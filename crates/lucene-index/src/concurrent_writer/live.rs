//! Searching documents still in the writer's buffers, with no flush and no
//! refresh.
//!
//! A writer built with [`ConcurrentIndexWriter::enable_live_search`] keeps,
//! per buffer, what a search reads besides the postings: each field's norms
//! as a column by document, each document's stored fields, and which
//! documents are deleted. Deletes are resolved as they arrive, against the
//! buffer's own postings (a delete term is a term lookup), so a search only
//! applies the deletes issued since the buffer's last add. A buffer taken
//! for a flush stays readable after its segment is published, until
//! [`ConcurrentIndexWriter::release_live`] says no reader lacks that segment
//! any more: a reader of an older commit gets it from the buffer. The
//! deletes issued meanwhile reach it as they reach the published segments.
//!
//! A searcher opens a [`LiveView`]: each buffer's document count and
//! deletes as of then. Its queries read the terms they need
//! ([`ConcurrentIndexWriter::live_view_snapshot`]) and its fetches the
//! stored fields ([`ConcurrentIndexWriter::live_document`]) of those
//! documents only, so every phase of a search sees the same documents. A
//! view keeps its buffers until it is closed.
//!
//! Opening a view holds every slot's lock, the control lock and the log's,
//! so it is one cut through the operations: an update's delete and its new
//! document are both in it or both not, and a buffer is either still
//! buffered, being flushed, or published (counted by
//! [`LiveView::published`]), never two of these. Commits are known by
//! generation, each holding the publishes before it. Reading a view's
//! terms holds the slots' locks while it decodes them: indexing waits for
//! what the requested terms' postings cost to decode.
//!
//! The extra columns, and the published buffers kept, are not charged to
//! the RAM buffer, so a writer with live search flushes after the same
//! documents as one without; [`ConcurrentIndexWriter::live_bytes`] reports
//! them.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::atomic::Ordering;
use std::sync::{Arc, MutexGuard};

use lucene_codecs::field_infos::{FieldInfo, IndexOptions};
use lucene_codecs::postings_writer::TermPostings;
use lucene_codecs::stored_fields::{FieldValue, StoredField};

use super::{lock, ConcurrentIndexWriter, Dwpt};
use crate::buffered_updates::{DocValuesUpdate, Term};
use crate::index_writer::pending_postings::{LiveFieldStats, PendingPostings};
use crate::index_writer::ExplicitFields;
use crate::index_writer::{DeleteNode, Error, IndexingConfig, Result};

/// What one search reads from the buffers: per field, the terms it looks
/// up.
#[derive(Debug, Clone, Default)]
pub struct LiveRequest {
    pub fields: Vec<LiveFieldRequest>,
}

/// One field of a [`LiveRequest`].
#[derive(Debug, Clone)]
pub struct LiveFieldRequest {
    pub field: String,
    /// The terms to read, in any order; repeats are read once.
    pub terms: Vec<Vec<u8>>,
    /// Read the terms' occurrences too. Without them the field is reported
    /// as indexing documents and frequencies only, which is all a search
    /// with no phrase, span or interval clause on it reads.
    pub positions: bool,
}

/// A consistent cut through the writer's buffers: see the module doc.
#[derive(Debug, Clone, Default)]
pub struct LiveSnapshot {
    /// Each buffer that holds documents: the slots' first, then the buffers
    /// being flushed in ticket order.
    pub buffers: Vec<LiveBuffer>,
    /// How many flushed buffers had been published as of the cut: a reader
    /// of the published segments as of the same count holds every other
    /// document exactly once.
    pub published: u64,
}

/// One buffer's documents as a segment of their own, doc ids in the order
/// they were added.
#[derive(Debug, Clone, Default)]
pub struct LiveBuffer {
    pub max_doc: i32,
    /// Deleted documents, ascending: deleted by term, or soft-deleted.
    pub deleted: Vec<i32>,
    /// The requested fields this buffer has terms for, in request order.
    pub fields: Vec<LiveBufferField>,
}

/// One requested field of a [`LiveBuffer`].
#[derive(Debug, Clone)]
pub struct LiveBufferField {
    /// The registered field, its index options lowered to documents and
    /// frequencies when positions were not requested.
    pub info: FieldInfo,
    /// The requested terms the buffer holds, in term order.
    pub terms: Vec<TermPostings>,
    /// Each document's norm (`max_doc` entries, `0` for a document without
    /// the field); `None` for a field without norms.
    pub norms: Option<Vec<i64>>,
    /// The buffer's statistics for the field, over all its documents.
    pub num_terms: i64,
    pub sum_total_term_freq: i64,
    pub sum_doc_freq: i64,
    pub doc_count: i32,
}

/// A growable bit per document.
#[derive(Debug, Clone, Default)]
struct Bits(Vec<u64>);

impl Bits {
    fn set(&mut self, doc: usize, on: bool) {
        let word = doc / 64;
        if word >= self.0.len() {
            if !on {
                return;
            }
            self.0.resize(word.saturating_add(1), 0);
        }
        let bit = 1u64 << (doc % 64);
        if on {
            self.0[word] |= bit;
        } else {
            self.0[word] &= !bit;
        }
    }

    fn ram_bytes(&self) -> usize {
        self.0.capacity().saturating_mul(8)
    }
}

/// One field's norms by document: a byte each while every norm fits one,
/// as BM25's do.
#[derive(Debug, Clone)]
enum NormColumn {
    Narrow(Vec<i8>),
    Wide(Vec<i64>),
}

impl NormColumn {
    fn set(&mut self, doc: usize, norm: i64) {
        if let NormColumn::Narrow(v) = self {
            if let Ok(b) = i8::try_from(norm) {
                if v.len() <= doc {
                    v.resize(doc.saturating_add(1), 0);
                }
                v[doc] = b;
                return;
            }
            *self = NormColumn::Wide(v.iter().map(|&b| i64::from(b)).collect());
        }
        if let NormColumn::Wide(v) = self {
            if v.len() <= doc {
                v.resize(doc.saturating_add(1), 0);
            }
            v[doc] = norm;
        }
    }

    fn to_vec(&self, max_doc: usize) -> Vec<i64> {
        let mut out: Vec<i64> = match self {
            NormColumn::Narrow(v) => v.iter().map(|&b| i64::from(b)).collect(),
            NormColumn::Wide(v) => v.clone(),
        };
        out.resize(max_doc, 0);
        out
    }

    fn ram_bytes(&self) -> usize {
        match self {
            NormColumn::Narrow(v) => v.capacity(),
            NormColumn::Wide(v) => v.capacity().saturating_mul(8),
        }
    }
}

/// A buffer's norms, deletes and stored fields, kept for live search.
#[derive(Debug, Clone, Default)]
pub(super) struct LiveColumns {
    /// Names the buffer from its first document on; `0` while it holds
    /// none.
    pub(super) id: u64,
    norms: HashMap<i32, NormColumn>,
    deletes: Deletes,
    stored: StoredArena,
}

/// A buffer's deleted documents.
#[derive(Debug, Clone, Default)]
struct Deletes {
    /// Deleted by term.
    hard: Bits,
    /// Soft-deleted: added with a soft-deletes value, or given one by an
    /// update.
    soft: Bits,
    /// A delete by query reached the buffer: its documents cannot be told
    /// apart from the postings alone.
    by_query: bool,
}

impl LiveColumns {
    /// Records the next document's stored fields.
    pub(super) fn record_stored(&mut self, stored: &[StoredField]) {
        self.stored.push(stored);
    }

    /// Records document `doc`'s norms, and whether it arrives soft-deleted
    /// (`soft` is the soft-deletes field's number).
    pub(super) fn record(&mut self, doc: i32, fields: &ExplicitFields, soft: Option<i32>) {
        let Ok(at) = usize::try_from(doc) else {
            return;
        };
        for f in &fields.inverted {
            if let Some(norm) = f.norm {
                self.norms
                    .entry(f.field_number)
                    .or_insert_with(|| NormColumn::Narrow(Vec::new()))
                    .set(at, norm);
            }
        }
        if let Some(soft) = soft {
            if fields.doc_values.iter().any(|v| v.field_number == soft) {
                self.deletes.soft.set(at, true);
            }
        }
    }

    /// See [`Deletes::apply`].
    pub(super) fn apply(
        &mut self,
        nodes: &[Arc<DeleteNode>],
        limit: i32,
        postings: &PendingPostings,
        cfg: &IndexingConfig,
    ) {
        self.deletes.apply(nodes, limit, postings, cfg);
    }

    pub(super) fn ram_bytes(&self) -> usize {
        self.norms
            .values()
            .map(NormColumn::ram_bytes)
            .fold(self.deletes.hard.ram_bytes(), usize::saturating_add)
            .saturating_add(self.deletes.soft.ram_bytes())
            .saturating_add(self.stored.ram_bytes())
    }
}

impl Deletes {
    /// Applies delete-log entries to the buffer's first `limit` documents,
    /// resolving each term against `postings`.
    fn apply(
        &mut self,
        nodes: &[Arc<DeleteNode>],
        limit: i32,
        postings: &PendingPostings,
        cfg: &IndexingConfig,
    ) {
        let mut docs = Vec::new();
        let matching = |field: &str, bytes: &[u8], docs: &mut Vec<(i32, i32)>| {
            docs.clear();
            if let Some(runs) = cfg
                .field_by_name(field)
                .and_then(|f| postings.live_field(f.number))
            {
                runs.live_docs_of(bytes, docs);
            }
            docs.iter()
                .map(|&(d, _)| d)
                .take_while(move |&d| d < limit)
                .filter_map(|d| usize::try_from(d).ok())
                .collect::<Vec<usize>>()
        };
        let soft_field = cfg.soft_deletes_field().map(|f| f.name.as_str());
        for node in nodes {
            match node.as_ref() {
                DeleteNode::Terms(terms) => {
                    for t in terms {
                        for d in matching(&t.field, &t.bytes, &mut docs) {
                            self.hard.set(d, true);
                        }
                    }
                }
                DeleteNode::Queries(queries) => self.by_query |= !queries.is_empty(),
                DeleteNode::DocValuesUpdates(updates) => {
                    for u in updates {
                        let DocValuesUpdate::Numeric { term, field, value } = u else {
                            continue;
                        };
                        if Some(field.as_str()) != soft_field {
                            continue;
                        }
                        for d in matching(&term.field, &term.bytes, &mut docs) {
                            self.soft.set(d, value.is_some());
                        }
                    }
                }
            }
        }
    }

    /// The deleted documents below `max_doc`, ascending.
    fn docs(&self, max_doc: i32) -> Vec<i32> {
        let words = self.hard.0.len().max(self.soft.0.len());
        let mut out = Vec::new();
        for w in 0..words {
            let h = self.hard.0.get(w).copied().unwrap_or(0);
            let s = self.soft.0.get(w).copied().unwrap_or(0);
            let mut bits = h | s;
            while bits != 0 {
                let b = bits.trailing_zeros() as usize;
                bits &= bits.wrapping_sub(1);
                match i32::try_from(w.saturating_mul(64).saturating_add(b)) {
                    Ok(d) if d < max_doc => out.push(d),
                    _ => return out,
                }
            }
        }
        out
    }
}

/// A buffer taken for a flush: readable by live search until it is
/// published, and after, until a commit holding it is released
/// ([`ConcurrentIndexWriter::release_live`]).
pub(super) struct Flushing {
    pub(super) postings: Arc<PendingPostings>,
    pub(super) columns: LiveColumns,
    pub(super) max_doc: i32,
    /// The segment it was published as, as flushed: its documents in the
    /// buffer's order. `None` until then, and for a segment dropped as
    /// fully deleted.
    pub(super) segment: Option<crate::segment_infos::SegmentCommitInfo>,
}

/// The live-search state the control lock guards.
#[derive(Default)]
pub(super) struct LiveFlushes {
    /// Being built, by ticket.
    pub(super) flushing: BTreeMap<u64, Flushing>,
    /// Published, by publish number, until a released commit holds them.
    pub(super) retained: BTreeMap<u64, Flushing>,
    /// Buffers published so far: the next publish number.
    pub(super) published: u64,
    /// Recent commits, oldest first.
    commits: VecDeque<Commit>,
    /// Open views per buffer id: a published buffer a view reads is kept.
    pins: HashMap<u64, usize>,
    /// Publishes below this number go once no view reads them.
    released: u64,
    /// The deletes handed over to the published segments
    /// ([`ConcurrentIndexWriter::hand_over`]), in log order, from the log
    /// position `history_base` on: those after a commit's
    /// [`Commit::handed`] are the ones its reader lacks. Kept only with
    /// live search on.
    history: VecDeque<Arc<DeleteNode>>,
    history_base: usize,
    keeps_history: bool,
    /// What [`Self::history`]'s entries hold.
    history_bytes: usize,
}

/// A commit this writer made.
#[derive(Debug, Clone, Copy)]
struct Commit {
    generation: i64,
    /// How many publishes it holds.
    held: u64,
    /// The log position up to which its segments had every delete.
    handed: usize,
}

/// How many commits [`LiveFlushes::commits`] remembers.
const COMMITS_KEPT: usize = 64;

impl LiveFlushes {
    /// A buffer taken for a flush published (`published` set, with the
    /// segment it was published as) or dropped.
    pub(super) fn finish(
        &mut self,
        ticket: u64,
        published: bool,
        segment: Option<crate::segment_infos::SegmentCommitInfo>,
    ) {
        if let Some(mut f) = self.flushing.remove(&ticket) {
            if published {
                f.segment = segment;
                self.retained.insert(self.published, f);
                self.published = self.published.saturating_add(1);
            }
        }
    }

    /// A commit at `generation` holds every publish so far, and every
    /// delete up to log position `handed`. A commit that wrote nothing new
    /// keeps the generation, and the position, of the one before.
    pub(super) fn committed(&mut self, generation: i64, handed: usize) {
        if self
            .commits
            .back()
            .is_some_and(|c| c.generation == generation)
        {
            return;
        }
        if self.commits.len() == COMMITS_KEPT {
            // No view can name the forgotten commit any more.
            if let Some(c) = self.commits.pop_front() {
                self.release(c.held);
            }
        }
        self.commits.push_back(Commit {
            generation,
            held: self.published,
            handed,
        });
    }

    /// Keeps the deletes handed over from now on, the next of them at log
    /// position `handed`.
    pub(super) fn keep_history(&mut self, handed: usize) {
        if !self.keeps_history {
            self.keeps_history = true;
            self.history.clear();
            self.history_bytes = 0;
            self.history_base = handed;
        }
    }

    /// `nodes`, the log's next entries, were handed over.
    pub(super) fn handed(&mut self, nodes: &[Arc<DeleteNode>]) {
        if self.keeps_history {
            for node in nodes {
                self.history_bytes = self.history_bytes.saturating_add(node.ram_bytes());
                self.history.push_back(Arc::clone(node));
            }
        }
    }

    /// Drops the published buffers below `held` no view reads, now and as
    /// the views reading the others close, and the deletes every reader
    /// still searched has.
    fn release(&mut self, held: u64) {
        self.released = self.released.max(held);
        let (released, pins) = (self.released, &self.pins);
        self.retained
            .retain(|&n, f| n >= released || pins.contains_key(&f.columns.id));
        let end = self.history_base.saturating_add(self.history.len());
        let needed = self
            .commits
            .iter()
            .find(|c| c.held >= self.released)
            .map_or(end, |c| c.handed);
        while self.history_base < needed {
            let Some(node) = self.history.pop_front() else {
                break;
            };
            self.history_bytes = self.history_bytes.saturating_sub(node.ram_bytes());
            self.history_base = self.history_base.saturating_add(1);
        }
    }

    /// Every buffer goes: `rollback` or `delete_all` discarded their
    /// documents, and the log, now at position `handed`. A view of one
    /// reads it as all deleted; no earlier commit can be viewed, its
    /// segments having lost documents no delete names.
    pub(super) fn discard(&mut self, handed: usize) {
        self.flushing.clear();
        self.retained.clear();
        self.released = self.published;
        self.commits.clear();
        self.history.clear();
        self.history_bytes = 0;
        self.history_base = handed;
    }

    /// The deletes the reader of `commit` lacks: the handed ones after it,
    /// then `unhanded`, in log order. `None` when the history no longer
    /// reaches back to it.
    fn deletes_after(
        &self,
        commit: &Commit,
        unhanded: &[Arc<DeleteNode>],
    ) -> Option<Vec<Arc<DeleteNode>>> {
        if !self.keeps_history || commit.handed < self.history_base {
            return None;
        }
        let skip = commit.handed.saturating_sub(self.history_base);
        Some(
            self.history
                .iter()
                .skip(skip)
                .chain(unhanded)
                .cloned()
                .collect(),
        )
    }

    /// Whether every publish from `from` on is still kept.
    fn keeps_from(&self, from: u64) -> bool {
        let kept = self.retained.range(from..).count() as u64;
        kept == self.published.saturating_sub(from)
    }

    /// Buffer `id`, if it is being flushed or kept after its publish.
    fn find(&self, id: u64) -> Option<Source<'_>> {
        self.buffers().find(|f| f.columns.id == id).map(|f| Source {
            postings: &f.postings,
            columns: &f.columns,
        })
    }

    /// The commit at `generation`.
    fn commit(&self, generation: i64) -> Result<Commit> {
        self.commits
            .iter()
            .find(|c| c.generation == generation)
            .copied()
            .ok_or_else(|| {
                Error::LiveSearch(format!(
                    "commit generation {generation} is not one this writer made"
                ))
            })
    }

    /// Every buffer still read: being built, then published.
    fn buffers(&self) -> impl Iterator<Item = &Flushing> {
        self.flushing.values().chain(self.retained.values())
    }

    pub(super) fn buffers_mut(&mut self) -> impl Iterator<Item = &mut Flushing> {
        self.flushing.values_mut().chain(self.retained.values_mut())
    }
}

/// The terms `nodes` delete by, in order: hard deletes, and soft deletes
/// through `soft_field`. `None` when one of them is anything else that
/// changes which documents are live: a delete by query, or a soft delete
/// taken back.
fn delete_terms(nodes: &[Arc<DeleteNode>], soft_field: Option<&str>) -> Option<Vec<Term>> {
    let mut out = Vec::new();
    for node in nodes {
        match node.as_ref() {
            DeleteNode::Terms(terms) => out.extend(terms.iter().map(|t| Term::clone(t))),
            DeleteNode::Queries(queries) => {
                if !queries.is_empty() {
                    return None;
                }
            }
            DeleteNode::DocValuesUpdates(updates) => {
                for u in updates {
                    let DocValuesUpdate::Numeric { term, field, value } = u else {
                        continue;
                    };
                    if Some(field.as_str()) != soft_field {
                        continue;
                    }
                    value.as_ref()?;
                    out.push(term.clone());
                }
            }
        }
    }
    Some(out)
}

/// A request field resolved against the schema.
struct Wanted<'a> {
    info: &'a FieldInfo,
    terms: Vec<&'a [u8]>,
    positions: bool,
}

/// The buffers one searcher reads, fixed when it opens
/// ([`ConcurrentIndexWriter::open_live_view`]): each buffer's documents and
/// deletes as of then. Later documents, and later deletes, are not in it.
/// Its buffers stay readable until it is closed
/// ([`ConcurrentIndexWriter::close_live_view`]).
#[derive(Debug, Clone, Default)]
pub struct LiveView {
    pub buffers: Vec<ViewBuffer>,
    /// See [`LiveSnapshot::published`].
    pub published: u64,
    /// For a view of a commit, the delete terms issued since, in the order
    /// they were issued: each deletes every document of that commit's
    /// segments it matches, as its reader does not know yet. A later view
    /// of the same commit lists the same terms first.
    pub commit_deletes: Vec<Term>,
}

/// One buffer of a [`LiveView`].
#[derive(Debug, Clone, Default)]
pub struct ViewBuffer {
    /// The buffer, across its moves from a slot to a flush and a publish.
    pub id: u64,
    pub max_doc: i32,
    /// Deleted documents, ascending.
    pub deleted: Vec<i32>,
    /// Whether the buffer was still taking documents when the view opened:
    /// else it was being flushed, or published, and grows no more.
    pub in_slot: bool,
    /// Every field's statistics as of the view, by field number.
    stats: HashMap<i32, LiveFieldStats>,
}

impl ViewBuffer {
    /// By field number, the statistics of the view's documents of the
    /// buffer, deleted ones included.
    pub fn field_totals(&self) -> HashMap<i32, FieldTotals> {
        self.stats
            .iter()
            .map(|(&f, s)| {
                (
                    f,
                    FieldTotals {
                        num_terms: s.num_terms,
                        sum_total_term_freq: s.sum_total_term_freq,
                        sum_doc_freq: s.sum_doc_freq,
                        doc_count: s.doc_count,
                    },
                )
            })
            .collect()
    }
}

/// A field's statistics over a buffer's first documents: see
/// [`ViewBuffer::field_totals`]. Those of documents `a..b` are the
/// difference of the totals at `b` and at `a`, but for `num_terms` (the
/// distinct terms, which do not subtract).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FieldTotals {
    pub num_terms: i64,
    pub sum_total_term_freq: i64,
    pub sum_doc_freq: i64,
    pub doc_count: i32,
}

/// A buffered document's stored fields, in the order they were added.
#[derive(Debug, Clone, Default)]
pub(super) struct StoredArena {
    bytes: Vec<u8>,
    /// Document `d`'s fields end at `ends[d]`.
    ends: Vec<usize>,
}

const LONG: u8 = 0;
const INT: u8 = 1;
const FLOAT: u8 = 2;
const DOUBLE: u8 = 3;
const BINARY: u8 = 4;
const STRING: u8 = 5;

impl StoredArena {
    fn push(&mut self, fields: &[StoredField]) {
        let out = &mut self.bytes;
        for f in fields {
            out.extend_from_slice(&f.field_number.to_le_bytes());
            let (tag, bytes): (u8, &[u8]) = match &f.value {
                FieldValue::Long(v) => (LONG, &v.to_le_bytes()),
                FieldValue::Int(v) => (INT, &v.to_le_bytes()),
                FieldValue::Float(v) => (FLOAT, &v.to_bits().to_le_bytes()),
                FieldValue::Double(v) => (DOUBLE, &v.to_bits().to_le_bytes()),
                FieldValue::Binary(b) => (BINARY, b.as_slice()),
                FieldValue::String(s) => (STRING, s.as_bytes()),
            };
            out.push(tag);
            out.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
            out.extend_from_slice(bytes);
        }
        self.ends.push(out.len());
    }

    fn document(&self, doc: usize) -> Option<Vec<StoredField>> {
        let end = *self.ends.get(doc)?;
        let start = doc.checked_sub(1).map_or(0, |p| self.ends[p]);
        let mut r = self.bytes.get(start..end)?;
        let mut out = Vec::new();
        let take = |n: usize, r: &mut &[u8]| -> Option<Vec<u8>> {
            let (head, rest) = r.split_at_checked(n)?;
            *r = rest;
            Some(head.to_vec())
        };
        while !r.is_empty() {
            let number = i32::from_le_bytes(take(4, &mut r)?.try_into().ok()?);
            let tag = take(1, &mut r)?[0];
            let len = u64::from_le_bytes(take(8, &mut r)?.try_into().ok()?);
            let bytes = take(usize::try_from(len).ok()?, &mut r)?;
            let value = match tag {
                LONG => FieldValue::Long(i64::from_le_bytes(bytes.try_into().ok()?)),
                INT => FieldValue::Int(i32::from_le_bytes(bytes.try_into().ok()?)),
                FLOAT => {
                    FieldValue::Float(f32::from_bits(u32::from_le_bytes(bytes.try_into().ok()?)))
                }
                DOUBLE => {
                    FieldValue::Double(f64::from_bits(u64::from_le_bytes(bytes.try_into().ok()?)))
                }
                BINARY => FieldValue::Binary(bytes),
                STRING => FieldValue::String(String::from_utf8(bytes).ok()?),
                _ => return None,
            };
            out.push(StoredField {
                field_number: number,
                value,
            });
        }
        Some(out)
    }

    fn ram_bytes(&self) -> usize {
        self.bytes
            .capacity()
            .saturating_add(self.ends.capacity().saturating_mul(8))
    }
}

/// Where a buffer is: its postings and columns.
struct Source<'a> {
    postings: &'a PendingPostings,
    columns: &'a LiveColumns,
}

impl ConcurrentIndexWriter<'_> {
    /// Keeps what live search reads ([`Self::open_live_view`]) as documents
    /// arrive, from now on. Fails once a document is buffered: a buffer
    /// filled without it cannot be read.
    pub fn enable_live_search(&self) -> Result<()> {
        let slots: Vec<_> = self.slots.iter().map(lock).collect();
        if slots.iter().any(|d| !d.docs.is_empty()) {
            return Err(Error::LiveSearch(
                "live search must be enabled before the first document".into(),
            ));
        }
        let mut core = lock(&self.core);
        let handed = lock(&self.log).handed;
        core.live.keep_history(handed);
        self.live.store(true, Ordering::Release);
        Ok(())
    }

    /// Whether this writer keeps what live search reads.
    pub fn live_search(&self) -> bool {
        self.live()
    }

    pub(super) fn live(&self) -> bool {
        self.live.load(Ordering::Acquire)
    }

    /// The heap live search holds beyond the RAM buffer's charge: every
    /// buffer's columns and stored fields, the postings of published
    /// buffers still kept, and the deletes kept for readers of older
    /// commits.
    pub fn live_bytes(&self) -> usize {
        let slots = self
            .slots
            .iter()
            .map(|s| lock(s).live.ram_bytes())
            .fold(0usize, usize::saturating_add);
        let core = lock(&self.core);
        let retained = core
            .live
            .retained
            .values()
            .map(|f| f.postings.bytes())
            .fold(slots, usize::saturating_add);
        core.live
            .buffers()
            .map(|f| f.columns.ram_bytes())
            .fold(retained, usize::saturating_add)
            .saturating_add(core.live.history_bytes)
    }

    /// Stops reading the published buffers the commit at `generation` holds,
    /// once no open view reads them: call it once no search reads from an
    /// older commit. Their memory goes.
    pub fn release_live(&self, generation: i64) -> Result<()> {
        let mut core = lock(&self.core);
        let held = core.live.commit(generation)?.held;
        core.live.release(held);
        Ok(())
    }

    fn check_live(&self) -> Result<Arc<IndexingConfig>> {
        let cfg = self.cfg();
        if !self.live() || !cfg.takes_explicit() {
            return Err(Error::LiveSearch(
                "the writer keeps no live-search columns".into(),
            ));
        }
        Ok(cfg)
    }

    /// Opens a view of every buffered document a reader of the commit at
    /// `commit` lacks (with `None`, the documents in no published segment):
    /// see the module doc. Close it with [`Self::close_live_view`].
    ///
    /// Fails for a writer without [`Self::enable_live_search`], for one that
    /// takes no explicit documents (only those are inverted as they arrive),
    /// for a commit this writer did not make or no longer remembers, for one
    /// whose buffers [`Self::release_live`] dropped, and when a delete by
    /// query reached a buffer.
    pub fn open_live_view(&self, commit: Option<i64>) -> Result<LiveView> {
        let cfg = self.check_live()?;
        let slots: Vec<_> = self.slots.iter().map(lock).collect();
        let mut core = lock(&self.core);
        let commit = commit.map(|g| core.live.commit(g)).transpose()?;
        let from = commit.map_or(core.live.published, |c| c.held);
        if !core.live.keeps_from(from) {
            return Err(Error::LiveSearch(format!(
                "the buffers published after commit {:?} were released",
                commit.map(|c| c.generation)
            )));
        }
        let (slot_tails, flushing_tail) = {
            let log = lock(&self.log);
            let tails: Vec<_> = slots.iter().map(|d| log.since(d.slice)).collect();
            (tails, log.since(log.handed))
        };
        let commit_nodes = match &commit {
            Some(c) => Some(core.live.deletes_after(c, &flushing_tail).ok_or_else(|| {
                Error::LiveSearch(format!(
                    "the deletes since commit {} are no longer kept",
                    c.generation
                ))
            })?),
            None => None,
        };
        let mut out = LiveView {
            buffers: Vec::new(),
            published: core.live.published,
            commit_deletes: Vec::new(),
        };
        for (dwpt, tail) in slots.iter().zip(&slot_tails) {
            if dwpt.docs.is_empty() {
                continue;
            }
            let max_doc = i32::try_from(dwpt.docs.len()).unwrap_or(i32::MAX);
            let source = Source {
                postings: &dwpt.postings,
                columns: &dwpt.live,
            };
            let mut buffer = view_buffer(&source, tail, max_doc, &cfg)?;
            buffer.in_slot = true;
            out.buffers.push(buffer);
        }
        let published = core.live.retained.range(from..).map(|(_, f)| f);
        for f in core.live.flushing.values().chain(published) {
            let source = Source {
                postings: &f.postings,
                columns: &f.columns,
            };
            out.buffers
                .push(view_buffer(&source, &flushing_tail, f.max_doc, &cfg)?);
        }
        for b in &out.buffers {
            let pins = core.live.pins.entry(b.id).or_insert(0);
            *pins = pins.saturating_add(1);
        }
        drop(core);
        drop(slots);
        if let Some(nodes) = commit_nodes {
            let soft = cfg.soft_deletes_field().map(|f| f.name.as_str());
            match delete_terms(&nodes, soft) {
                Some(terms) => out.commit_deletes = terms,
                None => {
                    self.close_live_view(&out);
                    return Err(Error::LiveSearch(
                        "a delete since the commit is not by term".into(),
                    ));
                }
            }
        }
        Ok(out)
    }

    /// The segment buffer `id` was published as, in the writer's directory
    /// ([`Self::directory`]): every document of the buffer, which grows no
    /// more. `None` while it is buffered or being flushed, and once a view
    /// no longer pins it.
    pub fn live_published_segment(
        &self,
        id: u64,
    ) -> Option<crate::segment_infos::SegmentCommitInfo> {
        let core = lock(&self.core);
        core.live
            .retained
            .values()
            .find(|f| f.columns.id == id)
            .and_then(|f| f.segment.clone())
    }

    /// Writes documents `start..end` of buffer `id` into `dir` as segment
    /// `name`, numbered from `start`, as a flush of just those documents
    /// would write them, without taking them from the writer: what a search
    /// needing whole segments (doc values, points, every term) reads a
    /// view's buffers through. `doc_counts` is, by field number, how many
    /// of those documents have the field: the difference of the
    /// `doc_count`s of the [`ViewBuffer::field_totals`] of views of `end` and of `start`
    /// documents. Stored fields are not written ([`Self::live_document`]
    /// reads them); deletes are the caller's.
    ///
    /// A range starting where an earlier one ended reads the buffered
    /// postings from there on; `next_starts` names the documents ranges
    /// may start at next (the ends of the ranges kept), and where other
    /// ranges ended is forgotten. Without `postings` the segment has no
    /// terms (norms, doc values and points only), for a search that reads
    /// the terms it needs through [`Self::live_view_snapshot`].
    ///
    /// The buffer's slot is locked while the segment is written. Fails when
    /// the buffer has left its slot (taken for a flush since), when it
    /// holds fewer than `end` documents, and under an index sort, which
    /// would renumber the documents.
    #[allow(clippy::too_many_arguments)]
    pub fn live_buffer_segment(
        &self,
        id: u64,
        range: std::ops::Range<i32>,
        doc_counts: HashMap<i32, i32>,
        next_starts: &[i32],
        postings: bool,
        dir: &dyn lucene_store::Directory,
        name: &str,
    ) -> Result<crate::segment_infos::SegmentCommitInfo> {
        let cfg = self.check_live()?;
        if cfg.sorts_documents() {
            return Err(Error::LiveSearch(
                "an index sort renumbers the buffered documents".into(),
            ));
        }
        let slot = self
            .slot_of(id)
            .filter(|d| !d.docs.is_empty())
            .ok_or_else(|| Error::LiveSearch(format!("buffer {id} is being flushed")))?;
        let (Ok(start), Ok(end)) = (usize::try_from(range.start), usize::try_from(range.end))
        else {
            return Err(Error::LiveSearch("a negative document range".into()));
        };
        if start >= end || end > slot.docs.len() {
            return Err(Error::LiveSearch(format!(
                "documents {start}..{end} of a buffer of {}",
                slot.docs.len()
            )));
        }
        let limit = crate::index_writer::BufferLimit {
            start: range.start,
            docs: range.end,
            doc_counts,
            postings,
        };
        let buffer = crate::index_writer::DocumentBuffer {
            docs: slot.docs.get(start..end).unwrap_or_default(),
            custom_freq_terms: slot.custom_freq_terms.get(start..end).unwrap_or_default(),
            vectors: slot.vectors.get(start..end).unwrap_or_default(),
            explicit: crate::index_writer::ExplicitDocs::PackedRange(&slot.explicit, start, end),
            postings: Some(&slot.postings),
            old_to_new: None,
            has_blocks: slot.has_blocks,
            stored: None,
            limit: Some(&limit),
        };
        let (sci, _) = cfg.build_and_write_segment(
            dir,
            &buffer,
            name,
            lucene_util::string_helper::random_id(),
            None,
        )?;
        if postings {
            slot.postings.keep_cursors_at(next_starts);
        }
        Ok(sci)
    }

    /// The directory the writer's segments are in.
    pub fn directory(&self) -> &dyn lucene_store::Directory {
        self.dir
    }

    /// Lets the buffers `view` read go once no other view reads them.
    pub fn close_live_view(&self, view: &LiveView) {
        let mut core = lock(&self.core);
        for b in &view.buffers {
            if let Some(n) = core.live.pins.get_mut(&b.id) {
                *n = n.saturating_sub(1);
                if *n == 0 {
                    core.live.pins.remove(&b.id);
                }
            }
        }
        let released = core.live.released;
        core.live.release(released);
    }

    /// The slot holding buffer `id`, locked. The free slots are looked at
    /// first, so a slot another thread holds (writing one of its own chunks,
    /// say) is waited for only when the buffer is in no free one. A buffer
    /// never moves into a slot, so the second look cannot miss it.
    fn slot_of(&self, id: u64) -> Option<MutexGuard<'_, Dwpt>> {
        let mut busy = Vec::new();
        for slot in &self.slots {
            match slot.try_lock() {
                Ok(d) if d.live.id == id => return Some(d),
                Ok(_) => {}
                Err(std::sync::TryLockError::Poisoned(p)) => {
                    let d = p.into_inner();
                    if d.live.id == id {
                        return Some(d);
                    }
                }
                Err(std::sync::TryLockError::WouldBlock) => busy.push(slot),
            }
        }
        busy.into_iter().map(lock).find(|d| d.live.id == id)
    }

    /// Runs `f` on buffer `id`'s postings and columns, wherever it is now.
    /// A buffer only moves from a slot to the control lock, so looking in
    /// the slots one at a time, then there, cannot miss it.
    fn with_buffer<T>(&self, id: u64, f: impl Fn(Option<&Source<'_>>) -> T) -> T {
        if let Some(d) = self.slot_of(id) {
            return f(Some(&Source {
                postings: &d.postings,
                columns: &d.live,
            }));
        }
        let core = lock(&self.core);
        f(core.live.find(id).as_ref())
    }

    /// The requested fields and terms of `view`'s documents, with each
    /// buffer's norms, deletes and field statistics as of the view. Fields
    /// the schema does not know, and terms no buffer holds, are left out.
    pub fn live_view_snapshot(
        &self,
        view: &LiveView,
        request: &LiveRequest,
    ) -> Result<LiveSnapshot> {
        let cfg = self.check_live()?;
        let wanted: Vec<Wanted<'_>> = request
            .fields
            .iter()
            .filter_map(|r| {
                let info = cfg.field_by_name(&r.field)?;
                let mut terms: Vec<&[u8]> = r.terms.iter().map(Vec::as_slice).collect();
                terms.sort_unstable();
                terms.dedup();
                Some(Wanted {
                    info,
                    terms,
                    positions: r.positions && info.index_options.subsumes_positions(),
                })
            })
            .collect();
        let slots: Vec<_> = self.slots.iter().map(lock).collect();
        let core = lock(&self.core);
        let mut out = LiveSnapshot {
            buffers: Vec::with_capacity(view.buffers.len()),
            published: view.published,
        };
        for b in &view.buffers {
            let source = slots
                .iter()
                .find(|d| d.live.id == b.id)
                .map(|d| Source {
                    postings: &d.postings,
                    columns: &d.live,
                })
                .or_else(|| core.live.find(b.id));
            // A buffer `delete_all` or `rollback` dropped: none of its
            // documents is live any more.
            let Some(source) = source else {
                out.buffers.push(LiveBuffer {
                    max_doc: b.max_doc,
                    deleted: (0..b.max_doc).collect(),
                    fields: Vec::new(),
                });
                continue;
            };
            out.buffers.push(read_buffer(&source, b, &wanted)?);
        }
        Ok(out)
    }

    /// The stored fields of document `doc` of `view`'s buffer `buffer`.
    pub fn live_document(
        &self,
        view: &LiveView,
        buffer: usize,
        doc: i32,
    ) -> Result<Vec<StoredField>> {
        let missing =
            || Error::LiveSearch(format!("no buffered document {doc} in buffer {buffer}"));
        let b = view.buffers.get(buffer).ok_or_else(missing)?;
        if doc < 0 || doc >= b.max_doc {
            return Err(missing());
        }
        let doc = usize::try_from(doc).map_err(|_| missing())?;
        self.with_buffer(b.id, |s| s.and_then(|s| s.columns.stored.document(doc)))
            .ok_or_else(missing)
    }

    /// [`Self::open_live_view`], [`Self::live_view_snapshot`] and
    /// [`Self::close_live_view`] in one: the requested terms of every
    /// buffered document a reader of the commit at `commit` lacks.
    pub fn live_snapshot(
        &self,
        request: &LiveRequest,
        commit: Option<i64>,
    ) -> Result<LiveSnapshot> {
        let view = self.open_live_view(commit)?;
        let out = self.live_view_snapshot(&view, request);
        self.close_live_view(&view);
        out
    }
}

/// One buffer's documents and deletes as of now, with `tail` (delete-log
/// entries it has not applied, all issued after its documents) applied.
fn view_buffer(
    source: &Source<'_>,
    tail: &[Arc<DeleteNode>],
    max_doc: i32,
    cfg: &IndexingConfig,
) -> Result<ViewBuffer> {
    let with_tail;
    let deletes = if tail.is_empty() {
        &source.columns.deletes
    } else {
        let mut d = source.columns.deletes.clone();
        d.apply(tail, max_doc, source.postings, cfg);
        with_tail = d;
        &with_tail
    };
    if deletes.by_query {
        return Err(Error::LiveSearch(
            "a delete by query reached a buffered document".into(),
        ));
    }
    Ok(ViewBuffer {
        id: source.columns.id,
        max_doc,
        deleted: deletes.docs(max_doc),
        in_slot: false,
        stats: source.postings.live_stats_all(),
    })
}

/// The requested terms of `view`'s documents of one buffer.
fn read_buffer(
    source: &Source<'_>,
    view: &ViewBuffer,
    wanted: &[Wanted<'_>],
) -> Result<LiveBuffer> {
    let max = usize::try_from(view.max_doc).unwrap_or(0);
    let mut fields = Vec::new();
    for w in wanted {
        let (Some(runs), Some(stats)) = (
            source.postings.live_field(w.info.number),
            view.stats.get(&w.info.number),
        ) else {
            continue;
        };
        let mut terms = Vec::new();
        for t in &w.terms {
            if let Some(mut p) = runs.live_term(t, w.positions)? {
                truncate(&mut p, view.max_doc);
                if !p.docs.is_empty() {
                    terms.push(p);
                }
            }
        }
        let mut info = w.info.clone();
        if w.positions {
            info.store_payloads = runs.has_payloads();
        } else if info.index_options.subsumes_positions() {
            info.index_options = IndexOptions::DocsAndFreqs;
            info.store_payloads = false;
        }
        let freqs = !matches!(info.index_options, IndexOptions::Docs | IndexOptions::None);
        let norms = source.columns.norms.get(&w.info.number);
        let norms =
            (!w.info.omit_norms).then(|| norms.map_or_else(|| vec![0; max], |c| c.to_vec(max)));
        fields.push(LiveBufferField {
            info,
            terms,
            norms,
            num_terms: stats.num_terms,
            sum_total_term_freq: if freqs {
                stats.sum_total_term_freq
            } else {
                stats.sum_doc_freq
            },
            sum_doc_freq: stats.sum_doc_freq,
            doc_count: stats.doc_count,
        });
    }
    Ok(LiveBuffer {
        max_doc: view.max_doc,
        deleted: view.deleted.clone(),
        fields,
    })
}

/// Drops `p`'s documents from `max_doc` on, with their occurrences.
fn truncate(p: &mut TermPostings, max_doc: i32) {
    let keep = p.docs.partition_point(|&(d, _)| d < max_doc);
    if keep == p.docs.len() {
        return;
    }
    if !p.payload_lengths.is_empty() {
        let occurrences: usize = p.positions[..keep].iter().map(Vec::len).sum();
        let bytes: usize = p.payload_lengths[..occurrences]
            .iter()
            .map(|&l| l as usize)
            .sum();
        p.payload_lengths.truncate(occurrences);
        p.payload_bytes.truncate(bytes);
    }
    p.docs.truncate(keep);
    p.positions.truncate(keep);
    p.offsets.truncate(keep);
}

#[cfg(test)]
mod tests;
