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
use std::sync::Arc;

use lucene_codecs::field_infos::{FieldInfo, IndexOptions};
use lucene_codecs::postings_writer::TermPostings;
use lucene_codecs::stored_fields::{FieldValue, StoredField};

use super::{lock, ConcurrentIndexWriter};
use crate::buffered_updates::DocValuesUpdate;
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
    /// Recent commits: generation, and how many publishes it holds.
    commits: VecDeque<(i64, u64)>,
    /// Open views per buffer id: a published buffer a view reads is kept.
    pins: HashMap<u64, usize>,
}

/// How many commits [`LiveFlushes::commits`] remembers.
const COMMITS_KEPT: usize = 64;

impl LiveFlushes {
    /// A buffer taken for a flush published (`published` set) or dropped.
    pub(super) fn finish(&mut self, ticket: u64, published: bool) {
        if let Some(f) = self.flushing.remove(&ticket) {
            if published {
                self.retained.insert(self.published, f);
                self.published = self.published.saturating_add(1);
            }
        }
    }

    /// A commit at `generation` holds every publish so far.
    pub(super) fn committed(&mut self, generation: i64) {
        if self.commits.back().is_some_and(|&(g, _)| g == generation) {
            return;
        }
        if self.commits.len() == COMMITS_KEPT {
            self.commits.pop_front();
        }
        self.commits.push_back((generation, self.published));
    }

    /// How many publishes the commit at `generation` holds.
    fn held_by(&self, generation: i64) -> Result<u64> {
        self.commits
            .iter()
            .find(|&&(g, _)| g == generation)
            .map(|&(_, p)| p)
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
}

/// One buffer of a [`LiveView`].
#[derive(Debug, Clone, Default)]
pub struct ViewBuffer {
    /// The buffer, across its moves from a slot to a flush and a publish.
    pub id: u64,
    pub max_doc: i32,
    /// Deleted documents, ascending.
    pub deleted: Vec<i32>,
    /// Every field's statistics as of the view, by field number.
    stats: HashMap<i32, LiveFieldStats>,
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
    /// buffer's columns and stored fields, and the postings of published
    /// buffers still kept.
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
    }

    /// Stops reading the published buffers the commit at `generation` holds,
    /// once no open view reads them: call it once no search reads from an
    /// older commit. Their memory goes.
    pub fn release_live(&self, generation: i64) -> Result<()> {
        let mut core = lock(&self.core);
        let held = core.live.held_by(generation)?;
        let live = &mut core.live;
        let pins = &live.pins;
        live.retained
            .retain(|&n, f| n >= held || pins.contains_key(&f.columns.id));
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
        let from = match commit {
            Some(generation) => core.live.held_by(generation)?,
            None => core.live.published,
        };
        if from < core.live.published && core.live.retained.keys().next().is_none_or(|&n| n > from)
        {
            return Err(Error::LiveSearch(format!(
                "the buffers published after commit {commit:?} were released"
            )));
        }
        let (slot_tails, flushing_tail) = {
            let log = lock(&self.log);
            let tails: Vec<_> = slots.iter().map(|d| log.since(d.slice)).collect();
            (tails, log.since(log.handed))
        };
        let mut out = LiveView {
            buffers: Vec::new(),
            published: core.live.published,
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
            out.buffers.push(view_buffer(&source, tail, max_doc, &cfg)?);
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
        Ok(out)
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
    }

    /// Runs `f` on buffer `id`'s postings and columns, wherever it is now.
    fn with_buffer<T>(&self, id: u64, f: impl Fn(Option<&Source<'_>>) -> T) -> T {
        let slots: Vec<_> = self.slots.iter().map(lock).collect();
        if let Some(d) = slots.iter().find(|d| d.live.id == id) {
            return f(Some(&Source {
                postings: &d.postings,
                columns: &d.live,
            }));
        }
        let core = lock(&self.core);
        let found = core.live.buffers().find(|b| b.columns.id == id);
        f(found
            .map(|b| Source {
                postings: &b.postings,
                columns: &b.columns,
            })
            .as_ref())
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
                .or_else(|| {
                    core.live
                        .buffers()
                        .find(|f| f.columns.id == b.id)
                        .map(|f| Source {
                            postings: &f.postings,
                            columns: &f.columns,
                        })
                });
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
