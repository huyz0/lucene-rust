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
use std::sync::{Arc, RwLockReadGuard};

use lucene_codecs::field_infos::{DocValuesType, FieldInfo, IndexOptions};
use lucene_codecs::postings_writer::TermPostings;
use lucene_codecs::stored_fields::{FieldValue, StoredField};

use super::{lock, read_slot, write_slot, ConcurrentIndexWriter, Dwpt};
use crate::buffered_updates::{DocValuesUpdate, Term};
use crate::index_writer::packed_fields::PackedFields;
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
    /// Read every buffered term any of these match too.
    pub patterns: Vec<LivePattern>,
}

/// A set of terms named by a pattern ([`LiveFieldRequest::patterns`]).
#[derive(Debug, Clone)]
pub enum LivePattern {
    /// A wildcard (or prefix) pattern.
    Wildcard(lucene_codecs::wildcard::WildcardPattern),
    /// A regexp with a compiled automaton
    /// ([`lucene_codecs::regexp::RegexpPattern::dfa_matches`]); one without
    /// matches nothing here.
    Regexp(lucene_codecs::regexp::RegexpPattern),
}

impl LivePattern {
    fn matches(&self, term: &[u8]) -> bool {
        match self {
            LivePattern::Wildcard(p) => p.matches(term),
            LivePattern::Regexp(p) => p.dfa_matches(term).unwrap_or(false),
        }
    }
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
    /// Each document's norm from document [`Self::norms_from`] on
    /// (`max_doc - norms_from` entries, `0` for a document without the
    /// field); `None` for a field without norms.
    pub norms: Option<Vec<i64>>,
    /// The first document [`Self::norms`] holds: where the buffer was read
    /// from.
    pub norms_from: i32,
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

    /// Document `doc`'s norm, `0` when it has none.
    fn get(&self, doc: usize) -> Option<i64> {
        match self {
            NormColumn::Narrow(v) => v.get(doc).map(|&b| i64::from(b)),
            NormColumn::Wide(v) => v.get(doc).copied(),
        }
    }

    /// Documents `from..max_doc`'s norms, 0 past the column's end.
    fn to_vec_from(&self, from: usize, max_doc: usize) -> Vec<i64> {
        let to = max_doc.max(from);
        let mut out: Vec<i64> = match self {
            NormColumn::Narrow(v) => v
                .get(from..to.min(v.len()))
                .unwrap_or_default()
                .iter()
                .map(|&b| i64::from(b))
                .collect(),
            NormColumn::Wide(v) => v.get(from..to.min(v.len())).unwrap_or_default().to_vec(),
        };
        out.resize(to.saturating_sub(from), 0);
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
    doc_values: doc_values::DocValuesColumns,
    chunk: chunk_fields::ChunkFields,
    /// The vector fields' graphs over the buffer's vectors.
    graphs: Vec<crate::index_writer::BufferGraphField>,
    /// Vector fields kept without a graph: their format has none, or a
    /// vector could not be added to it.
    no_graph: Vec<i32>,
}

/// Vector ordinals an inserter adds before letting a flush that waits for
/// the graph take over.
const INSERT_BATCH: usize = 256;

/// A buffered vector as its field's graph holds it: the same allocation.
fn buffer_vector(
    value: &crate::index_writer::VectorValue,
) -> lucene_codecs::buffer_vectors::BufferVector {
    use lucene_codecs::buffer_vectors::BufferVector;
    match value {
        crate::index_writer::VectorValue::Float32(v) => BufferVector::Float32(Arc::clone(v)),
        crate::index_writer::VectorValue::Byte(v) => BufferVector::Byte(Arc::clone(v)),
    }
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
    /// Adds document `doc`'s vectors to their fields' graphs, and starts
    /// an inserter on a field whose graph has none running.
    pub(super) fn record_vectors(
        &mut self,
        doc: i32,
        vectors: &[crate::index_writer::DocumentVector],
        cfg: &IndexingConfig,
    ) {
        for v in vectors {
            let Some(number) = cfg.vector_field_number(&v.field_name) else {
                continue;
            };
            if self.no_graph.contains(&number) {
                continue;
            }
            let at = match self.graphs.iter().position(|g| g.field_number == number) {
                Some(at) => at,
                None => match cfg.buffer_graph_for(&v.field_name) {
                    Some(graph) => {
                        self.graphs.push(crate::index_writer::BufferGraphField {
                            field_number: number,
                            graph: Arc::new(graph),
                        });
                        self.graphs.len().saturating_sub(1)
                    }
                    None => {
                        self.no_graph.push(number);
                        continue;
                    }
                },
            };
            let Some(field) = self.graphs.get(at) else {
                continue;
            };
            if field.graph.push(doc, buffer_vector(&v.value)).is_err() {
                // Its ordinals no longer follow the documents: searches
                // scan the field's vectors exactly and the flush builds
                // its graph.
                self.graphs.remove(at);
                self.no_graph.push(number);
                continue;
            }
            if field.graph.schedule() {
                let graph = Arc::clone(&field.graph);
                // An insertion that fails or panics leaves the ordinals after
                // it out of the graph: searches score them exactly, and the
                // flush builds the field's graph again.
                rayon::spawn(move || {
                    // A panic stops at this task: no process abort, and the
                    // graph is marked failed so the flush builds its own.
                    let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        graph.insert_pending(INSERT_BATCH)
                    }));
                    if !matches!(run, Ok(Ok(()))) {
                        graph.fail();
                    }
                });
            }
        }
    }

    /// The vector fields' graphs.
    pub(super) fn graphs(&self) -> &[crate::index_writer::BufferGraphField] {
        &self.graphs
    }

    /// Records the next document's stored fields.
    pub(super) fn record_stored(&mut self, stored: &[StoredField]) {
        self.stored.push(stored);
    }

    /// Records document `doc`'s norms, and whether it arrives soft-deleted
    /// (`soft` is the soft-deletes field's number).
    pub(super) fn record(
        &mut self,
        doc: i32,
        fields: &ExplicitFields,
        soft: Option<i32>,
        cfg: &IndexingConfig,
    ) {
        let Ok(at) = usize::try_from(doc) else {
            return;
        };
        self.doc_values.record(doc, &fields.doc_values, |n| {
            cfg.field_by_number(n).map(|f| f.doc_values_type)
        });
        self.chunk.record(doc, fields, cfg);
        if self.chunk.gave_up() || self.doc_values.refused() {
            // Chunks are decoded from now on: neither set of columns is read.
            self.chunk.give_up();
            self.doc_values.refuse();
        }
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

    /// Frees the columns kept only for writing chunks, keeping the points'
    /// bounds views read.
    pub(super) fn drop_chunk_columns(&mut self) {
        self.doc_values = doc_values::DocValuesColumns::default();
        self.chunk.drop_columns();
    }

    /// Documents `start..end` as columns, numbered from `start`; `None`
    /// when they must be decoded to be written.
    fn chunk_columns(&self, start: i32, end: i32) -> Option<crate::index_writer::ChunkColumns> {
        let doc_values = self.doc_values.range(start, end)?;
        self.columns_besides_doc_values(start, end, doc_values, true)
    }

    /// [`Self::chunk_columns`] with each field's doc values copied out as
    /// kept, to be put in the writer's shape only for the fields read, and
    /// the points only with `points`.
    fn lean_columns(
        &self,
        start: i32,
        end: i32,
        points: bool,
    ) -> Option<(crate::index_writer::ChunkColumns, Vec<(i32, DocValuesRun)>)> {
        let runs = self.doc_values.runs(start, end)?;
        Some((
            self.columns_besides_doc_values(start, end, Vec::new(), points)?,
            runs,
        ))
    }

    fn columns_besides_doc_values(
        &self,
        start: i32,
        end: i32,
        doc_values: Vec<(i32, crate::index_writer::PrebuiltDocValues)>,
        points: bool,
    ) -> Option<crate::index_writer::ChunkColumns> {
        let (present, points) = self.chunk.range(start, end, points)?;
        let (from, to) = (usize::try_from(start).ok()?, usize::try_from(end).ok()?);
        let norms = self
            .norms
            .iter()
            .filter_map(|(&n, column)| {
                let mut values: Vec<Option<i64>> = vec![None; to.saturating_sub(from)];
                let mut any = false;
                for (s, e) in self.chunk.normed_in(n, start, end) {
                    let (s, e) = (usize::try_from(s).ok()?, usize::try_from(e).ok()?);
                    for d in s..e {
                        if let Some(v) = values.get_mut(d.saturating_sub(from)) {
                            *v = Some(column.get(d).unwrap_or(0));
                            any = true;
                        }
                    }
                }
                any.then_some((n, values))
            })
            .collect();
        Some(crate::index_writer::ChunkColumns {
            present,
            norms,
            doc_values,
            points,
        })
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
            .saturating_add(
                self.graphs
                    .iter()
                    .map(|g| g.graph.ram_bytes())
                    .fold(0usize, usize::saturating_add),
            )
    }

    /// What the columns kept only for writing chunks hold.
    pub(super) fn chunk_ram_bytes(&self) -> usize {
        self.doc_values
            .ram_bytes()
            .saturating_add(self.chunk.ram_bytes())
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
                runs.live_docs_of(bytes, 0, docs);
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

/// One buffered document's doc values for a field
/// ([`ConcurrentIndexWriter::live_doc_values`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LiveDocValue {
    /// NUMERIC and SORTED_NUMERIC.
    Numbers(Vec<i64>),
    /// BINARY, SORTED and SORTED_SET.
    Bytes(Vec<Vec<u8>>),
}

impl LiveDocValue {
    /// `values` as a segment of field type `kind` holds them; `None` when
    /// there are none of the field's kind.
    fn of<'a>(kind: DocValuesType, values: impl Iterator<Item = &'a FieldValue>) -> Option<Self> {
        let number = |v: &FieldValue| match v {
            FieldValue::Long(x) => Some(*x),
            FieldValue::Int(x) => Some(i64::from(*x)),
            _ => None,
        };
        let bytes = |v: &FieldValue| match v {
            FieldValue::String(s) => Some(s.as_bytes().to_vec()),
            FieldValue::Binary(b) => Some(b.clone()),
            _ => None,
        };
        let out = match kind {
            DocValuesType::Numeric => Self::Numbers(values.filter_map(number).take(1).collect()),
            DocValuesType::SortedNumeric => {
                let mut v: Vec<i64> = values.filter_map(number).collect();
                v.sort_unstable();
                Self::Numbers(v)
            }
            DocValuesType::Binary | DocValuesType::Sorted => {
                Self::Bytes(values.filter_map(bytes).take(1).collect())
            }
            DocValuesType::SortedSet => {
                let mut v: Vec<Vec<u8>> = values.filter_map(bytes).collect();
                v.sort_unstable();
                v.dedup();
                Self::Bytes(v)
            }
            _ => return None,
        };
        match &out {
            Self::Numbers(v) if v.is_empty() => None,
            Self::Bytes(v) if v.is_empty() => None,
            _ => Some(out),
        }
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
    /// The published segment's files, pinned while the buffer is kept so a
    /// view still opens them after a merge replaced the segment.
    pub(super) held_files: Vec<String>,
    /// What chunks without postings are written from until the buffer is
    /// published, beside `columns`' chunk columns.
    pub(super) lean: Option<LeanSource>,
    /// The documents' fields as indexed (full blocks shared with the
    /// flush), for reading one document's doc values
    /// ([`ConcurrentIndexWriter::live_doc_values`]).
    pub(super) explicit: PackedFields,
}

/// What a chunk without postings is written from, copied out of its
/// buffer so the buffer's lock is not held while it is written.
struct Copied {
    explicit: PackedFields,
    /// Documents `start..end` of the chunk; empty for a flushing buffer.
    docs: Vec<lucene_codecs::stored_fields::Document>,
    custom: Vec<Vec<(String, i32)>>,
    vectors: Vec<Vec<crate::index_writer::DocumentVector>>,
    has_blocks: bool,
    partition: Option<Arc<str>>,
}

impl Flushing {
    /// Frees what chunks of the buffer were written from.
    fn drop_lean(&mut self) {
        self.columns.drop_chunk_columns();
        self.lean = None;
    }
}

/// What a chunk of a flushing buffer without postings is written from,
/// beside its packed fields ([`ConcurrentIndexWriter::live_lean_segment`]). Kept only for a buffer
/// whose documents carry no fields or vectors of their own.
pub(super) struct LeanSource {
    pub(super) has_blocks: bool,
    pub(super) partition: Option<Arc<str>>,
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
    /// Open views per buffer id that read it as chunks
    /// ([`ViewBuffer::chunks`]): its [`Flushing::lean`] source stays after
    /// its publish while there are any.
    lean_pins: HashMap<u64, usize>,
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
    /// Files of published buffers no longer kept, to unpin
    /// ([`Self::take_unpinned`]) under the writer's lock.
    unpinned: Vec<String>,
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
        held_files: Vec<String>,
    ) {
        if let Some(mut f) = self.flushing.remove(&ticket) {
            // A published buffer is read as its segment by views to come.
            if !published || !self.lean_pins.contains_key(&f.columns.id) {
                f.drop_lean();
            }
            if published {
                f.segment = segment;
                f.held_files = held_files;
                self.retained.insert(self.published, f);
                self.published = self.published.saturating_add(1);
                return;
            }
        }
        self.unpinned.extend(held_files);
    }

    /// The files to unpin: those of the published buffers dropped since.
    pub(super) fn take_unpinned(&mut self) -> Vec<String> {
        std::mem::take(&mut self.unpinned)
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
        let (released, pins, unpinned) = (self.released, &self.pins, &mut self.unpinned);
        self.retained.retain(|&n, f| {
            let keep = n >= released || pins.contains_key(&f.columns.id);
            if !keep {
                unpinned.append(&mut f.held_files);
            }
            keep
        });
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
        for f in self.retained.values_mut() {
            self.unpinned.append(&mut f.held_files);
        }
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
            explicit: &f.explicit,
            num_docs: usize::try_from(f.max_doc).unwrap_or(0),
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
    patterns: &'a [LivePattern],
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
    /// Whether the view reads the buffer as chunks: it was in its slot, or
    /// being flushed with what chunks are written from kept. Else it is
    /// read as the segment it is published as.
    pub chunks: bool,
    /// Every field's statistics as of the view, by field number.
    stats: HashMap<i32, LiveFieldStats>,
    /// By field number, its points' bounds as of the view.
    bounds: HashMap<i32, PointBounds>,
    /// The vector fields' graphs over the buffer, each with how many of
    /// its vectors the view holds.
    pub vector_graphs: Vec<ViewGraph>,
}

/// A vector field's graph over a [`ViewBuffer`]'s buffer.
#[derive(Debug, Clone)]
pub struct ViewGraph {
    pub field_number: i32,
    pub graph: Arc<lucene_codecs::buffer_vectors::BufferGraph>,
    /// The field's vectors among the view's documents.
    pub count: i32,
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
    explicit: &'a PackedFields,
    /// How many documents the buffer holds now.
    num_docs: usize,
}

impl ConcurrentIndexWriter<'_> {
    /// Keeps what live search reads ([`Self::open_live_view`]) as documents
    /// arrive, from now on. Fails once a document is buffered: a buffer
    /// filled without it cannot be read.
    pub fn enable_live_search(&self) -> Result<()> {
        let slots: Vec<_> = self.slots.iter().map(write_slot).collect();
        if slots.iter().any(|d| d.has_docs()) {
            return Err(Error::LiveSearch(
                "live search must be enabled before the first document".into(),
            ));
        }
        // Locked before the log, as `hand_over` holds it: a delete handed
        // over meanwhile is either before `handed` or kept in the history.
        let mut live = self.live_flushes();
        let handed = lock(&self.log).handed;
        live.keep_history(handed);
        drop(live);
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
            .map(|s| read_slot(s).live.ram_bytes())
            .fold(0usize, usize::saturating_add);
        let live = self.live_flushes();
        let retained = live
            .retained
            .values()
            .map(|f| f.postings.bytes())
            .fold(slots, usize::saturating_add);
        live.buffers()
            .map(|f| f.columns.ram_bytes().saturating_add(f.explicit.ram_bytes()))
            .fold(retained, usize::saturating_add)
            .saturating_add(live.history_bytes)
    }

    /// Stops reading the published buffers the commit at `generation` holds,
    /// once no open view reads them: call it once no search reads from an
    /// older commit. Their memory goes.
    pub fn release_live(&self, generation: i64) -> Result<()> {
        let unpinned = {
            let mut live = self.live_flushes();
            let held = live.commit(generation)?.held;
            live.release(held);
            live.take_unpinned()
        };
        self.unpin(unpinned);
        Ok(())
    }

    /// Hands the files of published buffers no longer kept back to the
    /// writer, which deletes those nothing else names. Takes the writer's
    /// lock, so is called without the live lock held.
    fn unpin(&self, files: Vec<String>) {
        if !files.is_empty() {
            self.unpin_with(&mut lock(&self.core).writer, files);
        }
    }

    /// [`Self::unpin`] under the writer's lock, held as `writer`. Files the
    /// writer fails to release are kept for the next try.
    pub(super) fn unpin_with(
        &self,
        writer: &mut crate::index_writer::IndexWriter<'_>,
        files: Vec<String>,
    ) {
        if !files.is_empty() && writer.release_files(&files).is_err() {
            self.live_flushes().unpinned.extend(files);
        }
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
        let opened = std::time::Instant::now();
        let cfg = self.check_live()?;
        let slots: Vec<_> = self.slots.iter().map(read_slot).collect();
        crate::live_timing::record(crate::live_timing::Phase::ViewOpenLocks, opened.elapsed());
        let core_wait = crate::live_timing::start(crate::live_timing::Phase::CoreWait);
        let mut live = self.live_flushes();
        drop(core_wait);
        let commit = commit.map(|g| live.commit(g)).transpose()?;
        let from = commit.map_or(live.published, |c| c.held);
        if !live.keeps_from(from) {
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
            Some(c) => Some(live.deletes_after(c, &flushing_tail).ok_or_else(|| {
                Error::LiveSearch(format!(
                    "the deletes since commit {} are no longer kept",
                    c.generation
                ))
            })?),
            None => None,
        };
        let mut out = LiveView {
            buffers: Vec::new(),
            published: live.published,
            commit_deletes: Vec::new(),
        };
        for (dwpt, tail) in slots.iter().zip(&slot_tails) {
            if !dwpt.has_docs() {
                continue;
            }
            let max_doc = i32::try_from(dwpt.num_docs()).unwrap_or(i32::MAX);
            let source = Source {
                postings: &dwpt.postings,
                columns: &dwpt.live,
                explicit: &dwpt.explicit,
                num_docs: dwpt.num_docs(),
            };
            let mut buffer = view_buffer(&source, tail, max_doc, &cfg)?;
            buffer.in_slot = true;
            buffer.chunks = true;
            out.buffers.push(buffer);
        }
        let published = live.retained.range(from..).map(|(_, f)| (f, false));
        let flushing = live.flushing.values().map(|f| (f, f.lean.is_some()));
        for (f, chunks) in flushing.chain(published) {
            let source = Source {
                postings: &f.postings,
                columns: &f.columns,
                explicit: &f.explicit,
                num_docs: usize::try_from(f.max_doc).unwrap_or(0),
            };
            let mut buffer = view_buffer(&source, &flushing_tail, f.max_doc, &cfg)?;
            buffer.chunks = chunks;
            out.buffers.push(buffer);
        }
        for b in &out.buffers {
            let pins = live.pins.entry(b.id).or_insert(0);
            *pins = pins.saturating_add(1);
            if b.chunks {
                let pins = live.lean_pins.entry(b.id).or_insert(0);
                *pins = pins.saturating_add(1);
            }
        }
        drop(live);
        drop(slots);
        crate::live_timing::record(crate::live_timing::Phase::ViewOpen, opened.elapsed());
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
        self.live_flushes()
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
    /// The buffer's slot is held for reading while a segment with postings
    /// is written; one without them copies the documents' columns out and
    /// is written with the slot free. Fails when
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
        self.live_segment_with(
            id,
            range,
            doc_counts,
            next_starts,
            postings,
            None,
            dir,
            name,
        )
        .and_then(|s| match s {
            LeanSegment::Written(sci, _) => Ok(sci),
            LeanSegment::Parts(..) => Err(Error::LiveSearch(
                "a chunk for a search alone was asked for another".into(),
            )),
        })
    }

    /// [`Self::live_buffer_segment`] without postings, for one search: the
    /// segment's doc values are not written but given back as columns, for
    /// its reader to encode a field when it first reads it, and its points
    /// are written only with `points`. Written as `live_buffer_segment`
    /// writes it, doc values and points included, when the buffer keeps no
    /// columns.
    #[allow(clippy::too_many_arguments)]
    pub fn live_lean_segment(
        &self,
        id: u64,
        range: std::ops::Range<i32>,
        doc_counts: HashMap<i32, i32>,
        next_starts: &[i32],
        points: bool,
        dir: &dyn lucene_store::Directory,
        name: &str,
    ) -> Result<LeanSegment> {
        self.live_segment_with(
            id,
            range,
            doc_counts,
            next_starts,
            false,
            Some(points),
            dir,
            name,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn live_segment_with(
        &self,
        id: u64,
        range: std::ops::Range<i32>,
        doc_counts: HashMap<i32, i32>,
        next_starts: &[i32],
        postings: bool,
        lazy: Option<bool>,
        dir: &dyn lucene_store::Directory,
        name: &str,
    ) -> Result<LeanSegment> {
        let started = std::time::Instant::now();
        let cfg = self.check_live()?;
        if cfg.sorts_documents() {
            return Err(Error::LiveSearch(
                "an index sort renumbers the buffered documents".into(),
            ));
        }
        let flushed = || Error::LiveSearch(format!("buffer {id} is being flushed"));
        let (Ok(start), Ok(end)) = (usize::try_from(range.start), usize::try_from(range.end))
        else {
            return Err(Error::LiveSearch("a negative document range".into()));
        };
        // A view copies each field's doc values out as kept, and puts in
        // the writer's shape only those its searches read.
        let columns_of = |c: &LiveColumns| {
            if decode_chunks() {
                None
            } else if lazy.is_some() && cfg.codec.is_none() {
                c.lean_columns(range.start, range.end, lazy == Some(true))
                    .map(|(c, runs)| (c, Some(runs)))
            } else {
                c.chunk_columns(range.start, range.end).map(|c| (c, None))
            }
        };
        let waiting = std::time::Instant::now();
        let slot = self.slot_of(id).filter(|d| d.has_docs());
        // A buffer gone from its slot is written, without postings, from
        // what it keeps for that until it is published and the views that
        // read it as chunks are closed.
        let (num_docs, columns, mut copied) = match &slot {
            Some(slot) => {
                crate::live_timing::record(
                    crate::live_timing::Phase::ChunkSlotWait,
                    waiting.elapsed(),
                );
                (slot.num_docs(), columns_of(&slot.live), None)
            }
            None if postings => return Err(flushed()),
            None => {
                let live = self.live_flushes();
                let f = live
                    .flushing
                    .values()
                    .chain(live.retained.values())
                    .find(|f| f.columns.id == id)
                    .ok_or_else(flushed)?;
                let source = f.lean.as_ref().ok_or_else(flushed)?;
                let copied = Copied {
                    explicit: f.explicit.snapshot(),
                    docs: Vec::new(),
                    custom: Vec::new(),
                    vectors: Vec::new(),
                    has_blocks: source.has_blocks,
                    partition: source.partition.clone(),
                };
                let num_docs = usize::try_from(f.max_doc).unwrap_or(0);
                (num_docs, columns_of(&f.columns), Some(copied))
            }
        };
        if start >= end || end > num_docs {
            return Err(Error::LiveSearch(format!(
                "documents {start}..{end} of a buffer of {num_docs}"
            )));
        }
        let (mut columns, runs) = columns.map_or((None, None), |(c, runs)| (Some(c), runs));
        let lazy_doc_values = match (runs, columns.as_mut()) {
            (Some(runs), Some(c))
                if crate::index_writer::LazyDocValues::covers(&cfg, &runs, &c.present) =>
            {
                Some(crate::index_writer::LazyDocValues::new(
                    Arc::clone(&cfg),
                    runs,
                    range.end.saturating_sub(range.start),
                ))
            }
            (Some(runs), Some(c)) => {
                match runs
                    .iter()
                    .map(|(n, r)| Some((*n, r.build()?)))
                    .collect::<Option<Vec<_>>>()
                {
                    // Columns copied without their points cannot write a
                    // segment that has them: the documents are decoded.
                    Some(doc_values) if lazy != Some(false) => c.doc_values = doc_values,
                    _ => columns = None,
                }
                None
            }
            _ => None,
        };
        // Only the columns hold the run's fields: no held documents,
        // custom-frequency terms, vectors or blocks of their own.
        let only_columns = match (&slot, &copied) {
            (Some(slot), _) => {
                let run = end.saturating_sub(start);
                !slot.has_blocks
                    && slot
                        .docs
                        .iter()
                        .skip(start)
                        .take(run)
                        .all(|d| d.fields.is_empty())
                    && slot
                        .custom_freq_terms
                        .iter()
                        .skip(start)
                        .take(run)
                        .all(Vec::is_empty)
                    && slot.vectors.iter().skip(start).take(run).all(Vec::is_empty)
            }
            (None, Some(c)) => {
                !c.has_blocks && c.docs.is_empty() && c.custom.is_empty() && c.vectors.is_empty()
            }
            (None, None) => false,
        };
        if let (Some(points), Some(c), true) = (lazy, columns.as_ref(), only_columns) {
            if lazy_doc_values.is_some() && cfg.takes_lean_parts(c) {
                // Built from the columns alone, with the slot free.
                let partition = match (&slot, &copied) {
                    (Some(slot), _) => slot.partition.clone(),
                    (None, Some(c)) => c.partition.clone(),
                    (None, None) => None,
                };
                drop(slot);
                let parts = cfg.build_lean_parts(
                    c,
                    range.end.saturating_sub(range.start),
                    points,
                    partition.as_deref(),
                    lucene_util::string_helper::random_id(),
                )?;
                crate::live_timing::record(crate::live_timing::Phase::ChunkLean, started.elapsed());
                let lazy_doc_values = lazy_doc_values.ok_or_else(flushed)?;
                return Ok(LeanSegment::Parts(Box::new(parts), lazy_doc_values));
            }
        }
        let lean = lazy_doc_values.is_some();
        let limit = crate::index_writer::BufferLimit {
            start: range.start,
            docs: range.end,
            doc_counts,
            postings,
            columns,
            doc_values: !lean,
            points: !lean || lazy == Some(true),
        };
        let write = |explicit: &PackedFields,
                     pending: &PendingPostings,
                     docs: &[_],
                     custom: &[_],
                     vectors: &[_],
                     has_blocks,
                     partition: Option<&str>| {
            let buffer = crate::index_writer::DocumentBuffer {
                docs,
                custom_freq_terms: custom,
                vectors,
                explicit: crate::index_writer::ExplicitDocs::PackedRange(explicit, start, end),
                postings: Some(pending),
                old_to_new: None,
                has_blocks,
                stored: None,
                limit: Some(&limit),
                partition,
                graphs: &[],
            };
            cfg.build_and_write_segment(
                dir,
                &buffer,
                name,
                lucene_util::string_helper::random_id(),
                None,
            )
            .map(|(sci, _)| sci)
        };
        let sci = match slot {
            Some(slot) if postings => {
                // Documents counted but not held (`Dwpt::stored_docs`) are
                // empty.
                let sci = write(
                    &slot.explicit,
                    &slot.postings,
                    &padded(&slot.docs, start, end),
                    &padded_or_empty(&slot.custom_freq_terms, start, end),
                    &padded_or_empty(&slot.vectors, start, end),
                    slot.has_blocks,
                    slot.partition.as_deref(),
                )?;
                slot.postings.keep_cursors_at(next_starts);
                sci
            }
            slot => {
                // Without postings the chunk reads only the documents' own
                // columns: they are copied out (the packed fields' full
                // blocks shared) and written with the slot free for
                // indexing.
                let c = match (slot, copied.take()) {
                    (Some(slot), _) => Copied {
                        explicit: slot.explicit.snapshot(),
                        docs: padded(&slot.docs, start, end).into_owned(),
                        custom: padded_or_empty(&slot.custom_freq_terms, start, end).into_owned(),
                        vectors: padded_or_empty(&slot.vectors, start, end).into_owned(),
                        has_blocks: slot.has_blocks,
                        partition: slot.partition.clone(),
                    },
                    (None, Some(c)) => c,
                    (None, None) => return Err(flushed()),
                };
                // A flushing buffer keeps no documents: they are all empty.
                let docs = padded(&c.docs, 0, end.saturating_sub(start));
                write(
                    &c.explicit,
                    &PendingPostings::default(),
                    &docs,
                    &c.custom,
                    &c.vectors,
                    c.has_blocks,
                    c.partition.as_deref(),
                )?
            }
        };
        crate::live_timing::record(
            if lazy.is_some() {
                crate::live_timing::Phase::ChunkLean
            } else {
                crate::live_timing::Phase::ChunkWrite
            },
            started.elapsed(),
        );
        Ok(LeanSegment::Written(sci, lazy_doc_values))
    }

    /// The directory the writer's segments are in.
    pub fn directory(&self) -> &dyn lucene_store::Directory {
        self.dir
    }

    /// Lets the buffers `view` read go once no other view reads them.
    pub fn close_live_view(&self, view: &LiveView) {
        let mut live = self.live_flushes();
        for b in &view.buffers {
            if let Some(n) = live.pins.get_mut(&b.id) {
                *n = n.saturating_sub(1);
                if *n == 0 {
                    live.pins.remove(&b.id);
                }
            }
            if !b.chunks {
                continue;
            }
            if let Some(n) = live.lean_pins.get_mut(&b.id) {
                *n = n.saturating_sub(1);
                if *n == 0 {
                    live.lean_pins.remove(&b.id);
                    if let Some(f) = live.retained.values_mut().find(|f| f.columns.id == b.id) {
                        f.drop_lean();
                    }
                }
            }
        }
        let released = live.released;
        live.release(released);
        let unpinned = live.take_unpinned();
        drop(live);
        self.unpin(unpinned);
    }

    /// The buffers in the slots now, each read under its own slot's lock,
    /// without their deletes and not pinned: what writing their newest
    /// documents into chunks ([`Self::live_buffer_segment`]) needs, more
    /// cheaply than a view.
    pub fn live_slot_buffers(&self) -> Vec<ViewBuffer> {
        if !self.live() {
            return Vec::new();
        }
        self.slots
            .iter()
            .filter_map(|s| {
                let d = read_slot(s);
                d.has_docs().then(|| ViewBuffer {
                    id: d.live.id,
                    max_doc: i32::try_from(d.num_docs()).unwrap_or(i32::MAX),
                    deleted: Vec::new(),
                    in_slot: true,
                    chunks: true,
                    stats: d.postings.live_stats_all(),
                    bounds: HashMap::new(),
                    vector_graphs: Vec::new(),
                })
            })
            .collect()
    }

    /// The slot holding buffer `id`, locked. The free slots are looked at
    /// first, so a slot another thread holds (writing one of its own chunks,
    /// say) is waited for only when the buffer is in no free one. A buffer
    /// never moves into a slot, so the second look cannot miss it.
    fn slot_of(&self, id: u64) -> Option<RwLockReadGuard<'_, Dwpt>> {
        let mut busy = Vec::new();
        for slot in &self.slots {
            match slot.try_read() {
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
        busy.into_iter().map(read_slot).find(|d| d.live.id == id)
    }

    /// Runs `f` on buffer `id`'s postings and columns, wherever it is now.
    /// A buffer only moves from a slot to the control lock, so looking in
    /// the slots one at a time, then there, cannot miss it.
    fn with_buffer<T>(&self, id: u64, f: impl Fn(Option<&Source<'_>>) -> T) -> T {
        if let Some(d) = self.slot_of(id) {
            return f(Some(&Source {
                postings: &d.postings,
                columns: &d.live,
                explicit: &d.explicit,
                num_docs: d.num_docs(),
            }));
        }
        let core_wait = crate::live_timing::start(crate::live_timing::Phase::CoreWait);
        let live = self.live_flushes();
        drop(core_wait);
        f(live.find(id).as_ref())
    }

    /// The requested fields and terms of `view`'s documents, with each
    /// buffer's norms, deletes and field statistics as of the view. Fields
    /// the schema does not know, and terms no buffer holds, are left out.
    pub fn live_view_snapshot(
        &self,
        view: &LiveView,
        request: &LiveRequest,
    ) -> Result<LiveSnapshot> {
        self.live_view_snapshot_from(view, request, &[])
    }

    /// [`Self::live_view_snapshot`], each buffer's postings without
    /// positions only from its document in `from` on (by buffer; 0 past
    /// its end): a term's documents before are left out, its read starting
    /// near there. A buffer read from its end on has no fields.
    pub fn live_view_snapshot_from(
        &self,
        view: &LiveView,
        request: &LiveRequest,
        from: &[i32],
    ) -> Result<LiveSnapshot> {
        crate::live_timing::timed(crate::live_timing::Phase::Snapshot, || {
            self.snapshot_from(view, request, from)
        })
    }

    fn snapshot_from(
        &self,
        view: &LiveView,
        request: &LiveRequest,
        from: &[i32],
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
                    patterns: &r.patterns,
                    positions: r.positions && info.index_options.subsumes_positions(),
                })
            })
            .collect();
        let mut out = LiveSnapshot {
            buffers: Vec::with_capacity(view.buffers.len()),
            published: view.published,
        };
        // Each buffer is read under its own slot lock: the view fixed its
        // documents and statistics, so the buffers need not be read at
        // one instant, and indexing into the other slots goes on.
        for (i, b) in view.buffers.iter().enumerate() {
            let from = from.get(i).copied().unwrap_or(0);
            let read = self.with_buffer(b.id, |source| {
                // A buffer `delete_all` or `rollback` dropped: none of its
                // documents is live any more.
                source.map_or_else(
                    || {
                        Ok(LiveBuffer {
                            max_doc: b.max_doc,
                            deleted: (0..b.max_doc).collect(),
                            fields: Vec::new(),
                        })
                    },
                    |source| read_buffer(source, b, &wanted, from),
                )
            });
            out.buffers.push(read?);
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

    /// The doc values of document `doc` of `view`'s buffer `buffer`, for
    /// each of `fields` the document has, as a segment would hold them:
    /// SORTED_NUMERIC values sorted, SORTED_SET values sorted without
    /// repeats, and a single-valued field's first value. They are the
    /// values the document was added with: a doc-values update to a
    /// buffered document is not applied. Fails where
    /// [`Self::live_document`] fails, and for a buffer whose documents were
    /// not all added as explicit fields.
    pub fn live_doc_values(
        &self,
        view: &LiveView,
        buffer: usize,
        doc: i32,
        fields: &[i32],
    ) -> Result<Vec<(i32, LiveDocValue)>> {
        let missing =
            || Error::LiveSearch(format!("no buffered document {doc} in buffer {buffer}"));
        let b = view.buffers.get(buffer).ok_or_else(missing)?;
        if doc < 0 || doc >= b.max_doc {
            return Err(missing());
        }
        let at = usize::try_from(doc).map_err(|_| missing())?;
        let values = self
            .with_buffer(b.id, |s| {
                let s = s?;
                // Each document has packed fields only when all were
                // added as explicit fields.
                if s.explicit.len() != s.num_docs {
                    return Some(Err(Error::LiveSearch(format!(
                        "buffer {buffer} holds documents not added as explicit fields"
                    ))));
                }
                let mut out = Vec::new();
                Some(
                    s.explicit
                        .for_each_range(at, at.saturating_add(1), |_, f| {
                            out = std::mem::take(&mut f.doc_values);
                            Ok(())
                        })
                        .map(|()| out),
                )
            })
            .ok_or_else(missing)??;
        let cfg = self.cfg();
        Ok(fields
            .iter()
            .filter_map(|&n| {
                let kind = cfg.field_by_number(n)?.doc_values_type;
                let of_field = values.iter().filter(|v| v.field_number == n);
                LiveDocValue::of(kind, of_field.map(|v| &v.value)).map(|v| (n, v))
            })
            .collect())
    }

    /// The least and greatest value of each indexed dimension of field
    /// `field`'s points in `view`'s buffer `buffer`, and how many points it
    /// held, as of the view: they bound the view's documents' points.
    /// `None` when no document of it gave the field a point.
    pub fn live_point_bounds(
        &self,
        view: &LiveView,
        buffer: usize,
        field: i32,
    ) -> Result<Option<PointBounds>> {
        self.check_live()?;
        let b = view
            .buffers
            .get(buffer)
            .ok_or_else(|| Error::LiveSearch(format!("no buffer {buffer} in the view")))?;
        Ok(b.bounds.get(&field).filter(|p| p.count > 0).cloned())
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

/// A run of buffered documents for one search
/// ([`ConcurrentIndexWriter::live_lean_segment`]).
pub enum LeanSegment {
    /// Built in memory: what a reader of the run needs besides its doc
    /// values, which are encoded a field at a time as they are read.
    Parts(
        Box<crate::index_writer::LeanParts>,
        crate::index_writer::LazyDocValues,
    ),
    /// Written to the directory, with its doc values when they were not.
    Written(
        crate::segment_infos::SegmentCommitInfo,
        Option<crate::index_writer::LazyDocValues>,
    ),
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
        chunks: false,
        stats: source.postings.live_stats_all(),
        bounds: source.columns.chunk.all_bounds().clone(),
        vector_graphs: source
            .columns
            .graphs
            .iter()
            .map(|g| ViewGraph {
                field_number: g.field_number,
                graph: Arc::clone(&g.graph),
                count: g.graph.len(),
            })
            .collect(),
    })
}

/// The requested terms of `view`'s documents of one buffer.
fn read_buffer(
    source: &Source<'_>,
    view: &ViewBuffer,
    wanted: &[Wanted<'_>],
    from: i32,
) -> Result<LiveBuffer> {
    let from_doc = usize::try_from(from).unwrap_or(0);
    if from >= view.max_doc {
        // Nothing of it is asked for: neither its norms nor its positions
        // are read.
        return Ok(LiveBuffer {
            max_doc: view.max_doc,
            deleted: view.deleted.clone(),
            fields: Vec::new(),
        });
    }
    let max = usize::try_from(view.max_doc).unwrap_or(0);
    let mut fields = Vec::new();
    for w in wanted {
        let (Some(runs), Some(stats)) = (
            source.postings.live_field(w.info.number),
            view.stats.get(&w.info.number),
        ) else {
            continue;
        };
        let matched: Vec<&[u8]>;
        let names: &[&[u8]] = if w.patterns.is_empty() {
            &w.terms
        } else {
            let mut all = w.terms.clone();
            all.extend(runs.live_matching(|t| w.patterns.iter().any(|p| p.matches(t))));
            all.sort_unstable();
            all.dedup();
            matched = all;
            &matched
        };
        let mut terms = Vec::new();
        for &t in names {
            if let Some(mut p) = runs.live_term(t, w.positions, from)? {
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
        // Only the documents from `from` on are read, as their terms are.
        let norms = source.columns.norms.get(&w.info.number);
        let norms = (!w.info.omit_norms).then(|| {
            norms.map_or_else(
                || vec![0; max.saturating_sub(from_doc)],
                |c| c.to_vec_from(from_doc, max),
            )
        });
        fields.push(LiveBufferField {
            info,
            terms,
            norms,
            norms_from: from.max(0),
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

/// `list[start..end]`, with default entries for the documents past the end of `list`.
fn padded<T: Clone + Default>(list: &[T], start: usize, end: usize) -> std::borrow::Cow<'_, [T]> {
    match list.get(start..end) {
        Some(held) => held.into(),
        None => {
            let mut out = list.get(start..).unwrap_or_default().to_vec();
            out.resize_with(end.saturating_sub(start), T::default);
            out.into()
        }
    }
}

/// Like [`padded`], but an empty list (no document has an entry) stays empty.
fn padded_or_empty<T: Clone + Default>(
    list: &[T],
    start: usize,
    end: usize,
) -> std::borrow::Cow<'_, [T]> {
    if list.is_empty() {
        list.into()
    } else {
        padded(list, start, end)
    }
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

mod chunk_fields;
pub use chunk_fields::PointBounds;

#[cfg(test)]
thread_local! {
    /// Chunks of this thread are written from their decoded documents.
    static DECODE_CHUNKS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn decode_chunks() -> bool {
    #[cfg(test)]
    {
        DECODE_CHUNKS.with(std::cell::Cell::get)
    }
    #[cfg(not(test))]
    {
        false
    }
}

mod doc_values;
pub(crate) use doc_values::DocValuesRun;
pub use doc_values::KeywordRun;
#[cfg(test)]
mod tests;
