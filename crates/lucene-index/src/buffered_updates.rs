//! Buffered deletes and doc-values updates, and the sequence numbers that
//! order them against document adds — this port of
//! `org/apache/lucene/index/{DocumentsWriterDeleteQueue,BufferedUpdates,
//! FrozenBufferedUpdates,DocValuesUpdate,BufferedUpdatesStream}.java`.
//!
//! # The contract this exists to provide
//!
//! Real Lucene's mutating `IndexWriter` methods (`addDocument`,
//! `updateDocument`, `deleteDocuments`, `updateDocValues`, …) all return a
//! **sequence number**: a monotonically increasing `long` that defines the
//! effective serialization of every operation on the index. Three properties
//! follow from it, and all three are what this module is for:
//!
//! 1. **Every mutating operation gets a strictly larger seqNo than the one
//!    before it** ([`DeleteQueue::next_sequence_number`]).
//! 2. **A delete applies to every document with a smaller seqNo and to no
//!    document with a larger one.** Java gets this from
//!    `BufferedUpdates.addTerm(term, docIDUpto)`: the delete records how many
//!    documents the in-RAM buffer held when it was issued, and at flush only
//!    doc IDs *below* that limit are deleted from the new segment.
//!    [`BufferedUpdates::add_term`] is that, with the same
//!    "highest `docIDUpto` wins" rule for a repeated term.
//! 3. **A segment applies exactly the deletes whose seqNo precedes its
//!    flush.** Java gets this from [`FrozenBufferedUpdates::del_gen`] versus
//!    `SegmentCommitInfo.getBufferedDeletesGen()`: a frozen packet applies to
//!    a segment iff the segment's buffered-deletes generation is *lower* than
//!    the packet's (`FrozenBufferedUpdates.applyTermDeletes`: `if
//!    (segState.delGen > delGen) continue;`), so a delete issued after a
//!    segment was flushed does not reach into it, and a delete issued before
//!    it does. [`BufferedUpdatesStream`] and
//!    [`crate::segment_infos::SegmentCommitInfo::buffered_deletes_gen`] are
//!    that pair.
//!
//! # What is deliberately different from Java
//!
//! **No lock-free linked list.** Java's `DocumentsWriterDeleteQueue` is a
//! singly linked list of `Node`s with a per-`DocumentsWriterPerThread`
//! `DeleteSlice` head, and a global slice for the deletes that apply to
//! already-written segments. The entire structure exists to let many indexing
//! threads append deletes without a lock and still agree on ordering — the
//! class doc says so explicitly ("a non-blocking linked pending deletes
//! queue"). This port has **one** indexing thread by construction (see
//! [`crate::index_writer`]'s module doc: one caller, one `Directory`,
//! sequential calls), so there is exactly one private slice, it is never
//! contended, and "apply the slice lazily at the next `finishDocuments`" is
//! observationally identical to "apply it eagerly when the delete is issued".
//! [`DeleteQueue`] therefore keeps the two things the slices actually
//! *compute* — the private [`BufferedUpdates`] for the segment being built
//! and the global [`BufferedUpdates`] for the segments already written — and
//! drops the list, the slice heads and the locks. See
//! `docs/sweep/m2/c7-delete-queue.md` for the equivalence argument
//! document-by-document.
//!
//! **`Query` is not available here.** Java's `deleteDocuments(Query...)`
//! takes `org.apache.lucene.search.Query` and resolves it with an
//! `IndexSearcher`. This port's dependency graph is strictly downward
//! (`util ← store ← codecs ← index ← search ← core ← ffi`) and
//! `lucene-search` depends on `lucene-index`, so `lucene-index` cannot name
//! a `lucene_search::Query` without inverting that edge — the same constraint
//! [`crate::term_delete`] and [`crate::points_delete`] already document.
//! [`DeleteQuery`] is therefore a small, closed enum of the query shapes this
//! crate can resolve with the primitives it *does* own (a blocktree term
//! dictionary and a postings reader): exact term, term prefix, term range,
//! match-all, and boolean composition of those. `lucene-search`/`lucene-ffi`
//! can lower their richer `Query` onto it. Anything outside that set is a
//! caller-side resolution today, exactly as it was before this module
//! existed.

use std::collections::HashMap;

/// A sequence number: real Lucene's `IndexWriter` return value from every
/// mutating method. Starts at 1 (`DocumentsWriterDeleteQueue`'s constructor
/// comment: "seqNo must start at 1 because some APIs negate this to also
/// return a boolean") and increases by one per operation.
pub type SeqNo = i64;

/// `DocumentsWriterDeleteQueue`'s starting sequence number.
pub const FIRST_SEQ_NO: SeqNo = 1;

/// The highest sequence number [`DeleteQueue::skip_sequence_numbers`] will
/// leave the counter at.
///
/// Half the `i64` range, reserved as headroom for exactly the same reason
/// [`crate::segment_infos::MAX_GENERATION`] reserves it: the counter is
/// otherwise stepped one at a time per indexing operation, and keeping it
/// below this ceiling is what makes that `+ 1` unable to overflow however
/// absurd a jump a caller asks for. Reaching `i64::MAX` from here would take
/// 2^62 further operations in one writer session.
pub const MAX_SEQ_NO: SeqNo = i64::MAX / 2;

/// Java's `BufferedUpdates.MAX_INT`: the `docIDUpto` that means "no limit —
/// this delete applies to every document in the segment". Used for the
/// global buffer, whose deletes target segments that were already complete
/// when the delete was issued.
pub const MAX_DOC_ID_UPTO: i32 = i32::MAX;

/// `org.apache.lucene.index.Term`: a `(field, bytes)` pair.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Term {
    pub field: String,
    pub bytes: Vec<u8>,
}

impl Term {
    pub fn new(field: impl Into<String>, bytes: impl Into<Vec<u8>>) -> Self {
        Term {
            field: field.into(),
            bytes: bytes.into(),
        }
    }
}

/// The query shapes `deleteDocuments(Query...)` can carry in this port — see
/// the module doc for why this is a closed enum rather than
/// `lucene_search::Query`. Resolution lives in
/// [`crate::index_writer::IndexWriter`], against one already-opened segment,
/// exactly like [`crate::term_delete::resolve_term_doc_ids`].
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum DeleteQuery {
    /// `TermQuery`.
    Term(Term),
    /// `PrefixQuery`: every term in `field` starting with `prefix`.
    Prefix { field: String, prefix: Vec<u8> },
    /// `TermRangeQuery`. `None` bounds are open.
    TermRange {
        field: String,
        lower: Option<Vec<u8>>,
        upper: Option<Vec<u8>>,
        include_lower: bool,
        include_upper: bool,
    },
    /// `MatchAllDocsQuery`. Java's `deleteDocuments(Query...)` specialises
    /// this into `deleteAll()` (LUCENE-6379); [`crate::index_writer`] does
    /// the same.
    MatchAll,
    /// `BooleanQuery` with every clause `SHOULD` (union).
    Any(Vec<DeleteQuery>),
    /// `BooleanQuery` with every clause `MUST` (intersection).
    All(Vec<DeleteQuery>),
    /// `BooleanQuery` with one `MUST_NOT` clause over `MatchAllDocsQuery`
    /// (complement, within the segment's live docs).
    Not(Box<DeleteQuery>),
}

/// `DocValuesUpdate.NumericDocValuesUpdate` / `BinaryDocValuesUpdate`: "set
/// every document matching `term`'s `field` doc-values value to `value`".
///
/// `value: None` is Java's `hasValue == false`, which
/// `FrozenBufferedUpdates.applyDocValuesUpdates` turns into
/// `DocValuesFieldUpdates.reset(doc)` — the doc's value is *removed*, not set
/// to zero. `IndexWriter.updateDocValues` reaches it by passing a field whose
/// value is null.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DocValuesUpdate {
    Numeric {
        term: Term,
        field: String,
        value: Option<i64>,
    },
    Binary {
        term: Term,
        field: String,
        value: Option<Vec<u8>>,
    },
}

impl DocValuesUpdate {
    /// `DocValuesUpdate.term`.
    pub fn term(&self) -> &Term {
        match self {
            DocValuesUpdate::Numeric { term, .. } | DocValuesUpdate::Binary { term, .. } => term,
        }
    }

    /// `DocValuesUpdate.field`.
    pub fn field(&self) -> &str {
        match self {
            DocValuesUpdate::Numeric { field, .. } | DocValuesUpdate::Binary { field, .. } => field,
        }
    }

    /// `DocValuesUpdate.type == NUMERIC`.
    pub fn is_numeric(&self) -> bool {
        matches!(self, DocValuesUpdate::Numeric { .. })
    }
}

/// One entry of Java's `FieldUpdatesBuffer`: the term that selects the
/// documents, the `docIDUpto` limit that bounds it inside a
/// segment-private packet, and the value (or its absence, a `reset`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BufferedUpdate {
    pub term: Term,
    pub doc_id_upto: i32,
    pub value: UpdateValue,
}

/// The value half of a [`BufferedUpdate`] — Java's
/// `FieldUpdatesBuffer.BufferedUpdate`'s `hasValue`/`numericValue`/
/// `binaryValue` triple, collapsed into the one enum Rust can express it as.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UpdateValue {
    /// `hasValue == false`: `DocValuesFieldUpdates.reset(doc)`.
    None,
    Numeric(i64),
    Binary(Vec<u8>),
}

/// Java's `FieldUpdatesBuffer`: every buffered update for one field, in
/// **arrival order**. Java is explicit that arrival order (not term order) is
/// what must be replayed, "so that we apply the updates in the correct order,
/// i.e. if two terms update the same document, the last one that came in
/// wins" (`FrozenBufferedUpdates.applyDocValuesUpdates`), so this is a `Vec`
/// and never a map.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FieldUpdatesBuffer {
    /// `FieldUpdatesBuffer.isNumeric()`. A field is either numeric or binary
    /// for the whole buffer's life — Java asserts the same via the update
    /// type it was constructed from.
    pub is_numeric: bool,
    pub updates: Vec<BufferedUpdate>,
}

/// `BufferedUpdates`: the deletes and doc-values updates buffered against one
/// in-progress segment (Java: one `DocumentsWriterPerThread`'s
/// `pendingUpdates`), or against every already-written segment (Java: the
/// delete queue's `globalBufferedUpdates`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BufferedUpdates {
    /// `BufferedUpdates.deleteTerms`: term -> the highest `docIDUpto` seen
    /// for it. See [`BufferedUpdates::add_term`] for why the highest wins.
    pub delete_terms: HashMap<Term, i32>,
    /// `BufferedUpdates.deleteQueries`: query -> `docIDUpto`. Java uses a
    /// `HashMap<Query,Integer>`, so a repeated query keeps the *last*
    /// `docIDUpto` (`Map.put` overwrites); this does the same.
    pub delete_queries: HashMap<DeleteQuery, i32>,
    /// `BufferedUpdates.fieldUpdates`, keyed by the *updated* field name.
    pub field_updates: HashMap<String, FieldUpdatesBuffer>,
    /// `BufferedUpdates.numFieldUpdates`.
    pub num_field_updates: usize,
}

impl BufferedUpdates {
    /// `BufferedUpdates.addTerm(Term, int)`.
    ///
    /// Java keeps the **higher** `docIDUpto` when the same term is buffered
    /// twice, and its comment explains exactly why: two threads replacing the
    /// same document can finish out of order, and blindly overwriting with a
    /// lower limit would leave the earlier document undeleted. The rule is
    /// still the right one single-threaded — `updateDocument(t, …)` twice in a
    /// row must delete both of the previously added docs, which only the
    /// higher limit does.
    pub fn add_term(&mut self, term: Term, doc_id_upto: i32) {
        match self.delete_terms.get_mut(&term) {
            Some(current) => {
                if doc_id_upto > *current {
                    *current = doc_id_upto;
                }
            }
            None => {
                self.delete_terms.insert(term, doc_id_upto);
            }
        }
    }

    /// `BufferedUpdates.addQuery(Query, int)`.
    pub fn add_query(&mut self, query: DeleteQuery, doc_id_upto: i32) {
        self.delete_queries.insert(query, doc_id_upto);
    }

    /// `BufferedUpdates.addNumericUpdate` / `addBinaryUpdate` — one method
    /// here because [`DocValuesUpdate`] already carries the discriminant Java
    /// splits on in `DocValuesUpdatesNode.apply`.
    pub fn add_doc_values_update(&mut self, update: &DocValuesUpdate, doc_id_upto: i32) {
        let is_numeric = update.is_numeric();
        let buffer = self
            .field_updates
            .entry(update.field().to_string())
            .or_insert_with(|| FieldUpdatesBuffer {
                is_numeric,
                updates: Vec::new(),
            });
        let value = match update {
            DocValuesUpdate::Numeric { value: Some(v), .. } => UpdateValue::Numeric(*v),
            DocValuesUpdate::Binary { value: Some(v), .. } => UpdateValue::Binary(v.clone()),
            // `hasValue == false` -> `reset(doc)`.
            _ => UpdateValue::None,
        };
        buffer.updates.push(BufferedUpdate {
            term: update.term().clone(),
            doc_id_upto,
            value,
        });
        // ARITH: one increment per `BufferedUpdate` pushed onto
        // `buffer.updates` immediately above, so this counter is bounded by
        // the total length of those in-memory `Vec`s -- a `usize` cannot
        // overflow counting things that fit in memory.
        #[allow(clippy::arithmetic_side_effects)]
        {
            self.num_field_updates += 1;
        }
    }

    /// `BufferedUpdates.any()`.
    pub fn any(&self) -> bool {
        !self.delete_terms.is_empty()
            || !self.delete_queries.is_empty()
            || self.num_field_updates > 0
    }

    /// `BufferedUpdates.clear()`.
    pub fn clear(&mut self) {
        self.delete_terms.clear();
        self.delete_queries.clear();
        self.field_updates.clear();
        self.num_field_updates = 0;
    }

    /// `BufferedUpdates.clearDeleteTerms()`.
    ///
    /// **Not called by this port**, deliberately. Java calls it from
    /// `DocumentsWriterPerThread.flush` once the segment-local term deletes
    /// have been folded into the new segment's live docs; this port resolves
    /// them from the written segment instead and so keeps them in the private
    /// packet (see [`DeleteQueue::freeze_private_buffer`]'s doc comment).
    /// Kept for parity, and because it is the natural spelling if that
    /// decision is ever revisited.
    pub fn clear_delete_terms(&mut self) {
        self.delete_terms.clear();
    }
}

/// `FrozenBufferedUpdates`: a [`BufferedUpdates`] made immutable and stamped
/// with the [`del_gen`](FrozenBufferedUpdates::del_gen)
/// `BufferedUpdatesStream::push` assigned it. The generation is the whole
/// point — it is what decides which segments the packet may touch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrozenBufferedUpdates {
    /// Java's `deleteTerms`, a `PrefixCodedTerms`. Kept sorted here for the
    /// same reason Java prefix-codes it sorted: a term dictionary seek walk
    /// is cheapest in term order (`TermDocsIterator(reader, true)`).
    pub delete_terms: Vec<(Term, i32)>,
    /// Java's parallel `deleteQueries`/`deleteQueryLimits` arrays.
    pub delete_queries: Vec<(DeleteQuery, i32)>,
    /// Java's `fieldUpdates`, in field-name order for determinism.
    pub field_updates: Vec<(String, FieldUpdatesBuffer)>,
    /// `FrozenBufferedUpdates.privateSegment`: `Some(segment_name)` when this
    /// packet belongs to exactly one freshly flushed segment, in which case
    /// the per-entry `docIDUpto` limits are honoured. `None` for a global
    /// packet, whose entries apply without a limit to every older segment.
    pub private_segment: Option<String>,
    /// `FrozenBufferedUpdates.delGen`, assigned by
    /// [`BufferedUpdatesStream::push`]. `-1` until then, matching Java.
    del_gen: i64,
}

impl FrozenBufferedUpdates {
    /// `new FrozenBufferedUpdates(infoStream, updates, privateSegment)`.
    pub fn new(updates: &BufferedUpdates, private_segment: Option<String>) -> Self {
        let mut delete_terms: Vec<(Term, i32)> = updates
            .delete_terms
            .iter()
            .map(|(t, &d)| (t.clone(), d))
            .collect();
        delete_terms.sort();
        let mut delete_queries: Vec<(DeleteQuery, i32)> = updates
            .delete_queries
            .iter()
            .map(|(q, &d)| (q.clone(), d))
            .collect();
        // `HashMap` iteration order is arbitrary; the apply order of
        // independent query deletes cannot change the outcome (each clears
        // bits in the same bitset), but a deterministic order keeps the
        // resulting `.liv` byte-identical run to run. Sorted by *value*, not
        // by a `Debug` rendering: no allocation per comparison, and the order
        // cannot drift when someone edits a `Debug` impl.
        delete_queries.sort_by(|a, b| a.0.cmp(&b.0));
        let mut field_updates: Vec<(String, FieldUpdatesBuffer)> = updates
            .field_updates
            .iter()
            .map(|(f, b)| (f.clone(), b.clone()))
            .collect();
        field_updates.sort_by(|a, b| a.0.cmp(&b.0));

        FrozenBufferedUpdates {
            delete_terms,
            delete_queries,
            field_updates,
            private_segment,
            del_gen: -1,
        }
    }

    /// `FrozenBufferedUpdates.any()`.
    pub fn any(&self) -> bool {
        !self.delete_terms.is_empty()
            || !self.delete_queries.is_empty()
            || !self.field_updates.is_empty()
    }

    /// `FrozenBufferedUpdates.delGen()`.
    pub fn del_gen(&self) -> i64 {
        debug_assert_ne!(self.del_gen, -1, "delGen is not yet set");
        self.del_gen
    }

    /// `FrozenBufferedUpdates.setDelGen(long)`. Java asserts it is set once;
    /// [`BufferedUpdatesStream::push`] is the only caller.
    fn set_del_gen(&mut self, del_gen: i64) {
        debug_assert_eq!(self.del_gen, -1, "delGen was already set");
        self.del_gen = del_gen;
    }

    /// Whether this packet's deletes may touch a segment whose
    /// `buffered_deletes_gen` is `segment_gen` — the single rule that makes a
    /// delete apply to the segments that existed when it was issued and to no
    /// others.
    ///
    /// Java, in `FrozenBufferedUpdates.applyTermDeletes`/`applyQueryDeletes`/
    /// `applyDocValuesUpdates`: `if (segState.delGen > delGen) continue;`
    /// — i.e. apply iff `segment_gen <= self.del_gen`. Equality happens only
    /// for a segment's own private packet, and it is exactly then that the
    /// per-entry `docIDUpto` limits are honoured
    /// ([`FrozenBufferedUpdates::limit_for`]).
    pub fn applies_to(&self, segment_gen: i64) -> bool {
        segment_gen <= self.del_gen()
    }

    /// The `docIDUpto` limit to enforce for an entry when applying this
    /// packet to a segment with generation `segment_gen`.
    ///
    /// Java: `if (delGen == segState.delGen) { limit = deleteQueryLimits[i]; }
    /// else { limit = Integer.MAX_VALUE; }`. A packet applied to an *older*
    /// segment needs no limit — every document in that segment predates every
    /// delete in the packet — while a segment's own private packet must
    /// respect the buffer position each delete was issued at.
    pub fn limit_for(&self, segment_gen: i64, entry_limit: i32) -> i32 {
        if segment_gen == self.del_gen() {
            entry_limit
        } else {
            MAX_DOC_ID_UPTO
        }
    }
}

/// `BufferedUpdatesStream`: the ordered set of frozen packets that have been
/// pushed but not yet applied, and the generation counter that stamps them.
///
/// Java's version additionally tracks in-flight resolution across threads
/// (`FinishedSegments`, `stillRunning`, `waitApply`); with one indexing
/// thread a packet is pushed and applied within the same call stack, so what
/// remains is the counter and the pending list.
#[derive(Debug, Default)]
pub struct BufferedUpdatesStream {
    updates: Vec<FrozenBufferedUpdates>,
    /// Java: "Starts at 1 so that SegmentInfos that have never had deletes
    /// applied (whose bufferedDelGen defaults to 0) will be correct."
    next_gen: i64,
}

impl BufferedUpdatesStream {
    pub fn new() -> Self {
        BufferedUpdatesStream {
            updates: Vec::new(),
            next_gen: 1,
        }
    }

    /// `BufferedUpdatesStream.push(FrozenBufferedUpdates)`: stamps the packet
    /// with the next generation and queues it. Returns that generation.
    // ARITH: `next_gen` starts at 1 and steps by exactly 1 per `push`/
    // `next_gen` call -- one per flushed delete packet or published segment,
    // each of which writes files. It is a session-local counter, never read
    // off disk (`bufferedDeletesGen` is not serialized), so 2^63 flushes in
    // one writer session is the bound and it is not reachable.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn push(&mut self, mut packet: FrozenBufferedUpdates) -> i64 {
        let gen = self.next_gen;
        self.next_gen += 1;
        packet.set_del_gen(gen);
        self.updates.push(packet);
        gen
    }

    /// `BufferedUpdatesStream.getNextGen()`: burn a generation without
    /// pushing a packet. Java calls this in `publishFlushedSegment` when a
    /// flush produced no segment-private deletes, so the new segment still
    /// gets a generation strictly above every packet pushed before it.
    // ARITH: see `push` -- same counter, same one-per-flush step.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn next_gen(&mut self) -> i64 {
        let gen = self.next_gen;
        self.next_gen += 1;
        gen
    }

    /// Every pending packet, oldest generation first. Applying in generation
    /// order is what makes "the last update wins" hold across packets.
    pub fn pending(&self) -> &[FrozenBufferedUpdates] {
        &self.updates
    }

    /// `BufferedUpdatesStream.any()`.
    pub fn any(&self) -> bool {
        !self.updates.is_empty()
    }

    /// Takes every pending packet, leaving the stream empty — Java's
    /// `FinishedSegments` bookkeeping collapses to this with one indexing
    /// thread, since a packet is fully applied by the time the call that
    /// pushed it returns.
    ///
    /// Moves rather than copies: a packet owns every delete term's bytes and
    /// every update value, and the caller is the only consumer.
    pub fn take_pending(&mut self) -> Vec<FrozenBufferedUpdates> {
        std::mem::take(&mut self.updates)
    }

    /// `BufferedUpdatesStream.clear()` — only used by `IndexWriter.rollback`.
    ///
    /// **The generation counter deliberately does not reset**, and this is the
    /// one place this port must diverge from Java's `clear()`, which does set
    /// `nextGen = 1`. Java can: `rollbackInternal` *closes* the writer, so no
    /// delete is ever issued again against the `SegmentCommitInfo`s the
    /// rollback restored. This port keeps the writer usable afterwards (see
    /// [`crate::index_writer::IndexWriter::rollback`]'s doc comment, which
    /// states that divergence), and those restored segments still carry the
    /// `buffered_deletes_gen` their original flush stamped on them — 1, 2, 3,
    /// … . Restarting the counter at 1 would make the next packet's
    /// `del_gen` lower than most of them, and
    /// [`FrozenBufferedUpdates::applies_to`] would then reject the delete for
    /// every segment above generation 1: a silent, partial delete.
    ///
    /// It is the same argument [`DeleteQueue::clear`] already makes for the
    /// sequence-number counter, and for the same reason: a generation, once
    /// handed out, must never be handed out again while anything that saw it
    /// is still reachable.
    pub fn clear(&mut self) {
        self.updates.clear();
    }
}

/// `DocumentsWriterDeleteQueue`: the sequence-number source, plus the two
/// [`BufferedUpdates`] Java's delete slices compute — see the module doc for
/// why the linked list itself is not ported.
#[derive(Debug)]
pub struct DeleteQueue {
    next_seq_no: SeqNo,
    /// Java's `globalBufferedUpdates`: what the *already written* segments
    /// must have applied to them. Entries carry [`MAX_DOC_ID_UPTO`] because
    /// every document in those segments predates the delete.
    global: BufferedUpdates,
    /// Java's `DocumentsWriterPerThread.pendingUpdates`: what the segment
    /// currently being buffered must have applied to it, each entry limited
    /// to the buffer position the delete was issued at.
    private: BufferedUpdates,
}

impl Default for DeleteQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl DeleteQueue {
    pub fn new() -> Self {
        DeleteQueue {
            next_seq_no: FIRST_SEQ_NO,
            global: BufferedUpdates::default(),
            private: BufferedUpdates::default(),
        }
    }

    /// `DocumentsWriterDeleteQueue.getNextSequenceNumber()`.
    // ARITH: `next_seq_no` starts at `FIRST_SEQ_NO` (1) and steps by 1 per
    // indexing operation in this session -- it is never read off disk, so the
    // bound is 2^63 operations in one writer session. `skip_sequence_numbers`
    // is the only other mutator and it clamps the counter at `MAX_SEQ_NO`
    // (`i64::MAX / 2`), so no jump, however absurd, can leave this `+ 1`
    // without 2^62 of headroom.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn next_sequence_number(&mut self) -> SeqNo {
        let seq_no = self.next_seq_no;
        self.next_seq_no += 1;
        seq_no
    }

    /// `DocumentsWriterDeleteQueue.getLastSequenceNumber()`.
    // ARITH: `next_seq_no >= FIRST_SEQ_NO` (1) always, per
    // `next_sequence_number`, so `- 1` cannot underflow.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn last_sequence_number(&self) -> SeqNo {
        self.next_seq_no - 1
    }

    /// `DocumentsWriterDeleteQueue.skipSequenceNumbers(long)`: Java inserts a
    /// gap at flush/commit so in-flight threads land inside it.
    ///
    /// **Not called by this port**: the gap exists so that operations already
    /// in flight on other threads get numbers inside it, and there are no
    /// other threads here. Kept for parity, and because it is what a
    /// multi-threaded `DocumentsWriterPerThreadPool` would need first.
    pub fn skip_sequence_numbers(&mut self, jump: i64) {
        // A *negative* jump would rewind the counter and hand the same
        // sequence number out twice -- two operations indistinguishable in the
        // ordering every `SeqNo` comparison in this module relies on -- so it
        // is clamped away rather than applied. The addition then saturates:
        // Java's `nextSeqNo.addAndGet(jump)` wraps to a negative sequence
        // number, which is strictly worse than a stuck-at-`MAX_SEQ_NO`
        // counter, and a wrap here is a bare `+` panic in a debug build. The
        // ceiling is what keeps `next_sequence_number`'s own `+ 1` provably
        // safe -- see `MAX_SEQ_NO`.
        self.next_seq_no = self.next_seq_no.saturating_add(jump.max(0)).min(MAX_SEQ_NO);
    }

    /// `addDelete(Term...)`: buffer term deletes and return the operation's
    /// sequence number. `doc_id_upto` is the number of documents currently
    /// buffered — the delete reaches those and not the ones after.
    pub fn add_term_deletes(&mut self, terms: &[Term], doc_id_upto: i32) -> SeqNo {
        for term in terms {
            self.global.add_term(term.clone(), MAX_DOC_ID_UPTO);
            self.private.add_term(term.clone(), doc_id_upto);
        }
        self.next_sequence_number()
    }

    /// `addDelete(Query...)`.
    pub fn add_query_deletes(&mut self, queries: &[DeleteQuery], doc_id_upto: i32) -> SeqNo {
        for query in queries {
            self.global.add_query(query.clone(), MAX_DOC_ID_UPTO);
            self.private.add_query(query.clone(), doc_id_upto);
        }
        self.next_sequence_number()
    }

    /// `addDocValuesUpdates(DocValuesUpdate...)`.
    pub fn add_doc_values_updates(
        &mut self,
        updates: &[DocValuesUpdate],
        doc_id_upto: i32,
    ) -> SeqNo {
        for update in updates {
            self.global.add_doc_values_update(update, MAX_DOC_ID_UPTO);
            self.private.add_doc_values_update(update, doc_id_upto);
        }
        self.next_sequence_number()
    }

    /// `DocumentsWriterDeleteQueue.anyChanges()`.
    pub fn any_changes(&self) -> bool {
        self.global.any() || self.private.any()
    }

    /// The private buffer, for the segment currently being built.
    pub fn private_updates(&self) -> &BufferedUpdates {
        &self.private
    }

    /// `DocumentsWriterDeleteQueue.freezeGlobalBuffer` + `BufferedUpdates.clear()`:
    /// takes everything buffered for the already-written segments and hands
    /// it over as an unstamped packet, leaving the global buffer empty.
    /// `None` when there is nothing to freeze (Java returns `null`).
    pub fn freeze_global_buffer(&mut self) -> Option<FrozenBufferedUpdates> {
        if !self.global.any() {
            return None;
        }
        let packet = FrozenBufferedUpdates::new(&self.global, None);
        self.global.clear();
        Some(packet)
    }

    /// `DocumentsWriterPerThread.flush`'s handling of `pendingUpdates`: freeze
    /// everything buffered against the segment being flushed as that segment's
    /// **private** packet, and reset the buffer for the next segment. `None`
    /// when nothing was buffered, matching Java's `segmentDeletes = null`
    /// branch.
    ///
    /// Resetting is not optional: every `docIDUpto` in this buffer is an index
    /// into the document buffer that the flush just emptied, so carrying an
    /// entry across would silently re-target it at the *next* segment's first
    /// documents.
    ///
    /// **One divergence from Java, deliberate.** Java resolves the private
    /// *term* deletes during the flush itself --
    /// `FreqProxTermsWriter.applyDeletes` walks the in-RAM `FreqProxFields`
    /// before the postings are written and folds the result straight into
    /// `SegmentWriteState.liveDocs` -- and then calls
    /// `pendingUpdates.clearDeleteTerms()` because they are already accounted
    /// for. This port resolves them from the *written* segment instead,
    /// through the same path every other segment's deletes take, so they stay
    /// in the packet. The outcome is identical (the `docIDUpto` limit is what
    /// bounds them either way); the cost is one extra `.liv` generation on a
    /// freshly flushed segment where Java writes none, recorded in
    /// `docs/sweep/m2/c7-delete-queue.md`.
    pub fn freeze_private_buffer(&mut self, segment_name: &str) -> Option<FrozenBufferedUpdates> {
        let packet = if self.private.any() {
            Some(FrozenBufferedUpdates::new(
                &self.private,
                Some(segment_name.to_string()),
            ))
        } else {
            None
        };
        self.private.clear();
        packet
    }

    /// `DocumentsWriterDeleteQueue.clear()` — `IndexWriter.rollback`'s
    /// `deleteQueue.clear()`. The sequence-number counter deliberately does
    /// **not** reset: Java's `rollbackInternal` builds a fresh queue whose
    /// seqNos continue past the aborted ones, so a caller can never see the
    /// same seqNo twice in one writer's life.
    pub fn clear(&mut self) {
        self.global.clear();
        self.private.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(field: &str, bytes: &str) -> Term {
        Term::new(field, bytes.as_bytes())
    }

    #[test]
    fn sequence_numbers_start_at_one_and_increase_by_one() {
        let mut q = DeleteQueue::new();
        assert_eq!(q.next_sequence_number(), 1);
        assert_eq!(q.next_sequence_number(), 2);
        assert_eq!(q.add_term_deletes(&[t("id", "a")], 0), 3);
        assert_eq!(q.add_query_deletes(&[DeleteQuery::MatchAll], 0), 4);
        assert_eq!(q.last_sequence_number(), 4);
    }

    #[test]
    fn skip_sequence_numbers_leaves_a_gap() {
        let mut q = DeleteQueue::new();
        assert_eq!(q.next_sequence_number(), 1);
        q.skip_sequence_numbers(10);
        assert_eq!(q.next_sequence_number(), 12);
    }

    /// A sequence number is only meaningful as an *order*, so a jump that
    /// rewinds the counter hands the same number out twice, and one that
    /// overflows it hands out a negative that sorts below every number already
    /// issued. Java's `nextSeqNo.addAndGet(jump)` does both silently; the bare
    /// `+=` here panicked on the second in a debug build.
    #[test]
    fn a_backwards_or_overflowing_jump_never_reissues_a_sequence_number() {
        let mut q = DeleteQueue::new();
        assert_eq!(q.next_sequence_number(), 1);
        q.skip_sequence_numbers(-100);
        assert_eq!(
            q.next_sequence_number(),
            2,
            "a negative jump must not rewind"
        );

        let mut q = DeleteQueue::new();
        q.skip_sequence_numbers(i64::MAX);
        q.skip_sequence_numbers(i64::MAX);
        assert_eq!(q.last_sequence_number(), MAX_SEQ_NO - 1);
        // ...and the counter is still far enough from the top that the very
        // next operation cannot overflow it.
        assert_eq!(q.next_sequence_number(), MAX_SEQ_NO);
        assert_eq!(q.next_sequence_number(), MAX_SEQ_NO + 1);
    }

    #[test]
    fn a_repeated_delete_term_keeps_the_highest_doc_id_upto() {
        // Java's `BufferedUpdates.addTerm`: the higher limit wins, so two
        // `updateDocument(t, ...)` calls in a row delete both of the earlier
        // documents rather than only the first.
        let mut b = BufferedUpdates::default();
        b.add_term(t("id", "a"), 3);
        b.add_term(t("id", "a"), 1);
        assert_eq!(b.delete_terms[&t("id", "a")], 3);
        b.add_term(t("id", "a"), 7);
        assert_eq!(b.delete_terms[&t("id", "a")], 7);
    }

    #[test]
    fn global_entries_are_unlimited_and_private_entries_are_bounded() {
        let mut q = DeleteQueue::new();
        q.add_term_deletes(&[t("id", "a")], 5);
        assert_eq!(q.private_updates().delete_terms[&t("id", "a")], 5);
        let global = q.freeze_global_buffer().expect("global packet");
        assert_eq!(global.delete_terms, vec![(t("id", "a"), MAX_DOC_ID_UPTO)]);
    }

    #[test]
    fn freeze_global_buffer_is_empty_the_second_time() {
        let mut q = DeleteQueue::new();
        q.add_term_deletes(&[t("id", "a")], 0);
        assert!(q.freeze_global_buffer().is_some());
        assert!(q.freeze_global_buffer().is_none());
    }

    #[test]
    fn a_private_packet_carries_the_buffers_limits_and_names_its_segment() {
        let mut q = DeleteQueue::new();
        q.add_term_deletes(&[t("id", "a")], 2);
        q.add_query_deletes(&[DeleteQuery::Term(t("id", "b"))], 3);
        let packet = q.freeze_private_buffer("_0").expect("private packet");
        assert_eq!(packet.delete_terms, vec![(t("id", "a"), 2)]);
        assert_eq!(packet.delete_queries.len(), 1);
        assert_eq!(packet.delete_queries[0].1, 3);
        assert_eq!(packet.private_segment.as_deref(), Some("_0"));
    }

    #[test]
    fn freezing_the_private_buffer_resets_it_for_the_next_segment() {
        // Every `docIDUpto` in the buffer indexes the document buffer the
        // flush just emptied; carrying one across would re-target it at the
        // next segment's first documents.
        let mut q = DeleteQueue::new();
        q.add_term_deletes(&[t("id", "a")], 2);
        assert!(q.freeze_private_buffer("_0").is_some());
        assert!(!q.private_updates().any());
        assert!(q.freeze_private_buffer("_1").is_none());
    }

    #[test]
    fn push_stamps_ascending_generations_starting_at_one() {
        let mut stream = BufferedUpdatesStream::new();
        let mut b = BufferedUpdates::default();
        b.add_term(t("id", "a"), MAX_DOC_ID_UPTO);
        let first = stream.push(FrozenBufferedUpdates::new(&b, None));
        let second = stream.push(FrozenBufferedUpdates::new(&b, None));
        assert_eq!((first, second), (1, 2));
        assert_eq!(stream.next_gen(), 3);
        assert_eq!(stream.pending().len(), 2);
    }

    #[test]
    fn a_packet_applies_to_older_segments_and_not_to_newer_ones() {
        let mut stream = BufferedUpdatesStream::new();
        let mut b = BufferedUpdates::default();
        b.add_term(t("id", "a"), MAX_DOC_ID_UPTO);
        stream.push(FrozenBufferedUpdates::new(&b, None));
        let packet = &stream.pending()[0];
        assert_eq!(packet.del_gen(), 1);
        // A segment flushed before this packet (generation 0) gets it...
        assert!(packet.applies_to(0));
        // ...and one flushed after (generation 2) does not.
        assert!(!packet.applies_to(2));
    }

    #[test]
    fn only_a_segments_own_private_packet_honours_the_doc_id_upto_limit() {
        let mut stream = BufferedUpdatesStream::new();
        let mut b = BufferedUpdates::default();
        b.add_query(DeleteQuery::MatchAll, 4);
        let gen = stream.push(FrozenBufferedUpdates::new(&b, Some("_0".into())));
        let packet = &stream.pending()[0];
        // Its own segment: the limit is enforced.
        assert_eq!(packet.limit_for(gen, 4), 4);
        // An older segment: no limit, every doc predates the delete.
        assert_eq!(packet.limit_for(gen - 1, 4), MAX_DOC_ID_UPTO);
    }

    #[test]
    fn doc_values_updates_record_value_reset_and_arrival_order() {
        let mut b = BufferedUpdates::default();
        b.add_doc_values_update(
            &DocValuesUpdate::Numeric {
                term: t("id", "a"),
                field: "n".into(),
                value: Some(7),
            },
            10,
        );
        b.add_doc_values_update(
            &DocValuesUpdate::Numeric {
                term: t("id", "a"),
                field: "n".into(),
                value: None,
            },
            11,
        );
        let buffer = &b.field_updates["n"];
        assert!(buffer.is_numeric);
        assert_eq!(buffer.updates[0].value, UpdateValue::Numeric(7));
        // A null-valued field is `reset(doc)`, not "set to 0".
        assert_eq!(buffer.updates[1].value, UpdateValue::None);
        assert_eq!(b.num_field_updates, 2);
    }

    #[test]
    fn binary_updates_are_kept_separate_from_numeric_ones() {
        let mut b = BufferedUpdates::default();
        b.add_doc_values_update(
            &DocValuesUpdate::Binary {
                term: t("id", "a"),
                field: "bin".into(),
                value: Some(b"hello".to_vec()),
            },
            0,
        );
        let buffer = &b.field_updates["bin"];
        assert!(!buffer.is_numeric);
        assert_eq!(
            buffer.updates[0].value,
            UpdateValue::Binary(b"hello".to_vec())
        );
    }

    #[test]
    fn clear_resets_every_bucket_but_not_the_sequence_number() {
        let mut q = DeleteQueue::new();
        q.add_term_deletes(&[t("id", "a")], 0);
        assert!(q.any_changes());
        q.clear();
        assert!(!q.any_changes());
        // Java's rollback builds a *new* queue that continues the seqNo
        // space; a caller must never see a seqNo twice.
        assert_eq!(q.next_sequence_number(), 2);
    }

    #[test]
    fn stream_clear_drops_the_packets_but_never_rewinds_the_generation() {
        // The bug this pins: `rollback()` restores committed segments that
        // still carry the `buffered_deletes_gen` their flush stamped on them.
        // Rewinding the counter to 1 (as Java's `clear()` does, safely, because
        // Java then closes the writer) would make every later packet sort
        // *below* those segments, and `applies_to` would reject the delete for
        // all but the oldest -- a silent, partial delete.
        let mut stream = BufferedUpdatesStream::new();
        let mut b = BufferedUpdates::default();
        b.add_term(t("id", "a"), MAX_DOC_ID_UPTO);
        assert_eq!(stream.push(FrozenBufferedUpdates::new(&b, None)), 1);
        assert_eq!(stream.push(FrozenBufferedUpdates::new(&b, None)), 2);
        assert!(stream.any());

        stream.clear();
        assert!(!stream.any());
        assert_eq!(
            stream.next_gen(),
            3,
            "the counter must continue past every generation already stamped \
             on a segment"
        );
    }

    #[test]
    fn take_pending_hands_over_the_packets_and_empties_the_stream() {
        let mut stream = BufferedUpdatesStream::new();
        let mut b = BufferedUpdates::default();
        b.add_term(t("id", "a"), MAX_DOC_ID_UPTO);
        stream.push(FrozenBufferedUpdates::new(&b, None));
        stream.push(FrozenBufferedUpdates::new(&b, None));
        let taken = stream.take_pending();
        assert_eq!(taken.len(), 2);
        assert_eq!(taken[0].del_gen(), 1);
        assert_eq!(taken[1].del_gen(), 2);
        assert!(!stream.any());
        // ...and the counter still moves forward.
        assert_eq!(stream.next_gen(), 3);
    }

    #[test]
    fn frozen_packet_sorts_its_terms() {
        let mut b = BufferedUpdates::default();
        b.add_term(t("id", "c"), 1);
        b.add_term(t("id", "a"), 2);
        b.add_term(t("body", "z"), 3);
        let packet = FrozenBufferedUpdates::new(&b, None);
        let terms: Vec<&Term> = packet.delete_terms.iter().map(|(t, _)| t).collect();
        assert_eq!(terms[0], &t("body", "z"));
        assert_eq!(terms[1], &t("id", "a"));
        assert_eq!(terms[2], &t("id", "c"));
    }

    #[test]
    fn an_empty_buffer_freezes_to_a_packet_that_reports_nothing() {
        let b = BufferedUpdates::default();
        let packet = FrozenBufferedUpdates::new(&b, None);
        assert!(!packet.any());
    }
}
