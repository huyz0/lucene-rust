//! Many indexing threads, one index -- Java's `DocumentsWriter`
//! (`DocumentsWriterPerThreadPool`, `DocumentsWriterFlushControl`,
//! `DocumentsWriterFlushQueue`) plus the executor half of
//! `ConcurrentMergeScheduler`, in the shape `PLAN.md` §3.5 point 6 sets:
//! ownership instead of `synchronized`.
//!
//! # Who owns what
//!
//! - **One buffer per indexing slot** ([`Dwpt`], a `DocumentsWriterPerThread`),
//!   each behind its own `Mutex`. A thread adding a document locks one free
//!   slot, and the delete log only long enough to copy pointers out of it --
//!   no writer-wide lock -- so indexing threads do not wait on each other's
//!   work (Java locks a DWPT per document
//!   the same way).
//! - **The control plane** -- the wrapped [`IndexWriter`]: segment list, file
//!   deleter, the buffered-deletes stream, merges' bookkeeping -- behind one
//!   `Mutex`, taken only to ticket or publish a segment, start or finish a
//!   merge, or commit. `PLAN.md` allows `Mutex` "only on control-plane
//!   state", and this is that state. Deletes do not take it (below).
//! - **Building a segment** -- inverting the documents, writing every file,
//!   resolving the buffer's own deletes -- happens on the thread whose slot
//!   filled up, **with no lock held** (`DocumentsWriterPerThread.flush` runs
//!   outside `IndexWriter`'s monitor too). Several threads build segments at
//!   once, each from its own buffer, through
//!   [`IndexingConfig::build_and_write_segment`], which reads only the shared,
//!   frozen [`IndexingConfig`].
//! - **Merging** likewise: [`ConcurrentIndexWriter::maybe_merge`] claims a
//!   merge under the control lock, reads and writes every file without it
//!   ([`IndexingConfig::run_merge`]), and takes the lock again only to publish
//!   -- in memory, carrying onto the merged segment any delete made meanwhile;
//!   the next commit makes it durable (`IndexWriter.commitMerge`). Whichever
//!   thread the caller dedicates to it is the merge scheduler's thread
//!   ([`ConcurrentIndexWriter::run_merges`]); nothing here spawns threads
//!   behind the caller's back, so a borrowed [`Directory`] needs no `'static`.
//!
//! # Deletes and their order
//!
//! A delete (or update, or doc-values update) must reach every document added
//! **before** it and none added after, in every buffer at once -- Java's
//! `DocumentsWriterDeleteQueue`, a lock-free list each DWPT reads a *slice* of.
//! Here: one append-only log behind its own lock, held only to append or to
//! copy out pointers, which also hands out the sequence numbers so they follow
//! the log's order. After each add, still holding its slot, a thread applies
//! the entries its slot has not seen to the slot's private deletes, limited to
//! the documents the slot held before the add --
//! `DeleteSlice.apply(pendingUpdates, docIDUpto)`. An update appends its
//! delete in that same step, after its document (`finishDocuments`), so the
//! two are never apart.
//!
//! When a buffer flushes, its segment takes a **ticket**: the control plane
//! freezes what the published segments owe right then, and the segment is
//! published strictly in ticket order, pushing that frozen packet immediately
//! before its own -- `DocumentsWriterFlushQueue` publishing
//! `publishFrozenUpdates` then `publishFlushedSegment`. That is what makes a
//! delete issued while a segment is being built reach it: the delete is frozen
//! by a *later* ticket, so its packet sorts above the segment and applies.
//!
//! # Commits
//!
//! A commit (`flushAllThreads`) takes every slot at once, tickets each
//! non-empty buffer, then takes a **cut**: a ticket carrying only what the
//! published segments owe as of now. With every slot held no update is half
//! done, so everything issued before the cut is in the commit and everything
//! after is not -- an update's delete and its document together. Segments
//! ticketed after the cut may build meanwhile but publish only once
//! `segments_N` is written, as Java holds back flushes during a full flush.
//!
//! # Not (yet) here
//!
//! Two-phase commit, and soft deletes of native documents, are single-writer
//! features for now -- [`ConcurrentIndexWriter::into_writer`] hands the
//! writer back for them. Explicit documents take soft updates here
//! ([`ConcurrentIndexWriter::soft_update_explicit_documents`]); a field they
//! use is registered through [`ConcurrentIndexWriter::register_field`], and
//! other settings change through [`ConcurrentIndexWriter::with_writer`].
//! (Vectors and custom-frequency postings ride in each slot's buffer with their
//! documents: [`ConcurrentIndexWriter::add_document_with_vectors`],
//! [`ConcurrentIndexWriter::add_document_with_custom_freq_terms`].) A merge
//! whose source takes a doc-values update while it runs is abandoned and
//! retried later rather than carried -- see [`IndexWriter::finish_merge`]. Each
//! ticket's frozen packet is applied to every published segment under the
//! control lock, where Java applies packets outside `IndexWriter`'s monitor,
//! per segment; that is the cost the update benchmark pays. The merge policy's
//! segment statistics are read from disk under the control lock too.

use std::collections::{HashMap, HashSet, VecDeque};
use std::ops::DerefMut;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard, Weak};
use std::time::Duration;

use lucene_codecs::stored_fields::Document;
use lucene_store::{Directory, FsIndexOutput, Input};

use crate::buffered_updates::{
    BufferedUpdates, DeleteQuery, DocValuesUpdate, FrozenBufferedUpdates, SeqNo, Term,
};
use crate::deletes;
use crate::flush_policy::{FlushByRamOrCountsPolicy, FlushControl, FlushPolicy};
use crate::index_writer::packed_fields::{PackedFields, StreamedStored};
use crate::index_writer::pending_postings::PendingPostings;
use crate::index_writer::{
    buffer_node, custom_freq_terms_ram_bytes, document_ram_bytes, DeleteNode, DocumentBuffer,
    DocumentVector, ExplicitDocs, ExplicitDocument, FlushDeletes, IndexWriter, IndexingConfig,
    InvertedTerms, Result, SegmentTicket, DISABLE_AUTO_FLUSH,
};
use crate::merge_policy::MergeTrigger;
use crate::merge_rate_limiter::{MergeRateLimiter, OneMergeProgress};
use crate::merge_scheduler::{MergeScheduler, MergeSource, PendingMerges, ScheduledMerge};
use crate::segment_infos::SegmentCommitInfo;
use crate::stall_control::DocumentsWriterStallControl;
use lucene_codecs::field_infos::FieldInfo;

mod live;
pub(crate) use live::DocValuesRun;
pub use live::{
    FieldTotals, KeywordRun, LeanSegment, LiveBuffer, LiveBufferField, LiveDocValue,
    LiveDocumentVectors, LiveFieldRequest, LivePattern, LiveRequest, LiveSnapshot, LiveView,
    PointBounds, ViewBuffer,
};

/// A value only reached through `&mut`: a slot's buffer is read by
/// several threads at once, while a part of it (a stored-fields stream)
/// is not `Sync`. The mutex is never locked, only passed through.
#[derive(Default)]
struct Exclusive<T>(Mutex<T>);

impl<T> Exclusive<T> {
    fn get_mut(&mut self) -> &mut T {
        self.0
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// One indexing slot's buffer -- `DocumentsWriterPerThread`, before its flush.
#[derive(Default)]
struct Dwpt {
    /// The buffered documents, but for the last `stored_docs`.
    docs: Vec<Document>,
    /// Documents after `docs` whose stored values went to `stored`, so they
    /// would be empty in `docs`: counted instead of held, as Lucene keeps no
    /// per-document object either. [`Dwpt::fill_docs`] makes them real
    /// empty documents where a [`Document`] list is needed.
    stored_docs: usize,
    /// Each document's custom-frequency terms
    /// ([`ConcurrentIndexWriter::add_document_with_custom_freq_terms`]),
    /// parallel to `docs` once the flush pads it: explicit documents carry
    /// none and add no entry, which saves an empty `Vec` per document.
    custom_freq_terms: Vec<Vec<(String, i32)>>,
    /// Each document's vectors
    /// ([`ConcurrentIndexWriter::add_document_with_vectors`]), short like
    /// `custom_freq_terms`.
    vectors: Vec<Vec<DocumentVector>>,
    /// Parallel to `docs` in a writer that takes explicit documents: each
    /// document's indexed fields, their terms moved into `postings`, packed.
    explicit: PackedFields,
    /// The explicit documents' stored values, compressed as they are added
    /// ([`IndexingConfig::streamed_stored_mode`]); `docs` then holds empty
    /// documents.
    stored: Exclusive<Option<StreamedStored>>,
    /// What `stored` counts toward `ram_bytes` now: its heap when last
    /// measured, so the charge falls as its bytes go to disk.
    stored_charged: usize,
    /// The explicit documents' postings, inverted as they are added.
    postings: PendingPostings,
    has_blocks: bool,
    ram_bytes: usize,
    /// This buffer's share of every delete issued since it last flushed, each
    /// limited to the documents it held at the time.
    private: BufferedUpdates,
    /// The absolute position in the delete log up to which `private` is
    /// current -- `DeleteSlice.sliceTail`.
    slice: usize,
    /// Norms and deletes for live search, kept only by a writer that has it
    /// ([`ConcurrentIndexWriter::enable_live_search`]).
    live: live::LiveColumns,
    /// The partition every buffered document was added under
    /// ([`IndexWriter::set_segment_partitioning`]); `None` while empty.
    partition: Option<Arc<str>>,
}

/// Pads `list` to `n` entries when any document has an entry, or when a
/// sort will permute it.
fn pad_lists<T>(list: &mut Vec<Vec<T>>, n: usize, sorting: bool) {
    if !list.is_empty() || sorting {
        list.resize_with(n, Vec::new);
    }
}

impl Dwpt {
    /// How many documents the buffer holds.
    fn num_docs(&self) -> usize {
        self.docs.len().saturating_add(self.stored_docs)
    }

    fn has_docs(&self) -> bool {
        self.num_docs() > 0
    }

    /// Turns the counted [`Dwpt::stored_docs`] into empty documents in
    /// `docs`, for a document added after them or a flush.
    fn fill_docs(&mut self) {
        let n = self.num_docs();
        self.docs.resize_with(n, Document::default);
        self.stored_docs = 0;
    }
}

/// Every delete issued, for the buffers to read their slices of and for the
/// published segments -- `DocumentsWriterDeleteQueue`, whose global buffer
/// likewise sits behind its own lock rather than `IndexWriter`'s. Entries
/// every buffer has applied and the control plane has taken are dropped from
/// the front.
///
/// The lock is held only to append an entry or to copy out the pointers a
/// reader has not seen; applying them happens after it is released. Java's
/// queue is a lock-free linked list its slices walk for the same reason:
/// every update appends and every add reads, so anything done under the
/// shared lock is paid by all indexing threads at once.
#[derive(Default)]
struct DeleteLog {
    /// The absolute position of `nodes[0]`.
    base: usize,
    nodes: VecDeque<Arc<DeleteNode>>,
    /// The absolute position up to which the control plane has buffered the
    /// log for the published segments ([`ConcurrentIndexWriter::hand_over`]).
    handed: usize,
    /// The next operation's sequence number, handed out under this lock so
    /// the numbers follow the log's order (`DocumentsWriterDeleteQueue`'s
    /// `nextSeqNo`, taken inside its `add` and `updateSlice`).
    next_seq: SeqNo,
    /// The RAM of the entries the control plane has not taken yet -- what
    /// the published segments' next packet will hold, the delete queue's
    /// `ramBytesUsed()` the flush policy weighs (`getDeleteBytesUsed`).
    unhanded_bytes: usize,
}

impl DeleteLog {
    fn end(&self) -> usize {
        self.base.saturating_add(self.nodes.len())
    }

    fn take_seq(&mut self) -> SeqNo {
        let seq = self.next_seq;
        self.next_seq = seq.saturating_add(1);
        seq
    }

    /// The entries from absolute position `from` on.
    fn since(&self, from: usize) -> Vec<Arc<DeleteNode>> {
        self.nodes
            .iter()
            .skip(from.max(self.base).saturating_sub(self.base))
            .cloned()
            .collect()
    }
}

/// The control plane: the wrapped writer, and the order segments publish in.
struct Core<'d> {
    writer: IndexWriter<'d>,
    /// The next ticket to hand out, and the ticket allowed to publish next.
    next_ticket: u64,
    next_publish: u64,
    /// While a commit is taking effect, the first ticket *after* its cut:
    /// that ticket and every later one wait to publish until `segments_N` is
    /// written -- Java's `DocumentsWriterFlushControl.blockedFlushes` during
    /// a full flush. They may build meanwhile.
    barrier: Option<u64>,
    /// Segments a merge has claimed and not yet finished.
    merging: HashSet<String>,
    /// The progress of every scheduled merge running now, through which
    /// [`ConcurrentIndexWriter::with_writer_merges_aborted`] aborts them.
    running: Vec<Arc<OneMergeProgress>>,
}

/// One document a thread adds, with what rides alongside it.
struct Entry {
    doc: Document,
    custom_freq_terms: Vec<(String, i32)>,
    vectors: Vec<DocumentVector>,
}

/// A buffer's per-document custom-frequency terms, vectors and explicit
/// fields, parallel to its documents, and its explicit documents' postings.
type Buffered = (
    Vec<Vec<(String, i32)>>,
    Vec<Vec<DocumentVector>>,
    PackedFields,
    Arc<PendingPostings>,
    Vec<crate::index_writer::BufferGraphField>,
);

/// Buffers an add's documents into a slot, returning the RAM they take.
type Fill = Box<dyn FnOnce(&mut Dwpt) -> Result<usize>>;

/// A buffer that filled (or was flushed): everything its segment is built from.
struct FlushBatch {
    ticket: SegmentTicket,
    ticket_no: u64,
    docs: Vec<Document>,
    custom_freq_terms: Vec<Vec<(String, i32)>>,
    vectors: Vec<Vec<DocumentVector>>,
    explicit: PackedFields,
    stored: Option<StreamedStored>,
    /// Shared with live search while the batch builds.
    postings: Arc<PendingPostings>,
    has_blocks: bool,
    private: BufferedUpdates,
    /// The slot's `ram_bytes` when it was taken: what this flush counts
    /// toward `flushBytes` until it is published.
    ram_bytes: usize,
    /// The partition the documents were added under.
    partition: Option<Arc<str>>,
    /// The vector fields' graphs live search kept over the buffer, which
    /// the flush writes instead of building them again.
    graphs: Vec<crate::index_writer::BufferGraphField>,
}

/// What [`ConcurrentIndexWriter::build`] hands to the publish.
struct Built {
    sci: SegmentCommitInfo,
    si_files: Vec<String>,
    fully_deleted: bool,
}

/// `IndexWriterConfig.getRAMPerThreadHardLimitMB()`'s default (1945 MB) in
/// bytes: a slot holding more is marked for flushing whatever the policy
/// says (`DocumentsWriterFlushControl.hardMaxBytesPerDWPT`).
pub(crate) const HARD_MAX_BYTES_PER_SLOT: usize = 1945 * 1024 * 1024;

/// A slot's advertised slice while it holds no document: it has nothing a
/// log entry could reach, so it holds back no trimming, and its next add
/// starts from the log's end.
const EMPTY_SLOT: usize = usize::MAX;

/// Hands each [`ConcurrentIndexWriter`] its `id`.
static NEXT_WRITER_ID: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    /// The writer and slot this thread last added to: see
    /// [`ConcurrentIndexWriter::acquire_slot`].
    static LAST_SLOT: std::cell::Cell<(usize, usize)> = const { std::cell::Cell::new((usize::MAX, 0)) };
}

/// See the module documentation.
pub struct ConcurrentIndexWriter<'d> {
    dir: &'d dyn Directory,
    /// The control plane's configuration as of its last change: a field
    /// registered ([`Self::register_field`]) or a setting changed through
    /// [`Self::with_writer`] replaces it. An add reads the snapshot current
    /// when it starts; a flush or merge the one current when it builds, which
    /// knows every field its documents use.
    cfg: RwLock<Arc<IndexingConfig>>,
    core: Mutex<Core<'d>>,
    /// Signalled when a segment publishes or the index changes.
    changed: Condvar,
    slots: Vec<RwLock<Dwpt>>,
    /// Where each slot's slice stands ([`EMPTY_SLOT`] while it is empty),
    /// readable without its lock, so the log can be trimmed.
    slices: Vec<AtomicUsize>,
    log: Mutex<DeleteLog>,
    /// Held for the whole of a [`Self::flush`] or [`Self::commit`]: one full
    /// flush at a time, as Java's `IndexWriter.fullFlushLock`.
    full_flush: Mutex<()>,
    /// Which slot to wait on when every slot is taken.
    next_slot: AtomicUsize,
    /// A partitioning writer: the slot each partition last started a buffer
    /// in, so a document finds its partition's buffer without a scan of
    /// every slot. A hint: the slot is checked under its lock.
    partition_slots: Mutex<HashMap<Arc<str>, usize>>,
    /// Tells this writer's slots apart from another writer's in a thread's
    /// [`LAST_SLOT`].
    id: usize,
    /// Tests: every add takes the next slot in rotation, so one thread can
    /// fill several buffers.
    #[cfg(any(test, feature = "test-support"))]
    rotate_slots: std::sync::atomic::AtomicBool,
    max_buffered_docs: Option<usize>,
    ram_buffer_bytes: Option<usize>,
    /// Set by [`Self::stop_merges`]: no merge is registered any more.
    merges_stopped: AtomicBool,
    /// How many [`Self::with_writer_merges_aborted`] calls are under way: no
    /// merge is registered or started by the caller meanwhile.
    merges_paused: AtomicUsize,
    /// `DocumentsWriterFlushControl`'s accounting -- RAM in the slots and in
    /// buffers being built, each slot's `flushPending` -- updated together
    /// with the flush policy's decision and the stall decision, as Java's is
    /// under its monitor. Kept only with a RAM buffer (see
    /// [`Self::update_ram`]).
    ram: Mutex<FlushControl>,
    /// `FlushByRamOrCountsPolicy`.
    policy: FlushByRamOrCountsPolicy,
    /// Each slot's `flushPending`, readable without the accounting lock --
    /// how an indexing thread finds a marked buffer to help flush.
    pending: Vec<AtomicBool>,
    /// `LiveIndexWriterConfig.checkPendingFlushOnUpdate`.
    check_pending_flush_on_update: AtomicBool,
    /// `DocumentsWriterFlushControl.stallControl`.
    stall: DocumentsWriterStallControl,
    /// `IndexWriter.pendingMerges`: merges registered for a scheduler and
    /// not yet taken by it. Their segments are in `Core::merging`.
    pending_merges: PendingMerges,
    /// `IndexWriterConfig.getMergeScheduler()`, when one was given
    /// ([`ConcurrentIndexWriter::with_merge_scheduler`]).
    scheduler: Option<SchedulerHook>,
    /// `IndexWriter.pendingNumDocs`, in two parts: documents in the slots or
    /// in a flush not yet published, and every segment's `maxDoc` as of the
    /// last change to the segment lists. A published flush's documents are
    /// briefly in both, which can only over-count.
    buffered_docs: AtomicUsize,
    segment_docs: AtomicUsize,
    /// [`IndexWriter::set_max_docs`]' limit, frozen with the configuration.
    max_docs: usize,
    /// [`Self::enable_live_search`].
    live: AtomicBool,
    /// Buffers being flushed or published, readable by live search, under
    /// a lock of their own: a view opens without waiting for a commit or a
    /// publish that holds the control lock. Taken after the control lock
    /// when both are held.
    live_flushes: Mutex<live::LiveFlushes>,
    /// The next buffer's id for live search, from 1.
    next_buffer_id: AtomicU64,
}

/// The scheduler this writer hands its pending merges to, and the writer
/// itself as the `MergeSource` the scheduler's threads call back into.
struct SchedulerHook {
    scheduler: Arc<dyn MergeScheduler>,
    source: Weak<dyn MergeSource>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    // A thread that panicked mid-operation leaves the state as it was at the
    // panic; later calls see it rather than a poison error. The writer's own
    // invariants are held by the operations themselves, not by the lock.
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// A slot's buffer to change: [`lock`]'s rules.
fn write_slot(m: &RwLock<Dwpt>) -> RwLockWriteGuard<'_, Dwpt> {
    m.write().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// A slot's buffer to read beside other readers -- a view opening, a
/// chunk being written -- while indexing into it waits.
fn read_slot(m: &RwLock<Dwpt>) -> RwLockReadGuard<'_, Dwpt> {
    m.read().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Lifts a commit's publish barrier however the commit ends, unwinding
/// included -- a barrier left up would stop every later segment for good.
struct BarrierGuard<'a, 'd>(&'a ConcurrentIndexWriter<'d>);

impl Drop for BarrierGuard<'_, '_> {
    fn drop(&mut self) {
        lock(&self.0.core).barrier = None;
        self.0.changed.notify_all();
    }
}

impl<'d> ConcurrentIndexWriter<'d> {
    /// The current configuration snapshot (see the `cfg` field).
    fn cfg(&self) -> Arc<IndexingConfig> {
        Arc::clone(
            &self
                .cfg
                .read()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        )
    }

    /// Runs `body` on the wrapped writer under the control lock -- for what
    /// only the single-threaded writer does (commit user data, commit points
    /// and holds, soft-deletes retention, force merges). The buffers are not
    /// flushed first; the configuration snapshot is refreshed after.
    pub fn with_writer<T>(&self, body: impl FnOnce(&mut IndexWriter<'d>) -> T) -> T {
        self.run_on_writer(lock(&self.core), body)
    }

    /// [`Self::with_writer`] once no merge is running: merges not started yet
    /// are dropped and running scheduled ones aborted, and `body` runs after
    /// every merge has let go of its segments. For a force merge, which
    /// merges those segments itself: a throttled natural merge of them would
    /// only keep their files on disk until it finished, to be merged again
    /// after. Merges a caller runs itself ([`Self::maybe_merge`]) cannot be
    /// aborted and are waited for. The policy registers merges again on the
    /// next flush or merge request.
    pub fn with_writer_merges_aborted<T>(&self, body: impl FnOnce(&mut IndexWriter<'d>) -> T) -> T {
        struct Paused<'a>(&'a AtomicUsize);
        impl Drop for Paused<'_> {
            fn drop(&mut self) {
                self.0.fetch_sub(1, Ordering::AcqRel);
            }
        }
        self.merges_paused.fetch_add(1, Ordering::AcqRel);
        let _paused = Paused(&self.merges_paused);
        let mut core = lock(&self.core);
        self.abort_merges(&mut core);
        let core = self.wait_for_merges(core);
        self.run_on_writer(core, body)
    }

    /// Whether no merge may be registered now.
    fn merges_blocked(&self) -> bool {
        self.merges_stopped.load(Ordering::Acquire)
            || self.merges_paused.load(Ordering::Acquire) > 0
    }

    /// `IndexWriter.abortMerges`' first half: drops the registered merges no
    /// merge thread has taken yet, releasing their segments, and aborts the
    /// running scheduled ones. [`Self::wait_for_merges`] waits for them.
    fn abort_merges(&self, core: &mut Core<'_>) {
        while let Some(merge) = self.pending_merges.pop() {
            for name in &merge.segments {
                core.merging.remove(name);
            }
        }
        for progress in &core.running {
            progress.abort();
        }
    }

    /// Waits until no merge holds a segment, aborting each scheduled merge
    /// that starts meanwhile -- one a merge thread took before
    /// [`Self::abort_merges`] ran.
    fn wait_for_merges<'g>(
        &'g self,
        mut core: MutexGuard<'g, Core<'d>>,
    ) -> MutexGuard<'g, Core<'d>> {
        while !core.merging.is_empty() {
            core = self
                .changed
                .wait(core)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            for progress in &core.running {
                progress.abort();
            }
        }
        core
    }

    fn run_on_writer<T>(
        &self,
        mut core: MutexGuard<'_, Core<'d>>,
        body: impl FnOnce(&mut IndexWriter<'d>) -> T,
    ) -> T {
        let out = body(&mut core.writer);
        *self
            .cfg
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = core.writer.shared_config();
        drop(core);
        self.changed.notify_all();
        out
    }

    /// How many fields are registered: the schema only grows, so a changed
    /// count means [`Self::field_infos`] changed.
    pub fn field_count(&self) -> usize {
        self.cfg().field_list().len()
    }

    /// [`IndexWriter::fields`], from any thread: the fields registered so far.
    pub fn field_infos(&self) -> Vec<FieldInfo> {
        self.cfg().field_list().to_vec()
    }

    /// [`IndexWriter::register_field`], from any thread: the field is known
    /// to every add that starts after this returns.
    pub fn register_field(&self, info: FieldInfo) -> Result<i32> {
        self.with_writer(|w| w.register_field(info))
    }

    /// Shares `writer` among up to `slots` indexing threads at once. Its
    /// configuration is frozen from here on -- the building threads all read
    /// it -- and merging passes to [`Self::maybe_merge`], so that merges run
    /// on the caller's merge thread rather than inside a commit. Anything it
    /// had buffered is flushed first.
    ///
    /// # Errors
    /// [`crate::index_writer::Error::ConcurrentUnsupported`] for a writer
    /// with a parent field that takes native documents: the building threads
    /// have no parent marker to give them.
    pub fn new(mut writer: IndexWriter<'d>, slots: usize) -> Result<Self> {
        if let Some(what) = writer.shared_config().concurrent_unsupported() {
            return Err(crate::index_writer::Error::ConcurrentUnsupported(what));
        }
        // The configuration is frozen from here: the parent field's number,
        // which each add marks its last document with, is fixed now.
        writer.register_parent_field()?;
        writer.validate_configured_term_vector_schema()?;
        writer.flush()?;
        writer.set_merges_by_caller(true);
        let slots = slots.max(1);
        let max_buffered_docs = usize::try_from(writer.max_buffered_docs())
            .ok()
            .filter(|_| writer.max_buffered_docs() != DISABLE_AUTO_FLUSH);
        let ram_buffer_bytes = (writer.ram_buffer_size_mb() > 0.0)
            .then(|| (writer.ram_buffer_size_mb() * 1024.0 * 1024.0) as usize);
        let log = DeleteLog {
            next_seq: writer.next_sequence_number_peek(),
            ..DeleteLog::default()
        };
        let segment_docs = writer.segment_doc_total()?;
        let mut live = live::LiveFlushes::default();
        live.committed(writer.segment_infos().version, 0);
        let max_docs = writer.max_docs_limit();
        Ok(ConcurrentIndexWriter {
            dir: writer.dir(),
            cfg: RwLock::new(writer.shared_config()),
            core: Mutex::new(Core {
                writer,
                next_ticket: 0,
                next_publish: 0,
                barrier: None,
                merging: HashSet::new(),
                running: Vec::new(),
            }),
            live_flushes: Mutex::new(live),
            changed: Condvar::new(),
            slots: (0..slots).map(|_| RwLock::new(Dwpt::default())).collect(),
            id: NEXT_WRITER_ID.fetch_add(1, Ordering::Relaxed),
            slices: (0..slots).map(|_| AtomicUsize::new(EMPTY_SLOT)).collect(),
            log: Mutex::new(log),
            full_flush: Mutex::new(()),
            next_slot: AtomicUsize::new(0),
            partition_slots: Mutex::new(HashMap::new()),
            #[cfg(any(test, feature = "test-support"))]
            rotate_slots: std::sync::atomic::AtomicBool::new(false),
            max_buffered_docs,
            ram_buffer_bytes,
            merges_stopped: AtomicBool::new(false),
            merges_paused: AtomicUsize::new(0),
            ram: Mutex::new(FlushControl::new(slots)),
            policy: FlushByRamOrCountsPolicy {
                max_buffered_docs,
                ram_buffer_bytes,
            },
            pending: (0..slots).map(|_| AtomicBool::new(false)).collect(),
            check_pending_flush_on_update: AtomicBool::new(true),
            stall: DocumentsWriterStallControl::new(),
            pending_merges: PendingMerges::new(),
            scheduler: None,
            buffered_docs: AtomicUsize::new(0),
            segment_docs: AtomicUsize::new(segment_docs),
            max_docs,
            live: AtomicBool::new(false),
            next_buffer_id: AtomicU64::new(1),
        })
    }

    /// `IndexWriter.updatePendingMerges`: registers every merge the policy
    /// proposes among segments not already merging -- claiming their
    /// segments -- for the scheduler to take. Nothing without a policy.
    fn register_merges(&self, core: &mut Core<'_>, trigger: MergeTrigger) -> Result<()> {
        if self.merges_blocked() {
            return Ok(());
        }
        for ((names, hooks), estimated) in
            core.writer.find_scheduled_merges(trigger, &core.merging)?
        {
            core.merging.extend(names.iter().cloned());
            self.pending_merges.push(Arc::new(
                ScheduledMerge::new(names, estimated).with_hooks(hooks),
            ));
        }
        Ok(())
    }

    /// `IndexWriter.maybeMerge(trigger)`: registers what the policy wants and
    /// hands it to the scheduler. A no-op for a writer without one, whose
    /// caller runs merges itself ([`Self::maybe_merge`], [`Self::run_merges`]).
    fn schedule_merges(&self, trigger: MergeTrigger) -> Result<()> {
        let Some(hook) = &self.scheduler else {
            return Ok(());
        };
        let Some(source) = hook.source.upgrade() else {
            return Ok(());
        };
        self.register_merges(&mut lock(&self.core), trigger)?;
        hook.scheduler.merge(&source, trigger)
    }

    /// Runs one scheduled merge on the calling (merge) thread, every output
    /// through `limiter` -- `IndexWriter.merge(OneMerge)`: `mergeInit`, the
    /// merge without the control lock, `commitMerge`, `mergeFinish`, then
    /// `updatePendingMerges(MERGE_FINISHED)` so a cascade of merges is
    /// registered for the scheduler's next round.
    fn run_scheduled_merge(
        &self,
        merge: &ScheduledMerge,
        limiter: Option<Arc<MergeRateLimiter>>,
    ) -> Result<()> {
        let release = |core: &mut Core<'_>| {
            for name in &merge.segments {
                core.merging.remove(name);
            }
            core.running.retain(|p| !Arc::ptr_eq(p, merge.progress()));
        };
        let (plan, listened) = {
            let mut core = lock(&self.core);
            match core.writer.begin_merge(&merge.segments) {
                Ok(mut plan) => {
                    core.writer
                        .attach_merge_hooks(&mut plan, merge.hooks.clone());
                    let listened = core.writer.merge_event(&plan);
                    core.running.push(Arc::clone(merge.progress()));
                    // A pending abort waits for merges to start, too.
                    drop(core);
                    self.changed.notify_all();
                    (plan, listened)
                }
                Err(e) => {
                    release(&mut core);
                    drop(core);
                    // `delete_all` waits for `merging` to empty.
                    self.changed.notify_all();
                    return Err(e);
                }
            }
        };
        // `OpenSearchConcurrentMergeScheduler.doMerge`: the listener hears of
        // the merge before it runs and after it ends, without the lock.
        let report =
            crate::index_writer::MergeReport::start(listened, Some(Arc::clone(merge.progress())));
        let tracking = TrackingDirectory::new(self.dir);
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            self.cfg()
                .run_merge_limited(&tracking, &plan, limiter, merge.progress())
        }));
        let mut core = lock(&self.core);
        release(&mut core);
        let result = match outcome {
            Ok(Ok(outcome)) => {
                let finished = core.writer.finish_merge(plan, outcome);
                self.refresh_segment_docs(&mut core);
                finished.and_then(|_| self.register_merges(&mut core, MergeTrigger::MergeFinished))
            }
            Ok(Err(e)) => {
                let _ = core.writer.abort_merge(plan);
                let _ = core.writer.delete_new_files(&tracking.created());
                Err(e)
            }
            Err(panic) => {
                let _ = core.writer.abort_merge(plan);
                let _ = core.writer.delete_new_files(&tracking.created());
                drop(core);
                self.changed.notify_all();
                drop(report);
                resume_unwind(panic);
            }
        };
        drop(core);
        self.changed.notify_all();
        report.finish(&result);
        result
    }

    /// Registers no merge from now on and drops the registered ones no merge
    /// thread has taken yet -- what `IndexWriter.rollback` does to merges
    /// (`stopMerges`, `abortMerges`) before the writer closes. Merges already
    /// running finish; a merge scheduler's `close` waits for them.
    pub fn stop_merges(&self) {
        self.merges_stopped.store(true, Ordering::Release);
        let mut core = lock(&self.core);
        while let Some(merge) = self.pending_merges.pop() {
            for name in &merge.segments {
                core.merging.remove(name);
            }
        }
        drop(core);
        self.changed.notify_all();
    }

    /// `IndexWriter.close()`'s merge half: registers what the policy still
    /// wants, lets the scheduler run it with throttling lifted
    /// (`MergeTrigger.CLOSING`), and waits for every merge the scheduler
    /// started -- returning the first merge failure. A no-op without a
    /// scheduler.
    pub fn close_merges(&self) -> Result<()> {
        let Some(hook) = &self.scheduler else {
            return Ok(());
        };
        self.schedule_merges(MergeTrigger::Closing)?;
        hook.scheduler.close()
    }

    /// `DocumentsWriterFlushControl.updateStallState`, after `change` has
    /// updated the accounting: indexing stalls while the RAM in slots and in
    /// flushes together exceeds twice the RAM buffer (`stallLimitBytes`) --
    /// flushing is falling behind -- but not when the slots alone do, since
    /// then no flush in progress could free enough to lift it.
    ///
    /// The decision and the flag change together under one lock, as Java's
    /// monitor makes them: two threads deciding from different snapshots
    /// could otherwise leave the flag stalled after the condition cleared,
    /// with every indexing thread waiting on it. With no RAM buffer
    /// (`stallLimitBytes` is `Long.MAX_VALUE`) indexing can never stall, so
    /// nothing is counted and no lock is taken.
    fn update_ram<T: Default>(&self, change: impl FnOnce(&mut FlushControl) -> T) -> T {
        let Some(buffer) = self.ram_buffer_bytes else {
            return T::default();
        };
        let mut ram = lock(&self.ram);
        let out = change(&mut ram);
        for i in ram.take_pending_changes() {
            if let Some(flag) = self.pending.get(i) {
                flag.store(ram.is_flush_pending(i), Ordering::Release);
            }
        }
        let limit = buffer.saturating_mul(2);
        let (active, flushing) = (ram.active_bytes(), ram.flush_bytes());
        let stall = active.saturating_add(flushing) > limit && active < limit;
        self.stall.update_stalled(stall);
        out
    }

    /// `LiveIndexWriterConfig.setCheckPendingFlushUpdate`: whether indexing
    /// threads flush buffers the flush policy marked in other slots (on by
    /// default). Off, a marked buffer is flushed by the next thread to index
    /// into it, by [`Self::flush_next_buffer`], or by a full flush -- and by
    /// indexing threads anyway while indexing is stalled.
    pub fn set_check_pending_flush_on_update(&self, check: bool) {
        self.check_pending_flush_on_update
            .store(check, Ordering::Relaxed);
    }

    /// `IndexWriter.hasPendingMerges`: whether merges are registered that no
    /// merge thread has taken yet. Takes only the queue's own lock.
    pub fn has_pending_merges(&self) -> bool {
        !self.pending_merges.is_empty()
    }

    /// See [`Self::set_check_pending_flush_on_update`].
    pub fn check_pending_flush_on_update(&self) -> bool {
        self.check_pending_flush_on_update.load(Ordering::Relaxed)
    }

    /// `DocumentsWriterFlushControl.nextPendingFlush` + `doFlush`: takes one
    /// buffer the flush policy marked, whose slot is free, and flushes it on
    /// this thread. Returns whether it flushed one.
    fn flush_one_pending(&self) -> Result<bool> {
        for (j, flag) in self.pending.iter().enumerate() {
            if !flag.load(Ordering::Acquire) {
                continue;
            }
            let Ok(mut dwpt) = self.slots[j].try_write() else {
                continue;
            };
            if !self.pending[j].load(Ordering::Acquire) {
                continue;
            }
            let batch = self.begin_flush(j, &mut dwpt);
            drop(dwpt);
            if let Some(batch) = batch {
                self.complete_flush(batch)?;
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Whether any slot is marked for flushing (`numQueuedFlushes() > 0`).
    fn any_pending(&self) -> bool {
        self.pending.iter().any(|p| p.load(Ordering::Acquire))
    }

    /// `DocumentsWriter.applyAllDeletes`: once the flush policy found the
    /// buffered deletes over the RAM buffer, freezes them into a packet of
    /// their own and applies it to the published segments -- a ticket that
    /// publishes in order with the segments before it, as Java's
    /// `ticketQueue.addTicket(maybeFreezeGlobalBuffer)`. Never during a full
    /// flush (the flag waits for the next chance), as Java's.
    fn maybe_apply_all_deletes(&self) -> Result<bool> {
        let Ok(_full_flush) = self.full_flush.try_lock() else {
            return Ok(false);
        };
        if !self.update_ram(FlushControl::get_and_reset_apply_all_deletes) {
            return Ok(false);
        }
        let (ticket_no, packet) = {
            let mut core = lock(&self.core);
            let mut live = self.live_flushes();
            let nodes = Self::take_unhanded(&mut lock(&self.log));
            Self::hand_over(&mut core, &mut live, &nodes);
            drop(live);
            let packet = core.writer.begin_deletes_ticket();
            let ticket_no = core.next_ticket;
            core.next_ticket = ticket_no.saturating_add(1);
            (ticket_no, packet)
        };
        self.update_ram(|c| c.set_delete_bytes_used(0));
        self.complete_cut(ticket_no, packet)?;
        Ok(true)
    }

    /// `DocumentsWriter.preUpdate`: while indexing is stalled, or while a
    /// marked buffer waits and `checkPendingFlushOnUpdate` is on, helps
    /// flush marked buffers, then waits while indexing is still stalled.
    fn pre_update(&self) -> Result<bool> {
        let mut flushed = false;
        while self.stall.any_stalled_threads()
            || (self.check_pending_flush_on_update.load(Ordering::Relaxed) && self.any_pending())
        {
            let helped = self.flush_one_pending()?;
            flushed |= helped;
            if !helped && !self.stall.any_stalled_threads() {
                // Every marked buffer is being indexed into or flushed by
                // another thread, which will flush it.
                break;
            }
            self.stall.wait_if_stalled();
        }
        Ok(flushed)
    }

    /// The stall control, for tests (`DocumentsWriterFlushControl`'s
    /// package-private `stallControl`).
    #[cfg(test)]
    fn stall_control(&self) -> &DocumentsWriterStallControl {
        &self.stall
    }

    /// `IndexWriter.addDocument`, from any thread.
    pub fn add_document(&self, doc: Document) -> Result<SeqNo> {
        self.add(None, vec![doc])
    }

    /// [`IndexWriter::add_document_with_vectors`], from any thread: the
    /// document and its KNN vectors go into this thread's buffer together,
    /// and its segment writes them (`IndexingChain` per
    /// `DocumentsWriterPerThread`).
    pub fn add_document_with_vectors(
        &self,
        doc: Document,
        vectors: Vec<DocumentVector>,
    ) -> Result<SeqNo> {
        self.add_entries(
            None,
            vec![Entry {
                doc,
                custom_freq_terms: Vec::new(),
                vectors,
            }],
        )
    }

    /// [`IndexWriter::add_document_with_custom_freq_terms`], from any
    /// thread.
    pub fn add_document_with_custom_freq_terms(
        &self,
        doc: Document,
        terms: Vec<(String, i32)>,
    ) -> Result<SeqNo> {
        crate::index_writer::summed_custom_freq_terms(&terms)?;
        self.add_entries(
            None,
            vec![Entry {
                doc,
                custom_freq_terms: terms,
                vectors: Vec::new(),
            }],
        )
    }

    /// `IndexWriter.addDocuments`: a block, kept contiguous in one segment.
    pub fn add_documents(&self, docs: Vec<Document>) -> Result<SeqNo> {
        self.add(None, docs)
    }

    /// `IndexWriter.updateDocument`: deletes every document matching `term`
    /// added before this call, then adds `doc`.
    pub fn update_document(&self, term: Term, doc: Document) -> Result<SeqNo> {
        self.add(Some(DeleteNode::terms(vec![term])), vec![doc])
    }

    /// `IndexWriter.deleteDocuments(Term...)`.
    pub fn delete_documents_by_term(&self, terms: &[Term]) -> Result<SeqNo> {
        self.buffer_delete(DeleteNode::terms(terms.to_vec()))
    }

    /// `IndexWriter.deleteDocuments(Query...)`, for the query shapes
    /// [`DeleteQuery`] covers. `MatchAll` anywhere is [`Self::delete_all`]
    /// (LUCENE-6379), as in [`IndexWriter::delete_documents_by_query`].
    pub fn delete_documents_by_query(&self, queries: &[DeleteQuery]) -> Result<SeqNo> {
        if queries.iter().any(|q| matches!(q, DeleteQuery::MatchAll)) {
            return self.delete_all();
        }
        self.buffer_delete(DeleteNode::Queries(queries.to_vec()))
    }

    /// `IndexWriter.deleteAll()`: drops every buffered document and every
    /// segment, as [`IndexWriter::delete_all`] -- not durable until the next
    /// [`Self::commit`]. Under the full-flush lock with every slot held
    /// (`DocumentsWriter.lockAndAbortAll`): the slots' buffers and their
    /// deletes are discarded, every flush already ticketed publishes first
    /// (`waitForFlush`), merges not yet started are dropped and running ones
    /// aborted and waited for (`abortMerges`), and every delete
    /// buffered so far is cleared. Returns the operation's sequence number.
    pub fn delete_all(&self) -> Result<SeqNo> {
        let _full_flush = lock(&self.full_flush);
        let mut slots: Vec<RwLockWriteGuard<'_, Dwpt>> =
            self.slots.iter().map(write_slot).collect();
        self.abort_slots(&mut slots);
        let mut core = lock(&self.core);
        self.abort_merges(&mut core);
        while core.next_publish != core.next_ticket || !core.merging.is_empty() {
            core = self
                .changed
                .wait(core)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            for progress in &core.running {
                progress.abort();
            }
        }
        let (seq_no, end) = {
            let mut log = lock(&self.log);
            let end = log.end();
            log.nodes.clear();
            log.base = end;
            log.handed = end;
            log.unhanded_bytes = 0;
            (log.take_seq(), end)
        };
        self.update_ram(|c| c.set_delete_bytes_used(0));
        let unpinned = {
            let mut live = self.live_flushes();
            live.discard(end);
            live.take_unpinned()
        };
        let deleted = core.writer.delete_all();
        self.unpin_with(&mut core.writer, unpinned);
        // Every slot was emptied above and no flush is in flight.
        self.buffered_docs.store(0, Ordering::Release);
        self.refresh_segment_docs(&mut core);
        deleted?;
        drop(core);
        drop(slots);
        self.changed.notify_all();
        Ok(seq_no)
    }

    /// `IndexWriter.updateDocValues(Term, Field...)`.
    pub fn update_doc_values(&self, term: Term, updates: &[DocValuesUpdate]) -> Result<SeqNo> {
        let node = self.cfg().doc_values_update_node(&term, updates)?;
        self.buffer_delete(node)
    }

    /// `IndexWriter.updateNumericDocValue(Term, String, long)`.
    pub fn update_numeric_doc_value(&self, term: Term, field: &str, value: i64) -> Result<SeqNo> {
        let update = DocValuesUpdate::Numeric {
            term: term.clone(),
            field: field.to_string(),
            value: Some(value),
        };
        self.update_doc_values(term, std::slice::from_ref(&update))
    }

    /// Appends `node` to the log -- `DocumentsWriterDeleteQueue.add`, which
    /// likewise takes no `IndexWriter` lock: a publish holds the control lock
    /// while it applies the published segments' share of the deletes, and a
    /// delete must not queue behind it. The sequence number is taken in the
    /// same step, so numbers follow the log's order.
    fn buffer_delete(&self, node: DeleteNode) -> Result<SeqNo> {
        let bytes = node.ram_bytes();
        let (seq_no, delete_bytes) = {
            let mut log = lock(&self.log);
            log.nodes.push_back(Arc::new(node));
            log.unhanded_bytes = log.unhanded_bytes.saturating_add(bytes);
            let seq_no = log.take_seq();
            self.trim(&mut log);
            (seq_no, log.unhanded_bytes)
        };
        // `DocumentsWriterFlushControl.doOnDelete`.
        self.update_ram(|c| {
            c.set_delete_bytes_used(delete_bytes);
            self.policy.on_change(c, None);
        });
        self.maybe_apply_all_deletes()?;
        Ok(seq_no)
    }

    /// Drops the entries every slot has applied and the control plane has
    /// taken.
    fn trim(&self, log: &mut DeleteLog) {
        if log.nodes.is_empty() {
            return;
        }
        let applied = self
            .slices
            .iter()
            .map(|s| s.load(Ordering::Acquire))
            .min()
            .unwrap_or(EMPTY_SLOT)
            .min(log.handed);
        while log.base < applied && !log.nodes.is_empty() {
            log.nodes.pop_front();
            log.base = log.base.saturating_add(1);
        }
    }

    /// The log entries the control plane has not taken yet, now marked taken
    /// -- the global slice `DocumentsWriterDeleteQueue.freezeGlobalBuffer`
    /// reads. The caller holds the control lock and buffers them
    /// ([`Self::hand_over`]) before anything freezes the writer's global
    /// buffer.
    fn take_unhanded(log: &mut DeleteLog) -> Vec<Arc<DeleteNode>> {
        let nodes = log.since(log.handed);
        log.handed = log.end();
        log.unhanded_bytes = 0;
        nodes
    }

    /// Buffers `nodes` for the published segments.
    /// `live` was locked before the log they were taken from, so a view
    /// sees them either in the log or in the buffers being flushed.
    /// The live segment list as an NRT reader opens it. With `handed` (live
    /// search on, taken at the cut of a flush whose barrier is still up),
    /// its version is also a reader live views can be opened against, as a
    /// commit's is: it holds every publish so far and every delete before
    /// log position `handed`.
    fn nrt_snapshot_keyed(&self, handed: Option<usize>) -> Result<crate::nrt::NrtSnapshot> {
        let mut core = lock(&self.core);
        let snapshot = core.writer.nrt_snapshot_of_live_view()?;
        if let Some(handed) = handed {
            let unpinned = {
                let mut live = self.live_flushes();
                live.committed(snapshot.segment_infos.version, handed);
                live.take_unpinned()
            };
            self.unpin_with(&mut core.writer, unpinned);
        }
        Ok(snapshot)
    }

    fn hand_over(core: &mut Core<'_>, live: &mut live::LiveFlushes, nodes: &[Arc<DeleteNode>]) {
        for node in nodes {
            core.writer.buffer_global_delete(node);
        }
        live.handed(nodes);
        // They reach the buffers being flushed once published; live search
        // sees them there now.
        if !nodes.is_empty() && live.buffers_mut().next().is_some() {
            let cfg = core.writer.shared_config();
            for f in live.buffers_mut() {
                f.columns.apply(nodes, f.max_doc, &f.postings, &cfg);
            }
        }
    }

    /// Applies `nodes` -- the log from `dwpt.slice` up to `end` -- to slot
    /// `i`'s private deletes, each reaching the slot's first `limit`
    /// documents -- `DeleteSlice.apply(pendingUpdates, docIDUpto)`.
    fn apply_slice(
        &self,
        i: usize,
        dwpt: &mut Dwpt,
        nodes: &[Arc<DeleteNode>],
        end: usize,
        limit: usize,
    ) {
        let limit = i32::try_from(limit).unwrap_or(i32::MAX);
        for node in nodes {
            buffer_node(&mut dwpt.private, node, limit);
        }
        if self.live() && !nodes.is_empty() {
            dwpt.live.apply(nodes, limit, &dwpt.postings, &self.cfg());
        }
        dwpt.slice = end;
        self.slices[i].store(end, Ordering::Release);
    }

    /// Tests: every add from now on takes the next slot in rotation (or stops doing so), so the
    /// buffers that fill do not depend on how the threads that add happen to be scheduled.
    #[cfg(any(test, feature = "test-support"))]
    pub fn set_rotate_slots(&self, on: bool) {
        self.rotate_slots.store(on, Ordering::Relaxed);
    }

    /// Locks a free slot -- `DocumentsWriterPerThreadPool.getAndLock`. A
    /// thread goes back to the slot it last filled while that slot is free,
    /// so `k` threads indexing at once fill `k` buffers evenly and a refresh
    /// builds `k` segments of about the same size, in parallel. Otherwise
    /// free slots are tried lowest first, so a thread indexing alone keeps
    /// filling one buffer, as Java hands out a free
    /// `DocumentsWriterPerThread` before it creates another. With every slot
    /// taken, it waits on one, picked in rotation.
    fn acquire_slot(&self) -> (usize, RwLockWriteGuard<'_, Dwpt>) {
        #[cfg(any(test, feature = "test-support"))]
        if self.rotate_slots.load(Ordering::Relaxed) {
            let i = self
                .next_slot
                .fetch_add(1, Ordering::Relaxed)
                .checked_rem(self.slots.len())
                .unwrap_or(0);
            return (i, write_slot(&self.slots[i]));
        }
        let (writer, last) = LAST_SLOT.get();
        if writer == self.id {
            if let Some(Ok(guard)) = self.slots.get(last).map(RwLock::try_write) {
                return (last, guard);
            }
        }
        for (i, slot) in self.slots.iter().enumerate() {
            if let Ok(guard) = slot.try_write() {
                LAST_SLOT.set((self.id, i));
                return (i, guard);
            }
        }
        // `n >= 1` (`new` makes at least one slot); `checked_rem` only because
        // the arithmetic gate cannot see that.
        let wait = self
            .next_slot
            .fetch_add(1, Ordering::Relaxed)
            .checked_rem(self.slots.len())
            .unwrap_or(0);
        // Not recorded as this thread's slot: it was only the one waited on
        // (another thread's, or every slot held by a commit), so the thread
        // keeps going back to its own once slots free up.
        (wait, write_slot(&self.slots[wait]))
    }

    /// [`Self::acquire_slot`] for a document of `partition`, in a
    /// partitioning writer: a free slot that is empty or already holds
    /// `partition` -- this thread's last one first, then one holding the
    /// partition, then an empty one -- else any free slot, which the caller
    /// flushes before using, else (every slot taken) the one waited on in
    /// rotation.
    fn acquire_slot_for(&self, partition: Option<&str>) -> (usize, RwLockWriteGuard<'_, Dwpt>) {
        let (writer, last) = LAST_SLOT.get();
        if writer == self.id {
            if let Some(Ok(guard)) = self.slots.get(last).map(RwLock::try_write) {
                if !guard.has_docs() || guard.partition.as_deref() == partition {
                    return (last, guard);
                }
            }
        }
        if let Some(p) = partition {
            let hint = lock(&self.partition_slots).get(p).copied();
            if let Some(i) = hint {
                if let Some(Ok(guard)) = self.slots.get(i).map(RwLock::try_write) {
                    if !guard.has_docs() || guard.partition.as_deref() == partition {
                        LAST_SLOT.set((self.id, i));
                        return (i, guard);
                    }
                }
            }
        }
        let mut empty = None;
        let mut other = None;
        for (i, slot) in self.slots.iter().enumerate() {
            let Ok(guard) = slot.try_write() else {
                continue;
            };
            if !guard.has_docs() {
                empty.get_or_insert(i);
            } else if guard.partition.as_deref() == partition {
                LAST_SLOT.set((self.id, i));
                return (i, guard);
            } else {
                other.get_or_insert(i);
            }
        }
        // Released above, so the slot may have been taken or filled since:
        // whatever it holds now, the caller flushes it if it must.
        for i in empty.into_iter().chain(other) {
            if let Ok(guard) = self.slots[i].try_write() {
                LAST_SLOT.set((self.id, i));
                return (i, guard);
            }
        }
        let wait = self
            .next_slot
            .fetch_add(1, Ordering::Relaxed)
            .checked_rem(self.slots.len())
            .unwrap_or(0);
        (wait, write_slot(&self.slots[wait]))
    }

    /// `DocumentsWriterPerThread.updateDocuments` + `finishDocuments`: the
    /// documents go into the slot first, then -- still under the slot's lock
    /// -- the delete (if any) is appended to the log and the slot applies
    /// every entry it has not seen, limited to the documents it held
    /// *before* these. So an update's delete never reaches its own document,
    /// and the delete and the document are never apart: a commit locks every
    /// slot to take its cut, and sees both or neither.
    fn add(&self, delete: Option<DeleteNode>, docs: Vec<Document>) -> Result<SeqNo> {
        let entries = docs
            .into_iter()
            .map(|doc| Entry {
                doc,
                custom_freq_terms: Vec::new(),
                vectors: Vec::new(),
            })
            .collect();
        self.add_entries(delete, entries)
    }

    fn add_entries(&self, delete: Option<DeleteNode>, docs: Vec<Entry>) -> Result<SeqNo> {
        let cfg = self.cfg();
        // The building threads have no parent marker for a native document.
        if cfg.parent_field_number().is_some() {
            return Err(crate::index_writer::Error::ConcurrentUnsupported(
                "a native document under a parent field (set_parent_field)",
            ));
        }
        let count = docs.len();
        self.add_with(delete, None, count, |dwpt| {
            let before = dwpt.num_docs();
            for (k, entry) in docs.iter().enumerate() {
                cfg.validate_document_vectors(before.saturating_add(k), &entry.vectors)?;
            }
            Ok(Box::new(move |dwpt: &mut Dwpt| {
                let mut added = 0usize;
                for entry in docs {
                    // A plain document is charged its stored fields only, as
                    // before; the extras only when it carries them.
                    let custom = if entry.custom_freq_terms.is_empty() {
                        0
                    } else {
                        custom_freq_terms_ram_bytes(&entry.custom_freq_terms)
                    };
                    added = added
                        .saturating_add(document_ram_bytes(&entry.doc))
                        .saturating_add(custom)
                        .saturating_add(
                            entry
                                .vectors
                                .iter()
                                .map(DocumentVector::ram_bytes)
                                .fold(0usize, usize::saturating_add),
                        );
                    // Explicit documents leave these short (see `Dwpt`).
                    dwpt.fill_docs();
                    let at = dwpt.docs.len();
                    dwpt.custom_freq_terms.resize_with(at, Vec::new);
                    dwpt.vectors.resize_with(at, Vec::new);
                    dwpt.docs.push(entry.doc);
                    dwpt.custom_freq_terms.push(entry.custom_freq_terms);
                    dwpt.vectors.push(entry.vectors);
                }
                Ok(added)
            }) as Fill)
        })
    }

    /// `IndexWriter.addDocuments` for explicit documents
    /// ([`IndexWriter::add_explicit_documents`]), from any thread: each
    /// document is inverted into its slot's postings as it is buffered.
    pub fn add_explicit_documents(&self, docs: Vec<ExplicitDocument>) -> Result<SeqNo> {
        self.add_explicit(None, None, docs, None)
    }

    /// [`Self::add_explicit_documents`] under `partition`
    /// ([`IndexWriter::set_segment_partitioning`]): the documents go to a
    /// buffer holding only that partition, so the segment they flush to
    /// records it. Without partitioning `partition` is ignored.
    pub fn add_explicit_documents_in(
        &self,
        partition: Option<Arc<str>>,
        docs: Vec<ExplicitDocument>,
    ) -> Result<SeqNo> {
        self.add_explicit(partition, None, docs, None)
    }

    /// [`Self::add_explicit_documents_in`] with each document's KNN vectors
    /// (`vectors[i]` is `docs[i]`'s, one per vector field it carries), as
    /// `IndexingChain` takes a `KnnFloatVectorField` beside the document's
    /// other fields.
    pub fn add_explicit_documents_with_vectors_in(
        &self,
        partition: Option<Arc<str>>,
        docs: Vec<ExplicitDocument>,
        vectors: Vec<Vec<DocumentVector>>,
    ) -> Result<SeqNo> {
        self.add_explicit(partition, None, docs, Some(vectors))
    }

    /// `IndexWriter.softUpdateDocuments(term, docs, softDeletes...)` for
    /// explicit documents ([`IndexWriter::soft_update_explicit_documents`]):
    /// `docs` are added and every document matching `term` added before
    /// them gets `soft_deletes`.
    pub fn soft_update_explicit_documents(
        &self,
        term: Term,
        docs: Vec<ExplicitDocument>,
        soft_deletes: &[DocValuesUpdate],
    ) -> Result<SeqNo> {
        self.soft_update_explicit_documents_in(None, term, docs, soft_deletes)
    }

    /// [`Self::soft_update_explicit_documents`] under `partition`, as
    /// [`Self::add_explicit_documents_in`]. The earlier documents matching
    /// `term` get `soft_deletes` whatever partition they are in.
    pub fn soft_update_explicit_documents_in(
        &self,
        partition: Option<Arc<str>>,
        term: Term,
        docs: Vec<ExplicitDocument>,
        soft_deletes: &[DocValuesUpdate],
    ) -> Result<SeqNo> {
        self.soft_update_explicit(partition, term, docs, None, soft_deletes)
    }

    /// [`Self::soft_update_explicit_documents_in`] with each document's KNN
    /// vectors, as [`Self::add_explicit_documents_with_vectors_in`].
    pub fn soft_update_explicit_documents_with_vectors_in(
        &self,
        partition: Option<Arc<str>>,
        term: Term,
        docs: Vec<ExplicitDocument>,
        vectors: Vec<Vec<DocumentVector>>,
        soft_deletes: &[DocValuesUpdate],
    ) -> Result<SeqNo> {
        self.soft_update_explicit(partition, term, docs, Some(vectors), soft_deletes)
    }

    fn soft_update_explicit(
        &self,
        partition: Option<Arc<str>>,
        term: Term,
        docs: Vec<ExplicitDocument>,
        vectors: Option<Vec<Vec<DocumentVector>>>,
        soft_deletes: &[DocValuesUpdate],
    ) -> Result<SeqNo> {
        if soft_deletes.is_empty() {
            return Err(crate::index_writer::Error::NoSoftDeletesSupplied);
        }
        let cfg = self.cfg();
        // `verifyOrCreateDvOnlyField`: an update to a field this writer has
        // never registered would resolve against nothing and vanish.
        for update in soft_deletes {
            cfg.verify_doc_values_update_field(update)?;
        }
        let updates = soft_deletes
            .iter()
            .map(|u| crate::index_writer::retarget_update(u, &term))
            .collect();
        self.add_explicit(
            partition,
            Some(DeleteNode::DocValuesUpdates(updates)),
            docs,
            vectors,
        )
    }

    fn add_explicit(
        &self,
        partition: Option<Arc<str>>,
        delete: Option<DeleteNode>,
        docs: Vec<ExplicitDocument>,
        vectors: Option<Vec<Vec<DocumentVector>>>,
    ) -> Result<SeqNo> {
        let cfg = self.cfg();
        // Every fallible lookup runs before anything is buffered, so a
        // refused document leaves nothing behind.
        let options = cfg.explicit_add_options(&docs)?;
        if let Some(vectors) = &vectors {
            if vectors.len() != docs.len() {
                return Err(crate::index_writer::explicit_error(
                    "one vector list per document",
                ));
            }
            for doc_vectors in vectors {
                cfg.validate_explicit_vectors(doc_vectors)?;
            }
        }
        // A list per document, empty for one without vectors.
        let mut vectors = vectors.map(Vec::into_iter);
        let mut docs = docs;
        // `IndexingChain.processDocument(docId, doc, lastDocInBlock)`: the
        // last document of the add is the parent, `-1` in the parent field.
        if let (Some(parent), Some(last)) = (cfg.parent_field_number(), docs.last_mut()) {
            last.fields
                .doc_values
                .push(lucene_codecs::stored_fields::StoredField {
                    field_number: parent,
                    value: lucene_codecs::stored_fields::FieldValue::Long(-1),
                });
        }
        let streamed = cfg.streamed_stored_mode();
        let count = docs.len();
        let live = self
            .live()
            .then(|| cfg.soft_deletes_field().map(|f| f.number));
        let live_cfg = Arc::clone(&cfg);
        self.add_with(delete, partition, count, move |_| {
            Ok(Box::new(move |dwpt: &mut Dwpt| {
                // The postings count what they grew by, as `bytesUsed` counts
                // `TermsHashPerField`'s pools; the documents' own inverted
                // terms are dropped once they are in.
                let postings_before = dwpt.postings.bytes();
                // Chunk columns copy the documents' doc values and points.
                let columns_before = dwpt.live.chunk_ram_bytes();
                let mut added = 0usize;
                if live.is_some() {
                    dwpt.postings.seal_blocks();
                }
                for (mut doc, options) in docs.into_iter().zip(options) {
                    let doc_vectors = vectors
                        .as_mut()
                        .and_then(Iterator::next)
                        .unwrap_or_default();
                    let stored = Document { fields: doc.stored };
                    if let Some(mode) = streamed {
                        let kept = dwpt
                            .stored
                            .get_mut()
                            .get_or_insert_with(|| StreamedStored::new(mode))
                            .add(&stored, live.is_some());
                        if live.is_some() {
                            // A big document is read from its compressed
                            // chunk rather than held twice.
                            match kept {
                                Some(kept) => dwpt.live.record_stored_chunk(kept),
                                None => dwpt.live.record_stored(&stored.fields),
                            }
                        }
                        dwpt.stored_docs = dwpt.stored_docs.saturating_add(1);
                    } else {
                        if live.is_some() {
                            dwpt.live.record_stored(&stored.fields);
                        }
                        added = added.saturating_add(document_ram_bytes(&stored));
                        dwpt.fill_docs();
                        dwpt.docs.push(stored);
                    }
                    let doc_id =
                        i32::try_from(dwpt.num_docs().saturating_sub(1)).map_err(|_| {
                            crate::index_writer::explicit_error("too many buffered documents")
                        })?;
                    for (inv, options) in doc.fields.inverted.iter_mut().zip(options) {
                        let Some(options) = options else {
                            continue;
                        };
                        dwpt.postings.add_field(doc_id, inv, options)?;
                        // the postings hold them now; the document keeps its norm
                        inv.terms = InvertedTerms::default();
                    }
                    if let Some(soft) = live {
                        dwpt.live.record(doc_id, &doc.fields, soft, &live_cfg);
                    }
                    if !doc_vectors.is_empty() {
                        // Short until a document has vectors (see `Dwpt`).
                        added = added.saturating_add(
                            doc_vectors
                                .iter()
                                .map(DocumentVector::ram_bytes)
                                .fold(0usize, usize::saturating_add),
                        );
                        // The graphs grow as the documents arrive, as
                        // `Lucene99HnswVectorsWriter` builds them, and are
                        // charged as its `ramBytesUsed` charges them: the
                        // flush writes them rather than building them.
                        let graphs_before = dwpt.live.graph_ram_bytes();
                        dwpt.live
                            .record_vectors(doc_id, &doc_vectors, &live_cfg, live.is_some());
                        let graphs_now = dwpt.live.graph_ram_bytes();
                        if graphs_now < graphs_before {
                            // A graph given up: its charge goes back.
                            dwpt.ram_bytes = dwpt
                                .ram_bytes
                                .saturating_sub(graphs_before.saturating_sub(graphs_now));
                        }
                        added = added.saturating_add(graphs_now.saturating_sub(graphs_before));
                        let at = usize::try_from(doc_id).unwrap_or(0);
                        dwpt.vectors.resize_with(at, Vec::new);
                        dwpt.vectors.push(doc_vectors);
                    }
                    added = added.saturating_add(dwpt.explicit.push(doc.fields));
                }
                let grown = dwpt.postings.bytes().saturating_sub(postings_before);
                let columns_now = dwpt.live.chunk_ram_bytes();
                if columns_now < columns_before {
                    // The columns gave up: their charge goes back.
                    dwpt.ram_bytes = dwpt
                        .ram_bytes
                        .saturating_sub(columns_before.saturating_sub(columns_now));
                }
                let columns = columns_now.saturating_sub(columns_before);
                Ok(added
                    .saturating_add(grown)
                    .saturating_add(columns)
                    .saturating_add(Self::recharge_stored(dwpt)))
            }) as Fill)
        })
    }

    /// Brings `dwpt`'s charge for its streamed stored fields to their heap
    /// now: returns what it grew by, and takes off what went to disk.
    fn recharge_stored(dwpt: &mut Dwpt) -> usize {
        let now = dwpt
            .stored
            .get_mut()
            .as_ref()
            .map_or(0, StreamedStored::ram_bytes);
        let before = std::mem::replace(&mut dwpt.stored_charged, now);
        if now < before {
            dwpt.ram_bytes = dwpt.ram_bytes.saturating_sub(before.saturating_sub(now));
        }
        now.saturating_sub(before)
    }

    /// Moves a buffer's stored fields to the `.fdt` of a segment reserved
    /// for it once they pass [`lucene_codecs::stored_fields::SPILL_BYTES`], so from then on
    /// they go to disk as documents arrive (as Lucene's `StoredFieldsConsumer`
    /// writes through the segment a `DocumentsWriterPerThread` names when it
    /// starts) instead of waiting in memory for the flush and counting
    /// against the RAM buffer. The file is pinned in the deleter until the
    /// buffer's flush publishes or fails. Never waits for the control lock:
    /// when it is busy (a commit syncing), the buffer stays in memory and
    /// tries again with its next document.
    fn move_stored_to_disk(&self, dwpt: &mut Dwpt) {
        let Some(stored) = dwpt.stored.get_mut().as_mut() else {
            return;
        };
        if !stored.wants_disk() {
            return;
        }
        let Ok(mut core) = self.core.try_lock() else {
            return;
        };
        let (name, id) = core.writer.reserve_segment();
        let fdt = stored.fdt_name(&name);
        core.writer.hold_new_files(std::slice::from_ref(&fdt));
        drop(core);
        match self.dir.create_output(&fdt) {
            Ok(out) => stored.move_to_disk(out, name, id),
            Err(_) => {
                let _ = lock(&self.core).writer.release_files(&[fdt]);
            }
        }
    }

    /// Empties every slot (`delete_all`, `rollback`), releasing the files
    /// their stored fields were writing.
    fn abort_slots(&self, slots: &mut [RwLockWriteGuard<'_, Dwpt>]) {
        let mut files = Vec::new();
        for (i, dwpt) in slots.iter_mut().enumerate() {
            if let Some(stored) = dwpt.stored.get_mut().take() {
                files.extend(stored.discard());
            }
            // Background routing and insertion stop with the documents.
            dwpt.live.retire_indexes();
            *dwpt.deref_mut() = Dwpt::default();
            self.update_ram(|c| {
                let bytes = c.checkout_for_flush(i);
                c.flush_done(bytes);
            });
            self.slices[i].store(EMPTY_SLOT, Ordering::Release);
        }
        if !files.is_empty() {
            let _ = lock(&self.core).writer.release_files(&files);
        }
    }

    /// The shared tail of every add: `check` validates against the slot
    /// (nothing buffered yet) and returns what buffers the documents; then
    /// the delete, the slice and the flush decision.
    fn add_with(
        &self,
        delete: Option<DeleteNode>,
        partition: Option<Arc<str>>,
        count: usize,
        check: impl FnOnce(&Dwpt) -> Result<Fill>,
    ) -> Result<SeqNo> {
        // BLOCK-GUARD: refused before anything is buffered, as
        // `DocumentsWriterPerThread.updateDocuments` refuses it; a buffered
        // block could never be flushed.
        self.cfg().check_block(count)?;
        let mut flushed = self.pre_update()?;
        let partitioned = self.cfg().partitions_segments();
        let mut evicted = None;
        let (seq_no, batch) = {
            let (i, mut dwpt) = if partitioned {
                self.acquire_slot_for(partition.as_deref())
            } else {
                self.acquire_slot()
            };
            // A buffer holds one partition: one holding another is flushed
            // first (its build runs once the slot is released), and the slot
            // starts over for this partition.
            if partitioned && dwpt.has_docs() && dwpt.partition != partition {
                evicted = self.begin_flush(i, &mut dwpt);
            }
            let before = dwpt.num_docs();
            let fill = match check(&dwpt).and_then(|fill| self.reserve_docs(count).map(|()| fill)) {
                Ok(fill) => fill,
                Err(e) => {
                    drop(dwpt);
                    return Err(self.fail_after_eviction(evicted, e));
                }
            };
            if count > 1 {
                dwpt.has_blocks = true;
            }
            let added = match fill(&mut dwpt) {
                Ok(added) => added,
                Err(e) => {
                    self.buffered_docs.fetch_sub(count, Ordering::AcqRel);
                    drop(dwpt);
                    return Err(self.fail_after_eviction(evicted, e));
                }
            };
            if partitioned {
                if before == 0 {
                    if let Some(p) = &partition {
                        let mut hints = lock(&self.partition_slots);
                        // Stale hints for partitions no longer written are dropped
                        // wholesale; a miss only costs one scan.
                        if hints.len() >= self.slots.len().saturating_mul(4) {
                            hints.clear();
                        }
                        hints.insert(Arc::clone(p), i);
                    }
                }
                dwpt.partition = partition;
            }
            dwpt.ram_bytes = dwpt.ram_bytes.saturating_add(added);
            self.move_stored_to_disk(&mut dwpt);
            if self.live() && dwpt.live.id == 0 {
                dwpt.live.id = self.next_buffer_id.fetch_add(1, Ordering::Relaxed);
            }
            let (nodes, end, seq_no, delete_bytes) = {
                let mut log = lock(&self.log);
                if let Some(node) = delete {
                    log.unhanded_bytes = log.unhanded_bytes.saturating_add(node.ram_bytes());
                    log.nodes.push_back(Arc::new(node));
                }
                let seq_no = log.take_seq();
                // A slot that held nothing has nothing an entry could reach.
                let nodes = if before == 0 {
                    Vec::new()
                } else {
                    log.since(dwpt.slice)
                };
                let end = log.end();
                self.trim(&mut log);
                (nodes, end, seq_no, log.unhanded_bytes)
            };
            self.apply_slice(i, &mut dwpt, &nodes, end, before);
            // `DocumentsWriterFlushControl.doAfterDocument`: the policy runs
            // unless the slot is already marked (its growth then counts as
            // flushing), then a marked slot is flushed by this thread. Without
            // a RAM buffer only the document count and the hard per-slot cap
            // can mark it, so they are checked here without the accounting
            // lock.
            let full = if self.ram_buffer_bytes.is_some() {
                let (num_docs, ram_bytes) = (dwpt.num_docs(), dwpt.ram_bytes);
                self.update_ram(|c| {
                    c.commit_slot_bytes(i, num_docs, ram_bytes);
                    c.set_delete_bytes_used(delete_bytes);
                    if !c.is_flush_pending(i) {
                        self.policy.on_change(c, Some(i));
                        if !c.is_flush_pending(i) && ram_bytes > HARD_MAX_BYTES_PER_SLOT {
                            c.set_flush_pending(i);
                        }
                    }
                    c.is_flush_pending(i)
                })
            } else {
                // `hardMaxBytesPerDWPT` applies whatever the RAM setting.
                self.max_buffered_docs.is_some_and(|m| dwpt.num_docs() >= m)
                    || dwpt.ram_bytes > HARD_MAX_BYTES_PER_SLOT
            };
            let batch = if full {
                self.begin_flush(i, &mut dwpt)
            } else {
                None
            };
            (seq_no, batch)
        };
        if let Some(evicted) = evicted {
            self.complete_flush(evicted)?;
            flushed = true;
        }
        // `DocumentsWriter.postUpdate`. The deletes are applied after this
        // thread's own flush publishes: their ticket comes after its ticket,
        // and Java's ticket queue does not make the applying thread wait for
        // earlier tickets the way publishing in turn here does.
        if let Some(batch) = batch {
            self.complete_flush(batch)?;
            flushed = true;
        } else if self.check_pending_flush_on_update.load(Ordering::Relaxed) {
            flushed |= self.flush_one_pending()?;
        }
        self.maybe_apply_all_deletes()?;
        if flushed {
            self.schedule_merges(MergeTrigger::SegmentFlush)?;
        }
        Ok(seq_no)
    }

    /// An add refused after its slot's buffer of another partition was taken
    /// for a flush: that flush still completes (its documents were
    /// acknowledged, and later tickets wait for its turn), and the add's own
    /// error is the one returned.
    fn fail_after_eviction(
        &self,
        evicted: Option<FlushBatch>,
        error: crate::index_writer::Error,
    ) -> crate::index_writer::Error {
        if let Some(evicted) = evicted {
            // A failed build is recorded by `complete_flush` itself (its
            // ticket retired, its files deleted); the add's error says why
            // this call failed.
            let _ = self.complete_flush(evicted);
        }
        error
    }

    /// Takes slot `i`'s buffer for a segment, with a ticket -- under the slot's
    /// lock, the control lock and the log's, so no delete lands between the
    /// slice being closed and the ticket freezing the published segments'
    /// share.
    fn begin_flush(&self, i: usize, dwpt: &mut Dwpt) -> Option<FlushBatch> {
        if !dwpt.has_docs() {
            return None;
        }
        dwpt.fill_docs();
        // Copied before the locks: the values are shared, only the lists are new.
        let vectors = if self.live() {
            dwpt.vectors.clone()
        } else {
            Vec::new()
        };
        let mut core = lock(&self.core);
        let mut live = self.live_flushes();
        // One read of the log for both, so the slice and the published
        // segments' share end at the same entry. Every entry since the slot's
        // last add came after all of its documents.
        let (slot_nodes, global_nodes, end) = {
            let mut log = lock(&self.log);
            let slot_nodes = log.since(dwpt.slice);
            let global_nodes = Self::take_unhanded(&mut log);
            (slot_nodes, global_nodes, log.end())
        };
        self.apply_slice(i, dwpt, &slot_nodes, end, dwpt.docs.len());
        Self::hand_over(&mut core, &mut live, &global_nodes);
        let ticket = core.writer.begin_segment(
            dwpt.stored
                .get_mut()
                .as_ref()
                .and_then(StreamedStored::reserved),
        );
        let ticket_no = core.next_ticket;
        core.next_ticket = core.next_ticket.saturating_add(1);
        let postings = Arc::new(std::mem::take(&mut dwpt.postings));
        let mut columns = std::mem::take(&mut dwpt.live);
        let graphs = columns.take_graphs();
        columns.drop_lists();
        // Until it is published, a buffer whose documents are all in its
        // columns and packed fields is still read as chunks without
        // postings; any other goes as one segment once published.
        let lean = (self.live()
            && dwpt.custom_freq_terms.is_empty()
            && dwpt.docs.iter().all(|d| d.fields.is_empty()))
        .then(|| live::LeanSource {
            has_blocks: dwpt.has_blocks,
            partition: dwpt.partition.clone(),
        });
        if lean.is_none() {
            columns.drop_chunk_columns();
        }
        if self.live() {
            let flushing = live::Flushing {
                postings: Arc::clone(&postings),
                columns,
                max_doc: i32::try_from(dwpt.docs.len()).unwrap_or(i32::MAX),
                segment: None,
                held_files: Vec::new(),
                vectors,
                lean,
                explicit: dwpt.explicit.snapshot(),
            };
            live.flushing.insert(ticket_no, flushing);
        }
        drop(live);
        drop(core);
        let batch = FlushBatch {
            ticket,
            ticket_no,
            docs: std::mem::take(&mut dwpt.docs),
            custom_freq_terms: std::mem::take(&mut dwpt.custom_freq_terms),
            vectors: std::mem::take(&mut dwpt.vectors),
            explicit: std::mem::take(&mut dwpt.explicit),
            stored: dwpt.stored.get_mut().take(),
            postings,
            has_blocks: std::mem::take(&mut dwpt.has_blocks),
            private: std::mem::take(&mut dwpt.private),
            ram_bytes: dwpt.ram_bytes,
            partition: dwpt.partition.take(),
            graphs,
        };
        dwpt.ram_bytes = 0;
        dwpt.stored_charged = 0;
        self.update_ram(|c| c.checkout_for_flush(i));
        self.slices[i].store(EMPTY_SLOT, Ordering::Release);
        Some(batch)
    }

    /// Waits, under the control lock, until ticket `ticket_no` may publish:
    /// every earlier ticket has, and no commit is holding it back.
    fn wait_turn(&self, ticket_no: u64) -> MutexGuard<'_, Core<'d>> {
        let mut core = lock(&self.core);
        while core.next_publish != ticket_no || core.barrier.is_some_and(|b| ticket_no >= b) {
            core = self
                .changed
                .wait(core)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        core
    }

    /// The buffers live search reads beyond the slots'.
    fn live_flushes(&self) -> MutexGuard<'_, live::LiveFlushes> {
        lock(&self.live_flushes)
    }

    /// Re-reads the segments' `maxDoc` total after the segment lists
    /// changed. A failure to read it leaves the old total in place.
    fn refresh_segment_docs(&self, core: &mut Core<'_>) {
        if let Ok(total) = core.writer.segment_doc_total() {
            self.segment_docs.store(total, Ordering::Release);
        }
    }

    /// `IndexWriter.reserveDocs`: claims room for `added` documents under
    /// the max-docs limit, or refuses them before anything is buffered.
    fn reserve_docs(&self, added: usize) -> Result<()> {
        let before = self.buffered_docs.fetch_add(added, Ordering::AcqRel);
        let total = self
            .segment_docs
            .load(Ordering::Acquire)
            .saturating_add(before)
            .saturating_add(added);
        if total > self.max_docs {
            self.buffered_docs.fetch_sub(added, Ordering::AcqRel);
            return Err(crate::index_writer::Error::TooManyDocs(self.max_docs));
        }
        Ok(())
    }

    /// Marks the ticket whose turn it was as done and wakes the waiters.
    fn retire(&self, mut core: MutexGuard<'_, Core<'d>>) {
        self.refresh_segment_docs(&mut core);
        core.next_publish = core.next_publish.saturating_add(1);
        drop(core);
        self.changed.notify_all();
    }

    /// Builds a batch's segment and resolves its own deletes -- everything a
    /// flush does before the publish, with no lock held.
    #[allow(clippy::too_many_arguments)]
    fn build(
        &self,
        ticket: &SegmentTicket,
        mut docs: Vec<Document>,
        (mut custom_freq_terms, mut vectors, mut packed, postings, graphs): Buffered,
        stored: Option<StreamedStored>,
        has_blocks: bool,
        mut private: BufferedUpdates,
        partition: Option<&str>,
        tracking: &TrackingDirectory<'_>,
    ) -> Result<Built> {
        // Whether or not the flush writes them, the graphs grow no more: an
        // inserter still running stops, and lets go of them and their
        // vectors, rather than finish a graph nothing reads.
        let _retire = RetireGraphs(&graphs);
        let cfg = self.cfg();
        // Kept short when no document has any: nothing reads past them then
        // but a sort, which permutes them.
        pad_lists(&mut custom_freq_terms, docs.len(), cfg.sorts_documents());
        pad_lists(&mut vectors, docs.len(), cfg.sorts_documents());
        // Native documents carry no explicit fields.
        let explicit_docs = cfg.takes_explicit();
        // The stored values were compressed as the documents arrived, in the
        // mode the configuration asked for then; a sort or a codec installed
        // since would need them as documents, which they no longer are.
        if let Some(s) = &stored {
            if cfg.streamed_stored_mode() != Some(s.mode()) {
                return Err(crate::index_writer::explicit_error(
                    "the index sort or codec changed while stored fields were buffered",
                ));
            }
        }
        // An index sort permutes whole documents, so their fields are
        // unpacked; otherwise the flush reads them packed, a document at a
        // time.
        let mut unpacked = if explicit_docs && cfg.sorts_documents() {
            std::mem::take(&mut packed).unpack()?
        } else {
            Vec::new()
        };
        let sort_map = cfg.sort_buffer(
            &mut docs,
            &mut custom_freq_terms,
            &mut vectors,
            &mut unpacked,
            has_blocks,
        )?;
        let explicit = if explicit_docs && !cfg.sorts_documents() {
            ExplicitDocs::Packed(&packed)
        } else {
            ExplicitDocs::Fields(&unpacked)
        };
        // The postings hold insertion-order doc ids; `sort_map` is
        // `new_to_old`.
        let old_to_new = sort_map.as_ref().map(|new_to_old| {
            let mut old_to_new = vec![0usize; new_to_old.len()];
            for (new, &old) in new_to_old.iter().enumerate() {
                old_to_new[old] = new;
            }
            old_to_new
        });
        let postings = explicit_docs.then_some(&*postings);
        let stored = stored.map(|s| s.finish(&ticket.segment_id)).transpose()?;
        let buffer = DocumentBuffer {
            docs: &docs,
            custom_freq_terms: &custom_freq_terms,
            vectors: &vectors,
            explicit,
            postings,
            old_to_new: old_to_new.as_deref(),
            has_blocks,
            stored: stored.as_ref(),
            limit: None,
            partition,
            // Under a sort the flush renumbers them (`old_to_new`).
            graphs: &graphs,
        };
        // This segment's own deletes are resolved here, with no lock held --
        // `DocumentsWriterPerThread.flush`'s `applyDeletes` -- so the publish,
        // which every other flushing thread waits on, is only bookkeeping and
        // the published segments' share. Term deletes first, against the
        // terms the build has just inverted (see `FlushDeletes`).
        let below_limit = |doc: i32, limit: i32| match &sort_map {
            None => doc < limit,
            Some(map) => usize::try_from(doc)
                .ok()
                .and_then(|d| map.get(d))
                .is_some_and(|&old| old < usize::try_from(limit).unwrap_or(0)),
        };
        let mut flush_deletes = FlushDeletes {
            terms: &private.delete_terms,
            below_limit: &below_limit,
            deleted: Vec::new(),
            resolved: false,
        };
        let (mut sci, si_files) = cfg.build_and_write_segment(
            tracking,
            &buffer,
            &ticket.segment_name,
            ticket.segment_id,
            (!private.delete_terms.is_empty()).then_some(&mut flush_deletes),
        )?;
        let FlushDeletes {
            mut deleted,
            resolved,
            ..
        } = flush_deletes;
        if resolved {
            private.delete_terms.clear();
            if !deleted.is_empty() {
                deleted.sort_unstable();
                deleted.dedup();
                sci = deletes::apply_deletes_with_codec(
                    tracking,
                    &sci,
                    None,
                    docs.len(),
                    deleted,
                    self.cfg().codec.as_deref(),
                )?;
            }
        }
        // Then whatever needs the written segment: query deletes, doc-values
        // updates, and term deletes the build did not resolve.
        let sort_map = sort_map.map(|m| (ticket.segment_name.clone(), m));
        let fully_deleted = if private.any() {
            cfg.apply_flush_private_updates(tracking, &private, &mut sci, sort_map.as_ref())?
        } else {
            usize::try_from(sci.del_count).is_ok_and(|n| n == docs.len())
        };
        Ok(Built {
            sci,
            si_files,
            fully_deleted,
        })
    }

    /// Builds `batch`'s segment with no lock held, then publishes it in ticket
    /// order. A build that fails -- or panics -- still retires its ticket,
    /// as `DocumentsWriterPerThread.abort` does, so the tickets behind it do
    /// not wait forever: its files are deleted, and the published segments
    /// still get what its ticket froze for them.
    fn complete_flush(&self, batch: FlushBatch) -> Result<()> {
        let FlushBatch {
            ticket,
            ticket_no,
            docs,
            custom_freq_terms,
            vectors,
            explicit,
            stored,
            postings,
            has_blocks,
            private,
            ram_bytes,
            partition,
            graphs,
        } = batch;
        let doc_count = docs.len();
        // Written as the buffer filled, outside the build's tracking.
        let reserved_fdt = stored
            .as_ref()
            .and_then(|s| s.reserved().map(|(name, _)| s.fdt_name(name)));
        let tracking = TrackingDirectory::new(self.dir);
        let built = catch_unwind(AssertUnwindSafe(|| {
            self.build(
                &ticket,
                docs,
                (custom_freq_terms, vectors, explicit, postings, graphs),
                stored,
                has_blocks,
                private,
                partition.as_deref(),
                &tracking,
            )
        }));
        let mut core = self.wait_turn(ticket_no);
        let mut segment = None;
        let outcome = catch_unwind(AssertUnwindSafe(|| match built {
            Ok(Ok(built)) => {
                let name = built.sci.segment_name.clone();
                if self.live() && !built.fully_deleted {
                    segment = Some(built.sci.clone());
                }
                Ok(core
                    .writer
                    .publish_segment_with(ticket, built.sci, built.si_files, None, None, false)
                    .and_then(|()| {
                        if built.fully_deleted {
                            core.writer.drop_fully_deleted_flushed(&name)
                        } else {
                            Ok(())
                        }
                    }))
            }
            Ok(Err(e)) => {
                Self::abandon(&mut core, ticket, &tracking);
                Ok(Err(e))
            }
            Err(panic) => {
                Self::abandon(&mut core, ticket, &tracking);
                Err(panic)
            }
        }));
        // The pin on the `.fdt` the buffer wrote: a published segment's
        // files are referenced by now, an abandoned one's are deleted.
        if let Some(fdt) = reserved_fdt {
            let _ = core.writer.release_files(&[fdt]);
        }
        // Published or abandoned, under the lock it happened under. A
        // published buffer's files stay while views may read it.
        let published = matches!(outcome, Ok(Ok(Ok(()))));
        // A segment whose files cannot be pinned is still read while they
        // last; a view that finds them gone fails and is answered without it.
        let held = match segment.as_ref().filter(|_| published) {
            Some(sci) => core.writer.hold_segment(sci).unwrap_or_default(),
            None => Vec::new(),
        };
        let unpinned = {
            let mut live = self.live_flushes();
            live.finish(ticket_no, published, segment, held);
            live.take_unpinned()
        };
        self.unpin_with(&mut core.writer, unpinned);
        self.retire(core);
        // Published or abandoned, its buffer is gone either way
        // (`doAfterFlush`), panics included. Published documents are now in
        // the segment total `retire` just read.
        self.update_ram(|c| c.flush_done(ram_bytes));
        let _ = self
            .buffered_docs
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                Some(n.saturating_sub(doc_count))
            });
        match outcome {
            Ok(Ok(result)) => result,
            Ok(Err(panic)) | Err(panic) => resume_unwind(panic),
        }
    }

    /// `DocumentsWriterPerThread.abort`, in the ticket's turn: the batch's
    /// files go, and the published segments still get what its ticket froze.
    /// Errors here are secondary to the build's own and are dropped.
    fn abandon(core: &mut Core<'_>, ticket: SegmentTicket, tracking: &TrackingDirectory<'_>) {
        // The batch held operations already acknowledged to their callers,
        // and they are gone. Java treats such an abort as a tragic event and
        // closes the writer; this writer goes on, but its next commit must
        // not report success without writing (skipping it as "no changes"
        // would hide a failing directory behind an `Ok`).
        core.writer.changed();
        core.writer.abandon_segment(ticket);
        let _ = core.writer.delete_new_files(&tracking.created());
        let _ = core.writer.apply_pending_packets();
    }

    /// Publishes a commit's cut ticket, in its turn.
    fn complete_cut(&self, ticket_no: u64, packet: Option<FrozenBufferedUpdates>) -> Result<()> {
        let mut core = self.wait_turn(ticket_no);
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            core.writer.publish_deletes_ticket(packet)
        }));
        self.retire(core);
        outcome.unwrap_or_else(|panic| resume_unwind(panic))
    }

    /// `DocumentsWriter.flushAllThreads`, with the caller holding
    /// `full_flush`: takes every slot at once, tickets each non-empty buffer
    /// and then a *cut* -- a ticket carrying only what the published segments
    /// owe as of now (`DocumentsWriterFlushQueue`'s global-only
    /// `FlushTicket`) -- and lets the slots go. With every slot held no
    /// update can be half done, so the cut splits the operations cleanly:
    /// everything before it is in these segments and this packet, everything
    /// after in later tickets. With `barrier`, those later tickets may build
    /// but not publish until the caller lifts it.
    ///
    /// The segments are then built at once, one scoped thread each, where
    /// Java's committing thread flushes them one after another: Java inverts
    /// each document as it is added, so its flush only writes, while here
    /// inverting is the build's own work. Returns once the cut is published,
    /// and so every ticket before it, with the sequence number of the last
    /// operation before the cut.
    fn flush_all(&self, barrier: bool) -> Result<(SeqNo, usize)> {
        let (batches, cut, packet, last_seq, handed) = {
            let mut slots: Vec<RwLockWriteGuard<'_, Dwpt>> =
                self.slots.iter().map(write_slot).collect();
            let batches: Vec<FlushBatch> = slots
                .iter_mut()
                .enumerate()
                .filter_map(|(i, dwpt)| self.begin_flush(i, dwpt))
                .collect();
            let mut core = lock(&self.core);
            let mut live = self.live_flushes();
            let (nodes, last_seq, handed) = {
                let mut log = lock(&self.log);
                let nodes = Self::take_unhanded(&mut log);
                (nodes, log.next_seq.saturating_sub(1), log.handed)
            };
            Self::hand_over(&mut core, &mut live, &nodes);
            drop(live);
            let packet = core.writer.begin_deletes_ticket();
            let cut = core.next_ticket;
            core.next_ticket = cut.saturating_add(1);
            if barrier {
                core.barrier = Some(cut.saturating_add(1));
            }
            (batches, cut, packet, last_seq, handed)
        };
        std::thread::scope(|scope| {
            let builds: Vec<_> = batches
                .into_iter()
                .map(|batch| scope.spawn(move || self.complete_flush(batch)))
                .collect();
            let mut result = self.complete_cut(cut, packet);
            for build in builds {
                let outcome = build.join().unwrap_or_else(|panic| resume_unwind(panic));
                result = outcome.and(result);
            }
            result
        })?;
        Ok((last_seq, handed))
    }

    /// Flushes every slot's buffer to a segment and waits until every segment
    /// ticketed so far is published -- `DocumentsWriter.flushAllThreads`.
    pub fn flush(&self) -> Result<()> {
        {
            let _full_flush = lock(&self.full_flush);
            self.flush_all(false)?;
        }
        self.schedule_merges(MergeTrigger::FullFlush)
    }

    /// `IndexWriter.commit`: flushes every buffer, applies every delete issued
    /// before it, and writes `segments_N`. The commit is a clean cut through
    /// the operations of every thread (see [`Self::flush_all`]): each one
    /// issued before it is wholly in the commit, each one after it wholly out
    /// of it, an update's delete and document included. Segments flushed by
    /// other threads meanwhile wait to publish until it is written, as Java
    /// holds back flushes during a full flush.
    ///
    /// Returns the sequence number of the last operation in the commit, as
    /// Java's does: every operation numbered at or below it is in, every one
    /// above it is not.
    pub fn commit(&self) -> Result<SeqNo> {
        self.commit_with(|| Ok::<(), crate::index_writer::Error>(()))
    }

    /// [`Self::commit`], running `before_commit` once every buffer holding
    /// operations from before the cut is flushed and before `segments_N` is
    /// written. Java's `IndexWriter.prepareCommitInternal` iterates the
    /// commit's user data at the same point (`setLiveCommitData` takes a lazy
    /// `Iterable`), so the user data may describe every operation in the
    /// commit -- a maximum sequence number read here is at least that of each
    /// one -- while other threads go on indexing into the emptied buffers.
    /// `before_commit` may set the user data through [`Self::with_writer`];
    /// when it fails, no commit is written.
    pub fn commit_with<E: From<crate::index_writer::Error>>(
        &self,
        before_commit: impl FnOnce() -> std::result::Result<(), E>,
    ) -> std::result::Result<SeqNo, E> {
        let last_seq = {
            let _full_flush = lock(&self.full_flush);
            let _barrier = BarrierGuard(self);
            let (last_seq, handed) = self.flush_all(true)?;
            before_commit()?;
            let mut core = lock(&self.core);
            // Merge-on-commit leaves the segments this writer's merge
            // threads hold alone (`registerMerge` rejects those merges).
            let Core {
                writer, merging, ..
            } = &mut *core;
            writer.prepare_commit_excluding(merging)?;
            writer.finish_commit()?;
            let version = writer.segment_infos().version;
            let unpinned = {
                let mut live = self.live_flushes();
                live.committed(version, handed);
                live.take_unpinned()
            };
            self.unpin_with(writer, unpinned);
            last_seq
        };
        self.schedule_merges(MergeTrigger::FullFlush)?;
        Ok(last_seq)
    }

    /// Runs every merge the merge policy wants right now, each without the
    /// control lock while it reads and writes -- `IndexWriter.maybeMerge` with
    /// the merge executed on the calling thread. Returns how many merges were
    /// published. Only committed segments are merged, as with [`IndexWriter`];
    /// a merged segment becomes durable with the next [`Self::commit`]. A
    /// merge abandoned because a source changed under it ends the round: it
    /// is proposed again next time, not straight away.
    ///
    /// With a merge scheduler ([`Self::with_merge_scheduler`]) the merges are
    /// registered and handed to it instead, and `0` is returned: they run on
    /// the scheduler's threads.
    pub fn maybe_merge(&self) -> Result<usize> {
        if self.scheduler.is_some() {
            self.schedule_merges(MergeTrigger::Explicit)?;
            return Ok(0);
        }
        let mut merged = 0usize;
        loop {
            let plan = {
                let mut core = lock(&self.core);
                if self.merges_blocked() {
                    break;
                }
                let Some(((names, hooks), _)) = core
                    .writer
                    .find_scheduled_merges(MergeTrigger::Explicit, &core.merging)?
                    .into_iter()
                    .next()
                else {
                    break;
                };
                let mut plan = core.writer.begin_merge(&names)?;
                core.writer.attach_merge_hooks(&mut plan, hooks);
                core.merging.extend(names);
                plan
            };
            let tracking = TrackingDirectory::new(self.dir);
            let outcome = catch_unwind(AssertUnwindSafe(|| self.cfg().run_merge(&tracking, &plan)));
            let mut core = lock(&self.core);
            for name in &plan.names {
                core.merging.remove(name);
            }
            // Every way out notifies: `delete_all` waits for `merging` to
            // empty.
            let published = match outcome {
                Ok(Ok(outcome)) => {
                    let published = core.writer.finish_merge(plan, outcome);
                    self.refresh_segment_docs(&mut core);
                    drop(core);
                    self.changed.notify_all();
                    published?
                }
                Ok(Err(e)) => {
                    let _ = core.writer.abort_merge(plan);
                    let _ = core.writer.delete_new_files(&tracking.created());
                    drop(core);
                    self.changed.notify_all();
                    return Err(e);
                }
                Err(panic) => {
                    let _ = core.writer.abort_merge(plan);
                    let _ = core.writer.delete_new_files(&tracking.created());
                    drop(core);
                    self.changed.notify_all();
                    resume_unwind(panic);
                }
            };
            if !published {
                break;
            }
            merged = merged.saturating_add(1);
        }
        Ok(merged)
    }

    /// The merge scheduler's thread: runs [`Self::maybe_merge`] whenever the
    /// index changes, until `stop` is set -- `ConcurrentMergeScheduler`'s merge
    /// thread, on a thread the caller owns (so the borrowed directory needs no
    /// `'static`). Returns how many merges ran.
    pub fn run_merges(&self, stop: &AtomicBool) -> Result<usize> {
        let mut merged = 0usize;
        loop {
            merged = merged.saturating_add(self.maybe_merge()?);
            if stop.load(Ordering::Acquire) {
                return Ok(merged);
            }
            let core = lock(&self.core);
            let _ = self
                .changed
                .wait_timeout(core, Duration::from_millis(20))
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }

    /// Flushes every buffer and hands the single-threaded writer back, merging
    /// on commit again, for what only it does (two-phase commit, rollback).
    /// Segments flushed and not yet committed stay pending in it, and so do
    /// deletes not yet applied; its sequence numbers continue from this
    /// writer's.
    pub fn into_writer(self) -> Result<IndexWriter<'d>> {
        self.flush()?;
        let mut core = self
            .core
            .into_inner()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut log = self
            .log
            .into_inner()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut live = self
            .live_flushes
            .into_inner()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let nodes = Self::take_unhanded(&mut log);
        Self::hand_over(&mut core, &mut live, &nodes);
        let mut writer = core.writer;
        writer.advance_sequence_numbers_to(log.next_seq);
        writer.set_merges_by_caller(false);
        Ok(writer)
    }

    /// `IndexWriter.tryDeleteDocument(reader, docID)`: see
    /// [`IndexWriter::try_delete_document`]. Numbered from this writer's
    /// log; `None` when the reader's segment is no longer in the writer.
    pub fn try_delete_document(
        &self,
        reader_infos: &crate::segment_infos::SegmentInfos,
        doc_id: i32,
    ) -> Result<Option<SeqNo>> {
        let done = lock(&self.core)
            .writer
            .try_delete_resolved(reader_infos, doc_id)?;
        Ok(done.then(|| lock(&self.log).take_seq()))
    }

    /// `IndexWriter.tryUpdateDocValue(reader, docID, fields...)`: see
    /// [`IndexWriter::try_update_doc_value`].
    pub fn try_update_doc_value(
        &self,
        reader_infos: &crate::segment_infos::SegmentInfos,
        doc_id: i32,
        updates: &[DocValuesUpdate],
    ) -> Result<Option<SeqNo>> {
        let done = lock(&self.core)
            .writer
            .try_update_resolved(reader_infos, doc_id, updates)?;
        Ok(done.then(|| lock(&self.log).take_seq()))
    }

    /// `IndexWriter.flushNextBuffer()`: flushes the slot holding the most
    /// buffered RAM (by document count when RAM is not tracked) into a
    /// segment, returning whether there was one to flush.
    pub fn flush_next_buffer(&self) -> Result<bool> {
        let largest = self
            .slots
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let d = read_slot(s);
                (i, (d.ram_bytes, d.num_docs()))
            })
            .filter(|(_, (_, docs))| *docs > 0)
            .max_by_key(|(_, size)| *size)
            .map(|(i, _)| i);
        let Some(i) = largest else {
            return Ok(false);
        };
        let batch = {
            let mut dwpt = write_slot(&self.slots[i]);
            self.begin_flush(i, &mut dwpt)
        };
        let Some(batch) = batch else {
            return Ok(false);
        };
        self.complete_flush(batch)?;
        self.schedule_merges(MergeTrigger::SegmentFlush)?;
        Ok(true)
    }

    /// Documents buffered in every slot, not yet in a segment.
    pub fn pending_doc_count(&self) -> usize {
        self.slots.iter().map(|s| read_slot(s).num_docs()).sum()
    }

    /// RAM held by the buffers, those being flushed included -- what the
    /// flush policy counts (`DocumentsWriterFlushControl`'s active plus
    /// flush bytes). Without a RAM buffer, the slots' own counts.
    pub fn ram_bytes_used(&self) -> usize {
        if self.ram_buffer_bytes.is_some() {
            let ram = lock(&self.ram);
            return ram.active_bytes().saturating_add(ram.flush_bytes());
        }
        self.slots
            .iter()
            .map(|s| read_slot(s).ram_bytes)
            .fold(0, usize::saturating_add)
    }

    /// `IndexWriter.getFlushingBytes()`: the RAM of the buffers marked for
    /// flushing or being built into segments, part of
    /// [`Self::ram_bytes_used`]. `0` without a RAM buffer, which is not
    /// accounted.
    pub fn flushing_bytes(&self) -> usize {
        if self.ram_buffer_bytes.is_some() {
            lock(&self.ram).flush_bytes()
        } else {
            0
        }
    }

    /// [`IndexWriter::has_uncommitted_changes`], counting the slots' buffers
    /// and the deletes not yet handed to the control plane.
    pub fn has_uncommitted_changes(&self) -> bool {
        self.pending_doc_count() > 0
            || {
                let log = lock(&self.log);
                log.handed < log.end()
            }
            || lock(&self.core).writer.has_uncommitted_changes()
    }

    /// `IndexWriter.rollback()`: drops every buffer and every delete not
    /// yet committed and takes the wrapped writer back to its last commit,
    /// as [`Self::delete_all`] takes every slot: flushes already ticketed
    /// publish first, merges not started are dropped and running ones
    /// aborted and waited for.
    pub fn rollback(&self) {
        let _full_flush = lock(&self.full_flush);
        let mut slots: Vec<RwLockWriteGuard<'_, Dwpt>> =
            self.slots.iter().map(write_slot).collect();
        self.abort_slots(&mut slots);
        let mut core = lock(&self.core);
        self.abort_merges(&mut core);
        while core.next_publish != core.next_ticket || !core.merging.is_empty() {
            core = self
                .changed
                .wait(core)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            for progress in &core.running {
                progress.abort();
            }
        }
        let end = {
            let mut log = lock(&self.log);
            let end = log.end();
            log.nodes.clear();
            log.base = end;
            log.handed = end;
            log.unhanded_bytes = 0;
            end
        };
        self.update_ram(|c| c.set_delete_bytes_used(0));
        // One hold of the live lock: nothing reads the buffers dropped but
        // not yet the commit gone back to.
        let mut live = self.live_flushes();
        live.discard(end);
        core.writer.rollback();
        // Back at its last commit, whose readers now hold every document.
        let version = core.writer.segment_infos().version;
        live.committed(version, end);
        let unpinned = live.take_unpinned();
        drop(live);
        self.unpin_with(&mut core.writer, unpinned);
        self.buffered_docs.store(0, Ordering::Release);
        self.refresh_segment_docs(&mut core);
        drop(core);
        drop(slots);
        self.changed.notify_all();
    }
}

impl ConcurrentIndexWriter<'static> {
    /// [`ConcurrentIndexWriter::new`] with `IndexWriterConfig.setMergeScheduler`:
    /// merges are registered after every flush and commit (and by
    /// [`Self::maybe_merge`]) and run as `scheduler` decides -- on the
    /// indexing thread for a [`crate::merge_scheduler::SerialMergeScheduler`],
    /// on a pool of merge threads, stalling indexing threads when merging
    /// falls behind, for a [`crate::merge_scheduler::ConcurrentMergeScheduler`].
    ///
    /// The writer is returned in an `Arc` because the scheduler's threads
    /// hold it while they merge, which is also why the directory must be
    /// `'static`. [`Self::close_merges`] waits for them; after it,
    /// `Arc::try_unwrap` gives the writer back for [`Self::into_writer`].
    pub fn with_merge_scheduler(
        writer: IndexWriter<'static>,
        slots: usize,
        scheduler: Arc<dyn MergeScheduler>,
    ) -> Result<Arc<Self>> {
        let mut w = Self::new(writer, slots)?;
        Ok(Arc::new_cyclic(|me: &Weak<Self>| {
            let source: Weak<dyn MergeSource> = me.clone();
            w.scheduler = Some(SchedulerHook { scheduler, source });
            w
        }))
    }
}

/// `DirectoryReader.open(IndexWriter)` over a concurrent writer: a full flush
/// (every slot's buffer to a segment, every delete issued before it applied)
/// and then the live segment list, pinned -- `IndexWriter.getReader`, which
/// also flushes all threads and then `maybeMerge(GET_READER)`.
impl<'d> crate::nrt::NrtSource for ConcurrentIndexWriter<'d> {
    fn directory(&self) -> &dyn Directory {
        self.dir
    }

    fn nrt_snapshot(
        &self,
        _apply_all_deletes: bool,
        _write_all_deletes: bool,
    ) -> Result<crate::nrt::NrtSnapshot> {
        let full_flush = lock(&self.full_flush);
        // With live search on, the snapshot is a reader's cut through the
        // buffers as a commit is: no later flush publishes before it is
        // taken, so it holds exactly the publishes and deletes before the
        // cut, and views of it read the rest from the buffers.
        let live = self.live();
        let barrier = live.then(|| BarrierGuard(self));
        let (_, handed) = self.flush_all(live)?;
        let mut core = lock(&self.core);
        // Java's `segmentInfos` holds what was just flushed, so merges see
        // it before any commit does.
        core.writer.fold_flushed_segments();
        let scheduled = !self.merges_blocked()
            && self
                .scheduler
                .as_ref()
                .is_some_and(|hook| hook.source.strong_count() > 0);
        let mut waiting: Option<(Vec<String>, std::time::Instant)> = None;
        if scheduled {
            // `maybeMerge(SEGMENT_FLUSH)`, which Java's indexing threads run
            // as each flushed segment publishes: natural merges take the new
            // segments first, so the full-flush merges below seldom find one
            // left to wait for.
            self.register_merges(&mut core, MergeTrigger::SegmentFlush)?;
            let Core {
                writer, merging, ..
            } = &mut *core;
            match writer.full_flush_merges(MergeTrigger::GetReader, merging)? {
                Some((merges, wait)) if merges.iter().all(|((_, hooks), _)| hooks.is_none()) => {
                    // `IndexWriter.getReader` with `maxFullFlushMergeWaitMillis`:
                    // the merges run on the merge threads, and the reader
                    // waits for them up to the wait.
                    let mut names = Vec::new();
                    for ((group, _), bytes) in merges {
                        merging.extend(group.iter().cloned());
                        names.extend(group.iter().cloned());
                        self.pending_merges
                            .push(Arc::new(ScheduledMerge::new(group, bytes)));
                    }
                    let now = std::time::Instant::now();
                    waiting = Some((names, now.checked_add(wait).unwrap_or(now)));
                }
                Some(_) => {
                    // Merges with hooks run here, as `merge_on_full_flush` runs them.
                    writer.merge_on_full_flush(MergeTrigger::GetReader, merging)?;
                }
                None => {}
            }
        } else {
            let Core {
                writer, merging, ..
            } = &mut *core;
            writer.merge_on_full_flush(MergeTrigger::GetReader, merging)?;
        }
        drop(core);
        let Some((names, until)) = waiting else {
            let snapshot = self.nrt_snapshot_keyed(live.then_some(handed));
            drop(barrier);
            drop(full_flush);
            self.schedule_merges(MergeTrigger::GetReader)?;
            return snapshot;
        };
        // The wait holds no full flush, as Java awaits the merges outside
        // `fullFlushLock`: commits and other refreshes go on meanwhile. With
        // live search on, the cut stays held until the snapshot is taken.
        let full_flush = if live {
            Some(full_flush)
        } else {
            drop(full_flush);
            None
        };
        self.schedule_merges(MergeTrigger::GetReader)?;
        {
            let mut core = lock(&self.core);
            while names.iter().any(|name| core.merging.contains(name)) {
                let now = std::time::Instant::now();
                if now >= until {
                    break;
                }
                core = self
                    .changed
                    .wait_timeout(core, until.saturating_duration_since(now))
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .0;
            }
        }
        // No flush half-published under the snapshot: take it between full
        // flushes.
        let _full_flush = full_flush.unwrap_or_else(|| lock(&self.full_flush));
        let snapshot = self.nrt_snapshot_keyed(live.then_some(handed));
        drop(barrier);
        snapshot
    }

    fn nrt_is_current(&self, infos: &crate::segment_infos::SegmentInfos) -> Result<bool> {
        if self.pending_doc_count() > 0 {
            return Ok(false);
        }
        {
            let log = lock(&self.log);
            if log.handed < log.end() {
                return Ok(false);
            }
        }
        Ok(lock(&self.core).writer.nrt_is_current(infos))
    }
}

/// The writer as its scheduler sees it: `IndexWriter`'s own `MergeSource`.
impl MergeSource for ConcurrentIndexWriter<'static> {
    fn next_merge(&self) -> Option<Arc<ScheduledMerge>> {
        self.pending_merges.pop()
    }

    fn on_merge_finished(&self, merge: &Arc<ScheduledMerge>) {
        let mut core = lock(&self.core);
        for name in &merge.segments {
            core.merging.remove(name);
        }
        drop(core);
        // `delete_all` waits for `merging` to empty.
        self.changed.notify_all();
    }

    fn has_pending_merges(&self) -> bool {
        !self.pending_merges.is_empty()
    }

    fn merge(
        &self,
        merge: &Arc<ScheduledMerge>,
        limiter: Option<Arc<MergeRateLimiter>>,
    ) -> Result<()> {
        self.run_scheduled_merge(merge, limiter)
    }
}

/// A [`Directory`] that records every file created through it, so a build
/// that fails can delete exactly its own files -- Java's
/// `TrackingDirectoryWrapper`. `IndexFileDeleter::refresh` would instead
/// delete every unreferenced file, including another thread's segment in the
/// middle of being written.
struct TrackingDirectory<'a> {
    inner: &'a dyn Directory,
    created: Mutex<Vec<String>>,
}

impl<'a> TrackingDirectory<'a> {
    fn new(inner: &'a dyn Directory) -> Self {
        TrackingDirectory {
            inner,
            created: Mutex::new(Vec::new()),
        }
    }

    fn created(&self) -> Vec<String> {
        lock(&self.created).clone()
    }
}

impl Directory for TrackingDirectory<'_> {
    fn list_all(&self) -> lucene_store::Result<Vec<String>> {
        self.inner.list_all()
    }

    fn open(&self, name: &str) -> lucene_store::Result<Input> {
        self.inner.open(name)
    }

    fn open_aligned(&self, name: &str, alignment: usize) -> lucene_store::Result<Input> {
        self.inner.open_aligned(name, alignment)
    }

    fn create_output(&self, name: &str) -> lucene_store::Result<FsIndexOutput> {
        let output = self.inner.create_output(name)?;
        lock(&self.created).push(name.to_string());
        Ok(output)
    }

    fn create_output_with_estimate(
        &self,
        name: &str,
        estimated_bytes: Option<u64>,
    ) -> lucene_store::Result<FsIndexOutput> {
        let output = self
            .inner
            .create_output_with_estimate(name, estimated_bytes)?;
        lock(&self.created).push(name.to_string());
        Ok(output)
    }

    fn sync(&self, names: &[String]) -> lucene_store::Result<()> {
        self.inner.sync(names)
    }

    fn rename(&self, source: &str, dest: &str) -> lucene_store::Result<()> {
        self.inner.rename(source, dest)
    }

    fn delete_file(&self, name: &str) -> lucene_store::Result<()> {
        self.inner.delete_file(name)
    }

    fn sync_meta_data(&self) -> lucene_store::Result<()> {
        self.inner.sync_meta_data()
    }

    fn obtain_lock(&self, name: &str) -> lucene_store::Result<Box<dyn lucene_store::Lock>> {
        self.inner.obtain_lock(name)
    }

    fn file_length(&self, name: &str) -> lucene_store::Result<u64> {
        self.inner.file_length(name)
    }

    fn pending_deletions(&self) -> lucene_store::Result<std::collections::BTreeSet<String>> {
        self.inner.pending_deletions()
    }
}

/// Retires a flushed buffer's graphs when dropped ([`BufferGraph::retire`]).
///
/// [`BufferGraph::retire`]: lucene_codecs::buffer_vectors::BufferGraph::retire
struct RetireGraphs<'a>(&'a [crate::index_writer::BufferGraphField]);

impl Drop for RetireGraphs<'_> {
    fn drop(&mut self) {
        for g in self.0 {
            g.graph.retire();
        }
    }
}

#[cfg(test)]
mod partition_tests;

#[cfg(test)]
mod tests;
