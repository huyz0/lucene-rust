//! Many indexing threads, one index -- M4's T4.3, Java's `DocumentsWriter`
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
//! Vectors and custom-frequency postings are single-writer features for now
//! ([`IndexWriter::add_document_with_vectors`],
//! [`IndexWriter::add_document_with_custom_freq_terms`]); soft deletes,
//! `deleteAll` and two-phase commit too -- [`ConcurrentIndexWriter::into_writer`]
//! hands the writer back for them. A merge whose source takes a doc-values
//! update while it runs is abandoned and retried later rather than carried --
//! see [`IndexWriter::finish_merge`]. Each ticket's frozen packet is applied to
//! every published segment under the control lock, where Java applies
//! packets outside `IndexWriter`'s monitor, per segment; that is the cost the
//! update benchmark still pays (`docs/milestones/m4-write-path-hardened.md`,
//! T4.3). The merge policy's segment statistics are read from disk under the
//! control lock too.

use std::collections::{HashSet, VecDeque};
use std::ops::DerefMut;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, Weak};
use std::time::Duration;

use lucene_codecs::stored_fields::Document;
use lucene_store::{Directory, FsIndexOutput, Input};

use crate::buffered_updates::{
    BufferedUpdates, DeleteQuery, DocValuesUpdate, FrozenBufferedUpdates, SeqNo, Term,
};
use crate::deletes;
use crate::flush_policy::{FlushByRamOrCountsPolicy, FlushControl, FlushPolicy};
use crate::index_writer::{
    buffer_node, document_ram_bytes, DeleteNode, DocumentBuffer, ExplicitFields, FlushDeletes,
    IndexWriter, IndexingConfig, Result, SegmentTicket, DISABLE_AUTO_FLUSH,
};
use crate::merge_policy::{MergePolicyConfig, MergeTrigger};
use crate::merge_rate_limiter::MergeRateLimiter;
use crate::merge_scheduler::{MergeScheduler, MergeSource, PendingMerges, ScheduledMerge};
use crate::segment_infos::SegmentCommitInfo;
use crate::stall_control::DocumentsWriterStallControl;

/// One indexing slot's buffer -- `DocumentsWriterPerThread`, before its flush.
#[derive(Default)]
struct Dwpt {
    docs: Vec<Document>,
    has_blocks: bool,
    ram_bytes: usize,
    /// This buffer's share of every delete issued since it last flushed, each
    /// limited to the documents it held at the time.
    private: BufferedUpdates,
    /// The absolute position in the delete log up to which `private` is
    /// current -- `DeleteSlice.sliceTail`.
    slice: usize,
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
}

/// A buffer that filled (or was flushed): everything its segment is built from.
struct FlushBatch {
    ticket: SegmentTicket,
    ticket_no: u64,
    docs: Vec<Document>,
    has_blocks: bool,
    private: BufferedUpdates,
    /// The slot's `ram_bytes` when it was taken: what this flush counts
    /// toward `flushBytes` until it is published.
    ram_bytes: usize,
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
const HARD_MAX_BYTES_PER_SLOT: usize = 1945 * 1024 * 1024;

/// A slot's advertised slice while it holds no document: it has nothing a
/// log entry could reach, so it holds back no trimming, and its next add
/// starts from the log's end.
const EMPTY_SLOT: usize = usize::MAX;

/// See the module documentation.
pub struct ConcurrentIndexWriter<'d> {
    dir: &'d dyn Directory,
    cfg: Arc<IndexingConfig>,
    core: Mutex<Core<'d>>,
    /// Signalled when a segment publishes or the index changes.
    changed: Condvar,
    slots: Vec<Mutex<Dwpt>>,
    /// Where each slot's slice stands ([`EMPTY_SLOT`] while it is empty),
    /// readable without its lock, so the log can be trimmed.
    slices: Vec<AtomicUsize>,
    log: Mutex<DeleteLog>,
    /// Held for the whole of a [`Self::flush`] or [`Self::commit`]: one full
    /// flush at a time, as Java's `IndexWriter.fullFlushLock`.
    full_flush: Mutex<()>,
    /// Round-robin start for choosing a slot.
    next_slot: AtomicUsize,
    max_buffered_docs: Option<usize>,
    ram_buffer_bytes: Option<usize>,
    merge_policy: Option<MergePolicyConfig>,
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
    /// Shares `writer` among up to `slots` indexing threads at once. Its
    /// configuration is frozen from here on -- the building threads all read
    /// it -- and merging passes to [`Self::maybe_merge`], so that merges run
    /// on the caller's merge thread rather than inside a commit. Anything it
    /// had buffered is flushed first.
    pub fn new(mut writer: IndexWriter<'d>, slots: usize) -> Result<Self> {
        writer.flush()?;
        writer.set_merges_by_caller(true);
        let slots = slots.max(1);
        let merge_policy = writer.merge_policy().cloned();
        let max_buffered_docs = usize::try_from(writer.max_buffered_docs())
            .ok()
            .filter(|_| writer.max_buffered_docs() != DISABLE_AUTO_FLUSH);
        let ram_buffer_bytes = (writer.ram_buffer_size_mb() > 0.0)
            .then(|| (writer.ram_buffer_size_mb() * 1024.0 * 1024.0) as usize);
        let log = DeleteLog {
            next_seq: writer.next_sequence_number_peek(),
            ..DeleteLog::default()
        };
        Ok(ConcurrentIndexWriter {
            dir: writer.dir(),
            cfg: writer.shared_config(),
            core: Mutex::new(Core {
                writer,
                next_ticket: 0,
                next_publish: 0,
                barrier: None,
                merging: HashSet::new(),
            }),
            changed: Condvar::new(),
            slots: (0..slots).map(|_| Mutex::new(Dwpt::default())).collect(),
            slices: (0..slots).map(|_| AtomicUsize::new(EMPTY_SLOT)).collect(),
            log: Mutex::new(log),
            full_flush: Mutex::new(()),
            next_slot: AtomicUsize::new(0),
            max_buffered_docs,
            ram_buffer_bytes,
            merge_policy,
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
        })
    }

    /// `IndexWriter.updatePendingMerges`: registers every merge the policy
    /// proposes among segments not already merging -- claiming their
    /// segments -- for the scheduler to take. Nothing without a policy.
    fn register_merges(&self, core: &mut Core<'_>) -> Result<()> {
        let Some(policy) = &self.merge_policy else {
            return Ok(());
        };
        while let Some((names, estimated)) = core.writer.next_merge_sized(policy, &core.merging)? {
            core.merging.extend(names.iter().cloned());
            self.pending_merges
                .push(Arc::new(ScheduledMerge::new(names, estimated)));
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
        self.register_merges(&mut lock(&self.core))?;
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
        };
        let plan = {
            let mut core = lock(&self.core);
            match core.writer.begin_merge(&merge.segments) {
                Ok(plan) => plan,
                Err(e) => {
                    release(&mut core);
                    return Err(e);
                }
            }
        };
        let tracking = TrackingDirectory::new(self.dir);
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            self.cfg.run_merge_limited(&tracking, &plan, limiter)
        }));
        let mut core = lock(&self.core);
        release(&mut core);
        let result = match outcome {
            Ok(Ok(outcome)) => core
                .writer
                .finish_merge(plan, outcome)
                .and_then(|_| self.register_merges(&mut core)),
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
                resume_unwind(panic);
            }
        };
        drop(core);
        self.changed.notify_all();
        result
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
        for (i, flag) in self.pending.iter().enumerate() {
            flag.store(ram.is_flush_pending(i), Ordering::Release);
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

    /// `DocumentsWriterFlushControl.nextPendingFlush` + `doFlush`: takes one
    /// buffer the flush policy marked, whose slot is free, and flushes it on
    /// this thread. Returns whether it flushed one.
    fn flush_one_pending(&self) -> Result<bool> {
        for (j, flag) in self.pending.iter().enumerate() {
            if !flag.load(Ordering::Acquire) {
                continue;
            }
            let Ok(mut dwpt) = self.slots[j].try_lock() else {
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
            let nodes = Self::take_unhanded(&mut lock(&self.log));
            Self::hand_over(&mut core, &nodes);
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
    /// finish first (`abortMerges` waits for them too), and every delete
    /// buffered so far is cleared. Returns the operation's sequence number.
    pub fn delete_all(&self) -> Result<SeqNo> {
        let _full_flush = lock(&self.full_flush);
        let mut slots: Vec<MutexGuard<'_, Dwpt>> = self.slots.iter().map(lock).collect();
        for (i, dwpt) in slots.iter_mut().enumerate() {
            *dwpt.deref_mut() = Dwpt::default();
            self.update_ram(|c| {
                let bytes = c.checkout_for_flush(i);
                c.flush_done(bytes);
            });
            self.slices[i].store(EMPTY_SLOT, Ordering::Release);
        }
        let mut core = lock(&self.core);
        while let Some(merge) = self.pending_merges.pop() {
            for name in &merge.segments {
                core.merging.remove(name);
            }
        }
        while core.next_publish != core.next_ticket || !core.merging.is_empty() {
            core = self
                .changed
                .wait(core)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        let seq_no = {
            let mut log = lock(&self.log);
            let end = log.end();
            log.nodes.clear();
            log.base = end;
            log.handed = end;
            log.unhanded_bytes = 0;
            log.take_seq()
        };
        self.update_ram(|c| c.set_delete_bytes_used(0));
        core.writer.delete_all()?;
        drop(core);
        drop(slots);
        self.changed.notify_all();
        Ok(seq_no)
    }

    /// `IndexWriter.updateDocValues(Term, Field...)`.
    pub fn update_doc_values(&self, term: Term, updates: &[DocValuesUpdate]) -> Result<SeqNo> {
        let node = self.cfg.doc_values_update_node(&term, updates)?;
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
    fn hand_over(core: &mut Core<'_>, nodes: &[Arc<DeleteNode>]) {
        for node in nodes {
            core.writer.buffer_global_delete(node);
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
        dwpt.slice = end;
        self.slices[i].store(end, Ordering::Release);
    }

    /// Locks a free slot -- any, trying each once from a rotating start, then
    /// waiting on one -- `DocumentsWriterPerThreadPool.getAndLock`.
    fn acquire_slot(&self) -> (usize, MutexGuard<'_, Dwpt>) {
        let n = self.slots.len();
        // `n >= 1` (`new` makes at least one slot); `checked_rem` only because
        // the arithmetic gate cannot see that.
        let start = self
            .next_slot
            .fetch_add(1, Ordering::Relaxed)
            .checked_rem(n)
            .unwrap_or(0);
        for k in 0..n {
            let i = start.saturating_add(k).checked_rem(n).unwrap_or(0);
            if let Ok(guard) = self.slots[i].try_lock() {
                return (i, guard);
            }
        }
        (start, lock(&self.slots[start]))
    }

    /// `DocumentsWriterPerThread.updateDocuments` + `finishDocuments`: the
    /// documents go into the slot first, then -- still under the slot's lock
    /// -- the delete (if any) is appended to the log and the slot applies
    /// every entry it has not seen, limited to the documents it held
    /// *before* these. So an update's delete never reaches its own document,
    /// and the delete and the document are never apart: a commit locks every
    /// slot to take its cut, and sees both or neither.
    fn add(&self, delete: Option<DeleteNode>, docs: Vec<Document>) -> Result<SeqNo> {
        let mut flushed = self.pre_update()?;
        let (seq_no, batch) = {
            let (i, mut dwpt) = self.acquire_slot();
            let before = dwpt.docs.len();
            if docs.len() > 1 {
                dwpt.has_blocks = true;
            }
            let mut added = 0usize;
            for doc in docs {
                added = added.saturating_add(document_ram_bytes(&doc));
                dwpt.docs.push(doc);
            }
            dwpt.ram_bytes = dwpt.ram_bytes.saturating_add(added);
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
            // a RAM buffer only the document count can mark it, so it is
            // checked here without the accounting lock.
            let full = if self.ram_buffer_bytes.is_some() {
                let (num_docs, ram_bytes) = (dwpt.docs.len(), dwpt.ram_bytes);
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
                self.max_buffered_docs.is_some_and(|m| dwpt.docs.len() >= m)
            };
            let batch = if full {
                self.begin_flush(i, &mut dwpt)
            } else {
                None
            };
            (seq_no, batch)
        };
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

    /// Takes slot `i`'s buffer for a segment, with a ticket -- under the slot's
    /// lock, the control lock and the log's, so no delete lands between the
    /// slice being closed and the ticket freezing the published segments'
    /// share.
    fn begin_flush(&self, i: usize, dwpt: &mut Dwpt) -> Option<FlushBatch> {
        if dwpt.docs.is_empty() {
            return None;
        }
        let mut core = lock(&self.core);
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
        Self::hand_over(&mut core, &global_nodes);
        let ticket = core.writer.begin_segment();
        let ticket_no = core.next_ticket;
        core.next_ticket = core.next_ticket.saturating_add(1);
        drop(core);
        let batch = FlushBatch {
            ticket,
            ticket_no,
            docs: std::mem::take(&mut dwpt.docs),
            has_blocks: std::mem::take(&mut dwpt.has_blocks),
            private: std::mem::take(&mut dwpt.private),
            ram_bytes: dwpt.ram_bytes,
        };
        dwpt.ram_bytes = 0;
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

    /// Marks the ticket whose turn it was as done and wakes the waiters.
    fn retire(&self, mut core: MutexGuard<'_, Core<'d>>) {
        core.next_publish = core.next_publish.saturating_add(1);
        drop(core);
        self.changed.notify_all();
    }

    /// Builds a batch's segment and resolves its own deletes -- everything a
    /// flush does before the publish, with no lock held.
    fn build(
        &self,
        ticket: &SegmentTicket,
        mut docs: Vec<Document>,
        has_blocks: bool,
        mut private: BufferedUpdates,
        tracking: &TrackingDirectory<'_>,
    ) -> Result<Built> {
        let mut custom_freq_terms = vec![Vec::new(); docs.len()];
        let mut vectors = vec![Vec::new(); docs.len()];
        let sort_map =
            self.cfg
                .sort_buffer(&mut docs, &mut custom_freq_terms, &mut vectors, has_blocks)?;
        // The concurrent writer takes native documents only.
        let explicit = vec![ExplicitFields::default(); docs.len()];
        let buffer = DocumentBuffer {
            docs: &docs,
            custom_freq_terms: &custom_freq_terms,
            vectors: &vectors,
            explicit: &explicit,
            has_blocks,
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
        let (mut sci, si_files) = self.cfg.build_and_write_segment(
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
                sci = deletes::apply_deletes(tracking, &sci, None, docs.len(), deleted)?;
            }
        }
        // Then whatever needs the written segment: query deletes, doc-values
        // updates, and term deletes the build did not resolve.
        let sort_map = sort_map.map(|m| (ticket.segment_name.clone(), m));
        let fully_deleted = if private.any() {
            self.cfg
                .apply_flush_private_updates(tracking, &private, &mut sci, sort_map.as_ref())?
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
            has_blocks,
            private,
            ram_bytes,
        } = batch;
        let tracking = TrackingDirectory::new(self.dir);
        let built = catch_unwind(AssertUnwindSafe(|| {
            self.build(&ticket, docs, has_blocks, private, &tracking)
        }));
        let mut core = self.wait_turn(ticket_no);
        let outcome = catch_unwind(AssertUnwindSafe(|| match built {
            Ok(Ok(built)) => {
                let name = built.sci.segment_name.clone();
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
        self.retire(core);
        // Published or abandoned, its buffer is gone either way
        // (`doAfterFlush`), panics included.
        self.update_ram(|c| c.flush_done(ram_bytes));
        match outcome {
            Ok(Ok(result)) => result,
            Ok(Err(panic)) | Err(panic) => resume_unwind(panic),
        }
    }

    /// `DocumentsWriterPerThread.abort`, in the ticket's turn: the batch's
    /// files go, and the published segments still get what its ticket froze.
    /// Errors here are secondary to the build's own and are dropped.
    fn abandon(core: &mut Core<'_>, ticket: SegmentTicket, tracking: &TrackingDirectory<'_>) {
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
    fn flush_all(&self, barrier: bool) -> Result<SeqNo> {
        let (batches, cut, packet, last_seq) = {
            let mut slots: Vec<MutexGuard<'_, Dwpt>> = self.slots.iter().map(lock).collect();
            let batches: Vec<FlushBatch> = slots
                .iter_mut()
                .enumerate()
                .filter_map(|(i, dwpt)| self.begin_flush(i, dwpt))
                .collect();
            let mut core = lock(&self.core);
            let (nodes, last_seq) = {
                let mut log = lock(&self.log);
                let nodes = Self::take_unhanded(&mut log);
                (nodes, log.next_seq.saturating_sub(1))
            };
            Self::hand_over(&mut core, &nodes);
            let packet = core.writer.begin_deletes_ticket();
            let cut = core.next_ticket;
            core.next_ticket = cut.saturating_add(1);
            if barrier {
                core.barrier = Some(cut.saturating_add(1));
            }
            (batches, cut, packet, last_seq)
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
        Ok(last_seq)
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
        let last_seq = {
            let _full_flush = lock(&self.full_flush);
            let _barrier = BarrierGuard(self);
            let last_seq = self.flush_all(true)?;
            let mut core = lock(&self.core);
            core.writer.commit()?;
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
        let Some(policy) = &self.merge_policy else {
            return Ok(0);
        };
        let mut merged = 0usize;
        loop {
            let plan = {
                let mut core = lock(&self.core);
                let Some(names) = core.writer.next_merge(policy, &core.merging)? else {
                    break;
                };
                let plan = core.writer.begin_merge(&names)?;
                core.merging.extend(names);
                plan
            };
            let tracking = TrackingDirectory::new(self.dir);
            let outcome = catch_unwind(AssertUnwindSafe(|| self.cfg.run_merge(&tracking, &plan)));
            let mut core = lock(&self.core);
            for name in &plan.names {
                core.merging.remove(name);
            }
            let published = match outcome {
                Ok(Ok(outcome)) => core.writer.finish_merge(plan, outcome)?,
                Ok(Err(e)) => {
                    let _ = core.writer.abort_merge(plan);
                    let _ = core.writer.delete_new_files(&tracking.created());
                    return Err(e);
                }
                Err(panic) => {
                    let _ = core.writer.abort_merge(plan);
                    let _ = core.writer.delete_new_files(&tracking.created());
                    drop(core);
                    resume_unwind(panic);
                }
            };
            drop(core);
            self.changed.notify_all();
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
    /// on commit again, for what only it does (two-phase commit, rollback,
    /// vectors). Segments flushed and not yet committed stay pending in it,
    /// and so do deletes not yet applied; its sequence numbers continue from
    /// this writer's.
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
        let nodes = Self::take_unhanded(&mut log);
        Self::hand_over(&mut core, &nodes);
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
                let d = lock(s);
                (i, (d.ram_bytes, d.docs.len()))
            })
            .filter(|(_, (_, docs))| *docs > 0)
            .max_by_key(|(_, size)| *size)
            .map(|(i, _)| i);
        let Some(i) = largest else {
            return Ok(false);
        };
        let batch = {
            let mut dwpt = lock(&self.slots[i]);
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
        self.slots.iter().map(|s| lock(s).docs.len()).sum()
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
        let snapshot = {
            let _full_flush = lock(&self.full_flush);
            self.flush_all(false)?;
            lock(&self.core).writer.nrt_snapshot_of_live_view()?
        };
        self.schedule_merges(MergeTrigger::GetReader)?;
        Ok(snapshot)
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

    fn create_output(&self, name: &str) -> lucene_store::Result<FsIndexOutput> {
        let output = self.inner.create_output(name)?;
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

#[cfg(test)]
mod tests {
    // The arithmetic gate is about values read off disk; a test's `i + 1` is
    // not one. See docs/arithmetic-gate.md.
    #![allow(clippy::arithmetic_side_effects)]

    use std::collections::BTreeMap;

    use lucene_codecs::field_infos::{FieldInfo, IndexOptions};
    use lucene_codecs::stored_fields::{self, FieldValue, StoredField};
    use lucene_store::FsDirectory;
    use lucene_util::test_support::TempDir;

    use super::*;
    use crate::index_writer::Error;
    use crate::segment_info::LuceneVersion;
    use crate::{deletes, segment_info, segment_infos};

    fn fields() -> Vec<FieldInfo> {
        vec![
            FieldInfo {
                index_options: IndexOptions::Docs,
                omit_norms: true,
                ..FieldInfo::new("id", 0)
            },
            FieldInfo {
                index_options: IndexOptions::DocsAndFreqs,
                ..FieldInfo::new("body", 1)
            },
        ]
    }

    fn doc(id: &str, version: u32) -> Document {
        Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String(id.to_string()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::String(format!("v{version} common")),
                },
            ],
        }
    }

    fn writer(dir: &FsDirectory, max_buffered_docs: i32) -> IndexWriter<'_> {
        let mut w = IndexWriter::open(
            dir,
            fields(),
            "Lucene104",
            LuceneVersion {
                major: 10,
                minor: 5,
                bugfix: 0,
            },
        )
        .unwrap();
        w.set_postings_field(Some("id")).unwrap();
        w.add_postings_field("body").unwrap();
        w.set_max_buffered_docs(max_buffered_docs).unwrap();
        w.set_ram_buffer_size_mb(DISABLE_AUTO_FLUSH_MB).unwrap();
        w.set_merge_policy(Some(MergePolicyConfig {
            max_merge_at_once: 4,
            segments_per_tier: 3,
            floor_segment_size: 1 << 30,
            ..MergePolicyConfig::default()
        }));
        w
    }

    use crate::index_writer::DISABLE_AUTO_FLUSH_MB;

    /// id -> body of every live document of the latest commit, read through
    /// the stored fields and `.liv` files -- plus the segment count.
    fn live_documents(dir: &FsDirectory) -> (BTreeMap<String, String>, usize) {
        let infos = segment_infos::read_latest(dir).unwrap();
        let mut out = BTreeMap::new();
        for sci in &infos.segments {
            let name = &sci.segment_name;
            let si =
                segment_info::parse(&dir.open(&format!("{name}.si")).unwrap(), &sci.segment_id)
                    .unwrap();
            let fdt = dir.open(&format!("{name}.fdt")).unwrap();
            let fdx = dir.open(&format!("{name}.fdx")).unwrap();
            let fdm = dir.open(&format!("{name}.fdm")).unwrap();
            let reader = stored_fields::open(&fdt, &fdx, &fdm, &sci.segment_id, "").unwrap();
            let live = (sci.del_gen >= 0).then(|| {
                lucene_codecs::live_docs::parse(
                    &dir.open(&deletes::liv_file_name(name, sci.del_gen))
                        .unwrap(),
                    &sci.segment_id,
                    sci.del_gen,
                    si.doc_count as usize,
                    sci.del_count as usize,
                )
                .unwrap()
            });
            for d in 0..si.doc_count {
                if live.as_ref().is_some_and(|l| !l.get(d as usize)) {
                    continue;
                }
                let stored = reader.document(d).unwrap();
                let text = |n: i32| match &stored
                    .fields
                    .iter()
                    .find(|f| f.field_number == n)
                    .unwrap()
                    .value
                {
                    FieldValue::String(s) => s.clone(),
                    other => panic!("{other:?}"),
                };
                if let Some(previous) = out.insert(text(0), text(1)) {
                    panic!(
                        "id {} is live twice: {previous:?} and {:?} (segment {name}, doc {d})",
                        text(0),
                        text(1)
                    );
                }
            }
        }
        (out, infos.segments.len())
    }

    fn assert_clean(dir: &FsDirectory) {
        for result in crate::check_index::check_directory(dir).unwrap() {
            assert!(result.all_passed(), "{:?}", result.failures());
        }
    }

    #[test]
    fn many_threads_add_while_merges_run() {
        let tmp = TempDir::new("concurrent-add");
        let dir = FsDirectory::open(&tmp);
        let w = ConcurrentIndexWriter::new(writer(&dir, 7), 4).unwrap();
        let stop = AtomicBool::new(false);
        let merges = std::thread::scope(|scope| {
            let merger = scope.spawn(|| w.run_merges(&stop).unwrap());
            let indexers: Vec<_> = (0..4)
                .map(|t| {
                    let w = &w;
                    scope.spawn(move || {
                        for i in 0..500 {
                            w.add_document(doc(&format!("t{t}x{i}"), 0)).unwrap();
                            if i % 97 == 0 {
                                w.commit().unwrap();
                            }
                        }
                    })
                })
                .collect();
            for h in indexers {
                h.join().unwrap();
            }
            w.commit().unwrap();
            stop.store(true, Ordering::Release);
            merger.join().unwrap()
        });
        // However the merge thread was scheduled, one more round runs here.
        let merges = merges + w.maybe_merge().unwrap();
        w.commit().unwrap();
        let (docs, segments) = live_documents(&dir);
        assert_eq!(docs.len(), 2000);
        assert!(merges > 0, "nothing was merged");
        assert!(
            segments < 2000 / 7,
            "{segments} segments: nothing was merged"
        );
        assert_clean(&dir);
    }

    #[test]
    fn updates_and_deletes_from_many_threads_leave_exactly_the_last_versions() {
        let tmp = TempDir::new("concurrent-update");
        let dir = FsDirectory::open(&tmp);
        let w = ConcurrentIndexWriter::new(writer(&dir, 5), 3).unwrap();
        let stop = AtomicBool::new(false);
        // Each thread owns its ids, so the expected end state is exact: id k
        // of thread t is updated k % 4 times, and deleted when k % 5 == 0.
        std::thread::scope(|scope| {
            let merger = scope.spawn(|| w.run_merges(&stop).unwrap());
            let threads: Vec<_> = (0..3)
                .map(|t| {
                    let w = &w;
                    scope.spawn(move || {
                        for k in 0..150u32 {
                            let id = format!("t{t}x{k}");
                            w.add_document(doc(&id, 0)).unwrap();
                            for v in 1..=(k % 4) {
                                w.update_document(
                                    Term::new("id", id.clone().into_bytes()),
                                    doc(&id, v),
                                )
                                .unwrap();
                            }
                            if k % 5 == 0 {
                                w.delete_documents_by_term(&[Term::new("id", id.into_bytes())])
                                    .unwrap();
                            }
                            if k % 41 == 0 {
                                w.commit().unwrap();
                            }
                        }
                    })
                })
                .collect();
            for h in threads {
                h.join().unwrap();
            }
            w.commit().unwrap();
            stop.store(true, Ordering::Release);
            merger.join().unwrap();
        });
        w.commit().unwrap();
        let (docs, _) = live_documents(&dir);
        let mut expected = BTreeMap::new();
        for t in 0..3 {
            for k in 0..150u32 {
                if k % 5 != 0 {
                    expected.insert(format!("t{t}x{k}"), format!("v{} common", k % 4));
                }
            }
        }
        assert_eq!(docs, expected);
        assert_clean(&dir);
    }

    #[test]
    fn a_delete_reaches_a_segment_that_was_being_built_when_it_was_issued() {
        // Two slots: slot A's buffer is ticketed (taken for a segment) before
        // the delete, and published after it. The delete must still reach it.
        let tmp = TempDir::new("concurrent-inflight");
        let dir = FsDirectory::open(&tmp);
        let w = ConcurrentIndexWriter::new(writer(&dir, 1000), 2).unwrap();
        w.add_document(doc("a", 0)).unwrap();
        let (i, mut dwpt) = w.acquire_slot();
        let batch = w.begin_flush(i, &mut dwpt);
        drop(dwpt);
        let batch = batch.or_else(|| {
            // The document went to the other slot.
            let j = 1 - i;
            let mut other = lock(&w.slots[j]);
            w.begin_flush(j, &mut other)
        });
        let batch = batch.expect("one slot held the document");
        w.delete_documents_by_term(&[Term::new("id", b"a".to_vec())])
            .unwrap();
        w.complete_flush(batch).unwrap();
        w.commit().unwrap();
        let (docs, _) = live_documents(&dir);
        assert!(docs.is_empty(), "{docs:?}");
        assert_clean(&dir);
    }

    /// A commit is a clean cut through concurrent updates: an update's delete
    /// and its new document are both in a commit or both out of it, so every
    /// id is live exactly once in every commit.
    #[test]
    fn every_commit_holds_each_updated_id_exactly_once() {
        let tmp = TempDir::new("concurrent-cut");
        let dir = FsDirectory::open(&tmp);
        let w = ConcurrentIndexWriter::new(writer(&dir, 3), 3).unwrap();
        for k in 0..30 {
            w.add_document(doc(&format!("u{k}"), 0)).unwrap();
        }
        w.commit().unwrap();
        let stop = AtomicBool::new(false);
        std::thread::scope(|scope| {
            for t in 0..2u32 {
                let (w, stop) = (&w, &stop);
                scope.spawn(move || {
                    let mut v = 1;
                    while !stop.load(Ordering::Acquire) {
                        for k in (t..30).step_by(2) {
                            let id = format!("u{k}");
                            w.update_document(
                                Term::new("id", id.clone().into_bytes()),
                                doc(&id, v),
                            )
                            .unwrap();
                        }
                        v += 1;
                    }
                });
            }
            // Stops the updaters however this loop ends, so a failed
            // assertion fails the test instead of hanging it.
            let _stop = StopOnDrop(&stop);
            for _ in 0..40 {
                w.commit().unwrap();
                // Panics on an id live twice.
                let (docs, _) = live_documents(&dir);
                assert_eq!(docs.len(), 30, "an update was cut in half: {docs:?}");
            }
        });
        w.commit().unwrap();
        assert_clean(&dir);
    }

    struct StopOnDrop<'a>(&'a AtomicBool);

    impl Drop for StopOnDrop<'_> {
        fn drop(&mut self) {
            self.0.store(true, Ordering::Release);
        }
    }

    /// Two segments in flight, a delete issued between their tickets: the
    /// first segment gets it only through the *second* ticket's frozen
    /// packet, pushed when that one publishes -- after the first, although
    /// the second finished building first.
    #[test]
    fn a_delete_between_two_tickets_reaches_the_first_through_the_second() {
        let tmp = TempDir::new("concurrent-two-tickets");
        let dir = FsDirectory::open(&tmp);
        let w = ConcurrentIndexWriter::new(writer(&dir, 1000), 2).unwrap();
        lock(&w.slots[0]).docs.push(doc("a", 0));
        lock(&w.slots[1]).docs.push(doc("b", 0));
        let first = w.begin_flush(0, &mut lock(&w.slots[0])).unwrap();
        w.delete_documents_by_term(&[Term::new("id", b"a".to_vec())])
            .unwrap();
        w.delete_documents_by_term(&[Term::new("id", b"b".to_vec())])
            .unwrap();
        let second = w.begin_flush(1, &mut lock(&w.slots[1])).unwrap();
        std::thread::scope(|scope| {
            let later = scope.spawn(|| w.complete_flush(second));
            // Let the second build finish and wait for its turn.
            std::thread::sleep(Duration::from_millis(50));
            w.complete_flush(first).unwrap();
            later.join().unwrap().unwrap();
        });
        // Published, not yet committed: both deletes already applied.
        {
            let core = lock(&w.core);
            let live = core.writer.live_infos();
            assert!(
                live.segments.is_empty(),
                "both segments are fully deleted and dropped"
            );
        }
        w.add_document(doc("c", 0)).unwrap();
        w.commit().unwrap();
        let (docs, _) = live_documents(&dir);
        assert_eq!(docs.keys().collect::<Vec<_>>(), ["c"]);
        assert_clean(&dir);
    }

    fn two_committed_segments(dir: &FsDirectory) -> ConcurrentIndexWriter<'_> {
        let mut single = sorted_writer(dir);
        single.set_index_sort(None).unwrap();
        single.set_merge_policy(Some(MergePolicyConfig {
            max_merge_at_once: 2,
            segments_per_tier: 2,
            floor_segment_size: 1 << 30,
            ..MergePolicyConfig::default()
        }));
        let w = ConcurrentIndexWriter::new(single, 1).unwrap();
        for (id, rank) in [("m0", 1), ("m1", 2)] {
            w.add_document(ranked(id, 0, rank)).unwrap();
        }
        w.commit().unwrap();
        for (id, rank) in [("m2", 3), ("m3", 4)] {
            w.add_document(ranked(id, 0, rank)).unwrap();
        }
        w.commit().unwrap();
        w.add_document(ranked("m4", 0, 5)).unwrap();
        w.commit().unwrap();
        w
    }

    /// `commitMergedDeletesAndUpdates`: a delete published on a source while
    /// the merge ran is carried onto the merged segment through the merge's
    /// doc map.
    #[test]
    fn a_delete_made_during_a_merge_is_carried_onto_the_merged_segment() {
        let tmp = TempDir::new("concurrent-merge-carry");
        let dir = FsDirectory::open(&tmp);
        let w = two_committed_segments(&dir);
        let plan = {
            let mut core = lock(&w.core);
            let names = core
                .writer
                .next_merge(w.merge_policy.as_ref().unwrap(), &HashSet::new())
                .unwrap()
                .unwrap();
            core.writer.begin_merge(&names).unwrap()
        };
        let outcome = w.cfg.run_merge(&dir, &plan).unwrap();
        // Meanwhile: deletes reach the first two segments -- at least one of
        // them a source, whichever the policy chose -- and are published
        // there. (Emptying a source would abandon the merge instead.)
        let sources = plan.names.len();
        for id in ["m0", "m2"] {
            w.delete_documents_by_term(&[Term::new("id", id.as_bytes().to_vec())])
                .unwrap();
        }
        w.flush().unwrap();
        let generation = segment_infos::read_latest(&dir).unwrap().generation;
        assert!(lock(&w.core).writer.finish_merge(plan, outcome).unwrap());
        // `commitMerge`: published in memory; nothing durable changes until
        // the next commit.
        assert_eq!(
            segment_infos::read_latest(&dir).unwrap().generation,
            generation
        );
        w.commit().unwrap();
        let (docs, segments) = live_documents(&dir);
        assert!(segments <= 3 - sources + 1, "{segments} segments");
        assert_eq!(docs.keys().collect::<Vec<_>>(), ["m1", "m3", "m4"]);
        assert_clean(&dir);
    }

    /// A doc-values update on a source while its merge ran is not carried:
    /// the merge is abandoned, its files deleted, the sources kept.
    #[test]
    fn a_doc_values_update_during_a_merge_abandons_it() {
        let tmp = TempDir::new("concurrent-merge-abandon");
        let dir = FsDirectory::open(&tmp);
        let w = two_committed_segments(&dir);
        let plan = {
            let mut core = lock(&w.core);
            let names = core
                .writer
                .next_merge(w.merge_policy.as_ref().unwrap(), &HashSet::new())
                .unwrap()
                .unwrap();
            core.writer.begin_merge(&names).unwrap()
        };
        let merged_name = plan.merged_name.clone();
        let outcome = w.cfg.run_merge(&dir, &plan).unwrap();
        for id in ["m0", "m2", "m4"] {
            w.update_numeric_doc_value(Term::new("id", id.as_bytes().to_vec()), "score", 7)
                .unwrap();
        }
        w.flush().unwrap();
        assert!(!lock(&w.core).writer.finish_merge(plan, outcome).unwrap());
        let leftovers: Vec<String> = dir
            .list_all()
            .unwrap()
            .into_iter()
            .filter(|f| {
                f.starts_with(&format!("{merged_name}."))
                    || f.starts_with(&format!("{merged_name}_"))
            })
            .collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");
        w.commit().unwrap();
        let (docs, segments) = live_documents(&dir);
        assert_eq!(segments, 3);
        assert_eq!(docs.len(), 5);
        assert_clean(&dir);
    }

    /// A writer over `id`, `body`, `rank` and `score` (NUMERIC doc values),
    /// sorted by `rank` ascending, so a flush permutes its buffer and every
    /// private delete's limit has to go through the sort map.
    fn sorted_writer(dir: &FsDirectory) -> IndexWriter<'_> {
        let mut fields = fields();
        fields.push(FieldInfo {
            doc_values_type: lucene_codecs::field_infos::DocValuesType::Numeric,
            ..FieldInfo::new("rank", 2)
        });
        fields.push(FieldInfo {
            doc_values_type: lucene_codecs::field_infos::DocValuesType::Numeric,
            ..FieldInfo::new("score", 3)
        });
        let mut w = IndexWriter::open(
            dir,
            fields,
            "Lucene104",
            LuceneVersion {
                major: 10,
                minor: 5,
                bugfix: 0,
            },
        )
        .unwrap();
        w.set_postings_field(Some("id")).unwrap();
        w.add_postings_field("body").unwrap();
        w.set_doc_values_field(Some("rank")).unwrap();
        w.add_doc_values_field("score").unwrap();
        w.set_index_sort(Some(&[segment_info::IndexSortField::long(
            "rank",
            false,
            Some(i64::MAX),
        )]))
        .unwrap();
        w.set_max_buffered_docs(1000).unwrap();
        w.set_ram_buffer_size_mb(DISABLE_AUTO_FLUSH_MB).unwrap();
        w
    }

    fn ranked(id: &str, version: u32, rank: i64) -> Document {
        let mut d = doc(id, version);
        d.fields.push(StoredField {
            field_number: 2,
            value: FieldValue::Long(rank),
        });
        d.fields.push(StoredField {
            field_number: 3,
            value: FieldValue::Long(rank * 100),
        });
        d
    }

    /// A buffer's own deletes of every kind, resolved by the flushing thread
    /// before the publish: term deletes against the terms just inverted,
    /// query deletes and doc-values updates against the written segment --
    /// all through the sort map, since the segment is sorted by `rank`.
    #[test]
    fn a_flush_resolves_its_own_deletes_and_updates_through_the_sort() {
        let tmp = TempDir::new("concurrent-sorted");
        let dir = FsDirectory::open(&tmp);
        let w = ConcurrentIndexWriter::new(sorted_writer(&dir), 1).unwrap();
        w.add_document(ranked("a", 0, 3)).unwrap();
        w.add_document(ranked("b", 0, 1)).unwrap();
        w.add_document(ranked("c", 0, 2)).unwrap();
        w.delete_documents_by_query(&[DeleteQuery::Term(Term::new("id", b"a".to_vec()))])
            .unwrap();
        w.update_numeric_doc_value(Term::new("id", b"b".to_vec()), "score", 10)
            .unwrap();
        w.update_document(Term::new("id", b"c".to_vec()), ranked("c", 1, 0))
            .unwrap();
        // After the delete: survives it.
        w.add_document(ranked("a", 1, 5)).unwrap();
        w.commit().unwrap();

        let (docs, segments) = live_documents(&dir);
        assert_eq!(segments, 1);
        let want: BTreeMap<String, String> = [("a", 1), ("b", 0), ("c", 1)]
            .into_iter()
            .map(|(id, v)| (id.to_string(), format!("v{v} common")))
            .collect();
        assert_eq!(docs, want);
        let infos = segment_infos::read_latest(&dir).unwrap();
        let scores = crate::index_writer::tests::read_numeric_column(&dir, &infos.segments[0], 3);
        // Sorted by rank (c1=0, b=1, c0=2, a0=3, a1=5), each score 100 x its
        // rank; only b's changed.
        assert_eq!(scores, [Some(0), Some(10), Some(200), Some(300), Some(500)]);
        assert_clean(&dir);
    }

    /// `DocumentsWriterPerThread.abort`: a buffer whose segment cannot be
    /// built loses its documents and leaves no file behind, the published
    /// segments still get the deletes its ticket froze, and the writer goes
    /// on.
    #[test]
    fn a_failed_build_is_abandoned_and_the_writer_goes_on() {
        let tmp = TempDir::new("concurrent-abort");
        let dir = FsDirectory::open(&tmp);
        let mut single = sorted_writer(&dir);
        single.set_merge_policy(Some(MergePolicyConfig {
            max_merge_at_once: 2,
            segments_per_tier: 2,
            floor_segment_size: 1 << 30,
            ..MergePolicyConfig::default()
        }));
        single.set_max_buffered_docs(2).unwrap();
        let w = ConcurrentIndexWriter::new(single, 1).unwrap();
        w.add_document(ranked("old", 0, 1)).unwrap();
        w.add_document(ranked("kept", 0, 1)).unwrap();
        w.commit().unwrap();
        w.add_document(ranked("also", 0, 1)).unwrap();
        w.commit().unwrap();
        // The delete is frozen into the next ticket, whose build then fails:
        // a block in a sorted index with no parent field fills the buffer
        // (`max_buffered_docs` is 2) and the automatic flush refuses it.
        w.delete_documents_by_term(&[Term::new("id", b"old".to_vec())])
            .unwrap();
        assert!(matches!(
            w.add_documents(vec![ranked("p", 0, 2), ranked("q", 0, 3)]),
            Err(Error::IndexSortWithBlocksAndNoParentField)
        ));
        let failed: Vec<String> = dir
            .list_all()
            .unwrap()
            .into_iter()
            .filter(|f| f.starts_with("_2"))
            .collect();
        assert!(failed.is_empty(), "the failed build left {failed:?}");
        // Nothing is left pending for a merge's segment to take by mistake
        // (`begin_merge` asserts the stream is empty).
        assert_eq!(w.maybe_merge().unwrap(), 1);
        w.add_document(ranked("r", 0, 4)).unwrap();
        w.commit().unwrap();
        let (docs, _) = live_documents(&dir);
        assert_eq!(docs.keys().collect::<Vec<_>>(), ["also", "kept", "r"]);
        assert_clean(&dir);
    }

    /// A directory whose `create_output` waits while its gate is shut: holds
    /// a flush in the middle of its build.
    struct GatedDirectory<'a> {
        inner: &'a FsDirectory,
        open: Mutex<bool>,
        opened: Condvar,
    }

    impl GatedDirectory<'_> {
        fn set_open(&self, open: bool) {
            *lock(&self.open) = open;
            self.opened.notify_all();
        }
    }

    impl Directory for GatedDirectory<'_> {
        fn list_all(&self) -> lucene_store::Result<Vec<String>> {
            self.inner.list_all()
        }
        fn open(&self, name: &str) -> lucene_store::Result<Input> {
            self.inner.open(name)
        }
        fn create_output(&self, name: &str) -> lucene_store::Result<FsIndexOutput> {
            let mut open = lock(&self.open);
            while !*open {
                open = self.opened.wait(open).unwrap();
            }
            drop(open);
            self.inner.create_output(name)
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
    }

    /// Waits (bounded) until `cond` holds.
    fn eventually(what: &str, cond: impl Fn() -> bool) {
        let start = std::time::Instant::now();
        while !cond() {
            assert!(
                start.elapsed() < std::time::Duration::from_secs(20),
                "never: {what}"
            );
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    /// `DocumentsWriterFlushControl.updateStallState` +
    /// `DocumentsWriter.preUpdate`: while flushes are held mid-build and the
    /// slots keep filling, RAM in slots and flushes passes twice the buffer
    /// and indexing stalls; the other threads' adds wait, and go on once the
    /// flushes complete and lift the stall. No document is lost.
    ///
    /// Each flush is taken by the thread whose add reached the RAM buffer
    /// (the largest slot, `FlushByRamOrCountsPolicy`) and holds that thread
    /// at the gate, so three held flushes of about four documents each plus
    /// nine documents in the slots pass the twenty-document stall limit
    /// while the slots alone stay under it.
    #[test]
    fn indexing_stalls_while_flushing_falls_behind() {
        let tmp = TempDir::new("concurrent-stall");
        let fs = FsDirectory::open(&tmp);
        let gated = GatedDirectory {
            inner: &fs,
            open: Mutex::new(true),
            opened: Condvar::new(),
        };
        let version = LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        };
        let mut single = IndexWriter::open(&gated, fields(), "Lucene104", version).unwrap();
        single.set_postings_field(Some("id")).unwrap();
        single.add_postings_field("body").unwrap();
        single.set_max_buffered_docs(DISABLE_AUTO_FLUSH).unwrap();
        // Ten documents fill a slot.
        let doc_bytes = document_ram_bytes(&doc("t1x00", 0));
        let mb = (doc_bytes * 10) as f64 / (1024.0 * 1024.0);
        single.set_ram_buffer_size_mb(mb).unwrap();
        let w = ConcurrentIndexWriter::new(single, 3).unwrap();
        assert!(w.stall_control().is_healthy());

        gated.set_open(false);
        /// Opens the gate however the scope ends: a failed assertion must
        /// not leave a thread blocked in `create_output` for the scope to
        /// wait on forever.
        struct OpenOnDrop<'a, 'b>(&'a GatedDirectory<'b>);
        impl Drop for OpenOnDrop<'_, '_> {
            fn drop(&mut self) {
                self.0.set_open(true);
            }
        }
        std::thread::scope(|scope| {
            let _open = OpenOnDrop(&gated);
            let fillers: Vec<_> = (0..4)
                .map(|t| {
                    let w = &w;
                    scope.spawn(move || {
                        for k in 0..30 {
                            w.add_document(doc(&format!("t{t}x{k:02}"), 0)).unwrap();
                        }
                    })
                })
                .collect();
            eventually("the held flushes stall indexing", || {
                w.stall_control().any_stalled_threads()
            });
            eventually("a thread waits on the stall", || {
                w.stall_control().has_blocked()
            });
            assert!(fillers.iter().any(|f| !f.is_finished()));
            gated.set_open(true);
            for f in fillers {
                f.join().unwrap();
            }
        });
        assert!(w.stall_control().was_stalled());
        assert!(
            w.stall_control().is_healthy(),
            "the flushes lifted the stall"
        );
        w.commit().unwrap();
        let (docs, _) = live_documents(&fs);
        assert_eq!(docs.len(), 120);
    }

    /// A concurrent writer flushing by RAM over three slots, the buffer
    /// `docs` documents of `doc("t1x00", 0)`'s size, and no document limit.
    fn ram_writer(dir: &FsDirectory, docs: usize) -> ConcurrentIndexWriter<'_> {
        let mut single = writer(dir, 1_000_000);
        let doc_bytes = document_ram_bytes(&doc("t1x00", 0));
        single
            .set_ram_buffer_size_mb((doc_bytes * docs) as f64 / (1024.0 * 1024.0))
            .unwrap();
        single.set_max_buffered_docs(DISABLE_AUTO_FLUSH).unwrap();
        ConcurrentIndexWriter::new(single, 3).unwrap()
    }

    /// `FlushByRamOrCountsPolicy.markLargestWriterPending`: the RAM buffer
    /// bounds every slot together, and reaching it flushes the slot holding
    /// the most, not the one the last document went into. One thread fills
    /// the slots in turn (4, 3, 3 documents): the tenth document fills the
    /// buffer and the first slot, the largest, is flushed.
    #[test]
    fn the_ram_buffer_bounds_all_slots_and_flushes_the_largest() {
        let tmp = TempDir::new("concurrent-largest");
        let dir = FsDirectory::open(&tmp);
        let w = ram_writer(&dir, 10);
        for k in 0..9 {
            w.add_document(doc(&format!("t1x{k:02}"), 0)).unwrap();
        }
        assert_eq!(w.pending_doc_count(), 9);
        w.add_document(doc("t1x09", 0)).unwrap();
        assert_eq!(w.pending_doc_count(), 6, "the four-document slot flushed");
        let counts: Vec<usize> = w.slots.iter().map(|s| lock(s).docs.len()).collect();
        assert_eq!(counts, [0, 3, 3]);
        let ram = lock(&w.ram);
        assert_eq!(ram.flush_bytes(), 0, "the flush is done");
        assert_eq!(ram.num_pending(), 0);
    }

    /// `FlushByRamOrCountsPolicy`: a marked slot the add did not go into is
    /// flushed by the adding thread (`postUpdate`'s `maybeFlush`) while
    /// `checkPendingFlushOnUpdate` is on; with it off the slot stays marked
    /// until a thread indexes into it.
    #[test]
    fn a_marked_slot_is_flushed_by_whoever_checks_pending_flushes() {
        for check in [true, false] {
            let tmp = TempDir::new("concurrent-pending");
            let dir = FsDirectory::open(&tmp);
            let w = ram_writer(&dir, 10);
            w.set_check_pending_flush_on_update(check);
            let big = Document {
                fields: vec![
                    StoredField {
                        field_number: 0,
                        value: FieldValue::String("big".into()),
                    },
                    StoredField {
                        field_number: 1,
                        value: FieldValue::String(
                            "x".repeat(document_ram_bytes(&doc("t1x00", 0)) * 15 / 2),
                        ),
                    },
                ],
            };
            // Slots 0, 1, 2 in turn; the big document (about 8.5 small ones)
            // is the largest, and the buffer fills on the third add, into
            // slot 2.
            w.add_document(doc("a", 0)).unwrap();
            w.add_document(big.clone()).unwrap();
            w.add_document(doc("c", 0)).unwrap();
            let counts: Vec<usize> = w.slots.iter().map(|s| lock(s).docs.len()).collect();
            if check {
                assert_eq!(counts, [1, 0, 1], "the adding thread flushed slot 1");
                assert!(!w.any_pending());
            } else {
                assert_eq!(counts, [1, 1, 1], "slot 1 waits, marked");
                assert!(w.pending[1].load(Ordering::Acquire));
                // Slot 0 next: the marked RAM no longer counts, so nothing
                // more is marked, and slot 1 still waits.
                w.add_document(doc("d", 0)).unwrap();
                let counts: Vec<usize> = w.slots.iter().map(|s| lock(s).docs.len()).collect();
                assert_eq!(counts, [2, 1, 1]);
                assert!(w.pending[1].load(Ordering::Acquire));
                // Into slot 1: the thread indexing there flushes it.
                w.add_document(doc("e", 0)).unwrap();
                let counts: Vec<usize> = w.slots.iter().map(|s| lock(s).docs.len()).collect();
                assert_eq!(counts, [2, 0, 1]);
                assert!(!w.any_pending());
            }
            w.commit().unwrap();
            let (docs, _) = live_documents(&dir);
            assert_eq!(docs.len(), if check { 3 } else { 5 });
        }
    }

    /// `FlushByRamOrCountsPolicy.flushDeletes` + `DocumentsWriter
    /// .applyAllDeletes`: once buffered deletes alone reach the RAM buffer
    /// they are applied to the published segments at once -- their `.liv`
    /// written before any flush or commit -- and stop counting.
    #[test]
    fn deletes_over_the_ram_buffer_are_applied_without_a_flush() {
        let tmp = TempDir::new("concurrent-delete-ram");
        let dir = FsDirectory::open(&tmp);
        let mut single = writer(&dir, 1_000_000);
        for k in 0..20 {
            single.add_document(doc(&format!("d{k:02}"), 0)).unwrap();
        }
        single.commit().unwrap();
        let node = DeleteNode::terms(vec![Term::new("id", "d00")]);
        // Three one-term deletes fill the buffer.
        single
            .set_ram_buffer_size_mb((node.ram_bytes() * 3) as f64 / (1024.0 * 1024.0))
            .unwrap();
        single.set_max_buffered_docs(DISABLE_AUTO_FLUSH).unwrap();
        let w = ConcurrentIndexWriter::new(single, 2).unwrap();
        let liv = || dir.list_all().unwrap().iter().any(|f| f.ends_with(".liv"));
        w.delete_documents_by_term(&[Term::new("id", "d00")])
            .unwrap();
        w.delete_documents_by_term(&[Term::new("id", "d01")])
            .unwrap();
        assert!(!liv(), "two deletes are under the buffer");
        assert_eq!(lock(&w.log).unhanded_bytes, node.ram_bytes() * 2);
        w.delete_documents_by_term(&[Term::new("id", "d02")])
            .unwrap();
        assert!(liv(), "the third applied all three");
        assert_eq!(lock(&w.log).unhanded_bytes, 0);
        assert_eq!(lock(&w.ram).delete_bytes_used(), 0);
        w.delete_documents_by_term(&[Term::new("id", "d03")])
            .unwrap();
        w.commit().unwrap();
        let (docs, _) = live_documents(&dir);
        assert_eq!(docs.len(), 16);
        assert!(!docs.contains_key("d02") && docs.contains_key("d04"));
    }

    /// With no RAM buffer (flushing by document count) nothing ever stalls.
    #[test]
    fn indexing_never_stalls_without_a_ram_buffer() {
        let tmp = TempDir::new("concurrent-no-stall");
        let dir = FsDirectory::open(&tmp);
        let w = ConcurrentIndexWriter::new(writer(&dir, 2), 2).unwrap();
        for k in 0..20 {
            w.add_document(doc(&format!("n{k}"), 0)).unwrap();
        }
        w.commit().unwrap();
        assert!(!w.stall_control().was_stalled());
    }

    /// `DocumentsWriterPerThread.flush`: a buffer its own deletes leave empty
    /// never becomes a published segment.
    #[test]
    fn a_buffer_its_own_deletes_empty_is_not_published() {
        let tmp = TempDir::new("concurrent-empty");
        let dir = FsDirectory::open(&tmp);
        let w = ConcurrentIndexWriter::new(writer(&dir, 1000), 1).unwrap();
        w.add_document(doc("x", 0)).unwrap();
        w.delete_documents_by_term(&[Term::new("id", b"x".to_vec())])
            .unwrap();
        w.commit().unwrap();
        let (docs, segments) = live_documents(&dir);
        assert!(docs.is_empty());
        assert_eq!(segments, 0);
        let leftovers: Vec<String> = dir
            .list_all()
            .unwrap()
            .into_iter()
            .filter(|f| f.starts_with("_0"))
            .collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");
    }

    /// Updates spread over several slots from one thread: each update's
    /// delete has to reach the slot the old version went to.
    #[test]
    fn one_thread_updating_across_slots_leaves_the_last_versions() {
        let tmp = TempDir::new("concurrent-repro");
        let dir = FsDirectory::open(&tmp);
        let w = ConcurrentIndexWriter::new(writer(&dir, 5), 3).unwrap();
        let mut expected = BTreeMap::new();
        for k in 0..40u32 {
            let id = format!("k{k}");
            w.add_document(doc(&id, 0)).unwrap();
            for v in 1..=(k % 4) {
                w.update_document(Term::new("id", id.clone().into_bytes()), doc(&id, v))
                    .unwrap();
            }
            expected.insert(id, format!("v{} common", k % 4));
        }
        w.commit().unwrap();
        let (docs, _) = live_documents(&dir);
        assert_eq!(docs, expected);
        assert_clean(&dir);
    }

    #[test]
    fn a_delete_does_not_reach_a_document_added_after_it() {
        let tmp = TempDir::new("concurrent-order");
        let dir = FsDirectory::open(&tmp);
        let w = ConcurrentIndexWriter::new(writer(&dir, 1000), 2).unwrap();
        w.add_document(doc("x", 0)).unwrap();
        w.delete_documents_by_term(&[Term::new("id", b"x".to_vec())])
            .unwrap();
        let added = w.add_document(doc("x", 1)).unwrap();
        let committed = w.commit().unwrap();
        // `commit` names the last operation it holds.
        assert_eq!(committed, added);
        assert!(w.add_document(doc("y", 0)).unwrap() > committed);
        let (docs, _) = live_documents(&dir);
        assert_eq!(
            docs.into_iter().collect::<Vec<_>>(),
            [("x".to_string(), "v1 common".to_string())]
        );
        assert_clean(&dir);
    }

    /// `IndexWriter.deleteAll()` on the concurrent writer: committed,
    /// flushed and buffered documents all go, deletes buffered before it
    /// reach nothing added after it, a `MatchAllDocsQuery` delete is the
    /// same operation, and a merge the scheduler had not started is dropped.
    #[test]
    fn delete_all_drops_every_segment_buffer_and_delete() {
        let tmp = TempDir::new("concurrent-delete-all");
        let dir = FsDirectory::open(&tmp);
        let w = ConcurrentIndexWriter::new(writer(&dir, 3), 2).unwrap();
        for k in 0..10 {
            w.add_document(doc(&format!("a{k}"), 0)).unwrap();
        }
        w.commit().unwrap();
        for k in 0..5 {
            w.add_document(doc(&format!("b{k}"), 0)).unwrap();
        }
        w.delete_documents_by_term(&[Term::new("id", "c0")])
            .unwrap();
        assert!(w.pending_doc_count() > 0);
        let before = w.add_document(doc("gone", 0)).unwrap();
        let seq = w.delete_all().unwrap();
        assert!(seq > before);
        assert_eq!(w.pending_doc_count(), 0);
        assert_eq!(lock(&w.log).nodes.len(), 0);
        // Not durable yet.
        assert_eq!(live_documents(&dir).0.len(), 10);
        // "c0" was deleted before the delete-all: the new one survives.
        w.add_document(doc("c0", 0)).unwrap();
        w.add_document(doc("c1", 0)).unwrap();
        w.commit().unwrap();
        let (docs, _) = live_documents(&dir);
        assert_eq!(docs.keys().collect::<Vec<_>>(), ["c0", "c1"]);
        assert_clean(&dir);

        // A merge registered for a scheduler but not taken is dropped.
        lock(&w.core).merging.insert("_zz".into());
        w.pending_merges
            .push(Arc::new(ScheduledMerge::new(vec!["_zz".into()], 0)));
        let seq2 = w
            .delete_documents_by_query(&[DeleteQuery::MatchAll])
            .unwrap();
        assert!(seq2 > seq);
        assert!(w.pending_merges.is_empty());
        assert!(lock(&w.core).merging.is_empty());
        w.commit().unwrap();
        assert!(live_documents(&dir).0.is_empty());
        assert_clean(&dir);
    }

    /// Handing the writer back: pending work, a delete not yet applied, and
    /// sequence numbers that keep climbing across both hand-overs.
    #[test]
    fn into_writer_hands_back_pending_work_and_the_numbering() {
        let tmp = TempDir::new("concurrent-into");
        let dir = FsDirectory::open(&tmp);
        let mut single = writer(&dir, 1000);
        let before = single.add_document(doc("q", 0)).unwrap();
        let w = ConcurrentIndexWriter::new(single, 2).unwrap();
        let during = w.add_document(doc("p", 0)).unwrap();
        assert!(during > before, "{during} after {before}");
        w.delete_documents_by_term(&[Term::new("id", b"q".to_vec())])
            .unwrap();
        assert_eq!(w.pending_doc_count(), 1);
        let mut single = w.into_writer().unwrap();
        let after = single.add_document(doc("r", 0)).unwrap();
        assert!(after > during + 1, "{after} after {during} and a delete");
        single.commit().unwrap();
        let (docs, _) = live_documents(&dir);
        assert_eq!(docs.keys().collect::<Vec<_>>(), ["p", "r"]);
        assert_clean(&dir);
    }

    /// **Merges and flushes between commits leave no per-segment state
    /// behind.** The concurrent writer merges and flushes without stamping a
    /// `segments_N`, so the writer's version cache had only a commit to prune
    /// it: it held every segment each merge retired or each delete emptied
    /// until then. Here one commit comes every 50 rounds, and every round must
    /// hold the cache to the writer's view, and the deleter and the directory
    /// to the view plus the last commit.
    #[test]
    fn merges_between_commits_keep_per_segment_state_to_the_view() {
        let tmp = TempDir::new("concurrent-resource-bounds");
        let dir = FsDirectory::open(&tmp);
        let w = ConcurrentIndexWriter::new(writer(&dir, 5), 1).unwrap();
        let mut merges = 0;
        for round in 0..300u32 {
            for k in 0..10u32 {
                let id = format!("{}", (round * 10 + k) % 400);
                w.update_document(Term::new("id", id.clone().into_bytes()), doc(&id, round))
                    .unwrap();
            }
            merges += w.maybe_merge().unwrap();
            if round % 50 == 49 {
                w.commit().unwrap();
            }
            let (view, versions, tracked) = lock(&w.core).writer.per_segment_sizes();
            assert!(
                versions <= view,
                "round {round}: {versions} versions for {view} segments"
            );
            // A caller-driven merge keeps its sources' files while the last
            // commit names them; the next commit releases them. So the bound
            // is the view plus that commit -- not every segment ever written.
            let committed = segment_infos::read_latest(&dir).map_or(0, |i| i.segments.len());
            assert!(
                tracked <= view + committed,
                "round {round}: {tracked} file sets for {view} segments + {committed} committed"
            );
            let on_disk = std::fs::read_dir(tmp.path())
                .unwrap()
                .filter(|e| {
                    e.as_ref()
                        .unwrap()
                        .file_name()
                        .to_string_lossy()
                        .ends_with(".si")
                })
                .count();
            assert!(
                on_disk <= view + committed,
                "round {round}: {on_disk} segments on disk for {view} + {committed} committed"
            );
        }
        assert!(merges > 50, "the policy must have merged: {merges}");
        w.commit().unwrap();
        assert_clean(&dir);
    }

    /// A directory that lives for the rest of the test process -- what a
    /// merge scheduler's threads need to hold the writer (the directory
    /// itself is removed by its `TempDir` guard).
    fn static_dir(tmp: &TempDir) -> &'static FsDirectory {
        Box::leak(Box::new(FsDirectory::open(tmp)))
    }

    /// `ConcurrentMergeScheduler` behind the writer: merges are registered
    /// after flushes and commits and run on merge threads while four threads
    /// index; nothing here runs a merge itself. After `close_merges` the
    /// writer can be taken back out of its `Arc`.
    #[test]
    fn a_concurrent_merge_scheduler_merges_while_threads_index() {
        let tmp = TempDir::new("concurrent-cms");
        let dir = static_dir(&tmp);
        let cms = crate::merge_scheduler::ConcurrentMergeScheduler::new();
        cms.set_max_merges_and_threads(3, 2).unwrap();
        let w =
            ConcurrentIndexWriter::with_merge_scheduler(writer(dir, 7), 4, Arc::new(cms.clone()))
                .unwrap();
        std::thread::scope(|scope| {
            for t in 0..4 {
                let w = &w;
                scope.spawn(move || {
                    for i in 0..400 {
                        w.add_document(doc(&format!("t{t}x{i}"), 0)).unwrap();
                        if i % 61 == 0 {
                            w.commit().unwrap();
                        }
                    }
                });
            }
        });
        w.commit().unwrap();
        assert_eq!(w.maybe_merge().unwrap(), 0, "merges are the scheduler's");
        w.close_merges().unwrap();
        assert_eq!(cms.merge_thread_count(), 0);
        w.commit().unwrap();
        let (docs, segments) = live_documents(dir);
        assert_eq!(docs.len(), 1600);
        assert!(
            segments < 1600 / 7 / 2,
            "{segments} segments: the scheduler merged too little"
        );
        assert_clean(dir);
        let w = Arc::try_unwrap(w)
            .ok()
            .expect("no merge thread still holds the writer");
        let mut single = w.into_writer().unwrap();
        single.commit().unwrap();
    }

    /// `SerialMergeScheduler` behind the writer: a commit's merges run on the
    /// committing thread before `commit` returns.
    #[test]
    fn a_serial_merge_scheduler_merges_on_the_committing_thread() {
        let tmp = TempDir::new("concurrent-serial");
        let dir = static_dir(&tmp);
        let w = ConcurrentIndexWriter::with_merge_scheduler(
            writer(dir, 5),
            1,
            Arc::new(crate::merge_scheduler::SerialMergeScheduler::new()),
        )
        .unwrap();
        for i in 0..60 {
            w.add_document(doc(&format!("d{i}"), 0)).unwrap();
            if i % 5 == 4 {
                w.commit().unwrap();
            }
        }
        // Twelve flushed segments, merged by the commits themselves.
        let merged = lock(&w.core).writer.segment_infos().segments.len();
        assert!(merged < 12, "{merged} segments: nothing was merged");
        w.close_merges().unwrap();
        w.commit().unwrap();
        assert_eq!(live_documents(dir).0.len(), 60);
        assert_clean(dir);

        // Without a policy, a scheduler has nothing to do.
        let tmp2 = TempDir::new("concurrent-serial-nopolicy");
        let dir2 = static_dir(&tmp2);
        let mut plain = writer(dir2, 5);
        plain.set_merge_policy(None);
        let w = ConcurrentIndexWriter::with_merge_scheduler(
            plain,
            1,
            Arc::new(crate::merge_scheduler::SerialMergeScheduler::new()),
        )
        .unwrap();
        for i in 0..20 {
            w.add_document(doc(&format!("d{i}"), 0)).unwrap();
        }
        w.commit().unwrap();
        w.flush().unwrap();
        assert_eq!(w.maybe_merge().unwrap(), 0);
        w.close_merges().unwrap();
        assert_eq!(lock(&w.core).writer.segment_infos().segments.len(), 4);
    }

    /// `onMergeFinished` releases a merge the scheduler took but will not
    /// run, and a writer with no scheduler has nothing to close.
    #[test]
    fn a_merge_the_scheduler_gives_back_releases_its_segments() {
        let tmp = TempDir::new("concurrent-sched-release");
        let dir = static_dir(&tmp);
        let w = ConcurrentIndexWriter::with_merge_scheduler(
            writer(dir, 5),
            1,
            Arc::new(crate::merge_scheduler::NoMergeScheduler),
        )
        .unwrap();
        for i in 0..40 {
            w.add_document(doc(&format!("d{i}"), 0)).unwrap();
            if i % 5 == 4 {
                w.commit().unwrap();
            }
        }
        // `NoMergeScheduler` leaves what the commits registered pending.
        assert!(w.has_pending_merges());
        let merge = w.next_merge().expect("a registered merge");
        assert!(merge
            .segments
            .iter()
            .all(|n| lock(&w.core).merging.contains(n)));
        w.on_merge_finished(&merge);
        assert!(merge
            .segments
            .iter()
            .all(|n| !lock(&w.core).merging.contains(n)));

        let tmp2 = TempDir::new("concurrent-no-sched");
        let dir2 = FsDirectory::open(&tmp2);
        let plain = ConcurrentIndexWriter::new(writer(&dir2, 5), 1).unwrap();
        plain.close_merges().unwrap();
    }

    /// `flushNextBuffer` takes the fullest slot; `tryDeleteDocument` and
    /// `tryUpdateDocValue` act on a snapshot's doc ids and are numbered from
    /// this writer's log.
    #[test]
    fn flush_next_buffer_and_try_delete_by_doc_id() {
        use crate::nrt::NrtSource;
        let tmp = TempDir::new("concurrent-try");
        let dir = FsDirectory::open(&tmp);
        let w = ConcurrentIndexWriter::new(writer(&dir, 1000), 2).unwrap();
        assert!(!w.flush_next_buffer().unwrap());
        for i in 0..3 {
            w.add_document(doc(&format!("a{i}"), 0)).unwrap();
        }
        // Round-robin over two slots: the fuller one (two documents) goes
        // first, then the other.
        assert!(w.flush_next_buffer().unwrap());
        assert_eq!(w.pending_doc_count(), 1);
        assert!(w.flush_next_buffer().unwrap());
        assert_eq!(w.pending_doc_count(), 0);
        let snap = w.nrt_snapshot(true, false).unwrap();
        let before = lock(&w.log).next_seq;
        let seq = w.try_delete_document(&snap.segment_infos, 1).unwrap();
        assert_eq!(seq, Some(before));
        let update = DocValuesUpdate::Numeric {
            term: Term::new("", ""),
            field: "id".into(),
            value: Some(1),
        };
        // `id` has no doc values: refused before anything is written.
        assert!(w
            .try_update_doc_value(&snap.segment_infos, 0, &[update])
            .is_err());
        w.commit().unwrap();
        let (docs, _) = live_documents(&dir);
        assert_eq!(docs.len(), 2);
        // Doc 1 of the snapshot is the second of the first segment flushed,
        // which (round-robin, fuller slot first) holds a0 and a2.
        assert!(!docs.contains_key("a2"), "{docs:?}");
        assert_clean(&dir);
    }
}
