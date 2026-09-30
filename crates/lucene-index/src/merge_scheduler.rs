//! Port of `org.apache.lucene.index.MergeScheduler` and its implementations:
//! [`SerialMergeScheduler`], [`ConcurrentMergeScheduler`] (a merge-thread
//! pool with `maxThreadCount`/`maxMergeCount`, stalling of the threads that
//! produce segments, pausing of the largest merges and the automatic IO
//! throttle) and [`MultiIndexMergeScheduler`] (several indexes sharing one
//! `ConcurrentMergeScheduler`'s threads).
//!
//! A scheduler is handed a [`MergeSource`] -- the writer's pending-merge
//! queue, Java's `MergeScheduler.MergeSource` -- and decides on which thread,
//! and at what IO rate, each pending merge runs.
//! [`crate::concurrent_writer::ConcurrentIndexWriter::with_merge_scheduler`]
//! is the writer that registers merges and calls one; the single-threaded
//! [`crate::index_writer::IndexWriter`] runs its merges inside `commit`,
//! which is [`SerialMergeScheduler`]'s behaviour.
//!
//! # What differs from Java
//!
//! - **Threads own what they use.** A merge thread outlives the `merge` call
//!   that started it, so it holds the source as an `Arc<dyn MergeSource>`
//!   (`'static`), where Java holds the `IndexWriter` by reference.
//! - **Merge failures.** Java's merge thread rethrows as
//!   `MergePolicy.MergeException` on itself (`handleMergeException`), which
//!   the JVM reports and the writer records as a tragedy. Here the error is
//!   kept, and [`ConcurrentMergeScheduler::sync`] (and so `close`) returns
//!   the first one: a Rust thread has no uncaught-exception handler to report
//!   it to. A panicking merge is caught and kept the same way.
//! - **`getIntraMergeExecutor`** is not ported: nothing in this port's merge
//!   splits one merge across threads (Java's only user is the parallel HNSW
//!   graph merge), so every merge's work runs on its merge thread, which is
//!   what Java's executor does for any merge under 50 MB.
//! - **`isAlive`** is a flag the merge thread clears as it exits, and a
//!   thread is identified by its `ThreadId`.
//! - **The clock** is injectable, for the IO throttle's tests.

use std::collections::VecDeque;
use std::fmt;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::thread::{JoinHandle, ThreadId};
use std::time::{Duration, Instant};

use lucene_store::RateLimiter;

use crate::index_writer::{Error, Result};
use crate::merge_policy::MergeTrigger;
use crate::merge_rate_limiter::{MergeRateLimiter, OneMergeProgress};

/// `MergePolicy.OneMerge` as a scheduler sees it: the segments it combines,
/// how big it is expected to be, whether it is a forced merge, and the
/// progress (abort flag, pauses) its rate limiter acts through.
pub struct ScheduledMerge {
    /// The merged segments' names, in merge order.
    pub segments: Vec<String>,
    /// `OneMerge.estimatedMergeBytes`.
    pub estimated_merge_bytes: u64,
    /// `OneMerge.maxNumSegments`: `-1` for a natural merge, the target count
    /// for a `forceMerge`.
    pub max_num_segments: i32,
    progress: Arc<OneMergeProgress>,
    /// `OneMerge.mergeStartNS`, `-1` until the merge starts.
    start_ns: AtomicI64,
}

impl fmt::Debug for ScheduledMerge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ScheduledMerge")
            .field("segments", &self.segments)
            .field("estimated_merge_bytes", &self.estimated_merge_bytes)
            .field("max_num_segments", &self.max_num_segments)
            .field("aborted", &self.is_aborted())
            .finish()
    }
}

impl ScheduledMerge {
    /// A natural merge of `segments`, expected to write about
    /// `estimated_merge_bytes`.
    pub fn new(segments: Vec<String>, estimated_merge_bytes: u64) -> Self {
        ScheduledMerge {
            segments,
            estimated_merge_bytes,
            max_num_segments: -1,
            progress: Arc::new(OneMergeProgress::new()),
            start_ns: AtomicI64::new(-1),
        }
    }

    /// The same merge, as part of `forceMerge(max_num_segments)`.
    pub fn forced(mut self, max_num_segments: i32) -> Self {
        self.max_num_segments = max_num_segments;
        self
    }

    /// `OneMerge.getMergeProgress()`.
    pub fn progress(&self) -> &Arc<OneMergeProgress> {
        &self.progress
    }

    /// `OneMerge.isAborted()`.
    pub fn is_aborted(&self) -> bool {
        self.progress.is_aborted()
    }

    /// `OneMerge.setAborted()`.
    pub fn abort(&self) {
        self.progress.abort();
    }

    fn start_ns(&self) -> i64 {
        self.start_ns.load(Ordering::SeqCst)
    }
}

/// `MergeScheduler.MergeSource`: the pending merges of one writer.
pub trait MergeSource: Send + Sync {
    /// `getNextMerge()`: takes the next pending merge, if any.
    fn next_merge(&self) -> Option<Arc<ScheduledMerge>>;
    /// `onMergeFinished(merge)`: `merge` was taken but will not run (the
    /// scheduler failed to start it); release what registering it claimed.
    fn on_merge_finished(&self, merge: &Arc<ScheduledMerge>);
    /// `hasPendingMerges()`.
    fn has_pending_merges(&self) -> bool;
    /// `merge(merge)`: runs `merge` on the calling thread. Every file it
    /// writes goes through `limiter` when one is given -- the directory
    /// `MergeScheduler.wrapForMerge` returns.
    fn merge(
        &self,
        merge: &Arc<ScheduledMerge>,
        limiter: Option<Arc<MergeRateLimiter>>,
    ) -> Result<()>;
}

/// `MergeScheduler`: decides when, and on which thread, a source's pending
/// merges run.
pub trait MergeScheduler: Send + Sync + fmt::Debug {
    /// `merge(mergeSource, trigger)`: run (or start) the source's pending
    /// merges.
    fn merge(&self, source: &Arc<dyn MergeSource>, trigger: MergeTrigger) -> Result<()>;
    /// `close()`: waits for the merges this scheduler started, returning the
    /// first failure among them.
    fn close(&self) -> Result<()> {
        Ok(())
    }
}

/// `SerialMergeScheduler`: every pending merge, one after another, on the
/// thread that asked. The `synchronized` of Java's `merge` is the mutex: two
/// threads never merge through one serial scheduler at once.
#[derive(Debug, Default)]
pub struct SerialMergeScheduler {
    lock: Mutex<()>,
}

impl SerialMergeScheduler {
    pub fn new() -> Self {
        Self::default()
    }
}

impl MergeScheduler for SerialMergeScheduler {
    fn merge(&self, source: &Arc<dyn MergeSource>, _trigger: MergeTrigger) -> Result<()> {
        let _guard = self.lock.lock().unwrap_or_else(|p| p.into_inner());
        while let Some(merge) = source.next_merge() {
            source.merge(&merge, None)?;
        }
        Ok(())
    }
}

/// `NoMergeScheduler`: never merges. Pending merges stay pending.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoMergeScheduler;

impl MergeScheduler for NoMergeScheduler {
    fn merge(&self, _source: &Arc<dyn MergeSource>, _trigger: MergeTrigger) -> Result<()> {
        Ok(())
    }
}

/// `ConcurrentMergeScheduler.AUTO_DETECT_MERGES_AND_THREADS`.
pub const AUTO_DETECT_MERGES_AND_THREADS: i32 = -1;

const MIN_MERGE_MB_PER_SEC: f64 = 5.0;
const MAX_MERGE_MB_PER_SEC: f64 = 10240.0;
const START_MB_PER_SEC: f64 = 20.0;
const MIN_BIG_MERGE_MB: f64 = 50.0;
const MIN_BIG_MERGE_BYTES: u64 = 50 * 1024 * 1024;

fn bytes_to_mb(bytes: u64) -> f64 {
    bytes as f64 / 1024. / 1024.
}

fn ns_to_sec(ns: i64) -> f64 {
    ns as f64 / 1_000_000_000.0
}

/// One `ConcurrentMergeScheduler.MergeThread`.
struct MergeThread {
    name: String,
    thread: Option<ThreadId>,
    handle: Option<JoinHandle<()>>,
    /// `isAlive()`: cleared by the thread itself as it exits.
    alive: Arc<AtomicBool>,
    /// [`MultiIndexMergeScheduler`]'s `TaggedMergeSource.getDirectory()`.
    tag: Option<String>,
    merge: Arc<ScheduledMerge>,
    limiter: Arc<MergeRateLimiter>,
}

impl MergeThread {
    fn is_alive(&self) -> bool {
        self.alive.load(Ordering::SeqCst)
    }
}

struct CmsState {
    merge_threads: Vec<MergeThread>,
    max_thread_count: i32,
    max_merge_count: i32,
    merge_thread_counter: u64,
    target_mb_per_sec: f64,
    do_auto_io_throttle: bool,
    force_merge_mb_per_sec: f64,
    /// Failures of merges run on merge threads, oldest first.
    errors: Vec<Error>,
    /// How many times a producer has waited in `maybeStall` (observability,
    /// and what the stall tests wait for).
    stall_waits: u64,
}

type Clock = Box<dyn Fn() -> i64 + Send + Sync>;

struct CmsInner {
    state: Mutex<CmsState>,
    /// `ConcurrentMergeScheduler.this`'s monitor: `wait`/`notifyAll`.
    cond: Condvar,
    clock: Clock,
}

fn lock(m: &Mutex<CmsState>) -> MutexGuard<'_, CmsState> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

/// `ConcurrentMergeScheduler`: each merge on a thread of its own.
///
/// - At most `max_merge_count` merges run at once; a thread that produces
///   segments (asks for merges) while that many are running and more are
///   pending **stalls** until one finishes (`maybeStall`), so indexing cannot
///   outrun merging without bound. A merge thread never stalls.
/// - Of the running merges, only `max_thread_count` of the *big* ones
///   (over 50 MB) proceed; the largest ones beyond that are paused through
///   their rate limiter (rate `0`) until smaller ones finish
///   (`updateMergeThreads`).
/// - With the automatic IO throttle on, each new big merge moves a shared
///   target rate: up 20% when a similarly sized merge is already running
///   (merging is falling behind), down 10% otherwise, between 5 MB/s and
///   10 GB/s (`updateIOThrottle`). A forced merge runs at
///   `force_merge_mb_per_sec` instead.
///
/// Cloning shares the scheduler.
#[derive(Clone)]
pub struct ConcurrentMergeScheduler {
    inner: Arc<CmsInner>,
}

impl fmt::Debug for ConcurrentMergeScheduler {
    /// `toString()`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = lock(&self.inner.state);
        write!(
            f,
            "ConcurrentMergeScheduler: maxThreadCount={}, maxMergeCount={}, ioThrottle={}",
            s.max_thread_count, s.max_merge_count, s.do_auto_io_throttle
        )
    }
}

impl Default for ConcurrentMergeScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl ConcurrentMergeScheduler {
    /// A scheduler that sizes itself on first use (`AUTO_DETECT`), with the
    /// automatic IO throttle off -- Java's defaults.
    pub fn new() -> Self {
        let epoch = Instant::now();
        Self::with_clock(Box::new(move || {
            i64::try_from(epoch.elapsed().as_nanos()).unwrap_or(i64::MAX)
        }))
    }

    fn with_clock(clock: Clock) -> Self {
        ConcurrentMergeScheduler {
            inner: Arc::new(CmsInner {
                state: Mutex::new(CmsState {
                    merge_threads: Vec::new(),
                    max_thread_count: AUTO_DETECT_MERGES_AND_THREADS,
                    max_merge_count: AUTO_DETECT_MERGES_AND_THREADS,
                    merge_thread_counter: 0,
                    target_mb_per_sec: START_MB_PER_SEC,
                    do_auto_io_throttle: false,
                    force_merge_mb_per_sec: f64::INFINITY,
                    errors: Vec::new(),
                    stall_waits: 0,
                }),
                cond: Condvar::new(),
                clock,
            }),
        }
    }

    /// `setMaxMergesAndThreads(maxMergeCount, maxThreadCount)`: both
    /// [`AUTO_DETECT_MERGES_AND_THREADS`], or both at least 1 with
    /// `max_thread_count <= max_merge_count`.
    pub fn set_max_merges_and_threads(
        &self,
        max_merge_count: i32,
        max_thread_count: i32,
    ) -> Result<()> {
        let bad = |m: &str| Err(Error::InvalidMergeScheduler(m.to_string()));
        let auto = AUTO_DETECT_MERGES_AND_THREADS;
        if (max_merge_count == auto) != (max_thread_count == auto) {
            return bad(
                "both maxMergeCount and maxThreadCount must be AUTO_DETECT_MERGES_AND_THREADS",
            );
        }
        if max_merge_count != auto {
            if max_thread_count < 1 {
                return bad("maxThreadCount should be at least 1");
            }
            if max_merge_count < 1 {
                return bad("maxMergeCount should be at least 1");
            }
            if max_thread_count > max_merge_count {
                return Err(Error::InvalidMergeScheduler(format!(
                    "maxThreadCount should be <= maxMergeCount (= {max_merge_count})"
                )));
            }
        }
        let mut s = lock(&self.inner.state);
        s.max_merge_count = max_merge_count;
        s.max_thread_count = max_thread_count;
        Ok(())
    }

    /// `setDefaultMaxMergesAndThreads(spins)`: one thread and six merges for
    /// a spinning disk; otherwise half the cores (at least one) and five more
    /// merges than threads.
    pub fn set_default_max_merges_and_threads(&self, spins: bool) {
        let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
        let mut s = lock(&self.inner.state);
        Self::apply_defaults(&mut s, spins, cores);
    }

    fn apply_defaults(s: &mut CmsState, spins: bool, cores: usize) {
        if spins {
            s.max_thread_count = 1;
            s.max_merge_count = 6;
        } else {
            s.max_thread_count = i32::try_from(cores / 2).unwrap_or(i32::MAX).max(1);
            s.max_merge_count = s.max_thread_count.saturating_add(5);
        }
    }

    /// `getMaxThreadCount()`.
    pub fn max_thread_count(&self) -> i32 {
        lock(&self.inner.state).max_thread_count
    }

    /// `getMaxMergeCount()`.
    pub fn max_merge_count(&self) -> i32 {
        lock(&self.inner.state).max_merge_count
    }

    /// `setForceMergeMBPerSec(v)`.
    pub fn set_force_merge_mb_per_sec(&self, v: f64) {
        let mut s = lock(&self.inner.state);
        s.force_merge_mb_per_sec = v;
        Self::update_merge_threads(&mut s);
    }

    /// `getForceMergeMBPerSec()`.
    pub fn force_merge_mb_per_sec(&self) -> f64 {
        lock(&self.inner.state).force_merge_mb_per_sec
    }

    /// `enableAutoIOThrottle()`: resets the target to 20 MB/s.
    pub fn enable_auto_io_throttle(&self) {
        let mut s = lock(&self.inner.state);
        s.do_auto_io_throttle = true;
        s.target_mb_per_sec = START_MB_PER_SEC;
        Self::update_merge_threads(&mut s);
    }

    /// `disableAutoIOThrottle()`.
    pub fn disable_auto_io_throttle(&self) {
        let mut s = lock(&self.inner.state);
        s.do_auto_io_throttle = false;
        Self::update_merge_threads(&mut s);
    }

    /// `getAutoIOThrottle()`.
    pub fn auto_io_throttle(&self) -> bool {
        lock(&self.inner.state).do_auto_io_throttle
    }

    /// `getIORateLimitMBPerSec()`: the throttle's target, or infinity when
    /// the throttle is off.
    pub fn io_rate_limit_mb_per_sec(&self) -> f64 {
        let s = lock(&self.inner.state);
        if s.do_auto_io_throttle {
            s.target_mb_per_sec
        } else {
            f64::INFINITY
        }
    }

    /// `mergeThreadCount()`: live merge threads, other than the calling one,
    /// whose merge is not aborted.
    pub fn merge_thread_count(&self) -> usize {
        Self::count_merge_threads(&lock(&self.inner.state))
    }

    fn count_merge_threads(s: &CmsState) -> usize {
        let current = std::thread::current().id();
        s.merge_threads
            .iter()
            .filter(|t| t.thread != Some(current) && t.is_alive() && !t.merge.is_aborted())
            .count()
    }

    /// How many times a producer thread has waited because too many merges
    /// were running.
    pub fn stall_waits(&self) -> u64 {
        lock(&self.inner.state).stall_waits
    }

    /// Every live merge thread's name and current IO rate (`0` paused,
    /// infinity unlimited), largest merge first.
    pub fn merge_thread_rates(&self) -> Vec<(String, u64, f64)> {
        let s = lock(&self.inner.state);
        let mut v: Vec<(String, u64, f64)> = s
            .merge_threads
            .iter()
            .filter(|t| t.is_alive())
            .map(|t| {
                (
                    t.name.clone(),
                    t.merge.estimated_merge_bytes,
                    t.limiter.mb_per_sec(),
                )
            })
            .collect();
        v.sort_by_key(|t| std::cmp::Reverse(t.1));
        v
    }

    /// `sync()`: waits for every merge thread to finish, then returns (and
    /// forgets) the first merge failure, if any.
    pub fn sync(&self) -> Result<()> {
        self.sync_where(|_| true)
    }

    /// `sync`, restricted to the threads `pred` selects by tag --
    /// `CombinedMergeScheduler.sync(directory)`.
    fn sync_where(&self, pred: impl Fn(Option<&str>) -> bool) -> Result<()> {
        let current = std::thread::current().id();
        loop {
            let handle = {
                let mut s = lock(&self.inner.state);
                s.merge_threads
                    .iter_mut()
                    .find(|t| {
                        t.is_alive()
                            && t.thread != Some(current)
                            && pred(t.tag.as_deref())
                            && t.handle.is_some()
                    })
                    .and_then(|t| t.handle.take())
            };
            match handle {
                Some(h) => {
                    let _ = h.join();
                }
                None => {
                    // A thread that removed itself from the list may still
                    // be finishing; wait for every selected one to be gone.
                    let s = lock(&self.inner.state);
                    if s.merge_threads.iter().any(|t| {
                        t.is_alive() && t.thread != Some(current) && pred(t.tag.as_deref())
                    }) {
                        let _ = self
                            .inner
                            .cond
                            .wait_timeout(s, Duration::from_millis(10))
                            .unwrap_or_else(|p| p.into_inner());
                        continue;
                    }
                    break;
                }
            }
        }
        let mut s = lock(&self.inner.state);
        if s.errors.is_empty() {
            Ok(())
        } else {
            Err(s.errors.remove(0))
        }
    }

    /// `initDynamicDefaults`.
    fn init_dynamic_defaults(s: &mut CmsState) {
        if s.max_thread_count == AUTO_DETECT_MERGES_AND_THREADS {
            let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
            Self::apply_defaults(s, false, cores);
        }
    }

    /// `merge(mergeSource, trigger)`, tagging the threads it starts.
    fn merge_tagged(
        &self,
        source: &Arc<dyn MergeSource>,
        trigger: MergeTrigger,
        tag: Option<&str>,
    ) -> Result<()> {
        let s = lock(&self.inner.state);
        let (s, result) = self.merge_locked(s, source, trigger, tag);
        drop(s);
        result
    }

    /// The body of Java's `synchronized merge`, entered with the monitor
    /// held -- from a producer thread, or from a merge thread that just
    /// finished (`runOnMergeFinished`).
    fn merge_locked<'a>(
        &'a self,
        mut s: MutexGuard<'a, CmsState>,
        source: &Arc<dyn MergeSource>,
        trigger: MergeTrigger,
        tag: Option<&str>,
    ) -> (MutexGuard<'a, CmsState>, Result<()>) {
        Self::init_dynamic_defaults(&mut s);
        if trigger == MergeTrigger::Closing {
            // Disable throttling on close.
            s.target_mb_per_sec = MAX_MERGE_MB_PER_SEC;
            Self::update_merge_threads(&mut s);
        }
        loop {
            let (next, go) = self.maybe_stall(s, source);
            s = next;
            if !go {
                return (s, Ok(()));
            }
            let Some(merge) = source.next_merge() else {
                return (s, Ok(()));
            };
            match self.start_merge_thread(&mut s, source, &merge, tag) {
                Ok(()) => Self::update_merge_threads(&mut s),
                Err(e) => {
                    source.on_merge_finished(&merge);
                    return (s, Err(e));
                }
            }
        }
    }

    /// `maybeStall`: while the source has merges pending and
    /// `max_merge_count` merges are already running, the producer waits
    /// (a quarter of a second at a time, in case a wakeup is missed). A merge
    /// thread is never stalled -- it returns `false`, ending its `merge`.
    fn maybe_stall<'a>(
        &'a self,
        mut s: MutexGuard<'a, CmsState>,
        source: &Arc<dyn MergeSource>,
    ) -> (MutexGuard<'a, CmsState>, bool) {
        let current = std::thread::current().id();
        while source.has_pending_merges()
            && i64::try_from(Self::count_merge_threads(&s)).unwrap_or(i64::MAX)
                >= i64::from(s.max_merge_count)
        {
            if s.merge_threads.iter().any(|t| t.thread == Some(current)) {
                return (s, false);
            }
            s.stall_waits = s.stall_waits.saturating_add(1);
            self.inner.cond.notify_all();
            s = self
                .inner
                .cond
                .wait_timeout(s, Duration::from_millis(250))
                .unwrap_or_else(|p| p.into_inner())
                .0;
        }
        (s, true)
    }

    /// `getMergeThread` + `mergeThreads.add` + `updateIOThrottle` +
    /// `start()`.
    fn start_merge_thread(
        &self,
        s: &mut CmsState,
        source: &Arc<dyn MergeSource>,
        merge: &Arc<ScheduledMerge>,
        tag: Option<&str>,
    ) -> Result<()> {
        let name = format!("Lucene Merge Thread #{}", s.merge_thread_counter);
        s.merge_thread_counter = s.merge_thread_counter.saturating_add(1);
        let limiter = Arc::new(MergeRateLimiter::new(Arc::clone(merge.progress())));
        let alive = Arc::new(AtomicBool::new(true));
        s.merge_threads.push(MergeThread {
            name: name.clone(),
            thread: None,
            handle: None,
            alive: Arc::clone(&alive),
            tag: tag.map(str::to_string),
            merge: Arc::clone(merge),
            limiter: Arc::clone(&limiter),
        });
        Self::update_io_throttle(s, &self.inner.clock, merge, &limiter);
        let this = self.clone();
        let thread_source = Arc::clone(source);
        let thread_merge = Arc::clone(merge);
        let thread_tag = tag.map(str::to_string);
        let spawned = std::thread::Builder::new().name(name).spawn(move || {
            this.run_merge_thread(thread_source, thread_merge, limiter, thread_tag, alive);
        });
        match spawned {
            Ok(handle) => {
                let id = handle.thread().id();
                if let Some(t) = s
                    .merge_threads
                    .iter_mut()
                    .rev()
                    .find(|t| Arc::ptr_eq(&t.merge, merge))
                {
                    t.thread = Some(id);
                    t.handle = Some(handle);
                }
                Ok(())
            }
            Err(e) => {
                s.merge_threads.retain(|t| !Arc::ptr_eq(&t.merge, merge));
                Err(Error::Store(lucene_store::Error::Io(e)))
            }
        }
    }

    /// `MergeThread.run`: `doMerge`, then `runOnMergeFinished` -- which asks
    /// for more merges from this (merge) thread, removes the thread and wakes
    /// any stalled producer.
    fn run_merge_thread(
        &self,
        source: Arc<dyn MergeSource>,
        merge: Arc<ScheduledMerge>,
        limiter: Arc<MergeRateLimiter>,
        tag: Option<String>,
        alive: Arc<AtomicBool>,
    ) {
        /// Clears `alive` however the thread ends.
        struct Exit(Arc<AtomicBool>, Arc<CmsInner>);
        impl Drop for Exit {
            fn drop(&mut self) {
                self.0.store(false, Ordering::SeqCst);
                self.1.cond.notify_all();
            }
        }
        let _exit = Exit(alive, Arc::clone(&self.inner));
        // The thread's id is recorded by the spawner under the monitor; take
        // it before starting so `mergeThreadCount` excludes this thread.
        drop(lock(&self.inner.state));
        merge.start_ns.store((self.inner.clock)(), Ordering::SeqCst);
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            source.merge(&merge, Some(Arc::clone(&limiter)))
        }));
        let failure = match outcome {
            Ok(Ok(())) => None,
            // `MergeAbortedException`: OK to ignore.
            Ok(Err(Error::MergeAborted)) => None,
            Ok(Err(e)) => Some(e),
            Err(panic) => Some(Error::MergeThreadPanicked(panic_message(&panic))),
        };
        let s = lock(&self.inner.state);
        let mut s = if failure.is_none() {
            // `runOnMergeFinished`: let the scheduler run new merges.
            let (s, result) =
                self.merge_locked(s, &source, MergeTrigger::MergeFinished, tag.as_deref());
            let mut s = s;
            if let Err(e) = result {
                s.errors.push(e);
            }
            s
        } else {
            s
        };
        if let Some(e) = failure {
            s.errors.push(e);
        }
        let current = std::thread::current().id();
        s.merge_threads.retain(|t| t.thread != Some(current));
        Self::update_merge_threads(&mut s);
        drop(s);
        // Let go of the source before `alive` clears: a caller that `sync`s
        // and then takes its writer back out of an `Arc` must find this
        // thread's reference gone.
        drop(source);
        drop(merge);
        self.inner.cond.notify_all();
    }

    /// `updateMergeThreads`: sorts the live merges largest first; pauses the
    /// largest big merges beyond `max_thread_count`; gives the rest the rate
    /// their kind of merge runs at.
    fn update_merge_threads(s: &mut CmsState) {
        s.merge_threads.retain(MergeThread::is_alive);
        let mut active: Vec<&MergeThread> = s.merge_threads.iter().collect();
        // Larger merges sort first; stable, as `CollectionUtil.timSort`.
        active.sort_by_key(|t| std::cmp::Reverse(t.merge.estimated_merge_bytes));
        let big_merge_count = active
            .iter()
            .rposition(|t| t.merge.estimated_merge_bytes > MIN_BIG_MERGE_BYTES)
            .map_or(0, |i| i.saturating_add(1));
        let max_thread_count = usize::try_from(s.max_thread_count).unwrap_or(usize::MAX);
        let pause_below = big_merge_count.saturating_sub(max_thread_count);
        for (idx, t) in active.iter().enumerate() {
            let merge = &t.merge;
            let rate = if idx < pause_below {
                0.0
            } else if merge.max_num_segments != -1 {
                s.force_merge_mb_per_sec
            } else if !s.do_auto_io_throttle || merge.estimated_merge_bytes < MIN_BIG_MERGE_BYTES {
                f64::INFINITY
            } else {
                s.target_mb_per_sec
            };
            t.limiter.set_mb_per_sec(rate);
        }
    }

    /// `isBacklog(now, merge)`: another big merge of similar size (a third
    /// to three times as big) has been running for more than three seconds.
    fn is_backlog(s: &CmsState, now: i64, merge: &ScheduledMerge) -> bool {
        let merge_mb = bytes_to_mb(merge.estimated_merge_bytes);
        s.merge_threads.iter().any(|t| {
            let start = t.merge.start_ns();
            t.is_alive()
                && !std::ptr::eq(Arc::as_ptr(&t.merge), merge)
                && start != -1
                && t.merge.estimated_merge_bytes >= MIN_BIG_MERGE_BYTES
                && ns_to_sec(now.saturating_sub(start)) > 3.0
                && {
                    let ratio = bytes_to_mb(t.merge.estimated_merge_bytes) / merge_mb;
                    ratio > 0.3 && ratio < 3.0
                }
        })
    }

    /// `updateIOThrottle(newMerge, rateLimiter)`: the automatic IO throttle's
    /// closed-loop step, taken for each new big merge.
    fn update_io_throttle(
        s: &mut CmsState,
        clock: &Clock,
        new_merge: &ScheduledMerge,
        limiter: &MergeRateLimiter,
    ) {
        if !s.do_auto_io_throttle {
            return;
        }
        if bytes_to_mb(new_merge.estimated_merge_bytes) < MIN_BIG_MERGE_MB {
            return;
        }
        let now = clock();
        let new_backlog = Self::is_backlog(s, now, new_merge);
        let cur_backlog = !new_backlog
            && (i64::try_from(s.merge_threads.len()).unwrap_or(i64::MAX)
                > i64::from(s.max_thread_count)
                || s.merge_threads
                    .iter()
                    .any(|t| Self::is_backlog(s, now, &t.merge)));
        if new_backlog {
            s.target_mb_per_sec = (s.target_mb_per_sec * 1.20).min(MAX_MERGE_MB_PER_SEC);
        } else if !cur_backlog {
            s.target_mb_per_sec = (s.target_mb_per_sec / 1.10).max(MIN_MERGE_MB_PER_SEC);
        }
        let rate = if new_merge.max_num_segments != -1 {
            s.force_merge_mb_per_sec
        } else {
            s.target_mb_per_sec
        };
        limiter.set_mb_per_sec(rate);
    }
}

fn panic_message(panic: &Box<dyn std::any::Any + Send>) -> String {
    panic
        .downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| panic.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "non-string panic payload".to_string())
}

impl MergeScheduler for ConcurrentMergeScheduler {
    fn merge(&self, source: &Arc<dyn MergeSource>, trigger: MergeTrigger) -> Result<()> {
        self.merge_tagged(source, trigger, None)
    }

    /// `close()`: `sync()`.
    fn close(&self) -> Result<()> {
        self.sync()
    }
}

/// `MultiIndexMergeScheduler`: a scheduler for one index among several that
/// share **one** [`ConcurrentMergeScheduler`] (Java's `CombinedMergeScheduler`),
/// so `max_merge_count`, the pausing of big merges and the IO throttle are
/// decided across every index at once -- a merge storm in one index stalls
/// producers of all of them rather than each index claiming its own threads.
///
/// Each index is identified by a key (Java: its `Directory`); closing one
/// waits only for that index's merges.
#[derive(Debug)]
pub struct MultiIndexMergeScheduler {
    key: String,
    combined: ConcurrentMergeScheduler,
    manage_singleton: bool,
}

/// `CombinedMergeScheduler.singleton` and its reference count.
fn singleton() -> &'static Mutex<Option<(ConcurrentMergeScheduler, usize)>> {
    static SINGLETON: OnceLock<Mutex<Option<(ConcurrentMergeScheduler, usize)>>> = OnceLock::new();
    SINGLETON.get_or_init(|| Mutex::new(None))
}

impl MultiIndexMergeScheduler {
    /// `new MultiIndexMergeScheduler(directory)`: shares the process-wide
    /// combined scheduler, created on first use and closed when the last
    /// index using it closes.
    pub fn new(key: impl Into<String>) -> Self {
        let mut slot = singleton().lock().unwrap_or_else(|p| p.into_inner());
        let (cms, refs) = slot.get_or_insert_with(|| (ConcurrentMergeScheduler::new(), 0));
        *refs = refs.saturating_add(1);
        MultiIndexMergeScheduler {
            key: key.into(),
            combined: cms.clone(),
            manage_singleton: true,
        }
    }

    /// `MultiIndexMergeScheduler(directory, combinedMergeScheduler)`: shares
    /// `combined`, which the caller owns.
    pub fn with_scheduler(key: impl Into<String>, combined: ConcurrentMergeScheduler) -> Self {
        MultiIndexMergeScheduler {
            key: key.into(),
            combined,
            manage_singleton: false,
        }
    }

    /// `getDirectory()`.
    pub fn key(&self) -> &str {
        &self.key
    }

    /// `getCombinedMergeScheduler()`.
    pub fn combined_merge_scheduler(&self) -> &ConcurrentMergeScheduler {
        &self.combined
    }

    /// `CombinedMergeScheduler.peekSingleton()`.
    pub fn peek_singleton() -> Option<ConcurrentMergeScheduler> {
        singleton()
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
            .map(|(c, _)| c.clone())
    }
}

impl MergeScheduler for MultiIndexMergeScheduler {
    /// Tags the source with this index's key (`TaggedMergeSource`) and hands
    /// it to the combined scheduler.
    fn merge(&self, source: &Arc<dyn MergeSource>, trigger: MergeTrigger) -> Result<()> {
        self.combined.merge_tagged(source, trigger, Some(&self.key))
    }

    /// Waits for this index's merges only (`sync(directory)`), then drops
    /// this index's reference to the shared singleton.
    fn close(&self) -> Result<()> {
        let key = self.key.clone();
        let result = self.combined.sync_where(|tag| tag == Some(key.as_str()));
        if self.manage_singleton {
            let mut slot = singleton().lock().unwrap_or_else(|p| p.into_inner());
            if let Some((cms, refs)) = slot.as_mut() {
                *refs = refs.saturating_sub(1);
                if *refs == 0 {
                    let cms = cms.clone();
                    *slot = None;
                    drop(slot);
                    cms.sync()?;
                }
            }
        }
        result
    }
}

/// A queue of pending merges -- the part of `IndexWriter` a `MergeSource`
/// exposes -- for a caller (or a test) that registers merges itself.
pub(crate) struct PendingMerges {
    queue: Mutex<VecDeque<Arc<ScheduledMerge>>>,
}

impl PendingMerges {
    pub(crate) fn new() -> Self {
        PendingMerges {
            queue: Mutex::new(VecDeque::new()),
        }
    }

    pub(crate) fn push(&self, merge: Arc<ScheduledMerge>) {
        self.queue
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push_back(merge);
    }

    pub(crate) fn pop(&self) -> Option<Arc<ScheduledMerge>> {
        self.queue
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .pop_front()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.queue
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .is_empty()
    }
}

#[cfg(test)]
mod tests;
