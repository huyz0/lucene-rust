//! Deterministic tests for the schedulers: every test merge blocks on a gate
//! the test opens, so which merges are running, paused or pending at each
//! assertion is decided by the test, not by thread timing. Waits that depend
//! on a thread reaching a state poll with a generous deadline and fail (not
//! hang) if the state never comes.
// Test code: see `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use super::*;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::AtomicUsize;

const MB: u64 = 1024 * 1024;

/// A merge source whose merges block until released.
#[derive(Default)]
struct TestSource {
    pending: Mutex<VecDeque<Arc<ScheduledMerge>>>,
    released: Mutex<HashSet<String>>,
    release_all: AtomicBool,
    gate: Condvar,
    running: AtomicUsize,
    ran: Mutex<Vec<(String, ThreadId)>>,
    returned: Mutex<Vec<String>>,
    /// Merges that fail instead of running: `"err"` returns an error,
    /// `"panic"` panics, `"abort"` reports an aborted merge.
    outcome: Mutex<HashMap<String, &'static str>>,
}

impl TestSource {
    fn with(merges: &[(&str, u64)]) -> Arc<Self> {
        let s = TestSource::default();
        for &(name, bytes) in merges {
            s.pending
                .lock()
                .unwrap()
                .push_back(Arc::new(ScheduledMerge::new(
                    vec![name.to_string()],
                    bytes * MB,
                )));
        }
        Arc::new(s)
    }

    fn push(&self, merge: ScheduledMerge) {
        self.pending.lock().unwrap().push_back(Arc::new(merge));
    }

    fn release(&self, name: &str) {
        self.released.lock().unwrap().insert(name.to_string());
        self.gate.notify_all();
    }

    fn release_everything(&self) {
        self.release_all.store(true, Ordering::SeqCst);
        let _g = self.released.lock().unwrap();
        self.gate.notify_all();
    }

    fn ran(&self) -> Vec<String> {
        self.ran
            .lock()
            .unwrap()
            .iter()
            .map(|(n, _)| n.clone())
            .collect()
    }
}

impl MergeSource for TestSource {
    fn next_merge(&self) -> Option<Arc<ScheduledMerge>> {
        self.pending.lock().unwrap().pop_front()
    }

    fn on_merge_finished(&self, merge: &Arc<ScheduledMerge>) {
        self.returned
            .lock()
            .unwrap()
            .push(merge.segments[0].clone());
    }

    fn has_pending_merges(&self) -> bool {
        !self.pending.lock().unwrap().is_empty()
    }

    fn merge(
        &self,
        merge: &Arc<ScheduledMerge>,
        _limiter: Option<Arc<MergeRateLimiter>>,
    ) -> Result<()> {
        let name = merge.segments[0].clone();
        self.running.fetch_add(1, Ordering::SeqCst);
        {
            let mut released = self.released.lock().unwrap();
            while !self.release_all.load(Ordering::SeqCst) && !released.contains(&name) {
                released = self.gate.wait(released).unwrap();
            }
        }
        self.running.fetch_sub(1, Ordering::SeqCst);
        self.ran
            .lock()
            .unwrap()
            .push((name.clone(), std::thread::current().id()));
        let outcome = self.outcome.lock().unwrap().get(&name).copied();
        match outcome {
            Some("err") => Err(Error::InvalidMergeScheduler(format!("{name} failed"))),
            Some("panic") => panic!("{name} blew up"),
            Some("abort") => Err(Error::MergeAborted),
            _ => Ok(()),
        }
    }
}

fn dyn_source(s: &Arc<TestSource>) -> Arc<dyn MergeSource> {
    Arc::clone(s) as Arc<dyn MergeSource>
}

/// Polls `cond` until it holds, failing the test after ten seconds.
fn wait_until(what: &str, cond: impl Fn() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !cond() {
        assert!(Instant::now() < deadline, "timed out waiting for: {what}");
        std::thread::sleep(Duration::from_millis(2));
    }
}

#[test]
fn serial_runs_every_pending_merge_in_order_on_the_calling_thread() {
    let source = TestSource::with(&[("a", 1), ("b", 1), ("c", 1)]);
    source.release_everything();
    SerialMergeScheduler::new()
        .merge(&dyn_source(&source), MergeTrigger::Explicit)
        .unwrap();
    assert_eq!(source.ran(), ["a", "b", "c"]);
    let me = std::thread::current().id();
    assert!(source.ran.lock().unwrap().iter().all(|(_, t)| *t == me));
    assert!(SerialMergeScheduler::new().close().is_ok());

    // A failure stops the loop and is returned.
    let source = TestSource::with(&[("x", 1), ("y", 1)]);
    source.outcome.lock().unwrap().insert("x".into(), "err");
    source.release_everything();
    let err = SerialMergeScheduler::new()
        .merge(&dyn_source(&source), MergeTrigger::Explicit)
        .unwrap_err();
    assert!(err.to_string().contains("x failed"), "{err}");
    assert_eq!(source.ran(), ["x"]);
}

#[test]
fn no_merge_scheduler_leaves_merges_pending() {
    let source = TestSource::with(&[("a", 1)]);
    NoMergeScheduler
        .merge(&dyn_source(&source), MergeTrigger::Explicit)
        .unwrap();
    assert!(source.has_pending_merges());
    assert!(source.ran().is_empty());
}

#[test]
fn max_merges_and_threads_are_validated_as_java_does() {
    let cms = ConcurrentMergeScheduler::new();
    assert_eq!(cms.max_thread_count(), AUTO_DETECT_MERGES_AND_THREADS);
    for (merges, threads) in [(-1, 2), (2, -1), (3, 0), (0, 0), (2, 3)] {
        assert!(
            matches!(
                cms.set_max_merges_and_threads(merges, threads),
                Err(Error::InvalidMergeScheduler(_))
            ),
            "{merges}/{threads}"
        );
    }
    cms.set_max_merges_and_threads(4, 2).unwrap();
    assert_eq!((cms.max_merge_count(), cms.max_thread_count()), (4, 2));
    cms.set_max_merges_and_threads(-1, -1).unwrap();
    assert_eq!(cms.max_merge_count(), AUTO_DETECT_MERGES_AND_THREADS);

    cms.set_default_max_merges_and_threads(true);
    assert_eq!((cms.max_merge_count(), cms.max_thread_count()), (6, 1));
    let mut s = lock(&cms.inner.state);
    ConcurrentMergeScheduler::apply_defaults(&mut s, false, 8);
    assert_eq!((s.max_merge_count, s.max_thread_count), (9, 4));
    ConcurrentMergeScheduler::apply_defaults(&mut s, false, 1);
    assert_eq!((s.max_merge_count, s.max_thread_count), (6, 1));
    drop(s);
    assert!(format!("{cms:?}").contains("maxThreadCount=1, maxMergeCount=6"));

    // AUTO_DETECT is resolved on first use.
    cms.set_max_merges_and_threads(-1, -1).unwrap();
    let source = TestSource::with(&[]);
    cms.merge(&dyn_source(&source), MergeTrigger::Explicit)
        .unwrap();
    assert!(cms.max_thread_count() >= 1);
    assert_eq!(cms.max_merge_count(), cms.max_thread_count() + 5);
}

#[test]
fn merges_run_on_named_merge_threads_and_sync_waits_for_them() {
    let cms = ConcurrentMergeScheduler::new();
    cms.set_max_merges_and_threads(8, 4).unwrap();
    let source = TestSource::with(&[("a", 1), ("b", 1), ("c", 1)]);
    cms.merge(&dyn_source(&source), MergeTrigger::SegmentFlush)
        .unwrap();
    wait_until("three merges running", || {
        source.running.load(Ordering::SeqCst) == 3
    });
    assert_eq!(cms.merge_thread_count(), 3);
    let mut names: Vec<String> = cms.merge_thread_rates().into_iter().map(|t| t.0).collect();
    names.sort();
    assert_eq!(
        names,
        [
            "Lucene Merge Thread #0",
            "Lucene Merge Thread #1",
            "Lucene Merge Thread #2"
        ]
    );
    source.release_everything();
    cms.close().unwrap();
    assert_eq!(cms.merge_thread_count(), 0);
    let mut ran = source.ran();
    ran.sort();
    assert_eq!(ran, ["a", "b", "c"]);
    let me = std::thread::current().id();
    assert!(source.ran.lock().unwrap().iter().all(|(_, t)| *t != me));
}

/// `maybeStall`: with `max_merge_count` merges running and more pending, the
/// producing thread waits until a merge finishes. The merge thread that
/// finishes then starts the pending merge itself (`runOnMergeFinished`), and
/// is never stalled.
///
/// Seen to fail with the stall disabled (`maybe_stall` returning `(s, true)`
/// at once): the producer returns having started all three merges, and the
/// "producer is still stalled" assertion fails.
#[test]
fn a_producer_stalls_while_max_merge_count_merges_run() {
    let cms = ConcurrentMergeScheduler::new();
    cms.set_max_merges_and_threads(2, 1).unwrap();
    let source = TestSource::with(&[("a", 1), ("b", 1), ("c", 1)]);
    let returned = Arc::new(AtomicBool::new(false));
    let producer = {
        let (cms, source, returned) = (cms.clone(), dyn_source(&source), Arc::clone(&returned));
        std::thread::spawn(move || {
            cms.merge(&source, MergeTrigger::SegmentFlush).unwrap();
            returned.store(true, Ordering::SeqCst);
        })
    };
    wait_until("the producer to stall", || cms.stall_waits() > 0);
    wait_until("a and b running", || {
        source.running.load(Ordering::SeqCst) == 2
    });
    assert!(source.has_pending_merges(), "c must still be pending");
    assert!(
        !returned.load(Ordering::SeqCst),
        "the producer must still be stalled"
    );

    source.release("a");
    producer.join().unwrap();
    assert!(returned.load(Ordering::SeqCst));
    assert!(!source.has_pending_merges());
    source.release_everything();
    cms.sync().unwrap();
    let mut ran = source.ran();
    ran.sort();
    assert_eq!(ran, ["a", "b", "c"]);
}

/// `updateMergeThreads`: of the big merges running, only
/// `max_thread_count` proceed -- the largest others are stopped (rate 0) --
/// and a finished merge lets the next largest resume. Small merges are never
/// paused.
///
/// Seen to fail with pausing disabled (`pause_below = 0`): every rate is
/// unlimited.
#[test]
fn the_largest_big_merges_beyond_max_thread_count_are_paused() {
    let cms = ConcurrentMergeScheduler::new();
    cms.set_max_merges_and_threads(5, 1).unwrap();
    let source = TestSource::with(&[("m100", 100), ("m300", 300), ("m200", 200), ("small", 10)]);
    cms.merge(&dyn_source(&source), MergeTrigger::SegmentFlush)
        .unwrap();
    let rates = |cms: &ConcurrentMergeScheduler| {
        cms.merge_thread_rates()
            .into_iter()
            .map(|(_, bytes, rate)| (bytes / MB, rate))
            .collect::<Vec<_>>()
    };
    assert_eq!(
        rates(&cms),
        [
            (300, 0.0),
            (200, 0.0),
            (100, f64::INFINITY),
            (10, f64::INFINITY)
        ]
    );
    source.release("m100");
    wait_until("m100 to finish", || cms.merge_thread_rates().len() == 3);
    assert_eq!(
        rates(&cms),
        [(300, 0.0), (200, f64::INFINITY), (10, f64::INFINITY)]
    );
    source.release_everything();
    cms.sync().unwrap();
}

/// A controllable clock for the IO throttle.
fn clocked() -> (ConcurrentMergeScheduler, Arc<AtomicI64>) {
    let now = Arc::new(AtomicI64::new(1_000_000_000));
    let clock = Arc::clone(&now);
    let cms = ConcurrentMergeScheduler::with_clock(Box::new(move || clock.load(Ordering::SeqCst)));
    (cms, now)
}

/// `updateIOThrottle`'s closed loop: a big merge with no similar merge
/// running lowers the target by 10%; one with a similar merge running for
/// more than three seconds raises it by 20%; a small merge leaves it alone;
/// a forced merge runs at the force-merge rate. Bounded at 5 MB/s below.
#[test]
fn the_auto_io_throttle_follows_the_merge_backlog() {
    let (cms, now) = clocked();
    cms.set_max_merges_and_threads(10, 10).unwrap();
    assert_eq!(cms.io_rate_limit_mb_per_sec(), f64::INFINITY);
    cms.enable_auto_io_throttle();
    assert!(cms.auto_io_throttle());
    assert_eq!(cms.io_rate_limit_mb_per_sec(), 20.0);

    let source = TestSource::with(&[("a", 100)]);
    let dyn_src = dyn_source(&source);
    cms.merge(&dyn_src, MergeTrigger::SegmentFlush).unwrap();
    let after_a = 20.0 / 1.10;
    assert_eq!(cms.io_rate_limit_mb_per_sec(), after_a);
    wait_until("a to start", || source.running.load(Ordering::SeqCst) == 1);
    assert_eq!(cms.merge_thread_rates()[0].2, after_a);

    // A small merge does not move the target, and is not throttled.
    source.push(ScheduledMerge::new(vec!["tiny".into()], 5 * MB));
    cms.merge(&dyn_src, MergeTrigger::SegmentFlush).unwrap();
    assert_eq!(cms.io_rate_limit_mb_per_sec(), after_a);

    // Four seconds later a similar merge arrives: backlog, +20%.
    now.fetch_add(4_000_000_000, Ordering::SeqCst);
    source.push(ScheduledMerge::new(vec!["b".into()], 150 * MB));
    cms.merge(&dyn_src, MergeTrigger::SegmentFlush).unwrap();
    let after_b = after_a * 1.20;
    assert_eq!(cms.io_rate_limit_mb_per_sec(), after_b);
    // A dissimilar one (ratio outside 0.3..3) while those run: the others
    // are backlogged against each other, so the rate is left as it is.
    source.push(ScheduledMerge::new(vec!["huge".into()], 2000 * MB));
    cms.merge(&dyn_src, MergeTrigger::SegmentFlush).unwrap();
    assert_eq!(cms.io_rate_limit_mb_per_sec(), after_b);

    // A forced merge runs at the force-merge rate -- but still moves the
    // shared target, as Java's `updateIOThrottle` does: `a` is similar to it
    // and has run for four seconds, so that is backlog again.
    cms.set_force_merge_mb_per_sec(7.0);
    assert_eq!(cms.force_merge_mb_per_sec(), 7.0);
    source.push(ScheduledMerge::new(vec!["forced".into()], 60 * MB).forced(1));
    cms.merge(&dyn_src, MergeTrigger::Explicit).unwrap();
    let after_forced = after_b * 1.20;
    assert_eq!(cms.io_rate_limit_mb_per_sec(), after_forced);
    wait_until("five merges running", || {
        source.running.load(Ordering::SeqCst) == 5
    });
    let rates: HashMap<u64, f64> = cms
        .merge_thread_rates()
        .into_iter()
        .map(|(_, b, r)| (b / MB, r))
        .collect();
    assert_eq!(rates[&60], 7.0);
    assert_eq!(rates[&5], f64::INFINITY);
    assert_eq!(rates[&100], after_forced);

    // Off: everything unlimited again, forced merges excepted.
    cms.disable_auto_io_throttle();
    let rates: HashMap<u64, f64> = cms
        .merge_thread_rates()
        .into_iter()
        .map(|(_, b, r)| (b / MB, r))
        .collect();
    assert_eq!(rates[&100], f64::INFINITY);
    assert_eq!(rates[&60], 7.0);
    source.release_everything();
    cms.sync().unwrap();

    // The floor: lone big merges keep lowering it, to 5 MB/s and no further.
    let (cms, _now) = clocked();
    cms.set_max_merges_and_threads(10, 10).unwrap();
    cms.enable_auto_io_throttle();
    let source = TestSource::with(&[]);
    source.release_everything();
    let dyn_src = dyn_source(&source);
    for i in 0..30 {
        source.push(ScheduledMerge::new(vec![format!("m{i}")], 100 * MB));
        cms.merge(&dyn_src, MergeTrigger::SegmentFlush).unwrap();
        cms.sync().unwrap();
    }
    assert_eq!(cms.io_rate_limit_mb_per_sec(), MIN_MERGE_MB_PER_SEC);

    // Closing lifts the throttle to its ceiling.
    cms.merge(&dyn_src, MergeTrigger::Closing).unwrap();
    assert_eq!(cms.io_rate_limit_mb_per_sec(), MAX_MERGE_MB_PER_SEC);
}

#[test]
fn merge_failures_are_kept_and_returned_by_sync() {
    let cms = ConcurrentMergeScheduler::new();
    cms.set_max_merges_and_threads(4, 4).unwrap();
    let source = TestSource::with(&[("ok", 1), ("err", 1), ("panic", 1), ("abort", 1)]);
    {
        let mut o = source.outcome.lock().unwrap();
        o.insert("err".into(), "err");
        o.insert("panic".into(), "panic");
        o.insert("abort".into(), "abort");
    }
    source.release_everything();
    cms.merge(&dyn_source(&source), MergeTrigger::Explicit)
        .unwrap();
    let mut errors = Vec::new();
    while let Err(e) = cms.sync() {
        errors.push(e.to_string());
    }
    errors.sort();
    // An aborted merge is not a failure (`MergeAbortedException`).
    assert_eq!(errors.len(), 2, "{errors:?}");
    assert!(
        errors[0].contains("a merge thread panicked: panic blew up"),
        "{errors:?}"
    );
    assert!(errors[1].contains("err failed"), "{errors:?}");
    assert_eq!(source.ran().len(), 4);
}

/// Two indexes on one combined scheduler: `max_merge_count` counts the
/// merges of both, so a producer of one index stalls behind the other's
/// merge; and closing one index waits for its own merges only.
#[test]
fn indexes_sharing_a_combined_scheduler_stall_each_other_and_close_separately() {
    let combined = ConcurrentMergeScheduler::new();
    combined.set_max_merges_and_threads(1, 1).unwrap();
    let a = MultiIndexMergeScheduler::with_scheduler("index-a", combined.clone());
    let b = MultiIndexMergeScheduler::with_scheduler("index-b", combined.clone());
    assert_eq!(a.key(), "index-a");
    assert_eq!(
        a.combined_merge_scheduler().max_merge_count(),
        combined.max_merge_count()
    );
    let source_a = TestSource::with(&[("a1", 1)]);
    let source_b = TestSource::with(&[("b1", 1)]);
    a.merge(&dyn_source(&source_a), MergeTrigger::SegmentFlush)
        .unwrap();
    wait_until("a1 running", || {
        source_a.running.load(Ordering::SeqCst) == 1
    });

    let b_done = Arc::new(AtomicBool::new(false));
    let producer_b = {
        let (b_src, done) = (dyn_source(&source_b), Arc::clone(&b_done));
        let b = MultiIndexMergeScheduler::with_scheduler("index-b", combined.clone());
        std::thread::spawn(move || {
            b.merge(&b_src, MergeTrigger::SegmentFlush).unwrap();
            done.store(true, Ordering::SeqCst);
        })
    };
    wait_until("b's producer to stall behind a's merge", || {
        combined.stall_waits() > 0
    });
    assert!(!b_done.load(Ordering::SeqCst));
    source_a.release("a1");
    producer_b.join().unwrap();
    wait_until("b1 running", || {
        source_b.running.load(Ordering::SeqCst) == 1
    });

    // Closing a does not wait for b's merge.
    a.close().unwrap();
    assert_eq!(source_b.running.load(Ordering::SeqCst), 1);
    source_b.release("b1");
    b.close().unwrap();
    assert_eq!(source_b.ran(), ["b1"]);
}

#[test]
fn the_process_wide_combined_scheduler_lives_as_long_as_its_users() {
    assert!(MultiIndexMergeScheduler::peek_singleton().is_none());
    let x = MultiIndexMergeScheduler::new("x");
    let y = MultiIndexMergeScheduler::new("y");
    let shared = MultiIndexMergeScheduler::peek_singleton().expect("created by the first user");
    shared.set_max_merges_and_threads(3, 2).unwrap();
    assert_eq!(y.combined_merge_scheduler().max_thread_count(), 2);
    let source = TestSource::with(&[("m", 1)]);
    source.release_everything();
    x.merge(&dyn_source(&source), MergeTrigger::Explicit)
        .unwrap();
    x.close().unwrap();
    assert!(MultiIndexMergeScheduler::peek_singleton().is_some());
    y.close().unwrap();
    assert!(MultiIndexMergeScheduler::peek_singleton().is_none());
    assert_eq!(source.ran(), ["m"]);
}

#[test]
fn a_scheduled_merge_reports_its_shape() {
    let m = ScheduledMerge::new(vec!["_0".into(), "_1".into()], 3 * MB).forced(2);
    assert_eq!(m.max_num_segments, 2);
    assert!(!m.is_aborted());
    m.abort();
    assert!(m.is_aborted() && m.progress().is_aborted());
    let debug = format!("{m:?}");
    assert!(
        debug.contains("aborted: true") && debug.contains("_1"),
        "{debug}"
    );
    let q = PendingMerges::new();
    assert!(q.is_empty());
    q.push(Arc::new(m));
    assert!(!q.is_empty());
    assert!(q.pop().is_some() && q.pop().is_none());
    assert_eq!(
        panic_message(&(Box::new(7u8) as Box<dyn std::any::Any + Send>)),
        "non-string panic payload"
    );
}
