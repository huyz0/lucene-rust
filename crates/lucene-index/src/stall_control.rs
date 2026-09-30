//! Port of `org.apache.lucene.index.DocumentsWriterStallControl`: holds
//! indexing threads back while flushing lags behind indexing, so buffered
//! RAM cannot grow without bound.
//!
//! [`crate::concurrent_writer::ConcurrentIndexWriter`] decides when to stall
//! (`DocumentsWriterFlushControl.updateStallState`) and waits here before
//! each add (`DocumentsWriter.preUpdate`). Java keeps its waiter bookkeeping
//! under `assert` only; here it is always kept, since it is two counters.
//! `isThreadQueued(Thread)` is not ported (a test hook keyed by thread
//! identity; [`DocumentsWriterStallControl::num_waiting`] serves the same
//! tests).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard};
use std::time::Duration;

#[derive(Debug, Default)]
struct State {
    /// `numWaiting`.
    num_waiting: usize,
    /// `wasStalled`.
    was_stalled: bool,
}

/// `DocumentsWriterStallControl`.
#[derive(Debug, Default)]
pub struct DocumentsWriterStallControl {
    /// `volatile boolean stalled`: read without the lock on the fast path.
    stalled: AtomicBool,
    state: Mutex<State>,
    cond: Condvar,
}

/// How long [`DocumentsWriterStallControl::wait_if_stalled`] waits at most
/// (Java's defensive `wait(1000)`): the caller re-checks and waits again if
/// still stalled.
const MAX_WAIT: Duration = Duration::from_secs(1);

impl DocumentsWriterStallControl {
    pub fn new() -> Self {
        Self::default()
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// `updateStalled(stalled)`: sets the flag, and on any change wakes every
    /// waiting thread.
    pub fn update_stalled(&self, stalled: bool) {
        let mut state = self.lock();
        if self.stalled.load(Ordering::SeqCst) != stalled {
            self.stalled.store(stalled, Ordering::SeqCst);
            if stalled {
                state.was_stalled = true;
            }
            self.cond.notify_all();
        }
    }

    /// `waitIfStalled()`: blocks while stalled -- until the first wakeup, or
    /// at most a second. It does not loop: the caller re-checks, and the
    /// stall is re-established by whoever set it if it is still due.
    pub fn wait_if_stalled(&self) {
        if !self.stalled.load(Ordering::SeqCst) {
            return;
        }
        let mut state = self.lock();
        if self.stalled.load(Ordering::SeqCst) {
            state.num_waiting = state.num_waiting.saturating_add(1);
            state = match self.cond.wait_timeout(state, MAX_WAIT) {
                Ok((s, _)) => s,
                Err(poisoned) => poisoned.into_inner().0,
            };
            state.num_waiting = state.num_waiting.saturating_sub(1);
        }
    }

    /// `anyStalledThreads()`: whether indexing is stalled right now.
    pub fn any_stalled_threads(&self) -> bool {
        self.stalled.load(Ordering::SeqCst)
    }

    /// `hasBlocked()`: whether any thread is waiting.
    pub fn has_blocked(&self) -> bool {
        self.lock().num_waiting > 0
    }

    /// `getNumWaiting()`.
    pub fn num_waiting(&self) -> usize {
        self.lock().num_waiting
    }

    /// `isHealthy()`: not stalled.
    pub fn is_healthy(&self) -> bool {
        !self.stalled.load(Ordering::SeqCst)
    }

    /// `wasStalled()`: whether it has ever stalled.
    pub fn was_stalled(&self) -> bool {
        self.lock().was_stalled
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::Instant;

    /// Waits (bounded) until `cond` holds.
    fn eventually(cond: impl Fn() -> bool) {
        let start = Instant::now();
        while !cond() {
            assert!(
                start.elapsed() < Duration::from_secs(10),
                "condition never held"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    #[test]
    fn a_healthy_control_never_blocks() {
        let ctrl = DocumentsWriterStallControl::new();
        assert!(ctrl.is_healthy());
        ctrl.update_stalled(false);
        ctrl.wait_if_stalled();
        assert!(!ctrl.any_stalled_threads());
        assert!(!ctrl.has_blocked());
        assert!(!ctrl.was_stalled());
    }

    /// `TestDocumentsWriterStallControl.testSimpleStall`: stalled threads
    /// wait until the stall is lifted, and then all go.
    #[test]
    fn stalled_threads_wait_until_the_stall_is_lifted() {
        let ctrl = Arc::new(DocumentsWriterStallControl::new());
        ctrl.update_stalled(true);
        assert!(ctrl.any_stalled_threads());
        assert!(!ctrl.is_healthy());
        assert!(ctrl.was_stalled());
        let threads: Vec<_> = (0..4)
            .map(|_| {
                let ctrl = ctrl.clone();
                std::thread::spawn(move || ctrl.wait_if_stalled())
            })
            .collect();
        eventually(|| ctrl.num_waiting() == 4);
        assert!(ctrl.has_blocked());
        ctrl.update_stalled(false);
        for t in threads {
            t.join().unwrap();
        }
        assert_eq!(ctrl.num_waiting(), 0);
        assert!(ctrl.is_healthy());
        assert!(ctrl.was_stalled(), "wasStalled is sticky");
    }

    /// A stall nobody lifts releases a waiter after the bounded wait, so a
    /// lost wakeup cannot hang a thread forever.
    #[test]
    fn a_waiter_gives_up_after_a_second() {
        let ctrl = DocumentsWriterStallControl::new();
        ctrl.update_stalled(true);
        let start = Instant::now();
        ctrl.wait_if_stalled();
        assert!(start.elapsed() >= Duration::from_millis(900));
        assert!(
            ctrl.any_stalled_threads(),
            "still stalled; the caller re-checks"
        );
    }
}
