//! `SuppressingConcurrentMergeScheduler`: a [`ConcurrentMergeScheduler`]
//! that swallows the merge failures a test expects.
//!
//! Differs: Java overrides `handleMergeException`, which the port's
//! scheduler has no hook for (a merge thread's failure is kept and handed
//! back by `sync`/`close`, and a failure to start a merge by `merge`). The
//! wrapper filters those instead: a failure whose error, or any error in
//! its `source()` chain, `is_ok` accepts is dropped and counted; any other
//! is returned as Java rethrows it. Java's `abstract isOK` is the `is_ok`
//! function passed to [`SuppressingConcurrentMergeScheduler::new`].

use std::error::Error as StdError;
use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use lucene_index::index_writer::{Error, Result};
use lucene_index::merge_policy::MergeTrigger;
use lucene_index::merge_scheduler::{ConcurrentMergeScheduler, MergeScheduler, MergeSource};

/// `isOK(Throwable)`: whether one error of a failure's cause chain is one
/// the test expects.
pub type IsOk = fn(&(dyn StdError + 'static)) -> bool;

/// `SuppressingConcurrentMergeScheduler`.
pub struct SuppressingConcurrentMergeScheduler {
    inner: ConcurrentMergeScheduler,
    is_ok: IsOk,
    suppressed: AtomicUsize,
}

impl fmt::Debug for SuppressingConcurrentMergeScheduler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SuppressingConcurrentMergeScheduler")
            .field("inner", &self.inner)
            .field("suppressed", &self.suppressed())
            .finish()
    }
}

impl SuppressingConcurrentMergeScheduler {
    /// A scheduler over a new [`ConcurrentMergeScheduler`] suppressing what
    /// `is_ok` accepts.
    pub fn new(is_ok: IsOk) -> Self {
        Self::wrapping(ConcurrentMergeScheduler::new(), is_ok)
    }

    /// The same over `inner` (configured by the caller).
    pub fn wrapping(inner: ConcurrentMergeScheduler, is_ok: IsOk) -> Self {
        SuppressingConcurrentMergeScheduler {
            inner,
            is_ok,
            suppressed: AtomicUsize::new(0),
        }
    }

    /// The scheduler underneath.
    pub fn inner(&self) -> &ConcurrentMergeScheduler {
        &self.inner
    }

    /// How many failures were suppressed so far.
    pub fn suppressed(&self) -> usize {
        self.suppressed.load(Ordering::SeqCst)
    }

    /// `handleMergeException(exc)`: walks the cause chain; `Ok` (counted)
    /// when any cause is OK, else the failure.
    fn handle(&self, e: Error) -> Result<()> {
        let mut cause: Option<&(dyn StdError + 'static)> = Some(&e);
        while let Some(c) = cause {
            if (self.is_ok)(c) {
                self.suppressed.fetch_add(1, Ordering::SeqCst);
                return Ok(());
            }
            cause = c.source();
        }
        Err(e)
    }

    /// `sync()`: waits for every merge, suppressing what is OK and
    /// returning the first failure that is not.
    pub fn sync(&self) -> Result<()> {
        let mut first = None;
        // Each `sync` hands back (and forgets) one kept failure.
        while let Err(e) = self.inner.sync() {
            if let Err(e) = self.handle(e) {
                first.get_or_insert(e);
            }
        }
        first.map_or(Ok(()), Err)
    }
}

impl MergeScheduler for SuppressingConcurrentMergeScheduler {
    fn merge(&self, source: &Arc<dyn MergeSource>, trigger: MergeTrigger) -> Result<()> {
        self.inner
            .merge(source, trigger)
            .or_else(|e| self.handle(e))
    }

    fn close(&self) -> Result<()> {
        self.sync()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::Mutex;

    use lucene_index::merge_rate_limiter::MergeRateLimiter;
    use lucene_index::merge_scheduler::ScheduledMerge;

    use super::*;

    /// A source whose merges fail with the queued errors, in order.
    struct Failing {
        pending: Mutex<VecDeque<Arc<ScheduledMerge>>>,
        errors: Mutex<VecDeque<Error>>,
    }

    impl Failing {
        fn source(errors: Vec<Error>) -> Arc<dyn MergeSource> {
            // One merge more than the errors: the last one succeeds.
            let pending = (0..=errors.len())
                .map(|i| Arc::new(ScheduledMerge::new(vec![format!("_{i}")], 1)))
                .collect();
            Arc::new(Failing {
                pending: Mutex::new(pending),
                errors: Mutex::new(errors.into()),
            })
        }
    }

    impl MergeSource for Failing {
        fn next_merge(&self) -> Option<Arc<ScheduledMerge>> {
            self.pending.lock().unwrap().pop_front()
        }
        fn on_merge_finished(&self, _merge: &Arc<ScheduledMerge>) {}
        fn has_pending_merges(&self) -> bool {
            !self.pending.lock().unwrap().is_empty()
        }
        fn merge(
            &self,
            _merge: &Arc<ScheduledMerge>,
            _limiter: Option<Arc<MergeRateLimiter>>,
        ) -> Result<()> {
            match self.errors.lock().unwrap().pop_front() {
                Some(e) => Err(e),
                None => Ok(()),
            }
        }
    }

    fn closed() -> Error {
        Error::Store(lucene_store::Error::AlreadyClosed(
            "this writer is closed".into(),
        ))
    }

    fn other() -> Error {
        Error::Store(lucene_store::Error::Io(std::io::Error::other(
            "disk on fire",
        )))
    }

    /// `isOK`: an `AlreadyClosedException`, the writer's or a cause's.
    fn already_closed(e: &(dyn StdError + 'static)) -> bool {
        let store = match e.downcast_ref::<Error>() {
            Some(Error::Store(s)) => Some(s),
            Some(_) => None,
            None => e.downcast_ref::<lucene_store::Error>(),
        };
        matches!(store, Some(lucene_store::Error::AlreadyClosed(_)))
    }

    /// An I/O failure caused by `source`.
    #[derive(Debug)]
    struct CausedBy(lucene_store::Error);

    impl fmt::Display for CausedBy {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("merge failed")
        }
    }

    impl StdError for CausedBy {
        fn source(&self) -> Option<&(dyn StdError + 'static)> {
            Some(&self.0)
        }
    }

    #[test]
    fn ok_failures_are_suppressed_and_counted() {
        let cms = SuppressingConcurrentMergeScheduler::new(already_closed);
        let source = Failing::source(vec![closed(), closed()]);
        cms.merge(&source, MergeTrigger::Explicit).unwrap();
        // Gate rule #10: both failures reached the filter.
        cms.close().unwrap();
        assert_eq!(2, cms.suppressed());
        assert!(!source.has_pending_merges());
        source.on_merge_finished(&Arc::new(ScheduledMerge::new(Vec::new(), 0)));
        assert!(format!("{cms:?}").contains("suppressed: 2"));
    }

    #[test]
    fn other_failures_are_rethrown() {
        let cms = SuppressingConcurrentMergeScheduler::new(already_closed);
        let source = Failing::source(vec![other(), closed()]);
        cms.merge(&source, MergeTrigger::Explicit).unwrap();
        let e = cms.close().expect_err("a disk failure is not OK");
        assert!(e.to_string().contains("disk on fire"), "{e}");
        assert_eq!(1, cms.suppressed());
        // Forgotten once returned.
        cms.sync().unwrap();
    }

    #[test]
    fn the_cause_chain_is_walked() {
        // The writer's error is an I/O failure; what caused it is the
        // `AlreadyClosed` the filter accepts, one level down.
        let cause = lucene_store::Error::AlreadyClosed("x".into());
        let wrapped = Error::Store(lucene_store::Error::Io(std::io::Error::other(CausedBy(
            cause,
        ))));
        assert!(!already_closed(&wrapped));
        assert!(wrapped.to_string().contains("merge failed"), "{wrapped}");
        assert!(!already_closed(&Error::AddIndexes("x".into())));
        let cms = SuppressingConcurrentMergeScheduler::wrapping(
            ConcurrentMergeScheduler::new(),
            already_closed,
        );
        assert!(cms.handle(wrapped).is_ok());
        assert!(cms.handle(other()).is_err());
        assert_eq!(1, cms.suppressed());
        assert_eq!(cms.inner().merge_thread_count(), 0);
    }
}
