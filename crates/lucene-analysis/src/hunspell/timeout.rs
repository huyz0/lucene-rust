//! `TimeoutPolicy`, `SuggestionTimeoutException` and the `checkCanceled`
//! runnable Lucene threads through spell checking and suggestion.
//!
//! Java stops a computation by throwing from `checkCanceled.run()`: the
//! caller's hook throws to cancel, and `Suggester.checkTimeLimit` throws a
//! `SuggestionTimeoutException` (carrying the suggestions found so far) once
//! its deadline has passed, looking at the clock every 100th call. The port
//! has no exceptions to unwind with, so a [`Canceler`] latches instead: once
//! it has said "stop", every later check says so too, each check site
//! returns at once (a speller check answers "not found"), and the suggestion
//! set refuses further changes -- so what the caller gets is exactly the set
//! as it stood when Java would have thrown.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering::Relaxed};
use std::time::{Duration, Instant};

/// `org.apache.lucene.analysis.hunspell.TimeoutPolicy`: what
/// [`Hunspell::suggest`](super::Hunspell::suggest) does when it runs out of
/// time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeoutPolicy {
    /// `NO_TIMEOUT`: let the computation complete even if it takes ages.
    NoTimeout,
    /// `RETURN_PARTIAL_RESULT`: stop, and return what has been computed so
    /// far.
    ReturnPartialResult,
    /// `THROW_EXCEPTION`: stop, and report a [`SuggestionTimeout`].
    ThrowException,
}

/// `Hunspell.SUGGEST_TIME_LIMIT`: 250 ms.
pub const SUGGEST_TIME_LIMIT: Duration = Duration::from_millis(250);

/// Java's `Runnable checkCanceled`: called periodically while spelling or
/// suggesting; returning `true` cancels the computation (where Java's hook
/// throws). A canceled `spell` answers `false`; a canceled suggestion is a
/// [`SuggestionTimeout`] with the suggestions found so far.
pub type CheckCanceled<'h> = &'h (dyn Fn() -> bool + Sync);

/// `org.apache.lucene.analysis.hunspell.SuggestionTimeoutException`: the
/// suggestion ran out of time (or was canceled) and
/// [`partial_result`](Self::partial_result) is what it had found.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct SuggestionTimeout {
    message: String,
    partial_result: Vec<String>,
}

impl SuggestionTimeout {
    pub(crate) fn new(message: String, partial_result: Vec<String>) -> Self {
        SuggestionTimeout {
            message,
            partial_result,
        }
    }

    /// `getMessage()`: `Time limit of <n>ms exceeded for <word>`, or
    /// `canceled` when the caller's hook stopped it.
    pub fn message(&self) -> &str {
        &self.message
    }

    /// `getPartialResult()`: the suggestions computed before the stop.
    pub fn partial_result(&self) -> &[String] {
        &self.partial_result
    }

    /// The suggestions computed before the stop, by value.
    pub fn into_partial_result(self) -> Vec<String> {
        self.partial_result
    }
}

/// `checkTimeLimit`'s `invocationCounter` start: the clock is read every
/// 100th check.
const CHECKS_PER_CLOCK_READ: u32 = 100;

/// Why a [`Canceler`] stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Stop {
    /// The caller's hook returned `true`.
    Canceled,
    /// The deadline passed.
    TimedOut,
}

/// One computation's `checkCanceled`: the caller's hook, an optional
/// deadline, and the latch both set. Atomics (relaxed, never contended: one
/// computation runs on one thread) keep [`Hunspell`](super::Hunspell)
/// `Sync`.
pub(crate) struct Canceler<'h> {
    hook: Option<CheckCanceled<'h>>,
    deadline: Option<Instant>,
    countdown: AtomicU32,
    stopped: AtomicBool,
    timed_out: AtomicBool,
}

impl<'h> Canceler<'h> {
    /// A canceler calling `hook` on every check and, with a `time_limit`,
    /// stopping once that much time has passed from now.
    pub(crate) fn new(hook: Option<CheckCanceled<'h>>, time_limit: Option<Duration>) -> Self {
        Canceler {
            hook,
            deadline: time_limit.and_then(|t| Instant::now().checked_add(t)),
            countdown: AtomicU32::new(CHECKS_PER_CLOCK_READ),
            stopped: AtomicBool::new(false),
            timed_out: AtomicBool::new(false),
        }
    }

    /// `checkCanceled.run()`: whether the computation must stop now.
    pub(crate) fn check(&self) -> bool {
        if self.stopped.load(Relaxed) {
            return true;
        }
        if self.hook.is_some_and(|hook| hook()) {
            self.stopped.store(true, Relaxed);
            return true;
        }
        if let Some(deadline) = self.deadline {
            let left = self.countdown.load(Relaxed).saturating_sub(1);
            if left > 0 {
                self.countdown.store(left, Relaxed);
            } else {
                self.countdown.store(CHECKS_PER_CLOCK_READ, Relaxed);
                if Instant::now() > deadline {
                    self.timed_out.store(true, Relaxed);
                    self.stopped.store(true, Relaxed);
                    return true;
                }
            }
        }
        false
    }

    /// Whether an earlier check said "stop", without checking again.
    pub(crate) fn stopped(&self) -> Option<Stop> {
        if !self.stopped.load(Relaxed) {
            None
        } else if self.timed_out.load(Relaxed) {
            Some(Stop::TimedOut)
        } else {
            Some(Stop::Canceled)
        }
    }
}

impl std::fmt::Debug for Canceler<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Canceler")
            .field("hook", &self.hook.is_some())
            .field("deadline", &self.deadline)
            .field("stopped", &self.stopped())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_deadline_is_read_every_100th_check_and_latches() {
        let c = Canceler::new(None, Some(Duration::ZERO));
        for _ in 1..CHECKS_PER_CLOCK_READ {
            assert!(!c.check());
        }
        assert!(c.check());
        assert_eq!(c.stopped(), Some(Stop::TimedOut));
        assert!(c.check());
        assert!(format!("{c:?}").contains("TimedOut"));

        let never = Canceler::new(None, None);
        for _ in 0..1000 {
            assert!(!never.check());
        }
        assert_eq!(never.stopped(), None);
        // A limit past `Instant`'s range is no deadline.
        let far = Canceler::new(None, Some(Duration::MAX));
        assert!(format!("{far:?}").contains("deadline: None"));

        let hook = || true;
        let c = Canceler::new(Some(&hook), Some(Duration::from_secs(60)));
        assert!(c.check());
        assert_eq!(c.stopped(), Some(Stop::Canceled));
    }

    #[test]
    fn a_timeout_carries_its_partial_result() {
        let e = SuggestionTimeout::new("m".into(), vec!["a".into()]);
        assert_eq!(
            (e.message(), e.partial_result()),
            ("m", &["a".to_string()][..])
        );
        assert_eq!(e.into_partial_result(), ["a"]);
    }
}
