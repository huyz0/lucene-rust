//! The bound on ranged requests in flight over every reader of one
//! [`crate::PageCache`]: at most so many requests, and so many bytes of
//! them, at once. A request takes a [`FetchPermit`] before it is sent and
//! gives it back when its pages have been cut from its buffer, so a node's
//! fetch threads, open connections and fetch buffers stay bounded however
//! many queries miss at once, and each query still sends its own requests
//! in parallel up to the bound.
//!
//! A permit is taken by the thread that will send the request, one at a
//! time, and held only while that request runs, so a reader waiting for a
//! permit never holds one another reader waits for: every wait ends when
//! some running request ends.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Condvar, Mutex};

use crate::lock::lock_ignoring_poison;

/// Requests and bytes in flight now.
#[derive(Default)]
struct InFlight {
    requests: usize,
    bytes: u64,
}

/// The in-flight bound of one cache.
pub(crate) struct FetchBudget {
    max_requests: usize,
    max_bytes: u64,
    now: Mutex<InFlight>,
    freed: Condvar,
    waits: AtomicU64,
}

/// One request's share of a [`FetchBudget`]; dropping it gives it back.
pub(crate) struct FetchPermit<'a> {
    budget: &'a FetchBudget,
    bytes: u64,
}

impl FetchBudget {
    /// A budget of `max_requests` requests and `max_bytes` bytes in flight,
    /// each at least 1.
    pub(crate) fn new(max_requests: usize, max_bytes: u64) -> FetchBudget {
        FetchBudget {
            max_requests: max_requests.max(1),
            max_bytes: max_bytes.max(1),
            now: Mutex::new(InFlight::default()),
            freed: Condvar::new(),
            waits: AtomicU64::new(0),
        }
    }

    /// A permit for one request of `bytes`, once one more request and
    /// `bytes` more fit; a request larger than the whole byte budget is
    /// counted as the whole budget, so it runs alone rather than never.
    pub(crate) fn acquire(&self, bytes: u64) -> FetchPermit<'_> {
        let bytes = bytes.min(self.max_bytes);
        let mut now = lock_ignoring_poison(&self.now);
        let fits = |n: &InFlight| {
            n.requests < self.max_requests && n.bytes.saturating_add(bytes) <= self.max_bytes
        };
        if !fits(&now) {
            self.waits.fetch_add(1, Ordering::Relaxed);
            while !fits(&now) {
                now = self
                    .freed
                    .wait(now)
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
            }
        }
        now.requests = now.requests.saturating_add(1);
        now.bytes = now.bytes.saturating_add(bytes);
        FetchPermit {
            budget: self,
            bytes,
        }
    }

    /// Requests that waited for a permit so far.
    pub(crate) fn waits(&self) -> u64 {
        self.waits.load(Ordering::Relaxed)
    }

    /// Requests and bytes in flight now.
    #[cfg(test)]
    fn in_flight(&self) -> (usize, u64) {
        let now = lock_ignoring_poison(&self.now);
        (now.requests, now.bytes)
    }
}

impl Drop for FetchPermit<'_> {
    fn drop(&mut self) {
        let mut now = lock_ignoring_poison(&self.budget.now);
        now.requests = now.requests.saturating_sub(1);
        now.bytes = now.bytes.saturating_sub(self.bytes);
        drop(now);
        // Waiters ask for different sizes: wake them all, each rechecks.
        self.budget.freed.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn permits_count_requests_and_bytes_and_give_them_back() {
        let budget = FetchBudget::new(2, 100);
        let a = budget.acquire(30);
        let b = budget.acquire(70);
        assert_eq!(budget.in_flight(), (2, 100));
        drop(a);
        assert_eq!(budget.in_flight(), (1, 70));
        drop(b);
        assert_eq!(budget.in_flight(), (0, 0));
        assert_eq!(budget.waits(), 0);
    }

    #[test]
    fn a_request_past_the_bound_waits_until_one_ends() {
        let budget = FetchBudget::new(1, 100);
        let held = budget.acquire(10);
        std::thread::scope(|s| {
            let waiter = s.spawn(|| {
                let _p = budget.acquire(10);
                budget.in_flight()
            });
            // The waiter cannot get in while `held` is out.
            while budget.waits() == 0 {
                std::thread::sleep(Duration::from_millis(1));
            }
            assert_eq!(budget.in_flight(), (1, 10));
            drop(held);
            assert_eq!(waiter.join().unwrap(), (1, 10));
        });
        assert_eq!(budget.in_flight(), (0, 0));
        assert_eq!(budget.waits(), 1);
    }

    #[test]
    fn bytes_bound_requests_and_an_oversized_one_runs_alone() {
        let budget = FetchBudget::new(10, 100);
        let held = budget.acquire(60);
        std::thread::scope(|s| {
            let waiter = s.spawn(|| {
                // 60 + 50 > 100: waits for the first.
                let _p = budget.acquire(50);
                budget.in_flight()
            });
            while budget.waits() == 0 {
                std::thread::sleep(Duration::from_millis(1));
            }
            drop(held);
            assert_eq!(waiter.join().unwrap(), (1, 50));
        });
        let big = budget.acquire(1_000);
        assert_eq!(budget.in_flight(), (1, 100));
        drop(big);
        assert_eq!(budget.in_flight(), (0, 0));
    }

    #[test]
    fn zero_limits_still_admit_one_request() {
        let budget = FetchBudget::new(0, 0);
        let p = budget.acquire(5);
        assert_eq!(budget.in_flight(), (1, 1));
        drop(p);
    }
}
