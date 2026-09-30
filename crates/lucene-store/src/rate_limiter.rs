//! Port of `org.apache.lucene.store.RateLimiter` (with its
//! `SimpleRateLimiter`) and `RateLimitedIndexOutput`: throttling of write
//! I/O, which `ConcurrentMergeScheduler` applies to merge outputs so a big
//! merge cannot starve searches of disk bandwidth.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::data_output::DataOutput;
use crate::index_output::IndexOutput;
use crate::lock::lock_ignoring_poison;

/// Port of `RateLimiter`: pauses a writer to hold it to a rate.
pub trait RateLimiter: Send + Sync {
    /// `setMBPerSec(mbPerSec)`.
    fn set_mb_per_sec(&self, mb_per_sec: f64);

    /// `getMBPerSec()`.
    fn mb_per_sec(&self) -> f64;

    /// `pause(bytes)`: blocks long enough that writing `bytes` stays within
    /// the rate. Returns the nanoseconds actually paused.
    fn pause(&self, bytes: u64) -> u64;

    /// `getMinPauseCheckBytes()`: how many bytes a caller may write between
    /// calls to [`RateLimiter::pause`].
    fn min_pause_check_bytes(&self) -> u64;
}

/// `SimpleRateLimiter.MIN_PAUSE_CHECK_MSEC`.
const MIN_PAUSE_CHECK_MSEC: f64 = 5.0;

/// Port of `RateLimiter.SimpleRateLimiter`: an instantaneous rate -- each
/// pause is measured from where the previous one left off, not averaged over
/// history.
#[derive(Debug)]
pub struct SimpleRateLimiter {
    /// `f64` bits: Java's `volatile double mbPerSec`.
    mb_per_sec: AtomicU64,
    min_pause_check_bytes: AtomicU64,
    /// `lastNS`, relative to `origin` (Java's `System.nanoTime()` has an
    /// arbitrary origin too).
    last: Mutex<Duration>,
    origin: Instant,
}

impl SimpleRateLimiter {
    /// `new SimpleRateLimiter(mbPerSec)`.
    pub fn new(mb_per_sec: f64) -> Self {
        let limiter = Self {
            mb_per_sec: AtomicU64::new(0),
            min_pause_check_bytes: AtomicU64::new(0),
            last: Mutex::new(Duration::ZERO),
            origin: Instant::now(),
        };
        limiter.set_mb_per_sec(mb_per_sec);
        limiter
    }

    fn now(&self) -> Duration {
        self.origin.elapsed()
    }
}

impl RateLimiter for SimpleRateLimiter {
    fn set_mb_per_sec(&self, mb_per_sec: f64) {
        self.mb_per_sec
            .store(mb_per_sec.to_bits(), Ordering::Release);
        // `(long) ((MIN_PAUSE_CHECK_MSEC / 1000.0) * mbPerSec * 1024 * 1024)`.
        let bytes = (MIN_PAUSE_CHECK_MSEC / 1000.0) * mb_per_sec * 1024.0 * 1024.0;
        self.min_pause_check_bytes
            .store(bytes as u64, Ordering::Release);
    }

    fn mb_per_sec(&self) -> f64 {
        f64::from_bits(self.mb_per_sec.load(Ordering::Acquire))
    }

    fn min_pause_check_bytes(&self) -> u64 {
        self.min_pause_check_bytes.load(Ordering::Acquire)
    }

    fn pause(&self, bytes: u64) -> u64 {
        let start = self.now();
        let seconds_to_pause = (bytes as f64 / 1024.0 / 1024.0) / self.mb_per_sec();
        let target = {
            let mut last = lock_ignoring_poison(&self.last);
            // The time we should sleep until: purely the instantaneous rate,
            // seconds added onto where the last pause ended.
            let pause = Duration::try_from_secs_f64(seconds_to_pause).unwrap_or(Duration::MAX);
            let target = last.saturating_add(pause);
            if start >= target {
                // Already past the target: no pausing. Set to `start`, not
                // `target`, to enforce the instant rate rather than the
                // average over all history.
                *last = start;
                return 0;
            }
            *last = target;
            target
        };
        // A loop because a sleep can end early.
        let mut cur = start;
        while cur < target {
            std::thread::sleep(target.saturating_sub(cur));
            cur = self.now();
        }
        u64::try_from(cur.saturating_sub(start).as_nanos()).unwrap_or(u64::MAX)
    }
}

/// Port of `RateLimitedIndexOutput`: an [`IndexOutput`] that calls
/// [`RateLimiter::pause`] every `min_pause_check_bytes` written.
///
/// As in Java, one `write_bytes` call is never split: a large array is
/// written without pauses after the check for it (LUCENE-10448).
pub struct RateLimitedIndexOutput<'a, O: IndexOutput> {
    out: O,
    rate_limiter: &'a dyn RateLimiter,
    bytes_since_last_pause: u64,
    current_min_pause_check_bytes: u64,
}

impl<'a, O: IndexOutput> RateLimitedIndexOutput<'a, O> {
    /// `new RateLimitedIndexOutput(rateLimiter, out)`.
    pub fn new(rate_limiter: &'a dyn RateLimiter, out: O) -> Self {
        Self {
            current_min_pause_check_bytes: rate_limiter.min_pause_check_bytes(),
            out,
            rate_limiter,
            bytes_since_last_pause: 0,
        }
    }

    /// The wrapped output, for closing it (`FilterIndexOutput.close`).
    pub fn into_inner(self) -> O {
        self.out
    }

    /// `checkRate()`.
    fn check_rate(&mut self, bytes: usize) {
        self.bytes_since_last_pause = self.bytes_since_last_pause.saturating_add(bytes as u64);
        if self.bytes_since_last_pause > self.current_min_pause_check_bytes {
            self.rate_limiter.pause(self.bytes_since_last_pause);
            self.bytes_since_last_pause = 0;
            self.current_min_pause_check_bytes = self.rate_limiter.min_pause_check_bytes();
        }
    }
}

impl<O: IndexOutput> DataOutput for RateLimitedIndexOutput<'_, O> {
    fn write_byte(&mut self, b: u8) {
        self.check_rate(1);
        self.out.write_byte(b);
    }

    fn write_bytes(&mut self, b: &[u8]) {
        self.check_rate(b.len());
        self.out.write_bytes(b);
    }

    fn write_i16(&mut self, v: i16) {
        self.check_rate(2);
        self.out.write_i16(v);
    }

    fn write_i32(&mut self, v: i32) {
        self.check_rate(4);
        self.out.write_i32(v);
    }

    fn write_i64(&mut self, v: i64) {
        self.check_rate(8);
        self.out.write_i64(v);
    }
}

impl<O: IndexOutput> IndexOutput for RateLimitedIndexOutput<'_, O> {
    fn name(&self) -> &str {
        self.out.name()
    }

    fn file_pointer(&self) -> u64 {
        self.out.file_pointer()
    }

    fn checksum(&self) -> u64 {
        self.out.checksum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ByteBuffersDirectory, Directory};

    /// Records every pause instead of sleeping.
    #[derive(Default)]
    struct Recording {
        pauses: Mutex<Vec<u64>>,
        check: AtomicU64,
    }

    impl RateLimiter for Recording {
        fn set_mb_per_sec(&self, _mb_per_sec: f64) {}
        fn mb_per_sec(&self) -> f64 {
            1.0
        }
        fn pause(&self, bytes: u64) -> u64 {
            self.pauses.lock().unwrap().push(bytes);
            0
        }
        fn min_pause_check_bytes(&self) -> u64 {
            self.check.load(Ordering::Relaxed)
        }
    }

    #[test]
    fn rate_limited_output_pauses_every_min_pause_check_bytes() {
        let limiter = Recording::default();
        limiter.check.store(10, Ordering::Relaxed);
        let dir = ByteBuffersDirectory::new();
        let mut out = RateLimitedIndexOutput::new(&limiter, dir.create_output("f").unwrap());
        for _ in 0..10 {
            out.write_byte(1);
        }
        assert!(limiter.pauses.lock().unwrap().is_empty());
        out.write_byte(1); // 11 > 10
        assert_eq!(*limiter.pauses.lock().unwrap(), vec![11]);
        // The limit is only re-read after a pause: 4 + 2 + 8 = 14 bytes
        // still wait for the old limit of 10.
        limiter.check.store(3, Ordering::Relaxed);
        out.write_i32(7);
        out.write_i16(1);
        assert_eq!(*limiter.pauses.lock().unwrap(), vec![11]);
        out.write_i64(2);
        assert_eq!(*limiter.pauses.lock().unwrap(), vec![11, 14]);
        // One array is one check, however large.
        out.write_bytes(&[0u8; 100]);
        assert_eq!(*limiter.pauses.lock().unwrap(), vec![11, 14, 100]);
        assert_eq!(out.file_pointer(), 11 + 4 + 2 + 8 + 100);
        assert_eq!(out.name(), "f");
        let checksum = out.checksum();
        assert_eq!(out.into_inner().close().unwrap(), checksum);
        assert_eq!(dir.file_length("f").unwrap(), 125);
    }

    #[test]
    fn simple_rate_limiter_settings() {
        let limiter = SimpleRateLimiter::new(1.0);
        assert_eq!(limiter.mb_per_sec(), 1.0);
        // 5 ms at 1 MB/s.
        assert_eq!(limiter.min_pause_check_bytes(), 5242);
        limiter.set_mb_per_sec(100.0);
        assert_eq!(limiter.mb_per_sec(), 100.0);
        assert_eq!(limiter.min_pause_check_bytes(), 524_288);
        assert!(format!("{limiter:?}").contains("SimpleRateLimiter"));
    }

    #[test]
    fn simple_rate_limiter_enforces_the_instant_rate() {
        // 1 MB/s: 50 KB is ~48.8 ms.
        let limiter = SimpleRateLimiter::new(1.0);
        let started = Instant::now();
        let first = limiter.pause(50 * 1024);
        let second = limiter.pause(50 * 1024);
        let elapsed = started.elapsed();
        // The first pause is measured from construction, the second from
        // where the first ended: together at least ~97 ms.
        assert!(first + second >= 90_000_000, "{first} + {second}");
        assert!(elapsed >= Duration::from_millis(90), "{elapsed:?}");

        // After an idle gap longer than a write's share, the target is in the
        // past: no pause, and `last` resets to now.
        std::thread::sleep(Duration::from_millis(60));
        assert_eq!(limiter.pause(1024), 0);
        // An unbounded rate never pauses.
        let fast = SimpleRateLimiter::new(f64::INFINITY);
        assert_eq!(fast.pause(u64::MAX), 0);
    }
}
