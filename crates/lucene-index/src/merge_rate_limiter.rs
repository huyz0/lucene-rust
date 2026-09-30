//! Port of `org.apache.lucene.index.MergeRateLimiter`, of the
//! `MergePolicy.OneMergeProgress` it pauses through, and of the merge
//! directory `IndexWriter` throttles a merge's outputs with
//! (`IndexWriter.addMergeRateLimiters`' `FilterDirectory`).
//!
//! # What differs from Java
//!
//! - **Abort.** Java's `pause` throws `MergeAbortedException` out of the
//!   write that hit it. [`lucene_store::RateLimiter::pause`] cannot fail --
//!   `DataOutput` writes are infallible here -- so an aborted merge's pause
//!   returns at once instead, and the merge learns it was aborted from
//!   [`OneMergeProgress::is_aborted`] (the writer checks it when the merge's
//!   writes are done, [`crate::index_writer::Error::MergeAborted`]).
//! - **The clock.** Java starts `lastNS` at `0` on `System.nanoTime()`, whose
//!   origin is far in the past (boot, on Linux), so a first write never
//!   pauses while a rate of zero stops at once. [`MergeRateLimiter`] keeps
//!   both by reading its own monotonic clock from an origin
//!   [`CLOCK_ORIGIN_NS`] in the past. `lastNS + pauseNS` saturates instead of
//!   wrapping, so a rate of zero keeps a merge stopped even after it has
//!   paused before -- where Java's `long` addition overflows and lets it run.
//! - `OneMergeProgress.setMergeThread`'s owner assertion is not kept: nothing
//!   here hands a merge to another thread midway.

use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use lucene_store::directory::{Directory, Input};
use lucene_store::{FsIndexOutput, Lock, RateLimiter};

/// Locks `m`, ignoring poison: the locks here guard no data (`()`), only a
/// wait, so a panic elsewhere while one was held leaves nothing inconsistent.
fn lock_ignoring_poison(m: &Mutex<()>) -> MutexGuard<'_, ()> {
    m.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// `OneMergeProgress.PauseReason`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PauseReason {
    /// Stopped (because of throughput rate set to 0, typically).
    Stopped,
    /// Temporarily paused because of exceeded throughput rate.
    Paused,
    /// Other reason.
    Other,
}

impl PauseReason {
    fn index(self) -> usize {
        match self {
            PauseReason::Stopped => 0,
            PauseReason::Paused => 1,
            PauseReason::Other => 2,
        }
    }
}

/// `MergePolicy.OneMergeProgress`: a running merge's abort flag and pause
/// accounting, and the condition a paused merge thread waits on.
#[derive(Debug, Default)]
pub struct OneMergeProgress {
    /// `pauseLock`; the merge thread holds it while it waits.
    pause_lock: Mutex<()>,
    /// `pausing`.
    pausing: Condvar,
    /// `pauseTimesNS`, indexed by [`PauseReason`].
    pause_times_ns: [AtomicU64; 3],
    aborted: AtomicBool,
}

impl OneMergeProgress {
    pub fn new() -> Self {
        Self::default()
    }

    /// `abort()`: abort the merge at the next possible moment, waking it if
    /// it is paused.
    pub fn abort(&self) {
        self.aborted.store(true, Ordering::SeqCst);
        self.wakeup();
    }

    /// `isAborted()`.
    pub fn is_aborted(&self) -> bool {
        self.aborted.load(Ordering::SeqCst)
    }

    /// `pauseNanos(pauseNanos, reason, condition)`: pauses the calling thread
    /// for at least `pause_nanos` unless the merge is aborted or `condition`
    /// turns false (checked on every wakeup, [`Self::wakeup`] or spurious),
    /// in which case it returns at once. The time spent counts toward
    /// `reason`.
    pub fn pause_nanos(&self, pause_nanos: u64, reason: PauseReason, condition: &dyn Fn() -> bool) {
        let start = Instant::now();
        let deadline = start.checked_add(Duration::from_nanos(pause_nanos));
        let mut guard = lock_ignoring_poison(&self.pause_lock);
        while !self.is_aborted() && condition() {
            let now = Instant::now();
            let remaining = match deadline {
                Some(d) if d > now => d.duration_since(now),
                Some(_) => break,
                // Beyond `Instant`'s range: wait in the longest step it has.
                None => Duration::from_secs(u64::from(u32::MAX)),
            };
            guard = match self.pausing.wait_timeout(guard, remaining) {
                Ok((g, _)) => g,
                Err(poisoned) => poisoned.into_inner().0,
            };
        }
        drop(guard);
        let elapsed = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
        self.pause_times_ns[reason.index()].fetch_add(elapsed, Ordering::SeqCst);
    }

    /// `wakeup()`: wakes every thread paused in [`Self::pause_nanos`].
    pub fn wakeup(&self) {
        let _guard = lock_ignoring_poison(&self.pause_lock);
        self.pausing.notify_all();
    }

    /// `getPauseTimes().get(reason)`: nanoseconds paused for `reason`.
    pub fn pause_time_ns(&self, reason: PauseReason) -> u64 {
        self.pause_times_ns[reason.index()].load(Ordering::SeqCst)
    }
}

/// `MergeRateLimiter.MIN_PAUSE_CHECK_MSEC`.
const MIN_PAUSE_CHECK_MSEC: f64 = 25.0;
/// `MergeRateLimiter.MIN_PAUSE_NS`: a pause shorter than 2 ms is not taken.
const MIN_PAUSE_NS: i64 = 2_000_000;
/// `MergeRateLimiter.MAX_PAUSE_NS`: no single wait is longer than 250 ms (the
/// loop in [`MergeRateLimiter`]'s `pause` waits again if it must).
const MAX_PAUSE_NS: i64 = 250_000_000;
/// How far in the past [`MergeRateLimiter`]'s clock starts: about 13 days,
/// standing for `System.nanoTime()`'s arbitrary, long-past origin (see the
/// module doc).
pub const CLOCK_ORIGIN_NS: i64 = 1 << 50;

/// `MergeRateLimiter`: the [`RateLimiter`] a merge's outputs pause through.
/// Starts unlimited; `mb_per_sec == 0.0` stops the merge until the rate
/// changes or the merge is aborted.
#[derive(Debug)]
pub struct MergeRateLimiter {
    /// `f64` bits: Java's `volatile double mbPerSec`.
    mb_per_sec: AtomicU64,
    /// `volatile long minPauseCheckBytes`.
    min_pause_check_bytes: AtomicU64,
    /// Makes updating the two fields above atomic, as Java's
    /// `synchronized (this)` in `setMBPerSec`.
    set_lock: Mutex<()>,
    /// `lastNS`, on this limiter's clock.
    last_ns: AtomicI64,
    /// `totalBytesWritten`.
    total_bytes_written: AtomicU64,
    merge_progress: Arc<OneMergeProgress>,
    epoch: Instant,
}

impl MergeRateLimiter {
    /// `new MergeRateLimiter(mergeProgress)`: no limit yet.
    pub fn new(merge_progress: Arc<OneMergeProgress>) -> Self {
        let limiter = MergeRateLimiter {
            mb_per_sec: AtomicU64::new(f64::INFINITY.to_bits()),
            min_pause_check_bytes: AtomicU64::new(0),
            set_lock: Mutex::new(()),
            last_ns: AtomicI64::new(0),
            total_bytes_written: AtomicU64::new(0),
            merge_progress,
            epoch: Instant::now(),
        };
        limiter.set_mb_per_sec(f64::INFINITY);
        limiter
    }

    /// `System.nanoTime()` on this limiter's clock.
    fn now_ns(&self) -> i64 {
        let elapsed = i64::try_from(self.epoch.elapsed().as_nanos()).unwrap_or(i64::MAX);
        elapsed.saturating_add(CLOCK_ORIGIN_NS)
    }

    /// `setMBPerSec(mbPerSec)`, checked: a negative rate is Java's
    /// `IllegalArgumentException`. `0.0` is allowed and stops the merge.
    pub fn try_set_mb_per_sec(&self, mb_per_sec: f64) -> Result<(), String> {
        if mb_per_sec < 0.0 || mb_per_sec.is_nan() {
            return Err(format!("mbPerSec must be positive; got: {mb_per_sec}"));
        }
        {
            let _guard = lock_ignoring_poison(&self.set_lock);
            self.mb_per_sec
                .store(mb_per_sec.to_bits(), Ordering::SeqCst);
            // `Math.min(1024 * 1024, (long) ((MIN_PAUSE_CHECK_MSEC / 1000.0)
            // * mbPerSec * 1024 * 1024))`; `as` saturates as Java's cast does
            // (`POSITIVE_INFINITY` to `Long.MAX_VALUE`).
            let bytes = ((MIN_PAUSE_CHECK_MSEC / 1000.0) * mb_per_sec * 1024.0 * 1024.0) as u64;
            self.min_pause_check_bytes
                .store(bytes.min(1024 * 1024), Ordering::SeqCst);
        }
        self.merge_progress.wakeup();
        Ok(())
    }

    /// `getTotalBytesWritten()`.
    pub fn total_bytes_written(&self) -> u64 {
        self.total_bytes_written.load(Ordering::SeqCst)
    }

    /// `getTotalStoppedNS()`.
    pub fn total_stopped_ns(&self) -> u64 {
        self.merge_progress.pause_time_ns(PauseReason::Stopped)
    }

    /// `getTotalPausedNS()`.
    pub fn total_paused_ns(&self) -> u64 {
        self.merge_progress.pause_time_ns(PauseReason::Paused)
    }

    /// The progress this limiter pauses through.
    pub fn merge_progress(&self) -> &Arc<OneMergeProgress> {
        &self.merge_progress
    }

    /// `maybePause(bytes)`: the nanoseconds spent paused, or `None` when no
    /// pause was due (Java's `-1`) -- or the merge is aborted (Java throws).
    fn maybe_pause(&self, bytes: u64) -> Option<u64> {
        if self.merge_progress.is_aborted() {
            return None;
        }
        let rate = self.mb_per_sec();
        let seconds_to_pause = (bytes as f64 / 1024. / 1024.) / rate;
        // `(long) (1000000000 * secondsToPause)`, saturating like Java's cast.
        let pause_ns = (1_000_000_000.0 * seconds_to_pause) as i64;
        let mut cur_pause_ns = 0i64;
        let _ = self
            .last_ns
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |last| {
                let cur_ns = self.now_ns();
                let target_ns = last.saturating_add(pause_ns);
                let pause = target_ns.saturating_sub(cur_ns);
                if pause <= MIN_PAUSE_NS {
                    // The instant rate, not the rate averaged over history.
                    cur_pause_ns = 0;
                    Some(cur_ns)
                } else {
                    cur_pause_ns = pause;
                    Some(last)
                }
            });
        if cur_pause_ns == 0 {
            return None;
        }
        let pause = u64::try_from(cur_pause_ns.min(MAX_PAUSE_NS)).unwrap_or(0);
        let start = Instant::now();
        let reason = if rate == 0.0 {
            PauseReason::Stopped
        } else {
            PauseReason::Paused
        };
        let rate_bits = rate.to_bits();
        self.merge_progress.pause_nanos(pause, reason, &|| {
            self.mb_per_sec.load(Ordering::SeqCst) == rate_bits
        });
        Some(u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX))
    }
}

impl RateLimiter for MergeRateLimiter {
    /// `setMBPerSec`; a negative rate is refused with no change (see
    /// [`MergeRateLimiter::try_set_mb_per_sec`] for the checked form).
    fn set_mb_per_sec(&self, mb_per_sec: f64) {
        let _ = self.try_set_mb_per_sec(mb_per_sec);
    }

    fn mb_per_sec(&self) -> f64 {
        f64::from_bits(self.mb_per_sec.load(Ordering::SeqCst))
    }

    /// `pause(bytes)`: waits, in steps of at most 250 ms and re-reading the
    /// rate between them, until writing `bytes` is within the rate.
    fn pause(&self, bytes: u64) -> u64 {
        self.total_bytes_written.fetch_add(bytes, Ordering::SeqCst);
        let mut paused = 0u64;
        while let Some(delta) = self.maybe_pause(bytes) {
            paused = paused.saturating_add(delta);
        }
        paused
    }

    fn min_pause_check_bytes(&self) -> u64 {
        self.min_pause_check_bytes.load(Ordering::SeqCst)
    }
}

/// `IndexWriter.addMergeRateLimiters`' merge directory: `dir`, with every
/// output it creates throttled by one merge's [`MergeRateLimiter`].
pub struct MergeDirectory<'a> {
    inner: &'a dyn Directory,
    limiter: Arc<MergeRateLimiter>,
}

impl<'a> MergeDirectory<'a> {
    pub fn new(inner: &'a dyn Directory, limiter: Arc<MergeRateLimiter>) -> Self {
        MergeDirectory { inner, limiter }
    }

    /// The limiter every output of this directory pauses through.
    pub fn limiter(&self) -> &Arc<MergeRateLimiter> {
        &self.limiter
    }

    fn throttled(&self, mut out: FsIndexOutput) -> FsIndexOutput {
        out.set_rate_limiter(self.limiter.clone());
        out
    }
}

impl Directory for MergeDirectory<'_> {
    fn list_all(&self) -> lucene_store::Result<Vec<String>> {
        self.inner.list_all()
    }
    fn open(&self, name: &str) -> lucene_store::Result<Input> {
        self.inner.open(name)
    }
    fn file_length(&self, name: &str) -> lucene_store::Result<u64> {
        self.inner.file_length(name)
    }
    fn create_output(&self, name: &str) -> lucene_store::Result<FsIndexOutput> {
        Ok(self.throttled(self.inner.create_output(name)?))
    }
    fn create_temp_output(
        &self,
        prefix: &str,
        suffix: &str,
    ) -> lucene_store::Result<FsIndexOutput> {
        Ok(self.throttled(self.inner.create_temp_output(prefix, suffix)?))
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
    fn obtain_lock(&self, name: &str) -> lucene_store::Result<Box<dyn Lock>> {
        self.inner.obtain_lock(name)
    }
    fn pending_deletions(&self) -> lucene_store::Result<std::collections::BTreeSet<String>> {
        self.inner.pending_deletions()
    }
    fn fs_directory_path(&self) -> Option<&std::path::Path> {
        self.inner.fs_directory_path()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_store::data_output::DataOutput;
    use lucene_store::ByteBuffersDirectory;

    #[test]
    fn settings_follow_java() {
        let progress = Arc::new(OneMergeProgress::new());
        let limiter = MergeRateLimiter::new(progress.clone());
        // Unlimited to start: `(long) Infinity` capped at 1 MiB.
        assert_eq!(limiter.mb_per_sec(), f64::INFINITY);
        assert_eq!(limiter.min_pause_check_bytes(), 1024 * 1024);
        // 25 ms at 10 MB/s is 262144 bytes.
        limiter.set_mb_per_sec(10.0);
        assert_eq!(limiter.min_pause_check_bytes(), 262_144);
        limiter.set_mb_per_sec(0.0);
        assert_eq!(limiter.min_pause_check_bytes(), 0);
        assert!(limiter.try_set_mb_per_sec(-1.0).is_err());
        assert!(limiter.try_set_mb_per_sec(f64::NAN).is_err());
        limiter.set_mb_per_sec(-1.0);
        assert_eq!(limiter.mb_per_sec(), 0.0, "a refused rate changes nothing");
        assert!(Arc::ptr_eq(limiter.merge_progress(), &progress));
    }

    #[test]
    fn an_unlimited_or_first_write_does_not_pause() {
        let limiter = MergeRateLimiter::new(Arc::new(OneMergeProgress::new()));
        assert_eq!(limiter.pause(10 << 20), 0);
        assert_eq!(limiter.pause(10 << 20), 0);
        assert_eq!(limiter.total_bytes_written(), 20 << 20);
        // The first write at a finite rate never pauses: `lastNS` starts far
        // in the past. (It is set to "now" by every write that does not
        // pause, so it is only the first.)
        let limiter = MergeRateLimiter::new(Arc::new(OneMergeProgress::new()));
        limiter.set_mb_per_sec(1.0);
        assert_eq!(limiter.pause(64 << 10), 0);
        assert_eq!(limiter.total_bytes_written(), 64 << 10);
        assert_eq!(limiter.total_paused_ns(), 0);
        assert_eq!(limiter.total_stopped_ns(), 0);
    }

    #[test]
    fn a_rate_is_enforced_by_pausing() {
        let limiter = MergeRateLimiter::new(Arc::new(OneMergeProgress::new()));
        limiter.set_mb_per_sec(10.0);
        let start = Instant::now();
        // Four 1 MiB writes at 10 MB/s: each after the first waits ~100 ms.
        for _ in 0..4 {
            limiter.pause(1 << 20);
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed >= Duration::from_millis(250),
            "{elapsed:?}: expected ~300 ms of pauses"
        );
        assert!(limiter.total_paused_ns() >= 250_000_000);
        assert_eq!(limiter.total_stopped_ns(), 0);
    }

    #[test]
    fn a_stopped_merge_waits_for_a_new_rate_or_an_abort() {
        let progress = Arc::new(OneMergeProgress::new());
        let limiter = Arc::new(MergeRateLimiter::new(progress.clone()));
        limiter.set_mb_per_sec(0.0);
        let l = limiter.clone();
        let t = std::thread::spawn(move || l.pause(1));
        std::thread::sleep(Duration::from_millis(300));
        assert!(!t.is_finished(), "a zero rate stops the merge");
        limiter.set_mb_per_sec(f64::INFINITY);
        t.join().unwrap();
        assert!(limiter.total_stopped_ns() >= 250_000_000);

        limiter.set_mb_per_sec(0.0);
        let l = limiter.clone();
        let t = std::thread::spawn(move || l.pause(1));
        std::thread::sleep(Duration::from_millis(50));
        progress.abort();
        t.join().unwrap();
        assert!(progress.is_aborted());
        // Aborted: no further pause, whatever the rate.
        assert_eq!(limiter.pause(1 << 30), 0);
    }

    #[test]
    fn pause_nanos_accounts_by_reason_and_returns_when_the_condition_fails() {
        let p = OneMergeProgress::new();
        p.pause_nanos(3_000_000, PauseReason::Other, &|| true);
        assert!(p.pause_time_ns(PauseReason::Other) >= 3_000_000);
        let start = Instant::now();
        p.pause_nanos(5_000_000_000, PauseReason::Paused, &|| false);
        assert!(start.elapsed() < Duration::from_secs(1));
        // Longer than `Instant` can represent: waits until the condition goes.
        let flag = AtomicBool::new(true);
        let p = Arc::new(p);
        std::thread::scope(|s| {
            s.spawn(|| {
                p.pause_nanos(u64::MAX, PauseReason::Stopped, &|| {
                    flag.load(Ordering::SeqCst)
                })
            });
            std::thread::sleep(Duration::from_millis(20));
            flag.store(false, Ordering::SeqCst);
            p.wakeup();
        });
    }

    #[test]
    fn the_merge_directory_throttles_every_output_and_forwards_the_rest() {
        let dir = ByteBuffersDirectory::new();
        let limiter = Arc::new(MergeRateLimiter::new(Arc::new(OneMergeProgress::new())));
        let merge_dir = MergeDirectory::new(&dir, limiter.clone());
        assert!(Arc::ptr_eq(merge_dir.limiter(), &limiter));
        let mut out = merge_dir.create_output("a").unwrap();
        out.write_bytes(&[1u8; 3 << 20]);
        out.close().unwrap();
        let mut tmp = merge_dir.create_temp_output("t", "tmp").unwrap();
        let tmp_name = lucene_store::index_output::IndexOutput::name(&tmp).to_string();
        tmp.write_bytes(&[2u8; 2 << 20]);
        tmp.close().unwrap();
        // Unlimited, so `minPauseCheckBytes` is 1 MiB: each output hands
        // `pause` its bytes once more than that built up -- every 129 chunks
        // of 8 KiB -- and, as in Java, only what reached `pause` is counted:
        // twice for the 3 MiB file, once for the 2 MiB one.
        assert_eq!(limiter.total_bytes_written(), 3 * 129 * 8192);
        assert_eq!(merge_dir.file_length("a").unwrap(), 3 << 20);
        assert_eq!(merge_dir.open("a").unwrap().len(), 3 << 20);
        merge_dir.rename(&tmp_name, "b").unwrap();
        merge_dir.sync(&["a".to_string(), "b".to_string()]).unwrap();
        merge_dir.sync_meta_data().unwrap();
        assert_eq!(merge_dir.list_all().unwrap(), vec!["a", "b"]);
        merge_dir.delete_file("b").unwrap();
        assert!(merge_dir.pending_deletions().unwrap().is_empty());
        assert!(merge_dir.fs_directory_path().is_none());
        drop(merge_dir.obtain_lock("write.lock").unwrap());
    }
}
