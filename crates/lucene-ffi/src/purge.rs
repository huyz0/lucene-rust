//! The allocator's purge delay follows the write load.
//!
//! mimalloc returns freed pages to the system `purge_delay` milliseconds
//! after they go idle. The default (10 ms) is right for searches, whose
//! scratch tables are freed and allocated again within the same request
//! stream: purging them at once costs 15-25% on allocation-heavy
//! aggregations. Indexing and merging free buffers of tens of megabytes
//! from many threads and leave them idle, and at the default the node held
//! 300-400 MB more at the peak of a bulk load. So the delay is 0 while
//! writes arrive and goes back to the default a few seconds after the last
//! one. A `MIMALLOC_PURGE_DELAY` set in the environment is left alone.

#[cfg(feature = "mimalloc")]
mod imp {
    use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    /// `mi_option_purge_delay` in mimalloc's option enum.
    const PURGE_DELAY: libmimalloc_sys::mi_option_t = 15;
    /// How long after the last write the default delay returns.
    const QUIET: Duration = Duration::from_secs(5);

    static LAST_WRITE_MS: AtomicU64 = AtomicU64::new(0);
    static EAGER: AtomicBool = AtomicBool::new(false);
    static STARTED: std::sync::Once = std::sync::Once::new();
    static EPOCH: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();

    fn now_ms() -> u64 {
        let ms = EPOCH.get_or_init(Instant::now).elapsed().as_millis();
        u64::try_from(ms).unwrap_or(u64::MAX)
    }

    fn overridden() -> bool {
        std::env::var_os("MIMALLOC_PURGE_DELAY").is_some()
    }

    /// A write is under way: purge freed pages at once until it stops.
    pub fn writing() -> Busy {
        ACTIVE.fetch_add(1, Ordering::Relaxed);
        LAST_WRITE_MS.store(now_ms(), Ordering::Relaxed);
        if EAGER.load(Ordering::Relaxed) || overridden() {
            return Busy;
        }
        STARTED.call_once(|| {
            // SAFETY: `mi_option_get` reads an option's value.
            let default = unsafe { libmimalloc_sys::mi_option_get(PURGE_DELAY) };
            DEFAULT.store(default, Ordering::Relaxed);
            // A failed spawn leaves the delay as it was.
            let _ = std::thread::Builder::new()
                .name("purge-delay".into())
                .spawn(restore_when_quiet);
        });
        EAGER.store(true, Ordering::Relaxed);
        // SAFETY: `mi_option_set` writes an option's value.
        unsafe { libmimalloc_sys::mi_option_set(PURGE_DELAY, 0) };
        Busy
    }

    /// Held while a write call runs; the default delay does not return before it ends.
    pub struct Busy;

    impl Drop for Busy {
        fn drop(&mut self) {
            LAST_WRITE_MS.store(now_ms(), Ordering::Relaxed);
            ACTIVE.fetch_sub(1, Ordering::Relaxed);
        }
    }

    static ACTIVE: AtomicUsize = AtomicUsize::new(0);

    static DEFAULT: AtomicI64 = std::sync::atomic::AtomicI64::new(10);

    fn restore_when_quiet() {
        loop {
            std::thread::sleep(Duration::from_secs(1));
            if !EAGER.load(Ordering::Relaxed) {
                continue;
            }
            if ACTIVE.load(Ordering::Relaxed) > 0 {
                continue;
            }
            let idle = now_ms().saturating_sub(LAST_WRITE_MS.load(Ordering::Relaxed));
            if idle >= QUIET.as_millis() as u64 {
                EAGER.store(false, Ordering::Relaxed);
                // SAFETY: as in `writing`.
                unsafe {
                    libmimalloc_sys::mi_option_set(
                        PURGE_DELAY,
                        DEFAULT.load(Ordering::Relaxed) as std::os::raw::c_long,
                    );
                }
            }
        }
    }
}

/// Marks a write call; hold the result until it returns. See the module documentation.
#[cfg(feature = "mimalloc")]
#[inline]
pub(crate) fn writing() -> imp::Busy {
    imp::writing()
}

/// Marks a write call (no allocator to tune without the `mimalloc` feature).
#[cfg(not(feature = "mimalloc"))]
#[inline]
pub(crate) fn writing() {}
