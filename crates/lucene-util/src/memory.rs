//! Accounting of the memory the library allocates, process-wide and per
//! request, so a host can bound it.
//!
//! The library's global allocator (the plugin's, in `lucene-ffi`) reports
//! every allocation and free to [`on_alloc`] and [`on_free`]. They never
//! allocate, lock or call out: each thread sums its own changes and folds
//! them into the shared counters once they reach [`STEP`] bytes either way,
//! so the shared counters lag the truth by at most [`STEP`] per thread.
//!
//! Two bounds are kept:
//!
//! - **Process**: [`live_bytes`] is what the library holds now. A host sets
//!   [`set_process_limit`]; past it, every request's next [`checkpoint`]
//!   fails, and work outside requests (indexing, merges) carries on, since
//!   the host bounds those with its own controls.
//! - **Request**: a host wraps a call in a [`Scope`] ([`enter`]). The bytes
//!   the call's threads gain while the scope is entered are its
//!   [`Scope::used`]. A request grows past what it was granted by asking the
//!   host's [`Grant`] for more (the plugin charges OpenSearch's `request`
//!   circuit breaker), and fails at its next [`checkpoint`] once the host
//!   refuses or its own hard limit is reached.
//!
//! An allocation itself never fails here: an allocator that returns null
//! aborts the process. Long loops call [`checkpoint`] and code about to make
//! one large allocation calls [`reserve`] first, both returning
//! [`Exceeded`], which ends the request with an error and frees what it
//! held. Memory freed during a scope that was allocated before it (a cache
//! entry evicted) counts against the scope's use, as the threads' net change
//! is what is measured; the error is in the safe direction for the process
//! bound and small for a request.

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU8, Ordering::Relaxed};
use std::sync::{Arc, Mutex};

/// A thread's unreported change is folded into the shared counters once it
/// reaches this many bytes either way.
pub const STEP: i64 = 128 << 10;

/// The smallest grant a scope asks its host for.
pub const MIN_GRANT: i64 = 1 << 20;

/// Bytes the library holds now, to within [`STEP`] per thread.
static LIVE: AtomicI64 = AtomicI64::new(0);
/// The most [`LIVE`] has been.
static PEAK: AtomicI64 = AtomicI64::new(0);
/// [`LIVE`] past which requests fail; `0` is no limit.
static PROCESS_LIMIT: AtomicI64 = AtomicI64::new(0);
/// Whether [`LIVE`] is past [`PROCESS_LIMIT`] (set and cleared by the folds).
static OVER_PROCESS: AtomicBool = AtomicBool::new(false);
/// Requests failed by each bound, for the host's stats.
static TRIPS_REQUEST: AtomicI64 = AtomicI64::new(0);
static TRIPS_PROCESS: AtomicI64 = AtomicI64::new(0);
static CANCELLED: AtomicI64 = AtomicI64::new(0);

struct Local {
    /// Bytes allocated minus freed on this thread, not yet folded.
    pending: Cell<i64>,
    /// The scope entered on this thread, or null: owned by the [`Entered`]
    /// guard that set it, which outlives every read of it here.
    scope: Cell<*const Scope>,
    /// Set while this thread's fold runs, so that nothing a fold does is
    /// counted twice (a fold never allocates; this is belt and braces).
    busy: Cell<bool>,
}

thread_local! {
    static LOCAL: Local = const {
        Local {
            pending: Cell::new(0),
            scope: Cell::new(std::ptr::null()),
            busy: Cell::new(false),
        }
    };
}

/// The allocator's report of `bytes` allocated on this thread.
#[inline]
pub fn on_alloc(bytes: usize) {
    change(i64::try_from(bytes).unwrap_or(i64::MAX));
}

/// The allocator's report of `bytes` freed on this thread.
#[inline]
pub fn on_free(bytes: usize) {
    change(i64::try_from(bytes).map_or(i64::MIN, |b| b.wrapping_neg()));
}

#[inline]
fn change(by: i64) {
    // A thread being torn down has no slot any more: its change goes
    // straight to the process counter.
    let _ = LOCAL
        .try_with(|l| {
            let p = l.pending.get().saturating_add(by);
            if (-STEP..STEP).contains(&p) || l.busy.get() {
                l.pending.set(p);
            } else {
                l.pending.set(0);
                l.busy.set(true);
                fold(p, l.scope.get());
                l.busy.set(false);
            }
        })
        .map_err(|_| fold(by, std::ptr::null()));
}

/// Adds `by` to the process counter and to `scope`'s use.
fn fold(by: i64, scope: *const Scope) {
    let live = LIVE.fetch_add(by, Relaxed).saturating_add(by);
    if by > 0 {
        PEAK.fetch_max(live, Relaxed);
    }
    let limit = PROCESS_LIMIT.load(Relaxed);
    let over = limit > 0 && live > limit;
    if over != OVER_PROCESS.load(Relaxed) {
        OVER_PROCESS.store(over, Relaxed);
    }
    // SAFETY: a non-null scope pointer is the one `enter` stored for this
    // thread, kept alive by the `Entered` guard (an `Arc`) until the guard
    // clears the slot again on this same thread.
    if let Some(s) = unsafe { scope.as_ref() } {
        let used = s.used.fetch_add(by, Relaxed).saturating_add(by);
        s.peak.fetch_max(used, Relaxed);
    }
}

/// Folds this thread's pending change now.
fn flush() {
    let _ = LOCAL.try_with(|l| {
        let p = l.pending.replace(0);
        if p != 0 {
            l.busy.set(true);
            fold(p, l.scope.get());
            l.busy.set(false);
        }
    });
}

/// Bytes the library holds now (to within [`STEP`] per thread).
pub fn live_bytes() -> i64 {
    LIVE.load(Relaxed)
}

/// The most the library has held.
pub fn peak_bytes() -> i64 {
    PEAK.load(Relaxed)
}

/// Sets the process bound: past `bytes` held, requests fail at their next
/// [`checkpoint`]. `0` or less removes it.
pub fn set_process_limit(bytes: i64) {
    PROCESS_LIMIT.store(bytes.max(0), Relaxed);
    let live = LIVE.load(Relaxed);
    OVER_PROCESS.store(bytes > 0 && live > bytes, Relaxed);
}

/// The process bound, `0` when there is none.
pub fn process_limit() -> i64 {
    PROCESS_LIMIT.load(Relaxed)
}

/// Counters for the host's stats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stats {
    pub live: i64,
    pub peak: i64,
    pub process_limit: i64,
    pub request_trips: i64,
    pub process_trips: i64,
    pub cancelled: i64,
}

/// The counters now.
pub fn stats() -> Stats {
    Stats {
        live: LIVE.load(Relaxed),
        peak: PEAK.load(Relaxed),
        process_limit: PROCESS_LIMIT.load(Relaxed),
        request_trips: TRIPS_REQUEST.load(Relaxed),
        process_trips: TRIPS_PROCESS.load(Relaxed),
        cancelled: CANCELLED.load(Relaxed),
    }
}

/// The host's answer to a scope that needs `bytes` more: `true` grants them
/// (and the host now holds them charged until [`Scope::granted`] is
/// released), `false` refuses. Called on whichever thread of the request
/// needs them, never from the allocator.
pub type Grant = dyn Fn(i64) -> bool + Send + Sync;

/// Why a request stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Reason {
    /// Its host refused more memory, or its own hard limit was reached.
    Request = 1,
    /// The process bound was reached.
    Process = 2,
    /// Its host cancelled it ([`Scope::cancel`]).
    Cancelled = 3,
}

/// Bytes a long-lived structure keeps, added to a node gauge while it
/// lives and taken off when it is dropped: what the host reports to its own
/// breakers for memory that outlives any request (OpenSearch's
/// `fielddata` breaker for global ordinals). Clones are not counted twice:
/// share one `Held` behind an `Arc`.
#[derive(Debug)]
pub struct Held {
    gauge: &'static AtomicI64,
    bytes: i64,
}

impl Held {
    /// `bytes` held from now on, added to `gauge`.
    pub fn new(gauge: &'static AtomicI64, bytes: usize) -> Self {
        let bytes = i64::try_from(bytes).unwrap_or(i64::MAX);
        gauge.fetch_add(bytes, Relaxed);
        Self { gauge, bytes }
    }

    /// The bytes counted.
    pub fn bytes(&self) -> i64 {
        self.bytes
    }
}

impl Drop for Held {
    fn drop(&mut self) {
        self.gauge.fetch_sub(self.bytes, Relaxed);
    }
}

/// Every [`Held`] is equal: the count is bookkeeping, not part of the value
/// that holds it.
impl PartialEq for Held {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl Eq for Held {}

/// A request stopped by a bound or cancelled.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct Exceeded {
    pub reason: Reason,
    pub message: String,
}

/// One request's accounting: what its threads hold, what the host granted,
/// and whether it was told to stop.
pub struct Scope {
    used: AtomicI64,
    peak: AtomicI64,
    granted: AtomicI64,
    /// The most the request may use whatever the host grants; `0` is none.
    hard_limit: i64,
    stop: AtomicU8,
    grant: Option<Box<Grant>>,
    /// Held while asking the host, so concurrent threads ask once.
    asking: Mutex<()>,
    label: String,
}

impl std::fmt::Debug for Scope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Scope")
            .field("label", &self.label)
            .field("used", &self.used())
            .field("granted", &self.granted())
            .finish_non_exhaustive()
    }
}

impl Scope {
    /// A scope named `label` (for messages), at most `hard_limit` bytes
    /// (`0`: no limit of its own), growing through `grant` (`None`: up to
    /// the hard limit without asking).
    pub fn new(label: impl Into<String>, hard_limit: i64, grant: Option<Box<Grant>>) -> Arc<Self> {
        Arc::new(Scope {
            used: AtomicI64::new(0),
            peak: AtomicI64::new(0),
            granted: AtomicI64::new(0),
            hard_limit: hard_limit.max(0),
            stop: AtomicU8::new(0),
            grant,
            asking: Mutex::new(()),
            label: label.into(),
        })
    }

    /// Net bytes its threads gained since it was entered.
    pub fn used(&self) -> i64 {
        self.used.load(Relaxed)
    }

    /// The most [`Self::used`] reached.
    pub fn peak(&self) -> i64 {
        self.peak.load(Relaxed)
    }

    /// Bytes the host granted (and must release when the request ends).
    pub fn granted(&self) -> i64 {
        self.granted.load(Relaxed)
    }

    /// Tells the request to stop at its next [`checkpoint`]: any thread may
    /// call it, while the request runs on others.
    pub fn cancel(&self) {
        let _ = self
            .stop
            .compare_exchange(0, Reason::Cancelled as u8, Relaxed, Relaxed);
    }

    fn stopped(&self, reason: Reason, message: String) -> Exceeded {
        let _ = self
            .stop
            .compare_exchange(0, reason as u8, Relaxed, Relaxed);
        match reason {
            Reason::Request => TRIPS_REQUEST.fetch_add(1, Relaxed),
            Reason::Process => TRIPS_PROCESS.fetch_add(1, Relaxed),
            Reason::Cancelled => CANCELLED.fetch_add(1, Relaxed),
        };
        Exceeded { reason, message }
    }

    fn check_stop(&self) -> Result<(), Exceeded> {
        match self.stop.load(Relaxed) {
            0 => Ok(()),
            r if r == Reason::Cancelled as u8 => Err(Exceeded {
                reason: Reason::Cancelled,
                message: format!("[{}] cancelled", self.label),
            }),
            r => Err(Exceeded {
                reason: if r == Reason::Process as u8 {
                    Reason::Process
                } else {
                    Reason::Request
                },
                message: format!("[{}] stopped: over its memory limit", self.label),
            }),
        }
    }

    /// Makes sure `more` bytes beyond what is used now are granted, asking
    /// the host when they are not.
    fn ensure(&self, more: i64) -> Result<(), Exceeded> {
        let want = self.used().saturating_add(more.max(0));
        if want <= self.granted() {
            return Ok(());
        }
        if self.hard_limit > 0 && want > self.hard_limit {
            return Err(self.stopped(
                Reason::Request,
                format!(
                    "[{}] would use {want} bytes, over its limit of {} bytes",
                    self.label, self.hard_limit
                ),
            ));
        }
        let _asking = self.asking.lock().unwrap_or_else(|e| e.into_inner());
        let granted = self.granted();
        if want <= granted {
            return Ok(());
        }
        // Grow by an eighth of what is held at least, so a request asks a
        // logarithmic number of times.
        let ask = want.saturating_sub(granted).max(MIN_GRANT).max(granted / 8);
        let ask = if self.hard_limit > 0 {
            ask.min(self.hard_limit.saturating_sub(granted))
                .max(want.saturating_sub(granted))
        } else {
            ask
        };
        let ok = match &self.grant {
            Some(g) => g(ask),
            None => true,
        };
        if !ok {
            return Err(self.stopped(
                Reason::Request,
                format!(
                    "[{}] refused {ask} more bytes (using {}, granted {granted})",
                    self.label,
                    self.used()
                ),
            ));
        }
        self.granted.fetch_add(ask, Relaxed);
        Ok(())
    }
}

/// The scope [`enter`] put on this thread, released when dropped.
pub struct Entered {
    scope: Arc<Scope>,
    previous: *const Scope,
    /// The previous scope's `Arc`, kept alive while it is shadowed.
    previous_arc: Option<Arc<Scope>>,
}

thread_local! {
    /// The `Arc` behind [`Local::scope`], for [`current`].
    static CURRENT: std::cell::RefCell<Option<Arc<Scope>>> = const { std::cell::RefCell::new(None) };
}

/// Puts `scope` on this thread until the guard drops: what the thread
/// allocates and frees meanwhile is the scope's.
pub fn enter(scope: Arc<Scope>) -> Entered {
    flush();
    let previous = LOCAL.with(|l| l.scope.replace(Arc::as_ptr(&scope)));
    let previous_arc = CURRENT.with(|c| c.replace(Some(Arc::clone(&scope))));
    Entered {
        scope,
        previous,
        previous_arc,
    }
}

impl Entered {
    /// The scope entered.
    pub fn scope(&self) -> &Arc<Scope> {
        &self.scope
    }
}

impl Drop for Entered {
    fn drop(&mut self) {
        flush();
        LOCAL.with(|l| l.scope.set(self.previous));
        let prev = self.previous_arc.take();
        CURRENT.with(|c| *c.borrow_mut() = prev);
    }
}

/// The scope on this thread, to carry to a thread the request hands work
/// to ([`enter`] it there).
pub fn current() -> Option<Arc<Scope>> {
    CURRENT.with(|c| c.borrow().clone())
}

/// Runs `f` with `scope` (from [`current`] on the handing thread) entered.
pub fn within<T>(scope: Option<&Arc<Scope>>, f: impl FnOnce() -> T) -> T {
    match scope {
        Some(s) => {
            let _entered = enter(Arc::clone(s));
            f()
        }
        None => f(),
    }
}

fn with_scope<T>(f: impl FnOnce(&Scope) -> Result<T, Exceeded>) -> Option<Result<T, Exceeded>> {
    let ptr = LOCAL
        .try_with(|l| l.scope.get())
        .unwrap_or(std::ptr::null());
    // SAFETY: as in `fold`: the pointer is this thread's entered scope, kept
    // alive by its guard for as long as it is set.
    unsafe { ptr.as_ref() }.map(f)
}

fn process_check(s: &Scope) -> Result<(), Exceeded> {
    if OVER_PROCESS.load(Relaxed) && s.used() > 0 {
        return Err(s.stopped(
            Reason::Process,
            format!(
                "[{}] the process holds {} bytes, over its limit of {} bytes",
                s.label,
                LIVE.load(Relaxed),
                PROCESS_LIMIT.load(Relaxed)
            ),
        ));
    }
    Ok(())
}

/// Whether the request on this thread may go on: `Err` once it was
/// cancelled, its host refused it more memory, or the process is over its
/// bound. Free outside a scope, and cheap inside one (a few loads) unless
/// the request has outgrown its grant.
#[inline]
pub fn checkpoint() -> Result<(), Exceeded> {
    with_scope(|s| {
        s.check_stop()?;
        process_check(s)?;
        if s.used() > s.granted() {
            s.ensure(0)?;
        }
        Ok(())
    })
    .unwrap_or(Ok(()))
}

/// [`checkpoint`] before allocating `bytes` at once: refused, rather than
/// allocated and then stopped, when the request may not grow by that much.
pub fn reserve(bytes: usize) -> Result<(), Exceeded> {
    let bytes = i64::try_from(bytes).unwrap_or(i64::MAX);
    with_scope(|s| {
        s.check_stop()?;
        if OVER_PROCESS.load(Relaxed)
            || (PROCESS_LIMIT.load(Relaxed) > 0
                && LIVE.load(Relaxed).saturating_add(bytes) > PROCESS_LIMIT.load(Relaxed))
        {
            return Err(s.stopped(
                Reason::Process,
                format!(
                    "[{}] {bytes} more bytes would take the process over its limit of {} bytes \
                     (it holds {})",
                    s.label,
                    PROCESS_LIMIT.load(Relaxed),
                    LIVE.load(Relaxed)
                ),
            ));
        }
        s.ensure(bytes)
    })
    .unwrap_or(Ok(()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicI64;

    // The allocator is not hooked in this crate's tests: they report
    // changes by hand, as the allocator would.

    #[test]
    fn a_scope_counts_what_its_thread_gains_and_asks_the_host_for_more() {
        let asked = Arc::new(AtomicI64::new(0));
        let a = Arc::clone(&asked);
        let s = Scope::new(
            "t",
            0,
            Some(Box::new(move |n| {
                a.fetch_add(n, Relaxed);
                true
            })),
        );
        {
            let _e = enter(Arc::clone(&s));
            on_alloc(3 << 20);
            assert!(checkpoint().is_ok());
            assert!(s.used() >= 3 << 20);
            assert!(s.granted() >= s.used());
            on_free(3 << 20);
        }
        assert_eq!(s.used(), 0);
        assert_eq!(asked.load(Relaxed), s.granted());
        assert_eq!(s.peak(), 3 << 20);
    }

    #[test]
    fn a_refused_grant_stops_the_request_and_stays_stopped() {
        let s = Scope::new("agg", 0, Some(Box::new(|_| false)));
        let _e = enter(Arc::clone(&s));
        on_alloc(2 << 20);
        let e = checkpoint().unwrap_err();
        assert_eq!(e.reason, Reason::Request);
        assert!(e.message.contains("[agg]"), "{}", e.message);
        on_free(2 << 20);
        assert!(checkpoint().is_err(), "a stopped request stays stopped");
    }

    #[test]
    fn reserve_refuses_before_the_allocation_over_the_hard_limit() {
        let s = Scope::new("r", 4 << 20, None);
        let _e = enter(Arc::clone(&s));
        assert!(reserve(1 << 20).is_ok());
        let e = reserve(8 << 20).unwrap_err();
        assert_eq!(e.reason, Reason::Request);
        assert_eq!(s.used(), 0, "nothing was allocated");
    }

    #[test]
    fn cancel_from_another_thread_stops_at_the_next_checkpoint() {
        let s = Scope::new("c", 0, None);
        let _e = enter(Arc::clone(&s));
        assert!(checkpoint().is_ok());
        let other = Arc::clone(&s);
        std::thread::spawn(move || other.cancel()).join().unwrap();
        assert_eq!(checkpoint().unwrap_err().reason, Reason::Cancelled);
    }

    #[test]
    fn small_changes_wait_on_the_thread_until_a_step() {
        let s = Scope::new("w", 0, None);
        let _e = enter(Arc::clone(&s));
        on_alloc(1000);
        assert_eq!(s.used(), 0, "below a step, not folded yet");
        on_alloc(usize::try_from(STEP).unwrap());
        assert_eq!(s.used(), STEP + 1000);
        on_free(usize::try_from(STEP).unwrap() + 1000);
    }

    #[test]
    fn work_handed_to_another_thread_carries_the_scope() {
        let s = Scope::new("h", 0, None);
        let _e = enter(Arc::clone(&s));
        let cur = current();
        std::thread::spawn(move || {
            within(cur.as_ref(), || on_alloc(1 << 20));
        })
        .join()
        .unwrap();
        assert_eq!(s.used(), 1 << 20);
        on_free(1 << 20);
        flush();
    }

    #[test]
    fn outside_a_scope_nothing_fails() {
        on_alloc(64 << 20);
        assert!(checkpoint().is_ok());
        assert!(reserve(1 << 40).is_ok());
        on_free(64 << 20);
    }

    #[test]
    fn held_bytes_leave_the_gauge_when_dropped() {
        static GAUGE: AtomicI64 = AtomicI64::new(0);
        let a = Arc::new(Held::new(&GAUGE, 100));
        let b = Arc::clone(&a);
        assert_eq!(GAUGE.load(Relaxed), 100);
        drop(a);
        assert_eq!(GAUGE.load(Relaxed), 100, "a clone still holds it");
        drop(b);
        assert_eq!(GAUGE.load(Relaxed), 0);
    }
}
