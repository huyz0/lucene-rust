//! The host's handle on the library's memory accounting
//! ([`lucene_util::memory`]): a scope per request, charged to the host's own
//! breaker through an upcall, and the process bound and counters.
//!
//! The plugin opens a scope around each search, aggregation and rescore
//! call on the calling thread ([`ffi_memory_scope_begin`], then
//! [`ffi_memory_scope_end`] after it), with a token of its own. When the
//! request grows past what it was granted, the library asks the
//! [`MemoryUpcall`] for more with that token; the plugin charges OpenSearch's
//! `request` circuit breaker, so native and Java memory of every concurrent
//! request share one limit, and refuses when the breaker trips. The request
//! then fails with [`FfiStatus::MemoryLimit`] at its next checkpoint, and the
//! plugin throws the breaker's own exception. Work the request hands to the
//! library's thread pool carries the scope with it.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, Weak};

use lucene_util::memory::{self, Entered, Reason, Scope};

use crate::error::{guard, set_last_error, FfiStatus};

/// `(token, bytes) -> 0` when the host granted `bytes` more to the scope
/// opened with `token`, anything else when it refused them. Called on any
/// thread of the request, never from inside the allocator.
pub type MemoryUpcall = unsafe extern "C" fn(i64, i64) -> i32;

static UPCALL: OnceLock<MemoryUpcall> = OnceLock::new();

/// Scopes open now, by token, for [`ffi_memory_scope_cancel`].
fn open_scopes() -> &'static Mutex<HashMap<i64, Weak<Scope>>> {
    static OPEN: OnceLock<Mutex<HashMap<i64, Weak<Scope>>>> = OnceLock::new();
    OPEN.get_or_init(Mutex::default)
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

thread_local! {
    /// The scopes this thread opened, innermost last.
    static ENTERED: std::cell::RefCell<Vec<(i64, Entered)>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Registers the grant upcall ([`MemoryUpcall`]), once per process. Without
/// one, a scope grows up to its hard limit without asking.
///
/// # Safety
/// `upcall` must have [`MemoryUpcall`]'s contract and stay callable for the
/// life of the process.
#[no_mangle]
pub unsafe extern "C" fn ffi_memory_set_upcall(upcall: Option<MemoryUpcall>) -> i32 {
    guard(|| {
        let upcall = upcall.ok_or(FfiStatus::NullPointer)?;
        let _ = UPCALL.set(upcall);
        Ok(())
    })
}

/// Opens a scope on this thread: what the next calls on it (and the pool
/// threads they hand work to) allocate is the request's `token`, at most
/// `hard_limit` bytes (`0`: only the host's grants bound it). `label` names
/// the request in messages.
///
/// # Safety
/// `label` valid for `label_len` bytes (null when 0).
#[no_mangle]
pub unsafe extern "C" fn ffi_memory_scope_begin(
    token: i64,
    hard_limit: i64,
    label: *const u8,
    label_len: usize,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        let label = unsafe { crate::raw::bytes_from_raw(label, label_len)? };
        let label = String::from_utf8_lossy(label).into_owned();
        let grant: Option<Box<memory::Grant>> = UPCALL.get().map(|&up| {
            Box::new(move |bytes: i64| {
                // SAFETY: the registered upcall's contract.
                unsafe { up(token, bytes) == 0 }
            }) as Box<memory::Grant>
        });
        let scope = Scope::new(label, hard_limit, grant);
        lock(open_scopes()).insert(token, Arc::downgrade(&scope));
        let entered = memory::enter(scope);
        ENTERED.with(|e| e.borrow_mut().push((token, entered)));
        Ok(())
    })
}

/// Closes this thread's innermost scope: `out[0..n]` gets the bytes granted
/// (the host releases them now), the most the request held, what it still
/// held at the end, and `1` when its deadline passed while it collected
/// ([`ffi_memory_scope_set_deadline`]), else `0`, as far as `n` reaches.
///
/// # Safety
/// `out` valid for `n` writes (null when 0).
#[no_mangle]
pub unsafe extern "C" fn ffi_memory_scope_end(out: *mut i64, n: usize) -> i32 {
    guard(|| {
        if out.is_null() && n > 0 {
            return Err(FfiStatus::NullPointer);
        }
        let Some((token, entered)) = ENTERED.with(|e| e.borrow_mut().pop()) else {
            set_last_error("no memory scope is open on this thread");
            return Err(FfiStatus::InvalidArgument);
        };
        let scope = Arc::clone(entered.scope());
        drop(entered);
        lock(open_scopes()).remove(&token);
        let values = [
            scope.granted(),
            scope.peak(),
            scope.used(),
            i64::from(scope.timed_out()),
        ];
        for (i, v) in values.iter().take(n).enumerate() {
            // SAFETY: `out` is valid for `n > i` writes (caller contract).
            unsafe { *out.add(i) = *v };
        }
        Ok(())
    })
}

/// Tells the request opened with `token` to stop at its next checkpoint;
/// it then fails with [`FfiStatus::Cancelled`]. Any thread may call it; a
/// token with no open scope is a no-op.
#[no_mangle]
pub extern "C" fn ffi_memory_scope_cancel(token: i64) -> i32 {
    guard(|| {
        let scope = lock(open_scopes()).get(&token).and_then(Weak::upgrade);
        if let Some(s) = scope {
            s.cancel();
        }
        Ok(())
    })
}

/// Gives the request opened with `token` a deadline `nanos` from now (`0` or
/// less: already passed): once it passes, the request collects no more and
/// answers with what it collected, a search `timeout`'s partial result. Any
/// thread may call it; a token with no open scope is a no-op.
#[no_mangle]
pub extern "C" fn ffi_memory_scope_set_deadline(token: i64, nanos: i64) -> i32 {
    guard(|| {
        let scope = lock(open_scopes()).get(&token).and_then(Weak::upgrade);
        if let Some(s) = scope {
            s.set_deadline(std::time::Duration::from_nanos(
                u64::try_from(nanos).unwrap_or(0),
            ));
        }
        Ok(())
    })
}

/// Sets the process bound in bytes (`0`: none): past it, requests fail at
/// their next checkpoint.
#[no_mangle]
pub extern "C" fn ffi_memory_set_process_limit(bytes: i64) -> i32 {
    guard(|| {
        memory::set_process_limit(bytes);
        Ok(())
    })
}

/// The accounting's counters into `out[0..n]`: bytes held now, the most
/// held, the process bound, requests stopped by their grant, requests
/// stopped by the process bound, requests cancelled; then the query caches'
/// bytes, their node budget and sets evicted, decoded sort columns' bytes,
/// the global ordinals' bytes (what the host charges its `fielddata`
/// breaker), and requests that stopped collecting at their deadline.
///
/// # Safety
/// `out` valid for `n` writes (null when 0).
#[no_mangle]
pub unsafe extern "C" fn ffi_memory_stats(out: *mut i64, n: usize) -> i32 {
    guard(|| {
        if out.is_null() && n > 0 {
            return Err(FfiStatus::NullPointer);
        }
        let s = memory::stats();
        let (qc_bytes, qc_budget, qc_evictions) = lucene_search::query_cache_budget::node_stats();
        let i = |v: usize| i64::try_from(v).unwrap_or(i64::MAX);
        let values = [
            s.live,
            s.peak,
            s.process_limit,
            s.request_trips,
            s.process_trips,
            s.cancelled,
            i(qc_bytes),
            i(qc_budget),
            i64::try_from(qc_evictions).unwrap_or(i64::MAX),
            i(lucene_search::sort_column_bytes()),
            lucene_search::ordinal_map::held_bytes(),
            s.timed_out,
        ];
        for (i, v) in values.iter().take(n).enumerate() {
            // SAFETY: `out` is valid for `n > i` writes (caller contract).
            unsafe { *out.add(i) = *v };
        }
        Ok(())
    })
}

/// Sets the bytes every segment's query cache may hold together (the
/// plugin passes `indices.queries.cache.size`); past them the least recently
/// used sets go, whichever segment holds them.
#[no_mangle]
pub extern "C" fn ffi_query_cache_set_budget(bytes: i64) -> i32 {
    guard(|| {
        lucene_search::query_cache_budget::set_node_budget(
            usize::try_from(bytes.max(0)).unwrap_or(usize::MAX),
        );
        Ok(())
    })
}

/// The status and message for a request a bound or cancel stopped.
pub(crate) fn exceeded(e: &memory::Exceeded) -> FfiStatus {
    set_last_error(e.message.clone());
    match e.reason {
        Reason::Cancelled => FfiStatus::Cancelled,
        Reason::Request | Reason::Process => FfiStatus::MemoryLimit,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_scope_opens_and_closes_on_its_thread() {
        let label = b"search";
        // SAFETY: valid label.
        assert_eq!(
            unsafe { ffi_memory_scope_begin(7, 0, label.as_ptr(), label.len()) },
            0
        );
        let mut out = [9i64; 3];
        // SAFETY: valid out.
        assert_eq!(unsafe { ffi_memory_scope_end(out.as_mut_ptr(), 3) }, 0);
        assert_eq!(out[0], 0, "nothing asked for");
        // SAFETY: valid out.
        let rc = unsafe { ffi_memory_scope_end(out.as_mut_ptr(), 3) };
        assert_eq!(
            rc,
            FfiStatus::InvalidArgument.code(),
            "nothing left to close"
        );
    }

    #[test]
    fn cancel_reaches_the_open_scope_by_token() {
        let label = b"c";
        // SAFETY: valid label.
        assert_eq!(
            unsafe { ffi_memory_scope_begin(42, 0, label.as_ptr(), label.len()) },
            0
        );
        std::thread::spawn(|| assert_eq!(ffi_memory_scope_cancel(42), 0))
            .join()
            .unwrap();
        let e = memory::checkpoint().unwrap_err();
        assert_eq!(exceeded(&e), FfiStatus::Cancelled);
        // SAFETY: null out with n = 0.
        assert_eq!(unsafe { ffi_memory_scope_end(std::ptr::null_mut(), 0) }, 0);
        assert_eq!(ffi_memory_scope_cancel(42), 0, "closed: a no-op");
    }

    #[test]
    fn a_deadline_reaches_the_open_scope_and_its_end_reports_it() {
        let label = b"d";
        // SAFETY: valid label.
        assert_eq!(
            unsafe { ffi_memory_scope_begin(43, 0, label.as_ptr(), label.len()) },
            0
        );
        assert_eq!(ffi_memory_scope_set_deadline(43, 3_600_000_000_000), 0);
        assert!(!memory::past_deadline());
        std::thread::spawn(|| assert_eq!(ffi_memory_scope_set_deadline(43, -1), 0))
            .join()
            .unwrap();
        assert!(memory::past_deadline());
        let mut out = [9i64; 4];
        // SAFETY: valid out.
        assert_eq!(unsafe { ffi_memory_scope_end(out.as_mut_ptr(), 4) }, 0);
        assert_eq!(out[3], 1);
        assert_eq!(ffi_memory_scope_set_deadline(43, 0), 0, "closed: a no-op");
    }

    #[test]
    fn stats_report_the_counters() {
        let mut out = [0i64; 12];
        // SAFETY: valid out.
        assert_eq!(unsafe { ffi_memory_stats(out.as_mut_ptr(), 12) }, 0);
        assert!(out[11] >= 0, "requests stopped at their deadline");
        assert!(out[1] >= out[0].min(0));
        assert!(out[7] > 0, "the query caches have a budget");
    }
}
