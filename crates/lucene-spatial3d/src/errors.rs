//! Exceptions thrown from inside geo3d's infallible interface methods.
//!
//! A handful of Java methods behind `isWithin`, the distance computations
//! and the bounds throw at query time -- `GeoComplexPolygon`'s "No
//! dual-plane travel strategies were found", `GeoStandardPath`'s "Can't find
//! world intersection" -- and the exception unwinds through every caller up
//! to whoever asked. Making every `is_within`/`compute_distance` in the
//! port return a `Result` for those rare cases would put a branch and a
//! `?` on every hot membership test. Instead the throwing site calls
//! [`raise`], which records the first exception in a thread-local slot and
//! returns a placeholder value, and the entry points that Java callers see
//! (`PointInGeo3DShapeQuery`, the distance comparators, the differential
//! tests) run under [`catch`], which turns a recorded exception back into
//! an `Err` -- the same answer Java gives, since Java's partial result is
//! discarded with the exception. Where Java itself catches (a `try` around a
//! call that may throw), the port wraps the call in [`catch`] too.
//!
//! A [`raise`] with no [`catch`] around it would have nobody to report to:
//! the placeholder value would be taken as the answer (failing open) and
//! the exception would sit in the slot. A catch-depth counter makes that a
//! `debug_assert!` failure in tests, and in release the stray exception is
//! dropped rather than left pending for an unrelated later [`catch`].

use std::cell::{Cell, RefCell};

use super::{Error, Result};

thread_local! {
    static PENDING: RefCell<Option<Error>> = const { RefCell::new(None) };
    /// How many [`catch`]es are running on this thread.
    static DEPTH: Cell<u32> = const { Cell::new(0) };
}

/// Leaves one [`catch`] level, also when `f` unwinds: a panic caught at
/// the FFI must not leave the thread believing it is still inside a catch.
struct DepthGuard;

impl Drop for DepthGuard {
    fn drop(&mut self) {
        DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
    }
}

/// Records `e` as thrown, unless an earlier exception already is: Java's
/// first throw is the one that propagates.
///
/// Must run inside a [`catch`]; outside one it is a port bug (a call site
/// that lets the placeholder through as the answer), asserted in debug
/// builds and ignored in release.
pub(crate) fn raise(e: Error) {
    let depth = DEPTH.with(Cell::get);
    debug_assert!(depth > 0, "spatial3d raise outside catch: {e}");
    if depth == 0 {
        return;
    }
    PENDING.with(|p| {
        let mut p = p.borrow_mut();
        if p.is_none() {
            *p = Some(e);
        }
    });
}

/// Runs `f`; `Err` with the first exception `f` raised, if any. Nests: an
/// exception pending outside is set aside and restored.
pub fn catch<T>(f: impl FnOnce() -> T) -> Result<T> {
    let outer = PENDING.with(|p| p.borrow_mut().take());
    DEPTH.with(|d| d.set(d.get().saturating_add(1)));
    let guard = DepthGuard;
    let value = f();
    drop(guard);
    let inner = PENDING.with(|p| std::mem::replace(&mut *p.borrow_mut(), outer));
    match inner {
        Some(e) => Err(e),
        None => Ok(value),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_raise_wins_and_catch_nests() {
        let r = catch(|| {
            raise(Error::Runtime("a".into()));
            raise(Error::Runtime("b".into()));
            let inner = catch(|| {
                raise(Error::IllegalArgument("c".into()));
                1
            });
            assert_eq!(inner, Err(Error::IllegalArgument("c".into())));
            2
        });
        assert_eq!(r, Err(Error::Runtime("a".into())));
        assert_eq!(catch(|| 3), Ok(3));
        assert_eq!(DEPTH.with(Cell::get), 0);
    }

    /// A raise outside any catch is a port bug: a debug assertion, and in
    /// release nothing is left pending for a later catch to report.
    #[test]
    fn raise_outside_catch_is_flagged() {
        let r = std::panic::catch_unwind(|| raise(Error::Runtime("stray".into())));
        assert_eq!(r.is_err(), cfg!(debug_assertions));
        assert_eq!(catch(|| 4), Ok(4));
    }

    /// A panic unwinding out of `f` still leaves the catch level.
    #[test]
    fn depth_is_restored_when_f_panics() {
        let r = std::panic::catch_unwind(|| catch(|| panic!("boom")));
        assert!(r.is_err());
        assert_eq!(DEPTH.with(Cell::get), 0);
    }
}
