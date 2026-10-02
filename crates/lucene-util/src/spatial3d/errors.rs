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

use std::cell::RefCell;

use super::{Error, Result};

thread_local! {
    static PENDING: RefCell<Option<Error>> = const { RefCell::new(None) };
}

/// Records `e` as thrown, unless an earlier exception already is: Java's
/// first throw is the one that propagates.
pub(crate) fn raise(e: Error) {
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
    let value = f();
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
    }
}
