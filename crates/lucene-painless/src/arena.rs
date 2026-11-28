//! The bytes a script may allocate in one invocation.
//!
//! Every string, array and collection a script makes is charged here, on the
//! thread that runs it. Past the limit the run is marked, and the next loop
//! iteration or the end of the invocation fails it with an exception; so
//! the peak is the limit plus one operation, and nothing accumulates across
//! invocations because each one begins at zero. A single value over the
//! per-value limit marks the run as well.

use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::vm::Throw;

/// Bytes one invocation may allocate under the limits a node sets: 4 MiB.
pub const DEFAULT_INVOCATION: usize = 4 << 20;

/// Bytes one string or collection may take under the limits a node sets: 1 MiB.
pub const DEFAULT_VALUE: usize = 1 << 20;

/// IR nodes one script may lower to under the limits a node sets.
pub const DEFAULT_IR_NODES: usize = 20_000;

// Unlimited until a node sets them, so a user of the library that never
// asks for limits runs scripts as before.
static INVOCATION: AtomicUsize = AtomicUsize::new(usize::MAX);
static VALUE: AtomicUsize = AtomicUsize::new(usize::MAX);
static IR_NODES: AtomicUsize = AtomicUsize::new(usize::MAX);

thread_local! {
    static USED: Cell<usize> = const { Cell::new(0) };
    static EXCEEDED: Cell<Option<usize>> = const { Cell::new(None) };
}

/// Sets the limits of every later invocation: `invocation` bytes in all and
/// `value` bytes in one string or collection. Process-wide.
pub fn set_limits(invocation: usize, value: usize) {
    INVOCATION.store(invocation, Ordering::Relaxed);
    VALUE.store(value, Ordering::Relaxed);
}

/// Sets the most IR nodes a script may lower to. Process-wide.
pub fn set_ir_nodes(nodes: usize) {
    IR_NODES.store(nodes, Ordering::Relaxed);
}

/// The most IR nodes a script may lower to now.
pub fn ir_nodes() -> usize {
    IR_NODES.load(Ordering::Relaxed)
}

/// The limits now: bytes an invocation and bytes a value.
pub fn limits() -> (usize, usize) {
    (
        INVOCATION.load(Ordering::Relaxed),
        VALUE.load(Ordering::Relaxed),
    )
}

/// Starts an invocation on this thread with nothing charged.
pub(crate) fn begin() {
    USED.with(|u| u.set(0));
    EXCEEDED.with(|e| e.set(None));
}

/// Charges `bytes` for one new value.
pub(crate) fn charge(bytes: usize) {
    let (invocation, value) = limits();
    let used = USED.with(|u| {
        let n = u.get().saturating_add(bytes);
        u.set(n);
        n
    });
    if bytes > value {
        EXCEEDED.with(|e| e.set(Some(value)));
    } else if used > invocation {
        EXCEEDED.with(|e| {
            if e.get().is_none() {
                e.set(Some(invocation));
            }
        });
    }
}

/// Bytes charged on this thread since the invocation began.
pub fn used() -> usize {
    USED.with(Cell::get)
}

/// Fails the invocation when it went past a limit.
///
/// # Errors
/// A `PainlessError` naming the limit.
pub(crate) fn check() -> Result<(), Throw> {
    match EXCEEDED.with(Cell::get) {
        None => Ok(()),
        Some(limit) => Err(Throw::new(
            "org.opensearch.painless.PainlessError",
            format!("script allocated more than the limit of {limit} bytes"),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One test: the limits are process-wide.
    #[test]
    fn a_value_or_an_invocation_past_its_limit_fails_the_run() {
        set_limits(1000, 600);
        begin();
        charge(500);
        assert!(check().is_ok());
        assert_eq!(used(), 500);
        charge(400);
        assert!(check().is_ok());
        charge(200);
        let e = check().expect_err("past the invocation limit");
        assert!(e.message().unwrap_or_default().contains("1000"));
        begin();
        assert!(check().is_ok(), "a new invocation starts clean");
        charge(601);
        assert!(check()
            .expect_err("past the value limit")
            .message()
            .unwrap_or_default()
            .contains("600"));
        set_limits(usize::MAX, usize::MAX);
        assert_eq!(limits(), (usize::MAX, usize::MAX));
        set_ir_nodes(7);
        assert_eq!(ir_nodes(), 7);
        set_ir_nodes(usize::MAX);
    }
}
