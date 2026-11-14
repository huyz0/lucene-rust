//! The status every engine entry point reports, and the thread-local message
//! that explains a failure.
//!
//! A failing call leaves one message for its thread ([`set_last_error`]);
//! the boundary reads it back with [`last_error`]. A call that failed
//! without recording one is given [`Status::default_message`], so a caller
//! never reads an older failure's text.

use std::cell::{Cell, RefCell};

/// Every entry point's return code: `0` is success, any other value a stable
/// failure reason the caller can branch on without parsing the message.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Ok = 0,
    NullPointer = 1,
    InvalidUtf8 = 2,
    InvalidHandle = 3,
    Io = 4,
    Decode = 5,
    Search = 6,
    IndexOutOfBounds = 7,
    BufferTooSmall = 8,
    Panic = 9,
    InvalidArgument = 10,
    /// A registry cannot issue any more handles of that kind. Only
    /// reachable by leaking handles (never closing them); the fix is on the
    /// caller's side, so this is a distinct code rather than a generic
    /// failure.
    HandleLimit = 11,
    /// The request outgrew the memory its host granted it, or the process
    /// is over its bound; the message says which.
    MemoryLimit = 12,
    /// The host cancelled the request mid-call.
    Cancelled = 13,
    /// An aggregation made more buckets than the request allows; the last
    /// error is the count, in decimal.
    TooManyBuckets = 14,
}

impl Status {
    pub fn code(self) -> i32 {
        self as i32
    }

    /// The message recorded for a failing call that returned this status
    /// without calling [`set_last_error`] itself: no error path may leave
    /// the thread-local slot holding an *older*, unrelated failure's text.
    pub fn default_message(self) -> &'static str {
        match self {
            Status::Ok => "ok",
            Status::NullPointer => {
                "null pointer: a required pointer argument was null (no further detail recorded)"
            }
            Status::InvalidUtf8 => "invalid UTF-8 in a caller-supplied string argument",
            Status::InvalidHandle => "unknown, already-closed, or wrong-kind handle",
            Status::Io => "I/O error",
            Status::Decode => "decode error",
            Status::Search => "search failed",
            Status::IndexOutOfBounds => "index out of bounds",
            Status::BufferTooSmall => {
                "caller-allocated buffer too small: call the matching `*_len` accessor first"
            }
            Status::Panic => "a Rust panic was caught at the FFI boundary",
            Status::InvalidArgument => "invalid argument",
            Status::HandleLimit => "handle registry exhausted: too many open handles",
            Status::MemoryLimit => "the request is over its memory limit",
            Status::Cancelled => "the request was cancelled",
            Status::TooManyBuckets => "an aggregation made too many buckets",
        }
    }
}

thread_local! {
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
    /// Set by [`set_last_error`], cleared by [`begin_call`]. Lets the
    /// boundary tell "this call recorded its own message" from "this call
    /// failed but left the slot holding a *previous* call's message".
    static ERROR_RECORDED: Cell<bool> = const { Cell::new(false) };
}

/// Records `message` as this thread's last error, replacing the previous one.
pub fn set_last_error(message: impl Into<String>) {
    ERROR_RECORDED.with(|c| c.set(true));
    LAST_ERROR.with(|slot| *slot.borrow_mut() = message.into());
}

/// This thread's last recorded error message.
pub fn last_error() -> String {
    LAST_ERROR.with(|slot| slot.borrow().clone())
}

/// Starts a call: nothing is recorded for it yet ([`error_recorded`]).
pub fn begin_call() {
    ERROR_RECORDED.with(|c| c.set(false));
}

/// Whether this thread recorded an error since [`begin_call`].
pub fn error_recorded() -> bool {
    ERROR_RECORDED.with(|c| c.get())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_recorded_error_is_read_back_and_flagged() {
        begin_call();
        assert!(!error_recorded());
        set_last_error("boom");
        assert!(error_recorded());
        assert_eq!(last_error(), "boom");
    }

    #[test]
    fn every_status_has_a_nonempty_default_message() {
        for status in [
            Status::Ok,
            Status::NullPointer,
            Status::InvalidUtf8,
            Status::InvalidHandle,
            Status::Io,
            Status::Decode,
            Status::Search,
            Status::IndexOutOfBounds,
            Status::BufferTooSmall,
            Status::Panic,
            Status::InvalidArgument,
            Status::HandleLimit,
            Status::MemoryLimit,
            Status::Cancelled,
            Status::TooManyBuckets,
        ] {
            assert!(!status.default_message().is_empty(), "{status:?}");
        }
    }
}

thread_local! {
    /// The shard paths the running call read through ([`note_shard`]).
    static CALL_SHARDS: std::cell::RefCell<Vec<std::sync::Arc<str>>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Notes that the running call reads the shard at `path`: if a page of a
/// file under it fails to decode (an encrypted chunk that fails
/// authentication) or cannot be read from disk while the call runs, the
/// boundary's panic guard fails the call with that error.
///
/// Noting a shard makes the call checked
/// ([`lucene_store::read_failures::set_checked`]) until it ends: only then
/// may a page that cannot be read be handed to it as zeros, since only then
/// does the failure fail the call.
pub fn note_shard(path: &std::sync::Arc<str>) {
    lucene_store::read_failures::set_checked(true);
    CALL_SHARDS.with(|c| {
        let mut shards = c.borrow_mut();
        if !shards.iter().any(|s| std::sync::Arc::ptr_eq(s, path)) {
            shards.push(std::sync::Arc::clone(path));
        }
    });
}

/// Runs `f` over the shards the running call has noted.
pub fn call_shards<R>(f: impl FnOnce(&[std::sync::Arc<str>]) -> R) -> R {
    CALL_SHARDS.with(|c| f(&c.borrow()))
}

/// Forgets the noted shards, as a call begins.
pub fn clear_call_shards() {
    CALL_SHARDS.with(|c| c.borrow_mut().clear());
}
