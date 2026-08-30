//! Status codes and the panic/last-error machinery every exported function
//! in this crate goes through — see the `ffi-safety` skill: "a panic must
//! never unwind into the JVM" and "all exported calls return a status code".
//!
//! Every `extern "C" fn` in this crate is a thin wrapper around
//! [`guard`], which:
//! 1. Runs the real body inside [`std::panic::catch_unwind`].
//! 2. On success, converts the body's `Result<(), FfiStatus>` into the
//!    plain `i32` the C ABI returns (`0` on `Ok`).
//! 3. On a caught panic, records a message in a thread-local slot (read
//!    back via [`crate::ffi_get_last_error_message`]) and returns
//!    [`FfiStatus::Panic`] instead of propagating the unwind.
//! 4. On *any* non-`Ok` return, guarantees the thread-local message slot
//!    describes **this** call: a body that returned a status without
//!    calling [`set_last_error`] itself (a bare null-pointer or
//!    buffer-too-small check, say) gets [`FfiStatus::default_message`]
//!    written for it. Without that backfill,
//!    [`crate::ffi_get_last_error_message`] would hand the caller a
//!    *previous*, unrelated failure's text -- a stale message read as a
//!    diagnosis of the call that just failed, which is worse than an
//!    uninformative one. The one exported function outside this invariant
//!    is [`crate::ffi_get_last_error_message`] itself, which bypasses
//!    [`guard`] deliberately: its own `BufferTooSmall`/`NullPointer`
//!    returns must *not* overwrite the message the caller is retrying to
//!    read, so it records nothing and leaves the slot intact for the retry.
//!
//! `catch_unwind`'s closure must be [`std::panic::UnwindSafe`]; every raw
//! pointer this crate receives is `*const`/`*mut`, and raw pointers are
//! already `UnwindSafe` (unlike `&mut T`, a raw pointer carries no
//! compiler-enforced invariant a panic could leave torn), so every exported
//! function's [`guard`]-wrapped closure — which only ever captures raw
//! pointers and other `Copy` primitives (handles, lengths) — satisfies
//! `UnwindSafe` on its own. No `AssertUnwindSafe` wrapping is used anywhere
//! in this crate.

use std::cell::RefCell;
use std::os::raw::c_char;
use std::sync::Once;

/// Every exported function's return code. `0` (`Ok`) is success; every
/// other value is a specific, stable failure reason a JNI caller can branch
/// on without parsing a string (the string is available too, via
/// [`crate::ffi_get_last_error_message`], for logging).
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfiStatus {
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
    /// A registry cannot issue any more handles of that kind -- see
    /// [`crate::handle::MAX_SLOTS`]. Only reachable by leaking handles
    /// (never closing them); the fix is on the caller's side, so this is a
    /// distinct code rather than a generic failure.
    HandleLimit = 11,
}

impl FfiStatus {
    pub fn code(self) -> i32 {
        self as i32
    }

    /// The message [`guard`] records when a failing body returned this
    /// status without calling [`set_last_error`] itself -- see `guard`'s doc
    /// comment for why no error path may leave the thread-local slot holding
    /// an *older*, unrelated failure's text.
    fn default_message(self) -> &'static str {
        match self {
            FfiStatus::Ok => "ok",
            FfiStatus::NullPointer => {
                "null pointer: a required pointer argument was null (no further detail recorded)"
            }
            FfiStatus::InvalidUtf8 => "invalid UTF-8 in a caller-supplied string argument",
            FfiStatus::InvalidHandle => "unknown, already-closed, or wrong-kind handle",
            FfiStatus::Io => "I/O error",
            FfiStatus::Decode => "decode error",
            FfiStatus::Search => "search failed",
            FfiStatus::IndexOutOfBounds => "index out of bounds",
            FfiStatus::BufferTooSmall => {
                "caller-allocated buffer too small: call the matching `*_len` accessor first"
            }
            FfiStatus::Panic => "a Rust panic was caught at the FFI boundary",
            FfiStatus::InvalidArgument => "invalid argument",
            FfiStatus::HandleLimit => "handle registry exhausted: too many open handles",
        }
    }
}

thread_local! {
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
    /// Set only while a [`guard`] call's `catch_unwind`-wrapped body is
    /// running on this thread -- lets [`install_panic_hook`]'s process-wide
    /// hook write straight to this thread's own [`LAST_ERROR`] slot without
    /// touching any other thread's panic (see `install_panic_hook`'s doc
    /// comment for why the hook, not `Any` downcasting, is this module's
    /// message-capture mechanism).
    static CAPTURING_PANIC: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Set by [`set_last_error`], cleared by [`guard`] before each body runs.
    /// Lets `guard` tell "this call recorded its own message" from "this call
    /// failed but left the slot holding a *previous* call's message", so the
    /// second case can be backfilled with [`FfiStatus::default_message`]
    /// rather than reported as if it were this call's diagnosis.
    static ERROR_RECORDED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

static INSTALL_PANIC_HOOK: Once = Once::new();

/// Installs a process-wide panic hook (once) that -- for any thread
/// currently inside a [`guard`] call -- writes the panic's formatted
/// message into that thread's [`LAST_ERROR`] slot.
///
/// **Why a hook instead of downcasting the `catch_unwind` payload
/// (`Box<dyn Any + Send>`) to `&str`/`String`**: the ordinary approach
/// (`payload.downcast_ref::<&str>()`) is what real Lucene-in-Rust FFI code
/// would reach for first, and it is what this module used originally: it
/// works for a plain literal (`panic!("boom")`, whose payload actually is a
/// `&'static str`) and a formatted one (`panic!("boom: {x}")`, whose
/// payload is a `String`). In this environment that downcast intermittently
/// reports neither type matches, even though a hook installed for the same
/// panic observes the correct payload type just before unwinding starts
/// (verified directly: `PanicHookInfo::payload().is::<&str>()` is `true` at
/// hook time for the same panic where `catch_unwind`'s returned
/// `Box<dyn Any + Send>` later fails both downcasts) -- i.e. some part of
/// this toolchain's `Box<dyn Any + Send>` plumbing does not preserve a
/// stable `TypeId` from hook time to `catch_unwind`-return time here. Rather
/// than build on that flaky primitive, this module captures the message at
/// the one point it's reliably observable -- inside the hook, via
/// `PanicHookInfo`'s `Display` impl (`"panicked at file:line:col:\nmessage"`,
/// the same text the default hook would print) -- and stores it directly,
/// sidestepping the downcast entirely.
fn install_panic_hook() {
    INSTALL_PANIC_HOOK.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info: &std::panic::PanicHookInfo<'_>| {
            if CAPTURING_PANIC.with(|c| c.get()) {
                ERROR_RECORDED.with(|c| c.set(true));
                LAST_ERROR.with(|slot| *slot.borrow_mut() = format!("panic: {info}"));
            } else {
                previous(info);
            }
        }));
    });
}

/// Records `message` in this thread's last-error slot, overwriting any
/// previous message. Called on every non-`Ok` path so
/// [`crate::ffi_get_last_error_message`] always reflects the most recent
/// failure on the calling thread (JNI callers are expected to check the
/// status code and only then, if non-zero, fetch the message — matching a
/// plain `errno`-style contract).
pub fn set_last_error(message: impl Into<String>) {
    ERROR_RECORDED.with(|c| c.set(true));
    LAST_ERROR.with(|slot| *slot.borrow_mut() = message.into());
}

fn last_error() -> String {
    LAST_ERROR.with(|slot| slot.borrow().clone())
}

/// Copies the calling thread's last-error message into `buf` (a
/// caller-allocated buffer of `buf_len` bytes), NUL-terminated, writing the
/// number of bytes written (excluding the NUL) to `*out_written`.
///
/// Returns [`FfiStatus::BufferTooSmall`] (without writing anything to
/// `buf`) if `buf_len` is too small to hold the message plus its
/// terminating NUL — the caller can retry with a larger buffer; the
/// message itself is left in the thread-local slot for a retry to read
/// (this call does not clear it).
///
/// # Safety
/// `buf` must be valid for writes of `buf_len` bytes, and `out_written`
/// must be valid for a single `usize` write (or null, in which case the
/// written length is not reported).
pub unsafe fn get_last_error_message(
    buf: *mut c_char,
    buf_len: usize,
    out_written: *mut usize,
) -> i32 {
    let message = last_error();
    let bytes = message.as_bytes();
    if bytes.len() + 1 > buf_len {
        return FfiStatus::BufferTooSmall.code();
    }
    if buf.is_null() {
        return FfiStatus::NullPointer.code();
    }
    // SAFETY: caller contract guarantees `buf` is valid for `buf_len` bytes,
    // and `bytes.len() + 1 <= buf_len` was just checked above.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr() as *const c_char, buf, bytes.len());
        *buf.add(bytes.len()) = 0;
    }
    if !out_written.is_null() {
        // SAFETY: caller contract guarantees `out_written` is valid for one write.
        unsafe {
            *out_written = bytes.len();
        }
    }
    FfiStatus::Ok.code()
}

/// Runs `body` under `catch_unwind`, converting a caught panic into
/// [`FfiStatus::Panic`] plus a last-error message instead of letting the
/// unwind cross into the JVM (see this module's doc comment). Every
/// exported function's implementation is this one call.
///
/// Also guarantees the "every non-`Ok` status leaves a retrievable message
/// about *this* call" invariant, backfilling
/// [`FfiStatus::default_message`] when `body` returned an error without
/// recording one -- see step 4 of this module's doc comment.
pub fn guard<F>(body: F) -> i32
where
    F: FnOnce() -> Result<(), FfiStatus> + std::panic::UnwindSafe,
{
    install_panic_hook();
    ERROR_RECORDED.with(|c| c.set(false));
    CAPTURING_PANIC.with(|c| c.set(true));
    let outcome = std::panic::catch_unwind(body);
    CAPTURING_PANIC.with(|c| c.set(false));
    match outcome {
        Ok(Ok(())) => FfiStatus::Ok.code(),
        Ok(Err(status)) => {
            // Every non-`Ok` return must leave a message describing *this*
            // call -- a body that returned early without calling
            // `set_last_error` (e.g. a bare null-pointer or buffer-too-small
            // check) would otherwise leave the slot holding an older,
            // unrelated failure, which is worse than no message at all.
            if !ERROR_RECORDED.with(|c| c.get()) {
                set_last_error(status.default_message());
            }
            status.code()
        }
        Err(_payload) => {
            // The installed panic hook (see `install_panic_hook`) already wrote
            // this thread's formatted panic message into `LAST_ERROR` while
            // `CAPTURING_PANIC` was `true` above -- nothing left to do here but
            // report the status code.
            FfiStatus::Panic.code()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guard_ok_returns_zero() {
        assert_eq!(guard(|| Ok(())), FfiStatus::Ok.code());
    }

    #[test]
    fn guard_err_returns_status_code_and_sets_message() {
        let code = guard(|| {
            set_last_error("boom");
            Err(FfiStatus::Decode)
        });
        assert_eq!(code, FfiStatus::Decode.code());
        assert_eq!(last_error(), "boom");
    }

    #[test]
    fn panic_outside_guard_falls_through_to_the_previous_hook() {
        // A panic on a thread that never called `guard` (so `CAPTURING_PANIC`
        // is still `false` there) must fall through to whatever hook was
        // installed before this crate's -- exercising `install_panic_hook`'s
        // `else` branch. `install_panic_hook` runs at least once before this
        // test via every other `guard`-calling test, so the `Once` here is a
        // no-op re-confirmation, not a fresh install.
        install_panic_hook();
        let result = std::thread::spawn(|| {
            panic!("panic on a non-capturing thread");
        })
        .join();
        assert!(result.is_err());
    }

    // These two tests deliberately panic inside `guard` and rely on this
    // crate's own installed panic hook (see `install_panic_hook`'s doc
    // comment) to suppress the default panic printout; they do not touch
    // `std::panic::set_hook` themselves, since only one process-wide hook
    // can be installed and `install_panic_hook`'s `Once` guarantees this
    // crate's hook -- which every other test also depends on -- is the one
    // left in place after this test returns.

    #[test]
    fn guard_catches_panic_and_reports_panic_status() {
        let code = guard(|| panic!("deliberate test panic"));
        assert_eq!(code, FfiStatus::Panic.code());
        assert!(last_error().contains("deliberate test panic"));
    }

    #[test]
    fn guard_catches_string_panic_payload() {
        let code = guard(|| panic!("{}", String::from("owned message")));
        assert_eq!(code, FfiStatus::Panic.code());
        assert!(last_error().contains("owned message"));
    }

    #[test]
    fn guard_backfills_a_message_when_the_body_records_none() {
        // A previous, unrelated failure leaves its message in the slot...
        assert_eq!(
            guard(|| {
                set_last_error("an older, unrelated failure");
                Err(FfiStatus::Decode)
            }),
            FfiStatus::Decode.code()
        );
        // ...and the next failing call, which records nothing itself, must
        // not let that stale text be read back as its own diagnosis.
        let code = guard(|| Err(FfiStatus::NullPointer));
        assert_eq!(code, FfiStatus::NullPointer.code());
        let msg = last_error();
        assert!(
            !msg.contains("older, unrelated"),
            "stale message leaked: {msg}"
        );
        assert_eq!(msg, FfiStatus::NullPointer.default_message());
    }

    #[test]
    fn guard_keeps_the_bodys_own_message_when_it_recorded_one() {
        let code = guard(|| {
            set_last_error("specific detail from the body");
            Err(FfiStatus::InvalidArgument)
        });
        assert_eq!(code, FfiStatus::InvalidArgument.code());
        assert_eq!(last_error(), "specific detail from the body");
    }

    #[test]
    fn every_status_has_a_nonempty_default_message() {
        for status in [
            FfiStatus::Ok,
            FfiStatus::NullPointer,
            FfiStatus::InvalidUtf8,
            FfiStatus::InvalidHandle,
            FfiStatus::Io,
            FfiStatus::Decode,
            FfiStatus::Search,
            FfiStatus::IndexOutOfBounds,
            FfiStatus::BufferTooSmall,
            FfiStatus::Panic,
            FfiStatus::InvalidArgument,
            FfiStatus::HandleLimit,
        ] {
            assert!(!status.default_message().is_empty(), "{status:?}");
        }
    }

    #[test]
    fn get_last_error_message_roundtrips_through_buffer() {
        set_last_error("hello");
        let mut buf = [0 as c_char; 16];
        let mut written: usize = 0;
        let code =
            unsafe { get_last_error_message(buf.as_mut_ptr(), buf.len(), &mut written as *mut _) };
        assert_eq!(code, FfiStatus::Ok.code());
        assert_eq!(written, 5);
        let s = unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) };
        assert_eq!(s.to_str().unwrap(), "hello");
    }

    #[test]
    fn get_last_error_message_reports_buffer_too_small() {
        set_last_error("a longer message than the buffer");
        let mut buf = [0 as c_char; 4];
        let code =
            unsafe { get_last_error_message(buf.as_mut_ptr(), buf.len(), std::ptr::null_mut()) };
        assert_eq!(code, FfiStatus::BufferTooSmall.code());
    }

    #[test]
    fn get_last_error_message_null_buf_is_null_pointer_error() {
        set_last_error("");
        let code = unsafe { get_last_error_message(std::ptr::null_mut(), 0, std::ptr::null_mut()) };
        // Empty message + NUL fits in buf_len == 0? No: 0+1 > 0, so BufferTooSmall
        // wins before the null check runs -- verify that ordering explicitly.
        assert_eq!(code, FfiStatus::BufferTooSmall.code());
    }

    #[test]
    fn get_last_error_message_null_buf_with_room_is_null_pointer_error() {
        set_last_error("");
        let mut written: usize = 0;
        let code =
            unsafe { get_last_error_message(std::ptr::null_mut(), 1, &mut written as *mut _) };
        assert_eq!(code, FfiStatus::NullPointer.code());
    }

    #[test]
    fn get_last_error_message_null_out_written_is_fine() {
        set_last_error("ok");
        let mut buf = [0 as c_char; 8];
        let code =
            unsafe { get_last_error_message(buf.as_mut_ptr(), buf.len(), std::ptr::null_mut()) };
        assert_eq!(code, FfiStatus::Ok.code());
    }
}
