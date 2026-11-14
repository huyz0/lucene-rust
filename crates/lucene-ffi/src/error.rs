//! Status codes and the panic and last-error handling every exported
//! function goes through.
//!
//! Each exported function's body runs inside [`guard`], which catches a panic
//! (a panic must never unwind into the JVM) and turns the body's
//! `Result<(), FfiStatus>` into the `i32` it returns, `0` on success. Every
//! non-`Ok` return leaves a thread-local message about that call: a body that
//! returned an error without recording one gets [`FfiStatus::default_message`],
//! so the caller never reads an older failure's text. The plugin reads the
//! message with `ffi_jvm_last_error`.
//!
//! The guarded closures capture only raw pointers and `Copy` values, which
//! are `UnwindSafe`, so no `AssertUnwindSafe` is needed.

use std::sync::Once;

/// Every exported function's return code ([`lucene_engine::Status`]): `0`
/// is success, any other value a stable failure reason the caller can
/// branch on without parsing the message.
pub use lucene_engine::status::Status as FfiStatus;
pub use lucene_engine::status::{last_error, set_last_error};

thread_local! {
    /// Set only while a [`guard`] call's `catch_unwind`-wrapped body is
    /// running on this thread -- lets [`install_panic_hook`]'s process-wide
    /// hook write straight to this thread's own the last-error slot slot without
    /// touching any other thread's panic (see `install_panic_hook`'s doc
    /// comment for why the hook, not `Any` downcasting, is this module's
    /// message-capture mechanism).
    static CAPTURING_PANIC: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// The shard paths the running [`guard`] call read through
    /// (`lucene_engine::status::note_shard`), and how deep in nested guard calls this thread is.
    static GUARD_DEPTH: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    /// [`lucene_store::read_failures::epoch`] when the outermost running
    /// [`guard`] call started.
    static CALL_EPOCH: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// A recorded read failure under one of the shards the running call read
/// ([`lucene_store::read_failures`]); one atomic load while none is.
fn shard_read_failure() -> Option<String> {
    if lucene_store::read_failures::count() == 0 {
        return None;
    }
    let since = CALL_EPOCH.with(|e| e.get());
    lucene_engine::status::call_shards(|shards| {
        shards
            .iter()
            .find_map(|path| lucene_store::read_failures::failure_under(path, since))
    })
}

static INSTALL_PANIC_HOOK: Once = Once::new();

/// Installs a process-wide panic hook (once) that -- for any thread
/// currently inside a [`guard`] call -- writes the panic's formatted
/// message into that thread's the last-error slot slot.
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
                set_last_error(format!("panic: {info}"));
            } else {
                previous(info);
            }
        }));
    });
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
    crate::register_codecs();
    lucene_engine::status::begin_call();
    let depth = GUARD_DEPTH.with(|d| {
        let depth = d.get();
        d.set(depth.saturating_add(1));
        depth
    });
    if depth == 0 {
        lucene_engine::status::clear_call_shards();
        CALL_EPOCH.with(|e| e.set(lucene_store::read_failures::epoch()));
    }
    CAPTURING_PANIC.with(|c| c.set(true));
    let outcome = std::panic::catch_unwind(body);
    GUARD_DEPTH.with(|d| d.set(depth));
    if depth == 0 {
        lucene_store::read_failures::set_checked(false);
    }
    // A nested call returns into its caller's guard, which is still running.
    CAPTURING_PANIC.with(|c| c.set(depth > 0));
    match outcome {
        Ok(Ok(())) => match shard_read_failure() {
            // The call may have read a page that failed to decode or could
            // not be read, and zeros stood in for it: its result cannot be
            // trusted.
            Some(message) => {
                set_last_error(format!("reading the shard failed: {message}"));
                FfiStatus::Io.code()
            }
            None => FfiStatus::Ok.code(),
        },
        Ok(Err(status)) => {
            // Every non-`Ok` return must leave a message describing *this*
            // call -- a body that returned early without calling
            // `set_last_error` (e.g. a bare null-pointer or buffer-too-small
            // check) would otherwise leave the slot holding an older,
            // unrelated failure, which is worse than no message at all.
            if !lucene_engine::status::error_recorded() {
                set_last_error(status.default_message());
            }
            status.code()
        }
        Err(_payload) => {
            // The installed panic hook (see `install_panic_hook`) already wrote
            // this thread's formatted panic message into the last-error slot while
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
}
