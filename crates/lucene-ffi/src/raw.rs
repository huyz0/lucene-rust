//! Raw-pointer-to-Rust-value conversions shared by every exported function.
//! Centralized here so every `unsafe` pointer dereference in this crate's
//! public surface goes through exactly these two helpers (see the
//! `ffi-safety` skill: `unsafe` is scoped, not sprinkled).

use crate::error::{set_last_error, FfiStatus};
use std::ffi::c_char;

/// Reads `len` bytes at `ptr` as a UTF-8 `&str` borrowing from the caller's
/// buffer -- no copy, no ownership transfer. `ptr` may be null only when
/// `len == 0` (an empty string), matching a common C-ABI convention for
/// "empty and possibly not backed by a real allocation".
///
/// Takes `*const c_char` because every caller is converting a C string. The
/// widening to `*const u8` happens here, once, via `.cast()` rather than an
/// `as` expression: `c_char` is `i8` on x86_64 but `u8` on aarch64, so an
/// `as *const u8` at each call site is a real cast on one target and a no-op
/// the linter rejects on the other.
///
/// # Safety
/// `ptr` must be valid for reads of `len` bytes for the duration of the
/// borrow returned.
pub unsafe fn str_from_raw<'a>(ptr: *const c_char, len: usize) -> Result<&'a str, FfiStatus> {
    if ptr.is_null() {
        return if len == 0 {
            Ok("")
        } else {
            Err(FfiStatus::NullPointer)
        };
    }
    // SAFETY: caller contract guarantees `ptr` is valid for `len` bytes.
    let bytes = unsafe { std::slice::from_raw_parts(ptr.cast::<u8>(), len) };
    std::str::from_utf8(bytes).map_err(|_| FfiStatus::InvalidUtf8)
}

/// Reads `len` bytes at `ptr` as a byte slice borrowing from the caller's
/// buffer. Same null/zero-length convention as [`str_from_raw`].
///
/// # Safety
/// `ptr` must be valid for reads of `len` bytes for the duration of the
/// borrow returned.
pub unsafe fn bytes_from_raw<'a>(ptr: *const u8, len: usize) -> Result<&'a [u8], FfiStatus> {
    if ptr.is_null() {
        return if len == 0 {
            Ok(&[])
        } else {
            Err(FfiStatus::NullPointer)
        };
    }
    // SAFETY: caller contract guarantees `ptr` is valid for `len` bytes.
    Ok(unsafe { std::slice::from_raw_parts(ptr, len) })
}

/// Allocates an empty `Vec<T>` with room for `len` elements, reporting an
/// allocation failure as [`FfiStatus::InvalidArgument`] instead of aborting.
///
/// **Why this exists rather than `Vec::with_capacity`**: every count reaching
/// this crate is caller-supplied over a C ABI, where a JNI wrapper bug (a
/// negative `int` widened to `usize`, an uninitialised length) turns into an
/// absurd allocation request. `Vec::with_capacity` responds to a failed
/// allocation by calling `handle_alloc_error`, which **aborts the process** --
/// and an abort is *not* an unwind, so [`crate::error::guard`]'s
/// `catch_unwind` cannot contain it. A single bad length on the Java side
/// would take the whole OpenSearch node down rather than returning a status
/// code. `try_reserve_exact` returns the failure instead, keeping this
/// boundary's "every failure is a status code" contract intact.
pub fn try_with_capacity<T>(len: usize) -> Result<Vec<T>, FfiStatus> {
    let mut out = Vec::new();
    out.try_reserve_exact(len).map_err(|_| {
        set_last_error(format!(
            "cannot allocate {len} x {} bytes for a caller-supplied length",
            std::mem::size_of::<T>()
        ));
        FfiStatus::InvalidArgument
    })?;
    Ok(out)
}

/// Copies `src` into a freshly allocated `Vec<T>` via [`try_with_capacity`],
/// so an allocation failure is a status code rather than a process abort --
/// the `to_vec()` every `*_slice_from_raw` helper in this crate would
/// otherwise call.
pub fn try_to_vec<T: Copy>(src: &[T]) -> Result<Vec<T>, FfiStatus> {
    let mut out = try_with_capacity(src.len())?;
    out.extend_from_slice(src);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn try_with_capacity_allocates_a_reasonable_length() {
        let v = try_with_capacity::<u64>(8).unwrap();
        assert!(v.is_empty());
        assert!(v.capacity() >= 8);
    }

    #[test]
    fn try_with_capacity_rejects_an_absurd_length_instead_of_aborting() {
        // The exact shape a JNI caller's negative `int` takes once widened to
        // `usize`: far larger than any allocator can serve. `Vec::with_capacity`
        // would abort the process here; this must be a status code.
        let got = try_with_capacity::<u64>(usize::MAX / 4);
        assert_eq!(got.err(), Some(FfiStatus::InvalidArgument));
    }

    #[test]
    fn try_to_vec_copies_the_slice() {
        assert_eq!(try_to_vec(&[1i32, 2, 3]).unwrap(), vec![1, 2, 3]);
        assert!(try_to_vec::<i32>(&[]).unwrap().is_empty());
    }

    #[test]
    fn str_from_raw_reads_valid_utf8() {
        let s = "hello";
        let got = unsafe { str_from_raw(s.as_ptr().cast::<c_char>(), s.len()) }.unwrap();
        assert_eq!(got, "hello");
    }

    #[test]
    fn str_from_raw_rejects_invalid_utf8() {
        let bytes = [0xFFu8, 0xFE];
        let got = unsafe { str_from_raw(bytes.as_ptr().cast::<c_char>(), bytes.len()) };
        assert_eq!(got, Err(FfiStatus::InvalidUtf8));
    }

    #[test]
    fn str_from_raw_null_with_zero_len_is_empty_string() {
        let got = unsafe { str_from_raw(std::ptr::null(), 0) }.unwrap();
        assert_eq!(got, "");
    }

    #[test]
    fn str_from_raw_null_with_nonzero_len_is_null_pointer_error() {
        let got = unsafe { str_from_raw(std::ptr::null(), 3) };
        assert_eq!(got, Err(FfiStatus::NullPointer));
    }

    #[test]
    fn bytes_from_raw_reads_bytes() {
        let b = [1u8, 2, 3];
        let got = unsafe { bytes_from_raw(b.as_ptr(), b.len()) }.unwrap();
        assert_eq!(got, &[1, 2, 3]);
    }

    #[test]
    fn bytes_from_raw_null_with_zero_len_is_empty_slice() {
        let got = unsafe { bytes_from_raw(std::ptr::null(), 0) }.unwrap();
        assert!(got.is_empty());
    }

    #[test]
    fn bytes_from_raw_null_with_nonzero_len_is_null_pointer_error() {
        let got = unsafe { bytes_from_raw(std::ptr::null(), 5) };
        assert_eq!(got, Err(FfiStatus::NullPointer));
    }
}

/// Tests that the whole crate really is free of the "allocate a
/// caller-supplied length before validating it" abort hazard -- a
/// source-scanning invariant test, since the failure mode
/// (`handle_alloc_error` aborting the JVM) cannot be reproduced from inside
/// a test.
///
/// **Why not a `clippy.toml` `disallowed-methods` entry**, which would be a
/// real Tier-1 gate: `clippy.toml` is per-*workspace*, not per-crate, and
/// `Vec::with_capacity` is legitimate everywhere else in this workspace (a
/// decoder sizing a buffer from an already-validated header count is fine).
/// Banning it workspace-wide would produce dozens of false positives in
/// crates that never see a caller-supplied length. Scoped lint config would
/// be the right tool the day cargo grows it.
#[cfg(test)]
mod allocation_hazard_tests {
    /// The lines of `source` that are *not* inside a `#[cfg(test)]` item.
    ///
    /// Brace-depth tracked rather than "everything before the first
    /// `#[cfg(test)]`": this crate has several files with a test module
    /// followed by *more* items (`segment.rs`'s `live_docs_tests`,
    /// `highlighter.rs`'s `formatter_knob_tests`,
    /// `results_fragments.rs`'s `span_tests` all come after an existing
    /// `mod tests`), so a first-occurrence split would silently exempt
    /// production code appended after them.
    fn production_lines(source: &str) -> Vec<(usize, &str)> {
        let mut out = Vec::new();
        let mut skip_until_depth: Option<i32> = None;
        let mut pending_cfg_test = false;
        let mut depth = 0i32;
        for (n, line) in source.lines().enumerate() {
            let opens = line.matches('{').count() as i32;
            let closes = line.matches('}').count() as i32;
            if let Some(target) = skip_until_depth {
                depth += opens - closes;
                if depth <= target {
                    skip_until_depth = None;
                }
                continue;
            }
            if line.trim() == "#[cfg(test)]" {
                pending_cfg_test = true;
                depth += opens - closes;
                continue;
            }
            if pending_cfg_test {
                // The attributed item starts here; skip until its braces
                // close back to the depth we were at before it opened.
                pending_cfg_test = false;
                if opens > 0 {
                    skip_until_depth = Some(depth);
                    depth += opens - closes;
                    if depth <= skip_until_depth.unwrap() {
                        skip_until_depth = None;
                    }
                    continue;
                }
                // An attributed item with no brace on this line (e.g. a
                // `#[cfg(test)] use ...;`) -- skip just this line.
                depth += opens - closes;
                continue;
            }
            out.push((n + 1, line));
            depth += opens - closes;
        }
        out
    }

    /// No production code in this crate may allocate a caller-supplied
    /// length through a method that aborts on failure: every length reaching
    /// this crate comes from a JNI caller, and `Vec`/`String::with_capacity`
    /// and `vec![x; n]` all call `handle_alloc_error`, which **aborts** --
    /// an abort is not an unwind, so `catch_unwind` cannot contain it and a
    /// single bad length takes the JVM down. [`super::try_with_capacity`]/
    /// [`super::try_to_vec`] are the replacements. Test code is exempt (its
    /// lengths are literals).
    #[test]
    fn no_production_call_site_allocates_a_length_that_can_abort() {
        // Only the *repeat* form of `vec!` (`vec![x; n]`) is length-driven;
        // `vec![a, b, c]` is a literal and cannot be caller-sized. Detected
        // structurally below rather than by substring, since `vec![` alone
        // matches both.
        // Substrings, not full paths: `::with_capacity(` catches the
        // turbofished `Vec::<T>::with_capacity` form and every other
        // capacity-preallocating constructor (`HashMap`, `HashSet`,
        // `BytesMut`, ...) that the two-name list used to miss. Everything
        // matched here is length-driven and aborts on failure; the
        // `alloc-ok:` opt-out below is how a provably Rust-side length is
        // allowed through.
        const BANNED_METHODS: [&str; 1] = ["::with_capacity("];
        // An `// alloc-ok: <reason>` comment on the same line opts a site out
        // -- for a length that provably comes from Rust-side state rather
        // than from the caller (e.g. `segments.len()` off an opened reader).
        // The reason is mandatory by convention: a bare marker with no
        // explanation is exactly the thing this test exists to prevent
        // someone from adding.
        const OPT_OUT: &str = "alloc-ok:";
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut offenders = Vec::new();
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().and_then(|e| e.to_str()) != Some("rs") {
                continue;
            }
            let source = std::fs::read_to_string(&path).unwrap();
            for (n, line) in production_lines(&source) {
                let code = line.trim_start();
                if code.starts_with("//") || code.starts_with("///") {
                    continue;
                }
                if code.contains(OPT_OUT) {
                    continue;
                }
                for banned in BANNED_METHODS {
                    if code.contains(banned) {
                        offenders.push(format!("{}:{n}: {banned}", path.display()));
                    }
                }
                if let Some(rest) = code.split_once("vec![").map(|(_, r)| r) {
                    // `vec![x; n]` -- a `;` before the closing bracket.
                    let end = rest.find(']').unwrap_or(rest.len());
                    if rest[..end].contains(';') {
                        offenders.push(format!("{}:{n}: vec![x; n]", path.display()));
                    }
                }
            }
        }
        assert!(
            offenders.is_empty(),
            "use `raw::try_with_capacity`/`raw::try_to_vec` instead -- these abort the \
             process on a bad caller-supplied length: {offenders:?}"
        );
    }

    /// The scanner itself must not be fooled by a file whose production code
    /// comes *after* a test module -- the shape this crate actually has, and
    /// the shape a first-`#[cfg(test)]`-wins split gets wrong.
    #[test]
    fn production_lines_skips_test_modules_wherever_they_sit() {
        let source = "fn a() {\n    let x = 1;\n}\n\n#[cfg(test)]\nmod tests {\n    fn t() {\n                              let v = Vec::with_capacity(4);\n    }\n}\n\nfn b() {\n    let y = 2;\n}\n";
        let kept: Vec<&str> = production_lines(source)
            .into_iter()
            .map(|(_, l)| l)
            .collect();
        assert!(kept.iter().any(|l| l.contains("let x = 1")));
        assert!(
            kept.iter().any(|l| l.contains("let y = 2")),
            "production code after a test module must still be scanned"
        );
        assert!(
            !kept.iter().any(|l| l.contains("Vec::with_capacity")),
            "test-module code must be skipped"
        );
    }

    /// The banned-substring list itself, checked against the forms it has to
    /// catch. `Vec::with_capacity` was the only spelling the original list
    /// named, which silently exempted the turbofished form and every other
    /// capacity-preallocating constructor -- all of which abort on a bad
    /// caller-supplied length exactly as `Vec::with_capacity` does.
    #[test]
    fn the_banned_substring_catches_every_spelling_that_can_abort() {
        const BANNED: &str = "::with_capacity(";
        for spelled in [
            "let v = Vec::with_capacity(n);",
            "let v = Vec::<u8>::with_capacity(n);",
            "let v: Vec<u8> = Vec::with_capacity(n);",
            "let s = String::with_capacity(n);",
            "let m = HashMap::with_capacity(n);",
            "let m = std::collections::HashSet::with_capacity(n);",
        ] {
            assert!(spelled.contains(BANNED), "not caught: {spelled}");
        }
        // The fallible replacements this crate actually uses are not caught
        // (they are the fix, not the hazard), and neither is a `vec![a, b]`
        // literal -- only the length-driven `vec![x; n]` repeat form is, and
        // that is detected structurally rather than by substring.
        for allowed in [
            "let v = try_with_capacity(n)?;",
            "let v = crate::raw::try_with_capacity(n)?;",
            "let v = vec![1, 2, 3];",
        ] {
            assert!(!allowed.contains(BANNED), "false positive: {allowed}");
        }
    }
}
