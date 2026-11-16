#![cfg(test)]
// Tests index and count freely: the gate is about values read off disk.
#![allow(clippy::arithmetic_side_effects)]
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
/// this crate comes from a Java caller, and `Vec`/`String::with_capacity`
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
    // `BytesMut`, ...) that a list of full names would miss. Everything
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
    // SPACES-OK: an indented Rust source, the test's input.
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
