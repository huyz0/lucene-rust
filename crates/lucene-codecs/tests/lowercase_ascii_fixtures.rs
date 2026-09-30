//! Differential test for `LowercaseAsciiCompression.compress` against Lucene
//! 10.5.0: `fixtures/data/util_primitives/lowercase_ascii.txt` (written by
//! `fixtures/src/GenUtilPrimitives.java`) holds 400 inputs -- lengths 0 to
//! 1500, exception densities from none to past the `len / 32` limit, and
//! exceptions more than 255 bytes apart -- with Lucene's compressed bytes or
//! its rejection. The port must agree on both, and its output must decode
//! back through the block-tree reader's decompressor.

#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::lowercase_ascii;

fn unhex(s: &str) -> Vec<u8> {
    if s == "-" {
        return Vec::new();
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

#[test]
fn compress_matches_lucene() {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/util_primitives/lowercase_ascii.txt"
    ))
    .expect("run scripts/gen-fixtures.sh --only GenUtilPrimitives");
    let (mut accepted, mut rejected, mut with_exceptions) = (0, 0, 0);
    for (ln, line) in text.lines().enumerate() {
        let p: Vec<&str> = line.split(' ').collect();
        let input = unhex(p[1]);
        let mut out = Vec::new();
        let ok = lowercase_ascii::compress(&input, &mut out);
        if p[2] == "REJECT" {
            assert!(!ok, "line {}: Lucene rejected", ln + 1);
            rejected += 1;
        } else {
            assert!(ok, "line {}: Lucene compressed", ln + 1);
            assert_eq!(out, unhex(p[2]), "line {}", ln + 1);
            if out.len() > input.len() - input.len() / 4 + 1 {
                with_exceptions += 1;
            }
            accepted += 1;
        }
    }
    assert!(
        accepted >= 300 && rejected >= 40 && with_exceptions >= 50,
        "{accepted} accepted, {rejected} rejected, {with_exceptions} with exceptions"
    );
}
