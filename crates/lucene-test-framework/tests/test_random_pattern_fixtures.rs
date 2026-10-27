//! Differential test against `fixtures/src/testframework/GenTestFrameworkPattern.java`:
//! for 500 seeds, [`random_pattern`] settles on the pattern Lucene 10.5.0's
//! `TestUtil.randomPattern` does, after the same retries (the random's next
//! `nextInt()` agrees too).
//!
//! Regenerate with `scripts/gen-test-framework-fixtures.sh --only TestFrameworkPattern`.

use std::fs;
use std::path::PathBuf;

use lucene_test_framework::util::test_util::{random_pattern, random_regexpish_string};
use lucene_util::java_random::JavaRandom;

fn units_hex(s: &str) -> String {
    if s.is_empty() {
        return "-".into();
    }
    s.encode_utf16().map(|u| format!("{u:04x}")).collect()
}

#[test]
fn random_pattern_matches_lucene() {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/testframework/data/pattern.txt");
    let fixture = fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    let mut seeds = 0;
    let mut retried = 0;
    for line in fixture.lines() {
        let mut parts = line.split(' ');
        let seed: i64 = parts.next().unwrap().parse().unwrap();
        let pattern = parts.next().unwrap();
        let next: i32 = parts.next().unwrap().parse().unwrap();
        if random_regexpish_string(&mut JavaRandom::new(seed)) != source_of(pattern) {
            retried += 1;
        }
        let mut r = JavaRandom::new(seed);
        let (source, _compiled) = random_pattern(&mut r);
        assert_eq!(units_hex(&source), pattern, "seed {seed}: {source:?}");
        assert_eq!(r.next_int(), next, "seed {seed}: draws after the pattern");
        seeds += 1;
    }
    assert_eq!(seeds, 500);
    // The seeds reach the retry: a first draw that does not compile or
    // that splits the surrogate pair.
    assert!(retried > 10, "only {retried} seeds retried");
}

/// The pattern a fixture line's hex units spell.
fn source_of(hex: &str) -> String {
    if hex == "-" {
        return String::new();
    }
    let units: Vec<u16> = (0..hex.len())
        .step_by(4)
        .map(|i| u16::from_str_radix(&hex[i..i + 4], 16).unwrap())
        .collect();
    String::from_utf16(&units).unwrap()
}
