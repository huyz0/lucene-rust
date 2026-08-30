//! Differential test against a real `FST<BytesRef>` that **accepts the empty
//! string** with a non-empty output -- the one `FSTMetadata` shape no other
//! `GenFst*` fixture reaches. Regenerate with
//! `fixtures/src/GenFstEmptyKey.java`.
//!
//! `FSTMetadata.save` does not write the empty output verbatim: it runs the
//! value through `outputs.writeFinalOutput` (for `ByteSequenceOutputs`, a
//! `vint` length then the payload), reverses that whole buffer, and writes
//! `vint(len)` plus the reversed buffer, so the reader decodes it with the
//! same reverse `BytesReader` every arc output uses. Reversing and keeping
//! the buffer verbatim silently prepends a length byte to every empty
//! output -- a divergence only real Lucene-written bytes can expose, which
//! is why this fixture exists.
// Test-support code opts out of the arithmetic gate at the file boundary:
// the gate exists for values read off disk in production decode paths, not
// for a fixture builder's own index arithmetic. See
// `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::fst::{write_fst, Fst};
use lucene_store::data_input::SliceInput;

fn dir() -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/fst_empty_key/"
    )
    .to_string()
}

fn manifest() -> Vec<(String, String)> {
    let text = std::fs::read_to_string(format!("{}manifest.properties", dir()))
        .expect("run fixtures generator first (GenFstEmptyKey)");
    text.lines()
        .filter_map(|l| l.split_once('='))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

fn get<'a>(m: &'a [(String, String)], key: &str) -> &'a str {
    m.iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.as_str())
        .unwrap_or_else(|| panic!("manifest key {key} missing"))
}

fn from_hex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

fn fst_bytes() -> Vec<u8> {
    std::fs::read(format!("{}fst.bin", dir()))
        .expect("run fixtures generator first (GenFstEmptyKey)")
}

#[test]
fn empty_output_decodes_without_its_serialization_framing() {
    let buf = fst_bytes();
    let mut input = SliceInput::new(&buf);
    let fst = Fst::read(&mut input).unwrap();
    let m = manifest();
    let expected = from_hex(get(&m, "empty_output_hex"));
    assert!(!expected.is_empty(), "fixture must use a non-empty output");
    assert_eq!(
        fst.metadata().empty_output.as_deref(),
        Some(expected.as_slice()),
        "emptyOutput must be the decoded payload, not the reversed `writeFinalOutput` buffer"
    );
}

#[test]
fn every_present_key_including_the_empty_one_resolves() {
    let buf = fst_bytes();
    let mut input = SliceInput::new(&buf);
    let fst = Fst::read(&mut input).unwrap();
    let m = manifest();
    let n: usize = get(&m, "num_present").parse().unwrap();
    assert!(n > 1);
    let mut saw_empty_key = false;
    for i in 0..n {
        let key = from_hex(get(&m, &format!("present.{i}.key_hex")));
        let want = from_hex(get(&m, &format!("present.{i}.output_hex")));
        saw_empty_key |= key.is_empty();
        assert_eq!(fst.get(&key).unwrap().as_deref(), Some(want.as_slice()));
        assert_eq!(
            fst.seek_exact(&key).unwrap().as_deref(),
            Some(want.as_slice())
        );
    }
    assert!(saw_empty_key, "fixture must contain the empty-string key");
}

#[test]
fn absent_keys_are_rejected() {
    let buf = fst_bytes();
    let mut input = SliceInput::new(&buf);
    let fst = Fst::read(&mut input).unwrap();
    let m = manifest();
    let n: usize = get(&m, "num_absent").parse().unwrap();
    for i in 0..n {
        let key = from_hex(get(&m, &format!("absent.{i}.key_hex")));
        assert_eq!(fst.get(&key).unwrap(), None, "key {key:?} must be absent");
    }
}

#[test]
fn enumeration_starts_at_the_empty_key_with_its_own_output() {
    let buf = fst_bytes();
    let mut input = SliceInput::new(&buf);
    let fst = Fst::read(&mut input).unwrap();
    let m = manifest();
    let n: usize = get(&m, "num_present").parse().unwrap();
    let expected: Vec<(Vec<u8>, Vec<u8>)> = (0..n)
        .map(|i| {
            (
                from_hex(get(&m, &format!("present.{i}.key_hex"))),
                from_hex(get(&m, &format!("present.{i}.output_hex"))),
            )
        })
        .collect();
    let got: Vec<(Vec<u8>, Vec<u8>)> = fst.iter().unwrap().map(|r| r.unwrap()).collect();
    assert_eq!(got, expected);
}

/// `write_fst` must re-emit byte-for-byte what real Lucene wrote, metadata
/// framing included -- a self-round-trip through this port's own reader
/// cannot see a framing error that both sides make symmetrically.
#[test]
fn write_fst_reproduces_the_real_lucene_bytes_exactly() {
    let buf = fst_bytes();
    let mut input = SliceInput::new(&buf);
    let fst = Fst::read(&mut input).unwrap();
    assert_eq!(write_fst(&fst), buf);
}
