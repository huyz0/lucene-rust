//! Differential test against real `FST<BytesRef>`s whose `FST.INPUT_TYPE` is
//! `BYTE2` and `BYTE4` (built via real Lucene's `FSTCompiler.Builder`'s
//! public `INPUT_TYPE` parameter and `FSTCompiler.add(IntsRef, T)`, not
//! `BytesRef`-derived labels), unlike every other `fst_*_fixtures.rs` test in
//! this crate, which exercises `BYTE1` (raw term bytes, used by the
//! BlockTree term index) fixtures. Regenerate with
//! `fixtures/src/GenFstWideInputTypes.java`.
//!
//! `Fst::get`/`Fst::iter` (the byte-keyed API) reject non-`BYTE1` FSTs
//! outright, since a `u8` key element can't represent a UTF-16 code unit
//! (`BYTE2`) or a full Unicode code point (`BYTE4`) in general -- see
//! `Fst::get`'s doc comment. This test instead exercises the label-domain
//! API (`Fst::get_labels`, `Fst::iter_labels`/`FstEnum::next_labels`) that
//! widens lookup/enumeration to any `INPUT_TYPE`.

use lucene_codecs::fst::{Fst, InputType};
use lucene_store::data_input::SliceInput;

fn dir(name: &str) -> String {
    format!("{}/../../fixtures/data/{name}/", env!("CARGO_MANIFEST_DIR"))
}

struct Manifest {
    kv: Vec<(String, String)>,
}

impl Manifest {
    fn load(name: &str) -> Self {
        let text = std::fs::read_to_string(format!("{}manifest.properties", dir(name)))
            .expect("run fixtures generator first (GenFstWideInputTypes)");
        let kv = text
            .lines()
            .filter_map(|l| l.split_once('='))
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        Manifest { kv }
    }

    fn get(&self, key: &str) -> &str {
        self.kv
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
            .unwrap_or_else(|| panic!("manifest key {key} missing"))
    }

    fn count(&self, prefix: &str) -> usize {
        self.get(prefix).parse().unwrap()
    }

    fn labels(&self, key: &str) -> Vec<i32> {
        let s = self.get(key);
        s.split(',').map(|n| n.parse().unwrap()).collect()
    }
}

fn from_hex(s: &str) -> Vec<u8> {
    if s.is_empty() {
        return Vec::new();
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

fn load_fst(name: &str) -> Fst<'static> {
    let buf = std::fs::read(format!("{}fst.bin", dir(name)))
        .expect("run fixtures generator first (GenFstWideInputTypes)");
    let mut input = SliceInput::new(&buf);
    Fst::read(&mut input).expect("decode real Lucene-written FST")
}

fn check_present_and_absent(name: &str, expected_input_type: InputType) {
    let fst = load_fst(name);
    assert_eq!(fst.metadata().input_type, expected_input_type);

    let manifest = Manifest::load(name);
    let n = manifest.count("num_present");
    assert!(n > 0);
    for i in 0..n {
        let key = manifest.labels(&format!("present.{i}.key"));
        let expected_output = from_hex(manifest.get(&format!("present.{i}.output_hex")));
        let got = fst
            .get_labels(&key)
            .unwrap_or_else(|e| panic!("get_labels({key:?}) errored: {e}"));
        assert_eq!(got, Some(expected_output), "key={key:?} ({i})");

        // `seek_exact_labels` is a thin wrapper around `get_labels` -- must
        // agree.
        assert_eq!(
            fst.seek_exact_labels(&key).unwrap(),
            fst.get_labels(&key).unwrap()
        );
    }

    let n = manifest.count("num_absent");
    assert!(n > 0);
    for i in 0..n {
        let key = manifest.labels(&format!("absent.{i}.key"));
        let got = fst
            .get_labels(&key)
            .unwrap_or_else(|e| panic!("get_labels({key:?}) errored: {e}"));
        assert_eq!(got, None, "key={key:?} ({i}) should be absent");
    }
}

#[test]
fn byte2_present_and_absent_keys() {
    check_present_and_absent("fst_byte2", InputType::Byte2);
}

#[test]
fn byte4_present_and_absent_keys() {
    check_present_and_absent("fst_byte4", InputType::Byte4);
}

/// `Fst::get`/`Fst::iter` (the byte-keyed API) must reject these FSTs
/// outright rather than silently misdecoding wide labels as bytes.
#[test]
fn byte_keyed_api_rejects_wide_input_types() {
    for name in ["fst_byte2", "fst_byte4"] {
        let fst = load_fst(name);
        assert!(fst.get(b"anything").is_err());
        assert!(fst.iter().is_err());
    }
}

fn check_iter_labels_enumerates_all_keys(name: &str) {
    let fst = load_fst(name);
    let manifest = Manifest::load(name);
    let mut expected: Vec<(Vec<i32>, Vec<u8>)> = (0..manifest.count("num_present"))
        .map(|i| {
            (
                manifest.labels(&format!("present.{i}.key")),
                from_hex(manifest.get(&format!("present.{i}.output_hex"))),
            )
        })
        .collect();
    expected.sort();

    let mut enumerator = fst.iter_labels();
    let mut got = Vec::new();
    while let Some(pair) = enumerator.next_labels() {
        got.push(pair.expect("enumeration should not error"));
    }

    assert_eq!(got, expected);
}

#[test]
fn byte2_iter_labels_enumerates_all_keys_in_ascending_order() {
    check_iter_labels_enumerates_all_keys("fst_byte2");
}

#[test]
fn byte4_iter_labels_enumerates_all_keys_in_ascending_order() {
    check_iter_labels_enumerates_all_keys("fst_byte4");
}

/// `FstEnum::seek_ceil_labels`/`seek_floor_labels`/`seek_exact_labels` must
/// agree with a full `iter_labels` walk on the BYTE4 fixture -- a rough
/// cross-check that seeking through wide labels lands on the same node the
/// full enumeration visits.
#[test]
fn byte4_seek_labels_matches_full_enumeration() {
    let fst = load_fst("fst_byte4");
    let manifest = Manifest::load("fst_byte4");
    let mut expected: Vec<(Vec<i32>, Vec<u8>)> = (0..manifest.count("num_present"))
        .map(|i| {
            (
                manifest.labels(&format!("present.{i}.key")),
                from_hex(manifest.get(&format!("present.{i}.output_hex"))),
            )
        })
        .collect();
    expected.sort();

    for (key, output) in &expected {
        let mut enumerator = fst.iter_labels();
        let found = enumerator
            .seek_exact_labels(key)
            .expect("seek_exact_labels should not error");
        assert_eq!(found, Some((key.clone(), output.clone())));

        let mut enumerator = fst.iter_labels();
        let ceil = enumerator
            .seek_ceil_labels(key)
            .expect("seek_ceil_labels should not error");
        assert_eq!(ceil, Some((key.clone(), output.clone())));

        let mut enumerator = fst.iter_labels();
        let floor = enumerator
            .seek_floor_labels(key)
            .expect("seek_floor_labels should not error");
        assert_eq!(floor, Some((key.clone(), output.clone())));
    }
}

/// Same cross-check as [`byte4_seek_labels_matches_full_enumeration`], on the
/// BYTE2 fixture -- seeking through UTF-16-range labels must agree with a
/// full `iter_labels` walk too, not just for BYTE4's wider codepoint range.
#[test]
fn byte2_seek_labels_matches_full_enumeration() {
    let fst = load_fst("fst_byte2");
    let manifest = Manifest::load("fst_byte2");
    let mut expected: Vec<(Vec<i32>, Vec<u8>)> = (0..manifest.count("num_present"))
        .map(|i| {
            (
                manifest.labels(&format!("present.{i}.key")),
                from_hex(manifest.get(&format!("present.{i}.output_hex"))),
            )
        })
        .collect();
    expected.sort();

    for (key, output) in &expected {
        let mut enumerator = fst.iter_labels();
        let found = enumerator
            .seek_exact_labels(key)
            .expect("seek_exact_labels should not error");
        assert_eq!(found, Some((key.clone(), output.clone())));

        let mut enumerator = fst.iter_labels();
        let ceil = enumerator
            .seek_ceil_labels(key)
            .expect("seek_ceil_labels should not error");
        assert_eq!(ceil, Some((key.clone(), output.clone())));

        let mut enumerator = fst.iter_labels();
        let floor = enumerator
            .seek_floor_labels(key)
            .expect("seek_floor_labels should not error");
        assert_eq!(floor, Some((key.clone(), output.clone())));
    }
}
