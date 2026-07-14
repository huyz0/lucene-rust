//! Differential test against a real `FST<BytesRef>` built via real Lucene's
//! `FSTCompiler` with `allowFixedLengthArcs(true)` and a sparse enough set of
//! single-byte root labels that the compiler actually expands the root node
//! into an `ARCS_FOR_BINARY_SEARCH` fixed-length-arc node (confirmed via
//! `FST.Arc#toString()` printing `"(bs)"` -- see
//! `fixtures/src/GenFstBinarySearch.java`'s self-check, which fails the
//! generator outright if the compiler picks a different encoding). This is
//! the proof that `Fst::find_target_arc`'s `ARCS_FOR_BINARY_SEARCH` decode
//! path (added alongside the pre-existing list-encoding support) works
//! against genuine Lucene-written bytes, not just the hand-built node in
//! `fst.rs`'s own unit tests.
//!
//! Regenerate with `fixtures/src/GenFstBinarySearch.java` (see
//! `fixtures/README.md`).

use lucene_codecs::fst::Fst;
use lucene_store::data_input::SliceInput;

fn dir() -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/fst_binary_search/"
    )
    .to_string()
}

struct Manifest {
    kv: Vec<(String, String)>,
}

impl Manifest {
    fn load() -> Self {
        let text = std::fs::read_to_string(format!("{}manifest.properties", dir()))
            .expect("run fixtures generator first (GenFstBinarySearch)");
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

fn load_fst() -> Fst<'static> {
    let buf = std::fs::read(format!("{}fst.bin", dir()))
        .expect("run fixtures generator first (GenFstBinarySearch)");
    let mut input = SliceInput::new(&buf);
    Fst::read(&mut input).expect("decode real Lucene-written ARCS_FOR_BINARY_SEARCH FST")
}

#[test]
fn present_keys_resolve_to_expected_outputs_through_binary_search_node() {
    let fst = load_fst();
    let manifest = Manifest::load();
    let n = manifest.count("num_present");
    assert!(n > 0);
    for i in 0..n {
        let key = from_hex(manifest.get(&format!("present.{i}.key_hex")));
        let expected_output = from_hex(manifest.get(&format!("present.{i}.output_hex")));
        let got = fst
            .get(&key)
            .unwrap_or_else(|e| panic!("get({key:?}) errored: {e}"));
        assert_eq!(got, Some(expected_output), "key={key:?} ({i})");
    }
}

#[test]
fn absent_keys_in_and_around_the_label_range_are_not_found() {
    let fst = load_fst();
    let manifest = Manifest::load();
    let n = manifest.count("num_absent");
    assert!(n > 0);
    for i in 0..n {
        let key = from_hex(manifest.get(&format!("absent.{i}.key_hex")));
        let got = fst
            .get(&key)
            .unwrap_or_else(|e| panic!("get({key:?}) errored: {e}"));
        assert_eq!(got, None, "key={key:?} ({i}) should be absent");
    }
}

/// Every key in this fixture is exactly one byte, so a lookup only ever
/// touches the root's `ARCS_FOR_BINARY_SEARCH` node itself -- confirms this
/// isn't accidentally passing because of some other, unrelated node in the
/// FST.
#[test]
fn all_manifest_keys_are_single_byte() {
    let manifest = Manifest::load();
    for i in 0..manifest.count("num_present") {
        let key = from_hex(manifest.get(&format!("present.{i}.key_hex")));
        assert_eq!(key.len(), 1);
    }
}

#[test]
fn read_borrowed_matches_read_on_binary_search_fixture() {
    let buf = std::fs::read(format!("{}fst.bin", dir()))
        .expect("run fixtures generator first (GenFstBinarySearch)");

    let mut owned_input = SliceInput::new(&buf);
    let owned = Fst::read(&mut owned_input).expect("owned decode");

    let mut borrowed_input = SliceInput::new(&buf);
    let borrowed = Fst::read_borrowed(&mut borrowed_input).expect("borrowed decode");
    assert!(borrowed.is_borrowed());

    let manifest = Manifest::load();
    for i in 0..manifest.count("num_present") {
        let key = from_hex(manifest.get(&format!("present.{i}.key_hex")));
        assert_eq!(owned.get(&key).unwrap(), borrowed.get(&key).unwrap());
        assert!(owned.get(&key).unwrap().is_some());
    }
    for i in 0..manifest.count("num_absent") {
        let key = from_hex(manifest.get(&format!("absent.{i}.key_hex")));
        assert_eq!(owned.get(&key).unwrap(), None);
        assert_eq!(borrowed.get(&key).unwrap(), None);
    }
}
