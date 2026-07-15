//! Differential test against a real `FST<BytesRef>` built via real Lucene's
//! `FSTCompiler` with `allowFixedLengthArcs(true)` and a small, fully
//! contiguous single-byte label set that the compiler actually expands into
//! an `ARCS_FOR_CONTINUOUS` fixed-length-arc node (confirmed via
//! `FST.Arc#toString()` printing `"(cs)"` -- see
//! `fixtures/src/GenFstContinuous.java`'s self-check, which fails the
//! generator outright if the compiler picks a different encoding). This is
//! the proof that `Fst::find_target_arc_continuous`'s `ARCS_FOR_CONTINUOUS`
//! decode path works against genuine Lucene-written bytes, not just the
//! hand-built node in `fst.rs`'s own unit tests.
//!
//! Regenerate with `fixtures/src/GenFstContinuous.java` (see
//! `fixtures/README.md`).

use lucene_codecs::fst::Fst;
use lucene_store::data_input::SliceInput;

fn dir() -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/fst_continuous/"
    )
    .to_string()
}

struct Manifest {
    kv: Vec<(String, String)>,
}

impl Manifest {
    fn load() -> Self {
        let text = std::fs::read_to_string(format!("{}manifest.properties", dir()))
            .expect("run fixtures generator first (GenFstContinuous)");
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
        .expect("run fixtures generator first (GenFstContinuous)");
    let mut input = SliceInput::new(&buf);
    Fst::read(&mut input).expect("decode real Lucene-written ARCS_FOR_CONTINUOUS FST")
}

#[test]
fn present_keys_resolve_to_expected_outputs_through_continuous_node() {
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

/// Every absent key here is strictly outside the contiguous label range
/// (there is no in-range gap to test -- `ARCS_FOR_CONTINUOUS`'s whole point
/// is that every label in its range is present), so this exercises
/// `find_target_arc_continuous`'s before/after-range bounds check.
#[test]
fn absent_keys_outside_the_label_range_are_not_found() {
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
/// touches the root's `ARCS_FOR_CONTINUOUS` node itself -- confirms this
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
fn read_borrowed_matches_read_on_continuous_fixture() {
    let buf = std::fs::read(format!("{}fst.bin", dir()))
        .expect("run fixtures generator first (GenFstContinuous)");

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

/// `Fst::iter` over a real Lucene-written `ARCS_FOR_CONTINUOUS` root node
/// (not just the hand-built one in `fst.rs`'s own unit tests): proves
/// `read_first_real_target_arc`/`read_next_real_arc`'s continuous branch
/// (shared with `ARCS_FOR_BINARY_SEARCH`'s incremental-slot-advance
/// arithmetic) enumerates every arc of a genuine continuous node in the
/// correct (ascending label) order, not just `find_target_arc`'s one-shot
/// lookup path.
#[test]
fn iter_enumerates_every_key_of_a_continuous_root_node() {
    let fst = load_fst();
    let manifest = Manifest::load();
    let mut expected: Vec<(Vec<u8>, Vec<u8>)> = (0..manifest.count("num_present"))
        .map(|i| {
            (
                from_hex(manifest.get(&format!("present.{i}.key_hex"))),
                from_hex(manifest.get(&format!("present.{i}.output_hex"))),
            )
        })
        .collect();
    expected.sort();

    let got: Vec<(Vec<u8>, Vec<u8>)> = fst
        .iter()
        .expect("iter should support this BYTE1 fixture")
        .collect::<Result<_, _>>()
        .expect("enumeration should not error");

    assert_eq!(got, expected);
}
