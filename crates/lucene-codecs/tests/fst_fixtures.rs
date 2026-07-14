//! Differential test against a real `FST<BytesRef>` built via real Lucene's
//! `FSTCompiler` (`ByteSequenceOutputs`, `allowFixedLengthArcs(false)` so
//! only list-encoded nodes are emitted -- the only encoding this port's
//! reader supports so far) and saved with `FST.save(Path)`. Regenerate with
//! fixtures/src/GenFst.java.

use lucene_codecs::fst::Fst;
use lucene_store::data_input::SliceInput;

fn dir() -> String {
    concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures/data/fst/").to_string()
}

struct Manifest {
    kv: Vec<(String, String)>,
}

impl Manifest {
    fn load() -> Self {
        let text = std::fs::read_to_string(format!("{}manifest.properties", dir()))
            .expect("run fixtures generator first (GenFst)");
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
    let buf =
        std::fs::read(format!("{}fst.bin", dir())).expect("run fixtures generator first (GenFst)");
    let mut input = SliceInput::new(&buf);
    Fst::read(&mut input).expect("decode real Lucene-written FST")
}

#[test]
fn present_keys_resolve_to_expected_outputs() {
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
        assert_eq!(
            got,
            Some(expected_output),
            "key={:?} ({})",
            String::from_utf8_lossy(&key),
            i
        );
    }
}

#[test]
fn absent_keys_are_not_found() {
    let fst = load_fst();
    let manifest = Manifest::load();
    let n = manifest.count("num_absent");
    assert!(n > 0);
    for i in 0..n {
        let key = from_hex(manifest.get(&format!("absent.{i}.key_hex")));
        let got = fst
            .get(&key)
            .unwrap_or_else(|e| panic!("get({key:?}) errored: {e}"));
        assert_eq!(
            got,
            None,
            "key={:?} ({}) should be absent",
            String::from_utf8_lossy(&key),
            i
        );
    }
}

#[test]
fn metadata_matches_expected_shape() {
    let fst = load_fst();
    let meta = fst.metadata();
    assert_eq!(meta.input_type, lucene_codecs::fst::InputType::Byte1);
    assert!(meta.empty_output.is_none());
    assert!(meta.num_bytes > 0);
}

/// `Fst::read_borrowed` (zero-copy body) must resolve every present/absent
/// key from the real fixture identically to `Fst::read` (owned-copy body)
/// over the exact same bytes -- the whole point of adding a borrowing
/// constructor is that it's a drop-in alternative for lookup, not a
/// different code path with different semantics.
#[test]
fn read_borrowed_matches_read_on_real_fixture() {
    let buf =
        std::fs::read(format!("{}fst.bin", dir())).expect("run fixtures generator first (GenFst)");

    let mut owned_input = SliceInput::new(&buf);
    let owned = Fst::read(&mut owned_input).expect("owned decode");

    let mut borrowed_input = SliceInput::new(&buf);
    let borrowed = Fst::read_borrowed(&mut borrowed_input).expect("borrowed decode");
    assert!(borrowed.is_borrowed());
    assert!(!owned.is_borrowed());

    let manifest = Manifest::load();
    for i in 0..manifest.count("num_present") {
        let key = from_hex(manifest.get(&format!("present.{i}.key_hex")));
        assert_eq!(owned.get(&key).unwrap(), borrowed.get(&key).unwrap());
    }
    for i in 0..manifest.count("num_absent") {
        let key = from_hex(manifest.get(&format!("absent.{i}.key_hex")));
        assert_eq!(owned.get(&key).unwrap(), borrowed.get(&key).unwrap());
        assert_eq!(borrowed.get(&key).unwrap(), None);
    }

    assert_eq!(owned.metadata().num_bytes, borrowed.metadata().num_bytes);
}
