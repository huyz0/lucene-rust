//! Differential test against a real `FST<BytesRef>` (`fixtures/data/fst_deep_trie/`)
//! built via real Lucene's `FSTCompiler` (`ByteSequenceOutputs`,
//! `allowFixedLengthArcs(false)`, same scope as `fst_fixtures.rs`) whose 9
//! keys (`abcaa`, `abcab`, `abcz`, `abda`, `abdz`, `acaa`, `aczz`, `baaa`,
//! `bzzz`) share prefixes deeply enough that a single seek must descend
//! (and, for several targets, backtrack) across at least 3 distinct trie
//! levels -- confirmed by `GenFstDeepTrie.java`'s own manual arc-walk
//! self-check (depth 5 along `"abcaa"`) before the fixture was written,
//! the same way `GenFstBinarySearch.java` self-checks its `"(bs)"` node
//! encoding. Every prior `fst_*_fixtures.rs` test's interesting structure
//! lives at or one level below the root; this fixture closes that gap for
//! genuinely deep, multi-level seeking specifically.
//!
//! Ground truth for every `seekCeil`/`seekFloor`/`seekExact` result in
//! `manifest.properties` comes from real Lucene's own
//! `BytesRefFSTEnum.seekCeil`/`seekFloor`/`seekExact` against the reloaded
//! FST (see `GenFstDeepTrie.java`), not hand-derived -- so this test is
//! cross-checked against real Lucene's actual multi-level backtracking
//! behavior, not just self-consistent with this port's own code.
//!
//! Regenerate with `fixtures/src/GenFstDeepTrie.java`.

use lucene_codecs::fst::Fst;
use lucene_store::data_input::SliceInput;
use std::collections::HashMap;

fn dir() -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/fst_deep_trie/"
    )
    .to_string()
}

struct Manifest {
    kv: HashMap<String, String>,
}

impl Manifest {
    fn load() -> Self {
        let text = std::fs::read_to_string(format!("{}manifest.properties", dir()))
            .expect("run fixtures generator first (GenFstDeepTrie)");
        let kv = text
            .lines()
            .filter_map(|l| l.split_once('='))
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        Manifest { kv }
    }

    fn get(&self, key: &str) -> &str {
        self.kv
            .get(key)
            .unwrap_or_else(|| panic!("manifest key {key} missing"))
            .as_str()
    }

    fn count(&self, key: &str) -> usize {
        self.get(key).parse().unwrap()
    }

    /// Reads a `<prefix>.present`/`.key_hex`/`.output_hex` triple written by
    /// `appendResult` in `GenFstDeepTrie.java` as `Option<(key, output)>`.
    fn result(&self, prefix: &str) -> Option<(Vec<u8>, Vec<u8>)> {
        if self.get(&format!("{prefix}.present")) == "true" {
            Some((
                from_hex(self.get(&format!("{prefix}.key_hex"))),
                from_hex(self.get(&format!("{prefix}.output_hex"))),
            ))
        } else {
            None
        }
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
        .expect("run fixtures generator first (GenFstDeepTrie)");
    let mut input = SliceInput::new(&buf);
    Fst::read(&mut input).expect("decode real Lucene-written FST")
}

#[test]
fn get_resolves_every_present_key_at_every_depth() {
    let fst = load_fst();
    let m = Manifest::load();
    for i in 0..m.count("num_present") {
        let key = from_hex(m.get(&format!("present.{i}.key_hex")));
        let want = from_hex(m.get(&format!("present.{i}.output_hex")));
        assert_eq!(
            fst.get(&key).unwrap(),
            Some(want),
            "key={key:?} ({})",
            String::from_utf8_lossy(&key)
        );
    }
}

#[test]
fn seek_ceil_matches_real_lucene_across_every_backtrack_target() {
    let fst = load_fst();
    let m = Manifest::load();
    let mut e = fst.iter().unwrap();
    for i in 0..m.count("num_targets") {
        let key = from_hex(m.get(&format!("target.{i}.key_hex")));
        let want = m.result(&format!("target.{i}.ceil"));
        assert_eq!(
            e.seek_ceil(&key).unwrap(),
            want,
            "seek_ceil target={key:?} ({})",
            String::from_utf8_lossy(&key)
        );
    }
}

#[test]
fn seek_floor_matches_real_lucene_across_every_backtrack_target() {
    let fst = load_fst();
    let m = Manifest::load();
    let mut e = fst.iter().unwrap();
    for i in 0..m.count("num_targets") {
        let key = from_hex(m.get(&format!("target.{i}.key_hex")));
        let want = m.result(&format!("target.{i}.floor"));
        assert_eq!(
            e.seek_floor(&key).unwrap(),
            want,
            "seek_floor target={key:?} ({})",
            String::from_utf8_lossy(&key)
        );
    }
}

#[test]
fn seek_exact_matches_real_lucene_across_every_backtrack_target() {
    let fst = load_fst();
    let m = Manifest::load();
    let mut e = fst.iter().unwrap();
    for i in 0..m.count("num_targets") {
        let key = from_hex(m.get(&format!("target.{i}.key_hex")));
        let want = m.result(&format!("target.{i}.exact"));
        assert_eq!(
            e.seek_exact(&key).unwrap(),
            want,
            "seek_exact target={key:?} ({})",
            String::from_utf8_lossy(&key)
        );

        // `Fst::seek_exact` (non-enum, direct API) must agree too.
        let want_output = want.map(|(_, output)| output);
        assert_eq!(
            fst.seek_exact(&key).unwrap(),
            want_output,
            "Fst::seek_exact target={key:?}"
        );
    }
}

#[test]
fn full_ascending_enumeration_yields_every_present_key_in_sorted_order() {
    let fst = load_fst();
    let m = Manifest::load();
    let mut expected: Vec<(Vec<u8>, Vec<u8>)> = (0..m.count("num_present"))
        .map(|i| {
            (
                from_hex(m.get(&format!("present.{i}.key_hex"))),
                from_hex(m.get(&format!("present.{i}.output_hex"))),
            )
        })
        .collect();
    expected.sort();

    let e = fst.iter().unwrap();
    let mut got = Vec::new();
    for kv in e {
        got.push(kv.unwrap());
    }
    assert_eq!(got, expected);
}
