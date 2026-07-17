//! Differential test against a real `.tim`/`.tip`/`.tmd` triple engineered to
//! force real `Lucene103BlockTreeTermsWriter` into a genuinely **4+-level
//! deep** blocktree: a chain of two or more nested non-leaf `.tim` blocks
//! between the root and the leaves, not just the single non-leaf layer
//! `blocktree_multilevel_fixture.rs`'s 8000-term/default-block-size fixture
//! produces. See `crates/lucene-codecs/src/blocktree.rs`'s
//! `deep_nesting_fixture_reaches_at_least_four_levels` unit test (in
//! `blocktree.rs` itself, since it needs private trie/block-walking internals
//! to verify the nesting depth structurally) for the proof that a 4+-deep
//! chain was actually reached, not just that lookups happen to still work --
//! the tests in *this* file are the behavioral half of that split (every
//! term still findable via the public API, matching real Lucene's own ground
//! truth), same convention as every other real-bytes fixture test in this
//! crate.
//!
//! One field, "many" (`IndexOptions.DOCS`), 2000 distinct pseudo-random
//! strings over the narrow `{a,b}` alphabet (16 bytes each,
//! `java.util.Random(12345)`, fully deterministic), written with
//! `Lucene104PostingsFormat`'s `minItemsInBlock=2`/`maxItemsInBlock=4`
//! (rather than the format's 25/48 defaults). Regenerate with
//! `fixtures/src/GenBlockTreeDeepNesting.java`; see that file's module doc
//! for why a narrow alphabet plus small block-size thresholds is what
//! actually forces deep chained non-leaf nesting, where a wide alphabet at
//! any term count plateaus at a single non-leaf layer.

use lucene_codecs::blocktree;
use lucene_codecs::field_infos;

fn dir() -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/blocktree_deep_nesting_index/"
    )
    .to_string()
}

struct Manifest {
    kv: Vec<(String, String)>,
}

impl Manifest {
    fn load() -> Self {
        let text = std::fs::read_to_string(format!("{}manifest.properties", dir()))
            .expect("run fixtures generator first (GenBlockTreeDeepNesting)");
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
}

fn id_from_hex(hex: &str) -> [u8; 16] {
    let mut id = [0u8; 16];
    for (i, slot) in id.iter_mut().enumerate() {
        *slot = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap();
    }
    id
}

fn read_raw(name: &str) -> Vec<u8> {
    std::fs::read(format!("{}{}.raw", dir(), name)).unwrap_or_else(|_| panic!("missing {name}.raw"))
}

fn open_fixture() -> (blocktree::BlockTreeFields, Manifest) {
    let m = Manifest::load();
    let id = id_from_hex(m.get("id_hex"));
    let suffix = m.get("segment_suffix").to_string();
    let max_doc: i32 = m.get("max_doc").parse().unwrap();

    let fnm = read_raw(m.get("fnm_file_name"));
    let field_infos = field_infos::parse(&fnm, &id, "").expect("parse .fnm");

    let tim = read_raw(m.get("tim_file_name"));
    let tip = read_raw(m.get("tip_file_name"));
    let tmd = read_raw(m.get("tmd_file_name"));

    let fields = blocktree::open(&tim, &tip, &tmd, &field_infos, &id, &suffix, max_doc)
        .expect("open blocktree");
    (fields, m)
}

fn expected_terms(m: &Manifest) -> Vec<String> {
    let file_name = m.get("field.many.termsFile");
    let text = std::fs::read_to_string(format!("{}{}", dir(), file_name))
        .unwrap_or_else(|_| panic!("missing {file_name}"));
    text.lines().map(|l| l.to_string()).collect()
}

/// Every one of the 2000 terms is independently reachable via `seek_exact`,
/// each with the real `docFreq == 1`/`totalTermFreq == 1` (one document per
/// distinct token) -- proves that decoding through a 4+-level-deep chain of
/// nested non-leaf `.tim` blocks (see the module doc, and
/// `deep_nesting_fixture_reaches_at_least_four_levels` for the structural
/// proof this fixture actually reaches that depth) still recovers
/// byte-correct terms and stats, not just that `open()` doesn't error.
#[test]
fn deep_nesting_field_seek_exact_matches_real_lucene() {
    let (fields, m) = open_fixture();
    let many = fields.field("many").expect("expected field \"many\"");

    let num_terms: i64 = m.get("field.many.numTerms").parse().unwrap();
    assert_eq!(num_terms, 2000);
    assert_eq!(many.num_terms, num_terms);

    for term in expected_terms(&m) {
        let stats = many
            .seek_exact(term.as_bytes())
            .unwrap_or_else(|| panic!("expected term {term:?} to be found"));
        assert_eq!(stats.doc_freq, 1, "term={term:?}");
        assert_eq!(stats.total_term_freq, 1, "term={term:?}");
    }

    assert!(many.seek_exact(b"zzzzzzzzzzzzzzzz").is_none());
    assert!(many.seek_exact(b"").is_none());
}

/// Ordered enumeration (`TermsEnum::next()`) across every level of this
/// deeply-nested field's block chain must reproduce real Lucene's exact
/// sorted term list -- a stronger check than sampled `seek_exact` calls,
/// since an off-by-one in how a sub-block's own key bytes get re-prefixed
/// across several levels of recursion (`decode_block`'s doc comment) would
/// show up as a missing, duplicated, or misordered term somewhere in this
/// 2000-term walk, not necessarily one of the specific terms a smaller
/// spot-check might sample.
#[test]
fn deep_nesting_field_enumeration_matches_real_lucene_terms_enum_next() {
    let (fields, m) = open_fixture();
    let many = fields.field("many").expect("expected field \"many\"");

    let expected = expected_terms(&m);
    let mut got = Vec::with_capacity(expected.len());
    let mut it = many.iter();
    while let Some((term, stats)) = it.next() {
        assert_eq!(stats.doc_freq, 1);
        assert_eq!(stats.total_term_freq, 1);
        got.push(String::from_utf8(term.to_vec()).unwrap());
    }
    assert_eq!(got, expected);
}
