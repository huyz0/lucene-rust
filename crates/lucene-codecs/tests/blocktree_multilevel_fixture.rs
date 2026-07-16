//! Differential test against a real `.tim`/`.tip`/`.tmd` triple large/varied
//! enough to force real `Lucene103BlockTreeTermsWriter` into a genuine
//! **multi-level blocktree**: at least one physical `.tim` block that is
//! itself non-leaf (some of its entries are pointers to further-nested
//! sub-blocks, addressed by an in-block delta-fp, rather than raw term
//! suffixes) -- distinct from both the `.tip` trie's own multi-level node
//! nesting and from floor sub-blocks, both already exercised by
//! `blocktree_fixtures.rs`'s "many" field (400 sequential-numeric-suffix
//! terms). See `crates/lucene-codecs/src/blocktree.rs`'s module doc for the
//! full three-way distinction, and this crate's
//! `multilevel_fixture_reaches_a_genuine_non_leaf_block` unit test (in
//! `blocktree.rs` itself, since it needs private trie-walking internals to
//! verify structurally) for the proof that a non-leaf block was actually
//! reached, not just that lookups happen to still work. **Caveat, stated
//! honestly**: this fixture's one non-leaf block's sub-block pointer happens
//! to also be independently reachable via the `.tip` trie, so it gets
//! skipped by `decode_block`'s dedup check rather than actually recursed
//! into here -- the tests in *this* file prove the non-leaf *shape* decodes
//! without error and that every term still resolves correctly overall, but
//! the recursive re-prefixing code path itself is exercised only by
//! `blocktree.rs`'s hand-built `decode_block_recurses_into_sub_block` unit
//! test, not by this real-Lucene differential.
//!
//! One field, "many" (`IndexOptions.DOCS`), 8000 distinct pseudo-random
//! lowercase strings (4-12 bytes, `java.util.Random(12345)`, fully
//! deterministic -- no external word list dependency). Regenerate with
//! `fixtures/src/GenBlockTreeMultilevel.java`; see that file's module doc for
//! why a synthetic random string set was used instead of `GenBlockTree`'s
//! existing sequential-numeric "many" field (which never produces a non-leaf
//! block no matter how large: real Lucene's writer resolves that shape via
//! floor blocks instead).

use lucene_codecs::blocktree;
use lucene_codecs::field_infos;

fn dir() -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/blocktree_multilevel_index/"
    )
    .to_string()
}

struct Manifest {
    kv: Vec<(String, String)>,
}

impl Manifest {
    fn load() -> Self {
        let text = std::fs::read_to_string(format!("{}manifest.properties", dir()))
            .expect("run fixtures generator first (GenBlockTreeMultilevel)");
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

/// Every one of the 8000 terms is independently reachable via `seek_exact`,
/// each with the real `docFreq == 1`/`totalTermFreq == 1` (one document per
/// distinct token) -- proves that decoding through at least one non-leaf
/// `.tim` block (see the module doc) still recovers byte-correct terms and
/// stats, not just that `open()` doesn't error.
#[test]
fn multilevel_field_seek_exact_matches_real_lucene() {
    let (fields, m) = open_fixture();
    let many = fields.field("many").expect("expected field \"many\"");

    let num_terms: i64 = m.get("field.many.numTerms").parse().unwrap();
    assert_eq!(num_terms, 8000);
    assert_eq!(many.num_terms, num_terms);

    for term in expected_terms(&m) {
        let stats = many
            .seek_exact(term.as_bytes())
            .unwrap_or_else(|| panic!("expected term {term:?} to be found"));
        assert_eq!(stats.doc_freq, 1, "term={term:?}");
        assert_eq!(stats.total_term_freq, 1, "term={term:?}");
    }

    assert!(many.seek_exact(b"zzz-definitely-missing").is_none());
    assert!(many.seek_exact(b"").is_none());
}

/// Ordered enumeration (`TermsEnum::next()`) across every leaf *and* non-leaf
/// block in this field must reproduce real Lucene's exact sorted term list --
/// a stronger check than sampled `seek_exact` calls, since an off-by-one in
/// how a non-leaf block's sub-block entries get re-prefixed
/// (`decode_block`'s doc comment) would show up as a missing, duplicated, or
/// misordered term somewhere in this 8000-term walk, not necessarily one of
/// the specific terms a smaller spot-check might sample.
#[test]
fn multilevel_field_enumeration_matches_real_lucene_terms_enum_next() {
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
