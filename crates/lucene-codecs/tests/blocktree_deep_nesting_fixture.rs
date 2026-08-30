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
// Test-support code opts out of the arithmetic gate at the file boundary:
// the gate exists for values read off disk in production decode paths, not
// for a fixture builder's own index arithmetic. See
// `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

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

/// Exhaustive `seekCeil` differential over the deepest block chain this port
/// has real Lucene bytes for.
///
/// `SegmentTermsEnum.seekCeil` is the hardest of the lazy navigator's entry
/// points: the trie descent can run out mid-target, the scan can end past a
/// block's last entry (in which case Java falls through to `next()`, which
/// pops the frame stack and may re-descend into a *deeper* sub-block to find
/// the true ceiling), and a non-leaf scan can stop on a sub-block pointer
/// that sorts after the target, which has to be descended into rather than
/// returned. None of those are visible from `seek_exact`.
///
/// So this checks `seek_ceil` against a brute-force answer computed from the
/// real Lucene term list, for four target families per term: the term itself
/// (`FOUND`), the term with a byte appended (a target strictly between two
/// real terms, or past the last one), the term with its last byte dropped,
/// and the term with its last byte incremented. Then it checks that `next()`
/// after a ceiling seek continues from where the seek landed.
#[test]
fn deep_nesting_field_seek_ceil_matches_a_brute_force_ceiling() {
    let (fields, m) = open_fixture();
    let many = fields.field("many").expect("expected field \"many\"");
    let terms = expected_terms(&m);
    assert!(terms.windows(2).all(|w| w[0] < w[1]), "terms are sorted");

    let mut targets: Vec<Vec<u8>> = Vec::new();
    for t in &terms {
        let b = t.as_bytes();
        targets.push(b.to_vec());
        let mut appended = b.to_vec();
        appended.push(b'a');
        targets.push(appended);
        if b.len() > 1 {
            targets.push(b[..b.len() - 1].to_vec());
            let mut bumped = b.to_vec();
            *bumped.last_mut().unwrap() += 1;
            targets.push(bumped);
        }
    }
    targets.push(Vec::new());
    targets.push(b"\xff".to_vec());

    // Deliberately *not* in sorted order, and deliberately down **one reused
    // enum**: the batch's single intentional divergence from Java is that
    // `SegmentTermsEnumFrame::rewind` resets a loaded frame's cursors in place
    // where Java forces a block reload, and that path is only reachable on the
    // second and later seek against the same frame stack -- most of all on a
    // *backwards* seek, which is what `rewind` exists for. A fresh
    // `many.iter()` per target would never touch it.
    let mut state = 0x243F_6A88_85A3_08D3u64;
    for i in (1..targets.len()).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        targets.swap(i, (state % (i as u64 + 1)) as usize);
    }
    let mut reused = many.iter();

    for target in &targets {
        // Brute force: the first term >= target, if any.
        let at = terms.partition_point(|t| t.as_bytes() < target.as_slice());
        let expected = terms.get(at);

        // Both spellings must agree: a fresh enum, and the reused one whose
        // frames still hold the previous target's blocks.
        let mut it = many.iter();
        let status = it.seek_ceil(target);
        assert_eq!(
            reused.seek_ceil(target),
            status,
            "reused enum disagreed on target={target:?}"
        );
        assert_eq!(
            reused.current().map(|(t, _)| t.to_vec()),
            it.current().map(|(t, _)| t.to_vec()),
            "reused enum landed elsewhere on target={target:?}"
        );
        match expected {
            None => {
                assert_eq!(
                    status,
                    blocktree::SeekStatus::End,
                    "target={target:?} is past the last term"
                );
                assert!(it.current().is_none());
                assert!(it.next().is_none());
            }
            Some(want) => {
                let expected_status = if want.as_bytes() == target.as_slice() {
                    blocktree::SeekStatus::Found
                } else {
                    blocktree::SeekStatus::NotFound
                };
                assert_eq!(status, expected_status, "target={target:?}");
                let (got, stats) = it
                    .current()
                    .unwrap_or_else(|| panic!("target={target:?} left the enum unpositioned"));
                assert_eq!(
                    std::str::from_utf8(got).unwrap(),
                    want.as_str(),
                    "target={target:?}"
                );
                assert_eq!(stats.doc_freq, 1);

                // `next()` must continue from the landed-on term, not restart.
                if let Some(after) = terms.get(at + 1) {
                    let (nxt, _) = it.next().expect("a term follows the ceiling");
                    assert_eq!(
                        std::str::from_utf8(nxt).unwrap(),
                        after.as_str(),
                        "next() after seek_ceil({target:?})"
                    );
                } else {
                    assert!(it.next().is_none());
                }
            }
        }
    }
}

/// The infallible lookups have `Result`-returning twins, and on intact bytes
/// the two must agree everywhere -- `try_seek_exact`/`try_next`/
/// `try_seek_ceil` are the spellings new callers are meant to use, so they
/// need the same fixture proof the older ones have.
#[test]
fn deep_nesting_field_fallible_lookups_agree_with_the_infallible_ones() {
    let (fields, m) = open_fixture();
    let many = fields.field("many").expect("expected field \"many\"");
    let terms = expected_terms(&m);

    for term in terms.iter().step_by(37) {
        assert_eq!(
            many.try_seek_exact(term.as_bytes()).expect("intact bytes"),
            many.seek_exact(term.as_bytes()),
            "term={term:?}"
        );
    }
    assert_eq!(many.try_seek_exact(b"zzzz").expect("intact bytes"), None);

    let mut a = many.iter();
    let mut b = many.iter();
    loop {
        let want = a
            .try_next()
            .expect("intact bytes")
            .map(|(t, s)| (t.to_vec(), s));
        let got = b.next().map(|(t, s)| (t.to_vec(), s));
        assert_eq!(want, got);
        if want.is_none() {
            break;
        }
    }

    let mut it = many.iter();
    assert_eq!(
        it.try_seek_ceil(terms[10].as_bytes())
            .expect("intact bytes"),
        blocktree::SeekStatus::Found
    );
}
