//! Differential test for `RegexpQuery` (task #43) against the same real
//! `IndexWriter`-produced segment `tests/wildcard_query_fixtures.rs`/
//! `tests/fuzzy_query_fixtures.rs` already validate
//! (`fixtures/data/blocktree_index/`) -- see those files' module docs for
//! the shared fixture-reuse rationale.
//!
//! `body`'s real terms/postings in this fixture (see `GenBlockTree.java`'s
//! own doc comment): `cat` = {0, 2}, `dog` = {0, 1}, `bird` = {1, 4}.
//!
//! `fixtures/src/AppendRegexpManifest.java` opens this exact fixture
//! directory read-only and runs genuine
//! `org.apache.lucene.search.RegexpQuery`s (exact literal, the
//! whole-term-match convention -- the single most important case, since
//! it's the exact "looks right in isolation, subtly wrong vs real regex
//! conventions" bug this task called out -- `.`/`*`/`+`/`?` quantifiers,
//! `[...]` classes, `|` alternation across two and three terms, a no-match
//! case, and a missing-field case) through a real `IndexSearcher`, recording
//! real Lucene's own matched doc IDs into `manifest.properties`'
//! `regexp.*` keys -- this is the actual cross-engine proof this port's
//! `RegexpQuery` matching is checked against, not a hand-derived
//! expectation.

use lucene_codecs::blocktree;
use lucene_codecs::postings::DocInput;
use lucene_search::{search_boolean_query, BooleanQuery, Clause, RegexpQuery, VecCollector};

fn dir() -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/blocktree_index/"
    )
    .to_string()
}

struct Manifest {
    kv: Vec<(String, String)>,
}

impl Manifest {
    fn load() -> Self {
        let text = std::fs::read_to_string(format!("{}manifest.properties", dir()))
            .expect("run fixtures generator first (GenBlockTree + AppendRegexpManifest)");
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

fn open_segment() -> (blocktree::BlockTreeFields, Vec<u8>, [u8; 16], String) {
    let m = Manifest::load();
    let id = id_from_hex(m.get("id_hex"));
    let suffix = m.get("segment_suffix").to_string();
    let max_doc: i32 = m.get("max_doc").parse().unwrap();

    let fnm = read_raw(m.get("fnm_file_name"));
    let field_infos = lucene_codecs::field_infos::parse(&fnm, &id, "").expect("parse .fnm");

    let tim = read_raw(m.get("tim_file_name"));
    let tip = read_raw(m.get("tip_file_name"));
    let tmd = read_raw(m.get("tmd_file_name"));
    let fields = blocktree::open(&tim, &tip, &tmd, &field_infos, &id, &suffix, max_doc)
        .expect("open blocktree");

    let doc = read_raw(m.get("doc_file_name"));
    (fields, doc, id, suffix)
}

/// Runs `search_boolean_query` with a single `Clause::Regexp` `must` clause
/// and returns the matched, ascending doc-ID list.
fn regexp_docs(field: &str, pattern: &str) -> Vec<i32> {
    let (fields, doc, id, suffix) = open_segment();
    let doc_in = DocInput::open(&doc, &id, &suffix).expect("open .doc");

    let query = BooleanQuery::new().with_must([Clause::from(RegexpQuery::new(field, pattern))]);
    let mut c = VecCollector::default();
    search_boolean_query(
        &fields,
        Some(&doc_in),
        None,
        None,
        None,
        None,
        &query,
        &mut c,
    )
    .unwrap();
    c.docs
}

/// Every `regexp.<case>.*` manifest entry `AppendRegexpManifest.java`
/// recorded must be reproduced exactly by this port's own `Clause::Regexp`
/// matching -- the actual cross-engine check (not a self-consistency test).
#[test]
fn regexp_matches_real_lucenes_own_regexpquery_output_for_every_case() {
    let m = Manifest::load();
    let cases: Vec<&str> = m.get("regexp.cases").split(',').collect();
    assert!(!cases.is_empty(), "manifest must record regexp cases");

    for case in cases {
        let field = m.get(&format!("regexp.{case}.field"));
        let pattern = m.get(&format!("regexp.{case}.pattern"));
        let docs_key = m.get(&format!("regexp.{case}.docs"));
        let expected: Vec<i32> = if docs_key.is_empty() {
            Vec::new()
        } else {
            docs_key.split(',').map(|d| d.parse().unwrap()).collect()
        };

        let got = regexp_docs(field, pattern);
        assert_eq!(got, expected, "case={case} field={field} pattern={pattern}");
    }
}

/// Unit-level checks of `RegexpQuery` matching semantics not already pinned
/// by the real-Lucene cases above.
#[test]
fn regexp_missing_field_matches_nothing() {
    assert_eq!(regexp_docs("no_such_field", "cat"), Vec::<i32>::new());
}

/// The whole-term-match convention, pinned once more directly against the
/// real fixture (also covered via the manifest's `wholeTermNoSubstringMatch`
/// case above): `ca` must not match `cat`.
#[test]
fn regexp_whole_term_match_against_real_fixture() {
    assert_eq!(regexp_docs("body", "ca"), Vec::<i32>::new());
    assert_eq!(regexp_docs("body", "cat"), vec![0, 2]);
}

/// Composing a `Clause::Regexp` inside a `BooleanQuery` alongside a plain
/// `Clause::Term` must intersect (`must`) exactly like any other clause pair
/// -- "c.t|d.g" (matching "cat" {0,2} and "dog" {0,1}) AND "dog" ({0,1}) ->
/// {0, 1}.
#[test]
fn regexp_composes_inside_boolean_query_must() {
    let (fields, doc, id, suffix) = open_segment();
    let doc_in = DocInput::open(&doc, &id, &suffix).expect("open .doc");

    let query = BooleanQuery::new().with_must([
        Clause::from(RegexpQuery::new("body", "c.t|d.g")),
        Clause::Term(lucene_search::TermQuery::new("body", "dog")),
    ]);
    let mut c = VecCollector::default();
    search_boolean_query(
        &fields,
        Some(&doc_in),
        None,
        None,
        None,
        None,
        &query,
        &mut c,
    )
    .unwrap();
    assert_eq!(c.docs, vec![0, 1]);
}

/// A malformed pattern (unsupported `{n,m}` syntax) surfaces as an
/// `Err(Error::Regexp(_))`, not a panic or a silent empty match.
#[test]
fn regexp_malformed_pattern_is_an_error_not_a_panic() {
    let (fields, doc, id, suffix) = open_segment();
    let doc_in = DocInput::open(&doc, &id, &suffix).expect("open .doc");

    let query = BooleanQuery::new().with_must([Clause::from(RegexpQuery::new("body", "a{2,3}"))]);
    let mut c = VecCollector::default();
    let result = search_boolean_query(
        &fields,
        Some(&doc_in),
        None,
        None,
        None,
        None,
        &query,
        &mut c,
    );
    assert!(matches!(result, Err(lucene_search::Error::Regexp(_))));
}
