//! Differential test for `FuzzyQuery` (task #42) against the same real
//! `IndexWriter`-produced segment `tests/wildcard_query_fixtures.rs`/
//! `tests/prefix_query_fixtures.rs` already validate
//! (`fixtures/data/blocktree_index/`) -- see those files' module docs for
//! the shared fixture-reuse rationale.
//!
//! `body`'s real terms/postings in this fixture (see `GenBlockTree.java`'s
//! own doc comment): `cat` = {0, 2}, `dog` = {0, 1}, `bird` = {1, 4}.
//!
//! `fixtures/src/AppendFuzzyManifest.java` opens this exact fixture
//! directory read-only and runs genuine
//! `org.apache.lucene.search.FuzzyQuery`s (exact match, single substitution/
//! insertion/deletion, a transposition case both with and without
//! `transpositions` enabled -- the single most important case, since it's
//! the exact subtlety real `FuzzyQuery`'s default `transpositions = true`
//! behavior hinges on -- `prefix_length` excluding an otherwise-in-range
//! candidate, `max_edits` boundary cases, and a no-match case) through a
//! real `IndexSearcher`, recording real Lucene's own matched doc IDs into
//! `manifest.properties`' `fuzzy.*` keys -- this is the actual cross-engine
//! proof this port's `FuzzyQuery` matching is checked against, not a
//! hand-derived expectation.

use lucene_codecs::blocktree;
use lucene_codecs::postings::DocInput;
use lucene_search::{search_boolean_query, BooleanQuery, Clause, FuzzyQuery, VecCollector};

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
            .expect("run fixtures generator first (GenBlockTree + AppendFuzzyManifest)");
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

/// Runs `search_boolean_query` with a single `Clause::Fuzzy` `must` clause
/// and returns the matched, ascending doc-ID list.
fn fuzzy_docs(
    field: &str,
    term: &str,
    max_edits: u8,
    prefix_length: usize,
    transpositions: bool,
) -> Vec<i32> {
    let (fields, doc, id, suffix) = open_segment();
    let doc_in = DocInput::open(&doc, &id, &suffix).expect("open .doc");

    let query = BooleanQuery::new().with_must([Clause::from(
        FuzzyQuery::new(field, term.as_bytes().to_vec())
            .with_max_edits(max_edits)
            .with_prefix_length(prefix_length)
            .with_transpositions(transpositions),
    )]);
    let mut c = VecCollector::default();
    search_boolean_query(&fields, Some(&doc_in), None, None, None, &query, &mut c).unwrap();
    c.docs
}

/// Every `fuzzy.<case>.*` manifest entry `AppendFuzzyManifest.java` recorded
/// must be reproduced exactly by this port's own `Clause::Fuzzy` matching --
/// the actual cross-engine check (not a self-consistency test).
#[test]
fn fuzzy_matches_real_lucenes_own_fuzzyquery_output_for_every_case() {
    let m = Manifest::load();
    let cases: Vec<&str> = m.get("fuzzy.cases").split(',').collect();
    assert!(!cases.is_empty(), "manifest must record fuzzy cases");

    for case in cases {
        let field = m.get(&format!("fuzzy.{case}.field"));
        let term = m.get(&format!("fuzzy.{case}.term"));
        let max_edits: u8 = m.get(&format!("fuzzy.{case}.maxEdits")).parse().unwrap();
        let prefix_length: usize = m
            .get(&format!("fuzzy.{case}.prefixLength"))
            .parse()
            .unwrap();
        let transpositions: bool = m
            .get(&format!("fuzzy.{case}.transpositions"))
            .parse()
            .unwrap();
        let docs_key = m.get(&format!("fuzzy.{case}.docs"));
        let expected: Vec<i32> = if docs_key.is_empty() {
            Vec::new()
        } else {
            docs_key.split(',').map(|d| d.parse().unwrap()).collect()
        };

        let got = fuzzy_docs(field, term, max_edits, prefix_length, transpositions);
        assert_eq!(
            got, expected,
            "case={case} field={field} term={term} max_edits={max_edits} \
             prefix_length={prefix_length} transpositions={transpositions}"
        );
    }
}

/// Unit-level checks of `FuzzyQuery` matching semantics not already pinned
/// by the real-Lucene cases above.
#[test]
fn fuzzy_missing_field_matches_nothing() {
    assert_eq!(
        fuzzy_docs("no_such_field", "cat", 2, 0, true),
        Vec::<i32>::new()
    );
}

/// Exact match (distance 0) against the real fixture: `cat` = {0, 2}.
#[test]
fn fuzzy_exact_match_against_real_fixture() {
    assert_eq!(fuzzy_docs("body", "cat", 0, 0, true), vec![0, 2]);
}

/// Composing a `Clause::Fuzzy` inside a `BooleanQuery` alongside a plain
/// `Clause::Term` must intersect (`must`) exactly like any other clause pair
/// -- fuzzy "birt" (distance 1 from "bird", matching {1, 4}) AND "dog"
/// ({0, 1}) -> {1}.
#[test]
fn fuzzy_composes_inside_boolean_query_must() {
    let (fields, doc, id, suffix) = open_segment();
    let doc_in = DocInput::open(&doc, &id, &suffix).expect("open .doc");

    let query = BooleanQuery::new().with_must([
        Clause::from(FuzzyQuery::new("body", b"birt".to_vec())),
        Clause::Term(lucene_search::TermQuery::new("body", "dog")),
    ]);
    let mut c = VecCollector::default();
    search_boolean_query(&fields, Some(&doc_in), None, None, None, &query, &mut c).unwrap();
    assert_eq!(c.docs, vec![1]);
}
