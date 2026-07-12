//! Differential/consistency test for BM25 scoring against the real
//! `IndexWriter`-produced segment `crates/lucene-codecs/tests/blocktree_fixtures.rs`
//! and `tests/term_query_fixtures.rs` already validate at the term-dictionary/
//! postings and matching-query layers (`fixtures/data/blocktree_index/`).
//!
//! **What this test does and does not prove**: `crates/lucene-search/src/similarity.rs`'s
//! module doc explains that this port has no norms *reader* yet, so every score
//! here uses a constant field-length substitution rather than this segment's real
//! per-document field lengths -- scores computed here are **not** expected to
//! numerically match real Lucene's BM25 output for the same query (that would
//! need a norms reader plus a Java-side BM25 fixture generator, deferred per
//! `docs/parity.md`'s scoring row). What this test *does* assert against real
//! Lucene data: that scoring the real segment's real postings (`docFreq`,
//! `doc_count`, per-doc `freq`, all read from the real fixture bytes) through
//! `search_term_query_scored`/`search_boolean_query_scored` produces internally
//! consistent, correctly-ordered results -- the same matched-doc set the
//! unscored `search_term_query`/`search_boolean_query` (already fixture-verified)
//! returns, real `TopDocsCollector` tie-break ordering, and every score computed
//! independently from the real `TermStats`/`Postings` this fixture's own decode
//! already produces (reimplementing the BM25 arithmetic in this test file, not
//! just calling `similarity::score` and trusting it -- that formula already has
//! its own hand-computed unit tests in `similarity.rs`).

use lucene_codecs::blocktree;
use lucene_codecs::field_infos;
use lucene_codecs::postings::DocInput;
use lucene_search::collector::TopDocsCollector;
use lucene_search::{
    search_boolean_query, search_boolean_query_scored, search_term_query, search_term_query_scored,
    BooleanQuery, TermQuery,
};

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
            .expect("run fixtures generator first (GenBlockTree)");
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

fn open_segment() -> (
    blocktree::BlockTreeFields,
    Vec<u8>,
    [u8; 16],
    String,
    Manifest,
) {
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

    let doc = read_raw(m.get("doc_file_name"));
    (fields, doc, id, suffix, m)
}

/// Reimplements the BM25 formula independently of `similarity::score` (see this
/// file's module doc), using this port's own constant field-length substitution
/// (`UNNORMED_FIELD_LENGTH == 1.0`, so the length-normalization term collapses to
/// `b`).
fn expected_bm25(doc_freq: i64, doc_count: i64, freq: f32) -> f32 {
    let idf = (1.0 + (doc_count as f64 - doc_freq as f64 + 0.5) / (doc_freq as f64 + 0.5)).ln();
    let k1 = 1.2_f64;
    let b = 0.75_f64;
    let tf_norm = (freq as f64) * (k1 + 1.0) / (freq as f64 + k1 * (1.0 - b + b));
    (idf * tf_norm) as f32
}

/// `body`'s three terms all have `docFreq == 2`: scoring each independently and
/// comparing the *set* of scored doc IDs against the already-fixture-verified
/// unscored `search_term_query` result proves the scored path doesn't drop or
/// invent matches, only attaches scores to the same real doc set.
#[test]
fn term_query_scored_matches_unscored_doc_set_and_hand_computed_scores() {
    let (fields, doc, id, suffix, m) = open_segment();
    let doc_in = DocInput::open(&doc, &id, &suffix).expect("open .doc");
    let doc_count = fields.field("body").unwrap().doc_count as i64;

    for term in ["cat", "dog", "bird"] {
        let doc_freq: i64 = m
            .get(&format!("field.body.term.{term}.docFreq"))
            .parse()
            .unwrap();

        let mut unscored = lucene_search::VecCollector::default();
        search_term_query(
            &fields,
            Some(&doc_in),
            None,
            &TermQuery::new("body", term),
            &mut unscored,
        )
        .unwrap();

        let mut top = TopDocsCollector::new(unscored.docs.len().max(1));
        search_term_query_scored(
            &fields,
            Some(&doc_in),
            None,
            &TermQuery::new("body", term),
            &mut top,
        )
        .unwrap();

        let scored_docs: Vec<i32> = top.top_docs().iter().map(|h| h.doc_id).collect();
        let mut sorted_scored_docs = scored_docs.clone();
        sorted_scored_docs.sort_unstable();
        assert_eq!(sorted_scored_docs, unscored.docs, "term={term:?}");

        // Real Lucene's `TopScoreDocCollector` ordering: strictly non-increasing
        // score, ascending doc ID on any tie.
        for pair in top.top_docs().windows(2) {
            assert!(
                pair[0].score > pair[1].score
                    || (pair[0].score == pair[1].score && pair[0].doc_id < pair[1].doc_id),
                "top_docs not correctly ordered for term={term:?}: {:?}",
                top.top_docs()
            );
        }

        // Independently re-derive each hit's expected score from the real
        // `Postings.freqs` this fixture's own decode already produces (not by
        // trusting `similarity::score`'s own unit tests).
        let postings = fields
            .field("body")
            .unwrap()
            .postings(term.as_bytes(), Some(&doc_in))
            .unwrap()
            .unwrap();
        for hit in top.top_docs() {
            let idx = postings.docs.iter().position(|&d| d == hit.doc_id).unwrap();
            let freq = postings.freqs[idx] as f32;
            let expected = expected_bm25(doc_freq, doc_count, freq);
            assert!(
                (hit.score - expected).abs() < 1e-4,
                "term={term:?} doc={} got={} expected={}",
                hit.doc_id,
                hit.score,
                expected
            );
        }
    }
}

/// `big`/`everywhere` (`docFreq == 300`): the multi-block `.doc`-backed scoring
/// path, checked with a small `top_n` to exercise real eviction against a real
/// 300-doc postings list.
#[test]
fn term_query_scored_top_n_eviction_keeps_highest_scoring_real_docs() {
    let (fields, doc, id, suffix, _m) = open_segment();
    let doc_in = DocInput::open(&doc, &id, &suffix).expect("open .doc");

    let mut top = TopDocsCollector::new(5);
    search_term_query_scored(
        &fields,
        Some(&doc_in),
        None,
        &TermQuery::new("big", "everywhere"),
        &mut top,
    )
    .unwrap();

    assert_eq!(top.top_docs().len(), 5);
    for pair in top.top_docs().windows(2) {
        assert!(
            pair[0].score > pair[1].score
                || (pair[0].score == pair[1].score && pair[0].doc_id < pair[1].doc_id)
        );
    }
}

/// `search_boolean_query_scored`'s matched-doc set must equal
/// `search_boolean_query`'s (already fixture-verified) result, and every score
/// must be the sum of the contributing clauses' individual BM25 scores.
#[test]
fn boolean_query_scored_matches_unscored_doc_set_and_sums_clause_scores() {
    let (fields, doc, id, suffix, _m) = open_segment();
    let doc_in = DocInput::open(&doc, &id, &suffix).expect("open .doc");

    let mut unscored = lucene_search::VecCollector::default();
    let should_query = BooleanQuery::new().with_should([
        TermQuery::new("body", "cat"),
        TermQuery::new("body", "bird"),
    ]);
    search_boolean_query(&fields, Some(&doc_in), None, &should_query, &mut unscored).unwrap();

    let mut top = TopDocsCollector::new(unscored.docs.len().max(1));
    search_boolean_query_scored(&fields, Some(&doc_in), None, &should_query, &mut top).unwrap();

    let mut scored_docs: Vec<i32> = top.top_docs().iter().map(|h| h.doc_id).collect();
    scored_docs.sort_unstable();
    assert_eq!(scored_docs, unscored.docs);

    // Independently compute each single-clause score via `search_term_query_scored`
    // and confirm the boolean disjunction's score is their sum for docs matching
    // both clauses, or exactly the one matching clause's score otherwise.
    let mut cat_scores = TopDocsCollector::new(unscored.docs.len().max(1));
    search_term_query_scored(
        &fields,
        Some(&doc_in),
        None,
        &TermQuery::new("body", "cat"),
        &mut cat_scores,
    )
    .unwrap();
    let mut bird_scores = TopDocsCollector::new(unscored.docs.len().max(1));
    search_term_query_scored(
        &fields,
        Some(&doc_in),
        None,
        &TermQuery::new("body", "bird"),
        &mut bird_scores,
    )
    .unwrap();

    let lookup = |top: &TopDocsCollector, doc_id: i32| -> Option<f32> {
        top.top_docs()
            .iter()
            .find(|h| h.doc_id == doc_id)
            .map(|h| h.score)
    };

    for hit in top.top_docs() {
        let expected = lookup(&cat_scores, hit.doc_id).unwrap_or(0.0)
            + lookup(&bird_scores, hit.doc_id).unwrap_or(0.0);
        assert!(
            (hit.score - expected).abs() < 1e-4,
            "doc={} got={} expected={}",
            hit.doc_id,
            hit.score,
            expected
        );
    }
}

/// `must_not` clauses never contribute to the score -- a pure filter, matching
/// real `Occur.MUST_NOT`'s contract.
#[test]
fn boolean_query_scored_must_not_clause_never_contributes_to_score() {
    let (fields, doc, id, suffix, _m) = open_segment();
    let doc_in = DocInput::open(&doc, &id, &suffix).expect("open .doc");

    let query = BooleanQuery::new()
        .with_must([TermQuery::new("body", "cat")])
        .with_must_not([TermQuery::new("body", "dog")]);

    let mut top = TopDocsCollector::new(10);
    search_boolean_query_scored(&fields, Some(&doc_in), None, &query, &mut top).unwrap();

    let mut cat_only = TopDocsCollector::new(10);
    search_term_query_scored(
        &fields,
        Some(&doc_in),
        None,
        &TermQuery::new("body", "cat"),
        &mut cat_only,
    )
    .unwrap();

    for hit in top.top_docs() {
        let cat_score = cat_only
            .top_docs()
            .iter()
            .find(|h| h.doc_id == hit.doc_id)
            .map(|h| h.score)
            .expect("doc must be one of cat's matches");
        assert!(
            (hit.score - cat_score).abs() < 1e-4,
            "must_not clause changed the score: got={} expected={}",
            hit.score,
            cat_score
        );
    }
}
