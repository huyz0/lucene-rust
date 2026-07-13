//! Differential/consistency test for `DisjunctionMaxQuery` (task #32) against
//! the same real `IndexWriter`-produced segment `tests/scoring_fixtures.rs`
//! and `tests/boolean_query_fixtures.rs` already validate
//! (`fixtures/data/blocktree_index/`) -- see those files' module docs for why
//! this "reuse an already-verified real segment, independently recompute the
//! expected combination from its already-fixture-proven single-clause
//! results" pattern counts as a real differential test rather than a
//! synthetic unit test: every doc ID and every BM25 score involved is decoded
//! straight from real Lucene-written bytes, only the dismax combination
//! arithmetic (`max(disjunct scores) + tie_breaker * sum(rest)`) is
//! independently re-derived here rather than trusted from `lib.rs`'s own
//! implementation.
//!
//! `body`'s real postings in this fixture (see `scoring_fixtures.rs`'s own
//! module doc and manifest): `cat` = {0, 2}, `dog` = {0, 1}, `bird` = {1, 4}.

use lucene_codecs::blocktree;
use lucene_codecs::field_infos;
use lucene_codecs::norms;
use lucene_codecs::postings::DocInput;
use lucene_search::collector::TopDocsCollector;
use lucene_search::{
    search_disjunction_max_query, search_disjunction_max_query_scored, search_term_query_scored,
    Clause, DisjunctionMaxQuery, FieldNorms, TermQuery, VecCollector,
};
use std::collections::HashMap;

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

fn read_raw_no_suffix(name: &str) -> Vec<u8> {
    std::fs::read(format!("{}{}", dir(), name)).unwrap_or_else(|_| panic!("missing {name}"))
}

/// Opens this fixture's real `_0.nvm`/`_0.nvd` (`body`'s real per-doc/avg
/// field length -- see `scoring_fixtures.rs`'s `open_body_norms`, same
/// pattern) -- needed because real Lucene's default `IndexSearcher` scores
/// with real norms, not the `UNNORMED_FIELD_LENGTH` constant fallback other
/// tests in this file deliberately exercise; matching real Lucene's own
/// recorded `DisjunctionMaxQuery` scores exactly requires the same real norms.
fn open_body_field_norms<'a>(id: &[u8; 16], data: &'a [u8], max_doc: i32) -> FieldNorms<'a> {
    let meta = read_raw_no_suffix("_0.nvm");
    let manifest = Manifest::load();
    let field_infos_buf = read_raw(manifest.get("fnm_file_name"));
    let field_infos = field_infos::parse(&field_infos_buf, id, "").expect("parse .fnm");
    let body_number = field_infos
        .fields
        .iter()
        .find(|f| f.name == "body")
        .expect("body field")
        .number;
    let (_, parsed) = norms::parse_meta(&meta, id, "").expect("parse .nvm");
    let entry = *parsed.entry(body_number).expect("body has a norms entry");
    FieldNorms::open(data, entry, max_doc, None).expect("open body norms")
}

fn open_segment() -> (blocktree::BlockTreeFields, Vec<u8>, [u8; 16], String) {
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
    (fields, doc, id, suffix)
}

/// `search_disjunction_max_query`'s matched set must equal the real, already
/// fixture-verified union of `cat`={0,2} and `bird`={1,4}: {0,1,2,4}.
#[test]
fn dismax_matches_the_real_union_of_its_disjuncts() {
    let (fields, doc, id, suffix) = open_segment();
    let doc_in = DocInput::open(&doc, &id, &suffix).expect("open .doc");

    let query = DisjunctionMaxQuery::new(
        [
            Clause::Term(TermQuery::new("body", "cat")),
            Clause::Term(TermQuery::new("body", "bird")),
        ],
        0.0,
    );
    let mut c = VecCollector::default();
    search_disjunction_max_query(&fields, Some(&doc_in), None, None, None, &query, &mut c).unwrap();
    assert_eq!(c.docs, vec![0, 1, 2, 4]);
}

/// Scored differential proof of the exact dismax formula against real BM25
/// scores decoded from this fixture's real postings: doc 0 is the only doc
/// matching both `cat` and `dog`, so its dismax score must equal
/// `max(cat_score(0), dog_score(0)) + tie_breaker * min(cat_score(0),
/// dog_score(0))` -- independently re-derived here from two standalone
/// `search_term_query_scored` calls against the same real segment, not from
/// trusting `dismax_scores`'s own implementation.
#[test]
fn dismax_scored_tie_breaker_formula_matches_real_bm25_scores() {
    let (fields, doc, id, suffix) = open_segment();
    let doc_in = DocInput::open(&doc, &id, &suffix).expect("open .doc");

    let mut cat = TopDocsCollector::new(10);
    search_term_query_scored(
        &fields,
        Some(&doc_in),
        None,
        &TermQuery::new("body", "cat"),
        None,
        &mut cat,
    )
    .unwrap();
    let mut dog = TopDocsCollector::new(10);
    search_term_query_scored(
        &fields,
        Some(&doc_in),
        None,
        &TermQuery::new("body", "dog"),
        None,
        &mut dog,
    )
    .unwrap();
    let score_of = |top: &TopDocsCollector, doc_id: i32| -> f32 {
        top.top_docs()
            .iter()
            .find(|h| h.doc_id == doc_id)
            .map(|h| h.score)
            .unwrap()
    };
    let cat0 = score_of(&cat, 0);
    let dog0 = score_of(&dog, 0);
    assert_ne!(
        cat0, dog0,
        "test needs distinct real scores to distinguish max-plus-tiebreak from a plain sum"
    );

    for tie_breaker in [0.0f32, 0.3, 1.0] {
        let expected = cat0.max(dog0) + tie_breaker * cat0.min(dog0);

        let query = DisjunctionMaxQuery::new(
            [
                Clause::Term(TermQuery::new("body", "cat")),
                Clause::Term(TermQuery::new("body", "dog")),
            ],
            tie_breaker,
        );
        let mut top = TopDocsCollector::new(10);
        search_disjunction_max_query_scored(
            &fields,
            Some(&doc_in),
            None,
            None,
            None,
            &query,
            None,
            &mut top,
        )
        .unwrap();
        assert!(
            (score_of(&top, 0) - expected).abs() < 1e-4,
            "tie_breaker={tie_breaker}: got={} expected={}",
            score_of(&top, 0),
            expected
        );
    }

    // `tie_breaker == 1.0` degenerates to a plain sum -- a useful sanity check
    // on the formula's other extreme, since real Lucene documents
    // `tie_breaker == 1.0` as equivalent to summing every matching disjunct.
    let query_sum = DisjunctionMaxQuery::new(
        [
            Clause::Term(TermQuery::new("body", "cat")),
            Clause::Term(TermQuery::new("body", "dog")),
        ],
        1.0,
    );
    let mut top_sum = TopDocsCollector::new(10);
    search_disjunction_max_query_scored(
        &fields,
        Some(&doc_in),
        None,
        None,
        None,
        &query_sum,
        None,
        &mut top_sum,
    )
    .unwrap();
    assert!((score_of(&top_sum, 0) - (cat0 + dog0)).abs() < 1e-4);
}

/// A doc matching only one disjunct (doc 2: `cat` only, not `dog`) must score
/// exactly that disjunct's own real BM25 score, regardless of `tie_breaker`
/// (the "sum of every *other* matching disjunct" term is zero).
#[test]
fn dismax_scored_single_matching_disjunct_gets_its_own_real_score_exactly() {
    let (fields, doc, id, suffix) = open_segment();
    let doc_in = DocInput::open(&doc, &id, &suffix).expect("open .doc");

    let mut cat = TopDocsCollector::new(10);
    search_term_query_scored(
        &fields,
        Some(&doc_in),
        None,
        &TermQuery::new("body", "cat"),
        None,
        &mut cat,
    )
    .unwrap();
    let cat2 = cat
        .top_docs()
        .iter()
        .find(|h| h.doc_id == 2)
        .map(|h| h.score)
        .expect("cat matches doc 2 in this fixture");

    let query = DisjunctionMaxQuery::new(
        [
            Clause::Term(TermQuery::new("body", "cat")),
            Clause::Term(TermQuery::new("body", "dog")),
        ],
        0.5,
    );
    let mut top = TopDocsCollector::new(10);
    search_disjunction_max_query_scored(
        &fields,
        Some(&doc_in),
        None,
        None,
        None,
        &query,
        None,
        &mut top,
    )
    .unwrap();
    let got = top
        .top_docs()
        .iter()
        .find(|h| h.doc_id == 2)
        .map(|h| h.score)
        .expect("doc 2 must appear in the dismax result");
    assert!((got - cat2).abs() < 1e-4);
}

/// **Real cross-engine ground truth** (not a self-consistency check like the
/// tests above): `fixtures/src/AppendDismaxManifest.java` opens this exact
/// fixture directory read-only and runs a genuine
/// `org.apache.lucene.search.DisjunctionMaxQuery` (over `body:cat`/`body:dog`,
/// `tieBreakerMultiplier=0.3`) through a real `IndexSearcher`, recording real
/// Lucene's own `TopDocs` (doc, score) pairs into `manifest.properties`'
/// `dismax.realLuceneDocScores` key (see that file's module doc for why it's
/// a standalone manifest-appending tool rather than folded into
/// `GenBlockTree.java`: regenerating the whole segment would assign a new
/// random segment ID, which `lucene-ffi`'s test suite hardcodes). This test
/// re-runs the equivalent query through this port's own
/// `search_disjunction_max_query_scored` and asserts doc-for-doc, score-for-
/// score agreement with real Lucene's recorded output -- the actual
/// cross-engine proof this port's dismax formula is checked against.
#[test]
fn dismax_scored_matches_real_lucenes_own_disjunctionmaxquery_output() {
    let (fields, doc, id, suffix) = open_segment();
    let doc_in = DocInput::open(&doc, &id, &suffix).expect("open .doc");
    let m = Manifest::load();
    let max_doc: i32 = m.get("max_doc").parse().unwrap();

    // Real Lucene's default `IndexSearcher` scores with real per-doc/avg
    // `body` norms (this fixture's `bodyType` never calls `setOmitNorms`),
    // not the `UNNORMED_FIELD_LENGTH` constant fallback -- matching its
    // recorded scores exactly requires the same real norms here.
    let norms_data = read_raw_no_suffix("_0.nvd");
    let body_norms = open_body_field_norms(&id, &norms_data, max_doc);
    let mut norms_map = HashMap::new();
    norms_map.insert("body".to_string(), body_norms);

    let tie_breaker: f32 = m.get("dismax.tieBreaker").parse().unwrap();
    let term_a_field = m.get("dismax.termA.field");
    let term_a_term = m.get("dismax.termA.term");
    let term_b_field = m.get("dismax.termB.field");
    let term_b_term = m.get("dismax.termB.term");

    let expected: Vec<(i32, f32)> = m
        .get("dismax.realLuceneDocScores")
        .split(',')
        .map(|pair| {
            let (doc_str, score_str) = pair.split_once(':').expect("doc:score pair");
            (doc_str.parse().unwrap(), score_str.parse().unwrap())
        })
        .collect();
    assert!(
        !expected.is_empty(),
        "manifest must record at least one real Lucene dismax hit"
    );

    let query = DisjunctionMaxQuery::new(
        [
            Clause::Term(TermQuery::new(term_a_field, term_a_term)),
            Clause::Term(TermQuery::new(term_b_field, term_b_term)),
        ],
        tie_breaker,
    );
    let mut top = TopDocsCollector::new(10);
    search_disjunction_max_query_scored(
        &fields,
        Some(&doc_in),
        None,
        None,
        None,
        &query,
        Some(&norms_map),
        &mut top,
    )
    .unwrap();

    let mut got: Vec<(i32, f32)> = top.top_docs().iter().map(|h| (h.doc_id, h.score)).collect();
    got.sort_unstable_by_key(|&(doc_id, _)| doc_id);
    let mut expected_sorted = expected.clone();
    expected_sorted.sort_unstable_by_key(|&(doc_id, _)| doc_id);

    assert_eq!(
        got.iter().map(|&(d, _)| d).collect::<Vec<_>>(),
        expected_sorted.iter().map(|&(d, _)| d).collect::<Vec<_>>(),
        "matched doc-ID set must equal real Lucene's own DisjunctionMaxQuery result"
    );
    for ((got_doc, got_score), (exp_doc, exp_score)) in got.iter().zip(expected_sorted.iter()) {
        assert_eq!(got_doc, exp_doc);
        assert!(
            (got_score - exp_score).abs() < 1e-4,
            "doc={got_doc}: got={got_score} real_lucene={exp_score}"
        );
    }
}
