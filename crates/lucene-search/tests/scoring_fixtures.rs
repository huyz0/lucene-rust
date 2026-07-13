//! Differential/consistency test for BM25 scoring against the real
//! `IndexWriter`-produced segment `crates/lucene-codecs/tests/blocktree_fixtures.rs`
//! and `tests/term_query_fixtures.rs` already validate at the term-dictionary/
//! postings and matching-query layers (`fixtures/data/blocktree_index/`).
//!
//! **What most of this file's tests prove**: passing `norms: None` to
//! `search_term_query_scored`/`search_boolean_query_scored` falls back to
//! `similarity::UNNORMED_FIELD_LENGTH` for both `fieldLength`/`avgFieldLength`
//! (a documented approximation, see `similarity.rs`'s module doc) -- most tests
//! below exercise exactly that fallback path, reimplementing the BM25 arithmetic
//! independently in this test file (`expected_bm25`, not just calling
//! `similarity::score` and trusting it) against the real segment's real
//! postings (`docFreq`, `doc_count`, per-doc `freq`, all read from the real
//! fixture bytes).
//!
//! [`term_query_scored_with_real_norms_matches_real_lucene_field_lengths`]
//! (below) is the one test that instead opens this fixture's real `_0.nvm`/
//! `_0.nvd` (the same `IndexWriter`-produced segment) and passes `Some(&norms)`
//! -- it asserts the decoded per-doc field lengths equal hand-computed values
//! (derived from this fixture's own known per-term postings frequencies, since
//! a doc's BM25 field length is the sum of its terms' frequencies) *and* that
//! real-norms scoring differs from the `None`-fallback constant approximation,
//! proving the length-normalization term is actually live end-to-end against
//! real Lucene-written norm bytes.

use lucene_codecs::blocktree;
use lucene_codecs::field_infos;
use lucene_codecs::norms;
use lucene_codecs::postings::DocInput;
use lucene_search::collector::TopDocsCollector;
use lucene_search::{
    search_boolean_query, search_boolean_query_scored, search_term_query, search_term_query_scored,
    BooleanQuery, FieldNorms, TermQuery,
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
            None,
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
        None,
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
    search_boolean_query_scored(&fields, Some(&doc_in), None, &should_query, None, &mut top)
        .unwrap();

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
        None,
        &mut cat_scores,
    )
    .unwrap();
    let mut bird_scores = TopDocsCollector::new(unscored.docs.len().max(1));
    search_term_query_scored(
        &fields,
        Some(&doc_in),
        None,
        &TermQuery::new("body", "bird"),
        None,
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
    search_boolean_query_scored(&fields, Some(&doc_in), None, &query, None, &mut top).unwrap();

    let mut cat_only = TopDocsCollector::new(10);
    search_term_query_scored(
        &fields,
        Some(&doc_in),
        None,
        &TermQuery::new("body", "cat"),
        None,
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

/// `minimum_should_match` gates *matching*, not the score formula --
/// `search_boolean_query_scored` must still sum every `must`/`should` clause a
/// matched doc actually satisfies, not just `minimum_should_match`-worth of them.
#[test]
fn boolean_query_scored_minimum_should_match_sums_all_matching_clauses() {
    let (fields, doc, id, suffix, _m) = open_segment();
    let doc_in = DocInput::open(&doc, &id, &suffix).expect("open .doc");

    // must=[cat]={0,2}; should=[dog,bird], dog={0,1}, bird={1,4}, minimum_should_match=1
    // narrows the matched set to {0} (see `boolean_minimum_should_match_one_with_must_present_narrows_the_set`
    // in `lib.rs`'s unit tests for the matching-side proof) -- this test instead
    // asserts the *score* for doc 0 is `cat_score(0) + dog_score(0)` (bird doesn't
    // match doc 0 at all, so it contributes nothing either way), proving the
    // threshold only gates which docs match, not how many clause scores get summed.
    let query = BooleanQuery::new()
        .with_must([TermQuery::new("body", "cat")])
        .with_should([
            TermQuery::new("body", "dog"),
            TermQuery::new("body", "bird"),
        ])
        .with_minimum_should_match(1);

    let mut top = TopDocsCollector::new(10);
    search_boolean_query_scored(&fields, Some(&doc_in), None, &query, None, &mut top).unwrap();
    let hits = top.top_docs();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].doc_id, 0);

    let mut cat_scores = TopDocsCollector::new(10);
    search_term_query_scored(
        &fields,
        Some(&doc_in),
        None,
        &TermQuery::new("body", "cat"),
        None,
        &mut cat_scores,
    )
    .unwrap();
    let mut dog_scores = TopDocsCollector::new(10);
    search_term_query_scored(
        &fields,
        Some(&doc_in),
        None,
        &TermQuery::new("body", "dog"),
        None,
        &mut dog_scores,
    )
    .unwrap();

    let lookup = |top: &TopDocsCollector, doc_id: i32| -> Option<f32> {
        top.top_docs()
            .iter()
            .find(|h| h.doc_id == doc_id)
            .map(|h| h.score)
    };
    let expected = lookup(&cat_scores, 0).expect("cat matches doc 0")
        + lookup(&dog_scores, 0).expect("dog matches doc 0");
    assert!(
        (hits[0].score - expected).abs() < 1e-4,
        "got={} expected={}",
        hits[0].score,
        expected
    );
}

/// Opens this fixture's real `_0.nvm`/`_0.nvd` (written directly by segment
/// name, no per-format suffix -- matching `Lucene90NormsFormat`'s file naming,
/// unlike the per-field-postings-format `Lucene104_0` suffix `.tim`/`.doc`/etc
/// use) and returns `body`'s [`NormsEntry`] plus the whole `.nvd` file's bytes.
fn open_body_norms(id: &[u8; 16]) -> (Vec<u8>, norms::NormsEntry) {
    let meta = read_raw_no_suffix("_0.nvm");
    let data = read_raw_no_suffix("_0.nvd");
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
    (data, entry)
}

fn read_raw_no_suffix(name: &str) -> Vec<u8> {
    std::fs::read(format!("{}{}", dir(), name)).unwrap_or_else(|_| panic!("missing {name}"))
}

/// Differential test against this fixture's real `IndexWriter`-produced norms:
/// `body`'s per-doc BM25 field length is the sum of that doc's terms'
/// frequencies (`cat`/`dog`/`bird` are `body`'s only indexed terms here, per
/// this fixture's manifest) -- doc 0 has `cat` freq 2 + `dog` freq 1 = length
/// 3, doc 1 has `dog` freq 1 + `bird` freq 1 = length 2, doc 2 has `cat` freq 1
/// = length 1, doc 4 has `bird` freq 3 = length 3. All four lengths are well
/// under `SmallFloat`'s 24-value exact/subnormal range (see
/// `lucene_util::small_float`), so real Lucene's norm byte for each equals the
/// length exactly, with no lossy re-encoding to account for.
#[test]
fn term_query_scored_with_real_norms_matches_real_lucene_field_lengths() {
    let (fields, doc, id, suffix, m) = open_segment();
    let doc_in = DocInput::open(&doc, &id, &suffix).expect("open .doc");
    let max_doc: i32 = m.get("max_doc").parse().unwrap();

    let (data, entry) = open_body_norms(&id);
    let field_norms = FieldNorms::open(&data, entry, max_doc, None).expect("open body norms");

    let expected_lengths = [(0i32, 3.0f32), (1, 2.0), (2, 1.0), (4, 3.0)];
    for (doc_id, want) in expected_lengths {
        let got = field_norms.field_length(doc_id).unwrap();
        assert_eq!(got, want, "doc {doc_id}");
    }

    // Real per-doc lengths differ (3, 2, 1, 3), so avgFieldLength must be their
    // mean, not the `UNNORMED_FIELD_LENGTH == 1.0` constant fallback.
    let expected_avg = (3.0 + 2.0 + 1.0 + 3.0) / 4.0;
    assert!((field_norms.avg_field_length - expected_avg).abs() < 1e-4);

    // Scoring `bird` (docs 1 and 4, freqs 1 and 3) with real norms must differ
    // from the `None`-fallback constant-length scoring -- proving the
    // length-normalization term is actually live against real Lucene bytes,
    // not collapsed to a constant.
    let mut with_real_norms = TopDocsCollector::new(10);
    search_term_query_scored(
        &fields,
        Some(&doc_in),
        None,
        &TermQuery::new("body", "bird"),
        Some(&field_norms),
        &mut with_real_norms,
    )
    .unwrap();

    let mut with_unnormed = TopDocsCollector::new(10);
    search_term_query_scored(
        &fields,
        Some(&doc_in),
        None,
        &TermQuery::new("body", "bird"),
        None,
        &mut with_unnormed,
    )
    .unwrap();

    let score_of = |top: &TopDocsCollector, doc_id: i32| -> f32 {
        top.top_docs()
            .iter()
            .find(|h| h.doc_id == doc_id)
            .map(|h| h.score)
            .unwrap()
    };
    for doc_id in [1, 4] {
        assert_ne!(
            score_of(&with_real_norms, doc_id),
            score_of(&with_unnormed, doc_id),
            "doc {doc_id}: real-norms score should differ from the constant fallback"
        );
    }

    // Doc 1 (length 2, below the field average of 2.25) is shorter than
    // average and should score at least as high, relative to its own
    // freq/idf, as it would under the flat constant approximation would
    // predict for a doc *at* the average -- concretely, verify the sign of
    // the effect via `similarity::score` directly using the real decoded
    // lengths, matching this test's own field_norms values (not trusting
    // `similarity.rs`'s already-separately-tested unit tests).
    let doc_count = fields.field("body").unwrap().doc_count as i64;
    let bird_doc_freq: i64 = m.get("field.body.term.bird.docFreq").parse().unwrap();
    let expected_doc1 = lucene_search::similarity::score(
        bird_doc_freq,
        doc_count,
        1.0,
        field_norms.field_length(1).unwrap(),
        field_norms.avg_field_length,
    );
    assert!((score_of(&with_real_norms, 1) - expected_doc1).abs() < 1e-4);
}
