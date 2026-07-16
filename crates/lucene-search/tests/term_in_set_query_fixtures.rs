//! Tests for `TermInSetQuery` (real Lucene's `org.apache.lucene.search.
//! TermInSetQuery`) against the same real `IndexWriter`-produced segment
//! `tests/boolean_query_fixtures.rs`/`tests/wildcard_query_fixtures.rs`
//! already validate (`fixtures/data/blocktree_index/`).
//!
//! `body`'s real terms/postings in this fixture (see `GenBlockTree.java`'s
//! own doc comment, already relied on by `wildcard_query_fixtures.rs`):
//! `cat` = {0, 2}, `dog` = {0, 1}, `bird` = {1, 4}.
//!
//! No separate Java fixture generator is needed here (unlike
//! `wildcard_query_fixtures.rs`'s `AppendWildcardManifest.java`):
//! `TermInSetQuery`'s matching semantics reduce to a plain per-term postings
//! union, already exercised term-by-term by the existing fixture/manifest
//! (each of `cat`/`dog`/`bird`'s doc sets is independently known), so the
//! cross-engine proof is inherited from that per-term data rather than
//! needing a new genuine `TermInSetQuery` run recorded separately.
//!
//! **Scoring**: verified directly against real `TermInSetQuery.java`'s class
//! doc comment ("NOTE: This query produces scores that are equal to its
//! boost") -- flat `1.0` per matching doc, not `max`-of-matched-terms and not
//! summed like a `BooleanQuery` of `SHOULD` clauses would. See
//! `term_in_set_scores_flat_1_0_not_summed_like_a_boolean_should_disjunction`
//! below for the test that pins this down against the summed-should
//! alternative.

use lucene_codecs::blocktree;
use lucene_codecs::postings::DocInput;
use lucene_search::{
    search_boolean_query, search_boolean_query_scored, BooleanQuery, Clause, TermInSetQuery,
    TermQuery, TopDocsCollector, VecCollector,
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

/// Runs `search_boolean_query` with a single `Clause::TermInSet(field,
/// terms)` `must` clause and returns the matched, ascending doc-ID list.
fn term_in_set_docs(field: &str, terms: &[&[u8]]) -> Vec<i32> {
    let (fields, doc, id, suffix) = open_segment();
    let doc_in = DocInput::open(&doc, &id, &suffix).expect("open .doc");

    let query = BooleanQuery::new().with_must([Clause::from(TermInSetQuery::new(
        field,
        terms.iter().map(|t| t.to_vec()),
    ))]);
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

/// A doc matches if it contains ANY of the given terms, not all of them:
/// `cat` = {0, 2}, `dog` = {0, 1} -- neither doc 1 nor doc 2 contains both,
/// yet both must appear in the union {0, 1, 2}.
#[test]
fn matches_any_of_the_given_terms_not_all() {
    assert_eq!(term_in_set_docs("body", &[b"cat", b"dog"]), vec![0, 1, 2]);
}

/// Three-term union: `cat` = {0, 2}, `dog` = {0, 1}, `bird` = {1, 4} ->
/// {0, 1, 2, 4}.
#[test]
fn unions_postings_across_more_than_two_terms() {
    assert_eq!(
        term_in_set_docs("body", &[b"cat", b"dog", b"bird"]),
        vec![0, 1, 2, 4]
    );
}

/// A term absent from the index contributes nothing -- no error, just no
/// extra doc IDs beyond what the present terms already match.
#[test]
fn absent_term_contributes_nothing_and_does_not_error() {
    assert_eq!(
        term_in_set_docs("body", &[b"cat", b"no_such_term"]),
        vec![0, 2]
    );
}

/// Every given term absent from the index -> matches nothing, no error.
#[test]
fn all_absent_terms_matches_nothing() {
    assert_eq!(
        term_in_set_docs("body", &[b"no_such_term", b"also_missing"]),
        Vec::<i32>::new()
    );
}

/// An empty term set matches nothing (an empty union is empty).
#[test]
fn empty_term_set_matches_nothing() {
    let empty: &[&[u8]] = &[];
    assert_eq!(term_in_set_docs("body", empty), Vec::<i32>::new());
}

/// A `TermInSetQuery` against a field that doesn't exist in this segment
/// matches nothing, same "missing field" convention every other clause
/// follows.
#[test]
fn missing_field_matches_nothing() {
    assert_eq!(
        term_in_set_docs("no_such_field", &[b"cat"]),
        Vec::<i32>::new()
    );
}

/// Composes correctly as a `BooleanQuery` clause: a `TermInSet(["cat",
/// "bird"])` (matches {0, 2, 1, 4}) `must`-combined with a plain
/// `Clause::Term("dog")` (matches {0, 1}) intersects to {0, 1}.
#[test]
fn composes_as_a_boolean_query_must_clause() {
    let (fields, doc, id, suffix) = open_segment();
    let doc_in = DocInput::open(&doc, &id, &suffix).expect("open .doc");

    let query = BooleanQuery::new().with_must([
        Clause::from(TermInSetQuery::new(
            "body",
            [b"cat".to_vec(), b"bird".to_vec()],
        )),
        Clause::Term(TermQuery::new("body", "dog")),
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

/// Composes correctly as a `BooleanQuery` `should` clause too: `TermInSet
/// (["bird"])` (matches {1, 4}) as one `should` alongside plain `Term("cat")`
/// (matches {0, 2}) unions to {0, 1, 2, 4}.
#[test]
fn composes_as_a_boolean_query_should_clause() {
    let (fields, doc, id, suffix) = open_segment();
    let doc_in = DocInput::open(&doc, &id, &suffix).expect("open .doc");

    let query = BooleanQuery::new().with_should([
        Clause::from(TermInSetQuery::new("body", [b"bird".to_vec()])),
        Clause::Term(TermQuery::new("body", "cat")),
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
    assert_eq!(c.docs, vec![0, 1, 2, 4]);
}

/// **Scoring formula check, verified against real Lucene's own
/// `TermInSetQuery.java` class doc comment**: "NOTE: This query produces
/// scores that are equal to its boost" -- i.e. every matching doc scores flat
/// `1.0` (no boost wrapper here), *regardless* of how many of the given terms
/// matched that doc. This is explicitly checked against two alternative
/// (wrong) formulas this task's own instructions called out as easy to
/// assume by mistake:
///
/// - NOT `max`-of-matched-terms' own idf-based scores (there is no such
///   per-term score computed at all in this port's unscored-clause
///   convention -- see `Clause::TermInSet`'s doc comment).
/// - NOT summed like a `BooleanQuery` of `SHOULD` clauses would sum each
///   matched clause's own score (doc 0 matches both `cat` and `dog`; if this
///   were sum-of-matched-should-clauses scoring with each clause scoring
///   1.0, doc 0 would score 2.0, not 1.0).
///
/// `body` doc 0 matches both `cat` and `dog` (double match); doc 2 matches
/// only `cat` (single match); doc 4 matches only `bird` (single match, not
/// in the term list here so it's absent). All matching docs must score
/// exactly `1.0`, doc 0 included, disproving the summed-should alternative.
#[test]
fn term_in_set_scores_flat_1_0_not_summed_like_a_boolean_should_disjunction() {
    let (fields, doc, id, suffix) = open_segment();
    let doc_in = DocInput::open(&doc, &id, &suffix).expect("open .doc");

    let query = BooleanQuery::new().with_must([Clause::from(TermInSetQuery::new(
        "body",
        [b"cat".to_vec(), b"dog".to_vec()],
    ))]);

    let mut collector = TopDocsCollector::new(10);
    search_boolean_query_scored(
        &fields,
        Some(&doc_in),
        None,
        None,
        None,
        None,
        &query,
        None,
        &mut collector,
    )
    .unwrap();
    let mut hits: Vec<(i32, f32)> = collector
        .top_docs()
        .iter()
        .map(|d| (d.doc_id, d.score))
        .collect();
    hits.sort_by_key(|&(doc_id, _)| doc_id);

    // cat={0,2}, dog={0,1} -> union {0,1,2}; doc 0 is in BOTH terms' postings
    // but must still score 1.0, not 2.0.
    assert_eq!(hits, vec![(0, 1.0), (1, 1.0), (2, 1.0)]);
}
