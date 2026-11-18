#![cfg(test)]
use super::*;

/// `BooleanWeight.count` passes a lone required leaf through a constant
/// score, a boost or a nested boolean; anything else beside it (a second
/// clause, a `SHOULD`, a `MUST_NOT`) is counted by the scorer.
#[test]
fn a_lone_leaf_is_found_through_wrappers_and_nothing_else() {
    use crate::query::{BoostQuery, ConstantScoreQuery, FieldExistsQuery, TermQuery};
    let exists = || Clause::Exists(FieldExistsQuery::new("f"));
    let with = |must: Vec<Clause>, filter: Vec<Clause>| BooleanQuery {
        must,
        filter,
        ..BooleanQuery::default()
    };
    let field = |q: &BooleanQuery| match lone_leaf(q) {
        Some(Clause::Exists(e)) => Some(e.field.clone()),
        Some(Clause::Term(t)) => Some(format!("term {}", t.field)),
        Some(_) => Some("other".to_string()),
        None => None,
    };
    let f = Some("f".to_string());
    assert_eq!(field(&with(vec![exists()], vec![])), f);
    assert_eq!(field(&with(vec![], vec![exists()])), f);
    let wrapped = Clause::Boost(Box::new(BoostQuery::new(
        Clause::ConstantScore(Box::new(ConstantScoreQuery::new(exists(), 1.0))),
        2.0,
    )));
    assert_eq!(field(&with(vec![wrapped], vec![])), f);
    let nested = Clause::Boolean(Box::new(with(vec![], vec![exists()])));
    assert_eq!(field(&with(vec![nested], vec![])), f);
    assert_eq!(field(&with(vec![exists()], vec![exists()])), None);
    let mut should = with(vec![exists()], vec![]);
    should.should.push(exists());
    assert_eq!(field(&should), None);
    let mut not = with(vec![exists()], vec![]);
    not.must_not.push(exists());
    assert_eq!(field(&not), None);
    // One required `SHOULD` and none to match it: nothing matches.
    let mut impossible = with(vec![exists()], vec![]);
    impossible.minimum_should_match = 1;
    assert_eq!(field(&impossible), None);
    // A fuzzy expansion's one term: boosted, scoring with its own
    // statistics.
    let term = TermQuery::new("f", "t").with_doc_freq(7);
    let boosted = Clause::Boost(Box::new(BoostQuery::new(Clause::Term(term), 0.8)));
    assert_eq!(
        field(&with(vec![boosted], vec![])),
        Some("term f".to_string())
    );
    assert_eq!(
        field(&with(
            vec![Clause::MatchAllDocs(crate::query::MatchAllDocsQuery::new(
                3
            ))],
            vec![]
        )),
        Some("other".to_string())
    );
}

#[test]
fn cutoff_doc_set_keeps_exactly_the_lowest_n_documents() {
    let mut set = CutoffDocSet::default();
    let docs = [5, 900, 64, 63, 1000, 127, 128, 3, 70000, 65];
    for &d in &docs {
        assert!(set.insert(d));
    }
    assert!(!set.insert(64), "a repeat is not a new document");
    assert!(!set.insert(-1), "a negative id is never inserted");
    assert_eq!(set.len(), docs.len());

    let mut sorted = docs.to_vec();
    sorted.sort_unstable();
    for n in [1usize, 3, 4, 7, 10] {
        let mut s = CutoffDocSet::default();
        for &d in &docs {
            s.insert(d);
        }
        let cutoff = s.shrink_to(n);
        let mut kept = Vec::new();
        s.for_each(|d| kept.push(d));
        assert_eq!(kept, sorted[..n], "n={n}");
        assert_eq!(cutoff, sorted[n - 1] + 1, "n={n}");
        assert_eq!(s.len(), n);
    }
    // Asking for more than the set holds leaves it alone.
    assert_eq!(set.shrink_to(100), lucene_codecs::postings::NO_MORE_DOCS);
}

// Reuses the same checked-in real-Lucene fixture
// (`fixtures/data/blocktree_index/`) the differential test in
// `crates/lucene-search/tests/term_query_fixtures.rs` opens -- that test
// is the real-Lucene proof; these unit tests instead focus on
// `search_term_query`'s own branches (missing field, missing term,
// singleton no-`.doc`-needed path, the `.doc`-required error path,
// `live_docs` filtering) using the same real segment data, rather than
// hand-building a synthetic one (see the `test-coverage` skill: a real
// fixture beats a hand-built one wherever one is already available).
fn open_fixture() -> (BlockTreeFields, Option<DocInputOwned>) {
    let dir = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/blocktree_index/"
    );
    let manifest = std::fs::read_to_string(format!("{dir}manifest.properties"))
        .expect("run fixtures generator first (GenBlockTree)");
    let get = |key: &str| -> String {
        manifest
            .lines()
            .find_map(|l| l.strip_prefix(&format!("{key}=")))
            .unwrap_or_else(|| panic!("manifest key {key} missing"))
            .to_string()
    };
    let id_hex = get("id_hex");
    let mut id = [0u8; 16];
    for (i, slot) in id.iter_mut().enumerate() {
        *slot = u8::from_str_radix(&id_hex[i * 2..i * 2 + 2], 16).unwrap();
    }
    let suffix = get("segment_suffix");
    let max_doc: i32 = get("max_doc").parse().unwrap();

    let read_raw = |name: &str| -> Vec<u8> {
        std::fs::read(format!("{dir}{name}.raw")).unwrap_or_else(|_| panic!("missing {name}"))
    };
    let fnm = read_raw(&get("fnm_file_name"));
    let field_infos = lucene_codecs::field_infos::parse(&fnm, &id, "").expect("parse .fnm");
    let tim = read_raw(&get("tim_file_name"));
    let tip = read_raw(&get("tip_file_name"));
    let tmd = read_raw(&get("tmd_file_name"));
    let fields = blocktree::open(&tim, &tip, &tmd, &field_infos, &id, &suffix, max_doc)
        .expect("open blocktree");
    let doc = read_raw(&get("doc_file_name"));
    let pos = read_raw(&get("pos_file_name"));
    let pay = read_raw(&get("pay_file_name"));
    (
        fields,
        Some(DocInputOwned {
            doc,
            pos,
            pay,
            id,
            suffix,
        }),
    )
}

// Owns the `.doc`/`.pos`/`.pay` bytes + segment id/suffix so `DocInput`/
// `PosInput`/`PayInput` can be constructed with a lifetime tied to a local
// variable in each test (each of these borrows its buffer).
struct DocInputOwned {
    doc: Vec<u8>,
    pos: Vec<u8>,
    pay: Vec<u8>,
    id: [u8; 16],
    suffix: String,
}

impl DocInputOwned {
    fn open(&self) -> DocInput<'_> {
        DocInput::open(&self.doc, &self.id, &self.suffix).expect("open .doc")
    }

    fn open_pos(&self) -> PosInput<'_> {
        PosInput::open(&self.pos, &self.id, &self.suffix).expect("open .pos")
    }

    fn open_pay(&self) -> PayInput<'_> {
        PayInput::open(&self.pay, &self.id, &self.suffix).expect("open .pay")
    }
}

#[test]
fn missing_field_yields_no_matches() {
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let mut c = CountCollector::default();
    search_term_query(
        &fields,
        doc_in.as_ref(),
        None,
        &TermQuery::new("nonexistent", "x"),
        &mut c,
    )
    .unwrap();
    assert_eq!(c.count, 0);
}

#[test]
fn missing_term_yields_no_matches() {
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let mut c = VecCollector::default();
    search_term_query(
        &fields,
        doc_in.as_ref(),
        None,
        &TermQuery::new("body", "zzz-missing"),
        &mut c,
    )
    .unwrap();
    assert!(c.docs.is_empty());
}

#[test]
fn multi_doc_term_collects_expected_docs_in_order() {
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let mut c = VecCollector::default();
    search_term_query(
        &fields,
        doc_in.as_ref(),
        None,
        &TermQuery::new("body", "cat"),
        &mut c,
    )
    .unwrap();
    assert_eq!(c.docs, vec![0, 2]);
}

#[test]
fn singleton_term_needs_no_doc_input() {
    let (fields, _doc) = open_fixture();
    let mut c = VecCollector::default();
    search_term_query(&fields, None, None, &TermQuery::new("id", "id2"), &mut c).unwrap();
    assert_eq!(c.docs, vec![2]);
}

#[test]
fn multi_doc_term_without_doc_input_is_an_error() {
    let (fields, _doc) = open_fixture();
    let mut c = CountCollector::default();
    let err =
        search_term_query(&fields, None, None, &TermQuery::new("body", "cat"), &mut c).unwrap_err();
    assert!(matches!(err, Error::BlockTree(_)));
}

#[test]
fn live_docs_filters_out_deleted_docs() {
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let max_doc: i32 = {
        let dir = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/blocktree_index/"
        );
        let manifest = std::fs::read_to_string(format!("{dir}manifest.properties")).unwrap();
        manifest
            .lines()
            .find_map(|l| l.strip_prefix("max_doc="))
            .unwrap()
            .parse()
            .unwrap()
    };
    let mut live_docs = FixedBitSet::new(max_doc as usize);
    for i in 0..max_doc {
        live_docs.set(i as usize);
    }
    // "cat" matches docs 0 and 2 (see manifest); mark doc 0 deleted.
    live_docs.clear(0);

    let mut c = VecCollector::default();
    search_term_query(
        &fields,
        doc_in.as_ref(),
        Some(&live_docs),
        &TermQuery::new("body", "cat"),
        &mut c,
    )
    .unwrap();
    assert_eq!(c.docs, vec![2]);
}

// `should_match_counts` unit tests: pure counting logic, no fixture needed.

#[test]
fn should_match_counts_tallies_doc_occurrences_across_clauses() {
    let counts = should_match_counts(&[vec![1, 2, 3], vec![2, 3], vec![3]]);
    assert_eq!(counts.get(&1), Some(&1));
    assert_eq!(counts.get(&2), Some(&2));
    assert_eq!(counts.get(&3), Some(&3));
    assert_eq!(counts.get(&4), None);
}

#[test]
fn should_match_counts_no_clauses_is_empty() {
    assert!(should_match_counts(&[]).is_empty());
}

#[test]
fn should_match_counts_disjoint_clauses_each_count_one() {
    let counts = should_match_counts(&[vec![1], vec![2], vec![3]]);
    assert_eq!(counts.get(&1), Some(&1));
    assert_eq!(counts.get(&2), Some(&1));
    assert_eq!(counts.get(&3), Some(&1));
}

// Boolean-query tests all reuse `body`'s known real-Lucene doc sets from
// `manifest.properties` (see `term_query_fixtures.rs`'s module doc for how these
// were captured): cat={0,2}, dog={0,1}, bird={1,4}.

#[test]
fn boolean_must_conjunction_matches_only_docs_in_every_clause() {
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let mut c = VecCollector::default();
    let q = BooleanQuery::new()
        .with_must([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")]);
    search_boolean_query(&fields, doc_in.as_ref(), None, None, None, None, &q, &mut c).unwrap();
    assert_eq!(c.docs, vec![0]);
}

/// The lazy leapfrog path must be indistinguishable from the general
/// materializing path, since it is an execution change and not a semantic
/// one. Same query, both routes, same `(doc, score)` output.
#[test]
fn lazy_conjunction_matches_the_general_path_exactly() {
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let q = BooleanQuery::new()
        .with_must([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")]);

    let mut lazy = collector::TopDocsCollector::new(10);
    search_boolean_query_scored(
        &fields,
        doc_in.as_ref(),
        None,
        None,
        None,
        None,
        &q,
        None,
        &mut lazy,
    )
    .unwrap();

    // Force the general path by giving the query a shape the gate rejects:
    // a `must_not` that excludes nothing changes the route, not the result.
    let general_q = BooleanQuery::new()
        .with_must([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")])
        .with_must_not([TermQuery::new("body", "no_such_term_anywhere")]);
    let mut general = collector::TopDocsCollector::new(10);
    search_boolean_query_scored(
        &fields,
        doc_in.as_ref(),
        None,
        None,
        None,
        None,
        &general_q,
        None,
        &mut general,
    )
    .unwrap();

    assert_eq!(lazy.top_docs().len(), general.top_docs().len());
    for (a, b) in lazy.top_docs().iter().zip(general.top_docs()) {
        assert_eq!(a.doc_id, b.doc_id);
        assert!(
            (a.score - b.score).abs() < 1e-6,
            "{} vs {}",
            a.score,
            b.score
        );
    }
}

/// A `docFreq == 1` term is pulsed into the term dictionary and has no
/// `.doc` bytes, so `lazy_postings` cannot open a cursor for it. Both lazy
/// paths must hand such a query to the general path rather than erroring.
///
/// Regression: an unguarded lazy conjunction would have propagated `Unsupported("docFreq <= 1")` to the caller. The
/// disjunction's version of the same bug is what the existing maxscore
/// singleton test caught; nothing covered the conjunction, hence this.
#[test]
fn lazy_paths_fall_back_for_pulsed_singleton_terms() {
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());

    // "id"/"id2" has docFreq == 1 in this fixture.
    for q in [
        BooleanQuery::new().with_must([TermQuery::new("body", "cat"), TermQuery::new("id", "id2")]),
        BooleanQuery::new()
            .with_should([TermQuery::new("body", "cat"), TermQuery::new("id", "id2")]),
    ] {
        let mut c = collector::TopDocsCollector::new(10);
        search_boolean_query_scored(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            None,
            &q,
            None,
            &mut c,
        )
        .expect("pulsed singleton must not error");
    }
}

/// A term absent from the dictionary makes the whole conjunction empty --
/// the lazy path must return that answer itself rather than falling through
/// and letting the general path recompute it.
#[test]
fn lazy_conjunction_with_absent_term_matches_nothing() {
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let mut c = collector::TopDocsCollector::new(10);
    let q = BooleanQuery::new().with_must([
        TermQuery::new("body", "cat"),
        TermQuery::new("body", "definitely_not_indexed"),
    ]);
    search_boolean_query_scored(
        &fields,
        doc_in.as_ref(),
        None,
        None,
        None,
        None,
        &q,
        None,
        &mut c,
    )
    .unwrap();
    assert!(c.top_docs().is_empty());
}

/// An absent *field* is the same story one level up.
#[test]
fn lazy_conjunction_with_absent_field_matches_nothing() {
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let mut c = collector::TopDocsCollector::new(10);
    let q = BooleanQuery::new().with_must([
        TermQuery::new("body", "cat"),
        TermQuery::new("no_such_field", "cat"),
    ]);
    search_boolean_query_scored(
        &fields,
        doc_in.as_ref(),
        None,
        None,
        None,
        None,
        &q,
        None,
        &mut c,
    )
    .unwrap();
    assert!(c.top_docs().is_empty());
}

#[test]
fn points_range_clause_without_points_input_is_missing_points_input_in_match() {
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let mut c = VecCollector::default();
    let q = BooleanQuery::new().with_must([Clause::PointsRange(
        crate::query::PointsRangeQuery::new("body", 0, 100),
    )]);
    let err = search_boolean_query(&fields, doc_in.as_ref(), None, None, None, None, &q, &mut c)
        .unwrap_err();
    assert!(matches!(err, Error::MissingPointsInput(field) if field == "body"));
}

#[test]
fn points_range_clause_without_points_input_is_missing_points_input_in_scored_match() {
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let mut c = collector::TopDocsCollector::new(10);
    let q = BooleanQuery::new().with_must([Clause::PointsRange(
        crate::query::PointsRangeQuery::new("body", 0, 100),
    )]);
    let err = search_boolean_query_scored(
        &fields,
        doc_in.as_ref(),
        None,
        None,
        None,
        None,
        &q,
        None,
        &mut c,
    )
    .unwrap_err();
    assert!(matches!(err, Error::MissingPointsInput(field) if field == "body"));
}

/// `field_infos` for a single synthetic `LongPoint`-shaped numeric field
/// named "price" -- same minimal-construction template
/// `multi_segment.rs`'s `numeric_field_infos` test helper uses for a
/// doc-values field, adapted for `point_dimension_count`/
/// `point_index_dimension_count`/`point_num_bytes` instead.
fn price_field_infos(field_number: i32) -> lucene_codecs::field_infos::FieldInfos {
    use lucene_codecs::field_infos::{
        DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions, VectorEncoding,
        VectorSimilarityFunction,
    };
    lucene_codecs::field_infos::FieldInfos {
        fields: vec![FieldInfo {
            name: "price".to_string(),
            number: field_number,
            store_term_vectors: false,
            omit_norms: false,
            store_payloads: false,
            soft_deletes_field: false,
            parent_field: false,
            index_options: IndexOptions::None,
            doc_values_type: DocValuesType::None,
            doc_values_skip_index_type: DocValuesSkipIndexType::None,
            doc_values_gen: -1,
            attributes: vec![],
            point_dimension_count: 1,
            point_index_dimension_count: 1,
            point_num_bytes: 8,
            vector_dimension: 0,
            vector_encoding: VectorEncoding::Float32,
            vector_similarity_function: VectorSimilarityFunction::Euclidean,
        }],
    }
}

/// End-to-end proof: a real query *string* (not a
/// hand-built [`Clause::PointsRange`]) parsed via
/// [`query_parser::parse_query`], resolved through
/// [`search_boolean_query`] against a real `PointsInput` -- the full
/// query-string-to-results path, matching this module's own doc comment on
/// [`Error::MissingPointsInput`].
#[test]
fn points_range_query_string_matches_real_segment_end_to_end() {
    use lucene_codecs::points::{self, WritePointsField};
    let segment_id = [7u8; lucene_store::codec_util::ID_LENGTH];
    // doc 0 -> 10, doc 1 -> 20, doc 2 -> 30, doc 3 -> 40, doc 4 -> 50.
    let points_field = WritePointsField {
        field_number: 3,
        num_dims: 1,
        num_index_dims: 1,
        bytes_per_dim: 8,
        points: (0..5)
            .map(|doc_id| {
                (
                    doc_id,
                    points_query::pack_i64((doc_id as i64 + 1) * 10).to_vec(),
                )
            })
            .collect(),
    };
    let (kdm, kdi, kdd) = points::write(&[points_field], 512, &segment_id, "").unwrap();
    let reader = points::open(&kdm, &kdi, &kdd, &segment_id, "").unwrap();
    let field_infos = price_field_infos(3);
    let points_input = PointsInput {
        reader,
        field_infos: &field_infos,
    };

    let fields = BlockTreeFields::empty();

    // Inclusive `[20 TO 40]` -- docs with values 20, 30, 40 (doc 1..=3).
    let clause = query_parser::parse_query("price:[20 TO 40]", None).unwrap();
    let q = BooleanQuery::new().with_must([clause]);
    let mut c = VecCollector::default();
    search_boolean_query(
        &fields,
        None,
        None,
        None,
        None,
        Some(&points_input),
        &q,
        &mut c,
    )
    .unwrap();
    assert_eq!(c.docs, vec![1, 2, 3]);

    // Exclusive `{20 TO 40}` -- only doc 2 (value 30)
    // is strictly between 20 and 40.
    let clause = query_parser::parse_query("price:{20 TO 40}", None).unwrap();
    let q = BooleanQuery::new().with_must([clause]);
    let mut c = VecCollector::default();
    search_boolean_query(
        &fields,
        None,
        None,
        None,
        None,
        Some(&points_input),
        &q,
        &mut c,
    )
    .unwrap();
    assert_eq!(c.docs, vec![2]);

    // Unknown field name: no matches, not an error (same "missing field
    // means no matches" convention every other clause follows).
    let clause = query_parser::parse_query("nope:[0 TO 100]", None).unwrap();
    let q = BooleanQuery::new().with_must([clause]);
    let mut c = VecCollector::default();
    search_boolean_query(
        &fields,
        None,
        None,
        None,
        None,
        Some(&points_input),
        &q,
        &mut c,
    )
    .unwrap();
    assert!(c.docs.is_empty());

    // Scored sibling: same matched set, constant `1.0` score per doc
    // (real `PointRangeQuery` is `ConstantScoreQuery`-shaped).
    let clause = query_parser::parse_query("price:[20 TO 40]", None).unwrap();
    let q = BooleanQuery::new().with_must([clause]);
    let mut top = collector::TopDocsCollector::new(10);
    search_boolean_query_scored(
        &fields,
        None,
        None,
        None,
        None,
        Some(&points_input),
        &q,
        None,
        &mut top,
    )
    .unwrap();
    let mut docs: Vec<i32> = top.top_docs().iter().map(|d| d.doc_id).collect();
    docs.sort_unstable();
    assert_eq!(docs, vec![1, 2, 3]);
    assert!(top.top_docs().iter().all(|d| d.score == 1.0));
}

#[test]
fn boolean_should_disjunction_matches_union_of_clauses() {
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let mut c = VecCollector::default();
    let q = BooleanQuery::new().with_should([
        TermQuery::new("body", "cat"),
        TermQuery::new("body", "bird"),
    ]);
    search_boolean_query(&fields, doc_in.as_ref(), None, None, None, None, &q, &mut c).unwrap();
    assert_eq!(c.docs, vec![0, 1, 2, 4]);
}

#[test]
fn boolean_must_not_excludes_matching_docs() {
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let mut c = VecCollector::default();
    let q = BooleanQuery::new()
        .with_must([TermQuery::new("body", "cat")])
        .with_must_not([TermQuery::new("body", "dog")]);
    search_boolean_query(&fields, doc_in.as_ref(), None, None, None, None, &q, &mut c).unwrap();
    assert_eq!(c.docs, vec![2]);
}

#[test]
fn boolean_pure_must_not_matches_nothing() {
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let mut c = VecCollector::default();
    let q = BooleanQuery::new().with_must_not([TermQuery::new("body", "dog")]);
    search_boolean_query(&fields, doc_in.as_ref(), None, None, None, None, &q, &mut c).unwrap();
    assert!(c.docs.is_empty());
}

#[test]
fn boolean_empty_query_matches_nothing() {
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let mut c = VecCollector::default();
    let q = BooleanQuery::new();
    search_boolean_query(&fields, doc_in.as_ref(), None, None, None, None, &q, &mut c).unwrap();
    assert!(c.docs.is_empty());
}

// ---- `Occur.FILTER` ---------------------------------------------------
//
// Matching-side unit tests. The scoring side is pinned bit-for-bit against
// real `IndexSearcher` in `tests/bm25_scoring_fixtures.rs`; these cover the
// branches of `matched_boolean_docs` and the conjunction bulk scorer that a filter
// clause reaches, using the same fixture segment
// (cat={0,2}, dog={0,1}, bird={1,4}).

/// Collects every `(doc, score)` the scorer emits, in emission order --
/// the scored analogue of [`VecCollector`], kept local to the tests because
/// production callers all want a top-`n` queue instead.
#[derive(Default)]
struct ScoreVecCollector {
    hits: Vec<(i32, f32)>,
}

impl collector::ScoringCollector for ScoreVecCollector {
    fn collect(&mut self, doc_id: i32, score: f32) {
        self.hits.push((doc_id, score));
    }
}

/// Helper: matched doc IDs for `query`, unscored.
fn boolean_docs(query: &BooleanQuery) -> Vec<i32> {
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let mut c = VecCollector::default();
    search_boolean_query(
        &fields,
        doc_in.as_ref(),
        None,
        None,
        None,
        None,
        query,
        &mut c,
    )
    .unwrap();
    c.docs
}

/// Helper: `(doc, score)` pairs for `query`, ascending by doc, no norms.
fn boolean_scores(query: &BooleanQuery) -> Vec<(i32, f32)> {
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let mut c = ScoreVecCollector::default();
    search_boolean_query_scored(
        &fields,
        doc_in.as_ref(),
        None,
        None,
        None,
        None,
        query,
        None,
        &mut c,
    )
    .unwrap();
    c.hits
}

#[test]
fn filter_only_query_matches_the_conjunction_it_describes() {
    // Java's pure-negative rewrite is `clauses.size() ==
    // clauseSets.get(MUST_NOT).size()`, which a FILTER clause fails -- so a
    // filter-only query is a *positive* query and matches, unlike the
    // must_not-only query tested above.
    let q = BooleanQuery::new()
        .with_filter([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")]);
    assert_eq!(boolean_docs(&q), vec![0]);
}

#[test]
fn filter_only_query_scores_every_match_zero() {
    let q = BooleanQuery::new().with_filter([TermQuery::new("body", "cat")]);
    assert_eq!(boolean_scores(&q), vec![(0, 0.0), (2, 0.0)]);
}

#[test]
fn a_filter_clause_selects_exactly_what_the_same_must_clause_would() {
    let must = BooleanQuery::new()
        .with_must([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")]);
    let filter = BooleanQuery::new()
        .with_filter([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")]);
    let mixed = BooleanQuery::new()
        .with_must([TermQuery::new("body", "cat")])
        .with_filter([TermQuery::new("body", "dog")]);
    assert_eq!(boolean_docs(&must), vec![0]);
    assert_eq!(boolean_docs(&filter), boolean_docs(&must));
    assert_eq!(boolean_docs(&mixed), boolean_docs(&must));
}

#[test]
fn a_filter_clause_contributes_exactly_zero_to_the_score() {
    // The difference between `+cat +dog` and `+cat #dog` must be exactly the
    // `dog` clause's own score, with the `cat` clause's contribution
    // bit-identical between the two -- which is what "does not perturb the
    // summation order" means when `f32` addition is not associative.
    let cat_only = boolean_scores(&BooleanQuery::new().with_must([TermQuery::new("body", "cat")]));
    let cat_and_dog = boolean_scores(
        &BooleanQuery::new()
            .with_must([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")]),
    );
    let cat_filter_dog = boolean_scores(
        &BooleanQuery::new()
            .with_must([TermQuery::new("body", "cat")])
            .with_filter([TermQuery::new("body", "dog")]),
    );
    assert_eq!(cat_filter_dog.len(), 1);
    assert_eq!(cat_filter_dog[0].0, 0);
    // Same matched set as the all-MUST form...
    assert_eq!(
        cat_and_dog.iter().map(|h| h.0).collect::<Vec<_>>(),
        cat_filter_dog.iter().map(|h| h.0).collect::<Vec<_>>()
    );
    // ... and the score is bit-for-bit the `cat` clause's alone.
    let cat_on_doc0 = cat_only
        .iter()
        .find(|h| h.0 == 0)
        .expect("cat matches doc 0")
        .1;
    assert_eq!(
        cat_filter_dog[0].1.to_bits(),
        cat_on_doc0.to_bits(),
        "filter contributed {} to the score",
        cat_filter_dog[0].1 - cat_on_doc0
    );
    assert!(cat_and_dog[0].1 > cat_filter_dog[0].1);
}

#[test]
fn filter_clauses_do_not_count_toward_minimum_should_match() {
    // filter=[cat]={0,2}; should=[dog,bird] with minimum_should_match=1.
    // Doc 2 matches the filter and neither optional clause. If a filter
    // counted toward the threshold it would survive; Java increments
    // `shouldMatchCount` only for `Occur.SHOULD`, so it must not.
    let q = BooleanQuery::new()
        .with_filter([TermQuery::new("body", "cat")])
        .with_should([
            TermQuery::new("body", "dog"),
            TermQuery::new("body", "bird"),
        ])
        .with_minimum_should_match(1);
    assert_eq!(boolean_docs(&q), vec![0]);

    // With the threshold off, doc 2 comes back -- proving the exclusion
    // above was the threshold and not the filter.
    let q = BooleanQuery::new()
        .with_filter([TermQuery::new("body", "cat")])
        .with_should([
            TermQuery::new("body", "dog"),
            TermQuery::new("body", "bird"),
        ]);
    assert_eq!(boolean_docs(&q), vec![0, 2]);
}

#[test]
fn a_filter_only_query_with_a_positive_minimum_should_match_matches_nothing() {
    // No `should` clauses at all, so no document can ever reach a positive
    // threshold -- the same rule as the `must`-only case, and the reason
    // `BooleanQuery::rewrite`'s single-clause unwrap excludes it.
    let q = BooleanQuery::new()
        .with_filter([TermQuery::new("body", "cat")])
        .with_minimum_should_match(1);
    assert!(boolean_docs(&q).is_empty());
}

#[test]
fn filter_combines_with_must_not() {
    // filter=[cat]={0,2}, must_not=[dog]={0,1} -> {2}.
    let q = BooleanQuery::new()
        .with_filter([TermQuery::new("body", "cat")])
        .with_must_not([TermQuery::new("body", "dog")]);
    assert_eq!(boolean_docs(&q), vec![2]);
}

#[test]
fn a_nested_boolean_query_works_as_a_filter_clause_and_still_scores_zero() {
    // filter=[nested], nested = should=[dog, bird] -> {0,1,4}; must=[cat]
    // -> {0,2}. Intersection {0}, scored as `cat` alone.
    let nested = BooleanQuery::new().with_should([
        TermQuery::new("body", "dog"),
        TermQuery::new("body", "bird"),
    ]);
    let q = BooleanQuery::new()
        .with_must([TermQuery::new("body", "cat")])
        .with_filter([Clause::from(nested)]);
    assert_eq!(boolean_docs(&q), vec![0]);

    let cat_only = boolean_scores(&BooleanQuery::new().with_must([TermQuery::new("body", "cat")]));
    let scored = boolean_scores(&q);
    assert_eq!(scored.len(), 1);
    assert_eq!(
        scored[0].1.to_bits(),
        cat_only
            .iter()
            .find(|h| h.0 == 0)
            .expect("cat matches doc 0")
            .1
            .to_bits(),
        "a nested boolean filter must contribute nothing either"
    );
}

#[test]
fn a_filter_clause_that_matches_nothing_empties_the_conjunction() {
    let q = BooleanQuery::new()
        .with_must([TermQuery::new("body", "cat")])
        .with_filter([TermQuery::new("body", "no_such_term_anywhere")]);
    assert!(boolean_docs(&q).is_empty());
    assert!(boolean_scores(&q).is_empty());
}

/// Which bulk scorer `exec::bulk_boolean` picks for `q`, for a collector
/// that keeps every hit.
fn bulk_kind(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    q: &BooleanQuery,
) -> Option<&'static str> {
    let ctx = exec::LeafContext {
        fields,
        doc_in,
        pos_in: None,
        pay_in: None,
        live_docs: None,
        points: None,
        norms: None,
        global: None,
        max_doc: None,
        cache: None,
        reader: None,
        similarity: None,
    };
    exec::bulk_boolean(&ctx, q, 1.0, exec::Mode::Complete)
        .unwrap()
        .map(|b| b.kind())
}

#[test]
fn a_filter_only_conjunction_takes_the_lazy_leapfrog_path_and_scores_zero() {
    // Not taking it cost 129.5ms against 8.8ms for the same conjunction
    // written with `MUST` clauses, because the general path materializes
    // each clause's whole doc list first -- see `benches/filter_vs_must.rs`.
    // The leapfrog takes it, and so does block-max pruning --
    // see `a_filter_only_conjunction_prunes_against_a_full_top_n_queue`.
    let q = BooleanQuery::new()
        .with_filter([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")]);
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    assert_eq!(
        bulk_kind(&fields, doc_in.as_ref(), &q),
        Some("conjunction"),
        "a filter-only conjunction must run on BlockMaxConjunctionBulkScorer"
    );
    assert_eq!(boolean_scores(&q), vec![(0, 0.0)]);
}

#[test]
fn a_filter_only_conjunction_prunes_against_a_full_top_n_queue() {
    // Every document scores 0, so the block-max bound is 0 and the
    // threshold a full queue publishes is 0 -- `0 <= 0` authorizes the
    // skip. This is not "pruning on a tie":
    // `TopScoreDocCollector` publishes `Math.nextUp(0f)` and Lucene's own
    // `bound < minCompetitiveScore` test skips exactly the same blocks.
    // The answer is unchanged because documents arrive ascending and
    // `HitQueue` gives a score tie to the lower doc id, so nothing skipped
    // could have entered the queue.
    //
    // On `big` (`everywhere` in 300 documents), not on `body`: the
    // pruning branch needs a decoded block to bound, and `body`'s two-doc
    // postings live in the vint tail, where `current_block_last_doc_id()`
    // is still `-1`.
    let q = BooleanQuery::new().with_filter([
        TermQuery::new("big", "everywhere"),
        TermQuery::new("big", "everywhere"),
    ]);
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let run = |threshold: u64| {
        let mut top = collector::TopDocsCollector::with_total_hits_threshold(20, threshold);
        search_boolean_query_scored(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            None,
            &q,
            None,
            &mut top,
        )
        .unwrap();
        (top.top_docs().to_vec(), top.total_hits())
    };
    // `u64::MAX` is `ScoreMode::COMPLETE`, i.e. pruning forbidden.
    let (exhaustive_hits, exhaustive_total) = run(u64::MAX);
    assert_eq!(exhaustive_total.value, 300);
    assert_eq!(
        exhaustive_total.relation,
        collector::TotalHitsRelation::EqualTo
    );
    // With pruning allowed most of the segment is skipped -- and the kept
    // hits are bit-identical, which is the property that makes it legal.
    let (pruned_hits, pruned_total) = run(0);
    assert!(
        pruned_total.value < 300,
        "documents must have been skipped, saw {}",
        pruned_total.value
    );
    assert_eq!(
        pruned_total.relation,
        collector::TotalHitsRelation::GreaterThanOrEqualTo,
        "a skipped document makes the count a lower bound"
    );
    assert_eq!(pruned_hits, exhaustive_hits);
    // `big`'s postings start at doc 5, every score is 0, and a score tie
    // goes to the lower doc id: the answer is the 20 lowest doc ids.
    assert_eq!(
        pruned_hits,
        (5..25)
            .map(|doc_id| collector::ScoreDoc { doc_id, score: 0.0 })
            .collect::<Vec<_>>()
    );
}

#[test]
fn the_lazy_conjunction_path_handles_a_filter_leg() {
    // must=[cat], filter=[dog]: both are `Clause::Term`, so this *is* the
    // lazy leapfrog's shape, with one scoring leg and one non-scoring one.
    let q = BooleanQuery::new()
        .with_must([TermQuery::new("body", "cat")])
        .with_filter([TermQuery::new("body", "dog")]);
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    assert_eq!(
        bulk_kind(&fields, doc_in.as_ref(), &q),
        Some("conjunction"),
        "a conjunction with a filter leg must run on BlockMaxConjunctionBulkScorer"
    );
    let hits = boolean_scores(&q);
    assert_eq!(hits.iter().map(|h| h.0).collect::<Vec<_>>(), vec![0]);
    // Same score as `+body:cat` alone: the filter leg contributes nothing.
    let cat = boolean_scores(&BooleanQuery::new().with_must([TermQuery::new("body", "cat")]));
    assert_eq!(
        hits[0].1.to_bits(),
        cat.iter()
            .find(|h| h.0 == 0)
            .expect("cat matches doc 0")
            .1
            .to_bits()
    );
}

#[test]
fn the_lazy_disjunction_and_maxscore_paths_decline_a_query_with_filters() {
    // Both are pure-SHOULD fast paths; a filter clause is a required
    // clause, so neither shape applies and both must fall back rather than
    // silently ignore it.
    let q = BooleanQuery::new()
        .with_filter([TermQuery::new("body", "cat")])
        .with_should([
            TermQuery::new("body", "dog"),
            TermQuery::new("body", "bird"),
        ]);
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    // FILTER + SHOULD with `minimum_should_match` 0 is `ReqOptSumScorer`'s
    // shape, which runs on its batch bulk scorer here.
    assert_eq!(bulk_kind(&fields, doc_in.as_ref(), &q), Some("req_opt"));

    // The MAXSCORE entry point falls back to the general path, which does
    // honour the filter: `cat` = {0,2}, so doc 1 and doc 4 are excluded
    // even though they match `dog`/`bird`.
    let mut top = collector::TopDocsCollector::new(10);
    search_boolean_query_scored_maxscore(
        &fields,
        doc_in.as_ref(),
        None,
        None,
        None,
        None,
        &q,
        None,
        &mut top,
    )
    .unwrap();
    let mut docs: Vec<i32> = top.top_docs().iter().map(|h| h.doc_id).collect();
    docs.sort_unstable();
    assert_eq!(docs, vec![0, 2]);
}

// ---- single-scoring-clause fast path + constant-score streaming --------
//
// Covers `search_boolean_query_scored`'s
// one-clause shortcut and the `stream_constant_score_clause` /
// `expanded_terms` machinery it is the only caller of, which the
// wildcard-family fixture suites (calling `search_prefix_query_scored` and
// friends directly) never reach. Fixture fields used below: `big` holds one term
// `everywhere` in 300 documents, `many` holds 400 terms `term0000`..
// `term0399` at one document each.

#[test]
fn a_single_prefix_clause_streams_through_the_constant_score_path() {
    // 1 expanded term at docFreq 300, so
    // `total_doc_freq >= terms * BLOCK_SIZE` and the streaming union runs
    // instead of materializing the postings. Every match scores a flat 1.0,
    // so a full top-`n` queue's bottom is already 1.0 and the stream stops
    // early -- `pruning_threshold().is_some_and(|s| s >= 1.0)`.
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let q = BooleanQuery::new().with_must([Clause::Prefix(PrefixQuery::new("big", "every"))]);
    let mut top = collector::TopDocsCollector::new(3);
    search_boolean_query_scored(
        &fields,
        doc_in.as_ref(),
        None,
        None,
        None,
        None,
        &q,
        None,
        &mut top,
    )
    .unwrap();
    assert_eq!(top.top_docs().len(), 3);
    assert!(top.top_docs().iter().all(|h| h.score == 1.0));
    // Stopped early: the stream returned before visiting all 300.
    assert!(
        top.total_hits().value < 300,
        "the early exit did not fire: {} hits visited",
        top.total_hits().value
    );
}

#[test]
fn a_single_wildcard_or_regexp_clause_streams_the_same_way() {
    // The other two `expanded_terms` arms. `e*e` and `ever.*` both select
    // the single term `everywhere`.
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    for clause in [
        Clause::Wildcard(WildcardQuery::new("big", "e*e")),
        Clause::Regexp(RegexpQuery::new("big", "ever.*")),
    ] {
        let q = BooleanQuery::new().with_must([clause.clone()]);
        let mut top = collector::TopDocsCollector::new(2);
        search_boolean_query_scored(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            None,
            &q,
            None,
            &mut top,
        )
        .unwrap();
        assert_eq!(top.top_docs().len(), 2, "{clause:?}");
        assert!(top.top_docs().iter().all(|h| h.score == 1.0));
    }
}

#[test]
fn the_streaming_path_declines_when_the_expansion_is_wider_than_it_is_deep() {
    // 400 terms at docFreq 1 each: `total_doc_freq (400) < terms (400) *
    // BLOCK_SIZE (128)`, so opening a lazy cursor per term would decode far
    // more than the whole union is worth. The clause falls back to
    // `resolve_clause_docs` and is collected at a flat 1.0 -- the other half
    // of the fast path's wildcard arm.
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let q = BooleanQuery::new().with_must([Clause::Prefix(PrefixQuery::new("many", "term00"))]);
    let mut all = ScoreVecCollector::default();
    search_boolean_query_scored(
        &fields,
        doc_in.as_ref(),
        None,
        None,
        None,
        None,
        &q,
        None,
        &mut all,
    )
    .unwrap();
    assert_eq!(all.hits.len(), 100, "term0000..term0099");
    assert!(all.hits.iter().all(|h| h.1 == 1.0));
    assert!(
        all.hits.windows(2).all(|w| w[0].0 < w[1].0),
        "the fallback must still emit ascending doc ids"
    );
}

#[test]
fn a_wide_expansion_into_a_top_n_collector_keeps_the_first_documents() {
    // 400 single-document terms into a collector that needs only `n`:
    // more than the small k-way merge takes, so `cutoff_constant_score_union`
    // keeps the lowest `n` documents under a shrinking cutoff -- the same
    // documents, at the same flat 1.0, as collecting every match.
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let q = BooleanQuery::new().with_must([Clause::Prefix(PrefixQuery::new("many", "term0"))]);
    let mut all = ScoreVecCollector::default();
    search_boolean_query_scored(
        &fields,
        doc_in.as_ref(),
        None,
        None,
        None,
        None,
        &q,
        None,
        &mut all,
    )
    .unwrap();
    let mut every: Vec<i32> = all.hits.iter().map(|h| h.0).collect();
    every.sort_unstable();
    assert_eq!(every.len(), 400, "term0000..term0399");
    for n in [1, 5, 40] {
        let mut top = collector::TopDocsCollector::new(n);
        search_boolean_query_scored(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            None,
            &q,
            None,
            &mut top,
        )
        .unwrap();
        let mut docs: Vec<i32> = top.top_docs().iter().map(|h| h.doc_id).collect();
        docs.sort_unstable();
        assert_eq!(docs, every[..n], "top {n}");
        assert!(top.top_docs().iter().all(|h| h.score == 1.0));
    }
}

// ---- `MaxNonCompetitiveBoostAttribute`'s feedback loop --------

/// The loop is a **pruning** channel, not a semantic one: whatever it
/// skips could not have entered the top-`maxExpansions` queue anyway. So
/// the two arms must agree term for term, boost for boost, and on the
/// blended document frequency -- and the pruned arm must actually have
/// dropped its edit budget, or this test proves only that a no-op is a
/// no-op.
///
/// The corpus is the fixture's `many` field: 400 terms `term0000`..
/// `term0399`, all eight codepoints long. Against `term0000` at
/// `maxEdits = 2` that is 1 exact match, 21 terms at distance 1
/// (`boost = 1 - 1/8 = 0.875`) and a large tail at distance 2
/// (`boost = 0.75`), so a queue of 5 fills entirely with 0.875s and
/// `bottomChanged`'s `bottom >= 1 - maxEdits/termLength` (0.875 >= 0.75)
/// holds immediately.
#[test]
fn pruning_never_changes_the_selected_expansion() {
    let (fields, _doc) = open_fixture();
    let field = fields.field("many").expect("the `many` field");
    let query = FuzzyQuery::new("many", b"term0000".to_vec())
        .with_max_edits(2)
        .with_max_expansions(5);

    let (pruned, pruned_df, pruned_edits) =
        bench_only_fuzzy_expansion(field, &query, true).unwrap();
    let (whole, whole_df, whole_edits) = bench_only_fuzzy_expansion(field, &query, false).unwrap();

    assert_eq!(pruned, whole, "pruning changed the selected expansion");
    assert_eq!(pruned_df, whole_df);
    assert_eq!(pruned.len(), 5);
    assert_eq!(
        whole_edits, 2,
        "the unpruned arm must keep the query's own budget"
    );
    assert!(
        pruned_edits < 2,
        "the feedback loop never fired: budget still {pruned_edits}"
    );
    // The queue is all exact/distance-1 terms, so their boosts are the two
    // values `bottomChanged`'s arithmetic is about.
    assert!(pruned.iter().all(|(_, boost, _)| *boost >= 0.875));

    // The same equality over a spread of shapes, so this is a property
    // rather than one case: different query-term lengths (which move
    // `bottomChanged`'s `1 - maxEdits/termLength` threshold), different
    // budgets, different caps, a query term that is not in the dictionary
    // at all, and a non-zero `prefixLength`.
    let mut ever_pruned = false;
    for (term, max_edits, prefix_length, max_expansions) in [
        (&b"term0000"[..], 2u8, 0usize, 1usize),
        (b"term0000", 2, 0, 3),
        (b"term0000", 2, 0, 400),
        (b"term0000", 1, 0, 5),
        (b"term0000", 0, 0, 5),
        (b"term0199", 2, 0, 7),
        (b"term0199", 2, 5, 7),
        (b"zzzzzzzz", 2, 0, 5),
        (b"term", 2, 0, 5),
    ] {
        let query = FuzzyQuery::new("many", term.to_vec())
            .with_max_edits(max_edits)
            .with_prefix_length(prefix_length)
            .with_max_expansions(max_expansions);
        let (pruned, pruned_df, pruned_edits) =
            bench_only_fuzzy_expansion(field, &query, true).unwrap();
        let (whole, whole_df, whole_edits) =
            bench_only_fuzzy_expansion(field, &query, false).unwrap();
        let described = format!("{term:?} ed{max_edits} pfx{prefix_length} exp{max_expansions}");
        assert_eq!(pruned, whole, "{described}: pruning changed the expansion");
        assert_eq!(pruned_df, whole_df, "{described}: blended df changed");
        assert_eq!(whole_edits, max_edits, "{described}");
        assert!(pruned_edits <= max_edits, "{described}: budget widened");
        ever_pruned |= pruned_edits < max_edits;
    }
    assert!(ever_pruned, "no case in the spread ever pruned");
}

/// The other direction: a query whose whole matching set fits inside
/// `maxExpansions` never fills the queue, so `MaxNonCompetitiveBoost` is
/// never published and the budget stays where the query put it. A pruner
/// that lowered it here would drop real hits.
#[test]
fn a_queue_that_never_fills_never_lowers_the_edit_budget() {
    let (fields, _doc) = open_fixture();
    let field = fields.field("body").expect("the `body` field");
    let query = FuzzyQuery::new("body", b"cat".to_vec())
        .with_max_edits(2)
        .with_max_expansions(50);
    let (pruned, _, pruned_edits) = bench_only_fuzzy_expansion(field, &query, true).unwrap();
    let (whole, _, _) = bench_only_fuzzy_expansion(field, &query, false).unwrap();
    assert_eq!(pruned, whole);
    assert!(
        pruned.len() < 50,
        "this corpus must not fill the queue: {} terms",
        pruned.len()
    );
    assert_eq!(pruned_edits, 2);
}

/// `maxExpansions = 0` is a queue that is "full" at size zero. Java never
/// reaches `bottomChanged` with a null bottom term (`TopTermsRewrite`'s
/// `peek()` would be null), so the guard is this port's, and it has to be
/// there: without it the bottom lookup is an unwrap on an empty queue.
#[test]
fn a_zero_expansion_cap_selects_nothing_and_does_not_prune() {
    let (fields, _doc) = open_fixture();
    let field = fields.field("many").expect("the `many` field");
    let query = FuzzyQuery::new("many", b"term0000".to_vec())
        .with_max_edits(2)
        .with_max_expansions(0);
    let (terms, df, edits) = bench_only_fuzzy_expansion(field, &query, true).unwrap();
    assert!(terms.is_empty());
    assert_eq!(df, 0);
    assert_eq!(edits, 2);
}

/// `FuzzyIntersect::set_max_edits` only ever narrows: a walk that has
/// already rejected terms at a tighter budget cannot honestly widen.
#[test]
fn the_edit_budget_can_only_be_narrowed_mid_walk() {
    let (fields, _doc) = open_fixture();
    let field = fields.field("many").expect("the `many` field");
    let pattern = FuzzyMatch::new(b"term0000", 2, 0, true);
    let mut walk = field.fuzzy_intersect(&pattern);
    assert_eq!(walk.max_edits(), 2);
    walk.set_max_edits(1);
    assert_eq!(walk.max_edits(), 1);
    walk.set_max_edits(2);
    assert_eq!(walk.max_edits(), 1, "widening must be ignored");
    walk.set_max_edits(0);
    assert_eq!(walk.max_edits(), 0);
    // At budget 0 only the exact term survives -- asserted as the whole
    // list, not `all(...)`, which an empty walk would satisfy vacuously.
    let got: Vec<Vec<u8>> = walk.map(|r| r.unwrap().0).collect();
    assert_eq!(got, vec![b"term0000".to_vec()]);
}

#[test]
fn a_single_fuzzy_clause_is_scored_not_constant_scored() {
    // Real `FuzzyQuery`'s rewrite is `TopTermsBlendedFreqScoringRewrite`,
    // so it scores; the rest of the multi-term family constant-scores. A
    // single-clause fuzzy query must therefore *not* take the constant-
    // score fast path, and must agree with `clause_scores` -- the arm that
    // calls `fuzzy_doc_scores` -- document for document and score for
    // score.
    //
    // Verified to fail against the unfixed code: with `Clause::Fuzzy` in
    // the fast path's `matches!` guard, every hit comes back at 1.0 and
    // both assertions below trip.
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let clause = Clause::Fuzzy(FuzzyQuery::new("body", "cat").with_max_edits(1));
    let q = BooleanQuery::new().with_must([clause.clone()]);

    let mut all = ScoreVecCollector::default();
    search_boolean_query_scored(
        &fields,
        doc_in.as_ref(),
        None,
        None,
        None,
        None,
        &q,
        None,
        &mut all,
    )
    .unwrap();
    assert!(!all.hits.is_empty());

    let mut expected: Vec<(i32, f32)> = clause_scores(
        &fields,
        doc_in.as_ref(),
        None,
        None,
        None,
        None,
        &clause,
        None,
        None,
    )
    .unwrap()
    .into_iter()
    .collect();
    expected.sort_by_key(|(doc_id, _)| *doc_id);
    let mut got = all.hits.clone();
    got.sort_by_key(|(doc_id, _)| *doc_id);
    assert_eq!(got, expected);

    // The regression this pins: a flat 1.0 per hit. BM25 over a fixture
    // whose matching documents differ in length cannot produce that.
    assert!(
        !all.hits.iter().all(|h| h.1 == 1.0),
        "fuzzy hits were constant-scored: {:?}",
        all.hits
    );
}

#[test]
fn a_single_phrase_clause_takes_the_fast_path_straight_to_the_collector() {
    // The `Clause::Phrase` arm: scored by `search_phrase_query_scored`
    // directly, with no intermediate `HashMap<i32, f32>`. It must agree
    // exactly with the standalone phrase search.
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let pos_in = doc.as_ref().map(|d| d.open_pos());
    let pay_in = doc.as_ref().map(|d| d.open_pay());
    let phrase = PhraseQuery::new("pos", ["alpha", "beta"]);

    let mut direct = ScoreVecCollector::default();
    search_phrase_query_scored(
        &fields,
        doc_in.as_ref(),
        pos_in.as_ref(),
        pay_in.as_ref(),
        None,
        &phrase,
        None,
        &mut direct,
    )
    .unwrap();

    let mut boxed = ScoreVecCollector::default();
    search_boolean_query_scored(
        &fields,
        doc_in.as_ref(),
        pos_in.as_ref(),
        pay_in.as_ref(),
        None,
        None,
        &BooleanQuery::new().with_must([Clause::Phrase(phrase.clone())]),
        None,
        &mut boxed,
    )
    .unwrap();

    assert!(!direct.hits.is_empty());
    assert_eq!(direct.hits.len(), boxed.hits.len());
    for (a, b) in direct.hits.iter().zip(&boxed.hits) {
        assert_eq!(a.0, b.0);
        assert_eq!(a.1.to_bits(), b.1.to_bits());
    }
}

#[test]
fn a_single_pulsed_term_clause_reaches_the_fast_paths_term_arm() {
    // A pulsed term -- docFreq <= 1, its one posting in the term
    // dictionary -- through the single-clause fast path's `Term` arm. The
    // `many` field is 400 such terms.
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let q = BooleanQuery::new().with_must([TermQuery::new("many", "term0123")]);
    let mut boxed = ScoreVecCollector::default();
    search_boolean_query_scored(
        &fields,
        doc_in.as_ref(),
        None,
        None,
        None,
        None,
        &q,
        None,
        &mut boxed,
    )
    .unwrap();
    let mut direct = ScoreVecCollector::default();
    search_term_query_scored(
        &fields,
        doc_in.as_ref(),
        None,
        &TermQuery::new("many", "term0123"),
        None,
        &mut direct,
    )
    .unwrap();
    assert_eq!(boxed.hits.len(), 1);
    assert_eq!(boxed.hits[0].0, direct.hits[0].0);
    assert_eq!(boxed.hits[0].1.to_bits(), direct.hits[0].1.to_bits());
}

// ---- `clause_scores`' wildcard-family and fuzzy arms -------------------
//
// Reached only when the clause is *not* alone in the query (otherwise the
// fast path above takes it), which is why nothing exercised them.

#[test]
fn the_wildcard_family_scores_a_flat_one_as_a_non_solo_boolean_clause() {
    // `should = [term, prefix|wildcard|regexp]`: the disjunction's lazy path
    // declines (not every clause is a `Clause::Term`), so this runs through
    // `clause_scores`, whose wildcard-family arms all score a flat 1.0 --
    // the port's stand-in for `ConstantScoreQuery`'s constant.
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    for clause in [
        Clause::Prefix(PrefixQuery::new("big", "every")),
        Clause::Wildcard(WildcardQuery::new("big", "e*e")),
        Clause::Regexp(RegexpQuery::new("big", "ever.*")),
    ] {
        let q = BooleanQuery::new()
            .with_should([Clause::Term(TermQuery::new("body", "cat")), clause.clone()]);
        let mut all = ScoreVecCollector::default();
        search_boolean_query_scored(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            None,
            &q,
            None,
            &mut all,
        )
        .unwrap();
        // 300 `big` documents at 1.0 plus the two `body:cat` documents.
        assert_eq!(all.hits.len(), 302, "{clause:?}");
        assert_eq!(
            all.hits.iter().filter(|h| h.1 == 1.0).count(),
            300,
            "{clause:?}"
        );
    }
}

#[test]
fn a_fuzzy_clause_scores_through_blended_term_statistics_as_a_boolean_clause() {
    // `clause_scores`' `Clause::Fuzzy` arm is the only caller of
    // `fuzzy_doc_scores`, which -- unlike the flat-1.0 wildcard family --
    // computes a real BM25 score per expanded term against the *blended*
    // doc frequency (`FuzzyQuery`'s `BlendedTermQuery` behaviour) and
    // weights it by that term's edit-distance boost.
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let q = BooleanQuery::new().with_should([
        Clause::Term(TermQuery::new("body", "bird")),
        Clause::Fuzzy(FuzzyQuery::new("body", "cat").with_max_edits(1)),
    ]);
    let mut all = ScoreVecCollector::default();
    search_boolean_query_scored(
        &fields,
        doc_in.as_ref(),
        None,
        None,
        None,
        None,
        &q,
        None,
        &mut all,
    )
    .unwrap();
    assert!(!all.hits.is_empty());
    assert!(
        all.hits.iter().any(|h| h.1 > 0.0 && h.1 != 1.0),
        "a fuzzy clause must score, not contribute a flat constant: {:?}",
        all.hits
    );
}

#[test]
fn a_fuzzy_clause_with_a_missing_field_scores_nothing() {
    // `fuzzy_doc_scores`' early return, and the only branch of it a
    // caller can trip without a term-dictionary walk.
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let q = BooleanQuery::new().with_should([
        Clause::Term(TermQuery::new("body", "cat")),
        Clause::Fuzzy(FuzzyQuery::new("no_such_field", "cat")),
    ]);
    let mut all = ScoreVecCollector::default();
    search_boolean_query_scored(
        &fields,
        doc_in.as_ref(),
        None,
        None,
        None,
        None,
        &q,
        None,
        &mut all,
    )
    .unwrap();
    assert_eq!(all.hits.len(), 2, "only body:cat's documents");
}

// ---- block-max pruning in the two lazy paths --------------------------
//
// Both lazy paths prune against the
// collector's bottom score, and neither pruning branch was reachable from
// the fixture's 5-document `body` field: the queue has to *fill* before a
// threshold exists at all. The `big` field (one term, 300 documents, term
// frequencies cycling 1..4) is the fixture's only field wide enough.

#[test]
fn the_lazy_conjunction_prunes_blocks_against_a_full_queue() {
    // Two legs over the same 300-document term, so the conjunction is the
    // whole postings list and the summed block-max bound is exactly twice
    // one leg's. Once the top-2 queue holds two maximum-frequency
    // documents, no later block can beat it and the leapfrog skips whole
    // spans on impacts alone.
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let q = BooleanQuery::new().with_must([
        TermQuery::new("big", "everywhere"),
        TermQuery::new("big", "everywhere"),
    ]);

    let mut pruned = collector::TopDocsCollector::new(2);
    search_boolean_query_scored(
        &fields,
        doc_in.as_ref(),
        None,
        None,
        None,
        None,
        &q,
        None,
        &mut pruned,
    )
    .unwrap();

    // The same query with pruning forbidden (`ScoreMode::Complete`), as the
    // reference: pruning may not change which documents come back on top.
    let mut exhaustive = collector::TopDocsCollector::with_total_hits_threshold(2, u64::MAX);
    search_boolean_query_scored(
        &fields,
        doc_in.as_ref(),
        None,
        None,
        None,
        None,
        &q,
        None,
        &mut exhaustive,
    )
    .unwrap();

    assert_eq!(exhaustive.total_hits().value, 300);
    assert_eq!(
        pruned
            .top_docs()
            .iter()
            .map(|h| h.score.to_bits())
            .collect::<Vec<_>>(),
        exhaustive
            .top_docs()
            .iter()
            .map(|h| h.score.to_bits())
            .collect::<Vec<_>>(),
        "pruning changed the top-2 scores"
    );
    assert!(
        pruned.total_hits().value < 300,
        "no block was skipped: {} of 300 documents visited",
        pruned.total_hits().value
    );
}

#[test]
fn the_lazy_disjunction_prunes_blocks_against_a_full_queue() {
    // Same idea one shape over: a two-clause union whose second clause is
    // the fixture's 8,250-document `l1` term, so there are many blocks for
    // the skip loop to walk forward over on impacts alone.
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let q = BooleanQuery::new().with_should([
        TermQuery::new("big", "everywhere"),
        TermQuery::new("l1", "l1term"),
    ]);

    let mut pruned = collector::TopDocsCollector::new(3);
    search_boolean_query_scored(
        &fields,
        doc_in.as_ref(),
        None,
        None,
        None,
        None,
        &q,
        None,
        &mut pruned,
    )
    .unwrap();

    let mut exhaustive = collector::TopDocsCollector::with_total_hits_threshold(3, u64::MAX);
    search_boolean_query_scored(
        &fields,
        doc_in.as_ref(),
        None,
        None,
        None,
        None,
        &q,
        None,
        &mut exhaustive,
    )
    .unwrap();

    assert_eq!(exhaustive.total_hits().value, 8550, "300 `big` + 8250 `l1`");
    assert_eq!(
        pruned
            .top_docs()
            .iter()
            .map(|h| h.score.to_bits())
            .collect::<Vec<_>>(),
        exhaustive
            .top_docs()
            .iter()
            .map(|h| h.score.to_bits())
            .collect::<Vec<_>>(),
        "pruning changed the top-3 scores"
    );
    assert!(
        pruned.total_hits().value < 8550,
        "no block was skipped: {} of 8550 documents visited",
        pruned.total_hits().value
    );
}

#[test]
fn maxscore_term_path_skips_whole_blocks_on_a_level1_term() {
    // `BatchScoreBulkScorer` + `ImpactsDISI`: once the queue is full, a
    // block whose impacts cannot beat its bottom is stepped over on its
    // header alone. `l1term` spans 32+ blocks whose impacts vary, so there
    // are blocks to skip; the result must still equal an exhaustive run.
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let q = TermQuery::new("l1", "l1term");

    let mut exhaustive = collector::TopDocsCollector::with_total_hits_threshold(3, u64::MAX);
    search_term_query_scored_maxscore(&fields, doc_in.as_ref(), None, &q, None, &mut exhaustive)
        .unwrap();
    assert_eq!(exhaustive.total_hits().value, 8250);

    crate::test_only_maxscore_block_skip_counter::reset();
    let mut pruned = collector::TopDocsCollector::new(3);
    search_term_query_scored_maxscore(&fields, doc_in.as_ref(), None, &q, None, &mut pruned)
        .unwrap();
    let skips = crate::test_only_maxscore_block_skip_counter::count();

    assert_eq!(pruned.top_docs(), exhaustive.top_docs());
    assert!(skips > 0, "no block of 8,250 documents was skipped");
    assert!(
        pruned.total_hits().value < 8250,
        "every document still reached the collector"
    );
}

// `minimum_should_match` tests: cat={0,2}, dog={0,1}, bird={1,4} (see above).

#[test]
fn boolean_minimum_should_match_zero_with_must_present_is_unchanged_regression() {
    // `minimum_should_match == 0` (the default) must
    // still leave `should` purely score-only once `must` is non-empty, exactly
    // as without `minimum_should_match`.
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let mut c = VecCollector::default();
    let q = BooleanQuery::new()
        .with_must([TermQuery::new("body", "cat")])
        .with_should([TermQuery::new("body", "bird")])
        .with_minimum_should_match(0);
    search_boolean_query(&fields, doc_in.as_ref(), None, None, None, None, &q, &mut c).unwrap();
    assert_eq!(c.docs, vec![0, 2]);
}

#[test]
fn boolean_minimum_should_match_one_with_must_present_narrows_the_set() {
    // must=[cat]={0,2}; should=[dog,bird], dog={0,1}, bird={1,4}. With
    // minimum_should_match=1, doc 2 (0 should-clause hits) is now excluded even
    // though it satisfies `must` -- `should` genuinely narrows the set once
    // minimum_should_match > 0, unlike the 0 case above.
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let mut c = VecCollector::default();
    let q = BooleanQuery::new()
        .with_must([TermQuery::new("body", "cat")])
        .with_should([
            TermQuery::new("body", "dog"),
            TermQuery::new("body", "bird"),
        ])
        .with_minimum_should_match(1);
    search_boolean_query(&fields, doc_in.as_ref(), None, None, None, None, &q, &mut c).unwrap();
    assert_eq!(c.docs, vec![0]);
}

#[test]
fn boolean_minimum_should_match_two_with_three_should_clauses_excludes_single_hits() {
    // should=[cat,dog,bird] (must empty): doc0 hits cat+dog (2), doc1 hits
    // dog+bird (2), doc2 hits only cat (1), doc4 hits only bird (1). With
    // minimum_should_match=2, only docs with 2+ hits survive.
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let mut c = VecCollector::default();
    let q = BooleanQuery::new()
        .with_should([
            TermQuery::new("body", "cat"),
            TermQuery::new("body", "dog"),
            TermQuery::new("body", "bird"),
        ])
        .with_minimum_should_match(2);
    search_boolean_query(&fields, doc_in.as_ref(), None, None, None, None, &q, &mut c).unwrap();
    assert_eq!(c.docs, vec![0, 1]);
}

#[test]
fn boolean_minimum_should_match_with_must_empty_still_requires_the_threshold() {
    // Same should set as above but explicitly with must empty and
    // minimum_should_match=1 -- equivalent to a plain disjunction (every should
    // clause hit counts as >= 1), confirming must-empty + minSSM=1 matches the
    // existing should-disjunction-is-the-matched-set behavior.
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let mut c = VecCollector::default();
    let q = BooleanQuery::new()
        .with_should([
            TermQuery::new("body", "cat"),
            TermQuery::new("body", "bird"),
        ])
        .with_minimum_should_match(1);
    search_boolean_query(&fields, doc_in.as_ref(), None, None, None, None, &q, &mut c).unwrap();
    assert_eq!(c.docs, vec![0, 1, 2, 4]);
}

#[test]
fn boolean_minimum_should_match_exceeding_clause_count_matches_nothing() {
    // Only 2 should clauses exist; minimum_should_match=5 can never be reached
    // by any doc -- mirrors real `BooleanQuery.rewrite()`'s `MatchNoDocsQuery`
    // for "shoulds.size() < minimumNumberShouldMatch", achieved here without a
    // separate branch (see `matched_boolean_docs`'s doc comment).
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let mut c = VecCollector::default();
    let q = BooleanQuery::new()
        .with_should([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")])
        .with_minimum_should_match(5);
    search_boolean_query(&fields, doc_in.as_ref(), None, None, None, None, &q, &mut c).unwrap();
    assert!(c.docs.is_empty());
}

#[test]
fn boolean_minimum_should_match_combines_with_must_not() {
    // must=[cat]={0,2}; should=[dog,bird] with minimum_should_match=1 keeps only
    // doc 0 (see the "narrows the set" test above); must_not=[dog]={0,1}
    // additionally excludes doc 0, leaving nothing.
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let mut c = VecCollector::default();
    let q = BooleanQuery::new()
        .with_must([TermQuery::new("body", "cat")])
        .with_should([
            TermQuery::new("body", "dog"),
            TermQuery::new("body", "bird"),
        ])
        .with_must_not([TermQuery::new("body", "dog")])
        .with_minimum_should_match(1);
    search_boolean_query(&fields, doc_in.as_ref(), None, None, None, None, &q, &mut c).unwrap();
    assert!(c.docs.is_empty());
}

#[test]
fn boolean_minimum_should_match_with_no_should_clauses_at_all_matches_nothing() {
    // Task #60 edge case: a must-only query with a nonzero
    // `minimum_should_match` but zero `should` clauses -- this is the same
    // "should.len() < minimum_should_match" rule the exceeding-clause-count
    // test above already exercises, just at the should.len() == 0 boundary.
    // Real Lucene's `Boolean2ScorerSupplier.get()` returns a null scorer
    // (matches nothing) whenever `minShouldMatch > optionalScorers.size()`,
    // with no special-case carve-out for "must is otherwise satisfied" --
    // `minimum_should_match` isn't silently ignored just because there's
    // nothing for it to apply to. This port's counting mechanism already
    // produces that exact outcome with no separate branch (an empty
    // `should_docs` list means every doc's should-match count is 0, which
    // never reaches a nonzero threshold), so this test locks in already-
    // correct behavior rather than fixing a bug.
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let mut c = VecCollector::default();
    let q = BooleanQuery::new()
        .with_must([TermQuery::new("body", "cat")])
        .with_minimum_should_match(1);
    search_boolean_query(&fields, doc_in.as_ref(), None, None, None, None, &q, &mut c).unwrap();
    assert!(c.docs.is_empty());
}

#[test]
fn boolean_duplicate_should_clause_counts_and_scores_twice() {
    // Task #60 edge case: the same `TermQuery` appears twice in `should`.
    // Real Lucene does not dedupe clauses at the `BooleanQuery` level --
    // two identical `TermQuery` clauses produce two independent scorers,
    // so a matching doc counts twice toward `minimum_should_match` and
    // scores twice (once per clause instance). This is the CORRECT
    // behavior to lock in, not a bug to fix: `should_match_counts` and
    // `clause_scores` already tally per clause instance, not per distinct
    // clause value, so duplicating a clause naturally double-counts both
    // the match tally and the score sum with no dedup logic anywhere.
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());

    // Matching: cat={0,2}. Duplicating "cat" in should with
    // minimum_should_match=2 requires 2 should-hits; a single distinct
    // should clause could never reach 2, so a match here proves the
    // duplicate is counted as two separate hits.
    let mut c = VecCollector::default();
    let q = BooleanQuery::new()
        .with_should([TermQuery::new("body", "cat"), TermQuery::new("body", "cat")])
        .with_minimum_should_match(2);
    search_boolean_query(&fields, doc_in.as_ref(), None, None, None, None, &q, &mut c).unwrap();
    assert_eq!(c.docs, vec![0, 2]);

    // Scoring: the duplicated clause's score contribution must be exactly
    // double a single instance's, not deduplicated to a single
    // contribution.
    let single = BooleanQuery::new().with_should([TermQuery::new("body", "cat")]);
    let duped = BooleanQuery::new()
        .with_should([TermQuery::new("body", "cat"), TermQuery::new("body", "cat")]);
    let mut single_scores = TopDocsCollector::new(10);
    search_boolean_query_scored(
        &fields,
        doc_in.as_ref(),
        None,
        None,
        None,
        None,
        &single,
        None,
        &mut single_scores,
    )
    .unwrap();
    let mut duped_scores = TopDocsCollector::new(10);
    search_boolean_query_scored(
        &fields,
        doc_in.as_ref(),
        None,
        None,
        None,
        None,
        &duped,
        None,
        &mut duped_scores,
    )
    .unwrap();
    let single_hits = single_scores.top_docs();
    let duped_hits = duped_scores.top_docs();
    assert_eq!(single_hits.len(), duped_hits.len());
    for (single_hit, duped_hit) in single_hits.iter().zip(duped_hits.iter()) {
        assert_eq!(single_hit.doc_id, duped_hit.doc_id);
        assert_eq!(duped_hit.score, single_hit.score * 2.0);
    }
}

// Nested `BooleanQuery` clause tests: cat={0,2}, dog={0,1},
// bird={1,4} (see above).

#[test]
fn nested_boolean_must_clause_narrows_the_matched_set() {
    // top.must = [dog, nested] where nested = should=[cat, bird]. dog={0,1};
    // nested's own disjunction = cat ∪ bird = {0,1,2,4}. Conjunction: {0,1}.
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let mut c = VecCollector::default();
    let nested = BooleanQuery::new().with_should([
        TermQuery::new("body", "cat"),
        TermQuery::new("body", "bird"),
    ]);
    let q = BooleanQuery::new().with_must([
        Clause::Term(TermQuery::new("body", "dog")),
        Clause::Boolean(Box::new(nested)),
    ]);
    search_boolean_query(&fields, doc_in.as_ref(), None, None, None, None, &q, &mut c).unwrap();
    assert_eq!(c.docs, vec![0, 1]);
}

#[test]
fn nested_boolean_should_clause_contributes_to_the_disjunction() {
    // top.should = [nested] where nested = must=[cat, dog] -- nested's own
    // conjunction is {0}, so top's disjunction (its only should clause) is {0}.
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let mut c = VecCollector::default();
    let nested = BooleanQuery::new()
        .with_must([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")]);
    let q = BooleanQuery::new().with_should([nested]);
    search_boolean_query(&fields, doc_in.as_ref(), None, None, None, None, &q, &mut c).unwrap();
    assert_eq!(c.docs, vec![0]);
}

#[test]
fn nested_boolean_clauses_own_minimum_should_match_does_not_leak_to_parent() {
    // nested = should=[dog, bird], minimum_should_match=2 (its own threshold):
    // dog={0,1}, bird={1,4}, so nested's own matched set is {1} (only doc 1 hits
    // both). Top level has no minimum_should_match of its own (defaults to 0),
    // and top.must = [nested] alone -- the parent's conjunction is exactly
    // nested's matched set, proving nested's threshold is evaluated
    // independently and does narrow nested's own contribution.
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let mut c = VecCollector::default();
    let nested = BooleanQuery::new()
        .with_should([
            TermQuery::new("body", "dog"),
            TermQuery::new("body", "bird"),
        ])
        .with_minimum_should_match(2);
    let q = BooleanQuery::new().with_must([nested]);
    search_boolean_query(&fields, doc_in.as_ref(), None, None, None, None, &q, &mut c).unwrap();
    assert_eq!(c.docs, vec![1]);
}

#[test]
fn parent_minimum_should_match_does_not_affect_nested_querys_own_matching() {
    // Same nested query as above (should=[dog,bird], min_should_match=2 => {1}
    // is nested's own matched set), but now the *parent* also sets its own
    // minimum_should_match=1 over should=[nested, cat]. must is empty, so the
    // matched set is should's disjunction: nested's {1} ∪ cat's {0,2} = {0,1,2},
    // gated by parent's own min_should_match=1 (trivially satisfied by any
    // should hit) -- confirms the parent's own threshold is a fully separate
    // setting from the nested query's, neither overriding the other.
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let mut c = VecCollector::default();
    let nested = BooleanQuery::new()
        .with_should([
            TermQuery::new("body", "dog"),
            TermQuery::new("body", "bird"),
        ])
        .with_minimum_should_match(2);
    let q = BooleanQuery::new()
        .with_should([
            Clause::Boolean(Box::new(nested)),
            Clause::Term(TermQuery::new("body", "cat")),
        ])
        .with_minimum_should_match(1);
    search_boolean_query(&fields, doc_in.as_ref(), None, None, None, None, &q, &mut c).unwrap();
    assert_eq!(c.docs, vec![0, 1, 2]);
}

#[test]
fn nested_boolean_must_not_at_different_levels_does_not_leak_between_them() {
    // Task #60 edge case: a `must_not` at the outer level must exclude based
    // on the outer clause's own matching set, independent of a nested inner
    // `BooleanQuery`'s own separate `must_not`. dog={0,1}, cat={0,2},
    // bird={1,4}.
    //
    // inner: must=[dog], must_not=[cat] -> dog={0,1} minus cat={0,2} = {1}.
    let inner = BooleanQuery::new()
        .with_must([TermQuery::new("body", "dog")])
        .with_must_not([TermQuery::new("body", "cat")]);

    // First confirm the inner query's own isolated result is {1}, so the
    // outer-level assertion below is unambiguous about which level's
    // must_not did the excluding.
    let mut inner_only = VecCollector::default();
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    search_boolean_query(
        &fields,
        doc_in.as_ref(),
        None,
        None,
        None,
        None,
        &inner.clone(),
        &mut inner_only,
    )
    .unwrap();
    assert_eq!(inner_only.docs, vec![1]);

    // outer: must=[inner], must_not=[bird] -> inner's own result {1} minus
    // bird={1,4} = {} -- the outer must_not (bird) excludes doc 1 on its own
    // criteria, entirely independent of inner's own must_not (cat), which
    // never even mentions bird.
    let mut c = VecCollector::default();
    let outer = BooleanQuery::new()
        .with_must([Clause::Boolean(Box::new(inner))])
        .with_must_not([TermQuery::new("body", "bird")]);
    search_boolean_query(
        &fields,
        doc_in.as_ref(),
        None,
        None,
        None,
        None,
        &outer,
        &mut c,
    )
    .unwrap();
    assert!(c.docs.is_empty());
}

#[test]
fn nested_boolean_clause_that_matches_nothing_contributes_an_empty_set() {
    // nested has only a must_not clause -- real `BooleanQuery.rewrite()`'s pure
    // negative case, matching nothing on its own. As a `should` clause of the
    // parent, it must contribute no docs, leaving only the sibling should
    // clause's matches.
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let mut c = VecCollector::default();
    let nested = BooleanQuery::new().with_must_not([TermQuery::new("body", "dog")]);
    let q = BooleanQuery::new().with_should([
        Clause::Boolean(Box::new(nested)),
        Clause::Term(TermQuery::new("body", "cat")),
    ]);
    search_boolean_query(&fields, doc_in.as_ref(), None, None, None, None, &q, &mut c).unwrap();
    assert_eq!(c.docs, vec![0, 2]);
}

#[test]
fn three_levels_of_nested_boolean_clauses_resolve_correctly() {
    // Genuine multi-level recursion, not just one extra level: innermost =
    // must=[cat, dog] => {0}. middle.should = [innermost] => {0} (its only
    // should clause). top.must = [middle] => {0}.
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let mut c = VecCollector::default();
    let innermost = BooleanQuery::new()
        .with_must([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")]);
    let middle = BooleanQuery::new().with_should([innermost]);
    let top = BooleanQuery::new().with_must([middle]);
    search_boolean_query(
        &fields,
        doc_in.as_ref(),
        None,
        None,
        None,
        None,
        &top,
        &mut c,
    )
    .unwrap();
    assert_eq!(c.docs, vec![0]);
}

#[test]
fn nested_boolean_clause_scoring_sums_its_own_matching_sub_clauses() {
    // top.should = [nested] alone, nested.should = [cat, bird]. Nested's own
    // matched set is cat ∪ bird = {0,1,2,4}; each matched doc's score must equal
    // the sum of whichever of cat/bird it actually matches -- same recursive
    // rule `boolean_query_scored_matches_unscored_doc_set_and_sums_clause_scores`
    // in `scoring_fixtures.rs` proves at the top level, now one level deeper.
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());

    let nested = BooleanQuery::new().with_should([
        TermQuery::new("body", "cat"),
        TermQuery::new("body", "bird"),
    ]);
    let top = BooleanQuery::new().with_should([nested]);

    let mut top_docs = TopDocsCollector::new(10);
    search_boolean_query_scored(
        &fields,
        doc_in.as_ref(),
        None,
        None,
        None,
        None,
        &top,
        None,
        &mut top_docs,
    )
    .unwrap();

    let mut cat_scores = TopDocsCollector::new(10);
    search_term_query_scored(
        &fields,
        doc_in.as_ref(),
        None,
        &TermQuery::new("body", "cat"),
        None,
        &mut cat_scores,
    )
    .unwrap();
    let mut bird_scores = TopDocsCollector::new(10);
    search_term_query_scored(
        &fields,
        doc_in.as_ref(),
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

    let hits = top_docs.top_docs();
    let mut hit_docs: Vec<i32> = hits.iter().map(|h| h.doc_id).collect();
    hit_docs.sort_unstable();
    assert_eq!(hit_docs, vec![0, 1, 2, 4]);

    for hit in hits {
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

#[test]
fn nested_boolean_clause_scoring_excludes_docs_the_nested_query_itself_rejects() {
    // nested = should=[dog, bird], minimum_should_match=2 -- nested's own
    // matched set is {1} alone (see the matching-side test above). As a
    // `should` clause of top (must empty), top's matched set must be exactly
    // {1}, and its score must be dog(1) + bird(1) (both of nested's own
    // sub-clauses that doc 1 actually satisfies) -- not a score for doc 0 or 4,
    // which nested's own threshold rejects even though dog/bird individually
    // match them.
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());

    let nested = BooleanQuery::new()
        .with_should([
            TermQuery::new("body", "dog"),
            TermQuery::new("body", "bird"),
        ])
        .with_minimum_should_match(2);
    let top = BooleanQuery::new().with_should([nested]);

    let mut top_docs = TopDocsCollector::new(10);
    search_boolean_query_scored(
        &fields,
        doc_in.as_ref(),
        None,
        None,
        None,
        None,
        &top,
        None,
        &mut top_docs,
    )
    .unwrap();
    let hits = top_docs.top_docs();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].doc_id, 1);

    let mut dog_scores = TopDocsCollector::new(10);
    search_term_query_scored(
        &fields,
        doc_in.as_ref(),
        None,
        &TermQuery::new("body", "dog"),
        None,
        &mut dog_scores,
    )
    .unwrap();
    let mut bird_scores = TopDocsCollector::new(10);
    search_term_query_scored(
        &fields,
        doc_in.as_ref(),
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
    let expected =
        lookup(&dog_scores, 1).expect("dog matches doc 1") + lookup(&bird_scores, 1).unwrap();
    assert!((hits[0].score - expected).abs() < 1e-4);
}

#[test]
fn boolean_must_with_missing_term_matches_nothing() {
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let mut c = VecCollector::default();
    let q = BooleanQuery::new().with_must([
        TermQuery::new("body", "cat"),
        TermQuery::new("body", "zzz-missing"),
    ]);
    search_boolean_query(&fields, doc_in.as_ref(), None, None, None, None, &q, &mut c).unwrap();
    assert!(c.docs.is_empty());
}

#[test]
fn boolean_live_docs_filters_before_conjunction() {
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let max_doc: i32 = {
        let dir = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/blocktree_index/"
        );
        let manifest = std::fs::read_to_string(format!("{dir}manifest.properties")).unwrap();
        manifest
            .lines()
            .find_map(|l| l.strip_prefix("max_doc="))
            .unwrap()
            .parse()
            .unwrap()
    };
    let mut live_docs = FixedBitSet::new(max_doc as usize);
    for i in 0..max_doc {
        live_docs.set(i as usize);
    }
    // cat={0,2}, dog={0,1}; conjunction is {0}. Marking doc 0 dead removes the
    // only shared doc, so the conjunction (computed post-filter) is empty.
    live_docs.clear(0);

    let mut c = VecCollector::default();
    let q = BooleanQuery::new()
        .with_must([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")]);
    search_boolean_query(
        &fields,
        doc_in.as_ref(),
        None,
        None,
        Some(&live_docs),
        None,
        &q,
        &mut c,
    )
    .unwrap();
    assert!(c.docs.is_empty());
}

#[test]
fn count_collector_matches_vec_collector_length() {
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let mut count = CountCollector::default();
    let mut docs = VecCollector::default();
    search_term_query(
        &fields,
        doc_in.as_ref(),
        None,
        &TermQuery::new("body", "bird"),
        &mut count,
    )
    .unwrap();
    search_term_query(
        &fields,
        doc_in.as_ref(),
        None,
        &TermQuery::new("body", "bird"),
        &mut docs,
    )
    .unwrap();
    assert_eq!(count.count as usize, docs.docs.len());
}

// `phrase_matches_in_doc` unit tests: synthetic per-term position lists, no
// fixture needed -- this is the pure alignment-checking function in isolation.

#[test]
fn phrase_matches_exact_alignment_at_position_zero() {
    assert!(phrase_matches_in_doc(&[&[0][..], &[1][..], &[2][..]]));
}

#[test]
fn phrase_matches_exact_alignment_at_a_later_position() {
    assert!(phrase_matches_in_doc(&[
        &[0, 5][..],
        &[1, 6][..],
        &[2, 7][..]
    ]));
}

#[test]
fn phrase_no_match_despite_every_term_present() {
    // "cat" at 0 and 10, "sat" at 1, "mat" at 5: no base position aligns all three
    // (0 -> needs 1 at "sat" (ok) and 2 at "mat" (missing); 10 -> needs 11 (missing)).
    assert!(!phrase_matches_in_doc(&[&[0, 10][..], &[1][..], &[5][..]]));
}

#[test]
fn phrase_multiple_candidates_only_one_aligns() {
    // Base 0 fails (needs 2 at term index 2, only 5/7 present); base 3 succeeds
    // (needs 4 at term index 1 -- present -- and 5 at term index 2 -- present).
    assert!(phrase_matches_in_doc(&[
        &[0, 3][..],
        &[1, 4][..],
        &[5, 7][..]
    ]));
}

#[test]
fn phrase_single_term_degenerates_to_any_occurrence() {
    assert!(phrase_matches_in_doc(&[&[2, 9][..]]));
}

#[test]
fn phrase_single_term_with_no_occurrences_is_false() {
    assert!(!phrase_matches_in_doc(&[&[][..]]));
}

#[test]
fn phrase_no_terms_at_all_is_false() {
    assert!(!phrase_matches_in_doc(&[]));
}

#[test]
fn phrase_a_term_with_no_occurrences_in_this_doc_is_false() {
    assert!(!phrase_matches_in_doc(&[&[0][..], &[][..]]));
}

#[test]
fn phrase_repeated_term_with_consecutive_occurrences_matches() {
    // "the the": both occurrence lists are the term "the"'s own positions --
    // 0 and 1 are consecutive, so "the" at 0 followed by "the" at 1 is a match.
    assert!(phrase_matches_in_doc(&[&[0, 1, 2][..], &[0, 1, 2][..]]));
}

#[test]
fn phrase_repeated_term_without_consecutive_occurrences_does_not_match() {
    // "the" only occurs at 0, 2, 4 -- no two consecutive occurrences exist.
    assert!(!phrase_matches_in_doc(&[&[0, 2, 4][..], &[0, 2, 4][..]]));
}

// Fixture-backed `search_phrase_query` tests: reuse the real-Lucene "pos" field
// (`IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS`) already checked into
// `fixtures/data/blocktree_index/` for `crates/lucene-codecs/tests/
// blocktree_fixtures.rs`'s `pos_field_positions_match_real_lucene_postings_enum`
// test -- per the manifest, doc 8555 has "alpha" at position 0 and "beta" at
// position 1 (adjacent), while doc 8556 has "alpha" at positions 0 and 1 but no
// "beta" at all. That's exactly the shape a real "alpha beta" phrase query
// differential test needs, already present without extending the fixture
// generator (see this module's `Testing` section in the task write-up: prefer
// reusing existing fixtures over adding new ones when the data already fits).

#[test]
fn phrase_query_two_terms_matches_only_the_adjacent_doc() {
    let (fields, doc) = open_fixture();
    let doc = doc.unwrap();
    let doc_in = doc.open();
    let pos_in = doc.open_pos();
    let pay_in = doc.open_pay();
    let mut c = VecCollector::default();
    search_phrase_query(
        &fields,
        Some(&doc_in),
        Some(&pos_in),
        Some(&pay_in),
        None,
        &PhraseQuery::new("pos", ["alpha", "beta"]),
        &mut c,
    )
    .unwrap();
    assert_eq!(c.docs, vec![8555]);
}

#[test]
fn phrase_query_single_term_degenerates_to_term_query() {
    let (fields, doc) = open_fixture();
    let doc = doc.unwrap();
    let doc_in = doc.open();
    let mut c = VecCollector::default();
    // No .pos/.pay opened at all -- the single-term case must not need them.
    search_phrase_query(
        &fields,
        Some(&doc_in),
        None,
        None,
        None,
        &PhraseQuery::new("pos", ["alpha"]),
        &mut c,
    )
    .unwrap();
    assert_eq!(c.docs, vec![8555, 8556, 8557]);
}

#[test]
fn phrase_query_empty_terms_matches_nothing() {
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let mut c = VecCollector::default();
    search_phrase_query(
        &fields,
        doc_in.as_ref(),
        None,
        None,
        None,
        &PhraseQuery::new("pos", Vec::<&str>::new()),
        &mut c,
    )
    .unwrap();
    assert!(c.docs.is_empty());
}

#[test]
fn phrase_query_missing_field_matches_nothing() {
    let (fields, doc) = open_fixture();
    let doc = doc.unwrap();
    let doc_in = doc.open();
    let pos_in = doc.open_pos();
    let mut c = VecCollector::default();
    search_phrase_query(
        &fields,
        Some(&doc_in),
        Some(&pos_in),
        None,
        None,
        &PhraseQuery::new("nonexistent", ["a", "b"]),
        &mut c,
    )
    .unwrap();
    assert!(c.docs.is_empty());
}

#[test]
fn phrase_query_missing_term_matches_nothing() {
    let (fields, doc) = open_fixture();
    let doc = doc.unwrap();
    let doc_in = doc.open();
    let pos_in = doc.open_pos();
    let pay_in = doc.open_pay();
    let mut c = VecCollector::default();
    search_phrase_query(
        &fields,
        Some(&doc_in),
        Some(&pos_in),
        Some(&pay_in),
        None,
        &PhraseQuery::new("pos", ["alpha", "zzz-missing"]),
        &mut c,
    )
    .unwrap();
    assert!(c.docs.is_empty());
}

#[test]
fn phrase_query_duplicate_term_matches_consecutive_occurrences() {
    let (fields, doc) = open_fixture();
    let doc = doc.unwrap();
    let doc_in = doc.open();
    let pos_in = doc.open_pos();
    let pay_in = doc.open_pay();
    let mut c = VecCollector::default();
    // doc 8555 has "alpha" only at position 0 (no consecutive pair); doc 8556
    // has "alpha" at 0 and 1, a real consecutive-repeated-term match.
    search_phrase_query(
        &fields,
        Some(&doc_in),
        Some(&pos_in),
        Some(&pay_in),
        None,
        &PhraseQuery::new("pos", ["alpha", "alpha"]),
        &mut c,
    )
    .unwrap();
    assert_eq!(c.docs, vec![8556]);
}

#[test]
fn phrase_query_sloppy_wiring_still_matches_the_exact_adjacent_doc() {
    // This module's own `open_fixture()` data (alpha/beta) is exact-adjacent
    // (gap 0). The non-adjacent-by-a-known-gap cross-engine case (doc7,
    // alpha@0/beta@3) now lives in
    // `crates/lucene-search/tests/phrase_query_fixtures.rs`'s
    // `sloppy_phrase_gap_matches_real_lucenes_phrase_query_set_slop_at_every_tested_value`,
    // verified against real Lucene's actual `PhraseQuery.setSlop(n)` results
    // recorded by `GenBlockTree.java` -- see `docs/parity.md`. This test
    // instead proves `search_phrase_query` itself correctly routes `slop > 0`
    // to the sloppy path end-to-end (not just the sloppy matcher
    // in isolation, which the unit tests above
    // already cover exhaustively): a generous slop must still find exactly the
    // same match as `slop == 0` for data that's already exact-adjacent.
    let (fields, doc) = open_fixture();
    let doc = doc.unwrap();
    let doc_in = doc.open();
    let pos_in = doc.open_pos();
    let pay_in = doc.open_pay();
    let mut c = VecCollector::default();
    search_phrase_query(
        &fields,
        Some(&doc_in),
        Some(&pos_in),
        Some(&pay_in),
        None,
        &PhraseQuery::new("pos", ["alpha", "beta"]).with_slop(5),
        &mut c,
    )
    .unwrap();
    // slop=5 also bridges doc7's gap (alpha@0, beta@3, needs 2 moves).
    assert_eq!(c.docs, vec![8555, 8557]);
}

#[test]
fn phrase_query_multi_term_without_pos_input_is_an_error() {
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let mut c = VecCollector::default();
    let err = search_phrase_query(
        &fields,
        doc_in.as_ref(),
        None,
        None,
        None,
        &PhraseQuery::new("pos", ["alpha", "beta"]),
        &mut c,
    )
    .unwrap_err();
    assert!(matches!(err, Error::MissingPosInput));
}

#[test]
fn phrase_query_live_docs_filters_before_alignment_check() {
    let (fields, doc) = open_fixture();
    let doc = doc.unwrap();
    let doc_in = doc.open();
    let pos_in = doc.open_pos();
    let pay_in = doc.open_pay();
    let max_doc: i32 = {
        let dir = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/blocktree_index/"
        );
        let manifest = std::fs::read_to_string(format!("{dir}manifest.properties")).unwrap();
        manifest
            .lines()
            .find_map(|l| l.strip_prefix("max_doc="))
            .unwrap()
            .parse()
            .unwrap()
    };
    let mut live_docs = FixedBitSet::new(max_doc as usize);
    for i in 0..max_doc {
        live_docs.set(i as usize);
    }
    // "alpha beta" only ever matches doc 8555 -- marking it dead removes the
    // only match.
    live_docs.clear(8555);

    let mut c = VecCollector::default();
    search_phrase_query(
        &fields,
        Some(&doc_in),
        Some(&pos_in),
        Some(&pay_in),
        Some(&live_docs),
        &PhraseQuery::new("pos", ["alpha", "beta"]),
        &mut c,
    )
    .unwrap();
    assert!(c.docs.is_empty());
}

// `phrase_freq_exact` unit tests: pure counting logic against
// hand-built position lists, no fixture needed.

/// The branch-free two-term count is the brute-force one: each `p0` of
/// the first (strictly ascending) list whose `p0 + 1` is anywhere in the
/// second, which may repeat a position (a multi-phrase union).
#[test]
fn the_branch_free_two_term_count_matches_brute_force() {
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move |m: u64| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed % m
    };
    for _ in 0..3_000 {
        let mut first: Vec<i32> = (0..next(12)).map(|_| next(40) as i32).collect();
        first.sort_unstable();
        first.dedup();
        let mut second: Vec<i32> = (0..next(12)).map(|_| next(40) as i32).collect();
        second.sort_unstable();
        let want = first.iter().filter(|&&p| second.contains(&(p + 1))).count() as i32;
        assert_eq!(
            phrase_freq_exact_two_count(&first, &second),
            want,
            "{first:?} {second:?}"
        );
        assert_eq!(phrase_freq_exact_two(&first, &second, false), want);
        assert_eq!(phrase_freq_exact_two(&first, &second, true), want.min(1));
    }
}

#[test]
fn phrase_alignment_walk_handles_a_phrase_longer_than_the_inline_cursor_array() {
    // `phrase_freq_exact_impl` keeps its cursors on the stack for up to
    // `MAX_INLINE_PHRASE_TERMS` non-leading terms and spills to a `Vec`
    // past that. Both sides of that branch must behave identically, and
    // the spill side has no other caller in this crate's tests.
    let n = MAX_INLINE_PHRASE_TERMS + 5;
    // A phrase of `n + 1` terms that aligns exactly once, at p0 == 0.
    let lists: Vec<Vec<i32>> = (0..=n).map(|i| vec![i as i32, 500 + i as i32]).collect();
    let refs: Vec<&[i32]> = lists.iter().map(|v| v.as_slice()).collect();
    assert!(phrase_matches_in_doc(&refs));
    // Both the p0 == 0 and the p0 == 500 alignments are valid.
    assert_eq!(phrase_freq_exact(&refs), 2);

    // Break one position in the middle and it must stop matching.
    let mut broken = lists.clone();
    broken[n / 2] = vec![9999];
    let broken_refs: Vec<&[i32]> = broken.iter().map(|v| v.as_slice()).collect();
    assert!(!phrase_matches_in_doc(&broken_refs));
    assert_eq!(phrase_freq_exact(&broken_refs), 0);
}

#[test]
fn phrase_alignment_walk_agrees_with_a_brute_force_check() {
    // The merge walk's early exits (an exhausted cursor breaks out of the
    // whole scan, a mismatch only skips this `p0`) are the parts most
    // likely to be subtly wrong, and no fixture exercises them densely.
    // Compare against an independent O(n^2) definition over many shapes.
    fn brute(term_positions: &[&[i32]]) -> i32 {
        let Some((first, rest)) = term_positions.split_first() else {
            return 0;
        };
        let mut freq = 0;
        for &p0 in first.iter() {
            if rest
                .iter()
                .enumerate()
                .all(|(i, ps)| ps.contains(&(p0 + i as i32 + 1)))
            {
                freq += 1;
            }
        }
        freq
    }

    // A deterministic pseudo-random spread of positions, several widths.
    let mut seed = 0x2545_F491_4F6C_DD1Du64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for terms in 1..=4usize {
        for _ in 0..200 {
            let lists: Vec<Vec<i32>> = (0..terms)
                .map(|_| {
                    let mut v: Vec<i32> = (0i32..24).filter(|_| next() % 3 != 0).collect();
                    v.dedup();
                    v
                })
                .collect();
            let refs: Vec<&[i32]> = lists.iter().map(|v| v.as_slice()).collect();
            let expected = if lists.iter().any(|l| l.is_empty()) {
                0
            } else {
                brute(&refs)
            };
            assert_eq!(phrase_freq_exact(&refs), expected, "{lists:?}");
            assert_eq!(phrase_matches_in_doc(&refs), expected > 0, "{lists:?}");
        }
    }
}

// ---- `sloppy_phrase::sloppy_phrase_freq` at slop 0 vs the exact count ----
//
// These live here, beside `phrase_freq_exact`, because what they pin is the
// *agreement* between the two matchers; the sloppy walk's own behaviour is
// unit-tested in `sloppy_phrase`.

/// Every slot a distinct term, i.e. `SloppyPhraseMatcher`'s no-repeats case.
fn sloppy_freq(term_positions: &[&[i32]], slop: u32) -> f32 {
    sloppy_phrase::sloppy_phrase_freq(
        term_positions,
        &sloppy_phrase::PhraseRepeats::none(term_positions.len()),
        slop,
    )
}

fn sloppy_matches(term_positions: &[&[i32]], slop: u32) -> bool {
    sloppy_phrase::sloppy_phrase_matches(
        term_positions,
        &sloppy_phrase::PhraseRepeats::none(term_positions.len()),
        slop,
    )
}

#[test]
fn sloppy_phrase_freq_at_slop_zero_equals_the_exact_count() {
    // The defining agreement: with a zero move budget every match has
    // `matchLength == 0`, so each weighs `1/(1+0) == 1` and the sum is
    // exactly `phrase_freq_exact`'s count. Checked over several shapes,
    // including a repeated-term phrase and a non-matching one.
    let cases: [(Vec<Vec<i32>>, u32); 4] = [
        (vec![vec![0, 5], vec![1, 6]], 2),
        (vec![vec![0, 1, 2], vec![1, 2, 3]], 3),
        (vec![vec![0, 4], vec![9]], 0),
        (vec![vec![7], vec![3]], 0),
    ];
    for (positions, expected) in cases {
        let refs: Vec<&[i32]> = positions.iter().map(|v| v.as_slice()).collect();
        assert_eq!(phrase_freq_exact(&refs), expected as i32, "{positions:?}");
        assert_eq!(
            sloppy_freq(&refs, 0),
            expected as f32,
            "slop-0 sloppy freq must equal the exact count for {positions:?}"
        );
    }
}

#[test]
fn sloppy_phrase_freq_weights_a_loose_match_below_a_tight_one() {
    // `alpha`@0 with `beta`@1 is `matchLength == 0` -> weight 1.
    let tight = sloppy_freq(&[&[0], &[1]], 3);
    // `alpha`@0 with `beta`@3 is `matchLength == 2` -> weight 1/3, the
    // exact case `fixtures/data/blocktree_index`'s doc 7 records against
    // real Lucene (see `tests/bm25_scoring_fixtures.rs`).
    let loose = sloppy_freq(&[&[0], &[3]], 3);
    assert_eq!(tight, 1.0);
    assert_eq!(loose, 1.0 / 3.0);
    assert!(loose < tight);
}

#[test]
fn sloppy_phrase_freq_sums_every_starting_position() {
    // Two starts: 0 (pairs with 1, matchLength 0) and 4 (pairs with 7,
    // matchLength 2). Sum = 1 + 1/3.
    let got = sloppy_freq(&[&[0, 4], &[1, 7]], 3);
    assert_eq!(got, 1.0 + 1.0 / 3.0);
}

#[test]
fn sloppy_phrase_freq_skips_starts_over_the_slop_budget() {
    // Start 0 needs 4 moves to reach `beta`@5, over a budget of 2; start 6
    // reaches `beta`@7 with 0. Only the second contributes.
    let got = sloppy_freq(&[&[0, 6], &[5, 7]], 2);
    assert_eq!(got, 1.0);
}

#[test]
fn sloppy_phrase_freq_edge_cases_match_the_matchers_own_contract() {
    // Empty phrase, an empty position list, and a single-term phrase --
    // the same three edge cases the sloppy matcher documents.
    assert_eq!(sloppy_freq(&[], 5), 0.0);
    assert_eq!(sloppy_freq(&[&[1, 2], &[]], 5), 0.0);
    assert_eq!(sloppy_freq(&[&[1, 2, 9]], 5), 3.0);
    assert_eq!(sloppy_freq(&[&[]], 5), 0.0);
}

#[test]
fn sloppy_phrase_freq_is_nonzero_exactly_when_the_matcher_says_it_matches() {
    // The frequency and the boolean matcher must never disagree about
    // whether a document matches -- they are used together (one gates the
    // unscored path, the other the scored one).
    let shapes: [Vec<Vec<i32>>; 6] = [
        vec![vec![0], vec![1]],
        vec![vec![0], vec![3]],
        vec![vec![0], vec![9]],
        vec![vec![0, 5], vec![2, 6], vec![3, 20]],
        vec![vec![4], vec![2]],
        vec![vec![1, 2, 3], vec![2, 3, 4], vec![3, 4, 5]],
    ];
    for slop in 0..=4u32 {
        for shape in &shapes {
            let refs: Vec<&[i32]> = shape.iter().map(|v| v.as_slice()).collect();
            assert_eq!(
                sloppy_freq(&refs, slop) > 0.0,
                sloppy_matches(&refs, slop),
                "slop={slop} shape={shape:?}"
            );
        }
    }
}

/// **The two-term merge counts what the general loop counts**, stopping
/// at the first match or not: random non-decreasing position lists of
/// every small length, dense and sparse, with and without alignments,
/// and with the repeated positions a multi-phrase union produces.
#[test]
fn the_two_term_phrase_merge_agrees_with_the_general_loop() {
    let mut s = 0x1234_5678_9ABC_DEF1u64;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    for _ in 0..20_000 {
        let spread = 1 + next() % 12;
        let (na, nb) = (next() % 9, next() % 9);
        let mut list = |n: u64| {
            let mut p = (next() % 4) as i32;
            (0..n)
                .map(|_| {
                    // A step of 0 repeats a position, as a union does.
                    p += (next() % (spread + 1)) as i32;
                    p
                })
                .collect::<Vec<i32>>()
        };
        let (a, b) = (list(na), list(nb));
        for stop in [false, true] {
            assert_eq!(
                phrase_freq_exact_two(&a, &b, stop),
                phrase_freq_exact_general(&[&a, &b], stop),
                "{a:?} {b:?} stop={stop}"
            );
        }
    }
}

#[test]
fn phrase_freq_exact_counts_one_match_when_phrase_occurs_once() {
    // "quick fox" at position 0/1 only.
    assert_eq!(phrase_freq_exact(&[&[0][..], &[1][..]]), 1);
}

#[test]
fn phrase_freq_exact_counts_every_repeated_occurrence() {
    // "the the": "the" at 0,1,2,3 -- valid starts at 0,1,2 (0+1=1 present,
    // 1+1=2 present, 2+1=3 present), 3 has no successor -- 3 matches, not 1.
    let positions = [vec![0, 1, 2, 3], vec![0, 1, 2, 3]];
    assert_eq!(
        phrase_freq_exact(&positions.iter().map(|v| v.as_slice()).collect::<Vec<_>>()),
        3
    );
}

#[test]
fn phrase_freq_exact_zero_when_no_alignment_exists() {
    assert_eq!(phrase_freq_exact(&[&[0][..], &[5][..]]), 0);
}

#[test]
fn phrase_freq_exact_zero_for_empty_term_positions() {
    assert_eq!(phrase_freq_exact(&[]), 0);
}

#[test]
fn phrase_freq_exact_zero_when_any_term_has_no_occurrences() {
    assert_eq!(phrase_freq_exact(&[&[0][..], &[][..]]), 0);
}

#[test]
fn phrase_freq_exact_single_term_counts_every_occurrence() {
    assert_eq!(phrase_freq_exact(&[&[0, 3, 7][..]]), 3);
}

#[test]
fn phrase_freq_exact_non_overlapping_repeats_counts_two() {
    // "quick fox ... quick fox": two disjoint adjacent pairs, no overlap
    // possible between them.
    let positions = [vec![0, 10], vec![1, 11]];
    assert_eq!(
        phrase_freq_exact(&positions.iter().map(|v| v.as_slice()).collect::<Vec<_>>()),
        2
    );
}

// `span_matches_in_doc` unit tests: synthetic per-leaf position
// maps, no fixture needed -- this is the pure span-computation function in
// isolation, mirroring `phrase_matches_in_doc`'s own test style above.

fn leaf_positions(pairs: &[(&str, &[i32])]) -> HashMap<SpanLeafKey, Vec<i32>> {
    pairs
        .iter()
        .map(|&(term, positions)| {
            (
                ("f".to_string(), term.as_bytes().to_vec()),
                positions.to_vec(),
            )
        })
        .collect()
}

#[test]
fn span_term_matches_every_occurrence_in_a_multi_occurrence_doc() {
    let positions = leaf_positions(&[("cat", &[0, 3, 7])]);
    let query = SpanQuery::span_term("f", "cat");
    assert_eq!(
        span_matches_in_doc(&query, &positions),
        vec![(0, 1), (3, 4), (7, 8)]
    );
}

#[test]
fn span_term_no_occurrences_yields_no_spans() {
    let positions = leaf_positions(&[]);
    let query = SpanQuery::span_term("f", "cat");
    assert!(span_matches_in_doc(&query, &positions).is_empty());
}

#[test]
fn span_near_in_order_matches_an_ordered_adjacent_pair() {
    // "cat" at 0, "sat" at 1 -- adjacent, in phrase order.
    let positions = leaf_positions(&[("cat", &[0]), ("sat", &[1])]);
    let query = SpanQuery::span_near(
        [
            SpanQuery::span_term("f", "cat"),
            SpanQuery::span_term("f", "sat"),
        ],
        0,
        true,
    );
    assert_eq!(span_matches_in_doc(&query, &positions), vec![(0, 2)]);
}

#[test]
fn span_near_in_order_does_not_match_a_reversed_pair() {
    // "sat" at 0, "cat" at 1 -- clauses are [cat, sat], but the doc has
    // "sat" occur first: in-order requires clause 0's span before clause 1's.
    let positions = leaf_positions(&[("cat", &[1]), ("sat", &[0])]);
    let query = SpanQuery::span_near(
        [
            SpanQuery::span_term("f", "cat"),
            SpanQuery::span_term("f", "sat"),
        ],
        0,
        true,
    );
    assert!(span_matches_in_doc(&query, &positions).is_empty());
}

#[test]
fn span_near_out_of_order_matches_a_reversed_pair_within_slop() {
    // Same reversed doc as above, but `in_order == false`: any relative
    // order is accepted, so this DOES match -- the key differentiator from
    // `PhraseQuery`'s in-order-only sloppy matching.
    let positions = leaf_positions(&[("cat", &[1]), ("sat", &[0])]);
    let query = SpanQuery::span_near(
        [
            SpanQuery::span_term("f", "cat"),
            SpanQuery::span_term("f", "sat"),
        ],
        0,
        false,
    );
    assert_eq!(span_matches_in_doc(&query, &positions), vec![(0, 2)]);
}

#[test]
fn span_near_respects_slop_boundary_exactly_at_limit_matches() {
    // "cat" at 0, "sat" at 2 -- one word gap, slack = (2 - 1) = 1.
    let positions = leaf_positions(&[("cat", &[0]), ("sat", &[2])]);
    let query = SpanQuery::span_near(
        [
            SpanQuery::span_term("f", "cat"),
            SpanQuery::span_term("f", "sat"),
        ],
        1,
        true,
    );
    assert_eq!(span_matches_in_doc(&query, &positions), vec![(0, 3)]);
}

#[test]
fn span_near_respects_slop_boundary_one_over_does_not_match() {
    let positions = leaf_positions(&[("cat", &[0]), ("sat", &[2])]);
    let query = SpanQuery::span_near(
        [
            SpanQuery::span_term("f", "cat"),
            SpanQuery::span_term("f", "sat"),
        ],
        0,
        true,
    );
    assert!(span_matches_in_doc(&query, &positions).is_empty());
}

#[test]
fn span_or_matches_if_either_sub_span_matches() {
    let cat_only = leaf_positions(&[("cat", &[0])]);
    let dog_only = leaf_positions(&[("dog", &[0])]);
    let neither = leaf_positions(&[]);
    let both = leaf_positions(&[("cat", &[0]), ("dog", &[5])]);
    let query = SpanQuery::span_or([
        SpanQuery::span_term("f", "cat"),
        SpanQuery::span_term("f", "dog"),
    ]);
    assert_eq!(span_matches_in_doc(&query, &cat_only), vec![(0, 1)]);
    assert_eq!(span_matches_in_doc(&query, &dog_only), vec![(0, 1)]);
    assert!(span_matches_in_doc(&query, &neither).is_empty());
    assert_eq!(span_matches_in_doc(&query, &both), vec![(0, 1), (5, 6)]);
}

#[test]
fn span_near_of_span_near_composes_correctly() {
    // (cat NEAR/0,in-order sat) NEAR/0,in-order mat: "cat" 0, "sat" 1, "mat" 2.
    let positions = leaf_positions(&[("cat", &[0]), ("sat", &[1]), ("mat", &[2])]);
    let inner = SpanQuery::span_near(
        [
            SpanQuery::span_term("f", "cat"),
            SpanQuery::span_term("f", "sat"),
        ],
        0,
        true,
    );
    let outer = SpanQuery::span_near([inner, SpanQuery::span_term("f", "mat")], 0, true);
    assert_eq!(span_matches_in_doc(&outer, &positions), vec![(0, 3)]);
}

#[test]
fn span_near_of_span_near_no_match_when_inner_does_not_align() {
    // Inner "cat sat" fails to align (gap too big for slop=0), so the outer
    // near can never find an inner span to combine with "mat".
    let positions = leaf_positions(&[("cat", &[0]), ("sat", &[5]), ("mat", &[6])]);
    let inner = SpanQuery::span_near(
        [
            SpanQuery::span_term("f", "cat"),
            SpanQuery::span_term("f", "sat"),
        ],
        0,
        true,
    );
    let outer = SpanQuery::span_near([inner, SpanQuery::span_term("f", "mat")], 0, true);
    assert!(span_matches_in_doc(&outer, &positions).is_empty());
}

#[test]
fn span_near_empty_clauses_never_matches() {
    let positions = leaf_positions(&[("cat", &[0])]);
    let query = SpanQuery::span_near(std::iter::empty(), 0, true);
    assert!(span_matches_in_doc(&query, &positions).is_empty());
}

#[test]
fn span_near_a_clause_with_no_occurrences_never_matches() {
    let positions = leaf_positions(&[("cat", &[0])]);
    let query = SpanQuery::span_near(
        [
            SpanQuery::span_term("f", "cat"),
            SpanQuery::span_term("f", "sat"),
        ],
        10,
        true,
    );
    assert!(span_matches_in_doc(&query, &positions).is_empty());
}

// `search_phrase_query_scored` fixture-driven tests: reuses the
// `pos` field's real alpha/beta postings this module's `search_phrase_query`
// tests already validate at the matching layer.

#[test]
fn phrase_query_scored_matches_unscored_doc_set_and_scores_positively() {
    let (fields, doc) = open_fixture();
    let doc = doc.unwrap();
    let doc_in = doc.open();
    let pos_in = doc.open_pos();
    let pay_in = doc.open_pay();

    let mut unscored = VecCollector::default();
    search_phrase_query(
        &fields,
        Some(&doc_in),
        Some(&pos_in),
        Some(&pay_in),
        None,
        &PhraseQuery::new("pos", ["alpha", "beta"]),
        &mut unscored,
    )
    .unwrap();
    assert_eq!(unscored.docs, vec![8555]);

    let mut top = TopDocsCollector::new(10);
    search_phrase_query_scored(
        &fields,
        Some(&doc_in),
        Some(&pos_in),
        Some(&pay_in),
        None,
        &PhraseQuery::new("pos", ["alpha", "beta"]),
        None,
        &mut top,
    )
    .unwrap();
    let hits = top.top_docs();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].doc_id, 8555);
    assert!(hits[0].score > 0.0);
}

#[test]
fn phrase_query_scored_single_term_delegates_to_term_scoring() {
    let (fields, doc) = open_fixture();
    let doc = doc.unwrap();
    let doc_in = doc.open();

    let mut phrase_top = TopDocsCollector::new(10);
    search_phrase_query_scored(
        &fields,
        Some(&doc_in),
        None,
        None,
        None,
        &PhraseQuery::new("pos", ["alpha"]),
        None,
        &mut phrase_top,
    )
    .unwrap();

    let mut term_top = TopDocsCollector::new(10);
    search_term_query_scored(
        &fields,
        Some(&doc_in),
        None,
        &TermQuery::new("pos", "alpha"),
        None,
        &mut term_top,
    )
    .unwrap();

    let mut phrase_hits: Vec<(i32, f32)> = phrase_top
        .top_docs()
        .iter()
        .map(|h| (h.doc_id, h.score))
        .collect();
    let mut term_hits: Vec<(i32, f32)> = term_top
        .top_docs()
        .iter()
        .map(|h| (h.doc_id, h.score))
        .collect();
    phrase_hits.sort_by_key(|h| h.0);
    term_hits.sort_by_key(|h| h.0);
    assert_eq!(phrase_hits.len(), term_hits.len());
    for ((pd, ps), (td, ts)) in phrase_hits.iter().zip(term_hits.iter()) {
        assert_eq!(pd, td);
        assert!((ps - ts).abs() < 1e-6);
    }
}

#[test]
fn phrase_query_scored_empty_terms_yields_no_hits() {
    let (fields, doc) = open_fixture();
    let doc = doc.unwrap();
    let doc_in = doc.open();
    let mut top = TopDocsCollector::new(10);
    search_phrase_query_scored(
        &fields,
        Some(&doc_in),
        None,
        None,
        None,
        &PhraseQuery::default(),
        None,
        &mut top,
    )
    .unwrap();
    assert!(top.top_docs().is_empty());
}

#[test]
fn phrase_query_scored_missing_term_yields_no_hits() {
    let (fields, doc) = open_fixture();
    let doc = doc.unwrap();
    let doc_in = doc.open();
    let pos_in = doc.open_pos();
    let pay_in = doc.open_pay();
    let mut top = TopDocsCollector::new(10);
    search_phrase_query_scored(
        &fields,
        Some(&doc_in),
        Some(&pos_in),
        Some(&pay_in),
        None,
        &PhraseQuery::new("pos", ["alpha", "zzz-missing"]),
        None,
        &mut top,
    )
    .unwrap();
    assert!(top.top_docs().is_empty());
}

#[test]
fn phrase_query_scored_multi_term_without_pos_input_is_an_error() {
    let (fields, doc) = open_fixture();
    let doc = doc.unwrap();
    let doc_in = doc.open();
    let mut top = TopDocsCollector::new(10);
    let err = search_phrase_query_scored(
        &fields,
        Some(&doc_in),
        None,
        None,
        None,
        &PhraseQuery::new("pos", ["alpha", "beta"]),
        None,
        &mut top,
    )
    .unwrap_err();
    assert!(matches!(err, Error::MissingPosInput));
}

#[test]
fn span_leaves_on_fields_the_segment_lacks_have_no_positions_without_pos_input() {
    let fields = BlockTreeFields::empty();
    let q = SpanQuery::span_near(
        [
            SpanQuery::span_term("f", "a"),
            SpanQuery::span_term("f", "b"),
        ],
        1,
        true,
    );
    assert!(span_leaf_positions(&fields, None, None, None, None, &q)
        .unwrap()
        .is_none());
}

/// `PhraseWeight.getPhraseMatcher`: a segment without the field has no
/// matcher, whether or not it has a `.pos` file -- an OpenSearch
/// tombstone segment (only `_id` and the sequence fields) is one.
#[test]
fn phrase_query_on_a_field_the_segment_lacks_matches_nothing_without_pos_input() {
    let (fields, doc) = open_fixture();
    let doc = doc.unwrap();
    let doc_in = doc.open();
    let q = PhraseQuery::new("nosuchfield", ["alpha", "beta"]);
    let mut c = VecCollector::default();
    search_phrase_query(&fields, Some(&doc_in), None, None, None, &q, &mut c).unwrap();
    assert!(c.docs.is_empty());
    let mut top = TopDocsCollector::new(10);
    search_phrase_query_scored(&fields, Some(&doc_in), None, None, None, &q, None, &mut top)
        .unwrap();
    assert!(top.top_docs().is_empty());
}

#[test]
fn phrase_query_scored_repeated_phrase_scores_higher_than_single_occurrence() {
    // doc 8556 has "alpha" at 0 and 1 -- "alpha alpha" matches twice there
    // (phrase_freq_exact counts both consecutive starts). A higher phraseFreq
    // must yield a strictly higher BM25 score than a doc with phraseFreq 1,
    // same monotonicity property term scoring already proves for `freq`.
    let (fields, doc) = open_fixture();
    let doc = doc.unwrap();
    let doc_in = doc.open();
    let pos_in = doc.open_pos();
    let pay_in = doc.open_pay();

    let mut top = TopDocsCollector::new(10);
    search_phrase_query_scored(
        &fields,
        Some(&doc_in),
        Some(&pos_in),
        Some(&pay_in),
        None,
        &PhraseQuery::new("pos", ["alpha", "alpha"]),
        None,
        &mut top,
    )
    .unwrap();
    // Only doc 8556 has a consecutive "alpha alpha" alignment (doc 8555 has
    // "alpha" only once, doc 8557 likewise) -- see
    // `phrase_query_duplicate_term_matches_consecutive_occurrences` above.
    let hits = top.top_docs();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].doc_id, 8556);
    assert!(hits[0].score > 0.0);
}

// `Clause::Phrase` inside a `BooleanQuery`: matching + scoring,
// plus one nested case.

#[test]
fn boolean_must_with_phrase_clause_narrows_the_matched_set() {
    // must = [phrase("alpha beta"), term("alpha")]: phrase matches only 8555;
    // term "alpha" matches 8555, 8556, 8557 -- conjunction is {8555}.
    let (fields, doc) = open_fixture();
    let doc = doc.unwrap();
    let doc_in = doc.open();
    let pos_in = doc.open_pos();
    let pay_in = doc.open_pay();

    let q = BooleanQuery::new().with_must([
        Clause::Phrase(PhraseQuery::new("pos", ["alpha", "beta"])),
        Clause::Term(TermQuery::new("pos", "alpha")),
    ]);
    let mut c = VecCollector::default();
    search_boolean_query(
        &fields,
        Some(&doc_in),
        Some(&pos_in),
        Some(&pay_in),
        None,
        None,
        &q,
        &mut c,
    )
    .unwrap();
    assert_eq!(c.docs, vec![8555]);
}

#[test]
fn boolean_should_with_phrase_clause_scores_additively() {
    let (fields, doc) = open_fixture();
    let doc = doc.unwrap();
    let doc_in = doc.open();
    let pos_in = doc.open_pos();
    let pay_in = doc.open_pay();

    let q = BooleanQuery::new().with_should([
        Clause::Phrase(PhraseQuery::new("pos", ["alpha", "beta"])),
        Clause::Term(TermQuery::new("pos", "alpha")),
    ]);
    let mut top = TopDocsCollector::new(10);
    search_boolean_query_scored(
        &fields,
        Some(&doc_in),
        Some(&pos_in),
        Some(&pay_in),
        None,
        None,
        &q,
        None,
        &mut top,
    )
    .unwrap();

    let mut phrase_only = TopDocsCollector::new(10);
    search_phrase_query_scored(
        &fields,
        Some(&doc_in),
        Some(&pos_in),
        Some(&pay_in),
        None,
        &PhraseQuery::new("pos", ["alpha", "beta"]),
        None,
        &mut phrase_only,
    )
    .unwrap();
    let mut term_only = TopDocsCollector::new(10);
    search_term_query_scored(
        &fields,
        Some(&doc_in),
        None,
        &TermQuery::new("pos", "alpha"),
        None,
        &mut term_only,
    )
    .unwrap();

    let lookup = |top: &TopDocsCollector, doc_id: i32| -> Option<f32> {
        top.top_docs()
            .iter()
            .find(|h| h.doc_id == doc_id)
            .map(|h| h.score)
    };
    let hits = top.top_docs();
    let mut hit_docs: Vec<i32> = hits.iter().map(|h| h.doc_id).collect();
    hit_docs.sort_unstable();
    assert_eq!(hit_docs, vec![8555, 8556, 8557]);
    for hit in hits {
        let expected = lookup(&phrase_only, hit.doc_id).unwrap_or(0.0)
            + lookup(&term_only, hit.doc_id).unwrap_or(0.0);
        assert!(
            (hit.score - expected).abs() < 1e-4,
            "doc={} got={} expected={}",
            hit.doc_id,
            hit.score,
            expected
        );
    }
}

#[test]
fn nested_boolean_clause_containing_a_phrase_clause_resolves_correctly() {
    // top.must = [nested], nested.should = [phrase("alpha beta"), term("gamma"
    // -- missing, contributes nothing)] -- nested's own disjunction is just the
    // phrase's matched set {8555}; the parent's conjunction (its only clause)
    // must equal that same set.
    let (fields, doc) = open_fixture();
    let doc = doc.unwrap();
    let doc_in = doc.open();
    let pos_in = doc.open_pos();
    let pay_in = doc.open_pay();

    let nested = BooleanQuery::new().with_should([
        Clause::Phrase(PhraseQuery::new("pos", ["alpha", "beta"])),
        Clause::Term(TermQuery::new("pos", "zzz-missing")),
    ]);
    let top_query = BooleanQuery::new().with_must([Clause::Boolean(Box::new(nested))]);

    let mut c = VecCollector::default();
    search_boolean_query(
        &fields,
        Some(&doc_in),
        Some(&pos_in),
        Some(&pay_in),
        None,
        None,
        &top_query,
        &mut c,
    )
    .unwrap();
    assert_eq!(c.docs, vec![8555]);

    // Scoring side: the nested clause's phrase contribution must equal the
    // phrase's own standalone score for doc 8555.
    let mut top = TopDocsCollector::new(10);
    search_boolean_query_scored(
        &fields,
        Some(&doc_in),
        Some(&pos_in),
        Some(&pay_in),
        None,
        None,
        &top_query,
        None,
        &mut top,
    )
    .unwrap();
    let mut phrase_only = TopDocsCollector::new(10);
    search_phrase_query_scored(
        &fields,
        Some(&doc_in),
        Some(&pos_in),
        Some(&pay_in),
        None,
        &PhraseQuery::new("pos", ["alpha", "beta"]),
        None,
        &mut phrase_only,
    )
    .unwrap();
    let hits = top.top_docs();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].doc_id, 8555);
    let expected = phrase_only.top_docs()[0].score;
    assert!((hits[0].score - expected).abs() < 1e-4);
}

#[test]
fn boolean_phrase_clause_without_pos_input_is_an_error() {
    let (fields, doc) = open_fixture();
    let doc = doc.unwrap();
    let doc_in = doc.open();
    let q =
        BooleanQuery::new().with_must([Clause::Phrase(PhraseQuery::new("pos", ["alpha", "beta"]))]);
    let mut c = VecCollector::default();
    let err = search_boolean_query(&fields, Some(&doc_in), None, None, None, None, &q, &mut c)
        .unwrap_err();
    assert!(matches!(err, Error::MissingPosInput));
}

// `DisjunctionMaxQuery`: matching is a pure union, scoring is
// `max(disjunct scores) + tie_breaker * sum(rest)`. `body`'s known postings
// (see the `Clause::Boolean`/BM25 tests above and `scoring_fixtures.rs`):
// cat={0,2}, dog={0,1}, bird={1,4}.

#[test]
fn dismax_matches_the_union_of_every_disjunct() {
    let (fields, doc) = open_fixture();
    let doc = doc.unwrap();
    let doc_in = doc.open();

    let q = DisjunctionMaxQuery::new(
        [
            Clause::Term(TermQuery::new("body", "cat")),
            Clause::Term(TermQuery::new("body", "bird")),
        ],
        0.0,
    );
    let mut c = VecCollector::default();
    search_disjunction_max_query(&fields, Some(&doc_in), None, None, None, None, &q, &mut c)
        .unwrap();
    // cat={0,2} union bird={1,4} = {0,1,2,4}, ascending.
    assert_eq!(c.docs, vec![0, 1, 2, 4]);
}

#[test]
fn dismax_with_no_disjuncts_matches_nothing() {
    let (fields, doc) = open_fixture();
    let doc = doc.unwrap();
    let doc_in = doc.open();

    let q = DisjunctionMaxQuery::new(Vec::<Clause>::new(), 0.0);
    let mut c = VecCollector::default();
    search_disjunction_max_query(&fields, Some(&doc_in), None, None, None, None, &q, &mut c)
        .unwrap();
    assert!(c.docs.is_empty());
}

#[test]
fn dismax_missing_term_disjunct_contributes_nothing_to_the_union() {
    let (fields, doc) = open_fixture();
    let doc = doc.unwrap();
    let doc_in = doc.open();

    let q = DisjunctionMaxQuery::new(
        [
            Clause::Term(TermQuery::new("body", "cat")),
            Clause::Term(TermQuery::new("body", "zzz-missing")),
        ],
        0.0,
    );
    let mut c = VecCollector::default();
    search_disjunction_max_query(&fields, Some(&doc_in), None, None, None, None, &q, &mut c)
        .unwrap();
    assert_eq!(c.docs, vec![0, 2]);
}

#[test]
fn dismax_scored_with_zero_tie_breaker_is_pure_max_of_disjunct_scores() {
    // doc 0 matches both cat and dog -- with tie_breaker == 0.0 its score
    // must be exactly max(cat_score(0), dog_score(0)), not their sum.
    let (fields, doc) = open_fixture();
    let doc = doc.unwrap();
    let doc_in = doc.open();

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
        "test needs distinct scores to prove max, not sum"
    );
    let expected_max = cat0.max(dog0);

    let q = DisjunctionMaxQuery::new(
        [
            Clause::Term(TermQuery::new("body", "cat")),
            Clause::Term(TermQuery::new("body", "dog")),
        ],
        0.0,
    );
    let mut top = TopDocsCollector::new(10);
    search_disjunction_max_query_scored(
        &fields,
        Some(&doc_in),
        None,
        None,
        None,
        None,
        &q,
        None,
        &mut top,
    )
    .unwrap();
    assert!((score_of(&top, 0) - expected_max).abs() < 1e-4);
}

#[test]
fn dismax_scored_tie_breaker_arithmetic_matches_the_exact_formula() {
    // Exact arithmetic proof of `max + tie_breaker * sum(rest)`, computed
    // both ways from independently-derived single-clause scores -- doc 0
    // matches both cat and dog.
    let (fields, doc) = open_fixture();
    let doc = doc.unwrap();
    let doc_in = doc.open();

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
    let tie_breaker = 0.3f32;
    let expected = cat0.max(dog0) + tie_breaker * cat0.min(dog0);

    let q = DisjunctionMaxQuery::new(
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
        None,
        &q,
        None,
        &mut top,
    )
    .unwrap();
    assert!((score_of(&top, 0) - expected).abs() < 1e-5);
}

#[test]
fn dismax_scored_doc_matching_only_one_disjunct_gets_exactly_that_score() {
    // doc 2 matches only cat (not dog): the tie_breaker term is multiplied
    // by zero "other" contributions, so the score is exactly cat's own
    // score regardless of tie_breaker.
    let (fields, doc) = open_fixture();
    let doc = doc.unwrap();
    let doc_in = doc.open();

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
    let score_of = |top: &TopDocsCollector, doc_id: i32| -> f32 {
        top.top_docs()
            .iter()
            .find(|h| h.doc_id == doc_id)
            .map(|h| h.score)
            .unwrap()
    };
    let cat2 = score_of(&cat, 2);

    let q = DisjunctionMaxQuery::new(
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
        None,
        &q,
        None,
        &mut top,
    )
    .unwrap();
    assert!((score_of(&top, 2) - cat2).abs() < 1e-4);
}

#[test]
fn dismax_nested_inside_a_boolean_clause_matches_and_scores_correctly() {
    // top.must = [term(dog), dismax([cat, bird])]: dog={0,1}, dismax union
    // cat∪bird = {0,1,2,4}, conjunction = {0,1}.
    let (fields, doc) = open_fixture();
    let doc = doc.unwrap();
    let doc_in = doc.open();

    let dismax = DisjunctionMaxQuery::new(
        [
            Clause::Term(TermQuery::new("body", "cat")),
            Clause::Term(TermQuery::new("body", "bird")),
        ],
        0.0,
    );
    let q = BooleanQuery::new().with_must([
        Clause::Term(TermQuery::new("body", "dog")),
        Clause::DisjunctionMax(Box::new(dismax)),
    ]);
    let mut c = VecCollector::default();
    search_boolean_query(&fields, Some(&doc_in), None, None, None, None, &q, &mut c).unwrap();
    assert_eq!(c.docs, vec![0, 1]);
}

#[test]
fn boolean_clause_nested_inside_a_dismax_disjunct_matches_and_scores_correctly() {
    // dismax([term(bird), boolean.must=[cat, dog]]): boolean's own
    // conjunction cat∩dog = {0}; bird = {1,4}. Union = {0,1,4}.
    let (fields, doc) = open_fixture();
    let doc = doc.unwrap();
    let doc_in = doc.open();

    let nested_bool = BooleanQuery::new()
        .with_must([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")]);
    let q = DisjunctionMaxQuery::new(
        [
            Clause::Term(TermQuery::new("body", "bird")),
            Clause::Boolean(Box::new(nested_bool)),
        ],
        0.0,
    );
    let mut c = VecCollector::default();
    search_disjunction_max_query(&fields, Some(&doc_in), None, None, None, None, &q, &mut c)
        .unwrap();
    assert_eq!(c.docs, vec![0, 1, 4]);
}

#[test]
fn dismax_nested_inside_another_dismax_recurses_to_multiple_levels() {
    // outer dismax([term(bird), inner dismax([cat, dog])]): inner union
    // cat∪dog = {0,1,2}; outer union with bird{1,4} = {0,1,2,4}.
    let (fields, doc) = open_fixture();
    let doc = doc.unwrap();
    let doc_in = doc.open();

    let inner = DisjunctionMaxQuery::new(
        [
            Clause::Term(TermQuery::new("body", "cat")),
            Clause::Term(TermQuery::new("body", "dog")),
        ],
        0.0,
    );
    let outer = DisjunctionMaxQuery::new(
        [
            Clause::Term(TermQuery::new("body", "bird")),
            Clause::DisjunctionMax(Box::new(inner)),
        ],
        0.0,
    );
    let mut c = VecCollector::default();
    search_disjunction_max_query(
        &fields,
        Some(&doc_in),
        None,
        None,
        None,
        None,
        &outer,
        &mut c,
    )
    .unwrap();
    assert_eq!(c.docs, vec![0, 1, 2, 4]);
}

#[test]
fn dismax_phrase_clause_without_pos_input_is_an_error() {
    let (fields, doc) = open_fixture();
    let doc = doc.unwrap();
    let doc_in = doc.open();
    let q = DisjunctionMaxQuery::new(
        [Clause::Phrase(PhraseQuery::new("pos", ["alpha", "beta"]))],
        0.0,
    );
    let mut c = VecCollector::default();
    let err =
        search_disjunction_max_query(&fields, Some(&doc_in), None, None, None, None, &q, &mut c)
            .unwrap_err();
    assert!(matches!(err, Error::MissingPosInput));
}

// `ConstantScoreQuery`/`BoostQuery`. `body`'s known real postings
// (see the dismax/boolean tests above): cat={0,2}, dog={0,1}, bird={1,4}.
//
// **Cross-engine verification scope, decided here rather than adding a new
// Java fixture (see the `differential-testing` skill)**: both wrappers are
// arithmetically trivial compositions over an inner clause whose own scoring
// is *already* cross-engine-verified -- `Clause::Term`/`Clause::Boolean`/
// `Clause::DisjunctionMax` scoring was checked against real Lucene's
// `IndexSearcher`/`TopDocs` output in earlier tasks (`scoring_fixtures.rs`,
// `dismax_query_fixtures.rs` -- see `docs/parity.md`'s `DisjunctionMaxQuery`
// row for the exact fixture and its real recorded scores). `ConstantScore`
// replaces that already-real inner score with a literal constant (no
// arithmetic to get wrong beyond "return `score` verbatim"); `Boost`
// multiplies it by a literal `f32` (one `*`, no order-of-operations
// ambiguity, no norms/idf/tf interaction of its own). Writing a brand-new
// `Gen*.java` generator to prove `x == x` (constant) or `y == a * b` (a
// single multiply of Rust's own `f32`, the same float type and operator
// Java's `float` multiply uses bit-for-bit under IEEE 754) would not
// exercise any Lucene-specific format or algorithm this port could get
// subtly wrong -- unlike BM25's `tfNorm`/`idf` formulas or the dismax
// tie-breaker formula, which *did* need real Lucene ground truth to catch
// a real bug (the BM25 `tfNorm` fix). Instead, these
// tests use `search_term_query_scored`'s already-cross-engine-consistent
// real BM25 score for `body:cat`/`body:dog` at specific docs as the "known
// real" inner score, and assert the wrapped score is exactly that constant,
// or exactly that real score times the boost -- i.e. they verify this
// task's arithmetic against a real (not hand-faked) inner score, just
// without a second Java fixture generator, which would add fixture
// maintenance burden without covering anything this reasoning doesn't
// already cover.

fn real_score(fields: &BlockTreeFields, doc_in: &DocInput<'_>, term: &str, doc_id: i32) -> f32 {
    let mut top = TopDocsCollector::new(10);
    search_term_query_scored(
        fields,
        Some(doc_in),
        None,
        &TermQuery::new("body", term),
        None,
        &mut top,
    )
    .unwrap();
    top.top_docs()
        .iter()
        .find(|h| h.doc_id == doc_id)
        .map(|h| h.score)
        .unwrap()
}

#[test]
fn constant_score_matching_set_equals_inner_matching_set() {
    let (fields, doc) = open_fixture();
    let doc = doc.unwrap();
    let doc_in = doc.open();

    let q = BooleanQuery::new().with_must([Clause::from(ConstantScoreQuery::new(
        TermQuery::new("body", "cat"),
        1.0,
    ))]);
    let mut c = VecCollector::default();
    search_boolean_query(&fields, Some(&doc_in), None, None, None, None, &q, &mut c).unwrap();
    assert_eq!(c.docs, vec![0, 2]);
}

#[test]
fn constant_score_with_a_missing_inner_term_matches_nothing() {
    let (fields, doc) = open_fixture();
    let doc = doc.unwrap();
    let doc_in = doc.open();

    let q = BooleanQuery::new().with_must([Clause::from(ConstantScoreQuery::new(
        TermQuery::new("body", "zzz-missing"),
        7.0,
    ))]);
    let mut c = VecCollector::default();
    search_boolean_query(&fields, Some(&doc_in), None, None, None, None, &q, &mut c).unwrap();
    assert!(c.docs.is_empty());
}

#[test]
fn constant_score_scores_exactly_the_configured_score_regardless_of_inner_score() {
    let (fields, doc) = open_fixture();
    let doc = doc.unwrap();
    let doc_in = doc.open();

    // Real per-doc BM25 scores for cat differ between doc 0 and doc 2 (real
    // Lucene never scores two different docs identically for the same term
    // unless their lengths/freqs coincide) -- proving the constant override
    // discards both, not just one.
    let cat0 = real_score(&fields, &doc_in, "cat", 0);
    let cat2 = real_score(&fields, &doc_in, "cat", 2);
    let constant = 4.25f32;
    assert_ne!(cat0, constant);
    assert_ne!(cat2, constant);

    let q = BooleanQuery::new().with_must([Clause::from(ConstantScoreQuery::new(
        TermQuery::new("body", "cat"),
        constant,
    ))]);
    let mut top = TopDocsCollector::new(10);
    search_boolean_query_scored(
        &fields,
        Some(&doc_in),
        None,
        None,
        None,
        None,
        &q,
        None,
        &mut top,
    )
    .unwrap();
    let score_of = |top: &TopDocsCollector, doc_id: i32| -> f32 {
        top.top_docs()
            .iter()
            .find(|h| h.doc_id == doc_id)
            .map(|h| h.score)
            .unwrap()
    };
    assert_eq!(score_of(&top, 0), constant);
    assert_eq!(score_of(&top, 2), constant);
}

#[test]
fn boost_matching_set_equals_inner_matching_set() {
    let (fields, doc) = open_fixture();
    let doc = doc.unwrap();
    let doc_in = doc.open();

    let q = BooleanQuery::new().with_must([Clause::from(BoostQuery::new(
        TermQuery::new("body", "dog"),
        2.0,
    ))]);
    let mut c = VecCollector::default();
    search_boolean_query(&fields, Some(&doc_in), None, None, None, None, &q, &mut c).unwrap();
    assert_eq!(c.docs, vec![0, 1]);
}

#[test]
fn boost_with_a_missing_inner_term_matches_nothing() {
    let (fields, doc) = open_fixture();
    let doc = doc.unwrap();
    let doc_in = doc.open();

    let q = BooleanQuery::new().with_must([Clause::from(BoostQuery::new(
        TermQuery::new("body", "zzz-missing"),
        2.0,
    ))]);
    let mut c = VecCollector::default();
    search_boolean_query(&fields, Some(&doc_in), None, None, None, None, &q, &mut c).unwrap();
    assert!(c.docs.is_empty());
}

#[test]
fn boost_score_is_exactly_the_inner_real_score_times_boost() {
    let (fields, doc) = open_fixture();
    let doc = doc.unwrap();
    let doc_in = doc.open();

    let dog0 = real_score(&fields, &doc_in, "dog", 0);
    let boost = 2.5f32;

    let q = BooleanQuery::new().with_must([Clause::from(BoostQuery::new(
        TermQuery::new("body", "dog"),
        boost,
    ))]);
    let mut top = TopDocsCollector::new(10);
    search_boolean_query_scored(
        &fields,
        Some(&doc_in),
        None,
        None,
        None,
        None,
        &q,
        None,
        &mut top,
    )
    .unwrap();
    let score0 = top
        .top_docs()
        .iter()
        .find(|h| h.doc_id == 0)
        .map(|h| h.score)
        .unwrap();
    assert!((score0 - dog0 * boost).abs() < 1e-5);
    assert_ne!(score0, dog0, "boost must actually change the score");
}

#[test]
fn constant_score_nested_inside_a_boolean_query_composes_with_other_clauses() {
    // must = [dog, constant_score(cat, 9.0)]: dog={0,1}, cat={0,2},
    // conjunction = {0}; doc 0's total score is dog's own real score plus
    // the constant 9.0 (real Lucene's additive `BooleanScorer`).
    let (fields, doc) = open_fixture();
    let doc = doc.unwrap();
    let doc_in = doc.open();

    let dog0 = real_score(&fields, &doc_in, "dog", 0);
    let constant = 9.0f32;

    let q = BooleanQuery::new().with_must([
        Clause::Term(TermQuery::new("body", "dog")),
        Clause::from(ConstantScoreQuery::new(
            TermQuery::new("body", "cat"),
            constant,
        )),
    ]);
    let mut c = VecCollector::default();
    search_boolean_query(&fields, Some(&doc_in), None, None, None, None, &q, &mut c).unwrap();
    assert_eq!(c.docs, vec![0]);

    let mut top = TopDocsCollector::new(10);
    search_boolean_query_scored(
        &fields,
        Some(&doc_in),
        None,
        None,
        None,
        None,
        &q,
        None,
        &mut top,
    )
    .unwrap();
    let score0 = top
        .top_docs()
        .iter()
        .find(|h| h.doc_id == 0)
        .map(|h| h.score)
        .unwrap();
    assert!((score0 - (dog0 + constant)).abs() < 1e-4);
}

#[test]
fn boost_nested_inside_a_dismax_disjunct_scores_correctly() {
    // dismax([boost(cat, 2.0), term(dog)], tie_breaker=0.0): doc 0 matches
    // both; with tie_breaker 0.0 the winner is whichever disjunct scores
    // higher, and the boosted disjunct's score must be exactly cat0 * 2.0.
    let (fields, doc) = open_fixture();
    let doc = doc.unwrap();
    let doc_in = doc.open();

    let cat0 = real_score(&fields, &doc_in, "cat", 0);
    let dog0 = real_score(&fields, &doc_in, "dog", 0);
    let boost = 2.0f32;
    let boosted_cat0 = cat0 * boost;
    let expected_max = boosted_cat0.max(dog0);

    let q = DisjunctionMaxQuery::new(
        [
            Clause::from(BoostQuery::new(TermQuery::new("body", "cat"), boost)),
            Clause::Term(TermQuery::new("body", "dog")),
        ],
        0.0,
    );
    let mut top = TopDocsCollector::new(10);
    search_disjunction_max_query_scored(
        &fields,
        Some(&doc_in),
        None,
        None,
        None,
        None,
        &q,
        None,
        &mut top,
    )
    .unwrap();
    let score0 = top
        .top_docs()
        .iter()
        .find(|h| h.doc_id == 0)
        .map(|h| h.score)
        .unwrap();
    assert!((score0 - expected_max).abs() < 1e-4);
}

#[test]
fn constant_score_wrapping_a_dismax_query_matches_the_dismax_union_and_scores_fixed() {
    let (fields, doc) = open_fixture();
    let doc = doc.unwrap();
    let doc_in = doc.open();

    let dismax = DisjunctionMaxQuery::new(
        [
            Clause::Term(TermQuery::new("body", "cat")),
            Clause::Term(TermQuery::new("body", "bird")),
        ],
        0.0,
    );
    let constant = 6.0f32;
    let q = BooleanQuery::new().with_must([Clause::from(ConstantScoreQuery::new(
        Clause::DisjunctionMax(Box::new(dismax)),
        constant,
    ))]);
    let mut c = VecCollector::default();
    search_boolean_query(&fields, Some(&doc_in), None, None, None, None, &q, &mut c).unwrap();
    // cat={0,2} union bird={1,4} = {0,1,2,4}.
    assert_eq!(c.docs, vec![0, 1, 2, 4]);

    let mut top = TopDocsCollector::new(10);
    search_boolean_query_scored(
        &fields,
        Some(&doc_in),
        None,
        None,
        None,
        None,
        &q,
        None,
        &mut top,
    )
    .unwrap();
    for doc_id in [0, 1, 2, 4] {
        let score = top
            .top_docs()
            .iter()
            .find(|h| h.doc_id == doc_id)
            .map(|h| h.score)
            .unwrap();
        assert_eq!(score, constant);
    }
}

#[test]
fn boost_wrapping_a_constant_score_query_multiplies_the_constant() {
    // BoostQuery(ConstantScoreQuery(cat, 3.0), 2.0) -- real Lucene composes
    // the two multiplicatively/replacement in that order: matching docs
    // score exactly 3.0 * 2.0 = 6.0.
    let (fields, doc) = open_fixture();
    let doc = doc.unwrap();
    let doc_in = doc.open();

    let inner_constant = 3.0f32;
    let boost = 2.0f32;
    let q = BooleanQuery::new().with_must([Clause::from(BoostQuery::new(
        Clause::from(ConstantScoreQuery::new(
            TermQuery::new("body", "cat"),
            inner_constant,
        )),
        boost,
    ))]);
    let mut c = VecCollector::default();
    search_boolean_query(&fields, Some(&doc_in), None, None, None, None, &q, &mut c).unwrap();
    assert_eq!(c.docs, vec![0, 2]);

    let mut top = TopDocsCollector::new(10);
    search_boolean_query_scored(
        &fields,
        Some(&doc_in),
        None,
        None,
        None,
        None,
        &q,
        None,
        &mut top,
    )
    .unwrap();
    for doc_id in [0, 2] {
        let score = top
            .top_docs()
            .iter()
            .find(|h| h.doc_id == doc_id)
            .map(|h| h.score)
            .unwrap();
        assert_eq!(score, inner_constant * boost);
    }
}

/// Differential proof for [`search_term_query_scored_maxscore`], against
/// the checked-in real-Lucene fixture's `big`/`"everywhere"` term
/// (`docFreq == 300`: one full 256-doc level-0 block with real,
/// Java-written impacts -- `field.big.term.everywhere.impacts.results` in
/// `manifest.properties` -- plus a 44-doc tail; see
/// `crates/lucene-codecs/tests/blocktree_fixtures.rs`'s
/// `big_field_impacts_match_real_lucene_impacts_enum`, which already
/// proves those decoded impacts match real Lucene byte-for-byte). Opens
/// this field's *real* per-doc norms (`_0.nvd`/`_0.nvm`) rather than
/// passing `norms: None` -- deliberately, because the impacts'
/// `(freq, norm)` bound is only a valid upper bound against the score
/// formula that actually consumes those same real norm bytes; comparing
/// it against the unrelated `UNNORMED_FIELD_LENGTH` fallback score
/// instead would not be a documented approximation but an unproven (and
/// in general unsound) mix, so this test doesn't exercise that
/// combination.
///
/// For several `top_n` values asserts:
/// - [`search_term_query_scored_maxscore`]'s `TopDocsCollector::top_docs()`
///   is *exactly* [`search_term_query_scored`] (the eager path)'s, proving
///   the skip never changes the result;
/// - for small `top_n` (1, 5, 50), at least one real block-skip actually
///   happened (via `test_only_maxscore_block_skip_counter`, reset/read
///   around each call) -- ruling out a vacuously-passing test where the
///   skip branch is simply never taken. Every doc in this fixture's block
///   is one of only four distinct `(freq, norm)` combinations, cycling
///   `1,2,3,4,1,2,3,...`, so the single best-scoring combination is
///   reached within the first few docs and the whole rest of the block
///   (a few hundred docs) becomes safely skippable from there on;
/// - for `top_n = 300` (the full `docFreq`: every doc must eventually be
///   collected, so the collector is never "full" until after the very
///   last doc), the skip count is 0 -- proving the threshold check itself
///   is conservative, not "sometimes skips, still happens to match".
#[test]
fn maxscore_lazy_path_matches_eager_path_on_real_fixture_and_actually_skips_blocks() {
    let (fields, doc) = open_fixture();
    let doc = doc.unwrap();
    let doc_in = doc.open();

    // Open real per-doc norms for "big" (see this test's doc comment for
    // why `norms: None` wouldn't be a sound comparison here).
    let dir = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/blocktree_index/"
    );
    let manifest = std::fs::read_to_string(format!("{dir}manifest.properties"))
        .expect("run fixtures generator first (GenBlockTree)");
    let get = |key: &str| -> String {
        manifest
            .lines()
            .find_map(|l| l.strip_prefix(&format!("{key}=")))
            .unwrap_or_else(|| panic!("manifest key {key} missing"))
            .to_string()
    };
    let id_hex = get("id_hex");
    let mut id = [0u8; 16];
    for (i, slot) in id.iter_mut().enumerate() {
        *slot = u8::from_str_radix(&id_hex[i * 2..i * 2 + 2], 16).unwrap();
    }
    let max_doc: i32 = get("max_doc").parse().unwrap();
    let fnm = std::fs::read(format!("{dir}{}.raw", get("fnm_file_name"))).expect("read .fnm");
    let field_infos = lucene_codecs::field_infos::parse(&fnm, &id, "").expect("parse .fnm");
    let big_field_number = field_infos
        .fields
        .iter()
        .find(|f| f.name == "big")
        .expect("fixture has a \"big\" field")
        .number;

    let nvm = std::fs::read(format!("{dir}_0.nvm")).expect("read _0.nvm");
    let nvd = std::fs::read(format!("{dir}_0.nvd")).expect("read _0.nvd");
    let (_, parsed_norms) = lucene_codecs::norms::parse_meta(&nvm, &id, "").expect("parse .nvm");
    let entry = *parsed_norms
        .entry(big_field_number)
        .expect("\"big\" field has a norms entry");
    let norms = FieldNorms::open(&nvd, entry, max_doc, None).expect("FieldNorms::open");

    let query = TermQuery::new("big", b"everywhere".as_slice());

    for &top_n in &[1usize, 5, 50, 300] {
        let mut eager = TopDocsCollector::new(top_n);
        search_term_query_scored(
            &fields,
            Some(&doc_in),
            None,
            &query,
            Some(&norms),
            &mut eager,
        )
        .expect("eager search");

        test_only_scored_docs_counter::reset();
        let mut lazy = TopDocsCollector::new(top_n);
        search_term_query_scored_maxscore(
            &fields,
            Some(&doc_in),
            None,
            &query,
            Some(&norms),
            &mut lazy,
        )
        .expect("maxscore search");
        let collected = test_only_scored_docs_counter::count();

        assert_eq!(
            eager.top_docs(),
            lazy.top_docs(),
            "top_{top_n} must match exactly between eager and maxscore paths"
        );

        // docFreq 300 is one full block plus a 44-doc tail, and Lucene's
        // `BatchScoreBulkScorer` scores a decoded block whole -- there is
        // no block here that pruning could leave undecoded (the tail
        // carries no impacts). What pruning does save is every document
        // that cannot beat the queue's bottom being handed to the
        // collector (`score >= minCompetitiveScore`); a multi-block term
        // proves the block skipping itself, see
        // `maxscore_term_path_skips_whole_blocks_on_a_level1_term`.
        if top_n < 300 {
            assert!(
                collected < 300,
                "top_{top_n}: the threshold must keep uncompetitive documents out \
                     of the collector (collected {collected} of 300)"
            );
        } else {
            assert_eq!(
                collected, 300,
                "top_{top_n} == the full docFreq: the collector is never full \
                     until the very last doc, so every document must reach it"
            );
        }
    }
}

/// `ScoreMode.isExhaustive()`'s whole purpose: a collector in
/// [`collector::ScoreMode::Complete`] promises an exact `totalHits`, so the
/// scorer must not skip a single block. Proven with the skip counter rather
/// than with the result -- the results are necessarily identical either
/// way, and it is exactly the kind of gate that can go dead without any
/// assertion about output noticing.
#[test]
fn an_exhaustive_score_mode_collector_disables_every_maxscore_block_skip() {
    let (fields, doc) = open_fixture();
    let doc = doc.unwrap();
    let doc_in = doc.open();
    // `l1term` (8,250 documents over 32+ blocks with varying impacts)
    // rather than the 300-document `everywhere`: Lucene's block-at-a-time
    // scorer only skips whole blocks, and one full block plus an
    // impact-less tail leaves it nothing to skip.
    let query = TermQuery::new("l1", "l1term");

    test_only_maxscore_block_skip_counter::reset();
    let mut pruning = TopDocsCollector::new(1);
    search_term_query_scored_maxscore(&fields, Some(&doc_in), None, &query, None, &mut pruning)
        .expect("maxscore search");
    let pruning_skips = test_only_maxscore_block_skip_counter::count();
    assert!(
        pruning_skips > 0,
        "the pruning baseline must actually skip, or this test proves nothing"
    );
    assert_eq!(pruning.score_mode(), collector::ScoreMode::TopScores);
    assert_eq!(
        pruning.total_hits().relation,
        collector::TotalHitsRelation::GreaterThanOrEqualTo,
        "a pruned search cannot claim an exact count"
    );

    test_only_maxscore_block_skip_counter::reset();
    let mut exhaustive = TopDocsCollector::with_total_hits_threshold(1, u64::MAX);
    assert!(exhaustive.score_mode().is_exhaustive());
    search_term_query_scored_maxscore(&fields, Some(&doc_in), None, &query, None, &mut exhaustive)
        .expect("maxscore search");
    assert_eq!(
        test_only_maxscore_block_skip_counter::count(),
        0,
        "no block may be skipped for an exhaustive collector"
    );
    assert_eq!(
        exhaustive.total_hits().relation,
        collector::TotalHitsRelation::EqualTo
    );
    assert_eq!(
        exhaustive.total_hits().value,
        8250,
        "the fixture's \"l1\"/\"l1term\" term has docFreq 8250, all of them counted"
    );
    // And the top hit is the same either way -- pruning changed the work,
    // not the answer.
    assert_eq!(pruning.top_docs(), exhaustive.top_docs());
}

#[test]
fn maxscore_returns_immediately_for_unknown_field_or_term() {
    let (fields, doc) = open_fixture();
    let doc = doc.unwrap();
    let doc_in = doc.open();

    let mut c = TopDocsCollector::new(5);
    search_term_query_scored_maxscore(
        &fields,
        Some(&doc_in),
        None,
        &TermQuery::new("nonexistent_field", "x"),
        None,
        &mut c,
    )
    .unwrap();
    assert!(c.top_docs().is_empty());

    let mut c = TopDocsCollector::new(5);
    search_term_query_scored_maxscore(
        &fields,
        Some(&doc_in),
        None,
        &TermQuery::new("body", "no_such_term_anywhere"),
        None,
        &mut c,
    )
    .unwrap();
    assert!(c.top_docs().is_empty());
}

#[test]
fn maxscore_falls_back_to_eager_path_when_doc_in_is_none() {
    // `body` is a multi-doc field/term, so both the eager and maxscore
    // paths require `.doc` bytes -- confirms the `doc_in == None`
    // fallback branch produces the same (error) outcome as calling the
    // eager path directly, rather than panicking or silently succeeding.
    let (fields, _doc) = open_fixture();
    let query = TermQuery::new("body", "cat");

    let mut eager = TopDocsCollector::new(5);
    let eager_result = search_term_query_scored(&fields, None, None, &query, None, &mut eager);

    let mut lazy = TopDocsCollector::new(5);
    let lazy_result =
        search_term_query_scored_maxscore(&fields, None, None, &query, None, &mut lazy);

    assert_eq!(eager_result.is_err(), lazy_result.is_err());
    if eager_result.is_ok() {
        assert_eq!(eager.top_docs(), lazy.top_docs());
    }
}

#[test]
fn maxscore_falls_back_to_eager_path_for_a_singleton_docfreq_one_term() {
    // "id"/"id2" is a `docFreq == 1` singleton (pulsed into the term
    // dictionary, no `.doc` bytes at all) -- exercises the `doc_freq <=
    // 1` fallback branch specifically, distinct from the `doc_in ==
    // None` branch above (here `doc_in` is `Some`).
    let (fields, doc) = open_fixture();
    let doc = doc.unwrap();
    let doc_in = doc.open();
    let query = TermQuery::new("id", "id2");

    let mut eager = TopDocsCollector::new(5);
    search_term_query_scored(&fields, Some(&doc_in), None, &query, None, &mut eager).unwrap();

    test_only_maxscore_block_skip_counter::reset();
    let mut lazy = TopDocsCollector::new(5);
    search_term_query_scored_maxscore(&fields, Some(&doc_in), None, &query, None, &mut lazy)
        .unwrap();

    assert_eq!(eager.top_docs(), lazy.top_docs());
    assert_eq!(
        test_only_maxscore_block_skip_counter::count(),
        0,
        "a singleton term has no blocks to skip"
    );
}

#[test]
fn maxscore_falls_back_to_unnormed_field_length_when_norms_is_none() {
    // Confirms the `norms: None` -> `UNNORMED_FIELD_LENGTH` branch is
    // exercised, that skipping actually happens (not vacuously passing
    // because nothing gets pruned), and -- the specific bug this test
    // caught during review -- that the block-skip bound is computed
    // against the *same* `UNNORMED_FIELD_LENGTH` scoring formula the
    // real per-doc score below uses, not against the field's real
    // on-wire norm bytes (which would be an unsound mix: this fixture's
    // "big"/"everywhere" impacts were written against real per-doc
    // norms, so blindly feeding them into `max_score_for_impacts` while
    // scoring downstream with `UNNORMED_FIELD_LENGTH` produced an
    // under-estimated bound that could -- and did -- skip a block
    // containing a doc that belonged in the top-K).
    let (fields, doc) = open_fixture();
    let doc = doc.unwrap();
    let doc_in = doc.open();
    // `everywhere` is the term whose impacts were written against real
    // per-doc norms (the unsound-mix scenario); `l1term` spans 32+ blocks,
    // so it also proves the unnormed bound still lets whole blocks go.
    for (field, term) in [("big", "everywhere"), ("l1", "l1term")] {
        let query = TermQuery::new(field, term);

        let mut eager = TopDocsCollector::new(5);
        search_term_query_scored(&fields, Some(&doc_in), None, &query, None, &mut eager).unwrap();

        test_only_maxscore_block_skip_counter::reset();
        let mut lazy = TopDocsCollector::new(5);
        search_term_query_scored_maxscore(&fields, Some(&doc_in), None, &query, None, &mut lazy)
            .unwrap();

        assert_eq!(eager.top_docs(), lazy.top_docs(), "{field}:{term}");
        if term == "l1term" {
            assert!(
                test_only_maxscore_block_skip_counter::count() > 0,
                "l1term's 8,250 documents span 32+ level-0 blocks, so some \
                     block should be skippable once the top-5 threshold is reached"
            );
        }
    }
}

/// Differential proof for
/// [`search_boolean_query_scored_maxscore`], analogous to
/// `maxscore_lazy_path_matches_eager_path_on_real_fixture_and_actually_skips_blocks`
/// one level up: a three-clause pure-SHOULD `BooleanQuery` over this
/// fixture's real Lucene-written data (`big`/`"everywhere"`, `docFreq ==
/// 300`, spanning a real full level-0 block plus a tail; `body`/`"cat"`
/// and `body`/`"dog"`, `docFreq == 2` each). For many `top_n` values,
/// asserts [`search_boolean_query_scored_maxscore`]'s
/// `TopDocsCollector::top_docs()` is byte-identical to
/// [`search_boolean_query_scored`] (the exhaustive path), proving the
/// skip logic never changes the result.
#[test]
fn boolean_maxscore_lazy_path_matches_eager_path_on_real_fixture() {
    let (fields, doc) = open_fixture();
    let doc = doc.unwrap();
    let doc_in = doc.open();

    let query = BooleanQuery::new().with_should([
        Clause::from(TermQuery::new("big", b"everywhere".as_slice())),
        Clause::from(TermQuery::new("body", "cat")),
        Clause::from(TermQuery::new("body", "dog")),
    ]);

    for &top_n in &[1usize, 2, 5, 20, 300] {
        let mut eager = TopDocsCollector::new(top_n);
        search_boolean_query_scored(
            &fields,
            Some(&doc_in),
            None,
            None,
            None,
            None,
            &query,
            None,
            &mut eager,
        )
        .expect("eager search");

        let mut lazy = TopDocsCollector::new(top_n);
        search_boolean_query_scored_maxscore(
            &fields,
            Some(&doc_in),
            None,
            None,
            None,
            None,
            &query,
            None,
            &mut lazy,
        )
        .expect("maxscore search");

        assert_eq!(
            eager.top_docs(),
            lazy.top_docs(),
            "top_{top_n} must match exactly between eager and maxscore boolean paths"
        );
    }
}

/// Counter-based proof (analogous to
/// `maxscore_lazy_path_matches_eager_path_on_real_fixture_and_actually_skips_blocks`'s
/// own skip-count assertion) that
/// [`search_boolean_query_scored_maxscore`] doesn't just happen to match
/// the eager path's output -- it genuinely skips real per-clause level-0
/// block decode for the same three-clause query above, reusing
/// [`test_only_maxscore_block_skip_counter`] (shared with the
/// single-term MAXSCORE path, since both increment the same counter at
/// the same kind of skip site: `advance()`ing straight past a level-0
/// block's last doc without decoding the rest of it).
#[test]
fn test_only_boolean_maxscore_block_skip_counter_records_real_skips() {
    let (fields, doc) = open_fixture();
    let doc = doc.unwrap();
    let doc_in = doc.open();

    let query = BooleanQuery::new().with_should([
        Clause::from(TermQuery::new("big", b"everywhere".as_slice())),
        Clause::from(TermQuery::new("body", "cat")),
        Clause::from(TermQuery::new("body", "dog")),
    ]);

    // A small top_n reaches its final threshold quickly relative to
    // "big"'s 300-doc postings list, making at least one of its blocks
    // (out of several) provably unable to beat that threshold once
    // combined with the other clauses' global bounds.
    test_only_maxscore_block_skip_counter::reset();
    let mut lazy = TopDocsCollector::new(1);
    search_boolean_query_scored_maxscore(
        &fields,
        Some(&doc_in),
        None,
        None,
        None,
        None,
        &query,
        None,
        &mut lazy,
    )
    .expect("maxscore search");
    assert!(
        test_only_maxscore_block_skip_counter::count() > 0,
        "top_1 should make at least one clause's block provably \
             uncompetitive against the other clauses' global bounds"
    );

    // The full docFreq: the collector is never full until the very last
    // candidate doc, so nothing should be skippable -- same conservative
    // check the single-term differential test above makes.
    test_only_maxscore_block_skip_counter::reset();
    let mut lazy_full = TopDocsCollector::new(9000);
    search_boolean_query_scored_maxscore(
        &fields,
        Some(&doc_in),
        None,
        None,
        None,
        None,
        &query,
        None,
        &mut lazy_full,
    )
    .expect("maxscore search");
    assert_eq!(
        test_only_maxscore_block_skip_counter::count(),
        0,
        "top_n covering every possible hit should never need to skip anything"
    );
}

#[test]
fn boolean_maxscore_falls_back_to_eager_path_when_must_clause_present() {
    // A non-empty `must` disqualifies the fast path entirely (see this
    // function's doc comment) -- confirms the fallback produces exactly
    // the eager path's output rather than silently mishandling `must`.
    let (fields, doc) = open_fixture();
    let doc = doc.unwrap();
    let doc_in = doc.open();

    let query = BooleanQuery::new()
        .with_must([Clause::from(TermQuery::new("body", "cat"))])
        .with_should([Clause::from(TermQuery::new("body", "dog"))]);

    let mut eager = TopDocsCollector::new(5);
    search_boolean_query_scored(
        &fields,
        Some(&doc_in),
        None,
        None,
        None,
        None,
        &query,
        None,
        &mut eager,
    )
    .unwrap();

    let mut lazy = TopDocsCollector::new(5);
    search_boolean_query_scored_maxscore(
        &fields,
        Some(&doc_in),
        None,
        None,
        None,
        None,
        &query,
        None,
        &mut lazy,
    )
    .unwrap();

    assert_eq!(eager.top_docs(), lazy.top_docs());
}

#[test]
fn boolean_maxscore_falls_back_to_eager_path_for_minimum_should_match_above_one() {
    let (fields, doc) = open_fixture();
    let doc = doc.unwrap();
    let doc_in = doc.open();

    let query = BooleanQuery::new()
        .with_should([
            Clause::from(TermQuery::new("body", "cat")),
            Clause::from(TermQuery::new("body", "dog")),
            Clause::from(TermQuery::new("body", "bird")),
        ])
        .with_minimum_should_match(2);

    let mut eager = TopDocsCollector::new(5);
    search_boolean_query_scored(
        &fields,
        Some(&doc_in),
        None,
        None,
        None,
        None,
        &query,
        None,
        &mut eager,
    )
    .unwrap();

    let mut lazy = TopDocsCollector::new(5);
    search_boolean_query_scored_maxscore(
        &fields,
        Some(&doc_in),
        None,
        None,
        None,
        None,
        &query,
        None,
        &mut lazy,
    )
    .unwrap();

    assert_eq!(eager.top_docs(), lazy.top_docs());
}

#[test]
fn boolean_maxscore_falls_back_to_eager_path_for_nested_boolean_clause() {
    // A nested `Clause::Boolean` isn't `Clause::Term`, so it disqualifies
    // the fast path (see this function's doc comment) -- confirms the
    // fallback still produces the exhaustive path's exact result.
    let (fields, doc) = open_fixture();
    let doc = doc.unwrap();
    let doc_in = doc.open();

    let query = BooleanQuery::new().with_should([
        Clause::from(TermQuery::new("body", "cat")),
        Clause::Boolean(Box::new(
            BooleanQuery::new().with_should([Clause::from(TermQuery::new("body", "dog"))]),
        )),
    ]);

    let mut eager = TopDocsCollector::new(5);
    search_boolean_query_scored(
        &fields,
        Some(&doc_in),
        None,
        None,
        None,
        None,
        &query,
        None,
        &mut eager,
    )
    .unwrap();

    let mut lazy = TopDocsCollector::new(5);
    search_boolean_query_scored_maxscore(
        &fields,
        Some(&doc_in),
        None,
        None,
        None,
        None,
        &query,
        None,
        &mut lazy,
    )
    .unwrap();

    assert_eq!(eager.top_docs(), lazy.top_docs());
}

#[test]
fn boolean_maxscore_falls_back_to_eager_path_for_singleton_docfreq_one_clause() {
    // "id"/"id2" has `docFreq == 1` (pulsed, no `.doc` bytes) -- exercises
    // the per-clause `doc_freq <= 1` fallback branch.
    let (fields, doc) = open_fixture();
    let doc = doc.unwrap();
    let doc_in = doc.open();

    let query = BooleanQuery::new().with_should([
        Clause::from(TermQuery::new("body", "cat")),
        Clause::from(TermQuery::new("id", "id2")),
    ]);

    let mut eager = TopDocsCollector::new(5);
    search_boolean_query_scored(
        &fields,
        Some(&doc_in),
        None,
        None,
        None,
        None,
        &query,
        None,
        &mut eager,
    )
    .unwrap();

    let mut lazy = TopDocsCollector::new(5);
    search_boolean_query_scored_maxscore(
        &fields,
        Some(&doc_in),
        None,
        None,
        None,
        None,
        &query,
        None,
        &mut lazy,
    )
    .unwrap();

    assert_eq!(eager.top_docs(), lazy.top_docs());
}

#[test]
fn boolean_maxscore_falls_back_to_eager_path_when_doc_in_is_none() {
    let (fields, _doc) = open_fixture();
    let query = BooleanQuery::new().with_should([
        Clause::from(TermQuery::new("body", "cat")),
        Clause::from(TermQuery::new("body", "dog")),
    ]);

    let mut eager = TopDocsCollector::new(5);
    let eager_result = search_boolean_query_scored(
        &fields, None, None, None, None, None, &query, None, &mut eager,
    );

    let mut lazy = TopDocsCollector::new(5);
    let lazy_result = search_boolean_query_scored_maxscore(
        &fields, None, None, None, None, None, &query, None, &mut lazy,
    );

    assert_eq!(eager_result.is_err(), lazy_result.is_err());
    if eager_result.is_ok() {
        assert_eq!(eager.top_docs(), lazy.top_docs());
    }
}

/// `search_boolean_query_scored_maxscore` is a delegate for
/// `search_boolean_query_scored`, so it must agree with the plain scored
/// entry point on *every* shape. Two shapes in the matrix are
/// easy to miss: a clause naming an absent field and a clause naming an
/// absent term. The disjunction bulk scorer *accepts* both (it drops the
/// clause from the union and keeps going).
#[test]
fn boolean_maxscore_agrees_with_the_plain_scored_path_on_every_shape() {
    let (fields, doc) = open_fixture();
    let doc = doc.unwrap();
    let doc_in = doc.open();

    let shapes: Vec<(&str, BooleanQuery)> = vec![
        (
            "pure should, both terms present",
            BooleanQuery::new().with_should([
                Clause::from(TermQuery::new("body", "cat")),
                Clause::from(TermQuery::new("body", "dog")),
            ]),
        ),
        (
            "a clause naming an absent field",
            BooleanQuery::new().with_should([
                Clause::from(TermQuery::new("body", "cat")),
                Clause::from(TermQuery::new("no_such_field", "cat")),
            ]),
        ),
        (
            "a clause naming an absent term in a present field",
            BooleanQuery::new().with_should([
                Clause::from(TermQuery::new("body", "cat")),
                Clause::from(TermQuery::new("body", "no_such_term")),
            ]),
        ),
        (
            "every clause absent",
            BooleanQuery::new().with_should([
                Clause::from(TermQuery::new("body", "no_such_term")),
                Clause::from(TermQuery::new("no_such_field", "cat")),
            ]),
        ),
        (
            "a must clause (the disjunction gate declines)",
            BooleanQuery::new()
                .with_must([Clause::from(TermQuery::new("body", "cat"))])
                .with_should([Clause::from(TermQuery::new("body", "dog"))]),
        ),
        (
            "a filter clause (c11's fourth list)",
            BooleanQuery::new()
                .with_should([Clause::from(TermQuery::new("body", "cat"))])
                .with_filter([Clause::from(TermQuery::new("body", "dog"))]),
        ),
        (
            "a must_not clause",
            BooleanQuery::new()
                .with_should([Clause::from(TermQuery::new("body", "cat"))])
                .with_must_not([Clause::from(TermQuery::new("body", "dog"))]),
        ),
        (
            "a pulsed docFreq == 1 clause",
            BooleanQuery::new().with_should([
                Clause::from(TermQuery::new("body", "cat")),
                Clause::from(TermQuery::new("id", "id2")),
            ]),
        ),
        (
            "a nested boolean clause",
            BooleanQuery::new().with_should([
                Clause::from(TermQuery::new("body", "cat")),
                Clause::Boolean(Box::new(
                    BooleanQuery::new().with_should([Clause::from(TermQuery::new("body", "dog"))]),
                )),
            ]),
        ),
        (
            "a 300-document clause that spans real level-0 blocks",
            BooleanQuery::new().with_should([
                Clause::from(TermQuery::new("big", b"everywhere".as_slice())),
                Clause::from(TermQuery::new("body", "cat")),
            ]),
        ),
    ];

    let mut any_hits = false;
    for (name, query) in &shapes {
        for top_n in [1usize, 3, 10, 50] {
            let mut eager = TopDocsCollector::new(top_n);
            search_boolean_query_scored(
                &fields,
                Some(&doc_in),
                None,
                None,
                None,
                None,
                query,
                None,
                &mut eager,
            )
            .unwrap();

            let mut delegated = TopDocsCollector::new(top_n);
            search_boolean_query_scored_maxscore(
                &fields,
                Some(&doc_in),
                None,
                None,
                None,
                None,
                query,
                None,
                &mut delegated,
            )
            .unwrap();

            any_hits |= !eager.top_docs().is_empty();
            assert_eq!(
                eager.top_docs(),
                delegated.top_docs(),
                "shape {name:?} at top_n={top_n} disagreed"
            );
        }
    }
    assert!(
        any_hits,
        "the matrix must actually produce hits or it proves nothing"
    );
}

#[test]
fn search_term_query_scored_with_similarity_using_defaults_matches_hardcoded_path() {
    // Task #214 regression proof: Bm25Params::default() through the new
    // parameterized path must reproduce `search_term_query_scored`'s
    // hardcoded-default scores byte-for-byte.
    let (fields, doc) = open_fixture();
    let doc = doc.unwrap();
    let doc_in = doc.open();
    let query = TermQuery::new("body", "cat");

    let mut default_path = TopDocsCollector::new(10);
    search_term_query_scored(
        &fields,
        Some(&doc_in),
        None,
        &query,
        None,
        &mut default_path,
    )
    .unwrap();

    let mut with_similarity = TopDocsCollector::new(10);
    search_term_query_scored_with_similarity(
        &fields,
        Some(&doc_in),
        None,
        &query,
        None,
        similarity::Bm25Params::default(),
        &mut with_similarity,
    )
    .unwrap();

    assert_eq!(default_path.top_docs(), with_similarity.top_docs());
    assert!(
        !default_path.top_docs().is_empty(),
        "test fixture must actually produce hits for this proof to be meaningful"
    );
}

#[test]
fn search_term_query_scored_with_similarity_using_different_k1_b_changes_scores() {
    // A genuinely different k1/b must produce a measurably different,
    // correctly-computed score -- computed by hand from the same BM25
    // formula `similarity.rs`'s doc comment documents, not just "some
    // different number".
    let (fields, doc) = open_fixture();
    let doc = doc.unwrap();
    let doc_in = doc.open();
    let query = TermQuery::new("body", "cat");

    let mut default_path = TopDocsCollector::new(10);
    search_term_query_scored(
        &fields,
        Some(&doc_in),
        None,
        &query,
        None,
        &mut default_path,
    )
    .unwrap();

    let custom_params = similarity::Bm25Params::new(2.0, 0.5).expect("in range");
    let mut with_similarity = TopDocsCollector::new(10);
    search_term_query_scored_with_similarity(
        &fields,
        Some(&doc_in),
        None,
        &query,
        None,
        custom_params,
        &mut with_similarity,
    )
    .unwrap();

    assert!(!default_path.top_docs().is_empty());
    assert_eq!(
        default_path.top_docs().len(),
        with_similarity.top_docs().len()
    );

    for (default_hit, custom_hit) in default_path
        .top_docs()
        .iter()
        .zip(with_similarity.top_docs().iter())
    {
        assert_eq!(default_hit.doc_id, custom_hit.doc_id);
        // Hand-computed expected score using `similarity::score_with_params`
        // directly (same formula the production path calls), with
        // UNNORMED_FIELD_LENGTH for both field-length terms since this
        // fixture is opened without norms (`None` passed above).
        let field_terms = fields.field("body").unwrap();
        let stats = field_terms.seek_exact(b"cat").unwrap();
        let doc_freqs = term_doc_freqs(&fields, Some(&doc_in), None, &query).unwrap();
        let freq = doc_freqs
            .iter()
            .find(|&&(d, _)| d == default_hit.doc_id)
            .unwrap()
            .1;
        let expected = similarity::score_with_params(
            stats.doc_freq as i64,
            field_terms.doc_count as i64,
            freq as f32,
            similarity::UNNORMED_FIELD_LENGTH,
            similarity::UNNORMED_FIELD_LENGTH,
            custom_params,
        );
        assert!(
            (custom_hit.score - expected).abs() < 1e-5,
            "doc={} got={} expected={}",
            custom_hit.doc_id,
            custom_hit.score,
            expected
        );
        assert!(
            (custom_hit.score - default_hit.score).abs() > 1e-3,
            "different k1/b must measurably change the score: default={} custom={}",
            default_hit.score,
            custom_hit.score
        );
    }
}

#[test]
fn search_term_query_scored_with_similarity_missing_field_yields_no_hits() {
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let mut collector = TopDocsCollector::new(10);
    search_term_query_scored_with_similarity(
        &fields,
        doc_in.as_ref(),
        None,
        &TermQuery::new("no_such_field", "cat"),
        None,
        similarity::Bm25Params::new(2.0, 0.5).expect("in range"),
        &mut collector,
    )
    .unwrap();
    assert!(collector.top_docs().is_empty());
}

#[test]
fn search_term_query_scored_with_similarity_missing_term_yields_no_hits() {
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let mut collector = TopDocsCollector::new(10);
    search_term_query_scored_with_similarity(
        &fields,
        doc_in.as_ref(),
        None,
        &TermQuery::new("body", "no_such_term_at_all"),
        None,
        similarity::Bm25Params::new(2.0, 0.5).expect("in range"),
        &mut collector,
    )
    .unwrap();
    assert!(collector.top_docs().is_empty());
}
