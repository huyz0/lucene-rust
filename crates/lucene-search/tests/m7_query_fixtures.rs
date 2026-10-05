//! M7's query and scoring half, differentially against Lucene 10.5.0:
//! `fixtures/src/GenM7Queries.java`.
//!
//! One three-segment, index-sorted index with deletions, searched with every
//! query M7 brings into the scorer tree -- synonyms, BM25F, n-gram phrases,
//! explicit phrase positions, multi-phrases, the multi-term rewrite methods,
//! blended terms, Indri, log-odds fusion, Bayesian calibration, doc-values and
//! index-sort ranges, multi-dimensional and 4/16-byte points, vector
//! similarity thresholds and KNN as a clause -- each under BM25 and, for the
//! scoring ones, `ClassicSimilarity` and a DFR similarity. Every top-20 hit
//! and its score bits must be Java's.

mod m7grammar;

use std::collections::HashMap;

use lucene_search::directory_reader::DirectoryReader;
use lucene_search::field_norms::FieldNorms;
use lucene_search::multi_segment::{
    search_boolean_query_multi_segment_with_deadline,
    search_boolean_query_multi_segment_with_similarity, OpenSegment,
};
use lucene_search::query::Clause;
use lucene_store::FsDirectory;
use m7grammar::{knn_clause, query, sim, Ctx, VectorFiles, QVEC};

fn data(path: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/data")
        .join(path)
}

fn want(hits: &str) -> Vec<(i32, u32)> {
    hits.split(',')
        .filter(|s| !s.is_empty())
        .map(|h| {
            let (d, b) = h.split_once(':').unwrap();
            (d.parse().unwrap(), u32::from_str_radix(b, 16).unwrap())
        })
        .collect()
}

#[test]
fn m7_queries_match_lucene_bit_for_bit() {
    let cases = check_fixture("m7_queries_index", &QVEC, 20, true);
    assert!(cases >= 200, "{cases} searches");
}

/// The vector queries over graphs large enough for filtered
/// (`FilteredHnswGraphSearcher`), patient and seeded walks to happen.
#[test]
fn m7_knn_queries_match_lucene_bit_for_bit() {
    let cases = check_fixture("m7_knn_index", &QVEC8, 50, false);
    assert!(cases >= 16, "{cases} searches");
}

/// Doc-values ranges over fields with a skip index (`SkipBlockRangeIterator`,
/// `DocValuesRangeIterator`), several skip blocks a segment.
#[test]
fn m7_doc_values_skip_ranges_match_lucene_bit_for_bit() {
    // The ranges must run on the skip-index path, not the plain-column one.
    let reader = DirectoryReader::open(&FsDirectory::open(data("m7_dv_index"))).unwrap();
    for seg in reader.segment_readers() {
        for field in ["sk", "msk"] {
            let number = seg.field_infos().field_by_name(field).unwrap().number;
            assert!(
                seg.doc_values_skip_index(number).unwrap().is_some(),
                "{field}"
            );
        }
        let body = seg.field_infos().field_by_name("body").unwrap().number;
        assert!(seg.doc_values_skip_index(body).unwrap().is_none());
    }
    let cases = check_fixture("m7_dv_index", &QVEC, 50, false);
    assert!(cases >= 9, "{cases} searches");
}

/// `GenM7Queries.QVEC8`.
const QVEC8: [f32; 8] = [0.1, -0.3, 0.25, 0.6, -0.05, 0.4, -0.2, 0.15];

/// Runs every search `name`'s `searches.tsv` records and fails on the first
/// difference from Lucene's hits; returns how many ran.
fn check_fixture(name: &str, qvec: &[f32], top: usize, estimator: bool) -> usize {
    let dir = data(name);
    let text = std::fs::read_to_string(dir.join("searches.tsv"))
        .expect("run scripts/gen-fixtures.sh --only GenM7Queries");
    let reader = DirectoryReader::open(&FsDirectory::open(&dir)).unwrap();
    assert!(reader.segment_readers().len() >= 2);
    assert!(
        reader
            .segment_readers()
            .iter()
            .filter(|s| s.live_docs().is_some())
            .count()
            >= 2
    );
    let mut opened = reader.open_segments().unwrap();
    opened.open_points().unwrap();
    let segments: Vec<OpenSegment<'_>> = opened.as_open_segments();
    let owned =
        reader.field_norms_by_field(&["body".to_string(), "title".to_string(), "gram".to_string()]);
    let norms: Vec<Option<&HashMap<String, FieldNorms<'_>>>> = owned.iter().map(Some).collect();
    let vectors = VectorFiles::read(&dir, &reader);
    let knn = |op: &str, field: &str, arg: &str, clause: Option<Clause>| -> Option<Clause> {
        knn_clause(
            op, field, arg, clause, &reader, &segments, &norms, &vectors, qvec,
        )
    };
    let ctx = Ctx { knn: &knn };

    let (mut cases, mut skipped, mut failures) = (0, 0, Vec::new());
    for line in text.lines() {
        let [name, q, hits] = line.split('\t').collect::<Vec<_>>()[..] else {
            panic!("bad line {line}");
        };
        let Some(bq) = query(q, &ctx) else {
            skipped += 1;
            continue;
        };
        let similarity = sim(name);
        let got = match search_boolean_query_multi_segment_with_similarity(
            &segments,
            &bq,
            &norms,
            top,
            similarity.as_ref(),
        ) {
            Ok(got) => got,
            Err(e) => {
                failures.push(format!("{name} [{q}]: {e}"));
                continue;
            }
        };
        let got: Vec<(i32, u32)> = got.iter().map(|h| (h.doc_id, h.score.to_bits())).collect();
        cases += 1;
        let want = want(hits);
        if got != want {
            failures.push(format!("{name} [{q}]\n  rust {got:x?}\n  java {want:x?}"));
        }
        if name == "bm25" {
            // `TimeLimitingBulkScorer`: a deadline that never comes changes
            // nothing; one already past stops before the first window.
            let far = std::time::Instant::now() + std::time::Duration::from_secs(3600);
            let (timed, cut) =
                search_boolean_query_multi_segment_with_deadline(&segments, &bq, &norms, top, far)
                    .unwrap();
            let timed: Vec<(i32, u32)> = timed
                .iter()
                .map(|h| (h.doc_id, h.score.to_bits()))
                .collect();
            if timed != want || cut {
                failures.push(format!(
                    "deadline [{q}]\n  rust {timed:x?}\n  java {want:x?}"
                ));
            }
            let past = std::time::Instant::now();
            let (none, cut) =
                search_boolean_query_multi_segment_with_deadline(&segments, &bq, &norms, top, past)
                    .unwrap();
            assert!(none.is_empty() && cut, "an expired deadline scores nothing");
        }
    }
    // `BayesianScoreEstimator.estimate` over `body`.
    let est = if estimator {
        std::fs::read_to_string(dir.join("estimator.tsv")).unwrap()
    } else {
        String::new()
    };
    let mut estimates = 0;
    for line in est.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        let (n, tokens, seed) = (
            f[0].parse().unwrap(),
            f[1].parse().unwrap(),
            f[2].parse().unwrap(),
        );
        let p =
            lucene_search::bayesian_estimator::estimate(&segments, &norms, "body", n, tokens, seed)
                .unwrap();
        let bits = |s: &str| u32::from_str_radix(s, 16).unwrap();
        let got = (p.alpha.to_bits(), p.beta.to_bits(), p.base_rate.to_bits());
        let want = (bits(f[3]), bits(f[4]), bits(f[5]));
        if got != want {
            failures.push(format!(
                "estimate {line}\n  rust {got:x?}\n  java {want:x?}"
            ));
        }
        estimates += 1;
    }
    assert_eq!(estimates, if estimator { 3 } else { 0 });
    eprintln!("{cases} searches compared, {skipped} skipped");
    assert_eq!(skipped, 0, "every recorded search runs");
    assert!(
        failures.is_empty(),
        "{} searches differ:\n{}",
        failures.len(),
        failures.join("\n")
    );
    cases
}

/// The range, point and doc-values-rewrite queries at their edges, each
/// against the query it must equal over `m7_queries_index` (Lucene's own
/// answers for the ordinary shapes are the searches above): a range over
/// every value is `FieldExistsQuery` (`rewrite`), an inverted or absent one
/// nothing; `IndexSortSortedNumericDocValuesRangeQuery` on an index not
/// sorted by its field is its fallback; a two-dimensional `PointInSetQuery`
/// (`SinglePointVisitor`, a walk per point) is the union of its points'
/// boxes; a point query of the wrong shape is refused with Java's message;
/// and `DOC_VALUES_REWRITE` matches what `CONSTANT_SCORE_REWRITE` does for
/// sources the recorded searches do not use (automata, term sets, open and
/// exclusive bounds, the empty prefix).
#[test]
fn range_and_point_queries_equal_their_rewrites_at_the_edges() {
    use lucene_search::extended_query::{
        AutomatonQuery, IndexSortSortedNumericDocValuesRangeQuery, MultiTermQuery, MultiTermSource,
        NumericDocValuesRangeQuery, PointInSetQuery, PointRangeQuery, RewriteMethod,
        TermRangeQuery,
    };
    use lucene_search::index_searcher::IndexSearcher;
    use lucene_search::query::{BooleanQuery, FieldExistsQuery, PrefixQuery};
    use std::sync::Arc;

    let reader = DirectoryReader::open(&FsDirectory::open(data("m7_queries_index"))).unwrap();
    let mut opened = reader.open_segments().unwrap();
    opened.open_points().unwrap();
    let segments = opened.as_open_segments();
    let norms: Vec<Option<&HashMap<String, FieldNorms<'_>>>> = vec![None; segments.len()];
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();
    let search = |c: Clause| {
        let mut q = BooleanQuery::new();
        q.filter.push(c);
        searcher.search(&q, 1 << 20).map(|t| {
            let mut d: Vec<i32> = t.score_docs.iter().map(|h| h.doc).collect();
            d.sort_unstable();
            d
        })
    };
    let docs = |c: Clause| search(c).unwrap();
    let exists = |f: &str| docs(Clause::Exists(FieldExistsQuery::new(f)));
    let num = exists("num");
    assert!(!num.is_empty());

    // Numeric doc-values ranges.
    let range = NumericDocValuesRangeQuery::new;
    assert_eq!(docs(range("num", i64::MIN, i64::MAX).into()), num);
    assert!(docs(range("num", 5, 4).into()).is_empty());
    assert!(docs(range("nosuch", 0, 10).into()).is_empty());

    // The index-sort range on an index sorted by no field: its fallback,
    // and the same rewrites first.
    let sorted = |lo: i64, hi: i64| -> Clause {
        IndexSortSortedNumericDocValuesRangeQuery::new("num", lo, hi, range("num", lo, hi)).into()
    };
    let some = docs(range("num", 10, 60).into());
    assert!(!some.is_empty() && some.len() < num.len());
    assert_eq!(docs(sorted(10, 60)), some);
    assert_eq!(docs(sorted(i64::MIN, i64::MAX)), num);
    assert!(docs(sorted(60, 10)).is_empty());

    // A two-dimensional point set: a walk per point, the union of their
    // single-point boxes.
    let corners: Vec<[i32; 2]> = (-50..=50).flat_map(|x| [[x, 0], [x, 7], [x, -3]]).collect();
    let packed = |p: &[i32; 2]| -> Vec<u8> {
        p.iter()
            .flat_map(|&v| ((v as u32) ^ 0x8000_0000).to_be_bytes())
            .collect()
    };
    let set = PointInSetQuery::new("p2", 2, 4, corners.iter().map(packed)).unwrap();
    let mut union = BooleanQuery::new();
    for p in &corners {
        union
            .should
            .push(PointRangeQuery::int_range("p2", p, p).unwrap().into());
    }
    let want = docs(Clause::Boolean(Box::new(union)));
    assert!(!want.is_empty());
    assert_eq!(docs(set.into()), want);
    // A field indexed with another shape.
    let e = search(PointInSetQuery::int_set("p2", &[1]).unwrap().into()).unwrap_err();
    assert!(
        e.to_string()
            .contains("field=\"p2\" was indexed with numDims=2 bytesPerDim=4 but this query has numDims=1 bytesPerDim=4"),
        "{e}"
    );

    // `DOC_VALUES_REWRITE` against `CONSTANT_SCORE_REWRITE` on `tag`
    // (indexed and in `SORTED_SET` doc values, several a document).
    let both = |source: MultiTermSource| {
        let dv = docs(MultiTermQuery::new(source.clone(), RewriteMethod::DocValues).into());
        let cs = docs(MultiTermQuery::new(source, RewriteMethod::ConstantScore).into());
        assert_eq!(dv, cs);
        dv
    };
    let range = |lo: Option<&str>, hi: Option<&str>, inc: bool| {
        MultiTermSource::TermRange(TermRangeQuery::new(
            "tag",
            lo.map(|s| s.as_bytes().to_vec()),
            hi.map(|s| s.as_bytes().to_vec()),
            inc,
            inc,
        ))
    };
    let tagged = exists("tag");
    assert_eq!(both(range(None, None, true)), tagged);
    assert!(!both(range(Some("k03"), Some("k07"), false)).is_empty());
    assert!(!both(range(None, Some("k05"), true)).is_empty());
    assert!(
        both(range(Some("k99"), None, true)).is_empty(),
        "past the last term"
    );
    assert_eq!(
        both(MultiTermSource::Prefix(PrefixQuery::new(
            "tag",
            b"".to_vec()
        ))),
        tagged
    );
    use lucene_util::automaton::{automata, operations, RegExp, DEFAULT_DETERMINIZE_WORK_LIMIT};
    let automaton = |re: &str| {
        let a = RegExp::new(re).unwrap().to_automaton().unwrap();
        let a = operations::determinize(&a, DEFAULT_DETERMINIZE_WORK_LIMIT).unwrap();
        MultiTermSource::Automaton(AutomatonQuery::new("tag", a, false))
    };
    assert!(!both(automaton("k0[2-5]")).is_empty());
    assert_eq!(both(automaton(".*")), tagged);
    assert!(!both(MultiTermSource::Automaton(AutomatonQuery::new(
        "tag",
        automata::make_string("k04"),
        false
    )))
    .is_empty());
    assert!(both(MultiTermSource::Automaton(AutomatonQuery::new(
        "tag",
        automata::make_empty(),
        false
    )))
    .is_empty());
    let terms: Vec<Vec<u8>> = ["k01", "k08", "k13", "kzz"]
        .iter()
        .map(|t| t.as_bytes().to_vec())
        .collect();
    let set = lucene_search::join::TermSetSource {
        field: "tag".to_string(),
        terms: terms.into(),
        from_field: "from".to_string(),
        from_query: Arc::new(Clause::Exists(FieldExistsQuery::new("tag"))),
    };
    assert!(!both(MultiTermSource::TermSet(set)).is_empty());
}

/// `IndexSortSortedNumericDocValuesRangeQuery`'s binary search on the sort
/// shapes the recorded fixture does not have (it is sorted ascending by a
/// `LONG` key without a missing value): an index this port writes sorted by
/// `LONG` and `INT` keys, ascending and descending, with and without a
/// missing value -- and by a `FLOAT` key, which the query does not search
/// (its fallback). Every range, inside and around the values, matches what
/// its fallback, a plain doc-values range, does.
#[test]
fn the_index_sort_range_search_equals_its_fallback_on_every_sort_shape() {
    use lucene_index::document::{Document, NumericDocValuesField};
    use lucene_index::index_writer::IndexWriter;
    use lucene_index::segment_info::{
        IndexSortField, IndexSortKind, LuceneVersion, NumericSortKey,
    };
    use lucene_search::extended_query::{
        IndexSortSortedNumericDocValuesRangeQuery, NumericDocValuesRangeQuery,
    };
    use lucene_search::index_searcher::IndexSearcher;
    use lucene_search::query::BooleanQuery;
    use lucene_util::test_support::TempDir;

    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let shapes = [
        ("long", false, NumericSortKey::Long(None)),
        ("long-desc-missing", true, NumericSortKey::Long(Some(3))),
        ("int-missing", false, NumericSortKey::Int(Some(-4))),
        ("int-desc", true, NumericSortKey::Int(None)),
        ("float", false, NumericSortKey::Float(None)),
    ];
    for (name, reverse, key) in shapes {
        let tmp = TempDir::new(&format!("index-sort-range-{name}"));
        let dir = FsDirectory::open(tmp.path());
        {
            let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", version).unwrap();
            w.enable_explicit_documents().unwrap();
            w.set_index_sort(Some(&[IndexSortField {
                field: "v".to_string(),
                reverse,
                kind: IndexSortKind::Numeric(key),
            }]))
            .unwrap();
            w.set_max_buffered_docs(97).unwrap();
            for i in 0..300i64 {
                let mut d = Document::new();
                // Every seventh document has no value.
                if i % 7 != 0 {
                    d.add(NumericDocValuesField::new("v", (i * 37) % 41 - 20));
                }
                w.add_fields_document(&d).unwrap();
            }
            w.commit().unwrap();
        }
        let reader = DirectoryReader::open(&dir).unwrap();
        assert!(reader.segment_readers().len() > 1, "{name}");
        let opened = reader.open_segments().unwrap();
        let segments = opened.as_open_segments();
        let norms: Vec<Option<&HashMap<String, FieldNorms<'_>>>> = vec![None; segments.len()];
        let searcher = IndexSearcher::new(&segments, &norms).unwrap();
        let docs = |c: Clause| {
            let mut q = BooleanQuery::new();
            q.filter.push(c);
            let mut d: Vec<i32> = searcher
                .search(&q, 1000)
                .unwrap()
                .score_docs
                .iter()
                .map(|h| h.doc)
                .collect();
            d.sort_unstable();
            d
        };
        for (lo, hi) in [
            (-20, 20),
            (-5, 5),
            (3, 3),
            (-4, -4),
            (0, 0),
            (-30, -21),
            (21, 30),
            (-25, 0),
        ] {
            let fallback = NumericDocValuesRangeQuery::new("v", lo, hi);
            let want = docs(fallback.clone().into());
            let got =
                docs(IndexSortSortedNumericDocValuesRangeQuery::new("v", lo, hi, fallback).into());
            assert_eq!(got, want, "{name} [{lo}, {hi}]");
        }
    }
}

/// Doc-values rewrites and point queries over an index this port writes with
/// the shapes `m7_queries_index` lacks: a `SORTED` and a single-valued
/// `SORTED_SET` keyword (`DOC_VALUES_REWRITE` against
/// `CONSTANT_SCORE_REWRITE`), a keyword without doc values and an absent one
/// (nothing, as `DocValuesRewriteMethod` finds no values), a long point on
/// every document (`IndexOrDocValuesQuery` costing a range that holds every
/// document at `maxDoc`), point queries on an absent field, and a point query
/// over segments whose points were not opened (refused).
#[test]
fn doc_values_rewrites_and_points_on_shapes_the_fixture_lacks() {
    use lucene_index::document::{
        Document, LongPoint, NumericDocValuesField, SortedDocValuesField, SortedSetDocValuesField,
        Store, StringField,
    };
    use lucene_index::index_writer::IndexWriter;
    use lucene_index::segment_info::LuceneVersion;
    use lucene_search::extended_query::{
        IndexOrDocValuesQuery, MultiTermQuery, MultiTermSource, NumericDocValuesRangeQuery,
        PointInSetQuery, PointRangeQuery, RewriteMethod, TermRangeQuery,
    };
    use lucene_search::index_searcher::IndexSearcher;
    use lucene_search::query::BooleanQuery;
    use lucene_util::test_support::TempDir;

    let tmp = TempDir::new("dv-rewrite-points");
    let dir = FsDirectory::open(tmp.path());
    {
        let version = LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        };
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", version).unwrap();
        w.set_max_buffered_docs(70).unwrap();
        for i in 0..200i64 {
            let mut d = Document::new();
            let k = format!("k{:02}", (i * 7) % 23);
            d.add(StringField::new("s", k.clone(), Store::No));
            d.add(SortedDocValuesField::new("s", k.clone().into_bytes()));
            if i % 5 != 0 {
                d.add(StringField::new("ss", k.clone(), Store::No));
                d.add(SortedSetDocValuesField::new("ss", k.clone().into_bytes()));
            }
            d.add(StringField::new("plain", k, Store::No));
            d.add(LongPoint::new("p", &[i - 100]).unwrap());
            d.add(NumericDocValuesField::new("p", i - 100));
            w.add_fields_document(&d).unwrap();
        }
        w.commit().unwrap();
    }
    let reader = DirectoryReader::open(&dir).unwrap();
    assert!(reader.segment_readers().len() > 1);
    let mut opened = reader.open_segments().unwrap();
    // Before the points are opened: a point query is refused.
    {
        let segments = opened.as_open_segments();
        let norms: Vec<Option<&HashMap<String, FieldNorms<'_>>>> = vec![None; segments.len()];
        let searcher = IndexSearcher::new(&segments, &norms).unwrap();
        let mut q = BooleanQuery::new();
        q.filter
            .push(PointRangeQuery::long_range("p", &[0], &[5]).unwrap().into());
        assert!(searcher.search(&q, 10).is_err());
    }
    opened.open_points().unwrap();
    let segments = opened.as_open_segments();
    let norms: Vec<Option<&HashMap<String, FieldNorms<'_>>>> = vec![None; segments.len()];
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();
    let docs = |c: Clause| {
        let mut q = BooleanQuery::new();
        q.filter.push(c);
        let mut d: Vec<i32> = searcher
            .search(&q, 1000)
            .unwrap()
            .score_docs
            .iter()
            .map(|h| h.doc)
            .collect();
        d.sort_unstable();
        d
    };
    let range = |field: &str, lo: &str, hi: &str| {
        MultiTermSource::TermRange(TermRangeQuery::new(
            field,
            Some(lo.as_bytes().to_vec()),
            Some(hi.as_bytes().to_vec()),
            true,
            false,
        ))
    };
    let rewritten = |source: MultiTermSource, method: RewriteMethod| {
        docs(MultiTermQuery::new(source, method).into())
    };
    for field in ["s", "ss"] {
        let source = range(field, "k03", "k11");
        let want = rewritten(source.clone(), RewriteMethod::ConstantScore);
        assert!(!want.is_empty(), "{field}");
        assert_eq!(rewritten(source, RewriteMethod::DocValues), want, "{field}");
        // Terms the dictionary does not have.
        assert!(rewritten(range(field, "x", "z"), RewriteMethod::DocValues).is_empty());
    }
    for field in ["plain", "nosuch"] {
        assert!(rewritten(range(field, "k00", "k99"), RewriteMethod::DocValues).is_empty());
    }

    // Points: a range over every document, by points or by doc values.
    let all: Vec<i32> = (0..200).collect();
    let every = IndexOrDocValuesQuery::new(
        PointRangeQuery::long_range("p", &[-100], &[99]).unwrap(),
        NumericDocValuesRangeQuery::new("p", -100, 99),
    );
    assert_eq!(docs(every.into()), all);
    let some = IndexOrDocValuesQuery::new(
        PointRangeQuery::long_range("p", &[-10], &[10]).unwrap(),
        NumericDocValuesRangeQuery::new("p", -10, 10),
    );
    assert_eq!(docs(some.into()), (90..=110).collect::<Vec<i32>>());
    let absent = IndexOrDocValuesQuery::new(
        PointRangeQuery::long_range("nosuch", &[0], &[1]).unwrap(),
        NumericDocValuesRangeQuery::new("nosuch", 0, 1),
    );
    assert!(docs(absent.into()).is_empty());
    assert!(docs(
        PointRangeQuery::long_range("nosuch", &[0], &[1])
            .unwrap()
            .into()
    )
    .is_empty());
    assert!(docs(PointInSetQuery::long_set("nosuch", &[0, 1]).unwrap().into()).is_empty());
}

/// `BayesianScoreEstimator.estimate` beyond the recorded runs: a field no
/// segment indexes has no vocabulary to sample and estimates Java's
/// fallback `(1, 0, 0.01)`; one single-term query over a whole vocabulary
/// keeps a reservoir of one, replaced by draws past it, and still estimates
/// finite parameters inside the base rate's clamp.
#[test]
fn the_estimator_falls_back_without_a_vocabulary_and_samples_past_its_reservoir() {
    use lucene_search::bayesian_estimator::estimate;
    let reader = DirectoryReader::open(&FsDirectory::open(data("m7_queries_index"))).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let owned = reader.field_norms_by_field(&["body".to_string(), "title".to_string()]);
    let norms: Vec<Option<&HashMap<String, FieldNorms<'_>>>> = owned.iter().map(Some).collect();
    let p = estimate(&segments, &norms, "nosuch", 5, 3, 1).unwrap();
    assert_eq!((p.alpha, p.beta, p.base_rate), (1.0, 0.0, 0.01));
    for seed in [1, 7, 42] {
        let p = estimate(&segments, &norms, "title", 1, 1, seed).unwrap();
        assert!(p.alpha.is_finite() && p.alpha > 0.0, "{seed}: {}", p.alpha);
        assert!(p.beta > 0.0, "{seed}: {}", p.beta);
        assert!(
            (1e-6..=0.5).contains(&p.base_rate),
            "{seed}: {}",
            p.base_rate
        );
    }
}
