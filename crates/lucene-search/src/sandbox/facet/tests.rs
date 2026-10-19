//! Unit tests of the facet package beside the Java fixtures
//! (`tests/sandbox_facet_fixtures.rs`): the boundaries the fixture index
//! does not reach -- interval building at `Long.MIN_VALUE`/`MAX_VALUE`,
//! Java's `Math.max` for floats, the histogram's skip-index and bulk paths
//! over an index written here, and the errors of reading too early.

use std::collections::BTreeMap;
use std::hash::Hasher;
use std::sync::Arc;

use lucene_index::document::{self as d, Document, IndexableField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_store::FsDirectory;
use lucene_util::numeric_utils::int_to_sortable_bytes;
use lucene_util::test_support::TempDir;

use super::builders::{FacetBuilder, FacetOrchestrator, NoCollector, NoMainCollector};
use super::comparable::{self, ByOrdinal};
use super::cutters::LongValueFacetCutter;
use super::histogram::{
    bucket_end, bytes_to_long, check_max_buckets, histogram_search, HistogramCollectorManager,
};
use super::iterators::CandidateSetOrdinalIterator;
use super::labels::{LabelToOrd, OrdToLabel, RangeOrdToLabel};
use super::post_collection::PostCollectionFaceting;
use super::ranges::{
    disjoint_intervals, overlapping_intervals, process_value, IntervalTracker, LongRangeFacetCutter,
};
use super::recorders::{CountFacetRecorder, LongAggregationsFacetRecorder, Reducer};
use super::*;
use crate::directory_reader::DirectoryReader;
use crate::extended_query::{IndexOrDocValuesQuery, PointRangeQuery};
use crate::facets::multi_long_values_source;
use crate::facets::range::LongRange;
use crate::facets::taxonomy::facet_label::FacetLabel;
use crate::facets::{FacetsCollector, FacetsCollectorManager};
use crate::function::FunctionScoreQuery;
use crate::grouping::{search_manager, GroupingCollectorManager};
use crate::index_searcher::{IndexSearcher, SegmentNorms};
use crate::leaf_collector::SegmentCollector;
use crate::query::{BooleanQuery, BoostQuery, Clause, ConstantScoreQuery, MatchAllDocsQuery};
use crate::values_source::{constant, long_from_long_field};
use crate::Error;

const VERSION: LuceneVersion = LuceneVersion {
    major: 10,
    minor: 5,
    bugfix: 0,
};

/// The first segment's document count.
const DOCS: i32 = 5000;

fn b(f: impl IndexableField + 'static) -> Box<dyn IndexableField> {
    Box::new(f)
}

/// Two segments: `DOCS` documents with an int point and value `p = i`, a
/// skip-indexed `n = i / 10`, a skip-indexed `sp` from document 4600 on,
/// two points and values `m`, and `rv = i` on even documents; then ten
/// documents with `p` only.
fn index(tmp: &TempDir) -> DirectoryReader {
    let dir = FsDirectory::open(tmp.path());
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
    for i in 0..DOCS {
        let v = i64::from(i);
        let mut fields = vec![
            b(d::IntPoint::new("p", &[i]).unwrap()),
            b(d::NumericDocValuesField::new("p", v)),
            b(d::NumericDocValuesField::indexed_field("n", v / 10)),
            b(d::IntPoint::new("m", &[i]).unwrap()),
            b(d::IntPoint::new("m", &[i + 1]).unwrap()),
            b(d::SortedNumericDocValuesField::new("m", v)),
            b(d::SortedNumericDocValuesField::new("m", v + 1)),
        ];
        if i >= 4600 {
            fields.push(b(d::NumericDocValuesField::indexed_field("sp", v)));
        }
        if i % 2 == 0 {
            fields.push(b(d::NumericDocValuesField::new("rv", v)));
        }
        let mut doc = Document::new();
        for f in fields {
            doc.add_boxed(f);
        }
        w.add_fields_document(&doc).unwrap();
    }
    w.commit().unwrap();
    for i in 0..10 {
        let mut doc = Document::new();
        doc.add_boxed(b(d::IntPoint::new("p", &[i]).unwrap()));
        doc.add_boxed(b(d::NumericDocValuesField::new("p", i64::from(i))));
        w.add_fields_document(&doc).unwrap();
    }
    w.commit().unwrap();
    drop(w);
    DirectoryReader::open(&dir).unwrap()
}

fn match_all() -> BooleanQuery {
    BooleanQuery::new().with_must([Clause::MatchAllDocs(MatchAllDocsQuery::new(0))])
}

/// `IntPoint.newRangeQuery(field, lo, hi)`.
fn int_range(field: &str, lo: i32, hi: i32) -> Clause {
    let enc = |v: i32| {
        let mut out = vec![0u8; 4];
        int_to_sortable_bytes(v, &mut out, 0);
        out
    };
    PointRangeQuery::new(field, 1, enc(lo), enc(hi))
        .unwrap()
        .into()
}

/// The naive histogram of `field`'s values over the documents `keep`s.
fn naive(values: impl Iterator<Item = i64>, width: i64) -> BTreeMap<i64, i32> {
    let mut out = BTreeMap::new();
    for v in values {
        *out.entry(v.div_euclid(width)).or_insert(0) += 1;
    }
    out
}

fn long_range(label: &str, min: i64, max: i64) -> LongRange {
    LongRange::new(label, min, true, max, true).unwrap()
}

#[test]
fn reducers_follow_javas_math() {
    assert_eq!(Reducer::Max.reduce_i32(3, -4).unwrap(), 3);
    assert_eq!(Reducer::Sum.reduce_i32(3, -4).unwrap(), -1);
    let e = Reducer::Sum.reduce_i32(i32::MAX, 1).unwrap_err();
    assert!(matches!(e, Error::IllegalArgument(m) if m == "integer overflow"));
    assert_eq!(Reducer::Max.reduce_i64(-3, -4).unwrap(), -3);
    let e = Reducer::Sum.reduce_i64(i64::MIN, -1).unwrap_err();
    assert!(matches!(e, Error::IllegalArgument(m) if m == "long overflow"));

    // `Math.max`: NaN wins, 0.0 beats -0.0 whichever side it is on.
    assert!(Reducer::Max.reduce_f32(f32::NAN, 1.0).is_nan());
    assert!(Reducer::Max.reduce_f32(1.0, f32::NAN).is_nan());
    assert!(Reducer::Max.reduce_f32(-0.0, 0.0).is_sign_positive());
    assert!(Reducer::Max.reduce_f32(0.0, -0.0).is_sign_positive());
    assert_eq!(Reducer::Max.reduce_f32(2.0, 1.0), 2.0);
    assert_eq!(Reducer::Max.reduce_f32(1.0, 2.0), 2.0);
    assert_eq!(Reducer::Sum.reduce_f32(1.5, 2.0), 3.5);
    assert!(Reducer::Max.reduce_f64(f64::NAN, 1.0).is_nan());
    assert!(Reducer::Max.reduce_f64(-0.0, 0.0).is_sign_positive());
    assert!(Reducer::Max.reduce_f64(0.0, -0.0).is_sign_positive());
    assert_eq!(Reducer::Max.reduce_f64(2.0, 1.0), 2.0);
    assert_eq!(Reducer::Max.reduce_f64(1.0, 2.0), 2.0);
    assert_eq!(Reducer::Sum.reduce_f64(1.5, 2.0), 3.5);
}

/// A cutter that asks for ordinal 0 to be rolled up but has no hierarchy.
struct RollupWithoutChildren;

impl FacetCutter for RollupWithoutChildren {
    fn create_leaf_cutter<'a>(
        &self,
        _reader: &'a crate::directory_reader::SegmentReader,
    ) -> Result<Box<dyn LeafFacetCutter + 'a>> {
        Err(Error::Unsupported("test cutter".into()))
    }

    fn ordinals_to_rollup(&self) -> Result<Option<BoxOrdinalIterator<'_>>> {
        Ok(Some(Box::new(from_array(vec![0]))))
    }
}

#[test]
fn recorders_refuse_reads_before_reduce_and_rollups_without_children() {
    let count = CountFacetRecorder::new();
    assert!(matches!(count.get_count(0), Err(Error::IllegalState(_))));
    assert!(matches!(
        count.reduce(&RollupWithoutChildren),
        Err(Error::IllegalState(_))
    ));
    assert!(RollupWithoutChildren.children_ords(0).unwrap().is_none());

    let e = LongAggregationsFacetRecorder::new(Vec::new(), vec![Reducer::Max])
        .err()
        .unwrap();
    assert!(matches!(e, Error::IllegalArgument(m) if m == "0 values sources but 1 reducers"));
    let agg =
        LongAggregationsFacetRecorder::new(vec![long_from_long_field("p")], vec![Reducer::Sum])
            .unwrap();
    assert!(agg.is_empty().is_err());
    assert!(agg.recorded_ords().is_err());
    assert!(agg.contains(0).is_err());
    agg.reduce(&LongValueFacetCutter::new("p")).unwrap();
    assert!(agg.is_empty().unwrap());
    assert_eq!(
        agg.recorded_ords().unwrap().next_ord().unwrap(),
        NO_MORE_ORDS
    );
    assert!(!agg.contains(0).unwrap());
}

#[test]
fn small_helpers() {
    assert_eq!(EmptyOrdinalIterator.next_ord().unwrap(), NO_MORE_ORDS);
    let mut h = MixHasher::default();
    h.write(b"ab");
    let first = h.finish();
    h.write(b"c");
    assert_ne!(first, h.finish());

    let mut ords = vec![3, 1, 2];
    comparable::sort(&mut ords, &&ByOrdinal).unwrap();
    assert_eq!(ords, [1, 2, 3]);

    assert!(check_max_buckets(4, 4).is_ok());
    let e = check_max_buckets(5, 4).unwrap_err();
    assert!(matches!(e, Error::IllegalState(m)
        if m == "Collected 5 buckets, which is more than the configured max number of buckets: 4"));
    assert_eq!(bucket_end(-5, 3, 100), -3);
    assert_eq!(bucket_end(7, 3, 8), 8);
    let mut four = [0u8; 4];
    int_to_sortable_bytes(-7, &mut four, 0);
    assert_eq!(bytes_to_long(4)(&four), -7);
    let mut eight = [0u8; 8];
    lucene_util::numeric_utils::long_to_sortable_bytes(-7, &mut eight, 0);
    assert_eq!(bytes_to_long(8)(&eight), -7);
    assert!(HistogramCollectorManager::new("p", 0).is_err());
}

fn intervals(v: Vec<ranges::InclusiveRange>) -> Vec<(i64, i64)> {
    v.into_iter().map(|r| (r.start, r.end)).collect()
}

#[test]
fn elementary_intervals_at_the_ends_of_long() {
    let none: Vec<(&LongRange, i32)> = Vec::new();
    assert_eq!(intervals(disjoint_intervals(&none)), [(i64::MIN, i64::MAX)]);
    let tail = long_range("t", 10, i64::MAX);
    assert_eq!(
        intervals(disjoint_intervals(&[(&tail, 0)])),
        [(i64::MIN, 9), (10, i64::MAX)]
    );
    let head = long_range("h", i64::MIN, 0);
    assert_eq!(
        intervals(disjoint_intervals(&[(&head, 0)])),
        [(i64::MIN, 0), (1, i64::MAX)]
    );

    // A one-value range at `Long.MIN_VALUE` is its own first interval, as
    // is a one-value range in the middle.
    let min = long_range("min", i64::MIN, i64::MIN);
    let five = long_range("five", 5, 5);
    let wide = long_range("wide", 3, 10);
    let up = long_range("up", 8, i64::MAX);
    let sorted = [(&min, 0), (&wide, 1), (&five, 2), (&up, 3)];
    assert_eq!(
        intervals(overlapping_intervals(&sorted)),
        [
            (i64::MIN, i64::MIN),
            (i64::MIN + 1, 2),
            (3, 4),
            (5, 5),
            (6, 7),
            (8, 10),
            (11, i64::MAX)
        ]
    );
}

#[test]
fn interval_lookup_and_tracking() {
    let boundaries = [10, 20, 30];
    assert_eq!(process_value(&boundaries, 5, None), 0);
    assert_eq!(process_value(&boundaries, 25, None), 2);
    assert_eq!(process_value(&boundaries, 15, Some(1)), 1);
    assert_eq!(process_value(&boundaries, 25, Some(1)), 2);
    assert_eq!(process_value(&boundaries, 100, Some(2)), 2);

    let mut t = IntervalTracker::new(5);
    t.set(-1);
    t.set(9);
    t.set(3);
    t.set(1);
    t.freeze();
    assert!(t.any_in(1, 2) && t.any_in(3, 4) && !t.any_in(2, 3) && !t.any_in(9, 10));
    assert!(t.any_in(0, 200) && !t.any_in(4, 4) && !t.any_in(5, 3));
    let mut wide = IntervalTracker::new(200);
    wide.set(130);
    assert!(wide.any_in(5, 131) && wide.any_in(130, 400) && !wide.any_in(0, 130));
    assert!(!wide.any_in(131, 200) && !wide.any_in(64, 128));
    assert_eq!(
        [t.next_ord(), t.next_ord(), t.next_ord()],
        [1, 3, NO_MORE_ORDS]
    );
    t.clear();
    t.freeze();
    assert_eq!(t.next_ord(), NO_MORE_ORDS);
}

/// A label's only component, parsed, is its ordinal.
struct ParsedLabels;

impl LabelToOrd for ParsedLabels {
    fn get_ord(&self, label: &FacetLabel) -> Result<i32> {
        Ok(label.components()[0].parse().unwrap())
    }
}

#[test]
fn searches_over_a_written_index() {
    let tmp = TempDir::new("sandbox-facet-unit");
    let reader = index(&tmp);
    let mut opened = reader.open_segments().unwrap();
    opened.open_points().unwrap();
    let segments = opened.as_open_segments();
    let norms: Vec<SegmentNorms<'_, '_>> = vec![None; segments.len()];
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();
    let all = match_all();

    // The value cutter over a field half the documents lack.
    let rv = Arc::new(LongValueFacetCutter::new("rv"));
    assert_eq!(rv.field(), "rv");
    let counts = Arc::new(CountFacetRecorder::new());
    let manager = FacetFieldCollectorManager::new(rv.clone(), counts.clone());
    assert!(Arc::ptr_eq(manager.recorder(), &counts));
    let _ = manager.cutter();
    search_manager(&searcher, &all, &manager).unwrap();
    let mut recorded = counts.recorded_ords().unwrap();
    let mut n = 0;
    while recorded.next_ord().unwrap() != NO_MORE_ORDS {
        n += 1;
    }
    assert_eq!(n, DOCS / 2);
    assert!(rv.get_label(-5).unwrap().is_none());

    // Candidates the recorder lacks are skipped.
    let labels = [
        FacetLabel::new(["0"]).unwrap(),
        FacetLabel::new(["999999"]).unwrap(),
    ];
    let mut candidates =
        CandidateSetOrdinalIterator::new(counts.as_ref(), &labels, &ParsedLabels).unwrap();
    let first = candidates.next_ord().unwrap();
    assert!(counts.contains(first).unwrap());
    assert_eq!(candidates.next_ord().unwrap(), NO_MORE_ORDS);

    // A single-valued range cutter over the same field, through the
    // orchestrator without a main collector.
    let ranges = [long_range("low", 0, 99), long_range("high", 100, 199)];
    let source = multi_long_values_source::from_single_valued(long_from_long_field("rv"));
    let cutter: Arc<dyn FacetCutter> = Arc::new(LongRangeFacetCutter::create(source, &ranges));
    let builder = FacetBuilder::common(
        "r",
        cutter,
        Arc::new(RangeOrdToLabel::for_long_ranges(&ranges)),
    );
    FacetOrchestrator::new()
        .add_builder(&builder)
        .collect(&all, &searcher)
        .unwrap();
    let result = builder.get_result().unwrap();
    let got: Vec<(String, i32)> = result
        .label_values
        .iter()
        .map(|lv| (lv.label.clone(), lv.count))
        .collect();
    assert_eq!(got, [("low".to_string(), 50), ("high".to_string(), 50)]);

    // The main collector's score mode, with and without facets.
    let scoring = FacetsCollectorManager::new(true);
    let plain = FacetsCollectorManager::new(false);
    let alone = FacetOrchestrator::new()
        .collect_with(&all, &searcher, Some(&scoring))
        .unwrap()
        .unwrap();
    assert!(alone.matching_docs()[0].scores.is_some());
    let values = FacetBuilder::long_value("rv");
    assert!(matches!(values.get_result(), Err(Error::IllegalState(_))));
    for (main, keeps) in [(&scoring, true), (&plain, false)] {
        let fc = FacetOrchestrator::new()
            .add_builder(&values)
            .collect_with(&all, &searcher, Some(main))
            .unwrap()
            .unwrap();
        assert_eq!(fc.matching_docs()[0].scores.is_some(), keeps);
    }
    assert_eq!(values.get_result().unwrap().child_count, DOCS / 2);

    // The never-used main collector.
    let none = NoMainCollector;
    let mut c: NoCollector = GroupingCollectorManager::new_collector(&none).unwrap();
    assert_eq!(
        c.score_mode(),
        crate::collector::ScoreMode::CompleteNoScores
    );
    c.set_next_reader(0, &segments[0]).unwrap();
    c.collect(0, 1.0).unwrap();
    GroupingCollectorManager::reduce(&none, vec![c]).unwrap();
}

#[test]
fn post_collection_scores_and_foreign_leaves() {
    let tmp = TempDir::new("sandbox-facet-post");
    let reader = index(&tmp);
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let norms: Vec<SegmentNorms<'_, '_>> = vec![None; segments.len()];
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();
    let all = match_all();
    let none: &[(&str, &FacetsCollectorManager)] = &[];
    let no_docs: &[(&str, &FacetsCollector<'_>)] = &[];

    let scoring = FacetsCollectorManager::new(true);
    let plain = FacetsCollectorManager::new(false);
    let scored = search_manager(&searcher, &all, &scoring).unwrap();
    let unscored = search_manager(&searcher, &all, &plain).unwrap();

    // Scores are carried over from the collected documents.
    let out = PostCollectionFaceting::new(&searcher, Some(&scoring), none, &scored, no_docs)
        .collect()
        .unwrap();
    let again = out.drill_down_result.unwrap();
    assert_eq!(
        again.matching_docs()[0].scores,
        scored.matching_docs()[0].scores
    );
    assert!(out.drill_sideways_results.is_empty());

    // A collector that needs scores the documents do not have.
    let e = PostCollectionFaceting::new(&searcher, Some(&scoring), none, &unscored, no_docs)
        .collect()
        .unwrap_err();
    assert!(matches!(e, Error::IllegalState(m)
        if m == "Collector requires scores, but FacetCollector doesn't have them."));

    // Documents of a leaf the searcher does not have.
    let first = IndexSearcher::new(&segments[..1], &norms[..1]).unwrap();
    let e = PostCollectionFaceting::new(&first, Some(&plain), none, &unscored, no_docs)
        .collect()
        .unwrap_err();
    assert!(matches!(e, Error::IllegalArgument(_)));
}

#[test]
fn histograms_over_points_and_skip_indexes() {
    let tmp = TempDir::new("sandbox-facet-histogram");
    let reader = index(&tmp);
    let mut opened = reader.open_segments().unwrap();
    opened.open_points().unwrap();
    let segments = opened.as_open_segments();
    let norms: Vec<SegmentNorms<'_, '_>> = vec![None; segments.len()];
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();
    let all = match_all();
    let ps = || (0..i64::from(DOCS)).chain(0..10);
    let hist = |q: &BooleanQuery, field: &str, width: i64| {
        histogram_search(
            &searcher,
            q,
            &HistogramCollectorManager::new(field, width).unwrap(),
        )
    };

    // Match-all over the points tree (the first segment) and the values
    // (the second, too small for the tree).
    assert_eq!(hist(&all, "p", 1000).unwrap(), naive(ps(), 1000));
    // Too many buckets, found while walking the tree.
    let few = HistogramCollectorManager::with_max_buckets("p", 1000, 2).unwrap();
    assert!(matches!(
        histogram_search(&searcher, &all, &few),
        Err(Error::IllegalState(_))
    ));

    // The range, under every wrapper Java unwraps; and not unwrapped when
    // it is not the query's only clause.
    let in_range = naive(ps().filter(|v| (100..=2999).contains(v)), 1000);
    let wrapped = [
        int_range("p", 100, 2999),
        BoostQuery::new(ConstantScoreQuery::new(int_range("p", 100, 2999), 1.0), 2.0).into(),
        FunctionScoreQuery::new(int_range("p", 100, 2999), constant(1.0)).into(),
        IndexOrDocValuesQuery::new(int_range("p", 100, 2999), int_range("p", 100, 2999)).into(),
        BooleanQuery::new()
            .with_filter([int_range("p", 100, 2999)])
            .into(),
    ];
    for clause in wrapped {
        let q = BooleanQuery::new().with_must([clause]);
        assert_eq!(hist(&q, "p", 1000).unwrap(), in_range);
    }
    let two = BooleanQuery::new().with_must([
        int_range("p", 100, 2999),
        Clause::MatchAllDocs(MatchAllDocsQuery::new(0)),
    ]);
    assert_eq!(hist(&two, "p", 1000).unwrap(), in_range);
    // A range over another field is not the histogram's.
    let other = BooleanQuery::new().with_must([int_range("m", 0, 99)]);
    assert_eq!(hist(&other, "p", 1000).unwrap(), naive(0..100, 1000));

    // Two points per document: the values, once per document per bucket.
    let m = (0..i64::from(DOCS)).flat_map(|v| {
        if v.div_euclid(1000) == (v + 1).div_euclid(1000) {
            vec![v]
        } else {
            vec![v, v + 1]
        }
    });
    assert_eq!(hist(&all, "m", 1000).unwrap(), naive(m, 1000));

    // The skip index: whole blocks in one bucket, then one value a bucket,
    // then a field starting late.
    let n = || (0..i64::from(DOCS)).map(|v| v / 10);
    assert_eq!(hist(&all, "n", 100_000).unwrap(), naive(n(), 100_000));
    assert_eq!(hist(&all, "n", 7).unwrap(), naive(n(), 7));
    assert_eq!(
        hist(&all, "sp", 100).unwrap(),
        naive(4600..i64::from(DOCS), 100)
    );
}
