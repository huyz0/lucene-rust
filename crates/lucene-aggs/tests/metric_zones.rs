#![allow(clippy::arithmetic_side_effects)]
//! Metrics read from `Adaptive90`'s per-1024-document statistics come out
//! bit for bit as the value-by-value fold over a stock Lucene90 copy of the
//! same documents: whole segments and ranges (the streamed and the seeking
//! read), every part asked for alone, a running sum past `2^53` and values
//! too wide for an exact window sum (both fall back to the values), a
//! `double` field (extremes and counts from the windows, NaN windows read),
//! sparse
//! and multi-valued fields (no statistics), deletions, and a merge.

use std::collections::HashMap;

use lucene_aggs::bucket_aggs::segment_cache::{AggSegmentCache, CacheRequest, SegmentIdentity};
use lucene_aggs::bucket_aggs::{
    aggregate_tree, aggregate_tree_cached, zone_windows_folded, AggNode, AggResult, DateRounding,
    DateUnit, Globals, RoundingKind,
};
use lucene_aggs::metric_aggs::{
    metric_states, MetricSpec, MetricState, Source, ValueKind, NEED_ALL, NEED_COUNT, NEED_MAX,
    NEED_MIN, NEED_MIN_MAX, NEED_SUM,
};
use lucene_codecs::field_infos::{DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions};
use lucene_codecs::per_field_doc_values::Lucene90DocValuesFormat;
use lucene_codecs::stored_fields::{FieldValue, StoredField};
use lucene_index::buffered_updates::Term;
use lucene_index::index_writer::{
    ExplicitDocument, ExplicitFields, IndexWriter, InvertedField, InvertedTerm,
};
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::extended_query::{ExtendedQuery, NumericDocValuesRangeQuery};
use lucene_search::multi_segment::OpenSegment;
use lucene_search::query::MatchAllDocsQuery;
use lucene_search::segment_match::MinScore;
use lucene_search::{BooleanQuery, Clause, TermQuery};
use lucene_store::FsDirectory;
use lucene_util::test_support::TempDir;

const VERSION: LuceneVersion = LuceneVersion {
    major: 10,
    minor: 5,
    bugfix: 0,
};

/// A timestamp: every window's sum is exact, the running sum passes `2^53`
/// after a few windows.
const TS: i32 = 0;
/// Small values that drift (smaller as adaptive blocks than as one Lucene90
/// array), as a SORTED_NUMERIC field with one value per document.
const SMALL: i32 = 1;
/// Values near `-2^50`: a window of them may not sum exactly as doubles.
const WIDE: i32 = 2;
/// Every 9th document has none.
const SPARSE: i32 = 3;
/// Every 5th document has two.
const MULTI: i32 = 4;
/// A `double` (no window sums: its sortable longs overflow), with a NaN in
/// one window and both zeros in another.
const DBL: i32 = 5;
const ID: i32 = 6;
const BODY: i32 = 7;

/// `NumericUtils.doubleToSortableLong`.
fn sortable(d: f64) -> i64 {
    let bits = d.to_bits() as i64;
    bits ^ ((bits >> 63) & 0x7fff_ffff_ffff_ffff)
}

const DOCS: usize = 9_000;

fn term(t: &[u8]) -> InvertedTerm {
    InvertedTerm {
        term: t.to_vec(),
        freq: 1,
        positions: Vec::new(),
        offsets: Vec::new(),
        payloads: Vec::new(),
    }
}

fn long(field_number: i32, v: i64) -> StoredField {
    StoredField {
        field_number,
        value: FieldValue::Long(v),
    }
}

fn doc(i: usize) -> ExplicitDocument {
    let i64i = i as i64;
    let mut doc_values = vec![
        long(TS, 1_700_000_000_000 + i64i * 37 + (i64i * 7919) % 1000),
        long(SMALL, (i64i / 100) % 1000 - 300 + (i64i * 31) % 3),
        long(WIDE, -(1 << 50) + i64i * 1000 + (i64i * 13) % 101),
    ];
    if i % 9 != 8 {
        doc_values.push(long(SPARSE, (i64i * 17) % 1000));
    }
    let d = match i {
        2_000 => f64::NAN,
        5_000 => -0.0,
        5_001 => 0.0,
        _ => (i64i % 4000) as f64 / 100.0 - 20.0,
    };
    doc_values.push(long(DBL, sortable(d)));
    doc_values.push(long(MULTI, i64i));
    if i.is_multiple_of(5) {
        doc_values.push(long(MULTI, i64i + 3));
    }
    ExplicitDocument {
        stored: Vec::new(),
        fields: ExplicitFields {
            inverted: vec![
                InvertedField {
                    field_number: ID,
                    terms: vec![term(format!("{i}").as_bytes())].into(),
                    norm: None,
                },
                InvertedField {
                    field_number: BODY,
                    terms: vec![term(format!("w{}", (i / 3000) % 3).as_bytes())].into(),
                    norm: None,
                },
            ],
            doc_values,
            ..ExplicitFields::default()
        },
    }
}

fn writer(dir: &FsDirectory, format: Lucene90DocValuesFormat) -> IndexWriter<'_> {
    let mut w = IndexWriter::open(dir, Vec::new(), "Lucene104", VERSION).unwrap();
    w.enable_explicit_documents().unwrap();
    w.set_default_doc_values_format(format);
    let dv = |name: &str, number, t| {
        FieldInfo::new(name, number).with_doc_values(t, DocValuesSkipIndexType::None, -1)
    };
    w.register_field(dv("ts", TS, DocValuesType::Numeric))
        .unwrap();
    w.register_field(dv("small", SMALL, DocValuesType::SortedNumeric))
        .unwrap();
    w.register_field(dv("wide", WIDE, DocValuesType::Numeric))
        .unwrap();
    w.register_field(dv("sparse", SPARSE, DocValuesType::Numeric))
        .unwrap();
    w.register_field(dv("multi", MULTI, DocValuesType::SortedNumeric))
        .unwrap();
    w.register_field(dv("dbl", DBL, DocValuesType::Numeric))
        .unwrap();
    for (name, number) in [("id", ID), ("body", BODY)] {
        w.register_field(
            FieldInfo::new(name, number)
                .with_index_options(IndexOptions::Docs)
                .with_omit_norms(true),
        )
        .unwrap();
    }
    w
}

/// Both copies, built in lockstep.
struct Pair {
    _tmp: [TempDir; 2],
    dirs: [FsDirectory; 2],
}

impl Pair {
    fn new(name: &str) -> Pair {
        let tmp = [
            TempDir::new(&format!("{name}-stock")),
            TempDir::new(&format!("{name}-adaptive")),
        ];
        let dirs = [
            FsDirectory::open(tmp[0].path()),
            FsDirectory::open(tmp[1].path()),
        ];
        Pair { _tmp: tmp, dirs }
    }
}

fn clause_query(c: Clause) -> BooleanQuery {
    let mut q = BooleanQuery::new();
    q.filter = vec![c];
    q
}

fn queries() -> Vec<BooleanQuery> {
    let ts = |lo: usize, hi: usize| {
        clause_query(Clause::Extended(Box::new(
            ExtendedQuery::NumericDocValuesRange(NumericDocValuesRangeQuery::new(
                "ts",
                1_700_000_000_000 + lo as i64 * 37,
                1_700_000_000_000 + hi as i64 * 37,
            )),
        )))
    };
    vec![
        clause_query(Clause::MatchAllDocs(MatchAllDocsQuery::new(i32::MAX))),
        // Most of the segment (streamed), a few windows (seeking), less
        // than a window.
        ts(500, 8_200),
        ts(1_000, 3_500),
        ts(1_000, 2_100),
        ts(2_100, 2_600),
        clause_query(Clause::Term(TermQuery::new("body", b"w1".as_slice()))),
    ]
}

fn bits(s: &MetricState) -> [u64; 7] {
    [
        s.count,
        s.sum.to_bits(),
        s.delta.to_bits(),
        s.min.to_bits(),
        s.max.to_bits(),
        s.min_of_mins.to_bits(),
        s.max_of_maxes.to_bits(),
    ]
}

/// Every field, every part, every query: the adaptive copy answers as the
/// stock one. Returns how many fields of the adaptive copy's segments carry
/// statistics.
fn check(pair: &Pair) -> usize {
    let readers = [
        DirectoryReader::open(&pair.dirs[0]).unwrap(),
        DirectoryReader::open(&pair.dirs[1]).unwrap(),
    ];
    let mut opened = [
        readers[0].open_segments().unwrap(),
        readers[1].open_segments().unwrap(),
    ];
    let [o0, o1] = &mut opened;
    let segments: [Vec<OpenSegment<'_>>; 2] = [o0.as_open_segments(), o1.as_open_segments()];
    for needs in [
        NEED_ALL,
        NEED_MIN,
        NEED_MAX,
        NEED_COUNT,
        NEED_SUM,
        NEED_COUNT | NEED_SUM,
        NEED_MIN_MAX,
    ] {
        let specs: Vec<MetricSpec> = ["ts", "small", "wide", "sparse", "multi", "dbl"]
            .iter()
            .map(|&f| MetricSpec {
                field: f.to_string(),
                kind: if f == "dbl" {
                    ValueKind::Double
                } else {
                    ValueKind::Long
                },
                source: Source::DocValues,
                needs,
            })
            .collect();
        for (qi, q) in queries().iter().enumerate() {
            let [stock, adaptive] = [0, 1].map(|k| {
                metric_states(&segments[k], readers[k].segment_readers(), q, &specs).unwrap()
            });
            for (f, (s, a)) in stock.iter().zip(&adaptive).enumerate() {
                assert_eq!(
                    bits(s),
                    bits(a),
                    "query {qi} field {} needs {needs}:\n stock    {s:?}\n adaptive {a:?}",
                    specs[f].field
                );
            }
            assert!(
                needs & NEED_COUNT == 0 || stock[0].count > 0,
                "query {qi} matches nothing"
            );
        }
    }
    let mut with_zones = 0;
    for r in readers[1].segment_readers() {
        for number in [TS, SMALL, WIDE, SPARSE, MULTI, DBL] {
            let (meta, _) = r.doc_values_for_field(number).unwrap();
            let entry = meta
                .numeric_entry(number)
                .or_else(|| meta.sorted_numeric_entry(number).map(|e| &e.numeric))
                .unwrap();
            let zones = entry.adaptive.as_ref().is_some_and(|a| a.zones);
            assert!(
                !zones || matches!(number, TS | SMALL | WIDE | DBL),
                "field {number}"
            );
            with_zones += usize::from(zones);
        }
    }
    with_zones
}

/// `date_histogram`s with metric sub-aggregations, over every query and
/// slicing: the adaptive copy (whose whole windows inside one bucket are
/// folded from block statistics) answers as the stock one, bit for bit.
fn check_histograms(pair: &Pair) {
    let readers = [
        DirectoryReader::open(&pair.dirs[0]).unwrap(),
        DirectoryReader::open(&pair.dirs[1]).unwrap(),
    ];
    let mut opened = [
        readers[0].open_segments().unwrap(),
        readers[1].open_segments().unwrap(),
    ];
    let [o0, o1] = &mut opened;
    let segments: [Vec<OpenSegment<'_>>; 2] = [o0.as_open_segments(), o1.as_open_segments()];
    let metric = |field: &str, needs: u8| AggNode::Metric {
        field: field.to_string(),
        missing: None,
        kind: if field == "dbl" {
            ValueKind::Double
        } else {
            ValueKind::Long
        },
        source: Source::DocValues,
        needs,
        script: None,
    };
    // Every part of every field with statistics, but no `double` sum
    // (which needs the values: a window holding one is collected) and no
    // `wide` sum (past the exact-integer bound, so also collected).
    let every_metric = || {
        let mut subs = Vec::new();
        for f in ["ts", "small", "wide", "dbl"] {
            for needs in [
                NEED_ALL,
                NEED_MIN,
                NEED_MAX,
                NEED_COUNT,
                NEED_COUNT | NEED_SUM,
            ] {
                if (f != "dbl" && f != "wide") || needs & NEED_SUM == 0 {
                    subs.push(metric(f, needs));
                }
            }
        }
        subs
    };
    let histogram = |kind, zone_ms, offset, hard_bounds, subs| AggNode::DateHistogram {
        field: "ts".to_string(),
        missing: None,
        rounding: DateRounding {
            kind,
            zone_ms,
            offset,
        },
        hard_bounds,
        subs,
    };
    let t0 = 1_700_000_000_000i64;
    let trees = vec![
        // Minutes: about 1600 documents a bucket, so some windows inside
        // one and some across two.
        vec![histogram(
            RoundingKind::Interval(60_000),
            0,
            0,
            (None, None),
            every_metric(),
        )],
        // Calendar minutes in a shifted zone, and an offset.
        vec![histogram(
            RoundingKind::Unit(DateUnit::Minute),
            5_400_000 + 17_000,
            0,
            (None, None),
            every_metric(),
        )],
        vec![histogram(
            RoundingKind::Unit(DateUnit::Hour),
            0,
            1_234,
            (None, None),
            every_metric(),
        )],
        // Ten seconds: no window inside one bucket.
        vec![histogram(
            RoundingKind::Interval(10_000),
            0,
            0,
            (None, None),
            every_metric(),
        )],
        // Hard bounds cutting the range: whole windows fall in buckets
        // both inside and outside them.
        vec![histogram(
            RoundingKind::Interval(60_000),
            0,
            0,
            (Some(t0 + 60_000), Some(t0 + 240_000)),
            every_metric(),
        )],
        // A sparse metric (no statistics) beside one with them: the window
        // is collected; and a histogram with no sub-aggregation, beside a
        // top-level metric.
        vec![
            histogram(
                RoundingKind::Interval(60_000),
                0,
                0,
                (None, None),
                vec![metric("sparse", NEED_ALL), metric("small", NEED_ALL)],
            ),
            histogram(
                RoundingKind::Interval(120_000),
                0,
                0,
                (None, None),
                Vec::new(),
            ),
            metric("small", NEED_COUNT | NEED_SUM),
        ],
        // A `double` sum beside the rest: its windows are collected.
        vec![histogram(
            RoundingKind::Interval(60_000),
            0,
            0,
            (None, None),
            vec![metric("dbl", NEED_ALL), metric("small", NEED_ALL)],
        )],
        // A `wide` sum, past the exact-integer bound: its windows are
        // collected.
        vec![histogram(
            RoundingKind::Interval(60_000),
            0,
            0,
            (None, None),
            vec![
                metric("wide", NEED_ALL),
                metric("ts", NEED_COUNT | NEED_SUM),
            ],
        )],
        // Nested: the inner histogram's owner varies, so it is collected.
        vec![histogram(
            RoundingKind::Interval(120_000),
            0,
            0,
            (None, None),
            vec![histogram(
                RoundingKind::Interval(60_000),
                0,
                0,
                (None, None),
                every_metric(),
            )],
        )],
    ];
    let ords = HashMap::new();
    let globals = Globals::new(&ords);
    let n = segments[0].len();
    let slicings = [
        vec![(0..n).collect::<Vec<_>>()],
        (0..n).map(|i| vec![i]).collect::<Vec<_>>(),
    ];
    for (t, nodes) in trees.iter().enumerate() {
        let folded = zone_windows_folded();
        for (qi, q) in queries().iter().enumerate() {
            for slices in &slicings {
                let [stock, adaptive]: [Vec<Vec<AggResult>>; 2] = [0, 1].map(|k| {
                    aggregate_tree(
                        &segments[k],
                        readers[k].segment_readers(),
                        q,
                        nodes,
                        &globals,
                        slices,
                        None,
                    )
                    .unwrap()
                });
                assert_eq!(
                    format!("{stock:?}"),
                    format!("{adaptive:?}"),
                    "tree {t} query {qi} slices {slices:?}"
                );
            }
        }
        // Minute buckets (with or without hard bounds) hold whole windows.
        if matches!(t, 0 | 1 | 4) {
            assert!(zone_windows_folded() > folded, "tree {t} folded no window");
        }
    }
    // Scored matches (a `min_score`, listing the documents) and the
    // per-segment cache, which keeps the folded states.
    let tree = &trees[0];
    let norms = vec![None; n];
    let min_score = MinScore {
        min: 0.0,
        norms: &norms,
    };
    let cache = [
        AggSegmentCache::new(64 << 20),
        AggSegmentCache::new(64 << 20),
    ];
    let ids: [Vec<Option<SegmentIdentity>>; 2] = [0, 1].map(|k| {
        readers[k]
            .segment_readers()
            .iter()
            .zip(&segments[k])
            .map(|(r, s)| Some(SegmentIdentity::of(r, s.live_docs)))
            .collect()
    });
    for (qi, q) in queries().iter().enumerate() {
        for slices in &slicings {
            let folded = zone_windows_folded();
            let [stock, adaptive]: [Vec<Vec<AggResult>>; 2] = [0, 1].map(|k| {
                aggregate_tree(
                    &segments[k],
                    readers[k].segment_readers(),
                    q,
                    tree,
                    &globals,
                    slices,
                    Some(&min_score),
                )
                .unwrap()
            });
            assert_eq!(
                format!("{stock:?}"),
                format!("{adaptive:?}"),
                "min_score query {qi} slices {slices:?}"
            );
            if qi == 0 {
                assert!(zone_windows_folded() > folded, "min_score folded no window");
            }
            for pass in 0..2 {
                let [stock, adaptive]: [Vec<Vec<AggResult>>; 2] = [0, 1].map(|k| {
                    let request = CacheRequest {
                        cache: &cache[k],
                        segments: &ids[k],
                    };
                    aggregate_tree_cached(
                        &segments[k],
                        readers[k].segment_readers(),
                        q,
                        tree,
                        &globals,
                        slices,
                        None,
                        Some(&request),
                    )
                    .unwrap()
                });
                assert_eq!(
                    format!("{stock:?}"),
                    format!("{adaptive:?}"),
                    "cached pass {pass} query {qi} slices {slices:?}"
                );
            }
        }
    }
}

#[test]
fn zone_folds_answer_bit_for_bit_as_the_values() {
    let pair = Pair::new("metric-zones");
    let mut w = [
        writer(&pair.dirs[0], Lucene90DocValuesFormat::default()),
        writer(&pair.dirs[1], Lucene90DocValuesFormat::rust_adaptive()),
    ];
    for w in &mut w {
        for i in 0..DOCS {
            w.add_explicit_documents(vec![doc(i)]).unwrap();
            if i == 6_143 {
                w.commit().unwrap();
            }
        }
        w.commit().unwrap();
    }
    // Two segments, each field with statistics in both.
    assert_eq!(check(&pair), 8);
    check_histograms(&pair);

    // Deletions: the windows holding them are read value by value.
    for w in &mut w {
        w.delete_documents_by_term(&[
            Term::new("id", b"5".as_slice()),
            Term::new("id", b"4100".as_slice()),
        ])
        .unwrap();
        w.commit().unwrap();
    }
    check(&pair);
    check_histograms(&pair);

    // Merged: the statistics are rewritten for the merged values.
    for w in &mut w {
        w.force_merge(1).unwrap();
        w.commit().unwrap();
    }
    assert_eq!(check(&pair), 4);
    check_histograms(&pair);
}
