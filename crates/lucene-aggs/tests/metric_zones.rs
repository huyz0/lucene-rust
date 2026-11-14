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
    DateUnit, Globals, RoundingKind, TermsOrder,
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
use lucene_search::java_math::{java_max, java_min};
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
/// A level in runs of 3,000 documents: most windows hold one value.
const LEVEL: i32 = 8;
/// A `double` whose first two windows alternate `-0.0` and `0.5` (two
/// histogram buckets of key zero, told apart by sign), the third all
/// negative infinity (no histogram bucket), then tenths.
const SIGNED: i32 = 9;

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
    doc_values.push(long(LEVEL, (i64i / 3000) % 4));
    let signed = match i {
        0..2048 if i.is_multiple_of(2) => -0.0,
        0..2048 => 0.5,
        2048..3072 => f64::NEG_INFINITY,
        _ => (i % 50) as f64 / 10.0,
    };
    doc_values.push(long(SIGNED, sortable(signed)));
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
    w.register_field(dv("level", LEVEL, DocValuesType::SortedNumeric))
        .unwrap();
    w.register_field(dv("signed", SIGNED, DocValuesType::Numeric))
        .unwrap();
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
        for number in [TS, SMALL, WIDE, SPARSE, MULTI, DBL, LEVEL, SIGNED] {
            let (meta, _) = r.doc_values_for_field(number).unwrap();
            let entry = meta
                .numeric_entry(number)
                .or_else(|| meta.sorted_numeric_entry(number).map(|e| &e.numeric))
                .unwrap();
            let zones = entry.adaptive.as_ref().is_some_and(|a| a.zones);
            assert!(
                !zones || matches!(number, TS | SMALL | WIDE | DBL | LEVEL | SIGNED),
                "field {number}"
            );
            with_zones += usize::from(zones);
        }
    }
    with_zones
}

/// Document `i`'s stored value of field `number`, if it has one.
fn value_of(i: usize, number: i32) -> Option<i64> {
    doc(i)
        .fields
        .doc_values
        .iter()
        .find(|f| f.field_number == number)
        .map(|f| match f.value {
            FieldValue::Long(v) => v,
            _ => unreachable!("every test field is a long"),
        })
}

/// `CompensatedSum.add` (Java's order; non-finite values added straight).
fn compensated(sum: &mut f64, delta: &mut f64, v: f64) {
    if !v.is_finite() {
        let tally = v + *sum;
        *sum = if tally.is_nan() { f64::NAN } else { tally };
    }
    if sum.is_finite() {
        let corrected = v + *delta;
        let updated = *sum + corrected;
        *delta = corrected - (updated - *sum);
        *sum = updated;
    }
}

/// The single-valued fields' metrics over matches read a window at a time
/// (whole windows from statistics, dense spans with one window read, scattered
/// matches one by one), in both copies, against the documents' own values
/// folded in document order, for every part asked for alone or together.
/// Each case names its matches by their original id (the `id` term or the
/// `ts` value), and so does `live`: it drops the deleted ids. Each case
/// matches at most a quarter of either segment, so it is read by seeking.
fn check_seek_against_values(pair: &Pair, live: impl Fn(usize) -> bool) {
    let ids = |docs: Vec<usize>| {
        let mut q = BooleanQuery::new();
        q.should = docs
            .iter()
            .map(|i| Clause::Term(TermQuery::new("id", format!("{i}").as_bytes())))
            .collect();
        (q, docs)
    };
    let ts_value = |i: usize| value_of(i, TS).unwrap();
    let ts = |lo: usize, hi: usize| {
        let (lo, hi) = (
            1_700_000_000_000 + lo as i64 * 37,
            1_700_000_000_000 + hi as i64 * 37,
        );
        let q = clause_query(Clause::Extended(Box::new(
            ExtendedQuery::NumericDocValuesRange(NumericDocValuesRangeQuery::new("ts", lo, hi)),
        )));
        let docs = (0..DOCS)
            .filter(|&i| (lo..=hi).contains(&ts_value(i)))
            .collect();
        (q, docs)
    };
    let cases = [
        // The whole window 1024..2048 (its statistics), and dense partial
        // windows on both sides of it.
        ts(1_000, 2_100),
        // One document in five of a few hundred: one window read with gaps.
        ids((1_000..1_400).step_by(5).collect()),
        // One in fifty: one by one.
        ids((0..3_000).step_by(50).collect()),
        // Dense windows holding the deleted documents 5 and 4,100.
        ids((0..40).chain((4_064..4_200).step_by(2)).collect()),
        // Every document of an aligned span, first in the second segment
        // before the merge, then in the middle of the merged one.
        ids((6_144..6_400).collect()),
        // Across the first segment's end, and a dense run to the last
        // document.
        ids((6_000..6_300)
            .step_by(3)
            .chain((8_900..9_000).step_by(2))
            .chain([8_999])
            .collect()),
    ];
    let readers = pair
        .dirs
        .each_ref()
        .map(|d| DirectoryReader::open(d).unwrap());
    let mut opened = readers.each_ref().map(|r| r.open_segments().unwrap());
    let [o0, o1] = &mut opened;
    let segments: [Vec<OpenSegment<'_>>; 2] = [o0.as_open_segments(), o1.as_open_segments()];
    let fields = [
        ("ts", TS, ValueKind::Long),
        ("small", SMALL, ValueKind::Long),
        ("wide", WIDE, ValueKind::Long),
        ("sparse", SPARSE, ValueKind::Long),
        ("dbl", DBL, ValueKind::Double),
        ("signed", SIGNED, ValueKind::Double),
    ];
    let all_needs = [
        NEED_ALL,
        NEED_MIN,
        NEED_MAX,
        NEED_COUNT,
        NEED_SUM,
        NEED_COUNT | NEED_SUM,
        NEED_MIN_MAX,
    ];
    let specs: Vec<Vec<MetricSpec>> = all_needs
        .iter()
        .map(|&needs| {
            fields
                .iter()
                .map(|&(f, _, kind)| MetricSpec {
                    field: f.to_string(),
                    kind,
                    source: Source::DocValues,
                    needs,
                })
                .collect()
        })
        .collect();
    // The parts `needs` asks for; the others may stay at their defaults.
    let asked = |s: &MetricState, needs: u8| {
        let b = bits(s);
        let part = |bit: u8, v: u64| (needs & bit != 0).then_some(v);
        [
            part(NEED_COUNT, b[0]),
            part(NEED_SUM, b[1]),
            part(NEED_SUM, b[2]),
            part(NEED_MIN_MAX, b[3]),
            part(NEED_MIN_MAX, b[4]),
            part(NEED_MIN, b[5]),
            part(NEED_MAX, b[6]),
        ]
    };
    for (ci, (q, docs)) in cases.iter().enumerate() {
        for (fi, &(name, number, kind)) in fields.iter().enumerate() {
            let mut want = MetricState::default();
            for &i in docs.iter().filter(|&&i| live(i)) {
                let Some(v) = value_of(i, number) else {
                    continue;
                };
                let v = match kind {
                    ValueKind::Double => {
                        f64::from_bits((v ^ ((v >> 63) & 0x7fff_ffff_ffff_ffff)) as u64)
                    }
                    _ => v as f64,
                };
                want.count += 1;
                compensated(&mut want.sum, &mut want.delta, v);
                want.min = java_min(want.min, v);
                want.max = java_max(want.max, v);
            }
            want.min_of_mins = want.min;
            want.max_of_maxes = want.max;
            for (specs, &needs) in specs.iter().zip(&all_needs) {
                for k in 0..2 {
                    let got = metric_states(&segments[k], readers[k].segment_readers(), q, specs)
                        .unwrap();
                    assert_eq!(
                        asked(&got[fi], needs),
                        asked(&want, needs),
                        "case {ci} field {name} needs {needs} copy {k}:\n got  {:?}\n want {want:?}",
                        got[fi]
                    );
                }
            }
        }
    }
}

/// Top-level bucket aggregations (`date_histogram`, `histogram`, numeric
/// `terms`, `range`) with metric sub-aggregations, over every query and
/// slicing: the adaptive copy (whose whole windows that put every document
/// in the same buckets are folded from block statistics) answers as the
/// stock one, bit for bit.
fn check_buckets(pair: &Pair) {
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
        kind: if field == "dbl" || field == "signed" {
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
    let numeric_histogram =
        |field: &str, kind, interval, offset, hard_bounds, subs| AggNode::Histogram {
            field: field.to_string(),
            missing: None,
            kind,
            interval,
            offset,
            hard_bounds,
            subs,
        };
    let numeric_terms = |field: &str, kind, subs| AggNode::NumericTerms {
        field: field.to_string(),
        missing: None,
        kind,
        shard_size: 100,
        order: TermsOrder::CountDesc,
        subs,
    };
    let range = |field: &str, kind, ranges, subs| AggNode::Range {
        field: field.to_string(),
        missing: None,
        kind,
        ranges,
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
        // 9: a `histogram` whose keys hold whole windows.
        vec![numeric_histogram(
            "small",
            ValueKind::Long,
            50.0,
            0.0,
            (None, None),
            every_metric(),
        )],
        // 10: with an offset and hard bounds cutting the range.
        vec![numeric_histogram(
            "small",
            ValueKind::Long,
            25.0,
            3.0,
            (Some(-260.0), Some(-230.0)),
            every_metric(),
        )],
        // 11: `-0.0` and `0.0` round to two buckets; a window of negative
        // infinity makes none; a NaN window.
        vec![
            numeric_histogram(
                "signed",
                ValueKind::Double,
                10.0,
                0.0,
                (None, None),
                vec![metric("small", NEED_ALL)],
            ),
            numeric_histogram(
                "dbl",
                ValueKind::Double,
                50.0,
                0.0,
                (None, None),
                vec![metric("small", NEED_ALL)],
            ),
        ],
        // 12: numeric `terms` on runs of one value, long and double.
        vec![
            numeric_terms("level", ValueKind::Long, every_metric()),
            numeric_terms("signed", ValueKind::Double, Vec::new()),
        ],
        // 13: overlapping `range`s, open at either end.
        vec![range(
            "small",
            ValueKind::Long,
            vec![
                (f64::NEG_INFINITY, -250.0),
                (-260.0, -230.0),
                (-240.0, f64::INFINITY),
            ],
            every_metric(),
        )],
        // 14: a `double` range around both zeros and a NaN; a level range
        // whose windows fall wholly in a range or outside all of them (level
        // 1, in no range).
        vec![
            range(
                "dbl",
                ValueKind::Double,
                vec![(f64::NEG_INFINITY, 0.0), (0.0, 10.0), (5.0, f64::INFINITY)],
                vec![metric("small", NEED_ALL)],
            ),
            range(
                "level",
                ValueKind::Long,
                vec![(0.0, 1.0), (2.0, 3.0)],
                every_metric(),
            ),
        ],
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
        // Trees whose buckets hold whole windows: minute date buckets (with
        // or without hard bounds), and every other kind's tree.
        if matches!(t, 0 | 1 | 4 | 9..=14) {
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
    // Two segments, each field with statistics in both, but `level`, one
    // value throughout the second segment (Lucene90's constant wins there).
    assert_eq!(check(&pair), 11);
    check_buckets(&pair);
    check_seek_against_values(&pair, |_| true);

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
    check_buckets(&pair);
    check_seek_against_values(&pair, |i| i != 5 && i != 4_100);

    // Merged: the statistics are rewritten for the merged values.
    for w in &mut w {
        w.force_merge(1).unwrap();
        w.commit().unwrap();
    }
    assert_eq!(check(&pair), 6);
    check_buckets(&pair);
    check_seek_against_values(&pair, |i| i != 5 && i != 4_100);
}
