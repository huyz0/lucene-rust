#![allow(clippy::arithmetic_side_effects)]
//! The aggregation segment cache answers exactly as the pass without it:
//! dashboard trees (`date_histogram`, `terms`, `range`, `filters`,
//! `histogram`, numeric `terms`, metrics) over time-range queries, across
//! refreshes that add segments, deletions, doc-values updates and a force
//! merge, over every slicing and over a reader holding only some of the
//! segments (other global ordinals for the same segment's terms).

use std::collections::HashMap;
use std::sync::Arc;

use lucene_codecs::field_infos::{DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions};
use lucene_codecs::stored_fields::{FieldValue, StoredField};
use lucene_index::buffered_updates::Term;
use lucene_index::index_writer::{
    ExplicitDocument, ExplicitFields, IndexWriter, InvertedField, InvertedTerm,
};
use lucene_index::segment_info::LuceneVersion;
use lucene_search::aggs::{self, Source, ValueKind};
use lucene_search::bucket_aggs::segment_cache::{AggSegmentCache, CacheRequest, SegmentIdentity};
use lucene_search::bucket_aggs::{
    aggregate_tree, aggregate_tree_cached, AggNode, AggResult, DateRounding, Globals, RoundingKind,
    TermsOrder,
};
use lucene_search::directory_reader::{DirectoryReader, SegmentReader};
use lucene_search::multi_segment::OpenSegment;
use lucene_search::query::{BoostQuery, ConstantScoreQuery, MatchAllDocsQuery, PointsRangeQuery};
use lucene_search::{BooleanQuery, Clause, TermQuery};
use lucene_store::FsDirectory;
use lucene_util::test_support::TempDir;

const VERSION: LuceneVersion = LuceneVersion {
    major: 10,
    minor: 5,
    bugfix: 0,
};

const HOST: i32 = 0;
const TS: i32 = 1;
const DUR: i32 = 2;
const BODY: i32 = 3;
const ID: i32 = 4;

fn term(t: &[u8]) -> InvertedTerm {
    InvertedTerm {
        term: t.to_vec(),
        freq: 1,
        positions: Vec::new(),
        offsets: Vec::new(),
        payloads: Vec::new(),
    }
}

/// Document `i`: a host among a growing set (later segments bring new
/// terms, which move the global ordinals of the earlier ones), a time that
/// grows with `i`, a duration (every 9th has none) and a body word.
fn doc(i: usize, big: bool) -> ExplicitDocument {
    // New hosts sort first: the earlier hosts' global ordinals move up.
    let host = format!("h{:02}", 50 - (i * 7) % (3 + i / 150));
    let ts = 1_000 + i as i64 * 10;
    let mut doc_values = vec![
        StoredField {
            field_number: HOST,
            value: FieldValue::Binary(host.clone().into_bytes()),
        },
        StoredField {
            field_number: TS,
            value: FieldValue::Long(ts),
        },
    ];
    if i % 9 != 8 {
        doc_values.push(StoredField {
            field_number: DUR,
            value: FieldValue::Long(((i * 31) % 97) as i64 - 20 + if big { 1 << 51 } else { 0 }),
        });
    }
    let mut packed = vec![0u8; 8];
    lucene_util::numeric_utils::long_to_sortable_bytes(ts, &mut packed, 0);
    ExplicitDocument {
        stored: Vec::new(),
        fields: ExplicitFields {
            inverted: vec![
                InvertedField {
                    field_number: HOST,
                    terms: vec![term(host.as_bytes())].into(),
                    norm: None,
                },
                InvertedField {
                    field_number: BODY,
                    terms: vec![term(format!("w{}", i % 4).as_bytes())].into(),
                    norm: None,
                },
                InvertedField {
                    field_number: ID,
                    terms: vec![term(format!("{i}").as_bytes())].into(),
                    norm: None,
                },
            ],
            doc_values,
            points: vec![StoredField {
                field_number: TS,
                value: FieldValue::Binary(packed),
            }],
            ..ExplicitFields::default()
        },
    }
}

fn writer(dir: &FsDirectory) -> IndexWriter<'_> {
    let mut w = IndexWriter::open(dir, Vec::new(), "Lucene104", VERSION).unwrap();
    w.enable_explicit_documents().unwrap();
    let docs = |name: &str, number| {
        FieldInfo::new(name, number)
            .with_index_options(IndexOptions::Docs)
            .with_omit_norms(true)
    };
    w.register_field(docs("host", HOST).with_doc_values(
        DocValuesType::SortedSet,
        DocValuesSkipIndexType::None,
        -1,
    ))
    .unwrap();
    w.register_field(
        FieldInfo::new("ts", TS)
            .with_points(1, 1, 8)
            .with_doc_values(DocValuesType::Numeric, DocValuesSkipIndexType::None, -1),
    )
    .unwrap();
    w.register_field(FieldInfo::new("dur", DUR).with_doc_values(
        DocValuesType::Numeric,
        DocValuesSkipIndexType::None,
        -1,
    ))
    .unwrap();
    w.register_field(docs("body", BODY)).unwrap();
    w.register_field(docs("id", ID)).unwrap();
    w
}

fn add(w: &mut IndexWriter<'_>, from: usize, to: usize, every: usize) {
    add_docs(w, from, to, every, false);
}

fn add_docs(w: &mut IndexWriter<'_>, from: usize, to: usize, every: usize, big: bool) {
    for i in from..to {
        w.add_explicit_documents(vec![doc(i, big)]).unwrap();
        if i % every == every - 1 {
            w.commit().unwrap();
        }
    }
    w.commit().unwrap();
}

fn metric(field: &str, needs: u8) -> AggNode {
    AggNode::Metric {
        field: field.to_string(),
        missing: None,
        kind: ValueKind::Long,
        source: Source::DocValues,
        needs,
        script: None,
    }
}

fn terms(field: &str, shard_size: usize, order: TermsOrder, subs: Vec<AggNode>) -> AggNode {
    AggNode::Terms {
        field: field.to_string(),
        missing: None,
        accepted: None,
        shard_size,
        order,
        zero_buckets: false,
        subs,
    }
}

fn date_histogram(interval: i64, subs: Vec<AggNode>) -> AggNode {
    AggNode::DateHistogram {
        field: "ts".to_string(),
        missing: None,
        rounding: DateRounding {
            kind: RoundingKind::Interval(interval),
            zone_ms: 0,
            offset: 0,
        },
        hard_bounds: (None, None),
        subs,
    }
}

fn t(field: &str, value: &str) -> Clause {
    Clause::Term(TermQuery::new(field, value.as_bytes()))
}

fn range(min: i64, max: i64) -> Clause {
    Clause::PointsRange(PointsRangeQuery::new("ts", min, max))
}

fn filtered(clauses: Vec<Clause>) -> BooleanQuery {
    let mut q = BooleanQuery::new();
    q.filter = clauses;
    q
}

/// Dashboard-shaped trees.
fn trees() -> Vec<Vec<AggNode>> {
    vec![
        // Count per interval per top host, with the hosts' durations.
        vec![date_histogram(
            700,
            vec![terms(
                "host",
                3,
                TermsOrder::CountDesc,
                vec![metric("dur", aggs::NEED_COUNT | aggs::NEED_SUM)],
            )],
        )],
        // Top hosts, each over time with stats; and a top-level count.
        vec![
            terms(
                "host",
                2,
                TermsOrder::CountDesc,
                vec![date_histogram(2_000, vec![metric("dur", aggs::NEED_ALL)])],
            ),
            metric("dur", aggs::NEED_COUNT),
        ],
        // Duration ranges by body word, a histogram of durations by time
        // value, and the top hosts by key.
        vec![
            AggNode::Range {
                field: "dur".to_string(),
                missing: Some(0),
                kind: ValueKind::Long,
                ranges: vec![(f64::NEG_INFINITY, 0.0), (0.0, 30.0), (30.0, f64::INFINITY)],
                subs: vec![AggNode::Filters {
                    filters: vec![
                        filtered(vec![t("body", "w1")]),
                        filtered(vec![t("body", "w2")]),
                    ],
                    other: true,
                    subs: vec![metric("dur", aggs::NEED_MIN | aggs::NEED_MAX)],
                }],
            },
            AggNode::Histogram {
                field: "dur".to_string(),
                missing: None,
                kind: ValueKind::Long,
                interval: 7.0,
                offset: 1.0,
                hard_bounds: (None, None),
                subs: vec![AggNode::NumericTerms {
                    field: "ts".to_string(),
                    missing: None,
                    kind: ValueKind::Long,
                    shard_size: 2,
                    order: TermsOrder::KeyDesc,
                    subs: Vec::new(),
                }],
            },
            terms("host", 4, TermsOrder::KeyAsc, Vec::new()),
        ],
        // A top-level date_histogram alone, counted from the points where
        // every document of a segment matches.
        vec![date_histogram(1_000, Vec::new())],
    ]
}

/// Last-N windows sliding forward, a whole-index range, a term with a
/// range, and match-all.
fn queries(end: i64) -> Vec<BooleanQuery> {
    let mut all = BooleanQuery::new();
    all.must
        .push(Clause::MatchAllDocs(MatchAllDocsQuery::new(i32::MAX)));
    let mut out = vec![all, filtered(vec![range(i64::MIN, i64::MAX)])];
    for (span, step) in [(2_500, 0), (2_500, 30), (6_000, 70), (40_000, 10)] {
        out.push(filtered(vec![range(end - span + step, end + step)]));
    }
    out.push(filtered(vec![t("body", "w3"), range(end - 5_000, end)]));
    out.push(filtered(vec![t("host", "h49")]));
    // Ranges under a boolean, a constant score and a boost; a range on a
    // field without points.
    out.push(filtered(vec![Clause::Boolean(Box::new(filtered(vec![
        range(end - 3_000, end),
    ])))]));
    out.push(filtered(vec![Clause::ConstantScore(Box::new(
        ConstantScoreQuery::new(range(end - 4_000, end + 50), 1.0),
    ))]));
    out.push(filtered(vec![Clause::Boost(Box::new(BoostQuery::new(
        Clause::Boolean(Box::new(filtered(vec![
            t("body", "w1"),
            range(end - 2_000, end),
        ]))),
        2.0,
    )))]));
    out.push(filtered(vec![Clause::PointsRange(PointsRangeQuery::new(
        "dur", 0, 100,
    ))]));
    out
}

/// Every segment's identity, as the reader's live documents have it.
fn identities(
    segments: &[OpenSegment<'_>],
    readers: &[SegmentReader],
) -> Vec<Option<SegmentIdentity>> {
    readers
        .iter()
        .zip(segments)
        .map(|(r, s)| Some(SegmentIdentity::of(r, s.live_docs)))
        .collect()
}

/// Checks every tree and query over `reader`'s segments `keep`, sliced
/// whole and in two, against the pass without the cache, twice (the second
/// from the cache).
fn check(reader: &DirectoryReader, keep: &[usize], cache: &AggSegmentCache, end: i64) {
    let mut opened = reader.open_segments().unwrap();
    opened.open_points().unwrap();
    let segments: Vec<OpenSegment<'_>> = opened
        .as_open_segments()
        .into_iter()
        .enumerate()
        .filter(|(i, _)| keep.contains(i))
        .map(|(_, s)| s)
        .collect();
    let readers: Vec<SegmentReader> = keep
        .iter()
        .map(|&i| reader.segment_readers()[i].clone())
        .collect();
    let mut ords = HashMap::new();
    ords.insert(
        "host".to_string(),
        Arc::new(lucene_search::terms_agg::GlobalOrds::build(&readers, "host").unwrap()),
    );
    let globals = Globals::new(&ords);
    let ids = identities(&segments, &readers);
    let request = CacheRequest {
        cache,
        segments: &ids,
    };
    let n = segments.len();
    let whole: Vec<usize> = (0..n).collect();
    let halves = vec![
        (0..n).filter(|i| i % 2 == 0).collect::<Vec<_>>(),
        (0..n).filter(|i| i % 2 == 1).collect(),
    ];
    for (t, nodes) in trees().iter().enumerate() {
        for (q, query) in queries(end).iter().enumerate() {
            for slices in [vec![whole.clone()], halves.clone()] {
                let plain: Vec<Vec<AggResult>> =
                    aggregate_tree(&segments, &readers, query, nodes, &globals, &slices, None)
                        .unwrap();
                for round in 0..2 {
                    let cached = aggregate_tree_cached(
                        &segments,
                        &readers,
                        query,
                        nodes,
                        &globals,
                        &slices,
                        None,
                        Some(&request),
                    )
                    .unwrap();
                    assert_eq!(
                        cached, plain,
                        "tree {t}, query {q}, slices {slices:?}, segments {keep:?}, round {round}"
                    );
                }
            }
        }
    }
}

fn open(dir: &FsDirectory) -> DirectoryReader {
    DirectoryReader::open(dir).unwrap()
}

#[test]
fn cached_parts_answer_as_collecting_does_through_refreshes_deletes_updates_and_merges() {
    let cache = AggSegmentCache::new(64 << 20);
    let merged = phases("agg-segment-cache", &cache);
    // The merged segment, under 2^16 documents, is one entry per tree and
    // query.
    assert!(merged.entries <= 5 * 9, "{merged:?}");
}

/// The same with blocks of 64 documents: a range cutting a segment keys
/// each block by its own values and live documents.
#[test]
fn cached_blocks_answer_as_collecting_does() {
    let cache = AggSegmentCache::with_block_docs(64 << 20, 64);
    let merged = phases("agg-segment-cache-blocks", &cache);
    // The merged segment's 800 documents in blocks, for the ranges cutting
    // it.
    assert!(merged.entries > 5 * 9, "{merged:?}");
}

/// Refreshes, deletes, a doc-values update and a merge, each checked; what
/// the cache holds for the merged segment alone.
fn phases(
    name: &str,
    cache: &AggSegmentCache,
) -> lucene_search::bucket_aggs::segment_cache::CacheStats {
    let tmp = TempDir::new(name);
    let dir = FsDirectory::open(tmp.path());
    let mut w = writer(&dir);
    add(&mut w, 0, 600, 150);
    let r = open(&dir);
    assert!(
        r.segment_readers().len() >= 3,
        "segments {}",
        r.segment_readers().len()
    );
    let every: Vec<usize> = (0..r.segment_readers().len()).collect();
    check(&r, &every, cache, 1_000 + 600 * 10);
    let first = cache.stats();
    assert!(
        first.hits > 0 && first.misses > 0 && first.entries > 0,
        "{first:?}"
    );

    // A refresh adds segments: the old ones' parts are read back, under the
    // new reader's global ordinals.
    add(&mut w, 600, 800, 100);
    let r = open(&dir);
    let every: Vec<usize> = (0..r.segment_readers().len()).collect();
    let before = cache.stats();
    check(&r, &every, cache, 1_000 + 800 * 10);
    let after = cache.stats();
    assert!(after.hits > before.hits);
    // Only some segments: other global ordinals again.
    check(&r, &every[1..], cache, 1_000 + 800 * 10);
    check(
        &r,
        &[every[0], every[every.len() - 1]],
        cache,
        1_000 + 800 * 10,
    );

    // Deletions in two segments.
    w.delete_documents_by_term(&[
        Term::new("id", b"3".as_slice()),
        Term::new("id", b"640".as_slice()),
        Term::new("body", b"w2".as_slice()),
    ])
    .unwrap();
    w.commit().unwrap();
    let r = open(&dir);
    let every: Vec<usize> = (0..r.segment_readers().len()).collect();
    check(&r, &every, cache, 1_000 + 800 * 10);

    // A doc-values update.
    for id in ["5", "7", "777"] {
        w.update_numeric_doc_value(Term::new("id", id.as_bytes()), "dur", 1_000)
            .unwrap();
    }
    w.commit().unwrap();
    let r = open(&dir);
    let every: Vec<usize> = (0..r.segment_readers().len()).collect();
    check(&r, &every, cache, 1_000 + 800 * 10);

    // A force merge.
    w.force_merge(1).unwrap();
    w.commit().unwrap();
    let r = open(&dir);
    assert_eq!(r.segment_readers().len(), 1);
    cache.clear();
    check(&r, &[0], cache, 1_000 + 800 * 10);
    let merged = cache.stats();

    // A small budget keeps the cache within it.
    cache.set_budget(4 << 10);
    let s = cache.stats();
    assert!(s.bytes <= 4 << 10, "{s:?}");
    assert!(s.evictions > 0);
    merged
}

/// Durations near 2^51: a part's sum could round differently added in
/// another order, so the pass collects without the cache, and answers as
/// it does.
#[test]
fn a_sum_that_could_round_runs_without_the_cache() {
    let tmp = TempDir::new("agg-segment-cache-big");
    let dir = FsDirectory::open(tmp.path());
    let cache = AggSegmentCache::new(64 << 20);
    let mut w = writer(&dir);
    add_docs(&mut w, 0, 300, 100, true);
    let r = open(&dir);
    let mut opened = r.open_segments().unwrap();
    opened.open_points().unwrap();
    let segments = opened.as_open_segments();
    let readers = r.segment_readers();
    let ords = HashMap::new();
    let globals = Globals::new(&ords);
    let ids = identities(&segments, readers);
    let request = CacheRequest {
        cache: &cache,
        segments: &ids,
    };
    let mut all = BooleanQuery::new();
    all.must
        .push(Clause::MatchAllDocs(MatchAllDocsQuery::new(i32::MAX)));
    let slices = vec![(0..segments.len()).collect::<Vec<_>>()];
    for needs in [
        aggs::NEED_SUM,
        aggs::NEED_COUNT | aggs::NEED_SUM,
        aggs::NEED_ALL,
    ] {
        let nodes = vec![date_histogram(1_000, vec![metric("dur", needs)])];
        let plain =
            aggregate_tree(&segments, readers, &all, &nodes, &globals, &slices, None).unwrap();
        for _ in 0..2 {
            let cached = aggregate_tree_cached(
                &segments,
                readers,
                &all,
                &nodes,
                &globals,
                &slices,
                None,
                Some(&request),
            )
            .unwrap();
            assert_eq!(cached, plain, "needs {needs}");
        }
    }
}

/// A budget a few parts fill: parts are dropped as new ones come in, a part
/// past an eighth of the budget is not kept, and the answers stay the same.
#[test]
fn a_small_budget_evicts_as_parts_come_in() {
    let tmp = TempDir::new("agg-segment-cache-small");
    let dir = FsDirectory::open(tmp.path());
    let cache = AggSegmentCache::new(48 << 10);
    let mut w = writer(&dir);
    add(&mut w, 0, 600, 150);
    let r = open(&dir);
    let every: Vec<usize> = (0..r.segment_readers().len()).collect();
    check(&r, &every, &cache, 1_000 + 600 * 10);
    let s = cache.stats();
    assert!(s.evictions > 0 && s.bytes <= 48 << 10, "{s:?}");
    // The process's cache is off until given a budget.
    assert_eq!(AggSegmentCache::global().budget(), 0);
}
