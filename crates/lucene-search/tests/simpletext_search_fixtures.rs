//! Searches over SimpleText segments, differentially against Lucene 10.5.0:
//! `fixtures/src/GenSimpleTextSearch.java`.
//!
//! `GenM7Queries`' three-segment, index-sorted index with deletions, written
//! by `SimpleTextCodec`, searched with every `GenM7Queries` query (synonyms,
//! combined fields, phrases, multi-phrases, fuzzy, the multi-term rewrite
//! methods, blended terms, Indri, log-odds fusion, Bayesian calibration,
//! doc-values and index-sort ranges, points, `IndexOrDocValuesQuery`, vector
//! thresholds, KNN as a clause), `GenSpans`' span queries, and sorted
//! searches (`TopFieldCollector`). Every hit and its score bits, or its sort
//! values, must be Java's.

// Test fixtures' own arithmetic -- see `docs/arithmetic-gate.md`'s "Test code".
#![allow(clippy::arithmetic_side_effects)]

mod m7grammar;

use std::collections::HashMap;

use lucene_codecs::vectors::FlatVectorsReader;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::extended_query::{MultiTermQuery, MultiTermSource, RewriteMethod};
use lucene_search::field_norms::FieldNorms;
use lucene_search::index_searcher::{IndexSearcher, SegmentNorms};
use lucene_search::multi_segment::{
    search_boolean_query_multi_segment_with_similarity, OpenSegment,
};
use lucene_search::query::{BooleanQuery, Clause, PrefixQuery, WildcardQuery};
use lucene_search::spans::SpanNode;
use lucene_search::top_field::{FieldDoc, Selector, SortField, SortType};
use lucene_search::vector_query::{KnnSegment, VectorsInput};
use lucene_store::FsDirectory;
use lucene_util::fixed_bit_set::FixedBitSet;
use m7grammar::{knn_clause_from, query, sim, Ctx, KnnSource, QVEC};

fn data() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/simpletext_search")
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

/// Each segment's vectors through its flat vectors reader, with no graph:
/// the exhaustive scan `SimpleTextKnnVectorsReader.search` is.
struct TextVectors<'r> {
    reader: &'r DirectoryReader,
    flat: Vec<Option<FlatVectorsReader<'r>>>,
}

impl KnnSource for TextVectors<'_> {
    fn knn_segments<'s>(&'s self, filters: Option<&'s [FixedBitSet]>) -> Vec<KnnSegment<'s>> {
        self.reader
            .segment_readers()
            .iter()
            .zip(&self.flat)
            .enumerate()
            .filter_map(|(i, (seg, flat))| {
                Some(KnnSegment {
                    vectors: VectorsInput {
                        flat: flat.clone()?,
                        hnsw: None,
                        field_infos: seg.field_infos(),
                        live_docs: seg.live_docs(),
                        filter: filters.map(|f| &f[i]),
                        max_doc: seg.max_doc,
                    },
                    doc_base: seg.doc_base,
                })
            })
            .collect()
    }
}

fn open() -> DirectoryReader {
    let reader = DirectoryReader::open(&FsDirectory::open(data().join("index"))).unwrap();
    assert_eq!(reader.segment_readers().len(), 3);
    assert!(reader
        .segment_readers()
        .iter()
        .all(|s| s.check_integrity().is_ok() && s.segment_name.starts_with('_')));
    reader
}

fn norms(reader: &DirectoryReader) -> Vec<HashMap<String, FieldNorms<'_>>> {
    reader
        .field_norms_by_field(&["body".to_string(), "title".to_string(), "gram".to_string()])
        .unwrap()
}

#[test]
fn m7_queries_over_simpletext_segments_match_lucene_bit_for_bit() {
    let text = std::fs::read_to_string(data().join("searches.tsv"))
        .expect("run scripts/gen-fixtures.sh --only GenSimpleTextSearch");
    let reader = open();
    let mut opened = reader.open_segments().unwrap();
    opened.open_points().unwrap();
    let segments: Vec<OpenSegment<'_>> = opened.as_open_segments();
    assert!(segments.iter().filter(|s| s.live_docs.is_some()).count() >= 2);
    let owned = norms(&reader);
    let norms: Vec<Option<&HashMap<String, FieldNorms<'_>>>> = owned.iter().map(Some).collect();
    let vectors = TextVectors {
        reader: &reader,
        flat: reader
            .segment_readers()
            .iter()
            .map(|seg| seg.flat_vectors_reader().unwrap())
            .collect(),
    };
    assert!(vectors.flat.iter().all(Option::is_some));
    let knn = |op: &str, field: &str, arg: &str, clause: Option<Clause>| -> Option<Clause> {
        knn_clause_from(op, field, arg, clause, &segments, &norms, &vectors, &QVEC)
    };
    let ctx = Ctx { knn: &knn };

    let (mut cases, mut failures) = (0, Vec::new());
    for line in text.lines() {
        let [name, q, hits] = line.split('\t').collect::<Vec<_>>()[..] else {
            panic!("bad line {line}");
        };
        let bq = query(q, &ctx).expect("every recorded search parses");
        let similarity = sim(name);
        let got = match search_boolean_query_multi_segment_with_similarity(
            &segments,
            &bq,
            &norms,
            20,
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
        if got != want(hits) {
            failures.push(format!(
                "{name} [{q}]\n  rust {got:x?}\n  java {:x?}",
                want(hits)
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} searches differ:\n{}",
        failures.len(),
        failures.join("\n")
    );
    assert!(cases >= 200, "{cases} searches");
}

/// `GenSpans.query`, as far as `GenSimpleTextSearch.SPANS` uses it.
fn span(spec: &str) -> SpanNode {
    let (name, args) = match spec.find('(') {
        None => (spec, Vec::new()),
        Some(p) => (&spec[..p], split(&spec[p + 1..spec.len() - 1])),
    };
    let int = |s: &str| -> i32 { s.parse().unwrap() };
    let spans = |from: usize| -> Vec<SpanNode> { args[from..].iter().map(|s| span(s)).collect() };
    let multi =
        |source| SpanNode::multi_term(MultiTermQuery::new(source, RewriteMethod::default()));
    match name {
        "t" => SpanNode::term(args[0], args[1]),
        "near" => SpanNode::near(spans(2), int(args[0]), args[1] == "true").unwrap(),
        "or" => SpanNode::or(spans(0)).unwrap(),
        "first" => SpanNode::first(span(args[1]), int(args[0])),
        "range" => SpanNode::position_range(span(args[2]), int(args[0]), int(args[1])),
        "not" => SpanNode::not(span(args[0]), span(args[1]), int(args[2]), int(args[3])).unwrap(),
        "containing" => SpanNode::containing(span(args[0]), span(args[1])).unwrap(),
        "within" => SpanNode::within(span(args[0]), span(args[1])).unwrap(),
        "prefix" => multi(MultiTermSource::Prefix(PrefixQuery::new(args[0], args[1]))),
        "wildcard" => multi(MultiTermSource::Wildcard(WildcardQuery::new(
            args[0], args[1],
        ))),
        _ => {
            assert!(args.is_empty(), "{spec}");
            SpanNode::term("body", spec)
        }
    }
}

fn split(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let (mut depth, mut start) = (0i32, 0usize);
    for (i, c) in s.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => depth -= 1,
            ',' if depth == 0 => {
                out.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    out.push(&s[start..]);
    out
}

#[test]
fn span_queries_over_simpletext_segments_match_lucene_bit_for_bit() {
    let text = std::fs::read_to_string(data().join("spans.tsv")).unwrap();
    let reader = open();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let owned = norms(&reader);
    let norms: Vec<SegmentNorms<'_, '_>> = owned.iter().map(Some).collect();
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();
    let mut failures = Vec::new();
    for line in text.lines() {
        let (spec, hits) = line.split_once('\t').unwrap();
        let q = BooleanQuery::new().with_must([Clause::from(span(spec))]);
        let got: Vec<(i32, u32)> = searcher
            .search(&q, 20)
            .unwrap()
            .score_docs
            .iter()
            .map(|h| (h.doc, h.score.to_bits()))
            .collect();
        if got != want(hits) {
            failures.push(format!("{spec}\n  rust {got:x?}\n  java {:x?}", want(hits)));
        }
    }
    assert_eq!(text.lines().count(), 13);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// `GenSortedSearch.spec`'s sort keys.
fn sort(spec: &str) -> Vec<SortField> {
    spec.split(',')
        .map(|k| {
            let p: Vec<&str> = k.split(':').collect();
            SortField {
                nested: None,
                rewritten: None,
                field: p[0].to_string(),
                ty: match p[1] {
                    "score" => SortType::Score,
                    "doc" => SortType::Doc,
                    "long" => SortType::Long,
                    other => panic!("type {other}"),
                },
                selector: if p[2] == "max" {
                    Selector::Max
                } else {
                    Selector::Min
                },
                reverse: p[3] == "true",
                missing: p[4].parse().unwrap(),
            }
        })
        .collect()
}

fn field_docs(s: &str) -> Vec<FieldDoc> {
    s.split(',')
        .filter(|s| !s.is_empty())
        .map(|h| {
            let mut parts = h.split(':');
            FieldDoc {
                doc: parts.next().unwrap().parse().unwrap(),
                values: parts.map(|v| v.parse().unwrap()).collect(),
                terms: Vec::new(),
            }
        })
        .collect()
}

#[test]
fn sorted_searches_over_simpletext_segments_match_lucene() {
    let text = std::fs::read_to_string(data().join("sorted.tsv")).unwrap();
    let reader = open();
    let mut opened = reader.open_segments().unwrap();
    opened.open_points().unwrap();
    let segments = opened.as_open_segments();
    let owned = norms(&reader);
    let norms: Vec<SegmentNorms<'_, '_>> = owned.iter().map(Some).collect();
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();
    let no_knn = |_: &str, _: &str, _: &str, _: Option<Clause>| -> Option<Clause> {
        unreachable!("no vector op is sorted")
    };
    let ctx = Ctx { knn: &no_knn };
    let mut failures = Vec::new();
    for line in text.lines() {
        let [q, spec, n, hits, total, relation] = line.split('\t').collect::<Vec<_>>()[..] else {
            panic!("bad line {line}");
        };
        let got = searcher
            .search_sorted(
                reader.segment_readers(),
                &query(q, &ctx).unwrap(),
                n.parse().unwrap(),
                &sort(spec),
                None,
            )
            .unwrap();
        let gte =
            got.total.relation == lucene_search::collector::TotalHitsRelation::GreaterThanOrEqualTo;
        let total_ok = if relation == "eq" {
            !gte && got.total.value.to_string() == total
        } else {
            gte && got.total.value >= 1000
        };
        if got.hits != field_docs(hits) || !total_ok {
            failures.push(format!(
                "[{q}] by {spec} top {n}\n  rust {:?} total {} gte {gte}\n  java {hits} total {total} {relation}",
                got.hits, got.total.value
            ));
        }
    }
    assert_eq!(text.lines().count(), 48);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
