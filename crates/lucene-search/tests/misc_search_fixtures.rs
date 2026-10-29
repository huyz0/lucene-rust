//! M14 T14.3 `misc`, differentially against Lucene 10.5.0:
//! `fixtures/src/GenMiscSearch.java`'s `misc/search.tsv` over
//! `misc/search_src` (two segments): the doc-values statistics, the
//! diversified top-docs collector (one collector, and one per segment reduced
//! by the manager) and the memory-accounting bitset collector.
#![allow(clippy::arithmetic_side_effects)]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use lucene_search::collector::ScoringCollector;
use lucene_search::collectors::CollectorManager;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::index_searcher::IndexSearcher;
use lucene_search::misc::memory::{CollectorMemoryTracker, MemoryTracker};
use lucene_search::misc::search::{
    DiversifiedTopDocs, DiversifiedTopDocsCollector, DiversifiedTopDocsCollectorManager,
    DocValuesStats, DocValuesStatsCollectorManager, DoubleDocValuesStats, KeySource, Leaves,
    LongDocValuesStats, MemoryAccountingBitsetCollectorManager, SortedDocValuesStats,
    SortedDoubleDocValuesStats, SortedLongDocValuesStats, SortedSetDocValuesStats,
};
use lucene_search::query::{BooleanQuery, Clause, TermQuery};
use lucene_search::reader::LeafReader;
use lucene_store::FsDirectory;

fn data() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/misc")
}

fn query(term: &str) -> BooleanQuery {
    BooleanQuery {
        must: vec![Clause::Term(TermQuery::new(
            "txt",
            term.as_bytes().to_vec(),
        ))],
        ..Default::default()
    }
}

fn hex(b: Option<&[u8]>) -> String {
    b.map_or("null".to_string(), |b| {
        b.iter().map(|x| format!("{x:02x}")).collect()
    })
}

fn bits(v: f64) -> String {
    format!("{:x}", v.to_bits())
}

macro_rules! with_searcher {
    ($reader:ident, $searcher:ident, $body:block) => {{
        let opened = $reader.open_segments().unwrap();
        let segments = opened.as_open_segments();
        let owned = $reader.field_norms_by_field(&["txt".to_string()]).unwrap();
        let norms: Vec<_> = owned.iter().map(Some).collect();
        let $searcher = IndexSearcher::new(&segments, &norms).unwrap();
        $body
    }};
}

#[test]
fn doc_values_stats_equal_lucenes() {
    let tsv = std::fs::read_to_string(data().join("search.tsv")).unwrap();
    let dir = FsDirectory::open(data().join("search_src"));
    let reader = DirectoryReader::open(&dir).unwrap();
    let leaf_refs: Vec<&dyn LeafReader> = reader
        .segment_readers()
        .iter()
        .map(|s| s as &dyn LeafReader)
        .collect();
    let leaves = Leaves::new(&leaf_refs);
    let mut checked = 0;
    for line in tsv.lines().filter(|l| l.starts_with("S\t")) {
        let c: Vec<&str> = line.split('\t').collect();
        let (q, kind_field) = (c[1], c[2]);
        let (kind, field) = kind_field.split_once(':').unwrap();
        let want = c[3..].join("\t");
        let query = query(q);
        let got = with_searcher!(reader, searcher, {
            macro_rules! run {
                ($make:expr) => {{
                    let manager = DocValuesStatsCollectorManager::new(|| $make);
                    let mut collectors = Vec::new();
                    for i in 0..leaf_refs.len() {
                        let mut c = manager.new_collector(leaves);
                        searcher
                            .search_slice_collector(&query, &[i], &mut c)
                            .unwrap();
                        collectors.push(c);
                    }
                    manager.reduce(collectors).unwrap()
                }};
            }
            match kind {
                "long" => {
                    let s: LongDocValuesStats = run!(LongDocValuesStats::new(field));
                    format!(
                        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t-",
                        s.count(),
                        s.missing(),
                        s.min(),
                        s.max(),
                        s.sum(),
                        bits(s.mean()),
                        bits(s.variance()),
                        bits(s.stdev()),
                        ""
                    )
                    .replace("\t\t-", "\t-")
                }
                "double" => {
                    let s: DoubleDocValuesStats = run!(DoubleDocValuesStats::new(field));
                    format!(
                        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t-",
                        s.count(),
                        s.missing(),
                        bits(s.min()),
                        bits(s.max()),
                        bits(s.sum()),
                        bits(s.mean()),
                        bits(s.variance()),
                        bits(s.stdev()),
                    )
                }
                "sortedlong" => {
                    let s: SortedLongDocValuesStats = run!(SortedLongDocValuesStats::new(field));
                    format!(
                        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                        s.count(),
                        s.missing(),
                        s.min(),
                        s.max(),
                        s.sum(),
                        bits(s.mean()),
                        bits(s.variance()),
                        bits(s.stdev()),
                        s.values_count()
                    )
                }
                "sorteddouble" => {
                    let s: SortedDoubleDocValuesStats =
                        run!(SortedDoubleDocValuesStats::new(field));
                    format!(
                        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                        s.count(),
                        s.missing(),
                        bits(s.min()),
                        bits(s.max()),
                        bits(s.sum()),
                        bits(s.mean()),
                        bits(s.variance()),
                        bits(s.stdev()),
                        s.values_count()
                    )
                }
                "sorted" => {
                    let s: SortedDocValuesStats = run!(SortedDocValuesStats::new(field));
                    format!(
                        "{}\t{}\t{}\t{}\t-\t-\t-\t-\t-",
                        s.count(),
                        s.missing(),
                        hex(s.min()),
                        hex(s.max())
                    )
                }
                "sortedset" => {
                    let s: SortedSetDocValuesStats = run!(SortedSetDocValuesStats::new(field));
                    format!(
                        "{}\t{}\t{}\t{}\t-\t-\t-\t-\t-",
                        s.count(),
                        s.missing(),
                        hex(s.min()),
                        hex(s.max())
                    )
                }
                other => panic!("{other}"),
            }
        });
        assert_eq!(got, want, "{q} {kind_field}");
        checked += 1;
    }
    assert_eq!(checked, 16);
}

/// The keys of the `key` numeric doc values, by global document id.
fn key_source(leaves: Vec<Arc<lucene_search::directory_reader::SegmentReader>>) -> KeySource {
    let mut bases = Vec::new();
    let mut base = 0;
    for l in &leaves {
        bases.push(base);
        base += l.max_doc;
    }
    Box::new(move |doc| {
        let i = bases.partition_point(|b| *b <= doc) - 1;
        let Some(mut dv) = leaves[i].numeric_doc_values("key").unwrap() else {
            return 0;
        };
        if dv.advance_exact(doc - bases[i]).unwrap() {
            dv.long_value()
        } else {
            0
        }
    })
}

fn rows(tsv: &str, config: &str) -> Vec<(i32, i32, u32, i64)> {
    tsv.lines()
        .map(|l| l.split('\t').collect::<Vec<_>>())
        .filter(|c| c[0] == "D" && c[1] == config)
        .map(|c| {
            (
                c[2].parse().unwrap(),
                c[3].parse().unwrap(),
                c[4].parse::<i32>().unwrap() as u32,
                c[5].parse().unwrap(),
            )
        })
        .collect()
}

fn check(tsv: &str, config: &str, top: &DiversifiedTopDocs) {
    let want = rows(tsv, config);
    let (total, hits) = want.split_last().map_or((0, &[][..]), |(t, h)| (t.1, h));
    let hits: Vec<(i32, i32, u32, i64)> = hits.to_vec();
    assert_eq!(top.total_hits, total as u64, "{config}: total");
    let got: Vec<(i32, i32, u32, i64)> = top
        .hits
        .iter()
        .enumerate()
        .map(|(r, h)| (r as i32, h.doc, h.score.to_bits(), h.key))
        .collect();
    assert_eq!(got, hits, "{config}");
}

#[test]
fn diversified_top_docs_equal_lucenes() {
    let tsv = std::fs::read_to_string(data().join("search.tsv")).unwrap();
    let dir = FsDirectory::open(data().join("search_src"));
    let reader = DirectoryReader::open(&dir).unwrap();
    let leaves: Vec<_> = reader
        .segment_readers()
        .iter()
        .cloned()
        .map(Arc::new)
        .collect();
    let q = query("a");
    with_searcher!(reader, searcher, {
        for (n, m) in [(10, 2), (5, 1), (50, 3)] {
            let name = format!("{n}_{m}");
            let mut single = DiversifiedTopDocsCollector::new(n, m, key_source(leaves.clone()));
            searcher.search_collector(&q, &mut single).unwrap();
            check(&tsv, &format!("c{name}"), &single.top_docs());

            let manager =
                DiversifiedTopDocsCollectorManager::new(n, m, || key_source(leaves.clone()));
            let mut collectors = Vec::new();
            for i in 0..leaves.len() {
                let mut c = manager.new_collector().unwrap();
                searcher.search_slice_collector(&q, &[i], &mut c).unwrap();
                collectors.push(c);
            }
            check(
                &tsv,
                &format!("m{name}"),
                &manager.reduce(collectors).unwrap(),
            );
        }
    });
}

#[test]
fn memory_accounting_bitset_equals_lucenes() {
    let tsv = std::fs::read_to_string(data().join("search.tsv")).unwrap();
    let dir = FsDirectory::open(data().join("search_src"));
    let reader = DirectoryReader::open(&dir).unwrap();
    let tracker = Arc::new(CollectorMemoryTracker::new("t", i64::MAX));
    let manager = MemoryAccountingBitsetCollectorManager::new(tracker.clone());
    let q = query("b");
    let result = with_searcher!(reader, searcher, {
        let mut collectors = Vec::new();
        for i in 0..reader.segment_readers().len() {
            let mut c = manager.new_collector().unwrap();
            searcher.search_slice_collector(&q, &[i], &mut c).unwrap();
            collectors.push(c);
        }
        manager.reduce(collectors).unwrap()
    });
    let mut want_len = 0usize;
    let mut want_docs = Vec::new();
    for c in tsv
        .lines()
        .map(|l| l.split('\t').collect::<Vec<_>>())
        .filter(|c| c[0] == "B")
    {
        let doc: i64 = c[2].parse().unwrap();
        if doc < 0 {
            want_len = c[3].parse().unwrap();
        } else {
            want_docs.push(doc as usize);
        }
    }
    assert_eq!(result.bit_set.len(), want_len);
    let got: Vec<usize> = (0..result.bit_set.len())
        .filter(|d| result.bit_set.get(*d))
        .collect();
    assert_eq!(got, want_docs);
    assert!(!want_docs.is_empty());
    assert_eq!(result.total_bytes_used, tracker.bytes());
    assert!(result.total_bytes_used > 0);
}

#[test]
fn memory_accounting_bitset_reports_a_refusing_tracker() {
    let tracker = Arc::new(CollectorMemoryTracker::new("t", 1));
    assert!(MemoryAccountingBitsetCollectorManager::new(tracker)
        .new_collector()
        .is_err());
    // room for the empty set only: growing it is refused when reduced
    let tracker = Arc::new(CollectorMemoryTracker::new("t", 40));
    let manager = MemoryAccountingBitsetCollectorManager::new(tracker);
    let mut c = manager.new_collector().unwrap();
    c.collect(5, 0.0);
    assert!(manager.reduce(vec![c]).is_err());
}

/// Lucene 10.6's `testResultBitSetSizedToHighestMatchedDoc*` and
/// `testResultBitSetEmptyOnNoMatches` (branch_10_6, #16452): the union is
/// sized to the highest matched document plus one, however many documents
/// the collectors' slices spanned, and to one empty bit when nothing matched.
#[test]
fn memory_accounting_bitset_is_sized_to_the_highest_match() {
    let tracker = Arc::new(CollectorMemoryTracker::new("t", i64::MAX));
    let manager = MemoryAccountingBitsetCollectorManager::new(tracker.clone());
    // Two slices of an index of 1,000 documents, one match in each.
    let mut low = manager.new_collector().unwrap();
    low.collect(5, 0.0);
    let mut high = manager.new_collector().unwrap();
    high.collect(500, 0.0);
    let empty = manager.new_collector().unwrap();
    let result = manager.reduce(vec![low, high, empty]).unwrap();
    assert_eq!(result.bit_set.len(), 501);
    assert_eq!(result.bit_set.cardinality(), 2);
    assert!(result.bit_set.get(5) && result.bit_set.get(500));
    assert_eq!(result.total_bytes_used, tracker.bytes());

    let manager = MemoryAccountingBitsetCollectorManager::new(tracker);
    let result = manager
        .reduce(vec![manager.new_collector().unwrap()])
        .unwrap();
    assert_eq!(result.bit_set.len(), 1);
    assert_eq!(result.bit_set.next_set_bit(0), None);
    assert!(!result.bit_set.get(0));
}

#[test]
fn human_readable_query_prefixes_its_description() {
    use lucene_search::misc::search::HumanReadableQuery;
    let inner = Clause::Term(TermQuery::new("txt", b"a".to_vec()));
    let q = HumanReadableQuery::new(inner.clone(), "users");
    assert_eq!(q.to_string_field("txt"), "users:a");
    assert_eq!(q.to_string_field("other"), "users:txt:a");
    assert_eq!(q.description(), "users");
    assert_eq!(q.wrapped_query(), &inner);
    assert_eq!(q.rewrite(), inner);
    assert_eq!(q, HumanReadableQuery::new(inner, "users"));
}
