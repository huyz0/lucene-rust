//! Full-text search over one shard shared by many tenants: the Rust side of
//! `scripts/bench-fts-partition.sh`, over the indexes and query list the
//! Java side (`benchmarks/java-runner/src/PartitionBench.java`) writes with
//! stock Lucene.
//!
//! ```text
//! cargo run --release -p lucene-search --example fts_partition -- query <dir> <queries.tsv> <passes>
//! cargo run --release -p lucene-search --example fts_partition -- hits <dir> <queries.tsv>
//! ```
//!
//! `query` prints the cold pass first (mean latency and the bytes read from
//! the device, run it on a dropped page cache), then per tenant class and
//! query kind the hot mean and median. `hits` prints each query's total and
//! top hits (stored id and score bits, or sort value), as the Java side does.
//! A search sorted by time (`newest`) ends a segment early where the index
//! sort's tenant key is pinned, as the plugin tells the native collector.

#![allow(clippy::arithmetic_side_effects)]

use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

use lucene_codecs::stored_fields::FieldValue;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::field_norms::FieldNorms;
use lucene_search::index_searcher::{IndexSearcher, SegmentNorms};
use lucene_search::multi_segment::OpenSegment;
use lucene_search::query::{
    BooleanQuery, Clause, FieldExistsQuery, PhraseQuery, PointsRangeQuery, TermQuery,
};
use lucene_search::top_field::{search_sorted, SortField, SortType};
use lucene_store::MmapDirectory;

const TOP_N: usize = 10;
const THRESHOLD: u64 = 1000;

struct Q {
    id: String,
    class: String,
    kind: String,
    query: BooleanQuery,
    sorted: bool,
}

fn body(t: &str) -> Clause {
    Clause::Term(TermQuery::new("body", t.as_bytes().to_vec()))
}

fn load(path: &str, blocks: bool) -> Vec<Q> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| {
            let f: Vec<&str> = line.split('\t').collect();
            let words: Vec<&str> = f[6].split(' ').filter(|w| !w.is_empty()).collect();
            let kind = f[2].to_string();
            let must = match kind.as_str() {
                "term" | "newest_term" => vec![body(words[0])],
                "and" => vec![body(words[0]), body(words[1])],
                "or" => vec![Clause::Boolean(Box::new(BooleanQuery {
                    should: vec![body(words[0]), body(words[1])],
                    ..BooleanQuery::default()
                }))],
                "phrase" => vec![Clause::Phrase(PhraseQuery::new(
                    "body",
                    words.iter().map(|w| w.to_string()),
                ))],
                "newest" => Vec::new(),
                k => panic!("query kind {k}"),
            };
            let mut filter = vec![Clause::Term(TermQuery::new(
                "tenant",
                f[3].as_bytes().to_vec(),
            ))];
            let (lo, hi): (i64, i64) = (f[4].parse().unwrap(), f[5].parse().unwrap());
            if lo >= 0 {
                filter.push(Clause::PointsRange(PointsRangeQuery::new("ts", lo, hi)));
            }
            if blocks {
                filter.push(Clause::Exists(FieldExistsQuery::new("_primary_term")));
            }
            Q {
                id: f[0].to_string(),
                class: f[1].to_string(),
                sorted: kind.starts_with("newest"),
                kind,
                query: BooleanQuery {
                    must,
                    filter,
                    ..BooleanQuery::default()
                },
            }
        })
        .collect()
}

fn read_bytes() -> u64 {
    std::fs::read_to_string("/proc/self/io")
        .unwrap_or_default()
        .lines()
        .find_map(|l| {
            l.strip_prefix("read_bytes: ")
                .map(|v| v.trim().parse().unwrap_or(0))
        })
        .unwrap_or(0)
}

/// One query's top hits: (doc, score bits or sort value), and the total.
fn run(
    searcher: &IndexSearcher<'_, '_>,
    segments: &[OpenSegment<'_>],
    sorted_segments: &std::cell::RefCell<Vec<OpenSegment<'_>>>,
    reader: &DirectoryReader,
    norms: &[Option<&HashMap<String, FieldNorms<'_>>>],
    q: &Q,
) -> (Vec<(i32, i64)>, String) {
    if q.sorted {
        // The plugin flags a segment whose index sort's keys after the
        // first are the search's sort; the native side checks the query
        // pins the first.
        let mut early = sorted_segments.borrow_mut();
        for s in early.iter_mut() {
            s.index_sort_prefix = pinned(s, &q.query);
        }
        let _ = segments;
        let top = search_sorted(
            &early[..],
            reader.segment_readers(),
            &q.query,
            norms,
            &[SortField::numeric("ts", SortType::Long, true)],
            TOP_N,
            THRESHOLD,
            None,
        )
        .unwrap();
        let total = format!("{}{}", top.total.value, relation(&top.total));
        (
            top.hits.iter().map(|h| (h.doc, h.values[0])).collect(),
            total,
        )
    } else {
        let top = searcher.search(&q.query, TOP_N).unwrap();
        let total = format!("{}{}", top.total_hits.value, relation(&top.total_hits));
        (
            top.score_docs
                .iter()
                .map(|h| (h.doc, i64::from(h.score.to_bits())))
                .collect(),
            total,
        )
    }
}

fn relation(t: &lucene_search::TotalHits) -> &'static str {
    if t.relation == lucene_search::TotalHitsRelation::EqualTo {
        ""
    } else {
        "+"
    }
}

/// Whether the segment's documents come in the `newest` sort's order: its
/// index sort is tenant then time, newest first, and the query pins the
/// tenant.
fn pinned(s: &OpenSegment<'_>, q: &BooleanQuery) -> bool {
    let sorted = s
        .reader
        .and_then(|r| r.index_sort())
        .is_some_and(|k| k.len() >= 2 && k[1].field == "ts" && k[1].reverse);
    sorted && lucene_search::pinned_sort_keys(s, q).unwrap() >= 1
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = &args[2];
    let dir = MmapDirectory::open(path);
    let reader = DirectoryReader::open(&dir).unwrap();
    let blocks = reader.segment_readers()[0].has_blocks();
    let qs = load(&args[3], blocks);
    let mut opened = reader.open_segments().unwrap();
    opened.open_points().unwrap();
    let segments = opened.as_open_segments();
    let owned = reader.field_norms_by_field(&["body".to_string()]).unwrap();
    let norms: Vec<SegmentNorms<'_, '_>> = owned.iter().map(Some).collect();
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();
    let sorted_segments = std::cell::RefCell::new(opened.as_open_segments());
    let go = |q: &Q| run(&searcher, &segments, &sorted_segments, &reader, &norms, q);
    match args[1].as_str() {
        "hits" => {
            let leaves = reader.segment_readers();
            for q in &qs {
                let (hits, total) = go(q);
                let mut line = format!("{}\t{total}", q.id);
                for (doc, v) in hits {
                    let (mut d, mut id) = (doc, -1);
                    for seg in leaves {
                        if d < seg.max_doc {
                            let stored = seg.stored_document(d).unwrap().unwrap();
                            if let Some(FieldValue::Int(i)) =
                                stored.fields.iter().map(|f| &f.value).next()
                            {
                                id = *i;
                            }
                            break;
                        }
                        d -= seg.max_doc;
                    }
                    let v = if q.sorted {
                        v.to_string()
                    } else {
                        format!("{:x}", v as u32)
                    };
                    line.push_str(&format!("\t{id}:{v}"));
                }
                println!("{line}");
            }
        }
        "query" => {
            let passes: usize = args[4].parse().unwrap();
            let before = read_bytes();
            let t = Instant::now();
            for q in &qs {
                std::hint::black_box(go(q));
            }
            println!(
                "cold,all,all,{:.0},{}",
                t.elapsed().as_secs_f64() * 1e6 / qs.len() as f64,
                read_bytes() - before
            );
            for _ in 0..3 {
                for q in &qs {
                    std::hint::black_box(go(q));
                }
            }
            let mut lat: BTreeMap<String, Vec<u64>> = BTreeMap::new();
            for _ in 0..passes {
                for q in &qs {
                    let t = Instant::now();
                    std::hint::black_box(go(q));
                    lat.entry(format!("{},{}", q.class, q.kind))
                        .or_default()
                        .push(t.elapsed().as_micros() as u64);
                }
            }
            for (k, mut v) in lat {
                v.sort_unstable();
                let mean = v.iter().sum::<u64>() as f64 / v.len() as f64;
                println!("hot,{k},{mean:.0},{}", v[v.len() / 2]);
            }
        }
        m => panic!("mode {m}"),
    }
}
