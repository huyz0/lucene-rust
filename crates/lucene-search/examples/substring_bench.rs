//! Substring search over a log corpus: the Rust side of
//! `docs/benchmarks/substring-field.md`. One line per document, `msg` a
//! BINARY doc values field in the `RustSubstring` format (or plain BINARY),
//! `line` its line number.
//!
//! ```text
//! cargo run --profile dist -p lucene-search --example substring_bench -- build <corpus> <dir> <block_shift|plain>
//! cargo run --profile dist -p lucene-search --example substring_bench -- query <dir> <queries.tsv> [secs]
//! ```
//!
//! `build` flushes without merging, then force-merges to one segment, and
//! prints the seconds of each and the index size. `query` prints each
//! query's median latency, hit count and the sum of `mix(line)` over its
//! hits, the same hash the Java side prints for stock Lucene's wildcard
//! field.

#![allow(clippy::arithmetic_side_effects)]

use std::io::BufRead;
use std::time::Instant;

use lucene_codecs::doc_values::Lucene90DocValuesFormat;
use lucene_codecs::substring::Candidates;
use lucene_index::document::{BinaryDocValuesField, Document, NumericDocValuesField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::collector::ScoringCollector;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::extended_query::{SubstringKind, SubstringQuery};
use lucene_search::index_searcher::{IndexSearcher, SegmentNorms};
use lucene_search::query::{BooleanQuery, Clause};
use lucene_search::reader::LeafReader;
use lucene_store::{FsDirectory, MmapDirectory};

const VERSION: LuceneVersion = LuceneVersion {
    major: 10,
    minor: 5,
    bugfix: 0,
};

fn dir_size(dir: &str) -> (u64, u64) {
    let mut total = 0;
    let mut dv = 0;
    for e in std::fs::read_dir(dir).unwrap() {
        let e = e.unwrap();
        let n = e.file_name().to_string_lossy().into_owned();
        let len = e.metadata().unwrap().len();
        total += len;
        if n.ends_with(".dvd") || n.ends_with(".dvm") {
            dv += len;
        }
    }
    (total, dv)
}

fn writer<'d>(fs: &'d FsDirectory, shift: Option<u8>) -> IndexWriter<'d> {
    let mut w = IndexWriter::open(fs, Vec::new(), "Lucene104", VERSION).unwrap();
    w.set_ram_buffer_size_mb(256.0).unwrap();
    if let Some(s) = shift {
        w.set_doc_values_format_for_field("msg", Lucene90DocValuesFormat::rust_substring(s));
    }
    w
}

fn build(corpus: &str, dir: &str, shift: &str) {
    let shift: Option<u8> = (shift != "plain").then(|| shift.parse().unwrap());
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).unwrap();
    let fs = FsDirectory::open(std::path::Path::new(dir));
    let file = std::io::BufReader::with_capacity(1 << 20, std::fs::File::open(corpus).unwrap());
    let t = Instant::now();
    let mut n = 0i64;
    {
        let mut w = writer(&fs, shift);
        for line in file.split(b'\n') {
            let line = line.unwrap();
            let mut doc = Document::new();
            doc.add(BinaryDocValuesField::new("msg", line));
            doc.add(NumericDocValuesField::new("line", n));
            w.add_fields_document(&doc).unwrap();
            n += 1;
        }
        w.commit().unwrap();
    }
    let index_s = t.elapsed().as_secs_f64();
    let segments = std::fs::read_dir(dir)
        .unwrap()
        .filter(|e| {
            e.as_ref()
                .unwrap()
                .file_name()
                .to_string_lossy()
                .ends_with(".si")
        })
        .count();
    let (flushed, _) = dir_size(dir);
    let t = Instant::now();
    {
        let mut w = writer(&fs, shift);
        w.force_merge(1).unwrap();
        w.commit().unwrap();
    }
    let merge_s = t.elapsed().as_secs_f64();
    let (total, dv) = dir_size(dir);
    println!(
        "docs={n} index_s={index_s:.1} docs_per_s={:.0} segments={segments} flushed_mib={:.1} merge_s={merge_s:.1} total_mib={:.1} dv_mib={:.1}",
        n as f64 / index_s,
        flushed as f64 / 1048576.0,
        total as f64 / 1048576.0,
        dv as f64 / 1048576.0
    );
}

struct Q {
    id: String,
    query: SubstringQuery,
}

fn queries(path: &str) -> Vec<Q> {
    let mut out = Vec::new();
    for l in std::fs::read_to_string(path).unwrap().lines() {
        if l.starts_with('#') || l.trim().is_empty() {
            continue;
        }
        let c: Vec<&str> = l.split('\t').collect();
        let kind = match c[1] {
            "contains" => SubstringKind::Contains,
            "wildcard" => SubstringKind::Wildcard,
            "regexp" => SubstringKind::Regexp,
            "exact" => SubstringKind::Exact,
            k => panic!("kind {k}"),
        };
        out.push(Q {
            id: c[0].to_string(),
            query: SubstringQuery::new("msg", kind, c[3], c[2] == "1").unwrap(),
        });
    }
    out
}

fn mix(x: i64) -> i64 {
    let x = x.wrapping_mul(0x9E37_79B9_7F4A_7C15_u64 as i64);
    x ^ ((x as u64) >> 31) as i64
}

struct Hash<'l> {
    lines: &'l [i64],
    n: u64,
    h: i64,
}

impl ScoringCollector for Hash<'_> {
    fn collect(&mut self, doc: i32, _score: f32) {
        self.n += 1;
        self.h = self.h.wrapping_add(mix(self.lines[doc as usize]));
    }
}

fn query(dir: &str, path: &str, secs: f64) {
    let reader = DirectoryReader::open(&MmapDirectory::open(std::path::Path::new(dir))).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let norms: Vec<SegmentNorms<'_, '_>> = segments.iter().map(|_| None).collect();
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();
    let mut lines = vec![0i64; searcher.max_doc() as usize];
    for s in &segments {
        let r = s.reader.expect("a segment reader");
        let mut dv = r.numeric_doc_values("line").unwrap().unwrap();
        while dv.next_doc().unwrap() != i32::MAX {
            lines[(s.doc_base + dv.doc_id()) as usize] = dv.long_value();
        }
    }
    println!("id\trust_ms\trust_hits\trust_hash");
    for q in queries(path) {
        let bq = BooleanQuery {
            must: vec![Clause::from(q.query.clone())],
            ..BooleanQuery::default()
        };
        let run = || {
            let mut c = Hash {
                lines: &lines,
                n: 0,
                h: 0,
            };
            searcher.search_collector(&bq, &mut c).unwrap();
            (c.n, c.h)
        };
        let (n, h) = run();
        let warm = Instant::now();
        while warm.elapsed().as_secs_f64() < secs * 0.3 {
            run();
        }
        let mut ts = Vec::new();
        let t = Instant::now();
        while (t.elapsed().as_secs_f64() < secs || ts.len() < 3) && ts.len() < 100_000 {
            let s = Instant::now();
            run();
            ts.push(s.elapsed().as_secs_f64() * 1e3);
        }
        ts.sort_by(f64::total_cmp);
        println!("{}\t{:.3}\t{n}\t{h}", q.id, ts[ts.len() / 2]);
    }
}

fn explain(dir: &str, path: &str) {
    let reader = DirectoryReader::open(&MmapDirectory::open(std::path::Path::new(dir))).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    println!("id\tgrams\tcandidate_blocks\tblocks\tplan_ms\tread_ms\tread_test_ms");
    for q in queries(path) {
        let (mut cands, mut blocks) = (0i64, 0i64);
        let (mut plan_ms, mut read_ms, mut all_ms) = (0f64, 0f64, 0f64);
        for s in &segments {
            let r = s.reader.expect("a segment reader");
            let Some(index) = r.substring_index("msg").unwrap() else {
                continue;
            };
            blocks += index.num_blocks();
            let t = Instant::now();
            let list: Vec<i64> = match index.candidate_blocks(q.query.matcher().grams()).unwrap() {
                Candidates::All => (0..index.num_blocks()).collect(),
                Candidates::Blocks(b) => b.into_iter().map(i64::from).collect(),
            };
            plan_ms += t.elapsed().as_secs_f64() * 1e3;
            cands += list.len() as i64;
            let mut block = lucene_codecs::substring::Block::new();
            let t = Instant::now();
            for &b in &list {
                index.read_block(b, &mut block).unwrap();
            }
            read_ms += t.elapsed().as_secs_f64() * 1e3;
            let (mut hits, mut scratch) = (Vec::new(), Vec::new());
            let t = Instant::now();
            for &b in &list {
                index.read_block(b, &mut block).unwrap();
                q.query.matcher().collect(&block, &mut hits, &mut scratch);
            }
            all_ms += t.elapsed().as_secs_f64() * 1e3;
        }
        println!(
            "{}\t{}\t{cands}\t{blocks}\t{plan_ms:.2}\t{read_ms:.2}\t{all_ms:.2}",
            q.id,
            q.query.matcher().grams().len()
        );
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    match a[1].as_str() {
        "build" => build(&a[2], &a[3], &a[4]),
        "query" => query(&a[2], &a[3], a.get(4).map_or(2.0, |s| s.parse().unwrap())),
        "explain" => explain(&a[2], &a[3]),
        c => panic!("command {c}"),
    }
}
