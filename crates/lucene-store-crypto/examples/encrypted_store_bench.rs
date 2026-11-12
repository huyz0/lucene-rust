//! Open time, resident memory, query latency and merge time of one index
//! stored plain or encrypted, read through each store mode.
//!
//! `cargo run --release -p lucene-store-crypto --example encrypted_store_bench -- <root> <mode> [docs]`
//!
//! builds `<root>/plain` and `<root>/enc` (the same documents, in eight
//! segments) on first use, then in a fresh process per run:
//!
//! - opens a reader over the mode's directory and prints the open time and
//!   resident memory,
//! - runs a pass of term queries (the first, cold when the caller dropped
//!   the kernel's caches before the run) and then the best of 20 passes,
//! - force-merges a copy of the index to one segment through the mode's
//!   directory.
//!
//! Modes: `plain-mmap`, `plain-paged` (the demand pager), `enc-whole`
//! (encrypted, decrypted whole at open), `enc-paged` (encrypted, decrypted
//! page by page into the demand pager). `BUDGET_MB` sets the pager's
//! budget.
#![allow(clippy::arithmetic_side_effects)]

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use lucene_index::document::{Document, Store, StringField, TextField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::index_searcher::{IndexSearcher, SegmentNorms};
use lucene_search::query::{BooleanQuery, Clause, TermQuery};
use lucene_store::{Directory, FsDirectory, MmapDirectory, PagedFsDirectory};
use lucene_store_crypto::{EncryptedDirectory, KeyProvider, StaticKeyProvider};

const WORDS: [&str; 16] = [
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa",
    "lambda", "mu", "nu", "xi", "omicron", "pi",
];
const TAGS: u64 = 2000;
const SEGMENTS: usize = 8;

fn keys() -> Arc<dyn KeyProvider> {
    Arc::new(StaticKeyProvider::new("bench", &[9; 32]))
}

fn version() -> LuceneVersion {
    LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    }
}

/// Anonymous and file-backed resident memory, in MB.
fn rss() -> (f64, f64) {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let get = |key: &str| {
        status
            .lines()
            .find(|l| l.starts_with(key))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|kb| kb.parse::<f64>().ok())
            .map_or(0.0, |kb| kb / 1024.0)
    };
    (get("RssAnon:"), get("RssFile:"))
}

fn pager_mb() -> f64 {
    lucene_store::paged_input::Pager::global().map_or(0.0, |p| p.resident_bytes() as f64 / 1e6)
}

fn pager_faults() -> u64 {
    lucene_store::paged_input::Pager::global().map_or(0, |p| p.stats().faults)
}

fn report(phase: &str, ms: f64) {
    let (anon, file) = rss();
    println!(
        "{phase:<10} {ms:>9.1} ms  anon {anon:>7.1} MB  file {file:>7.1} MB  pager {:>7.1} MB  faults {}",
        pager_mb(),
        pager_faults()
    );
}

fn build(dir: &dyn Directory, docs: usize) {
    let mut w = IndexWriter::open(dir, Vec::new(), "Lucene104", version()).unwrap();
    let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    let per_segment = docs.div_ceil(SEGMENTS);
    for i in 0..docs {
        let mut d = Document::new();
        d.add(StringField::new("id", format!("doc-{i:08}"), Store::Yes));
        d.add(StringField::new(
            "tag",
            format!("t{}", next() % TAGS),
            Store::No,
        ));
        let body: Vec<&str> = (0..40)
            .map(|_| WORDS[(next() % WORDS.len() as u64) as usize])
            .collect();
        d.add(TextField::new("body", body.join(" "), Store::Yes));
        w.add_fields_document(&d).unwrap();
        if (i + 1) % per_segment == 0 {
            w.flush().unwrap();
        }
    }
    w.commit().unwrap();
}

fn dir_size(path: &Path) -> u64 {
    std::fs::read_dir(path)
        .unwrap()
        .map(|e| e.unwrap().metadata().unwrap().len())
        .sum()
}

fn copy_dir(from: &Path, to: &Path) {
    let _ = std::fs::remove_dir_all(to);
    std::fs::create_dir_all(to).unwrap();
    for e in std::fs::read_dir(from).unwrap() {
        let e = e.unwrap();
        std::fs::copy(e.path(), to.join(e.file_name())).unwrap();
    }
}

fn open_mode(mode: &str, path: &Path) -> Box<dyn Directory> {
    match mode {
        "plain-mmap" => Box::new(MmapDirectory::open(path)),
        "plain-paged" => Box::new(PagedFsDirectory::open(path)),
        "enc-whole" => Box::new(EncryptedDirectory::new(MmapDirectory::open(path), keys())),
        "enc-paged" => Box::new(EncryptedDirectory::new(
            PagedFsDirectory::open(path),
            keys(),
        )),
        other => panic!("unknown mode {other}"),
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let root = PathBuf::from(args.next().expect("root dir"));
    let mode = args.next().expect("mode");
    let docs: usize = args.next().map_or(2_000_000, |s| s.parse().unwrap());
    if let Ok(mb) = std::env::var("BUDGET_MB") {
        let pager = lucene_store::paged_input::Pager::global().expect("pager");
        pager.set_budget(mb.parse::<u64>().unwrap() << 20);
    }
    let plain = root.join("plain");
    let enc = root.join("enc");
    for (path, encrypted) in [(&plain, false), (&enc, true)] {
        std::fs::create_dir_all(path).unwrap();
        if std::fs::read_dir(path).unwrap().next().is_none() {
            let t = Instant::now();
            if encrypted {
                build(
                    &EncryptedDirectory::new(FsDirectory::open(path), keys()),
                    docs,
                );
            } else {
                build(&FsDirectory::open(path), docs);
            }
            println!(
                "built {} ({docs} docs) in {:.1}s",
                path.display(),
                t.elapsed().as_secs_f64()
            );
        }
    }
    let path = if mode.starts_with("enc") {
        &enc
    } else {
        &plain
    };
    println!("mode {mode}, index {:.1} MB", dir_size(path) as f64 / 1e6);
    report("start", 0.0);
    {
        let dir = open_mode(&mode, path);
        let t = Instant::now();
        let reader = DirectoryReader::open(&*dir).unwrap();
        let opened = reader.open_segments().unwrap();
        let segments = opened.as_open_segments();
        let owned = reader.field_norms_by_field(&["body".to_string()]).unwrap();
        let norms: Vec<SegmentNorms<'_, '_>> = owned.iter().map(Some).collect();
        let searcher = IndexSearcher::new(&segments, &norms).unwrap();
        report("open", t.elapsed().as_secs_f64() * 1e3);
        let run = || {
            let t = Instant::now();
            let mut hits = 0u64;
            for w in WORDS {
                let q = BooleanQuery {
                    must: vec![Clause::Term(TermQuery::new("body", w.as_bytes().to_vec()))],
                    ..Default::default()
                };
                hits += searcher.search(&q, 10).unwrap().total_hits.value;
            }
            for t in (0..TAGS).step_by(20) {
                let q = BooleanQuery {
                    must: vec![Clause::Term(TermQuery::new(
                        "tag",
                        format!("t{t}").into_bytes(),
                    ))],
                    ..Default::default()
                };
                hits += searcher.search(&q, 10).unwrap().total_hits.value;
            }
            (t.elapsed().as_secs_f64() * 1e3, hits)
        };
        let (first, hits) = run();
        report("query1", first);
        let mut passes: Vec<f64> = (0..20).map(|_| run().0).collect();
        passes.sort_by(f64::total_cmp);
        report("query-p50", passes[passes.len() / 2]);
        report("query-min", passes[0]);
        println!("hits {hits}");
    }
    let scratch = root.join(format!("merge-{mode}"));
    copy_dir(path, &scratch);
    // The copy sits in the kernel's cache: drop it so the merge reads disk.
    let _ = std::fs::write("/proc/sys/vm/drop_caches", "1");
    {
        let dir = open_mode(&mode, &scratch);
        let t = Instant::now();
        let mut w = IndexWriter::open(&*dir, Vec::new(), "Lucene104", version()).unwrap();
        w.force_merge(1).unwrap();
        w.commit().unwrap();
        report("merge", t.elapsed().as_secs_f64() * 1e3);
    }
    let _ = std::fs::remove_dir_all(&scratch);
}
