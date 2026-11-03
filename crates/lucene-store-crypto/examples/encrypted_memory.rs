//! Resident memory and query latency of an encrypted index held open by a
//! reader, demand-decrypted (default) or decrypted whole
//! (`LAZY=0`).
//!
//! `cargo run --release -p lucene-store-crypto --example encrypted_memory -- <dir> [docs]`
//! builds the index in `<dir>` when it is empty, then opens it, runs term
//! queries and prints the process's resident set after each phase.
#![allow(clippy::arithmetic_side_effects)]

use std::sync::Arc;
use std::time::Instant;

use lucene_index::document::{Document, Store, StringField, TextField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::index_searcher::{IndexSearcher, SegmentNorms};
use lucene_search::query::{BooleanQuery, Clause, TermQuery};
use lucene_store::{Directory, FsDirectory, MmapDirectory};
use lucene_store_crypto::{EncryptedDirectory, KeyProvider, StaticKeyProvider};

const WORDS: [&str; 16] = [
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa",
    "lambda", "mu", "nu", "xi", "omicron", "pi",
];

fn keys() -> Arc<dyn KeyProvider> {
    Arc::new(StaticKeyProvider::new("bench", &[9; 32]))
}

/// Anonymous (heap and decrypted) and file-backed (page cache) resident
/// memory, in MB.
fn rss_mb() -> String {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let get = |key: &str| {
        status
            .lines()
            .find(|l| l.starts_with(key))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|kb| kb.parse::<f64>().ok())
            .map_or(0.0, |kb| kb / 1024.0)
    };
    format!(
        "anon {:.1} MB, file {:.1} MB",
        get("RssAnon:"),
        get("RssFile:")
    )
}

fn build(dir: &dyn Directory, docs: usize) {
    let v = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut w = IndexWriter::open(dir, Vec::new(), "Lucene104", v).unwrap();
    let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
    for i in 0..docs {
        let mut d = Document::new();
        d.add(StringField::new("id", format!("doc-{i:08}"), Store::Yes));
        let body: Vec<&str> = (0..40)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                WORDS[(x % WORDS.len() as u64) as usize]
            })
            .collect();
        d.add(TextField::new("body", body.join(" "), Store::Yes));
        w.add_fields_document(&d).unwrap();
    }
    w.force_merge(1).unwrap();
    w.commit().unwrap();
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = std::path::PathBuf::from(args.next().expect("index dir"));
    let docs: usize = args.next().map_or(1_000_000, |s| s.parse().unwrap());
    std::fs::create_dir_all(&path).unwrap();
    if std::fs::read_dir(&path).unwrap().next().is_none() {
        let t = Instant::now();
        build(
            &EncryptedDirectory::new(FsDirectory::open(&path), keys()),
            docs,
        );
        println!("built {docs} docs in {:.1}s", t.elapsed().as_secs_f64());
    }
    let size: u64 = std::fs::read_dir(&path)
        .unwrap()
        .map(|e| e.unwrap().metadata().unwrap().len())
        .sum();
    let lazy = std::env::var("LAZY").map_or(true, |v| v != "0");
    println!("index {:.1} MB, lazy={}", size as f64 / 1e6, lazy);
    println!("rss start {}", rss_mb());
    let dir = EncryptedDirectory::new(MmapDirectory::open(&path), keys()).with_lazy_decrypt(lazy);
    let t = Instant::now();
    let reader = DirectoryReader::open(&dir).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let owned = reader.field_norms_by_field(&["body".to_string()]).unwrap();
    let norms: Vec<SegmentNorms<'_, '_>> = owned.iter().map(Some).collect();
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();
    println!(
        "open {:.0} ms, rss {}",
        t.elapsed().as_secs_f64() * 1e3,
        rss_mb()
    );
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
        (t.elapsed().as_secs_f64() * 1e3, hits)
    };
    let (first, hits) = run();
    println!(
        "first pass: 16 term queries {first:.1} ms ({hits} hits), rss {}",
        rss_mb()
    );
    let best = (0..30).map(|_| run().0).fold(f64::MAX, f64::min);
    println!("warm: best of 30 passes {best:.2} ms, rss {}", rss_mb());
}
