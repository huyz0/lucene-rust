//! Fetch-phase cost of the stored-fields files (`.fdt`/`.fdx`) under each
//! mmap read advice (`lucene_store::read_advice`), and a merge running
//! beside that fetch load.
//!
//! ```text
//! cargo run --release -p lucene-search --example stored_fetch -- build <dir> <docs> <doc_bytes> [speed|compression|zstd] [segments]
//! cargo run --release -p lucene-search --example stored_fetch -- fetch <dir> [--n 10] [--queries 2000] [--threads 4] [--seconds 60]
//! cargo run --release -p lucene-search --example stored_fetch -- merge <dir> [--threads 4]
//! ```
//!
//! `build` writes `docs` documents, each an `_id` and a `_source` of about
//! `doc_bytes` of JSON-like text, with the given stored-fields mode
//! (`speed` is the default codec, `compression` is `best_compression`), then
//! merges down to `segments` segments (1 by default).
//!
//! `fetch` runs `queries` fetches; each reads the stored fields of `n`
//! documents drawn uniformly over the index, in doc-id order, after the
//! reader's prefetch, as a fetch phase does. It prints latency percentiles,
//! throughput and the bytes the process read from the device.
//!
//! `merge` force-merges the index to one segment while `threads` threads
//! run the fetch load against a reader opened before it, and prints the
//! merge time and the fetch latency during it.
//!
//! `NO_PREFETCH=1` skips the prefetch, as a get by id does. `ID_ONLY=1` reads
//! only `_id` of each document, as a `_source: false` fetch does.
//!
//! The read advice comes from `READ_ADVICE` (a `lucene_store::read_advice`
//! spec; unset keeps the defaults). Drop the page cache before a cold run
//! and run inside a memory cgroup (and a blkio throttle to emulate network
//! block storage) to see the starved case.

#![allow(clippy::arithmetic_side_effects, clippy::unwrap_used)]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use lucene_index::document::{Document, Field, Store, StoredValue, StringField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::{DirectoryReader, SegmentReader};
use lucene_store::{Directory, FsDirectory, MmapDirectory};

const VERSION: LuceneVersion = LuceneVersion {
    major: 10,
    minor: 5,
    bugfix: 0,
};

fn next(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn unit(state: &mut u64) -> f64 {
    (next(state) >> 11) as f64 / (1u64 << 53) as f64
}

/// A Zipf-ish word of a 50k-word vocabulary.
fn word(state: &mut u64) -> String {
    let rank = (50_000f64.powf(unit(state))) as u64;
    format!("w{rank}")
}

/// About `bytes` of a log-like JSON source.
fn source(state: &mut u64, i: usize, bytes: usize) -> Vec<u8> {
    let mut s = format!(
        "{{\"@timestamp\":{},\"host\":\"h{}\",\"status\":{},\"msg\":\"",
        1_730_000_000_000u64 + i as u64 * 37,
        next(state) % 500,
        [200, 200, 200, 404, 500][(next(state) % 5) as usize]
    );
    while s.len() + 2 < bytes {
        s.push_str(&word(state));
        s.push(' ');
    }
    s.push_str("\"}");
    s.into_bytes()
}

fn build(args: &[String]) {
    let path = &args[2];
    let docs: usize = args[3].parse().unwrap();
    let doc_bytes: usize = args[4].parse().unwrap();
    let mode = args.get(5).map_or("speed", String::as_str);
    let segments: usize = args.get(6).map_or(1, |s| s.parse().unwrap());
    let dir = FsDirectory::open(path);
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
    match mode {
        "speed" => {}
        "compression" => w
            .set_codec(Some(Arc::new(
                lucene_codecs::codec::BestCompressionLucene104Codec,
            )))
            .unwrap(),
        "zstd" => w
            .set_codec(Some(Arc::new(
                lucene_codecs::codec::ZstdLucene104Codec::new(
                    lucene_codecs::stored_fields::ZstdMode::zstd(3),
                )
                .unwrap(),
            )))
            .unwrap(),
        m => panic!("unknown mode {m}"),
    }
    w.set_ram_buffer_size_mb(256.0).unwrap();
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let t = Instant::now();
    for i in 0..docs {
        let mut d = Document::new();
        d.add(StringField::new("_id", format!("{i:010}"), Store::Yes));
        d.add(Field::stored(
            "_source",
            StoredValue::Binary(source(&mut state, i, doc_bytes)),
        ));
        w.add_fields_document(&d).unwrap();
    }
    w.commit().unwrap();
    w.force_merge(segments).unwrap();
    w.commit().unwrap();
    let mut total = 0u64;
    for name in dir.list_all().unwrap() {
        let len = dir.file_length(&name).unwrap();
        total += len;
        if len > 1 << 20 {
            eprintln!("  {name:<28} {:>10.1} MiB", len as f64 / (1 << 20) as f64);
        }
    }
    eprintln!(
        "indexed {docs} docs ({mode}, {segments} segments) in {:.1?}, {:.0} MiB",
        t.elapsed(),
        total as f64 / (1 << 20) as f64
    );
}

fn read_bytes() -> u64 {
    std::fs::read_to_string("/proc/self/io")
        .unwrap_or_default()
        .lines()
        .find_map(|l| l.strip_prefix("read_bytes: ")?.trim().parse().ok())
        .unwrap_or(0)
}

fn opt<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}

fn set_advice() {
    if let Ok(spec) = std::env::var("READ_ADVICE") {
        lucene_store::read_advice::set(&spec).unwrap();
    }
}

/// One fetch of `n` random documents, in doc-id order, after the prefetch.
fn fetch_one(leaves: &[SegmentReader], max_doc: i32, n: usize, state: &mut u64) -> usize {
    let prefetch = std::env::var_os("NO_PREFETCH").is_none();
    let id_only = std::env::var_os("ID_ONLY").is_some();
    let mut docs: Vec<i32> = (0..n)
        .map(|_| (next(state) % max_doc as u64) as i32)
        .collect();
    docs.sort_unstable();
    docs.dedup();
    let mut base = 0;
    let mut per_seg: Vec<Vec<i32>> = Vec::with_capacity(leaves.len());
    for seg in leaves {
        per_seg.push(
            docs.iter()
                .filter(|&&d| d >= base && d - base < seg.max_doc)
                .map(|&d| d - base)
                .collect(),
        );
        base += seg.max_doc;
    }
    let mut bytes = 0;
    for (seg, docs) in leaves.iter().zip(&per_seg) {
        let id = seg
            .field_infos()
            .field_by_name("_id")
            .map_or(0, |f| f.number);
        if prefetch {
            seg.prefetch_stored_documents(docs).unwrap();
        }
        for &d in docs {
            if id_only {
                let mut v = lucene_codecs::stored_fields::DocumentVisitor::for_fields(&[id]);
                assert!(seg.visit_stored_document(d, &mut v).unwrap());
                bytes += std::hint::black_box(v.document().fields.len());
                continue;
            }
            let doc = seg.stored_document(d).unwrap().unwrap();
            bytes += std::hint::black_box(doc.fields.len());
        }
    }
    bytes
}

fn pct(sorted: &[Duration], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let i = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[i].as_secs_f64() * 1e3
}

/// Runs fetches on `threads` threads until `queries` are done, `deadline`
/// passes or `stop` is set; returns every fetch's latency.
fn load(
    leaves: &[SegmentReader],
    n: usize,
    queries: usize,
    threads: usize,
    deadline: Instant,
    stop: &AtomicBool,
) -> Vec<Duration> {
    if threads == 0 {
        return Vec::new();
    }
    let max_doc: i32 = leaves.iter().map(|s| s.max_doc).sum();
    let per = queries.div_ceil(threads);
    let mut lat = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads)
            .map(|t| {
                s.spawn(move || {
                    let mut state = 0x2545_f491_4f6c_dd1du64 ^ ((t as u64 + 1) * 0x9e37);
                    let mut lat = Vec::with_capacity(per.min(1 << 20));
                    for _ in 0..per {
                        if Instant::now() > deadline || stop.load(Ordering::Relaxed) {
                            break;
                        }
                        let t = Instant::now();
                        fetch_one(leaves, max_doc, n, &mut state);
                        lat.push(t.elapsed());
                    }
                    lat
                })
            })
            .collect();
        hs.into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect::<Vec<_>>()
    });
    lat.sort_unstable();
    lat
}

fn report(label: &str, lat: &[Duration], wall: Duration, read: u64) {
    let mean = lat.iter().sum::<Duration>().as_secs_f64() * 1e3 / lat.len().max(1) as f64;
    println!(
        "{label} fetches={} p50_ms={:.3} p90_ms={:.3} p99_ms={:.3} mean_ms={:.3} fetches_per_s={:.0} read_mib={:.1} read_kib_per_fetch={:.1}",
        lat.len(),
        pct(lat, 0.5),
        pct(lat, 0.9),
        pct(lat, 0.99),
        mean,
        lat.len() as f64 / wall.as_secs_f64(),
        read as f64 / (1 << 20) as f64,
        read as f64 / 1024.0 / lat.len().max(1) as f64,
    );
}

fn fetch(args: &[String]) {
    let path = &args[2];
    let n: usize = opt(args, "--n").map_or(10, |v| v.parse().unwrap());
    let queries: usize = opt(args, "--queries").map_or(2000, |v| v.parse().unwrap());
    let threads: usize = opt(args, "--threads").map_or(4, |v| v.parse().unwrap());
    let seconds: u64 = opt(args, "--seconds").map_or(60, |v| v.parse().unwrap());
    set_advice();
    let dir = MmapDirectory::open(path);
    let r0 = read_bytes();
    let t = Instant::now();
    let reader = DirectoryReader::open(&dir).unwrap();
    let leaves = reader.segment_readers();
    let open = t.elapsed();
    let stop = AtomicBool::new(false);
    let wall = Instant::now();
    let lat = load(
        leaves,
        n,
        queries,
        threads,
        wall + Duration::from_secs(seconds),
        &stop,
    );
    let wall = wall.elapsed();
    println!(
        "advice={} open_ms={:.1} n={n} threads={threads}",
        std::env::var("READ_ADVICE").unwrap_or_else(|_| "default".into()),
        open.as_secs_f64() * 1e3
    );
    report("fetch", &lat, wall, read_bytes() - r0);
}

fn merge(args: &[String]) {
    let path = &args[2];
    let threads: usize = opt(args, "--threads").map_or(4, |v| v.parse().unwrap());
    let n: usize = opt(args, "--n").map_or(10, |v| v.parse().unwrap());
    set_advice();
    let dir = MmapDirectory::open(path);
    let reader = DirectoryReader::open(&dir).unwrap();
    let leaves = reader.segment_readers();
    let stop = AtomicBool::new(false);
    let r0 = read_bytes();
    let wall = Instant::now();
    let (lat, merge_time) = std::thread::scope(|s| {
        let h = s.spawn(|| {
            load(
                leaves,
                n,
                usize::MAX / 2,
                threads,
                Instant::now() + Duration::from_secs(3600),
                &stop,
            )
        });
        let t = Instant::now();
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
        w.force_merge(1).unwrap();
        w.commit().unwrap();
        let merge_time = t.elapsed();
        stop.store(true, Ordering::Relaxed);
        (h.join().unwrap(), merge_time)
    });
    let wall = wall.elapsed();
    println!(
        "advice={} merge_s={:.2}",
        std::env::var("READ_ADVICE").unwrap_or_else(|_| "default".into()),
        merge_time.as_secs_f64()
    );
    report("fetch_during_merge", &lat, wall, read_bytes() - r0);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("build") => build(&args),
        Some("fetch") => fetch(&args),
        Some("merge") => merge(&args),
        _ => eprintln!("usage: stored_fetch build|fetch|merge ..."),
    }
}
