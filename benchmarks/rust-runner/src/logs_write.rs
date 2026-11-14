//! Write-path cost on a log-shaped document: the Rust half of
//! `scripts/bench-logs-write.sh`, over the corpus this program writes and
//! the Java side (`benchmarks/java-runner/src/LogsWriteBench.java`) reads.
//! It allocates with mimalloc, as the plugin library does.
//!
//! ```text
//! logs_write corpus <out.tsv> <n> <seed>
//! logs_write index <corpus.tsv> <dir> <threads> <refresh ms>
//! logs_write forcemerge <dir> <segments>
//! ```
//!
//! `ADAPTIVE=on` writes adaptive doc values, `ADAPTIVE=auto` lets the writer
//! choose from the data; unset is stock doc values.
//!
//! A document is what an OpenSearch log index holds: `_id` (stored and
//! indexed), `_source` (stored JSON), six keyword fields (indexed without
//! norms, sorted-set doc values), four longs (a point and a sorted-numeric
//! doc value each), `_seq_no` (point and numeric doc value), `_version` and
//! `_primary_term` (numeric doc values) and a 20-word `message` with
//! positions and norms. Each indexing thread builds its documents as it
//! adds them, as the plugin decodes them, so the thread that allocates a
//! document is the one that frees it. `index` writes as
//! `partition_cost index` does: a concurrent writer with one buffer per CPU,
//! a 256 MB buffer, `TieredMergePolicy` and `ConcurrentMergeScheduler`
//! defaults, a near-real-time reader refreshed every `refresh ms`, then a
//! commit and the merges it scheduled.

#![allow(clippy::arithmetic_side_effects)]

use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use lucene_codecs::field_infos::{DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions};
use lucene_codecs::stored_fields::{FieldValue, StoredField};
use lucene_index::concurrent_writer::ConcurrentIndexWriter;
use lucene_index::index_writer::{ExplicitDocument, IndexWriter, InvertedField, InvertedTerms};
use lucene_index::merge_policy::MergePolicyConfig;
use lucene_index::merge_scheduler::ConcurrentMergeScheduler;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_store::directory::Directory;
use lucene_store::MmapDirectory;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

const VERSION: LuceneVersion = LuceneVersion {
    major: 10,
    minor: 5,
    bugfix: 0,
};

const KEYWORDS: [(&str, u64); 6] = [
    ("host", 1000),
    ("status", 12),
    ("method", 6),
    ("service", 200),
    ("region", 24),
    ("agent", 500),
];
const LONGS: [&str; 4] = ["ts", "bytes", "latency", "count"];

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let mut x = self.0;
        x ^= x >> 33;
        x = x.wrapping_mul(0xff51afd7ed558ccd);
        x ^ (x >> 33)
    }
    /// Roughly Zipf over `0..n`: small values far more often.
    fn skewed(&mut self, n: u64) -> u64 {
        let u = (self.next() >> 11) as f64 / (1u64 << 53) as f64;
        ((n as f64).powf(u) - 1.0) as u64 % n
    }
}

fn corpus(out: &str, n: usize, seed: u64) {
    let mut r = Rng(seed);
    let mut w = std::io::BufWriter::new(std::fs::File::create(out).unwrap());
    let mut ts = 1_700_000_000_000i64;
    for i in 0..n {
        ts += (r.next() % 20) as i64;
        let mut cols: Vec<String> = vec![format!("{i:016x}")];
        for (name, card) in KEYWORDS {
            cols.push(format!("{name}{}", r.skewed(card)));
        }
        cols.push(ts.to_string());
        cols.push((r.skewed(1 << 20)).to_string());
        cols.push((r.next() % 5000).to_string());
        cols.push((r.next() % 100).to_string());
        let words: Vec<String> = (0..20).map(|_| format!("w{}", r.skewed(50_000))).collect();
        cols.push(words.join(" "));
        writeln!(w, "{}", cols.join("\t")).unwrap();
    }
}

struct Row {
    cols: Vec<String>,
}

fn read_corpus(path: &str) -> Vec<Row> {
    let f = std::io::BufReader::new(std::fs::File::open(path).unwrap());
    f.lines()
        .map(|l| Row {
            cols: l.unwrap().split('\t').map(str::to_string).collect(),
        })
        .collect()
}

/// The `_source` both engines store: the row as a JSON object.
fn source(r: &Row) -> Vec<u8> {
    let c = &r.cols;
    let mut s = String::with_capacity(400);
    s.push('{');
    for (i, (name, _)) in KEYWORDS.iter().enumerate() {
        s.push_str(&format!("\"{name}\":\"{}\",", c[1 + i]));
    }
    for (i, name) in LONGS.iter().enumerate() {
        s.push_str(&format!("\"{name}\":{},", c[7 + i]));
    }
    s.push_str(&format!("\"message\":\"{}\"}}", c[11]));
    s.into_bytes()
}

fn kw(name: &str, n: i32) -> FieldInfo {
    FieldInfo::new(name, n)
        .with_index_options(IndexOptions::Docs)
        .with_omit_norms(true)
        .with_doc_values(DocValuesType::SortedSet, DocValuesSkipIndexType::None, -1)
}

fn long(name: &str, n: i32, dv: DocValuesType) -> FieldInfo {
    FieldInfo::new(name, n)
        .with_points(1, 1, 8)
        .with_doc_values(dv, DocValuesSkipIndexType::None, -1)
}

/// Field numbers: `_id` 0, `_source` 1, keywords 2..8, longs 8..12,
/// `_seq_no` 12, `_version` 13, `_primary_term` 14, `message` 15.
fn writer(dir: &dyn Directory) -> IndexWriter<'_> {
    let mut w = IndexWriter::open(dir, Vec::new(), "Lucene104", VERSION).unwrap();
    w.enable_explicit_documents().unwrap();
    w.register_field(
        FieldInfo::new("_id", 0)
            .with_index_options(IndexOptions::Docs)
            .with_omit_norms(true),
    )
    .unwrap();
    w.register_field(FieldInfo::new("_source", 1)).unwrap();
    for (i, (name, _)) in KEYWORDS.iter().enumerate() {
        w.register_field(kw(name, 2 + i as i32)).unwrap();
    }
    for (i, name) in LONGS.iter().enumerate() {
        w.register_field(long(name, 8 + i as i32, DocValuesType::SortedNumeric))
            .unwrap();
    }
    w.register_field(long("_seq_no", 12, DocValuesType::Numeric))
        .unwrap();
    w.register_field(FieldInfo::new("_version", 13).with_doc_values(
        DocValuesType::Numeric,
        DocValuesSkipIndexType::None,
        -1,
    ))
    .unwrap();
    w.register_field(FieldInfo::new("_primary_term", 14).with_doc_values(
        DocValuesType::Numeric,
        DocValuesSkipIndexType::None,
        -1,
    ))
    .unwrap();
    w.register_field(
        FieldInfo::new("message", 15).with_index_options(IndexOptions::DocsAndFreqsAndPositions),
    )
    .unwrap();
    w.set_merge_policy(Some(MergePolicyConfig::default()));
    w.set_forced_merge_max_segment_size(Some(u64::MAX));
    w.set_max_full_flush_merge_wait_millis(0);
    w.set_use_compound_file(false);
    match std::env::var("ADAPTIVE").as_deref() {
        Ok("on") => w.set_default_doc_values_format(
            lucene_codecs::per_field_doc_values::Lucene90DocValuesFormat::rust_adaptive(),
        ),
        Ok("auto") => w.set_adaptive_doc_values_auto(),
        _ => {}
    }
    let ram_mb = std::env::var("RAM_MB").map_or(256.0, |v| v.parse().unwrap());
    w.set_ram_buffer_size_mb(ram_mb).unwrap();
    w
}

fn single(term: &[u8]) -> InvertedTerms {
    let mut t = InvertedTerms::default();
    t.push::<Vec<u8>>(term, 1, &[], &[], &[]);
    t
}

fn point(v: i64) -> FieldValue {
    FieldValue::Binary(((v as u64) ^ 0x8000_0000_0000_0000).to_be_bytes().to_vec())
}

fn document(seq: i64, r: &Row) -> ExplicitDocument {
    let c = &r.cols;
    let mut d = ExplicitDocument::default();
    let id = c[0].as_bytes();
    d.stored.push(StoredField {
        field_number: 0,
        value: FieldValue::Binary(id.to_vec()),
    });
    d.stored.push(StoredField {
        field_number: 1,
        value: FieldValue::Binary(source(r)),
    });
    d.fields.inverted.push(InvertedField {
        field_number: 0,
        terms: single(id),
        norm: None,
    });
    for i in 0..KEYWORDS.len() {
        let n = 2 + i as i32;
        let v = c[1 + i].as_bytes();
        d.fields.inverted.push(InvertedField {
            field_number: n,
            terms: single(v),
            norm: None,
        });
        d.fields.doc_values.push(StoredField {
            field_number: n,
            value: FieldValue::Binary(v.to_vec()),
        });
    }
    for i in 0..LONGS.len() {
        let n = 8 + i as i32;
        let v: i64 = c[7 + i].parse().unwrap();
        d.fields.points.push(StoredField {
            field_number: n,
            value: point(v),
        });
        d.fields.doc_values.push(StoredField {
            field_number: n,
            value: FieldValue::Long(v),
        });
    }
    d.fields.points.push(StoredField {
        field_number: 12,
        value: point(seq),
    });
    for (n, v) in [(12, seq), (13, 1), (14, 1)] {
        d.fields.doc_values.push(StoredField {
            field_number: n,
            value: FieldValue::Long(v),
        });
    }
    // WhitespaceAnalyzer: one token per word, positions in order.
    let mut terms: Vec<(&str, Vec<i32>)> = Vec::new();
    let mut at: HashMap<&str, usize> = HashMap::new();
    let mut len = 0u32;
    for (p, w) in c[11].split(' ').filter(|w| !w.is_empty()).enumerate() {
        let k = *at.entry(w).or_insert_with(|| {
            terms.push((w, Vec::new()));
            terms.len() - 1
        });
        terms[k].1.push(p as i32);
        len += 1;
    }
    let mut inv = InvertedTerms::default();
    for (term, pos) in &terms {
        inv.push::<Vec<u8>>(term.as_bytes(), pos.len() as i32, pos, &[], &[]);
    }
    d.fields.inverted.push(InvertedField {
        field_number: 15,
        terms: inv,
        norm: Some(i64::from(lucene_util::small_float::int_to_byte4(len) as i8)),
    });
    d
}

fn dir_bytes(path: &str) -> u64 {
    std::fs::read_dir(path)
        .unwrap()
        .filter_map(|e| e.ok()?.metadata().ok())
        .map(|m| m.len())
        .sum()
}

fn cpu_seconds() -> f64 {
    let s = std::fs::read_to_string("/proc/self/stat").unwrap();
    let rest = &s[s.rfind(')').unwrap() + 2..];
    let f: Vec<&str> = rest.split(' ').collect();
    let ticks: f64 = f[11].parse::<f64>().unwrap() + f[12].parse::<f64>().unwrap();
    ticks / 100.0
}

fn index(args: &[String]) {
    let (corpus_path, path) = (&args[2], &args[3]);
    let threads: usize = args[4].parse().unwrap();
    let refresh_ms: u64 = args[5].parse().unwrap();
    let docs = read_corpus(corpus_path);
    let _ = std::fs::remove_dir_all(path);
    std::fs::create_dir_all(path).unwrap();
    let dir: &'static MmapDirectory = Box::leak(Box::new(MmapDirectory::open(path)));
    let w = writer(dir);
    let cms = ConcurrentMergeScheduler::new();
    let slots = std::thread::available_parallelism().map_or(1, |n| n.get());
    let w = ConcurrentIndexWriter::with_merge_scheduler(w, slots, Arc::new(cms.clone())).unwrap();
    let next = AtomicUsize::new(0);
    let done = AtomicBool::new(false);
    let refreshes = Mutex::new(Vec::<f64>::new());
    let cpu0 = cpu_seconds();
    let start = Instant::now();
    let added = std::thread::scope(|s| {
        s.spawn(|| {
            let mut reader: Option<DirectoryReader> = None;
            while !done.load(Ordering::Acquire) {
                std::thread::sleep(std::time::Duration::from_millis(refresh_ms));
                let t = Instant::now();
                reader = Some(match reader.take() {
                    None => DirectoryReader::open_from_writer(&*w).unwrap(),
                    Some(r) => r.open_if_changed_nrt(&*w).unwrap().unwrap_or(r),
                });
                refreshes
                    .lock()
                    .unwrap()
                    .push(t.elapsed().as_secs_f64() * 1e3);
            }
        });
        let adders: Vec<_> = (0..threads)
            .map(|_| {
                s.spawn(|| loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(row) = docs.get(i) else { break };
                    w.add_explicit_documents(vec![document(i as i64, row)])
                        .unwrap();
                })
            })
            .collect();
        for a in adders {
            a.join().unwrap();
        }
        let added = start.elapsed().as_secs_f64();
        done.store(true, Ordering::Release);
        added
    });
    w.commit().unwrap();
    w.close_merges().unwrap();
    let all = start.elapsed().as_secs_f64();
    let cpu = cpu_seconds() - cpu0;
    w.commit().unwrap();
    let rf = refreshes.into_inner().unwrap();
    println!(
        "rust,{},{added:.1},{:.0},{all:.1},{cpu:.1},{},{:.1},{:.1}",
        docs.len(),
        docs.len() as f64 / added,
        rf.len(),
        rf.iter().sum::<f64>() / rf.len().max(1) as f64,
        dir_bytes(path) as f64 / 1e6
    );
    drop(w);
}

fn force_merge(args: &[String]) {
    let path = &args[2];
    let segments: usize = args[3].parse().unwrap();
    let dir = MmapDirectory::open(path);
    let mut w = writer(&dir);
    let cpu0 = cpu_seconds();
    let t = Instant::now();
    w.force_merge(segments).unwrap();
    w.commit().unwrap();
    let took = t.elapsed().as_secs_f64();
    let cpu = cpu_seconds() - cpu0;
    w.close().unwrap();
    println!(
        "rust,forcemerge,{took:.1},{cpu:.1},{:.1}",
        dir_bytes(path) as f64 / 1e6
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args[1].as_str() {
        "corpus" => corpus(&args[2], args[3].parse().unwrap(), args[4].parse().unwrap()),
        "index" => index(&args),
        "forcemerge" => force_merge(&args),
        o => panic!("unknown mode {o}"),
    }
}
