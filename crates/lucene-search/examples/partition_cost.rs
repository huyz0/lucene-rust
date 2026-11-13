//! What a partition layout costs on the write side: the Rust half of
//! `scripts/bench-partition-cost.sh`, over the corpus the Java side
//! (`benchmarks/java-runner/src/PartitionCostBench.java`) writes.
//!
//! ```text
//! partition_cost vectors <out.fbin> <n> <dim> <seed>
//! partition_cost index <corpus.tsv> <dir> <layout> <threads> <refresh ms> [<vectors.fbin> <hnsw|ivf|ivf-p>]
//! partition_cost forcemerge <dir> <layout> <segments> [<vectors.fbin> <hnsw|ivf|ivf-p>]
//! partition_cost knn <dir> <corpus.tsv> <vectors.fbin> <queries.fbin>
//! ```
//!
//! Layouts: `plain` (arrival order), `tenant` (index sort on the tenant),
//! `composite` (tenant, then time newest first) and `segments` (each
//! tenant's documents in segments of their own,
//! `IndexWriter::set_segment_partitioning`, as `OPEN_PARTITIONED` opens a
//! shard). `index` writes as the plugin's engine writer does: a concurrent
//! writer with one buffer per CPU (1024 with `segments`), a 256 MB buffer,
//! `TieredMergePolicy` and `ConcurrentMergeScheduler` defaults, a
//! near-real-time reader refreshed every `refresh ms` while `threads`
//! threads add the corpus in arrival order, then a commit. It prints one CSV
//! row: docs/s, commit time, refresh count and times, merge count and
//! time, segments and size. `forcemerge` times `force_merge` and a commit.
//! `knn` prints tenant-filtered top-10 latency and recall per tenant class.
//!
//! `STORE_SOURCE=1` also stores the body as a `_source` field, and `HC=1`
//! turns on `IndexWriter::set_merge_lz4_high_compression`, as
//! `docs/benchmarks/settings-defaults.md` runs it.

#![allow(clippy::arithmetic_side_effects)]

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use lucene_codecs::field_infos::{
    DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions, VectorEncoding,
    VectorSimilarityFunction,
};
use lucene_codecs::per_field_knn_vectors::KnnVectorsFormat;
use lucene_codecs::stored_fields::{FieldValue, StoredField};
use lucene_index::concurrent_writer::ConcurrentIndexWriter;
use lucene_index::index_writer::{
    DocumentVector, ExplicitDocument, IndexWriter, InvertedField, InvertedTerms, MergeEvent,
    MergeListener, MergeTimes, SegmentPartitioning,
};
use lucene_index::merge_policy::MergePolicyConfig;
use lucene_index::merge_scheduler::ConcurrentMergeScheduler;
use lucene_index::segment_info::{
    IndexSortField, IndexSortKind, LuceneVersion, NumericSortKey, SortedNumericSelector,
    SortedSetSelector, StringMissingValue,
};
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::knn_search::{knn_top_hits_with, KnnTarget};
use lucene_search::query::{Clause, TermQuery};
use lucene_store::directory::Directory;
use lucene_store::MmapDirectory;

const VERSION: LuceneVersion = LuceneVersion {
    major: 10,
    minor: 5,
    bugfix: 0,
};

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
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn gauss(&mut self) -> f32 {
        let (u, v) = (self.unit().max(1e-7), self.unit());
        (-2.0 * u.ln()).sqrt() * (std::f32::consts::TAU * v).cos()
    }
}

fn normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    v.iter_mut().for_each(|x| *x /= n);
}

fn read_bin(path: &str) -> (usize, Vec<f32>) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let d = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    (
        d,
        bytes[8..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect(),
    )
}

/// Vectors around 256 random centres, unrelated to the tenant.
fn vectors(out: &str, n: usize, dim: usize, seed: u64) {
    let mut c = Rng(1);
    let centres: Vec<f32> = (0..256 * dim).map(|_| c.gauss()).collect();
    let mut r = Rng(seed);
    let mut f = Vec::with_capacity(8 + n * dim * 4);
    f.extend_from_slice(&(n as u32).to_le_bytes());
    f.extend_from_slice(&(dim as u32).to_le_bytes());
    let mut v = vec![0f32; dim];
    for _ in 0..n {
        let k = (r.next() % 256) as usize;
        for (j, x) in v.iter_mut().enumerate() {
            *x = centres[k * dim + j] + 0.7 * r.gauss();
        }
        normalize(&mut v);
        for x in &v {
            f.extend_from_slice(&x.to_le_bytes());
        }
    }
    std::fs::write(out, f).unwrap();
}

struct Row {
    id: i32,
    tenant: String,
    ts: i64,
    body: String,
}

fn corpus(path: &str) -> Vec<Row> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|l| {
            let f: Vec<&str> = l.splitn(4, '\t').collect();
            Row {
                id: f[0].parse().unwrap(),
                tenant: f[1].to_string(),
                ts: f[2].parse().unwrap(),
                body: f[3].to_string(),
            }
        })
        .collect()
}

#[derive(Default)]
struct Merges {
    count: AtomicU64,
    ms: AtomicU64,
}

impl MergeListener for Merges {
    fn before_merge(&self, _: &MergeEvent) {}
    fn after_merge(
        &self,
        _: &MergeEvent,
        times: MergeTimes,
        _: Option<&lucene_index::index_writer::Error>,
    ) {
        self.count.fetch_add(1, Ordering::Relaxed);
        self.ms.fetch_add(times.took_ms, Ordering::Relaxed);
    }
}

fn sort(layout: &str) -> Option<Vec<IndexSortField>> {
    let tenant = IndexSortField {
        field: "tenant".into(),
        reverse: false,
        kind: IndexSortKind::SortedSet {
            selector: SortedSetSelector::Min,
            missing: StringMissingValue::None,
        },
    };
    let ts = IndexSortField {
        field: "ts".into(),
        reverse: true,
        kind: IndexSortKind::SortedNumeric {
            key: NumericSortKey::Long(None),
            selector: SortedNumericSelector::Min,
        },
    };
    match layout {
        "plain" | "segments" => None,
        "tenant" => Some(vec![tenant]),
        "composite" => Some(vec![tenant, ts]),
        o => panic!("layout {o}"),
    }
}

/// A writer configured as the engine writer opens a shard.
fn writer<'d>(
    dir: &'d dyn Directory,
    layout: &str,
    vec: Option<(usize, &str)>,
    merges: Arc<Merges>,
) -> IndexWriter<'d> {
    let mut w = IndexWriter::open(dir, Vec::new(), "Lucene104", VERSION).unwrap();
    w.enable_explicit_documents().unwrap();
    if let Some(s) = sort(layout) {
        w.set_index_sort(Some(&s)).unwrap();
    }
    if layout == "segments" {
        w.set_segment_partitioning(Some(SegmentPartitioning::opensearch()));
    }
    w.register_field(FieldInfo::new("id", 0)).unwrap();
    w.register_field(
        FieldInfo::new("tenant", 1)
            .with_index_options(IndexOptions::Docs)
            .with_omit_norms(true)
            .with_doc_values(DocValuesType::SortedSet, DocValuesSkipIndexType::None, -1),
    )
    .unwrap();
    w.register_field(
        FieldInfo::new("ts", 2)
            .with_points(1, 1, 8)
            .with_doc_values(
                DocValuesType::SortedNumeric,
                DocValuesSkipIndexType::None,
                -1,
            ),
    )
    .unwrap();
    match vec {
        None => {
            w.register_field(
                FieldInfo::new("body", 3)
                    .with_index_options(IndexOptions::DocsAndFreqsAndPositions),
            )
            .unwrap();
        }
        Some((dim, fmt)) => {
            w.register_field(FieldInfo::new("v", 3).with_vectors(
                dim as i32,
                VectorEncoding::Float32,
                VectorSimilarityFunction::Cosine,
            ))
            .unwrap();
            w.set_knn_vectors_format_for_field(
                "v",
                match fmt {
                    "ivf" | "ivf-p" => KnnVectorsFormat::ivf(384, 1).unwrap(),
                    "hnsw" => KnnVectorsFormat::hnsw(16, 100).unwrap(),
                    o => panic!("{o}"),
                },
            );
            w.set_ivf_partition_clusters((fmt == "ivf-p").then_some(20_000));
        }
    }
    if std::env::var("STORE_SOURCE").is_ok() {
        w.register_field(FieldInfo::new("_source", 4)).unwrap();
    }
    w.set_merge_lz4_high_compression(std::env::var("HC").is_ok());
    w.set_merge_policy(Some(MergePolicyConfig::default()));
    w.set_forced_merge_max_segment_size(Some(u64::MAX));
    w.set_max_full_flush_merge_wait_millis(0);
    w.set_use_compound_file(false);
    let ram_mb = std::env::var("RAM_MB").map_or(256.0, |v| v.parse().unwrap());
    w.set_ram_buffer_size_mb(ram_mb).unwrap();
    w.set_merge_listener(Some(merges));
    w
}

fn document(r: &Row, vector: Option<&[f32]>) -> (ExplicitDocument, Vec<DocumentVector>) {
    let mut d = ExplicitDocument::default();
    d.stored.push(StoredField {
        field_number: 0,
        value: FieldValue::Int(r.id),
    });
    if std::env::var("STORE_SOURCE").is_ok() {
        d.stored.push(StoredField {
            field_number: 4,
            value: FieldValue::Binary(r.body.as_bytes().to_vec()),
        });
    }
    let mut t = InvertedTerms::default();
    t.push::<Vec<u8>>(r.tenant.as_bytes(), 1, &[], &[], &[]);
    d.fields.inverted.push(InvertedField {
        field_number: 1,
        terms: t,
        norm: None,
    });
    d.fields.doc_values.push(StoredField {
        field_number: 1,
        value: FieldValue::Binary(r.tenant.as_bytes().to_vec()),
    });
    d.fields.points.push(StoredField {
        field_number: 2,
        value: FieldValue::Binary(
            ((r.ts as u64) ^ 0x8000_0000_0000_0000)
                .to_be_bytes()
                .to_vec(),
        ),
    });
    d.fields.doc_values.push(StoredField {
        field_number: 2,
        value: FieldValue::Long(r.ts),
    });
    let mut vectors = Vec::new();
    match vector {
        None => {
            // WhitespaceAnalyzer: one token per word, positions in order.
            let mut terms: HashMap<&str, Vec<i32>> = HashMap::new();
            let mut n = 0u32;
            for (p, w) in r.body.split(' ').filter(|w| !w.is_empty()).enumerate() {
                terms.entry(w).or_default().push(p as i32);
                n += 1;
            }
            let mut inv = InvertedTerms::default();
            for (term, pos) in &terms {
                inv.push::<Vec<u8>>(term.as_bytes(), pos.len() as i32, pos, &[], &[]);
            }
            d.fields.inverted.push(InvertedField {
                field_number: 3,
                terms: inv,
                // BM25Similarity.computeNorm: a signed byte.
                norm: Some(i64::from(lucene_util::small_float::int_to_byte4(n) as i8)),
            });
        }
        Some(v) => vectors.push(DocumentVector::float32("v", v.to_vec())),
    }
    (d, vectors)
}

fn dir_bytes(path: &str) -> u64 {
    std::fs::read_dir(path)
        .unwrap()
        .filter_map(|e| e.ok()?.metadata().ok())
        .map(|m| m.len())
        .sum()
}

fn pct(v: &[f64], p: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v[((v.len() as f64 * p) as usize).min(v.len() - 1)]
}

fn index(args: &[String]) {
    let (corpus_path, path, layout) = (&args[2], &args[3], args[4].as_str());
    let threads: usize = args[5].parse().unwrap();
    let refresh_ms: u64 = args[6].parse().unwrap();
    let rows = corpus(corpus_path);
    let vecs = args.get(7).map(|p| read_bin(p));
    let fmt = args.get(8).map(String::as_str);
    let _ = std::fs::remove_dir_all(path);
    std::fs::create_dir_all(path).unwrap();
    let dir: &'static MmapDirectory = Box::leak(Box::new(MmapDirectory::open(path)));
    let merges = Arc::new(Merges::default());
    let w = writer(
        dir,
        layout,
        vecs.as_ref().map(|(d, _)| (*d, fmt.unwrap())),
        merges.clone(),
    );
    let cms = ConcurrentMergeScheduler::new();
    let mut slots = std::thread::available_parallelism().map_or(1, |n| n.get());
    if layout == "segments" {
        slots = slots.max(1024);
    }
    let w = ConcurrentIndexWriter::with_merge_scheduler(w, slots, Arc::new(cms.clone())).unwrap();
    let next = AtomicUsize::new(0);
    let done = AtomicBool::new(false);
    let refreshes = Mutex::new(Vec::<f64>::new());
    let start = Instant::now();
    let added = std::thread::scope(|s| {
        s.spawn(|| {
            let mut reader: Option<DirectoryReader> = None;
            while !done.load(Ordering::Acquire) {
                std::thread::sleep(std::time::Duration::from_millis(refresh_ms));
                let t = Instant::now();
                reader = Some(match reader.take() {
                    None => DirectoryReader::open_from_writer(&*w).unwrap(),
                    Some(r) => match r.open_if_changed_nrt(&*w).unwrap() {
                        Some(n) => n,
                        None => r,
                    },
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
                    let Some(r) = rows.get(i) else { break };
                    let v = vecs.as_ref().map(|(d, v)| &v[i * d..(i + 1) * d]);
                    let (doc, vectors) = document(r, v);
                    let partition: Option<Arc<str>> =
                        (layout == "segments").then(|| Arc::from(r.tenant.as_str()));
                    w.add_explicit_documents_with_vectors_in(partition, vec![doc], vec![vectors])
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
    let c0 = Instant::now();
    w.commit().unwrap();
    let commit = c0.elapsed().as_secs_f64();
    // Let the merges the commit scheduled finish, as Java's `sync`.
    w.close_merges().unwrap();
    let all = start.elapsed().as_secs_f64();
    let segs = DirectoryReader::open_from_writer(&*w)
        .unwrap()
        .segment_readers()
        .len();
    w.commit().unwrap();
    let mut rf = refreshes.into_inner().unwrap();
    rf.sort_by(f64::total_cmp);
    let kind = match fmt {
        None => "fts",
        Some(f) => f,
    };
    println!(
        "rust,{layout},{kind},{},{added:.1},{:.0},{commit:.1},{},{:.1},{:.1},{:.1},{},{:.1},{segs},{all:.1},{:.1}",
        rows.len(),
        rows.len() as f64 / added,
        rf.len(),
        rf.iter().sum::<f64>() / rf.len().max(1) as f64,
        pct(&rf, 0.5),
        pct(&rf, 0.99),
        merges.count.load(Ordering::Relaxed),
        merges.ms.load(Ordering::Relaxed) as f64 / 1e3,
        dir_bytes(path) as f64 / 1e6
    );
    drop(w);
}

fn force_merge(args: &[String]) {
    let (path, layout) = (&args[2], args[3].as_str());
    let segments: usize = args[4].parse().unwrap();
    let vec = args.get(5).map(|p| (read_bin(p).0, args[6].as_str()));
    let dir = MmapDirectory::open(path);
    let mut w = writer(&dir, layout, vec, Arc::new(Merges::default()));
    let t = Instant::now();
    w.force_merge(segments).unwrap();
    w.commit().unwrap();
    let took = t.elapsed().as_secs_f64();
    w.close().unwrap();
    let segs = DirectoryReader::open(&dir).unwrap().segment_readers().len();
    println!(
        "rust,{layout},forcemerge,{segments},{took:.1},{:.1},{segs}",
        dir_bytes(path) as f64 / 1e6
    );
}

fn knn(args: &[String]) {
    const K: usize = 10;
    let (path, corpus_path) = (&args[2], &args[3]);
    let rows = corpus(corpus_path);
    let (dim, base) = read_bin(&args[4]);
    let (_, queries) = read_bin(&args[5]);
    let dir = MmapDirectory::open(path);
    let reader = DirectoryReader::open(&dir).unwrap();
    let opened = reader.open_segments().unwrap();
    let segs = opened.as_open_segments();
    let mut ids = Vec::new();
    for seg in reader.segment_readers() {
        for doc in 0..seg.max_doc {
            match seg.stored_document(doc).unwrap().unwrap().fields[0].value {
                FieldValue::Int(id) => ids.push(id as usize),
                ref o => panic!("{o:?}"),
            }
        }
    }
    let mut by_tenant: HashMap<&str, Vec<usize>> = HashMap::new();
    for r in &rows {
        by_tenant.entry(&r.tenant).or_default().push(r.id as usize);
    }
    let nq = queries.len() / dim;
    println!("tenant_class,queries,mean_us,p50_us,p99_us,recall");
    for (class, tenants) in [
        ("large", (0..5).collect::<Vec<u32>>()),
        ("mid", (20..60).collect()),
        ("small", (300..900).step_by(5).collect()),
    ] {
        let mut times = Vec::new();
        let (mut found, mut total) = (0, 0);
        for pass in 0..2 {
            for (qi, t) in (0..nq).zip(tenants.iter().cycle()) {
                let tenant = format!("t{t}");
                let Some(members) = by_tenant.get(tenant.as_str()) else {
                    continue;
                };
                let q = &queries[qi * dim..(qi + 1) * dim];
                let clause = Clause::Term(TermQuery::new("tenant", tenant.as_bytes()));
                let t0 = Instant::now();
                let hits = knn_top_hits_with(
                    &segs,
                    "v",
                    &KnnTarget::Float(q.to_vec()),
                    K,
                    100,
                    Some(&clause),
                    0,
                )
                .unwrap();
                let us = t0.elapsed().as_secs_f64() * 1e6;
                if pass == 0 {
                    continue;
                }
                times.push(us);
                let mut s: Vec<(f32, usize)> = members
                    .iter()
                    .map(|&i| {
                        let v = &base[i * dim..(i + 1) * dim];
                        (v.iter().zip(q).map(|(a, b)| a * b).sum::<f32>(), i)
                    })
                    .collect();
                s.sort_by(|a, b| b.0.total_cmp(&a.0));
                let truth: Vec<usize> = s.iter().take(K).map(|p| p.1).collect();
                found += hits
                    .iter()
                    .take(K)
                    .filter(|h| truth.contains(&ids[h.doc_id as usize]))
                    .count();
                total += truth.len();
            }
        }
        times.sort_by(f64::total_cmp);
        println!(
            "{class},{},{:.0},{:.0},{:.0},{:.4}",
            times.len(),
            times.iter().sum::<f64>() / times.len().max(1) as f64,
            pct(&times, 0.5),
            pct(&times, 0.99),
            found as f64 / total.max(1) as f64
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args[1].as_str() {
        "vectors" => vectors(
            &args[2],
            args[3].parse().unwrap(),
            args[4].parse().unwrap(),
            args[5].parse().unwrap(),
        ),
        "index" => index(&args),
        "forcemerge" => force_merge(&args),
        "knn" => knn(&args),
        m => panic!("mode {m}"),
    }
}
