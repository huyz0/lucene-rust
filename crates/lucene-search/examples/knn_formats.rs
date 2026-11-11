//! Build time, size, recall and latency of one KNN field through the real
//! index path (IndexWriter, segment reader, `knn_top_hits`) per vectors
//! format: HNSW, scalar-quantized HNSW and `LuceneRustIvfVectorsFormat`.
//!
//! ```text
//! cargo run --release -p lucene-search --example knn_formats -- <prefix> <l2|cosine> [options]
//!   --formats L   comma list of hnsw, hnsw-int7, hnsw-sq1/sq2/sq4 (Lucene104
//!                 scalar quantized HNSW), hnsw-<quantizer> (the graph walked on
//!                 any `IVF_QUANTIZERS` codes: hnsw-erq2, hnsw-rabitq, ...), ivf,
//!                 ivf-f16 and ivf-<quantizer> (f16 rerank, ivf-erq4, ivf-osq2, ...)
//!                 (default hnsw, hnsw-int7, ivf, ivf-f16)
//!   --queries N   measured queries (default 1000), after 200 warm-up ones
//!   --segments N  flushed segments, force-merged to one when --merge is set (default 1)
//!   --nprobes L   IVF probes per segment, 0 = calibrated (default 0)
//!   --ef L        HNSW candidate counts to sweep (default 100)
//!   --dir D       where the indexes are written (default a temporary directory)
//!   --reuse       keep each format's index under D and search it again on
//!                 later runs with the same data and build options
//!   --targets L   recall targets: each format searched recall-targeted
//!                 (`knn_recall`), after --tune-queries tuning queries, then
//!                 again at the tuned effort without sampling (the monitor's
//!                 cost is the difference)
//!   --sample-rate R  the targeted searches' sample rate (default 0.01)
//!   --budget B    the targeted searches' check budget (default 0.05)
//!   --tune-queries N queries run before measuring a target (default 3000)
//! ```
//!
//! `<prefix>` names `.base.fbin`, `.query.fbin` and `.gt.ibin` as
//! `benchmarks/ivf/prepare.py` writes them. Prints one CSV row per format
//! and setting.
#![allow(clippy::arithmetic_side_effects)]

use std::time::Instant;

use lucene_codecs::field_infos::{FieldInfo, VectorEncoding, VectorSimilarityFunction};
use lucene_codecs::per_field_knn_vectors::{KnnVectorsFormat, IVF_QUANTIZERS};
use lucene_codecs::stored_fields::{FieldValue, StoredField};
use lucene_index::concurrent_writer::ConcurrentIndexWriter;
use lucene_index::index_writer::{DocumentVector, ExplicitDocument, IndexWriter};
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::knn_recall::{
    knn_top_hits_at_effort, knn_top_hits_exact, knn_top_hits_tuned, RecallTarget, RecallTuner,
};
use lucene_search::knn_search::{knn_top_hits_with, KnnTarget};
use lucene_store::FsDirectory;
use lucene_util::quantization::ScalarEncoding;

const K: usize = 10;

fn read_bin<T: Copy>(path: &str, decode: fn([u8; 4]) -> T) -> (usize, usize, Vec<T>) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let n = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let d = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let data = bytes[8..]
        .chunks_exact(4)
        .map(|c| decode(c.try_into().unwrap()))
        .collect();
    (n, d, data)
}

fn format(name: &str) -> KnnVectorsFormat {
    match name {
        "hnsw" => KnnVectorsFormat::hnsw(16, 100).unwrap(),
        "hnsw-int7" => {
            KnnVectorsFormat::hnsw_scalar_quantized(ScalarEncoding::SevenBit, 16, 100).unwrap()
        }
        "hnsw-sq1" => {
            KnnVectorsFormat::hnsw_scalar_quantized(ScalarEncoding::SingleBitQueryNibble, 16, 100)
                .unwrap()
        }
        "hnsw-sq2" => {
            KnnVectorsFormat::hnsw_scalar_quantized(ScalarEncoding::DibitQueryNibble, 16, 100)
                .unwrap()
        }
        "hnsw-sq4" => {
            KnnVectorsFormat::hnsw_scalar_quantized(ScalarEncoding::PackedNibble, 16, 100).unwrap()
        }
        "ivf" => KnnVectorsFormat::ivf(384, 0).unwrap(),
        "ivf-f16" => KnnVectorsFormat::ivf(384, 1).unwrap(),
        other if other.starts_with("hnsw-") => {
            // `hnsw-<quantizer>`: the graph walked on the named codes.
            let quantizer = IVF_QUANTIZERS
                .iter()
                .position(|&n| Some(n) == other.strip_prefix("hnsw-"))
                .unwrap_or_else(|| panic!("unknown format {other}"));
            KnnVectorsFormat::hnsw_quantized(quantizer as u8, 16, 100).unwrap()
        }
        other => {
            // `ivf-<quantizer>`: f16 rerank vectors, the named cluster codes.
            let quantizer = other
                .strip_prefix("ivf-")
                .and_then(|q| IVF_QUANTIZERS.iter().position(|&n| n == q))
                .unwrap_or_else(|| panic!("unknown format {other}"));
            KnnVectorsFormat::ivf_quantized(384, 1, quantizer as u8).unwrap()
        }
    }
}

fn dir_size(path: &std::path::Path) -> u64 {
    std::fs::read_dir(path)
        .unwrap()
        .filter_map(|e| e.ok()?.metadata().ok())
        .map(|m| m.len())
        .sum()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let prefix = &args[1];
    let similarity = match args[2].as_str() {
        "l2" => VectorSimilarityFunction::Euclidean,
        "cosine" => VectorSimilarityFunction::Cosine,
        m => panic!("unknown metric {m}"),
    };
    let opt = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .map(|i| args[i + 1].clone())
    };
    let formats = opt("--formats").unwrap_or_else(|| "hnsw,hnsw-int7,ivf,ivf-f16".into());
    let segments: usize = opt("--segments").map_or(1, |s| s.parse().unwrap());
    let merge = args.iter().any(|a| a == "--merge");
    let nprobes: Vec<usize> = opt("--nprobes")
        .unwrap_or_else(|| "0".into())
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    let efs: Vec<usize> = opt("--ef")
        .unwrap_or_else(|| "100".into())
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    let (n, dim, base) = read_bin(&format!("{prefix}.base.fbin"), f32::from_le_bytes);
    let (nq_all, _, queries) = read_bin(&format!("{prefix}.query.fbin"), f32::from_le_bytes);
    let (_, gt_k, gt) = read_bin(&format!("{prefix}.gt.ibin"), i32::from_le_bytes);
    let nq = opt("--queries")
        .map_or(1000, |s| s.parse().unwrap())
        .min(nq_all.saturating_sub(200));
    let reuse = args.iter().any(|a| a == "--reuse");
    let targets: Vec<f32> = opt("--targets").map_or_else(Vec::new, |s| {
        s.split(',').map(|s| s.parse().unwrap()).collect()
    });
    let sample_rate: f32 = opt("--sample-rate").map_or(0.01, |s| s.parse().unwrap());
    let budget: f32 = opt("--budget").map_or(0.05, |s| s.parse().unwrap());
    let tune_queries: usize = opt("--tune-queries").map_or(3000, |s| s.parse().unwrap());
    let root = opt("--dir").map_or_else(std::env::temp_dir, std::path::PathBuf::from);
    eprintln!("{prefix}: {n} x {dim}, {nq} queries, {segments} segments, merge {merge}");
    println!("format,setting,segments,build_s,index_mb,recall@{K},mean_us,p99_us");
    for name in formats.split(',') {
        let path = if reuse {
            root.join(format!("knn-formats-{name}"))
        } else {
            root.join(format!("knn-formats-{}-{name}", std::process::id()))
        };
        // The marker names what the kept index was built from; a missing
        // marker (an interrupted build) or another build rebuilds it.
        let marker = root.join(format!("knn-formats-{name}.built"));
        let key = format!("{prefix} {} {segments} {merge}", args[2]);
        let built = reuse && std::fs::read_to_string(&marker).is_ok_and(|k| k == key);
        if !built {
            let _ = std::fs::remove_file(&marker);
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).unwrap();
        }
        let dir = FsDirectory::open(&path);
        let t = Instant::now();
        if !built {
            let mut w = IndexWriter::open(
                &dir,
                Vec::new(),
                "Lucene104",
                LuceneVersion {
                    major: 10,
                    minor: 5,
                    bugfix: 0,
                },
            )
            .unwrap();
            w.enable_explicit_documents().unwrap();
            w.register_field(FieldInfo::new("id", 0)).unwrap();
            w.register_field(FieldInfo::new("v", 0).with_vectors(
                dim as i32,
                VectorEncoding::Float32,
                similarity,
            ))
            .unwrap();
            w.set_knn_vectors_format_for_field("v", format(name));
            w.set_ram_buffer_size_mb(4096.0).unwrap();
            w.set_merge_policy(None);
            let w = ConcurrentIndexWriter::new(w, 1).unwrap();
            let per = n.div_ceil(segments);
            for start in (0..n).step_by(per) {
                let (docs, vectors): (Vec<_>, Vec<_>) = (start..(start + per).min(n))
                    .map(|i| {
                        let mut d = ExplicitDocument::default();
                        d.stored.push(StoredField {
                            field_number: 0,
                            value: FieldValue::Int(i as i32),
                        });
                        (
                            d,
                            vec![DocumentVector::float32(
                                "v",
                                base[i * dim..(i + 1) * dim].to_vec(),
                            )],
                        )
                    })
                    .unzip();
                w.add_explicit_documents_with_vectors_in(None, docs, vectors)
                    .unwrap();
                w.flush().unwrap();
            }
            w.commit().unwrap();
            let mut w = w.into_writer().unwrap();
            if merge {
                w.force_merge(1).unwrap();
                w.commit().unwrap();
            }
            w.close().unwrap();
            if reuse {
                std::fs::write(&marker, &key).unwrap();
            }
        }
        let build = t.elapsed().as_secs_f64();
        let size = dir_size(&path) as f64 / 1e6;
        let reader = DirectoryReader::open(&dir).unwrap();
        let opened = reader.open_segments().unwrap();
        let segs = opened.as_open_segments();
        // A merge may reorder documents: hits map back to base ordinals
        // through the stored id.
        let mut ids: Vec<i32> = Vec::with_capacity(n);
        for seg in reader.segment_readers() {
            for doc in 0..seg.max_doc {
                let stored = seg.stored_document(doc).unwrap().unwrap();
                match stored.fields[0].value {
                    FieldValue::Int(id) => ids.push(id),
                    ref other => panic!("id {other:?}"),
                }
            }
        }
        // IVF sweeps probes and sizes its own rerank; HNSW sweeps its beam.
        let settings: Vec<(usize, usize)> = if name.starts_with("ivf") {
            nprobes.iter().map(|&p| (p, K)).collect()
        } else {
            efs.iter().map(|&ef| (0, ef)).collect()
        };
        let pool = |q: usize| KnnTarget::Float(queries[q * dim..(q + 1) * dim].to_vec());
        let recall_of = |q: usize, hits: &[lucene_search::ScoreDoc]| {
            let truth = &gt[q * gt_k..q * gt_k + K];
            hits.iter()
                .filter(|h| truth.contains(&ids[h.doc_id as usize]))
                .count()
        };
        if !targets.is_empty() {
            // The exact search the recall checks compare with.
            let mut found = 0usize;
            let t = Instant::now();
            for q in 0..200.min(nq) {
                let hits = knn_top_hits_exact(&segs, "v", &pool(q), K, None, None).unwrap();
                found += recall_of(q, &hits);
            }
            println!(
                "{name},exact,{},{build:.1},{size:.1},{:.4},{:.0},",
                reader.segment_readers().len(),
                found as f64 / (200.min(nq) * K) as f64,
                t.elapsed().as_secs_f64() * 1e6 / 200.min(nq) as f64
            );
        }
        // Recall-targeted rows: tuned (sampling) and fixed at the tuned effort.
        for &target in &targets {
            let candidates = if name.starts_with("ivf") { K } else { efs[0] };
            let tuner = RecallTuner::new(
                RecallTarget::new(target, sample_rate)
                    .unwrap()
                    .with_budget(budget)
                    .unwrap(),
            );
            let tuned = |q: usize| {
                knn_top_hits_tuned(
                    &segs,
                    "v",
                    &pool(q),
                    K,
                    candidates,
                    None,
                    None,
                    0,
                    &[],
                    &tuner,
                )
                .unwrap()
            };
            // Tuning queries come from past the measured ones, cycling.
            let spare = nq_all - nq;
            let t0 = Instant::now();
            for i in 0..tune_queries {
                tuned(nq + i % spare);
            }
            let tune_s = t0.elapsed().as_secs_f64();
            let before = tuner.stats();
            let mut runs: Vec<(String, Vec<f64>, usize)> = Vec::new();
            // `fixed-tuneset`: the tuned effort over the tuning queries.
            for mode in ["tuned", "fixed", "fixed-tuneset"] {
                let effort = tuner.stats().effort;
                let mut times = Vec::with_capacity(nq);
                let mut found = 0usize;
                for i in 0..nq {
                    let q = if mode == "fixed-tuneset" {
                        nq + i % spare
                    } else {
                        i
                    };
                    let t = Instant::now();
                    let hits = if mode == "tuned" {
                        tuned(q)
                    } else {
                        knn_top_hits_at_effort(
                            &segs,
                            "v",
                            &pool(q),
                            K,
                            candidates,
                            None,
                            None,
                            0,
                            &[],
                            effort,
                        )
                        .unwrap()
                    };
                    times.push(t.elapsed().as_secs_f64() * 1e6);
                    found += recall_of(q, &hits);
                }
                runs.push((mode.to_string(), times, found));
            }
            let after = tuner.stats();
            for (mode, mut times, found) in runs {
                times.sort_by(f64::total_cmp);
                let mean = times.iter().sum::<f64>() / nq as f64;
                let p50 = times[nq / 2];
                let p99 = times[(nq * 99 / 100).min(nq - 1)];
                println!(
                    "{name},target={target}/{mode}/effort={:.3}->{:.3}/est={:.4}/lb={:.4}/samples={}/check_share={:.3}/tune_s={tune_s:.1},{},{build:.1},{size:.1},{:.4},{mean:.0},{p99:.0},p50={p50:.0}",
                    before.effort,
                    after.effort,
                    after.recall.unwrap_or(f64::NAN),
                    after.recall_lower_bound.unwrap_or(f64::NAN),
                    after.samples,
                    after.check_nanos as f64 / after.search_nanos.max(1) as f64,
                    reader.segment_readers().len(),
                    found as f64 / (nq * K) as f64
                );
            }
        }
        for &(nprobe, candidates) in &settings {
            let search = |q: usize| {
                knn_top_hits_with(
                    &segs,
                    "v",
                    &KnnTarget::Float(queries[q * dim..(q + 1) * dim].to_vec()),
                    K,
                    candidates,
                    None,
                    nprobe,
                )
                .unwrap()
            };
            for q in nq..(nq + 200).min(nq_all) {
                search(q);
            }
            let mut times = Vec::with_capacity(nq);
            let mut found = 0usize;
            for q in 0..nq {
                let t = Instant::now();
                let hits = search(q);
                times.push(t.elapsed().as_secs_f64() * 1e6);
                let truth = &gt[q * gt_k..q * gt_k + K];
                found += hits
                    .iter()
                    .filter(|h| truth.contains(&ids[h.doc_id as usize]))
                    .count();
            }
            times.sort_by(f64::total_cmp);
            let mean = times.iter().sum::<f64>() / nq as f64;
            let p99 = times[(nq * 99 / 100).min(nq - 1)];
            let setting = if name.starts_with("ivf") {
                format!("nprobes={nprobe}")
            } else {
                format!("ef={candidates}")
            };
            println!(
                "{name},{setting},{},{build:.1},{size:.1},{:.4},{mean:.0},{p99:.0}",
                reader.segment_readers().len(),
                found as f64 / (nq * K) as f64
            );
        }
        drop(opened);
        drop(reader);
        if !reuse {
            let _ = std::fs::remove_dir_all(&path);
        }
    }
}
