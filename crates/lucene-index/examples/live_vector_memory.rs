//! Peak memory of a vector ingest burst, live search on or off.
//!
//! Several threads add documents with one `knn_vector` field as fast as
//! they can while another opens and closes a live view every 100 ms, with
//! no commit until the end. With live search on, each buffer keeps an HNSW
//! graph that a background task fills; this measures what that costs in
//! peak RSS beside the same burst with live search off.
//!
//! ```text
//! cargo build -p lucene-index --release --example live_vector_memory
//! ./target/release/examples/live_vector_memory <dims> <docs> <on|off> [threads] [ram-mb]
//! ```
//!
//! RSS is read from `/proc/self/status` (`VmHWM`, the peak), so this is
//! Linux-only.
#![allow(clippy::arithmetic_side_effects)] // A measurement harness's own sizes.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use lucene_codecs::field_infos::{
    FieldInfo, IndexOptions, VectorEncoding, VectorSimilarityFunction,
};
use lucene_index::concurrent_writer::ConcurrentIndexWriter;
use lucene_index::index_writer::{
    DocumentVector, ExplicitDocument, ExplicitFields, IndexWriter, InvertedField, InvertedTerm,
};
use lucene_index::segment_info::LuceneVersion;
use lucene_store::FsDirectory;

fn status_kb(key: &str) -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with(key))
                .and_then(|l| l.split_whitespace().nth(1).and_then(|v| v.parse().ok()))
        })
        .unwrap_or(0)
}

/// A pseudo-random unit vector, the same for the same `i`.
fn vector(i: u64, dims: usize) -> Vec<f32> {
    let mut s = i.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut v: Vec<f32> = (0..dims)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        })
        .collect();
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
    v.iter_mut().for_each(|x| *x /= norm);
    v
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let dims: usize = a.get(1).and_then(|v| v.parse().ok()).unwrap_or(128);
    let docs: usize = a.get(2).and_then(|v| v.parse().ok()).unwrap_or(200_000);
    let live = a.get(3).is_none_or(|v| v == "on");
    let threads: usize = a.get(4).and_then(|v| v.parse().ok()).unwrap_or(4);
    let ram_mb: f64 = a.get(5).and_then(|v| v.parse().ok()).unwrap_or(64.0);

    let tmp = std::env::temp_dir().join(format!(
        "live-vector-memory-{}-{}",
        std::process::id(),
        dims
    ));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    let dir = FsDirectory::open(&tmp);
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", version).unwrap();
    w.set_ram_buffer_size_mb(ram_mb).unwrap();
    w.enable_explicit_documents().unwrap();
    let id = w
        .register_field(
            FieldInfo::new("_id", 0)
                .with_index_options(IndexOptions::Docs)
                .with_omit_norms(true),
        )
        .unwrap();
    w.register_field(FieldInfo::new("v", 0).with_vectors(
        dims as i32,
        VectorEncoding::Float32,
        VectorSimilarityFunction::DotProduct,
    ))
    .unwrap();
    let w = ConcurrentIndexWriter::new(w, threads).unwrap();
    if live {
        w.enable_live_search().unwrap();
    }
    let base_kb = status_kb("VmRSS:");

    let next = AtomicUsize::new(0);
    let peak_live = AtomicUsize::new(0);
    let views = AtomicUsize::new(0);
    let open_us = AtomicUsize::new(0);
    let lag = AtomicUsize::new(0);
    let start = Instant::now();
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| loop {
                let at = next.fetch_add(64, Ordering::Relaxed);
                if at >= docs {
                    break;
                }
                let end = (at + 64).min(docs);
                let batch: Vec<ExplicitDocument> = (at..end)
                    .map(|i| ExplicitDocument {
                        stored: Vec::new(),
                        fields: ExplicitFields {
                            inverted: vec![InvertedField {
                                field_number: id,
                                terms: vec![InvertedTerm {
                                    term: format!("d{i}").into_bytes(),
                                    freq: 1,
                                    positions: Vec::new(),
                                    offsets: Vec::new(),
                                    payloads: Vec::new(),
                                }]
                                .into(),
                                norm: None,
                            }],
                            ..ExplicitFields::default()
                        },
                    })
                    .collect();
                for (k, doc) in batch.into_iter().enumerate() {
                    let v = vec![DocumentVector::float32("v", vector((at + k) as u64, dims))];
                    w.add_explicit_documents_with_vectors_in(None, vec![doc], vec![v])
                        .unwrap();
                }
            });
        }
        s.spawn(|| {
            loop {
                if live {
                    let t = Instant::now();
                    if let Ok(view) = w.open_live_view(None) {
                        open_us.fetch_max(t.elapsed().as_micros() as usize, Ordering::Relaxed);
                        views.fetch_add(1, Ordering::Relaxed);
                        // Vectors in the view not yet in their buffer's graph.
                        let behind: i32 = view
                            .buffers
                            .iter()
                            .flat_map(|b| &b.vector_graphs)
                            .map(|g| g.count - g.graph.inserted().min(g.count))
                            .sum();
                        lag.fetch_max(behind.max(0) as usize, Ordering::Relaxed);
                        w.close_live_view(&view);
                    }
                    peak_live.fetch_max(w.live_bytes(), Ordering::Relaxed);
                }
                std::thread::sleep(Duration::from_millis(100));
                if next.load(Ordering::Relaxed) >= docs {
                    break;
                }
            }
        });
    });
    let ingest = start.elapsed();
    let before_commit = status_kb("VmHWM:");
    w.commit().unwrap();
    let total = start.elapsed();
    let peak_kb = status_kb("VmHWM:");
    let raw_mb = (docs * dims * 4) as f64 / 1_048_576.0;
    println!(
        "dims={dims} docs={docs} live={} threads={threads} ram_mb={ram_mb}",
        if live { "on" } else { "off" }
    );
    println!(
        "raw vectors {raw_mb:.1} MiB, base RSS {} MiB",
        base_kb / 1024
    );
    println!(
        "ingest {:.2}s ({:.0} docs/s), with commit {:.2}s, views opened {} (slowest open {} ms)",
        ingest.as_secs_f64(),
        docs as f64 / ingest.as_secs_f64(),
        total.as_secs_f64(),
        views.load(Ordering::Relaxed),
        open_us.load(Ordering::Relaxed) / 1000
    );
    println!(
        "peak RSS during ingest {} MiB, through commit {} MiB, peak live_bytes {} MiB, most vectors not yet in a graph {}",
        before_commit / 1024,
        peak_kb / 1024,
        peak_live.load(Ordering::Relaxed) / 1_048_576,
        lag.load(Ordering::Relaxed)
    );
    drop(w);
    let _ = std::fs::remove_dir_all(&tmp);
}
