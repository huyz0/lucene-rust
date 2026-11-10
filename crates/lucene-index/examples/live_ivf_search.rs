//! Realtime search of an IVF field's buffered vectors through cluster
//! lists, against scoring every buffered vector.
//!
//! Indexes `base` vectors and flushes them (the flush clusters them), then
//! buffers `buffered` more, which live search routes to those clusters in
//! the background. Reports the routing rate, and per query the latency and
//! recall@10 (against an exact scan of the buffer) of the lists' search
//! and of an exact one.
//!
//! ```text
//! cargo run --release -p lucene-index --example live_ivf_search -- \
//!     /tmp/data/glove100.base.fbin [base] [buffered] [cluster-size]
//! ```
//!
//! The file is `.fbin`: `u32` count, `u32` dimension, then the floats.
//! The last 200 vectors of the file are the queries.
#![allow(clippy::arithmetic_side_effects)] // A measurement harness's own sizes.

use std::time::{Duration, Instant};

use lucene_codecs::buffer_vectors::BufferQuery;
use lucene_codecs::field_infos::{
    FieldInfo, IndexOptions, VectorEncoding, VectorSimilarityFunction,
};
use lucene_codecs::per_field_knn_vectors::KnnVectorsFormat;
use lucene_index::concurrent_writer::ConcurrentIndexWriter;
use lucene_index::index_writer::{
    DocumentVector, ExplicitDocument, ExplicitFields, IndexWriter, InvertedField, InvertedTerm,
};
use lucene_index::segment_info::LuceneVersion;
use lucene_store::FsDirectory;

fn read_fbin(path: &str) -> (usize, Vec<f32>) {
    let bytes = std::fs::read(path).expect("read the .fbin file");
    let n = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let dim = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let floats = bytes[8..8 + n * dim * 4]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    (dim, floats)
}

fn percentile(v: &mut [Duration], p: f64) -> u128 {
    v.sort();
    v[((v.len() - 1) as f64 * p) as usize].as_micros()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let path = a.get(1).expect("an .fbin file");
    let base: usize = a.get(2).and_then(|v| v.parse().ok()).unwrap_or(100_000);
    let buffered: usize = a.get(3).and_then(|v| v.parse().ok()).unwrap_or(100_000);
    let cluster: u32 = a.get(4).and_then(|v| v.parse().ok()).unwrap_or(128);
    let (dim, floats) = read_fbin(path);
    let total = floats.len() / dim;
    assert!(
        base + buffered + 200 <= total,
        "the file has {total} vectors"
    );
    let vector = |i: usize| floats[i * dim..(i + 1) * dim].to_vec();
    let similarity = VectorSimilarityFunction::Euclidean;

    let tmp = std::env::temp_dir().join(format!("live-ivf-search-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    let dir = FsDirectory::open(&tmp);
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", version).unwrap();
    w.set_ram_buffer_size_mb(1024.0).unwrap();
    w.enable_explicit_documents().unwrap();
    let id = w
        .register_field(
            FieldInfo::new("_id", 0)
                .with_index_options(IndexOptions::Docs)
                .with_omit_norms(true),
        )
        .unwrap();
    w.register_field(FieldInfo::new("v", 0).with_vectors(
        dim as i32,
        VectorEncoding::Float32,
        similarity,
    ))
    .unwrap();
    w.set_knn_vectors_format_for_field(
        "v",
        KnnVectorsFormat::ivf_quantized(cluster, 1, 0).unwrap(),
    );
    let w = ConcurrentIndexWriter::new(w, 1).unwrap();
    w.enable_live_search().unwrap();
    let add = |i: usize| {
        let doc = ExplicitDocument {
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
        };
        let v = vec![DocumentVector::float32("v", vector(i))];
        w.add_explicit_documents_with_vectors_in(None, vec![doc], vec![v])
            .unwrap();
    };
    for i in 0..base {
        add(i);
    }
    let t = Instant::now();
    w.flush().unwrap();
    println!(
        "flush of {base} (k-means): {:.2}s",
        t.elapsed().as_secs_f64()
    );
    let t = Instant::now();
    for i in base..base + buffered {
        add(i);
    }
    let added = t.elapsed();
    let view = w.open_live_view(None).unwrap();
    let b = view.buffers.iter().find(|b| b.in_slot).unwrap();
    let l = &b.vector_lists[0];
    while l.lists.assigned() < l.count {
        std::thread::sleep(Duration::from_millis(1));
    }
    let routed = t.elapsed();
    println!(
        "{buffered} buffered: added in {:.2}s, routed by {:.2}s ({:.1} us a vector), {:?}",
        added.as_secs_f64(),
        routed.as_secs_f64(),
        routed.as_secs_f64() * 1e6 / buffered as f64,
        l.lists
    );

    let queries: Vec<Vec<f32>> = (total - 200..total).map(vector).collect();
    let mut exact_times = Vec::new();
    let mut truth = Vec::new();
    for q in &queries {
        let t = Instant::now();
        let mut all: Vec<(i32, f32)> = (0..buffered)
            .map(|d| {
                let s = similarity.score(q, &floats[(base + d) * dim..(base + d + 1) * dim]);
                (d as i32, s)
            })
            .collect();
        all.select_nth_unstable_by(10, |a, b| b.1.total_cmp(&a.1));
        all.truncate(10);
        exact_times.push(t.elapsed());
        truth.push(all.into_iter().map(|h| h.0).collect::<Vec<_>>());
    }
    println!(
        "exact scan: p50 {} us, p99 {} us",
        percentile(&mut exact_times, 0.5),
        percentile(&mut exact_times, 0.99)
    );
    for nprobe in [0usize, 16, 32, 64, 128, 256] {
        let mut times = Vec::new();
        let mut found = 0usize;
        for (q, want) in queries.iter().zip(&truth) {
            let t = Instant::now();
            let hits = l
                .lists
                .search(&BufferQuery::Float32(q.clone()), 10, l.count, None, nprobe)
                .unwrap();
            times.push(t.elapsed());
            found += hits.iter().filter(|h| want.contains(&h.0)).count();
        }
        println!(
            "lists nprobe {nprobe} (probes {}): p50 {} us, p99 {} us, recall@10 {:.3}",
            if nprobe == 0 {
                "default".to_string()
            } else {
                nprobe.to_string()
            },
            percentile(&mut times, 0.5),
            percentile(&mut times, 0.99),
            found as f64 / (10 * queries.len()) as f64
        );
    }
    w.close_live_view(&view);
    drop(view);
    drop(w);
    let _ = std::fs::remove_dir_all(&tmp);
}
