//! Picking a field's quantizer per segment from its own vectors.
#![allow(clippy::arithmetic_side_effects)]

use std::sync::Arc;

use lucene_ivf::{
    AutoQuantizer, FieldVectors, IvfParams, IvfReader, IvfWriter, SearchParams, Similarity,
    AUTO_MIN_VECTORS,
};
use lucene_vector_quant::osq::OsqFactory;
use lucene_vector_quant::rabitq::ExRabitqFactory;
use lucene_vector_quant::{QuantizerFactory, QuantizerRegistry};

const ID: [u8; 16] = [3; 16];

/// `n` vectors of `dim` around 40 centres.
fn dataset(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    let mut next = move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    };
    let centres: Vec<Vec<f32>> = (0..40)
        .map(|_| (0..dim).map(|_| next() * 4.0).collect())
        .collect();
    (0..n)
        .flat_map(|i| {
            centres[i % centres.len()]
                .iter()
                .map(|x| x + next())
                .collect::<Vec<_>>()
        })
        .collect()
}

/// The quantizer name `auto` wrote for an `n`-vector field, and the reader.
fn written(n: usize, read_bytes: usize) -> (String, IvfReader, Vec<f32>) {
    let dim = 128;
    let data = dataset(n, dim, 5);
    let docs: Vec<i32> = (0..n as i32).collect();
    let auto = AutoQuantizer::new(vec![
        Box::new(OsqFactory::bbq()),
        Box::new(ExRabitqFactory::new(4)),
    ]);
    let mut w = IvfWriter::new(&ID, "");
    w.add_field(
        &FieldVectors {
            number: 1,
            similarity: Similarity::Euclidean,
            dimension: dim,
            docs: &docs,
            vectors: &data,
        },
        &auto,
        &IvfParams {
            target_cluster_size: 128,
            calibration_recall: Some(0.9),
            auto_rerank_read_bytes: read_bytes,
            ..IvfParams::default()
        },
    )
    .unwrap();
    let files = w.finish();
    let reader = IvfReader::open(
        &files.meta,
        Arc::new(files.postings.clone()),
        Arc::new(files.vectors.clone()),
        &ID,
        "",
        &QuantizerRegistry::with_builtins(),
    )
    .unwrap();
    let name = reader.field(1).unwrap().quantizer().name().to_string();
    (name, reader, data)
}

#[test]
fn auto_records_the_quantizer_it_picked() {
    let auto = AutoQuantizer::with_defaults();
    assert_eq!(auto.name(), "auto");
    assert_eq!(auto.candidates().map(<[_]>::len), Some(5));
    // A small field gets the first candidate.
    let (small, _, _) = written(AUTO_MIN_VECTORS - 1, 4096);
    assert_eq!(small, "osq");
}

/// Where a reranked vector costs a disk page, the codes that need the
/// fewest reranks win; where it costs nothing, the smallest codes do.
#[test]
fn the_pick_follows_the_cost_of_a_rerank() {
    let n = AUTO_MIN_VECTORS + 2000;
    let (disk, reader, data) = written(n, 1 << 20);
    assert_eq!(disk, "erq");
    let (memory, _, _) = written(n, 0);
    assert_eq!(memory, "osq");
    // The picked codes search as any other.
    let mut found = 0;
    for q in data.chunks_exact(128).step_by(997).take(10) {
        let truth = reader.exact_search(1, q, 10, None).unwrap();
        let hits = reader
            .search(1, q, &SearchParams::new(10, 16, 50), None)
            .unwrap();
        found += hits
            .iter()
            .filter(|h| truth.iter().any(|t| t.doc == h.doc))
            .count();
    }
    assert!(found >= 90, "{found} of 100");
}
