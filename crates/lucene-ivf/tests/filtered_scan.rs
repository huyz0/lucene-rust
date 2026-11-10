//! A filtered search counts the accepted vectors of the clusters it scans,
//! which is what a caller grows its probes by.
#![allow(clippy::arithmetic_side_effects)]

use std::sync::Arc;

use lucene_ivf::{FieldVectors, IvfParams, IvfReader, IvfWriter, SearchParams, Similarity};
use lucene_vector_quant::osq::OsqFactory;
use lucene_vector_quant::QuantizerRegistry;

const ID: [u8; 16] = [3; 16];

/// `n` vectors around 12 centres, vector `i` around centre `i % 12`.
fn dataset(n: usize, dim: usize) -> Vec<f32> {
    let mut x = 0x2545_F491_4F6C_DD1Du64;
    let mut next = move || {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((x >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    };
    let centres: Vec<Vec<f32>> = (0..12)
        .map(|_| (0..dim).map(|_| next() * 8.0).collect())
        .collect();
    (0..n)
        .flat_map(|i| {
            centres[i % 12]
                .iter()
                .map(|c| c + next())
                .collect::<Vec<_>>()
        })
        .collect()
}

#[test]
fn filtered_scans_count_the_accepted_vectors_and_skip_empty_clusters() {
    let (dim, n) = (16, 2400);
    let data = dataset(n, dim);
    let docs: Vec<i32> = (0..n as i32).collect();
    let mut w = IvfWriter::new(&ID, "");
    let params = IvfParams {
        target_cluster_size: 100,
        ..IvfParams::default()
    };
    w.add_field(
        &FieldVectors {
            number: 1,
            similarity: Similarity::Euclidean,
            dimension: dim,
            docs: &docs,
            vectors: &data,
        },
        &OsqFactory::bbq(),
        &params,
    )
    .unwrap();
    let files = w.finish();
    let r = IvfReader::open(
        &files.meta,
        Arc::new(files.postings.clone()),
        Arc::new(files.vectors.clone()),
        &ID,
        "",
        &QuantizerRegistry::with_builtins(),
    )
    .unwrap();
    let clusters = r.field(1).unwrap().clusters();
    let all = SearchParams::new(10, clusters, 50);
    let q = &data[5 * dim..6 * dim];
    let out = r.search_with(1, q, &all, None, None).unwrap();
    assert_eq!(out.stats.accepted, n);
    // One centre's vectors: a few clusters hold them all.
    let centre = |d: i32| d % 12 == 7;
    let accept: &dyn Fn(i32) -> bool = &centre;
    let out = r.search_with(1, q, &all, Some(accept), None).unwrap();
    assert_eq!(out.stats.accepted, n / 12);
    assert_eq!(out.stats.clusters_scanned, clusters);
    assert!(out.hits.iter().all(|h| centre(h.doc)));
    assert_eq!(out.hits.len(), 10);
    // A filter nothing passes scores nothing.
    let none: &dyn Fn(i32) -> bool = &|_| false;
    let out = r.search_with(1, q, &all, Some(none), None).unwrap();
    assert_eq!(out.stats.accepted, 0);
    assert_eq!(out.stats.candidates, 0);
    assert!(out.hits.is_empty());
}
