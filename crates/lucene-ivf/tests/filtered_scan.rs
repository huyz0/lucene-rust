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

fn open(data: &[f32], dim: usize, spill: Option<f32>) -> IvfReader {
    let n = data.len() / dim;
    let docs: Vec<i32> = (0..n as i32).collect();
    let mut w = IvfWriter::new(&ID, "");
    let params = IvfParams {
        target_cluster_size: 100,
        spill,
        ..IvfParams::default()
    };
    w.add_field(
        &FieldVectors {
            number: 1,
            similarity: Similarity::Euclidean,
            dimension: dim,
            docs: &docs,
            vectors: data,
        },
        &OsqFactory::bbq(),
        &params,
    )
    .unwrap();
    let files = w.finish();
    IvfReader::open(
        &files.meta,
        Arc::new(files.postings.clone()),
        Arc::new(files.vectors.clone()),
        &ID,
        "",
        &QuantizerRegistry::with_builtins(),
    )
    .unwrap()
}

#[test]
fn filtered_scans_count_the_accepted_vectors_and_skip_empty_clusters() {
    let (dim, n) = (16, 2400);
    let data = dataset(n, dim);
    let r = open(&data, dim, None);
    let clusters = r.field(1).unwrap().clusters();
    let all = SearchParams::new(10, clusters, 50);
    let q = &data[5 * dim..6 * dim];
    let out = r.search_with(1, q, &all, None, None).unwrap();
    assert_eq!(out.stats.accepted, n);
    assert_eq!(
        (out.stats.clusters_scanned, out.stats.clusters_empty),
        (clusters, 0)
    );
    // One centre's vectors: a few clusters hold them all.
    let centre = |d: i32| d % 12 == 7;
    let accept: &dyn Fn(i32) -> bool = &centre;
    let out = r.search_with(1, q, &all, Some(accept), None).unwrap();
    assert_eq!(out.stats.accepted, n / 12);
    assert!(out.stats.clusters_empty > clusters / 2, "{:?}", out.stats);
    assert_eq!(
        out.stats.clusters_scanned + out.stats.clusters_empty,
        clusters
    );
    assert!(out.hits.iter().all(|h| centre(h.doc)));
    assert_eq!(out.hits.len(), 10);
    // A filter nothing passes scores nothing.
    let none: &dyn Fn(i32) -> bool = &|_| false;
    let out = r.search_with(1, q, &all, Some(none), None).unwrap();
    assert_eq!(out.stats.accepted, 0);
    assert_eq!(out.stats.clusters_scanned, 0);
    assert!(out.hits.is_empty());
}

#[test]
fn spilled_vectors_count_once_per_cluster_they_are_in() {
    let (dim, n) = (16, 2400);
    let data = dataset(n, dim);
    let r = open(&data, dim, Some(1.0));
    let f = r.field(1).unwrap();
    assert_eq!(f.entries(), 2 * n);
    let all = SearchParams::new(10, f.clusters(), 50);
    let out = r.search_with(1, &data[..dim], &all, None, None).unwrap();
    assert_eq!(out.stats.accepted, f.entries());
    let even: &dyn Fn(i32) -> bool = &|d| d % 2 == 0;
    let out = r
        .search_with(1, &data[..dim], &all, Some(even), None)
        .unwrap();
    assert_eq!(out.stats.accepted, n);
}

#[test]
fn clusters_of_partitions_hold_one_partition_each_and_prune_by_range() {
    let (dim, n) = (16, 4000);
    let data = dataset(n, dim);
    let docs: Vec<i32> = (0..n as i32).collect();
    let mut w = IvfWriter::new(&ID, "");
    let params = IvfParams {
        target_cluster_size: 100,
        partition_min_vectors: 500,
        ..IvfParams::default()
    };
    let field = FieldVectors {
        number: 1,
        similarity: Similarity::Euclidean,
        dimension: dim,
        docs: &docs,
        vectors: &data,
    };
    // Partitions of 1000, 1000, 980, 20 and 1000 documents: the small one
    // is a run of its own between two large ones.
    w.add_partitioned_field(
        &field,
        &[0, 1000, 2000, 2980, 3000],
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
    let f = r.field(1).unwrap();
    let ranges = f.doc_ranges().expect("narrow ranges");
    assert_eq!(ranges.len(), f.clusters());
    for &(lo, hi) in ranges {
        assert_eq!(
            lo / 1000,
            hi / 1000,
            "cluster spans partitions: {lo}..={hi}"
        );
    }
    // A filter on docs 1000..2000: only that run's clusters are probed.
    let among: Vec<usize> = (0..f.clusters())
        .filter(|&c| ranges[c].0 >= 1000 && ranges[c].1 < 2000)
        .collect();
    assert!(!among.is_empty() && among.len() < f.clusters());
    let run: &dyn Fn(i32) -> bool = &|d| (1000..2000).contains(&d);
    let q = &data[1500 * dim..1501 * dim];
    let p = SearchParams::new(10, among.len(), 50);
    let out = r.search_within(1, q, &p, Some(run), &among, None).unwrap();
    assert_eq!(out.stats.clusters_scanned, among.len());
    assert_eq!(out.stats.accepted, 1000);
    assert_eq!(out.hits.len(), 10);
    assert!(out.hits.iter().all(|h| run(h.doc)));
    // Probing them all finds what a search of every cluster finds.
    let all = r
        .search_with(
            1,
            q,
            &SearchParams::new(10, f.clusters(), 50),
            Some(run),
            None,
        )
        .unwrap();
    assert_eq!(out.hits, all.hits);
    // Clustered as one, the field's ranges are wide: none are kept.
    let r = open(&data, dim, None);
    assert!(r.field(1).unwrap().doc_ranges().is_none());
}
