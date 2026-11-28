#![allow(clippy::arithmetic_side_effects)]
//! The exported Faiss `Clustering`: coarse centroids trained from a sample
//! the caller chose, with the points-per-centroid cap passed in.

use lucene_faiss::kmeans::cluster;
use lucene_faiss::{Error, Metric};

/// `groups` well-separated blobs of `per` points in `d` dimensions.
fn blobs(groups: usize, per: usize, d: usize) -> Vec<f32> {
    let mut out = Vec::new();
    for g in 0..groups {
        for i in 0..per {
            for j in 0..d {
                let jitter = ((i * 7 + j * 3) % 11) as f32 * 0.01;
                out.push(if j == g % d { 10.0 } else { 0.0 } + jitter);
            }
        }
    }
    out
}

fn nearest(centroids: &[f32], d: usize, v: &[f32]) -> usize {
    centroids
        .chunks_exact(d)
        .enumerate()
        .map(|(i, c)| {
            (
                i,
                c.iter().zip(v).map(|(a, b)| (a - b) * (a - b)).sum::<f32>(),
            )
        })
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap()
        .0
}

#[test]
fn separates_blobs_into_one_centroid_each() {
    let (d, per) = (4, 50);
    let x = blobs(4, per, d);
    let c = cluster(&x, d, 4, Metric::L2, 256, 7).unwrap();
    assert_eq!((c.d, c.len(), c.metric), (d, 4, Metric::L2));
    let mut seen: Vec<usize> = (0..4)
        .map(|g| nearest(&c.data, d, &x[g * per * d..][..d]))
        .collect();
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(seen.len(), 4, "each blob owns a centroid");
}

#[test]
fn same_seed_same_centroids_and_cap_changes_the_sample() {
    let (d, per) = (4, 200);
    let x = blobs(4, per, d);
    let a = cluster(&x, d, 4, Metric::L2, 256, 3).unwrap();
    assert_eq!(a.data, cluster(&x, d, 4, Metric::L2, 256, 3).unwrap().data);
    // 39 points a centroid trains on 156 of the 800 points: still valid.
    let b = cluster(&x, d, 4, Metric::L2, 39, 3).unwrap();
    assert_eq!(b.len(), 4);
}

#[test]
fn inner_product_centroids_are_unit_length() {
    let x = blobs(3, 40, 4);
    let c = cluster(&x, 4, 3, Metric::InnerProduct, 256, 1).unwrap();
    for v in c.data.chunks_exact(4) {
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((n - 1.0).abs() < 1e-4);
    }
}

#[test]
fn refuses_bad_input() {
    let x = blobs(2, 5, 4);
    let msg = |r: Result<_, Error>| {
        r.map(|_: lucene_faiss::format::Flat| ())
            .unwrap_err()
            .to_string()
    };
    assert!(msg(cluster(&x, 4, 11, Metric::L2, 256, 0)).contains(
        "Number of training points (10) should be at least as large as number of clusters (11)"
    ));
    assert!(msg(cluster(&x, 0, 2, Metric::L2, 256, 0)).contains("dimension"));
    assert!(msg(cluster(&x[..9], 4, 2, Metric::L2, 256, 0)).contains("whole"));
    assert!(msg(cluster(&x, 4, 0, Metric::L2, 256, 0)).contains("clusters"));
    assert!(msg(cluster(&x, 4, 2, Metric::L2, 0, 0)).contains("points per centroid"));
}
