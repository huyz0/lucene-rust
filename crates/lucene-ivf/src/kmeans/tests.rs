#![cfg(test)]
#![allow(clippy::arithmetic_side_effects)]
use super::*;

/// `blobs` well-separated Gaussian-ish blobs of `per` points each.
fn blobs(blobs: usize, per: usize, dim: usize) -> Vec<f32> {
    let mut rng = SplitMix64::new(1);
    let mut out = Vec::new();
    for b in 0..blobs {
        for _ in 0..per {
            for j in 0..dim {
                let center = if j == b % dim {
                    100.0 * (b / dim + 1) as f32
                } else {
                    0.0
                };
                let noise = (rng.below(1000) as f32 / 1000.0) - 0.5;
                out.push(center + noise);
            }
        }
    }
    out
}

#[test]
fn separated_blobs_become_their_own_leaves() {
    let dim = 8;
    let data = blobs(6, 50, dim);
    let c = cluster(&data, dim, 60, &KMeansParams::default(), Metric::L2);
    let k = c.len(dim);
    assert!(k >= 6, "{k} clusters");
    assert!(!c.is_empty());
    // No cluster mixes blobs.
    for b in 0..6 {
        let first = c.assignments[b * 50];
        for i in 0..50 {
            assert_eq!(c.assignments[b * 50 + i], first);
        }
    }
    let mut sizes = vec![0; k];
    for &a in &c.assignments {
        sizes[a as usize] += 1;
    }
    assert!(sizes.iter().all(|&s| s > 0 && s <= 60), "{sizes:?}");
}

#[test]
fn deterministic_for_a_seed() {
    let data = blobs(10, 40, 4);
    let p = KMeansParams::default();
    assert_eq!(
        cluster(&data, 4, 25, &p, Metric::L2),
        cluster(&data, 4, 25, &p, Metric::L2)
    );
    let ip = cluster(&data, 4, 25, &p, Metric::InnerProduct);
    assert_eq!(ip.assignments.len(), 400);
    assert_eq!(Metric::InnerProduct.cost(&[1.0, 2.0], &[3.0, 1.0]), -5.0);
}

#[test]
fn small_and_degenerate_inputs() {
    let empty = cluster(&[], 3, 10, &KMeansParams::default(), Metric::L2);
    assert!(empty.is_empty() && empty.assignments.is_empty());
    assert_eq!(empty.len(3), 0);
    let one = cluster(
        &[1.0, 2.0, 3.0, 5.0],
        2,
        10,
        &KMeansParams::default(),
        Metric::L2,
    );
    assert_eq!(one.centroids, vec![2.0, 3.5]);
    assert_eq!(one.assignments, vec![0, 0]);
    // Identical vectors: cut in order.
    let same = vec![1.0f32; 2 * 25];
    let c = cluster(&same, 2, 10, &KMeansParams::default(), Metric::L2);
    assert_eq!(c.len(2), 3);
    // A zero target behaves as one.
    let c = cluster(&[0.0, 1.0, 2.0], 1, 0, &KMeansParams::default(), Metric::L2);
    assert_eq!(c.len(1), 3);
}

#[test]
fn refinement_off_still_partitions() {
    let data = blobs(5, 30, 6);
    let p = KMeansParams {
        refine_rounds: 0,
        full_rounds: 0,
        ..KMeansParams::default()
    };
    let c = cluster(&data, 6, 40, &p, Metric::L2);
    assert_eq!(c.assignments.len(), 150);
    assert!(c.len(6) >= 4);
}

#[test]
fn empty_clusters_are_compacted() {
    let c = compact(vec![0.0, 1.0, 2.0], vec![0, 2, 2], 1, 3);
    assert_eq!(c.centroids, vec![0.0, 2.0]);
    assert_eq!(c.assignments, vec![0, 1, 1]);
}

#[test]
fn samples_are_distinct_and_sorted() {
    let mut rng = SplitMix64::new(3);
    let s = sample_indices(100, 30, &mut rng);
    assert_eq!(s.len(), 30);
    assert!(s.windows(2).all(|w| w[0] < w[1]) && s[29] < 100);
    assert_eq!(sample_indices(3, 5, &mut rng), vec![0, 1, 2]);
}

#[test]
fn soar_picks_a_different_nearby_cluster() {
    let dim = 4;
    let data = blobs(6, 30, dim);
    let c = cluster(&data, dim, 30, &KMeansParams::default(), Metric::L2);
    let spill = soar_assign(&data, dim, &c, 1.0);
    assert_eq!(spill.len(), c.assignments.len());
    let k = c.len(dim) as u32;
    for (s, a) in spill.iter().zip(&c.assignments) {
        assert!(*s == u32::MAX || (*s != *a && *s < k));
    }
    assert!(spill.iter().any(|&s| s != u32::MAX));
    let one = cluster(&data[..8], dim, 30, &KMeansParams::default(), Metric::L2);
    assert_eq!(soar_assign(&data[..8], dim, &one, 1.0), vec![u32::MAX; 2]);
    // A vector on its centroid does not spill.
    let two = Clustering {
        centroids: vec![0.0, 0.0, 5.0, 5.0],
        assignments: vec![0, 1],
    };
    assert_eq!(
        soar_assign(&[0.0, 0.0, 5.0, 4.0], 2, &two, 1.0),
        vec![u32::MAX, 0]
    );
}

#[test]
fn neighbors_exclude_self_and_sort() {
    let n = neighbor_lists(&[0.0, 10.0, 1.0, 3.0], 1, 2);
    assert_eq!(n[0], vec![2, 3]);
    assert_eq!(n[1], vec![3, 2]);
    assert_eq!(neighbor_lists(&[0.0, 1.0], 1, 5)[0], vec![1]);
}

#[test]
fn full_assignment_matches_brute_force() {
    let mut rng = SplitMix64::new(7);
    let mut random = |len: usize| -> Vec<f32> {
        (0..len)
            .map(|_| rng.below(2001) as f32 / 1000.0 - 1.0)
            .collect()
    };
    // Dimensions around the lane width; counts that leave partial
    // tiles and partial blocks.
    for dim in [1, 7, 8, 13] {
        for (n, k) in [(1, 1), (5, 2), (300, 7), (517, 100)] {
            let data = random(n * dim);
            let centroids = random(k * dim);
            let current = vec![0u32; n];
            for metric in [Metric::L2, Metric::InnerProduct] {
                let got = full_assign(&data, dim, &centroids, &current, metric);
                for (i, &g) in got.iter().enumerate() {
                    let v = row(&data, dim, i);
                    let (want, best) = nearest(v, &centroids, dim, metric);
                    let cost = metric.cost(v, row(&centroids, dim, g as usize));
                    assert!(
                        g as usize == want || (cost - best).abs() <= 1e-4 * (1.0 + best.abs()),
                        "dim {dim} n {n} k {k} {metric:?} vector {i}: {g} vs {want}"
                    );
                }
            }
        }
    }
}

#[test]
fn full_assignment_keeps_ties_in_place() {
    // Two identical centroids: each vector stays with the one it has.
    let centroids = [1.0, 0.0, 1.0, 0.0, -1.0, 0.0];
    let data = [1.0, 0.0, 1.0, 0.0, 0.9, 0.1];
    for metric in [Metric::L2, Metric::InnerProduct] {
        assert_eq!(
            full_assign(&data, 2, &centroids, &[1, 0, 2], metric),
            vec![1, 0, 0]
        );
    }
}

#[test]
fn full_rounds_alone_partition() {
    let data = blobs(6, 30, 3);
    let p = KMeansParams {
        refine_rounds: 0,
        full_rounds: 3,
        ..KMeansParams::default()
    };
    let c = cluster(&data, 3, 30, &p, Metric::L2);
    // No cluster mixes blobs.
    let mut blob_of = vec![usize::MAX; c.len(3)];
    for (i, &a) in c.assignments.iter().enumerate() {
        let b = &mut blob_of[a as usize];
        assert!(*b == usize::MAX || *b == i / 30, "cluster {a} mixes blobs");
        *b = i / 30;
    }
}

fn sizes(c: &Clustering, dim: usize) -> Vec<usize> {
    let mut out = vec![0; c.len(dim)];
    for &a in &c.assignments {
        out[a as usize] += 1;
    }
    out
}

/// Each blob's mean, as `cluster` would compute it.
fn blob_means(data: &[f32], dim: usize, blobs: usize, per: usize) -> Vec<f32> {
    let labels: Vec<u32> = (0..blobs * per).map(|i| (i / per) as u32).collect();
    let members = all_members(blobs * per);
    means(data, dim, &members, &labels, blobs, &vec![0.0; blobs * dim]).0
}

#[test]
fn reuse_keeps_hinted_vectors_and_places_the_rest() {
    let dim = 4;
    let data = blobs(6, 40, dim);
    let centroids = blob_means(&data, dim, 5, 40);
    // Blobs 0..5 keep their cluster; blob 5 is new.
    let hints: Vec<[u32; 2]> = (0..240)
        .map(|i| {
            if i < 200 {
                [(i / 40) as u32, NO_CLUSTER]
            } else {
                [NO_CLUSTER; 2]
            }
        })
        .collect();
    for (rounds, metric) in [(0, Metric::L2), (2, Metric::L2), (1, Metric::InnerProduct)] {
        let p = KMeansParams::default();
        let c = reuse(&data, dim, &centroids, &hints, 40, &p, rounds, metric);
        assert_eq!(
            c,
            reuse(&data, dim, &centroids, &hints, 40, &p, rounds, metric)
        );
        if metric == Metric::InnerProduct {
            continue;
        }
        assert_eq!(c.len(dim), 5, "{rounds}");
        for i in 0..200 {
            assert_eq!(c.assignments[i], (i / 40) as u32);
        }
        // Blob 5 sits nearest blob 1 and joins it whole (80 is not over
        // twice the target).
        assert!(c.assignments[200..].iter().all(|&a| a == 1));
        assert_eq!(sizes(&c, dim), vec![40, 80, 40, 40, 40]);
        // Centroids are the clusters' means.
        assert_eq!(&c.centroids[..dim], &centroids[..dim]);
    }
}

#[test]
fn reuse_splits_large_clusters() {
    let dim = 4;
    let data = blobs(6, 40, dim);
    let centroids = blob_means(&data, dim, 2, 40);
    let hints = vec![[NO_CLUSTER; 2]; 240];
    let c = reuse(
        &data,
        dim,
        &centroids,
        &hints,
        40,
        &KMeansParams::default(),
        1,
        Metric::L2,
    );
    let s = sizes(&c, dim);
    assert_eq!(s.iter().sum::<usize>(), 240);
    assert!(s.iter().all(|&x| x <= 80), "{s:?}");
    assert!(s.len() >= 3, "{s:?}");
    // Identical vectors over the limit are cut in order.
    let same = vec![1.0f32; 2 * 25];
    let c = reuse(
        &same,
        2,
        &[1.0, 1.0],
        &[[0, NO_CLUSTER]; 25],
        10,
        &KMeansParams::default(),
        0,
        Metric::L2,
    );
    assert_eq!(sizes(&c, 2), vec![10, 10, 5]);
}

#[test]
fn reuse_dissolves_small_clusters() {
    let dim = 4;
    let all = blobs(6, 40, dim);
    let centroids = blob_means(&all, dim, 6, 40);
    // Blob 5 lost all but five vectors.
    let data = &all[..205 * dim];
    let hints: Vec<[u32; 2]> = (0..205).map(|i| [(i / 40) as u32, NO_CLUSTER]).collect();
    let p = KMeansParams::default();
    let c = reuse(data, dim, &centroids, &hints, 40, &p, 1, Metric::L2);
    assert_eq!(sizes(&c, dim), vec![40, 45, 40, 40, 40]);
    assert!(c.assignments[200..].iter().all(|&a| a == 1));
    // When every cluster is small the largest one takes everything.
    let c = reuse(
        &[0.0, 1.0, 10.0, 11.0, 20.0],
        1,
        &[0.5, 10.5, 20.0],
        &[
            [0, 1],
            [0, NO_CLUSTER],
            [1, NO_CLUSTER],
            [1, 0],
            [2, NO_CLUSTER],
        ],
        100,
        &p,
        1,
        Metric::L2,
    );
    assert_eq!(c.assignments, vec![0; 5]);
    assert_eq!(c.centroids, vec![8.4]);
}

#[test]
fn refinement_moves_only_vectors_of_active_clusters() {
    let data = [4.0, 6.0];
    let centroids = [0.0, 10.0];
    let neighbors = neighbor_lists(&centroids, 1, 1);
    let r = |active: Option<&[bool]>| {
        refine_round(
            &data,
            1,
            &centroids,
            &[1, 0],
            &neighbors,
            Metric::L2,
            active,
        )
    };
    assert_eq!(r(None), vec![0, 1]);
    assert_eq!(r(Some(&[true, false])), vec![1, 1]);
}

#[test]
fn reuse_picks_the_nearer_of_two_hinted_clusters() {
    let c = reuse(
        &[9.0, 0.0, 1.0, 0.0],
        2,
        &[0.0, 0.0, 10.0, 0.0],
        &[[0, 1], [1, 0]],
        1,
        &KMeansParams::default(),
        0,
        Metric::L2,
    );
    assert_eq!(c.assignments, vec![1, 0]);
    assert_eq!(c.centroids, vec![1.0, 0.0, 9.0, 0.0]);
}

#[test]
fn reuse_without_clusters_or_vectors_clusters_afresh() {
    let data = blobs(3, 20, 2);
    let p = KMeansParams::default();
    let hints = vec![[NO_CLUSTER; 2]; 60];
    assert_eq!(
        reuse(&data, 2, &[], &hints, 20, &p, 1, Metric::L2),
        cluster(&data, 2, 20, &p, Metric::L2)
    );
    assert!(reuse(&[], 2, &[1.0, 2.0], &[], 20, &p, 1, Metric::L2).is_empty());
}

#[test]
#[should_panic(expected = "hint names no cluster")]
fn reuse_rejects_a_hint_past_the_clusters() {
    reuse(
        &[1.0],
        1,
        &[1.0],
        &[[1, NO_CLUSTER]],
        4,
        &KMeansParams::default(),
        0,
        Metric::L2,
    );
}

#[test]
#[should_panic(expected = "one hint per vector")]
fn reuse_needs_one_hint_per_vector() {
    reuse(
        &[1.0],
        1,
        &[1.0],
        &[],
        4,
        &KMeansParams::default(),
        0,
        Metric::L2,
    );
}

#[test]
#[should_panic(expected = "ragged")]
fn ragged_input_panics() {
    cluster(&[1.0, 2.0, 3.0], 2, 4, &KMeansParams::default(), Metric::L2);
}
