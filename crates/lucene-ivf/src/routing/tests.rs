#![cfg(test)]
use super::*;
use crate::kmeans::SplitMix64;

fn random(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = SplitMix64::new(seed);
    (0..n * dim)
        .map(|_| (rng.next_u64() >> 40) as f32 / (1u64 << 24) as f32 - 0.5)
        .collect()
}

const SIMILARITIES: [Similarity; 4] = [
    Similarity::Euclidean,
    Similarity::Cosine,
    Similarity::DotProduct,
    Similarity::MaximumInnerProduct,
];

#[test]
fn exact_routing_matches_the_f32_scan() {
    let dim = 24;
    let centroids = random(300, dim, 1);
    let queries = random(20, dim, 2);
    for s in SIMILARITIES {
        let r = Router::new(&centroids, dim, s, Routing::Exact, None);
        for q in queries.chunks_exact(dim) {
            assert_eq!(
                r.route(&centroids, q, 17),
                exact_route(&centroids, dim, s, q, 17)
            );
        }
    }
}

#[test]
fn quantized_routing_finds_the_nearest_centroids() {
    let dim = 64;
    let centroids = random(2000, dim, 3);
    let queries = random(50, dim, 4);
    for s in SIMILARITIES {
        let r = Router::new(&centroids, dim, s, Routing::Quantized, None);
        let mut found = 0;
        for q in queries.chunks_exact(dim) {
            let got = r.route(&centroids, q, 32);
            let want = exact_route(&centroids, dim, s, q, 32);
            assert_eq!(got.len(), 32);
            // The nearest centroid is always found and ranked first.
            assert_eq!(got[0], want[0], "{s:?}");
            found += got.iter().filter(|c| want.contains(c)).count();
        }
        assert!(found * 100 >= 99 * 50 * 32, "{s:?}: {found}");
    }
}

#[test]
fn tree_routing_returns_distinct_clusters_nearest_first() {
    let dim = 16;
    let centroids = random(3000, dim, 5);
    let queries = random(30, dim, 6);
    for s in SIMILARITIES {
        let r = Router::new(&centroids, dim, s, Routing::Tree { cover: 32 }, None);
        assert!(r.has_tree());
        let exact = ExactScore::new(s, &centroids, dim);
        let mut found = 0;
        for q in queries.chunks_exact(dim) {
            let got = r.route(&centroids, q, 20);
            assert_eq!(got.len(), 20);
            let mut seen = got.clone();
            seen.sort_unstable();
            seen.dedup();
            assert_eq!(seen.len(), 20);
            let score = |c: usize| exact.score(&centroids, dim, q, c);
            assert!(got.windows(2).all(|p| score(p[0]) >= score(p[1])));
            let want = exact_route(&centroids, dim, s, q, 20);
            found += got.iter().filter(|c| want.contains(c)).count();
        }
        assert!(found * 10 >= 8 * 30 * 20, "{s:?}: {found}");
    }
}

#[test]
fn small_and_empty_fields() {
    let dim = 8;
    for routing in [Routing::Exact, Routing::Quantized, Routing::Auto] {
        let r = Router::new(&[], dim, Similarity::Euclidean, routing, None);
        assert!(r.is_empty());
        assert!(r.route(&[], &[0.0; 8], 4).is_empty());
        let centroids = random(5, dim, 7);
        let r = Router::new(&centroids, dim, Similarity::Cosine, routing, None);
        let mut got = r.route(&centroids, &centroids[..dim], 10);
        assert_eq!(got[0], 0);
        got.sort_unstable();
        assert_eq!(got, vec![0, 1, 2, 3, 4]);
    }
    // Too few clusters for a tree: routes flat.
    let centroids = random(50, dim, 8);
    let r = Router::new(
        &centroids,
        dim,
        Similarity::Euclidean,
        Routing::Tree { cover: 4 },
        None,
    );
    assert!(!r.has_tree());
    assert_eq!(r.route(&centroids, &centroids[..dim], 50).len(), 50);
}

#[test]
fn zero_query_and_identical_centroids() {
    let dim = 8;
    let centroids = vec![0.25f32; 200 * dim];
    for s in SIMILARITIES {
        for routing in [Routing::Quantized, Routing::Tree { cover: 2 }] {
            let r = Router::new(&centroids, dim, s, routing, None);
            for q in [vec![0f32; dim], vec![1f32; dim]] {
                let got = r.route(&centroids, &q, 7);
                assert_eq!(got.len(), 7);
            }
        }
    }
}

#[test]
fn euclidean_routing_far_from_the_origin() {
    let dim = 32;
    let s = Similarity::Euclidean;
    // How many of the exact 16 nearest centroids each routing finds, with
    // every coordinate shifted by `offset`.
    let found = |offset: f32| {
        let mut centroids = random(2000, dim, 9);
        let mut queries = random(40, dim, 10);
        for x in centroids.iter_mut().chain(queries.iter_mut()) {
            *x += offset;
        }
        // The exact path is the f32 squared distance itself.
        let r = Router::new(&centroids, dim, s, Routing::Exact, None);
        for q in queries.chunks_exact(dim) {
            let got = r.route(&centroids, q, 8);
            let d = |c: usize| simd::square_distance_f32(q, kmeans::row(&centroids, dim, c));
            let best = (0..2000).map(d).fold(f32::INFINITY, f32::min);
            assert_eq!(d(got[0]), best);
        }
        [Routing::Quantized, Routing::Tree { cover: 32 }].map(|routing| {
            let r = Router::new(&centroids, dim, s, routing, None);
            queries
                .chunks_exact(dim)
                .map(|q| {
                    let want = exact_route(&centroids, dim, s, q, 16);
                    let got = r.route(&centroids, q, 16);
                    got.iter().filter(|c| want.contains(c)).count()
                })
                .sum::<usize>()
        })
    };
    let near = found(0.0);
    let far = found(1000.0);
    // The codes are centred, so routing far from the origin is as good
    // as near it (the tree is weak on structureless data either way).
    assert!(far[0] * 10 >= 9 * 40 * 16, "{far:?}");
    assert!(far[1] * 20 >= near[1] * 19, "{near:?} {far:?}");
}

#[test]
fn bad_parents_are_ignored() {
    let dim = 8;
    let centroids = random(300, dim, 11);
    let good = tree_parents(&centroids, dim, Similarity::Euclidean);
    assert_eq!(parent_count(&good), {
        let mut p = good.clone();
        p.sort_unstable();
        p.dedup();
        p.len()
    });
    let short = &good[..299];
    let mut gap = good.clone();
    gap.iter_mut().for_each(|p| *p *= 2);
    let mut huge = good.clone();
    huge[0] = u32::MAX;
    for parents in [short, &gap[..], &huge[..]] {
        let r = Router::new(
            &centroids,
            dim,
            Similarity::Euclidean,
            Routing::Auto,
            Some(parents),
        );
        assert!(!r.has_tree());
        assert_eq!(r.route(&centroids, &centroids[..dim], 5)[0], 0);
    }
    let r = Router::new(
        &centroids,
        dim,
        Similarity::Euclidean,
        Routing::Auto,
        Some(&good),
    );
    assert!(r.has_tree());
}
