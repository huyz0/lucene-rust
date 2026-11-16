#![cfg(test)]
// ARITH: test fixtures of a few thousand vectors.
#![allow(clippy::arithmetic_side_effects)]

use super::*;

/// Deterministic pseudo-random unit vectors, around `centers` clusters.
fn vectors(n: usize, dim: usize, centers: usize) -> Vec<Vec<f32>> {
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    };
    let means: Vec<Vec<f32>> = (0..centers)
        .map(|_| (0..dim).map(|_| next() * 4.0).collect())
        .collect();
    (0..n)
        .map(|i| {
            let m = &means[i % centers];
            let mut v: Vec<f32> = m.iter().map(|x| x + next()).collect();
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.iter_mut().for_each(|x| *x /= norm);
            v
        })
        .collect()
}

/// Brute-force routing over fixed centroids; vectors whose best dot
/// product is below `outlier` are outliers.
struct Brute {
    centroids: Vec<Vec<f32>>,
    outlier: f32,
    nprobe: usize,
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

impl Brute {
    fn ranked(&self, q: &[f32]) -> Vec<(usize, f32)> {
        let mut r: Vec<(usize, f32)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(c, m)| (c, dot(q, m)))
            .collect();
        r.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        r
    }
}

impl ClusterRouter for Brute {
    fn clusters(&self) -> usize {
        self.centroids.len()
    }
    fn assign(&self, v: &[f32]) -> Option<usize> {
        let (c, s) = self.ranked(v)[0];
        (s >= self.outlier).then_some(c)
    }
    fn probe(&self, q: &[f32], nprobe: usize) -> Vec<usize> {
        self.ranked(q)
            .into_iter()
            .take(nprobe)
            .map(|(c, _)| c)
            .collect()
    }
    fn default_nprobe(&self) -> usize {
        self.nprobe
    }
}

fn unit_means(dim: usize, centers: usize) -> Vec<Vec<f32>> {
    // The data's own means: every fourth vector of a fresh draw is near
    // its cluster's mean, so averaging recovers them.
    let vs = vectors(centers * 50, dim, centers);
    (0..centers)
        .map(|c| {
            let mut m = vec![0.0f32; dim];
            for v in vs.iter().skip(c).step_by(centers) {
                m.iter_mut().zip(v).for_each(|(a, b)| *a += b);
            }
            let norm = m.iter().map(|x| x * x).sum::<f32>().sqrt();
            m.iter_mut().for_each(|x| *x /= norm);
            m
        })
        .collect()
}

fn lists(vs: &[Vec<f32>], router: Brute) -> BufferLists {
    let dim = vs[0].len();
    let l = BufferLists::new(VectorSimilarityFunction::DotProduct, dim, Arc::new(router));
    for (i, v) in vs.iter().enumerate() {
        l.push(i as i32 * 2, BufferVector::Float32(Arc::new(v.clone())))
            .unwrap();
    }
    l
}

fn exact(vs: &[Vec<f32>], q: &[f32], k: usize, keep: impl Fn(usize) -> bool) -> Vec<i32> {
    let mut all: Vec<(i32, f32)> = vs
        .iter()
        .enumerate()
        .filter(|(i, _)| keep(*i))
        .map(|(i, v)| {
            (
                i as i32 * 2,
                VectorSimilarityFunction::DotProduct.score(q, v),
            )
        })
        .collect();
    all.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    all.into_iter().take(k).map(|(d, _)| d).collect()
}

/// Probing the nearest clusters finds what an exact scan finds, with
/// the unrouted tail and the outliers scored too.
#[test]
fn probed_lists_find_the_exact_neighbours() {
    let (dim, centers) = (32, 40);
    let vs = vectors(20_000, dim, centers);
    let l = lists(
        &vs,
        Brute {
            centroids: unit_means(dim, centers),
            outlier: 0.5,
            nprobe: 4,
        },
    );
    assert!(l.schedule());
    assert!(!l.schedule(), "one run at a time");
    l.assign_pending(1000);
    assert_eq!(l.assigned(), 20_000);
    let outliers = l.lists[centers].read().unwrap().len();
    assert!(outliers < 2000, "{outliers} outliers");
    let mut recall = 0usize;
    for q in vectors(50, dim, centers).iter().skip(3) {
        let hits = l
            .search(&BufferQuery::Float32(q.clone()), 10, 20_000, None, 0)
            .unwrap();
        let want = exact(&vs, q, 10, |_| true);
        recall += hits.iter().filter(|h| want.contains(&h.0)).count();
    }
    assert!(recall as f64 / 470.0 > 0.95, "recall {recall}/470");
}

/// A reader that saw fewer vectors than the lists hold reads only its
/// own; one that saw more scores the unrouted ones exactly.
#[test]
fn a_reader_reads_its_own_count() {
    let (dim, centers) = (32, 40);
    let vs = vectors(20_000, dim, centers);
    let l = lists(
        &vs,
        Brute {
            centroids: unit_means(dim, centers),
            outlier: -1.0,
            nprobe: 4,
        },
    );
    l.assign_batch(15_000);
    assert_eq!(l.assigned(), 15_000);
    let q = vs[17_000].clone();
    let hits = l
        .search(&BufferQuery::Float32(q.clone()), 1, 20_000, None, 0)
        .unwrap();
    assert_eq!(hits[0].0, 34_000, "unrouted, scored exactly");
    let hits = l
        .search(&BufferQuery::Float32(q), 5, 16_000, None, 0)
        .unwrap();
    assert!(hits.iter().all(|h| h.0 < 32_000), "{hits:?}");
}

/// A filter passing no more than the accepted target is scored
/// exactly, wherever its vectors lie.
#[test]
fn a_small_filter_is_scored_exactly() {
    let (dim, centers) = (32, 40);
    let vs = vectors(20_000, dim, centers);
    let l = lists(
        &vs,
        Brute {
            centroids: unit_means(dim, centers),
            outlier: -1.0,
            nprobe: 1,
        },
    );
    l.assign_pending(usize::MAX);
    // Only cluster 7's vectors pass: far from most queries.
    let mut accept = FixedBitSet::new(20_000);
    for i in (7..20_000).step_by(centers).take(100) {
        // FBS: `i < 20_000`, the bitset's length.
        accept.set(i);
    }
    let q = vs[3].clone();
    let hits = l
        .search(
            &BufferQuery::Float32(q.clone()),
            10,
            20_000,
            Some(&accept),
            1,
        )
        .unwrap();
    let want = exact(&vs, &q, 10, |i| accept.get(i));
    assert_eq!(hits.iter().map(|h| h.0).collect::<Vec<_>>(), want);
}

/// A filter passing more than the accepted target widens the probes
/// from the filter's share until they have scanned the target: here
/// six clusters pass, far from the query's, and the nearest of them are
/// found at one probe asked.
#[test]
fn a_filter_widens_the_probes() {
    let (dim, centers) = (32, 40);
    let vs = vectors(20_000, dim, centers);
    let l = lists(
        &vs,
        Brute {
            centroids: unit_means(dim, centers),
            outlier: -1.0,
            nprobe: 1,
        },
    );
    l.assign_pending(usize::MAX);
    let mut accept = FixedBitSet::new(20_000);
    for i in 0..20_000 {
        if (7..13).contains(&(i % centers)) {
            // FBS: `i < 20_000`, the bitset's length.
            accept.set(i);
        }
    }
    assert!(accept.cardinality() > ivf_accepted_target(10));
    let q = vs[3].clone();
    let hits = l
        .search(
            &BufferQuery::Float32(q.clone()),
            10,
            20_000,
            Some(&accept),
            1,
        )
        .unwrap();
    let want = exact(&vs, &q, 10, |i| accept.get(i));
    assert_eq!(hits.iter().map(|h| h.0).collect::<Vec<_>>(), want);
}

#[test]
fn accepted_target_grows_with_the_root_of_k() {
    assert_eq!(ivf_accepted_target(1), IVF_ACCEPTED_FOR_10);
    assert_eq!(ivf_accepted_target(10), IVF_ACCEPTED_FOR_10);
    assert_eq!(ivf_accepted_target(40), 2 * IVF_ACCEPTED_FOR_10);
}

#[test]
fn small_bad_and_retired() {
    let vs = vectors(100, 8, 4);
    let l = lists(
        &vs,
        Brute {
            centroids: unit_means(8, 4),
            outlier: -1.0,
            nprobe: 1,
        },
    );
    assert!(l
        .push(1000, BufferVector::Float32(Arc::new(vec![0.0; 3])))
        .is_err());
    assert!(l
        .push(1000, BufferVector::Byte(Arc::new(vec![0; 8])))
        .is_err());
    assert!(l
        .search(&BufferQuery::Byte(vec![0; 8]), 1, 100, None, 0)
        .is_err());
    l.retire();
    assert!(!l.schedule());
    l.assign_pending(10);
    assert_eq!(l.assigned(), 0);
    // Small: exact.
    let hits = l
        .search(&BufferQuery::Float32(vs[5].clone()), 1, 100, None, 0)
        .unwrap();
    assert_eq!(hits[0].0, 10);
    assert!(l.ram_bytes() > 0);
    assert!(format!("{l:?}").contains("BufferLists"));
}
