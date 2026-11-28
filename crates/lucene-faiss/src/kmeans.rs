//! Faiss's training of an IVF's coarse centroids (`Clustering`): Lloyd's
//! iterations over a sample of at most 256 points per centroid, 25
//! iterations, centroids seeded from distinct sample points; spherical (the
//! centroids renormalised each iteration) for inner-product indexes.

use rayon::prelude::*;

use crate::distance::dot;
use crate::error::{Error, Result};
use crate::format::{Flat, Metric};
use lucene_util::gemm;

const ITERATIONS: usize = 25;
/// Faiss's default `max_points_per_centroid`.
pub const MAX_POINTS_PER_CENTROID: usize = 256;

/// Small deterministic generator (SplitMix64) for sampling and seeding.
pub(crate) struct Rng(pub(crate) u64);

impl Rng {
    pub(crate) fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    pub(crate) fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    pub(crate) fn unit_float(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Rows per matrix-product block: the block's scores (`BLOCK * nlist`
/// floats) stay near the cache.
const BLOCK: usize = 256;

/// Every row of `x`'s nearest centroid by one matrix product per block, as
/// Faiss does with BLAS: Euclidean ranks `c.c / 2 - x.c`, inner product
/// `-x.c`. Ties go to the lower centroid.
pub(crate) fn assign_all(x: &[f32], d: usize, centroids: &Flat) -> Vec<usize> {
    let k = centroids.len();
    let half_norms: Vec<f32> = match centroids.metric {
        Metric::L2 => centroids
            .data
            .chunks_exact(d)
            .map(|c| 0.5 * dot(c, c))
            .collect(),
        Metric::InnerProduct => vec![0.0; k],
    };
    let blocks: Vec<Vec<usize>> = x
        .par_chunks(BLOCK * d)
        .map(|rows| {
            let mut scores = vec![0f32; rows.len() / d * k];
            gemm::dot_rows(rows, &centroids.data, d, &mut scores);
            scores
                .chunks_exact(k)
                .map(|row| {
                    let mut best = 0;
                    let mut best_s = f32::INFINITY;
                    for (c, (&dot, &h)) in row.iter().zip(&half_norms).enumerate() {
                        let s = h - dot;
                        if s < best_s {
                            best = c;
                            best_s = s;
                        }
                    }
                    best
                })
                .collect()
        })
        .collect();
    blocks.concat()
}

fn normalize(v: &mut [f32]) {
    let norm = dot(v, v).sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
}

/// Trains `nlist` centroids over `x` (`n` vectors of `d` floats) on at most
/// 256 points a centroid.
pub(crate) fn train(x: &[f32], d: usize, nlist: usize, metric: Metric, seed: u64) -> Flat {
    train_capped(x, d, nlist, metric, MAX_POINTS_PER_CENTROID, seed)
}

/// Faiss's `Clustering` over `x` (`n` vectors of `d` floats): `k` centroids
/// by 25 Lloyd iterations over a random sample of at most
/// `points_per_centroid * k` points (`max_points_per_centroid`; Faiss warns
/// below 39 and uses 256), seeded from distinct sample points, empty clusters
/// split from a populated one, and, for [`Metric::InnerProduct`], spherical
/// (every centroid renormalised each iteration). The result is the same for
/// the same `seed`.
///
/// # Errors
/// [`Error::Invalid`] for a zero dimension, `k` or `points_per_centroid`, a
/// sample that is not whole vectors, or fewer points than centroids (Faiss's
/// `nx >= k` check, in its message's shape).
pub fn cluster(
    x: &[f32],
    d: usize,
    k: usize,
    metric: Metric,
    points_per_centroid: usize,
    seed: u64,
) -> Result<Flat> {
    if d == 0 {
        return Err(Error::Invalid("dimension must be positive".into()));
    }
    if !x.len().is_multiple_of(d) {
        return Err(Error::Invalid(format!(
            "sample of {} floats is not a whole number of {d}-dimensional vectors",
            x.len()
        )));
    }
    if k == 0 {
        return Err(Error::Invalid("number of clusters must be positive".into()));
    }
    if points_per_centroid == 0 {
        return Err(Error::Invalid(
            "points per centroid must be positive".into(),
        ));
    }
    let n = x.len() / d;
    if n < k {
        return Err(Error::Invalid(format!(
            "Number of training points ({n}) should be at least as large as number of clusters ({k})"
        )));
    }
    Ok(train_capped(x, d, k, metric, points_per_centroid, seed))
}

fn train_capped(
    x: &[f32],
    d: usize,
    nlist: usize,
    metric: Metric,
    points_per_centroid: usize,
    seed: u64,
) -> Flat {
    let n = x.len() / d;
    let mut rng = Rng(seed);
    let want = nlist.saturating_mul(points_per_centroid).min(n);
    let mut order: Vec<usize> = (0..n).collect();
    for i in 0..want {
        let j = i + rng.below(n - i);
        order.swap(i, j);
    }
    let mut sample: Vec<f32> = Vec::with_capacity(want * d);
    for &i in &order[..want] {
        sample.extend_from_slice(&x[i * d..(i + 1) * d]);
    }
    let mut centroids = Flat {
        d,
        metric,
        data: Vec::with_capacity(nlist * d),
    };
    for c in 0..nlist {
        // Fewer points than centroids: points repeat, as Faiss warns.
        centroids
            .data
            .extend_from_slice(&sample[(c % want.max(1)) * d..][..d]);
    }
    if metric == Metric::InnerProduct {
        centroids.data.chunks_exact_mut(d).for_each(normalize);
    }
    for _ in 0..ITERATIONS {
        let assigned = assign_all(&sample, d, &centroids);
        let mut sums = vec![0f64; nlist * d];
        let mut counts = vec![0usize; nlist];
        for (v, &c) in sample.chunks_exact(d).zip(&assigned) {
            counts[c] += 1;
            for (s, &x) in sums[c * d..(c + 1) * d].iter_mut().zip(v) {
                *s += f64::from(x);
            }
        }
        for c in 0..nlist {
            if counts[c] == 0 {
                // An empty cluster takes a point of a populated one, which
                // is how Faiss splits clusters to keep all centroids used.
                let mut donor = rng.below(nlist);
                while counts[donor] < 2 {
                    donor = (donor + 1) % nlist;
                    if donor == c {
                        break;
                    }
                }
                let src: Vec<f32> = centroids.vector(donor).to_vec();
                centroids.data[c * d..(c + 1) * d].copy_from_slice(&src);
                for (i, v) in centroids.data[c * d..(c + 1) * d].iter_mut().enumerate() {
                    // Perturb symmetrically so the pair separates.
                    *v *= if i % 2 == 0 {
                        1.0 + 1.0 / 1024.0
                    } else {
                        1.0 - 1.0 / 1024.0
                    };
                }
                continue;
            }
            let inv = 1.0 / counts[c] as f64;
            for i in 0..d {
                centroids.data[c * d + i] = (sums[c * d + i] * inv) as f32;
            }
        }
        if metric == Metric::InnerProduct {
            centroids.data.chunks_exact_mut(d).for_each(normalize);
        }
    }
    centroids
}
