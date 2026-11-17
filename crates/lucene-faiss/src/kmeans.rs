//! Faiss's training of an IVF's coarse centroids (`Clustering`): Lloyd's
//! iterations over a sample of at most 256 points per centroid, 25
//! iterations, centroids seeded from distinct sample points; spherical (the
//! centroids renormalised each iteration) for inner-product indexes.

use rayon::prelude::*;

use crate::distance::{distance, distance4, dot};
use crate::format::{Flat, Metric};

const ITERATIONS: usize = 25;
const MAX_POINTS_PER_CENTROID: usize = 256;

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

/// The index of the centroid nearest to `v` under `metric`.
pub(crate) fn nearest(centroids: &Flat, v: &[f32]) -> usize {
    let mut best = 0;
    let mut best_d = distance(centroids.metric, v, centroids.vector(0));
    for c in 1..centroids.len() {
        let d = distance(centroids.metric, v, centroids.vector(c));
        let better = match centroids.metric {
            Metric::L2 => d < best_d,
            Metric::InnerProduct => d > best_d,
        };
        if better {
            best = c;
            best_d = d;
        }
    }
    best
}

/// [`nearest`] for each of `group` (four at a time share the centroid loads).
fn nearest_group(centroids: &Flat, group: &[&[f32]]) -> Vec<usize> {
    let [a, b, c, d] = group else {
        return group.iter().map(|v| nearest(centroids, v)).collect();
    };
    let mut best = [0usize; 4];
    let mut best_d = distance4(centroids.metric, [a, b, c, d], centroids.vector(0));
    for ci in 1..centroids.len() {
        let ds = distance4(centroids.metric, [a, b, c, d], centroids.vector(ci));
        for r in 0..4 {
            let better = match centroids.metric {
                Metric::L2 => ds[r] < best_d[r],
                Metric::InnerProduct => ds[r] > best_d[r],
            };
            if better {
                best[r] = ci;
                best_d[r] = ds[r];
            }
        }
    }
    best.to_vec()
}

fn normalize(v: &mut [f32]) {
    let norm = dot(v, v).sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
}

/// Trains `nlist` centroids over `x` (`n` vectors of `d` floats).
pub(crate) fn train(x: &[f32], d: usize, nlist: usize, metric: Metric, seed: u64) -> Flat {
    let n = x.len() / d;
    let mut rng = Rng(seed);
    // Sample at most `nlist * 256` points.
    let want = nlist.saturating_mul(MAX_POINTS_PER_CENTROID).min(n);
    let mut order: Vec<usize> = (0..n).collect();
    for i in 0..want {
        let j = i + rng.below(n - i);
        order.swap(i, j);
    }
    let sample: Vec<&[f32]> = order[..want]
        .iter()
        .map(|&i| &x[i * d..(i + 1) * d])
        .collect();
    let mut centroids = Flat {
        d,
        metric,
        data: Vec::with_capacity(nlist * d),
    };
    for c in 0..nlist {
        // Fewer points than centroids: points repeat, as Faiss warns.
        centroids
            .data
            .extend_from_slice(sample[c % sample.len().max(1)]);
    }
    if metric == Metric::InnerProduct {
        centroids.data.chunks_exact_mut(d).for_each(normalize);
    }
    for _ in 0..ITERATIONS {
        let assigned: Vec<usize> = sample
            .par_chunks(4)
            .flat_map_iter(|group| nearest_group(&centroids, group))
            .collect();
        let mut sums = vec![0f64; nlist * d];
        let mut counts = vec![0usize; nlist];
        for (v, &c) in sample.iter().zip(&assigned) {
            counts[c] += 1;
            for (s, &x) in sums[c * d..(c + 1) * d].iter_mut().zip(*v) {
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
