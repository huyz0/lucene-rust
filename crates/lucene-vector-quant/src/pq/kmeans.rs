//! Lloyd's k-means over one subspace's vectors, as Faiss's `Clustering`
//! runs it for each sub-quantizer of a product quantizer: 25 iterations,
//! centroids seeded from distinct sample points, an empty cluster split off
//! a populated one.

use lucene_util::gemm;

const ITERATIONS: usize = 25;
/// Rows per matrix-product block: a block's scores stay near the cache.
const BLOCK: usize = 256;

/// SplitMix64.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Every row of `x`'s nearest centroid by Euclidean distance (ranking
/// `c.c / 2 - x.c`, one matrix product per block); ties go to the lower
/// centroid.
pub(super) fn assign(x: &[f32], d: usize, centroids: &[f32]) -> Vec<usize> {
    let k = centroids.len() / d;
    let half_norms: Vec<f32> = centroids
        .chunks_exact(d)
        .map(|c| 0.5 * c.iter().map(|v| v * v).sum::<f32>())
        .collect();
    let mut out = Vec::with_capacity(x.len() / d);
    let mut scores = vec![0f32; BLOCK * k];
    for rows in x.chunks(BLOCK * d) {
        let n = rows.len() / d;
        gemm::dot_rows(rows, centroids, d, &mut scores[..n * k]);
        for row in scores[..n * k].chunks_exact(k) {
            let mut best = 0;
            let mut best_s = f32::INFINITY;
            for (c, (&dot, &h)) in row.iter().zip(&half_norms).enumerate() {
                let s = h - dot;
                if s < best_s {
                    best = c;
                    best_s = s;
                }
            }
            out.push(best);
        }
    }
    out
}

/// `k` centroids (row-major, `k * d` floats) over `x` (`n >= k` vectors of
/// `d` floats), deterministic in `seed`.
pub(super) fn train(x: &[f32], d: usize, k: usize, seed: u64) -> Vec<f32> {
    let n = x.len() / d;
    let mut rng = Rng(seed);
    let mut order: Vec<usize> = (0..n).collect();
    for i in 0..k {
        let j = i + rng.below(n - i);
        order.swap(i, j);
    }
    let mut centroids = Vec::with_capacity(k * d);
    for &i in &order[..k] {
        centroids.extend_from_slice(&x[i * d..(i + 1) * d]);
    }
    for _ in 0..ITERATIONS {
        let assigned = assign(x, d, &centroids);
        let mut sums = vec![0f64; k * d];
        let mut counts = vec![0usize; k];
        for (v, &c) in x.chunks_exact(d).zip(&assigned) {
            counts[c] += 1;
            for (s, &x) in sums[c * d..(c + 1) * d].iter_mut().zip(v) {
                *s += f64::from(x);
            }
        }
        for c in 0..k {
            if counts[c] == 0 {
                continue;
            }
            let inv = 1.0 / counts[c] as f64;
            for i in 0..d {
                centroids[c * d + i] = (sums[c * d + i] * inv) as f32;
            }
        }
        for c in 0..k {
            if counts[c] != 0 {
                continue;
            }
            // Split the first populated cluster at or after a random one: the
            // empty centroid becomes a copy nudged one way, the donor the
            // other, so the pair separates next iteration.
            let mut donor = rng.below(k);
            while counts[donor] < 2 {
                donor = (donor + 1) % k;
            }
            let src: Vec<f32> = centroids[donor * d..(donor + 1) * d].to_vec();
            for (i, &v) in src.iter().enumerate() {
                let f = if i % 2 == 0 {
                    1.0 / 1024.0
                } else {
                    -1.0 / 1024.0
                };
                centroids[c * d + i] = v * (1.0 + f);
                centroids[donor * d + i] = v * (1.0 - f);
            }
            counts[donor] /= 2;
            counts[c] = counts[donor];
        }
    }
    centroids
}
