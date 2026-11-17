//! The two distances of a Faiss index over float vectors.

use crate::format::Metric;

/// `a * b + c`, fused where the build targets FMA (as Faiss's AVX2 kernels).
#[inline(always)]
fn fma(a: f32, b: f32, c: f32) -> f32 {
    #[cfg(any(target_feature = "fma", target_arch = "aarch64"))]
    {
        a.mul_add(b, c)
    }
    #[cfg(not(any(target_feature = "fma", target_arch = "aarch64")))]
    {
        a * b + c
    }
}

/// Squared Euclidean distance (`fvec_L2sqr`).
pub(crate) fn l2sqr(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for i in 0..8 {
            let d = x[i] - y[i];
            acc[i] = fma(d, d, acc[i]);
        }
    }
    let mut sum = acc.iter().sum::<f32>();
    for (x, y) in ra.iter().zip(rb) {
        let d = x - y;
        sum += d * d;
    }
    sum
}

/// Inner product (`fvec_inner_product`).
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for i in 0..8 {
            acc[i] = fma(x[i], y[i], acc[i]);
        }
    }
    let mut sum = acc.iter().sum::<f32>();
    for (x, y) in ra.iter().zip(rb) {
        sum += x * y;
    }
    sum
}

/// The distance of `metric`: squared L2, or the inner product.
pub(crate) fn distance(metric: Metric, a: &[f32], b: &[f32]) -> f32 {
    match metric {
        Metric::L2 => l2sqr(a, b),
        Metric::InnerProduct => dot(a, b),
    }
}

/// `distance(metric, a[i], b)` for four vectors at once: `b` is loaded once
/// per lane chunk and shared, and every result equals the one-vector call.
pub(crate) fn distance4(metric: Metric, a: [&[f32]; 4], b: &[f32]) -> [f32; 4] {
    let mut acc = [[0f32; 8]; 4];
    let cb = b.chunks_exact(8);
    let rb = cb.remainder();
    let mut chunks = [
        a[0].chunks_exact(8),
        a[1].chunks_exact(8),
        a[2].chunks_exact(8),
        a[3].chunks_exact(8),
    ];
    for y in cb {
        for (r, ch) in chunks.iter_mut().enumerate() {
            let Some(x) = ch.next() else { continue };
            for i in 0..8 {
                acc[r][i] = match metric {
                    Metric::L2 => {
                        let d = x[i] - y[i];
                        fma(d, d, acc[r][i])
                    }
                    Metric::InnerProduct => fma(x[i], y[i], acc[r][i]),
                };
            }
        }
    }
    let mut out = [0f32; 4];
    for r in 0..4 {
        let mut sum = acc[r].iter().sum::<f32>();
        let tail = &a[r][a[r].len().saturating_sub(rb.len())..];
        for (x, y) in tail.iter().zip(rb) {
            sum += match metric {
                Metric::L2 => (x - y) * (x - y),
                Metric::InnerProduct => x * y,
            };
        }
        out[r] = sum;
    }
    out
}
