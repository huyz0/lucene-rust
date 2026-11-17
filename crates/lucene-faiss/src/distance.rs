//! The two distances of a Faiss index over float vectors.

use crate::format::Metric;

/// Squared Euclidean distance (`fvec_L2sqr`).
pub(crate) fn l2sqr(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for i in 0..8 {
            let d = x[i] - y[i];
            acc[i] += d * d;
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
            acc[i] += x[i] * y[i];
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
