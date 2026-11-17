//! The two distances of a Faiss index over float vectors.

use crate::format::Metric;

/// Squared Euclidean distance (`fvec_L2sqr`), by the shared AVX2/NEON kernel.
#[inline]
pub(crate) fn l2sqr(a: &[f32], b: &[f32]) -> f32 {
    lucene_util::simd::square_distance_f32(a, b)
}

/// Inner product (`fvec_inner_product`), by the shared AVX2/NEON kernel.
#[inline]
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    lucene_util::simd::dot_f32(a, b)
}

/// The distance of `metric`: squared L2, or the inner product.
pub(crate) fn distance(metric: Metric, a: &[f32], b: &[f32]) -> f32 {
    match metric {
        Metric::L2 => l2sqr(a, b),
        Metric::InnerProduct => dot(a, b),
    }
}
