//! The single vector a multi-vector folds into for candidate generation:
//! the mean of its tokens, normalized to unit length. Indexed in a
//! `knn_vector` field it finds, with BM25, the candidates `MaxSim` reranks
//! (measured on BEIR SciFact and NFCorpus: BM25 plus pooled-mean candidates,
//! 100 to 300 of them reranked, match exact `MaxSim` over the whole corpus).

/// The unit-length mean of `tokens` (token-major, `dim` components each);
/// all zeros when the mean is zero or there are no tokens.
///
/// # Panics
/// When `dim` is zero.
pub fn pooled_mean(tokens: &[f32], dim: usize) -> Vec<f32> {
    assert!(dim > 0, "dimension must be positive");
    let mut mean = vec![0.0f64; dim];
    for t in tokens.chunks_exact(dim) {
        for (m, &x) in mean.iter_mut().zip(t) {
            *m += f64::from(x);
        }
    }
    let norm = mean.iter().map(|m| m * m).sum::<f64>().sqrt();
    if norm == 0.0 || !norm.is_finite() {
        return vec![0.0; dim];
    }
    mean.iter().map(|m| (m / norm) as f32).collect()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::arithmetic_side_effects)]
    use super::*;

    #[test]
    fn the_mean_is_normalized() {
        let v = pooled_mean(&[1.0, 0.0, 0.0, 1.0, 1.0, 1.0], 2);
        assert!(
            (v[0] - v[1]).abs() < 1e-7 && (v[0] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6
        );
        assert_eq!(pooled_mean(&[1.0, -1.0, -1.0, 1.0], 2), vec![0.0, 0.0]);
        assert_eq!(pooled_mean(&[], 3), vec![0.0; 3]);
    }
}
