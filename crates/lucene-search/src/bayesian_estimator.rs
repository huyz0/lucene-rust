//! `BayesianScoreEstimator`: `BayesianScoreQuery`'s `alpha`, `beta` and base
//! rate estimated from the index itself, by sampling pseudo-queries from a
//! field's vocabulary and reading the BM25 score distribution they produce.
//!
//! The sampling is `java.util.Random`'s, seeded as Java seeds it, so the same
//! seed picks the same terms and the estimate matches Lucene's to the bit.

use std::collections::{BTreeSet, HashMap};

use crate::field_norms::FieldNorms;
use crate::multi_segment::{search_boolean_query_multi_segment_maxscore, OpenSegment};
use crate::query::{BooleanQuery, Clause, TermQuery};
use crate::{Error, Result};

/// `BayesianScoreEstimator.Parameters`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Parameters {
    pub alpha: f32,
    pub beta: f32,
    pub base_rate: f32,
}

/// What an empty index or field estimates: `new Parameters(1.0f, 0.0f, 0.01f)`.
const FALLBACK: Parameters = Parameters {
    alpha: 1.0,
    beta: 0.0,
    base_rate: 0.01,
};

const PERCENTILE_THRESHOLD: f64 = 0.95;
const BASE_RATE_MIN: f32 = 1e-6;
const BASE_RATE_MAX: f32 = 0.5;
/// `estimate(searcher, field)`'s defaults.
pub const DEFAULT_N_SAMPLES: usize = 50;
pub const DEFAULT_TOKENS_PER_QUERY: usize = 5;
pub const DEFAULT_SEED: i64 = 42;

/// `java.util.Random`: the 48-bit linear congruential generator.
#[derive(Debug, Clone)]
pub struct JavaRandom {
    seed: u64,
}

impl JavaRandom {
    const MULTIPLIER: u64 = 0x5_DEEC_E66D;
    const ADDEND: u64 = 0xB;
    const MASK: u64 = (1 << 48) - 1;

    /// `new Random(seed)`: the seed scrambled with the multiplier.
    pub fn new(seed: i64) -> Self {
        Self {
            seed: (seed as u64 ^ Self::MULTIPLIER) & Self::MASK,
        }
    }

    /// `next(bits)`.
    fn next(&mut self, bits: u32) -> i32 {
        self.seed = self
            .seed
            .wrapping_mul(Self::MULTIPLIER)
            .wrapping_add(Self::ADDEND)
            & Self::MASK;
        (self.seed >> (48 - bits)) as i32
    }

    /// `nextLong()`: `((long) next(32) << 32) + next(32)`.
    pub fn next_long(&mut self) -> i64 {
        let hi = i64::from(self.next(32));
        let lo = i64::from(self.next(32));
        (hi << 32).wrapping_add(lo)
    }
}

/// `BayesianScoreEstimator.nextLong(rng, bound)`: a uniform value in
/// `[0, bound)`, rejecting the draws that would bias it.
fn next_long_bounded(rng: &mut JavaRandom, bound: i64) -> i64 {
    loop {
        let bits = ((rng.next_long() as u64) >> 1) as i64;
        let value = bits % bound;
        // `bits - value + (bound - 1) < 0`: the overflow check.
        if bits.wrapping_sub(value).wrapping_add(bound - 1) >= 0 {
            return value;
        }
    }
}

/// `sampleVocabularyTerms`: reservoir-sample `sample_size` terms of `field`
/// from the reader's merged, sorted vocabulary (`MultiTerms`).
fn sample_vocabulary_terms(
    segments: &[OpenSegment<'_>],
    field: &str,
    sample_size: usize,
    rng: &mut JavaRandom,
) -> Result<Vec<Vec<u8>>> {
    let mut vocabulary = BTreeSet::new();
    for seg in segments {
        let Some(ft) = seg.fields.field(field) else {
            continue;
        };
        let mut it = ft.iter();
        while let Some(t) = it.try_next_term()? {
            vocabulary.insert(t.to_vec());
        }
    }
    let mut reservoir: Vec<Vec<u8>> = Vec::with_capacity(sample_size);
    let mut seen = 0i64;
    for term in vocabulary {
        seen += 1;
        if reservoir.len() < sample_size {
            reservoir.push(term);
        } else {
            let replacement = next_long_bounded(rng, seen);
            if let Ok(i) = usize::try_from(replacement) {
                if i < sample_size {
                    reservoir[i] = term;
                }
            }
        }
    }
    Ok(reservoir)
}

/// `BayesianScoreEstimator.estimate(searcher, field, nSamples,
/// tokensPerQuery, seed)`: `beta` is the median BM25 score of the sampled
/// pseudo-queries' hits, `alpha` the inverse of their standard deviation,
/// and the base rate the mean share of documents at or above each query's
/// 95th percentile, clamped to `[1e-6, 0.5]`.
pub fn estimate(
    segments: &[OpenSegment<'_>],
    norms: &[Option<&HashMap<String, FieldNorms<'_>>>],
    field: &str,
    n_samples: usize,
    tokens_per_query: usize,
    seed: i64,
) -> Result<Parameters> {
    if n_samples == 0 {
        return Err(Error::InvalidQuery(format!(
            "nSamples must be positive, got {n_samples}"
        )));
    }
    if tokens_per_query == 0 {
        return Err(Error::InvalidQuery(format!(
            "tokensPerQuery must be positive, got {tokens_per_query}"
        )));
    }
    let max_doc: i64 = segments
        .iter()
        .map(|s| i64::from(s.max_doc.unwrap_or(0)))
        .sum();
    if max_doc == 0 {
        return Ok(FALLBACK);
    }
    let sample_size = n_samples
        .checked_mul(tokens_per_query)
        .filter(|&n| n <= i32::MAX as usize)
        .ok_or_else(|| Error::InvalidQuery("integer overflow".into()))?;
    let mut rng = JavaRandom::new(seed);
    let sampled = sample_vocabulary_terms(segments, field, sample_size, &mut rng)?;
    if sampled.is_empty() {
        return Ok(FALLBACK);
    }
    let top_n = usize::try_from(max_doc.min(10_000)).unwrap_or(10_000);
    let mut all_scores: Vec<f32> = Vec::new();
    let mut base_rate_fractions: Vec<f32> = Vec::new();
    for chunk in sampled.chunks(tokens_per_query) {
        let query = BooleanQuery::new().with_should(
            chunk
                .iter()
                .map(|t| Clause::Term(TermQuery::new(field, t.clone())))
                .collect::<Vec<_>>(),
        );
        let hits = search_boolean_query_multi_segment_maxscore(segments, &query, norms, top_n)?;
        let scores: Vec<f32> = hits.iter().map(|h| h.score).collect();
        if scores.is_empty() {
            continue;
        }
        let mut sorted = scores.clone();
        sorted.sort_by(f32::total_cmp);
        let p_idx = ((sorted.len() as f64 * PERCENTILE_THRESHOLD) as usize).min(sorted.len() - 1);
        let threshold = sorted[p_idx];
        let high = scores.iter().filter(|&&s| s >= threshold).count();
        base_rate_fractions.push(high as f32 / max_doc as f32);
        all_scores.extend_from_slice(&scores);
    }
    if all_scores.is_empty() {
        return Ok(FALLBACK);
    }
    all_scores.sort_by(f32::total_cmp);
    let beta = all_scores[all_scores.len() / 2];
    let n = all_scores.len() as f64;
    let mean = all_scores.iter().map(|&s| f64::from(s)).sum::<f64>() / n;
    let variance = all_scores
        .iter()
        .map(|&s| {
            let d = f64::from(s) - mean;
            d * d
        })
        .sum::<f64>()
        / n;
    let std = variance.sqrt();
    let alpha = if std > 0.0 { (1.0 / std) as f32 } else { 1.0 };
    let mut base_rate = 0.0f32;
    for f in &base_rate_fractions {
        base_rate += f;
    }
    base_rate /= base_rate_fractions.len() as f32;
    let base_rate = base_rate.clamp(BASE_RATE_MIN, BASE_RATE_MAX);
    Ok(Parameters {
        alpha,
        beta,
        base_rate,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn java_random_matches_its_reference_sequence() {
        // `new Random(42).nextLong()` in Java.
        let mut r = JavaRandom::new(42);
        assert_eq!(r.next_long(), -5_025_562_857_975_149_833);
        assert_eq!(r.next_long(), -5_843_495_416_241_995_736);
        let mut r = JavaRandom::new(0);
        assert_eq!(r.next_long(), -4_962_768_465_676_381_896);
    }

    #[test]
    fn bounded_draws_stay_below_the_bound() {
        let mut r = JavaRandom::new(7);
        for bound in [1i64, 2, 3, 10, 1 << 40] {
            for _ in 0..50 {
                let v = next_long_bounded(&mut r, bound);
                assert!((0..bound).contains(&v));
            }
        }
    }

    #[test]
    fn empty_input_and_bad_arguments() {
        assert!(estimate(&[], &[], "f", 0, 5, 1).is_err());
        assert!(estimate(&[], &[], "f", 5, 0, 1).is_err());
        assert_eq!(estimate(&[], &[], "f", 5, 5, 1).unwrap(), FALLBACK);
    }
}
