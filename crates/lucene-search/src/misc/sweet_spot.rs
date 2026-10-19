//! `SweetSpotSimilarity`: `ClassicSimilarity` with a length norm that is flat
//! over the good lengths, and a baseline or hyperbolic `tf`.

use std::fmt;
use std::sync::Arc;

use lucene_index::similarity::NormSimilarity;
use lucene_util::geo::{java_double_string, java_float_string};
use lucene_util::{small_float, strict_math};

use crate::similarities::{
    summed_idf, ClassicSimilarity, CollectionStatistics, SimScorer, Similarity, TermStatistics,
    TfIdfSimilarity,
};

/// `SweetSpotSimilarity`. The defaults are Java's: a flat length norm at one
/// term, the baseline `tf` with no floor, the hyperbolic `tf` from 0 to 2.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SweetSpotSimilarity {
    discount_overlaps: bool,
    ln_min: i32,
    ln_max: i32,
    ln_steep: f32,
    tf_base: f32,
    tf_min: f32,
    tf_hyper_min: f32,
    tf_hyper_max: f32,
    tf_hyper_base: f64,
    tf_hyper_xoffset: f32,
}

impl Default for SweetSpotSimilarity {
    /// `new SweetSpotSimilarity()`: overlaps discounted.
    fn default() -> Self {
        Self::new(true)
    }
}

impl SweetSpotSimilarity {
    /// `new SweetSpotSimilarity(discountOverlaps)`.
    pub fn new(discount_overlaps: bool) -> Self {
        Self {
            discount_overlaps,
            ln_min: 1,
            ln_max: 1,
            ln_steep: 0.5,
            tf_base: 0.0,
            tf_min: 0.0,
            tf_hyper_min: 0.0,
            tf_hyper_max: 2.0,
            tf_hyper_base: 1.3,
            tf_hyper_xoffset: 10.0,
        }
    }

    /// `setBaselineTfFactors(base, min)`.
    pub fn set_baseline_tf_factors(&mut self, base: f32, min: f32) {
        self.tf_min = min;
        self.tf_base = base;
    }

    /// `setHyperbolicTfFactors(min, max, base, xoffset)`.
    pub fn set_hyperbolic_tf_factors(&mut self, min: f32, max: f32, base: f64, xoffset: f32) {
        self.tf_hyper_min = min;
        self.tf_hyper_max = max;
        self.tf_hyper_base = base;
        self.tf_hyper_xoffset = xoffset;
    }

    /// `setLengthNormFactors(min, max, steepness)`.
    pub fn set_length_norm_factors(&mut self, min: i32, max: i32, steepness: f32) {
        self.ln_min = min;
        self.ln_max = max;
        self.ln_steep = steepness;
    }

    /// `lengthNorm(numTerms)`: `1 / sqrt(steepness * (|n - min| + |n - max| -
    /// (max - min)) + 1)`, which is 1 anywhere between `min` and `max`.
    pub fn length_norm(&self, num_terms: i32) -> f32 {
        // Java's `int` arithmetic wraps.
        let off_sweet_spot = num_terms
            .wrapping_sub(self.ln_min)
            .wrapping_abs()
            .wrapping_add(num_terms.wrapping_sub(self.ln_max).wrapping_abs())
            .wrapping_sub(self.ln_max.wrapping_sub(self.ln_min));
        let v = self.ln_steep * off_sweet_spot as f32 + 1.0;
        (1.0f32 as f64 / f64::from(v).sqrt()) as f32
    }

    /// `baselineTf(freq)`: `sqrt(freq + base^2 - min)`, `base` at or below
    /// `min`, and 0 for no occurrences.
    pub fn baseline_tf(&self, freq: f32) -> f32 {
        if freq == 0.0 {
            return 0.0;
        }
        if freq <= self.tf_min {
            self.tf_base
        } else {
            f64::from(freq + self.tf_base * self.tf_base - self.tf_min).sqrt() as f32
        }
    }

    /// `hyperbolicTf(freq)`: a `tanh` curve from `min` to `max` centred on
    /// `xoffset`, and `max` where it overflows to `NaN`.
    pub fn hyperbolic_tf(&self, freq: f32) -> f32 {
        if freq == 0.0 {
            return 0.0;
        }
        let (min, max) = (self.tf_hyper_min, self.tf_hyper_max);
        let base = self.tf_hyper_base;
        let x = f64::from(freq - self.tf_hyper_xoffset);
        let (up, down) = (strict_math::pow(base, x), strict_math::pow(base, -x));
        let result =
            min + (f64::from((max - min) / 2.0) * ((up - down) / (up + down) + 1.0)) as f32;
        if result.is_nan() {
            max
        } else {
            result
        }
    }

    /// `TFIDFSimilarity.scorer`'s `normTable`, from this `lengthNorm`.
    fn norm_table(&self) -> [f32; 256] {
        let mut norm_table = [0.0f32; 256];
        for (i, slot) in norm_table.iter_mut().enumerate().skip(1) {
            *slot = self.length_norm(small_float::byte4_to_int(i as u8) as i32);
        }
        norm_table[0] = 1.0 / norm_table[255];
        norm_table
    }
}

impl fmt::Display for SweetSpotSimilarity {
    /// `toString()`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SweetSpotSimilarity(ln_min={}, ln_max={}, ln_steep={}, tf_base={}, tf_min={}, \
             tf_hyper_min={}, tf_hyper_max={}, tf_hyper_base={}, tf_hyper_xoffset={})",
            self.ln_min,
            self.ln_max,
            java_float_string(self.ln_steep),
            java_float_string(self.tf_base),
            java_float_string(self.tf_min),
            java_float_string(self.tf_hyper_min),
            java_float_string(self.tf_hyper_max),
            java_double_string(self.tf_hyper_base),
            java_float_string(self.tf_hyper_xoffset),
        )
    }
}

struct SweetSpotScorer {
    query_weight: f32,
    sim: SweetSpotSimilarity,
    norm_table: [f32; 256],
}

impl SimScorer for SweetSpotScorer {
    fn score(&self, freq: f32, norm: i64) -> f32 {
        let raw = self.sim.baseline_tf(freq) * self.query_weight;
        raw * self.norm_table[(norm & 0xFF) as usize]
    }
}

impl NormSimilarity for SweetSpotSimilarity {
    fn discount_overlaps(&self) -> bool {
        self.discount_overlaps
    }
}

impl Similarity for SweetSpotSimilarity {
    fn simple_name(&self) -> &str {
        "SweetSpotSimilarity"
    }

    fn scorer(
        &self,
        _field: &str,
        boost: f32,
        collection: &CollectionStatistics,
        terms: &[TermStatistics],
    ) -> Arc<dyn SimScorer> {
        let idf = summed_idf(
            |t| ClassicSimilarity::idf(t.doc_freq, collection.doc_count),
            terms,
        );
        Arc::new(SweetSpotScorer {
            query_weight: boost * idf,
            sim: *self,
            norm_table: self.norm_table(),
        })
    }

    fn as_tfidf(&self, _field: &str) -> Option<&dyn TfIdfSimilarity> {
        Some(self)
    }

    fn shared(&self) -> Option<Arc<dyn Similarity>> {
        Some(Arc::new(*self))
    }
}

impl TfIdfSimilarity for SweetSpotSimilarity {
    /// `tf(freq)`: the baseline, as Java overrides it.
    fn tf(&self, freq: f32) -> f32 {
        self.baseline_tf(freq)
    }

    fn idf(&self, doc_freq: i64, doc_count: i64) -> f32 {
        ClassicSimilarity::idf(doc_freq, doc_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn length_norm_is_flat_between_min_and_max() {
        let mut s = SweetSpotSimilarity::default();
        s.set_length_norm_factors(3, 9, 0.5);
        for n in 3..=9 {
            assert_eq!(s.length_norm(n), 1.0);
        }
        assert!(s.length_norm(2) < 1.0 && s.length_norm(10) < 1.0);
        // `abs` of the most negative `int` wraps, as in Java.
        let _ = s.length_norm(i32::MIN);
    }

    #[test]
    fn baseline_tf_floors_at_base() {
        let mut s = SweetSpotSimilarity::default();
        s.set_baseline_tf_factors(1.5, 2.0);
        assert_eq!(s.baseline_tf(0.0), 0.0);
        assert_eq!(s.baseline_tf(2.0), 1.5);
        assert!(s.baseline_tf(10.0) > 1.5);
    }

    #[test]
    fn hyperbolic_tf_saturates_to_max_instead_of_nan() {
        let s = SweetSpotSimilarity::default();
        assert_eq!(s.hyperbolic_tf(0.0), 0.0);
        assert_eq!(s.hyperbolic_tf(1.0e6), 2.0);
        assert!(s.hyperbolic_tf(10.0) > 0.9 && s.hyperbolic_tf(10.0) < 1.1);
    }

    #[test]
    fn to_string_uses_java_number_layout() {
        assert_eq!(
            SweetSpotSimilarity::default().to_string(),
            "SweetSpotSimilarity(ln_min=1, ln_max=1, ln_steep=0.5, tf_base=0.0, tf_min=0.0, \
             tf_hyper_min=0.0, tf_hyper_max=2.0, tf_hyper_base=1.3, tf_hyper_xoffset=10.0)"
        );
    }
}
