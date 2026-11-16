//! `StableTflSimilarity` (Lucene 10.6, `branch_10_6`, #16243): BM25's shape
//! with IDF replaced by a term rarity estimated from the term's length and
//! the document's length alone, so the collection and term statistics a
//! scorer is handed do not change its scores.
//!
//! ```text
//! tf = freq / (freq + k1)
//! p  = 1 - (1 - m * 2 ^ (-c * |t|)) ^ |d|
//! tr = log(1 + (1 - p + 0.05) / (p + 0.05))
//! score = boost * tr * tf
//! ```
//!
//! Ported operand for operand: `tr` in `double` from `float` inputs and
//! narrowed once, the per-norm-byte cache summed in `float`, and the score
//! as `weight - weight / (1 + freq / k1)`. The norm is `Similarity`'s
//! default (overlaps discounted).
//!
//! Differs: Java's `TermStatistics` always carries its term; this port's
//! may not ([`TermStatistics::term`] is `None` for statistics a caller built
//! without one), and such a term is scored at the length Java gives a term
//! that is not valid UTF-8 (5). The bulk scorer is `SimScorer`'s default,
//! which Java's `asBulkSimScorer` is written to equal bit for bit.

use std::fmt;
use std::sync::Arc;

use lucene_util::small_float;

use super::{
    length_table, CollectionStatistics, NormSimilarity, SimScorer, Similarity, TermStatistics,
};
use crate::explain::{java_float, Explanation};

/// `StableTflSimilarity`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StableTflSimilarity {
    k1: f32,
    c: f32,
}

impl StableTflSimilarity {
    /// `DEFAULT_K1`: BM25's default `k1`.
    pub const DEFAULT_K1: f32 = 1.2;
    /// `DEFAULT_C`: the decay fitted to English term frequencies.
    pub const DEFAULT_C: f32 = 0.917;
    /// `M`: the base term-match probability. Not adjustable.
    const M: f32 = 0.00781;
    /// `NON_UTF_8_DEFAULT_TERM_LENGTH`: about the average English word.
    const NON_UTF_8_DEFAULT_TERM_LENGTH: f32 = 5.0;

    /// `StableTflSimilarity(k1, c)`: both must be finite and non-negative.
    ///
    /// # Errors
    /// Java's `IllegalArgumentException` message when either is not.
    pub fn new(k1: f32, c: f32) -> Result<Self, String> {
        if !k1.is_finite() || k1 < 0.0 {
            return Err(format!(
                "illegal k1 value: {}, must be a non-negative finite value",
                java_float(k1)
            ));
        }
        if !c.is_finite() || c < 0.0 {
            return Err(format!(
                "illegal c value: {}, must be a non-negative finite value",
                java_float(c)
            ));
        }
        Ok(Self { k1, c })
    }

    /// `getK1()`.
    pub fn k1(&self) -> f32 {
        self.k1
    }

    /// `getC()`.
    pub fn c(&self) -> f32 {
        self.c
    }

    /// `getTermLength(term)`: the term's code points, Java's
    /// `UnicodeUtil.codePointCount`, or the default length when that throws
    /// (an invalid lead byte, or a sequence cut short).
    fn term_length(term: Option<&[u8]>) -> f32 {
        term.and_then(code_point_count)
            .map_or(Self::NON_UTF_8_DEFAULT_TERM_LENGTH, |n| n as f32)
    }
}

impl Default for StableTflSimilarity {
    /// `StableTflSimilarity()`: `k1 = 1.2`, `c = 0.917`.
    fn default() -> Self {
        Self {
            k1: Self::DEFAULT_K1,
            c: Self::DEFAULT_C,
        }
    }
}

impl fmt::Display for StableTflSimilarity {
    /// `toString()`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "StableTflSimilarity(k1={}, c={})",
            java_float(self.k1),
            java_float(self.c)
        )
    }
}

/// `UnicodeUtil.codePointCount(BytesRef)`: `None` where Java throws.
fn code_point_count(utf8: &[u8]) -> Option<usize> {
    let mut pos = 0usize;
    let mut count = 0usize;
    while pos < utf8.len() {
        let step = match utf8[pos] {
            0x00..=0x7f => 1,
            0xc0..=0xdf => 2,
            0xe0..=0xef => 3,
            0xf0..=0xf7 => 4,
            // continuation bytes as a lead, and 5- and 6-byte sequences
            _ => return None,
        };
        pos += step;
        count += 1;
    }
    (pos <= utf8.len()).then_some(count)
}

/// `p`: the probability a term of `term_length` code points occurs in a
/// document of `doc_length` terms, `1 - (1 - m * 2 ^ (-c * tl)) ^ dl`.
fn probability(c: f32, term_length: f32, doc_length: f32) -> f64 {
    let decay = 2f64.powf(f64::from(-c * term_length));
    1.0 - (1.0 - f64::from(StableTflSimilarity::M) * decay).powf(f64::from(doc_length))
}

/// `log(1 + (1 - p + 0.05) / (p + 0.05))`, narrowed to `float`.
fn rarity(p: f64) -> f32 {
    (1.0 + (1.0 - p + 0.05) / (p + 0.05)).ln() as f32
}

impl NormSimilarity for StableTflSimilarity {}

impl Similarity for StableTflSimilarity {
    fn simple_name(&self) -> &str {
        "StableTflSimilarity"
    }

    /// `scorer(boost, fieldStats, termStats...)`: the summed rarity of every
    /// term at each norm byte's length, times `boost`.
    fn scorer(
        &self,
        _field: &str,
        boost: f32,
        _collection: &CollectionStatistics,
        terms: &[TermStatistics],
    ) -> Arc<dyn SimScorer> {
        let lengths = length_table();
        let mut cache = [0f32; 256];
        let mut term_lengths = Vec::with_capacity(terms.len());
        for t in terms {
            let term_length = Self::term_length(t.term.as_deref());
            term_lengths.push(term_length);
            for (slot, &doc_length) in cache.iter_mut().zip(&lengths) {
                *slot += rarity(probability(self.c, term_length, doc_length));
            }
        }
        for slot in &mut cache {
            *slot *= boost;
        }
        Arc::new(TflScorer {
            boost,
            k1: self.k1,
            c: self.c,
            term_lengths,
            cache,
        })
    }

    fn shared(&self) -> Option<Arc<dyn Similarity>> {
        Some(Arc::new(*self))
    }
}

/// `StableTflSimilarity.TflScorer`.
struct TflScorer {
    boost: f32,
    k1: f32,
    c: f32,
    /// Each query term's length in code points.
    term_lengths: Vec<f32>,
    /// `boost` times the summed rarity, per norm byte.
    cache: [f32; 256],
}

impl TflScorer {
    /// `explainTR(float termLength, long norm)`.
    fn explain_term_rarity(&self, term_length: f32, norm: i64) -> Explanation {
        let doc_length = small_float::byte4_to_int(norm as u8) as f32;
        let p = probability(self.c, term_length, doc_length);
        let factors = vec![
            Explanation::match_(
                StableTflSimilarity::M,
                "m, multiplicative constant to term match probability",
            ),
            Explanation::match_(self.c, "c, decaying constant for term length"),
            Explanation::match_(term_length, "tl, term length"),
            Explanation::match_(doc_length, "dl, document length"),
        ];
        let probability = Explanation::match_double(
            p,
            "p, probability that the term appears in the doc, \
             computed as 1 - (1 - m * 2 ^ (-c * tl)) ^ dl from:",
        )
        .with_details(factors);
        Explanation::match_(
            rarity(p),
            "tr, term rarity, computed as log(1 + (1 - p + 0.05) / (p + 0.05)) from:",
        )
        .with_details(vec![probability])
    }

    /// `explainTR(float[] termLengths, long norm)`: one term's, or every
    /// term's summed as a `double`.
    fn explain_rarities(&self, norm: i64) -> Explanation {
        if let [one] = self.term_lengths.as_slice() {
            return self.explain_term_rarity(*one, norm);
        }
        let details: Vec<Explanation> = self
            .term_lengths
            .iter()
            .map(|&tl| self.explain_term_rarity(tl, norm))
            .collect();
        let tr = details.iter().map(|d| f64::from(d.value)).sum::<f64>();
        Explanation::match_double(tr, "tr, term rarity, computed as the sum of:")
            .with_details(details)
    }
}

impl SimScorer for TflScorer {
    /// `weight - weight / (1 + freq / k1)`: `freq / (freq + k1)` rewritten
    /// so it stays monotonic in `freq` without widening to `double`.
    fn score(&self, freq: f32, norm: i64) -> f32 {
        let weight = self.cache[norm as u8 as usize];
        weight - weight / (1.0 + freq / self.k1)
    }

    fn explain(&self, freq: Explanation, norm: i64) -> Explanation {
        let f = freq.value;
        let label = freq.value_string();
        let tf = Explanation::match_(
            1.0 - 1.0 / (1.0 + f / self.k1),
            "tf, computed as freq / (freq + k1) from:",
        )
        .with_details(vec![
            freq,
            Explanation::match_(self.k1, "k1, term saturation parameter"),
        ]);
        let weight = self.cache[norm as u8 as usize];
        Explanation::match_(
            weight - weight / (1.0 + f / self.k1),
            format!("score(freq={label}), computed as boost * tr * tf from:"),
        )
        .with_details(vec![
            Explanation::match_(self.boost, "boost"),
            tf,
            self.explain_rarities(norm),
        ])
    }
}

/// Ports `TestStableTflSimilarity` (`branch_10_6`, #16243) with its exact
/// inputs and expected values; the randomized `BaseSimilarityTestCase` half
/// runs from `lucene-test-framework`'s similarity tests.
#[cfg(test)]
mod tests;
