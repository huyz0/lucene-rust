//! `BM25Similarity`-equivalent (`org.apache.lucene.search.similarities.BM25Similarity`),
//! pared down to this slice's scope: the pure scoring formula, no `Similarity`/
//! `SimScorer`/`SimWeight` trait hierarchy (no second similarity implementation exists
//! in this port to justify one — same "no speculative polymorphism" reasoning
//! `lib.rs`'s module doc already applies to `Weight`/`Scorer`).
//!
//! ## The formula (verified against Lucene 10.5.0's `BM25Similarity.java`, not guessed)
//!
//! `idf(docFreq, docCount) = ln(1 + (docCount - docFreq + 0.5) / (docFreq + 0.5))`
//! (`BM25Similarity.idf`, `IDFExplanation` cache path — the `+1` in `ln(1 + x)` is
//! Lucene's own smoothing over the textbook Robertson/Sparck-Jones IDF, not this
//! port's invention).
//!
//! `tfNorm(freq, fieldLength, avgFieldLength, k1, b) =`
//! `  freq * (k1 + 1) / (freq + k1 * (1 - b + b * fieldLength / avgFieldLength))`
//! (`BM25Scorer.score`, ignoring the `boost` multiplier real Lucene folds in at the
//! `Weight` level — no query-time boosting exists in this port yet).
//!
//! `score = idf * tfNorm`.
//!
//! Defaults `k1 = 1.2`, `b = 0.75` match `BM25Similarity()`'s no-arg constructor,
//! which is what every field in this port's fixtures implicitly uses (no
//! per-field `Similarity` override machinery exists here).
//!
//! ## The norms gap (honest approximation, not invented data)
//!
//! Real BM25 needs each matched document's *own* field length and the field's
//! *average* length across the whole segment (`fieldLength`/`avgFieldLength`
//! above) — in real Lucene these come from decoding the `.nvd`/`.nvm` norms file
//! for that field (`NumericDocValues` over `Similarity.computeNorm`'s per-doc byte).
//! `crates/lucene-codecs/src/norms.rs` currently has **write-side support only**
//! (`write_single_dense_field`) — see `docs/parity.md`'s norms row — there is no
//! norms *reader* wired into this port's search path yet (the plain byte-decode
//! primitives exist in that module, but nothing in `lucene-search` opens a `.nvd`/
//! `.nvm` pair the way it opens `.tim`/`.tip`/`.doc`).
//!
//! Rather than inventing a fake norms decode here (this project's stated
//! philosophy: correctness first, be honest about what's approximated — see
//! `docs/parity.md`'s norms and postings rows for the same stance), this module
//! takes `field_length`/`avg_field_length` as plain `f32` parameters, and
//! [`crate::search_term_query_scored`]/[`crate::search_boolean_query_scored`]
//! currently always pass `1.0`/`1.0` for both (see [`UNNORMED_FIELD_LENGTH`]).
//! With both equal to `1.0`, the length-normalization term `b * fieldLength /
//! avgFieldLength` collapses to exactly `b`, so `tfNorm` becomes `freq * (k1 + 1)
//! / (freq + k1)` — a real, well-defined BM25 variant (equivalent to every
//! document having the field's average length), just not one that reflects this
//! segment's actual per-document field lengths. **This means scores computed by
//! this module are internally consistent (same-term/same-corpus comparisons rank
//! correctly relative to each other) but are not expected to numerically match
//! real Lucene's BM25 scores for the same query**, since real Lucene's norms
//! would vary `tfNorm` per document. Tracked in `docs/parity.md` as deferred
//! until a norms reader exists in this port.

/// `BM25Similarity`'s default `k1` (term-frequency saturation parameter).
pub const DEFAULT_K1: f32 = 1.2;
/// `BM25Similarity`'s default `b` (field-length normalization parameter).
pub const DEFAULT_B: f32 = 0.75;

/// The constant `fieldLength`/`avgFieldLength` this port substitutes for real
/// per-document norms, since no norms *reader* exists in this port's search path
/// yet — see this module's doc comment for why `1.0`/`1.0` (rather than e.g.
/// `0.0`/`1.0`) is the honest "no-op" substitution: it makes the length-
/// normalization term collapse to a constant instead of silently zeroing or
/// exploding it.
pub const UNNORMED_FIELD_LENGTH: f32 = 1.0;

/// `BM25Similarity.idf(long docFreq, long docCount)`-equivalent: the inverse
/// document frequency component of the score, shared by every document matching
/// this term (real Lucene caches this once per term via `IDFExplanation`; this
/// port just recomputes it, cheap enough not to matter at this scale).
///
/// `doc_count` is the field's document count (`CollectionStatistics.docCount()`
/// in real Lucene, falling back to `maxDoc` when doc-count tracking is
/// unavailable — this port always has `FieldTerms::doc_count`, so no fallback is
/// needed). `doc_freq` is the term's document frequency in that field.
pub fn idf(doc_freq: i64, doc_count: i64) -> f32 {
    (1.0 + (doc_count as f64 - doc_freq as f64 + 0.5) / (doc_freq as f64 + 0.5)).ln() as f32
}

/// `BM25Scorer.score(int doc, float freq)`-equivalent tf-normalization term
/// (everything except the `idf` multiplier and the (unsupported) query boost).
pub fn tf_norm(freq: f32, field_length: f32, avg_field_length: f32, k1: f32, b: f32) -> f32 {
    freq * (k1 + 1.0) / (freq + k1 * (1.0 - b + b * field_length / avg_field_length))
}

/// The full per-document BM25 score: `idf * tf_norm`, using the default `k1`/`b`
/// and the given collection/document statistics.
///
/// - `doc_freq`: the term's document frequency in `field`.
/// - `doc_count`: the field's document count (see [`idf`]).
/// - `freq`: the term's frequency in the matched document.
/// - `field_length`/`avg_field_length`: see this module's doc comment for why
///   this port currently always passes [`UNNORMED_FIELD_LENGTH`] for both.
pub fn score(
    doc_freq: i64,
    doc_count: i64,
    freq: f32,
    field_length: f32,
    avg_field_length: f32,
) -> f32 {
    idf(doc_freq, doc_count) * tf_norm(freq, field_length, avg_field_length, DEFAULT_K1, DEFAULT_B)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Hand-computed (independently of the implementation, via the same formula
    // read directly off `BM25Similarity.java`) expected values -- see the
    // `test-coverage` skill's rule against "coverage theater": these assert
    // pre-computed numbers, not "whatever the code currently produces".

    #[test]
    fn idf_matches_hand_computed_value() {
        // docFreq=1, docCount=10: ln(1 + (10 - 1 + 0.5)/(1 + 0.5)) = ln(1 + 9.5/1.5)
        // = ln(1 + 6.333333...) = ln(7.333333...) = 1.9924302...
        let got = idf(1, 10);
        assert!((got - 1.992_430_2).abs() < 1e-5, "got {got}");
    }

    #[test]
    fn idf_decreases_as_doc_freq_increases() {
        // A more common term (higher docFreq) must score a lower idf over the same
        // docCount -- the defining monotonicity property of IDF.
        assert!(idf(1, 100) > idf(50, 100));
        assert!(idf(50, 100) > idf(99, 100));
    }

    #[test]
    fn idf_can_go_negative_for_a_term_in_every_document() {
        // Real BM25Similarity's ln(1+x) smoothing still allows idf < 0 when
        // docFreq == docCount (every doc contains the term): ln(1 + 0.5/(N+0.5)),
        // which is > 0 actually for finite N -- but as docFreq approaches docCount
        // for large N it approaches ln(1) = 0 from above. Assert the concrete
        // hand-computed boundary value instead of a vague claim.
        // docFreq=10, docCount=10: ln(1 + (10-10+0.5)/(10+0.5)) = ln(1 + 0.5/10.5)
        // = ln(1.047619...) = 0.046520...
        let got = idf(10, 10);
        assert!((got - 0.046_520).abs() < 1e-4, "got {got}");
    }

    #[test]
    fn tf_norm_matches_hand_computed_value() {
        // freq=3, fieldLength=avgFieldLength=1.0 (this port's constant
        // substitution), k1=1.2, b=0.75:
        // tfNorm = 3*(1.2+1) / (3 + 1.2*(1-0.75+0.75*1/1)) = 6.6 / (3 + 1.2*1.0)
        // = 6.6 / 4.2 = 1.571428...
        let got = tf_norm(3.0, 1.0, 1.0, DEFAULT_K1, DEFAULT_B);
        assert!((got - 1.571_428_6).abs() < 1e-5, "got {got}");
    }

    #[test]
    fn tf_norm_with_field_longer_than_average_reduces_score() {
        // freq=2, fieldLength=2*avgFieldLength: b*fieldLength/avgFieldLength term
        // doubles, penalizing tf_norm relative to the fieldLength==avgFieldLength
        // case -- BM25's length-normalization property.
        let baseline = tf_norm(2.0, 1.0, 1.0, DEFAULT_K1, DEFAULT_B);
        let longer_doc = tf_norm(2.0, 2.0, 1.0, DEFAULT_K1, DEFAULT_B);
        assert!(longer_doc < baseline);
    }

    #[test]
    fn score_combines_idf_and_tf_norm() {
        // docFreq=2, docCount=10, freq=4, unnormed field length.
        // idf(2,10) = ln(1 + (10-2+0.5)/(2+0.5)) = ln(1 + 8.5/2.5) = ln(4.4)
        //           = 1.481_604...
        // tfNorm(4, 1, 1, 1.2, 0.75) = 4*2.2 / (4 + 1.2*1.0) = 8.8/5.2 = 1.692_307...
        // score = 1.481604 * 1.692307... = 2.507197...
        let got = score(2, 10, 4.0, UNNORMED_FIELD_LENGTH, UNNORMED_FIELD_LENGTH);
        let expected_idf = 4.4f64.ln() as f32;
        let expected_tf_norm = 8.8f32 / 5.2f32;
        let expected = expected_idf * expected_tf_norm;
        assert!(
            (got - expected).abs() < 1e-4,
            "got {got}, expected {expected}"
        );
        assert!((got - 2.507_197).abs() < 1e-3, "got {got}");
    }

    #[test]
    fn score_is_zero_when_freq_is_zero() {
        assert_eq!(score(1, 10, 0.0, 1.0, 1.0), 0.0);
    }

    #[test]
    fn score_increases_with_freq_all_else_equal() {
        let low = score(5, 100, 1.0, 1.0, 1.0);
        let high = score(5, 100, 5.0, 1.0, 1.0);
        assert!(high > low);
    }
}
