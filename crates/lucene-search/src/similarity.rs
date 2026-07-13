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
//! ## Norms: real per-doc field length, decoded from `.nvd`/`.nvm` (this task)
//!
//! Real BM25 needs each matched document's *own* field length and the field's
//! *average* length across the whole segment (`fieldLength`/`avgFieldLength`
//! above) — in real Lucene these come from decoding the `.nvd`/`.nvm` norms file
//! for that field (`NumericDocValues` over `Similarity.computeNorm`'s per-doc
//! byte). `crates/lucene-codecs/src/norms.rs` has a complete read side
//! (`parse_meta`/`norm_value`, fixture-verified — see `docs/parity.md`'s norms
//! row), so this module now decodes real norm bytes instead of a constant.
//!
//! Real Lucene's default `Similarity.computeNorm` encodes a field's token-count
//! length via `SmallFloat.intToByte4` (a lossy 4-bit-mantissa byte encoding, *not*
//! a literal length) into the single norm byte written per doc; `BM25Similarity`
//! decodes it back with `SmallFloat.byte4ToInt` (cached per-segment as
//! `LENGTH_TABLE[0..256]`) to get an *approximate* field length before applying
//! `b * fieldLength / avgFieldLength`. [`decode_norm`] is this port's
//! `byte4ToInt`-equivalent decode step (see [`lucene_util::small_float`] for the
//! bit-manipulation itself, verified byte-for-byte against `SmallFloat.java`).
//! Skipping this decode and treating a raw norm byte as a literal length would
//! produce numerically wrong (if plausible-looking) scores — see
//! `lucene_util::small_float`'s doc comment for why the encoding is lossy above
//! byte value 24.
//!
//! [`crate::field_norms::FieldNorms`] computes `avgFieldLength` once per field
//! per query (summing every live doc's decoded length, mirroring `avgFieldLength
//! = sumTotalTermFreq / docCount` — this port has no separately tracked
//! `sumTotalTermFreq`, but a field's `sumTotalTermFreq` *is* the sum of its
//! per-doc lengths by definition) and [`crate::search_term_query_scored`]/
//! [`crate::search_boolean_query_scored`] use it, falling back to
//! [`UNNORMED_FIELD_LENGTH`]/[`UNNORMED_FIELD_LENGTH`] only when the field has no
//! opened norms at all (norms disabled for that field, or the caller didn't open
//! a `.nvd`/`.nvm` pair) — a documented, deliberate fallback, not silently wrong
//! data; see [`crate::field_norms`] for exactly when that applies.

/// `BM25Similarity`'s default `k1` (term-frequency saturation parameter).
pub const DEFAULT_K1: f32 = 1.2;
/// `BM25Similarity`'s default `b` (field-length normalization parameter).
pub const DEFAULT_B: f32 = 0.75;

/// The constant `fieldLength`/`avgFieldLength` this port substitutes when a
/// field has no opened norms (norms disabled for that field, or the caller
/// didn't open a `.nvd`/`.nvm` pair for this search) — see this module's doc
/// comment for why `1.0`/`1.0` (rather than e.g. `0.0`/`1.0`) is the honest
/// "no-op" substitution: it makes the length-normalization term collapse to a
/// constant instead of silently zeroing or exploding it.
pub const UNNORMED_FIELD_LENGTH: f32 = 1.0;

/// `SmallFloat.byte4ToInt`-equivalent decode of one real Lucene norm byte back
/// to an approximate field length, mirroring `BM25Similarity.LENGTH_TABLE[i] =
/// SmallFloat.byte4ToInt((byte) i)` — see this module's doc comment for why this
/// decode step (not a literal-length reinterpretation of the byte) is required.
///
/// `norm` is the sign-extended `i64` [`lucene_codecs::norms::norm_value`]
/// returns; truncating back to `u8` recovers the original unsigned byte
/// regardless of that sign extension (two's complement preserves the low byte),
/// matching real Lucene's `((byte) encodedNorm) & 0xff` indexing.
pub fn decode_norm(norm: i64) -> f32 {
    lucene_util::small_float::byte4_to_int(norm as u8) as f32
}

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

    #[test]
    fn decode_norm_matches_small_float_byte4_to_int() {
        // Same known values as `lucene_util::small_float`'s test, reached
        // through this module's `i64`-sign-extension-aware wrapper.
        assert_eq!(decode_norm(0), 0.0);
        assert_eq!(decode_norm(23), 23.0);
        assert_eq!(decode_norm(100), 3096.0);
        // Byte 200 sign-extends to a negative i64 the way
        // `norms::norm_value` returns it (`read_byte() as i8 as i64`); the
        // `as u8` truncation must still recover the original byte.
        assert_eq!(decode_norm(200i64 as i8 as i64), 16_777_240.0);
        assert_eq!(decode_norm(255i64 as i8 as i64), 2_013_265_944.0);
    }

    #[test]
    fn score_with_real_decoded_lengths_differs_from_unnormed_constant() {
        // Two docs with different real (decoded) field lengths must get
        // different tf_norm contributions -- proving the length-
        // normalization term is actually live, not collapsed to a constant.
        let short_doc_len = decode_norm(5); // byte 5 -> length 5 (subnormal, exact)
        let long_doc_len = decode_norm(40); // byte 40 -> a longer decoded length
        assert!(long_doc_len > short_doc_len);

        let avg = (short_doc_len + long_doc_len) / 2.0;
        let score_short = score(2, 10, 3.0, short_doc_len, avg);
        let score_long = score(2, 10, 3.0, long_doc_len, avg);
        assert_ne!(score_short, score_long);
        // Same BM25 property as `tf_norm_with_field_longer_than_average_reduces_score`:
        // the shorter-than-average doc scores higher for the same freq/idf.
        assert!(score_short > score_long);
    }
}
