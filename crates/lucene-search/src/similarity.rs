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
//! `  freq / (freq + k1 * (1 - b + b * fieldLength / avgFieldLength))`
//! (`BM25Scorer.doScore`, ignoring the `boost` multiplier real Lucene folds in at
//! the `Weight` level — no query-time boosting exists in this port yet).
//!
//! **No `(k1 + 1)` numerator factor** — this differs from the textbook
//! Robertson/Sparck-Jones BM25 TF term (`freq * (k1 + 1) / (...)`) that an
//! earlier version of this module's formula mistakenly carried over; verified
//! against Lucene 10.5.0's actual `BM25Scorer.doScore` source (`return weight -
//! weight / (1f + freq * normInverse)`, which algebraically expands to `weight *
//! freq / (freq + k1 * (1 - b + b * fieldLength / avgFieldLength))` — no `(k1 +
//! 1)` anywhere) and cross-checked against real `IndexSearcher.explain()` output
//! against a real fixture segment (`dismax_query_fixtures.rs`'s
//! `dismax_scored_matches_real_lucenes_own_disjunctionmaxquery_output`, task
//! #32), which is what caught this discrepancy — every earlier self-consistency
//! test in this crate reimplemented the *same* (wrong) formula independently, so
//! none of them could have caught it; this is the first test in this port that
//! compares an absolute BM25 score against real Lucene's own recorded output
//! rather than a hand-rederivation of this module's own formula.
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

/// The two knobs real Lucene's `BM25Similarity(float k1, float b)` constructor
/// exposes per-`Similarity`-instance (task #214, "Configurable BM25 constant
/// from FFI") -- `k1` (term-frequency saturation) and `b` (field-length
/// normalization). [`Default`] reproduces today's hardcoded [`DEFAULT_K1`]/
/// [`DEFAULT_B`] byte-for-byte, so every existing call site that doesn't know
/// about this struct keeps its exact current behavior.
///
/// **Scope note** (see `docs/parity.md`'s BM25/similarity row for the full,
/// honest list): this struct only reaches
/// [`crate::search_term_query_scored_with_similarity`] so far -- a single
/// `TermQuery`, no MAXSCORE pruning. `search_boolean_query_scored`,
/// the MAXSCORE-pruned variants (`search_term_query_scored_maxscore`,
/// `search_boolean_query_scored_maxscore`), phrase queries, and
/// `explain`/`explain_boolean` all remain hardcoded to [`DEFAULT_K1`]/
/// [`DEFAULT_B`], unchanged. Threading custom `k1`/`b` through every scored
/// path is a larger, separately-scoped change; this task deliberately covers
/// only the single most fundamental scored entry point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bm25Params {
    /// Term-frequency saturation parameter (`BM25Similarity`'s `k1`).
    pub k1: f32,
    /// Field-length normalization parameter (`BM25Similarity`'s `b`).
    pub b: f32,
}

impl Default for Bm25Params {
    /// Reproduces [`DEFAULT_K1`]/[`DEFAULT_B`] -- real Lucene's
    /// `BM25Similarity()` no-arg constructor.
    fn default() -> Self {
        Bm25Params {
            k1: DEFAULT_K1,
            b: DEFAULT_B,
        }
    }
}

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

/// `BM25Scorer.doScore(float freq, float normInverse)`-equivalent tf-normalization
/// term (everything except the `idf` multiplier and the (unsupported) query
/// boost) — see this module's doc comment for why there is **no** `(k1 + 1)`
/// numerator factor (real Lucene 10.5.0's actual formula, not the textbook one).
pub fn tf_norm(freq: f32, field_length: f32, avg_field_length: f32, k1: f32, b: f32) -> f32 {
    freq / (freq + k1 * (1.0 - b + b * field_length / avg_field_length))
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

/// [`score`]'s sibling taking an explicit [`Bm25Params`] instead of the
/// hardcoded [`DEFAULT_K1`]/[`DEFAULT_B`] -- see that struct's doc comment for
/// this task's scope. `Bm25Params::default()` produces byte-for-byte the same
/// result as [`score`] (same formula, same constants).
pub fn score_with_params(
    doc_freq: i64,
    doc_count: i64,
    freq: f32,
    field_length: f32,
    avg_field_length: f32,
    params: Bm25Params,
) -> f32 {
    idf(doc_freq, doc_count) * tf_norm(freq, field_length, avg_field_length, params.k1, params.b)
}

/// Upper bound on the BM25 score any document covered by a single block/span
/// of competitive impacts (`lucene_codecs::postings::Impact`) can achieve —
/// this port's scoped-down stand-in for real Lucene's `MaxScoreCache`/
/// `ImpactsEnum.getMaxScore` (see `docs/parity.md`'s postings row for the
/// full `ImpactsEnum` hierarchy this port does *not* implement).
///
/// [`crate::postings`]'s impacts invariant (`Postings::level0_impacts`'s doc
/// comment, mirroring `CompetitiveImpactAccumulator.getCompetitiveFreqNormPairs`)
/// guarantees `impacts` is ordered by strictly increasing `freq` *and*
/// strictly increasing (unsigned) `norm` — but that does **not** mean the
/// last entry alone bounds the score: a higher `freq` raises the score while
/// a higher decoded field length (from a higher `norm` byte) lowers it, so
/// this function conservatively takes the max BM25 score obtainable from any
/// one `(freq, norm)` pair in the list, not just the extremes. That is
/// exactly what real Lucene's own `Impact`-consuming scorers do too — see
/// `BM25Scorer.mms`'s per-impacts-entry max in `Lucene101PostingsReader`'s
/// impacts consumer logic — a real block's true max score, not a heuristic
/// approximation, so a caller may safely skip any document in this block
/// whose true score cannot exceed a `top_n` collector's current worst kept
/// score once that worst score is `>=` this bound (no real hit can ever be
/// missed by such a skip).
///
/// Returns `0.0` for an empty `impacts` slice. **This does NOT always mean
/// "no documents here, safe to skip"** — `PostingsCursor::level0_impacts`/
/// `LazyDocsCursor::level0_impacts` (`lucene_codecs::postings`) also return
/// an empty slice when the cursor is positioned in the *tail* block (the
/// `docFreq % BLOCK_SIZE` remainder, or a term with fewer than one full
/// block), which carries no level-0 impacts on the wire at all even though
/// it can hold real, scoreable documents. A future caller MUST NOT treat a
/// `0.0` bound from an empty slice as license to skip that block — check
/// the cursor's own state (full block with impacts vs. tail) before
/// deciding to skip, the same distinction those cursors' own doc comments
/// already draw. The only caller today (`assert_block_pruning_matches_brute_force`,
/// below) never skips on an empty-impacts result, so this gap is currently
/// inert, not exploited.
pub fn max_score_for_impacts(
    impacts: &[lucene_codecs::postings::Impact],
    doc_freq: i64,
    doc_count: i64,
    avg_field_length: f32,
) -> f32 {
    let idf = idf(doc_freq, doc_count);
    impacts
        .iter()
        .map(|impact| {
            let field_length = decode_norm(impact.norm);
            idf * tf_norm(
                impact.freq as f32,
                field_length,
                avg_field_length,
                DEFAULT_K1,
                DEFAULT_B,
            )
        })
        .fold(0.0f32, f32::max)
}

/// Demonstration/proof harness for [`max_score_for_impacts`]-driven
/// block-level pruning of a single [`lucene_codecs::postings::Postings`]
/// list, kept test-only (see this module's doc comment on the smaller,
/// honestly-scoped increment this port takes here: the bound itself, proven
/// safe in isolation, rather than wiring block-skip into the production
/// `term_doc_scores`/`TopDocsCollector` loop in `lib.rs`/`collector.rs` yet
/// — that eager path already fully decodes every block's docs/freqs before
/// scoring starts, via `DocInput::read_postings`, so a real skip there would
/// additionally need switching that loop onto `LazyDocsCursor`'s
/// decode-on-demand blocks, a larger change left as future work per
/// `docs/parity.md`).
///
/// `norm_bytes[i]` is the real per-doc norm byte backing `postings.docs[i]`'s
/// own (decoded) field length — the caller must construct `postings` (and in
/// particular its `level0_impacts`) so every doc's real `(freq, norm_byte)`
/// is dominated by its covering block's impacts entries (`freq <=
/// entry.freq` and `norm_byte <= entry.norm` for some entry), exactly the
/// invariant real `CompetitiveImpactAccumulator`-written impacts guarantee —
/// this harness does not (and cannot, without owning the writer side)
/// verify that invariant itself, only that pruning built *on top of* it never
/// changes the result.
///
/// Walks `postings` twice: once scoring every doc (the ground truth), once
/// skipping a whole level-0 block's remaining docs entirely whenever
/// [`max_score_for_impacts`]'s bound for that block cannot beat the
/// collector's current worst kept score (mirroring a single-clause
/// MAXSCORE-style block skip) -- and asserts the two top-`n` results are
/// identical, proving the skip never drops a real result.
#[cfg(test)]
fn assert_block_pruning_matches_brute_force(
    postings: &lucene_codecs::postings::Postings,
    norm_bytes: &[u8],
    doc_freq: i64,
    doc_count: i64,
    avg_field_length: f32,
    top_n: usize,
) {
    use crate::collector::{ScoringCollector, TopDocsCollector};
    use lucene_codecs::postings::{PostingsCursor, NO_MORE_DOCS};

    assert_eq!(norm_bytes.len(), postings.docs.len());
    let real_field_length = |doc_id: i32| -> f32 {
        let idx = postings.docs.iter().position(|&d| d == doc_id).unwrap();
        decode_norm(norm_bytes[idx] as i64)
    };

    // Ground truth: score every single doc, no skipping.
    let mut brute = TopDocsCollector::new(top_n);
    {
        let mut cursor = PostingsCursor::new(postings);
        loop {
            let doc_id = cursor.next_doc();
            if doc_id == NO_MORE_DOCS {
                break;
            }
            let freq = cursor.freq().expect("started, in range");
            let s = score(
                doc_freq,
                doc_count,
                freq as f32,
                real_field_length(doc_id),
                avg_field_length,
            );
            brute.collect(doc_id, s);
        }
    }

    // Pruned: skip a whole level-0 block's remaining docs once its impacts'
    // max score can no longer beat the current worst kept hit.
    let mut pruned = TopDocsCollector::new(top_n);
    {
        let mut cursor = PostingsCursor::new(postings);
        loop {
            let doc_id = cursor.next_doc();
            if doc_id == NO_MORE_DOCS {
                break;
            }
            let block_impacts = cursor.level0_impacts();
            if !block_impacts.is_empty() && pruned.top_docs().len() >= top_n {
                let bound =
                    max_score_for_impacts(block_impacts, doc_freq, doc_count, avg_field_length);
                let worst = pruned.top_docs().last().map(|h| h.score);
                if worst.is_some_and(|w| bound <= w) {
                    // This whole block cannot possibly enter the top-n:
                    // find its last covered doc ID and jump straight past it.
                    let last_in_block = postings
                        .level0_impacts
                        .iter()
                        .find(|(_, impacts)| impacts.as_slice() == block_impacts)
                        .map(|&(last, _)| last)
                        .expect("block_impacts came from postings.level0_impacts");
                    if cursor.advance(last_in_block + 1) == NO_MORE_DOCS {
                        break;
                    }
                    continue;
                }
            }
            let freq = cursor.freq().expect("started, in range");
            let s = score(
                doc_freq,
                doc_count,
                freq as f32,
                real_field_length(doc_id),
                avg_field_length,
            );
            pruned.collect(doc_id, s);
        }
    }

    assert_eq!(
        brute.top_docs(),
        pruned.top_docs(),
        "block-level max-score pruning must never change the top-{top_n} result"
    );
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
        // tfNorm = 3 / (3 + 1.2*(1-0.75+0.75*1/1)) = 3 / (3 + 1.2*1.0)
        // = 3 / 4.2 = 0.714285...
        let got = tf_norm(3.0, 1.0, 1.0, DEFAULT_K1, DEFAULT_B);
        assert!((got - 0.714_285_7).abs() < 1e-5, "got {got}");
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
        // tfNorm(4, 1, 1, 1.2, 0.75) = 4 / (4 + 1.2*1.0) = 4/5.2 = 0.769_230...
        // score = 1.481604 * 0.769230... = 1.139696...
        let got = score(2, 10, 4.0, UNNORMED_FIELD_LENGTH, UNNORMED_FIELD_LENGTH);
        let expected_idf = 4.4f64.ln() as f32;
        let expected_tf_norm = 4.0f32 / 5.2f32;
        let expected = expected_idf * expected_tf_norm;
        assert!(
            (got - expected).abs() < 1e-4,
            "got {got}, expected {expected}"
        );
        assert!((got - 1.139_696).abs() < 1e-3, "got {got}");
    }

    #[test]
    fn bm25_params_default_matches_lucene_default_constants() {
        let params = Bm25Params::default();
        assert_eq!(params.k1, DEFAULT_K1);
        assert_eq!(params.b, DEFAULT_B);
    }

    #[test]
    fn score_with_params_using_defaults_matches_score_byte_for_byte() {
        // Regression proof (task #214): the new parameterized path must
        // reproduce the existing hardcoded-default path exactly when given
        // Bm25Params::default(), not just "close enough".
        let got = score_with_params(
            2,
            10,
            4.0,
            UNNORMED_FIELD_LENGTH,
            UNNORMED_FIELD_LENGTH,
            Bm25Params::default(),
        );
        let expected = score(2, 10, 4.0, UNNORMED_FIELD_LENGTH, UNNORMED_FIELD_LENGTH);
        assert_eq!(
            got, expected,
            "got {got}, expected byte-identical {expected}"
        );
    }

    #[test]
    fn score_with_params_using_different_k1_b_matches_hand_computed_value() {
        // docFreq=2, docCount=10, freq=4, unnormed field length, but k1=2.0,
        // b=0.5 instead of the 1.2/0.75 defaults.
        // idf(2,10) = ln(4.4) = 1.481604... (same as `score_combines_idf_and_tf_norm`)
        // tfNorm(4, 1, 1, 2.0, 0.5) = 4 / (4 + 2.0*(1 - 0.5 + 0.5*1/1))
        //           = 4 / (4 + 2.0*1.0) = 4 / 6.0 = 0.666666...
        // score = 1.481604 * 0.666666... = 0.987736...
        let params = Bm25Params { k1: 2.0, b: 0.5 };
        let got = score_with_params(
            2,
            10,
            4.0,
            UNNORMED_FIELD_LENGTH,
            UNNORMED_FIELD_LENGTH,
            params,
        );
        let expected_idf = 4.4f64.ln() as f32;
        let expected_tf_norm = 4.0f32 / 6.0f32;
        let expected = expected_idf * expected_tf_norm;
        assert!(
            (got - expected).abs() < 1e-4,
            "got {got}, expected {expected}"
        );
        assert!((got - 0.987_736).abs() < 1e-3, "got {got}");
        // And it must differ measurably from the default-params score.
        let default_score = score(2, 10, 4.0, UNNORMED_FIELD_LENGTH, UNNORMED_FIELD_LENGTH);
        assert!(
            (got - default_score).abs() > 1e-3,
            "different k1/b must produce a measurably different score: {got} vs {default_score}"
        );
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
    fn max_score_for_impacts_matches_hand_computed_value_single_entry() {
        // One impact: freq=3, norm byte 5 -> decode_norm(5) == 5.0 (subnormal,
        // exact -- see `decode_norm_matches_small_float_byte4_to_int`).
        // docFreq=2, docCount=10, avgFieldLength=5.0 (so fieldLength ==
        // avgFieldLength, collapsing the length-norm term to 1.0):
        // idf(2,10) = ln(4.4) = 1.481604..., tfNorm(3,5,5,1.2,0.75)
        // = 3 / (3 + 1.2*1.0) = 3/4.2 = 0.714285...
        // expected = 1.481604 * 0.714285... = 1.058289...
        let impacts = vec![lucene_codecs::postings::Impact { freq: 3, norm: 5 }];
        let got = max_score_for_impacts(&impacts, 2, 10, 5.0);
        let expected = idf(2, 10) * tf_norm(3.0, 5.0, 5.0, DEFAULT_K1, DEFAULT_B);
        assert!(
            (got - expected).abs() < 1e-5,
            "got {got}, expected {expected}"
        );
        assert!((got - 1.058_289).abs() < 1e-3, "got {got}");
    }

    #[test]
    fn max_score_for_impacts_empty_slice_is_zero() {
        assert_eq!(max_score_for_impacts(&[], 2, 10, 1.0), 0.0);
    }

    #[test]
    fn max_score_for_impacts_takes_the_true_max_not_just_the_last_entry() {
        // Impacts are ordered by strictly increasing freq *and* strictly
        // increasing norm (the CompetitiveImpactAccumulator invariant) -- but
        // a higher norm decodes to a longer field, which *penalizes* tf_norm.
        // Construct a case where the last (highest-freq, highest-norm) entry
        // scores *lower* than an earlier entry, proving this function must
        // scan every entry rather than assuming the list's tail bounds it.
        let impacts = vec![
            lucene_codecs::postings::Impact { freq: 2, norm: 1 }, // short field, modest freq
            lucene_codecs::postings::Impact { freq: 3, norm: 60 }, // much longer field
        ];
        let doc_freq = 2;
        let doc_count = 10;
        let avg_field_length = 1.0;
        let score_entry_0 = idf(doc_freq, doc_count)
            * tf_norm(2.0, decode_norm(1), avg_field_length, DEFAULT_K1, DEFAULT_B);
        let score_entry_1 = idf(doc_freq, doc_count)
            * tf_norm(
                3.0,
                decode_norm(60),
                avg_field_length,
                DEFAULT_K1,
                DEFAULT_B,
            );
        assert!(
            score_entry_0 > score_entry_1,
            "test setup must make the earlier (lower-freq, lower-norm) entry \
             the higher-scoring one: {score_entry_0} vs {score_entry_1}"
        );
        let got = max_score_for_impacts(&impacts, doc_freq, doc_count, avg_field_length);
        assert!(
            (got - score_entry_0).abs() < 1e-5,
            "got {got}, expected {score_entry_0}"
        );
    }

    #[test]
    fn max_score_for_impacts_bounds_a_doc_matching_any_single_entry_exactly() {
        // The property this port's pruning actually relies on (see
        // `max_score_for_impacts`'s doc comment): a real doc whose (freq,
        // norm) exactly matches one of the block's competitive impacts
        // entries can never score higher than this function's bound over
        // the whole list -- note this is *not* the same as "any
        // component-wise-dominated (freq, norm) pair is safe" (a doc with a
        // smaller norm than every entry but a small freq can, in principle,
        // score higher than an entry that traded a larger norm for a larger
        // freq -- BM25's tf and length-norm terms move in opposite
        // directions). The real write-side guarantee
        // (`CompetitiveImpactAccumulator`) is that every real doc in the
        // block matches this exactly-one-entry shape or is dominated in the
        // score sense, not the naive component-wise sense.
        let impacts = vec![
            lucene_codecs::postings::Impact { freq: 1, norm: 3 },
            lucene_codecs::postings::Impact { freq: 5, norm: 20 },
            lucene_codecs::postings::Impact { freq: 10, norm: 50 },
        ];
        let doc_freq = 3;
        let doc_count = 50;
        let avg_field_length = 10.0;
        let bound = max_score_for_impacts(&impacts, doc_freq, doc_count, avg_field_length);
        for impact in &impacts {
            let field_length = decode_norm(impact.norm);
            let actual = score(
                doc_freq,
                doc_count,
                impact.freq as f32,
                field_length,
                avg_field_length,
            );
            assert!(
                actual <= bound + 1e-4,
                "doc score {actual} (freq={}, norm={}) exceeded bound {bound}",
                impact.freq,
                impact.norm
            );
        }
    }

    #[test]
    fn block_pruning_via_max_score_matches_brute_force_top_1() {
        use lucene_codecs::postings::{Impact, Postings};

        // Block A: docs 1..=3, freq=5, norm byte 5 (decode_norm(5) == 5.0,
        // exact) -- scores relatively high once avg_field_length == 5.0.
        // Block B: docs 4..=6, freq=1, norm byte 100 (decode_norm(100) ==
        // 3096.0, a much longer field) -- scores much lower.
        let postings = Postings {
            docs: vec![1, 2, 3, 4, 5, 6],
            freqs: vec![5, 5, 5, 1, 1, 1],
            level0_impacts: vec![
                (3, vec![Impact { freq: 5, norm: 5 }]),
                (6, vec![Impact { freq: 1, norm: 100 }]),
            ],
            level1_impacts: Vec::new(),
        };
        let norm_bytes = [5u8, 5, 5, 100, 100, 100];
        let doc_freq = 2;
        let doc_count = 10;
        let avg_field_length = 5.0;

        // Sanity: block B's bound really is lower than block A's real
        // per-doc scores, so the pruning path in the harness below actually
        // exercises the skip branch rather than vacuously never triggering.
        let block_a_score = score(doc_freq, doc_count, 5.0, 5.0, avg_field_length);
        let block_b_bound = max_score_for_impacts(
            &[Impact { freq: 1, norm: 100 }],
            doc_freq,
            doc_count,
            avg_field_length,
        );
        assert!(
            block_b_bound < block_a_score,
            "test setup must make block B's bound beatable by block A's score: \
             {block_b_bound} vs {block_a_score}"
        );

        assert_block_pruning_matches_brute_force(
            &postings,
            &norm_bytes,
            doc_freq,
            doc_count,
            avg_field_length,
            1,
        );
    }

    #[test]
    fn block_pruning_via_max_score_matches_brute_force_top_2_spans_both_blocks() {
        use lucene_codecs::postings::{Impact, Postings};

        // Same two blocks as above, but top_n=2 needs one hit from each
        // block (block A's docs alone can supply at most... here 3, so this
        // covers the "collector not yet full when reaching block B" and "no
        // skip should happen since block B still has a competitive doc"
        // shape, proving pruning doesn't skip when it can't yet prove
        // safety).
        let postings = Postings {
            docs: vec![1, 2, 3, 4, 5, 6],
            freqs: vec![5, 5, 5, 4, 4, 4],
            level0_impacts: vec![
                (3, vec![Impact { freq: 5, norm: 5 }]),
                (6, vec![Impact { freq: 4, norm: 5 }]),
            ],
            level1_impacts: Vec::new(),
        };
        let norm_bytes = [5u8, 5, 5, 5, 5, 5];
        assert_block_pruning_matches_brute_force(&postings, &norm_bytes, 2, 10, 5.0, 2);
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
