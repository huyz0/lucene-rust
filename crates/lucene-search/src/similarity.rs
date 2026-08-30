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
//! `score = idf * tfNorm` **algebraically**, but the expression this module
//! actually evaluates is real Lucene's own `BM25Scorer.doScore`,
//! `weight - weight / (1 + freq * normInverse)` (see [`do_score`]). The two
//! agree in exact arithmetic and disagree in `f32`: they round at different
//! points, so the gap is *a few* ULP in general (see
//! `do_score_reproduces_the_algebraic_form_to_within_rounding`, which allows
//! four), and was exactly one ULP on every hit of the fixture that caught this.
//! Every scored path in this crate used the multiply form until the b12 sweep;
//! `tests/bm25_scoring_fixtures.rs` now pins term, boolean, phrase and
//! multi-phrase scores against real Lucene's own recorded `TopDocs` output
//! **bit for bit**.
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
//! per query, and [`crate::search_term_query_scored`]/
//! [`crate::search_boolean_query_scored`] use it, falling back to
//! [`UNNORMED_FIELD_LENGTH`] for both lengths only when the field has no opened
//! norms at all (norms disabled for that field, or the caller didn't open a
//! `.nvd`/`.nvm` pair) — a documented, deliberate fallback, not silently wrong
//! data; see [`crate::field_norms`] for exactly when that applies.
//!
//! **`FieldNorms` has two constructors and they do not compute the same
//! `avgFieldLength`**; only `from_field_stats` matches Java, and it is not the
//! one the production callers use. That is a live divergence, not a note about
//! this module: see `docs/sweep/m2/LEDGER.md`'s carry-over row and
//! `docs/sweep/m2/b12-search-core.md` F-7, owned by b13 and b15.

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
///
/// **The fields are private and [`Bm25Params::new`] is the only way to build a
/// non-default one.** Real Lucene validates `k1`/`b` in its constructor and has
/// no setters, and the validation is load-bearing rather than decorative here:
/// `b` outside `0..=1` makes the length-normalization term non-monotonic in the
/// norm, which invalidates every impacts-derived upper bound
/// ([`max_score_for_impacts`]) and turns MAXSCORE block skipping into a source
/// of *missing hits*. Public fields plus a validating constructor would leave
/// `Bm25Params { k1, b }` as an unchecked back door, which is exactly what the
/// FFI entry point -- taking two floats straight off the C ABI -- would have
/// gone through.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bm25Params {
    /// Term-frequency saturation parameter (`BM25Similarity`'s `k1`).
    k1: f32,
    /// Field-length normalization parameter (`BM25Similarity`'s `b`).
    b: f32,
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

impl Bm25Params {
    /// `BM25Similarity(float k1, float b)`'s **validating** constructor.
    ///
    /// Real Lucene rejects out-of-range parameters at construction, and the
    /// rejections are not decorative -- they are what keeps the score
    /// monotonic. `b > 1` or `b < 0` makes the length-normalization term
    /// non-monotonic in the norm, which silently invalidates every
    /// impacts-derived upper bound this crate computes
    /// ([`max_score_for_impacts`]), turning MAXSCORE block skipping from an
    /// optimization into a source of missing hits. A negative `k1` can make
    /// the denominator zero or negative and produce infinities. This port had
    /// no validation at all before the b12 sweep: [`Bm25Params`]'s fields are
    /// public and a caller (including the FFI one) could set any float.
    ///
    /// The error messages are Lucene's own, verbatim, so a caller diagnosing a
    /// rejection finds the same string in either engine.
    pub fn new(k1: f32, b: f32) -> std::result::Result<Self, String> {
        if !k1.is_finite() || k1 < 0.0 {
            return Err(format!(
                "illegal k1 value: {k1}, must be a non-negative finite value"
            ));
        }
        if b.is_nan() || !(0.0..=1.0).contains(&b) {
            return Err(format!("illegal b value: {b}, must be between 0 and 1"));
        }
        Ok(Bm25Params { k1, b })
    }

    /// `BM25Similarity.getK1()`.
    pub fn k1(self) -> f32 {
        self.k1
    }

    /// `BM25Similarity.getB()`.
    pub fn b(self) -> f32 {
        self.b
    }
}

/// The constant `fieldLength`/`avgFieldLength` this port substitutes when a
/// field has no opened norms (norms disabled for that field, or the caller
/// didn't open a `.nvd`/`.nvm` pair for this search) — see this module's doc
/// comment for why `1.0`/`1.0` (rather than e.g. `0.0`/`1.0`) is the honest
/// "no-op" substitution: it makes the length-normalization term collapse to a
/// constant instead of silently zeroing or exploding it.
pub const UNNORMED_FIELD_LENGTH: f32 = 1.0;

/// [`norm_inverse`] evaluated at [`UNNORMED_FIELD_LENGTH`] for both lengths and
/// the default `k1`/`b` -- the length-normalization reciprocal every scoring
/// path uses for a document whose field has no opened norms. Spelled as a
/// constant so the no-norms branch of a per-document scoring loop is a load,
/// not a divide.
pub const UNNORMED_NORM_INVERSE: f32 = 1.0
    / (DEFAULT_K1
        * ((1.0 - DEFAULT_B) + DEFAULT_B * UNNORMED_FIELD_LENGTH / UNNORMED_FIELD_LENGTH));

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

/// `BM25Similarity.scorer`'s per-norm-byte cache entry, `cache[i] = 1f / (k1 *
/// ((1 - b) + b * LENGTH_TABLE[i] / avgdl))` -- the *reciprocal* of the
/// length-normalization denominator, which is the only form real Lucene's
/// scorer ever holds. [`crate::field_norms::FieldNorms`] precomputes all 256 of
/// these once per field; this function is the single-value form, for the paths
/// that have a field length rather than a norm byte.
pub fn norm_inverse(field_length: f32, avg_field_length: f32, k1: f32, b: f32) -> f32 {
    1.0 / (k1 * ((1.0 - b) + b * field_length / avg_field_length))
}

/// `BM25Scorer.doScore(float freq, float normInverse)` **verbatim**:
/// `weight - weight / (1 + freq * normInverse)`, where `weight` is
/// `boost * idf`.
///
/// This is not a stylistic choice, and it is not interchangeable with the
/// algebraically-equal `weight * freq / (freq + k1 * (1 - b + b * dl / avgdl))`
/// that [`tf_norm`] spells out. Two reasons, both from `BM25Similarity.java`'s
/// own comment on this method:
///
/// 1. **Bit-for-bit agreement with real Lucene.** Every scored path in this
///    port used the multiply form until the b12 sweep, and every one of them
///    came out exactly one ULP away from real Lucene's own recorded `TopDocs`
///    scores on a real fixture segment (see
///    `crates/lucene-search/tests/bm25_scoring_fixtures.rs`). Float addition,
///    subtraction and division do not commute across the rewrite; only this
///    expression reproduces Lucene's bits.
/// 2. **Monotonicity, which block-max pruning depends on.** Lucene rewrites
///    `freq / (freq + norm)` to `1 - 1 / (1 + freq * (1/norm))` precisely
///    because the latter is guaranteed monotonic in both `freq` and `norm`
///    without promoting to `double`. An impacts-derived upper bound
///    ([`max_score_for_impacts`]) is only sound if it is computed by the same
///    expression the per-document score is; mixing the two forms can put the
///    bound one ULP *below* a real document's score, and MAXSCORE will then
///    skip a document that belonged in the top-`n`.
pub fn do_score(weight: f32, freq: f32, norm_inverse: f32) -> f32 {
    weight - weight / (1.0 + freq * norm_inverse)
}

/// The full per-document BM25 score, using the default `k1`/`b` and the given
/// collection/document statistics -- real Lucene's `BM25Scorer.score(freq,
/// encodedNorm)`, i.e. [`do_score`] over [`idf`] and [`norm_inverse`].
///
/// - `doc_freq`: the term's document frequency in `field`.
/// - `doc_count`: the field's document count (see [`idf`]).
/// - `freq`: the term's frequency in the matched document.
/// - `field_length`/`avg_field_length`: [`UNNORMED_FIELD_LENGTH`] for both when
///   the field has no opened norms; see this module's doc comment.
pub fn score(
    doc_freq: i64,
    doc_count: i64,
    freq: f32,
    field_length: f32,
    avg_field_length: f32,
) -> f32 {
    score_with_params(
        doc_freq,
        doc_count,
        freq,
        field_length,
        avg_field_length,
        Bm25Params::default(),
    )
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
    do_score(
        idf(doc_freq, doc_count),
        freq,
        norm_inverse(field_length, avg_field_length, params.k1(), params.b()),
    )
}

/// Upper bound on the BM25 score any document covered by a single block/span
/// of competitive impacts (`lucene_codecs::postings::Impact`) can achieve —
/// this port's scoped-down stand-in for real Lucene's `MaxScoreCache`/
/// `ImpactsEnum.getMaxScore` (see `docs/parity.md`'s postings row for the
/// full `ImpactsEnum` hierarchy this port does *not* implement).
///
/// [`lucene_codecs::postings`]'s impacts invariant (`Postings::level0_impacts`'s doc
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
    let weight = idf(doc_freq, doc_count);
    impacts
        .iter()
        .map(|impact| {
            let field_length = decode_norm(impact.norm);
            // Deliberately `do_score`, not `idf * tf_norm`: the bound and the
            // per-document score it gates must be the *same* float expression,
            // or the bound can land one ULP under a real score and skip a
            // document that belonged in the top-`n`. See [`do_score`].
            do_score(
                weight,
                impact.freq as f32,
                norm_inverse(field_length, avg_field_length, DEFAULT_K1, DEFAULT_B),
            )
        })
        .fold(0.0f32, f32::max)
}

/// [`max_score_for_impacts`]'s sibling for the `norms == None` scoring path.
///
/// When a search runs without opened norms every document is scored with
/// `field_length == avg_field_length == `[`UNNORMED_FIELD_LENGTH`], **not** with
/// whatever real norm byte the wire impacts happen to carry. Bounding with the
/// wire norms there would bound a different scoring formula than the one
/// actually applied, which can underestimate the bound and skip a document that
/// should have been collected. Four call sites in `lib.rs` open-coded this and
/// had to keep the rule in step; it lives here once instead.
pub fn max_score_for_impacts_unnormed(
    impacts: &[lucene_codecs::postings::Impact],
    doc_freq: i64,
    doc_count: i64,
) -> f32 {
    let weight = idf(doc_freq, doc_count);
    let norm_inverse = norm_inverse(
        UNNORMED_FIELD_LENGTH,
        UNNORMED_FIELD_LENGTH,
        DEFAULT_K1,
        DEFAULT_B,
    );
    impacts
        .iter()
        .map(|impact| do_score(weight, impact.freq as f32, norm_inverse))
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
    fn bm25_params_new_rejects_exactly_what_lucene_rejects() {
        // `BM25Similarity(float k1, float b, boolean, float)`'s own guards,
        // with its own messages.
        assert_eq!(
            Bm25Params::new(-0.1, 0.5).unwrap_err(),
            "illegal k1 value: -0.1, must be a non-negative finite value"
        );
        assert!(Bm25Params::new(f32::NAN, 0.5).is_err());
        assert!(Bm25Params::new(f32::INFINITY, 0.5).is_err());
        assert_eq!(
            Bm25Params::new(1.2, 1.5).unwrap_err(),
            "illegal b value: 1.5, must be between 0 and 1"
        );
        assert!(Bm25Params::new(1.2, -0.001).is_err());
        assert!(Bm25Params::new(1.2, f32::NAN).is_err());
        // The boundaries Lucene accepts.
        assert_eq!(
            Bm25Params::new(0.0, 0.0).unwrap(),
            Bm25Params { k1: 0.0, b: 0.0 }
        );
        assert_eq!(
            Bm25Params::new(1.2, 1.0).unwrap(),
            Bm25Params { k1: 1.2, b: 1.0 }
        );
        assert_eq!(
            Bm25Params::new(DEFAULT_K1, DEFAULT_B).unwrap(),
            Bm25Params::default()
        );
    }

    #[test]
    fn do_score_reproduces_the_algebraic_form_to_within_rounding() {
        // The two expressions are equal in exact arithmetic; the point of
        // `do_score` is that it is the one real Lucene evaluates. Pin both
        // facts: close, and not always identical.
        let mut differed = false;
        for freq in [1.0f32, 2.0, 3.0, 7.0, 41.0] {
            for len in [1.0f32, 2.0, 5.0, 17.0, 96.0] {
                let weight = idf(3, 97);
                let got = do_score(weight, freq, norm_inverse(len, 12.5, DEFAULT_K1, DEFAULT_B));
                let algebraic = weight * tf_norm(freq, len, 12.5, DEFAULT_K1, DEFAULT_B);
                // A few ULP, not one: the two expressions differ in *where*
                // they round (one division and a subtraction versus two
                // divisions and a multiplication), so the gap compounds.
                assert!(
                    (got - algebraic).abs() <= 4.0 * f32::EPSILON * got.abs().max(1.0),
                    "freq={freq} len={len}: {got} vs {algebraic}"
                );
                differed |= got.to_bits() != algebraic.to_bits();
            }
        }
        assert!(
            differed,
            "if the two forms never differ, this crate's insistence on `do_score` \
             would be pointless -- they do differ, and Lucene's bits are do_score's"
        );
    }

    #[test]
    fn norm_inverse_matches_lucenes_scorer_cache_entry() {
        // `cache[i] = 1f / (k1 * ((1 - b) + b * LENGTH_TABLE[i] / avgdl))`.
        let avgdl = 4.0f32;
        for byte in [0u8, 1, 23, 40, 100, 255] {
            let len = decode_norm(byte as i64);
            let expected = 1.0f32 / (DEFAULT_K1 * ((1.0 - DEFAULT_B) + DEFAULT_B * len / avgdl));
            assert_eq!(norm_inverse(len, avgdl, DEFAULT_K1, DEFAULT_B), expected);
        }
        // And the constant the no-norms path uses is the same function at
        // `UNNORMED_FIELD_LENGTH`.
        assert_eq!(
            UNNORMED_NORM_INVERSE,
            norm_inverse(
                UNNORMED_FIELD_LENGTH,
                UNNORMED_FIELD_LENGTH,
                DEFAULT_K1,
                DEFAULT_B
            )
        );
    }

    #[test]
    fn max_score_for_impacts_is_never_below_a_real_scored_document() {
        // The soundness property MAXSCORE rests on, and the reason the bound
        // and the score must be the same float expression: for every
        // (freq, norm) pair the bound covers, the bound must be >= the score
        // that pair actually produces. One ULP the wrong way here drops hits.
        let impacts: Vec<lucene_codecs::postings::Impact> = [(1i32, 0i64), (3, 40), (9, 100)]
            .iter()
            .map(|&(freq, norm)| lucene_codecs::postings::Impact { freq, norm })
            .collect();
        let avgdl = 7.5f32;
        let bound = max_score_for_impacts(&impacts, 5, 200, avgdl);
        for impact in &impacts {
            let actual = score(5, 200, impact.freq as f32, decode_norm(impact.norm), avgdl);
            assert!(actual <= bound, "{actual} > bound {bound} for {impact:?}");
        }
        // Unnormed sibling, same property against the unnormed scoring path.
        let unnormed = max_score_for_impacts_unnormed(&impacts, 5, 200);
        for impact in &impacts {
            let actual = score(
                5,
                200,
                impact.freq as f32,
                UNNORMED_FIELD_LENGTH,
                UNNORMED_FIELD_LENGTH,
            );
            assert!(actual <= unnormed, "{actual} > bound {unnormed}");
        }
    }

    #[test]
    fn bm25_params_default_matches_lucene_default_constants() {
        let params = Bm25Params::default();
        assert_eq!(params.k1(), DEFAULT_K1);
        assert_eq!(params.b(), DEFAULT_B);
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
        let params = Bm25Params::new(2.0, 0.5).expect("in range");
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
