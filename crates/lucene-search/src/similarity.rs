//! `BM25Similarity`-equivalent (`org.apache.lucene.search.similarities.BM25Similarity`)
//! as a pure scoring formula; the `Similarity` trait and the other
//! similarities live in [`crate::similarities`].
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
//! four), and is exactly one ULP on every hit of the BM25 fixture.
//! `tests/bm25_scoring_fixtures.rs` pins term, boolean, phrase and
//! multi-phrase scores against real Lucene's own recorded `TopDocs` output
//! **bit for bit**.
//!
//! Defaults `k1 = 1.2`, `b = 0.75` match `BM25Similarity()`'s no-arg constructor,
//! which is what every field in this port's fixtures implicitly uses (no
//! per-field `Similarity` override machinery exists here).
//!
//! ## Norms: real per-doc field length, decoded from `.nvd`/`.nvm`
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
//! one the production callers use. That is a live divergence.

/// `BM25Similarity`'s default `k1` (term-frequency saturation parameter).
pub const DEFAULT_K1: f32 = 1.2;
/// `BM25Similarity`'s default `b` (field-length normalization parameter).
pub const DEFAULT_B: f32 = 0.75;

/// The two knobs real Lucene's `BM25Similarity(float k1, float b)` constructor
/// exposes per-`Similarity`-instance -- `k1` (term-frequency saturation) and `b` (field-length
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
/// [`DEFAULT_B`]. Threading custom `k1`/`b` through every scored path is a
/// larger change; only the single most fundamental scored entry point takes it.
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
    /// the denominator zero or negative and produce infinities. [`Bm25Params`]'s
    /// fields are public, so a caller could set any float.
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
/// 1. **Bit-for-bit agreement with real Lucene.** The multiply form comes out
///    exactly one ULP away from real Lucene's own recorded `TopDocs`
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

/// [`do_score`] over a whole block: `out[i] = do_score(weight, freqs[i], norm_inverse[i])`
/// -- `BM25Similarity.BM25Scorer`'s bulk `score(size, freqs, norms, scores)`.
///
/// The same expression element by element, so every score is bit-for-bit
/// what [`do_score`] gives; written as a flat loop over equal-length slices so
/// it compiles to vector divides (eight lanes at a time on x86-64-v3) rather
/// than one scalar divide per document.
///
/// # Panics
///
/// If `freqs` and `norm_inverse` differ in length.
pub fn do_score_batch(weight: f32, freqs: &[i32], norm_inverse: &[f32], out: &mut Vec<f32>) {
    assert_eq!(freqs.len(), norm_inverse.len());
    out.clear();
    out.extend(
        freqs
            .iter()
            .zip(norm_inverse)
            .map(|(&f, &n)| do_score(weight, f as f32, n)),
    );
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
/// scope. `Bm25Params::default()` produces byte-for-byte the same
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
    max_score_for_impacts_weighted(impacts, idf(doc_freq, doc_count), avg_field_length)
}

/// [`max_score_for_impacts`] with the clause's `weight` (`boost * idf`)
/// already computed -- what a scorer holds, so bounding a block does not pay
/// an `ln()` every time. Same expression, so the same bits.
pub fn max_score_for_impacts_weighted(
    impacts: &[lucene_codecs::postings::Impact],
    weight: f32,
    avg_field_length: f32,
) -> f32 {
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

/// [`max_score_for_impacts_weighted`] over a field's norm-inverse table
/// ([`crate::field_norms::FieldNormsCursor::norm_inverse_table`]) -- what
/// `MaxScoreCache.computeMaxScore` does with `BM25Scorer`'s `cache[]`.
///
/// One multiply per impact instead of a norm decode, a `norm_inverse` division
/// and a score division: [`do_score`] is `w - w / (1 + x)` with
/// `x = freq * normInverse`, and every IEEE operation in it is monotone, so
/// the largest `x` gives the largest score, bit for bit. The documents are
/// scored from the same table, so the bound and the scores it gates share
/// one source. `0.0` for no impacts, as the fold it replaces.
pub fn max_score_for_impacts_table(
    impacts: &[lucene_codecs::postings::Impact],
    weight: f32,
    norm_inverse_table: &[f32; 256],
) -> f32 {
    let mut max_x = None::<f32>;
    for impact in impacts {
        let x = impact.freq as f32 * norm_inverse_table[impact.norm as u8 as usize];
        max_x = Some(max_x.map_or(x, |m| m.max(x)));
    }
    max_x.map_or(0.0, |x| (weight - weight / (1.0 + x)).max(0.0))
}

/// [`max_score_for_impacts_table`] over a level's still-encoded impacts,
/// folded as [`lucene_codecs::postings::for_each_impact`] decodes them: the
/// same maximum of `freq * normInverse`, then the same one BM25 evaluation,
/// so the same float. `None` for an empty run or one that does not decode --
/// the "no bound" a decoded list that is empty gives.
pub fn max_score_for_encoded_impacts_table(
    bytes: &[u8],
    weight: f32,
    norm_inverse_table: &[f32; 256],
) -> Option<f32> {
    let mut max_x = None::<f32>;
    lucene_codecs::postings::for_each_impact(bytes, |freq, norm| {
        let x = freq as f32 * norm_inverse_table[norm as u8 as usize];
        max_x = Some(max_x.map_or(x, |m| m.max(x)));
    })
    .ok()?;
    max_x.map(|x| (weight - weight / (1.0 + x)).max(0.0))
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
    max_score_for_impacts_unnormed_weighted(impacts, idf(doc_freq, doc_count))
}

/// [`max_score_for_impacts_unnormed`] with the clause's `weight` already
/// computed; see [`max_score_for_impacts_weighted`].
pub fn max_score_for_impacts_unnormed_weighted(
    impacts: &[lucene_codecs::postings::Impact],
    weight: f32,
) -> f32 {
    let norm_inverse = norm_inverse(
        UNNORMED_FIELD_LENGTH,
        UNNORMED_FIELD_LENGTH,
        DEFAULT_K1,
        DEFAULT_B,
    );
    // One `norm_inverse` for every document, so the largest frequency gives
    // the largest score (see [`max_score_for_impacts_table`]).
    impacts
        .iter()
        .map(|impact| impact.freq)
        .max()
        .map_or(0.0, |freq| {
            do_score(weight, freq as f32, norm_inverse).max(0.0)
        })
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
mod tests;
