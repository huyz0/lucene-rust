//! `Explanation`-equivalent (`org.apache.lucene.search.Explanation`) and
//! `explain_clause`, this port's `IndexSearcher.explain(query, doc)`
//! counterpart: a per-doc, human-readable breakdown of exactly how a
//! [`crate::Clause`]'s score for one document was computed.
//!
//! **This task does not change any scoring behavior.** Every arm below
//! recomputes a score using the *exact same* functions and arguments the
//! already-verified `search_*_scored` functions in `lib.rs` already call
//! ([`crate::similarity::idf`], [`crate::similarity::tf_norm`],
//! [`crate::term_doc_freqs`], [`crate::dismax_scores`]'s max+tie-breaker
//! formula, etc) — `explain_clause`'s reported top-level `value` is therefore
//! bit-for-bit identical to what `search_term_query_scored`/
//! `search_boolean_query_scored`/`search_disjunction_max_query_scored` already
//! produce for the same doc, not a second, independently-computed
//! approximation. This crate's own unit tests below assert that equality
//! directly (`assert_eq!`, not an epsilon comparison) against those functions'
//! actual output.
//!
//! ## [`Explanation`]'s shape
//!
//! Mirrors real Lucene's `Explanation` class exactly: `value` (the computed
//! score contribution), `description` (what this node represents),
//! `details` (child `Explanation`s the value was derived from), and `matched`
//! (real Lucene's own `isMatch()`/internal `match` boolean — `true` for every
//! node built via [`Explanation::match_`], `false` with `value == 0.0` for
//! every node built via [`Explanation::no_match`], the same
//! `Explanation.match(...)`/`Explanation.noMatch(...)` factory-method split
//! real Lucene's own class provides).
//!
//! ## Which [`crate::Clause`] variants get a "real" explanation vs a flat one
//!
//! - **Real, detailed explanations** (mirroring real Lucene's own
//!   `TermWeight.explain`/`BooleanWeight.explain`/`PhraseWeight.explain`/
//!   `DisjunctionMaxQuery`'s explain/`ConstantScoreWeight.explain`/
//!   `BoostQuery.BoostWeight.explain` as closely as this port's own scoring
//!   math allows): `Clause::Term`, `Clause::Boolean`, `Clause::Phrase`,
//!   `Clause::DisjunctionMax`, `Clause::ConstantScore`, `Clause::Boost`.
//! - **Flat, one-level explanations** ("matches, constant score 1.0" or "no
//!   match" — these clauses have no single term's frequency/idf to break
//!   down further, see each query type's own doc comment in `query.rs` for
//!   why they're unscored): `Clause::Wildcard`, `Clause::Prefix`,
//!   `Clause::Fuzzy`, `Clause::Regexp`, `Clause::Span`.

use std::collections::HashMap;

use lucene_codecs::blocktree::BlockTreeFields;
use lucene_codecs::postings::{DocInput, PayInput, PosInput};
use lucene_util::fixed_bit_set::FixedBitSet;

use crate::query::{BoostQuery, ConstantScoreQuery, DisjunctionMaxQuery, PhraseQuery, TermQuery};
use crate::{similarity, BooleanQuery, Clause, FieldNorms, Result};

/// The Rust analogue of real Lucene's `Explanation` — see this module's doc
/// comment for the exact shape and factory-method correspondence.
#[derive(Debug, Clone, PartialEq)]
pub struct Explanation {
    /// Real Lucene's `isMatch()` — `true` for a matching node (built via
    /// [`Explanation::match_`]), `false` for a non-matching node (built via
    /// [`Explanation::no_match`], always paired with `value == 0.0`).
    pub matched: bool,
    /// The computed score contribution this node represents. Always `0.0`
    /// for a non-matching node, same as real `Explanation.noMatch`.
    pub value: f32,
    /// Human-readable description of what this node's `value` represents.
    pub description: String,
    /// Child explanations `value` was derived from — empty for a leaf node.
    pub details: Vec<Explanation>,
}

impl Explanation {
    /// Real `Explanation.match(value, description, details...)`-equivalent:
    /// builds a matching, leaf explanation (`details` empty; use
    /// [`Self::with_details`] to attach children).
    pub fn match_(value: f32, description: impl Into<String>) -> Self {
        Self {
            matched: true,
            value,
            description: description.into(),
            details: Vec::new(),
        }
    }

    /// Real `Explanation.noMatch(description, details...)`-equivalent: a
    /// non-matching explanation, `value` fixed at `0.0` — real Lucene's own
    /// convention (a non-match has no score to report).
    pub fn no_match(description: impl Into<String>) -> Self {
        Self {
            matched: false,
            value: 0.0,
            description: description.into(),
            details: Vec::new(),
        }
    }

    /// Builder-style: attaches `details` (child explanations) to this node.
    pub fn with_details(mut self, details: Vec<Explanation>) -> Self {
        self.details = details;
        self
    }
}

/// Real `IndexSearcher.explain(query, doc)`-equivalent for one already-opened
/// segment and one already-resolved [`Clause`] (`query`), matching whatever
/// `must`/`should`/`must_not`/nesting the clause tree describes — see this
/// module's doc comment for exactly which variants get a detailed vs flat
/// explanation, and `lib.rs`'s `search_*_scored` functions this mirrors
/// (`search_term_query_scored`, `search_boolean_query_scored`,
/// `search_phrase_query_scored`, `search_disjunction_max_query_scored`).
///
/// - `doc`: the single doc ID to explain (real Lucene's `explain(query,
///   int doc)` signature takes exactly one doc for exactly this reason —
///   explain is a diagnostic tool for one result, not a bulk scoring path).
/// - `norms`: same contract as [`crate::search_boolean_query_scored`]'s —
///   per-field real norms, falling back to
///   [`crate::similarity::UNNORMED_FIELD_LENGTH`] for an unlisted field or
///   when `norms` itself is `None`.
#[allow(clippy::too_many_arguments)]
pub fn explain_clause(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    pos_in: Option<&PosInput<'_>>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    clause: &Clause,
    doc: i32,
    norms: Option<&HashMap<String, FieldNorms<'_>>>,
) -> Result<Explanation> {
    match clause {
        Clause::Term(query) => {
            let clause_norms = norms.and_then(|m| m.get(&query.field));
            explain_term(fields, doc_in, live_docs, query, doc, clause_norms)
        }
        Clause::Phrase(query) => {
            let clause_norms = norms.and_then(|m| m.get(&query.field));
            explain_phrase(
                fields,
                doc_in,
                pos_in,
                pay_in,
                live_docs,
                query,
                doc,
                clause_norms,
            )
        }
        Clause::Boolean(nested) => explain_boolean(
            fields, doc_in, pos_in, pay_in, live_docs, nested, doc, norms,
        ),
        Clause::DisjunctionMax(nested) => explain_dismax(
            fields, doc_in, pos_in, pay_in, live_docs, nested, doc, norms,
        ),
        Clause::ConstantScore(nested) => {
            explain_constant_score(fields, doc_in, pos_in, pay_in, live_docs, nested, doc)
        }
        Clause::Boost(nested) => explain_boost(
            fields, doc_in, pos_in, pay_in, live_docs, nested, doc, norms,
        ),
        Clause::Wildcard(query) => {
            let matched = crate::wildcard_doc_ids(fields, doc_in, live_docs, query)?.contains(&doc);
            Ok(explain_flat_match(matched))
        }
        Clause::Prefix(query) => {
            let matched = crate::prefix_doc_ids(fields, doc_in, live_docs, query)?.contains(&doc);
            Ok(explain_flat_match(matched))
        }
        Clause::Fuzzy(query) => {
            let matched = crate::fuzzy_doc_ids(fields, doc_in, live_docs, query)?.contains(&doc);
            Ok(explain_flat_match(matched))
        }
        Clause::Regexp(query) => {
            let matched = crate::regexp_doc_ids(fields, doc_in, live_docs, query)?.contains(&doc);
            Ok(explain_flat_match(matched))
        }
        Clause::Span(query) => {
            let matched = crate::span_doc_ids(fields, doc_in, pos_in, pay_in, live_docs, query)?
                .contains(&doc);
            Ok(explain_flat_match(matched))
        }
        Clause::PointsRange(query) => {
            Err(crate::Error::UnexecutablePointsRange(query.field.clone()))
        }
        Clause::MatchAllDocs(query) => {
            let matched = crate::match_all_doc_ids(live_docs, query.max_doc).contains(&doc);
            Ok(explain_flat_match(matched))
        }
        Clause::MatchNoDocs(_) => Ok(Explanation::no_match("MatchNoDocsQuery never matches")),
        Clause::TermInSet(query) => {
            let matched =
                crate::term_in_set_doc_ids(fields, doc_in, live_docs, query)?.contains(&doc);
            Ok(explain_flat_match(matched))
        }
    }
}

/// A leaf clause with no per-term breakdown (`Wildcard`/`Prefix`/`Fuzzy`/
/// `Regexp`/`Span` — see this module's doc comment): matches score exactly
/// `1.0` (same flat constant every `clause_scores` arm for these variants
/// already reports), non-matches are a clean `no_match` at `0.0`.
fn explain_flat_match(matched: bool) -> Explanation {
    if matched {
        Explanation::match_(1.0, "1.0 = matches, unscored constant score")
    } else {
        Explanation::no_match("no matching term")
    }
}

/// Whether `clause` matches `doc` at all — used by [`explain_boolean`]/
/// [`explain_dismax`]/[`explain_constant_score`] to decide "does this
/// sub-clause participate" without needing its full score breakdown.
fn clause_matches(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    pos_in: Option<&PosInput<'_>>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    clause: &Clause,
    doc: i32,
) -> Result<bool> {
    Ok(
        crate::resolve_clause_docs(fields, doc_in, pos_in, pay_in, live_docs, clause)?
            .contains(&doc),
    )
}

/// [`Clause::Term`]'s explanation: mirrors real `TermWeight.explain`/
/// `BM25Scorer.explain` — a `weight(field:term)` node wrapping a
/// `score(freq=...)` node, itself wrapping `idf` (with `docFreq`/`docCount`
/// leaf details) and `tfNorm` (with `freq`/`k1`/`b`/`fieldLength`/
/// `avgFieldLength` leaf details). `value` is computed via the exact same
/// [`similarity::idf`]/[`similarity::tf_norm`] calls, in the same order, as
/// [`crate::term_doc_scores`] — bit-for-bit identical to
/// `search_term_query_scored`'s own output for this doc (verified by this
/// module's own unit tests).
fn explain_term(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &TermQuery,
    doc: i32,
    norms: Option<&FieldNorms<'_>>,
) -> Result<Explanation> {
    let Some(field_terms) = fields.field(&query.field) else {
        return Ok(Explanation::no_match(format!(
            "no matching term, field '{}' not found",
            query.field
        )));
    };
    let Some(stats) = field_terms.seek_exact(&query.term) else {
        return Ok(Explanation::no_match("no matching term"));
    };
    let doc_freqs = crate::term_doc_freqs(fields, doc_in, live_docs, query)?;
    let Some(&(_, freq)) = doc_freqs.iter().find(|&&(d, _)| d == doc) else {
        return Ok(Explanation::no_match(format!(
            "no matching term, doc={doc} does not contain this term or is not live"
        )));
    };

    let doc_count = field_terms.doc_count as i64;
    let (field_length, avg_field_length) = match norms {
        Some(fn_) => (fn_.field_length(doc)?, fn_.avg_field_length),
        None => (
            similarity::UNNORMED_FIELD_LENGTH,
            similarity::UNNORMED_FIELD_LENGTH,
        ),
    };
    let idf = similarity::idf(stats.doc_freq as i64, doc_count);
    let tf_norm = similarity::tf_norm(
        freq as f32,
        field_length,
        avg_field_length,
        similarity::DEFAULT_K1,
        similarity::DEFAULT_B,
    );
    let value = idf * tf_norm;

    let idf_explanation = Explanation::match_(
        idf,
        format!(
            "idf, computed as log(1 + (docCount - docFreq + 0.5) / (docFreq + 0.5)) from docFreq={}, docCount={doc_count}",
            stats.doc_freq
        ),
    )
    .with_details(vec![
        Explanation::match_(stats.doc_freq as f32, "docFreq, number of documents containing term"),
        Explanation::match_(doc_count as f32, "docCount, total number of documents with field"),
    ]);

    let tf_norm_explanation = Explanation::match_(
        tf_norm,
        "tfNorm, computed as freq / (freq + k1 * (1 - b + b * fieldLength / avgFieldLength)) from:",
    )
    .with_details(vec![
        Explanation::match_(freq as f32, "freq, occurrences of term within document"),
        Explanation::match_(similarity::DEFAULT_K1, "k1, term saturation parameter"),
        Explanation::match_(similarity::DEFAULT_B, "b, length normalization parameter"),
        Explanation::match_(field_length, "fieldLength"),
        Explanation::match_(avg_field_length, "avgFieldLength"),
    ]);

    let score_explanation = Explanation::match_(
        value,
        format!("score(freq={freq}), computed as idf * tfNorm from:"),
    )
    .with_details(vec![idf_explanation, tf_norm_explanation]);

    Ok(Explanation::match_(
        value,
        format!(
            "weight({}:{}), result of:",
            query.field,
            String::from_utf8_lossy(&query.term)
        ),
    )
    .with_details(vec![score_explanation]))
}

/// [`Clause::Phrase`]'s explanation: same shape as [`explain_term`], but the
/// idf is the sum of every constituent term's own idf and `tfNorm` uses the
/// doc's phrase frequency in place of a single term's `freq` — mirroring
/// [`crate::search_phrase_query_scored`]'s exact formula (see that
/// function's doc comment). A single-term phrase delegates straight to
/// [`explain_term`], same degenerate-case convention
/// `search_phrase_query_scored` itself uses.
#[allow(clippy::too_many_arguments)]
fn explain_phrase(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    pos_in: Option<&PosInput<'_>>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &PhraseQuery,
    doc: i32,
    norms: Option<&FieldNorms<'_>>,
) -> Result<Explanation> {
    if query.terms.is_empty() {
        return Ok(Explanation::no_match(
            "PhraseQuery with no terms matches nothing",
        ));
    }
    if query.terms.len() == 1 {
        let term_query = TermQuery::new(query.field.clone(), query.terms[0].clone());
        return explain_term(fields, doc_in, live_docs, &term_query, doc, norms);
    }
    let Some(pos_in) = pos_in else {
        return Err(crate::Error::MissingPosInput);
    };

    let Some(field_terms) = fields.field(&query.field) else {
        return Ok(Explanation::no_match(format!(
            "no matching term, field '{}' not found",
            query.field
        )));
    };

    let doc_count = field_terms.doc_count as i64;
    let mut idf_sum = 0.0f32;
    let mut idf_details = Vec::with_capacity(query.terms.len());
    for term in &query.terms {
        let Some(stats) = field_terms.seek_exact(term) else {
            return Ok(Explanation::no_match(format!(
                "no matching term, phrase term '{}' not found",
                String::from_utf8_lossy(term)
            )));
        };
        let term_idf = similarity::idf(stats.doc_freq as i64, doc_count);
        idf_sum += term_idf;
        idf_details.push(Explanation::match_(
            term_idf,
            format!(
                "idf({}), docFreq={}, docCount={doc_count}",
                String::from_utf8_lossy(term),
                stats.doc_freq
            ),
        ));
    }

    let mut per_term_docs: Vec<Vec<i32>> = Vec::with_capacity(query.terms.len());
    let mut per_term_maps: Vec<HashMap<i32, Vec<i32>>> = Vec::with_capacity(query.terms.len());
    for term in &query.terms {
        let Some((docs, map)) = crate::term_doc_positions(
            fields,
            doc_in,
            pos_in,
            pay_in,
            live_docs,
            &query.field,
            term,
        )?
        else {
            return Ok(Explanation::no_match(format!(
                "no matching term, phrase term '{}' not found",
                String::from_utf8_lossy(term)
            )));
        };
        per_term_docs.push(docs);
        per_term_maps.push(map);
    }

    if !per_term_docs.iter().all(|docs| docs.contains(&doc)) {
        return Ok(Explanation::no_match(format!(
            "no matching phrase, doc={doc} is missing at least one phrase term"
        )));
    }

    let term_positions: Vec<Vec<i32>> = per_term_maps
        .iter()
        .map(|m| {
            m.get(&doc)
                .cloned()
                .expect("doc came from the conjunction of every term's own doc list")
        })
        .collect();
    let phrase_freq = if query.slop == 0 {
        crate::phrase_freq_exact(&term_positions)
    } else if crate::phrase_matches_in_doc_sloppy(&term_positions, query.slop) {
        1
    } else {
        0
    };
    if phrase_freq == 0 {
        return Ok(Explanation::no_match(format!(
            "no matching phrase alignment for doc={doc}"
        )));
    }

    let (field_length, avg_field_length) = match norms {
        Some(fn_) => (fn_.field_length(doc)?, fn_.avg_field_length),
        None => (
            similarity::UNNORMED_FIELD_LENGTH,
            similarity::UNNORMED_FIELD_LENGTH,
        ),
    };
    let tf_norm = similarity::tf_norm(
        phrase_freq as f32,
        field_length,
        avg_field_length,
        similarity::DEFAULT_K1,
        similarity::DEFAULT_B,
    );
    let value = idf_sum * tf_norm;

    let idf_explanation =
        Explanation::match_(idf_sum, "idf, sum of each phrase term's own idf, from:")
            .with_details(idf_details);

    let tf_norm_explanation = Explanation::match_(
        tf_norm,
        "tfNorm, computed as phraseFreq / (phraseFreq + k1 * (1 - b + b * fieldLength / avgFieldLength)) from:",
    )
    .with_details(vec![
        Explanation::match_(phrase_freq as f32, "phraseFreq, count of valid phrase alignments"),
        Explanation::match_(similarity::DEFAULT_K1, "k1, term saturation parameter"),
        Explanation::match_(similarity::DEFAULT_B, "b, length normalization parameter"),
        Explanation::match_(field_length, "fieldLength"),
        Explanation::match_(avg_field_length, "avgFieldLength"),
    ]);

    let score_explanation = Explanation::match_(
        value,
        format!("score(phraseFreq={phrase_freq}), computed as idf * tfNorm from:"),
    )
    .with_details(vec![idf_explanation, tf_norm_explanation]);

    Ok(Explanation::match_(
        value,
        format!("weight({}:\"...\"), result of:", query.field),
    )
    .with_details(vec![score_explanation]))
}

/// [`Clause::Boolean`]'s explanation: mirrors real `BooleanWeight.explain` --
/// `no_match` when the doc doesn't satisfy `must`'s conjunction / the
/// `should`'s disjunction (when `must` is empty) / `minimum_should_match`, or
/// falls in `must_not`'s exclusion (see [`crate::matched_boolean_docs`], the
/// same matched-doc-set computation `search_boolean_query`/
/// `search_boolean_query_scored` already use); otherwise a `sum of:` node
/// whose value is the sum of every `must` clause's own explanation (always
/// included, since a matching doc satisfies every `must` clause by
/// definition) plus every `should` clause that itself matches this doc --
/// exactly [`crate::search_boolean_query_scored`]'s own summation (`must`
/// chained with `should`, via [`crate::clause_scores`]), so `value` is
/// bit-for-bit identical to that function's output for this doc.
#[allow(clippy::too_many_arguments)]
fn explain_boolean(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    pos_in: Option<&PosInput<'_>>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &BooleanQuery,
    doc: i32,
    norms: Option<&HashMap<String, FieldNorms<'_>>>,
) -> Result<Explanation> {
    let Some(matched) =
        crate::matched_boolean_docs(fields, doc_in, pos_in, pay_in, live_docs, query)?
    else {
        return Ok(Explanation::no_match(
            "BooleanQuery with no must/should clauses matches nothing",
        ));
    };
    let matched_set: std::collections::HashSet<i32> = matched.collect();
    if !matched_set.contains(&doc) {
        return Ok(Explanation::no_match(
            "Failure to meet condition(s) of required/prohibited clause(s)",
        ));
    }

    let mut details = Vec::new();
    let mut total = 0.0f32;
    for clause in &query.must {
        let e = explain_clause(
            fields, doc_in, pos_in, pay_in, live_docs, clause, doc, norms,
        )?;
        total += e.value;
        details.push(e);
    }
    for clause in &query.should {
        if clause_matches(fields, doc_in, pos_in, pay_in, live_docs, clause, doc)? {
            let e = explain_clause(
                fields, doc_in, pos_in, pay_in, live_docs, clause, doc, norms,
            )?;
            total += e.value;
            details.push(e);
        }
    }

    Ok(Explanation::match_(total, format!("{total} = sum of:")).with_details(details))
}

/// [`Clause::DisjunctionMax`]'s explanation: mirrors real
/// `DisjunctionMaxQuery`'s explain -- the matching disjunct with the highest
/// score, plus `tie_breaker * sum(every other matching disjunct's score)`;
/// see [`crate::dismax_scores`]'s doc comment for the exact formula this
/// mirrors (`max + tieBreaker * sum(rest)`), computed here from the same
/// per-disjunct explanations (in `query.disjuncts`' own order, same
/// summation order `dismax_scores` uses, so floating-point results match
/// bit-for-bit).
#[allow(clippy::too_many_arguments)]
fn explain_dismax(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    pos_in: Option<&PosInput<'_>>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &DisjunctionMaxQuery,
    doc: i32,
    norms: Option<&HashMap<String, FieldNorms<'_>>>,
) -> Result<Explanation> {
    if query.disjuncts.is_empty() {
        return Ok(Explanation::no_match(
            "DisjunctionMaxQuery with no disjuncts matches nothing",
        ));
    }

    let mut sub_explanations = Vec::new();
    for clause in &query.disjuncts {
        if clause_matches(fields, doc_in, pos_in, pay_in, live_docs, clause, doc)? {
            sub_explanations.push(explain_clause(
                fields, doc_in, pos_in, pay_in, live_docs, clause, doc, norms,
            )?);
        }
    }
    if sub_explanations.is_empty() {
        return Ok(Explanation::no_match(format!(
            "no matching disjunct for doc={doc}"
        )));
    }

    let mut max_score = f32::NEG_INFINITY;
    let mut sum_score = 0.0f32;
    for e in &sub_explanations {
        sum_score += e.value;
        if e.value > max_score {
            max_score = e.value;
        }
    }
    let other_sum = sum_score - max_score;
    let value = max_score + query.tie_breaker * other_sum;

    Ok(Explanation::match_(
        value,
        format!(
            "{value} = max of:, plus {} times others of:",
            query.tie_breaker
        ),
    )
    .with_details(sub_explanations))
}

/// [`Clause::ConstantScore`]'s explanation: mirrors real
/// `ConstantScoreWeight.explain` -- matches iff the wrapped clause matches
/// (its own score discarded entirely), always scoring exactly
/// `nested.score`, same as [`crate::clause_scores`]'s `Clause::ConstantScore`
/// arm.
fn explain_constant_score(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    pos_in: Option<&PosInput<'_>>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    nested: &ConstantScoreQuery,
    doc: i32,
) -> Result<Explanation> {
    if !clause_matches(
        fields,
        doc_in,
        pos_in,
        pay_in,
        live_docs,
        &nested.inner,
        doc,
    )? {
        return Ok(Explanation::no_match("no matching clause"));
    }
    Ok(Explanation::match_(
        nested.score,
        format!(
            "{} = ConstantScore, discarding the wrapped clause's own score",
            nested.score
        ),
    ))
}

/// [`Clause::Boost`]'s explanation: mirrors real `BoostQuery.BoostWeight.
/// explain` -- matches iff the wrapped clause matches, scoring the wrapped
/// clause's own score multiplied by `boost`, same as [`crate::clause_scores`]'s
/// `Clause::Boost` arm (`inner.value * nested.boost`, same multiplication
/// order, so bit-for-bit identical).
#[allow(clippy::too_many_arguments)]
fn explain_boost(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    pos_in: Option<&PosInput<'_>>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    nested: &BoostQuery,
    doc: i32,
    norms: Option<&HashMap<String, FieldNorms<'_>>>,
) -> Result<Explanation> {
    let inner = explain_clause(
        fields,
        doc_in,
        pos_in,
        pay_in,
        live_docs,
        &nested.inner,
        doc,
        norms,
    )?;
    if !inner.matched {
        return Ok(Explanation::no_match("no matching clause"));
    }
    let value = inner.value * nested.boost;
    Ok(Explanation::match_(value, format!("{value} = product of:"))
        .with_details(vec![inner, Explanation::match_(nested.boost, "boost")]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::{
        BoostQuery, ConstantScoreQuery, DisjunctionMaxQuery, PrefixQuery, WildcardQuery,
    };
    use crate::{
        search_boolean_query_scored, search_disjunction_max_query_scored, search_term_query_scored,
        BooleanQuery, ScoringCollector, TermQuery,
    };
    use lucene_codecs::blocktree;

    /// Test-only collector capturing every `(doc_id, score)` pair in
    /// collection order -- this test module's ground truth to compare
    /// `explain_clause`'s reported value against (see this module's own doc
    /// comment on why bit-for-bit equality, not an epsilon comparison, is
    /// the correctness bar here).
    #[derive(Default)]
    struct ScoreCapture {
        scores: Vec<(i32, f32)>,
    }
    impl ScoringCollector for ScoreCapture {
        fn collect(&mut self, doc_id: i32, score: f32) {
            self.scores.push((doc_id, score));
        }
    }

    fn open_fixture() -> (BlockTreeFields, Option<DocInputOwned>) {
        let dir = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/blocktree_index/"
        );
        let manifest = std::fs::read_to_string(format!("{dir}manifest.properties"))
            .expect("run fixtures generator first (GenBlockTree)");
        let get = |key: &str| -> String {
            manifest
                .lines()
                .find_map(|l| l.strip_prefix(&format!("{key}=")))
                .unwrap_or_else(|| panic!("manifest key {key} missing"))
                .to_string()
        };
        let id_hex = get("id_hex");
        let mut id = [0u8; 16];
        for (i, slot) in id.iter_mut().enumerate() {
            *slot = u8::from_str_radix(&id_hex[i * 2..i * 2 + 2], 16).unwrap();
        }
        let suffix = get("segment_suffix");
        let max_doc: i32 = get("max_doc").parse().unwrap();

        let read_raw = |name: &str| -> Vec<u8> {
            std::fs::read(format!("{dir}{name}.raw")).unwrap_or_else(|_| panic!("missing {name}"))
        };
        let fnm = read_raw(&get("fnm_file_name"));
        let field_infos = lucene_codecs::field_infos::parse(&fnm, &id, "").expect("parse .fnm");
        let tim = read_raw(&get("tim_file_name"));
        let tip = read_raw(&get("tip_file_name"));
        let tmd = read_raw(&get("tmd_file_name"));
        let fields = blocktree::open(&tim, &tip, &tmd, &field_infos, &id, &suffix, max_doc)
            .expect("open blocktree");
        let doc = read_raw(&get("doc_file_name"));
        let pos = read_raw(&get("pos_file_name"));
        let pay = read_raw(&get("pay_file_name"));
        (
            fields,
            Some(DocInputOwned {
                doc,
                pos,
                pay,
                id,
                suffix,
            }),
        )
    }

    struct DocInputOwned {
        doc: Vec<u8>,
        pos: Vec<u8>,
        pay: Vec<u8>,
        id: [u8; 16],
        suffix: String,
    }

    impl DocInputOwned {
        fn open(&self) -> DocInput<'_> {
            DocInput::open(&self.doc, &self.id, &self.suffix).expect("open .doc")
        }
        fn open_pos(&self) -> PosInput<'_> {
            PosInput::open(&self.pos, &self.id, &self.suffix).expect("open .pos")
        }
        fn open_pay(&self) -> PayInput<'_> {
            PayInput::open(&self.pay, &self.id, &self.suffix).expect("open .pay")
        }
    }

    #[test]
    fn points_range_clause_is_not_yet_explainable() {
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let clause = Clause::PointsRange(crate::query::PointsRangeQuery::new("body", 0, 100));
        let err = explain_clause(&fields, doc_in.as_ref(), None, None, None, &clause, 0, None)
            .unwrap_err();
        assert!(matches!(err, crate::Error::UnexecutablePointsRange(field) if field == "body"));
    }

    #[test]
    fn term_explain_matching_doc_equals_scored_search_output_exactly() {
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let query = TermQuery::new("body", "cat");

        let mut capture = ScoreCapture::default();
        search_term_query_scored(&fields, doc_in.as_ref(), None, &query, None, &mut capture)
            .unwrap();
        assert!(
            !capture.scores.is_empty(),
            "fixture must have a doc matching body:cat"
        );
        let (target_doc, expected_score) = capture.scores[0];

        let explanation = explain_clause(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            &Clause::Term(query),
            target_doc,
            None,
        )
        .unwrap();

        assert!(explanation.matched);
        assert_eq!(explanation.value, expected_score);

        // The nested idf/tfNorm sub-values must themselves multiply back to
        // the same top-level value -- not just an equal top-level number by
        // coincidence.
        let score_node = &explanation.details[0];
        let idf_node = &score_node.details[0];
        let tf_norm_node = &score_node.details[1];
        assert_eq!(idf_node.value * tf_norm_node.value, expected_score);
        assert_eq!(score_node.value, expected_score);
    }

    /// Confirms `explain_clause` has no interaction whatsoever with
    /// `crate::search_term_query_scored_maxscore`'s MAXSCORE-skip path (task
    /// #135): `explain_term` only ever calls `crate::term_doc_freqs` (the
    /// eager, non-pruning path this module's top doc comment already
    /// documents), so it can correctly explain a doc that a MAXSCORE search
    /// over the *same term* would have safely skipped decoding entirely once
    /// its `TopDocsCollector` filled up. This uses the same "big"/"everywhere"
    /// fixture term (`docFreq == 300`, real impacts) as
    /// `lib.rs`'s `maxscore_lazy_path_matches_eager_path_on_real_fixture_and_actually_skips_blocks`:
    /// a `top_n = 1` MAXSCORE search only ever collects/returns its single
    /// best-scoring doc, skipping real-Lucene-block decode for the rest of
    /// the term's docs -- yet `explain_clause` asked to explain one of those
    /// *not*-returned docs still produces the exact score
    /// `search_term_query_scored` (the eager path) computes for it,
    /// unaffected by the unrelated search having pruned blocks. This is not a
    /// bug fix (`explain_clause` never touches the maxscore path in the first
    /// place, by construction), just a regression test making that
    /// non-interaction explicit and permanent.
    #[test]
    fn explain_is_unaffected_by_an_unrelated_maxscore_search_pruning_the_same_term() {
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let query = TermQuery::new("big", b"everywhere".as_slice());

        // Eager, ground-truth per-doc scores for every doc matching this term.
        let mut eager = ScoreCapture::default();
        search_term_query_scored(&fields, doc_in.as_ref(), None, &query, None, &mut eager).unwrap();
        assert!(
            eager.scores.len() > 1,
            "fixture term must match more than one doc for this test to be meaningful"
        );

        // A top_n=1 MAXSCORE search over the identical term/doc_in/norms
        // returns only its single best doc.
        let mut maxscore = crate::TopDocsCollector::new(1);
        crate::search_term_query_scored_maxscore(
            &fields,
            doc_in.as_ref(),
            None,
            &query,
            None,
            &mut maxscore,
        )
        .unwrap();
        let kept_docs: std::collections::HashSet<i32> = maxscore
            .top_docs()
            .iter()
            .map(|score_doc| score_doc.doc_id)
            .collect();
        assert_eq!(kept_docs.len(), 1, "top_n=1 keeps exactly one doc");

        // Pick a doc the maxscore search did NOT keep (and thus, per its own
        // design, may not have even decoded) and confirm explain_clause still
        // reports the correct eager score for it.
        let (pruned_doc, expected_score) = *eager
            .scores
            .iter()
            .find(|(doc_id, _)| !kept_docs.contains(doc_id))
            .expect("at least one doc must have been pruned by top_n=1");

        let explanation = explain_clause(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            &Clause::Term(query),
            pruned_doc,
            None,
        )
        .unwrap();

        assert!(explanation.matched);
        assert_eq!(explanation.value, expected_score);
    }

    /// Builds a synthetic, dense [`FieldNorms`] covering every doc up to
    /// `max_doc` with the same norm byte -- exercises the "real opened
    /// norms" (`Some(fn_)`) branch in [`explain_term`]/[`explain_phrase`],
    /// which the fixture's own real `.nvd`/`.nvm` files aren't wired up for
    /// in this crate's existing tests (same synthetic-entry pattern
    /// `field_norms.rs`'s own unit tests already use).
    fn synthetic_norms(data: &[u8], max_doc: i32) -> FieldNorms<'_> {
        let entry = lucene_codecs::norms::NormsEntry {
            field_number: 0,
            docs_with_field_offset: -1,
            docs_with_field_length: 0,
            jump_table_entry_count: 0,
            dense_rank_power: 0,
            num_docs_with_field: max_doc,
            bytes_per_norm: 1,
            norms_offset: 0,
        };
        FieldNorms::open(data, entry, max_doc, None).unwrap()
    }

    #[test]
    fn term_explain_with_real_norms_matches_scored_search_output_exactly() {
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let query = TermQuery::new("body", "cat");

        let mut capture = ScoreCapture::default();
        search_term_query_scored(&fields, doc_in.as_ref(), None, &query, None, &mut capture)
            .unwrap();
        let (target_doc, _) = capture.scores[0];

        // Norms must cover every doc the term matches, not just `target_doc`
        // -- `search_term_query_scored` computes every matched doc's field
        // length eagerly, in doc-ID order.
        let max_doc = capture.scores.iter().map(|&(d, _)| d).max().unwrap() + 1;
        let data = vec![10u8; max_doc as usize];
        let norms = synthetic_norms(&data, max_doc);

        let mut capture_normed = ScoreCapture::default();
        search_term_query_scored(
            &fields,
            doc_in.as_ref(),
            None,
            &query,
            Some(&norms),
            &mut capture_normed,
        )
        .unwrap();
        let (_, expected_score) = capture_normed
            .scores
            .into_iter()
            .find(|&(d, _)| d == target_doc)
            .unwrap();

        let explanation = explain_clause(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            &Clause::Term(query),
            target_doc,
            Some(&HashMap::from([("body".to_string(), norms)])),
        )
        .unwrap();
        assert!(explanation.matched);
        assert_eq!(explanation.value, expected_score);
    }

    #[test]
    fn term_explain_non_matching_doc_is_no_match_with_zero_value() {
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let query = TermQuery::new("body", "cat");

        // doc 999999 is far outside this small fixture's doc range.
        let explanation = explain_clause(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            &Clause::Term(query),
            999_999,
            None,
        )
        .unwrap();
        assert!(!explanation.matched);
        assert_eq!(explanation.value, 0.0);
    }

    #[test]
    fn term_explain_missing_field_is_no_match() {
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let explanation = explain_clause(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            &Clause::Term(TermQuery::new("nonexistent", "x")),
            0,
            None,
        )
        .unwrap();
        assert!(!explanation.matched);
        assert_eq!(explanation.value, 0.0);
    }

    #[test]
    fn term_explain_missing_term_is_no_match() {
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let explanation = explain_clause(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            &Clause::Term(TermQuery::new("body", "zzz-missing")),
            0,
            None,
        )
        .unwrap();
        assert!(!explanation.matched);
    }

    #[test]
    fn boolean_explain_matching_doc_equals_scored_search_output_exactly() {
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let query = BooleanQuery::new()
            .with_must([TermQuery::new("body", "cat")])
            .with_should([TermQuery::new("body", "dog")]);

        let mut capture = ScoreCapture::default();
        search_boolean_query_scored(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            &query,
            None,
            &mut capture,
        )
        .unwrap();
        assert!(!capture.scores.is_empty());
        let (target_doc, expected_score) = capture.scores[0];

        let explanation = explain_clause(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            &Clause::Boolean(Box::new(query)),
            target_doc,
            None,
        )
        .unwrap();
        assert!(explanation.matched);
        assert_eq!(explanation.value, expected_score);

        // Sub-clause explanation values must sum to the top-level value.
        let sum: f32 = explanation.details.iter().map(|d| d.value).sum();
        assert_eq!(sum, expected_score);
    }

    #[test]
    fn boolean_explain_no_match_when_must_clause_fails() {
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let query = BooleanQuery::new().with_must([
            TermQuery::new("body", "cat"),
            TermQuery::new("body", "zzz-missing"),
        ]);
        let explanation = explain_clause(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            &Clause::Boolean(Box::new(query)),
            0,
            None,
        )
        .unwrap();
        assert!(!explanation.matched);
        assert_eq!(explanation.value, 0.0);
    }

    #[test]
    fn boolean_explain_no_match_for_empty_query() {
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let explanation = explain_clause(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            &Clause::Boolean(Box::new(BooleanQuery::new())),
            0,
            None,
        )
        .unwrap();
        assert!(!explanation.matched);
    }

    #[test]
    fn boolean_explain_no_match_for_pure_must_not_query() {
        // Task #60 edge case: a `BooleanQuery` with only `must_not` clauses
        // (no `must`/`should`) matches nothing -- `matched_boolean_docs` already
        // folds this into the same "no must/should clauses" `Ok(None)` case an
        // entirely empty query hits (see that function's doc comment), so
        // `explain_boolean` must report a no-match here too, not "everything
        // except the excluded set."
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let query = BooleanQuery::new().with_must_not([TermQuery::new("body", "dog")]);
        // Doc 2 doesn't contain "dog" at all -- if pure must_not were buggily
        // treated as "match everything except the excluded set," this doc would
        // wrongly explain as a match.
        let explanation = explain_clause(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            &Clause::Boolean(Box::new(query)),
            2,
            None,
        )
        .unwrap();
        assert!(!explanation.matched);
        assert_eq!(explanation.value, 0.0);
    }

    #[test]
    fn boolean_explain_no_match_when_minimum_should_match_exceeds_should_clause_count() {
        // Task #60 edge case: `minimum_should_match` greater than the number of
        // `should` clauses can never be satisfied -- must explain as no-match,
        // not panic or (worse) accidentally match.
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let query = BooleanQuery::new()
            .with_should([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")])
            .with_minimum_should_match(5);
        let explanation = explain_clause(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            &Clause::Boolean(Box::new(query)),
            0,
            None,
        )
        .unwrap();
        assert!(!explanation.matched);
        assert_eq!(explanation.value, 0.0);
    }

    #[test]
    fn boolean_explain_duplicate_should_clause_sums_twice_matching_scored_search() {
        // Task #60 edge case: a duplicated `should` clause must contribute its
        // score twice in the explanation, exactly matching
        // `search_boolean_query_scored`'s own double-counting (real Lucene does
        // not dedupe clauses -- see the `lib.rs` regression test
        // `boolean_duplicate_should_clause_counts_and_scores_twice` for the
        // full rationale). Verified here by requiring bit-for-bit equality
        // against the scored search path, the same technique
        // `boolean_explain_matching_doc_equals_scored_search_output_exactly`
        // already uses.
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let query = BooleanQuery::new()
            .with_should([TermQuery::new("body", "cat"), TermQuery::new("body", "cat")]);

        let mut capture = ScoreCapture::default();
        search_boolean_query_scored(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            &query,
            None,
            &mut capture,
        )
        .unwrap();
        assert!(!capture.scores.is_empty());
        let (target_doc, expected_score) = capture.scores[0];

        let explanation = explain_clause(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            &Clause::Boolean(Box::new(query)),
            target_doc,
            None,
        )
        .unwrap();
        assert!(explanation.matched);
        assert_eq!(explanation.value, expected_score);
        // Two should-clause detail entries, both for "cat", summing to the
        // duplicated total -- not deduplicated to one.
        assert_eq!(explanation.details.len(), 2);
    }

    #[test]
    fn dismax_explain_matching_doc_equals_scored_search_output_exactly() {
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let query = DisjunctionMaxQuery::new(
            [
                Clause::Term(TermQuery::new("body", "cat")),
                Clause::Term(TermQuery::new("body", "dog")),
            ],
            0.5,
        );

        let mut capture = ScoreCapture::default();
        search_disjunction_max_query_scored(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            &query,
            None,
            &mut capture,
        )
        .unwrap();
        assert!(!capture.scores.is_empty());
        let (target_doc, expected_score) = capture.scores[0];

        let explanation = explain_clause(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            &Clause::DisjunctionMax(Box::new(query)),
            target_doc,
            None,
        )
        .unwrap();
        assert!(explanation.matched);
        assert_eq!(explanation.value, expected_score);
    }

    #[test]
    fn dismax_explain_no_disjuncts_is_no_match() {
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let query = DisjunctionMaxQuery::new(Vec::<Clause>::new(), 0.0);
        let explanation = explain_clause(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            &Clause::DisjunctionMax(Box::new(query)),
            0,
            None,
        )
        .unwrap();
        assert!(!explanation.matched);
    }

    #[test]
    fn dismax_explain_no_matching_disjunct_is_no_match() {
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let query =
            DisjunctionMaxQuery::new([Clause::Term(TermQuery::new("body", "zzz-missing"))], 0.0);
        let explanation = explain_clause(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            &Clause::DisjunctionMax(Box::new(query)),
            0,
            None,
        )
        .unwrap();
        assert!(!explanation.matched);
    }

    #[test]
    fn constant_score_explain_matching_doc_reports_the_constant() {
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let query = ConstantScoreQuery::new(TermQuery::new("body", "cat"), 2.5);

        let mut capture = ScoreCapture::default();
        search_term_query_scored(
            &fields,
            doc_in.as_ref(),
            None,
            &TermQuery::new("body", "cat"),
            None,
            &mut capture,
        )
        .unwrap();
        let (target_doc, _) = capture.scores[0];

        let explanation = explain_clause(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            &Clause::ConstantScore(Box::new(query)),
            target_doc,
            None,
        )
        .unwrap();
        assert!(explanation.matched);
        assert_eq!(explanation.value, 2.5);
    }

    #[test]
    fn constant_score_explain_non_matching_doc_is_no_match() {
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let query = ConstantScoreQuery::new(TermQuery::new("body", "zzz-missing"), 2.5);
        let explanation = explain_clause(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            &Clause::ConstantScore(Box::new(query)),
            0,
            None,
        )
        .unwrap();
        assert!(!explanation.matched);
        assert_eq!(explanation.value, 0.0);
    }

    #[test]
    fn boost_explain_matching_doc_multiplies_inner_score() {
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());

        let mut capture = ScoreCapture::default();
        search_term_query_scored(
            &fields,
            doc_in.as_ref(),
            None,
            &TermQuery::new("body", "cat"),
            None,
            &mut capture,
        )
        .unwrap();
        let (target_doc, inner_score) = capture.scores[0];

        let query = BoostQuery::new(TermQuery::new("body", "cat"), 3.0);
        let explanation = explain_clause(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            &Clause::Boost(Box::new(query)),
            target_doc,
            None,
        )
        .unwrap();
        assert!(explanation.matched);
        assert_eq!(explanation.value, inner_score * 3.0);
    }

    #[test]
    fn boost_explain_non_matching_doc_is_no_match() {
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let query = BoostQuery::new(TermQuery::new("body", "zzz-missing"), 3.0);
        let explanation = explain_clause(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            &Clause::Boost(Box::new(query)),
            0,
            None,
        )
        .unwrap();
        assert!(!explanation.matched);
    }

    #[test]
    fn wildcard_explain_matching_doc_is_flat_constant_score() {
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let matched = crate::wildcard_doc_ids(
            &fields,
            doc_in.as_ref(),
            None,
            &WildcardQuery::new("body", "ca*"),
        )
        .unwrap();
        assert!(!matched.is_empty());
        let explanation = explain_clause(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            &Clause::Wildcard(WildcardQuery::new("body", "ca*")),
            matched[0],
            None,
        )
        .unwrap();
        assert!(explanation.matched);
        assert_eq!(explanation.value, 1.0);
    }

    #[test]
    fn wildcard_explain_non_matching_doc_is_no_match() {
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let explanation = explain_clause(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            &Clause::Wildcard(WildcardQuery::new("body", "zzz-nomatch*")),
            0,
            None,
        )
        .unwrap();
        assert!(!explanation.matched);
        assert_eq!(explanation.value, 0.0);
    }

    #[test]
    fn prefix_explain_matching_doc_is_flat_constant_score() {
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let matched = crate::prefix_doc_ids(
            &fields,
            doc_in.as_ref(),
            None,
            &PrefixQuery::new("body", "ca"),
        )
        .unwrap();
        assert!(!matched.is_empty());
        let explanation = explain_clause(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            &Clause::Prefix(PrefixQuery::new("body", "ca")),
            matched[0],
            None,
        )
        .unwrap();
        assert!(explanation.matched);
        assert_eq!(explanation.value, 1.0);
    }

    #[test]
    fn fuzzy_explain_matching_and_non_matching_docs() {
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let query = crate::FuzzyQuery::new("body", "cat");
        let matched = crate::fuzzy_doc_ids(&fields, doc_in.as_ref(), None, &query).unwrap();
        assert!(!matched.is_empty());

        let hit = explain_clause(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            &Clause::Fuzzy(query.clone()),
            matched[0],
            None,
        )
        .unwrap();
        assert!(hit.matched);
        assert_eq!(hit.value, 1.0);

        let miss = explain_clause(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            &Clause::Fuzzy(crate::FuzzyQuery::new("body", "zzzzzzzzzz")),
            0,
            None,
        )
        .unwrap();
        assert!(!miss.matched);
        assert_eq!(miss.value, 0.0);
    }

    #[test]
    fn regexp_explain_matching_and_non_matching_docs() {
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let query = crate::RegexpQuery::new("body", "ca.*");
        let matched = crate::regexp_doc_ids(&fields, doc_in.as_ref(), None, &query).unwrap();
        assert!(!matched.is_empty());

        let hit = explain_clause(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            &Clause::Regexp(query),
            matched[0],
            None,
        )
        .unwrap();
        assert!(hit.matched);
        assert_eq!(hit.value, 1.0);

        let miss = explain_clause(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            &Clause::Regexp(crate::RegexpQuery::new("body", "zzz-nomatch.*")),
            0,
            None,
        )
        .unwrap();
        assert!(!miss.matched);
    }

    #[test]
    fn span_explain_matching_and_non_matching_docs() {
        let (fields, doc) = open_fixture();
        let doc = doc.expect("fixture has an opened .doc file");
        let doc_in = doc.open();
        let pos_in = doc.open_pos();
        let pay_in = doc.open_pay();
        let query = crate::SpanQuery::span_term("pos", "alpha");
        let matched = crate::span_doc_ids(
            &fields,
            Some(&doc_in),
            Some(&pos_in),
            Some(&pay_in),
            None,
            &query,
        )
        .unwrap();
        assert!(!matched.is_empty());

        let hit = explain_clause(
            &fields,
            Some(&doc_in),
            Some(&pos_in),
            Some(&pay_in),
            None,
            &Clause::Span(query),
            matched[0],
            None,
        )
        .unwrap();
        assert!(hit.matched);
        assert_eq!(hit.value, 1.0);

        let miss_query = crate::SpanQuery::span_term("pos", "zzz-missing");
        let miss = explain_clause(
            &fields,
            Some(&doc_in),
            Some(&pos_in),
            Some(&pay_in),
            None,
            &Clause::Span(miss_query),
            0,
            None,
        )
        .unwrap();
        assert!(!miss.matched);
    }

    #[test]
    fn phrase_explain_multi_term_matching_doc_equals_scored_search_output_exactly() {
        let (fields, doc) = open_fixture();
        let doc = doc.expect("fixture has an opened .doc file");
        let doc_in = doc.open();
        let pos_in = doc.open_pos();
        let pay_in = doc.open_pay();

        // "alpha beta" is known (see `lib.rs`'s own phrase tests) to match
        // real doc 8555 in the "pos" field exactly, at slop 0.
        let query = crate::PhraseQuery::new("pos", ["alpha", "beta"]);

        let mut capture = ScoreCapture::default();
        crate::search_phrase_query_scored(
            &fields,
            Some(&doc_in),
            Some(&pos_in),
            Some(&pay_in),
            None,
            &query,
            None,
            &mut capture,
        )
        .unwrap();
        assert!(
            !capture.scores.is_empty(),
            "fixture must have a doc matching \"alpha beta\""
        );
        let (target_doc, expected_score) = capture.scores[0];

        let explanation = explain_clause(
            &fields,
            Some(&doc_in),
            Some(&pos_in),
            Some(&pay_in),
            None,
            &Clause::Phrase(query),
            target_doc,
            None,
        )
        .unwrap();
        assert!(explanation.matched);
        assert_eq!(explanation.value, expected_score);

        // The nested idf/tfNorm sub-values must themselves multiply back to
        // the same top-level value.
        let score_node = &explanation.details[0];
        let idf_node = &score_node.details[0];
        let tf_norm_node = &score_node.details[1];
        assert_eq!(idf_node.value * tf_norm_node.value, expected_score);
    }

    /// Closes a coverage gap the field norms/sloppy-match branches of
    /// `explain_phrase` weren't exercised by: `slop != 0` and the real-norms
    /// path. Reuses `GenBlockTree.java`'s known sloppy-gap fixture doc
    /// (`"alpha"@0`, `"beta"@3` -- a real 2-move gap, task #28) with a real
    /// `FieldNorms` built the same way `term_explain_with_real_norms_...`
    /// does for `Clause::Term`, at a slop large enough to match.
    #[test]
    fn phrase_explain_sloppy_match_with_real_norms_equals_scored_search_output_exactly() {
        let (fields, doc) = open_fixture();
        let doc = doc.expect("fixture has an opened .doc file");
        let doc_in = doc.open();
        let pos_in = doc.open_pos();
        let pay_in = doc.open_pay();

        let dir = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/blocktree_index/"
        );
        let manifest = std::fs::read_to_string(format!("{dir}manifest.properties")).unwrap();
        let get = |key: &str| -> String {
            manifest
                .lines()
                .find_map(|l| l.strip_prefix(&format!("{key}=")))
                .unwrap_or_else(|| panic!("manifest key {key} missing"))
                .to_string()
        };
        let gap_doc: i32 = get("field.pos.sloppyGapDoc").parse().unwrap();
        let moves_needed: u32 = get("field.pos.sloppyGap.movesNeeded").parse().unwrap();
        assert_eq!(
            moves_needed, 2,
            "fixture's known gap size changed underneath this test"
        );

        let query = crate::PhraseQuery::new("pos", ["alpha", "beta"]).with_slop(moves_needed);

        let max_doc = gap_doc + 1;
        let data = vec![10u8; max_doc as usize];
        let norms = synthetic_norms(&data, max_doc);

        let mut capture = ScoreCapture::default();
        crate::search_phrase_query_scored(
            &fields,
            Some(&doc_in),
            Some(&pos_in),
            Some(&pay_in),
            None,
            &query,
            Some(&norms),
            &mut capture,
        )
        .unwrap();
        let (target_doc, expected_score) = *capture
            .scores
            .iter()
            .find(|(d, _)| *d == gap_doc)
            .unwrap_or_else(|| panic!("expected doc {gap_doc} to match at slop={moves_needed}"));

        let norms_map: HashMap<String, FieldNorms> = HashMap::from([("pos".to_string(), norms)]);
        let explanation = explain_clause(
            &fields,
            Some(&doc_in),
            Some(&pos_in),
            Some(&pay_in),
            None,
            &Clause::Phrase(query),
            target_doc,
            Some(&norms_map),
        )
        .unwrap();
        assert!(explanation.matched);
        assert_eq!(explanation.value, expected_score);
    }

    #[test]
    fn phrase_explain_no_alignment_is_no_match() {
        let (fields, doc) = open_fixture();
        let doc = doc.expect("fixture has an opened .doc file");
        let doc_in = doc.open();
        let pos_in = doc.open_pos();
        let pay_in = doc.open_pay();

        // Reversed order ("beta alpha") never aligns in this fixture.
        let query = crate::PhraseQuery::new("pos", ["beta", "alpha"]);
        let explanation = explain_clause(
            &fields,
            Some(&doc_in),
            Some(&pos_in),
            Some(&pay_in),
            None,
            &Clause::Phrase(query),
            8555,
            None,
        )
        .unwrap();
        assert!(!explanation.matched);
    }

    #[test]
    fn phrase_explain_missing_field_is_no_match() {
        let (fields, doc) = open_fixture();
        let doc = doc.expect("fixture has an opened .doc file");
        let doc_in = doc.open();
        let pos_in = doc.open_pos();
        let pay_in = doc.open_pay();
        let query = crate::PhraseQuery::new("nonexistent", ["a", "b"]);
        let explanation = explain_clause(
            &fields,
            Some(&doc_in),
            Some(&pos_in),
            Some(&pay_in),
            None,
            &Clause::Phrase(query),
            0,
            None,
        )
        .unwrap();
        assert!(!explanation.matched);
    }

    #[test]
    fn phrase_explain_missing_term_in_multi_term_phrase_is_no_match() {
        let (fields, doc) = open_fixture();
        let doc = doc.expect("fixture has an opened .doc file");
        let doc_in = doc.open();
        let pos_in = doc.open_pos();
        let pay_in = doc.open_pay();
        let query = crate::PhraseQuery::new("pos", ["alpha", "zzz-missing"]);
        let explanation = explain_clause(
            &fields,
            Some(&doc_in),
            Some(&pos_in),
            Some(&pay_in),
            None,
            &Clause::Phrase(query),
            8555,
            None,
        )
        .unwrap();
        assert!(!explanation.matched);
    }

    #[test]
    fn phrase_explain_empty_terms_is_no_match() {
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let explanation = explain_clause(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            &Clause::Phrase(crate::PhraseQuery::default()),
            0,
            None,
        )
        .unwrap();
        assert!(!explanation.matched);
    }

    #[test]
    fn phrase_explain_single_term_delegates_to_term_explain() {
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let query = crate::PhraseQuery::new("body", ["cat"]);

        let mut capture = ScoreCapture::default();
        search_term_query_scored(
            &fields,
            doc_in.as_ref(),
            None,
            &TermQuery::new("body", "cat"),
            None,
            &mut capture,
        )
        .unwrap();
        let (target_doc, expected_score) = capture.scores[0];

        let explanation = explain_clause(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            &Clause::Phrase(query),
            target_doc,
            None,
        )
        .unwrap();
        assert!(explanation.matched);
        assert_eq!(explanation.value, expected_score);
    }

    #[test]
    fn phrase_explain_missing_multi_term_without_pos_input_is_an_error() {
        let (fields, _doc) = open_fixture();
        let query = crate::PhraseQuery::new("body", ["quick", "fox"]);
        let err = explain_clause(
            &fields,
            None,
            None,
            None,
            None,
            &Clause::Phrase(query),
            0,
            None,
        )
        .unwrap_err();
        assert!(matches!(err, crate::Error::MissingPosInput));
    }

    #[test]
    fn explanation_match_and_no_match_constructors() {
        let m = Explanation::match_(1.5, "matched")
            .with_details(vec![Explanation::match_(1.0, "child")]);
        assert!(m.matched);
        assert_eq!(m.value, 1.5);
        assert_eq!(m.details.len(), 1);

        let nm = Explanation::no_match("nope");
        assert!(!nm.matched);
        assert_eq!(nm.value, 0.0);
        assert!(nm.details.is_empty());
    }
}
