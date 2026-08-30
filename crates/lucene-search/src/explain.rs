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
//! Every `description` string this module emits is real Lucene 10.5.0's
//! verbatim (`"weight(field:term in doc) [BM25Similarity], result of:"`,
//! `"score(freq=..), computed as boost * idf * tf from:"`, `"idf, computed as
//! log(1 + (N - n + 0.5) / (n + 0.5)) from:"`, `"sum of:"`, `"max of:"`, ...)
//! -- downstream tooling parses these, so they are a compatibility contract,
//! not prose. [`Explanation`]'s [`std::fmt::Display`] is likewise a port of
//! `Explanation.toString()`'s two-space-per-level indented rendering.
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

impl std::fmt::Display for Explanation {
    /// Port of real `Explanation.toString()`: one line per node,
    /// `"{value} = {description}"`, each nesting level indented by two
    /// spaces, every line (including the last) terminated by `\n` -- the
    /// exact rendering `IndexSearcher.explain(...).toString()` produces and
    /// that downstream tooling parses.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.fmt_at_depth(f, 0)
    }
}

impl Explanation {
    fn fmt_at_depth(&self, f: &mut std::fmt::Formatter<'_>, depth: usize) -> std::fmt::Result {
        for _ in 0..depth {
            f.write_str("  ")?;
        }
        writeln!(f, "{} = {}", java_float(self.value), self.description)?;
        for detail in &self.details {
            detail.fmt_at_depth(f, depth + 1)?;
        }
        Ok(())
    }
}

/// Renders `v` the way Java's `Float.toString`/`String.valueOf(float)` does
/// -- shortest representation that round-trips, but **always** with a decimal
/// point (`2` renders as `"2.0"`, not `"2"`). Rust's own `Display` for `f32`
/// drops the trailing `.0`, which would make every `score(freq=2.0)`-style
/// description in this module differ from real Lucene's by one character;
/// `{:?}` keeps it, and matches Java's shortest-round-trip choice for every
/// value a score/idf/tf can take.
fn java_float(v: f32) -> String {
    format!("{v:?}")
}

/// Renders `clause` the way real Lucene's `Query.toString()` does for the
/// corresponding Java query class -- the string every `ConstantScoreWeight`/
/// `BooleanWeight` explanation embeds (`"no match on required clause (" +
/// c.query() + ")"`, `getQuery().toString() + " doesn't match id " + doc`,
/// ...). Real Lucene's `toString(String field)` omits the `field:` prefix
/// when the field equals the enclosing "default" field; explanations always
/// call the no-argument `toString()`, i.e. default field `""`, so every
/// field is printed.
fn describe_clause(clause: &Clause) -> String {
    fn term(bytes: &[u8]) -> String {
        String::from_utf8_lossy(bytes).into_owned()
    }
    match clause {
        Clause::Term(q) => format!("{}:{}", q.field, term(&q.term)),
        Clause::Phrase(q) => {
            let body = q
                .terms
                .iter()
                .map(|t| term(t))
                .collect::<Vec<_>>()
                .join(" ");
            let slop = if q.slop == 0 {
                String::new()
            } else {
                format!("~{}", q.slop)
            };
            format!("{}:\"{body}\"{slop}", q.field)
        }
        Clause::MultiPhrase(q) => {
            let body = q
                .term_arrays
                .iter()
                .map(|alts| {
                    if alts.len() == 1 {
                        term(&alts[0])
                    } else {
                        format!(
                            "({})",
                            alts.iter().map(|t| term(t)).collect::<Vec<_>>().join(" ")
                        )
                    }
                })
                .collect::<Vec<_>>()
                .join(" ");
            let slop = if q.slop == 0 {
                String::new()
            } else {
                format!("~{}", q.slop)
            };
            format!("{}:\"{body}\"{slop}", q.field)
        }
        Clause::Boolean(q) => {
            let mut parts = Vec::new();
            for c in &q.must {
                parts.push(format!("+{}", describe_clause(c)));
            }
            // `Occur.FILTER.toString()` is `"#"`.
            for c in &q.filter {
                parts.push(format!("#{}", describe_clause(c)));
            }
            for c in &q.should {
                parts.push(describe_clause(c));
            }
            for c in &q.must_not {
                parts.push(format!("-{}", describe_clause(c)));
            }
            let mm = if q.minimum_should_match == 0 {
                String::new()
            } else {
                format!("~{}", q.minimum_should_match)
            };
            format!("({}){mm}", parts.join(" "))
        }
        Clause::DisjunctionMax(q) => {
            let body = q
                .disjuncts
                .iter()
                .map(describe_clause)
                .collect::<Vec<_>>()
                .join(" | ");
            let tie = if q.tie_breaker == 0.0 {
                String::new()
            } else {
                format!("~{}", java_float(q.tie_breaker))
            };
            format!("({body}){tie}")
        }
        Clause::ConstantScore(q) => format!("ConstantScore({})", describe_clause(&q.inner)),
        Clause::Boost(q) => format!("({})^{}", describe_clause(&q.inner), java_float(q.boost)),
        Clause::Wildcard(q) => format!("{}:{}", q.field, term(&q.pattern)),
        Clause::Prefix(q) => format!("{}:{}*", q.field, term(&q.prefix)),
        Clause::Fuzzy(q) => format!("{}:{}~{}", q.field, term(&q.term), q.max_edits),
        Clause::Regexp(q) => format!("{}:/{}/", q.field, q.pattern),
        Clause::Span(q) => describe_span(q),
        Clause::PointsRange(q) => format!("{}:[{} TO {}]", q.field, q.min, q.max),
        Clause::MatchAllDocs(_) => "*:*".to_string(),
        Clause::MatchNoDocs(q) => format!("MatchNoDocsQuery(\"{}\")", q.reason),
        Clause::TermInSet(q) => format!(
            "{}:({})",
            q.field,
            q.terms
                .iter()
                .map(|t| term(t))
                .collect::<Vec<_>>()
                .join(" ")
        ),
    }
}

/// `describe_clause`'s [`crate::query::SpanQuery`] arm, mirroring real
/// `SpanTermQuery`/`SpanNearQuery`/`SpanOrQuery`'s own `toString`.
fn describe_span(span: &crate::query::SpanQuery) -> String {
    use crate::query::SpanQuery;
    match span {
        SpanQuery::SpanTerm { field, term } => {
            format!("spanTerm({field}:{})", String::from_utf8_lossy(term))
        }
        SpanQuery::SpanNear {
            clauses,
            slop,
            in_order,
        } => format!(
            "spanNear([{}], {slop}, {in_order})",
            clauses
                .iter()
                .map(describe_span)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        SpanQuery::SpanOr { clauses } => format!(
            "spanOr([{}])",
            clauses
                .iter()
                .map(describe_span)
                .collect::<Vec<_>>()
                .join(", ")
        ),
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
            Ok(explain_flat_match(matched, clause, doc))
        }
        Clause::Prefix(query) => {
            let matched = crate::prefix_doc_ids(fields, doc_in, live_docs, query)?.contains(&doc);
            Ok(explain_flat_match(matched, clause, doc))
        }
        Clause::Fuzzy(query) => {
            let matched = crate::fuzzy_doc_ids(fields, doc_in, live_docs, query)?.contains(&doc);
            Ok(explain_flat_match(matched, clause, doc))
        }
        Clause::Regexp(query) => {
            let matched = crate::regexp_doc_ids(fields, doc_in, live_docs, query)?.contains(&doc);
            Ok(explain_flat_match(matched, clause, doc))
        }
        Clause::Span(query) => {
            let matched = crate::span_doc_ids(fields, doc_in, pos_in, pay_in, live_docs, query)?
                .contains(&doc);
            Ok(explain_flat_match(matched, clause, doc))
        }
        Clause::PointsRange(query) => Err(crate::Error::MissingPointsInput(query.field.clone())),
        Clause::MatchAllDocs(query) => {
            let matched = crate::match_all_doc_ids(live_docs, query.max_doc).contains(&doc);
            Ok(explain_flat_match(matched, clause, doc))
        }
        Clause::MatchNoDocs(_) => Ok(Explanation::no_match(format!(
            "{} doesn't match id {doc}",
            describe_clause(clause)
        ))),
        Clause::MultiPhrase(query) => {
            // No per-position breakdown yet (real `MultiPhraseWeight.explain`
            // has one): this reports the real score the scorer produces for
            // this doc, obtained from the scorer itself so the two can never
            // disagree, rather than a re-derivation that could drift.
            let mut hit: Option<f32> = None;
            struct Pick<'a> {
                doc: i32,
                out: &'a mut Option<f32>,
            }
            impl crate::collector::ScoringCollector for Pick<'_> {
                fn collect(&mut self, doc_id: i32, score: f32) {
                    if doc_id == self.doc {
                        *self.out = Some(score);
                    }
                }
            }
            let mut pick = Pick { doc, out: &mut hit };
            crate::search_multi_phrase_query_scored(
                fields,
                doc_in,
                pos_in,
                pay_in,
                live_docs,
                query,
                norms.and_then(|m| m.get(&query.field)),
                &mut pick,
            )?;
            Ok(match hit {
                Some(score) => Explanation::match_(
                    score,
                    format!(
                        "weight({} in {doc}) [BM25Similarity], result of:",
                        describe_clause(clause)
                    ),
                ),
                // `MultiPhraseWeight` inherits `PhraseWeight.explain`, whose
                // no-match description is exactly `"no matching terms"`.
                None => Explanation::no_match("no matching terms"),
            })
        }
        Clause::TermInSet(query) => {
            let matched =
                crate::term_in_set_doc_ids(fields, doc_in, live_docs, query)?.contains(&doc);
            Ok(explain_flat_match(matched, clause, doc))
        }
    }
}

/// A leaf clause with no per-term breakdown (`Wildcard`/`Prefix`/`Fuzzy`/
/// `Regexp`/`Span` — see this module's doc comment): matches score exactly
/// `1.0` (same flat constant every `clause_scores` arm for these variants
/// already reports), non-matches are a clean `no_match` at `0.0`.
///
/// The descriptions are real `ConstantScoreWeight.explain`'s verbatim --
/// `getQuery().toString()` on a match, `getQuery() + " doesn't match id " +
/// doc` on a non-match -- since every one of these clauses is a
/// `MultiTermQuery`/`SpanQuery` that real Lucene rewrites to a
/// constant-scoring weight before explaining it.
fn explain_flat_match(matched: bool, clause: &Clause, doc: i32) -> Explanation {
    let described = describe_clause(clause);
    if matched {
        // `ConstantScoreWeight.explain`: `Explanation.match(score,
        // getQuery().toString() + (score == 1f ? "" : "^" + score))` -- these
        // clauses always score a flat `1.0` here, so the `^score` suffix is
        // never appended.
        Explanation::match_(1.0, described)
    } else {
        Explanation::no_match(format!("{described} doesn't match id {doc}"))
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
        crate::resolve_clause_docs(fields, doc_in, pos_in, pay_in, live_docs, None, clause)?
            .contains(&doc),
    )
}

/// [`Clause::Term`]'s explanation: mirrors real `TermWeight.explain`/
/// `BM25Scorer.explain` — a `weight(field:term in doc) [BM25Similarity]` node
/// wrapping a `score(freq=...)` node, itself wrapping `idf` (with `n`/`N`
/// leaf details) and `tf` (with `freq`/`k1`/`b`/`dl`/`avgdl` leaf details),
/// every description string verbatim from Java. `value` is computed via the exact same
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
        return Ok(Explanation::no_match("no matching term"));
    };
    let Some(stats) = field_terms.seek_exact(&query.term) else {
        return Ok(Explanation::no_match("no matching term"));
    };
    let doc_freqs = crate::term_doc_freqs(fields, doc_in, live_docs, query)?;
    let Some(&(_, freq)) = doc_freqs.iter().find(|&&(d, _)| d == doc) else {
        return Ok(Explanation::no_match("no matching term"));
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
    // `BM25Scorer.explain`/`explainTF` verbatim: the reported score is
    // `weight - weight / (1 + freq * normInverse)` -- the *same* expression the
    // scorer evaluates, which is why this stays bit-identical to
    // `search_term_query_scored`'s output -- while the `tf` sub-explanation is
    // Lucene's own `1f - 1f / (1 + freq * normInverse)`. Real Lucene notes that
    // it deliberately does not present this as a "product of", because the
    // rewrite introduces a rounding difference against `idf * tf`.
    let norm_inverse = similarity::norm_inverse(
        field_length,
        avg_field_length,
        similarity::DEFAULT_K1,
        similarity::DEFAULT_B,
    );
    let tf_norm = 1.0 - 1.0 / (1.0 + freq as f32 * norm_inverse);
    let value = similarity::do_score(idf, freq as f32, norm_inverse);

    let idf_explanation = idf_explanation(idf, stats.doc_freq as i64, doc_count);

    let tf_explanation = Explanation::match_(
        tf_norm,
        "tf, computed as freq / (freq + k1 * (1 - b + b * dl / avgdl)) from:",
    )
    .with_details(vec![
        Explanation::match_(freq as f32, "freq, occurrences of term within document"),
        Explanation::match_(similarity::DEFAULT_K1, "k1, term saturation parameter"),
        Explanation::match_(similarity::DEFAULT_B, "b, length normalization parameter"),
        Explanation::match_(field_length, dl_description(field_length)),
        Explanation::match_(avg_field_length, "avgdl, average length of field"),
    ]);

    let score_explanation = Explanation::match_(
        value,
        format!(
            "score(freq={}), computed as boost * idf * tf from:",
            java_float(freq as f32)
        ),
    )
    .with_details(vec![idf_explanation, tf_explanation]);

    Ok(Explanation::match_(
        value,
        format!(
            "weight({} in {doc}) [BM25Similarity], result of:",
            describe_clause(&Clause::Term(query.clone()))
        ),
    )
    .with_details(vec![score_explanation]))
}

/// One term's `idf` node, verbatim from `BM25Similarity.idfExplain(FieldStats,
/// TermStats)` -- the description and both leaf-detail descriptions are real
/// Lucene's exact strings (`N`/`n`, not `docCount`/`docFreq`).
fn idf_explanation(idf: f32, doc_freq: i64, doc_count: i64) -> Explanation {
    Explanation::match_(
        idf,
        "idf, computed as log(1 + (N - n + 0.5) / (n + 0.5)) from:",
    )
    .with_details(vec![
        Explanation::match_(doc_freq as f32, "n, number of documents containing term"),
        Explanation::match_(doc_count as f32, "N, total number of documents with field"),
    ])
}

/// `BM25Similarity.BM25Scorer.explainTF`'s `dl` leaf description: real Lucene
/// prints `"dl, length of field (approximate)"` when the *encoded* norm byte
/// is `> 39`, and `"dl, length of field"` otherwise. This port only carries
/// the already-decoded length, but `BM25Similarity.LENGTH_TABLE[i] == i` for
/// every `i <= 39` and the table is monotonically non-decreasing, so
/// `field_length > 39.0` is exactly the same predicate (asserted against the
/// real table in this module's tests).
fn dl_description(field_length: f32) -> &'static str {
    if field_length > 39.0 {
        "dl, length of field (approximate)"
    } else {
        "dl, length of field"
    }
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
        return Ok(Explanation::no_match("no matching terms"));
    };

    let doc_count = field_terms.doc_count as i64;
    let mut idf_sum = 0.0f32;
    let mut idf_details = Vec::with_capacity(query.terms.len());
    for term in &query.terms {
        let Some(stats) = field_terms.seek_exact(term) else {
            return Ok(Explanation::no_match("no matching terms"));
        };
        let term_idf = similarity::idf(stats.doc_freq as i64, doc_count);
        idf_sum += term_idf;
        idf_details.push(idf_explanation(term_idf, stats.doc_freq as i64, doc_count));
    }

    let mut per_term_docs: Vec<Vec<i32>> = Vec::with_capacity(query.terms.len());
    let mut per_term_maps: Vec<HashMap<i32, Vec<i32>>> = Vec::with_capacity(query.terms.len());
    for term in &query.terms {
        let Some((docs, positions, spans)) = crate::term_doc_positions(
            fields,
            doc_in,
            pos_in,
            pay_in,
            live_docs,
            &query.field,
            term,
        )?
        else {
            return Ok(Explanation::no_match("no matching terms"));
        };
        // Not a hot path (one document, on demand): rebuild the doc -> positions
        // map the explain code expects from the aligned vectors the phrase
        // reader now returns.
        per_term_maps.push(
            docs.iter()
                .copied()
                .zip(spans.iter())
                .map(|(d, &(start, end))| (d, positions[start as usize..end as usize].to_vec()))
                .collect::<std::collections::HashMap<i32, Vec<i32>>>(),
        );
        per_term_docs.push(docs);
    }

    if !per_term_docs.iter().all(|docs| docs.contains(&doc)) {
        return Ok(Explanation::no_match("no matching terms"));
    }

    let term_positions: Vec<Vec<i32>> = per_term_maps
        .iter()
        .map(|m| {
            m.get(&doc)
                .cloned()
                .expect("doc came from the conjunction of every term's own doc list")
        })
        .collect();
    let term_positions: Vec<&[i32]> = term_positions.iter().map(|v| v.as_slice()).collect();
    // `PhraseScorer.score()`'s frequency: `ExactPhraseMatcher`'s match count
    // for `slop == 0`, `SloppyPhraseMatcher`'s summed `1/(1+matchLength)`
    // otherwise -- see `crate::phrase_freq_sloppy`.
    let phrase_freq = if query.slop == 0 {
        crate::phrase_freq_exact(&term_positions) as f32
    } else {
        crate::phrase_freq_sloppy(&term_positions, query.slop)
    };
    if phrase_freq == 0.0 {
        return Ok(Explanation::no_match("no matching phrase"));
    }

    let (field_length, avg_field_length) = match norms {
        Some(fn_) => (fn_.field_length(doc)?, fn_.avg_field_length),
        None => (
            similarity::UNNORMED_FIELD_LENGTH,
            similarity::UNNORMED_FIELD_LENGTH,
        ),
    };
    // Same `doScore`/`explainTF` split as `explain_term` above.
    let norm_inverse = similarity::norm_inverse(
        field_length,
        avg_field_length,
        similarity::DEFAULT_K1,
        similarity::DEFAULT_B,
    );
    let tf_norm = 1.0 - 1.0 / (1.0 + phrase_freq * norm_inverse);
    let value = similarity::do_score(idf_sum, phrase_freq, norm_inverse);

    // `BM25Similarity.idfExplain(FieldStats, TermStats[])` -- the phrase's idf
    // is the sum of every constituent term's own idf, reported under Java's
    // exact `"idf, sum of:"` description with one child per term.
    let idf_node = Explanation::match_(idf_sum, "idf, sum of:").with_details(idf_details);

    // `PhraseWeight.explain` builds `Explanation.match(freq, "phraseFreq=" +
    // freq)` and hands it to the very same `BM25Scorer.explain`/`explainTF`
    // pair `explain_term` uses, so the `tf`/`score` nodes below are the
    // term case's strings verbatim, only with the phrase freq inside.
    let tf_explanation = Explanation::match_(
        tf_norm,
        "tf, computed as freq / (freq + k1 * (1 - b + b * dl / avgdl)) from:",
    )
    .with_details(vec![
        Explanation::match_(
            phrase_freq,
            format!("phraseFreq={}", java_float(phrase_freq)),
        ),
        Explanation::match_(similarity::DEFAULT_K1, "k1, term saturation parameter"),
        Explanation::match_(similarity::DEFAULT_B, "b, length normalization parameter"),
        Explanation::match_(field_length, dl_description(field_length)),
        Explanation::match_(avg_field_length, "avgdl, average length of field"),
    ]);

    let score_explanation = Explanation::match_(
        value,
        format!(
            "score(freq={}), computed as boost * idf * tf from:",
            java_float(phrase_freq)
        ),
    )
    .with_details(vec![idf_node, tf_explanation]);

    Ok(Explanation::match_(
        value,
        format!(
            "weight({} in {doc}) [BM25Similarity], result of:",
            describe_clause(&Clause::Phrase(query.clone()))
        ),
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
    // `BooleanWeight.explain`'s exact control flow: build every clause's own
    // explanation first, tracking `fail` (a required clause that didn't match,
    // or a prohibited clause that did), `matchCount` and `shouldMatchCount`,
    // then pick one of Java's four outcomes from those counters.
    let mut details: Vec<Explanation> = Vec::new();
    let mut failing_optionals: Vec<Explanation> = Vec::new();
    let mut fail = false;
    let mut match_count = 0usize;
    let mut should_match_count = 0usize;
    let mut total = 0.0f32;

    for clause in &query.must {
        let e = explain_clause(
            fields, doc_in, pos_in, pay_in, live_docs, clause, doc, norms,
        )?;
        if e.matched {
            match_count += 1;
            total += e.value;
            details.push(e);
        } else {
            fail = true;
            details.push(
                Explanation::no_match(format!(
                    "no match on required clause ({})",
                    describe_clause(clause)
                ))
                .with_details(vec![e]),
            );
        }
    }
    // `BooleanWeight.explain`'s `FILTER` arm, verbatim:
    //
    //     subs.add(Explanation.match(0f, "match on required clause, product of:",
    //         Explanation.match(0f, Occur.FILTER + " clause"), e));
    //
    // `Occur.FILTER + " clause"` is `"# clause"`. The wrapper's value is `0f`,
    // which is what keeps a filter clause out of the parent's sum -- the
    // clause's own explanation `e` survives as a child for diagnosis, but its
    // value is not what the parent adds up. `matchCount` counts it (Java
    // increments for every non-prohibited match); `shouldMatchCount` does not.
    for clause in &query.filter {
        let e = explain_clause(
            fields, doc_in, pos_in, pay_in, live_docs, clause, doc, norms,
        )?;
        if e.matched {
            match_count += 1;
            details.push(
                Explanation::match_(0.0, "match on required clause, product of:")
                    .with_details(vec![Explanation::match_(0.0, "# clause"), e]),
            );
        } else {
            fail = true;
            details.push(
                Explanation::no_match(format!(
                    "no match on required clause ({})",
                    describe_clause(clause)
                ))
                .with_details(vec![e]),
            );
        }
    }
    for clause in &query.should {
        let e = explain_clause(
            fields, doc_in, pos_in, pay_in, live_docs, clause, doc, norms,
        )?;
        if e.matched {
            match_count += 1;
            should_match_count += 1;
            total += e.value;
            details.push(e);
        } else {
            failing_optionals.push(
                Explanation::no_match(format!(
                    "no match on optional clause ({})",
                    describe_clause(clause)
                ))
                .with_details(vec![e]),
            );
        }
    }
    for clause in &query.must_not {
        let e = explain_clause(
            fields, doc_in, pos_in, pay_in, live_docs, clause, doc, norms,
        )?;
        if e.matched {
            fail = true;
            details.push(
                Explanation::no_match(format!(
                    "match on prohibited clause ({})",
                    describe_clause(clause)
                ))
                .with_details(vec![e]),
            );
        }
    }

    if fail {
        return Ok(Explanation::no_match(
            "Failure to meet condition(s) of required/prohibited clause(s)",
        )
        .with_details(details));
    }
    if match_count == 0 {
        details.extend(failing_optionals);
        return Ok(Explanation::no_match("No matching clauses").with_details(details));
    }
    if should_match_count < query.minimum_should_match {
        details.extend(failing_optionals);
        return Ok(Explanation::no_match(format!(
            "Failure to match minimum number of optional clauses: {}, matched: {should_match_count}",
            query.minimum_should_match
        ))
        .with_details(details));
    }
    Ok(Explanation::match_(total, "sum of:").with_details(details))
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
    // `DisjunctionMaxWeight.explain`: sub-explanations of every *matching*
    // disjunct on a match, of every non-matching one on a no-match.
    let mut subs_on_match = Vec::new();
    let mut subs_on_no_match = Vec::new();
    for clause in &query.disjuncts {
        let e = explain_clause(
            fields, doc_in, pos_in, pay_in, live_docs, clause, doc, norms,
        )?;
        if e.matched {
            subs_on_match.push(e);
        } else if subs_on_match.is_empty() {
            subs_on_no_match.push(e);
        }
    }
    if subs_on_match.is_empty() {
        return Ok(Explanation::no_match("No matching clause").with_details(subs_on_no_match));
    }

    let mut max_score = f32::NEG_INFINITY;
    let mut sum_score = 0.0f32;
    for e in &subs_on_match {
        sum_score += e.value;
        if e.value > max_score {
            max_score = e.value;
        }
    }
    let other_sum = sum_score - max_score;
    let value = max_score + query.tie_breaker * other_sum;

    // `DisjunctionMaxWeight.explain`'s exact description switch.
    let desc = if query.tie_breaker == 0.0 {
        "max of:".to_string()
    } else {
        format!(
            "max plus {} times others of:",
            java_float(query.tie_breaker)
        )
    };
    Ok(Explanation::match_(value, desc).with_details(subs_on_match))
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
        return Ok(Explanation::no_match(format!(
            "ConstantScore({}) doesn't match id {doc}",
            describe_clause(&nested.inner)
        )));
    }
    // `ConstantScoreWeight.explain`: `getQuery().toString() + (score == 1f ?
    // "" : "^" + score)`.
    let described = format!("ConstantScore({})", describe_clause(&nested.inner));
    let description = if nested.score == 1.0 {
        described
    } else {
        format!("{described}^{}", java_float(nested.score))
    };
    Ok(Explanation::match_(nested.score, description))
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
        return Ok(Explanation::no_match(format!(
            "{} doesn't match id {doc}",
            describe_clause(&Clause::Boost(Box::new(nested.clone())))
        )));
    }
    let value = inner.value * nested.boost;
    Ok(Explanation::match_(value, "product of:")
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

    // --- Description-string fidelity ---
    //
    // Real Lucene's explanation *strings* are a public contract: downstream
    // tooling (OpenSearch's `_explain` API, relevance-debugging UIs) parses
    // them. These tests pin every description this module emits to the exact
    // literal real Lucene 10.5.0 produces, quoted from
    // `BM25Similarity.idfExplain`/`BM25Scorer.explain`/`explainTF`,
    // `TermQuery.TermWeight.explain`, `PhraseWeight.explain`,
    // `BooleanWeight.explain`, `DisjunctionMaxQuery.DisjunctionMaxWeight.
    // explain` and `ConstantScoreWeight.explain`.

    #[test]
    fn term_explanation_descriptions_are_real_lucenes_verbatim() {
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let clause = Clause::Term(TermQuery::new("body", "cat"));
        let e =
            explain_clause(&fields, doc_in.as_ref(), None, None, None, &clause, 0, None).unwrap();

        assert_eq!(
            e.description,
            "weight(body:cat in 0) [BM25Similarity], result of:"
        );
        let score = &e.details[0];
        assert!(
            score
                .description
                .ends_with("), computed as boost * idf * tf from:"),
            "{}",
            score.description
        );
        assert!(
            score.description.starts_with("score(freq="),
            "{}",
            score.description
        );
        // Java renders the freq as a `Float`, i.e. always with a decimal point.
        assert!(
            score.description.contains(".0)"),
            "freq must render Java-style (\"2.0\", not \"2\"): {}",
            score.description
        );

        let idf = &score.details[0];
        assert_eq!(
            idf.description,
            "idf, computed as log(1 + (N - n + 0.5) / (n + 0.5)) from:"
        );
        assert_eq!(
            idf.details[0].description,
            "n, number of documents containing term"
        );
        assert_eq!(
            idf.details[1].description,
            "N, total number of documents with field"
        );

        let tf = &score.details[1];
        assert_eq!(
            tf.description,
            "tf, computed as freq / (freq + k1 * (1 - b + b * dl / avgdl)) from:"
        );
        let descs: Vec<&str> = tf.details.iter().map(|d| d.description.as_str()).collect();
        assert_eq!(
            descs,
            vec![
                "freq, occurrences of term within document",
                "k1, term saturation parameter",
                "b, length normalization parameter",
                "dl, length of field",
                "avgdl, average length of field",
            ]
        );
    }

    #[test]
    fn phrase_explanation_descriptions_are_real_lucenes_verbatim() {
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let owned = doc.as_ref().unwrap();
        let pos_in = owned.open_pos();
        let pay_in = owned.open_pay();
        // "alpha beta" matches real doc 8555 in the "pos" field at slop 0 --
        // the same known-good phrase this module's other phrase tests use.
        let query = crate::query::PhraseQuery::new("pos", ["alpha", "beta"]);
        let clause = Clause::Phrase(query);

        // Whichever doc actually contains the phrase; the no-match strings on
        // the way there are just as much a part of the contract.
        let mut checked = false;
        for doc_id in [0, 8555] {
            let e = explain_clause(
                &fields,
                doc_in.as_ref(),
                Some(&pos_in),
                Some(&pay_in),
                None,
                &clause,
                doc_id,
                None,
            )
            .unwrap();
            if !e.matched {
                assert!(
                    e.description == "no matching terms" || e.description == "no matching phrase",
                    "{}",
                    e.description
                );
                continue;
            }
            checked = true;
            assert_eq!(
                e.description,
                format!("weight(pos:\"alpha beta\" in {doc_id}) [BM25Similarity], result of:")
            );
            let score = &e.details[0];
            assert!(
                score.description.starts_with("score(freq=")
                    && score
                        .description
                        .ends_with("), computed as boost * idf * tf from:"),
                "{}",
                score.description
            );
            assert_eq!(score.details[0].description, "idf, sum of:");
            for per_term in &score.details[0].details {
                assert_eq!(
                    per_term.description,
                    "idf, computed as log(1 + (N - n + 0.5) / (n + 0.5)) from:"
                );
            }
            let tf = &score.details[1];
            assert_eq!(
                tf.description,
                "tf, computed as freq / (freq + k1 * (1 - b + b * dl / avgdl)) from:"
            );
            assert!(
                tf.details[0].description.starts_with("phraseFreq="),
                "{}",
                tf.details[0].description
            );
        }
        assert!(checked, "fixture must contain the phrase \"alpha beta\"");
    }

    #[test]
    fn filter_clause_explanation_is_real_lucenes_verbatim() {
        // `BooleanWeight.explain`'s FILTER arm:
        //
        //     subs.add(Explanation.match(0f, "match on required clause, product of:",
        //         Explanation.match(0f, Occur.FILTER + " clause"), e));
        //
        // `Occur.FILTER.toString()` is `"#"`, so the inner description is
        // exactly `"# clause"` and both values are `0f`.
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());

        let clause = Clause::Boolean(Box::new(
            BooleanQuery::new()
                .with_must([TermQuery::new("body", "cat")])
                .with_filter([TermQuery::new("body", "dog")]),
        ));
        let e =
            explain_clause(&fields, doc_in.as_ref(), None, None, None, &clause, 0, None).unwrap();
        assert!(e.matched);
        assert_eq!(e.description, "sum of:");
        assert_eq!(e.details.len(), 2, "one MUST sub, one FILTER sub");

        let filter_sub = &e.details[1];
        assert!(filter_sub.matched);
        assert_eq!(filter_sub.value, 0.0);
        assert_eq!(
            filter_sub.description,
            "match on required clause, product of:"
        );
        assert_eq!(filter_sub.details.len(), 2);
        assert_eq!(filter_sub.details[0].description, "# clause");
        assert_eq!(filter_sub.details[0].value, 0.0);
        assert!(
            filter_sub.details[1]
                .description
                .starts_with("weight(body:dog"),
            "the filtered clause's own explanation survives as a child: {}",
            filter_sub.details[1].description
        );

        // ... and the total is the MUST clause's value alone -- bit for bit.
        assert_eq!(e.value.to_bits(), e.details[0].value.to_bits());
    }

    #[test]
    fn a_failing_filter_clause_explains_as_a_failing_required_clause() {
        // Java routes the non-matching case through `c.isRequired()`, which is
        // true for FILTER, so the description is the same
        // `"no match on required clause (...)"` a MUST clause gets.
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let clause = Clause::Boolean(Box::new(
            BooleanQuery::new()
                .with_must([TermQuery::new("body", "cat")])
                .with_filter([TermQuery::new("body", "nonexistentterm")]),
        ));
        let e =
            explain_clause(&fields, doc_in.as_ref(), None, None, None, &clause, 0, None).unwrap();
        assert!(!e.matched);
        assert_eq!(
            e.description,
            "Failure to meet condition(s) of required/prohibited clause(s)"
        );
        assert!(e
            .details
            .iter()
            .any(|d| d.description == "no match on required clause (body:nonexistentterm)"));
    }

    #[test]
    fn a_filter_only_query_explains_as_a_match_of_zero() {
        // `matchCount` counts a matching FILTER clause (Java increments it for
        // every non-prohibited match), so this is a match, and the sum is 0.
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let clause = Clause::Boolean(Box::new(
            BooleanQuery::new().with_filter([TermQuery::new("body", "cat")]),
        ));
        let e =
            explain_clause(&fields, doc_in.as_ref(), None, None, None, &clause, 0, None).unwrap();
        assert!(e.matched);
        assert_eq!(e.description, "sum of:");
        assert_eq!(e.value, 0.0);
    }

    #[test]
    fn a_filter_clause_does_not_count_toward_minimum_should_match_in_explain() {
        // Java increments `shouldMatchCount` only for `Occur.SHOULD`. Doc 2
        // matches the `cat` filter and neither optional clause, so the
        // explanation must be the minimum-not-reached failure with
        // `matched: 0` -- not a match.
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let clause = Clause::Boolean(Box::new(
            BooleanQuery::new()
                .with_filter([TermQuery::new("body", "cat")])
                .with_should([TermQuery::new("body", "dog")])
                .with_minimum_should_match(1),
        ));
        let e =
            explain_clause(&fields, doc_in.as_ref(), None, None, None, &clause, 2, None).unwrap();
        assert!(!e.matched);
        assert_eq!(
            e.description,
            "Failure to match minimum number of optional clauses: 1, matched: 0"
        );
    }

    #[test]
    fn describe_clause_prints_a_filter_clause_with_lucenes_hash_prefix() {
        // `Occur.FILTER.toString()` is `"#"`, and `BooleanQuery.toString`
        // prefixes each clause with its occur.
        assert_eq!(
            describe_clause(&Clause::Boolean(Box::new(
                BooleanQuery::new()
                    .with_must([TermQuery::new("body", "cat")])
                    .with_filter([TermQuery::new("body", "dog")])
                    .with_should([TermQuery::new("body", "bird")])
                    .with_must_not([TermQuery::new("body", "fish")])
            ))),
            "(+body:cat #body:dog body:bird -body:fish)"
        );
    }

    #[test]
    fn boolean_explanation_descriptions_are_real_lucenes_verbatim() {
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());

        // A matching boolean: `BooleanWeight.explain` emits a bare "sum of:",
        // with no `"{value} = "` prefix (the value lives in the node itself,
        // and `Explanation.toString` re-renders it).
        let clause = Clause::Boolean(Box::new(BooleanQuery {
            must: vec![Clause::Term(TermQuery::new("body", "cat"))],
            filter: Vec::new(),
            should: vec![Clause::Term(TermQuery::new("body", "nonexistentterm"))],
            must_not: Vec::new(),
            minimum_should_match: 0,
        }));
        let e =
            explain_clause(&fields, doc_in.as_ref(), None, None, None, &clause, 0, None).unwrap();
        assert!(e.matched);
        assert_eq!(e.description, "sum of:");

        // A required clause that fails.
        let clause = Clause::Boolean(Box::new(BooleanQuery {
            must: vec![Clause::Term(TermQuery::new("body", "nonexistentterm"))],
            filter: Vec::new(),
            should: Vec::new(),
            must_not: Vec::new(),
            minimum_should_match: 0,
        }));
        let e =
            explain_clause(&fields, doc_in.as_ref(), None, None, None, &clause, 0, None).unwrap();
        assert!(!e.matched);
        assert_eq!(
            e.description,
            "Failure to meet condition(s) of required/prohibited clause(s)"
        );
        assert_eq!(
            e.details[0].description,
            "no match on required clause (body:nonexistentterm)"
        );

        // A prohibited clause that matches.
        let clause = Clause::Boolean(Box::new(BooleanQuery {
            must: vec![Clause::Term(TermQuery::new("body", "cat"))],
            filter: Vec::new(),
            should: Vec::new(),
            must_not: vec![Clause::Term(TermQuery::new("body", "cat"))],
            minimum_should_match: 0,
        }));
        let e =
            explain_clause(&fields, doc_in.as_ref(), None, None, None, &clause, 0, None).unwrap();
        assert!(!e.matched);
        assert_eq!(
            e.description,
            "Failure to meet condition(s) of required/prohibited clause(s)"
        );
        assert!(e
            .details
            .iter()
            .any(|d| d.description == "match on prohibited clause (body:cat)"));

        // Nothing matched at all.
        let clause = Clause::Boolean(Box::new(BooleanQuery {
            must: Vec::new(),
            filter: Vec::new(),
            should: vec![Clause::Term(TermQuery::new("body", "nonexistentterm"))],
            must_not: Vec::new(),
            minimum_should_match: 0,
        }));
        let e =
            explain_clause(&fields, doc_in.as_ref(), None, None, None, &clause, 0, None).unwrap();
        assert_eq!(e.description, "No matching clauses");
        assert_eq!(
            e.details[0].description,
            "no match on optional clause (body:nonexistentterm)"
        );

        // minimumShouldMatch not reached, but at least one clause matched.
        let clause = Clause::Boolean(Box::new(BooleanQuery {
            must: Vec::new(),
            filter: Vec::new(),
            should: vec![
                Clause::Term(TermQuery::new("body", "cat")),
                Clause::Term(TermQuery::new("body", "nonexistentterm")),
            ],
            must_not: Vec::new(),
            minimum_should_match: 2,
        }));
        let e =
            explain_clause(&fields, doc_in.as_ref(), None, None, None, &clause, 0, None).unwrap();
        assert_eq!(
            e.description,
            "Failure to match minimum number of optional clauses: 2, matched: 1"
        );
    }

    #[test]
    fn dismax_explanation_descriptions_are_real_lucenes_verbatim() {
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());

        // `tieBreakerMultiplier == 0` -> "max of:"; otherwise
        // "max plus {tie} times others of:".
        let zero_tie = Clause::DisjunctionMax(Box::new(DisjunctionMaxQuery::new(
            [Clause::Term(TermQuery::new("body", "cat"))],
            0.0,
        )));
        let e = explain_clause(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            &zero_tie,
            0,
            None,
        )
        .unwrap();
        assert_eq!(e.description, "max of:");

        let tie = Clause::DisjunctionMax(Box::new(DisjunctionMaxQuery::new(
            [Clause::Term(TermQuery::new("body", "cat"))],
            0.5,
        )));
        let e = explain_clause(&fields, doc_in.as_ref(), None, None, None, &tie, 0, None).unwrap();
        assert_eq!(e.description, "max plus 0.5 times others of:");

        let none = Clause::DisjunctionMax(Box::new(DisjunctionMaxQuery::new(
            [Clause::Term(TermQuery::new("body", "nonexistentterm"))],
            0.0,
        )));
        let e = explain_clause(&fields, doc_in.as_ref(), None, None, None, &none, 0, None).unwrap();
        assert_eq!(e.description, "No matching clause");
    }

    #[test]
    fn constant_score_and_flat_clause_descriptions_are_real_lucenes_verbatim() {
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());

        // Score exactly 1.0 -> no "^score" suffix (ConstantScoreWeight.explain).
        let one = Clause::ConstantScore(Box::new(ConstantScoreQuery::new(
            Clause::Term(TermQuery::new("body", "cat")),
            1.0,
        )));
        let e = explain_clause(&fields, doc_in.as_ref(), None, None, None, &one, 0, None).unwrap();
        assert_eq!(e.description, "ConstantScore(body:cat)");

        let boosted = Clause::ConstantScore(Box::new(ConstantScoreQuery::new(
            Clause::Term(TermQuery::new("body", "cat")),
            2.5,
        )));
        let e = explain_clause(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            &boosted,
            0,
            None,
        )
        .unwrap();
        assert_eq!(e.description, "ConstantScore(body:cat)^2.5");

        let miss = Clause::ConstantScore(Box::new(ConstantScoreQuery::new(
            Clause::Term(TermQuery::new("body", "nonexistentterm")),
            1.0,
        )));
        let e = explain_clause(&fields, doc_in.as_ref(), None, None, None, &miss, 3, None).unwrap();
        assert_eq!(
            e.description,
            "ConstantScore(body:nonexistentterm) doesn't match id 3"
        );

        // A flat (constant-scoring) multi-term clause renders as the query
        // itself on a match and `"<query> doesn't match id <doc>"` otherwise.
        let prefix = Clause::Prefix(PrefixQuery::new("body", "ca"));
        let e =
            explain_clause(&fields, doc_in.as_ref(), None, None, None, &prefix, 0, None).unwrap();
        assert_eq!(e.description, "body:ca*");
        let missing = Clause::Prefix(PrefixQuery::new("body", "zzzz"));
        let e = explain_clause(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            &missing,
            0,
            None,
        )
        .unwrap();
        assert_eq!(e.description, "body:zzzz* doesn't match id 0");
    }

    #[test]
    fn describe_clause_matches_lucene_query_to_string() {
        assert_eq!(
            describe_clause(&Clause::Term(TermQuery::new("f", "t"))),
            "f:t"
        );
        assert_eq!(
            describe_clause(&Clause::Phrase(crate::query::PhraseQuery::new(
                "f",
                ["a", "b"]
            ))),
            "f:\"a b\""
        );
        assert_eq!(
            describe_clause(&Clause::Phrase(
                crate::query::PhraseQuery::new("f", ["a", "b"]).with_slop(3)
            )),
            "f:\"a b\"~3"
        );
        assert_eq!(
            describe_clause(&Clause::Boolean(Box::new(BooleanQuery {
                must: vec![Clause::Term(TermQuery::new("f", "a"))],
                filter: Vec::new(),
                should: vec![Clause::Term(TermQuery::new("f", "b"))],
                must_not: vec![Clause::Term(TermQuery::new("f", "c"))],
                minimum_should_match: 0,
            }))),
            "(+f:a f:b -f:c)"
        );
        assert_eq!(
            describe_clause(&Clause::DisjunctionMax(Box::new(DisjunctionMaxQuery::new(
                [
                    Clause::Term(TermQuery::new("f", "a")),
                    Clause::Term(TermQuery::new("f", "b")),
                ],
                0.3,
            )))),
            "(f:a | f:b)~0.3"
        );
        assert_eq!(
            describe_clause(&Clause::Boost(Box::new(BoostQuery::new(
                Clause::Term(TermQuery::new("f", "a")),
                2.0,
            )))),
            "(f:a)^2.0"
        );
        assert_eq!(
            describe_clause(&Clause::Wildcard(WildcardQuery::new("f", "a*b"))),
            "f:a*b"
        );
        assert_eq!(
            describe_clause(&Clause::Fuzzy(crate::query::FuzzyQuery::new("f", "a"))),
            "f:a~2"
        );
        assert_eq!(
            describe_clause(&Clause::Regexp(crate::query::RegexpQuery::new("f", "a.*"))),
            "f:/a.*/"
        );
        assert_eq!(
            describe_clause(&Clause::MatchAllDocs(crate::query::MatchAllDocsQuery::new(
                7
            ))),
            "*:*"
        );
        assert_eq!(
            describe_clause(&Clause::Span(crate::query::SpanQuery::span_near(
                [
                    crate::query::SpanQuery::span_term("f", "a"),
                    crate::query::SpanQuery::span_term("f", "b"),
                ],
                2,
                true,
            ))),
            "spanNear([spanTerm(f:a), spanTerm(f:b)], 2, true)"
        );
    }

    #[test]
    fn display_renders_java_explanation_to_string_layout() {
        let e = Explanation::match_(2.0, "sum of:").with_details(vec![
            Explanation::match_(1.5, "weight(f:a in 0) [BM25Similarity], result of:")
                .with_details(vec![Explanation::match_(1.5, "score(freq=1.0), ...")]),
            Explanation::match_(0.5, "boost"),
        ]);
        assert_eq!(
            e.to_string(),
            "2.0 = sum of:\n  \
             1.5 = weight(f:a in 0) [BM25Similarity], result of:\n    \
             1.5 = score(freq=1.0), ...\n  \
             0.5 = boost\n"
        );
    }

    #[test]
    fn java_float_always_keeps_a_decimal_point() {
        // Rust's own `Display` for f32 prints "2"; Java's `Float.toString`
        // prints "2.0", and every explanation string in this module has to
        // agree with Java's.
        assert_eq!(java_float(2.0), "2.0");
        assert_eq!(java_float(0.5), "0.5");
        assert_eq!(java_float(1.2), "1.2");
    }

    #[test]
    fn dl_description_flips_exactly_where_the_encoded_norm_byte_does() {
        // `BM25Scorer.explainTF` picks "(approximate)" when the encoded norm
        // byte is > 39. `dl_description` uses the *decoded* length instead;
        // that is only equivalent because `LENGTH_TABLE[i] == i` for every
        // `i <= 39` and the table is monotonically non-decreasing. Assert both
        // properties against this port's own `decode_norm`, so a change to the
        // table would fail here rather than silently drift from Java.
        let mut previous = f32::NEG_INFINITY;
        for i in 0..256u32 {
            let decoded = similarity::decode_norm(i as i64);
            assert!(decoded >= previous, "LENGTH_TABLE must not decrease at {i}");
            previous = decoded;
            if i <= 39 {
                assert_eq!(decoded, i as f32, "LENGTH_TABLE[{i}] must be exactly {i}");
            }
            let java_says_approximate = i > 39;
            assert_eq!(
                dl_description(decoded) == "dl, length of field (approximate)",
                java_says_approximate,
                "encoded norm {i} (decoded {decoded})"
            );
        }
    }

    #[test]
    fn points_range_clause_is_not_yet_explainable() {
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let clause = Clause::PointsRange(crate::query::PointsRangeQuery::new("body", 0, 100));
        let err = explain_clause(&fields, doc_in.as_ref(), None, None, None, &clause, 0, None)
            .unwrap_err();
        assert!(matches!(err, crate::Error::MissingPointsInput(field) if field == "body"));
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

        // The nested idf/tfNorm sub-values must multiply back to the top-level
        // value -- but only to within rounding, not exactly. Real Lucene says
        // so itself in `BM25Scorer.explain`: "not using 'product of' since the
        // rewrite that we do in score() introduces a small rounding error that
        // CheckHits complains about". The score is
        // `weight - weight / (1 + freq * normInverse)` and the `tf` node is
        // `1 - 1 / (1 + freq * normInverse)`; those are equal in exact
        // arithmetic and a few ULP apart in `f32`. The tolerance is
        // **relative**: `f32::EPSILON` on its own is the gap between 1.0 and
        // the next float, which is smaller than one ULP for any score >= 2 --
        // the multi-phrase fixture already scores 1.57 -- so an absolute
        // epsilon would make this flake rather than fail honestly.
        let score_node = &explanation.details[0];
        let idf_node = &score_node.details[0];
        let tf_norm_node = &score_node.details[1];
        assert!(
            (idf_node.value * tf_norm_node.value - expected_score).abs()
                <= 4.0 * f32::EPSILON * expected_score.abs().max(1.0),
            "idf {} * tfNorm {} = {}, score {expected_score}",
            idf_node.value,
            tf_norm_node.value,
            idf_node.value * tf_norm_node.value
        );
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
            // `0xFF` is Java's `denseRankPower == -1`, "no rank table". `0` is
            // not in `IndexedDISI`'s legal set (`-1`, or `7..=15`), so
            // `dense_rank_bytes(0)` rejects it and any DENSE block reached with
            // it fails to decode. Unreachable here -- this entry is dense, so
            // nothing looks at the field -- but it described metadata no writer
            // can produce, which is the kind of almost-right test input that
            // hides a real decode bug. Same fix `field_norms.rs`'s `NO_RANK`
            // constant records. (`field_norms.rs`'s own literal `0` is
            // deliberate and stays: it is the input to
            // `an_illegal_dense_rank_power_is_rejected_rather_than_guessed`,
            // which asserts the decode *refuses* it. Named rather than cited by
            // line number, which drifts.)
            dense_rank_power: 0xFF,
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

        // Multiply back to the top-level value to within rounding -- see
        // `term_explain_matching_doc_equals_scored_search_output_exactly` for
        // why real Lucene's own `explain` refuses to call this a "product of".
        let score_node = &explanation.details[0];
        let idf_node = &score_node.details[0];
        let tf_norm_node = &score_node.details[1];
        assert!(
            (idf_node.value * tf_norm_node.value - expected_score).abs()
                <= 4.0 * f32::EPSILON * expected_score.abs().max(1.0),
            "idf {} * tfNorm {} vs score {expected_score}",
            idf_node.value,
            tf_norm_node.value
        );
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
