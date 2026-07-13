#![forbid(unsafe_code)]
//! lucene-search: see /PLAN.md for scope.
//!
//! First slice: single-segment `TermQuery` execution — find every live doc
//! ID matching one exact `(field, term)` pair, against a segment already
//! opened the way `crates/lucene-codecs/tests/blocktree_fixtures.rs` opens
//! one today (a `blocktree::BlockTreeFields` plus, when the term's `docFreq
//! > 1`, an opened `.doc` file).
//!
//! ## Scope of this slice (see PLAN.md's Phase 3 section for the full plan)
//!
//! **In scope:**
//! - [`query::TermQuery`]: field + exact term, no scoring metadata.
//! - [`search_term_query`]: `seekExact` the term via
//!   `blocktree::FieldTerms::postings`, then feed every `(docID, freq)` pair
//!   through a [`collector::Collector`], filtered by `live_docs` (deleted
//!   docs excluded — matches `IndexSearcher`'s `Bits liveDocs` handling,
//!   `null` meaning "no deletions, every doc is live").
//! - [`collector::CountCollector`]/[`collector::VecCollector`]: the two
//!   simplest observationally-useful collectors (`TotalHitCountCollector`
//!   and "give me the doc IDs", respectively).
//!
//! **Deliberately out of scope, left for later PLAN.md Phase 3 slices:**
//! - **Relevance scoring.** Real `TermQuery`'s `Weight`/`Scorer` pair always
//!   computes a BM25 (or configured `Similarity`) score per doc, using norms
//!   and collection statistics (`docFreq`/`sumTotalTermFreq`) this port can
//!   already read (`blocktree::TermStats`) but has no `Similarity` module
//!   for yet. A "does this doc match" query is honestly a different, simpler
//!   problem than "how well does this doc match", and PLAN.md's own phase
//!   plan lists Similarity/BM25 as a separate line item — this slice proves
//!   the matching/collection plumbing first, without inventing scoring math
//!   ahead of schedule.
//! - **Multi-segment search / `IndexSearcher`/`IndexReader` federation.**
//!   This module runs against one already-opened segment's term dictionary
//!   and postings file — there is no `SegmentReader`/`DirectoryReader`/
//!   `IndexReader` abstraction in this port yet (the write side only
//!   produces fully-stored-only segments so far — see
//!   `crates/lucene-index/examples/write_segment_infos_fixture.rs`'s module
//!   doc — and no unified "open every file this segment's `.si` names"
//!   reader exists on the read side either). Building that abstraction is
//!   its own task; this slice takes already-opened
//!   `blocktree::BlockTreeFields`/`postings::DocInput` values as parameters,
//!   the same shape the differential tests in
//!   `crates/lucene-codecs/tests/blocktree_fixtures.rs` already use.
//! - **Dynamic pruning (WAND/MAXSCORE), skip-ahead-driven early
//!   termination, `TopScoreDocCollector`.** All meaningless without scoring;
//!   deferred alongside it. This slice also doesn't use
//!   `postings::LazyDocsCursor`'s decode-on-demand `advance()` — a full
//!   `seekExact` + eager `postings()` materializes every matching doc up
//!   front (same tradeoff `blocktree.rs` itself already made for term
//!   lookup — see that module's doc comment — "correctness first, profile
//!   before optimizing" per the `rust-performance` skill). A future slice
//!   that adds a real multi-term/skip-driven query shape is the right place
//!   to switch to the lazy cursor for genuine sub-linear skipping.
//!
//! **Design note — why a plain function, not a `Weight`/`Scorer` trait
//! hierarchy:** real Lucene's `Query -> Weight -> Scorer/BulkScorer` chain
//! exists to support many query types composing arbitrarily (a
//! `BooleanQuery` wraps `Weight`s recursively) and per-segment reuse of
//! collection statistics across a multi-segment `IndexSearcher`. With
//! exactly one query type and exactly one segment, none of that
//! polymorphism has a second caller yet — introducing the trait hierarchy
//! now would be speculative generality with a single implementation, the
//! opposite of what `rust-performance` asks for. When `BooleanQuery` and
//! multi-segment search land, revisit whether an enum-based `Scorer`
//! (`rust-performance`'s "enums where the closed set allows" guidance)
//! earns its keep.
//!
//! One concrete piece of rework this design note still defers, named explicitly
//! so the next contributor isn't surprised by its size: **[`collector::Collector`]
//! will need a breaking signature change for relevance scoring** --
//! `collect(&mut self, doc_id: i32)` has no way to receive a score the way
//! real Lucene's `LeafCollector` does via `setScorer`/`Scorer.score()`; this
//! isn't a small addition, every existing `Collector` impl's signature
//! changes.
//!
//! ## `BooleanQuery` (this slice's addition)
//!
//! [`query::BooleanQuery`]/[`search_boolean_query`] add `MUST`/`SHOULD`/`MUST_NOT`
//! conjunction, disjunction, and exclusion over `TermQuery` clauses, built on the new
//! [`docid_set`] module's [`docid_set::Conjunction`]/[`docid_set::Disjunction`]/
//! [`docid_set::Excluding`] merge combinators (see that module's doc comment for why
//! they're plain `Iterator<Item = i32>` adapters rather than a bespoke
//! `next_doc`/`advance` trait). `search_term_query` itself is refactored to share the
//! same `term_doc_ids` helper `search_boolean_query`'s per-clause lookups use, rather
//! than duplicating the field-lookup/`postings`/`live_docs`-filter sequence — a clean
//! simplification since both now want exactly "one clause's ascending, live-filtered
//! doc-ID sequence", with no behavior change to `search_term_query`'s own contract.
//!
//! Matching semantics follow real `BooleanQuery.rewrite()`/`BooleanWeight`
//! (`org.apache.lucene.search.BooleanQuery`/`BooleanWeight`, verified against that
//! source rather than guessed): a query with **no `must` and no `should` clauses
//! matches nothing**, regardless of `must_not` — real Lucene rewrites both "no
//! clauses at all" (`clauses.isEmpty()`) and "only `MUST_NOT` clauses"
//! (`clauses.size() == clauseSets.get(MUST_NOT).size()`) to a `MatchNoDocsQuery`,
//! i.e. a **pure negative query does not mean "match every doc except the excluded
//! ones"** — it means match nothing.
//!
//! `query.minimum_should_match` (task #24's addition; `query::BooleanQuery`'s doc
//! comment has the full field-level accounting) gates `should` **regardless of
//! whether `must` is also non-empty** — this is the one place it's easy to get
//! backwards, so it's called out explicitly: real `BooleanWeight.scorer`/
//! `bulkScorer`/`explain` all compute `shouldMatchCount` and reject a doc with
//! `shouldMatchCount < minShouldMatch` unconditionally, not just when `must` is
//! empty. Concretely:
//! - `minimum_should_match == 0` (the default): when `must` is non-empty, `should`
//!   clauses do **not** narrow the matched set at all (scoring-only once a
//!   `MUST`/`FILTER` clause exists); the matched set is `must`'s conjunction alone.
//!   When `must` is empty, the matched set is `should`'s disjunction (a doc needs at
//!   least one `should` hit — `minimum_should_match`'s implicit floor of 1 in that
//!   case).
//! - `minimum_should_match > 0`: **this is a real behavior change from the
//!   `must`-present case above** — a doc drawn from `must`'s conjunction (or, when
//!   `must` is empty, from `should`'s disjunction) is only kept if it *also* matches
//!   at least `minimum_should_match` of the `should` clauses. See
//!   [`should_match_counts`] for the per-doc counting mechanism this needs (a plain
//!   `Disjunction` only reports doc-is-in-the-union, not how many clauses agreed).
//! - `minimum_should_match` exceeding `should.len()`: real
//!   `BooleanQuery.rewrite()` turns this into an explicit `MatchNoDocsQuery`
//!   ("SHOULD clause count less than minimumNumberShouldMatch") at query-construction
//!   time. This port doesn't add a separate branch for it — no doc's should-match
//!   count can ever exceed `should.len()`, so the threshold comparison above already
//!   yields the same "matches nothing" outcome for free.
//!
//! Either way, `must_not`'s disjunction is subtracted from whatever the above
//! produces, same as before `minimum_should_match` existed.
//!
//! **Nested `BooleanQuery` clauses** (task #25's addition): a `must`/`should`/
//! `must_not` clause can itself be a [`query::Clause::Boolean`] (a boxed, nested
//! `BooleanQuery`), to arbitrary depth — see [`query::Clause`]'s doc comment for
//! why an enum (not a `Weight`/`Scorer`-style trait object) is the right shape
//! here, and [`resolve_clause_docs`]/[`clause_scores`] for the recursive
//! matching/scoring algorithms. A nested query resolves its own
//! `must`/`should`/`must_not`/`minimum_should_match` completely independently of
//! its parent's before the parent treats the result as one more clause to merge
//! or score.
//!
//! Deferred, tracked in `docs/parity.md`: `PhraseQuery` as a boolean clause (only
//! `TermQuery` and nested `BooleanQuery` are `Clause` variants today — see
//! `query::Clause`'s doc comment), and — same as `search_term_query` — relevance
//! scoring (a separate task, #13, since implemented — see below).
//!
//! ## Relevance scoring (task #13's addition)
//!
//! [`search_term_query_scored`]/[`search_boolean_query_scored`] add BM25 relevance
//! scoring (see [`similarity`] for the formula and [`field_norms`] for how real
//! per-doc/avg-field-length norms are decoded and fed in). Both take an optional
//! opened [`field_norms::FieldNorms`] (single field) / `HashMap<String,
//! FieldNorms>` (boolean, keyed by clause field) for real BM25 length
//! normalization; passing `None` falls back to a documented constant
//! approximation (`similarity::UNNORMED_FIELD_LENGTH`) for a field with no
//! opened norms — see [`similarity`]'s module doc for the honest accounting of
//! when that fallback applies. [`collector::ScoringCollector`] is the scored
//! sibling of [`collector::Collector`] (a new trait, not a breaking change —
//! see `collector.rs`'s module doc for why), and [`collector::TopDocsCollector`]
//! is the `TopScoreDocCollector`-equivalent that consumes it.
//!
//! `search_term_query_scored` mirrors `search_term_query`'s field/term lookup
//! exactly, additionally reading each matched doc's `freq` (already decoded by
//! `blocktree::FieldTerms::postings`, just previously discarded by
//! `term_doc_ids`) and computing `similarity::score(docFreq, docCount, freq,
//! fieldLength, avgFieldLength)` per doc, using real decoded norms when `norms`
//! is `Some`.
//!
//! `search_boolean_query_scored` computes the same matched-doc set as
//! `search_boolean_query` (reusing `term_doc_ids` for the pure set algebra), then
//! sums each matching doc's per-clause BM25 scores across every `must`/`should`
//! clause that doc satisfies — mirroring real Lucene's additive `BooleanScorer`
//! (`must_not` clauses never contribute a score, matching real
//! `Occur.MUST_NOT`'s "filters, never scores" contract).
//!
//! ## Doc-values-driven range query and sort (this slice's addition)
//!
//! [`doc_value_query`] adds a numeric range filter ([`search_numeric_range`]), a
//! single-valued SORTED ordinal range/equality filter ([`search_sorted_ord_range`]),
//! and a "sort an already-matched doc set by a numeric doc value" post-processing
//! helper ([`sort_by_numeric_doc_value`]), all built directly on
//! `lucene_codecs::doc_values`' already-complete read side (`numeric_value`/
//! `sorted_ord`). See that module's doc comment for the full scope accounting —
//! notably, multi-valued SORTED_NUMERIC/SORTED_SET range/sort (needs a selector
//! concept this port doesn't have yet) and skip-index-driven range pruning (this
//! port doesn't parse doc-values skip indexes) are both deliberately deferred.

pub mod collector;
pub mod directory_reader;
pub mod doc_value_query;
pub mod docid_set;
pub mod facets;
pub mod field_norms;
pub mod highlighter;
pub mod multi_segment;
pub mod query;
pub mod query_parser;
pub mod similarity;
pub mod soft_deletes;
pub mod term_vectors_query;

pub use collector::{
    Collector, CountCollector, ScoreDoc, ScoringCollector, TopDocsCollector, VecCollector,
};
pub use doc_value_query::{
    search_numeric_range, search_sorted_ord_range, sort_by_numeric_doc_value, MissingValue,
};
pub use field_norms::FieldNorms;
pub use multi_segment::{
    merge_multi_segment_scored, search_boolean_query_multi_segment,
    search_term_query_multi_segment, OpenSegment,
};
pub use query::{
    BooleanQuery, BoostQuery, Clause, ConstantScoreQuery, DisjunctionMaxQuery, FuzzyQuery,
    PhraseQuery, PrefixQuery, RegexpQuery, SpanQuery, TermQuery, WildcardQuery,
};
pub use term_vectors_query::{matched_term_offsets, term_vector_for_doc};

use std::collections::HashMap;

use docid_set::{BoxDocIter, Conjunction, Disjunction, Excluding};

use lucene_codecs::blocktree::{self, BlockTreeFields};
use lucene_codecs::fuzzy::FuzzyMatch;
use lucene_codecs::postings::{DocInput, PayInput, PosInput};
use lucene_codecs::regexp::RegexpPattern;
use lucene_codecs::wildcard::WildcardPattern;
use lucene_util::fixed_bit_set::FixedBitSet;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    BlockTree(#[from] blocktree::Error),
    /// A multi-term [`PhraseQuery`] needs an opened `.pos` file to check
    /// position alignment -- the single-term degenerate case (see
    /// [`search_phrase_query`]'s doc comment) never reaches this, since it
    /// delegates straight to [`search_term_query`] without touching
    /// positions at all.
    #[error("phrase query needs an opened .pos file for a multi-term phrase")]
    MissingPosInput,
    /// Surfaced by [`doc_value_query::search_numeric_range`]/
    /// [`doc_value_query::search_sorted_ord_range`]/
    /// [`doc_value_query::sort_by_numeric_doc_value`] when the underlying
    /// `.dvd`/`.dvm` decode fails (e.g. a doc ID out of range for the entry,
    /// or a truncated/corrupt values region).
    #[error(transparent)]
    DocValues(#[from] lucene_codecs::doc_values::Error),
    /// Surfaced by [`field_norms::FieldNorms`] / [`term_doc_scores`] when
    /// decoding a norm byte for a scored query's field fails (a doc ID out of
    /// range for the norms entry, or a truncated/corrupt `.nvd` region).
    #[error(transparent)]
    Norms(#[from] lucene_codecs::norms::Error),
    /// Surfaced by [`term_vectors_query::term_vector_for_doc`] when the
    /// underlying `.tvd`/`.tvx` decode fails (e.g. a doc ID out of range, or
    /// a truncated/corrupt term-vectors region).
    #[error(transparent)]
    TermVectors(#[from] lucene_codecs::term_vectors::Error),
    /// Surfaced by [`regexp_doc_ids`] (task #43's `Clause::Regexp`) when
    /// [`RegexpQuery::pattern`] uses syntax
    /// [`lucene_codecs::regexp::RegexpPattern::new`] doesn't support (see
    /// that module's doc comment for exactly which subset is supported) --
    /// unlike a missing field/term (an empty, non-error match result every
    /// other clause returns), a malformed pattern is a caller mistake
    /// worth surfacing distinctly, the same way a truncated `.tim`/`.tip`
    /// decode is an [`Error::BlockTree`] rather than an empty result.
    #[error(transparent)]
    Regexp(#[from] lucene_codecs::regexp::RegexpError),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Executes `query` against one already-opened segment's term dictionary
/// (and, when needed, its `.doc` postings file), feeding every matching
/// **live** doc ID to `collector` in ascending order.
///
/// - `fields`: the segment's decoded term dictionaries
///   (`blocktree::open(...)`'s result).
/// - `doc_in`: the segment's opened `.doc` file, or `None` if the segment
///   has none opened. Only actually needed when the matched term's `docFreq
///   > 1` (see [`blocktree::FieldTerms::postings`]); a `None` here is fine
///   for a field where every term is a `docFreq == 1` singleton (pulsed
///   entirely into the term dictionary, e.g. this port's `id` fixture
///   field) — passing `None` for a term that turns out to need it surfaces
///   as an [`Error`].
/// - `live_docs`: the segment's `.liv` bitset (set bit == live), or `None`
///   for "no deletions in this segment" (mirrors `IndexSearcher`'s `Bits
///   liveDocs == null` convention) — every matched doc is then reported as
///   live.
///
/// Returns `Ok(())` with no doc reported to `collector` when the query's
/// field doesn't exist in this segment or the term isn't found in that
/// field's dictionary (mirrors `TermQuery.createWeight`'s `null`-`Scorer`
/// "no matches" outcome — not an error).
pub fn search_term_query<C: Collector>(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &TermQuery,
    collector: &mut C,
) -> Result<()> {
    for doc_id in term_doc_ids(fields, doc_in, live_docs, query)? {
        collector.collect(doc_id);
    }
    Ok(())
}

/// Shared per-clause lookup: `seekExact`s `query`'s term via
/// `blocktree::FieldTerms::postings`, then returns every matching doc ID (ascending,
/// per `Postings`' own contract), filtered by `live_docs` the same way
/// `search_term_query` always has. Returns an empty `Vec` — not an error — when the
/// query's field doesn't exist in this segment or the term isn't in that field's
/// dictionary, matching `TermQuery.createWeight`'s `null`-`Scorer` "no matches"
/// outcome. Used by both `search_term_query` and `search_boolean_query` so the
/// field-lookup/`postings`/`live_docs`-filter sequence lives in exactly one place.
fn term_doc_ids(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &TermQuery,
) -> Result<Vec<i32>> {
    let Some(field_terms) = fields.field(&query.field) else {
        return Ok(Vec::new());
    };
    let Some(postings) = field_terms.postings(&query.term, doc_in)? else {
        return Ok(Vec::new());
    };
    Ok(postings
        .docs
        .iter()
        .copied()
        .filter(|&doc_id| live_docs.is_none_or(|bits| bits.get(doc_id as usize)))
        .collect())
}

/// [`Clause::Prefix`]'s matched doc-ID list: same union-across-matching-terms
/// mechanism as [`wildcard_doc_ids`], built on
/// [`lucene_codecs::wildcard::WildcardPattern::prefix`] (a literal-bytes-only
/// pattern -- see [`PrefixQuery`]'s doc comment for why this avoids
/// `WildcardPattern::new`'s glob-escaping entirely) instead of a general glob
/// pattern. Returns an empty `Vec` -- not an error -- when `query.field`
/// doesn't exist in this segment, same "missing field means no matches"
/// convention every other clause follows.
fn prefix_doc_ids(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &PrefixQuery,
) -> Result<Vec<i32>> {
    let Some(field_terms) = fields.field(&query.field) else {
        return Ok(Vec::new());
    };
    let pattern = WildcardPattern::prefix(&query.prefix);
    let matching_terms: Vec<Vec<u8>> = field_terms
        .intersect(&pattern)
        .map(|(term, _stats)| term.to_vec())
        .collect();
    let mut doc_ids: Vec<i32> = Vec::new();
    for term in &matching_terms {
        let Some(postings) = field_terms.postings(term, doc_in)? else {
            continue;
        };
        doc_ids.extend(
            postings
                .docs
                .iter()
                .copied()
                .filter(|&doc_id| live_docs.is_none_or(|bits| bits.get(doc_id as usize))),
        );
    }
    doc_ids.sort_unstable();
    doc_ids.dedup();
    Ok(doc_ids)
}

/// [`Clause::Wildcard`]'s matched doc-ID list: every term
/// [`lucene_codecs::blocktree::FieldTerms::intersect`] finds matching `query`'s
/// compiled pattern (for `query.field`) contributes its own postings' doc IDs,
/// **union**ed across every matching term (real `WildcardQuery`'s
/// `MultiTermQuery` matching contract -- a doc matches if *any* accepted term
/// occurs in it) and deduplicated (a doc can hold more than one term the
/// pattern accepts in a multi-valued field), then filtered by `live_docs` same
/// as [`term_doc_ids`]. Returns an empty `Vec` -- not an error -- when
/// `query.field` doesn't exist in this segment, matching every other clause's
/// "missing field means no matches" convention.
fn wildcard_doc_ids(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &WildcardQuery,
) -> Result<Vec<i32>> {
    let Some(field_terms) = fields.field(&query.field) else {
        return Ok(Vec::new());
    };
    let pattern = WildcardPattern::new(&query.pattern);
    let matching_terms: Vec<Vec<u8>> = field_terms
        .intersect(&pattern)
        .map(|(term, _stats)| term.to_vec())
        .collect();
    let mut doc_ids: Vec<i32> = Vec::new();
    for term in &matching_terms {
        let Some(postings) = field_terms.postings(term, doc_in)? else {
            continue;
        };
        doc_ids.extend(
            postings
                .docs
                .iter()
                .copied()
                .filter(|&doc_id| live_docs.is_none_or(|bits| bits.get(doc_id as usize))),
        );
    }
    doc_ids.sort_unstable();
    doc_ids.dedup();
    Ok(doc_ids)
}

/// [`Clause::Fuzzy`]'s matched doc-ID list (task #42): same
/// union-across-matching-terms mechanism as [`wildcard_doc_ids`]/
/// [`prefix_doc_ids`], built on
/// [`lucene_codecs::blocktree::FieldTerms::fuzzy_intersect`] and
/// [`lucene_codecs::fuzzy::FuzzyMatch`] instead of a glob pattern. Returns an
/// empty `Vec` -- not an error -- when `query.field` doesn't exist in this
/// segment, same "missing field means no matches" convention every other
/// clause follows.
fn fuzzy_doc_ids(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &FuzzyQuery,
) -> Result<Vec<i32>> {
    let Some(field_terms) = fields.field(&query.field) else {
        return Ok(Vec::new());
    };
    let pattern = FuzzyMatch::new(
        &query.term,
        query.max_edits,
        query.prefix_length,
        query.transpositions,
    );
    let matching_terms: Vec<Vec<u8>> = field_terms
        .fuzzy_intersect(&pattern)
        .map(|(term, _stats)| term.to_vec())
        .collect();
    let mut doc_ids: Vec<i32> = Vec::new();
    for term in &matching_terms {
        let Some(postings) = field_terms.postings(term, doc_in)? else {
            continue;
        };
        doc_ids.extend(
            postings
                .docs
                .iter()
                .copied()
                .filter(|&doc_id| live_docs.is_none_or(|bits| bits.get(doc_id as usize))),
        );
    }
    doc_ids.sort_unstable();
    doc_ids.dedup();
    Ok(doc_ids)
}

/// [`Clause::Regexp`]'s matched doc-ID list (task #43): same
/// union-across-matching-terms mechanism as [`wildcard_doc_ids`]/
/// [`prefix_doc_ids`]/[`fuzzy_doc_ids`], built on
/// [`lucene_codecs::blocktree::FieldTerms::regexp_intersect`] and
/// [`lucene_codecs::regexp::RegexpPattern`] instead of a glob/edit-distance
/// pattern. Returns an empty `Vec` -- not an error -- when `query.field`
/// doesn't exist in this segment, same "missing field means no matches"
/// convention every other clause follows; a malformed `query.pattern`
/// (unsupported regexp syntax) instead surfaces as [`Error::Regexp`],
/// propagated via `?` from [`RegexpPattern::new`] -- distinct from the
/// missing-field/missing-term case because a bad pattern is a caller
/// mistake, not a legitimate "matches nothing" outcome.
fn regexp_doc_ids(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &RegexpQuery,
) -> Result<Vec<i32>> {
    let Some(field_terms) = fields.field(&query.field) else {
        return Ok(Vec::new());
    };
    let pattern = RegexpPattern::new(query.pattern.as_bytes())?;
    let matching_terms: Vec<Vec<u8>> = field_terms
        .regexp_intersect(&pattern)
        .map(|(term, _stats)| term.to_vec())
        .collect();
    let mut doc_ids: Vec<i32> = Vec::new();
    for term in &matching_terms {
        let Some(postings) = field_terms.postings(term, doc_in)? else {
            continue;
        };
        doc_ids.extend(
            postings
                .docs
                .iter()
                .copied()
                .filter(|&doc_id| live_docs.is_none_or(|bits| bits.get(doc_id as usize))),
        );
    }
    doc_ids.sort_unstable();
    doc_ids.dedup();
    Ok(doc_ids)
}

/// Executes `query` (see [`query::BooleanQuery`] and this module's doc comment for
/// the exact matching semantics) against one already-opened segment, feeding every
/// matching **live** doc ID to `collector` in ascending order — same parameter
/// contract as [`search_term_query`], generalized to a `must`/`should`/`must_not`
/// clause list of `TermQuery`s instead of exactly one.
///
/// `pos_in`/`pay_in`: the segment's opened `.pos`/`.pay` files, needed only when
/// `query` (at any nesting depth) contains a `Clause::Phrase` with more than one
/// term (task #29's addition — see [`resolve_clause_docs`]). `None` is fine for
/// a query with no multi-term phrase clause; passing `None` for a query that
/// turns out to need it surfaces as [`Error::MissingPosInput`], same convention
/// as [`search_phrase_query`].
#[allow(clippy::too_many_arguments)]
pub fn search_boolean_query<C: Collector>(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    pos_in: Option<&PosInput<'_>>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &BooleanQuery,
    collector: &mut C,
) -> Result<()> {
    let Some(matched) = matched_boolean_docs(fields, doc_in, pos_in, pay_in, live_docs, query)?
    else {
        return Ok(());
    };
    for doc_id in matched {
        collector.collect(doc_id);
    }
    Ok(())
}

/// Counts, per doc ID, how many of `should_docs` (one ascending, live-filtered doc-ID
/// list per `should` clause, same shape [`term_doc_ids`] returns per clause) contain
/// that doc — the mechanism [`matched_boolean_docs`] needs to enforce
/// `minimum_should_match`, since a plain [`Disjunction`] only reports "this doc is in
/// the union of at least one clause", not "how many clauses agreed on it". Doc order
/// and duplicates across clauses (a doc appearing in more than one clause's list) are
/// both handled the same way a `HashMap<i32, usize>` tally naturally handles them —
/// same "count occurrences via a map" shape [`term_doc_positions`]'s per-term maps
/// already use in this module for a different purpose.
pub(crate) fn should_match_counts(should_docs: &[Vec<i32>]) -> HashMap<i32, usize> {
    let mut counts = HashMap::new();
    for docs in should_docs {
        for &doc_id in docs {
            *counts.entry(doc_id).or_insert(0) += 1;
        }
    }
    counts
}

/// Resolves one `must`/`should`/`must_not` [`Clause`] to its ascending,
/// live-filtered doc-ID list — the recursive core that lets [`matched_boolean_docs`]
/// treat a `Clause::Term` and a `Clause::Boolean` identically once resolved.
///
/// - `Clause::Term`: delegates straight to [`term_doc_ids`], same as before this
///   task's `Clause` generalization.
/// - `Clause::Boolean`: recursively calls [`matched_boolean_docs`] on the nested
///   query, which independently resolves *that* query's own
///   `must`/`should`/`must_not`/`minimum_should_match` (its own call to this same
///   function for each of its own clauses, however deep the nesting goes — genuine
///   recursion, not a hardcoded second level) before this function materializes the
///   result as one more doc-ID list for the parent's `Conjunction`/`Disjunction` to
///   merge like any other clause. A nested query that itself resolves to "matches
///   nothing" (`Ok(None)`, see `matched_boolean_docs`'s doc comment) contributes an
///   empty list here, not an error.
/// - `Clause::Phrase` (task #29's addition): collects [`search_phrase_query`]'s
///   matches into a `Vec` via a local [`VecCollector`], reusing that function's
///   matching logic (missing field/term/degenerate-single-term handling and
///   all) rather than duplicating it.
fn resolve_clause_docs(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    pos_in: Option<&PosInput<'_>>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    clause: &Clause,
) -> Result<Vec<i32>> {
    match clause {
        Clause::Term(query) => term_doc_ids(fields, doc_in, live_docs, query),
        Clause::Phrase(query) => {
            let mut collector = collector::VecCollector::default();
            search_phrase_query(
                fields,
                doc_in,
                pos_in,
                pay_in,
                live_docs,
                query,
                &mut collector,
            )?;
            Ok(collector.docs)
        }
        Clause::Boolean(nested) => Ok(matched_boolean_docs(
            fields, doc_in, pos_in, pay_in, live_docs, nested,
        )?
        .map(Iterator::collect)
        .unwrap_or_default()),
        Clause::DisjunctionMax(nested) => {
            resolve_dismax_docs(fields, doc_in, pos_in, pay_in, live_docs, nested)
        }
        Clause::ConstantScore(nested) => {
            resolve_clause_docs(fields, doc_in, pos_in, pay_in, live_docs, &nested.inner)
        }
        Clause::Boost(nested) => {
            resolve_clause_docs(fields, doc_in, pos_in, pay_in, live_docs, &nested.inner)
        }
        Clause::Wildcard(query) => wildcard_doc_ids(fields, doc_in, live_docs, query),
        Clause::Prefix(query) => prefix_doc_ids(fields, doc_in, live_docs, query),
        Clause::Fuzzy(query) => fuzzy_doc_ids(fields, doc_in, live_docs, query),
        Clause::Regexp(query) => regexp_doc_ids(fields, doc_in, live_docs, query),
        Clause::Span(query) => span_doc_ids(fields, doc_in, pos_in, pay_in, live_docs, query),
    }
}

/// Resolves a [`DisjunctionMaxQuery`]'s matched doc-ID list -- a doc matches
/// iff **any** disjunct matches (real `DisjunctionMaxQuery`'s matching
/// contract: it's a pure union, unlike `BooleanQuery.should`'s
/// `minimum_should_match`-gated disjunction). Each disjunct is resolved via
/// [`resolve_clause_docs`], same recursive treatment `Clause::Boolean` gets,
/// then merged through the same [`Disjunction`] iterator `matched_boolean_docs`
/// uses for its own `should`-only case, deduplicated and sorted ascending by
/// construction.
fn resolve_dismax_docs(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    pos_in: Option<&PosInput<'_>>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &DisjunctionMaxQuery,
) -> Result<Vec<i32>> {
    if query.disjuncts.is_empty() {
        // Real `DisjunctionMaxQuery` with no disjuncts matches nothing --
        // mirrors `BooleanQuery`'s own "no must/should clauses" -> matches
        // nothing rule (see `matched_boolean_docs`'s doc comment).
        return Ok(Vec::new());
    }
    let doc_lists: Vec<Vec<i32>> = query
        .disjuncts
        .iter()
        .map(|clause| resolve_clause_docs(fields, doc_in, pos_in, pay_in, live_docs, clause))
        .collect::<Result<_>>()?;
    let iters: Vec<BoxDocIter<'static>> = doc_lists
        .into_iter()
        .map(|v| Box::new(v.into_iter()) as BoxDocIter<'static>)
        .collect();
    Ok(Disjunction::new(iters).collect())
}

/// Shared matched-doc-set computation for [`search_boolean_query`] and
/// [`search_boolean_query_scored`] (previously duplicated between the two; unified
/// here since `minimum_should_match` handling would otherwise need implementing
/// twice) — see this module's doc comment for the exact semantics, including the
/// `minimum_should_match` interaction rules. Returns `Ok(None)` for the "no `must`
/// and no `should` clauses" case (real `BooleanQuery.rewrite()`'s `MatchNoDocsQuery`
/// outcome — see the doc comment), `Ok(Some(iter))` of the ascending, live-filtered
/// matched doc IDs otherwise.
///
/// Each clause is resolved via [`resolve_clause_docs`], which recurses into a
/// nested `Clause::Boolean`'s own call to this same function — this is also what
/// makes this function itself recursive (a nested query resolves via a fresh call
/// to `matched_boolean_docs`, not a duplicated copy of this algorithm).
#[allow(clippy::too_many_arguments)]
fn matched_boolean_docs(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    pos_in: Option<&PosInput<'_>>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &BooleanQuery,
) -> Result<Option<BoxDocIter<'static>>> {
    if query.must.is_empty() && query.should.is_empty() {
        // Real `BooleanQuery.rewrite()` turns both "no clauses at all" and "only
        // MUST_NOT clauses" into a `MatchNoDocsQuery` -- see this module's doc
        // comment. Neither case reaches the merge machinery below.
        return Ok(None);
    }

    let clause_docs = |clauses: &[Clause]| -> Result<Vec<Vec<i32>>> {
        clauses
            .iter()
            .map(|clause| resolve_clause_docs(fields, doc_in, pos_in, pay_in, live_docs, clause))
            .collect()
    };

    let to_iters = |docs: Vec<Vec<i32>>| -> Vec<BoxDocIter<'static>> {
        docs.into_iter()
            .map(|v| Box::new(v.into_iter()) as BoxDocIter<'static>)
            .collect()
    };

    let min_should_match = query.minimum_should_match;
    // `should_docs` is only needed when `should` actually participates in matching:
    // either as the base set (`must` empty) or as a `minimum_should_match` gate on
    // top of `must`'s conjunction. When `must` is non-empty and
    // `minimum_should_match == 0`, `should` stays purely score-only (matching
    // pre-task-#24 behavior exactly) and this never touches it.
    let should_docs = if query.must.is_empty() || min_should_match > 0 {
        Some(clause_docs(&query.should)?)
    } else {
        None
    };

    let base: BoxDocIter<'static> =
        if !query.must.is_empty() {
            let conjunction = Conjunction::new(to_iters(clause_docs(&query.must)?));
            if min_should_match > 0 {
                let counts = should_match_counts(should_docs.as_ref().expect("computed above"));
                Box::new(conjunction.filter(move |doc_id| {
                    counts.get(doc_id).copied().unwrap_or(0) >= min_should_match
                }))
            } else {
                Box::new(conjunction)
            }
        } else {
            let should_docs = should_docs.expect("computed above (must is empty)");
            if min_should_match > 1 {
                let counts = should_match_counts(&should_docs);
                let disjunction = Disjunction::new(to_iters(should_docs));
                Box::new(disjunction.filter(move |doc_id| {
                    counts.get(doc_id).copied().unwrap_or(0) >= min_should_match
                }))
            } else {
                // `min_should_match` is 0 or 1: a plain disjunction already requires "at
                // least one should clause matched", so no counting is needed.
                Box::new(Disjunction::new(to_iters(should_docs)))
            }
        };

    let matched: BoxDocIter<'static> = if query.must_not.is_empty() {
        base
    } else {
        let excluded: BoxDocIter<'static> =
            Box::new(Disjunction::new(to_iters(clause_docs(&query.must_not)?)));
        Box::new(Excluding::new(base, excluded))
    };

    Ok(Some(matched))
}

/// Shared per-clause lookup, scored sibling of [`term_doc_ids`]: same field/term/
/// `live_docs` handling, but returns `(doc_id, freq)` pairs (ascending by
/// `doc_id`) instead of discarding `freq`, so callers can compute a BM25 score
/// per doc. Returns an empty `Vec` for a missing field/term, same as
/// [`term_doc_ids`].
fn term_doc_freqs(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &TermQuery,
) -> Result<Vec<(i32, i32)>> {
    let Some(field_terms) = fields.field(&query.field) else {
        return Ok(Vec::new());
    };
    let Some(postings) = field_terms.postings(&query.term, doc_in)? else {
        return Ok(Vec::new());
    };
    Ok(postings
        .docs
        .iter()
        .copied()
        .zip(postings.freqs.iter().copied())
        .filter(|&(doc_id, _)| live_docs.is_none_or(|bits| bits.get(doc_id as usize)))
        .collect())
}

/// One clause's BM25 score per matching, live doc (ascending by `doc_id`) — see
/// [`similarity`]'s module doc for the formula. `norms`, when `Some`, supplies
/// this query's field's real per-doc/avg field length (see [`field_norms`]);
/// `None` falls back to [`similarity::UNNORMED_FIELD_LENGTH`] for both, a
/// documented approximation for a field with no opened norms. Returns an empty
/// `Vec` for a missing field/term (no score to compute), same as
/// [`term_doc_ids`].
fn term_doc_scores(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &TermQuery,
    norms: Option<&FieldNorms<'_>>,
) -> Result<Vec<(i32, f32)>> {
    let Some(field_terms) = fields.field(&query.field) else {
        return Ok(Vec::new());
    };
    let Some(stats) = field_terms.seek_exact(&query.term) else {
        return Ok(Vec::new());
    };
    let doc_freqs = term_doc_freqs(fields, doc_in, live_docs, query)?;
    let doc_count = field_terms.doc_count as i64;
    doc_freqs
        .into_iter()
        .map(|(doc_id, freq)| {
            let (field_length, avg_field_length) = match norms {
                Some(fn_) => (fn_.field_length(doc_id)?, fn_.avg_field_length),
                None => (
                    similarity::UNNORMED_FIELD_LENGTH,
                    similarity::UNNORMED_FIELD_LENGTH,
                ),
            };
            let score = similarity::score(
                stats.doc_freq as i64,
                doc_count,
                freq as f32,
                field_length,
                avg_field_length,
            );
            Ok((doc_id, score))
        })
        .collect()
}

/// Scored sibling of [`search_term_query`]: same matching semantics, but feeds
/// each matched, live doc's BM25 score (see [`similarity`]) to a
/// [`ScoringCollector`] instead of a plain [`Collector`]. `norms`: see
/// [`term_doc_scores`]'s doc comment — `Some(&FieldNorms)` for
/// `query.field`'s real per-doc/avg field length, `None` to fall back to the
/// constant approximation.
pub fn search_term_query_scored<C: ScoringCollector>(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &TermQuery,
    norms: Option<&FieldNorms<'_>>,
    collector: &mut C,
) -> Result<()> {
    for (doc_id, score) in term_doc_scores(fields, doc_in, live_docs, query, norms)? {
        collector.collect(doc_id, score);
    }
    Ok(())
}

/// Recursive per-clause `(doc_id -> score)` contribution, the scored sibling of
/// [`resolve_clause_docs`] used by [`search_boolean_query_scored`]:
///
/// - `Clause::Term`: this clause's own BM25 score per matching, live doc (via
///   [`term_doc_scores`]), keyed by `query.field` in `norms` same as before this
///   task's `Clause` generalization.
/// - `Clause::Boolean`: real Lucene sums a nested `BooleanQuery`'s own internal
///   score — itself the sum of *its* matching `must`/`should` sub-clauses' scores
///   — as one contribution to the parent's total. Implemented here by first
///   resolving the nested query's own matched-doc set (respecting its own
///   `must_not`/`minimum_should_match`, same as matching), then recursing into
///   this same function for each of the nested query's own `must`/`should`
///   sub-clauses and summing, restricted to docs the nested query itself
///   actually matched — a should-clause hit the nested query's own `must_not` or
///   `minimum_should_match` excludes must not leak a score contribution into the
///   parent. This recursion has no depth limit: a `Clause::Boolean` nested inside
///   another `Clause::Boolean` resolves the same way, one level at a time.
/// - `Clause::Phrase` (task #29's addition): this clause's own BM25 score per
///   matching, live doc via [`search_phrase_query_scored`], collected through a
///   local [`ScoringCollector`] (a tiny inline impl, since neither existing
///   collector in `collector.rs` needs to be shared for this one-shot use), keyed
///   by `query.field` in `norms` same as `Clause::Term`.
fn clause_scores(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    pos_in: Option<&PosInput<'_>>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    clause: &Clause,
    norms: Option<&HashMap<String, FieldNorms<'_>>>,
) -> Result<HashMap<i32, f32>> {
    match clause {
        Clause::Term(query) => {
            let clause_norms = norms.and_then(|m| m.get(&query.field));
            let mut scores = HashMap::new();
            for (doc_id, score) in term_doc_scores(fields, doc_in, live_docs, query, clause_norms)?
            {
                *scores.entry(doc_id).or_insert(0.0) += score;
            }
            Ok(scores)
        }
        Clause::Phrase(query) => {
            let clause_norms = norms.and_then(|m| m.get(&query.field));
            let mut scores: HashMap<i32, f32> = HashMap::new();
            struct SumCollector<'a>(&'a mut HashMap<i32, f32>);
            impl collector::ScoringCollector for SumCollector<'_> {
                fn collect(&mut self, doc_id: i32, score: f32) {
                    *self.0.entry(doc_id).or_insert(0.0) += score;
                }
            }
            let mut collector = SumCollector(&mut scores);
            search_phrase_query_scored(
                fields,
                doc_in,
                pos_in,
                pay_in,
                live_docs,
                query,
                clause_norms,
                &mut collector,
            )?;
            Ok(scores)
        }
        Clause::Boolean(nested) => {
            let Some(matched) =
                matched_boolean_docs(fields, doc_in, pos_in, pay_in, live_docs, nested)?
            else {
                return Ok(HashMap::new());
            };
            let matched: std::collections::HashSet<i32> = matched.collect();

            let mut scores: HashMap<i32, f32> = HashMap::new();
            for sub_clause in nested.must.iter().chain(nested.should.iter()) {
                for (doc_id, score) in
                    clause_scores(fields, doc_in, pos_in, pay_in, live_docs, sub_clause, norms)?
                {
                    if matched.contains(&doc_id) {
                        *scores.entry(doc_id).or_insert(0.0) += score;
                    }
                }
            }
            Ok(scores)
        }
        Clause::DisjunctionMax(nested) => {
            dismax_scores(fields, doc_in, pos_in, pay_in, live_docs, nested, norms)
        }
        Clause::ConstantScore(nested) => {
            let matched =
                resolve_clause_docs(fields, doc_in, pos_in, pay_in, live_docs, &nested.inner)?;
            Ok(matched
                .into_iter()
                .map(|doc_id| (doc_id, nested.score))
                .collect())
        }
        Clause::Boost(nested) => {
            let inner_scores = clause_scores(
                fields,
                doc_in,
                pos_in,
                pay_in,
                live_docs,
                &nested.inner,
                norms,
            )?;
            Ok(inner_scores
                .into_iter()
                .map(|(doc_id, score)| (doc_id, score * nested.boost))
                .collect())
        }
        Clause::Wildcard(query) => {
            // Unscored: flat 1.0 per matching doc -- see `WildcardQuery`'s doc
            // comment in `query.rs` for why (no single term's frequency/idf to
            // score against for a multi-term match).
            let matched = wildcard_doc_ids(fields, doc_in, live_docs, query)?;
            Ok(matched
                .into_iter()
                .map(|doc_id| (doc_id, 1.0_f32))
                .collect())
        }
        Clause::Prefix(query) => {
            // Unscored: flat 1.0 per matching doc -- see `PrefixQuery`'s doc
            // comment for why (same rationale as `Clause::Wildcard`'s arm
            // above).
            let matched = prefix_doc_ids(fields, doc_in, live_docs, query)?;
            Ok(matched
                .into_iter()
                .map(|doc_id| (doc_id, 1.0_f32))
                .collect())
        }
        Clause::Fuzzy(query) => {
            // Unscored: flat 1.0 per matching doc -- see `FuzzyQuery`'s doc
            // comment for why (same rationale as `Clause::Wildcard`'s arm
            // above).
            let matched = fuzzy_doc_ids(fields, doc_in, live_docs, query)?;
            Ok(matched
                .into_iter()
                .map(|doc_id| (doc_id, 1.0_f32))
                .collect())
        }
        Clause::Regexp(query) => {
            // Unscored: flat 1.0 per matching doc -- see `RegexpQuery`'s doc
            // comment for why (same rationale as `Clause::Wildcard`'s arm
            // above).
            let matched = regexp_doc_ids(fields, doc_in, live_docs, query)?;
            Ok(matched
                .into_iter()
                .map(|doc_id| (doc_id, 1.0_f32))
                .collect())
        }
        Clause::Span(query) => {
            // Unscored: flat 1.0 per matching doc -- see `SpanQuery`'s doc
            // comment for why (same rationale as `Clause::Wildcard`'s arm
            // above -- real span-aware scoring is a separate, unscoped
            // problem).
            let matched = span_doc_ids(fields, doc_in, pos_in, pay_in, live_docs, query)?;
            Ok(matched
                .into_iter()
                .map(|doc_id| (doc_id, 1.0_f32))
                .collect())
        }
    }
}

/// Real `DisjunctionMaxQuery.DisjunctionMaxWeight`/`DisjunctionMaxScorer`'s
/// scoring formula: for each doc matching at least one disjunct, its score is
/// `max(disjunct_scores) + tie_breaker * sum(every other matching disjunct's
/// score)` -- exactly Lucene's own formula (**not** an approximation; see this
/// function's doc comment on `DisjunctionMaxQuery` in `query.rs` for the
/// citation), computed per-disjunct via [`clause_scores`] (the same recursive
/// per-clause scorer `Clause::Boolean` already uses, so a `Clause::Boolean` or
/// nested `Clause::DisjunctionMax` disjunct scores correctly to arbitrary
/// depth). A doc appearing in zero disjuncts' score maps never appears in the
/// result at all (matching [`resolve_dismax_docs`]'s "union" matching
/// contract -- scoring and matching agree on which docs are present).
fn dismax_scores(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    pos_in: Option<&PosInput<'_>>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &DisjunctionMaxQuery,
    norms: Option<&HashMap<String, FieldNorms<'_>>>,
) -> Result<HashMap<i32, f32>> {
    let per_disjunct: Vec<HashMap<i32, f32>> = query
        .disjuncts
        .iter()
        .map(|clause| clause_scores(fields, doc_in, pos_in, pay_in, live_docs, clause, norms))
        .collect::<Result<_>>()?;

    // Every doc appearing in at least one disjunct's score map.
    let mut all_docs: std::collections::HashSet<i32> = std::collections::HashSet::new();
    for scores in &per_disjunct {
        all_docs.extend(scores.keys().copied());
    }

    let mut result = HashMap::new();
    for doc_id in all_docs {
        let mut max_score = f32::NEG_INFINITY;
        let mut sum_score = 0.0f32;
        for scores in &per_disjunct {
            if let Some(&score) = scores.get(&doc_id) {
                sum_score += score;
                if score > max_score {
                    max_score = score;
                }
            }
        }
        let other_sum = sum_score - max_score;
        result.insert(doc_id, max_score + query.tie_breaker * other_sum);
    }
    Ok(result)
}

/// Scored sibling of [`search_boolean_query`]: computes the same matched-doc set
/// (`must`'s conjunction, else `should`'s disjunction, minus `must_not`'s
/// disjunction — identical rules to [`search_boolean_query`], see this module's
/// doc comment), but reports each matched doc's score as the **sum of its BM25
/// score across every `must`/`should` clause it satisfies** (mirroring real
/// Lucene's additive `BooleanScorer`; `must_not` clauses never contribute to the
/// score, matching `Occur.MUST_NOT`'s filter-only contract). A `Clause::Boolean`
/// clause contributes its own nested score recursively — see [`clause_scores`]'s
/// doc comment for the exact recursive rule and how it stays correct to
/// arbitrary nesting depth.
///
/// `norms`: real per-doc/avg field length, keyed by field name, for every
/// scored (`must`/`should`) clause's field, at every nesting depth — a clause
/// whose field has no entry in this map (or when `norms` itself is `None`) falls
/// back to [`similarity::UNNORMED_FIELD_LENGTH`] for that clause, same documented
/// approximation as [`term_doc_scores`]. A `BooleanQuery`'s clauses can span
/// multiple fields, unlike a single [`TermQuery`], hence the map instead of one
/// `FieldNorms`.
///
/// `pos_in`/`pay_in`: see [`search_boolean_query`]'s doc comment -- same
/// contract, needed only when `query` contains a multi-term `Clause::Phrase` at
/// any nesting depth.
#[allow(clippy::too_many_arguments)]
pub fn search_boolean_query_scored<C: ScoringCollector>(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    pos_in: Option<&PosInput<'_>>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &BooleanQuery,
    norms: Option<&HashMap<String, FieldNorms<'_>>>,
    collector: &mut C,
) -> Result<()> {
    let Some(matched) = matched_boolean_docs(fields, doc_in, pos_in, pay_in, live_docs, query)?
    else {
        return Ok(());
    };

    // Sum each scoring clause's (doc_id -> score) contributions across `must`
    // and `should` (never `must_not`, which only filters -- see doc comment).
    let mut scores: HashMap<i32, f32> = HashMap::new();
    for clause in query.must.iter().chain(query.should.iter()) {
        for (doc_id, score) in
            clause_scores(fields, doc_in, pos_in, pay_in, live_docs, clause, norms)?
        {
            *scores.entry(doc_id).or_insert(0.0) += score;
        }
    }

    for doc_id in matched {
        collector.collect(doc_id, scores.get(&doc_id).copied().unwrap_or(0.0));
    }
    Ok(())
}

/// `DisjunctionMaxQuery`-equivalent (task #32): reports every doc matching at
/// least one of `query.disjuncts` (a pure union -- see [`resolve_dismax_docs`]'s
/// doc comment) to `collector`, in ascending doc-ID order. Same
/// `pos_in`/`pay_in` contract as [`search_boolean_query`]: `None` is fine
/// unless a disjunct contains a multi-term `Clause::Phrase` at any nesting
/// depth, surfacing `Error::MissingPosInput` only then.
#[allow(clippy::too_many_arguments)]
pub fn search_disjunction_max_query<C: Collector>(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    pos_in: Option<&PosInput<'_>>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &DisjunctionMaxQuery,
    collector: &mut C,
) -> Result<()> {
    let matched = resolve_dismax_docs(fields, doc_in, pos_in, pay_in, live_docs, query)?;
    for doc_id in matched {
        collector.collect(doc_id);
    }
    Ok(())
}

/// Scored sibling of [`search_disjunction_max_query`]: computes the identical
/// matched-doc set, reporting each doc's score via real Lucene's exact dismax
/// formula (see [`dismax_scores`]'s doc comment) -- `max(disjunct scores) +
/// tie_breaker * sum(every other matching disjunct's score)`. `norms`: same
/// contract as [`search_boolean_query_scored`]'s -- per-field real norms,
/// keyed by field name, for every disjunct's field at any nesting depth;
/// falls back to [`similarity::UNNORMED_FIELD_LENGTH`] for an unlisted field
/// or when `norms` itself is `None`.
#[allow(clippy::too_many_arguments)]
pub fn search_disjunction_max_query_scored<C: ScoringCollector>(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    pos_in: Option<&PosInput<'_>>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &DisjunctionMaxQuery,
    norms: Option<&HashMap<String, FieldNorms<'_>>>,
    collector: &mut C,
) -> Result<()> {
    let scores = dismax_scores(fields, doc_in, pos_in, pay_in, live_docs, query, norms)?;
    let mut docs: Vec<i32> = scores.keys().copied().collect();
    docs.sort_unstable();
    for doc_id in docs {
        collector.collect(doc_id, scores[&doc_id]);
    }
    Ok(())
}

/// Checks whether `term_positions` (one sorted, ascending position list per phrase
/// term, in phrase order, all for the *same* doc) has some base position `p` such
/// that `term_positions[i]` contains `p + i` for every `i` -- `ExactPhraseScorer`'s
/// core test (`org.apache.lucene.search.ExactPhraseScorer`, slop == 0 case), done
/// here as a straightforward candidate-and-check rather than Java's stateful
/// per-postings merge: every position in `term_positions[0]` is a candidate base
/// `p`, and each candidate is checked against every other term's position list via
/// binary search (each list is already sorted, since positions are decoded and
/// grouped in increasing order by [`lucene_codecs::postings::read_positions`]).
///
/// **Edge cases, verified by this function's own unit tests below**: an empty
/// `term_positions` (no terms at all) or any single empty position list (a term
/// with zero occurrences in this doc, which callers should never actually pass in
/// practice -- doc-level conjunction already guarantees every term occurs at least
/// once) both yield `false` rather than panicking. A single-term phrase
/// (`term_positions.len() == 1`) degenerates to "does this term occur at all in
/// this doc": the inner loop over `1..len` is empty, so the first candidate
/// position always succeeds. A repeated term (e.g. "the the") works unmodified --
/// the two position lists happen to be identical, but the check only ever compares
/// `p + i` against list `i`, never compares lists against each other by identity.
pub(crate) fn phrase_matches_in_doc(term_positions: &[Vec<i32>]) -> bool {
    let Some((first, rest)) = term_positions.split_first() else {
        return false;
    };
    if rest.iter().any(|positions| positions.is_empty()) {
        return false;
    }
    'candidate: for &p0 in first {
        for (i, positions) in rest.iter().enumerate() {
            let target = p0 + (i as i32 + 1);
            if positions.binary_search(&target).is_err() {
                continue 'candidate;
            }
        }
        return true;
    }
    false
}

/// Sloppy (`slop > 0`) sibling of [`phrase_matches_in_doc`]: checks whether
/// `term_positions` (same shape/contract as `phrase_matches_in_doc` — one sorted,
/// ascending position list per phrase term, in phrase order, all for the same doc)
/// has some strictly-increasing, in-order alignment `p_0 < p_1 < ... <
/// p_{n-1}` (one position per term, `p_i` drawn from `term_positions[i]`) whose
/// **total "move" distance** is at most `slop`.
///
/// **Exact formula implemented, and where it comes from**: real Lucene's
/// `PhraseQuery` Javadoc (`org.apache.lucene.search.PhraseQuery`, "the slop
/// parameter") describes slop as "the number of positions all words need to move
/// to line up in order" — a term one word away from its expected adjacent slot
/// costs one "move", two words away costs two, and so on. For an alignment
/// `p_0 < p_1 < ... < p_{n-1}` in that order, the total moves needed is the sum of
/// each adjacent gap's slack: `sum_{i=1}^{n-1} (p_i - p_{i-1} - 1)`, which
/// telescopes to `(p_{n-1} - p_0) - (n - 1)` regardless of the intermediate
/// positions chosen. A doc matches iff some such alignment has
/// `(p_{n-1} - p_0) - (n - 1) <= slop`.
///
/// **Scope, stated precisely (see `docs/parity.md`)**: this is an **in-order-only**
/// implementation of real Lucene's sloppy matching — it requires
/// `p_0 < p_1 < ... < p_{n-1}` strictly increasing in phrase order, matching real
/// Lucene's common case (phrase terms found in their query order, just spread
/// apart by up to `slop` extra words). Real Lucene's general `SloppyPhraseMatcher`
/// (`org.apache.lucene.search.SloppyPhraseMatcher`) additionally allows term
/// **reordering** within the slop budget (e.g. "quick fox" matching "fox... quick"
/// at a high enough slop, via a priority-queue-based edit-distance computation over
/// `PhrasePositions`) — that general algorithm is *not* implemented here; this
/// port could not confidently re-derive/verify its exact edit-distance formula
/// against real Lucene's source within this task's scope, so reordering is
/// deliberately out of scope rather than guessed at. Every test below proves only
/// this function's own stated in-order formula, not full Lucene byte-for-byte
/// parity for the reordering case.
///
/// For a fixed starting position `p_0`, the smallest valid alignment (and hence
/// the minimum possible `p_{n-1}`, and thus the minimum possible move count for
/// that `p_0`) is found by a simple greedy scan: for each subsequent term, pick
/// the smallest position in its list that's strictly greater than the previous
/// term's chosen position. Picking any larger position could only increase (never
/// decrease) the running total, so greedy is optimal for a fixed `p_0`; every
/// `p_0` in the first term's own position list is tried in turn (same
/// candidate-and-check structure as `phrase_matches_in_doc`).
///
/// **Edge cases** (matching `phrase_matches_in_doc`'s own contract): an empty
/// `term_positions` or any single empty position list both yield `false`. A
/// single-term phrase (`term_positions.len() == 1`) degenerates to "does this term
/// occur at all" regardless of `slop`. `slop == 0` is equivalent to
/// `phrase_matches_in_doc` (a zero move budget forces every gap to be exactly
/// `0`, i.e. exact adjacency) — [`search_phrase_query`] still calls the dedicated
/// exact-match fast path for `slop == 0` rather than this function, but this
/// function's own unit tests confirm the `slop == 0` case agrees.
pub(crate) fn phrase_matches_in_doc_sloppy(term_positions: &[Vec<i32>], slop: u32) -> bool {
    let Some((first, rest)) = term_positions.split_first() else {
        return false;
    };
    if rest.iter().any(|positions| positions.is_empty()) {
        return false;
    }
    if rest.is_empty() {
        // Single-term phrase: any occurrence at all is a match, same as
        // `phrase_matches_in_doc`.
        return !first.is_empty();
    }
    let slop = slop as i64;
    'candidate: for &p0 in first {
        let mut prev = p0;
        let mut total_moves: i64 = 0;
        for positions in rest {
            // Smallest position strictly greater than `prev` -- `partition_point`
            // finds the first index where `positions[idx] > prev` since the list is
            // sorted ascending.
            let idx = positions.partition_point(|&x| x <= prev);
            let Some(&pos) = positions.get(idx) else {
                continue 'candidate;
            };
            total_moves += i64::from(pos - prev - 1);
            if total_moves > slop {
                continue 'candidate;
            }
            prev = pos;
        }
        return true;
    }
    false
}

/// `ExactPhraseScorer`'s per-doc `phraseFreq`-equivalent: counts every valid base
/// position `p0` in `term_positions[0]` for which the rest of `term_positions`
/// align exactly (`term_positions[i]` contains `p0 + i` for every `i`) — the
/// same alignment condition [`phrase_matches_in_doc`] checks, except this counts
/// every satisfying `p0` instead of stopping at the first one.
///
/// **Why counting distinct `p0` values in the first term's own (already
/// deduplicated, strictly ascending) position list can't double-count the same
/// occurrence**: each `p0` is one real position of `term_positions[0]` in the
/// doc, and a real doc position occurs at most once in that list (positions are
/// decoded in strictly increasing order — see
/// [`lucene_codecs::postings::read_positions`]), so every counted match starts
/// at a genuinely distinct occurrence of the phrase's first word — this is
/// exactly `ExactPhraseScorer`'s own counting granularity: one match per
/// starting position of the phrase's first term that the rest of the phrase
/// aligns to. A repeated phrase (e.g. "the the" appearing twice, positions
/// 0,1,2,3 for "the") is counted once per valid starting position (0 and 2 for
/// non-overlapping repeats, or 0 *and* 1 if "the the the" — position 1 also
/// starts a valid "the the" alignment against positions 2 — matching real
/// Lucene's own per-start-position counting, which does not suppress
/// overlapping matches).
///
/// **Edge cases** (same contract as [`phrase_matches_in_doc`]): an empty
/// `term_positions`, or any single empty position list, both yield `0`. A
/// single-term phrase (`term_positions.len() == 1`) counts every occurrence of
/// that lone term (the inner alignment loop is empty, so every `p0` counts).
pub(crate) fn phrase_freq_exact(term_positions: &[Vec<i32>]) -> i32 {
    let Some((first, rest)) = term_positions.split_first() else {
        return 0;
    };
    if rest.iter().any(|positions| positions.is_empty()) {
        return 0;
    }
    let mut freq = 0;
    'candidate: for &p0 in first {
        for (i, positions) in rest.iter().enumerate() {
            let target = p0 + (i as i32 + 1);
            if positions.binary_search(&target).is_err() {
                continue 'candidate;
            }
        }
        freq += 1;
    }
    freq
}

/// Key type for [`span_matches_in_doc`]'s `doc_positions` map: one leaf
/// `SpanQuery::SpanTerm`'s `(field, term)` pair, the finest granularity a span
/// query ever needs a position list for (unlike `PhraseQuery`, a `SpanQuery`'s
/// leaves aren't all implicitly the same field — see [`SpanQuery`]'s doc
/// comment).
type SpanLeafKey = (String, Vec<u8>);

/// Computes `query`'s matching span ranges (`[start, end)` position pairs, real
/// `SpanTermQuery`/`SpanNearQuery`/`SpanOrQuery`'s per-doc result shape) against
/// one doc's already-decoded position lists, `doc_positions` -- one sorted,
/// ascending position list per distinct `SpanQuery::SpanTerm` leaf appearing
/// anywhere in `query` (a leaf whose `(field, term)` pair has no entry, or an
/// empty entry, is treated as "no occurrences in this doc", same convention
/// [`phrase_matches_in_doc`]'s callers already rely on -- see
/// [`span_doc_ids`]'s doc comment for how this map is built).
///
/// **Scope**: this is the direct, in-memory span computation
/// [`SpanQuery`]'s own doc comment describes (not a lazy `Spans` iterator) --
/// callers needing every matching span range for a doc call this once per
/// doc, the same shape `phrase_matches_in_doc`/`phrase_matches_in_doc_sloppy`
/// already use for `PhraseQuery`.
///
/// - `SpanQuery::SpanTerm`: every occurrence in `doc_positions` becomes exactly
///   one `(position, position + 1)` span -- real `SpanTermQuery`'s exact
///   semantics (`termFreq` occurrences, not just "does it occur").
/// - `SpanQuery::SpanOr`: the union (sorted, deduplicated) of every
///   sub-`SpanQuery`'s own spans -- real `SpanOrQuery`'s exact semantics.
/// - `SpanQuery::SpanNear`: delegates to [`span_near_matches`] -- see that
///   function's doc comment for the `slop`/`in_order` algorithm, including the
///   `in_order == false` any-order case that's this type's key differentiator
///   from `PhraseQuery`'s in-order-only sloppy matching.
///
/// Returned spans are sorted ascending and deduplicated (`(start, end)`
/// lexicographic order) regardless of variant, so a caller can treat "matches"
/// as simply "the returned `Vec` is non-empty" without caring which variant
/// produced it -- exactly how [`span_doc_ids`] uses this function.
pub(crate) fn span_matches_in_doc(
    query: &SpanQuery,
    doc_positions: &HashMap<SpanLeafKey, Vec<i32>>,
) -> Vec<(i32, i32)> {
    match query {
        SpanQuery::SpanTerm { field, term } => {
            let key = (field.clone(), term.clone());
            doc_positions
                .get(&key)
                .map(|positions| positions.iter().map(|&p| (p, p + 1)).collect())
                .unwrap_or_default()
        }
        SpanQuery::SpanOr { clauses } => {
            let mut spans: Vec<(i32, i32)> = clauses
                .iter()
                .flat_map(|clause| span_matches_in_doc(clause, doc_positions))
                .collect();
            spans.sort_unstable();
            spans.dedup();
            spans
        }
        SpanQuery::SpanNear {
            clauses,
            slop,
            in_order,
        } => span_near_matches(clauses, *slop, *in_order, doc_positions),
    }
}

/// [`SpanQuery::SpanNear`]'s matching algorithm (real `SpanNearQuery`'s
/// `NearSpansOrdered`/`NearSpansUnordered` equivalent, computed directly
/// rather than via a lazy iterator -- see [`span_matches_in_doc`]'s doc
/// comment for the scope decision): every `clauses[i]`'s own spans (computed
/// recursively via [`span_matches_in_doc`], so a `SpanNear` of `SpanNear`s
/// composes for free) are combined, one span chosen per clause, and a
/// combination is a match iff its chosen spans satisfy the `in_order`
/// arrangement below with total positional slack at most `slop`.
///
/// **`in_order == true`**: the chosen spans must already be non-overlapping
/// and increasing in `clauses`' own order -- `chosen[i].1 <= chosen[i + 1].0`
/// for every adjacent pair (span `i` ends at or before span `i + 1` starts).
/// This is real `SpanNearQuery(clauses, slop, true)`'s ordering requirement:
/// a reversed pair (clause 1's occurrence sits before clause 0's) never
/// satisfies it, at any slop -- exactly the case [`PhraseQuery`]'s own
/// in-order sloppy matching also rejects.
///
/// **`in_order == false`**: the chosen spans, **sorted by start position**
/// (not by `clauses`' order), must satisfy that same non-overlapping,
/// increasing condition -- any relative order among the clauses is accepted,
/// provided the spans still fit together without overlapping. This is the
/// capability [`PhraseQuery`]'s sloppy matching does *not* have (see that
/// function's own doc comment) -- a reversed pair (clause 1's occurrence
/// before clause 0's) matches here as long as the total slack fits `slop`,
/// which is exactly what distinguishes `SpanNearQuery(slop, false)` from a
/// sloppy phrase.
///
/// **Slop formula**, applied to the arranged (in-order or sorted, per above)
/// spans: the total slack is `sum(next.start - prev.end)` over every adjacent
/// pair -- `0` when spans touch exactly end-to-start with no gap, growing by
/// one for every extra intervening position, the same "moves needed to line
/// up" accounting [`phrase_matches_in_doc_sloppy`]'s doc comment derives for
/// `PhraseQuery`, generalized from single positions to `[start, end)` span
/// ranges. A combination whose arranged spans overlap (`next.start <
/// prev.end`) is rejected outright, regardless of `slop` -- overlapping
/// sub-spans have no defined "gap" to charge against the budget.
///
/// The overall span reported for a matching combination is `(min start, max
/// end)` across every chosen sub-span -- the smallest range containing every
/// sub-span, matching real `Spans`' own near-match span extent.
///
/// **Complexity**: this evaluates every combination of one span per clause
/// (a cartesian product) -- acceptable for this port's honestly-scoped MVP
/// (see [`SpanQuery`]'s doc comment) given the same "correctness first,
/// profile before optimizing" call this crate's other multi-term matchers
/// already make (`rust-performance` skill), but not the sub-linear
/// early-termination a real lazy `NearSpans` iterator gets.
///
/// **Edge cases**: an empty `clauses` list, or any clause whose own spans are
/// empty (the sub-query doesn't occur at all in this doc), both yield no
/// spans -- a `SpanNear` needs every sub-clause to contribute at least one
/// occurrence.
fn span_near_matches(
    clauses: &[SpanQuery],
    slop: u32,
    in_order: bool,
    doc_positions: &HashMap<SpanLeafKey, Vec<i32>>,
) -> Vec<(i32, i32)> {
    if clauses.is_empty() {
        return Vec::new();
    }
    let per_clause_spans: Vec<Vec<(i32, i32)>> = clauses
        .iter()
        .map(|clause| span_matches_in_doc(clause, doc_positions))
        .collect();
    if per_clause_spans.iter().any(Vec::is_empty) {
        return Vec::new();
    }

    let slop = i64::from(slop);
    let mut results: Vec<(i32, i32)> = Vec::new();
    let mut chosen: Vec<(i32, i32)> = Vec::with_capacity(clauses.len());
    combine_span_clauses(&per_clause_spans, &mut chosen, slop, in_order, &mut results);
    results.sort_unstable();
    results.dedup();
    results
}

/// Recursive cartesian-product helper for [`span_near_matches`]: picks one
/// span per entry in `per_clause_spans` (via `chosen`, built up one clause at
/// a time) and, once a full combination is chosen, checks its `in_order`/
/// `slop` validity and appends the resulting overall span to `results` if
/// valid -- see [`span_near_matches`]'s doc comment for the exact checks.
fn combine_span_clauses(
    per_clause_spans: &[Vec<(i32, i32)>],
    chosen: &mut Vec<(i32, i32)>,
    slop: i64,
    in_order: bool,
    results: &mut Vec<(i32, i32)>,
) {
    let Some((spans, rest)) = per_clause_spans.split_first() else {
        // Every clause has now contributed one span -- validate this
        // combination.
        let mut arranged = chosen.clone();
        if !in_order {
            arranged.sort_unstable_by_key(|span| span.0);
        }
        let mut slack: i64 = 0;
        for pair in arranged.windows(2) {
            let (prev, next) = (pair[0], pair[1]);
            if next.0 < prev.1 {
                // Overlapping sub-spans have no defined gap -- invalid at any
                // slop.
                return;
            }
            slack += i64::from(next.0 - prev.1);
        }
        if slack <= slop {
            let start = arranged
                .iter()
                .map(|span| span.0)
                .min()
                .expect("non-empty: at least one clause");
            let end = arranged
                .iter()
                .map(|span| span.1)
                .max()
                .expect("non-empty: at least one clause");
            results.push((start, end));
        }
        return;
    };
    for &span in spans {
        chosen.push(span);
        combine_span_clauses(rest, chosen, slop, in_order, results);
        chosen.pop();
    }
}

/// Collects every distinct `SpanQuery::SpanTerm` leaf's `(field, term)` pair
/// appearing anywhere in `query` (recursively through `SpanNear`/`SpanOr`),
/// deduplicated -- the set of position lists [`span_doc_ids`] needs to fetch
/// before it can evaluate [`span_matches_in_doc`] for any candidate doc.
fn collect_span_leaves(query: &SpanQuery, leaves: &mut Vec<SpanLeafKey>) {
    match query {
        SpanQuery::SpanTerm { field, term } => leaves.push((field.clone(), term.clone())),
        SpanQuery::SpanNear { clauses, .. } | SpanQuery::SpanOr { clauses } => {
            for clause in clauses {
                collect_span_leaves(clause, leaves);
            }
        }
    }
}

/// [`Clause::Span`]'s matched doc-ID list (task #55): gathers every distinct
/// leaf `(field, term)` pair `query` touches (via [`collect_span_leaves`]),
/// fetches each one's live-filtered `doc_id -> position list` map (via
/// [`term_doc_positions`], the same helper [`search_phrase_query`] uses), then
/// for every doc appearing in **any** leaf's doc list (a safe, simple
/// over-approximation of the true candidate set -- see this function's own
/// doc comment below for why that's fine) builds a per-doc `doc_positions` map
/// and checks [`span_matches_in_doc`] for a non-empty result.
///
/// **Why "any leaf's doc list" instead of computing each variant's own tighter
/// candidate set** (e.g. a `SpanNear`'s candidates could be the *conjunction*
/// of its sub-clauses' doc lists, `SpanOr`'s the union): this port takes the
/// simpler, uniformly-correct union-of-every-leaf approach for
/// [`SpanQuery`]'s honestly-scoped MVP (see that type's doc comment) --
/// [`span_matches_in_doc`] itself already correctly reports "no match" for a
/// candidate doc that doesn't actually satisfy a `SpanNear`'s stricter
/// requirement, so the wider candidate set costs only some wasted position-
/// list lookups, never an incorrect result. A future optimization pass could
/// tighten this per-variant if profiling shows it matters (same "correctness
/// first, profile before optimizing" call this crate's other multi-term
/// matchers already make).
///
/// Returns an empty `Vec` -- not an error -- when `query` has no leaves at all
/// (a `SpanNear`/`SpanOr` with empty `clauses`, which can never match) or when
/// every leaf's field/term is missing from this segment. Requires `pos_in` (an
/// `Err(Error::MissingPosInput)` otherwise) whenever `query` has at least one
/// leaf -- unlike `PhraseQuery`, even a single-leaf `SpanTerm` needs a real
/// position list (its spans are per-occurrence `(position, position + 1)`
/// pairs, not just doc-level presence), so there is no single-term fast path
/// that skips positions the way [`search_phrase_query`] has for a length-1
/// phrase.
#[allow(clippy::too_many_arguments)]
fn span_doc_ids(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    pos_in: Option<&PosInput<'_>>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &SpanQuery,
) -> Result<Vec<i32>> {
    let mut leaves: Vec<SpanLeafKey> = Vec::new();
    collect_span_leaves(query, &mut leaves);
    if leaves.is_empty() {
        return Ok(Vec::new());
    }
    let Some(pos_in) = pos_in else {
        return Err(Error::MissingPosInput);
    };
    leaves.sort_unstable();
    leaves.dedup();

    let mut candidate_docs: Vec<i32> = Vec::new();
    let mut per_leaf_maps: Vec<(SpanLeafKey, HashMap<i32, Vec<i32>>)> =
        Vec::with_capacity(leaves.len());
    for (field, term) in &leaves {
        let Some((docs, map)) =
            term_doc_positions(fields, doc_in, pos_in, pay_in, live_docs, field, term)?
        else {
            // A missing leaf term contributes no occurrences anywhere --
            // `span_matches_in_doc` already treats an absent map entry the
            // same way, so this leaf simply never adds candidate docs.
            continue;
        };
        candidate_docs.extend(docs);
        per_leaf_maps.push(((field.clone(), term.clone()), map));
    }
    candidate_docs.sort_unstable();
    candidate_docs.dedup();

    let mut result = Vec::new();
    for doc_id in candidate_docs {
        let mut doc_positions: HashMap<SpanLeafKey, Vec<i32>> = HashMap::new();
        for (key, map) in &per_leaf_maps {
            if let Some(positions) = map.get(&doc_id) {
                doc_positions.insert(key.clone(), positions.clone());
            }
        }
        if !span_matches_in_doc(query, &doc_positions).is_empty() {
            result.push(doc_id);
        }
    }
    Ok(result)
}

/// Executes `query` (see [`query::SpanQuery`] for the exact matching
/// semantics and this port's Spans-API-vs-direct-computation scope decision)
/// against one already-opened segment, feeding every matching **live** doc ID
/// to `collector` in ascending order -- same parameter contract as
/// [`search_phrase_query`], with `pos_in` required whenever `query` has at
/// least one leaf (see [`span_doc_ids`]'s doc comment for why `SpanQuery` has
/// no single-leaf fast path that skips positions the way a length-1
/// `PhraseQuery` does).
pub fn search_span_query<C: Collector>(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    pos_in: Option<&PosInput<'_>>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &SpanQuery,
    collector: &mut C,
) -> Result<()> {
    for doc_id in span_doc_ids(fields, doc_in, pos_in, pay_in, live_docs, query)? {
        collector.collect(doc_id);
    }
    Ok(())
}

/// One phrase-query term's live-filtered doc-ID list plus a `doc_id -> sorted
/// position list` map for that same term, or `None` when the field/term doesn't
/// exist (mirrors [`term_doc_ids`]'s "missing is not an error" convention). The map
/// (rather than a `Vec` aligned to the doc list) is what [`search_phrase_query`]
/// needs: after computing the doc-level conjunction across every term, it looks up
/// each candidate doc's position list per term by doc ID, not by index.
type TermDocPositions = (Vec<i32>, HashMap<i32, Vec<i32>>);

fn term_doc_positions(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    pos_in: &PosInput<'_>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    field: &str,
    term: &[u8],
) -> Result<Option<TermDocPositions>> {
    let Some(field_terms) = fields.field(field) else {
        return Ok(None);
    };
    let Some(postings) = field_terms.postings(term, doc_in)? else {
        return Ok(None);
    };
    let Some(positions) = field_terms.positions(term, doc_in, pos_in, pay_in)? else {
        return Ok(None);
    };

    let mut docs = Vec::with_capacity(postings.docs.len());
    let mut map = HashMap::with_capacity(postings.docs.len());
    for (doc_id, doc_positions) in postings.docs.into_iter().zip(positions) {
        if !live_docs.is_none_or(|bits| bits.get(doc_id as usize)) {
            continue;
        }
        docs.push(doc_id);
        map.insert(
            doc_id,
            doc_positions.into_iter().map(|p| p.position).collect(),
        );
    }
    Ok(Some((docs, map)))
}

/// Executes `query` (see [`query::PhraseQuery`] for `slop`'s exact semantics)
/// against one already-opened segment, feeding every matching **live** doc ID to
/// `collector` in ascending order -- same
/// parameter contract as [`search_term_query`], plus the segment's opened `.pos`/
/// `.pay` files (needed to check position alignment for a real, multi-term
/// phrase). Note `live_docs` sits *after* `pos_in`/`pay_in` here, unlike
/// [`search_term_query`]/[`search_boolean_query`]'s "`live_docs` right after
/// `doc_in`" ordering -- deliberate, to keep the two positions-file parameters
/// adjacent to each other and to `doc_in`.
///
/// - `pos_in`: the segment's opened `.pos` file. Required (an `Err(Error::
///   MissingPosInput)` otherwise) for any phrase with **more than one term** --
///   never touched for a single-term phrase, which degenerates to a plain
///   [`search_term_query`] call (see below). `None` is fine for that case.
/// - `pay_in`: the segment's opened `.pay` file, or `None` when the field has
///   neither offsets nor payloads, or its total occurrence count never spans a
///   full postings block -- same optionality contract as
///   [`lucene_codecs::blocktree::FieldTerms::positions`].
///
/// **Matching semantics**: a doc matches iff it contains every phrase term (a
/// pure doc-ID conjunction, computed first as a cheap pre-filter -- phrase match
/// implies term match, so this never does position work for a doc that couldn't
/// possibly qualify) *and* an alignment check finds a valid alignment for that
/// doc's per-term position lists: `query.slop == 0` uses
/// [`phrase_matches_in_doc`]'s exact-adjacency fast path (unchanged from before
/// `slop` existed), `query.slop > 0` uses
/// [`phrase_matches_in_doc_sloppy`]'s in-order sloppy check -- see that
/// function's doc comment for the precise formula and its in-order-only scope.
///
/// **Edge cases** (see `query::PhraseQuery`'s doc comment and this port's
/// `docs/parity.md` for the full accounting):
/// - **Empty `terms`**: matches nothing, mirroring real
///   `PhraseQuery.Builder.build()`'s `MatchNoDocsQuery` result for zero added
///   terms. Not an error.
/// - **Single term**: degenerates to [`search_term_query`] on a `TermQuery` for
///   that one term -- a length-1 phrase trivially "aligns" wherever the term
///   occurs, so there's no position work to do (also means a caller running only
///   single-term "phrase" queries never needs an opened `.pos` file at all).
/// - **A term missing from the field**: matches nothing, not an error -- same
///   convention as [`search_term_query`]/[`search_boolean_query`].
/// - **Duplicate terms** (e.g. "the the"): handled correctly by
///   [`phrase_matches_in_doc`] without special-casing -- see that function's doc
///   comment.
pub fn search_phrase_query<C: Collector>(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    pos_in: Option<&PosInput<'_>>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &PhraseQuery,
    collector: &mut C,
) -> Result<()> {
    if query.terms.is_empty() {
        return Ok(());
    }
    if query.terms.len() == 1 {
        let term_query = TermQuery::new(query.field.clone(), query.terms[0].clone());
        return search_term_query(fields, doc_in, live_docs, &term_query, collector);
    }
    let Some(pos_in) = pos_in else {
        return Err(Error::MissingPosInput);
    };

    let mut per_term_docs: Vec<Vec<i32>> = Vec::with_capacity(query.terms.len());
    let mut per_term_maps: Vec<HashMap<i32, Vec<i32>>> = Vec::with_capacity(query.terms.len());
    for term in &query.terms {
        let Some((docs, map)) = term_doc_positions(
            fields,
            doc_in,
            pos_in,
            pay_in,
            live_docs,
            &query.field,
            term,
        )?
        else {
            // A missing term means the phrase can never match -- same convention
            // as `term_doc_ids`/`search_term_query`.
            return Ok(());
        };
        per_term_docs.push(docs);
        per_term_maps.push(map);
    }

    let candidate_docs: Vec<i32> = Conjunction::new(
        per_term_docs
            .into_iter()
            .map(|v| Box::new(v.into_iter()) as BoxDocIter<'static>)
            .collect(),
    )
    .collect();

    for doc_id in candidate_docs {
        let term_positions: Vec<Vec<i32>> = per_term_maps
            .iter()
            .map(|m| {
                m.get(&doc_id)
                    .cloned()
                    .expect("doc_id came from the conjunction of every term's own doc list")
            })
            .collect();
        let is_match = if query.slop == 0 {
            phrase_matches_in_doc(&term_positions)
        } else {
            phrase_matches_in_doc_sloppy(&term_positions, query.slop)
        };
        if is_match {
            collector.collect(doc_id);
        }
    }
    Ok(())
}

/// Scored sibling of [`search_phrase_query`] (task #29): same matching
/// semantics and parameter contract, but feeds each matched, live doc's BM25
/// score to a [`ScoringCollector`] instead of a plain [`Collector`].
///
/// **Formula, verified against real Lucene's `PhraseWeight`/`BM25Similarity`
/// source rather than guessed**: a multi-term phrase's `idf` is the *sum* of
/// each constituent term's own `idf(docFreq, docCount)` — this is
/// `BM25Similarity.idf(CollectionStatistics, TermStatistics[])`'s actual
/// behavior for a phrase's combined term statistics (it iterates every term
/// and sums each one's `idf`, then reports that sum as the phrase's overall
/// idf), not this port's invention. `tfNorm` is computed exactly like
/// [`term_doc_scores`]'s, except with the doc's **phrase frequency** in place
/// of a single term's `freq`:
/// - `query.slop == 0`: phrase frequency is [`phrase_freq_exact`]'s count of
///   valid alignments (`ExactPhraseScorer`'s real `phraseFreq` accumulation —
///   see that function's doc comment for the exact counting rule and why it
///   doesn't double-count).
/// - `query.slop > 0`: phrase frequency is simplified to `1` if
///   [`phrase_matches_in_doc_sloppy`] finds any valid alignment, `0`
///   otherwise — **a deliberate, honestly-scoped simplification**, not a
///   verified port of real Lucene's `SloppyPhraseMatcher` scoring. Real
///   Lucene's sloppy scorer accumulates a graduated per-match contribution of
///   `1.0 / (matchLength + 1)` (favoring tighter alignments) summed across
///   every valid alignment its priority-queue-based algorithm finds — this
///   port could not confidently re-derive/verify that exact per-match
///   weighting formula (or the surrounding alignment-enumeration algorithm,
///   already scoped down to in-order-only by
///   [`phrase_matches_in_doc_sloppy`]'s own doc comment) within this task's
///   scope, so graduated sloppy match-quality scoring is deliberately
///   deferred (see `docs/parity.md`) in favor of this simpler matches-or-not
///   boolean signal, consistent with this port's established "scope down
///   honestly rather than guess at unverified Lucene internals" practice (see
///   BKD's split heuristic, `phrase_matches_in_doc_sloppy` itself).
///
/// `norms`/`collector`: same contract as [`search_term_query_scored`]'s.
#[allow(clippy::too_many_arguments)]
pub fn search_phrase_query_scored<C: ScoringCollector>(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    pos_in: Option<&PosInput<'_>>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &PhraseQuery,
    norms: Option<&FieldNorms<'_>>,
    collector: &mut C,
) -> Result<()> {
    if query.terms.is_empty() {
        return Ok(());
    }
    if query.terms.len() == 1 {
        let term_query = TermQuery::new(query.field.clone(), query.terms[0].clone());
        return search_term_query_scored(fields, doc_in, live_docs, &term_query, norms, collector);
    }
    let Some(pos_in) = pos_in else {
        return Err(Error::MissingPosInput);
    };

    let Some(field_terms) = fields.field(&query.field) else {
        return Ok(());
    };

    // Real BM25's phrase idf is the sum of every constituent term's own idf --
    // see this function's doc comment. A missing term means the phrase can
    // never match, same convention as `search_phrase_query`.
    let doc_count = field_terms.doc_count as i64;
    let mut idf_sum = 0.0f32;
    for term in &query.terms {
        let Some(stats) = field_terms.seek_exact(term) else {
            return Ok(());
        };
        idf_sum += similarity::idf(stats.doc_freq as i64, doc_count);
    }

    let mut per_term_docs: Vec<Vec<i32>> = Vec::with_capacity(query.terms.len());
    let mut per_term_maps: Vec<HashMap<i32, Vec<i32>>> = Vec::with_capacity(query.terms.len());
    for term in &query.terms {
        let Some((docs, map)) = term_doc_positions(
            fields,
            doc_in,
            pos_in,
            pay_in,
            live_docs,
            &query.field,
            term,
        )?
        else {
            return Ok(());
        };
        per_term_docs.push(docs);
        per_term_maps.push(map);
    }

    let candidate_docs: Vec<i32> = Conjunction::new(
        per_term_docs
            .into_iter()
            .map(|v| Box::new(v.into_iter()) as BoxDocIter<'static>)
            .collect(),
    )
    .collect();

    for doc_id in candidate_docs {
        let term_positions: Vec<Vec<i32>> = per_term_maps
            .iter()
            .map(|m| {
                m.get(&doc_id)
                    .cloned()
                    .expect("doc_id came from the conjunction of every term's own doc list")
            })
            .collect();
        let phrase_freq = if query.slop == 0 {
            phrase_freq_exact(&term_positions)
        } else if phrase_matches_in_doc_sloppy(&term_positions, query.slop) {
            1
        } else {
            0
        };
        if phrase_freq == 0 {
            continue;
        }
        let (field_length, avg_field_length) = match norms {
            Some(fn_) => (fn_.field_length(doc_id)?, fn_.avg_field_length),
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
        collector.collect(doc_id, idf_sum * tf_norm);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reuses the same checked-in real-Lucene fixture
    // (`fixtures/data/blocktree_index/`) the differential test in
    // `crates/lucene-search/tests/term_query_fixtures.rs` opens -- that test
    // is the real-Lucene proof; these unit tests instead focus on
    // `search_term_query`'s own branches (missing field, missing term,
    // singleton no-`.doc`-needed path, the `.doc`-required error path,
    // `live_docs` filtering) using the same real segment data, rather than
    // hand-building a synthetic one (see the `test-coverage` skill: a real
    // fixture beats a hand-built one wherever one is already available).
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

    // Owns the `.doc`/`.pos`/`.pay` bytes + segment id/suffix so `DocInput`/
    // `PosInput`/`PayInput` can be constructed with a lifetime tied to a local
    // variable in each test (each of these borrows its buffer).
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
    fn missing_field_yields_no_matches() {
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let mut c = CountCollector::default();
        search_term_query(
            &fields,
            doc_in.as_ref(),
            None,
            &TermQuery::new("nonexistent", "x"),
            &mut c,
        )
        .unwrap();
        assert_eq!(c.count, 0);
    }

    #[test]
    fn missing_term_yields_no_matches() {
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let mut c = VecCollector::default();
        search_term_query(
            &fields,
            doc_in.as_ref(),
            None,
            &TermQuery::new("body", "zzz-missing"),
            &mut c,
        )
        .unwrap();
        assert!(c.docs.is_empty());
    }

    #[test]
    fn multi_doc_term_collects_expected_docs_in_order() {
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let mut c = VecCollector::default();
        search_term_query(
            &fields,
            doc_in.as_ref(),
            None,
            &TermQuery::new("body", "cat"),
            &mut c,
        )
        .unwrap();
        assert_eq!(c.docs, vec![0, 2]);
    }

    #[test]
    fn singleton_term_needs_no_doc_input() {
        let (fields, _doc) = open_fixture();
        let mut c = VecCollector::default();
        search_term_query(&fields, None, None, &TermQuery::new("id", "id2"), &mut c).unwrap();
        assert_eq!(c.docs, vec![2]);
    }

    #[test]
    fn multi_doc_term_without_doc_input_is_an_error() {
        let (fields, _doc) = open_fixture();
        let mut c = CountCollector::default();
        let err = search_term_query(&fields, None, None, &TermQuery::new("body", "cat"), &mut c)
            .unwrap_err();
        assert!(matches!(err, Error::BlockTree(_)));
    }

    #[test]
    fn live_docs_filters_out_deleted_docs() {
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let max_doc: i32 = {
            let dir = concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../fixtures/data/blocktree_index/"
            );
            let manifest = std::fs::read_to_string(format!("{dir}manifest.properties")).unwrap();
            manifest
                .lines()
                .find_map(|l| l.strip_prefix("max_doc="))
                .unwrap()
                .parse()
                .unwrap()
        };
        let mut live_docs = FixedBitSet::new(max_doc as usize);
        for i in 0..max_doc {
            live_docs.set(i as usize);
        }
        // "cat" matches docs 0 and 2 (see manifest); mark doc 0 deleted.
        live_docs.clear(0);

        let mut c = VecCollector::default();
        search_term_query(
            &fields,
            doc_in.as_ref(),
            Some(&live_docs),
            &TermQuery::new("body", "cat"),
            &mut c,
        )
        .unwrap();
        assert_eq!(c.docs, vec![2]);
    }

    // `should_match_counts` unit tests: pure counting logic, no fixture needed.

    #[test]
    fn should_match_counts_tallies_doc_occurrences_across_clauses() {
        let counts = should_match_counts(&[vec![1, 2, 3], vec![2, 3], vec![3]]);
        assert_eq!(counts.get(&1), Some(&1));
        assert_eq!(counts.get(&2), Some(&2));
        assert_eq!(counts.get(&3), Some(&3));
        assert_eq!(counts.get(&4), None);
    }

    #[test]
    fn should_match_counts_no_clauses_is_empty() {
        assert!(should_match_counts(&[]).is_empty());
    }

    #[test]
    fn should_match_counts_disjoint_clauses_each_count_one() {
        let counts = should_match_counts(&[vec![1], vec![2], vec![3]]);
        assert_eq!(counts.get(&1), Some(&1));
        assert_eq!(counts.get(&2), Some(&1));
        assert_eq!(counts.get(&3), Some(&1));
    }

    // Boolean-query tests all reuse `body`'s known real-Lucene doc sets from
    // `manifest.properties` (see `term_query_fixtures.rs`'s module doc for how these
    // were captured): cat={0,2}, dog={0,1}, bird={1,4}.

    #[test]
    fn boolean_must_conjunction_matches_only_docs_in_every_clause() {
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let mut c = VecCollector::default();
        let q = BooleanQuery::new()
            .with_must([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")]);
        search_boolean_query(&fields, doc_in.as_ref(), None, None, None, &q, &mut c).unwrap();
        assert_eq!(c.docs, vec![0]);
    }

    #[test]
    fn boolean_should_disjunction_matches_union_of_clauses() {
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let mut c = VecCollector::default();
        let q = BooleanQuery::new().with_should([
            TermQuery::new("body", "cat"),
            TermQuery::new("body", "bird"),
        ]);
        search_boolean_query(&fields, doc_in.as_ref(), None, None, None, &q, &mut c).unwrap();
        assert_eq!(c.docs, vec![0, 1, 2, 4]);
    }

    #[test]
    fn boolean_must_not_excludes_matching_docs() {
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let mut c = VecCollector::default();
        let q = BooleanQuery::new()
            .with_must([TermQuery::new("body", "cat")])
            .with_must_not([TermQuery::new("body", "dog")]);
        search_boolean_query(&fields, doc_in.as_ref(), None, None, None, &q, &mut c).unwrap();
        assert_eq!(c.docs, vec![2]);
    }

    #[test]
    fn boolean_pure_must_not_matches_nothing() {
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let mut c = VecCollector::default();
        let q = BooleanQuery::new().with_must_not([TermQuery::new("body", "dog")]);
        search_boolean_query(&fields, doc_in.as_ref(), None, None, None, &q, &mut c).unwrap();
        assert!(c.docs.is_empty());
    }

    #[test]
    fn boolean_empty_query_matches_nothing() {
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let mut c = VecCollector::default();
        let q = BooleanQuery::new();
        search_boolean_query(&fields, doc_in.as_ref(), None, None, None, &q, &mut c).unwrap();
        assert!(c.docs.is_empty());
    }

    // `minimum_should_match` tests: cat={0,2}, dog={0,1}, bird={1,4} (see above).

    #[test]
    fn boolean_minimum_should_match_zero_with_must_present_is_unchanged_regression() {
        // Explicit regression test: `minimum_should_match == 0` (the default) must
        // still leave `should` purely score-only once `must` is non-empty, exactly
        // like before task #24 added `minimum_should_match` at all.
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let mut c = VecCollector::default();
        let q = BooleanQuery::new()
            .with_must([TermQuery::new("body", "cat")])
            .with_should([TermQuery::new("body", "bird")])
            .with_minimum_should_match(0);
        search_boolean_query(&fields, doc_in.as_ref(), None, None, None, &q, &mut c).unwrap();
        assert_eq!(c.docs, vec![0, 2]);
    }

    #[test]
    fn boolean_minimum_should_match_one_with_must_present_narrows_the_set() {
        // must=[cat]={0,2}; should=[dog,bird], dog={0,1}, bird={1,4}. With
        // minimum_should_match=1, doc 2 (0 should-clause hits) is now excluded even
        // though it satisfies `must` -- `should` genuinely narrows the set once
        // minimum_should_match > 0, unlike the 0 case above.
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let mut c = VecCollector::default();
        let q = BooleanQuery::new()
            .with_must([TermQuery::new("body", "cat")])
            .with_should([
                TermQuery::new("body", "dog"),
                TermQuery::new("body", "bird"),
            ])
            .with_minimum_should_match(1);
        search_boolean_query(&fields, doc_in.as_ref(), None, None, None, &q, &mut c).unwrap();
        assert_eq!(c.docs, vec![0]);
    }

    #[test]
    fn boolean_minimum_should_match_two_with_three_should_clauses_excludes_single_hits() {
        // should=[cat,dog,bird] (must empty): doc0 hits cat+dog (2), doc1 hits
        // dog+bird (2), doc2 hits only cat (1), doc4 hits only bird (1). With
        // minimum_should_match=2, only docs with 2+ hits survive.
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let mut c = VecCollector::default();
        let q = BooleanQuery::new()
            .with_should([
                TermQuery::new("body", "cat"),
                TermQuery::new("body", "dog"),
                TermQuery::new("body", "bird"),
            ])
            .with_minimum_should_match(2);
        search_boolean_query(&fields, doc_in.as_ref(), None, None, None, &q, &mut c).unwrap();
        assert_eq!(c.docs, vec![0, 1]);
    }

    #[test]
    fn boolean_minimum_should_match_with_must_empty_still_requires_the_threshold() {
        // Same should set as above but explicitly with must empty and
        // minimum_should_match=1 -- equivalent to a plain disjunction (every should
        // clause hit counts as >= 1), confirming must-empty + minSSM=1 matches the
        // existing should-disjunction-is-the-matched-set behavior.
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let mut c = VecCollector::default();
        let q = BooleanQuery::new()
            .with_should([
                TermQuery::new("body", "cat"),
                TermQuery::new("body", "bird"),
            ])
            .with_minimum_should_match(1);
        search_boolean_query(&fields, doc_in.as_ref(), None, None, None, &q, &mut c).unwrap();
        assert_eq!(c.docs, vec![0, 1, 2, 4]);
    }

    #[test]
    fn boolean_minimum_should_match_exceeding_clause_count_matches_nothing() {
        // Only 2 should clauses exist; minimum_should_match=5 can never be reached
        // by any doc -- mirrors real `BooleanQuery.rewrite()`'s `MatchNoDocsQuery`
        // for "shoulds.size() < minimumNumberShouldMatch", achieved here without a
        // separate branch (see `matched_boolean_docs`'s doc comment).
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let mut c = VecCollector::default();
        let q = BooleanQuery::new()
            .with_should([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")])
            .with_minimum_should_match(5);
        search_boolean_query(&fields, doc_in.as_ref(), None, None, None, &q, &mut c).unwrap();
        assert!(c.docs.is_empty());
    }

    #[test]
    fn boolean_minimum_should_match_combines_with_must_not() {
        // must=[cat]={0,2}; should=[dog,bird] with minimum_should_match=1 keeps only
        // doc 0 (see the "narrows the set" test above); must_not=[dog]={0,1}
        // additionally excludes doc 0, leaving nothing.
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let mut c = VecCollector::default();
        let q = BooleanQuery::new()
            .with_must([TermQuery::new("body", "cat")])
            .with_should([
                TermQuery::new("body", "dog"),
                TermQuery::new("body", "bird"),
            ])
            .with_must_not([TermQuery::new("body", "dog")])
            .with_minimum_should_match(1);
        search_boolean_query(&fields, doc_in.as_ref(), None, None, None, &q, &mut c).unwrap();
        assert!(c.docs.is_empty());
    }

    // Nested `BooleanQuery` clause tests (task #25): cat={0,2}, dog={0,1},
    // bird={1,4} (see above).

    #[test]
    fn nested_boolean_must_clause_narrows_the_matched_set() {
        // top.must = [dog, nested] where nested = should=[cat, bird]. dog={0,1};
        // nested's own disjunction = cat ∪ bird = {0,1,2,4}. Conjunction: {0,1}.
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let mut c = VecCollector::default();
        let nested = BooleanQuery::new().with_should([
            TermQuery::new("body", "cat"),
            TermQuery::new("body", "bird"),
        ]);
        let q = BooleanQuery::new().with_must([
            Clause::Term(TermQuery::new("body", "dog")),
            Clause::Boolean(Box::new(nested)),
        ]);
        search_boolean_query(&fields, doc_in.as_ref(), None, None, None, &q, &mut c).unwrap();
        assert_eq!(c.docs, vec![0, 1]);
    }

    #[test]
    fn nested_boolean_should_clause_contributes_to_the_disjunction() {
        // top.should = [nested] where nested = must=[cat, dog] -- nested's own
        // conjunction is {0}, so top's disjunction (its only should clause) is {0}.
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let mut c = VecCollector::default();
        let nested = BooleanQuery::new()
            .with_must([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")]);
        let q = BooleanQuery::new().with_should([nested]);
        search_boolean_query(&fields, doc_in.as_ref(), None, None, None, &q, &mut c).unwrap();
        assert_eq!(c.docs, vec![0]);
    }

    #[test]
    fn nested_boolean_clauses_own_minimum_should_match_does_not_leak_to_parent() {
        // nested = should=[dog, bird], minimum_should_match=2 (its own threshold):
        // dog={0,1}, bird={1,4}, so nested's own matched set is {1} (only doc 1 hits
        // both). Top level has no minimum_should_match of its own (defaults to 0),
        // and top.must = [nested] alone -- the parent's conjunction is exactly
        // nested's matched set, proving nested's threshold is evaluated
        // independently and does narrow nested's own contribution.
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let mut c = VecCollector::default();
        let nested = BooleanQuery::new()
            .with_should([
                TermQuery::new("body", "dog"),
                TermQuery::new("body", "bird"),
            ])
            .with_minimum_should_match(2);
        let q = BooleanQuery::new().with_must([nested]);
        search_boolean_query(&fields, doc_in.as_ref(), None, None, None, &q, &mut c).unwrap();
        assert_eq!(c.docs, vec![1]);
    }

    #[test]
    fn parent_minimum_should_match_does_not_affect_nested_querys_own_matching() {
        // Same nested query as above (should=[dog,bird], min_should_match=2 => {1}
        // is nested's own matched set), but now the *parent* also sets its own
        // minimum_should_match=1 over should=[nested, cat]. must is empty, so the
        // matched set is should's disjunction: nested's {1} ∪ cat's {0,2} = {0,1,2},
        // gated by parent's own min_should_match=1 (trivially satisfied by any
        // should hit) -- confirms the parent's own threshold is a fully separate
        // setting from the nested query's, neither overriding the other.
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let mut c = VecCollector::default();
        let nested = BooleanQuery::new()
            .with_should([
                TermQuery::new("body", "dog"),
                TermQuery::new("body", "bird"),
            ])
            .with_minimum_should_match(2);
        let q = BooleanQuery::new()
            .with_should([
                Clause::Boolean(Box::new(nested)),
                Clause::Term(TermQuery::new("body", "cat")),
            ])
            .with_minimum_should_match(1);
        search_boolean_query(&fields, doc_in.as_ref(), None, None, None, &q, &mut c).unwrap();
        assert_eq!(c.docs, vec![0, 1, 2]);
    }

    #[test]
    fn nested_boolean_clause_that_matches_nothing_contributes_an_empty_set() {
        // nested has only a must_not clause -- real `BooleanQuery.rewrite()`'s pure
        // negative case, matching nothing on its own. As a `should` clause of the
        // parent, it must contribute no docs, leaving only the sibling should
        // clause's matches.
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let mut c = VecCollector::default();
        let nested = BooleanQuery::new().with_must_not([TermQuery::new("body", "dog")]);
        let q = BooleanQuery::new().with_should([
            Clause::Boolean(Box::new(nested)),
            Clause::Term(TermQuery::new("body", "cat")),
        ]);
        search_boolean_query(&fields, doc_in.as_ref(), None, None, None, &q, &mut c).unwrap();
        assert_eq!(c.docs, vec![0, 2]);
    }

    #[test]
    fn three_levels_of_nested_boolean_clauses_resolve_correctly() {
        // Genuine multi-level recursion, not just one extra level: innermost =
        // must=[cat, dog] => {0}. middle.should = [innermost] => {0} (its only
        // should clause). top.must = [middle] => {0}.
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let mut c = VecCollector::default();
        let innermost = BooleanQuery::new()
            .with_must([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")]);
        let middle = BooleanQuery::new().with_should([innermost]);
        let top = BooleanQuery::new().with_must([middle]);
        search_boolean_query(&fields, doc_in.as_ref(), None, None, None, &top, &mut c).unwrap();
        assert_eq!(c.docs, vec![0]);
    }

    #[test]
    fn nested_boolean_clause_scoring_sums_its_own_matching_sub_clauses() {
        // top.should = [nested] alone, nested.should = [cat, bird]. Nested's own
        // matched set is cat ∪ bird = {0,1,2,4}; each matched doc's score must equal
        // the sum of whichever of cat/bird it actually matches -- same recursive
        // rule `boolean_query_scored_matches_unscored_doc_set_and_sums_clause_scores`
        // in `scoring_fixtures.rs` proves at the top level, now one level deeper.
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());

        let nested = BooleanQuery::new().with_should([
            TermQuery::new("body", "cat"),
            TermQuery::new("body", "bird"),
        ]);
        let top = BooleanQuery::new().with_should([nested]);

        let mut top_docs = TopDocsCollector::new(10);
        search_boolean_query_scored(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            &top,
            None,
            &mut top_docs,
        )
        .unwrap();

        let mut cat_scores = TopDocsCollector::new(10);
        search_term_query_scored(
            &fields,
            doc_in.as_ref(),
            None,
            &TermQuery::new("body", "cat"),
            None,
            &mut cat_scores,
        )
        .unwrap();
        let mut bird_scores = TopDocsCollector::new(10);
        search_term_query_scored(
            &fields,
            doc_in.as_ref(),
            None,
            &TermQuery::new("body", "bird"),
            None,
            &mut bird_scores,
        )
        .unwrap();

        let lookup = |top: &TopDocsCollector, doc_id: i32| -> Option<f32> {
            top.top_docs()
                .iter()
                .find(|h| h.doc_id == doc_id)
                .map(|h| h.score)
        };

        let hits = top_docs.top_docs();
        let mut hit_docs: Vec<i32> = hits.iter().map(|h| h.doc_id).collect();
        hit_docs.sort_unstable();
        assert_eq!(hit_docs, vec![0, 1, 2, 4]);

        for hit in hits {
            let expected = lookup(&cat_scores, hit.doc_id).unwrap_or(0.0)
                + lookup(&bird_scores, hit.doc_id).unwrap_or(0.0);
            assert!(
                (hit.score - expected).abs() < 1e-4,
                "doc={} got={} expected={}",
                hit.doc_id,
                hit.score,
                expected
            );
        }
    }

    #[test]
    fn nested_boolean_clause_scoring_excludes_docs_the_nested_query_itself_rejects() {
        // nested = should=[dog, bird], minimum_should_match=2 -- nested's own
        // matched set is {1} alone (see the matching-side test above). As a
        // `should` clause of top (must empty), top's matched set must be exactly
        // {1}, and its score must be dog(1) + bird(1) (both of nested's own
        // sub-clauses that doc 1 actually satisfies) -- not a score for doc 0 or 4,
        // which nested's own threshold rejects even though dog/bird individually
        // match them.
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());

        let nested = BooleanQuery::new()
            .with_should([
                TermQuery::new("body", "dog"),
                TermQuery::new("body", "bird"),
            ])
            .with_minimum_should_match(2);
        let top = BooleanQuery::new().with_should([nested]);

        let mut top_docs = TopDocsCollector::new(10);
        search_boolean_query_scored(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            &top,
            None,
            &mut top_docs,
        )
        .unwrap();
        let hits = top_docs.top_docs();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, 1);

        let mut dog_scores = TopDocsCollector::new(10);
        search_term_query_scored(
            &fields,
            doc_in.as_ref(),
            None,
            &TermQuery::new("body", "dog"),
            None,
            &mut dog_scores,
        )
        .unwrap();
        let mut bird_scores = TopDocsCollector::new(10);
        search_term_query_scored(
            &fields,
            doc_in.as_ref(),
            None,
            &TermQuery::new("body", "bird"),
            None,
            &mut bird_scores,
        )
        .unwrap();
        let lookup = |top: &TopDocsCollector, doc_id: i32| -> Option<f32> {
            top.top_docs()
                .iter()
                .find(|h| h.doc_id == doc_id)
                .map(|h| h.score)
        };
        let expected =
            lookup(&dog_scores, 1).expect("dog matches doc 1") + lookup(&bird_scores, 1).unwrap();
        assert!((hits[0].score - expected).abs() < 1e-4);
    }

    #[test]
    fn boolean_must_with_missing_term_matches_nothing() {
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let mut c = VecCollector::default();
        let q = BooleanQuery::new().with_must([
            TermQuery::new("body", "cat"),
            TermQuery::new("body", "zzz-missing"),
        ]);
        search_boolean_query(&fields, doc_in.as_ref(), None, None, None, &q, &mut c).unwrap();
        assert!(c.docs.is_empty());
    }

    #[test]
    fn boolean_live_docs_filters_before_conjunction() {
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let max_doc: i32 = {
            let dir = concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../fixtures/data/blocktree_index/"
            );
            let manifest = std::fs::read_to_string(format!("{dir}manifest.properties")).unwrap();
            manifest
                .lines()
                .find_map(|l| l.strip_prefix("max_doc="))
                .unwrap()
                .parse()
                .unwrap()
        };
        let mut live_docs = FixedBitSet::new(max_doc as usize);
        for i in 0..max_doc {
            live_docs.set(i as usize);
        }
        // cat={0,2}, dog={0,1}; conjunction is {0}. Marking doc 0 dead removes the
        // only shared doc, so the conjunction (computed post-filter) is empty.
        live_docs.clear(0);

        let mut c = VecCollector::default();
        let q = BooleanQuery::new()
            .with_must([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")]);
        search_boolean_query(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            Some(&live_docs),
            &q,
            &mut c,
        )
        .unwrap();
        assert!(c.docs.is_empty());
    }

    #[test]
    fn count_collector_matches_vec_collector_length() {
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let mut count = CountCollector::default();
        let mut docs = VecCollector::default();
        search_term_query(
            &fields,
            doc_in.as_ref(),
            None,
            &TermQuery::new("body", "bird"),
            &mut count,
        )
        .unwrap();
        search_term_query(
            &fields,
            doc_in.as_ref(),
            None,
            &TermQuery::new("body", "bird"),
            &mut docs,
        )
        .unwrap();
        assert_eq!(count.count as usize, docs.docs.len());
    }

    // `phrase_matches_in_doc` unit tests: synthetic per-term position lists, no
    // fixture needed -- this is the pure alignment-checking function in isolation.

    #[test]
    fn phrase_matches_exact_alignment_at_position_zero() {
        assert!(phrase_matches_in_doc(&[vec![0], vec![1], vec![2]]));
    }

    #[test]
    fn phrase_matches_exact_alignment_at_a_later_position() {
        assert!(phrase_matches_in_doc(&[vec![0, 5], vec![1, 6], vec![2, 7]]));
    }

    #[test]
    fn phrase_no_match_despite_every_term_present() {
        // "cat" at 0 and 10, "sat" at 1, "mat" at 5: no base position aligns all three
        // (0 -> needs 1 at "sat" (ok) and 2 at "mat" (missing); 10 -> needs 11 (missing)).
        assert!(!phrase_matches_in_doc(&[vec![0, 10], vec![1], vec![5]]));
    }

    #[test]
    fn phrase_multiple_candidates_only_one_aligns() {
        // Base 0 fails (needs 2 at term index 2, only 5/7 present); base 3 succeeds
        // (needs 4 at term index 1 -- present -- and 5 at term index 2 -- present).
        assert!(phrase_matches_in_doc(&[vec![0, 3], vec![1, 4], vec![5, 7]]));
    }

    #[test]
    fn phrase_single_term_degenerates_to_any_occurrence() {
        assert!(phrase_matches_in_doc(&[vec![2, 9]]));
    }

    #[test]
    fn phrase_single_term_with_no_occurrences_is_false() {
        assert!(!phrase_matches_in_doc(&[vec![]]));
    }

    #[test]
    fn phrase_no_terms_at_all_is_false() {
        assert!(!phrase_matches_in_doc(&[]));
    }

    #[test]
    fn phrase_a_term_with_no_occurrences_in_this_doc_is_false() {
        assert!(!phrase_matches_in_doc(&[vec![0], vec![]]));
    }

    #[test]
    fn phrase_repeated_term_with_consecutive_occurrences_matches() {
        // "the the": both occurrence lists are the term "the"'s own positions --
        // 0 and 1 are consecutive, so "the" at 0 followed by "the" at 1 is a match.
        assert!(phrase_matches_in_doc(&[vec![0, 1, 2], vec![0, 1, 2]]));
    }

    #[test]
    fn phrase_repeated_term_without_consecutive_occurrences_does_not_match() {
        // "the" only occurs at 0, 2, 4 -- no two consecutive occurrences exist.
        assert!(!phrase_matches_in_doc(&[vec![0, 2, 4], vec![0, 2, 4]]));
    }

    // `phrase_matches_in_doc_sloppy` unit tests: hand-computed slop values against
    // the formula documented on that function -- `(p_last - p_first) - (n - 1)`.

    #[test]
    fn sloppy_exact_alignment_needs_zero_slop() {
        // positions 0,1,2: (2-0)-2 = 0 moves needed -- matches at slop=0.
        assert!(phrase_matches_in_doc_sloppy(
            &[vec![0], vec![1], vec![2]],
            0
        ));
    }

    #[test]
    fn sloppy_agrees_with_exact_for_slop_zero_no_match_case() {
        // "cat" at 0, "sat" at 2: (2-0)-1 = 1 move needed, slop=0 is one short.
        assert!(!phrase_matches_in_doc_sloppy(&[vec![0], vec![2]], 0));
        assert!(!phrase_matches_in_doc(&[vec![0], vec![2]]));
    }

    #[test]
    fn sloppy_gap_of_one_extra_word_needs_slop_one() {
        // "quick" at 0, "fox" at 2 (one word -- "brown" -- skipped in between):
        // (2-0)-1 = 1 move needed.
        assert!(!phrase_matches_in_doc_sloppy(&[vec![0], vec![2]], 0));
        assert!(phrase_matches_in_doc_sloppy(&[vec![0], vec![2]], 1));
    }

    #[test]
    fn sloppy_boundary_exactly_enough_slop_matches() {
        // "a" at 0, "b" at 4: (4-0)-1 = 3 moves needed. slop=3 matches, slop=2 (one
        // less than enough) does not.
        assert!(phrase_matches_in_doc_sloppy(&[vec![0], vec![4]], 3));
        assert!(!phrase_matches_in_doc_sloppy(&[vec![0], vec![4]], 2));
    }

    #[test]
    fn sloppy_three_term_gap_sums_across_both_intervals() {
        // "the" at 0, "quick" at 2 (gap 1), "fox" at 5 (gap 2): total moves =
        // (5-0)-2 = 3, matching the sum of per-interval gaps (1 + 2).
        assert!(phrase_matches_in_doc_sloppy(
            &[vec![0], vec![2], vec![5]],
            3
        ));
        assert!(!phrase_matches_in_doc_sloppy(
            &[vec![0], vec![2], vec![5]],
            2
        ));
    }

    #[test]
    fn sloppy_picks_the_best_of_multiple_candidate_base_positions() {
        // First term at {0, 10}; second term at {1, 11}. Base 0 -> 1 needs 0 moves;
        // base 10 -> 11 also needs 0 moves -- either way it should match at slop=0,
        // proving every base candidate is tried (not just the first).
        assert!(phrase_matches_in_doc_sloppy(&[vec![0, 10], vec![1, 11]], 0));
    }

    #[test]
    fn sloppy_greedy_finds_smallest_valid_next_position() {
        // First term at 0; second term's list has {1, 2, 100} -- greedy must pick 1
        // (smallest valid), needing 0 moves, not be confused by the far-away 100.
        assert!(phrase_matches_in_doc_sloppy(&[vec![0], vec![1, 2, 100]], 0));
    }

    #[test]
    fn sloppy_no_increasing_alignment_exists_still_fails_at_high_slop() {
        // Second term's only occurrence (0) is not strictly after the first term's
        // only occurrence (0) -- no in-order alignment exists at any slop, since
        // this port's scope excludes reordering/ties.
        assert!(!phrase_matches_in_doc_sloppy(&[vec![0], vec![0]], 100));
    }

    #[test]
    fn sloppy_single_term_degenerates_to_any_occurrence_regardless_of_slop() {
        assert!(phrase_matches_in_doc_sloppy(&[vec![2, 9]], 0));
        assert!(phrase_matches_in_doc_sloppy(&[vec![2, 9]], 5));
    }

    #[test]
    fn sloppy_single_term_with_no_occurrences_is_false() {
        assert!(!phrase_matches_in_doc_sloppy(&[vec![]], 5));
    }

    #[test]
    fn sloppy_no_terms_at_all_is_false() {
        assert!(!phrase_matches_in_doc_sloppy(&[], 5));
    }

    #[test]
    fn sloppy_a_term_with_no_occurrences_in_this_doc_is_false() {
        assert!(!phrase_matches_in_doc_sloppy(&[vec![0], vec![]], 5));
    }

    #[test]
    fn sloppy_repeated_term_with_a_gap_matches_at_sufficient_slop() {
        // "the" at 0, 3 -- as a two-term "the the" phrase, base 0 needs the second
        // "the" strictly after 0: smallest is 3, (3-0)-1 = 2 moves.
        assert!(phrase_matches_in_doc_sloppy(&[vec![0, 3], vec![0, 3]], 2));
        assert!(!phrase_matches_in_doc_sloppy(&[vec![0, 3], vec![0, 3]], 1));
    }

    // Fixture-backed `search_phrase_query` tests: reuse the real-Lucene "pos" field
    // (`IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS`) already checked into
    // `fixtures/data/blocktree_index/` for `crates/lucene-codecs/tests/
    // blocktree_fixtures.rs`'s `pos_field_positions_match_real_lucene_postings_enum`
    // test -- per the manifest, doc 8555 has "alpha" at position 0 and "beta" at
    // position 1 (adjacent), while doc 8556 has "alpha" at positions 0 and 1 but no
    // "beta" at all. That's exactly the shape a real "alpha beta" phrase query
    // differential test needs, already present without extending the fixture
    // generator (see this module's `Testing` section in the task write-up: prefer
    // reusing existing fixtures over adding new ones when the data already fits).

    #[test]
    fn phrase_query_two_terms_matches_only_the_adjacent_doc() {
        let (fields, doc) = open_fixture();
        let doc = doc.unwrap();
        let doc_in = doc.open();
        let pos_in = doc.open_pos();
        let pay_in = doc.open_pay();
        let mut c = VecCollector::default();
        search_phrase_query(
            &fields,
            Some(&doc_in),
            Some(&pos_in),
            Some(&pay_in),
            None,
            &PhraseQuery::new("pos", ["alpha", "beta"]),
            &mut c,
        )
        .unwrap();
        assert_eq!(c.docs, vec![8555]);
    }

    #[test]
    fn phrase_query_single_term_degenerates_to_term_query() {
        let (fields, doc) = open_fixture();
        let doc = doc.unwrap();
        let doc_in = doc.open();
        let mut c = VecCollector::default();
        // No .pos/.pay opened at all -- the single-term case must not need them.
        search_phrase_query(
            &fields,
            Some(&doc_in),
            None,
            None,
            None,
            &PhraseQuery::new("pos", ["alpha"]),
            &mut c,
        )
        .unwrap();
        assert_eq!(c.docs, vec![8555, 8556, 8557]);
    }

    #[test]
    fn phrase_query_empty_terms_matches_nothing() {
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let mut c = VecCollector::default();
        search_phrase_query(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            &PhraseQuery::new("pos", Vec::<&str>::new()),
            &mut c,
        )
        .unwrap();
        assert!(c.docs.is_empty());
    }

    #[test]
    fn phrase_query_missing_field_matches_nothing() {
        let (fields, doc) = open_fixture();
        let doc = doc.unwrap();
        let doc_in = doc.open();
        let pos_in = doc.open_pos();
        let mut c = VecCollector::default();
        search_phrase_query(
            &fields,
            Some(&doc_in),
            Some(&pos_in),
            None,
            None,
            &PhraseQuery::new("nonexistent", ["a", "b"]),
            &mut c,
        )
        .unwrap();
        assert!(c.docs.is_empty());
    }

    #[test]
    fn phrase_query_missing_term_matches_nothing() {
        let (fields, doc) = open_fixture();
        let doc = doc.unwrap();
        let doc_in = doc.open();
        let pos_in = doc.open_pos();
        let pay_in = doc.open_pay();
        let mut c = VecCollector::default();
        search_phrase_query(
            &fields,
            Some(&doc_in),
            Some(&pos_in),
            Some(&pay_in),
            None,
            &PhraseQuery::new("pos", ["alpha", "zzz-missing"]),
            &mut c,
        )
        .unwrap();
        assert!(c.docs.is_empty());
    }

    #[test]
    fn phrase_query_duplicate_term_matches_consecutive_occurrences() {
        let (fields, doc) = open_fixture();
        let doc = doc.unwrap();
        let doc_in = doc.open();
        let pos_in = doc.open_pos();
        let pay_in = doc.open_pay();
        let mut c = VecCollector::default();
        // doc 8555 has "alpha" only at position 0 (no consecutive pair); doc 8556
        // has "alpha" at 0 and 1, a real consecutive-repeated-term match.
        search_phrase_query(
            &fields,
            Some(&doc_in),
            Some(&pos_in),
            Some(&pay_in),
            None,
            &PhraseQuery::new("pos", ["alpha", "alpha"]),
            &mut c,
        )
        .unwrap();
        assert_eq!(c.docs, vec![8556]);
    }

    #[test]
    fn phrase_query_sloppy_wiring_still_matches_the_exact_adjacent_doc() {
        // This module's own `open_fixture()` data (alpha/beta) is exact-adjacent
        // (gap 0). The non-adjacent-by-a-known-gap cross-engine case (doc7,
        // alpha@0/beta@3) now lives in
        // `crates/lucene-search/tests/phrase_query_fixtures.rs`'s
        // `sloppy_phrase_gap_matches_real_lucenes_phrase_query_set_slop_at_every_tested_value`,
        // verified against real Lucene's actual `PhraseQuery.setSlop(n)` results
        // recorded by `GenBlockTree.java` -- see `docs/parity.md`. This test
        // instead proves `search_phrase_query` itself correctly routes `slop > 0`
        // to the sloppy path end-to-end (not just `phrase_matches_in_doc_sloppy`
        // in isolation, which the unit tests above
        // already cover exhaustively): a generous slop must still find exactly the
        // same match as `slop == 0` for data that's already exact-adjacent.
        let (fields, doc) = open_fixture();
        let doc = doc.unwrap();
        let doc_in = doc.open();
        let pos_in = doc.open_pos();
        let pay_in = doc.open_pay();
        let mut c = VecCollector::default();
        search_phrase_query(
            &fields,
            Some(&doc_in),
            Some(&pos_in),
            Some(&pay_in),
            None,
            &PhraseQuery::new("pos", ["alpha", "beta"]).with_slop(5),
            &mut c,
        )
        .unwrap();
        // slop=5 also bridges doc7's gap (alpha@0, beta@3, needs 2 moves).
        assert_eq!(c.docs, vec![8555, 8557]);
    }

    #[test]
    fn phrase_query_multi_term_without_pos_input_is_an_error() {
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let mut c = VecCollector::default();
        let err = search_phrase_query(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            &PhraseQuery::new("pos", ["alpha", "beta"]),
            &mut c,
        )
        .unwrap_err();
        assert!(matches!(err, Error::MissingPosInput));
    }

    #[test]
    fn phrase_query_live_docs_filters_before_alignment_check() {
        let (fields, doc) = open_fixture();
        let doc = doc.unwrap();
        let doc_in = doc.open();
        let pos_in = doc.open_pos();
        let pay_in = doc.open_pay();
        let max_doc: i32 = {
            let dir = concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../fixtures/data/blocktree_index/"
            );
            let manifest = std::fs::read_to_string(format!("{dir}manifest.properties")).unwrap();
            manifest
                .lines()
                .find_map(|l| l.strip_prefix("max_doc="))
                .unwrap()
                .parse()
                .unwrap()
        };
        let mut live_docs = FixedBitSet::new(max_doc as usize);
        for i in 0..max_doc {
            live_docs.set(i as usize);
        }
        // "alpha beta" only ever matches doc 8555 -- marking it dead removes the
        // only match.
        live_docs.clear(8555);

        let mut c = VecCollector::default();
        search_phrase_query(
            &fields,
            Some(&doc_in),
            Some(&pos_in),
            Some(&pay_in),
            Some(&live_docs),
            &PhraseQuery::new("pos", ["alpha", "beta"]),
            &mut c,
        )
        .unwrap();
        assert!(c.docs.is_empty());
    }

    // `phrase_freq_exact` unit tests (task #29): pure counting logic against
    // hand-built position lists, no fixture needed.

    #[test]
    fn phrase_freq_exact_counts_one_match_when_phrase_occurs_once() {
        // "quick fox" at position 0/1 only.
        assert_eq!(phrase_freq_exact(&[vec![0], vec![1]]), 1);
    }

    #[test]
    fn phrase_freq_exact_counts_every_repeated_occurrence() {
        // "the the": "the" at 0,1,2,3 -- valid starts at 0,1,2 (0+1=1 present,
        // 1+1=2 present, 2+1=3 present), 3 has no successor -- 3 matches, not 1.
        let positions = vec![vec![0, 1, 2, 3], vec![0, 1, 2, 3]];
        assert_eq!(phrase_freq_exact(&positions), 3);
    }

    #[test]
    fn phrase_freq_exact_zero_when_no_alignment_exists() {
        assert_eq!(phrase_freq_exact(&[vec![0], vec![5]]), 0);
    }

    #[test]
    fn phrase_freq_exact_zero_for_empty_term_positions() {
        assert_eq!(phrase_freq_exact(&[]), 0);
    }

    #[test]
    fn phrase_freq_exact_zero_when_any_term_has_no_occurrences() {
        assert_eq!(phrase_freq_exact(&[vec![0], vec![]]), 0);
    }

    #[test]
    fn phrase_freq_exact_single_term_counts_every_occurrence() {
        assert_eq!(phrase_freq_exact(&[vec![0, 3, 7]]), 3);
    }

    #[test]
    fn phrase_freq_exact_non_overlapping_repeats_counts_two() {
        // "quick fox ... quick fox": two disjoint adjacent pairs, no overlap
        // possible between them.
        let positions = vec![vec![0, 10], vec![1, 11]];
        assert_eq!(phrase_freq_exact(&positions), 2);
    }

    // `span_matches_in_doc` unit tests (task #55): synthetic per-leaf position
    // maps, no fixture needed -- this is the pure span-computation function in
    // isolation, mirroring `phrase_matches_in_doc`'s own test style above.

    fn leaf_positions(pairs: &[(&str, &[i32])]) -> HashMap<SpanLeafKey, Vec<i32>> {
        pairs
            .iter()
            .map(|&(term, positions)| {
                (
                    ("f".to_string(), term.as_bytes().to_vec()),
                    positions.to_vec(),
                )
            })
            .collect()
    }

    #[test]
    fn span_term_matches_every_occurrence_in_a_multi_occurrence_doc() {
        let positions = leaf_positions(&[("cat", &[0, 3, 7])]);
        let query = SpanQuery::span_term("f", "cat");
        assert_eq!(
            span_matches_in_doc(&query, &positions),
            vec![(0, 1), (3, 4), (7, 8)]
        );
    }

    #[test]
    fn span_term_no_occurrences_yields_no_spans() {
        let positions = leaf_positions(&[]);
        let query = SpanQuery::span_term("f", "cat");
        assert!(span_matches_in_doc(&query, &positions).is_empty());
    }

    #[test]
    fn span_near_in_order_matches_an_ordered_adjacent_pair() {
        // "cat" at 0, "sat" at 1 -- adjacent, in phrase order.
        let positions = leaf_positions(&[("cat", &[0]), ("sat", &[1])]);
        let query = SpanQuery::span_near(
            [
                SpanQuery::span_term("f", "cat"),
                SpanQuery::span_term("f", "sat"),
            ],
            0,
            true,
        );
        assert_eq!(span_matches_in_doc(&query, &positions), vec![(0, 2)]);
    }

    #[test]
    fn span_near_in_order_does_not_match_a_reversed_pair() {
        // "sat" at 0, "cat" at 1 -- clauses are [cat, sat], but the doc has
        // "sat" occur first: in-order requires clause 0's span before clause 1's.
        let positions = leaf_positions(&[("cat", &[1]), ("sat", &[0])]);
        let query = SpanQuery::span_near(
            [
                SpanQuery::span_term("f", "cat"),
                SpanQuery::span_term("f", "sat"),
            ],
            0,
            true,
        );
        assert!(span_matches_in_doc(&query, &positions).is_empty());
    }

    #[test]
    fn span_near_out_of_order_matches_a_reversed_pair_within_slop() {
        // Same reversed doc as above, but `in_order == false`: any relative
        // order is accepted, so this DOES match -- the key differentiator from
        // `PhraseQuery`'s in-order-only sloppy matching.
        let positions = leaf_positions(&[("cat", &[1]), ("sat", &[0])]);
        let query = SpanQuery::span_near(
            [
                SpanQuery::span_term("f", "cat"),
                SpanQuery::span_term("f", "sat"),
            ],
            0,
            false,
        );
        assert_eq!(span_matches_in_doc(&query, &positions), vec![(0, 2)]);
    }

    #[test]
    fn span_near_respects_slop_boundary_exactly_at_limit_matches() {
        // "cat" at 0, "sat" at 2 -- one word gap, slack = (2 - 1) = 1.
        let positions = leaf_positions(&[("cat", &[0]), ("sat", &[2])]);
        let query = SpanQuery::span_near(
            [
                SpanQuery::span_term("f", "cat"),
                SpanQuery::span_term("f", "sat"),
            ],
            1,
            true,
        );
        assert_eq!(span_matches_in_doc(&query, &positions), vec![(0, 3)]);
    }

    #[test]
    fn span_near_respects_slop_boundary_one_over_does_not_match() {
        let positions = leaf_positions(&[("cat", &[0]), ("sat", &[2])]);
        let query = SpanQuery::span_near(
            [
                SpanQuery::span_term("f", "cat"),
                SpanQuery::span_term("f", "sat"),
            ],
            0,
            true,
        );
        assert!(span_matches_in_doc(&query, &positions).is_empty());
    }

    #[test]
    fn span_or_matches_if_either_sub_span_matches() {
        let cat_only = leaf_positions(&[("cat", &[0])]);
        let dog_only = leaf_positions(&[("dog", &[0])]);
        let neither = leaf_positions(&[]);
        let both = leaf_positions(&[("cat", &[0]), ("dog", &[5])]);
        let query = SpanQuery::span_or([
            SpanQuery::span_term("f", "cat"),
            SpanQuery::span_term("f", "dog"),
        ]);
        assert_eq!(span_matches_in_doc(&query, &cat_only), vec![(0, 1)]);
        assert_eq!(span_matches_in_doc(&query, &dog_only), vec![(0, 1)]);
        assert!(span_matches_in_doc(&query, &neither).is_empty());
        assert_eq!(span_matches_in_doc(&query, &both), vec![(0, 1), (5, 6)]);
    }

    #[test]
    fn span_near_of_span_near_composes_correctly() {
        // (cat NEAR/0,in-order sat) NEAR/0,in-order mat: "cat" 0, "sat" 1, "mat" 2.
        let positions = leaf_positions(&[("cat", &[0]), ("sat", &[1]), ("mat", &[2])]);
        let inner = SpanQuery::span_near(
            [
                SpanQuery::span_term("f", "cat"),
                SpanQuery::span_term("f", "sat"),
            ],
            0,
            true,
        );
        let outer = SpanQuery::span_near([inner, SpanQuery::span_term("f", "mat")], 0, true);
        assert_eq!(span_matches_in_doc(&outer, &positions), vec![(0, 3)]);
    }

    #[test]
    fn span_near_of_span_near_no_match_when_inner_does_not_align() {
        // Inner "cat sat" fails to align (gap too big for slop=0), so the outer
        // near can never find an inner span to combine with "mat".
        let positions = leaf_positions(&[("cat", &[0]), ("sat", &[5]), ("mat", &[6])]);
        let inner = SpanQuery::span_near(
            [
                SpanQuery::span_term("f", "cat"),
                SpanQuery::span_term("f", "sat"),
            ],
            0,
            true,
        );
        let outer = SpanQuery::span_near([inner, SpanQuery::span_term("f", "mat")], 0, true);
        assert!(span_matches_in_doc(&outer, &positions).is_empty());
    }

    #[test]
    fn span_near_empty_clauses_never_matches() {
        let positions = leaf_positions(&[("cat", &[0])]);
        let query = SpanQuery::span_near(std::iter::empty(), 0, true);
        assert!(span_matches_in_doc(&query, &positions).is_empty());
    }

    #[test]
    fn span_near_a_clause_with_no_occurrences_never_matches() {
        let positions = leaf_positions(&[("cat", &[0])]);
        let query = SpanQuery::span_near(
            [
                SpanQuery::span_term("f", "cat"),
                SpanQuery::span_term("f", "sat"),
            ],
            10,
            true,
        );
        assert!(span_matches_in_doc(&query, &positions).is_empty());
    }

    // `search_phrase_query_scored` fixture-driven tests (task #29): reuses the
    // `pos` field's real alpha/beta postings this module's `search_phrase_query`
    // tests already validate at the matching layer.

    #[test]
    fn phrase_query_scored_matches_unscored_doc_set_and_scores_positively() {
        let (fields, doc) = open_fixture();
        let doc = doc.unwrap();
        let doc_in = doc.open();
        let pos_in = doc.open_pos();
        let pay_in = doc.open_pay();

        let mut unscored = VecCollector::default();
        search_phrase_query(
            &fields,
            Some(&doc_in),
            Some(&pos_in),
            Some(&pay_in),
            None,
            &PhraseQuery::new("pos", ["alpha", "beta"]),
            &mut unscored,
        )
        .unwrap();
        assert_eq!(unscored.docs, vec![8555]);

        let mut top = TopDocsCollector::new(10);
        search_phrase_query_scored(
            &fields,
            Some(&doc_in),
            Some(&pos_in),
            Some(&pay_in),
            None,
            &PhraseQuery::new("pos", ["alpha", "beta"]),
            None,
            &mut top,
        )
        .unwrap();
        let hits = top.top_docs();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, 8555);
        assert!(hits[0].score > 0.0);
    }

    #[test]
    fn phrase_query_scored_single_term_delegates_to_term_scoring() {
        let (fields, doc) = open_fixture();
        let doc = doc.unwrap();
        let doc_in = doc.open();

        let mut phrase_top = TopDocsCollector::new(10);
        search_phrase_query_scored(
            &fields,
            Some(&doc_in),
            None,
            None,
            None,
            &PhraseQuery::new("pos", ["alpha"]),
            None,
            &mut phrase_top,
        )
        .unwrap();

        let mut term_top = TopDocsCollector::new(10);
        search_term_query_scored(
            &fields,
            Some(&doc_in),
            None,
            &TermQuery::new("pos", "alpha"),
            None,
            &mut term_top,
        )
        .unwrap();

        let mut phrase_hits: Vec<(i32, f32)> = phrase_top
            .top_docs()
            .iter()
            .map(|h| (h.doc_id, h.score))
            .collect();
        let mut term_hits: Vec<(i32, f32)> = term_top
            .top_docs()
            .iter()
            .map(|h| (h.doc_id, h.score))
            .collect();
        phrase_hits.sort_by_key(|h| h.0);
        term_hits.sort_by_key(|h| h.0);
        assert_eq!(phrase_hits.len(), term_hits.len());
        for ((pd, ps), (td, ts)) in phrase_hits.iter().zip(term_hits.iter()) {
            assert_eq!(pd, td);
            assert!((ps - ts).abs() < 1e-6);
        }
    }

    #[test]
    fn phrase_query_scored_empty_terms_yields_no_hits() {
        let (fields, doc) = open_fixture();
        let doc = doc.unwrap();
        let doc_in = doc.open();
        let mut top = TopDocsCollector::new(10);
        search_phrase_query_scored(
            &fields,
            Some(&doc_in),
            None,
            None,
            None,
            &PhraseQuery::default(),
            None,
            &mut top,
        )
        .unwrap();
        assert!(top.top_docs().is_empty());
    }

    #[test]
    fn phrase_query_scored_missing_term_yields_no_hits() {
        let (fields, doc) = open_fixture();
        let doc = doc.unwrap();
        let doc_in = doc.open();
        let pos_in = doc.open_pos();
        let pay_in = doc.open_pay();
        let mut top = TopDocsCollector::new(10);
        search_phrase_query_scored(
            &fields,
            Some(&doc_in),
            Some(&pos_in),
            Some(&pay_in),
            None,
            &PhraseQuery::new("pos", ["alpha", "zzz-missing"]),
            None,
            &mut top,
        )
        .unwrap();
        assert!(top.top_docs().is_empty());
    }

    #[test]
    fn phrase_query_scored_multi_term_without_pos_input_is_an_error() {
        let (fields, doc) = open_fixture();
        let doc = doc.unwrap();
        let doc_in = doc.open();
        let mut top = TopDocsCollector::new(10);
        let err = search_phrase_query_scored(
            &fields,
            Some(&doc_in),
            None,
            None,
            None,
            &PhraseQuery::new("pos", ["alpha", "beta"]),
            None,
            &mut top,
        )
        .unwrap_err();
        assert!(matches!(err, Error::MissingPosInput));
    }

    #[test]
    fn phrase_query_scored_repeated_phrase_scores_higher_than_single_occurrence() {
        // doc 8556 has "alpha" at 0 and 1 -- "alpha alpha" matches twice there
        // (phrase_freq_exact counts both consecutive starts). A higher phraseFreq
        // must yield a strictly higher BM25 score than a doc with phraseFreq 1,
        // same monotonicity property term scoring already proves for `freq`.
        let (fields, doc) = open_fixture();
        let doc = doc.unwrap();
        let doc_in = doc.open();
        let pos_in = doc.open_pos();
        let pay_in = doc.open_pay();

        let mut top = TopDocsCollector::new(10);
        search_phrase_query_scored(
            &fields,
            Some(&doc_in),
            Some(&pos_in),
            Some(&pay_in),
            None,
            &PhraseQuery::new("pos", ["alpha", "alpha"]),
            None,
            &mut top,
        )
        .unwrap();
        // Only doc 8556 has a consecutive "alpha alpha" alignment (doc 8555 has
        // "alpha" only once, doc 8557 likewise) -- see
        // `phrase_query_duplicate_term_matches_consecutive_occurrences` above.
        let hits = top.top_docs();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, 8556);
        assert!(hits[0].score > 0.0);
    }

    // `Clause::Phrase` inside a `BooleanQuery` (task #29): matching + scoring,
    // plus one nested case.

    #[test]
    fn boolean_must_with_phrase_clause_narrows_the_matched_set() {
        // must = [phrase("alpha beta"), term("alpha")]: phrase matches only 8555;
        // term "alpha" matches 8555, 8556, 8557 -- conjunction is {8555}.
        let (fields, doc) = open_fixture();
        let doc = doc.unwrap();
        let doc_in = doc.open();
        let pos_in = doc.open_pos();
        let pay_in = doc.open_pay();

        let q = BooleanQuery::new().with_must([
            Clause::Phrase(PhraseQuery::new("pos", ["alpha", "beta"])),
            Clause::Term(TermQuery::new("pos", "alpha")),
        ]);
        let mut c = VecCollector::default();
        search_boolean_query(
            &fields,
            Some(&doc_in),
            Some(&pos_in),
            Some(&pay_in),
            None,
            &q,
            &mut c,
        )
        .unwrap();
        assert_eq!(c.docs, vec![8555]);
    }

    #[test]
    fn boolean_should_with_phrase_clause_scores_additively() {
        let (fields, doc) = open_fixture();
        let doc = doc.unwrap();
        let doc_in = doc.open();
        let pos_in = doc.open_pos();
        let pay_in = doc.open_pay();

        let q = BooleanQuery::new().with_should([
            Clause::Phrase(PhraseQuery::new("pos", ["alpha", "beta"])),
            Clause::Term(TermQuery::new("pos", "alpha")),
        ]);
        let mut top = TopDocsCollector::new(10);
        search_boolean_query_scored(
            &fields,
            Some(&doc_in),
            Some(&pos_in),
            Some(&pay_in),
            None,
            &q,
            None,
            &mut top,
        )
        .unwrap();

        let mut phrase_only = TopDocsCollector::new(10);
        search_phrase_query_scored(
            &fields,
            Some(&doc_in),
            Some(&pos_in),
            Some(&pay_in),
            None,
            &PhraseQuery::new("pos", ["alpha", "beta"]),
            None,
            &mut phrase_only,
        )
        .unwrap();
        let mut term_only = TopDocsCollector::new(10);
        search_term_query_scored(
            &fields,
            Some(&doc_in),
            None,
            &TermQuery::new("pos", "alpha"),
            None,
            &mut term_only,
        )
        .unwrap();

        let lookup = |top: &TopDocsCollector, doc_id: i32| -> Option<f32> {
            top.top_docs()
                .iter()
                .find(|h| h.doc_id == doc_id)
                .map(|h| h.score)
        };
        let hits = top.top_docs();
        let mut hit_docs: Vec<i32> = hits.iter().map(|h| h.doc_id).collect();
        hit_docs.sort_unstable();
        assert_eq!(hit_docs, vec![8555, 8556, 8557]);
        for hit in hits {
            let expected = lookup(&phrase_only, hit.doc_id).unwrap_or(0.0)
                + lookup(&term_only, hit.doc_id).unwrap_or(0.0);
            assert!(
                (hit.score - expected).abs() < 1e-4,
                "doc={} got={} expected={}",
                hit.doc_id,
                hit.score,
                expected
            );
        }
    }

    #[test]
    fn nested_boolean_clause_containing_a_phrase_clause_resolves_correctly() {
        // top.must = [nested], nested.should = [phrase("alpha beta"), term("gamma"
        // -- missing, contributes nothing)] -- nested's own disjunction is just the
        // phrase's matched set {8555}; the parent's conjunction (its only clause)
        // must equal that same set.
        let (fields, doc) = open_fixture();
        let doc = doc.unwrap();
        let doc_in = doc.open();
        let pos_in = doc.open_pos();
        let pay_in = doc.open_pay();

        let nested = BooleanQuery::new().with_should([
            Clause::Phrase(PhraseQuery::new("pos", ["alpha", "beta"])),
            Clause::Term(TermQuery::new("pos", "zzz-missing")),
        ]);
        let top_query = BooleanQuery::new().with_must([Clause::Boolean(Box::new(nested))]);

        let mut c = VecCollector::default();
        search_boolean_query(
            &fields,
            Some(&doc_in),
            Some(&pos_in),
            Some(&pay_in),
            None,
            &top_query,
            &mut c,
        )
        .unwrap();
        assert_eq!(c.docs, vec![8555]);

        // Scoring side: the nested clause's phrase contribution must equal the
        // phrase's own standalone score for doc 8555.
        let mut top = TopDocsCollector::new(10);
        search_boolean_query_scored(
            &fields,
            Some(&doc_in),
            Some(&pos_in),
            Some(&pay_in),
            None,
            &top_query,
            None,
            &mut top,
        )
        .unwrap();
        let mut phrase_only = TopDocsCollector::new(10);
        search_phrase_query_scored(
            &fields,
            Some(&doc_in),
            Some(&pos_in),
            Some(&pay_in),
            None,
            &PhraseQuery::new("pos", ["alpha", "beta"]),
            None,
            &mut phrase_only,
        )
        .unwrap();
        let hits = top.top_docs();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, 8555);
        let expected = phrase_only.top_docs()[0].score;
        assert!((hits[0].score - expected).abs() < 1e-4);
    }

    #[test]
    fn boolean_phrase_clause_without_pos_input_is_an_error() {
        let (fields, doc) = open_fixture();
        let doc = doc.unwrap();
        let doc_in = doc.open();
        let q = BooleanQuery::new()
            .with_must([Clause::Phrase(PhraseQuery::new("pos", ["alpha", "beta"]))]);
        let mut c = VecCollector::default();
        let err =
            search_boolean_query(&fields, Some(&doc_in), None, None, None, &q, &mut c).unwrap_err();
        assert!(matches!(err, Error::MissingPosInput));
    }

    // `DisjunctionMaxQuery` (task #32): matching is a pure union, scoring is
    // `max(disjunct scores) + tie_breaker * sum(rest)`. `body`'s known postings
    // (see the `Clause::Boolean`/BM25 tests above and `scoring_fixtures.rs`):
    // cat={0,2}, dog={0,1}, bird={1,4}.

    #[test]
    fn dismax_matches_the_union_of_every_disjunct() {
        let (fields, doc) = open_fixture();
        let doc = doc.unwrap();
        let doc_in = doc.open();

        let q = DisjunctionMaxQuery::new(
            [
                Clause::Term(TermQuery::new("body", "cat")),
                Clause::Term(TermQuery::new("body", "bird")),
            ],
            0.0,
        );
        let mut c = VecCollector::default();
        search_disjunction_max_query(&fields, Some(&doc_in), None, None, None, &q, &mut c).unwrap();
        // cat={0,2} union bird={1,4} = {0,1,2,4}, ascending.
        assert_eq!(c.docs, vec![0, 1, 2, 4]);
    }

    #[test]
    fn dismax_with_no_disjuncts_matches_nothing() {
        let (fields, doc) = open_fixture();
        let doc = doc.unwrap();
        let doc_in = doc.open();

        let q = DisjunctionMaxQuery::new(Vec::<Clause>::new(), 0.0);
        let mut c = VecCollector::default();
        search_disjunction_max_query(&fields, Some(&doc_in), None, None, None, &q, &mut c).unwrap();
        assert!(c.docs.is_empty());
    }

    #[test]
    fn dismax_missing_term_disjunct_contributes_nothing_to_the_union() {
        let (fields, doc) = open_fixture();
        let doc = doc.unwrap();
        let doc_in = doc.open();

        let q = DisjunctionMaxQuery::new(
            [
                Clause::Term(TermQuery::new("body", "cat")),
                Clause::Term(TermQuery::new("body", "zzz-missing")),
            ],
            0.0,
        );
        let mut c = VecCollector::default();
        search_disjunction_max_query(&fields, Some(&doc_in), None, None, None, &q, &mut c).unwrap();
        assert_eq!(c.docs, vec![0, 2]);
    }

    #[test]
    fn dismax_scored_with_zero_tie_breaker_is_pure_max_of_disjunct_scores() {
        // doc 0 matches both cat and dog -- with tie_breaker == 0.0 its score
        // must be exactly max(cat_score(0), dog_score(0)), not their sum.
        let (fields, doc) = open_fixture();
        let doc = doc.unwrap();
        let doc_in = doc.open();

        let mut cat = TopDocsCollector::new(10);
        search_term_query_scored(
            &fields,
            Some(&doc_in),
            None,
            &TermQuery::new("body", "cat"),
            None,
            &mut cat,
        )
        .unwrap();
        let mut dog = TopDocsCollector::new(10);
        search_term_query_scored(
            &fields,
            Some(&doc_in),
            None,
            &TermQuery::new("body", "dog"),
            None,
            &mut dog,
        )
        .unwrap();
        let score_of = |top: &TopDocsCollector, doc_id: i32| -> f32 {
            top.top_docs()
                .iter()
                .find(|h| h.doc_id == doc_id)
                .map(|h| h.score)
                .unwrap()
        };
        let cat0 = score_of(&cat, 0);
        let dog0 = score_of(&dog, 0);
        assert_ne!(
            cat0, dog0,
            "test needs distinct scores to prove max, not sum"
        );
        let expected_max = cat0.max(dog0);

        let q = DisjunctionMaxQuery::new(
            [
                Clause::Term(TermQuery::new("body", "cat")),
                Clause::Term(TermQuery::new("body", "dog")),
            ],
            0.0,
        );
        let mut top = TopDocsCollector::new(10);
        search_disjunction_max_query_scored(
            &fields,
            Some(&doc_in),
            None,
            None,
            None,
            &q,
            None,
            &mut top,
        )
        .unwrap();
        assert!((score_of(&top, 0) - expected_max).abs() < 1e-4);
    }

    #[test]
    fn dismax_scored_tie_breaker_arithmetic_matches_the_exact_formula() {
        // Exact arithmetic proof of `max + tie_breaker * sum(rest)`, computed
        // both ways from independently-derived single-clause scores -- doc 0
        // matches both cat and dog.
        let (fields, doc) = open_fixture();
        let doc = doc.unwrap();
        let doc_in = doc.open();

        let mut cat = TopDocsCollector::new(10);
        search_term_query_scored(
            &fields,
            Some(&doc_in),
            None,
            &TermQuery::new("body", "cat"),
            None,
            &mut cat,
        )
        .unwrap();
        let mut dog = TopDocsCollector::new(10);
        search_term_query_scored(
            &fields,
            Some(&doc_in),
            None,
            &TermQuery::new("body", "dog"),
            None,
            &mut dog,
        )
        .unwrap();
        let score_of = |top: &TopDocsCollector, doc_id: i32| -> f32 {
            top.top_docs()
                .iter()
                .find(|h| h.doc_id == doc_id)
                .map(|h| h.score)
                .unwrap()
        };
        let cat0 = score_of(&cat, 0);
        let dog0 = score_of(&dog, 0);
        let tie_breaker = 0.3f32;
        let expected = cat0.max(dog0) + tie_breaker * cat0.min(dog0);

        let q = DisjunctionMaxQuery::new(
            [
                Clause::Term(TermQuery::new("body", "cat")),
                Clause::Term(TermQuery::new("body", "dog")),
            ],
            tie_breaker,
        );
        let mut top = TopDocsCollector::new(10);
        search_disjunction_max_query_scored(
            &fields,
            Some(&doc_in),
            None,
            None,
            None,
            &q,
            None,
            &mut top,
        )
        .unwrap();
        assert!((score_of(&top, 0) - expected).abs() < 1e-5);
    }

    #[test]
    fn dismax_scored_doc_matching_only_one_disjunct_gets_exactly_that_score() {
        // doc 2 matches only cat (not dog): the tie_breaker term is multiplied
        // by zero "other" contributions, so the score is exactly cat's own
        // score regardless of tie_breaker.
        let (fields, doc) = open_fixture();
        let doc = doc.unwrap();
        let doc_in = doc.open();

        let mut cat = TopDocsCollector::new(10);
        search_term_query_scored(
            &fields,
            Some(&doc_in),
            None,
            &TermQuery::new("body", "cat"),
            None,
            &mut cat,
        )
        .unwrap();
        let score_of = |top: &TopDocsCollector, doc_id: i32| -> f32 {
            top.top_docs()
                .iter()
                .find(|h| h.doc_id == doc_id)
                .map(|h| h.score)
                .unwrap()
        };
        let cat2 = score_of(&cat, 2);

        let q = DisjunctionMaxQuery::new(
            [
                Clause::Term(TermQuery::new("body", "cat")),
                Clause::Term(TermQuery::new("body", "dog")),
            ],
            0.5,
        );
        let mut top = TopDocsCollector::new(10);
        search_disjunction_max_query_scored(
            &fields,
            Some(&doc_in),
            None,
            None,
            None,
            &q,
            None,
            &mut top,
        )
        .unwrap();
        assert!((score_of(&top, 2) - cat2).abs() < 1e-4);
    }

    #[test]
    fn dismax_nested_inside_a_boolean_clause_matches_and_scores_correctly() {
        // top.must = [term(dog), dismax([cat, bird])]: dog={0,1}, dismax union
        // cat∪bird = {0,1,2,4}, conjunction = {0,1}.
        let (fields, doc) = open_fixture();
        let doc = doc.unwrap();
        let doc_in = doc.open();

        let dismax = DisjunctionMaxQuery::new(
            [
                Clause::Term(TermQuery::new("body", "cat")),
                Clause::Term(TermQuery::new("body", "bird")),
            ],
            0.0,
        );
        let q = BooleanQuery::new().with_must([
            Clause::Term(TermQuery::new("body", "dog")),
            Clause::DisjunctionMax(Box::new(dismax)),
        ]);
        let mut c = VecCollector::default();
        search_boolean_query(&fields, Some(&doc_in), None, None, None, &q, &mut c).unwrap();
        assert_eq!(c.docs, vec![0, 1]);
    }

    #[test]
    fn boolean_clause_nested_inside_a_dismax_disjunct_matches_and_scores_correctly() {
        // dismax([term(bird), boolean.must=[cat, dog]]): boolean's own
        // conjunction cat∩dog = {0}; bird = {1,4}. Union = {0,1,4}.
        let (fields, doc) = open_fixture();
        let doc = doc.unwrap();
        let doc_in = doc.open();

        let nested_bool = BooleanQuery::new()
            .with_must([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")]);
        let q = DisjunctionMaxQuery::new(
            [
                Clause::Term(TermQuery::new("body", "bird")),
                Clause::Boolean(Box::new(nested_bool)),
            ],
            0.0,
        );
        let mut c = VecCollector::default();
        search_disjunction_max_query(&fields, Some(&doc_in), None, None, None, &q, &mut c).unwrap();
        assert_eq!(c.docs, vec![0, 1, 4]);
    }

    #[test]
    fn dismax_nested_inside_another_dismax_recurses_to_multiple_levels() {
        // outer dismax([term(bird), inner dismax([cat, dog])]): inner union
        // cat∪dog = {0,1,2}; outer union with bird{1,4} = {0,1,2,4}.
        let (fields, doc) = open_fixture();
        let doc = doc.unwrap();
        let doc_in = doc.open();

        let inner = DisjunctionMaxQuery::new(
            [
                Clause::Term(TermQuery::new("body", "cat")),
                Clause::Term(TermQuery::new("body", "dog")),
            ],
            0.0,
        );
        let outer = DisjunctionMaxQuery::new(
            [
                Clause::Term(TermQuery::new("body", "bird")),
                Clause::DisjunctionMax(Box::new(inner)),
            ],
            0.0,
        );
        let mut c = VecCollector::default();
        search_disjunction_max_query(&fields, Some(&doc_in), None, None, None, &outer, &mut c)
            .unwrap();
        assert_eq!(c.docs, vec![0, 1, 2, 4]);
    }

    #[test]
    fn dismax_phrase_clause_without_pos_input_is_an_error() {
        let (fields, doc) = open_fixture();
        let doc = doc.unwrap();
        let doc_in = doc.open();
        let q = DisjunctionMaxQuery::new(
            [Clause::Phrase(PhraseQuery::new("pos", ["alpha", "beta"]))],
            0.0,
        );
        let mut c = VecCollector::default();
        let err =
            search_disjunction_max_query(&fields, Some(&doc_in), None, None, None, &q, &mut c)
                .unwrap_err();
        assert!(matches!(err, Error::MissingPosInput));
    }

    // `ConstantScoreQuery`/`BoostQuery` (task #33). `body`'s known real postings
    // (see the dismax/boolean tests above): cat={0,2}, dog={0,1}, bird={1,4}.
    //
    // **Cross-engine verification scope, decided here rather than adding a new
    // Java fixture (see the `differential-testing` skill)**: both wrappers are
    // arithmetically trivial compositions over an inner clause whose own scoring
    // is *already* cross-engine-verified -- `Clause::Term`/`Clause::Boolean`/
    // `Clause::DisjunctionMax` scoring was checked against real Lucene's
    // `IndexSearcher`/`TopDocs` output in earlier tasks (`scoring_fixtures.rs`,
    // `dismax_query_fixtures.rs` -- see `docs/parity.md`'s `DisjunctionMaxQuery`
    // row for the exact fixture and its real recorded scores). `ConstantScore`
    // replaces that already-real inner score with a literal constant (no
    // arithmetic to get wrong beyond "return `score` verbatim"); `Boost`
    // multiplies it by a literal `f32` (one `*`, no order-of-operations
    // ambiguity, no norms/idf/tf interaction of its own). Writing a brand-new
    // `Gen*.java` generator to prove `x == x` (constant) or `y == a * b` (a
    // single multiply of Rust's own `f32`, the same float type and operator
    // Java's `float` multiply uses bit-for-bit under IEEE 754) would not
    // exercise any Lucene-specific format or algorithm this port could get
    // subtly wrong -- unlike BM25's `tfNorm`/`idf` formulas or the dismax
    // tie-breaker formula, which *did* need real Lucene ground truth to catch
    // a real bug (see the BM25 `tfNorm` fix task #32 found). Instead, these
    // tests use `search_term_query_scored`'s already-cross-engine-consistent
    // real BM25 score for `body:cat`/`body:dog` at specific docs as the "known
    // real" inner score, and assert the wrapped score is exactly that constant,
    // or exactly that real score times the boost -- i.e. they verify this
    // task's arithmetic against a real (not hand-faked) inner score, just
    // without a second Java fixture generator, which would add fixture
    // maintenance burden without covering anything this reasoning doesn't
    // already cover.

    fn real_score(fields: &BlockTreeFields, doc_in: &DocInput<'_>, term: &str, doc_id: i32) -> f32 {
        let mut top = TopDocsCollector::new(10);
        search_term_query_scored(
            fields,
            Some(doc_in),
            None,
            &TermQuery::new("body", term),
            None,
            &mut top,
        )
        .unwrap();
        top.top_docs()
            .iter()
            .find(|h| h.doc_id == doc_id)
            .map(|h| h.score)
            .unwrap()
    }

    #[test]
    fn constant_score_matching_set_equals_inner_matching_set() {
        let (fields, doc) = open_fixture();
        let doc = doc.unwrap();
        let doc_in = doc.open();

        let q = BooleanQuery::new().with_must([Clause::from(ConstantScoreQuery::new(
            TermQuery::new("body", "cat"),
            1.0,
        ))]);
        let mut c = VecCollector::default();
        search_boolean_query(&fields, Some(&doc_in), None, None, None, &q, &mut c).unwrap();
        assert_eq!(c.docs, vec![0, 2]);
    }

    #[test]
    fn constant_score_with_a_missing_inner_term_matches_nothing() {
        let (fields, doc) = open_fixture();
        let doc = doc.unwrap();
        let doc_in = doc.open();

        let q = BooleanQuery::new().with_must([Clause::from(ConstantScoreQuery::new(
            TermQuery::new("body", "zzz-missing"),
            7.0,
        ))]);
        let mut c = VecCollector::default();
        search_boolean_query(&fields, Some(&doc_in), None, None, None, &q, &mut c).unwrap();
        assert!(c.docs.is_empty());
    }

    #[test]
    fn constant_score_scores_exactly_the_configured_score_regardless_of_inner_score() {
        let (fields, doc) = open_fixture();
        let doc = doc.unwrap();
        let doc_in = doc.open();

        // Real per-doc BM25 scores for cat differ between doc 0 and doc 2 (real
        // Lucene never scores two different docs identically for the same term
        // unless their lengths/freqs coincide) -- proving the constant override
        // discards both, not just one.
        let cat0 = real_score(&fields, &doc_in, "cat", 0);
        let cat2 = real_score(&fields, &doc_in, "cat", 2);
        let constant = 4.25f32;
        assert_ne!(cat0, constant);
        assert_ne!(cat2, constant);

        let q = BooleanQuery::new().with_must([Clause::from(ConstantScoreQuery::new(
            TermQuery::new("body", "cat"),
            constant,
        ))]);
        let mut top = TopDocsCollector::new(10);
        search_boolean_query_scored(&fields, Some(&doc_in), None, None, None, &q, None, &mut top)
            .unwrap();
        let score_of = |top: &TopDocsCollector, doc_id: i32| -> f32 {
            top.top_docs()
                .iter()
                .find(|h| h.doc_id == doc_id)
                .map(|h| h.score)
                .unwrap()
        };
        assert_eq!(score_of(&top, 0), constant);
        assert_eq!(score_of(&top, 2), constant);
    }

    #[test]
    fn boost_matching_set_equals_inner_matching_set() {
        let (fields, doc) = open_fixture();
        let doc = doc.unwrap();
        let doc_in = doc.open();

        let q = BooleanQuery::new().with_must([Clause::from(BoostQuery::new(
            TermQuery::new("body", "dog"),
            2.0,
        ))]);
        let mut c = VecCollector::default();
        search_boolean_query(&fields, Some(&doc_in), None, None, None, &q, &mut c).unwrap();
        assert_eq!(c.docs, vec![0, 1]);
    }

    #[test]
    fn boost_with_a_missing_inner_term_matches_nothing() {
        let (fields, doc) = open_fixture();
        let doc = doc.unwrap();
        let doc_in = doc.open();

        let q = BooleanQuery::new().with_must([Clause::from(BoostQuery::new(
            TermQuery::new("body", "zzz-missing"),
            2.0,
        ))]);
        let mut c = VecCollector::default();
        search_boolean_query(&fields, Some(&doc_in), None, None, None, &q, &mut c).unwrap();
        assert!(c.docs.is_empty());
    }

    #[test]
    fn boost_score_is_exactly_the_inner_real_score_times_boost() {
        let (fields, doc) = open_fixture();
        let doc = doc.unwrap();
        let doc_in = doc.open();

        let dog0 = real_score(&fields, &doc_in, "dog", 0);
        let boost = 2.5f32;

        let q = BooleanQuery::new().with_must([Clause::from(BoostQuery::new(
            TermQuery::new("body", "dog"),
            boost,
        ))]);
        let mut top = TopDocsCollector::new(10);
        search_boolean_query_scored(&fields, Some(&doc_in), None, None, None, &q, None, &mut top)
            .unwrap();
        let score0 = top
            .top_docs()
            .iter()
            .find(|h| h.doc_id == 0)
            .map(|h| h.score)
            .unwrap();
        assert!((score0 - dog0 * boost).abs() < 1e-5);
        assert_ne!(score0, dog0, "boost must actually change the score");
    }

    #[test]
    fn constant_score_nested_inside_a_boolean_query_composes_with_other_clauses() {
        // must = [dog, constant_score(cat, 9.0)]: dog={0,1}, cat={0,2},
        // conjunction = {0}; doc 0's total score is dog's own real score plus
        // the constant 9.0 (real Lucene's additive `BooleanScorer`).
        let (fields, doc) = open_fixture();
        let doc = doc.unwrap();
        let doc_in = doc.open();

        let dog0 = real_score(&fields, &doc_in, "dog", 0);
        let constant = 9.0f32;

        let q = BooleanQuery::new().with_must([
            Clause::Term(TermQuery::new("body", "dog")),
            Clause::from(ConstantScoreQuery::new(
                TermQuery::new("body", "cat"),
                constant,
            )),
        ]);
        let mut c = VecCollector::default();
        search_boolean_query(&fields, Some(&doc_in), None, None, None, &q, &mut c).unwrap();
        assert_eq!(c.docs, vec![0]);

        let mut top = TopDocsCollector::new(10);
        search_boolean_query_scored(&fields, Some(&doc_in), None, None, None, &q, None, &mut top)
            .unwrap();
        let score0 = top
            .top_docs()
            .iter()
            .find(|h| h.doc_id == 0)
            .map(|h| h.score)
            .unwrap();
        assert!((score0 - (dog0 + constant)).abs() < 1e-4);
    }

    #[test]
    fn boost_nested_inside_a_dismax_disjunct_scores_correctly() {
        // dismax([boost(cat, 2.0), term(dog)], tie_breaker=0.0): doc 0 matches
        // both; with tie_breaker 0.0 the winner is whichever disjunct scores
        // higher, and the boosted disjunct's score must be exactly cat0 * 2.0.
        let (fields, doc) = open_fixture();
        let doc = doc.unwrap();
        let doc_in = doc.open();

        let cat0 = real_score(&fields, &doc_in, "cat", 0);
        let dog0 = real_score(&fields, &doc_in, "dog", 0);
        let boost = 2.0f32;
        let boosted_cat0 = cat0 * boost;
        let expected_max = boosted_cat0.max(dog0);

        let q = DisjunctionMaxQuery::new(
            [
                Clause::from(BoostQuery::new(TermQuery::new("body", "cat"), boost)),
                Clause::Term(TermQuery::new("body", "dog")),
            ],
            0.0,
        );
        let mut top = TopDocsCollector::new(10);
        search_disjunction_max_query_scored(
            &fields,
            Some(&doc_in),
            None,
            None,
            None,
            &q,
            None,
            &mut top,
        )
        .unwrap();
        let score0 = top
            .top_docs()
            .iter()
            .find(|h| h.doc_id == 0)
            .map(|h| h.score)
            .unwrap();
        assert!((score0 - expected_max).abs() < 1e-4);
    }

    #[test]
    fn constant_score_wrapping_a_dismax_query_matches_the_dismax_union_and_scores_fixed() {
        let (fields, doc) = open_fixture();
        let doc = doc.unwrap();
        let doc_in = doc.open();

        let dismax = DisjunctionMaxQuery::new(
            [
                Clause::Term(TermQuery::new("body", "cat")),
                Clause::Term(TermQuery::new("body", "bird")),
            ],
            0.0,
        );
        let constant = 6.0f32;
        let q = BooleanQuery::new().with_must([Clause::from(ConstantScoreQuery::new(
            Clause::DisjunctionMax(Box::new(dismax)),
            constant,
        ))]);
        let mut c = VecCollector::default();
        search_boolean_query(&fields, Some(&doc_in), None, None, None, &q, &mut c).unwrap();
        // cat={0,2} union bird={1,4} = {0,1,2,4}.
        assert_eq!(c.docs, vec![0, 1, 2, 4]);

        let mut top = TopDocsCollector::new(10);
        search_boolean_query_scored(&fields, Some(&doc_in), None, None, None, &q, None, &mut top)
            .unwrap();
        for doc_id in [0, 1, 2, 4] {
            let score = top
                .top_docs()
                .iter()
                .find(|h| h.doc_id == doc_id)
                .map(|h| h.score)
                .unwrap();
            assert_eq!(score, constant);
        }
    }

    #[test]
    fn boost_wrapping_a_constant_score_query_multiplies_the_constant() {
        // BoostQuery(ConstantScoreQuery(cat, 3.0), 2.0) -- real Lucene composes
        // the two multiplicatively/replacement in that order: matching docs
        // score exactly 3.0 * 2.0 = 6.0.
        let (fields, doc) = open_fixture();
        let doc = doc.unwrap();
        let doc_in = doc.open();

        let inner_constant = 3.0f32;
        let boost = 2.0f32;
        let q = BooleanQuery::new().with_must([Clause::from(BoostQuery::new(
            Clause::from(ConstantScoreQuery::new(
                TermQuery::new("body", "cat"),
                inner_constant,
            )),
            boost,
        ))]);
        let mut c = VecCollector::default();
        search_boolean_query(&fields, Some(&doc_in), None, None, None, &q, &mut c).unwrap();
        assert_eq!(c.docs, vec![0, 2]);

        let mut top = TopDocsCollector::new(10);
        search_boolean_query_scored(&fields, Some(&doc_in), None, None, None, &q, None, &mut top)
            .unwrap();
        for doc_id in [0, 2] {
            let score = top
                .top_docs()
                .iter()
                .find(|h| h.doc_id == doc_id)
                .map(|h| h.score)
                .unwrap();
            assert_eq!(score, inner_constant * boost);
        }
    }
}
