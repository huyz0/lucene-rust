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
//! Matching semantics follow real `BooleanQuery.rewrite()`
//! (`org.apache.lucene.search.BooleanQuery`, verified against that source rather than
//! guessed): a query with **no `must` and no `should` clauses matches nothing**,
//! regardless of `must_not` — real Lucene rewrites both "no clauses at all" (`clauses
//! .isEmpty()`) and "only `MUST_NOT` clauses" (`clauses.size() ==
//! clauseSets.get(MUST_NOT).size()`) to a `MatchNoDocsQuery`, i.e. a **pure negative
//! query does not mean "match every doc except the excluded ones"** — it means match
//! nothing. When `must` is non-empty, `should` clauses do **not** narrow the matched
//! set at all (they're scoring-only once a `MUST`/`FILTER` clause exists, absent
//! `minimumNumberShouldMatch` — unimplemented here, see `query::BooleanQuery`'s doc
//! comment); the matched set is `must`'s conjunction alone, then `must_not`'s
//! disjunction is subtracted. When `must` is empty but `should` is not, the matched
//! set is `should`'s disjunction, then `must_not` is subtracted the same way.
//!
//! Deferred, tracked in `docs/parity.md`: nested `BooleanQuery` clauses (every clause
//! here is a flat `TermQuery`), `minimumNumberShouldMatch`, and — same as
//! `search_term_query` — relevance scoring (a separate task, #13).
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
pub mod doc_value_query;
pub mod docid_set;
pub mod field_norms;
pub mod query;
pub mod similarity;

pub use collector::{
    Collector, CountCollector, ScoreDoc, ScoringCollector, TopDocsCollector, VecCollector,
};
pub use doc_value_query::{
    search_numeric_range, search_sorted_ord_range, sort_by_numeric_doc_value, MissingValue,
};
pub use field_norms::FieldNorms;
pub use query::{BooleanQuery, PhraseQuery, TermQuery};

use std::collections::HashMap;

use docid_set::{BoxDocIter, Conjunction, Disjunction, Excluding};

use lucene_codecs::blocktree::{self, BlockTreeFields};
use lucene_codecs::postings::{DocInput, PayInput, PosInput};
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

/// Executes `query` (see [`query::BooleanQuery`] and this module's doc comment for
/// the exact matching semantics) against one already-opened segment, feeding every
/// matching **live** doc ID to `collector` in ascending order — same parameter
/// contract as [`search_term_query`], generalized to a `must`/`should`/`must_not`
/// clause list of `TermQuery`s instead of exactly one.
pub fn search_boolean_query<C: Collector>(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &BooleanQuery,
    collector: &mut C,
) -> Result<()> {
    if query.must.is_empty() && query.should.is_empty() {
        // Real `BooleanQuery.rewrite()` turns both "no clauses at all" and "only
        // MUST_NOT clauses" into a `MatchNoDocsQuery` -- see this module's doc
        // comment. Neither case reaches the merge machinery below.
        return Ok(());
    }

    let clause_docs = |clauses: &[TermQuery]| -> Result<Vec<Vec<i32>>> {
        clauses
            .iter()
            .map(|q| term_doc_ids(fields, doc_in, live_docs, q))
            .collect()
    };

    let to_iters = |docs: Vec<Vec<i32>>| -> Vec<BoxDocIter<'static>> {
        docs.into_iter()
            .map(|v| Box::new(v.into_iter()) as BoxDocIter<'static>)
            .collect()
    };

    let base: BoxDocIter<'static> = if !query.must.is_empty() {
        Box::new(Conjunction::new(to_iters(clause_docs(&query.must)?)))
    } else {
        Box::new(Disjunction::new(to_iters(clause_docs(&query.should)?)))
    };

    let matched: BoxDocIter<'static> = if query.must_not.is_empty() {
        base
    } else {
        let excluded: BoxDocIter<'static> =
            Box::new(Disjunction::new(to_iters(clause_docs(&query.must_not)?)));
        Box::new(Excluding::new(base, excluded))
    };

    for doc_id in matched {
        collector.collect(doc_id);
    }
    Ok(())
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

/// Scored sibling of [`search_boolean_query`]: computes the same matched-doc set
/// (`must`'s conjunction, else `should`'s disjunction, minus `must_not`'s
/// disjunction — identical rules to [`search_boolean_query`], see this module's
/// doc comment), but reports each matched doc's score as the **sum of its BM25
/// score across every `must`/`should` clause it satisfies** (mirroring real
/// Lucene's additive `BooleanScorer`; `must_not` clauses never contribute to the
/// score, matching `Occur.MUST_NOT`'s filter-only contract).
///
/// `norms`: real per-doc/avg field length, keyed by field name, for every
/// scored (`must`/`should`) clause's field — a clause whose field has no entry
/// in this map (or when `norms` itself is `None`) falls back to
/// [`similarity::UNNORMED_FIELD_LENGTH`] for that clause, same documented
/// approximation as [`term_doc_scores`]. A `BooleanQuery`'s clauses can span
/// multiple fields, unlike a single [`TermQuery`], hence the map instead of one
/// `FieldNorms`.
pub fn search_boolean_query_scored<C: ScoringCollector>(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &BooleanQuery,
    norms: Option<&HashMap<String, FieldNorms<'_>>>,
    collector: &mut C,
) -> Result<()> {
    if query.must.is_empty() && query.should.is_empty() {
        return Ok(());
    }

    let clause_docs = |clauses: &[TermQuery]| -> Result<Vec<Vec<i32>>> {
        clauses
            .iter()
            .map(|q| term_doc_ids(fields, doc_in, live_docs, q))
            .collect()
    };

    let to_iters = |docs: Vec<Vec<i32>>| -> Vec<BoxDocIter<'static>> {
        docs.into_iter()
            .map(|v| Box::new(v.into_iter()) as BoxDocIter<'static>)
            .collect()
    };

    let base: BoxDocIter<'static> = if !query.must.is_empty() {
        Box::new(Conjunction::new(to_iters(clause_docs(&query.must)?)))
    } else {
        Box::new(Disjunction::new(to_iters(clause_docs(&query.should)?)))
    };

    let matched: BoxDocIter<'static> = if query.must_not.is_empty() {
        base
    } else {
        let excluded: BoxDocIter<'static> =
            Box::new(Disjunction::new(to_iters(clause_docs(&query.must_not)?)));
        Box::new(Excluding::new(base, excluded))
    };

    // Sum each scoring clause's (doc_id -> score) contributions across `must`
    // and `should` (never `must_not`, which only filters -- see doc comment).
    let mut scores: HashMap<i32, f32> = HashMap::new();
    for clause in query.must.iter().chain(query.should.iter()) {
        let clause_norms = norms.and_then(|m| m.get(&clause.field));
        for (doc_id, score) in term_doc_scores(fields, doc_in, live_docs, clause, clause_norms)? {
            *scores.entry(doc_id).or_insert(0.0) += score;
        }
    }

    for doc_id in matched {
        collector.collect(doc_id, scores.get(&doc_id).copied().unwrap_or(0.0));
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

/// Executes `query` (see [`query::PhraseQuery`] for the exact exact-adjacent-
/// position, `slop == 0` semantics) against one already-opened segment, feeding
/// every matching **live** doc ID to `collector` in ascending order -- same
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
/// possibly qualify) *and* [`phrase_matches_in_doc`] finds a valid alignment for
/// that doc's per-term position lists.
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
        if phrase_matches_in_doc(&term_positions) {
            collector.collect(doc_id);
        }
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
        search_boolean_query(&fields, doc_in.as_ref(), None, &q, &mut c).unwrap();
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
        search_boolean_query(&fields, doc_in.as_ref(), None, &q, &mut c).unwrap();
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
        search_boolean_query(&fields, doc_in.as_ref(), None, &q, &mut c).unwrap();
        assert_eq!(c.docs, vec![2]);
    }

    #[test]
    fn boolean_pure_must_not_matches_nothing() {
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let mut c = VecCollector::default();
        let q = BooleanQuery::new().with_must_not([TermQuery::new("body", "dog")]);
        search_boolean_query(&fields, doc_in.as_ref(), None, &q, &mut c).unwrap();
        assert!(c.docs.is_empty());
    }

    #[test]
    fn boolean_empty_query_matches_nothing() {
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let mut c = VecCollector::default();
        let q = BooleanQuery::new();
        search_boolean_query(&fields, doc_in.as_ref(), None, &q, &mut c).unwrap();
        assert!(c.docs.is_empty());
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
        search_boolean_query(&fields, doc_in.as_ref(), None, &q, &mut c).unwrap();
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
        search_boolean_query(&fields, doc_in.as_ref(), Some(&live_docs), &q, &mut c).unwrap();
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
        assert_eq!(c.docs, vec![8555, 8556]);
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
}
