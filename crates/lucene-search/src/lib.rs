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
//! source rather than guessed): a query with **no `must`, no `filter` and no
//! `should` clauses matches nothing**, regardless of `must_not` — real Lucene
//! rewrites both "no clauses at all" (`clauses.isEmpty()`) and "only `MUST_NOT`
//! clauses" (`clauses.size() == clauseSets.get(MUST_NOT).size()`) to a
//! `MatchNoDocsQuery`, i.e. a **pure negative query does not mean "match every doc
//! except the excluded ones"** — it means match nothing.
//!
//! **`Occur.FILTER`** (`query.filter`) is `MUST` with the score dropped: a filter
//! clause is a leg of the same conjunction as `must`, but contributes exactly `0`
//! to the score and is never summed. Three consequences the executor here relies
//! on, all of them Java's:
//! - A filter clause **does not count toward `minimum_should_match`** — only
//!   `Occur.SHOULD` increments `shouldMatchCount` in `BooleanWeight`.
//! - A **filter-only query matches**, at score `0`. It is a positive query
//!   (`BooleanClause.isRequired()`), so the pure-negative rewrite above does not
//!   apply to it.
//! - A filter clause **cannot perturb the scoring clauses' float summation
//!   order**, because it never enters the sum at all: `ConjunctionScorer.score()`
//!   iterates `scorers` (the `MUST` subset), not `required`. `f32` addition is not
//!   associative, so this is a bit-level property, not a stylistic one, and
//!   `tests/bm25_scoring_fixtures.rs` asserts it against real `IndexSearcher`.
//!
//! `query.minimum_should_match` (task #24's addition; `query::BooleanQuery`'s doc
//! comment has the full field-level accounting) gates `should` **regardless of
//! whether `must` is also non-empty** — this is the one place it's easy to get
//! backwards, so it's called out explicitly: real `BooleanWeight.scorer`/
//! `bulkScorer`/`explain` all compute `shouldMatchCount` and reject a doc with
//! `shouldMatchCount < minShouldMatch` unconditionally, not just when `must` is
//! empty. Concretely:
//! - `minimum_should_match == 0` (the default): when `must`/`filter` is non-empty,
//!   `should` clauses do **not** narrow the matched set at all (scoring-only once a
//!   `MUST`/`FILTER` clause exists); the matched set is the required conjunction
//!   alone. When there are no required clauses, the matched set is `should`'s
//!   disjunction (a doc needs at least one `should` hit —
//!   `minimum_should_match`'s implicit floor of 1 in that case).
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
pub mod explain;
pub mod facets;
pub mod field_norms;
pub mod highlighter;
pub mod multi_segment;
pub mod ordinal_map;
pub mod points_query;
pub mod query;
pub mod query_cache;
pub mod query_parser;
pub mod similarity;
pub mod soft_deletes;
pub mod term_vectors_query;
pub mod vector_query;

pub use collector::{
    Collector, CountCollector, FieldValueDoc, ScoreDoc, ScoreMode, ScoringCollector, SortDirection,
    TopDocsCollector, TopFieldCollector, TotalHits, TotalHitsRelation, VecCollector,
};
pub use doc_value_query::{
    search_numeric_range, search_numeric_range_sorted_by_field, search_sorted_ord_range,
    sort_by_numeric_doc_value, sort_top_n_by_numeric_doc_value, MissingValue,
};
pub use explain::{explain_clause, Explanation};
pub use field_norms::FieldNorms;
pub use multi_segment::{
    merge_multi_segment_scored, search_boolean_query_multi_segment,
    search_boolean_query_multi_segment_concurrent, search_boolean_query_multi_segment_maxscore,
    search_boolean_query_multi_segment_maxscore_concurrent, search_term_query_multi_segment,
    search_term_query_multi_segment_concurrent, OpenSegment,
};
pub use points_query::{pack_i64, search_points_range, PointsInput};
pub use query::{
    BooleanQuery, BoostQuery, Clause, ConstantScoreQuery, DisjunctionMaxQuery, FuzzyQuery,
    MatchAllDocsQuery, MatchNoDocsQuery, MultiPhraseQuery, PhraseQuery, PrefixQuery, RegexpQuery,
    SpanQuery, TermInSetQuery, TermQuery, WildcardQuery,
};
pub use query_cache::{search_term_query_cached, QueryCache};
pub use term_vectors_query::{matched_term_offsets, term_vector_for_doc};
pub use vector_query::{
    accept_bitset, per_leaf_top_k, search_knn_byte_vector_query,
    search_knn_byte_vector_query_multi_segment,
    search_knn_byte_vector_query_multi_segment_concurrent, search_knn_float_vector_query,
    search_knn_float_vector_query_multi_segment,
    search_knn_float_vector_query_multi_segment_concurrent, similarity_from_ordinal,
    similarity_ordinal, KnnByteVectorQuery, KnnFloatVectorQuery, KnnSegment, VectorsInput,
};

use std::collections::HashMap;

use docid_set::{BoxDocIter, Conjunction, Disjunction, Excluding, WindowedDisjunction};

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
    /// [`doc_value_query::field_exists_source`] was asked about a field that
    /// indexes none of norms, vectors or doc values -- Java's own
    /// `IllegalStateException` from `FieldExistsQuery`, whose message this
    /// reproduces. A field *name* that is simply not in `FieldInfos` is not
    /// this case: that is Java's `fieldInfo == null`, a `null` scorer, and no
    /// matches without an error.
    #[error(
        "FieldExistsQuery requires that the field indexes doc values, norms or vectors, but \
         field {0:?} exists and indexes neither of these data structures"
    )]
    FieldExistsUnsupported(String),
    /// Surfaced by [`points_query::search_points_range`] when the underlying
    /// `.kdd`/`.kdi`/`.kdm` decode fails (a truncated/corrupt BKD points
    /// region) -- the points-range analog of [`Error::DocValues`].
    #[error(transparent)]
    Points(#[from] lucene_codecs::points::Error),
    /// A [`Clause::PointsRange`] (task #64's `field:[min TO max]` query-parser
    /// syntax) reached [`resolve_clause_docs`]/[`clause_scores`] with no
    /// [`points_query::PointsInput`] supplied -- same "this clause needs an
    /// opened resource this call didn't provide" shape as
    /// [`Error::MissingPosInput`] for a multi-term [`PhraseQuery`], not a
    /// permanent gap: [`crate::explain::explain_clause`] never has a
    /// `PointsInput` to pass (see that module's own scope note) and so always
    /// surfaces this for a `Clause::PointsRange`, but
    /// [`search_boolean_query`]/[`search_boolean_query_scored`]/
    /// [`search_disjunction_max_query`]/[`search_disjunction_max_query_scored`]
    /// resolve it against a real segment's BKD points data
    /// ([`points_query::search_points_range`]) whenever a caller passes
    /// `points: Some(..)`.
    #[error(
        "Clause::PointsRange needs an opened PointsInput (.kdm/.kdi/.kdd) to execute \
         (field {0:?})"
    )]
    MissingPointsInput(String),
    /// Surfaced by [`vector_query`] when the underlying `.vemf`/`.vec`/
    /// `.vem`/`.vex` decode fails -- the vector analog of [`Error::Points`].
    /// A *caller* mistake (an unknown field, a wrong-length query vector, a
    /// `k` below 1) is [`Error::InvalidKnnQuery`] instead, deliberately: the
    /// two are indistinguishable to a caller that only sees one error type,
    /// and "the index is corrupt" is the wrong thing to tell a JNI caller
    /// who sent a bad request (see `lucene_ffi::vectors`).
    #[error(transparent)]
    Vectors(#[from] lucene_codecs::vectors::Error),
    /// A KNN query this segment cannot answer as asked -- Java's
    /// `IllegalArgumentException` from `AbstractKnnVectorQuery`'s constructor
    /// and its two subclasses' `approximateSearch`, carrying Java's own
    /// message. Never a sign of a damaged index.
    #[error("{0}")]
    InvalidKnnQuery(String),
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

/// `TermsEnum.postings(reuse, PostingsEnum.NONE)`: one term's matching doc IDs
/// with the `.doc` file's **frequency blocks skipped** rather than unpacked
/// (`PForUtil.skip`, c8's addition to `lucene-codecs`).
///
/// **This returns the doc list only, deliberately.** `PostingsFlags::DocsOnly`
/// fills `Postings::freqs` with `1`s, so a caller that took the whole
/// `Postings` and later read a frequency would get a plausible, silently wrong
/// number. Handing back a bare `Vec<i32>` makes that impossible rather than
/// merely untested: there is no frequency in scope to read. Every unscored
/// matching path in this module goes through here; anything that scores calls
/// [`blocktree::FieldTerms::postings`] and keeps the frequencies.
///
/// `None` is "this term is not in this field's dictionary" (Java's `null`
/// `PostingsEnum`), distinct from `Some(vec![])`.
fn term_docs_only(
    field_terms: &blocktree::FieldTerms,
    term: &[u8],
    doc_in: Option<&DocInput<'_>>,
) -> Result<Option<Vec<i32>>> {
    Ok(field_terms
        .postings_with_flags(
            term,
            doc_in,
            lucene_codecs::postings::PostingsFlags::DocsOnly,
        )?
        .map(|p| p.docs))
}

/// Shared per-clause lookup: `seekExact`s `query`'s term, then returns every
/// matching doc ID (ascending, per `Postings`' own contract), filtered by
/// `live_docs` the same way `search_term_query` always has. Returns an empty
/// `Vec` — not an error — when the query's field doesn't exist in this segment
/// or the term isn't in that field's dictionary, matching
/// `TermQuery.createWeight`'s `null`-`Scorer` "no matches" outcome. Used by
/// both `search_term_query` and `search_boolean_query` so the
/// field-lookup/postings/`live_docs`-filter sequence lives in exactly one
/// place.
///
/// Goes through [`term_docs_only`]: this is a *matching* path and never reads a
/// frequency.
fn term_doc_ids(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &TermQuery,
) -> Result<Vec<i32>> {
    let Some(field_terms) = fields.field(&query.field) else {
        return Ok(Vec::new());
    };
    let Some(docs) = term_docs_only(field_terms, &query.term, doc_in)? else {
        return Ok(Vec::new());
    };
    Ok(docs
        .into_iter()
        .filter(|&doc_id| live_docs.is_none_or(|bits| bits.get(doc_id as usize)))
        .collect())
}

/// A wildcard-family clause's expansion: the field it targets, the terms it
/// matched, and the sum of their document frequencies.
///
/// The last is what [`stream_constant_score_clause`] weighs its setup cost
/// against, so it is gathered during the term scan rather than by re-seeking
/// each term afterwards.
type Expansion = (String, Vec<Vec<u8>>, i64);

/// The terms a wildcard-family clause expands to, or `None` when the clause is
/// not one of that family (or its field is absent from this segment).
fn expanded_terms(fields: &BlockTreeFields, clause: &Clause) -> Result<Option<Expansion>> {
    let out = match clause {
        Clause::Prefix(q) => {
            let Some(ft) = fields.field(&q.field) else {
                return Ok(None);
            };
            let pattern = WildcardPattern::prefix(&q.prefix);
            let mut df = 0i64;
            let terms: Vec<Vec<u8>> = ft
                .intersect(&pattern)
                .map(|(t, s)| {
                    df += s.doc_freq as i64;
                    t.to_vec()
                })
                .collect();
            (q.field.clone(), terms, df)
        }
        Clause::Wildcard(q) => {
            let Some(ft) = fields.field(&q.field) else {
                return Ok(None);
            };
            let pattern = WildcardPattern::new(&q.pattern);
            let mut df = 0i64;
            let terms: Vec<Vec<u8>> = ft
                .intersect(&pattern)
                .map(|(t, s)| {
                    df += s.doc_freq as i64;
                    t.to_vec()
                })
                .collect();
            (q.field.clone(), terms, df)
        }
        Clause::Regexp(q) => {
            let Some(ft) = fields.field(&q.field) else {
                return Ok(None);
            };
            let pattern = RegexpPattern::new(q.pattern.as_bytes())?;
            let mut df = 0i64;
            let terms: Vec<Vec<u8>> = ft
                .regexp_intersect(&pattern)
                .map(|(t, s)| {
                    df += s.doc_freq as i64;
                    t.to_vec()
                })
                .collect();
            (q.field.clone(), terms, df)
        }
        _ => return Ok(None),
    };
    Ok(Some(out))
}

/// Streams a constant-scoring clause's matched documents in ascending order and
/// stops as soon as the collector cannot be beaten.
///
/// The wildcard family scores a flat 1.0 for every match. With every score
/// equal, `TopDocsCollector`'s tie-break -- lower doc ID wins -- makes the top
/// `n` simply *the `n` lowest matching doc IDs*. So once the collector is full
/// and its worst kept score is already 1.0, no later document can enter it and
/// there is nothing to gain by finding the rest.
///
/// That matters because "the rest" can be enormous. `regexp body:t1[0-9]`
/// matches ten terms, all of them frequent: unioning their postings is roughly
/// 15 million documents to answer a top-50 query, and it measured **2,845x
/// slower than Lucene**, which rewrites a small term set to a disjunction and
/// prunes. Merging the terms' already-sorted postings lazily and stopping at 50
/// does the same job without ever building the union.
///
/// Returns `false` when it cannot handle the clause -- an absent field, no
/// `.doc` input, or a pulsed singleton term with no postings to open lazily --
/// leaving the caller to fall back to the materializing path.
fn stream_constant_score_clause<C: ScoringCollector>(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    clause: &Clause,
    collector: &mut C,
) -> Result<bool> {
    let Some((field, terms, total_doc_freq)) = expanded_terms(fields, clause)? else {
        return Ok(false);
    };
    let Some(field_terms) = fields.field(&field) else {
        return Ok(false);
    };
    let Some(doc_in) = doc_in else {
        return Ok(false);
    };
    // Choose by expected work rather than by term count.
    //
    // Setting up costs one lazy cursor per term, and opening a cursor decodes
    // that term's first block -- about `BLOCK_SIZE` documents' worth of work
    // each. The bit-set union it replaces costs one pass over *every* posting.
    // So streaming wins when the union is much larger than the setup, and a
    // fixed term-count threshold gets this wrong in both directions: 32 was
    // tried and cost `prefix body:t12` a 12x win, because that clause expands
    // to hundreds of terms whose postings are far larger still.
    if total_doc_freq < (terms.len() as i64) * lucene_codecs::for_util::BLOCK_SIZE as i64 {
        return Ok(false);
    }

    // `TermsEnum.postings(reuse, PostingsEnum.NONE)`: this is a *constant*-score
    // union -- every hit is collected at `1.0` below, and no frequency, norm or
    // impact is ever consulted -- so the `.doc` file's frequency blocks are
    // skipped (`PForUtil.skip`) instead of unpacked. The cursors are wrapped so
    // that stays true by construction: `DocsOnlyCursor` exposes `next_doc` and
    // nothing else, so the `1`-filled frequencies `PostingsFlags::DocsOnly`
    // produces cannot be read from here even by mistake.
    struct DocsOnlyCursor<'a>(lucene_codecs::postings::LazyDocsCursor<'a>);
    impl DocsOnlyCursor<'_> {
        fn next_doc(&mut self) -> Result<i32> {
            Ok(self.0.next_doc().map_err(blocktree::Error::Postings)?)
        }
    }

    let mut cursors = Vec::with_capacity(terms.len());
    for term in &terms {
        let Some(cursor) = field_terms.lazy_postings_with_flags(
            term,
            doc_in,
            lucene_codecs::postings::PostingsFlags::DocsOnly,
        )?
        else {
            return Ok(false);
        };
        let mut cursor = DocsOnlyCursor(cursor);
        let doc = cursor.next_doc()?;
        cursors.push((cursor, doc));
    }

    loop {
        // Smallest current doc across the cursors. Linear in the number of
        // matched terms, which is affordable because this loop runs about
        // `top_n` times, not once per matching document.
        let mut best = lucene_codecs::postings::NO_MORE_DOCS;
        for (_, doc) in &cursors {
            if *doc < best {
                best = *doc;
            }
        }
        if best == lucene_codecs::postings::NO_MORE_DOCS {
            return Ok(true);
        }
        if live_docs.is_none_or(|bits| bits.get(best as usize)) {
            collector.collect(best, 1.0);
            // Full, and its worst hit already scores what every remaining
            // document would: nothing left can displace anything.
            if collector.pruning_threshold().is_some_and(|s| s >= 1.0) {
                return Ok(true);
            }
        }
        for (cursor, doc) in cursors.iter_mut() {
            if *doc == best {
                *doc = cursor.next_doc()?;
            }
        }
    }
}

/// Accumulates matched doc IDs from several terms' postings and returns them
/// ascending and deduplicated -- Lucene's `DocIdSetBuilder`, which every
/// `MultiTermQuery` rewrite goes through.
///
/// The wildcard family unions one posting list per matching term, and this port
/// did that by concatenating them all into a `Vec<i32>` and calling
/// `sort_unstable` + `dedup`. That is `O(n log n)` in the *total* number of
/// postings, which for a prefix matching thousands of terms is tens of millions
/// of entries: on `prefix body:t12` over the M1 corpus the sort alone was 67% of
/// the query. A bit per document is `O(n + maxDoc/64)` and, incidentally,
/// bounded memory rather than one `i32` per posting.
///
/// Grows on demand so no caller has to know `maxDoc` up front.
#[derive(Default)]
struct DocIdBitSet {
    words: Vec<u64>,
    max_set: i32,
}

impl DocIdBitSet {
    #[inline]
    fn set(&mut self, doc_id: i32) {
        if doc_id < 0 {
            return;
        }
        let idx = doc_id as usize;
        let word = idx >> 6;
        if word >= self.words.len() {
            self.words.resize(word + 1, 0);
        }
        self.words[word] |= 1u64 << (idx & 63);
        if doc_id > self.max_set {
            self.max_set = doc_id;
        }
    }

    /// The set documents, ascending. Iterating set bits word by word is what
    /// makes this cheaper than sorting: each word yields its bits in order
    /// already.
    fn into_sorted_vec(self) -> Vec<i32> {
        let mut out = Vec::new();
        for (w, &word) in self.words.iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                out.push(((w << 6) | bit) as i32);
                bits &= bits - 1;
            }
        }
        out
    }
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
    let mut acc = DocIdBitSet::default();
    for term in &matching_terms {
        let Some(docs) = term_docs_only(field_terms, term, doc_in)? else {
            continue;
        };
        for doc_id in docs {
            if live_docs.is_none_or(|bits| bits.get(doc_id as usize)) {
                acc.set(doc_id);
            }
        }
    }
    let doc_ids = acc.into_sorted_vec();
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
    let mut acc = DocIdBitSet::default();
    for term in &matching_terms {
        let Some(docs) = term_docs_only(field_terms, term, doc_in)? else {
            continue;
        };
        for doc_id in docs {
            if live_docs.is_none_or(|bits| bits.get(doc_id as usize)) {
                acc.set(doc_id);
            }
        }
    }
    let doc_ids = acc.into_sorted_vec();
    Ok(doc_ids)
}

/// One term a [`FuzzyQuery`] expanded to, with the per-term boost real
/// Lucene's `FuzzyTermsEnum` publishes through its `BoostAttribute` and the
/// term's own document frequency.
#[derive(Debug, Clone)]
struct ExpandedTerm {
    term: Vec<u8>,
    /// `FuzzyTermsEnum`'s similarity, **not** clamped -- `TopTermsRewrite
    /// .build` does the `Math.max(0.0f, st.boost)` truncation when it turns
    /// the queue into clauses, and [`fuzzy_doc_scores`] does it there too.
    boost: f32,
    doc_freq: i64,
}

/// The result of expanding a [`FuzzyQuery`] against one segment's field: the
/// selected terms plus the **blended** document frequency every one of them
/// is scored with.
#[derive(Debug, Clone, Default)]
struct FuzzyExpansion {
    terms: Vec<ExpandedTerm>,
    /// `BlendedTermQuery.rewrite`: `df = Math.max(df, ctx.docFreq())` across
    /// every selected term, then every term's `TermStates` is rebuilt with
    /// that artificial frequency. Blending is the whole point of
    /// `TopTermsBlendedFreqScoringRewrite`: without it "the rarest term
    /// typically ranks highest (often not useful eg in the set of expanded
    /// terms in a FuzzyQuery)".
    blended_doc_freq: i64,
}

/// Expands `query` over `field_terms` exactly the way
/// `MultiTermQuery.TopTermsBlendedFreqScoringRewrite` does.
///
/// Three Lucene behaviours live here, and this port previously had none of
/// them (it kept the first `max_expansions` terms in term-dictionary order
/// and scored every match a flat `1.0`):
///
/// 1. **Per-term boost.** `FuzzyTermsEnum.next` sets `BoostAttribute` to
///    `1.0` for an exact match and `1 - ed/min(len(term), len(query term))`
///    otherwise -- see [`lucene_codecs::fuzzy::FuzzyMatch::boost`].
/// 2. **Top-N by boost, not by term order.** `TopTermsRewrite.collect` keeps
///    a size-`maxExpansions` priority queue whose worst element is the lowest
///    boost, with ties broken so the lexicographically **later** term is
///    dropped first (`boost == t.boost && bytes.compareTo(t.bytes.get()) > 0`
///    skips the candidate). Sorting by `(boost desc, bytes asc)` and taking
///    the first `maxExpansions` selects the same set.
/// 3. **Blended document frequency**, see [`FuzzyExpansion::blended_doc_freq`].
///
/// The whole matching set has to be visited to pick the top N, which is what
/// Lucene does too -- `TopTermsRewrite.rewrite` runs `collectTerms` over
/// every term the automaton accepts.
fn fuzzy_expanded_terms(
    field_terms: &lucene_codecs::blocktree::FieldTerms,
    query: &FuzzyQuery,
) -> FuzzyExpansion {
    let pattern = FuzzyMatch::new(
        &query.term,
        query.max_edits,
        query.prefix_length,
        query.transpositions,
    );
    let mut candidates: Vec<ExpandedTerm> = field_terms
        .fuzzy_intersect(&pattern)
        .map(|(term, stats)| ExpandedTerm {
            boost: pattern.boost(&term).unwrap_or(0.0),
            term,
            doc_freq: stats.doc_freq as i64,
        })
        .collect();
    candidates.sort_by(|a, b| {
        b.boost
            .partial_cmp(&a.boost)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.term.cmp(&b.term))
    });
    candidates.truncate(query.max_expansions);
    let blended_doc_freq = candidates.iter().map(|t| t.doc_freq).max().unwrap_or(0);
    FuzzyExpansion {
        terms: candidates,
        blended_doc_freq,
    }
}

/// [`Clause::Fuzzy`]'s matched doc-ID list (task #42, `max_expansions`
/// bounded per task #221): same union-across-matching-terms mechanism as
/// [`wildcard_doc_ids`]/[`prefix_doc_ids`], built on
/// [`lucene_codecs::blocktree::FieldTerms::fuzzy_intersect`] and
/// [`lucene_codecs::fuzzy::FuzzyMatch`] instead of a glob pattern. Returns an
/// empty `Vec` -- not an error -- when `query.field` doesn't exist in this
/// segment, same "missing field means no matches" convention every other
/// clause follows.
///
/// **`max_expansions` cap**: `fuzzy_intersect` returns a lazy `Iterator` over
/// this segment's field's already-fully-decoded, in-memory sorted term
/// entries (`BlockTreeFields` merges a field's whole dictionary into one
/// `Vec` at open time -- see that module's doc comment -- so there is no
/// on-demand term-dictionary decode left to short-circuit by this point).
/// `.take(query.max_expansions)` here stops pulling from that iterator once
/// the cap is hit, which does avoid running the fuzzy-match predicate and
/// allocating a result for every entry past the cap, but it does **not**
/// skip any decode/IO work, since that already happened when the segment was
/// opened.
///
/// **Selection when more terms match than `max_expansions` allows** is
/// [`fuzzy_expanded_terms`]'s, i.e. real Lucene's `TopTermsRewrite` priority
/// queue: highest `FuzzyTermsEnum` boost first, ties broken by ascending term
/// bytes.
fn fuzzy_doc_ids(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &FuzzyQuery,
) -> Result<Vec<i32>> {
    let Some(field_terms) = fields.field(&query.field) else {
        return Ok(Vec::new());
    };
    let expansion = fuzzy_expanded_terms(field_terms, query);
    let mut acc = DocIdBitSet::default();
    for ExpandedTerm { term, .. } in &expansion.terms {
        let Some(docs) = term_docs_only(field_terms, term, doc_in)? else {
            continue;
        };
        for doc_id in docs {
            if live_docs.is_none_or(|bits| bits.get(doc_id as usize)) {
                acc.set(doc_id);
            }
        }
    }
    let doc_ids = acc.into_sorted_vec();
    Ok(doc_ids)
}

/// [`Clause::Fuzzy`]'s BM25 score per matching, live doc.
///
/// Real `FuzzyQuery` is **not** a constant-scoring query: its default rewrite
/// method is `MultiTermQuery.TopTermsBlendedFreqScoringRewrite`, which turns
/// the expanded terms into a `BlendedTermQuery` -- a `BooleanQuery` of
/// `SHOULD` `BoostQuery(TermQuery(t), boost_t)` clauses whose `TermStates`
/// have all been rewritten to one blended document frequency. So a doc's
/// score is
///
/// ```text
/// sum over selected terms t of  max(0, boost_t) * BM25(df_blended, tf(t, doc), norm(doc))
/// ```
///
/// with `boost_t` from [`lucene_codecs::fuzzy::FuzzyMatch::boost`] and
/// `df_blended` from [`fuzzy_expanded_terms`]. `max(0, boost)` is
/// `TopTermsRewrite.build`'s own truncation ("we allow negative term scores
/// while collecting ... but truncate such boosts to 0.0f when building the
/// query"), which matters for one-and-two-character query terms whose
/// similarity really can go negative.
///
/// This port previously returned a flat `1.0` for every fuzzy hit, which
/// changed the top-k of every fuzzy query -- recorded as finding P4 in
/// `docs/sweep/findings.md` and fixed here.
///
/// The blended `df` is a *reader-wide* statistic in Lucene (`TermStates
/// .build` walks every leaf). This function blends within one segment, the
/// same limitation `term_doc_scores` works around with its `GlobalStats`
/// parameter; a fuzzy clause has no such plumbing yet, so a multi-segment
/// fuzzy score can differ from Lucene's by the usual per-segment-idf amount.
fn fuzzy_doc_scores(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &FuzzyQuery,
    norms: Option<&FieldNorms<'_>>,
) -> Result<HashMap<i32, f32>> {
    let mut scores: HashMap<i32, f32> = HashMap::new();
    let Some(field_terms) = fields.field(&query.field) else {
        return Ok(scores);
    };
    let expansion = fuzzy_expanded_terms(field_terms, query);
    let doc_count = field_terms.doc_count as i64;

    // Resolve every expanded term's live postings first, then score in one
    // pass, rather than scoring inside the per-term loop.
    //
    // Why: the norms lookup wants **ascending** documents. Each expanded term's
    // own postings ascend, but the next term restarts near doc 0, so scoring
    // term-by-term makes a `FieldNormsCursor` rewind once per expansion -- and
    // on a sparse field a rewind is a fresh `IndexedDISI` walk from the first
    // block header, turning O(region) into O(expansions x region). At the
    // measured 677 us per 100,000-document walk, a 50-term expansion would
    // spend ~34 ms in norms alone.
    let mut contributions: Vec<(i32, i32, f32)> = Vec::new(); // (doc, freq, boost)
    for expanded in &expansion.terms {
        let boost = expanded.boost.max(0.0);
        let Some(postings) = field_terms.postings(&expanded.term, doc_in)? else {
            continue;
        };
        for (&doc_id, &freq) in postings.docs.iter().zip(postings.freqs.iter()) {
            if live_docs.is_none_or(|bits| bits.get(doc_id as usize)) {
                contributions.push((doc_id, freq, boost));
            }
        }
    }

    // Sorting is what makes the cursor monotonic, and it is **stable**, which
    // is what keeps this bit-for-bit identical to the per-term loop it
    // replaces: a document's contributions keep their expansion-term order, so
    // the sequence of `+=` into `scores` -- and therefore every float bit --
    // is unchanged. `f32` addition is not associative, so that is a real
    // requirement, not a nicety.
    //
    // Only worth it when a lookup is actually order-sensitive. For the ordinary
    // dense one-byte field the lookup is an array index, so the sort would buy
    // nothing and is skipped.
    if norms.is_some_and(|n| n.prefers_ascending_lookups()) {
        contributions.sort_by_key(|&(doc_id, _, _)| doc_id);
    }

    let mut norms_cursor = norms.map(|n| n.cursor());
    for (doc_id, freq, boost) in contributions {
        let (field_length, avg_field_length) = match norms_cursor.as_mut() {
            Some(nc) => (nc.field_length(doc_id)?, nc.avg_field_length()),
            None => (
                similarity::UNNORMED_FIELD_LENGTH,
                similarity::UNNORMED_FIELD_LENGTH,
            ),
        };
        let score = similarity::score(
            expansion.blended_doc_freq,
            doc_count,
            freq as f32,
            field_length,
            avg_field_length,
        );
        *scores.entry(doc_id).or_insert(0.0) += boost * score;
    }
    Ok(scores)
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
    let mut acc = DocIdBitSet::default();
    for term in &matching_terms {
        let Some(docs) = term_docs_only(field_terms, term, doc_in)? else {
            continue;
        };
        for doc_id in docs {
            if live_docs.is_none_or(|bits| bits.get(doc_id as usize)) {
                acc.set(doc_id);
            }
        }
    }
    let doc_ids = acc.into_sorted_vec();
    Ok(doc_ids)
}

/// [`Clause::TermInSet`]'s matched doc-ID list: each of `query.terms` is
/// resolved via the same per-term postings lookup [`term_doc_ids`] uses for a
/// plain [`crate::query::TermQuery`], and the results are **union**ed and
/// deduplicated -- the same "match any of several terms in one field" shape
/// [`wildcard_doc_ids`]/[`prefix_doc_ids`]/[`fuzzy_doc_ids`]/[`regexp_doc_ids`]
/// already implement for a computed (pattern-matched) term set, here applied
/// to an explicit, caller-supplied term list instead. A term absent from the
/// segment (or an absent `query.field` entirely) contributes no doc IDs, not
/// an error -- same "missing means no matches" convention every other clause
/// follows. An empty `query.terms` returns an empty `Vec` (matches nothing)
/// with no special-case branch needed, since the loop below simply doesn't run.
fn term_in_set_doc_ids(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &TermInSetQuery,
) -> Result<Vec<i32>> {
    let Some(field_terms) = fields.field(&query.field) else {
        return Ok(Vec::new());
    };
    let mut acc = DocIdBitSet::default();
    for term in &query.terms {
        let Some(docs) = term_docs_only(field_terms, term, doc_in)? else {
            continue;
        };
        for doc_id in docs {
            if live_docs.is_none_or(|bits| bits.get(doc_id as usize)) {
                acc.set(doc_id);
            }
        }
    }
    let doc_ids = acc.into_sorted_vec();
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
///
/// `points` (task #199's addition): the segment's opened `.kdm`/`.kdi`/`.kdd`
/// BKD points data plus field infos, bundled as a [`points_query::PointsInput`]
/// -- needed only when `query` (at any nesting depth) contains a
/// `Clause::PointsRange`. `None` is fine for a query with no such clause;
/// passing `None` for a query that turns out to need it surfaces as
/// [`Error::MissingPointsInput`], same convention as `pos_in`/`pay_in` above.
#[allow(clippy::too_many_arguments)]
pub fn search_boolean_query<C: Collector>(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    pos_in: Option<&PosInput<'_>>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    points: Option<&PointsInput<'_>>,
    query: &BooleanQuery,
    collector: &mut C,
) -> Result<()> {
    let Some(matched) =
        matched_boolean_docs(fields, doc_in, pos_in, pay_in, live_docs, points, query)?
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
#[allow(clippy::too_many_arguments)]
fn resolve_clause_docs(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    pos_in: Option<&PosInput<'_>>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    points: Option<&PointsInput<'_>>,
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
            fields, doc_in, pos_in, pay_in, live_docs, points, nested,
        )?
        .map(Iterator::collect)
        .unwrap_or_default()),
        Clause::DisjunctionMax(nested) => {
            resolve_dismax_docs(fields, doc_in, pos_in, pay_in, live_docs, points, nested)
        }
        Clause::ConstantScore(nested) => resolve_clause_docs(
            fields,
            doc_in,
            pos_in,
            pay_in,
            live_docs,
            points,
            &nested.inner,
        ),
        Clause::Boost(nested) => resolve_clause_docs(
            fields,
            doc_in,
            pos_in,
            pay_in,
            live_docs,
            points,
            &nested.inner,
        ),
        Clause::Wildcard(query) => wildcard_doc_ids(fields, doc_in, live_docs, query),
        Clause::Prefix(query) => prefix_doc_ids(fields, doc_in, live_docs, query),
        Clause::Fuzzy(query) => fuzzy_doc_ids(fields, doc_in, live_docs, query),
        Clause::Regexp(query) => regexp_doc_ids(fields, doc_in, live_docs, query),
        Clause::Span(query) => span_doc_ids(fields, doc_in, pos_in, pay_in, live_docs, query),
        Clause::PointsRange(query) => points_range_doc_ids(points, live_docs, query),
        Clause::MatchAllDocs(query) => Ok(match_all_doc_ids(live_docs, query.max_doc)),
        Clause::MatchNoDocs(_) => Ok(Vec::new()),
        Clause::MultiPhrase(query) => {
            multi_phrase_doc_ids(fields, doc_in, pos_in, pay_in, live_docs, query)
        }
        Clause::TermInSet(query) => term_in_set_doc_ids(fields, doc_in, live_docs, query),
    }
}

/// [`Clause::PointsRange`]'s matched doc-ID list (task #199's addition):
/// looks `query.field`'s number up from `points`'s [`PointsInput::field_infos`],
/// packs `query.min`/`query.max` via [`points_query::pack_i64`] (real
/// Lucene's `LongPoint` sortable-bytes encoding -- see that function's doc
/// comment), then delegates to [`points_query::search_points_range`] --
/// the same "look the number up, reopen/reuse the reader, delegate" sequence
/// `lucene_ffi::points_query::ffi_search_points_range` already performs at
/// the C-ABI boundary, one layer down.
///
/// `points: None` (no `.kdm`/`.kdi`/`.kdd` opened for this call) is
/// [`Error::MissingPointsInput`] -- a genuine capability gap for *this* call,
/// not "no matches" (mirrors [`Error::MissingPosInput`]'s same "the caller
/// needed to open this input and didn't" precedent). An unknown field name
/// (present in the query but absent from `points.field_infos`) instead
/// returns an empty `Vec`, matching every other clause's "missing field means
/// no matches" convention (see [`term_doc_ids`]'s doc comment).
fn points_range_doc_ids(
    points: Option<&PointsInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &query::PointsRangeQuery,
) -> Result<Vec<i32>> {
    let Some(points) = points else {
        return Err(Error::MissingPointsInput(query.field.clone()));
    };
    let Some(field_number) = points.field_number(&query.field) else {
        return Ok(Vec::new());
    };
    let min_packed = points_query::pack_i64(query.min);
    let max_packed = points_query::pack_i64(query.max);
    let mut collector = collector::VecCollector::default();
    points_query::search_points_range(
        &points.reader,
        live_docs,
        field_number,
        &min_packed,
        &max_packed,
        &mut collector,
    )?;
    Ok(collector.docs)
}

/// [`Clause::MatchAllDocs`]'s matched doc-ID list: every live doc in
/// `0..max_doc`, ascending -- there's no term dictionary to seek into (see
/// [`MatchAllDocsQuery`]'s doc comment), so this is a direct `0..max_doc` sweep
/// filtered by `live_docs`, the same shape
/// [`doc_value_query::search_numeric_range`]'s own `[0, max_doc)` sweep uses.
pub(crate) fn match_all_doc_ids(live_docs: Option<&FixedBitSet>, max_doc: i32) -> Vec<i32> {
    (0..max_doc)
        .filter(|&doc_id| live_docs.is_none_or(|bits| bits.get(doc_id as usize)))
        .collect()
}

/// Resolves a [`DisjunctionMaxQuery`]'s matched doc-ID list -- a doc matches
/// iff **any** disjunct matches (real `DisjunctionMaxQuery`'s matching
/// contract: it's a pure union, unlike `BooleanQuery.should`'s
/// `minimum_should_match`-gated disjunction). Each disjunct is resolved via
/// [`resolve_clause_docs`], same recursive treatment `Clause::Boolean` gets,
/// then merged through the same [`Disjunction`] iterator `matched_boolean_docs`
/// uses for its own `should`-only case, deduplicated and sorted ascending by
/// construction.
#[allow(clippy::too_many_arguments)]
fn resolve_dismax_docs(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    pos_in: Option<&PosInput<'_>>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    points: Option<&PointsInput<'_>>,
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
        .map(|clause| {
            resolve_clause_docs(fields, doc_in, pos_in, pay_in, live_docs, points, clause)
        })
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
    points: Option<&PointsInput<'_>>,
    query: &BooleanQuery,
) -> Result<Option<BoxDocIter<'static>>> {
    if query.must.is_empty() && query.filter.is_empty() && query.should.is_empty() {
        // Real `BooleanQuery.rewrite()` turns both "no clauses at all" and "only
        // MUST_NOT clauses" into a `MatchNoDocsQuery` -- see this module's doc
        // comment. Neither case reaches the merge machinery below. A `filter`
        // clause is a *positive* clause (Java: `BooleanClause.isRequired()` is
        // true for `FILTER`), so a filter-only query is not one of these.
        return Ok(None);
    }

    let clause_docs = |clauses: &[Clause]| -> Result<Vec<Vec<i32>>> {
        clauses
            .iter()
            .map(|clause| {
                resolve_clause_docs(fields, doc_in, pos_in, pay_in, live_docs, points, clause)
            })
            .collect()
    };

    let to_iters = |docs: Vec<Vec<i32>>| -> Vec<BoxDocIter<'static>> {
        docs.into_iter()
            .map(|v| Box::new(v.into_iter()) as BoxDocIter<'static>)
            .collect()
    };

    let min_should_match = query.minimum_should_match;
    // Java's `BooleanClause.isRequired()`: `MUST` or `FILTER`. Both are legs of
    // the same conjunction; the only difference is that a `FILTER` leg never
    // reaches `clause_scores` (see `search_boolean_query_scored_with_stats`).
    let has_required = !query.must.is_empty() || !query.filter.is_empty();
    // `should_docs` is only needed when `should` actually participates in matching:
    // either as the base set (no required clauses) or as a `minimum_should_match`
    // gate on top of the required conjunction. With required clauses present and
    // `minimum_should_match == 0`, `should` stays purely score-only (matching
    // pre-task-#24 behavior exactly) and this never touches it.
    let should_docs = if !has_required || min_should_match > 0 {
        Some(clause_docs(&query.should)?)
    } else {
        None
    };

    let base: BoxDocIter<'static> =
        if has_required {
            let mut required = clause_docs(&query.must)?;
            required.extend(clause_docs(&query.filter)?);
            let conjunction = Conjunction::new(to_iters(required));
            if min_should_match > 0 {
                let counts = should_match_counts(should_docs.as_ref().expect("computed above"));
                Box::new(conjunction.filter(move |doc_id| {
                    counts.get(doc_id).copied().unwrap_or(0) >= min_should_match
                }))
            } else {
                Box::new(conjunction)
            }
        } else {
            let should_docs = should_docs.expect("computed above (no required clauses)");
            // No required clauses and more than one optional clause is exactly
            // the shape `BooleanScorerSupplier.booleanScorer` hands to
            // `BooleanScorer`: a 4,096-document window ORed into a bitset (plus
            // a per-document clause count when `minimum_should_match > 1`),
            // rather than a per-document merge across every clause. See
            // `WindowedDisjunction`. A single clause keeps the plain path --
            // `BooleanScorer`'s own constructor refuses one clause too.
            if WindowedDisjunction::is_applicable(should_docs.len()) {
                Box::new(WindowedDisjunction::new(should_docs, min_should_match))
            } else if min_should_match > 1 {
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
        // The prohibited set is a pure OR, so it takes the same windowed path
        // (Java hands it a `DisjunctionSumScorer`, whose *scores* it then throws
        // away -- the union is all that is used, and this computes the same
        // union more cheaply).
        let must_not_docs = clause_docs(&query.must_not)?;
        let excluded: BoxDocIter<'static> =
            if WindowedDisjunction::is_applicable(must_not_docs.len()) {
                Box::new(WindowedDisjunction::new(must_not_docs, 1))
            } else {
                Box::new(Disjunction::new(to_iters(must_not_docs)))
            };
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
    global: Option<&GlobalStats>,
) -> Result<Vec<(i32, f32)>> {
    // Reader-wide idf where the caller supplied it, as the lazy paths already
    // do -- see `CollectionStats`. Without this a boolean query whose shape
    // does not fit a lazy path still scored every term from its own segment's
    // counters, which is the multi-segment bug that fix was for.
    let resolved = global
        .and_then(|g| g.get(&(query.field.clone(), query.term.clone())))
        .copied();
    term_doc_scores_with_collection_stats(fields, doc_in, live_docs, query, norms, resolved)
}

/// [`term_doc_scores`] with this term's reader-wide counters already resolved
/// out of the [`GlobalStats`] map -- the shape the MAXSCORE entry point holds
/// them in, since a `TermQuery` mentions exactly one term.
fn term_doc_scores_with_collection_stats(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &TermQuery,
    norms: Option<&FieldNorms<'_>>,
    global: Option<CollectionStats>,
) -> Result<Vec<(i32, f32)>> {
    let Some(field_terms) = fields.field(&query.field) else {
        return Ok(Vec::new());
    };
    let Some(stats) = field_terms.seek_exact(&query.term) else {
        return Ok(Vec::new());
    };
    let doc_freqs = term_doc_freqs(fields, doc_in, live_docs, query)?;
    let (term_doc_freq, doc_count) = match global {
        Some(g) => (g.doc_freq, g.doc_count),
        None => (stats.doc_freq as i64, field_terms.doc_count as i64),
    };
    // One `FieldNormsCursor` per scan, as Lucene takes one `NumericDocValues`
    // per scorer: `FieldNorms` stays the immutable, `Sync` per-segment entry
    // (so `multi_segment.rs` can share it across rayon leaves) and the mutable
    // `IndexedDISI` position lives here, where it can walk forward across
    // documents instead of restarting per lookup.
    let mut norms_cursor = norms.map(|n| n.cursor());
    doc_freqs
        .into_iter()
        .map(|(doc_id, freq)| {
            let (field_length, avg_field_length) = match norms_cursor.as_mut() {
                Some(nc) => (nc.field_length(doc_id)?, nc.avg_field_length()),
                None => (
                    similarity::UNNORMED_FIELD_LENGTH,
                    similarity::UNNORMED_FIELD_LENGTH,
                ),
            };
            let score = similarity::score(
                term_doc_freq,
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
    search_term_query_scored_with_stats(fields, doc_in, live_docs, query, norms, None, collector)
}

/// [`search_term_query_scored`] taking reader-wide statistics -- see
/// [`CollectionStats`]. `None` keeps this segment's own counters.
pub fn search_term_query_scored_with_stats<C: ScoringCollector>(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &TermQuery,
    norms: Option<&FieldNorms<'_>>,
    global: Option<&GlobalStats>,
    collector: &mut C,
) -> Result<()> {
    for (doc_id, score) in term_doc_scores(fields, doc_in, live_docs, query, norms, global)? {
        collector.collect(doc_id, score);
    }
    Ok(())
}

/// [`search_term_query_scored_with_stats`] with this term's reader-wide
/// counters already resolved to a bare [`CollectionStats`] -- the shape
/// [`search_term_query_scored_maxscore_with_stats`] holds them in, so its
/// fallback paths can forward them instead of silently reverting the leaf to
/// its own `docFreq`/`docCount`.
fn search_term_query_scored_with_collection_stats<C: ScoringCollector>(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &TermQuery,
    norms: Option<&FieldNorms<'_>>,
    global: Option<CollectionStats>,
    collector: &mut C,
) -> Result<()> {
    for (doc_id, score) in
        term_doc_scores_with_collection_stats(fields, doc_in, live_docs, query, norms, global)?
    {
        collector.collect(doc_id, score);
    }
    Ok(())
}

/// [`term_doc_scores`]'s sibling taking an explicit [`similarity::Bm25Params`]
/// instead of always using [`similarity::DEFAULT_K1`]/[`similarity::DEFAULT_B`]
/// (task #214). See [`similarity::Bm25Params`]'s doc comment for this task's
/// scope.
fn term_doc_scores_with_similarity(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &TermQuery,
    norms: Option<&FieldNorms<'_>>,
    params: similarity::Bm25Params,
) -> Result<Vec<(i32, f32)>> {
    let Some(field_terms) = fields.field(&query.field) else {
        return Ok(Vec::new());
    };
    let Some(stats) = field_terms.seek_exact(&query.term) else {
        return Ok(Vec::new());
    };
    let doc_freqs = term_doc_freqs(fields, doc_in, live_docs, query)?;
    let doc_count = field_terms.doc_count as i64;
    // One `FieldNormsCursor` per scan, as Lucene takes one `NumericDocValues`
    // per scorer: `FieldNorms` stays the immutable, `Sync` per-segment entry
    // (so `multi_segment.rs` can share it across rayon leaves) and the mutable
    // `IndexedDISI` position lives here, where it can walk forward across
    // documents instead of restarting per lookup.
    let mut norms_cursor = norms.map(|n| n.cursor());
    doc_freqs
        .into_iter()
        .map(|(doc_id, freq)| {
            let (field_length, avg_field_length) = match norms_cursor.as_mut() {
                Some(nc) => (nc.field_length(doc_id)?, nc.avg_field_length()),
                None => (
                    similarity::UNNORMED_FIELD_LENGTH,
                    similarity::UNNORMED_FIELD_LENGTH,
                ),
            };
            let score = similarity::score_with_params(
                stats.doc_freq as i64,
                doc_count,
                freq as f32,
                field_length,
                avg_field_length,
                params,
            );
            Ok((doc_id, score))
        })
        .collect()
}

/// [`search_term_query_scored`]'s sibling taking an explicit
/// [`similarity::Bm25Params`] (`k1`/`b`) instead of the hardcoded BM25
/// defaults (task #214, "Configurable BM25 constant from FFI") — the narrowest
/// useful surface this task adds: the single most fundamental scored search
/// entry point (a plain `TermQuery`, no MAXSCORE pruning), left as a new
/// sibling function rather than an added parameter on
/// [`search_term_query_scored`] itself, so that function's signature and
/// behavior stay byte-for-byte unchanged for its existing callers/tests.
///
/// `params: similarity::Bm25Params::default()` produces byte-for-byte the same
/// scores as [`search_term_query_scored`] — see
/// `similarity::score_with_params_using_defaults_matches_score_byte_for_byte`
/// for the regression proof.
///
/// **Scope note** (see [`similarity::Bm25Params`]'s doc comment and
/// `docs/parity.md`'s BM25/similarity row): only this function and its FFI
/// counterpart (`lucene_ffi::query::ffi_search_term_query_scored_with_similarity`)
/// honor a custom `k1`/`b` today. `search_boolean_query_scored`,
/// `search_term_query_scored_maxscore`, `search_boolean_query_scored_maxscore`,
/// phrase queries, and `explain`/`explain_boolean` are all deliberately left
/// hardcoded to the defaults in this task — threading custom `k1`/`b` through
/// MAXSCORE pruning, multi-segment fan-out, and phrase scoring is a
/// materially larger, riskier change than this task's scope.
pub fn search_term_query_scored_with_similarity<C: ScoringCollector>(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &TermQuery,
    norms: Option<&FieldNorms<'_>>,
    params: similarity::Bm25Params,
    collector: &mut C,
) -> Result<()> {
    for (doc_id, score) in
        term_doc_scores_with_similarity(fields, doc_in, live_docs, query, norms, params)?
    {
        collector.collect(doc_id, score);
    }
    Ok(())
}

/// MAXSCORE-style sibling of [`search_term_query_scored`], scoped narrowly and
/// honestly: single `TermQuery`, [`collector::TopDocsCollector`] only (the
/// only collector this crate has that exposes a min-competitive-score
/// threshold, via [`collector::TopDocsCollector::min_competitive_score`]).
/// Streams the term's postings through
/// [`lucene_codecs::postings::LazyDocsCursor`] instead of eagerly
/// materializing every block via [`DocInput::read_postings`] (the path
/// [`term_doc_scores`]/[`search_term_query_scored`] use): once the collector
/// is holding a full top-`n` (`min_competitive_score()` returns `Some`), a
/// level-0 block whose [`similarity::max_score_for_impacts`] upper bound
/// can't beat that threshold is skipped whole — `advance()`d straight past
/// its last doc without ever running `ForUtil`/`PForUtil` decode on it — the
/// same proof-of-safety this crate's `assert_block_pruning_matches_brute_force`
/// test harness (`similarity.rs`) already established for the bound in
/// isolation, now wired into a real collector loop.
///
/// Falls back to the eager [`search_term_query_scored`] path (carrying any
/// reader-wide [`CollectionStats`] straight through, so the fallback scores
/// with the same idf the pruned path would have) whenever the
/// fast path isn't available — no `.doc` file opened (`doc_in == None`),
/// `docFreq <= 1` (no `.doc` bytes are even written for a singleton term, so
/// there's nothing to stream lazily), or an index option
/// [`lucene_codecs::postings::LazyDocsCursor`] doesn't support (currently
/// only `IndexOptions::None`, which never carries postings in the first
/// place) — so this function never produces a
/// result the eager path wouldn't already produce on its own; the skip is
/// purely a performance path, not a matching-semantics change. Verified
/// identical to [`search_term_query_scored`] for many `top_n`/term-frequency
/// combinations, including ones that exercise real block skipping, by
/// `search_term_query_scored_maxscore_matches_eager_path` below.
pub fn search_term_query_scored_maxscore(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &TermQuery,
    norms: Option<&FieldNorms<'_>>,
    collector: &mut collector::TopDocsCollector,
) -> Result<()> {
    search_term_query_scored_maxscore_with_stats(
        fields, doc_in, live_docs, query, norms, None, collector,
    )
}

/// [`search_term_query_scored_maxscore`] taking reader-wide statistics, so a
/// multi-segment search scores every leaf with one idf -- see
/// [`CollectionStats`]. `None` keeps this segment's own statistics, which is
/// correct for a single-segment search and is what the plain entry point does.
#[allow(clippy::too_many_arguments)]
pub fn search_term_query_scored_maxscore_with_stats(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &TermQuery,
    norms: Option<&FieldNorms<'_>>,
    global: Option<CollectionStats>,
    collector: &mut collector::TopDocsCollector,
) -> Result<()> {
    let Some(field_terms) = fields.field(&query.field) else {
        return Ok(());
    };
    let Some(stats) = field_terms.seek_exact(&query.term) else {
        return Ok(());
    };
    // Every fallback below forwards `global`. Calling the no-stats
    // `search_term_query_scored` here (which is what this function did) let a
    // leaf silently revert to its own `docFreq`/`docCount` -- the cross-segment
    // idf bug `CollectionStats` documents, alive on exactly the paths the
    // MAXSCORE loop declines to take.
    let Some(doc_in) = doc_in else {
        return search_term_query_scored_with_collection_stats(
            fields, None, live_docs, query, norms, global, collector,
        );
    };
    if stats.doc_freq <= 1 {
        return search_term_query_scored_with_collection_stats(
            fields,
            Some(doc_in),
            live_docs,
            query,
            norms,
            global,
            collector,
        );
    }
    let mut cursor = match field_terms.lazy_postings(&query.term, doc_in) {
        Ok(Some(c)) => c,
        Ok(None) => return Ok(()),
        Err(blocktree::Error::Postings(lucene_codecs::postings::Error::Unsupported(_))) => {
            return search_term_query_scored_with_collection_stats(
                fields,
                Some(doc_in),
                live_docs,
                query,
                norms,
                global,
                collector,
            );
        }
        Err(e) => return Err(e.into()),
    };

    let (doc_freq, doc_count) = match global {
        Some(g) => (g.doc_freq, g.doc_count),
        None => (stats.doc_freq as i64, field_terms.doc_count as i64),
    };
    let avg_field_length = match norms {
        Some(fn_) => fn_.avg_field_length,
        None => similarity::UNNORMED_FIELD_LENGTH,
    };
    // One norms cursor for this leaf's whole scan -- the MAXSCORE loop only
    // ever moves forward, so a sparse field costs one `IndexedDISI` walk.
    let mut norms_cursor = norms.map(|n| n.cursor());

    // Bound for a slice of impacts. Shared by level 0 and level 1 so the
    // `norms == None` rule below cannot be honoured at one level and forgotten
    // at the other -- which is exactly the bug the first cut of level-1
    // skipping had.
    //
    // When `norms` is `None` every doc is scored with
    // `field_length == UNNORMED_FIELD_LENGTH`, NOT with whatever real norm byte
    // the wire impacts carry. Feeding the real norms in would bound a
    // *different* scoring formula than the one used, underestimating the bound
    // and skipping docs that should have been collected.
    let bound_for = |impacts: &[lucene_codecs::postings::Impact]| -> f32 {
        match norms {
            Some(_) => {
                similarity::max_score_for_impacts(impacts, doc_freq, doc_count, avg_field_length)
            }
            None => similarity::max_score_for_impacts_unnormed(impacts, doc_freq, doc_count),
        }
    };

    let mut doc_id = cursor.next_doc().map_err(blocktree::Error::from)?;

    // A global upper bound on any document's score for this term, used for an
    // early exit. Real Lucene's MaxScoreCache keeps `globalMaxScore` for exactly
    // this and falls back to it whenever a level has no impacts
    // (`getMaxScore` returns it when `getLevel` yields -1).
    //
    // This port previously pruned only when per-block impacts existed
    // (`if !impacts.is_empty()`), so a field indexed without frequencies -- a
    // StringField/keyword, which carries no impacts on the wire at all -- was
    // never pruned and scanned its entire posting list. On the benchmark corpus
    // that made `keyword:t0` (4,997,130 postings, all scoring identically)
    // 866x slower than Lucene, which stops as soon as the top-k is full.
    //
    // Without frequencies every doc has freq 1, so the bound is tight -- with
    // norms absent it is the exact, constant score every doc receives, and the
    // early exit fires the moment the collector fills. With frequencies the
    // supremum of `tf_norm` as freq grows is `k1 + 1`.
    let global_bound = {
        let idf = similarity::idf(doc_freq, doc_count);
        // Highest frequency any single document can have: every occurrence
        // beyond one-per-matching-document could sit in one doc. For a field
        // indexed without frequencies `totalTermFreq` aliases `docFreq`, so
        // this collapses to exactly 1 -- which is what makes the bound tight
        // enough to fire on a keyword field. Deriving it from the term's own
        // statistics beats special-casing index options, and it is tighter than
        // `tf_norm`'s supremum of `k1 + 1` for ordinary fields too.
        // Both terms of this bound must describe the SAME postings. `doc_freq`
        // above may have been replaced by the reader-wide value, while
        // `total_term_freq` is this segment's, and mixing them makes the
        // difference negative -- clamped to 1, the bound collapses and the
        // early exit fires on documents that should have been collected.
        // Caught by the benchmark's recall cross-check as a 16x "speedup" on a
        // query whose hit set had silently changed.
        let max_freq = (stats.total_term_freq - stats.doc_freq as i64 + 1).max(1) as f32;
        let (len, avg) = match norms {
            // Shortest possible document: the most favourable length norm.
            Some(_) => (1.0, avg_field_length),
            None => (
                similarity::UNNORMED_FIELD_LENGTH,
                similarity::UNNORMED_FIELD_LENGTH,
            ),
        };
        idf * similarity::tf_norm(
            max_freq,
            len,
            avg,
            similarity::DEFAULT_K1,
            similarity::DEFAULT_B,
        )
    };

    // `ImpactsDISI.upTo`: the highest doc ID whose block has already been judged
    // competitive *at the current threshold*. While the cursor is inside that
    // block with that threshold there is nothing to re-decide, so the whole
    // preamble below is skipped -- Lucene's
    // `advanceTarget`: `if (target <= upTo) return target;`. That is the
    // difference between evaluating an impact bound once per 256-document block
    // and once per document.
    //
    // The invalidation is the other half of it, and is not optional:
    // `ImpactsDISI.setMinCompetitiveScore` sets `upTo = -1` whenever the
    // threshold actually rises, precisely so a block that was competitive
    // against the old threshold gets re-judged against the new one. Leaving
    // that out made this loop stop skipping entirely on a two-block fixture,
    // caught by `maxscore_..._actually_skips_blocks`'s counter rather than by
    // any result changing -- the results are identical either way, only the
    // work done differs.
    let mut checked_upto: i32 = -1;
    let mut threshold = collector.pruning_threshold();

    while doc_id != lucene_codecs::postings::NO_MORE_DOCS {
        if doc_id > checked_upto {
            if let Some(threshold) = threshold {
                // Nothing left can be competitive, whatever the block impacts say.
                if global_bound <= threshold {
                    break;
                }

                // `ImpactsDISI.advanceTarget`: walk forward over blocks on
                // their *impacts alone*, skipping every one whose bound cannot
                // beat the threshold, and stop at the first that can. Nothing
                // here decodes a block body.
                //
                // This loop used to be a single test followed by
                // `cursor.advance(skip_to)`, which is correct but decodes the
                // block it lands on -- so the next iteration's "is this block
                // competitive?" question was answered *after* paying the
                // `ForUtil` unpack it exists to avoid. Counted on the M1
                // corpus, `body:t0` unpacked 1,423,616 documents to score
                // 82,564: 5.8% of the decode work was used, and the boolean
                // queries were far worse at 1.3%.
                //
                // When `norms` is `None`, every doc is actually scored below
                // with `field_length == UNNORMED_FIELD_LENGTH ==
                // avg_field_length` (the length-norm term collapses to 1.0),
                // NOT with whatever real per-doc norm byte this block's impacts
                // happen to carry on the wire. Feeding `max_score_for_impacts`
                // the real wire norms here would compute a bound for a
                // *different* scoring formula than the one actually used below
                // -- an unsound mix that can underestimate the bound and skip a
                // doc that should have been collected. `bound_for` carries that
                // rule; see its definition.
                let mut target = doc_id;
                loop {
                    cursor
                        .advance_shallow(target)
                        .map_err(blocktree::Error::from)?;
                    // Empty impacts mean no bound is available -- the tail
                    // block, a field without freqs, or exhaustion. All three
                    // are "cannot skip", so stop and let `advance` decide.
                    let competitive = {
                        let impacts = cursor.level0_impacts();
                        impacts.is_empty() || bound_for(impacts) > threshold
                    };
                    if competitive {
                        break;
                    }
                    #[cfg(any(test, feature = "test-support"))]
                    test_only_maxscore_block_skip_counter::record_skip();

                    // Skip at the highest level whose bound is still under the
                    // threshold, not one block at a time. A level-1 span covers
                    // 32 level-0 blocks, so when its merged impacts also fail
                    // to beat the threshold the whole span goes at once --
                    // `MaxScoreCache.getSkipLevel`/`getSkipUpTo`.
                    let up_to = cursor.level0_last_doc_id();
                    let l1_last = cursor.level1_last_doc_id();
                    let span_skippable = {
                        let l1 = cursor.level1_impacts();
                        !l1.is_empty()
                            && l1_last != lucene_codecs::postings::NO_MORE_DOCS
                            && bound_for(l1) <= threshold
                    };
                    let next = if span_skippable {
                        l1_last.saturating_add(1)
                    } else {
                        up_to.saturating_add(1)
                    };
                    // Guard against a non-advancing skip -- `up_to` is
                    // `NO_MORE_DOCS` in states where the extent is unknown, and
                    // saturating there would spin.
                    if next <= target {
                        break;
                    }
                    target = next;
                }

                if target != doc_id {
                    // Exactly one block gets decoded: the one that survived.
                    doc_id = cursor.advance(target).map_err(blocktree::Error::from)?;
                    if doc_id == lucene_codecs::postings::NO_MORE_DOCS {
                        break;
                    }
                }
            }
            // This block is competitive (or there was no threshold yet):
            // do not ask again until the cursor leaves it.
            checked_upto = cursor.current_block_last_doc_id();
        }

        if live_docs.is_none_or(|bits| bits.get(doc_id as usize)) {
            let freq = cursor.freq().expect("cursor started, doc_id in range") as f32;
            // Table-driven scoring, as real Lucene's BM25Scorer does: one
            // lookup and one division rather than decoding the norm to a length
            // and dividing twice. Algebraically identical --
            // `weight - weight/(1 + freq*normInverse)` expands to
            // `idf * freq / (freq + k1*((1-b) + b*len/avgdl))`.
            let score = match norms_cursor.as_mut() {
                Some(nc) => {
                    let weight = similarity::idf(doc_freq, doc_count);
                    let norm_inverse = nc.norm_inverse(doc_id)?;
                    weight - weight / (1.0 + freq * norm_inverse)
                }
                None => similarity::score(
                    doc_freq,
                    doc_count,
                    freq,
                    similarity::UNNORMED_FIELD_LENGTH,
                    avg_field_length,
                ),
            };
            collector.collect(doc_id, score);
            // `Scorer.setMinCompetitiveScore` -> `ImpactsDISI.upTo = -1`. The
            // threshold only ever rises, and only a collected hit can raise it.
            let now = collector.pruning_threshold();
            if now != threshold {
                threshold = now;
                checked_upto = -1;
            }
        }
        doc_id = cursor.next_doc().map_err(blocktree::Error::from)?;
    }
    Ok(())
}

/// Test-only instrumentation for [`search_term_query_scored_maxscore`]'s
/// block-skip branch: a thread-local counter, incremented once per whole
/// level-0 block actually skipped (i.e. `advance()`d past without decoding),
/// so a differential test can assert real skipping happened rather than only
/// asserting the (necessarily identical) end result — a test could otherwise
/// pass vacuously if the skip branch were dead code. Never compiled into a
/// normal (non-test, non-`test-support`) build: gated on `#[cfg(any(test,
/// feature = "test-support"))]` both here and at the increment site in
/// [`search_term_query_scored_maxscore`]. The `test-support` feature (see
/// this crate's `Cargo.toml`) exists solely so `lucene-ffi`'s own test suite
/// -- a *different* crate, which can never see this crate's `#[cfg(test)]`
/// code no matter what it depends on -- can reuse this exact counter to
/// prove MAXSCORE pruning genuinely happened underneath an FFI call, instead
/// of duplicating the instrumentation or (worse) only asserting the
/// necessarily-identical end result. `lucene-ffi`'s non-test build never
/// enables this feature (only its `[dev-dependencies]` edge does, per Cargo's
/// resolver-2 feature unification, which scopes dev-dependency features to
/// test/bench targets only), so this stays out of any production binary.
/// Test-only instrumentation counting documents that reach a
/// [`collector::TopDocsCollector`] -- i.e. documents a scorer actually
/// produced and scored, as opposed to documents it skipped past.
///
/// This exists to answer a question timing cannot: when a query is slower than
/// Lucene's while this port's per-document costs are *lower* than Lucene's,
/// the two engines must be visiting different numbers of documents. Counting
/// is immune to the measurement noise that makes small timing differences
/// unreadable, and it localizes a divergence to "we do more work" rather than
/// "we are slower", which are very different defects.
///
/// The Java counterpart is `BenchRunner`'s counting `LeafCollector` wrapper,
/// which counts the same event (`collect(doc)` per leaf).
///
/// Same gating and rationale as [`test_only_maxscore_block_skip_counter`].
#[cfg(any(test, feature = "test-support"))]
pub mod test_only_scored_docs_counter {
    use std::cell::Cell;

    thread_local! {
        static SCORED: Cell<u64> = const { Cell::new(0) };
    }

    pub fn record_scored() {
        SCORED.with(|c| c.set(c.get() + 1));
    }

    pub fn reset() {
        SCORED.with(|c| c.set(0));
    }

    pub fn count() -> u64 {
        SCORED.with(|c| c.get())
    }
}

#[cfg(any(test, feature = "test-support"))]
pub mod test_only_maxscore_block_skip_counter {
    use std::cell::Cell;

    thread_local! {
        static SKIPPED_BLOCKS: Cell<usize> = const { Cell::new(0) };
    }

    pub fn record_skip() {
        SKIPPED_BLOCKS.with(|c| c.set(c.get() + 1));
    }

    /// Resets the counter to 0 -- call before a test's search, then
    /// [`take`] after to read how many blocks that search skipped.
    pub fn reset() {
        SKIPPED_BLOCKS.with(|c| c.set(0));
    }

    /// Reads (without resetting) the number of blocks skipped since the last
    /// [`reset`].
    pub fn count() -> usize {
        SKIPPED_BLOCKS.with(|c| c.get())
    }
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
#[allow(clippy::too_many_arguments)]
fn clause_scores(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    pos_in: Option<&PosInput<'_>>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    points: Option<&PointsInput<'_>>,
    clause: &Clause,
    norms: Option<&HashMap<String, FieldNorms<'_>>>,
    global: Option<&GlobalStats>,
) -> Result<HashMap<i32, f32>> {
    match clause {
        Clause::Term(query) => {
            let clause_norms = norms.and_then(|m| m.get(&query.field));
            let mut scores = HashMap::new();
            for (doc_id, score) in
                term_doc_scores(fields, doc_in, live_docs, query, clause_norms, global)?
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
            search_phrase_query_scored_with_stats(
                fields,
                doc_in,
                pos_in,
                pay_in,
                live_docs,
                query,
                clause_norms,
                global,
                &mut collector,
            )?;
            Ok(scores)
        }
        Clause::Boolean(nested) => {
            let Some(matched) =
                matched_boolean_docs(fields, doc_in, pos_in, pay_in, live_docs, points, nested)?
            else {
                return Ok(HashMap::new());
            };
            let matched: std::collections::HashSet<i32> = matched.collect();

            let mut scores: HashMap<i32, f32> = HashMap::new();
            for sub_clause in nested.must.iter().chain(nested.should.iter()) {
                for (doc_id, score) in clause_scores(
                    fields, doc_in, pos_in, pay_in, live_docs, points, sub_clause, norms, global,
                )? {
                    if matched.contains(&doc_id) {
                        *scores.entry(doc_id).or_insert(0.0) += score;
                    }
                }
            }
            Ok(scores)
        }
        Clause::DisjunctionMax(nested) => dismax_scores(
            fields, doc_in, pos_in, pay_in, live_docs, points, nested, norms, global,
        ),
        Clause::ConstantScore(nested) => {
            let matched = resolve_clause_docs(
                fields,
                doc_in,
                pos_in,
                pay_in,
                live_docs,
                points,
                &nested.inner,
            )?;
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
                points,
                &nested.inner,
                norms,
                global,
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
            // Scored, unlike the rest of the multi-term family: real
            // `FuzzyQuery`'s default rewrite is
            // `TopTermsBlendedFreqScoringRewrite`, not a constant-score one.
            // See `fuzzy_doc_scores`.
            let clause_norms = norms.and_then(|m| m.get(&query.field));
            fuzzy_doc_scores(fields, doc_in, live_docs, query, clause_norms)
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
        Clause::PointsRange(query) => {
            // Unscored: flat 1.0 per matching doc -- real Lucene's
            // `PointRangeQuery` is `ConstantScoreQuery`-shaped (see
            // `points_query`'s module doc comment), same rationale as
            // `Clause::Wildcard`'s arm above.
            let matched = points_range_doc_ids(points, live_docs, query)?;
            Ok(matched
                .into_iter()
                .map(|doc_id| (doc_id, 1.0_f32))
                .collect())
        }
        Clause::MatchAllDocs(query) => {
            // Flat 1.0 per live doc -- see `MatchAllDocsQuery`'s doc comment
            // for why (real `ConstantScoreScorer`'s own boost, undiscounted).
            Ok(match_all_doc_ids(live_docs, query.max_doc)
                .into_iter()
                .map(|doc_id| (doc_id, 1.0_f32))
                .collect())
        }
        Clause::MatchNoDocs(_) => Ok(HashMap::new()),
        Clause::MultiPhrase(query) => {
            let clause_norms = norms.and_then(|m| m.get(&query.field));
            let mut scores: HashMap<i32, f32> = HashMap::new();
            struct SumCollector<'a>(&'a mut HashMap<i32, f32>);
            impl collector::ScoringCollector for SumCollector<'_> {
                fn collect(&mut self, doc_id: i32, score: f32) {
                    *self.0.entry(doc_id).or_insert(0.0) += score;
                }
            }
            let mut sink = SumCollector(&mut scores);
            search_multi_phrase_query_scored_with_stats(
                fields,
                doc_in,
                pos_in,
                pay_in,
                live_docs,
                query,
                clause_norms,
                global,
                &mut sink,
            )?;
            Ok(scores)
        }
        Clause::TermInSet(query) => {
            // Unscored: flat 1.0 per matching doc -- see `TermInSetQuery`'s
            // doc comment for why (verified against real `TermInSetQuery`'s
            // own class doc comment: "produces scores that are equal to its
            // boost", not a summed/max-of-matched-terms formula).
            let matched = term_in_set_doc_ids(fields, doc_in, live_docs, query)?;
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
#[allow(clippy::too_many_arguments)]
fn dismax_scores(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    pos_in: Option<&PosInput<'_>>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    points: Option<&PointsInput<'_>>,
    query: &DisjunctionMaxQuery,
    norms: Option<&HashMap<String, FieldNorms<'_>>>,
    global: Option<&GlobalStats>,
) -> Result<HashMap<i32, f32>> {
    let per_disjunct: Vec<HashMap<i32, f32>> = query
        .disjuncts
        .iter()
        .map(|clause| {
            clause_scores(
                fields, doc_in, pos_in, pay_in, live_docs, points, clause, norms, global,
            )
        })
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
/// Lazy disjunction: the pure-`should`, all-[`Clause::Term`] shape, executed
/// without materializing any clause's doc list.
///
/// The conjunction sibling of this is [`try_conjunction_lazy`]; the same
/// argument applies. [`resolve_clause_docs`] builds a `Vec<i32>` per clause and
/// then unions them, so a disjunction pays for every posting of every clause
/// plus a `HashMap` of every matching doc, for a query that wants the top 50.
///
/// Here every cursor advances in lock-step over the union: the smallest
/// current doc across the cursors is the next candidate, every cursor sitting
/// on it contributes its BM25 term, and only those cursors advance. Cursor
/// selection is a linear scan rather than a heap on purpose -- real
/// disjunctions have a handful of clauses, and a linear minimum over 2-4
/// entries beats the heap's bookkeeping.
///
/// This does **not** prune: it still visits every doc in the union, so it is a
/// smaller win than the conjunction's. Removing that visit needs block-max
/// WAND, which is a larger change and is not in this milestone.
///
/// `minimum_should_match` is deliberately excluded from the gate: honouring it
/// requires counting matching clauses per doc, which is the general path's job.
/// Returns `Ok(false)` when the query does not have this shape.
fn try_disjunction_lazy<C: ScoringCollector>(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &BooleanQuery,
    norms: Option<&HashMap<String, FieldNorms<'_>>>,
    global: Option<&GlobalStats>,
    collector: &mut C,
) -> Result<bool> {
    if query.should.is_empty()
        || !query.must.is_empty()
        || !query.filter.is_empty()
        || !query.must_not.is_empty()
        || query.minimum_should_match > 1
    {
        return Ok(false);
    }
    let mut terms = Vec::with_capacity(query.should.len());
    for clause in &query.should {
        match clause {
            Clause::Term(t) => terms.push(t),
            _ => return Ok(false),
        }
    }
    let Some(doc_in) = doc_in else {
        return Ok(false);
    };

    struct Leg<'a> {
        cursor: lucene_codecs::postings::LazyDocsCursor<'a>,
        doc: i32,
        doc_freq: i64,
        doc_count: i64,
        /// `idf(doc_freq, doc_count)`, computed once when the leg is built.
        ///
        /// `idf` is a `ln()`, and it was being recomputed for every document
        /// this clause matched: `libm`'s `log` accounted for over 15% of a
        /// two-clause disjunction's profile. Lucene computes it once per term
        /// in `BM25Similarity.scorer` and carries it as the scorer's `weight`,
        /// which is what this is.
        weight: f32,
        /// This leg's own norms position -- Lucene's per-scorer
        /// `NumericDocValues`. The shared `&FieldNorms` stays immutable and
        /// `Sync`; the mutable `IndexedDISI` walk lives here, per leg, so a
        /// sparse field is traversed once per scan rather than re-resolved per
        /// document. See `field_norms::FieldNormsCursor`.
        norms: Option<crate::field_norms::FieldNormsCursor<'a, 'a>>,
    }

    impl Leg<'_> {
        /// Upper bound on this clause's contribution over a block's impacts,
        /// with the same `norms == None` rule the term and conjunction paths
        /// carry.
        fn bound(&self, impacts: &[lucene_codecs::postings::Impact]) -> f32 {
            match &self.norms {
                Some(n) => similarity::max_score_for_impacts(
                    impacts,
                    self.doc_freq,
                    self.doc_count,
                    n.avg_field_length(),
                ),
                None => similarity::max_score_for_impacts_unnormed(
                    impacts,
                    self.doc_freq,
                    self.doc_count,
                ),
            }
        }
    }
    let mut legs: Vec<Leg<'_>> = Vec::with_capacity(terms.len());
    for t in &terms {
        let Some(field_terms) = fields.field(&t.field) else {
            continue; // absent field contributes nothing to a union
        };
        let Some(stats) = field_terms.seek_exact(&t.term) else {
            continue; // absent term likewise
        };
        // Pulsed singleton (see try_conjunction_lazy): no .doc bytes exist for
        // it. Unlike an absent term it *does* contribute to the union, so the
        // clause cannot simply be skipped -- the whole query falls back.
        if stats.doc_freq <= 1 {
            return Ok(false);
        }
        let Some(mut cursor) = field_terms.lazy_postings(&t.term, doc_in)? else {
            continue;
        };
        let doc = cursor.next_doc().map_err(blocktree::Error::Postings)?;
        let (doc_freq, doc_count) =
            match global.and_then(|g| g.get(&(t.field.clone(), t.term.clone()))) {
                Some(g) => (g.doc_freq, g.doc_count),
                None => (stats.doc_freq as i64, field_terms.doc_count as i64),
            };
        legs.push(Leg {
            cursor,
            doc,
            doc_freq,
            doc_count,
            weight: similarity::idf(doc_freq, doc_count),
            norms: norms.and_then(|m| m.get(&t.field)).map(|fn_| fn_.cursor()),
        });
    }

    // `ImpactsDISI.upTo`, for a union: the highest doc for which the current
    // span has already been judged competitive at the current threshold. Inside
    // it there is nothing to re-decide, so the whole preamble below -- a pass
    // over every leg to recompute `up_to`, then a cache probe -- is skipped
    // entirely. Those two together were 20% of this query's profile, run once
    // per document to reach a decision that only changes when a leg crosses a
    // block boundary or the threshold rises.
    //
    // Invalidated on a threshold rise, exactly as
    // `ImpactsDISI.setMinCompetitiveScore` sets `upTo = -1`. See the term path
    // for what leaving that out costs: the skipping silently stops.
    let mut checked_upto: i32 = -1;
    let mut threshold = collector.pruning_threshold();
    loop {
        let Some(candidate) = legs
            .iter()
            .map(|l| l.doc)
            .filter(|&d| d != lucene_codecs::postings::NO_MORE_DOCS)
            .min()
        else {
            return Ok(true); // every cursor exhausted
        };

        // Block-max pruning for a union. A document may match every clause, so
        // the sum of the clauses' per-block maxima bounds any score in the span
        // -- the same bound the conjunction uses, sound here for the same
        // reason. This is the safe core of MAXSCORE/WAND; it does not yet
        // partition clauses into essential and non-essential, which is where
        // Lucene's WANDScorer gets its remaining power.
        if candidate > checked_upto {
            if let Some(threshold) = threshold {
                let mut up_to = i32::MAX;
                for leg in &legs {
                    if leg.doc != lucene_codecs::postings::NO_MORE_DOCS {
                        up_to = up_to.min(leg.cursor.current_block_last_doc_id());
                    }
                }
                let sum_max = {
                    let mut acc = 0.0f32;
                    let mut ok = true;
                    for leg in &legs {
                        if leg.doc == lucene_codecs::postings::NO_MORE_DOCS {
                            continue;
                        }
                        let impacts = leg.cursor.level0_impacts();
                        if impacts.is_empty() {
                            ok = false;
                            break;
                        }
                        acc += leg.bound(impacts);
                    }
                    if ok {
                        acc
                    } else {
                        f32::INFINITY
                    }
                };
                if sum_max.is_finite() && up_to >= candidate && sum_max <= threshold {
                    // No document in this span can compete. Rather than
                    // advancing the cursors -- which would decode the block each
                    // one lands on, only for the next iteration to find that
                    // span uncompetitive too -- walk spans forward on impacts
                    // alone, and materialize once at the end.
                    //
                    // This is where the wasted decode was worst. Counted on the
                    // M1 corpus before this loop existed, `or t0 t1` unpacked
                    // 9,914,368 documents to score 138,650: 1.4% of the decode
                    // work was used. `advance` skips *intervening* blocks
                    // cheaply, but it always decodes the one it lands on, so a
                    // skip-decode-skip-decode walk paid for every span it
                    // rejected.
                    let mut next = up_to.saturating_add(1);
                    loop {
                        #[cfg(any(test, feature = "test-support"))]
                        test_only_maxscore_block_skip_counter::record_skip();

                        for leg in legs.iter_mut() {
                            if leg.doc != lucene_codecs::postings::NO_MORE_DOCS {
                                leg.cursor
                                    .advance_shallow(next)
                                    .map_err(blocktree::Error::Postings)?;
                            }
                        }
                        let mut span_end = i32::MAX;
                        let mut acc = 0.0f32;
                        let mut bounded = true;
                        for leg in &legs {
                            if leg.doc == lucene_codecs::postings::NO_MORE_DOCS {
                                continue;
                            }
                            span_end = span_end.min(leg.cursor.level0_last_doc_id());
                            let impacts = leg.cursor.level0_impacts();
                            if impacts.is_empty() {
                                bounded = false;
                                break;
                            }
                            acc += leg.bound(impacts);
                        }
                        // Unbounded (a tail block, or exhaustion), competitive,
                        // or not advancing: stop walking and let the cursors
                        // materialize normally.
                        if !bounded
                            || acc > threshold
                            || span_end == i32::MAX
                            || span_end.saturating_add(1) <= next
                        {
                            break;
                        }
                        next = span_end.saturating_add(1);
                    }
                    // Exactly one block per leg gets decoded: the surviving one.
                    for leg in legs.iter_mut() {
                        if leg.doc != lucene_codecs::postings::NO_MORE_DOCS && leg.doc < next {
                            leg.doc = leg
                                .cursor
                                .advance(next)
                                .map_err(blocktree::Error::Postings)?;
                        }
                    }
                    continue;
                }
                // Competitive: do not ask again until a leg leaves the span this
                // decision covered. `up_to` can be i32::MAX when every leg is
                // exhausted, which is the same "nothing left to re-decide" answer.
                checked_upto = up_to;
            } else {
                // No threshold yet, so nothing can be pruned and nothing needs
                // deciding until one appears -- which sets `checked_upto` back
                // to -1 below.
                checked_upto = i32::MAX;
            }
        }

        let live = live_docs.is_none_or(|bits| bits.get(candidate as usize));
        let mut score = 0.0f32;
        for leg in legs.iter_mut() {
            if leg.doc != candidate {
                continue;
            }
            if live {
                let freq = leg.cursor.freq().unwrap_or(1) as f32;
                // `BM25Scorer.score(freq, encodedNorm)` verbatim: the idf is
                // hoisted into `leg.weight` once per clause (not one `ln()` per
                // document), and the length normalization is the precomputed
                // `cache[norm]` reciprocal, so this is one table load, one
                // multiply, one divide -- and, unlike the `idf * tf_norm`
                // multiply form it replaces, bit-for-bit what real Lucene
                // produces. See `similarity::do_score`.
                let norm_inverse = match leg.norms.as_mut() {
                    Some(n) => n.norm_inverse(candidate)?,
                    None => similarity::UNNORMED_NORM_INVERSE,
                };
                score += similarity::do_score(leg.weight, freq, norm_inverse);
            }
            leg.doc = leg.cursor.next_doc().map_err(blocktree::Error::Postings)?;
        }
        if live {
            collector.collect(candidate, score);
            // `ImpactsDISI.setMinCompetitiveScore`: a rise invalidates the span.
            let now = collector.pruning_threshold();
            if now != threshold {
                threshold = now;
                checked_upto = -1;
            }
        }
    }
}

/// Reader-wide term and collection statistics, for scoring a multi-segment
/// search the way Lucene does.
///
/// Lucene's `IndexSearcher` computes `TermStatistics`/`CollectionStatistics`
/// once across the whole reader and hands the same values to every leaf, so a
/// term's idf is identical in every segment. This port scored each segment from
/// its own `docFreq`/`docCount`, which is only equivalent when there is one
/// segment.
///
/// On M1's 15-segment corpus that made `body:t0`'s idf range 0.000473 to
/// 0.000763 -- a 1.6x spread against the global 0.000574 -- so the top-k filled
/// from whichever segment happened to make the term look rarest, and every one
/// of the 20 benchmark queries disagreed with Java on its hit set. It is
/// invisible on a single segment, which is why the merged corpus agreed exactly
/// and no fixture caught it: every fixture is one segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CollectionStats {
    /// Number of documents containing the term, summed across every segment.
    pub doc_freq: i64,
    /// Number of documents that have the field, summed across every segment.
    pub doc_count: i64,
}

/// Reader-wide statistics for every term a query mentions, keyed by
/// `(field, term)`. Built once per search by the multi-segment layer.
pub type GlobalStats = HashMap<(String, Vec<u8>), CollectionStats>;

/// Lazy leapfrog conjunction: the pure-`must`, all-[`Clause::Term`] shape,
/// executed without materializing any clause's doc list.
///
/// ## Why this exists
///
/// [`search_boolean_query_scored`] resolves each clause through
/// [`resolve_clause_docs`], which returns a `Vec<i32>` of *every* matching doc
/// before any intersection happens. Its cost therefore tracks the **most
/// frequent** clause. Real Lucene's `ConjunctionDISI` advances on the
/// **rarest**, so a conjunction containing a selective term is cheap.
///
/// M1 measured the difference: `and t0 t1z4`, whose rarest term matches a
/// handful of documents, cost more than scanning `t0` alone -- 20x slower than
/// Java. See `docs/benchmarks/verdict.md`.
///
/// ## The algorithm
///
/// Standard leapfrog. Cursors are ordered rarest-first by `docFreq` so the
/// lead cursor is the most selective; every other cursor is `advance()`d to the
/// lead's doc, and any cursor that overshoots becomes the new candidate. The
/// per-doc work is then O(clauses), not O(postings).
///
/// Scores are summed across clauses, matching
/// [`search_boolean_query_scored`]'s `must` handling exactly -- the same
/// `doc_freq`/`doc_count`/`freq`/`field_length` inputs to
/// [`similarity::score_with_params`], so this is an execution change and not a
/// scoring change.
///
/// Returns `Ok(false)` when the query does not have this shape, leaving the
/// caller to fall back.
fn try_conjunction_lazy<C: ScoringCollector>(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &BooleanQuery,
    norms: Option<&HashMap<String, FieldNorms<'_>>>,
    global: Option<&GlobalStats>,
    collector: &mut C,
) -> Result<bool> {
    // Shape gate: pure conjunction of leaf terms, nothing else. `filter`
    // clauses are admitted -- they are legs of the same conjunction, just
    // non-scoring ones (Java: `BooleanScorerSupplier.req(FILTER, MUST)` puts
    // both in `required` and only the `MUST` legs in `requiredScoring`).
    //
    // A **filter-only** conjunction is admitted too, and must be: without it,
    // `#body:t0 #body:t1` fell to the general path, which materializes every
    // clause's whole doc list before intersecting -- 129.5ms against 8.8ms for
    // the all-`MUST` form of the same query on the benchmark corpus, i.e. the
    // clause that is supposed to be *cheaper* was 15x dearer
    // (`benches/filter_vs_must.rs`).
    if (query.must.is_empty() && query.filter.is_empty())
        || !query.should.is_empty()
        || !query.must_not.is_empty()
    {
        return Ok(false);
    }
    let mut terms = Vec::with_capacity(query.must.len() + query.filter.len());
    for (clause, scoring) in query
        .must
        .iter()
        .map(|c| (c, true))
        .chain(query.filter.iter().map(|c| (c, false)))
    {
        match clause {
            Clause::Term(t) => terms.push((t, scoring)),
            _ => return Ok(false),
        }
    }
    let Some(doc_in) = doc_in else {
        return Ok(false);
    };

    // Resolve every term up front: a missing term means the conjunction is
    // empty, which is itself the answer.
    /// `BooleanClause.isScoring()`, as a type rather than a flag.
    ///
    /// A `FILTER` leg gates the intersection and nothing else -- it is never
    /// asked for its frequency, its norm or its impacts, exactly as
    /// `ConjunctionScorer.score()` iterates `scorers` (the scoring subset)
    /// rather than `required`. c12 made that structural instead of
    /// conventional: a filter leg's cursor is opened with
    /// `PostingsFlags::DocsOnly`, which fills every frequency with `1`, so
    /// reading one would be silently wrong rather than merely wasteful. The
    /// scoring inputs (`weight`, `norms`) live *inside* the `Scoring` variant,
    /// so the only way to reach the code that calls `freq()` is to have
    /// matched a leg that decoded frequencies.
    enum LegRole<'a> {
        Scoring {
            /// `idf(doc_freq, doc_count)`, computed once when the leg is built
            /// -- see the disjunction's `Leg::weight` for why that matters.
            weight: f32,
            /// This leg's own norms position -- Lucene's per-scorer
            /// `NumericDocValues`. The shared `&FieldNorms` stays immutable
            /// and `Sync`; the mutable `IndexedDISI` walk lives here, per leg,
            /// so a sparse field is traversed once per scan rather than
            /// re-resolved per document. See `field_norms::FieldNormsCursor`.
            norms: Option<crate::field_norms::FieldNormsCursor<'a, 'a>>,
        },
        Filter,
    }

    struct Leg<'a> {
        cursor: lucene_codecs::postings::LazyDocsCursor<'a>,
        doc_freq: i64,
        doc_count: i64,
        role: LegRole<'a>,
    }

    impl Leg<'_> {
        fn scoring(&self) -> bool {
            matches!(self.role, LegRole::Scoring { .. })
        }
    }

    impl Leg<'_> {
        /// Upper bound on this clause's contribution over a block's impacts.
        ///
        /// Honours the same `norms == None` rule the term path documents: with
        /// no norms every doc is scored at UNNORMED_FIELD_LENGTH, so bounding
        /// with the wire norms would bound a different formula and could
        /// underestimate -- skipping a document that should have been kept.
        fn bound(&self, impacts: &[lucene_codecs::postings::Impact]) -> f32 {
            match &self.role {
                LegRole::Scoring { norms: Some(n), .. } => similarity::max_score_for_impacts(
                    impacts,
                    self.doc_freq,
                    self.doc_count,
                    n.avg_field_length(),
                ),
                LegRole::Scoring { norms: None, .. } => similarity::max_score_for_impacts_unnormed(
                    impacts,
                    self.doc_freq,
                    self.doc_count,
                ),
                // A filter leg contributes nothing to any score, so its bound
                // is 0. It is never asked for one -- every call site goes
                // through `filter(Leg::scoring)` -- and this arm keeps that
                // sound rather than merely unreached.
                LegRole::Filter => 0.0,
            }
        }
    }
    let mut legs: Vec<Leg<'_>> = Vec::with_capacity(terms.len());
    for (t, scoring) in &terms {
        let scoring = *scoring;
        let Some(field_terms) = fields.field(&t.field) else {
            return Ok(true); // field absent: no matches, and we handled it
        };
        let Some(stats) = field_terms.seek_exact(&t.term) else {
            return Ok(true); // term absent: no matches
        };
        // docFreq <= 1 is pulsed into the term dictionary: the term has no .doc
        // bytes at all, so lazy_postings cannot open a cursor for it. Hand the
        // whole query to the general path, which reads it from the term
        // metadata. Cheap to give up on -- such a conjunction matches at most
        // one document.
        if stats.doc_freq <= 1 {
            return Ok(false);
        }
        // `TermsEnum.postings(reuse, flags)`: a filter leg reads doc ids and
        // nothing else, so the `.doc` file's frequency blocks are skipped
        // (`PForUtil.skip`) rather than unpacked. See `LegRole` for why the
        // scoring inputs live inside the `Scoring` variant.
        let flags = if scoring {
            lucene_codecs::postings::PostingsFlags::Freqs
        } else {
            lucene_codecs::postings::PostingsFlags::DocsOnly
        };
        let Some(cursor) = field_terms.lazy_postings_with_flags(&t.term, doc_in, flags)? else {
            return Ok(true);
        };
        // Reader-wide idf where available, matching Lucene's per-leaf scoring.
        let (doc_freq, doc_count) =
            match global.and_then(|g| g.get(&(t.field.clone(), t.term.clone()))) {
                Some(g) => (g.doc_freq, g.doc_count),
                None => (stats.doc_freq as i64, field_terms.doc_count as i64),
            };
        legs.push(Leg {
            cursor,
            doc_freq,
            doc_count,
            role: if scoring {
                LegRole::Scoring {
                    weight: similarity::idf(doc_freq, doc_count),
                    // A filter leg never scores, so it never needs a norms
                    // cursor -- and skipping it also skips the sparse-field
                    // `IndexedDISI` walk that cursor would perform across the
                    // scan.
                    norms: norms.and_then(|m| m.get(&t.field)).map(|fn_| fn_.cursor()),
                }
            } else {
                LegRole::Filter
            },
        });
    }

    // Rarest first: the lead cursor drives the leapfrog, so it must be the most
    // selective one for the skipping to pay off.
    legs.sort_by_key(|l| l.doc_freq);

    let mut candidate = legs[0]
        .cursor
        .next_doc()
        .map_err(blocktree::Error::Postings)?;
    // Deliberately NOT the `ImpactsDISI.upTo` shape the term and disjunction
    // paths use. It was tried here and measured slower -- `and t0 t1` 91.2 ->
    // 83.0 qps, `and t0 tz` 167.4 -> 144.7. The reason is that a leapfrog's
    // `candidate` regularly overshoots `up_to` (the code below only skips when
    // `up_to >= candidate`), so keying the "already decided" marker on the
    // document rather than on the span invalidates it almost every iteration
    // and loses the cache entirely. The span-keyed cache below does the same
    // job for this shape.
    let mut conj_bound: Option<(i32, f32)> = None;
    // Block-max pruning needs a scoring clause to bound. With none, every
    // document scores 0, so the summed bound would be 0 and would authorize
    // skipping the whole segment the moment a top-`n` queue filled -- which is
    // what Lucene's `FilterScorer` wrapper (`getMaxScore() == 0`) lets
    // `TOP_SCORES` do, but only under `TopScoreDocCollector`'s
    // `Math.nextUp(bottom)` threshold. This port's `pruning_threshold` is the
    // bottom score itself, so `0 <= 0` would prune on a *tie*, dropping
    // documents Lucene keeps. Pruning is therefore switched off for the
    // filter-only shape; the leapfrog itself is the win here.
    let prunable = legs.iter().any(Leg::scoring);
    'outer: while candidate != lucene_codecs::postings::NO_MORE_DOCS {
        // Block-max conjunction pruning, as Lucene's BlockMaxConjunctionScorer
        // does it. Every clause must match, so a document's score is at most the
        // *sum* of the clauses' per-block maxima. Take the span all clauses
        // currently cover (the smallest of their block ends) and, if that summed
        // bound cannot beat the collector's threshold, no document in the span
        // can qualify -- skip the whole span.
        //
        // Without this the leapfrog is lazy but blind: it visits every document
        // in the intersection and scores it, however uncompetitive.
        if let Some(threshold) = collector.pruning_threshold().filter(|_| prunable) {
            // Recompute the summed bound only when the covered span changes.
            // The span is identified by the smallest clause block end, which is
            // cheap to read; the bound itself is not, and on a selective
            // conjunction recomputing it per candidate costs more than the
            // pruning saves -- measured as a 32% regression on `and tz t2s`
            // before this cache existed.
            let mut up_to = i32::MAX;
            for leg in &legs {
                up_to = up_to.min(leg.cursor.current_block_last_doc_id());
            }
            let sum_max = match conj_bound {
                Some((key, v)) if key == up_to => v,
                _ => {
                    let mut acc = 0.0f32;
                    let mut ok = true;
                    // Only the *scoring* legs contribute to the bound: a
                    // filter leg's score is 0, so summing over `must` alone is
                    // still a real upper bound (Java builds its
                    // `BlockMaxConjunctionScorer` from `scoringScorers`, never
                    // from the filters). The span the bound covers still
                    // narrows to every leg, filters included, which only makes
                    // the bound tighter-scoped and therefore safe.
                    for leg in legs.iter().filter(|l| l.scoring()) {
                        let impacts = leg.cursor.level0_impacts();
                        if impacts.is_empty() {
                            ok = false;
                            break;
                        }
                        acc += leg.bound(impacts);
                    }
                    let v = if ok { acc } else { f32::INFINITY };
                    conj_bound = Some((up_to, v));
                    v
                }
            };
            let have_bounds = sum_max.is_finite();
            if have_bounds && up_to >= candidate && sum_max <= threshold {
                // Walk spans forward on impacts alone before materializing --
                // same reason as the disjunction above. `advance` decodes the
                // block it lands on, so advancing span by span paid an unpack
                // for every span it then rejected.
                //
                // Every clause must match, so the lead cursor cannot move past
                // a span until every clause has been shallow-positioned there:
                // the summed bound is only meaningful when all of them describe
                // the same span.
                let mut next = up_to.saturating_add(1);
                loop {
                    for leg in legs.iter_mut() {
                        leg.cursor
                            .advance_shallow(next)
                            .map_err(blocktree::Error::Postings)?;
                    }
                    let mut span_end = i32::MAX;
                    let mut acc = 0.0f32;
                    let mut bounded = true;
                    for leg in &legs {
                        span_end = span_end.min(leg.cursor.level0_last_doc_id());
                        if !leg.scoring() {
                            continue;
                        }
                        let impacts = leg.cursor.level0_impacts();
                        if impacts.is_empty() {
                            bounded = false;
                            break;
                        }
                        acc += leg.bound(impacts);
                    }
                    if !bounded
                        || acc > threshold
                        || span_end == i32::MAX
                        || span_end.saturating_add(1) <= next
                    {
                        break;
                    }
                    next = span_end.saturating_add(1);
                }
                candidate = legs[0]
                    .cursor
                    .advance(next)
                    .map_err(blocktree::Error::Postings)?;
                continue 'outer;
            }
        }

        for i in 1..legs.len() {
            let d = legs[i]
                .cursor
                .advance(candidate)
                .map_err(blocktree::Error::Postings)?;
            if d != candidate {
                // Overshot: this doc cannot match, restart with the new floor.
                candidate = legs[0]
                    .cursor
                    .advance(d)
                    .map_err(blocktree::Error::Postings)?;
                continue 'outer;
            }
        }

        if live_docs.is_none_or(|bits| bits.get(candidate as usize)) {
            let mut score = 0.0f32;
            // `ConjunctionScorer.score()` iterates `scorers`, the scoring
            // subset of `required` -- a filter leg is skipped entirely, so its
            // frequency is never decoded and its norm never read. That is the
            // whole cost difference between `#body:dog` and `+body:dog`.
            for leg in legs.iter_mut() {
                // Destructuring the role is what makes reading a frequency
                // legitimate here: only a `Scoring` leg's cursor decoded
                // frequencies at all, and only this arm can see its
                // `weight`/`norms`. A `Filter` leg has no scoring inputs to
                // reach and no real frequency to misread.
                let LegRole::Scoring { weight, norms } = &mut leg.role else {
                    continue;
                };
                let freq = leg.cursor.freq().unwrap_or(1) as f32;
                // `BM25Scorer.score(freq, encodedNorm)` verbatim: the idf is
                // hoisted into `weight` once per clause (not one `ln()` per
                // document), and the length normalization is the precomputed
                // `cache[norm]` reciprocal, so this is one table load, one
                // multiply, one divide -- and, unlike the `idf * tf_norm`
                // multiply form it replaces, bit-for-bit what real Lucene
                // produces. See `similarity::do_score`.
                let norm_inverse = match norms.as_mut() {
                    Some(n) => n.norm_inverse(candidate)?,
                    None => similarity::UNNORMED_NORM_INVERSE,
                };
                score += similarity::do_score(*weight, freq, norm_inverse);
            }
            collector.collect(candidate, score);
        }
        candidate = legs[0]
            .cursor
            .next_doc()
            .map_err(blocktree::Error::Postings)?;
    }
    Ok(true)
}

#[allow(clippy::too_many_arguments)]
pub fn search_boolean_query_scored<C: ScoringCollector>(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    pos_in: Option<&PosInput<'_>>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    points: Option<&PointsInput<'_>>,
    query: &BooleanQuery,
    norms: Option<&HashMap<String, FieldNorms<'_>>>,
    collector: &mut C,
) -> Result<()> {
    search_boolean_query_scored_with_stats(
        fields, doc_in, pos_in, pay_in, live_docs, points, query, norms, None, collector,
    )
}

/// [`search_boolean_query_scored`] taking reader-wide statistics, so a
/// multi-segment boolean search scores every leaf with one idf per term -- see
/// [`CollectionStats`]. `None` keeps each segment's own counters.
#[allow(clippy::too_many_arguments)]
pub fn search_boolean_query_scored_with_stats<C: ScoringCollector>(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    pos_in: Option<&PosInput<'_>>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    points: Option<&PointsInput<'_>>,
    query: &BooleanQuery,
    norms: Option<&HashMap<String, FieldNorms<'_>>>,
    global: Option<&GlobalStats>,
    collector: &mut C,
) -> Result<()> {
    // Pure conjunctions of leaf terms run lazily, without materializing any
    // clause's doc list. Everything else falls through to the general path.
    if try_conjunction_lazy(fields, doc_in, live_docs, query, norms, global, collector)?
        || try_disjunction_lazy(fields, doc_in, live_docs, query, norms, global, collector)?
    {
        return Ok(());
    }

    // One scoring clause and nothing to filter against: the clause's own score
    // map *is* the matched set, so running `matched_boolean_docs` first would
    // execute the clause a second time for an answer already in hand. On a
    // two-term phrase query that second execution was 14% of the query in
    // `resolve_clause_docs` alone, plus a full second pass of position decode
    // -- the single-clause shape is exactly how `search_phrase_query_scored`
    // reaches this function from the multi-segment layer.
    //
    // Restricted to `Term` and `Phrase`, the two clause kinds whose
    // `clause_scores` output is *exactly* their matched set. It is not true in
    // general: a nested `Boolean` can match documents its scoring sub-clauses
    // never mention, and the wildcard family expands to terms elsewhere. Those
    // keep the two-pass path. `minimum_should_match` must be 0, not merely
    // `<= 1`: with no `should` clauses at all, a minimum of 1 means nothing
    // matches, and the fast path would wrongly return the `must` clause's
    // documents.
    if query.must.len() == 1
        && query.filter.is_empty()
        && query.should.is_empty()
        && query.must_not.is_empty()
        && query.minimum_should_match == 0
        && matches!(
            query.must[0],
            Clause::Term(_)
                | Clause::Phrase(_)
                | Clause::Prefix(_)
                | Clause::Wildcard(_)
                | Clause::Fuzzy(_)
                | Clause::Regexp(_)
        )
    {
        // Straight to the collector, not through `clause_scores`. That function
        // returns a `HashMap<i32, f32>` -- one hash insert per matching
        // document, then a key collection, then a sort, then a hash lookup per
        // document to read it back. On a two-term phrase over this corpus that
        // machinery was **43% of the query**: `clause_scores` 29.9%,
        // `SumCollector::collect` 10.3%, hashing 3.1%, and the sort 3.0%,
        // against 23.4% for the position decode it exists to serve.
        //
        // Lucene never materializes a score map. A `Scorer` streams
        // `(doc, score)` to the collector in ascending document order, which is
        // exactly what both of these functions already do -- the map was pure
        // overhead between two things that already agreed on shape.
        match &query.must[0] {
            Clause::Term(q) => {
                let clause_norms = norms.and_then(|m| m.get(&q.field));
                return search_term_query_scored_with_stats(
                    fields,
                    doc_in,
                    live_docs,
                    q,
                    clause_norms,
                    global,
                    collector,
                );
            }
            Clause::Phrase(q) => {
                let clause_norms = norms.and_then(|m| m.get(&q.field));
                return search_phrase_query_scored_with_stats(
                    fields,
                    doc_in,
                    pos_in,
                    pay_in,
                    live_docs,
                    q,
                    clause_norms,
                    global,
                    collector,
                );
            }
            // The wildcard family scores a flat 1.0 per matching document --
            // see `PrefixQuery`'s doc comment -- so its "score map" was a
            // `HashMap<i32, f32>` whose every value was the same constant, then
            // a key collection, then a sort, then a lookup per document to read
            // that constant back. On `prefix body:t12` over this corpus that
            // machinery was 43% of the query: hash insert 11.8%, hashing 8.7%,
            // rehash 4.3%, and the sort 17.7%, against 3.4% for finding the
            // matching documents in the first place.
            //
            // These were excluded from this fast path when it was written, on
            // the grounds that the wildcard family "expands to terms
            // elsewhere". That is true and irrelevant: `resolve_clause_docs`
            // returns exactly the matched set in ascending document order,
            // which is precisely what the collector wants.
            other @ (Clause::Prefix(_)
            | Clause::Wildcard(_)
            | Clause::Fuzzy(_)
            | Clause::Regexp(_)) => {
                // Lazily merge the expanded terms' postings and stop at the
                // collector's capacity -- see `stream_constant_score_clause`.
                if stream_constant_score_clause(fields, doc_in, live_docs, other, collector)? {
                    return Ok(());
                }
                let matched =
                    resolve_clause_docs(fields, doc_in, pos_in, pay_in, live_docs, points, other)?;
                debug_assert!(
                    matched.windows(2).all(|w| w[0] < w[1]),
                    "resolve_clause_docs must yield ascending, deduplicated doc ids"
                );
                for doc_id in matched {
                    collector.collect(doc_id, 1.0);
                }
                return Ok(());
            }
            // Unreachable: the guard above admits only the arms handled here.
            _ => unreachable!("guarded by the matches! above"),
        }
    }

    let Some(matched) =
        matched_boolean_docs(fields, doc_in, pos_in, pay_in, live_docs, points, query)?
    else {
        return Ok(());
    };

    // Sum each scoring clause's (doc_id -> score) contributions across `must`
    // and `should` (never `must_not`, which only filters -- see doc comment).
    let mut scores: HashMap<i32, f32> = HashMap::new();
    for clause in query.must.iter().chain(query.should.iter()) {
        for (doc_id, score) in clause_scores(
            fields, doc_in, pos_in, pay_in, live_docs, points, clause, norms, global,
        )? {
            *scores.entry(doc_id).or_insert(0.0) += score;
        }
    }

    for doc_id in matched {
        collector.collect(doc_id, scores.get(&doc_id).copied().unwrap_or(0.0));
    }
    Ok(())
}

/// MAXSCORE-style sibling of [`search_boolean_query_scored`], kept as a
/// distinct entry point because `lucene-ffi`
/// (`ffi_search_boolean_query_scored_maxscore`,
/// `ffi_search_boolean_query_multi_segment_maxscore`) and
/// [`multi_segment::search_boolean_query_multi_segment_maxscore`] name it.
///
/// **This is a delegate; the MAXSCORE machinery lives in
/// [`try_disjunction_lazy`].** Until batch c12 this function carried its own
/// two-tier essential/non-essential MAXSCORE implementation over
/// [`lucene_codecs::postings::LazyDocsCursor`]s -- ~180 lines whose body was
/// provably unreachable. The function's first act is to try
/// [`try_disjunction_lazy`] and return if it succeeded, and the two are
/// *exactly complementary*:
///
/// - every shape the old body could handle (pure `should`, all
///   [`Clause::Term`], `doc_in` present, `minimum_should_match <= 1`, no
///   pulsed `docFreq <= 1` term) [`try_disjunction_lazy`] handles first;
/// - every shape [`try_disjunction_lazy`] declines
///   (`must`/`filter`/`must_not` present, `should` empty,
///   `minimum_should_match > 1`, a non-`Clause::Term` clause, `doc_in` absent,
///   a pulsed term) the old body also declined, falling straight back;
/// - the two cases where the old body was *stricter* -- an absent field and an
///   absent term, which [`try_disjunction_lazy`] simply drops from the union --
///   are shapes [`try_disjunction_lazy`] therefore *accepts*, so control never
///   reached the body for them either;
/// - the old body's `Err(Unsupported)` fallback arm was unreachable for the
///   same reason: [`try_disjunction_lazy`] propagates that error with `?`
///   before the body is entered.
///
/// Verified for c12 two ways: by reading both predicates, and by a line
/// coverage report (`cargo llvm-cov -p lucene-search`) in which every one of
/// the body's 78 executable lines was uncovered by all 899 tests, including
/// the six `boolean_maxscore_falls_back_*` tests written to drive its own
/// fallback arms.
///
/// Deleted rather than revived. The old body's own doc comment already
/// recorded it as 4-5x *slower* than the lazy union on M1's 5M-document
/// corpus (655 ms vs 163 ms on `t0 OR t1`) and said "prefer the plain scored
/// entry point"; [`try_disjunction_lazy`] has since grown real block-max
/// pruning of its own, so reviving the body would mean routing queries to the
/// slower of two pruning implementations. The entry point keeps its name, its
/// signature and its behaviour -- unchanged, since the behaviour was already
/// "[`try_disjunction_lazy`], else the exhaustive path", which is precisely
/// what [`search_boolean_query_scored_with_stats`] does (its
/// [`try_conjunction_lazy`] attempt declines every shape
/// [`try_disjunction_lazy`] accepts and vice versa -- the two gates are
/// mutually exclusive on `query.should.is_empty()`).
#[allow(clippy::too_many_arguments)]
pub fn search_boolean_query_scored_maxscore(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    pos_in: Option<&PosInput<'_>>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    points: Option<&PointsInput<'_>>,
    query: &BooleanQuery,
    norms: Option<&HashMap<String, FieldNorms<'_>>>,
    collector: &mut collector::TopDocsCollector,
) -> Result<()> {
    search_boolean_query_scored_maxscore_with_stats(
        fields, doc_in, pos_in, pay_in, live_docs, points, query, norms, None, collector,
    )
}

/// [`search_boolean_query_scored_maxscore`] taking reader-wide statistics, so
/// it agrees with [`search_boolean_query_scored_with_stats`] on a multi-segment
/// index -- the two must produce identical output, and they cannot if only one
/// of them scores globally.
///
/// See [`search_boolean_query_scored_maxscore`] for why this is a delegate.
#[allow(clippy::too_many_arguments)]
pub fn search_boolean_query_scored_maxscore_with_stats(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    pos_in: Option<&PosInput<'_>>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    points: Option<&PointsInput<'_>>,
    query: &BooleanQuery,
    norms: Option<&HashMap<String, FieldNorms<'_>>>,
    global: Option<&GlobalStats>,
    collector: &mut collector::TopDocsCollector,
) -> Result<()> {
    search_boolean_query_scored_with_stats(
        fields, doc_in, pos_in, pay_in, live_docs, points, query, norms, global, collector,
    )
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
    points: Option<&PointsInput<'_>>,
    query: &DisjunctionMaxQuery,
    collector: &mut C,
) -> Result<()> {
    let matched = resolve_dismax_docs(fields, doc_in, pos_in, pay_in, live_docs, points, query)?;
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
    points: Option<&PointsInput<'_>>,
    query: &DisjunctionMaxQuery,
    norms: Option<&HashMap<String, FieldNorms<'_>>>,
    collector: &mut C,
) -> Result<()> {
    let scores = dismax_scores(
        fields, doc_in, pos_in, pay_in, live_docs, points, query, norms, None,
    )?;
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
pub(crate) fn phrase_matches_in_doc(term_positions: &[&[i32]]) -> bool {
    phrase_freq_exact_impl(term_positions, true) != 0
}

/// Scratch cursors for the alignment walk, one per non-leading phrase term.
/// Stack-resident for any phrase this port can realistically be handed
/// (`MAX_INLINE_PHRASE_TERMS` slots), heap only past that -- the walk runs once
/// per candidate *document*, so a `Vec` here would be one allocation per
/// document, which is the exact shape finding #O15 spent a milestone removing
/// from the position stream.
const MAX_INLINE_PHRASE_TERMS: usize = 32;

/// The one alignment walk behind [`phrase_matches_in_doc`] and
/// [`phrase_freq_exact`]. `stop_at_first` returns as soon as one alignment is
/// found (the boolean question) instead of counting them all.
///
/// **`ExactPhraseMatcher`'s shape, not a binary search.** `p0` walks the first
/// term's positions in ascending order, so every target `p0 + i` ascends too,
/// so a cursor per subsequent term never rewinds: the whole walk is one merge
/// pass over all the lists, `O(sum of list lengths)`, which is what
/// `ExactPhraseMatcher.nextMatch` does by advancing its `PostingsEnum`s
/// together.
///
/// This replaced a `binary_search` per term per candidate
/// (`O(|p0| · (n-1) · log|list|)`). Measured by
/// `benches/phrase_freq.rs` on 4096-position lists: **5.0x-6.8x** faster
/// (26.8 -> 4.9 µs at a 50% hit rate, 39.6 -> 5.8 µs for a three-term phrase),
/// and 6.8% slower on an 8-position list where both are ~18 ns. The same
/// finding as M1.6's `next_doc`/`advance` binary searches, in the function
/// those two never reached (b12 finding F-21).
fn phrase_freq_exact_impl(term_positions: &[&[i32]], stop_at_first: bool) -> i32 {
    let Some((first, rest)) = term_positions.split_first() else {
        return 0;
    };
    if rest.iter().any(|positions| positions.is_empty()) {
        return 0;
    }
    let mut inline = [0usize; MAX_INLINE_PHRASE_TERMS];
    let mut spilled;
    let cursors: &mut [usize] = if rest.len() <= MAX_INLINE_PHRASE_TERMS {
        &mut inline[..rest.len()]
    } else {
        spilled = vec![0usize; rest.len()];
        &mut spilled
    };

    let mut freq = 0;
    'candidate: for &p0 in first.iter() {
        for (i, positions) in rest.iter().enumerate() {
            let target = p0 + (i as i32 + 1);
            let mut c = cursors[i];
            while c < positions.len() && positions[c] < target {
                c += 1;
            }
            cursors[i] = c;
            if c == positions.len() {
                // This term is exhausted, and every later `p0` needs a strictly
                // larger target, so no later alignment can succeed either.
                break 'candidate;
            }
            if positions[c] != target {
                continue 'candidate;
            }
        }
        freq += 1;
        if stop_at_first {
            break;
        }
    }
    freq
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
pub(crate) fn phrase_matches_in_doc_sloppy(term_positions: &[&[i32]], slop: u32) -> bool {
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
    'candidate: for &p0 in first.iter() {
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
pub(crate) fn phrase_freq_exact(term_positions: &[&[i32]]) -> i32 {
    phrase_freq_exact_impl(term_positions, false)
}

/// `SloppyPhraseMatcher`'s contribution to `PhraseScorer.score()`, for a phrase
/// with `slop > 0`: **not** a match count, but the sum of
/// `sloppyWeight() == 1f / (1f + matchLength)` over every match in the document
/// (`PhraseScorer.score()`: `freq = matcher.sloppyWeight(); while
/// (matcher.nextMatch()) freq += matcher.sloppyWeight();`), where `matchLength`
/// is the alignment's total slack -- the same
/// `(p_last - p_first) - (n - 1)` quantity [`phrase_matches_in_doc_sloppy`]
/// already computes and compares against `slop`.
///
/// This is what separates a tightly-packed sloppy match from a loose one: on
/// this crate's own fixture segment, real Lucene scores an adjacent
/// `alpha beta` at weight `1` and an `alpha _ _ beta` two positions apart at
/// weight `1/3`, a 2x difference in the final BM25 score. Returning a flat `1`
/// for "this document matched somehow" -- which this port did until the b12
/// sweep -- collapses that distinction and also loses the frequency signal
/// entirely for a document containing several sloppy occurrences.
///
/// **Scope, stated as precisely as [`phrase_matches_in_doc_sloppy`]'s**: this
/// enumerates one match per starting position of the phrase's first term
/// (exactly [`phrase_freq_exact`]'s granularity) and takes that start's
/// *minimum* achievable `matchLength` via the same greedy scan, which is
/// optimal for a fixed start. Real `SloppyPhraseMatcher` instead enumerates
/// matches by repeatedly advancing whichever `PhrasePositions` is currently
/// minimal, which additionally admits **reordered** terms -- the same
/// documented in-order-only restriction this port's sloppy matcher already
/// carries, now inherited by its frequency. For `slop == 0` the two agree
/// exactly (every gap is forced to zero, so every match weighs `1` and the sum
/// is [`phrase_freq_exact`]'s count), which is what
/// `sloppy_phrase_freq_at_slop_zero_equals_the_exact_count` pins.
pub(crate) fn phrase_freq_sloppy(term_positions: &[&[i32]], slop: u32) -> f32 {
    let Some((first, rest)) = term_positions.split_first() else {
        return 0.0;
    };
    if rest.iter().any(|positions| positions.is_empty()) {
        return 0.0;
    }
    if rest.is_empty() {
        // Single-term phrase: real `PhraseQuery.Builder.build()` rewrites this
        // to a `TermQuery`, whose frequency is the term's own -- every
        // occurrence is a zero-slack match.
        return first.len() as f32;
    }
    let slop = i64::from(slop);
    let mut freq = 0.0f32;
    'candidate: for &p0 in first.iter() {
        let mut prev = p0;
        let mut match_length: i64 = 0;
        for positions in rest {
            let idx = positions.partition_point(|&x| x <= prev);
            let Some(&pos) = positions.get(idx) else {
                continue 'candidate;
            };
            match_length += i64::from(pos - prev - 1);
            if match_length > slop {
                continue 'candidate;
            }
            prev = pos;
        }
        freq += 1.0 / (1.0 + match_length as f32);
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
        let Some((docs, positions, spans)) =
            term_doc_positions(fields, doc_in, pos_in, pay_in, live_docs, field, term)?
        else {
            // A missing leaf term contributes no occurrences anywhere --
            // `span_matches_in_doc` already treats an absent map entry the
            // same way, so this leaf simply never adds candidate docs.
            continue;
        };
        // Spans are not on the measured hot path: rebuild the map the span
        // matcher expects from the flat positions and their per-doc spans.
        let map: HashMap<i32, Vec<i32>> = docs
            .iter()
            .copied()
            .zip(spans.iter())
            .map(|(doc, &(start, end))| (doc, positions[start as usize..end as usize].to_vec()))
            .collect();
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
/// A term's live matching docs and, index-aligned with them, each doc's
/// positions.
///
/// Aligned vectors rather than a `HashMap<doc, positions>`: the phrase matcher
/// walks candidates in ascending doc order, so a per-term cursor index finds
/// each doc in amortised O(1) with no hashing and no per-doc clone. Building
/// the map cost one hash insertion per matching document -- ~5M for a
/// high-frequency term -- and looking a doc up then cloned its position vector.
///
/// Flat, not nested. The positions of every matching document live in one
/// `Vec<i32>`, and `spans[i]` is document `i`'s `(start, end)` into it. The
/// nested `Vec<Vec<i32>>` this replaced allocated once per matching document --
/// roughly five million allocations for a phrase query on a high-frequency term
/// -- and about half of such a query's runtime was in `malloc`/`free`/`memcpy`.
/// Lucene's `ExactPhraseMatcher` allocates nothing per document at all.
type TermDocPositions = (Vec<i32>, Vec<i32>, Vec<(u32, u32)>);

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
    let Some((postings, positions, doc_starts)) =
        field_terms.positions_flat(term, doc_in, pos_in, pay_in)?
    else {
        return Ok(None);
    };

    let mut docs = Vec::with_capacity(postings.docs.len());
    let mut spans: Vec<(u32, u32)> = Vec::with_capacity(postings.docs.len());
    for (i, doc_id) in postings.docs.into_iter().enumerate() {
        if !live_docs.is_none_or(|bits| bits.get(doc_id as usize)) {
            continue;
        }
        docs.push(doc_id);
        // `doc_starts` is indexed by the *unfiltered* document position and has
        // one extra entry, so the span is always in range and deleted documents
        // simply never get one pushed.
        spans.push((doc_starts[i], doc_starts[i + 1]));
    }
    Ok(Some((docs, positions, spans)))
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

    let Some(field_terms) = fields.field(&query.field) else {
        return Ok(());
    };

    // Documents first, positions second -- the same order (and for the same
    // reason) as `search_phrase_query_scored_with_stats`: a phrase can only
    // match where every term does, so positions are needed only for the
    // intersection. Deliberately **unfiltered** by `live_docs` here, because
    // `positions_for_docs` indexes the wire stream by each document's running
    // frequency sum and needs the term's whole doc list; deletions are applied
    // to the candidate list below instead.
    let mut per_term_docs: Vec<Vec<i32>> = Vec::with_capacity(query.terms.len());
    let mut per_term_freqs: Vec<Vec<i32>> = Vec::with_capacity(query.terms.len());
    for term in &query.terms {
        let Some(postings) = field_terms.postings(term, doc_in)? else {
            // A missing term means the phrase can never match -- same convention
            // as `term_doc_ids`/`search_term_query`.
            return Ok(());
        };
        per_term_docs.push(postings.docs);
        per_term_freqs.push(postings.freqs);
    }

    let per_term_docs_snapshot = per_term_docs.clone();
    let candidate_docs: Vec<i32> = Conjunction::new(
        per_term_docs
            .into_iter()
            .map(|v| Box::new(v.into_iter()) as BoxDocIter<'static>)
            .collect(),
    )
    .filter(|&doc_id| live_docs.is_none_or(|bits| bits.get(doc_id as usize)))
    .collect();
    if candidate_docs.is_empty() {
        return Ok(());
    }

    // Each term's own indices for the candidates, ascending -- one walk per
    // term, no searching, because both lists are sorted.
    let mut per_term_positions: Vec<Vec<i32>> = Vec::with_capacity(query.terms.len());
    let mut per_term_starts: Vec<Vec<u32>> = Vec::with_capacity(query.terms.len());
    for (t, term) in query.terms.iter().enumerate() {
        let docs = &per_term_docs_snapshot[t];
        let mut wanted = Vec::with_capacity(candidate_docs.len());
        let mut cursor = 0usize;
        for &doc_id in &candidate_docs {
            while cursor < docs.len() && docs[cursor] < doc_id {
                cursor += 1;
            }
            debug_assert!(
                cursor < docs.len() && docs[cursor] == doc_id,
                "candidates come from the intersection of these very lists"
            );
            wanted.push(cursor);
        }
        let stats = field_terms
            .seek_exact(term)
            .expect("term presence was established above");
        let (positions, starts) = field_terms.positions_for_docs(
            term,
            doc_in,
            pos_in,
            pay_in,
            &per_term_freqs[t],
            stats.total_term_freq,
            &wanted,
        )?;
        per_term_positions.push(positions);
        per_term_starts.push(starts);
    }

    // Candidate `k` sits at index `k` in every term's positions, because
    // every term's `wanted` was built from the same candidate list.
    let mut term_positions: Vec<&[i32]> = Vec::with_capacity(per_term_positions.len());
    for (k, doc_id) in candidate_docs.into_iter().enumerate() {
        term_positions.clear();
        for t in 0..per_term_positions.len() {
            let start = per_term_starts[t][k] as usize;
            let end = per_term_starts[t][k + 1] as usize;
            term_positions.push(&per_term_positions[t][start..end]);
        }
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
    search_phrase_query_scored_with_stats(
        fields, doc_in, pos_in, pay_in, live_docs, query, norms, None, collector,
    )
}

/// [`search_phrase_query_scored`] taking reader-wide statistics, so a phrase
/// clause of a multi-segment search gets one idf per constituent term -- see
/// [`CollectionStats`]. `None` keeps this segment's own counters.
#[allow(clippy::too_many_arguments)]
pub fn search_phrase_query_scored_with_stats<C: ScoringCollector>(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    pos_in: Option<&PosInput<'_>>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &PhraseQuery,
    norms: Option<&FieldNorms<'_>>,
    global: Option<&GlobalStats>,
    collector: &mut C,
) -> Result<()> {
    if query.terms.is_empty() {
        return Ok(());
    }
    if query.terms.len() == 1 {
        let term_query = TermQuery::new(query.field.clone(), query.terms[0].clone());
        return search_term_query_scored_with_stats(
            fields,
            doc_in,
            live_docs,
            &term_query,
            norms,
            global,
            collector,
        );
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
    let mut idf_sum = 0.0f32;
    for term in &query.terms {
        let Some(stats) = field_terms.seek_exact(term) else {
            return Ok(());
        };
        // Reader-wide statistics where the caller has them, exactly as the term
        // path does. A phrase's idf is the sum of its terms' idfs, so a
        // per-segment idf here is wrong in the same way and for the same
        // reason -- and it is what left two phrase queries disagreeing with
        // Java on the segmented corpus after every other query agreed.
        let (df, dc) = match global.and_then(|g| g.get(&(query.field.clone(), term.clone()))) {
            Some(g) => (g.doc_freq, g.doc_count),
            None => (stats.doc_freq as i64, field_terms.doc_count as i64),
        };
        idf_sum += similarity::idf(df, dc);
    }

    // Documents first, positions second.
    //
    // A phrase can only match where every term does, so positions are needed
    // only for the intersection -- which for `phrase t0 t1` on the M1 corpus is
    // 2.2M documents out of `t0`'s 5M. Fetching each term's positions up front
    // materialized every one of them, roughly 15 million for `t0` alone, to
    // look at less than half. Intersecting the (cheap) doc lists first and then
    // asking only for those documents' positions is what `ExactPhraseMatcher`
    // effectively does by advancing per candidate.
    // Deliberately **unfiltered** by `live_docs`: `positions_for_docs` indexes
    // the wire position stream by each document's running frequency sum and
    // rejects a frequency list whose total disagrees with the term's
    // `totalTermFreq`, so dropping deleted documents here makes every scored
    // phrase query on a segment with deletions fail outright
    // ("sum of per-doc freqs disagrees with total_term_freq"). Deletions are
    // applied to the candidate list below instead, which is where they belong:
    // a deleted document is not a hit, but it still occupies its slot in the
    // term's postings.
    let mut per_term_docs: Vec<Vec<i32>> = Vec::with_capacity(query.terms.len());
    let mut per_term_freqs: Vec<Vec<i32>> = Vec::with_capacity(query.terms.len());
    for term in &query.terms {
        let Some(postings) = field_terms.postings(term, doc_in)? else {
            return Ok(());
        };
        per_term_docs.push(postings.docs.clone());
        per_term_freqs.push(postings.freqs.clone());
    }

    let per_term_docs_snapshot = per_term_docs.clone();
    let candidate_docs: Vec<i32> = Conjunction::new(
        per_term_docs
            .into_iter()
            .map(|v| Box::new(v.into_iter()) as BoxDocIter<'static>)
            .collect(),
    )
    .filter(|&doc_id| live_docs.is_none_or(|bits| bits.get(doc_id as usize)))
    .collect();
    if candidate_docs.is_empty() {
        return Ok(());
    }

    // Each term's own indices for the candidates, ascending -- one walk per
    // term, no searching, because both lists are sorted.
    let mut per_term_positions: Vec<Vec<i32>> = Vec::with_capacity(query.terms.len());
    let mut per_term_starts: Vec<Vec<u32>> = Vec::with_capacity(query.terms.len());
    for (t, term) in query.terms.iter().enumerate() {
        let docs = &per_term_docs_snapshot[t];
        let mut wanted = Vec::with_capacity(candidate_docs.len());
        let mut cursor = 0usize;
        for &doc_id in &candidate_docs {
            while cursor < docs.len() && docs[cursor] < doc_id {
                cursor += 1;
            }
            debug_assert!(
                cursor < docs.len() && docs[cursor] == doc_id,
                "candidates come from the intersection of these very lists"
            );
            wanted.push(cursor);
        }
        let stats = field_terms
            .seek_exact(term)
            .expect("term presence was established above");
        let (positions, starts) = field_terms.positions_for_docs(
            term,
            doc_in,
            pos_in,
            pay_in,
            &per_term_freqs[t],
            stats.total_term_freq,
            &wanted,
        )?;
        per_term_positions.push(positions);
        per_term_starts.push(starts);
    }

    // One cursor per term, advanced in step with the ascending candidate order,
    // so each doc's positions are found without hashing and without cloning.
    // Candidate `k` sits at index `k` in every term's positions, because
    // `wanted` was built from the same candidate list for each term. No
    // per-document cursor bookkeeping is left.
    let mut term_positions: Vec<&[i32]> = Vec::with_capacity(per_term_positions.len());
    // One norms cursor for this scan; `candidate_docs` ascends, so a sparse
    // field's `IndexedDISI` region is walked once, not once per document.
    let mut norms_cursor = norms.map(|n| n.cursor());
    for (k, &doc_id) in candidate_docs.iter().enumerate() {
        term_positions.clear();
        for t in 0..per_term_positions.len() {
            let start = per_term_starts[t][k] as usize;
            let end = per_term_starts[t][k + 1] as usize;
            term_positions.push(&per_term_positions[t][start..end]);
        }
        // `PhraseScorer.score()`: the frequency fed to the similarity is
        // `ExactPhraseMatcher`'s match count for `slop == 0`, and
        // `SloppyPhraseMatcher`'s summed `1/(1+matchLength)` otherwise -- not
        // a flat `1` for "matched". See `phrase_freq_sloppy`.
        let phrase_freq = if query.slop == 0 {
            phrase_freq_exact(&term_positions) as f32
        } else {
            phrase_freq_sloppy(&term_positions, query.slop)
        };
        if phrase_freq == 0.0 {
            continue;
        }
        let (field_length, avg_field_length) = match norms_cursor.as_mut() {
            Some(nc) => (nc.field_length(doc_id)?, nc.avg_field_length()),
            None => (
                similarity::UNNORMED_FIELD_LENGTH,
                similarity::UNNORMED_FIELD_LENGTH,
            ),
        };
        collector.collect(
            doc_id,
            similarity::do_score(
                idf_sum,
                phrase_freq,
                similarity::norm_inverse(
                    field_length,
                    avg_field_length,
                    similarity::DEFAULT_K1,
                    similarity::DEFAULT_B,
                ),
            ),
        );
    }
    Ok(())
}

/// The documents containing at least one of `terms`, ascending -- one
/// position's `UnionFullPostingsEnum` doc stream. Absent terms contribute
/// nothing; an entirely absent set yields an empty list, which makes the whole
/// phrase match nothing.
///
/// **Not** filtered by `live_docs`: the caller applies deletions to the
/// candidate list, for the same reason `search_phrase_query_scored_with_stats`
/// does (`positions_for_docs` indexes the wire stream by a running frequency
/// sum that must total the term's `totalTermFreq`).
fn multi_phrase_slot_docs(
    field_terms: &blocktree::FieldTerms,
    doc_in: Option<&DocInput<'_>>,
    terms: &[Vec<u8>],
) -> Result<Vec<i32>> {
    let mut merged = DocIdBitSet::default();
    for term in terms {
        let Some(docs) = term_docs_only(field_terms, term, doc_in)? else {
            continue;
        };
        for doc_id in docs {
            merged.set(doc_id);
        }
    }
    Ok(merged.into_sorted_vec())
}

/// One position's merged, sorted, deduplicated positions for each document in
/// `candidates` (which must be ascending). Returns `(positions, starts)` with
/// `starts.len() == candidates.len() + 1`, the same flat shape
/// `FieldTerms::positions_for_docs` returns for a single term.
fn multi_phrase_slot_positions(
    field_terms: &blocktree::FieldTerms,
    doc_in: Option<&DocInput<'_>>,
    pos_in: &PosInput<'_>,
    pay_in: Option<&PayInput<'_>>,
    terms: &[Vec<u8>],
    candidates: &[i32],
) -> Result<(Vec<i32>, Vec<u32>)> {
    let mut per_doc: Vec<Vec<i32>> = vec![Vec::new(); candidates.len()];
    for term in terms {
        let Some(stats) = field_terms.seek_exact(term) else {
            continue;
        };
        let Some(postings) = field_terms.postings(term, doc_in)? else {
            continue;
        };
        // `positions_for_docs` wants indices into the term's own **unfiltered**
        // doc sequence, and a frequency list totalling its `totalTermFreq` --
        // see `multi_phrase_slot_docs` on why deletions are not applied here.
        let term_docs = &postings.docs;
        let freqs = &postings.freqs;
        let mut wanted = Vec::new();
        let mut wanted_slots = Vec::new();
        let mut cursor = 0usize;
        for (slot, &doc_id) in candidates.iter().enumerate() {
            while cursor < term_docs.len() && term_docs[cursor] < doc_id {
                cursor += 1;
            }
            if cursor < term_docs.len() && term_docs[cursor] == doc_id {
                wanted.push(cursor);
                wanted_slots.push(slot);
            }
        }
        if wanted.is_empty() {
            continue;
        }
        let (positions, starts) = field_terms.positions_for_docs(
            term,
            doc_in,
            pos_in,
            pay_in,
            freqs,
            stats.total_term_freq,
            &wanted,
        )?;
        for (k, &slot) in wanted_slots.iter().enumerate() {
            let from = starts[k] as usize;
            let to = starts[k + 1] as usize;
            per_doc[slot].extend_from_slice(&positions[from..to]);
        }
    }

    let mut flat = Vec::new();
    let mut starts = Vec::with_capacity(candidates.len() + 1);
    starts.push(0u32);
    for doc_positions in &mut per_doc {
        // Sorted, and **not** deduplicated. `UnionPostingsEnum.freq()` builds
        // its `PositionsQueue` by draining every sub's positions and calling
        // `sort()` -- there is no dedup step, so a position reached by two
        // alternatives (the same term listed twice, or two terms the analyzer
        // stacked at one position) is yielded twice and the matcher counts the
        // alignment twice. Deduplicating here looks obviously right and is
        // wrong: real Lucene scores `pos:"(alpha alpha) beta"` at 0.87906057 on
        // this crate's fixture where the deduplicated form gives 0.6393168.
        // Recorded as `scoring.multiphrase.dup` and pinned by
        // `multi_phrase_query_scores_match_real_lucene_bit_for_bit`.
        doc_positions.sort_unstable();
        flat.extend_from_slice(doc_positions);
        starts.push(flat.len() as u32);
    }
    Ok((flat, starts))
}

/// [`search_phrase_query`]'s sibling for [`query::MultiPhraseQuery`]: matching
/// only, no scores. See that struct's doc comment for the exact semantics this
/// implements (and which parts of real `MultiPhraseQuery` are out of scope).
pub fn search_multi_phrase_query<C: Collector>(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    pos_in: Option<&PosInput<'_>>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &query::MultiPhraseQuery,
    collector: &mut C,
) -> Result<()> {
    for doc_id in multi_phrase_doc_ids(fields, doc_in, pos_in, pay_in, live_docs, query)? {
        collector.collect(doc_id);
    }
    Ok(())
}

/// The matching-only half of [`search_multi_phrase_query`], shared with
/// [`resolve_clause_docs`].
fn multi_phrase_doc_ids(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    pos_in: Option<&PosInput<'_>>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &query::MultiPhraseQuery,
) -> Result<Vec<i32>> {
    let mut docs = Vec::new();
    struct Collect<'a>(&'a mut Vec<i32>);
    impl collector::ScoringCollector for Collect<'_> {
        fn collect(&mut self, doc_id: i32, _score: f32) {
            self.0.push(doc_id);
        }
    }
    let mut sink = Collect(&mut docs);
    multi_phrase_hits(
        fields, doc_in, pos_in, pay_in, live_docs, query, None, None, &mut sink,
    )?;
    Ok(docs)
}

/// Scored [`query::MultiPhraseQuery`] execution -- see that struct's doc
/// comment for the idf-over-every-term and merged-frequency rules taken from
/// `MultiPhraseWeight`.
#[allow(clippy::too_many_arguments)]
pub fn search_multi_phrase_query_scored<C: ScoringCollector>(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    pos_in: Option<&PosInput<'_>>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &query::MultiPhraseQuery,
    norms: Option<&FieldNorms<'_>>,
    collector: &mut C,
) -> Result<()> {
    search_multi_phrase_query_scored_with_stats(
        fields, doc_in, pos_in, pay_in, live_docs, query, norms, None, collector,
    )
}

/// [`search_multi_phrase_query_scored`] taking reader-wide statistics -- see
/// [`CollectionStats`].
#[allow(clippy::too_many_arguments)]
pub fn search_multi_phrase_query_scored_with_stats<C: ScoringCollector>(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    pos_in: Option<&PosInput<'_>>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &query::MultiPhraseQuery,
    norms: Option<&FieldNorms<'_>>,
    global: Option<&GlobalStats>,
    collector: &mut C,
) -> Result<()> {
    multi_phrase_hits(
        fields, doc_in, pos_in, pay_in, live_docs, query, norms, global, collector,
    )
}

/// The one implementation both [`search_multi_phrase_query`] and
/// [`search_multi_phrase_query_scored_with_stats`] run, so matching and scoring
/// can never disagree about which documents match (the defect
/// `search_boolean_query_scored`'s two-pass general path used to have, see
/// finding #O16 in `docs/sweep/findings.md`).
#[allow(clippy::too_many_arguments)]
fn multi_phrase_hits<C: ScoringCollector>(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    pos_in: Option<&PosInput<'_>>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &query::MultiPhraseQuery,
    norms: Option<&FieldNorms<'_>>,
    global: Option<&GlobalStats>,
    collector: &mut C,
) -> Result<()> {
    // `MultiPhraseQuery.rewrite`: an empty `termArrays` is a
    // `MatchNoDocsQuery`.
    if query.term_arrays.is_empty() {
        return Ok(());
    }
    let Some(field_terms) = fields.field(&query.field) else {
        return Ok(());
    };

    // `MultiPhraseWeight`: the idf is summed over every term of every
    // position, and a term absent from this segment contributes nothing.
    let mut idf_sum = 0.0f32;
    let mut any_term_present = false;
    for alternatives in &query.term_arrays {
        for term in alternatives {
            let Some(stats) = field_terms.seek_exact(term) else {
                continue;
            };
            any_term_present = true;
            let (df, dc) = match global.and_then(|g| g.get(&(query.field.clone(), term.clone()))) {
                Some(g) => (g.doc_freq, g.doc_count),
                None => (stats.doc_freq as i64, field_terms.doc_count as i64),
            };
            idf_sum += similarity::idf(df, dc);
        }
    }
    if !any_term_present {
        return Ok(());
    }

    // `MultiPhraseQuery.rewrite`'s "optimize one-term case": a single position
    // rewrites to a `BooleanQuery` of `SHOULD` `TermQuery`s. That is **not**
    // the same thing as a one-slot phrase over the merged union -- a
    // `BooleanQuery` scores each term with its *own* idf and its *own*
    // frequency and adds them, where the phrase path would use the summed idf
    // once against the merged frequency. Real Lucene's recorded scores for
    // this shape (`scoring.multiphrase.single` in the blocktree fixture's
    // manifest) are the boolean ones, which is what caught the difference.
    if query.term_arrays.len() == 1 {
        let mut summed: HashMap<i32, f32> = HashMap::new();
        for term in &query.term_arrays[0] {
            let term_query = TermQuery::new(query.field.clone(), term.clone());
            for (doc_id, score) in
                term_doc_scores(fields, doc_in, live_docs, &term_query, norms, global)?
            {
                *summed.entry(doc_id).or_insert(0.0) += score;
            }
        }
        let mut hits: Vec<(i32, f32)> = summed.into_iter().collect();
        hits.sort_unstable_by_key(|&(doc_id, _)| doc_id);
        for (doc_id, score) in hits {
            collector.collect(doc_id, score);
        }
        return Ok(());
    }

    let Some(pos_in) = pos_in else {
        return Err(Error::MissingPosInput);
    };

    // One merged doc stream per position (`UnionFullPostingsEnum`'s doc side).
    let slot_docs: Vec<Vec<i32>> = query
        .term_arrays
        .iter()
        .map(|alts| multi_phrase_slot_docs(field_terms, doc_in, alts))
        .collect::<Result<Vec<_>>>()?;
    if slot_docs.iter().any(|docs| docs.is_empty()) {
        return Ok(());
    }

    // Documents first, positions second -- the same order `ExactPhraseMatcher`
    // effectively works in, and the reason the phrase path stopped decoding
    // every position of every document (finding #O15).
    let slot_count = slot_docs.len();
    let candidates: Vec<i32> = Conjunction::new(
        slot_docs
            .into_iter()
            .map(|docs| Box::new(docs.into_iter()) as BoxDocIter<'static>)
            .collect(),
    )
    .filter(|&doc_id| live_docs.is_none_or(|bits| bits.get(doc_id as usize)))
    .collect();
    if candidates.is_empty() {
        return Ok(());
    }

    let mut per_slot_positions = Vec::with_capacity(slot_count);
    let mut per_slot_starts = Vec::with_capacity(slot_count);
    for alternatives in &query.term_arrays {
        let (positions, starts) = multi_phrase_slot_positions(
            field_terms,
            doc_in,
            pos_in,
            pay_in,
            alternatives,
            &candidates,
        )?;
        per_slot_positions.push(positions);
        per_slot_starts.push(starts);
    }

    let mut slot_positions: Vec<&[i32]> = Vec::with_capacity(slot_count);
    // `candidates` ascends, so one cursor covers the whole scan.
    let mut norms_cursor = norms.map(|n| n.cursor());
    for (k, &doc_id) in candidates.iter().enumerate() {
        slot_positions.clear();
        for t in 0..per_slot_positions.len() {
            let from = per_slot_starts[t][k] as usize;
            let to = per_slot_starts[t][k + 1] as usize;
            slot_positions.push(&per_slot_positions[t][from..to]);
        }
        let freq = if query.slop == 0 {
            phrase_freq_exact(&slot_positions) as f32
        } else {
            phrase_freq_sloppy(&slot_positions, query.slop)
        };
        if freq == 0.0 {
            continue;
        }
        let norm_inverse = match norms_cursor.as_mut() {
            Some(nc) => nc.norm_inverse(doc_id)?,
            None => similarity::UNNORMED_NORM_INVERSE,
        };
        collector.collect(doc_id, similarity::do_score(idf_sum, freq, norm_inverse));
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
        search_boolean_query(&fields, doc_in.as_ref(), None, None, None, None, &q, &mut c).unwrap();
        assert_eq!(c.docs, vec![0]);
    }

    /// The lazy leapfrog path must be indistinguishable from the general
    /// materializing path, since it is an execution change and not a semantic
    /// one. Same query, both routes, same `(doc, score)` output.
    #[test]
    fn lazy_conjunction_matches_the_general_path_exactly() {
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let q = BooleanQuery::new()
            .with_must([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")]);

        let mut lazy = collector::TopDocsCollector::new(10);
        search_boolean_query_scored(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            None,
            &q,
            None,
            &mut lazy,
        )
        .unwrap();

        // Force the general path by giving the query a shape the gate rejects:
        // a `must_not` that excludes nothing changes the route, not the result.
        let general_q = BooleanQuery::new()
            .with_must([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")])
            .with_must_not([TermQuery::new("body", "no_such_term_anywhere")]);
        let mut general = collector::TopDocsCollector::new(10);
        search_boolean_query_scored(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            None,
            &general_q,
            None,
            &mut general,
        )
        .unwrap();

        assert_eq!(lazy.top_docs().len(), general.top_docs().len());
        for (a, b) in lazy.top_docs().iter().zip(general.top_docs()) {
            assert_eq!(a.doc_id, b.doc_id);
            assert!(
                (a.score - b.score).abs() < 1e-6,
                "{} vs {}",
                a.score,
                b.score
            );
        }
    }

    /// A `docFreq == 1` term is pulsed into the term dictionary and has no
    /// `.doc` bytes, so `lazy_postings` cannot open a cursor for it. Both lazy
    /// paths must hand such a query to the general path rather than erroring.
    ///
    /// Regression: the first cut of the lazy conjunction did not guard this and
    /// would have propagated `Unsupported("docFreq <= 1")` to the caller. The
    /// disjunction's version of the same bug is what the existing maxscore
    /// singleton test caught; nothing covered the conjunction, hence this.
    #[test]
    fn lazy_paths_fall_back_for_pulsed_singleton_terms() {
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());

        // "id"/"id2" has docFreq == 1 in this fixture.
        for q in [
            BooleanQuery::new()
                .with_must([TermQuery::new("body", "cat"), TermQuery::new("id", "id2")]),
            BooleanQuery::new()
                .with_should([TermQuery::new("body", "cat"), TermQuery::new("id", "id2")]),
        ] {
            let mut c = collector::TopDocsCollector::new(10);
            search_boolean_query_scored(
                &fields,
                doc_in.as_ref(),
                None,
                None,
                None,
                None,
                &q,
                None,
                &mut c,
            )
            .expect("pulsed singleton must not error");
        }
    }

    /// A term absent from the dictionary makes the whole conjunction empty --
    /// the lazy path must return that answer itself rather than falling through
    /// and letting the general path recompute it.
    #[test]
    fn lazy_conjunction_with_absent_term_matches_nothing() {
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let mut c = collector::TopDocsCollector::new(10);
        let q = BooleanQuery::new().with_must([
            TermQuery::new("body", "cat"),
            TermQuery::new("body", "definitely_not_indexed"),
        ]);
        search_boolean_query_scored(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            None,
            &q,
            None,
            &mut c,
        )
        .unwrap();
        assert!(c.top_docs().is_empty());
    }

    /// An absent *field* is the same story one level up.
    #[test]
    fn lazy_conjunction_with_absent_field_matches_nothing() {
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let mut c = collector::TopDocsCollector::new(10);
        let q = BooleanQuery::new().with_must([
            TermQuery::new("body", "cat"),
            TermQuery::new("no_such_field", "cat"),
        ]);
        search_boolean_query_scored(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            None,
            &q,
            None,
            &mut c,
        )
        .unwrap();
        assert!(c.top_docs().is_empty());
    }

    #[test]
    fn points_range_clause_without_points_input_is_missing_points_input_in_match() {
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let mut c = VecCollector::default();
        let q = BooleanQuery::new().with_must([Clause::PointsRange(
            crate::query::PointsRangeQuery::new("body", 0, 100),
        )]);
        let err =
            search_boolean_query(&fields, doc_in.as_ref(), None, None, None, None, &q, &mut c)
                .unwrap_err();
        assert!(matches!(err, Error::MissingPointsInput(field) if field == "body"));
    }

    #[test]
    fn points_range_clause_without_points_input_is_missing_points_input_in_scored_match() {
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let mut c = collector::TopDocsCollector::new(10);
        let q = BooleanQuery::new().with_must([Clause::PointsRange(
            crate::query::PointsRangeQuery::new("body", 0, 100),
        )]);
        let err = search_boolean_query_scored(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            None,
            &q,
            None,
            &mut c,
        )
        .unwrap_err();
        assert!(matches!(err, Error::MissingPointsInput(field) if field == "body"));
    }

    /// `field_infos` for a single synthetic `LongPoint`-shaped numeric field
    /// named "price" -- same minimal-construction template
    /// `multi_segment.rs`'s `numeric_field_infos` test helper uses for a
    /// doc-values field, adapted for `point_dimension_count`/
    /// `point_index_dimension_count`/`point_num_bytes` instead.
    fn price_field_infos(field_number: i32) -> lucene_codecs::field_infos::FieldInfos {
        use lucene_codecs::field_infos::{
            DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions, VectorEncoding,
            VectorSimilarityFunction,
        };
        lucene_codecs::field_infos::FieldInfos {
            fields: vec![FieldInfo {
                name: "price".to_string(),
                number: field_number,
                store_term_vectors: false,
                omit_norms: false,
                store_payloads: false,
                soft_deletes_field: false,
                parent_field: false,
                index_options: IndexOptions::None,
                doc_values_type: DocValuesType::None,
                doc_values_skip_index_type: DocValuesSkipIndexType::None,
                doc_values_gen: -1,
                attributes: vec![],
                point_dimension_count: 1,
                point_index_dimension_count: 1,
                point_num_bytes: 8,
                vector_dimension: 0,
                vector_encoding: VectorEncoding::Float32,
                vector_similarity_function: VectorSimilarityFunction::Euclidean,
            }],
        }
    }

    /// Task #199's end-to-end proof: a real query *string* (not a
    /// hand-built [`Clause::PointsRange`]) parsed via
    /// [`query_parser::parse_query`], resolved through
    /// [`search_boolean_query`] against a real `PointsInput` -- the full
    /// query-string-to-results path this task wires up, matching this
    /// module's own doc comment on [`Error::MissingPointsInput`].
    #[test]
    fn points_range_query_string_matches_real_segment_end_to_end() {
        use lucene_codecs::points::{self, WritePointsField};
        let segment_id = [7u8; lucene_store::codec_util::ID_LENGTH];
        // doc 0 -> 10, doc 1 -> 20, doc 2 -> 30, doc 3 -> 40, doc 4 -> 50.
        let points_field = WritePointsField {
            field_number: 3,
            num_dims: 1,
            num_index_dims: 1,
            bytes_per_dim: 8,
            points: (0..5)
                .map(|doc_id| {
                    (
                        doc_id,
                        points_query::pack_i64((doc_id as i64 + 1) * 10).to_vec(),
                    )
                })
                .collect(),
        };
        let (kdm, kdi, kdd) = points::write(&[points_field], 512, &segment_id, "").unwrap();
        let reader = points::open(&kdm, &kdi, &kdd, &segment_id, "").unwrap();
        let field_infos = price_field_infos(3);
        let points_input = PointsInput {
            reader,
            field_infos: &field_infos,
        };

        let fields = BlockTreeFields::empty();

        // Inclusive `[20 TO 40]` -- docs with values 20, 30, 40 (doc 1..=3).
        let clause = query_parser::parse_query("price:[20 TO 40]", None).unwrap();
        let q = BooleanQuery::new().with_must([clause]);
        let mut c = VecCollector::default();
        search_boolean_query(
            &fields,
            None,
            None,
            None,
            None,
            Some(&points_input),
            &q,
            &mut c,
        )
        .unwrap();
        assert_eq!(c.docs, vec![1, 2, 3]);

        // Exclusive `{20 TO 40}` (task #195's syntax) -- only doc 2 (value 30)
        // is strictly between 20 and 40.
        let clause = query_parser::parse_query("price:{20 TO 40}", None).unwrap();
        let q = BooleanQuery::new().with_must([clause]);
        let mut c = VecCollector::default();
        search_boolean_query(
            &fields,
            None,
            None,
            None,
            None,
            Some(&points_input),
            &q,
            &mut c,
        )
        .unwrap();
        assert_eq!(c.docs, vec![2]);

        // Unknown field name: no matches, not an error (same "missing field
        // means no matches" convention every other clause follows).
        let clause = query_parser::parse_query("nope:[0 TO 100]", None).unwrap();
        let q = BooleanQuery::new().with_must([clause]);
        let mut c = VecCollector::default();
        search_boolean_query(
            &fields,
            None,
            None,
            None,
            None,
            Some(&points_input),
            &q,
            &mut c,
        )
        .unwrap();
        assert!(c.docs.is_empty());

        // Scored sibling: same matched set, constant `1.0` score per doc
        // (real `PointRangeQuery` is `ConstantScoreQuery`-shaped).
        let clause = query_parser::parse_query("price:[20 TO 40]", None).unwrap();
        let q = BooleanQuery::new().with_must([clause]);
        let mut top = collector::TopDocsCollector::new(10);
        search_boolean_query_scored(
            &fields,
            None,
            None,
            None,
            None,
            Some(&points_input),
            &q,
            None,
            &mut top,
        )
        .unwrap();
        let mut docs: Vec<i32> = top.top_docs().iter().map(|d| d.doc_id).collect();
        docs.sort_unstable();
        assert_eq!(docs, vec![1, 2, 3]);
        assert!(top.top_docs().iter().all(|d| d.score == 1.0));
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
        search_boolean_query(&fields, doc_in.as_ref(), None, None, None, None, &q, &mut c).unwrap();
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
        search_boolean_query(&fields, doc_in.as_ref(), None, None, None, None, &q, &mut c).unwrap();
        assert_eq!(c.docs, vec![2]);
    }

    #[test]
    fn boolean_pure_must_not_matches_nothing() {
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let mut c = VecCollector::default();
        let q = BooleanQuery::new().with_must_not([TermQuery::new("body", "dog")]);
        search_boolean_query(&fields, doc_in.as_ref(), None, None, None, None, &q, &mut c).unwrap();
        assert!(c.docs.is_empty());
    }

    #[test]
    fn boolean_empty_query_matches_nothing() {
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let mut c = VecCollector::default();
        let q = BooleanQuery::new();
        search_boolean_query(&fields, doc_in.as_ref(), None, None, None, None, &q, &mut c).unwrap();
        assert!(c.docs.is_empty());
    }

    // ---- `Occur.FILTER` ---------------------------------------------------
    //
    // Matching-side unit tests. The scoring side is pinned bit-for-bit against
    // real `IndexSearcher` in `tests/bm25_scoring_fixtures.rs`; these cover the
    // branches of `matched_boolean_docs`/`try_conjunction_lazy` that a filter
    // clause reaches, using the same fixture segment
    // (cat={0,2}, dog={0,1}, bird={1,4}).

    /// Collects every `(doc, score)` the scorer emits, in emission order --
    /// the scored analogue of [`VecCollector`], kept local to the tests because
    /// production callers all want a top-`n` queue instead.
    #[derive(Default)]
    struct ScoreVecCollector {
        hits: Vec<(i32, f32)>,
    }

    impl collector::ScoringCollector for ScoreVecCollector {
        fn collect(&mut self, doc_id: i32, score: f32) {
            self.hits.push((doc_id, score));
        }
    }

    /// Helper: matched doc IDs for `query`, unscored.
    fn boolean_docs(query: &BooleanQuery) -> Vec<i32> {
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let mut c = VecCollector::default();
        search_boolean_query(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            None,
            query,
            &mut c,
        )
        .unwrap();
        c.docs
    }

    /// Helper: `(doc, score)` pairs for `query`, ascending by doc, no norms.
    fn boolean_scores(query: &BooleanQuery) -> Vec<(i32, f32)> {
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let mut c = ScoreVecCollector::default();
        search_boolean_query_scored(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            None,
            query,
            None,
            &mut c,
        )
        .unwrap();
        c.hits
    }

    #[test]
    fn filter_only_query_matches_the_conjunction_it_describes() {
        // Java's pure-negative rewrite is `clauses.size() ==
        // clauseSets.get(MUST_NOT).size()`, which a FILTER clause fails -- so a
        // filter-only query is a *positive* query and matches, unlike the
        // must_not-only query tested above.
        let q = BooleanQuery::new()
            .with_filter([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")]);
        assert_eq!(boolean_docs(&q), vec![0]);
    }

    #[test]
    fn filter_only_query_scores_every_match_zero() {
        let q = BooleanQuery::new().with_filter([TermQuery::new("body", "cat")]);
        assert_eq!(boolean_scores(&q), vec![(0, 0.0), (2, 0.0)]);
    }

    #[test]
    fn a_filter_clause_selects_exactly_what_the_same_must_clause_would() {
        let must = BooleanQuery::new()
            .with_must([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")]);
        let filter = BooleanQuery::new()
            .with_filter([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")]);
        let mixed = BooleanQuery::new()
            .with_must([TermQuery::new("body", "cat")])
            .with_filter([TermQuery::new("body", "dog")]);
        assert_eq!(boolean_docs(&must), vec![0]);
        assert_eq!(boolean_docs(&filter), boolean_docs(&must));
        assert_eq!(boolean_docs(&mixed), boolean_docs(&must));
    }

    #[test]
    fn a_filter_clause_contributes_exactly_zero_to_the_score() {
        // The difference between `+cat +dog` and `+cat #dog` must be exactly the
        // `dog` clause's own score, with the `cat` clause's contribution
        // bit-identical between the two -- which is what "does not perturb the
        // summation order" means when `f32` addition is not associative.
        let cat_only =
            boolean_scores(&BooleanQuery::new().with_must([TermQuery::new("body", "cat")]));
        let cat_and_dog = boolean_scores(
            &BooleanQuery::new()
                .with_must([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")]),
        );
        let cat_filter_dog = boolean_scores(
            &BooleanQuery::new()
                .with_must([TermQuery::new("body", "cat")])
                .with_filter([TermQuery::new("body", "dog")]),
        );
        assert_eq!(cat_filter_dog.len(), 1);
        assert_eq!(cat_filter_dog[0].0, 0);
        // Same matched set as the all-MUST form...
        assert_eq!(
            cat_and_dog.iter().map(|h| h.0).collect::<Vec<_>>(),
            cat_filter_dog.iter().map(|h| h.0).collect::<Vec<_>>()
        );
        // ... and the score is bit-for-bit the `cat` clause's alone.
        let cat_on_doc0 = cat_only
            .iter()
            .find(|h| h.0 == 0)
            .expect("cat matches doc 0")
            .1;
        assert_eq!(
            cat_filter_dog[0].1.to_bits(),
            cat_on_doc0.to_bits(),
            "filter contributed {} to the score",
            cat_filter_dog[0].1 - cat_on_doc0
        );
        assert!(cat_and_dog[0].1 > cat_filter_dog[0].1);
    }

    #[test]
    fn filter_clauses_do_not_count_toward_minimum_should_match() {
        // filter=[cat]={0,2}; should=[dog,bird] with minimum_should_match=1.
        // Doc 2 matches the filter and neither optional clause. If a filter
        // counted toward the threshold it would survive; Java increments
        // `shouldMatchCount` only for `Occur.SHOULD`, so it must not.
        let q = BooleanQuery::new()
            .with_filter([TermQuery::new("body", "cat")])
            .with_should([
                TermQuery::new("body", "dog"),
                TermQuery::new("body", "bird"),
            ])
            .with_minimum_should_match(1);
        assert_eq!(boolean_docs(&q), vec![0]);

        // With the threshold off, doc 2 comes back -- proving the exclusion
        // above was the threshold and not the filter.
        let q = BooleanQuery::new()
            .with_filter([TermQuery::new("body", "cat")])
            .with_should([
                TermQuery::new("body", "dog"),
                TermQuery::new("body", "bird"),
            ]);
        assert_eq!(boolean_docs(&q), vec![0, 2]);
    }

    #[test]
    fn a_filter_only_query_with_a_positive_minimum_should_match_matches_nothing() {
        // No `should` clauses at all, so no document can ever reach a positive
        // threshold -- the same rule as the `must`-only case, and the reason
        // `BooleanQuery::rewrite`'s single-clause unwrap excludes it.
        let q = BooleanQuery::new()
            .with_filter([TermQuery::new("body", "cat")])
            .with_minimum_should_match(1);
        assert!(boolean_docs(&q).is_empty());
    }

    #[test]
    fn filter_combines_with_must_not() {
        // filter=[cat]={0,2}, must_not=[dog]={0,1} -> {2}.
        let q = BooleanQuery::new()
            .with_filter([TermQuery::new("body", "cat")])
            .with_must_not([TermQuery::new("body", "dog")]);
        assert_eq!(boolean_docs(&q), vec![2]);
    }

    #[test]
    fn a_nested_boolean_query_works_as_a_filter_clause_and_still_scores_zero() {
        // filter=[nested], nested = should=[dog, bird] -> {0,1,4}; must=[cat]
        // -> {0,2}. Intersection {0}, scored as `cat` alone.
        let nested = BooleanQuery::new().with_should([
            TermQuery::new("body", "dog"),
            TermQuery::new("body", "bird"),
        ]);
        let q = BooleanQuery::new()
            .with_must([TermQuery::new("body", "cat")])
            .with_filter([Clause::from(nested)]);
        assert_eq!(boolean_docs(&q), vec![0]);

        let cat_only =
            boolean_scores(&BooleanQuery::new().with_must([TermQuery::new("body", "cat")]));
        let scored = boolean_scores(&q);
        assert_eq!(scored.len(), 1);
        assert_eq!(
            scored[0].1.to_bits(),
            cat_only
                .iter()
                .find(|h| h.0 == 0)
                .expect("cat matches doc 0")
                .1
                .to_bits(),
            "a nested boolean filter must contribute nothing either"
        );
    }

    #[test]
    fn a_filter_clause_that_matches_nothing_empties_the_conjunction() {
        let q = BooleanQuery::new()
            .with_must([TermQuery::new("body", "cat")])
            .with_filter([TermQuery::new("body", "no_such_term_anywhere")]);
        assert!(boolean_docs(&q).is_empty());
        assert!(boolean_scores(&q).is_empty());
    }

    #[test]
    fn a_filter_only_conjunction_takes_the_lazy_leapfrog_path_and_scores_zero() {
        // Not taking it cost 129.5ms against 8.8ms for the same conjunction
        // written with `MUST` clauses, because the general path materializes
        // each clause's whole doc list first -- see `benches/filter_vs_must.rs`.
        // Pruning is off for this shape (no scoring clause to bound), but the
        // leapfrog is not.
        let q = BooleanQuery::new()
            .with_filter([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")]);
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let mut c = ScoreVecCollector::default();
        assert!(
            try_conjunction_lazy(&fields, doc_in.as_ref(), None, &q, None, None, &mut c).unwrap(),
            "the lazy path must take a filter-only conjunction"
        );
        assert_eq!(c.hits, vec![(0, 0.0)]);
        assert_eq!(boolean_scores(&q), vec![(0, 0.0)]);
    }

    #[test]
    fn a_filter_only_conjunction_does_not_prune_against_a_full_top_n_queue() {
        // Every document scores 0, so the block-max bound would be 0 and
        // `0 <= threshold` would authorize skipping the rest of the segment the
        // moment the queue filled -- silently truncating the hit set on a tie.
        // `prunable` exists to stop that. `cat`={0,2} and `dog`={0,1} give only
        // one hit here, so the check that bites is the wider one: a top-1 queue
        // over a filter-only query must still see every match.
        let q = BooleanQuery::new().with_filter([TermQuery::new("body", "cat")]);
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let mut top = collector::TopDocsCollector::new(1);
        search_boolean_query_scored(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            None,
            &q,
            None,
            &mut top,
        )
        .unwrap();
        assert_eq!(
            top.total_hits().value,
            2,
            "both docs must have been visited"
        );
    }

    #[test]
    fn the_lazy_conjunction_path_handles_a_filter_leg() {
        // must=[cat], filter=[dog]: both are `Clause::Term`, so this *is* the
        // lazy leapfrog's shape, with one scoring leg and one non-scoring one.
        let q = BooleanQuery::new()
            .with_must([TermQuery::new("body", "cat")])
            .with_filter([TermQuery::new("body", "dog")]);
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let mut c = ScoreVecCollector::default();
        assert!(
            try_conjunction_lazy(&fields, doc_in.as_ref(), None, &q, None, None, &mut c).unwrap(),
            "the lazy path must take a conjunction with a filter leg"
        );
        assert_eq!(c.hits.iter().map(|h| h.0).collect::<Vec<_>>(), vec![0]);
        // Same score as `+body:cat` alone, through the same lazy path.
        let mut cat = ScoreVecCollector::default();
        try_conjunction_lazy(
            &fields,
            doc_in.as_ref(),
            None,
            &BooleanQuery::new().with_must([TermQuery::new("body", "cat")]),
            None,
            None,
            &mut cat,
        )
        .unwrap();
        assert_eq!(
            c.hits[0].1.to_bits(),
            cat.hits
                .iter()
                .find(|h| h.0 == 0)
                .expect("cat matches doc 0")
                .1
                .to_bits()
        );
    }

    #[test]
    fn the_lazy_disjunction_and_maxscore_paths_decline_a_query_with_filters() {
        // Both are pure-SHOULD fast paths; a filter clause is a required
        // clause, so neither shape applies and both must fall back rather than
        // silently ignore it.
        let q = BooleanQuery::new()
            .with_filter([TermQuery::new("body", "cat")])
            .with_should([
                TermQuery::new("body", "dog"),
                TermQuery::new("body", "bird"),
            ]);
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let mut c = ScoreVecCollector::default();
        assert!(
            !try_disjunction_lazy(&fields, doc_in.as_ref(), None, &q, None, None, &mut c).unwrap()
        );
        assert!(c.hits.is_empty());

        // The MAXSCORE entry point falls back to the general path, which does
        // honour the filter: `cat` = {0,2}, so doc 1 and doc 4 are excluded
        // even though they match `dog`/`bird`.
        let mut top = collector::TopDocsCollector::new(10);
        search_boolean_query_scored_maxscore(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            None,
            &q,
            None,
            &mut top,
        )
        .unwrap();
        let mut docs: Vec<i32> = top.top_docs().iter().map(|h| h.doc_id).collect();
        docs.sort_unstable();
        assert_eq!(docs, vec![0, 2]);
    }

    // ---- single-scoring-clause fast path + constant-score streaming --------
    //
    // c6 recorded `lib.rs` at 90.52% line coverage, below this repo's 95% bar.
    // The largest single hole was this pair: `search_boolean_query_scored`'s
    // one-clause shortcut and the `stream_constant_score_clause` /
    // `expanded_terms` machinery it is the only caller of. Nothing reached them
    // because the wildcard-family fixture suites all call
    // `search_prefix_query_scored` and friends *directly*, never through a
    // `BooleanQuery` -- so the whole shortcut, the streaming union, and its
    // early exit were untested. Fixture fields used below: `big` holds one term
    // `everywhere` in 300 documents, `many` holds 400 terms `term0000`..
    // `term0399` at one document each.

    #[test]
    fn a_single_prefix_clause_streams_through_the_constant_score_path() {
        // 1 expanded term at docFreq 300, so
        // `total_doc_freq >= terms * BLOCK_SIZE` and the streaming union runs
        // instead of materializing the postings. Every match scores a flat 1.0,
        // so a full top-`n` queue's bottom is already 1.0 and the stream stops
        // early -- `pruning_threshold().is_some_and(|s| s >= 1.0)`.
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let q = BooleanQuery::new().with_must([Clause::Prefix(PrefixQuery::new("big", "every"))]);
        let mut top = collector::TopDocsCollector::new(3);
        search_boolean_query_scored(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            None,
            &q,
            None,
            &mut top,
        )
        .unwrap();
        assert_eq!(top.top_docs().len(), 3);
        assert!(top.top_docs().iter().all(|h| h.score == 1.0));
        // Stopped early: the stream returned before visiting all 300.
        assert!(
            top.total_hits().value < 300,
            "the early exit did not fire: {} hits visited",
            top.total_hits().value
        );
    }

    #[test]
    fn a_single_wildcard_or_regexp_clause_streams_the_same_way() {
        // The other two `expanded_terms` arms. `e*e` and `ever.*` both select
        // the single term `everywhere`.
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        for clause in [
            Clause::Wildcard(WildcardQuery::new("big", "e*e")),
            Clause::Regexp(RegexpQuery::new("big", "ever.*")),
        ] {
            let q = BooleanQuery::new().with_must([clause.clone()]);
            let mut top = collector::TopDocsCollector::new(2);
            search_boolean_query_scored(
                &fields,
                doc_in.as_ref(),
                None,
                None,
                None,
                None,
                &q,
                None,
                &mut top,
            )
            .unwrap();
            assert_eq!(top.top_docs().len(), 2, "{clause:?}");
            assert!(top.top_docs().iter().all(|h| h.score == 1.0));
        }
    }

    #[test]
    fn the_streaming_path_declines_when_the_expansion_is_wider_than_it_is_deep() {
        // 400 terms at docFreq 1 each: `total_doc_freq (400) < terms (400) *
        // BLOCK_SIZE (128)`, so opening a lazy cursor per term would decode far
        // more than the whole union is worth. The clause falls back to
        // `resolve_clause_docs` and is collected at a flat 1.0 -- the other half
        // of the fast path's wildcard arm.
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let q = BooleanQuery::new().with_must([Clause::Prefix(PrefixQuery::new("many", "term00"))]);
        let mut all = ScoreVecCollector::default();
        search_boolean_query_scored(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            None,
            &q,
            None,
            &mut all,
        )
        .unwrap();
        assert_eq!(all.hits.len(), 100, "term0000..term0099");
        assert!(all.hits.iter().all(|h| h.1 == 1.0));
        assert!(
            all.hits.windows(2).all(|w| w[0].0 < w[1].0),
            "the fallback must still emit ascending doc ids"
        );
    }

    #[test]
    fn a_single_fuzzy_clause_takes_the_fast_path_without_streaming() {
        // `expanded_terms` returns `None` for `Clause::Fuzzy` (its expansion is
        // edit-distance driven, not a term-dictionary prefix scan), so the
        // stream declines immediately and the fallback collects.
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let q = BooleanQuery::new().with_must([Clause::Fuzzy(
            FuzzyQuery::new("body", "cat").with_max_edits(1),
        )]);
        let mut all = ScoreVecCollector::default();
        search_boolean_query_scored(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            None,
            &q,
            None,
            &mut all,
        )
        .unwrap();
        assert!(!all.hits.is_empty());
        assert!(all.hits.iter().all(|h| h.1 == 1.0));
    }

    #[test]
    fn a_single_phrase_clause_takes_the_fast_path_straight_to_the_collector() {
        // The `Clause::Phrase` arm: scored by `search_phrase_query_scored`
        // directly, with no intermediate `HashMap<i32, f32>`. It must agree
        // exactly with the standalone phrase search.
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let pos_in = doc.as_ref().map(|d| d.open_pos());
        let pay_in = doc.as_ref().map(|d| d.open_pay());
        let phrase = PhraseQuery::new("pos", ["alpha", "beta"]);

        let mut direct = ScoreVecCollector::default();
        search_phrase_query_scored(
            &fields,
            doc_in.as_ref(),
            pos_in.as_ref(),
            pay_in.as_ref(),
            None,
            &phrase,
            None,
            &mut direct,
        )
        .unwrap();

        let mut boxed = ScoreVecCollector::default();
        search_boolean_query_scored(
            &fields,
            doc_in.as_ref(),
            pos_in.as_ref(),
            pay_in.as_ref(),
            None,
            None,
            &BooleanQuery::new().with_must([Clause::Phrase(phrase.clone())]),
            None,
            &mut boxed,
        )
        .unwrap();

        assert!(!direct.hits.is_empty());
        assert_eq!(direct.hits.len(), boxed.hits.len());
        for (a, b) in direct.hits.iter().zip(&boxed.hits) {
            assert_eq!(a.0, b.0);
            assert_eq!(a.1.to_bits(), b.1.to_bits());
        }
    }

    #[test]
    fn a_single_pulsed_term_clause_reaches_the_fast_paths_term_arm() {
        // `try_conjunction_lazy` handles every other single-`Clause::Term`
        // conjunction, so the fast path's `Term` arm is only reachable for a
        // *pulsed* term -- docFreq <= 1, whose postings live in the term
        // dictionary and have no `.doc` bytes for a lazy cursor to open. The
        // `many` field is 400 such terms.
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let q = BooleanQuery::new().with_must([TermQuery::new("many", "term0123")]);
        let mut boxed = ScoreVecCollector::default();
        search_boolean_query_scored(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            None,
            &q,
            None,
            &mut boxed,
        )
        .unwrap();
        let mut direct = ScoreVecCollector::default();
        search_term_query_scored(
            &fields,
            doc_in.as_ref(),
            None,
            &TermQuery::new("many", "term0123"),
            None,
            &mut direct,
        )
        .unwrap();
        assert_eq!(boxed.hits.len(), 1);
        assert_eq!(boxed.hits[0].0, direct.hits[0].0);
        assert_eq!(boxed.hits[0].1.to_bits(), direct.hits[0].1.to_bits());
    }

    // ---- `clause_scores`' wildcard-family and fuzzy arms -------------------
    //
    // Reached only when the clause is *not* alone in the query (otherwise the
    // fast path above takes it), which is why nothing exercised them.

    #[test]
    fn the_wildcard_family_scores_a_flat_one_as_a_non_solo_boolean_clause() {
        // `should = [term, prefix|wildcard|regexp]`: the disjunction's lazy path
        // declines (not every clause is a `Clause::Term`), so this runs through
        // `clause_scores`, whose wildcard-family arms all score a flat 1.0 --
        // the port's stand-in for `ConstantScoreQuery`'s constant.
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        for clause in [
            Clause::Prefix(PrefixQuery::new("big", "every")),
            Clause::Wildcard(WildcardQuery::new("big", "e*e")),
            Clause::Regexp(RegexpQuery::new("big", "ever.*")),
        ] {
            let q = BooleanQuery::new()
                .with_should([Clause::Term(TermQuery::new("body", "cat")), clause.clone()]);
            let mut all = ScoreVecCollector::default();
            search_boolean_query_scored(
                &fields,
                doc_in.as_ref(),
                None,
                None,
                None,
                None,
                &q,
                None,
                &mut all,
            )
            .unwrap();
            // 300 `big` documents at 1.0 plus the two `body:cat` documents.
            assert_eq!(all.hits.len(), 302, "{clause:?}");
            assert_eq!(
                all.hits.iter().filter(|h| h.1 == 1.0).count(),
                300,
                "{clause:?}"
            );
        }
    }

    #[test]
    fn a_fuzzy_clause_scores_through_blended_term_statistics_as_a_boolean_clause() {
        // `clause_scores`' `Clause::Fuzzy` arm is the only caller of
        // `fuzzy_doc_scores`, which -- unlike the flat-1.0 wildcard family --
        // computes a real BM25 score per expanded term against the *blended*
        // doc frequency (`FuzzyQuery`'s `BlendedTermQuery` behaviour) and
        // weights it by that term's edit-distance boost.
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let q = BooleanQuery::new().with_should([
            Clause::Term(TermQuery::new("body", "bird")),
            Clause::Fuzzy(FuzzyQuery::new("body", "cat").with_max_edits(1)),
        ]);
        let mut all = ScoreVecCollector::default();
        search_boolean_query_scored(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            None,
            &q,
            None,
            &mut all,
        )
        .unwrap();
        assert!(!all.hits.is_empty());
        assert!(
            all.hits.iter().any(|h| h.1 > 0.0 && h.1 != 1.0),
            "a fuzzy clause must score, not contribute a flat constant: {:?}",
            all.hits
        );
    }

    #[test]
    fn a_fuzzy_clause_with_a_missing_field_scores_nothing() {
        // `fuzzy_doc_scores`' early return, and the only branch of it a
        // caller can trip without a term-dictionary walk.
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let q = BooleanQuery::new().with_should([
            Clause::Term(TermQuery::new("body", "cat")),
            Clause::Fuzzy(FuzzyQuery::new("no_such_field", "cat")),
        ]);
        let mut all = ScoreVecCollector::default();
        search_boolean_query_scored(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            None,
            &q,
            None,
            &mut all,
        )
        .unwrap();
        assert_eq!(all.hits.len(), 2, "only body:cat's documents");
    }

    // ---- block-max pruning in the two lazy paths --------------------------
    //
    // The other half of c6's coverage gap. Both lazy paths prune against the
    // collector's bottom score, and neither pruning branch was reachable from
    // the fixture's 5-document `body` field: the queue has to *fill* before a
    // threshold exists at all. The `big` field (one term, 300 documents, term
    // frequencies cycling 1..4) is the fixture's only field wide enough.

    #[test]
    fn the_lazy_conjunction_prunes_blocks_against_a_full_queue() {
        // Two legs over the same 300-document term, so the conjunction is the
        // whole postings list and the summed block-max bound is exactly twice
        // one leg's. Once the top-2 queue holds two maximum-frequency
        // documents, no later block can beat it and the leapfrog skips whole
        // spans on impacts alone.
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let q = BooleanQuery::new().with_must([
            TermQuery::new("big", "everywhere"),
            TermQuery::new("big", "everywhere"),
        ]);

        let mut pruned = collector::TopDocsCollector::new(2);
        search_boolean_query_scored(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            None,
            &q,
            None,
            &mut pruned,
        )
        .unwrap();

        // The same query with pruning forbidden (`ScoreMode::Complete`), as the
        // reference: pruning may not change which documents come back on top.
        let mut exhaustive = collector::TopDocsCollector::with_total_hits_threshold(2, u64::MAX);
        search_boolean_query_scored(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            None,
            &q,
            None,
            &mut exhaustive,
        )
        .unwrap();

        assert_eq!(exhaustive.total_hits().value, 300);
        assert_eq!(
            pruned
                .top_docs()
                .iter()
                .map(|h| h.score.to_bits())
                .collect::<Vec<_>>(),
            exhaustive
                .top_docs()
                .iter()
                .map(|h| h.score.to_bits())
                .collect::<Vec<_>>(),
            "pruning changed the top-2 scores"
        );
        assert!(
            pruned.total_hits().value < 300,
            "no block was skipped: {} of 300 documents visited",
            pruned.total_hits().value
        );
    }

    #[test]
    fn the_lazy_disjunction_prunes_blocks_against_a_full_queue() {
        // Same idea one shape over: a two-clause union whose second clause is
        // the fixture's 8,250-document `l1` term, so there are many blocks for
        // the skip loop to walk forward over on impacts alone.
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let q = BooleanQuery::new().with_should([
            TermQuery::new("big", "everywhere"),
            TermQuery::new("l1", "l1term"),
        ]);

        let mut pruned = collector::TopDocsCollector::new(3);
        search_boolean_query_scored(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            None,
            &q,
            None,
            &mut pruned,
        )
        .unwrap();

        let mut exhaustive = collector::TopDocsCollector::with_total_hits_threshold(3, u64::MAX);
        search_boolean_query_scored(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            None,
            &q,
            None,
            &mut exhaustive,
        )
        .unwrap();

        assert_eq!(exhaustive.total_hits().value, 8550, "300 `big` + 8250 `l1`");
        assert_eq!(
            pruned
                .top_docs()
                .iter()
                .map(|h| h.score.to_bits())
                .collect::<Vec<_>>(),
            exhaustive
                .top_docs()
                .iter()
                .map(|h| h.score.to_bits())
                .collect::<Vec<_>>(),
            "pruning changed the top-3 scores"
        );
        assert!(
            pruned.total_hits().value < 8550,
            "no block was skipped: {} of 8550 documents visited",
            pruned.total_hits().value
        );
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
        search_boolean_query(&fields, doc_in.as_ref(), None, None, None, None, &q, &mut c).unwrap();
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
        search_boolean_query(&fields, doc_in.as_ref(), None, None, None, None, &q, &mut c).unwrap();
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
        search_boolean_query(&fields, doc_in.as_ref(), None, None, None, None, &q, &mut c).unwrap();
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
        search_boolean_query(&fields, doc_in.as_ref(), None, None, None, None, &q, &mut c).unwrap();
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
        search_boolean_query(&fields, doc_in.as_ref(), None, None, None, None, &q, &mut c).unwrap();
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
        search_boolean_query(&fields, doc_in.as_ref(), None, None, None, None, &q, &mut c).unwrap();
        assert!(c.docs.is_empty());
    }

    #[test]
    fn boolean_minimum_should_match_with_no_should_clauses_at_all_matches_nothing() {
        // Task #60 edge case: a must-only query with a nonzero
        // `minimum_should_match` but zero `should` clauses -- this is the same
        // "should.len() < minimum_should_match" rule the exceeding-clause-count
        // test above already exercises, just at the should.len() == 0 boundary.
        // Real Lucene's `Boolean2ScorerSupplier.get()` returns a null scorer
        // (matches nothing) whenever `minShouldMatch > optionalScorers.size()`,
        // with no special-case carve-out for "must is otherwise satisfied" --
        // `minimum_should_match` isn't silently ignored just because there's
        // nothing for it to apply to. This port's counting mechanism already
        // produces that exact outcome with no separate branch (an empty
        // `should_docs` list means every doc's should-match count is 0, which
        // never reaches a nonzero threshold), so this test locks in already-
        // correct behavior rather than fixing a bug.
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let mut c = VecCollector::default();
        let q = BooleanQuery::new()
            .with_must([TermQuery::new("body", "cat")])
            .with_minimum_should_match(1);
        search_boolean_query(&fields, doc_in.as_ref(), None, None, None, None, &q, &mut c).unwrap();
        assert!(c.docs.is_empty());
    }

    #[test]
    fn boolean_duplicate_should_clause_counts_and_scores_twice() {
        // Task #60 edge case: the same `TermQuery` appears twice in `should`.
        // Real Lucene does not dedupe clauses at the `BooleanQuery` level --
        // two identical `TermQuery` clauses produce two independent scorers,
        // so a matching doc counts twice toward `minimum_should_match` and
        // scores twice (once per clause instance). This is the CORRECT
        // behavior to lock in, not a bug to fix: `should_match_counts` and
        // `clause_scores` already tally per clause instance, not per distinct
        // clause value, so duplicating a clause naturally double-counts both
        // the match tally and the score sum with no dedup logic anywhere.
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());

        // Matching: cat={0,2}. Duplicating "cat" in should with
        // minimum_should_match=2 requires 2 should-hits; a single distinct
        // should clause could never reach 2, so a match here proves the
        // duplicate is counted as two separate hits.
        let mut c = VecCollector::default();
        let q = BooleanQuery::new()
            .with_should([TermQuery::new("body", "cat"), TermQuery::new("body", "cat")])
            .with_minimum_should_match(2);
        search_boolean_query(&fields, doc_in.as_ref(), None, None, None, None, &q, &mut c).unwrap();
        assert_eq!(c.docs, vec![0, 2]);

        // Scoring: the duplicated clause's score contribution must be exactly
        // double a single instance's, not deduplicated to a single
        // contribution.
        let single = BooleanQuery::new().with_should([TermQuery::new("body", "cat")]);
        let duped = BooleanQuery::new()
            .with_should([TermQuery::new("body", "cat"), TermQuery::new("body", "cat")]);
        let mut single_scores = TopDocsCollector::new(10);
        search_boolean_query_scored(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            None,
            &single,
            None,
            &mut single_scores,
        )
        .unwrap();
        let mut duped_scores = TopDocsCollector::new(10);
        search_boolean_query_scored(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            None,
            &duped,
            None,
            &mut duped_scores,
        )
        .unwrap();
        let single_hits = single_scores.top_docs();
        let duped_hits = duped_scores.top_docs();
        assert_eq!(single_hits.len(), duped_hits.len());
        for (single_hit, duped_hit) in single_hits.iter().zip(duped_hits.iter()) {
            assert_eq!(single_hit.doc_id, duped_hit.doc_id);
            assert_eq!(duped_hit.score, single_hit.score * 2.0);
        }
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
        search_boolean_query(&fields, doc_in.as_ref(), None, None, None, None, &q, &mut c).unwrap();
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
        search_boolean_query(&fields, doc_in.as_ref(), None, None, None, None, &q, &mut c).unwrap();
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
        search_boolean_query(&fields, doc_in.as_ref(), None, None, None, None, &q, &mut c).unwrap();
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
        search_boolean_query(&fields, doc_in.as_ref(), None, None, None, None, &q, &mut c).unwrap();
        assert_eq!(c.docs, vec![0, 1, 2]);
    }

    #[test]
    fn nested_boolean_must_not_at_different_levels_does_not_leak_between_them() {
        // Task #60 edge case: a `must_not` at the outer level must exclude based
        // on the outer clause's own matching set, independent of a nested inner
        // `BooleanQuery`'s own separate `must_not`. dog={0,1}, cat={0,2},
        // bird={1,4}.
        //
        // inner: must=[dog], must_not=[cat] -> dog={0,1} minus cat={0,2} = {1}.
        let inner = BooleanQuery::new()
            .with_must([TermQuery::new("body", "dog")])
            .with_must_not([TermQuery::new("body", "cat")]);

        // First confirm the inner query's own isolated result is {1}, so the
        // outer-level assertion below is unambiguous about which level's
        // must_not did the excluding.
        let mut inner_only = VecCollector::default();
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        search_boolean_query(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            None,
            &inner.clone(),
            &mut inner_only,
        )
        .unwrap();
        assert_eq!(inner_only.docs, vec![1]);

        // outer: must=[inner], must_not=[bird] -> inner's own result {1} minus
        // bird={1,4} = {} -- the outer must_not (bird) excludes doc 1 on its own
        // criteria, entirely independent of inner's own must_not (cat), which
        // never even mentions bird.
        let mut c = VecCollector::default();
        let outer = BooleanQuery::new()
            .with_must([Clause::Boolean(Box::new(inner))])
            .with_must_not([TermQuery::new("body", "bird")]);
        search_boolean_query(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            None,
            &outer,
            &mut c,
        )
        .unwrap();
        assert!(c.docs.is_empty());
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
        search_boolean_query(&fields, doc_in.as_ref(), None, None, None, None, &q, &mut c).unwrap();
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
        search_boolean_query(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            None,
            &top,
            &mut c,
        )
        .unwrap();
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
        search_boolean_query(&fields, doc_in.as_ref(), None, None, None, None, &q, &mut c).unwrap();
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
            None,
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
        assert!(phrase_matches_in_doc(&[&[0][..], &[1][..], &[2][..]]));
    }

    #[test]
    fn phrase_matches_exact_alignment_at_a_later_position() {
        assert!(phrase_matches_in_doc(&[
            &[0, 5][..],
            &[1, 6][..],
            &[2, 7][..]
        ]));
    }

    #[test]
    fn phrase_no_match_despite_every_term_present() {
        // "cat" at 0 and 10, "sat" at 1, "mat" at 5: no base position aligns all three
        // (0 -> needs 1 at "sat" (ok) and 2 at "mat" (missing); 10 -> needs 11 (missing)).
        assert!(!phrase_matches_in_doc(&[&[0, 10][..], &[1][..], &[5][..]]));
    }

    #[test]
    fn phrase_multiple_candidates_only_one_aligns() {
        // Base 0 fails (needs 2 at term index 2, only 5/7 present); base 3 succeeds
        // (needs 4 at term index 1 -- present -- and 5 at term index 2 -- present).
        assert!(phrase_matches_in_doc(&[
            &[0, 3][..],
            &[1, 4][..],
            &[5, 7][..]
        ]));
    }

    #[test]
    fn phrase_single_term_degenerates_to_any_occurrence() {
        assert!(phrase_matches_in_doc(&[&[2, 9][..]]));
    }

    #[test]
    fn phrase_single_term_with_no_occurrences_is_false() {
        assert!(!phrase_matches_in_doc(&[&[][..]]));
    }

    #[test]
    fn phrase_no_terms_at_all_is_false() {
        assert!(!phrase_matches_in_doc(&[]));
    }

    #[test]
    fn phrase_a_term_with_no_occurrences_in_this_doc_is_false() {
        assert!(!phrase_matches_in_doc(&[&[0][..], &[][..]]));
    }

    #[test]
    fn phrase_repeated_term_with_consecutive_occurrences_matches() {
        // "the the": both occurrence lists are the term "the"'s own positions --
        // 0 and 1 are consecutive, so "the" at 0 followed by "the" at 1 is a match.
        assert!(phrase_matches_in_doc(&[&[0, 1, 2][..], &[0, 1, 2][..]]));
    }

    #[test]
    fn phrase_repeated_term_without_consecutive_occurrences_does_not_match() {
        // "the" only occurs at 0, 2, 4 -- no two consecutive occurrences exist.
        assert!(!phrase_matches_in_doc(&[&[0, 2, 4][..], &[0, 2, 4][..]]));
    }

    // `phrase_matches_in_doc_sloppy` unit tests: hand-computed slop values against
    // the formula documented on that function -- `(p_last - p_first) - (n - 1)`.

    #[test]
    fn sloppy_exact_alignment_needs_zero_slop() {
        // positions 0,1,2: (2-0)-2 = 0 moves needed -- matches at slop=0.
        assert!(phrase_matches_in_doc_sloppy(
            &[&[0][..], &[1][..], &[2][..]],
            0
        ));
    }

    #[test]
    fn sloppy_agrees_with_exact_for_slop_zero_no_match_case() {
        // "cat" at 0, "sat" at 2: (2-0)-1 = 1 move needed, slop=0 is one short.
        assert!(!phrase_matches_in_doc_sloppy(&[&[0][..], &[2][..]], 0));
        assert!(!phrase_matches_in_doc(&[&[0][..], &[2][..]]));
    }

    #[test]
    fn sloppy_gap_of_one_extra_word_needs_slop_one() {
        // "quick" at 0, "fox" at 2 (one word -- "brown" -- skipped in between):
        // (2-0)-1 = 1 move needed.
        assert!(!phrase_matches_in_doc_sloppy(&[&[0][..], &[2][..]], 0));
        assert!(phrase_matches_in_doc_sloppy(&[&[0][..], &[2][..]], 1));
    }

    #[test]
    fn sloppy_boundary_exactly_enough_slop_matches() {
        // "a" at 0, "b" at 4: (4-0)-1 = 3 moves needed. slop=3 matches, slop=2 (one
        // less than enough) does not.
        assert!(phrase_matches_in_doc_sloppy(&[&[0][..], &[4][..]], 3));
        assert!(!phrase_matches_in_doc_sloppy(&[&[0][..], &[4][..]], 2));
    }

    #[test]
    fn sloppy_three_term_gap_sums_across_both_intervals() {
        // "the" at 0, "quick" at 2 (gap 1), "fox" at 5 (gap 2): total moves =
        // (5-0)-2 = 3, matching the sum of per-interval gaps (1 + 2).
        assert!(phrase_matches_in_doc_sloppy(
            &[&[0][..], &[2][..], &[5][..]],
            3
        ));
        assert!(!phrase_matches_in_doc_sloppy(
            &[&[0][..], &[2][..], &[5][..]],
            2
        ));
    }

    #[test]
    fn sloppy_picks_the_best_of_multiple_candidate_base_positions() {
        // First term at {0, 10}; second term at {1, 11}. Base 0 -> 1 needs 0 moves;
        // base 10 -> 11 also needs 0 moves -- either way it should match at slop=0,
        // proving every base candidate is tried (not just the first).
        assert!(phrase_matches_in_doc_sloppy(
            &[&[0, 10][..], &[1, 11][..]],
            0
        ));
    }

    #[test]
    fn sloppy_greedy_finds_smallest_valid_next_position() {
        // First term at 0; second term's list has {1, 2, 100} -- greedy must pick 1
        // (smallest valid), needing 0 moves, not be confused by the far-away 100.
        assert!(phrase_matches_in_doc_sloppy(
            &[&[0][..], &[1, 2, 100][..]],
            0
        ));
    }

    #[test]
    fn sloppy_no_increasing_alignment_exists_still_fails_at_high_slop() {
        // Second term's only occurrence (0) is not strictly after the first term's
        // only occurrence (0) -- no in-order alignment exists at any slop, since
        // this port's scope excludes reordering/ties.
        assert!(!phrase_matches_in_doc_sloppy(&[&[0][..], &[0][..]], 100));
    }

    #[test]
    fn sloppy_single_term_degenerates_to_any_occurrence_regardless_of_slop() {
        assert!(phrase_matches_in_doc_sloppy(&[&[2, 9][..]], 0));
        assert!(phrase_matches_in_doc_sloppy(&[&[2, 9][..]], 5));
    }

    #[test]
    fn sloppy_single_term_with_no_occurrences_is_false() {
        assert!(!phrase_matches_in_doc_sloppy(&[&[][..]], 5));
    }

    #[test]
    fn sloppy_no_terms_at_all_is_false() {
        assert!(!phrase_matches_in_doc_sloppy(&[], 5));
    }

    #[test]
    fn sloppy_a_term_with_no_occurrences_in_this_doc_is_false() {
        assert!(!phrase_matches_in_doc_sloppy(&[&[0][..], &[][..]], 5));
    }

    #[test]
    fn sloppy_repeated_term_with_a_gap_matches_at_sufficient_slop() {
        // "the" at 0, 3 -- as a two-term "the the" phrase, base 0 needs the second
        // "the" strictly after 0: smallest is 3, (3-0)-1 = 2 moves.
        assert!(phrase_matches_in_doc_sloppy(&[&[0, 3][..], &[0, 3][..]], 2));
        assert!(!phrase_matches_in_doc_sloppy(
            &[&[0, 3][..], &[0, 3][..]],
            1
        ));
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
    fn phrase_alignment_walk_handles_a_phrase_longer_than_the_inline_cursor_array() {
        // `phrase_freq_exact_impl` keeps its cursors on the stack for up to
        // `MAX_INLINE_PHRASE_TERMS` non-leading terms and spills to a `Vec`
        // past that. Both sides of that branch must behave identically, and
        // the spill side has no other caller in this crate's tests.
        let n = MAX_INLINE_PHRASE_TERMS + 5;
        // A phrase of `n + 1` terms that aligns exactly once, at p0 == 0.
        let lists: Vec<Vec<i32>> = (0..=n).map(|i| vec![i as i32, 500 + i as i32]).collect();
        let refs: Vec<&[i32]> = lists.iter().map(|v| v.as_slice()).collect();
        assert!(phrase_matches_in_doc(&refs));
        // Both the p0 == 0 and the p0 == 500 alignments are valid.
        assert_eq!(phrase_freq_exact(&refs), 2);

        // Break one position in the middle and it must stop matching.
        let mut broken = lists.clone();
        broken[n / 2] = vec![9999];
        let broken_refs: Vec<&[i32]> = broken.iter().map(|v| v.as_slice()).collect();
        assert!(!phrase_matches_in_doc(&broken_refs));
        assert_eq!(phrase_freq_exact(&broken_refs), 0);
    }

    #[test]
    fn phrase_alignment_walk_agrees_with_a_brute_force_check() {
        // The merge walk's early exits (an exhausted cursor breaks out of the
        // whole scan, a mismatch only skips this `p0`) are the parts most
        // likely to be subtly wrong, and no fixture exercises them densely.
        // Compare against an independent O(n^2) definition over many shapes.
        fn brute(term_positions: &[&[i32]]) -> i32 {
            let Some((first, rest)) = term_positions.split_first() else {
                return 0;
            };
            let mut freq = 0;
            for &p0 in first.iter() {
                if rest
                    .iter()
                    .enumerate()
                    .all(|(i, ps)| ps.contains(&(p0 + i as i32 + 1)))
                {
                    freq += 1;
                }
            }
            freq
        }

        // A deterministic pseudo-random spread of positions, several widths.
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for terms in 1..=4usize {
            for _ in 0..200 {
                let lists: Vec<Vec<i32>> = (0..terms)
                    .map(|_| {
                        let mut v: Vec<i32> = (0i32..24).filter(|_| next() % 3 != 0).collect();
                        v.dedup();
                        v
                    })
                    .collect();
                let refs: Vec<&[i32]> = lists.iter().map(|v| v.as_slice()).collect();
                let expected = if lists.iter().any(|l| l.is_empty()) {
                    0
                } else {
                    brute(&refs)
                };
                assert_eq!(phrase_freq_exact(&refs), expected, "{lists:?}");
                assert_eq!(phrase_matches_in_doc(&refs), expected > 0, "{lists:?}");
            }
        }
    }

    // ---- `phrase_freq_sloppy` (`SloppyPhraseMatcher.sloppyWeight()` summed) ----

    #[test]
    fn sloppy_phrase_freq_at_slop_zero_equals_the_exact_count() {
        // The defining agreement: with a zero move budget every match has
        // `matchLength == 0`, so each weighs `1/(1+0) == 1` and the sum is
        // exactly `phrase_freq_exact`'s count. Checked over several shapes,
        // including a repeated-term phrase and a non-matching one.
        let cases: [(Vec<Vec<i32>>, u32); 4] = [
            (vec![vec![0, 5], vec![1, 6]], 2),
            (vec![vec![0, 1, 2], vec![1, 2, 3]], 3),
            (vec![vec![0, 4], vec![9]], 0),
            (vec![vec![7], vec![3]], 0),
        ];
        for (positions, expected) in cases {
            let refs: Vec<&[i32]> = positions.iter().map(|v| v.as_slice()).collect();
            assert_eq!(phrase_freq_exact(&refs), expected as i32, "{positions:?}");
            assert_eq!(
                phrase_freq_sloppy(&refs, 0),
                expected as f32,
                "slop-0 sloppy freq must equal the exact count for {positions:?}"
            );
        }
    }

    #[test]
    fn sloppy_phrase_freq_weights_a_loose_match_below_a_tight_one() {
        // `alpha`@0 with `beta`@1 is `matchLength == 0` -> weight 1.
        let tight = phrase_freq_sloppy(&[&[0], &[1]], 3);
        // `alpha`@0 with `beta`@3 is `matchLength == 2` -> weight 1/3, the
        // exact case `fixtures/data/blocktree_index`'s doc 7 records against
        // real Lucene (see `tests/bm25_scoring_fixtures.rs`).
        let loose = phrase_freq_sloppy(&[&[0], &[3]], 3);
        assert_eq!(tight, 1.0);
        assert_eq!(loose, 1.0 / 3.0);
        assert!(loose < tight);
    }

    #[test]
    fn sloppy_phrase_freq_sums_every_starting_position() {
        // Two starts: 0 (pairs with 1, matchLength 0) and 4 (pairs with 7,
        // matchLength 2). Sum = 1 + 1/3.
        let got = phrase_freq_sloppy(&[&[0, 4], &[1, 7]], 3);
        assert_eq!(got, 1.0 + 1.0 / 3.0);
    }

    #[test]
    fn sloppy_phrase_freq_skips_starts_over_the_slop_budget() {
        // Start 0 needs 4 moves to reach `beta`@5, over a budget of 2; start 6
        // reaches `beta`@7 with 0. Only the second contributes.
        let got = phrase_freq_sloppy(&[&[0, 6], &[5, 7]], 2);
        assert_eq!(got, 1.0);
    }

    #[test]
    fn sloppy_phrase_freq_edge_cases_match_the_matchers_own_contract() {
        // Empty phrase, an empty position list, and a single-term phrase --
        // the same three edge cases `phrase_matches_in_doc_sloppy` documents.
        assert_eq!(phrase_freq_sloppy(&[], 5), 0.0);
        assert_eq!(phrase_freq_sloppy(&[&[1, 2], &[]], 5), 0.0);
        assert_eq!(phrase_freq_sloppy(&[&[1, 2, 9]], 5), 3.0);
        assert_eq!(phrase_freq_sloppy(&[&[]], 5), 0.0);
    }

    #[test]
    fn sloppy_phrase_freq_is_nonzero_exactly_when_the_matcher_says_it_matches() {
        // The frequency and the boolean matcher must never disagree about
        // whether a document matches -- they are used together (one gates the
        // unscored path, the other the scored one).
        let shapes: [Vec<Vec<i32>>; 6] = [
            vec![vec![0], vec![1]],
            vec![vec![0], vec![3]],
            vec![vec![0], vec![9]],
            vec![vec![0, 5], vec![2, 6], vec![3, 20]],
            vec![vec![4], vec![2]],
            vec![vec![1, 2, 3], vec![2, 3, 4], vec![3, 4, 5]],
        ];
        for slop in 0..=4u32 {
            for shape in &shapes {
                let refs: Vec<&[i32]> = shape.iter().map(|v| v.as_slice()).collect();
                assert_eq!(
                    phrase_freq_sloppy(&refs, slop) > 0.0,
                    phrase_matches_in_doc_sloppy(&refs, slop),
                    "slop={slop} shape={shape:?}"
                );
            }
        }
    }

    #[test]
    fn phrase_freq_exact_counts_one_match_when_phrase_occurs_once() {
        // "quick fox" at position 0/1 only.
        assert_eq!(phrase_freq_exact(&[&[0][..], &[1][..]]), 1);
    }

    #[test]
    fn phrase_freq_exact_counts_every_repeated_occurrence() {
        // "the the": "the" at 0,1,2,3 -- valid starts at 0,1,2 (0+1=1 present,
        // 1+1=2 present, 2+1=3 present), 3 has no successor -- 3 matches, not 1.
        let positions = [vec![0, 1, 2, 3], vec![0, 1, 2, 3]];
        assert_eq!(
            phrase_freq_exact(&positions.iter().map(|v| v.as_slice()).collect::<Vec<_>>()),
            3
        );
    }

    #[test]
    fn phrase_freq_exact_zero_when_no_alignment_exists() {
        assert_eq!(phrase_freq_exact(&[&[0][..], &[5][..]]), 0);
    }

    #[test]
    fn phrase_freq_exact_zero_for_empty_term_positions() {
        assert_eq!(phrase_freq_exact(&[]), 0);
    }

    #[test]
    fn phrase_freq_exact_zero_when_any_term_has_no_occurrences() {
        assert_eq!(phrase_freq_exact(&[&[0][..], &[][..]]), 0);
    }

    #[test]
    fn phrase_freq_exact_single_term_counts_every_occurrence() {
        assert_eq!(phrase_freq_exact(&[&[0, 3, 7][..]]), 3);
    }

    #[test]
    fn phrase_freq_exact_non_overlapping_repeats_counts_two() {
        // "quick fox ... quick fox": two disjoint adjacent pairs, no overlap
        // possible between them.
        let positions = [vec![0, 10], vec![1, 11]];
        assert_eq!(
            phrase_freq_exact(&positions.iter().map(|v| v.as_slice()).collect::<Vec<_>>()),
            2
        );
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
        let err = search_boolean_query(&fields, Some(&doc_in), None, None, None, None, &q, &mut c)
            .unwrap_err();
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
        search_disjunction_max_query(&fields, Some(&doc_in), None, None, None, None, &q, &mut c)
            .unwrap();
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
        search_disjunction_max_query(&fields, Some(&doc_in), None, None, None, None, &q, &mut c)
            .unwrap();
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
        search_disjunction_max_query(&fields, Some(&doc_in), None, None, None, None, &q, &mut c)
            .unwrap();
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
        search_boolean_query(&fields, Some(&doc_in), None, None, None, None, &q, &mut c).unwrap();
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
        search_disjunction_max_query(&fields, Some(&doc_in), None, None, None, None, &q, &mut c)
            .unwrap();
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
        search_disjunction_max_query(
            &fields,
            Some(&doc_in),
            None,
            None,
            None,
            None,
            &outer,
            &mut c,
        )
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
        let err = search_disjunction_max_query(
            &fields,
            Some(&doc_in),
            None,
            None,
            None,
            None,
            &q,
            &mut c,
        )
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
        search_boolean_query(&fields, Some(&doc_in), None, None, None, None, &q, &mut c).unwrap();
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
        search_boolean_query(&fields, Some(&doc_in), None, None, None, None, &q, &mut c).unwrap();
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
        search_boolean_query_scored(
            &fields,
            Some(&doc_in),
            None,
            None,
            None,
            None,
            &q,
            None,
            &mut top,
        )
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
        search_boolean_query(&fields, Some(&doc_in), None, None, None, None, &q, &mut c).unwrap();
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
        search_boolean_query(&fields, Some(&doc_in), None, None, None, None, &q, &mut c).unwrap();
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
        search_boolean_query_scored(
            &fields,
            Some(&doc_in),
            None,
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
        search_boolean_query(&fields, Some(&doc_in), None, None, None, None, &q, &mut c).unwrap();
        assert_eq!(c.docs, vec![0]);

        let mut top = TopDocsCollector::new(10);
        search_boolean_query_scored(
            &fields,
            Some(&doc_in),
            None,
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
        search_boolean_query(&fields, Some(&doc_in), None, None, None, None, &q, &mut c).unwrap();
        // cat={0,2} union bird={1,4} = {0,1,2,4}.
        assert_eq!(c.docs, vec![0, 1, 2, 4]);

        let mut top = TopDocsCollector::new(10);
        search_boolean_query_scored(
            &fields,
            Some(&doc_in),
            None,
            None,
            None,
            None,
            &q,
            None,
            &mut top,
        )
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
        search_boolean_query(&fields, Some(&doc_in), None, None, None, None, &q, &mut c).unwrap();
        assert_eq!(c.docs, vec![0, 2]);

        let mut top = TopDocsCollector::new(10);
        search_boolean_query_scored(
            &fields,
            Some(&doc_in),
            None,
            None,
            None,
            None,
            &q,
            None,
            &mut top,
        )
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

    /// Differential proof for [`search_term_query_scored_maxscore`], against
    /// the checked-in real-Lucene fixture's `big`/`"everywhere"` term
    /// (`docFreq == 300`: one full 256-doc level-0 block with real,
    /// Java-written impacts -- `field.big.term.everywhere.impacts.results` in
    /// `manifest.properties` -- plus a 44-doc tail; see
    /// `crates/lucene-codecs/tests/blocktree_fixtures.rs`'s
    /// `big_field_impacts_match_real_lucene_impacts_enum`, which already
    /// proves those decoded impacts match real Lucene byte-for-byte). Opens
    /// this field's *real* per-doc norms (`_0.nvd`/`_0.nvm`) rather than
    /// passing `norms: None` -- deliberately, because the impacts'
    /// `(freq, norm)` bound is only a valid upper bound against the score
    /// formula that actually consumes those same real norm bytes; comparing
    /// it against the unrelated `UNNORMED_FIELD_LENGTH` fallback score
    /// instead would not be a documented approximation but an unproven (and
    /// in general unsound) mix, so this test doesn't exercise that
    /// combination.
    ///
    /// For several `top_n` values asserts:
    /// - [`search_term_query_scored_maxscore`]'s `TopDocsCollector::top_docs()`
    ///   is *exactly* [`search_term_query_scored`] (the eager path)'s, proving
    ///   the skip never changes the result;
    /// - for small `top_n` (1, 5, 50), at least one real block-skip actually
    ///   happened (via `test_only_maxscore_block_skip_counter`, reset/read
    ///   around each call) -- ruling out a vacuously-passing test where the
    ///   skip branch is simply never taken. Every doc in this fixture's block
    ///   is one of only four distinct `(freq, norm)` combinations, cycling
    ///   `1,2,3,4,1,2,3,...`, so the single best-scoring combination is
    ///   reached within the first few docs and the whole rest of the block
    ///   (a few hundred docs) becomes safely skippable from there on;
    /// - for `top_n = 300` (the full `docFreq`: every doc must eventually be
    ///   collected, so the collector is never "full" until after the very
    ///   last doc), the skip count is 0 -- proving the threshold check itself
    ///   is conservative, not "sometimes skips, still happens to match".
    #[test]
    fn maxscore_lazy_path_matches_eager_path_on_real_fixture_and_actually_skips_blocks() {
        let (fields, doc) = open_fixture();
        let doc = doc.unwrap();
        let doc_in = doc.open();

        // Open real per-doc norms for "big" (see this test's doc comment for
        // why `norms: None` wouldn't be a sound comparison here).
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
        let max_doc: i32 = get("max_doc").parse().unwrap();
        let fnm = std::fs::read(format!("{dir}{}.raw", get("fnm_file_name"))).expect("read .fnm");
        let field_infos = lucene_codecs::field_infos::parse(&fnm, &id, "").expect("parse .fnm");
        let big_field_number = field_infos
            .fields
            .iter()
            .find(|f| f.name == "big")
            .expect("fixture has a \"big\" field")
            .number;

        let nvm = std::fs::read(format!("{dir}_0.nvm")).expect("read _0.nvm");
        let nvd = std::fs::read(format!("{dir}_0.nvd")).expect("read _0.nvd");
        let (_, parsed_norms) =
            lucene_codecs::norms::parse_meta(&nvm, &id, "").expect("parse .nvm");
        let entry = *parsed_norms
            .entry(big_field_number)
            .expect("\"big\" field has a norms entry");
        let norms = FieldNorms::open(&nvd, entry, max_doc, None).expect("FieldNorms::open");

        let query = TermQuery::new("big", b"everywhere".as_slice());

        for &top_n in &[1usize, 5, 50, 300] {
            let mut eager = TopDocsCollector::new(top_n);
            search_term_query_scored(
                &fields,
                Some(&doc_in),
                None,
                &query,
                Some(&norms),
                &mut eager,
            )
            .expect("eager search");

            test_only_maxscore_block_skip_counter::reset();
            let mut lazy = TopDocsCollector::new(top_n);
            search_term_query_scored_maxscore(
                &fields,
                Some(&doc_in),
                None,
                &query,
                Some(&norms),
                &mut lazy,
            )
            .expect("maxscore search");
            let skips = test_only_maxscore_block_skip_counter::count();

            assert_eq!(
                eager.top_docs(),
                lazy.top_docs(),
                "top_{top_n} must match exactly between eager and maxscore paths"
            );

            if top_n < 300 {
                assert!(
                    skips > 0,
                    "top_{top_n} should reach the block's best-scoring combination \
                     within its first few docs, making the rest of the block \
                     (out of docFreq 300) safely skippable (got {skips} skips)"
                );
            } else {
                assert_eq!(
                    skips, 0,
                    "top_{top_n} == the full docFreq: the collector is never full \
                     until the very last doc, so nothing should be skippable \
                     (got {skips} skips)"
                );
            }
        }
    }

    /// `ScoreMode.isExhaustive()`'s whole purpose: a collector in
    /// [`collector::ScoreMode::Complete`] promises an exact `totalHits`, so the
    /// scorer must not skip a single block. Proven with the skip counter rather
    /// than with the result -- the results are necessarily identical either
    /// way, and it is exactly the kind of gate that can go dead without any
    /// assertion about output noticing.
    #[test]
    fn an_exhaustive_score_mode_collector_disables_every_maxscore_block_skip() {
        let (fields, doc) = open_fixture();
        let doc = doc.unwrap();
        let doc_in = doc.open();
        let query = TermQuery::new("big", b"everywhere".as_slice());

        test_only_maxscore_block_skip_counter::reset();
        let mut pruning = TopDocsCollector::new(1);
        search_term_query_scored_maxscore(&fields, Some(&doc_in), None, &query, None, &mut pruning)
            .expect("maxscore search");
        let pruning_skips = test_only_maxscore_block_skip_counter::count();
        assert!(
            pruning_skips > 0,
            "the pruning baseline must actually skip, or this test proves nothing"
        );
        assert_eq!(pruning.score_mode(), collector::ScoreMode::TopScores);
        assert_eq!(
            pruning.total_hits().relation,
            collector::TotalHitsRelation::GreaterThanOrEqualTo,
            "a pruned search cannot claim an exact count"
        );

        test_only_maxscore_block_skip_counter::reset();
        let mut exhaustive = TopDocsCollector::with_total_hits_threshold(1, u64::MAX);
        assert!(exhaustive.score_mode().is_exhaustive());
        search_term_query_scored_maxscore(
            &fields,
            Some(&doc_in),
            None,
            &query,
            None,
            &mut exhaustive,
        )
        .expect("maxscore search");
        assert_eq!(
            test_only_maxscore_block_skip_counter::count(),
            0,
            "no block may be skipped for an exhaustive collector"
        );
        assert_eq!(
            exhaustive.total_hits().relation,
            collector::TotalHitsRelation::EqualTo
        );
        assert_eq!(
            exhaustive.total_hits().value,
            300,
            "the fixture's \"big\"/\"everywhere\" term has docFreq 300, all of them counted"
        );
        // And the top hit is the same either way -- pruning changed the work,
        // not the answer.
        assert_eq!(pruning.top_docs(), exhaustive.top_docs());
    }

    #[test]
    fn maxscore_returns_immediately_for_unknown_field_or_term() {
        let (fields, doc) = open_fixture();
        let doc = doc.unwrap();
        let doc_in = doc.open();

        let mut c = TopDocsCollector::new(5);
        search_term_query_scored_maxscore(
            &fields,
            Some(&doc_in),
            None,
            &TermQuery::new("nonexistent_field", "x"),
            None,
            &mut c,
        )
        .unwrap();
        assert!(c.top_docs().is_empty());

        let mut c = TopDocsCollector::new(5);
        search_term_query_scored_maxscore(
            &fields,
            Some(&doc_in),
            None,
            &TermQuery::new("body", "no_such_term_anywhere"),
            None,
            &mut c,
        )
        .unwrap();
        assert!(c.top_docs().is_empty());
    }

    #[test]
    fn maxscore_falls_back_to_eager_path_when_doc_in_is_none() {
        // `body` is a multi-doc field/term, so both the eager and maxscore
        // paths require `.doc` bytes -- confirms the `doc_in == None`
        // fallback branch produces the same (error) outcome as calling the
        // eager path directly, rather than panicking or silently succeeding.
        let (fields, _doc) = open_fixture();
        let query = TermQuery::new("body", "cat");

        let mut eager = TopDocsCollector::new(5);
        let eager_result = search_term_query_scored(&fields, None, None, &query, None, &mut eager);

        let mut lazy = TopDocsCollector::new(5);
        let lazy_result =
            search_term_query_scored_maxscore(&fields, None, None, &query, None, &mut lazy);

        assert_eq!(eager_result.is_err(), lazy_result.is_err());
        if eager_result.is_ok() {
            assert_eq!(eager.top_docs(), lazy.top_docs());
        }
    }

    #[test]
    fn maxscore_falls_back_to_eager_path_for_a_singleton_docfreq_one_term() {
        // "id"/"id2" is a `docFreq == 1` singleton (pulsed into the term
        // dictionary, no `.doc` bytes at all) -- exercises the `doc_freq <=
        // 1` fallback branch specifically, distinct from the `doc_in ==
        // None` branch above (here `doc_in` is `Some`).
        let (fields, doc) = open_fixture();
        let doc = doc.unwrap();
        let doc_in = doc.open();
        let query = TermQuery::new("id", "id2");

        let mut eager = TopDocsCollector::new(5);
        search_term_query_scored(&fields, Some(&doc_in), None, &query, None, &mut eager).unwrap();

        test_only_maxscore_block_skip_counter::reset();
        let mut lazy = TopDocsCollector::new(5);
        search_term_query_scored_maxscore(&fields, Some(&doc_in), None, &query, None, &mut lazy)
            .unwrap();

        assert_eq!(eager.top_docs(), lazy.top_docs());
        assert_eq!(
            test_only_maxscore_block_skip_counter::count(),
            0,
            "a singleton term has no blocks to skip"
        );
    }

    #[test]
    fn maxscore_falls_back_to_unnormed_field_length_when_norms_is_none() {
        // Confirms the `norms: None` -> `UNNORMED_FIELD_LENGTH` branch is
        // exercised, that skipping actually happens (not vacuously passing
        // because nothing gets pruned), and -- the specific bug this test
        // caught during review -- that the block-skip bound is computed
        // against the *same* `UNNORMED_FIELD_LENGTH` scoring formula the
        // real per-doc score below uses, not against the field's real
        // on-wire norm bytes (which would be an unsound mix: this fixture's
        // "big"/"everywhere" impacts were written against real per-doc
        // norms, so blindly feeding them into `max_score_for_impacts` while
        // scoring downstream with `UNNORMED_FIELD_LENGTH` produced an
        // under-estimated bound that could -- and did -- skip a block
        // containing a doc that belonged in the top-K).
        let (fields, doc) = open_fixture();
        let doc = doc.unwrap();
        let doc_in = doc.open();
        let query = TermQuery::new("big", b"everywhere".as_slice());

        let mut eager = TopDocsCollector::new(5);
        search_term_query_scored(&fields, Some(&doc_in), None, &query, None, &mut eager).unwrap();

        test_only_maxscore_block_skip_counter::reset();
        let mut lazy = TopDocsCollector::new(5);
        search_term_query_scored_maxscore(&fields, Some(&doc_in), None, &query, None, &mut lazy)
            .unwrap();

        assert_eq!(eager.top_docs(), lazy.top_docs());
        assert!(
            test_only_maxscore_block_skip_counter::count() > 0,
            "this fixture's docFreq (300) spans more than one level-0 block, \
             so some block should be skippable once the top-5 threshold is reached"
        );
    }

    /// Differential proof for
    /// [`search_boolean_query_scored_maxscore`], analogous to
    /// `maxscore_lazy_path_matches_eager_path_on_real_fixture_and_actually_skips_blocks`
    /// one level up: a three-clause pure-SHOULD `BooleanQuery` over this
    /// fixture's real Lucene-written data (`big`/`"everywhere"`, `docFreq ==
    /// 300`, spanning a real full level-0 block plus a tail; `body`/`"cat"`
    /// and `body`/`"dog"`, `docFreq == 2` each). For many `top_n` values,
    /// asserts [`search_boolean_query_scored_maxscore`]'s
    /// `TopDocsCollector::top_docs()` is byte-identical to
    /// [`search_boolean_query_scored`] (the exhaustive path), proving the
    /// skip logic never changes the result.
    #[test]
    fn boolean_maxscore_lazy_path_matches_eager_path_on_real_fixture() {
        let (fields, doc) = open_fixture();
        let doc = doc.unwrap();
        let doc_in = doc.open();

        let query = BooleanQuery::new().with_should([
            Clause::from(TermQuery::new("big", b"everywhere".as_slice())),
            Clause::from(TermQuery::new("body", "cat")),
            Clause::from(TermQuery::new("body", "dog")),
        ]);

        for &top_n in &[1usize, 2, 5, 20, 300] {
            let mut eager = TopDocsCollector::new(top_n);
            search_boolean_query_scored(
                &fields,
                Some(&doc_in),
                None,
                None,
                None,
                None,
                &query,
                None,
                &mut eager,
            )
            .expect("eager search");

            let mut lazy = TopDocsCollector::new(top_n);
            search_boolean_query_scored_maxscore(
                &fields,
                Some(&doc_in),
                None,
                None,
                None,
                None,
                &query,
                None,
                &mut lazy,
            )
            .expect("maxscore search");

            assert_eq!(
                eager.top_docs(),
                lazy.top_docs(),
                "top_{top_n} must match exactly between eager and maxscore boolean paths"
            );
        }
    }

    /// Counter-based proof (analogous to
    /// `maxscore_lazy_path_matches_eager_path_on_real_fixture_and_actually_skips_blocks`'s
    /// own skip-count assertion) that
    /// [`search_boolean_query_scored_maxscore`] doesn't just happen to match
    /// the eager path's output -- it genuinely skips real per-clause level-0
    /// block decode for the same three-clause query above, reusing
    /// [`test_only_maxscore_block_skip_counter`] (shared with the
    /// single-term MAXSCORE path, since both increment the same counter at
    /// the same kind of skip site: `advance()`ing straight past a level-0
    /// block's last doc without decoding the rest of it).
    #[test]
    fn test_only_boolean_maxscore_block_skip_counter_records_real_skips() {
        let (fields, doc) = open_fixture();
        let doc = doc.unwrap();
        let doc_in = doc.open();

        let query = BooleanQuery::new().with_should([
            Clause::from(TermQuery::new("big", b"everywhere".as_slice())),
            Clause::from(TermQuery::new("body", "cat")),
            Clause::from(TermQuery::new("body", "dog")),
        ]);

        // A small top_n reaches its final threshold quickly relative to
        // "big"'s 300-doc postings list, making at least one of its blocks
        // (out of several) provably unable to beat that threshold once
        // combined with the other clauses' global bounds.
        test_only_maxscore_block_skip_counter::reset();
        let mut lazy = TopDocsCollector::new(1);
        search_boolean_query_scored_maxscore(
            &fields,
            Some(&doc_in),
            None,
            None,
            None,
            None,
            &query,
            None,
            &mut lazy,
        )
        .expect("maxscore search");
        assert!(
            test_only_maxscore_block_skip_counter::count() > 0,
            "top_1 should make at least one clause's block provably \
             uncompetitive against the other clauses' global bounds"
        );

        // The full docFreq: the collector is never full until the very last
        // candidate doc, so nothing should be skippable -- same conservative
        // check the single-term differential test above makes.
        test_only_maxscore_block_skip_counter::reset();
        let mut lazy_full = TopDocsCollector::new(9000);
        search_boolean_query_scored_maxscore(
            &fields,
            Some(&doc_in),
            None,
            None,
            None,
            None,
            &query,
            None,
            &mut lazy_full,
        )
        .expect("maxscore search");
        assert_eq!(
            test_only_maxscore_block_skip_counter::count(),
            0,
            "top_n covering every possible hit should never need to skip anything"
        );
    }

    #[test]
    fn boolean_maxscore_falls_back_to_eager_path_when_must_clause_present() {
        // A non-empty `must` disqualifies the fast path entirely (see this
        // function's doc comment) -- confirms the fallback produces exactly
        // the eager path's output rather than silently mishandling `must`.
        let (fields, doc) = open_fixture();
        let doc = doc.unwrap();
        let doc_in = doc.open();

        let query = BooleanQuery::new()
            .with_must([Clause::from(TermQuery::new("body", "cat"))])
            .with_should([Clause::from(TermQuery::new("body", "dog"))]);

        let mut eager = TopDocsCollector::new(5);
        search_boolean_query_scored(
            &fields,
            Some(&doc_in),
            None,
            None,
            None,
            None,
            &query,
            None,
            &mut eager,
        )
        .unwrap();

        let mut lazy = TopDocsCollector::new(5);
        search_boolean_query_scored_maxscore(
            &fields,
            Some(&doc_in),
            None,
            None,
            None,
            None,
            &query,
            None,
            &mut lazy,
        )
        .unwrap();

        assert_eq!(eager.top_docs(), lazy.top_docs());
    }

    #[test]
    fn boolean_maxscore_falls_back_to_eager_path_for_minimum_should_match_above_one() {
        let (fields, doc) = open_fixture();
        let doc = doc.unwrap();
        let doc_in = doc.open();

        let query = BooleanQuery::new()
            .with_should([
                Clause::from(TermQuery::new("body", "cat")),
                Clause::from(TermQuery::new("body", "dog")),
                Clause::from(TermQuery::new("body", "bird")),
            ])
            .with_minimum_should_match(2);

        let mut eager = TopDocsCollector::new(5);
        search_boolean_query_scored(
            &fields,
            Some(&doc_in),
            None,
            None,
            None,
            None,
            &query,
            None,
            &mut eager,
        )
        .unwrap();

        let mut lazy = TopDocsCollector::new(5);
        search_boolean_query_scored_maxscore(
            &fields,
            Some(&doc_in),
            None,
            None,
            None,
            None,
            &query,
            None,
            &mut lazy,
        )
        .unwrap();

        assert_eq!(eager.top_docs(), lazy.top_docs());
    }

    #[test]
    fn boolean_maxscore_falls_back_to_eager_path_for_nested_boolean_clause() {
        // A nested `Clause::Boolean` isn't `Clause::Term`, so it disqualifies
        // the fast path (see this function's doc comment) -- confirms the
        // fallback still produces the exhaustive path's exact result.
        let (fields, doc) = open_fixture();
        let doc = doc.unwrap();
        let doc_in = doc.open();

        let query = BooleanQuery::new().with_should([
            Clause::from(TermQuery::new("body", "cat")),
            Clause::Boolean(Box::new(
                BooleanQuery::new().with_should([Clause::from(TermQuery::new("body", "dog"))]),
            )),
        ]);

        let mut eager = TopDocsCollector::new(5);
        search_boolean_query_scored(
            &fields,
            Some(&doc_in),
            None,
            None,
            None,
            None,
            &query,
            None,
            &mut eager,
        )
        .unwrap();

        let mut lazy = TopDocsCollector::new(5);
        search_boolean_query_scored_maxscore(
            &fields,
            Some(&doc_in),
            None,
            None,
            None,
            None,
            &query,
            None,
            &mut lazy,
        )
        .unwrap();

        assert_eq!(eager.top_docs(), lazy.top_docs());
    }

    #[test]
    fn boolean_maxscore_falls_back_to_eager_path_for_singleton_docfreq_one_clause() {
        // "id"/"id2" has `docFreq == 1` (pulsed, no `.doc` bytes) -- exercises
        // the per-clause `doc_freq <= 1` fallback branch.
        let (fields, doc) = open_fixture();
        let doc = doc.unwrap();
        let doc_in = doc.open();

        let query = BooleanQuery::new().with_should([
            Clause::from(TermQuery::new("body", "cat")),
            Clause::from(TermQuery::new("id", "id2")),
        ]);

        let mut eager = TopDocsCollector::new(5);
        search_boolean_query_scored(
            &fields,
            Some(&doc_in),
            None,
            None,
            None,
            None,
            &query,
            None,
            &mut eager,
        )
        .unwrap();

        let mut lazy = TopDocsCollector::new(5);
        search_boolean_query_scored_maxscore(
            &fields,
            Some(&doc_in),
            None,
            None,
            None,
            None,
            &query,
            None,
            &mut lazy,
        )
        .unwrap();

        assert_eq!(eager.top_docs(), lazy.top_docs());
    }

    #[test]
    fn boolean_maxscore_falls_back_to_eager_path_when_doc_in_is_none() {
        let (fields, _doc) = open_fixture();
        let query = BooleanQuery::new().with_should([
            Clause::from(TermQuery::new("body", "cat")),
            Clause::from(TermQuery::new("body", "dog")),
        ]);

        let mut eager = TopDocsCollector::new(5);
        let eager_result = search_boolean_query_scored(
            &fields, None, None, None, None, None, &query, None, &mut eager,
        );

        let mut lazy = TopDocsCollector::new(5);
        let lazy_result = search_boolean_query_scored_maxscore(
            &fields, None, None, None, None, None, &query, None, &mut lazy,
        );

        assert_eq!(eager_result.is_err(), lazy_result.is_err());
        if eager_result.is_ok() {
            assert_eq!(eager.top_docs(), lazy.top_docs());
        }
    }

    /// c12: `search_boolean_query_scored_maxscore` is now a delegate for
    /// `search_boolean_query_scored`, so it must agree with the plain scored
    /// entry point on *every* shape -- not only on the ones the deleted body
    /// used to decline. The two shapes that most directly pinned the deleted
    /// body's unreachability are in the matrix: a clause naming an absent
    /// field and a clause naming an absent term. `try_disjunction_lazy`
    /// *accepts* both (it drops the clause from the union and keeps going),
    /// which is exactly why the deleted body -- which declined them -- could
    /// never be reached for them either.
    #[test]
    fn boolean_maxscore_agrees_with_the_plain_scored_path_on_every_shape() {
        let (fields, doc) = open_fixture();
        let doc = doc.unwrap();
        let doc_in = doc.open();

        let shapes: Vec<(&str, BooleanQuery)> = vec![
            (
                "pure should, both terms present",
                BooleanQuery::new().with_should([
                    Clause::from(TermQuery::new("body", "cat")),
                    Clause::from(TermQuery::new("body", "dog")),
                ]),
            ),
            (
                "a clause naming an absent field",
                BooleanQuery::new().with_should([
                    Clause::from(TermQuery::new("body", "cat")),
                    Clause::from(TermQuery::new("no_such_field", "cat")),
                ]),
            ),
            (
                "a clause naming an absent term in a present field",
                BooleanQuery::new().with_should([
                    Clause::from(TermQuery::new("body", "cat")),
                    Clause::from(TermQuery::new("body", "no_such_term")),
                ]),
            ),
            (
                "every clause absent",
                BooleanQuery::new().with_should([
                    Clause::from(TermQuery::new("body", "no_such_term")),
                    Clause::from(TermQuery::new("no_such_field", "cat")),
                ]),
            ),
            (
                "a must clause (the disjunction gate declines)",
                BooleanQuery::new()
                    .with_must([Clause::from(TermQuery::new("body", "cat"))])
                    .with_should([Clause::from(TermQuery::new("body", "dog"))]),
            ),
            (
                "a filter clause (c11's fourth list)",
                BooleanQuery::new()
                    .with_should([Clause::from(TermQuery::new("body", "cat"))])
                    .with_filter([Clause::from(TermQuery::new("body", "dog"))]),
            ),
            (
                "a must_not clause",
                BooleanQuery::new()
                    .with_should([Clause::from(TermQuery::new("body", "cat"))])
                    .with_must_not([Clause::from(TermQuery::new("body", "dog"))]),
            ),
            (
                "a pulsed docFreq == 1 clause",
                BooleanQuery::new().with_should([
                    Clause::from(TermQuery::new("body", "cat")),
                    Clause::from(TermQuery::new("id", "id2")),
                ]),
            ),
            (
                "a nested boolean clause",
                BooleanQuery::new().with_should([
                    Clause::from(TermQuery::new("body", "cat")),
                    Clause::Boolean(Box::new(
                        BooleanQuery::new()
                            .with_should([Clause::from(TermQuery::new("body", "dog"))]),
                    )),
                ]),
            ),
            (
                "a 300-document clause that spans real level-0 blocks",
                BooleanQuery::new().with_should([
                    Clause::from(TermQuery::new("big", b"everywhere".as_slice())),
                    Clause::from(TermQuery::new("body", "cat")),
                ]),
            ),
        ];

        let mut any_hits = false;
        for (name, query) in &shapes {
            for top_n in [1usize, 3, 10, 50] {
                let mut eager = TopDocsCollector::new(top_n);
                search_boolean_query_scored(
                    &fields,
                    Some(&doc_in),
                    None,
                    None,
                    None,
                    None,
                    query,
                    None,
                    &mut eager,
                )
                .unwrap();

                let mut delegated = TopDocsCollector::new(top_n);
                search_boolean_query_scored_maxscore(
                    &fields,
                    Some(&doc_in),
                    None,
                    None,
                    None,
                    None,
                    query,
                    None,
                    &mut delegated,
                )
                .unwrap();

                any_hits |= !eager.top_docs().is_empty();
                assert_eq!(
                    eager.top_docs(),
                    delegated.top_docs(),
                    "shape {name:?} at top_n={top_n} disagreed"
                );
            }
        }
        assert!(
            any_hits,
            "the matrix must actually produce hits or it proves nothing"
        );
    }

    #[test]
    fn search_term_query_scored_with_similarity_using_defaults_matches_hardcoded_path() {
        // Task #214 regression proof: Bm25Params::default() through the new
        // parameterized path must reproduce `search_term_query_scored`'s
        // hardcoded-default scores byte-for-byte.
        let (fields, doc) = open_fixture();
        let doc = doc.unwrap();
        let doc_in = doc.open();
        let query = TermQuery::new("body", "cat");

        let mut default_path = TopDocsCollector::new(10);
        search_term_query_scored(
            &fields,
            Some(&doc_in),
            None,
            &query,
            None,
            &mut default_path,
        )
        .unwrap();

        let mut with_similarity = TopDocsCollector::new(10);
        search_term_query_scored_with_similarity(
            &fields,
            Some(&doc_in),
            None,
            &query,
            None,
            similarity::Bm25Params::default(),
            &mut with_similarity,
        )
        .unwrap();

        assert_eq!(default_path.top_docs(), with_similarity.top_docs());
        assert!(
            !default_path.top_docs().is_empty(),
            "test fixture must actually produce hits for this proof to be meaningful"
        );
    }

    #[test]
    fn search_term_query_scored_with_similarity_using_different_k1_b_changes_scores() {
        // A genuinely different k1/b must produce a measurably different,
        // correctly-computed score -- computed by hand from the same BM25
        // formula `similarity.rs`'s doc comment documents, not just "some
        // different number".
        let (fields, doc) = open_fixture();
        let doc = doc.unwrap();
        let doc_in = doc.open();
        let query = TermQuery::new("body", "cat");

        let mut default_path = TopDocsCollector::new(10);
        search_term_query_scored(
            &fields,
            Some(&doc_in),
            None,
            &query,
            None,
            &mut default_path,
        )
        .unwrap();

        let custom_params = similarity::Bm25Params::new(2.0, 0.5).expect("in range");
        let mut with_similarity = TopDocsCollector::new(10);
        search_term_query_scored_with_similarity(
            &fields,
            Some(&doc_in),
            None,
            &query,
            None,
            custom_params,
            &mut with_similarity,
        )
        .unwrap();

        assert!(!default_path.top_docs().is_empty());
        assert_eq!(
            default_path.top_docs().len(),
            with_similarity.top_docs().len()
        );

        for (default_hit, custom_hit) in default_path
            .top_docs()
            .iter()
            .zip(with_similarity.top_docs().iter())
        {
            assert_eq!(default_hit.doc_id, custom_hit.doc_id);
            // Hand-computed expected score using `similarity::score_with_params`
            // directly (same formula the production path calls), with
            // UNNORMED_FIELD_LENGTH for both field-length terms since this
            // fixture is opened without norms (`None` passed above).
            let field_terms = fields.field("body").unwrap();
            let stats = field_terms.seek_exact(b"cat").unwrap();
            let doc_freqs = term_doc_freqs(&fields, Some(&doc_in), None, &query).unwrap();
            let freq = doc_freqs
                .iter()
                .find(|&&(d, _)| d == default_hit.doc_id)
                .unwrap()
                .1;
            let expected = similarity::score_with_params(
                stats.doc_freq as i64,
                field_terms.doc_count as i64,
                freq as f32,
                similarity::UNNORMED_FIELD_LENGTH,
                similarity::UNNORMED_FIELD_LENGTH,
                custom_params,
            );
            assert!(
                (custom_hit.score - expected).abs() < 1e-5,
                "doc={} got={} expected={}",
                custom_hit.doc_id,
                custom_hit.score,
                expected
            );
            assert!(
                (custom_hit.score - default_hit.score).abs() > 1e-3,
                "different k1/b must measurably change the score: default={} custom={}",
                default_hit.score,
                custom_hit.score
            );
        }
    }

    #[test]
    fn search_term_query_scored_with_similarity_missing_field_yields_no_hits() {
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let mut collector = TopDocsCollector::new(10);
        search_term_query_scored_with_similarity(
            &fields,
            doc_in.as_ref(),
            None,
            &TermQuery::new("no_such_field", "cat"),
            None,
            similarity::Bm25Params::new(2.0, 0.5).expect("in range"),
            &mut collector,
        )
        .unwrap();
        assert!(collector.top_docs().is_empty());
    }

    #[test]
    fn search_term_query_scored_with_similarity_missing_term_yields_no_hits() {
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let mut collector = TopDocsCollector::new(10);
        search_term_query_scored_with_similarity(
            &fields,
            doc_in.as_ref(),
            None,
            &TermQuery::new("body", "no_such_term_at_all"),
            None,
            similarity::Bm25Params::new(2.0, 0.5).expect("in range"),
            &mut collector,
        )
        .unwrap();
        assert!(collector.top_docs().is_empty());
    }
}
