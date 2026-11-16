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
//! `query.minimum_should_match` (`query::BooleanQuery`'s doc comment has the
//! full field-level accounting) gates `should` **regardless of
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
//! **Nested `BooleanQuery` clauses**: a `must`/`should`/
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
//! ## Relevance scoring
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

pub mod bayesian_estimator;
mod bit_set_pool;
mod bulk_scorer;
pub mod collector;
pub mod collectors;
pub mod common_terms;
pub mod derived_source;
pub mod directory_reader;
pub mod doc_value_query;
pub mod docid_set;
pub mod document;
pub mod drill_sideways_query;
mod drill_sideways_scorer;
mod exec;
pub use exec::pinned_sort_keys;
/// One segment's readers as the query executor reads them
/// ([`LeafContext::for_spans`]).
pub use exec::LeafContext;
pub mod explain;
pub mod extended_query;
pub mod field_norms;
pub mod function;
pub mod fuzzy_like_this;
pub mod global_ords;
pub mod index_searcher;
pub mod intervals;
pub mod java_math;
pub mod java_priority_queue;
pub mod join;
pub mod knn_collectors;
pub mod knn_recall;
pub mod knn_search;
pub mod leaf_collector;
pub mod live_segment;
pub mod matches;
pub mod memory;
pub mod mlt;
pub mod multi_bits;
pub mod multi_segment;
pub mod multi_terms;
pub mod near_spans;
pub mod nrt_reader;
pub mod ordinal_map;
mod phrase_scorer;
pub mod points_query;
pub mod prefetch;
pub mod profile_search;
pub mod query;
pub mod query_auto_stop_word_analyzer;
pub mod query_builder;
pub mod query_cache;
pub mod query_parser;
pub mod query_visitor;
pub mod reader;
pub mod reader_manager;
pub mod reference_manager;
pub mod rescorer;
pub mod sandbox;
pub mod segment_cacheable;
pub mod segment_match;
pub mod segment_order;
pub mod similarities;
pub mod similarity;
pub mod simpletext_native;
pub mod simpletext_transcode;
pub mod simpletext_writer;
pub mod slices;
pub mod sloppy_phrase;
pub mod soft_deletes;
pub mod spans;
mod sparse_docs_cache;
pub mod term_vectors_query;
pub mod terminate;
#[cfg(any(test, feature = "asserting"))]
pub mod test_support;
pub mod top_docs;
pub mod top_field;
pub mod tree_set;
pub mod values_source;
pub mod vector_query;
pub mod weight_count;
pub mod word2vec;

pub use collector::{
    Collector, CountCollector, FieldValueDoc, MaxScoreAccumulator, ScoreDoc, ScoreMode,
    ScoringCollector, SortDirection, TopDocsCollector, TopFieldCollector, TotalHits,
    TotalHitsRelation, VecCollector,
};
pub use doc_value_query::{
    search_numeric_range, search_numeric_range_sorted_by_field, search_sorted_ord_range,
    sort_by_numeric_doc_value, sort_top_n_by_numeric_doc_value, MissingValue,
};
pub use exec::cache::{sort_column_bytes, SegmentQueryCache};
/// The node-wide budget of every segment's query cache.
pub mod query_cache_budget {
    pub use crate::exec::cache::{node_stats, set_node_budget, DEFAULT_NODE_BYTES};
}
pub use explain::{explain_clause, Explanation};
pub use field_norms::FieldNorms;
pub use multi_segment::{
    merge_multi_segment_scored, merge_multi_segment_scored_after,
    merge_multi_segment_scored_concurrent_shared_max_score, search_boolean_query_multi_segment,
    search_boolean_query_multi_segment_concurrent, search_boolean_query_multi_segment_maxscore,
    search_boolean_query_multi_segment_maxscore_concurrent,
    search_boolean_query_multi_segment_maxscore_counting, search_leaves_shared_counting,
    search_term_query_multi_segment, search_term_query_multi_segment_after,
    search_term_query_multi_segment_concurrent, search_term_query_multi_segment_counting,
    OpenSegment,
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

/// `NodeCommunicationException`: the other node of a replication group
/// could not be reached (`lucene-replicator`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeCommunicationError {
    /// What was being attempted.
    pub when: String,
    /// The failure underneath.
    pub cause: String,
}

impl std::fmt::Display for NodeCommunicationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.when)
    }
}

impl std::error::Error for NodeCommunicationError {}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The request was stopped by its memory bound or cancelled
    /// ([`lucene_util::memory`]).
    #[error(transparent)]
    Memory(#[from] lucene_util::memory::Exceeded),
    /// An aggregation made more buckets than the request allows
    /// (`search.max_buckets`): at least the first number, over the second.
    #[error("too many buckets: at least {0}, over the limit of {1}")]
    TooManyBuckets(usize, usize),
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
    /// Surfaced by [`regexp_doc_ids`] (`Clause::Regexp`) when
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
    /// A [`Clause::PointsRange`] (`field:[min TO max]` in the query parser)
    /// reached [`resolve_clause_docs`]/[`clause_scores`] with no
    /// [`points_query::PointsInput`] supplied -- same "this clause needs an
    /// opened resource this call didn't provide" shape as
    /// [`Error::MissingPosInput`] for a multi-term [`PhraseQuery`], not a
    /// permanent gap: [`crate::explain::explain_clause`] never has a
    /// `PointsInput` to pass and so always surfaces this for a
    /// `Clause::PointsRange` (`explain_clause_with_stats`, which
    /// `IndexSearcher::explain` calls, takes the segment's), but
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
    /// A `Clause::Exists` reached a path that has no segment reader to read
    /// its norms or doc values from -- the materializing and explain paths;
    /// the scorer tree ([`exec`]) runs it with the segment's reader.
    #[error("Clause::Exists needs the segment's reader to execute (field {0:?})")]
    MissingSegmentReader(String),
    /// Surfaced by [`vector_query`] when the underlying `.vemf`/`.vec`/
    /// `.vem`/`.vex` decode fails -- the vector analog of [`Error::Points`].
    /// A *caller* mistake (an unknown field, a wrong-length query vector, a
    /// `k` below 1) is [`Error::InvalidKnnQuery`] instead, deliberately: the
    /// two are indistinguishable to a caller that only sees one error type,
    /// and "the index is corrupt" is the wrong thing to tell a Java caller
    /// who sent a bad request (see `lucene_ffi::vectors`).
    #[error(transparent)]
    Vectors(#[from] lucene_codecs::vectors::Error),
    /// The IVF index of a `IvfVectorsFormat` field.
    #[error(transparent)]
    Ivf(#[from] lucene_ivf::Error),
    /// The disk files of a `DiskHnswVectorsFormat` instance.
    #[error(transparent)]
    DiskHnsw(#[from] lucene_disk_hnsw::Error),
    /// A KNN query this segment cannot answer as asked -- Java's
    /// `IllegalArgumentException` from `AbstractKnnVectorQuery`'s constructor
    /// and its two subclasses' `approximateSearch`, carrying Java's own
    /// message. Never a sign of a damaged index.
    #[error("{0}")]
    InvalidKnnQuery(String),
    /// A query built with arguments its Java constructor rejects
    /// (`IllegalArgumentException`), or that expands past
    /// `IndexSearcher.getMaxClauseCount()` (`TooManyClauses`).
    #[error("{0}")]
    InvalidQuery(String),
    /// [`query_cache::search_term_query_cached`] was handed a `num_docs` the
    /// segment's own postings disagree with: a term's `.doc` file produced a
    /// doc id at or past the bitset that `num_docs` sizes. Java has no
    /// equivalent because its `LRUQueryCache` builds the set from a
    /// `LeafReaderContext` that owns both numbers; here they are two
    /// parameters, and the alternative to reporting the disagreement is
    /// `FixedBitSet::set` writing a ghost bit or panicking (see
    /// `docs/mechanical-gates.md`'s `fixed-bitset-bound` rule).
    /// A `MatchAllDocsQuery` built without a `maxDoc` (`i32::MAX`, as the
    /// JVM's decoded queries carry) reached a segment that did not supply
    /// its own: it would walk 2^31 documents past the segment's end.
    #[error("match-all query has no maxDoc and the segment supplied none")]
    MatchAllWithoutMaxDoc,
    #[error("cached query produced docID={doc_id}, outside the segment's 0..{num_docs}")]
    CachedDocOutOfRange { doc_id: i32, num_docs: usize },
    #[error(transparent)]
    Sort(#[from] top_field::SortError),
    /// A concurrent-search slice names a segment the reader does not have.
    #[error("aggregation slice names segment {segment} of {segments}")]
    SliceOutOfRange { segment: usize, segments: usize },
    /// A `terms` aggregation's field has doc values but not keyword ones.
    #[error("terms aggregation field {0} has non-keyword doc values")]
    TermsAggType(String),
    /// A `lucene_search::document` query refused its input -- Java's
    /// `IllegalArgumentException`/`IllegalStateException` from the document
    /// package.
    #[error("document query: {0}")]
    DocumentQuery(String),
    /// Java's `IllegalArgumentException` from an API precondition (search
    /// infrastructure: `TopDocs.merge`/`rrf`, rescorers, values
    /// sources, the Matches API).
    #[error("{0}")]
    IllegalArgument(String),
    /// Java's `IllegalStateException` (a reference manager used after close,
    /// a collector replayed after its cache overflowed).
    #[error("{0}")]
    IllegalState(String),
    /// Java's `AlreadyClosedException` (a reference manager used after
    /// `close`).
    #[error("{0}")]
    AlreadyClosed(String),
    /// Opening or reopening a [`directory_reader::DirectoryReader`].
    #[error(transparent)]
    DirectoryReader(#[from] directory_reader::Error),
    /// A low-level decode (an `IndexedDISI` region, a doc-values terms
    /// dictionary) read through the [`reader`] layer.
    #[error(transparent)]
    Store(#[from] lucene_store::Error),
    /// A term's postings or positions read through the [`reader`] layer.
    #[error(transparent)]
    Postings(#[from] lucene_codecs::postings::Error),
    /// Stored fields read through the [`reader`] layer.
    #[error(transparent)]
    StoredFields(#[from] lucene_codecs::stored_fields::Error),
    /// `ExitableDirectoryReader.ExitingReaderException`: a
    /// [`reader::exitable::QueryTimeout`] said to stop while a wrapped
    /// reader was being enumerated.
    #[error("{0}")]
    ExitingReader(String),
    /// Java's `UnsupportedOperationException` (a `FilteredTermsEnum` asked
    /// to seek, `ord()` on a dictionary without ordinals).
    #[error("{0}")]
    Unsupported(String),
    /// A Spatial4j / spatial-extras exception (`lucene_spatial4j::spatial4j`):
    /// what a spatial strategy, its queries and value sources throw, with
    /// Java's class ([`lucene_spatial4j::spatial4j::Error::java_class`]).
    #[error(transparent)]
    Spatial(#[from] lucene_spatial4j::spatial4j::Error),
    /// `NodeCommunicationException`: the other node of a replication group
    /// could not be reached.
    #[error(transparent)]
    NodeCommunication(#[from] NodeCommunicationError),
    /// Reading or writing a `segments_N`.
    #[error(transparent)]
    SegmentInfos(#[from] lucene_index::segment_infos::Error),
    /// A file deleter's deletion failed.
    #[error(transparent)]
    FileDeleter(#[from] lucene_index::index_file_deleter::Error),
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
/// (`PForUtil.skip`).
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
        .filter(|&doc_id| live_docs.is_none_or(|bits| bits.get_doc(doc_id)))
        .collect())
}

/// A wildcard-family clause's expansion: the field it targets, the terms it
/// matched, and the sum of their document frequencies.
///
/// The last is what [`stream_constant_score_clause`] weighs its setup cost
/// against, so it is gathered during the term scan rather than by re-seeking
/// each term afterwards.
type Expansion = (String, Vec<(Vec<u8>, blocktree::SeekedTerm)>, i64);

/// The terms a wildcard-family clause expands to, or `None` when the clause is
/// not one of that family (or its field is absent from this segment).
pub(crate) fn expanded_terms(
    fields: &BlockTreeFields,
    clause: &Clause,
) -> Result<Option<Expansion>> {
    let out = match clause {
        // `TermInSetQuery`: each distinct term that is in this segment, in
        // term order, as its `TermsEnum` over the sorted set visits them.
        Clause::TermInSet(q) => {
            let Some(ft) = fields.field(&q.field) else {
                return Ok(None);
            };
            let mut sorted: Vec<&Vec<u8>> = q.terms.iter().collect();
            sorted.sort_unstable();
            sorted.dedup();
            let mut df = 0i64;
            let mut terms = Vec::with_capacity(sorted.len());
            for t in sorted {
                if let Some(seeked) = ft.seek_term_state(t)? {
                    df += seeked.stats.doc_freq as i64;
                    terms.push((t.clone(), seeked));
                }
            }
            (q.field.clone(), terms, df)
        }
        Clause::Prefix(q) => {
            let Some(ft) = fields.field(&q.field) else {
                return Ok(None);
            };
            let pattern = WildcardPattern::prefix(&q.prefix);
            let mut df = 0i64;
            let terms: Vec<(Vec<u8>, blocktree::SeekedTerm)> = ft
                .intersect_states(&pattern)
                .map(|r| {
                    let (t, s) = r?;
                    df += s.stats.doc_freq as i64;
                    Ok((t, s))
                })
                .collect::<lucene_codecs::blocktree::Result<_>>()?;
            (q.field.clone(), terms, df)
        }
        Clause::Wildcard(q) => {
            let Some(ft) = fields.field(&q.field) else {
                return Ok(None);
            };
            let pattern = WildcardPattern::new(&q.pattern);
            let mut df = 0i64;
            let terms: Vec<(Vec<u8>, blocktree::SeekedTerm)> = ft
                .intersect_states(&pattern)
                .map(|r| {
                    let (t, s) = r?;
                    df += s.stats.doc_freq as i64;
                    Ok((t, s))
                })
                .collect::<lucene_codecs::blocktree::Result<_>>()?;
            (q.field.clone(), terms, df)
        }
        Clause::Regexp(q) => {
            let Some(ft) = fields.field(&q.field) else {
                return Ok(None);
            };
            let pattern = RegexpPattern::new(q.pattern.as_bytes())?;
            let mut df = 0i64;
            let terms: Vec<(Vec<u8>, blocktree::SeekedTerm)> = ft
                .regexp_intersect_states(&pattern)
                .map(|r| {
                    let (t, s) = r?;
                    df += s.stats.doc_freq as i64;
                    Ok((t, s))
                })
                .collect::<lucene_codecs::blocktree::Result<_>>()?;
            (q.field.clone(), terms, df)
        }
        _ => return Ok(None),
    };
    Ok(Some(out))
}

/// Whether the multi-term `clause` is one the scorer tree runs through the
/// segment's query cache: a term set of 2 to 16 terms (Lucene's
/// `rewriteAsBooleanQuery`) on a segment the cache takes. Its size bounds its
/// expansion without seeking a term (a lone term present runs in the tree as
/// a plain term, as Lucene's rewrite does). A prefix, wildcard or regexp
/// would have to be expanded to know -- a walk of the term dictionary that
/// is the whole cost of a `regexp`, paid twice -- so those keep streaming,
/// uncached.
fn small_rewrite_is_cached(
    cache: Option<&SegmentQueryCache>,
    max_doc: Option<i32>,
    clause: &Clause,
) -> bool {
    let Clause::TermInSet(q) = clause else {
        return false;
    };
    cache.is_some()
        && max_doc.is_some_and(|m| m >= exec::cache::MIN_SEGMENT_SIZE)
        && q.terms.len() > 1
        && q.terms.len() <= exec::multi_term::BOOLEAN_REWRITE_TERM_COUNT_THRESHOLD
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
    // A collector already full at this constant score, with its threshold
    // published, can take no document from here on: every one ties on score
    // and loses on doc id (a later leaf of a shared collector, or a second
    // pass). Expanding the clause's terms -- thousands of them for a short
    // prefix, the whole cost of such a query -- would buy nothing.
    if collector.constant_score_hits_needed() == Some(0) {
        return Ok(true);
    }
    let Some((field, terms, total_doc_freq)) = expanded_terms(fields, clause)? else {
        return Ok(false);
    };
    let Some(field_terms) = fields.field(&field) else {
        return Ok(false);
    };
    let Some(doc_in) = doc_in else {
        return Ok(false);
    };
    // A collector whose fill size is known needs only the first `need`
    // matching documents in doc-id order -- see `cutoff_constant_score_union`,
    // which reads each term only below a shrinking doc-id cutoff.
    if let Some(need) = collector.constant_score_hits_needed() {
        // A handful of terms (a `terms` query, a narrow prefix): Lucene's
        // rewrite is a constant-scored disjunction of them, and a plain
        // k-way merge that stops at `need` beats the cutoff machinery built
        // for hundreds of expansions.
        if terms.len() <= 16 {
            return small_constant_score_union(
                field_terms,
                &terms,
                doc_in,
                live_docs,
                need,
                collector,
            );
        }
        return cutoff_constant_score_union(
            field_terms,
            &terms,
            doc_in,
            live_docs,
            need,
            collector,
        );
    }
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
    for (_, seeked) in &terms {
        let cursor = field_terms.lazy_postings_for(
            seeked,
            doc_in,
            lucene_codecs::postings::PostingsFlags::DocsOnly,
        )?;
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
        if live_docs.is_none_or(|bits| bits.get_doc(best)) {
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

/// A constant-scoring clause's union when the collector's fill size is known:
/// the first `need` matching documents in doc-id order are the whole answer,
/// so no term needs reading past the point where `need` documents below it
/// are already known.
///
/// Every match scores the same, and the collector breaks a score tie by the
/// lower doc id, so after `need` documents
/// ([`ScoringCollector::constant_score_hits_needed`]) nothing later can enter
/// it. This keeps a set of the matching documents below a *cutoff* and reads
/// each term's postings only while they are below it; whenever the set holds
/// twice what is needed, the cutoff drops to just past the `need`-th document
/// and everything above is forgotten. After the first few terms the cutoff is
/// a few thousand doc ids in, and every later term costs one block.
///
/// Lucene cannot stop early here: `MultiTermQuery`'s constant-score rewrite
/// builds the whole `DocIdSet` (every posting of every expanded term) before
/// the first document is scored, which is what a prefix with thousands of
/// terms pays. This returns the same documents -- the lowest `need` matching
/// doc ids -- without building it.
///
/// Opens one cursor at a time (the old streaming merge held one per term,
/// thousands of multi-kilobyte cursors for a wide prefix, and profiled mostly
/// as page faults), and reads a pulsed single-document term straight out of
/// its term metadata.
/// The first `need` documents of the union of a few terms, in doc-id order,
/// each collected at `1.0`: `BooleanQuery(SHOULD terms)` under
/// `ConstantScoreQuery`, stopped as `ConstantScoreScorer` stops once the
/// collector can take nothing more. A window-at-a-time bitset union
/// (`BooleanScorer`'s shape), each term's bit-set blocks ORed in a word at a
/// time ([`lucene_codecs::postings::LazyDocsCursor::into_window`]), with
/// windows sized to the documents still needed so it reads little past the
/// `need`-th match.
fn small_constant_score_union<C: ScoringCollector>(
    field_terms: &blocktree::FieldTerms,
    terms: &[(Vec<u8>, blocktree::SeekedTerm)],
    doc_in: &DocInput<'_>,
    live_docs: Option<&FixedBitSet>,
    need: u64,
    collector: &mut C,
) -> Result<bool> {
    use lucene_codecs::postings::NO_MORE_DOCS;
    let pe = |e| -> Error { blocktree::Error::Postings(e).into() };
    let mut cursors = Vec::with_capacity(terms.len());
    let mut docs = Vec::with_capacity(terms.len());
    for (_, seeked) in terms {
        let mut c = field_terms.lazy_postings_for(
            seeked,
            doc_in,
            lucene_codecs::postings::PostingsFlags::DocsOnly,
        )?;
        docs.push(c.next_doc().map_err(pe)?);
        cursors.push(c);
    }
    // A window of documents at a time, as `BooleanScorer` does: every term
    // marks its documents in the window's bits, then the bits are collected
    // in order -- deduplicated and sorted without a per-document scan of the
    // terms' heads.
    //
    // The window is sized to what is still needed rather than fixed at
    // `BooleanScorer`'s 4,096: a union of frequent terms finds its `need`
    // documents in the first few hundred doc ids, and every posting read past
    // them is wasted (a fixed window read `regexp body:t1[0-9]` two to three
    // times further than Lucene's document-at-a-time disjunction stops). The
    // first window is `need` documents long -- the union can hold at most one
    // match per doc id -- and each later one is stretched by the density the
    // windows so far showed.
    const WINDOW: i64 = 4096;
    let mut words = [0u64; (WINDOW / 64) as usize];
    let mut left = need;
    let (mut spanned, mut found) = (0i64, 0i64);
    loop {
        let base = docs.iter().copied().min().unwrap_or(NO_MORE_DOCS);
        if base == NO_MORE_DOCS || left == 0 {
            return Ok(true);
        }
        let want = i64::try_from(left).unwrap_or(WINDOW);
        // ARITH: `found <= spanned`, both bounded by the segment's doc count;
        // `want <= WINDOW` after the clamp, so the product stays small.
        #[allow(clippy::arithmetic_side_effects)]
        let win = {
            let want = want.min(WINDOW);
            let est = match (found, spanned) {
                (0, 0) => want,
                (0, _) => WINDOW,
                _ => want * spanned / found + want / 8,
            };
            ((est.clamp(64, WINDOW) + 63) & !63) as i32
        };
        let end = base.saturating_add(win);
        for (d, c) in docs.iter_mut().zip(cursors.iter_mut()) {
            if *d < end {
                *d = c.into_window(base, end, &mut words).map_err(pe)?;
            }
        }
        // ARITH: `end - base <= WINDOW`.
        #[allow(clippy::arithmetic_side_effects)]
        {
            spanned += i64::from(end - base);
            found += words.iter().map(|w| i64::from(w.count_ones())).sum::<i64>();
        }
        for (w, word) in words.iter_mut().enumerate() {
            let mut bits = std::mem::take(word);
            while bits != 0 {
                // ARITH: w < WINDOW / 64 and the bit < 64, so the offset is
                // below WINDOW and base + offset below `end`.
                let doc = base + (w * 64) as i32 + bits.trailing_zeros() as i32;
                bits &= bits - 1;
                if left > 0 && live_docs.is_none_or(|l| l.get_doc(doc)) {
                    collector.collect(doc, 1.0);
                    left -= 1;
                }
            }
        }
    }
}

fn cutoff_constant_score_union<C: ScoringCollector>(
    field_terms: &blocktree::FieldTerms,
    terms: &[(Vec<u8>, blocktree::SeekedTerm)],
    doc_in: &DocInput<'_>,
    live_docs: Option<&FixedBitSet>,
    need: u64,
    collector: &mut C,
) -> Result<bool> {
    let need = usize::try_from(need).unwrap_or(usize::MAX).max(1);
    let mut set = CutoffDocSet::default();
    let mut cutoff = lucene_codecs::postings::NO_MORE_DOCS;
    let add = |set: &mut CutoffDocSet, cutoff: &mut i32, doc: i32| {
        if live_docs.is_none_or(|bits| bits.get_doc(doc))
            && set.insert(doc)
            && set.len() >= need.saturating_mul(2)
        {
            *cutoff = set.shrink_to(need);
        }
    };
    // One cursor, reset per term (`TermsEnum.postings(reuse, ...)`).
    let mut reuse = None;
    for (term, seeked) in terms {
        if seeked.stats.doc_freq <= 1 {
            // Pulsed into the term dictionary: no `.doc` bytes to open lazily.
            if let Some(postings) = field_terms.postings_with_flags(
                term,
                Some(doc_in),
                lucene_codecs::postings::PostingsFlags::DocsOnly,
            )? {
                for &doc in &postings.docs {
                    if doc < cutoff {
                        add(&mut set, &mut cutoff, doc);
                    }
                }
            }
            continue;
        }
        // A rare term starting past the cutoff has nothing to add: learn
        // that from its first delta instead of decoding the whole block.
        if field_terms
            .tail_only_first_doc(seeked, doc_in)?
            .is_some_and(|first| first >= cutoff)
        {
            continue;
        }
        // Opened from the state the expansion walk already decoded: no
        // second dictionary seek per term.
        let cursor = field_terms.reuse_postings_for(
            seeked,
            doc_in,
            lucene_codecs::postings::PostingsFlags::DocsOnly,
            &mut reuse,
        )?;
        // `advance(doc + 1)`, not `next_doc`: this walk reads a term only up
        // to the cutoff -- a dozen documents once the set has filled -- and
        // `next_doc` expands a bit-set block's 256 documents on first use,
        // where `advance` searches its bits. The expansion was 13% of
        // `regexp body:t1[0-9]`.
        let mut doc = -1i32;
        loop {
            doc = cursor
                .advance(doc.saturating_add(1))
                .map_err(blocktree::Error::Postings)?;
            // `NO_MORE_DOCS` is never below the cutoff either.
            if doc >= cutoff {
                break;
            }
            add(&mut set, &mut cutoff, doc);
        }
    }
    let mut done = false;
    set.for_each(|doc| {
        if done {
            return;
        }
        collector.collect(doc, 1.0);
        if collector.pruning_threshold().is_some_and(|s| s >= 1.0) {
            done = true;
        }
    });
    Ok(true)
}

/// The doc-id set [`cutoff_constant_score_union`] keeps: a growable bitset
/// that also knows how many bits are set, and can forget everything above its
/// `n`-th document.
#[derive(Default)]
struct CutoffDocSet {
    words: Vec<u64>,
    len: usize,
}

impl CutoffDocSet {
    /// Sets `doc`'s bit; `true` when it was not already set.
    fn insert(&mut self, doc: i32) -> bool {
        let Ok(idx) = usize::try_from(doc) else {
            return false;
        };
        let (w, bit) = (idx >> 6, 1u64 << (idx & 63));
        if w >= self.words.len() {
            self.words.resize(w + 1, 0);
        }
        if self.words[w] & bit != 0 {
            return false;
        }
        self.words[w] |= bit;
        self.len += 1;
        true
    }

    fn len(&self) -> usize {
        self.len
    }

    /// Keeps only the first `n` documents and returns the new cutoff: one past
    /// the `n`-th, so a document below it is exactly one that could still be
    /// among the first `n`.
    fn shrink_to(&mut self, n: usize) -> i32 {
        let mut remaining = n;
        for w in 0..self.words.len() {
            let ones = self.words[w].count_ones() as usize;
            if ones < remaining {
                remaining -= ones;
                continue;
            }
            // The `remaining`-th set bit of this word ends the kept range.
            let mut word = self.words[w];
            for _ in 1..remaining {
                word &= word - 1;
            }
            let bit = word.trailing_zeros() as usize;
            let keep_mask = if bit == 63 {
                u64::MAX
            } else {
                (1u64 << (bit + 1)) - 1
            };
            self.words[w] &= keep_mask;
            self.words.truncate(w + 1);
            self.len = n;
            return ((w << 6) | bit) as i32 + 1;
        }
        lucene_codecs::postings::NO_MORE_DOCS
    }

    /// Every document in the set, ascending.
    fn for_each(&self, mut f: impl FnMut(i32)) {
        for (w, &word) in self.words.iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                f(((w << 6) | bits.trailing_zeros() as usize) as i32);
                bits &= bits - 1;
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
/// of entries: on `prefix body:t12` over the benchmark corpus the sort alone was 67% of
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
        .map(|r| r.map(|(term, _stats)| term))
        .collect::<lucene_codecs::blocktree::Result<_>>()?;
    let mut acc = DocIdBitSet::default();
    for term in &matching_terms {
        let Some(docs) = term_docs_only(field_terms, term, doc_in)? else {
            continue;
        };
        for doc_id in docs {
            if live_docs.is_none_or(|bits| bits.get_doc(doc_id)) {
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
        .map(|r| r.map(|(term, _stats)| term))
        .collect::<lucene_codecs::blocktree::Result<_>>()?;
    let mut acc = DocIdBitSet::default();
    for term in &matching_terms {
        let Some(docs) = term_docs_only(field_terms, term, doc_in)? else {
            continue;
        };
        for doc_id in docs {
            if live_docs.is_none_or(|bits| bits.get_doc(doc_id)) {
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
    /// `CollectionStatistics.docCount` for the field over whatever scope the
    /// expansion was run on -- one segment for the single-segment path, the
    /// whole reader for [`multi_segment::global_fuzzy_stats`]'s. It travels
    /// with the blended frequency because the two are one `TermStatistics`/
    /// `CollectionStatistics` pair in Java and separating them is how the
    /// per-leaf/reader-wide mismatch got in.
    doc_count: i64,
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
/// Four Lucene behaviours live here (as opposed to keeping the first
/// `max_expansions` terms in term-dictionary order and scoring every match a
/// flat `1.0`):
///
/// 1. **Per-term boost.** `FuzzyTermsEnum.next` sets `BoostAttribute` to
///    `1.0` for an exact match and `1 - ed/min(len(term), len(query term))`
///    otherwise -- see [`lucene_codecs::fuzzy::FuzzyMatch::boost`].
/// 2. **Top-N by boost, not by term order.** `TopTermsRewrite.collect` keeps
///    a size-`maxExpansions` priority queue whose worst element is the lowest
///    boost, with ties broken so the lexicographically **later** term is
///    dropped first (`boost == t.boost && bytes.compareTo(t.bytes.get()) > 0`
///    skips the candidate). [`fuzzy_expanded_terms_pruned`] keeps that queue
///    as a sorted `Vec`, in the final `(boost desc, bytes asc)` order.
/// 3. **Blended document frequency**, see [`FuzzyExpansion::blended_doc_freq`].
/// 4. **The `MaxNonCompetitiveBoostAttribute` feedback loop**: the worst
///    boost still in that queue is fed *back* into the term enumeration, which
///    drops its edit budget as soon as no term at the higher distance could
///    still compete (`FuzzyTermsEnum.bottomChanged`). See
///    [`lucene_codecs::blocktree::FuzzyIntersect::set_max_edits`].
///
/// The whole matching set is still *visited* -- `TopTermsRewrite.rewrite` runs
/// `collectTerms` over every term the automaton accepts -- but from the point
/// the queue fills, each remaining term is tested against a narrower band.
fn fuzzy_expanded_terms(
    field_terms: &lucene_codecs::blocktree::FieldTerms,
    query: &FuzzyQuery,
) -> Result<FuzzyExpansion> {
    fuzzy_expanded_terms_pruned(field_terms, query, true)
}

/// [`fuzzy_expanded_terms`] with the feedback loop switchable, so
/// `examples/fuzzy_pruning.rs` can time both arms in one process from one
/// build. Production always passes `true`; `false` is the pre-c40 behaviour
/// (`FuzzyTermsEnum` with `MaxNonCompetitiveBoostAttribute` never consulted),
/// and it must produce the **same expansion** -- which
/// `pruning_never_changes_the_selected_expansion` asserts.
fn fuzzy_expanded_terms_pruned(
    field_terms: &lucene_codecs::blocktree::FieldTerms,
    query: &FuzzyQuery,
    prune: bool,
) -> Result<FuzzyExpansion> {
    Ok(fuzzy_expanded_terms_instrumented(field_terms, query, prune)?.0)
}

/// [`fuzzy_expanded_terms_pruned`] also reporting the edit budget the walk
/// ended on -- `FuzzyTermsEnum.maxEdits` after the last `bottomChanged`. Only
/// [`bench_only_fuzzy_expansion`] wants it; the search path takes the tuple's
/// first half and drops it.
fn fuzzy_expanded_terms_instrumented(
    field_terms: &lucene_codecs::blocktree::FieldTerms,
    query: &FuzzyQuery,
    prune: bool,
) -> Result<(FuzzyExpansion, u8)> {
    let pattern = FuzzyMatch::new(
        &query.term,
        query.max_edits,
        query.prefix_length,
        query.transpositions,
    );
    let mut collector = FuzzyTopTerms::new(query, &pattern);
    let final_max_edits = collector.collect_segment(field_terms, &pattern, prune)?;
    Ok((
        collector.finish(field_terms.doc_count as i64),
        final_max_edits,
    ))
}

/// The terms [`fuzzy_expanded_terms`] selects, over a one-document leaf
/// whose terms are held in memory (`MemoryIndex`'s, each with `docFreq` 1),
/// in the queue's order.
pub(crate) fn fuzzy_expanded_terms_in_memory<'t>(
    terms: impl Iterator<Item = &'t [u8]>,
    query: &FuzzyQuery,
) -> Vec<Vec<u8>> {
    let pattern = FuzzyMatch::new(
        &query.term,
        query.max_edits,
        query.prefix_length,
        query.transpositions,
    );
    let mut collector = FuzzyTopTerms::new(query, &pattern);
    collector.collect_sorted(terms.map(|t| (t, 1)), &pattern);
    collector.queue.into_iter().map(|e| e.term).collect()
}

/// `TopTermsRewrite.rewrite`'s `stQueue` + `visitedTerms` + the shared
/// `MaxNonCompetitiveBoostAttribute`, as one object that
/// `TermCollectingRewrite.collectTerms` drives **once per leaf**.
///
/// Java's collector is created once and fed every leaf's `TermsEnum` in turn
/// (`for (LeafReaderContext context : topReaderContext.leaves())`), so the
/// queue, the visited-term map and the published max-non-competitive boost are
/// all reader-wide. That is what makes `BlendedTermQuery`'s `df` a reader-wide
/// number, and it is why this type exists rather than a per-segment loop:
/// a fuzzy clause scored from one segment's own `docFreq`/`docCount` is the
/// same defect [`CollectionStats`] exists to fix for `TermQuery`.
struct FuzzyTopTerms {
    /// `Math.min(size, getMaxSize())`.
    max_size: usize,
    /// `FuzzyAutomatonBuilder.getTermLength()`: the whole query term's
    /// codepoint count, which is `bottomChanged`'s denominator.
    term_length: usize,
    /// The query's own `maxEdits`, which every leaf's `FuzzyTermsEnum` starts
    /// from (`this.maxEdits = this.automata.length - 1` in its constructor).
    query_max_edits: u8,
    /// `stQueue`, as a sorted `Vec` of at most `max_size` entries in the final
    /// order (boost descending, then term bytes ascending) rather than a
    /// binary heap keyed on its inverse. Java's `ScoreTerm.compareTo` -- "if
    /// the boosts are equal, `other.bytes.compareTo(this.bytes)`" -- makes
    /// `peek()` the lowest boost and, among equal boosts, the
    /// lexicographically *largest* term; that is exactly the last element
    /// here. A phrase-sized `maxExpansions` (50 by default) makes the sorted
    /// insert cheaper than a heap, and it removes the final sort.
    queue: Vec<ExpandedTerm>,
    /// `MaxNonCompetitiveBoostAttribute`'s two values, published on the
    /// collector's shared `AttributeSource` and therefore surviving from one
    /// leaf's enum to the next. `None` is Java's
    /// `(Float.NEGATIVE_INFINITY, null)` initial state.
    published_bottom: Option<(f32, Vec<u8>)>,
}

impl FuzzyTopTerms {
    fn new(query: &FuzzyQuery, pattern: &FuzzyMatch<'_>) -> Self {
        FuzzyTopTerms {
            max_size: query.max_expansions,
            term_length: pattern.term_length(),
            query_max_edits: query.max_edits,
            queue: Vec::new(),
            published_bottom: None,
        }
    }

    /// `FuzzyTermsEnum.bottomChanged`'s `while` loop against the currently
    /// published bottom, returning the tightened budget.
    ///
    /// `term_after` is Java's
    /// `bottomTerm == null || (lastTerm != null && lastTerm.compareTo(bottomTerm) >= 0)`.
    fn tighten(&self, mut edits: u8, last_term: Option<&[u8]>) -> u8 {
        let (bottom, bottom_term) = match &self.published_bottom {
            // No bottom published yet: `bottom` is `Float.NEGATIVE_INFINITY`,
            // which is below every finite `maxBoost`, so the loop breaks at
            // once and nothing is tightened.
            None => return edits,
            Some((boost, term)) => (*boost, term.as_slice()),
        };
        let term_after = last_term.is_some_and(|t| t >= bottom_term);
        while edits > 0 {
            // `1.0f - ((float) maxEdits / (float) termLength)`. A zero-length
            // query term makes this `-inf`, which never breaks the loop -- the
            // same answer Java's float division gives.
            let max_boost = 1.0f32 - f32::from(edits) / (self.term_length as f32);
            if bottom < max_boost || (bottom == max_boost && !term_after) {
                break;
            }
            edits = edits.saturating_sub(1);
        }
        edits
    }

    /// `collectTerms`' body for one accepted term with its `boost` and
    /// `docFreq`: whether it was inserted into the queue (and so may have
    /// moved its bottom).
    fn offer(&mut self, term: &[u8], boost: f32, doc_freq: i64) -> bool {
        // `collectTerms`' own rejection, *before* the visited-term lookup:
        // once the queue is full, a term that cannot beat the worst entry
        // is dropped without being inserted -- and, in a second leaf,
        // without its `docFreq` being accumulated either.
        // `maxSize == 0` makes the queue full from the start with no
        // bottom to compare against, so nothing is ever competitive --
        // which is the same "select nothing" a truncate-to-zero gives.
        if self.queue.len() == self.max_size {
            match self.queue.last() {
                None => return false,
                Some(bottom) => {
                    if boost < bottom.boost
                        || (boost == bottom.boost && term > bottom.term.as_slice())
                    {
                        return false;
                    }
                }
            }
        }

        // `ScoreTerm t = visitedTerms.get(bytes)`: a term already in the
        // queue only has its `docFreq` accumulated
        // (`t.termState.register(state, ord, docFreq, totalTermFreq)`),
        // and nothing is republished. Only reachable across leaves --
        // within one leaf the term dictionary yields each term once.
        //
        // The queue is sorted by `(boost desc, term asc)` and a term's
        // boost is a function of the term alone, so an entry for it can
        // only sit where it would be inserted: the binary search for the
        // insertion point answers the lookup too, where a scan compared
        // the term against every entry.
        let at = self
            .queue
            .partition_point(|e| e.boost > boost || (e.boost == boost && e.term.as_slice() < term));
        if let Some(entry) = self
            .queue
            .get_mut(at)
            .filter(|e| e.boost == boost && e.term == term)
        {
            entry.doc_freq = entry.doc_freq.saturating_add(doc_freq);
            return false;
        }
        self.queue.insert(
            at,
            ExpandedTerm {
                boost,
                term: term.to_vec(),
                doc_freq,
            },
        );
        // `if (stQueue.size() > maxSize) { st = stQueue.poll(); visitedTerms
        // .remove(st.bytes.get()); st.termState.clear(); }` -- an evicted
        // term loses the `docFreq` earlier leaves contributed, which is
        // Java's behaviour and not an oversight here.
        if self.queue.len() > self.max_size {
            self.queue.pop();
        }
        true
    }

    /// [`Self::collect_segment`] over a leaf whose terms are held in memory
    /// (`MemoryIndex`'s): each of `terms`, in term order with its `docFreq`,
    /// that the pattern accepts within the query's `maxEdits`. Without the
    /// feedback loop, which only prunes the walk and never changes the
    /// selected expansion (`pruning_never_changes_the_selected_expansion`).
    fn collect_sorted<'t>(
        &mut self,
        terms: impl Iterator<Item = (&'t [u8], i64)>,
        pattern: &FuzzyMatch<'_>,
    ) {
        for (term, doc_freq) in terms {
            if let Some(ed) = pattern.edits_within(term, pattern.max_edits()) {
                self.offer(term, pattern.boost_from_edits(term, ed), doc_freq);
            }
        }
    }

    /// One leaf: `collectTerms`' inner `while ((bytes = termsEnum.next()) != null)`
    /// loop over `getTermsEnum(query, terms, collector.attributes)`.
    ///
    /// Returns the edit budget the leaf's walk ended on -- `FuzzyTermsEnum
    /// .maxEdits` after its last `bottomChanged`, which is the only
    /// *observable* of the feedback loop itself.
    fn collect_segment(
        &mut self,
        field_terms: &lucene_codecs::blocktree::FieldTerms,
        pattern: &FuzzyMatch<'_>,
        prune: bool,
    ) -> Result<u8> {
        // `FuzzyTermsEnum`'s constructor: a *fresh* enum per leaf, starting
        // from the query's own `maxEdits`, then immediately
        // `bottom = maxBoostAtt.getMaxNonCompetitiveBoost(); bottomChanged(null)`
        // against the boost an earlier leaf published. `lastTerm` is null
        // there, so `termAfter` is `bottomTerm == null`, i.e. false whenever a
        // bottom exists.
        let mut effective_max_edits = self.query_max_edits;
        let mut walk = field_terms.fuzzy_intersect(pattern);
        if prune {
            let tightened = self.tighten(effective_max_edits, None);
            if tightened != effective_max_edits {
                effective_max_edits = tightened;
                walk.set_max_edits(tightened);
            }
        }
        while let Some(next) = walk.next() {
            let (term, stats) = next?;
            // `FuzzyTermsEnum.next` sets `boostAtt` from the `ed` it has
            // already computed; the walk carries the same distance out so this
            // does not run the banded DP a second time per accepted term.
            let boost = pattern.boost_from_edits(&term, walk.last_edits());

            if !self.offer(&term, boost, stats.doc_freq as i64) {
                continue;
            }

            // `maxBoostAtt.setMaxNonCompetitiveBoost(peek().boost)` +
            // `setCompetitiveTerm(peek().bytes)`, then `FuzzyTermsEnum.next`
            // noticing they changed and running `bottomChanged(lastTerm)`.
            //
            // Two details of *when* Java does this, both reproduced. It
            // publishes only from the insert branch -- `collectTerms` returns
            // before the publish block both when the term was not competitive
            // and when it was already visited -- and `next()` acts only when
            // the published pair actually *changed*, which is the same
            // condition, since an insert into a full queue always moves the
            // bottom. Without that gate this loop would re-run on later,
            // non-competitive terms and could decrement once more than Java
            // does (`termAfter` flips to true as the walk passes the bottom
            // term). That extra step is still result-preserving -- past the
            // bottom term no equal-boost term can displace it -- but it is not
            // what Java does, and "prunes a bit more than Lucene" is not a
            // divergence worth having.
            //
            // Java also delays the swap to the *next* `next()` call so the
            // current term's `docFreq()` stays valid; here the term has already
            // been consumed, so "after this term" is the same point.
            if self.queue.len() == self.max_size {
                let bottom = self.queue.last().expect("non-empty: len == max_size > 0");
                self.published_bottom = Some((bottom.boost, bottom.term.clone()));
                if prune && effective_max_edits > 0 {
                    let tightened = self.tighten(effective_max_edits, Some(&term));
                    if tightened != effective_max_edits {
                        effective_max_edits = tightened;
                        walk.set_max_edits(tightened);
                    }
                }
            }
        }
        Ok(effective_max_edits)
    }

    /// `BlendedTermQuery.rewrite`'s fold: `df = Math.max(df, ctx.docFreq())`
    /// over every selected term, with `doc_count` the matching
    /// `CollectionStatistics.docCount`.
    fn finish(self, doc_count: i64) -> FuzzyExpansion {
        let blended_doc_freq = self.queue.iter().map(|t| t.doc_freq).max().unwrap_or(0);
        FuzzyExpansion {
            terms: self.queue,
            blended_doc_freq,
            doc_count,
        }
    }
}

/// `TopTermsRewrite.rewrite` run over **every leaf**, which is what
/// [`multi_segment::global_fuzzy_stats`] needs and what
/// [`fuzzy_expanded_terms`] deliberately is not.
///
/// `field_terms` is one entry per segment that has the field, in reader order
/// (`topReaderContext.leaves()`'s order, which is what decides the outcome of
/// an equal-boost tie and which leaf's `docFreq` an evicted term loses);
/// `doc_count` is the reader-wide `CollectionStatistics.docCount`, i.e. the
/// same segments' `docCount`s summed.
///
/// Pruning is always on here: it is a Lucene behaviour, not an option, and
/// `fuzzy_expanded_terms_pruned`'s switch exists only so
/// `examples/fuzzy_pruning.rs` can time both arms.
pub(crate) fn fuzzy_expansion_across_leaves(
    field_terms: &[&lucene_codecs::blocktree::FieldTerms],
    query: &FuzzyQuery,
    doc_count: i64,
) -> Result<FuzzyCollectionStats> {
    let pattern = FuzzyMatch::new(
        &query.term,
        query.max_edits,
        query.prefix_length,
        query.transpositions,
    );
    let mut collector = FuzzyTopTerms::new(query, &pattern);
    for leaf in field_terms {
        collector.collect_segment(leaf, &pattern, true)?;
    }
    let expansion = collector.finish(doc_count);
    Ok(FuzzyCollectionStats {
        terms: expansion
            .terms
            .iter()
            .map(|t| (t.term.clone(), t.boost))
            .collect(),
        blended_doc_freq: expansion.blended_doc_freq,
        doc_count: expansion.doc_count,
    })
}

/// One selected term as [`bench_only_fuzzy_expansion`] reports it:
/// `(term bytes, FuzzyTermsEnum's BoostAttribute, docFreq)`.
#[doc(hidden)]
pub type BenchExpandedTerm = (Vec<u8>, f32, i64);

/// Measurement-only entry point for `examples/fuzzy_pruning.rs`: the term
/// expansion [`fuzzy_doc_ids`]/[`fuzzy_doc_scores`] run, with the
/// `MaxNonCompetitiveBoostAttribute` feedback loop switchable so both arms are
/// one build in one process (criterion is unusable on this project's machine).
/// Returns the selected terms with their boosts, the blended document
/// frequency, and the edit budget the walk ended on -- the last of which is the
/// only *observable* of the feedback loop itself, so a test can require that it
/// actually fired.
///
/// `#[doc(hidden)] pub` rather than `#[cfg(feature = "test-support")]`, and
/// deliberately: a feature-gated example is one `cargo clippy --all-targets`
/// cannot build, and this project has already been bitten twice by a
/// measurement target left broken for several batches while a stale binary
/// kept producing plausible numbers (see `scripts/gate.sh`'s note on
/// `benchmarks/rust-runner`). The cost of the alternative is one hidden
/// function that forwards to private code.
#[doc(hidden)]
pub fn bench_only_fuzzy_expansion(
    field_terms: &lucene_codecs::blocktree::FieldTerms,
    query: &FuzzyQuery,
    prune: bool,
) -> Result<(Vec<BenchExpandedTerm>, i64, u8)> {
    let (expansion, final_max_edits) =
        fuzzy_expanded_terms_instrumented(field_terms, query, prune)?;
    Ok((
        expansion
            .terms
            .iter()
            .map(|t| (t.term.clone(), t.boost, t.doc_freq))
            .collect(),
        expansion.blended_doc_freq,
        final_max_edits,
    ))
}

/// [`Clause::Fuzzy`]'s matched doc-ID list (`max_expansions`
/// bounded): same union-across-matching-terms mechanism as
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
    let expansion = fuzzy_expanded_terms(field_terms, query)?;
    let mut acc = DocIdBitSet::default();
    for ExpandedTerm { term, .. } in &expansion.terms {
        let Some(docs) = term_docs_only(field_terms, term, doc_in)? else {
            continue;
        };
        for doc_id in docs {
            if live_docs.is_none_or(|bits| bits.get_doc(doc_id)) {
                acc.set(doc_id);
            }
        }
    }
    let doc_ids = acc.into_sorted_vec();
    Ok(doc_ids)
}

/// A lone [`FuzzyQuery`] scored the way Lucene scores it:
/// `TopTermsBlendedFreqScoringRewrite` makes it a `BlendedTermQuery` whose
/// `BOOLEAN_REWRITE` is a pure-`SHOULD` `BooleanQuery` of `TermQuery`s -- each
/// boosted by its edit distance, all sharing the blended document frequency --
/// and that disjunction is scored by `MaxScoreBulkScorer`
/// ([`bulk_scorer::score_disjunction`]).
///
/// [`fuzzy_doc_scores`] materialized every posting of every selected term
/// (up to `maxExpansions`, 1024 by default) into one list and summed scores
/// through a `HashMap<i32, f32>`, so a top-10 fuzzy query paid for every
/// matching document of every term. Here the terms whose block maxima cannot
/// reach the threshold become non-essential and are only advanced to the
/// candidates the others produce, and scores are summed in `f64` as Lucene
/// sums them.
///
/// A pulsed single-document term takes part as a one-document cursor
/// ([`lucene_codecs::blocktree::FieldTerms::lazy_postings_for`]).
fn try_fuzzy_bulk<C: ScoringCollector>(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &FuzzyQuery,
    norms: Option<&FieldNorms<'_>>,
    global: Option<&GlobalStats>,
    collector: &mut C,
) -> Result<bool> {
    let Some(doc_in) = doc_in else {
        return Ok(false);
    };
    let Some(field_terms) = fields.field(&query.field) else {
        return Ok(true); // no such field: nothing matches
    };
    let (selected, blended_doc_freq, doc_count) = match global.and_then(|g| g.fuzzy(query)) {
        Some(g) => (g.terms.clone(), g.blended_doc_freq, g.doc_count),
        None => {
            let expansion = fuzzy_expanded_terms(field_terms, query)?;
            (
                expansion
                    .terms
                    .iter()
                    .map(|t| (t.term.clone(), t.boost))
                    .collect::<Vec<_>>(),
                expansion.blended_doc_freq,
                expansion.doc_count,
            )
        }
    };
    let idf = similarity::idf(blended_doc_freq, doc_count);
    let mut legs: Vec<bulk_scorer::TermLeg<'_>> = Vec::with_capacity(selected.len());
    for (term, raw_boost) in &selected {
        let Some(seeked) = field_terms.seek_term_state(term)? else {
            continue;
        };
        let stats = seeked.stats;
        // A pulsed single-document term opens as a one-document cursor
        // (`lazy_postings_for`), so it is a leg like any other.
        let cursor = field_terms.lazy_postings_for(
            &seeked,
            doc_in,
            lucene_codecs::postings::PostingsFlags::Freqs,
        )?;
        legs.push(bulk_scorer::TermLeg::scoring(
            cursor,
            // `BoostQuery` folds the boost into the similarity weight.
            raw_boost.max(0.0) * idf,
            norms.map(|n| n.cursor()),
            stats.doc_freq as i64,
            (stats.total_term_freq - stats.doc_freq as i64 + 1).max(1) as f32,
        ));
    }
    bulk_scorer::score_disjunction(&mut legs, live_docs, collector)?;
    Ok(true)
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
/// A flat `1.0` per fuzzy hit would change the top-k of every fuzzy query.
///
/// The blended `df`, the selected term set and the `docCount` are all
/// *reader-wide* statistics in Lucene: `TopTermsRewrite.rewrite` drives one
/// collector across every leaf and `BlendedTermQuery.rewrite` then folds a
/// single `df` over the result. `global` carries them
/// ([`multi_segment::global_fuzzy_stats`]); without it this expands and blends
/// within the one segment it is given, which is the right answer for a
/// single-segment index and the same per-leaf/reader-wide divergence
/// [`CollectionStats`] documents for `TermQuery` on any other.
fn fuzzy_doc_scores(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &FuzzyQuery,
    norms: Option<&FieldNorms<'_>>,
    global: Option<&GlobalStats>,
) -> Result<HashMap<i32, f32>> {
    let mut scores: HashMap<i32, f32> = HashMap::new();
    let Some(field_terms) = fields.field(&query.field) else {
        return Ok(scores);
    };
    // Reader-wide expansion where the caller gathered one, this segment's own
    // otherwise. The terms are the *selection*, not just the statistics: which
    // terms win the `maxExpansions` queue is itself a reader-wide decision.
    let (selected, blended_doc_freq, doc_count) = match global.and_then(|g| g.fuzzy(query)) {
        Some(g) => (g.terms.clone(), g.blended_doc_freq, g.doc_count),
        None => {
            let expansion = fuzzy_expanded_terms(field_terms, query)?;
            (
                expansion
                    .terms
                    .iter()
                    .map(|t| (t.term.clone(), t.boost))
                    .collect(),
                expansion.blended_doc_freq,
                expansion.doc_count,
            )
        }
    };

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
    for (term, raw_boost) in &selected {
        let boost = raw_boost.max(0.0);
        let Some(postings) = field_terms.postings(term, doc_in)? else {
            continue;
        };
        for (&doc_id, &freq) in postings.docs.iter().zip(postings.freqs.iter()) {
            if live_docs.is_none_or(|bits| bits.get_doc(doc_id)) {
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

    // `BM25Similarity.scorer(boost, ...)` folds the boost into the *weight*
    // -- `this.weight = boost * idf.getValue().floatValue()` -- and
    // `BM25Scorer.score` then evaluates `weight - weight / (1 + freq *
    // normInverse)`. Multiplying the boost into the finished score instead is
    // the algebraically-equal, bitwise-different rewrite `similarity::do_score`
    // exists to warn about: it was one ULP out on this fixture's doc 2.
    let idf = similarity::idf(blended_doc_freq, doc_count);
    let mut norms_cursor = norms.map(|n| n.cursor());
    for (doc_id, freq, boost) in contributions {
        let norm_inverse = match norms_cursor.as_mut() {
            Some(nc) => nc.norm_inverse(doc_id)?,
            None => similarity::UNNORMED_NORM_INVERSE,
        };
        let score = similarity::do_score(boost * idf, freq as f32, norm_inverse);
        *scores.entry(doc_id).or_insert(0.0) += score;
    }
    Ok(scores)
}

/// [`Clause::Regexp`]'s matched doc-ID list: same
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
        .map(|r| r.map(|(term, _stats)| term))
        .collect::<lucene_codecs::blocktree::Result<_>>()?;
    let mut acc = DocIdBitSet::default();
    for term in &matching_terms {
        let Some(docs) = term_docs_only(field_terms, term, doc_in)? else {
            continue;
        };
        for doc_id in docs {
            if live_docs.is_none_or(|bits| bits.get_doc(doc_id)) {
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
            if live_docs.is_none_or(|bits| bits.get_doc(doc_id)) {
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
/// term (see [`resolve_clause_docs`]). `None` is fine for
/// a query with no multi-term phrase clause; passing `None` for a query that
/// turns out to need it surfaces as [`Error::MissingPosInput`], same convention
/// as [`search_phrase_query`].
///
/// `points`: the segment's opened `.kdm`/`.kdi`/`.kdd`
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
/// - `Clause::Phrase`: collects [`search_phrase_query`]'s
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
        Clause::Exists(q) => Err(crate::Error::MissingSegmentReader(q.field.clone())),
        Clause::Extended(q) => Ok(exec::extended::resolve(
            fields, doc_in, pos_in, pay_in, live_docs, points, None, None, q, false,
        )?
        .into_iter()
        .map(|(d, _)| d)
        .collect()),
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

/// [`Clause::PointsRange`]'s matched doc-ID list:
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
    // A field with no points in this segment (`PointValues` is `null`).
    if points.reader.field(field_number).is_none() {
        return Ok(Vec::new());
    }
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
        .filter(|&doc_id| live_docs.is_none_or(|bits| bits.get_doc(doc_id)))
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
/// [`search_boolean_query_scored`] (shared so `minimum_should_match` handling is
/// implemented once) — see this module's doc comment for the exact semantics, including the
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
        .filter(|&(doc_id, _)| live_docs.is_none_or(|bits| bits.get_doc(doc_id)))
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
        .and_then(|g| g.term(&query.field, &query.term))
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
    let Some(stats) = field_terms.try_seek_exact(&query.term)? else {
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
/// See [`similarity::Bm25Params`]'s doc comment for scope.
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
    let Some(stats) = field_terms.try_seek_exact(&query.term)? else {
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
/// defaults. This is the single most fundamental scored search entry point (a
/// plain `TermQuery`, no MAXSCORE pruning), a sibling function rather than an
/// added parameter on [`search_term_query_scored`], so that function's
/// signature and behavior stay unchanged for its callers.
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
/// hardcoded to the defaults: threading custom `k1`/`b` through MAXSCORE
/// pruning, multi-segment fan-out, and phrase scoring is a materially larger
/// change.
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
pub fn search_term_query_scored_maxscore_with_stats<C: ScoringCollector>(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &TermQuery,
    norms: Option<&FieldNorms<'_>>,
    global: Option<CollectionStats>,
    collector: &mut C,
) -> Result<()> {
    let Some(field_terms) = fields.field(&query.field) else {
        return Ok(());
    };
    // One trie walk for both the statistics and the postings pointers.
    let Some(seeked) = field_terms.seek_term_state(&query.term)? else {
        return Ok(());
    };
    let stats = seeked.stats;
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
    let flags = lucene_codecs::postings::PostingsFlags::Freqs;
    let cursor = match field_terms.lazy_postings_for(&seeked, doc_in, flags) {
        Ok(c) => c,
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
    // The highest frequency any one document can have for this term, which
    // bounds every score from above (`MaxScoreCache.globalMaxScore`, but
    // tight). Both statistics must describe the SAME postings: `doc_freq`
    // above may be the reader-wide value while `total_term_freq` is this
    // segment's, and mixing them makes the difference negative -- clamped to
    // 1, the bound collapses and the early exit fires on documents that should
    // have been collected. For a field indexed without frequencies
    // `totalTermFreq` aliases `docFreq`, so this is exactly 1 -- which is what
    // lets a keyword field's scan stop as soon as the top-k fills instead of
    // reading every posting.
    let max_freq = (stats.total_term_freq - stats.doc_freq as i64 + 1).max(1) as f32;
    // `BatchScoreBulkScorer`: a block of postings at a time, scored in one
    // vectorized pass, with `ImpactsDISI` skipping every block whose impacts
    // cannot beat the threshold before it is decoded. See `bulk_scorer`.
    let mut leg = bulk_scorer::TermLeg::scoring(
        cursor,
        similarity::idf(doc_freq, doc_count),
        norms.map(|n| n.cursor()),
        stats.doc_freq as i64,
        max_freq,
    );
    bulk_scorer::score_term(&mut leg, live_docs, collector)
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
    /// [`count`] after to read how many blocks that search skipped.
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
/// - `Clause::Phrase`: this clause's own BM25 score per
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
        Clause::Exists(q) => Err(crate::Error::MissingSegmentReader(q.field.clone())),
        Clause::Extended(q) => Ok(exec::extended::resolve(
            fields, doc_in, pos_in, pay_in, live_docs, points, norms, global, q, true,
        )?
        .into_iter()
        .collect()),
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
            fuzzy_doc_scores(fields, doc_in, live_docs, query, clause_norms, global)
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
        // `SpanWeight`/`SpanScorer` under BM25: see `exec::span`.
        Clause::Span(query) => Ok(exec::span::resolve_span(
            fields, doc_in, pos_in, pay_in, live_docs, norms, global, query,
        )?
        .into_iter()
        .collect()),
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
        let scores = per_disjunct
            .iter()
            .filter_map(|scores| scores.get(&doc_id).copied());
        result.insert(doc_id, dismax_value(scores, query.tie_breaker));
    }
    Ok(result)
}

/// `DisjunctionMaxScorer.score`'s and `DisjunctionMaxWeight.explain`'s
/// arithmetic over the matching disjuncts' scores, in doubles as Java's
/// (`max = 0`, `double otherScoreSum = 0`; `(float) (max + otherScoreSum *
/// tieBreakerMultiplier)`): in floats, `28344.309 + 0.5 * 0.2834431` rounds
/// one ulp below Java's score.
pub(crate) fn dismax_value(scores: impl Iterator<Item = f32>, tie_breaker: f32) -> f32 {
    let mut max = 0.0f64;
    let mut other_sum = 0.0f64;
    for score in scores.map(f64::from) {
        if score >= max {
            other_sum += max;
            max = score;
        } else {
            other_sum += score;
        }
    }
    #[allow(clippy::cast_possible_truncation)]
    let value = (max + other_sum * f64::from(tie_breaker)) as f32;
    value
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
/// On the 15-segment corpus that made `body:t0`'s idf range 0.000473 to
/// 0.000763 -- a 1.6x spread against the global 0.000574 -- so the top-k filled
/// from whichever segment happened to make the term look rarest, and every one
/// of the 20 benchmark queries disagreed with Java on its hit set. It is
/// invisible on a single segment, which is why the merged corpus agreed exactly
/// and no fixture caught it: every fixture is one segment.
///
/// Beside the two numbers BM25 reads, it carries the rest of what
/// `IndexSearcher.termStatistics`/`collectionStatistics` hand any
/// [`similarities::Similarity`]: the term's `totalTermFreq` and the field's
/// `maxDoc`, `sumTotalTermFreq` and `sumDocFreq`, each summed the same way.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CollectionStats {
    /// Number of documents containing the term, summed across every segment.
    pub doc_freq: i64,
    /// Number of documents that have the field, summed across every segment.
    pub doc_count: i64,
    /// The term's occurrences, summed across every segment
    /// (`TermStates.totalTermFreq`).
    pub total_term_freq: i64,
    /// `IndexReader.maxDoc()`: every segment's `maxDoc`, deleted documents
    /// included.
    pub max_doc: i64,
    /// The field's tokens (`Terms.getSumTotalTermFreq`), summed.
    pub sum_total_term_freq: i64,
    /// The field's postings (`Terms.getSumDocFreq`), summed.
    pub sum_doc_freq: i64,
}

impl CollectionStats {
    /// `IndexSearcher.collectionStatistics(field)`, as a similarity takes it.
    /// The counts come from a valid index, so `CollectionStatistics`' own
    /// invariants hold; they are clamped rather than checked so a statistic
    /// a caller built by hand with only `doc_freq`/`doc_count` still scores.
    pub fn collection_statistics(&self) -> similarities::CollectionStatistics {
        let doc_count = self.doc_count.max(1);
        let sum_doc_freq = self.sum_doc_freq.max(doc_count);
        similarities::CollectionStatistics {
            max_doc: self.max_doc.max(doc_count),
            doc_count,
            sum_total_term_freq: self.sum_total_term_freq.max(sum_doc_freq),
            sum_doc_freq,
        }
    }

    /// `IndexSearcher.termStatistics(term, docFreq, totalTermFreq)`, clamped
    /// like [`Self::collection_statistics`]; `term` is `term.bytes()`.
    pub fn term_statistics(&self, term: &[u8]) -> similarities::TermStatistics {
        let doc_freq = self.doc_freq.max(1);
        similarities::TermStatistics {
            doc_freq,
            total_term_freq: self.total_term_freq.max(doc_freq),
            term: Some(std::sync::Arc::from(term)),
        }
    }
}

/// Reader-wide statistics for one [`FuzzyQuery`] clause: the expansion
/// `MultiTermQuery.TopTermsBlendedFreqScoringRewrite` produces over the
/// **whole reader**, not one leaf.
///
/// Java's `TopTermsRewrite.rewrite` drives one collector across every leaf's
/// `TermsEnum` (`TermCollectingRewrite.collectTerms`), so the selected terms,
/// their boosts and each one's `docFreq` are all reader-wide; then
/// `BlendedTermQuery.rewrite` takes `df = max` over them and every clause
/// scores with that one number against the reader-wide
/// `CollectionStatistics.docCount`. Expanding per segment instead selects a
/// *different term set* in each leaf and blends a different frequency -- the
/// multi-term twin of the defect [`CollectionStats`] documents.
#[derive(Debug, Clone, PartialEq)]
pub struct FuzzyCollectionStats {
    /// The selected terms and their `FuzzyTermsEnum` boosts, in
    /// `TopTermsRewrite`'s final order (boost descending, then term bytes
    /// ascending). Boosts are **not** clamped -- `TopTermsRewrite.build` does
    /// `Math.max(0.0f, st.boost)` when it turns the queue into clauses, and so
    /// does the scorer.
    pub terms: Vec<(Vec<u8>, f32)>,
    /// `BlendedTermQuery.rewrite`'s `df = Math.max(df, ctx.docFreq())` over
    /// every selected term's reader-wide `docFreq`.
    pub blended_doc_freq: i64,
    /// `CollectionStatistics.docCount` for the field, summed over every
    /// segment that has it.
    pub doc_count: i64,
}

/// Reader-wide statistics for the terms and fuzzy clauses a query mentions.
/// Built once per search by the multi-segment layer.
///
/// Two maps rather than one, because a fuzzy clause's statistic is not a
/// term's: it is keyed by the *query* (a fuzzy clause and a `TermQuery` on the
/// same `(field, term)` can appear in one boolean query and mean different
/// things), and it carries the whole expansion rather than a single
/// `docFreq`/`docCount` pair.
/// One term's entry in [`GlobalStats`].
#[derive(Debug, Clone, PartialEq)]
struct TermEntry {
    stats: CollectionStats,
    states: Vec<(usize, Option<lucene_codecs::blocktree::SeekedTerm>)>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct GlobalStats {
    /// By field, then term: the reader-wide statistics, and where gathering
    /// them found the term in each segment.
    terms: HashMap<String, HashMap<Vec<u8>, TermEntry>>,
    fuzzy: HashMap<FuzzyQuery, FuzzyCollectionStats>,
    /// A multi-term query's reader-wide rewrite (`TermCollectingRewrite`),
    /// keyed by the query's `Debug` form; see
    /// [`crate::exec::extended::rewrite_multi_term`].
    extended: HashMap<String, Clause>,
    /// The function queries' reader-wide state: each value source's
    /// `createWeight` context and each values source's rewrite.
    functions: function::FunctionStats,
}

impl GlobalStats {
    /// An empty set of statistics -- every lookup falls back to the segment's
    /// own counters, which is what a single-segment search wants.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records one term's reader-wide `docFreq`/`docCount`.
    pub fn insert_term(
        &mut self,
        field: impl Into<String>,
        term: impl Into<Vec<u8>>,
        stats: CollectionStats,
    ) {
        self.terms.entry(field.into()).or_default().insert(
            term.into(),
            TermEntry {
                stats,
                states: Vec::new(),
            },
        );
    }

    /// [`Self::insert_term`] with the term's state in each segment the
    /// statistics were summed over (keyed by the segment's
    /// [`BlockTreeFields`]' address; `None` where the term is absent) -- what
    /// Lucene's `TermStates` keeps so `TermWeight.scorer` never seeks a term
    /// twice.
    pub(crate) fn insert_term_states(
        &mut self,
        field: String,
        term: Vec<u8>,
        stats: CollectionStats,
        states: Vec<(usize, Option<lucene_codecs::blocktree::SeekedTerm>)>,
    ) {
        self.terms
            .entry(field)
            .or_default()
            .insert(term, TermEntry { stats, states });
    }

    /// The term's state in the segment whose fields are `fields`, as the
    /// statistics pass found it: `Some(None)` when it is absent there, `None`
    /// when that segment was not recorded (the caller seeks).
    pub(crate) fn term_state(
        &self,
        field: &str,
        term: &[u8],
        fields: &BlockTreeFields,
    ) -> Option<Option<lucene_codecs::blocktree::SeekedTerm>> {
        let addr = std::ptr::from_ref(fields).addr();
        self.terms
            .get(field)?
            .get(term)?
            .states
            .iter()
            .find(|(a, _)| *a == addr)
            .map(|(_, s)| *s)
    }

    /// Replaces every gathered term's statistics with what `source` says
    /// (`IndexSearcher.termStatistics` / `collectionStatistics` overridden):
    /// each field's collection statistics once, from the reader's, then
    /// each term the reader holds (a term absent here has no
    /// `TermStatistics` to ask about, as in Java, and matches nothing here).
    ///
    /// # Errors
    /// Whatever `source` reports.
    pub(crate) fn override_statistics(
        &mut self,
        source: &dyn index_searcher::StatisticsSource,
    ) -> Result<()> {
        for (field, terms) in &mut self.terms {
            let Some(local) = terms.values().map(|e| e.stats).find(|s| s.max_doc > 0) else {
                continue;
            };
            let local = similarities::CollectionStatistics {
                max_doc: local.max_doc,
                doc_count: local.doc_count,
                sum_total_term_freq: local.sum_total_term_freq,
                sum_doc_freq: local.sum_doc_freq,
            };
            let collection = source
                .collection_statistics(field, Some(local))?
                .unwrap_or(local);
            for (term, entry) in terms.iter_mut() {
                let stats = &mut entry.stats;
                if stats.doc_freq <= 0 {
                    continue;
                }
                let t = source.term_statistics(
                    field,
                    term,
                    similarities::TermStatistics {
                        doc_freq: stats.doc_freq,
                        total_term_freq: stats.total_term_freq,
                        term: Some(std::sync::Arc::from(term.as_slice())),
                    },
                )?;
                *stats = CollectionStats {
                    doc_freq: t.doc_freq,
                    doc_count: collection.doc_count,
                    total_term_freq: t.total_term_freq,
                    max_doc: collection.max_doc,
                    sum_total_term_freq: collection.sum_total_term_freq,
                    sum_doc_freq: collection.sum_doc_freq,
                };
            }
        }
        Ok(())
    }

    /// One term's reader-wide statistics, or `None` when the caller did not
    /// gather them (in which case the segment's own are already right).
    pub fn term(&self, field: &str, term: &[u8]) -> Option<&CollectionStats> {
        self.terms.get(field)?.get(term).map(|e| &e.stats)
    }

    /// Every fuzzy clause's recorded expansion.
    pub(crate) fn fuzzy_entries(
        &self,
    ) -> impl Iterator<Item = (&FuzzyQuery, &FuzzyCollectionStats)> {
        self.fuzzy.iter()
    }

    /// Records a multi-term query's reader-wide rewrite under `key`.
    pub(crate) fn insert_extended_rewrite(&mut self, key: String, rewritten: Clause) {
        self.extended.insert(key, rewritten);
    }

    /// The reader-wide rewrite recorded under `key`, if any.
    pub(crate) fn extended_rewrite(&self, key: &str) -> Option<&Clause> {
        self.extended.get(key)
    }

    /// Records one fuzzy clause's reader-wide expansion.
    pub fn insert_fuzzy(&mut self, query: FuzzyQuery, stats: FuzzyCollectionStats) {
        self.fuzzy.insert(query, stats);
    }

    /// One fuzzy clause's reader-wide expansion, or `None` when the caller did
    /// not gather it.
    pub fn fuzzy(&self, query: &FuzzyQuery) -> Option<&FuzzyCollectionStats> {
        self.fuzzy.get(query)
    }

    /// True when nothing was gathered at all.
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty() && self.fuzzy.is_empty() && self.functions.is_empty()
    }

    /// The function queries' state the statistics pass computed.
    pub(crate) fn functions(&self) -> &function::FunctionStats {
        &self.functions
    }

    /// Records the function queries' state.
    pub(crate) fn set_functions(&mut self, functions: function::FunctionStats) {
        self.functions = functions;
    }
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
    search_boolean_query_scored_impl(
        fields, doc_in, pos_in, pay_in, live_docs, points, query, norms, global, None, None, None,
        collector,
    )
}

/// [`search_boolean_query_scored_with_stats`] over one segment of a
/// multi-segment search, which also knows the segment's `maxDoc` (for a
/// `MatchAllDocsQuery` clause; see [`multi_segment::OpenSegment::max_doc`]).
pub(crate) fn search_boolean_query_scored_segment<C: ScoringCollector>(
    seg: &multi_segment::OpenSegment<'_>,
    query: &BooleanQuery,
    norms: Option<&HashMap<String, FieldNorms<'_>>>,
    global: Option<&GlobalStats>,
    collector: &mut C,
) -> Result<()> {
    // The searcher's own similarity skips the BM25 shortcuts.
    if let (Some(similarity), Some(global)) = (seg.similarity, global) {
        return search_boolean_query_scored_segment_with_similarity(
            seg, query, norms, global, similarity, collector,
        );
    }
    let max_doc = seg.max_doc;
    search_boolean_query_scored_impl(
        seg.fields,
        seg.doc_in,
        seg.pos_in,
        seg.pay_in,
        seg.live_docs,
        seg.points,
        query,
        norms,
        global,
        max_doc,
        seg.cache,
        seg.reader,
        collector,
    )
}

/// [`search_boolean_query_scored_segment`] under a similarity other than the
/// default BM25 (`IndexSearcher.setSimilarity`): straight to the scorer tree,
/// whose term and phrase clauses score through `similarity`'s
/// [`similarities::SimScorer`]s from the reader-wide statistics in `global`.
/// None of the single-shape shortcuts above it apply: each scores BM25.
pub(crate) fn search_boolean_query_scored_segment_with_similarity<C: ScoringCollector>(
    seg: &multi_segment::OpenSegment<'_>,
    query: &BooleanQuery,
    norms: Option<&HashMap<String, FieldNorms<'_>>>,
    global: &GlobalStats,
    similarity: &dyn similarities::Similarity,
    collector: &mut C,
) -> Result<()> {
    let ctx = exec::LeafContext {
        fields: seg.fields,
        doc_in: seg.doc_in,
        pos_in: seg.pos_in,
        pay_in: seg.pay_in,
        live_docs: seg.live_docs,
        points: seg.points,
        norms,
        global: Some(global),
        max_doc: seg.max_doc,
        cache: seg.cache,
        reader: seg.reader,
        similarity: Some(similarity),
    };
    let mode = exec::Mode::of(collector);
    if let Some(mut bulk) = exec::bulk_boolean(&ctx, query, 1.0, mode)? {
        exec::score_segment(&mut bulk, mode, seg.live_docs, collector, ctx.max_doc)?;
    }
    Ok(())
}

/// `IndexSearcher.searchLeaf` under `setTimeout(queryTimeout)`: the scorer
/// tree's bulk scorer wrapped in a `TimeLimitingBulkScorer`, which asks
/// `should_exit` before each window of documents
/// ([`exec::score_segment_time_limited`]). `similarity` `None` is the default
/// BM25. Returns whether the timeout cut this segment short
/// (`TimeExceededException`); the hits collected before it stand.
pub(crate) fn search_boolean_query_scored_segment_time_limited<C: ScoringCollector>(
    seg: &multi_segment::OpenSegment<'_>,
    query: &BooleanQuery,
    norms: Option<&HashMap<String, FieldNorms<'_>>>,
    global: &GlobalStats,
    similarity: Option<&dyn similarities::Similarity>,
    collector: &mut C,
    should_exit: impl FnMut() -> bool,
) -> Result<bool> {
    let ctx = exec::LeafContext {
        fields: seg.fields,
        doc_in: seg.doc_in,
        pos_in: seg.pos_in,
        pay_in: seg.pay_in,
        live_docs: seg.live_docs,
        points: seg.points,
        norms,
        global: Some(global),
        max_doc: seg.max_doc,
        cache: seg.cache,
        reader: seg.reader,
        similarity: similarity.or(seg.similarity),
    };
    let mode = exec::Mode::of(collector);
    match exec::bulk_boolean(&ctx, query, 1.0, mode)? {
        Some(mut bulk) => exec::score_segment_time_limited(
            &mut bulk,
            mode,
            seg.live_docs,
            collector,
            seg.max_doc,
            should_exit,
        ),
        None => Ok(false),
    }
}

/// `Weight.count`-free counting: how many live documents of one segment
/// `query` matches, run on the same scorer tree and bulk scorers as the
/// scored search, in `COMPLETE_NO_SCORES` -- frequencies never decoded, no
/// score computed, no pruning. A query holding a function query needs its
/// reader-wide preparation: [`count_boolean_query_segment_with`].
pub fn count_boolean_query_segment(
    seg: &multi_segment::OpenSegment<'_>,
    query: &BooleanQuery,
) -> Result<u64> {
    count_boolean_query_segment_with(seg, query, None)
}

/// `query` as `IndexSearcher.rewrite(query, needsScores = false)` leaves it
/// for a search that takes no scores (a count, a sort by field): wrapped in a
/// `ConstantScoreQuery`, whose rewrite takes the boost off every clause of a
/// boolean. Only a `min_score`d function score can see the difference (it
/// scores its wrapped query whatever the search takes), so any other query
/// is returned as it is.
pub fn without_scores_rewrite(query: &BooleanQuery) -> std::borrow::Cow<'_, BooleanQuery> {
    if function::holds_min_score(query) {
        std::borrow::Cow::Owned(count_unboosted(query))
    } else {
        std::borrow::Cow::Borrowed(query)
    }
}

/// The boosts off the clauses of every boolean of `query`, which it reaches
/// through the clauses that are themselves booleans.
fn count_unboosted(query: &BooleanQuery) -> BooleanQuery {
    fn clause(c: &Clause) -> Clause {
        match c {
            Clause::Boost(b) => clause(&b.inner),
            Clause::Boolean(inner) => Clause::Boolean(Box::new(count_unboosted(inner))),
            other => other.clone(),
        }
    }
    let mut out = query.clone();
    for list in [
        &mut out.must,
        &mut out.should,
        &mut out.filter,
        &mut out.must_not,
    ] {
        for c in list.iter_mut() {
            *c = clause(c);
        }
    }
    out
}

/// The reader-wide preparation of `query`'s function queries over every
/// segment of the reader (Java creates each weight over the whole
/// searcher), for [`count_boolean_query_segment_with`]; `None` when it holds
/// none.
///
/// # Errors
/// A function source's preparation failing.
pub fn count_preparation(
    segments: &[multi_segment::OpenSegment<'_>],
    query: &BooleanQuery,
) -> Result<Option<GlobalStats>> {
    multi_segment::global_function_stats(segments, query, None)
}

/// [`count_boolean_query_segment`] with `global`, the reader's
/// [`count_preparation`] of `query`.
pub fn count_boolean_query_segment_with(
    seg: &multi_segment::OpenSegment<'_>,
    query: &BooleanQuery,
    global: Option<&GlobalStats>,
) -> Result<u64> {
    let query = &*without_scores_rewrite(query);
    struct Count(u64);
    impl ScoringCollector for Count {
        fn collect(&mut self, _doc_id: i32, _score: f32) {
            self.0 += 1;
        }
        fn score_mode(&self) -> collector::ScoreMode {
            collector::ScoreMode::CompleteNoScores
        }
        fn add_hits(&mut self, n: u64) -> bool {
            self.0 = self.0.saturating_add(n);
            true
        }
        fn collect_range(&mut self, min: i32, max: i32, _score: f32) {
            self.0 = self
                .0
                .saturating_add(u64::try_from(max.saturating_sub(min)).unwrap_or(0));
        }
    }
    // A count scores nothing, but the queries a function scores (a
    // `min_score`'s wrapped query) score with the reader's norms.
    let mut norms = HashMap::new();
    if let (Some(g), Some(reader)) = (global, seg.reader) {
        for (field, avg) in g.functions().avg_lengths() {
            if let Some(n) = reader.field_norms_with_avg_field_length(field, *avg)? {
                norms.insert(field.clone(), n);
            }
        }
    }
    let ctx = exec::LeafContext {
        fields: seg.fields,
        doc_in: seg.doc_in,
        pos_in: seg.pos_in,
        pay_in: seg.pay_in,
        live_docs: seg.live_docs,
        points: seg.points,
        norms: (!norms.is_empty()).then_some(&norms),
        global,
        max_doc: seg.max_doc,
        cache: seg.cache,
        reader: seg.reader,
        similarity: seg.similarity,
    };
    if let Some(n) = count_boolean_query_shortcut(seg, query)? {
        return Ok(n);
    }
    // `CachingWrapperWeight`: a count's weight is built without scores, so
    // the whole query asks the segment's query cache before it runs.
    if let (Some(cache), Some(max_doc)) = (ctx.cache, ctx.max_doc) {
        let clause = Clause::Boolean(Box::new(query.clone()));
        let cacheable = segment_cacheable::is_cacheable(&clause, ctx.reader);
        // The core's matches, before deletions.
        let core = exec::LeafContext {
            live_docs: None,
            cache: None,
            ..ctx
        };
        match cache.scorer_if_cacheable(
            &clause,
            max_doc,
            cacheable,
            || Ok(false),
            || exec::build::build(&core, &clause, 1.0, exec::Mode::NoScores, true),
        )? {
            Some(exec::cache::CacheResult::Hit(set)) => return Ok(set.live_count(seg.live_docs)),
            Some(exec::cache::CacheResult::Empty) => return Ok(0),
            None => {}
        }
    }
    let mut count = Count(0);
    if let Some(mut bulk) = exec::bulk_boolean(&ctx, query, 1.0, exec::Mode::NoScores)? {
        exec::score_segment(
            &mut bulk,
            exec::Mode::NoScores,
            seg.live_docs,
            &mut count,
            ctx.max_doc,
        )?;
    }
    Ok(count.0)
}

/// `Weight.count` for a segment when it is answered without iterating
/// matches -- a lone term with no deletions, an `exists` from exact field
/// statistics -- else `None`.
pub(crate) fn count_boolean_query_shortcut(
    seg: &multi_segment::OpenSegment<'_>,
    query: &BooleanQuery,
) -> Result<Option<u64>> {
    // `BooleanWeight.count` of a lone required clause is that clause's
    // `Weight.count`: an `exists` is answered from the index statistics
    // (`FieldExistsQuery.count`) when they are exact for this segment, a term
    // from its `docFreq` when the segment has no deletions (`TermWeight.count`
    // -- the dictionary's own, whatever statistics the term scores with).
    // `IndexOrDocValuesQuery`'s weight counts with its index side's.
    let lone = lone_leaf(query).map(|c| match c {
        Clause::Extended(e) => match &**e {
            extended_query::ExtendedQuery::IndexOrDocValues(q) => &*q.index_query,
            _ => c,
        },
        c => c,
    });
    match (lone, seg.reader) {
        (Some(Clause::Exists(e)), Some(reader)) => {
            let mut leaf = reader.field_exists_leaf(&e.field, None)?;
            // The deletions the searcher sees, not the commit's.
            leaf.num_docs = match seg.live_docs {
                Some(live) => i32::try_from(live.cardinality()).unwrap_or(i32::MAX),
                None => leaf.max_doc,
            };
            if let Some(n) = weight_count::count_field_exists_leaf(&leaf) {
                #[cfg(any(test, feature = "asserting"))]
                exec::asserting::check_count(n, seg.max_doc, seg.live_docs);
                return Ok(Some(u64::try_from(n).unwrap_or(0)));
            }
        }
        // `MatchAllDocsQuery`'s `ConstantScoreWeight.count`: `numDocs`.
        (Some(Clause::MatchAllDocs(_)), _) => {
            if let Some(max_doc) = seg.max_doc {
                let n = match seg.live_docs {
                    Some(live) => live.cardinality() as u64,
                    None => u64::try_from(max_doc).unwrap_or(0),
                };
                return Ok(Some(n));
            }
        }
        (Some(Clause::Term(t)), _) => {
            if let Some(n) = weight_count::count_term_query_shortcut(seg.fields, seg.live_docs, t)?
            {
                #[cfg(any(test, feature = "asserting"))]
                exec::asserting::check_count(n, seg.max_doc, seg.live_docs);
                return Ok(Some(u64::try_from(n).unwrap_or(0)));
            }
        }
        // `PointRangeQuery.count`, on a segment without deletions.
        (Some(leaf @ (Clause::PointsRange(_) | Clause::Extended(_))), _)
            if seg.live_docs.is_none() && seg.points.is_some() =>
        {
            let points = seg.points.expect("checked");
            let n = match leaf {
                Clause::PointsRange(q) => exec::ranges::point_range_count(
                    points,
                    &q.field,
                    &points_query::pack_i64(q.min),
                    &points_query::pack_i64(q.max),
                    1,
                    8,
                )?,
                Clause::Extended(e) => match &**e {
                    extended_query::ExtendedQuery::PointRange(q) => {
                        exec::ranges::point_range_count(
                            points,
                            &q.field,
                            &q.lower,
                            &q.upper,
                            q.num_dims,
                            q.bytes_per_dim,
                        )?
                    }
                    _ => None,
                },
                _ => None,
            };
            if let Some(n) = n {
                #[cfg(any(test, feature = "asserting"))]
                exec::asserting::check_count(
                    i64::try_from(n).unwrap_or(i64::MAX),
                    seg.max_doc,
                    seg.live_docs,
                );
                return Ok(Some(n));
            }
        }
        _ => {}
    }
    Ok(None)
}

/// The one leaf a query is and nothing else -- the only `MUST` or `FILTER`
/// clause, possibly under a constant score, a boost or a nested boolean of
/// the same shape, which `BooleanWeight.count` and those wrappers' `count`
/// pass through. `None` for anything else -- including a
/// `minimum_should_match` above 0 with no `SHOULD` clause, which matches
/// nothing.
fn lone_leaf(query: &BooleanQuery) -> Option<&Clause> {
    fn inner(c: &Clause) -> Option<&Clause> {
        match c {
            Clause::ConstantScore(c) => inner(&c.inner),
            Clause::Boost(b) => inner(&b.inner),
            Clause::Boolean(b) => lone_leaf(b),
            leaf => Some(leaf),
        }
    }
    if !query.should.is_empty() || !query.must_not.is_empty() || query.minimum_should_match > 0 {
        return None;
    }
    match (query.must.as_slice(), query.filter.as_slice()) {
        ([c], []) | ([], [c]) => inner(c),
        _ => None,
    }
}

#[allow(clippy::too_many_arguments)]
fn search_boolean_query_scored_impl<C: ScoringCollector>(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    pos_in: Option<&PosInput<'_>>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    points: Option<&PointsInput<'_>>,
    query: &BooleanQuery,
    norms: Option<&HashMap<String, FieldNorms<'_>>>,
    global: Option<&GlobalStats>,
    max_doc: Option<i32>,
    cache: Option<&SegmentQueryCache>,
    reader: Option<&directory_reader::SegmentReader>,
    collector: &mut C,
) -> Result<()> {
    // One scoring clause and nothing to filter against: the clause's own score
    // map *is* the matched set, so running `matched_boolean_docs` first would
    // execute the clause a second time for an answer already in hand. On a
    // two-term phrase query that second execution was 14% of the query in
    // `resolve_clause_docs` alone, plus a full second pass of position decode
    // -- the single-clause shape is exactly how `search_phrase_query_scored`
    // reaches this function from the multi-segment layer.
    //
    // Restricted to `Phrase` and the wildcard family, the clause kinds whose
    // matched set this path streams exactly (a lone `Term` takes the scorer
    // tree below, which prunes). It is not true in general: a nested
    // `Boolean` can match documents its scoring sub-clauses never mention.
    // Those keep the two-pass path. `minimum_should_match` must be 0, not merely
    // `<= 1`: with no `should` clauses at all, a minimum of 1 means nothing
    // matches, and the fast path would wrongly return the `must` clause's
    // documents.
    //
    // A lone `FuzzyQuery` is a scored disjunction over its expanded terms in
    // Lucene (`TopTermsBlendedFreqScoringRewrite`), and is run as one --
    // `try_fuzzy_bulk` -- rather than through the score map below.
    if query.must.len() == 1
        && query.filter.is_empty()
        && query.should.is_empty()
        && query.must_not.is_empty()
        && query.minimum_should_match == 0
    {
        if let Clause::Fuzzy(fq) = &query.must[0] {
            let clause_norms = norms.and_then(|m| m.get(&fq.field));
            if try_fuzzy_bulk(
                fields,
                doc_in,
                live_docs,
                fq,
                clause_norms,
                global,
                collector,
            )? {
                return Ok(());
            }
        }
    }
    if query.must.len() == 1
        && query.filter.is_empty()
        && query.should.is_empty()
        && query.must_not.is_empty()
        && query.minimum_should_match == 0
        // A phrase with explicit positions takes the scorer tree.
        && !matches!(&query.must[0], Clause::Phrase(p) if !p.has_implicit_positions())
        && matches!(
            query.must[0],
            Clause::Phrase(_)
                | Clause::Prefix(_)
                | Clause::Wildcard(_)
                | Clause::Regexp(_)
                | Clause::TermInSet(_)
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
        //
        // A lone `Term` is not here: it takes the scorer tree below, whose
        // term bulk scorer skips blocks on impacts once the collector has a
        // threshold (`TermScorer` + `BatchScoreBulkScorer`). This path scored
        // every posting -- 0.08x Lucene on a dense term, found by the REST
        // benchmark's shapes once a lone term reached it through the query
        // tree.
        match &query.must[0] {
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
            //
            // `Clause::Fuzzy` is **not** admitted here, and must not be: alone
            // in this family it is a *scoring* query, because `FuzzyQuery`
            // overrides the constant-score rewrite with
            // `MultiTermQuery.TopTermsBlendedFreqScoringRewrite`. Collecting it
            // at a flat 1.0 does not merely mis-score -- it changes the hit
            // *set*, since every document ties and the collector then keeps the
            // first `n` by document id instead of the top `n` by score. It
            // takes the two-pass path below, whose `clause_scores` arm calls
            // `fuzzy_doc_scores`; the test pinning this is
            // `a_single_fuzzy_clause_is_scored_not_constant_scored`.
            other @ (Clause::Prefix(_)
            | Clause::Wildcard(_)
            | Clause::Regexp(_)
            | Clause::TermInSet(_)) => {
                // A small term set on a segment with a query cache: the
                // scorer tree below, whose rewrite of them (a `BooleanQuery` of
                // `SHOULD` terms, built without scores) is cached as Lucene's
                // `CachingWrapperWeight` caches it -- a term set used again
                // is then iterated from a bitset. Streaming here would never
                // count a use, so it would never be cached.
                if !small_rewrite_is_cached(cache, max_doc, other) {
                    // Lazily merge the expanded terms' postings and stop at
                    // the collector's capacity -- see
                    // `stream_constant_score_clause`.
                    if stream_constant_score_clause(fields, doc_in, live_docs, other, collector)? {
                        return Ok(());
                    }
                    let matched = resolve_clause_docs(
                        fields, doc_in, pos_in, pay_in, live_docs, points, other,
                    )?;
                    debug_assert!(
                        matched.windows(2).all(|w| w[0] < w[1]),
                        "resolve_clause_docs must yield ascending, deduplicated doc ids"
                    );
                    for doc_id in matched {
                        collector.collect(doc_id, 1.0);
                    }
                    return Ok(());
                }
            }
            // Unreachable: the guard above admits only the arms handled here.
            _ => unreachable!("guarded by the matches! above"),
        }
    }

    // Everything else runs as a Lucene scorer tree: iterators composed the way
    // `BooleanScorerSupplier` composes them, advanced lazily, pruned by block
    // maxima once the collector publishes a threshold. See `exec`.
    let ctx = exec::LeafContext {
        fields,
        doc_in,
        pos_in,
        pay_in,
        live_docs,
        points,
        norms,
        global,
        max_doc,
        cache,
        reader,
        similarity: None,
    };
    let mode = exec::Mode::of(collector);
    if let Some(mut bulk) = exec::bulk_boolean(&ctx, query, 1.0, mode)? {
        exec::score_segment(&mut bulk, mode, live_docs, collector, ctx.max_doc)?;
    }
    Ok(())
}

/// MAXSCORE-style sibling of [`search_boolean_query_scored`], kept as a
/// distinct entry point because `lucene-ffi`
/// (`ffi_search_boolean_query_scored_maxscore`,
/// `ffi_search_boolean_query_multi_segment_maxscore`) and
/// [`multi_segment::search_boolean_query_multi_segment_maxscore`] name it.
///
/// **This is a delegate.** The MAXSCORE machinery is
/// `bulk_scorer::MaxScore` (`MaxScoreBulkScorer`), which the bulk-scorer
/// dispatch in `exec` picks for a disjunction of term clauses exactly as
/// `BooleanScorerSupplier.booleanScorer()` does; this entry point runs the
/// same dispatch as [`search_boolean_query_scored_with_stats`].
///
/// The name and signature are kept for the callers above.
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
pub fn search_boolean_query_scored_maxscore_with_stats<C: ScoringCollector>(
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
    search_boolean_query_scored_with_stats(
        fields, doc_in, pos_in, pay_in, live_docs, points, query, norms, global, collector,
    )
}

/// `DisjunctionMaxQuery`-equivalent: reports every doc matching at
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
/// change as the `next_doc`/`advance` binary searches.
fn phrase_freq_exact_impl(term_positions: &[&[i32]], stop_at_first: bool) -> i32 {
    if let [first, second] = term_positions {
        return phrase_freq_exact_two(first, second, stop_at_first);
    }
    phrase_freq_exact_general(term_positions, stop_at_first)
}

/// [`phrase_freq_exact_impl`] for any number of terms.
fn phrase_freq_exact_general(term_positions: &[&[i32]], stop_at_first: bool) -> i32 {
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

/// [`phrase_freq_exact_impl`] for a two-term phrase -- every phrase in the
/// benchmark mix, and the common case: one merge of two ascending lists, counting each
/// `p` of `first` whose `p + 1` is in `second`. The general loop's cursor
/// table, slice-of-slices indexing and per-term bookkeeping were two thirds
/// of its cost on `"t0 t1"`, whose documents hold a handful of positions each.
fn phrase_freq_exact_two(first: &[i32], second: &[i32], stop_at_first: bool) -> i32 {
    if !stop_at_first {
        return phrase_freq_exact_two_count(first, second);
    }
    let (mut i, mut j, mut freq) = (0, 0, 0);
    while let (Some(&p0), Some(&p1)) = (first.get(i), second.get(j)) {
        // The same `p0 + 1` the general loop aligns on.
        let target = p0 + 1;
        if p1 < target {
            j += 1;
            continue;
        }
        if p1 == target {
            freq += 1;
            if stop_at_first {
                break;
            }
        }
        // `j` stays: a multi-phrase position list is a union and may repeat
        // a position, and the next `p0` may equal this one.
        i += 1;
    }
    freq
}

/// [`phrase_freq_exact_two`]'s counting merge without a branch per step:
/// whether `second` or `first` moves on, and whether the step counts, are
/// both data, not control flow. A document holds a handful of positions per
/// term, and which list moves next is close to a coin flip, so the branchy
/// merge mispredicted most of its steps. The steps are the same: `j` moves
/// past a `second` position below `p0 + 1`; otherwise `i` moves, counting
/// when the position is exactly `p0 + 1` (and `j` stays, as there).
fn phrase_freq_exact_two_count(first: &[i32], second: &[i32]) -> i32 {
    let (mut i, mut j, mut freq) = (0usize, 0usize, 0i32);
    while i < first.len() && j < second.len() {
        let target = first[i].wrapping_add(1);
        let p1 = second[j];
        let behind = p1 < target;
        freq += i32::from(p1 == target);
        j += usize::from(behind);
        i += usize::from(!behind);
    }
    freq
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

/// Key type for [`span_matches_in_doc`]'s `doc_positions` map: one leaf
/// `SpanQuery::SpanTerm`'s `(field, term)` pair, the finest granularity a span
/// query ever needs a position list for (unlike `PhraseQuery`, a `SpanQuery`'s
/// leaves aren't all implicitly the same field — see [`SpanQuery`]'s doc
/// comment).
pub(crate) type SpanLeafKey = (String, Vec<u8>);

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
/// doc, the same shape `phrase_matches_in_doc`/[`sloppy_phrase`]
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

/// [`SpanQuery::SpanNear`]'s matching algorithm: `NearSpansOrdered`'s or
/// `NearSpansUnordered`'s walk (`crate::near_spans`) over every clause's own
/// spans, which are computed recursively via [`span_matches_in_doc`] so a
/// `SpanNear` of `SpanNear`s composes for free.
///
/// **This is a walk, not a cartesian product.** Both Java classes are `Spans`
/// iterators whose sub-span cursors only ever move forward, and the sequence
/// of `(startPosition(), endPosition())` pairs they emit is *narrower* than
/// "every arrangement of one span per clause that fits the budget":
///
/// - `in_order == true` is `NearSpansOrdered`. `stretchToOrder` advances each
///   later clause to the first span starting at or after the previous
///   clause's end and never rewinds, so each position of clause 0 contributes
///   exactly one arrangement -- the minimum-slop one. Its own class doc:
///   "the formed spans only contains minimum slop matches."
/// - `in_order == false` is `NearSpansUnordered`. `atMatch()` is a pure width
///   test -- `maxEndPosition - top().startPosition() - totalSpanLength <=
///   slop` -- with **no** non-overlap requirement and no relative-order
///   requirement at all, evaluated at each step of a priority-queue walk that
///   advances only the least span. A reversed pair matches as long as the
///   width fits, and two clauses may settle on the *same* span, giving a
///   negative width, which matches: real Lucene returns doc 8555 -- a document
///   with a single `alpha` -- for `SpanNearQuery([alpha, alpha], 0, false)`.
///   Its `endPosition()` is the *running* `maxEndPosition`, which can exceed
///   the current arrangement's own maximum end.
///
/// Enumerating the whole cartesian product and reporting `(min start, max
/// end)` per accepted arrangement would produce extents Java's walk never
/// visits. The hit set would not change ([`span_doc_ids`] only asks whether
/// the span list is non-empty), but a nested `SpanNear`-of-`SpanNear`
/// consumes the inner clause's extents. Ground truth for the sequences themselves:
/// `fixtures/src/AppendSpanExtentManifest.java`'s `spanextent.*` keys, which
/// are real `Spans.nextStartPosition()`/`endPosition()` walks.
///
/// **Slop formula**: for a *non-overlapping* arrangement the two arms compute
/// the same number -- `sum(next.start - prev.end)` telescopes to
/// `maxEnd - minStart - sum(lengths)` -- so `slop` means the same thing on
/// both wherever both accept. It is **not** the same quantity as a sloppy
/// phrase's `matchLength` (see [`sloppy_phrase`]), which is a window width
/// over slot-shifted positions.
///
/// Returned spans are sorted ascending and deduplicated, which loses nothing:
/// both walks emit `(start, end)` pairs that are already non-decreasing in
/// both components (clause 0's cursor and the queue's minimum only rise, and
/// `maxEndPosition` only rises), so sorting is a no-op and `dedup` removes
/// only the consecutive repeats a caller asking "which extents exist" does not
/// want twice.
///
/// **Complexity**: linear in the total number of sub-spans, not exponential in
/// the clause count -- the cartesian product this replaced was the latter.
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
    let slices: Vec<&[(i32, i32)]> = per_clause_spans.iter().map(Vec::as_slice).collect();

    let slop = i64::from(slop);
    let mut results: Vec<(i32, i32)> = Vec::new();
    let mut record = |_: crate::near_spans::Arrangement<'_>, start: i32, end: i32| {
        results.push((start, end));
    };
    if in_order {
        crate::near_spans::for_each_ordered_match(&slices, slop, &mut record);
    } else {
        crate::near_spans::for_each_unordered_match(&slices, slop, &mut record);
    }
    results.sort_unstable();
    results.dedup();
    results
}

/// Collects every distinct `SpanQuery::SpanTerm` leaf's `(field, term)` pair
/// appearing anywhere in `query` (recursively through `SpanNear`/`SpanOr`),
/// deduplicated -- the set of position lists [`span_doc_ids`] needs to fetch
/// before it can evaluate [`span_matches_in_doc`] for any candidate doc.
pub(crate) fn collect_span_leaves(query: &SpanQuery, leaves: &mut Vec<SpanLeafKey>) {
    match query {
        SpanQuery::SpanTerm { field, term } => leaves.push((field.clone(), term.clone())),
        SpanQuery::SpanNear { clauses, .. } | SpanQuery::SpanOr { clauses } => {
            for clause in clauses {
                collect_span_leaves(clause, leaves);
            }
        }
    }
}

/// `SpanTermQuery.SpanTermWeight.getSpans`' positions check, reached as
/// `SpanWeight.getSpans` walks `query`: a term this segment has, in a field it
/// indexed without positions, is Java's `IllegalStateException`. The walk
/// stops where Java's does -- `SpanNearQuery` at its first clause without
/// spans, `SpanOrQuery` after every clause -- so a near whose earlier term
/// is absent here reports nothing. A query whose fields all have positions
/// (or are absent) is not walked.
pub(crate) fn require_span_positions(fields: &BlockTreeFields, query: &SpanQuery) -> Result<()> {
    fn lacks(fields: &BlockTreeFields, q: &SpanQuery) -> bool {
        match q {
            SpanQuery::SpanTerm { field, .. } => fields
                .field(field)
                .is_some_and(|ft| !ft.index_options().subsumes_positions()),
            SpanQuery::SpanNear { clauses, .. } | SpanQuery::SpanOr { clauses } => {
                clauses.iter().any(|c| lacks(fields, c))
            }
        }
    }
    /// Whether `q` has spans in this segment (Java's non-`null`).
    fn walk(fields: &BlockTreeFields, q: &SpanQuery) -> Result<bool> {
        match q {
            SpanQuery::SpanTerm { field, term } => {
                let Some(ft) = fields.field(field) else {
                    return Ok(false);
                };
                if ft.try_seek_exact(term)?.is_none() {
                    return Ok(false);
                }
                if !ft.index_options().subsumes_positions() {
                    return Err(Error::IllegalState(format!(
                        "field \"{field}\" was indexed without position data; cannot run SpanTermQuery (term={})",
                        String::from_utf8_lossy(term)
                    )));
                }
                Ok(true)
            }
            SpanQuery::SpanNear { clauses, .. } => {
                for c in clauses {
                    if !walk(fields, c)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            SpanQuery::SpanOr { clauses } => {
                let mut any = false;
                for c in clauses {
                    any |= walk(fields, c)?;
                }
                Ok(any)
            }
        }
    }
    if lacks(fields, query) {
        walk(fields, query)?;
    }
    Ok(())
}

/// [`Clause::Span`]'s matched doc-ID list: [`span_doc_extents`]
/// with the extents dropped.
///
/// [`span_doc_extents`] gathers every distinct leaf `(field, term)` pair
/// `query` touches (via [`collect_span_leaves`]), fetches each one's
/// live-filtered `doc_id -> position list` map (via [`term_doc_positions`],
/// the same helper [`search_phrase_query`] uses), then for every doc appearing
/// in **any** leaf's doc list (a safe, simple over-approximation of the true
/// candidate set -- see below for why that's fine) builds a per-doc
/// `doc_positions` map and runs [`span_matches_in_doc`] over it.
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
    Ok(
        span_doc_extents(fields, doc_in, pos_in, pay_in, live_docs, query)?
            .into_iter()
            .map(|(doc_id, _)| doc_id)
            .collect(),
    )
}

/// One document's matching span extents, as [`span_doc_extents`] returns them:
/// the document id and its `(startPosition(), endPosition())` pairs.
pub type DocSpans = (i32, Vec<(i32, i32)>);

/// Every matching document's own **span extents**, which is what real
/// `SpanWeight.getSpans(context, Postings.POSITIONS)` walked to
/// `NO_MORE_DOCS`/`NO_MORE_POSITIONS` produces: one
/// `(startPosition(), endPosition())` pair per emitted match, ascending, per
/// document, live documents only.
///
/// [`span_doc_ids`] is this with the extents dropped; a caller that wants the
/// ranges -- a highlighter, or anything composing spans further -- wants this.
/// A document with no matching span is not listed, so the returned document
/// list is exactly [`span_doc_ids`]'.
///
/// The extents are the walks' own (see [`span_near_matches`]), **not** the
/// bounding box of every arrangement that fits the slop budget: for
/// `in_order == true` Java emits only minimum-slop arrangements, and for
/// `in_order == false` its `endPosition()` is a running maximum. Ground truth
/// is `fixtures/src/AppendSpanExtentManifest.java`'s `spanextent.*` keys,
/// recorded from real `Spans` walks.
#[allow(clippy::too_many_arguments)]
pub fn span_doc_extents(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    pos_in: Option<&PosInput<'_>>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &SpanQuery,
) -> Result<Vec<DocSpans>> {
    let Some((candidate_docs, per_leaf_maps)) =
        span_leaf_positions(fields, doc_in, pos_in, pay_in, live_docs, query)?
    else {
        return Ok(Vec::new());
    };
    let mut result: Vec<DocSpans> = Vec::new();
    for doc_id in candidate_docs {
        let mut doc_positions: HashMap<SpanLeafKey, Vec<i32>> = HashMap::new();
        for (key, map) in &per_leaf_maps {
            if let Some(positions) = map.get(&doc_id) {
                doc_positions.insert(key.clone(), positions.clone());
            }
        }
        let extents = span_matches_in_doc(query, &doc_positions);
        if !extents.is_empty() {
            result.push((doc_id, extents));
        }
    }
    Ok(result)
}

/// Every leaf of a span query with its live documents' positions: the
/// candidate documents (any leaf's, ascending) and, per distinct leaf
/// `(field, term)` present in the segment, a `doc -> positions` map. `None`
/// when the query has no leaf. See [`span_doc_ids`] for why the candidates
/// are the union of every leaf's documents.
#[allow(clippy::type_complexity)]
pub(crate) fn span_leaf_positions(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    pos_in: Option<&PosInput<'_>>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &SpanQuery,
) -> Result<Option<(Vec<i32>, Vec<(SpanLeafKey, HashMap<i32, Vec<i32>>)>)>> {
    require_span_positions(fields, query)?;
    let mut leaves: Vec<SpanLeafKey> = Vec::new();
    collect_span_leaves(query, &mut leaves);
    // A segment without any leaf's field has no spans (`SpanTermWeight`
    // returns none), `.pos` file or not.
    if !leaves.iter().any(|(f, _)| fields.field(f).is_some()) {
        return Ok(None);
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
    Ok(Some((candidate_docs, per_leaf_maps)))
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
pub(crate) type TermDocPositions = (Vec<i32>, Vec<i32>, Vec<(u32, u32)>);

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
        if !live_docs.is_none_or(|bits| bits.get_doc(doc_id)) {
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

/// `getPhraseMatcher`'s `fieldTerms.hasPositions() == false` check: a phrase
/// that `rewrite` leaves a phrase (two or more terms for `PhraseQuery`, two or
/// more positions for `MultiPhraseQuery`) over a field this segment indexed
/// without positions is Java's `IllegalStateException`, with the query as its
/// `toString()`. A segment without the field has no matcher, so no error;
/// the check comes before any term lookup, as Java's does.
pub(crate) fn require_phrase_positions(fields: &BlockTreeFields, q: &PhraseQuery) -> Result<()> {
    phrase_positions_check(field_index_options(fields, &q.field), q)
}

/// [`require_phrase_positions`] for `MultiPhraseQuery`.
pub(crate) fn require_multi_phrase_positions(
    fields: &BlockTreeFields,
    q: &query::MultiPhraseQuery,
) -> Result<()> {
    multi_phrase_positions_check(field_index_options(fields, &q.field), q)
}

/// [`require_phrase_positions`] given the field's index options in the leaf
/// (`None`: the leaf has no terms in it).
pub(crate) fn phrase_positions_check(
    options: Option<lucene_codecs::field_infos::IndexOptions>,
    q: &PhraseQuery,
) -> Result<()> {
    if q.terms.len() < 2 {
        return Ok(());
    }
    no_positions(options, &q.field, "PhraseQuery", || {
        Clause::Phrase(q.clone())
    })
}

/// [`phrase_positions_check`] for `MultiPhraseQuery`.
pub(crate) fn multi_phrase_positions_check(
    options: Option<lucene_codecs::field_infos::IndexOptions>,
    q: &query::MultiPhraseQuery,
) -> Result<()> {
    if q.term_arrays.len() < 2 {
        return Ok(());
    }
    no_positions(options, &q.field, "MultiPhraseQuery", || {
        Clause::MultiPhrase(q.clone())
    })
}

fn field_index_options(
    fields: &BlockTreeFields,
    field: &str,
) -> Option<lucene_codecs::field_infos::IndexOptions> {
    fields.field(field).map(|ft| ft.index_options())
}

fn no_positions(
    options: Option<lucene_codecs::field_infos::IndexOptions>,
    field: &str,
    name: &str,
    clause: impl FnOnce() -> Clause,
) -> Result<()> {
    match options {
        Some(o) if !o.subsumes_positions() => Err(Error::IllegalState(format!(
            "field \"{field}\" was indexed without position data; cannot run {name} (phrase={})",
            query_builder::to_query_string(Some(&clause()), "")
        ))),
        _ => Ok(()),
    }
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
/// `slop` existed), `query.slop > 0` runs
/// [`sloppy_phrase::sloppy_phrase_matches`], the ported `SloppyPhraseMatcher`
/// -- which admits terms appearing out of phrase order within the budget.
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
    require_phrase_positions(fields, query)?;
    if query.terms.is_empty() {
        return Ok(());
    }
    if query.terms.len() == 1 {
        let term_query = TermQuery::new(query.field.clone(), query.terms[0].clone());
        return search_term_query(fields, doc_in, live_docs, &term_query, collector);
    }
    let Some(field_terms) = fields.field(&query.field) else {
        return Ok(());
    };
    let Some(pos_in) = pos_in else {
        return Err(Error::MissingPosInput);
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
    .filter(|&doc_id| live_docs.is_none_or(|bits| bits.get_doc(doc_id)))
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
            .try_seek_exact(term)?
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

    // Which slots hold the same term -- `SloppyPhraseMatcher`'s `rptGroups`,
    // which Java computes once per scorer and this port computes once per
    // query. Unused at `slop == 0`, where the exact matcher runs instead.
    let repeats = sloppy_phrase::PhraseRepeats::for_phrase(&query.terms);

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
            sloppy_phrase::sloppy_phrase_matches(&term_positions, &repeats, query.slop)
        };
        if is_match {
            collector.collect(doc_id);
        }
    }
    Ok(())
}

/// Scored sibling of [`search_phrase_query`]: same matching
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
/// - `query.slop > 0`: phrase frequency is
///   [`sloppy_phrase::sloppy_phrase_freq`], which is `PhraseScorer.score()`'s
///   own `freq = sloppyWeight(); while (nextMatch()) freq += sloppyWeight();`
///   over the ported `SloppyPhraseMatcher` walk -- a graduated
///   `1 / (1 + matchLength)` per match, summed in Lucene's own emission order
///   (`f32` addition is not associative, so the order is part of the answer).
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
    require_phrase_positions(fields, query)?;
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
    let Some(field_terms) = fields.field(&query.field) else {
        return Ok(());
    };
    let Some(pos_in) = pos_in else {
        return Err(Error::MissingPosInput);
    };

    // Real BM25's phrase idf is the sum of every constituent term's own idf --
    // see this function's doc comment. A missing term means the phrase can
    // never match, same convention as `search_phrase_query`.
    // `BM25Similarity.idfExplain(TermStatistics[])` sums in a double and
    // casts once: an `f32` sum of three or more idfs can be an ulp off.
    let mut idf_acc = 0.0f64;
    let mut term_doc_freqs = Vec::with_capacity(query.terms.len());
    for term in &query.terms {
        let Some(stats) = field_terms.try_seek_exact(term)? else {
            return Ok(());
        };
        term_doc_freqs.push(stats.doc_freq);
        // Reader-wide statistics where the caller has them, exactly as the term
        // path does. A phrase's idf is the sum of its terms' idfs, so a
        // per-segment idf here is wrong in the same way and for the same
        // reason -- and it is what left two phrase queries disagreeing with
        // Java on the segmented corpus after every other query agreed.
        let (df, dc) = match global.and_then(|g| g.term(&query.field, term)) {
            Some(g) => (g.doc_freq, g.doc_count),
            None => (stats.doc_freq as i64, field_terms.doc_count as i64),
        };
        idf_acc += f64::from(similarity::idf(df, dc));
    }
    let idf_sum = idf_acc as f32;

    // `PhraseScorer` over lazy positions: a leapfrog over the terms' documents,
    // a `maxFreq` score check before any position is read, and positions
    // decoded only for the documents that survive it. See `phrase_scorer`.
    // Needs a real `.doc` stream for every term, so a pulsed singleton
    // (`docFreq == 1`, whose one posting lives in the term dictionary) keeps
    // the eager path below.
    if let Some(doc_in) = doc_in {
        if term_doc_freqs.iter().all(|&df| df > 1) {
            let mut cursors = Vec::with_capacity(query.terms.len());
            for (term, &df) in query.terms.iter().zip(&term_doc_freqs) {
                let Some(cursor) = field_terms.lazy_positions(term, doc_in, pos_in)? else {
                    return Ok(());
                };
                cursors.push((cursor, df as i64));
            }
            let repeats = sloppy_phrase::PhraseRepeats::for_phrase(&query.terms);
            return phrase_scorer::score_phrase(
                cursors,
                idf_sum,
                query.slop,
                &repeats,
                norms.map(|n| n.cursor()),
                live_docs,
                collector,
            );
        }
    }

    let matched = phrase_doc_freqs(field_terms, doc_in, pos_in, pay_in, live_docs, query)?;
    // One norms cursor for this scan; the matches ascend, so a sparse
    // field's `IndexedDISI` region is walked once, not once per document.
    let mut norms_cursor = norms.map(|n| n.cursor());
    for (doc_id, phrase_freq) in matched {
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

/// The eager phrase path: every live document `query` matches in this
/// segment, ascending, with its phrase frequency (`ExactPhraseMatcher`'s
/// count, or `SloppyPhraseMatcher`'s summed weights). What a phrase whose
/// term is a pulsed singleton -- no `.doc` stream for a lazy cursor -- is
/// scored from, by BM25 below and by any other similarity in the scorer
/// tree. `field_terms` holds every one of the phrase's terms.
pub(crate) fn phrase_doc_freqs(
    field_terms: &blocktree::FieldTerms,
    doc_in: Option<&DocInput<'_>>,
    pos_in: &PosInput<'_>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &PhraseQuery,
) -> Result<Vec<(i32, f32)>> {
    // Documents first, positions second.
    //
    // A phrase can only match where every term does, so positions are needed
    // only for the intersection -- which for `phrase t0 t1` on the benchmark corpus is
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
            return Ok(Vec::new());
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
    .filter(|&doc_id| live_docs.is_none_or(|bits| bits.get_doc(doc_id)))
    .collect();
    if candidate_docs.is_empty() {
        return Ok(Vec::new());
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
            .try_seek_exact(term)?
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
    // `wanted` was built from the same candidate list for each term.
    let mut term_positions: Vec<&[i32]> = Vec::with_capacity(per_term_positions.len());
    // `SloppyPhraseMatcher`'s `rptGroups`, computed once per query -- see
    // `search_phrase_query`. Unused at `slop == 0`.
    let repeats = sloppy_phrase::PhraseRepeats::for_phrase(&query.terms);
    let mut out = Vec::with_capacity(candidate_docs.len());
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
        // a flat `1` for "matched". See `sloppy_phrase::sloppy_phrase_freq`.
        let phrase_freq = if query.slop == 0 {
            phrase_freq_exact(&term_positions) as f32
        } else {
            sloppy_phrase::sloppy_phrase_freq(&term_positions, &repeats, query.slop)
        };
        if phrase_freq != 0.0 {
            out.push((doc_id, phrase_freq));
        }
    }
    Ok(out)
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
pub(crate) fn multi_phrase_slot_docs(
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
pub(crate) fn multi_phrase_slot_positions(
    field_terms: &blocktree::FieldTerms,
    doc_in: Option<&DocInput<'_>>,
    pos_in: &PosInput<'_>,
    pay_in: Option<&PayInput<'_>>,
    terms: &[Vec<u8>],
    candidates: &[i32],
) -> Result<(Vec<i32>, Vec<u32>)> {
    let mut per_doc: Vec<Vec<i32>> = vec![Vec::new(); candidates.len()];
    for term in terms {
        let Some(stats) = field_terms.try_seek_exact(term)? else {
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
/// can never disagree about which documents match .
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
    require_multi_phrase_positions(fields, query)?;
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
    // `BM25Similarity.idfExplain(TermStatistics[])` sums in a double and
    // casts once: an `f32` sum of three or more idfs can be an ulp off.
    let mut idf_acc = 0.0f64;
    let mut any_term_present = false;
    for alternatives in &query.term_arrays {
        for term in alternatives {
            let Some(stats) = field_terms.try_seek_exact(term)? else {
                continue;
            };
            any_term_present = true;
            let (df, dc) = match global.and_then(|g| g.term(&query.field, term)) {
                Some(g) => (g.doc_freq, g.doc_count),
                None => (stats.doc_freq as i64, field_terms.doc_count as i64),
            };
            idf_acc += f64::from(similarity::idf(df, dc));
        }
    }
    let idf_sum = idf_acc as f32;
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

    // The matches ascend, so one norms cursor covers the whole scan.
    let mut norms_cursor = norms.map(|n| n.cursor());
    multi_phrase_freqs(
        field_terms,
        doc_in,
        pos_in,
        pay_in,
        live_docs,
        query,
        |doc_id, freq| {
            let norm_inverse = match norms_cursor.as_mut() {
                Some(nc) => nc.norm_inverse(doc_id)?,
                None => similarity::UNNORMED_NORM_INVERSE,
            };
            collector.collect(doc_id, similarity::do_score(idf_sum, freq, norm_inverse));
            Ok(())
        },
    )
}

/// `doc`'s frequency for a `MultiPhraseQuery` of two or more positions
/// (`PhraseMatcher`'s, what its weight's `SimScorer` reads), `None` when the
/// phrase does not match it: what an explanation without scores prints.
#[allow(clippy::too_many_arguments)]
pub(crate) fn multi_phrase_freq(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    pos_in: Option<&PosInput<'_>>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &query::MultiPhraseQuery,
    doc: i32,
) -> Result<Option<f32>> {
    require_multi_phrase_positions(fields, query)?;
    let Some(field_terms) = fields.field(&query.field) else {
        return Ok(None);
    };
    let mut found = None;
    multi_phrase_freqs(
        field_terms,
        doc_in,
        pos_in,
        pay_in,
        live_docs,
        query,
        |doc_id, freq| {
            if doc_id == doc {
                found = Some(freq);
            }
            Ok(())
        },
    )?;
    Ok(found)
}

/// Every live document a `MultiPhraseQuery` of two or more positions
/// matches, ascending, with its phrase frequency: the matching half of
/// [`multi_phrase_hits`].
fn multi_phrase_freqs(
    field_terms: &blocktree::FieldTerms,
    doc_in: Option<&DocInput<'_>>,
    pos_in: Option<&PosInput<'_>>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &query::MultiPhraseQuery,
    mut on_match: impl FnMut(i32, f32) -> Result<()>,
) -> Result<()> {
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
    .filter(|&doc_id| live_docs.is_none_or(|bits| bits.get_doc(doc_id)))
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

    // See `search_phrase_query`'s own binding: `MultiPhraseQuery` slots hold a
    // *set* of terms each, which is the `hasMultiTermRpts` case of Java's
    // repeat detection.
    let repeats = sloppy_phrase::PhraseRepeats::for_multi_phrase(&query.term_arrays);

    let mut slot_positions: Vec<&[i32]> = Vec::with_capacity(slot_count);
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
            sloppy_phrase::sloppy_phrase_freq(&slot_positions, &repeats, query.slop)
        };
        if freq == 0.0 {
            continue;
        }
        on_match(doc_id, freq)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
