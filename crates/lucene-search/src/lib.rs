#![forbid(unsafe_code)]
//! Query execution over opened segments: Lucene's `search` package, its
//! collectors and similarities, and the shard-side pieces built on them.
//!
//! Queries are the enums of [`query`] (`TermQuery`, `BooleanQuery` with
//! nested [`query::Clause`]s, phrase, span, multi-term, `DisjunctionMax`,
//! constant-score, points, vector). A segment is searched by the free
//! functions of this crate root ([`search_term_query`],
//! [`search_boolean_query`], [`search_phrase_query`], ...), or across segments
//! through [`multi_segment`] and [`index_searcher`], feeding a
//! [`collector::Collector`] (matches only) or [`collector::ScoringCollector`]
//! (matches with BM25 or another [`similarities::Similarity`] score; see
//! [`field_norms`] for the length norms it reads). The function groups live in
//! modules re-exported here: `multi_term_expansion` (prefix, wildcard, fuzzy,
//! regexp and term-in-set expansion), `phrase_search`, `span_search` and
//! `global_stats` (reader-wide statistics so every leaf scores with the same
//! idf). [`doc_value_query`] holds the doc-values range filters and sorts;
//! [`docid_set`] the `Conjunction`/`Disjunction`/`Excluding` merge combinators.
//!
//! ## `BooleanQuery` matching
//!
//! Matching follows `BooleanQuery.rewrite()` and `BooleanWeight`:
//!
//! - A query with no `must`, no `filter` and no `should` clause matches
//!   nothing, whatever its `must_not` clauses say (Lucene rewrites it to
//!   `MatchNoDocsQuery`); a pure negative query is not "everything except".
//! - `filter` is `MUST` with the score dropped: it contributes `0`, never
//!   counts toward `minimum_should_match`, and never enters the float
//!   summation, whose order is observable because `f32` addition is not
//!   associative. A filter-only query matches, at score `0`.
//! - `minimum_should_match` gates `should` whether or not `must` is present.
//!   At `0` (the default) `should` clauses only score once a `must` or
//!   `filter` clause exists, and with none a document needs one `should` hit;
//!   above `0` a document must also match that many `should` clauses
//!   (`should_match_counts` counts them). A value above `should.len()`
//!   matches nothing, as Lucene's rewrite does.
//! - `must_not`'s disjunction is subtracted from the result and never scores.
//! - A clause can itself be a [`query::Clause::Boolean`], to any depth,
//!   resolved independently of its parent (`resolve_clause_docs`,
//!   `clause_scores`).
//! - Scores sum each matching document's per-clause scores over the `must`
//!   and `should` clauses it satisfies, as Lucene's additive `BooleanScorer`.
//!
//! A numeric `NumericDocValues` range or sorted-ordinal filter and the sort by
//! a numeric doc value are in [`doc_value_query`].

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
mod scores_at;
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

mod multi_term_expansion;
pub use multi_term_expansion::*;
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
/// - `Clause::Term`: delegates straight to [`term_doc_ids`].
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
///   [`term_doc_scores`]), keyed by `query.field` in `norms`.
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

mod global_stats;
pub use global_stats::*;
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

mod span_search;
pub use span_search::*;
mod phrase_search;
pub use phrase_search::*;
#[cfg(test)]
mod tests;
