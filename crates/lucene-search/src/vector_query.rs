//! Search-side KNN (approximate nearest neighbour) vector queries: the
//! *query-level* half of Lucene's vector search, on top of the codec half in
//! [`lucene_codecs::vectors`] (the
//! `Lucene99FlatVectorsFormat` `.vec`/`.vemf` store),
//! [`lucene_codecs::hnsw`] (`org.apache.lucene.util.hnsw.*`) and
//! [`lucene_codecs::hnsw_vectors`] (the `.vem`/`.vex` graph).
//!
//! Java counterparts (Lucene 10.5.0, `lucene/core/src/java/`):
//! `org/apache/lucene/search/{AbstractKnnVectorQuery, KnnFloatVectorQuery,
//! KnnByteVectorQuery, AcceptDocs, KnnCollector, AbstractKnnCollector,
//! TopKnnCollector, VectorScorer}.java`,
//! `org/apache/lucene/search/knn/{KnnCollectorManager,
//! TopKnnCollectorManager, KnnSearchStrategy}.java`, and the dispatch half of
//! `org/apache/lucene/codecs/lucene99/Lucene99HnswVectorsReader.search`.
//!
//! ## What lives here, and what was already ported elsewhere
//!
//! `TopKnnCollector`/`AbstractKnnCollector` are **already ported**, as
//! [`lucene_codecs::hnsw::KnnCollector`] -- one type, because Java's
//! collector is a `NeighborQueue` plus a visit limit and the graph builder
//! needs the same thing. `VectorScorer` is
//! [`lucene_codecs::hnsw::VectorScorer`] plus
//! [`lucene_codecs::vectors::FloatVectorScorer`]/`ByteVectorScorer`, and the
//! graph walk is [`lucene_codecs::hnsw::HnswGraphSearcher`]. None of them is
//! re-implemented here. This module is the layer above: which collector
//! size, which accept set, and graph-walk-or-exact -- exactly what
//! `AbstractKnnVectorQuery.rewrite`/`getLeafResults`/`exactSearch` decide.
//!
//! ## Where the filter comes from
//!
//! Java's `AbstractKnnVectorQuery` holds a filter `Query` and resolves it per
//! leaf through `Weight`/`Scorer` inside `rewrite`. This port has no
//! `IndexSearcher`/`Weight`, so the resolved per-segment doc set is an
//! **input** ([`VectorsInput::filter`]), exactly as `live_docs` already is
//! for every other query function in this crate. A caller builds it from
//! whatever query it likes -- [`crate::resolve_clause_docs`] turns a
//! [`crate::query::BooleanQuery`] (`Occur::FILTER` clauses included) into a
//! `Vec<i32>` and [`accept_bitset`] turns that into the bitset this module
//! wants. Java's implicit `FieldExistsQuery(field)` conjunct is folded in here
//! too, for free: translating the accept set into **ordinal** space drops every
//! document that has no vector, which is all that conjunct does.
//!
//! ## Per-leaf `k` is pro-rata in 10.5.0, not `k`
//!
//! Worth stating loudly, because it changed and the older shape is the
//! intuitive one: `TopKnnCollectorManager.isOptimistic()` returns **true**,
//! so `AbstractKnnVectorQuery.rewrite` wraps it in an
//! `OptimisticKnnCollectorManager` that sizes each leaf's collector at
//! [`per_leaf_top_k`]`(k, leafMaxDoc / indexMaxDoc)` -- *not* `k` -- and then
//! runs a second, re-entrant pass over any leaf whose worst collected hit is
//! still at or above the merged top-`k`'s worst. Searching every leaf for `k`
//! returns a different, usually *better* answer, which is exactly why it
//! cannot be substituted: the differential fixture pins Lucene's answer, not
//! the best one.

use lucene_codecs::backward_codecs::hnsw_vectors::RetiredHnswVectorsReader;
use lucene_codecs::backward_codecs::quantized_vectors::{QuantizedVectorsReader, SearchKind};
use lucene_codecs::buffer_lists::ivf_accepted_target;
use lucene_codecs::field_infos::{FieldInfos, VectorEncoding, VectorSimilarityFunction};
use lucene_codecs::hnsw::{KnnCollect, KnnCollector, VectorScorer};
use lucene_codecs::hnsw_vectors::{self, HnswVectorsReader, OffHeapHnswGraph, SearchOptions};
use lucene_codecs::vectors::{FlatFieldEntry, FlatVectorsReader};
use lucene_util::fixed_bit_set::FixedBitSet;

use crate::collector::{ScoreDoc, ScoringCollector};
use crate::knn_collectors::{AdaptiveBeamCollector, FilteredWalkPlanner};
use crate::multi_segment::merge_multi_segment_scored;
use crate::{Error, Result};

mod grouped;
pub use grouped::{
    search_grouped_byte_knn_per_segment, search_grouped_float_knn_per_segment, GroupFill,
    GroupMembers, SegmentGroups,
};

/// How a leaf's collector diversifies its hits.
#[derive(Clone, Copy)]
enum Diversify<'a> {
    /// The best child of each parent the bit set closes; `None`: a leaf
    /// without parents, which has no results.
    Parents(Option<&'a FixedBitSet>),
    /// The best documents of each group ([`grouped`]).
    Groups(SegmentGroups<'a>),
}

/// `Lucene99HnswVectorsReader.EXHAUSTIVE_BULK_SCORE_ORDS`: how many ordinals
/// an exhaustive scan scores per batch, so one `bulkScore` maximum can retire
/// a whole batch against the collector's competitive threshold.
const EXHAUSTIVE_BULK_SCORE_ORDS: usize = 64;

/// `AbstractKnnVectorQuery.LAMBDA`: "constant controlling the degree of
/// additional result exploration done during pro-rata search of segments".
const LAMBDA: f64 = 16.0;

/// Port of `AbstractKnnVectorQuery.perLeafTopKCalculation`: a leaf's expected
/// share of the global top `k` (`k * leafProportion`) plus three standard
/// deviations of the binomial, so there is ~95% probability the leaf's true
/// contribution is no larger.
///
/// The float/double split is Java's and is kept deliberately: `k *
/// leafProportion` and the variance are `float` (Java's `int * float`),
/// `Math.sqrt` widens to `double`, and the `(int)` cast truncates toward
/// zero. A rounding difference here moves the collector size by one and
/// therefore moves which documents an approximate search returns, so this is
/// a bit-level port, not a formula that merely looks the same.
pub fn per_leaf_top_k(k: usize, leaf_proportion: f32) -> usize {
    let kp: f32 = k as f32 * leaf_proportion;
    let variance: f32 = kp * (1.0 - leaf_proportion);
    let v: f64 = kp as f64 + LAMBDA * (variance as f64).sqrt();
    // `Math.max(1, ..)` widens the 1 to a double. One deliberate divergence
    // lives here: a zero-document index makes `leafProportion` NaN, and Java's
    // `Math.max` *propagates* NaN where Rust's `f64::max` returns the non-NaN
    // operand -- so Java yields `(int) NaN == 0` and this yields `1`. Lucene
    // treats its own 0 as a bug (`AbstractKnnVectorQuery`: "if we divided by
    // zero above, leafProportion can be NaN and then this would be 0",
    // immediately above `assert perLeafTopK > 0`), and the two are observably
    // the same anyway: an index with no documents has no vectors, so a
    // collector of 0 and one of 1 both come back empty. `1` is chosen because
    // a zero-sized collector is a worse thing to hand downstream.
    let clamped = v.max(1.0);
    if clamped >= i32::MAX as f64 {
        i32::MAX as usize
    } else {
        clamped as usize
    }
}

/// One already-opened segment's vector inputs -- the KNN sibling of
/// [`crate::points_query::PointsInput`], and the reason this module needs no
/// term dictionary at all: a vector field has none, and real Lucene's
/// `KnnVectorsReader` is a per-segment reader entirely independent of
/// `FieldsProducer`.
pub struct VectorsInput<'d> {
    /// `Lucene99FlatVectorsReader` over this segment's `.vemf`/`.vec`.
    pub flat: FlatVectorsReader<'d>,
    /// `Lucene99HnswVectorsReader` over this segment's `.vem`/`.vex` (or a
    /// retired 9.x reader, see [`GraphReader`]), or `None` when the caller
    /// opened no graph. `None` makes every search the exhaustive scan Java
    /// also falls back to -- exact, just `O(size)`.
    pub hnsw: Option<GraphReader<'d>>,
    /// The segment's `.fnm`: the only place a field *name* maps to the field
    /// *number* the vector formats key everything by.
    pub field_infos: &'d FieldInfos,
    /// `LeafReader.getLiveDocs()`; `None` for a segment with no deletions.
    pub live_docs: Option<&'d FixedBitSet>,
    /// The filter query's matching documents in this segment, if any -- see
    /// this module's doc comment for why it is an input rather than a
    /// `Query`. `None` is Java's `filterWeight == null`, which is a
    /// *different path* and not merely a filter that accepts everything:
    /// Java then skips the cost heuristic and the `visitedLimit` cap
    /// entirely.
    pub filter: Option<&'d FixedBitSet>,
    /// `SegmentInfo.maxDoc()`.
    pub max_doc: i32,
}

/// The KNN reader a segment's vector fields were written with, as far as the
/// graph walk is concerned: `PerFieldKnnVectorsFormat`'s per-field delegate.
///
/// A retired (`Lucene90`..`Lucene95`) reader also owns its flat vectors --
/// [`RetiredHnswVectorsReader::flat`] is what [`VectorsInput::flat`] should
/// be for such a segment -- and so does a retired quantized one
/// ([`QuantizedVectorsReader::flat`]).
#[derive(Debug, Clone)]
pub enum GraphReader<'d> {
    /// `Lucene99HnswVectorsReader`.
    Lucene99(HnswVectorsReader<'d>),
    /// `Lucene9{0,1,2,4,5}HnswVectorsReader`: no exhaustive-scan branch, and
    /// `Lucene90` walks its own single-level graph.
    Retired(RetiredHnswVectorsReader<'d>),
    /// `Lucene99(Hnsw)ScalarQuantizedVectorsFormat` or
    /// `Lucene102(Hnsw)BinaryQuantizedVectorsFormat`: a float query is scored
    /// on the quantized codes, and each format searches its own way
    /// ([`SearchKind`]).
    Quantized(QuantizedVectorsReader<'d>),
    /// `Lucene104HnswScalarQuantizedVectorsFormat`: the HNSW graph (`None`
    /// for a field indexed without one) walked, and the exhaustive branch
    /// scanned, on the scalar-quantized codes (`getRandomVectorScorer` of
    /// `Lucene104ScalarQuantizedVectorsReader`).
    ScalarQuantized {
        graph: Option<HnswVectorsReader<'d>>,
        codes: lucene_codecs::scalar_quantized_vectors::ScalarQuantizedVectorsReader<'d>,
    },
    /// `IvfVectorsFormat`: a float query probes the field's
    /// clusters and reranks on its rerank vectors ([`ivf_leaf_results`]),
    /// keeping the best child per parent for a diversifying search; a byte
    /// field and a selective filter scan the raw vectors exhaustively.
    Ivf(std::sync::Arc<lucene_ivf::IvfReader>),
    /// `HnswQuantizedVectorsFormat`: the HNSW graph (`None` for a
    /// field indexed without one) walked on the field's quantized codes,
    /// the candidates reranked on the raw vectors; the exhaustive branch, a
    /// diversifying search and a byte field score the raw vectors.
    HnswQuantized {
        graph: Option<HnswVectorsReader<'d>>,
        codes: std::sync::Arc<lucene_index::hnsw_quant_vectors::HnswCodesReader>,
    },
    /// `DiskHnswVectorsFormat`: the HNSW graph, plus the disk layout of the
    /// field. A float search without parents walks the disk layout
    /// ([`disk_hnsw_leaf_results`]); a diversifying search, a leaf with a
    /// deadline or a patience collector, and a filter too selective to walk
    /// take the graph on the raw vectors.
    HnswDisk {
        graph: Option<HnswVectorsReader<'d>>,
        disk: std::sync::Arc<lucene_disk_hnsw::DiskHnswReader>,
    },
}

impl<'d> From<HnswVectorsReader<'d>> for GraphReader<'d> {
    fn from(r: HnswVectorsReader<'d>) -> Self {
        GraphReader::Lucene99(r)
    }
}

impl<'d> From<RetiredHnswVectorsReader<'d>> for GraphReader<'d> {
    fn from(r: RetiredHnswVectorsReader<'d>) -> Self {
        GraphReader::Retired(r)
    }
}

impl<'d> From<QuantizedVectorsReader<'d>> for GraphReader<'d> {
    fn from(r: QuantizedVectorsReader<'d>) -> Self {
        GraphReader::Quantized(r)
    }
}

/// One leaf field's graph, resolved: what [`leaf_results`] walks -- the
/// field's reader's `search(field, target, knnCollector, acceptDocs)`.
enum LeafGraph<'a, 'd> {
    /// `Lucene99HnswVectorsReader.search`: the graph (or `None` for none),
    /// with its own graph-versus-scan choice.
    Lucene99(Option<OffHeapHnswGraph<'d>>),
    /// A retired reader's `search`, for `field_number`.
    Retired(&'a RetiredHnswVectorsReader<'d>, i32),
    /// `FlatVectorsReader.search`: a flat format that indexed no graph
    /// collects nothing ([`SearchKind::Nothing`]).
    Nothing,
    /// `Lucene102BinaryQuantizedVectorsReader.search` ([`SearchKind::ScanAll`]).
    ScanAll,
}

impl LeafGraph<'_, '_> {
    /// The reader's `search` into `collector`.
    fn search_with<S: VectorScorer, C: KnnCollect + ?Sized>(
        &self,
        scorer: &mut S,
        collector: &mut C,
        options: SearchOptions<'_>,
    ) -> Result<()> {
        match self {
            LeafGraph::Lucene99(graph) => {
                hnsw_vectors::search_with(scorer, graph.as_ref(), collector, options)?
            }
            LeafGraph::Retired(reader, field) => reader.search_with(
                *field,
                scorer,
                collector,
                options.accept_ords,
                options.seed_ords,
            )?,
            LeafGraph::Nothing => {}
            LeafGraph::ScanAll => {
                // `if (knnCollector.k() == 0) return;` then every accepted
                // ordinal, collected and counted one at a time, with no
                // early-termination check.
                if collector.k() == 0 {
                    return Ok(());
                }
                for ord in 0..scorer.max_ord() {
                    if options.accept_ords.is_none_or(|b| b.get_doc(ord)) {
                        let score = scorer.score(ord)?;
                        collector.collect(ord, score);
                        collector.inc_visited_count(1);
                    }
                }
            }
        }
        Ok(())
    }

    /// The reader's `search` for a collector of `k` (Java's
    /// `TopKnnCollector`, decorated by `extras`), returning its hits, best
    /// first, and whether it early-terminated.
    fn search<S: VectorScorer>(
        &self,
        scorer: &mut S,
        k: usize,
        visit_limit: u64,
        options: SearchOptions<'_>,
        extras: &LeafExtras,
    ) -> Result<(Vec<(i32, f32)>, bool)> {
        if extras.patience.is_some() || extras.deadline.is_some() {
            // Patience and deadlines wrap the collector; see [`approximate`].
            return approximate(scorer, self, k, visit_limit, options, extras);
        }
        if let (Some((hits, slack)), Some(similarity)) =
            (extras.exploration_slack, extras.similarity)
        {
            if scorer.max_ord() <= 0 || k == 0 {
                return Ok((Vec::new(), false));
            }
            if let Some(mut c) = AdaptiveBeamCollector::new(
                KnnCollector::new(k, visit_limit),
                hits,
                slack,
                similarity,
            ) {
                self.search_with(scorer, &mut c, options)?;
                let c = c.into_inner();
                let early = c.early_terminated();
                return Ok((c.top_docs(), early));
            }
        }
        Ok(match self {
            LeafGraph::Lucene99(graph) => {
                hnsw_vectors::search(scorer, graph.as_ref(), k, visit_limit, options)?
            }
            LeafGraph::Retired(reader, field) => reader.search(
                *field,
                scorer,
                k,
                visit_limit,
                options.accept_ords,
                options.seed_ords,
            )?,
            LeafGraph::Nothing | LeafGraph::ScanAll => {
                if scorer.max_ord() <= 0 || k == 0 {
                    return Ok((Vec::new(), false));
                }
                let mut collector = KnnCollector::new(k, visit_limit);
                self.search_with(scorer, &mut collector, options)?;
                let early = collector.early_terminated();
                (collector.top_docs(), early)
            }
        })
    }

    /// `Lucene90HnswVectorsReader.search` collects ordinals as documents; see
    /// [`RetiredHnswVectorsReader::hits_are_ordinals`].
    fn hits_are_ordinals(&self) -> bool {
        matches!(self, LeafGraph::Retired(r, _) if r.hits_are_ordinals())
    }
}

/// `Lucene99HnswVectorsReader`'s graph for a field, with its "unknown field"
/// mapped to the caller mistake it is.
fn lucene99_graph<'d>(
    reader: &HnswVectorsReader<'d>,
    field_number: i32,
) -> Result<Option<OffHeapHnswGraph<'d>>> {
    reader.graph(field_number).map_err(|e| match e {
        // A field the `.vemf` has and the `.vem` does not is a caller
        // mistake, not a damaged index -- Java's `getFieldEntryOrThrow`
        // raises `IllegalArgumentException` for it. Unreachable with
        // Lucene-written files (both metas list every vector field),
        // but the two must not be confused: `lucene-ffi` turns a decode
        // error into "this index is corrupt".
        lucene_codecs::vectors::Error::UnknownField(number) => Error::InvalidKnnQuery(format!(
            "field number {number} has vectors but no HNSW graph entry in this segment's .vem"
        )),
        other => Error::Vectors(other),
    })
}

/// The field's reader's search, resolved for one leaf. The graph is
/// optional twice over for `Lucene99HnswVectorsReader`: the caller may have
/// opened no `.vem`/`.vex`, and a field written below
/// `HNSW_GRAPH_THRESHOLD` documents carries none even when they were
/// opened. Both mean the same thing -- take the exhaustive branch.
fn leaf_graph<'a, 'd>(
    input: &'a VectorsInput<'d>,
    field_number: i32,
    float_target: bool,
) -> Result<LeafGraph<'a, 'd>> {
    Ok(match &input.hnsw {
        None | Some(GraphReader::Ivf(_)) => LeafGraph::Lucene99(None),
        Some(GraphReader::Retired(reader)) => LeafGraph::Retired(reader, field_number),
        Some(GraphReader::Lucene99(reader)) => {
            LeafGraph::Lucene99(lucene99_graph(reader, field_number)?)
        }
        Some(GraphReader::ScalarQuantized { graph, .. })
        | Some(GraphReader::HnswQuantized { graph, .. })
        | Some(GraphReader::HnswDisk { graph, .. }) => LeafGraph::Lucene99(match graph {
            Some(reader) => lucene99_graph(reader, field_number)?,
            None => None,
        }),
        Some(GraphReader::Quantized(reader)) => {
            match (reader.format().search_kind(), reader.graph()) {
                (SearchKind::Hnsw, Some(graphs)) => {
                    LeafGraph::Lucene99(lucene99_graph(graphs, field_number)?)
                }
                // A quantized field with the HNSW kind always has its graph
                // reader (`QuantizedVectorsReader::open` requires it).
                (SearchKind::Hnsw, None) | (SearchKind::Nothing, _) => LeafGraph::Nothing,
                // A byte field of the binary format goes to the raw reader's
                // `search`, which is `FlatVectorsReader`'s: nothing.
                (SearchKind::ScanAll, _) if float_target => LeafGraph::ScanAll,
                (SearchKind::ScanAll, _) => LeafGraph::Nothing,
            }
        }
    })
}

/// A float query's scorer over a `Lucene104` scalar-quantized field's codes;
/// `None` for any other reader, and for a field the instance holds no codes
/// for or whose codes are empty, which Java scores on the raw vectors.
fn scalar_quantized_scorer<'d>(
    input: &VectorsInput<'d>,
    resolved: &ResolvedField,
    target: &[f32],
) -> Result<Option<lucene_codecs::scalar_quantized_vectors::OwnedQuantizedQueryScorer<'d>>> {
    let Some(GraphReader::ScalarQuantized { codes, .. }) = &input.hnsw else {
        return Ok(None);
    };
    match codes.field(resolved.field_number) {
        Some(entry) if entry.size > 0 => {}
        _ => return Ok(None),
    }
    let values = codes.quantized_vector_values(resolved.field_number)?;
    Ok(Some(
        lucene_codecs::scalar_quantized_vectors::OwnedQuantizedQueryScorer::new(values, target)?,
    ))
}

/// Turns a doc-id list -- e.g. straight out of
/// [`crate::resolve_clause_docs`] -- into the bitset
/// [`VectorsInput::filter`] wants.
///
/// A doc id at or past `max_doc` is dropped rather than panicking or
/// widening the bitset: a filter resolved against a *different* segment is a
/// caller mistake this module cannot detect, and widening would let it
/// accept documents that do not exist.
pub fn accept_bitset(docs: impl IntoIterator<Item = i32>, max_doc: i32) -> FixedBitSet {
    let mut bits = FixedBitSet::new(max_doc.max(0) as usize);
    for doc in docs {
        if doc >= 0 && (doc as usize) < bits.len() {
            bits.set(doc as usize);
        }
    }
    bits
}

/// `KnnFloatVectorQuery`: a field, a target vector and `k`.
///
/// The three fields after `k` have no counterpart on Java's query object;
/// [`KnnFloatVectorQuery::new`] leaves all three at the values that
/// reproduce `KnnFloatVectorQuery` exactly.
#[derive(Debug, Clone, PartialEq)]
pub struct KnnFloatVectorQuery {
    pub field: String,
    pub target: Vec<f32>,
    pub k: usize,
    /// OpenSearch's `num_candidates`, which Lucene has no equivalent of:
    /// `KnnFloatVectorQuery` searches each leaf with a collector of exactly
    /// that leaf's `k`. `0` therefore reproduces Lucene exactly; a larger
    /// value widens the beam and truncates back down -- strictly more work
    /// for strictly better recall, never a different *kind* of answer.
    pub ef_search: usize,
    /// The collector's `visitLimit`. `0` is Java's unfiltered default
    /// (`Integer.MAX_VALUE`, i.e. never early-terminate); the filtered path
    /// caps it at `cost + 1` the way Java does regardless.
    pub visited_limit: u64,
    /// A cross-check, not an override. `None` means "the field's own", which
    /// is what Lucene always does (`FieldInfo` owns the similarity and
    /// `KnnFloatVectorQuery` has no such parameter); `Some(s)` requires the
    /// field to have been written with `s`. The reason is not stylistic: the
    /// HNSW graph's arcs encode the build-time similarity's neighbourhood, so
    /// walking it under another one silently degrades recall with no error at
    /// all.
    pub similarity: Option<VectorSimilarityFunction>,
    /// `KnnSearchStrategy.Hnsw(filteredSearchThreshold)`: a filtered leaf
    /// passing fewer than this percentage of its graph walks level 0 with
    /// `FilteredHnswGraphSearcher`. `0`, Lucene's default, never does.
    pub filtered_search_threshold: i32,
    /// The clusters an IVF field's search probes in each segment
    /// (`IvfVectorsFormat`); `0` leaves it to the depth each
    /// segment calibrated when it was written. Other formats ignore it.
    pub ivf_nprobes: usize,
    /// `(hits, slack)`: an HNSW walk stops once its best unexplored
    /// candidate is farther than `1 + slack` times the `hits`-th best node
    /// it collected, instead of once it no longer beats the worst of its
    /// `k` ([`crate::knn_collectors::AdaptiveBeamCollector`], Vespa's
    /// `exploration-slack`). Opt-in, with no Lucene equivalent: the walk's
    /// depth then follows the query's neighbourhood, which finds more of the
    /// true `hits` nearest per visited node than a wider `ef_search`
    /// (`docs/benchmarks/hnsw.md`, "Adaptive beam"). `k` still sizes the
    /// collector. Ignored by maximum inner product fields, IVF fields, and
    /// searches with a patience or a deadline.
    pub exploration_slack: Option<(usize, f32)>,
    /// A scale on an IVF leaf's search depth, `1.0` (the default) leaving
    /// it as asked: its probes and its rerank cap are multiplied by it. A
    /// recall-targeted search ([`crate::knn_recall`]) sets it, scaling an
    /// HNSW search's `k` (its candidates) itself. Other leaves ignore it.
    pub search_effort: f32,
}

/// `KnnByteVectorQuery`: [`KnnFloatVectorQuery`] over a BYTE-encoded field.
///
/// `target` is Java's *signed* `byte[]` verbatim -- the byte kernels in
/// [`lucene_codecs::vectors`] sign-extend it exactly as Java's `byte` does,
/// so a caller passes the same bytes it would hand `KnnByteVectorQuery`.
#[derive(Debug, Clone, PartialEq)]
pub struct KnnByteVectorQuery {
    pub field: String,
    pub target: Vec<u8>,
    pub k: usize,
    /// See [`KnnFloatVectorQuery::ef_search`].
    pub ef_search: usize,
    /// See [`KnnFloatVectorQuery::visited_limit`].
    pub visited_limit: u64,
    /// See [`KnnFloatVectorQuery::similarity`].
    pub similarity: Option<VectorSimilarityFunction>,
    /// `KnnSearchStrategy.Hnsw(filteredSearchThreshold)`: a filtered leaf
    /// passing fewer than this percentage of its graph walks level 0 with
    /// `FilteredHnswGraphSearcher`. `0`, Lucene's default, never does.
    pub filtered_search_threshold: i32,
    /// The clusters an IVF field's search probes in each segment
    /// (`IvfVectorsFormat`); `0` leaves it to the depth each
    /// segment calibrated when it was written. Other formats ignore it.
    pub ivf_nprobes: usize,
    /// See [`KnnFloatVectorQuery::exploration_slack`].
    pub exploration_slack: Option<(usize, f32)>,
    /// A scale on an IVF leaf's search depth, `1.0` (the default) leaving
    /// it as asked: its probes and its rerank cap are multiplied by it. A
    /// recall-targeted search ([`crate::knn_recall`]) sets it, scaling an
    /// HNSW search's `k` (its candidates) itself. Other leaves ignore it.
    pub search_effort: f32,
}

macro_rules! knn_query_impl {
    ($t:ident, $elem:ty, $encoding:expr, $variant:ident) => {
        impl $t {
            /// Java's constructor, including its `k < 1` rejection
            /// (`IllegalArgumentException`, *"k must be at least 1"*).
            pub fn new(field: impl Into<String>, target: Vec<$elem>, k: usize) -> Result<Self> {
                check_k(k)?;
                Ok(Self {
                    field: field.into(),
                    target,
                    k,
                    ef_search: 0,
                    visited_limit: 0,
                    similarity: None,
                    filtered_search_threshold: 0,
                    ivf_nprobes: 0,
                    exploration_slack: None,
                    search_effort: 1.0,
                })
            }

            /// See [`KnnFloatVectorQuery::ivf_nprobes`].
            pub fn with_ivf_nprobes(mut self, nprobes: usize) -> Self {
                self.ivf_nprobes = nprobes;
                self
            }

            /// See [`KnnFloatVectorQuery::search_effort`].
            pub fn with_search_effort(mut self, effort: f32) -> Self {
                self.search_effort = effort;
                self
            }

            /// See [`KnnFloatVectorQuery::ef_search`].
            pub fn with_ef_search(mut self, ef_search: usize) -> Self {
                self.ef_search = ef_search;
                self
            }

            /// See [`KnnFloatVectorQuery::visited_limit`].
            pub fn with_visited_limit(mut self, visited_limit: u64) -> Self {
                self.visited_limit = visited_limit;
                self
            }

            /// See [`KnnFloatVectorQuery::similarity`].
            pub fn with_similarity(mut self, similarity: VectorSimilarityFunction) -> Self {
                self.similarity = Some(similarity);
                self
            }

            /// See [`KnnFloatVectorQuery::filtered_search_threshold`].
            pub fn with_filtered_search_threshold(mut self, threshold: i32) -> Self {
                self.filtered_search_threshold = threshold;
                self
            }

            /// See [`KnnFloatVectorQuery::exploration_slack`]. A `slack`
            /// that is not a non-negative number, or `hits` of `0`, is
            /// rejected.
            pub fn with_exploration_slack(mut self, hits: usize, slack: f32) -> Result<Self> {
                if hits == 0 || !slack.is_finite() || slack < 0.0 {
                    return Err(Error::InvalidKnnQuery(format!(
                        "exploration slack needs hits >= 1 and a finite slack >= 0, got {hits} and {slack}"
                    )));
                }
                self.exploration_slack = Some((hits, slack));
                Ok(self)
            }
        }

        impl KnnQuery for $t {
            const ENCODING: VectorEncoding = $encoding;

            fn k(&self) -> usize {
                self.k
            }
            fn field(&self) -> &str {
                &self.field
            }
            fn target(&self) -> Target<'_> {
                Target::$variant(&self.target)
            }
            fn ef_search(&self) -> usize {
                self.ef_search
            }
            fn visited_limit(&self) -> u64 {
                self.visited_limit
            }
            fn similarity(&self) -> Option<VectorSimilarityFunction> {
                self.similarity
            }
            fn filtered_search_threshold(&self) -> i32 {
                self.filtered_search_threshold
            }
            fn ivf_nprobes(&self) -> usize {
                self.ivf_nprobes
            }
            fn exploration_slack(&self) -> Option<(usize, f32)> {
                self.exploration_slack
            }
            fn search_effort(&self) -> f32 {
                self.search_effort
            }
        }
    };
}

knn_query_impl!(KnnFloatVectorQuery, f32, VectorEncoding::Float32, Float);
knn_query_impl!(KnnByteVectorQuery, u8, VectorEncoding::Byte, Byte);

/// What the two encodings' queries share, so the fan-out, the per-leaf plan
/// and the field preflight are written once (`AbstractKnnVectorQuery` is
/// exactly this seam in Java).
trait KnnQuery {
    const ENCODING: VectorEncoding;
    fn k(&self) -> usize;
    fn field(&self) -> &str;
    fn target(&self) -> Target<'_>;
    fn ef_search(&self) -> usize;
    fn visited_limit(&self) -> u64;
    fn similarity(&self) -> Option<VectorSimilarityFunction>;
    fn filtered_search_threshold(&self) -> i32;
    fn ivf_nprobes(&self) -> usize;
    fn exploration_slack(&self) -> Option<(usize, f32)> {
        None
    }
    fn search_effort(&self) -> f32;
}

#[derive(Clone, Copy)]
enum Target<'q> {
    Float(&'q [f32]),
    Byte(&'q [u8]),
}

fn check_k(k: usize) -> Result<()> {
    if k < 1 {
        return Err(Error::InvalidKnnQuery(format!(
            "k must be at least 1, got: {k}"
        )));
    }
    Ok(())
}

/// The `.vemf`/`.vem` similarity ordinals, which are
/// `Lucene94FieldInfosFormat`'s pinned list and **not** the Java enum's
/// declaration order -- the same four values
/// `lucene_codecs::vectors::read_similarity_function` decodes from the file.
pub fn similarity_ordinal(s: VectorSimilarityFunction) -> i32 {
    match s {
        VectorSimilarityFunction::Euclidean => 0,
        VectorSimilarityFunction::DotProduct => 1,
        VectorSimilarityFunction::Cosine => 2,
        VectorSimilarityFunction::MaximumInnerProduct => 3,
    }
}

/// The inverse of [`similarity_ordinal`]; `None` for a value that is not one
/// of the four.
pub fn similarity_from_ordinal(ordinal: i32) -> Option<VectorSimilarityFunction> {
    match ordinal {
        0 => Some(VectorSimilarityFunction::Euclidean),
        1 => Some(VectorSimilarityFunction::DotProduct),
        2 => Some(VectorSimilarityFunction::Cosine),
        3 => Some(VectorSimilarityFunction::MaximumInnerProduct),
        _ => None,
    }
}

/// One leaf's resolved field: the field number plus its `.vemf` entry.
struct ResolvedField {
    field_number: i32,
    entry: FlatFieldEntry,
}

/// `AbstractKnnVectorQuery`'s per-leaf preflight: name -> number, the
/// encoding check both subclasses make, the similarity cross-check, and
/// Java's own dimension check.
fn resolve_field<Q: KnnQuery>(input: &VectorsInput<'_>, query: &Q) -> Result<ResolvedField> {
    let field = query.field();
    let Some(info) = input.field_infos.field_by_name(field) else {
        return Err(Error::InvalidKnnQuery(format!(
            "unknown field {field:?} in this segment's .fnm"
        )));
    };
    let field_number = info.number;
    let Some(entry) = input.flat.field(field_number) else {
        return Err(Error::InvalidKnnQuery(format!(
            "field {field:?} (number {field_number}) has no vectors in this segment"
        )));
    };
    // Java's own check, in `AbstractKnnVectorQuery`'s two subclasses: a
    // `KnnByteVectorQuery` on a FLOAT32 field (or vice versa) is an error,
    // not a reinterpretation of the bytes.
    if entry.encoding != Q::ENCODING {
        return Err(Error::InvalidKnnQuery(format!(
            "field {field:?} is {:?}-encoded, but this call searches {:?} vectors",
            entry.encoding,
            Q::ENCODING
        )));
    }
    if let Some(requested) = query.similarity() {
        if requested != entry.similarity {
            return Err(Error::InvalidKnnQuery(format!(
                "similarity {} does not match the field's own {} -- the HNSW graph's arcs were \
                 built with the field's similarity, so searching it with another one silently \
                 degrades recall",
                similarity_ordinal(requested),
                similarity_ordinal(entry.similarity)
            )));
        }
    }
    // Java's own check and its own message shape (`AbstractKnnVectorQuery`:
    // "vector query dimension: X differs from field dimension: Y"). Made
    // here, before the scorer is built, because the reader reports the same
    // mismatch as a *decode* error -- right for a corrupt file, wrong for a
    // caller who passed a wrong-length vector.
    let target_len = match query.target() {
        Target::Float(t) => t.len(),
        Target::Byte(t) => t.len(),
    };
    if target_len != entry.dimension as usize {
        return Err(Error::InvalidKnnQuery(format!(
            "vector query dimension: {target_len} differs from field dimension: {}",
            entry.dimension
        )));
    }
    Ok(ResolvedField {
        field_number,
        entry: entry.clone(),
    })
}

/// Port of `AcceptDocs` reduced to what this port can express: one leaf's
/// accepted documents, already translated into **ordinal** space.
///
/// Ordinal space rather than doc space is what makes Java's implicit
/// `FieldExistsQuery(field)` conjunct free -- an ordinal exists only for a
/// document that has a vector -- and it makes `cardinality()` here exactly
/// `AcceptDocs.cost()` there. Java's cost is *also* exact (a
/// `BitSet.cardinality()`, not an estimate), which is what makes the
/// exact-search heuristic in [`leaf_results`] portable rather than a guessed
/// threshold.
enum AcceptOrds<'d> {
    /// The identity case: the field is dense, so ordinal == doc id and the
    /// caller's doc-space bitset already *is* an ordinal-space one. Nothing
    /// is copied; Java always allocates a `Bits` wrapper here
    /// (`KnnVectorValues.getAcceptOrds`) and pays a virtual call per visited
    /// node.
    Borrowed(&'d FixedBitSet),
    Owned(FixedBitSet),
}

impl AcceptOrds<'_> {
    fn bits(&self) -> &FixedBitSet {
        match self {
            AcceptOrds::Borrowed(b) => b,
            AcceptOrds::Owned(b) => b,
        }
    }
}

/// `KnnVectorValues.getAcceptOrds(acceptDocs)`, over Java's
/// `liveDocs`-intersected filter set.
///
/// `None` means everything is accepted (no deletions, no filter), which is
/// Java's `null` `Bits` and the fastest graph walk.
fn accept_ords<'a, 'v>(
    input: &VectorsInput<'a>,
    resolved: &ResolvedField,
    values: &OrdDocMaps<'_, 'v>,
) -> Result<Option<AcceptOrds<'a>>> {
    let size = resolved.entry.size;
    if input.live_docs.is_none() && input.filter.is_none() {
        return Ok(None);
    }
    // A field whose ordinals *are* its doc ids (`OrdToDoc::Dense`, and
    // trivially `Empty`) needs no translation at all, as long as the bitset
    // covers every ordinal the walk can ask about.
    let identity = resolved.entry.ord_to_doc.is_dense() || resolved.entry.ord_to_doc.is_empty();
    if identity && input.filter.is_none() {
        if let Some(live) = input.live_docs {
            if live.len() >= size as usize {
                return Ok(Some(AcceptOrds::Borrowed(live)));
            }
        }
    }
    if identity {
        return Ok(Some(AcceptOrds::Owned(identity_accept_ords(
            input.live_docs,
            input.filter,
            size.max(0) as usize,
        ))));
    }
    if let Some(bits) = accept_ords_from_docs(input, size, values)? {
        return Ok(Some(AcceptOrds::Owned(bits)));
    }
    let mut bits = FixedBitSet::new(size.max(0) as usize);
    // Every ordinal's document, decoded in one bulk pass (`ordToDoc` per
    // ordinal is a `DirectMonotonicReader` lookup each).
    let mut docs = Vec::new();
    (values.ord_to_docs)(&mut docs)?;
    for ord in 0..size {
        // `ord_to_docs` yields exactly `size` documents; a short list
        // would only leave the rest unaccepted.
        let doc = docs.get(ord as usize).copied().unwrap_or(-1);
        if doc < 0 {
            continue;
        }
        // `get_doc` is the sanctioned way to ask a bitset about an id that did
        // not come from it -- here, a doc id the flat vector store's
        // ordinal-to-doc map produced, against the caller's live-docs and
        // filter bitsets. See `FixedBitSet::get_doc`.
        let live = input.live_docs.is_none_or(|b| b.get_doc(doc));
        let passes = input.filter.is_none_or(|b| b.get_doc(doc));
        if live && passes {
            // FBS: `bits` is `FixedBitSet::new(size.max(0))` immediately above
            // and `ord` runs `0..size`.
            bits.set(ord as usize);
        }
    }
    Ok(Some(AcceptOrds::Owned(bits)))
}

/// [`accept_ords`] for a field whose ordinals are its doc ids: `live AND
/// filter` a word at a time, instead of two `get_doc` probes and a `set` per
/// ordinal. A source bitset shorter than `size` reads as zero past its end
/// (its own ghost bits are clear), which is `get_doc`'s "past the end is not
/// accepted"; the last word is masked to `size`.
fn identity_accept_ords(
    live: Option<&FixedBitSet>,
    filter: Option<&FixedBitSet>,
    size: usize,
) -> FixedBitSet {
    let num_words = size.div_ceil(64);
    let word = |b: Option<&FixedBitSet>, i: usize| match b {
        None => u64::MAX,
        Some(b) => b.words().get(i).copied().unwrap_or(0),
    };
    let mut words: Vec<u64> = (0..num_words)
        .map(|i| word(live, i) & word(filter, i))
        .collect();
    if !size.is_multiple_of(64) {
        if let Some(last) = words.last_mut() {
            *last &= (1u64 << (size % 64)) - 1;
        }
    }
    FixedBitSet::from_words(words, size)
}

/// A leaf's two directions between ordinals and documents, as
/// [`accept_ords`] uses them.
struct OrdDocMaps<'f, 'v> {
    /// `ordToDoc`.
    ord_to_doc: &'f dyn Fn(i32) -> Result<i32>,
    /// `ordToDoc` for every ordinal, in order.
    ord_to_docs: &'f dyn Fn(&mut Vec<i32>) -> Result<()>,
    /// The doc -> ordinal iterator (`KnnVectorValues.iterator()`).
    doc_to_ord: &'f dyn Fn() -> Result<lucene_codecs::vectors::DocToOrdCursor<'v>>,
}

/// [`accept_ords`] for a sparse field, from the **document** side: the
/// filter's documents (or, with no filter, the deleted ones) looked up in the
/// field's doc -> ordinal iterator, rather than every ordinal translated to
/// its document. The same set: an ordinal is accepted exactly when its
/// document is live and passes the filter. Java answers the same question
/// lazily (`getAcceptOrds` tests a node's document when the walk visits it);
/// this touches only the documents that can change the answer, where the
/// ordinal-side loop decodes the whole `ordToDoc` map per query. `None`
/// when the field's last document lies past the live-docs bitset (a
/// mismatched reader), which the ordinal-side loop handles.
fn accept_ords_from_docs(
    input: &VectorsInput<'_>,
    size: i32,
    values: &OrdDocMaps<'_, '_>,
) -> Result<Option<FixedBitSet>> {
    let ords = size.max(0) as usize;
    if ords == 0 {
        return Ok(None);
    }
    let mut cursor = (values.doc_to_ord)()?;
    // ARITH: `ords >= 1` above.
    #[allow(clippy::arithmetic_side_effects)]
    let last_doc = (values.ord_to_doc)(size - 1)?;
    let mut bits = FixedBitSet::new(ords);
    match (input.filter, input.live_docs) {
        (Some(filter), live) if filter.cardinality() >= ords / RANK_WALK_INVERSE_DENSITY => {
            accept_ords_by_rank(&cursor, filter, live, last_doc, &mut bits)?;
        }
        (Some(filter), live) => {
            let mut next = filter.next_set_bit(0);
            while let Some(doc) = next {
                let doc_id = i32::try_from(doc).unwrap_or(i32::MAX);
                if live.is_none_or(|l| l.get_doc(doc_id)) {
                    if let Some(ord) = cursor.ordinal(doc_id)? {
                        if (ord as usize) < ords {
                            // FBS: bounded by the check above.
                            bits.set(ord as usize);
                        }
                    }
                }
                next = doc
                    .checked_add(1)
                    .and_then(|from| filter.next_set_bit(from));
            }
        }
        (None, Some(live)) => {
            if usize::try_from(last_doc).map_or(true, |d| d >= live.len()) {
                return Ok(None);
            }
            bits.set_range(0, ords);
            let mut next = live.next_clear_bit(0);
            while let Some(doc) = next {
                // Every document the field has is below `live.len()`.
                if doc > last_doc as usize {
                    break;
                }
                if let Some(ord) = cursor.ordinal(doc as i32)? {
                    if (ord as usize) < ords {
                        // FBS: bounded by the check above.
                        bits.clear(ord as usize);
                    }
                }
                next = doc
                    .checked_add(1)
                    .and_then(|from| live.next_clear_bit(from));
            }
        }
        (None, None) => return Ok(None),
    }
    Ok(Some(bits))
}

/// A filter passing at least one in this many of a field's vectors is
/// turned into ordinals by [`accept_ords_by_rank`] rather than a lookup per
/// document: the rank walk reads the field's whole document set once, which
/// a few lookups beat and many do not.
const RANK_WALK_INVERSE_DENSITY: usize = 16;

/// [`accept_ords_from_docs`] for a filter passing many documents: the
/// field's documents as a bit set
/// ([`lucene_codecs::vectors::DocToOrdCursor::or_docs_into`]), each
/// accepted one's ordinal its rank among them -- the set bits before it --
/// a word at a time, where a lookup per document would seek the field's
/// `IndexedDISI` each time. The same set of ordinals.
fn accept_ords_by_rank(
    cursor: &lucene_codecs::vectors::DocToOrdCursor<'_>,
    filter: &FixedBitSet,
    live: Option<&FixedBitSet>,
    last_doc: i32,
    bits: &mut FixedBitSet,
) -> Result<()> {
    let span = usize::try_from(last_doc).map_or(0, |d| (d >> 6).saturating_add(1));
    let mut field = vec![0u64; span.max(filter.words().len())];
    cursor.or_docs_into(&mut field)?;
    let ords = bits.len();
    let mut rank = 0usize;
    for (i, &has) in field.iter().enumerate() {
        if has == 0 {
            continue;
        }
        let mut accepted = has & filter.words().get(i).copied().unwrap_or(0);
        if let Some(live) = live {
            accepted &= live.words().get(i).copied().unwrap_or(0);
        }
        while accepted != 0 {
            let bit = accepted.trailing_zeros();
            let below = has & ((1u64 << bit) - 1);
            let ord = rank + below.count_ones() as usize;
            if ord < ords {
                // FBS: bounded by the check above.
                bits.set(ord);
            }
            accepted &= accepted - 1;
        }
        rank += has.count_ones() as usize;
    }
    Ok(())
}

fn flush_bulk<S: VectorScorer>(
    collector: &mut KnnCollector,
    scorer: &mut S,
    ords: &[i32],
    scores: &mut [f32],
    num_ords: usize,
) -> Result<()> {
    collector.inc_visited_count(num_ords);
    if scorer.bulk_score(&ords[..num_ords], &mut scores[..num_ords])?
        > collector.min_competitive_similarity()
    {
        for j in 0..num_ords {
            collector.collect(ords[j], scores[j]);
        }
    }
    Ok(())
}

/// Port of `AbstractKnnVectorQuery.exactSearch`: score every accepted ordinal
/// and keep the best `min(k, cost)`.
///
/// Java's `HitQueue` is prefilled with `(Integer.MAX_VALUE, -Infinity)`
/// sentinels and drained with `while (queue.top().score < 0) pop()`. That
/// loop removes exactly the unfilled slots, because every
/// `VectorSimilarityFunction` maps into a non-negative range (the byte
/// `DOT_PRODUCT` transform bottoms out at exactly `0`), so a collector that
/// simply holds fewer than `k` hits is the same answer and no sentinel
/// machinery is reproduced. The tie-break is the same either way:
/// `HitQueue.lessThan` prefers the lower doc id on an equal score, and
/// [`KnnCollector`]'s `NeighborQueue` prefers the lower *ordinal* -- the same
/// order, since ordinals ascend with doc ids by construction.
fn exact_search<S: VectorScorer>(
    scorer: &mut S,
    accept_ords: &FixedBitSet,
    cost: usize,
    k: usize,
) -> Result<Vec<(i32, f32)>> {
    let queue_size = k.min(cost);
    if queue_size == 0 {
        return Ok(Vec::new());
    }
    let mut collector = KnnCollector::new(queue_size, u64::MAX);
    let mut ords = [0i32; EXHAUSTIVE_BULK_SCORE_ORDS];
    let mut scores = [0.0f32; EXHAUSTIVE_BULK_SCORE_ORDS];
    let mut num_ords = 0usize;
    // `scorer.max_ord()` and `accept_ords` reach this function from two
    // different places (the flat vector store's own count, and whatever
    // `accept_ords` the leaf plan built), so the ordinal loop is bounded
    // against the bitset that is actually indexed rather than against the
    // scorer -- an ordinal the accept set does not cover is, by definition,
    // not accepted. Hoisted out of the loop: one load, not one per ordinal.
    // Walked a set bit at a time (`next_set_bit` skips whole zero words):
    // this path runs when the filter is selective, so most ordinals are not
    // accepted and a per-ordinal probe would dominate the scoring.
    let end = scorer.max_ord().max(0) as usize;
    let span = scorer.read_ahead_span();
    let mut read_until = 0usize;
    let mut next = accept_ords.next_set_bit(0);
    while let Some(ord_idx) = next.filter(|&o| o < end) {
        next = accept_ords.next_set_bit(ord_idx + 1);
        if span > 0 && ord_idx >= read_until {
            read_until = read_ahead_dense(scorer, accept_ords, ord_idx, span, end);
        }
        let ord = ord_idx as i32;
        ords[num_ords] = ord;
        num_ords += 1;
        if num_ords == EXHAUSTIVE_BULK_SCORE_ORDS {
            flush_bulk(&mut collector, scorer, &ords, &mut scores, num_ords)?;
            num_ords = 0;
        }
    }
    if num_ords > 0 {
        flush_bulk(&mut collector, scorer, &ords, &mut scores, num_ords)?;
    }
    Ok(collector.top_docs())
}

/// The cost of one node a filtered graph walk visits, in vectors an exact
/// scan scores: a visit reads a neighbour list and a vector from a random
/// place and updates the walk's queues, the scan scores the passing vectors
/// in order (`docs/benchmarks/hnsw.md`, "Filtered searches").
const HNSW_VISIT_COST: f64 = 4.0;

/// A filter whose exact scan costs less than this many unfiltered walks is
/// scanned without probing the graph: even a walk that found only passing
/// nodes would save little.
const HNSW_PROBE_GAIN: f64 = 4.0;

/// The visits over which a probing walk measures the share of nodes near
/// the query that the filter passes.
const HNSW_PROBE_VISITS: u64 = 256;

/// How a filtered leaf passing more than `perLeafTopK` documents is searched.
#[derive(Debug, Clone, Copy, PartialEq)]
enum FilteredWalk {
    /// Score the passing vectors exactly.
    Scan,
    /// Lucene's walk, with its fallback to the scan after visiting as many
    /// nodes as pass.
    Walk,
    /// A walk that switches to the scan after [`HNSW_PROBE_VISITS`] when the
    /// filter passes too few of the nodes it met for the rest of the walk
    /// to cost less than the scan ([`FilteredWalkPlanner`]).
    Probe {
        unfiltered_visits: f64,
        share: f64,
        scan_visits: u64,
    },
}

/// The cost model choosing between [`FilteredWalk`]s for a filter passing
/// `passing` of a graph's `size` vectors, walked with a queue of
/// `collector_k`. An unfiltered walk visits about `ln(size) * collector_k`
/// nodes (Lucene's `expectedVisitedNodes`); one under a filter unrelated to
/// the query about that divided by the share of vectors passing. The scan
/// costs `passing / HNSW_VISIT_COST` visits. A walk predicted to cost no
/// more than the scan is Lucene's. Otherwise the filter may still be
/// correlated with the query's neighbourhood (a tenant queried from inside
/// its own region), which only the walk itself can see, so it probes --
/// unless the scan is cheap next to any walk.
///
/// Lucene walks every filter passing more than `perLeafTopK` documents and
/// falls back to the scan after visiting as many nodes as pass, which at a
/// few percent of a segment costs the walk and then the scan.
fn filtered_walk_plan(passing: usize, size: usize, collector_k: usize) -> FilteredWalk {
    let unfiltered_visits = (size.max(1) as f64).ln() * collector_k as f64;
    let share = passing as f64 / size.max(1) as f64;
    let scan = passing as f64 / HNSW_VISIT_COST;
    if unfiltered_visits <= scan * share {
        FilteredWalk::Walk
    } else if scan < HNSW_PROBE_GAIN * unfiltered_visits {
        FilteredWalk::Scan
    } else {
        FilteredWalk::Probe {
            unfiltered_visits,
            share,
            scan_visits: scan as u64,
        }
    }
}

/// The fewest accepted ordinals in a read-ahead window for
/// [`read_ahead_dense`] to read it ahead: a quarter of the ~256 pages it
/// spans, below which the scan touches too few of its pages for one large
/// read to beat a page per accepted vector.
const READ_AHEAD_MIN_ACCEPTED: usize = 64;

/// Reads the window of `span` ordinals from `from` ahead when enough of
/// them are accepted ([`READ_AHEAD_MIN_ACCEPTED`]), and returns where the
/// window ends. An exhaustive scan visits ordinals in order: over vectors
/// advised random access this keeps a dense scan (a contiguous tenant, a
/// broad filter) to large reads while a sparse one still reads only the
/// pages it scores.
fn read_ahead_dense<S: VectorScorer>(
    scorer: &S,
    accept_ords: &FixedBitSet,
    from: usize,
    span: usize,
    end: usize,
) -> usize {
    let to = from.saturating_add(span).min(end);
    if count_set_bits(accept_ords.words(), from, to) >= READ_AHEAD_MIN_ACCEPTED {
        let as_i32 = |o: usize| i32::try_from(o).unwrap_or(i32::MAX);
        scorer.read_ahead(as_i32(from), as_i32(to));
    }
    to
}

/// Set bits of `words` in `from..to`.
fn count_set_bits(words: &[u64], from: usize, to: usize) -> usize {
    if from >= to {
        return 0;
    }
    let (first, last) = (from / 64, (to - 1) / 64);
    let mut count = 0usize;
    for (i, &w) in words.iter().enumerate().take(last + 1).skip(first) {
        let mut w = w;
        if i == first {
            w &= u64::MAX << (from % 64);
        }
        if i == last {
            w &= u64::MAX >> (63 - (to - 1) % 64);
        }
        count += w.count_ones() as usize;
    }
    count
}

/// One leaf's collector sizing, the part of `getLeafResults` that does not
/// depend on the target's encoding.
#[derive(Debug, Clone, Copy)]
struct LeafPlan {
    /// `k` exactly as the query asked for it -- `exactSearch`'s queue size.
    k: usize,
    /// Java's `perLeafTopK`: `perLeafTopKCalculation(k, leafProportion)` for a
    /// multi-leaf search, `k` for a single one. **Not** widened by
    /// `ef_search`, deliberately: this is the number Java's two cost tests
    /// compare against (`cost <= perLeafTopK` and
    /// `scoreDocs.length >= perLeafTopK`), and widening it there would take
    /// the exact-search branch where Java walks the graph -- a different
    /// *kind* of answer from a knob documented as only ever buying recall.
    per_leaf_top_k: usize,
    /// The collector's size: [`Self::per_leaf_top_k`] widened to `ef_search`
    /// when the caller asked for a wider beam. Only the collector, never a
    /// threshold.
    collector_k: usize,
    visited_limit: u64,
    /// Java's `filterWeight != null`.
    filtered: bool,
    /// The collector decorators beyond `TopKnnCollector`.
    extras: LeafExtras,
}

/// What wraps a leaf's `TopKnnCollector`: `PatienceKnnVectorQuery`'s
/// `HnswQueueSaturationCollector` and `TimeLimitingKnnCollectorManager`'s
/// deadline.
#[derive(Debug, Clone, Copy, Default)]
struct LeafExtras {
    /// `(saturationThreshold, patience)`.
    patience: Option<(f64, usize)>,
    deadline: Option<std::time::Instant>,
    /// `KnnSearchStrategy.Hnsw.filteredSearchThreshold`.
    filtered_search_threshold: i32,
    /// [`KnnFloatVectorQuery::ivf_nprobes`].
    ivf_nprobes: usize,
    /// [`KnnFloatVectorQuery::exploration_slack`].
    exploration_slack: Option<(usize, f32)>,
    /// The field's similarity, set per leaf for
    /// [`Self::exploration_slack`].
    similarity: Option<VectorSimilarityFunction>,
    /// The live documents the filter passes over every leaf of the search,
    /// counted only for a search reaching an IVF field; `None` otherwise or
    /// when a leaf is searched alone (its own count stands in). It counts
    /// documents without a vector too, so it errs towards probing.
    search_passing: Option<usize>,
    /// [`KnnFloatVectorQuery::search_effort`]; `0.0` (the default) is `1.0`.
    search_effort: f32,
}

impl LeafExtras {
    fn timed_out(&self) -> bool {
        self.deadline
            .is_some_and(|d| std::time::Instant::now() >= d)
    }

    /// The search effort when it scales anything: positive, finite and not
    /// `1.0`.
    fn effort(&self) -> Option<f32> {
        let e = self.search_effort;
        (e.is_finite() && e > 0.0 && e != 1.0).then_some(e)
    }
}

/// `n` scaled by `effort`, rounded up, at least `min` (and at least 1).
fn scaled(n: usize, effort: f32, min: usize) -> usize {
    let v = (n as f64 * f64::from(effort)).ceil();
    // A float above `usize::MAX` saturates in the cast.
    (v as usize).max(min).max(1)
}

/// `searchNearestVectors` into the leaf's collector chain when it is
/// decorated: `PatienceKnnVectorQuery`'s `HnswQueueSaturationCollector`
/// and/or `TimeLimitingKnnCollectorManager`'s deadline around the
/// `TopKnnCollector`, handed to whichever reader the field has -- retired
/// ones included, since their `search` takes any collector. Returns the
/// hits, best first, and whether they are partial
/// (`TotalHits.Relation.GREATER_THAN_OR_EQUAL_TO`).
fn approximate<S: VectorScorer>(
    scorer: &mut S,
    graph: &LeafGraph<'_, '_>,
    k: usize,
    limit: u64,
    options: SearchOptions<'_>,
    extras: &LeafExtras,
) -> Result<(Vec<(i32, f32)>, bool)> {
    use crate::knn_collectors::{HnswQueueSaturationCollector, TimeLimitingKnnCollector};
    if scorer.max_ord() == 0 || k == 0 {
        return Ok((Vec::new(), false));
    }
    let top = KnnCollector::new(k, limit);
    let (top, partial) = match (extras.patience, extras.deadline) {
        (Some((threshold, patience)), deadline) => {
            let patient = HnswQueueSaturationCollector::new(top, threshold, patience);
            let patient = match deadline {
                Some(d) => {
                    let mut c = TimeLimitingKnnCollector::new(patient, d);
                    graph.search_with(scorer, &mut c, options)?;
                    let timed_out = c.timed_out();
                    let p = c.into_inner();
                    let partial = p.partial() || timed_out;
                    return Ok((p.into_inner().top_docs(), partial));
                }
                None => {
                    let mut c = patient;
                    graph.search_with(scorer, &mut c, options)?;
                    c
                }
            };
            let partial = patient.partial();
            (patient.into_inner(), partial)
        }
        (None, Some(d)) => {
            let mut c = TimeLimitingKnnCollector::new(top, d);
            graph.search_with(scorer, &mut c, options)?;
            let timed_out = c.timed_out();
            let inner = c.into_inner();
            let partial = KnnCollect::early_terminated(&inner) || timed_out;
            (inner, partial)
        }
        (None, None) => {
            let mut c = top;
            graph.search_with(scorer, &mut c, options)?;
            let early = c.early_terminated();
            (c, early)
        }
    };
    Ok((top.top_docs(), partial))
}

/// One leaf's phase-1 output: the hits a caller wants, plus the ordinals a
/// *seeded* second pass over the same leaf would start from.
///
/// The two are the same hits twice over, which is deliberate: Java rebuilds
/// the ordinals in `ReentrantKnnCollectorManager` by running phase 1's doc
/// ids back through `MappedDISI` -- an `advance` per seed over the whole
/// `IndexedDISI` -- because its `TopDocs` carry only doc ids. Keeping them
/// from the walk that already had them costs one `i32` per hit (at most
/// `perLeafTopK` of them) and no lookups at all.
#[derive(Debug, Default)]
struct LeafHits {
    /// Local-doc-space hits, best first.
    hits: Vec<ScoreDoc>,
    /// The same hits' ordinals, ascending -- `SeededHnswGraphSearcher`'s
    /// entry points. Empty when nothing was collected.
    ords: Vec<i32>,
}

/// Port of `AbstractKnnVectorQuery.getLeafResults` for one leaf whose scorer
/// is already built. Returns local-doc-space hits, best first, and whether
/// the search early-terminated.
#[allow(clippy::too_many_arguments)]
fn leaf_results<S: VectorScorer>(
    scorer: &mut S,
    graph: &LeafGraph<'_, '_>,
    accept: Option<&AcceptOrds<'_>>,
    ord_to_doc: &impl Fn(i32) -> Result<i32>,
    max_doc: i32,
    plan: &LeafPlan,
    seed_ords: Option<&[i32]>,
    diversify: Option<Diversify<'_>>,
) -> Result<(LeafHits, bool)> {
    match diversify {
        Some(Diversify::Parents(parents)) => {
            return diversified_leaf_results(
                scorer, graph, accept, ord_to_doc, max_doc, plan, seed_ords, parents,
            );
        }
        Some(Diversify::Groups(groups)) => {
            return grouped::grouped_leaf_results(
                scorer, graph, accept, ord_to_doc, max_doc, plan, groups,
            );
        }
        None => {}
    }
    leaf_results_rescored(
        scorer, None, false, graph, accept, ord_to_doc, max_doc, plan, seed_ords,
    )
}

/// An exhaustive search's hits, scored on `exact` when given (then they need
/// no rescoring: the flag is `false`), else on `scorer`.
fn exact_hits<S: VectorScorer>(
    scorer: &mut S,
    exact: Option<&mut (dyn VectorScorer + '_)>,
    bits: &FixedBitSet,
    cost: usize,
    k: usize,
) -> Result<(Vec<(i32, f32)>, bool)> {
    Ok(match exact {
        Some(exact) => (exact_search(&mut DynScorer(exact), bits, cost, k)?, false),
        None => (exact_search(scorer, bits, cost, k)?, true),
    })
}

/// A `dyn` scorer as a sized one, for the generic search functions.
struct DynScorer<'s, 'd>(&'s mut (dyn VectorScorer + 'd));

impl VectorScorer for DynScorer<'_, '_> {
    fn score(&mut self, node: i32) -> lucene_codecs::vectors::Result<f32> {
        self.0.score(node)
    }

    fn max_ord(&self) -> i32 {
        self.0.max_ord()
    }

    fn prefetch(&self, nodes: &[i32]) {
        self.0.prefetch(nodes)
    }

    fn read_ahead_span(&self) -> usize {
        self.0.read_ahead_span()
    }

    fn read_ahead(&self, from: i32, to: i32) {
        self.0.read_ahead(from, to)
    }

    fn bulk_score(
        &mut self,
        nodes: &[i32],
        scores: &mut [f32],
    ) -> lucene_codecs::vectors::Result<f32> {
        self.0.bulk_score(nodes, scores)
    }
}

/// [`leaf_results`] without diversification, with every collected hit
/// rescored by `rescore` (and re-ranked, best first) when one is given.
/// With `exact_on_rescore`, an exhaustive search (a selective filter, or a
/// graph walk that visited too many nodes) scores with `rescore` itself
/// rather than collecting `k` on `scorer`'s estimates.
#[allow(clippy::too_many_arguments)]
fn leaf_results_rescored<S: VectorScorer>(
    scorer: &mut S,
    mut rescore: Option<&mut dyn VectorScorer>,
    exact_on_rescore: bool,
    graph: &LeafGraph<'_, '_>,
    accept: Option<&AcceptOrds<'_>>,
    ord_to_doc: &impl Fn(i32) -> Result<i32>,
    max_doc: i32,
    plan: &LeafPlan,
    seed_ords: Option<&[i32]>,
) -> Result<(LeafHits, bool)> {
    let size = scorer.max_ord().max(0) as usize;
    // Clamped to the field's own vector count, which the reader validated
    // against the `.vec` file's length when it opened. That clamp is what
    // keeps a caller-supplied `k` of `usize::MAX` from reaching
    // `KnnCollector::new`'s heap allocation -- an allocation failure
    // *aborts*, which no `catch_unwind` can contain (see the `ffi-safety`
    // skill). It changes no result: a queue larger than the population can
    // never fill.
    let collector_k = plan.collector_k.min(size);
    let per_leaf_top_k = plan.per_leaf_top_k.min(size);
    let accept_bits = accept.map(|a| a.bits());

    // `from_graph`: whether the hits came out of the reader's `search` (and
    // so, for `Lucene90`, are ordinals standing in for documents) rather than
    // out of `exactSearch`, which always reports real documents.
    let mut needs_rescore = true;
    let (hits, early, from_graph) = if !plan.filtered {
        // Java's `filterWeight == null` branch: `AcceptDocs.fromLiveDocs`,
        // `visitedLimit = Integer.MAX_VALUE`, and no cost heuristic.
        // `filteredDocCount` is `min(maxDoc, graphSize)` here even on a
        // segment with deletions -- see [`SearchOptions::filtered_doc_count`]
        // for why that is not the bug it looks like.
        let hits = graph.search(
            scorer,
            collector_k,
            plan.visited_limit,
            SearchOptions {
                accept_ords: accept_bits,
                filtered_doc_count: Some(max_doc),
                seed_ords,
                filtered_search_threshold: plan.extras.filtered_search_threshold,
            },
            &plan.extras,
        )?;
        (hits.0, hits.1, true)
    } else {
        let bits = accept_bits.expect("a filtered leaf always has an accept set");
        let cost = bits.cardinality();
        // Lucene scans exactly below `perLeafTopK` passing documents; past
        // that the leaf's cost model decides ([`filtered_walk_plan`]). An
        // exact scan it chooses scores on the rescoring vectors when there
        // are any: the exact top `k`, not `k` codes' estimates rescored.
        let walk = if cost <= per_leaf_top_k {
            None
        } else if !matches!(graph, LeafGraph::Lucene99(Some(_))) {
            // Retired formats keep Lucene's rule: `Lucene90` reports a
            // walk's ordinals as documents, which the scan does not.
            Some(FilteredWalk::Walk)
        } else {
            match filtered_walk_plan(cost, size, collector_k) {
                FilteredWalk::Scan => None,
                walk => Some(walk),
            }
        };
        // Whether the cost model rather than Lucene's rule chose the scan.
        let mut planned_scan = cost > per_leaf_top_k && walk.is_none();
        let walked = match walk {
            None => None,
            Some(walk) => {
                // "We pass cost + 1 here to account for the edge case when
                // we explore exactly cost vectors."
                let limit = plan.visited_limit.min(cost as u64 + 1);
                let options = SearchOptions {
                    accept_ords: Some(bits),
                    filtered_doc_count: Some(cost as i32),
                    seed_ords,
                    filtered_search_threshold: plan.extras.filtered_search_threshold,
                };
                let (hits, early, gave_up) = match walk {
                    FilteredWalk::Probe {
                        unfiltered_visits,
                        share,
                        scan_visits,
                    } if plan.extras.patience.is_none()
                        && plan.extras.deadline.is_none()
                        && plan.extras.exploration_slack.is_none() =>
                    {
                        let mut c = FilteredWalkPlanner::new(
                            KnnCollector::new(collector_k, limit),
                            unfiltered_visits,
                            share,
                            scan_visits,
                            HNSW_PROBE_VISITS,
                        );
                        graph.search_with(scorer, &mut c, options)?;
                        let gave_up = c.gave_up();
                        let inner = c.into_inner();
                        let early = inner.early_terminated();
                        (inner.top_docs(), early, gave_up)
                    }
                    _ => {
                        let (hits, early) =
                            graph.search(scorer, collector_k, limit, options, &plan.extras)?;
                        (hits, early, false)
                    }
                };
                planned_scan = gave_up;
                if !gave_up && ((!early && hits.len() >= per_leaf_top_k) || plan.extras.timed_out())
                {
                    Some((hits, early))
                } else {
                    // "We stopped the kNN search because it visited too
                    // many nodes, so fall back to exact search."
                    None
                }
            }
        };
        match walked {
            Some((hits, early)) => (hits, early, true),
            None => {
                // Seeding does not reach here, exactly as in Java: the
                // search strategy is read by `HnswGraphSearcher.search`, and
                // this branch never calls it.
                let (hits, rescore) = exact_hits(
                    scorer,
                    rescore
                        .as_deref_mut()
                        .filter(|_| exact_on_rescore || planned_scan),
                    bits,
                    cost,
                    plan.k,
                )?;
                needs_rescore = rescore;
                (hits, false, false)
            }
        }
    };

    // Java's `OrdinalTranslatedKnnCollector`, plus this leaf's own seed set
    // for a possible second pass: `SeededKnnVectorQuery.TopDocsDISI` sorts
    // the hits' local doc ids ascending and `MappedDISI` turns each into its
    // ordinal, and ordinals ascend with doc ids by construction -- so the
    // seed list is this hit set's ordinals, ascending.
    let mut hits = hits;
    if let Some(rescore) = rescore.filter(|_| needs_rescore) {
        if hits.len() > 1 {
            let ords: Vec<i32> = hits.iter().map(|h| h.0).collect();
            rescore.prefetch(&ords);
        }
        for hit in &mut hits {
            hit.1 = rescore.score(hit.0)?;
        }
        // Ordinals ascend with doc ids: ties keep the lower doc first.
        hits.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    }
    let ordinals_as_docs = from_graph && graph.hits_are_ordinals();
    let mut out = Vec::with_capacity(hits.len());
    let mut ords = Vec::with_capacity(hits.len());
    for (ord, score) in hits {
        ords.push(ord);
        out.push(ScoreDoc {
            doc_id: if ordinals_as_docs {
                ord
            } else {
                ord_to_doc(ord)?
            },
            score,
        });
    }
    ords.sort_unstable();
    Ok((LeafHits { hits: out, ords }, early))
}

/// The encoding-specific half: open this field's values, build the scorer and
/// the accept set, then run [`leaf_results`].
fn search_leaf(
    input: &VectorsInput<'_>,
    resolved: &ResolvedField,
    target: Target<'_>,
    plan: &LeafPlan,
    seed_ords: Option<&[i32]>,
    diversify: Option<Diversify<'_>>,
) -> Result<(LeafHits, bool)> {
    let plan = &LeafPlan {
        extras: LeafExtras {
            similarity: Some(resolved.entry.similarity),
            ..plan.extras
        },
        ..*plan
    };
    let graph = leaf_graph(
        input,
        resolved.field_number,
        matches!(target, Target::Float(_)),
    )?;
    match target {
        Target::Float(t) => {
            if let Some(GraphReader::Ivf(reader)) = &input.hnsw {
                match diversify {
                    None => {
                        return match ivf_leaf_results(reader, input, resolved, t, plan, None)? {
                            Some(found) => Ok(found),
                            None => Ok((
                                exact_float_leaf(input, resolved, t, plan.collector_k.max(1))?,
                                false,
                            )),
                        };
                    }
                    // A leaf without parents has no results.
                    Some(Diversify::Parents(None)) => return Ok((LeafHits::default(), false)),
                    // Probing for the best child per parent, or the best
                    // documents per group; a filter too selective for
                    // probing takes the exhaustive walk below.
                    Some(Diversify::Parents(Some(parents))) => {
                        let group = |doc: i32| -> i32 {
                            usize::try_from(doc)
                                .ok()
                                .and_then(|d| parents.next_set_bit(d))
                                .map_or(i32::MAX, |p| i32::try_from(p).unwrap_or(i32::MAX))
                        };
                        // The parents that can be found: at most those still live.
                        let reachable = live_passing(Some(parents), input.live_docs, input.max_doc);
                        let groups = IvfGroups {
                            group: &group,
                            groups: parents.cardinality(),
                            reachable,
                            per_group: 1,
                        };
                        if let Some(found) =
                            ivf_leaf_results(reader, input, resolved, t, plan, Some(groups))?
                        {
                            return Ok(found);
                        }
                    }
                    // Groups scored whole take the exhaustive path.
                    Some(Diversify::Groups(SegmentGroups { fill: Some(_), .. })) => {}
                    Some(Diversify::Groups(g)) => {
                        let groups = IvfGroups {
                            group: g.group,
                            groups: g.groups,
                            reachable: g.groups,
                            per_group: g.per_group,
                        };
                        if let Some(found) =
                            ivf_leaf_results(reader, input, resolved, t, plan, Some(groups))?
                        {
                            return Ok(found);
                        }
                    }
                }
            }
            if let (Some(GraphReader::HnswDisk { disk, .. }), None) = (&input.hnsw, diversify) {
                if let Some(found) = disk_hnsw_leaf_results(disk, input, resolved, t, plan)? {
                    return Ok(found);
                }
            }
            let values = input.flat.float_vector_values(resolved.field_number)?;
            let ord_to_doc = |ord: i32| Ok(values.ord_to_doc(ord)?);
            let ord_to_docs = |out: &mut Vec<i32>| Ok(values.ord_to_docs(out)?);
            let doc_to_ord = || Ok(values.doc_to_ord()?);
            let maps = OrdDocMaps {
                ord_to_doc: &ord_to_doc,
                ord_to_docs: &ord_to_docs,
                doc_to_ord: &doc_to_ord,
            };
            let accept = accept_ords(input, resolved, &maps)?;
            if let Some(GraphReader::Quantized(reader)) = &input.hnsw {
                // `getRandomVectorScorer(field, target)` of the quantized
                // flat reader -- which is also `FloatVectorValues.scorer`, so
                // the exact-search fallback scores on the codes too.
                let mut scorer = reader.float_scorer(resolved.field_number, t)?;
                return leaf_results(
                    &mut scorer,
                    &graph,
                    accept.as_ref(),
                    &ord_to_doc,
                    input.max_doc,
                    plan,
                    seed_ords,
                    diversify,
                );
            }
            let mut scorer = values.scorer(t)?;
            if let (
                Some(GraphReader::HnswQuantized { codes, .. }),
                None,
                LeafGraph::Lucene99(Some(_)),
            ) = (&input.hnsw, diversify, &graph)
            {
                if let Some(field) = codes.field(resolved.field_number) {
                    field.check_matches(
                        values.dimension(),
                        values.similarity(),
                        usize::try_from(values.size()).unwrap_or(0),
                    )?;
                    // The graph is walked on the codes, the collected
                    // candidates are rescored on the float vectors, and an
                    // exhaustive search scores the float vectors directly
                    // (as does a segment too small for a graph, below).
                    let mut estimates = codes.scorer(field, t);
                    return leaf_results_rescored(
                        &mut estimates,
                        Some(&mut scorer),
                        true,
                        &graph,
                        accept.as_ref(),
                        &ord_to_doc,
                        input.max_doc,
                        plan,
                        seed_ords,
                    );
                }
            }
            if diversify.is_none() {
                if let Some(mut codes) = scalar_quantized_scorer(input, resolved, t)? {
                    // The graph is walked on the codes, then the collected
                    // candidates are rescored on the float vectors.
                    return leaf_results_rescored(
                        &mut codes,
                        Some(&mut scorer),
                        false,
                        &graph,
                        accept.as_ref(),
                        &ord_to_doc,
                        input.max_doc,
                        plan,
                        seed_ords,
                    );
                }
            }
            leaf_results(
                &mut scorer,
                &graph,
                accept.as_ref(),
                &ord_to_doc,
                input.max_doc,
                plan,
                seed_ords,
                diversify,
            )
        }
        Target::Byte(t) => {
            let values = input.flat.byte_vector_values(resolved.field_number)?;
            let ord_to_doc = |ord: i32| Ok(values.ord_to_doc(ord)?);
            let ord_to_docs = |out: &mut Vec<i32>| Ok(values.ord_to_docs(out)?);
            let doc_to_ord = || Ok(values.doc_to_ord()?);
            let maps = OrdDocMaps {
                ord_to_doc: &ord_to_doc,
                ord_to_docs: &ord_to_docs,
                doc_to_ord: &doc_to_ord,
            };
            let accept = accept_ords(input, resolved, &maps)?;
            let mut scorer = values.scorer(t)?;
            leaf_results(
                &mut scorer,
                &graph,
                accept.as_ref(),
                &ord_to_doc,
                input.max_doc,
                plan,
                seed_ords,
                diversify,
            )
        }
    }
}

/// The most a nested field's shortlist first widens for the children each
/// parent holds; a search finding fewer than `k` parents widens it further
/// (see [`ivf_leaf_results`]).
const IVF_MAX_GROUP_SPREAD: usize = 64;

/// The most live filtered documents a filtered IVF search counts one by one,
/// in a segment where some documents have none: past it, it samples this
/// many points over the segment for the share that has one.
const IVF_PASSING_SAMPLE: usize = 4096;

/// The cost of scoring one raw vector exactly, in IVF code scans (with their
/// share of routing, block reads and rerank): a filter goes to the exact scan
/// when its documents times this are at most the entries the probes would
/// scan. Measured in `docs/benchmarks/ivf.md` ("Filtered searches by tenant
/// size").
const IVF_EXACT_COST_RATIO: usize = 8;

/// One leaf's search through its IVF index: the `collector_k` best live
/// documents the filter passes, best first. `None` sends the leaf to the
/// exhaustive scan instead, when it costs less than probing: the filter
/// passes no more vectors than the probes would have to scan for it
/// ([`ivf_accepted_target`]), or scoring them exactly costs less than the
/// planned probes ([`IVF_EXACT_COST_RATIO`]). The flag is set when the
/// leaf's deadline passed before the probes reached `k` accepted documents.
/// A filter on a partition of a field clustered per partition is scored
/// exactly when it passes at most this many times the accepted target over
/// every leaf of the search: a partition's few clusters are probed nearly
/// all, with quantization error, where its documents' exact scores cost
/// little more. Counted over the search and not per leaf, so a partition
/// spread over many segments is probed as it would be in a few
/// (`docs/benchmarks/ivf.md`, "Clusters per partition").
const IVF_PARTITION_EXACT_RATIO: usize = 32;

/// Most runs of set bits a filter may have for a filtered IVF search to
/// prune clusters by document range: a partition's documents are one run
/// per segment of an index sorted by the partition field.
const IVF_FILTER_RUNS: usize = 256;

/// The clusters whose document range (`ranges`, lowest and highest
/// document) holds a document `filter` passes, or `None` when the filter
/// has more than [`IVF_FILTER_RUNS`] runs of documents.
/// The most documents a leaf's filter on a partition may pass to be scored
/// exactly: all of them while the search's documents of the partition
/// (`search_passing`, or the leaf's own `passing`) fit
/// [`IVF_PARTITION_EXACT_RATIO`], else the probes' accepted target `want`.
fn partition_exact_below(
    passing: usize,
    search_passing: Option<usize>,
    want: usize,
    k: usize,
) -> usize {
    let all = search_passing.unwrap_or(passing).max(passing);
    if all <= ivf_accepted_target(k).saturating_mul(IVF_PARTITION_EXACT_RATIO) {
        passing
    } else {
        want
    }
}

fn clusters_holding(ranges: &[(i32, i32)], filter: &FixedBitSet) -> Option<Vec<usize>> {
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut from = 0usize;
    while let Some(lo) = filter.next_set_bit(from) {
        if runs.len() == IVF_FILTER_RUNS {
            return None;
        }
        let hi = filter.next_clear_bit(lo).unwrap_or(filter.len());
        runs.push((lo, hi));
        from = hi;
        if from >= filter.len() {
            break;
        }
    }
    Some(
        ranges
            .iter()
            .enumerate()
            .filter(|(_, &(lo, hi))| {
                let (Ok(lo), Ok(hi)) = (usize::try_from(lo), usize::try_from(hi)) else {
                    return false;
                };
                // The first run ending after `lo` holds a document of the
                // range when it starts at or below `hi`.
                let i = runs.partition_point(|r| r.1 <= lo);
                runs.get(i).is_some_and(|r| r.0 <= hi)
            })
            .map(|(c, _)| c)
            .collect(),
    )
}

/// How an IVF leaf's documents fall into groups: a nested field's parents,
/// or a field's values.
#[derive(Clone, Copy)]
struct IvfGroups<'a> {
    /// A document's group.
    group: &'a dyn Fn(i32) -> i32,
    /// The groups in the segment (at least): the shortlist widens by the
    /// documents per group.
    groups: usize,
    /// The groups a search can find, at most: fewer than `k` found while
    /// more are in reach deepens the probes.
    reachable: usize,
    /// The documents kept of each group.
    per_group: usize,
}

fn ivf_leaf_results(
    reader: &lucene_ivf::IvfReader,
    input: &VectorsInput<'_>,
    resolved: &ResolvedField,
    target: &[f32],
    plan: &LeafPlan,
    grouping: Option<IvfGroups<'_>>,
) -> Result<Option<(LeafHits, bool)>> {
    let Some(field) = reader.field(resolved.field_number) else {
        return Ok(None);
    };
    let k = plan.collector_k.max(1);
    let clusters = field.clusters().max(1);
    // A nested field's children grouped by parent (the next parent bit at
    // or after the child, as `DiversifyingNearestChildrenKnnCollector`
    // finds it), or documents by a field's value; the shortlist widened by
    // the documents per group and the documents kept of each.
    let no_group = |_: i32| 0;
    let group: &dyn Fn(i32) -> i32 = grouping.map_or(&no_group, |g| g.group);
    let per_group = grouping.map_or(1, |g| g.per_group.max(1));
    let spread = std::cell::Cell::new(grouping.map_or(1, |g| {
        field
            .count()
            .div_ceil(g.groups.max(1))
            .clamp(1, IVF_MAX_GROUP_SPREAD)
            .saturating_mul(per_group)
    }));
    let reachable = grouping.map_or(0, |g| g.reachable);
    let grouped = grouping.is_some();
    // The groups among a search's hits.
    let groups_in = |hits: &[lucene_ivf::ScoredDoc]| -> usize {
        if per_group == 1 {
            return hits.len();
        }
        let mut seen: Vec<i32> = hits.iter().map(|h| group(h.doc)).collect();
        seen.sort_unstable();
        seen.dedup();
        seen.len()
    };
    // Once the shortlist holds every entry, a grouped search's candidates
    // are every probed child.
    let shortlist_full =
        || spread.get().saturating_mul(k.max(field.auto_rerank(k))) >= field.entries();
    let effort = plan.extras.effort();
    let base_nprobe = match plan.extras.ivf_nprobes {
        0 => field.auto_nprobe(),
        n => n,
    };
    let base_nprobe = effort
        .map_or(base_nprobe, |e| scaled(base_nprobe, e, 1))
        .clamp(1, clusters);
    // The calibrated rerank cap, scaled with the probes.
    let rerank = match effort {
        Some(e) => lucene_ivf::Rerank::Bounded {
            cap: scaled(field.auto_rerank(k), e, k),
        },
        None => lucene_ivf::Rerank::Auto,
    };
    let accept = |doc: i32| {
        input.live_docs.is_none_or(|l| l.get_doc(doc))
            && input.filter.is_none_or(|f| f.get_doc(doc))
    };
    let restricted = input.live_docs.is_some() || input.filter.is_some();
    let filtered: Option<&dyn Fn(i32) -> bool> = restricted.then_some(&accept);
    // A search of every cluster, or (`among`) of those clusters only.
    let search = |nprobe: lucene_ivf::Probes, among: Option<&[usize]>| {
        let params = lucene_ivf::SearchParams { k, nprobe, rerank };
        let field = resolved.field_number;
        if grouped {
            let g = lucene_ivf::Grouping {
                group,
                spread: spread.get(),
                per_group,
            };
            reader.search_grouped(field, target, &params, filtered, among, &g)
        } else {
            match among {
                Some(cs) => reader.search_within(field, target, &params, filtered, cs, None),
                None => reader.search_with(field, target, &params, filtered, None),
            }
        }
    };
    // Grouped, `k` counts parents: while fewer than `k` (of those live) are
    // found, the probes deepen and the shortlist widens, so many siblings
    // scoring alike cannot crowd the other parents out; both grow until
    // every cluster's every child is a candidate, an exhaustive search.
    let short = |found: usize| grouped && found < k.min(reachable);
    // Whether a short round can still find more.
    // Whether a short round over `limit` clusters can still find more.
    let spent = |nprobe: usize, limit: usize, found: usize| {
        nprobe >= limit && (!short(found) || shortlist_full())
    };
    let widen = |found: usize| {
        if short(found) {
            spread.set(spread.get().saturating_mul(2));
        }
    };
    let (found, partial) = match input.filter {
        // Without a filter, deletions only: the calibrated (or asked) depth,
        // doubled while fewer than `k` live documents are in reach.
        None => {
            let live = input
                .live_docs
                .map_or(field.count(), |l| l.cardinality().min(field.count()));
            let mut nprobe = base_nprobe;
            loop {
                let probes =
                    if plan.extras.ivf_nprobes == 0 && effort.is_none() && nprobe == base_nprobe {
                        lucene_ivf::Probes::Auto
                    } else {
                        lucene_ivf::Probes::Fixed(nprobe)
                    };
                let out = search(probes, None)?;
                let found = groups_in(&out.hits);
                let enough = if grouped {
                    !short(found)
                } else {
                    !restricted || out.hits.len() >= k.min(live)
                };
                if enough || spent(nprobe, clusters, found) {
                    break (out.hits, false);
                }
                if plan.extras.timed_out() {
                    break (out.hits, true);
                }
                widen(found);
                nprobe = nprobe.saturating_mul(2).min(clusters);
            }
        }
        Some(filter) => {
            // The live documents the filter passes that have a vector here.
            let filter_live = live_passing(Some(filter), input.live_docs, input.max_doc);
            let passing = if field.count() >= input.max_doc as usize {
                filter_live
            } else {
                // Some documents have no vector (a nested field's parents, a
                // sparse field): a filter passing at most
                // [`IVF_PASSING_SAMPLE`] live documents is counted exactly;
                // a larger one by the share of its documents with a vector
                // at that many points spread evenly over the segment. Walking
                // all of a filter passing half a large segment costs more
                // than probing for it.
                let values = input.flat.float_vector_values(resolved.field_number)?;
                let mut ords = values.doc_to_ord()?;
                let max_doc = usize::try_from(input.max_doc).unwrap_or(0);
                let exact = filter_live <= IVF_PASSING_SAMPLE;
                let step = if exact {
                    1
                } else {
                    max_doc.div_ceil(IVF_PASSING_SAMPLE).max(1)
                };
                let (mut n, mut seen) = (0usize, 0usize);
                let mut at = 0usize;
                while at < max_doc {
                    let Some(bit) = filter.next_set_bit(at) else {
                        break;
                    };
                    // The next point: each set bit is counted once.
                    at = if exact {
                        bit.saturating_add(1)
                    } else {
                        bit.saturating_add(1).max(at.saturating_add(step))
                    };
                    // A set bit is below `max_doc`, an `i32`.
                    let doc = i32::try_from(bit).unwrap_or(i32::MAX);
                    if input.live_docs.is_none_or(|l| l.get_doc(doc)) {
                        seen = seen.saturating_add(1);
                        if ords.ordinal(doc)?.is_some() {
                            n = n.saturating_add(1);
                        }
                    }
                }
                if exact {
                    n
                } else {
                    ((filter_live as f64) * n as f64 / seen.max(1) as f64) as usize
                }
            };
            // A filter passing no more documents than the accepted target
            // is scored exactly: probing would scan at least as many.
            if passing <= ivf_accepted_target(k) {
                return Ok(None);
            }
            // A field clustered per partition (an index sorted by a
            // partition field) probes only the clusters whose document range
            // holds a document the filter passes, as an index of those
            // documents alone would.
            let among = field
                .doc_ranges()
                .and_then(|ranges| clusters_holding(ranges, filter));
            // The clusters searched and their entries, as the probes count
            // them: a spilled vector is in two clusters.
            let (space_clusters, space_entries) = match &among {
                Some(cs) => (
                    cs.len(),
                    cs.iter().map(|&c| field.cluster_size(c)).sum::<usize>(),
                ),
                None => (clusters, field.entries()),
            };
            let space_clusters = space_clusters.max(1);
            let space_entries = space_entries.max(1);
            let per_cluster = space_entries.div_ceil(space_clusters).max(1);
            // The calibrated (or asked) depth, as a share of the clusters
            // searched.
            let base = if among.is_some() {
                let fraction = base_nprobe as f64 / clusters as f64;
                ((fraction * space_clusters as f64).ceil() as usize).clamp(1, space_clusters)
            } else {
                base_nprobe
            };
            let entries_per_vector = field.entries().max(1) as f64 / field.count().max(1) as f64;
            let share = (passing as f64 * entries_per_vector / space_entries as f64).min(1.0);
            // Across the whole field, never more than the filter's share of
            // what the unfiltered search scans would hold on average, with
            // slack for clusters below the average size. A partition's own
            // clusters hold mostly its documents, so it needs no such cap.
            let cap = match &among {
                Some(_) => usize::MAX,
                None => ((base.saturating_mul(per_cluster) as f64 * 0.75) as usize).max(k),
            };
            let want = ivf_accepted_target(k)
                .min(cap)
                .min((passing as f64 * entries_per_vector) as usize);
            // The first try probes as many clusters as the filter's share of
            // the searched clusters needs on average, at least the
            // calibrated (or asked) depth; the probes double while the
            // accepted entries they held fall short, as they do where the
            // filter's documents lie far from the query.
            let need = (want as f64 / (per_cluster as f64 * share.max(f64::MIN_POSITIVE))).ceil();
            let mut nprobe = base.max(need.min(space_clusters as f64) as usize);
            let planned = nprobe.saturating_mul(per_cluster);
            let exact_below = match &among {
                Some(_) => partition_exact_below(passing, plan.extras.search_passing, want, k),
                None => want,
            };
            if passing <= exact_below || passing.saturating_mul(IVF_EXACT_COST_RATIO) <= planned {
                return Ok(None);
            }
            loop {
                let out = search(lucene_ivf::Probes::Fixed(nprobe), among.as_deref())?;
                let found = groups_in(&out.hits);
                if (out.stats.accepted >= want && !short(found))
                    || spent(nprobe, space_clusters, found)
                {
                    break (out.hits, false);
                }
                if plan.extras.timed_out() {
                    break (out.hits, found < k);
                }
                widen(found);
                nprobe = nprobe.saturating_mul(2).min(space_clusters);
            }
        }
    };
    let mut hits: Vec<ScoreDoc> = found
        .into_iter()
        .map(|h| ScoreDoc {
            doc_id: h.doc,
            score: h.score,
        })
        .collect();
    // Rerank vectors stored narrower than `f32` score approximately: the
    // hits are scored again on the raw vectors, as every other leaf is. A
    // nested search's best child per parent was chosen on those scores, an
    // approximation as the probes are.
    if field.encoding() != lucene_ivf::VectorEncoding::F32 {
        let values = input.flat.float_vector_values(resolved.field_number)?;
        let mut ords = values.doc_to_ord()?;
        let mut scratch = vec![0f32; target.len()];
        hits.sort_unstable_by_key(|h| h.doc_id);
        for h in &mut hits {
            if let Some(ord) = ords.ordinal(h.doc_id)? {
                h.score = resolved
                    .entry
                    .similarity
                    .score(target, values.vector_ref(ord, &mut scratch)?);
            }
        }
        hits.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.doc_id.cmp(&b.doc_id)));
    }
    Ok(Some((
        LeafHits {
            hits,
            ords: Vec::new(),
        },
        partial,
    )))
}

/// Nodes a disk HNSW search fetches together per round trip (DiskANN's `W`).
const DISK_HNSW_BEAM_WIDTH: usize = 8;

/// The most a restricted walk's candidate list grows over `collector_k`.
const DISK_HNSW_MAX_LIST_GROWTH: usize = 8;

/// A float search of a field's disk HNSW layout: the `collector_k` best
/// live documents the filter passes, scored exactly from the node records
/// the walk expanded. `None` leaves the leaf to the graph on the raw
/// vectors: a deadline, patience or exploration slack to honor, a filter
/// cheaper to scan exactly, or a walk that found too few documents.
fn disk_hnsw_leaf_results(
    disk: &lucene_disk_hnsw::DiskHnswReader,
    input: &VectorsInput<'_>,
    resolved: &ResolvedField,
    target: &[f32],
    plan: &LeafPlan,
) -> Result<Option<(LeafHits, bool)>> {
    let Some(field) = disk.field(resolved.field_number) else {
        return Ok(None);
    };
    let extras = &plan.extras;
    if extras.patience.is_some() || extras.deadline.is_some() || extras.exploration_slack.is_some()
    {
        return Ok(None);
    }
    let size = field.count();
    let collector_k = plan.collector_k.min(size).max(1);
    // The documents the filter passes that are live; a filter passing no
    // more than `perLeafTopK` is scored exactly, as Lucene does, and so is
    // one the cost model prices below a walk.
    let mut wanted_passing = size;
    let wanted = match input.filter {
        None => plan.per_leaf_top_k.min(size),
        Some(filter) => {
            let cost = live_passing(Some(filter), input.live_docs, input.max_doc);
            wanted_passing = cost.min(size);
            if cost <= plan.per_leaf_top_k
                || matches!(
                    filtered_walk_plan(cost, size, collector_k),
                    FilteredWalk::Scan
                )
            {
                return Ok(None);
            }
            plan.per_leaf_top_k.min(cost)
        }
    };
    let accept = |doc: i32| {
        input.live_docs.is_none_or(|l| l.get_doc(doc))
            && input.filter.is_none_or(|f| f.get_doc(doc))
    };
    let restricted = input.live_docs.is_some() || input.filter.is_some();
    // Rejected documents are walked but fill no slot of the candidate list,
    // so a list sized for `collector_k` finds fewer accepted ones as the
    // share of accepted documents falls.
    let passing = match input.filter {
        Some(_) => wanted_passing,
        None => input.live_docs.map_or(size, |l| l.cardinality().min(size)),
    };
    let share = passing.max(1) as f64 / size.max(1) as f64;
    let list_len = ((collector_k as f64 / share).ceil() as usize).clamp(
        collector_k,
        collector_k.saturating_mul(DISK_HNSW_MAX_LIST_GROWTH),
    );
    // `ef_search` widens the walk's candidate list, as it widens Lucene's
    // beam; the leaf returns the `perLeafTopK` best of it.
    let params = lucene_disk_hnsw::SearchParams {
        k: plan.per_leaf_top_k.min(size).max(1),
        list_len,
        beam_width: DISK_HNSW_BEAM_WIDTH,
    };
    let out = disk.search(
        resolved.field_number,
        target,
        &params,
        restricted.then_some(&accept as &dyn Fn(i32) -> bool),
    )?;
    if out.hits.len() < wanted {
        return Ok(None);
    }
    let mut hits: Vec<ScoreDoc> = out
        .hits
        .into_iter()
        .map(|h| ScoreDoc {
            doc_id: h.doc,
            score: h.score,
        })
        .collect();
    // The layout stores a cosine field's vectors normalized, so its scores
    // differ from the raw vectors' in the last bits: the hits are scored
    // again on the raw vectors, as every other leaf scores them.
    if resolved.entry.similarity == VectorSimilarityFunction::Cosine {
        let values = input.flat.float_vector_values(resolved.field_number)?;
        let mut ords = values.doc_to_ord()?;
        let mut scratch = vec![0f32; target.len()];
        hits.sort_unstable_by_key(|h| h.doc_id);
        for h in &mut hits {
            if let Some(ord) = ords.ordinal(h.doc_id)? {
                h.score = resolved
                    .entry
                    .similarity
                    .score(target, values.vector_ref(ord, &mut scratch)?);
            }
        }
        hits.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.doc_id.cmp(&b.doc_id)));
    }
    Ok(Some((
        LeafHits {
            hits,
            ords: Vec::new(),
        },
        false,
    )))
}

/// The exact `k` best live documents the filter passes, best first (the
/// lower doc on a tie), scored on the leaf's raw vectors.
fn exact_float_leaf(
    input: &VectorsInput<'_>,
    resolved: &ResolvedField,
    target: &[f32],
    k: usize,
) -> Result<LeafHits> {
    let values = input.flat.float_vector_values(resolved.field_number)?;
    let similarity = resolved.entry.similarity;
    let mut scratch = vec![0f32; target.len()];
    let mut top: std::collections::BinaryHeap<std::cmp::Reverse<(OrdScore, i32)>> =
        std::collections::BinaryHeap::with_capacity(k.saturating_add(1));
    let mut offer = |doc: i32, score: f32| {
        // Ties keep the lower doc: it ranks above a higher one.
        top.push(std::cmp::Reverse((OrdScore(score), -doc)));
        if top.len() > k {
            top.pop();
        }
    };
    let live = |doc: i32| input.live_docs.is_none_or(|l| l.get_doc(doc));
    match input.filter {
        // A filter visits only the documents it passes, in doc order, so
        // the cost follows the filter and not the field: a small tenant's
        // range of a large segment reads only that range's vectors.
        Some(filter) => {
            let mut ords = values.doc_to_ord()?;
            let mut from = 0usize;
            while let Some(bit) = filter.next_set_bit(from) {
                from = bit.saturating_add(1);
                // A set bit is below `max_doc`, an `i32`.
                let doc = i32::try_from(bit).unwrap_or(i32::MAX);
                if !live(doc) {
                    continue;
                }
                if let Some(ord) = ords.ordinal(doc)? {
                    offer(
                        doc,
                        similarity.score(target, values.vector_ref(ord, &mut scratch)?),
                    );
                }
            }
        }
        None => {
            for ord in 0..values.size() {
                let doc = values.ord_to_doc(ord)?;
                if live(doc) {
                    offer(
                        doc,
                        similarity.score(target, values.vector_ref(ord, &mut scratch)?),
                    );
                }
            }
        }
    }
    let mut hits: Vec<ScoreDoc> = top
        .into_iter()
        .map(|std::cmp::Reverse((OrdScore(score), neg))| ScoreDoc {
            doc_id: -neg,
            score,
        })
        .collect();
    hits.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.doc_id.cmp(&b.doc_id)));
    Ok(LeafHits {
        hits,
        ords: Vec::new(),
    })
}

/// A score ordered by `total_cmp`, for a heap.
#[derive(Clone, Copy, PartialEq)]
struct OrdScore(f32);

impl Eq for OrdScore {}

impl PartialOrd for OrdScore {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdScore {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// Runs a `KnnFloatVectorQuery` against one already-opened segment, exactly
/// as `IndexSearcher.search(KnnFloatVectorQuery, k)` does over a single-leaf
/// reader (`leafProportion == 1`, so `perLeafTopK == k` and the re-entrant
/// second pass cannot trigger).
///
/// Hits come back best-first in this segment's **local** doc-id space; see
/// [`search_knn_float_vector_query_multi_segment`] for the global one.
pub fn search_knn_float_vector_query(
    input: &VectorsInput<'_>,
    query: &KnnFloatVectorQuery,
) -> Result<Vec<ScoreDoc>> {
    search_one_segment(input, query)
}

/// `KnnByteVectorQuery`'s equivalent of [`search_knn_float_vector_query`].
pub fn search_knn_byte_vector_query(
    input: &VectorsInput<'_>,
    query: &KnnByteVectorQuery,
) -> Result<Vec<ScoreDoc>> {
    search_one_segment(input, query)
}

fn search_one_segment<Q: KnnQuery>(input: &VectorsInput<'_>, query: &Q) -> Result<Vec<ScoreDoc>> {
    check_k(query.k())?;
    let resolved = resolve_field(input, query)?;
    let plan = LeafPlan {
        k: query.k(),
        per_leaf_top_k: query.k(),
        collector_k: query.k().max(query.ef_search()),
        visited_limit: visit_limit(query),
        filtered: input.filter.is_some(),
        extras: LeafExtras {
            filtered_search_threshold: query.filtered_search_threshold(),
            ivf_nprobes: query.ivf_nprobes(),
            exploration_slack: query.exploration_slack(),
            search_effort: query.search_effort(),
            ..LeafExtras::default()
        },
    };
    let (mut leaf, _) = search_leaf(input, &resolved, query.target(), &plan, None, None)?;
    leaf.hits.truncate(query.k());
    Ok(leaf.hits)
}

fn visit_limit<Q: KnnQuery>(query: &Q) -> u64 {
    if query.visited_limit() == 0 {
        u64::MAX
    } else {
        query.visited_limit()
    }
}

/// One leaf of a multi-segment KNN search: this segment's vector inputs plus
/// its `doc_base`, the KNN sibling of
/// [`crate::multi_segment::OpenSegment`].
pub struct KnnSegment<'d> {
    pub vectors: VectorsInput<'d>,
    /// This segment's starting global doc id (`SegmentReader.docBase`) -- the
    /// same caller-computed value [`crate::multi_segment::OpenSegment`] takes,
    /// with the same warning: a wrong value here silently produces wrong
    /// global doc ids and this module cannot detect it.
    pub doc_base: i32,
}

/// `IndexSearcher.search(KnnFloatVectorQuery, k)` over a multi-segment index:
/// `AbstractKnnVectorQuery.rewrite`'s per-leaf fan-out, pro-rata collector
/// sizing, optimistic re-entry pass, and `TopDocs.merge`.
///
/// Three things that are easy to get wrong, spelled out:
///
/// 1. **Per-leaf `k` is pro-rata, not `k`.** `TopKnnCollectorManager` is
///    optimistic (`isOptimistic() == true`), so each leaf is searched with a
///    collector of [`per_leaf_top_k`]`(k, leafMaxDoc/indexMaxDoc)`. For a
///    handful of similar-sized segments that is *larger* than `k` (the
///    `LAMBDA = 16` term dominates); for one segment it is exactly `k`.
/// 2. **A second, re-entrant pass** runs over every leaf whose worst phase-1
///    hit is still at or above the merged top-`k`'s worst -- those leaves are
///    not "tapped out" and are searched again with a full-`k` collector.
/// 3. **The merge is `TopDocs.merge(k, ..)`**, which this port already has as
///    [`merge_multi_segment_scored`] (`doc_base` translation plus one more
///    `TopDocsCollector`, i.e. `HitQueue`'s score-desc/doc-asc order).
///    Nothing about it is re-implemented here.
///
/// The second pass is **seeded**, as Java's is:
/// `ReentrantKnnCollectorManager` wraps phase 1's hits for that leaf in a
/// `KnnSearchStrategy.Seeded`, which `HnswGraphSearcher.search` honours by
/// delegating to `SeededHnswGraphSearcher` -- so level 0's beam restarts from
/// the nodes phase 1 already reached rather than descending from the graph's
/// entry node again. See
/// [`lucene_codecs::hnsw::HnswGraphSearcher::search_seeded`]; it changes only
/// *where* the walk starts, never the collector size, the accept set or the
/// merge, and it is what makes the second pass cost a fraction of the first.
pub fn search_knn_float_vector_query_multi_segment(
    segments: &[KnnSegment<'_>],
    query: &KnnFloatVectorQuery,
) -> Result<Vec<ScoreDoc>> {
    knn_multi_segment(segments, query, false)
}

/// `KnnByteVectorQuery`'s equivalent of
/// [`search_knn_float_vector_query_multi_segment`].
pub fn search_knn_byte_vector_query_multi_segment(
    segments: &[KnnSegment<'_>],
    query: &KnnByteVectorQuery,
) -> Result<Vec<ScoreDoc>> {
    knn_multi_segment(segments, query, false)
}

/// The concurrent sibling of
/// [`search_knn_float_vector_query_multi_segment`]: each leaf's search runs
/// on rayon's pool, as real Lucene runs each leaf on its `TaskExecutor`. The
/// merge stays sequential and in segment order, so the two functions' results
/// are provably identical rather than merely usually equal.
///
/// **Measured, and the measurement says do not reach for this by default.**
/// On the four-leaf, 4000-document fixture (`benches/knn_multi_segment.rs`) a
/// `k = 10` query costs **38 us sequentially and 198 us concurrently**, and
/// at `k = 100` -- ten times the work per leaf -- **128 us against 232 us**.
/// The gap narrows with the work per leaf, as a fixed dispatch cost should,
/// but it has not closed even at `k = 100` over 4000 vectors: a leaf search
/// here is tens of microseconds and rayon's per-task cost is comparable.
/// This entry point earns its keep on leaves large enough for one search to
/// dominate that -- a real OpenSearch shard's millions of documents, not a
/// fixture's thousands.
pub fn search_knn_float_vector_query_multi_segment_concurrent(
    segments: &[KnnSegment<'_>],
    query: &KnnFloatVectorQuery,
) -> Result<Vec<ScoreDoc>> {
    knn_multi_segment(segments, query, true)
}

/// `KnnByteVectorQuery`'s equivalent of
/// [`search_knn_float_vector_query_multi_segment_concurrent`].
pub fn search_knn_byte_vector_query_multi_segment_concurrent(
    segments: &[KnnSegment<'_>],
    query: &KnnByteVectorQuery,
) -> Result<Vec<ScoreDoc>> {
    knn_multi_segment(segments, query, true)
}

/// Per-leaf field resolution and collector sizing -- the part of the fan-out
/// that is identical sequential or concurrent.
fn plan_leaves<Q: KnnQuery>(
    segments: &[KnnSegment<'_>],
    query: &Q,
    extras: LeafExtras,
) -> Result<(Vec<ResolvedField>, Vec<LeafPlan>)> {
    check_k(query.k())?;
    // `ctx.parent.reader().maxDoc()`: the whole index's document count.
    let index_max_doc: i64 = segments.iter().map(|s| s.vectors.max_doc as i64).sum();
    let mut resolved = Vec::with_capacity(segments.len());
    let mut plans = Vec::with_capacity(segments.len());
    let search_passing = segments
        .iter()
        .any(|s| s.vectors.filter.is_some() && matches!(s.vectors.hnsw, Some(GraphReader::Ivf(_))))
        .then(|| {
            segments
                .iter()
                .map(|s| live_passing(s.vectors.filter, s.vectors.live_docs, s.vectors.max_doc))
                .fold(0usize, usize::saturating_add)
        });
    for seg in segments {
        resolved.push(resolve_field(&seg.vectors, query)?);
        // Java's `ctx.reader().maxDoc() / (float) ctx.parent.reader().maxDoc()`.
        // A zero-document index makes this NaN, which `per_leaf_top_k`'s
        // `max(1.0, ..)` turns into 1 -- Java's documented `assert
        // perLeafTopK > 0`.
        let proportion = seg.vectors.max_doc as f32 / index_max_doc as f32;
        let leaf_top_k = per_leaf_top_k(query.k(), proportion);
        plans.push(LeafPlan {
            k: query.k(),
            per_leaf_top_k: leaf_top_k,
            collector_k: leaf_top_k.max(query.ef_search()),
            visited_limit: visit_limit(query),
            filtered: seg.vectors.filter.is_some(),
            extras: LeafExtras {
                filtered_search_threshold: query.filtered_search_threshold(),
                ivf_nprobes: query.ivf_nprobes(),
                exploration_slack: query.exploration_slack(),
                search_passing,
                search_effort: query.search_effort(),
                ..extras
            },
        });
    }
    Ok((resolved, plans))
}

/// The live documents of one leaf its filter passes, every live document
/// when it has none.
fn live_passing(
    filter: Option<&FixedBitSet>,
    live_docs: Option<&FixedBitSet>,
    max_doc: i32,
) -> usize {
    match (filter, live_docs) {
        (Some(f), Some(l)) => FixedBitSet::intersection_count(f, l),
        (Some(f), None) => f.cardinality(),
        (None, Some(l)) => l.cardinality(),
        (None, None) => usize::try_from(max_doc).unwrap_or(0),
    }
}

fn merge_leaves(
    segments: &[KnnSegment<'_>],
    per_leaf: &[Vec<ScoreDoc>],
    k: usize,
) -> Result<Vec<ScoreDoc>> {
    let doc_bases: Vec<i32> = segments.iter().map(|s| s.doc_base).collect();
    merge_multi_segment_scored(&doc_bases, k, |i, local| {
        for hit in &per_leaf[i] {
            local.collect(hit.doc_id, hit.score);
        }
        Ok(())
    })
}

/// The re-entry decision of `AbstractKnnVectorQuery.rewrite`: which leaves
/// are still worth exploring once phase 1's merged top-`k` is known. A leaf
/// qualifies when its own worst collected hit is at or above the merged
/// top-`k`'s worst, i.e. "all this leaf's hits are at or above the global
/// topK min score; explore it further".
fn reentry_leaves(per_leaf: &[Vec<ScoreDoc>], merged: &[ScoreDoc]) -> Vec<usize> {
    let Some(worst) = merged.last() else {
        return Vec::new();
    };
    let min_top_k_score = worst.score;
    (0..per_leaf.len())
        .filter(|&i| {
            per_leaf[i]
                .last()
                .is_some_and(|h| h.score >= min_top_k_score)
        })
        .collect()
}

/// Phase 2's collector: `getKnnCollectorManager(k, searcher)` **without** the
/// optimistic wrapper, i.e. the full `k` (widened by `ef_search` like every
/// other collector here).
///
/// **Only the collector changes.** `perLeafTopK` -- the number
/// `getLeafResults` compares `cost` and `scoreDocs.length` against -- is
/// recomputed there from `ctx.parent` on *every* call, so it is the pro-rata
/// value in phase 2 exactly as in phase 1; the full `k` reaches phase 2
/// through the collector manager (`ReentrantKnnCollectorManager` delegates to
/// a fresh `TopKnnCollectorManager(k, searcher)`), which `getLeafResults`
/// never inspects. Raising the threshold with the collector would take Java's
/// `cost <= perLeafTopK` exact-search branch where Java walks the graph, and
/// take its `scoreDocs.length >= perLeafTopK` fall-back where Java keeps the
/// approximate result -- a different answer on any filtered re-entered leaf.
fn reentry_plan(phase1: &LeafPlan, ef_search: usize) -> LeafPlan {
    LeafPlan {
        collector_k: phase1.k.max(ef_search),
        ..*phase1
    }
}

fn knn_multi_segment<Q: KnnQuery + Sync>(
    segments: &[KnnSegment<'_>],
    query: &Q,
    concurrent: bool,
) -> Result<Vec<ScoreDoc>> {
    knn_multi_segment_with(
        segments,
        query,
        concurrent,
        LeafExtras::default(),
        None,
        None,
    )
}

/// [`knn_multi_segment`] with its collectors decorated (`extras`) and phase 1
/// seeded per leaf (`SeededKnnVectorQuery`'s seed hits, as ordinals; an
/// empty leaf entry is unseeded).
fn knn_multi_segment_with<Q: KnnQuery + Sync>(
    segments: &[KnnSegment<'_>],
    query: &Q,
    concurrent: bool,
    extras: LeafExtras,
    phase1_seeds: Option<&[Vec<i32>]>,
    diversify: Option<&[Option<&FixedBitSet>]>,
) -> Result<Vec<ScoreDoc>> {
    let k = query.k();
    let (resolved, plans) = plan_leaves(segments, query, extras)?;

    let phase1 = run_leaves(
        segments,
        &resolved,
        query,
        &(0..segments.len()).collect::<Vec<_>>(),
        &plans,
        phase1_seeds,
        concurrent,
        diversify,
    )?;
    let mut early = false;
    let mut per_leaf: Vec<Vec<ScoreDoc>> = Vec::with_capacity(segments.len());
    let mut per_leaf_ords: Vec<Vec<i32>> = Vec::with_capacity(segments.len());
    for (leaf, e) in phase1 {
        early |= e;
        per_leaf.push(leaf.hits);
        per_leaf_ords.push(leaf.ords);
    }
    let mut merged = merge_leaves(segments, &per_leaf, k)?;

    // "only re-enter if we used the optimistic collection" (always, here --
    // `TopKnnCollectorManager.isOptimistic()`), there is more than one leaf,
    // something was collected, and nothing early-terminated.
    if segments.len() > 1 && !merged.is_empty() && !early {
        let reenter = reentry_leaves(&per_leaf, &merged);
        if !reenter.is_empty() {
            let plans2: Vec<LeafPlan> = plans
                .iter()
                .map(|p| reentry_plan(p, query.ef_search()))
                .collect();
            // `ReentrantKnnCollectorManager`: phase 2 is *seeded* with phase
            // 1's own hits for that leaf, so the walk resumes where it left
            // off instead of descending from the graph's entry node again.
            let phase2 = run_leaves(
                segments,
                &resolved,
                query,
                &reenter,
                &plans2,
                Some(&per_leaf_ords),
                concurrent,
                diversify,
            )?;
            for (&i, (leaf, _)) in reenter.iter().zip(phase2) {
                per_leaf[i] = leaf.hits;
            }
            merged = merge_leaves(segments, &per_leaf, k)?;
        }
    }
    // `IndexSearcher.search(rewritten, n)` collects the rewritten
    // `DocAndScoreQuery` with the leaves' live docs as `acceptDocs`. Every
    // hit a reader translated through `ordToDoc` is live already (the accept
    // set says so); a `Lucene90` hit is an *ordinal* standing in for a
    // document, which can name a deleted one -- and Java drops it here,
    // after the top `k` was cut, so the answer comes back short.
    merged.retain(|h| hit_is_live(segments, h.doc_id));
    Ok(merged)
}

/// Whether global doc `doc` is live in the leaf that holds it.
fn hit_is_live(segments: &[KnnSegment<'_>], doc: i32) -> bool {
    segments
        .iter()
        .rev()
        .find(|s| s.doc_base <= doc)
        .is_none_or(|s| {
            s.vectors
                .live_docs
                .is_none_or(|live| live.get_doc(doc - s.doc_base))
        })
}

/// Phase 2's entry points for leaf `i`, or `None` for "not seeded".
///
/// Java's `ReentrantKnnCollectorManager` falls back to the **unseeded**
/// collector when a leaf's phase-1 `TopDocs` is empty ("shouldn't happen - we
/// only come here when there are results", and its `assert false` says so),
/// and `HnswGraphSearcher.search` ignores a `KnnSearchStrategy.Seeded` whose
/// `numberOfEntryPoints()` is zero. An empty seed list is therefore not an
/// empty entry-point set to be passed down -- which
/// [`lucene_codecs::hnsw::HnswGraphSearcher::search_seeded`] rejects outright,
/// as Java's `fromEntryPoints` does -- but "not seeded at all".
fn seed_slice(seeds: Option<&[Vec<i32>]>, i: usize) -> Option<&[i32]> {
    seeds
        .map(|s| s[i].as_slice())
        .filter(|ords| !ords.is_empty())
}

/// Runs `search_leaf` for each named leaf, sequentially or on rayon's pool.
///
/// The fan-out is *not* expressed as
/// [`crate::multi_segment::merge_multi_segment_scored_concurrent`] even
/// though the shape matches, for one concrete reason: that function's
/// per-segment closure writes into a `TopDocsCollector::new(top_n)`, which
/// truncates each leaf's contribution to `top_n == k` -- and the re-entry
/// decision above needs each leaf's **untruncated** `perLeafTopK` list (Java
/// compares `perLeaf.scoreDocs[len-1].score`, the `perLeafTopK`-th score, not
/// the `k`-th). The merge, which is the part with the doc-base translation
/// and the `HitQueue` ordering in it, *is* that module's
/// [`merge_multi_segment_scored`] -- see [`merge_leaves`].
#[allow(clippy::too_many_arguments)]
fn run_leaves<Q: KnnQuery + Sync>(
    segments: &[KnnSegment<'_>],
    resolved: &[ResolvedField],
    query: &Q,
    leaves: &[usize],
    plans: &[LeafPlan],
    seeds: Option<&[Vec<i32>]>,
    concurrent: bool,
    diversify: Option<&[Option<&FixedBitSet>]>,
) -> Result<Vec<(LeafHits, bool)>> {
    let seed_for = |i: usize| seed_slice(seeds, i);
    let parents_for = |i: usize| diversify.map(|d| Diversify::Parents(d.get(i).copied().flatten()));
    if concurrent {
        use rayon::prelude::*;
        leaves
            .par_iter()
            .map(|&i| {
                search_leaf(
                    &segments[i].vectors,
                    &resolved[i],
                    query.target(),
                    &plans[i],
                    seed_for(i),
                    parents_for(i),
                )
            })
            .collect()
    } else {
        leaves
            .iter()
            .map(|&i| {
                search_leaf(
                    &segments[i].vectors,
                    &resolved[i],
                    query.target(),
                    &plans[i],
                    seed_for(i),
                    parents_for(i),
                )
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// KNN as a query clause, patience, seeds, timeouts, similarity thresholds
// ---------------------------------------------------------------------------

/// `DocAndScoreQuery.createDocAndScoreQuery`: KNN hits (global doc ids, as
/// the multi-segment searches return them) as a clause that can sit in a
/// boolean like any other -- `AbstractKnnVectorQuery.rewrite`'s result. No
/// hit is `MatchNoDocsQuery`.
pub fn knn_hits_to_clause(segments: &[KnnSegment<'_>], hits: &[ScoreDoc]) -> crate::query::Clause {
    if hits.is_empty() {
        return crate::query::Clause::MatchNoDocs(crate::query::MatchNoDocsQuery::new());
    }
    let doc_bases: Vec<i32> = segments.iter().map(|s| s.doc_base).collect();
    crate::extended_query::DocAndScoreQuery::new(
        hits.iter().map(|h| (h.doc_id, h.score)).collect(),
        &doc_bases,
    )
    .into()
}

/// `IndexSearcher.setTimeout`: the KNN search of `query` with each leaf's
/// collector wrapped by `TimeLimitingKnnCollectorManager` -- past
/// `deadline`, walks stop and what they found stands.
pub fn search_knn_float_vector_query_multi_segment_with_deadline(
    segments: &[KnnSegment<'_>],
    query: &KnnFloatVectorQuery,
    deadline: std::time::Instant,
) -> Result<Vec<ScoreDoc>> {
    let extras = LeafExtras {
        deadline: Some(deadline),
        ..LeafExtras::default()
    };
    knn_multi_segment_with(segments, query, false, extras, None, None)
}

/// `KnnByteVectorQuery`'s equivalent of
/// [`search_knn_float_vector_query_multi_segment_with_deadline`].
pub fn search_knn_byte_vector_query_multi_segment_with_deadline(
    segments: &[KnnSegment<'_>],
    query: &KnnByteVectorQuery,
    deadline: std::time::Instant,
) -> Result<Vec<ScoreDoc>> {
    let extras = LeafExtras {
        deadline: Some(deadline),
        ..LeafExtras::default()
    };
    knn_multi_segment_with(segments, query, false, extras, None, None)
}

/// `PatienceKnnVectorQuery`: a KNN query whose graph walks stop once the
/// result queue saturates (`HnswQueueSaturationCollector`).
#[derive(Debug, Clone, PartialEq)]
pub struct PatienceKnnVectorQuery<Q> {
    pub query: Q,
    pub saturation_threshold: f64,
    pub patience: usize,
}

impl<Q> PatienceKnnVectorQuery<Q> {
    /// `DEFAULT_SATURATION_THRESHOLD`.
    pub const DEFAULT_SATURATION_THRESHOLD: f64 = 0.995;

    pub fn new(query: Q, saturation_threshold: f64, patience: usize) -> Self {
        Self {
            query,
            saturation_threshold,
            patience,
        }
    }

    /// `defaultPatience`: `max(7, (int) (k * 0.3))`.
    fn default_patience(k: usize) -> usize {
        7usize.max((k as f64 * 0.3) as usize)
    }
}

impl PatienceKnnVectorQuery<KnnFloatVectorQuery> {
    /// `fromFloatQuery(knnQuery)`.
    pub fn from_float_query(query: KnnFloatVectorQuery) -> Self {
        let patience = Self::default_patience(query.k);
        Self::new(query, Self::DEFAULT_SATURATION_THRESHOLD, patience)
    }
}

impl PatienceKnnVectorQuery<KnnByteVectorQuery> {
    /// `fromByteQuery(knnQuery)`.
    pub fn from_byte_query(query: KnnByteVectorQuery) -> Self {
        let patience = Self::default_patience(query.k);
        Self::new(query, Self::DEFAULT_SATURATION_THRESHOLD, patience)
    }
}

/// `IndexSearcher.search(PatienceKnnVectorQuery, k)` over a multi-segment
/// index: [`search_knn_float_vector_query_multi_segment`] with every leaf's
/// collector (both passes) a `HnswQueueSaturationCollector`.
pub fn search_patience_knn_float_vector_query_multi_segment(
    segments: &[KnnSegment<'_>],
    query: &PatienceKnnVectorQuery<KnnFloatVectorQuery>,
) -> Result<Vec<ScoreDoc>> {
    let extras = LeafExtras {
        patience: Some((query.saturation_threshold, query.patience)),
        ..LeafExtras::default()
    };
    knn_multi_segment_with(segments, &query.query, false, extras, None, None)
}

/// The byte-vector twin of
/// [`search_patience_knn_float_vector_query_multi_segment`].
pub fn search_patience_knn_byte_vector_query_multi_segment(
    segments: &[KnnSegment<'_>],
    query: &PatienceKnnVectorQuery<KnnByteVectorQuery>,
) -> Result<Vec<ScoreDoc>> {
    let extras = LeafExtras {
        patience: Some((query.saturation_threshold, query.patience)),
        ..LeafExtras::default()
    };
    knn_multi_segment_with(segments, &query.query, false, extras, None, None)
}

/// `SeededKnnVectorQuery`'s first pass: each leaf's walk starts from the
/// vectors of `seed_docs[i]` (that leaf's local doc ids, the seed query's
/// top `k` there -- [`knn_seed_docs`]) instead of the graph's entry node. A
/// leaf with no seed walks as usual; the second pass is the plain query's.
pub fn search_seeded_knn_float_vector_query_multi_segment(
    segments: &[KnnSegment<'_>],
    query: &KnnFloatVectorQuery,
    seed_docs: &[Vec<i32>],
) -> Result<Vec<ScoreDoc>> {
    let seeds = seed_ords(segments, query, seed_docs)?;
    knn_multi_segment_with(
        segments,
        query,
        false,
        LeafExtras::default(),
        Some(&seeds),
        None,
    )
}

/// The byte-vector twin of
/// [`search_seeded_knn_float_vector_query_multi_segment`].
pub fn search_seeded_knn_byte_vector_query_multi_segment(
    segments: &[KnnSegment<'_>],
    query: &KnnByteVectorQuery,
    seed_docs: &[Vec<i32>],
) -> Result<Vec<ScoreDoc>> {
    let seeds = seed_ords(segments, query, seed_docs)?;
    knn_multi_segment_with(
        segments,
        query,
        false,
        LeafExtras::default(),
        Some(&seeds),
        None,
    )
}

/// `SeededKnnVectorQuery.MappedDISI` over `TopDocsDISI`: each seed doc
/// (sorted) to the ordinal of the first vector at or after it.
fn seed_ords<Q: KnnQuery>(
    segments: &[KnnSegment<'_>],
    query: &Q,
    seed_docs: &[Vec<i32>],
) -> Result<Vec<Vec<i32>>> {
    let mut out = Vec::with_capacity(segments.len());
    for (i, seg) in segments.iter().enumerate() {
        let mut docs = seed_docs.get(i).cloned().unwrap_or_default();
        docs.sort_unstable();
        if docs.is_empty() {
            out.push(Vec::new());
            continue;
        }
        let resolved = resolve_field(&seg.vectors, query)?;
        // Every ordinal's document, decoded once: the vector values are
        // opened once per leaf, not per probe of the search below.
        let mut ord_docs = Vec::new();
        match Q::ENCODING {
            VectorEncoding::Float32 => seg
                .vectors
                .flat
                .float_vector_values(resolved.field_number)?
                .ord_to_docs(&mut ord_docs)?,
            VectorEncoding::Byte => seg
                .vectors
                .flat
                .byte_vector_values(resolved.field_number)?
                .ord_to_docs(&mut ord_docs)?,
        }
        let mut ords = Vec::with_capacity(docs.len());
        for doc in docs {
            // The first ordinal whose document is at or after `doc`
            // (`advance(doc)` on the vector iterator, then `index()`).
            let ord = ord_docs.partition_point(|&d| d < doc);
            if ord < ord_docs.len() {
                // An ordinal indexes an `i32`-sized field.
                ords.push(ord as i32);
            }
        }
        out.push(ords);
    }
    Ok(out)
}

/// `filterWeight.scorer(ctx)` for every leaf: the documents `filter`
/// matches in each segment (deletions not applied, as a scorer's iterator
/// does not apply them), as the bitset [`VectorsInput::filter`] takes.
pub fn filter_bitsets(
    segments: &[crate::multi_segment::OpenSegment<'_>],
    filter: &crate::query::Clause,
) -> Result<Vec<FixedBitSet>> {
    // `filterWeight` is created over the whole searcher: a function query
    // in the filter reads the reader's statistics.
    let as_query = crate::query::BooleanQuery::new().with_filter([filter.clone()]);
    let global = crate::multi_segment::global_function_stats(segments, &as_query, None)?;
    segments
        .iter()
        .map(|seg| crate::exec::extended::segment_match_bits(seg, filter, global.as_ref()))
        .collect()
}

/// `SeededKnnVectorQuery.SeededCollectorManager`'s seed search: per leaf,
/// the top `k` live documents of `seed` that have a vector for `field` (the
/// `FieldExistsQuery` filter) and pass `filter`, scored with reader-wide
/// statistics -- `TopScoreDocCollector` over `seedWeight`. `open` and
/// `knn` are the same segments in the same order.
pub fn knn_seed_docs(
    open: &[crate::multi_segment::OpenSegment<'_>],
    norms: &[Option<&std::collections::HashMap<String, crate::FieldNorms<'_>>>],
    knn: &[KnnSegment<'_>],
    field: &str,
    seed: &crate::query::Clause,
    filter: Option<&[FixedBitSet]>,
    k: usize,
) -> Result<Vec<Vec<i32>>> {
    struct All(Vec<(i32, f32)>);
    impl ScoringCollector for All {
        fn collect(&mut self, doc_id: i32, score: f32) {
            self.0.push((doc_id, score));
        }
    }
    let q = crate::query::BooleanQuery::new().with_must([seed.clone()]);
    let global = crate::multi_segment::global_boolean_stats(open, &q)?;
    let mut out = Vec::with_capacity(open.len());
    for (i, seg) in open.iter().enumerate() {
        let mut all = All(Vec::new());
        crate::search_boolean_query_scored_segment(
            seg,
            &q,
            norms.get(i).copied().flatten(),
            Some(&global),
            &mut all,
        )?;
        let mut hits: Vec<(i32, f32)> = all
            .0
            .into_iter()
            .filter(|&(d, _)| {
                seg.live_docs.is_none_or(|l| l.get_doc(d))
                    // FBS: `sets` is the slice of per-segment filter sets,
                    // not a bit set; `get(i)` is the slice's checked `get`.
                    && filter
                        .and_then(|sets| sets.get(i))
                        .is_none_or(|b| b.get_doc(d))
            })
            .collect();
        if let Some(s) = knn.get(i) {
            // The `FieldExistsQuery` clause, as Java's conjunction runs it:
            // each hit looked up in the field's doc -> ord iterator, forward
            // in doc order.
            hits.sort_unstable_by_key(|&(d, _)| d);
            retain_vector_docs(&s.vectors, field, &mut hits)?;
        }
        hits.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        hits.truncate(k);
        out.push(hits.into_iter().map(|(d, _)| d).collect());
    }
    Ok(out)
}

/// Keeps the `hits` (ascending doc ids) that have a vector for `field` in one
/// segment.
fn retain_vector_docs(
    input: &VectorsInput<'_>,
    field: &str,
    hits: &mut Vec<(i32, f32)>,
) -> Result<()> {
    let entry = input
        .field_infos
        .field_by_name(field)
        .and_then(|info| Some((info.number, input.flat.field(info.number)?)));
    let Some((number, entry)) = entry else {
        hits.clear();
        return Ok(());
    };
    match entry.encoding {
        VectorEncoding::Float32 => {
            let values = input.flat.float_vector_values(number)?;
            retain_with_ordinal(values.doc_to_ord()?, hits)
        }
        VectorEncoding::Byte => {
            let values = input.flat.byte_vector_values(number)?;
            retain_with_ordinal(values.doc_to_ord()?, hits)
        }
    }
}

/// [`retain_vector_docs`]' filter: one forward pass of the cursor.
fn retain_with_ordinal(
    mut cursor: lucene_codecs::vectors::DocToOrdCursor<'_>,
    hits: &mut Vec<(i32, f32)>,
) -> Result<()> {
    let mut kept = 0;
    for i in 0..hits.len() {
        if cursor.ordinal(hits[i].0)?.is_some() {
            hits.swap(kept, i);
            // ARITH: `kept <= i < hits.len()`.
            #[allow(clippy::arithmetic_side_effects)]
            {
                kept += 1;
            }
        }
    }
    hits.truncate(kept);
    Ok(())
}

/// `FloatVectorSimilarityQuery`/`ByteVectorSimilarityQuery`
/// (`AbstractVectorSimilarityQuery`): every vector whose similarity to the
/// target is at least `result_similarity`, found by a graph walk whose
/// traversal bound decays by `decay` (`VectorSimilarityCollector`), or
/// exhaustively at `decay == 1` or when a filtered walk runs out of visits.
///
/// Lucene 10.5.2 (#16654) split the Java class into `.Adaptive` (10.5's
/// decaying walk, [`Self::new`]) and the deprecated `.Explicit` (10.4's fixed
/// traversal bound, [`Self::explicit`], `ExplicitVectorSimilarityCollector`);
/// here one struct carries both, `traversal_similarity` telling them apart.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorSimilarityQuery<T> {
    pub field: String,
    pub target: Vec<T>,
    pub result_similarity: f32,
    pub decay: f32,
    /// `Explicit.traversalSimilarity`; `None` for `Adaptive`.
    pub traversal_similarity: Option<f32>,
}

/// `FloatVectorSimilarityQuery`.
pub type FloatVectorSimilarityQuery = VectorSimilarityQuery<f32>;
/// `ByteVectorSimilarityQuery`.
pub type ByteVectorSimilarityQuery = VectorSimilarityQuery<u8>;

impl<T> VectorSimilarityQuery<T> {
    /// `AbstractVectorSimilarityQuery.DEFAULT_DECAY`.
    pub const DEFAULT_DECAY: f32 = 0.5;

    /// `Explicit(field, target, traversalSimilarity, resultSimilarity)`
    /// (deprecated in Java): Lucene 10.4's query, collecting with
    /// `ExplicitVectorSimilarityCollector`; decay `0`. Java's
    /// `Explicit(field, target, resultSimilarity)` is
    /// `explicit(field, target, r, r)`.
    pub fn explicit(
        field: impl Into<String>,
        target: Vec<T>,
        traversal_similarity: f32,
        result_similarity: f32,
    ) -> Result<Self> {
        let mut q = Self::new(field, target, result_similarity, 0.0)?;
        if traversal_similarity > result_similarity {
            return Err(Error::InvalidKnnQuery(
                "traversalSimilarity should be <= resultSimilarity".to_string(),
            ));
        }
        q.traversal_similarity = Some(traversal_similarity);
        Ok(q)
    }

    /// `Adaptive(field, target, resultSimilarity, decay)`, the private
    /// constructor's checks: a similarity that is a number, a decay in
    /// `[0, 1]`.
    pub fn new(
        field: impl Into<String>,
        target: Vec<T>,
        result_similarity: f32,
        decay: f32,
    ) -> Result<Self> {
        if result_similarity.is_nan() {
            return Err(Error::InvalidKnnQuery(format!(
                "resultSimilarity must have a valid value; got {result_similarity}"
            )));
        }
        if decay.is_nan() || !(0.0..=1.0).contains(&decay) {
            return Err(Error::InvalidKnnQuery(format!(
                "decay must lie in range [DECAY_MAX_APPROXIMATION = 0, DECAY_MAX_QUALITY = 1]; got {decay}"
            )));
        }
        Ok(Self {
            field: field.into(),
            target,
            result_similarity,
            decay,
            traversal_similarity: None,
        })
    }
}

/// `AbstractVectorSimilarityQuery`'s scorer over every leaf, as the clause
/// it scores like: each matching vector's document with its similarity. A
/// leaf's filter is its [`VectorsInput::filter`].
pub fn float_vector_similarity_clause(
    segments: &[KnnSegment<'_>],
    query: &FloatVectorSimilarityQuery,
) -> Result<crate::query::Clause> {
    vector_similarity_clause(
        segments,
        query,
        Target::Float(&query.target),
        VectorEncoding::Float32,
    )
}

/// The byte-vector twin of [`float_vector_similarity_clause`].
pub fn byte_vector_similarity_clause(
    segments: &[KnnSegment<'_>],
    query: &ByteVectorSimilarityQuery,
) -> Result<crate::query::Clause> {
    vector_similarity_clause(
        segments,
        query,
        Target::Byte(&query.target),
        VectorEncoding::Byte,
    )
}

/// Every document [`float_vector_similarity_clause`] matches, with its
/// score: global doc ids, leaf by leaf in segment order, each leaf's in the
/// order its scorer supplier yields them (`fromScoreDocs` or
/// `fromAcceptDocs`), which a caller sorts as it needs.
pub fn float_vector_similarity_hits(
    segments: &[KnnSegment<'_>],
    query: &FloatVectorSimilarityQuery,
) -> Result<Vec<ScoreDoc>> {
    vector_similarity_hits(
        segments,
        query,
        Target::Float(&query.target),
        VectorEncoding::Float32,
    )
}

/// The byte-vector twin of [`float_vector_similarity_hits`].
pub fn byte_vector_similarity_hits(
    segments: &[KnnSegment<'_>],
    query: &ByteVectorSimilarityQuery,
) -> Result<Vec<ScoreDoc>> {
    vector_similarity_hits(
        segments,
        query,
        Target::Byte(&query.target),
        VectorEncoding::Byte,
    )
}

fn vector_similarity_hits<T>(
    segments: &[KnnSegment<'_>],
    query: &VectorSimilarityQuery<T>,
    target: Target<'_>,
    encoding: VectorEncoding,
) -> Result<Vec<ScoreDoc>> {
    let mut hits = Vec::new();
    for seg in segments {
        for (doc, score) in similarity_leaf(&seg.vectors, query, target, encoding)? {
            // The scorer is iterated under the leaf's live docs; only a
            // `Lucene90` ordinal-as-document can name a deleted one.
            if seg.vectors.live_docs.is_some_and(|live| !live.get_doc(doc)) {
                continue;
            }
            hits.push(ScoreDoc {
                doc_id: doc + seg.doc_base,
                score,
            });
        }
    }
    Ok(hits)
}

fn vector_similarity_clause<T>(
    segments: &[KnnSegment<'_>],
    query: &VectorSimilarityQuery<T>,
    target: Target<'_>,
    encoding: VectorEncoding,
) -> Result<crate::query::Clause> {
    let hits = vector_similarity_hits(segments, query, target, encoding)?;
    Ok(knn_hits_to_clause(segments, &hits))
}

/// One leaf of [`vector_similarity_clause`]: `(local doc, similarity)`.
fn similarity_leaf<T>(
    input: &VectorsInput<'_>,
    query: &VectorSimilarityQuery<T>,
    target: Target<'_>,
    encoding: VectorEncoding,
) -> Result<Vec<(i32, f32)>> {
    let Some(info) = input.field_infos.field_by_name(&query.field) else {
        return Ok(Vec::new());
    };
    let Some(entry) = input.flat.field(info.number) else {
        return Ok(Vec::new());
    };
    if entry.encoding != encoding {
        return Err(Error::InvalidKnnQuery(format!(
            "field {:?} is {:?}-encoded, but this query searches {encoding:?} vectors",
            query.field, entry.encoding
        )));
    }
    let resolved = ResolvedField {
        field_number: info.number,
        entry: entry.clone(),
    };
    // `acceptDocs.cost()`: in document space, the filter's live documents
    // (with or without a vector) -- or every document without a filter.
    let cardinality: usize = match input.filter {
        Some(f) => (0..input.max_doc)
            .filter(|&d| f.get_doc(d) && input.live_docs.is_none_or(|l| l.get_doc(d)))
            .count(),
        None => input.max_doc.max(0) as usize,
    };
    if input.filter.is_some() && cardinality == 0 {
        return Ok(Vec::new());
    }
    let graph = leaf_graph(
        input,
        resolved.field_number,
        matches!(target, Target::Float(_)),
    )?;
    let leaf = SimilarityLeaf {
        input,
        resolved: &resolved,
        result_similarity: query.result_similarity,
        decay: query.decay,
        traversal_similarity: query.traversal_similarity,
        cardinality,
    };
    match target {
        Target::Float(t) => {
            let values = input.flat.float_vector_values(resolved.field_number)?;
            let ord_to_doc = |ord: i32| -> Result<i32> { Ok(values.ord_to_doc(ord)?) };
            let ord_to_docs = |out: &mut Vec<i32>| -> Result<()> { Ok(values.ord_to_docs(out)?) };
            let doc_to_ord = || Ok(values.doc_to_ord()?);
            let ord_to_docs = OrdDocMaps {
                ord_to_doc: &ord_to_doc,
                ord_to_docs: &ord_to_docs,
                doc_to_ord: &doc_to_ord,
            };
            // `createVectorScorer`: `FloatVectorValues.scorer(target)`, which a
            // quantized format answers on its codes.
            if let Some(GraphReader::Quantized(reader)) = &input.hnsw {
                let mut scorer = reader.float_scorer(resolved.field_number, t)?;
                return leaf.run(&mut scorer, &graph, &ord_to_doc, &ord_to_docs);
            }
            let mut scorer = values.scorer(t)?;
            leaf.run(&mut scorer, &graph, &ord_to_doc, &ord_to_docs)
        }
        Target::Byte(t) => {
            let values = input.flat.byte_vector_values(resolved.field_number)?;
            let ord_to_doc = |ord: i32| -> Result<i32> { Ok(values.ord_to_doc(ord)?) };
            let ord_to_docs = |out: &mut Vec<i32>| -> Result<()> { Ok(values.ord_to_docs(out)?) };
            let doc_to_ord = || Ok(values.doc_to_ord()?);
            let ord_to_docs = OrdDocMaps {
                ord_to_doc: &ord_to_doc,
                ord_to_docs: &ord_to_docs,
                doc_to_ord: &doc_to_ord,
            };
            let mut scorer = values.scorer(t)?;
            leaf.run(&mut scorer, &graph, &ord_to_doc, &ord_to_docs)
        }
    }
}

/// `AbstractVectorSimilarityQuery.scorerSupplier` for one leaf, once its
/// scorer and graph are open.
struct SimilarityLeaf<'q, 'd> {
    input: &'q VectorsInput<'d>,
    resolved: &'q ResolvedField,
    result_similarity: f32,
    decay: f32,
    /// `Explicit.traversalSimilarity`: collect with
    /// `ExplicitVectorSimilarityCollector` (its `getKnnCollectorManager`).
    traversal_similarity: Option<f32>,
    /// `acceptDocs.cost()`.
    cardinality: usize,
}

impl SimilarityLeaf<'_, '_> {
    fn run<S: VectorScorer>(
        &self,
        scorer: &mut S,
        graph: &LeafGraph<'_, '_>,
        ord_to_doc: &impl Fn(i32) -> Result<i32>,
        ord_to_docs: &OrdDocMaps<'_, '_>,
    ) -> Result<Vec<(i32, f32)>> {
        use crate::knn_collectors::{
            ExplicitVectorSimilarityCollector, VectorSimilarityCollector, DECAY_MAX_QUALITY,
        };
        let accept = accept_ords(self.input, self.resolved, ord_to_docs)?;
        let accept_bits = accept.as_ref().map(|a| a.bits());
        let filtered = self.input.filter.is_some();
        // Whether the hits are the reader's `search` output (and so, for
        // `Lucene90`, ordinals standing in for documents) rather than
        // `fromAcceptDocs`' real documents.
        let mut from_graph = false;
        let hits = if self.decay == DECAY_MAX_QUALITY {
            self.exact(scorer, accept_bits)?
        } else {
            // `approximateSearch(context, acceptDocs, visitLimit, ...)`:
            // unlimited without a filter, the filter's cardinality with one.
            let limit = if filtered {
                self.cardinality as u64
            } else {
                i32::MAX as u64
            };
            let options = SearchOptions {
                accept_ords: accept_bits,
                filtered_doc_count: Some(i32::try_from(self.cardinality).unwrap_or(i32::MAX)),
                seed_ords: None,
                // `AbstractVectorSimilarityQuery.DEFAULT_STRATEGY`: `Hnsw(0)`.
                filtered_search_threshold: 0,
            };
            let (hits, early) = match self.traversal_similarity {
                Some(traversal) => {
                    let mut collector = ExplicitVectorSimilarityCollector::new(
                        traversal,
                        self.result_similarity,
                        limit,
                    )?;
                    graph.search_with(scorer, &mut collector, options)?;
                    collector.into_hits()
                }
                None => {
                    let mut collector =
                        VectorSimilarityCollector::new(self.result_similarity, self.decay, limit);
                    graph.search_with(scorer, &mut collector, options)?;
                    collector.into_hits()
                }
            };
            if filtered && early {
                // The walk ran out of visits: `fromAcceptDocs`, exhaustive.
                self.exact(scorer, accept_bits)?
            } else {
                from_graph = true;
                hits
            }
        };
        let ordinals_as_docs = from_graph && graph.hits_are_ordinals();
        let mut out = Vec::with_capacity(hits.len());
        for (ord, score) in hits {
            out.push((
                if ordinals_as_docs {
                    ord
                } else {
                    ord_to_doc(ord)?
                },
                score,
            ));
        }
        Ok(out)
    }

    /// `VectorSimilarityScorerSupplier.fromAcceptDocs`: every accepted vector
    /// at or above the threshold.
    fn exact<S: VectorScorer>(
        &self,
        scorer: &mut S,
        accept: Option<&FixedBitSet>,
    ) -> Result<Vec<(i32, f32)>> {
        let mut out = Vec::new();
        for ord in 0..scorer.max_ord() {
            if accept.is_some_and(|a| !a.get_doc(ord)) {
                continue;
            }
            let score = scorer.score(ord)?;
            if score >= self.result_similarity {
                out.push((ord, score));
            }
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// lucene-join's diversifying child KNN queries
// ---------------------------------------------------------------------------

/// `DiversifyingNearestChildrenKnnCollector.ParentChildScore`: a child, its
/// parent and its similarity -- and the child's ordinal, which a seeded second
/// pass starts from (Java maps the doc back through `MappedDISI`).
#[derive(Debug, Clone, Copy)]
struct ParentChildScore {
    child: i32,
    ord: i32,
    parent: i32,
    score: f32,
}

impl ParentChildScore {
    /// `compareTo`: by score, then the lower child first (it compares
    /// greater: "lower ids are preferred").
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        java_float_compare(self.score, o.score).then_with(|| o.child.cmp(&self.child))
    }
}

/// `DiversifyingNearestChildrenKnnCollector.NodeIdCachingHeap`: a min-heap
/// of the best child per parent, 1-based, with each parent's heap position.
struct NodeIdCachingHeap {
    max_size: usize,
    /// `heapNodes[1..=size]`; slot 0 unused.
    nodes: Vec<ParentChildScore>,
    size: usize,
    index: std::collections::HashMap<i32, usize>,
}

impl NodeIdCachingHeap {
    fn new(max_size: usize) -> Self {
        let empty = ParentChildScore {
            child: 0,
            ord: 0,
            parent: 0,
            score: 0.0,
        };
        Self {
            max_size,
            nodes: vec![empty; max_size.saturating_add(1).min(1 << 20)],
            size: 0,
            index: std::collections::HashMap::new(),
        }
    }

    fn top(&self) -> ParentChildScore {
        self.nodes[1]
    }

    fn set(&mut self, i: usize, v: ParentChildScore) {
        if i >= self.nodes.len() {
            self.nodes.resize(i + 1, v);
        }
        self.nodes[i] = v;
    }

    /// `insertWithOverflow(node, parentNode, score)`.
    fn insert_with_overflow(&mut self, v: ParentChildScore) -> bool {
        if let Some(&at) = self.index.get(&v.parent) {
            if self.nodes[at].score < v.score {
                self.update_element(at, v);
                return true;
            }
            return false;
        }
        if self.size >= self.max_size {
            let top = self.nodes[1];
            if v.score < top.score || (v.score == top.score && v.child > top.child) {
                return false;
            }
            // `updateTop`.
            self.index.remove(&top.parent);
            self.nodes[1] = v;
            self.down_heap(1, true);
            return true;
        }
        // `pushIn`.
        self.size += 1;
        let at = self.size;
        self.set(at, v);
        self.up_heap(at);
        true
    }

    /// `updateElement(heapIndex, nodeId, parentId, score)`.
    fn update_element(&mut self, at: usize, v: ParentChildScore) {
        let old = self.nodes[at].score;
        self.nodes[at] = v;
        if v.score < old {
            self.up_heap(at);
        } else {
            self.down_heap(at, true);
        }
    }

    fn up_heap(&mut self, orig: usize) {
        let mut i = orig;
        let bottom = self.nodes[i];
        let mut j = i >> 1;
        while j > 0 && bottom.cmp(&self.nodes[j]).is_lt() {
            self.nodes[i] = self.nodes[j];
            self.index.insert(self.nodes[i].parent, i);
            i = j;
            j >>= 1;
        }
        self.index.insert(bottom.parent, i);
        self.nodes[i] = bottom;
    }

    /// `downHeap(i)`, or `downHeapWithoutCacheUpdate(i)` without `cache`.
    fn down_heap(&mut self, mut i: usize, cache: bool) {
        let node = self.nodes[i];
        let mut j = i << 1;
        let mut k = j + 1;
        if k <= self.size && self.nodes[k].cmp(&self.nodes[j]).is_lt() {
            j = k;
        }
        while j <= self.size && self.nodes[j].cmp(&node).is_lt() {
            self.nodes[i] = self.nodes[j];
            if cache {
                self.index.insert(self.nodes[i].parent, i);
            }
            i = j;
            j = i << 1;
            k = j + 1;
            if k <= self.size && self.nodes[k].cmp(&self.nodes[j]).is_lt() {
                j = k;
            }
        }
        if cache {
            self.index.insert(node.parent, i);
        }
        self.nodes[i] = node;
    }

    /// `popToDrain()`.
    fn pop_to_drain(&mut self) {
        if self.size > 0 {
            self.nodes[1] = self.nodes[self.size];
            self.size -= 1;
            self.down_heap(1, false);
        }
    }
}

/// `DiversifyingNearestChildrenKnnCollector`, behind the reader's
/// `OrdinalTranslatedKnnCollector`: an ordinal is translated to its document
/// before the parent is looked up.
struct DiversifyingCollector<'p, 'f> {
    k: usize,
    visit_limit: u64,
    visited: u64,
    parents: &'p FixedBitSet,
    ord_to_doc: &'f dyn Fn(i32) -> Result<i32>,
    heap: NodeIdCachingHeap,
    error: Option<Error>,
}

impl<'p, 'f> DiversifyingCollector<'p, 'f> {
    fn new(
        k: usize,
        visit_limit: u64,
        parents: &'p FixedBitSet,
        ord_to_doc: &'f dyn Fn(i32) -> Result<i32>,
    ) -> Self {
        Self {
            k,
            visit_limit,
            visited: 0,
            parents,
            ord_to_doc,
            heap: NodeIdCachingHeap::new(k.max(1)),
            error: None,
        }
    }

    /// `topDocs()`: the heap drained, best first, as `(doc, ord, score)`.
    fn top_docs(mut self) -> Result<Vec<(i32, i32, f32)>> {
        if let Some(e) = self.error.take() {
            return Err(e);
        }
        while self.heap.size > self.k {
            self.heap.pop_to_drain();
        }
        let n = self.heap.size;
        let mut out = vec![(0, 0, 0.0f32); n];
        for i in 1..=n {
            let top = self.heap.top();
            out[n - i] = (top.child, top.ord, top.score);
            self.heap.pop_to_drain();
        }
        Ok(out)
    }
}

impl KnnCollect for DiversifyingCollector<'_, '_> {
    fn k(&self) -> usize {
        self.k
    }
    fn early_terminated(&self) -> bool {
        self.visited >= self.visit_limit
    }
    fn inc_visited_count(&mut self, count: usize) {
        self.visited = self.visited.saturating_add(count as u64);
    }
    fn visited_count(&self) -> u64 {
        self.visited
    }
    fn visit_limit(&self) -> u64 {
        self.visit_limit
    }
    fn collect(&mut self, ord: i32, similarity: f32) -> bool {
        let doc = match (self.ord_to_doc)(ord) {
            Ok(d) => d,
            Err(e) => {
                self.error.get_or_insert(e);
                return false;
            }
        };
        let parent = usize::try_from(doc)
            .ok()
            .and_then(|d| self.parents.next_set_bit(d))
            .and_then(|p| i32::try_from(p).ok())
            .unwrap_or(i32::MAX);
        self.heap.insert_with_overflow(ParentChildScore {
            child: doc,
            ord,
            parent,
            score: similarity,
        })
    }
    fn min_competitive_similarity(&self) -> f32 {
        if self.heap.size >= self.k {
            self.heap.top().score
        } else {
            f32::NEG_INFINITY
        }
    }
}

/// A diversified hit: the child's document, its ordinal and its similarity.
type ChildHit = (i32, i32, f32);

/// `DiversifyingChildren*KnnVectorQuery.exactSearch`: each parent's best
/// accepted child (the first on a tie), the best `min(k, cost)` of those.
/// Java walks the accepted documents; this walks their ordinals, which
/// ascend with them. The two are the same walk: Java's filter is
/// `childFilter AND FieldExistsQuery(field)`, so every accepted document has
/// a vector, and so an ordinal.
fn diversified_exact_search<S: VectorScorer>(
    scorer: &mut S,
    accept_ords: &FixedBitSet,
    cost: usize,
    k: usize,
    parents: &FixedBitSet,
    ord_to_doc: &dyn Fn(i32) -> Result<i32>,
) -> Result<Vec<(i32, i32, f32)>> {
    let queue_size = k.min(cost);
    if queue_size == 0 {
        return Ok(Vec::new());
    }
    // `HitQueue(queueSize, true)`: prefilled with sentinels (`-inf`,
    // `Integer.MAX_VALUE`), its top the worst entry -- the lowest score, then
    // the highest document -- replaced whenever a parent's best beats it.
    let mut queue: std::collections::BinaryHeap<WorstFirst> = (0..queue_size)
        .map(|_| WorstFirst((i32::MAX, -1, f32::NEG_INFINITY)))
        .collect();
    let mut current_parent = -1i64;
    let mut current: Option<(i32, i32, f32)> = None;
    let offer = |queue: &mut std::collections::BinaryHeap<WorstFirst>, hit: (i32, i32, f32)| {
        if let Some(mut top) = queue.peek_mut() {
            if hit.2 > top.0 .2 {
                top.0 = hit;
            }
        }
    };
    let end = scorer.max_ord().max(0) as usize;
    let span = scorer.read_ahead_span();
    let mut read_until = 0usize;
    let mut next = accept_ords.next_set_bit(0);
    while let Some(o) = next.filter(|&o| o < end) {
        next = accept_ords.next_set_bit(o + 1);
        if span > 0 && o >= read_until {
            read_until = read_ahead_dense(scorer, accept_ords, o, span, end);
        }
        let ord = o as i32;
        let doc = ord_to_doc(ord)?;
        let parent = usize::try_from(doc)
            .ok()
            .and_then(|d| parents.next_set_bit(d))
            .map_or(i64::MAX, |p| p as i64);
        if parent != current_parent {
            if let Some(hit) = current.take() {
                offer(&mut queue, hit);
            }
            current_parent = parent;
        }
        let score = scorer.score(ord)?;
        if current.is_none_or(|(_, _, s)| score > s) {
            current = Some((doc, ord, score));
        }
    }
    if let Some(hit) = current {
        offer(&mut queue, hit);
    }
    // `while (queue.size() > 0 && queue.top().score < 0) pop()`, then the
    // rest popped worst first into the array from its end.
    while queue.peek().is_some_and(|w| w.0 .2 < 0.0) {
        queue.pop();
    }
    // Ascending under `WorstFirst` is best first.
    Ok(queue.into_sorted_vec().into_iter().map(|w| w.0).collect())
}

/// A `HitQueue` entry ordered so a max-heap's top is `HitQueue`'s: the lowest
/// score (`Float.compare`), then the highest document.
struct WorstFirst((i32, i32, f32));

impl PartialEq for WorstFirst {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o).is_eq()
    }
}
impl Eq for WorstFirst {}
impl PartialOrd for WorstFirst {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for WorstFirst {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        java_float_compare(o.0 .2, self.0 .2).then(self.0 .0.cmp(&o.0 .0))
    }
}

/// `Float.compare`: `-0.0` below `0.0`, every `NaN` one value above all.
fn java_float_compare(a: f32, b: f32) -> std::cmp::Ordering {
    let canon = |x: f32| if x.is_nan() { f32::NAN } else { x };
    canon(a).total_cmp(&canon(b))
}

/// `DiversifyingChildren*KnnVectorQuery.approximateSearch`: the reader's
/// `searchNearestVectors` into a `DiversifyingNearestChildrenKnnCollector`.
fn diversified_approximate<S: VectorScorer>(
    scorer: &mut S,
    graph: &LeafGraph<'_, '_>,
    k: usize,
    limit: u64,
    options: SearchOptions<'_>,
    parents: &FixedBitSet,
    ord_to_doc: &dyn Fn(i32) -> Result<i32>,
) -> Result<(Vec<ChildHit>, bool)> {
    // `Lucene90HnswVectorsReader.search` collects documents already.
    let identity = |o: i32| -> Result<i32> { Ok(o) };
    let translate: &dyn Fn(i32) -> Result<i32> = if graph.hits_are_ordinals() {
        &identity
    } else {
        ord_to_doc
    };
    let mut c = DiversifyingCollector::new(k, limit, parents, translate);
    if scorer.max_ord() > 0 && k > 0 {
        graph.search_with(scorer, &mut c, options)?;
    }
    let early = KnnCollect::early_terminated(&c);
    Ok((c.top_docs()?, early))
}

/// [`leaf_results`] for the diversifying child queries: the same
/// `getLeafResults` branches, with `DiversifyingNearestChildrenKnnCollector`
/// and the parent-grouping `exactSearch`. A leaf without parents has no
/// results (its collector manager returns `null`).
#[allow(clippy::too_many_arguments)]
fn diversified_leaf_results<S: VectorScorer>(
    scorer: &mut S,
    graph: &LeafGraph<'_, '_>,
    accept: Option<&AcceptOrds<'_>>,
    ord_to_doc: &dyn Fn(i32) -> Result<i32>,
    max_doc: i32,
    plan: &LeafPlan,
    seed_ords: Option<&[i32]>,
    parents: Option<&FixedBitSet>,
) -> Result<(LeafHits, bool)> {
    let Some(parents) = parents else {
        return Ok((LeafHits::default(), false));
    };
    let size = scorer.max_ord().max(0) as usize;
    let collector_k = plan.collector_k.min(size);
    let per_leaf_top_k = plan.per_leaf_top_k.min(size);
    let accept_bits = accept.map(|a| a.bits());
    let (hits, early) = if !plan.filtered {
        diversified_approximate(
            scorer,
            graph,
            collector_k,
            plan.visited_limit,
            SearchOptions {
                accept_ords: accept_bits,
                filtered_doc_count: Some(max_doc),
                seed_ords,
                filtered_search_threshold: plan.extras.filtered_search_threshold,
            },
            parents,
            ord_to_doc,
        )?
    } else {
        let bits = accept_bits.expect("a filtered leaf always has an accept set");
        let cost = bits.cardinality();
        if cost <= per_leaf_top_k {
            (
                diversified_exact_search(scorer, bits, cost, plan.k, parents, ord_to_doc)?,
                false,
            )
        } else {
            let limit = plan.visited_limit.min(cost as u64 + 1);
            let (hits, early) = diversified_approximate(
                scorer,
                graph,
                collector_k,
                limit,
                SearchOptions {
                    accept_ords: Some(bits),
                    filtered_doc_count: Some(cost as i32),
                    seed_ords,
                    filtered_search_threshold: plan.extras.filtered_search_threshold,
                },
                parents,
                ord_to_doc,
            )?;
            if (!early && hits.len() >= per_leaf_top_k) || plan.extras.timed_out() {
                (hits, early)
            } else {
                (
                    diversified_exact_search(scorer, bits, cost, plan.k, parents, ord_to_doc)?,
                    false,
                )
            }
        }
    };
    let mut ords: Vec<i32> = hits.iter().map(|h| h.1).collect();
    ords.sort_unstable();
    Ok((
        LeafHits {
            hits: hits
                .into_iter()
                .map(|(doc_id, _, score)| ScoreDoc { doc_id, score })
                .collect(),
            ords,
        },
        early,
    ))
}

/// `IndexSearcher.search(DiversifyingChildrenFloatKnnVectorQuery, ..)`'s
/// rewrite over a multi-segment index: [`search_knn_float_vector_query_multi_segment`]'s
/// fan-out, pro-rata sizing, optimistic re-entry and merge, each leaf's
/// collector keeping only the best child per parent
/// (`DiversifyingNearestChildrenKnnCollectorManager`, optimistic too). Hits
/// are children. `parents` has one entry per segment: its parent filter's
/// bit set, `None` (or no entry) for a segment without parents -- no results
/// there.
pub fn search_diversifying_children_float_knn_multi_segment(
    segments: &[KnnSegment<'_>],
    parents: &[Option<&FixedBitSet>],
    query: &KnnFloatVectorQuery,
) -> Result<Vec<ScoreDoc>> {
    knn_multi_segment_with(
        segments,
        query,
        false,
        LeafExtras::default(),
        None,
        Some(parents),
    )
}

/// `DiversifyingChildrenByteKnnVectorQuery`'s equivalent of
/// [`search_diversifying_children_float_knn_multi_segment`].
pub fn search_diversifying_children_byte_knn_multi_segment(
    segments: &[KnnSegment<'_>],
    parents: &[Option<&FixedBitSet>],
    query: &KnnByteVectorQuery,
) -> Result<Vec<ScoreDoc>> {
    knn_multi_segment_with(
        segments,
        query,
        false,
        LeafExtras::default(),
        None,
        Some(parents),
    )
}

#[cfg(test)]
mod tests;
