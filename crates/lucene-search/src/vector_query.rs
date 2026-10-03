//! Search-side KNN (approximate nearest neighbour) vector queries: the
//! *query-level* half of Lucene's vector search, on top of the codec half
//! `c5-vectors` ported into [`lucene_codecs::vectors`] (the
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
//! [`crate::query::BooleanQuery`] (`Occur::FILTER` clauses included, since
//! c11) into a `Vec<i32>` and [`accept_bitset`] turns that into the bitset
//! this module wants. Java's implicit `FieldExistsQuery(field)` conjunct is
//! folded in here too, for free: translating the accept set into **ordinal**
//! space drops every document that has no vector, which is all that conjunct
//! does.
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
use lucene_codecs::field_infos::{FieldInfos, VectorEncoding, VectorSimilarityFunction};
use lucene_codecs::hnsw::{KnnCollect, KnnCollector, VectorScorer};
use lucene_codecs::hnsw_vectors::{self, HnswVectorsReader, OffHeapHnswGraph, SearchOptions};
use lucene_codecs::vectors::{FlatFieldEntry, FlatVectorsReader};
use lucene_util::fixed_bit_set::FixedBitSet;

use crate::collector::{ScoreDoc, ScoringCollector};
use crate::multi_segment::merge_multi_segment_scored;
use crate::{Error, Result};

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
        None => LeafGraph::Lucene99(None),
        Some(GraphReader::Retired(reader)) => LeafGraph::Retired(reader, field_number),
        Some(GraphReader::Lucene99(reader)) => {
            LeafGraph::Lucene99(lucene99_graph(reader, field_number)?)
        }
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
                })
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
    if !identity {
        if let Some(bits) = accept_ords_from_docs(input, size, values)? {
            return Ok(Some(AcceptOrds::Owned(bits)));
        }
    }
    let mut bits = FixedBitSet::new(size.max(0) as usize);
    // Every ordinal's document, decoded in one bulk pass (`ordToDoc` per
    // ordinal is a `DirectMonotonicReader` lookup each).
    let mut docs = Vec::new();
    if !identity {
        (values.ord_to_docs)(&mut docs)?;
    }
    for ord in 0..size {
        let doc = if identity {
            ord
        } else {
            // `ord_to_docs` yields exactly `size` documents; a short list
            // would only leave the rest unaccepted.
            docs.get(ord as usize).copied().unwrap_or(-1)
        };
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
    let accepted_ords = accept_ords.len();
    for ord in 0..scorer.max_ord() {
        let ord_idx = ord as usize;
        if ord_idx >= accepted_ords || !accept_ords.get(ord_idx) {
            continue;
        }
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
}

impl LeafExtras {
    fn timed_out(&self) -> bool {
        self.deadline
            .is_some_and(|d| std::time::Instant::now() >= d)
    }
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
    diversify: Option<Option<&FixedBitSet>>,
) -> Result<(LeafHits, bool)> {
    if let Some(parents) = diversify {
        return diversified_leaf_results(
            scorer, graph, accept, ord_to_doc, max_doc, plan, seed_ords, parents,
        );
    }
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
        if cost <= per_leaf_top_k {
            // "If there are <= perLeafTopK possible matches, short-circuit
            // and perform exact search, since HNSW must always visit at
            // least perLeafTopK documents."
            //
            // Seeding does not reach here, exactly as in Java: the search
            // strategy is read by `HnswGraphSearcher.search`, and this branch
            // never calls it.
            (exact_search(scorer, bits, cost, plan.k)?, false, false)
        } else {
            // "We pass cost + 1 here to account for the edge case when we
            // explore exactly cost vectors."
            let limit = plan.visited_limit.min(cost as u64 + 1);
            let (hits, early) = graph.search(
                scorer,
                collector_k,
                limit,
                SearchOptions {
                    accept_ords: Some(bits),
                    filtered_doc_count: Some(cost as i32),
                    seed_ords,
                    filtered_search_threshold: plan.extras.filtered_search_threshold,
                },
                &plan.extras,
            )?;
            if (!early && hits.len() >= per_leaf_top_k) || plan.extras.timed_out() {
                (hits, early, true)
            } else {
                // "We stopped the kNN search because it visited too many
                // nodes, so fall back to exact search."
                (exact_search(scorer, bits, cost, plan.k)?, false, false)
            }
        }
    };

    // Java's `OrdinalTranslatedKnnCollector`, plus this leaf's own seed set
    // for a possible second pass: `SeededKnnVectorQuery.TopDocsDISI` sorts
    // the hits' local doc ids ascending and `MappedDISI` turns each into its
    // ordinal, and ordinals ascend with doc ids by construction -- so the
    // seed list is this hit set's ordinals, ascending.
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
    diversify: Option<Option<&FixedBitSet>>,
) -> Result<(LeafHits, bool)> {
    let graph = leaf_graph(
        input,
        resolved.field_number,
        matches!(target, Target::Float(_)),
    )?;
    match target {
        Target::Float(t) => {
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
                ..extras
            },
        });
    }
    Ok((resolved, plans))
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
    let parents_for = |i: usize| diversify.map(|d| d.get(i).copied().flatten());
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
    segments
        .iter()
        .map(|seg| {
            let docs = crate::exec::extended::segment_matches(seg, filter)?;
            Ok(accept_bitset(docs, seg.max_doc.unwrap_or(0)))
        })
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
                    && filter.and_then(|f| f.get(i)).is_none_or(|b| b.get_doc(d))
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
#[derive(Debug, Clone, PartialEq)]
pub struct VectorSimilarityQuery<T> {
    pub field: String,
    pub target: Vec<T>,
    pub result_similarity: f32,
    pub decay: f32,
}

/// `FloatVectorSimilarityQuery`.
pub type FloatVectorSimilarityQuery = VectorSimilarityQuery<f32>;
/// `ByteVectorSimilarityQuery`.
pub type ByteVectorSimilarityQuery = VectorSimilarityQuery<u8>;

impl<T> VectorSimilarityQuery<T> {
    /// `AbstractVectorSimilarityQuery.DEFAULT_DECAY`.
    pub const DEFAULT_DECAY: f32 = 0.5;

    /// The constructor's checks: a similarity that is a number, a decay in
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
        use crate::knn_collectors::{VectorSimilarityCollector, DECAY_MAX_QUALITY};
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
            let mut collector =
                VectorSimilarityCollector::new(self.result_similarity, self.decay, limit);
            graph.search_with(
                scorer,
                &mut collector,
                SearchOptions {
                    accept_ords: accept_bits,
                    filtered_doc_count: Some(i32::try_from(self.cardinality).unwrap_or(i32::MAX)),
                    seed_ords: None,
                    // `AbstractVectorSimilarityQuery.DEFAULT_STRATEGY`: `Hnsw(0)`.
                    filtered_search_threshold: 0,
                },
            )?;
            let (hits, early) = collector.into_hits();
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
    let accepted = accept_ords.len();
    for ord in 0..scorer.max_ord() {
        let o = ord as usize;
        if o >= accepted || !accept_ords.get(o) {
            continue;
        }
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
mod tests {
    use super::*;

    #[test]
    fn per_leaf_top_k_is_javas_pro_rata_formula() {
        // One leaf: proportion 1, variance 0, so it is exactly `k`.
        assert_eq!(per_leaf_top_k(10, 1.0), 10);
        assert_eq!(per_leaf_top_k(1, 1.0), 1);
        // Four equal leaves, k = 10: 2.5 + 16*sqrt(10*0.25*0.75) = 24.4 -> 24.
        assert_eq!(per_leaf_top_k(10, 0.25), 24);
        // Two equal leaves, k = 10: 5 + 16*sqrt(2.5) = 30.29 -> 30.
        assert_eq!(per_leaf_top_k(10, 0.5), 30);
        // A vanishing leaf still gets a slot (Java's `Math.max(1, ..)`, and
        // its `assert perLeafTopK > 0`).
        assert_eq!(per_leaf_top_k(10, 0.0), 1);
        // A zero-document index divides by zero. Java's `Math.max`
        // propagates the NaN and yields 0 (and trips its own `assert
        // perLeafTopK > 0`); this returns 1 -- see the function's own comment
        // for why that divergence is observably nothing.
        assert_eq!(per_leaf_top_k(10, f32::NAN), 1);
        // And an absurd k cannot overflow the cast.
        assert_eq!(per_leaf_top_k(usize::MAX, 1.0), i32::MAX as usize);
    }

    #[test]
    fn k_zero_is_rejected_like_javas_constructor() {
        let e = KnnFloatVectorQuery::new("f", vec![1.0], 0).unwrap_err();
        assert!(e.to_string().contains("k must be at least 1"), "{e}");
        let e = KnnByteVectorQuery::new("f", vec![1], 0).unwrap_err();
        assert!(e.to_string().contains("k must be at least 1"), "{e}");
        assert!(KnnFloatVectorQuery::new("f", vec![1.0], 1).is_ok());
    }

    #[test]
    fn query_builders_set_exactly_what_they_name() {
        let q = KnnFloatVectorQuery::new("f", vec![1.0], 3)
            .unwrap()
            .with_ef_search(50)
            .with_visited_limit(7)
            .with_similarity(VectorSimilarityFunction::Cosine);
        assert_eq!((q.k, q.ef_search, q.visited_limit), (3, 50, 7));
        assert_eq!(q.similarity, Some(VectorSimilarityFunction::Cosine));
        assert_eq!(visit_limit(&q), 7);
        let plain = KnnByteVectorQuery::new("f", vec![1], 3).unwrap();
        // `visited_limit == 0` is Java's "unlimited".
        assert_eq!(visit_limit(&plain), u64::MAX);
        assert_eq!(plain.similarity, None);
    }

    #[test]
    fn similarity_ordinals_are_the_pinned_file_format_order() {
        for ordinal in 0..4 {
            let s = similarity_from_ordinal(ordinal).unwrap();
            assert_eq!(similarity_ordinal(s), ordinal);
        }
        assert_eq!(similarity_from_ordinal(-1), None);
        assert_eq!(similarity_from_ordinal(4), None);
        assert_eq!(similarity_ordinal(VectorSimilarityFunction::Euclidean), 0);
        assert_eq!(similarity_ordinal(VectorSimilarityFunction::DotProduct), 1);
        assert_eq!(similarity_ordinal(VectorSimilarityFunction::Cosine), 2);
        assert_eq!(
            similarity_ordinal(VectorSimilarityFunction::MaximumInnerProduct),
            3
        );
    }

    #[test]
    fn accept_bitset_drops_out_of_range_doc_ids() {
        let bits = accept_bitset([0, 3, 7, 99, -1], 8);
        assert_eq!(bits.len(), 8);
        assert!(bits.get(0) && bits.get(3) && bits.get(7));
        assert_eq!(bits.cardinality(), 3);
        assert_eq!(accept_bitset([1], -5).len(), 0);
    }

    #[test]
    fn reentry_picks_exactly_the_leaves_that_are_not_tapped_out() {
        let sd = |doc, score| ScoreDoc { doc_id: doc, score };
        let per_leaf = vec![
            vec![sd(0, 0.9), sd(1, 0.8)], // worst 0.8 >= 0.75 -> re-enter
            vec![sd(2, 0.7), sd(3, 0.6)], // worst 0.6 <  0.75 -> tapped out
            vec![],                       // nothing at all    -> tapped out
        ];
        let merged = vec![sd(0, 0.9), sd(1, 0.8), sd(2, 0.75)];
        assert_eq!(reentry_leaves(&per_leaf, &merged), vec![0]);
        // With no merged hits at all there is nothing to compare against.
        assert!(reentry_leaves(&per_leaf, &[]).is_empty());
    }

    /// A scorer that counts the vector comparisons the walk performs, so a
    /// test can assert on *work done* rather than on a recall number (c5's
    /// Tier-2 lesson: recall does not discriminate here -- mutating the
    /// diversity rule took graph agreement to 1/4273 while recall rose).
    struct Counting<S> {
        inner: S,
        comparisons: usize,
    }

    impl<S: VectorScorer> VectorScorer for Counting<S> {
        fn score(&mut self, node: i32) -> lucene_codecs::vectors::Result<f32> {
            self.comparisons += 1;
            self.inner.score(node)
        }

        fn max_ord(&self) -> i32 {
            self.inner.max_ord()
        }

        fn bulk_score(
            &mut self,
            nodes: &[i32],
            scores: &mut [f32],
        ) -> lucene_codecs::vectors::Result<f32> {
            self.comparisons += nodes.len();
            self.inner.bulk_score(nodes, scores)
        }
    }

    /// What seeding actually buys, over the real fixture graph and asserted
    /// structurally rather than by a metric.
    ///
    /// Two properties, and both fail on a "seeded" search that quietly
    /// ignored its entry points:
    ///
    /// 1. **Seeding a walk with its own answer is a fixpoint.** Feeding the
    ///    unseeded top-`k`'s ordinals back in as entry points returns exactly
    ///    the same hits -- which is the reason Java can substitute phase 2's
    ///    seeded walk for a fresh descent at all.
    /// 2. **It skips `findBestEntryPoint`.** The seeded walk performs
    ///    strictly fewer vector comparisons, because the entire hill climb
    ///    over every level above 0 is not run.
    ///
    /// A third assertion pins that the seeds are *used* and not merely
    /// accepted: seeding from a single far-away ordinal reaches a different
    /// (worse) answer.
    #[test]
    fn a_seeded_walk_restarts_from_its_entry_points_and_skips_the_descent() {
        let dir = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/vectors_index/"
        );
        let text = std::fs::read_to_string(format!("{dir}manifest.properties"))
            .expect("run scripts/gen-fixtures.sh first (GenVectors)");
        let kv: std::collections::HashMap<&str, &str> =
            text.lines().filter_map(|l| l.split_once('=')).collect();
        let mut id = [0u8; 16];
        let hex = kv["id_hex"];
        for (i, slot) in id.iter_mut().enumerate() {
            *slot = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap();
        }
        let suffix = kv["segment_suffix"];
        let read = |name: &str| std::fs::read(format!("{dir}{name}")).expect("fixture file");
        let (vemf, vec_file) = (read(kv["vemf_file"]), read(kv["vec_file"]));
        let (vem, vex) = (read(kv["vem_file"]), read(kv["vex_file"]));
        let flat = FlatVectorsReader::open(&vemf, &vec_file, &id, suffix).unwrap();
        let hnsw = HnswVectorsReader::open(&vem, &vex, &id, suffix).unwrap();
        let field_number: i32 = kv["f0.number"].parse().unwrap();
        let values = flat.float_vector_values(field_number).unwrap();
        let graph = hnsw.graph(field_number).unwrap();
        assert!(graph.is_some(), "the dense fixture field carries a graph");
        let target: Vec<f32> = kv["q.f0.0.vec"]
            .split(',')
            .map(|s| f32::from_bits(s.parse::<i32>().unwrap() as u32))
            .collect();
        let k = 10;

        let run = |seeds: Option<&[i32]>, visit_limit: u64| {
            let mut scorer = Counting {
                inner: values.scorer(&target).unwrap(),
                comparisons: 0,
            };
            let (hits, _) = hnsw_vectors::search(
                &mut scorer,
                graph.as_ref(),
                k,
                visit_limit,
                SearchOptions {
                    seed_ords: seeds,
                    ..SearchOptions::default()
                },
            )
            .unwrap();
            (hits, scorer.comparisons)
        };

        let (plain, plain_cost) = run(None, u64::MAX);
        assert_eq!(plain.len(), k);
        let mut seeds: Vec<i32> = plain.iter().map(|(ord, _)| *ord).collect();
        seeds.sort_unstable();

        let (seeded, seeded_cost) = run(Some(&seeds), u64::MAX);
        assert_eq!(seeded, plain, "seeding a walk with its own answer moved it");
        assert!(
            seeded_cost < plain_cost,
            "the seeded walk did {seeded_cost} comparisons against {plain_cost}: it did not \
             skip the entry-point descent"
        );

        // And the seeds are genuinely *the starting set*, not a hint the walk
        // may discard. Capped at exactly one visit per seed, the beam cannot
        // move at all, so the answer is the seed set itself -- and one
        // arbitrary seed under the same cap is not it. (Without the cap this
        // graph is well-connected enough that even a single far-away entry
        // point converges to the same top 10, which is why the assertion
        // needs the cap to discriminate.)
        let far = values.scorer(&target).unwrap().max_ord() - 1;
        let (pinned, _) = run(Some(&seeds), seeds.len() as u64);
        assert_eq!(pinned, plain, "a visit-capped seeded walk left its seeds");
        let (stranded, _) = run(Some(&[far]), 1);
        assert_eq!(stranded.len(), 1);
        assert_eq!(stranded[0].0, far);
    }

    /// An empty phase-1 hit list for a leaf is "not seeded", not "seeded
    /// with nothing" -- Java's `ReentrantKnnCollectorManager` falls back to
    /// the unseeded collector there, and the codec's seeded entry point
    /// rejects an empty entry-point set outright (as Java's
    /// `fromEntryPoints` does). Getting this wrong turns a leaf that
    /// collected nothing in phase 1 into a hard error.
    #[test]
    fn an_empty_phase_one_hit_list_means_not_seeded() {
        let seeds = vec![vec![3, 7], Vec::new()];
        assert_eq!(seed_slice(Some(&seeds), 0), Some(&[3, 7][..]));
        assert_eq!(seed_slice(Some(&seeds), 1), None);
        // Phase 1 itself is never seeded.
        assert_eq!(seed_slice(None, 0), None);
    }

    /// Phase 2 restores the full `k` **on the collector only**. Java
    /// recomputes `perLeafTopK` inside `getLeafResults` from `ctx.parent` on
    /// every call, so the two cost thresholds stay pro-rata across both
    /// passes; only the collector manager changes.
    #[test]
    fn reentry_restores_the_full_k_on_the_collector_and_nothing_else() {
        let phase1 = LeafPlan {
            k: 10,
            per_leaf_top_k: 24,
            collector_k: 24,
            visited_limit: 99,
            filtered: true,
            extras: LeafExtras::default(),
        };
        let phase2 = reentry_plan(&phase1, 0);
        assert_eq!(phase2.collector_k, 10);
        assert_eq!(phase2.per_leaf_top_k, 24, "the threshold stays pro-rata");
        assert_eq!(phase2.k, 10);
        assert_eq!(phase2.visited_limit, 99);
        assert!(phase2.filtered);
        // A caller-widened beam widens the collector and nothing else: the
        // cost thresholds stay Java's.
        let widened = reentry_plan(&phase1, 40);
        assert_eq!(widened.collector_k, 40);
        assert_eq!(widened.per_leaf_top_k, 24);
    }
    /// A scorer over a fixed table of similarities, one per ordinal.
    struct Table(Vec<f32>);

    impl VectorScorer for Table {
        fn score(&mut self, node: i32) -> lucene_codecs::vectors::Result<f32> {
            Ok(self.0[node as usize])
        }

        fn max_ord(&self) -> i32 {
            self.0.len() as i32
        }
    }

    fn bits(len: usize, set: &[usize]) -> FixedBitSet {
        let mut b = FixedBitSet::new(len);
        for &i in set {
            b.set(i);
        }
        b
    }

    fn doc_of(ord: i32) -> Result<i32> {
        Ok(ord * 2)
    }

    /// Children at even docs (ordinal `o` is doc `2o`), parents closing
    /// blocks at docs 5, 11 and 19: blocks {0,2,4}, {6,8,10}, {12..18}, and
    /// doc 20 after the last parent (`NO_MORE_DOCS` is its parent).
    fn block_parents() -> FixedBitSet {
        bits(21, &[5, 11, 19])
    }

    #[test]
    fn diversified_exact_search_keeps_each_parents_best_child() {
        let parents = block_parents();
        //             block 0          block 1          block 2                block after
        let mut t = Table(vec![0.1, 0.5, 0.5, 0.2, 0.9, 0.3, 0.4, 0.4, 0.4, 0.4, 0.6]);
        let all = bits(11, &(0..11).collect::<Vec<_>>());
        let hits = diversified_exact_search(&mut t, &all, 11, 10, &parents, &doc_of).unwrap();
        // One per block, best first; a tie inside a block keeps the first
        // child (`score > currentScore`), the orphan block counts too.
        assert_eq!(
            hits,
            vec![(8, 4, 0.9), (20, 10, 0.6), (2, 1, 0.5), (12, 6, 0.4)]
        );
        // `k` (and `cost`) bound the queue; a tie between blocks keeps the
        // lower document (`HitQueue`'s order), whatever came first.
        let mut t = Table(vec![0.7, 0.0, 0.0, 0.7, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0]);
        let hits = diversified_exact_search(&mut t, &all, 11, 2, &parents, &doc_of).unwrap();
        assert_eq!(hits, vec![(0, 0, 0.7), (6, 3, 0.7)]);
        let hits = diversified_exact_search(&mut t, &all, 1, 5, &parents, &doc_of).unwrap();
        assert_eq!(hits, vec![(0, 0, 0.7)]);
        // Only accepted ordinals count; nothing accepted is nothing.
        let some = bits(11, &[2, 9]);
        let hits = diversified_exact_search(&mut t, &some, 2, 5, &parents, &doc_of).unwrap();
        assert_eq!(hits, vec![(4, 2, 0.0), (18, 9, 0.0)]);
        assert!(
            diversified_exact_search(&mut t, &some, 0, 5, &parents, &doc_of)
                .unwrap()
                .is_empty()
        );
        // Negative similarities are dropped with the sentinels.
        let mut t = Table(vec![
            -0.5, -0.5, -0.5, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]);
        let hits = diversified_exact_search(&mut t, &all, 11, 2, &parents, &doc_of).unwrap();
        assert_eq!(hits, vec![(6, 3, 0.25), (12, 6, 0.0)]);
        // A failing ordinal lookup is the caller's error.
        let broken = |_: i32| -> Result<i32> { Err(Error::IllegalArgument("ord".into())) };
        assert!(diversified_exact_search(&mut t, &all, 11, 2, &parents, &broken).is_err());
    }

    #[test]
    fn the_diversifying_collector_follows_javas_heap() {
        let parents = block_parents();
        let mut c = DiversifyingCollector::new(2, 100, &parents, &doc_of);
        assert_eq!(KnnCollect::k(&c), 2);
        assert_eq!(c.min_competitive_similarity(), f32::NEG_INFINITY);
        assert!(c.collect(0, 0.5)); // block 0
        assert!(!c.collect(1, 0.4)); // block 0, worse: kept 0
        assert!(c.collect(1, 0.6)); // block 0, better: replaces 0
        assert!(c.collect(4, 0.2)); // block 1
        assert_eq!(c.min_competitive_similarity(), 0.2);
        assert!(!c.collect(6, 0.1)); // block 2, below the floor
        assert!(!c.collect(7, 0.2)); // block 2, a tie with a higher child
        assert!(c.collect(10, 0.3)); // the orphan block evicts block 1
        assert!(c.collect(2, 0.7)); // block 0 improves again, in place
        assert!(c.collect(9, 0.3)); // block 2, a tie with a lower child: evicts doc 20
        c.inc_visited_count(3);
        assert_eq!((c.visited_count(), c.visit_limit()), (3, 100));
        assert!(!KnnCollect::early_terminated(&c));
        assert_eq!(c.top_docs().unwrap(), vec![(4, 2, 0.7), (18, 9, 0.3)]);

        // A worse score for a parent in the heap that must move down.
        let mut c = DiversifyingCollector::new(3, 1, &parents, &doc_of);
        for (ord, score) in [(0, 0.9), (3, 0.8), (6, 0.7)] {
            assert!(c.collect(ord, score));
        }
        assert!(c.collect(1, 0.95));
        assert!(c.collect(4, 0.85));
        c.inc_visited_count(1);
        assert!(KnnCollect::early_terminated(&c));
        assert_eq!(
            c.top_docs().unwrap(),
            vec![(2, 1, 0.95), (8, 4, 0.85), (12, 6, 0.7)]
        );

        // An ordinal that does not translate fails the search.
        let broken = |_: i32| -> Result<i32> { Err(Error::IllegalArgument("ord".into())) };
        let mut c = DiversifyingCollector::new(1, 10, &parents, &broken);
        assert!(!c.collect(0, 1.0));
        assert!(c.top_docs().is_err());
    }

    #[test]
    fn diversified_leaf_results_take_javas_branches() {
        let parents = block_parents();
        let plan = |filtered: bool, per_leaf_top_k: usize| LeafPlan {
            k: 2,
            per_leaf_top_k,
            collector_k: 2,
            visited_limit: u64::MAX,
            filtered,
            extras: LeafExtras::default(),
        };
        let mut t = Table(vec![0.1, 0.5, 0.5, 0.2, 0.9, 0.3, 0.4, 0.4, 0.4, 0.4, 0.6]);
        // No parents in the segment: no results.
        let (hits, early) = diversified_leaf_results(
            &mut t,
            &LeafGraph::ScanAll,
            None,
            &doc_of,
            21,
            &plan(false, 2),
            None,
            None,
        )
        .unwrap();
        assert!(hits.hits.is_empty() && !early);
        // Unfiltered: the reader's search (a full scan here).
        let (hits, _) = diversified_leaf_results(
            &mut t,
            &LeafGraph::ScanAll,
            None,
            &doc_of,
            21,
            &plan(false, 2),
            None,
            Some(&parents),
        )
        .unwrap();
        let docs: Vec<(i32, f32)> = hits.hits.iter().map(|h| (h.doc_id, h.score)).collect();
        assert_eq!(docs, vec![(8, 0.9), (20, 0.6)]);
        assert_eq!(hits.ords, vec![4, 10]);
        // Filtered below perLeafTopK: the exact search.
        let some = bits(11, &[0, 1, 6]);
        let accept = AcceptOrds::Borrowed(&some);
        let (hits, _) = diversified_leaf_results(
            &mut t,
            &LeafGraph::Nothing,
            Some(&accept),
            &doc_of,
            21,
            &plan(true, 5),
            None,
            Some(&parents),
        )
        .unwrap();
        let docs: Vec<i32> = hits.hits.iter().map(|h| h.doc_id).collect();
        assert_eq!(docs, vec![2, 12]);
        // Filtered above it: the graph first; one that finds too little (a
        // flat format collects nothing) falls back to the exact search, one
        // that finds enough is kept.
        let (hits, _) = diversified_leaf_results(
            &mut t,
            &LeafGraph::Nothing,
            Some(&accept),
            &doc_of,
            21,
            &plan(true, 1),
            None,
            Some(&parents),
        )
        .unwrap();
        assert_eq!(hits.hits.len(), 2);
        let (hits, early) = diversified_leaf_results(
            &mut t,
            &LeafGraph::ScanAll,
            Some(&accept),
            &doc_of,
            21,
            &plan(true, 1),
            None,
            Some(&parents),
        )
        .unwrap();
        let docs: Vec<i32> = hits.hits.iter().map(|h| h.doc_id).collect();
        assert_eq!((docs, early), (vec![2, 12], false));
    }
}
