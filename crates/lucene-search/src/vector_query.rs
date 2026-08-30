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

use lucene_codecs::field_infos::{FieldInfos, VectorEncoding, VectorSimilarityFunction};
use lucene_codecs::hnsw::{HnswGraphView, KnnCollector, VectorScorer};
use lucene_codecs::hnsw_vectors::{self, HnswVectorsReader, SearchOptions};
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
    /// `Lucene99HnswVectorsReader` over this segment's `.vem`/`.vex`, or
    /// `None` when the caller opened no graph. `None` makes every search the
    /// exhaustive scan Java also falls back to -- exact, just `O(size)`.
    pub hnsw: Option<HnswVectorsReader<'d>>,
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
fn accept_ords<'a>(
    input: &VectorsInput<'a>,
    resolved: &ResolvedField,
    ord_to_doc: &impl Fn(i32) -> Result<i32>,
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
    let mut bits = FixedBitSet::new(size.max(0) as usize);
    for ord in 0..size {
        let doc = if identity { ord } else { ord_to_doc(ord)? };
        if doc < 0 {
            continue;
        }
        let doc = doc as usize;
        let live = input.live_docs.is_none_or(|b| doc < b.len() && b.get(doc));
        let passes = input.filter.is_none_or(|b| doc < b.len() && b.get(doc));
        if live && passes {
            bits.set(ord as usize);
        }
    }
    Ok(Some(AcceptOrds::Owned(bits)))
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
    for ord in 0..scorer.max_ord() {
        if !accept_ords.get(ord as usize) {
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
fn leaf_results<S: VectorScorer, G: HnswGraphView>(
    scorer: &mut S,
    graph: Option<&G>,
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

    let (hits, early) = if !plan.filtered {
        // Java's `filterWeight == null` branch: `AcceptDocs.fromLiveDocs`,
        // `visitedLimit = Integer.MAX_VALUE`, and no cost heuristic.
        // `filteredDocCount` is `min(maxDoc, graphSize)` here even on a
        // segment with deletions -- see [`SearchOptions::filtered_doc_count`]
        // for why that is not the bug it looks like.
        hnsw_vectors::search(
            scorer,
            graph,
            collector_k,
            plan.visited_limit,
            SearchOptions {
                accept_ords: accept_bits,
                filtered_doc_count: Some(max_doc),
                seed_ords,
            },
        )?
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
            (exact_search(scorer, bits, cost, plan.k)?, false)
        } else {
            // "We pass cost + 1 here to account for the edge case when we
            // explore exactly cost vectors."
            let limit = plan.visited_limit.min(cost as u64 + 1);
            let (hits, early) = hnsw_vectors::search(
                scorer,
                graph,
                collector_k,
                limit,
                SearchOptions {
                    accept_ords: Some(bits),
                    filtered_doc_count: Some(cost as i32),
                    seed_ords,
                },
            )?;
            if !early && hits.len() >= per_leaf_top_k {
                (hits, early)
            } else {
                // "We stopped the kNN search because it visited too many
                // nodes, so fall back to exact search."
                (exact_search(scorer, bits, cost, plan.k)?, false)
            }
        }
    };

    // Java's `OrdinalTranslatedKnnCollector`, plus this leaf's own seed set
    // for a possible second pass: `SeededKnnVectorQuery.TopDocsDISI` sorts
    // the hits' local doc ids ascending and `MappedDISI` turns each into its
    // ordinal, and ordinals ascend with doc ids by construction -- so the
    // seed list is this hit set's ordinals, ascending.
    let mut out = Vec::with_capacity(hits.len());
    let mut ords = Vec::with_capacity(hits.len());
    for (ord, score) in hits {
        ords.push(ord);
        out.push(ScoreDoc {
            doc_id: ord_to_doc(ord)?,
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
) -> Result<(LeafHits, bool)> {
    // The graph is optional twice over: the caller may have opened no
    // `.vem`/`.vex`, and a field written below `HNSW_GRAPH_THRESHOLD`
    // documents carries none even when they were opened. Both mean the same
    // thing -- take the exhaustive branch.
    let graph = match &input.hnsw {
        None => None,
        Some(reader) => reader.graph(resolved.field_number).map_err(|e| match e {
            // A field the `.vemf` has and the `.vem` does not is a caller
            // mistake, not a damaged index -- Java's `getFieldEntryOrThrow`
            // raises `IllegalArgumentException` for it. Unreachable with
            // Lucene-written files (both metas list every vector field), but
            // the two must not be confused: `lucene-ffi` turns a decode error
            // into "this index is corrupt".
            lucene_codecs::vectors::Error::UnknownField(number) => Error::InvalidKnnQuery(format!(
                "field number {number} has vectors but no HNSW graph entry in this segment's .vem"
            )),
            other => Error::Vectors(other),
        })?,
    };
    match target {
        Target::Float(t) => {
            let values = input.flat.float_vector_values(resolved.field_number)?;
            let ord_to_doc = |ord: i32| Ok(values.ord_to_doc(ord)?);
            let accept = accept_ords(input, resolved, &ord_to_doc)?;
            let mut scorer = values.scorer(t)?;
            leaf_results(
                &mut scorer,
                graph.as_ref(),
                accept.as_ref(),
                &ord_to_doc,
                input.max_doc,
                plan,
                seed_ords,
            )
        }
        Target::Byte(t) => {
            let values = input.flat.byte_vector_values(resolved.field_number)?;
            let ord_to_doc = |ord: i32| Ok(values.ord_to_doc(ord)?);
            let accept = accept_ords(input, resolved, &ord_to_doc)?;
            let mut scorer = values.scorer(t)?;
            leaf_results(
                &mut scorer,
                graph.as_ref(),
                accept.as_ref(),
                &ord_to_doc,
                input.max_doc,
                plan,
                seed_ords,
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
    };
    let (mut leaf, _) = search_leaf(input, &resolved, query.target(), &plan, None)?;
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
    let k = query.k();
    let (resolved, plans) = plan_leaves(segments, query)?;

    let phase1 = run_leaves(
        segments,
        &resolved,
        query,
        &(0..segments.len()).collect::<Vec<_>>(),
        &plans,
        None,
        concurrent,
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
            )?;
            for (&i, (leaf, _)) in reenter.iter().zip(phase2) {
                per_leaf[i] = leaf.hits;
            }
            merged = merge_leaves(segments, &per_leaf, k)?;
        }
    }
    Ok(merged)
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
) -> Result<Vec<(LeafHits, bool)>> {
    let seed_for = |i: usize| seed_slice(seeds, i);
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
                )
            })
            .collect()
    }
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
}
