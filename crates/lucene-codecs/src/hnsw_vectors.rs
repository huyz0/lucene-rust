//! `Lucene99HnswVectorsFormat`: the serialized HNSW graph, `.vex` (index) +
//! `.vem` (metadata).
//!
//! Port of `org.apache.lucene.codecs.lucene99.Lucene99HnswVectors{Format,
//! Reader,Writer}`, including its `OffHeapHnswGraph`.
//!
//! This format stores **only the graph**. The vectors it connects live in a
//! [`crate::vectors`] `.vec`/`.vemf` pair written by the same flush, and a
//! search needs both: the graph decides which ordinals to look at, the flat
//! store answers what they are.
//!
//! # Wire format
//!
//! `.vex` (vector index):
//! ```text
//! IndexHeader(codec="Lucene99HnswVectorsFormatIndex", version=1, id, suffix)
//! for each field:
//!   for level in 0..numLevels:            (level 0 first, nodes ascending)
//!     for each node on that level:
//!       NumNeighbors  --> vint            (after sorting and de-duplicating)
//!       Neighbors     --> group-vint deltas; the first is the absolute
//!                         ordinal, the rest are gaps
//!   NodeOffsets       --> DirectMonotonicWriter data: the cumulative byte
//!                         offset of each node's neighbour list, level-major
//! Footer
//! ```
//!
//! `.vem` (vector metadata):
//! ```text
//! IndexHeader(codec="Lucene99HnswVectorsFormatMeta", version=1, id, suffix)
//! for each field:
//!   FieldNumber            --> int32
//!   VectorEncoding         --> int32
//!   VectorSimilarityFunction --> int32
//!   VectorIndexOffset      --> vlong    (into .vex; excludes NodeOffsets)
//!   VectorIndexLength      --> vlong
//!   Dimension              --> vint
//!   Count                  --> int32
//!   M                      --> vint     (maxConn)
//!   NumLevels              --> vint     (0 when no graph was built)
//!   for level in 1..numLevels:
//!     NumNodesOnLevel      --> vint
//!     Nodes                --> vint * n (first absolute, rest deltas)
//!   if numLevels > 0:
//!     OffsetsOffset        --> int64    (into .vex)
//!     BlockShift           --> vint     (16)
//!     DirectMonotonicMeta  --> one tuple per block
//!     OffsetsLength        --> int64
//! -1                       --> int32
//! Footer
//! ```
//!
//! Note level 0 is **not** listed in `.vem`: it always holds every node, so
//! the reader reconstructs `0..count` itself.
//!
//! # Scope
//!
//! - `VERSION_GROUPVARINT` (1) is what this port writes; both 0 (plain vints)
//!   and 1 are read, because `versionMeta` selects the neighbour encoding and
//!   a Lucene 9.9-era segment is still version 0.
//! - No scalar quantization (`lucene104`'s formats
//!   layer on the same pair and are separate work) and no sorted-index
//!   (`writeSortingField`) path. `mergeOneField` **is** ported -- see
//!   [`merge_one_field`].
//! - [`search`] carries Java's `acceptOrds`, `filteredDocCount` and
//!   `KnnSearchStrategy.Seeded` entry points (see [`SearchOptions`]), so the
//!   filtered, deletions-aware and re-entrant searches
//!   `AbstractKnnVectorQuery` performs all run through this one dispatch
//!   rather than through a copy of it in the search layer.

use std::cell::RefCell;
use std::sync::Arc;

use lucene_store::codec_util::{self, ID_LENGTH};
use lucene_store::data_input::{DataInput, SliceInput};
use lucene_store::data_output::DataOutput;
use lucene_util::fixed_bit_set::FixedBitSet;

use crate::direct_monotonic;
use crate::field_infos::{VectorEncoding, VectorSimilarityFunction};
use crate::hnsw::{
    self, expected_visited_nodes, HnswGraphSearcher, HnswGraphView, KnnCollect, KnnCollector,
    OnHeapHnswGraph, UpdateableVectorScorer, VectorScorer,
};
use crate::vectors::{
    encoding_ordinal, read_similarity_function, read_vector_encoding, similarity_ordinal, Error,
    Result,
};

/// `Lucene99HnswVectorsFormat.META_CODEC_NAME`.
pub const META_CODEC: &str = "Lucene99HnswVectorsFormatMeta";
/// `Lucene99HnswVectorsFormat.VECTOR_INDEX_CODEC_NAME`.
pub const INDEX_CODEC: &str = "Lucene99HnswVectorsFormatIndex";
/// `Lucene99HnswVectorsFormat.META_EXTENSION`.
pub const META_EXTENSION: &str = "vem";
/// `Lucene99HnswVectorsFormat.VECTOR_INDEX_EXTENSION`.
pub const INDEX_EXTENSION: &str = "vex";

/// `Lucene99HnswVectorsFormat.VERSION_START`.
pub const VERSION_START: i32 = 0;
/// `Lucene99HnswVectorsFormat.VERSION_GROUPVARINT`.
pub const VERSION_GROUPVARINT: i32 = 1;
/// `Lucene99HnswVectorsFormat.VERSION_CURRENT`.
pub const VERSION_CURRENT: i32 = VERSION_GROUPVARINT;

/// `Lucene99HnswVectorsFormat.DIRECT_MONOTONIC_BLOCK_SHIFT`.
pub const DIRECT_MONOTONIC_BLOCK_SHIFT: u32 = 16;

/// `Lucene99HnswVectorsReader.EXHAUSTIVE_BULK_SCORE_ORDS`.
const EXHAUSTIVE_BULK_SCORE_ORDS: usize = 64;

fn corrupt<T>(msg: impl Into<String>) -> Result<T> {
    Err(Error::CorruptMeta(msg.into()))
}

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

/// One field's graph, as handed to [`write_hnsw_vectors`].
///
/// `graph == None` is Lucene's "tiny segment" case: the writer skips graph
/// construction when `expectedVisitedNodes(k, n) >= n` (see
/// [`crate::hnsw::should_create_graph`]) and records `numLevels = 0`, which
/// the reader turns into `HnswGraph.EMPTY` and an exhaustive search.
#[derive(Debug)]
pub struct HnswVectorsField<'g> {
    pub field_number: i32,
    pub encoding: VectorEncoding,
    pub similarity: VectorSimilarityFunction,
    pub dimension: i32,
    /// The number of vectors in the matching flat field.
    pub count: i32,
    pub graph: Option<&'g OnHeapHnswGraph>,
    /// `M`, recorded in `.vem` even when no graph was built (Java writes the
    /// format's configured `M` there).
    pub m: i32,
}

/// Port of `Lucene99HnswVectorsWriter.{writeField,writeGraph,writeMeta,
/// finish}`. Returns `(vex_bytes, vem_bytes)`.
pub fn write_hnsw_vectors(
    fields: &[HnswVectorsField<'_>],
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> Result<(Vec<u8>, Vec<u8>)> {
    let mut index = Vec::new();
    codec_util::write_index_header(
        &mut index,
        INDEX_CODEC,
        VERSION_CURRENT,
        segment_id,
        segment_suffix,
    );
    let mut meta = Vec::new();
    codec_util::write_index_header(
        &mut meta,
        META_CODEC,
        VERSION_CURRENT,
        segment_id,
        segment_suffix,
    );

    for field in fields {
        // The writer must not be able to emit a `.vem` its own reader (or
        // Lucene's) rejects. These are exactly `read_field_entry`'s bounds and
        // `Lucene99HnswVectorsFormat`'s constructor checks.
        if field.dimension <= 0 {
            return Err(Error::InvalidGraphParameter(format!(
                "field {}: vector dimension must be positive, got {}",
                field.field_number, field.dimension
            )));
        }
        if field.count < 0 {
            return Err(Error::InvalidGraphParameter(format!(
                "field {}: negative vector count {}",
                field.field_number, field.count
            )));
        }
        let m = field.graph.map_or(field.m, |g| g.max_conn());
        if !(1..=crate::hnsw::MAXIMUM_MAX_CONN).contains(&m) {
            return Err(Error::InvalidGraphParameter(format!(
                "field {}: M must be in 1..={}, got {m}",
                field.field_number,
                crate::hnsw::MAXIMUM_MAX_CONN
            )));
        }
        let vector_index_offset = index.len() as i64;
        let level_node_offsets = match field.graph {
            Some(graph) => write_graph(&mut index, graph)?,
            None => Vec::new(),
        };
        // ARITH: `vector_index_offset` was `index.len()` before `write_graph`
        // appended to it and nothing truncates `index`, so the current length
        // is at least that; a `Vec`'s length fits `i64` on every supported
        // target.
        #[allow(clippy::arithmetic_side_effects)]
        let vector_index_length = index.len() as i64 - vector_index_offset;
        write_meta(
            &mut meta,
            &mut index,
            field,
            vector_index_offset,
            vector_index_length,
            &level_node_offsets,
        )?;
    }

    meta.write_i32(-1);
    codec_util::write_footer(&mut meta);
    codec_util::write_footer(&mut index);
    Ok((index, meta))
}

/// One segment being merged, as far as one vector field's **graph** is
/// concerned. Pair it with the matching
/// [`crate::vectors::FlatVectorMergeSource`], in the same order: the merged
/// ordinal space this assumes is the one the flat merge assigns.
#[derive(Debug, Clone, Copy)]
pub struct GraphMergeSource<'a, G: HnswGraphView> {
    /// This segment's graph for the field, or `None` when it has none (a
    /// sub-threshold segment, or a codec that stores no graph). A source
    /// without a graph still contributes vectors -- they are simply inserted
    /// from scratch.
    pub graph: Option<&'a G>,
    /// The source's ordinal -> its own doc id, ascending
    /// (`KnnVectorValues.ordToDoc`, materialized). One entry per vector.
    pub ord_to_doc: &'a [i32],
    /// `MergeState.docMaps[i]`: merged doc id per source doc id, `-1` when
    /// dropped. Java also carries `MergeState.liveDocs[i]` to count live
    /// vectors; that count is exactly "ordinals whose doc maps to something",
    /// so this port derives it here rather than taking a second, redundant
    /// input that could disagree with the first.
    pub doc_map: &'a [i32],
}

/// `IncrementalHnswGraphMerger.DELETE_PCT_THRESHOLD`: a graph more than 40%
/// deleted has degraded connectivity and is not used as the base.
const DELETE_PCT_THRESHOLD: i32 = 40;

/// The most a base graph may have deleted for the *concurrent* merge to reuse
/// it, given `workers` merge threads. Reuse repairs every damaged node on the
/// merge thread (`fix_disconnected_nodes`), which costs roughly 1.9 s per
/// percent deleted at 100k x 128 vectors (M 16, beam 100) and saturates near
/// 60 s, while a rebuild spreads over the workers: 15-19 s on 4 workers. Break
/// even is therefore near `40 / workers` percent (about 10% on 4 workers); past
/// it the rebuild wins, 4-5x at 20-35% deleted (`hnsw_merge_repair_bench`).
/// One worker keeps Java's 40.
fn reuse_delete_pct_limit(workers: usize) -> i32 {
    let workers = i32::try_from(workers.max(1)).unwrap_or(i32::MAX);
    (DELETE_PCT_THRESHOLD / workers).max(5)
}

/// Port of `Lucene99HnswVectorsWriter.mergeOneField`'s graph half, i.e.
/// `buildAndWriteGraph` + `createGraphMerger` +
/// `IncrementalHnswGraphMerger.{addReader,createBuilder,getNewOrdMapping,merge}`.
///
/// Returns the merged segment's graph, or `None` when Lucene would write none
/// (`shouldCreateGraph` false, or no vectors at all) -- the caller then passes
/// `graph: None` to [`write_hnsw_vectors`], which records `numLevels = 0`.
///
/// **The point of this entry point is that it does not rebuild.** The largest
/// usable source graph is copied into the merged ordinal space and the others
/// are folded into it, so merging a big segment with a small one costs roughly
/// the small one's insertions instead of the sum. A from-scratch rebuild is the
/// fallback, taken only when no source graph qualifies.
///
/// `scorer` scores **merged** ordinals -- open it over the merged `.vec` the
/// flat merge just wrote, exactly as `mergeOneField` reopens the flat writer's
/// output before touching the graph. `merged_ord_to_doc` is that same merged
/// field's ordinal -> doc mapping; its length is the merged vector count.
///
/// `sources` must contain **exactly** the segments whose `FieldInfo` for this
/// field has a positive dimension, in merge order -- not every segment in the
/// merge. Java's `buildAndWriteGraph` filters with
/// `hasVectorValues(mergeState.fieldInfos[i], fieldInfo.name)` before calling
/// `addReader`, and the count matters: "every source contributed a usable
/// graph" is what decides whether any merged ordinal still has to be inserted
/// from scratch.
///
/// Note `M`: when a base graph is reused the merged graph inherits **its**
/// `maxConn`, not the `m` argument -- Java does the same
/// (`new OnHeapHnswGraph(initializerGraph.maxConn(), ...)`), so reconfiguring a
/// writer's `M` does not re-shape an existing graph on merge. `m` is used only
/// for the from-scratch rebuild.
pub fn merge_one_field<G: HnswGraphView, S: UpdateableVectorScorer>(
    scorer: S,
    m: i32,
    beam_width: i32,
    seed: u64,
    merged_ord_to_doc: &[i32],
    sources: &[GraphMergeSource<'_, G>],
) -> Result<Option<OnHeapHnswGraph>> {
    // `as i32` on a length is the shape `cast_sign_loss` exists to catch: a
    // slice longer than `i32::MAX` truncates, and a truncated *negative*
    // count then sizes `FixedBitSet::new(count as usize)` at ~2^64 bits. An
    // ordinal space is an `i32` in this format, so refusing one that does not
    // fit is the honest answer.
    let Ok(total_vector_count) = i32::try_from(merged_ord_to_doc.len()) else {
        return Err(Error::InvalidGraphParameter(format!(
            "{} merged vectors is more than an ordinal can name",
            merged_ord_to_doc.len()
        )));
    };
    if total_vector_count == 0
        || !hnsw::should_create_graph(hnsw::HNSW_GRAPH_THRESHOLD, total_vector_count)
    {
        return Ok(None);
    }
    // Checked before anything is allocated: `OnHeapHnswGraph::with_size`
    // asserts on a non-positive `M`, and a caller mistake must be an error, not
    // a panic. Same bounds `HnswGraphBuilder::new` enforces.
    if !(1..=hnsw::MAXIMUM_MAX_CONN).contains(&m) {
        return Err(Error::InvalidGraphParameter(format!(
            "M (max connections) must be in 1..={}, got {m}",
            hnsw::MAXIMUM_MAX_CONN
        )));
    }
    if !(1..=hnsw::MAXIMUM_BEAM_WIDTH).contains(&beam_width) {
        return Err(Error::InvalidGraphParameter(format!(
            "beamWidth must be in 1..={}, got {beam_width}",
            hnsw::MAXIMUM_BEAM_WIDTH
        )));
    }

    // `IncrementalHnswGraphMerger.addReader`, once per source.
    let (usable, largest) = add_readers(sources, DELETE_PCT_THRESHOLD)?;

    let Some(largest) = largest else {
        // `HnswGraphBuilder.create(scorerSupplier, M, beamWidth, randSeed, maxOrd)`:
        // nothing worth reusing, so build the whole graph. Note the `maxOrd`
        // argument -- the merge path passes the eventual node count where the
        // flush path passes `-1`, so the searcher's visited bitset is allocated
        // once at full size instead of growing.
        return Ok(Some(
            hnsw::HnswGraphBuilder::with_graph(
                scorer,
                beam_width,
                seed,
                OnHeapHnswGraph::with_size(m, total_vector_count),
            )?
            .build(total_vector_count)?,
        ));
    };

    // `createBuilder`: the base graph goes first; otherwise largest first.
    let order: Vec<usize> = if usable.contains(&largest) {
        let mut o = usable.clone();
        o.sort_by_key(|i| std::cmp::Reverse(sources[*i].graph.expect("checked above").size()));
        o
    } else {
        let mut o = vec![largest];
        o.extend(usable.iter().copied());
        o
    };

    // `graphReaders.size() == numReaders ? null : new FixedBitSet(maxOrd)`:
    // when every source contributed a usable graph, every merged ordinal is
    // covered by one of them and nothing is left to insert from scratch.
    let mut initialized = if order.len() == sources.len() {
        None
    } else {
        Some(FixedBitSet::new(total_vector_count as usize))
    };

    let ord_maps = new_ord_mapping(&order, sources, merged_ord_to_doc, initialized.as_mut())?;
    let graphs: Vec<&G> = order
        .iter()
        .map(|i| {
            sources[*i]
                .graph
                .expect("only sources with graphs are ordered")
        })
        .collect();

    Ok(Some(hnsw::merge_graphs(
        scorer,
        beam_width,
        seed,
        &graphs,
        &ord_maps,
        total_vector_count,
        initialized.as_ref(),
    )?))
}

/// Port of `ConcurrentHnswMerger` (`Lucene99HnswVectorsFormat` with
/// `numMergeWorkers > 1`): the same source selection as [`merge_one_field`],
/// but only the base graph is reused -- copied into the merged ordinal space
/// by `InitializedHnswGraphBuilder.initGraph` -- and every other merged
/// ordinal is inserted by `num_workers` threads
/// through [`crate::hnsw_concurrent::build_concurrent`], each with its own
/// scorer from `new_scorer` (Java's `scorerSupplier.copy()`).
///
/// With no usable base graph the whole graph is built concurrently. The
/// result depends on thread scheduling, as in Java.
#[allow(clippy::too_many_arguments)]
pub fn merge_one_field_concurrent<G, S, F>(
    new_scorer: F,
    m: i32,
    beam_width: i32,
    seed: u64,
    merged_ord_to_doc: &[i32],
    sources: &[GraphMergeSource<'_, G>],
    num_workers: usize,
    batch_size: i32,
) -> Result<Option<OnHeapHnswGraph>>
where
    G: HnswGraphView,
    S: UpdateableVectorScorer + Send,
    F: Fn() -> Result<S>,
{
    let Ok(total_vector_count) = i32::try_from(merged_ord_to_doc.len()) else {
        return Err(Error::InvalidGraphParameter(format!(
            "{} merged vectors is more than an ordinal can name",
            merged_ord_to_doc.len()
        )));
    };
    if total_vector_count == 0
        || !hnsw::should_create_graph(hnsw::HNSW_GRAPH_THRESHOLD, total_vector_count)
    {
        return Ok(None);
    }
    if !(1..=hnsw::MAXIMUM_MAX_CONN).contains(&m) || num_workers == 0 {
        return Err(Error::InvalidGraphParameter(format!(
            "M must be in 1..={} and at least one worker is needed (M {m}, {num_workers} workers)",
            hnsw::MAXIMUM_MAX_CONN
        )));
    }
    let (_, largest) = add_readers(sources, reuse_delete_pct_limit(num_workers))?;
    let (graph, initialized) = match largest {
        None => (OnHeapHnswGraph::with_size(m, total_vector_count), None),
        Some(largest) => {
            // `ConcurrentHnswMerger.getNewOrdMapping`: the base graph only.
            let mut bits = FixedBitSet::new(total_vector_count as usize);
            let maps = new_ord_mapping(&[largest], sources, merged_ord_to_doc, Some(&mut bits))?;
            let base = sources[largest].graph.expect("add_readers picks graphs");
            let graph = hnsw::HnswGraphBuilder::init_graph(
                new_scorer()?,
                beam_width,
                seed,
                base,
                &maps[0],
                total_vector_count,
            )?;
            (graph, Some(bits))
        }
    };
    let scorers = (0..num_workers)
        .map(|_| new_scorer())
        .collect::<Result<Vec<S>>>()?;
    Ok(Some(crate::hnsw_concurrent::build_concurrent(
        scorers,
        m,
        beam_width,
        seed,
        graph,
        initialized.as_ref(),
        total_vector_count,
        batch_size,
    )?))
}

/// `IncrementalHnswGraphMerger.addReader` over every source: the sources
/// whose graphs are usable as they are (no deletions), and the one chosen as
/// the base graph (`largestGraphReader`), if any qualifies.
fn add_readers<G: HnswGraphView>(
    sources: &[GraphMergeSource<'_, G>],
    max_delete_pct: i32,
) -> Result<(Vec<usize>, Option<usize>)> {
    let mut usable: Vec<usize> = Vec::new();
    let mut largest: Option<usize> = None;
    let mut largest_live = -1i64;
    for (i, source) in sources.iter().enumerate() {
        let Some(graph) = source.graph else { continue };
        let graph_size = graph.size();
        if graph_size == 0 {
            continue;
        }
        // `!=` against `Ok(graph_size)`, not `len() as i32 != graph_size`: a
        // source with more than `i32::MAX` vectors would otherwise truncate
        // into agreement with a small `graph_size`, and `new_ord_mapping`
        // would then index `old_to_new` with a wrapped-negative ordinal.
        if i32::try_from(source.ord_to_doc.len()) != Ok(graph_size) {
            return Err(Error::InvalidGraphParameter(format!(
                "source {i}: graph has {graph_size} nodes but {} vectors",
                source.ord_to_doc.len()
            )));
        }
        let live = count_live_vectors(source)?;
        // ARITH: `live` counts a subset of `ord_to_doc`, whose length was just
        // checked to equal `graph_size`, so `0 <= live <= graph_size`, and
        // `graph_size > 0` was checked above. Java's
        // `((graphSize - candidateVectorCount) * 100) / graphSize` is `int`
        // arithmetic that silently wraps once `graphSize` passes ~21.5M
        // vectors -- a segment size this port can reach -- so the product is
        // taken in `i64` here, where `i32::MAX * 100` is three orders of
        // magnitude short of overflow.
        #[allow(clippy::arithmetic_side_effects)]
        let delete_pct = (i64::from(graph_size) - live) * 100 / i64::from(graph_size);
        // The candidate's live count is compared against the incumbent's live
        // count, so a graph with deletions does not stay the base over a
        // larger deletion-free one because of reader order.
        if delete_pct <= i64::from(max_delete_pct) && (largest.is_none() || live > largest_live) {
            largest = Some(i);
            largest_live = live;
        }
        if live == i64::from(graph_size) {
            usable.push(i);
        }
    }
    Ok((usable, largest))
}

/// `IncrementalHnswGraphMerger.countLiveVectors`.
fn count_live_vectors<G: HnswGraphView>(source: &GraphMergeSource<'_, G>) -> Result<i64> {
    let mut live = 0i64;
    for doc in source.ord_to_doc {
        // `*doc as usize` sign-extends a negative doc id to ~2^64, which no
        // `doc_map` covers, so a corrupt ordinal-to-doc mapping is reported
        // rather than indexed with.
        match source.doc_map.get(*doc as usize) {
            Some(&new_doc) => {
                if new_doc >= 0 {
                    // ARITH: one increment per element of `ord_to_doc`, and a
                    // slice of 4-byte elements cannot hold `i64::MAX` of them.
                    #[allow(clippy::arithmetic_side_effects)]
                    {
                        live += 1;
                    }
                }
            }
            None => {
                return Err(Error::InvalidGraphParameter(format!(
                    "doc map has no entry for source doc {doc}"
                )))
            }
        }
    }
    Ok(live)
}

/// `IncrementalHnswGraphMerger.getNewOrdMapping`: for each reused graph, the
/// merged ordinal of each of its own ordinals (`-1` for a vector this merge
/// drops). Also marks every merged ordinal some source graph covers, so the
/// caller knows which ones still have to be inserted from scratch.
fn new_ord_mapping<G: HnswGraphView>(
    order: &[usize],
    sources: &[GraphMergeSource<'_, G>],
    merged_ord_to_doc: &[i32],
    mut initialized: Option<&mut FixedBitSet>,
) -> Result<Vec<Vec<i32>>> {
    // Per graph: merged doc id -> that graph's own ordinal.
    let mut new_doc_to_old_ord: Vec<lucene_util::fx_hash::FxHashMap<i32, i32>> =
        Vec::with_capacity(order.len());
    let mut old_to_new: Vec<Vec<i32>> = Vec::with_capacity(order.len());
    for i in order {
        let source = &sources[*i];
        let mut map = lucene_util::fx_hash::FxHashMap::with_capacity_and_hasher(
            source.ord_to_doc.len(),
            Default::default(),
        );
        for (ord, doc) in source.ord_to_doc.iter().enumerate() {
            let new_doc = match source.doc_map.get(*doc as usize) {
                Some(&d) => d,
                None => {
                    return Err(Error::InvalidGraphParameter(format!(
                        "doc map has no entry for source doc {doc}"
                    )))
                }
            };
            if new_doc >= 0 {
                map.insert(new_doc, ord as i32);
            }
        }
        new_doc_to_old_ord.push(map);
        old_to_new.push(vec![-1; source.ord_to_doc.len()]);
    }

    for (new_ord, doc) in merged_ord_to_doc.iter().enumerate() {
        for i in 0..order.len() {
            if let Some(&old_ord) = new_doc_to_old_ord[i].get(doc) {
                old_to_new[i][old_ord as usize] = new_ord as i32;
                if let Some(bits) = initialized.as_deref_mut() {
                    // FBS: `initialized` is the caller's
                    // `FixedBitSet::new(total_vector_count)` and
                    // `total_vector_count` is `merged_ord_to_doc.len()`
                    // (`:285`) -- the very slice `new_ord` enumerates. The two
                    // reach this function as separate parameters, so the proof
                    // is the caller's; `write_hnsw_merged` is its only caller.
                    bits.set(new_ord);
                }
                break;
            }
        }
    }
    Ok(old_to_new)
}

/// Port of `Lucene99HnswVectorsWriter.writeGraph`: returns the *per-node*
/// (non-cumulative) byte lengths, level-major.
fn write_graph(index: &mut Vec<u8>, graph: &OnHeapHnswGraph) -> Result<Vec<Vec<i32>>> {
    let count_on_level0 = graph.size();
    let mut offsets: Vec<Vec<i32>> = Vec::with_capacity(graph.num_levels() as usize);
    // ARITH: `write_hnsw_vectors` -- this function's only caller -- rejects a
    // field whose `M` (the graph's own `max_conn()` when it has a graph) is
    // outside `1..=MAXIMUM_MAX_CONN` (512) before calling it, so the product
    // is at most 1024.
    #[allow(clippy::arithmetic_side_effects)]
    let mut scratch: Vec<u32> = Vec::with_capacity((graph.max_conn() * 2) as usize);
    let mut nnodes: Vec<i32> = Vec::new();
    for level in 0..graph.num_levels() {
        let sorted_nodes = graph.sorted_nodes_on_level(level)?;
        let mut level_offsets = Vec::with_capacity(sorted_nodes.len());
        for node in sorted_nodes {
            let neighbors = graph.neighbors(level, node);
            let offset_start = index.len();
            nnodes.clear();
            nnodes.extend_from_slice(neighbors.nodes());
            nnodes.sort_unstable();
            // Delta-encode after sorting, dropping duplicates -- Java's
            // `writeGraph` does the same and writes the *post-dedup* size.
            scratch.clear();
            // Java asserts `nnodes[i] < countOnLevel0` -- disabled in
            // production -- and only for `i >= 1`, leaving the *first* node
            // unchecked even though it is the one written as an absolute
            // ordinal. Every node is range-checked here instead: that is what
            // makes each delta below provably non-negative and in range, and
            // it stops a corrupt first ordinal being written as a ~4-billion
            // absolute ordinal that this module's own reader then rejects only
            // once the file is on disk.
            for (i, &node) in nnodes.iter().enumerate() {
                if node < 0 || node >= count_on_level0 {
                    return corrupt(format!(
                        "node out of range: {node} not in 0..{count_on_level0}"
                    ));
                }
                if i == 0 {
                    scratch.push(node as u32);
                    continue;
                }
                // ARITH: every element is in `0..count_on_level0` and the
                // slice was sorted ascending just above, so the difference is
                // in `0..count_on_level0` too.
                #[allow(clippy::arithmetic_side_effects)]
                if nnodes[i - 1] != node {
                    scratch.push((node - nnodes[i - 1]) as u32);
                }
            }
            index.write_vint(scratch.len() as i32);
            index.write_group_vints(&scratch);
            // ARITH: `offset_start` is `index.len()` from before the two
            // writes above, and `index` only grows. The `as i32` cannot
            // truncate either: one node's list is a vint plus at most
            // `2 * MAXIMUM_MAX_CONN` group-varints, under 6 KB. (Java spells
            // the same fact `Math.toIntExact`.)
            #[allow(clippy::arithmetic_side_effects)]
            level_offsets.push((index.len() - offset_start) as i32);
        }
        offsets.push(level_offsets);
    }
    Ok(offsets)
}

/// Port of `Lucene99HnswVectorsWriter.writeMeta`.
fn write_meta(
    meta: &mut Vec<u8>,
    index: &mut Vec<u8>,
    field: &HnswVectorsField<'_>,
    vector_index_offset: i64,
    vector_index_length: i64,
    level_node_offsets: &[Vec<i32>],
) -> Result<()> {
    meta.write_i32(field.field_number);
    meta.write_i32(encoding_ordinal(field.encoding));
    meta.write_i32(similarity_ordinal(field.similarity));
    meta.write_vlong(vector_index_offset);
    meta.write_vlong(vector_index_length);
    meta.write_vint(field.dimension);
    meta.write_i32(field.count);

    let Some(graph) = field.graph else {
        meta.write_vint(field.m);
        meta.write_vint(0);
        return Ok(());
    };

    meta.write_vint(graph.max_conn());
    meta.write_vint(graph.num_levels());
    let mut value_count = 0i64;
    for level in 0..graph.num_levels() {
        let nodes = graph.sorted_nodes_on_level(level)?;
        // ARITH: a sum of in-memory `Vec` lengths, so it is bounded by the
        // total number of graph entries held in memory -- a `usize`, hence
        // well inside `i64`.
        #[allow(clippy::arithmetic_side_effects)]
        {
            value_count += nodes.len() as i64;
        }
        if level > 0 {
            meta.write_vint(nodes.len() as i32);
            // Delta-encode from the back, exactly as Java's in-place loop
            // `for (i = n-1; i > 0; --i) nol[i] -= nol[i-1]` does.
            let mut deltas = nodes.clone();
            for i in (1..deltas.len()).rev() {
                // Checked rather than proved: `sorted_nodes_on_level` is a
                // public trait method, so "ascending, non-negative" is a
                // promise made by whatever `HnswGraphView` the caller handed
                // in, not an invariant of this module. A descending pair would
                // otherwise overflow (`i32::MAX - i32::MIN`) or write a
                // negative delta this format cannot represent.
                // ARITH: `i` comes from `1..deltas.len()`, so `i - 1` is in
                // range.
                #[allow(clippy::arithmetic_side_effects)]
                let previous = deltas[i - 1];
                deltas[i] = deltas[i]
                    .checked_sub(previous)
                    .filter(|d| *d >= 0)
                    .ok_or_else(|| {
                        Error::CorruptMeta(format!(
                            "HNSW level {level} nodes are not in ascending order"
                        ))
                    })?;
            }
            for n in deltas {
                meta.write_vint(n);
            }
        } else if nodes.len() as i32 != field.count {
            return corrupt(format!(
                "Level 0 expects to have all nodes: {} != {}",
                nodes.len(),
                field.count
            ));
        }
    }

    let start = index.len() as i64;
    meta.write_i64(start);
    meta.write_vint(DIRECT_MONOTONIC_BLOCK_SHIFT as i32);
    let mut cumulative = Vec::with_capacity(value_count as usize);
    let mut sum = 0i64;
    for level_offsets in level_node_offsets {
        for &v in level_offsets {
            cumulative.push(sum);
            // ARITH: every `v` is a byte length `write_graph` measured inside
            // `index`, and they partition it, so the running sum is at most
            // `index.len()`.
            #[allow(clippy::arithmetic_side_effects)]
            {
                sum += v as i64;
            }
        }
    }
    let (dm_meta, dm_data) = direct_monotonic::write(&cumulative, DIRECT_MONOTONIC_BLOCK_SHIFT);
    meta.extend_from_slice(&dm_meta);
    index.extend_from_slice(&dm_data);
    // ARITH: `start` was `index.len()` before the append above, and `index`
    // only grows.
    #[allow(clippy::arithmetic_side_effects)]
    meta.write_i64(index.len() as i64 - start);
    Ok(())
}

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

/// One field's `.vem` entry: Java's `Lucene99HnswVectorsReader.FieldEntry`.
#[derive(Debug, Clone)]
pub struct HnswFieldEntry {
    pub field_number: i32,
    pub encoding: VectorEncoding,
    pub similarity: VectorSimilarityFunction,
    pub vector_index_offset: i64,
    pub vector_index_length: i64,
    pub m: i32,
    pub num_levels: i32,
    pub dimension: i32,
    pub size: i32,
    /// `nodes_by_level[0]` is always empty (level 0 is `0..size` implicitly).
    /// Shared with every [`OffHeapHnswGraph`] opened over this field, which a
    /// kNN query does once per segment: a refcount bump, not a copy of every
    /// upper-level node list.
    pub nodes_by_level: Arc<[Vec<i32>]>,
    offsets_meta: Option<direct_monotonic::Meta>,
    offsets_offset: i64,
    offsets_length: i64,
}

/// Port of `Lucene99HnswVectorsReader`.
#[derive(Debug, Clone)]
pub struct HnswVectorsReader<'a> {
    index: &'a [u8],
    version: i32,
    fields: std::sync::Arc<Vec<HnswFieldEntry>>,
}

impl<'a> HnswVectorsReader<'a> {
    /// Port of the `Lucene99HnswVectorsReader` constructor + `readFields`.
    pub fn open(
        meta_buf: &[u8],
        index_buf: &'a [u8],
        segment_id: &[u8; ID_LENGTH],
        segment_suffix: &str,
    ) -> Result<Self> {
        let mut meta = SliceInput::new(meta_buf);
        let version_meta = codec_util::check_index_header(
            &mut meta,
            META_CODEC,
            VERSION_START,
            VERSION_CURRENT,
            segment_id,
            segment_suffix,
        )?;
        let mut index_in = SliceInput::new(index_buf);
        let version_index = codec_util::check_index_header(
            &mut index_in,
            INDEX_CODEC,
            VERSION_START,
            VERSION_CURRENT,
            segment_id,
            segment_suffix,
        )?;
        let (version_meta, version_index) = (version_meta.version, version_index.version);
        if version_meta != version_index {
            return corrupt(format!(
                "Format versions mismatch: meta={version_meta}, {INDEX_CODEC}={version_index}"
            ));
        }
        // Checked rather than proved: `check_index_header` above happens to
        // guarantee both files are longer than a footer, but that is an
        // argument about a header this function does not own, and the
        // underflow it would license (`0usize - 16`) is a slice panic.
        let meta_end = meta_buf
            .len()
            .checked_sub(codec_util::FOOTER_LENGTH)
            .ok_or_else(|| Error::CorruptMeta("`.vem` is shorter than its footer".into()))?;
        codec_util::check_whole_file_footer(meta_buf, meta_end)?;
        let index_end = index_buf
            .len()
            .checked_sub(codec_util::FOOTER_LENGTH)
            .ok_or_else(|| Error::CorruptMeta("`.vex` is shorter than its footer".into()))?;
        codec_util::check_whole_file_footer(index_buf, index_end)?;

        let mut fields = Vec::new();
        loop {
            let field_number = meta.read_i32()?;
            if field_number == -1 {
                break;
            }
            if field_number < 0 {
                return corrupt(format!("Invalid field number: {field_number}"));
            }
            fields.push(read_field_entry(&mut meta, field_number, index_buf.len())?);
        }

        Ok(HnswVectorsReader {
            index: index_buf,
            version: version_meta,
            fields: std::sync::Arc::new(fields),
        })
    }

    /// A reader over entries another format's `.vem` described: the retired
    /// `Lucene95HnswVectorsFormat` wrote its graph exactly as
    /// `Lucene99HnswVectorsFormat` version 0 does (plain-vint neighbour
    /// deltas, `DirectMonotonic` node offsets), only framed in a different
    /// metadata file.
    pub(crate) fn from_entries(index: &'a [u8], version: i32, fields: Vec<HnswFieldEntry>) -> Self {
        HnswVectorsReader {
            index,
            version,
            fields: std::sync::Arc::new(fields),
        }
    }

    pub fn version(&self) -> i32 {
        self.version
    }

    pub fn fields(&self) -> &[HnswFieldEntry] {
        &self.fields
    }

    /// This reader without its data file, to keep beside the segment's
    /// files: [`Self::attach`] it to the same data for each search, so the
    /// metadata is parsed and checked once and shared, not copied.
    pub fn detached(&self) -> HnswVectorsReader<'static> {
        HnswVectorsReader {
            index: &[],
            version: self.version,
            fields: self.fields.clone(),
        }
    }

    /// A detached reader ([`Self::detached`]) over `data`, which must be the
    /// file the reader was opened on.
    pub fn attach<'b>(&self, data: &'b [u8]) -> HnswVectorsReader<'b> {
        HnswVectorsReader {
            index: data,
            version: self.version,
            fields: self.fields.clone(),
        }
    }

    pub fn field(&self, field_number: i32) -> Option<&HnswFieldEntry> {
        self.fields.iter().find(|f| f.field_number == field_number)
    }

    /// `getVectorCount(fieldInfo)` (Lucene 10.6, `branch_10_6`, #16647): the
    /// field's vector count as its metadata entry holds it, without opening
    /// its vector values.
    /// `Lucene99HnswVectorsReader`'s: `getFieldEntryOrThrow(name).size()`.
    ///
    /// # Errors
    /// [`Error::UnknownField`] for a field this reader does not hold.
    pub fn vector_count(&self, field_number: i32) -> Result<i32> {
        self.field(field_number)
            .map(|f| f.size)
            .ok_or(Error::UnknownField(field_number))
    }

    /// Port of `Lucene99HnswVectorsReader.getGraph(FieldEntry)`. `None` is
    /// Java's `HnswGraph.EMPTY`: the field exists but carries no graph, so a
    /// search must be exhaustive.
    pub fn graph(&self, field_number: i32) -> Result<Option<OffHeapHnswGraph<'a>>> {
        let entry = self
            .field(field_number)
            .ok_or(Error::UnknownField(field_number))?;
        if entry.vector_index_length == 0 {
            return Ok(None);
        }
        OffHeapHnswGraph::new(entry, self.index, self.version).map(Some)
    }
}

fn read_field_entry(
    meta: &mut SliceInput<'_>,
    field_number: i32,
    index_len: usize,
) -> Result<HnswFieldEntry> {
    let encoding = read_vector_encoding(meta)?;
    let similarity = read_similarity_function(meta)?;
    let vector_index_offset = meta.read_vlong()?;
    let vector_index_length = meta.read_vlong()?;
    let dimension = meta.read_vint()?;
    let size = meta.read_i32()?;
    read_graph_meta(
        meta,
        GraphMetaHead {
            field_number,
            encoding,
            similarity,
            vector_index_offset,
            vector_index_length,
            dimension,
            size,
        },
        index_len,
    )
}

/// What a `.vem` field entry says before its graph parameters: shared by
/// `Lucene99HnswVectorsReader` and the retired `Lucene95HnswVectorsReader`,
/// whose entries differ only in what sits between this head and `M` (the
/// latter keeps its vectors' own offsets and `ordToDoc` there).
pub(crate) struct GraphMetaHead {
    pub field_number: i32,
    pub encoding: VectorEncoding,
    pub similarity: VectorSimilarityFunction,
    pub vector_index_offset: i64,
    pub vector_index_length: i64,
    pub dimension: i32,
    pub size: i32,
}

/// The rest of a `Lucene99HnswVectorsReader.FieldEntry` (and of
/// `Lucene95HnswVectorsReader.FieldEntry`, whose tail is the same bytes): `M`,
/// the level count, the upper levels' node lists and the node-offsets
/// `DirectMonotonic` meta.
pub(crate) fn read_graph_meta(
    meta: &mut SliceInput<'_>,
    head: GraphMetaHead,
    index_len: usize,
) -> Result<HnswFieldEntry> {
    let GraphMetaHead {
        field_number,
        encoding,
        similarity,
        vector_index_offset,
        vector_index_length,
        dimension,
        size,
    } = head;
    let m = meta.read_vint()?;
    let num_levels = meta.read_vint()?;
    if dimension <= 0 {
        return corrupt(format!("illegal vector dimension {dimension}"));
    }
    if size < 0 {
        return corrupt(format!("illegal vector count {size}"));
    }
    if m <= 0 || m > crate::hnsw::MAXIMUM_MAX_CONN {
        return corrupt(format!("illegal maxConn {m}"));
    }
    if num_levels < 0 {
        return corrupt(format!("illegal level count {num_levels}"));
    }
    // ARITH: the `||` short-circuits, so the sum is only formed once both
    // operands are known non-negative; each then widens to at most `2^63 - 1`
    // and their `u128` sum cannot approach `u128::MAX`. Widening rather than
    // `offset > index_len - length` on purpose: the latter is the shape that
    // forms the very sum it guards.
    #[allow(clippy::arithmetic_side_effects)]
    let index_region_past_end = vector_index_offset < 0
        || vector_index_length < 0
        || vector_index_offset as u128 + vector_index_length as u128 > index_len as u128;
    if index_region_past_end {
        return corrupt(format!(
            "graph region [{vector_index_offset}, +{vector_index_length}) past the end of a \
             {index_len} byte .vex file"
        ));
    }

    // Java pre-allocates `new int[numLevels][]` from the vint it just read.
    // Here that is `vec![Vec::new(); 2^31 - 1]` -- 51 GB of empty vectors, an
    // **abort** rather than an exception -- for a `.vem` whose level count byte
    // flipped. Grown one level at a time instead: every level above 0 reads at
    // least two bytes below, so the file itself bounds the count.
    let mut nodes_by_level: Vec<Vec<i32>> = Vec::new();
    let mut number_of_offsets = 0i64;
    for level in 0..num_levels {
        if level > 0 {
            let num_nodes_on_level = meta.read_vint()?;
            if num_nodes_on_level <= 0 || num_nodes_on_level > size {
                return corrupt(format!(
                    "illegal node count {num_nodes_on_level} on level {level} (size {size})"
                ));
            }
            // `size` is an unvalidated `.vem` int, so `num_nodes_on_level`
            // alone admits an 8.6 GB reservation. Every node on the level is
            // written as a vint, so it costs at least one further byte of
            // `.vem`: a real file always has that many left.
            if num_nodes_on_level as usize > meta.remaining() {
                return corrupt(format!(
                    "level {level} claims {num_nodes_on_level} nodes, more than the \
                     {} bytes left in the .vem",
                    meta.remaining()
                ));
            }
            let Some(offsets) = number_of_offsets.checked_add(i64::from(num_nodes_on_level)) else {
                return corrupt("HNSW node count overflows the node-offsets sequence");
            };
            number_of_offsets = offsets;
            let mut nodes = Vec::with_capacity(num_nodes_on_level as usize);
            nodes.push(meta.read_vint()?);
            for i in 1..num_nodes_on_level as usize {
                let delta = meta.read_vint()?;
                if delta < 0 {
                    return corrupt("negative node delta on an upper HNSW level");
                }
                // Java's `nodesByLevel[level][i - 1] + readVInt()` is an `int`
                // add that wraps; wrapping here would break the monotonicity
                // the single `nodes.last() < size` check below relies on.
                // ARITH: `i` comes from `1..num_nodes_on_level`, so `i - 1`
                // is in range.
                #[allow(clippy::arithmetic_side_effects)]
                let previous = nodes[i - 1];
                let Some(node) = previous.checked_add(delta) else {
                    return corrupt("HNSW level node ordinals overflow");
                };
                nodes.push(node);
            }
            // Non-negative deltas make the sequence non-decreasing, so the
            // last entry is the largest and these two checks bound all of them.
            if nodes.last().is_some_and(|&n| n >= size) || nodes[0] < 0 {
                return corrupt("HNSW level node ordinal out of range");
            }
            nodes_by_level.push(nodes);
        } else {
            let Some(offsets) = number_of_offsets.checked_add(i64::from(size)) else {
                return corrupt("HNSW node count overflows the node-offsets sequence");
            };
            number_of_offsets = offsets;
            // Level 0 is `0..size` implicitly; the slot exists so upper levels
            // keep their index.
            nodes_by_level.push(Vec::new());
        }
    }

    let (offsets_meta, offsets_offset, offsets_length) = if number_of_offsets > 0 {
        let offsets_offset = meta.read_i64()?;
        let block_shift = meta.read_vint()?;
        if !(0..=31).contains(&block_shift) {
            return corrupt(format!("illegal DirectMonotonic block shift {block_shift}"));
        }
        let m = direct_monotonic::load_meta(meta, number_of_offsets, block_shift as u32)?;
        let offsets_length = meta.read_i64()?;
        // ARITH: same shape and same proof as the vector-index region above.
        #[allow(clippy::arithmetic_side_effects)]
        let offsets_region_past_end = offsets_offset < 0
            || offsets_length < 0
            || offsets_offset as u128 + offsets_length as u128 > index_len as u128;
        if offsets_region_past_end {
            return corrupt("graph node-offsets region past the end of the .vex file");
        }
        (Some(m), offsets_offset, offsets_length)
    } else {
        (None, 0, 0)
    };

    Ok(HnswFieldEntry {
        field_number,
        encoding,
        similarity,
        vector_index_offset,
        vector_index_length,
        m,
        num_levels,
        dimension,
        size,
        nodes_by_level: nodes_by_level.into(),
        offsets_meta,
        offsets_offset,
        offsets_length,
    })
}

/// Port of `Lucene99HnswVectorsReader.OffHeapHnswGraph`: reads a node's
/// neighbours straight out of the mapped `.vex` bytes.
#[derive(Debug, Clone)]
pub struct OffHeapHnswGraph<'a> {
    /// Just this field's graph region.
    data: &'a [u8],
    /// The node-offsets region (a `DirectMonotonicReader` payload).
    offsets_data: &'a [u8],
    offsets_meta: direct_monotonic::Meta,
    nodes_by_level: Arc<[Vec<i32>]>,
    /// `graphLevelNodeIndexOffsets`: how many nodes precede each level in the
    /// flat node-offsets sequence.
    level_index_offsets: Vec<i64>,
    num_levels: i32,
    entry_node: i32,
    size: i32,
    max_conn: i32,
    version: i32,
    /// Java's `currentNeighborsBuffer`, an `int[M * 2]` allocated once for the
    /// graph's lifetime. `RefCell` rather than `&mut self` because
    /// [`HnswGraphView::neighbors_into`] takes `&self` deliberately -- the
    /// builder reads the graph while holding a mutable borrow of a neighbour
    /// array inside it. Single-threaded, like every searcher here.
    scratch: RefCell<Vec<u64>>,
}

impl<'a> OffHeapHnswGraph<'a> {
    fn new(entry: &HnswFieldEntry, index: &'a [u8], version: i32) -> Result<Self> {
        let Some(offsets_meta) = entry.offsets_meta.clone() else {
            return corrupt("graph has data but no node offsets");
        };
        // `read_field_entry` has already established that both regions lie
        // inside `index`, but this constructor takes a `&HnswFieldEntry` and
        // proving it from here would mean trusting a caller's struct. Both
        // slices are taken fallibly instead -- the same cost, once per field.
        let region = |offset: i64, length: i64| -> Option<&'a [u8]> {
            let start = usize::try_from(offset).ok()?;
            let end = start.checked_add(usize::try_from(length).ok()?)?;
            index.get(start..end)
        };
        let data = region(entry.vector_index_offset, entry.vector_index_length)
            .ok_or_else(|| Error::CorruptMeta("HNSW graph region is not inside the .vex".into()))?;
        let offsets_data = region(entry.offsets_offset, entry.offsets_length).ok_or_else(|| {
            Error::CorruptMeta("HNSW node-offsets region is not inside the .vex".into())
        })?;

        let num_levels = usize::try_from(entry.num_levels)
            .map_err(|_| Error::CorruptMeta("negative HNSW level count".into()))?;
        if entry.nodes_by_level.len() != num_levels {
            return corrupt("HNSW level count does not match the per-level node lists");
        }
        let mut level_index_offsets = vec![0i64; num_levels.max(1)];
        for i in 1..num_levels {
            // ARITH: `i` comes from `1..num_levels` and this arm needs
            // `i != 1`, so `i - 1` is in `1..num_levels - 1`.
            #[allow(clippy::arithmetic_side_effects)]
            let node_count = if i == 1 {
                i64::from(entry.size)
            } else {
                entry.nodes_by_level[i - 1].len() as i64
            };
            // ARITH: a sum of `num_levels` counts, each of which is either
            // `size` (a non-negative `i32`) or the length of a level's node
            // list -- and `read_field_entry` bounded every one of those by the
            // `.vem` bytes it had left, so the total is bounded by
            // `i32::MAX + .vem length` and cannot approach `i64::MAX`.
            #[allow(clippy::arithmetic_side_effects)]
            {
                level_index_offsets[i] = level_index_offsets[i - 1] + node_count;
            }
        }
        // ARITH: guarded by `num_levels > 1`.
        #[allow(clippy::arithmetic_side_effects)]
        let entry_node = if num_levels > 1 {
            *entry.nodes_by_level[num_levels - 1]
                .first()
                .ok_or_else(|| Error::CorruptMeta("top HNSW level is empty".into()))?
        } else {
            0
        };
        Ok(OffHeapHnswGraph {
            data,
            offsets_data,
            offsets_meta,
            nodes_by_level: Arc::clone(&entry.nodes_by_level),
            level_index_offsets,
            num_levels: entry.num_levels,
            entry_node,
            size: entry.size,
            max_conn: entry.m,
            version,
            // ARITH: `read_field_entry` rejects any `m` outside
            // `1..=MAXIMUM_MAX_CONN` (512), so this is at most 1024 `u64`s --
            // Java's `int[M * 2] currentNeighborsBuffer`.
            #[allow(clippy::arithmetic_side_effects)]
            scratch: RefCell::new(vec![0u64; (entry.m as usize) * 2]),
        })
    }
}

impl HnswGraphView for OffHeapHnswGraph<'_> {
    fn size(&self) -> i32 {
        self.size
    }

    fn num_levels(&self) -> i32 {
        self.num_levels
    }

    fn entry_node(&self) -> i32 {
        self.entry_node
    }

    fn max_conn(&self) -> i32 {
        self.max_conn
    }

    /// Port of `OffHeapHnswGraph.seek` + the `nextNeighbor` drain.
    fn neighbors_into(&self, level: i32, node: i32, out: &mut Vec<i32>) -> Result<()> {
        out.clear();
        let target_index = if level == 0 {
            // Java `assert`s nothing here: an ordinal past `size` would read
            // the first upper-level node's offset and decode *its* list as
            // this node's. Every caller in this crate passes a checked
            // ordinal, but the method is public.
            if node < 0 || node >= self.size {
                return corrupt(format!("seek level=0 target={node} out of range"));
            }
            node as i64
        } else {
            let nodes = self
                .nodes_by_level
                .get(level as usize)
                .ok_or_else(|| Error::CorruptMeta(format!("no such HNSW level: {level}")))?;
            match nodes.binary_search(&node) {
                Ok(i) => i as i64,
                Err(_) => {
                    return corrupt(format!("seek level={level} target={node} not found"));
                }
            }
        };
        let level_offset = self
            .level_index_offsets
            .get(level as usize)
            .copied()
            .ok_or_else(|| Error::CorruptMeta(format!("no such HNSW level: {level}")))?;
        // ARITH: `target_index` is either a level-0 node ordinal or an index
        // into one level's node list, and `level_offset` is the number of
        // nodes on the levels below it; both were bounded by the `.vem` above,
        // so the sum is far short of `i64::MAX`. An out-of-range sum is not a
        // problem in any case: `direct_monotonic::get` rejects an index past
        // its value count.
        #[allow(clippy::arithmetic_side_effects)]
        let flat_index = target_index + level_offset;
        let offset = direct_monotonic::get(self.offsets_data, &self.offsets_meta, flat_index)?;
        if offset < 0 || offset as usize > self.data.len() {
            return corrupt(format!("HNSW node offset {offset} out of range"));
        }
        let mut input = SliceInput::new(self.data);
        input.seek(offset as usize)?;
        let arc_count = input.read_vint()?;
        // ARITH: `max_conn` is `read_field_entry`'s `m`, rejected outside
        // `1..=MAXIMUM_MAX_CONN` (512), so `max_conn * 2 <= 1024` -- and it is
        // exactly the length of `scratch`, which the slice below indexes.
        #[allow(clippy::arithmetic_side_effects)]
        let max_arcs = self.max_conn * 2;
        if arc_count < 0 || arc_count > max_arcs {
            return corrupt(format!("too many neighbors: {arc_count}"));
        }
        if arc_count == 0 {
            return Ok(());
        }
        let mut scratch = self.scratch.borrow_mut();
        let buf = &mut scratch[..arc_count as usize];
        if self.version >= VERSION_GROUPVARINT {
            input.read_group_vints(buf)?;
        } else {
            for slot in buf.iter_mut() {
                *slot = input.read_vint()? as u32 as u64;
            }
        }
        let mut sum = 0i64;
        out.reserve(arc_count as usize);
        for delta in buf.iter() {
            // ARITH: the loop rejects any `sum` outside `0..size`, so on entry
            // `sum <= i32::MAX`; each delta is a group-varint or a vint, i.e.
            // at most `u32::MAX`, so the sum stays under 2^33.
            #[allow(clippy::arithmetic_side_effects)]
            {
                sum += *delta as i64;
            }
            if sum < 0 || sum >= self.size as i64 {
                return corrupt(format!("HNSW neighbor ordinal {sum} out of range"));
            }
            out.push(sum as i32);
        }
        Ok(())
    }

    fn prefetch_neighbors(&self, level: i32, node: i32) {
        // Level 0 only, where the walk spends its time; the node-offsets
        // table is small enough to stay cached, the lists are not.
        if level != 0 || node < 0 || node >= self.size {
            return;
        }
        if let Ok(offset) =
            direct_monotonic::get(self.offsets_data, &self.offsets_meta, i64::from(node))
        {
            if let Some(list) = usize::try_from(offset)
                .ok()
                .and_then(|start| self.data.get(start..))
            {
                // A vint count and up to `2 * M` group-varint deltas,
                // budgeted at 3 bytes each; a longer list's tail just misses.
                let len = (self.max_conn.max(0) as usize)
                    .saturating_mul(6)
                    .min(list.len());
                lucene_util::simd::prefetch_bytes(&list[..len]);
            }
        }
    }

    fn sorted_nodes_on_level(&self, level: i32) -> Result<Vec<i32>> {
        if level == 0 {
            return Ok((0..self.size).collect());
        }
        self.nodes_by_level
            .get(level as usize)
            .cloned()
            .ok_or_else(|| Error::CorruptMeta(format!("no such HNSW level: {level}")))
    }
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

/// The parameters `Lucene99HnswVectorsReader.search` carries besides the
/// scorer, the graph and `k` -- the ones a plain top-`k` search leaves at
/// their defaults and a *filtered* or *seeded* one does not.
///
/// One struct rather than three more positional arguments because all three
/// are optional and all three mean "narrow the walk"; `SearchOptions::default()`
/// is exactly Java's unfiltered, unseeded `KnnFloatVectorQuery`.
#[derive(Debug, Clone, Copy, Default)]
pub struct SearchOptions<'a> {
    /// Java's `acceptOrds` (`KnnVectorValues.getAcceptOrds(acceptDocs)`), in
    /// **ordinal** space: the ordinals a hit may come from. `None` accepts
    /// everything, which is Java's `null` `Bits` and the fastest walk.
    ///
    /// It gates *collection*, not *traversal*: the walk still crosses a
    /// rejected node's arcs, exactly as `HnswGraphSearcher.searchLevel` does.
    pub accept_ords: Option<&'a FixedBitSet>,
    /// Java's `filteredDocCount = min(acceptDocs.cost(), graphSize)`, i.e.
    /// the *exact* number of documents that pass the filter
    /// (`AcceptDocs.cost()` materialises the bitset and returns
    /// `BitSet.cardinality()`; it is not an estimate). It decides the graph
    /// walk against the exhaustive scan: a walk that would visit more nodes
    /// than there are acceptable documents is not worth its overhead.
    ///
    /// `None` means "every vector passes", which is the unfiltered case:
    /// Java gets `maxDoc` from `BitsAcceptDocs.cost()` there and then takes
    /// `min(maxDoc, graphSize)`, and `graphSize <= maxDoc` always, so `None`
    /// and `Some(max_doc)` are the same number. Note that this holds on a
    /// **deleted** segment too, which looks wrong and is not:
    /// `BitsAcceptDocs` uses `bitSet.cardinality()` only
    /// `if (bits instanceof BitSet)`, and `Lucene90LiveDocsFormat.readLiveDocs`
    /// returns a `DenseLiveDocs`/`SparseLiveDocs`, which implement
    /// `LiveDocs extends Bits` and are not `BitSet`s. So deletions alone
    /// never drop a segment to the exhaustive scan.
    pub filtered_doc_count: Option<i32>,
    /// Java's `KnnSearchStrategy.Seeded` entry points, already resolved to
    /// level-0 ordinals: level 0's beam starts here instead of at the end of
    /// `findBestEntryPoint`'s descent. See
    /// [`crate::hnsw::HnswGraphSearcher::search_seeded`].
    ///
    /// Ignored when the search takes the exhaustive branch, as Java's is
    /// (nothing consults the strategy there).
    pub seed_ords: Option<&'a [i32]>,
    /// `KnnSearchStrategy.Hnsw.filteredSearchThreshold`: with an accept set
    /// passing fewer than this percentage of the graph, level 0 is walked by
    /// [`crate::hnsw::HnswGraphSearcher::search_filtered`]
    /// (`FilteredHnswGraphSearcher`). `0` -- Lucene 10.5.0's
    /// `DEFAULT_FILTERED_SEARCH_THRESHOLD` -- never does.
    pub filtered_search_threshold: i32,
}

/// Port of `Lucene99HnswVectorsReader.search(FieldEntry, KnnCollector,
/// AcceptDocs, ...)`, including its choice between a graph walk and an
/// exhaustive scan.
///
/// `graph` is `None` for a field written without one (a "tiny segment"), and
/// the scan is then the only option -- which is also what Java does when `k`
/// is large enough that a graph walk would visit more nodes than there are
/// vectors.
///
/// Returns the `(ordinal, score)` pairs, best first, and whether the
/// collector early-terminated (Java's
/// `TotalHits.Relation.GREATER_THAN_OR_EQUAL_TO`, which
/// `AbstractKnnVectorQuery.getLeafResults` reads to decide whether to fall
/// back to an exact search). Translating ordinals to doc ids is the caller's
/// job (Java's `OrdinalTranslatedKnnCollector`); use
/// [`crate::vectors::FloatVectorValues::ord_to_doc`].
pub fn search<G: HnswGraphView, S: VectorScorer>(
    scorer: &mut S,
    graph: Option<&G>,
    k: usize,
    visit_limit: u64,
    options: SearchOptions<'_>,
) -> Result<(Vec<(i32, f32)>, bool)> {
    let num_vectors = scorer.max_ord();
    let accept_ords = options.accept_ords;
    // Java's `Bits` is unbounded-by-construction (`getAcceptOrds` wraps the
    // doc-space bits behind `ordToDoc`), so it has nothing to check here. A
    // `FixedBitSet` does: `get` past the end is an out-of-bounds index, i.e.
    // a panic on a caller mistake rather than an error, and the accept set is
    // the one argument a caller derives from something outside this segment.
    //
    // The bound is the larger of the field's vector count and the graph's own
    // ordinal space, not just the former: the exhaustive branch asks about
    // `0..maxOrd`, but the **graph** branch asks about whatever ordinals the
    // arcs name, bounded by `maxNodeId()` -- and nothing cross-checks the
    // `.vem`'s node count against the `.vec`'s vector count, so a corrupt meta
    // or a graph handed in for the wrong field would index past the end of an
    // otherwise correctly-sized accept set. Same bound
    // [`crate::hnsw::HnswGraphSearcher::search_seeded`] applies to its entry
    // points, for the same reason.
    if let Some(bits) = accept_ords {
        let graph_bound = graph.map_or(0i64, |g| i64::from(g.max_node_id()).saturating_add(1));
        let needed = i64::from(num_vectors).max(graph_bound).max(0);
        if (bits.len() as i64) < needed {
            return Err(Error::InvalidGraphParameter(format!(
                "the accept-ordinal set covers {} ordinals, short of the {needed} this field's \
                 {num_vectors} vectors and its graph between them can name",
                bits.len()
            )));
        }
    }
    if let Some(count) = options.filtered_doc_count {
        if count < 0 {
            return Err(Error::InvalidGraphParameter(format!(
                "filtered_doc_count {count} is negative (it is a cardinality)"
            )));
        }
    }
    if num_vectors == 0 || k == 0 {
        return Ok((Vec::new(), false));
    }
    let mut collector = KnnCollector::new(k, visit_limit);
    search_with(scorer, graph, &mut collector, options)?;
    let early = collector.early_terminated();
    Ok((collector.top_docs(), early))
}

/// [`search`] into any [`KnnCollect`]or: `Lucene99HnswVectorsReader.search`'s
/// graph-walk-or-scan choice made with `collector.k()`, the results left in
/// the collector. The argument checks are [`search`]'s; this one assumes a
/// caller that made them (or passes no accept set).
pub fn search_with<G: HnswGraphView, S: VectorScorer, C: KnnCollect + ?Sized>(
    scorer: &mut S,
    graph: Option<&G>,
    collector: &mut C,
    options: SearchOptions<'_>,
) -> Result<()> {
    let num_vectors = scorer.max_ord();
    let accept_ords = options.accept_ords;
    let k = collector.k();
    if num_vectors == 0 || k == 0 {
        return Ok(());
    }
    let graph_size = graph.map_or(0, |g| g.size());
    let mut do_hnsw = k < num_vectors as usize;
    // `int filteredDocCount = Math.min(acceptDocs.cost(), graphSize);`
    let filtered_doc_count = options
        .filtered_doc_count
        .unwrap_or(graph_size)
        .min(graph_size);
    let unfiltered_visit = expected_visited_nodes(k as i32, graph_size);
    if unfiltered_visit >= filtered_doc_count || graph_size == 0 {
        do_hnsw = false;
    }

    if do_hnsw {
        let graph = graph.expect("graph_size > 0 implies a graph");
        let mut searcher = HnswGraphSearcher::new(k, graph.size());
        // `HnswGraphSearcher.search`: the filtered searcher when the
        // strategy's threshold exceeds the share of the graph the filter
        // passes (`useFilteredSearch`: `ratio * 100 < threshold`).
        let filtered = match accept_ords {
            Some(bits)
                if graph.max_conn() != -1
                    && filtered_doc_count > 0
                    && (filtered_doc_count as f32 / graph.size() as f32) * 100.0
                        < options.filtered_search_threshold as f32 =>
            {
                Some(bits)
            }
            _ => None,
        };
        match (filtered, options.seed_ords) {
            (Some(bits), seeds) => searcher.search_filtered(
                collector,
                scorer,
                graph,
                bits,
                filtered_doc_count,
                seeds.filter(|s| !s.is_empty()),
            )?,
            (None, Some(seeds)) => {
                searcher.search_seeded(collector, scorer, graph, accept_ords, seeds)?
            }
            (None, None) => searcher.search(collector, scorer, graph, accept_ords)?,
        }
    } else {
        // Java's bulk-scored exhaustive branch, 64 ordinals at a time.
        let mut ords = [0i32; EXHAUSTIVE_BULK_SCORE_ORDS];
        let mut scores = [0.0f32; EXHAUSTIVE_BULK_SCORE_ORDS];
        let mut num_ords = 0usize;
        for i in 0..num_vectors {
            // `if (acceptedOrds == null || acceptedOrds.get(i))`: the
            // early-termination check is *inside* it, so a run of rejected
            // ordinals cannot end the scan.
            if accept_ords.is_some_and(|b| !b.get(i as usize)) {
                continue;
            }
            if collector.early_terminated() {
                break;
            }
            ords[num_ords] = i;
            // ARITH: reset to 0 the moment it reaches
            // `EXHAUSTIVE_BULK_SCORE_ORDS` (64), so it never leaves `0..=64`.
            #[allow(clippy::arithmetic_side_effects)]
            {
                num_ords += 1;
            }
            if num_ords == EXHAUSTIVE_BULK_SCORE_ORDS {
                flush_bulk(collector, scorer, &ords, &mut scores, num_ords)?;
                num_ords = 0;
            }
        }
        if num_ords > 0 {
            flush_bulk(collector, scorer, &ords, &mut scores, num_ords)?;
        }
    }
    Ok(())
}

fn flush_bulk<S: VectorScorer, C: KnnCollect + ?Sized>(
    collector: &mut C,
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

#[cfg(test)]
mod tests;
