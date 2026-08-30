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

use lucene_store::codec_util::{self, ID_LENGTH};
use lucene_store::data_input::{DataInput, SliceInput};
use lucene_store::data_output::DataOutput;
use lucene_util::fixed_bit_set::FixedBitSet;

use crate::direct_monotonic;
use crate::field_infos::{VectorEncoding, VectorSimilarityFunction};
use crate::hnsw::{
    self, expected_visited_nodes, HnswGraphSearcher, HnswGraphView, KnnCollector, OnHeapHnswGraph,
    UpdateableVectorScorer, VectorScorer,
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
    let mut usable: Vec<usize> = Vec::new();
    let mut largest: Option<usize> = None;
    let mut largest_graph_size = -1i32;
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
        // Java compares the candidate's *live* count against the incumbent's
        // *total* size, not against its live count. Kept as-is: this only ever
        // picks a base graph, and at equal size a graph with deletions is the
        // worse base, which is what the asymmetry expresses.
        if delete_pct <= i64::from(DELETE_PCT_THRESHOLD)
            && (largest.is_none() || live > i64::from(largest_graph_size))
        {
            largest = Some(i);
            largest_graph_size = graph_size;
        }
        if live == i64::from(graph_size) {
            usable.push(i);
        }
    }

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
    let mut new_doc_to_old_ord: Vec<std::collections::HashMap<i32, i32>> =
        Vec::with_capacity(order.len());
    let mut old_to_new: Vec<Vec<i32>> = Vec::with_capacity(order.len());
    for i in order {
        let source = &sources[*i];
        let mut map = std::collections::HashMap::with_capacity(source.ord_to_doc.len());
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
    pub nodes_by_level: Vec<Vec<i32>>,
    offsets_meta: Option<direct_monotonic::Meta>,
    offsets_offset: i64,
    offsets_length: i64,
}

/// Port of `Lucene99HnswVectorsReader`.
#[derive(Debug, Clone)]
pub struct HnswVectorsReader<'a> {
    index: &'a [u8],
    version: i32,
    fields: Vec<HnswFieldEntry>,
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
            fields,
        })
    }

    pub fn version(&self) -> i32 {
        self.version
    }

    pub fn fields(&self) -> &[HnswFieldEntry] {
        &self.fields
    }

    pub fn field(&self, field_number: i32) -> Option<&HnswFieldEntry> {
        self.fields.iter().find(|f| f.field_number == field_number)
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
        nodes_by_level,
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
    nodes_by_level: Vec<Vec<i32>>,
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
            nodes_by_level: entry.nodes_by_level.clone(),
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
        match options.seed_ords {
            Some(seeds) => {
                searcher.search_seeded(&mut collector, scorer, graph, accept_ords, seeds)?
            }
            None => searcher.search(&mut collector, scorer, graph, accept_ords)?,
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
                flush_bulk(&mut collector, scorer, &ords, &mut scores, num_ords)?;
                num_ords = 0;
            }
        }
        if num_ords > 0 {
            flush_bulk(&mut collector, scorer, &ords, &mut scores, num_ords)?;
        }
    }
    let early = collector.early_terminated();
    Ok((collector.top_docs(), early))
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

#[cfg(test)]
mod tests {
    // The arithmetic gate is about values read off disk; a test's `i + 1` is
    // not one. See docs/arithmetic-gate.md.
    #![allow(clippy::arithmetic_side_effects)]
    use super::*;
    use crate::hnsw::{HnswGraphBuilder, KnnCollector, UpdateableVectorScorer};

    const ID: [u8; ID_LENGTH] = *b"hnswvectorstest1";

    /// A one-dimensional scorer, so "nearest" is readable off the page.
    #[derive(Debug, Clone)]
    struct Line {
        values: Vec<f32>,
        query: f32,
    }

    impl VectorScorer for Line {
        fn score(&mut self, node: i32) -> Result<f32> {
            let d = self.values[node as usize] - self.query;
            Ok(1.0 / (1.0 + d * d))
        }

        fn max_ord(&self) -> i32 {
            self.values.len() as i32
        }
    }

    impl UpdateableVectorScorer for Line {
        fn set_scoring_ordinal(&mut self, ord: i32) -> Result<()> {
            self.query = self.values[ord as usize];
            Ok(())
        }
    }

    /// The unfiltered, unseeded defaults -- Java's plain
    /// `KnnFloatVectorQuery` arguments.
    fn opts<'a>() -> SearchOptions<'a> {
        SearchOptions::default()
    }

    fn line(n: usize) -> Line {
        Line {
            values: (0..n).map(|i| i as f32).collect(),
            query: 0.0,
        }
    }

    fn build(n: usize) -> OnHeapHnswGraph {
        HnswGraphBuilder::new(line(n), 8, 32, crate::hnsw::DEFAULT_RAND_SEED)
            .unwrap()
            .build(n as i32)
            .unwrap()
    }

    fn field<'g>(graph: Option<&'g OnHeapHnswGraph>, count: i32) -> HnswVectorsField<'g> {
        HnswVectorsField {
            field_number: 0,
            encoding: VectorEncoding::Float32,
            similarity: VectorSimilarityFunction::Euclidean,
            dimension: 1,
            count,
            graph,
            m: 8,
        }
    }

    fn repack(buf: &[u8], patch: impl FnOnce(&mut Vec<u8>)) -> Vec<u8> {
        let mut body = buf[..buf.len() - codec_util::FOOTER_LENGTH].to_vec();
        patch(&mut body);
        codec_util::write_footer(&mut body);
        body
    }

    fn open(meta: &[u8], index: &[u8]) -> Result<HnswVectorsReader<'static>> {
        let index: &'static [u8] = Box::leak(index.to_vec().into_boxed_slice());
        HnswVectorsReader::open(meta, index, &ID, "")
    }

    #[test]
    fn a_graph_round_trips_arc_for_arc() {
        let graph = build(300);
        assert!(graph.num_levels() >= 2);
        let (vex, vem) = write_hnsw_vectors(&[field(Some(&graph), 300)], &ID, "").unwrap();
        let reader = HnswVectorsReader::open(&vem, &vex, &ID, "").unwrap();
        assert_eq!(reader.version(), VERSION_CURRENT);
        assert_eq!(reader.fields().len(), 1);
        assert!(reader.field(9).is_none());
        let entry = reader.field(0).unwrap();
        assert_eq!(entry.dimension, 1);
        assert_eq!(entry.size, 300);
        assert_eq!(entry.m, 8);
        assert_eq!(entry.encoding, VectorEncoding::Float32);
        assert_eq!(entry.similarity, VectorSimilarityFunction::Euclidean);

        let off = reader.graph(0).unwrap().unwrap();
        assert_eq!(off.size(), graph.size());
        assert_eq!(off.num_levels(), graph.num_levels());
        assert_eq!(off.entry_node(), graph.entry_node());
        assert_eq!(off.max_conn(), graph.max_conn());
        let mut want = Vec::new();
        let mut got = Vec::new();
        for level in 0..graph.num_levels() {
            assert_eq!(
                off.sorted_nodes_on_level(level).unwrap(),
                graph.sorted_nodes_on_level(level).unwrap()
            );
            for node in graph.sorted_nodes_on_level(level).unwrap() {
                graph.neighbors_into(level, node, &mut want).unwrap();
                want.sort_unstable();
                want.dedup();
                off.neighbors_into(level, node, &mut got).unwrap();
                assert_eq!(got, want, "level {level} node {node}");
            }
        }
    }

    /// A field Lucene decided was too small for a graph: `numLevels == 0`, a
    /// zero-length `.vex` region, and a reader that must not go looking for
    /// node offsets that were never written.
    #[test]
    fn a_field_without_a_graph_round_trips() {
        let (vex, vem) = write_hnsw_vectors(&[field(None, 5)], &ID, "").unwrap();
        let reader = HnswVectorsReader::open(&vem, &vex, &ID, "").unwrap();
        let entry = reader.field(0).unwrap();
        assert_eq!(entry.num_levels, 0);
        assert_eq!(entry.vector_index_length, 0);
        assert_eq!(entry.m, 8);
        assert!(reader.graph(0).unwrap().is_none());
        assert!(matches!(reader.graph(7), Err(Error::UnknownField(7))));
    }

    #[test]
    fn several_fields_share_one_file_pair() {
        let a = build(200);
        let b = build(150);
        let mut fa = field(Some(&a), 200);
        let mut fb = field(Some(&b), 150);
        fb.field_number = 4;
        fb.encoding = VectorEncoding::Byte;
        fb.similarity = VectorSimilarityFunction::Cosine;
        fa.field_number = 2;
        let (vex, vem) = write_hnsw_vectors(&[fa, fb], &ID, "").unwrap();
        let reader = HnswVectorsReader::open(&vem, &vex, &ID, "").unwrap();
        assert_eq!(reader.fields().len(), 2);
        assert_eq!(reader.field(4).unwrap().encoding, VectorEncoding::Byte);
        let ga = reader.graph(2).unwrap().unwrap();
        let gb = reader.graph(4).unwrap().unwrap();
        assert_eq!(ga.size(), 200);
        assert_eq!(gb.size(), 150);
        let mut out = Vec::new();
        // The second field's region starts partway into `.vex`; a wrong
        // `vectorIndexOffset` decodes the first field's neighbours here.
        gb.neighbors_into(0, 0, &mut out).unwrap();
        let mut want = Vec::new();
        b.neighbors_into(0, 0, &mut want).unwrap();
        want.sort_unstable();
        want.dedup();
        assert_eq!(out, want);
    }

    #[test]
    fn writer_rejects_a_level_zero_that_is_not_every_node() {
        let graph = build(200);
        // A `count` that disagrees with the graph is the writer-side shape of
        // Java's `assert nodesOnLevel.size() == count`.
        assert!(matches!(
            write_hnsw_vectors(&[field(Some(&graph), 199)], &ID, ""),
            Err(Error::CorruptMeta(_))
        ));
    }

    /// The writer must not be able to emit a `.vem` its own reader rejects --
    /// a `M` past `MAXIMUM_MAX_CONN` builds a graph fine and then makes the
    /// segment unopenable by this port *and* by Lucene, whose
    /// `OffHeapHnswGraph` allocates `int[M * 2]` on the value.
    #[test]
    fn writer_rejects_parameters_its_own_reader_would_refuse() {
        let graph = build(200);
        let mut bad_dim = field(Some(&graph), 200);
        bad_dim.dimension = 0;
        assert!(matches!(
            write_hnsw_vectors(&[bad_dim], &ID, ""),
            Err(Error::InvalidGraphParameter(_))
        ));

        let mut bad_count = field(None, -1);
        bad_count.count = -1;
        assert!(matches!(
            write_hnsw_vectors(&[bad_count], &ID, ""),
            Err(Error::InvalidGraphParameter(_))
        ));

        // No graph, so `M` comes from the field rather than the graph.
        let mut bad_m = field(None, 5);
        bad_m.m = crate::hnsw::MAXIMUM_MAX_CONN + 1;
        assert!(matches!(
            write_hnsw_vectors(&[bad_m], &ID, ""),
            Err(Error::InvalidGraphParameter(_))
        ));
        let mut zero_m = field(None, 5);
        zero_m.m = 0;
        assert!(matches!(
            write_hnsw_vectors(&[zero_m], &ID, ""),
            Err(Error::InvalidGraphParameter(_))
        ));
    }

    #[test]
    fn seeking_a_node_that_is_not_on_the_level_is_rejected() {
        let graph = build(300);
        let (vex, vem) = write_hnsw_vectors(&[field(Some(&graph), 300)], &ID, "").unwrap();
        let reader = HnswVectorsReader::open(&vem, &vex, &ID, "").unwrap();
        let off = reader.graph(0).unwrap().unwrap();
        let top = off.num_levels() - 1;
        let on_top = off.sorted_nodes_on_level(top).unwrap();
        let missing = (0..300).find(|n| !on_top.contains(n)).unwrap();
        let mut out = Vec::new();
        assert!(matches!(
            off.neighbors_into(top, missing, &mut out),
            Err(Error::CorruptMeta(_))
        ));
        assert!(matches!(
            off.neighbors_into(top + 5, 0, &mut out),
            Err(Error::CorruptMeta(_))
        ));
        assert!(matches!(
            off.sorted_nodes_on_level(top + 5),
            Err(Error::CorruptMeta(_))
        ));
    }

    // ---------------- search ----------------

    #[test]
    fn search_over_a_graph_finds_the_nearest_neighbours() {
        let graph = build(300);
        let (vex, vem) = write_hnsw_vectors(&[field(Some(&graph), 300)], &ID, "").unwrap();
        let reader = HnswVectorsReader::open(&vem, &vex, &ID, "").unwrap();
        let off = reader.graph(0).unwrap().unwrap();
        let mut scorer = line(300);
        scorer.query = 150.4;
        let (hits, _) = search(&mut scorer, Some(&off), 3, u64::MAX, opts()).unwrap();
        let got: Vec<i32> = hits.into_iter().map(|(n, _)| n).collect();
        assert_eq!(got, vec![150, 151, 149]);
    }

    #[test]
    fn search_falls_back_to_an_exhaustive_scan_without_a_graph() {
        let mut scorer = line(300);
        scorer.query = 7.2;
        // `graph: None` is the tiny-segment case.
        let (hits, _) =
            search::<OnHeapHnswGraph, _>(&mut scorer, None, 3, u64::MAX, opts()).unwrap();
        let got: Vec<i32> = hits.into_iter().map(|(n, _)| n).collect();
        assert_eq!(got, vec![7, 8, 6]);
    }

    /// Java takes the exhaustive branch whenever a graph walk would visit at
    /// least as many nodes as the field holds -- so a large `k` over a small
    /// field never touches the graph, and must still be exact.
    #[test]
    fn a_large_k_takes_the_exhaustive_branch_and_is_exact() {
        let graph = build(120);
        let (vex, vem) = write_hnsw_vectors(&[field(Some(&graph), 120)], &ID, "").unwrap();
        let reader = HnswVectorsReader::open(&vem, &vex, &ID, "").unwrap();
        let off = reader.graph(0).unwrap().unwrap();
        assert!(expected_visited_nodes(60, 120) >= 120);
        let mut scorer = line(120);
        scorer.query = 60.0;
        let (hits, _) = search(&mut scorer, Some(&off), 60, u64::MAX, opts()).unwrap();
        assert_eq!(hits.len(), 60);
        assert_eq!(hits[0].0, 60);
    }

    /// The two arguments a caller derives from something outside this
    /// segment, and the two that would therefore turn a caller mistake into a
    /// panic rather than an error: an accept set shorter than the field
    /// (`FixedBitSet::get` past the end is an out-of-bounds index) and a
    /// negative "cardinality". Java has neither check -- its `Bits` is
    /// unbounded by construction and its `filteredDocCount` is an `assert`.
    #[test]
    fn a_short_accept_set_or_a_negative_filtered_doc_count_is_an_error_not_a_panic() {
        let mut scorer = line(300);
        let short = FixedBitSet::new(299);
        let e = search::<OnHeapHnswGraph, _>(
            &mut scorer,
            None,
            3,
            u64::MAX,
            SearchOptions {
                accept_ords: Some(&short),
                ..opts()
            },
        )
        .unwrap_err();
        assert!(e.to_string().contains("covers 299 ordinals"), "{e}");

        // Exactly the field's size is accepted, which is what every caller in
        // this workspace builds.
        let exact = FixedBitSet::new(300);
        let (hits, _) = search::<OnHeapHnswGraph, _>(
            &mut scorer,
            None,
            3,
            u64::MAX,
            SearchOptions {
                accept_ords: Some(&exact),
                ..opts()
            },
        )
        .unwrap();
        assert!(hits.is_empty(), "an all-zero accept set accepts nothing");

        let e = search::<OnHeapHnswGraph, _>(
            &mut scorer,
            None,
            3,
            u64::MAX,
            SearchOptions {
                filtered_doc_count: Some(-1),
                ..opts()
            },
        )
        .unwrap_err();
        assert!(e.to_string().contains("is negative"), "{e}");

        // And the bound is the *graph's* ordinal space when that is wider
        // than the field's -- the case the exhaustive branch cannot reach but
        // the graph walk can, since the arcs name whatever the `.vem` says.
        let wide = OnHeapHnswGraph::with_size(4, 400);
        let e = search(
            &mut scorer,
            Some(&wide),
            3,
            u64::MAX,
            SearchOptions {
                accept_ords: Some(&exact),
                ..opts()
            },
        )
        .unwrap_err();
        assert!(e.to_string().contains("short of the 400"), "{e}");
    }

    /// `filteredDocCount` is what decides the graph walk against the
    /// exhaustive scan, and it is the *filter's* cardinality, not the field's:
    /// a filter selective enough that a walk would visit more nodes than there
    /// are acceptable documents takes the scan, which is the whole point of
    /// Java's `unfilteredVisit >= filteredDocCount` test. The scan must still
    /// honour the accept set -- and its early-termination check sits *inside*
    /// the accept test, so a run of rejected ordinals cannot end it.
    #[test]
    fn a_selective_filtered_doc_count_drops_the_search_to_an_accept_aware_scan() {
        let graph = build(300);
        let (vex, vem) = write_hnsw_vectors(&[field(Some(&graph), 300)], &ID, "").unwrap();
        let reader = HnswVectorsReader::open(&vem, &vex, &ID, "").unwrap();
        let off = reader.graph(0).unwrap().unwrap();
        let mut accept = FixedBitSet::new(300);
        // Everything from 200 up, so the accepted set starts well past the
        // ordinals a scan would reach first.
        for ord in 200..300 {
            accept.set(ord);
        }
        let mut scorer = line(300);
        scorer.query = 150.4;
        // `expected_visited_nodes(3, 300) == 17`, above a cardinality of 10,
        // so this one takes the scan.
        let mut tiny = FixedBitSet::new(300);
        tiny.set(250);
        tiny.set(251);
        let (hits, _) = search(
            &mut scorer,
            Some(&off),
            3,
            u64::MAX,
            SearchOptions {
                accept_ords: Some(&tiny),
                filtered_doc_count: Some(2),
                ..opts()
            },
        )
        .unwrap();
        assert_eq!(
            hits.iter().map(|(n, _)| *n).collect::<Vec<_>>(),
            vec![250, 251]
        );

        // And with a cardinality large enough to keep the graph walk, the
        // accept set still gates what is collected.
        let (hits, _) = search(
            &mut scorer,
            Some(&off),
            3,
            u64::MAX,
            SearchOptions {
                accept_ords: Some(&accept),
                filtered_doc_count: Some(100),
                ..opts()
            },
        )
        .unwrap();
        assert_eq!(
            hits.iter().map(|(n, _)| *n).collect::<Vec<_>>(),
            vec![200, 201, 202]
        );
    }

    /// Seeding reaches the graph walk and nothing else: with the seeds set,
    /// `findBestEntryPoint`'s descent is skipped, and the exhaustive branch
    /// ignores them exactly as Java's does (nothing there consults the search
    /// strategy).
    #[test]
    fn seed_ordinals_reach_the_graph_walk_and_are_ignored_by_the_scan() {
        let graph = build(300);
        let (vex, vem) = write_hnsw_vectors(&[field(Some(&graph), 300)], &ID, "").unwrap();
        let reader = HnswVectorsReader::open(&vem, &vex, &ID, "").unwrap();
        let off = reader.graph(0).unwrap().unwrap();
        let mut scorer = line(300);
        scorer.query = 150.4;
        let seeded = SearchOptions {
            seed_ords: Some(&[149, 150, 151]),
            ..opts()
        };
        let (hits, _) = search(&mut scorer, Some(&off), 3, u64::MAX, seeded).unwrap();
        assert_eq!(
            hits.iter().map(|(n, _)| *n).collect::<Vec<_>>(),
            vec![150, 151, 149]
        );

        // `k >= numVectors` takes the scan, where the seeds are irrelevant
        // and the answer is exact regardless.
        let (hits, _) = search(&mut scorer, Some(&off), 300, u64::MAX, seeded).unwrap();
        assert_eq!(hits.len(), 300);
        assert_eq!(hits[0].0, 150);
    }

    #[test]
    fn search_of_an_empty_field_or_zero_k_collects_nothing() {
        let mut empty = Line {
            values: Vec::new(),
            query: 0.0,
        };
        assert!(
            search::<OnHeapHnswGraph, _>(&mut empty, None, 5, u64::MAX, opts())
                .unwrap()
                .0
                .is_empty()
        );
        let mut scorer = line(10);
        assert!(
            search::<OnHeapHnswGraph, _>(&mut scorer, None, 0, u64::MAX, opts())
                .unwrap()
                .0
                .is_empty()
        );
    }

    #[test]
    fn the_exhaustive_branch_honours_the_visit_limit() {
        let mut scorer = line(500);
        scorer.query = 400.0;
        // Fewer than one bulk block, so the loop stops mid-batch.
        let (hits, early) = search::<OnHeapHnswGraph, _>(&mut scorer, None, 5, 10, opts()).unwrap();
        assert!(early, "the visit limit is what stopped the scan");
        assert!(hits.len() <= 5);
        // The first `EXHAUSTIVE_BULK_SCORE_ORDS` ordinals are 0..64, none of
        // which is near the query, so an honoured limit shows up as a wrong
        // (but bounded) answer rather than the exact one.
        assert!(hits.iter().all(|(n, _)| *n < 64));
    }

    #[test]
    fn a_collector_reports_the_nodes_it_kept_in_descending_score_order() {
        let graph = build(300);
        let (vex, vem) = write_hnsw_vectors(&[field(Some(&graph), 300)], &ID, "").unwrap();
        let reader = HnswVectorsReader::open(&vem, &vex, &ID, "").unwrap();
        let off = reader.graph(0).unwrap().unwrap();
        let mut scorer = line(300);
        scorer.query = 10.0;
        let mut collector = KnnCollector::new(4, u64::MAX);
        let mut searcher = HnswGraphSearcher::new(4, off.size());
        searcher
            .search(&mut collector, &mut scorer, &off, None)
            .unwrap();
        let hits = collector.top_docs();
        for pair in hits.windows(2) {
            assert!(pair[0].1 >= pair[1].1);
        }
        assert_eq!(hits[0].0, 10);
    }

    // ---------------- corrupt metadata ----------------

    fn valid_pair() -> (Vec<u8>, Vec<u8>) {
        let graph = build(300);
        write_hnsw_vectors(&[field(Some(&graph), 300)], &ID, "").unwrap()
    }

    /// Byte offset of the first field's `dimension` vint in `.vem`, found by
    /// decoding rather than counting.
    fn meta_cursor(meta: &[u8]) -> SliceInput<'_> {
        let mut input = SliceInput::new(meta);
        codec_util::check_index_header(
            &mut input,
            META_CODEC,
            VERSION_START,
            VERSION_CURRENT,
            &ID,
            "",
        )
        .unwrap();
        input.read_i32().unwrap(); // field number
        input.read_i32().unwrap(); // encoding
        input.read_i32().unwrap(); // similarity
        input.read_vlong().unwrap(); // index offset
        input.read_vlong().unwrap(); // index length
        input
    }

    #[test]
    fn an_illegal_dimension_count_or_max_conn_is_rejected() {
        let (vex, vem) = valid_pair();
        let mut cur = meta_cursor(&vem);
        let dim_at = cur.position();
        assert_eq!(vem[dim_at], 1);
        assert!(matches!(
            open(&repack(&vem, |b| b[dim_at] = 0), &vex),
            Err(Error::CorruptMeta(_))
        ));
        cur.read_vint().unwrap();
        let count_at = cur.position();
        assert!(matches!(
            open(
                &repack(&vem, |b| b[count_at..count_at + 4]
                    .copy_from_slice(&(-1i32).to_le_bytes())),
                &vex
            ),
            Err(Error::CorruptMeta(_))
        ));
        cur.read_i32().unwrap();
        let m_at = cur.position();
        assert!(matches!(
            open(&repack(&vem, |b| b[m_at] = 0), &vex),
            Err(Error::CorruptMeta(_))
        ));
        // `MAXIMUM_MAX_CONN` is 512, so a two-byte vint above it is rejected.
        assert!(matches!(
            open(
                &repack(&vem, |b| {
                    b[m_at] = 0x80;
                    b[m_at + 1] = 0x40;
                }),
                &vex
            ),
            Err(Error::CorruptMeta(_))
        ));
    }

    #[test]
    fn an_upper_level_node_count_out_of_range_is_rejected() {
        let (vex, vem) = valid_pair();
        let mut cur = meta_cursor(&vem);
        cur.read_vint().unwrap(); // dimension
        cur.read_i32().unwrap(); // count
        cur.read_vint().unwrap(); // M
        cur.read_vint().unwrap(); // numLevels
        let level1_count_at = cur.position();
        // Zero nodes on an upper level is impossible: the level exists because
        // something is on it.
        assert!(matches!(
            open(&repack(&vem, |b| b[level1_count_at] = 0), &vex),
            Err(Error::CorruptMeta(_))
        ));
    }

    /// The level count is a vint, so a single flipped byte can name
    /// `Integer.MAX_VALUE` levels. Java answers that with
    /// `new int[numLevels][]`, an `OutOfMemoryError`; this port used to answer
    /// it with `vec![Vec::new(); 2^31 - 1]` -- **51 GB, an abort**, which
    /// `catch_unwind` cannot turn into a JVM exception. Reproduce the unfixed
    /// version under `( ulimit -v 4000000; cargo test ... )`.
    #[test]
    fn an_absurd_level_count_is_a_decode_error_not_an_allocation() {
        let (vex, vem) = valid_pair();
        let mut cur = meta_cursor(&vem);
        cur.read_vint().unwrap(); // dimension
        cur.read_i32().unwrap(); // count
        cur.read_vint().unwrap(); // M
        let num_levels_at = cur.position();
        // A five-byte vint for `i32::MAX`, in place of the one-byte real one.
        let mut body = vem[..num_levels_at].to_vec();
        body.write_vint(i32::MAX);
        body.extend_from_slice(&vem[num_levels_at + 1..vem.len() - codec_util::FOOTER_LENGTH]);
        codec_util::write_footer(&mut body);
        assert!(open(&body, &vex).is_err());
    }

    /// The per-level node count is bounded by `size`, which is a raw `.vem`
    /// `int`: together they used to size a `Vec::with_capacity` of up to
    /// `i32::MAX` ordinals -- 8.6 GB, the same abort shape. The file itself is
    /// the bound: one vint per node.
    #[test]
    fn an_upper_level_node_count_larger_than_the_file_is_a_decode_error() {
        let (vex, vem) = valid_pair();
        let mut cur = meta_cursor(&vem);
        cur.read_vint().unwrap(); // dimension
        let count_at = cur.position();
        cur.read_i32().unwrap(); // count
        cur.read_vint().unwrap(); // M
        cur.read_vint().unwrap(); // numLevels
        let level1_count_at = cur.position();
        let mut body = vem[..count_at].to_vec();
        body.write_i32(i32::MAX); // size, so the `<= size` check passes
        body.extend_from_slice(&vem[count_at + 4..level1_count_at]);
        body.write_vint(i32::MAX); // nodes on level 1
        body.extend_from_slice(&vem[level1_count_at + 1..vem.len() - codec_util::FOOTER_LENGTH]);
        codec_util::write_footer(&mut body);
        let err = open(&body, &vex).unwrap_err();
        assert!(
            format!("{err:?}").contains("bytes left in the .vem"),
            "expected the node count to be rejected against the file length, got {err:?}"
        );
    }

    /// Flip bit 0 and bit 7 of every `.vem` and `.vex` body byte, re-sign the
    /// footer so only a semantic invariant can reject the file, and require a
    /// typed error or a clean decode from `open`, `graph`, a full walk of
    /// every node on every level, and a search. Never a panic, never an abort.
    ///
    /// The fixture is the 300-vector, multi-level graph: a single-level or
    /// single-node graph would leave the per-level node lists, the
    /// `DirectMonotonic` node-offset array and the group-varint neighbour
    /// decoder untouched, which is measuring the fixture rather than the
    /// decoder.
    #[test]
    fn every_resigned_single_byte_vem_and_vex_corruption_is_an_error_or_a_clean_decode() {
        let (vex, vem) = valid_pair();
        let graph = build(300);
        assert!(graph.num_levels() >= 2, "fixture must be multi-level");
        let mut flipped = 0usize;
        let mut rejected = 0usize;
        for which in 0..2 {
            let original: &[u8] = if which == 0 { &vem } else { &vex };
            let body_len = original.len() - codec_util::FOOTER_LENGTH;
            for at in 0..body_len {
                for bit in [0u8, 7] {
                    let patched = repack(original, |b| b[at] ^= 1 << bit);
                    let (m, x) = if which == 0 {
                        (patched.clone(), vex.clone())
                    } else {
                        (vem.clone(), patched.clone())
                    };
                    flipped += 1;
                    if walk_everything(&m, &x).is_err() {
                        rejected += 1;
                    }
                }
            }
        }
        assert_eq!(
            flipped,
            2 * (vem.len() + vex.len() - 2 * codec_util::FOOTER_LENGTH)
        );
        assert!(
            rejected > flipped / 4,
            "only {rejected} of {flipped} flips rejected -- suspiciously few"
        );
        eprintln!(".vem+.vex byte-flip sweep: {rejected}/{flipped} rejected");
    }

    /// Every read a query performs, so a flip anywhere in either file has to
    /// surface as a typed error rather than a panic.
    fn walk_everything(meta: &[u8], index: &[u8]) -> Result<()> {
        let index: &'static [u8] = Box::leak(index.to_vec().into_boxed_slice());
        let reader = HnswVectorsReader::open(meta, index, &ID, "")?;
        for entry in reader.fields() {
            let number = entry.field_number;
            let Some(off) = reader.graph(number)? else {
                continue;
            };
            let mut out = Vec::new();
            for level in 0..off.num_levels() {
                for node in off.sorted_nodes_on_level(level)? {
                    off.neighbors_into(level, node, &mut out)?;
                }
            }
            let mut scorer = line(off.size().max(1) as usize);
            search(&mut scorer, Some(&off), 10, u64::MAX, opts())?;
        }
        Ok(())
    }

    #[test]
    fn a_graph_region_past_the_end_of_the_index_file_is_rejected() {
        let (vex, vem) = valid_pair();
        let mut input = SliceInput::new(&vem);
        codec_util::check_index_header(
            &mut input,
            META_CODEC,
            VERSION_START,
            VERSION_CURRENT,
            &ID,
            "",
        )
        .unwrap();
        input.read_i32().unwrap();
        input.read_i32().unwrap();
        input.read_i32().unwrap();
        let offset_at = input.position();
        assert!(matches!(
            open(
                &repack(&vem, |b| {
                    b[offset_at] = 0xFF;
                    b[offset_at + 1] = 0x7F;
                }),
                &vex
            ),
            Err(Error::CorruptMeta(_))
        ));
    }

    #[test]
    fn a_negative_field_number_is_rejected() {
        let (vex, vem) = valid_pair();
        let at = codec_util::index_header_length(META_CODEC, "");
        assert!(matches!(
            open(
                &repack(&vem, |b| b[at..at + 4]
                    .copy_from_slice(&(-5i32).to_le_bytes())),
                &vex
            ),
            Err(Error::CorruptMeta(_))
        ));
    }

    #[test]
    fn a_truncated_meta_file_is_rejected() {
        let (vex, vem) = valid_pair();
        assert!(open(&vem[..vem.len() - 1], &vex).is_err());
        assert!(open(&vem, &vex[..vex.len() - 1]).is_err());
    }

    #[test]
    fn an_out_of_range_neighbour_ordinal_is_rejected() {
        // A hand-built one-node graph whose only neighbour points past the end
        // of the graph: the delta decoder must refuse it rather than hand a
        // wild ordinal to the scorer.
        let mut graph = OnHeapHnswGraph::new(8);
        graph.add_node(0, 0);
        graph.try_set_new_entry_node(0, 0);
        graph.neighbors_mut(0, 0).add_in_order(0, 1.0);
        let (vex, vem) = write_hnsw_vectors(&[field(Some(&graph), 1)], &ID, "").unwrap();
        // Bump the single neighbour ordinal from 0 to 5. `writeGroupVInts`
        // emits whole groups of four behind a flag byte and the remainder as
        // plain vints, so a lone neighbour is one byte right after the vint
        // arc count.
        let payload_at = codec_util::index_header_length(INDEX_CODEC, "") + 1;
        let broken = repack(&vex, |b| b[payload_at] = 5);
        let reader = open(&vem, &broken).unwrap();
        let off = reader.graph(0).unwrap().unwrap();
        let mut out = Vec::new();
        assert!(matches!(
            off.neighbors_into(0, 0, &mut out),
            Err(Error::CorruptMeta(_))
        ));
    }
    // ---------------- merge ----------------

    /// A `Line` scorer that counts how many similarity computations it is
    /// asked for -- the unit the merge is supposed to save.
    #[derive(Debug, Clone)]
    struct CountingLine {
        inner: Line,
        scores: std::rc::Rc<std::cell::Cell<u64>>,
    }

    impl VectorScorer for CountingLine {
        fn score(&mut self, node: i32) -> Result<f32> {
            self.scores.set(self.scores.get() + 1);
            self.inner.score(node)
        }

        fn max_ord(&self) -> i32 {
            self.inner.max_ord()
        }
    }

    impl UpdateableVectorScorer for CountingLine {
        fn set_scoring_ordinal(&mut self, ord: i32) -> Result<()> {
            self.inner.set_scoring_ordinal(ord)
        }
    }

    fn counting(n: usize) -> (CountingLine, std::rc::Rc<std::cell::Cell<u64>>) {
        let scores = std::rc::Rc::new(std::cell::Cell::new(0u64));
        (
            CountingLine {
                inner: line(n),
                scores: scores.clone(),
            },
            scores,
        )
    }

    fn identity(n: usize) -> Vec<i32> {
        (0..n as i32).collect()
    }

    /// The claim the incremental merger exists to make: merging a big segment
    /// with a small one costs far less than rebuilding the whole graph. Counted
    /// in similarity computations, which is what dominates both.
    #[test]
    fn merging_reuses_the_largest_graph_instead_of_rebuilding() {
        let big = 1200usize;
        let small = 120usize;
        let total = big + small;
        let big_graph = build(big);
        let small_graph = build(small);

        // Source 0's docs land first in the merged segment, source 1's after.
        let big_docs = identity(big);
        let small_docs = identity(small);
        let big_map = identity(big);
        let small_map: Vec<i32> = (0..small as i32).map(|d| d + big as i32).collect();
        let merged_ord_to_doc = identity(total);
        let sources = [
            GraphMergeSource {
                graph: Some(&big_graph),
                ord_to_doc: &big_docs,
                doc_map: &big_map,
            },
            GraphMergeSource {
                graph: Some(&small_graph),
                ord_to_doc: &small_docs,
                doc_map: &small_map,
            },
        ];

        let (scorer, merge_scores) = counting(total);
        let merged = merge_one_field(
            scorer,
            8,
            32,
            crate::hnsw::DEFAULT_RAND_SEED,
            &merged_ord_to_doc,
            &sources,
        )
        .unwrap()
        .expect("a 1320-vector segment is well past the graph threshold");

        let (rebuild_scorer, rebuild_scores) = counting(total);
        let rebuilt = HnswGraphBuilder::new(rebuild_scorer, 8, 32, crate::hnsw::DEFAULT_RAND_SEED)
            .unwrap()
            .build(total as i32)
            .unwrap();

        assert_eq!(merged.sorted_nodes_on_level(0).unwrap().len(), total);
        assert!(
            merge_scores.get() * 3 < rebuild_scores.get(),
            "the incremental merge scored {} times against a rebuild's {} -- it is supposed to \
             cost a fraction, not a comparable amount",
            merge_scores.get(),
            rebuild_scores.get()
        );
        let _ = rebuilt;

        // And it must still be searchable: a query anywhere in either source's
        // range finds its own vector.
        let mut searcher = HnswGraphSearcher::new(10, merged.size());
        for target in [0usize, 5, big - 1, big, total - 1] {
            let mut collector = KnnCollector::new(10, u64::MAX);
            let mut scorer = Line {
                values: (0..total).map(|i| i as f32).collect(),
                query: target as f32,
            };
            searcher
                .search(&mut collector, &mut scorer, &merged, None)
                .unwrap();
            let found: Vec<i32> = collector.top_docs().into_iter().map(|(n, _)| n).collect();
            assert!(
                found.contains(&(target as i32)),
                "merged graph lost ordinal {target}: {found:?}"
            );
        }
    }

    /// The **mixed** case, and the only one that reaches `new_ord_mapping`'s
    /// `initialized` bitset: one source has a reusable graph and one does not,
    /// so `order.len() != sources.len()` and Java's
    /// `graphReaders.size() == numReaders ? null : new FixedBitSet(maxOrd)`
    /// allocates the set that records which merged ordinals a reused graph
    /// already covers. Every ordinal the graphless source contributes must be
    /// left for insertion from scratch, and every ordinal the reused graph
    /// covers must be marked -- so the merged graph has to end up complete,
    /// with both sources' vectors reachable.
    #[test]
    fn a_source_without_a_graph_leaves_its_ordinals_to_be_inserted_from_scratch() {
        let reusable = 600usize;
        let graphless = 300usize;
        let total = reusable + graphless;
        let big_graph = build(reusable);
        let docs_a = identity(reusable);
        let docs_b = identity(graphless);
        let map_a = identity(reusable);
        let map_b: Vec<i32> = (0..graphless as i32).map(|d| d + reusable as i32).collect();
        let merged_ord_to_doc = identity(total);
        let sources = [
            GraphMergeSource {
                graph: Some(&big_graph),
                ord_to_doc: &docs_a,
                doc_map: &map_a,
            },
            GraphMergeSource {
                graph: None,
                ord_to_doc: &docs_b,
                doc_map: &map_b,
            },
        ];
        let merged = merge_one_field(
            line(total),
            8,
            32,
            crate::hnsw::DEFAULT_RAND_SEED,
            &merged_ord_to_doc,
            &sources,
        )
        .unwrap()
        .expect("900 vectors is past the graph threshold");

        // Completeness is the property the `initialized` bookkeeping exists
        // for: an ordinal marked when it should not be is never inserted and
        // vanishes from the graph; one left unmarked is inserted twice.
        assert_eq!(merged.sorted_nodes_on_level(0).unwrap().len(), total);
        let mut seen = std::collections::HashSet::new();
        for node in merged.sorted_nodes_on_level(0).unwrap() {
            assert!(seen.insert(node), "ordinal {node} appears twice on level 0");
        }
        assert_eq!(seen.len(), total);
        // The reused graph's own ordinals and the graphless source's are both
        // present, which is what says the bitset covered the right half.
        assert!(seen.contains(&0));
        assert!(seen.contains(&(reusable as i32 - 1)));
        assert!(seen.contains(&(reusable as i32)));
        assert!(seen.contains(&(total as i32 - 1)));
    }

    /// No source has a graph (every segment was below the threshold), so
    /// there is nothing to reuse and Lucene rebuilds. The result must still be
    /// a complete graph.
    #[test]
    fn merging_rebuilds_when_no_source_has_a_graph() {
        let total = 800usize;
        let docs_a = identity(400);
        let docs_b = identity(400);
        let map_a = identity(400);
        let map_b: Vec<i32> = (0..400).map(|d| d + 400).collect();
        let merged_ord_to_doc = identity(total);
        let sources: [GraphMergeSource<'_, OnHeapHnswGraph>; 2] = [
            GraphMergeSource {
                graph: None,
                ord_to_doc: &docs_a,
                doc_map: &map_a,
            },
            GraphMergeSource {
                graph: None,
                ord_to_doc: &docs_b,
                doc_map: &map_b,
            },
        ];
        let merged = merge_one_field(
            line(total),
            8,
            32,
            crate::hnsw::DEFAULT_RAND_SEED,
            &merged_ord_to_doc,
            &sources,
        )
        .unwrap()
        .unwrap();
        assert_eq!(merged.sorted_nodes_on_level(0).unwrap().len(), total);
        // A rebuild is exactly `HnswGraphBuilder::build`, so it must be
        // arc-for-arc that graph.
        let rebuilt = build(total);
        assert_eq!(merged.entry_node(), rebuilt.entry_node());
        assert_eq!(merged.num_levels(), rebuilt.num_levels());
        let mut a = Vec::new();
        let mut b = Vec::new();
        for level in 0..merged.num_levels() {
            for node in merged.sorted_nodes_on_level(level).unwrap() {
                merged.neighbors_into(level, node, &mut a).unwrap();
                rebuilt.neighbors_into(level, node, &mut b).unwrap();
                assert_eq!(a, b, "level {level} node {node}");
            }
        }
    }

    /// Below `HNSW_GRAPH_THRESHOLD` Lucene writes no graph at all, on merge
    /// exactly as on flush -- a merged segment of 40 vectors must not suddenly
    /// acquire one.
    #[test]
    fn merging_writes_no_graph_below_the_threshold() {
        let docs = identity(40);
        let map = identity(40);
        let merged_ord_to_doc = identity(40);
        let graph = build(40);
        let sources = [GraphMergeSource {
            graph: Some(&graph),
            ord_to_doc: &docs,
            doc_map: &map,
        }];
        assert!(merge_one_field(
            line(40),
            8,
            32,
            crate::hnsw::DEFAULT_RAND_SEED,
            &merged_ord_to_doc,
            &sources,
        )
        .unwrap()
        .is_none());
        // ...and an empty merged field is `None` too.
        assert!(merge_one_field(
            line(1),
            8,
            32,
            crate::hnsw::DEFAULT_RAND_SEED,
            &[],
            &sources,
        )
        .unwrap()
        .is_none());
    }

    /// A graph more than `DELETE_PCT_THRESHOLD` deleted is not used as the
    /// base: its connectivity has degraded, so Lucene prefers a rebuild.
    #[test]
    fn a_heavily_deleted_graph_is_not_used_as_the_base() {
        let n = 3000usize;
        let graph = build(n);
        let docs = identity(n);
        // Drop 60% -- past the 40% threshold.
        let mut map = vec![-1i32; n];
        let mut kept = 0i32;
        for (d, slot) in map.iter_mut().enumerate() {
            if d % 5 < 2 {
                *slot = kept;
                kept += 1;
            }
        }
        let merged_ord_to_doc = identity(kept as usize);
        let sources = [GraphMergeSource {
            graph: Some(&graph),
            ord_to_doc: &docs,
            doc_map: &map,
        }];
        let merged = merge_one_field(
            line(kept as usize),
            8,
            32,
            crate::hnsw::DEFAULT_RAND_SEED,
            &merged_ord_to_doc,
            &sources,
        )
        .unwrap()
        .unwrap();
        // A rebuild, so it is arc-for-arc `HnswGraphBuilder::build(kept)`.
        let rebuilt = build(kept as usize);
        assert_eq!(merged.entry_node(), rebuilt.entry_node());
        let mut a = Vec::new();
        let mut b = Vec::new();
        for node in 0..kept {
            merged.neighbors_into(0, node, &mut a).unwrap();
            rebuilt.neighbors_into(0, node, &mut b).unwrap();
            assert_eq!(a, b, "node {node}");
        }
    }

    /// `M`/`beamWidth` out of range must be an error, not a panic -- the merge
    /// entry point allocates a graph sized by `M` before any builder sees it.
    #[test]
    fn merge_rejects_out_of_range_graph_parameters() {
        let graph = build(1400);
        let docs = identity(1400);
        let map = identity(1400);
        let sources = [GraphMergeSource {
            graph: Some(&graph),
            ord_to_doc: &docs,
            doc_map: &map,
        }];
        for (m, beam) in [(0, 32), (513, 32), (8, 0), (8, 3201)] {
            assert!(
                matches!(
                    merge_one_field(
                        line(1400),
                        m,
                        beam,
                        crate::hnsw::DEFAULT_RAND_SEED,
                        &identity(1400),
                        &sources
                    ),
                    Err(Error::InvalidGraphParameter(_))
                ),
                "M={m} beamWidth={beam} must be rejected"
            );
        }
    }

    /// A source whose graph size disagrees with its vector count is a caller
    /// bug that would otherwise index the ordinal map out of bounds.
    #[test]
    fn merge_rejects_a_source_whose_graph_and_vectors_disagree() {
        let graph = build(1500);
        let docs = identity(1400);
        let map = identity(1400);
        let merged_ord_to_doc = identity(1400);
        let sources = [GraphMergeSource {
            graph: Some(&graph),
            ord_to_doc: &docs,
            doc_map: &map,
        }];
        assert!(matches!(
            merge_one_field(
                line(1400),
                8,
                32,
                crate::hnsw::DEFAULT_RAND_SEED,
                &merged_ord_to_doc,
                &sources
            ),
            Err(Error::InvalidGraphParameter(_))
        ));

        // And a doc map with no entry for a doc the source's vectors name.
        let graph = build(1400);
        let docs = identity(1400);
        let short = identity(3);
        let sources = [GraphMergeSource {
            graph: Some(&graph),
            ord_to_doc: &docs,
            doc_map: &short,
        }];
        assert!(matches!(
            merge_one_field(
                line(1400),
                8,
                32,
                crate::hnsw::DEFAULT_RAND_SEED,
                &identity(1400),
                &sources
            ),
            Err(Error::InvalidGraphParameter(_))
        ));
    }

    /// The whole merge, end to end at the codec level: two segments' `.vec`/
    /// `.vemf`/`.vem`/`.vex` in, one segment's out, reopened and searched.
    #[test]
    fn a_merged_segment_round_trips_through_both_readers() {
        use crate::vectors::{
            FieldVectorData, FlatVectorMergeSource, FlatVectorsField, FlatVectorsReader,
            FlatVectorsWriter, MergeSourceValues, MergedFlatVectorField,
        };

        const DIM: i32 = 4;
        // Two source segments whose vectors interleave in value space, so the
        // merged graph has to link regions neither source graph did.
        let make = |offset: f32, count: i32| -> (Vec<u8>, Vec<u8>) {
            let values: Vec<f32> = (0..count)
                .flat_map(|i| (0..DIM).map(move |d| offset + i as f32 * 2.0 + d as f32 * 0.01))
                .collect();
            crate::vectors::write_flat_vectors(
                &[FlatVectorsField {
                    field_number: 0,
                    similarity: VectorSimilarityFunction::Euclidean,
                    dimension: DIM,
                    docs: (0..count).collect(),
                    values: FieldVectorData::Float32(values),
                }],
                count,
                &ID,
                "",
            )
            .unwrap()
        };
        let (a_vec, a_meta) = make(0.0, 900);
        let (b_vec, b_meta) = make(1.0, 700);
        let a = FlatVectorsReader::open(&a_meta, &a_vec, &ID, "").unwrap();
        let b = FlatVectorsReader::open(&b_meta, &b_vec, &ID, "").unwrap();

        let a_map = identity(900);
        let b_map: Vec<i32> = (0..700).map(|d| d + 900).collect();
        let flat_sources = vec![
            FlatVectorMergeSource {
                values: MergeSourceValues::Float32(a.float_vector_values(0).unwrap()),
                doc_map: &a_map,
            },
            FlatVectorMergeSource {
                values: MergeSourceValues::Float32(b.float_vector_values(0).unwrap()),
                doc_map: &b_map,
            },
        ];
        let mut writer = FlatVectorsWriter::new(1600, &ID, "");
        writer
            .merge_one_flat_vector_field(&MergedFlatVectorField {
                field_number: 0,
                encoding: VectorEncoding::Float32,
                similarity: VectorSimilarityFunction::Euclidean,
                dimension: DIM,
                sources: &flat_sources,
            })
            .unwrap();
        let (merged_vec, merged_vemf) = writer.finish();

        // Both source graphs, then the merged graph over the merged flat file.
        let a_graph = HnswGraphBuilder::new(
            a.float_vector_values(0).unwrap().ord_scorer(),
            8,
            32,
            crate::hnsw::DEFAULT_RAND_SEED,
        )
        .unwrap()
        .build(900)
        .unwrap();
        let b_graph = HnswGraphBuilder::new(
            b.float_vector_values(0).unwrap().ord_scorer(),
            8,
            32,
            crate::hnsw::DEFAULT_RAND_SEED,
        )
        .unwrap()
        .build(700)
        .unwrap();

        let merged_flat = FlatVectorsReader::open(&merged_vemf, &merged_vec, &ID, "").unwrap();
        let merged_values = merged_flat.float_vector_values(0).unwrap();
        let merged_ord_to_doc: Vec<i32> = (0..merged_values.size())
            .map(|o| merged_values.ord_to_doc(o).unwrap())
            .collect();
        let a_docs = identity(900);
        let b_docs = identity(700);
        let graph_sources = [
            GraphMergeSource {
                graph: Some(&a_graph),
                ord_to_doc: &a_docs,
                doc_map: &a_map,
            },
            GraphMergeSource {
                graph: Some(&b_graph),
                ord_to_doc: &b_docs,
                doc_map: &b_map,
            },
        ];
        let merged_graph = merge_one_field(
            merged_values.ord_scorer(),
            8,
            32,
            crate::hnsw::DEFAULT_RAND_SEED,
            &merged_ord_to_doc,
            &graph_sources,
        )
        .unwrap()
        .unwrap();

        let (vex, vem) = write_hnsw_vectors(
            &[HnswVectorsField {
                field_number: 0,
                encoding: VectorEncoding::Float32,
                similarity: VectorSimilarityFunction::Euclidean,
                dimension: DIM,
                count: 1600,
                graph: Some(&merged_graph),
                m: 8,
            }],
            &ID,
            "",
        )
        .unwrap();

        // Reopen the written bytes and search them, comparing against the
        // exhaustive answer over the same merged file.
        let reader = HnswVectorsReader::open(&vem, &vex, &ID, "").unwrap();
        let graph = reader.graph(0).unwrap().unwrap();
        assert_eq!(graph.size(), 1600);
        let mut hits = 0usize;
        let mut total = 0usize;
        for q in 0..10 {
            let query: Vec<f32> = (0..DIM)
                .map(|d| q as f32 * 137.0 + d as f32 * 0.01)
                .collect();
            let exact: Vec<i32> = merged_values
                .exhaustive_search(&query, 10)
                .unwrap()
                .into_iter()
                .map(|(d, _)| d)
                .collect();
            let mut scorer = merged_values.scorer(&query).unwrap();
            let approx: Vec<i32> = search(&mut scorer, Some(&graph), 10, u64::MAX, opts())
                .unwrap()
                .0
                .into_iter()
                .map(|(ord, _)| merged_values.ord_to_doc(ord).unwrap())
                .collect();
            total += exact.len();
            hits += approx.iter().filter(|d| exact.contains(d)).count();
        }
        let recall = hits as f64 / total as f64;
        assert!(
            recall >= 0.9,
            "search over the merged, serialized graph recalled {recall}"
        );
    }
}
