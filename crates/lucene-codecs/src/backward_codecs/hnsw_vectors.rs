//! The retired HNSW vector formats Lucene 9.0-9.8 wrote: ports of
//! `backward_codecs.lucene90.Lucene90HnswVectorsReader` (with the search of
//! `Lucene90OnHeapHnswGraph` and `Lucene90BoundsChecker`),
//! `lucene91.Lucene91HnswVectorsReader`, `lucene92.Lucene92HnswVectorsReader`
//! (and its `OffHeapFloatVectorValues`), `lucene94.Lucene94HnswVectorsReader`
//! (and its `OffHeap{Float,Byte}VectorValues`) and
//! `lucene95.Lucene95HnswVectorsReader`.
//!
//! Each keeps its vectors, its graph and its metadata in one `.vec`/`.vex`/
//! `.vem` triple (the current `Lucene99HnswVectorsFormat` splits the flat
//! vectors into a `.vec`/`.vemf` pair of their own). The vector bytes
//! themselves never changed -- `size * dimension` little-endian floats or
//! signed bytes, ordinal-addressed -- so a retired field is served by the
//! current [`FlatVectorsReader`] once its entry is described, and the search
//! layer needs nothing format-specific but the graph walk.
//!
//! | format (`.vem` codec) | Lucene | encodings | `ordToDoc` | graph |
//! |---|---|---|---|---|
//! | `Lucene90HnswVectorsFormat` | 9.0 | float | `vint` per ordinal, in `.vem` | one level; per-node offsets in `.vem`; `int` count + `vint` deltas from `-1` |
//! | `Lucene91HnswVectorsFormat` | 9.1 | float | dense marker, or `int` per ordinal | fixed `(1 + M) * 4`-byte slots on every level, `int` neighbours |
//! | `lucene92HnswVectorsFormat` | 9.2-9.3 | float | `IndexedDISI` + `DirectMonotonic` in `.vec` | fixed slots, `(1 + 2M) * 4` on level 0 |
//! | `lucene94HnswVectorsFormat` | 9.4 | float, byte | as 9.2 | as 9.2 |
//! | `Lucene95HnswVectorsFormat` | 9.5-9.8 | float, byte | as 9.2 (`OrdToDocDISIReaderConfiguration`) | `Lucene99HnswVectorsFormat` version 0: `vint` deltas, `DirectMonotonic` node offsets |
//!
//! Note the lower-case `l` of `lucene92`/`lucene94`: those two formats'
//! codec names (and 9.2's format name, hence its file names) really are
//! spelled that way.
//!
//! # Search
//!
//! `Lucene91` onwards search through the current
//! [`HnswGraphSearcher`] -- 10.5.0 calls `HnswGraphSearcher.search` straight
//! from each retired reader, with **no** exhaustive-scan branch (that choice
//! is `Lucene99HnswVectorsReader`'s own). `Lucene90` never had a hierarchy:
//! its reader still carries 9.0's single-level search, seeded from
//! `numSeed = k` random entry points drawn from a `SplittableRandom` whose
//! seed is the `.vex` footer's checksum, and [`RetiredHnswVectorsReader::search`]
//! ports it exactly.
//!
//! One Java quirk is kept on purpose: `Lucene90HnswVectorsReader.search`
//! hands the collector **ordinals**, not documents (`knnCollector.collect(node,
//! ...)` with no `ordToDoc`), so a 9.0 segment whose vector field is sparse
//! reports ordinal `n` as document `n`. Lucene 10.5.0 answers a KNN query that
//! way, and this port answers it the same way; see
//! [`RetiredHnswVectorsReader::hits_are_ordinals`].

use std::sync::Arc;

use lucene_store::codec_util::{self, ID_LENGTH};
use lucene_store::data_input::{DataInput, SliceInput};
use lucene_util::fixed_bit_set::FixedBitSet;
use lucene_util::splittable_random::SplittableRandom;

use crate::field_infos::{VectorEncoding, VectorSimilarityFunction};
use crate::hnsw::{
    HnswGraphSearcher, HnswGraphView, KnnCollect, KnnCollector, NeighborQueue, VectorScorer,
};
use crate::hnsw_vectors::{
    read_graph_meta, GraphMetaHead, HnswFieldEntry, HnswVectorsReader, OffHeapHnswGraph,
};
use crate::vectors::{
    check_vector_region, read_similarity_function, read_vector_encoding, Error, FlatFieldEntry,
    FlatVectorsReader, OrdToDoc, Result,
};

/// `HnswGraph.UNKNOWN_MAX_CONN`: what a `Lucene90` graph reports, since 9.0
/// did not record `M`.
const UNKNOWN_MAX_CONN: i32 = -1;

fn corrupt<T>(msg: impl Into<String>) -> Result<T> {
    Err(Error::CorruptMeta(msg.into()))
}

/// Which retired HNSW format a field was written with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetiredHnswFormat {
    Lucene90,
    Lucene91,
    Lucene92,
    Lucene94,
    Lucene95,
}

impl RetiredHnswFormat {
    /// Every retired format, oldest first.
    pub const ALL: [RetiredHnswFormat; 5] = [
        RetiredHnswFormat::Lucene90,
        RetiredHnswFormat::Lucene91,
        RetiredHnswFormat::Lucene92,
        RetiredHnswFormat::Lucene94,
        RetiredHnswFormat::Lucene95,
    ];

    /// `KnnVectorsFormat.forName`: the name `PerFieldKnnVectorsFormat.format`
    /// records for a field, and the middle of the field's file names.
    pub fn for_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|f| f.name() == name)
    }

    /// `KnnVectorsFormat.getName()`.
    pub fn name(self) -> &'static str {
        match self {
            RetiredHnswFormat::Lucene90 => "Lucene90HnswVectorsFormat",
            RetiredHnswFormat::Lucene91 => "Lucene91HnswVectorsFormat",
            RetiredHnswFormat::Lucene92 => "lucene92HnswVectorsFormat",
            RetiredHnswFormat::Lucene94 => "Lucene94HnswVectorsFormat",
            RetiredHnswFormat::Lucene95 => "Lucene95HnswVectorsFormat",
        }
    }

    /// The prefix of the three `*_CODEC_NAME`s (`...Meta`, `...Data`,
    /// `...Index`). Differs from [`Self::name`] for `Lucene94`, whose format
    /// is `Lucene94HnswVectorsFormat` but whose headers say `lucene94`.
    fn codec_prefix(self) -> &'static str {
        match self {
            RetiredHnswFormat::Lucene94 => "lucene94HnswVectorsFormat",
            other => other.name(),
        }
    }

    /// `VERSION_CURRENT` (`VERSION_START` is 0 for all five).
    fn version_current(self) -> i32 {
        match self {
            RetiredHnswFormat::Lucene94 | RetiredHnswFormat::Lucene95 => 1,
            _ => 0,
        }
    }

    /// Whether the format's writer sorted each node's neighbours by ordinal.
    /// `Lucene90` and `Lucene95` store deltas, so theirs are ascending by
    /// construction; `Lucene91`..`Lucene94` stored a `NeighborArray` as it
    /// stood, i.e. in score order -- not a defect a checker may report.
    pub fn neighbors_sorted(self) -> bool {
        matches!(
            self,
            RetiredHnswFormat::Lucene90 | RetiredHnswFormat::Lucene95
        )
    }
}

/// How one field's graph is laid out in `.vex`.
#[derive(Debug, Clone)]
enum GraphEntry {
    /// `Lucene90`: one level; `ord_offsets[ord]` is the node's position in
    /// the field's graph region.
    Lucene90 {
        index_offset: i64,
        index_length: i64,
        ord_offsets: Arc<[i64]>,
    },
    /// `Lucene91`/`92`/`94`: every node on a level has a fixed-size slot.
    Fixed {
        index_offset: i64,
        index_length: i64,
        max_conn: i32,
        num_levels: i32,
        size: i32,
        nodes_by_level: Arc<[Vec<i32>]>,
        /// `graphOffsetsByLevel`.
        level_offsets: Arc<[i64]>,
        /// Slot width on level 0 and above, in bytes.
        bytes_for_conns0: i64,
        bytes_for_conns: i64,
    },
    /// `Lucene95`: the `Lucene99HnswVectorsFormat` version 0 graph.
    Lucene95(HnswFieldEntry),
}

/// One retired field: its vectors (served by [`FlatVectorsReader`]) and its
/// graph.
#[derive(Debug, Clone)]
struct RetiredField {
    field_number: i32,
    graph: GraphEntry,
}

/// A retired HNSW vectors reader over one segment's `.vem`/`.vec`/`.vex`.
#[derive(Debug, Clone)]
pub struct RetiredHnswVectorsReader<'a> {
    format: RetiredHnswFormat,
    flat: FlatVectorsReader<'a>,
    index: &'a [u8],
    fields: Vec<RetiredField>,
    /// `Lucene90HnswVectorsReader.checksumSeed`: the `.vex` footer checksum.
    checksum_seed: u64,
}

impl<'a> RetiredHnswVectorsReader<'a> {
    /// The reader constructor of `format`: `readMetadata` (header, every
    /// field entry, footer) and `openDataInput` for `.vec` and `.vex`.
    ///
    /// Like the current readers this checks both data files' whole-file
    /// checksums at open, where Java defers that to `checkIntegrity`.
    pub fn open(
        format: RetiredHnswFormat,
        meta_buf: &[u8],
        data_buf: &'a [u8],
        index_buf: &'a [u8],
        segment_id: &[u8; ID_LENGTH],
        segment_suffix: &str,
    ) -> Result<Self> {
        let prefix = format.codec_prefix();
        let mut meta = SliceInput::new(meta_buf);
        let version_meta = codec_util::check_index_header(
            &mut meta,
            &format!("{prefix}Meta"),
            0,
            format.version_current(),
            segment_id,
            segment_suffix,
        )?
        .version;
        for (buf, kind) in [(data_buf, "Data"), (index_buf, "Index")] {
            let codec = format!("{prefix}{kind}");
            let version = codec_util::check_index_header(
                &mut SliceInput::new(buf),
                &codec,
                0,
                format.version_current(),
                segment_id,
                segment_suffix,
            )?
            .version;
            if version != version_meta {
                return corrupt(format!(
                    "Format versions mismatch: meta={version_meta}, {codec}={version}"
                ));
            }
        }
        for (buf, what) in [(meta_buf, ".vem"), (data_buf, ".vec"), (index_buf, ".vex")] {
            let Some(end) = buf.len().checked_sub(codec_util::FOOTER_LENGTH) else {
                return corrupt(format!("{what} is shorter than its footer"));
            };
            codec_util::check_whole_file_footer(buf, end)?;
        }
        let checksum_seed = codec_util::retrieve_checksum(index_buf)?;

        let mut flat_fields = Vec::new();
        let mut fields = Vec::new();
        loop {
            let field_number = meta.read_i32()?;
            if field_number == -1 {
                break;
            }
            if field_number < 0 {
                return corrupt(format!("Invalid field number: {field_number}"));
            }
            if fields
                .iter()
                .any(|f: &RetiredField| f.field_number == field_number)
            {
                return corrupt(format!("duplicate field number {field_number}"));
            }
            let (flat, graph) = read_field(
                format,
                &mut meta,
                field_number,
                data_buf.len(),
                index_buf.len(),
            )?;
            flat_fields.push(flat);
            fields.push(RetiredField {
                field_number,
                graph,
            });
        }
        Ok(RetiredHnswVectorsReader {
            format,
            flat: FlatVectorsReader::from_entries(data_buf, flat_fields),
            index: index_buf,
            fields,
            checksum_seed,
        })
    }

    pub fn format(&self) -> RetiredHnswFormat {
        self.format
    }

    /// The vectors, served by the current flat reader: `getFloatVectorValues`
    /// and `getByteVectorValues` of every retired reader.
    pub fn flat(&self) -> &FlatVectorsReader<'a> {
        &self.flat
    }

    /// Whether a KNN search of this format reports vector **ordinals** as
    /// its hits' doc ids. True for `Lucene90` only -- see the module docs.
    pub fn hits_are_ordinals(&self) -> bool {
        self.format == RetiredHnswFormat::Lucene90
    }

    fn field(&self, field_number: i32) -> Result<&RetiredField> {
        self.fields
            .iter()
            .find(|f| f.field_number == field_number)
            .ok_or(Error::UnknownField(field_number))
    }

    /// `getGraph(FieldEntry)`: the field's graph, read off `.vex` on demand.
    pub fn graph(&self, field_number: i32) -> Result<RetiredGraph<'a>> {
        let field = self.field(field_number)?;
        let region = |offset: i64, length: i64| -> Result<&'a [u8]> {
            let start = usize::try_from(offset).ok();
            let end = start.and_then(|s| s.checked_add(usize::try_from(length).ok()?));
            match (start, end) {
                (Some(s), Some(e)) => self.index.get(s..e),
                _ => None,
            }
            .ok_or_else(|| {
                Error::CorruptMeta(format!(
                    "graph region [{offset}, +{length}) is not inside a {} byte .vex",
                    self.index.len()
                ))
            })
        };
        match &field.graph {
            GraphEntry::Lucene90 {
                index_offset,
                index_length,
                ord_offsets,
            } => Ok(RetiredGraph::Lucene90(Lucene90Graph {
                data: region(*index_offset, *index_length)?,
                ord_offsets: Arc::clone(ord_offsets),
            })),
            GraphEntry::Fixed {
                index_offset,
                index_length,
                max_conn,
                num_levels,
                size,
                nodes_by_level,
                level_offsets,
                bytes_for_conns0,
                bytes_for_conns,
            } => Ok(RetiredGraph::Fixed(FixedGraph {
                data: region(*index_offset, *index_length)?,
                max_conn: *max_conn,
                num_levels: *num_levels,
                size: *size,
                entry_node: if *num_levels > 1 {
                    // `read_fixed_levels` rejects an empty upper level.
                    nodes_by_level
                        .last()
                        .and_then(|n| n.first())
                        .copied()
                        .unwrap_or(0)
                } else {
                    0
                },
                nodes_by_level: Arc::clone(nodes_by_level),
                level_offsets: Arc::clone(level_offsets),
                bytes_for_conns0: *bytes_for_conns0,
                bytes_for_conns: *bytes_for_conns,
            })),
            GraphEntry::Lucene95(entry) => {
                let reader = HnswVectorsReader::from_entries(self.index, 0, vec![entry.clone()]);
                match reader.graph(field_number)? {
                    Some(g) => Ok(RetiredGraph::Lucene95(g)),
                    None => Ok(RetiredGraph::Empty),
                }
            }
        }
    }

    /// The retired reader's `search(field, target, knnCollector, acceptDocs)`
    /// for a collector of `k` and `visit_limit` (Java's `TopKnnCollector`),
    /// over `scorer` (built from [`Self::flat`]'s values for the target).
    ///
    /// `accept_ords` is `getAcceptOrds(acceptDocs)` in ordinal space, as
    /// [`crate::hnsw_vectors::search`] takes it; `seed_ords` are
    /// `KnnSearchStrategy.Seeded` entry points, honoured by the formats that
    /// search through [`HnswGraphSearcher`] and ignored by `Lucene90`, whose
    /// reader never consults the strategy.
    ///
    /// Returns `(ordinal, score)` hits, best first, and whether the collector
    /// early-terminated. Translating an ordinal to a document is the caller's
    /// job -- unless [`Self::hits_are_ordinals`].
    pub fn search<S: VectorScorer>(
        &self,
        field_number: i32,
        scorer: &mut S,
        k: usize,
        visit_limit: u64,
        accept_ords: Option<&FixedBitSet>,
        seed_ords: Option<&[i32]>,
    ) -> Result<(Vec<(i32, f32)>, bool)> {
        let mut collector = KnnCollector::new(k, visit_limit);
        self.search_with(field_number, scorer, &mut collector, accept_ords, seed_ords)?;
        let early = collector.early_terminated();
        Ok((collector.top_docs(), early))
    }

    /// [`Self::search`] into any [`KnnCollect`]or -- whatever collector
    /// Java's `search(field, target, knnCollector, acceptDocs)` is handed
    /// (`VectorSimilarityCollector`, `HnswQueueSaturationCollector`,
    /// `TimeLimitingKnnCollector`, ...). The walk reads its `k()` and
    /// `visitLimit()`; the results stay in the collector.
    pub fn search_with<S: VectorScorer, C: KnnCollect + ?Sized>(
        &self,
        field_number: i32,
        scorer: &mut S,
        collector: &mut C,
        accept_ords: Option<&FixedBitSet>,
        seed_ords: Option<&[i32]>,
    ) -> Result<()> {
        let graph = self.graph(field_number)?;
        let size = scorer.max_ord();
        let k = collector.k();
        // `if (fieldEntry.size() == 0 || knnCollector.k() == 0) return;`
        if size <= 0 || k == 0 {
            return Ok(());
        }
        if let Some(bits) = accept_ords {
            if bits.len() < size as usize || bits.len() < graph.size().max(0) as usize {
                return Err(Error::InvalidGraphParameter(format!(
                    "the accept-ordinal set covers {} ordinals, short of this field's {size}",
                    bits.len()
                )));
            }
        }
        match &graph {
            RetiredGraph::Lucene90(g) => {
                let mut random = SplittableRandom::new(self.checksum_seed);
                lucene90_search(scorer, g, accept_ords, &mut random, collector)?;
            }
            RetiredGraph::Empty => {}
            _ => {
                let mut searcher = HnswGraphSearcher::new(k, graph.size());
                match seed_ords {
                    Some(seeds) => {
                        searcher.search_seeded(collector, scorer, &graph, accept_ords, seeds)?
                    }
                    None => searcher.search(collector, scorer, &graph, accept_ords)?,
                }
            }
        }
        Ok(())
    }
}

/// Reads one field's entry in `format`'s `.vem` layout: the flat half as a
/// [`FlatFieldEntry`], the graph half as a [`GraphEntry`].
fn read_field(
    format: RetiredHnswFormat,
    meta: &mut SliceInput<'_>,
    field_number: i32,
    data_len: usize,
    index_len: usize,
) -> Result<(FlatFieldEntry, GraphEntry)> {
    // `readVectorEncoding` exists from `Lucene94` on; before that every
    // vector was a float.
    let encoding = match format {
        RetiredHnswFormat::Lucene94 | RetiredHnswFormat::Lucene95 => read_vector_encoding(meta)?,
        _ => VectorEncoding::Float32,
    };
    // `VectorSimilarityFunction.values()[id]`: the declaration order, which is
    // the ordinal list `read_similarity_function` pins.
    let similarity: VectorSimilarityFunction = read_similarity_function(meta)?;
    let vector_data_offset = meta.read_vlong()?;
    let vector_data_length = meta.read_vlong()?;
    let index_offset = meta.read_vlong()?;
    let index_length = meta.read_vlong()?;
    let dimension = if format == RetiredHnswFormat::Lucene95 {
        meta.read_vint()?
    } else {
        meta.read_i32()?
    };
    let size = meta.read_i32()?;
    // `validateFieldEntry`: the dimension against `.fnm` is the caller's
    // check; the data-length identity is this one.
    check_vector_region(
        encoding,
        dimension,
        size,
        vector_data_offset,
        vector_data_length,
        data_len,
    )?;
    check_region(index_offset, index_length, index_len)?;

    let (ord_to_doc, graph) = match format {
        RetiredHnswFormat::Lucene90 => {
            let docs = read_int_docs(meta, size, |m| m.read_vint())?;
            let mut ord_offsets = Vec::with_capacity(docs.len());
            let mut offset = 0i64;
            for _ in 0..size {
                let delta = meta.read_vlong()?;
                offset = match offset.checked_add(delta) {
                    Some(o) if delta >= 0 && o <= index_length => o,
                    _ => {
                        return corrupt(format!(
                            "graph node offset past the field's {index_length} byte region"
                        ))
                    }
                };
                ord_offsets.push(offset);
            }
            (
                OrdToDoc::Explicit(docs),
                GraphEntry::Lucene90 {
                    index_offset,
                    index_length,
                    ord_offsets: ord_offsets.into(),
                },
            )
        }
        RetiredHnswFormat::Lucene91 => {
            let ord_to_doc = match meta.read_byte()? as i8 {
                -1 => OrdToDoc::Dense,
                0 => OrdToDoc::Explicit(read_int_docs(meta, size, |m| m.read_i32())?),
                other => return corrupt(format!("illegal dense/sparse marker {other}")),
            };
            let max_conn = meta.read_i32()?;
            let graph = read_fixed_graph(meta, index_offset, index_length, max_conn, size, false)?;
            (ord_to_doc, graph)
        }
        RetiredHnswFormat::Lucene92 | RetiredHnswFormat::Lucene94 => {
            let ord_to_doc = OrdToDoc::from_stored_meta(meta, size)?;
            let max_conn = meta.read_i32()?;
            let graph = read_fixed_graph(meta, index_offset, index_length, max_conn, size, true)?;
            (ord_to_doc, graph)
        }
        RetiredHnswFormat::Lucene95 => {
            let ord_to_doc = OrdToDoc::from_stored_meta(meta, size)?;
            let entry = read_graph_meta(
                meta,
                GraphMetaHead {
                    field_number,
                    encoding,
                    similarity,
                    vector_index_offset: index_offset,
                    vector_index_length: index_length,
                    dimension,
                    size,
                },
                index_len,
            )?;
            (ord_to_doc, GraphEntry::Lucene95(entry))
        }
    };
    Ok((
        FlatFieldEntry {
            field_number,
            encoding,
            similarity,
            vector_data_offset,
            vector_data_length,
            dimension,
            size,
            ord_to_doc,
        },
        graph,
    ))
}

/// `[offset, offset + length)` lies inside a file of `len` bytes.
fn check_region(offset: i64, length: i64, len: usize) -> Result<()> {
    let end = u64::try_from(offset)
        .ok()
        .zip(u64::try_from(length).ok())
        .and_then(|(o, l)| o.checked_add(l));
    match end {
        Some(end) if end <= len as u64 => Ok(()),
        _ => corrupt(format!(
            "graph region [{offset}, +{length}) past the end of a {len} byte .vex file"
        )),
    }
}

/// `int[] ordToDoc` of `size` entries, one `read` each. Java never checks
/// it; this port requires it strictly increasing and non-negative, because
/// [`OrdToDoc::Explicit`] answers doc -> ordinal by binary search.
fn read_int_docs(
    meta: &mut SliceInput<'_>,
    size: i32,
    read: impl Fn(&mut SliceInput<'_>) -> lucene_store::Result<i32>,
) -> Result<Arc<[i32]>> {
    // Every entry costs at least one byte of `.vem`.
    if size as usize > meta.remaining() {
        return corrupt(format!(
            "{size} ordToDoc entries claimed, more than the {} bytes left in the .vem",
            meta.remaining()
        ));
    }
    let mut docs = Vec::with_capacity(size as usize);
    let mut last = -1i32;
    for _ in 0..size {
        let doc = read(meta)?;
        if doc <= last {
            return corrupt(format!("ordToDoc is not increasing: {doc} after {last}"));
        }
        last = doc;
        docs.push(doc);
    }
    Ok(docs.into())
}

/// `FieldEntry.create`'s graph half for `Lucene91`/`92`/`94`: `M`, the level
/// count and every upper level's nodes (`int`s), then `graphOffsetsByLevel`.
fn read_fixed_graph(
    meta: &mut SliceInput<'_>,
    index_offset: i64,
    index_length: i64,
    max_conn: i32,
    size: i32,
    double_level0: bool,
) -> Result<GraphEntry> {
    let num_levels = meta.read_i32()?;
    if max_conn <= 0 || max_conn > crate::hnsw::MAXIMUM_MAX_CONN {
        return corrupt(format!("illegal maxConn {max_conn}"));
    }
    if num_levels < 0 {
        return corrupt(format!("illegal level count {num_levels}"));
    }
    let mut nodes_by_level: Vec<Vec<i32>> = Vec::new();
    for level in 0..num_levels {
        let num_nodes = meta.read_i32()?;
        if level == 0 {
            // `assert numNodesOnLevel == size`
            if num_nodes != size {
                return corrupt(format!(
                    "level 0 has {num_nodes} nodes, not the field's {size}"
                ));
            }
            nodes_by_level.push(Vec::new());
            continue;
        }
        // Four `.vem` bytes per node, so the file bounds the allocation.
        if num_nodes <= 0 || num_nodes > size || num_nodes as usize > meta.remaining() / 4 {
            return corrupt(format!(
                "illegal node count {num_nodes} on level {level} (size {size})"
            ));
        }
        let mut nodes = Vec::with_capacity(num_nodes as usize);
        let mut last = -1i32;
        for _ in 0..num_nodes {
            let node = meta.read_i32()?;
            // `Arrays.binarySearch` in `seek` needs them ascending.
            if node <= last || node >= size {
                return corrupt(format!(
                    "HNSW level {level} node {node} out of order or range"
                ));
            }
            last = node;
            nodes.push(node);
        }
        nodes_by_level.push(nodes);
    }
    // `connectionsAndSizeBytes`: `(1 + M) * 4`, and `(1 + 2M) * 4` on level 0
    // from `Lucene92` on.
    // ARITH: `max_conn` was checked into `1..=MAXIMUM_MAX_CONN` (512) above,
    // so both products are at most 4100.
    #[allow(clippy::arithmetic_side_effects)]
    let (bytes_for_conns, bytes_for_conns0) = {
        let upper = (i64::from(max_conn) + 1) * 4;
        let level0 = if double_level0 {
            (2 * i64::from(max_conn) + 1) * 4
        } else {
            upper
        };
        (upper, level0)
    };
    let mut level_offsets = Vec::with_capacity(nodes_by_level.len());
    let mut offset = 0i64;
    for (level, nodes) in nodes_by_level.iter().enumerate() {
        level_offsets.push(offset);
        let (count, width) = if level == 0 {
            (i64::from(size), bytes_for_conns0)
        } else {
            (nodes.len() as i64, bytes_for_conns)
        };
        // `Math.addExact`/`multiplyExact`, and every level must fit the region.
        offset = match count.checked_mul(width).and_then(|b| offset.checked_add(b)) {
            Some(o) if o <= index_length => o,
            _ => {
                return corrupt(format!(
                    "HNSW level {level} runs past the field's {index_length} byte graph region"
                ))
            }
        };
    }
    Ok(GraphEntry::Fixed {
        index_offset,
        index_length,
        max_conn,
        num_levels,
        size,
        nodes_by_level: nodes_by_level.into(),
        level_offsets: level_offsets.into(),
        bytes_for_conns0,
        bytes_for_conns,
    })
}

// ---------------------------------------------------------------------------
// Graphs
// ---------------------------------------------------------------------------

/// One retired field's graph, as [`HnswGraphView`].
#[derive(Debug, Clone)]
pub enum RetiredGraph<'a> {
    /// `Lucene90HnswVectorsReader.OffHeapHnswGraph`.
    Lucene90(Lucene90Graph<'a>),
    /// `Lucene9{1,2,4}HnswVectorsReader.OffHeapHnswGraph`.
    Fixed(FixedGraph<'a>),
    /// `Lucene95HnswVectorsReader.OffHeapHnswGraph`, which is
    /// [`OffHeapHnswGraph`] at version 0.
    Lucene95(OffHeapHnswGraph<'a>),
    /// A `Lucene95` field with no graph data (`numLevels == 0`).
    Empty,
}

/// The single-level graph of 9.0: a node's neighbours are an `int` count
/// then `vint` deltas starting from `-1`.
#[derive(Debug, Clone)]
pub struct Lucene90Graph<'a> {
    data: &'a [u8],
    ord_offsets: Arc<[i64]>,
}

/// The fixed-slot graph of 9.1-9.4.
#[derive(Debug, Clone)]
pub struct FixedGraph<'a> {
    data: &'a [u8],
    max_conn: i32,
    num_levels: i32,
    size: i32,
    entry_node: i32,
    nodes_by_level: Arc<[Vec<i32>]>,
    level_offsets: Arc<[i64]>,
    bytes_for_conns0: i64,
    bytes_for_conns: i64,
}

impl Lucene90Graph<'_> {
    fn size(&self) -> i32 {
        // `size` was an `i32` in `.vem`, and this has one entry per ordinal.
        self.ord_offsets.len() as i32
    }

    /// `seek(0, ord)` then the `nextNeighbor` drain.
    fn neighbors_into(&self, node: i32, out: &mut Vec<i32>) -> Result<()> {
        out.clear();
        let Some(&offset) = usize::try_from(node)
            .ok()
            .and_then(|n| self.ord_offsets.get(n))
        else {
            return corrupt(format!("seek target {node} out of range"));
        };
        let mut input = SliceInput::new(self.data);
        // `read_field` bounded every offset by the region's length.
        input.seek(offset as usize)?;
        let arc_count = input.read_i32()?;
        if arc_count < 0 || arc_count >= self.size().max(1) {
            return corrupt(format!("node {node} claims {arc_count} neighbours"));
        }
        let mut arc = -1i64;
        for _ in 0..arc_count {
            // ARITH: `arc` is kept in `-1..size` by the check below and a
            // vint is at most `u32::MAX`, so the sum stays inside `i64`.
            #[allow(clippy::arithmetic_side_effects)]
            {
                arc += i64::from(input.read_vint()? as u32);
            }
            if arc < 0 || arc >= i64::from(self.size()) {
                return corrupt(format!("HNSW neighbour ordinal {arc} out of range"));
            }
            out.push(arc as i32);
        }
        Ok(())
    }
}

impl FixedGraph<'_> {
    /// `seek(level, target)` then the `nextNeighbor` drain.
    fn neighbors_into(&self, level: i32, node: i32, out: &mut Vec<i32>) -> Result<()> {
        out.clear();
        let Some(level_index) = usize::try_from(level)
            .ok()
            .filter(|&l| l < self.nodes_by_level.len())
        else {
            return corrupt(format!("no such HNSW level: {level}"));
        };
        let target_index = if level == 0 {
            if node < 0 || node >= self.size {
                return corrupt(format!("seek target {node} out of range"));
            }
            i64::from(node)
        } else {
            match self.nodes_by_level[level_index].binary_search(&node) {
                Ok(i) => i as i64,
                Err(_) => return corrupt(format!("seek level={level} target={node} not found")),
            }
        };
        let width = if level == 0 {
            self.bytes_for_conns0
        } else {
            self.bytes_for_conns
        };
        // ARITH: `read_fixed_graph` proved `level_offsets[level] + count *
        // width` fits the region for the level's whole node count, and
        // `target_index < count`.
        #[allow(clippy::arithmetic_side_effects)]
        let offset = self.level_offsets[level_index] + target_index * width;
        let mut input = SliceInput::new(self.data);
        input.seek(offset as usize)?;
        let arc_count = input.read_i32()?;
        // The slot holds `width / 4 - 1` neighbours; more is a count that
        // would read into the next node's slot.
        // ARITH: `width` is `(1 + M) * 4` or `(1 + 2M) * 4` with `M >= 1`.
        #[allow(clippy::arithmetic_side_effects)]
        let slot = width / 4 - 1;
        if arc_count < 0 || i64::from(arc_count) > slot {
            return corrupt(format!("node {node} claims {arc_count} neighbours"));
        }
        for _ in 0..arc_count {
            let arc = input.read_i32()?;
            if arc < 0 || arc >= self.size {
                return corrupt(format!("HNSW neighbour ordinal {arc} out of range"));
            }
            out.push(arc);
        }
        Ok(())
    }
}

impl HnswGraphView for RetiredGraph<'_> {
    fn size(&self) -> i32 {
        match self {
            RetiredGraph::Lucene90(g) => g.size(),
            RetiredGraph::Fixed(g) => g.size,
            RetiredGraph::Lucene95(g) => g.size(),
            RetiredGraph::Empty => 0,
        }
    }

    /// `Lucene90`'s graph throws on `numLevels()`; it is one level.
    fn num_levels(&self) -> i32 {
        match self {
            RetiredGraph::Lucene90(_) => 1,
            RetiredGraph::Fixed(g) => g.num_levels,
            RetiredGraph::Lucene95(g) => g.num_levels(),
            RetiredGraph::Empty => 0,
        }
    }

    fn entry_node(&self) -> i32 {
        match self {
            RetiredGraph::Lucene90(_) => 0,
            RetiredGraph::Fixed(g) => g.entry_node,
            RetiredGraph::Lucene95(g) => g.entry_node(),
            // `HnswGraph.EMPTY.entryNode()`; a zero-node graph is never walked.
            RetiredGraph::Empty => 0,
        }
    }

    fn max_conn(&self) -> i32 {
        match self {
            RetiredGraph::Lucene90(_) => UNKNOWN_MAX_CONN,
            RetiredGraph::Fixed(g) => g.max_conn,
            RetiredGraph::Lucene95(g) => g.max_conn(),
            RetiredGraph::Empty => 1,
        }
    }

    fn neighbors_into(&self, level: i32, node: i32, out: &mut Vec<i32>) -> Result<()> {
        match self {
            RetiredGraph::Lucene90(g) if level == 0 => g.neighbors_into(node, out),
            RetiredGraph::Lucene90(_) => corrupt(format!("no such HNSW level: {level}")),
            RetiredGraph::Fixed(g) => g.neighbors_into(level, node, out),
            RetiredGraph::Lucene95(g) => g.neighbors_into(level, node, out),
            RetiredGraph::Empty => corrupt("the field has no graph"),
        }
    }

    fn sorted_nodes_on_level(&self, level: i32) -> Result<Vec<i32>> {
        match self {
            RetiredGraph::Lucene90(g) if level == 0 => Ok((0..g.size()).collect()),
            RetiredGraph::Fixed(g) if level == 0 => Ok((0..g.size).collect()),
            RetiredGraph::Fixed(g) => usize::try_from(level)
                .ok()
                .and_then(|l| g.nodes_by_level.get(l))
                .cloned()
                .ok_or_else(|| Error::CorruptMeta(format!("no such HNSW level: {level}"))),
            RetiredGraph::Lucene95(g) => g.sorted_nodes_on_level(level),
            _ => corrupt(format!("no such HNSW level: {level}")),
        }
    }
}

// ---------------------------------------------------------------------------
// Lucene90 search
// ---------------------------------------------------------------------------

/// `Lucene90OnHeapHnswGraph.search(query, topK = k, numSeed = k, ...)` plus
/// what `Lucene90HnswVectorsReader.search` does with its result: `k` random
/// entry points (distinct draws, at most `2 * size` of them), then a greedy
/// best-first expansion bounded by `Lucene90BoundsChecker.Max`, all on one
/// level. The popped results go to `collector` worst first, as Java's loop
/// hands them to the `KnnCollector`.
// ARITH: `num_visited` counts distinct ordinals of a `size`-node graph, so it
// is at most `size <= i32::MAX`.
#[allow(clippy::arithmetic_side_effects)]
fn lucene90_search<S: VectorScorer, C: KnnCollect + ?Sized>(
    scorer: &mut S,
    graph: &Lucene90Graph<'_>,
    accept_ords: Option<&FixedBitSet>,
    random: &mut SplittableRandom,
    collector: &mut C,
) -> Result<()> {
    // `Lucene90OnHeapHnswGraph.search(target, knnCollector.k(),
    // knnCollector.k(), ..., knnCollector.visitLimit(), random)`.
    let k = collector.k();
    let visited_limit = collector.visit_limit();
    let size = graph.size();
    let mut results = NeighborQueue::new(k, false);
    let mut candidates = NeighborQueue::new(k, true);
    let mut num_visited = 0u64;
    let mut incomplete = false;
    let mut visited = FixedBitSet::new(size as usize);
    // FBS: `search` checked `accept_ords.len() >= size` before calling this,
    // and every ordinal asked about is an entry point (`nextInt(size)`) or a
    // neighbour `Lucene90Graph::neighbors_into` range-checked against `size`.
    let accepted = |ord: i32| accept_ords.is_none_or(|b| b.get(ord as usize));
    let bounded_num_seed = (k as i64).min(2 * i64::from(size));
    for _ in 0..bounded_num_seed {
        let entry_point = random.next_int_bounded(size);
        // `visited.getAndSet(entryPoint) == false`.
        // FBS: `visited` is `FixedBitSet::new(size)` and `nextInt(size)` is in
        // `0..size`.
        if !visited.get(entry_point as usize) {
            // FBS: as above.
            visited.set(entry_point as usize);
            if num_visited >= visited_limit {
                incomplete = true;
                break;
            }
            let score = scorer.score(entry_point)?;
            candidates.add(entry_point, score);
            if accepted(entry_point) {
                results.add(entry_point, score);
            }
            num_visited += 1;
        }
    }
    // `Lucene90BoundsChecker.create(false)` is `Max`: `check(s)` is `s < bound`.
    let mut bound = results.top_score();
    let mut neighbors = Vec::new();
    while candidates.size() > 0 && !incomplete {
        let top_candidate_similarity = candidates.top_score();
        if results.size() >= k && top_candidate_similarity < bound {
            break;
        }
        let top_candidate_node = candidates.pop();
        graph.neighbors_into(top_candidate_node, &mut neighbors)?;
        for &friend in &neighbors {
            // FBS: `neighbors_into` rejects any neighbour outside `0..size`,
            // and `visited` is `FixedBitSet::new(size)`.
            if visited.get(friend as usize) {
                continue;
            }
            // FBS: as above.
            visited.set(friend as usize);
            if num_visited >= visited_limit {
                incomplete = true;
                break;
            }
            let friend_similarity = scorer.score(friend)?;
            if results.size() < k || friend_similarity >= bound {
                candidates.add(friend, friend_similarity);
                if accepted(friend) {
                    results.insert_with_overflow(friend, friend_similarity);
                    bound = results.top_score();
                }
            }
            num_visited += 1;
        }
    }
    while results.size() > k {
        results.pop();
    }
    collector.inc_visited_count(num_visited as usize);
    while results.size() > 0 {
        let score = results.top_score();
        let node = results.pop();
        collector.collect(node, score);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    // The arithmetic gate is about values read off disk; a test's `i + 1` is
    // not one. See docs/arithmetic-gate.md.
    #![allow(clippy::arithmetic_side_effects)]
    use super::*;

    /// The five retired fixture indices of `fixtures/data/bwc` and the format
    /// each one's vector fields use.
    const VERSIONS: [(&str, RetiredHnswFormat); 5] = [
        ("9.0.0", RetiredHnswFormat::Lucene90),
        ("9.1.0", RetiredHnswFormat::Lucene91),
        ("9.3.0", RetiredHnswFormat::Lucene92),
        ("9.4.2", RetiredHnswFormat::Lucene94),
        ("9.8.0", RetiredHnswFormat::Lucene95),
    ];

    struct Files {
        vem: Vec<u8>,
        vec: Vec<u8>,
        vex: Vec<u8>,
        id: [u8; ID_LENGTH],
        suffix: String,
    }

    /// Segment `_1` (200 vectors a field) of `version`, with the segment id
    /// and suffix read back off the `.vem` header itself.
    fn files(version: &str, format: RetiredHnswFormat) -> Files {
        let dir = format!(
            "{}/../../fixtures/data/bwc/{version}/",
            env!("CARGO_MANIFEST_DIR")
        );
        let read = |ext: &str| {
            std::fs::read(format!("{dir}_1_{}_0.{ext}", format.name())).expect("bwc fixture")
        };
        let vem = read("vem");
        let mut input = SliceInput::new(&vem);
        input.read_be_u32().unwrap();
        input.read_string().unwrap();
        input.read_be_u32().unwrap();
        let mut id = [0u8; ID_LENGTH];
        input.read_bytes(&mut id).unwrap();
        let len = input.read_byte().unwrap() as usize;
        let mut suffix = vec![0u8; len];
        input.read_bytes(&mut suffix).unwrap();
        Files {
            vec: read("vec"),
            vex: read("vex"),
            vem,
            id,
            suffix: String::from_utf8(suffix).unwrap(),
        }
    }

    fn open<'a>(
        format: RetiredHnswFormat,
        f: &'a Files,
        vem: &[u8],
    ) -> Result<RetiredHnswVectorsReader<'a>> {
        RetiredHnswVectorsReader::open(format, vem, &f.vec, &f.vex, &f.id, &f.suffix)
    }

    /// Reads everything a reader can serve: every vector, both directions
    /// of the ordinal map, every node's neighbours on every level, and a
    /// search per field.
    fn walk_everything(r: &RetiredHnswVectorsReader<'_>) -> Result<()> {
        for entry in r.flat().fields().to_vec() {
            let n = entry.field_number;
            let graph = r.graph(n)?;
            let mut out = Vec::new();
            for level in 0..graph.num_levels() {
                for node in graph.sorted_nodes_on_level(level)? {
                    graph.neighbors_into(level, node, &mut out)?;
                }
            }
            match entry.encoding {
                VectorEncoding::Float32 => {
                    let v = r.flat().float_vector_values(n)?;
                    let mut cursor = v.doc_to_ord()?;
                    for ord in 0..v.size() {
                        v.vector(ord)?;
                        let doc = v.ord_to_doc(ord)?;
                        if doc < 0 || cursor.ordinal(doc)? != Some(ord) {
                            return corrupt("doc_to_ord disagrees with ord_to_doc");
                        }
                    }
                    let q = vec![0.25f32; v.dimension()];
                    r.search(n, &mut v.scorer(&q)?, 10, u64::MAX, None, None)?;
                }
                VectorEncoding::Byte => {
                    let v = r.flat().byte_vector_values(n)?;
                    for ord in 0..v.size() {
                        v.vector(ord)?;
                        v.ord_to_doc(ord)?;
                    }
                    let q = vec![3u8; v.dimension()];
                    r.search(n, &mut v.scorer(&q)?, 10, u64::MAX, None, None)?;
                }
            }
        }
        Ok(())
    }

    #[test]
    fn format_names_round_trip_and_keep_their_spelling() {
        for f in RetiredHnswFormat::ALL {
            assert_eq!(RetiredHnswFormat::for_name(f.name()), Some(f));
        }
        assert_eq!(
            RetiredHnswFormat::Lucene92.name(),
            "lucene92HnswVectorsFormat"
        );
        assert_eq!(
            RetiredHnswFormat::Lucene94.codec_prefix(),
            "lucene94HnswVectorsFormat"
        );
        assert_eq!(
            RetiredHnswFormat::for_name("Lucene99HnswVectorsFormat"),
            None
        );
        assert_eq!(
            RetiredHnswFormat::for_name("Lucene92HnswVectorsFormat"),
            None
        );
        assert!(RetiredHnswFormat::Lucene90.neighbors_sorted());
        assert!(!RetiredHnswFormat::Lucene91.neighbors_sorted());
        assert!(RetiredHnswFormat::Lucene95.neighbors_sorted());
    }

    #[test]
    fn every_retired_fixture_opens_walks_and_searches() {
        for (version, format) in VERSIONS {
            let f = files(version, format);
            let r = open(format, &f, &f.vem).unwrap_or_else(|e| panic!("{version}: {e}"));
            assert_eq!(r.format(), format);
            assert_eq!(r.hits_are_ordinals(), format == RetiredHnswFormat::Lucene90);
            walk_everything(&r).unwrap_or_else(|e| panic!("{version}: {e}"));
            let fields = r.flat().fields();
            assert!(!fields.is_empty());
            for e in fields {
                assert_eq!(e.size, 200, "{version}: every other of 400 docs has one");
                assert_eq!(r.graph(e.field_number).unwrap().size(), 200);
            }
            assert!(matches!(r.graph(999), Err(Error::UnknownField(999))));
        }
    }

    /// Graph shape per format: `Lucene90` is one level with an unknown `M`;
    /// the others are hierarchical, and a seek off a level is an error.
    #[test]
    fn graphs_report_their_shape_and_reject_bad_seeks() {
        for (version, format) in VERSIONS {
            let f = files(version, format);
            let r = open(format, &f, &f.vem).unwrap();
            let n = r.flat().fields()[0].field_number;
            let g = r.graph(n).unwrap();
            let mut out = Vec::new();
            assert!(g.neighbors_into(0, -1, &mut out).is_err(), "{version}");
            assert!(g.neighbors_into(0, 200, &mut out).is_err(), "{version}");
            assert!(
                g.neighbors_into(g.num_levels(), 0, &mut out).is_err(),
                "{version}"
            );
            assert!(
                g.sorted_nodes_on_level(g.num_levels()).is_err(),
                "{version}"
            );
            if format == RetiredHnswFormat::Lucene90 {
                assert_eq!(
                    (g.num_levels(), g.entry_node(), g.max_conn()),
                    (1, 0, UNKNOWN_MAX_CONN)
                );
            } else {
                assert!(g.num_levels() > 1, "{version}");
                assert!(g.max_conn() >= 16, "{version}");
                let top = g.sorted_nodes_on_level(g.num_levels() - 1).unwrap();
                assert_eq!(g.entry_node(), top[0]);
                // A node that is not on the top level cannot be sought there.
                let absent = (0..200).find(|x| !top.contains(x)).unwrap();
                assert!(g
                    .neighbors_into(g.num_levels() - 1, absent, &mut out)
                    .is_err());
            }
        }
    }

    #[test]
    fn search_edge_cases() {
        for (version, format) in VERSIONS {
            let f = files(version, format);
            let r = open(format, &f, &f.vem).unwrap();
            let n = r
                .flat()
                .fields()
                .iter()
                .find(|e| e.encoding == VectorEncoding::Float32)
                .unwrap()
                .field_number;
            let v = r.flat().float_vector_values(n).unwrap();
            let q = vec![0.5f32; v.dimension()];
            let mut scorer = v.scorer(&q).unwrap();
            // k = 0 collects nothing.
            assert!(r
                .search(n, &mut scorer, 0, u64::MAX, None, None)
                .unwrap()
                .0
                .is_empty());
            // A short accept set is a caller error, not a panic.
            let short = FixedBitSet::new(10);
            assert!(matches!(
                r.search(n, &mut scorer, 5, u64::MAX, Some(&short), None),
                Err(Error::InvalidGraphParameter(_))
            ));
            // Accepting only even ordinals returns only even ordinals.
            let mut even = FixedBitSet::new(200);
            for o in (0..200).step_by(2) {
                even.set(o);
            }
            let (hits, early) = r
                .search(n, &mut scorer, 5, u64::MAX, Some(&even), None)
                .unwrap();
            assert!(!early);
            assert_eq!(hits.len(), 5, "{version}");
            assert!(hits.iter().all(|(o, _)| o % 2 == 0), "{version}: {hits:?}");
            assert!(hits.windows(2).all(|w| w[0].1 >= w[1].1));
            // A visit limit of one stops the walk early.
            let (_, early) = r.search(n, &mut scorer, 5, 1, None, None).unwrap();
            assert!(early, "{version}");
            // Seeds reach the hierarchical formats; Lucene90 ignores them.
            let (seeded, _) = r
                .search(n, &mut scorer, 5, u64::MAX, None, Some(&[1, 2, 3]))
                .unwrap();
            assert_eq!(seeded.len(), 5);
        }
    }

    /// Every single-bit flip of each retired `.vem` (and of the two smallest
    /// `.vex` files), re-signed so the checksum passes, either decodes
    /// cleanly or is rejected with an error -- never a panic.
    #[test]
    fn every_resigned_single_byte_corruption_is_an_error_or_a_clean_decode() {
        fn repack(buf: &[u8], at: usize, bit: u8) -> Vec<u8> {
            let mut body = buf[..buf.len() - codec_util::FOOTER_LENGTH].to_vec();
            body[at] ^= 1 << bit;
            codec_util::write_footer(&mut body);
            body
        }
        let mut meta_rejected = 0usize;
        let mut meta_flipped = 0usize;
        for (version, format) in VERSIONS {
            let f = files(version, format);
            for at in 0..f.vem.len() - codec_util::FOOTER_LENGTH {
                for bit in [0u8, 7] {
                    let vem = repack(&f.vem, at, bit);
                    meta_flipped += 1;
                    let bad = open(format, &f, &vem)
                        .and_then(|r| walk_everything(&r))
                        .is_err();
                    meta_rejected += usize::from(bad);
                }
            }
            if matches!(
                format,
                RetiredHnswFormat::Lucene90 | RetiredHnswFormat::Lucene95
            ) {
                for at in 0..f.vex.len() - codec_util::FOOTER_LENGTH {
                    let vex = repack(&f.vex, at, 0);
                    let _ = RetiredHnswVectorsReader::open(
                        format, &f.vem, &f.vec, &vex, &f.id, &f.suffix,
                    )
                    .and_then(|r| walk_everything(&r));
                }
            }
            let _ = version;
        }
        assert!(
            meta_rejected > meta_flipped / 2,
            "only {meta_rejected} of {meta_flipped} .vem flips rejected"
        );
    }

    #[test]
    fn mismatched_headers_and_short_files_are_rejected() {
        let (version, format) = VERSIONS[1];
        let f = files(version, format);
        // The wrong format for the files.
        assert!(open(RetiredHnswFormat::Lucene90, &f, &f.vem).is_err());
        // A data file that is not this format's.
        assert!(
            RetiredHnswVectorsReader::open(format, &f.vem, &f.vex, &f.vec, &f.id, &f.suffix)
                .is_err()
        );
        // A wrong segment id.
        assert!(RetiredHnswVectorsReader::open(
            format,
            &f.vem,
            &f.vec,
            &f.vex,
            &[0; ID_LENGTH],
            &f.suffix
        )
        .is_err());
        // Truncation.
        assert!(open(format, &f, &f.vem[..f.vem.len() - 1]).is_err());
        assert!(RetiredHnswVectorsReader::open(
            format,
            &f.vem,
            &f.vec[..8],
            &f.vex,
            &f.id,
            &f.suffix
        )
        .is_err());
        assert!(check_region(-1, 1, 10).is_err());
        assert!(check_region(5, 6, 10).is_err());
        assert!(check_region(5, 5, 10).is_ok());
    }
}
