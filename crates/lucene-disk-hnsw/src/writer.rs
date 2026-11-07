//! Writing a segment's disk HNSW files.
//!
//! `.dhm`, after its index header: the field count (vint), then per field:
//!
//! - number (vint), similarity ordinal (byte), dimension (vint), vector
//!   encoding ordinal (byte, `VectorEncoding::ordinal`), the quantizer's
//!   name (string) and state (vint length, bytes), the field's mean vector
//!   (dimension x f32 LE; the quantizer's centroid and the `int8`
//!   encoding's reference);
//! - node count (vint), `M` (vint), page length (vint), the most
//!   neighbours a record holds (vint), inline codes (byte, 0 or 1), the
//!   field's offsets in `.dhn` and `.dhc` (vlongs);
//! - the entry node plus one (vint, 0 for an empty field), the level count
//!   including level 0 (vint), then per upper level, lowest first: its node
//!   count (vint) and per node, ascending, the gap from the previous node
//!   (vint), its neighbour count (vint) and its neighbours (vints);
//! - the upper nodes (every node above level 0): their count (vint) and
//!   their codes in ascending node order (count x code length bytes).
//!
//! `.dhc` holds each field's codes in node order, `.dhn` each field's
//! records ([`crate::layout`]) from a page boundary; the gaps are zero.

use lucene_codecs::hnsw::HnswGraphView;
use lucene_store::codec_util::{self, ID_LENGTH};
use lucene_store::DataOutput;
use lucene_vector_quant::QuantizerFactory;
use rayon::prelude::*;

use crate::graph::{self, GraphParams};
use crate::layout::{NodeLayout, DEFAULT_PAGE_LEN};
use crate::{
    similarity_ordinal, Error, Result, Similarity, VectorEncoding, CODES_CODEC, MAX_LEVELS,
    META_CODEC, NODES_CODEC, VERSION_CURRENT,
};

/// How a field is written.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiskHnswParams {
    /// The graph built by [`DiskHnswWriter::add_field`].
    pub graph: GraphParams,
    /// Bytes of a page; records never cross one. Default 4096.
    pub page_len: usize,
    /// How records store the full-precision vectors. Default `f32`.
    pub encoding: VectorEncoding,
    /// Store each neighbour's code in the record too (LM-DiskANN), so a
    /// reader can search without the per-node codes in memory, at the cost
    /// of `2 * M` codes per record on disk. Default off.
    pub inline_codes: bool,
    /// Number nodes breadth first from the entry node, so neighbours tend
    /// to share pages. Default on.
    pub reorder: bool,
    /// Vectors the quantizer is trained on (an even stride over the field).
    pub training_sample: usize,
}

impl Default for DiskHnswParams {
    fn default() -> Self {
        DiskHnswParams {
            graph: GraphParams::default(),
            page_len: DEFAULT_PAGE_LEN,
            encoding: VectorEncoding::F32,
            inline_codes: false,
            reorder: true,
            training_sample: 25_000,
        }
    }
}

/// One field's vectors, one per document.
#[derive(Debug, Clone, Copy)]
pub struct FieldVectors<'a> {
    /// The field number.
    pub number: i32,
    /// How vectors are compared.
    pub similarity: Similarity,
    /// Floats per vector.
    pub dimension: usize,
    /// The documents, strictly ascending.
    pub docs: &'a [i32],
    /// The vectors, `docs.len()` rows of `dimension` floats.
    pub vectors: &'a [f32],
}

/// The bytes of the three files.
#[derive(Debug, Clone)]
pub struct DiskHnswFiles {
    /// `.dhm`.
    pub meta: Vec<u8>,
    /// `.dhc`.
    pub codes: Vec<u8>,
    /// `.dhn`.
    pub nodes: Vec<u8>,
}

/// Writes a segment's fields into the three files.
#[derive(Debug)]
pub struct DiskHnswWriter {
    meta: Vec<u8>,
    fields: Vec<u8>,
    numbers: Vec<i32>,
    codes: Vec<u8>,
    nodes: Vec<u8>,
}

fn bad<T>(m: String) -> Result<T> {
    Err(Error::IllegalArgument(m))
}

/// Writes a count that fits an int: the field's vector count, dimension,
/// page length and M are checked to, and every other count is bounded by
/// one of them.
fn write_count(out: &mut Vec<u8>, v: usize) {
    out.write_vint(i32::try_from(v).unwrap_or(i32::MAX));
}

fn write_offset(out: &mut Vec<u8>, v: usize) {
    // In-memory offsets fit an i64.
    out.write_vlong(i64::try_from(v).unwrap_or(i64::MAX));
}

/// The node ids of `neighbors` (old ids) renumbered by `rank`, ascending:
/// the order stock `.vex` files keep, so a graph read from one writes the
/// same bytes as the graph it was written from.
fn renumber(neighbors: &[i32], rank: &[u32], size: usize) -> Result<Vec<u32>> {
    let mut out: Vec<u32> = neighbors
        .iter()
        .map(|&nb| {
            usize::try_from(nb)
                .ok()
                .filter(|&nb| nb < size)
                .map(|nb| rank[nb])
                .ok_or_else(|| Error::Graph(format!("neighbour {nb} outside the graph")))
        })
        .collect::<Result<_>>()?;
    out.sort_unstable();
    Ok(out)
}

/// `Cosine` vectors scaled to unit length, which keeps every score, so the
/// mean and the codes are taken over directions; `None` for the other
/// similarities, whose vectors are stored as they are.
fn unit_vectors(field: &FieldVectors<'_>) -> Option<Vec<f32>> {
    if field.similarity != Similarity::Cosine {
        return None;
    }
    let mut out = field.vectors.to_vec();
    out.par_chunks_mut(field.dimension).for_each(|v| {
        // A zero vector has no direction and stays zero.
        let _ = lucene_util::vector_util::l2normalize(v, false);
    });
    Some(out)
}

/// The mean of `vectors` (rows of `dim` floats), summed in `f64`.
fn mean(vectors: &[f32], dim: usize) -> Vec<f32> {
    let mut sum = vec![0f64; dim];
    let mut rows = 0u32;
    for row in vectors.chunks_exact(dim) {
        for (s, &x) in sum.iter_mut().zip(row) {
            *s += f64::from(x);
        }
        rows = rows.saturating_add(1);
    }
    let rows = f64::from(rows.max(1));
    // A mean of f32 values is inside the f32 range.
    #[allow(clippy::cast_possible_truncation)]
    sum.iter().map(|s| (s / rows) as f32).collect()
}

/// Every `stride`-th row of `vectors`, at most `count` of them.
fn sample(vectors: &[f32], dim: usize, count: usize) -> Vec<f32> {
    let rows = vectors.len().checked_div(dim).unwrap_or(0);
    let stride = rows.div_ceil(count.max(1)).max(1);
    vectors
        .chunks_exact(dim)
        .step_by(stride)
        .flatten()
        .copied()
        .collect()
}

impl DiskHnswWriter {
    /// A writer for the segment `segment_id`, its files named with `suffix`.
    pub fn new(segment_id: &[u8; ID_LENGTH], suffix: &str) -> Self {
        let mut meta = Vec::new();
        let mut codes = Vec::new();
        let mut nodes = Vec::new();
        codec_util::write_index_header(&mut meta, META_CODEC, VERSION_CURRENT, segment_id, suffix);
        codec_util::write_index_header(
            &mut codes,
            CODES_CODEC,
            VERSION_CURRENT,
            segment_id,
            suffix,
        );
        codec_util::write_index_header(
            &mut nodes,
            NODES_CODEC,
            VERSION_CURRENT,
            segment_id,
            suffix,
        );
        DiskHnswWriter {
            meta,
            fields: Vec::new(),
            numbers: Vec::new(),
            codes,
            nodes,
        }
    }

    fn check(&self, field: &FieldVectors<'_>, params: &DiskHnswParams) -> Result<()> {
        if self.numbers.contains(&field.number) {
            return bad(format!("field {} written twice", field.number));
        }
        if field.dimension == 0 {
            return bad("dimension must be positive".into());
        }
        if Some(field.vectors.len()) != field.docs.len().checked_mul(field.dimension) {
            return bad(format!(
                "{} floats for {} documents of dimension {}",
                field.vectors.len(),
                field.docs.len(),
                field.dimension
            ));
        }
        if field.docs.first().is_some_and(|&d| d < 0) || field.docs.windows(2).any(|w| w[0] >= w[1])
        {
            return bad("documents must be non-negative and strictly ascending".into());
        }
        if field.vectors.iter().any(|x| !x.is_finite()) {
            return bad("vectors must be finite".into());
        }
        if i32::try_from(field.docs.len()).is_err() || i32::try_from(field.dimension).is_err() {
            return bad("too many vectors or dimensions".into());
        }
        if params.page_len == 0
            || !params.page_len.is_multiple_of(4)
            || i32::try_from(params.page_len).is_err()
        {
            return bad(format!(
                "page length {} is not a positive multiple of 4 that fits an int",
                params.page_len
            ));
        }
        Ok(())
    }

    /// Builds Lucene's HNSW graph over `field` ([`graph::build_graph`] with
    /// `params.graph`), trains a quantizer from `factory` and writes the
    /// field.
    ///
    /// # Errors
    ///
    /// As [`DiskHnswWriter::add_field_with_graph`], and the graph build's.
    pub fn add_field(
        &mut self,
        field: &FieldVectors<'_>,
        factory: &dyn QuantizerFactory,
        params: &DiskHnswParams,
    ) -> Result<()> {
        self.check(field, params)?;
        let normalized = unit_vectors(field);
        let vectors = normalized.as_deref().unwrap_or(field.vectors);
        let graph = graph::build_graph(vectors, field.dimension, field.similarity, &params.graph)?;
        self.write_field(field, vectors, &graph, factory, params)
    }

    /// Writes `field` with `graph`, whose ordinal `i` is the field's `i`-th
    /// vector: a graph just built, or a stock segment's
    /// (`lucene_codecs::hnsw_vectors`), so a segment converts without a
    /// rebuild. `params.graph` is not used.
    ///
    /// # Errors
    ///
    /// [`Error::IllegalArgument`] for a field written twice, a zero
    /// dimension, a vector count that does not match the documents,
    /// documents out of order, a non-finite value, a value the encoding
    /// cannot hold, a bad page length, or a graph of another size;
    /// [`Error::Graph`] for a graph whose neighbours fall outside it or
    /// exceed `2 * M` on level 0; the quantizer's training errors.
    pub fn add_field_with_graph<G: HnswGraphView>(
        &mut self,
        field: &FieldVectors<'_>,
        graph: &G,
        factory: &dyn QuantizerFactory,
        params: &DiskHnswParams,
    ) -> Result<()> {
        self.check(field, params)?;
        let normalized = unit_vectors(field);
        let vectors = normalized.as_deref().unwrap_or(field.vectors);
        self.write_field(field, vectors, graph, factory, params)
    }

    /// Writes a checked `field` whose vectors, as stored, are `vectors`.
    fn write_field(
        &mut self,
        field: &FieldVectors<'_>,
        vectors: &[f32],
        graph: &dyn HnswGraphView,
        factory: &dyn QuantizerFactory,
        params: &DiskHnswParams,
    ) -> Result<()> {
        let dim = field.dimension;
        let n = field.docs.len();
        if usize::try_from(graph.size()).ok() != Some(n) {
            return bad(format!("a graph of {} nodes for {n} vectors", graph.size()));
        }
        let max_conn = usize::try_from(graph.max_conn())
            .ok()
            .filter(|&m| m > 0)
            .ok_or_else(|| Error::Graph(format!("M {}", graph.max_conn())))?;
        let max_neighbors = max_conn
            .checked_mul(2)
            .filter(|&m| m <= usize::from(u16::MAX))
            .ok_or_else(|| Error::Graph(format!("M {max_conn}")))?;
        let centroid = mean(vectors, dim);
        let quantizer = factory.train(
            dim,
            field.similarity,
            &sample(vectors, dim, params.training_sample),
        )?;
        let prepared = quantizer
            .prepare_centroid(&centroid)
            .unwrap_or_else(|| centroid.clone());
        let code_len = quantizer.code_len();
        let mut old_codes = vec![0u8; n.checked_mul(code_len).ok_or_else(overflow)?];
        if code_len > 0 {
            old_codes
                .par_chunks_mut(code_len)
                .zip(vectors.par_chunks(dim))
                .for_each(|(out, v)| quantizer.encode_prepared(v, &prepared, out));
        }
        let code = |old: usize| -> &[u8] {
            // ARITH: `old < n`, and `old_codes` holds `n` codes.
            #[allow(clippy::arithmetic_side_effects)]
            let start = old * code_len;
            &old_codes[start..][..code_len]
        };

        let order: Vec<u32> = if params.reorder {
            graph::breadth_first_order(graph, n)?
        } else {
            // `n` fits `i32`, so every id fits `u32`.
            (0..n)
                .map(|i| u32::try_from(i).unwrap_or(u32::MAX))
                .collect()
        };
        let mut rank = vec![0u32; n];
        for (new, &old) in order.iter().enumerate() {
            rank[old as usize] = u32::try_from(new).unwrap_or(u32::MAX);
        }

        // Level-0 adjacency, renumbered, by new id.
        let mut counts = vec![0u16; n];
        let mut adjacency = vec![0u32; n.checked_mul(max_neighbors).ok_or_else(overflow)?];
        let mut scratch = Vec::new();
        for (new, &old) in order.iter().enumerate() {
            graph.neighbors_into(0, i32::try_from(old).unwrap_or(i32::MAX), &mut scratch)?;
            if scratch.len() > max_neighbors {
                return Err(Error::Graph(format!(
                    "node {old} has {} level-0 neighbours, more than 2 * M = {max_neighbors}",
                    scratch.len()
                )));
            }
            let nbs = renumber(&scratch, &rank, n)?;
            // ARITH: `new < n`; the slice holds `max_neighbors` per node.
            #[allow(clippy::arithmetic_side_effects)]
            adjacency[new * max_neighbors..][..nbs.len()].copy_from_slice(&nbs);
            // At most `max_neighbors`, which fits u16.
            counts[new] = u16::try_from(nbs.len()).unwrap_or(u16::MAX);
        }

        let vector_bytes = params
            .encoding
            .vector_bytes(dim)
            .ok_or_else(|| Error::IllegalArgument(format!("dimension {dim} overflows")))?;
        let layout = NodeLayout::new(
            params.page_len,
            vector_bytes,
            max_neighbors,
            if params.inline_codes { code_len } else { 0 },
        )
        .ok_or_else(overflow)?;

        // Records, by page group, in parallel. Nothing is added to the
        // writer's buffers until the whole field has been encoded, so a
        // field that fails leaves the files as they were.
        let page_start = self.nodes.len().next_multiple_of(params.page_len);
        let region_len = layout.region_len(n).ok_or_else(overflow)?;
        let mut region = vec![0u8; region_len];
        let group_len = layout.page_len().saturating_mul(layout.pages_per_record());
        let per_group = layout.records_per_page();
        let encoding = params.encoding;
        let finite = region
            .par_chunks_mut(group_len.max(1))
            .enumerate()
            .map(|(group, bytes)| {
                let first = group.saturating_mul(per_group);
                let last = first.saturating_add(per_group).min(n);
                (first..last).all(|new| {
                    // ARITH: `new - first < records_per_page`, so the record
                    // lies inside its page group.
                    #[allow(clippy::arithmetic_side_effects)]
                    let rec =
                        &mut bytes[(new - first) * layout.record_len()..][..layout.record_len()];
                    let old = order[new] as usize;
                    rec[..4].copy_from_slice(&field.docs[old].to_le_bytes());
                    rec[4..6].copy_from_slice(&counts[new].to_le_bytes());
                    // ARITH: `old < n`; rows are `dim` floats.
                    #[allow(clippy::arithmetic_side_effects)]
                    let v = &vectors[old * dim..][..dim];
                    let ok = encoding.encode(
                        v,
                        &centroid,
                        &mut rec[layout.vector_start()..layout.neighbors_start()],
                    );
                    let count = usize::from(counts[new]);
                    // ARITH: `new < n`; the slice holds `max_neighbors` per node.
                    #[allow(clippy::arithmetic_side_effects)]
                    let nbs = &adjacency[new * max_neighbors..][..count];
                    let at = layout.neighbors_start();
                    for (o, nb) in rec[at..].chunks_exact_mut(4).zip(nbs) {
                        o.copy_from_slice(&nb.to_le_bytes());
                    }
                    if layout.inline_code_len() > 0 {
                        let at = layout.codes_start();
                        for (o, &nb) in rec[at..].chunks_exact_mut(code_len).zip(nbs) {
                            o.copy_from_slice(code(order[nb as usize] as usize));
                        }
                    }
                    ok
                })
            })
            .reduce(|| true, |a, b| a && b);
        if !finite {
            return bad(format!(
                "a vector does not fit the {} encoding",
                params.encoding.name()
            ));
        }

        // Upper levels, renumbered.
        let levels = usize::try_from(graph.num_levels()).unwrap_or(0).max(1);
        if levels > MAX_LEVELS {
            return Err(Error::Graph(format!(
                "{levels} levels, more than {MAX_LEVELS}"
            )));
        }
        let mut upper_levels = Vec::new();
        let mut upper_nodes = Vec::new();
        for level in 1..levels {
            let level_i32 = i32::try_from(level).unwrap_or(i32::MAX);
            let mut nodes: Vec<(u32, i32)> = graph
                .sorted_nodes_on_level(level_i32)?
                .into_iter()
                .map(|old| {
                    usize::try_from(old)
                        .ok()
                        .filter(|&o| o < n)
                        .map(|o| (rank[o], old))
                        .ok_or_else(|| Error::Graph(format!("node {old} outside the graph")))
                })
                .collect::<Result<_>>()?;
            nodes.sort_unstable();
            let mut adj = Vec::with_capacity(nodes.len());
            for &(new, old) in &nodes {
                graph.neighbors_into(level_i32, old, &mut scratch)?;
                if scratch.len() > max_neighbors {
                    return Err(Error::Graph(format!(
                        "node {old} has {} level-{level} neighbours, more than 2 * M = \
                         {max_neighbors}",
                        scratch.len()
                    )));
                }
                adj.push((new, renumber(&scratch, &rank, n)?));
                upper_nodes.push(new);
            }
            upper_levels.push(adj);
        }
        upper_nodes.sort_unstable();
        upper_nodes.dedup();
        for adj in &upper_levels {
            for (_, nbs) in adj {
                if nbs.iter().any(|nb| upper_nodes.binary_search(nb).is_err()) {
                    return Err(Error::Graph(
                        "an upper-level neighbour is not on an upper level".into(),
                    ));
                }
            }
        }

        let codes_start = self.codes.len();
        let mut field_meta = Vec::new();
        let out = &mut field_meta;
        out.write_vint(field.number);
        out.write_byte(similarity_ordinal(field.similarity));
        write_count(out, dim);
        out.write_byte(params.encoding.ordinal());
        out.write_string(quantizer.name());
        let state = quantizer.state();
        let state_len = i32::try_from(state.len())
            .map_err(|_| Error::IllegalArgument("the quantizer state is too large".into()))?;
        out.write_vint(state_len);
        out.write_bytes(&state);
        for c in &centroid {
            out.write_bytes(&c.to_le_bytes());
        }
        write_count(out, n);
        write_count(out, max_conn);
        write_count(out, params.page_len);
        write_count(out, max_neighbors);
        out.write_byte(u8::from(params.inline_codes));
        write_offset(out, page_start);
        write_offset(out, codes_start);
        let entry = usize::try_from(graph.entry_node())
            .ok()
            .filter(|&e| e < n)
            .map_or(0, |e| (rank[e] as usize).saturating_add(1));
        write_count(out, if n == 0 { 0 } else { entry.max(1) });
        write_count(out, levels);
        for adj in &upper_levels {
            write_count(out, adj.len());
            let mut prev = 0u32;
            for (node, nbs) in adj {
                write_count(out, node.saturating_sub(prev) as usize);
                prev = *node;
                write_count(out, nbs.len());
                for &nb in nbs {
                    write_count(out, nb as usize);
                }
            }
        }
        write_count(out, upper_nodes.len());
        for &node in &upper_nodes {
            out.write_bytes(code(order[node as usize] as usize));
        }

        self.nodes.resize(page_start, 0);
        self.nodes.extend_from_slice(&region);
        for &old in &order {
            self.codes.extend_from_slice(code(old as usize));
        }
        self.fields.extend_from_slice(&field_meta);
        self.numbers.push(field.number);
        Ok(())
    }

    /// The finished files.
    pub fn finish(mut self) -> DiskHnswFiles {
        // At most `i32::MAX` fields can be added before memory runs out.
        self.meta
            .write_vint(i32::try_from(self.numbers.len()).unwrap_or(i32::MAX));
        self.meta.extend_from_slice(&self.fields);
        codec_util::write_footer(&mut self.meta);
        codec_util::write_footer(&mut self.codes);
        codec_util::write_footer(&mut self.nodes);
        DiskHnswFiles {
            meta: self.meta,
            codes: self.codes,
            nodes: self.nodes,
        }
    }
}

fn overflow() -> Error {
    Error::IllegalArgument("the field is too large".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samples_and_means() {
        let v: Vec<f32> = (0..10).map(|i| i as f32).collect();
        assert_eq!(sample(&v, 1, 5), vec![0.0, 2.0, 4.0, 6.0, 8.0]);
        assert_eq!(sample(&v, 1, 100).len(), 10);
        assert_eq!(mean(&v, 2), vec![4.0, 5.0]);
        assert_eq!(mean(&[], 2), vec![0.0, 0.0]);
    }

    #[test]
    fn renumbering_refuses_outside_ids() {
        assert_eq!(renumber(&[1, 0], &[6, 5], 2).unwrap(), vec![5, 6]);
        assert!(renumber(&[2], &[5, 6], 2).is_err());
        assert!(renumber(&[-1], &[5, 6], 2).is_err());
    }
}
