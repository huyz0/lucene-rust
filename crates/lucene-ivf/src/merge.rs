//! Merging IVF fields: the merged segment's vectors are re-quantized and
//! written afresh, never patched in place. Their clusters either come from
//! a full re-clustering, or, when one input segment holds most of the
//! merged vectors, from that segment's clusters rebalanced locally
//! ([`MergeClustering::ReuseLargest`], [`kmeans::reuse`]): re-clustering a
//! segment of millions of vectors costs a few full Lloyd rounds of
//! `n * k * dim`, while the clusters of the large input already fit most of
//! its data.

use lucene_vector_quant::QuantizerFactory;

use crate::kmeans::{self, NO_CLUSTER};
use crate::reader::{IvfField, IvfReader};
use crate::writer::{FieldVectors, IvfParams, IvfWriter};
use crate::{Error, Result};

/// How a merge finds the merged segment's clusters
/// ([`IvfParams::merge_clustering`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MergeClustering {
    /// Cluster every live vector from scratch, as a fresh build does.
    Recluster,
    /// Start from the clusters of the source with the most live vectors when
    /// it holds at least `min_fraction` of the merged ones and has two or
    /// more clusters: its vectors keep their cluster, the others go to their
    /// nearest one, and the partition is rebalanced with `refine_rounds`
    /// neighbourhood rounds, splits and dissolves ([`kmeans::reuse`]).
    /// Otherwise [`MergeClustering::Recluster`].
    ReuseLargest {
        /// The least share of the merged live vectors, in `0.0..=1.0`, the
        /// largest source must hold.
        min_fraction: f32,
        /// Neighbourhood Lloyd rounds over the clusters that received
        /// vectors and their neighbours.
        refine_rounds: usize,
    },
}

impl Default for MergeClustering {
    /// [`MergeClustering::ReuseLargest`] for a source of at least half the
    /// vectors, with one refinement round.
    fn default() -> Self {
        MergeClustering::ReuseLargest {
            min_fraction: 0.5,
            refine_rounds: 1,
        }
    }
}

/// One merged segment's field: its reader and how its documents map into
/// the merged segment (`None` for a deleted document).
pub struct MergeSource<'a> {
    /// The input segment.
    pub reader: &'a IvfReader,
    /// Old document to new document, `None` when the document is dropped.
    pub doc_map: &'a dyn Fn(i32) -> Option<i32>,
}

impl std::fmt::Debug for MergeSource<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MergeSource")
            .field("reader", self.reader)
            .finish_non_exhaustive()
    }
}

/// The source whose clusters `policy` reuses for the `live` (new doc,
/// source, slot) vectors of field `number`, if any: its index, its field
/// and the refinement rounds.
// ARITH: `counts` has one entry per source and the counts sum to `live.len()`.
#[allow(clippy::arithmetic_side_effects)]
fn reused_source<'a>(
    policy: MergeClustering,
    number: i32,
    sources: &[MergeSource<'a>],
    live: &[(i32, usize, u32)],
) -> Option<(usize, &'a IvfField, usize)> {
    let MergeClustering::ReuseLargest {
        min_fraction,
        refine_rounds,
    } = policy
    else {
        return None;
    };
    let mut counts = vec![0usize; sources.len()];
    for &(_, s, _) in live {
        counts[s] += 1;
    }
    // The first of equally large sources.
    let (largest, &count) = counts
        .iter()
        .enumerate()
        .max_by_key(|&(i, &c)| (c, std::cmp::Reverse(i)))?;
    let field = sources[largest].reader.field(number)?;
    (count > 0
        && field.clusters() > 1
        && count as f64 >= f64::from(min_fraction) * live.len() as f64)
        .then_some((largest, field, refine_rounds))
}

/// Writes field `number` of the merged segment into `writer`: the live
/// vectors of every source that has the field, under their new document
/// numbers, clustered as `params.merge_clustering` says and quantized
/// afresh with `factory`. Sources may store their vectors in any
/// [`crate::VectorEncoding`]; they are decoded, and written in
/// `params.rerank_encoding`.
///
/// # Errors
///
/// Sources that disagree on the field's similarity or dimension, two
/// sources mapping onto the same document, a read failure, or the writer's
/// own errors. A field no source has is not written.
pub fn merge_field(
    writer: &mut IvfWriter,
    number: i32,
    sources: &[MergeSource<'_>],
    factory: &dyn QuantizerFactory,
    params: &IvfParams,
) -> Result<()> {
    let mut shape = None;
    // (new doc, source, slot) for every live vector.
    let mut live: Vec<(i32, usize, u32)> = Vec::new();
    for (i, source) in sources.iter().enumerate() {
        let Some(field) = source.reader.field(number) else {
            continue;
        };
        let this = (field.similarity(), field.dimension());
        if *shape.get_or_insert(this) != this {
            return Err(Error::IllegalArgument(format!(
                "field {number}: sources disagree on similarity or dimension"
            )));
        }
        for (doc, slot) in source.reader.doc_table(number)? {
            if let Some(new_doc) = (source.doc_map)(doc) {
                live.push((new_doc, i, slot));
            }
        }
    }
    let Some((similarity, dimension)) = shape else {
        return Ok(());
    };
    live.sort_unstable_by_key(|&(d, _, _)| d);
    if live.windows(2).any(|w| w[0].0 == w[1].0) {
        return Err(Error::IllegalArgument(format!(
            "field {number}: two sources map onto the same document"
        )));
    }
    let docs: Vec<i32> = live.iter().map(|&(d, _, _)| d).collect();
    let mut vectors = vec![0f32; live.len().saturating_mul(dimension)];
    // One sequential read per source rather than one per vector.
    for (i, source) in sources.iter().enumerate() {
        let Some(field) = source.reader.field(number) else {
            continue;
        };
        let (encoding, stride) = (field.encoding(), field.vector_len());
        let bytes = source.reader.slot_vectors(number)?;
        // ARITH: slots were checked against the field's count at open, so
        // `(slot + 1) * stride` bytes lie inside `bytes`, and `out` indexes
        // `live`, whose length sized `vectors`.
        #[allow(clippy::arithmetic_side_effects)]
        for (out, &(_, _, slot)) in live.iter().enumerate().filter(|(_, l)| l.1 == i) {
            let from = slot as usize * stride;
            encoding.decode(
                &bytes[from..from + stride],
                field.slot_centroid(slot as usize),
                &mut vectors[out * dimension..(out + 1) * dimension],
            );
        }
    }
    let field = FieldVectors {
        number,
        similarity,
        dimension,
        docs: &docs,
        vectors: &vectors,
    };
    let Some((base, base_field, refine_rounds)) =
        reused_source(params.merge_clustering, number, sources, &live)
    else {
        return writer.add_field(&field, factory, params);
    };
    let slots = sources[base].reader.slot_clusters(number)?;
    let hints: Vec<[u32; 2]> = live
        .iter()
        .map(|&(_, s, slot)| {
            if s == base {
                slots[slot as usize]
            } else {
                [NO_CLUSTER; 2]
            }
        })
        .collect();
    let centroids = base_field.centroids();
    writer.add_clustered(&field, factory, params, &mut |prepared, metric| {
        kmeans::reuse(
            prepared,
            dimension,
            centroids,
            &hints,
            params.target_cluster_size,
            &params.kmeans,
            refine_rounds,
            metric,
        )
    })
}
