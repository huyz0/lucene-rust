//! Writing a segment's IVF files.
//!
//! `.ivm`:
//! ```text
//! IndexHeader(META_CODEC, version, segment id, suffix)
//! vint field count
//! for each field:
//!   i32 field number, byte similarity ordinal,
//!   byte vector encoding ordinal (from VERSION_VECTOR_ENCODING),
//!   vint dimension, vint vector count,
//!   string quantizer name, vint state length, state bytes,
//!   vint cluster count,
//!   vlong postings offset, vlong vectors offset, vlong doc table offset,
//!   centroids (cluster count x dimension f32 LE),
//!   cluster entry counts (cluster count x vint),
//!   for the int8 encoding, each cluster's own (primary) vector count
//!   (cluster count x vint): cluster c's vectors are the slots from the sum
//!   of the counts before it,
//!   from VERSION_CALIBRATION: cluster radii (cluster count x f32 LE: the
//!   largest distance from the centroid of an entry's stored vector,
//!   unit-normalized for Cosine), byte calibrated (0 or 1), and when 1:
//!   f32 probe fraction, f32 rerank multiple (`Calibration`),
//!   from VERSION_ROUTING_TREE: vint routing parent count (0 for none:
//!   fields below `routing::TREE_MIN_CLUSTERS` clusters), and when not 0,
//!   each cluster's parent (cluster count x vint, every parent used),
//!   from VERSION_PARTITIONS: byte partitioned (1 when each cluster holds
//!   documents of one partition run, `IvfWriter::add_partitioned_field`)
//! Footer
//! ```
//!
//! `.ivp`, from the field's postings offset, one block per cluster in order:
//! the cluster's entries as vector slots (`u32` LE, ascending), then their
//! codes, in the same order and encoded relative to this cluster's centroid,
//! as one block in the quantizer's layout (`Quantizer::write_block`,
//! `Quantizer::block_len` bytes: end to end by default, 32 codes interleaved
//! per block for OSQ's 1- and 2-bit encodings). A cluster's entries are its own
//! vectors and, with spilling, the vectors spilled into it from their
//! primary cluster; every vector has one primary entry and at most one
//! spilled one.
//!
//! `.ivv`, from the field's vectors offset (page aligned, [`crate::VECTORS_ALIGNMENT`]): every vector
//! (`VectorEncoding::vector_bytes(dimension)` bytes in the field's
//! encoding, int8 relative to its primary cluster's centroid; `f32` LE
//! before `VERSION_VECTOR_ENCODING`) in primary-cluster order, then
//! document order within a cluster; a vector's position is its slot. From
//! the doc table offset: the field's documents ascending (`i32` LE) and, for
//! each, its slot (`u32` LE).
//!
//! A `Cosine` field stores its vectors unit-normalized.

use rayon::prelude::*;

use lucene_store::codec_util::{self, ID_LENGTH};
use lucene_store::DataOutput;
use std::sync::Arc;

use lucene_vector_quant::{Quantizer, QuantizerFactory};

use crate::calibrate::{self, DEFAULT_CALIBRATION_RECALL};
use crate::kmeans::{self, KMeansParams, SplitMix64};
use crate::merge::MergeClustering;
use crate::model::SharedModel;
use crate::reader::{IvfField, DEFAULT_PROBE_FRACTION};
use crate::routing;
use crate::{
    similarity_ordinal, Error, Result, Similarity, VectorEncoding, META_CODEC, POSTINGS_CODEC,
    VECTORS_CODEC, VERSION_CURRENT,
};

/// Build parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IvfParams {
    /// The most vectors a cluster is split down to. A field with no more
    /// vectors than this is one cluster: a flat scan of its codes. The
    /// refinement that follows the split can leave a cluster somewhat over
    /// it, and spilling adds entries on top.
    pub target_cluster_size: usize,
    /// How the clusters are found.
    pub kmeans: KMeansParams,
    /// The most vectors the quantizer trains on.
    pub training_sample: usize,
    /// Spill every vector into a second cluster with SOAR
    /// ([`kmeans::soar_assign`]), weighting the residual term by this
    /// `lambda`; `None` keeps one cluster per vector. Spilling doubles the
    /// codes (the full-precision vectors are stored once) and lets a search
    /// probe fewer clusters for the same recall.
    pub spill: Option<f32>,
    /// How `.ivv` stores the full-precision vectors that rerank and exact
    /// search read: `f32` exactly, or `f16`, `bf16` or 8 bits for a half or
    /// about a quarter of the bytes per reranked candidate.
    pub rerank_encoding: VectorEncoding,
    /// Calibrate the field's search depth for this recall at 10 hits
    /// ([`crate::calibrate`]), in `(0, 1]`, and record it for
    /// [`crate::Probes::Auto`] and [`crate::Rerank::Auto`] searches; `None`
    /// records none, and those searches fall back to
    /// [`crate::DEFAULT_PROBE_FRACTION`] and
    /// [`crate::DEFAULT_RERANK_MULTIPLE`]. Calibration searches a sample of
    /// 64 of the field's own vectors, a small fraction of the clustering's
    /// cost. A field coded by [`crate::AutoQuantizer`] still compares its
    /// candidates at [`DEFAULT_CALIBRATION_RECALL`] when this is `None`,
    /// and records no calibration.
    pub calibration_recall: Option<f32>,
    /// How [`crate::merge_field`] clusters a merged segment: afresh, or from
    /// the clusters of its largest source.
    pub merge_clustering: MergeClustering,
    /// Store a centroid routing tree ([`crate::routing`]) from this many
    /// clusters on; `usize::MAX` stores none.
    pub routing_tree_min: usize,
    /// The fewest vectors a partition of
    /// [`IvfWriter::add_partitioned_field`] is clustered alone with; smaller
    /// partitions are clustered together with their neighbours, in runs of
    /// at least this many vectors.
    pub partition_min_vectors: usize,
    /// For a field whose quantizer is picked from candidates
    /// ([`crate::AutoQuantizer`]): the fewest bytes a reranked vector is
    /// counted as reading, a disk page by default; a search of vectors held
    /// in memory reads only the vector's own bytes (`0`).
    pub auto_rerank_read_bytes: usize,
}

impl Default for IvfParams {
    fn default() -> Self {
        IvfParams {
            target_cluster_size: 384,
            kmeans: KMeansParams::default(),
            training_sample: 25_000,
            spill: None,
            rerank_encoding: VectorEncoding::F32,
            calibration_recall: Some(DEFAULT_CALIBRATION_RECALL),
            merge_clustering: MergeClustering::default(),
            routing_tree_min: routing::TREE_MIN_CLUSTERS,
            partition_min_vectors: 20_000,
            auto_rerank_read_bytes: 4096,
        }
    }
}

/// One field's vectors, as a segment holds them.
#[derive(Debug, Clone, Copy)]
pub struct FieldVectors<'a> {
    /// The field number.
    pub number: i32,
    /// The field's similarity.
    pub similarity: Similarity,
    /// The vector dimension.
    pub dimension: usize,
    /// The documents with a vector, strictly ascending.
    pub docs: &'a [i32],
    /// One vector per document, laid end to end.
    pub vectors: &'a [f32],
}

/// The three files of a segment's IVF index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IvfFiles {
    /// `.ivm`.
    pub meta: Vec<u8>,
    /// `.ivp`.
    pub postings: Vec<u8>,
    /// `.ivv`.
    pub vectors: Vec<u8>,
}

/// A written field's clusters, as its reader routes with them: what an
/// indexing buffer routes its vectors to before they are flushed.
#[derive(Debug, Clone)]
pub struct FieldClusters {
    pub number: i32,
    pub similarity: Similarity,
    pub dimension: usize,
    /// `clusters * dimension` floats, as stored.
    pub centroids: Vec<f32>,
    /// Each centroid's parent in the routing tree stored with the field.
    pub tree_parents: Option<Vec<u32>>,
    /// Per cluster, the largest distance from its centroid of an entry's
    /// vector (unit-normalized for `Cosine`).
    pub radii: Vec<f32>,
    /// The share of the clusters a search probes by default.
    pub probe_fraction: f32,
}

impl FieldClusters {
    /// How many clusters there are.
    pub fn len(&self) -> usize {
        self.centroids
            .len()
            .checked_div(self.dimension)
            .unwrap_or(0)
    }

    /// Whether there are none.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The router the field's reader builds ([`Routing::Auto`]).
    ///
    /// [`Routing::Auto`]: crate::Routing::Auto
    pub fn router(&self) -> routing::Router {
        routing::Router::new(
            &self.centroids,
            self.dimension,
            self.similarity,
            crate::Routing::Auto,
            self.tree_parents.as_deref(),
        )
    }

    /// The probes a search makes by default: [`IvfField::auto_nprobe`].
    ///
    /// [`IvfField::auto_nprobe`]: crate::IvfField::auto_nprobe
    pub fn nprobe(&self) -> usize {
        crate::reader::probes_for(self.probe_fraction, self.len())
    }
}

/// Writes the IVF files of one segment, field by field.
#[derive(Debug)]
pub struct IvfWriter {
    fields: Vec<u8>,
    field_numbers: Vec<i32>,
    clusters: Vec<FieldClusters>,
    postings: Vec<u8>,
    vectors: VectorsBuf,
    meta: Vec<u8>,
}

/// The bytes written so far to the `.ivv` file: held in memory, or, once the
/// writer is given an output with [`IvfWriter::stream_vectors_to`], passed on
/// to it whenever a few megabytes have gathered. Offsets and alignment count
/// every byte either way.
#[derive(Debug)]
struct VectorsBuf {
    mem: Vec<u8>,
    out: Option<lucene_store::FsIndexOutput>,
    spilled: u64,
}

/// The most a streaming [`VectorsBuf`] holds before passing it on.
const VECTORS_SPILL_BYTES: usize = 4 << 20;

impl VectorsBuf {
    fn new(mem: Vec<u8>) -> Self {
        VectorsBuf {
            mem,
            out: None,
            spilled: 0,
        }
    }

    /// The file's length so far.
    fn len(&self) -> usize {
        usize::try_from(self.spilled)
            .unwrap_or(usize::MAX)
            .saturating_add(self.mem.len())
    }

    /// Pads the file with zeros to the next [`crate::VECTORS_ALIGNMENT`].
    fn align(&mut self) {
        let aligned = self.len().next_multiple_of(crate::VECTORS_ALIGNMENT);
        let pad = aligned.saturating_sub(self.len());
        self.mem.resize(self.mem.len().saturating_add(pad), 0);
    }

    fn write_bytes(&mut self, bytes: &[u8]) {
        self.mem.write_bytes(bytes);
        if self.out.is_some() && self.mem.len() >= VECTORS_SPILL_BYTES {
            self.spill();
        }
    }

    fn spill(&mut self) {
        if let Some(out) = self.out.as_mut() {
            DataOutput::write_bytes(out, &self.mem);
            self.spilled = self
                .spilled
                .saturating_add(u64::try_from(self.mem.len()).unwrap_or(u64::MAX));
            self.mem.clear();
        }
    }
}

/// Spare bytes a buffer sized for a field keeps for the file's footer.
const FOOTER_SLACK: usize = 64;

/// A field's vectors: unit-normalized for `Cosine`.
///
/// Borrowed as given for every other similarity: a copy of the whole input
/// would double what a merge of a large field holds.
fn prepared_vectors<'a>(field: &FieldVectors<'a>) -> std::borrow::Cow<'a, [f32]> {
    if field.similarity != Similarity::Cosine {
        return std::borrow::Cow::Borrowed(field.vectors);
    }
    let mut out = field.vectors.to_vec();
    prepare_vectors(field.similarity, field.dimension, &mut out);
    std::borrow::Cow::Owned(out)
}

/// Unit-normalizes `vectors` (`dimension` floats each) in place when
/// `similarity` is `Cosine`, which is what a field handed to
/// [`crate::merge_field_with_prepared_vectors`] must be: the writer then
/// reads it as given instead of normalizing a copy, which for a large merged
/// field is a third of what the build holds. Other similarities are left
/// as they are.
pub fn prepare_vectors(similarity: Similarity, dimension: usize, vectors: &mut [f32]) {
    if similarity != Similarity::Cosine || dimension == 0 {
        return;
    }
    vectors.par_chunks_mut(dimension).for_each(|v| {
        // A zero vector has no direction and stays zero.
        let _ = lucene_util::vector_util::l2normalize(v, false);
    });
}

/// The lengths of the runs of `docs` (ascending) to cluster on their own:
/// one per partition (`starts` holds each partition's first document,
/// ascending) of at least `min` vectors, and the partitions between them
/// together, in runs of at least `min` except where a large partition or
/// the end cuts one short.
// ARITH: run lengths are bounded by `docs.len()`.
#[allow(clippy::arithmetic_side_effects)]
fn partition_runs(docs: &[i32], starts: &[i32], min: usize) -> Vec<usize> {
    let mut runs = Vec::new();
    let mut pooled = 0usize;
    let mut at = 0usize;
    let mut bounds = starts
        .iter()
        .skip_while(|&&s| docs.first().is_some_and(|&d| s <= d));
    while at < docs.len() {
        let end = match bounds.next() {
            Some(&next) => at + docs[at..].partition_point(|&d| d < next),
            None => docs.len(),
        };
        let len = end - at;
        at = end;
        if len == 0 {
            continue;
        }
        if len >= min {
            if pooled > 0 {
                runs.push(pooled);
                pooled = 0;
            }
            runs.push(len);
        } else {
            pooled += len;
            if pooled >= min {
                runs.push(pooled);
                pooled = 0;
            }
        }
    }
    if pooled > 0 {
        runs.push(pooled);
    }
    runs
}

/// Whether a field's vectors are unit length and clustered spherically.
pub(crate) fn spherical(similarity: Similarity) -> bool {
    matches!(similarity, Similarity::Cosine | Similarity::DotProduct)
}

/// `clustering` checked against a field of `n` vectors of dimension `dim`,
/// with clusters that hold no vector dropped.
fn checked_clustering(
    clustering: kmeans::Clustering,
    n: usize,
    dim: usize,
) -> Result<kmeans::Clustering> {
    let bad = |m: &str| Err(Error::IllegalArgument(format!("clustering: {m}")));
    if !clustering.centroids.len().is_multiple_of(dim) {
        return bad("centroids are not whole vectors");
    }
    if clustering.centroids.iter().any(|x| !x.is_finite()) {
        return bad("centroids must be finite");
    }
    let k = clustering.len(dim);
    if clustering.assignments.len() != n {
        return bad("not one assignment per vector");
    }
    if clustering.assignments.iter().any(|&a| a as usize >= k) {
        return bad("an assignment names no centroid");
    }
    Ok(kmeans::compact(
        clustering.centroids,
        clustering.assignments,
        dim,
        k,
    ))
}

/// Where a field's rerank vectors start in `.ivv`: a page boundary
/// ([`crate::VECTORS_ALIGNMENT`]), so a vector that divides a page never
/// straddles two.
fn write_len(out: &mut Vec<u8>, v: usize) {
    // In-memory lengths fit an i64.
    out.write_vlong(i64::try_from(v).unwrap_or(i64::MAX));
}

fn write_count(out: &mut Vec<u8>, v: usize) -> Result<()> {
    let v = i32::try_from(v)
        .map_err(|_| Error::IllegalArgument(format!("count {v} does not fit an int")))?;
    out.write_vint(v);
    Ok(())
}

/// A candidate quantizer an [`crate::AutoQuantizer`] field was encoded
/// with: its postings, what calibration found and the bytes per query.
struct Candidate {
    quantizer: Arc<dyn lucene_vector_quant::Quantizer>,
    postings: Vec<u8>,
    evaluation: calibrate::Evaluation,
    cost: f64,
}

/// A field's `.ivp` section: each cluster's entries as slots, then their
/// codes against the cluster's centroid as one `quantizer` block. `flat`
/// lists every entry as (cluster, slot) in cluster order, `order` each
/// slot's vector.
// ARITH: offsets are within the codes buffer, sized for every entry.
#[allow(clippy::arithmetic_side_effects)]
fn encode_postings(
    quantizer: &dyn lucene_vector_quant::Quantizer,
    centroids: &[f32],
    dim: usize,
    entries: &[Vec<u32>],
    flat: &[(u32, u32)],
    order: &[u32],
    vectors: &[f32],
) -> Vec<u8> {
    let code_len = quantizer.code_len();
    let mut codes = vec![0u8; flat.len() * code_len];
    if code_len > 0 {
        // Each centroid as the quantizer encodes against it, mapped once
        // per cluster rather than once per vector.
        let prepared: Vec<Option<Vec<f32>>> = centroids
            .par_chunks(dim)
            .map(|c| quantizer.prepare_centroid(c))
            .collect();
        codes
            .par_chunks_mut(code_len)
            .zip(flat.par_iter())
            .for_each(|(out, &(c, slot))| {
                let vector = kmeans::row(vectors, dim, order[slot as usize] as usize);
                match &prepared[c as usize] {
                    Some(p) => quantizer.encode_prepared(vector, p, out),
                    None => quantizer.encode(vector, kmeans::row(centroids, dim, c as usize), out),
                }
            });
    }
    let mut postings = Vec::new();
    let mut at = 0usize;
    let mut block = Vec::new();
    for e in entries {
        for slot in e {
            postings.write_bytes(&slot.to_le_bytes());
        }
        block.clear();
        quantizer.write_block(&codes[at * code_len..(at + e.len()) * code_len], &mut block);
        postings.write_bytes(&block);
        at += e.len();
    }
    postings
}

impl IvfWriter {
    /// A writer for the segment `segment_id`, its files named with `suffix`.
    pub fn new(segment_id: &[u8; ID_LENGTH], suffix: &str) -> Self {
        let mut meta = Vec::new();
        let mut postings = Vec::new();
        let mut vectors = Vec::new();
        codec_util::write_index_header(&mut meta, META_CODEC, VERSION_CURRENT, segment_id, suffix);
        codec_util::write_index_header(
            &mut postings,
            POSTINGS_CODEC,
            VERSION_CURRENT,
            segment_id,
            suffix,
        );
        codec_util::write_index_header(
            &mut vectors,
            VECTORS_CODEC,
            VERSION_CURRENT,
            segment_id,
            suffix,
        );
        IvfWriter {
            fields: Vec::new(),
            field_numbers: Vec::new(),
            clusters: Vec::new(),
            postings,
            vectors: VectorsBuf::new(vectors),
            meta,
        }
    }

    /// Writes the `.ivv` file to `out` as it is built instead of holding it,
    /// which for a large field is its biggest buffer. Call before the first
    /// field is added and finish with [`IvfWriter::finish_streamed`]; the
    /// bytes are the same as [`IvfWriter::finish`] returns.
    pub fn stream_vectors_to(&mut self, out: lucene_store::FsIndexOutput) {
        self.vectors.out = Some(out);
    }

    fn check(&self, field: &FieldVectors<'_>) -> Result<()> {
        let bad = |m: String| Err(Error::IllegalArgument(m));
        if self.field_numbers.contains(&field.number) {
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
        // Slots are u32 and u32::MAX means "no cluster".
        if field.docs.len() >= u32::MAX as usize {
            return bad("too many vectors".into());
        }
        Ok(())
    }

    /// Clusters, quantizes and writes `field`, training a quantizer from
    /// `factory`.
    ///
    /// # Errors
    ///
    /// [`Error::IllegalArgument`] for a calibration recall outside `(0, 1]`,
    /// a field written twice, a zero dimension, a vector count that does not match the documents,
    /// documents out of order, a non-finite value, or a value the rerank
    /// encoding cannot hold (beyond `f16`'s range); the quantizer's
    /// training errors.
    pub fn add_field(
        &mut self,
        field: &FieldVectors<'_>,
        factory: &dyn QuantizerFactory,
        params: &IvfParams,
    ) -> Result<()> {
        self.add_clustered(field, factory, params, false, &mut |vectors, metric| {
            kmeans::cluster(
                vectors,
                field.dimension,
                params.target_cluster_size,
                &params.kmeans,
                metric,
            )
        })
    }

    /// Like [`IvfWriter::add_field`], for a field whose documents are
    /// grouped into partitions, `partition_starts` holding each partition's
    /// first document, ascending (an index sorted by a partition field):
    /// each partition of at least [`IvfParams::partition_min_vectors`]
    /// vectors is clustered on its own, and the smaller ones between them
    /// together, so every cluster holds the documents of one narrow range
    /// ([`crate::IvfField::doc_ranges`]). A search filtered to a partition
    /// then probes only its clusters, as an index of its documents alone
    /// would.
    ///
    /// # Errors
    ///
    /// As [`IvfWriter::add_field`].
    pub fn add_partitioned_field(
        &mut self,
        field: &FieldVectors<'_>,
        partition_starts: &[i32],
        factory: &dyn QuantizerFactory,
        params: &IvfParams,
    ) -> Result<()> {
        self.add_partitioned(field, false, partition_starts, factory, params)
    }

    /// [`IvfWriter::add_partitioned_field`] for vectors the caller has
    /// already passed through [`prepare_vectors`].
    ///
    /// # Errors
    ///
    /// As [`IvfWriter::add_field`].
    pub fn add_partitioned_prepared_field(
        &mut self,
        field: &FieldVectors<'_>,
        partition_starts: &[i32],
        factory: &dyn QuantizerFactory,
        params: &IvfParams,
    ) -> Result<()> {
        self.add_partitioned(field, true, partition_starts, factory, params)
    }

    fn add_partitioned(
        &mut self,
        field: &FieldVectors<'_>,
        prepared: bool,
        partition_starts: &[i32],
        factory: &dyn QuantizerFactory,
        params: &IvfParams,
    ) -> Result<()> {
        let runs = partition_runs(
            field.docs,
            partition_starts,
            params.partition_min_vectors.max(1),
        );
        self.add_prepared(
            field,
            prepared,
            factory,
            params,
            true,
            &mut |vectors, metric| {
                kmeans::cluster_runs(
                    vectors,
                    field.dimension,
                    &runs,
                    params.target_cluster_size,
                    &params.kmeans,
                    metric,
                )
            },
        )
    }

    /// Like [`IvfWriter::add_field`], but with the clusters given instead of
    /// found by k-means: `clustering` holds the centroids and each vector's
    /// cluster, in `field`'s document order. Clusters with no vector are
    /// dropped. Quantizer training, spilling and encoding are as for
    /// [`IvfWriter::add_field`]; `params.target_cluster_size` and
    /// `params.kmeans` are not used.
    ///
    /// The centroids are stored as given and probed by the field's metric:
    /// Euclidean distance, or the inner product with the unit centroid for
    /// `Cosine` and `DotProduct`, whose clusters should be spherical (see
    /// [`kmeans::Metric::InnerProduct`]).
    ///
    /// # Errors
    ///
    /// As [`IvfWriter::add_field`], and [`Error::IllegalArgument`] for
    /// centroids that are not whole vectors or not finite, or assignments
    /// that are not one per vector or name no centroid.
    pub fn add_field_with_clustering(
        &mut self,
        field: &FieldVectors<'_>,
        clustering: &kmeans::Clustering,
        factory: &dyn QuantizerFactory,
        params: &IvfParams,
    ) -> Result<()> {
        self.add_clustered(field, factory, params, false, &mut |_, _| {
            clustering.clone()
        })
    }

    /// Writes `field` against the shared `model`: each vector goes to the
    /// model's nearest centroid (a block matrix product, ties to the lower
    /// centroid), its codes are the model quantizer's against that centroid,
    /// and the field's `.ivm` entry names the model (id and checksum) and
    /// holds only the cluster sizes and radii: no centroids, no quantizer
    /// state. A cluster the segment has no vector for stays, empty, so a
    /// list has the same number in every segment of the model. `params`'
    /// clustering, quantizer, calibration and routing fields are not used;
    /// `rerank_encoding` is.
    ///
    /// # Errors
    ///
    /// As [`IvfWriter::add_field`], and [`Error::IllegalArgument`] when the
    /// field's dimension or similarity differs from the model's or
    /// `params.spill` is set (a spilled entry needs the second list's
    /// codes against a centroid the writer would have to choose).
    pub fn add_field_with_model(
        &mut self,
        field: &FieldVectors<'_>,
        model: &Arc<SharedModel>,
        params: &IvfParams,
    ) -> Result<()> {
        // The factory is not used: the model holds the quantizer.
        self.add_prepared_in(
            field,
            false,
            &lucene_vector_quant::flat::FlatFactory,
            params,
            false,
            &mut |_, _| kmeans::Clustering {
                centroids: Vec::new(),
                assignments: Vec::new(),
            },
            Some(model),
        )
    }

    /// Checks `field`, clusters its prepared vectors (unit-normalized for
    /// `Cosine`) with `clusterer`, then quantizes and writes it.
    // ARITH: every index and count below is bounded by the field's vector
    // count (checked to fit a u32) times its dimension, which is the length
    // of an existing buffer; offsets are lengths of in-memory buffers.
    #[allow(clippy::arithmetic_side_effects)]
    pub(crate) fn add_clustered(
        &mut self,
        field: &FieldVectors<'_>,
        factory: &dyn QuantizerFactory,
        params: &IvfParams,
        partitioned: bool,
        clusterer: &mut dyn FnMut(&[f32], kmeans::Metric) -> kmeans::Clustering,
    ) -> Result<()> {
        self.add_prepared(field, false, factory, params, partitioned, clusterer)
    }

    /// [`IvfWriter::add_clustered`], with `prepared` saying `field.vectors`
    /// are already as [`prepare_vectors`] leaves them, so a cosine field is
    /// not copied to be normalized.
    // ARITH: as `add_clustered`.
    #[allow(clippy::arithmetic_side_effects)]
    pub(crate) fn add_prepared(
        &mut self,
        field: &FieldVectors<'_>,
        prepared: bool,
        factory: &dyn QuantizerFactory,
        params: &IvfParams,
        partitioned: bool,
        clusterer: &mut dyn FnMut(&[f32], kmeans::Metric) -> kmeans::Clustering,
    ) -> Result<()> {
        self.add_prepared_in(
            field,
            prepared,
            factory,
            params,
            partitioned,
            clusterer,
            None,
        )
    }

    /// [`IvfWriter::add_prepared`] for a field of `model` when given: its
    /// vectors go to the model's nearest centroids (`clusterer` is not
    /// called), its codes are the model quantizer's, and its meta entry
    /// names the model instead of holding centroids and quantizer state.
    // ARITH: as `add_clustered`.
    #[allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]
    pub(crate) fn add_prepared_in(
        &mut self,
        field: &FieldVectors<'_>,
        prepared: bool,
        factory: &dyn QuantizerFactory,
        params: &IvfParams,
        partitioned: bool,
        clusterer: &mut dyn FnMut(&[f32], kmeans::Metric) -> kmeans::Clustering,
        model: Option<&Arc<SharedModel>>,
    ) -> Result<()> {
        if let Some(m) = model {
            if (m.dimension(), m.similarity()) != (field.dimension, field.similarity) {
                return Err(Error::IllegalArgument(format!(
                    "field {}: model [{}] is {} dimensions, {:?}; the field is {}, {:?}",
                    field.number,
                    m.id(),
                    m.dimension(),
                    m.similarity(),
                    field.dimension,
                    field.similarity
                )));
            }
            if params.spill.is_some() {
                return Err(Error::IllegalArgument(format!(
                    "field {}: a field of a shared model is not spilled",
                    field.number
                )));
            }
        }
        if params
            .calibration_recall
            .is_some_and(|t| !(t > 0.0 && t <= 1.0))
        {
            return Err(Error::IllegalArgument(format!(
                "calibration recall {:?} is not in (0, 1]",
                params.calibration_recall
            )));
        }
        self.check(field)?;
        let dim = field.dimension;
        let n = field.docs.len();
        let vectors = if prepared {
            std::borrow::Cow::Borrowed(field.vectors)
        } else {
            prepared_vectors(field)
        };
        let metric = if spherical(field.similarity) {
            kmeans::Metric::InnerProduct
        } else {
            kmeans::Metric::L2
        };
        let clustering = match model {
            // Clusters that hold none of this segment's vectors stay: the
            // model's list numbers are the same in every segment.
            Some(m) => kmeans::Clustering {
                centroids: m.centroids().to_vec(),
                assignments: m.assign(&vectors),
            },
            None => checked_clustering(clusterer(&vectors, metric), n, dim)?,
        };
        let nlist = clustering.len(dim);
        // Encoded before anything is written, so a refused field leaves the
        // writer as it was; int8 encodes the residual from the primary
        // centroid.
        let encoding = params.rerank_encoding;
        let vector_bytes = encoding
            .vector_bytes(dim)
            .ok_or_else(|| Error::IllegalArgument("dimension too large".into()))?;
        // `f32` vectors are written straight from `vectors` (whose values
        // `check` found finite) rather than copied into a second buffer.
        let mut encoded = if encoding == VectorEncoding::F32 {
            Vec::new()
        } else {
            vec![0u8; n * vector_bytes]
        };
        let representable = encoded
            .par_chunks_mut(vector_bytes)
            .zip(vectors.par_chunks(dim))
            .zip(clustering.assignments.par_iter())
            .all(|((out, v), &a)| {
                encoding.encode(v, kmeans::row(&clustering.centroids, dim, a as usize), out)
            });
        if !representable {
            return Err(Error::IllegalArgument(format!(
                "field {}: a vector is out of {}'s range",
                field.number,
                encoding.name()
            )));
        }

        let mut rng = SplitMix64::new(params.kmeans.seed ^ 0x7a11_0f5a_3b1e);
        let mut sample = Vec::new();
        for i in kmeans::sample_indices(n, params.training_sample, &mut rng) {
            sample.extend_from_slice(kmeans::row(&vectors, dim, i));
        }
        // Slots: vectors ordered by cluster, then by document (the input is
        // in document order, and the counting sort is stable).
        let mut sizes = vec![0usize; nlist];
        for &a in &clustering.assignments {
            sizes[a as usize] += 1;
        }
        let mut next = Vec::with_capacity(nlist);
        let mut start = 0usize;
        for &s in &sizes {
            next.push(start);
            start += s;
        }
        let mut slot_of = vec![0u32; n];
        let mut order = vec![0u32; n];
        for (i, &a) in clustering.assignments.iter().enumerate() {
            let slot = next[a as usize];
            next[a as usize] += 1;
            slot_of[i] = slot as u32;
            order[slot] = i as u32;
        }

        // Posting entries: every cluster's own vectors (its slot range),
        // plus the vectors spilled into it, each cluster's in slot order.
        let mut entries: Vec<Vec<u32>> = Vec::with_capacity(nlist);
        let mut first = 0usize;
        for &size in &sizes {
            entries.push((first as u32..(first + size) as u32).collect());
            first += size;
        }
        // Each vector's primary and spilled cluster.
        let mut clusters_of: Vec<[u32; 2]> = clustering
            .assignments
            .iter()
            .map(|&a| [a, u32::MAX])
            .collect();
        if let Some(lambda) = params.spill {
            // Spill by the metric search probes with: for unit vectors the
            // distance to a unit centroid ranks clusters as their dot does.
            let mut probe_space = clustering.clone();
            metric.adjust(&mut probe_space.centroids, dim);
            let secondary = kmeans::soar_assign(&vectors, dim, &probe_space, lambda);
            for (i, &c) in secondary.iter().enumerate() {
                if c != u32::MAX {
                    entries[c as usize].push(slot_of[i]);
                }
                clusters_of[i][1] = c;
            }
            for e in &mut entries {
                e.sort_unstable();
            }
        }
        let flat: Vec<(u32, u32)> = entries
            .iter()
            .enumerate()
            .flat_map(|(c, e)| e.iter().map(move |&slot| (c as u32, slot)))
            .collect();
        let tree = (model.is_none() && nlist >= params.routing_tree_min.max(2))
            .then(|| routing::tree_parents(&clustering.centroids, dim, field.similarity));
        let sizes_of_entries: Vec<usize> = entries.iter().map(Vec::len).collect();
        let slot_docs: Vec<i32> = order.iter().map(|&i| i as i32).collect();
        let vector_cost = vector_bytes.max(params.auto_rerank_read_bytes);
        let target = params
            .calibration_recall
            .unwrap_or(DEFAULT_CALIBRATION_RECALL);
        let calibration_seed = params.kmeans.seed ^ 0xca1_1b8a_7e5e_ed00;
        // The quantizer, its postings and, when it was picked by calibrating
        // every candidate, the chosen one's calibration.
        let (quantizer, postings, picked) = if let Some(m) = model {
            let q = m.quantizer_arc();
            let postings = encode_postings(
                q.as_ref(),
                &clustering.centroids,
                dim,
                &entries,
                &flat,
                &order,
                &vectors,
            );
            (q, postings, None)
        } else {
            match factory.candidates() {
                Some(candidates) if n >= crate::auto::AUTO_MIN_VECTORS && candidates.len() > 1 => {
                    let truths = calibrate::Truths::sample(
                        &vectors,
                        dim,
                        field.similarity,
                        calibration_seed,
                    );
                    let mut best: Option<Candidate> = None;
                    for candidate in candidates {
                        let quantizer: Arc<dyn Quantizer> =
                            Arc::from(candidate.train(dim, field.similarity, &sample)?);
                        let code_len = quantizer.code_len();
                        let postings = encode_postings(
                            quantizer.as_ref(),
                            &clustering.centroids,
                            dim,
                            &entries,
                            &flat,
                            &order,
                            &vectors,
                        );
                        let mut probe = IvfField::in_memory(
                            field.number,
                            field.similarity,
                            dim,
                            quantizer,
                            clustering.centroids.clone(),
                            &sizes_of_entries,
                            0,
                            slot_docs.clone(),
                        )?;
                        probe.set_tree(tree.clone());
                        let Some(e) = truths
                            .as_ref()
                            .map(|t| {
                                calibrate::evaluate(
                                    &probe,
                                    &postings,
                                    &vectors,
                                    &clusters_of,
                                    target,
                                    t,
                                )
                            })
                            .transpose()?
                            .flatten()
                        else {
                            continue;
                        };
                        let cost = e.scanned * code_len as f64 + e.reranked * vector_cost as f64;
                        let better = best.as_ref().is_none_or(|b| {
                            let (b, c) = (&b.evaluation, b.cost);
                            match (e.reached, b.reached) {
                                (true, false) => true,
                                (false, true) => false,
                                (true, true) => cost < c,
                                (false, false) => {
                                    e.recall > b.recall || (e.recall == b.recall && cost < c)
                                }
                            }
                        });
                        if better {
                            best = Some(Candidate {
                                quantizer: probe.into_quantizer(),
                                postings,
                                evaluation: e,
                                cost,
                            });
                        }
                    }
                    match best {
                        Some(b) => (b.quantizer, b.postings, Some(b.evaluation.calibration)),
                        None => {
                            let q: Arc<dyn Quantizer> =
                                Arc::from(candidates[0].train(dim, field.similarity, &sample)?);
                            let postings = encode_postings(
                                q.as_ref(),
                                &clustering.centroids,
                                dim,
                                &entries,
                                &flat,
                                &order,
                                &vectors,
                            );
                            (q, postings, None)
                        }
                    }
                }
                _ => {
                    let q: Arc<dyn Quantizer> =
                        Arc::from(factory.train(dim, field.similarity, &sample)?);
                    let postings = encode_postings(
                        q.as_ref(),
                        &clustering.centroids,
                        dim,
                        &entries,
                        &flat,
                        &order,
                        &vectors,
                    );
                    (q, postings, None)
                }
            }
        };
        let postings_offset = self.postings.len();
        self.postings
            .reserve_exact(postings.len().saturating_add(FOOTER_SLACK));
        self.postings.extend_from_slice(&postings);
        drop(postings);

        self.vectors.align();
        let vectors_offset = self.vectors.len();
        // Sized once: growing a buffer of this field's vectors by doubling
        // holds up to twice their bytes, plus the old copy while it moves.
        // The slack covers the footer, which would otherwise double it.
        if self.vectors.out.is_none() {
            self.vectors.mem.reserve_exact(
                n.saturating_mul(vector_bytes)
                    .saturating_add(n.saturating_mul(8))
                    .saturating_add(FOOTER_SLACK),
            );
        }
        for &i in &order {
            let i = i as usize;
            if encoding == VectorEncoding::F32 {
                for x in kmeans::row(&vectors, dim, i) {
                    self.vectors.write_bytes(&x.to_le_bytes());
                }
            } else {
                self.vectors
                    .write_bytes(&encoded[i * vector_bytes..(i + 1) * vector_bytes]);
            }
        }
        let doc_table_offset = self.vectors.len();
        for d in field.docs {
            self.vectors.write_bytes(&d.to_le_bytes());
        }
        for s in &slot_of {
            self.vectors.write_bytes(&s.to_le_bytes());
        }

        // Each cluster's radius over its entries' vectors as search decodes
        // them, for the score bounds of shared-threshold searches.
        let cosine = field.similarity == Similarity::Cosine;
        let radii: Vec<f32> = entries
            .par_iter()
            .enumerate()
            .map(|(c, e)| {
                let centroid = kmeans::row(&clustering.centroids, dim, c);
                let mut decoded = vec![0f32; dim];
                let mut r2 = 0f32;
                for &slot in e {
                    let i = order[slot as usize] as usize;
                    let v = if encoding == VectorEncoding::F32 {
                        kmeans::row(&vectors, dim, i)
                    } else {
                        let primary = clustering.assignments[i] as usize;
                        encoding.decode(
                            &encoded[i * vector_bytes..(i + 1) * vector_bytes],
                            kmeans::row(&clustering.centroids, dim, primary),
                            &mut decoded,
                        );
                        if cosine {
                            let _ = lucene_util::vector_util::l2normalize(&mut decoded, false);
                        }
                        &decoded
                    };
                    r2 = r2.max(lucene_util::simd::square_distance_f32(v, centroid));
                }
                r2.sqrt()
            })
            .collect();

        let name = quantizer.name().to_string();
        let state = quantizer.state();
        let calibration = match params.calibration_recall {
            _ if model.is_some() => None,
            Some(_) if picked.is_some() => picked,
            Some(target) => {
                let mut probe = IvfField::in_memory(
                    field.number,
                    field.similarity,
                    dim,
                    quantizer,
                    clustering.centroids.clone(),
                    &sizes_of_entries,
                    postings_offset,
                    slot_docs,
                )?;
                probe.set_tree(tree.clone());
                calibrate::calibrate(
                    &probe,
                    &self.postings,
                    &vectors,
                    &clusters_of,
                    target,
                    calibration_seed,
                )?
            }
            None => None,
        };

        let m = &mut self.fields;
        m.write_bytes(&field.number.to_le_bytes());
        m.write_byte(
            similarity_ordinal(field.similarity)
                | if model.is_some() {
                    crate::reader::SHARED_FLAG
                } else {
                    0
                },
        );
        m.write_byte(encoding.ordinal());
        write_count(m, dim)?;
        write_count(m, n)?;
        match model {
            Some(model) => {
                m.write_string(model.id());
                m.write_bytes(&model.crc().to_le_bytes());
            }
            None => {
                m.write_string(&name);
                write_count(m, state.len())?;
                m.write_bytes(&state);
            }
        }
        write_count(m, nlist)?;
        write_len(m, postings_offset);
        write_len(m, vectors_offset);
        write_len(m, doc_table_offset);
        if model.is_none() {
            for x in &clustering.centroids {
                m.write_bytes(&x.to_le_bytes());
            }
        }
        for e in &entries {
            write_count(m, e.len())?;
        }
        if encoding.residual() {
            for &size in &sizes {
                write_count(m, size)?;
            }
        }
        for r in &radii {
            m.write_bytes(&r.to_le_bytes());
        }
        match calibration {
            Some(c) => {
                m.write_byte(1);
                m.write_bytes(&c.probe_fraction.to_le_bytes());
                m.write_bytes(&c.rerank_multiple.to_le_bytes());
            }
            None => m.write_byte(0),
        }
        match &tree {
            Some(parents) => {
                write_count(m, routing::parent_count(parents))?;
                for &p in parents {
                    write_count(m, p as usize)?;
                }
            }
            None => write_count(m, 0)?,
        }
        m.write_byte(u8::from(partitioned));
        self.field_numbers.push(field.number);
        self.clusters.push(FieldClusters {
            number: field.number,
            similarity: field.similarity,
            dimension: dim,
            centroids: clustering.centroids.clone(),
            tree_parents: tree,
            radii,
            probe_fraction: calibration.map_or(DEFAULT_PROBE_FRACTION, |c| c.probe_fraction),
        });
        Ok(())
    }

    /// The clusters of the fields written so far, in order.
    pub fn clusters(&self) -> &[FieldClusters] {
        &self.clusters
    }

    /// The finished files.
    pub fn finish(mut self) -> IvfFiles {
        // Field counts are tiny.
        self.meta
            .write_vint(i32::try_from(self.field_numbers.len()).unwrap_or(i32::MAX));
        self.meta.write_bytes(&self.fields);
        codec_util::write_footer(&mut self.meta);
        codec_util::write_footer(&mut self.postings);
        codec_util::write_footer(&mut self.vectors.mem);
        IvfFiles {
            meta: self.meta,
            postings: self.postings,
            vectors: self.vectors.mem,
        }
    }

    /// [`IvfWriter::finish`] for a writer given [`IvfWriter::stream_vectors_to`]:
    /// the `.ivv` is complete in the output, which is handed back for the
    /// caller to close, and the files' `vectors` is empty.
    pub fn finish_streamed(mut self) -> (IvfFiles, Option<lucene_store::FsIndexOutput>) {
        self.meta
            .write_vint(i32::try_from(self.field_numbers.len()).unwrap_or(i32::MAX));
        self.meta.write_bytes(&self.fields);
        codec_util::write_footer(&mut self.meta);
        codec_util::write_footer(&mut self.postings);
        let Some(mut out) = self.vectors.out.take() else {
            codec_util::write_footer(&mut self.vectors.mem);
            let files = IvfFiles {
                meta: self.meta,
                postings: self.postings,
                vectors: self.vectors.mem,
            };
            return (files, None);
        };
        DataOutput::write_bytes(&mut out, &self.vectors.mem);
        codec_util::write_footer_to(&mut out);
        let files = IvfFiles {
            meta: self.meta,
            postings: self.postings,
            vectors: Vec::new(),
        };
        (files, Some(out))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::arithmetic_side_effects)]
    use super::*;
    use lucene_vector_quant::osq::OsqFactory;

    #[test]
    fn small_partitions_are_pooled_between_large_ones() {
        let docs: Vec<i32> = (0..100).collect();
        // Partitions of 40, 5, 5, 5, 30, 3, 3 and 9 documents.
        let starts = [0, 40, 45, 50, 55, 85, 88, 91];
        assert_eq!(partition_runs(&docs, &starts, 10), vec![40, 10, 5, 30, 15]);
        assert_eq!(
            partition_runs(&docs, &starts, 1),
            vec![40, 5, 5, 5, 30, 3, 3, 9]
        );
        assert_eq!(partition_runs(&docs, &[], 10), vec![100]);
        // Partitions without vectors are skipped.
        let sparse = [0, 1, 2, 50, 51, 52];
        assert_eq!(partition_runs(&sparse, &[0, 10, 20, 50], 2), vec![3, 3]);
    }

    #[test]
    fn rejects_bad_fields() {
        let mut w = IvfWriter::new(&[0; ID_LENGTH], "");
        let f = OsqFactory::bbq();
        let p = IvfParams::default();
        let ok = FieldVectors {
            number: 1,
            similarity: Similarity::Euclidean,
            dimension: 2,
            docs: &[0, 3],
            vectors: &[1.0, 2.0, 3.0, 4.0],
        };
        w.add_field(&ok, &f, &p).unwrap();
        let cases = [
            ok,
            FieldVectors {
                number: 2,
                dimension: 0,
                ..ok
            },
            FieldVectors {
                number: 2,
                docs: &[0],
                ..ok
            },
            FieldVectors {
                number: 2,
                docs: &[3, 3],
                ..ok
            },
            FieldVectors {
                number: 2,
                docs: &[-1, 3],
                ..ok
            },
            FieldVectors {
                number: 2,
                vectors: &[1.0, f32::NAN, 3.0, 4.0],
                ..ok
            },
        ];
        for c in cases {
            assert!(
                matches!(w.add_field(&c, &f, &p), Err(Error::IllegalArgument(_))),
                "{c:?}"
            );
        }
    }

    fn field<'a>(docs: &'a [i32], v: &'a [f32]) -> FieldVectors<'a> {
        FieldVectors {
            number: 1,
            similarity: Similarity::Cosine,
            dimension: 24,
            docs,
            vectors: v,
        }
    }

    fn cosine_field(n: usize, dim: usize) -> (Vec<i32>, Vec<f32>) {
        let mut state = 0x9e37_79b9_u32;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        };
        let docs = (0..n as i32).collect();
        // Magnitudes that vary, one zero vector, so normalizing matters.
        let mut vectors: Vec<f32> = (0..n * dim)
            .map(|i| next() * (1.0 + (i / dim % 7) as f32))
            .collect();
        vectors[3 * dim..4 * dim].fill(0.0);
        (docs, vectors)
    }

    #[test]
    fn vectors_prepared_by_the_caller_write_the_same_files() {
        let (docs, vectors) = cosine_field(900, 24);
        let params = IvfParams {
            target_cluster_size: 64,
            ..IvfParams::default()
        };
        let factory = OsqFactory::bbq();
        let mut copied = IvfWriter::new(&[7; ID_LENGTH], "");
        copied
            .add_field(&field(&docs, &vectors), &factory, &params)
            .unwrap();
        let mut prepared_vectors = vectors.clone();
        prepare_vectors(Similarity::Cosine, 24, &mut prepared_vectors);
        let mut in_place = IvfWriter::new(&[7; ID_LENGTH], "");
        crate::merge_field_with_prepared_vectors(
            &mut in_place,
            &field(&docs, &prepared_vectors),
            &[],
            &factory,
            &params,
        )
        .unwrap();
        assert_eq!(copied.finish(), in_place.finish());

        let starts = [0, 300, 600];
        let params = IvfParams {
            partition_min_vectors: 100,
            ..params
        };
        let mut copied = IvfWriter::new(&[7; ID_LENGTH], "");
        copied
            .add_partitioned_field(&field(&docs, &vectors), &starts, &factory, &params)
            .unwrap();
        let mut in_place = IvfWriter::new(&[7; ID_LENGTH], "");
        in_place
            .add_partitioned_prepared_field(
                &field(&docs, &prepared_vectors),
                &starts,
                &factory,
                &params,
            )
            .unwrap();
        assert_eq!(copied.finish(), in_place.finish());
    }

    #[test]
    fn preparing_leaves_other_similarities_alone() {
        let mut v = vec![3.0, 4.0, 0.0, 5.0];
        prepare_vectors(Similarity::Euclidean, 2, &mut v);
        assert_eq!(v, [3.0, 4.0, 0.0, 5.0]);
        prepare_vectors(Similarity::Cosine, 2, &mut v);
        assert_eq!(v[..2], [0.6, 0.8]);
        assert_eq!(v[2..], [0.0, 1.0]);
    }

    #[test]
    fn empty_writer_has_headers_and_footers() {
        let files = IvfWriter::new(&[1; ID_LENGTH], "x").finish();
        for f in [&files.meta, &files.postings, &files.vectors] {
            assert!(codec_util::retrieve_checksum(f).is_ok());
        }
    }

    #[test]
    fn write_count_rejects_huge_values() {
        let mut out = Vec::new();
        assert!(write_count(&mut out, usize::MAX).is_err());
    }
}
