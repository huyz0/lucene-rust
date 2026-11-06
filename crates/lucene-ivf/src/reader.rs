//! Reading a segment's IVF files and searching them.

use std::borrow::Cow;
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicU32, Ordering as AtomicOrdering};
use std::sync::Arc;

use lucene_store::codec_util::{self, FOOTER_LENGTH, ID_LENGTH};
use lucene_store::{DataInput, SliceInput};
use lucene_util::{simd, vector_util};
use lucene_vector_quant::{Quantizer, QuantizerRegistry, QueryScorer};

use crate::calibrate::Calibration;
use crate::kmeans::NO_CLUSTER;
use crate::{
    similarity_from_ordinal, Error, Result, Similarity, VectorEncoding, META_CODEC, POSTINGS_CODEC,
    VECTORS_CODEC, VERSION_BLOCK_LAYOUT, VERSION_CALIBRATION, VERSION_CURRENT,
    VERSION_VECTOR_ENCODING,
};

/// A file read by byte range: a memory-mapped or in-memory file, or a blob
/// fetched through a local cache. An index reads the postings and vectors
/// files only through this, one range per probed cluster and one per
/// reranked vector.
pub trait RangeSource: Send + Sync {
    /// The file length.
    fn len(&self) -> usize;

    /// Whether the file is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// `len` bytes from `offset`.
    ///
    /// # Errors
    ///
    /// A range past the end, or the source's own read failure.
    fn read(&self, offset: usize, len: usize) -> Result<Cow<'_, [u8]>>;

    /// A hint that `len` bytes from `offset` will be read soon. In-memory
    /// sources start loading them into the CPU cache; by default nothing
    /// happens.
    fn prefetch(&self, offset: usize, len: usize) {
        let _ = (offset, len);
    }
}

impl<T> RangeSource for T
where
    T: std::ops::Deref<Target = [u8]> + Send + Sync,
{
    fn len(&self) -> usize {
        <[u8]>::len(self)
    }

    fn read(&self, offset: usize, len: usize) -> Result<Cow<'_, [u8]>> {
        offset
            .checked_add(len)
            .and_then(|stop| self.get(offset..stop))
            .map(Cow::Borrowed)
            .ok_or_else(|| {
                Error::Corrupted(format!(
                    "read of {len} bytes at {offset} beyond a file of {} bytes",
                    <[u8]>::len(self)
                ))
            })
    }

    fn prefetch(&self, offset: usize, len: usize) {
        let stop = offset.saturating_add(len).min(<[u8]>::len(self));
        for at in (offset..stop).step_by(64) {
            simd::prefetch(self, at);
        }
    }
}

/// A search hit: a document and its score (exact when reranked, estimated
/// otherwise).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoredDoc {
    /// The document.
    pub doc: i32,
    /// Its score; higher is closer.
    pub score: f32,
}

/// How many clusters a search scans.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Probes {
    /// This many clusters (at least one).
    Fixed(usize),
    /// The field's calibrated share of its clusters
    /// ([`IvfField::auto_nprobe`]).
    Auto,
}

/// How many candidates, by estimated score, a search reranks with the
/// full-precision vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rerank {
    /// Exactly the best `n` candidates (at least `k` are kept). Zero returns
    /// estimated scores without reading any vector.
    Fixed(usize),
    /// Up to the field's calibrated depth ([`IvfField::auto_rerank`]),
    /// stopping early as [`Rerank::Bounded`] does.
    Auto,
    /// Up to `cap` candidates (at least `k`), in estimate order, stopping
    /// once no remaining candidate's optimistic score (its estimate plus the
    /// quantizer's error margin,
    /// [`lucene_vector_quant::QueryScorer::error_bounds`]) can beat the
    /// `k`-th exact score so far, and skipping each candidate whose own
    /// optimistic score cannot. Without margins from the quantizer this is
    /// [`Rerank::Fixed`]`(cap)`. A `cap` of zero returns estimates.
    Bounded {
        /// The most candidates reranked.
        cap: usize,
    },
}

/// How a search trades recall for work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SearchParams {
    /// Hits to return.
    pub k: usize,
    /// Clusters to scan.
    pub nprobe: Probes,
    /// Candidates to rerank.
    pub rerank: Rerank,
}

impl SearchParams {
    /// `k` hits from `nprobe` clusters, reranking `rerank` candidates: fixed
    /// work whatever the field.
    pub fn new(k: usize, nprobe: usize, rerank: usize) -> Self {
        SearchParams {
            k,
            nprobe: Probes::Fixed(nprobe),
            rerank: Rerank::Fixed(rerank),
        }
    }

    /// `k` hits at each field's calibrated depth: [`Probes::Auto`] and
    /// [`Rerank::Auto`].
    pub fn auto(k: usize) -> Self {
        SearchParams {
            k,
            nprobe: Probes::Auto,
            rerank: Rerank::Auto,
        }
    }
}

/// The share of its clusters [`Probes::Auto`] scans in a field written
/// without calibration.
pub const DEFAULT_PROBE_FRACTION: f32 = 0.05;

/// The most candidates per hit [`Rerank::Auto`] reranks in a field written
/// without calibration.
pub const DEFAULT_RERANK_MULTIPLE: f32 = 10.0;

/// The clusters one segment probes when a search spreads `nprobe` probes
/// over segments holding `total_vectors` vectors in all: the segment's share
/// by its `segment_vectors`, rounded up, and at least one. Clusters hold a
/// similar number of vectors in every segment, so the segments together
/// probe about the clusters one index of all the vectors would.
pub fn segment_nprobe(nprobe: usize, total_vectors: usize, segment_vectors: usize) -> usize {
    if total_vectors == 0 {
        return nprobe.max(1);
    }
    // ARITH: a u128 holds the product of two usizes; the divisor is non-zero.
    #[allow(clippy::arithmetic_side_effects)]
    let share = (nprobe as u128 * segment_vectors.min(total_vectors) as u128)
        .div_ceil(total_vectors as u128);
    usize::try_from(share).unwrap_or(usize::MAX).max(1)
}

/// The `k`-th best exact score found so far by a search over several fields
/// or segments, shared by their searches (across threads too) and only ever
/// raised. Each search raises it once it holds `k` exact hits, and skips the
/// clusters and candidates that cannot beat it, so later searches do less
/// work. Each search still returns its best hits that can enter the shared
/// top `k`; merging every search's hits gives the result.
#[derive(Debug)]
pub struct SharedThreshold {
    /// The score as a key that orders as scores do.
    key: AtomicU32,
}

impl Default for SharedThreshold {
    fn default() -> Self {
        Self::new()
    }
}

impl SharedThreshold {
    /// A threshold every score beats.
    pub fn new() -> Self {
        SharedThreshold {
            key: AtomicU32::new(Self::key(f32::NEG_INFINITY)),
        }
    }

    /// The highest score raised so far, or negative infinity.
    pub fn get(&self) -> f32 {
        Self::score(self.key.load(AtomicOrdering::Relaxed))
    }

    /// Raises the threshold to `score` when that is higher; NaN is ignored.
    pub fn raise(&self, score: f32) {
        if !score.is_nan() {
            self.key
                .fetch_max(Self::key(score), AtomicOrdering::Relaxed);
        }
    }

    /// `score`'s bits, mapped so unsigned order is score order.
    fn key(score: f32) -> u32 {
        let bits = score.to_bits();
        if bits & 0x8000_0000 == 0 {
            bits | 0x8000_0000
        } else {
            !bits
        }
    }

    /// The score of a key.
    fn score(key: u32) -> f32 {
        if key & 0x8000_0000 == 0 {
            f32::from_bits(!key)
        } else {
            f32::from_bits(key & 0x7fff_ffff)
        }
    }

    /// The threshold, or `None` while every score beats it.
    fn floor(&self) -> Option<f32> {
        let t = self.get();
        (t > f32::NEG_INFINITY).then_some(t)
    }
}

/// The work one search did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SearchStats {
    /// Clusters whose codes were scored.
    pub clusters_scanned: usize,
    /// Probed clusters skipped because none of their vectors could beat the
    /// shared threshold.
    pub clusters_skipped: usize,
    /// Candidates kept by estimated score for rerank.
    pub candidates: usize,
    /// Candidates whose full-precision vectors were read and scored.
    pub reranked: usize,
}

/// A search's hits, best first, and the work it took.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchOutput {
    /// The hits.
    pub hits: Vec<ScoredDoc>,
    /// The work.
    pub stats: SearchStats,
}

/// One field of an IVF index.
#[derive(Debug)]
pub struct IvfField {
    number: i32,
    similarity: Similarity,
    encoding: VectorEncoding,
    dimension: usize,
    count: usize,
    /// Bytes of one stored vector in `encoding`.
    vector_bytes: usize,
    quantizer: Box<dyn Quantizer>,
    centroids: Vec<f32>,
    /// The centroids as the quantizer encodes and scores against them
    /// (`Quantizer::prepare_centroid`), `prepared_len` floats each, mapped
    /// once at open; empty when the quantizer uses centroids as they are.
    prepared: Vec<f32>,
    prepared_len: usize,
    /// How a query ranks centroids: by `1 / |c|` times the inner product
    /// for a spherically clustered field (one factor per centroid), by
    /// inner product for `MaximumInnerProduct`, by Euclidean distance
    /// otherwise (both empty).
    centroid_scale: Vec<f32>,
    /// `cluster count + 1` prefix sums of the clusters' entry counts:
    /// cluster `c`'s entries are `starts[c]..starts[c + 1]`.
    starts: Vec<usize>,
    /// For a residual encoding, `cluster count + 1` prefix sums of the
    /// clusters' own vector counts: cluster `c`'s vectors are the slots
    /// `primary_starts[c]..primary_starts[c + 1]`. Empty otherwise.
    primary_starts: Vec<usize>,
    /// `cluster count + 1` prefix sums of the clusters' posting block
    /// lengths in bytes (slots, then the quantizer's block of codes), from
    /// the field's postings offset.
    block_starts: Vec<usize>,
    /// Each slot's document, from the doc table, kept in memory: posting
    /// entries name slots, and a filter or a hit needs the document.
    slot_docs: Vec<i32>,
    postings_offset: usize,
    vectors_offset: usize,
    doc_table_offset: usize,
    /// Per cluster, the largest distance from its centroid of an entry's
    /// stored vector (unit-normalized for `Cosine`); empty before
    /// `VERSION_CALIBRATION`.
    radii: Vec<f32>,
    calibration: Option<Calibration>,
}

impl IvfField {
    /// A field over the postings of `postings_offset` being written, with
    /// no vectors file: what calibration searches. `slot_docs` names each
    /// slot's document; `entry_sizes` are the clusters' entry counts.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn in_memory(
        number: i32,
        similarity: Similarity,
        dimension: usize,
        quantizer: Box<dyn Quantizer>,
        centroids: Vec<f32>,
        entry_sizes: &[usize],
        postings_offset: usize,
        slot_docs: Vec<i32>,
    ) -> Result<Self> {
        let mut starts = Vec::with_capacity(entry_sizes.len().saturating_add(1));
        starts.push(0);
        let mut total = 0usize;
        for &size in entry_sizes {
            total = add(total, size, "cluster sizes")?;
            starts.push(total);
        }
        let (block_starts, _) = block_layout(quantizer.as_ref(), &starts)?;
        let (prepared, prepared_len) = prepare_centroids(quantizer.as_ref(), &centroids, dimension);
        Ok(IvfField {
            number,
            similarity,
            encoding: VectorEncoding::F32,
            dimension,
            count: slot_docs.len(),
            vector_bytes: 0,
            quantizer,
            prepared,
            prepared_len,
            centroid_scale: centroid_scale(similarity, &centroids, dimension),
            centroids,
            starts,
            primary_starts: Vec::new(),
            block_starts,
            slot_docs,
            postings_offset,
            vectors_offset: 0,
            doc_table_offset: 0,
            radii: Vec::new(),
            calibration: None,
        })
    }

    /// The search depth calibrated when the field was written, if it was.
    pub fn calibration(&self) -> Option<Calibration> {
        self.calibration
    }

    /// Cluster `c`'s radius: the largest distance from its centroid of an
    /// entry's stored vector (unit-normalized for `Cosine`), or `None` in a
    /// field written before radii were recorded.
    ///
    /// # Panics
    ///
    /// When `c` is not below [`IvfField::clusters`].
    pub fn radius(&self, c: usize) -> Option<f32> {
        assert!(c < self.clusters(), "cluster {c} out of range");
        self.radii.get(c).copied()
    }

    /// The clusters [`Probes::Auto`] scans: the calibrated share of the
    /// field's clusters ([`DEFAULT_PROBE_FRACTION`] without calibration),
    /// rounded up, and at least one.
    pub fn auto_nprobe(&self) -> usize {
        let fraction = self
            .calibration
            .map_or(DEFAULT_PROBE_FRACTION, |c| c.probe_fraction);
        let clusters = self.clusters();
        // The calibrated fraction is a whole number of clusters over this
        // count; the slack keeps float rounding from adding one.
        let n = (f64::from(fraction) * clusters as f64 - 1e-6).ceil();
        (n.max(1.0) as usize).min(clusters.max(1))
    }

    /// The most candidates [`Rerank::Auto`] reranks for `k` hits: the
    /// calibrated multiple of `k` ([`DEFAULT_RERANK_MULTIPLE`] without
    /// calibration), rounded up.
    pub fn auto_rerank(&self, k: usize) -> usize {
        let multiple = self
            .calibration
            .map_or(DEFAULT_RERANK_MULTIPLE, |c| c.rerank_multiple);
        // The cast saturates.
        ((f64::from(multiple) * k as f64 - 1e-6).ceil() as usize).max(k)
    }
    /// The field number.
    pub fn number(&self) -> i32 {
        self.number
    }

    /// The similarity.
    pub fn similarity(&self) -> Similarity {
        self.similarity
    }

    /// How `.ivv` stores the field's full-precision vectors.
    pub fn encoding(&self) -> VectorEncoding {
        self.encoding
    }

    /// The dimension.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// The number of vectors.
    pub fn count(&self) -> usize {
        self.count
    }

    /// The number of clusters.
    pub fn clusters(&self) -> usize {
        self.starts.len().saturating_sub(1)
    }

    /// The quantizer the codes were made with.
    pub fn quantizer(&self) -> &dyn Quantizer {
        self.quantizer.as_ref()
    }

    /// Cluster `c`'s centroid.
    ///
    /// # Panics
    ///
    /// When `c` is not below [`IvfField::clusters`].
    pub fn centroid(&self, c: usize) -> &[f32] {
        crate::kmeans::row(&self.centroids, self.dimension, c)
    }

    /// Every centroid, `clusters * dimension` floats.
    pub(crate) fn centroids(&self) -> &[f32] {
        &self.centroids
    }

    /// Cluster `c`'s centroid as the quantizer encodes and scores against it
    /// (`Quantizer::prepare_centroid`), or `None` when the quantizer uses
    /// [`IvfField::centroid`] as it is.
    ///
    /// # Panics
    ///
    /// When `c` is not below [`IvfField::clusters`].
    pub fn prepared_centroid(&self, c: usize) -> Option<&[f32]> {
        if self.prepared_len == 0 {
            assert!(c < self.clusters(), "cluster {c} out of range");
            return None;
        }
        Some(crate::kmeans::row(&self.prepared, self.prepared_len, c))
    }

    /// Points `scorer` at cluster `c`'s codes, through the prepared centroid
    /// when there is one.
    fn target(&self, scorer: &mut dyn QueryScorer, c: usize) {
        match self.prepared_centroid(c) {
            Some(prepared) => scorer.set_prepared_centroid(prepared),
            None => scorer.set_centroid(self.centroid(c)),
        }
    }

    /// Posting entries over all clusters: the vector count, plus the
    /// spilled entries.
    pub fn entries(&self) -> usize {
        self.starts.last().copied().unwrap_or(0)
    }

    /// Cluster `c`'s entry count (its own vectors and those spilled into it).
    ///
    /// # Panics
    ///
    /// When `c` is not below [`IvfField::clusters`].
    pub fn cluster_size(&self, c: usize) -> usize {
        // ARITH: starts are non-decreasing prefix sums.
        #[allow(clippy::arithmetic_side_effects)]
        let size = self.starts[c + 1] - self.starts[c];
        size
    }

    /// Bytes of one stored full-precision vector.
    pub(crate) fn vector_len(&self) -> usize {
        self.vector_bytes
    }

    /// The byte range of cluster `c`'s posting block in `.ivp`.
    // ARITH: the field's whole postings region was checked to fit the file
    // when it was opened, and every block lies inside it.
    #[allow(clippy::arithmetic_side_effects)]
    fn block_range(&self, c: usize) -> (usize, usize) {
        (
            self.postings_offset + self.block_starts[c],
            self.block_starts[c + 1] - self.block_starts[c],
        )
    }

    /// The centroid slot `s`'s vector is stored relative to: its primary
    /// cluster's for a residual encoding, none otherwise.
    pub(crate) fn slot_centroid(&self, s: usize) -> &[f32] {
        if self.primary_starts.is_empty() {
            return &[];
        }
        // ARITH: `primary_starts[0] == 0 <= s`, so the partition point is
        // at least 1; `s < count == primary_starts[clusters]` keeps it a
        // cluster.
        #[allow(clippy::arithmetic_side_effects)]
        let c = self.primary_starts.partition_point(|&start| start <= s) - 1;
        self.centroid(c)
    }

    /// The byte offset of slot `s`'s vector in `.ivv`.
    // ARITH: `s < count`, and the vectors region was checked at open.
    #[allow(clippy::arithmetic_side_effects)]
    fn vector_offset(&self, s: usize) -> usize {
        self.vectors_offset + s * self.vector_len()
    }
}

/// A reader over one segment's IVF files.
pub struct IvfReader {
    fields: Vec<IvfField>,
    postings: Arc<dyn RangeSource>,
    vectors: Arc<dyn RangeSource>,
}

impl std::fmt::Debug for IvfReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IvfReader")
            .field("fields", &self.fields)
            .field("postings", &self.postings.len())
            .field("vectors", &self.vectors.len())
            .finish()
    }
}

fn corrupt<T>(msg: impl Into<String>) -> Result<T> {
    Err(Error::Corrupted(msg.into()))
}

fn read_count(input: &mut SliceInput<'_>, what: &str) -> Result<usize> {
    let v = input.read_vint()?;
    usize::try_from(v).or_else(|_| corrupt(format!("negative {what}: {v}")))
}

fn read_offset(input: &mut SliceInput<'_>, what: &str) -> Result<usize> {
    let v = input.read_vlong()?;
    usize::try_from(v).or_else(|_| corrupt(format!("bad {what}: {v}")))
}

fn read_f32(input: &mut SliceInput<'_>) -> Result<f32> {
    let mut b = [0u8; 4];
    input.read_bytes(&mut b)?;
    Ok(f32::from_le_bytes(b))
}

/// Checks a range-read file's header and the shape of its footer, and
/// returns its format version and where its payload ends.
fn check_source(
    source: &dyn RangeSource,
    codec: &str,
    segment_id: &[u8; ID_LENGTH],
    suffix: &str,
) -> Result<(i32, usize)> {
    let header_len = codec_util::index_header_length(codec, suffix);
    let Some(payload_end) = source.len().checked_sub(FOOTER_LENGTH) else {
        return corrupt(format!(
            "{codec}: file of {} bytes has no footer",
            source.len()
        ));
    };
    if payload_end < header_len {
        return corrupt(format!(
            "{codec}: file of {} bytes is truncated",
            source.len()
        ));
    }
    let header = source.read(0, header_len)?;
    let header = codec_util::check_index_header(
        &mut SliceInput::new(&header),
        codec,
        VERSION_BLOCK_LAYOUT,
        VERSION_CURRENT,
        segment_id,
        suffix,
    )?;
    let footer = source.read(payload_end, FOOTER_LENGTH)?;
    codec_util::retrieve_checksum(&footer)?;
    Ok((header.version, payload_end))
}

/// `a * b`, or a corruption error naming `what`.
fn mul(a: usize, b: usize, what: &str) -> Result<usize> {
    a.checked_mul(b)
        .map_or_else(|| corrupt(format!("{what} overflows")), Ok)
}

/// `a + b`, or a corruption error naming `what`.
fn add(a: usize, b: usize, what: &str) -> Result<usize> {
    a.checked_add(b)
        .map_or_else(|| corrupt(format!("{what} overflows")), Ok)
}

/// `centroids` (rows of `dimension` floats) through
/// `Quantizer::prepare_centroid`, laid end to end, and the length of one; empty
/// and 0 unless the quantizer prepares every centroid to one non-zero length.
fn prepare_centroids(
    quantizer: &dyn Quantizer,
    centroids: &[f32],
    dimension: usize,
) -> (Vec<f32>, usize) {
    let rows = centroids.chunks_exact(dimension.max(1));
    let clusters = rows.len();
    let prepared: Vec<Vec<f32>> = rows.map_while(|c| quantizer.prepare_centroid(c)).collect();
    let len = prepared.first().map_or(0, Vec::len);
    if len == 0 || prepared.len() != clusters || prepared.iter().any(|p| p.len() != len) {
        return (Vec::new(), 0);
    }
    (prepared.concat(), len)
}

/// Reads one field's meta entry and checks its regions against the files.
fn read_field(
    input: &mut SliceInput<'_>,
    version: i32,
    registry: &QuantizerRegistry,
    postings_end: usize,
    vectors_end: usize,
) -> Result<IvfField> {
    let mut number = [0u8; 4];
    input.read_bytes(&mut number)?;
    let number = i32::from_le_bytes(number);
    let ordinal = input.read_byte()?;
    let Some(similarity) = similarity_from_ordinal(ordinal) else {
        return corrupt(format!("field {number}: unknown similarity {ordinal}"));
    };
    let encoding = if version >= VERSION_VECTOR_ENCODING {
        let ordinal = input.read_byte()?;
        let Some(encoding) = VectorEncoding::from_ordinal(ordinal) else {
            return corrupt(format!("field {number}: unknown vector encoding {ordinal}"));
        };
        encoding
    } else {
        VectorEncoding::F32
    };
    let dimension = read_count(input, "dimension")?;
    if dimension == 0 {
        return corrupt(format!("field {number}: zero dimension"));
    }
    let count = read_count(input, "vector count")?;
    let name = input.read_string()?;
    let state_len = read_count(input, "state length")?;
    if state_len > input.remaining() {
        return corrupt(format!(
            "field {number}: state of {state_len} bytes past the end"
        ));
    }
    let mut state = vec![0u8; state_len];
    input.read_bytes(&mut state)?;
    let quantizer = registry.get(&name)?.read(dimension, similarity, &state)?;
    let nlist = read_count(input, "cluster count")?;
    // Each cluster's entry count takes at least a byte of what is left, which
    // bounds the allocations below by the meta file's size.
    if (nlist == 0) != (count == 0) || nlist > count || nlist > input.remaining() {
        return corrupt(format!(
            "field {number}: {nlist} clusters for {count} vectors"
        ));
    }
    let postings_offset = read_offset(input, "postings offset")?;
    let vectors_offset = read_offset(input, "vectors offset")?;
    let doc_table_offset = read_offset(input, "doc table offset")?;
    let centroid_bytes = mul(mul(nlist, dimension, "centroids")?, 4, "centroids")?;
    if centroid_bytes > input.remaining() {
        return corrupt(format!("field {number}: centroids past the end"));
    }
    let mut centroids = Vec::with_capacity(centroid_bytes / 4);
    for _ in 0..centroid_bytes / 4 {
        centroids.push(read_f32(input)?);
    }
    let mut starts = Vec::with_capacity(nlist.saturating_add(1));
    let mut total = 0usize;
    starts.push(0);
    for _ in 0..nlist {
        let size = read_count(input, "cluster size")?;
        if size == 0 {
            return corrupt(format!("field {number}: empty cluster"));
        }
        total = add(total, size, "cluster sizes")?;
        starts.push(total);
    }
    // Every vector has a primary entry and at most one spilled one.
    if total < count || total > mul(count, 2, "entries")? {
        return corrupt(format!(
            "field {number}: {total} entries for {count} vectors"
        ));
    }
    let mut primary_starts = Vec::new();
    if encoding.residual() {
        primary_starts.reserve(starts.len());
        primary_starts.push(0);
        let mut total = 0usize;
        for w in starts.windows(2) {
            let size = read_count(input, "primary cluster size")?;
            // ARITH: `starts` ascend.
            #[allow(clippy::arithmetic_side_effects)]
            let entries = w[1] - w[0];
            if size > entries {
                return corrupt(format!(
                    "field {number}: {size} vectors in a cluster of {entries} entries"
                ));
            }
            total = add(total, size, "primary cluster sizes")?;
            primary_starts.push(total);
        }
        if total != count {
            return corrupt(format!(
                "field {number}: {total} primary entries for {count} vectors"
            ));
        }
    }
    let mut radii = Vec::new();
    let mut calibration = None;
    if version >= VERSION_CALIBRATION {
        if mul(nlist, 4, "radii")? > input.remaining() {
            return corrupt(format!("field {number}: radii past the end"));
        }
        radii.reserve(nlist);
        for _ in 0..nlist {
            let r = read_f32(input)?;
            if !(r >= 0.0 && r.is_finite()) {
                return corrupt(format!("field {number}: cluster radius {r}"));
            }
            radii.push(r);
        }
        match input.read_byte()? {
            0 => {}
            1 => {
                let c = Calibration {
                    probe_fraction: read_f32(input)?,
                    rerank_multiple: read_f32(input)?,
                };
                if !c.is_valid() {
                    return corrupt(format!("field {number}: calibration {c:?}"));
                }
                calibration = Some(c);
            }
            b => return corrupt(format!("field {number}: calibration flag {b}")),
        }
    }
    let (block_starts, postings_bytes) = block_layout(quantizer.as_ref(), &starts)?;
    if add(postings_offset, postings_bytes, "postings")? > postings_end {
        return corrupt(format!("field {number}: postings past the end of .ivp"));
    }
    let vector_bytes = encoding
        .vector_bytes(dimension)
        .map_or_else(|| corrupt("vector length overflows"), Ok)?;
    let vectors_bytes = mul(count, vector_bytes, "vectors")?;
    if !vectors_offset.is_multiple_of(4)
        || add(vectors_offset, vectors_bytes, "vectors")? > doc_table_offset
        || add(doc_table_offset, mul(count, 8, "doc table")?, "doc table")? > vectors_end
    {
        return corrupt(format!(
            "field {number}: vectors or doc table out of bounds"
        ));
    }
    let (prepared, prepared_len) = prepare_centroids(quantizer.as_ref(), &centroids, dimension);
    Ok(IvfField {
        number,
        similarity,
        encoding,
        dimension,
        count,
        vector_bytes,
        quantizer,
        prepared,
        prepared_len,
        centroid_scale: centroid_scale(similarity, &centroids, dimension),
        centroids,
        starts,
        primary_starts,
        block_starts,
        slot_docs: Vec::new(),
        postings_offset,
        vectors_offset,
        doc_table_offset,
        radii,
        calibration,
    })
}

/// How a query ranks a field's centroids (see `IvfField::centroid_scale`).
fn centroid_scale(similarity: Similarity, centroids: &[f32], dimension: usize) -> Vec<f32> {
    if !crate::writer::spherical(similarity) {
        return Vec::new();
    }
    centroids
        .chunks_exact(dimension.max(1))
        .map(|c| {
            let norm = simd::dot_f32(c, c).sqrt();
            if norm > 0.0 {
                1.0 / norm
            } else {
                0.0
            }
        })
        .collect()
}

/// The `cluster count + 1` prefix sums of the posting block lengths of
/// clusters whose entries start at `starts`, and their total.
fn block_layout(quantizer: &dyn Quantizer, starts: &[usize]) -> Result<(Vec<usize>, usize)> {
    let mut block_starts = Vec::with_capacity(starts.len());
    let mut postings_bytes = 0usize;
    block_starts.push(0);
    for w in starts.windows(2) {
        // ARITH: `starts` ascend (checked sums of sizes).
        #[allow(clippy::arithmetic_side_effects)]
        let size = w[1] - w[0];
        let block = add(
            mul(size, 4, "slots")?,
            quantizer.block_len(size),
            "postings",
        )?;
        postings_bytes = add(postings_bytes, block, "postings")?;
        block_starts.push(postings_bytes);
    }
    Ok((block_starts, postings_bytes))
}

/// Reads a field's doc table and inverts it: each slot's document. Checks
/// the documents ascend and the slots are a permutation.
fn slot_docs(vectors: &dyn RangeSource, f: &IvfField) -> Result<Vec<i32>> {
    // The doc table was checked to fit the file.
    let len = mul(f.count, 8, "doc table")?;
    let bytes = vectors.read(f.doc_table_offset, len)?;
    let (docs, slots) = bytes.split_at(len / 2);
    let mut out = vec![-1i32; f.count];
    let mut previous = -1i32;
    for (d, s) in docs.chunks_exact(4).zip(slots.chunks_exact(4)) {
        let doc = le_i32(d, 0);
        let slot = u32::from_le_bytes([s[0], s[1], s[2], s[3]]) as usize;
        if doc <= previous {
            return corrupt(format!("field {}: documents out of order", f.number));
        }
        previous = doc;
        match out.get_mut(slot) {
            Some(o) if *o < 0 => *o = doc,
            _ => return corrupt(format!("field {}: bad slot {slot}", f.number)),
        }
    }
    Ok(out)
}

/// A candidate: higher scores are better, and of equal scores the lower
/// document. `bound` is the margin its exact score may exceed `score` by
/// (infinite when unknown); it takes no part in the order.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Candidate {
    pub(crate) score: f32,
    pub(crate) doc: i32,
    pub(crate) slot: u32,
    pub(crate) bound: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| other.doc.cmp(&self.doc))
    }
}

/// The best `size` candidates offered. With `distinct`, a slot offered
/// again (a spilled vector met in a second cluster) is kept once, at the
/// first score offered.
struct TopN {
    size: usize,
    heap: BinaryHeap<Reverse<Candidate>>,
    held: Option<std::collections::HashSet<u32>>,
}

impl TopN {
    fn new(size: usize, distinct: bool) -> Self {
        TopN {
            size,
            heap: BinaryHeap::with_capacity(size.saturating_add(1)),
            held: distinct.then(std::collections::HashSet::new),
        }
    }

    /// Whether `score` could enter: the heap is not full, or it beats the
    /// worst held (ties are settled by `offer`).
    #[inline]
    fn competitive(&self, score: f32) -> bool {
        self.heap.len() < self.size || self.heap.peek().is_some_and(|w| score >= w.0.score)
    }

    /// The worst score held once the heap is full.
    #[inline]
    fn floor(&self) -> Option<f32> {
        if self.heap.len() < self.size {
            return None;
        }
        self.heap.peek().map(|w| w.0.score)
    }

    #[inline]
    fn offer(&mut self, c: Candidate) {
        if let Some(held) = &mut self.held {
            if held.contains(&c.slot) {
                return;
            }
        }
        if self.heap.len() < self.size {
            self.heap.push(Reverse(c));
            if let Some(held) = &mut self.held {
                held.insert(c.slot);
            }
        } else if let Some(mut worst) = self.heap.peek_mut() {
            if c > worst.0 {
                let evicted = worst.0.slot;
                *worst = Reverse(c);
                if let Some(held) = &mut self.held {
                    held.remove(&evicted);
                    held.insert(c.slot);
                }
            }
        }
    }

    /// The candidates, best first.
    fn into_sorted(self) -> Vec<Candidate> {
        let mut v: Vec<Candidate> = self.heap.into_iter().map(|r| r.0).collect();
        v.sort_unstable_by(|a, b| b.cmp(a));
        v
    }
}

/// The higher of two optional floors.
fn higher(a: Option<f32>, b: Option<f32>) -> Option<f32> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (a, None) => a,
        (None, b) => b,
    }
}

impl IvfField {
    /// The `nprobe` clusters nearest to `query`, nearest first: by L2 for
    /// Euclidean, by dot with the unit centroid for the spherically
    /// clustered Cosine and DotProduct (see `centroid_scale`), and by raw
    /// dot for MaximumInnerProduct, whose score is the dot product itself.
    pub(crate) fn probe_order(&self, query: &[f32], nprobe: usize) -> Vec<usize> {
        let euclidean = self.similarity == Similarity::Euclidean;
        let mut scored: Vec<(f32, usize)> = self
            .centroids
            .chunks_exact(self.dimension)
            .enumerate()
            .map(|(c, centroid)| {
                let s = if euclidean {
                    -simd::square_distance_f32(query, centroid)
                } else {
                    let scale = self.centroid_scale.get(c).copied().unwrap_or(1.0);
                    simd::dot_f32(query, centroid) * scale
                };
                (s, c)
            })
            .collect();
        let by_score = |a: &(f32, usize), b: &(f32, usize)| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1));
        let n = nprobe.min(scored.len());
        if n < scored.len() {
            scored.select_nth_unstable_by(n, by_score);
            scored.truncate(n);
        }
        scored.sort_unstable_by(by_score);
        scored.into_iter().map(|(_, c)| c).collect()
    }

    /// An upper bound on the exact score of every entry of cluster `c` for
    /// `query` (prepared, of norm `query_norm`), from the cluster's radius
    /// `r`: an entry is within `r` of the centroid `m`, so its squared
    /// distance is at least `max(0, |q - m| - r)^2` and its inner product at
    /// most `<q, m> + |q| r`. A small slack covers the rounding of these
    /// sums and of the exact score's. Infinite without radii.
    fn cluster_bound(&self, query: &[f32], query_norm: f32, c: usize) -> f32 {
        let Some(&r) = self.radii.get(c) else {
            return f32::INFINITY;
        };
        let centroid = self.centroid(c);
        if self.similarity == Similarity::Euclidean {
            let d = simd::square_distance_f32(query, centroid).sqrt();
            let gap = (d - r - (1e-4 * (d + r) + 1e-6)).max(0.0);
            return 1.0 / (1.0 + gap * gap);
        }
        let m = simd::dot_f32(centroid, centroid).sqrt();
        let raw =
            simd::dot_f32(query, centroid) + query_norm * r + (1e-4 * query_norm * (m + r) + 1e-6);
        if self.similarity == Similarity::MaximumInnerProduct {
            lucene_util::quantization::scale_max_inner_product_score(raw)
        } else {
            ((1.0 + raw) / 2.0).max(0.0)
        }
    }

    /// Scores the codes of the clusters `probes` for `query` (prepared) and
    /// returns the best `size` candidates by estimate, best first, among the
    /// documents `accept` lets through. With `bounds`, each candidate
    /// carries the quantizer's error margin (infinite when it has none).
    /// Clusters that cannot beat `shared` are skipped.
    // ARITH: cluster sizes and slot offsets are bounded by the field's
    // checked regions; `scores` is resized to each block's size.
    #[allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]
    pub(crate) fn scan(
        &self,
        postings: &dyn RangeSource,
        query: &[f32],
        probes: &[usize],
        size: usize,
        accept: Option<&dyn Fn(i32) -> bool>,
        bounds: bool,
        shared: Option<&SharedThreshold>,
        stats: &mut SearchStats,
    ) -> Result<Vec<Candidate>> {
        let mut scorer = self.quantizer.query_scorer(query);
        let uses_centroid = self.quantizer.uses_centroid();
        if !uses_centroid {
            self.target(scorer.as_mut(), 0);
        }
        let mut top = TopN::new(size.min(self.entries()), self.entries() > self.count);
        let query_norm = simd::dot_f32(query, query).sqrt();
        let (mut scores, mut margins) = (Vec::new(), Vec::new());
        let mut bounded = bounds;
        for &c in probes {
            if let Some(floor) = shared.and_then(SharedThreshold::floor) {
                if self.cluster_bound(query, query_norm, c) < floor {
                    stats.clusters_skipped += 1;
                    continue;
                }
            }
            stats.clusters_scanned += 1;
            let (offset, len) = self.block_range(c);
            let block = postings.read(offset, len)?;
            let size = self.cluster_size(c);
            let (slots, codes) = block.split_at(size * 4);
            if uses_centroid {
                self.target(scorer.as_mut(), c);
            }
            scores.resize(size, 0f32);
            scorer.score_block(codes, &mut scores);
            if bounded {
                margins.resize(size, 0f32);
                bounded = scorer.error_bounds(codes, &scores, &mut margins);
            }
            for (i, &score) in scores.iter().enumerate() {
                if !top.competitive(score) {
                    continue;
                }
                let slot = le_i32(slots, i * 4) as u32;
                let Some(&doc) = self.slot_docs.get(slot as usize) else {
                    return corrupt(format!("cluster {c}: slot {slot} out of range"));
                };
                if accept.is_some_and(|a| !a(doc)) {
                    continue;
                }
                let bound = match margins.get(i) {
                    Some(&b) if bounded && !b.is_nan() => b.max(0.0),
                    _ => f32::INFINITY,
                };
                top.offer(Candidate {
                    score,
                    doc,
                    slot,
                    bound,
                });
            }
        }
        let candidates = top.into_sorted();
        stats.candidates = candidates.len();
        Ok(candidates)
    }
}

fn le_i32(b: &[u8], i: usize) -> i32 {
    let mut x = [0u8; 4];
    x.copy_from_slice(&b[i..][..4]);
    i32::from_le_bytes(x)
}

impl IvfReader {
    /// Opens the index whose meta file is `meta` (checksummed whole) and
    /// whose postings and vectors files are read by range (header and footer
    /// shape checked; see [`IvfReader::check_integrity`]). Quantizers are
    /// rebuilt through `registry`.
    ///
    /// # Errors
    ///
    /// A header, footer or checksum mismatch; a region that does not fit its
    /// file; a quantizer the registry does not know.
    pub fn open(
        meta: &[u8],
        postings: Arc<dyn RangeSource>,
        vectors: Arc<dyn RangeSource>,
        segment_id: &[u8; ID_LENGTH],
        suffix: &str,
        registry: &QuantizerRegistry,
    ) -> Result<Self> {
        let (postings_version, postings_end) =
            check_source(postings.as_ref(), POSTINGS_CODEC, segment_id, suffix)?;
        let (vectors_version, vectors_end) =
            check_source(vectors.as_ref(), VECTORS_CODEC, segment_id, suffix)?;
        let mut input = SliceInput::new(meta);
        let version = codec_util::check_index_header(
            &mut input,
            META_CODEC,
            VERSION_BLOCK_LAYOUT,
            VERSION_CURRENT,
            segment_id,
            suffix,
        )?
        .version;
        if postings_version != version || vectors_version != version {
            return corrupt(format!(
                "format versions differ: meta {version}, postings {postings_version}, \
                 vectors {vectors_version}"
            ));
        }
        let field_count = read_count(&mut input, "field count")?;
        let mut fields: Vec<IvfField> = Vec::new();
        for _ in 0..field_count {
            let mut field = read_field(&mut input, version, registry, postings_end, vectors_end)?;
            if fields.iter().any(|f| f.number == field.number) {
                return corrupt(format!("field {} recorded twice", field.number));
            }
            field.slot_docs = slot_docs(vectors.as_ref(), &field)?;
            fields.push(field);
        }
        codec_util::check_footer(&mut input, meta.len())?;
        Ok(IvfReader {
            fields,
            postings,
            vectors,
        })
    }

    /// Verifies the postings and vectors files' checksums, reading both
    /// whole.
    ///
    /// # Errors
    ///
    /// A checksum mismatch, or a read failure.
    pub fn check_integrity(&self) -> Result<()> {
        for source in [&self.postings, &self.vectors] {
            let all = source.read(0, source.len())?;
            let end = all.len().saturating_sub(FOOTER_LENGTH);
            codec_util::check_whole_file_footer(&all, end)?;
        }
        Ok(())
    }

    /// The fields, in the order written.
    pub fn fields(&self) -> &[IvfField] {
        &self.fields
    }

    /// The field numbered `number`.
    pub fn field(&self, number: i32) -> Option<&IvfField> {
        self.fields.iter().find(|f| f.number == number)
    }

    fn field_or_err(&self, number: i32) -> Result<&IvfField> {
        self.field(number)
            .ok_or_else(|| Error::IllegalArgument(format!("no IVF field {number}")))
    }

    /// Cluster `c`'s entries, as the documents of their vectors (in slot
    /// order), and their codes as stored: one block in the quantizer's
    /// layout (`Quantizer::write_block`), `Quantizer::block_len` bytes.
    ///
    /// # Errors
    ///
    /// An unknown field or cluster, or a read failure.
    pub fn cluster(&self, field: i32, c: usize) -> Result<(Vec<i32>, Vec<u8>)> {
        let f = self.field_or_err(field)?;
        if c >= f.clusters() {
            return Err(Error::IllegalArgument(format!("no cluster {c}")));
        }
        let (offset, len) = f.block_range(c);
        let block = self.postings.read(offset, len)?;
        // ARITH: a block is `size * 4` slot bytes, then the codes.
        #[allow(clippy::arithmetic_side_effects)]
        let (slots, codes) = block.split_at(f.cluster_size(c) * 4);
        let docs = slots
            .chunks_exact(4)
            .map(|b| {
                let slot = le_i32(b, 0) as u32 as usize;
                f.slot_docs
                    .get(slot)
                    .copied()
                    .ok_or_else(|| Error::Corrupted(format!("slot {slot} out of range")))
            })
            .collect::<Result<Vec<i32>>>()?;
        Ok((docs, codes.to_vec()))
    }

    /// The full-precision vector in `slot` (unit-normalized for `Cosine`),
    /// decoded from the field's encoding.
    fn read_vector(&self, f: &IvfField, slot: usize, out: &mut Vec<f32>) -> Result<()> {
        let bytes = self.vectors.read(f.vector_offset(slot), f.vector_len())?;
        out.resize(f.dimension, 0.0);
        f.encoding.decode(&bytes, f.slot_centroid(slot), out);
        Ok(())
    }

    /// Every vector of `field` in slot order, as stored (`IvfField::encoding`,
    /// `IvfField::vector_len` bytes each), in one range read: what a merge
    /// streams through.
    pub(crate) fn slot_vectors(&self, field: i32) -> Result<Cow<'_, [u8]>> {
        let f = self.field_or_err(field)?;
        // ARITH: the vectors region was checked to fit at open.
        #[allow(clippy::arithmetic_side_effects)]
        let len = f.count * f.vector_len();
        self.vectors.read(f.vectors_offset, len)
    }

    /// Each slot's cluster, as a [`crate::kmeans::reuse`] hint:
    /// `[primary, NO_CLUSTER]` where the field records the primary cluster
    /// (a residual encoding stores the clusters' own vector counts, and
    /// without spilling a cluster's entries are its own vectors), otherwise
    /// the one or two clusters whose postings hold the slot, read from
    /// `.ivp` in one range.
    ///
    /// # Errors
    ///
    /// An unknown field, a read failure, or postings that do not hold every
    /// slot once or twice.
    // ARITH: prefix sums are ascending and end at the vector count; block
    // ranges were checked to fit the postings at open.
    #[allow(clippy::arithmetic_side_effects)]
    pub(crate) fn slot_clusters(&self, field: i32) -> Result<Vec<[u32; 2]>> {
        let f = self.field_or_err(field)?;
        let own = if !f.primary_starts.is_empty() {
            Some(&f.primary_starts)
        } else if f.entries() == f.count {
            Some(&f.starts)
        } else {
            None
        };
        if let Some(starts) = own {
            let mut out = Vec::with_capacity(f.count);
            for (c, w) in starts.windows(2).enumerate() {
                out.resize(w[1], [c as u32, NO_CLUSTER]);
            }
            return Ok(out);
        }
        let mut out = vec![[NO_CLUSTER; 2]; f.count];
        let len = f.block_starts.last().copied().unwrap_or(0);
        let bytes = self.postings.read(f.postings_offset, len)?;
        for c in 0..f.clusters() {
            let from = f.block_starts[c];
            for s in bytes[from..from + f.cluster_size(c) * 4].chunks_exact(4) {
                let slot = le_i32(s, 0) as u32 as usize;
                let Some(entry) = out.get_mut(slot) else {
                    return corrupt(format!("slot {slot} out of range"));
                };
                let free = entry.iter_mut().find(|e| **e == NO_CLUSTER);
                let Some(free) = free else {
                    return corrupt(format!("slot {slot} in more than two clusters"));
                };
                *free = c as u32;
            }
        }
        if out.iter().any(|e| e[0] == NO_CLUSTER) {
            return corrupt("a slot in no cluster");
        }
        Ok(out)
    }

    /// The field's documents (ascending) and the slot of each one's vector.
    ///
    /// # Errors
    ///
    /// An unknown field, a read failure, or a slot out of range.
    pub fn doc_table(&self, field: i32) -> Result<Vec<(i32, u32)>> {
        let f = self.field_or_err(field)?;
        // ARITH: the doc table was checked to fit at open.
        #[allow(clippy::arithmetic_side_effects)]
        let bytes = self.vectors.read(f.doc_table_offset, f.count * 8)?;
        // ARITH: as above, `count * 4 <= bytes.len()`.
        #[allow(clippy::arithmetic_side_effects)]
        let (docs, slots) = bytes.split_at(f.count * 4);
        let mut out = Vec::with_capacity(f.count);
        for (d, s) in docs.chunks_exact(4).zip(slots.chunks_exact(4)) {
            let slot = u32::from_le_bytes([s[0], s[1], s[2], s[3]]);
            if slot as usize >= f.count {
                return corrupt(format!("slot {slot} out of range"));
            }
            out.push((le_i32(d, 0), slot));
        }
        Ok(out)
    }

    /// The vector in `slot` (see [`IvfReader::doc_table`]).
    ///
    /// # Errors
    ///
    /// An unknown field or slot, or a read failure.
    pub fn slot_vector(&self, field: i32, slot: u32) -> Result<Vec<f32>> {
        let f = self.field_or_err(field)?;
        if slot as usize >= f.count {
            return Err(Error::IllegalArgument(format!("no slot {slot}")));
        }
        let mut out = Vec::with_capacity(f.dimension);
        self.read_vector(f, slot as usize, &mut out)?;
        Ok(out)
    }

    /// The vector of document `doc`, or `None` when it has none.
    ///
    /// # Errors
    ///
    /// An unknown field, or a read failure.
    pub fn vector(&self, field: i32, doc: i32) -> Result<Option<Vec<f32>>> {
        let f = self.field_or_err(field)?;
        // Binary search over the doc table's sorted docs, a few 4-byte reads
        // rather than the whole table.
        let read_u32 = |offset: usize| -> Result<u32> {
            let b = self.vectors.read(offset, 4)?;
            Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        };
        let (mut lo, mut hi) = (0usize, f.count);
        // ARITH: `lo < hi <= count`, and the table's `count * 8` bytes were
        // checked to fit at open.
        #[allow(clippy::arithmetic_side_effects)]
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let d = read_u32(f.doc_table_offset + mid * 4)? as i32;
            match d.cmp(&doc) {
                Ordering::Less => lo = mid + 1,
                Ordering::Greater => hi = mid,
                Ordering::Equal => {
                    let slot = read_u32(f.doc_table_offset + (f.count + mid) * 4)? as usize;
                    if slot >= f.count {
                        return corrupt(format!("slot {slot} out of range"));
                    }
                    let mut out = Vec::with_capacity(f.dimension);
                    self.read_vector(f, slot, &mut out)?;
                    return Ok(Some(out));
                }
            }
        }
        Ok(None)
    }

    /// The approximate `params.k` nearest documents to `query` among those
    /// `accept` lets through.
    ///
    /// # Errors
    ///
    /// An unknown field, a query of the wrong dimension, or a read failure.
    pub fn search(
        &self,
        field: i32,
        query: &[f32],
        params: &SearchParams,
        accept: Option<&dyn Fn(i32) -> bool>,
    ) -> Result<Vec<ScoredDoc>> {
        Ok(self.search_with(field, query, params, accept, None)?.hits)
    }

    /// [`IvfReader::search`], also reporting the work done, and sharing the
    /// `k`-th best exact score with searches of other fields or segments
    /// through `shared` (see [`SharedThreshold`]): clusters and, with
    /// [`Rerank::Auto`] or [`Rerank::Bounded`], candidates that cannot beat
    /// it are skipped, and once this search holds `k` exact hits it raises
    /// it. Without `shared`, the hits are [`IvfReader::search`]'s.
    ///
    /// # Errors
    ///
    /// An unknown field, a query of the wrong dimension, or a read failure.
    pub fn search_with(
        &self,
        field: i32,
        query: &[f32],
        params: &SearchParams,
        accept: Option<&dyn Fn(i32) -> bool>,
        shared: Option<&SharedThreshold>,
    ) -> Result<SearchOutput> {
        let f = self.field_or_err(field)?;
        let query = self.prepare_query(f, query)?;
        let mut stats = SearchStats::default();
        if params.k == 0 || f.count == 0 {
            return Ok(SearchOutput {
                hits: Vec::new(),
                stats,
            });
        }
        let nprobe = match params.nprobe {
            Probes::Fixed(n) => n.max(1),
            Probes::Auto => f.auto_nprobe(),
        };
        let (rerank, early_stop) = match params.rerank {
            Rerank::Fixed(n) => (n, false),
            Rerank::Bounded { cap } => (cap, true),
            Rerank::Auto => (f.auto_rerank(params.k), true),
        };
        let probes = f.probe_order(&query, nprobe);
        let candidates = f.scan(
            self.postings.as_ref(),
            &query,
            &probes,
            params.k.max(rerank),
            accept,
            early_stop && rerank > 0,
            shared,
            &mut stats,
        )?;
        let hits = if rerank == 0 {
            candidates
                .into_iter()
                .take(params.k)
                .map(|c| ScoredDoc {
                    doc: c.doc,
                    score: c.score,
                })
                .collect()
        } else {
            self.rerank(f, &query, &candidates, params.k, shared, &mut stats)?
        };
        Ok(SearchOutput { hits, stats })
    }

    /// The best `k` of `candidates` (best estimate first) by exact score.
    /// Candidates go in order; once the `k`-th exact score so far, or the
    /// shared threshold, is above every remaining candidate's optimistic
    /// score (estimate plus margin), the rest are dropped, and a candidate
    /// whose own optimistic score is below it is skipped.
    // ARITH: candidate indices are below `candidates.len()`.
    #[allow(clippy::arithmetic_side_effects)]
    fn rerank(
        &self,
        f: &IvfField,
        query: &[f32],
        candidates: &[Candidate],
        k: usize,
        shared: Option<&SharedThreshold>,
        stats: &mut SearchStats,
    ) -> Result<Vec<ScoredDoc>> {
        let mut exact = TopN::new(k.min(f.count), false);
        // `reach[i]`: the most any candidate from `i` on can score.
        let mut reach = vec![f32::NEG_INFINITY; candidates.len()];
        let mut most = f32::NEG_INFINITY;
        for (r, c) in reach.iter_mut().zip(candidates).rev() {
            most = most.max(c.score + c.bound);
            *r = most;
        }
        let mut v = Vec::with_capacity(f.dimension);
        // Candidates' vectors are scattered over the file: ask for a few
        // ahead so their cache misses overlap.
        const AHEAD: usize = 8;
        let ahead = |c: Option<&Candidate>| {
            if let Some(c) = c {
                self.vectors
                    .prefetch(f.vector_offset(c.slot as usize), f.vector_len());
            }
        };
        for c in candidates.iter().take(AHEAD) {
            ahead(Some(c));
        }
        for (i, c) in candidates.iter().enumerate() {
            if let Some(floor) = higher(exact.floor(), shared.and_then(SharedThreshold::floor)) {
                if reach[i] < floor {
                    break;
                }
                if c.score + c.bound < floor {
                    continue;
                }
            }
            ahead(candidates.get(i + AHEAD));
            self.read_vector(f, c.slot as usize, &mut v)?;
            stats.reranked += 1;
            exact.offer(Candidate {
                score: f.similarity.score(query, &v),
                ..*c
            });
            // Only `k` real hits bound the shared top `k`.
            if let (Some(shared), Some(floor), true) = (shared, exact.floor(), exact.size == k) {
                shared.raise(floor);
            }
        }
        Ok(exact
            .into_sorted()
            .into_iter()
            .map(|c| ScoredDoc {
                doc: c.doc,
                score: c.score,
            })
            .collect())
    }

    /// The exact `k` nearest documents to `query` among those `accept` lets
    /// through: a scan of their full-precision vectors, for selective
    /// filters and small segments.
    ///
    /// # Errors
    ///
    /// An unknown field, a query of the wrong dimension, or a read failure.
    pub fn exact_search(
        &self,
        field: i32,
        query: &[f32],
        k: usize,
        accept: Option<&dyn Fn(i32) -> bool>,
    ) -> Result<Vec<ScoredDoc>> {
        let f = self.field_or_err(field)?;
        let query = self.prepare_query(f, query)?;
        let mut top = TopN::new(k.min(f.count), false);
        let mut v = Vec::with_capacity(f.dimension);
        for (doc, slot) in self.doc_table(field)? {
            if accept.is_some_and(|a| !a(doc)) {
                continue;
            }
            self.read_vector(f, slot as usize, &mut v)?;
            top.offer(Candidate {
                score: f.similarity.score(&query, &v),
                doc,
                slot,
                bound: 0.0,
            });
        }
        Ok(top
            .into_sorted()
            .into_iter()
            .map(|c| ScoredDoc {
                doc: c.doc,
                score: c.score,
            })
            .collect())
    }

    fn prepare_query(&self, f: &IvfField, query: &[f32]) -> Result<Vec<f32>> {
        if query.len() != f.dimension {
            return Err(Error::IllegalArgument(format!(
                "query dimension {} != field dimension {}",
                query.len(),
                f.dimension
            )));
        }
        if query.iter().any(|x| !x.is_finite()) {
            return Err(Error::IllegalArgument("query must be finite".into()));
        }
        let mut q = query.to_vec();
        if f.similarity == Similarity::Cosine {
            let _ = vector_util::l2normalize(&mut q, false);
        }
        Ok(q)
    }
}
