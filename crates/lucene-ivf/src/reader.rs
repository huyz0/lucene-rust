//! Reading a segment's IVF files and searching them.

use std::borrow::Cow;
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::sync::Arc;

use lucene_store::codec_util::{self, FOOTER_LENGTH, ID_LENGTH};
use lucene_store::{DataInput, SliceInput};
use lucene_util::{simd, vector_util};
use lucene_vector_quant::{Quantizer, QuantizerRegistry};

use crate::{
    similarity_from_ordinal, Error, Result, Similarity, META_CODEC, POSTINGS_CODEC, VECTORS_CODEC,
    VERSION_CURRENT, VERSION_START,
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

/// How a search trades recall for work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SearchParams {
    /// Hits to return.
    pub k: usize,
    /// Clusters to scan.
    pub nprobe: usize,
    /// Candidates, by estimated score, to rerank with the full-precision
    /// vectors (at least `k` are kept). Zero returns estimated scores
    /// without reading any vector.
    pub rerank: usize,
}

impl SearchParams {
    /// `k` hits from `nprobe` clusters, reranking `rerank` candidates.
    pub fn new(k: usize, nprobe: usize, rerank: usize) -> Self {
        SearchParams { k, nprobe, rerank }
    }
}

/// One field of an IVF index.
#[derive(Debug)]
pub struct IvfField {
    number: i32,
    similarity: Similarity,
    dimension: usize,
    count: usize,
    quantizer: Box<dyn Quantizer>,
    centroids: Vec<f32>,
    /// How a query ranks centroids: by `1 / |c|` times the inner product
    /// for a spherically clustered field (one factor per centroid), by
    /// inner product for `MaximumInnerProduct`, by Euclidean distance
    /// otherwise (both empty).
    centroid_scale: Vec<f32>,
    /// `cluster count + 1` prefix sums of the clusters' entry counts:
    /// cluster `c`'s entries are `starts[c]..starts[c + 1]`.
    starts: Vec<usize>,
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
}

impl IvfField {
    /// The field number.
    pub fn number(&self) -> i32 {
        self.number
    }

    /// The similarity.
    pub fn similarity(&self) -> Similarity {
        self.similarity
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

    /// Bytes of one full-precision vector.
    fn vector_len(&self) -> usize {
        // ARITH: bounded when the field was opened.
        #[allow(clippy::arithmetic_side_effects)]
        let len = self.dimension * 4;
        len
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
/// returns where its payload ends.
fn check_source(
    source: &dyn RangeSource,
    codec: &str,
    segment_id: &[u8; ID_LENGTH],
    suffix: &str,
) -> Result<usize> {
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
    codec_util::check_index_header(
        &mut SliceInput::new(&header),
        codec,
        VERSION_START,
        VERSION_CURRENT,
        segment_id,
        suffix,
    )?;
    let footer = source.read(payload_end, FOOTER_LENGTH)?;
    codec_util::retrieve_checksum(&footer)?;
    Ok(payload_end)
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

/// Reads one field's meta entry and checks its regions against the files.
fn read_field(
    input: &mut SliceInput<'_>,
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
    let mut block_starts = Vec::with_capacity(starts.len());
    let mut postings_bytes = 0usize;
    block_starts.push(0);
    for w in starts.windows(2) {
        // ARITH: `starts` ascend (checked sums of non-zero sizes).
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
    if add(postings_offset, postings_bytes, "postings")? > postings_end {
        return corrupt(format!("field {number}: postings past the end of .ivp"));
    }
    let vectors_bytes = mul(mul(count, dimension, "vectors")?, 4, "vectors")?;
    if !vectors_offset.is_multiple_of(4)
        || add(vectors_offset, vectors_bytes, "vectors")? > doc_table_offset
        || add(doc_table_offset, mul(count, 8, "doc table")?, "doc table")? > vectors_end
    {
        return corrupt(format!(
            "field {number}: vectors or doc table out of bounds"
        ));
    }
    Ok(IvfField {
        number,
        similarity,
        dimension,
        count,
        quantizer,
        centroid_scale: if crate::writer::spherical(similarity) {
            centroids
                .chunks_exact(dimension)
                .map(|c| {
                    let norm = simd::dot_f32(c, c).sqrt();
                    if norm > 0.0 {
                        1.0 / norm
                    } else {
                        0.0
                    }
                })
                .collect()
        } else {
            Vec::new()
        },
        centroids,
        starts,
        block_starts,
        slot_docs: Vec::new(),
        postings_offset,
        vectors_offset,
        doc_table_offset,
    })
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
/// document.
#[derive(Debug, Clone, Copy)]
struct Candidate {
    score: f32,
    doc: i32,
    slot: u32,
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
        let postings_end = check_source(postings.as_ref(), POSTINGS_CODEC, segment_id, suffix)?;
        let vectors_end = check_source(vectors.as_ref(), VECTORS_CODEC, segment_id, suffix)?;
        let mut input = SliceInput::new(meta);
        codec_util::check_index_header(
            &mut input,
            META_CODEC,
            VERSION_START,
            VERSION_CURRENT,
            segment_id,
            suffix,
        )?;
        let field_count = read_count(&mut input, "field count")?;
        let mut fields: Vec<IvfField> = Vec::new();
        for _ in 0..field_count {
            let mut field = read_field(&mut input, registry, postings_end, vectors_end)?;
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
        // ARITH: a block is `size * (4 + code_len)` bytes.
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

    /// The full-precision vector in `slot` (unit-normalized for `Cosine`).
    fn read_vector(&self, f: &IvfField, slot: usize, out: &mut Vec<f32>) -> Result<()> {
        let bytes = self.vectors.read(f.vector_offset(slot), f.vector_len())?;
        out.clear();
        match vector_util::le_f32s(&bytes) {
            Some(v) => out.extend_from_slice(v),
            None => out.extend(
                bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
            ),
        }
        Ok(())
    }

    /// Every vector of `field` in slot order, little-endian `f32`s, in one
    /// range read: what a merge streams through.
    pub(crate) fn slot_vectors(&self, field: i32) -> Result<Cow<'_, [u8]>> {
        let f = self.field_or_err(field)?;
        // ARITH: the vectors region was checked to fit at open.
        #[allow(clippy::arithmetic_side_effects)]
        let len = f.count * f.vector_len();
        self.vectors.read(f.vectors_offset, len)
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
    /// `accept` lets through (all when `None`).
    ///
    /// # Errors
    ///
    /// An unknown field, a query of the wrong dimension, or a read failure.
    // ARITH: cluster sizes, block lengths and slot numbers are bounded by the
    // field's checked regions; `scores` is resized to each block's size.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn search(
        &self,
        field: i32,
        query: &[f32],
        params: &SearchParams,
        accept: Option<&dyn Fn(i32) -> bool>,
    ) -> Result<Vec<ScoredDoc>> {
        let f = self.field_or_err(field)?;
        let query = self.prepare_query(f, query)?;
        if params.k == 0 || f.count == 0 {
            return Ok(Vec::new());
        }
        let probes = self.probe_order(f, &query, params.nprobe.max(1));
        let mut scorer = f.quantizer.query_scorer(&query);
        let uses_centroid = f.quantizer.uses_centroid();
        if !uses_centroid {
            scorer.set_centroid(f.centroid(0));
        }
        let mut top = TopN::new(
            params.k.max(params.rerank).min(f.entries()),
            f.entries() > f.count,
        );
        let mut scores = Vec::new();
        for c in probes {
            let (offset, len) = f.block_range(c);
            let block = self.postings.read(offset, len)?;
            let size = f.cluster_size(c);
            let (slots, codes) = block.split_at(size * 4);
            if uses_centroid {
                scorer.set_centroid(f.centroid(c));
            }
            scores.resize(size, 0f32);
            scorer.score_block(codes, &mut scores);
            for (i, &score) in scores.iter().enumerate() {
                if !top.competitive(score) {
                    continue;
                }
                let slot = le_i32(slots, i * 4) as u32;
                let Some(&doc) = f.slot_docs.get(slot as usize) else {
                    return corrupt(format!("cluster {c}: slot {slot} out of range"));
                };
                if accept.is_some_and(|a| !a(doc)) {
                    continue;
                }
                top.offer(Candidate { score, doc, slot });
            }
        }
        let candidates = top.into_sorted();
        if params.rerank == 0 {
            return Ok(candidates
                .into_iter()
                .take(params.k)
                .map(|c| ScoredDoc {
                    doc: c.doc,
                    score: c.score,
                })
                .collect());
        }
        let mut exact = TopN::new(params.k.min(f.count), false);
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
            ahead(candidates.get(i + AHEAD));
            self.read_vector(f, c.slot as usize, &mut v)?;
            exact.offer(Candidate {
                score: f.similarity.score(&query, &v),
                ..*c
            });
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

    /// The `nprobe` clusters nearest to `query`, nearest first: by L2 for
    /// Euclidean, by dot with the unit centroid for the spherically
    /// clustered Cosine and DotProduct (see `centroid_scale`), and by raw
    /// dot for MaximumInnerProduct, whose score is the dot product itself.
    fn probe_order(&self, f: &IvfField, query: &[f32], nprobe: usize) -> Vec<usize> {
        let euclidean = f.similarity == Similarity::Euclidean;
        let mut scored: Vec<(f32, usize)> = f
            .centroids
            .chunks_exact(f.dimension)
            .enumerate()
            .map(|(c, centroid)| {
                let s = if euclidean {
                    -simd::square_distance_f32(query, centroid)
                } else {
                    let scale = f.centroid_scale.get(c).copied().unwrap_or(1.0);
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
}
