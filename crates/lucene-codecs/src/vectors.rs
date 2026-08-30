//! `Lucene99FlatVectorsFormat`: flat (non-graph) KNN vector storage.
//!
//! Port of
//! `org.apache.lucene.codecs.lucene99.Lucene99FlatVectors{Format,Reader,Writer}`,
//! `org.apache.lucene.codecs.lucene95.{OffHeapFloatVectorValues,
//! OffHeapByteVectorValues,OrdToDocDISIReaderConfiguration}` and the
//! `org.apache.lucene.codecs.hnsw.FlatVectors{Format,Reader,Writer}` /
//! `FlatVectorsScorer` contract they implement, plus
//! `org.apache.lucene.util.VectorUtil` and
//! `org.apache.lucene.index.VectorSimilarityFunction`.
//!
//! The graph that sits *on top* of this store lives in [`crate::hnsw`]
//! (`org.apache.lucene.util.hnsw.*`) and [`crate::hnsw_vectors`]
//! (`Lucene99HnswVectorsFormat`, the `.vem`/`.vex` pair).
//!
//! # Wire format
//!
//! Vectors are addressed by **ordinal**, densely, in ascending doc-id order.
//! The doc id a given ordinal belongs to is recovered from the per-field
//! `OrdToDocDISIReaderConfiguration`: a field every document has a value for
//! is "dense" and needs no mapping at all (ordinal == doc id); a sparse field
//! carries an [`crate::indexed_disi`] bitset (doc -> ordinal) *and* a
//! [`crate::direct_monotonic`] sequence (ordinal -> doc), both stored in the
//! `.vec` file after that field's vectors.
//!
//! `.vec` (vector data):
//! ```text
//! IndexHeader(codec="Lucene99FlatVectorsFormatData", version=0, id, suffix)
//! for each field, in the order given to `write_flat_vectors`:
//!   zero padding to a multiple of 64 bytes (FLOAT32) or 4 bytes (BYTE)
//!   for each ordinal 0..count:
//!     Component_0 .. Component_{dim-1}  --> f32 little-endian, or i8
//!   if sparse (0 < count < maxDoc):
//!     IndexedDISI bitset of the docs that have a value
//!     DirectMonotonicWriter data for the ordinal -> doc mapping
//! Footer
//! ```
//!
//! `.vemf` (vector metadata):
//! ```text
//! IndexHeader(codec="Lucene99FlatVectorsFormatMeta", version=0, id, suffix)
//! for each field:
//!   FieldNumber            --> int32   (little-endian, like every readInt here)
//!   VectorEncoding         --> int32   (ordinal: BYTE=0, FLOAT32=1)
//!   VectorSimilarityFunction --> int32 (ordinal: EUCLIDEAN=0, DOT_PRODUCT=1,
//!                                       COSINE=2, MAXIMUM_INNER_PRODUCT=3)
//!   VectorDataOffset       --> vlong   (absolute offset into .vec)
//!   VectorDataLength       --> vlong   (count * dim * byteSize)
//!   Dimension              --> vint
//!   Count                  --> int32
//!   DocsWithFieldOffset    --> int64   (-2 empty, -1 dense, else offset in .vec)
//!   DocsWithFieldLength    --> int64
//!   JumpTableEntryCount    --> int16
//!   DenseRankPower         --> u8
//!   if DocsWithFieldOffset > -1:       (i.e. sparse)
//!     AddressesOffset      --> int64
//!     BlockShift           --> vint    (16)
//!     DirectMonotonicMeta  --> per block: (min: int64, avg: f32 bits as int32,
//!                                          offset: int64, bpv: u8)
//!     AddressesLength      --> int64
//! -1                       --> int32   (end-of-fields marker)
//! Footer
//! ```
//!
//! # Why the `.vec` file is padded
//!
//! `Lucene99FlatVectorsWriter.alignOutput` aligns each field's vector region
//! to 64 bytes for `FLOAT32` (4 for `BYTE`), because unaligned 64-byte reads
//! cost extra on Arm Neoverse. The padding is part of the format -- the
//! reader finds the region through the recorded offset, so a writer that
//! skipped the padding would still round-trip through itself and still be
//! wrong.

use lucene_store::codec_util::{self, ID_LENGTH};
use lucene_store::data_input::{DataInput, SliceInput};
use lucene_store::data_output::DataOutput;

use crate::direct_monotonic;
use crate::field_infos::{VectorEncoding, VectorSimilarityFunction};
use crate::indexed_disi::{self, DisiCursor};

/// `Lucene99FlatVectorsFormat.META_CODEC_NAME`.
pub const META_CODEC: &str = "Lucene99FlatVectorsFormatMeta";
/// `Lucene99FlatVectorsFormat.VECTOR_DATA_CODEC_NAME`.
pub const DATA_CODEC: &str = "Lucene99FlatVectorsFormatData";
/// `Lucene99FlatVectorsFormat.VERSION_START`.
pub const VERSION_START: i32 = 0;
/// `Lucene99FlatVectorsFormat.VERSION_CURRENT`.
pub const VERSION_CURRENT: i32 = 0;
/// `Lucene99FlatVectorsFormat.DIRECT_MONOTONIC_BLOCK_SHIFT`.
pub const DIRECT_MONOTONIC_BLOCK_SHIFT: u32 = 16;

/// `Lucene99FlatVectorsFormat.META_EXTENSION`.
pub const META_EXTENSION: &str = "vemf";
/// `Lucene99FlatVectorsFormat.VECTOR_DATA_EXTENSION`.
pub const DATA_EXTENSION: &str = "vec";

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Store(#[from] lucene_store::Error),
    #[error("field {0}: vector dimension mismatch: expected {1}, got {2}")]
    DimensionMismatch(i32, i32, i32),
    #[error("field {0}: empty vector not allowed")]
    EmptyVector(i32),
    #[error("unknown field number {0} in vector metadata")]
    UnknownField(i32),
    /// Port of `VectorUtil.{dotProduct,cosine,squareDistance}`'s
    /// `IllegalArgumentException`, whose message begins
    /// `"vector dimensions differ: "` and then concatenates the two lengths
    /// around a `!=`. Every `VectorSimilarityFunction.compare` inherits it. Silently scoring
    /// over the shorter of the two vectors (what a bare `zip` does) returns
    /// a plausible-looking but meaningless number.
    #[error("vector dimensions differ: {0}!={1}")]
    QueryDimensionMismatch(i32, i32),
    /// Port of `Lucene99FlatVectorsReader.getFieldEntry`'s
    /// `IllegalArgumentException`, whose message concatenates the field name
    /// with `"\" is encoded as: "` and then `" expected: "`.
    #[error("field {0} is encoded as {1:?}, expected {2:?}")]
    EncodingMismatch(i32, VectorEncoding, VectorEncoding),
    /// Port of the `FieldEntry` compact-constructor's
    /// `IllegalStateException("Vector data length ... not matching size ...")`
    /// and the other `.vemf` self-consistency checks.
    #[error("corrupt vector metadata: {0}")]
    CorruptMeta(String),
    #[error("vector ordinal {0} out of range (size {1})")]
    OrdOutOfRange(i32, i32),
    /// Port of `HnswGraphBuilder`'s and `Lucene99HnswVectorsFormat`'s
    /// constructor `IllegalArgumentException`s. Distinct from
    /// [`Error::CorruptMeta`] because these are *caller* mistakes, not bad
    /// bytes -- reporting "M must be positive" as "corrupt vector metadata"
    /// sends the reader looking at the wrong file.
    #[error("invalid graph parameter: {0}")]
    InvalidGraphParameter(String),
    /// Port of `VectorUtil.checkFinite`'s `IllegalArgumentException`, whose
    /// message begins `"non-finite value at vector["` and closes with the
    /// index, `"]="` and the value. It is what
    /// `KnnFloatVectorField`'s constructor applies to every indexed vector.
    #[error("field {0}: non-finite value at vector[{1}]={2}")]
    NonFiniteValue(i32, usize, f32),
}

pub type Result<T> = std::result::Result<T, Error>;

fn corrupt<T>(msg: impl Into<String>) -> Result<T> {
    Err(Error::CorruptMeta(msg.into()))
}

/// `VectorEncoding.byteSize`.
fn byte_size(encoding: VectorEncoding) -> usize {
    match encoding {
        VectorEncoding::Byte => 1,
        VectorEncoding::Float32 => 4,
    }
}

/// `Lucene99FlatVectorsWriter.alignOutput`'s alignment per encoding: 64 for
/// FLOAT32 (Arm Neoverse cache-line reads), `Float.BYTES` for BYTE.
fn alignment(encoding: VectorEncoding) -> usize {
    match encoding {
        VectorEncoding::Byte => 4,
        VectorEncoding::Float32 => 64,
    }
}

// ---------------------------------------------------------------------------
// VectorUtil
// ---------------------------------------------------------------------------

/// Port of `VectorUtil.squareDistance(float[], float[])`.
///
/// Eight independent accumulators over `chunks_exact(8)`, not
/// `a.iter().zip(b).map(..).sum()`: float addition is not associative, so a
/// single running `sum` forms a serial dependency chain that LLVM is
/// forbidden to reassociate, and the loop stays scalar. Splitting the sum
/// into eight lanes up front is what lets it emit packed SIMD adds -- the
/// same trick real Lucene applies via the Panama Vector API in
/// `PanamaVectorUtilSupport` (and, 2-wide, in the scalar
/// `DefaultVectorUtilSupport` fallback). The lane split changes the summation
/// order and therefore the last ulp or two of the result, exactly as
/// switching between Lucene's own two implementations does.
pub fn square_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "vector dimensions differ");
    let mut acc = [0.0f32; 8];
    let mut ca = a.chunks_exact(8);
    let mut cb = b.chunks_exact(8);
    for (x, y) in ca.by_ref().zip(cb.by_ref()) {
        for j in 0..8 {
            let d = x[j] - y[j];
            acc[j] += d * d;
        }
    }
    let mut sum = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    for (x, y) in ca.remainder().iter().zip(cb.remainder()) {
        let d = x - y;
        sum += d * d;
    }
    sum
}

/// Port of `VectorUtil.dotProduct(float[], float[])`; see [`square_distance`]
/// for why the sum is split across eight lanes.
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "vector dimensions differ");
    let mut acc = [0.0f32; 8];
    let mut ca = a.chunks_exact(8);
    let mut cb = b.chunks_exact(8);
    for (x, y) in ca.by_ref().zip(cb.by_ref()) {
        for j in 0..8 {
            acc[j] += x[j] * y[j];
        }
    }
    let mut sum = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    for (x, y) in ca.remainder().iter().zip(cb.remainder()) {
        sum += x * y;
    }
    sum
}

/// Port of `DefaultVectorUtilSupport.cosine(float[], float[])`. The final
/// division deliberately mirrors Java's exact shape --
/// `(float) (sum / Math.sqrt((double) norm1 * (double) norm2))`: **one**
/// square root, taken in `f64`, of the *product* of the two squared norms.
///
/// The zero-vector guard is this port's own: Java would return `NaN` here
/// (and asserts `Float.isFinite`), because indexing a zero vector with
/// COSINE is rejected up front by `FieldType`/`KnnFloatVectorField`
/// validation this port doesn't have yet. Returning 0 (a score of 0.5,
/// "no information") keeps every heap's ordering total instead of poisoning
/// it with NaN.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "vector dimensions differ");
    let mut sum_acc = [0.0f32; 8];
    let mut n1_acc = [0.0f32; 8];
    let mut n2_acc = [0.0f32; 8];
    let mut ca = a.chunks_exact(8);
    let mut cb = b.chunks_exact(8);
    for (x, y) in ca.by_ref().zip(cb.by_ref()) {
        for j in 0..8 {
            sum_acc[j] += x[j] * y[j];
            n1_acc[j] += x[j] * x[j];
            n2_acc[j] += y[j] * y[j];
        }
    }
    let fold = |acc: [f32; 8]| {
        ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]))
    };
    let mut sum = fold(sum_acc);
    let mut norm1 = fold(n1_acc);
    let mut norm2 = fold(n2_acc);
    for (x, y) in ca.remainder().iter().zip(cb.remainder()) {
        sum += x * y;
        norm1 += x * x;
        norm2 += y * y;
    }
    if norm1 == 0.0 || norm2 == 0.0 {
        return 0.0;
    }
    (sum as f64 / ((norm1 as f64) * (norm2 as f64)).sqrt()) as f32
}

/// Port of `VectorUtil.squareDistance(byte[], byte[])` -- an **`i32`** sum,
/// exactly as Java: "this will not overflow if dim < 2^18, since
/// max(byte * byte) = 2^14".
///
/// # Byte vectors are signed
///
/// Java's `byte` is signed and Lucene relies on that: a `BYTE`-encoded vector
/// component is in `[-128, 127]`. This crate is `#![forbid(unsafe_code)]`, so
/// it cannot reinterpret the `&[u8]` it mmaps as `&[i8]` for free -- instead
/// every byte kernel takes `&[u8]` straight off the `.vec` file and widens
/// each element through `as i8 as i32`, which is exactly Java's sign
/// extension and costs one `movsx` per element.
pub fn square_distance_bytes(a: &[u8], b: &[u8]) -> i32 {
    debug_assert_eq!(a.len(), b.len(), "vector dimensions differ");
    let mut sum = 0i32;
    for (x, y) in a.iter().zip(b) {
        // ARITH: both operands are sign-extended bytes, so `d` is in
        // -255..=255 and `d * d` in 0..=65_025 -- neither can overflow `i32`.
        // Only the accumulator can, and only for a `dimension` off `.vemf`
        // above ~2^18 (Java's own bound, `VectorUtil`: "this will not overflow
        // if dim < 2^18"). Java accumulates into an `int` and wraps there; the
        // `wrapping_add` is that same semantics rather than a debug panic on a
        // corrupt dimension.
        #[allow(clippy::arithmetic_side_effects)]
        let d = *x as i8 as i32 - *y as i8 as i32;
        // ARITH: `|d| <= 255`, so `d * d <= 65_025`.
        #[allow(clippy::arithmetic_side_effects)]
        let square = d * d;
        sum = sum.wrapping_add(square);
    }
    sum
}

/// Port of `VectorUtil.dotProduct(byte[], byte[])` -- an `i32` sum. See
/// [`square_distance_bytes`] on why `&[u8]` and not `&[i8]`.
pub fn dot_product_bytes(a: &[u8], b: &[u8]) -> i32 {
    debug_assert_eq!(a.len(), b.len(), "vector dimensions differ");
    let mut sum = 0i32;
    for (x, y) in a.iter().zip(b) {
        // ARITH: a product of two sign-extended bytes is in
        // -16_256..=16_384, well inside `i32`. See [`square_distance_bytes`]
        // for why the accumulator wraps rather than panics.
        #[allow(clippy::arithmetic_side_effects)]
        let product = (*x as i8 as i32) * (*y as i8 as i32);
        sum = sum.wrapping_add(product);
    }
    sum
}

/// Port of `DefaultVectorUtilSupport.cosine(byte[], byte[])`: `i32`
/// accumulators, then the same single `f64` square root of the product.
pub fn cosine_bytes(a: &[u8], b: &[u8]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "vector dimensions differ");
    let mut sum = 0i32;
    let mut norm1 = 0i32;
    let mut norm2 = 0i32;
    for (x, y) in a.iter().zip(b) {
        let e1 = *x as i8 as i32;
        let e2 = *y as i8 as i32;
        // ARITH: every product of two sign-extended bytes is in
        // -16_256..=16_384. See [`square_distance_bytes`] for why the three
        // accumulators wrap rather than panic.
        #[allow(clippy::arithmetic_side_effects)]
        let (dot, sq1, sq2) = (e1 * e2, e1 * e1, e2 * e2);
        sum = sum.wrapping_add(dot);
        norm1 = norm1.wrapping_add(sq1);
        norm2 = norm2.wrapping_add(sq2);
    }
    if norm1 == 0 || norm2 == 0 {
        return 0.0;
    }
    (sum as f64 / ((norm1 as f64) * (norm2 as f64)).sqrt()) as f32
}

/// Port of `VectorUtil.scaleMaxInnerProductScore`.
pub fn scale_max_inner_product_score(dot: f32) -> f32 {
    if dot < 0.0 {
        1.0 / (1.0 + -dot)
    } else {
        dot + 1.0
    }
}

/// Port of `VectorUtil.normalizeToUnitInterval`: `max((1 + value) / 2, 0)`.
pub fn normalize_to_unit_interval(value: f32) -> f32 {
    ((1.0 + value) / 2.0).max(0.0)
}

/// Port of `VectorUtil.normalizeDistanceToUnitInterval`: `1 / (1 + d)`.
pub fn normalize_distance_to_unit_interval(square_distance: f32) -> f32 {
    1.0 / (1.0 + square_distance)
}

/// Port of `VectorUtil.dotProductScore(byte[], byte[])`:
/// `0.5f + dot / (len * 2^15)`.
pub fn dot_product_score_bytes(a: &[u8], b: &[u8]) -> f32 {
    // Java is `(float) (a.length * (1 << 15))` -- `int` arithmetic that wraps
    // once the dimension reaches 2^16. `wrapping_mul` reproduces that exactly
    // instead of panicking on a `.vemf` dimension a real writer never emits.
    let denom = (a.len() as i32).wrapping_mul(1 << 15) as f32;
    0.5f32 + dot_product_bytes(a, b) as f32 / denom
}

impl VectorSimilarityFunction {
    /// Port of `VectorSimilarityFunction.compare(float[], float[])`.
    ///
    /// Every function maps its raw comparison onto a range where **higher is
    /// always better**:
    ///
    /// - `Euclidean`: `1 / (1 + squareDistance)`
    /// - `DotProduct`: `max((1 + dot) / 2, 0)` (assumes unit-normalized inputs)
    /// - `Cosine`: `max((1 + cosine) / 2, 0)`
    /// - `MaximumInnerProduct`: `scaleMaxInnerProductScore(dot)`
    pub fn score(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            VectorSimilarityFunction::Euclidean => {
                normalize_distance_to_unit_interval(square_distance(a, b))
            }
            VectorSimilarityFunction::DotProduct => normalize_to_unit_interval(dot_product(a, b)),
            VectorSimilarityFunction::Cosine => normalize_to_unit_interval(cosine(a, b)),
            VectorSimilarityFunction::MaximumInnerProduct => {
                scale_max_inner_product_score(dot_product(a, b))
            }
        }
    }

    /// Port of `VectorSimilarityFunction.compare(byte[], byte[])`.
    ///
    /// Note these are **not** the float formulas with an integer sum
    /// substituted -- Java uses genuinely different transforms for two of the
    /// four:
    ///
    /// - `Euclidean`: `1 / (1 + squareDistance)` (same shape, integer distance)
    /// - `DotProduct`: `dotProductScore` = `0.5 + dot / (dim * 2^15)`, **not**
    ///   `normalizeToUnitInterval` -- byte vectors are not unit-normalized, so
    ///   the raw dot is rescaled by its own maximum magnitude instead
    /// - `Cosine`: `(1 + cosine) / 2`, **without** the `max(_, 0)` clamp the
    ///   float branch has
    /// - `MaximumInnerProduct`: `scaleMaxInnerProductScore(dot)`
    pub fn score_bytes(&self, a: &[u8], b: &[u8]) -> f32 {
        match self {
            VectorSimilarityFunction::Euclidean => 1.0 / (1.0 + square_distance_bytes(a, b) as f32),
            VectorSimilarityFunction::DotProduct => dot_product_score_bytes(a, b),
            VectorSimilarityFunction::Cosine => (1.0 + cosine_bytes(a, b)) / 2.0,
            VectorSimilarityFunction::MaximumInnerProduct => {
                scale_max_inner_product_score(dot_product_bytes(a, b) as f32)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// OrdToDocDISIReaderConfiguration
// ---------------------------------------------------------------------------

/// Port of `org.apache.lucene.codecs.lucene95.OrdToDocDISIReaderConfiguration`:
/// how a field's vector ordinals map to document ids.
#[derive(Debug, Clone)]
pub enum OrdToDoc {
    /// `docsWithFieldOffset == -2`: no vectors at all.
    Empty,
    /// `docsWithFieldOffset == -1`: every document has a value, so ordinal
    /// and doc id are the same number and no mapping is stored.
    Dense,
    /// Some documents are missing a value: an `IndexedDISI` bitset gives
    /// doc -> ordinal, a `DirectMonotonicReader` gives ordinal -> doc.
    Sparse {
        docs_with_field_offset: i64,
        docs_with_field_length: i64,
        jump_table_entry_count: i16,
        dense_rank_power: u8,
        addresses_offset: i64,
        addresses_length: i64,
        meta: direct_monotonic::Meta,
    },
}

/// `[offset, offset + length)` of `file`, or `None` if that is not a range
/// inside it.
///
/// The sparse `ordToDoc` structures are the one place in this format where a
/// region's offset and length are read off `.vemf` with **nothing on the wire
/// relating them to the `.vec` they address** -- `read_field_entry` proves the
/// `size * dim * byteSize == vectorDataLength` identity for the vector data
/// and validates that region against the file, but the two `ordToDoc` regions
/// get neither treatment.
///
/// The guard they used to carry formed the very sum it existed to guard
/// (`if start + length > file.len()`, the shape `docs/arithmetic-gate.md`
/// names): a negative `addressesOffset` arrives as `usize::MAX` through
/// `as usize`, the addition wraps to something small, the comparison
/// *passes*, and the slice then panics with `start > end`. c30 reached it
/// from `check_index`'s side with a re-signed `.vemf` overwrite -- and this
/// decoder is on the query path, so through the FFI that is a dead JVM.
fn file_region(file: &[u8], offset: i64, length: i64) -> Option<&[u8]> {
    let start = usize::try_from(offset).ok()?;
    let end = start.checked_add(usize::try_from(length).ok()?)?;
    file.get(start..end)
}

impl OrdToDoc {
    /// Port of `OrdToDocDISIReaderConfiguration.fromStoredMeta`.
    fn from_stored_meta(input: &mut SliceInput<'_>, size: i32) -> Result<Self> {
        let docs_with_field_offset = input.read_i64()?;
        let docs_with_field_length = input.read_i64()?;
        let jump_table_entry_count = input.read_i16()?;
        let dense_rank_power = input.read_byte()?;
        if docs_with_field_offset > -1 {
            let addresses_offset = input.read_i64()?;
            let block_shift = input.read_vint()?;
            if !(0..=31).contains(&block_shift) {
                return corrupt(format!("illegal DirectMonotonic block shift {block_shift}"));
            }
            let meta = direct_monotonic::load_meta(input, size as i64, block_shift as u32)?;
            let addresses_length = input.read_i64()?;
            Ok(OrdToDoc::Sparse {
                docs_with_field_offset,
                docs_with_field_length,
                jump_table_entry_count,
                dense_rank_power,
                addresses_offset,
                addresses_length,
                meta,
            })
        } else if docs_with_field_offset == -1 {
            Ok(OrdToDoc::Dense)
        } else if docs_with_field_offset == -2 {
            Ok(OrdToDoc::Empty)
        } else {
            corrupt(format!(
                "illegal docsWithFieldOffset {docs_with_field_offset}"
            ))
        }
    }

    /// Port of `OrdToDocDISIReaderConfiguration.isEmpty`.
    pub fn is_empty(&self) -> bool {
        matches!(self, OrdToDoc::Empty)
    }

    /// Port of `OrdToDocDISIReaderConfiguration.isDense`.
    pub fn is_dense(&self) -> bool {
        matches!(self, OrdToDoc::Dense)
    }
}

/// Port of `OrdToDocDISIReaderConfiguration.writeStoredMeta`. Appends the
/// sparse structures to `data` (the `.vec` payload being built) and the
/// four/eight metadata fields to `meta`.
///
/// `docs` must be strictly ascending; `count` is `docs.len()`.
fn write_stored_meta(meta: &mut Vec<u8>, data: &mut Vec<u8>, docs: &[i32], max_doc: i32) {
    let count = docs.len() as i32;
    if count == 0 {
        meta.write_i64(-2); // docsWithFieldOffset
        meta.write_i64(0); // docsWithFieldLength
        meta.write_i16(-1); // jumpTableEntryCount
        meta.write_byte(0xFF); // denseRankPower == (byte) -1
    } else if count == max_doc {
        meta.write_i64(-1);
        meta.write_i64(0);
        meta.write_i16(-1);
        meta.write_byte(0xFF);
    } else {
        let offset = data.len() as i64;
        meta.write_i64(offset);
        let (disi, jump_table_entry_count) =
            indexed_disi::write_with_dense_rank_power(docs, indexed_disi::DEFAULT_DENSE_RANK_POWER);
        data.extend_from_slice(&disi);
        // ARITH: `offset` was `data.len()` before the `extend_from_slice`
        // above and `data` only grows, so the difference is the number of
        // bytes just appended and is >= 0. The length spans the block jump
        // table too -- it is what the reader subtracts the table's bytes from
        // (`createBlockSlice`).
        #[allow(clippy::arithmetic_side_effects)]
        let disi_len = data.len() as i64 - offset;
        meta.write_i64(disi_len);
        meta.write_i16(jump_table_entry_count);
        meta.write_byte(indexed_disi::DEFAULT_DENSE_RANK_POWER);

        let start = data.len() as i64;
        meta.write_i64(start);
        meta.write_vint(DIRECT_MONOTONIC_BLOCK_SHIFT as i32);
        let values: Vec<i64> = docs.iter().map(|d| *d as i64).collect();
        let (dm_meta, dm_data) = direct_monotonic::write(&values, DIRECT_MONOTONIC_BLOCK_SHIFT);
        meta.extend_from_slice(&dm_meta);
        data.extend_from_slice(&dm_data);
        // ARITH: `start` was `data.len()` before the `extend_from_slice`
        // above; same argument as `disi_len`.
        #[allow(clippy::arithmetic_side_effects)]
        let addresses_len = data.len() as i64 - start;
        meta.write_i64(addresses_len);
    }
}

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

/// One field's `.vemf` entry: Java's `Lucene99FlatVectorsReader.FieldEntry`.
#[derive(Debug, Clone)]
pub struct FlatFieldEntry {
    pub field_number: i32,
    pub encoding: VectorEncoding,
    pub similarity: VectorSimilarityFunction,
    pub vector_data_offset: i64,
    pub vector_data_length: i64,
    pub dimension: i32,
    pub size: i32,
    pub ord_to_doc: OrdToDoc,
}

/// Port of `Lucene99FlatVectorsReader`. Holds a borrow of the whole `.vec`
/// file; every vector read is a slice of it, so a 1M x 768 field costs zero
/// heap here (Java holds one `IndexInput` for exactly the same reason).
#[derive(Debug, Clone)]
pub struct FlatVectorsReader<'a> {
    data: &'a [u8],
    fields: Vec<FlatFieldEntry>,
}

impl<'a> FlatVectorsReader<'a> {
    /// Port of the `Lucene99FlatVectorsReader` constructor + `readFields`.
    ///
    /// `meta_buf` is the `.vemf` file, `data_buf` the `.vec` file.
    pub fn open(
        meta_buf: &[u8],
        data_buf: &'a [u8],
        segment_id: &[u8; ID_LENGTH],
        segment_suffix: &str,
    ) -> Result<Self> {
        let mut meta = SliceInput::new(meta_buf);
        let meta_version = codec_util::check_index_header(
            &mut meta,
            META_CODEC,
            VERSION_START,
            VERSION_CURRENT,
            segment_id,
            segment_suffix,
        )?;
        let mut data_in = SliceInput::new(data_buf);
        let data_version = codec_util::check_index_header(
            &mut data_in,
            DATA_CODEC,
            VERSION_START,
            VERSION_CURRENT,
            segment_id,
            segment_suffix,
        )?;
        let (meta_version, data_version) = (meta_version.version, data_version.version);
        if meta_version != data_version {
            return corrupt(format!(
                "Format versions mismatch: meta={meta_version}, {DATA_CODEC}={data_version}"
            ));
        }
        // A file shorter than its own footer is corruption, not a subtraction:
        // `len - FOOTER_LENGTH` on a truncated `.vemf` underflows to a huge
        // offset (release) or panics (debug).
        let (Some(meta_footer), Some(data_footer)) = (
            meta_buf.len().checked_sub(codec_util::FOOTER_LENGTH),
            data_buf.len().checked_sub(codec_util::FOOTER_LENGTH),
        ) else {
            return corrupt(format!(
                "vector files shorter than a codec footer: .vemf={}, .vec={}",
                meta_buf.len(),
                data_buf.len()
            ));
        };
        codec_util::check_whole_file_footer(meta_buf, meta_footer)?;
        codec_util::check_whole_file_footer(data_buf, data_footer)?;

        let mut fields = Vec::new();
        loop {
            let field_number = meta.read_i32()?;
            if field_number == -1 {
                break;
            }
            if field_number < 0 {
                return corrupt(format!("Invalid field number: {field_number}"));
            }
            fields.push(read_field_entry(&mut meta, field_number, data_buf.len())?);
        }

        Ok(FlatVectorsReader {
            data: data_buf,
            fields,
        })
    }

    pub fn fields(&self) -> &[FlatFieldEntry] {
        &self.fields
    }

    pub fn field(&self, field_number: i32) -> Option<&FlatFieldEntry> {
        self.fields.iter().find(|f| f.field_number == field_number)
    }

    fn entry(&self, field_number: i32, expected: VectorEncoding) -> Result<&FlatFieldEntry> {
        let entry = self
            .field(field_number)
            .ok_or(Error::UnknownField(field_number))?;
        if entry.encoding != expected {
            return Err(Error::EncodingMismatch(
                field_number,
                entry.encoding,
                expected,
            ));
        }
        Ok(entry)
    }

    /// Port of `Lucene99FlatVectorsReader.getFloatVectorValues` ->
    /// `OffHeapFloatVectorValues.load`.
    pub fn float_vector_values(&self, field_number: i32) -> Result<FloatVectorValues<'a>> {
        let entry = self.entry(field_number, VectorEncoding::Float32)?;
        Ok(FloatVectorValues {
            values: self.raw_values(entry)?,
        })
    }

    /// Port of `Lucene99FlatVectorsReader.getByteVectorValues` ->
    /// `OffHeapByteVectorValues.load`.
    pub fn byte_vector_values(&self, field_number: i32) -> Result<ByteVectorValues<'a>> {
        let entry = self.entry(field_number, VectorEncoding::Byte)?;
        Ok(ByteVectorValues {
            values: self.raw_values(entry)?,
        })
    }

    fn raw_values(&self, entry: &FlatFieldEntry) -> Result<RawVectorValues<'a>> {
        // `start + len` is formed with `checked_add`, not compared after the
        // fact: a `vectorDataOffset` near `usize::MAX` would otherwise wrap the
        // sum to something small and walk straight past the guard.
        let start = entry.vector_data_offset as usize;
        let Some(region) = start
            .checked_add(entry.vector_data_length as usize)
            .and_then(|end| self.data.get(start..end))
        else {
            return corrupt(format!(
                "vector data [{start}, +{}) out of bounds for a {} byte .vec file",
                entry.vector_data_length,
                self.data.len()
            ));
        };
        // Both factors are validated in `read_field_entry`
        // (`0 < dimension <= i32::MAX`, `byteSize <= 4`), and their product is
        // `vectorDataLength / size` for a non-empty field, so it is bounded by
        // the `.vec` file itself. `checked_mul` covers the `size == 0` case,
        // where the identity check says nothing about `dimension`, and a
        // 32-bit `usize`.
        let Some(vector_bytes) = (entry.dimension as usize).checked_mul(byte_size(entry.encoding))
        else {
            return corrupt(format!(
                "vector dimension {} too large for this platform",
                entry.dimension
            ));
        };
        Ok(RawVectorValues {
            slice: region,
            file: self.data,
            dimension: entry.dimension as usize,
            vector_bytes,
            size: entry.size,
            similarity: entry.similarity,
            ord_to_doc: entry.ord_to_doc.clone(),
        })
    }
}

/// `FieldEntry.create` plus the compact constructor's validation. Java
/// cross-checks against `FieldInfos`; this port has no `FieldInfos` here, so
/// it checks what the file can check about itself -- the `size * dim *
/// byteSize == vectorDataLength` identity Java asserts, which catches a
/// truncated or misaligned region before any read does.
fn read_field_entry(
    meta: &mut SliceInput<'_>,
    field_number: i32,
    data_len: usize,
) -> Result<FlatFieldEntry> {
    let encoding = read_vector_encoding(meta)?;
    let similarity = read_similarity_function(meta)?;
    let vector_data_offset = meta.read_vlong()?;
    let vector_data_length = meta.read_vlong()?;
    let dimension = meta.read_vint()?;
    let size = meta.read_i32()?;
    if dimension <= 0 {
        return corrupt(format!("illegal vector dimension {dimension}"));
    }
    if size < 0 {
        return corrupt(format!("illegal vector count {size}"));
    }
    if vector_data_offset < 0 || vector_data_length < 0 {
        return corrupt(format!(
            "illegal vector data region [{vector_data_offset}, +{vector_data_length})"
        ));
    }
    // `Math.multiplyExact` in Java's `FieldEntry` compact constructor. The
    // product of two `i32`s and a byte size genuinely overflows `i64`
    // (`(2^31-1) * 4 * (2^31-1) > 2^63`), and a wrapped `expected` is not
    // merely a wrong number: it is one an attacker picks to make the identity
    // below hold for an absurd `dimension`/`size` pair.
    let Some(expected) = (dimension as i64)
        .checked_mul(byte_size(encoding) as i64)
        .and_then(|v| v.checked_mul(size as i64))
    else {
        return corrupt(format!(
            "vector data length overflows: size={size} * dim={dimension} * byteSize={}",
            byte_size(encoding)
        ));
    };
    if expected != vector_data_length {
        return corrupt(format!(
            "Vector data length {vector_data_length} not matching size={size} * dim={dimension} \
             * byteSize={} = {expected}",
            byte_size(encoding)
        ));
    }
    // ARITH: both operands are validated non-negative `i64` above, so each
    // widens to a `u128` below 2^63 and the sum is below 2^64 -- a `u128`
    // cannot overflow here, and unlike an `i64`/`usize` sum it cannot wrap
    // past the very bound it is being compared against.
    #[allow(clippy::arithmetic_side_effects)]
    let region_end = vector_data_offset as u128 + vector_data_length as u128;
    if region_end > data_len as u128 {
        return corrupt(format!(
            "vector data region [{vector_data_offset}, +{vector_data_length}) past the end of a \
             {data_len} byte .vec file"
        ));
    }
    let ord_to_doc = OrdToDoc::from_stored_meta(meta, size)?;
    Ok(FlatFieldEntry {
        field_number,
        encoding,
        similarity,
        vector_data_offset,
        vector_data_length,
        dimension,
        size,
        ord_to_doc,
    })
}

/// Port of `Lucene99HnswVectorsReader.readVectorEncoding` (shared by the flat
/// reader): a **4-byte** ordinal, not a single byte.
pub(crate) fn read_vector_encoding(input: &mut SliceInput<'_>) -> Result<VectorEncoding> {
    match input.read_i32()? {
        0 => Ok(VectorEncoding::Byte),
        1 => Ok(VectorEncoding::Float32),
        // Lucene 10.5.0's `VectorEncoding` has exactly two constants,
        // BYTE (ordinal 0) and FLOAT32 (ordinal 1); a third ordinal is
        // corruption. (Lucene `main` appends FLOAT16 as ordinal 2 -- a
        // post-10.5.0 feature this port pins away from and must reject.)
        other => corrupt(format!("Invalid vector encoding id: {other}")),
    }
}

/// Port of `Lucene99HnswVectorsReader.readSimilarityFunction`, whose ordinal
/// list (`SIMILARITY_FUNCTIONS`) is pinned to
/// `Lucene94FieldInfosFormat`'s and deliberately independent of the Java
/// enum's declaration order.
pub(crate) fn read_similarity_function(
    input: &mut SliceInput<'_>,
) -> Result<VectorSimilarityFunction> {
    match input.read_i32()? {
        0 => Ok(VectorSimilarityFunction::Euclidean),
        1 => Ok(VectorSimilarityFunction::DotProduct),
        2 => Ok(VectorSimilarityFunction::Cosine),
        3 => Ok(VectorSimilarityFunction::MaximumInnerProduct),
        other => corrupt(format!("invalid distance function: {other}")),
    }
}

/// Encoding/similarity ordinals as written to `.vemf`/`.vem`.
pub(crate) fn encoding_ordinal(e: VectorEncoding) -> i32 {
    match e {
        VectorEncoding::Byte => 0,
        VectorEncoding::Float32 => 1,
    }
}

pub(crate) fn similarity_ordinal(s: VectorSimilarityFunction) -> i32 {
    match s {
        VectorSimilarityFunction::Euclidean => 0,
        VectorSimilarityFunction::DotProduct => 1,
        VectorSimilarityFunction::Cosine => 2,
        VectorSimilarityFunction::MaximumInnerProduct => 3,
    }
}

/// The encoding-independent half of `OffHeap{Float,Byte}VectorValues`.
#[derive(Debug, Clone)]
struct RawVectorValues<'a> {
    /// Just this field's vector bytes.
    slice: &'a [u8],
    /// The whole `.vec` file, needed for the sparse ord->doc structures
    /// (which live outside `slice`).
    file: &'a [u8],
    dimension: usize,
    /// `dimension * VectorEncoding.byteSize`, i.e. one vector's stride in
    /// `slice`. Computed once when the field is opened rather than at every
    /// `vector(ord)`: the encoding is fixed per field, and the multiply was
    /// otherwise on the per-candidate KNN path.
    vector_bytes: usize,
    size: i32,
    similarity: VectorSimilarityFunction,
    ord_to_doc: OrdToDoc,
}

impl RawVectorValues<'_> {
    /// One vector's bytes. `slice.len() == size * vector_bytes` is the
    /// identity `read_field_entry` enforced against `.vemf`, so an in-range
    /// ordinal always addresses a whole vector -- but the slice is taken with
    /// `get`, not `[..]`, so a future caller that breaks the identity gets an
    /// error rather than an index panic in a release build.
    fn bytes(&self, ord: i32) -> Result<&[u8]> {
        if ord < 0 || ord >= self.size {
            return Err(Error::OrdOutOfRange(ord, self.size));
        }
        // ARITH: `read_field_entry` enforces
        // `vectorDataLength == size * dimension * byteSize`, and `raw_values`
        // slices exactly `vectorDataLength` bytes with the same `byteSize`, so
        // `slice.len() == size * vector_bytes` exactly. With `0 <= ord < size`
        // that makes `(ord + 1) * vector_bytes <= slice.len()`: neither the
        // product nor the sum can overflow, both being bounded by the `.vec`
        // file's own length. `get` rather than `[..]` so that a future caller
        // who breaks the identity gets an error instead of an index panic --
        // and `get` performs exactly the bounds check `[..]` would, so it
        // costs nothing (a `checked_add` here measured 6% on a 20k-vector
        // scan; the proof above is what makes it unnecessary).
        #[allow(clippy::arithmetic_side_effects)]
        let (start, end) = {
            let start = ord as usize * self.vector_bytes;
            (start, start + self.vector_bytes)
        };
        // A `match`, not `.ok_or(..)`: `Error` carries `String` variants and
        // is wide, so an eagerly built error value is materialised on the
        // *success* path too -- 40% of this call in c31's A/B, on the
        // per-candidate KNN scoring path. (`ok_or_else` reads better but trips
        // `clippy::unnecessary_lazy_evaluations`, which does not know the
        // enum's size.)
        match self.slice.get(start..end) {
            Some(v) => Ok(v),
            None => Err(Error::OrdOutOfRange(ord, self.size)),
        }
    }

    /// The bytes of a contiguous ordinal range, for the merge's bulk copy.
    fn raw_range(&self, from_ord: i32, to_ord: i32) -> Result<&[u8]> {
        if from_ord < 0 || to_ord > self.size || from_ord > to_ord {
            return Err(Error::OrdOutOfRange(from_ord.max(to_ord), self.size));
        }
        // ARITH: as in `bytes` -- both ordinals are in `0..=size` and
        // `size * vector_bytes == slice.len()`.
        #[allow(clippy::arithmetic_side_effects)]
        let (start, end) = (
            from_ord as usize * self.vector_bytes,
            to_ord as usize * self.vector_bytes,
        );
        match self.slice.get(start..end) {
            Some(v) => Ok(v),
            None => Err(Error::OrdOutOfRange(to_ord, self.size)),
        }
    }

    /// Port of `KnnVectorValues.ordToDoc`.
    fn ord_to_doc(&self, ord: i32) -> Result<i32> {
        if ord < 0 || ord >= self.size {
            return Err(Error::OrdOutOfRange(ord, self.size));
        }
        match &self.ord_to_doc {
            OrdToDoc::Empty | OrdToDoc::Dense => Ok(ord),
            OrdToDoc::Sparse {
                addresses_offset,
                addresses_length,
                meta,
                ..
            } => {
                let Some(region) = file_region(self.file, *addresses_offset, *addresses_length)
                else {
                    return corrupt(format!(
                        "ordToDoc addresses region [{addresses_offset}, +{addresses_length}) is \
                         not inside a {} byte .vec file",
                        self.file.len()
                    ));
                };
                Ok(direct_monotonic::get(region, meta, ord as i64)? as i32)
            }
        }
    }
}

/// A cursor resolving **doc id -> ordinal**, the inverse of
/// [`FloatVectorValues::ord_to_doc`]. Port of what
/// `OrdToDocDISIReaderConfiguration.getIndexedDISI` is used for.
///
/// Forward-only over doc ids, like the underlying [`DisiCursor`]; call
/// [`reset`](DocToOrdCursor::reset) to go backwards.
#[derive(Debug)]
pub enum DocToOrdCursor<'a> {
    Empty,
    Dense { size: i32 },
    Sparse(Box<DisiCursor<'a>>),
}

impl DocToOrdCursor<'_> {
    /// The ordinal of `doc`, or `None` when `doc` has no vector.
    pub fn ordinal(&mut self, doc: i32) -> Result<Option<i32>> {
        match self {
            DocToOrdCursor::Empty => Ok(None),
            DocToOrdCursor::Dense { size } => Ok((doc >= 0 && doc < *size).then_some(doc)),
            DocToOrdCursor::Sparse(cursor) => Ok(cursor.advance_exact(doc)?.map(|o| o as i32)),
        }
    }

    pub fn reset(&mut self) {
        if let DocToOrdCursor::Sparse(cursor) = self {
            cursor.reset();
        }
    }
}

macro_rules! vector_values_common {
    ($t:ty) => {
        impl $t {
            /// `KnnVectorValues.dimension()`.
            pub fn dimension(&self) -> usize {
                self.values.dimension
            }

            /// `KnnVectorValues.size()`: the number of vectors, i.e. the
            /// number of documents that have a value for this field.
            pub fn size(&self) -> i32 {
                self.values.size
            }

            pub fn is_empty(&self) -> bool {
                self.values.size == 0
            }

            pub fn similarity(&self) -> VectorSimilarityFunction {
                self.values.similarity
            }

            /// `KnnVectorValues.ordToDoc(ord)`.
            pub fn ord_to_doc(&self, ord: i32) -> Result<i32> {
                self.values.ord_to_doc(ord)
            }

            /// Opens the doc -> ordinal direction. Cheap: no allocation for a
            /// dense or empty field, one `DisiCursor` otherwise.
            pub fn doc_to_ord(&self) -> Result<DocToOrdCursor<'_>> {
                match &self.values.ord_to_doc {
                    OrdToDoc::Empty => Ok(DocToOrdCursor::Empty),
                    // A dense field's ordinals are its doc ids, and every doc
                    // in `0..size` has one.
                    OrdToDoc::Dense => Ok(DocToOrdCursor::Dense {
                        size: self.values.size,
                    }),
                    OrdToDoc::Sparse {
                        docs_with_field_offset,
                        docs_with_field_length,
                        jump_table_entry_count,
                        dense_rank_power,
                        ..
                    } => {
                        let Some(region) = file_region(
                            self.values.file,
                            *docs_with_field_offset,
                            *docs_with_field_length,
                        ) else {
                            return corrupt(format!(
                                "docsWithField region [{docs_with_field_offset}, \
                                 +{docs_with_field_length}) is not inside a {} byte .vec file",
                                self.values.file.len()
                            ));
                        };
                        Ok(DocToOrdCursor::Sparse(Box::new(DisiCursor::new(
                            region,
                            *dense_rank_power,
                            *jump_table_entry_count,
                        ))))
                    }
                }
            }
        }
    };
}

/// Port of `org.apache.lucene.codecs.lucene95.OffHeapFloatVectorValues`.
#[derive(Debug, Clone)]
pub struct FloatVectorValues<'a> {
    values: RawVectorValues<'a>,
}

vector_values_common!(FloatVectorValues<'_>);

impl FloatVectorValues<'_> {
    /// `FloatVectorValues.vectorValue(ord)`, decoded into a caller-owned
    /// buffer so a scan allocates nothing.
    ///
    /// `out.len()` must be exactly `dimension()`. A wrong-sized buffer is an
    /// error rather than a silent partial fill: the zip below would otherwise
    /// stop at the shorter of the two and leave a plausible, wrong vector --
    /// the same failure mode b7 found in the old `search`, which scored the
    /// common prefix of a mismatched query.
    pub fn vector_into(&self, ord: i32, out: &mut [f32]) -> Result<()> {
        if out.len() != self.values.dimension {
            return Err(Error::QueryDimensionMismatch(
                self.values.dimension as i32,
                out.len() as i32,
            ));
        }
        let bytes = self.values.bytes(ord)?;
        for (slot, chunk) in out.iter_mut().zip(bytes.chunks_exact(4)) {
            *slot = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        Ok(())
    }

    /// The `.vec` bytes of ordinals `from..to`, verbatim.
    ///
    /// `.vec` stores FLOAT32 components as little-endian `f32`, which is
    /// byte-for-byte what this port would write for the same values -- so a
    /// merge can copy a whole run of vectors straight across with one memcpy
    /// instead of decoding and re-encoding each one (Java re-encodes through a
    /// `ByteBuffer` per vector).
    pub fn raw_range(&self, from_ord: i32, to_ord: i32) -> Result<&[u8]> {
        self.values.raw_range(from_ord, to_ord)
    }

    /// Allocating convenience wrapper around [`vector_into`](Self::vector_into).
    pub fn vector(&self, ord: i32) -> Result<Vec<f32>> {
        let mut out = vec![0.0f32; self.values.dimension];
        self.vector_into(ord, &mut out)?;
        Ok(out)
    }

    /// Exhaustive (brute-force, exact) top-`k` search over every stored
    /// vector, returning `(doc_id, score)` descending by score with doc id as
    /// the tie-break -- the same answer Java produces when
    /// `Lucene99HnswVectorsReader.search` takes its non-HNSW branch.
    ///
    /// `O(size * dimension)`. [`crate::hnsw_vectors`] is the approximate,
    /// `O(log size)` alternative.
    pub fn exhaustive_search(&self, query: &[f32], k: usize) -> Result<Vec<(i32, f32)>> {
        if query.len() != self.values.dimension {
            return Err(Error::QueryDimensionMismatch(
                self.values.dimension as i32,
                query.len() as i32,
            ));
        }
        let mut heap = crate::hnsw::NeighborQueue::new(k.max(1), false);
        if k == 0 {
            return Ok(Vec::new());
        }
        let mut scratch = vec![0.0f32; self.values.dimension];
        for ord in 0..self.values.size {
            self.vector_into(ord, &mut scratch)?;
            heap.insert_with_overflow(ord, self.values.similarity.score(query, &scratch));
        }
        collect_descending(heap, |ord| self.values.ord_to_doc(ord))
    }
}

/// Port of `org.apache.lucene.codecs.lucene95.OffHeapByteVectorValues`.
#[derive(Debug, Clone)]
pub struct ByteVectorValues<'a> {
    values: RawVectorValues<'a>,
}

vector_values_common!(ByteVectorValues<'_>);

impl ByteVectorValues<'_> {
    /// `ByteVectorValues.vectorValue(ord)` -- a borrow straight into the
    /// `.vec` bytes, since a byte vector needs no decoding at all.
    pub fn vector(&self, ord: i32) -> Result<&[u8]> {
        self.values.bytes(ord)
    }

    /// The `.vec` bytes of ordinals `from..to`, verbatim -- see
    /// [`FloatVectorValues::raw_range`].
    pub fn raw_range(&self, from_ord: i32, to_ord: i32) -> Result<&[u8]> {
        self.values.raw_range(from_ord, to_ord)
    }

    /// Exhaustive (brute-force, exact) top-`k` search; see
    /// [`FloatVectorValues::exhaustive_search`].
    pub fn exhaustive_search(&self, query: &[u8], k: usize) -> Result<Vec<(i32, f32)>> {
        if query.len() != self.values.dimension {
            return Err(Error::QueryDimensionMismatch(
                self.values.dimension as i32,
                query.len() as i32,
            ));
        }
        if k == 0 {
            return Ok(Vec::new());
        }
        let mut heap = crate::hnsw::NeighborQueue::new(k, false);
        for ord in 0..self.values.size {
            let v = self.vector(ord)?;
            heap.insert_with_overflow(ord, self.values.similarity.score_bytes(query, v));
        }
        collect_descending(heap, |ord| self.values.ord_to_doc(ord))
    }
}

/// Drains a `NeighborQueue` (min-heap on score) into `(doc, score)` pairs
/// sorted descending by score, doc id ascending on ties -- Lucene's
/// `TopDocs` ordering.
fn collect_descending(
    mut heap: crate::hnsw::NeighborQueue,
    ord_to_doc: impl Fn(i32) -> Result<i32>,
) -> Result<Vec<(i32, f32)>> {
    let mut out = Vec::with_capacity(heap.size());
    while heap.size() > 0 {
        let score = heap.top_score();
        let ord = heap.pop();
        out.push((ord_to_doc(ord)?, score));
    }
    out.reverse();
    Ok(out)
}

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

/// One field's vectors, as handed to [`write_flat_vectors`].
#[derive(Debug, Clone)]
pub struct FlatVectorsField {
    pub field_number: i32,
    pub similarity: VectorSimilarityFunction,
    pub dimension: i32,
    /// The doc ids that have a vector, strictly ascending. Ordinal `i` is
    /// `docs[i]`.
    pub docs: Vec<i32>,
    pub values: FieldVectorData,
}

/// A field's vector components, laid out flat: `docs.len() * dimension`
/// values in ordinal-major order. The variant also fixes the field's
/// `VectorEncoding`, so the two can never disagree.
#[derive(Debug, Clone)]
pub enum FieldVectorData {
    Float32(Vec<f32>),
    Byte(Vec<u8>),
}

impl FieldVectorData {
    pub fn encoding(&self) -> VectorEncoding {
        match self {
            FieldVectorData::Float32(_) => VectorEncoding::Float32,
            FieldVectorData::Byte(_) => VectorEncoding::Byte,
        }
    }

    fn component_count(&self) -> usize {
        match self {
            FieldVectorData::Float32(v) => v.len(),
            FieldVectorData::Byte(v) => v.len(),
        }
    }
}

/// Port of `Lucene99FlatVectorsWriter`: accumulates the `.vec` data file and
/// the `.vemf` metadata file one field at a time.
///
/// Java's writer buffers each field's vectors on the heap
/// (`FlatFieldVectorsWriter`) and serializes them in `flush`; this port's
/// caller already has a finished field in hand, so [`Self::write_field`] takes
/// one and encodes it immediately. The *merge* entry point
/// ([`Self::merge_one_flat_vector_field`]) never materializes a field at all --
/// it copies bytes straight from the source segments, which is the same
/// property Java's `mergeOneFlatVectorField` has ("we can just write the
/// vectors directly to the new segment") and the reason it is a separate entry
/// point rather than a `write_field` caller.
pub struct FlatVectorsWriter {
    data: Vec<u8>,
    meta: Vec<u8>,
    max_doc: i32,
}

impl FlatVectorsWriter {
    /// `max_doc` is the segment's document count -- it is what decides whether
    /// a field is written dense (`count == maxDoc`, no ord->doc structures at
    /// all) or sparse.
    pub fn new(max_doc: i32, segment_id: &[u8; ID_LENGTH], segment_suffix: &str) -> Self {
        let mut data = Vec::new();
        codec_util::write_index_header(
            &mut data,
            DATA_CODEC,
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
        FlatVectorsWriter {
            data,
            meta,
            max_doc,
        }
    }

    /// `Lucene99FlatVectorsWriter.writeField`: one field, from an in-memory
    /// batch of vectors.
    pub fn write_field(&mut self, field: &FlatVectorsField) -> Result<()> {
        validate_field(field, self.max_doc)?;
        let encoding = field.values.encoding();
        let vector_data_offset = self.align_output(encoding);
        match &field.values {
            FieldVectorData::Float32(v) => {
                for component in v {
                    self.data.extend_from_slice(&component.to_le_bytes());
                }
            }
            FieldVectorData::Byte(v) => self.data.extend_from_slice(v),
        }
        // ARITH: `align_output` returned `self.data.len()` and `self.data`
        // only grows, so this is the byte count just appended.
        #[allow(clippy::arithmetic_side_effects)]
        let vector_data_length = self.data.len() as i64 - vector_data_offset;
        self.write_meta(
            field.field_number,
            encoding,
            field.similarity,
            field.dimension,
            vector_data_offset,
            vector_data_length,
            &field.docs,
        );
        Ok(())
    }

    /// Port of `Lucene99FlatVectorsWriter.mergeOneFlatVectorField` (plus the
    /// `MergedVectorValues.merge*VectorValues` iteration it consumes): one
    /// field's vectors, taken from the segments being merged and written
    /// straight into this segment's `.vec`.
    ///
    /// Two properties Java has and this keeps. First, **no decoded buffering**:
    /// no vector is ever materialized, so merging a 1M-vector x 768-dim field
    /// costs one 12-byte plan entry per vector on top of the output buffer,
    /// not one decoded vector. Second, the merged ordinal space is defined
    /// here and nowhere else, which is exactly the space
    /// [`crate::hnsw_vectors::merge_one_field`]'s ordinal maps are resolved
    /// against.
    ///
    /// **Ordinals are assigned in merged-document order**, which is Java's
    /// `DocIDMerger.of(subs, mergeState.needsIndexSort)`: without an index
    /// sort that is source order (each source owns a contiguous, increasing
    /// merged range, so the scan below finds the plan already ascending and
    /// moves nothing); with one the sources interleave, and source order would
    /// produce a descending step in `.vemf`'s `IndexedDISI` doc list.
    ///
    /// It goes one step further than Java: a run of consecutive ordinals that
    /// all survive is copied with a single `memcpy` from the source's mapped
    /// `.vec`, because the on-disk representation of a FLOAT32 vector is
    /// little-endian `f32` on both sides. Java re-encodes every vector through
    /// a `ByteBuffer`. With no deletions and no index sort -- the common case
    /// -- that is one copy per source segment.
    ///
    /// `VectorUtil.checkFinite` is deliberately not re-run: a vector reaching
    /// here was checked when the source segment was written, and Java does not
    /// re-check on merge either.
    /// On any error the writer is left exactly as it was before the call, so a
    /// caller that recovers (a test, or a merge that skips a field) cannot
    /// carry half of a field's bytes forward into the next one.
    pub fn merge_one_flat_vector_field(&mut self, field: &MergedFlatVectorField<'_>) -> Result<()> {
        let rollback_to = self.data.len();
        match self.merge_one_flat_vector_field_inner(field) {
            Ok(()) => Ok(()),
            Err(e) => {
                self.data.truncate(rollback_to);
                Err(e)
            }
        }
    }

    fn merge_one_flat_vector_field_inner(
        &mut self,
        field: &MergedFlatVectorField<'_>,
    ) -> Result<()> {
        if field.dimension <= 0 {
            return Err(Error::DimensionMismatch(
                field.field_number,
                field.dimension,
                0,
            ));
        }
        let Some(vector_bytes) = (field.dimension as usize).checked_mul(byte_size(field.encoding))
        else {
            return corrupt(format!(
                "field {}: vector dimension {} too large for this platform",
                field.field_number, field.dimension
            ));
        };
        let vector_data_offset = self.align_output(field.encoding);

        // One entry per surviving vector: the merged doc id it lands on, and
        // where to copy it from. The merged ordinal space is this list's
        // order, so it is built before a single byte is copied.
        let mut plan: Vec<MergedVectorSlot> = Vec::new();
        for (source_index, source) in field.sources.iter().enumerate() {
            if source.values.encoding() != field.encoding {
                return Err(Error::EncodingMismatch(
                    field.field_number,
                    source.values.encoding(),
                    field.encoding,
                ));
            }
            if source.values.dimension() != field.dimension as usize {
                return Err(Error::DimensionMismatch(
                    field.field_number,
                    field.dimension,
                    source.values.dimension() as i32,
                ));
            }
            let size = source.values.size();
            // A `MergeState.DocMap` never reorders documents *within* a
            // source: a merge drops documents and interleaves sources, and
            // even an index-sorted merge keeps each source's own order (every
            // source is already sorted by the same key). So a source whose
            // surviving documents do not map to strictly increasing merged ids
            // is a caller defect, and one that would otherwise be laundered
            // into a well-formed segment by the interleaving sort below.
            let mut last_new_doc = -1i32;
            for ord in 0..size {
                let old_doc = source.values.ord_to_doc(ord)?;
                let new_doc = match source.doc_map.get(old_doc as usize) {
                    Some(&d) => d,
                    None => {
                        return corrupt(format!(
                            "field {}: doc map has no entry for source doc {old_doc}",
                            field.field_number
                        ))
                    }
                };
                if new_doc < 0 {
                    continue;
                }
                if new_doc <= last_new_doc {
                    return corrupt(format!(
                        "field {}: source {source_index}'s doc map is not increasing ({new_doc} \
                         after {last_new_doc})",
                        field.field_number
                    ));
                }
                last_new_doc = new_doc;
                plan.push(MergedVectorSlot {
                    new_doc,
                    source_index: source_index as u32,
                    ord,
                });
            }
        }
        // `DocIDMerger.of(subs, mergeState.needsIndexSort)`: with no index
        // sort the merged doc ids already ascend across sources (each source
        // owns a contiguous, increasing range), and this scan confirms it
        // without moving anything. With one, the merge interleaves sources and
        // the ordinals have to be re-ordered -- `.vemf`'s doc list is an
        // `IndexedDISI`, so a merged ordinal space in source order would be
        // rejected outright by `validate_docs` below rather than silently
        // mis-associated.
        if !plan.windows(2).all(|w| w[0].new_doc < w[1].new_doc) {
            plan.sort_unstable_by_key(|slot| slot.new_doc);
        }

        // Copy in merged-ordinal order, coalescing every run of consecutive
        // ordinals from one source into a single `memcpy` -- with no index
        // sort and no deletions that is one copy per source, exactly as
        // before this became order-driven.
        let mut docs: Vec<i32> = Vec::with_capacity(plan.len());
        let mut i = 0usize;
        while i < plan.len() {
            let slot = plan[i];
            // ARITH: `i < plan.len()`, so `i + 1 <= plan.len()` and every
            // `end += 1` below is guarded by `end < plan.len()`; `end - 1`
            // cannot underflow because `end` starts at `i + 1 >= 1`. `ord` is
            // a file-derived ordinal, so the run test compares a
            // `wrapping_sub` difference rather than adding one to it.
            #[allow(clippy::arithmetic_side_effects)]
            let mut end = i + 1;
            // ARITH: see above -- `end += 1` runs only while
            // `end < plan.len()`, and `end - 1 >= i >= 0`.
            #[allow(clippy::arithmetic_side_effects)]
            while end < plan.len()
                && plan[end].source_index == slot.source_index
                && plan[end].ord.wrapping_sub(plan[end - 1].ord) == 1
            {
                end += 1;
            }
            let source = &field.sources[slot.source_index as usize];
            // ARITH: the loop above only extends the run while each ordinal is
            // exactly one past its predecessor, so the run occupies
            // `slot.ord ..= slot.ord + (end - i - 1)`, and every one of those
            // ordinals came from `0..size` -- the end of the run is therefore
            // at most `size <= i32::MAX`. `end - i >= 1` cannot underflow.
            #[allow(clippy::arithmetic_side_effects)]
            let run_end = slot.ord + (end - i) as i32;
            let bytes = source.values.raw_range(slot.ord, run_end)?;
            self.data.extend_from_slice(bytes);
            docs.extend(plan[i..end].iter().map(|s| s.new_doc));
            i = end;
        }

        // ARITH: as in `write_field`, `vector_data_offset` is a length
        // `self.data` has already passed.
        #[allow(clippy::arithmetic_side_effects)]
        let vector_data_length = self.data.len() as i64 - vector_data_offset;
        let Some(expected) = (docs.len() as i64).checked_mul(vector_bytes as i64) else {
            return corrupt(format!(
                "field {}: merged {} vectors of {vector_bytes} bytes overflows an i64",
                field.field_number,
                docs.len()
            ));
        };
        if vector_data_length != expected {
            return corrupt(format!(
                "field {}: merged {} vectors but wrote {vector_data_length} bytes, expected \
                 {expected}",
                field.field_number,
                docs.len()
            ));
        }
        validate_docs(field.field_number, &docs, self.max_doc)?;
        self.write_meta(
            field.field_number,
            field.encoding,
            field.similarity,
            field.dimension,
            vector_data_offset,
            vector_data_length,
            &docs,
        );
        Ok(())
    }

    /// `Lucene99FlatVectorsWriter.finish`: the `-1` end-of-fields marker and
    /// both footers. Returns `(vec_bytes, vemf_bytes)`.
    pub fn finish(mut self) -> (Vec<u8>, Vec<u8>) {
        self.meta.write_i32(-1);
        codec_util::write_footer(&mut self.meta);
        codec_util::write_footer(&mut self.data);
        (self.data, self.meta)
    }

    /// `Lucene99FlatVectorsWriter.alignOutput`: pad to the encoding's
    /// alignment and return the aligned offset. Part of the format, not an
    /// optimisation -- see finding 7 in `docs/sweep/m2/c5-vectors.md`.
    fn align_output(&mut self, encoding: VectorEncoding) -> i64 {
        let align = alignment(encoding);
        let padded = self.data.len().next_multiple_of(align);
        self.data.resize(padded, 0);
        self.data.len() as i64
    }

    /// `Lucene99FlatVectorsWriter.writeMeta`.
    #[allow(clippy::too_many_arguments)]
    fn write_meta(
        &mut self,
        field_number: i32,
        encoding: VectorEncoding,
        similarity: VectorSimilarityFunction,
        dimension: i32,
        vector_data_offset: i64,
        vector_data_length: i64,
        docs: &[i32],
    ) {
        self.meta.write_i32(field_number);
        self.meta.write_i32(encoding_ordinal(encoding));
        self.meta.write_i32(similarity_ordinal(similarity));
        self.meta.write_vlong(vector_data_offset);
        self.meta.write_vlong(vector_data_length);
        self.meta.write_vint(dimension);
        self.meta.write_i32(docs.len() as i32);
        write_stored_meta(&mut self.meta, &mut self.data, docs, self.max_doc);
    }
}

/// One surviving vector, as [`FlatVectorsWriter::merge_one_flat_vector_field`]
/// plans the merged ordinal space: which merged document it belongs to, and
/// which source ordinal holds its bytes. Sorting these by `new_doc` is what
/// makes the merged `.vec` ordinals ascend by document, which `.vemf`'s
/// `IndexedDISI` doc list requires.
#[derive(Debug, Clone, Copy)]
struct MergedVectorSlot {
    new_doc: i32,
    source_index: u32,
    ord: i32,
}

/// One source segment's vectors for the field being merged.
#[derive(Debug, Clone)]
pub enum MergeSourceValues<'a> {
    Float32(FloatVectorValues<'a>),
    Byte(ByteVectorValues<'a>),
}

impl MergeSourceValues<'_> {
    pub fn encoding(&self) -> VectorEncoding {
        match self {
            MergeSourceValues::Float32(_) => VectorEncoding::Float32,
            MergeSourceValues::Byte(_) => VectorEncoding::Byte,
        }
    }

    pub fn size(&self) -> i32 {
        match self {
            MergeSourceValues::Float32(v) => v.size(),
            MergeSourceValues::Byte(v) => v.size(),
        }
    }

    pub fn dimension(&self) -> usize {
        match self {
            MergeSourceValues::Float32(v) => v.dimension(),
            MergeSourceValues::Byte(v) => v.dimension(),
        }
    }

    pub fn ord_to_doc(&self, ord: i32) -> Result<i32> {
        match self {
            MergeSourceValues::Float32(v) => v.ord_to_doc(ord),
            MergeSourceValues::Byte(v) => v.ord_to_doc(ord),
        }
    }

    fn raw_range(&self, from_ord: i32, to_ord: i32) -> Result<&[u8]> {
        match self {
            MergeSourceValues::Float32(v) => v.raw_range(from_ord, to_ord),
            MergeSourceValues::Byte(v) => v.raw_range(from_ord, to_ord),
        }
    }
}

/// One segment being merged, as far as one vector field is concerned.
#[derive(Debug, Clone)]
pub struct FlatVectorMergeSource<'a> {
    pub values: MergeSourceValues<'a>,
    /// `MergeState.docMaps[i]`: the merged doc id of each of this segment's
    /// doc ids, or `-1` for a document the merge drops (deleted, or filtered
    /// out). Indexed by the source segment's own doc id, so it must have one
    /// entry per source document (`maxDoc`), not per vector.
    ///
    /// A map **shorter** than that is rejected as corruption rather than read
    /// as "the rest is deleted": the two are indistinguishable from here, and
    /// a truncated map silently drops every vector past its end. Callers that
    /// want trailing deletions spell them as trailing `-1`s.
    pub doc_map: &'a [i32],
}

/// One field's merge inputs, as handed to
/// [`FlatVectorsWriter::merge_one_flat_vector_field`]. The encoding,
/// similarity and dimension come from the merged `FieldInfo`, not from the
/// sources -- that is the value `Lucene99FlatVectorsReader.FieldEntry` will
/// cross-check the written `.vemf` against.
#[derive(Debug, Clone)]
pub struct MergedFlatVectorField<'a> {
    pub field_number: i32,
    pub encoding: VectorEncoding,
    pub similarity: VectorSimilarityFunction,
    pub dimension: i32,
    /// In merge order. Merged ordinals are assigned across these by ascending
    /// merged doc id (which, with no index sort, is exactly this order).
    pub sources: &'a [FlatVectorMergeSource<'a>],
}

/// Port of `Lucene99FlatVectorsWriter.{addField,flush,writeField,writeMeta,
/// finish}`. Returns `(vec_bytes, vemf_bytes)`.
///
/// A convenience over [`FlatVectorsWriter`] for the flush path, which always
/// has every field in hand at once.
pub fn write_flat_vectors(
    fields: &[FlatVectorsField],
    max_doc: i32,
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> Result<(Vec<u8>, Vec<u8>)> {
    let mut writer = FlatVectorsWriter::new(max_doc, segment_id, segment_suffix);
    for field in fields {
        writer.write_field(field)?;
    }
    Ok(writer.finish())
}

/// The doc-list rules `DefaultFieldWriter.addValue` and
/// `IndexedDISI.writeBitSet` enforce between them: strictly ascending,
/// non-negative, below `maxDoc`. Shared by the flush and merge write paths --
/// a merge whose doc maps are wrong produces exactly this shape of bad list,
/// and a DISI bitset built from it counts a document twice.
fn validate_docs(field_number: i32, docs: &[i32], max_doc: i32) -> Result<()> {
    if !docs.windows(2).all(|w| w[0] < w[1]) {
        return corrupt(format!(
            "field {field_number}: doc ids must be strictly ascending"
        ));
    }
    if let Some(&last) = docs.last() {
        if last >= max_doc {
            return corrupt(format!(
                "field {field_number}: doc id {last} >= maxDoc {max_doc}"
            ));
        }
    }
    if docs.first().is_some_and(|&d| d < 0) {
        return corrupt(format!("field {field_number}: negative doc id"));
    }
    Ok(())
}

fn validate_field(field: &FlatVectorsField, max_doc: i32) -> Result<()> {
    if field.dimension <= 0 {
        return Err(Error::DimensionMismatch(
            field.field_number,
            field.dimension,
            0,
        ));
    }
    let Some(expected) = field.docs.len().checked_mul(field.dimension as usize) else {
        return Err(Error::DimensionMismatch(
            field.field_number,
            field.dimension,
            0,
        ));
    };
    if field.values.component_count() != expected {
        return Err(Error::DimensionMismatch(
            field.field_number,
            expected as i32,
            field.values.component_count() as i32,
        ));
    }
    if field.docs.is_empty() && field.values.component_count() != 0 {
        return Err(Error::EmptyVector(field.field_number));
    }
    validate_docs(field.field_number, &field.docs, max_doc)?;
    // `VectorUtil.checkFinite`, which `KnnFloatVectorField`'s constructor runs
    // on every indexed vector -- so a Lucene-written `.vec` can never contain a
    // NaN or an infinity, and this port's must not either. It is not cosmetic:
    // a NaN component poisons every comparison downstream, and `f32::max`
    // (unlike Java's `Math.max`) *drops* a NaN rather than propagating it, so a
    // NaN score would be silently excluded from the bulk-score maximum that
    // gates graph expansion instead of loudly breaking.
    if let FieldVectorData::Float32(values) = &field.values {
        if let Some((i, v)) = values.iter().enumerate().find(|(_, v)| !v.is_finite()) {
            return Err(Error::NonFiniteValue(field.field_number, i, *v));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Scorers
// ---------------------------------------------------------------------------

/// Port of `DefaultFlatVectorScorer`'s float scorer -- both roles Java splits
/// across `RandomVectorScorer` (query is an external vector) and
/// `UpdateableRandomVectorScorer` (query is a stored ordinal, re-targetable).
/// One type covers both because the only difference is where `query` came
/// from.
///
/// Decodes each candidate into a reusable scratch buffer, so a whole graph
/// search allocates nothing.
#[derive(Debug, Clone)]
pub struct FloatVectorScorer<'a> {
    values: FloatVectorValues<'a>,
    similarity: VectorSimilarityFunction,
    query: Vec<f32>,
    scratch: Vec<f32>,
}

impl crate::hnsw::VectorScorer for FloatVectorScorer<'_> {
    fn score(&mut self, node: i32) -> Result<f32> {
        self.values.vector_into(node, &mut self.scratch)?;
        Ok(self.similarity.score(&self.query, &self.scratch))
    }

    fn max_ord(&self) -> i32 {
        self.values.size()
    }
}

impl crate::hnsw::UpdateableVectorScorer for FloatVectorScorer<'_> {
    fn set_scoring_ordinal(&mut self, ord: i32) -> Result<()> {
        self.values.vector_into(ord, &mut self.query)
    }
}

/// Port of `DefaultFlatVectorScorer`'s byte scorer; see
/// [`FloatVectorScorer`]. No scratch buffer: a byte vector needs no decoding,
/// so the candidate is scored straight out of the mapped `.vec` bytes.
#[derive(Debug, Clone)]
pub struct ByteVectorScorer<'a> {
    values: ByteVectorValues<'a>,
    similarity: VectorSimilarityFunction,
    query: Vec<u8>,
}

impl crate::hnsw::VectorScorer for ByteVectorScorer<'_> {
    fn score(&mut self, node: i32) -> Result<f32> {
        let candidate = self.values.vector(node)?;
        Ok(self.similarity.score_bytes(&self.query, candidate))
    }

    fn max_ord(&self) -> i32 {
        self.values.size()
    }
}

impl crate::hnsw::UpdateableVectorScorer for ByteVectorScorer<'_> {
    fn set_scoring_ordinal(&mut self, ord: i32) -> Result<()> {
        let v = self.values.vector(ord)?;
        self.query.clear();
        self.query.extend_from_slice(v);
        Ok(())
    }
}

impl<'a> FloatVectorValues<'a> {
    /// `FlatVectorsScorer.getRandomVectorScorer(similarity, values, target)`.
    pub fn scorer(&self, query: &[f32]) -> Result<FloatVectorScorer<'a>> {
        if query.len() != self.values.dimension {
            return Err(Error::QueryDimensionMismatch(
                self.values.dimension as i32,
                query.len() as i32,
            ));
        }
        Ok(FloatVectorScorer {
            values: self.clone(),
            similarity: self.values.similarity,
            query: query.to_vec(),
            scratch: vec![0.0; self.values.dimension],
        })
    }

    /// `FlatVectorsScorer.getRandomVectorScorerSupplier(...).scorer()`: a
    /// scorer whose query is set later with `set_scoring_ordinal`.
    pub fn ord_scorer(&self) -> FloatVectorScorer<'a> {
        FloatVectorScorer {
            values: self.clone(),
            similarity: self.values.similarity,
            query: vec![0.0; self.values.dimension],
            scratch: vec![0.0; self.values.dimension],
        }
    }
}

impl<'a> ByteVectorValues<'a> {
    /// `FlatVectorsScorer.getRandomVectorScorer(similarity, values, target)`.
    pub fn scorer(&self, query: &[u8]) -> Result<ByteVectorScorer<'a>> {
        if query.len() != self.values.dimension {
            return Err(Error::QueryDimensionMismatch(
                self.values.dimension as i32,
                query.len() as i32,
            ));
        }
        Ok(ByteVectorScorer {
            values: self.clone(),
            similarity: self.values.similarity,
            query: query.to_vec(),
        })
    }

    /// See [`FloatVectorValues::ord_scorer`].
    pub fn ord_scorer(&self) -> ByteVectorScorer<'a> {
        ByteVectorScorer {
            values: self.clone(),
            similarity: self.values.similarity,
            query: vec![0; self.values.dimension],
        }
    }
}

#[cfg(test)]
mod tests {
    // A test's `i + 1` is not a length read off disk; see
    // `docs/arithmetic-gate.md`'s "Test code" section.
    #![allow(clippy::arithmetic_side_effects)]

    use super::*;

    /// Rebuilds a file's footer after a test has patched a byte in its body.
    /// Every corruption test needs this: `open` verifies the whole-file CRC
    /// *before* it decodes any field, so a raw byte flip is rejected as a bad
    /// checksum and never reaches the check under test.
    fn repack(buf: &[u8], patch: impl FnOnce(&mut Vec<u8>)) -> Vec<u8> {
        let mut body = buf[..buf.len() - codec_util::FOOTER_LENGTH].to_vec();
        patch(&mut body);
        codec_util::write_footer(&mut body);
        body
    }

    const ID: [u8; ID_LENGTH] = *b"vectorstestid001";

    // ---------------- re-signed byte-flip sweep ----------------

    /// A three-field segment that spans every structure the `.vemf`/`.vec`
    /// pair has: a **dense** float field (no ord->doc structures at all), a
    /// **sparse** float field (an `IndexedDISI` bitset *and* a
    /// `DirectMonotonic` ordinal->doc sequence, both living in `.vec` after
    /// the vectors), and a **byte** field (a different `byteSize`, and
    /// therefore a different `alignOutput` padding). 600 documents, so the
    /// DISI's dense-block path and its rank table are both populated -- a
    /// single-document fixture leaves all three structures degenerate and
    /// measures the fixture rather than the decoder.
    fn sweep_fixture() -> (Vec<u8>, Vec<u8>) {
        let max_doc = 600;
        let dense: Vec<i32> = (0..max_doc).collect();
        let sparse: Vec<i32> = (0..max_doc).filter(|d| d % 3 == 0).collect();
        let byte_docs: Vec<i32> = (0..max_doc).filter(|d| d % 7 == 0).collect();
        let byte_values: Vec<u8> = (0..byte_docs.len() * 2).map(|i| (i % 251) as u8).collect();
        let fields = vec![
            float_field(0, 4, dense),
            float_field(1, 3, sparse),
            FlatVectorsField {
                field_number: 2,
                similarity: VectorSimilarityFunction::DotProduct,
                dimension: 2,
                docs: byte_docs,
                values: FieldVectorData::Byte(byte_values),
            },
        ];
        write(&fields, max_doc)
    }

    /// Everything a reader does with a `.vemf`/`.vec` pair, so that a flip
    /// anywhere in either file has to surface as a typed error (or a clean,
    /// self-consistent decode) rather than a panic or an allocation abort.
    fn walk_everything(meta: &[u8], data: &[u8]) -> Result<()> {
        let reader = open(meta, data)?;
        for entry in reader.fields().to_vec() {
            let number = entry.field_number;
            match entry.encoding {
                VectorEncoding::Float32 => {
                    let values = reader.float_vector_values(number)?;
                    let mut scratch = vec![0.0f32; values.dimension()];
                    for ord in 0..values.size() {
                        values.vector_into(ord, &mut scratch)?;
                        values.ord_to_doc(ord)?;
                    }
                    if values.size() > 1 {
                        values.raw_range(0, values.size())?;
                    }
                    let query = vec![0.25f32; values.dimension()];
                    values.exhaustive_search(&query, 5)?;
                    let mut cursor = values.doc_to_ord()?;
                    for doc in 0..600 {
                        cursor.ordinal(doc)?;
                    }
                }
                VectorEncoding::Byte => {
                    let values = reader.byte_vector_values(number)?;
                    for ord in 0..values.size() {
                        values.vector(ord)?;
                        values.ord_to_doc(ord)?;
                    }
                    let mut cursor = values.doc_to_ord()?;
                    for doc in 0..600 {
                        cursor.ordinal(doc)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Flip bit 0 and bit 7 of every `.vemf` and `.vec` body byte, re-sign the
    /// footer so only a semantic invariant can reject the file, and require a
    /// typed error or a clean decode from the full walk above. Never a panic,
    /// never an abort.
    ///
    /// A low rejection rate is expected and is not a gap: almost every `.vec`
    /// byte is a vector *component*, and flipping one yields a different but
    /// perfectly well-formed float. The bar is that nothing panics, nothing
    /// aborts, and nothing reserves memory proportional to a number it just
    /// read.
    #[test]
    fn every_resigned_single_byte_vemf_and_vec_corruption_is_an_error_or_a_clean_decode() {
        let (data, meta) = sweep_fixture();
        walk_everything(&meta, &data).expect("the fixture itself must decode");
        let mut flipped = 0usize;
        let mut rejected = 0usize;
        let mut meta_flipped = 0usize;
        let mut meta_rejected = 0usize;
        for which in 0..2 {
            let original: &[u8] = if which == 0 { &meta } else { &data };
            let body_len = original.len() - codec_util::FOOTER_LENGTH;
            for at in 0..body_len {
                for bit in [0u8, 7] {
                    let patched = repack(original, |b| b[at] ^= 1 << bit);
                    let (m, d) = if which == 0 {
                        (patched.as_slice(), data.as_slice())
                    } else {
                        (meta.as_slice(), patched.as_slice())
                    };
                    flipped += 1;
                    let bad = walk_everything(m, d).is_err();
                    if bad {
                        rejected += 1;
                    }
                    if which == 0 {
                        meta_flipped += 1;
                        meta_rejected += usize::from(bad);
                    }
                }
            }
        }
        assert_eq!(
            flipped,
            2 * (meta.len() + data.len() - 2 * codec_util::FOOTER_LENGTH)
        );
        // The metadata is all lengths, offsets and counts, so almost every
        // flip there has to be caught by something.
        assert!(
            meta_rejected > meta_flipped / 2,
            "only {meta_rejected} of {meta_flipped} .vemf flips rejected"
        );
        eprintln!(
            ".vemf+.vec byte-flip sweep: {rejected}/{flipped} rejected \
             (.vemf alone {meta_rejected}/{meta_flipped})"
        );
    }

    fn float_field(number: i32, dim: i32, docs: Vec<i32>) -> FlatVectorsField {
        let values = (0..docs.len() as i32 * dim)
            .map(|i| i as f32 * 0.5 - 1.0)
            .collect();
        FlatVectorsField {
            field_number: number,
            similarity: VectorSimilarityFunction::Euclidean,
            dimension: dim,
            docs,
            values: FieldVectorData::Float32(values),
        }
    }

    fn write(fields: &[FlatVectorsField], max_doc: i32) -> (Vec<u8>, Vec<u8>) {
        write_flat_vectors(fields, max_doc, &ID, "").unwrap()
    }

    /// Position of a field's `dimension` vint in the `.vemf`, for the
    /// corruption tests. The layout up to it is fixed-width apart from the two
    /// vlongs, which are re-derived here rather than hard-coded.
    fn open(meta: &[u8], data: &[u8]) -> Result<FlatVectorsReader<'static>> {
        // `FlatVectorsReader` borrows `data`; the tests only need the error, so
        // leak the buffer rather than thread lifetimes through every case.
        let data: &'static [u8] = Box::leak(data.to_vec().into_boxed_slice());
        FlatVectorsReader::open(meta, data, &ID, "")
    }

    // ---------------- kernels ----------------

    fn naive_square_distance(a: &[f32], b: &[f32]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| ((*x - *y) as f64).powi(2))
            .sum()
    }

    #[test]
    fn float_kernels_agree_with_a_naive_reference_at_every_length() {
        // Lengths either side of the eight-lane split, so the `chunks_exact`
        // remainder is exercised too.
        for len in [1usize, 5, 8, 9, 16, 17, 31, 128] {
            let a: Vec<f32> = (0..len).map(|i| (i as f32 * 0.37 + 0.3).sin()).collect();
            let b: Vec<f32> = (0..len).map(|i| (i as f32 * 0.91 + 0.7).cos()).collect();
            let sq = square_distance(&a, &b) as f64;
            assert!(
                (sq - naive_square_distance(&a, &b)).abs() < 1e-4,
                "len {len}: {sq}"
            );
            let dot = dot_product(&a, &b) as f64;
            let naive_dot: f64 = a.iter().zip(&b).map(|(x, y)| *x as f64 * *y as f64).sum();
            assert!((dot - naive_dot).abs() < 1e-4, "len {len}: {dot}");
            let n1: f64 = a.iter().map(|x| *x as f64 * *x as f64).sum();
            let n2: f64 = b.iter().map(|x| *x as f64 * *x as f64).sum();
            let cos = cosine(&a, &b) as f64;
            assert!(
                (cos - naive_dot / (n1 * n2).sqrt()).abs() < 1e-4,
                "len {len}"
            );
        }
    }

    #[test]
    fn cosine_returns_zero_for_a_zero_vector() {
        // Java returns NaN here (and asserts `Float.isFinite`); this port
        // refuses to poison the heap ordering with a NaN.
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 2.0]), 0.0);
        assert_eq!(cosine(&[1.0, 2.0], &[0.0, 0.0]), 0.0);
        assert_eq!(cosine_bytes(&[0, 0], &[1, 2]), 0.0);
        assert_eq!(cosine_bytes(&[1, 2], &[0, 0]), 0.0);
    }

    #[test]
    fn byte_kernels_sign_extend_like_javas_byte() {
        // 0xFF is -1, not 255: Java's `byte` is signed and the on-disk bytes
        // are those signed values.
        assert_eq!(dot_product_bytes(&[0xFF], &[1]), -1);
        assert_eq!(square_distance_bytes(&[0xFF], &[1]), 4);
        assert_eq!(dot_product_bytes(&[0x80], &[1]), -128);
        assert!((cosine_bytes(&[0xFF], &[1]) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn scale_max_inner_product_score_is_piecewise() {
        assert_eq!(scale_max_inner_product_score(0.0), 1.0);
        assert_eq!(scale_max_inner_product_score(3.0), 4.0);
        assert_eq!(scale_max_inner_product_score(-3.0), 0.25);
    }

    #[test]
    fn normalize_helpers_match_java() {
        assert_eq!(normalize_to_unit_interval(1.0), 1.0);
        assert_eq!(normalize_to_unit_interval(-1.0), 0.0);
        // The `max(_, 0)` floor: an unnormalized dot below -1 must not score
        // negative.
        assert_eq!(normalize_to_unit_interval(-5.0), 0.0);
        assert_eq!(normalize_distance_to_unit_interval(0.0), 1.0);
        assert_eq!(normalize_distance_to_unit_interval(3.0), 0.25);
    }

    #[test]
    fn float_similarity_transforms_match_java() {
        let a = [1.0f32, 0.0, 0.0];
        let b = [0.0f32, 1.0, 0.0];
        assert_eq!(VectorSimilarityFunction::Euclidean.score(&a, &b), 1.0 / 3.0);
        assert_eq!(VectorSimilarityFunction::DotProduct.score(&a, &b), 0.5);
        assert_eq!(VectorSimilarityFunction::Cosine.score(&a, &b), 0.5);
        assert_eq!(
            VectorSimilarityFunction::MaximumInnerProduct.score(&a, &b),
            1.0
        );
        assert_eq!(VectorSimilarityFunction::DotProduct.score(&a, &a), 1.0);
        assert_eq!(
            VectorSimilarityFunction::MaximumInnerProduct.score(&a, &[-2.0, 0.0, 0.0]),
            1.0 / 3.0
        );
    }

    /// Byte vectors do **not** reuse the float transforms: DOT_PRODUCT is
    /// `dotProductScore` (0.5 + dot / (dim * 2^15)), and COSINE has no
    /// `max(_, 0)` clamp.
    #[test]
    fn byte_similarity_transforms_differ_from_the_float_ones() {
        let a = [1u8, 0, 0];
        let b = [0u8, 1, 0];
        assert_eq!(
            VectorSimilarityFunction::Euclidean.score_bytes(&a, &b),
            1.0 / 3.0
        );
        assert_eq!(
            VectorSimilarityFunction::DotProduct.score_bytes(&a, &b),
            0.5
        );
        // 3 dims, dot = 1 -> 0.5 + 1 / (3 * 32768)
        assert_eq!(
            VectorSimilarityFunction::DotProduct.score_bytes(&a, &a),
            0.5 + 1.0 / (3.0 * 32768.0)
        );
        assert_eq!(VectorSimilarityFunction::Cosine.score_bytes(&a, &b), 0.5);
        assert_eq!(
            VectorSimilarityFunction::MaximumInnerProduct.score_bytes(&a, &a),
            2.0
        );
    }

    // ---------------- writer / reader ----------------

    #[test]
    fn empty_field_writes_the_minus_two_marker() {
        let field = FlatVectorsField {
            field_number: 4,
            similarity: VectorSimilarityFunction::Cosine,
            dimension: 3,
            docs: Vec::new(),
            values: FieldVectorData::Float32(Vec::new()),
        };
        let (data, meta) = write(&[field], 10);
        let reader = FlatVectorsReader::open(&meta, &data, &ID, "").unwrap();
        let entry = reader.field(4).unwrap();
        assert!(entry.ord_to_doc.is_empty());
        assert!(!entry.ord_to_doc.is_dense());
        let values = reader.float_vector_values(4).unwrap();
        assert_eq!(values.size(), 0);
        assert!(values.is_empty());
        assert!(values
            .exhaustive_search(&[0.0, 0.0, 0.0], 3)
            .unwrap()
            .is_empty());
        let mut cursor = values.doc_to_ord().unwrap();
        assert_eq!(cursor.ordinal(0).unwrap(), None);
        cursor.reset();
        assert_eq!(cursor.ordinal(9).unwrap(), None);
    }

    #[test]
    fn dense_cursor_answers_out_of_range_docs_with_none() {
        let (data, meta) = write(&[float_field(0, 2, (0..5).collect())], 5);
        let reader = FlatVectorsReader::open(&meta, &data, &ID, "").unwrap();
        let values = reader.float_vector_values(0).unwrap();
        let mut cursor = values.doc_to_ord().unwrap();
        assert_eq!(cursor.ordinal(4).unwrap(), Some(4));
        assert_eq!(cursor.ordinal(5).unwrap(), None);
        cursor.reset();
        assert_eq!(cursor.ordinal(0).unwrap(), Some(0));
    }

    #[test]
    fn exhaustive_search_ranks_by_score_then_doc_id() {
        // Two vectors equidistant from the query: the smaller doc id wins.
        let field = FlatVectorsField {
            field_number: 0,
            similarity: VectorSimilarityFunction::Euclidean,
            dimension: 1,
            docs: vec![0, 1, 2],
            values: FieldVectorData::Float32(vec![-1.0, 1.0, 5.0]),
        };
        let (data, meta) = write(&[field], 3);
        let reader = FlatVectorsReader::open(&meta, &data, &ID, "").unwrap();
        let values = reader.float_vector_values(0).unwrap();
        let hits = values.exhaustive_search(&[0.0], 3).unwrap();
        assert_eq!(hits[0].0, 0);
        assert_eq!(hits[1].0, 1);
        assert_eq!(hits[2].0, 2);
        assert_eq!(values.exhaustive_search(&[0.0], 0).unwrap(), Vec::new());
        assert_eq!(values.exhaustive_search(&[0.0], 2).unwrap().len(), 2);
    }

    #[test]
    fn byte_exhaustive_search_and_scorer() {
        let field = FlatVectorsField {
            field_number: 1,
            similarity: VectorSimilarityFunction::Cosine,
            dimension: 2,
            docs: vec![0, 2],
            values: FieldVectorData::Byte(vec![1, 0, 0, 1]),
        };
        let (data, meta) = write(&[field], 4);
        let reader = FlatVectorsReader::open(&meta, &data, &ID, "").unwrap();
        let values = reader.byte_vector_values(1).unwrap();
        assert_eq!(values.dimension(), 2);
        assert_eq!(values.similarity(), VectorSimilarityFunction::Cosine);
        assert!(!values.is_empty());
        let hits = values.exhaustive_search(&[1, 0], 2).unwrap();
        assert_eq!(hits[0].0, 0);
        assert_eq!(hits[1].0, 2);
        assert_eq!(values.exhaustive_search(&[1, 0], 0).unwrap(), Vec::new());
        assert!(matches!(
            values.exhaustive_search(&[1], 1),
            Err(Error::QueryDimensionMismatch(2, 1))
        ));
        assert!(matches!(
            values.scorer(&[1]),
            Err(Error::QueryDimensionMismatch(2, 1))
        ));
        assert!(values.scorer(&[1, 2]).is_ok());
    }

    #[test]
    fn search_rejects_a_query_of_the_wrong_dimension() {
        let (data, meta) = write(&[float_field(0, 4, (0..3).collect())], 3);
        let reader = FlatVectorsReader::open(&meta, &data, &ID, "").unwrap();
        let values = reader.float_vector_values(0).unwrap();
        assert!(matches!(
            values.exhaustive_search(&[1.0, 2.0], 1),
            Err(Error::QueryDimensionMismatch(4, 2))
        ));
        assert!(matches!(
            values.scorer(&[1.0, 2.0]),
            Err(Error::QueryDimensionMismatch(4, 2))
        ));
    }

    #[test]
    fn unknown_field_and_encoding_mismatch_are_rejected() {
        let byte_field = FlatVectorsField {
            field_number: 2,
            similarity: VectorSimilarityFunction::DotProduct,
            dimension: 2,
            docs: vec![0, 1],
            values: FieldVectorData::Byte(vec![1, 2, 3, 4]),
        };
        let (data, meta) = write(&[float_field(0, 2, vec![0, 1]), byte_field], 2);
        let reader = FlatVectorsReader::open(&meta, &data, &ID, "").unwrap();
        assert!(matches!(
            reader.float_vector_values(9),
            Err(Error::UnknownField(9))
        ));
        assert!(matches!(
            reader.float_vector_values(2),
            Err(Error::EncodingMismatch(
                2,
                VectorEncoding::Byte,
                VectorEncoding::Float32
            ))
        ));
        assert!(matches!(
            reader.byte_vector_values(0),
            Err(Error::EncodingMismatch(
                0,
                VectorEncoding::Float32,
                VectorEncoding::Byte
            ))
        ));
        assert!(reader.field(9).is_none());
        assert_eq!(reader.fields().len(), 2);
    }

    #[test]
    fn vector_into_rejects_a_wrong_sized_buffer() {
        let (data, meta) = write(&[float_field(0, 4, vec![0, 1])], 2);
        let reader = FlatVectorsReader::open(&meta, &data, &ID, "").unwrap();
        let values = reader.float_vector_values(0).unwrap();
        let mut short = [0.0f32; 3];
        assert!(matches!(
            values.vector_into(0, &mut short),
            Err(Error::QueryDimensionMismatch(4, 3))
        ));
        let mut long = [0.0f32; 5];
        assert!(matches!(
            values.vector_into(0, &mut long),
            Err(Error::QueryDimensionMismatch(4, 5))
        ));
    }

    #[test]
    fn out_of_range_ordinals_are_rejected() {
        let (data, meta) = write(&[float_field(0, 2, vec![0, 1])], 2);
        let reader = FlatVectorsReader::open(&meta, &data, &ID, "").unwrap();
        let values = reader.float_vector_values(0).unwrap();
        assert!(matches!(values.vector(2), Err(Error::OrdOutOfRange(2, 2))));
        assert!(matches!(
            values.vector(-1),
            Err(Error::OrdOutOfRange(-1, 2))
        ));
        assert!(matches!(
            values.ord_to_doc(2),
            Err(Error::OrdOutOfRange(2, 2))
        ));
    }

    #[test]
    fn sparse_ord_to_doc_rejects_an_out_of_range_ordinal() {
        let (data, meta) = write(&[float_field(0, 2, vec![0, 3])], 8);
        let reader = FlatVectorsReader::open(&meta, &data, &ID, "").unwrap();
        let values = reader.float_vector_values(0).unwrap();
        assert_eq!(values.ord_to_doc(1).unwrap(), 3);
        assert!(matches!(
            values.ord_to_doc(5),
            Err(Error::OrdOutOfRange(5, 2))
        ));
    }

    // ---------------- write-side validation ----------------

    #[test]
    fn writer_rejects_malformed_fields() {
        let mut bad = float_field(0, 0, vec![0]);
        bad.values = FieldVectorData::Float32(Vec::new());
        assert!(matches!(
            write_flat_vectors(&[bad], 1, &ID, ""),
            Err(Error::DimensionMismatch(0, 0, 0))
        ));

        let mut wrong_len = float_field(0, 2, vec![0, 1]);
        wrong_len.values = FieldVectorData::Float32(vec![1.0, 2.0, 3.0]);
        assert!(matches!(
            write_flat_vectors(&[wrong_len], 2, &ID, ""),
            Err(Error::DimensionMismatch(0, 4, 3))
        ));

        let descending = float_field(0, 1, vec![3, 1]);
        assert!(matches!(
            write_flat_vectors(&[descending], 5, &ID, ""),
            Err(Error::CorruptMeta(_))
        ));

        let past_max_doc = float_field(0, 1, vec![0, 9]);
        assert!(matches!(
            write_flat_vectors(&[past_max_doc], 5, &ID, ""),
            Err(Error::CorruptMeta(_))
        ));

        let negative = float_field(0, 1, vec![-1, 2]);
        assert!(matches!(
            write_flat_vectors(&[negative], 5, &ID, ""),
            Err(Error::CorruptMeta(_))
        ));

        let mut orphan = float_field(0, 2, Vec::new());
        orphan.values = FieldVectorData::Float32(vec![1.0, 2.0]);
        assert!(matches!(
            write_flat_vectors(&[orphan], 5, &ID, ""),
            Err(Error::DimensionMismatch(0, 0, 2))
        ));
    }

    /// `VectorUtil.checkFinite`, which `KnnFloatVectorField`'s constructor runs
    /// on every indexed vector -- so a Lucene-written `.vec` never contains one
    /// of these and this port's must not either.
    #[test]
    fn writer_rejects_non_finite_components() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let field = FlatVectorsField {
                field_number: 2,
                similarity: VectorSimilarityFunction::Euclidean,
                dimension: 2,
                docs: vec![0, 1],
                values: FieldVectorData::Float32(vec![1.0, 2.0, bad, 4.0]),
            };
            assert!(
                matches!(
                    write_flat_vectors(&[field], 2, &ID, ""),
                    Err(Error::NonFiniteValue(2, 2, _))
                ),
                "{bad} accepted"
            );
        }
        // A byte field has no non-finite values to reject.
        let bytes = FlatVectorsField {
            field_number: 2,
            similarity: VectorSimilarityFunction::Euclidean,
            dimension: 2,
            docs: vec![0, 1],
            values: FieldVectorData::Byte(vec![1, 2, 3, 4]),
        };
        assert!(write_flat_vectors(&[bytes], 2, &ID, "").is_ok());
    }

    #[test]
    fn field_vector_data_reports_its_encoding() {
        assert_eq!(
            FieldVectorData::Float32(Vec::new()).encoding(),
            VectorEncoding::Float32
        );
        assert_eq!(
            FieldVectorData::Byte(Vec::new()).encoding(),
            VectorEncoding::Byte
        );
    }

    // ---------------- corrupt metadata ----------------

    #[test]
    fn version_mismatch_between_meta_and_data_is_rejected() {
        let (data, meta) = write(&[float_field(0, 2, vec![0, 1])], 2);
        // The version is the 4 bytes right after CODEC_MAGIC + the codec name.
        let version_at = codec_util::header_length(DATA_CODEC) - 4;
        let bumped = repack(&data, |b| {
            b[version_at..version_at + 4].copy_from_slice(&1i32.to_be_bytes());
        });
        // A version outside [VERSION_START, VERSION_CURRENT] is caught by
        // `check_index_header` itself, which is the same rejection.
        assert!(open(&meta, &bumped).is_err());
    }

    /// Corrupts the first field's encoding/similarity ordinal, which sit at a
    /// fixed offset right after the `.vemf` header's field number.
    fn meta_field_start() -> usize {
        codec_util::index_header_length(META_CODEC, "") + 4
    }

    #[test]
    fn invalid_encoding_and_similarity_ordinals_are_rejected() {
        let (data, meta) = write(&[float_field(0, 2, vec![0, 1])], 2);
        let at = meta_field_start();
        let bad_encoding = repack(&meta, |b| {
            b[at..at + 4].copy_from_slice(&7i32.to_le_bytes());
        });
        assert!(matches!(
            open(&bad_encoding, &data),
            Err(Error::CorruptMeta(_))
        ));
        let bad_similarity = repack(&meta, |b| {
            b[at + 4..at + 8].copy_from_slice(&9i32.to_le_bytes());
        });
        assert!(matches!(
            open(&bad_similarity, &data),
            Err(Error::CorruptMeta(_))
        ));
    }

    #[test]
    fn a_negative_field_number_is_rejected() {
        let (data, meta) = write(&[float_field(0, 2, vec![0, 1])], 2);
        let at = codec_util::index_header_length(META_CODEC, "");
        let bad = repack(&meta, |b| {
            b[at..at + 4].copy_from_slice(&(-5i32).to_le_bytes());
        });
        assert!(matches!(open(&bad, &data), Err(Error::CorruptMeta(_))));
    }

    #[test]
    fn a_vector_data_length_that_disagrees_with_size_is_rejected() {
        // `dimension` is a vint at a known offset: header, fieldNumber(4),
        // encoding(4), similarity(4), then two vlongs. Both vlongs here are
        // small enough to be one or two bytes, so find the dimension by
        // decoding rather than guessing.
        let (data, meta) = write(&[float_field(0, 2, vec![0, 1])], 2);
        let mut input = SliceInput::new(&meta);
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
        input.read_vlong().unwrap(); // data offset
        input.read_vlong().unwrap(); // data length
        let dim_at = input.position();
        assert_eq!(meta[dim_at], 2); // a one-byte vint
        let bad = repack(&meta, |b| b[dim_at] = 3);
        assert!(matches!(open(&bad, &data), Err(Error::CorruptMeta(_))));
        // ... and a dimension of zero, which Java's `FieldEntry` would divide
        // by.
        let zero = repack(&meta, |b| b[dim_at] = 0);
        assert!(matches!(open(&zero, &data), Err(Error::CorruptMeta(_))));
        // ... and a negative count.
        let count_at = dim_at + 1;
        let negative_count = repack(&meta, |b| {
            b[count_at..count_at + 4].copy_from_slice(&(-1i32).to_le_bytes());
        });
        assert!(matches!(
            open(&negative_count, &data),
            Err(Error::CorruptMeta(_))
        ));
        // ... and a `docsWithFieldOffset` that is neither -1, -2 nor a real
        // offset.
        let offset_at = count_at + 4;
        let bad_marker = repack(&meta, |b| {
            b[offset_at..offset_at + 8].copy_from_slice(&(-3i64).to_le_bytes());
        });
        assert!(matches!(
            open(&bad_marker, &data),
            Err(Error::CorruptMeta(_))
        ));
    }

    #[test]
    fn a_vector_region_past_the_end_of_the_data_file_is_rejected() {
        let (data, meta) = write(&[float_field(0, 2, vec![0, 1])], 2);
        let mut input = SliceInput::new(&meta);
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
        // The offset is a vlong; the writer's is 64 (one byte can't hold it,
        // so it is two). Overwrite with a huge two-byte vlong.
        let bad = repack(&meta, |b| {
            b[offset_at] = 0xFF;
            b[offset_at + 1] = 0x7F;
        });
        assert!(matches!(open(&bad, &data), Err(Error::CorruptMeta(_))));
    }

    #[test]
    fn a_truncated_footer_is_rejected() {
        let (data, meta) = write(&[float_field(0, 2, vec![0, 1])], 2);
        assert!(open(&meta[..meta.len() - 1], &data).is_err());
    }
    // ---------------- merge ----------------

    /// Two source segments, written as this port writes any segment, reopened
    /// and merged. The merged files must be **byte for byte** what a single
    /// flush of the same documents in the same order produces -- which is the
    /// whole contract: a merge may not be observable in the output.
    #[test]
    fn a_merge_produces_exactly_what_one_flush_of_the_same_documents_would() {
        // Segment A: 5 documents, dense. Segment B: 4 documents, sparse
        // (docs 0 and 2 only).
        let a_docs: Vec<i32> = (0..5).collect();
        let b_docs: Vec<i32> = vec![0, 2];
        let dim = 3;
        let a_values: Vec<f32> = (0..5 * dim).map(|i| i as f32 * 0.25 - 2.0).collect();
        let b_values: Vec<f32> = (0..2 * dim).map(|i| i as f32 * -0.75 + 9.0).collect();

        let (a_vec, a_meta) = write(
            &[FlatVectorsField {
                field_number: 7,
                similarity: VectorSimilarityFunction::Cosine,
                dimension: dim,
                docs: a_docs.clone(),
                values: FieldVectorData::Float32(a_values.clone()),
            }],
            5,
        );
        let (b_vec, b_meta) = write(
            &[FlatVectorsField {
                field_number: 7,
                similarity: VectorSimilarityFunction::Cosine,
                dimension: dim,
                docs: b_docs.clone(),
                values: FieldVectorData::Float32(b_values.clone()),
            }],
            4,
        );

        let a = FlatVectorsReader::open(&a_meta, &a_vec, &ID, "").unwrap();
        let b = FlatVectorsReader::open(&b_meta, &b_vec, &ID, "").unwrap();
        // Nothing deleted: A's docs keep their ids, B's are appended after
        // them. This is the shape `merge.rs` produces for a plain merge.
        let a_map: Vec<i32> = (0..5).collect();
        let b_map: Vec<i32> = (0..4).map(|d| d + 5).collect();
        let sources = vec![
            FlatVectorMergeSource {
                values: MergeSourceValues::Float32(a.float_vector_values(7).unwrap()),
                doc_map: &a_map,
            },
            FlatVectorMergeSource {
                values: MergeSourceValues::Float32(b.float_vector_values(7).unwrap()),
                doc_map: &b_map,
            },
        ];
        let mut writer = FlatVectorsWriter::new(9, &ID, "");
        writer
            .merge_one_flat_vector_field(&MergedFlatVectorField {
                field_number: 7,
                encoding: VectorEncoding::Float32,
                similarity: VectorSimilarityFunction::Cosine,
                dimension: dim,
                sources: &sources,
            })
            .unwrap();
        let (merged_vec, merged_meta) = writer.finish();

        // The same documents, flushed in one go.
        let mut all_docs = a_docs.clone();
        all_docs.extend(b_docs.iter().map(|d| d + 5));
        let mut all_values = a_values.clone();
        all_values.extend_from_slice(&b_values);
        let (flush_vec, flush_meta) = write(
            &[FlatVectorsField {
                field_number: 7,
                similarity: VectorSimilarityFunction::Cosine,
                dimension: dim,
                docs: all_docs,
                values: FieldVectorData::Float32(all_values),
            }],
            9,
        );
        assert_eq!(
            merged_vec, flush_vec,
            ".vec differs between merge and flush"
        );
        assert_eq!(
            merged_meta, flush_meta,
            ".vemf differs between merge and flush"
        );
    }

    /// A document the merge drops takes its vector with it, and the ordinals
    /// of everything after it shift down -- which is the part a bulk byte copy
    /// can get wrong, since the surviving run is no longer the whole field.
    #[test]
    fn a_merge_drops_deleted_documents_and_renumbers_the_rest() {
        let dim = 2;
        let values: Vec<f32> = (0..6 * dim).map(|i| i as f32).collect();
        let (src_vec, src_meta) = write(
            &[FlatVectorsField {
                field_number: 1,
                similarity: VectorSimilarityFunction::Euclidean,
                dimension: dim,
                docs: (0..6).collect(),
                values: FieldVectorData::Float32(values.clone()),
            }],
            6,
        );
        let src = FlatVectorsReader::open(&src_meta, &src_vec, &ID, "").unwrap();
        // Drop docs 1 and 4 -- one inside a run, one at the end but not last,
        // so both the "flush the run" and "start a new run" branches fire.
        let map: Vec<i32> = vec![0, -1, 1, 2, -1, 3];
        let sources = vec![FlatVectorMergeSource {
            values: MergeSourceValues::Float32(src.float_vector_values(1).unwrap()),
            doc_map: &map,
        }];
        let mut writer = FlatVectorsWriter::new(4, &ID, "");
        writer
            .merge_one_flat_vector_field(&MergedFlatVectorField {
                field_number: 1,
                encoding: VectorEncoding::Float32,
                similarity: VectorSimilarityFunction::Euclidean,
                dimension: dim,
                sources: &sources,
            })
            .unwrap();
        let (merged_vec, merged_meta) = writer.finish();

        let merged = FlatVectorsReader::open(&merged_meta, &merged_vec, &ID, "").unwrap();
        let out = merged.float_vector_values(1).unwrap();
        assert_eq!(out.size(), 4);
        // Dense again: four vectors over four documents.
        for (new_ord, old_doc) in [0i32, 2, 3, 5].into_iter().enumerate() {
            let new_ord = new_ord as i32;
            assert_eq!(out.ord_to_doc(new_ord).unwrap(), new_ord);
            assert_eq!(
                out.vector(new_ord).unwrap(),
                values[old_doc as usize * dim as usize..(old_doc as usize + 1) * dim as usize]
                    .to_vec()
            );
        }
    }

    /// A BYTE field merges the same way, and the encoding/dimension the caller
    /// declares is cross-checked against every source rather than trusted --
    /// a source opened for the wrong field would otherwise be copied in
    /// verbatim and produce a `.vec` whose length no longer matches its
    /// `.vemf`.
    #[test]
    fn merge_checks_every_source_against_the_declared_field() {
        let (f32_vec, f32_meta) = write(&[float_field(0, 4, (0..3).collect())], 3);
        let f32_reader = FlatVectorsReader::open(&f32_meta, &f32_vec, &ID, "").unwrap();
        let map: Vec<i32> = (0..3).collect();

        let sources = vec![FlatVectorMergeSource {
            values: MergeSourceValues::Float32(f32_reader.float_vector_values(0).unwrap()),
            doc_map: &map,
        }];
        let mut writer = FlatVectorsWriter::new(3, &ID, "");
        let clean = writer.data.len();
        // Declared BYTE, source is FLOAT32.
        assert!(matches!(
            writer.merge_one_flat_vector_field(&MergedFlatVectorField {
                field_number: 0,
                encoding: VectorEncoding::Byte,
                similarity: VectorSimilarityFunction::Euclidean,
                dimension: 4,
                sources: &sources,
            }),
            Err(Error::EncodingMismatch(
                0,
                VectorEncoding::Float32,
                VectorEncoding::Byte
            ))
        ));
        // Declared dimension disagrees with the source's.
        assert!(matches!(
            writer.merge_one_flat_vector_field(&MergedFlatVectorField {
                field_number: 0,
                encoding: VectorEncoding::Float32,
                similarity: VectorSimilarityFunction::Euclidean,
                dimension: 8,
                sources: &sources,
            }),
            Err(Error::DimensionMismatch(0, 8, 4))
        ));
        // A non-positive dimension.
        assert!(matches!(
            writer.merge_one_flat_vector_field(&MergedFlatVectorField {
                field_number: 0,
                encoding: VectorEncoding::Float32,
                similarity: VectorSimilarityFunction::Euclidean,
                dimension: 0,
                sources: &sources,
            }),
            Err(Error::DimensionMismatch(0, 0, 0))
        ));
        // A failed call must leave the writer untouched -- otherwise the
        // alignment padding (and, for a later failure, whole runs of copied
        // vectors) would leak into whatever field is written next.
        assert_eq!(writer.data.len(), clean);
        writer
            .merge_one_flat_vector_field(&MergedFlatVectorField {
                field_number: 0,
                encoding: VectorEncoding::Float32,
                similarity: VectorSimilarityFunction::Euclidean,
                dimension: 4,
                sources: &sources,
            })
            .unwrap();
        let (data, meta) = writer.finish();
        let reader = FlatVectorsReader::open(&meta, &data, &ID, "").unwrap();
        assert_eq!(reader.float_vector_values(0).unwrap().size(), 3);
    }

    /// An **index-sorted** merge interleaves its sources: source 0's vectors
    /// land on merged docs 0 and 3, source 1's on 1 and 2. The merged ordinal
    /// space must then follow the merged *document* order, not source order --
    /// `.vemf`'s doc list is an `IndexedDISI`, which only encodes a strictly
    /// ascending list, so source order here is not merely a different
    /// convention but an unwritable one.
    ///
    /// The vectors themselves are checked component by component against the
    /// document they came from: that is the assertion a wrong ordinal
    /// assignment breaks while leaving a segment that decodes cleanly.
    #[test]
    fn an_interleaving_doc_map_assigns_merged_ordinals_in_document_order() {
        // Two sources of two vectors each, dimension 2, with distinguishable
        // values.
        let (vec_a, meta_a) = write(&[float_field(0, 2, vec![0, 1])], 2);
        let (vec_b, meta_b) = write(&[float_field(0, 2, vec![0, 1])], 2);
        let src_a = FlatVectorsReader::open(&meta_a, &vec_a, &ID, "").unwrap();
        let src_b = FlatVectorsReader::open(&meta_b, &vec_b, &ID, "").unwrap();
        let values_a = src_a.float_vector_values(0).unwrap();
        let values_b = src_b.float_vector_values(0).unwrap();
        let want_a: Vec<Vec<f32>> = (0..2)
            .map(|o| values_a.vector(o).unwrap().to_vec())
            .collect();
        let want_b: Vec<Vec<f32>> = (0..2)
            .map(|o| values_b.vector(o).unwrap().to_vec())
            .collect();

        // source 0 doc 0 -> merged 0, doc 1 -> merged 3;
        // source 1 doc 0 -> merged 1, doc 1 -> merged 2.
        let map_a = vec![0i32, 3];
        let map_b = vec![1i32, 2];
        let sources = vec![
            FlatVectorMergeSource {
                values: MergeSourceValues::Float32(values_a),
                doc_map: &map_a,
            },
            FlatVectorMergeSource {
                values: MergeSourceValues::Float32(values_b),
                doc_map: &map_b,
            },
        ];
        let mut writer = FlatVectorsWriter::new(4, &ID, "");
        writer
            .merge_one_flat_vector_field(&MergedFlatVectorField {
                field_number: 0,
                encoding: VectorEncoding::Float32,
                similarity: VectorSimilarityFunction::Euclidean,
                dimension: 2,
                sources: &sources,
            })
            .unwrap();
        let (data, meta) = writer.finish();
        let merged = FlatVectorsReader::open(&meta, &data, &ID, "").unwrap();
        let values = merged.float_vector_values(0).unwrap();
        assert_eq!(values.size(), 4);
        let expected: Vec<(i32, &Vec<f32>)> = vec![
            (0, &want_a[0]),
            (1, &want_b[0]),
            (2, &want_b[1]),
            (3, &want_a[1]),
        ];
        for (ord, (doc, want)) in expected.iter().enumerate() {
            assert_eq!(values.ord_to_doc(ord as i32).unwrap(), *doc);
            assert_eq!(
                values.vector(ord as i32).unwrap(),
                want.as_slice(),
                "merged ordinal {ord} holds another document's vector"
            );
        }
    }

    /// A doc map that maps two source documents onto the same merged id, or
    /// that runs backwards, produces a doc list `IndexedDISI.writeBitSet`
    /// cannot encode -- and would otherwise yield a segment that decodes
    /// cleanly and answers wrongly. Same rule the flush path enforces.
    #[test]
    fn merge_rejects_a_doc_map_that_is_not_ascending_or_is_short() {
        let (src_vec, src_meta) = write(&[float_field(0, 2, (0..3).collect())], 3);
        let src = FlatVectorsReader::open(&src_meta, &src_vec, &ID, "").unwrap();

        for map in [vec![2i32, 1, 0], vec![0, 0, 1]] {
            let sources = vec![FlatVectorMergeSource {
                values: MergeSourceValues::Float32(src.float_vector_values(0).unwrap()),
                doc_map: &map,
            }];
            let mut writer = FlatVectorsWriter::new(3, &ID, "");
            assert!(
                matches!(
                    writer.merge_one_flat_vector_field(&MergedFlatVectorField {
                        field_number: 0,
                        encoding: VectorEncoding::Float32,
                        similarity: VectorSimilarityFunction::Euclidean,
                        dimension: 2,
                        sources: &sources,
                    }),
                    Err(Error::CorruptMeta(_))
                ),
                "doc map {map:?} must be rejected"
            );
        }

        // A doc map shorter than the source's document count.
        let short: Vec<i32> = vec![0];
        let sources = vec![FlatVectorMergeSource {
            values: MergeSourceValues::Float32(src.float_vector_values(0).unwrap()),
            doc_map: &short,
        }];
        let mut writer = FlatVectorsWriter::new(3, &ID, "");
        assert!(matches!(
            writer.merge_one_flat_vector_field(&MergedFlatVectorField {
                field_number: 0,
                encoding: VectorEncoding::Float32,
                similarity: VectorSimilarityFunction::Euclidean,
                dimension: 2,
                sources: &sources,
            }),
            Err(Error::CorruptMeta(_))
        ));
    }

    /// A field with no vectors in any source is legal: Java's
    /// `DocsWithFieldSet` is simply empty and `writeStoredMeta` records the
    /// `-2` empty marker.
    #[test]
    fn merging_a_field_no_source_has_vectors_for_writes_an_empty_field() {
        let mut writer = FlatVectorsWriter::new(10, &ID, "");
        writer
            .merge_one_flat_vector_field(&MergedFlatVectorField {
                field_number: 3,
                encoding: VectorEncoding::Float32,
                similarity: VectorSimilarityFunction::DotProduct,
                dimension: 5,
                sources: &[],
            })
            .unwrap();
        let (data, meta) = writer.finish();
        let reader = FlatVectorsReader::open(&meta, &data, &ID, "").unwrap();
        let values = reader.float_vector_values(3).unwrap();
        assert_eq!(values.size(), 0);
        assert!(values.is_empty());
        assert!(reader.field(3).unwrap().ord_to_doc.is_empty());
    }

    /// The writer can carry flushed and merged fields in one file pair, which
    /// is what a merge that gains a brand-new field needs.
    #[test]
    fn a_flushed_field_and_a_merged_field_share_one_file_pair() {
        let (src_vec, src_meta) = write(&[float_field(0, 2, (0..4).collect())], 4);
        let src = FlatVectorsReader::open(&src_meta, &src_vec, &ID, "").unwrap();
        let map: Vec<i32> = (0..4).collect();
        let sources = vec![FlatVectorMergeSource {
            values: MergeSourceValues::Float32(src.float_vector_values(0).unwrap()),
            doc_map: &map,
        }];

        let mut writer = FlatVectorsWriter::new(4, &ID, "");
        writer
            .merge_one_flat_vector_field(&MergedFlatVectorField {
                field_number: 0,
                encoding: VectorEncoding::Float32,
                similarity: VectorSimilarityFunction::Euclidean,
                dimension: 2,
                sources: &sources,
            })
            .unwrap();
        writer
            .write_field(&FlatVectorsField {
                field_number: 1,
                similarity: VectorSimilarityFunction::DotProduct,
                dimension: 3,
                docs: vec![1, 3],
                values: FieldVectorData::Byte(vec![1, 2, 3, 250, 251, 252]),
            })
            .unwrap();
        let (data, meta) = writer.finish();
        let reader = FlatVectorsReader::open(&meta, &data, &ID, "").unwrap();
        assert_eq!(reader.fields().len(), 2);
        assert_eq!(reader.float_vector_values(0).unwrap().size(), 4);
        let bytes = reader.byte_vector_values(1).unwrap();
        assert_eq!(bytes.vector(0).unwrap(), &[1, 2, 3]);
        assert_eq!(bytes.ord_to_doc(1).unwrap(), 3);
        // The merged field's region is still 64-byte aligned, and the byte
        // field's 4-byte aligned, exactly as a flush would leave them.
        assert_eq!(reader.field(0).unwrap().vector_data_offset % 64, 0);
        assert_eq!(reader.field(1).unwrap().vector_data_offset % 4, 0);
    }
}
