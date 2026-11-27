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
use lucene_util::vector_util;

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

/// The OpenSearch k-NN plugin's `KNN1040HnswHalfFloatVectorsFormat`: a
/// `Lucene99HnswVectorsFormat` graph over `KNN1040HalfFloatFlatVectorsFormat`,
/// which stores FLOAT32 fields as IEEE half floats (two little-endian bytes a
/// component) and scores them decoded. Its `.vemf` records no similarity: the
/// field's `FieldInfo` does ([`FlatVectorsReader::resolve_similarities`]).
pub const HALF_FLOAT_FORMAT_NAME: &str = "KNN1040HnswHalfFloatVectorsFormat";
/// `KNN1040HalfFloatFlatVectorsFormat`'s `.vemf` codec name.
pub const HALF_FLOAT_META_CODEC: &str = "KNN1040HalfFloatFlatVectorsFormatMeta";
/// `KNN1040HalfFloatFlatVectorsFormat`'s `.vec` codec name.
pub const HALF_FLOAT_DATA_CODEC: &str = "KNN1040HalfFloatFlatVectorsFormatData";
/// The half-float flat files' only version.
const HALF_FLOAT_VERSION: i32 = 0;

/// Whether `segment_suffix` (`<format>_<n>`) names a per-field instance of
/// `format`.
fn instance_of(segment_suffix: &str, format: &str) -> bool {
    segment_suffix
        .strip_prefix(format)
        .and_then(|rest| rest.strip_prefix('_'))
        .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
}

/// Whether a per-field instance with codec suffix `segment_suffix` is
/// [`HALF_FLOAT_FORMAT_NAME`]'s, whose flat files hold half floats.
pub fn stores_half_floats(segment_suffix: &str) -> bool {
    instance_of(segment_suffix, HALF_FLOAT_FORMAT_NAME)
}

/// `Float.floatToFloat16`: round to nearest even, overflow to infinity.
// ARITH: past the early returns `2^-25 < |f| < 65520`, so the exponent is in
// `-25..=15`, `expdelta <= 11`, every shift is below 32, and a rounded-up
// significand is at most `0x400`.
#[allow(clippy::arithmetic_side_effects)]
pub fn f32_to_f16(f: f32) -> u16 {
    let doppel = f.to_bits();
    let sign = ((doppel >> 16) & 0x8000) as u16;
    // HotSpot's intrinsic (`vcvtps2ph`): the NaN quieted, its significand's
    // top bits kept. k-NN never stores one.
    if f.is_nan() {
        return sign | 0x7e00 | ((doppel >> 13) & 0x03ff) as u16;
    }
    let abs = f.abs();
    // 0x1.ffcp15 + 0x0.002p15: halfway past the largest half float.
    if abs >= 65520.0 {
        return sign | 0x7c00;
    }
    // At most half the smallest subnormal: rounds to zero.
    if abs <= f32::from_bits(0x3300_0000) {
        return sign;
    }
    // `Math.getExponent`: the unbiased exponent (-127 for a subnormal f32,
    // which is far below any half float and returned as zero above).
    let mut exp = ((doppel >> 23) & 0xff) as i32 - 127;
    let mut expdelta = 0u32;
    let mut msb = 0u32;
    if exp < -14 {
        expdelta = (-14 - exp) as u32;
        exp = -15;
        msb = 0x0080_0000;
    }
    let signif = (doppel & 0x007f_ffff) | msb;
    let mut bits = (signif >> (13 + expdelta)) as u16;
    let lsb = signif & (1 << (13 + expdelta));
    let round = signif & (1 << (12 + expdelta));
    let sticky = signif & ((1 << (12 + expdelta)) - 1);
    if round != 0 && (lsb | sticky) != 0 {
        bits += 1;
    }
    // Adding, not or-ing: a significand that rounded up to 0x400 carries into
    // the exponent, as Java's `+` does.
    sign | ((((exp + 15) as u16) << 10) + bits)
}

/// `Float.float16ToFloat`.
// ARITH: a five-bit exponent plus 112 and a ten-bit significand.
#[allow(clippy::arithmetic_side_effects)]
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((h >> 10) & 0x1f) as i32;
    let mant = (h & 0x03ff) as u32;
    match exp {
        0 => sign * mant as f32 * f32::from_bits(0x3380_0000), // 2^-24
        31 => f32::from_bits((u32::from(h & 0x8000) << 16) | 0x7f80_0000 | (mant << 13)),
        _ => f32::from_bits(
            (u32::from(h & 0x8000) << 16) | (((exp + 112) as u32) << 23) | (mant << 13),
        ),
    }
}

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

/// One stored component's bytes: [`byte_size`], or two for half floats.
pub(crate) fn element_bytes(encoding: VectorEncoding, half: bool) -> usize {
    if half {
        2
    } else {
        byte_size(encoding)
    }
}

/// Where each field's vector data starts, whatever the encoding: a page
/// (Lucene main's `Lucene99FlatVectorsWriter.alignOutput`). Readers find the
/// data through the offset in the metadata, so stock readers open these
/// files. A vector whose size divides a page, or is a multiple of one, then
/// never straddles a page boundary: a read touches one page instead of two
/// when memory is short. Costs under a page of padding per field and
/// segment.
pub const VECTOR_DATA_ALIGNMENT: usize = 4096;

/// What Lucene 10.5 and 10.6 write: 64 bytes (Arm Neoverse cache-line
/// reads) for float vectors, 4 for bytes. Byte-for-byte comparisons against
/// fixtures made by that Lucene turn it on with
/// [`use_stock_vector_data_alignment`].
fn stock_alignment(encoding: VectorEncoding, half: bool) -> usize {
    match (encoding, half) {
        (VectorEncoding::Byte, false) => 4,
        _ => 64,
    }
}

static STOCK_ALIGNMENT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Makes later-written flat vector data start where Lucene 10.5 and 10.6
/// write it, for the whole process, instead of at
/// [`VECTOR_DATA_ALIGNMENT`].
pub fn use_stock_vector_data_alignment() {
    STOCK_ALIGNMENT.store(true, std::sync::atomic::Ordering::Relaxed);
}

// ---------------------------------------------------------------------------
// VectorUtil
// ---------------------------------------------------------------------------

/// Port of `VectorUtil.squareDistance(float[], float[])`.
///
/// The kernel is [`lucene_util::simd::square_distance_f32`]: explicit
/// AVX2+FMA with `PanamaVectorUtilSupport`'s layout -- four 8-lane
/// accumulators over 32-element steps, fused multiply-adds -- and a scalar
/// specification with the same layout and fold order for every other target.
/// A single running `sum` would be a serial dependency chain LLVM may not
/// reassociate; an accumulator array it did not vectorize either (0.3x of
/// Lucene), which is why this is written with intrinsics. The lane split
/// changes the summation order and so the last ulp or two against Java's
/// scalar `DefaultVectorUtilSupport`, exactly as switching between Lucene's
/// own two implementations does.
pub fn square_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "vector dimensions differ");
    lucene_util::simd::square_distance_f32(a, b)
}

/// Port of `VectorUtil.dotProduct(float[], float[])`; see [`square_distance`]
/// for the kernel.
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "vector dimensions differ");
    lucene_util::simd::dot_f32(a, b)
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
/// [`cosine`] with `a . a` given: the same value bit for bit when
/// `norm1` is `lucene_util::simd::cosine_parts_f32(a, _).1`.
pub fn cosine_with_norm(a: &[f32], norm1: f32, b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "vector dimensions differ");
    let (sum, norm2) = lucene_util::simd::dot_and_norm_f32(a, b);
    if norm1 == 0.0 || norm2 == 0.0 {
        return 0.0;
    }
    (sum as f64 / ((norm1 as f64) * (norm2 as f64)).sqrt()) as f32
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "vector dimensions differ");
    // The three sums, two 8-lane accumulators each (Panama's `cosineBody`).
    let (sum, norm1, norm2) = lucene_util::simd::cosine_parts_f32(a, b);
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
/// every byte kernel takes `&[u8]` straight off the `.vec` file and
/// sign-extends each element, exactly as Java does. The kernels are
/// `lucene_util::vector_util`'s AVX2 ones (`vpmovsxbw` + `vpmaddwd`, 32 bytes
/// a step), whose scalar specifications are the `as i8 as i32` loops.
pub fn square_distance_bytes(a: &[u8], b: &[u8]) -> i32 {
    debug_assert_eq!(a.len(), b.len(), "vector dimensions differ");
    // Java's own bound (`VectorUtil`: "this will not overflow if dim <
    // 2^18"): above it the `int` accumulator wraps, and the kernel wraps the
    // same way. Each kernel scores the common prefix, as the zip it replaced
    // did, so a corrupt dimension stays a wrong score and never a panic.
    let n = a.len().min(b.len());
    vector_util::square_distance_i8(&a[..n], &b[..n])
}

/// Port of `VectorUtil.dotProduct(byte[], byte[])` -- an `i32` sum. See
/// [`square_distance_bytes`] on why `&[u8]` and not `&[i8]`.
pub fn dot_product_bytes(a: &[u8], b: &[u8]) -> i32 {
    debug_assert_eq!(a.len(), b.len(), "vector dimensions differ");
    let n = a.len().min(b.len());
    vector_util::dot_product_i8(&a[..n], &b[..n])
}

/// Port of `DefaultVectorUtilSupport.cosine(byte[], byte[])`: `i32`
/// accumulators, then the same single `f64` square root of the product.
pub fn cosine_bytes(a: &[u8], b: &[u8]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "vector dimensions differ");
    let n = a.len().min(b.len());
    let (sum, norm1, norm2) = vector_util::cosine_sums_i8(&a[..n], &b[..n]);
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
    /// The retired `Lucene90`/`Lucene91` HNSW formats' `int[] ordToDoc`, read
    /// whole into memory at open (`Lucene90HnswVectorsReader.FieldEntry`,
    /// `Lucene91HnswVectorsReader.FieldEntry`): `ords[ord]` is the doc id.
    /// Strictly increasing -- the backward-codecs reader rejects anything else
    /// when it builds this, because [`DocToOrdCursor`] binary-searches it.
    Explicit(std::sync::Arc<[i32]>),
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
/// A guard written `if start + length > file.len()` forms the very sum it exists
/// to guard (the shape `docs/arithmetic-gate.md` names): a negative `addressesOffset` arrives as `usize::MAX` through
/// `as usize`, the addition wraps to something small, the comparison
/// *passes*, and the slice then panics with `start > end`. This decoder is on the query path, so through the FFI that is a dead JVM.
pub(crate) fn file_region(file: &[u8], offset: i64, length: i64) -> Option<&[u8]> {
    let start = usize::try_from(offset).ok()?;
    let end = start.checked_add(usize::try_from(length).ok()?)?;
    file.get(start..end)
}

impl OrdToDoc {
    /// Port of `OrdToDocDISIReaderConfiguration.fromStoredMeta`.
    pub(crate) fn from_stored_meta(input: &mut SliceInput<'_>, size: i32) -> Result<Self> {
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
pub(crate) fn write_stored_meta(
    meta: &mut Vec<u8>,
    data: &mut Vec<u8>,
    data_before: i64,
    docs: &[i32],
    max_doc: i32,
) {
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
        let offset = data_before.saturating_add(data.len() as i64);
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
        let disi_len = data_before + data.len() as i64 - offset;
        meta.write_i64(disi_len);
        meta.write_i16(jump_table_entry_count);
        meta.write_byte(indexed_disi::DEFAULT_DENSE_RANK_POWER);

        let start = data_before.saturating_add(data.len() as i64);
        meta.write_i64(start);
        meta.write_vint(DIRECT_MONOTONIC_BLOCK_SHIFT as i32);
        let values: Vec<i64> = docs.iter().map(|d| *d as i64).collect();
        let (dm_meta, dm_data) = direct_monotonic::write(&values, DIRECT_MONOTONIC_BLOCK_SHIFT);
        meta.extend_from_slice(&dm_meta);
        data.extend_from_slice(&dm_data);
        // ARITH: `start` was `data.len()` before the `extend_from_slice`
        // above; same argument as `disi_len`.
        #[allow(clippy::arithmetic_side_effects)]
        let addresses_len = data_before + data.len() as i64 - start;
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
    fields: std::sync::Arc<Vec<FlatFieldEntry>>,
    /// Byte vectors are scored by Hamming distance
    /// ([`scores_hamming`]) whatever their recorded similarity.
    hamming: bool,
    /// Float vectors are stored as half floats ([`stores_half_floats`]).
    half: bool,
    /// Whether every field's similarity is known: a half-float `.vemf`
    /// leaves it to [`Self::resolve_similarities`].
    resolved: bool,
}

/// The OpenSearch k-NN plugin's `KNN990HnswBinaryVectorsFormat`: the
/// `Lucene99HnswVectorsFormat` files over packed bit vectors, scored by
/// `KNN9120BinaryVectorScorer` as `1 / (1 + hamming distance)`. The fields
/// record `EUCLIDEAN`, which nothing reads.
pub const HAMMING_FORMAT_NAME: &str = "KNN990HnswBinaryVectorsFormat";

/// Whether a per-field instance with codec suffix `segment_suffix`
/// (`<format>_<n>`) is [`HAMMING_FORMAT_NAME`]'s, whose reader and writer
/// score byte vectors by Hamming distance.
pub fn scores_hamming(segment_suffix: &str) -> bool {
    instance_of(segment_suffix, HAMMING_FORMAT_NAME)
}

/// `KNNVectorSimilarityFunction.HAMMING.compare`: `1 / (1 + popcount(a ^ b))`.
#[inline]
pub fn hamming_score(a: &[u8], b: &[u8]) -> f32 {
    1.0 / (1.0 + lucene_util::vector_util::xor_bit_count(a, b) as f32)
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
        let half = stores_half_floats(segment_suffix);
        // `KNN1040HalfFloatFlatVectorsFormat` has its own codec names, at
        // version 0 only.
        let (meta_codec, data_codec, start, current) = if half {
            (
                HALF_FLOAT_META_CODEC,
                HALF_FLOAT_DATA_CODEC,
                HALF_FLOAT_VERSION,
                HALF_FLOAT_VERSION,
            )
        } else {
            (META_CODEC, DATA_CODEC, VERSION_START, VERSION_CURRENT)
        };
        let mut meta = SliceInput::new(meta_buf);
        let meta_version = codec_util::check_index_header(
            &mut meta,
            meta_codec,
            start,
            current,
            segment_id,
            segment_suffix,
        )?;
        let mut data_in = SliceInput::new(data_buf);
        let data_version = codec_util::check_index_header(
            &mut data_in,
            data_codec,
            start,
            current,
            segment_id,
            segment_suffix,
        )?;
        let (meta_version, data_version) = (meta_version.version, data_version.version);
        if meta_version != data_version {
            return corrupt(format!(
                "Format versions mismatch: meta={meta_version}, {data_codec}={data_version}"
            ));
        }
        // A file shorter than its own footer is corruption, not a subtraction:
        // `len - FOOTER_LENGTH` on a truncated `.vemf` underflows to a huge
        // offset (release) or panics (debug).
        let (Some(meta_footer), Some(_)) = (
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
        // `Lucene99FlatVectorsReader` checksums the small `.vemf` as it reads
        // it but only `CodecUtil.retrieveChecksum`s the `.vec`: re-reading
        // the whole data file belongs to `checkIntegrity`
        // ([`Self::check_integrity`]), not to every open.
        codec_util::retrieve_checksum(data_buf)?;

        let mut fields = Vec::new();
        loop {
            let field_number = meta.read_i32()?;
            if field_number == -1 {
                break;
            }
            if field_number < 0 {
                return corrupt(format!("Invalid field number: {field_number}"));
            }
            fields.push(read_field_entry(
                &mut meta,
                field_number,
                data_buf.len(),
                half,
            )?);
        }

        Ok(FlatVectorsReader {
            data: data_buf,
            fields: std::sync::Arc::new(fields),
            hamming: scores_hamming(segment_suffix),
            half,
            resolved: !half,
        })
    }

    /// Whether this reader's float vectors are stored as half floats.
    pub fn stores_half_floats(&self) -> bool {
        self.half
    }

    /// Sets each field's similarity from its `FieldInfo`
    /// (`similarity(field_number)`), as `KNN1040HalfFloatFlatVectorsReader`
    /// takes it from `FieldInfos`. A no-op for a `.vemf` that records its
    /// own; required before reading a half-float reader's vectors.
    ///
    /// # Errors
    /// A field `similarity` does not know.
    pub fn resolve_similarities(
        &mut self,
        similarity: impl Fn(i32) -> Option<VectorSimilarityFunction>,
    ) -> Result<()> {
        if self.resolved {
            return Ok(());
        }
        for field in std::sync::Arc::make_mut(&mut self.fields) {
            field.similarity =
                similarity(field.field_number).ok_or(Error::UnknownField(field.field_number))?;
        }
        self.resolved = true;
        Ok(())
    }

    /// [`Self::resolve_similarities`] from the segment's `FieldInfos`.
    ///
    /// # Errors
    /// A vector field `fields` does not hold.
    pub fn resolve_similarities_from(
        &mut self,
        fields: &[crate::field_infos::FieldInfo],
    ) -> Result<()> {
        self.resolve_similarities(|number| {
            fields
                .iter()
                .find(|f| f.number == number)
                .map(|f| f.vector_similarity_function)
        })
    }

    /// A reader over field entries some other format's metadata described
    /// -- the retired HNSW formats (`Lucene90`..`Lucene95`) kept their vectors
    /// in the same `.vec` layout this reader serves, but described them in
    /// their own `.vem`. Every entry must already have passed
    /// [`check_vector_region`] against `data`.
    pub(crate) fn from_entries(data: &'a [u8], fields: Vec<FlatFieldEntry>) -> Self {
        FlatVectorsReader {
            data,
            fields: std::sync::Arc::new(fields),
            hamming: false,
            half: false,
            resolved: true,
        }
    }

    /// `Lucene99FlatVectorsReader.checkIntegrity()`:
    /// `CodecUtil.checksumEntireFile` over the `.vec` this reader serves.
    ///
    /// # Errors
    /// A footer or checksum that does not verify.
    pub fn check_integrity(&self) -> Result<()> {
        let end = self.data.len().saturating_sub(codec_util::FOOTER_LENGTH);
        codec_util::check_whole_file_footer(self.data, end)?;
        Ok(())
    }

    pub fn fields(&self) -> &[FlatFieldEntry] {
        &self.fields
    }

    /// This reader without its data file, to keep beside the segment's
    /// files: [`Self::attach`] it to the same data for each search, so the
    /// metadata is parsed and checked once and shared, not copied.
    pub fn detached(&self) -> FlatVectorsReader<'static> {
        FlatVectorsReader {
            data: &[],
            fields: self.fields.clone(),
            hamming: self.hamming,
            half: self.half,
            resolved: self.resolved,
        }
    }

    /// A detached reader ([`Self::detached`]) over `data`, which must be the
    /// file the reader was opened on.
    pub fn attach<'b>(&self, data: &'b [u8]) -> FlatVectorsReader<'b> {
        FlatVectorsReader {
            data,
            fields: self.fields.clone(),
            hamming: self.hamming,
            half: self.half,
            resolved: self.resolved,
        }
    }

    pub fn field(&self, field_number: i32) -> Option<&FlatFieldEntry> {
        self.fields.iter().find(|f| f.field_number == field_number)
    }

    /// `getVectorCount(fieldInfo)` (Lucene 10.6, `branch_10_6`, #16647): the
    /// field's vector count as its metadata entry holds it, without opening
    /// its vector values.
    /// `Lucene99FlatVectorsReader`'s: `getFieldEntryOrThrow(name).size()`.
    ///
    /// # Errors
    /// [`Error::UnknownField`] for a field this reader does not hold (Java's
    /// `IllegalArgumentException`).
    pub fn vector_count(&self, field_number: i32) -> Result<i32> {
        self.field(field_number)
            .map(|f| f.size)
            .ok_or(Error::UnknownField(field_number))
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
        if !self.resolved {
            return corrupt(format!(
                "field {}: half-float vectors read before their similarity was resolved",
                entry.field_number
            ));
        }
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
        let Some(vector_bytes) =
            (entry.dimension as usize).checked_mul(element_bytes(entry.encoding, self.half))
        else {
            return corrupt(format!(
                "vector dimension {} too large for this platform",
                entry.dimension
            ));
        };
        // k-NN normalizes half-float cosine vectors (and queries) before they
        // get here, and scores them with its FP16 cosine kernel, `(1 + dot) /
        // 2`: the dot-product score, which is the cosine one for unit vectors.
        let similarity = if self.half && entry.similarity == VectorSimilarityFunction::Cosine {
            VectorSimilarityFunction::DotProduct
        } else {
            entry.similarity
        };
        Ok(RawVectorValues {
            slice: region,
            file: self.data,
            dimension: entry.dimension as usize,
            vector_bytes,
            size: entry.size,
            similarity,
            hamming: self.hamming && entry.encoding == VectorEncoding::Byte,
            half: self.half,
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
    half: bool,
) -> Result<FlatFieldEntry> {
    let encoding = read_vector_encoding(meta)?;
    if half && encoding != VectorEncoding::Float32 {
        return corrupt(format!(
            "field {field_number}: half-float vectors with encoding {encoding:?}"
        ));
    }
    // A half-float `.vemf` has no similarity; `resolve_similarities` sets it.
    let similarity = if half {
        VectorSimilarityFunction::Euclidean
    } else {
        read_similarity_function(meta)?
    };
    let vector_data_offset = meta.read_vlong()?;
    let vector_data_length = meta.read_vlong()?;
    let dimension = meta.read_vint()?;
    let size = meta.read_i32()?;
    check_vector_region(
        element_bytes(encoding, half),
        dimension,
        size,
        vector_data_offset,
        vector_data_length,
        data_len,
    )?;
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

/// The checks every `FieldEntry` of a flat vector store gets before any
/// vector is read: a positive dimension, a non-negative count, and a data
/// region that is exactly `size * dimension * byteSize` bytes and lies inside
/// the `.vec` file. Shared by [`read_field_entry`] and the retired HNSW
/// formats of `crate::backward_codecs::hnsw_vectors`, whose readers assert
/// the same identity (`validateFieldEntry`).
pub(crate) fn check_vector_region(
    element_bytes: usize,
    dimension: i32,
    size: i32,
    vector_data_offset: i64,
    vector_data_length: i64,
    data_len: usize,
) -> Result<()> {
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
        .checked_mul(element_bytes as i64)
        .and_then(|v| v.checked_mul(size as i64))
    else {
        return corrupt(format!(
            "vector data length overflows: size={size} * dim={dimension} * byteSize={element_bytes}"
        ));
    };
    if expected != vector_data_length {
        return corrupt(format!(
            "Vector data length {vector_data_length} not matching size={size} * dim={dimension} \
             * byteSize={element_bytes} = {expected}"
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
    Ok(())
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
    /// Byte vectors scored by [`hamming_score`] instead of `similarity`.
    hamming: bool,
    /// Float components stored as little-endian half floats.
    half: bool,
    ord_to_doc: OrdToDoc,
}

impl RawVectorValues<'_> {
    /// [`crate::hnsw::VectorScorer::read_ahead_span`].
    fn read_ahead_span(&self) -> usize {
        crate::hnsw::read_ahead_span(self.vector_bytes)
    }

    /// [`crate::hnsw::VectorScorer::read_ahead`]: the vectors of ordinals
    /// `from..to`, clamped to the field.
    fn read_ahead(&self, from: i32, to: i32) {
        crate::hnsw::read_ahead_rows(self.slice, self.size, self.vector_bytes, from, to);
    }

    /// Two byte vectors' score under this field's scoring: its similarity,
    /// or [`hamming_score`] for a [`HAMMING_FORMAT_NAME`] field.
    #[inline]
    fn score_bytes(&self, a: &[u8], b: &[u8]) -> f32 {
        if self.hamming {
            hamming_score(a, b)
        } else {
            self.similarity.score_bytes(a, b)
        }
    }

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
        // *success* path too -- 40% of this call in an A/B, on the
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
            // `ord < size` above, and the reader built `docs` with exactly
            // `size` entries -- but `get`, so a mismatch is an error.
            OrdToDoc::Explicit(docs) => match docs.get(ord as usize) {
                Some(&doc) => Ok(doc),
                None => Err(Error::OrdOutOfRange(ord, self.size)),
            },
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

    /// [`ord_to_doc`](Self::ord_to_doc) for every ordinal, in order, into
    /// `out` (cleared first): a sparse field's addresses decoded a block run
    /// at a time rather than one lookup per ordinal.
    fn ord_to_docs(&self, out: &mut Vec<i32>) -> Result<()> {
        out.clear();
        let size = self.size.max(0);
        match &self.ord_to_doc {
            OrdToDoc::Empty | OrdToDoc::Dense => out.extend(0..size),
            OrdToDoc::Explicit(docs) => out.extend(docs.iter().take(size as usize)),
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
                let mut wide = vec![0i64; size as usize];
                direct_monotonic::fill(region, meta, 0, &mut wide)?;
                out.extend(wide.iter().map(|&d| d as i32));
            }
        }
        Ok(())
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
    Dense {
        size: i32,
    },
    Sparse(Box<DisiCursor<'a>>),
    /// [`OrdToDoc::Explicit`], answered by binary search.
    Explicit(&'a [i32]),
}

impl DocToOrdCursor<'_> {
    /// The ordinal of `doc`, or `None` when `doc` has no vector.
    pub fn ordinal(&mut self, doc: i32) -> Result<Option<i32>> {
        match self {
            DocToOrdCursor::Empty => Ok(None),
            DocToOrdCursor::Dense { size } => Ok((doc >= 0 && doc < *size).then_some(doc)),
            DocToOrdCursor::Sparse(cursor) => Ok(cursor.advance_exact(doc)?.map(|o| o as i32)),
            // An ordinal is an index into an `i32`-sized array, so it fits.
            DocToOrdCursor::Explicit(docs) => Ok(docs.binary_search(&doc).ok().map(|o| o as i32)),
        }
    }

    /// ORs every document that has a vector into `words`, a bit set of
    /// `words.len() * 64` documents: the field's documents in ordinal order,
    /// so a document's ordinal is the number of set bits before it. A
    /// document past the set is corruption (a dense or explicit field's
    /// documents are dropped there instead, as no caller sizes `words` below
    /// the field's last document).
    pub fn or_docs_into(&self, words: &mut [u64]) -> Result<()> {
        let mut set = |doc: usize| {
            if let Some(w) = words.get_mut(doc >> 6) {
                *w |= 1u64 << (doc & 63);
            }
        };
        match self {
            DocToOrdCursor::Empty => {}
            DocToOrdCursor::Dense { size } => {
                for doc in 0..usize::try_from(*size).unwrap_or(0) {
                    set(doc);
                }
            }
            DocToOrdCursor::Sparse(cursor) => cursor.or_into_words(words)?,
            DocToOrdCursor::Explicit(docs) => {
                for &doc in docs.iter() {
                    if let Ok(d) = usize::try_from(doc) {
                        set(d);
                    }
                }
            }
        }
        Ok(())
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

            /// [`ord_to_doc`](Self::ord_to_doc) for every ordinal, in order,
            /// into `out` (cleared first), decoded in bulk.
            pub fn ord_to_docs(&self, out: &mut Vec<i32>) -> Result<()> {
                self.values.ord_to_docs(out)
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
                    OrdToDoc::Explicit(docs) => Ok(DocToOrdCursor::Explicit(docs)),
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
    /// as scoring the common prefix of a mismatched query would.
    pub fn vector_into(&self, ord: i32, out: &mut [f32]) -> Result<()> {
        if out.len() != self.values.dimension {
            return Err(Error::QueryDimensionMismatch(
                self.values.dimension as i32,
                out.len() as i32,
            ));
        }
        let bytes = self.values.bytes(ord)?;
        if self.values.half {
            for (slot, chunk) in out.iter_mut().zip(bytes.chunks_exact(2)) {
                *slot = f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
            }
        } else {
            for (slot, chunk) in out.iter_mut().zip(bytes.chunks_exact(4)) {
                *slot = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
        }
        Ok(())
    }

    /// `ord`'s vector, viewed in place in the `.vec` bytes when they can be
    /// ([`lucene_util::vector_util::le_f32s`]: the usual case), else decoded
    /// into `scratch` (which must be `dimension()` long, as for
    /// [`Self::vector_into`]). Either way a scorer reads the components once
    /// rather than once to decode and again to score.
    pub fn vector_ref<'s>(&'s self, ord: i32, scratch: &'s mut [f32]) -> Result<&'s [f32]> {
        let bytes = self.values.bytes(ord)?;
        if !self.values.half {
            if let Some(floats) = lucene_util::vector_util::le_f32s(bytes) {
                return Ok(floats);
            }
        }
        self.vector_into(ord, scratch)?;
        Ok(scratch)
    }

    /// The `.vec` bytes of ordinals `from..to`, verbatim.
    ///
    /// `.vec` stores FLOAT32 components as little-endian `f32`, which is
    /// byte-for-byte what this port would write for the same values -- so a
    /// merge can copy a whole run of vectors straight across with one memcpy
    /// instead of decoding and re-encoding each one (Java re-encodes through a
    /// `ByteBuffer` per vector). A half-float field's are half floats
    /// ([`Self::stores_half_floats`]).
    pub fn raw_range(&self, from_ord: i32, to_ord: i32) -> Result<&[u8]> {
        self.values.raw_range(from_ord, to_ord)
    }

    /// Whether the `.vec` holds these components as half floats.
    pub fn stores_half_floats(&self) -> bool {
        self.values.half
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
            let candidate = self.vector_ref(ord, &mut scratch)?;
            heap.insert_with_overflow(ord, self.values.similarity.score(query, candidate));
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
            heap.insert_with_overflow(ord, self.values.score_bytes(query, v));
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
    /// Where the `.vec` bytes go as they are written, when
    /// [`Self::stream_to`] gave one; `spilled` counts what it already holds.
    out: Option<lucene_store::FsIndexOutput>,
    spilled: u64,
    meta: Vec<u8>,
    max_doc: i32,
    /// Writes `KNN1040HalfFloatFlatVectorsFormat` ([`stores_half_floats`]).
    half: bool,
}

impl FlatVectorsWriter {
    /// `max_doc` is the segment's document count -- it is what decides whether
    /// a field is written dense (`count == maxDoc`, no ord->doc structures at
    /// all) or sparse.
    ///
    /// A [`HALF_FLOAT_FORMAT_NAME`] instance's `segment_suffix` makes it write
    /// `KNN1040HalfFloatFlatVectorsFormat`: float components as half floats,
    /// no similarity in `.vemf`.
    pub fn new(max_doc: i32, segment_id: &[u8; ID_LENGTH], segment_suffix: &str) -> Self {
        let half = stores_half_floats(segment_suffix);
        let (meta_codec, data_codec, version) = if half {
            (
                HALF_FLOAT_META_CODEC,
                HALF_FLOAT_DATA_CODEC,
                HALF_FLOAT_VERSION,
            )
        } else {
            (META_CODEC, DATA_CODEC, VERSION_CURRENT)
        };
        let mut data = Vec::new();
        codec_util::write_index_header(&mut data, data_codec, version, segment_id, segment_suffix);
        let mut meta = Vec::new();
        codec_util::write_index_header(&mut meta, meta_codec, version, segment_id, segment_suffix);
        FlatVectorsWriter {
            data,
            out: None,
            spilled: 0,
            meta,
            max_doc,
            half,
        }
    }

    /// Writes the `.vec` file to `out` as merged fields are copied in,
    /// instead of holding it. Only [`Self::merge_one_flat_vector_field`]
    /// streams, and a failed field is not rolled back; finish with
    /// [`Self::finish_streamed`]. The bytes are those [`Self::finish`] would
    /// return.
    pub fn stream_to(&mut self, out: lucene_store::FsIndexOutput) {
        self.out = Some(out);
    }

    /// The `.vec` file's length so far.
    fn data_len(&self) -> usize {
        usize::try_from(self.spilled)
            .unwrap_or(usize::MAX)
            .saturating_add(self.data.len())
    }

    /// Appends `bytes` to the file, passing them straight to the output
    /// when streaming.
    fn append(&mut self, bytes: &[u8]) {
        if let Some(out) = self.out.as_mut() {
            lucene_store::DataOutput::write_bytes(out, &self.data);
            self.spilled = self
                .spilled
                .saturating_add(u64::try_from(self.data.len()).unwrap_or(u64::MAX));
            self.data.clear();
            lucene_store::DataOutput::write_bytes(out, bytes);
            self.spilled = self
                .spilled
                .saturating_add(u64::try_from(bytes.len()).unwrap_or(u64::MAX));
        } else {
            self.data.extend_from_slice(bytes);
        }
    }

    /// [`Self::finish`] for a writer given [`Self::stream_to`]: the `.vec`
    /// is complete in the output, handed back for the caller to close, with
    /// the `.vemf` bytes.
    pub fn finish_streamed(mut self) -> (Option<lucene_store::FsIndexOutput>, Vec<u8>) {
        self.meta.write_i32(-1);
        codec_util::write_footer(&mut self.meta);
        let Some(mut out) = self.out.take() else {
            return (None, self.meta);
        };
        lucene_store::DataOutput::write_bytes(&mut out, &self.data);
        codec_util::write_footer_to(&mut out);
        (Some(out), self.meta)
    }

    /// One field's vectors to the streamed output, encoded a bounded piece at
    /// a time.
    fn stream_field(&mut self, values: &FieldVectorData) {
        const PIECE: usize = 1 << 16;
        let mut piece: Vec<u8> = Vec::with_capacity(PIECE);
        match values {
            FieldVectorData::Byte(v) => self.append(v),
            FieldVectorData::Float32(v) => {
                for chunk in v.chunks(PIECE / 4) {
                    piece.clear();
                    for &component in chunk {
                        if self.half {
                            piece.extend_from_slice(&f32_to_f16(component).to_le_bytes());
                        } else {
                            piece.extend_from_slice(&component.to_le_bytes());
                        }
                    }
                    self.append(&piece);
                }
            }
        }
    }

    /// `Lucene99FlatVectorsWriter.writeField`: one field, from an in-memory
    /// batch of vectors.
    pub fn write_field(&mut self, field: &FlatVectorsField) -> Result<()> {
        validate_field(field, self.max_doc)?;
        let encoding = field.values.encoding();
        let vector_data_offset = self.align_output(encoding);
        if self.out.is_some() {
            self.stream_field(&field.values);
        } else {
            match &field.values {
                FieldVectorData::Float32(v) if self.half => {
                    for &component in v {
                        self.data
                            .extend_from_slice(&f32_to_f16(component).to_le_bytes());
                    }
                }
                FieldVectorData::Float32(v) => {
                    // Sized once and filled in place: the loop compiles to a
                    // straight copy on little-endian targets, where pushing
                    // component by component checked capacity every 4 bytes.
                    let start = self.data.len();
                    // ARITH: the components are already in memory, so their
                    // byte count (and the buffer grown by it) fits a usize.
                    #[allow(clippy::arithmetic_side_effects)]
                    self.data.resize(start + v.len() * 4, 0);
                    for (out, component) in self.data[start..].chunks_exact_mut(4).zip(v) {
                        out.copy_from_slice(&component.to_le_bytes());
                    }
                }
                FieldVectorData::Byte(v) => self.data.extend_from_slice(v),
            }
        }
        // ARITH: `align_output` returned the length so far and the file only
        // grows, so this is the byte count just appended.
        #[allow(clippy::arithmetic_side_effects)]
        let vector_data_length = self.data_len() as i64 - vector_data_offset;
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
                if self.out.is_none() {
                    self.data.truncate(rollback_to);
                }
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
        let Some(vector_bytes) =
            (field.dimension as usize).checked_mul(element_bytes(field.encoding, self.half))
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
        // sort and no deletions that is one copy per source,
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
            match &source.values {
                // Stored the other way: decode and re-encode each vector.
                MergeSourceValues::Float32(v) if v.stores_half_floats() != self.half => {
                    let mut scratch = vec![0.0f32; v.dimension()];
                    for ord in slot.ord..run_end {
                        v.vector_into(ord, &mut scratch)?;
                        for &c in &scratch {
                            if self.half {
                                self.append(&f32_to_f16(c).to_le_bytes());
                            } else {
                                self.append(&c.to_le_bytes());
                            }
                        }
                    }
                }
                values => self.append(values.raw_range(slot.ord, run_end)?),
            }
            docs.extend(plan[i..end].iter().map(|s| s.new_doc));
            i = end;
        }

        // ARITH: as in `write_field`, `vector_data_offset` is a length
        // `self.data` has already passed.
        #[allow(clippy::arithmetic_side_effects)]
        let vector_data_length = self.data_len() as i64 - vector_data_offset;
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
        // The footer must not make the buffer grow by doubling: the file can
        // be most of a segment's memory.
        self.data.reserve_exact(codec_util::FOOTER_LENGTH);
        codec_util::write_footer(&mut self.data);
        (self.data, self.meta)
    }

    /// `Lucene99FlatVectorsWriter.alignOutput`: pad to the data alignment
    /// ([`VECTOR_DATA_ALIGNMENT`]) and return the aligned offset.
    fn align_output(&mut self, encoding: VectorEncoding) -> i64 {
        let align = if STOCK_ALIGNMENT.load(std::sync::atomic::Ordering::Relaxed) {
            stock_alignment(encoding, self.half)
        } else {
            VECTOR_DATA_ALIGNMENT
        };
        let padded = self.data_len().next_multiple_of(align);
        let pad = padded.saturating_sub(self.data_len());
        self.data.resize(self.data.len().saturating_add(pad), 0);
        self.data_len() as i64
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
        // `KNN1040HalfFloatFlatVectorsWriter.writeMeta` writes none.
        if !self.half {
            self.meta.write_i32(similarity_ordinal(similarity));
        }
        self.meta.write_vlong(vector_data_offset);
        self.meta.write_vlong(vector_data_length);
        self.meta.write_vint(dimension);
        self.meta.write_i32(docs.len() as i32);
        write_stored_meta(
            &mut self.meta,
            &mut self.data,
            self.spilled as i64,
            docs,
            self.max_doc,
        );
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
        // The branch-free pass first (it vectorizes); the position only when
        // it fails.
        if !values.iter().fold(true, |ok, v| ok & v.is_finite()) {
            if let Some((i, v)) = values.iter().enumerate().find(|(_, v)| !v.is_finite()) {
                return Err(Error::NonFiniteValue(field.field_number, i, *v));
            }
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
    /// `query . query`, for a cosine scorer: the one sum of the three that
    /// does not depend on the candidate.
    query_norm: f32,
    scratch: Vec<f32>,
    /// Whether the `.vec` is read by page ([`lucene_store::is_prefetchable`]):
    /// a bulk score then fetches every candidate's page in one batch before
    /// scoring, rather than faulting them in one at a time.
    paged: bool,
}

/// `query . query` as [`cosine`] sums it.
pub(crate) fn query_norm(query: &[f32]) -> f32 {
    lucene_util::simd::cosine_parts_f32(query, query).1
}

impl crate::hnsw::VectorScorer for FloatVectorScorer<'_> {
    fn read_ahead_span(&self) -> usize {
        self.values.values.read_ahead_span()
    }

    fn read_ahead(&self, from: i32, to: i32) {
        self.values.values.read_ahead(from, to)
    }

    fn score(&mut self, node: i32) -> Result<f32> {
        let candidate = self.values.vector_ref(node, &mut self.scratch)?;
        Ok(if self.similarity == VectorSimilarityFunction::Cosine {
            normalize_to_unit_interval(cosine_with_norm(&self.query, self.query_norm, candidate))
        } else {
            self.similarity.score(&self.query, candidate)
        })
    }

    fn bulk_score(&mut self, nodes: &[i32], scores: &mut [f32]) -> Result<f32> {
        if nodes.len() > 1 {
            self.prefetch(nodes);
        }
        let mut max = f32::NEG_INFINITY;
        for (i, &node) in nodes.iter().enumerate() {
            scores[i] = self.score(node)?;
            max = max.max(scores[i]);
        }
        Ok(max)
    }

    fn max_ord(&self) -> i32 {
        self.values.size()
    }

    /// A `.vec` read by page fetches every candidate's page in one batch
    /// rather than faulting them in one at a time; one in memory loads every
    /// cache line of every candidate before the first is scored (see
    /// `docs/benchmarks/hnsw.md` for the variants measured).
    fn prefetch(&self, nodes: &[i32]) {
        if self.paged {
            // A graph node's neighbours fit one stack batch; more go in
            // several. Only a hint: a page it misses faults in on its own.
            const BATCH: usize = 64;
            let mut ranges: [&[u8]; BATCH] = [&[]; BATCH];
            for chunk in nodes.chunks(BATCH) {
                let mut count = 0usize;
                for &n in chunk {
                    if let (Ok(bytes), Some(slot)) =
                        (self.values.values.bytes(n), ranges.get_mut(count))
                    {
                        *slot = bytes;
                        count = count.saturating_add(1);
                    }
                }
                let _ = lucene_store::prefetch(ranges.get(..count).unwrap_or(&[]));
            }
            return;
        }
        for &n in nodes {
            if let Ok(bytes) = self.values.values.bytes(n) {
                lucene_util::simd::prefetch_bytes(bytes);
            }
        }
    }
}

impl crate::hnsw::UpdateableVectorScorer for FloatVectorScorer<'_> {
    fn set_scoring_ordinal(&mut self, ord: i32) -> Result<()> {
        self.values.vector_into(ord, &mut self.query)?;
        self.query_norm = query_norm(&self.query);
        Ok(())
    }
}

/// Port of `DefaultFlatVectorScorer`'s byte scorer; see
/// [`FloatVectorScorer`]. No scratch buffer: a byte vector needs no decoding,
/// so the candidate is scored straight out of the mapped `.vec` bytes.
#[derive(Debug, Clone)]
pub struct ByteVectorScorer<'a> {
    values: ByteVectorValues<'a>,
    query: Vec<u8>,
}

impl crate::hnsw::VectorScorer for ByteVectorScorer<'_> {
    fn read_ahead_span(&self) -> usize {
        self.values.values.read_ahead_span()
    }

    fn read_ahead(&self, from: i32, to: i32) {
        self.values.values.read_ahead(from, to)
    }

    fn score(&mut self, node: i32) -> Result<f32> {
        let candidate = self.values.vector(node)?;
        Ok(self.values.values.score_bytes(&self.query, candidate))
    }

    fn max_ord(&self) -> i32 {
        self.values.size()
    }

    fn prefetch(&self, nodes: &[i32]) {
        for &n in nodes {
            if let Ok(bytes) = self.values.vector(n) {
                lucene_util::simd::prefetch_bytes(bytes);
            }
        }
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
    /// Every vector, ordinal after ordinal, as one run of components:
    /// borrowed straight from the file when it holds aligned little-endian
    /// `f32`s, decoded otherwise (half floats, a misaligned buffer).
    ///
    /// # Errors
    /// A vector that does not decode.
    pub fn all_vectors(&self) -> Result<std::borrow::Cow<'a, [f32]>> {
        let slice: &'a [u8] = self.values.slice;
        if !self.values.half {
            if let Some(floats) = lucene_util::vector_util::le_f32s(slice) {
                return Ok(std::borrow::Cow::Borrowed(floats));
            }
        }
        let size = usize::try_from(self.values.size).unwrap_or(0);
        let mut out = vec![0.0f32; size.saturating_mul(self.values.dimension)];
        for (ord, chunk) in out
            .chunks_exact_mut(self.values.dimension.max(1))
            .enumerate()
        {
            self.vector_into(ord as i32, chunk)?;
        }
        Ok(std::borrow::Cow::Owned(out))
    }

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
            query_norm: query_norm(query),
            scratch: vec![0.0; self.values.dimension],
            paged: self.is_paged(),
        })
    }

    /// Whether these vectors lie in a file read by page.
    fn is_paged(&self) -> bool {
        self.size() > 0
            && self
                .values
                .bytes(0)
                .is_ok_and(lucene_store::is_prefetchable)
    }

    /// `FlatVectorsScorer.getRandomVectorScorerSupplier(...).scorer()`: a
    /// scorer whose query is set later with `set_scoring_ordinal`.
    pub fn ord_scorer(&self) -> FloatVectorScorer<'a> {
        FloatVectorScorer {
            values: self.clone(),
            similarity: self.values.similarity,
            query: vec![0.0; self.values.dimension],
            query_norm: 0.0,
            scratch: vec![0.0; self.values.dimension],
            paged: self.is_paged(),
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
            query: query.to_vec(),
        })
    }

    /// See [`FloatVectorValues::ord_scorer`].
    pub fn ord_scorer(&self) -> ByteVectorScorer<'a> {
        ByteVectorScorer {
            values: self.clone(),
            query: vec![0; self.values.dimension],
        }
    }
}

#[cfg(test)]
mod tests;
