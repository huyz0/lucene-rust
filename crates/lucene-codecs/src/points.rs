//! Port of `org.apache.lucene.codecs.lucene90.Lucene90PointsFormat` /
//! `org.apache.lucene.util.bkd.BKDReader` (`.kdm` meta + `.kdi` index +
//! `.kdd` data) — block KD-tree point values (used by numeric range/point
//! fields: `IntPoint`, `LongPoint`, `LatLonPoint`, etc.). The write side
//! ([`write_packed`] for a flush, [`write_merged`] for a merge) is
//! `Lucene90PointsWriter` over [`crate::bkd_writer::BkdWriter`], the
//! byte-identical port of `BKDWriter`.
//!
//! - `.kdm`: `IndexHeader`, then per field: `fieldNumber` (i32, terminated
//!   by `-1`) followed by a per-field BKD header (plain `Header`, not
//!   `IndexHeader` -- no id/suffix) and the tree's shape (`numDims`,
//!   `numIndexDims`, `maxPointsInLeafNode`, `bytesPerDim`, `numLeaves`,
//!   `minPackedValue`/`maxPackedValue`, `pointCount`, `docCount`, then the
//!   `.kdi`-relative `indexStartPointer`/`numIndexBytes` this field's packed
//!   tree occupies), then `indexLength`/`dataLength`, then `Footer`.
//! - `.kdi`: `IndexHeader`, then each field's **packed index** back to back
//!   (a compact binary-tree encoding of split dimensions/values and leaf
//!   file-pointer deltas -- see [`decode_leaf_pointers`]), then `Footer`.
//! - `.kdd`: `IndexHeader`, then every field's **leaf blocks** back to back
//!   (each independently seekable via the leaf pointers recovered from
//!   `.kdi`), then `Footer`.
//!
//! This port only supports the version real Lucene 10.5.0 always writes
//! (`BKDWriter.VERSION_CURRENT` = 10, vectorized BPV24 + BPV21) -- older
//! on-disk versions (balanced legacy trees, non-vectorized bpv24, no
//! low-cardinality leaves) are rejected outright rather than replicated,
//! same stance as elsewhere in this port (only the current format is a
//! real write target).
//!
//! **Decode-fully, not lazy tree navigation**: Java's `BKDReader` walks the
//! packed index with a query's bounding box to prune whole subtrees
//! (`IntersectVisitor.compare`), seeking past whichever half doesn't
//! matter. This port has no query-pruning phase yet, so
//! [`decode_leaf_pointers`] always visits **every** node in strict
//! left-to-right order and never seeks: the packed index's `leftNumBytes`
//! field (which exists so a query can skip the entire left subtree without
//! parsing it) is read and discarded, and reading through the left
//! subtree's bytes recursively naturally lands the cursor exactly where
//! the right subtree begins -- the same trade-off already made for
//! `IndexedDISI`, stored fields, and the terms dictionary.

use lucene_store::codec_util;
use lucene_store::data_input::{DataInput, SliceInput};
use lucene_store::data_output::DataOutput;

use crate::bkd_writer;

/// Default `BKDConfig`/`Lucene90PointsWriter` leaf size.
pub const DEFAULT_MAX_POINTS_IN_LEAF_NODE: i32 = 512;

const DATA_CODEC_NAME: &str = "Lucene90PointsFormatData";
const INDEX_CODEC_NAME: &str = "Lucene90PointsFormatIndex";
const META_CODEC_NAME: &str = "Lucene90PointsFormatMeta";
const VERSION_START: i32 = 0;
const VERSION_CURRENT: i32 = 1;

const BKD_CODEC_NAME: &str = "BKD";
/// The only BKD version this port understands -- current Lucene always
/// writes this one (vectorized BPV24, BPV21 introduced).
const BKD_VERSION_CURRENT: i32 = 10;
/// `BKDWriter.VERSION_META_FILE`: the oldest BKD version this reader opens,
/// the one `Lucene90PointsFormat` version 0 always wrote. Older versions
/// (balanced trees, index-file-resident metadata) only ever came from the
/// Lucene 8 `Lucene60`/`Lucene86` points formats, which neither Lucene 10
/// nor OpenSearch 3.x reads.
const BKD_VERSION_META_FILE: i32 = 9;
/// `BKDWriter.VERSION_VECTORIZE_BPV24_AND_INTRODUCE_BPV21`.
const BKD_VERSION_VECTORIZE_BPV24: i32 = 10;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Store(#[from] lucene_store::Error),
    /// The BKD writer refused a configuration or failed on a temp file.
    #[error(transparent)]
    Bkd(#[from] crate::bkd_writer::BkdError),
    #[error("illegal field number: {0}")]
    IllegalFieldNumber(i32),
    #[error("unsupported doc-ids bits-per-value byte: {0}")]
    UnsupportedDocIdsEncoding(i8),
    #[error("unsupported compressed dimension marker: {0}")]
    UnsupportedCompressedDim(i8),
    #[error("sub-blocks do not add up to the expected count: {expected} != {actual}")]
    SubBlockCountMismatch { expected: usize, actual: usize },
    #[error("field {field_number}: write() requires at least one point (empty fields aren't supported by this write path)")]
    EmptyField { field_number: i32 },
    #[error("field {field_number}: point {index} has packed_value.len() == {actual}, expected bytes_per_dim == {expected}")]
    WrongPackedValueLength {
        field_number: i32,
        index: usize,
        expected: i32,
        actual: usize,
    },
    /// A [`PackedPointsField`] whose `values` is not exactly `docs.len()`
    /// packed values of `num_dims * bytes_per_dim` bytes.
    #[error("field {field_number}: {points} points of {stride} bytes need {points}x{stride} value bytes, got {actual}")]
    PackedValuesLength {
        field_number: i32,
        points: usize,
        stride: usize,
        actual: usize,
    },
    #[error("field {field_number}: num_index_dims ({num_index_dims}) must be between 1 and num_dims ({num_dims}) inclusive")]
    InvalidNumIndexDims {
        field_number: i32,
        num_dims: i32,
        num_index_dims: i32,
    },
    /// Port of `BKDConfig`'s constructor bounds checks (`numDims must be
    /// 1 .. 16`, `numIndexDims must be 1 .. 8`, `numIndexDims cannot exceed
    /// numDims`, `bytesPerDim must be > 0`, `maxPointsInLeafNode must be >
    /// 0`), plus `BKDReader`'s own `assert numLeaves > 0`. Java raises
    /// `IllegalArgumentException` from `BKDConfig.of(...)` while reading
    /// `.kdm`; here it is one error variant carrying the offending value.
    #[error("invalid BKD config: {0}")]
    InvalidConfig(String),
    /// Port of `BKDReader`'s
    /// `"minPackedValue ... is > maxPackedValue ... for dim=N"`
    /// `CorruptIndexException`.
    #[error("minPackedValue is > maxPackedValue for dim={0}")]
    MinGreaterThanMax(usize),
}

pub type Result<T> = std::result::Result<T, Error>;

/// A `CorruptIndexException`-shaped failure: something the `.kdm`/`.kdi`/`.kdd`
/// bytes claim that no writer could have produced. Java raises
/// `CorruptIndexException`; here it rides in on `lucene_store`'s own
/// `Corrupted` so callers that already match on it keep working.
fn corrupt(message: String) -> Error {
    Error::Store(lucene_store::Error::Corrupted(message))
}

/// `BKDConfig.MAX_DIMS` -- the most data dimensions a BKD tree may carry.
pub const MAX_DIMS: i32 = 16;
/// `BKDConfig.MAX_INDEX_DIMS` -- the most dimensions that may take part in
/// the tree's split structure.
pub const MAX_INDEX_DIMS: i32 = 8;

/// `ArrayUtil.MAX_ARRAY_LENGTH`, the ceiling `BKDConfig`'s constructor puts on
/// `maxPointsInLeafNode`. Java's value is `Integer.MAX_VALUE -
/// RamUsageEstimator.NUM_BYTES_ARRAY_HEADER` (= `i32::MAX - 16`), a JVM
/// array-header allowance; the number is reproduced rather than reinterpreted
/// so this port rejects exactly the values Java rejects.
pub const MAX_POINTS_IN_LEAF_NODE: i32 = i32::MAX - 16;

/// `PointValues.MAX_NUM_BYTES`, the ceiling `FieldInfo`'s constructor and
/// `FieldType.setDimensions` put on a point field's `bytesPerDim`. It is a
/// **write-side** bound: `BKDConfig` itself never checks it, so a `.kdm` Java
/// would happily read can in principle carry more, and the read path here does
/// not enforce it either (see [`check_config`]). What it does establish is that
/// every value this port *writes* has a `bytesPerDim` a `FieldInfo` would
/// accept -- the same guarantee real `BKDWriter` gets for free by only ever
/// being handed a `FieldInfo`-validated width.
pub const MAX_NUM_BYTES: i32 = 16;

/// Port of `BKDConfig`'s canonical constructor validation, shared by the
/// read side (`.kdm` per-field header, where Java calls `BKDConfig.of` and
/// lets its `IllegalArgumentException` escape) and the write side (where
/// Java's `BKDWriter` constructs the same config up front). Without this a
/// corrupt/hostile `.kdm` reaches `vec![0u8; (num_index_dims *
/// bytes_per_dim) as usize]` with an attacker-chosen (possibly negative,
/// hence huge-when-cast) length, and `write` reaches
/// `count.div_ceil(max_points_in_leaf_node)` with a zero divisor.
///
/// All five of `BKDConfig`'s constructor guards are reproduced, plus one Java
/// does not have: `numDims * bytesPerDim` overflowing. Java lets that wrap and
/// surface as `NegativeArraySizeException` from `new byte[...]`; in Rust it is
/// a panic, which is not an outcome a caller can handle. See
/// `docs/arithmetic-gate.md`.
fn check_config(
    num_dims: i32,
    num_index_dims: i32,
    bytes_per_dim: i32,
    max_points_in_leaf_node: i32,
) -> Result<()> {
    if !(1..=MAX_DIMS).contains(&num_dims) {
        return Err(Error::InvalidConfig(format!(
            "numDims must be 1 .. {MAX_DIMS} (got: {num_dims})"
        )));
    }
    if !(1..=MAX_INDEX_DIMS).contains(&num_index_dims) {
        return Err(Error::InvalidConfig(format!(
            "numIndexDims must be 1 .. {MAX_INDEX_DIMS} (got: {num_index_dims})"
        )));
    }
    if num_index_dims > num_dims {
        return Err(Error::InvalidConfig(format!(
            "numIndexDims cannot exceed numDims ({num_dims}) (got: {num_index_dims})"
        )));
    }
    if bytes_per_dim <= 0 {
        return Err(Error::InvalidConfig(format!(
            "bytesPerDim must be > 0; got {bytes_per_dim}"
        )));
    }
    // Java stops at `bytesPerDim > 0` and lets `numDims * bytesPerDim`
    // silently wrap an `int`, which surfaces one step later as
    // `NegativeArraySizeException` from `new byte[...]` -- a caught,
    // reportable corruption. In Rust the same product *panics* on overflow in
    // a debug build (and a large-but-not-overflowing one is an aborting
    // allocation), neither of which a caller can catch: a `.kdm` claiming
    // `numDims=8, bytesPerDim=2^30` is enough. Rejecting the product here
    // reproduces Java's outcome without Java's mechanism.
    if num_dims.checked_mul(bytes_per_dim).is_none() {
        return Err(Error::InvalidConfig(format!(
            "numDims ({num_dims}) x bytesPerDim ({bytes_per_dim}) overflows"
        )));
    }
    if max_points_in_leaf_node <= 0 {
        return Err(Error::InvalidConfig(format!(
            "maxPointsInLeafNode must be > 0; got {max_points_in_leaf_node}"
        )));
    }
    // `BKDConfig`'s last guard, `maxPointsInLeafNode > ArrayUtil.MAX_ARRAY_LENGTH`,
    // reproduced with Java's exact ceiling so this port rejects exactly the
    // values Java rejects -- 16 of them.
    if max_points_in_leaf_node > MAX_POINTS_IN_LEAF_NODE {
        return Err(Error::InvalidConfig(format!(
            "maxPointsInLeafNode must be <= {MAX_POINTS_IN_LEAF_NODE}; got \
             {max_points_in_leaf_node}"
        )));
    }
    Ok(())
}

/// One field's BKD tree shape and root-level bounds, plus enough to locate
/// its packed index slice in `.kdi` and walk its leaves in `.kdd`.
#[derive(Debug, Clone)]
pub struct PointsField {
    pub num_dims: i32,
    pub num_index_dims: i32,
    pub bytes_per_dim: i32,
    pub max_points_in_leaf_node: i32,
    pub num_leaves: i32,
    pub min_packed_value: Vec<u8>,
    pub max_packed_value: Vec<u8>,
    pub point_count: i64,
    pub doc_count: i32,
    index_start_pointer: i64,
    num_index_bytes: i32,
    /// The field's `BKDWriter` version, from its `.kdm` header: 9
    /// (`VERSION_META_FILE`, what `Lucene90PointsFormat` version 0 wrote --
    /// every Lucene 9.x and 10.0/10.1 index) or 10
    /// (`VERSION_VECTORIZE_BPV24_AND_INTRODUCE_BPV21`). The only difference
    /// a reader sees is the `BPV_24` doc-id layout -- see [`read_doc_ids_into`].
    bkd_version: i32,
}

impl PointsField {
    /// ARITH: every `PointsField` is built by [`read_field_meta`], which runs
    /// [`check_config`] first; that rejects the field unless `num_dims >= 1`,
    /// `bytes_per_dim >= 1` and `num_dims.checked_mul(bytes_per_dim)` is
    /// `Some`. So the product is in `1..=i32::MAX` and the widening cast is
    /// exact. The `debug_assert` re-states the `check_config` postcondition so
    /// `cargo test` exercises it on every leaf decode rather than trusting the
    /// prose.
    #[allow(clippy::arithmetic_side_effects)]
    fn packed_bytes_length(&self) -> usize {
        debug_assert!(self.num_dims.checked_mul(self.bytes_per_dim).is_some());
        (self.num_dims * self.bytes_per_dim) as usize
    }

    /// `num_index_dims * bytes_per_dim`, the length of every cell-bounds and
    /// split-value buffer.
    ///
    /// ARITH: `check_config` establishes `num_index_dims <= num_dims`, so this
    /// product is bounded by [`packed_bytes_length`](Self::packed_bytes_length)'s.
    #[allow(clippy::arithmetic_side_effects)]
    fn packed_index_bytes_length(&self) -> usize {
        debug_assert!(self.num_index_dims <= self.num_dims);
        (self.num_index_dims * self.bytes_per_dim) as usize
    }

    /// `BKDReader.BKDPointTree.size()`: how many points live under the
    /// subtree rooted at `node_id` -- derived from the node id and the
    /// field's own counts, with no `.kdi` or `.kdd` read at all. That is what
    /// makes [`PointsReader::estimate_point_count`] cheap.
    ///
    /// Only Java's **unbalanced** arm is ported, and that is complete rather
    /// than partial: `BKDReader.isTreeBalanced()` returns `false` outright for
    /// `version >= VERSION_META_FILE`, and this module accepts BKD version 10
    /// alone ([`BKD_VERSION_CURRENT`]), far past it. A balanced tree is a
    /// pre-8.6 index this reader rejects at `open`.
    ///
    /// Java's `numLeaves == 1` special case needs no counterpart either: it
    /// exists only to force `isTreeBalanced` false for a one-leaf tree, which
    /// is already the only arm here.
    ///
    /// ARITH: `num_leaves >= 1` and `max_points_in_leaf_node >= 1` (both
    /// checked at [`read_field_meta`]), and `node_id >= 1` on every call --
    /// [`PointsReader::estimate_point_count`] starts at the root and only ever
    /// descends through [`child_ids`], which errors before a node id can
    /// reach zero or wrap. Both doubling loops therefore terminate with
    /// `left`/`right` under `2 * num_leaves`, which is why the whole
    /// computation is done in `i64` where Java uses `int`: the intermediate
    /// values cannot overflow, so the only checked step left is the final
    /// point-count multiplication, which multiplies two `.kdm`-supplied
    /// numbers.
    #[allow(clippy::arithmetic_side_effects)]
    fn subtree_size(&self, node_id: i32) -> Result<i64> {
        let num_leaves = i64::from(self.num_leaves);
        let max_points = i64::from(self.max_points_in_leaf_node);
        debug_assert!(node_id >= 1 && num_leaves >= 1 && max_points >= 1);

        let mut left_most_leaf_node = i64::from(node_id);
        while left_most_leaf_node < num_leaves {
            left_most_leaf_node *= 2;
        }
        let mut right_most_leaf_node = i64::from(node_id);
        while right_most_leaf_node < num_leaves {
            right_most_leaf_node = right_most_leaf_node * 2 + 1;
        }
        let leaves = if right_most_leaf_node >= left_most_leaf_node {
            // Both are on the same level.
            right_most_leaf_node - left_most_leaf_node + 1
        } else {
            // Left is one level deeper than right.
            right_most_leaf_node - left_most_leaf_node + 1 + num_leaves
        };

        // `treeDepth = MathUtil.log(numLeaves, 2) + 2`, then
        // `rightMostLeafNode = (1 << treeDepth - 1) - 1`.
        let log2 = 63 - (num_leaves as u64).leading_zeros() as i64;
        let tree_right_most_leaf_node = (1i64 << (log2 + 1)) - 1;
        // `pointCount % maxPointsInLeafNode`, with Java's `== 0` promotion to
        // a full leaf. `rem_euclid` rather than `%` so a negative `pointCount`
        // in a corrupt `.kdm` cannot produce a negative subtree size.
        let last_leaf_node_point_count = match self.point_count.rem_euclid(max_points) {
            0 => max_points,
            n => n,
        };

        // `numLeaves` and `maxPointsInLeafNode` are both `.kdm` vints bounded
        // only by `i32::MAX`, so a corrupt pair can put this product a hair
        // past `i64`. Java lets it wrap and hands a *negative* cost to the
        // query planner; saturating keeps the one property a cost estimate
        // has to have -- monotonicity -- without an error path no legal file
        // can reach.
        Ok(if right_most_leaf_node == tree_right_most_leaf_node {
            (leaves - 1)
                .saturating_mul(max_points)
                .saturating_add(last_leaf_node_point_count)
        } else {
            leaves.saturating_mul(max_points)
        })
    }
}

/// One decoded point: its owning document id and its full packed value
/// (`num_dims * bytes_per_dim` bytes, big-endian-per-dimension unsigned
/// magnitude -- the same encoding `NumericUtils.intToSortableBytes`/
/// `longToSortableBytes` produce, unchanged here).
#[derive(Debug, Clone, PartialEq)]
pub struct Point {
    pub doc_id: i32,
    pub packed_value: Vec<u8>,
}

/// One decoded leaf block: every point it contains, plus (when present on
/// disk) its own per-leaf bounding box over the index dimensions --
/// `num_index_dims * bytes_per_dim` bytes each of `min_bound`/`max_bound`.
/// [`read_leaf_block`] only writes/reads this box when `num_index_dims !=
/// 1` (see that function's doc comment), so `bound` is `None` for
/// single-index-dimension fields -- there is no independent on-disk value
/// to cross-check a point against in that case beyond the field-wide
/// `min_packed_value`/`max_packed_value` in `.kdm`.
#[derive(Debug, Clone)]
pub struct Leaf {
    pub points: Vec<Point>,
    pub bound: Option<(Vec<u8>, Vec<u8>)>,
}

pub struct PointsReader<'d> {
    kdi: &'d [u8],
    kdd: &'d [u8],
    fields: std::borrow::Cow<'d, [(i32, PointsField)]>,
}

impl PointsReader<'static> {
    /// A segment that indexes no points: every field is absent, as
    /// `PointValues` is `null` for each of them in Lucene.
    pub fn empty() -> Self {
        PointsReader {
            kdi: &[],
            kdd: &[],
            fields: std::borrow::Cow::Borrowed(&[]),
        }
    }
}

impl<'d> PointsReader<'d> {
    /// A reader over `kdi`/`kdd` with the fields [`open_meta`] parsed from
    /// the same segment, borrowed: what a reader that keeps the parsed
    /// metadata (Lucene's `BKDReader`s live as long as the segment) builds
    /// per search instead of re-parsing `.kdm`.
    pub fn with_meta(kdi: &'d [u8], kdd: &'d [u8], fields: &'d [(i32, PointsField)]) -> Self {
        PointsReader {
            kdi,
            kdd,
            fields: std::borrow::Cow::Borrowed(fields),
        }
    }
}

/// Parses `.kdm`+`.kdi`+`.kdd` (already read into memory).
pub fn open<'d>(
    kdm: &[u8],
    kdi: &'d [u8],
    kdd: &'d [u8],
    segment_id: &[u8; codec_util::ID_LENGTH],
    segment_suffix: &str,
) -> Result<PointsReader<'d>> {
    let fields = open_meta(kdm, kdi, kdd, segment_id, segment_suffix)?;
    Ok(PointsReader {
        kdi,
        kdd,
        fields: std::borrow::Cow::Owned(fields),
    })
}

/// [`open`]'s checks and `.kdm` parse, returning the fields' metadata alone
/// (owned), for [`PointsReader::with_meta`].
pub fn open_meta(
    kdm: &[u8],
    kdi: &[u8],
    kdd: &[u8],
    segment_id: &[u8; codec_util::ID_LENGTH],
    segment_suffix: &str,
) -> Result<Vec<(i32, PointsField)>> {
    let mut kdi_input = SliceInput::new(kdi);
    codec_util::check_index_header(
        &mut kdi_input,
        INDEX_CODEC_NAME,
        VERSION_START,
        VERSION_CURRENT,
        segment_id,
        segment_suffix,
    )?;
    codec_util::retrieve_checksum(kdi)?;

    let mut kdd_input = SliceInput::new(kdd);
    codec_util::check_index_header(
        &mut kdd_input,
        DATA_CODEC_NAME,
        VERSION_START,
        VERSION_CURRENT,
        segment_id,
        segment_suffix,
    )?;
    codec_util::retrieve_checksum(kdd)?;

    let mut meta_input = SliceInput::new(kdm);
    codec_util::check_index_header(
        &mut meta_input,
        META_CODEC_NAME,
        VERSION_START,
        VERSION_CURRENT,
        segment_id,
        segment_suffix,
    )?;

    let mut fields = Vec::new();
    loop {
        let field_number = meta_input.read_i32()?;
        if field_number == -1 {
            break;
        }
        if field_number < 0 {
            return Err(Error::IllegalFieldNumber(field_number));
        }
        fields.push((field_number, read_field_meta(&mut meta_input)?));
    }
    let _index_length = meta_input.read_i64()?;
    let _data_length = meta_input.read_i64()?;
    codec_util::check_footer(&mut meta_input, kdm.len())?;

    Ok(fields)
}

fn read_field_meta(meta_input: &mut SliceInput) -> Result<PointsField> {
    // `check_header` enforces the exact version (min == max == CURRENT)
    // itself, surfacing a mismatch as a `Corrupted` error -- no separate
    // check needed here.
    let bkd_version = codec_util::check_header(
        meta_input,
        BKD_CODEC_NAME,
        BKD_VERSION_META_FILE,
        BKD_VERSION_CURRENT,
    )?
    .version;

    let num_dims = meta_input.read_vint()?;
    let num_index_dims = meta_input.read_vint()?;
    let max_points_in_leaf_node = meta_input.read_vint()?;
    let bytes_per_dim = meta_input.read_vint()?;
    // `BKDReader`'s constructor funnels these four through `BKDConfig.of`,
    // which throws on out-of-range values; do the same before any of them is
    // used as an allocation length or a divisor.
    check_config(
        num_dims,
        num_index_dims,
        bytes_per_dim,
        max_points_in_leaf_node,
    )?;
    let num_leaves = meta_input.read_vint()?;
    if num_leaves <= 0 {
        return Err(Error::InvalidConfig(format!(
            "numLeaves must be > 0; got {num_leaves}"
        )));
    }

    // ARITH: `check_config` proved `num_dims * bytes_per_dim` fits an `i32`
    // and `num_index_dims <= num_dims`, so this product is bounded by that one
    // and lands in `1..=i32::MAX`.
    #[allow(clippy::arithmetic_side_effects)]
    let packed_index_bytes_length = (num_index_dims * bytes_per_dim) as usize;
    // Two allocations sized by a value read off disk. `bytesPerDim` has no
    // upper bound in Lucene either, so `numIndexDims=1, bytesPerDim=2^30` is
    // a legal-looking `.kdm` that asks for two 1 GB buffers -- in Java an
    // `OutOfMemoryError` the caller can catch, here an **abort** no
    // `catch_unwind` at the FFI boundary can. The bytes have to be in the
    // `.kdm` for the read to succeed anyway, so requiring them up front
    // costs nothing and turns the abort into a decode error.
    if meta_input.remaining() < packed_index_bytes_length.saturating_mul(2) {
        return Err(Error::InvalidConfig(format!(
            "numIndexDims ({num_index_dims}) x bytesPerDim ({bytes_per_dim}) needs \
             {packed_index_bytes_length} bytes of min and max packed value but only {} \
             bytes of .kdm remain",
            meta_input.remaining()
        )));
    }
    let mut min_packed_value = vec![0u8; packed_index_bytes_length];
    let mut max_packed_value = vec![0u8; packed_index_bytes_length];
    meta_input.read_bytes(&mut min_packed_value)?;
    meta_input.read_bytes(&mut max_packed_value)?;
    // `BKDReader`: "minPackedValue ... is > maxPackedValue ... for dim=N".
    // Compared per dimension (unsigned byte-wise), not as one whole value.
    let bpd = bytes_per_dim as usize;
    for dim in 0..num_index_dims as usize {
        // ARITH: `dim < num_index_dims`, so `(dim + 1) * bpd <=
        // num_index_dims * bytes_per_dim = packed_index_bytes_length`, which
        // `check_config` already proved fits an `i32`. Both buffers are
        // exactly that long.
        #[allow(clippy::arithmetic_side_effects)]
        let (lo, hi) = (dim * bpd, (dim + 1) * bpd);
        if min_packed_value[lo..hi] > max_packed_value[lo..hi] {
            return Err(Error::MinGreaterThanMax(dim));
        }
    }

    let point_count = meta_input.read_vlong()?;
    let doc_count = meta_input.read_vint()?;
    let num_index_bytes = meta_input.read_vint()?;
    let _min_leaf_block_fp = meta_input.read_i64()?;
    let index_start_pointer = meta_input.read_i64()?;

    Ok(PointsField {
        num_dims,
        num_index_dims,
        bytes_per_dim,
        max_points_in_leaf_node,
        num_leaves,
        min_packed_value,
        max_packed_value,
        point_count,
        doc_count,
        index_start_pointer,
        num_index_bytes,
        bkd_version,
    })
}

impl<'d> PointsReader<'d> {
    /// The `.kdi` byte range holding `field`'s packed index.
    ///
    /// Both bounds come straight off the `.kdm`: `indexStartPointer` is a raw
    /// `i64` and `numIndexBytes` a vint, neither bounded by anything Java
    /// checks (`BKDReader` just seeks and lets the read fail). Adding them as
    /// `i64` overflows for a hostile pair -- a panic in a debug build, and in a
    /// release build a wrap to a *plausible in-range* end offset that hands the
    /// tree walker somebody else's field's bytes. Both conversions and the sum
    /// are therefore checked, and the range is resolved against `.kdi`'s actual
    /// length exactly once, before the walk.
    fn inner_nodes(&self, field: &PointsField) -> Result<&'d [u8]> {
        let range = usize::try_from(field.index_start_pointer)
            .ok()
            .zip(usize::try_from(field.num_index_bytes).ok())
            .and_then(|(start, len)| Some(start..start.checked_add(len)?));
        let Some(range) = range else {
            return Err(corrupt(format!(
                "field's packed index is out of range: indexStartPointer={}, numIndexBytes={}",
                field.index_start_pointer, field.num_index_bytes
            )));
        };
        self.kdi
            .get(range)
            .ok_or_else(|| lucene_store::Error::Eof { offset: 0 }.into())
    }

    /// The field numbers this segment's `.kdm` has an entry for, in file
    /// order.
    pub fn field_numbers(&self) -> impl Iterator<Item = i32> + '_ {
        self.fields.iter().map(|(n, _)| *n)
    }

    pub fn field(&self, field_number: i32) -> Option<&PointsField> {
        self.fields
            .iter()
            .find(|(n, _)| *n == field_number)
            .map(|(_, f)| f)
    }

    /// Decodes every point (doc id + full packed value) for `field_number`,
    /// across all its leaves, in leaf (left-to-right) order.
    pub fn decode_all_points(&self, field_number: i32) -> Result<Vec<Point>> {
        Ok(self
            .decode_leaves(field_number)?
            .into_iter()
            .flat_map(|leaf| leaf.points)
            .collect())
    }

    /// Decodes every leaf block for `field_number` individually (in
    /// left-to-right order), keeping each leaf's own points and (when
    /// present) its own bounding box separate -- the structural-invariant
    /// checker (`lucene_index::check_index`) needs per-leaf boundaries that
    /// [`decode_all_points`](Self::decode_all_points)'s flattened view discards.
    pub fn decode_leaves(&self, field_number: i32) -> Result<Vec<Leaf>> {
        let field = self
            .field(field_number)
            .ok_or(Error::IllegalFieldNumber(field_number))?;

        let inner_nodes = self.inner_nodes(field)?;
        let leaf_fps = decode_leaf_pointers(inner_nodes, field)?;

        let mut leaves = Vec::with_capacity(leaf_fps.len());
        let mut kdd_input = SliceInput::new(self.kdd);
        for &fp in &leaf_fps {
            seek_leaf_block(&mut kdd_input, fp)?;
            let mut points = Vec::new();
            let bound = read_leaf_block(&mut kdd_input, field, &mut points)?;
            leaves.push(Leaf { points, bound });
        }
        Ok(leaves)
    }

    /// Port of `PointValues.intersect(IntersectVisitor)` on top of
    /// `BKDReader.BKDPointTree` -- the *pruning* traversal, as opposed to
    /// [`decode_all_points`](Self::decode_all_points)'s
    /// decode-every-leaf scan.
    ///
    /// Reproduces Java's exact three-way dispatch on
    /// [`IntersectVisitor::compare`]:
    /// - [`Relation::CellOutsideQuery`] -- the whole subtree is skipped
    ///   without reading a single one of its `.kdd` bytes (and, for a left
    ///   child, without parsing its packed-index bytes either: the
    ///   `leftNumBytes` skip-ahead vint this port previously read and threw
    ///   away is what makes the seek to the right sibling possible, exactly
    ///   as `BKDPointTree.pushRight`'s `rightNodePositions[level]` does).
    /// - [`Relation::CellInsideQuery`] -- only the leaf's doc-id block is
    ///   decoded (`visitDocIDs`/`addAll`); the packed values are never
    ///   touched, so [`IntersectVisitor::visit`] is called instead of
    ///   [`IntersectVisitor::visit_with_value`].
    /// - [`Relation::CellCrossesQuery`] -- descend; at a leaf, decode the
    ///   full block and hand every point to
    ///   [`IntersectVisitor::visit_with_value`] for per-point filtering
    ///   (`visitDocValues`).
    ///
    /// The cell bounds handed to `compare` are maintained the way
    /// `pushBoundsLeft`/`pushBoundsRight`/`popBounds` maintain them: start
    /// from the field-wide `minPackedValue`/`maxPackedValue`, and at each
    /// inner node replace exactly one dimension's `bytes_per_dim` slice with
    /// the node's reconstructed split value (the max for the left child, the
    /// min for the right), restoring it on the way back up. The split value
    /// itself is reconstructed from the packed index's prefix/first-diff-byte
    /// coding against the last split value seen *in that dimension* along the
    /// current root path, with `negative_deltas` tracking left-vs-right
    /// exactly like `BKDReader.readNodeData` -- i.e. this is the read-side
    /// inverse of `BKDWriter.recursePackIndex`, and the first thing in this module that
    /// actually *uses* the reconstructed split values rather than skipping
    /// past them.
    ///
    /// Both bounds slices passed to `compare` are `num_index_dims *
    /// bytes_per_dim` long (the non-indexed trailing data dimensions have no
    /// cell bounds, matching Java).
    pub fn intersect<V: IntersectVisitor>(&self, field_number: i32, visitor: &mut V) -> Result<()> {
        self.intersect_with_scratch(field_number, visitor, true)
    }

    /// [`intersect`](Self::intersect) with the traversal scratch's reuse
    /// switchable -- see [`IntersectCtx::reuse_scratch`]. Production always
    /// passes `true`; `false` is the pre-`c43-final-cleanup` allocation shape,
    /// kept only so `examples/bkd_walk_scratch.rs` can time both arms in one
    /// process, and it must produce the **same answer**, which
    /// `the_two_scratch_arms_visit_exactly_the_same_documents` asserts.
    #[doc(hidden)]
    pub fn intersect_with_scratch<V: IntersectVisitor>(
        &self,
        field_number: i32,
        visitor: &mut V,
        reuse_scratch: bool,
    ) -> Result<()> {
        let field = self
            .field(field_number)
            .ok_or(Error::IllegalFieldNumber(field_number))?;
        let inner_nodes = self.inner_nodes(field)?;
        let mut input = SliceInput::new(inner_nodes);
        let mut ctx = IntersectCtx::new(field, self.kdd, reuse_scratch);
        // The root's leading FP-delta vlong, same as `decode_leaf_pointers`.
        let root_fp = input.read_vlong()?;
        intersect_node(&mut input, 1, root_fp, &mut ctx, visitor, 0)
    }

    /// `PointValues.estimatePointCount(IntersectVisitor)`: how many points
    /// [`intersect`](Self::intersect) *would* visit, without visiting them.
    ///
    /// The same descent as `intersect`, with the two expensive steps removed:
    /// a cell entirely inside the query contributes its whole subtree's
    /// `BKDPointTree.size()` (derived from the node id -- no `.kdd` read, no
    /// `visitDocIDs`), a cell entirely outside contributes nothing, and a leaf
    /// the walk cannot descend past is assumed half-matched, exactly as Java's
    /// `(size() + 1) / 2` does. No leaf block is decoded on any path, so the
    /// cost is bounded by the number of *inner* nodes whose cell crosses the
    /// query boundary.
    ///
    /// Feed the result to `lucene_search::points_query::estimate_doc_count`
    /// (Java's `estimateDocCount`, which is `estimatePointCount` plus the
    /// points-per-document correction) to get the cost an
    /// `IndexOrDocValuesQuery` planner wants.
    ///
    /// The visitor's [`visit`](IntersectVisitor::visit) and
    /// [`visit_with_value`](IntersectVisitor::visit_with_value) are never
    /// called -- Java's estimate uses only `compare`.
    pub fn estimate_point_count<V: IntersectVisitor>(
        &self,
        field_number: i32,
        visitor: &mut V,
    ) -> Result<i64> {
        self.estimate_point_count_bounded(field_number, visitor, i64::MAX)
    }

    /// `PointValues.isEstimatedPointCountGreaterThanOrEqualTo(visitor, tree,
    /// upperBound)`'s engine: [`estimate_point_count`](Self::estimate_point_count)
    /// that stops descending as soon as the running estimate reaches
    /// `upper_bound`.
    ///
    /// The answer is therefore only meaningful as "did it reach the bound":
    /// Java's own caller compares `>= upperBound` and discards the number.
    pub fn estimate_point_count_bounded<V: IntersectVisitor>(
        &self,
        field_number: i32,
        visitor: &mut V,
        upper_bound: i64,
    ) -> Result<i64> {
        self.estimate_point_count_with_scratch(field_number, visitor, upper_bound, true)
    }

    /// [`estimate_point_count_bounded`](Self::estimate_point_count_bounded)
    /// with the traversal scratch's reuse switchable -- see
    /// [`intersect_with_scratch`](Self::intersect_with_scratch).
    #[doc(hidden)]
    pub fn estimate_point_count_with_scratch<V: IntersectVisitor>(
        &self,
        field_number: i32,
        visitor: &mut V,
        upper_bound: i64,
        reuse_scratch: bool,
    ) -> Result<i64> {
        let field = self
            .field(field_number)
            .ok_or(Error::IllegalFieldNumber(field_number))?;
        let inner_nodes = self.inner_nodes(field)?;
        let mut input = SliceInput::new(inner_nodes);
        let mut ctx = IntersectCtx::new(field, self.kdd, reuse_scratch);
        let root_fp = input.read_vlong()?;
        estimate_node(&mut input, 1, root_fp, &mut ctx, visitor, upper_bound, 0)
    }

    /// [`intersect`](Self::intersect) walking with `scratch`'s buffers, so a
    /// caller that intersects one tree many times -- `NumericComparator`'s
    /// competitive iterator re-intersects on every new bottom -- allocates
    /// them once rather than once per walk.
    pub fn intersect_in<V: IntersectVisitor>(
        &self,
        field_number: i32,
        visitor: &mut V,
        scratch: &mut PointsScratch,
    ) -> Result<()> {
        let field = self
            .field(field_number)
            .ok_or(Error::IllegalFieldNumber(field_number))?;
        let inner_nodes = self.inner_nodes(field)?;
        let mut input = SliceInput::new(inner_nodes);
        let mut ctx = IntersectCtx::from_scratch(field, self.kdd, scratch);
        let root_fp = input.read_vlong();
        let out = match root_fp {
            Ok(fp) => intersect_node(&mut input, 1, fp, &mut ctx, visitor, 0),
            Err(e) => Err(e.into()),
        };
        ctx.into_scratch(scratch);
        out
    }

    /// `PointRangeQuery`'s `Weight.count` walk (its private `pointCount`),
    /// exact: a cell entirely inside the query adds its point count
    /// (`BKDPointTree.size()`) without reading a leaf, a cell outside adds
    /// nothing, and a leaf the query crosses is decoded and handed to the
    /// visitor value by value ([`IntersectVisitor::visit_with_value`]), as
    /// [`intersect`](Self::intersect) does. Returns the inside cells' points;
    /// the visitor counts what it accepted of the crossing leaves.
    pub fn count_points<V: IntersectVisitor>(
        &self,
        field_number: i32,
        visitor: &mut V,
    ) -> Result<i64> {
        let field = self
            .field(field_number)
            .ok_or(Error::IllegalFieldNumber(field_number))?;
        let inner_nodes = self.inner_nodes(field)?;
        let mut input = SliceInput::new(inner_nodes);
        let mut ctx = IntersectCtx::new(field, self.kdd, true);
        let root_fp = input.read_vlong()?;
        count_node(&mut input, 1, root_fp, &mut ctx, visitor, 0)
    }

    /// [`estimate_point_count_bounded`](Self::estimate_point_count_bounded)
    /// walking with `scratch`'s buffers -- see [`intersect_in`](Self::intersect_in).
    pub fn estimate_point_count_bounded_in<V: IntersectVisitor>(
        &self,
        field_number: i32,
        visitor: &mut V,
        upper_bound: i64,
        scratch: &mut PointsScratch,
    ) -> Result<i64> {
        let field = self
            .field(field_number)
            .ok_or(Error::IllegalFieldNumber(field_number))?;
        let inner_nodes = self.inner_nodes(field)?;
        let mut input = SliceInput::new(inner_nodes);
        let mut ctx = IntersectCtx::from_scratch(field, self.kdd, scratch);
        let root_fp = input.read_vlong();
        let out = match root_fp {
            Ok(fp) => estimate_node(&mut input, 1, fp, &mut ctx, visitor, upper_bound, 0),
            Err(e) => Err(e.into()),
        };
        ctx.into_scratch(scratch);
        out
    }

    /// Convenience wrapper over [`intersect`](Self::intersect) implementing
    /// `PointRangeQuery`'s visitor: every doc whose packed value falls inside
    /// the inclusive per-dimension box `[lower, upper]` (compared unsigned
    /// byte-wise per dimension, the ordering
    /// `NumericUtils.intToSortableBytes`/`longToSortableBytes` produce).
    /// `lower`/`upper` are `num_index_dims * bytes_per_dim` bytes.
    ///
    /// Returned doc ids are in leaf order and may repeat when a document has
    /// several matching points for the field -- exactly what Java's
    /// `PointRangeQuery` visitor sees before it folds them into a bitset.
    pub fn range_query(&self, field_number: i32, lower: &[u8], upper: &[u8]) -> Result<Vec<i32>> {
        let field = self
            .field(field_number)
            .ok_or(Error::IllegalFieldNumber(field_number))?;
        let mut visitor = self.range_visitor(field, lower, upper)?;
        self.intersect(field_number, &mut visitor)?;
        Ok(visitor.docs)
    }

    /// [`estimate_point_count`](Self::estimate_point_count) under
    /// [`range_query`](Self::range_query)'s visitor -- Java's
    /// `PointRangeQuery.ScorerSupplier`, which calls
    /// `values.estimatePointCount(visitor)` to size its `cost()` without
    /// running the query.
    ///
    /// The result is an *estimate*, deliberately: it over-counts a leaf the
    /// query only partly covers and never decodes one. Compare
    /// [`range_query`](Self::range_query), which answers exactly and costs a full intersect.
    pub fn estimate_range_point_count(
        &self,
        field_number: i32,
        lower: &[u8],
        upper: &[u8],
    ) -> Result<i64> {
        let field = self
            .field(field_number)
            .ok_or(Error::IllegalFieldNumber(field_number))?;
        let mut visitor = self.range_visitor(field, lower, upper)?;
        self.estimate_point_count(field_number, &mut visitor)
    }

    /// `PointRangeQuery`'s inclusive-box visitor over `field`, with the
    /// bounds-width check Java's `PointRangeQuery` constructor makes.
    fn range_visitor(
        &self,
        field: &PointsField,
        lower: &[u8],
        upper: &[u8],
    ) -> Result<RangeVisitor> {
        // `RangeVisitor` slices `lower`/`upper` per index dimension against
        // the *field's* shape, so a caller-supplied box of the wrong width
        // would index out of bounds mid-traversal. Java's `PointRangeQuery`
        // constructor makes the same check up front
        // (`checkArgs`/`Arrays.equals` on `numDims * bytesPerDim`); this is
        // the one place it can be made here.
        let expected = field.packed_index_bytes_length();
        if lower.len() != expected || upper.len() != expected {
            return Err(Error::InvalidConfig(format!(
                "range query bounds must be {expected} bytes (numIndexDims x bytesPerDim); \
                 got lower={}, upper={}",
                lower.len(),
                upper.len()
            )));
        }
        let (num_index_dims, bytes_per_dim) =
            (field.num_index_dims as usize, field.bytes_per_dim as usize);
        Ok(RangeVisitor {
            lower: lower.to_vec(),
            upper: upper.to_vec(),
            num_index_dims,
            bytes_per_dim,
            fixed: FixedWidthRange::new(num_index_dims, bytes_per_dim, lower, upper),
            docs: Vec::new(),
        })
    }
}

/// Port of `org.apache.lucene.index.PointValues.Relation`, defined in
/// `lucene-util` so the geo relations can answer in it too.
pub use lucene_util::point_values_relation::Relation;

/// Port of `org.apache.lucene.index.PointValues.IntersectVisitor`.
///
/// Java's `grow(int)` hint has no counterpart here: it exists to pre-size a
/// `DocIdSetBuilder`, and a Rust visitor that wants the same can size its own
/// storage from [`PointsField::point_count`].
pub trait IntersectVisitor {
    /// `IntersectVisitor.compare` -- how the query relates to the cell
    /// `[min_packed, max_packed]` (both `num_index_dims * bytes_per_dim`).
    fn compare(&mut self, min_packed: &[u8], max_packed: &[u8]) -> Relation;
    /// `IntersectVisitor.visit(int)` -- called for every doc in a cell that
    /// is entirely inside the query; the packed value is not decoded at all.
    fn visit(&mut self, doc_id: i32);
    /// `IntersectVisitor.visit(DocIdSetIterator)` in spirit: every doc id of a
    /// leaf entirely inside the query, at once. Defaults to [`Self::visit`]
    /// per id; a visitor that can take a run in bulk overrides it.
    fn visit_many(&mut self, doc_ids: &[i32]) {
        for &doc_id in doc_ids {
            self.visit(doc_id);
        }
    }
    /// `IntersectVisitor.visit(int, byte[])` -- called for every point in a
    /// cell that crosses the query boundary; the visitor must do its own
    /// per-point check.
    fn visit_with_value(&mut self, doc_id: i32, packed_value: &[u8]);
}

/// One inner node's split descriptor, decoded the way
/// `BKDReader.readNodeData` decodes it: `BKDWriter.recursePackIndex` packs
/// `splitDim`, the common-prefix length against the last split value seen in
/// that dimension, and the signed first-differing-byte delta into a single
/// vint as `(firstDiffByteDelta * (1 + bytesPerDim) + prefix) * numIndexDims
/// + splitDim`.
///
/// The three fields carry the invariants every caller's indexing rests on:
/// `split_dim < num_index_dims`, `prefix <= bytes_per_dim`, and
/// `first_diff_byte_delta >= 0` (the `negativeDeltas` sign flip is the
/// caller's, since only the pruning walk tracks it).
struct SplitDescriptor {
    split_dim: usize,
    prefix: usize,
    first_diff_byte_delta: i32,
}

/// Reads and unpacks one inner node's split descriptor.
///
/// The vint is attacker-shaped. Java's `code % numIndexDims` on a negative
/// `code` yields a negative `splitDim`, which indexes `splitValuesStack` out
/// of bounds one line later; the same value cast to `usize` here would be
/// astronomically large. Rejecting a negative `code` up front is what makes
/// the struct's three invariants true rather than assumed -- and it costs one
/// predictable branch per *inner node*, not per point.
fn read_split_descriptor(input: &mut SliceInput, field: &PointsField) -> Result<SplitDescriptor> {
    let code = input.read_vint()?;
    if code < 0 {
        return Err(corrupt(format!(
            "negative BKD split descriptor in .kdi: code={code}"
        )));
    }
    // ARITH: done in `i64` because `bytes_per_dim` is only bounded by
    // `num_dims * bytes_per_dim <= i32::MAX` (`check_config`), so
    // `1 + bytes_per_dim` can overflow an `i32` -- Java lets it wrap and
    // derives a nonsense prefix from the result. Widened, every operand is
    // non-negative (`code >= 0` by the guard above) and both divisors are
    // >= 1 (`num_index_dims >= 1` and `bytes_per_dim >= 1`, both from
    // `check_config`), so neither division can trap and no product is formed
    // at all. `code <= i32::MAX` bounds every quotient, so the casts back are
    // exact.
    #[allow(clippy::arithmetic_side_effects)]
    let descriptor = {
        let num_index_dims = i64::from(field.num_index_dims);
        let bytes_per_dim_plus_one = i64::from(field.bytes_per_dim) + 1;
        let code = i64::from(code);
        let split_dim = code % num_index_dims;
        let code = code / num_index_dims;
        SplitDescriptor {
            split_dim: split_dim as usize,
            prefix: (code % bytes_per_dim_plus_one) as usize,
            first_diff_byte_delta: (code / bytes_per_dim_plus_one) as i32,
        }
    };
    debug_assert!(descriptor.split_dim < field.num_index_dims as usize);
    debug_assert!(descriptor.prefix <= field.bytes_per_dim as usize);
    Ok(descriptor)
}

/// `nodeID * 2` and `nodeID * 2 + 1`, the packed tree's child ids.
///
/// Java lets both wrap an `int`. A wrap is not just a debug-build panic here:
/// a wrapped id compares `< numLeaves` all over again, and since the walk only
/// stops when it reaches a leaf, the recursion would keep descending -- the
/// depth is then bounded by nothing but the `.kdi` slice's length, i.e. a
/// large enough packed index overflows the *stack*. Checking the multiply caps
/// the depth at 31 levels instead, because `node_id` at least doubles per
/// level and has to stay below `num_leaves <= i32::MAX` to recurse at all.
fn child_ids(node_id: i32) -> Result<(i32, i32)> {
    let children = node_id
        .checked_mul(2)
        .and_then(|left| Some((left, left.checked_add(1)?)));
    children.ok_or_else(|| corrupt(format!("BKD node id overflows an i32: nodeID={node_id}")))
}

/// A leaf block's `.kdd` file pointer: the parent's baseline plus the right
/// child's delta. Both halves are `.kdi` values, and Java lets the `long` add
/// wrap before failing at `seek`.
fn child_block_fp(fp: i64, delta: i64) -> Result<i64> {
    fp.checked_add(delta).ok_or_else(|| {
        corrupt(format!(
            "BKD leaf block pointer overflows: fp={fp}, delta={delta}"
        ))
    })
}

/// Positions `.kdd` at a leaf block's file pointer. `fp` is reconstructed from
/// `.kdi` deltas, so a negative one is reachable; `as usize` would turn it into
/// a huge offset that merely looks like EOF.
fn seek_leaf_block(input: &mut SliceInput, fp: i64) -> Result<()> {
    let offset = usize::try_from(fp)
        .map_err(|_| corrupt(format!("negative BKD leaf block pointer: fp={fp}")))?;
    input.seek(offset)?;
    Ok(())
}

struct IntersectCtx<'a> {
    field: &'a PointsField,
    kdd: &'a [u8],
    min: Vec<u8>,
    max: Vec<u8>,
    split_values: Vec<u8>,
    negative_deltas: Vec<bool>,
    /// Java's per-level traversal stacks (`BKDPointTree`'s
    /// `splitValuesStack`, `splitDimValueStack` and the saved cell bound),
    /// which it sizes once from `getTreeDepth(numLeaves)`.
    ///
    /// Grown on demand instead, one entry the first time the walk reaches a
    /// level and reused for every node at that level thereafter, so a descent
    /// allocates `O(depth)` times per query rather than four times per inner
    /// node visited. Sizing it up front from `num_leaves` would be closer to
    /// Java but would let a corrupt `bytesPerDim` multiply a plausible
    /// allocation by the tree depth, which is the shape
    /// `docs/arithmetic-gate.md` singles out as the one that aborts.
    stack: Vec<LevelScratch>,
    /// Measurement-only. `false` reallocates every scratch buffer at each node
    /// instead of reusing it, which is exactly the pre-`c43-final-cleanup`
    /// shape -- four `Vec`s per inner node visited. It exists so
    /// `examples/bkd_walk_scratch.rs` can time both arms in **one process from
    /// one build**, which is the only shape of A/B this project trusts (see
    /// `docs/sweep/m2/c24-arith-codecs.md` on criterion, and `c42`'s stale
    /// binary). Production always passes `true`; the cost is one predictable
    /// branch per node against a `.kdi` read.
    reuse_scratch: bool,
    /// Java's `int[maxPointsInLeafNode]` doc-id buffer: every leaf the walk
    /// decodes (inside a matching subtree or crossing the query) reuses it.
    doc_ids: Vec<i32>,
}

/// A tree walk's buffers, kept between walks by a caller that makes many
/// ([`PointsReader::intersect_in`], [`PointsReader::estimate_point_count_bounded_in`]).
#[derive(Default)]
pub struct PointsScratch {
    min: Vec<u8>,
    max: Vec<u8>,
    split_values: Vec<u8>,
    negative_deltas: Vec<bool>,
    stack: Vec<LevelScratch>,
    doc_ids: Vec<i32>,
}

/// One level's reusable byte buffers -- see [`IntersectCtx::stack`].
///
/// The two cell bounds share one buffer because the walk never holds both:
/// `intersect_node` restores `max` before it saves `min`.
#[derive(Default)]
struct LevelScratch {
    /// `splitValuesStack[level]`: the reconstructed split value for this
    /// node's dimension, `bytes_per_dim` long.
    split_value: Vec<u8>,
    /// `ctx.split_values[dim_prefix_pos..dim_end]` as it was on entry, so the
    /// parent's state can be restored on the way back up
    /// (`recursePackIndex`'s `savSplitValue`, read side).
    saved_split_tail: Vec<u8>,
    /// The cell bound the descent is about to overwrite.
    saved_bound: Vec<u8>,
}

impl<'a> IntersectCtx<'a> {
    /// The root cell: the field's own bounding box, with the split-value
    /// buffer zeroed as `BKDReader`'s is.
    fn new(field: &'a PointsField, kdd: &'a [u8], reuse_scratch: bool) -> Self {
        IntersectCtx {
            field,
            kdd,
            min: field.min_packed_value.clone(),
            max: field.max_packed_value.clone(),
            split_values: vec![0u8; field.min_packed_value.len()],
            negative_deltas: vec![false; field.num_index_dims as usize],
            stack: Vec::new(),
            reuse_scratch,
            doc_ids: Vec::new(),
        }
    }

    /// [`Self::new`], over `scratch`'s buffers: the bounds and split values
    /// reset exactly as `new` sets them, the per-level stack and the leaf
    /// doc-id buffer kept (every walk overwrites a level's entry before it
    /// reads it, and a leaf's ids before it visits them).
    fn from_scratch(field: &'a PointsField, kdd: &'a [u8], scratch: &mut PointsScratch) -> Self {
        let mut min = std::mem::take(&mut scratch.min);
        min.clear();
        min.extend_from_slice(&field.min_packed_value);
        let mut max = std::mem::take(&mut scratch.max);
        max.clear();
        max.extend_from_slice(&field.max_packed_value);
        let mut split_values = std::mem::take(&mut scratch.split_values);
        split_values.clear();
        split_values.resize(field.min_packed_value.len(), 0);
        let mut negative_deltas = std::mem::take(&mut scratch.negative_deltas);
        negative_deltas.clear();
        negative_deltas.resize(field.num_index_dims as usize, false);
        IntersectCtx {
            field,
            kdd,
            min,
            max,
            split_values,
            negative_deltas,
            stack: std::mem::take(&mut scratch.stack),
            reuse_scratch: true,
            doc_ids: std::mem::take(&mut scratch.doc_ids),
        }
    }

    /// Hands the buffers back for the next walk.
    fn into_scratch(self, scratch: &mut PointsScratch) {
        scratch.min = self.min;
        scratch.max = self.max;
        scratch.split_values = self.split_values;
        scratch.negative_deltas = self.negative_deltas;
        scratch.stack = self.stack;
        scratch.doc_ids = self.doc_ids;
    }

    /// Makes `self.stack[level]` addressable. The walk only ever descends one
    /// level at a time, so this appends at most one entry per call.
    fn ensure_level(&mut self, level: usize) {
        if self.stack.len() <= level {
            self.stack
                .resize_with(level.saturating_add(1), Default::default);
        }
    }
}

/// One node of [`PointsReader::intersect`]'s traversal. `input` is
/// positioned at this node's own packed-index data (the caller has already
/// consumed the FP-delta vlong, if any, and passes the resulting `fp`).
/// Returning early on [`Relation::CellOutsideQuery`] deliberately leaves
/// `input` where it was: the caller always seeks to the right sibling's
/// recorded position before descending into it, so an unconsumed left
/// subtree costs nothing.
fn intersect_node<V: IntersectVisitor>(
    input: &mut SliceInput,
    node_id: i32,
    fp: i64,
    ctx: &mut IntersectCtx,
    visitor: &mut V,
    level: usize,
) -> Result<()> {
    let relation = visitor.compare(&ctx.min, &ctx.max);
    if relation == Relation::CellOutsideQuery {
        return Ok(());
    }

    if relation == Relation::CellInsideQuery {
        // `PointValues.intersect`: an entirely-inside cell short-circuits to
        // `visitDocIDs`, which never calls `compare` again and never decodes
        // a packed value anywhere in the subtree.
        return add_all(input, node_id, fp, ctx, visitor);
    }

    if node_id >= ctx.field.num_leaves {
        // Leaf reached with CELL_CROSSES_QUERY: `visitDocValues`, i.e. decode
        // the block and let the visitor filter point by point.
        let mut kdd_input = SliceInput::new(ctx.kdd);
        seek_leaf_block(&mut kdd_input, fp)?;
        read_leaf_block_into(
            &mut kdd_input,
            ctx.field,
            &mut ctx.doc_ids,
            &mut VisitSink(visitor),
        )?;
        return Ok(());
    }

    let node = read_inner_node(input, node_id, ctx, level)?;
    // ARITH: `child_ids` caps the descent at ~31 levels (`node_id` at least
    // doubles per level and must stay below `i32::MAX / 2` to recurse), so the
    // increment cannot overflow a `usize`.
    #[allow(clippy::arithmetic_side_effects)]
    let child_level = level + 1;

    // Left child: cell max is clamped down to the split value.
    clamp_bound(
        &mut ctx.max,
        &mut ctx.stack[level],
        node.dim_pos..node.dim_end,
        ctx.reuse_scratch,
    );
    ctx.negative_deltas[node.split_dim] = true;
    intersect_node(input, node.left_child, fp, ctx, visitor, child_level)?;
    restore_bound(&mut ctx.max, &ctx.stack[level], node.dim_pos..node.dim_end);

    // Right child: cell min is clamped up to the split value. Its own
    // packed-index bytes start at `right_node_position` whether or not the
    // left subtree was actually parsed.
    input.seek(node.right_node_position)?;
    let right_delta = input.read_vlong()?;
    let right_fp = child_block_fp(fp, right_delta)?;
    clamp_bound(
        &mut ctx.min,
        &mut ctx.stack[level],
        node.dim_pos..node.dim_end,
        ctx.reuse_scratch,
    );
    ctx.negative_deltas[node.split_dim] = false;
    intersect_node(input, node.right_child, right_fp, ctx, visitor, child_level)?;
    restore_bound(&mut ctx.min, &ctx.stack[level], node.dim_pos..node.dim_end);

    node.restore(ctx);
    Ok(())
}

/// One inner node's decoded `.kdi` data -- everything both tree walks need
/// after `BKDReader.readNodeData` has run.
///
/// `PointValues.intersect` and `PointValues.estimatePointCount` differ only in
/// what they do *at* a node; the descent itself is one piece of parsing and
/// lives here so the two cannot drift apart.
struct InnerNode {
    /// Which index dimension this node splits on.
    split_dim: usize,
    /// Byte range of `split_dim` inside a cell-bounds/split-value buffer.
    dim_pos: usize,
    dim_end: usize,
    /// Where inside that range this node's own bytes start.
    dim_prefix_pos: usize,
    /// `ctx.negative_deltas[split_dim]` as it was on entry.
    saved_negative_delta: bool,
    /// Which [`IntersectCtx::stack`] entry holds this node's split value and
    /// saved state -- `BKDPointTree`'s `level`.
    level: usize,
    /// `.kdi` offset of the right child's leading FP-delta vlong --
    /// `BKDPointTree.pushRight`'s `rightNodePositions[level]`.
    right_node_position: usize,
    left_child: i32,
    right_child: i32,
}

impl InnerNode {
    /// Undoes this node's edits to the shared traversal state --
    /// `BKDPointTree.pop`'s half that is not a cell bound.
    fn restore(&self, ctx: &mut IntersectCtx) {
        ctx.negative_deltas[self.split_dim] = self.saved_negative_delta;
        copy_dim(
            &mut ctx.split_values[self.dim_prefix_pos..self.dim_end],
            &ctx.stack[self.level].saved_split_tail,
        );
    }
}

/// `BKDPointTree.pushLeft`/`pushRight`'s cell-bound half: clamp `bound[range]`
/// to this level's split value, stashing what was there in the level's own
/// scratch so [`restore_bound`] can put it back.
///
/// Free functions rather than [`InnerNode`] methods so the caller can hand
/// over `&mut ctx.max` and `&mut ctx.stack[level]` as two disjoint field
/// borrows -- which is what lets the buffers be reused instead of cloned.
fn clamp_bound(
    bound: &mut [u8],
    scratch: &mut LevelScratch,
    range: std::ops::Range<usize>,
    reuse: bool,
) {
    refill(&mut scratch.saved_bound, &bound[range.clone()], reuse);
    copy_dim(&mut bound[range], &scratch.split_value);
}

/// `dst.copy_from_slice(src)` for one dimension's bytes. The walk copies a
/// split value or a cell bound several times per node, and a slice copy of a
/// length only known at run time is a libc `memmove` call each time -- 7% of
/// a numeric sort's competitive-iterator rebuilds. An `int`, `float`, `long`
/// or `double` point (4 or 8 bytes) moves as one word instead.
#[inline(always)]
fn copy_dim(dst: &mut [u8], src: &[u8]) {
    // A fixed-size move per length up to a `long`'s 8 bytes -- a split
    // value's suffix can be any of them -- and a slice copy past that.
    macro_rules! fixed {
        ($($n:literal)*) => {
            match (dst.len(), src.len()) {
                $(($n, $n) => {
                    let d: &mut [u8; $n] = (&mut *dst).try_into().expect("length matched");
                    *d = src.try_into().expect("length matched");
                })*
                _ => dst.copy_from_slice(src),
            }
        };
    }
    fixed!(1 2 3 4 5 6 7 8);
}

/// `buf` holding exactly `src`: in place when it already has that length
/// (a field's dimensions are all `bytes_per_dim` long, so after the first
/// node it usually does), else emptied and refilled. The measurement arm
/// (`reuse == false`) always takes the refill, which allocates.
#[inline(always)]
fn refill(buf: &mut Vec<u8>, src: &[u8], reuse: bool) {
    if reuse && buf.len() == src.len() {
        copy_dim(buf, src);
    } else {
        reset(buf, reuse);
        buf.extend_from_slice(src);
    }
}

/// Empties `buf` for reuse, or -- in the measurement arm -- throws it away so
/// the refill allocates. See [`IntersectCtx::reuse_scratch`].
fn reset(buf: &mut Vec<u8>, reuse: bool) {
    if reuse {
        buf.clear();
    } else {
        *buf = Vec::new();
    }
}

/// Undoes [`clamp_bound`] -- `BKDPointTree.pop`'s cell-bound half.
fn restore_bound(bound: &mut [u8], scratch: &LevelScratch, range: std::ops::Range<usize>) {
    copy_dim(&mut bound[range], &scratch.saved_bound);
}

/// `BKDReader.readNodeData`: decode one inner node's split descriptor into
/// `ctx.split_values` and work out where its right child's data begins.
///
/// `input` must be positioned at the node's own packed-index data (the caller
/// has already consumed its leading FP-delta vlong).
fn read_inner_node(
    input: &mut SliceInput,
    node_id: i32,
    ctx: &mut IntersectCtx,
    level: usize,
) -> Result<InnerNode> {
    ctx.ensure_level(level);
    let bytes_per_dim = ctx.field.bytes_per_dim as usize;

    // `BKDReader.readNodeData`: split dim, prefix and first-diff-byte delta
    // packed into one vint, then the split value's raw suffix.
    let SplitDescriptor {
        split_dim,
        prefix,
        first_diff_byte_delta,
    } = read_split_descriptor(input, ctx.field)?;
    // ARITH: `read_split_descriptor` guarantees `prefix <= bytes_per_dim` and
    // `split_dim < num_index_dims`, so `suffix` cannot underflow and
    // `dim_pos + bytes_per_dim <= num_index_dims * bytes_per_dim`, which is
    // exactly `ctx.split_values.len()` (it is cloned from
    // `field.min_packed_value`). `check_config` proved that product fits an
    // `i32`.
    #[allow(clippy::arithmetic_side_effects)]
    let (suffix, dim_pos) = (bytes_per_dim - prefix, split_dim * bytes_per_dim);
    // ARITH: same bounds -- `prefix <= bytes_per_dim` keeps the range
    // non-inverted and inside the buffer.
    #[allow(clippy::arithmetic_side_effects)]
    let (dim_prefix_pos, dim_end) = (dim_pos + prefix, dim_pos + bytes_per_dim);
    debug_assert!(dim_prefix_pos <= dim_end && dim_end <= ctx.split_values.len());
    let reuse = ctx.reuse_scratch;
    let scratch = &mut ctx.stack[level];
    refill(
        &mut scratch.saved_split_tail,
        &ctx.split_values[dim_prefix_pos..dim_end],
        reuse,
    );
    if suffix > 0 {
        // ARITH: `first_diff_byte_delta >= 0` (`read_split_descriptor`
        // rejects a negative `code`), so the negation cannot overflow -- only
        // `i32::MIN` does. The `wrapping_add` is Java's own semantics: it
        // writes `(byte) (oldByte + firstDiffByteDelta)`, so an out-of-range
        // delta truncates rather than trapping, and truncating is what
        // reproduces the split value a real `.kdi` encodes.
        let mut delta = first_diff_byte_delta;
        if ctx.negative_deltas[split_dim] {
            // ARITH: `read_split_descriptor` rejects a negative `code`, so
            // `first_diff_byte_delta >= 0` and the negation cannot overflow --
            // only `i32::MIN` does.
            #[allow(clippy::arithmetic_side_effects)]
            {
                delta = -delta;
            }
        }
        let old = i32::from(ctx.split_values[dim_prefix_pos]);
        ctx.split_values[dim_prefix_pos] = old.wrapping_add(delta) as u8;
        // ARITH: `suffix > 0` means `prefix < bytes_per_dim`, so
        // `dim_prefix_pos + 1 <= dim_end`.
        #[allow(clippy::arithmetic_side_effects)]
        let tail_start = dim_prefix_pos + 1;
        // `readBytes` of the suffix, through `copy_dim`: a slice copy of a
        // length only known at run time is a libc call, once per node.
        let at = input.position();
        let src = input.slice(at, at.saturating_add(dim_end.saturating_sub(tail_start)))?;
        input.seek(at.saturating_add(src.len()))?;
        copy_dim(&mut ctx.split_values[tail_start..dim_end], src);
    }
    // else: this node's split value is byte-identical to the last one seen in
    // this dimension (many duplicate values) -- nothing to read or change.

    let (left_child, right_child) = child_ids(node_id)?;
    let left_num_bytes = if left_child < ctx.field.num_leaves {
        let left_num_bytes = input.read_vint()?;
        usize::try_from(left_num_bytes).map_err(|_| {
            corrupt(format!(
                "negative leftNumBytes in .kdi: nodeID={node_id}, leftNumBytes={left_num_bytes}"
            ))
        })?
    } else {
        0
    };
    // `leftNumBytes` is a `.kdi` vint; the sum is what the seek to the right
    // sibling uses, so an overflow here would silently land the walk on the
    // wrong node in a release build.
    let Some(right_node_position) = input.position().checked_add(left_num_bytes) else {
        return Err(corrupt(format!(
            "leftNumBytes overruns .kdi: nodeID={node_id}, leftNumBytes={left_num_bytes}"
        )));
    };
    let scratch = &mut ctx.stack[level];
    refill(
        &mut scratch.split_value,
        &ctx.split_values[dim_pos..dim_end],
        reuse,
    );
    Ok(InnerNode {
        split_dim,
        dim_pos,
        dim_end,
        dim_prefix_pos,
        saved_negative_delta: ctx.negative_deltas[split_dim],
        level,
        right_node_position,
        left_child,
        right_child,
    })
}

/// One node of [`PointsReader::estimate_point_count`]'s traversal -- the
/// private `PointValues.estimatePointCount(visitor, pointTree, upperBound)`.
///
/// Reads strictly less than [`intersect_node`]: an entirely-inside cell is
/// answered from the node id alone (`BKDPointTree.size()`), never by
/// descending, and no leaf block is ever decoded.
fn estimate_node<V: IntersectVisitor>(
    input: &mut SliceInput,
    node_id: i32,
    fp: i64,
    ctx: &mut IntersectCtx,
    visitor: &mut V,
    upper_bound: i64,
    level: usize,
) -> Result<i64> {
    match visitor.compare(&ctx.min, &ctx.max) {
        // "This cell is fully outside the query shape: no points added."
        Relation::CellOutsideQuery => Ok(0),
        // "This cell is fully inside the query shape: add all points."
        Relation::CellInsideQuery => ctx.field.subtree_size(node_id),
        Relation::CellCrossesQuery => {
            if node_id >= ctx.field.num_leaves {
                // `moveToChild()` said no: "Assume half the points matched."
                // ARITH: `subtree_size` returns `1..=point_count`, so `+ 1`
                // cannot overflow an `i64` that already holds a point count.
                #[allow(clippy::arithmetic_side_effects)]
                return Ok((ctx.field.subtree_size(node_id)? + 1) / 2);
            }
            let node = read_inner_node(input, node_id, ctx, level)?;
            // ARITH: bounded by `child_ids`' ~31-level cap, as in
            // `intersect_node`.
            #[allow(clippy::arithmetic_side_effects)]
            let child_level = level + 1;

            clamp_bound(
                &mut ctx.max,
                &mut ctx.stack[level],
                node.dim_pos..node.dim_end,
                ctx.reuse_scratch,
            );
            ctx.negative_deltas[node.split_dim] = true;
            let left = estimate_node(
                input,
                node.left_child,
                fp,
                ctx,
                visitor,
                upper_bound,
                child_level,
            )?;
            restore_bound(&mut ctx.max, &ctx.stack[level], node.dim_pos..node.dim_end);

            // Java's `while (cost < upperBound && pointTree.moveToSibling())`:
            // once the running cost has reached the bound the caller only
            // wanted a yes/no answer, so the right sibling is never read.
            let mut cost = left;
            if cost < upper_bound {
                input.seek(node.right_node_position)?;
                let right_delta = input.read_vlong()?;
                let right_fp = child_block_fp(fp, right_delta)?;
                clamp_bound(
                    &mut ctx.min,
                    &mut ctx.stack[level],
                    node.dim_pos..node.dim_end,
                    ctx.reuse_scratch,
                );
                ctx.negative_deltas[node.split_dim] = false;
                // ARITH: `upperBound - cost` in Java. The `if` above proves
                // `cost < upper_bound`, and `cost` is non-negative (it is
                // either `0`, a `subtree_size` -- which saturates at or above
                // `1` -- or a `saturating_add` of two such), so the difference
                // is in `1..=upper_bound` and cannot underflow.
                #[allow(clippy::arithmetic_side_effects)]
                let remaining = upper_bound - cost;
                let right = estimate_node(
                    input,
                    node.right_child,
                    right_fp,
                    ctx,
                    visitor,
                    remaining,
                    child_level,
                )?;
                restore_bound(&mut ctx.min, &ctx.stack[level], node.dim_pos..node.dim_end);
                // Both halves are bounded by the field's point count, which is
                // an `i64` read off `.kdm`; saturating rather than wrapping
                // keeps a corrupt count from turning a cost estimate negative.
                cost = cost.saturating_add(right);
            }

            node.restore(ctx);
            Ok(cost)
        }
    }
}

/// One node of [`PointsReader::count_points`]' walk: [`estimate_node`]'s
/// descent with [`intersect_node`]'s leaf.
fn count_node<V: IntersectVisitor>(
    input: &mut SliceInput,
    node_id: i32,
    fp: i64,
    ctx: &mut IntersectCtx,
    visitor: &mut V,
    level: usize,
) -> Result<i64> {
    match visitor.compare(&ctx.min, &ctx.max) {
        Relation::CellOutsideQuery => Ok(0),
        Relation::CellInsideQuery => ctx.field.subtree_size(node_id),
        Relation::CellCrossesQuery if node_id >= ctx.field.num_leaves => {
            let mut kdd_input = SliceInput::new(ctx.kdd);
            seek_leaf_block(&mut kdd_input, fp)?;
            read_leaf_block_into(
                &mut kdd_input,
                ctx.field,
                &mut ctx.doc_ids,
                &mut VisitSink(visitor),
            )?;
            Ok(0)
        }
        Relation::CellCrossesQuery => {
            let node = read_inner_node(input, node_id, ctx, level)?;
            // ARITH: bounded by `child_ids`' ~31-level cap, as in
            // `intersect_node`.
            #[allow(clippy::arithmetic_side_effects)]
            let child_level = level + 1;
            clamp_bound(
                &mut ctx.max,
                &mut ctx.stack[level],
                node.dim_pos..node.dim_end,
                ctx.reuse_scratch,
            );
            ctx.negative_deltas[node.split_dim] = true;
            let left = count_node(input, node.left_child, fp, ctx, visitor, child_level)?;
            restore_bound(&mut ctx.max, &ctx.stack[level], node.dim_pos..node.dim_end);

            input.seek(node.right_node_position)?;
            let right_delta = input.read_vlong()?;
            let right_fp = child_block_fp(fp, right_delta)?;
            clamp_bound(
                &mut ctx.min,
                &mut ctx.stack[level],
                node.dim_pos..node.dim_end,
                ctx.reuse_scratch,
            );
            ctx.negative_deltas[node.split_dim] = false;
            let right = count_node(input, node.right_child, right_fp, ctx, visitor, child_level)?;
            restore_bound(&mut ctx.min, &ctx.stack[level], node.dim_pos..node.dim_end);

            node.restore(ctx);
            // Each half is at most the field's point count, an `i64`.
            Ok(left.saturating_add(right))
        }
    }
}

/// Port of `BKDReader.BKDPointTree.addAll` (reached from `visitDocIDs`):
/// every leaf under `node_id` contributes its doc ids and nothing else --
/// no `compare`, no packed values, no per-leaf bounding box. Consumes the
/// whole subtree's packed-index bytes, exactly like [`walk_node`], so the
/// caller's cursor arithmetic is unaffected by which branch it took.
fn add_all<V: IntersectVisitor>(
    input: &mut SliceInput,
    node_id: i32,
    fp: i64,
    ctx: &mut IntersectCtx,
    visitor: &mut V,
) -> Result<()> {
    if node_id >= ctx.field.num_leaves {
        let mut kdd_input = SliceInput::new(ctx.kdd);
        seek_leaf_block(&mut kdd_input, fp)?;
        let count = read_leaf_count(&mut kdd_input, ctx.field)?;
        read_doc_ids_into(
            &mut kdd_input,
            count,
            &mut ctx.doc_ids,
            ctx.field.bkd_version,
        )?;
        visitor.visit_many(&ctx.doc_ids);
        return Ok(());
    }

    // Same split-descriptor skip as `walk_node`: the split value itself is
    // irrelevant once the whole subtree is known to match.
    let descriptor = read_split_descriptor(input, ctx.field)?;
    skip_split_value_suffix(input, ctx.field, &descriptor)?;
    let (left_child, right_child) = child_ids(node_id)?;
    if left_child < ctx.field.num_leaves {
        input.read_vint()?; // leftNumBytes: not needed, we visit both halves
    }
    add_all(input, left_child, fp, ctx, visitor)?;
    let right_delta = input.read_vlong()?;
    add_all(
        input,
        right_child,
        child_block_fp(fp, right_delta)?,
        ctx,
        visitor,
    )
}

/// `PointRangeQuery`'s `IntersectVisitor`, inclusive on both ends.
struct RangeVisitor {
    lower: Vec<u8>,
    upper: Vec<u8>,
    num_index_dims: usize,
    bytes_per_dim: usize,
    /// The bounds as integers, for the one-dimensional 4- and 8-byte shapes.
    fixed: Option<FixedWidthRange>,
    docs: Vec<i32>,
}

/// `ArrayUtil.getUnsignedComparator`'s specialisation, which
/// `PointRangeQuery`'s visitor compares with: a 4- or 8-byte value (`int`,
/// `float`, `long`, `double` points) is one unsigned big-endian integer
/// comparison, which orders exactly as the byte-wise one does. Compared as
/// `[u8]` slices, each point cost two `memcmp` calls -- a fifth of a
/// `num:[1000 TO 2000]` query.
#[derive(Clone, Copy)]
struct FixedWidthRange {
    width: usize,
    lower: u64,
    upper: u64,
}

impl FixedWidthRange {
    fn new(
        num_index_dims: usize,
        bytes_per_dim: usize,
        lower: &[u8],
        upper: &[u8],
    ) -> Option<Self> {
        if num_index_dims != 1 || !matches!(bytes_per_dim, 4 | 8) {
            return None;
        }
        Some(Self {
            width: bytes_per_dim,
            lower: Self::key(lower, bytes_per_dim)?,
            upper: Self::key(upper, bytes_per_dim)?,
        })
    }

    /// The first `width` (4 or 8) bytes, big-endian; `None` if too short.
    #[inline]
    fn key(bytes: &[u8], width: usize) -> Option<u64> {
        if width == 8 {
            Some(u64::from_be_bytes(bytes.get(..8)?.try_into().ok()?))
        } else {
            Some(u64::from(u32::from_be_bytes(
                bytes.get(..4)?.try_into().ok()?,
            )))
        }
    }

    /// Whether `packed` (at least `width` bytes) is inside the range; `None`
    /// if it is too short to say.
    #[inline]
    fn contains(&self, packed: &[u8]) -> Option<bool> {
        let v = Self::key(packed, self.width)?;
        Some(self.lower <= v && v <= self.upper)
    }
}

impl IntersectVisitor for RangeVisitor {
    fn compare(&mut self, min_packed: &[u8], max_packed: &[u8]) -> Relation {
        let mut crosses = false;
        for dim in 0..self.num_index_dims {
            // ARITH: `dim < num_index_dims`, so `hi <= num_index_dims *
            // bytes_per_dim` -- the length `range_query` verified for
            // `lower`/`upper`, the length `PointsReader::intersect` gives the
            // cell bounds it passes in, and a product `check_config` proved
            // fits an `i32`.
            #[allow(clippy::arithmetic_side_effects)]
            let (lo, hi) = (dim * self.bytes_per_dim, (dim + 1) * self.bytes_per_dim);
            if max_packed[lo..hi] < self.lower[lo..hi] || min_packed[lo..hi] > self.upper[lo..hi] {
                return Relation::CellOutsideQuery;
            }
            crosses |=
                min_packed[lo..hi] < self.lower[lo..hi] || max_packed[lo..hi] > self.upper[lo..hi];
        }
        if crosses {
            Relation::CellCrossesQuery
        } else {
            Relation::CellInsideQuery
        }
    }

    fn visit(&mut self, doc_id: i32) {
        self.docs.push(doc_id);
    }

    fn visit_with_value(&mut self, doc_id: i32, packed_value: &[u8]) {
        if let Some(inside) = self.fixed.and_then(|f| f.contains(packed_value)) {
            if inside {
                self.docs.push(doc_id);
            }
            return;
        }
        for dim in 0..self.num_index_dims {
            // ARITH: same bound as `compare`; `packed_value` is
            // `num_dims * bytes_per_dim` long, which is at least as long as
            // the `num_index_dims * bytes_per_dim` scanned here.
            #[allow(clippy::arithmetic_side_effects)]
            let (lo, hi) = (dim * self.bytes_per_dim, (dim + 1) * self.bytes_per_dim);
            if packed_value[lo..hi] < self.lower[lo..hi]
                || packed_value[lo..hi] > self.upper[lo..hi]
            {
                return;
            }
        }
        self.docs.push(doc_id);
    }
}

/// Walks the packed binary tree in `.kdi` and returns every leaf's `.kdd`
/// file pointer, in left-to-right (in-order) order. See the module doc for
/// why this never seeks: a leaf is a node whose id is `>= num_leaves`
/// (`leafNodeOffset`), and the tree's root is node 1.
fn decode_leaf_pointers(inner_nodes: &[u8], field: &PointsField) -> Result<Vec<i64>> {
    let mut input = SliceInput::new(inner_nodes);
    let mut leaves = Vec::with_capacity(leaf_pointer_capacity(field.num_leaves, inner_nodes.len()));
    // The root is always reached as if it were a "right" child of an
    // implicit level 0 baseline of 0 -- `BKDReader`'s constructor calls
    // `readNodeData(false)` for the root, which always reads one leading
    // FP-delta vlong regardless of leaf-ness.
    let root_fp = input.read_vlong()?;
    walk_node(&mut input, 1, root_fp, field, &mut leaves)?;
    Ok(leaves)
}

/// How many leaf pointers to reserve up front for a `.kdm` claiming
/// `num_leaves` leaves against a `.kdi` slice of `inner_nodes_len` bytes.
///
/// `numLeaves` is a `.kdm` vint bounded only by `> 0`, and eight bytes per
/// leaf of `i32::MAX` leaves is a 17 GB reservation -- an allocation failure,
/// which **aborts**, and no `catch_unwind` at the FFI boundary can intercept
/// an abort. (Batch b7 found the sibling of this defect in the same header.)
///
/// The packed index itself is the ceiling. Every leaf but the leftmost is
/// reached through a right-child FP-delta vlong of at least one byte, and the
/// root costs one such vlong too, so a `.kdi` slice of `b` bytes can describe
/// at most `b + 1` leaves. That looser bound is used rather than the tighter
/// `b / 2 + 1` (each of the `n - 1` inner nodes also costs a split-descriptor
/// vint) so an off-by-one in the accounting can only cost a reallocation,
/// never reject a real file: for a well-formed `.kdi` the `min` always picks
/// `num_leaves` and the reservation is exactly what it was before.
fn leaf_pointer_capacity(num_leaves: i32, inner_nodes_len: usize) -> usize {
    // `read_field_meta` rejects `num_leaves <= 0`, so the `max(0)` is only
    // belt-and-braces for a hand-built `PointsField` in a test.
    debug_assert!(num_leaves > 0);
    (num_leaves.max(0) as usize).min(inner_nodes_len.saturating_add(1))
}

fn walk_node(
    input: &mut SliceInput,
    node_id: i32,
    fp: i64,
    field: &PointsField,
    leaves: &mut Vec<i64>,
) -> Result<()> {
    if node_id >= field.num_leaves {
        leaves.push(fp);
        return Ok(());
    }

    // Split descriptor: one vint encodes splitDim, prefix, and (if the
    // dimension's suffix is nonempty) a signed firstDiffByteDelta, all via
    // modulo/division -- we only need to consume the right number of
    // trailing raw bytes, not the actual split value, since we visit every
    // node regardless of any query bound.
    let descriptor = read_split_descriptor(input, field)?;
    skip_split_value_suffix(input, field, &descriptor)?;

    let (left_child, right_child) = child_ids(node_id)?;
    if left_child < field.num_leaves {
        input.read_vint()?; // leftNumBytes: a skip-ahead hint, unused (see module doc)
    }

    // Left child inherits this node's FP baseline unchanged.
    walk_node(input, left_child, fp, field, leaves)?;
    // Right child's FP is a delta from this node's baseline, read
    // immediately after the (fully consumed) left subtree.
    let right_delta = input.read_vlong()?;
    walk_node(
        input,
        right_child,
        child_block_fp(fp, right_delta)?,
        field,
        leaves,
    )?;
    Ok(())
}

/// Steps past the raw bytes of a split value without reconstructing it -- what
/// both non-pruning walks ([`walk_node`] and [`add_all`]) need. The first
/// differing byte is carried in the descriptor's delta rather than written, so
/// only `suffix - 1` bytes are on the wire.
fn skip_split_value_suffix(
    input: &mut SliceInput,
    field: &PointsField,
    descriptor: &SplitDescriptor,
) -> Result<()> {
    // ARITH: `read_split_descriptor` guarantees `prefix <= bytes_per_dim`, so
    // `suffix` cannot underflow, and the `suffix > 0` guard makes
    // `suffix - 1` safe in turn.
    #[allow(clippy::arithmetic_side_effects)]
    let suffix = field.bytes_per_dim as usize - descriptor.prefix;
    if suffix > 0 {
        // ARITH: `suffix > 0` is the guard, so `suffix - 1` cannot underflow.
        #[allow(clippy::arithmetic_side_effects)]
        input.skip(suffix - 1)?;
    }
    Ok(())
}

/// Decodes one leaf block (doc ids + packed values) at the data input's
/// current position, appending every point to `out`.
/// A leaf block's leading point count.
///
/// Java bounds this implicitly but absolutely: `DocIdsWriter.readInts` decodes
/// into `BKDReaderDocIDSetIterator.docIDs`, a `new int[maxPointsInLeafNode]`
/// allocated once when the point tree is built, so a `count` past
/// `maxPointsInLeafNode` throws `ArrayIndexOutOfBoundsException` before a
/// single doc id lands. This port allocates per leaf instead, so the same
/// `.kdd` vint sizes a fresh `Vec` -- and a count of `i32::MAX` is a multi-GB
/// reservation whose failure **aborts**, which `catch_unwind` at the FFI
/// boundary cannot intercept. Restating Java's invariant explicitly costs one
/// comparison per leaf and cannot reject a file any writer produced.
fn read_leaf_count(input: &mut SliceInput, field: &PointsField) -> Result<usize> {
    let count = input.read_vint()?;
    if count < 0 || count > field.max_points_in_leaf_node {
        return Err(corrupt(format!(
            "leaf block claims {count} points, outside 0..={} (maxPointsInLeafNode)",
            field.max_points_in_leaf_node
        )));
    }
    Ok(count as usize)
}

/// Reads one point's non-prefix suffix bytes into `scratch_value`, dimension by
/// dimension -- the inner `readBytes` loop shared by
/// `visitSparseRawDocValues` and `visitCompressedDocValues`.
#[inline]
fn read_suffix_bytes(
    input: &mut SliceInput,
    common_prefix_lengths: &[usize],
    bytes_per_dim: usize,
    scratch_value: &mut [u8],
) -> Result<()> {
    for (dim, &prefix) in common_prefix_lengths.iter().enumerate() {
        // ARITH: `read_leaf_block` bounds every entry of
        // `common_prefix_lengths` by `bytes_per_dim` before this runs (and the
        // one `+= 1` it applies is guarded by a `prefix < bytes_per_dim`
        // check), and `dim < num_dims` because the slice is `num_dims` long.
        // So `dim * bytes_per_dim + prefix <= (dim + 1) * bytes_per_dim <=
        // num_dims * bytes_per_dim`, which `check_config` proved fits an `i32`
        // and which is exactly `scratch_value.len()`.
        #[allow(clippy::arithmetic_side_effects)]
        let (lo, hi) = (dim * bytes_per_dim + prefix, (dim + 1) * bytes_per_dim);
        debug_assert!(lo <= hi && hi <= scratch_value.len());
        input.read_bytes(&mut scratch_value[lo..hi])?;
    }
    Ok(())
}

/// Where a decoded leaf's points go: a `Vec<Point>` for the callers that want
/// them all in hand, or straight to an [`IntersectVisitor`] for a crossing
/// leaf, which is `BKDReader.visitDocValues` -- one reused scratch value,
/// handed to the visitor point by point, never copied.
trait LeafSink {
    fn point(&mut self, doc_id: i32, packed_value: &[u8]);
}

impl LeafSink for Vec<Point> {
    fn point(&mut self, doc_id: i32, packed_value: &[u8]) {
        self.push(Point {
            doc_id,
            packed_value: packed_value.to_vec(),
        });
    }
}

/// A crossing leaf's points, streamed to the visitor. This used to decode the
/// leaf into a `Vec<Point>` first -- one heap allocation *per point* for its
/// packed value -- which made a small range query 2.4x slower than Lucene.
struct VisitSink<'v, V>(&'v mut V);

impl<V: IntersectVisitor> LeafSink for VisitSink<'_, V> {
    #[inline]
    fn point(&mut self, doc_id: i32, packed_value: &[u8]) {
        self.0.visit_with_value(doc_id, packed_value);
    }
}

fn read_leaf_block(
    input: &mut SliceInput,
    field: &PointsField,
    out: &mut Vec<Point>,
) -> Result<Option<(Vec<u8>, Vec<u8>)>> {
    read_leaf_block_into(input, field, &mut Vec::new(), out)
}

/// `doc_ids` is the caller's reusable decode buffer; its contents are
/// replaced.
fn read_leaf_block_into<S: LeafSink>(
    input: &mut SliceInput,
    field: &PointsField,
    doc_ids: &mut Vec<i32>,
    out: &mut S,
) -> Result<Option<(Vec<u8>, Vec<u8>)>> {
    let count = read_leaf_count(input, field)?;
    read_doc_ids_into(input, count, doc_ids, field.bkd_version)?;
    let doc_ids: &[i32] = doc_ids;

    let num_dims = field.num_dims as usize;
    let num_index_dims = field.num_index_dims as usize;
    let bytes_per_dim = field.bytes_per_dim as usize;
    let packed_bytes_length = field.packed_bytes_length();

    let mut common_prefix_lengths = vec![0usize; num_dims];
    let mut scratch_value = vec![0u8; packed_bytes_length];
    for (dim, prefix_len) in common_prefix_lengths.iter_mut().enumerate() {
        let prefix = input.read_vint()?;
        // A common prefix longer than the dimension it belongs to is not
        // something `BKDWriter` can emit (it is the length of a byte-wise
        // common prefix of that dimension's `bytesPerDim` bytes). Java lets it
        // through and `readBytes` spills into the *next* dimension's bytes --
        // silently decoding wrong point values for an all-equal leaf, and
        // throwing on `bytesPerDim - prefix` later otherwise. Here the same
        // spill inverts every later suffix range, which panics. One
        // comparison per dimension per leaf turns both into a decode error.
        if prefix < 0 || prefix > field.bytes_per_dim {
            return Err(corrupt(format!(
                "leaf common prefix for dim={dim} is {prefix} bytes, past bytesPerDim={}",
                field.bytes_per_dim
            )));
        }
        let prefix = prefix as usize;
        *prefix_len = prefix;
        if prefix > 0 {
            // ARITH: `dim < num_dims` and `prefix <= bytes_per_dim` (just
            // checked), so `dim * bytes_per_dim + prefix <= num_dims *
            // bytes_per_dim = scratch_value.len()`, a product `check_config`
            // proved fits an `i32`.
            #[allow(clippy::arithmetic_side_effects)]
            let (lo, hi) = (dim * bytes_per_dim, dim * bytes_per_dim + prefix);
            input.read_bytes(&mut scratch_value[lo..hi])?;
        }
    }

    // Port of `BKDReader.visitDocValuesWithCardinality` (the current-version
    // leaf layout, `version >= VERSION_LOW_CARDINALITY_LEAVES`, which every
    // BKD tree this port reads/writes uses -- see the module doc): the
    // compressed-dimension marker comes **before** the per-leaf bounding
    // box, not after -- an earlier version of this decoder read the box
    // first (matching the *older*, no-longer-written
    // `visitDocValuesNoCardinality` layout instead), which happened to go
    // unnoticed while this port's write side was single-dimension-only
    // (`num_index_dims == 1` never exercises the box at all) and was only
    // caught once a real multi-dimension fixture round-tripped through real
    // Lucene. The box is also only present when `compressed_dim != -1`
    // (real Lucene's `visitDocValuesWithCardinality` only calls
    // `readMinMax` inside the non-`-1` branch).
    let compressed_dim = input.read_byte()? as i8;
    if compressed_dim < -2 || i32::from(compressed_dim) >= field.num_dims {
        return Err(Error::UnsupportedCompressedDim(compressed_dim));
    }

    if compressed_dim == -1 {
        // Every point in this leaf has the identical value (common prefixes
        // already cover every byte of every dimension).
        for &doc_id in doc_ids {
            out.point(doc_id, &scratch_value);
        }
        return Ok(None);
    }

    let mut bound: Option<(Vec<u8>, Vec<u8>)> = None;
    if num_index_dims != 1 {
        // The index gives a (possibly looser) per-leaf bounding box for the
        // indexed dimensions when there's more than one; read it (rather
        // than merely skipping past it) so callers -- notably
        // `lucene_index::check_index`'s structural-invariant checker -- can
        // cross-check every point's value against it independently of the
        // field-wide box in `.kdm`.
        let mut min_bound = vec![0u8; field.packed_index_bytes_length()];
        let mut max_bound = vec![0u8; field.packed_index_bytes_length()];
        for (dim, &prefix) in common_prefix_lengths
            .iter()
            .take(num_index_dims)
            .enumerate()
        {
            // ARITH: `dim < num_index_dims`, so `hi <= num_index_dims *
            // bytes_per_dim`, the exact length of both bound buffers and a
            // product bounded by `check_config`'s `num_dims * bytes_per_dim`.
            // `prefix <= bytes_per_dim` was established when the prefix was
            // read, so `mid` sits inside `lo..=hi`.
            #[allow(clippy::arithmetic_side_effects)]
            let (lo, mid, hi) = (
                dim * bytes_per_dim,
                dim * bytes_per_dim + prefix,
                (dim + 1) * bytes_per_dim,
            );
            // The leading `prefix` bytes of both min and max are the leaf's
            // already-decoded common prefix (identical for every point in
            // this leaf, hence not re-sent on the wire for the box either).
            min_bound[lo..mid].copy_from_slice(&scratch_value[lo..mid]);
            max_bound[lo..mid].copy_from_slice(&scratch_value[lo..mid]);
            input.read_bytes(&mut min_bound[mid..hi])?;
            input.read_bytes(&mut max_bound[mid..hi])?;
        }
        bound = Some((min_bound, max_bound));
    }

    if compressed_dim == -2 {
        let mut i = 0usize;
        while i < count {
            let length = input.read_vint()?;
            let Ok(length) = usize::try_from(length) else {
                return Err(corrupt(format!(
                    "negative low-cardinality run length in .kdd leaf: {length}"
                )));
            };
            // ARITH: `i < count` is the loop condition, so `count - i` cannot
            // underflow. Comparing against the remainder rather than forming
            // `i + length` is what keeps a hostile vint from overflowing the
            // sum before the check that would have caught it.
            #[allow(clippy::arithmetic_side_effects)]
            let remaining_points = count - i;
            if length > remaining_points {
                return Err(Error::SubBlockCountMismatch {
                    expected: count,
                    actual: i.saturating_add(length),
                });
            }
            read_suffix_bytes(
                input,
                &common_prefix_lengths,
                bytes_per_dim,
                &mut scratch_value,
            )?;
            // ARITH: `length <= count - i`, so `end <= count`.
            #[allow(clippy::arithmetic_side_effects)]
            let end = i + length;
            for &doc_id in &doc_ids[i..end] {
                out.point(doc_id, &scratch_value);
            }
            i = end;
        }
        debug_assert_eq!(i, count);
    } else {
        let compressed_dim = compressed_dim as usize;
        let prefix = common_prefix_lengths[compressed_dim];
        // `BKDWriter.writeLeafBlockPackedValues` asserts
        // `commonPrefixLengths[sortedDim] < bytesPerDim` before it can pick a
        // non-negative compressed dimension: the run-length-compressed byte is
        // the first byte *after* that dimension's common prefix, so a
        // full-width prefix means there is no such byte. Java reads on
        // regardless -- addressing the next dimension's bytes and then asking
        // `readBytes` for a negative length; here the `+= 1` below would invert
        // every later suffix range instead.
        if prefix >= bytes_per_dim {
            return Err(corrupt(format!(
                "compressed dim {compressed_dim} has a full-width common prefix of {prefix} bytes"
            )));
        }
        // ARITH: `compressed_dim < num_dims` (checked when the marker was
        // read) and `prefix < bytes_per_dim`, so the offset is strictly inside
        // `scratch_value` and `prefix + 1 <= bytes_per_dim` keeps every later
        // suffix range non-inverted.
        #[allow(clippy::arithmetic_side_effects)]
        let compressed_byte_offset = compressed_dim * bytes_per_dim + prefix;
        // ARITH: `prefix < bytes_per_dim` was just checked, so `prefix + 1`
        // is at most `bytes_per_dim` and cannot overflow.
        #[allow(clippy::arithmetic_side_effects)]
        {
            common_prefix_lengths[compressed_dim] = prefix + 1;
        }
        let mut i = 0usize;
        while i < count {
            scratch_value[compressed_byte_offset] = input.read_byte()?;
            let run_len = usize::from(input.read_byte()?);
            // ARITH: `i < count` is the loop condition (see the `-2` branch).
            #[allow(clippy::arithmetic_side_effects)]
            let remaining_points = count - i;
            if run_len > remaining_points {
                return Err(Error::SubBlockCountMismatch {
                    expected: count,
                    actual: i.saturating_add(run_len),
                });
            }
            // ARITH: `run_len <= count - i`, so `end <= count`.
            #[allow(clippy::arithmetic_side_effects)]
            let end = i + run_len;
            for &doc_id in &doc_ids[i..end] {
                read_suffix_bytes(
                    input,
                    &common_prefix_lengths,
                    bytes_per_dim,
                    &mut scratch_value,
                )?;
                out.point(doc_id, &scratch_value);
            }
            i = end;
        }
        debug_assert_eq!(i, count);
    }

    Ok(bound)
}

const CONTINUOUS_IDS: i8 = -2;
const BITSET_IDS: i8 = -1;
const DELTA_BPV_16: i8 = 16;
const BPV_21: i8 = 21;
const BPV_24: i8 = 24;
const BPV_32: i8 = 32;
/// Legacy marker: per Java's own comment on `DocIdsWriter.LEGACY_DELTA_VINT`,
/// "these signs are legacy, should no longer be used in the writing side."
/// No Lucene 10.5.0 writer emits this, but `DocIdsWriter.readInts` still
/// decodes it for backward compatibility with indices written by very old
/// versions, so this port mirrors that read path.
const LEGACY_DELTA_VINT: i8 = 0;

/// Refuses a leaf's doc-id block *before* its buffer is reserved, when the
/// encoding's own fixed cost already exceeds the bytes left in `.kdd`.
///
/// Each of these encodings spends a known (or known-minimum) number of bytes
/// per doc id, so this is exactly the EOF the decode loop would hit a moment
/// later -- except that it happens before `Vec::with_capacity(count)` /
/// `vec![0i32; count]` turns a large count into a multi-gigabyte reservation
/// whose failure **aborts** rather than unwinding. A saturated `needed` (only
/// reachable on a 32-bit target, and only for a count no real leaf carries)
/// always fails the comparison, which is the right answer for such a count.
fn require_bytes(input: &SliceInput, needed: usize, what: &str) -> Result<()> {
    if input.remaining() < needed {
        return Err(corrupt(format!(
            "{what}: {needed} bytes needed but only {} remain",
            input.remaining()
        )));
    }
    Ok(())
}

/// [`read_doc_ids_into`] into a fresh `Vec`, for tests.
#[cfg(test)]
pub(crate) fn read_doc_ids(input: &mut SliceInput, count: usize) -> Result<Vec<i32>> {
    let mut out = Vec::new();
    read_doc_ids_into(input, count, &mut out, BKD_VERSION_CURRENT)?;
    Ok(out)
}

/// Port of `DocIdsWriter.readInts` -- decodes `count` doc ids using
/// whichever encoding the leaf's leading marker byte selects, into a
/// caller-owned buffer whose contents it replaces: Java's reusable
/// `int[maxPointsInLeafNode]`. A range query visits one leaf after another;
/// decoding each into a fresh zeroed `Vec` (two, for the packed encodings) was
/// a third of a large range's cost.
///
/// `count` must have come from [`read_leaf_count`], i.e. be bounded by the
/// field's `maxPointsInLeafNode`; that is what Java's fixed
/// `int[maxPointsInLeafNode]` decode buffer enforces implicitly.
///
/// `bkd_version` is the field's `BKDWriter` version: before
/// `VERSION_VECTORIZE_BPV24_AND_INTRODUCE_BPV21` (10) a `BPV_24` leaf is the
/// scalar big-endian packing [`read_scalar_bpv24`] decodes, as
/// `DocIdsWriter.readInts` dispatches on `version`.
fn read_doc_ids_into(
    input: &mut SliceInput,
    count: usize,
    out: &mut Vec<i32>,
    bkd_version: i32,
) -> Result<()> {
    let bpv = input.read_byte()? as i8;
    out.clear();
    match bpv {
        CONTINUOUS_IDS => {
            let start = input.read_vint()?;
            // `wrapping_add` is Java's own `docIDs[i] = start + i`, plain
            // `int` arithmetic: a `start` near `i32::MAX` yields wrapped
            // (garbage) doc ids there too, and wrapping is what reproduces
            // them instead of trapping. `count as i32` is exact --
            // `read_leaf_count` bounds `count` by `maxPointsInLeafNode`,
            // itself `<= i32::MAX - 16`.
            out.extend((0..count as i32).map(|i| start.wrapping_add(i)));
            Ok(())
        }
        BITSET_IDS => read_bitset_ids(input, count, out),
        DELTA_BPV_16 => read_delta_bpv16(input, count, out),
        BPV_21 => read_bpv21(input, count, out),
        BPV_24 if bkd_version < BKD_VERSION_VECTORIZE_BPV24 => read_scalar_bpv24(input, count, out),
        BPV_24 => read_bpv24(input, count, out),
        BPV_32 => {
            let bytes = take_bytes(input, count.saturating_mul(4), "BPV_32 doc ids")?;
            out.extend(
                bytes
                    .chunks_exact(4)
                    .map(|w| i32::from_le_bytes([w[0], w[1], w[2], w[3]])),
            );
            Ok(())
        }
        LEGACY_DELTA_VINT => read_legacy_delta_vint(input, count, out),
        other => Err(Error::UnsupportedDocIdsEncoding(other)),
    }
}

/// The next `n` bytes of `input` as a slice of the underlying buffer, and
/// `input` moved past them: [`require_bytes`] plus the read, for decoders
/// that unpack whole words straight from the mapped file instead of one
/// checked `read_i32` at a time.
fn take_bytes<'a>(input: &mut SliceInput<'a>, n: usize, what: &str) -> Result<&'a [u8]> {
    require_bytes(input, n, what)?;
    let bytes = &input.as_slice()[..n];
    input.skip(n)?;
    Ok(bytes)
}

/// Port of `DocIdsWriter.readLegacyDeltaVInts`: each doc id is a vint delta
/// from the previous one (starting at 0), the encoding used by index
/// versions that predate `DELTA_BPV_16`/`BPV_21`/`BPV_24`/`BPV_32`. No
/// current writer in this port (or in Lucene 10.5.0) produces this, so it
/// is exercised only by hand-built unit tests, not a real-Lucene fixture.
fn read_legacy_delta_vint(input: &mut SliceInput, count: usize, out: &mut Vec<i32>) -> Result<()> {
    // Every doc id costs at least one vint byte, so this is a true lower
    // bound: it caps the reservation without rejecting anything the decode
    // loop would have accepted.
    require_bytes(input, count, "legacy delta-vint doc ids")?;
    out.reserve(count);
    let mut doc = 0i32;
    for _ in 0..count {
        // Java: `doc += in.readVInt()`, `int` arithmetic that wraps.
        doc = doc.wrapping_add(input.read_vint()?);
        out.push(doc);
    }
    Ok(())
}

fn read_bitset_ids(input: &mut SliceInput, count: usize, out: &mut Vec<i32>) -> Result<()> {
    let offset_words = input.read_vint()?;
    let long_len = input.read_vint()?;
    if offset_words < 0 || long_len < 0 {
        return Err(corrupt(format!(
            "negative bitset doc-id header: offsetWords={offset_words}, longLen={long_len}"
        )));
    }
    // `vec![0i64; long_len]` on an `i32::MAX` vint is a 17 GB reservation --
    // Java grows a reusable `long[]` here and merely risks an
    // `OutOfMemoryError` the caller can catch. `read_i64s` needs exactly
    // eight bytes per word, so the stream itself is the ceiling.
    let long_len = long_len as usize;
    require_bytes(input, long_len.saturating_mul(8), "bitset doc ids")?;
    // The largest doc id this block could name, established once so that the
    // per-set-bit arithmetic below needs no checks at all. Java computes the
    // same base as `offsetWords << 6` and lets it wrap into a negative doc
    // base; `DocIdsWriter.writeIdsAsBitSet` derives `offsetWords` from a real
    // doc id, so no writer can reach past `i32::MAX`, and a wrapped value
    // would reach the visitor looking like a valid document number.
    let doc_base = i64::from(offset_words).saturating_mul(64);
    let past_end = doc_base.saturating_add(i64::from(long_len as i32).saturating_mul(64));
    if past_end > i64::from(i32::MAX) {
        return Err(corrupt(format!(
            "bitset doc ids run to {past_end}, past i32::MAX: \
             offsetWords={offset_words}, longLen={long_len}"
        )));
    }
    let doc_base = doc_base as i32;

    // Read in place off the mapped file, eight bytes per word.
    let words = take_bytes(input, long_len.saturating_mul(8), "bitset doc ids")?;

    // `count` is the writer's own set-bit count, so a well-formed block
    // reserves exactly what it needs; a corrupt one is capped by the number of
    // bits that actually exist.
    out.reserve(count.min(long_len.saturating_mul(64)));
    for (word_idx, word) in words.chunks_exact(8).enumerate() {
        let word = u64::from_le_bytes([
            word[0], word[1], word[2], word[3], word[4], word[5], word[6], word[7],
        ]);
        // ARITH: the guard above established `doc_base + long_len * 64 <=
        // i32::MAX` with `doc_base >= 0`, so `long_len * 64 <= i32::MAX` and
        // `word_idx < long_len` makes `word_idx * 64` fit an `i32`; the sum
        // with `doc_base` is bounded by the same inequality.
        #[allow(clippy::arithmetic_side_effects)]
        let word_base = doc_base + (word_idx as i32) * 64;
        let mut w = word;
        while w != 0 {
            let bit = w.trailing_zeros();
            // ARITH: `bit <= 63` and `word_base + 64 <= i32::MAX` by the same
            // bound, so this stays inside `0..=i32::MAX`.
            #[allow(clippy::arithmetic_side_effects)]
            out.push(word_base + bit as i32);
            // ARITH: `w != 0` is the loop condition, so `w - 1` cannot
            // underflow.
            #[allow(clippy::arithmetic_side_effects)]
            {
                w &= w - 1;
            }
        }
    }
    if out.len() != count {
        return Err(Error::SubBlockCountMismatch {
            expected: count,
            actual: out.len(),
        });
    }
    Ok(())
}

fn read_delta_bpv16(input: &mut SliceInput, count: usize, out: &mut Vec<i32>) -> Result<()> {
    let min = input.read_vint()?;
    // ARITH: `count <= maxPointsInLeafNode <= i32::MAX - 16`
    // (`read_leaf_count`), so neither the halving nor the remainder can
    // overflow. The byte cost is exact: one four-byte word per pair, two more
    // bytes for an odd tail.
    #[allow(clippy::arithmetic_side_effects)]
    let (half, odd) = (count / 2, count % 2);
    let needed = half.saturating_mul(4).saturating_add(odd.saturating_mul(2));
    require_bytes(input, needed, "delta-16 doc ids")?;
    out.resize(count, 0);
    for i in 0..half {
        let word = input.read_i32()?;
        // ARITH: `i < half`, so `i + half < 2 * half <= count = out.len()`.
        // The `wrapping_add`s are Java's `(docId >>> 16) + min` / `(docId &
        // 0xFFFF) + min`, plain `int` arithmetic.
        #[allow(clippy::arithmetic_side_effects)]
        {
            out[i] = (((word as u32) >> 16) as i32).wrapping_add(min);
            out[i + half] = (word & 0xFFFF).wrapping_add(min);
        }
    }
    if odd == 1 {
        // ARITH: an odd `count` is at least 1.
        #[allow(clippy::arithmetic_side_effects)]
        let last = count - 1;
        out[last] = i32::from(input.read_u16()?).wrapping_add(min);
    }
    Ok(())
}

fn floor_to_multiple_of_16(n: usize) -> usize {
    n & !0xF
}

fn read_bpv21(input: &mut SliceInput, count: usize, out: &mut Vec<i32>) -> Result<()> {
    // ARITH: `one_third <= count / 3` (rounding *down* to a multiple of 16),
    // so `num_ints <= 2 * (count / 3) < count` and `tail_start <= count`;
    // `count <= maxPointsInLeafNode <= i32::MAX - 16` by `read_leaf_count`, so
    // no product here comes near `usize`'s ceiling.
    #[allow(clippy::arithmetic_side_effects)]
    let (one_third, num_ints, tail_start) = {
        let one_third = floor_to_multiple_of_16(count / 3);
        (one_third, one_third * 2, one_third * 3)
    };
    // The `num_ints` scratch words are read verbatim, and each of the
    // remaining `count - tail_start` ids costs at least one further byte -- a
    // true lower bound, so a well-formed block is never rejected. It is enough
    // to bound `count` itself by the bytes left, since `num_ints * 4` alone is
    // already ~2.6 bytes per id.
    //
    // ARITH: `tail_start = 3 * one_third <= 3 * (count / 3) <= count`, so the
    // subtraction cannot underflow; the two saturating operations cannot.
    #[allow(clippy::arithmetic_side_effects)]
    let needed = num_ints
        .saturating_mul(4)
        .saturating_add(count - tail_start);
    require_bytes(input, needed, "BPV_21 doc ids")?;
    // The packed words go straight into `out[..num_ints]` (one bulk copy,
    // no scratch array), the ids they carry in their low bits are assembled
    // into `out[num_ints..tail_start]` from them, and only then are the words
    // shifted down in place to the ids in their high bits.
    let words = take_bytes(input, num_ints.saturating_mul(4), "BPV_21 doc id words")?;
    out.resize(count, 0);
    for (slot, w) in out[..num_ints].iter_mut().zip(words.chunks_exact(4)) {
        *slot = i32::from_le_bytes([w[0], w[1], w[2], w[3]]);
    }
    for i in 0..one_third {
        // ARITH: `i < one_third`, so `i + one_third < num_ints`
        // (the packed words' count) and `i + num_ints < 3 * one_third = tail_start <=
        // count = out.len()` (and `num_ints <= count` covers the plain
        // `out[i]` loop above). `<< 11` is applied to a value masked to 11
        // bits, so it cannot leave `i32`.
        #[allow(clippy::arithmetic_side_effects)]
        {
            out[i + num_ints] = (out[i] & 0x7FF) | ((out[i + one_third] & 0x7FF) << 11);
        }
    }
    for slot in &mut out[..num_ints] {
        *slot = ((*slot as u32) >> 11) as i32;
    }

    let mut i = tail_start;
    // ARITH: the loop only runs while `i + 2 < count`, so `i + 2` indexes
    // `out` in range and `i += 3` lands at most on `count + 2`, far below
    // `usize::MAX` for a `count` bounded by `i32::MAX`.
    #[allow(clippy::arithmetic_side_effects)]
    while i + 2 < count {
        let l = input.read_i64()?;
        out[i] = (l & 0x1FFFFF) as i32;
        out[i + 1] = ((l >> 21) & 0x1FFFFF) as i32;
        // Java is `(int) (l >>> 42)`, an *unsigned* shift: the top field is
        // 22 bits wide and zero-extended. A signed `>>` here would turn a
        // corrupt block's negative word into a negative doc id instead of the
        // in-range one Java produces.
        out[i + 2] = ((l as u64) >> 42) as i32;
        i += 3;
    }
    // ARITH: `i < count <= i32::MAX` bounds the increment.
    #[allow(clippy::arithmetic_side_effects)]
    while i < count {
        let lo = i32::from(input.read_u16()?);
        let hi = i32::from(input.read_byte()?);
        // `hi` is a byte, so `hi << 16` is at most 0x00FF_0000.
        out[i] = lo | (hi << 16);
        i += 1;
    }
    Ok(())
}

/// Port of `DocIdsWriter.readScalarInts24`: the `BPV_24` layout BKD versions
/// before 10 wrote -- every eight ids packed big-endian-first into three
/// little-endian `long`s, then each remaining id as a `short` (its high 16
/// bits) followed by a byte (its low 8 bits).
// ARITH: `count / 8` and `count % 8` by a constant; every shift below moves a
// `u64` by a constant under 64 and is masked to at most 24 bits before the
// narrowing cast, and `hi << 8` of a 16-bit value stays under 2^24.
#[allow(clippy::arithmetic_side_effects)]
fn read_scalar_bpv24(input: &mut SliceInput, count: usize, out: &mut Vec<i32>) -> Result<()> {
    let (blocks, tail) = (count / 8, count % 8);
    let needed = blocks
        .saturating_mul(24)
        .saturating_add(tail.saturating_mul(3));
    require_bytes(input, needed, "scalar BPV_24 doc ids")?;
    out.reserve(count);
    // The `3 * blocks` longs (`readLong`, little-endian) straight from the
    // mapped bytes, eight ids per three longs.
    let longs = take_bytes(
        input,
        blocks.saturating_mul(24),
        "scalar BPV_24 doc id longs",
    )?;
    let long_at = |c: &[u8], i: usize| -> u64 {
        let mut b = [0u8; 8];
        b.copy_from_slice(&c[i..i + 8]);
        u64::from_le_bytes(b)
    };
    for c in longs.chunks_exact(24) {
        let (l1, l2, l3) = (long_at(c, 0), long_at(c, 8), long_at(c, 16));
        // Every value is masked (or shifted) down to at most 24 bits before
        // the narrowing cast, so none of these can change sign.
        out.extend_from_slice(&[
            (l1 >> 40) as i32,
            ((l1 >> 16) & 0xff_ffff) as i32,
            (((l1 & 0xffff) << 8) | (l2 >> 56)) as i32,
            ((l2 >> 32) & 0xff_ffff) as i32,
            ((l2 >> 8) & 0xff_ffff) as i32,
            (((l2 & 0xff) << 16) | (l3 >> 48)) as i32,
            ((l3 >> 24) & 0xff_ffff) as i32,
            (l3 & 0xff_ffff) as i32,
        ]);
    }
    for _ in 0..tail {
        let hi = i32::from(input.read_u16()?);
        let lo = i32::from(input.read_byte()?);
        // `hi` is 16 bits, so `hi << 8` is at most 0x00FF_FF00.
        out.push((hi << 8) | lo);
    }
    Ok(())
}

fn read_bpv24(input: &mut SliceInput, count: usize, out: &mut Vec<i32>) -> Result<()> {
    // ARITH: `quarter = count / 4`, so `num_ints = 3 * quarter < count` and
    // `tail_start = 4 * quarter <= count`; `count <= maxPointsInLeafNode <=
    // i32::MAX - 16` by `read_leaf_count`.
    #[allow(clippy::arithmetic_side_effects)]
    let (quarter, num_ints, tail_start) = {
        let quarter = count / 4;
        (quarter, quarter * 3, quarter * 4)
    };
    // Exact: `num_ints` four-byte words, then three bytes for each remaining
    // id.
    //
    // ARITH: `tail_start = 4 * (count / 4) <= count`, so the subtraction
    // cannot underflow; the saturating operations cannot.
    #[allow(clippy::arithmetic_side_effects)]
    let needed = num_ints
        .saturating_mul(4)
        .saturating_add((count - tail_start).saturating_mul(3));
    require_bytes(input, needed, "BPV_24 doc ids")?;
    // As for `BPV_21`: words into `out[..num_ints]`, the low-byte ids
    // assembled from them, then the words shifted down in place.
    let words = take_bytes(input, num_ints.saturating_mul(4), "BPV_24 doc id words")?;
    out.resize(count, 0);
    for (slot, w) in out[..num_ints].iter_mut().zip(words.chunks_exact(4)) {
        *slot = i32::from_le_bytes([w[0], w[1], w[2], w[3]]);
    }
    for i in 0..quarter {
        // ARITH: `i < quarter`, so `i + quarter * 2 < 3 * quarter = num_ints`
        // (the packed words' count) and `i + num_ints < 4 * quarter = tail_start <=
        // count = out.len()`. Every shifted operand is masked to eight bits
        // first, so no shift can leave `i32`.
        #[allow(clippy::arithmetic_side_effects)]
        {
            out[i + num_ints] = (out[i] & 0xFF)
                | ((out[i + quarter] & 0xFF) << 8)
                | ((out[i + quarter * 2] & 0xFF) << 16);
        }
    }
    for slot in &mut out[..num_ints] {
        *slot = ((*slot as u32) >> 8) as i32;
    }

    let mut i = tail_start;
    // ARITH: `i < count <= i32::MAX` bounds the increment.
    #[allow(clippy::arithmetic_side_effects)]
    while i < count {
        let lo = i32::from(input.read_u16()?);
        let hi = i32::from(input.read_byte()?);
        // `hi` is a byte, so `hi << 16` is at most 0x00FF_0000.
        out[i] = lo | (hi << 16);
        i += 1;
    }
    Ok(())
}

/// One field's input to [`write()`]: `(docID, packedValue)` pairs for a field
/// with `num_dims` dimensions of `bytes_per_dim` bytes each (`packedValue`
/// is `num_dims * bytes_per_dim` bytes, each dimension's slice the sortable
/// big-endian encoding `NumericUtils.longToSortableBytes`/
/// `intToSortableBytes` already produce -- this module doesn't do that
/// conversion itself, same division of labor as the read side, which also
/// just hands back raw packed bytes). `num_index_dims` may be less than
/// `num_dims` (the trailing dimensions are data-only, as in
/// `BKDConfig.numIndexDims`).
#[derive(Debug, Clone)]
pub struct WritePointsField {
    pub field_number: i32,
    pub num_dims: i32,
    /// How many of `num_dims` leading dimensions are used to build the tree's
    /// split structure; must be in `1..=num_dims`.
    pub num_index_dims: i32,
    pub bytes_per_dim: i32,
    /// `(docID, packedValue)`, in insertion order -- the order
    /// `PointValuesWriter` buffers them in, which is the order
    /// `BKDWriter.writeField` starts from.
    pub points: Vec<(i32, Vec<u8>)>,
}

/// `Lucene90PointsWriter`'s default `maxMBSortInHeap`
/// (`BKDWriter.DEFAULT_MAX_MB_SORT_IN_HEAP`).
pub const DEFAULT_MAX_MB_SORT_IN_HEAP: f64 = bkd_writer::DEFAULT_MAX_MB_SORT_IN_HEAP;

/// [`write_packed`] for callers holding `(docID, packedValue)` pairs, with
/// the segment's `maxDoc` taken as one past the largest doc id -- right for
/// a segment whose last document has a point, which is what this
/// convenience is for (tests and tools building a points file on its own).
/// A segment writer knows its real `maxDoc` and calls [`write_packed`].
pub fn write(
    fields: &[WritePointsField],
    max_points_in_leaf_node: i32,
    segment_id: &[u8; codec_util::ID_LENGTH],
    segment_suffix: &str,
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let mut packed = Vec::with_capacity(fields.len());
    let mut max_doc = 0i32;
    for field in fields {
        check_write_config(
            field.field_number,
            field.points.len(),
            field.num_dims,
            field.num_index_dims,
            field.bytes_per_dim,
            max_points_in_leaf_node,
        )?;
        // ARITH: `check_write_config` proved both factors positive and their
        // product inside an `i32`, so the `usize` product is exact.
        #[allow(clippy::arithmetic_side_effects)]
        let stride = field.num_dims as usize * field.bytes_per_dim as usize;
        let mut values = Vec::with_capacity(field.points.len().saturating_mul(stride));
        for (i, (doc, value)) in field.points.iter().enumerate() {
            if value.len() != stride {
                return Err(Error::WrongPackedValueLength {
                    field_number: field.field_number,
                    index: i,
                    expected: stride as i32,
                    actual: value.len(),
                });
            }
            values.extend_from_slice(value);
            max_doc = max_doc.max(doc.saturating_add(1));
        }
        packed.push(PackedPointsField {
            field_number: field.field_number,
            num_dims: field.num_dims,
            num_index_dims: field.num_index_dims,
            bytes_per_dim: field.bytes_per_dim,
            docs: field.points.iter().map(|(doc, _)| *doc).collect(),
            values,
        });
    }
    write_packed(
        packed,
        max_points_in_leaf_node,
        max_doc,
        segment_id,
        segment_suffix,
    )
}

/// One field's points in the flat layout `BKDWriter` works on
/// (`MutablePointTree`: a doc-id array beside one contiguous buffer of
/// fixed-width packed values) -- what [`write_packed`] takes. Point `i` is
/// `docs[i]` with value `values[i * stride .. (i + 1) * stride]`, where
/// `stride = num_dims * bytes_per_dim`, in insertion order.
#[derive(Debug, Clone, Default)]
pub struct PackedPointsField {
    pub field_number: i32,
    pub num_dims: i32,
    pub num_index_dims: i32,
    pub bytes_per_dim: i32,
    pub docs: Vec<i32>,
    pub values: Vec<u8>,
}

/// The three outputs of one `Lucene90PointsWriter`, headers written.
struct PointsOutputs {
    meta: Vec<u8>,
    index: Vec<u8>,
    data: Vec<u8>,
}

impl PointsOutputs {
    /// The `Lucene90PointsWriter` constructor: an index header on each file.
    fn new(segment_id: &[u8; codec_util::ID_LENGTH], segment_suffix: &str) -> Self {
        let header = |codec: &str| {
            let mut out = Vec::new();
            codec_util::write_index_header(
                &mut out,
                codec,
                VERSION_CURRENT,
                segment_id,
                segment_suffix,
            );
            out
        };
        PointsOutputs {
            data: header(DATA_CODEC_NAME),
            meta: header(META_CODEC_NAME),
            index: header(INDEX_CODEC_NAME),
        }
    }

    /// The `finalizer` `BKDWriter` hands back: the field number, then the
    /// field's `.kdm` entry and its packed index.
    fn finish_field(
        &mut self,
        field_number: i32,
        writer: &bkd_writer::BkdWriter<'_>,
        plan: Option<bkd_writer::BkdIndexPlan>,
    ) {
        if let Some(plan) = plan {
            self.meta.write_i32(field_number);
            writer.write_index(&mut self.meta, &mut self.index, &plan);
        }
    }

    /// `Lucene90PointsWriter.finish`: the field-list terminator, the index
    /// and data footers, their lengths (footers included) in `.kdm`, and
    /// `.kdm`'s own footer.
    fn finish(mut self) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        self.meta.write_i32(-1);
        codec_util::write_footer(&mut self.index);
        codec_util::write_footer(&mut self.data);
        self.meta.write_i64(self.index.len() as i64);
        self.meta.write_i64(self.data.len() as i64);
        codec_util::write_footer(&mut self.meta);
        (self.meta, self.index, self.data)
    }
}

/// `new BKDConfig(numDims, numIndexDims, bytesPerDim, maxPointsInLeafNode)`
/// for a field that passed [`check_write_config`].
fn bkd_config(
    num_dims: i32,
    num_index_dims: i32,
    bytes_per_dim: i32,
    max_points_in_leaf_node: i32,
) -> Result<bkd_writer::BkdConfig> {
    Ok(bkd_writer::BkdConfig::new(
        num_dims as usize,
        num_index_dims as usize,
        bytes_per_dim as usize,
        max_points_in_leaf_node as usize,
    )?)
}

/// `Lucene90PointsWriter.writeField` over the flush path's buffered points
/// (`PointValuesWriter.flush`, a `MutablePointTree`): each field through
/// [`bkd_writer::BkdWriter::write_field`] -- the byte-identical port of
/// `BKDWriter` (`writeField1Dim` for one dimension, `writeFieldNDims`
/// otherwise), which reorders the points in place. `max_doc` is the
/// segment's: `BKDWriter` sizes its doc-id radix keys by
/// `bitsRequired(maxDoc - 1)`, so it is part of the bytes. Fields are
/// written in the order given (`FieldInfos` order for a real flush); a field
/// with no points is refused ([`Error::EmptyField`]) -- Java never hands one
/// to the writer.
pub fn write_packed(
    fields: Vec<PackedPointsField>,
    max_points_in_leaf_node: i32,
    max_doc: i32,
    segment_id: &[u8; codec_util::ID_LENGTH],
    segment_suffix: &str,
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let mut out = PointsOutputs::new(segment_id, segment_suffix);
    for field in fields {
        check_packed_field(&field, max_points_in_leaf_node)?;
        let config = bkd_config(
            field.num_dims,
            field.num_index_dims,
            field.bytes_per_dim,
            max_points_in_leaf_node,
        )?;
        let mut writer = bkd_writer::BkdWriter::new(
            usize::try_from(max_doc).unwrap_or(0),
            None,
            "",
            config,
            DEFAULT_MAX_MB_SORT_IN_HEAP,
            field.docs.len() as u64,
            bkd_writer::VERSION_CURRENT,
        )?;
        let mut tree = bkd_writer::MutablePointTree::new(config.packed_bytes_length());
        tree.docs = field.docs;
        tree.values = field.values;
        let plan = writer.write_field(&mut out.data, &mut tree)?;
        out.finish_field(field.field_number, &writer, plan);
    }
    Ok(out.finish())
}

/// One source segment's points for one merged field, in its tree's leaf
/// order with doc ids already mapped into the merged segment and deleted
/// documents dropped -- what `BKDWriter.MergeReader` and
/// `PointsWriter.mergeOneField`'s visitor both walk. Point `i` is `docs[i]`
/// with value `values[i * stride..]`.
#[derive(Debug, Clone, Default)]
pub struct PointsMergeSource {
    pub docs: Vec<i32>,
    pub values: Vec<u8>,
}

/// One merged field for [`write_merged`].
#[derive(Debug, Clone, Default)]
pub struct MergePointsField {
    pub field_number: i32,
    pub num_dims: i32,
    pub num_index_dims: i32,
    pub bytes_per_dim: i32,
    /// The contributing sources, in merge order.
    pub sources: Vec<PointsMergeSource>,
    /// `totMaxSize`/`maxPointCount`: the sum of the sources' `size()`,
    /// *including* points of deleted documents -- `BKDWriter` decides from it
    /// whether `add` buffers on heap or spills to a temp file, which decides
    /// the bytes.
    pub max_point_count: u64,
}

/// `Lucene90PointsWriter.merge`: a one-dimensional field (`numDims == 1`)
/// through [`bkd_writer::BkdWriter::merge`] -- a k-way merge of the sources'
/// sorted leaves, as Java's `BKDWriter.merge` -- and every other field
/// through `PointsWriter.mergeOneField`'s shape: each source's points
/// [`add`](bkd_writer::BkdWriter::add)ed in merge order, then
/// [`finish`](bkd_writer::BkdWriter::finish), spilling to temp files in
/// `temp_dir` (named after `temp_prefix`, the segment name, as Java names
/// them) past `maxMBSortInHeap`. A field whose every point was deleted
/// writes nothing, as Java's `finish` returns no finalizer.
#[allow(clippy::too_many_arguments)]
pub fn write_merged(
    fields: Vec<MergePointsField>,
    max_points_in_leaf_node: i32,
    max_doc: i32,
    temp_dir: &dyn lucene_store::directory::Directory,
    temp_prefix: &str,
    segment_id: &[u8; codec_util::ID_LENGTH],
    segment_suffix: &str,
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let mut out = PointsOutputs::new(segment_id, segment_suffix);
    for field in fields {
        let count: usize = field.sources.iter().map(|s| s.docs.len()).sum();
        check_write_config(
            field.field_number,
            count.max(1),
            field.num_dims,
            field.num_index_dims,
            field.bytes_per_dim,
            max_points_in_leaf_node,
        )?;
        let config = bkd_config(
            field.num_dims,
            field.num_index_dims,
            field.bytes_per_dim,
            max_points_in_leaf_node,
        )?;
        let stride = config.packed_bytes_length();
        for source in &field.sources {
            if source.values.len() != source.docs.len().saturating_mul(stride) {
                return Err(Error::PackedValuesLength {
                    field_number: field.field_number,
                    points: source.docs.len(),
                    stride,
                    actual: source.values.len(),
                });
            }
        }
        let mut writer = bkd_writer::BkdWriter::new(
            usize::try_from(max_doc).unwrap_or(0),
            Some(temp_dir),
            temp_prefix,
            config,
            DEFAULT_MAX_MB_SORT_IN_HEAP,
            field.max_point_count,
            bkd_writer::VERSION_CURRENT,
        )?;
        let plan = if field.num_dims == 1 {
            let sources: Vec<Box<dyn Iterator<Item = (Vec<u8>, i32)> + '_>> = field
                .sources
                .iter()
                .map(|s| {
                    Box::new(
                        s.values
                            .chunks_exact(stride)
                            .zip(s.docs.iter().copied())
                            .map(|(v, d)| (v.to_vec(), d)),
                    ) as Box<dyn Iterator<Item = (Vec<u8>, i32)> + '_>
                })
                .collect();
            writer.merge(&mut out.data, sources)?
        } else {
            for source in &field.sources {
                for (value, &doc) in source.values.chunks_exact(stride).zip(&source.docs) {
                    writer.add(value, doc)?;
                }
            }
            writer.finish(&mut out.data)?
        };
        out.finish_field(field.field_number, &writer, plan);
    }
    Ok(out.finish())
}

/// [`check_write_config`] plus the flat buffers' own consistency.
fn check_packed_field(field: &PackedPointsField, max_points_in_leaf_node: i32) -> Result<()> {
    check_write_config(
        field.field_number,
        field.docs.len(),
        field.num_dims,
        field.num_index_dims,
        field.bytes_per_dim,
        max_points_in_leaf_node,
    )?;
    // ARITH: `check_write_config` proved both factors positive and their
    // product inside an `i32`.
    #[allow(clippy::arithmetic_side_effects)]
    let stride = field.num_dims as usize * field.bytes_per_dim as usize;
    if field.values.len() != field.docs.len().saturating_mul(stride) {
        return Err(Error::PackedValuesLength {
            field_number: field.field_number,
            points: field.docs.len(),
            stride,
            actual: field.values.len(),
        });
    }
    Ok(())
}

/// The shape checks every field passes before a byte of it is written, in
/// the order the errors are reported: empty field, `num_index_dims` outside
/// `1..=num_dims`, `BKDConfig`'s bounds, then `PointValues.MAX_NUM_BYTES`.
fn check_write_config(
    field_number: i32,
    count: usize,
    num_dims: i32,
    num_index_dims: i32,
    bytes_per_dim: i32,
    max_points_in_leaf_node: i32,
) -> Result<()> {
    if count == 0 {
        return Err(Error::EmptyField { field_number });
    }
    if num_index_dims < 1 || num_index_dims > num_dims {
        return Err(Error::InvalidNumIndexDims {
            field_number,
            num_dims,
            num_index_dims,
        });
    }
    check_config(
        num_dims,
        num_index_dims,
        bytes_per_dim,
        max_points_in_leaf_node,
    )?;
    // `FieldInfo`'s and `FieldType.setDimensions`'s ceiling, which every
    // `BKDWriter` in Java sits behind. `BKDConfig` itself does not check it,
    // so `check_config` (shared with the read side, which must accept exactly
    // what Java's `BKDReader` accepts) does not either.
    if bytes_per_dim > MAX_NUM_BYTES {
        return Err(Error::InvalidConfig(format!(
            "bytesPerDim must be <= PointValues.MAX_NUM_BYTES (= {MAX_NUM_BYTES}); got {bytes_per_dim}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    // The arithmetic gate is about values read off disk; a test's `i + 1` is
    // not one. See docs/arithmetic-gate.md.
    #![allow(clippy::arithmetic_side_effects)]

    use super::*;

    #[test]
    fn copy_dim_and_refill_copy_every_width() {
        for len in [1usize, 2, 3, 4, 5, 6, 7, 8, 16] {
            let src: Vec<u8> = (1..=len as u8).collect();
            let mut dst = vec![0u8; len];
            copy_dim(&mut dst, &src);
            assert_eq!(dst, src, "copy_dim, {len} bytes");
            // In place once the length matches; refilled when it does not, or
            // when the measurement arm asks for a fresh allocation.
            for (start, reuse) in [
                (vec![9u8; len], true),
                (vec![9u8; len + 1], true),
                (vec![9u8; len], false),
            ] {
                let mut buf = start;
                refill(&mut buf, &src, reuse);
                assert_eq!(buf, src, "refill, {len} bytes, reuse {reuse}");
            }
        }
    }

    fn write_vint(out: &mut Vec<u8>, mut v: i32) {
        loop {
            let mut b = (v & 0x7f) as u8;
            v = ((v as u32) >> 7) as i32;
            if v != 0 {
                b |= 0x80;
                out.push(b);
            } else {
                out.push(b);
                break;
            }
        }
    }

    fn write_vlong(out: &mut Vec<u8>, mut v: i64) {
        loop {
            let mut b = (v & 0x7f) as u8;
            v = ((v as u64) >> 7) as i64;
            if v != 0 {
                b |= 0x80;
                out.push(b);
            } else {
                out.push(b);
                break;
            }
        }
    }

    #[test]
    fn continuous_ids_decode() {
        let mut bytes = vec![CONTINUOUS_IDS as u8];
        write_vint(&mut bytes, 100);
        let mut input = SliceInput::new(&bytes);
        assert_eq!(
            read_doc_ids(&mut input, 5).unwrap(),
            vec![100, 101, 102, 103, 104]
        );
    }

    #[test]
    fn bitset_ids_decode() {
        // docs 2, 5, 130 (offsetWords=0, spans 3 64-bit words).
        let mut bytes = vec![BITSET_IDS as u8];
        write_vint(&mut bytes, 0); // offsetWords
        write_vint(&mut bytes, 3); // longLen
        let mut word0 = 0u64;
        word0 |= 1 << 2;
        word0 |= 1 << 5;
        bytes.extend_from_slice(&word0.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        let mut word2 = 0u64;
        word2 |= 1 << (130 - 128);
        bytes.extend_from_slice(&word2.to_le_bytes());
        let mut input = SliceInput::new(&bytes);
        assert_eq!(read_doc_ids(&mut input, 3).unwrap(), vec![2, 5, 130]);
    }

    #[test]
    fn delta_bpv16_even_count() {
        let min = 1000i32;
        let deltas = [0i32, 5, 2, 9]; // docIds = min+delta
        let mut bytes = vec![DELTA_BPV_16 as u8];
        write_vint(&mut bytes, min);
        let half = deltas.len() / 2;
        for i in 0..half {
            let word = ((deltas[i] as u32) << 16) | (deltas[half + i] as u32);
            bytes.extend_from_slice(&(word as i32).to_le_bytes());
        }
        // out[i] = min+deltas[i], out[i+half] = min+deltas[half+i] -- the
        // pairing is (index, index+half) sharing one packed word, not
        // consecutive indices.
        let mut input = SliceInput::new(&bytes);
        assert_eq!(
            read_doc_ids(&mut input, 4).unwrap(),
            vec![1000, 1005, 1002, 1009]
        );
    }

    #[test]
    fn delta_bpv16_odd_count() {
        let min = 10i32;
        let mut bytes = vec![DELTA_BPV_16 as u8];
        write_vint(&mut bytes, min);
        // count=1: half=0, no packed words, then one trailing u16.
        bytes.extend_from_slice(&7u16.to_le_bytes());
        let mut input = SliceInput::new(&bytes);
        assert_eq!(read_doc_ids(&mut input, 1).unwrap(), vec![17]);
    }

    #[test]
    fn bpv32_decode() {
        let mut bytes = vec![BPV_32 as u8];
        for v in [1i32, 1_000_000, 70_000] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let mut input = SliceInput::new(&bytes);
        assert_eq!(
            read_doc_ids(&mut input, 3).unwrap(),
            vec![1, 1_000_000, 70_000]
        );
    }

    #[test]
    fn unsupported_doc_ids_encoding_rejected() {
        let bytes = [1u8]; // no such marker byte is defined
        let mut input = SliceInput::new(&bytes);
        assert!(matches!(
            read_doc_ids(&mut input, 1),
            Err(Error::UnsupportedDocIdsEncoding(1))
        ));
    }

    #[test]
    fn legacy_delta_vint_decode() {
        // marker 0 (LEGACY_DELTA_VINT): each id is a vint delta from the
        // previous one, starting at 0. Real Lucene 10.5.0 writers never
        // produce this marker (it predates DELTA_BPV_16/BPV_21/BPV_24/
        // BPV_32), so this is a hand-built fixture, not a real-Lucene one.
        let mut bytes = vec![0u8]; // LEGACY_DELTA_VINT
        write_vint(&mut bytes, 5); // doc 0: 0 + 5 = 5
        write_vint(&mut bytes, 3); // doc 1: 5 + 3 = 8
        write_vint(&mut bytes, 100); // doc 2: 8 + 100 = 108
        let mut input = SliceInput::new(&bytes);
        assert_eq!(read_doc_ids(&mut input, 3).unwrap(), vec![5, 8, 108]);
    }

    #[test]
    fn legacy_delta_vint_empty() {
        let bytes = vec![0u8];
        let mut input = SliceInput::new(&bytes);
        assert_eq!(read_doc_ids(&mut input, 0).unwrap(), Vec::<i32>::new());
    }

    /// A single-leaf field (numLeaves=1): the packed index is just the root
    /// FP delta vlong, no split descriptor bytes at all.
    #[test]
    fn single_leaf_decode_leaf_pointers() {
        let field = PointsField {
            num_dims: 1,
            num_index_dims: 1,
            bytes_per_dim: 4,
            max_points_in_leaf_node: 512,
            num_leaves: 1,
            min_packed_value: vec![0; 4],
            max_packed_value: vec![0; 4],
            point_count: 3,
            doc_count: 3,
            index_start_pointer: 0,
            num_index_bytes: 0,
            bkd_version: BKD_VERSION_CURRENT,
        };
        let mut inner = Vec::new();
        write_vlong(&mut inner, 300_000); // large enough to need vlong continuation bytes
        assert_eq!(decode_leaf_pointers(&inner, &field).unwrap(), vec![300_000]);
    }

    /// A 3-leaf field (root splits into leaf 2 (left) and an inner node 3
    /// that splits into leaves 6/7): exercises the recursive descent,
    /// inherited-vs-delta FP baselines, and the `leftNumBytes` skip.
    #[test]
    fn three_leaf_decode_leaf_pointers() {
        let field = PointsField {
            num_dims: 1,
            num_index_dims: 1,
            bytes_per_dim: 4,
            max_points_in_leaf_node: 512,
            num_leaves: 3,
            min_packed_value: vec![0; 4],
            max_packed_value: vec![0; 4],
            point_count: 3,
            doc_count: 3,
            index_start_pointer: 0,
            num_index_bytes: 0,
            bkd_version: BKD_VERSION_CURRENT,
        };
        // leafNodeOffset=3. node1 (root) is not a leaf (1<3); its children
        // are node2 (leaf, 2>=3? no wait 2<3 so node2 is NOT a leaf either;
        // recompute: leafNodeOffset=3 means leaves are nodeId>=3. node1's
        // children are 2,3. node2<3 -> inner; node3>=3 -> leaf.
        // node2's children are 4,5, both >=3 -> leaves.
        let mut inner = Vec::new();
        write_vlong(&mut inner, 100); // root FP baseline (node1)

        // node1 split descriptor: splitDim=0 (numIndexDims=1 so code%1=0
        // always), prefix=4 (== bytesPerDim so suffix=0, no extra bytes).
        // code = splitDim + numIndexDims*(prefix + (1+bytesPerDim)*firstDiffByteDelta)
        // with numIndexDims=1: code = 0 + 1*(prefix + 5*0) = prefix = 4.
        write_vint(&mut inner, 4);
        // left child (node2) < leafNodeOffset(3) -> leftNumBytes follows.
        // node2's own subtree (below) is 5 bytes; set leftNumBytes=5
        // (unused by this port, but must still be present/consumed).
        write_vint(&mut inner, 5);

        // -- node2's subtree (left of root) --
        // node2 split descriptor (same shape as node1's).
        write_vint(&mut inner, 4);
        // node2's left child (node4) >= leafNodeOffset(3) -> no leftNumBytes.
        // node4 (leaf) inherits node2's FP baseline (100, unchanged).
        // node5 (leaf, right child of node2): FP delta.
        write_vlong(&mut inner, 7); // node5 FP = 100+7=107

        // -- node3 (right child of root, a leaf): FP delta from root's 100.
        write_vlong(&mut inner, 50); // node3 FP = 100+50=150

        assert_eq!(
            decode_leaf_pointers(&inner, &field).unwrap(),
            vec![100, 107, 150]
        );
    }

    fn field_1d(bytes_per_dim: i32) -> PointsField {
        PointsField {
            num_dims: 1,
            num_index_dims: 1,
            bytes_per_dim,
            max_points_in_leaf_node: 512,
            num_leaves: 1,
            min_packed_value: vec![0; bytes_per_dim as usize],
            max_packed_value: vec![0; bytes_per_dim as usize],
            point_count: 0,
            doc_count: 0,
            index_start_pointer: 0,
            num_index_bytes: 0,
            bkd_version: BKD_VERSION_CURRENT,
        }
    }

    #[test]
    fn leaf_unique_value_all_points_share_one_value() {
        let field = field_1d(2);
        let mut bytes = Vec::new();
        write_vint(&mut bytes, 3); // count
        bytes.push(CONTINUOUS_IDS as u8);
        write_vint(&mut bytes, 5); // docBase -> docs 5,6,7
        write_vint(&mut bytes, 2); // common prefix = full bytesPerDim
        bytes.extend_from_slice(&[0x12, 0x34]);
        bytes.push(0xFF); // compressedDim = -1 (unique)

        let mut input = SliceInput::new(&bytes);
        let mut out = Vec::new();
        read_leaf_block(&mut input, &field, &mut out).unwrap();
        assert_eq!(out.len(), 3);
        for (i, p) in out.iter().enumerate() {
            assert_eq!(p.doc_id, 5 + i as i32);
            assert_eq!(p.packed_value, vec![0x12, 0x34]);
        }
    }

    #[test]
    fn leaf_sparse_low_cardinality_two_runs() {
        let field = field_1d(1);
        let mut bytes = Vec::new();
        write_vint(&mut bytes, 4); // count
        bytes.push(CONTINUOUS_IDS as u8);
        write_vint(&mut bytes, 10); // docBase -> docs 10,11,12,13
        write_vint(&mut bytes, 0); // common prefix = 0
        bytes.push(0xFE); // compressedDim = -2 (sparse)
                          // run 1: length=2, value=0xAA
        write_vint(&mut bytes, 2);
        bytes.push(0xAA);
        // run 2: length=2, value=0xBB
        write_vint(&mut bytes, 2);
        bytes.push(0xBB);

        let mut input = SliceInput::new(&bytes);
        let mut out = Vec::new();
        read_leaf_block(&mut input, &field, &mut out).unwrap();
        assert_eq!(
            out.iter()
                .map(|p| (p.doc_id, p.packed_value[0]))
                .collect::<Vec<_>>(),
            vec![(10, 0xAA), (11, 0xAA), (12, 0xBB), (13, 0xBB)]
        );
    }

    #[test]
    fn leaf_run_length_compressed_dim() {
        let field = field_1d(2);
        let mut bytes = Vec::new();
        write_vint(&mut bytes, 3); // count
        bytes.push(CONTINUOUS_IDS as u8);
        write_vint(&mut bytes, 0); // docBase -> docs 0,1,2
        write_vint(&mut bytes, 0); // common prefix = 0
        bytes.push(0x00); // compressedDim = 0
        bytes.push(0x01); // run byte value (shared high byte)
        bytes.push(3); // runLen = 3 (all points in one run)
        bytes.push(0x11); // point0 low byte
        bytes.push(0x22); // point1 low byte
        bytes.push(0x33); // point2 low byte

        let mut input = SliceInput::new(&bytes);
        let mut out = Vec::new();
        read_leaf_block(&mut input, &field, &mut out).unwrap();
        assert_eq!(
            out.iter()
                .map(|p| p.packed_value.clone())
                .collect::<Vec<_>>(),
            vec![vec![0x01, 0x11], vec![0x01, 0x22], vec![0x01, 0x33]]
        );
    }

    #[test]
    fn leaf_multi_index_dim_skips_min_max_box() {
        let field = PointsField {
            num_dims: 2,
            num_index_dims: 2,
            bytes_per_dim: 1,
            max_points_in_leaf_node: 512,
            num_leaves: 1,
            min_packed_value: vec![0; 2],
            max_packed_value: vec![0; 2],
            point_count: 0,
            doc_count: 0,
            index_start_pointer: 0,
            num_index_bytes: 0,
            bkd_version: BKD_VERSION_CURRENT,
        };
        let mut bytes = Vec::new();
        write_vint(&mut bytes, 1); // count
        bytes.push(CONTINUOUS_IDS as u8);
        write_vint(&mut bytes, 0); // docBase
        write_vint(&mut bytes, 0); // prefix dim0 = 0
        write_vint(&mut bytes, 0); // prefix dim1 = 0
        bytes.push(0x00); // compressedDim = 0 -- comes *before* the box
                          // (real `BKDReader.visitDocValuesWithCardinality`'s current-version
                          // layout, see `read_leaf_block`'s doc comment).
                          // min/max box (2 dims x (min tail + max tail), 1 byte each) -- values
                          // are irrelevant, just skipped -- only present because
                          // compressedDim != -1.
        bytes.extend_from_slice(&[0xEE, 0xEE, 0xEE, 0xEE]);
        bytes.push(0xAA); // run byte -> dim0's only byte
        bytes.push(1); // runLen = 1
        bytes.push(0xBB); // dim1's suffix byte for the one point

        let mut input = SliceInput::new(&bytes);
        let mut out = Vec::new();
        read_leaf_block(&mut input, &field, &mut out).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].packed_value, vec![0xAA, 0xBB]);
    }

    #[test]
    fn leaf_unsupported_compressed_dim_rejected() {
        let field = field_1d(1);
        let mut bytes = Vec::new();
        write_vint(&mut bytes, 1); // count
        bytes.push(CONTINUOUS_IDS as u8);
        write_vint(&mut bytes, 0);
        write_vint(&mut bytes, 0); // prefix = 0
        bytes.push(5); // compressedDim=5, but numDims=1 -> invalid

        let mut input = SliceInput::new(&bytes);
        let mut out = Vec::new();
        assert!(matches!(
            read_leaf_block(&mut input, &field, &mut out),
            Err(Error::UnsupportedCompressedDim(5))
        ));
    }

    #[test]
    fn leaf_sparse_sub_block_count_mismatch_rejected() {
        let field = field_1d(1);
        let mut bytes = Vec::new();
        write_vint(&mut bytes, 3); // count
        bytes.push(CONTINUOUS_IDS as u8);
        write_vint(&mut bytes, 0);
        write_vint(&mut bytes, 0); // prefix = 0
        bytes.push(0xFE); // compressedDim = -2
        write_vint(&mut bytes, 5); // run length overshoots count(3)
        bytes.push(0xAA);

        let mut input = SliceInput::new(&bytes);
        let mut out = Vec::new();
        assert!(matches!(
            read_leaf_block(&mut input, &field, &mut out),
            Err(Error::SubBlockCountMismatch {
                expected: 3,
                actual: 5
            })
        ));
    }

    #[test]
    fn leaf_compressed_dim_sub_block_count_mismatch_rejected() {
        let field = field_1d(1);
        let mut bytes = Vec::new();
        write_vint(&mut bytes, 3); // count
        bytes.push(CONTINUOUS_IDS as u8);
        write_vint(&mut bytes, 0);
        write_vint(&mut bytes, 0); // prefix = 0
        bytes.push(0x00); // compressedDim = 0
        bytes.push(0xAA); // run byte
        bytes.push(5); // runLen overshoots count(3) -- caught before reading further

        let mut input = SliceInput::new(&bytes);
        let mut out = Vec::new();
        assert!(matches!(
            read_leaf_block(&mut input, &field, &mut out),
            Err(Error::SubBlockCountMismatch {
                expected: 3,
                actual: 5
            })
        ));
    }

    #[test]
    fn bpv21_round_trips() {
        // count=100 makes one_third=32 (nonzero -- exercises the vectorized
        // main loop) with a 4-value remainder split across the triple-pack
        // loop (3 values) and the final scalar tail (1 value).
        let count = 100usize;
        let ids: Vec<i32> = (0..count as i32).map(|i| 1000 + i * 37).collect();
        let mut bytes = vec![BPV_21 as u8];
        write_bpv21_test(&mut bytes, &ids);
        let mut input = SliceInput::new(&bytes);
        assert_eq!(read_doc_ids(&mut input, count).unwrap(), ids);
    }

    /// BKD version 9's `BPV_24` (`DocIdsWriter.writeScalarInts24`'s layout):
    /// eight ids in three little-endian longs, most significant bits first,
    /// then short+byte per remaining id. The same bytes under version 10
    /// decode to different ids, so the version must be what picks the
    /// decoder.
    #[test]
    fn scalar_bpv24_is_bkd_version_9s_layout() {
        let ids: Vec<i32> = (0..11).map(|i| 0x0A_0000 + i * 0x1_0203).collect();
        let mut bytes = vec![BPV_24 as u8];
        let v = |i: usize| ids[i] as u64;
        let l1 = (v(0) << 40) | (v(1) << 16) | (v(2) >> 8);
        let l2 = ((v(2) & 0xff) << 56) | (v(3) << 32) | (v(4) << 8) | (v(5) >> 16);
        let l3 = ((v(5) & 0xffff) << 48) | (v(6) << 24) | v(7);
        for l in [l1, l2, l3] {
            bytes.extend_from_slice(&l.to_le_bytes());
        }
        for &id in &ids[8..] {
            bytes.extend_from_slice(&((id >> 8) as u16).to_le_bytes());
            bytes.push(id as u8);
        }
        let mut out = Vec::new();
        read_doc_ids_into(&mut SliceInput::new(&bytes), ids.len(), &mut out, 9).unwrap();
        assert_eq!(out, ids);
        let mut vectorized = Vec::new();
        read_doc_ids_into(
            &mut SliceInput::new(&bytes),
            ids.len(),
            &mut vectorized,
            BKD_VERSION_CURRENT,
        )
        .unwrap();
        assert_ne!(vectorized, ids);
        // Too few bytes for the ids it claims.
        let mut out = Vec::new();
        assert!(
            read_doc_ids_into(&mut SliceInput::new(&bytes[..20]), ids.len(), &mut out, 9).is_err()
        );
    }

    #[test]
    fn bpv24_round_trips() {
        // count=42 makes quarter=10 (nonzero -- exercises the vectorized
        // main loop) with a 2-value remainder for the final scalar tail.
        let count = 42usize;
        let ids: Vec<i32> = (0..count as i32).map(|i| 100_000 + i * 41).collect();
        let mut bytes = vec![BPV_24 as u8];
        write_bpv24_test(&mut bytes, &ids);
        let mut input = SliceInput::new(&bytes);
        assert_eq!(read_doc_ids(&mut input, count).unwrap(), ids);
    }

    #[test]
    fn bitset_ids_count_mismatch_rejected() {
        let mut bytes = vec![BITSET_IDS as u8];
        write_vint(&mut bytes, 0); // offsetWords
        write_vint(&mut bytes, 1); // longLen
        let word = (1u64 << 2) | (1u64 << 5); // 2 bits set
        bytes.extend_from_slice(&word.to_le_bytes());
        let mut input = SliceInput::new(&bytes);
        // Claim 3 expected, but only 2 bits are set.
        assert!(matches!(
            read_doc_ids(&mut input, 3),
            Err(Error::SubBlockCountMismatch {
                expected: 3,
                actual: 2
            })
        ));
    }

    /// Mirrors `DocIdsWriter.writeDocIds`'s `BPV_21` branch exactly enough to
    /// produce bytes `read_bpv21` can decode -- for test purposes only.
    fn write_bpv21_test(out: &mut Vec<u8>, ids: &[i32]) {
        let count = ids.len();
        let one_third = floor_to_multiple_of_16(count / 3);
        let num_ints = one_third * 2;
        let mut scratch = vec![0i32; num_ints];
        for i in 0..num_ints {
            scratch[i] = ids[i] << 11;
        }
        for i in 0..one_third {
            let long_idx = i + num_ints;
            scratch[i] |= ids[long_idx] & 0x7FF;
            scratch[i + one_third] |= (ids[long_idx] >> 11) & 0x7FF;
        }
        for &v in &scratch {
            out.extend_from_slice(&v.to_le_bytes());
        }
        let mut i = one_third * 3;
        while i + 2 < count {
            let l = (ids[i] as i64) | ((ids[i + 1] as i64) << 21) | ((ids[i + 2] as i64) << 42);
            out.extend_from_slice(&l.to_le_bytes());
            i += 3;
        }
        while i < count {
            out.extend_from_slice(&(ids[i] as u16).to_le_bytes());
            out.push((ids[i] >> 16) as u8);
            i += 1;
        }
    }

    /// Mirrors `DocIdsWriter.writeDocIds`'s vectorized `BPV_24` branch.
    fn write_bpv24_test(out: &mut Vec<u8>, ids: &[i32]) {
        let count = ids.len();
        let quarter = count / 4;
        let num_ints = quarter * 3;
        let mut scratch = vec![0i32; num_ints];
        for i in 0..num_ints {
            scratch[i] = ids[i] << 8;
        }
        for i in 0..quarter {
            let long_idx = i + num_ints;
            scratch[i] |= ids[long_idx] & 0xFF;
            scratch[i + quarter] |= (ids[long_idx] >> 8) & 0xFF;
            scratch[i + quarter * 2] |= (ids[long_idx] >> 16) & 0xFF;
        }
        for &v in &scratch {
            out.extend_from_slice(&v.to_le_bytes());
        }
        let mut i = quarter * 4;
        while i < count {
            out.extend_from_slice(&(ids[i] as u16).to_le_bytes());
            out.push((ids[i] >> 16) as u8);
            i += 1;
        }
    }

    fn id() -> [u8; codec_util::ID_LENGTH] {
        [7u8; codec_util::ID_LENGTH]
    }

    fn write_vint_i32(out: &mut Vec<u8>, v: i32) {
        write_vint(out, v);
    }

    /// Hand-encodes a minimal valid `.kdm`/`.kdi`/`.kdd` triple with zero
    /// fields (meta stream's field loop terminates on the first `-1`).
    fn build_empty_points_index() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        fn write_string(out: &mut Vec<u8>, s: &str) {
            write_vint_i32(out, s.len() as i32);
            out.extend_from_slice(s.as_bytes());
        }
        fn index_header(codec: &str, version: i32) -> Vec<u8> {
            let mut out = Vec::new();
            out.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
            write_string(&mut out, codec);
            out.extend_from_slice(&(version as u32).to_be_bytes());
            out.extend_from_slice(&id());
            out.push(0); // empty suffix
            out
        }
        fn footer(buf: &mut Vec<u8>) {
            buf.extend_from_slice(&codec_util::FOOTER_MAGIC.to_be_bytes());
            buf.extend_from_slice(&0u32.to_be_bytes());
            let checksum = crc32fast::hash(buf) as u64;
            buf.extend_from_slice(&checksum.to_be_bytes());
        }

        let mut kdi = index_header(INDEX_CODEC_NAME, VERSION_CURRENT);
        footer(&mut kdi);
        let mut kdd = index_header(DATA_CODEC_NAME, VERSION_CURRENT);
        footer(&mut kdd);
        let mut kdm = index_header(META_CODEC_NAME, VERSION_CURRENT);
        // Field numbers are a plain 4-byte little-endian i32 (`readInt`),
        // not a vint.
        kdm.extend_from_slice(&(-1i32).to_le_bytes()); // field loop terminator, zero fields
        kdm.extend_from_slice(&0i64.to_le_bytes()); // indexLength
        kdm.extend_from_slice(&0i64.to_le_bytes()); // dataLength
        footer(&mut kdm);

        (kdm, kdi, kdd)
    }

    #[test]
    fn empty_points_index_opens_with_zero_fields() {
        let (kdm, kdi, kdd) = build_empty_points_index();
        let reader = open(&kdm, &kdi, &kdd, &id(), "").unwrap();
        assert!(reader.field(0).is_none());
    }

    #[test]
    fn wrong_segment_id_rejected() {
        let (kdm, kdi, kdd) = build_empty_points_index();
        let wrong_id = [9u8; codec_util::ID_LENGTH];
        assert!(open(&kdm, &kdi, &kdd, &wrong_id, "").is_err());
    }

    #[test]
    fn illegal_field_number_rejected() {
        let (kdm, kdi, kdd) = build_empty_points_index();
        // `build_empty_points_index` writes the field-loop terminator (-1)
        // as the very first bytes after the index header; splice in an
        // illegal (negative, non -1) field number before it instead.
        let mut header = Vec::new();
        header.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
        write_vint_i32(&mut header, META_CODEC_NAME.len() as i32);
        header.extend_from_slice(META_CODEC_NAME.as_bytes());
        header.extend_from_slice(&(VERSION_CURRENT as u32).to_be_bytes());
        header.extend_from_slice(&id());
        header.push(0);
        assert_eq!(&kdm[..header.len()], header.as_slice());

        let mut patched = header;
        patched.extend_from_slice(&(-5i32).to_le_bytes()); // illegal field number
        patched.extend_from_slice(&0i64.to_le_bytes()); // indexLength
        patched.extend_from_slice(&0i64.to_le_bytes()); // dataLength
        patched.extend_from_slice(&codec_util::FOOTER_MAGIC.to_be_bytes());
        patched.extend_from_slice(&0u32.to_be_bytes());
        let checksum = crc32fast::hash(&patched) as u64;
        patched.extend_from_slice(&checksum.to_be_bytes());

        assert!(matches!(
            open(&patched, &kdi, &kdd, &id(), ""),
            Err(Error::IllegalFieldNumber(-5))
        ));
    }

    fn long_sortable_bytes(v: i64) -> Vec<u8> {
        // NumericUtils.longToSortableBytes: flip the sign bit, then big-endian.
        ((v ^ i64::MIN) as u64).to_be_bytes().to_vec()
    }

    #[test]
    fn write_then_read_single_leaf_continuous_ids_round_trips() {
        let points: Vec<(i32, Vec<u8>)> = (0..10)
            .map(|i| (i, long_sortable_bytes((i as i64) * 100 - 500)))
            .collect();
        let field = WritePointsField {
            field_number: 3,
            num_dims: 1,
            num_index_dims: 1,
            bytes_per_dim: 8,
            points: points.clone(),
        };
        let (kdm, kdi, kdd) = write(&[field], 512, &id(), "").unwrap();

        let reader = open(&kdm, &kdi, &kdd, &id(), "").unwrap();
        let meta = reader.field(3).unwrap();
        assert_eq!(meta.num_dims, 1);
        assert_eq!(meta.num_index_dims, 1);
        assert_eq!(meta.bytes_per_dim, 8);
        assert_eq!(meta.num_leaves, 1);
        assert_eq!(meta.point_count, 10);
        assert_eq!(meta.doc_count, 10);
        assert_eq!(meta.max_points_in_leaf_node, 512);

        let mut decoded = reader.decode_all_points(3).unwrap();
        decoded.sort_by_key(|p| p.doc_id);
        let mut expected: Vec<Point> = points
            .into_iter()
            .map(|(doc_id, packed_value)| Point {
                doc_id,
                packed_value,
            })
            .collect();
        expected.sort_by_key(|p| p.doc_id);
        assert_eq!(decoded, expected);
    }

    #[test]
    fn write_then_read_single_leaf_non_continuous_ids_round_trips() {
        // Every third doc skips the field, like GenPoints.java's real fixture
        // -- forces the BPV_32 doc-id path instead of CONTINUOUS_IDS.
        let points: Vec<(i32, Vec<u8>)> = (0..30)
            .filter(|i| i % 3 != 0)
            .map(|i| (i, long_sortable_bytes((i as i64) * 7919 - 1_000_000)))
            .collect();
        let field = WritePointsField {
            field_number: 0,
            num_dims: 1,
            num_index_dims: 1,
            bytes_per_dim: 8,
            points: points.clone(),
        };
        let (kdm, kdi, kdd) = write(&[field], 512, &id(), "").unwrap();

        let reader = open(&kdm, &kdi, &kdd, &id(), "").unwrap();
        let meta = reader.field(0).unwrap();
        assert_eq!(meta.point_count, points.len() as i64);
        assert_eq!(meta.doc_count, points.len() as i32);

        let mut decoded = reader.decode_all_points(0).unwrap();
        decoded.sort_by_key(|p| p.doc_id);
        let mut expected: Vec<Point> = points
            .into_iter()
            .map(|(doc_id, packed_value)| Point {
                doc_id,
                packed_value,
            })
            .collect();
        expected.sort_by_key(|p| p.doc_id);
        assert_eq!(decoded, expected);
    }

    #[test]
    fn write_multiple_fields_round_trips() {
        let field_a = WritePointsField {
            field_number: 0,
            num_dims: 1,
            num_index_dims: 1,
            bytes_per_dim: 4,
            points: vec![
                (0, vec![0, 0, 0, 1]),
                (1, vec![0, 0, 0, 2]),
                (2, vec![0, 0, 0, 3]),
            ],
        };
        let field_b = WritePointsField {
            field_number: 1,
            num_dims: 1,
            num_index_dims: 1,
            bytes_per_dim: 8,
            points: vec![(5, long_sortable_bytes(42)), (7, long_sortable_bytes(-1))],
        };
        let (kdm, kdi, kdd) = write(&[field_a.clone(), field_b.clone()], 512, &id(), "").unwrap();

        let reader = open(&kdm, &kdi, &kdd, &id(), "").unwrap();
        assert!(reader.field(0).is_some());
        assert!(reader.field(1).is_some());

        let mut got_a = reader.decode_all_points(0).unwrap();
        got_a.sort_by_key(|p| p.doc_id);
        assert_eq!(
            got_a,
            vec![
                Point {
                    doc_id: 0,
                    packed_value: vec![0, 0, 0, 1]
                },
                Point {
                    doc_id: 1,
                    packed_value: vec![0, 0, 0, 2]
                },
                Point {
                    doc_id: 2,
                    packed_value: vec![0, 0, 0, 3]
                },
            ]
        );

        let mut got_b = reader.decode_all_points(1).unwrap();
        got_b.sort_by_key(|p| p.doc_id);
        assert_eq!(
            got_b,
            vec![
                Point {
                    doc_id: 5,
                    packed_value: long_sortable_bytes(42)
                },
                Point {
                    doc_id: 7,
                    packed_value: long_sortable_bytes(-1)
                },
            ]
        );
    }

    #[test]
    fn write_single_point_round_trips() {
        let field = WritePointsField {
            field_number: 0,
            num_dims: 1,
            num_index_dims: 1,
            bytes_per_dim: 8,
            points: vec![(9, long_sortable_bytes(123_456_789))],
        };
        let (kdm, kdi, kdd) = write(&[field], 512, &id(), "").unwrap();
        let reader = open(&kdm, &kdi, &kdd, &id(), "").unwrap();
        let decoded = reader.decode_all_points(0).unwrap();
        assert_eq!(
            decoded,
            vec![Point {
                doc_id: 9,
                packed_value: long_sortable_bytes(123_456_789)
            }]
        );
    }

    #[test]
    fn write_then_read_two_leaves_round_trips() {
        // 8 points, max 4 => exactly 2 leaves (numLeftLeafNodes(2) == 1).
        let points: Vec<(i32, Vec<u8>)> = (0..8)
            .map(|i| (i, long_sortable_bytes((i as i64) * 1000)))
            .collect();
        let field = WritePointsField {
            field_number: 0,
            num_dims: 1,
            num_index_dims: 1,
            bytes_per_dim: 8,
            points: points.clone(),
        };
        let (kdm, kdi, kdd) = write(&[field], 4, &id(), "").unwrap();

        let reader = open(&kdm, &kdi, &kdd, &id(), "").unwrap();
        let meta = reader.field(0).unwrap();
        assert_eq!(meta.num_leaves, 2);
        assert_eq!(meta.point_count, 8);

        let mut decoded = reader.decode_all_points(0).unwrap();
        decoded.sort_by_key(|p| p.doc_id);
        let mut expected: Vec<Point> = points
            .into_iter()
            .map(|(doc_id, packed_value)| Point {
                doc_id,
                packed_value,
            })
            .collect();
        expected.sort_by_key(|p| p.doc_id);
        assert_eq!(decoded, expected);
    }

    #[test]
    fn write_then_read_many_leaves_round_trips() {
        // 173 points (deliberately not a multiple of the leaf size,
        // deliberately not a power of two leaf count), max 4/leaf => 44
        // leaves, forcing several levels of recursion, an unbalanced final
        // level, and both the `numLeftLeafNodes == 1` and `> 1` branches of
        // `pack_index`. Every third doc skips the field (like
        // `GenPoints.java`) so doc ids aren't a trivial consecutive run
        // within every leaf.
        let points: Vec<(i32, Vec<u8>)> = (0..300)
            .filter(|i| i % 3 != 0)
            .map(|i| (i, long_sortable_bytes((i as i64) * 7919 - 1_000_000)))
            .collect();
        let expected_leaves = points.len().div_ceil(4) as i32;
        let field = WritePointsField {
            field_number: 0,
            num_dims: 1,
            num_index_dims: 1,
            bytes_per_dim: 8,
            points: points.clone(),
        };
        let (kdm, kdi, kdd) = write(&[field], 4, &id(), "").unwrap();

        let reader = open(&kdm, &kdi, &kdd, &id(), "").unwrap();
        let meta = reader.field(0).unwrap();
        assert_eq!(meta.num_leaves, expected_leaves);
        assert_eq!(meta.point_count, points.len() as i64);
        assert_eq!(meta.doc_count, points.len() as i32);

        let mut decoded = reader.decode_all_points(0).unwrap();
        decoded.sort_by_key(|p| p.doc_id);
        let mut expected: Vec<Point> = points
            .into_iter()
            .map(|(doc_id, packed_value)| Point {
                doc_id,
                packed_value,
            })
            .collect();
        expected.sort_by_key(|p| p.doc_id);
        assert_eq!(decoded, expected);
    }

    #[test]
    fn write_then_read_2d_multi_leaf_round_trips() {
        // LatLonPoint-shaped: 2 dimensions, 4 bytes each. 100 points, max
        // 4/leaf => 25 leaves, exercising multi-dimension split-dimension
        // selection (widest_dim) across several recursion levels together
        // with multi-leaf packing.
        let points: Vec<(i32, Vec<u8>)> = (0..100i32)
            .map(|i| {
                let lat = (i * 37) % 1000; // narrower range
                let lon = (i * 9973) % 1_000_000; // much wider range
                let mut v = Vec::with_capacity(8);
                v.extend_from_slice(&lat.to_be_bytes());
                v.extend_from_slice(&lon.to_be_bytes());
                (i, v)
            })
            .collect();
        let field = WritePointsField {
            field_number: 0,
            num_dims: 2,
            num_index_dims: 2,
            bytes_per_dim: 4,
            points: points.clone(),
        };
        let (kdm, kdi, kdd) = write(&[field], 4, &id(), "").unwrap();

        let reader = open(&kdm, &kdi, &kdd, &id(), "").unwrap();
        let meta = reader.field(0).unwrap();
        assert_eq!(meta.num_dims, 2);
        assert_eq!(meta.num_index_dims, 2);
        assert_eq!(meta.num_leaves, points.len().div_ceil(4) as i32);
        assert_eq!(meta.point_count, points.len() as i64);

        let mut decoded = reader.decode_all_points(0).unwrap();
        decoded.sort_by_key(|p| p.doc_id);
        let mut expected: Vec<Point> = points
            .into_iter()
            .map(|(doc_id, packed_value)| Point {
                doc_id,
                packed_value,
            })
            .collect();
        expected.sort_by_key(|p| p.doc_id);
        assert_eq!(decoded, expected);
    }

    #[test]
    fn write_then_read_3d_multi_leaf_round_trips() {
        // 3 dimensions, 2 bytes each, non-continuous doc ids (every third
        // doc skips the field, like the 1D fixtures above).
        let points: Vec<(i32, Vec<u8>)> = (0..150i32)
            .filter(|i| i % 3 != 0)
            .map(|i| {
                let d0 = ((i * 41) % 500) as u16;
                let d1 = ((i * 173) % 30000) as u16;
                let d2 = ((i * 7) % 60000) as u16;
                let mut v = Vec::with_capacity(6);
                v.extend_from_slice(&d0.to_be_bytes());
                v.extend_from_slice(&d1.to_be_bytes());
                v.extend_from_slice(&d2.to_be_bytes());
                (i, v)
            })
            .collect();
        let expected_leaves = points.len().div_ceil(8) as i32;
        let field = WritePointsField {
            field_number: 3,
            num_dims: 3,
            num_index_dims: 3,
            bytes_per_dim: 2,
            points: points.clone(),
        };
        let (kdm, kdi, kdd) = write(&[field], 8, &id(), "").unwrap();

        let reader = open(&kdm, &kdi, &kdd, &id(), "").unwrap();
        let meta = reader.field(3).unwrap();
        assert_eq!(meta.num_dims, 3);
        assert_eq!(meta.num_index_dims, 3);
        assert_eq!(meta.num_leaves, expected_leaves);
        assert_eq!(meta.point_count, points.len() as i64);
        assert_eq!(meta.doc_count, points.len() as i32);

        let mut decoded = reader.decode_all_points(3).unwrap();
        decoded.sort_by_key(|p| p.doc_id);
        let mut expected: Vec<Point> = points
            .into_iter()
            .map(|(doc_id, packed_value)| Point {
                doc_id,
                packed_value,
            })
            .collect();
        expected.sort_by_key(|p| p.doc_id);
        assert_eq!(decoded, expected);
    }

    #[test]
    fn write_then_read_num_index_dims_less_than_num_dims_round_trips() {
        // 3 dimensions, but only the first 2 are index dims -- like a
        // `LatLonShape`-style bounding box where the 3rd dimension is a
        // non-indexed, data-only payload dimension. Multiple leaves, so both
        // the split-dimension selection (must never pick dim 2) and the
        // per-leaf/per-field bounding boxes (must only cover dims 0-1) get
        // exercised, not just single-leaf storage.
        let points: Vec<(i32, Vec<u8>)> = (0..40i32)
            .map(|i| {
                let d0 = ((i * 41) % 500) as u16;
                let d1 = ((i * 173) % 30000) as u16;
                // Data-only dim: deliberately made *wider-ranging* than the
                // two index dims, so a widest-dim implementation that (wrongly)
                // considered all `num_dims` dimensions instead of only the
                // first `num_index_dims` would pick this one instead.
                let d2 = (i * 1777) as u16;
                let mut v = Vec::with_capacity(6);
                v.extend_from_slice(&d0.to_be_bytes());
                v.extend_from_slice(&d1.to_be_bytes());
                v.extend_from_slice(&d2.to_be_bytes());
                (i, v)
            })
            .collect();
        let expected_leaves = points.len().div_ceil(8) as i32;
        let field = WritePointsField {
            field_number: 4,
            num_dims: 3,
            num_index_dims: 2,
            bytes_per_dim: 2,
            points: points.clone(),
        };
        let (kdm, kdi, kdd) = write(&[field], 8, &id(), "").unwrap();

        let reader = open(&kdm, &kdi, &kdd, &id(), "").unwrap();
        let meta = reader.field(4).unwrap();
        assert_eq!(meta.num_dims, 3);
        assert_eq!(meta.num_index_dims, 2);
        assert_eq!(meta.num_leaves, expected_leaves);
        assert_eq!(meta.point_count, points.len() as i64);
        assert_eq!(meta.doc_count, points.len() as i32);
        // min/max packed value is sized for index dims only (2 * 2 bytes),
        // not the full 3 dims.
        assert_eq!(meta.min_packed_value.len(), 4);
        assert_eq!(meta.max_packed_value.len(), 4);

        let mut decoded = reader.decode_all_points(4).unwrap();
        decoded.sort_by_key(|p| p.doc_id);
        let mut expected: Vec<Point> = points
            .into_iter()
            .map(|(doc_id, packed_value)| Point {
                doc_id,
                packed_value,
            })
            .collect();
        expected.sort_by_key(|p| p.doc_id);
        // Every point's full packed value -- including the non-indexed 3rd
        // dimension -- must survive the round trip unchanged.
        assert_eq!(decoded, expected);
    }

    #[test]
    fn write_then_read_three_leaves_unbalanced_round_trips() {
        // 9 points, max 4 => ceil(9/4) = 3 leaves (numLeftLeafNodes(3) ==
        // 2), exercising the same 2-leaves-left/1-leaf-right shape as the
        // hand-built `three_leaf_decode_leaf_pointers` unit test above, but
        // now produced by the writer instead of hand-encoded.
        let points: Vec<(i32, Vec<u8>)> =
            (0..9).map(|i| (100 + i, vec![(255 - i) as u8])).collect();
        let field = WritePointsField {
            field_number: 7,
            num_dims: 1,
            num_index_dims: 1,
            bytes_per_dim: 1,
            points: points.clone(),
        };
        let (kdm, kdi, kdd) = write(&[field], 4, &id(), "").unwrap();

        let reader = open(&kdm, &kdi, &kdd, &id(), "").unwrap();
        let meta = reader.field(7).unwrap();
        assert_eq!(meta.num_leaves, 3);

        let mut decoded = reader.decode_all_points(7).unwrap();
        decoded.sort_by_key(|p| p.doc_id);
        let mut expected: Vec<Point> = points
            .into_iter()
            .map(|(doc_id, packed_value)| Point {
                doc_id,
                packed_value,
            })
            .collect();
        expected.sort_by_key(|p| p.doc_id);
        assert_eq!(decoded, expected);
    }

    #[test]
    fn write_then_read_exactly_at_max_points_per_leaf_stays_single_leaf() {
        // count == max exactly: must stay a single leaf (num_leaves ==
        // ceil(count / max) == 1), the boundary just below the split trigger.
        let points: Vec<(i32, Vec<u8>)> = (0..4)
            .map(|i| (i, long_sortable_bytes((i as i64) * 1000)))
            .collect();
        let field = WritePointsField {
            field_number: 0,
            num_dims: 1,
            num_index_dims: 1,
            bytes_per_dim: 8,
            points: points.clone(),
        };
        let (kdm, kdi, kdd) = write(&[field], 4, &id(), "").unwrap();
        let reader = open(&kdm, &kdi, &kdd, &id(), "").unwrap();
        let meta = reader.field(0).unwrap();
        assert_eq!(meta.num_leaves, 1);
        assert_eq!(meta.point_count, 4);

        let mut decoded = reader.decode_all_points(0).unwrap();
        decoded.sort_by_key(|p| p.doc_id);
        let mut expected: Vec<Point> = points
            .into_iter()
            .map(|(doc_id, packed_value)| Point {
                doc_id,
                packed_value,
            })
            .collect();
        expected.sort_by_key(|p| p.doc_id);
        assert_eq!(decoded, expected);
    }

    #[test]
    fn write_then_read_one_over_max_points_per_leaf_splits_into_two() {
        // count == max + 1: exactly one point over the threshold must
        // trigger a split into 2 leaves (the classic BKD off-by-one
        // boundary), with the left leaf getting exactly `max` points and the
        // right leaf getting the single leftover point.
        let points: Vec<(i32, Vec<u8>)> = (0..5)
            .map(|i| (i, long_sortable_bytes((i as i64) * 1000)))
            .collect();
        let field = WritePointsField {
            field_number: 0,
            num_dims: 1,
            num_index_dims: 1,
            bytes_per_dim: 8,
            points: points.clone(),
        };
        let (kdm, kdi, kdd) = write(&[field], 4, &id(), "").unwrap();
        let reader = open(&kdm, &kdi, &kdd, &id(), "").unwrap();
        let meta = reader.field(0).unwrap();
        assert_eq!(meta.num_leaves, 2);
        assert_eq!(meta.point_count, 5);

        let mut decoded = reader.decode_all_points(0).unwrap();
        decoded.sort_by_key(|p| p.doc_id);
        let mut expected: Vec<Point> = points
            .into_iter()
            .map(|(doc_id, packed_value)| Point {
                doc_id,
                packed_value,
            })
            .collect();
        expected.sort_by_key(|p| p.doc_id);
        assert_eq!(decoded, expected);
    }

    #[test]
    fn write_then_read_all_points_identical_degenerate_case() {
        // Every point identical in every dimension -- no dimension has any
        // variance to split on. A real BKD tree still must split purely by
        // count (never by value), producing several valid leaves rather
        // than looping forever or panicking. 2 dimensions so widest_dim's
        // all-zero-range tie-break (dimension 0) is actually exercised.
        let value = vec![7u8, 7, 7, 7, 9, 9, 9, 9]; // 2 dims x 4 bytes, identical
        let points: Vec<(i32, Vec<u8>)> = (0..10).map(|i| (i, value.clone())).collect();
        let field = WritePointsField {
            field_number: 0,
            num_dims: 2,
            num_index_dims: 2,
            bytes_per_dim: 4,
            points: points.clone(),
        };
        let (kdm, kdi, kdd) = write(&[field], 4, &id(), "").unwrap();
        let reader = open(&kdm, &kdi, &kdd, &id(), "").unwrap();
        let meta = reader.field(0).unwrap();
        assert_eq!(meta.num_leaves, 3); // ceil(10/4)
        assert_eq!(meta.point_count, 10);

        let mut decoded = reader.decode_all_points(0).unwrap();
        decoded.sort_by_key(|p| p.doc_id);
        let mut expected: Vec<Point> = points
            .into_iter()
            .map(|(doc_id, packed_value)| Point {
                doc_id,
                packed_value,
            })
            .collect();
        expected.sort_by_key(|p| p.doc_id);
        assert_eq!(decoded, expected);
    }

    #[test]
    fn write_then_read_last_dim_only_varies_multi_leaf_round_trips() {
        // Full write/read round-trip of the same shape as
        // `widest_dim_picks_last_dim_when_only_it_varies` above, but through
        // the real writer with enough points to force multiple leaves --
        // proves compute_leaf_plan actually splits on dimension 2 at every
        // level rather than stalling because dims 0/1 look unsplittable.
        let points: Vec<(i32, Vec<u8>)> =
            (0..40i32).map(|i| (i, vec![5, 5, (i * 3) as u8])).collect();
        let field = WritePointsField {
            field_number: 0,
            num_dims: 3,
            num_index_dims: 3,
            bytes_per_dim: 1,
            points: points.clone(),
        };
        let (kdm, kdi, kdd) = write(&[field], 4, &id(), "").unwrap();
        let reader = open(&kdm, &kdi, &kdd, &id(), "").unwrap();
        let meta = reader.field(0).unwrap();
        assert_eq!(meta.num_leaves, 10); // ceil(40/4)
        assert_eq!(meta.point_count, 40);

        let mut decoded = reader.decode_all_points(0).unwrap();
        decoded.sort_by_key(|p| p.doc_id);
        let mut expected: Vec<Point> = points
            .into_iter()
            .map(|(doc_id, packed_value)| Point {
                doc_id,
                packed_value,
            })
            .collect();
        expected.sort_by_key(|p| p.doc_id);
        assert_eq!(decoded, expected);
    }

    #[test]
    fn write_packed_rejects_values_that_do_not_fill_the_points() {
        for len in [7, 9, 0] {
            let field = PackedPointsField {
                field_number: 3,
                num_dims: 1,
                num_index_dims: 1,
                bytes_per_dim: 4,
                docs: vec![0, 1],
                values: vec![0; len],
            };
            assert!(
                matches!(
                    write_packed(vec![field], 512, 2, &id(), ""),
                    Err(Error::PackedValuesLength {
                        field_number: 3,
                        points: 2,
                        stride: 4,
                        actual,
                    }) if actual == len
                ),
                "values of {len} bytes"
            );
        }
    }

    #[test]
    fn write_rejects_empty_field() {
        let field = WritePointsField {
            field_number: 0,
            num_dims: 1,
            num_index_dims: 1,
            bytes_per_dim: 4,
            points: vec![],
        };
        assert!(matches!(
            write(&[field], 512, &id(), ""),
            Err(Error::EmptyField { field_number: 0 })
        ));
    }

    #[test]
    fn write_rejects_num_index_dims_zero() {
        let field = WritePointsField {
            field_number: 0,
            num_dims: 2,
            num_index_dims: 0,
            bytes_per_dim: 4,
            points: vec![(0, vec![0u8; 8])],
        };
        assert!(matches!(
            write(&[field], 512, &id(), ""),
            Err(Error::InvalidNumIndexDims {
                field_number: 0,
                num_dims: 2,
                num_index_dims: 0,
            })
        ));
    }

    #[test]
    fn write_rejects_num_index_dims_greater_than_num_dims() {
        let field = WritePointsField {
            field_number: 0,
            num_dims: 2,
            num_index_dims: 3,
            bytes_per_dim: 4,
            points: vec![(0, vec![0u8; 8])],
        };
        assert!(matches!(
            write(&[field], 512, &id(), ""),
            Err(Error::InvalidNumIndexDims {
                field_number: 0,
                num_dims: 2,
                num_index_dims: 3,
            })
        ));
    }

    #[test]
    fn write_rejects_wrong_packed_value_length() {
        let field = WritePointsField {
            field_number: 0,
            num_dims: 1,
            num_index_dims: 1,
            bytes_per_dim: 8,
            points: vec![(0, vec![1, 2, 3])],
        };
        assert!(matches!(
            write(&[field], 512, &id(), ""),
            Err(Error::WrongPackedValueLength {
                field_number: 0,
                index: 0,
                expected: 8,
                actual: 3,
            })
        ));
    }

    #[test]
    fn write_then_read_rejects_wrong_segment_id() {
        let field = WritePointsField {
            field_number: 0,
            num_dims: 1,
            num_index_dims: 1,
            bytes_per_dim: 4,
            points: vec![(0, vec![0, 0, 0, 1])],
        };
        let (kdm, kdi, kdd) = write(&[field], 512, &id(), "").unwrap();
        let wrong_id = [9u8; codec_util::ID_LENGTH];
        assert!(open(&kdm, &kdi, &kdd, &wrong_id, "").is_err());
    }

    // ------------------------------------------------------------------
    // Arithmetic-gate controls: values off a `.kdd` leaf block.
    // ------------------------------------------------------------------

    /// Java's `DocIdsWriter.readInts` decodes into
    /// `BKDReaderDocIDSetIterator.docIDs`, a `new int[maxPointsInLeafNode]`
    /// allocated once, so a larger `count` throws there before a doc id lands.
    /// This port allocates per leaf, so the same vint sized a fresh `Vec`: a
    /// *negative* one became `usize::MAX` and `Vec::with_capacity` panicked
    /// with "capacity overflow", and a merely huge one is the abort shape.
    #[test]
    fn leaf_count_outside_max_points_in_leaf_node_is_a_decode_error() {
        let field = field_1d(1);
        for count in [-1i32, field.max_points_in_leaf_node + 1, i32::MAX] {
            let mut bytes = Vec::new();
            write_vint(&mut bytes, count);
            bytes.push(BPV_32 as u8);
            let mut input = SliceInput::new(&bytes);
            let mut out = Vec::new();
            let err = read_leaf_block(&mut input, &field, &mut out).unwrap_err();
            assert!(
                format!("{err}").contains("maxPointsInLeafNode"),
                "count={count}: {err}"
            );
        }
    }

    /// A leaf common prefix wider than the dimension it belongs to.
    /// `BKDWriter` computes it as the common prefix of that dimension's
    /// `bytesPerDim` bytes, so no writer emits one; Java's `readBytes` spills
    /// into the *next* dimension's bytes, and here the slice bound leaves
    /// `scratch_value` outright.
    #[test]
    fn leaf_common_prefix_outside_bytes_per_dim_is_a_decode_error() {
        let field = field_1d(2);
        for prefix in [3i32, -1] {
            let mut bytes = Vec::new();
            write_vint(&mut bytes, 1); // count
            bytes.push(CONTINUOUS_IDS as u8);
            write_vint(&mut bytes, 0); // docBase
            write_vint(&mut bytes, prefix);
            bytes.extend_from_slice(&[0u8; 8]);
            let mut input = SliceInput::new(&bytes);
            let mut out = Vec::new();
            let err = read_leaf_block(&mut input, &field, &mut out).unwrap_err();
            assert!(
                format!("{err}").contains("past bytesPerDim"),
                "prefix={prefix}: {err}"
            );
        }
    }

    /// `BKDWriter.writeLeafBlockPackedValues` asserts
    /// `commonPrefixLengths[sortedDim] < bytesPerDim` before it can pick a
    /// non-negative compressed dimension -- the run-length-compressed byte is
    /// the first byte *after* that prefix, so a full-width prefix means there
    /// is no such byte. Without the check `compressed_byte_offset` addressed
    /// one past `scratch_value`.
    #[test]
    fn full_width_common_prefix_on_the_compressed_dim_is_a_decode_error() {
        let field = field_1d(2);
        let mut bytes = Vec::new();
        write_vint(&mut bytes, 1); // count
        bytes.push(CONTINUOUS_IDS as u8);
        write_vint(&mut bytes, 0); // docBase
        write_vint(&mut bytes, 2); // common prefix == bytesPerDim
        bytes.extend_from_slice(&[0xAA, 0xBB]);
        bytes.push(0x00); // compressedDim = 0
        bytes.push(0x01); // run byte
        bytes.push(1); // runLen
        let mut input = SliceInput::new(&bytes);
        let mut out = Vec::new();
        let err = read_leaf_block(&mut input, &field, &mut out).unwrap_err();
        assert!(
            format!("{err}").contains("full-width common prefix"),
            "{err}"
        );
    }

    /// A negative sub-block length in the low-cardinality (`-2`) layout. The
    /// `i + length > count` guard was supposed to catch it, but `i + length`
    /// overflowed first once `i` had advanced past the first run.
    #[test]
    fn negative_low_cardinality_run_length_is_a_decode_error() {
        let field = field_1d(1);
        let mut bytes = Vec::new();
        write_vint(&mut bytes, 4); // count
        bytes.push(CONTINUOUS_IDS as u8);
        write_vint(&mut bytes, 0); // docBase
        write_vint(&mut bytes, 0); // common prefix
        bytes.push(0xFE); // compressedDim = -2
        write_vint(&mut bytes, 2); // first run: length 2
        bytes.push(0xAA);
        write_vint(&mut bytes, -1); // second run: negative length
        let mut input = SliceInput::new(&bytes);
        let mut out = Vec::new();
        let err = read_leaf_block(&mut input, &field, &mut out).unwrap_err();
        assert!(
            format!("{err}").contains("negative low-cardinality run length"),
            "{err}"
        );
    }

    /// `read_bitset_ids`' three unbounded header values: `longLen` sized
    /// `vec![0i64; n]` with nothing between it and the allocator, and
    /// `offsetWords * 64` overflowed an `i32` on the way to the doc base.
    #[test]
    fn corrupt_bitset_doc_id_header_is_a_decode_error() {
        // Negative `longLen` -> `vec![0i64; usize::MAX]`.
        let mut bytes = Vec::new();
        bytes.push(BITSET_IDS as u8);
        write_vint(&mut bytes, 0);
        write_vint(&mut bytes, -1);
        let err = read_doc_ids(&mut SliceInput::new(&bytes), 1).unwrap_err();
        assert!(
            format!("{err}").contains("negative bitset doc-id header"),
            "{err}"
        );

        // A plausible-but-absurd `longLen`: 2^28 words is a 2 GB reservation
        // out of a nine-byte block.
        let mut bytes = Vec::new();
        bytes.push(BITSET_IDS as u8);
        write_vint(&mut bytes, 0);
        write_vint(&mut bytes, 1 << 28);
        let err = read_doc_ids(&mut SliceInput::new(&bytes), 1).unwrap_err();
        assert!(format!("{err}").contains("bitset doc ids"), "{err}");

        // `offsetWords * 64` past `i32::MAX`.
        let mut bytes = Vec::new();
        bytes.push(BITSET_IDS as u8);
        write_vint(&mut bytes, i32::MAX);
        write_vint(&mut bytes, 1);
        bytes.extend_from_slice(&1i64.to_le_bytes());
        let err = read_doc_ids(&mut SliceInput::new(&bytes), 1).unwrap_err();
        assert!(format!("{err}").contains("past i32::MAX"), "{err}");
    }

    /// Java decodes BPV_21's top field as `(int) (l >>> 42)`, an *unsigned*
    /// shift over a 22-bit field. A signed `>>` turned a corrupt block's
    /// negative word into a negative doc id instead of the in-range value
    /// Java produces -- a silently different answer, not a rejected one.
    #[test]
    fn bpv21_top_field_is_zero_extended_like_java() {
        let mut bytes = Vec::new();
        bytes.push(BPV_21 as u8);
        bytes.extend_from_slice(&(-1i64).to_le_bytes());
        let ids = read_doc_ids(&mut SliceInput::new(&bytes), 3).unwrap();
        assert_eq!(ids, vec![0x1F_FFFF, 0x1F_FFFF, 0x3F_FFFF]);
    }

    /// A doc-id block whose fixed per-id cost already exceeds the bytes left
    /// is refused before its buffer is reserved, not after.
    #[test]
    fn doc_id_blocks_are_bounded_by_the_bytes_that_remain() {
        for (marker, needle) in [
            (BPV_32, "BPV_32 doc ids"),
            (BPV_24, "BPV_24 doc ids"),
            (BPV_21, "BPV_21 doc ids"),
            (DELTA_BPV_16, "delta-16 doc ids"),
            (LEGACY_DELTA_VINT, "legacy delta-vint doc ids"),
        ] {
            let mut bytes = vec![marker as u8];
            bytes.extend_from_slice(&[0u8; 8]);
            let err = read_doc_ids(&mut SliceInput::new(&bytes), 100_000).unwrap_err();
            assert!(format!("{err}").contains(needle), "{marker}: {err}");
        }
    }

    /// The reservation cap that keeps a `.kdm`'s `numLeaves` from asking for
    /// 17 GB. A well-formed file is unaffected: the `min` picks `numLeaves`.
    #[test]
    fn leaf_pointer_reservation_is_capped_by_the_packed_index_length() {
        assert_eq!(leaf_pointer_capacity(3, 64), 3);
        assert_eq!(leaf_pointer_capacity(1, 1), 1);
        // 2^31 leaves x 8 bytes = 17 GB, out of a 40-byte packed index.
        assert_eq!(leaf_pointer_capacity(i32::MAX, 40), 41);
    }

    /// The packed tree's node ids double at every level, so an `i32` runs out
    /// after 31 of them. Java lets `nodeID * 2` wrap; a wrapped id compares
    /// `< numLeaves` again, and since the walk only stops at a leaf the
    /// recursion is then bounded by nothing but the `.kdi`'s length -- i.e. a
    /// large enough packed index overflows the *stack*. Checking the multiply
    /// caps the depth at 31 instead.
    #[test]
    fn a_packed_index_deeper_than_the_node_id_space_is_a_decode_error() {
        let mut field = field_1d(1);
        field.num_leaves = i32::MAX;
        // Root FP-delta vlong, then a chain of `(code=0, leftNumBytes=0)`
        // inner nodes at two bytes each -- 63 levels' worth, twice what it
        // takes to drive `node_id` past 2^30.
        let inner_nodes = vec![0u8; 128];
        let err = decode_leaf_pointers(&inner_nodes, &field).unwrap_err();
        assert!(format!("{err}").contains("node id overflows"), "{err}");
    }

    /// `PointValues.MAX_NUM_BYTES`, the ceiling `FieldInfo` and
    /// `FieldType.setDimensions` put on every point field Java can index.
    /// `BKDConfig` does not check it, so the read side does not either; the
    /// write side refuses what `FieldInfo` would.
    #[test]
    fn write_rejects_bytes_per_dim_past_max_num_bytes() {
        let field = WritePointsField {
            field_number: 0,
            num_dims: 1,
            num_index_dims: 1,
            bytes_per_dim: MAX_NUM_BYTES + 1,
            points: vec![(0, vec![0u8; (MAX_NUM_BYTES + 1) as usize])],
        };
        match write(&[field], 512, &id(), "") {
            Err(Error::InvalidConfig(msg)) => {
                assert!(msg.contains("MAX_NUM_BYTES"), "{msg}")
            }
            other => panic!("expected InvalidConfig, got {other:?}"),
        }
    }
}

#[cfg(test)]
mod config_validation_tests {
    // The arithmetic gate is about values read off disk; a test's `i + 1` is
    // not one. See docs/arithmetic-gate.md.
    #![allow(clippy::arithmetic_side_effects)]

    use super::*;

    fn id() -> [u8; codec_util::ID_LENGTH] {
        [3u8; codec_util::ID_LENGTH]
    }

    fn long_sortable_bytes(v: i64) -> Vec<u8> {
        ((v as u64) ^ 0x8000_0000_0000_0000).to_be_bytes().to_vec()
    }

    /// A valid 1-dim / 8-bytes-per-dim / 4-points-per-leaf index, plus the
    /// `.kdm` offset of its per-field BKD shape (`numDims`), so a test can
    /// corrupt exactly one field of it. Every shape value is small enough to
    /// be a one-byte vint, so the layout after that offset is
    /// `numDims, numIndexDims, maxPointsInLeafNode, bytesPerDim, numLeaves,
    /// minPackedValue[8], maxPackedValue[8]`.
    fn valid_index_and_shape_offset() -> (Vec<u8>, Vec<u8>, Vec<u8>, usize) {
        let points: Vec<(i32, Vec<u8>)> = (0..9)
            .map(|i| (i, long_sortable_bytes(i as i64 * 100)))
            .collect();
        let field = WritePointsField {
            field_number: 0,
            num_dims: 1,
            num_index_dims: 1,
            bytes_per_dim: 8,
            points,
        };
        let (kdm, kdi, kdd) = write(&[field], 4, &id(), "").unwrap();
        // Locate the BKD plain header (`write_header`: magic + "BKD" +
        // version) and step past it.
        let mut needle = Vec::new();
        codec_util::write_header(&mut needle, BKD_CODEC_NAME, BKD_VERSION_CURRENT);
        let pos = kdm
            .windows(needle.len())
            .position(|w| w == needle.as_slice())
            .expect("BKD header present in .kdm");
        (kdm, kdi, kdd, pos + needle.len())
    }

    fn expect_config_error(kdm: Vec<u8>, kdi: Vec<u8>, kdd: Vec<u8>, needle: &str) {
        match open(&kdm, &kdi, &kdd, &id(), "") {
            Err(Error::InvalidConfig(msg)) => {
                assert!(msg.contains(needle), "unexpected message: {msg}")
            }
            Err(other) => panic!("expected InvalidConfig containing {needle:?}, got {other:?}"),
            Ok(_) => panic!("expected InvalidConfig containing {needle:?}, got Ok"),
        }
    }

    #[test]
    fn sanity_baseline_index_opens() {
        let (kdm, kdi, kdd, _) = valid_index_and_shape_offset();
        let reader = open(&kdm, &kdi, &kdd, &id(), "").unwrap();
        assert_eq!(reader.field(0).unwrap().num_leaves, 3);
    }

    #[test]
    fn num_dims_zero_rejected() {
        let (mut kdm, kdi, kdd, off) = valid_index_and_shape_offset();
        kdm[off] = 0;
        expect_config_error(kdm, kdi, kdd, "numDims must be 1 .. 16");
    }

    #[test]
    fn num_dims_above_max_rejected() {
        let (mut kdm, kdi, kdd, off) = valid_index_and_shape_offset();
        kdm[off] = 17;
        expect_config_error(kdm, kdi, kdd, "numDims must be 1 .. 16");
    }

    #[test]
    fn num_index_dims_above_max_rejected() {
        let (mut kdm, kdi, kdd, off) = valid_index_and_shape_offset();
        kdm[off] = 16; // numDims = 16, so numIndexDims = 9 is only capped by MAX_INDEX_DIMS
        kdm[off + 1] = 9;
        expect_config_error(kdm, kdi, kdd, "numIndexDims must be 1 .. 8");
    }

    #[test]
    fn num_index_dims_exceeding_num_dims_rejected() {
        let (mut kdm, kdi, kdd, off) = valid_index_and_shape_offset();
        kdm[off] = 2;
        kdm[off + 1] = 3;
        expect_config_error(kdm, kdi, kdd, "numIndexDims cannot exceed numDims");
    }

    #[test]
    fn max_points_in_leaf_node_zero_rejected() {
        let (mut kdm, kdi, kdd, off) = valid_index_and_shape_offset();
        kdm[off + 2] = 0;
        expect_config_error(kdm, kdi, kdd, "maxPointsInLeafNode must be > 0");
    }

    #[test]
    fn bytes_per_dim_zero_rejected() {
        let (mut kdm, kdi, kdd, off) = valid_index_and_shape_offset();
        kdm[off + 3] = 0;
        expect_config_error(kdm, kdi, kdd, "bytesPerDim must be > 0");
    }

    /// A five-byte vint for `bytesPerDim`, spliced in place of the one-byte
    /// one the writer emits. Everything before it is unchanged and
    /// `check_config` runs before the next field is read, so the rest of the
    /// `.kdm` never matters.
    fn splice_huge_bytes_per_dim(kdm: &mut Vec<u8>, off: usize) {
        // vint(2^30) = 0x80 0x80 0x80 0x80 0x04.
        kdm.splice(
            off + 3..off + 4,
            [0x80u8, 0x80, 0x80, 0x80, 0x04].iter().copied(),
        );
    }

    /// `numDims x bytesPerDim` overflowing an `i32`.
    ///
    /// Lucene bounds `numDims` (1..=16) and `numIndexDims` (1..=8) but puts
    /// **no upper bound at all** on `bytesPerDim` -- `BKDConfig`'s constructor
    /// checks only `bytesPerDim > 0`. In Java the product then wraps an `int`
    /// and `new byte[negative]` throws `NegativeArraySizeException`, which is
    /// caught and reported as corruption. In Rust the same multiplication is
    /// a **panic** in a debug build, which through the FFI is not a reported
    /// corruption but a dead JVM.
    #[test]
    fn num_dims_times_bytes_per_dim_overflowing_is_rejected() {
        let (mut kdm, kdi, kdd, off) = valid_index_and_shape_offset();
        kdm[off] = 8; // numDims
        kdm[off + 1] = 8; // numIndexDims
        splice_huge_bytes_per_dim(&mut kdm, off);
        expect_config_error(kdm, kdi, kdd, "overflows");
    }

    /// The same field read off disk, one step further out: a product that
    /// does *not* overflow but is far larger than the file. `numIndexDims=1,
    /// bytesPerDim=2^30` asks `vec![0u8; n]` for two 1 GB buffers out of a
    /// few hundred bytes of `.kdm`. A failed allocation **aborts**, and no
    /// `catch_unwind` at the FFI boundary can intercept an abort -- so this
    /// has to be refused before the allocation, not after.
    #[test]
    fn a_packed_value_length_larger_than_the_kdm_is_rejected_before_allocating() {
        let (mut kdm, kdi, kdd, off) = valid_index_and_shape_offset();
        splice_huge_bytes_per_dim(&mut kdm, off);
        expect_config_error(kdm, kdi, kdd, "bytes of .kdm remain");
    }

    /// `BKDConfig`'s `maxPointsInLeafNode > ArrayUtil.MAX_ARRAY_LENGTH` guard,
    /// the one this port had not reproduced. `maxPointsInLeafNode` sizes the
    /// per-leaf point buffer, so an absurd value is an allocation request the
    /// `.kdm` has no bytes to back.
    #[test]
    fn max_points_in_leaf_node_above_the_array_ceiling_is_rejected() {
        let (mut kdm, kdi, kdd, off) = valid_index_and_shape_offset();
        // vint(i32::MAX) = 0xFF 0xFF 0xFF 0xFF 0x07, spliced over the
        // one-byte `maxPointsInLeafNode` the writer emitted.
        kdm.splice(
            off + 2..off + 3,
            [0xFFu8, 0xFF, 0xFF, 0xFF, 0x07].iter().copied(),
        );
        expect_config_error(kdm, kdi, kdd, "maxPointsInLeafNode must be <=");
    }

    #[test]
    fn num_leaves_zero_rejected() {
        let (mut kdm, kdi, kdd, off) = valid_index_and_shape_offset();
        kdm[off + 4] = 0;
        expect_config_error(kdm, kdi, kdd, "numLeaves must be > 0");
    }

    /// `BKDReader`: "minPackedValue ... is > maxPackedValue ... for dim=N".
    /// Without this check the field would open and every query bound would
    /// be nonsense.
    #[test]
    fn min_packed_value_greater_than_max_rejected() {
        let (mut kdm, kdi, kdd, off) = valid_index_and_shape_offset();
        for b in kdm.iter_mut().skip(off + 5).take(8) {
            *b = 0xFF;
        }
        assert!(matches!(
            open(&kdm, &kdi, &kdd, &id(), ""),
            Err(Error::MinGreaterThanMax(0))
        ));
    }

    /// `maxPointsInLeafNode == 0`, which `BKDConfig` refuses.
    #[test]
    fn write_rejects_zero_max_points_in_leaf_node() {
        let field = WritePointsField {
            field_number: 0,
            num_dims: 1,
            num_index_dims: 1,
            bytes_per_dim: 8,
            points: vec![(0, long_sortable_bytes(1))],
        };
        assert!(matches!(
            write(&[field], 0, &id(), ""),
            Err(Error::InvalidConfig(_))
        ));
    }

    #[test]
    fn write_rejects_zero_bytes_per_dim() {
        let field = WritePointsField {
            field_number: 0,
            num_dims: 1,
            num_index_dims: 1,
            bytes_per_dim: 0,
            points: vec![(0, Vec::new())],
        };
        assert!(matches!(
            write(&[field], 4, &id(), ""),
            Err(Error::InvalidConfig(_))
        ));
    }

    #[test]
    fn write_rejects_too_many_dims() {
        let field = WritePointsField {
            field_number: 0,
            num_dims: 17,
            num_index_dims: 1,
            bytes_per_dim: 1,
            points: vec![(0, vec![0u8; 17])],
        };
        assert!(matches!(
            write(&[field], 4, &id(), ""),
            Err(Error::InvalidConfig(_))
        ));
    }

    #[test]
    fn write_rejects_too_many_index_dims() {
        let field = WritePointsField {
            field_number: 0,
            num_dims: 12,
            num_index_dims: 9,
            bytes_per_dim: 1,
            points: vec![(0, vec![0u8; 12])],
        };
        assert!(matches!(
            write(&[field], 4, &id(), ""),
            Err(Error::InvalidConfig(_))
        ));
    }

    // ------------------------------------------------------------------
    // Arithmetic-gate controls: the `.kdm` header's file-pointer fields, and
    // a re-signed byte-flip sweep over all three files.
    // ------------------------------------------------------------------

    /// The `.kdm`'s per-field record, laid out from `off` (the first byte
    /// after the plain BKD header). Every shape value in
    /// [`valid_index_and_shape_offset`]'s index is a one-byte vint, so the
    /// fixed offsets below are exact -- the asserts pin that.
    const POINT_COUNT: usize = 21;
    const NUM_INDEX_BYTES: usize = 23;
    const INDEX_START_POINTER: usize = 32;

    fn assert_meta_layout(kdm: &[u8], off: usize) {
        assert_eq!(kdm[off], 1, "numDims");
        assert_eq!(kdm[off + 1], 1, "numIndexDims");
        assert_eq!(kdm[off + 2], 4, "maxPointsInLeafNode");
        assert_eq!(kdm[off + 3], 8, "bytesPerDim");
        assert_eq!(kdm[off + 4], 3, "numLeaves");
        assert_eq!(kdm[off + POINT_COUNT], 9, "pointCount");
        assert_eq!(kdm[off + POINT_COUNT + 1], 9, "docCount");
        assert!(
            kdm[off + NUM_INDEX_BYTES] < 0x80,
            "numIndexBytes is one vint byte"
        );
    }

    /// `indexStartPointer + numIndexBytes` is the `.kdi` range every traversal
    /// starts from, and both halves are unbounded values off the `.kdm`
    /// (`BKDReader` just seeks and lets the read fail). Adding them as `i64`
    /// overflows: a panic in a debug build, and in a release build a wrap to a
    /// *plausible in-range* end offset that hands the tree walker some other
    /// field's bytes.
    #[test]
    fn a_packed_index_range_that_overflows_is_a_decode_error() {
        let (kdm, kdi, kdd, off) = valid_index_and_shape_offset();
        assert_meta_layout(&kdm, off);
        for (start, num_index_bytes) in [(i64::MAX, 0x7Fu8), (-1i64, 0x7F), (0, 0x7F)] {
            let mut kdm = kdm.clone();
            kdm[off + NUM_INDEX_BYTES] = num_index_bytes;
            kdm[off + INDEX_START_POINTER..off + INDEX_START_POINTER + 8]
                .copy_from_slice(&start.to_le_bytes()); // DataInput.readLong is little-endian
            resign(&mut kdm);
            let reader = open(&kdm, &kdi, &kdd, &id(), "").unwrap();
            assert!(
                reader.decode_all_points(0).is_err(),
                "start={start} numIndexBytes={num_index_bytes} decoded instead of failing"
            );
            assert!(reader
                .intersect(0, &mut CountingVisitor::default())
                .is_err());
        }
    }

    #[derive(Default)]
    struct CountingVisitor {
        seen: usize,
    }

    impl IntersectVisitor for CountingVisitor {
        fn compare(&mut self, _min: &[u8], _max: &[u8]) -> Relation {
            Relation::CellCrossesQuery
        }
        fn visit(&mut self, _doc_id: i32) {
            self.seen += 1;
        }
        fn visit_with_value(&mut self, _doc_id: i32, _packed_value: &[u8]) {
            self.seen += 1;
        }
    }

    /// Rewrites the codec footer so a mutated file still passes its CRC. The
    /// `.kdm` is checksum-verified on open, so without this a byte flip would
    /// be caught by the checksum and no semantic invariant would ever run.
    fn resign(buf: &mut Vec<u8>) {
        buf.truncate(buf.len() - codec_util::FOOTER_LENGTH);
        codec_util::write_footer(buf);
    }

    /// Re-signed byte-flip sweep over `.kdm`, `.kdi` and `.kdd`.
    ///
    /// For every payload byte of each file in turn, flip one bit, re-sign the
    /// footer, and drive the whole read surface -- `open`, `decode_leaves`,
    /// `decode_all_points` and a `range_query`. The bar is that every outcome
    /// is either a decoded result or a typed error: never a panic, never a
    /// reservation big enough to abort, never a hang.
    ///
    /// The rejection *rate* is reported rather than asserted exactly: most
    /// `.kdd` payload bytes are packed point values, and flipping one of those
    /// yields a different but perfectly well-formed point, which is not
    /// corruption the reader can or should detect.
    #[test]
    fn resigned_byte_flip_sweep_never_panics() {
        let (kdm, kdi, kdd) = {
            let (kdm, kdi, kdd, _) = valid_index_and_shape_offset();
            (kdm, kdi, kdd)
        };
        let names = ["kdm", "kdi", "kdd"];
        let previous_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let mut rejected = [0usize; 3];
        let mut total = [0usize; 3];
        let mut panics: Vec<String> = Vec::new();
        for which in 0..3 {
            let len = [kdm.len(), kdi.len(), kdd.len()][which];
            for offset in 0..len - codec_util::FOOTER_LENGTH {
                for bit in [0u8, 3, 7] {
                    let mut files = [kdm.clone(), kdi.clone(), kdd.clone()];
                    files[which][offset] ^= 1 << bit;
                    resign(&mut files[which]);
                    total[which] += 1;
                    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let reader = open(&files[0], &files[1], &files[2], &id(), "")?;
                        reader.decode_leaves(0)?;
                        reader.decode_all_points(0)?;
                        reader.range_query(
                            0,
                            &long_sortable_bytes(100),
                            &long_sortable_bytes(700),
                        )?;
                        reader.intersect(0, &mut CountingVisitor::default())?;
                        Ok::<(), Error>(())
                    }));
                    match outcome {
                        Ok(Ok(())) => {}
                        Ok(Err(_)) => rejected[which] += 1,
                        Err(_) => {
                            panics.push(format!("{}: offset={offset} bit={bit}", names[which]))
                        }
                    }
                }
            }
        }
        std::panic::set_hook(previous_hook);
        assert!(
            panics.is_empty(),
            "{} of {} flips panicked, e.g. {:?}",
            panics.len(),
            total.iter().sum::<usize>(),
            &panics[..panics.len().min(8)]
        );
        // Printed under `cargo test -- --nocapture`; the assertions below are
        // the regression guard (a rate collapsing to zero means the reader has
        // started accepting corruption silently).
        println!(
            "re-signed byte-flip rejection rate: kdm {}/{}, kdi {}/{}, kdd {}/{}",
            rejected[0], total[0], rejected[1], total[1], rejected[2], total[2]
        );
        assert!(
            rejected[0] * 2 > total[0],
            "kdm: {}/{}",
            rejected[0],
            total[0]
        );
        assert!(rejected[1] > 0, "kdi: {}/{}", rejected[1], total[1]);
        assert!(rejected[2] > 0, "kdd: {}/{}", rejected[2], total[2]);
    }
}

#[cfg(test)]
mod intersect_tests {
    // The arithmetic gate is about values read off disk; a test's `i + 1` is
    // not one. See docs/arithmetic-gate.md.
    #![allow(clippy::arithmetic_side_effects)]

    use super::*;

    fn id() -> [u8; codec_util::ID_LENGTH] {
        [9u8; codec_util::ID_LENGTH]
    }

    /// `NumericUtils.longToSortableBytes`: flip the sign bit so unsigned
    /// big-endian byte order matches signed numeric order.
    fn long_sortable_bytes(v: i64) -> Vec<u8> {
        ((v as u64) ^ 0x8000_0000_0000_0000).to_be_bytes().to_vec()
    }

    /// Records which relations `intersect` saw, so a test can prove that
    /// pruning really happened rather than just that the answer was right.
    struct CountingRange {
        inner: RangeVisitor,
        cells_compared: usize,
        leaves_fully_inside: usize,
        points_examined: usize,
    }

    impl IntersectVisitor for CountingRange {
        fn compare(&mut self, min_packed: &[u8], max_packed: &[u8]) -> Relation {
            self.cells_compared += 1;
            self.inner.compare(min_packed, max_packed)
        }
        fn visit(&mut self, doc_id: i32) {
            self.leaves_fully_inside += 1;
            self.inner.visit(doc_id);
        }
        fn visit_with_value(&mut self, doc_id: i32, packed_value: &[u8]) {
            self.points_examined += 1;
            self.inner.visit_with_value(doc_id, packed_value);
        }
    }

    fn counting(field: &PointsField, lower: &[u8], upper: &[u8]) -> CountingRange {
        CountingRange {
            inner: RangeVisitor {
                lower: lower.to_vec(),
                upper: upper.to_vec(),
                num_index_dims: field.num_index_dims as usize,
                bytes_per_dim: field.bytes_per_dim as usize,
                fixed: FixedWidthRange::new(
                    field.num_index_dims as usize,
                    field.bytes_per_dim as usize,
                    lower,
                    upper,
                ),
                docs: Vec::new(),
            },
            cells_compared: 0,
            leaves_fully_inside: 0,
            points_examined: 0,
        }
    }

    /// Brute-force reference: exactly what `decode_all_points` + an in-memory
    /// filter would return, which is what `lucene-search`'s points query does
    /// today. `intersect` must agree with it on every box.
    fn brute_force(
        reader: &PointsReader<'_>,
        field_number: i32,
        lower: &[u8],
        upper: &[u8],
    ) -> Vec<i32> {
        let field = reader.field(field_number).unwrap();
        let bpd = field.bytes_per_dim as usize;
        reader
            .decode_all_points(field_number)
            .unwrap()
            .into_iter()
            .filter(|p| {
                (0..field.num_index_dims as usize).all(|dim| {
                    let r = dim * bpd..(dim + 1) * bpd;
                    p.packed_value[r.clone()] >= lower[r.clone()]
                        && p.packed_value[r.clone()] <= upper[r]
                })
            })
            .map(|p| p.doc_id)
            .collect()
    }

    /// The two arms of [`PointsReader::intersect_with_scratch`] are the same
    /// walk with the per-level scratch either reused or reallocated at every
    /// inner node (the pre-`c43-final-cleanup` shape, kept so
    /// `examples/bkd_walk_scratch.rs` can time both in one process). They must
    /// visit exactly the same documents, in the same order, and produce the
    /// same estimate -- otherwise the measurement compares two algorithms
    /// rather than two allocation strategies.
    ///
    /// A multi-dimensional tree, because the buffers being reused are indexed
    /// by split dimension: a 1D tree would restore the one dimension it has
    /// even if the level indexing were wrong.
    /// **The integer comparison orders exactly as the byte-wise one.** Every
    /// 4- and 8-byte value against inclusive bounds, including the bytes
    /// whose high bit is set (a signed load would put them first) and the
    /// bounds themselves; and the shapes it does not cover fall back.
    #[test]
    fn a_fixed_width_range_agrees_with_the_byte_wise_comparison() {
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for width in [4usize, 8] {
            for _ in 0..200 {
                let mut a = next().to_be_bytes()[..width].to_vec();
                let mut b = next().to_be_bytes()[..width].to_vec();
                if a > b {
                    std::mem::swap(&mut a, &mut b);
                }
                let range = FixedWidthRange::new(1, width, &a, &b).unwrap();
                let mut probes: Vec<Vec<u8>> = (0..20)
                    .map(|_| next().to_be_bytes()[..width].to_vec())
                    .collect();
                probes.extend([a.clone(), b.clone(), vec![0x80; width], vec![0xff; width]]);
                for v in probes {
                    let bytewise = a.as_slice() <= v.as_slice() && v.as_slice() <= b.as_slice();
                    assert_eq!(range.contains(&v), Some(bytewise), "{a:x?} {v:x?} {b:x?}");
                }
            }
        }
        assert!(FixedWidthRange::new(2, 8, &[0; 16], &[0; 16]).is_none());
        assert!(FixedWidthRange::new(1, 16, &[0; 16], &[0; 16]).is_none());
        let r = FixedWidthRange::new(1, 8, &[0; 8], &[1; 8]).unwrap();
        assert_eq!(r.contains(&[0; 4]), None, "too short to say");
    }

    #[test]
    fn the_two_scratch_arms_visit_exactly_the_same_documents() {
        let points: Vec<(i32, Vec<u8>)> = (0..400)
            .map(|i| {
                let mut packed = long_sortable_bytes((i as i64 * 7919) % 1301);
                packed.extend_from_slice(&long_sortable_bytes((i as i64 * 104_729) % 907));
                packed.extend_from_slice(&long_sortable_bytes(i as i64 % 13));
                (i, packed)
            })
            .collect();
        let field = WritePointsField {
            field_number: 0,
            num_dims: 3,
            num_index_dims: 3,
            bytes_per_dim: 8,
            points,
        };
        let (kdm, kdi, kdd) = write(&[field], 4, &id(), "").unwrap();
        let reader = open(&kdm, &kdi, &kdd, &id(), "").unwrap();
        let meta = reader.field(0).unwrap().clone();

        let boxes = [
            (-2_000i64, 2_000i64),
            (0, 400),
            (100, 900),
            (-1, 1),
            (500, 500),
            (12, 1_300),
        ];
        let mut any_hits = false;
        for (lo, hi) in boxes {
            let lower: Vec<u8> = (0..3).flat_map(|_| long_sortable_bytes(lo)).collect();
            let upper: Vec<u8> = (0..3).flat_map(|_| long_sortable_bytes(hi)).collect();

            let mut reuse = counting(&meta, &lower, &upper);
            reader.intersect_with_scratch(0, &mut reuse, true).unwrap();
            let mut realloc = counting(&meta, &lower, &upper);
            reader
                .intersect_with_scratch(0, &mut realloc, false)
                .unwrap();
            assert_eq!(
                reuse.inner.docs, realloc.inner.docs,
                "reused and reallocated scratch disagree on [{lo}, {hi}]"
            );
            assert_eq!(reuse.cells_compared, realloc.cells_compared);
            assert_eq!(reuse.points_examined, realloc.points_examined);
            any_hits |= !reuse.inner.docs.is_empty();

            let mut v = counting(&meta, &lower, &upper);
            let a = reader
                .estimate_point_count_with_scratch(0, &mut v, i64::MAX, true)
                .unwrap();
            let mut v = counting(&meta, &lower, &upper);
            let b = reader
                .estimate_point_count_with_scratch(0, &mut v, i64::MAX, false)
                .unwrap();
            assert_eq!(a, b, "estimates disagree on [{lo}, {hi}]");
        }
        assert!(any_hits, "every box was empty -- the comparison is vacuous");
    }

    #[test]
    fn intersect_1d_matches_brute_force_on_every_boundary_box() {
        // 173 points across 44 leaves (same shape as
        // `write_then_read_many_leaves_round_trips`), so the traversal has
        // several levels and an unbalanced deepest level.
        let points: Vec<(i32, Vec<u8>)> = (0..300)
            .filter(|i| i % 3 != 0)
            .map(|i| (i, long_sortable_bytes((i as i64) * 7919 - 1_000_000)))
            .collect();
        let field = WritePointsField {
            field_number: 0,
            num_dims: 1,
            num_index_dims: 1,
            bytes_per_dim: 8,
            points,
        };
        let (kdm, kdi, kdd) = write(&[field], 4, &id(), "").unwrap();
        let reader = open(&kdm, &kdi, &kdd, &id(), "").unwrap();
        let meta = reader.field(0).unwrap().clone();

        // Boxes chosen to straddle every interesting case: empty below the
        // whole range, empty above it, exactly the whole range, a single
        // value, and several interior windows.
        let bounds = [
            (-5_000_000i64, -2_000_000i64),
            (10_000_000, 20_000_000),
            (-1_000_000, 1_400_000),
            (0, 0),
            (-1_000_000, -1_000_000),
            (-500_000, 500_000),
            (123_456, 987_654),
            (-1_000_000 + 7919, -1_000_000 + 7919 * 5),
        ];
        for (lo, hi) in bounds {
            let lower = long_sortable_bytes(lo);
            let upper = long_sortable_bytes(hi);
            let mut expected = brute_force(&reader, 0, &lower, &upper);
            expected.sort_unstable();
            let mut got = reader.range_query(0, &lower, &upper).unwrap();
            got.sort_unstable();
            assert_eq!(got, expected, "range [{lo}, {hi}]");
        }

        // Pruning really happens: a narrow interior window must not compare
        // (let alone decode) anything close to all 44 leaves' cells, and a
        // whole-range query must take the CELL_INSIDE_QUERY shortcut for
        // every point (no packed value decoded at all).
        let lower = long_sortable_bytes(-1_000_000 + 7919);
        let upper = long_sortable_bytes(-1_000_000 + 7919 * 5);
        let mut narrow = counting(&meta, &lower, &upper);
        reader.intersect(0, &mut narrow).unwrap();
        assert!(
            narrow.cells_compared < 2 * meta.num_leaves as usize,
            "narrow query compared {} cells for {} leaves -- no pruning?",
            narrow.cells_compared,
            meta.num_leaves
        );
        assert!(narrow.points_examined < 40, "{}", narrow.points_examined);

        let lower = long_sortable_bytes(i64::MIN);
        let upper = long_sortable_bytes(i64::MAX);
        let mut everything = counting(&meta, &lower, &upper);
        reader.intersect(0, &mut everything).unwrap();
        assert_eq!(everything.points_examined, 0);
        assert_eq!(everything.leaves_fully_inside, meta.point_count as usize);
        assert_eq!(everything.cells_compared, 1, "root alone should be INSIDE");
    }

    #[test]
    fn intersect_2d_matches_brute_force() {
        // Multi-index-dimension: split dimension alternates, so the
        // per-dimension bound clamping and the split-value reconstruction's
        // per-dimension `last_split_values`/`negative_deltas` state both get
        // exercised (a single-dimension tree can't tell them apart).
        fn int_sortable_bytes(v: i32) -> [u8; 4] {
            ((v as u32) ^ 0x8000_0000).to_be_bytes()
        }
        let mut points: Vec<(i32, Vec<u8>)> = Vec::new();
        for i in 0..120i32 {
            let mut packed = Vec::new();
            packed.extend_from_slice(&int_sortable_bytes(i * 13 % 97 - 40));
            packed.extend_from_slice(&int_sortable_bytes(i * 31 % 71 - 30));
            points.push((i, packed));
        }
        let field = WritePointsField {
            field_number: 3,
            num_dims: 2,
            num_index_dims: 2,
            bytes_per_dim: 4,
            points,
        };
        let (kdm, kdi, kdd) = write(&[field], 5, &id(), "").unwrap();
        let reader = open(&kdm, &kdi, &kdd, &id(), "").unwrap();

        for (x0, x1, y0, y1) in [
            (-40i32, 60i32, -30i32, 45i32),
            (0, 10, 0, 10),
            (-100, 100, -100, 100),
            (1000, 2000, 0, 5),
            (-5, -5, -5, 40),
        ] {
            let mut lower = Vec::new();
            lower.extend_from_slice(&int_sortable_bytes(x0));
            lower.extend_from_slice(&int_sortable_bytes(y0));
            let mut upper = Vec::new();
            upper.extend_from_slice(&int_sortable_bytes(x1));
            upper.extend_from_slice(&int_sortable_bytes(y1));

            let mut expected = brute_force(&reader, 3, &lower, &upper);
            expected.sort_unstable();
            let mut got = reader.range_query(3, &lower, &upper).unwrap();
            got.sort_unstable();
            assert_eq!(got, expected, "box ({x0}..{x1}, {y0}..{y1})");
        }
    }

    #[test]
    fn intersect_single_leaf_tree_still_works() {
        // `num_leaves == 1`: the root *is* a leaf, so `intersect_node` takes
        // the leaf branch on its very first call with no packed-index inner
        // node bytes to parse at all.
        let points: Vec<(i32, Vec<u8>)> = (0..5)
            .map(|i| (i, long_sortable_bytes(i as i64 * 10)))
            .collect();
        let field = WritePointsField {
            field_number: 0,
            num_dims: 1,
            num_index_dims: 1,
            bytes_per_dim: 8,
            points,
        };
        let (kdm, kdi, kdd) = write(&[field], 512, &id(), "").unwrap();
        let reader = open(&kdm, &kdi, &kdd, &id(), "").unwrap();
        assert_eq!(reader.field(0).unwrap().num_leaves, 1);

        let got = reader
            .range_query(0, &long_sortable_bytes(10), &long_sortable_bytes(30))
            .unwrap();
        assert_eq!(got, vec![1, 2, 3]);
        // Fully-outside box: the single leaf is never decoded.
        let got = reader
            .range_query(0, &long_sortable_bytes(1000), &long_sortable_bytes(2000))
            .unwrap();
        assert!(got.is_empty());
    }

    /// Exercises the packed index's `suffix == 0` branch: when many points
    /// share a value, consecutive splits in the same dimension can produce a
    /// split value byte-identical to the previous one, and
    /// `recursePackIndex` then writes `prefix == bytesPerDim` with no suffix
    /// bytes at all ("our split value is == last split value in this dim,
    /// which can happen when there are many duplicate values", per
    /// `BKDReader.readNodeData`'s own comment). The reconstruction must leave
    /// the running split value untouched in that case.
    #[test]
    fn intersect_with_heavy_duplicate_values_matches_brute_force() {
        // 400 points over only 5 distinct values, 4 per leaf => 100 leaves,
        // so most splits land inside a run of equal values.
        let points: Vec<(i32, Vec<u8>)> = (0..400)
            .map(|i| (i, long_sortable_bytes((i % 5) as i64)))
            .collect();
        let field = WritePointsField {
            field_number: 0,
            num_dims: 1,
            num_index_dims: 1,
            bytes_per_dim: 8,
            points: points.clone(),
        };
        let (kdm, kdi, kdd) = write(&[field], 4, &id(), "").unwrap();
        let reader = open(&kdm, &kdi, &kdd, &id(), "").unwrap();
        assert_eq!(reader.field(0).unwrap().num_leaves, 100);

        for (lo, hi) in [(0i64, 4i64), (2, 2), (1, 3), (5, 10), (-1, 0)] {
            let lower = long_sortable_bytes(lo);
            let upper = long_sortable_bytes(hi);
            let mut expected: Vec<i32> = points
                .iter()
                .filter(|(_, v)| *v >= lower && *v <= upper)
                .map(|(d, _)| *d)
                .collect();
            expected.sort_unstable();
            let mut got = reader.range_query(0, &lower, &upper).unwrap();
            got.sort_unstable();
            assert_eq!(got, expected, "duplicate-heavy range [{lo}, {hi}]");
        }
    }

    #[test]
    fn intersect_unknown_field_rejected() {
        let points = vec![(0, long_sortable_bytes(1))];
        let field = WritePointsField {
            field_number: 0,
            num_dims: 1,
            num_index_dims: 1,
            bytes_per_dim: 8,
            points,
        };
        let (kdm, kdi, kdd) = write(&[field], 512, &id(), "").unwrap();
        let reader = open(&kdm, &kdi, &kdd, &id(), "").unwrap();
        assert!(matches!(
            reader.range_query(7, &long_sortable_bytes(0), &long_sortable_bytes(1)),
            Err(Error::IllegalFieldNumber(7))
        ));
    }

    // --- BKDWriter.merge's one-pass path ---

    /// `n` single-dimension points with distinct 4-byte values, in ascending
    /// value order -- the shape an already-merged 1-D points stream has.
    fn sorted_1d_points(n: usize) -> Vec<(i32, Vec<u8>)> {
        (0..n)
            .map(|i| {
                let v = i as u32 * 37 + 11;
                (i as i32, v.to_be_bytes().to_vec())
            })
            .collect()
    }

    #[test]
    fn presorted_plan_matches_the_general_plan_byte_for_byte() {
        // The whole safety argument for skipping the sort: for a
        // single-index-dimension field whose points are already sorted, the
        // one-pass plan and the general recursive plan must be the *same*
        // plan. Distinct values make the general path's stable sort produce
        // exactly the ascending order, whatever order it is handed.
        for n in [1usize, 2, 8, 9, 16, 17, 100, 1_000, 4_097] {
            let sorted = sorted_1d_points(n);
            // A deterministic shuffle, so the general (sorting) path runs.
            let mut shuffled = sorted.clone();
            let len = shuffled.len();
            for i in 0..len {
                let j = (i * 7919 + 13) % len;
                shuffled.swap(i, j);
            }

            let field = |points: Vec<(i32, Vec<u8>)>| WritePointsField {
                field_number: 3,
                num_dims: 1,
                num_index_dims: 1,
                bytes_per_dim: 4,
                points,
            };
            let a = write(&[field(sorted)], 16, &[5u8; codec_util::ID_LENGTH], "").unwrap();
            let b = write(&[field(shuffled)], 16, &[5u8; codec_util::ID_LENGTH], "").unwrap();
            assert_eq!(a.0, b.0, "n={n}: .kdm differs");
            assert_eq!(a.1, b.1, "n={n}: .kdi differs");
            assert_eq!(a.2, b.2, "n={n}: .kdd differs");
        }
    }

    #[test]
    fn a_presorted_one_dimension_field_round_trips_through_the_reader() {
        const N: usize = 5_000;
        let points = sorted_1d_points(N);
        let field = WritePointsField {
            field_number: 0,
            num_dims: 1,
            num_index_dims: 1,
            bytes_per_dim: 4,
            points: points.clone(),
        };
        let id = [6u8; codec_util::ID_LENGTH];
        let (kdm, kdi, kdd) = write(&[field], 64, &id, "").unwrap();
        let reader = open(&kdm, &kdi, &kdd, &id, "").unwrap();
        let got = reader.decode_all_points(0).unwrap();
        assert_eq!(got.len(), N);
        for (i, point) in got.iter().enumerate() {
            assert_eq!(point.doc_id, points[i].0, "point {i}");
            assert_eq!(point.packed_value, points[i].1, "point {i}");
        }
        let meta = reader.field(0).unwrap();
        assert_eq!(meta.point_count, N as i64);
        assert_eq!(meta.min_packed_value, points[0].1);
        assert_eq!(meta.max_packed_value, points[N - 1].1);
    }

    #[test]
    fn a_presorted_field_with_trailing_data_only_dimensions_still_takes_the_one_pass_path() {
        // `num_index_dims == 1 < num_dims`: the trailing dimension rides along
        // in every packed value but never splits, so sortedness of dimension 0
        // is all the one-pass plan needs. Same byte-for-byte equivalence test.
        let n = 300usize;
        let sorted: Vec<(i32, Vec<u8>)> = (0..n)
            .map(|i| {
                let mut v = (i as u16).to_be_bytes().to_vec();
                v.extend_from_slice(&((n - i) as u16).to_be_bytes());
                (i as i32, v)
            })
            .collect();
        let mut shuffled = sorted.clone();
        for i in 0..n {
            let j = (i * 131 + 5) % n;
            shuffled.swap(i, j);
        }
        let field = |points: Vec<(i32, Vec<u8>)>| WritePointsField {
            field_number: 1,
            num_dims: 2,
            num_index_dims: 1,
            bytes_per_dim: 2,
            points,
        };
        let id = [4u8; codec_util::ID_LENGTH];
        let a = write(&[field(sorted)], 32, &id, "").unwrap();
        let b = write(&[field(shuffled)], 32, &id, "").unwrap();
        assert_eq!((a.0, a.1, a.2), (b.0, b.1, b.2));
    }

    #[test]
    fn points_with_equal_values_keep_their_input_order_on_both_paths() {
        // Ties are where a stable sort and a plain chunking could diverge:
        // the one-pass path preserves input order by construction, and the
        // general path's `sort_by` is stable, so they agree.
        let points: Vec<(i32, Vec<u8>)> = (0..64)
            .map(|i| (i, ((i / 8) as u32).to_be_bytes().to_vec()))
            .collect();
        let field = WritePointsField {
            field_number: 0,
            num_dims: 1,
            num_index_dims: 1,
            bytes_per_dim: 4,
            points: points.clone(),
        };
        let id = [3u8; codec_util::ID_LENGTH];
        let (kdm, kdi, kdd) = write(&[field], 8, &id, "").unwrap();
        let reader = open(&kdm, &kdi, &kdd, &id, "").unwrap();
        let got = reader.decode_all_points(0).unwrap();
        let doc_ids: Vec<i32> = got.iter().map(|p| p.doc_id).collect();
        assert_eq!(doc_ids, (0..64).collect::<Vec<i32>>());
    }

    // ------------------------------------------------------------------
    // Arithmetic-gate controls: the packed-index (`.kdi`) walk. These build a
    // `PointsReader` over a hand-written packed index, which is the only way
    // to reach the pruning traversal with values no writer emits.
    // ------------------------------------------------------------------

    fn hand_built_field(num_leaves: i32, num_index_bytes: i32) -> PointsField {
        PointsField {
            num_dims: 1,
            num_index_dims: 1,
            bytes_per_dim: 1,
            max_points_in_leaf_node: 512,
            num_leaves,
            min_packed_value: vec![0x00],
            max_packed_value: vec![0xFF],
            point_count: 2,
            doc_count: 2,
            index_start_pointer: 0,
            num_index_bytes,
            bkd_version: BKD_VERSION_CURRENT,
        }
    }

    fn negative_vint(out: &mut Vec<u8>, v: i32) {
        let mut u = v as u32;
        loop {
            let b = (u & 0x7F) as u8;
            u >>= 7;
            if u != 0 {
                out.push(b | 0x80);
            } else {
                out.push(b);
                break;
            }
        }
    }

    /// A negative split-descriptor vint. Java's `code % numIndexDims` yields a
    /// negative `splitDim` that indexes `splitValuesStack` out of bounds one
    /// line later; here the same value cast to `usize` indexed
    /// `negative_deltas` with an astronomic offset -- a panic, not a decode
    /// error. The `0x10..=0x20` query box forces `CELL_CROSSES_QUERY` at the
    /// root so the pruning path (not `addAll`) reads the descriptor.
    #[test]
    fn a_negative_split_descriptor_is_a_decode_error() {
        let mut kdi = vec![0u8]; // root FP-delta vlong = 0
        negative_vint(&mut kdi, -1);
        kdi.extend_from_slice(&[0u8; 16]);
        let field = hand_built_field(2, kdi.len() as i32);
        let kdd = vec![0u8; 64];
        let reader = PointsReader {
            kdi: &kdi,
            kdd: &kdd,
            fields: std::borrow::Cow::Owned(vec![(0, field)]),
        };
        let err = reader.range_query(0, &[0x10], &[0x20]).unwrap_err();
        assert!(
            format!("{err}").contains("negative BKD split descriptor"),
            "{err}"
        );
    }

    /// `leftNumBytes` is the skip-ahead hint the walk adds to the current
    /// position to find the right sibling. A negative one became a huge
    /// `usize` and overflowed the sum -- in a release build, a wrap to a
    /// position inside the packed index, i.e. the walk silently resuming on
    /// the wrong node.
    #[test]
    fn a_negative_left_num_bytes_is_a_decode_error() {
        // numLeaves = 3 so the root's left child is itself an inner node and
        // the `leftNumBytes` vint is present.
        let mut kdi = vec![0u8, 0u8]; // root FP delta = 0, split descriptor = 0
        negative_vint(&mut kdi, -1); // leftNumBytes
        kdi.extend_from_slice(&[0u8; 16]);
        let field = hand_built_field(3, kdi.len() as i32);
        let kdd = vec![0u8; 64];
        let reader = PointsReader {
            kdi: &kdi,
            kdd: &kdd,
            fields: std::borrow::Cow::Owned(vec![(0, field)]),
        };
        let err = reader.range_query(0, &[0x10], &[0x20]).unwrap_err();
        assert!(format!("{err}").contains("leftNumBytes"), "{err}");
    }

    /// A leaf's file pointer is its parent's baseline plus a `.kdi` delta, and
    /// Java lets the `long` add wrap before failing at `seek`. The left child
    /// here is entirely outside the query, so the walk reaches the right
    /// child's delta without touching `.kdd` first.
    #[test]
    fn a_leaf_pointer_that_overflows_is_a_decode_error() {
        let mut kdi = Vec::new();
        write_vlong_test(&mut kdi, i64::MAX); // root FP baseline
        kdi.push(0); // split descriptor: splitDim 0, prefix 0, delta 0
        write_vlong_test(&mut kdi, 1); // right child's FP delta
        let field = hand_built_field(2, kdi.len() as i32);
        let kdd = vec![0u8; 64];
        let reader = PointsReader {
            kdi: &kdi,
            kdd: &kdd,
            fields: std::borrow::Cow::Owned(vec![(0, field)]),
        };
        // The split value decodes to 0x00, so the left cell is [0x00, 0x00],
        // entirely below the query box.
        let err = reader.range_query(0, &[0x80], &[0xFF]).unwrap_err();
        assert!(
            format!("{err}").contains("leaf block pointer overflows"),
            "{err}"
        );
    }

    fn write_vlong_test(out: &mut Vec<u8>, mut v: i64) {
        loop {
            let b = (v & 0x7F) as u8;
            v = ((v as u64) >> 7) as i64;
            if v != 0 {
                out.push(b | 0x80);
            } else {
                out.push(b);
                break;
            }
        }
    }

    /// `RangeVisitor` slices the caller's box per index dimension against the
    /// *field's* shape, so a box of the wrong width indexed out of bounds
    /// mid-traversal. Java's `PointRangeQuery` constructor checks the same
    /// thing up front.
    #[test]
    fn range_query_rejects_bounds_of_the_wrong_width() {
        let points: Vec<(i32, Vec<u8>)> = (0..8)
            .map(|i| (i, long_sortable_bytes(i as i64 * 10)))
            .collect();
        let field = WritePointsField {
            field_number: 0,
            num_dims: 1,
            num_index_dims: 1,
            bytes_per_dim: 8,
            points,
        };
        let (kdm, kdi, kdd) = write(&[field], 4, &id(), "").unwrap();
        let reader = open(&kdm, &kdi, &kdd, &id(), "").unwrap();
        match reader.range_query(0, &[0u8; 4], &long_sortable_bytes(50)) {
            Err(Error::InvalidConfig(msg)) => assert!(msg.contains("8 bytes"), "{msg}"),
            other => panic!("expected InvalidConfig, got {other:?}"),
        }
        // The correctly-sized box still works.
        assert!(!reader
            .range_query(0, &long_sortable_bytes(0), &long_sortable_bytes(70))
            .unwrap()
            .is_empty());
    }

    /// The same 173-point / 44-leaf index
    /// `intersect_1d_matches_brute_force_on_every_boundary_box` builds: deep
    /// enough that the estimate walk has several levels, and unbalanced at the
    /// deepest one.
    fn single_dim_index() -> (Vec<u8>, Vec<u8>, Vec<u8>, [u8; codec_util::ID_LENGTH]) {
        let points: Vec<(i32, Vec<u8>)> = (0..300)
            .filter(|i| i % 3 != 0)
            .map(|i| (i, long_sortable_bytes((i as i64) * 7919 - 1_000_000)))
            .collect();
        let field = WritePointsField {
            field_number: 0,
            num_dims: 1,
            num_index_dims: 1,
            bytes_per_dim: 8,
            points,
        };
        let (kdm, kdi, kdd) = write(&[field], 4, &id(), "").unwrap();
        (kdm, kdi, kdd, id())
    }

    /// A `PointsField` with nothing but the four counts `subtree_size` reads.
    /// Built by hand because the formula is a function of those counts alone
    /// -- no `.kdi`/`.kdd` is involved, which is the property that makes
    /// `estimate_point_count` cheap in the first place.
    fn counts_only_field(
        num_leaves: i32,
        max_points_in_leaf_node: i32,
        point_count: i64,
    ) -> PointsField {
        PointsField {
            num_dims: 1,
            num_index_dims: 1,
            bytes_per_dim: 8,
            max_points_in_leaf_node,
            num_leaves,
            min_packed_value: vec![0u8; 8],
            max_packed_value: vec![0xFFu8; 8],
            point_count,
            doc_count: 1,
            index_start_pointer: 0,
            num_index_bytes: 0,
            bkd_version: BKD_VERSION_CURRENT,
        }
    }

    /// `BKDPointTree.size()` over the shape the committed `points_index`
    /// fixture has for its `val` field: 1 333 points, 512 per leaf, so three
    /// leaves -- the unbalanced case where the left subtree is one level
    /// deeper than the right, and where the last leaf holds 309 points rather
    /// than 512.
    ///
    /// Hand-derived from Java's own arithmetic rather than round-tripped, so a
    /// self-consistent-but-wrong formula cannot pass: `numLeaves = 3` gives
    /// `treeDepth = log2(3) + 2 = 3` and `rightMostLeafNode = (1 << 2) - 1 = 3`.
    #[test]
    fn subtree_size_matches_javas_unbalanced_formula() {
        let field = counts_only_field(3, 512, 1333);
        // Root: every leaf, so the whole point count.
        assert_eq!(field.subtree_size(1).unwrap(), 1333);
        // Node 2's subtree is leaves 4 and 5 (it is above the leaf offset), a
        // full 512 each, and its rightmost leaf is not the tree's.
        assert_eq!(field.subtree_size(2).unwrap(), 1024);
        // Node 3 *is* a leaf (3 >= numLeaves), and it is the tree's rightmost,
        // so it holds `pointCount % 512 == 309`.
        assert_eq!(field.subtree_size(3).unwrap(), 309);
        assert_eq!(field.subtree_size(4).unwrap(), 512);
        assert_eq!(field.subtree_size(5).unwrap(), 512);
        // The three leaves partition the field.
        assert_eq!(
            field.subtree_size(4).unwrap()
                + field.subtree_size(5).unwrap()
                + field.subtree_size(3).unwrap(),
            1333
        );
    }

    /// A point count that is an exact multiple of the leaf size takes Java's
    /// `lastLeafNodePointCount == 0 ? maxPointsInLeafNode` promotion -- a full
    /// last leaf, not an empty one.
    #[test]
    fn subtree_size_treats_an_exact_multiple_as_a_full_last_leaf() {
        let field = counts_only_field(2, 512, 1024);
        assert_eq!(field.subtree_size(1).unwrap(), 1024);
        assert_eq!(field.subtree_size(2).unwrap(), 512);
        assert_eq!(field.subtree_size(3).unwrap(), 512);
    }

    /// One leaf: Java forces `isTreeBalanced` false for this case, and the
    /// unbalanced formula answers the whole point count.
    #[test]
    fn subtree_size_handles_a_single_leaf_tree() {
        let field = counts_only_field(1, 512, 7);
        assert_eq!(field.subtree_size(1).unwrap(), 7);
    }

    /// The largest `numLeaves`/`maxPointsInLeafNode` a `.kdm` can encode still
    /// answers a non-negative size. Java wraps here and hands the query
    /// planner a negative cost.
    #[test]
    fn subtree_size_saturates_rather_than_wrapping_negative() {
        let field = counts_only_field(i32::MAX, i32::MAX, 1);
        assert!(field.subtree_size(1).unwrap() > 0);
        assert!(field.subtree_size(2).unwrap() > 0);
    }

    /// The two estimate entry points reject an unknown field the same way
    /// `intersect`/`range_query` do, rather than answering zero.
    #[test]
    fn estimate_point_count_rejects_an_unknown_field() {
        let (kdm, kdi, kdd, id) = single_dim_index();
        let reader = open(&kdm, &kdi, &kdd, &id, "").unwrap();
        assert!(matches!(
            reader.estimate_range_point_count(99, &long_sortable_bytes(0), &long_sortable_bytes(1)),
            Err(Error::IllegalFieldNumber(99))
        ));
        assert!(matches!(
            reader.estimate_point_count_bounded(99, &mut CountingVisitor::default(), i64::MAX),
            Err(Error::IllegalFieldNumber(99))
        ));
    }

    /// A bounds box of the wrong width is rejected before the walk starts, as
    /// it is for `range_query`.
    #[test]
    fn estimate_range_point_count_rejects_mis_sized_bounds() {
        let (kdm, kdi, kdd, id) = single_dim_index();
        let reader = open(&kdm, &kdi, &kdd, &id, "").unwrap();
        let err = reader
            .estimate_range_point_count(0, &[0u8; 4], &[0xFFu8; 4])
            .unwrap_err();
        assert!(
            format!("{err}").contains("range query bounds must be 8 bytes"),
            "unexpected error: {err}"
        );
    }

    /// `count_points` is exact: the inside cells' sizes plus what the
    /// visitor accepts of the crossing leaves is the number of points
    /// `range_query` finds, for ranges inside one leaf, across many, over
    /// everything and over nothing; an unknown field is an error.
    #[test]
    fn count_points_counts_what_range_query_finds() {
        let (kdm, kdi, kdd, id) = single_dim_index();
        let reader = open(&kdm, &kdi, &kdd, &id, "").unwrap();
        let field = reader.field(0).unwrap();
        let bounds = [
            i64::MIN,
            -1_000_001,
            -1_000_000,
            -992_081,
            -500_000,
            0,
            7_919,
            1_000_000,
            1_367_000,
            i64::MAX,
        ];
        let (mut saw_inside, mut saw_crossing) = (false, false);
        for &lo in &bounds {
            for &hi in bounds.iter().filter(|&&h| h >= lo) {
                let (lower, upper) = (long_sortable_bytes(lo), long_sortable_bytes(hi));
                let want = reader.range_query(0, &lower, &upper).unwrap().len() as i64;
                let mut visitor = reader.range_visitor(field, &lower, &upper).unwrap();
                let inside = reader.count_points(0, &mut visitor).unwrap();
                let crossing = visitor.docs.len() as i64;
                assert_eq!(inside + crossing, want, "[{lo}, {hi}]");
                saw_inside |= inside > 0;
                saw_crossing |= crossing > 0;
            }
        }
        assert!(saw_inside && saw_crossing, "both kinds of cell counted");
        assert!(matches!(
            reader.count_points(99, &mut CountingVisitor::default()),
            Err(Error::IllegalFieldNumber(99))
        ));
    }

    /// `estimate_point_count` never visits a document and never decodes a
    /// point -- Java's estimate uses only `compare`. The visitor here panics
    /// if either happens, so this is a shape assertion, not a count.
    #[derive(Default)]
    struct CountingVisitor {
        compares: usize,
    }

    impl IntersectVisitor for CountingVisitor {
        fn compare(&mut self, _min_packed: &[u8], _max_packed: &[u8]) -> Relation {
            self.compares += 1;
            Relation::CellCrossesQuery
        }
        fn visit(&mut self, _doc_id: i32) {
            panic!("estimate_point_count must not visit documents");
        }
        fn visit_with_value(&mut self, _doc_id: i32, _packed_value: &[u8]) {
            panic!("estimate_point_count must not decode points");
        }
    }

    /// An always-`CELL_CROSSES_QUERY` visitor drives the walk to every leaf
    /// and gets Java's "assume half the points matched" at each -- and touches
    /// no `.kdd` byte on the way, which is the whole point of the estimate.
    #[test]
    fn a_crossing_estimate_halves_every_leaf_and_decodes_nothing() {
        let (kdm, kdi, kdd, id) = single_dim_index();
        let reader = open(&kdm, &kdi, &kdd, &id, "").unwrap();
        let field = reader.field(0).unwrap();
        let mut visitor = CountingVisitor::default();
        let got = reader.estimate_point_count(0, &mut visitor).unwrap();
        let mut want = 0i64;
        for leaf in 0..field.num_leaves {
            want += (field.subtree_size(field.num_leaves + leaf).unwrap() + 1) / 2;
        }
        assert_eq!(got, want);
        assert!(visitor.compares > 0);
    }
}
