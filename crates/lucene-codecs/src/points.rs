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
    /// A [`PointTreeNode`] handed to a reader whose tree for that field has
    /// another shape: a node of another segment's reader.
    #[error("point tree node of field {0} does not belong to this reader's tree")]
    ForeignPointTreeNode(i32),
    /// [`PointsReader::count_points`] on a field of more than one
    /// dimension, which `PointRangeQuery.pointCount` is never run on.
    #[error("field {field_number}: counting points needs one dimension, the field has {num_dims}")]
    MultiDimensionCount { field_number: i32, num_dims: i32 },
    #[error("unsupported doc-ids bits-per-value byte: {0}")]
    UnsupportedDocIdsEncoding(i8),
    #[error("unsupported compressed dimension marker: {0}")]
    UnsupportedCompressedDim(i8),
    #[error("sub-blocks do not add up to the expected count: {expected} != {actual}")]
    SubBlockCountMismatch { expected: usize, actual: usize },
    #[error(
        "field {field_number}: write() requires at least one point (empty fields aren't supported by this write path)"
    )]
    EmptyField { field_number: i32 },
    #[error(
        "field {field_number}: point {index} has packed_value.len() == {actual}, expected bytes_per_dim == {expected}"
    )]
    WrongPackedValueLength {
        field_number: i32,
        index: usize,
        expected: i32,
        actual: usize,
    },
    /// A [`PackedPointsField`] whose `values` is not exactly `docs.len()`
    /// packed values of `num_dims * bytes_per_dim` bytes.
    #[error(
        "field {field_number}: {points} points of {stride} bytes need {points}x{stride} value bytes, got {actual}"
    )]
    PackedValuesLength {
        field_number: i32,
        points: usize,
        stride: usize,
        actual: usize,
    },
    #[error(
        "field {field_number}: num_index_dims ({num_index_dims}) must be between 1 and num_dims ({num_dims}) inclusive"
    )]
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
    ///   `leftNumBytes` skip-ahead vint is what makes the seek to the right
    ///   sibling possible, exactly
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
    /// inverse of `BKDWriter.recursePackIndex`.
    ///
    /// Both bounds slices passed to `compare` are `num_index_dims *
    /// bytes_per_dim` long (the non-indexed trailing data dimensions have no
    /// cell bounds, matching Java).
    pub fn intersect<V: IntersectVisitor>(&self, field_number: i32, visitor: &mut V) -> Result<()> {
        self.intersect_with_scratch(field_number, visitor, true)
    }

    /// [`intersect`](Self::intersect) with the traversal scratch's reuse
    /// switchable -- see [`IntersectCtx::reuse_scratch`]. Production always
    /// passes `true`; `false` reallocates at every node, kept only so
    /// `examples/bkd_walk_scratch.rs` can time both arms in one process, and it
    /// must produce the **same answer**, which
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

    /// The root of the field's index tree -- `PointValues.getPointTree()` --
    /// for a walk that is not depth first (`NearestNeighbor`'s best-first
    /// search over cells): [`point_tree_children`](Self::point_tree_children)
    /// is `moveToChild`/`moveToSibling`, a node is its own `clone()`, and a
    /// leaf's points are read with [`visit_leaf`](Self::visit_leaf).
    pub fn point_tree(&self, field_number: i32) -> Result<PointTreeNode> {
        let field = self
            .field(field_number)
            .ok_or(Error::IllegalFieldNumber(field_number))?;
        let mut input = SliceInput::new(self.inner_nodes(field)?);
        let fp = input.read_vlong()?;
        Ok(PointTreeNode {
            field_number,
            node_id: 1,
            num_leaves: field.num_leaves,
            pos: input.position(),
            fp,
            min: field.min_packed_value.clone(),
            max: field.max_packed_value.clone(),
            split_values: vec![0u8; field.min_packed_value.len()],
            negative_deltas: vec![false; field.num_index_dims as usize],
        })
    }

    /// The field `node` belongs to, refusing a node whose shape is not that
    /// field's tree's -- one taken from another reader (another segment),
    /// whose cell, split values and per-dimension flags would index past
    /// this field's.
    fn node_field(&self, node: &PointTreeNode) -> Result<&PointsField> {
        let field = self
            .field(node.field_number)
            .ok_or(Error::IllegalFieldNumber(node.field_number))?;
        let len = field.min_packed_value.len();
        if node.num_leaves != field.num_leaves
            || node.min.len() != len
            || node.max.len() != len
            || node.split_values.len() != len
            || node.negative_deltas.len() != field.num_index_dims as usize
        {
            return Err(Error::ForeignPointTreeNode(node.field_number));
        }
        Ok(field)
    }

    /// `moveToChild()` then `moveToSibling()` on a clone of `node`: its two
    /// children, each with its own cell (`BKDPointTree.pushLeft` /
    /// `pushRight`), or `None` at a leaf.
    ///
    /// # Errors
    /// A node of a tree other than this reader's (see [`point_tree`](Self::point_tree)),
    /// or corrupt index bytes.
    pub fn point_tree_children(
        &self,
        node: &PointTreeNode,
    ) -> Result<Option<(PointTreeNode, PointTreeNode)>> {
        let field = self.node_field(node)?;
        if node.node_id >= field.num_leaves {
            return Ok(None);
        }
        let mut input = SliceInput::new(self.inner_nodes(field)?);
        input.seek(node.pos)?;
        let mut ctx = IntersectCtx {
            field,
            kdd: self.kdd,
            min: Vec::new(),
            max: Vec::new(),
            split_values: node.split_values.clone(),
            negative_deltas: node.negative_deltas.clone(),
            stack: Vec::new(),
            reuse_scratch: true,
            leaf: LeafBuf::default(),
        };
        let inner = read_inner_node(&mut input, node.node_id, &mut ctx, 0)?;
        let split = &ctx.stack[0].split_value;
        let range = inner.dim_pos..inner.dim_end;
        let mut left_max = node.max.clone();
        left_max[range.clone()].copy_from_slice(split);
        let mut left_deltas = ctx.negative_deltas.clone();
        left_deltas[inner.split_dim] = true;
        let left = PointTreeNode {
            field_number: node.field_number,
            node_id: inner.left_child,
            num_leaves: field.num_leaves,
            pos: input.position(),
            fp: node.fp,
            min: node.min.clone(),
            max: left_max,
            split_values: ctx.split_values.clone(),
            negative_deltas: left_deltas,
        };
        input.seek(inner.right_node_position)?;
        let right_fp = child_block_fp(node.fp, input.read_vlong()?)?;
        let mut right_min = node.min.clone();
        right_min[range].copy_from_slice(split);
        let mut right_deltas = ctx.negative_deltas;
        right_deltas[inner.split_dim] = false;
        let right = PointTreeNode {
            field_number: node.field_number,
            node_id: inner.right_child,
            num_leaves: field.num_leaves,
            pos: input.position(),
            fp: right_fp,
            min: right_min,
            max: node.max.clone(),
            split_values: ctx.split_values,
            negative_deltas: right_deltas,
        };
        Ok(Some((left, right)))
    }

    /// `PointTree.visitDocValues(visitor)` at a leaf of
    /// [`point_tree`](Self::point_tree): every point of the leaf, through
    /// the visitor (a multi-dimension leaf's own box asked first, as
    /// [`intersect`](Self::intersect) asks it). An inner node visits
    /// nothing.
    pub fn visit_leaf<V: IntersectVisitor>(
        &self,
        node: &PointTreeNode,
        visitor: &mut V,
    ) -> Result<()> {
        let field = self.node_field(node)?;
        if node.node_id < field.num_leaves {
            return Ok(());
        }
        let mut kdd_input = SliceInput::new(self.kdd);
        seek_leaf_block(&mut kdd_input, node.fp)?;
        read_leaf_block_into(
            &mut kdd_input,
            field,
            &mut LeafBuf::default(),
            &mut VisitSink(visitor),
        )?;
        Ok(())
    }

    /// `PointTree.size()` at `node`: its points, from the tree's shape
    /// alone (`BKDPointTree.sizeFromBalancedTree`), the count
    /// [`count_points`](Self::count_points) adds for a cell entirely inside
    /// a query.
    ///
    /// # Errors
    /// A node of a tree other than this reader's.
    pub fn point_tree_size(&self, node: &PointTreeNode) -> Result<i64> {
        self.node_field(node)?.subtree_size(node.node_id)
    }

    /// `PointTree.visitDocIDs(visitor)` at `node`: `grow(size())`, then every
    /// document id under it through [`IntersectVisitor::visit_many`], with
    /// no value decoded (`BKDPointTree.addAll`) -- the walk
    /// [`intersect`](Self::intersect) takes for a cell inside the query.
    ///
    /// # Errors
    /// A node of a tree other than this reader's, or corrupt index bytes.
    pub fn visit_doc_ids<V: IntersectVisitor>(
        &self,
        node: &PointTreeNode,
        visitor: &mut V,
    ) -> Result<()> {
        let field = self.node_field(node)?;
        let size = field.subtree_size(node.node_id)?;
        if let Ok(size) = usize::try_from(size) {
            if size <= i32::MAX as usize {
                visitor.grow(size);
            }
        }
        let mut input = SliceInput::new(self.inner_nodes(field)?);
        input.seek(node.pos)?;
        let mut ctx = IntersectCtx {
            field,
            kdd: self.kdd,
            min: Vec::new(),
            max: Vec::new(),
            split_values: node.split_values.clone(),
            negative_deltas: node.negative_deltas.clone(),
            stack: Vec::new(),
            reuse_scratch: true,
            leaf: LeafBuf::default(),
        };
        add_all(&mut input, node.node_id, node.fp, &mut ctx, visitor)
    }

    /// `PointRangeQuery`'s `Weight.count` walk (its private `pointCount`),
    /// exact: a cell entirely inside the query adds its point count
    /// (`BKDPointTree.size()`) without reading a leaf, a cell outside adds
    /// nothing, and a leaf the query crosses is decoded and handed to the
    /// visitor value by value ([`IntersectVisitor::visit_with_value`]).
    /// Returns the inside cells' points; the visitor counts what it accepted
    /// of the crossing leaves.
    ///
    /// One-dimension fields only, the only ones Java runs `pointCount` on
    /// (`numDims == 1`): a multi-dimension leaf is related by its own box
    /// first, so it could be skipped or handed over whole through
    /// [`IntersectVisitor::visit_many`] -- where Java's counting visitor
    /// throws `UnsupportedOperationException` -- rather than value by value.
    ///
    /// # Errors
    /// [`Error::MultiDimensionCount`] for a field of more than one
    /// dimension, or corrupt index bytes.
    pub fn count_points<V: IntersectVisitor>(
        &self,
        field_number: i32,
        visitor: &mut V,
    ) -> Result<i64> {
        let field = self
            .field(field_number)
            .ok_or(Error::IllegalFieldNumber(field_number))?;
        if field.num_dims != 1 {
            return Err(Error::MultiDimensionCount {
                field_number,
                num_dims: field.num_dims,
            });
        }
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
        Ok(RangeVisitor {
            matcher: RangeMatcher::new(field, lower, upper),
            docs: Vec::new(),
        })
    }

    /// [`range_query`](Self::range_query)'s box over `field_number`, for a
    /// visitor of the caller's that collects the matches its own way (into a
    /// bit set, or clearing the documents outside the box).
    pub fn range_matcher(
        &self,
        field_number: i32,
        lower: &[u8],
        upper: &[u8],
    ) -> Result<RangeMatcher> {
        let field = self
            .field(field_number)
            .ok_or(Error::IllegalFieldNumber(field_number))?;
        Ok(self.range_visitor(field, lower, upper)?.matcher)
    }
}

/// Port of `org.apache.lucene.index.PointValues.Relation`, defined in
/// `lucene-util` so the geo relations can answer in it too.
pub use lucene_util::point_values_relation::Relation;

/// Port of `org.apache.lucene.index.PointValues.IntersectVisitor`.
///
/// Java's `grow(int)` hint is [`IntersectVisitor::grow`], called where
/// `BKDReader` calls it.
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
    /// `IntersectVisitor.visit(DocIdSetIterator, byte[])`: every doc of
    /// `doc_ids` has the value `packed_value`, in a leaf crossing the query
    /// (a run of equal values, or a leaf of one value). Defaults to
    /// [`Self::visit_with_value`] per id; a visitor whose test depends only
    /// on the value overrides it to test once.
    fn visit_run(&mut self, doc_ids: &[i32], packed_value: &[u8]) {
        for &doc_id in doc_ids {
            self.visit_with_value(doc_id, packed_value);
        }
    }
    /// `IntersectVisitor.visit(DocIdSetIterator.range(start, start +
    /// count))`: a leaf entirely inside the query whose ids are the
    /// consecutive `start..start + count`. Return `false` (the default) to
    /// receive them through [`Self::visit_many`] instead.
    fn visit_range(&mut self, start: i32, count: usize) -> bool {
        let _ = (start, count);
        false
    }
    /// `IntersectVisitor.visit(DocBaseBitSetIterator)`: a leaf entirely
    /// inside the query whose `count` ids are `base + i` for every set bit
    /// `i` of `words` (bit `i % 64` of word `i / 64`). Return `false` (the
    /// default) to receive them through [`Self::visit_many`] instead.
    fn visit_bitset(&mut self, words: &[u64], base: i32, count: usize) -> bool {
        let _ = (words, base, count);
        false
    }
    /// `IntersectVisitor.grow(count)`: how many documents the walk is about
    /// to hand over -- a whole matching subtree's size before its doc ids, a
    /// leaf's point count before it is visited. A hint (Java sizes a
    /// `DocIdSetBuilder` from it, and a builder's cost follows from it);
    /// ignored by default.
    fn grow(&mut self, count: usize) {
        let _ = count;
    }
    /// A subtree of `count` points entirely inside the query, which
    /// [`PointsReader::count_points`] counts by its size without visiting a
    /// document. The cell is the one the last [`Self::compare`] saw, so a
    /// visitor counting several intervals in one walk can tell which one the
    /// points fall in. Ignored by default.
    fn visit_inside_count(&mut self, count: i64) {
        let _ = count;
    }
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
    /// instead of reusing it -- four `Vec`s per inner node visited. It exists
    /// so `examples/bkd_walk_scratch.rs` can time both arms in **one process
    /// from one build**, which is the only shape of A/B this project trusts.
    /// Production always passes `true`; the cost is one predictable branch per
    /// node against a `.kdi` read.
    reuse_scratch: bool,
    /// Java's `int[maxPointsInLeafNode]` doc-id buffer and the leaf
    /// decoder's scratch: every leaf the walk decodes (inside a matching
    /// subtree or crossing the query) reuses them.
    leaf: LeafBuf,
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
    leaf: LeafBuf,
}

/// A leaf decode's buffers, reused from leaf to leaf as `BKDReader` reuses
/// its scratch arrays.
#[derive(Default)]
struct LeafBuf {
    doc_ids: Vec<i32>,
    /// A `BITSET_IDS` leaf's words (see [`input_doc_ids_shape`]).
    words: Vec<u64>,
    /// Per-dimension common prefix lengths.
    prefix_lens: Vec<usize>,
    /// The packed value being reconstructed.
    value: Vec<u8>,
    /// The leaf's own bounding box (multi-dimension leaves).
    min: Vec<u8>,
    max: Vec<u8>,
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
            leaf: LeafBuf::default(),
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
            leaf: std::mem::take(&mut scratch.leaf),
        }
    }

    /// Hands the buffers back for the next walk.
    fn into_scratch(self, scratch: &mut PointsScratch) {
        scratch.min = self.min;
        scratch.max = self.max;
        scratch.split_values = self.split_values;
        scratch.negative_deltas = self.negative_deltas;
        scratch.stack = self.stack;
        scratch.leaf = self.leaf;
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
        // a packed value anywhere in the subtree. `addAll` announces the
        // subtree's size first (`grow`), when it fits an `int`.
        let size = ctx.field.subtree_size(node_id)?;
        if let Ok(size) = usize::try_from(size) {
            if size <= i32::MAX as usize {
                visitor.grow(size);
            }
        }
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
            &mut ctx.leaf,
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

/// One node of a field's BKD index tree, with its cell -- what
/// `BKDReader.BKDPointTree` is positioned on, as a value: cloning it is
/// `PointTree.clone()`. See [`PointsReader::point_tree`].
///
/// Rust shape: Java keeps one set of per-level stacks and moves up and down;
/// a node here carries its own copy of the state its children are decoded
/// from (the split values seen along its path, the negative-delta flags,
/// the `.kdi` position and leaf pointer).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PointTreeNode {
    /// The field whose tree this node belongs to: the reader's methods take
    /// the node alone, so a node cannot be walked as another field's.
    field_number: i32,
    node_id: i32,
    num_leaves: i32,
    /// `.kdi` offset of this node's own data (after its FP delta).
    pos: usize,
    /// The leaf block pointer this node inherits (`leafBlockFPStack`).
    fp: i64,
    min: Vec<u8>,
    max: Vec<u8>,
    split_values: Vec<u8>,
    negative_deltas: Vec<bool>,
}

impl PointTreeNode {
    /// `getMinPackedValue()`.
    pub fn min_packed(&self) -> &[u8] {
        &self.min
    }

    /// `getMaxPackedValue()`.
    pub fn max_packed(&self) -> &[u8] {
        &self.max
    }

    /// Whether `moveToChild()` would answer `false`.
    pub fn is_leaf(&self) -> bool {
        self.node_id >= self.num_leaves
    }
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
        Relation::CellInsideQuery => {
            let count = ctx.field.subtree_size(node_id)?;
            visitor.visit_inside_count(count);
            Ok(count)
        }
        Relation::CellCrossesQuery if node_id >= ctx.field.num_leaves => {
            let mut kdd_input = SliceInput::new(ctx.kdd);
            seek_leaf_block(&mut kdd_input, fp)?;
            read_leaf_block_into(
                &mut kdd_input,
                ctx.field,
                &mut ctx.leaf,
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
        // `DocIdsWriter.readInts(in, count, visitor, buf)`: a leaf of
        // consecutive ids or a bit set is offered to the visitor whole.
        let at = kdd_input.position();
        match input_doc_ids_shape(&mut kdd_input, count, &mut ctx.leaf.words)? {
            DocIdsShape::Range(start) if visitor.visit_range(start, count) => return Ok(()),
            DocIdsShape::Bits(base) if visitor.visit_bitset(&ctx.leaf.words, base, count) => {
                return Ok(())
            }
            _ => kdd_input.seek(at)?,
        }
        read_doc_ids_into(
            &mut kdd_input,
            count,
            &mut ctx.leaf.doc_ids,
            ctx.field.bkd_version,
        )?;
        visitor.visit_many(&ctx.leaf.doc_ids);
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

/// `PointRangeQuery`'s inclusive box over one field: how a cell relates to
/// it and whether a point is inside, for a visitor that collects the
/// matches its own way. Made by [`PointsReader::range_matcher`].
#[derive(Clone)]
pub struct RangeMatcher {
    lower: Vec<u8>,
    upper: Vec<u8>,
    num_index_dims: usize,
    bytes_per_dim: usize,
    /// The bounds as integers, for the one-dimensional 4- and 8-byte shapes.
    fixed: Option<FixedWidthRange>,
}

impl RangeMatcher {
    fn new(field: &PointsField, lower: &[u8], upper: &[u8]) -> Self {
        let (num_index_dims, bytes_per_dim) =
            (field.num_index_dims as usize, field.bytes_per_dim as usize);
        Self {
            lower: lower.to_vec(),
            upper: upper.to_vec(),
            num_index_dims,
            bytes_per_dim,
            fixed: FixedWidthRange::new(num_index_dims, bytes_per_dim, lower, upper),
        }
    }

    /// How the cell `[min_packed, max_packed]` relates to the box.
    pub fn compare(&self, min_packed: &[u8], max_packed: &[u8]) -> Relation {
        let mut crosses = false;
        for dim in 0..self.num_index_dims {
            // ARITH: `dim < num_index_dims`, so `hi <= num_index_dims *
            // bytes_per_dim` -- the length `range_matcher` verified for
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

    /// Whether the point `packed_value` (the field's packed value) is inside
    /// the box.
    #[inline]
    pub fn matches(&self, packed_value: &[u8]) -> bool {
        if let Some(inside) = self.fixed.and_then(|f| f.contains(packed_value)) {
            return inside;
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
                return false;
            }
        }
        true
    }
}

/// `PointRangeQuery`'s `IntersectVisitor`, inclusive on both ends, into a
/// list of doc ids.
struct RangeVisitor {
    matcher: RangeMatcher,
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
        self.matcher.compare(min_packed, max_packed)
    }

    fn visit(&mut self, doc_id: i32) {
        self.docs.push(doc_id);
    }

    fn visit_many(&mut self, doc_ids: &[i32]) {
        self.docs.extend_from_slice(doc_ids);
    }

    fn visit_range(&mut self, start: i32, count: usize) -> bool {
        // ARITH: the caller guarantees `start + count <= i32::MAX`.
        #[allow(clippy::arithmetic_side_effects)]
        self.docs.extend(start..start + count as i32);
        true
    }

    fn visit_with_value(&mut self, doc_id: i32, packed_value: &[u8]) {
        if self.matcher.matches(packed_value) {
            self.docs.push(doc_id);
        }
    }

    fn grow(&mut self, count: usize) {
        self.docs.reserve(count);
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
/// an abort.
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

    /// Every document of `doc_ids` has the value `packed_value`.
    #[inline]
    fn run(&mut self, doc_ids: &[i32], packed_value: &[u8]) {
        for &doc_id in doc_ids {
            self.point(doc_id, packed_value);
        }
    }

    /// `visitor.compare(leafMin, leafMax)` on a multi-dimension leaf's own
    /// bounding box, which `BKDReader.visitDocValuesWithCardinality` asks
    /// before it decodes a value: a leaf outside the query is skipped, one
    /// inside has its doc ids visited without values ([`Self::inside`]). A
    /// sink that wants every point answers [`Relation::CellCrossesQuery`].
    fn leaf_bounds(&mut self, _min: &[u8], _max: &[u8]) -> Relation {
        Relation::CellCrossesQuery
    }

    /// The doc ids of a leaf [`Self::leaf_bounds`] put inside the query
    /// (`visitor.visit(IntsRef)`).
    fn inside(&mut self, _doc_ids: &[i32]) {}

    /// `visitor.grow(count)` before a leaf's documents are handed over.
    fn grow(&mut self, _count: usize) {}
}

impl LeafSink for Vec<Point> {
    fn point(&mut self, doc_id: i32, packed_value: &[u8]) {
        self.push(Point {
            doc_id,
            packed_value: packed_value.to_vec(),
        });
    }
}

/// A crossing leaf's points, streamed to the visitor without materialising a
/// `Vec<Point>` first (one heap allocation per point for its packed value).
struct VisitSink<'v, V>(&'v mut V);

impl<V: IntersectVisitor> LeafSink for VisitSink<'_, V> {
    #[inline]
    fn point(&mut self, doc_id: i32, packed_value: &[u8]) {
        self.0.visit_with_value(doc_id, packed_value);
    }

    #[inline]
    fn run(&mut self, doc_ids: &[i32], packed_value: &[u8]) {
        self.0.visit_run(doc_ids, packed_value);
    }

    fn leaf_bounds(&mut self, min: &[u8], max: &[u8]) -> Relation {
        self.0.compare(min, max)
    }

    fn inside(&mut self, doc_ids: &[i32]) {
        self.0.visit_many(doc_ids);
    }

    fn grow(&mut self, count: usize) {
        self.0.grow(count);
    }
}

fn read_leaf_block(
    input: &mut SliceInput,
    field: &PointsField,
    out: &mut Vec<Point>,
) -> Result<Option<(Vec<u8>, Vec<u8>)>> {
    let mut buf = LeafBuf::default();
    Ok(read_leaf_block_into(input, field, &mut buf, out)?.then_some((buf.min, buf.max)))
}

/// `buf` is the caller's reusable decode scratch; its contents are
/// replaced. Returns whether the leaf carried its own bounding box, which
/// is then in `buf.min`/`buf.max`.
fn read_leaf_block_into<S: LeafSink>(
    input: &mut SliceInput,
    field: &PointsField,
    buf: &mut LeafBuf,
    out: &mut S,
) -> Result<bool> {
    let count = read_leaf_count(input, field)?;
    read_doc_ids_into(input, count, &mut buf.doc_ids, field.bkd_version)?;
    let doc_ids: &[i32] = &buf.doc_ids;

    let num_dims = field.num_dims as usize;
    let num_index_dims = field.num_index_dims as usize;
    let bytes_per_dim = field.bytes_per_dim as usize;
    let packed_bytes_length = field.packed_bytes_length();

    let common_prefix_lengths = &mut buf.prefix_lens;
    common_prefix_lengths.clear();
    common_prefix_lengths.resize(num_dims, 0);
    let scratch_value = &mut buf.value;
    scratch_value.clear();
    scratch_value.resize(packed_bytes_length, 0);
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
        // already cover every byte of every dimension):
        // `visitUniqueRawDocValues`, one `visit(iterator, value)`.
        out.grow(count);
        out.run(doc_ids, scratch_value);
        return Ok(false);
    }

    let mut bound = false;
    if num_index_dims != 1 {
        // The index gives a (possibly looser) per-leaf bounding box for the
        // indexed dimensions when there's more than one; read it (rather
        // than merely skipping past it) so callers -- notably
        // `lucene_index::check_index`'s structural-invariant checker -- can
        // cross-check every point's value against it independently of the
        // field-wide box in `.kdm`.
        let min_bound = &mut buf.min;
        min_bound.clear();
        min_bound.resize(field.packed_index_bytes_length(), 0);
        let max_bound = &mut buf.max;
        max_bound.clear();
        max_bound.resize(field.packed_index_bytes_length(), 0);
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
        // `visitDocValuesWithCardinality`'s second look at the leaf: its own
        // box is often much tighter than the cell the index gave it.
        match out.leaf_bounds(min_bound, max_bound) {
            Relation::CellOutsideQuery => return Ok(true),
            Relation::CellInsideQuery => {
                out.grow(count);
                out.inside(doc_ids);
                return Ok(true);
            }
            Relation::CellCrossesQuery => out.grow(count),
        }
        bound = true;
    } else {
        out.grow(count);
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
            read_suffix_bytes(input, common_prefix_lengths, bytes_per_dim, scratch_value)?;
            // ARITH: `length <= count - i`, so `end <= count`.
            #[allow(clippy::arithmetic_side_effects)]
            let end = i + length;
            // `visitSparseRawDocValues`: one `visit(iterator, value)` per run.
            out.run(&doc_ids[i..end], scratch_value);
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
                read_suffix_bytes(input, common_prefix_lengths, bytes_per_dim, scratch_value)?;
                out.point(doc_id, scratch_value);
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
/// How a leaf's doc ids are encoded, for the two encodings a visitor can
/// take without the ids being listed.
enum DocIdsShape {
    /// `CONTINUOUS_IDS`: `start..start + count`.
    Range(i32),
    /// `BITSET_IDS`: the set bits of the words read into the caller's
    /// buffer, bit 0 of word 0 being this doc id.
    Bits(i32),
    /// Anything else (also a range that would wrap past `i32::MAX`): read
    /// the ids with [`read_doc_ids_into`] from the same position.
    List,
}

/// Reads a leaf's doc-id header and, for a `BITSET_IDS` leaf, its words
/// into `words`, checking them as [`read_doc_ids_into`] does. Leaves
/// `input` anywhere; a caller that gets [`DocIdsShape::List`] or does not
/// take the shape seeks back and reads the ids.
fn input_doc_ids_shape(
    input: &mut SliceInput,
    count: usize,
    words: &mut Vec<u64>,
) -> Result<DocIdsShape> {
    match input.read_byte()? as i8 {
        CONTINUOUS_IDS => {
            let start = input.read_vint()?;
            let fits =
                start >= 0 && i64::from(start).saturating_add(count as i64) <= i64::from(i32::MAX);
            Ok(if fits {
                DocIdsShape::Range(start)
            } else {
                DocIdsShape::List
            })
        }
        BITSET_IDS => {
            let offset_words = input.read_vint()?;
            let long_len = input.read_vint()?;
            // The same bounds `read_bitset_ids` enforces; anything it would
            // reject goes back to it for the error.
            let base = i64::from(offset_words).saturating_mul(64);
            let past_end = base.saturating_add(i64::from(long_len).saturating_mul(64));
            if offset_words < 0 || long_len < 0 || past_end > i64::from(i32::MAX) {
                return Ok(DocIdsShape::List);
            }
            let long_len = long_len as usize;
            if input.remaining() < long_len.saturating_mul(8) {
                return Ok(DocIdsShape::List);
            }
            let bytes = take_bytes(input, long_len.saturating_mul(8), "bitset doc ids")?;
            words.clear();
            words.extend(
                bytes
                    .chunks_exact(8)
                    .map(|w| u64::from_le_bytes([w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]])),
            );
            let set: u64 = words.iter().map(|w| u64::from(w.count_ones())).sum();
            if set != count as u64 {
                return Ok(DocIdsShape::List);
            }
            Ok(DocIdsShape::Bits(base as i32))
        }
        _ => Ok(DocIdsShape::List),
    }
}

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
    let (doc_base, words) = read_bitset_header(input)?;
    let long_len = words.len() / 8;

    // `count` is the writer's own set-bit count, so a well-formed block
    // reserves exactly what it needs; a corrupt one is capped by the number of
    // bits that actually exist.
    out.reserve(count.min(long_len.saturating_mul(64)));
    for (word_idx, word) in BitWords(words).iter().enumerate() {
        // ARITH: `read_bitset_header` established `doc_base + long_len * 64
        // <= i32::MAX` with `doc_base >= 0`, so `long_len * 64 <= i32::MAX`
        // and `word_idx < long_len` makes `word_idx * 64` fit an `i32`; the
        // sum with `doc_base` is bounded by the same inequality.
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

/// A bitset leaf's header (`offsetWords`, `longLen`), checked, and its words:
/// the doc id of the first word's bit 0, and the words' little-endian bytes
/// (`longLen * 8` of them).
fn read_bitset_header<'a>(input: &mut SliceInput<'a>) -> Result<(i32, &'a [u8])> {
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
    // Read in place off the mapped file, eight bytes per word.
    let words = take_bytes(input, long_len.saturating_mul(8), "bitset doc ids")?;
    Ok((doc_base as i32, words))
}

/// A bitset leaf's words as stored: little-endian `u64`s over a byte slice
/// whose length is a multiple of eight. Bit `b` of word `i` is doc
/// `doc_base + 64 * i + b`.
#[derive(Clone, Copy)]
pub struct BitWords<'a>(&'a [u8]);

impl BitWords<'_> {
    /// The words in order.
    pub fn iter(&self) -> impl Iterator<Item = u64> + '_ {
        self.0
            .chunks_exact(8)
            .map(|w| u64::from_le_bytes([w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]]))
    }

    /// How many words there are.
    pub fn len(&self) -> usize {
        self.0.len() / 8
    }

    /// Whether there are none.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
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

/// A little-endian `u32` from a four-byte chunk.
#[inline(always)]
fn le_u32(w: &[u8]) -> u32 {
    u32::from_le_bytes([w[0], w[1], w[2], w[3]])
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
    // Straight from the words, each pass a plain map the compiler unrolls
    // and vectorises: their high 21 bits are the first `num_ints` ids, and
    // the low 11 bits of word `i` and of word `i + one_third` make id
    // `num_ints + i`.
    out.resize(count, 0);
    let (high, low) = out.split_at_mut(num_ints);
    for (slot, w) in high.iter_mut().zip(words.chunks_exact(4)) {
        *slot = (le_u32(w) >> 11) as i32;
    }
    let (first, second) = words.split_at(one_third.saturating_mul(4));
    for ((slot, a), b) in low
        .iter_mut()
        .zip(first.chunks_exact(4))
        .zip(second.chunks_exact(4))
    {
        *slot = ((le_u32(a) & 0x7FF) | ((le_u32(b) & 0x7FF) << 11)) as i32;
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
    // As for `BPV_21`, straight from the words: their high 24 bits are the
    // first `num_ints` ids, and the low bytes of words `i`, `i + quarter`
    // and `i + 2 * quarter` make id `num_ints + i`.
    out.resize(count, 0);
    let (high, low) = out.split_at_mut(num_ints);
    for (slot, w) in high.iter_mut().zip(words.chunks_exact(4)) {
        *slot = (le_u32(w) >> 8) as i32;
    }
    let (first, rest) = words.split_at(quarter.saturating_mul(4));
    let (second, third) = rest.split_at(quarter.saturating_mul(4));
    for (((slot, a), b), c) in low
        .iter_mut()
        .zip(first.chunks_exact(4))
        .zip(second.chunks_exact(4))
        .zip(third.chunks_exact(4))
    {
        *slot =
            ((le_u32(a) & 0xFF) | ((le_u32(b) & 0xFF) << 8) | ((le_u32(c) & 0xFF) << 16)) as i32;
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
    /// The merged field's name, which a `writeField` override may key on
    /// (`RandomCodec` seeds its split choices from its hash).
    pub field_name: String,
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
    write_merged_with_mode(
        fields,
        max_points_in_leaf_node,
        max_doc,
        temp_dir,
        temp_prefix,
        segment_id,
        segment_suffix,
        true,
    )
}

/// `Lucene90PointsWriter.merge`'s native-reader dispatch. `native_merge` is
/// true only when every original source has a native points reader; even a
/// null reader selects Java's generic add/finish path for one dimension.
/// Empty fields retain their declared schema outside these files and add no
/// BKD entry; an empty operation still writes the complete header/footer triple.
#[allow(clippy::too_many_arguments)]
pub fn write_merged_with_mode(
    fields: Vec<MergePointsField>,
    max_points_in_leaf_node: i32,
    max_doc: i32,
    temp_dir: &dyn lucene_store::directory::Directory,
    temp_prefix: &str,
    segment_id: &[u8; codec_util::ID_LENGTH],
    segment_suffix: &str,
    native_merge: bool,
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    write_fields(
        fields.into_iter().map(Ok::<_, Error>),
        max_points_in_leaf_node,
        DEFAULT_MAX_MB_SORT_IN_HEAP,
        None,
        max_doc,
        temp_dir,
        (temp_prefix, segment_id, segment_suffix),
        native_merge,
    )
}

/// [`write_merged_with_mode`] taking each field from `fields` as it is
/// written, so a merge gathers one field's points, writes its tree and drops
/// them before gathering the next, instead of holding every field at once.
#[allow(clippy::too_many_arguments)]
pub fn write_merged_streaming<E: From<Error>>(
    fields: impl IntoIterator<Item = std::result::Result<MergePointsField, E>>,
    max_points_in_leaf_node: i32,
    max_doc: i32,
    temp_dir: &dyn lucene_store::directory::Directory,
    temp_prefix: &str,
    segment_id: &[u8; codec_util::ID_LENGTH],
    segment_suffix: &str,
    native_merge: bool,
) -> std::result::Result<PointsFiles, E> {
    write_fields(
        fields,
        max_points_in_leaf_node,
        DEFAULT_MAX_MB_SORT_IN_HEAP,
        None,
        max_doc,
        temp_dir,
        (temp_prefix, segment_id, segment_suffix),
        native_merge,
    )
}

/// A points writer's finished `.kdm`, `.kdi` and `.kdd`.
type PointsFiles = (Vec<u8>, Vec<u8>, Vec<u8>);

/// One sorted segment's `(packed value, doc)` points, fed to the 1-D BKD merge.
type MergeSource<'a> = Box<dyn Iterator<Item = (&'a [u8], i32)> + 'a>;

/// The `.kdm`/`.kdi`/`.kdd` of [`write_merged_with_mode`] and [`write_added`]:
/// `Lucene90PointsWriter` with its leaf size and heap budget, and a
/// subclass's `split` override when `split` is given.
#[allow(clippy::too_many_arguments)]
fn write_fields<E: From<Error>>(
    fields: impl IntoIterator<Item = std::result::Result<MergePointsField, E>>,
    max_points_in_leaf_node: i32,
    max_mb_sort_in_heap: f64,
    split: Option<SplitFor<'_>>,
    max_doc: i32,
    temp_dir: &dyn lucene_store::directory::Directory,
    (temp_prefix, segment_id, segment_suffix): (&str, &[u8; codec_util::ID_LENGTH], &str),
    native_merge: bool,
) -> std::result::Result<PointsFiles, E> {
    let mut out = PointsOutputs::new(segment_id, segment_suffix);
    for field in fields {
        write_field(
            field?,
            &mut out,
            max_points_in_leaf_node,
            max_mb_sort_in_heap,
            split,
            max_doc,
            temp_dir,
            temp_prefix,
            native_merge,
        )?;
    }
    Ok(out.finish())
}

/// One field's BKD tree into `out`.
#[allow(clippy::too_many_arguments)]
fn write_field(
    field: MergePointsField,
    out: &mut PointsOutputs,
    max_points_in_leaf_node: i32,
    max_mb_sort_in_heap: f64,
    split: Option<SplitFor<'_>>,
    max_doc: i32,
    temp_dir: &dyn lucene_store::directory::Directory,
    temp_prefix: &str,
    native_merge: bool,
) -> Result<()> {
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
        max_mb_sort_in_heap,
        field.max_point_count,
        bkd_writer::VERSION_CURRENT,
    )?;
    if let Some(split) = split.and_then(|split| split(&field.field_name)) {
        writer = writer.with_split(split);
    }
    let plan = if native_merge && field.num_dims == 1 {
        let sources: Vec<MergeSource<'_>> = field
            .sources
            .iter()
            .map(|s| {
                Box::new(s.values.chunks_exact(stride).zip(s.docs.iter().copied()))
                    as MergeSource<'_>
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
    Ok(())
}

/// A field's `BKDWriter.split` override, by field name, if any.
pub type SplitFor<'s> = &'s dyn Fn(&str) -> Option<bkd_writer::SplitOverride<'static>>;

/// A `Lucene90PointsWriter` subclass whose `writeField` adds every point
/// of a field to a fresh `BKDWriter` and finishes it, as `RandomCodec`'s
/// points format does at flush and, through `PointsWriter.merge`'s
/// `mergeOneField`, at merge.
pub struct AddFinishPoints<'s> {
    pub max_points_in_leaf_node: i32,
    pub max_mb_sort_in_heap: f64,
    /// The field's `BKDWriter.split` override.
    pub split: SplitFor<'s>,
}

/// The `.kdm`, `.kdi` and `.kdd` of an [`AddFinishPoints`] writer: each
/// field's points added in source order (a flush passes one source, its
/// insertion order), `totalPointCount` its `max_point_count`, spilling to
/// `temp_dir` past the heap budget. A field whose every point was deleted
/// writes nothing.
#[allow(clippy::too_many_arguments)]
pub fn write_added(
    fields: Vec<MergePointsField>,
    options: &AddFinishPoints<'_>,
    max_doc: i32,
    temp_dir: &dyn lucene_store::directory::Directory,
    temp_prefix: &str,
    segment_id: &[u8; codec_util::ID_LENGTH],
    segment_suffix: &str,
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    write_fields(
        fields.into_iter().map(Ok::<_, Error>),
        options.max_points_in_leaf_node,
        options.max_mb_sort_in_heap,
        Some(options.split),
        max_doc,
        temp_dir,
        (temp_prefix, segment_id, segment_suffix),
        false,
    )
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
mod tests;

#[cfg(test)]
mod config_validation_tests;

#[cfg(test)]
mod intersect_tests;
