//! Port of `org.apache.lucene.codecs.lucene90.Lucene90PointsFormat` /
//! `org.apache.lucene.util.bkd.BKDReader` (`.kdm` meta + `.kdi` index +
//! `.kdd` data) — read-only, block KD-tree point values (used by numeric
//! range/point fields: `IntPoint`, `LongPoint`, `LatLonPoint`, etc.).
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

/// Default `BKDConfig`/`Lucene90PointsWriter` leaf size -- the only leaf size
/// this port's write side has been verified against (see [`write`]'s module
/// doc for the single-leaf scope).
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

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Store(#[from] lucene_store::Error),
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
}

pub type Result<T> = std::result::Result<T, Error>;

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
}

impl PointsField {
    fn packed_bytes_length(&self) -> usize {
        (self.num_dims * self.bytes_per_dim) as usize
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

pub struct PointsReader<'d> {
    kdi: &'d [u8],
    kdd: &'d [u8],
    fields: Vec<(i32, PointsField)>,
}

/// Parses `.kdm`+`.kdi`+`.kdd` (already read into memory).
pub fn open<'d>(
    kdm: &[u8],
    kdi: &'d [u8],
    kdd: &'d [u8],
    segment_id: &[u8; codec_util::ID_LENGTH],
    segment_suffix: &str,
) -> Result<PointsReader<'d>> {
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

    Ok(PointsReader { kdi, kdd, fields })
}

fn read_field_meta(meta_input: &mut SliceInput) -> Result<PointsField> {
    // `check_header` enforces the exact version (min == max == CURRENT)
    // itself, surfacing a mismatch as a `Corrupted` error -- no separate
    // check needed here.
    codec_util::check_header(
        meta_input,
        BKD_CODEC_NAME,
        BKD_VERSION_CURRENT,
        BKD_VERSION_CURRENT,
    )?;

    let num_dims = meta_input.read_vint()?;
    let num_index_dims = meta_input.read_vint()?;
    let max_points_in_leaf_node = meta_input.read_vint()?;
    let bytes_per_dim = meta_input.read_vint()?;
    let num_leaves = meta_input.read_vint()?;

    let packed_index_bytes_length = (num_index_dims * bytes_per_dim) as usize;
    let mut min_packed_value = vec![0u8; packed_index_bytes_length];
    let mut max_packed_value = vec![0u8; packed_index_bytes_length];
    meta_input.read_bytes(&mut min_packed_value)?;
    meta_input.read_bytes(&mut max_packed_value)?;

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
    })
}

impl<'d> PointsReader<'d> {
    pub fn field(&self, field_number: i32) -> Option<&PointsField> {
        self.fields
            .iter()
            .find(|(n, _)| *n == field_number)
            .map(|(_, f)| f)
    }

    /// Decodes every point (doc id + full packed value) for `field_number`,
    /// across all its leaves, in leaf (left-to-right) order.
    pub fn decode_all_points(&self, field_number: i32) -> Result<Vec<Point>> {
        let field = self
            .field(field_number)
            .ok_or(Error::IllegalFieldNumber(field_number))?;

        let inner_nodes = self
            .kdi
            .get(
                field.index_start_pointer as usize
                    ..(field.index_start_pointer + field.num_index_bytes as i64) as usize,
            )
            .ok_or(lucene_store::Error::Eof { offset: 0 })?;
        let leaf_fps = decode_leaf_pointers(inner_nodes, field)?;

        let mut points = Vec::with_capacity(field.point_count as usize);
        let mut kdd_input = SliceInput::new(self.kdd);
        for &fp in &leaf_fps {
            kdd_input.seek(fp as usize)?;
            read_leaf_block(&mut kdd_input, field, &mut points)?;
        }
        Ok(points)
    }
}

/// Walks the packed binary tree in `.kdi` and returns every leaf's `.kdd`
/// file pointer, in left-to-right (in-order) order. See the module doc for
/// why this never seeks: a leaf is a node whose id is `>= num_leaves`
/// (`leafNodeOffset`), and the tree's root is node 1.
fn decode_leaf_pointers(inner_nodes: &[u8], field: &PointsField) -> Result<Vec<i64>> {
    let mut input = SliceInput::new(inner_nodes);
    let mut leaves = Vec::with_capacity(field.num_leaves as usize);
    // The root is always reached as if it were a "right" child of an
    // implicit level 0 baseline of 0 -- `BKDReader`'s constructor calls
    // `readNodeData(false)` for the root, which always reads one leading
    // FP-delta vlong regardless of leaf-ness.
    let root_fp = input.read_vlong()?;
    walk_node(&mut input, 1, root_fp, field, &mut leaves)?;
    Ok(leaves)
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
    let code = input.read_vint()?;
    let code = code / field.num_index_dims;
    let prefix = code % (1 + field.bytes_per_dim);
    let suffix = field.bytes_per_dim - prefix;
    if suffix > 0 {
        input.skip((suffix - 1) as usize)?;
    }

    let left_child = node_id * 2;
    if left_child < field.num_leaves {
        input.read_vint()?; // leftNumBytes: a skip-ahead hint, unused (see module doc)
    }

    // Left child inherits this node's FP baseline unchanged.
    walk_node(input, left_child, fp, field, leaves)?;
    // Right child's FP is a delta from this node's baseline, read
    // immediately after the (fully consumed) left subtree.
    let right_delta = input.read_vlong()?;
    walk_node(input, node_id * 2 + 1, fp + right_delta, field, leaves)?;
    Ok(())
}

/// Decodes one leaf block (doc ids + packed values) at the data input's
/// current position, appending every point to `out`.
fn read_leaf_block(
    input: &mut SliceInput,
    field: &PointsField,
    out: &mut Vec<Point>,
) -> Result<()> {
    let count = input.read_vint()? as usize;
    let doc_ids = read_doc_ids(input, count)?;

    let num_dims = field.num_dims as usize;
    let num_index_dims = field.num_index_dims as usize;
    let bytes_per_dim = field.bytes_per_dim as usize;
    let packed_bytes_length = field.packed_bytes_length();

    let mut common_prefix_lengths = vec![0usize; num_dims];
    let mut scratch_value = vec![0u8; packed_bytes_length];
    for (dim, prefix_len) in common_prefix_lengths.iter_mut().enumerate() {
        let prefix = input.read_vint()? as usize;
        *prefix_len = prefix;
        if prefix > 0 {
            input.read_bytes(
                &mut scratch_value[dim * bytes_per_dim..dim * bytes_per_dim + prefix],
            )?;
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
    if compressed_dim < -2 || compressed_dim as i32 >= field.num_dims {
        return Err(Error::UnsupportedCompressedDim(compressed_dim));
    }

    if compressed_dim == -1 {
        // Every point in this leaf has the identical value (common prefixes
        // already cover every byte of every dimension).
        for &doc_id in &doc_ids {
            out.push(Point {
                doc_id,
                packed_value: scratch_value.clone(),
            });
        }
        return Ok(());
    }

    if num_index_dims != 1 {
        // The index gives a (possibly looser) per-leaf bounding box for the
        // indexed dimensions when there's more than one; without a query to
        // prune against, this port just reads past it to stay aligned.
        for &prefix in common_prefix_lengths.iter().take(num_index_dims) {
            input.skip(bytes_per_dim - prefix)?;
            input.skip(bytes_per_dim - prefix)?;
        }
    }

    if compressed_dim == -2 {
        let mut i = 0usize;
        while i < count {
            let length = input.read_vint()? as usize;
            if i + length > count {
                return Err(Error::SubBlockCountMismatch {
                    expected: count,
                    actual: i + length,
                });
            }
            for dim in 0..num_dims {
                let prefix = common_prefix_lengths[dim];
                input.read_bytes(
                    &mut scratch_value[dim * bytes_per_dim + prefix..(dim + 1) * bytes_per_dim],
                )?;
            }
            for &doc_id in &doc_ids[i..i + length] {
                out.push(Point {
                    doc_id,
                    packed_value: scratch_value.clone(),
                });
            }
            i += length;
        }
        debug_assert_eq!(i, count);
    } else {
        let compressed_dim = compressed_dim as usize;
        let compressed_byte_offset =
            compressed_dim * bytes_per_dim + common_prefix_lengths[compressed_dim];
        common_prefix_lengths[compressed_dim] += 1;
        let mut i = 0usize;
        while i < count {
            scratch_value[compressed_byte_offset] = input.read_byte()?;
            let run_len = input.read_byte()? as usize;
            if i + run_len > count {
                return Err(Error::SubBlockCountMismatch {
                    expected: count,
                    actual: i + run_len,
                });
            }
            for j in 0..run_len {
                for dim in 0..num_dims {
                    let prefix = common_prefix_lengths[dim];
                    input.read_bytes(
                        &mut scratch_value[dim * bytes_per_dim + prefix..(dim + 1) * bytes_per_dim],
                    )?;
                }
                out.push(Point {
                    doc_id: doc_ids[i + j],
                    packed_value: scratch_value.clone(),
                });
            }
            i += run_len;
        }
        debug_assert_eq!(i, count);
    }

    Ok(())
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

/// Port of `DocIdsWriter.readInts` -- decodes `count` doc ids using
/// whichever encoding the leaf's leading marker byte selects.
fn read_doc_ids(input: &mut SliceInput, count: usize) -> Result<Vec<i32>> {
    let bpv = input.read_byte()? as i8;
    match bpv {
        CONTINUOUS_IDS => {
            let start = input.read_vint()?;
            Ok((0..count as i32).map(|i| start + i).collect())
        }
        BITSET_IDS => read_bitset_ids(input, count),
        DELTA_BPV_16 => read_delta_bpv16(input, count),
        BPV_21 => read_bpv21(input, count),
        BPV_24 => read_bpv24(input, count),
        BPV_32 => {
            let mut out = Vec::with_capacity(count);
            for _ in 0..count {
                out.push(input.read_i32()?);
            }
            Ok(out)
        }
        LEGACY_DELTA_VINT => read_legacy_delta_vint(input, count),
        other => Err(Error::UnsupportedDocIdsEncoding(other)),
    }
}

/// Port of `DocIdsWriter.readLegacyDeltaVInts`: each doc id is a vint delta
/// from the previous one (starting at 0), the encoding used by index
/// versions that predate `DELTA_BPV_16`/`BPV_21`/`BPV_24`/`BPV_32`. No
/// current writer in this port (or in Lucene 10.5.0) produces this, so it
/// is exercised only by hand-built unit tests, not a real-Lucene fixture.
fn read_legacy_delta_vint(input: &mut SliceInput, count: usize) -> Result<Vec<i32>> {
    let mut out = Vec::with_capacity(count);
    let mut doc = 0i32;
    for _ in 0..count {
        doc += input.read_vint()?;
        out.push(doc);
    }
    Ok(out)
}

fn read_bitset_ids(input: &mut SliceInput, count: usize) -> Result<Vec<i32>> {
    let offset_words = input.read_vint()?;
    let long_len = input.read_vint()? as usize;
    let mut words = vec![0i64; long_len];
    input.read_i64s(&mut words)?;

    let doc_base = offset_words * 64;
    let mut out = Vec::with_capacity(count);
    for (word_idx, &word) in words.iter().enumerate() {
        let mut w = word as u64;
        while w != 0 {
            let bit = w.trailing_zeros();
            out.push(doc_base + (word_idx as i32) * 64 + bit as i32);
            w &= w - 1;
        }
    }
    if out.len() != count {
        return Err(Error::SubBlockCountMismatch {
            expected: count,
            actual: out.len(),
        });
    }
    Ok(out)
}

fn read_delta_bpv16(input: &mut SliceInput, count: usize) -> Result<Vec<i32>> {
    let min = input.read_vint()?;
    let half = count / 2;
    let mut out = vec![0i32; count];
    for i in 0..half {
        let word = input.read_i32()?;
        out[i] = ((word as u32) >> 16) as i32 + min;
        out[i + half] = (word & 0xFFFF) + min;
    }
    if count % 2 == 1 {
        out[count - 1] = input.read_u16()? as i32 + min;
    }
    Ok(out)
}

fn floor_to_multiple_of_16(n: usize) -> usize {
    n & !0xF
}

fn read_bpv21(input: &mut SliceInput, count: usize) -> Result<Vec<i32>> {
    let one_third = floor_to_multiple_of_16(count / 3);
    let num_ints = one_third * 2;
    let mut scratch = vec![0i32; num_ints];
    for slot in scratch.iter_mut() {
        *slot = input.read_i32()?;
    }
    let mut out = vec![0i32; count];
    for i in 0..num_ints {
        out[i] = ((scratch[i] as u32) >> 11) as i32;
    }
    for i in 0..one_third {
        out[i + num_ints] = (scratch[i] & 0x7FF) | ((scratch[i + one_third] & 0x7FF) << 11);
    }

    let mut i = one_third * 3;
    while i + 2 < count {
        let l = input.read_i64()?;
        out[i] = (l & 0x1FFFFF) as i32;
        out[i + 1] = ((l >> 21) & 0x1FFFFF) as i32;
        out[i + 2] = (l >> 42) as i32;
        i += 3;
    }
    while i < count {
        let lo = input.read_u16()? as i32;
        let hi = input.read_byte()? as i32;
        out[i] = lo | (hi << 16);
        i += 1;
    }
    Ok(out)
}

fn read_bpv24(input: &mut SliceInput, count: usize) -> Result<Vec<i32>> {
    let quarter = count / 4;
    let num_ints = quarter * 3;
    let mut scratch = vec![0i32; num_ints];
    for slot in scratch.iter_mut() {
        *slot = input.read_i32()?;
    }
    let mut out = vec![0i32; count];
    for i in 0..num_ints {
        out[i] = ((scratch[i] as u32) >> 8) as i32;
    }
    for i in 0..quarter {
        out[i + num_ints] = (scratch[i] & 0xFF)
            | ((scratch[i + quarter] & 0xFF) << 8)
            | ((scratch[i + quarter * 2] & 0xFF) << 16);
    }

    let mut i = quarter * 4;
    while i < count {
        let lo = input.read_u16()? as i32;
        let hi = input.read_byte()? as i32;
        out[i] = lo | (hi << 16);
        i += 1;
    }
    Ok(out)
}

/// One field's input to [`write`]: `(docID, packedValue)` pairs for a field
/// with `num_dims` dimensions of `bytes_per_dim` bytes each (`packedValue`
/// is `num_dims * bytes_per_dim` bytes, each dimension's slice the sortable
/// big-endian encoding `NumericUtils.longToSortableBytes`/
/// `intToSortableBytes` already produce -- this module doesn't do that
/// conversion itself, same division of labor as the read side, which also
/// just hands back raw packed bytes). `num_dims == 1` is `LongPoint`/
/// `IntPoint`'s shape; `num_dims > 1` (e.g. 2 for `LatLonPoint`) is also
/// supported -- see [`write`]'s doc comment for the scope of that support
/// (`num_index_dims` is always treated as equal to `num_dims`; a field with
/// extra *data-only*, non-indexed dimensions is not supported).
#[derive(Debug, Clone)]
pub struct WritePointsField {
    pub field_number: i32,
    pub num_dims: i32,
    pub bytes_per_dim: i32,
    /// `(docID, packedValue)`, in any order -- [`write`] sorts (recursively,
    /// per split node -- see [`compute_leaf_plan`]) a local copy before
    /// splitting into leaves, so caller order never affects correctness.
    pub points: Vec<(i32, Vec<u8>)>,
}

/// Port of `Lucene90PointsWriter`/`BKDWriter`, scoped to **any number of
/// dimensions, any number of leaves** (multi-leaf trees and multi-dimension
/// points, e.g. `LatLonPoint`-shaped 2D fields, are both supported -- see
/// `docs/parity.md`). Produces `(.kdm, .kdi, .kdd)` bytes.
///
/// **Split algorithm**: at every recursive split, the split *dimension* is
/// chosen by [`widest_dim`] -- the dimension with the widest value range
/// (`max - min`, unsigned byte-wise) across the current point subset, ties
/// broken toward the lowest dimension index. This is a real, correct
/// heuristic in the same spirit as real `BKDWriter`'s own range-driven
/// dimension choice, not an arbitrary simplification -- for `num_dims == 1`
/// it always picks dimension 0, so the single-dimension path is unchanged.
/// Once a dimension is chosen, the current subset is sorted by that
/// dimension's bytes (unsigned byte-wise, i.e. numeric order for the
/// sortable big-endian encoding `LongPoint`/`IntPoint` produce) and split
/// exactly the way real `BKDWriter.build()` sizes its two halves --
/// `numLeaves = ceil(count / maxPointsInLeafNode)`, `numLeftLeafNodes =
/// getNumLeftLeafNodes(numLeaves)` (fill the deepest full level, then push
/// any remainder left -- see [`get_num_left_leaf_nodes`]), `mid =
/// numLeftLeafNodes * maxPointsInLeafNode`. Recursing on the left/right
/// halves with `numLeftLeafNodes`/`numLeaves - numLeftLeafNodes` leaves
/// respectively reproduces the same nearly-balanced binary tree real
/// Lucene's writer builds (verified: this is the exact formula in
/// `BKDWriter.getNumLeftLeafNodes`/`build`, not a simplification of it), so
/// no follow-up rebalancing is needed. Because each dimension can be
/// resorted at every split (a different dimension may be chosen at each
/// level), [`compute_leaf_plan`] partitions an owned `Vec` at each
/// recursion step (`Vec::split_off`) rather than reusing one shared,
/// globally-presorted array the way the single-dimension path used to --
/// there's still no per-node `radix select` the way real Lucene's
/// multi-pass/on-disk sort does; a plain `sort_by` at each node is enough at
/// the sizes this port's fixtures and tests exercise, and is stated as a
/// deliberate simplification (see `docs/parity.md`), not an attempt to
/// replicate `BKDRadixSelector`.
///
/// **Packed index (`.kdi`) construction**: leaves are written to `.kdd` in
/// left-to-right (in-order) order, recording each leaf's file pointer; a
/// second pass ([`pack_index`]) walks the same recursive split plan to
/// build the `.kdi` bytes, matching real `BKDWriter.recursePackIndex`'s
/// node layout exactly: `numLeaves == 1` writes nothing (left child) or one
/// FP-delta vlong (right child, relative to the caller's `minBlockFP`);
/// otherwise it writes (if not the tree's top call) the left subtree's FP
/// delta, then a split descriptor vint encoding `splitDim` (the dimension
/// [`widest_dim`] picked for that node) together with the split value's
/// prefix/first-diff-byte, then the left subtree's own packed bytes
/// (prefixed by a `leftNumBytes` skip-ahead vint whenever the left subtree
/// itself has more than one leaf, matching real Lucene's reader-side skip
/// optimization), then the right subtree's bytes. **Split-value delta
/// encoding matches real `BKDWriter` exactly, including across dimensions**:
/// each split's value is prefix-coded against the *previous split value
/// seen in that same dimension* via a running `last_split_values`/
/// `negative_deltas` pair **indexed by dimension** (one slot per index
/// dimension, exactly `BKDWriter.recursePackIndex`'s per-dimension
/// `lastSplitValues`/`negativeDeltas` arrays), saved and restored around
/// each child call the same way `pack_index`'s own doc comment describes --
/// see that function for the exact algorithm. This makes the packed index
/// byte-for-byte reconstructible by real `Lucene90PointsReader`'s pruning
/// path (`BKDReader.readNodeData`), which really does use the reconstructed
/// split value to decide whether to descend into a subtree at all -- see
/// `fixtures/src/VerifyPoints.java`'s bounding-box query, which forces
/// exactly that path and fails if this encoding were wrong.
///
/// **Leaf encoding choices made freely** (unchanged from the single-
/// dimension slice -- this port writes bytes real `Lucene90PointsReader`
/// can decode, not necessarily what real `BKDWriter` would have chosen):
/// common-prefix length is always written as 0, the compressed-dimension
/// marker is always `-2` with every run forced to length 1, and doc ids use
/// `CONTINUOUS_IDS` when a leaf's own ids are already an exact consecutive
/// run or plain `BPV_32` otherwise. When `num_dims > 1` each leaf also
/// writes its own (per-leaf, tighter-than-field) min/max bounding box, one
/// pair of `bytes_per_dim`-byte values per dimension -- the read side
/// ([`read_leaf_block`]) already decodes/skips this box unconditionally
/// whenever `num_index_dims != 1`, so this was already a real read-side
/// requirement, just never previously exercised by this module's own write
/// path.
///
/// **Scope**: `num_index_dims` is always treated as equal to `num_dims` --
/// a field with extra data-only (non-indexed) dimensions, real `BKDWriter`'s
/// `numDims > numIndexDims` case, is not supported (see `docs/parity.md`).
/// Empty fields (`points.is_empty()` returns [`Error::EmptyField`]) also
/// remain out of scope: real Lucene's `finish()` returns `null` and the
/// field is omitted from `.kdm` entirely in that case; this port's callers
/// are expected to simply not pass an empty field rather than replicate
/// that omission path for a case this slice's scope doesn't need.
pub fn write(
    fields: &[WritePointsField],
    max_points_in_leaf_node: i32,
    segment_id: &[u8; codec_util::ID_LENGTH],
    segment_suffix: &str,
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let mut data_out: Vec<u8> = Vec::new();
    codec_util::write_index_header(
        &mut data_out,
        DATA_CODEC_NAME,
        VERSION_CURRENT,
        segment_id,
        segment_suffix,
    );
    let mut meta_out: Vec<u8> = Vec::new();
    codec_util::write_index_header(
        &mut meta_out,
        META_CODEC_NAME,
        VERSION_CURRENT,
        segment_id,
        segment_suffix,
    );
    let mut index_out: Vec<u8> = Vec::new();
    codec_util::write_index_header(
        &mut index_out,
        INDEX_CODEC_NAME,
        VERSION_CURRENT,
        segment_id,
        segment_suffix,
    );

    for field in fields {
        write_field(
            field,
            max_points_in_leaf_node,
            &mut data_out,
            &mut index_out,
            &mut meta_out,
        )?;
    }

    // Field-loop terminator, then the two file-length fields real
    // `Lucene90PointsWriter.finish()` writes right after the footers of
    // `.kdi`/`.kdd` (so they capture each file's *total* length including
    // its own footer).
    meta_out.write_i32(-1);
    codec_util::write_footer(&mut index_out);
    codec_util::write_footer(&mut data_out);
    meta_out.write_i64(index_out.len() as i64);
    meta_out.write_i64(data_out.len() as i64);
    codec_util::write_footer(&mut meta_out);

    Ok((meta_out, index_out, data_out))
}

/// Real `BKDWriter.getNumLeftLeafNodes`: fill the deepest full level of a
/// perfect binary tree with `numLeaves` leaves, put half of that level on
/// the left, then push any leftover (unbalanced) leaves left too.
fn get_num_left_leaf_nodes(num_leaves: usize) -> usize {
    debug_assert!(num_leaves > 1);
    let last_full_level = usize::BITS - 1 - num_leaves.leading_zeros();
    let leaves_full_level = 1usize << last_full_level;
    let mut num_left = leaves_full_level / 2;
    let unbalanced = num_leaves - leaves_full_level;
    num_left += unbalanced.min(num_left);
    num_left
}

/// Computes `a - b` as an unsigned big-endian byte array the same length as
/// `a`/`b` (which must be equal length and non-empty), assuming `a >= b`
/// byte-wise -- true here since `a`/`b` are always a dimension's own max/min
/// over the same point subset. Used only to *compare* per-dimension value
/// ranges in [`widest_dim`], never written to disk: comparing two such
/// nonnegative, equal-length differences byte-wise (unsigned) orders them
/// the same way comparing the underlying numeric widths would, for any
/// `bytes_per_dim`, not just lengths that fit in a native integer.
fn unsigned_byte_sub(a: &[u8], b: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; a.len()];
    let mut borrow = 0i32;
    for i in (0..a.len()).rev() {
        let diff = a[i] as i32 - b[i] as i32 - borrow;
        if diff < 0 {
            out[i] = (diff + 256) as u8;
            borrow = 1;
        } else {
            out[i] = diff as u8;
            borrow = 0;
        }
    }
    out
}

/// This port's split-dimension heuristic (see [`write`]'s doc comment for
/// how it compares to real `BKDWriter`'s own choice): the dimension with the
/// widest value range (`max - min`, unsigned byte-wise, via
/// [`unsigned_byte_sub`]) across `points`, ties broken toward the lowest
/// dimension index. `num_dims == 1` always returns `0`.
fn widest_dim(points: &[(i32, Vec<u8>)], num_dims: usize, bytes_per_dim: usize) -> usize {
    debug_assert!(!points.is_empty());
    let mut best_dim = 0usize;
    let mut best_range: Option<Vec<u8>> = None;
    for dim in 0..num_dims {
        let lo = dim * bytes_per_dim;
        let hi = lo + bytes_per_dim;
        let mut min = &points[0].1[lo..hi];
        let mut max = min;
        for (_, v) in &points[1..] {
            let slice = &v[lo..hi];
            if slice < min {
                min = slice;
            }
            if slice > max {
                max = slice;
            }
        }
        let range = unsigned_byte_sub(max, min);
        let is_wider = match &best_range {
            Some(current_best) => range.as_slice() > current_best.as_slice(),
            None => true,
        };
        if is_wider {
            best_range = Some(range);
            best_dim = dim;
        }
    }
    best_dim
}

/// Recursively computes this field's leaves (each leaf's own point
/// sublist, left-to-right) and, for every split node, the packed value and
/// dimension that becomes/was used for the split (indexed the same way real
/// `BKDWriter` indexes `splitDimensionValues`/`splitValues`: at
/// `rightOffset - 1`, where `rightOffset = leavesOffset + numLeftLeafNodes`).
/// Mirrors real `BKDWriter.build`'s `mid = numLeftLeafNodes *
/// maxPointsInLeafNode` exactly -- see [`write`]'s doc comment. Unlike the
/// single-dimension predecessor of this function, `points` is consumed by
/// value and split with `Vec::split_off` at each node rather than indexing
/// into one shared, globally-presorted array, since a different call to
/// [`widest_dim`] (and therefore a different sort order) can happen at
/// every recursion level.
#[allow(clippy::too_many_arguments)]
fn compute_leaf_plan(
    points: Vec<(i32, Vec<u8>)>,
    leaves_offset: usize,
    num_leaves: usize,
    max_points_in_leaf_node: usize,
    num_dims: usize,
    bytes_per_dim: usize,
    leaves: &mut Vec<Vec<(i32, Vec<u8>)>>,
    split_values: &mut [Vec<u8>],
    split_dims: &mut [usize],
) {
    if num_leaves == 1 {
        leaves.push(points);
        return;
    }
    let dim = widest_dim(&points, num_dims, bytes_per_dim);
    let lo = dim * bytes_per_dim;
    let hi = lo + bytes_per_dim;
    let mut points = points;
    points.sort_by(|a, b| a.1[lo..hi].cmp(&b.1[lo..hi]));

    let num_left = get_num_left_leaf_nodes(num_leaves);
    let mid = num_left * max_points_in_leaf_node;
    let right_offset = leaves_offset + num_left;
    split_values[right_offset - 1] = points[mid].1[lo..hi].to_vec();
    split_dims[right_offset - 1] = dim;

    let right_points = points.split_off(mid);
    compute_leaf_plan(
        points,
        leaves_offset,
        num_left,
        max_points_in_leaf_node,
        num_dims,
        bytes_per_dim,
        leaves,
        split_values,
        split_dims,
    );
    compute_leaf_plan(
        right_points,
        right_offset,
        num_leaves - num_left,
        max_points_in_leaf_node,
        num_dims,
        bytes_per_dim,
        leaves,
        split_values,
        split_dims,
    );
}

/// Port of `BKDWriter.recursePackIndex`, matching real Lucene's split-value
/// prefix-coding exactly, including across dimensions: `last_split_values`/
/// `negative_deltas` are this port's `lastSplitValues`/`negativeDeltas`,
/// **one slot per index dimension** (real Lucene's own per-dimension
/// arrays -- `last_split_values[dim]` is `lastSplitValues[dim * bytesPerDim
/// .. (dim+1) * bytesPerDim]`). Both are threaded through the recursion by
/// mutable reference and saved/restored around each child call exactly the
/// way `recursePackIndex` does (see real Lucene's own comment:
/// "lastSplitValues is per-dimension split value previously seen; we use
/// this to prefix-code the split byte\[\] on each inner node") -- a left
/// child always sees `negative_deltas[splitDim] = true` while a right child
/// sees `false` (only the dimension actually split on at this node is
/// touched; every other dimension's slot is inherited unchanged from the
/// parent, exactly like real Lucene's single shared per-dimension arrays),
/// and `last_split_values[splitDim]`'s `[prefix..]` tail is temporarily
/// overwritten with this node's own split value for both children, then
/// restored to the caller's original bytes before returning (siblings must
/// see the *parent*'s state, not each other's post-recursion state).
///
/// Returns this subtree's own packed-index bytes -- the caller prefixes them
/// with a `leftNumBytes` vint when appending as a left child with more than
/// one leaf, matching real Lucene's `IndexTree` skip-ahead hint.
#[allow(clippy::too_many_arguments)]
fn pack_index(
    leaves_offset: usize,
    num_leaves: usize,
    min_block_fp: i64,
    is_left: bool,
    leaf_fps: &[i64],
    split_values: &[Vec<u8>],
    split_dims: &[usize],
    num_index_dims: usize,
    bytes_per_dim: usize,
    last_split_values: &mut [Vec<u8>],
    negative_deltas: &mut [bool],
) -> Vec<u8> {
    let mut out = Vec::new();
    if num_leaves == 1 {
        if !is_left {
            let delta = leaf_fps[leaves_offset] - min_block_fp;
            out.write_vlong(delta);
        }
        return out;
    }

    let left_block_fp = if is_left {
        min_block_fp
    } else {
        let left_fp = leaf_fps[leaves_offset];
        out.write_vlong(left_fp - min_block_fp);
        left_fp
    };

    let num_left = get_num_left_leaf_nodes(num_leaves);
    let right_offset = leaves_offset + num_left;
    let split_value = &split_values[right_offset - 1];
    let dim = split_dims[right_offset - 1];
    let last_split_value = &last_split_values[dim];

    // Find the common prefix length with the last split value seen in this
    // dimension (real Lucene's `commonPrefixComparator.compare`, a byte-wise
    // mismatch scan capped at `bytesPerDim`).
    let mut prefix = 0usize;
    while prefix < bytes_per_dim && split_value[prefix] == last_split_value[prefix] {
        prefix += 1;
    }

    let first_diff_byte_delta = if prefix < bytes_per_dim {
        let mut delta = split_value[prefix] as i32 - last_split_value[prefix] as i32;
        if negative_deltas[dim] {
            delta = -delta;
        }
        delta
    } else {
        0
    };

    // Pack the prefix, delta first-diff byte, and split dimension into a
    // single vInt: `(firstDiffByteDelta * (1 + bytesPerDim) + prefix) *
    // numIndexDims + splitDim` -- real `BKDWriter.recursePackIndex`'s exact
    // formula (for `numIndexDims == 1` this collapses to the single-
    // dimension path's old `... * 1 + 0`).
    let code = (first_diff_byte_delta * (1 + bytes_per_dim as i32) + prefix as i32)
        * num_index_dims as i32
        + dim as i32;
    out.write_vint(code);

    // Write the split value's suffix, prefix-coded vs. the parent's split
    // value: the first differing byte itself is never written raw (it's
    // recovered from `firstDiffByteDelta`), only the bytes after it.
    let suffix = bytes_per_dim - prefix;
    if suffix > 1 {
        out.write_bytes(&split_value[prefix + 1..bytes_per_dim]);
    }

    // Save the parent's tail before overwriting it so it can be restored
    // once both children have been packed. Only `last_split_values[dim]` (the
    // dimension this node split on) is touched -- every other dimension's
    // slot is untouched by this node.
    let saved_tail = last_split_values[dim][prefix..].to_vec();
    last_split_values[dim][prefix..].copy_from_slice(&split_value[prefix..]);

    let saved_negative_delta = negative_deltas[dim];
    negative_deltas[dim] = true;
    let left_bytes = pack_index(
        leaves_offset,
        num_left,
        left_block_fp,
        true,
        leaf_fps,
        split_values,
        split_dims,
        num_index_dims,
        bytes_per_dim,
        last_split_values,
        negative_deltas,
    );
    if num_left != 1 {
        out.write_vint(left_bytes.len() as i32);
    }
    out.extend_from_slice(&left_bytes);

    negative_deltas[dim] = false;
    let right_bytes = pack_index(
        right_offset,
        num_leaves - num_left,
        left_block_fp,
        false,
        leaf_fps,
        split_values,
        split_dims,
        num_index_dims,
        bytes_per_dim,
        last_split_values,
        negative_deltas,
    );
    out.extend_from_slice(&right_bytes);

    negative_deltas[dim] = saved_negative_delta;
    last_split_values[dim][prefix..].copy_from_slice(&saved_tail);

    out
}

fn write_field(
    field: &WritePointsField,
    max_points_in_leaf_node: i32,
    data_out: &mut Vec<u8>,
    index_out: &mut Vec<u8>,
    meta_out: &mut Vec<u8>,
) -> Result<()> {
    let count = field.points.len();
    if count == 0 {
        return Err(Error::EmptyField {
            field_number: field.field_number,
        });
    }
    let num_dims = field.num_dims as usize;
    let num_index_dims = num_dims; // scope: no data-only dims -- see `write`'s doc comment.
    let bytes_per_dim = field.bytes_per_dim as usize;
    let packed_bytes_length = num_dims * bytes_per_dim;
    for (i, (_, value)) in field.points.iter().enumerate() {
        if value.len() != packed_bytes_length {
            return Err(Error::WrongPackedValueLength {
                field_number: field.field_number,
                index: i,
                expected: (num_dims * bytes_per_dim) as i32,
                actual: value.len(),
            });
        }
    }

    // -- min/max packed value: computed *per dimension independently*
    // (unsigned byte-wise compare of each dimension's own bytes, not a
    // whole-value compare), matching real `BKDWriter`'s
    // `minPackedValue`/`maxPackedValue` -- for `num_dims == 1` this is the
    // same single-dimension whole-value compare the old code did. Computed
    // over caller order, independent of the split-planning sort below.
    let mut min_packed_value = vec![0u8; num_index_dims * bytes_per_dim];
    let mut max_packed_value = vec![0u8; num_index_dims * bytes_per_dim];
    for dim in 0..num_index_dims {
        let lo = dim * bytes_per_dim;
        let hi = lo + bytes_per_dim;
        let mut min = &field.points[0].1[lo..hi];
        let mut max = min;
        for (_, value) in &field.points[1..] {
            let slice = &value[lo..hi];
            if slice < min {
                min = slice;
            }
            if slice > max {
                max = slice;
            }
        }
        min_packed_value[lo..hi].copy_from_slice(min);
        max_packed_value[lo..hi].copy_from_slice(max);
    }
    let doc_count = {
        let mut docs: Vec<i32> = field.points.iter().map(|(d, _)| *d).collect();
        docs.sort_unstable();
        docs.dedup();
        docs.len() as i32
    };

    let max = max_points_in_leaf_node as usize;
    let num_leaves = count.div_ceil(max);

    let mut leaves: Vec<Vec<(i32, Vec<u8>)>> = Vec::with_capacity(num_leaves);
    let mut split_values: Vec<Vec<u8>> = vec![Vec::new(); num_leaves];
    let mut split_dims: Vec<usize> = vec![0; num_leaves];
    compute_leaf_plan(
        field.points.clone(),
        0,
        num_leaves,
        max,
        num_dims,
        bytes_per_dim,
        &mut leaves,
        &mut split_values,
        &mut split_dims,
    );
    debug_assert_eq!(leaves.len(), num_leaves);

    let mut leaf_fps: Vec<i64> = Vec::with_capacity(num_leaves);
    for leaf_points in &leaves {
        leaf_fps.push(data_out.len() as i64);
        write_leaf(
            data_out,
            leaf_points,
            num_dims,
            num_index_dims,
            bytes_per_dim,
        );
    }
    let min_leaf_block_fp = leaf_fps[0];

    // -- packed index (index_out) --
    let index_start_pointer = index_out.len() as i64;
    let mut last_split_values: Vec<Vec<u8>> = vec![vec![0u8; bytes_per_dim]; num_index_dims];
    let mut negative_deltas: Vec<bool> = vec![false; num_index_dims];
    let packed = pack_index(
        0,
        num_leaves,
        0,
        false,
        &leaf_fps,
        &split_values,
        &split_dims,
        num_index_dims,
        bytes_per_dim,
        &mut last_split_values,
        &mut negative_deltas,
    );
    index_out.write_bytes(&packed);
    let num_index_bytes = (index_out.len() as i64 - index_start_pointer) as i32;

    // -- per-field meta (meta_out) --
    meta_out.write_i32(field.field_number);
    codec_util::write_header(meta_out, BKD_CODEC_NAME, BKD_VERSION_CURRENT);
    meta_out.write_vint(num_dims as i32);
    meta_out.write_vint(num_index_dims as i32);
    meta_out.write_vint(max_points_in_leaf_node);
    meta_out.write_vint(field.bytes_per_dim);
    meta_out.write_vint(num_leaves as i32);
    meta_out.write_bytes(&min_packed_value);
    meta_out.write_bytes(&max_packed_value);
    meta_out.write_vlong(count as i64); // pointCount
    meta_out.write_vint(doc_count);
    meta_out.write_vint(num_index_bytes);
    meta_out.write_i64(min_leaf_block_fp);
    meta_out.write_i64(index_start_pointer);

    Ok(())
}

/// Writes one leaf block (doc ids + packed values) for `points` to
/// `data_out`. When `num_index_dims != 1` this also writes the leaf's own
/// (tighter-than-field) per-dimension min/max bounding box, matching what
/// [`read_leaf_block`] decodes/skips in that case.
///
/// **Field order matches real `BKDReader.visitDocValuesWithCardinality`
/// exactly: the compressed-dimension marker comes before the box, not
/// after.** This port's own read side got this wrong for one revision (see
/// [`read_leaf_block`]'s doc comment) -- the box is written (and, on the
/// read side, only decoded) when the marker isn't `-1`; this writer never
/// emits `-1`, so in practice the box is always written whenever
/// `num_index_dims != 1`.
fn write_leaf(
    data_out: &mut Vec<u8>,
    points: &[(i32, Vec<u8>)],
    num_dims: usize,
    num_index_dims: usize,
    bytes_per_dim: usize,
) {
    data_out.write_vint(points.len() as i32);
    write_leaf_doc_ids(data_out, points);
    // Common prefixes: one entry per dimension, always length 0 -- see the
    // module doc for why this is correct-but-not-maximally-compact.
    for _ in 0..num_dims {
        data_out.write_vint(0);
    }
    // compressedDim = -2 (sparse/low-cardinality run encoding), every run
    // forced to length 1.
    data_out.write_byte((-2i8) as u8);
    if num_index_dims != 1 {
        // Per-leaf min/max bounding box, one (min, max) pair of full
        // `bytes_per_dim`-byte values per index dimension (common prefix is
        // always 0 above, so nothing is elided here).
        for dim in 0..num_index_dims {
            let lo = dim * bytes_per_dim;
            let hi = lo + bytes_per_dim;
            let mut min = &points[0].1[lo..hi];
            let mut max = min;
            for (_, value) in &points[1..] {
                let slice = &value[lo..hi];
                if slice < min {
                    min = slice;
                }
                if slice > max {
                    max = slice;
                }
            }
            data_out.write_bytes(min);
            data_out.write_bytes(max);
        }
    }
    for (_, value) in points {
        data_out.write_vint(1);
        data_out.write_bytes(value);
    }
}

/// Writes this leaf's doc ids: `CONTINUOUS_IDS` when they're already an
/// exact consecutive run (cheap, common case for this slice's fixture),
/// `BPV_32` (plain 4-byte little-endian per doc) otherwise -- always
/// correct regardless of order or duplicates, unlike the bitset/delta-
/// packed encodings this port doesn't bother choosing between on write.
fn write_leaf_doc_ids(data_out: &mut Vec<u8>, points: &[(i32, Vec<u8>)]) {
    let ids: Vec<i32> = points.iter().map(|(d, _)| *d).collect();
    let is_continuous = ids.windows(2).all(|w| w[1] == w[0] + 1);
    if is_continuous {
        data_out.write_byte(CONTINUOUS_IDS as u8);
        data_out.write_vint(ids[0]);
    } else {
        data_out.write_byte(BPV_32 as u8);
        for &id in &ids {
            data_out.write_i32(id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn get_num_left_leaf_nodes_matches_bkdwriter_formula() {
        // Hand-verified against `BKDWriter.getNumLeftLeafNodes`'s own
        // worked examples (see the module doc): 3 leaves splits 2/1 (the
        // deepest full level for 3 has 2 leaves, half go left, then the one
        // unbalanced leaf also goes left).
        assert_eq!(get_num_left_leaf_nodes(2), 1);
        assert_eq!(get_num_left_leaf_nodes(3), 2);
        assert_eq!(get_num_left_leaf_nodes(4), 2);
        assert_eq!(get_num_left_leaf_nodes(5), 3);
        assert_eq!(get_num_left_leaf_nodes(7), 4);
        assert_eq!(get_num_left_leaf_nodes(8), 4);
        assert_eq!(get_num_left_leaf_nodes(9), 5);
    }

    #[test]
    fn compute_leaf_plan_distributes_all_points_and_stays_balanced() {
        // 17 points, max 4 per leaf => ceil(17/4) = 5 leaves. Every leaf
        // must respect the max, every point must appear exactly once across
        // all leaves (order across leaves isn't fixed for num_dims==1 either
        // since sorting happens per node, but total coverage must match),
        // and no leaf may be empty.
        let sorted: Vec<(i32, Vec<u8>)> = (0..17).map(|i| (i, vec![i as u8])).collect();
        let num_leaves = 5usize;
        let mut leaves = Vec::new();
        let mut split_values = vec![Vec::new(); num_leaves];
        let mut split_dims = vec![0usize; num_leaves];
        compute_leaf_plan(
            sorted.clone(),
            0,
            num_leaves,
            4,
            1,
            1,
            &mut leaves,
            &mut split_values,
            &mut split_dims,
        );
        assert_eq!(leaves.len(), num_leaves);
        let mut covered = 0usize;
        let mut all_docs: Vec<i32> = Vec::new();
        for leaf in &leaves {
            assert!(!leaf.is_empty(), "leaf must be non-empty");
            assert!(leaf.len() <= 4, "leaf exceeds max_points_in_leaf_node");
            covered += leaf.len();
            all_docs.extend(leaf.iter().map(|(doc_id, _)| *doc_id));
        }
        assert_eq!(covered, 17);
        all_docs.sort_unstable();
        assert_eq!(all_docs, (0..17).collect::<Vec<i32>>());
    }

    #[test]
    fn widest_dim_picks_the_dimension_with_the_larger_range() {
        // dim0 spans 0..=5 (range 5), dim1 spans 10..=11 (range 1) -- dim0 is
        // clearly wider.
        let points: Vec<(i32, Vec<u8>)> =
            vec![(0, vec![0, 10]), (1, vec![5, 11]), (2, vec![2, 10])];
        assert_eq!(widest_dim(&points, 2, 1), 0);
    }

    #[test]
    fn widest_dim_ties_break_toward_lowest_index() {
        let points: Vec<(i32, Vec<u8>)> = vec![(0, vec![0, 0]), (1, vec![5, 5])];
        assert_eq!(widest_dim(&points, 2, 1), 0);
    }

    #[test]
    fn widest_dim_single_dimension_always_zero() {
        let points: Vec<(i32, Vec<u8>)> = vec![(0, vec![9]), (1, vec![1])];
        assert_eq!(widest_dim(&points, 1, 1), 0);
    }

    /// Regression test for `crates/lucene-codecs/examples/write_points_fixture.rs`'s
    /// `make_points_2d` generator: its two dimensions must have comparable
    /// value ranges so [`widest_dim`] genuinely alternates between dimension
    /// 0 and dimension 1 across the tree's internal nodes, exercising
    /// `pack_index`'s per-dimension `last_split_values`/`negative_deltas`
    /// save/restore for *both* dimensions. An earlier version of that
    /// generator derived dim1 as `dim0 * 3000 + noise`, making dim1 ~3000x
    /// wider than dim0 at every node -- `widest_dim` picked dimension 1 at
    /// every single split, so dimension 0's delta-coding state was never
    /// exercised despite the module doc above claiming full interleaved-
    /// dimension coverage. This test reproduces that generator's exact data
    /// (same formulas, same `i % 3 != 0` filter, `NUM_POINTS == 200`) and
    /// asserts `compute_leaf_plan` actually records splits on both
    /// dimensions.
    #[test]
    fn widest_dim_alternates_across_tree_for_2d_fixture_data() {
        fn int_sortable_bytes(v: i32) -> [u8; 4] {
            ((v ^ i32::MIN) as u32).to_be_bytes()
        }

        const NUM_POINTS: usize = 200;
        let mut points: Vec<(i32, Vec<u8>)> = Vec::new();
        for i in 0..NUM_POINTS {
            if i % 3 != 0 {
                let d0 = ((i as i32) * 41) % 500 - 250;
                let noise = ((i as i32) * 97) % 400 - 200;
                let d1 = d0 + noise;
                let mut v = Vec::with_capacity(8);
                v.extend_from_slice(&int_sortable_bytes(d0));
                v.extend_from_slice(&int_sortable_bytes(d1));
                points.push((i as i32, v));
            }
        }

        let max_points_in_leaf_node = 8usize;
        let num_leaves = points.len().div_ceil(max_points_in_leaf_node);
        let mut leaves = Vec::new();
        let mut split_values = vec![Vec::new(); num_leaves];
        let mut split_dims = vec![usize::MAX; num_leaves];
        compute_leaf_plan(
            points,
            0,
            num_leaves,
            max_points_in_leaf_node,
            2,
            4,
            &mut leaves,
            &mut split_values,
            &mut split_dims,
        );

        // Every index except the last corresponds to a real internal-node
        // split (see compute_leaf_plan_distributes_all_points_and_stays_balanced
        // above for why); none should be left at the usize::MAX sentinel.
        let recorded_splits = &split_dims[..num_leaves - 1];
        assert!(
            recorded_splits.iter().all(|&d| d != usize::MAX),
            "expected every internal node to record a split dimension: {recorded_splits:?}"
        );
        assert!(
            recorded_splits.contains(&0),
            "expected at least one split on dimension 0: {recorded_splits:?}"
        );
        assert!(
            recorded_splits.contains(&1),
            "expected at least one split on dimension 1: {recorded_splits:?}"
        );
    }

    #[test]
    fn unsigned_byte_sub_multi_byte_borrow() {
        assert_eq!(
            unsigned_byte_sub(&[0x01, 0x00], &[0x00, 0x01]),
            vec![0x00, 0xFF]
        );
        assert_eq!(
            unsigned_byte_sub(&[0xFF, 0xFF], &[0x00, 0x00]),
            vec![0xFF, 0xFF]
        );
        assert_eq!(unsigned_byte_sub(&[0x05], &[0x05]), vec![0x00]);
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
    fn widest_dim_picks_last_dim_when_only_it_varies() {
        // dims 0 and 1 are identical across every point (zero range); only
        // dim 2 varies. A naive "cycle through dimensions" splitter would
        // pick 0 or 1 at some point; the real range-driven heuristic must
        // pick dim 2 every time since it's the only one with any spread.
        let points: Vec<(i32, Vec<u8>)> = (0..8).map(|i| (i, vec![1, 1, i as u8])).collect();
        assert_eq!(widest_dim(&points, 3, 1), 2);
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
    fn write_rejects_empty_field() {
        let field = WritePointsField {
            field_number: 0,
            num_dims: 1,
            bytes_per_dim: 4,
            points: vec![],
        };
        assert!(matches!(
            write(&[field], 512, &id(), ""),
            Err(Error::EmptyField { field_number: 0 })
        ));
    }

    #[test]
    fn write_rejects_wrong_packed_value_length() {
        let field = WritePointsField {
            field_number: 0,
            num_dims: 1,
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
            bytes_per_dim: 4,
            points: vec![(0, vec![0, 0, 0, 1])],
        };
        let (kdm, kdi, kdd) = write(&[field], 512, &id(), "").unwrap();
        let wrong_id = [9u8; codec_util::ID_LENGTH];
        assert!(open(&kdm, &kdi, &kdd, &wrong_id, "").is_err());
    }

    /// Mirrors real `BKDReader.readNodeData`'s split-value reconstruction,
    /// generalized to `num_index_dims >= 1`, closely enough to prove
    /// [`pack_index`]'s delta-coding round-trips: walks the packed index the
    /// same way [`walk_node`] does, but also decodes each inner node's
    /// `code` into `splitDim`/`prefix`/`firstDiffByteDelta` and reconstructs
    /// the split value against running per-dimension
    /// `last_split_values`/`negative_deltas` arrays, exactly like the real
    /// reader's `splitValuesStack`/`negativeDeltas`.
    #[allow(clippy::too_many_arguments)]
    fn reconstruct_split_values(
        input: &mut SliceInput,
        node_id: usize,
        num_leaves: usize,
        num_index_dims: usize,
        bytes_per_dim: usize,
        last_split_values: &mut [Vec<u8>],
        negative_deltas: &mut [bool],
        out: &mut Vec<(usize, usize, Vec<u8>)>,
    ) {
        if node_id >= num_leaves {
            return;
        }

        let code = input.read_vint().unwrap();
        let dim = (code as usize) % num_index_dims;
        let code = code / num_index_dims as i32;
        let prefix = (code % (1 + bytes_per_dim as i32)) as usize;
        let suffix = bytes_per_dim - prefix;

        let mut value = last_split_values[dim].clone();
        if suffix > 0 {
            let mut first_diff_byte_delta = code / (1 + bytes_per_dim as i32);
            if negative_deltas[dim] {
                first_diff_byte_delta = -first_diff_byte_delta;
            }
            value[prefix] = (value[prefix] as i32 + first_diff_byte_delta) as u8;
            if suffix > 1 {
                input
                    .read_bytes(&mut value[prefix + 1..bytes_per_dim])
                    .unwrap();
            }
        }
        out.push((node_id, dim, value.clone()));

        let left_child = node_id * 2;
        if left_child < num_leaves {
            input.read_vint().unwrap(); // leftNumBytes: skip-ahead hint, unused here too
        }

        let saved_tail = last_split_values[dim][prefix..].to_vec();
        last_split_values[dim][prefix..].copy_from_slice(&value[prefix..]);

        let saved_negative_delta = negative_deltas[dim];
        negative_deltas[dim] = true;
        reconstruct_split_values(
            input,
            left_child,
            num_leaves,
            num_index_dims,
            bytes_per_dim,
            last_split_values,
            negative_deltas,
            out,
        );

        let _right_fp_delta = input.read_vlong().unwrap();

        negative_deltas[dim] = false;
        reconstruct_split_values(
            input,
            node_id * 2 + 1,
            num_leaves,
            num_index_dims,
            bytes_per_dim,
            last_split_values,
            negative_deltas,
            out,
        );

        negative_deltas[dim] = saved_negative_delta;
        last_split_values[dim][prefix..].copy_from_slice(&saved_tail);
    }

    /// Builds a 5-leaf, single-dimension packed index (3 levels deep -- see
    /// the worked-out tree shape in this test's body) directly via
    /// [`compute_leaf_plan`] + [`pack_index`], then walks it with
    /// [`reconstruct_split_values`] (a close mirror of real
    /// `BKDReader.readNodeData`'s reconstruction) and asserts every inner
    /// node's reconstructed split value equals the original, at every depth
    /// -- not just the root. This is the case the bug this test guards
    /// against would have broken: with the old `prefix=0`,
    /// `firstDiffByteDelta=splitValue[0]` simplification, only the very
    /// first split (whichever inner node happens to be visited first with
    /// `last_split_value` still all zero) reconstructs correctly; every
    /// subsequent one silently reconstructs garbage once `last_split_value`
    /// has diverged from zero.
    #[test]
    fn pack_index_split_values_reconstruct_exactly_at_every_depth() {
        let bytes_per_dim = 2usize;
        // 5 leaves, distinct 2-byte big-endian values so every split value
        // differs from every other in more than trivial ways.
        let sorted: Vec<(i32, Vec<u8>)> = (0..40)
            .map(|i| (i, ((i as u16) * 137 + 11).to_be_bytes().to_vec()))
            .collect();
        let num_leaves = 5usize;
        let max_points_in_leaf_node = 8usize;
        let mut leaves = Vec::new();
        let mut split_values = vec![Vec::new(); num_leaves];
        let mut split_dims = vec![0usize; num_leaves];
        compute_leaf_plan(
            sorted,
            0,
            num_leaves,
            max_points_in_leaf_node,
            1,
            bytes_per_dim,
            &mut leaves,
            &mut split_values,
            &mut split_dims,
        );
        assert_eq!(leaves.len(), num_leaves);

        // Arbitrary but strictly increasing leaf file pointers -- pack_index
        // only cares about their deltas, and this test only checks split
        // values, not the pointers.
        let leaf_fps: Vec<i64> = (0..num_leaves as i64).map(|i| i * 1000 + 1).collect();

        let mut last_split_values = vec![vec![0u8; bytes_per_dim]; 1];
        let mut negative_deltas = vec![false; 1];
        let packed = pack_index(
            0,
            num_leaves,
            0,
            false,
            &leaf_fps,
            &split_values,
            &split_dims,
            1,
            bytes_per_dim,
            &mut last_split_values,
            &mut negative_deltas,
        );

        // Mirror decode_leaf_pointers: the top-level `is_left=false` call
        // always writes one leading root FP-delta vlong before any split
        // descriptor.
        let mut input = SliceInput::new(&packed);
        let _root_fp_delta = input.read_vlong().unwrap();

        let mut reader_last_split_values = vec![vec![0u8; bytes_per_dim]; 1];
        let mut reader_negative_deltas = vec![false; 1];
        let mut reconstructed = Vec::new();
        reconstruct_split_values(
            &mut input,
            1,
            num_leaves,
            1,
            bytes_per_dim,
            &mut reader_last_split_values,
            &mut reader_negative_deltas,
            &mut reconstructed,
        );

        // Expected split value per node id, worked out from
        // get_num_left_leaf_nodes's formula for this exact 5-leaf shape:
        // node1 (depth 0, root) splits at split_values[2];
        // node2 (depth 1, root's left child) splits at split_values[1];
        // node4 (depth 2, node2's left child) splits at split_values[0];
        // node3 (depth 1, root's right child) splits at split_values[3].
        // (node5/6/7/8/9 -- everything else -- are leaves, no split value.)
        let expected: Vec<(usize, usize, Vec<u8>)> = vec![
            (1, 0, split_values[2].clone()),
            (2, 0, split_values[1].clone()),
            (4, 0, split_values[0].clone()),
            (3, 0, split_values[3].clone()),
        ];
        assert_eq!(reconstructed.len(), expected.len());
        for (got, want) in reconstructed.iter().zip(expected.iter()) {
            assert_eq!(got, want, "node {} split value mismatch", got.0);
        }
    }

    /// Same idea as the single-dimension test above, but with `num_dims == 3`
    /// and enough points/leaves that [`widest_dim`] is guaranteed to pick
    /// different dimensions at different recursion depths (each dimension's
    /// values are drawn from a disjoint, distinctly-sized range so the
    /// widest-range dimension actually varies): proves `pack_index`'s
    /// per-dimension `last_split_values`/`negative_deltas` arrays (not a
    /// single shared one) are required for correct multi-dimension
    /// reconstruction, and that [`walk_node`]/[`read_leaf_block`] (the
    /// pre-existing, already-generic read side) agree with what this test's
    /// own `reconstruct_split_values` mirror computes.
    #[test]
    fn pack_index_multi_dim_split_values_reconstruct_exactly() {
        let num_dims = 3usize;
        let bytes_per_dim = 2usize;
        // These specific multipliers/moduli were found by brute-force search
        // (see this task's commit message/report) to be one arrangement
        // where the root and at least one deeper node pick different split
        // dimensions -- the property this test needs, not any particular
        // "geometric" meaning per dimension.
        let sorted: Vec<(i32, Vec<u8>)> = (0..80i32)
            .map(|i| {
                let d0 = ((i * 37) % 300) as u16;
                let d1 = ((i * 251) % 15000) as u16;
                let d2 = ((i * 29) % 4000) as u16;
                let mut v = Vec::with_capacity(num_dims * bytes_per_dim);
                v.extend_from_slice(&d0.to_be_bytes());
                v.extend_from_slice(&d1.to_be_bytes());
                v.extend_from_slice(&d2.to_be_bytes());
                (i, v)
            })
            .collect();
        let num_leaves = 10usize; // ceil(80 / 8), matching max_points_in_leaf_node below
        let max_points_in_leaf_node = 8usize;
        let mut leaves = Vec::new();
        let mut split_values = vec![Vec::new(); num_leaves];
        let mut split_dims = vec![0usize; num_leaves];
        compute_leaf_plan(
            sorted.clone(),
            0,
            num_leaves,
            max_points_in_leaf_node,
            num_dims,
            bytes_per_dim,
            &mut leaves,
            &mut split_values,
            &mut split_dims,
        );
        assert_eq!(leaves.len(), num_leaves);
        // Not every split need choose the same dimension -- if this
        // assertion ever fails because the test data changed, `widest_dim`
        // may still be correct; the point of this test is the multi-dim
        // decode, so relax/replace the assertion rather than the encoder.
        assert!(
            split_dims
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len()
                > 1,
            "test fixture should exercise more than one split dimension"
        );

        let leaf_fps: Vec<i64> = (0..num_leaves as i64).map(|i| i * 1000 + 1).collect();
        let mut last_split_values = vec![vec![0u8; bytes_per_dim]; num_dims];
        let mut negative_deltas = vec![false; num_dims];
        let packed = pack_index(
            0,
            num_leaves,
            0,
            false,
            &leaf_fps,
            &split_values,
            &split_dims,
            num_dims,
            bytes_per_dim,
            &mut last_split_values,
            &mut negative_deltas,
        );

        let mut input = SliceInput::new(&packed);
        let _root_fp_delta = input.read_vlong().unwrap();
        let mut reader_last_split_values = vec![vec![0u8; bytes_per_dim]; num_dims];
        let mut reader_negative_deltas = vec![false; num_dims];
        let mut reconstructed = Vec::new();
        reconstruct_split_values(
            &mut input,
            1,
            num_leaves,
            num_dims,
            bytes_per_dim,
            &mut reader_last_split_values,
            &mut reader_negative_deltas,
            &mut reconstructed,
        );

        let mut expected: Vec<(usize, usize, Vec<u8>)> = Vec::new();
        collect_expected_split_values(1, 0, num_leaves, &split_values, &split_dims, &mut expected);
        expected.sort_by_key(|(id, _, _)| *id);
        let mut reconstructed_sorted = reconstructed.clone();
        reconstructed_sorted.sort_by_key(|(id, _, _)| *id);
        assert_eq!(reconstructed_sorted, expected);
    }

    /// Walks the same recursive node-id/`leaves_offset` shape
    /// [`pack_index`]/[`walk_node`] use and collects every inner node's
    /// `(node_id, split_dim, split_value)`, purely so
    /// `pack_index_multi_dim_split_values_reconstruct_exactly` can build its
    /// own expected list without hand-working out the tree shape.
    fn collect_expected_split_values(
        node_id: usize,
        leaves_offset: usize,
        num_leaves: usize,
        split_values: &[Vec<u8>],
        split_dims: &[usize],
        out: &mut Vec<(usize, usize, Vec<u8>)>,
    ) {
        if num_leaves == 1 {
            return;
        }
        let num_left = get_num_left_leaf_nodes(num_leaves);
        let right_offset = leaves_offset + num_left;
        let idx = right_offset - 1;
        out.push((node_id, split_dims[idx], split_values[idx].clone()));
        collect_expected_split_values(
            node_id * 2,
            leaves_offset,
            num_left,
            split_values,
            split_dims,
            out,
        );
        collect_expected_split_values(
            node_id * 2 + 1,
            right_offset,
            num_leaves - num_left,
            split_values,
            split_dims,
            out,
        );
    }
}
