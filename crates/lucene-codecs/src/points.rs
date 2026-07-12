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

    // The index gives a (possibly looser) per-leaf bounding box for the
    // indexed dimensions when there's more than one; without a query to
    // prune against, this port just reads past it to stay aligned.
    if num_index_dims != 1 {
        for &prefix in common_prefix_lengths.iter().take(num_index_dims) {
            input.skip(bytes_per_dim - prefix)?;
            input.skip(bytes_per_dim - prefix)?;
        }
    }

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
    } else if compressed_dim == -2 {
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

/// Port of `DocIdsWriter.readInts` -- decodes `count` doc ids using
/// whichever of the current encodings the leaf's leading marker byte
/// selects. `LEGACY_DELTA_VINT` (marker 0) is not supported: per Java's own
/// comment, "these signs are legacy, should no longer be used in the
/// writing side," so no current write can produce it.
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
        other => Err(Error::UnsupportedDocIdsEncoding(other)),
    }
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
        let bytes = [0u8]; // LEGACY_DELTA_VINT marker, not supported
        let mut input = SliceInput::new(&bytes);
        assert!(matches!(
            read_doc_ids(&mut input, 1),
            Err(Error::UnsupportedDocIdsEncoding(0))
        ));
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
                                   // min/max box (2 dims x (min tail + max tail), 1 byte each) -- values
                                   // are irrelevant, just skipped.
        bytes.extend_from_slice(&[0xEE, 0xEE, 0xEE, 0xEE]);
        bytes.push(0x00); // compressedDim = 0
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
}
