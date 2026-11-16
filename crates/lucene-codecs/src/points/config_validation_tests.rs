#![cfg(test)]
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
/// `maxPointsInLeafNode` sizes the
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
                    reader.range_query(0, &long_sortable_bytes(100), &long_sortable_bytes(700))?;
                    reader.intersect(0, &mut CountingVisitor::default())?;
                    Ok::<(), Error>(())
                }));
                match outcome {
                    Ok(Ok(())) => {}
                    Ok(Err(_)) => rejected[which] += 1,
                    Err(_) => panics.push(format!("{}: offset={offset} bit={bit}", names[which])),
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
