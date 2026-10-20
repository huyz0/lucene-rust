//! `SimpleTextBKDWriter`: SimpleText's fork of an older `BKDWriter`, building
//! one field's tree into the `.dim` file as text.
//!
//! The tree is a full binary one: `numLeaves` is the power of two that brings
//! `pointCount` down to at most `maxPointsInLeafNode` per leaf by halving
//! (`countPerLeaf = (countPerLeaf + 1) / 2`), node 1 is the root, the leaves
//! are nodes `numLeaves..2 * numLeaves`, and an inner node sends
//! `count - count / 2` points left. The split dimension is the one whose
//! cell is widest (the first such, by `NumericUtils.subtract`); the split
//! value comes from `BKDRadixSelector.select`, and each leaf is sorted with
//! `BKDRadixSelector.heapRadixSort` -- both shared with the default codec's
//! writer ([`crate::bkd_writer`]), which keeps their swap-for-swap tie order.
//!
//! Only the `add` + `finish` entry point is ported: it is the one
//! `SimpleTextPointsWriter` uses, for flushes and merges alike. Java's
//! `writeField(MutablePointTree)` / `OneDimensionBKDWriter` / `rotateToTree`
//! have no caller there.
//!
//! What differs from Java: `add`'s over-count error is
//! [`crate::bkd_writer::BkdError::IllegalArgument`] (Java: an
//! `IllegalStateException`); a doc id past `maxDoc` is not counted (Java's
//! `FixedBitSet` may throw); the tree is built into a `Vec<u8>`, not an
//! `IndexOutput`. Spilled points are switched to the heap through the shared
//! writer's `switch_to_heap`, which deletes the spill file it read (Java's
//! fork keeps it until `close`). A writer dropped before `finish` succeeds
//! deletes the temp files it made, as Java's `close` does; the default
//! codec's [`crate::bkd_writer::BkdWriter`] alone does not.

use lucene_store::directory::Directory;
use lucene_util::numeric_utils;

use super::points_writer::{
    BLOCK_COUNT, BLOCK_DOC_ID, BLOCK_FP, BLOCK_VALUE, BYTES_PER_DIM, DOC_COUNT, INDEX_COUNT,
    MAX_LEAF_POINTS, MAX_VALUE, MIN_VALUE, NUM_DATA_DIMS, NUM_INDEX_DIMS, POINT_COUNT, SPLIT_COUNT,
    SPLIT_DIM, SPLIT_VALUE,
};
use super::util::{bytes_ref_to_string, write_line};
use super::{Error, Result};
use crate::bkd_writer::{BkdConfig, BkdWriter, PathSlice, PointStore};

/// `SimpleTextBKDWriter.DEFAULT_MAX_MB_SORT_IN_HEAP`.
pub const DEFAULT_MAX_MB_SORT_IN_HEAP: f64 = 16.0;

/// `ArrayUtil.MAX_ARRAY_LENGTH` (`Integer.MAX_VALUE - 8`, as
/// `checkMaxLeafNodeCount` compares against it).
const MAX_ARRAY_LENGTH: u64 = (i32::MAX - 8) as u64;

/// `org.apache.lucene.codecs.simpletext.SimpleTextBKDWriter`.
pub struct SimpleTextBkdWriter<'d> {
    config: BkdConfig,
    inner: BkdWriter<'d>,
}

impl Drop for SimpleTextBkdWriter<'_> {
    /// Java: `SimpleTextBKDWriter.close` deleting `tempInput`: a build that
    /// failed part way leaves no temp file behind.
    fn drop(&mut self) {
        self.inner.delete_temp_files();
    }
}

impl<'d> SimpleTextBkdWriter<'d> {
    /// `new SimpleTextBKDWriter(maxDoc, tempDir, tempFileNamePrefix, config,
    /// maxMBSortInHeap, totalPointCount)`. `temp_dir` is needed only once more
    /// points come than the heap budget holds.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] with Java's messages: a negative or NaN
    /// heap size, or one that holds fewer points than a leaf.
    pub fn new(
        max_doc: usize,
        temp_dir: Option<&'d dyn Directory>,
        temp_file_name_prefix: &str,
        config: BkdConfig,
        max_mb_sort_in_heap: f64,
        total_point_count: u64,
    ) -> Result<Self> {
        // Java: SimpleTextBKDWriter.verifyParams
        if max_mb_sort_in_heap < 0.0 {
            return Err(Error::IllegalArgument(format!(
                "maxMBSortInHeap must be >= 0.0 (got: {max_mb_sort_in_heap})"
            )));
        }
        // Java: (int) ((maxMBSortInHeap * 1024 * 1024) / (bytesPerDoc * numDims)),
        // a saturating cast; `bytes_per_doc * num_dims` is at most 260 * 16.
        let divisor = config.bytes_per_doc() as f64 * config.num_dims as f64;
        let max_points_sort_in_heap = ((max_mb_sort_in_heap * 1024.0 * 1024.0) / divisor) as i32;
        if i64::from(max_points_sort_in_heap) < config.max_points_in_leaf_node as i64 {
            return Err(Error::IllegalArgument(format!(
                "maxMBSortInHeap={max_mb_sort_in_heap} only allows for maxPointsSortInHeap={max_points_sort_in_heap}, \
                 but this is less than config.maxPointsInLeafNode()={}; either increase maxMBSortInHeap or decrease config.maxPointsInLeafNode()",
                config.max_points_in_leaf_node
            )));
        }
        let heap = usize::try_from(max_points_sort_in_heap).unwrap_or(0);
        Ok(SimpleTextBkdWriter {
            config,
            inner: BkdWriter::for_simple_text(
                max_doc,
                temp_dir,
                temp_file_name_prefix,
                config,
                heap,
                total_point_count,
            ),
        })
    }

    /// `SimpleTextBKDWriter.add(packedValue, docID)`.
    ///
    /// # Errors
    /// A value of the wrong length, more points than `totalPointCount`, or a
    /// temp file that cannot be written.
    pub fn add(&mut self, packed_value: &[u8], doc_id: i32) -> Result<()> {
        Ok(self.inner.add(packed_value, doc_id)?)
    }

    /// `SimpleTextBKDWriter.getPointCount()`.
    pub fn point_count(&self) -> u64 {
        self.inner.point_count()
    }

    /// `SimpleTextBKDWriter.finish(out)`: writes the leaf blocks, then the
    /// index, to `out`; returns the index's file pointer.
    ///
    /// # Errors
    /// [`Error::IllegalState`] with Java's messages for no points or a second
    /// call; [`Error::Runtime`] for too many leaves (Java's
    /// `IllegalStateException`); temp-file errors.
    pub fn finish(&mut self, out: &mut Vec<u8>) -> Result<i64> {
        // Java: SimpleTextBKDWriter.finish
        let point_count = self.inner.point_count();
        if point_count == 0 {
            return Err(Error::IllegalState("must index at least one point"));
        }
        if self.inner.is_finished() {
            return Err(Error::IllegalState("already finished"));
        }
        let points = self.inner.take_points()?;
        let c = self.config;
        let max_leaf = c.max_points_in_leaf_node as u64;
        let mut count_per_leaf = point_count;
        let mut inner_node_count: u64 = 1;
        while count_per_leaf > max_leaf {
            // ARITH: `count_per_leaf > max_leaf >= 1`, so the sum cannot pass
            // `u64::MAX`; `inner_node_count` doubles while `count_per_leaf`
            // halves, so it stays at most `2 * point_count`.
            #[allow(clippy::arithmetic_side_effects)]
            {
                count_per_leaf = count_per_leaf.div_ceil(2);
                inner_node_count *= 2;
            }
        }
        check_max_leaf_node_count(&c, inner_node_count)?;
        // ARITH: `check_max_leaf_node_count` bounds `(1 + bytesPerDim) *
        // numLeaves` by `MAX_ARRAY_LENGTH`.
        #[allow(clippy::arithmetic_side_effects)]
        let (num_leaves, entry) = (inner_node_count as usize, 1 + c.bytes_per_dim);
        // ARITH: the same bound: `entry * num_leaves <= MAX_ARRAY_LENGTH`.
        #[allow(clippy::arithmetic_side_effects)]
        let mut split_packed_values = vec![0u8; num_leaves * entry];
        let mut leaf_block_fps = vec![0i64; num_leaves];
        let (min, max) = self.inner.bounds();
        let (min, max) = (min.to_vec(), max.to_vec());
        self.build(
            1,
            num_leaves,
            points,
            out,
            &min,
            &max,
            &mut split_packed_values,
            &mut leaf_block_fps,
        )?;
        let index_fp = fp(out);
        // `countPerLeaf <= maxPointsInLeafNode`, an `int`.
        let max_points = i32::try_from(count_per_leaf).unwrap_or(i32::MAX);
        self.write_index(out, &leaf_block_fps, &split_packed_values, max_points);
        Ok(index_fp)
    }

    /// `SimpleTextBKDWriter.writeIndex`.
    fn write_index(
        &self,
        out: &mut Vec<u8>,
        leaf_block_fps: &[i64],
        split_packed_values: &[u8],
        max_points_in_leaf_node: i32,
    ) {
        let c = self.config;
        let (min, max) = self.inner.bounds();
        write_line(out, NUM_DATA_DIMS, c.num_dims.to_string().as_bytes());
        write_line(out, NUM_INDEX_DIMS, c.num_index_dims.to_string().as_bytes());
        write_line(out, BYTES_PER_DIM, c.bytes_per_dim.to_string().as_bytes());
        write_line(
            out,
            MAX_LEAF_POINTS,
            max_points_in_leaf_node.to_string().as_bytes(),
        );
        write_line(
            out,
            INDEX_COUNT,
            leaf_block_fps.len().to_string().as_bytes(),
        );
        write_line(out, MIN_VALUE, bytes_ref_to_string(min).as_bytes());
        write_line(out, MAX_VALUE, bytes_ref_to_string(max).as_bytes());
        write_line(
            out,
            POINT_COUNT,
            self.inner.point_count().to_string().as_bytes(),
        );
        write_line(
            out,
            DOC_COUNT,
            self.inner.docs_seen_count().to_string().as_bytes(),
        );
        for fp in leaf_block_fps {
            write_line(out, BLOCK_FP, fp.to_string().as_bytes());
        }
        // ARITH: `1 + bytes_per_dim <= 17`.
        #[allow(clippy::arithmetic_side_effects)]
        let entry = 1 + c.bytes_per_dim;
        let count = split_packed_values.len().checked_div(entry).unwrap_or(0);
        write_line(out, SPLIT_COUNT, count.to_string().as_bytes());
        for e in split_packed_values.chunks_exact(entry) {
            if let Some((dim, value)) = e.split_first() {
                write_line(out, SPLIT_DIM, dim.to_string().as_bytes());
                write_line(out, SPLIT_VALUE, bytes_ref_to_string(value).as_bytes());
            }
        }
    }

    /// `SimpleTextBKDWriter.split`: the index dim whose cell is widest, the
    /// first on a tie.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] if a cell's max is below its min (Java's
    /// `NumericUtils.subtract`: `a < b`), which the build never produces.
    fn split(&self, min: &[u8], max: &[u8]) -> Result<usize> {
        let c = self.config;
        let mut split_dim = None;
        let mut best = vec![0u8; c.bytes_per_dim];
        let mut diff = vec![0u8; c.bytes_per_dim];
        for dim in 0..c.num_index_dims {
            numeric_utils::subtract(c.bytes_per_dim, dim, max, min, &mut diff)
                .map_err(|_| Error::IllegalArgument("a < b".into()))?;
            if split_dim.is_none() || diff > best {
                best.copy_from_slice(&diff);
                split_dim = Some(dim);
            }
        }
        Ok(split_dim.unwrap_or(0))
    }

    /// `SimpleTextBKDWriter.build(nodeID, leafNodeOffset, PathSlice, ...)`.
    // ARITH: node ids below `2 * leaf_node_offset` (a checked leaf count),
    // dimension offsets below `packedBytesLength`, and slice counts of at
    // least 2 points per inner node (`pointCount >= numLeaves`).
    #[allow(clippy::too_many_arguments, clippy::arithmetic_side_effects)]
    fn build(
        &mut self,
        node_id: usize,
        leaf_node_offset: usize,
        points: PathSlice,
        out: &mut Vec<u8>,
        min_packed_value: &[u8],
        max_packed_value: &[u8],
        split_packed_values: &mut [u8],
        leaf_block_fps: &mut [i64],
    ) -> Result<()> {
        let c = self.config;
        let bpd = c.bytes_per_dim;
        if node_id >= leaf_node_offset {
            self.write_leaf(node_id - leaf_node_offset, &points, out, leaf_block_fps)?;
            return Ok(());
        }
        let split_dim = if c.num_index_dims > 1 {
            self.split(min_packed_value, max_packed_value)?
        } else {
            0
        };
        let right_count = points.count / 2;
        let left_count = points.count - right_count;
        let o = split_dim * bpd;
        let common_prefix_len = min_packed_value[o..o + bpd]
            .iter()
            .zip(&max_packed_value[o..o + bpd])
            .position(|(a, b)| a != b)
            .unwrap_or(bpd);
        let (split_value, left, right) = self.inner.select(
            points.clone(),
            points.start,
            points.start + points.count,
            points.start + left_count,
            split_dim,
            common_prefix_len,
        )?;
        let address = node_id * (1 + bpd);
        // Java: (byte) splitDim, below `numIndexDims <= 8`.
        split_packed_values[address] = split_dim as u8;
        split_packed_values[address + 1..address + 1 + bpd].copy_from_slice(&split_value);
        let mut min_split = min_packed_value.to_vec();
        let mut max_split = max_packed_value.to_vec();
        min_split[o..o + bpd].copy_from_slice(&split_value);
        max_split[o..o + bpd].copy_from_slice(&split_value);
        self.build(
            2 * node_id,
            leaf_node_offset,
            left,
            out,
            min_packed_value,
            &max_split,
            split_packed_values,
            leaf_block_fps,
        )?;
        self.build(
            2 * node_id + 1,
            leaf_node_offset,
            right,
            out,
            &min_split,
            max_packed_value,
            split_packed_values,
            leaf_block_fps,
        )
    }

    /// The leaf half of `build`: the common prefixes and the dimension to
    /// sort by are computed over every record of the heap store the slice
    /// lives in (Java's `computeCommonPrefixLength(heapSource, ...)` and its
    /// `usedBytes` loop run `0..heapSource.count()`, not over the slice), then
    /// the slice is sorted and written.
    // ARITH: record offsets below `bytesPerDoc`, ranges inside the heap.
    #[allow(clippy::arithmetic_side_effects)]
    fn write_leaf(
        &mut self,
        leaf: usize,
        points: &PathSlice,
        out: &mut Vec<u8>,
        leaf_block_fps: &mut [i64],
    ) -> Result<()> {
        let c = self.config;
        let bpd = c.bytes_per_dim;
        let (store, from, to) = self.inner.switch_to_heap(points)?;
        let mut store = store.borrow_mut();
        let PointStore::Heap(heap) = &mut *store else {
            return Err(Error::IllegalState("a leaf's points are not on heap"));
        };
        // Java: SimpleTextBKDWriter.computeCommonPrefixLength
        let mut common_prefix_lengths = vec![bpd; c.num_dims];
        if heap.count() > 0 {
            let first = heap.record(0)[..c.packed_bytes_length()].to_vec();
            for i in 1..heap.count() {
                let r = heap.record(i);
                for (dim, cpl) in common_prefix_lengths.iter_mut().enumerate() {
                    if *cpl != 0 {
                        let o = dim * bpd;
                        if let Some(j) = first[o..o + *cpl]
                            .iter()
                            .zip(&r[o..o + *cpl])
                            .position(|(a, b)| a != b)
                        {
                            *cpl = j;
                        }
                    }
                }
            }
        }
        let mut sorted_dim = 0;
        let mut sorted_dim_cardinality = usize::MAX;
        for (dim, &prefix) in common_prefix_lengths.iter().enumerate() {
            if prefix < bpd {
                let offset = dim * bpd + prefix;
                let mut used = [false; 256];
                for i in 0..heap.count() {
                    used[usize::from(heap.record(i)[offset])] = true;
                }
                let cardinality = used.iter().filter(|u| **u).count();
                if cardinality < sorted_dim_cardinality {
                    sorted_dim = dim;
                    sorted_dim_cardinality = cardinality;
                }
            }
        }
        self.inner.heap_radix_sort(
            heap,
            from,
            to,
            sorted_dim,
            common_prefix_lengths[sorted_dim],
        );
        leaf_block_fps[leaf] = fp(out);
        let p = c.packed_bytes_length();
        // Java: SimpleTextBKDWriter.writeLeafBlockDocs
        write_line(out, BLOCK_COUNT, (to - from).to_string().as_bytes());
        for i in from..to {
            let r = heap.record(i);
            let doc = i32::from_be_bytes([r[p], r[p + 1], r[p + 2], r[p + 3]]);
            write_line(out, BLOCK_DOC_ID, doc.to_string().as_bytes());
        }
        // Java: SimpleTextBKDWriter.writeLeafBlockPackedValues
        for i in from..to {
            write_line(
                out,
                BLOCK_VALUE,
                bytes_ref_to_string(&heap.record(i)[..p]).as_bytes(),
            );
        }
        Ok(())
    }
}

/// `SimpleTextBKDWriter.checkMaxLeafNodeCount`.
fn check_max_leaf_node_count(c: &BkdConfig, num_leaves: u64) -> Result<()> {
    let nodes = (c.bytes_per_dim as u64)
        .checked_add(1)
        .and_then(|e| e.checked_mul(num_leaves));
    if nodes.is_none_or(|n| n > MAX_ARRAY_LENGTH) {
        return Err(Error::Runtime(format!(
            "too many nodes; increase config.maxPointsInLeafNode() (currently {}) and reindex",
            c.max_points_in_leaf_node
        )));
    }
    Ok(())
}

/// `out.getFilePointer()`.
fn fp(out: &[u8]) -> i64 {
    i64::try_from(out.len()).unwrap_or(i64::MAX)
}

#[cfg(test)]
#[allow(clippy::arithmetic_side_effects)]
mod tests {
    use super::*;

    fn config(dims: usize, index_dims: usize, bpd: usize, leaf: usize) -> BkdConfig {
        BkdConfig::new(dims, index_dims, bpd, leaf).unwrap()
    }

    #[test]
    fn a_failed_build_leaves_no_temp_file() {
        use lucene_store::directory::FsDirectory;
        let path = std::env::temp_dir().join(format!(
            "stext_bkd_tmp_{}_{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&path).unwrap();
        let dir = FsDirectory::open(&path);
        let c = config(1, 1, 1, 512);
        // A heap of exactly 512 points (5 bytes each): 600 spill from the
        // first `add`.
        let mb = 512.0 * 5.0 / (1024.0 * 1024.0);
        let tmp = || {
            std::fs::read_dir(&path)
                .unwrap()
                .filter(|f| {
                    let n = f.as_ref().unwrap().file_name();
                    n.to_string_lossy().ends_with(".tmp")
                })
                .count()
        };
        for bad in [&[0u8, 0][..], &[1]] {
            let mut w = SimpleTextBkdWriter::new(700, Some(&dir), "_0", c, mb, 600).unwrap();
            for i in 0..600 {
                w.add(&[(i % 251) as u8], i).unwrap();
            }
            assert_eq!(tmp(), 1, "spilled");
            // A wrong length, then one point past `totalPointCount`.
            assert!(w.add(bad, 0).is_err());
            drop(w);
            assert_eq!(tmp(), 0, "cleaned up");
        }
        // A finished build cleans up after itself too.
        let mut w = SimpleTextBkdWriter::new(700, Some(&dir), "_0", c, mb, 600).unwrap();
        for i in 0..600 {
            w.add(&[(i % 251) as u8], i).unwrap();
        }
        w.finish(&mut Vec::new()).unwrap();
        assert_eq!(tmp(), 0);
        drop(w);
        std::fs::remove_dir_all(&path).unwrap();
    }

    #[test]
    fn rejects_what_java_rejects() {
        let c = config(1, 1, 4, 512);
        for mb in [-1.0, f64::NAN] {
            assert!(matches!(
                SimpleTextBkdWriter::new(1, None, "_0", c, mb, 1),
                Err(Error::IllegalArgument(_))
            ));
        }
        // NaN takes Java's path: `(int) NaN` is a heap of no points.
        let e = SimpleTextBkdWriter::new(1, None, "_0", c, f64::NAN, 1)
            .err()
            .unwrap();
        assert!(e.to_string().contains("maxPointsSortInHeap=0"), "{e}");
        let e = SimpleTextBkdWriter::new(1, None, "_0", c, 0.001, 1)
            .err()
            .unwrap();
        assert!(e
            .to_string()
            .contains("only allows for maxPointsSortInHeap=131"));
        let mut w = SimpleTextBkdWriter::new(1, None, "_0", c, 16.0, 1).unwrap();
        let mut out = Vec::new();
        assert_eq!(
            w.finish(&mut out).err().unwrap().to_string(),
            "must index at least one point"
        );
        assert!(w.add(&[0; 3], 0).is_err(), "wrong length");
        w.add(&[0, 0, 0, 1], 0).unwrap();
        assert!(w.add(&[0, 0, 0, 2], 0).is_err(), "past totalPointCount");
        assert_eq!(w.point_count(), 1);
        w.finish(&mut out).unwrap();
        assert_eq!(
            w.finish(&mut out).err().unwrap().to_string(),
            "already finished"
        );
    }

    #[test]
    fn too_many_leaves_and_a_bad_cell() {
        let c = config(2, 2, 16, 1);
        assert!(check_max_leaf_node_count(&c, 1 << 26).is_ok());
        assert!(check_max_leaf_node_count(&c, 1 << 27).is_err());
        assert!(check_max_leaf_node_count(&c, u64::MAX).is_err());
        let w = SimpleTextBkdWriter::new(1, None, "_0", c, 16.0, 1).unwrap();
        assert!(w.split(&[1; 32], &[0; 32]).is_err());
        assert_eq!(w.split(&[0; 32], &[0; 32]).unwrap(), 0);
    }
}
