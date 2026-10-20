//! `SimpleTextBKDReader`: one field's text BKD tree, as `PointValues`, with
//! the `PointTree` cursor Java walks it with and `PointValues`' own
//! `intersect` and `estimatePointCount` over that cursor.
//!
//! The index (leaf pointers, split values, bounds) is held in memory, as in
//! Java; each leaf is read from the `.dim` bytes when visited.
//!
//! What differs from Java: every walk returns a `Result` -- a leaf pointer
//! outside the file, a line without its prefix, a bad number, a value of the
//! wrong length, a block of more docs than `max leaf points` (Java's array
//! index error) or of a negative count is `Corrupted`; `size()`'s
//! `Math.toIntExact` failure too. The doc-id scratch grows with the blocks
//! read rather than being sized from the file's `max leaf points` up front.
//! `PointTree.clone()` is [`SimpleTextPointTree::clone_tree`].

use super::points_writer::{BLOCK_COUNT, BLOCK_DOC_ID, BLOCK_VALUE};
use super::util::{from_bytes_ref_string, parse_integer, read_line, read_value};
use super::{corrupt, file_pointer, Result};
use crate::bkd_writer::BkdConfig;
use crate::points::{IntersectVisitor, Relation};

/// `org.apache.lucene.codecs.simpletext.SimpleTextBKDReader`.
#[derive(Debug, Clone)]
pub struct SimpleTextBkdReader<'a> {
    data: &'a [u8],
    /// Packed `bytesPerIndexEntry` entries, one per node id (entry 0 unused).
    split_packed_values: Vec<u8>,
    leaf_block_fps: Vec<i64>,
    leaf_node_offset: usize,
    config: BkdConfig,
    bytes_per_index_entry: usize,
    min_packed_value: Vec<u8>,
    max_packed_value: Vec<u8>,
    point_count: i64,
    doc_count: i32,
}

impl<'a> SimpleTextBkdReader<'a> {
    /// `new SimpleTextBKDReader(in, numDims, numIndexDims,
    /// maxPointsInLeafNode, bytesPerDim, leafBlockFPs, splitPackedValues,
    /// minPackedValue, maxPackedValue, pointCount, docCount)` over the `.dim`
    /// bytes. `split_packed_values` holds `bytesPerIndexEntry` bytes per node
    /// (`bytesPerDim`, behind the split dim for more than one index dim).
    ///
    /// # Errors
    /// `Corrupted` for bounds that are not `numIndexDims * bytesPerDim`
    /// bytes (Java's `assert`), a split table too short for the inner nodes,
    /// a split dim past the index dims, or more leaves than an `int` counts.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        data: &'a [u8],
        config: BkdConfig,
        leaf_block_fps: Vec<i64>,
        split_packed_values: Vec<u8>,
        min_packed_value: Vec<u8>,
        max_packed_value: Vec<u8>,
        point_count: i64,
        doc_count: i32,
    ) -> Result<Self> {
        let bytes_per_index_entry = bytes_per_index_entry(&config);
        let len = config.packed_index_bytes_length();
        if min_packed_value.len() != len || max_packed_value.len() != len {
            return Err(corrupt(format!(
                "SimpleText points: min/max values of {}/{} bytes, expected {len}",
                min_packed_value.len(),
                max_packed_value.len()
            )));
        }
        let leaf_node_offset = leaf_block_fps.len();
        // Java's writer never writes a tree without leaves; its reader would
        // recurse without end sizing one.
        if leaf_node_offset == 0 || i32::try_from(leaf_node_offset).is_err() {
            return Err(corrupt(format!(
                "SimpleText points: {leaf_node_offset} leaves"
            )));
        }
        // Inner nodes are 1..leafNodeOffset; Java indexes their entries.
        let entries = split_packed_values
            .len()
            .checked_div(bytes_per_index_entry)
            .unwrap_or(0);
        if leaf_node_offset > 1 && entries < leaf_node_offset {
            return Err(corrupt(format!(
                "SimpleText points: {entries} split values for {leaf_node_offset} leaves"
            )));
        }
        if config.num_index_dims > 1 {
            for (node, e) in split_packed_values
                .chunks_exact(bytes_per_index_entry)
                .enumerate()
                .take(leaf_node_offset)
                .skip(1)
            {
                if usize::from(e[0]) >= config.num_index_dims {
                    return Err(corrupt(format!(
                        "SimpleText points: split dim {} of node {node} past {} index dims",
                        e[0], config.num_index_dims
                    )));
                }
            }
        }
        Ok(SimpleTextBkdReader {
            data,
            split_packed_values,
            leaf_block_fps,
            leaf_node_offset,
            config,
            bytes_per_index_entry,
            min_packed_value,
            max_packed_value,
            point_count,
            doc_count,
        })
    }

    /// `getPointTree()`: a cursor on the root.
    pub fn point_tree(&self) -> SimpleTextPointTree<'_, 'a> {
        SimpleTextPointTree::new(
            self,
            1,
            1,
            self.min_packed_value.clone(),
            self.max_packed_value.clone(),
        )
    }

    /// `getMinPackedValue()`.
    pub fn min_packed_value(&self) -> &[u8] {
        &self.min_packed_value
    }

    /// `getMaxPackedValue()`.
    pub fn max_packed_value(&self) -> &[u8] {
        &self.max_packed_value
    }

    /// `getNumDimensions()`.
    pub fn num_dimensions(&self) -> usize {
        self.config.num_dims
    }

    /// `getNumIndexDimensions()`.
    pub fn num_index_dimensions(&self) -> usize {
        self.config.num_index_dims
    }

    /// `getBytesPerDimension()`.
    pub fn bytes_per_dimension(&self) -> usize {
        self.config.bytes_per_dim
    }

    /// The `max leaf points` line: the tree's per-leaf point bound.
    pub fn max_points_in_leaf_node(&self) -> usize {
        self.config.max_points_in_leaf_node
    }

    /// `size()`: the `point count` line.
    pub fn size(&self) -> i64 {
        self.point_count
    }

    /// `getDocCount()`.
    pub fn doc_count(&self) -> i32 {
        self.doc_count
    }

    /// The leaf blocks' file pointers.
    pub fn leaf_block_fps(&self) -> &[i64] {
        &self.leaf_block_fps
    }

    /// `PointValues.intersect(visitor)`: the cursor walk -- `visitDocIDs`
    /// under a cell inside the query, `visitDocValues` at a crossing leaf.
    ///
    /// # Errors
    /// `Corrupted` for a damaged leaf block.
    pub fn intersect<V: IntersectVisitor>(&self, visitor: &mut V) -> Result<()> {
        // Java: PointValues.intersect
        let mut tree = self.point_tree();
        loop {
            match visitor.compare(&tree.min_packed_value, &tree.max_packed_value) {
                Relation::CellInsideQuery => tree.visit_doc_ids(visitor)?,
                Relation::CellCrossesQuery => {
                    if tree.move_to_child()? {
                        continue;
                    }
                    tree.visit_doc_values(visitor)?;
                }
                Relation::CellOutsideQuery => {}
            }
            while !tree.move_to_sibling()? {
                if !tree.move_to_parent() {
                    return Ok(());
                }
            }
        }
    }

    /// `PointValues.estimatePointCount(visitor)`.
    ///
    /// # Errors
    /// `Corrupted` when a node's size cannot be computed.
    pub fn estimate_point_count<V: IntersectVisitor>(&self, visitor: &mut V) -> Result<i64> {
        estimate_point_count(visitor, &mut self.point_tree(), i64::MAX)
    }
}

/// `numIndexDims == 1 ? bytesPerDim : bytesPerDim + 1`.
pub(crate) fn bytes_per_index_entry(config: &BkdConfig) -> usize {
    if config.num_index_dims == 1 {
        config.bytes_per_dim
    } else {
        // ARITH: `bytes_per_dim <= 16` (`BkdConfig::new`).
        #[allow(clippy::arithmetic_side_effects)]
        let e = config.bytes_per_dim + 1;
        e
    }
}

/// `PointValues.estimatePointCount(visitor, pointTree, upperBound)`.
fn estimate_point_count<V: IntersectVisitor>(
    visitor: &mut V,
    tree: &mut SimpleTextPointTree<'_, '_>,
    upper_bound: i64,
) -> Result<i64> {
    match visitor.compare(&tree.min_packed_value, &tree.max_packed_value) {
        Relation::CellOutsideQuery => Ok(0),
        Relation::CellInsideQuery => tree.size(),
        Relation::CellCrossesQuery => {
            if tree.move_to_child()? {
                let mut cost: i64 = 0;
                loop {
                    let c = estimate_point_count(visitor, tree, upper_bound.wrapping_sub(cost))?;
                    cost = cost.wrapping_add(c);
                    if cost >= upper_bound || !tree.move_to_sibling()? {
                        break;
                    }
                }
                tree.move_to_parent();
                Ok(cost)
            } else {
                // Assume half the points matched.
                Ok(tree.size()?.wrapping_add(1) / 2)
            }
        }
    }
}

/// `SimpleTextBKDReader.SimpleTextPointTree`: a cursor over the tree. Node 1
/// is the root, node `n`'s children are `2n` and `2n + 1`, and nodes from
/// `leafNodeOffset` on are leaves.
#[derive(Debug, Clone)]
pub struct SimpleTextPointTree<'r, 'a> {
    reader: &'r SimpleTextBkdReader<'a>,
    node_id: usize,
    level: usize,
    root_node: usize,
    min_packed_value: Vec<u8>,
    max_packed_value: Vec<u8>,
    /// The split dim's previous bound, per level.
    split_dim_value_stack: Vec<Vec<u8>>,
    /// The split dim, per level.
    split_dims: Vec<usize>,
    scratch: Vec<u8>,
    doc_ids: Vec<i32>,
}

impl<'r, 'a> SimpleTextPointTree<'r, 'a> {
    fn new(
        reader: &'r SimpleTextBkdReader<'a>,
        node_id: usize,
        level: usize,
        min_packed_value: Vec<u8>,
        max_packed_value: Vec<u8>,
    ) -> Self {
        // Java: getTreeDepth(leafNodeOffset) + 1 levels; `MathUtil.log(n, 2)`
        // is `n`'s bit length minus one (0 for no leaves).
        let tree_depth = usize::BITS
            .saturating_sub(reader.leaf_node_offset.leading_zeros())
            .saturating_add(2) as usize;
        SimpleTextPointTree {
            reader,
            node_id,
            level,
            root_node: node_id,
            min_packed_value,
            max_packed_value,
            split_dim_value_stack: vec![Vec::new(); tree_depth],
            split_dims: vec![0; tree_depth],
            scratch: Vec::new(),
            doc_ids: Vec::new(),
        }
    }

    /// `PointTree.clone()`: a cursor rooted at the current node, which
    /// cannot move above it.
    pub fn clone_tree(&self) -> Self {
        let mut index = SimpleTextPointTree::new(
            self.reader,
            self.node_id,
            self.level,
            self.min_packed_value.clone(),
            self.max_packed_value.clone(),
        );
        if !self.is_leaf_node() {
            if let (Some(d), Some(v), Some(dd), Some(vv)) = (
                self.split_dims.get(self.level),
                self.split_dim_value_stack.get(self.level),
                index.split_dims.get_mut(self.level),
                index.split_dim_value_stack.get_mut(self.level),
            ) {
                *dd = *d;
                vv.clone_from(v);
            }
        }
        index
    }

    /// `isLeafNode()`.
    pub fn is_leaf_node(&self) -> bool {
        self.node_id >= self.reader.leaf_node_offset
    }

    /// `getMinPackedValue()`: the current cell's lower bound.
    pub fn min_packed_value(&self) -> &[u8] {
        &self.min_packed_value
    }

    /// `getMaxPackedValue()`: the current cell's upper bound.
    pub fn max_packed_value(&self) -> &[u8] {
        &self.max_packed_value
    }

    /// `moveToChild()`: to the left child; `false` on a leaf.
    ///
    /// # Errors
    /// `Corrupted` for a split table or depth the tree does not have.
    pub fn move_to_child(&mut self) -> Result<bool> {
        if self.is_leaf_node() {
            return Ok(false);
        }
        self.push(true)?;
        Ok(true)
    }

    /// `moveToSibling()`: from a left child to its right sibling.
    ///
    /// # Errors
    /// As [`Self::move_to_child`].
    pub fn move_to_sibling(&mut self) -> Result<bool> {
        if self.node_id != self.root_node && self.node_id & 1 == 0 {
            self.pop(true);
            self.push(false)?;
            return Ok(true);
        }
        Ok(false)
    }

    /// `moveToParent()`.
    pub fn move_to_parent(&mut self) -> bool {
        if self.node_id == self.root_node {
            return false;
        }
        self.pop(self.node_id & 1 == 0);
        true
    }

    /// `pushLeft()` (`left`) or `pushRight()`: saves the split dim's bound
    /// at this level and narrows it to the split value.
    fn push(&mut self, left: bool) -> Result<()> {
        let r = self.reader;
        let bpd = r.config.bytes_per_dim;
        let entry = self
            .node_id
            .checked_mul(r.bytes_per_index_entry)
            .and_then(|a| {
                r.split_packed_values
                    .get(a..a.checked_add(r.bytes_per_index_entry)?)
            })
            .ok_or_else(|| {
                corrupt(format!(
                    "SimpleText points: no split for node {}",
                    self.node_id
                ))
            })?;
        let (split_dim, value) = if r.config.num_index_dims == 1 {
            (0, entry)
        } else {
            // `new` checked the dim is below `numIndexDims`.
            (usize::from(entry[0]), &entry[1..])
        };
        let level = self.level;
        let (Some(dim_slot), Some(saved)) = (
            self.split_dims.get_mut(level),
            self.split_dim_value_stack.get_mut(level),
        ) else {
            return Err(corrupt(
                "SimpleText points: tree deeper than its leaf count",
            ));
        };
        *dim_slot = split_dim;
        // ARITH: `split_dim < numIndexDims`, so the range is inside the
        // `numIndexDims * bytesPerDim` bounds; `node_id < leafNodeOffset <=
        // i32::MAX` (`new`), so the child ids fit a `usize`; `level` is below
        // the stacks' length, just checked.
        #[allow(clippy::arithmetic_side_effects)]
        {
            let range = split_dim * bpd..(split_dim + 1) * bpd;
            let bound = if left {
                &mut self.max_packed_value
            } else {
                &mut self.min_packed_value
            };
            saved.clear();
            saved.extend_from_slice(&bound[range.clone()]);
            bound[range].copy_from_slice(value);
            self.node_id = if left {
                self.node_id * 2
            } else {
                self.node_id * 2 + 1
            };
            self.level += 1;
        }
        Ok(())
    }

    /// `pop(isLeft)`: back to the parent, restoring its bound.
    fn pop(&mut self, is_left: bool) {
        self.node_id /= 2;
        self.level = self.level.saturating_sub(1);
        let bpd = self.reader.config.bytes_per_dim;
        let (Some(&dim), Some(saved)) = (
            self.split_dims.get(self.level),
            self.split_dim_value_stack.get(self.level),
        ) else {
            return;
        };
        let bound = if is_left {
            &mut self.max_packed_value
        } else {
            &mut self.min_packed_value
        };
        // ARITH: `dim` was checked below `numIndexDims` when pushed.
        #[allow(clippy::arithmetic_side_effects)]
        if let Some(b) = bound.get_mut(dim * bpd..(dim + 1) * bpd) {
            if b.len() == saved.len() {
                b.copy_from_slice(saved);
            }
        }
    }

    /// `size()`: the points under the current node, as Java's balanced-tree
    /// arithmetic counts them (`sizeFromBalancedTree`).
    ///
    /// # Errors
    /// `Corrupted` where Java's `Math.toIntExact` throws.
    pub fn size(&self) -> Result<i64> {
        let offset = self.reader.leaf_node_offset;
        let mut left_most = self.node_id;
        while left_most < offset {
            left_most = left_most.saturating_mul(2);
        }
        let mut right_most = self.node_id;
        while right_most < offset {
            right_most = right_most.saturating_mul(2).saturating_add(1);
        }
        self.size_from_balanced_tree(left_most, right_most)
    }

    /// `sizeFromBalancedTree(leftMostLeafNode, rightMostLeafNode)`.
    fn size_from_balanced_tree(&self, left_most: usize, right_most: usize) -> Result<i64> {
        let r = self.reader;
        let max_leaf = i64::try_from(r.config.max_points_in_leaf_node).unwrap_or(i64::MAX);
        // `leaf_node_offset <= i32::MAX` (`new`).
        let offset_int = i32::try_from(r.leaf_node_offset).unwrap_or(i32::MAX);
        let offset = i64::from(offset_int);
        let extra_points = max_leaf
            .checked_mul(offset)
            .and_then(|n| n.checked_sub(r.point_count))
            .and_then(|e| i32::try_from(e).ok())
            .ok_or_else(|| corrupt("SimpleText points: integer overflow in size"))?;
        // Java's `int` subtraction, which wraps.
        let node_offset = offset_int.wrapping_sub(extra_points);
        let mut count: i64 = 0;
        let mut node = left_most;
        while node <= right_most {
            let leaf = i64::try_from(node)
                .unwrap_or(i64::MAX)
                .saturating_sub(offset);
            let points =
                if balance_tree_node_position(0, offset, leaf, 0, 0) < i64::from(node_offset) {
                    max_leaf
                } else {
                    max_leaf.saturating_sub(1)
                };
            count = count.saturating_add(points);
            node = node.saturating_add(1);
            if node == usize::MAX {
                break;
            }
        }
        Ok(count)
    }

    /// `visitDocIDs(visitor)`: every doc id under the current node, no
    /// values read.
    ///
    /// # Errors
    /// `Corrupted` for a damaged leaf block.
    pub fn visit_doc_ids<V: IntersectVisitor>(&mut self, visitor: &mut V) -> Result<()> {
        self.add_all(visitor, false)
    }

    /// `addAll(visitor, grown)`.
    fn add_all<V: IntersectVisitor>(&mut self, visitor: &mut V, mut grown: bool) -> Result<()> {
        if !grown {
            let size = self.size()?;
            if size <= i64::from(i32::MAX) {
                if let Ok(size) = usize::try_from(size) {
                    visitor.grow(size);
                }
                grown = true;
            }
        }
        if self.is_leaf_node() {
            let mut pos = self.seek_leaf()?;
            let count = self.read_count(&mut pos)?;
            for _ in 0..count {
                let doc = self.read_int(&mut pos, BLOCK_DOC_ID)?;
                visitor.visit(doc);
            }
        } else {
            self.push(true)?;
            self.add_all(visitor, grown)?;
            self.pop(true);
            self.push(false)?;
            self.add_all(visitor, grown)?;
            self.pop(false);
        }
        Ok(())
    }

    /// `visitDocValues(visitor)`: every point under the current node.
    ///
    /// # Errors
    /// `Corrupted` for a damaged leaf block.
    pub fn visit_doc_values<V: IntersectVisitor>(&mut self, visitor: &mut V) -> Result<()> {
        if self.is_leaf_node() {
            let mut pos = self.seek_leaf()?;
            // Java: readDocIDs
            let count = self.read_count(&mut pos)?;
            let mut doc_ids = std::mem::take(&mut self.doc_ids);
            doc_ids.clear();
            for _ in 0..count {
                doc_ids.push(self.read_int(&mut pos, BLOCK_DOC_ID)?);
            }
            visitor.grow(count);
            let p = self.reader.config.packed_bytes_length();
            for &doc in &doc_ids {
                let tail = read_value(self.reader.data, &mut pos, &mut self.scratch, BLOCK_VALUE)?;
                let value = from_bytes_ref_string(&String::from_utf8_lossy(tail))
                    .map_err(|e| corrupt(e.to_string()))?;
                if value.len() != p {
                    return Err(corrupt(format!(
                        "SimpleText points: a value of {} bytes, expected {p}",
                        value.len()
                    )));
                }
                visitor.visit_with_value(doc, &value);
            }
            self.doc_ids = doc_ids;
        } else {
            self.push(true)?;
            self.visit_doc_values(visitor)?;
            self.pop(true);
            self.push(false)?;
            self.visit_doc_values(visitor)?;
            self.pop(false);
        }
        Ok(())
    }

    /// `in.seek(leafBlockFPs[nodeID - leafNodeOffset])`.
    fn seek_leaf(&self) -> Result<usize> {
        let r = self.reader;
        let fp = self
            .node_id
            .checked_sub(r.leaf_node_offset)
            .and_then(|leaf| r.leaf_block_fps.get(leaf))
            .ok_or_else(|| corrupt(format!("SimpleText points: no leaf {}", self.node_id)))?;
        file_pointer(r.data, *fp, "SimpleText points: block fp")
    }

    /// The `block count` line: non-negative, and at most `max leaf points`
    /// (Java's `scratchDocIDs` is that long).
    fn read_count(&mut self, pos: &mut usize) -> Result<usize> {
        let count = self.read_int(pos, BLOCK_COUNT)?;
        usize::try_from(count)
            .ok()
            .filter(|&c| c <= self.reader.config.max_points_in_leaf_node)
            .ok_or_else(|| {
                corrupt(format!(
                    "SimpleText points: block count {count} outside 0..={}",
                    self.reader.config.max_points_in_leaf_node
                ))
            })
    }

    fn read_int(&mut self, pos: &mut usize, prefix: &[u8]) -> Result<i32> {
        read_line(self.reader.data, pos, &mut self.scratch)?;
        let tail = self.scratch.strip_prefix(prefix).ok_or_else(|| {
            corrupt(format!(
                "SimpleText failure: expected line starting with {:?} but got {:?}",
                String::from_utf8_lossy(prefix),
                String::from_utf8_lossy(&self.scratch)
            ))
        })?;
        parse_integer(tail)
    }
}

/// `balanceTreeNodePosition(minNode, maxNode, node, position, level)`, as a
/// loop (Java recurses once per level; the depth is `log2` of the leaves).
fn balance_tree_node_position(
    mut min_node: i64,
    mut max_node: i64,
    node: i64,
    mut position: i64,
    mut level: u32,
) -> i64 {
    while max_node.saturating_sub(min_node) > 1 {
        let mid = min_node.saturating_add(max_node).saturating_add(1) / 2;
        if mid > node {
            max_node = mid;
        } else {
            min_node = mid;
            position = position.saturating_add(1i64.checked_shl(level).unwrap_or(0));
        }
        level = level.saturating_add(1);
    }
    position
}

#[cfg(test)]
#[allow(clippy::arithmetic_side_effects)]
mod tests {
    use super::*;

    const LEAF: &[u8] = b"block count 1\n  doc 7\n  block value [5]\n";

    fn reader(data: &[u8], fps: Vec<i64>, point_count: i64) -> Result<SimpleTextBkdReader<'_>> {
        let config = BkdConfig::new(1, 1, 1, 1).unwrap();
        let leaves = fps.len();
        SimpleTextBkdReader::new(
            data,
            config,
            fps,
            vec![3; leaves],
            vec![0],
            vec![9],
            point_count,
            1,
        )
    }

    struct Count(usize, Vec<usize>);

    impl IntersectVisitor for Count {
        fn compare(&mut self, _: &[u8], _: &[u8]) -> Relation {
            Relation::CellCrossesQuery
        }
        fn visit(&mut self, _: i32) {
            self.0 += 1;
        }
        fn visit_with_value(&mut self, _: i32, _: &[u8]) {
            self.0 += 1;
        }
        fn grow(&mut self, n: usize) {
            self.1.push(n);
        }
    }

    #[test]
    fn walks_a_two_leaf_tree() {
        let r = reader(LEAF, vec![0, 0], 2).unwrap();
        assert_eq!(r.max_points_in_leaf_node(), 1);
        let mut c = Count(0, Vec::new());
        r.intersect(&mut c).unwrap();
        assert_eq!(c.0, 2);
        assert_eq!(r.estimate_point_count(&mut c).unwrap(), 2);
        let mut t = r.point_tree();
        assert!(t.move_to_child().unwrap());
        assert_eq!(t.max_packed_value(), [3]);
        assert!(t.move_to_sibling().unwrap());
        assert_eq!(t.min_packed_value(), [3]);
        assert!(!t.move_to_sibling().unwrap());
        assert!(t.move_to_parent());
        assert_eq!(
            (t.min_packed_value(), t.max_packed_value()),
            (&[0][..], &[9][..])
        );
        let mut ids = Count(0, Vec::new());
        t.visit_doc_ids(&mut ids).unwrap();
        assert_eq!((ids.0, ids.1), (2, vec![2]));
        // A clone of an inner node keeps its split.
        let mut c = t.clone_tree();
        assert!(c.move_to_child().unwrap());
        assert!(c.move_to_parent());
        assert!(!c.move_to_parent());
    }

    #[test]
    fn refuses_a_bad_index() {
        // No leaves, too many split dims, a short split table, bounds of the
        // wrong length.
        assert!(reader(LEAF, Vec::new(), 0).is_err());
        let config = BkdConfig::new(2, 2, 1, 1).unwrap();
        let two = |split: Vec<u8>, min: Vec<u8>| {
            SimpleTextBkdReader::new(LEAF, config, vec![0, 0], split, min, vec![9, 9], 2, 1)
        };
        assert!(two(vec![0, 0, 2, 3], vec![0, 0]).is_err());
        assert!(two(vec![0, 0], vec![0, 0]).is_err());
        assert!(two(vec![0, 0, 1, 3], vec![0]).is_err());
        let r = two(vec![0, 0, 1, 3], vec![0, 0]).unwrap();
        let mut t = r.point_tree();
        assert!(t.move_to_child().unwrap());
        assert_eq!(t.max_packed_value(), [9, 3]);
    }

    #[test]
    fn refuses_bad_leaves() {
        let mut c = Count(0, Vec::new());
        // A leaf pointer past the end, or negative.
        for fp in [1000, -1] {
            let r = reader(LEAF, vec![0, fp], 2).unwrap();
            assert!(r.intersect(&mut c).is_err());
            assert!(r.point_tree().visit_doc_ids(&mut c).is_err());
        }
        // More docs than a leaf holds, a negative count, a bad doc, a
        // missing value, a value of the wrong length.
        for leaf in [
            &b"block count 2\n  doc 7\n  doc 8\n"[..],
            b"block count -1\n",
            b"block count 1\n  doc x\n",
            b"block count 1\n  dog 7\n",
            b"block count 1\n  doc 7\n",
            b"block count 1\n  doc 7\n  block value [5 6]\n",
            b"block count 1\n  doc 7\n  block value [zz]\n",
        ] {
            let r = reader(leaf, vec![0], 1).unwrap();
            assert!(r.point_tree().visit_doc_values(&mut c).is_err());
        }
        // `size()` past what Java's `Math.toIntExact` takes.
        let r = reader(LEAF, vec![0], -(1 << 40)).unwrap();
        assert!(r.point_tree().size().is_err());
        assert!(r.point_tree().visit_doc_ids(&mut c).is_err());
    }

    #[test]
    fn balanced_positions() {
        // Java: leaves 0..4 of a 4-leaf tree sit at positions 0, 2, 1, 3.
        let got: Vec<i64> = (0..4)
            .map(|n| balance_tree_node_position(0, 4, n, 0, 0))
            .collect();
        assert_eq!(got, [0, 2, 1, 3]);
    }
}
