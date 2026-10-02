//! `LegacyCell` and `LegacyPrefixTree`: the byte-per-level cells
//! `QuadPrefixTree` and `GeohashPrefixTree` share. A cell's term is its
//! bytes, one per level, plus `+` when it is a leaf above the last level.

use std::any::Any;
use std::fmt;
use std::sync::{Arc, OnceLock};

use super::{Cell, CellIterator, FilterCellIterator, SingletonCellIterator};
use crate::spatial4j::{Error, Point, Result, Shape, SpatialContext, SpatialRelation};

/// `LegacyCell.LEAF_BYTE` (sorts before letters and digits).
pub(crate) const LEAF_BYTE: u8 = b'+';

/// What a `LegacyCell` asks its tree (Java's inner-class access to
/// `QuadPrefixTree`/`GeohashPrefixTree`).
pub(crate) trait LegacyGrid: fmt::Debug + Send + Sync {
    /// `getMaxLevels()`.
    fn max_levels(&self) -> i32;
    /// `getSpatialContext()`.
    fn ctx(&self) -> &Arc<SpatialContext>;
    /// `getCell(p, level)`: the cell containing `p` at `level`.
    fn get_cell(self: Arc<Self>, p: &dyn Point, level: i32) -> Result<LegacyCell>;
    /// `getSubCells()`.
    fn sub_cells(self: Arc<Self>, cell: &LegacyCell) -> Result<Vec<LegacyCell>>;
    /// `getSubCellsSize()`.
    fn sub_cells_size(&self) -> i32;
    /// `getShape()` before caching.
    fn make_shape(&self, cell: &LegacyCell) -> Result<Arc<dyn Shape>>;
}

/// `LegacyCell` (with `QuadCell`'s and `GhCell`'s tree-specific parts in
/// [`LegacyGrid`]).
#[derive(Clone)]
pub struct LegacyCell {
    pub(crate) grid: Arc<dyn LegacyGrid>,
    /// The token without the leaf byte; its length is the level.
    pub(crate) bytes: Vec<u8>,
    pub(crate) is_leaf: bool,
    pub(crate) shape_rel: Option<SpatialRelation>,
    pub(crate) shape: OnceLock<Arc<dyn Shape>>,
}

impl fmt::Debug for LegacyCell {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl LegacyCell {
    /// `new LegacyCell(bytes, off, len)` then `readLeafAdjust()`: a trailing
    /// `+` marks a leaf; a cell at the last level is always a leaf.
    pub(crate) fn new(grid: Arc<dyn LegacyGrid>, bytes: &[u8]) -> Self {
        let mut c = LegacyCell {
            grid,
            bytes: bytes.to_vec(),
            is_leaf: false,
            shape_rel: None,
            shape: OnceLock::new(),
        };
        c.read_leaf_adjust();
        c
    }

    /// `readLeafAdjust()`.
    fn read_leaf_adjust(&mut self) {
        self.is_leaf = self.bytes.last() == Some(&LEAF_BYTE);
        if self.is_leaf {
            self.bytes.pop();
        }
        if self.level() == self.grid.max_levels() {
            self.is_leaf = true;
        }
    }

    /// The token bytes, without the leaf byte.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// `getSubCell(p)`: the child containing `p` (built from the point).
    fn sub_cell(&self, p: &dyn Point) -> Result<LegacyCell> {
        self.grid.clone().get_cell(p, self.level() + 1)
    }
}

impl fmt::Display for LegacyCell {
    /// `getTokenBytesWithLeaf(null).utf8ToString()`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&String::from_utf8_lossy(&self.token_bytes_with_leaf()))
    }
}

/// `LegacyCell.compare(..)`: unsigned bytes, then length.
pub(crate) fn compare(a: &[u8], b: &[u8]) -> i32 {
    super::compare_bytes(a, b)
}

impl Cell for LegacyCell {
    fn shape_rel(&self) -> Option<SpatialRelation> {
        self.shape_rel
    }

    fn set_shape_rel(&mut self, rel: Option<SpatialRelation>) {
        self.shape_rel = rel;
    }

    fn is_leaf(&self) -> bool {
        self.is_leaf
    }

    fn set_leaf(&mut self) {
        self.is_leaf = true;
    }

    fn token_bytes_with_leaf(&self) -> Vec<u8> {
        let mut result = self.bytes.clone();
        if !self.is_leaf || self.level() == self.grid.max_levels() {
            return result;
        }
        result.push(LEAF_BYTE);
        result
    }

    fn token_bytes_no_leaf(&self) -> Vec<u8> {
        self.bytes.clone()
    }

    fn level(&self) -> i32 {
        self.bytes.len() as i32
    }

    fn next_level_cells(
        &self,
        shape_filter: Option<&Arc<dyn Shape>>,
    ) -> Result<Box<dyn CellIterator>> {
        if let Some(p) = shape_filter.and_then(|s| s.as_point()) {
            let mut cell = self.sub_cell(p)?;
            cell.shape_rel = Some(SpatialRelation::Contains);
            return Ok(Box::new(SingletonCellIterator::new(Box::new(cell))));
        }
        let cells: Vec<Box<dyn Cell>> = self
            .grid
            .clone()
            .sub_cells(self)?
            .into_iter()
            .map(|c| Box::new(c) as Box<dyn Cell>)
            .collect();
        Ok(Box::new(FilterCellIterator::new(
            cells,
            shape_filter.cloned(),
        )))
    }

    fn shape(&self) -> Result<Arc<dyn Shape>> {
        if let Some(s) = self.shape.get() {
            return Ok(s.clone());
        }
        let s = self.grid.make_shape(self)?;
        Ok(self.shape.get_or_init(|| s).clone())
    }

    fn is_prefix_of(&self, c: &dyn Cell) -> bool {
        // Java casts to `LegacyCell` and compares its byte slice.
        let other = c.token_bytes_no_leaf();
        other.len() >= self.bytes.len() && other[..self.bytes.len()] == self.bytes[..]
    }

    fn compare_to_no_leaf(&self, from_cell: &dyn Cell) -> i32 {
        match from_cell.as_any().downcast_ref::<LegacyCell>() {
            Some(b) => compare(&self.bytes, &b.bytes),
            None => compare(&self.bytes, &from_cell.token_bytes_no_leaf()),
        }
    }

    fn sub_cells_size(&self) -> Option<i32> {
        Some(self.grid.sub_cells_size())
    }

    fn clone_box(&self) -> Box<dyn Cell> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// `LegacyPrefixTree.getDistanceForLevel(level)`: the diagonal of the cell
/// at `level` containing the world's center.
pub(crate) fn distance_for_level(grid: Arc<dyn LegacyGrid>, level: i32) -> Result<f64> {
    if level < 1 || level > grid.max_levels() {
        return Err(Error::IllegalArgument(
            "Level must be in 1 to maxLevels range".into(),
        ));
    }
    let ctx = grid.ctx().clone();
    let center = ctx.world_bounds().center_point();
    let cell = grid.get_cell(&center, level)?;
    let bbox = cell.shape()?.bounding_box()?;
    let width = bbox.width();
    let height = bbox.height();
    Ok((width * width + height * height).sqrt())
}

/// `LegacyPrefixTree.getTreeCellIterator(shape, detailLevel)` for a point:
/// the point's cell at each level, from its full-detail term.
pub(crate) fn point_cell_iterator(
    grid: Arc<dyn LegacyGrid>,
    p: &dyn Point,
    detail_level: i32,
) -> Result<Box<dyn CellIterator>> {
    let cell = grid.clone().get_cell(p, detail_level)?;
    let full_bytes = cell.bytes.clone();
    let mut cells: Vec<Box<dyn Cell>> = Vec::with_capacity(detail_level.max(0) as usize);
    for i in 1..detail_level {
        let end = (i as usize).min(full_bytes.len());
        cells.push(Box::new(LegacyCell::new(grid.clone(), &full_bytes[..end])));
    }
    cells.push(Box::new(cell));
    Ok(Box::new(FilterCellIterator::new(cells, None)))
}
