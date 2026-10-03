//! `LegacyCell` and `LegacyPrefixTree`: the byte-per-level cells
//! `QuadPrefixTree` and `GeohashPrefixTree` share. A cell's term is its
//! bytes, one per level, plus `+` when it is a leaf above the last level.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

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
    /// `getShape().relate(other)`.
    fn relate_cell(&self, cell: &LegacyCell, other: &dyn Shape) -> Result<SpatialRelation> {
        self.make_shape(cell)?.relate(other)
    }
}

/// `LegacyCell` (with `QuadCell`'s and `GhCell`'s tree-specific parts in
/// [`LegacyGrid`]).
#[derive(Clone)]
pub struct LegacyCell {
    pub(crate) grid: Arc<dyn LegacyGrid>,
    /// The token without the leaf byte; its length is the level.
    pub(crate) bytes: CellBytes,
    pub(crate) is_leaf: bool,
    pub(crate) shape_rel: Option<SpatialRelation>,
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
        Self::from_bytes(grid, CellBytes::from_slice(bytes))
    }

    /// [`Self::new`] taking the bytes.
    pub(crate) fn from_bytes(grid: Arc<dyn LegacyGrid>, bytes: CellBytes) -> Self {
        let mut c = LegacyCell {
            grid,
            bytes,
            is_leaf: false,
            shape_rel: None,
        };
        c.read_leaf_adjust();
        c
    }

    /// `readCell(term, scratch)` into this cell: its bytes, leaf flag,
    /// relation and shape reset to `term`'s, the allocation kept.
    pub(crate) fn reset(&mut self, bytes: &[u8]) {
        self.bytes = CellBytes::from_slice(bytes);
        self.is_leaf = false;
        self.shape_rel = None;
        self.read_leaf_adjust();
    }

    /// [`super::SpatialPrefixTree::read_cell_into`] for a legacy tree: the
    /// scratch reused when it is a legacy cell of this grid.
    pub(crate) fn read_into<G: LegacyGrid + 'static>(
        grid: &Arc<G>,
        term: &[u8],
        scratch: &mut Box<dyn Cell>,
    ) {
        let any: &mut dyn Any = &mut **scratch;
        if let Some(c) = any.downcast_mut::<LegacyCell>() {
            if std::ptr::addr_eq(Arc::as_ptr(&c.grid), Arc::as_ptr(grid)) {
                c.reset(term);
                return;
            }
        }
        let grid: Arc<dyn LegacyGrid> = grid.clone();
        *scratch = Box::new(LegacyCell::new(grid, term));
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
        let mut result = self.bytes.to_vec();
        if !self.is_leaf || self.level() == self.grid.max_levels() {
            return result;
        }
        result.push(LEAF_BYTE);
        result
    }

    fn token_bytes_no_leaf(&self) -> Vec<u8> {
        self.bytes.to_vec()
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
        // Java caches the shape in the cell; making it again gives the same
        // rectangle, and a cache's synchronisation costs more than the
        // arithmetic (most cells' shapes are asked for once).
        self.grid.make_shape(self)
    }

    fn relate_shape(&self, other: &dyn Shape) -> Result<SpatialRelation> {
        self.grid.relate_cell(self, other)
    }

    fn is_prefix_of(&self, c: &dyn Cell) -> bool {
        // Java casts to `LegacyCell` and compares its byte slice.
        match c.as_any().downcast_ref::<LegacyCell>() {
            Some(o) => o.bytes.starts_with(self.bytes.as_slice()),
            None => c.token_bytes_no_leaf().starts_with(self.bytes.as_slice()),
        }
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
    let full_bytes = cell.bytes.to_vec();
    let mut cells: Vec<Box<dyn Cell>> = Vec::with_capacity(detail_level.max(0) as usize);
    for i in 1..detail_level {
        let end = (i as usize).min(full_bytes.len());
        cells.push(Box::new(LegacyCell::new(grid.clone(), &full_bytes[..end])));
    }
    cells.push(Box::new(cell));
    Ok(Box::new(FilterCellIterator::new(cells, None)))
}

/// How many token bytes a cell keeps inline (deeper cells spill to the
/// heap): enough for every level of the default quad (12), packed quad
/// and geohash trees, so a cell is copied without allocating.
const INLINE_BYTES: usize = 30;

/// A legacy cell's token bytes (without the leaf byte): inline when short.
#[derive(Clone)]
pub(crate) enum CellBytes {
    Inline(u8, [u8; INLINE_BYTES]),
    Heap(Vec<u8>),
}

impl CellBytes {
    pub(crate) fn from_slice(b: &[u8]) -> Self {
        if b.len() <= INLINE_BYTES {
            let mut buf = [0u8; INLINE_BYTES];
            buf[..b.len()].copy_from_slice(b);
            CellBytes::Inline(b.len() as u8, buf)
        } else {
            CellBytes::Heap(b.to_vec())
        }
    }

    /// The bytes followed by `b` (a child's).
    pub(crate) fn with(&self, b: u8) -> Self {
        match self {
            CellBytes::Inline(len, buf) if usize::from(*len) < INLINE_BYTES => {
                let mut buf = *buf;
                buf[usize::from(*len)] = b;
                CellBytes::Inline(len + 1, buf)
            }
            _ => {
                let mut v = self.to_vec();
                v.push(b);
                CellBytes::Heap(v)
            }
        }
    }

    pub(crate) fn as_slice(&self) -> &[u8] {
        match self {
            CellBytes::Inline(len, buf) => &buf[..usize::from(*len)],
            CellBytes::Heap(v) => v,
        }
    }

    /// Drops the last byte.
    pub(crate) fn pop(&mut self) {
        match self {
            CellBytes::Inline(len, _) => *len = len.saturating_sub(1),
            CellBytes::Heap(v) => {
                v.pop();
            }
        }
    }
}

impl std::ops::Deref for CellBytes {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}
