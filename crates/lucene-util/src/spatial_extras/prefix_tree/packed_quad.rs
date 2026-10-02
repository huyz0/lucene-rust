//! `PackedQuadPrefixTree`: `QuadPrefixTree`'s quadrants packed into one
//! 8-byte big-endian term -- two bits per level from the top, the level in
//! bits 1-5 and the leaf flag in bit 0 -- with its own depth-first cell
//! iterator that can prune a branch whose four children all intersect.

use std::any::Any;
use std::fmt;
use std::sync::{Arc, OnceLock};

use super::quad::QuadGeometry;
use super::{Cell, CellIterator, FilterCellIterator, SingletonCellIterator, SpatialPrefixTree};
use crate::spatial4j::{
    Error, Point, RectangleImpl, Result, Shape, SpatialContext, SpatialRelation,
};

/// `PackedQuadPrefixTree.MAX_LEVELS_POSSIBLE`.
pub const MAX_LEVELS_POSSIBLE: i32 = 29;

/// `PackedQuadPrefixTree`.
#[derive(Debug, Clone)]
pub struct PackedQuadPrefixTree {
    inner: Arc<PackedGrid>,
}

#[derive(Debug)]
struct PackedGrid {
    geom: QuadGeometry,
    leafy_prune: std::sync::atomic::AtomicBool,
}

impl PackedQuadPrefixTree {
    /// `new PackedQuadPrefixTree(ctx, maxLevels)`.
    pub fn new(ctx: Arc<SpatialContext>, max_levels: i32) -> Result<Self> {
        if max_levels > MAX_LEVELS_POSSIBLE {
            return Err(Error::IllegalArgument(format!(
                "maxLevels of {max_levels} exceeds limit of {MAX_LEVELS_POSSIBLE}"
            )));
        }
        let bounds = ctx.world_bounds_values();
        Ok(PackedQuadPrefixTree {
            inner: Arc::new(PackedGrid {
                geom: QuadGeometry::new(ctx, bounds, max_levels),
                leafy_prune: std::sync::atomic::AtomicBool::new(true),
            }),
        })
    }

    /// `isPruneLeafyBranches()`.
    pub fn is_prune_leafy_branches(&self) -> bool {
        self.inner
            .leafy_prune
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// `setPruneLeafyBranches(prune)`.
    pub fn set_prune_leafy_branches(&self, prune: bool) {
        self.inner
            .leafy_prune
            .store(prune, std::sync::atomic::Ordering::Relaxed);
    }

    /// `getCell(p, level)`.
    pub fn get_cell(&self, p: &dyn Point, level: i32) -> Box<dyn Cell> {
        Box::new(self.inner.clone().get_cell(p, level))
    }
}

impl PackedGrid {
    /// `getCell(p, level)`.
    fn get_cell(self: Arc<Self>, p: &dyn Point, level: i32) -> PackedQuadCell {
        let mut term: u64 = 0;
        for (lvl, quad) in self.geom.quads_for_point(p, level).into_iter().enumerate() {
            // set bits for next level
            term |= (quad as u64) << (64 - ((lvl as u32 + 1) << 1));
            // increment level
            term = ((term >> 1).wrapping_add(1)) << 1;
        }
        let mut cell = PackedQuadCell::new(self, term);
        cell.shape_rel = Some(SpatialRelation::Contains);
        cell
    }
}

/// `PackedQuadPrefixTree.PackedQuadCell`.
#[derive(Clone)]
pub struct PackedQuadCell {
    grid: Arc<PackedGrid>,
    term: u64,
    is_leaf: bool,
    shape_rel: Option<SpatialRelation>,
    shape: OnceLock<Arc<dyn Shape>>,
}

impl fmt::Debug for PackedQuadCell {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl PackedQuadCell {
    /// `new PackedQuadCell(term)` (then `readLeafAdjust()`).
    fn new(grid: Arc<PackedGrid>, term: u64) -> Self {
        let mut c = PackedQuadCell {
            grid,
            term,
            is_leaf: false,
            shape_rel: None,
            shape: OnceLock::new(),
        };
        c.read_leaf_adjust();
        c
    }

    /// `readLeafAdjust()`.
    fn read_leaf_adjust(&mut self) {
        self.is_leaf = (self.term & 1) == 1;
        if self.level() == self.grid.geom.max_levels {
            self.is_leaf = true;
        }
    }

    /// `getShiftForLevel(level)`: Java's `>>>` masks the count to 6 bits.
    fn shift_for_level(level: i32) -> u32 {
        (64 - (level << 1)) as u32 & 63
    }

    /// `isEnd(level, shift)`.
    fn is_end(&self, level: i32, shift: u32) -> bool {
        self.term != 0
            && ((1u64.wrapping_shl((level << 1) as u32)).wrapping_sub(1))
                .wrapping_sub(self.term.wrapping_shr(shift))
                == 0
    }

    /// `nextCell(descend)`: the next cell in depth-first order, descending
    /// first when asked (and possible).
    fn next_cell(&self, descend: bool) -> Option<PackedQuadCell> {
        let max_levels = self.grid.geom.max_levels;
        let level = self.level();
        let shift = Self::shift_for_level(level);
        if (!descend && self.is_end(level, shift))
            || self.is_end(max_levels, Self::shift_for_level(max_levels))
        {
            return None;
        }
        let is_leaf = (self.term & 1) == 1;
        let new_term = if (descend && !is_leaf && level != max_levels) || level == 0 {
            ((self.term >> 1).wrapping_add(1)) << 1
        } else {
            let mut new_term = self.term.wrapping_add(1u64.wrapping_shl(shift));
            if (self.term.wrapping_shr(shift) & 3) == 3 {
                let tz = new_term.wrapping_shr(shift).trailing_zeros() as u64;
                new_term = ((new_term >> 1).wrapping_sub(tz >> 1)) << 1;
            }
            new_term
        };
        Some(PackedQuadCell::new(self.grid.clone(), new_term))
    }

    /// `getSubCells()`.
    fn sub_cells(&self) -> Vec<PackedQuadCell> {
        let base = if (self.term & 1) == 1 {
            self.term - 1
        } else {
            self.term
        };
        let mut cells = Vec::with_capacity(4);
        let mut pqc = PackedQuadCell::new(self.grid.clone(), base).next_cell(true);
        for _ in 0..4 {
            match pqc {
                Some(c) => {
                    pqc = c.next_cell(false);
                    cells.push(c);
                }
                None => break,
            }
        }
        cells
    }

    /// `makeShape()`: a `RectangleImpl` built directly (not through the
    /// context's factory).
    fn make_shape(&self) -> Arc<dyn Shape> {
        let level = self.level();
        let quads = (1..=level).map(|i| ((self.term >> (64 - (i << 1))) & 3) as u8);
        let [a, b, c, d] = self.grid.geom.cell_rect(quads, level as usize);
        Arc::new(RectangleImpl::new(a, b, c, d, self.grid.geom.ctx.clone()))
    }

    /// The packed term (leaf bit included).
    pub fn term(&self) -> u64 {
        self.term
    }
}

impl fmt::Display for PackedQuadCell {
    /// `toString()`: the term in binary, 64 digits.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.term == 0 {
            write!(f, "{}", "0".repeat(64))
        } else {
            write!(f, "{:064b}", self.term)
        }
    }
}

impl Cell for PackedQuadCell {
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
        let mut b = self.token_bytes_no_leaf();
        if self.is_leaf {
            b[7] |= 1;
        }
        b
    }

    fn token_bytes_no_leaf(&self) -> Vec<u8> {
        let mut b = self.term.to_be_bytes().to_vec();
        b[7] &= !1;
        b
    }

    fn level(&self) -> i32 {
        ((self.term >> 1) & 0x1F) as i32
    }

    fn next_level_cells(
        &self,
        shape_filter: Option<&Arc<dyn Shape>>,
    ) -> Result<Box<dyn CellIterator>> {
        if let Some(p) = shape_filter.and_then(|s| s.as_point()) {
            let mut cell = self.grid.clone().get_cell(p, self.level() + 1);
            cell.shape_rel = Some(SpatialRelation::Contains);
            return Ok(Box::new(SingletonCellIterator::new(Box::new(cell))));
        }
        let cells: Vec<Box<dyn Cell>> = self
            .sub_cells()
            .into_iter()
            .map(|c| Box::new(c) as Box<dyn Cell>)
            .collect();
        Ok(Box::new(FilterCellIterator::new(
            cells,
            shape_filter.cloned(),
        )))
    }

    fn shape(&self) -> Result<Arc<dyn Shape>> {
        Ok(self.shape.get_or_init(|| self.make_shape()).clone())
    }

    fn is_prefix_of(&self, c: &dyn Cell) -> bool {
        let Some(cell) = c.as_any().downcast_ref::<PackedQuadCell>() else {
            return false;
        };
        let shift = (64 - (self.level() << 1)) as u32;
        self.term == 0
            || self
                .term
                .wrapping_shr(shift)
                .wrapping_sub(cell.term.wrapping_shr(shift))
                == 0
    }

    fn compare_to_no_leaf(&self, from_cell: &dyn Cell) -> i32 {
        let from_term = match from_cell.as_any().downcast_ref::<PackedQuadCell>() {
            Some(b) => b.term,
            None => u64::from_be_bytes(
                from_cell
                    .token_bytes_no_leaf()
                    .try_into()
                    .unwrap_or([0u8; 8]),
            ),
        };
        let this_term = if (self.term & 1) == 1 {
            self.term - 1
        } else {
            self.term
        };
        let from_term = if (from_term & 1) == 1 {
            from_term - 1
        } else {
            from_term
        };
        this_term.cmp(&from_term) as i32
    }

    fn sub_cells_size(&self) -> Option<i32> {
        Some(4)
    }

    fn clone_box(&self) -> Box<dyn Cell> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// `PackedQuadPrefixTree.PrefixTreeIterator`.
struct PrefixTreeIterator {
    shape: Arc<dyn Shape>,
    this_cell: Option<PackedQuadCell>,
    next_cell: Option<PackedQuadCell>,
    level: i32,
    detail_level: i32,
    leafy_prune: bool,
    last: Option<Box<dyn Cell>>,
}

impl PrefixTreeIterator {
    /// `pruned(rel)`: an intersecting cell one level above the detail
    /// level, all four of whose children intersect, is indexed as a leaf.
    fn pruned(&self, rel: SpatialRelation) -> Result<bool> {
        if rel == SpatialRelation::Intersects
            && self.leafy_prune
            && self.level == self.detail_level - 1
        {
            let this = self.this_cell.as_ref().expect("pruned is asked of a cell");
            let mut prune_iter = this.next_level_cells(Some(&self.shape))?;
            let mut leaves = 0;
            while prune_iter.has_next()? {
                prune_iter.next()?;
                leaves += 1;
            }
            return Ok(leaves == 4);
        }
        Ok(false)
    }
}

impl CellIterator for PrefixTreeIterator {
    fn has_next(&mut self) -> Result<bool> {
        if self.next_cell.is_some() {
            return Ok(true);
        }
        while let Some(this) = self.this_cell.as_mut() {
            let rel = this.shape()?.relate(&*self.shape)?;
            if rel == SpatialRelation::Disjoint {
                self.this_cell = this.next_cell(false);
            } else {
                this.set_shape_rel(Some(rel));
                if rel == SpatialRelation::Within {
                    this.set_leaf();
                    self.next_cell = Some(this.clone());
                    self.this_cell = this.next_cell(false);
                } else {
                    self.level = this.level();
                    if self.level == self.detail_level || self.pruned(rel)? {
                        let this = self.this_cell.as_mut().expect("still set");
                        this.set_leaf();
                        if self.shape.as_point().is_some() {
                            this.set_shape_rel(Some(SpatialRelation::Within));
                            self.next_cell = Some(this.clone());
                            self.this_cell = None;
                        } else {
                            self.next_cell = Some(this.clone());
                            self.this_cell = this.next_cell(false);
                        }
                        break;
                    }
                    let this = self.this_cell.as_ref().expect("still set");
                    self.next_cell = Some(this.clone());
                    self.this_cell = this.next_cell(true);
                }
                break;
            }
        }
        Ok(self.next_cell.is_some())
    }

    fn next(&mut self) -> Result<Box<dyn Cell>> {
        if self.next_cell.is_none() && !self.has_next()? {
            return Err(Error::Runtime("java.util.NoSuchElementException".into()));
        }
        let c: Box<dyn Cell> = Box::new(self.next_cell.take().expect("filled above"));
        self.last = Some(c.clone_box());
        Ok(c)
    }

    fn this_cell(&self) -> Option<&dyn Cell> {
        self.last.as_deref()
    }
}

impl fmt::Display for PackedQuadPrefixTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PackedQuadPrefixTree(maxLevels:{},ctx:{},prune:{})",
            self.inner.geom.max_levels,
            self.inner.geom.ctx,
            self.is_prune_leafy_branches()
        )
    }
}

impl SpatialPrefixTree for PackedQuadPrefixTree {
    fn spatial_context(&self) -> &Arc<SpatialContext> {
        &self.inner.geom.ctx
    }

    fn max_levels(&self) -> i32 {
        self.inner.geom.max_levels
    }

    fn level_for_distance(&self, dist: f64) -> i32 {
        self.inner.geom.level_for_distance(dist)
    }

    /// `LegacyPrefixTree.getDistanceForLevel(level)`.
    fn distance_for_level(&self, level: i32) -> Result<f64> {
        if level < 1 || level > self.max_levels() {
            return Err(Error::IllegalArgument(
                "Level must be in 1 to maxLevels range".into(),
            ));
        }
        let ctx = self.spatial_context().clone();
        let center = ctx.world_bounds().center_point();
        let cell = self.inner.clone().get_cell(&center, level);
        let bbox = cell.shape()?.bounding_box()?;
        let width = bbox.width();
        let height = bbox.height();
        Ok((width * width + height * height).sqrt())
    }

    fn world_cell(&self) -> Box<dyn Cell> {
        Box::new(PackedQuadCell::new(self.inner.clone(), 0))
    }

    /// `readCell(term, null)`: the first eight bytes, big-endian.
    fn read_cell(&self, term: &[u8]) -> Result<Box<dyn Cell>> {
        if term.len() < 8 {
            return Err(Error::ArrayIndexOutOfBounds(format!(
                "Index {} out of bounds for length {}",
                term.len(),
                term.len()
            )));
        }
        let mut b = [0u8; 8];
        b.copy_from_slice(&term[..8]);
        Ok(Box::new(PackedQuadCell::new(
            self.inner.clone(),
            u64::from_be_bytes(b),
        )))
    }

    fn tree_cell_iterator(
        &self,
        shape: &Arc<dyn Shape>,
        detail_level: i32,
    ) -> Result<Box<dyn CellIterator>> {
        if detail_level > self.max_levels() {
            return Err(Error::IllegalArgument(format!(
                "detailLevel:{detail_level} exceed max: {}",
                self.max_levels()
            )));
        }
        let world = PackedQuadCell::new(self.inner.clone(), 0);
        Ok(Box::new(PrefixTreeIterator {
            shape: shape.clone(),
            this_cell: world.next_cell(true),
            next_cell: None,
            level: 0,
            detail_level,
            leafy_prune: self.is_prune_leafy_branches(),
            last: None,
        }))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
