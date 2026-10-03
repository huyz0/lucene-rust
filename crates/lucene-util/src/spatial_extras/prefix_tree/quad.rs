//! `QuadPrefixTree`: each level splits a cell into four quadrants, `A`
//! (upper left), `B` (upper right), `C` (lower left), `D` (lower right) --
//! one byte of the term per level.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use super::legacy::{self, LegacyCell, LegacyGrid};
use super::{Cell, CellIterator, SpatialPrefixTree};
use crate::spatial4j::{
    Error, Point, RectangleImpl, Result, Shape, ShapeFactoryImpl, SpatialContext, SpatialRelation,
};

/// `QuadPrefixTree.MAX_LEVELS_POSSIBLE`.
pub const MAX_LEVELS_POSSIBLE: i32 = 50;
/// `QuadPrefixTree.DEFAULT_MAX_LEVELS`.
pub const DEFAULT_MAX_LEVELS: i32 = 12;

/// The grid geometry `QuadPrefixTree` and `PackedQuadPrefixTree` share.
#[derive(Debug, Clone)]
pub(crate) struct QuadGeometry {
    pub(crate) ctx: Arc<SpatialContext>,
    pub(crate) max_levels: i32,
    pub(crate) xmin: f64,
    pub(crate) ymin: f64,
    pub(crate) xmid: f64,
    pub(crate) ymid: f64,
    pub(crate) grid_w: f64,
    pub(crate) grid_h: f64,
    pub(crate) level_w: Vec<f64>,
    pub(crate) level_h: Vec<f64>,
}

impl QuadGeometry {
    /// `QuadPrefixTree(ctx, bounds, maxLevels)`'s fields.
    pub(crate) fn new(ctx: Arc<SpatialContext>, bounds: [f64; 4], max_levels: i32) -> Self {
        let [xmin, xmax, ymin, ymax] = bounds;
        let n = (max_levels.max(0) + 1) as usize;
        let mut level_w = vec![0.0; n];
        let mut level_h = vec![0.0; n];
        let grid_w = xmax - xmin;
        let grid_h = ymax - ymin;
        let xmid = xmin + grid_w / 2.0;
        let ymid = ymin + grid_h / 2.0;
        level_w[0] = grid_w / 2.0;
        level_h[0] = grid_h / 2.0;
        for i in 1..n {
            level_w[i] = level_w[i - 1] / 2.0;
            level_h[i] = level_h[i - 1] / 2.0;
        }
        QuadGeometry {
            ctx,
            max_levels,
            xmin,
            ymin,
            xmid,
            ymid,
            grid_w,
            grid_h,
            level_w,
            level_h,
        }
    }

    /// `getLevelForDistance(dist)`.
    pub(crate) fn level_for_distance(&self, dist: f64) -> i32 {
        if dist == 0.0 {
            return self.max_levels;
        }
        for i in 0..(self.max_levels - 1).max(0) as usize {
            // note: level[i] is actually a lookup for level i+1
            if dist > self.level_w[i] && dist > self.level_h[i] {
                return i as i32 + 1;
            }
        }
        self.max_levels
    }

    /// `battenberg(xmid, ymid, xp, yp)`: the quadrant, 0-3.
    pub(crate) fn battenberg(xmid: f64, ymid: f64, xp: f64, yp: f64) -> u8 {
        if ymid <= yp {
            if xmid >= xp {
                return 0;
            }
            1
        } else {
            if xmid >= xp {
                return 2;
            }
            3
        }
    }

    /// `getCell(p, level)`'s walk: the quadrant at each level.
    pub(crate) fn quads_for_point(&self, p: &dyn Point, level: i32) -> Vec<u8> {
        let mut current_xmid = self.xmid;
        let mut current_ymid = self.ymid;
        let xp = p.x();
        let yp = p.y();
        let level_limit = if level > self.max_levels {
            self.max_levels
        } else {
            level
        };
        let mut quads = Vec::with_capacity(level_limit.max(0) as usize);
        for lvl in 0..level_limit.max(0) as usize {
            let c = Self::battenberg(current_xmid, current_ymid, xp, yp);
            let half_width = self.level_w[lvl + 1];
            let half_height = self.level_h[lvl + 1];
            match c {
                0 => {
                    current_xmid -= half_width;
                    current_ymid += half_height;
                }
                1 => {
                    current_xmid += half_width;
                    current_ymid += half_height;
                }
                2 => {
                    current_xmid -= half_width;
                    current_ymid -= half_height;
                }
                _ => {
                    current_xmid += half_width;
                    current_ymid -= half_height;
                }
            }
            quads.push(c);
        }
        quads
    }

    /// `QuadCell.makeShape()`'s corner and size for the quadrant path
    /// (`len` quadrants). A path deeper than `levelW` -- a term of a deeper
    /// tree, or a corrupt one -- is Java's `ArrayIndexOutOfBoundsException`
    /// at the first index it reads past the end.
    pub(crate) fn cell_rect(
        &self,
        quads: impl Iterator<Item = u8>,
        len: usize,
    ) -> Result<[f64; 4]> {
        let n = self.level_w.len();
        if len > n {
            // the first quadrant past the end that reads `levelW` (`C`
            // reads nothing), else `levelW[len - 1]`
            let i = quads
                .enumerate()
                .skip(n)
                .find(|&(_, c)| c != 2)
                .map_or(len - 1, |(i, _)| i);
            return Err(array_index_out_of_bounds(i, n));
        }
        let mut xmin = self.xmin;
        let mut ymin = self.ymin;
        // Java's switch -- 0 (`A`): y; 1 (`B`): x and y; 2 (`C`): neither;
        // 3 (`D`): x -- without its branches, which the cells' random
        // quadrants mispredict: a skipped addition adds `-0.0`, the one
        // value that leaves every double (`-0.0` included) unchanged.
        for ((&w, &h), c) in self.level_w.iter().zip(&self.level_h).zip(quads) {
            xmin += if c == 1 || c >= 3 { w } else { -0.0 };
            ymin += if c <= 1 { h } else { -0.0 };
        }
        let (width, height) = match len.checked_sub(1) {
            Some(last) => (self.level_w[last], self.level_h[last]),
            None => (self.grid_w, self.grid_h),
        };
        Ok([xmin, xmin + width, ymin, ymin + height])
    }
}

/// Java's `ArrayIndexOutOfBoundsException` message for `array[index]`.
pub(crate) fn array_index_out_of_bounds(index: usize, length: usize) -> Error {
    Error::ArrayIndexOutOfBounds(format!("Index {index} out of bounds for length {length}"))
}

/// `QuadPrefixTree`.
#[derive(Debug, Clone)]
pub struct QuadPrefixTree {
    inner: Arc<QuadGrid>,
}

#[derive(Debug)]
struct QuadGrid {
    geom: QuadGeometry,
}

impl QuadPrefixTree {
    /// `new QuadPrefixTree(ctx, maxLevels)`: over the world bounds.
    pub fn new(ctx: Arc<SpatialContext>, max_levels: i32) -> Result<Self> {
        let bounds = ctx.world_bounds_values();
        Self::with_bounds(ctx, bounds, max_levels)
    }

    /// `new QuadPrefixTree(ctx, bounds, maxLevels)`; `bounds` is `[minX,
    /// maxX, minY, maxY]`.
    pub fn with_bounds(
        ctx: Arc<SpatialContext>,
        bounds: [f64; 4],
        max_levels: i32,
    ) -> Result<Self> {
        Ok(QuadPrefixTree {
            inner: Arc::new(QuadGrid {
                geom: QuadGeometry::new(ctx, bounds, max_levels),
            }),
        })
    }

    /// `getCell(p, level)`.
    pub fn get_cell(&self, p: &dyn Point, level: i32) -> Result<Box<dyn Cell>> {
        Ok(Box::new(self.inner.clone().get_cell(p, level)?))
    }

    fn grid(&self) -> Arc<dyn LegacyGrid> {
        self.inner.clone()
    }
}

impl QuadGrid {
    /// `QuadCell.makeShape()`'s arithmetic: the cell's
    /// `[minX, maxX, minY, maxY]`.
    fn cell_bounds(&self, cell: &LegacyCell) -> Result<[f64; 4]> {
        self.bytes_bounds(cell.bytes.as_slice())
    }

    /// [`Self::cell_bounds`] of the cell whose token (without the leaf
    /// byte) is `bytes`.
    fn bytes_bounds(&self, bytes: &[u8]) -> Result<[f64; 4]> {
        // The common case in one pass: a cell no deeper than the tree, all
        // of whose bytes are quadrants. The same additions in the same
        // order as `cell_rect` (a skipped one adds `-0.0`), so the same
        // bits; anything else takes the checking path below for Java's
        // exception.
        let g = &self.geom;
        if bytes.len() <= g.level_w.len() {
            let (mut xmin, mut ymin) = (g.xmin, g.ymin);
            let mut quadrants = true;
            for ((&b, &w), &h) in bytes.iter().zip(&g.level_w).zip(&g.level_h) {
                let c = b.wrapping_sub(b'A');
                quadrants &= c <= 3;
                xmin += if c == 1 || c == 3 { w } else { -0.0 };
                ymin += if c <= 1 { h } else { -0.0 };
            }
            if quadrants {
                return Ok(self.finish(xmin, ymin, bytes.len()));
            }
        }
        self.bounds_error(bytes)
    }

    /// The bounds of a cell `len` levels deep whose running corner sums
    /// are `xmin`/`ymin`.
    fn finish(&self, xmin: f64, ymin: f64, len: usize) -> [f64; 4] {
        let g = &self.geom;
        let (width, height) = match len.checked_sub(1) {
            Some(last) => (g.level_w[last], g.level_h[last]),
            None => (g.grid_w, g.grid_h),
        };
        [xmin, xmin + width, ymin, ymin + height]
    }

    /// Java's switch meets an unexpected byte, or reads `levelW` past its
    /// end, at the first index either happens.
    fn bounds_error(&self, bytes: &[u8]) -> Result<[f64; 4]> {
        let n = self.geom.level_w.len();
        for (i, &c) in bytes.iter().enumerate() {
            match c {
                b'C' => {}
                b'A' | b'B' | b'D' if i >= n => return Err(array_index_out_of_bounds(i, n)),
                b'A' | b'B' | b'D' => {}
                _ => return Err(Error::Runtime(format!("unexpected char: {}", c as i8))),
            }
        }
        let quads = bytes.iter().map(|&c| c - b'A');
        self.geom.cell_rect(quads, bytes.len())
    }
}

/// Relates the cells of a [`QuadPrefixTree`], named by their token bytes,
/// to one shape: `cell.getShape().relate(shape)` for a stream of cells --
/// the visiting traversal's scanned terms, a query cell's children.
///
/// Two things make it cheaper than asking each cell, and neither changes
/// an answer. A cell's corner is a running sum over its quadrants
/// ([`QuadGrid::bytes_bounds`]); consecutive cells share most of their
/// path, so the sums of the shared prefix are kept and only the levels
/// after it are added -- the same additions in the same order, so the same
/// bits. And for a context whose rectangles are `RectangleImpl`s, one
/// rectangle is reset to each cell's (validated, normalised) bounds and
/// related, where `rect(..).relate(shape)` would make one per cell.
pub struct QuadCellRelater {
    grid: Arc<QuadGrid>,
    shape: Arc<dyn Shape>,
    /// The reused rectangle; `None` when the context's factory makes
    /// another kind of rectangle (Geo3D).
    scratch: Option<RectangleImpl>,
    /// The query shape when it is a `RectangleImpl`: related without
    /// dynamic calls.
    query_rect: Option<RectangleImpl>,
    /// The quadrant bytes whose running sums are in `sums`.
    path: Vec<u8>,
    /// `(xmin, ymin)` after each byte of `path`.
    sums: Vec<(f64, f64)>,
}

impl fmt::Debug for QuadCellRelater {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "QuadCellRelater({})", self.shape)
    }
}

impl QuadCellRelater {
    fn new(grid: Arc<QuadGrid>, shape: Arc<dyn Shape>) -> Self {
        let ctx = &grid.geom.ctx;
        let scratch = ctx
            .plain_rect_bounds(-180.0, 180.0, -90.0, 90.0)
            .map(|_| RectangleImpl::new(0.0, 0.0, 0.0, 0.0, ctx.clone()));
        let query_rect = shape.as_any().downcast_ref::<RectangleImpl>().cloned();
        QuadCellRelater {
            grid,
            shape,
            scratch,
            query_rect,
            path: Vec::new(),
            sums: Vec::new(),
        }
    }

    /// The tree's `maxLevels`.
    pub fn max_levels(&self) -> i32 {
        self.grid.geom.max_levels
    }

    /// `readCell(term)`'s `readLeafAdjust()` on raw term bytes: the token
    /// without the leaf byte, and whether the cell is a leaf.
    pub fn split_term<'t>(&self, term: &'t [u8]) -> (&'t [u8], bool) {
        let (bytes, leaf) = match term.split_last() {
            Some((&legacy::LEAF_BYTE, rest)) => (rest, true),
            _ => (term, false),
        };
        let leaf = leaf || bytes.len() as i32 == self.grid.geom.max_levels;
        (bytes, leaf)
    }

    /// The bounds of the cell whose token (without the leaf byte) is
    /// `bytes`, before the factory's validation: `QuadCell.makeShape()`'s
    /// arithmetic.
    ///
    /// # Errors
    /// A byte that is not a quadrant, or a path deeper than the tree
    /// (Java's exceptions, as [`QuadGrid::bytes_bounds`]).
    pub fn bounds(&mut self, bytes: &[u8]) -> Result<[f64; 4]> {
        let g = &self.grid.geom;
        if bytes.len() > g.level_w.len() {
            return self.grid.bytes_bounds(bytes);
        }
        let mut keep = 0;
        while keep < self.path.len() && keep < bytes.len() && self.path[keep] == bytes[keep] {
            keep += 1;
        }
        self.path.truncate(keep);
        self.sums.truncate(keep);
        let (mut xmin, mut ymin) = self.sums.last().copied().unwrap_or((g.xmin, g.ymin));
        for (i, &b) in bytes.iter().enumerate().skip(keep) {
            let c = b.wrapping_sub(b'A');
            if c > 3 {
                return self.grid.bytes_bounds(bytes);
            }
            xmin += if c == 1 || c == 3 { g.level_w[i] } else { -0.0 };
            ymin += if c <= 1 { g.level_h[i] } else { -0.0 };
            self.path.push(b);
            self.sums.push((xmin, ymin));
        }
        Ok(self.grid.finish(xmin, ymin, bytes.len()))
    }

    /// `cell.getShape().relate(shape)` for the cell whose token (without
    /// the leaf byte) is `bytes`.
    ///
    /// # Errors
    /// [`Self::bounds`]', the factory's validation, or the relation's.
    pub fn relate(&mut self, bytes: &[u8]) -> Result<SpatialRelation> {
        let [a, b, c, d] = self.bounds(bytes)?;
        let ctx = &self.grid.geom.ctx;
        match &mut self.scratch {
            Some(rect) => {
                rect.reset(ShapeFactoryImpl::rect_bounds(ctx, a, b, c, d)?);
                match &self.query_rect {
                    // `RectangleImpl.relate(Shape)` for a rectangle
                    Some(q) if !q.is_empty() && !rect.is_empty() => rect.relate_rect(q),
                    _ => rect.relate(&*self.shape),
                }
            }
            None => ctx.rect_relate(a, b, c, d, &*self.shape),
        }
    }
}

impl LegacyGrid for QuadGrid {
    fn max_levels(&self) -> i32 {
        self.geom.max_levels
    }

    fn ctx(&self) -> &Arc<SpatialContext> {
        &self.geom.ctx
    }

    fn get_cell(self: Arc<Self>, p: &dyn Point, level: i32) -> Result<LegacyCell> {
        let bytes: Vec<u8> = self
            .geom
            .quads_for_point(p, level)
            .into_iter()
            .map(|c| b'A' + c)
            .collect();
        let mut cell = LegacyCell::new(self, &bytes);
        cell.shape_rel = Some(crate::spatial4j::SpatialRelation::Contains);
        Ok(cell)
    }

    fn sub_cells(self: Arc<Self>, cell: &LegacyCell) -> Result<Vec<LegacyCell>> {
        let grid: Arc<dyn LegacyGrid> = self;
        Ok(b"ABCD"
            .iter()
            .map(|&b| LegacyCell::from_bytes(grid.clone(), cell.bytes.with(b)))
            .collect())
    }

    fn sub_cells_size(&self) -> i32 {
        4
    }

    fn child_labels(&self) -> Option<&'static [u8]> {
        Some(b"ABCD")
    }

    fn make_shape(&self, cell: &LegacyCell) -> Result<Arc<dyn Shape>> {
        let [a, b, c, d] = self.cell_bounds(cell)?;
        Ok(self.geom.ctx.rect(a, b, c, d)?)
    }

    fn rect_bounds(&self, cell: &LegacyCell) -> Option<Result<[f64; 4]>> {
        let [a, b, c, d] = match self.cell_bounds(cell) {
            Ok(v) => v,
            Err(e) => return Some(Err(e)),
        };
        self.geom.ctx.plain_rect_bounds(a, b, c, d)
    }

    fn relate_cell(&self, cell: &LegacyCell, other: &dyn Shape) -> Result<SpatialRelation> {
        let [a, b, c, d] = self.cell_bounds(cell)?;
        self.geom.ctx.rect_relate(a, b, c, d, other)
    }
}

impl fmt::Display for QuadPrefixTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "QuadPrefixTree(maxLevels:{},ctx:{})",
            self.inner.geom.max_levels, self.inner.geom.ctx
        )
    }
}

impl SpatialPrefixTree for QuadPrefixTree {
    fn spatial_context(&self) -> &Arc<SpatialContext> {
        &self.inner.geom.ctx
    }

    fn max_levels(&self) -> i32 {
        self.inner.geom.max_levels
    }

    fn level_for_distance(&self, dist: f64) -> i32 {
        self.inner.geom.level_for_distance(dist)
    }

    fn distance_for_level(&self, level: i32) -> Result<f64> {
        legacy::distance_for_level(self.grid(), level)
    }

    fn world_cell(&self) -> Box<dyn Cell> {
        Box::new(LegacyCell::new(self.grid(), &[]))
    }

    fn read_cell(&self, term: &[u8]) -> Result<Box<dyn Cell>> {
        Ok(Box::new(LegacyCell::new(self.grid(), term)))
    }

    fn read_cell_into(&self, term: &[u8], scratch: &mut Box<dyn Cell>) -> Result<()> {
        LegacyCell::read_into(&self.inner, term, scratch);
        Ok(())
    }

    fn quad_relater(&self, shape: &Arc<dyn Shape>) -> Option<QuadCellRelater> {
        Some(QuadCellRelater::new(self.inner.clone(), shape.clone()))
    }

    fn tree_cell_iterator(
        &self,
        shape: &Arc<dyn Shape>,
        detail_level: i32,
    ) -> Result<Box<dyn CellIterator>> {
        match shape.as_point() {
            Some(p) => legacy::point_cell_iterator(self.grid(), p, detail_level),
            None => super::default_tree_cell_iterator(self, shape, detail_level),
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
