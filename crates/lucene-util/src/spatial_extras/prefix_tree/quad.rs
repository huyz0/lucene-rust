//! `QuadPrefixTree`: each level splits a cell into four quadrants, `A`
//! (upper left), `B` (upper right), `C` (lower left), `D` (lower right) --
//! one byte of the term per level.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use super::legacy::{self, LegacyCell, LegacyGrid};
use super::{Cell, CellIterator, SpatialPrefixTree};
use crate::spatial4j::{Error, Point, Result, Shape, SpatialContext};

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

    /// `QuadCell.makeShape()`'s corner and size for the quadrant path.
    pub(crate) fn cell_rect(&self, quads: impl Iterator<Item = u8>, len: usize) -> [f64; 4] {
        let mut xmin = self.xmin;
        let mut ymin = self.ymin;
        for (i, c) in quads.enumerate() {
            match c {
                0 => ymin += self.level_h[i],
                1 => {
                    xmin += self.level_w[i];
                    ymin += self.level_h[i];
                }
                2 => {}
                _ => xmin += self.level_w[i],
            }
        }
        let (width, height) = if len > 0 {
            (self.level_w[len - 1], self.level_h[len - 1])
        } else {
            (self.grid_w, self.grid_h)
        };
        [xmin, xmin + width, ymin, ymin + height]
    }
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
            .map(|&b| {
                let mut bytes = cell.bytes.clone();
                bytes.push(b);
                LegacyCell::new(grid.clone(), &bytes)
            })
            .collect())
    }

    fn sub_cells_size(&self) -> i32 {
        4
    }

    fn make_shape(&self, cell: &LegacyCell) -> Result<Arc<dyn Shape>> {
        let mut quads = Vec::with_capacity(cell.bytes.len());
        for &c in &cell.bytes {
            match c {
                b'A'..=b'D' => quads.push(c - b'A'),
                _ => return Err(Error::Runtime(format!("unexpected char: {}", c as i8))),
            }
        }
        let [a, b, c, d] = self.geom.cell_rect(quads.into_iter(), cell.bytes.len());
        Ok(self.geom.ctx.rect(a, b, c, d)?)
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
