//! `GeohashPrefixTree`: cells are geohashes, one base-32 character (32
//! cells) per level.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use super::legacy::{self, LegacyCell, LegacyGrid};
use super::{Cell, CellIterator, SpatialPrefixTree};
use crate::spatial4j::{geohash, Error, Point, Result, Shape, SpatialContext};

/// `GeohashPrefixTree`.
#[derive(Debug, Clone)]
pub struct GeohashPrefixTree {
    inner: Arc<GhGrid>,
}

#[derive(Debug)]
struct GhGrid {
    ctx: Arc<SpatialContext>,
    max_levels: i32,
}

impl GeohashPrefixTree {
    /// `new GeohashPrefixTree(ctx, maxLevels)`: lat/lon world bounds only.
    pub fn new(ctx: Arc<SpatialContext>, max_levels: i32) -> Result<Self> {
        let bounds = ctx.world_bounds_values();
        if bounds[0] != -180.0 {
            return Err(Error::IllegalArgument(format!(
                "Geohash only supports lat-lon world bounds. Got {}",
                ctx.world_bounds()
            )));
        }
        let maxp = Self::max_levels_possible();
        if max_levels <= 0 || max_levels > maxp {
            return Err(Error::IllegalArgument(format!(
                "maxLevels must be [1-{maxp}] but got {max_levels}"
            )));
        }
        Ok(GeohashPrefixTree {
            inner: Arc::new(GhGrid { ctx, max_levels }),
        })
    }

    /// `getMaxLevelsPossible()`.
    pub fn max_levels_possible() -> i32 {
        geohash::MAX_PRECISION as i32
    }

    /// `getCell(p, level)`.
    pub fn get_cell(&self, p: &dyn Point, level: i32) -> Result<Box<dyn Cell>> {
        Ok(Box::new(self.inner.clone().get_cell(p, level)?))
    }

    fn grid(&self) -> Arc<dyn LegacyGrid> {
        self.inner.clone()
    }
}

impl LegacyGrid for GhGrid {
    fn max_levels(&self) -> i32 {
        self.max_levels
    }

    fn ctx(&self) -> &Arc<SpatialContext> {
        &self.ctx
    }

    fn get_cell(self: Arc<Self>, p: &dyn Point, level: i32) -> Result<LegacyCell> {
        // args are lat,lon (y,x)
        let hash = geohash::encode_lat_lon(p.y(), p.x(), level.max(0) as usize);
        Ok(LegacyCell::new(self, hash.as_bytes()))
    }

    fn sub_cells(self: Arc<Self>, cell: &LegacyCell) -> Result<Vec<LegacyCell>> {
        let base = String::from_utf8_lossy(&cell.bytes).into_owned();
        let grid: Arc<dyn LegacyGrid> = self;
        Ok(geohash::sub_geohashes(&base)
            .into_iter()
            .map(|h| LegacyCell::new(grid.clone(), h.as_bytes()))
            .collect())
    }

    fn sub_cells_size(&self) -> i32 {
        32 // 8x4
    }

    fn make_shape(&self, cell: &LegacyCell) -> Result<Arc<dyn Shape>> {
        let hash = String::from_utf8_lossy(&cell.bytes).into_owned();
        Ok(geohash::decode_boundary(&hash, &self.ctx)?)
    }
}

impl fmt::Display for GeohashPrefixTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GeohashPrefixTree(maxLevels:{},ctx:{})",
            self.inner.max_levels, self.inner.ctx
        )
    }
}

impl SpatialPrefixTree for GeohashPrefixTree {
    fn spatial_context(&self) -> &Arc<SpatialContext> {
        &self.inner.ctx
    }

    fn max_levels(&self) -> i32 {
        self.inner.max_levels
    }

    fn level_for_distance(&self, dist: f64) -> i32 {
        if dist == 0.0 {
            return self.inner.max_levels;
        }
        let level = geohash::lookup_hash_len_for_width_height(dist, dist) as i32;
        level.min(self.inner.max_levels).max(1)
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
