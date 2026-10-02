//! `org.apache.lucene.spatial.prefix.tree`: spatial prefix trees.

use std::sync::Arc;

use crate::s2::S2CellId;
use crate::spatial4j::{Result, Shape, SpatialContext};

/// `S2ShapeFactory`: a shape factory that can make the shape of an S2 cell.
pub trait S2ShapeFactory {
    /// `getS2CellShape(cellId)`.
    fn s2_cell_shape(&self, ctx: &Arc<SpatialContext>, cell_id: S2CellId)
        -> Result<Arc<dyn Shape>>;
}
