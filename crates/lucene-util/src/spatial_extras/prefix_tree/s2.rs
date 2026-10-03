//! `S2PrefixTree` and `S2PrefixTreeCell`: S2's cells on the six cube faces,
//! `arity` S2 levels (`4^arity` children) per tree level. A term is the
//! face's token, then one token per level from a 64-symbol alphabet, plus
//! `+` for a leaf above the last level. Cell shapes come from the
//! context's [`S2ShapeFactory`](super::S2ShapeFactory) (Lucene's is the Geo3D factory).

use std::any::Any;
use std::fmt;
use std::sync::{Arc, OnceLock};

use super::{Cell, CellIterator, FilterCellIterator, SpatialPrefixTree};
use crate::s2::{projections, S2CellId, S2LatLng, MAX_LEVEL};
use crate::spatial4j::{DistanceUtils, Error, Result, Shape, SpatialContext, SpatialRelation};

/// `S2PrefixTreeCell.LEAF`.
const LEAF: u8 = b'+';

/// `S2PrefixTreeCell.TOKENS`.
const TOKENS: &[u8; 64] = b"./0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

/// `S2PrefixTreeCell.PIXELS`: a token's value.
fn pixel(b: u8) -> Option<i64> {
    TOKENS.iter().position(|&t| t == b).map(|p| p as i64)
}

/// `S2PrefixTreeCell.FACES`.
fn face(i: i32) -> S2CellId {
    S2CellId::from_face_pos_level(i, 0, 0)
}

/// `S2PrefixTree`.
#[derive(Debug, Clone)]
pub struct S2PrefixTree {
    inner: Arc<S2Grid>,
}

#[derive(Debug)]
struct S2Grid {
    ctx: Arc<SpatialContext>,
    max_levels: i32,
    arity: i32,
}

impl S2PrefixTree {
    /// `new S2PrefixTree(ctx, maxLevels, arity)`: the context's shape
    /// factory must be an `S2ShapeFactory`; `arity` is 1, 2 or 3.
    pub fn new(ctx: Arc<SpatialContext>, max_levels: i32, arity: i32) -> Result<Self> {
        if ctx.shape_factory().as_s2().is_none() {
            return Err(Error::IllegalArgument(
                "Spatial context does not support S2 spatial index.".into(),
            ));
        }
        if !(1..=3).contains(&arity) {
            return Err(Error::IllegalArgument(format!(
                "Invalid value for S2 tree arity. Possible values are 1, 2 or 3. Provided value is {arity}."
            )));
        }
        Ok(S2PrefixTree {
            inner: Arc::new(S2Grid {
                ctx,
                max_levels,
                arity,
            }),
        })
    }

    /// `getMaxLevels(arity)`.
    pub fn max_levels_for_arity(arity: i32) -> i32 {
        MAX_LEVEL / arity + 1
    }

    fn cell(&self, cell_id: Option<S2CellId>) -> S2PrefixTreeCell {
        S2PrefixTreeCell::new(self.inner.clone(), cell_id)
    }
}

/// `S2PrefixTreeCell`: `cell_id` is `None` for the world cell.
#[derive(Clone)]
pub struct S2PrefixTreeCell {
    tree: Arc<S2Grid>,
    cell_id: Option<S2CellId>,
    level: i32,
    shape_rel: Option<SpatialRelation>,
    is_leaf: bool,
    shape: OnceLock<Arc<dyn Shape>>,
}

impl fmt::Debug for S2PrefixTreeCell {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl S2PrefixTreeCell {
    fn new(tree: Arc<S2Grid>, cell_id: Option<S2CellId>) -> Self {
        let mut c = S2PrefixTreeCell {
            tree,
            cell_id,
            level: 0,
            shape_rel: None,
            is_leaf: false,
            shape: OnceLock::new(),
        };
        c.set_level();
        if c.level == c.tree.max_levels {
            c.is_leaf = true;
        }
        c
    }

    /// `setLevel()`.
    fn set_level(&mut self) {
        self.level = match self.cell_id {
            None => 0,
            Some(id) => id.level() / self.tree.arity + 1,
        };
    }

    /// `readCell(tree, ref)`.
    fn read(tree: Arc<S2Grid>, term: &[u8]) -> Result<Self> {
        let is_leaf = term.last() == Some(&LEAF);
        let cell_id = Self::cell_id_from_bytes(&tree, term)?;
        let mut c = S2PrefixTreeCell {
            tree,
            cell_id,
            level: 0,
            shape_rel: None,
            is_leaf: false,
            shape: OnceLock::new(),
        };
        c.set_level();
        if is_leaf || c.level == c.tree.max_levels {
            c.is_leaf = true;
        }
        Ok(c)
    }

    /// `getS2CellIdFromBytesRef(ref)`.
    fn cell_id_from_bytes(tree: &S2Grid, term: &[u8]) -> Result<Option<S2CellId>> {
        // Java reads `bytes[end - 1]` before anything else, so an empty term
        // is an `ArrayIndexOutOfBoundsException` (not the world cell).
        if term.is_empty() {
            return Err(Error::ArrayIndexOutOfBounds(
                "Index -1 out of bounds for length 0".into(),
            ));
        }
        let mut length = term.len();
        if term.last() == Some(&LEAF) {
            length -= 1;
        }
        if length == 0 {
            return Ok(None); // world cell
        }
        let bad = |b: u8| Error::Runtime(format!("java.lang.NullPointerException: no token {b}"));
        let f = pixel(term[0]).ok_or_else(|| bad(term[0]))?;
        let mut id = face(f as i32).id();
        for (i, &b) in term.iter().enumerate().take(length).skip(1) {
            let this_level = i as i32;
            let pos = pixel(b).ok_or_else(|| bad(b))?;
            // first child at level
            id = id.wrapping_sub(id & id.wrapping_neg()).wrapping_add(
                1i64.wrapping_shl((2 * (MAX_LEVEL - this_level * tree.arity)) as u32),
            );
            // next until pos
            id = id.wrapping_add(pos.wrapping_mul((id & id.wrapping_neg()).wrapping_shl(1)));
        }
        Ok(Some(S2CellId::new(id)))
    }

    /// `getBytesRefFromS2CellId(cellId, bref)`.
    fn token(&self) -> Vec<u8> {
        let Some(cell_id) = self.cell_id else {
            return Vec::new();
        };
        let arity = self.tree.arity;
        // An id with no level (only a corrupt term decodes to one): Java's
        // `new byte[level]` then `b[0]` is an `ArrayIndexOutOfBoundsException`
        // (`NegativeArraySizeException` below zero); a token has no error
        // to return, so it is empty.
        if self.level < 1 {
            return Vec::new();
        }
        let mut b = vec![0u8; self.level as usize];
        b[0] = TOKENS[cell_id.face() as usize];
        for i in 1..self.level {
            let mut offset = 0;
            let level = arity * i;
            for j in 1..arity {
                offset = 4 * offset + cell_id.child_position(level - arity + j);
            }
            b[i as usize] = TOKENS[(4 * offset + cell_id.child_position(level)) as usize];
        }
        b
    }

    /// The cell's S2 id; `None` for the world.
    pub fn cell_id(&self) -> Option<S2CellId> {
        self.cell_id
    }
}

impl fmt::Display for S2PrefixTreeCell {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.cell_id {
            None => f.write_str("0"),
            Some(id) => write!(f, "{id}"),
        }
    }
}

impl Cell for S2PrefixTreeCell {
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
        let mut result = self.token();
        // max levels do not have leaf
        if self.is_leaf && self.level != self.tree.max_levels {
            result.push(LEAF);
        }
        result
    }

    fn token_bytes_no_leaf(&self) -> Vec<u8> {
        self.token()
    }

    fn level(&self) -> i32 {
        self.level
    }

    fn next_level_cells(
        &self,
        shape_filter: Option<&Arc<dyn Shape>>,
    ) -> Result<Box<dyn CellIterator>> {
        let children: Vec<S2CellId> = match self.cell_id {
            None => (0..6).map(face).collect(),
            Some(id) => {
                let n = 4usize.pow(self.tree.arity as u32);
                let mut v = Vec::with_capacity(n);
                let mut c = id.child_begin(id.level() + self.tree.arity);
                v.push(c);
                for _ in 1..n {
                    c = c.next();
                    v.push(c);
                }
                v
            }
        };
        let cells: Vec<Box<dyn Cell>> = children
            .into_iter()
            .map(|id| Box::new(S2PrefixTreeCell::new(self.tree.clone(), Some(id))) as Box<dyn Cell>)
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
        let ctx = &self.tree.ctx;
        let s: Arc<dyn Shape> = match self.cell_id {
            None => Arc::new(ctx.world_bounds()),
            Some(id) => ctx
                .shape_factory()
                .as_s2()
                .expect("checked at construction")
                .s2_cell_shape(ctx, id)?,
        };
        Ok(self.shape.get_or_init(|| s).clone())
    }

    fn is_prefix_of(&self, c: &dyn Cell) -> bool {
        let Some(id) = self.cell_id else {
            return true;
        };
        match c
            .as_any()
            .downcast_ref::<S2PrefixTreeCell>()
            .and_then(|c| c.cell_id)
        {
            Some(other) => id.contains(&other),
            None => false,
        }
    }

    fn compare_to_no_leaf(&self, from_cell: &dyn Cell) -> i32 {
        let Some(id) = self.cell_id else {
            return 1;
        };
        match from_cell
            .as_any()
            .downcast_ref::<S2PrefixTreeCell>()
            .and_then(|c| c.cell_id)
        {
            Some(other) => id.cmp(&other) as i32,
            // Java's `compareTo` dereferences the world cell's null id.
            None => 1,
        }
    }

    fn sub_cells_size(&self) -> Option<i32> {
        Some(match self.cell_id {
            None => 6,
            Some(_) => 4i32.pow(self.tree.arity as u32),
        })
    }

    fn clone_box(&self) -> Box<dyn Cell> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl fmt::Display for S2PrefixTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "S2PrefixTree(maxLevels:{},ctx:{})",
            self.inner.max_levels, self.inner.ctx
        )
    }
}

impl SpatialPrefixTree for S2PrefixTree {
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
        let level = projections::MAX_WIDTH.get_min_level(dist * DistanceUtils::DEGREES_TO_RADIANS);
        let arity = self.inner.arity;
        let round_level = if level % arity != 0 { 1 } else { 0 };
        let level = level / arity + round_level;
        self.inner.max_levels.min(level + 1)
    }

    fn distance_for_level(&self, level: i32) -> Result<f64> {
        if level == 0 {
            return Ok(180.0);
        }
        Ok(
            projections::MAX_WIDTH.get_value(self.inner.arity * (level - 1))
                * DistanceUtils::RADIANS_TO_DEGREES,
        )
    }

    fn world_cell(&self) -> Box<dyn Cell> {
        Box::new(self.cell(None))
    }

    fn read_cell(&self, term: &[u8]) -> Result<Box<dyn Cell>> {
        Ok(Box::new(S2PrefixTreeCell::read(self.inner.clone(), term)?))
    }

    fn tree_cell_iterator(
        &self,
        shape: &Arc<dyn Shape>,
        detail_level: i32,
    ) -> Result<Box<dyn CellIterator>> {
        let Some(p) = shape.as_point() else {
            return super::default_tree_cell_iterator(self, shape, detail_level);
        };
        let arity = self.inner.arity;
        let id = S2CellId::from_lat_lng(&S2LatLng::from_degrees(p.y(), p.x()))
            .parent(arity * (detail_level - 1));
        let mut cells: Vec<Box<dyn Cell>> = Vec::with_capacity(detail_level.max(0) as usize);
        for i in 0..(detail_level - 1).max(0) {
            cells.push(Box::new(self.cell(Some(id.parent(i * arity)))));
        }
        cells.push(Box::new(self.cell(Some(id))));
        Ok(Box::new(FilterCellIterator::new(cells, None)))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
