//! `org.apache.lucene.spatial.prefix.tree`: spatial prefix trees -- a
//! hierarchy of cells, each named by a byte string extending its parent's,
//! so that a shape indexes as the terms of the cells covering it.
//!
//! - [`QuadPrefixTree`] (four cells per level, one byte `A`-`D` each),
//!   [`PackedQuadPrefixTree`] (the same quads packed into an 8-byte term),
//!   [`GeohashPrefixTree`] (32 cells per level, a geohash character each),
//!   [`S2PrefixTree`] (S2's Hilbert-curve cells, `4^arity` per level, a
//!   base-64 character each);
//! - [`NumberRangePrefixTree`] and its [`DateRangePrefixTree`]: a
//!   one-dimensional tree whose cells are also shapes (`UnitNRShape`),
//!   indexing numbers or instants and their ranges.
//!
//! # What Rust changes
//!
//! - `Cell` and `CellIterator` are traits. Java's iterators hand out cells
//!   they may later reuse (an `NRCell` *is* its level's iterator); here
//!   every cell handed out is an owned value ([`Cell::clone_box`]) and an
//!   iterator keeps its own copy of the last one ([`CellIterator::this_cell`]).
//!   No caller in spatial-extras keeps a cell across `next()` expecting it to
//!   change, so the terms and relations are the same.
//! - The "scratch" `BytesRef` arguments (`getTokenBytes*(BytesRef)`) are
//!   fresh values, except `getTokenBytesNoLeaf(scratch)`, which is
//!   [`Cell::token_bytes_no_leaf_into`]. `readCell(term, scratch)` is
//!   [`SpatialPrefixTree::read_cell_into`].
//! - A quad cell's children (`getNextLevelCells` over `getSubCells`) are
//!   made one at a time as the filter accepts them, from one scratch
//!   child, instead of as a list of all four first: the same cells, in the
//!   same order, with the same relations.
//! - [`CellIterator::next_detached`] is `next()` without keeping the copy
//!   `thisCell()` would return, for the visiting traversal, which never asks
//!   for it.
//! - A quad tree's cells can be related to a shape from their token bytes
//!   alone ([`QuadCellRelater`], [`SpatialPrefixTree::quad_relater`]): the
//!   corner sums of the path a stream of cells shares are kept, and one
//!   `RectangleImpl` is reset to each cell's bounds and related, where
//!   Java makes a `QuadCell` and its rectangle per cell. [`Cell::rect_bounds`]
//!   gives a cell's rectangle as bounds, for a heatmap that only reads them.
//!   Same relations and bounds, bit for bit (stage 3 of the port).
//! - `QuadPrefixTree.buildNotRobustly`/`checkBattenbergNotRobustly` and
//!   `printInfo`, which nothing in the module calls, are not ported.

use std::any::Any;
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use crate::s2::S2CellId;
use crate::spatial4j::{DistanceUtils, Error, Result, Shape, SpatialContext, SpatialRelation};

pub mod date_range;
pub mod geohash;
pub mod java_calendar;
mod legacy;
pub mod number_range;
pub mod packed_quad;
pub mod quad;
pub mod s2;

pub use date_range::{DateRangePrefixTree, YEAR_LEVEL};
pub use geohash::GeohashPrefixTree;
pub use java_calendar::Calendar;
pub use legacy::LegacyCell;
pub use number_range::{
    dummy_ctx, NRShape, NrBase, NumberRangePrefixTree, NumberRangeTree, SpanUnitsNRShape,
    UnitNRShape,
};
pub use packed_quad::PackedQuadPrefixTree;
pub use quad::{QuadCellRelater, QuadPrefixTree};
pub use s2::S2PrefixTree;

/// `S2ShapeFactory`: a shape factory that can make the shape of an S2 cell.
pub trait S2ShapeFactory {
    /// `getS2CellShape(cellId)`.
    fn s2_cell_shape(&self, ctx: &Arc<SpatialContext>, cell_id: S2CellId)
        -> Result<Arc<dyn Shape>>;
}

/// `SpatialPrefixTree`. `Display` is Java's `toString()`.
pub trait SpatialPrefixTree: fmt::Debug + fmt::Display + Send + Sync + Any {
    /// `getSpatialContext()`.
    fn spatial_context(&self) -> &Arc<SpatialContext>;

    /// `getMaxLevels()`.
    fn max_levels(&self) -> i32;

    /// `getLevelForDistance(dist)`: the coarsest level whose cells are no
    /// larger than `dist` (degrees for geo).
    fn level_for_distance(&self, dist: f64) -> i32;

    /// `getDistanceForLevel(level)`.
    fn distance_for_level(&self, level: i32) -> Result<f64>;

    /// `getWorldCell()`: the level-0 cell.
    fn world_cell(&self) -> Box<dyn Cell>;

    /// `readCell(term, null)`: the cell an indexed term names (a leaf if the
    /// term carries the leaf marker).
    fn read_cell(&self, term: &[u8]) -> Result<Box<dyn Cell>>;

    /// `readCell(term, scratch)`: [`Self::read_cell`] into `scratch`, reusing
    /// its storage when the tree can (Java's scratch cell). The default
    /// replaces it.
    fn read_cell_into(&self, term: &[u8], scratch: &mut Box<dyn Cell>) -> Result<()> {
        *scratch = self.read_cell(term)?;
        Ok(())
    }

    /// `getTreeCellIterator(shape, detailLevel)`: the cells covering
    /// `shape`, in term order, down to `detail_level`.
    fn tree_cell_iterator(
        &self,
        shape: &Arc<dyn Shape>,
        detail_level: i32,
    ) -> Result<Box<dyn CellIterator>> {
        default_tree_cell_iterator(self, shape, detail_level)
    }

    /// A [`QuadCellRelater`] of this tree's cells against `shape`: `Some`
    /// for a [`QuadPrefixTree`], whose cells' relations it answers from
    /// their term bytes.
    fn quad_relater(&self, _shape: &Arc<dyn Shape>) -> Option<QuadCellRelater> {
        None
    }

    /// For downcasting.
    fn as_any(&self) -> &dyn Any;
}

/// `SpatialPrefixTree.getTreeCellIterator`'s base implementation.
pub fn default_tree_cell_iterator<T: SpatialPrefixTree + ?Sized>(
    tree: &T,
    shape: &Arc<dyn Shape>,
    detail_level: i32,
) -> Result<Box<dyn CellIterator>> {
    if detail_level > tree.max_levels() {
        return Err(Error::IllegalArgument("detailLevel > maxLevels".into()));
    }
    Ok(Box::new(TreeCellIterator::new(
        Some(shape.clone()),
        detail_level,
        tree.world_cell(),
    )?))
}

/// `Cell` (and `CellCanPrune`). `Display` is Java's `toString()`.
pub trait Cell: fmt::Debug + fmt::Display + Send + Sync + Any {
    /// `getShapeRel()`: the relation to the shape that found this cell, if
    /// any.
    fn shape_rel(&self) -> Option<SpatialRelation>;
    /// `setShapeRel(rel)`.
    fn set_shape_rel(&mut self, rel: Option<SpatialRelation>);
    /// `isLeaf()`.
    fn is_leaf(&self) -> bool;
    /// `setLeaf()`.
    fn set_leaf(&mut self);
    /// `getTokenBytesWithLeaf(null)`: the term, with the leaf marker for a
    /// leaf.
    fn token_bytes_with_leaf(&self) -> Vec<u8>;
    /// `getTokenBytesNoLeaf(null)`.
    fn token_bytes_no_leaf(&self) -> Vec<u8>;
    /// `getTokenBytesNoLeaf(scratch)`: the same bytes into `out` (cleared
    /// first), reusing its storage -- the visiting traversal's
    /// `curVNodeTerm`, one buffer for every seek target.
    fn token_bytes_no_leaf_into(&self, out: &mut Vec<u8>) {
        out.clear();
        out.extend_from_slice(&self.token_bytes_no_leaf());
    }
    /// `getLevel()`.
    fn level(&self) -> i32;
    /// `getNextLevelCells(shapeFilter)`: the children, filtered to those
    /// intersecting the shape (with their relations set) when one is given.
    fn next_level_cells(
        &self,
        shape_filter: Option<&Arc<dyn Shape>>,
    ) -> Result<Box<dyn CellIterator>>;
    /// `getShape()`.
    fn shape(&self) -> Result<Arc<dyn Shape>>;
    /// `getShape().relate(other)`: what the traversals ask of every cell
    /// they meet, answered without keeping the shape where a tree can.
    fn relate_shape(&self, other: &dyn Shape) -> Result<SpatialRelation> {
        self.shape()?.relate(other)
    }
    /// `getShape()`'s `[minX, maxX, minY, maxY]` when the shape is a
    /// plain rectangle (`RectangleImpl`) the tree can describe without
    /// making it; `None` otherwise (ask [`Self::shape`]).
    fn rect_bounds(&self) -> Option<Result<[f64; 4]>> {
        None
    }
    /// `isPrefixOf(c)`.
    fn is_prefix_of(&self, c: &dyn Cell) -> bool;
    /// `compareToNoLeaf(fromCell)`: term order ignoring the leaf marker.
    fn compare_to_no_leaf(&self, from_cell: &dyn Cell) -> i32;
    /// `CellCanPrune.getSubCellsSize()`, or `None` for a cell that is not a
    /// `CellCanPrune`.
    fn sub_cells_size(&self) -> Option<i32>;
    /// A copy of this cell.
    fn clone_box(&self) -> Box<dyn Cell>;
    /// For downcasting.
    fn as_any(&self) -> &dyn Any;
}

impl Clone for Box<dyn Cell> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// `CellIterator`: Java's `hasNext()`/`next()` over cells, plus
/// [`Self::this_cell`].
pub trait CellIterator: Send {
    /// `hasNext()`.
    fn has_next(&mut self) -> Result<bool>;
    /// `next()`: the next cell (`NoSuchElementException` past the end).
    fn next(&mut self) -> Result<Box<dyn Cell>>;
    /// `thisCell()`: the cell `next()` last returned.
    fn this_cell(&self) -> Option<&dyn Cell>;
    /// `next()` for a caller that never asks [`Self::this_cell`] before the
    /// following `hasNext()`: an iterator may hand the cell over without
    /// keeping the copy `thisCell` needs (`this_cell` is then `None`).
    fn next_detached(&mut self) -> Result<Box<dyn Cell>> {
        self.next()
    }
    /// `nextFrom(fromCell)`: the next cell at or after `from_cell`.
    fn next_from(&mut self, from_cell: &dyn Cell) -> Result<Option<Box<dyn Cell>>> {
        loop {
            if !self.has_next()? {
                return Ok(None);
            }
            let c = self.next()?;
            if c.compare_to_no_leaf(from_cell) >= 0 {
                return Ok(Some(c));
            }
        }
    }
    /// `remove()`: for a tree iterator, do not descend into the cell just
    /// returned.
    fn remove(&mut self) {}
}

/// `CellIterator`'s `nextCell`/`thisCell` pair and its `next()`.
#[derive(Debug, Default)]
pub struct IterState {
    pub next_cell: Option<Box<dyn Cell>>,
    pub this_cell: Option<Box<dyn Cell>>,
}

impl IterState {
    /// `CellIterator.next()` once `hasNext()` has filled `next_cell`.
    pub fn take_next(&mut self) -> Result<Box<dyn Cell>> {
        let Some(c) = self.next_cell.take() else {
            return Err(Error::Runtime("java.util.NoSuchElementException".into()));
        };
        let out = c.clone_box();
        self.this_cell = Some(c);
        Ok(out)
    }
}

/// `FilterCellIterator`: cells from a list, keeping those whose shape
/// intersects the filter (with the relation set, and `WITHIN` cells made
/// leaves).
#[derive(Debug)]
pub struct FilterCellIterator {
    base: std::vec::IntoIter<Box<dyn Cell>>,
    shape_filter: Option<Arc<dyn Shape>>,
    st: IterState,
}

impl FilterCellIterator {
    /// `new FilterCellIterator(baseIter, shapeFilter)`.
    pub fn new(cells: Vec<Box<dyn Cell>>, shape_filter: Option<Arc<dyn Shape>>) -> Self {
        FilterCellIterator {
            base: cells.into_iter(),
            shape_filter,
            st: IterState::default(),
        }
    }
}

impl CellIterator for FilterCellIterator {
    fn has_next(&mut self) -> Result<bool> {
        self.st.this_cell = None;
        if self.st.next_cell.is_some() {
            return Ok(true);
        }
        for mut cell in self.base.by_ref() {
            match &self.shape_filter {
                None => {
                    self.st.next_cell = Some(cell);
                    return Ok(true);
                }
                Some(filter) => {
                    let rel = cell.relate_shape(&**filter)?;
                    if rel.intersects() {
                        cell.set_shape_rel(Some(rel));
                        if rel == SpatialRelation::Within {
                            cell.set_leaf();
                        }
                        self.st.next_cell = Some(cell);
                        return Ok(true);
                    }
                }
            }
        }
        Ok(false)
    }

    fn next(&mut self) -> Result<Box<dyn Cell>> {
        if self.st.next_cell.is_none() && !self.has_next()? {
            return Err(Error::Runtime("java.util.NoSuchElementException".into()));
        }
        self.st.take_next()
    }

    fn this_cell(&self) -> Option<&dyn Cell> {
        self.st.this_cell.as_deref()
    }

    fn next_detached(&mut self) -> Result<Box<dyn Cell>> {
        if self.st.next_cell.is_none() && !self.has_next()? {
            return Err(Error::Runtime("java.util.NoSuchElementException".into()));
        }
        self.st.this_cell = None;
        self.st
            .next_cell
            .take()
            .ok_or_else(|| Error::Runtime("java.util.NoSuchElementException".into()))
    }
}

/// `SingletonCellIterator`.
#[derive(Debug)]
pub struct SingletonCellIterator {
    st: IterState,
}

impl SingletonCellIterator {
    pub fn new(cell: Box<dyn Cell>) -> Self {
        SingletonCellIterator {
            st: IterState {
                next_cell: Some(cell),
                this_cell: None,
            },
        }
    }
}

impl CellIterator for SingletonCellIterator {
    fn has_next(&mut self) -> Result<bool> {
        self.st.this_cell = None;
        Ok(self.st.next_cell.is_some())
    }

    fn next(&mut self) -> Result<Box<dyn Cell>> {
        self.st.take_next()
    }

    fn this_cell(&self) -> Option<&dyn Cell> {
        self.st.this_cell.as_deref()
    }
}

/// `TreeCellIterator`: a depth-first walk of the cells intersecting a
/// shape, down to a detail level (whose cells are leaves).
pub struct TreeCellIterator {
    shape_filter: Option<Arc<dyn Shape>>,
    iter_stack: Vec<Option<Box<dyn CellIterator>>>,
    stack_idx: i32,
    descend: bool,
    shape_is_point: bool,
    st: IterState,
}

impl fmt::Debug for TreeCellIterator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TreeCellIterator")
            .field("stack_idx", &self.stack_idx)
            .finish()
    }
}

impl TreeCellIterator {
    /// `new TreeCellIterator(shapeFilter, detailLevel, parentCell)`.
    pub fn new(
        shape_filter: Option<Arc<dyn Shape>>,
        detail_level: i32,
        parent_cell: Box<dyn Cell>,
    ) -> Result<Self> {
        let n = detail_level.max(1) as usize;
        let mut iter_stack: Vec<Option<Box<dyn CellIterator>>> = (0..n).map(|_| None).collect();
        iter_stack[0] = Some(parent_cell.next_level_cells(shape_filter.as_ref())?);
        let shape_is_point = shape_filter
            .as_ref()
            .is_some_and(|s| s.as_point().is_some());
        Ok(TreeCellIterator {
            shape_filter,
            iter_stack,
            stack_idx: 0,
            descend: false,
            shape_is_point,
            st: IterState::default(),
        })
    }

    fn top(&mut self) -> &mut Box<dyn CellIterator> {
        self.iter_stack[self.stack_idx as usize]
            .as_mut()
            .expect("the stack index always points to an iterator")
    }
}

impl CellIterator for TreeCellIterator {
    fn has_next(&mut self) -> Result<bool> {
        if self.st.next_cell.is_some() {
            return Ok(true);
        }
        let last = self.iter_stack.len() as i32 - 1;
        loop {
            if self.stack_idx == -1 {
                return Ok(false);
            }
            // If we can descend...
            if self.descend {
                let this_cell_leaf = self
                    .top()
                    .this_cell()
                    .expect("descend only after a cell was returned")
                    .is_leaf();
                if !(self.stack_idx == last || this_cell_leaf) {
                    let filter = self.shape_filter.clone();
                    let next_iter = self
                        .top()
                        .this_cell()
                        .expect("checked above")
                        .next_level_cells(filter.as_ref())?;
                    self.stack_idx += 1;
                    self.iter_stack[self.stack_idx as usize] = Some(next_iter);
                }
            }
            // Get sibling...
            if self.top().has_next()? {
                let mut next = self.top().next()?;
                if self.stack_idx == last && !self.shape_is_point {
                    next.set_leaf(); // because at bottom
                }
                self.st.next_cell = Some(next);
                break;
            }
            // Couldn't get next; go up...
            self.iter_stack[self.stack_idx as usize] = None;
            self.stack_idx -= 1;
            self.descend = false;
        }
        self.descend = true;
        Ok(true)
    }

    fn next(&mut self) -> Result<Box<dyn Cell>> {
        if self.st.next_cell.is_none() && !self.has_next()? {
            return Err(Error::Runtime("java.util.NoSuchElementException".into()));
        }
        self.st.take_next()
    }

    fn this_cell(&self) -> Option<&dyn Cell> {
        self.st.this_cell.as_deref()
    }

    fn remove(&mut self) {
        self.descend = false;
    }
}

/// `SpatialPrefixTreeFactory.makeSPT(args, classLoader, ctx)`: the tree
/// named by `prefixTree` (`geohash`, `quad`, `packedQuad`, `s2`; geohash
/// for geo and quad otherwise by default), with `maxLevels` or the level
/// of `maxDistErr` (1 m for geo by default). `version` is read and
/// ignored, as Java ignores it.
pub fn make_spt(
    args: &BTreeMap<String, String>,
    ctx: &Arc<SpatialContext>,
) -> Result<Arc<dyn SpatialPrefixTree>> {
    let cname = args
        .get("prefixTree")
        .cloned()
        .unwrap_or_else(|| if ctx.is_geo() { "geohash" } else { "quad" }.into());
    let kind = cname.to_ascii_lowercase();
    let level_for_distance = |degrees: f64| -> Result<i32> {
        Ok(match kind.as_str() {
            "geohash" => {
                GeohashPrefixTree::new(ctx.clone(), GeohashPrefixTree::max_levels_possible())?
                    .level_for_distance(degrees)
            }
            "quad" => QuadPrefixTree::new(ctx.clone(), quad::MAX_LEVELS_POSSIBLE)?
                .level_for_distance(degrees),
            // `PackedQuadPrefixTree.Factory` inherits `QuadPrefixTree.Factory`'s
            // `getLevelForDistance`, which asks its own `newSPT()`.
            "packedquad" => {
                PackedQuadPrefixTree::new(ctx.clone(), packed_quad::MAX_LEVELS_POSSIBLE)?
                    .level_for_distance(degrees)
            }
            _ => S2PrefixTree::new(ctx.clone(), S2PrefixTree::max_levels_for_arity(1), 1)?
                .level_for_distance(degrees),
        })
    };
    if !matches!(kind.as_str(), "geohash" | "quad" | "packedquad" | "s2") {
        return Err(Error::Runtime(format!(
            "java.lang.ClassNotFoundException: {cname}"
        )));
    }
    let max_levels: Option<i32> = if let Some(ml) = args.get("maxLevels") {
        Some(
            super::prefix_tree::date_range::java_parse_int(ml)
                .ok_or_else(|| Error::NumberFormat(format!("For input string: \"{ml}\"")))?,
        )
    } else {
        match args.get("maxDistErr") {
            None if !ctx.is_geo() => None,
            None => Some(level_for_distance(DistanceUtils::dist2_degrees(
                0.001,
                DistanceUtils::EARTH_MEAN_RADIUS_KM,
            ))?),
            Some(d) => {
                let degrees = crate::spatial4j::wkt::java_parse_double(d.trim()).map_err(|m| {
                    Error::NumberFormat(
                        m.trim_start_matches("java.lang.NumberFormatException: ")
                            .into(),
                    )
                })?;
                Some(level_for_distance(degrees)?)
            }
        }
    };
    Ok(match kind.as_str() {
        "geohash" => Arc::new(GeohashPrefixTree::new(
            ctx.clone(),
            max_levels.unwrap_or(GeohashPrefixTree::max_levels_possible()),
        )?),
        "quad" => Arc::new(QuadPrefixTree::new(
            ctx.clone(),
            max_levels.unwrap_or(quad::MAX_LEVELS_POSSIBLE),
        )?),
        "packedquad" => Arc::new(PackedQuadPrefixTree::new(
            ctx.clone(),
            max_levels.unwrap_or(packed_quad::MAX_LEVELS_POSSIBLE),
        )?),
        _ => Arc::new(S2PrefixTree::new(
            ctx.clone(),
            max_levels.unwrap_or(S2PrefixTree::max_levels_for_arity(1)),
            1,
        )?),
    })
}

/// Java's `StringHelper.startsWith(ref, prefix)` / `BytesRef.compareTo`
/// helpers on raw bytes.
pub(crate) fn compare_bytes(a: &[u8], b: &[u8]) -> i32 {
    for (x, y) in a.iter().zip(b.iter()) {
        let diff = *x as i32 - *y as i32;
        if diff != 0 {
            return diff;
        }
    }
    a.len() as i32 - b.len() as i32
}

#[cfg(test)]
mod tests;
