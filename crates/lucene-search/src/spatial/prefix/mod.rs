//! `org.apache.lucene.spatial.prefix`: the strategies over a
//! [`SpatialPrefixTree`] -- a shape indexes as the terms of the cells
//! covering it (`CellToBytesRefIterator` and `BytesRefIteratorTokenStream`
//! become the field's pre-analyzed tokens) -- their queries ([`query`]) and
//! facets ([`facets`]).

pub mod facets;
pub mod query;

use std::fmt;
use std::sync::Arc;

use lucene_index::document::{
    Field, FieldToken, FieldTokens, FieldType, IndexOptions, IndexableField,
};
use lucene_util::spatial4j::{Point, Shape, SpatialContext};
use lucene_util::spatial_extras::prefix_tree::{
    Cell, CellIterator, DateRangePrefixTree, NumberRangePrefixTree, SpatialPrefixTree, UnitNRShape,
};
use lucene_util::spatial_extras::query::{SpatialArgs, SpatialOperation};

use self::facets::{AcceptDocs, Facets, Heatmap};
use self::query::{
    ContainsPrefixTreeQuery, IntersectsPrefixTreeQuery, PrefixTreeTermQuery, PrefixTreeTermsQuery,
    WithinPrefixTreeQuery,
};
use super::util::{PointPrefixTreeFieldCacheProvider, ShapeFieldCacheDistanceValueSource};
use super::{check_field_name, Fields, SpatialStrategy};
use crate::document::DocumentQuery;
use crate::multi_segment::OpenSegment;
use crate::values_source::DoubleValuesSource;
use crate::{Error, Result};

/// `PrefixTreeStrategy.FIELD_TYPE`: indexed (`DOCS`), tokenized, not
/// stored, norms omitted.
pub fn field_type() -> FieldType {
    let mut ft = FieldType::new();
    // A fresh type is unfrozen: the setters cannot fail.
    let _ = ft.set_tokenized(true);
    let _ = ft.set_omit_norms(true);
    let _ = ft.set_index_options(IndexOptions::Docs);
    ft.frozen()
}

/// `PrefixTreeStrategy`: the configuration every prefix-tree strategy
/// shares -- the tree, the field, the default precision of non-point shapes
/// and whether only points are indexed.
#[derive(Clone)]
pub struct PrefixTreeStrategy {
    ctx: Arc<SpatialContext>,
    field_name: String,
    grid: Arc<dyn SpatialPrefixTree>,
    default_field_values_array_len: usize,
    dist_err_pct: f64,
    points_only: bool,
}

impl fmt::Debug for PrefixTreeStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PrefixTreeStrategy({}, {})", self.field_name, self.grid)
    }
}

impl PrefixTreeStrategy {
    /// `new PrefixTreeStrategy(grid, fieldName)`.
    ///
    /// # Errors
    /// An empty field name.
    pub fn new(grid: Arc<dyn SpatialPrefixTree>, field_name: &str) -> Result<Self> {
        check_field_name(field_name)?;
        Ok(PrefixTreeStrategy {
            ctx: grid.spatial_context().clone(),
            field_name: field_name.to_string(),
            grid,
            default_field_values_array_len: 2,
            dist_err_pct: SpatialArgs::DEFAULT_DISTERRPCT,
            points_only: false,
        })
    }

    /// `getGrid()`.
    pub fn grid(&self) -> &Arc<dyn SpatialPrefixTree> {
        &self.grid
    }

    /// `getFieldName()`.
    pub fn field_name(&self) -> &str {
        &self.field_name
    }

    /// `getSpatialContext()`.
    pub fn spatial_context(&self) -> &Arc<SpatialContext> {
        &self.ctx
    }

    /// `setDefaultFieldValuesArrayLen(len)`: a memory hint for the distance
    /// source's per-document point lists.
    pub fn set_default_field_values_array_len(&mut self, len: usize) {
        self.default_field_values_array_len = len;
    }

    /// `getDistErrPct()`.
    pub fn dist_err_pct(&self) -> f64 {
        self.dist_err_pct
    }

    /// `setDistErrPct(distErrPct)`: the default precision of non-point
    /// shapes, at index and query time.
    pub fn set_dist_err_pct(&mut self, dist_err_pct: f64) {
        self.dist_err_pct = dist_err_pct;
    }

    /// `isPointsOnly()`.
    pub fn is_points_only(&self) -> bool {
        self.points_only
    }

    /// `setPointsOnly(pointsOnly)`: only points are indexed, so there are no
    /// leaves but at the last level.
    pub fn set_points_only(&mut self, points_only: bool) {
        self.points_only = points_only;
    }

    /// The field of a shape's cells: their terms as a pre-analyzed token
    /// stream (`BytesRefIteratorTokenStream` -- each token at increment 1,
    /// offsets 0; `end()` leaves both at 0).
    fn field_of(&self, mut cells: Box<dyn CellIterator>, with_leaf: bool) -> Result<Fields> {
        let mut tokens = Vec::new();
        while cells.has_next()? {
            let cell = cells.next()?;
            let term = if with_leaf {
                cell.token_bytes_with_leaf()
            } else {
                cell.token_bytes_no_leaf()
            };
            tokens.push(FieldToken::new(term, 0, 0));
        }
        let field = Field::from_token_stream(
            self.field_name.clone(),
            FieldTokens {
                tokens,
                final_position_increment: 0,
                final_offset: 0,
                end_attributes: None,
            },
            field_type(),
        )
        .map_err(|e| Error::IllegalArgument(e.to_string()))?;
        let field: Box<dyn IndexableField> = Box::new(field);
        Ok(vec![field])
    }

    /// `createCellIteratorToIndex(shape, detailLevel, reuse)`'s base:
    /// the tree's cells, refusing a non-point on a points-only field.
    fn default_cells(
        &self,
        shape: &Arc<dyn Shape>,
        is_point: bool,
        detail_level: i32,
    ) -> Result<Box<dyn CellIterator>> {
        if self.points_only && !is_point {
            return Err(Error::IllegalArgument(format!(
                "pointsOnly is true yet a {} is given for indexing",
                super::shape_class(&**shape)
            )));
        }
        Ok(self.grid.tree_cell_iterator(shape, detail_level)?)
    }

    /// `makeDistanceValueSource(queryPoint, multiplier)`: the distance to a
    /// document's nearest indexed point, read from the field's terms at the
    /// last level (a points-only field).
    pub fn make_distance_value_source(
        &self,
        query_point: &Arc<dyn Point>,
        multiplier: f64,
    ) -> Arc<dyn DoubleValuesSource> {
        let provider = PointPrefixTreeFieldCacheProvider::new(
            self.grid.clone(),
            &self.field_name,
            self.default_field_values_array_len,
        );
        Arc::new(ShapeFieldCacheDistanceValueSource::new(
            self.ctx.clone(),
            Arc::new(provider),
            query_point.clone(),
            multiplier,
        ))
    }

    /// `calcFacets(context, topAcceptDocs, inputShape, facetLevel,
    /// maxCells)`: a heatmap ([`facets::calc_heatmap`]).
    ///
    /// # Errors
    /// As [`facets::calc_heatmap`].
    pub fn calc_facets(
        &self,
        leaves: &[OpenSegment<'_>],
        top_accept_docs: AcceptDocs<'_>,
        input_shape: Option<&Arc<dyn Shape>>,
        facet_level: i32,
        max_cells: i32,
    ) -> Result<Heatmap> {
        facets::calc_heatmap(
            self,
            leaves,
            top_accept_docs,
            input_shape,
            facet_level,
            max_cells,
        )
    }
}

/// `RecursivePrefixTreeStrategy`: any shape, queried by walking the indexed
/// cells against the query shape's (intersects, within, contains).
#[derive(Clone, Debug)]
pub struct RecursivePrefixTreeStrategy {
    base: PrefixTreeStrategy,
    prefix_grid_scan_level: i32,
    prune_leafy_branches: bool,
    multi_overlapping_indexed_shapes: bool,
    /// `NumberRangePrefixTreeStrategy`'s overrides: its point and
    /// grid-aligned shapes are units.
    number_range: bool,
    class_name: &'static str,
}

impl RecursivePrefixTreeStrategy {
    /// `new RecursivePrefixTreeStrategy(grid, fieldName)`: scan the last
    /// four levels, prune leafy branches.
    ///
    /// # Errors
    /// An empty field name.
    pub fn new(grid: Arc<dyn SpatialPrefixTree>, field_name: &str) -> Result<Self> {
        let max_levels = grid.max_levels();
        Ok(RecursivePrefixTreeStrategy {
            base: PrefixTreeStrategy::new(grid, field_name)?,
            prefix_grid_scan_level: max_levels - 4,
            prune_leafy_branches: true,
            multi_overlapping_indexed_shapes: true,
            number_range: false,
            class_name: "RecursivePrefixTreeStrategy",
        })
    }

    /// The shared configuration.
    pub fn base(&self) -> &PrefixTreeStrategy {
        &self.base
    }

    /// The shared configuration, to change.
    pub fn base_mut(&mut self) -> &mut PrefixTreeStrategy {
        &mut self.base
    }

    /// `getPrefixGridScanLevel()`.
    pub fn prefix_grid_scan_level(&self) -> i32 {
        self.prefix_grid_scan_level
    }

    /// `setPrefixGridScanLevel(level)`: the level from which indexed terms
    /// are scanned rather than sought.
    pub fn set_prefix_grid_scan_level(&mut self, level: i32) {
        self.prefix_grid_scan_level = level;
    }

    /// `isMultiOverlappingIndexedShapes()`.
    pub fn is_multi_overlapping_indexed_shapes(&self) -> bool {
        self.multi_overlapping_indexed_shapes
    }

    /// `setMultiOverlappingIndexedShapes(v)`: see
    /// [`ContainsPrefixTreeQuery::multi_overlapping_indexed_shapes`].
    pub fn set_multi_overlapping_indexed_shapes(&mut self, v: bool) {
        self.multi_overlapping_indexed_shapes = v;
    }

    /// `isPruneLeafyBranches()`.
    pub fn is_prune_leafy_branches(&self) -> bool {
        self.prune_leafy_branches
    }

    /// `setPruneLeafyBranches(v)`: a full set of sibling leaves indexes as
    /// their parent (for cells that can prune).
    pub fn set_prune_leafy_branches(&mut self, v: bool) {
        self.prune_leafy_branches = v;
    }

    /// `isPointShape(shape)`.
    fn is_point_shape(&self, shape: &dyn Shape) -> bool {
        if self.number_range {
            return shape
                .as_any()
                .downcast_ref::<UnitNRShape>()
                .is_some_and(|u| u.level() == self.base.grid.max_levels());
        }
        shape.as_point().is_some()
    }

    /// `isGridAlignedShape(shape)`: one cell -- a point, or a unit other
    /// than the world.
    fn is_grid_aligned_shape(&self, shape: &dyn Shape) -> bool {
        if self.number_range {
            return shape
                .as_any()
                .downcast_ref::<UnitNRShape>()
                .is_some_and(|u| u.level() > 0);
        }
        self.is_point_shape(shape)
    }

    /// `createIndexableFields(shape, distErr)`.
    ///
    /// # Errors
    /// As [`Self::create_indexable_fields_at_level`].
    pub fn create_indexable_fields_with_dist_err(
        &self,
        shape: &Arc<dyn Shape>,
        dist_err: f64,
    ) -> Result<Fields> {
        let level = self.base.grid.level_for_distance(dist_err);
        self.create_indexable_fields_at_level(shape, level)
    }

    /// `createIndexableFields(shape, detailLevel)`.
    ///
    /// # Errors
    /// A non-point on a points-only field, or the tree's error.
    pub fn create_indexable_fields_at_level(
        &self,
        shape: &Arc<dyn Shape>,
        detail_level: i32,
    ) -> Result<Fields> {
        let cells = self.cells_to_index(shape, detail_level)?;
        self.base.field_of(cells, true)
    }

    /// `createCellIteratorToIndex(shape, detailLevel, reuse)`: the tree's
    /// cells, or, pruning leafy branches of a non-point,
    /// `recursiveTraverseAndPrune`'s.
    fn cells_to_index(
        &self,
        shape: &Arc<dyn Shape>,
        detail_level: i32,
    ) -> Result<Box<dyn CellIterator>> {
        if !self.prune_leafy_branches || self.is_grid_aligned_shape(&**shape) {
            return self
                .base
                .default_cells(shape, self.is_point_shape(&**shape), detail_level);
        }
        let mut cells = Vec::with_capacity(4096);
        let world = self.base.grid.world_cell();
        recursive_traverse_and_prune(world, shape, detail_level, &mut cells)?;
        Ok(Box::new(VecCellIterator::new(cells)))
    }

    /// `makeGridShapeIntersectsQuery(gridShape)`: a points-only field's
    /// single term, else an intersects query scanning only the last level.
    fn make_grid_shape_intersects_query(
        &self,
        grid_shape: &Arc<dyn Shape>,
    ) -> Result<Box<dyn DocumentQuery>> {
        let grid = &self.base.grid;
        if self.base.points_only {
            // equivalent to a TermQuery: the last cell
            let mut it = grid.tree_cell_iterator(grid_shape, grid.max_levels())?;
            let mut cell = it.next()?;
            while it.has_next()? {
                cell = it.next()?;
            }
            return Ok(Box::new(PrefixTreeTermQuery::new(
                &self.base.field_name,
                cell.token_bytes_with_leaf(),
            )));
        }
        // There could be parent cells; reduce the scan level instead.
        Ok(Box::new(IntersectsPrefixTreeQuery::new(
            grid_shape.clone(),
            &self.base.field_name,
            grid.clone(),
            grid.max_levels(),
            grid.max_levels() + 1,
        )))
    }
}

/// `recursiveTraverseAndPrune(cell, shape, detailLevel, result)`: whether
/// `cell` was added as a leaf; otherwise it descends.
fn recursive_traverse_and_prune(
    mut cell: Box<dyn Cell>,
    shape: &Arc<dyn Shape>,
    detail_level: i32,
    result: &mut Vec<Box<dyn Cell>>,
) -> Result<bool> {
    if cell.level() == detail_level {
        cell.set_leaf(); // might already be a leaf
    }
    if cell.is_leaf() {
        result.push(cell);
        return Ok(true);
    }
    // The children are read from `cell` before it is (maybe) added.
    let mut sub_cells = cell.next_level_cells(Some(shape))?;
    let sub_cells_size = cell.sub_cells_size();
    let level = cell.level();
    let added = level != 0;
    if added {
        result.push(cell);
    }
    let mut leaves = 0i32;
    while sub_cells.has_next()? {
        let sub_cell = sub_cells.next()?;
        if recursive_traverse_and_prune(sub_cell, shape, detail_level, result)? {
            leaves += 1;
        }
    }
    // Cannot prune a cell that is not a `CellCanPrune`
    let Some(size) = sub_cells_size else {
        return Ok(false);
    };
    // can we prune?
    if leaves == size && level != 0 {
        // the parent as a leaf instead of all its children: remove the
        // leaves, then mark the cell (now last) a leaf
        let keep = result.len() - usize::try_from(leaves).unwrap_or(0);
        result.truncate(keep);
        result
            .last_mut()
            .expect("the parent precedes its children")
            .set_leaf();
        return Ok(true);
    }
    Ok(false)
}

/// The cells of a pruned traversal, as an iterator.
struct VecCellIterator {
    cells: std::vec::IntoIter<Box<dyn Cell>>,
    this_cell: Option<Box<dyn Cell>>,
}

impl VecCellIterator {
    fn new(cells: Vec<Box<dyn Cell>>) -> Self {
        VecCellIterator {
            cells: cells.into_iter(),
            this_cell: None,
        }
    }
}

impl CellIterator for VecCellIterator {
    fn has_next(&mut self) -> lucene_util::spatial4j::Result<bool> {
        Ok(self.cells.len() > 0)
    }

    fn next(&mut self) -> lucene_util::spatial4j::Result<Box<dyn Cell>> {
        let cell = self.cells.next().ok_or_else(|| {
            lucene_util::spatial4j::Error::Runtime("java.util.NoSuchElementException".into())
        })?;
        self.this_cell = Some(cell.clone_box());
        Ok(cell)
    }

    fn this_cell(&self) -> Option<&dyn Cell> {
        self.this_cell.as_deref()
    }
}

impl fmt::Display for RecursivePrefixTreeStrategy {
    /// `toString()`: the tree and the settings off their defaults.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut s = format!("{}(SPG:({})", self.class_name, self.base.grid);
        if self.base.points_only {
            s.push_str(",pointsOnly");
        }
        if self.prune_leafy_branches {
            s.push_str(",pruneLeafyBranches");
        }
        if self.prefix_grid_scan_level != self.base.grid.max_levels() - 4 {
            s.push_str(&format!(
                ",prefixGridScanLevel:{}",
                self.prefix_grid_scan_level
            ));
        }
        if !self.multi_overlapping_indexed_shapes {
            s.push_str(",!multiOverlappingIndexedShapes");
        }
        s.push(')');
        f.write_str(&s)
    }
}

impl SpatialStrategy for RecursivePrefixTreeStrategy {
    fn spatial_context(&self) -> &Arc<SpatialContext> {
        &self.base.ctx
    }

    fn field_name(&self) -> &str {
        &self.base.field_name
    }

    fn create_indexable_fields(&self, shape: &Arc<dyn Shape>) -> Result<Fields> {
        let dist_err = SpatialArgs::calc_distance_from_err_pct(
            &**shape,
            self.base.dist_err_pct,
            &self.base.ctx,
        )?;
        self.create_indexable_fields_with_dist_err(shape, dist_err)
    }

    fn make_distance_value_source(
        &self,
        query_point: &Arc<dyn Point>,
        multiplier: f64,
    ) -> Result<Arc<dyn DoubleValuesSource>> {
        if self.number_range {
            return Err(Error::Spatial(
                lucene_util::spatial4j::Error::UnsupportedOperation(None),
            ));
        }
        Ok(self
            .base
            .make_distance_value_source(query_point, multiplier))
    }

    fn make_query(&self, args: &SpatialArgs) -> Result<Box<dyn DocumentQuery>> {
        let op = args.operation;
        let shape = &args.shape;
        let grid = &self.base.grid;
        let detail_level =
            grid.level_for_distance(args.resolve_dist_err(&self.base.ctx, self.base.dist_err_pct)?);
        match op {
            SpatialOperation::Intersects => {
                if self.is_grid_aligned_shape(&**shape) {
                    return self.make_grid_shape_intersects_query(shape);
                }
                Ok(Box::new(IntersectsPrefixTreeQuery::new(
                    shape.clone(),
                    &self.base.field_name,
                    grid.clone(),
                    detail_level,
                    self.prefix_grid_scan_level,
                )))
            }
            SpatialOperation::IsWithin => Ok(Box::new(WithinPrefixTreeQuery::new(
                shape.clone(),
                &self.base.field_name,
                grid.clone(),
                detail_level,
                self.prefix_grid_scan_level,
                -1.0, // slower, but ensures correct results
            )?)),
            SpatialOperation::Contains => Ok(Box::new(ContainsPrefixTreeQuery::new(
                shape.clone(),
                &self.base.field_name,
                grid.clone(),
                detail_level,
                self.multi_overlapping_indexed_shapes,
            ))),
            _ => Err(Error::Spatial(op.unsupported())),
        }
    }

    fn as_prefix_tree(&self) -> Option<&PrefixTreeStrategy> {
        Some(&self.base)
    }
}

/// `TermQueryPrefixTreeStrategy`: a `TermInSetQuery` of the query shape's
/// leaf cells; indexed cells carry no leaf marker. For indexed points.
#[derive(Clone, Debug)]
pub struct TermQueryPrefixTreeStrategy {
    base: PrefixTreeStrategy,
}

impl TermQueryPrefixTreeStrategy {
    /// `new TermQueryPrefixTreeStrategy(grid, fieldName)`.
    ///
    /// # Errors
    /// An empty field name.
    pub fn new(grid: Arc<dyn SpatialPrefixTree>, field_name: &str) -> Result<Self> {
        Ok(TermQueryPrefixTreeStrategy {
            base: PrefixTreeStrategy::new(grid, field_name)?,
        })
    }

    /// The shared configuration.
    pub fn base(&self) -> &PrefixTreeStrategy {
        &self.base
    }

    /// The shared configuration, to change.
    pub fn base_mut(&mut self) -> &mut PrefixTreeStrategy {
        &mut self.base
    }

    /// `createIndexableFields(shape, detailLevel)`: the cells without leaf
    /// markers.
    ///
    /// # Errors
    /// A non-point on a points-only field, or the tree's error.
    pub fn create_indexable_fields_at_level(
        &self,
        shape: &Arc<dyn Shape>,
        detail_level: i32,
    ) -> Result<Fields> {
        let cells = self
            .base
            .default_cells(shape, shape.as_point().is_some(), detail_level)?;
        self.base.field_of(cells, false)
    }
}

impl fmt::Display for TermQueryPrefixTreeStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&super::strategy_string(
            "TermQueryPrefixTreeStrategy",
            &self.base.field_name,
            &self.base.ctx,
        ))
    }
}

impl SpatialStrategy for TermQueryPrefixTreeStrategy {
    fn spatial_context(&self) -> &Arc<SpatialContext> {
        &self.base.ctx
    }

    fn field_name(&self) -> &str {
        &self.base.field_name
    }

    fn create_indexable_fields(&self, shape: &Arc<dyn Shape>) -> Result<Fields> {
        let dist_err = SpatialArgs::calc_distance_from_err_pct(
            &**shape,
            self.base.dist_err_pct,
            &self.base.ctx,
        )?;
        let level = self.base.grid.level_for_distance(dist_err);
        self.create_indexable_fields_at_level(shape, level)
    }

    fn make_distance_value_source(
        &self,
        query_point: &Arc<dyn Point>,
        multiplier: f64,
    ) -> Result<Arc<dyn DoubleValuesSource>> {
        Ok(self
            .base
            .make_distance_value_source(query_point, multiplier))
    }

    fn make_query(&self, args: &SpatialArgs) -> Result<Box<dyn DocumentQuery>> {
        let op = args.operation;
        if op != SpatialOperation::Intersects {
            return Err(Error::Spatial(op.unsupported()));
        }
        let shape = &args.shape;
        let grid = &self.base.grid;
        let detail_level =
            grid.level_for_distance(args.resolve_dist_err(&self.base.ctx, self.base.dist_err_pct)?);
        // the leaf cells' terms (no parents, no leaf bytes)
        let mut terms = Vec::new();
        let mut cells = grid.tree_cell_iterator(shape, detail_level)?;
        while cells.has_next()? {
            let cell = cells.next()?;
            if !cell.is_leaf() {
                continue;
            }
            terms.push(cell.token_bytes_no_leaf());
        }
        terms.sort_unstable();
        terms.dedup();
        Ok(Box::new(PrefixTreeTermsQuery {
            field: self.base.field_name.clone(),
            terms,
        }))
    }

    fn as_prefix_tree(&self) -> Option<&PrefixTreeStrategy> {
        Some(&self.base)
    }
}

/// `NumberRangePrefixTreeStrategy`: number or date ranges indexed in a
/// [`NumberRangePrefixTree`] -- an RPT that does not prune, scans the last
/// two levels, indexes ranges as well as instants, and is exact
/// (`distErrPct` 0).
#[derive(Clone, Debug)]
pub struct NumberRangePrefixTreeStrategy {
    rpt: RecursivePrefixTreeStrategy,
    tree: NumberRangePrefixTree,
}

impl NumberRangePrefixTreeStrategy {
    /// `new NumberRangePrefixTreeStrategy(prefixTree, fieldName)`: `grid`
    /// is the tree (e.g. a [`DateRangePrefixTree`]), `tree` its
    /// number-range handle.
    ///
    /// # Errors
    /// An empty field name.
    pub fn new(
        grid: Arc<dyn SpatialPrefixTree>,
        tree: NumberRangePrefixTree,
        field_name: &str,
    ) -> Result<Self> {
        let mut rpt = RecursivePrefixTreeStrategy::new(grid, field_name)?;
        rpt.number_range = true;
        rpt.class_name = "NumberRangePrefixTreeStrategy";
        rpt.set_prune_leafy_branches(false);
        let max_levels = rpt.base.grid.max_levels();
        rpt.set_prefix_grid_scan_level(max_levels - 2); // user might want to change
        rpt.base.set_points_only(false);
        rpt.base.set_dist_err_pct(0.0);
        Ok(NumberRangePrefixTreeStrategy { rpt, tree })
    }

    /// The strategy over a [`DateRangePrefixTree`].
    ///
    /// # Errors
    /// An empty field name.
    pub fn for_dates(tree: Arc<DateRangePrefixTree>, field_name: &str) -> Result<Self> {
        let nr = tree.number_range_tree().clone();
        Self::new(tree, nr, field_name)
    }

    /// The recursive strategy it is.
    pub fn rpt(&self) -> &RecursivePrefixTreeStrategy {
        &self.rpt
    }

    /// The recursive strategy it is, to change.
    pub fn rpt_mut(&mut self) -> &mut RecursivePrefixTreeStrategy {
        &mut self.rpt
    }

    /// `getGrid()`: the number-range tree.
    pub fn number_range_tree(&self) -> &NumberRangePrefixTree {
        &self.tree
    }

    /// `calcFacets(context, topAcceptDocs, start, end)`: facets between two
    /// units, one level below the deeper of them.
    ///
    /// # Errors
    /// As [`Self::calc_facets`], or a range the tree refuses.
    pub fn calc_facets_between(
        &self,
        leaves: &[OpenSegment<'_>],
        top_accept_docs: AcceptDocs<'_>,
        start: &UnitNRShape,
        end: &UnitNRShape,
    ) -> Result<Facets> {
        let facet_range = self.tree.to_range_shape(start, end)?.into_shape();
        let detail_level = start.level().max(end.level()) + 1;
        self.calc_facets(leaves, top_accept_docs, &facet_range, detail_level)
    }

    /// `calcFacets(context, topAcceptDocs, facetRange, level)`: the counts
    /// at `level` within the range, grouped by parent.
    ///
    /// # Errors
    /// A decode error, or Java's exceptions for a world parent.
    pub fn calc_facets(
        &self,
        leaves: &[OpenSegment<'_>],
        top_accept_docs: AcceptDocs<'_>,
        facet_range: &Arc<dyn Shape>,
        level: i32,
    ) -> Result<Facets> {
        facets::calc_number_range_facets(
            &self.rpt.base,
            leaves,
            top_accept_docs,
            facet_range,
            level,
        )
    }
}

impl fmt::Display for NumberRangePrefixTreeStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.rpt.fmt(f)
    }
}

impl SpatialStrategy for NumberRangePrefixTreeStrategy {
    fn spatial_context(&self) -> &Arc<SpatialContext> {
        self.rpt.spatial_context()
    }

    fn field_name(&self) -> &str {
        self.rpt.field_name()
    }

    fn create_indexable_fields(&self, shape: &Arc<dyn Shape>) -> Result<Fields> {
        self.rpt.create_indexable_fields(shape)
    }

    /// Unsupported.
    fn make_distance_value_source(
        &self,
        query_point: &Arc<dyn Point>,
        multiplier: f64,
    ) -> Result<Arc<dyn DoubleValuesSource>> {
        self.rpt.make_distance_value_source(query_point, multiplier)
    }

    fn make_query(&self, args: &SpatialArgs) -> Result<Box<dyn DocumentQuery>> {
        self.rpt.make_query(args)
    }

    fn as_prefix_tree(&self) -> Option<&PrefixTreeStrategy> {
        Some(&self.rpt.base)
    }
}
