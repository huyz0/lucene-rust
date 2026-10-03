//! `PrefixTreeFacetCounter`, `HeatmapFacetCounter` and
//! `NumberRangePrefixTreeStrategy`'s facets: counts of documents per cell,
//! from the visiting traversal of [`super::query`].

use std::collections::BTreeMap;
use std::sync::Arc;

use lucene_util::fixed_bit_set::FixedBitSet;
use lucene_util::spatial4j::{Rectangle, Shape, SpatialRelation};
use lucene_util::spatial_extras::prefix_tree::{Cell, UnitNRShape};

use super::query::{visit, Traverser, VisitingQuery, Visitor};
use super::PrefixTreeStrategy;
use crate::multi_segment::OpenSegment;
use crate::{Error, Result};

/// `PrefixTreeFacetCounter.FacetVisitor`.
pub trait FacetVisitor {
    /// `startOfSegment()`: a segment with indexed data starts.
    fn start_of_segment(&mut self) {}

    /// `visit(cell, count)`: a leaf, or a cell at the facet level, with
    /// `count > 0` documents.
    ///
    /// # Errors
    /// The visitor's own (a shape of the wrong kind, Java's casts).
    fn visit(&mut self, cell: &dyn Cell, count: i32) -> Result<()>;
}

/// The documents a count accepts: Java's `topAcceptDocs` (global doc ids)
/// or, when `None`, each segment's live documents.
pub type AcceptDocs<'a> = Option<&'a FixedBitSet>;

/// `PrefixTreeFacetCounter.compute(strategy, context, topAcceptDocs,
/// queryShape, facetLevel, facetVisitor)`: every segment of `leaves`.
///
/// # Errors
/// A term or postings decode error, or the visitor's.
pub fn compute(
    strategy: &PrefixTreeStrategy,
    leaves: &[OpenSegment<'_>],
    top_accept_docs: AcceptDocs<'_>,
    query_shape: &Arc<dyn Shape>,
    facet_level: i32,
    facet_visitor: &mut dyn FacetVisitor,
) -> Result<()> {
    for leaf in leaves {
        // determine leaf acceptDocs
        let accept: LeafAccept<'_> = match top_accept_docs {
            None => LeafAccept::Live(leaf.live_docs),
            Some(top) => LeafAccept::Top(top, leaf.doc_base),
        };
        compute_leaf(
            strategy,
            leaf,
            &accept,
            query_shape,
            facet_level,
            facet_visitor,
        )?;
    }
    Ok(())
}

/// A segment's accepted documents.
enum LeafAccept<'a> {
    /// The live documents (`None`: all of them).
    Live(Option<&'a FixedBitSet>),
    /// The global `topAcceptDocs`, offset by the segment's doc base.
    Top(&'a FixedBitSet, i32),
}

impl LeafAccept<'_> {
    /// Whether `acceptDocs == null` (every document, so `docFreq` counts).
    fn all(&self) -> bool {
        matches!(self, LeafAccept::Live(None))
    }

    fn get(&self, doc: i32) -> bool {
        match self {
            LeafAccept::Live(bits) => bits.is_none_or(|b| b.get_doc(doc)),
            LeafAccept::Top(bits, base) => bits.get_doc(base.saturating_add(doc)),
        }
    }
}

/// The per-segment `compute(strategy, context, acceptDocs, ...)`.
fn compute_leaf(
    strategy: &PrefixTreeStrategy,
    leaf: &OpenSegment<'_>,
    accept: &LeafAccept<'_>,
    query_shape: &Arc<dyn Shape>,
    facet_level: i32,
    facet_visitor: &mut dyn FacetVisitor,
) -> Result<()> {
    let tree = strategy.grid();
    // scanLevel is an optimization knob; the deepest one.
    let scan_level = tree.max_levels();
    let q = VisitingQuery::new(
        query_shape.clone(),
        strategy.field_name(),
        tree.clone(),
        facet_level,
        scan_level,
    );
    let mut t = Traverser::new(leaf, strategy.field_name(), &**tree)?;
    let mut v = CountingVisitor {
        accept,
        facet_level,
        facet_visitor,
    };
    visit(&q, &mut t, &mut v)?;
    t.check(strategy.field_name())
}

struct CountingVisitor<'v, 'a> {
    accept: &'v LeafAccept<'a>,
    facet_level: i32,
    facet_visitor: &'v mut dyn FacetVisitor,
}

impl CountingVisitor<'_, '_> {
    /// `countDocsAtThisTerm()`.
    fn count_docs_at_this_term(&self, t: &mut Traverser<'_>) -> Result<i32> {
        if self.accept.all() {
            return t.doc_freq();
        }
        let mut count = 0i32;
        t.for_each_doc(|doc| {
            if self.accept.get(doc) {
                count = count.saturating_add(1);
            }
            true
        })?;
        Ok(count)
    }

    /// `hasDocsAtThisTerm()`.
    fn has_docs_at_this_term(&self, t: &mut Traverser<'_>) -> Result<bool> {
        if self.accept.all() {
            return Ok(true);
        }
        let mut any = false;
        t.for_each_doc(|doc| {
            any = self.accept.get(doc);
            !any
        })?;
        Ok(any)
    }
}

impl Visitor for CountingVisitor<'_, '_> {
    fn start(&mut self, _t: &mut Traverser<'_>) -> Result<()> {
        self.facet_visitor.start_of_segment();
        Ok(())
    }

    fn visit_prefix(
        &mut self,
        q: &VisitingQuery,
        t: &mut Traverser<'_>,
        cell: &dyn Cell,
    ) -> Result<bool> {
        // At facetLevel...
        if cell.level() == self.facet_level {
            // Count docs: not a leaf, but treated as one at the facet level
            self.visit_leaf(q, t, cell)?;
            return Ok(false);
        }
        // Short-circuits a discriminating filter near the facet level, or
        // where it is cheap (docFreq 1).
        if (cell.level() == self.facet_level - 1 || t.doc_freq()? == 1)
            && !self.has_docs_at_this_term(t)?
        {
            return Ok(false);
        }
        Ok(true)
    }

    fn visit_leaf(
        &mut self,
        _q: &VisitingQuery,
        t: &mut Traverser<'_>,
        cell: &dyn Cell,
    ) -> Result<()> {
        let count = self.count_docs_at_this_term(t)?;
        if count > 0 {
            self.facet_visitor.visit(cell, count)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// HeatmapFacetCounter
// ---------------------------------------------------------------------------

/// `HeatmapFacetCounter.MAX_ROWS_OR_COLUMNS`: `(int)
/// Math.sqrt(ArrayUtil.MAX_ARRAY_LENGTH)`.
pub const MAX_ROWS_OR_COLUMNS: i32 = 46_340;

/// `HeatmapFacetCounter.Heatmap`: counts in a grid of `columns x rows`,
/// column by column.
#[derive(Debug, Clone)]
pub struct Heatmap {
    pub columns: i32,
    pub rows: i32,
    /// First column (all rows), then the second, ...
    pub counts: Vec<i32>,
    pub region: Arc<dyn Rectangle>,
}

impl Heatmap {
    /// `getCount(x, y)`.
    pub fn get_count(&self, x: i32, y: i32) -> i32 {
        let i = x.saturating_mul(self.rows).saturating_add(y);
        usize::try_from(i)
            .ok()
            .and_then(|i| self.counts.get(i))
            .copied()
            .unwrap_or(0)
    }
}

impl std::fmt::Display for Heatmap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Heatmap{{{}x{} {}}}",
            self.columns, self.rows, self.region
        )
    }
}

fn rect_of(cell: &dyn Cell) -> Result<Arc<dyn Shape>> {
    let shape = cell.shape()?;
    if shape.as_rectangle().is_none() {
        return Err(crate::spatial::class_cast(
            &*shape,
            "org.locationtech.spatial4j.shape.Rectangle",
        ));
    }
    Ok(shape)
}

/// `HeatmapFacetCounter.calcFacets(strategy, context, topAcceptDocs,
/// inputShape, facetLevel, maxCells)`: the counts of the cells at
/// `facet_level` over the input shape's bounding box (the world for
/// `None`), each ancestor cell's count added to the cells beneath it.
///
/// # Errors
/// Too many cells, a tree whose cells are not rectangles (Java's
/// `ClassCastException`), or a decode error.
pub fn calc_heatmap(
    strategy: &PrefixTreeStrategy,
    leaves: &[OpenSegment<'_>],
    top_accept_docs: AcceptDocs<'_>,
    input_shape: Option<&Arc<dyn Shape>>,
    facet_level: i32,
    max_cells: i32,
) -> Result<Heatmap> {
    if i64::from(max_cells) > i64::from(MAX_ROWS_OR_COLUMNS) * i64::from(MAX_ROWS_OR_COLUMNS) {
        return Err(Error::IllegalArgument(format!(
            "maxCells ({max_cells}) should be <= {MAX_ROWS_OR_COLUMNS}"
        )));
    }
    let grid = strategy.grid();
    let ctx = grid.spatial_context().clone();
    let input_shape: Arc<dyn Shape> = match input_shape {
        Some(s) => s.clone(),
        None => Arc::new(ctx.world_bounds()),
    };
    let input_rect = input_shape.bounding_box()?;
    // First get the rect of the cell at the bottom-left at depth facetLevel
    let corner_pt: Arc<dyn Shape> = ctx.point_xy(input_rect.min_x(), input_rect.min_y())?;
    let mut cell_iterator = grid.tree_cell_iterator(&corner_pt, facet_level)?;
    let mut corner_cell = None;
    while cell_iterator.has_next()? {
        corner_cell = Some(cell_iterator.next()?);
    }
    let corner_cell = corner_cell
        .ok_or_else(|| Error::IllegalState(format!("Cell not at target level: {facet_level}")))?;
    let corner_shape = rect_of(&*corner_cell)?;
    let corner_rect = corner_shape.as_rectangle().expect("checked");
    // Now calculate the number of columns and rows necessary to cover the
    // inputRect
    let mut heat_min_x = corner_rect.min_x(); // note: might change below
    let cell_width = corner_rect.width();
    let [w_min_x, w_max_x, w_min_y, w_max_y] = ctx.world_bounds_values();
    let world_width = w_max_x - w_min_x;
    let world_height = w_max_y - w_min_y;
    let columns = calc_rows_or_cols(
        cell_width,
        heat_min_x,
        input_rect.width(),
        input_rect.min_x(),
        world_width,
    );
    let heat_min_y = corner_rect.min_y();
    let cell_height = corner_rect.height();
    let rows = calc_rows_or_cols(
        cell_height,
        heat_min_y,
        input_rect.height(),
        input_rect.min_y(),
        world_height,
    );
    if columns > MAX_ROWS_OR_COLUMNS
        || rows > MAX_ROWS_OR_COLUMNS
        || i64::from(columns) * i64::from(rows) > i64::from(max_cells)
    {
        return Err(Error::IllegalArgument(format!(
            "Too many cells ({columns} x {rows}) for level {facet_level} shape {input_rect}"
        )));
    }

    // Create resulting heatmap bounding rectangle & Heatmap object.
    let half_cell_width = cell_width / 2.0;
    // if X world-wraps, use world bounds' range
    if f64::from(columns) * cell_width + half_cell_width > world_width {
        heat_min_x = w_min_x;
    }
    let mut heat_max_x = heat_min_x + f64::from(columns) * cell_width;
    if (heat_max_x - w_max_x).abs() < half_cell_width {
        // numeric conditioning issue
        heat_max_x = w_max_x;
    } else if heat_max_x > w_max_x {
        // wraps dateline (won't happen if !geo)
        heat_max_x = heat_max_x - w_max_x + w_min_x;
    }
    let half_cell_height = cell_height / 2.0;
    let mut heat_max_y = heat_min_y + f64::from(rows) * cell_height;
    if (heat_max_y - w_max_y).abs() < half_cell_height {
        // numeric conditioning issue
        heat_max_y = w_max_y;
    }

    let cells = usize::try_from(i64::from(columns) * i64::from(rows)).unwrap_or(0);
    let region = ctx.rect(heat_min_x, heat_max_x, heat_min_y, heat_max_y)?;
    let mut heatmap = Heatmap {
        columns,
        rows,
        counts: vec![0; cells],
        region,
    };

    // All ancestor cell counts (of facetLevel) are captured during facet
    // visiting and applied later.
    let mut counter = HeatmapVisitor {
        heatmap: &mut heatmap,
        facet_level,
        cell_width,
        cell_height,
        heat_min_y,
        all_cells_ancestor_count: 0,
        ancestors: Vec::new(),
    };
    compute(
        strategy,
        leaves,
        top_accept_docs,
        &input_shape,
        facet_level,
        &mut counter,
    )?;
    let all_cells_ancestor_count = counter.all_cells_ancestor_count;
    let ancestors = std::mem::take(&mut counter.ancestors);

    // Apply allCellsAncestorCount
    if all_cells_ancestor_count > 0 {
        for c in &mut heatmap.counts {
            *c = c.wrapping_add(all_cells_ancestor_count);
        }
    }

    // Apply ancestors (each cell's count once per visit; the sums commute,
    // so Java's hash order does not matter)
    let crosses = heatmap.region.crosses_date_line();
    for (rect, count) in ancestors {
        let [r_min_x, r_max_x, r_min_y, r_max_y] = rect;
        let (start_row, end_row) =
            intersect_interval(heat_min_y, heat_max_y, cell_height, rows, r_min_y, r_max_y);
        if !crosses {
            let (start_col, end_col) = intersect_interval(
                heat_min_x, heat_max_x, cell_width, columns, r_min_x, r_max_x,
            );
            increment_range(&mut heatmap, start_col, end_col, start_row, end_row, count);
        } else {
            // the cell rect might intersect 2 disjoint parts of the heatmap,
            // so the left & right separately
            let left_columns = ((180.0 - heat_min_x) / cell_width).round() as i32;
            let right_columns = heatmap.columns.wrapping_sub(left_columns);
            // left half of dateline:
            if r_max_x > heat_min_x {
                let (start_col, end_col) = intersect_interval(
                    heat_min_x,
                    180.0,
                    cell_width,
                    left_columns,
                    r_min_x,
                    r_max_x,
                );
                increment_range(&mut heatmap, start_col, end_col, start_row, end_row, count);
            }
            // right half of dateline
            if r_min_x < heat_max_x {
                let (start_col, end_col) = intersect_interval(
                    -180.0,
                    heat_max_x,
                    cell_width,
                    right_columns,
                    r_min_x,
                    r_max_x,
                );
                increment_range(
                    &mut heatmap,
                    start_col.wrapping_add(left_columns),
                    end_col.wrapping_add(left_columns),
                    start_row,
                    end_row,
                    count,
                );
            }
        }
    }
    Ok(heatmap)
}

struct HeatmapVisitor<'h> {
    heatmap: &'h mut Heatmap,
    facet_level: i32,
    cell_width: f64,
    cell_height: f64,
    heat_min_y: f64,
    all_cells_ancestor_count: i32,
    /// `ancestors`: each ancestor cell's rectangle and count.
    ancestors: Vec<([f64; 4], i32)>,
}

impl FacetVisitor for HeatmapVisitor<'_> {
    fn visit(&mut self, cell: &dyn Cell, count: i32) -> Result<()> {
        let heat_min_x = self.heatmap.region.min_x();
        let shape = rect_of(cell)?;
        let rect = shape.as_rectangle().expect("checked");
        if cell.level() == self.facet_level {
            // heatmap level; count it directly: convert to col & row
            let column = if rect.min_x() >= heat_min_x {
                java_round((rect.min_x() - heat_min_x) / self.cell_width)
            } else {
                // due to dateline wrap
                java_round((rect.min_x() + 360.0 - heat_min_x) / self.cell_width)
            };
            let row = java_round((rect.min_y() - self.heat_min_y) / self.cell_height);
            // the tree may hand out adjacent cells overlapping the seam:
            // skip them
            if column < 0 || column >= self.heatmap.columns || row < 0 || row >= self.heatmap.rows {
                return Ok(());
            }
            let i =
                usize::try_from(i64::from(column) * i64::from(self.heatmap.rows) + i64::from(row))
                    .unwrap_or(usize::MAX);
            if let Some(c) = self.heatmap.counts.get_mut(i) {
                *c = c.wrapping_add(count);
            }
        } else if rect.relate(&*self.heatmap.region)? == SpatialRelation::Contains {
            self.all_cells_ancestor_count = self.all_cells_ancestor_count.wrapping_add(count);
        } else {
            // ancestor
            self.ancestors.push((
                [rect.min_x(), rect.max_x(), rect.min_y(), rect.max_y()],
                count,
            ));
        }
        Ok(())
    }
}

/// `(int) Math.round(d)`: half up, saturating as Java's `long` to `int`
/// narrowing would not -- callers' values are small.
fn java_round(d: f64) -> i32 {
    (d + 0.5).floor() as i64 as i32
}

/// `intersectInterval(...)`: the cells of a heatmap axis an ancestor's
/// range covers (known to intersect).
pub(crate) fn intersect_interval(
    heat_min: f64,
    heat_max: f64,
    heat_cell_len: f64,
    num_cells: i32,
    cell_min: f64,
    cell_max: f64,
) -> (i32, i32) {
    let start = if heat_min >= cell_min {
        0
    } else {
        java_round((cell_min - heat_min) / heat_cell_len)
    };
    let end = if heat_max <= cell_max {
        num_cells.wrapping_sub(1)
    } else {
        java_round((cell_max - heat_min) / heat_cell_len).wrapping_sub(1)
    };
    (start, end)
}

/// `incrementRange(...)`: `count` added to a block of the heatmap, its
/// start clamped (the end moved with it, as Java's) and its end too.
pub(crate) fn increment_range(
    heatmap: &mut Heatmap,
    mut start_column: i32,
    mut end_column: i32,
    mut start_row: i32,
    mut end_row: i32,
    count: i32,
) {
    if start_column < 0 {
        end_column = end_column.wrapping_add(start_column);
        start_column = 0;
    }
    end_column = end_column.min(heatmap.columns - 1);
    if start_row < 0 {
        end_row = end_row.wrapping_add(start_row);
        start_row = 0;
    }
    end_row = end_row.min(heatmap.rows - 1);
    if start_row > end_row {
        return; // short-circuit
    }
    for c in start_column..=end_column {
        let base = i64::from(c) * i64::from(heatmap.rows);
        for r in start_row..=end_row {
            if let Some(v) = usize::try_from(base + i64::from(r))
                .ok()
                .and_then(|i| heatmap.counts.get_mut(i))
            {
                *v = v.wrapping_add(count);
            }
        }
    }
}

/// `calcRowsOrCols(...)`: the intervals of `cell_range` covering the
/// request, at most the world's.
pub(crate) fn calc_rows_or_cols(
    cell_range: f64,
    cell_min: f64,
    request_range: f64,
    request_min: f64,
    world_range: f64,
) -> i32 {
    let range = request_range + (request_min - cell_min);
    if range == 0.0 {
        return 1;
    }
    let intervals = (range / cell_range).ceil();
    if intervals > f64::from(i32::MAX) {
        return i32::MAX; // should result in an error soon
    }
    // no more intervals than world bounds (rounding/edge issue)
    let intervals_max = java_round_long(world_range / cell_range);
    if intervals_max > i64::from(i32::MAX) {
        return intervals as i32;
    }
    (intervals_max as i32).min(intervals as i32)
}

/// `Math.round(double)` to `long`.
fn java_round_long(d: f64) -> i64 {
    (d + 0.5).floor() as i64
}

// ---------------------------------------------------------------------------
// NumberRangePrefixTreeStrategy.Facets
// ---------------------------------------------------------------------------

/// `NumberRangePrefixTreeStrategy.Facets.FacetParentVal`: a block of
/// detail-level counts under one parent.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FacetParentVal {
    /// `parentLeaves`: ranges spanning all of the children.
    pub parent_leaves: i32,
    /// `childCountsLen`.
    pub child_counts_len: i32,
    /// `childCounts`, `None` until a child is counted.
    pub child_counts: Option<Vec<i32>>,
}

/// `NumberRangePrefixTreeStrategy.Facets`: counts at a detail level,
/// grouped by parent (in the parents' order).
#[derive(Debug, Clone)]
pub struct Facets {
    /// `detailLevel`.
    pub detail_level: i32,
    /// `topLeaves`: ranges spanning the parents of the detail level.
    pub top_leaves: i32,
    /// `parents`, keyed by the parent's cell numbers (whose order is
    /// `UnitNRShape.compareTo`'s), with the parent itself.
    pub parents: BTreeMap<Vec<i32>, (UnitNRShape, FacetParentVal)>,
}

impl std::fmt::Display for Facets {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut buf = format!(
            "Facets: level={} topLeaves={} parentCount={}",
            self.detail_level,
            self.top_leaves,
            self.parents.len()
        );
        for (shape, p_val) in self.parents.values() {
            buf.push('\n');
            if buf.len() > 1000 {
                buf.push_str("...");
                break;
            }
            buf.push_str(&format!(" {shape} leafCount={}", p_val.parent_leaves));
            if let Some(c) = &p_val.child_counts {
                let parts: Vec<String> = c.iter().map(i32::to_string).collect();
                buf.push_str(&format!(" [{}]", parts.join(", ")));
            }
        }
        f.write_str(&buf)
    }
}

/// `UnitNRShape.clone()`: Java re-reads the cell from its term, which for
/// the world cell (no term) indexes its term buffer at `-1`.
fn clone_unit(u: &UnitNRShape) -> Result<UnitNRShape> {
    if u.level() == 0 {
        let len = u.tree().tree().base().max_term_len();
        return Err(Error::Spatial(
            lucene_util::spatial4j::Error::ArrayIndexOutOfBounds(format!(
                "Index -1 out of bounds for length {len}"
            )),
        ));
    }
    Ok(u.unit_clone())
}

/// The facet visitor of `calcFacets(context, topAcceptDocs, facetRange,
/// level)`.
struct NumberRangeFacetVisitor<'f> {
    facets: &'f mut Facets,
    level: i32,
    /// `parentShape`/`parentFacet`: the key of the current parent.
    parent: Option<Vec<i32>>,
}

impl NumberRangeFacetVisitor<'_> {
    fn setup_parent(&mut self, unit_shape: &UnitNRShape) -> Result<()> {
        let parent_shape = clone_unit(unit_shape)?;
        let key = parent_shape.vals().to_vec();
        if !self.facets.parents.contains_key(&key) {
            let len = parent_shape.tree().num_sub_cells(&parent_shape)?;
            self.facets.parents.insert(
                key.clone(),
                (
                    parent_shape,
                    FacetParentVal {
                        child_counts_len: len,
                        ..FacetParentVal::default()
                    },
                ),
            );
        }
        self.parent = Some(key);
        Ok(())
    }

    fn parent_val(&mut self) -> &mut FacetParentVal {
        let key = self.parent.as_ref().expect("a parent was set up");
        &mut self
            .facets
            .parents
            .get_mut(key)
            .expect("set up with its key")
            .1
    }
}

fn unit_of(cell: &dyn Cell) -> Result<UnitNRShape> {
    let shape = cell.shape()?;
    match shape.as_any().downcast_ref::<UnitNRShape>() {
        Some(u) => Ok(u.clone()),
        None => Err(crate::spatial::class_cast(
            &*shape,
            "org.apache.lucene.spatial.prefix.tree.NumberRangePrefixTree$UnitNRShape",
        )),
    }
}

impl FacetVisitor for NumberRangeFacetVisitor<'_> {
    fn visit(&mut self, cell: &dyn Cell, count: i32) -> Result<()> {
        if cell.level() < self.level - 1 {
            // some ancestor of parent facet level, direct or distant
            self.parent = None;
            self.facets.top_leaves = self.facets.top_leaves.wrapping_add(count);
        } else if cell.level() == self.level - 1 {
            // parent
            let u = unit_of(cell)?;
            self.setup_parent(&u)?;
            let p = self.parent_val();
            p.parent_leaves = p.parent_leaves.wrapping_add(count);
        } else {
            // at facet level
            let unit_shape = unit_of(cell)?;
            let unit_shape_parent = unit_shape.shape_at_level(unit_shape.level() - 1);
            let same_parent = self
                .parent
                .as_ref()
                .is_some_and(|k| k.as_slice() == unit_shape_parent.vals());
            if !same_parent {
                self.setup_parent(&unit_shape_parent)?;
            }
            let p = self.parent_val();
            let len = usize::try_from(p.child_counts_len).unwrap_or(0);
            let counts = p.child_counts.get_or_insert_with(|| vec![0; len]);
            let i = unit_shape.val_at_level(cell.level());
            let slot = usize::try_from(i)
                .ok()
                .and_then(|i| counts.get_mut(i))
                .ok_or_else(|| {
                    Error::Spatial(lucene_util::spatial4j::Error::ArrayIndexOutOfBounds(
                        format!("Index {i} out of bounds for length {len}"),
                    ))
                })?;
            *slot = slot.wrapping_add(count);
        }
        Ok(())
    }
}

/// `NumberRangePrefixTreeStrategy.calcFacets(context, topAcceptDocs,
/// facetRange, level)`.
pub(crate) fn calc_number_range_facets(
    strategy: &PrefixTreeStrategy,
    leaves: &[OpenSegment<'_>],
    top_accept_docs: AcceptDocs<'_>,
    facet_range: &Arc<dyn Shape>,
    level: i32,
) -> Result<Facets> {
    let mut facets = Facets {
        detail_level: level,
        top_leaves: 0,
        parents: BTreeMap::new(),
    };
    let mut v = NumberRangeFacetVisitor {
        facets: &mut facets,
        level,
        parent: None,
    };
    compute(
        strategy,
        leaves,
        top_accept_docs,
        facet_range,
        level,
        &mut v,
    )?;
    Ok(facets)
}
