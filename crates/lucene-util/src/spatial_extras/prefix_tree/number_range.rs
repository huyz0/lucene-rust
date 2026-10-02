//! `NumberRangePrefixTree`: a one-dimensional prefix tree whose cells are
//! also shapes. A `UnitNRShape` is a cell -- a stack of cell numbers, one
//! per level -- and a `SpanUnitsNRShape` a range between two of them; both
//! relate to each other as ranges of the number line.
//!
//! A level's cell number is one byte (`n + 1`) when the level has fewer than
//! 256 cells, else two (`n >> 7`, `(n & 0x7F) + 1`), so no term byte but a
//! two-byte level's high byte is 0 -- which leaves a trailing 0 byte free to
//! mark a leaf.
//!
//! # What Rust changes
//!
//! Java's `NRCell` is at once a cell, its level's cell iterator and a
//! `UnitNRShape`, all sharing one term buffer per cell stack and decoding
//! lazily. Here a [`UnitNRShape`] is a value: its cell numbers, plus the
//! cell state (leaf, relation, the filter it was iterated with), and
//! [`NRCellIterator`] is a separate iterator. `initIter`'s fast path, which
//! reuses the parent's iteration state to skip two prefix comparisons,
//! is not ported: it answers what the general path answers.

use std::any::Any;
use std::cmp::Ordering;
use std::fmt;
use std::sync::{Arc, OnceLock};

use super::{Cell, CellIterator, IterState, SpatialPrefixTree};
use crate::spatial4j::{
    Error, Point, Rectangle, Result, Shape, SpatialContext, SpatialContextFactory, SpatialRelation,
};

/// `a[i]` with Java's bounds check: `ArrayIndexOutOfBoundsException`'s
/// message on a miss.
pub(crate) fn java_index(a: &[i32], i: usize) -> Result<i32> {
    a.get(i).copied().ok_or_else(|| {
        Error::ArrayIndexOutOfBounds(format!("Index {i} out of bounds for length {}", a.len()))
    })
}

thread_local! {
    /// Whether this thread is inside [`relate_transposed`].
    static RELATING_TRANSPOSED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// `shape.relate(this).transpose()`: Java's fallback for a shape that is
/// neither number-range shape ("probably a UnitNRShape").
///
/// A geometric shape answers by the same fallback, so Java recurses until
/// `StackOverflowError`. A Rust stack overflow aborts the process instead,
/// so a second fallback while one is in progress is an error naming it.
fn relate_transposed(me: &dyn Shape, other: &dyn Shape) -> Result<SpatialRelation> {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            RELATING_TRANSPOSED.with(|r| r.set(false));
        }
    }
    if RELATING_TRANSPOSED.with(|r| r.replace(true)) {
        return Err(Error::Runtime(format!(
            "java.lang.StackOverflowError: a number-range shape cannot relate to {other}"
        )));
    }
    let _reset = Reset;
    Ok(other.relate(me)?.transpose())
}

/// What a concrete number-range tree adds (Java's abstract methods).
pub trait NumberRangeTree: fmt::Debug + Send + Sync + Any {
    /// The level layout.
    fn base(&self) -> &NrBase;
    /// `getNumSubCells(lv)`.
    ///
    /// Java indexes its arrays unchecked, so a cell at the tree's last level
    /// throws `ArrayIndexOutOfBoundsException`; that is an error here (see
    /// [`java_index`]). Java's base implementation,
    /// `maxSubCellsByLevel[level]`, is the date tree's fallback.
    fn num_sub_cells(&self, lv: &[i32]) -> Result<i32>;
    /// `toString(UnitNRShape)`.
    fn unit_to_string(&self, lv: &[i32]) -> String;
    /// `parseUnitShape(str)`: the cell numbers.
    fn parse_unit_shape(&self, s: &str) -> Result<Vec<i32>>;
    /// `getClass().getSimpleName()`.
    fn name(&self) -> &'static str;
    /// For downcasting.
    fn as_any(&self) -> &dyn Any;
}

/// `NumberRangePrefixTree`'s per-level layout.
#[derive(Debug, Clone)]
pub struct NrBase {
    pub max_sub_cells_by_level: Vec<i32>,
    term_len_by_level: Vec<usize>,
    level_by_term_len: Vec<i32>,
    max_term_len: usize,
}

impl NrBase {
    /// `NumberRangePrefixTree(maxSubCellsByLevel)`.
    pub fn new(max_sub_cells_by_level: Vec<i32>) -> Result<Self> {
        let max_levels = max_sub_cells_by_level.len();
        let mut term_len_by_level = vec![0usize; max_levels + 1];
        const MAX_STATES: i32 = 1 << 15;
        for level in 1..=max_levels {
            let states = max_sub_cells_by_level[level - 1];
            if states >= MAX_STATES || states <= 1 {
                return Err(Error::IllegalArgument(format!(
                    "Max states is {MAX_STATES}, given {states} at level {level}"
                )));
            }
            let two_bytes = states >= 256;
            term_len_by_level[level] = term_len_by_level[level - 1] + if two_bytes { 2 } else { 1 };
        }
        let max_term_len = term_len_by_level[max_levels] + 1;
        let mut level_by_term_len = vec![0i32; max_term_len];
        for level in 1..term_len_by_level.len() {
            let term_len = term_len_by_level[level];
            let prev = term_len_by_level[level - 1];
            if term_len - prev == 2 {
                level_by_term_len[term_len - 1] = -1;
            }
            level_by_term_len[term_len] = level as i32;
        }
        Ok(NrBase {
            max_sub_cells_by_level,
            term_len_by_level,
            level_by_term_len,
            max_term_len,
        })
    }

    /// `maxTermLen`: the longest term (a leaf at the last level).
    pub fn max_term_len(&self) -> usize {
        self.max_term_len
    }

    /// `getMaxLevels()`.
    pub fn max_levels(&self) -> i32 {
        self.max_sub_cells_by_level.len() as i32
    }

    /// The term (no leaf byte) of a stack of cell numbers.
    fn encode(&self, vals: &[i32]) -> Vec<u8> {
        let mut term = Vec::with_capacity(self.term_len_by_level[vals.len()] + 1);
        for (i, &n) in vals.iter().enumerate() {
            let level = i + 1;
            let two_bytes = self.term_len_by_level[level] - self.term_len_by_level[level - 1] > 1;
            if two_bytes {
                term.push((n >> 7) as u8);
                term.push(((n & 0x7F) + 1) as u8);
            } else {
                term.push((n + 1) as u8);
            }
        }
        term
    }

    /// `readCell(term)`'s decoding: the cell numbers and the leaf flag.
    fn decode(&self, term: &[u8]) -> Result<(Vec<i32>, bool)> {
        let is_leaf = term.last() == Some(&0);
        let len_no_leaf = if is_leaf { term.len() - 1 } else { term.len() };
        let level = match self.level_by_term_len.get(len_no_leaf) {
            Some(&l) if l >= 0 => l as usize,
            _ => {
                return Err(Error::ArrayIndexOutOfBounds(format!(
                    "no level for a term of length {len_no_leaf}"
                )))
            }
        };
        let mut vals = Vec::with_capacity(level);
        for l in 1..=level {
            let term_len = self.term_len_by_level[l];
            let two_bytes = term_len - self.term_len_by_level[l - 1] > 1;
            if two_bytes {
                let h = term[term_len - 2] as i32;
                let lo = term[term_len - 1] as i32;
                vals.push((h << 7) + (lo - 1));
            } else {
                vals.push(term[term_len - 1] as i32 - 1);
            }
        }
        Ok((vals, is_leaf))
    }
}

/// `NumberRangePrefixTree.DUMMY_CTX`: planar, x unbounded, y pinned to 0.
pub fn dummy_ctx() -> Arc<SpatialContext> {
    static CTX: OnceLock<Arc<SpatialContext>> = OnceLock::new();
    CTX.get_or_init(|| {
        let mut f = SpatialContextFactory::new();
        f.geo = false;
        f.world_bounds = Some([f64::NEG_INFINITY, f64::INFINITY, 0.0, 0.0]);
        f.new_spatial_context().expect("the dummy context is valid")
    })
    .clone()
}

/// `comparePrefix(a, b)`.
pub fn compare_prefix(a: &[i32], b: &[i32]) -> i32 {
    for (x, y) in a.iter().zip(b.iter()) {
        let diff = x - y;
        if diff != 0 {
            return diff;
        }
    }
    0
}

/// `NumberRangePrefixTree`: a handle on a concrete tree.
#[derive(Debug, Clone)]
pub struct NumberRangePrefixTree {
    tree: Arc<dyn NumberRangeTree>,
}

impl NumberRangePrefixTree {
    pub fn new(tree: Arc<dyn NumberRangeTree>) -> Self {
        NumberRangePrefixTree { tree }
    }

    /// The concrete tree.
    pub fn tree(&self) -> &Arc<dyn NumberRangeTree> {
        &self.tree
    }

    /// A unit shape (no cell state) of cell numbers.
    pub fn unit(&self, vals: Vec<i32>) -> UnitNRShape {
        UnitNRShape {
            tree: self.clone(),
            vals,
            leaf: false,
            shape_rel: None,
            iter_filter: None,
        }
    }

    /// `getNumSubCells(lv)`.
    pub fn num_sub_cells(&self, lv: &UnitNRShape) -> Result<i32> {
        self.tree.num_sub_cells(&lv.vals)
    }

    /// `getNumSubCells` of a strict prefix of a cell, which is always below
    /// the last level and so always has a count.
    fn prefix_sub_cells(&self, prefix: &[i32]) -> i32 {
        self.tree
            .num_sub_cells(prefix)
            .expect("a strict prefix is above the tree's last level")
    }

    /// `truncateStartVals(lv, endLevel)`: chops trailing zeros.
    fn truncate_start_vals(&self, lv: &[i32], end_level: usize) -> usize {
        let mut level = lv.len();
        while level > end_level {
            if lv[level - 1] != 0 {
                return level;
            }
            level -= 1;
        }
        end_level
    }

    /// `truncateEndVals(lv, endLevel)`: chops trailing maximum values.
    fn truncate_end_vals(&self, lv: &[i32], end_level: usize) -> usize {
        let mut level = lv.len();
        while level > end_level {
            let max = self.prefix_sub_cells(&lv[..level - 1]) - 1;
            if lv[level - 1] != max {
                return level;
            }
            level -= 1;
        }
        end_level
    }

    /// `toRangeShape(startUnit, endUnit)`: normalised (trailing minimum and
    /// maximum values dropped), and a unit when the range is one.
    pub fn to_range_shape(&self, start: &UnitNRShape, end: &UnitNRShape) -> Result<NRShape> {
        let start = self.unit(start.vals[..self.truncate_start_vals(&start.vals, 0)].to_vec());
        let end = self.unit(end.vals[..self.truncate_end_vals(&end.vals, 0)].to_vec());
        let cmp = compare_prefix(&start.vals, &end.vals);
        if cmp > 0 {
            return Err(Error::IllegalArgument(format!(
                "Wrong order: {start} TO {end}"
            )));
        }
        if cmp == 0 {
            let (sl, el) = (start.level(), end.level());
            if sl == el {
                return Ok(NRShape::Unit(start));
            } else if el > sl {
                if self.truncate_start_vals(&end.vals, sl as usize) == sl as usize {
                    return Ok(NRShape::Unit(end));
                }
            } else if self.truncate_end_vals(&start.vals, el as usize) == el as usize {
                return Ok(NRShape::Unit(start));
            }
        }
        Ok(NRShape::Span(SpanUnitsNRShape::new(start, end)))
    }

    /// `parseShape(str)`: a unit, or `[a TO b]`.
    pub fn parse_shape(&self, s: &str) -> Result<NRShape> {
        if s.is_empty() {
            return Err(Error::IllegalArgument("str is null or blank".into()));
        }
        let chars: Vec<char> = s.chars().collect();
        if chars[0] == '[' {
            if chars[chars.len() - 1] != ']' {
                return Err(Error::Parse {
                    message: format!("If starts with [ must end with ]; got {s}"),
                    offset: chars.len() as i32 - 1,
                });
            }
            let Some(middle) = s.find(" TO ") else {
                return Err(Error::Parse {
                    message: format!("If starts with [ must contain ' TO '; got {s}"),
                    offset: -1,
                });
            };
            let left = &s[1..middle];
            let right = &s[middle + 4..s.len() - 1];
            let l = self.unit(self.tree.parse_unit_shape(left)?);
            let r = self.unit(self.tree.parse_unit_shape(right)?);
            self.to_range_shape(&l, &r)
        } else if chars[0] == '{' {
            Err(Error::Parse {
                message: format!("Exclusive ranges not supported; got {s}"),
                offset: 0,
            })
        } else {
            Ok(NRShape::Unit(self.unit(self.tree.parse_unit_shape(s)?)))
        }
    }

    /// `toStringUnitRaw(lv)`: `[n,n,...]`.
    pub fn to_string_unit_raw(lv: &UnitNRShape) -> String {
        let parts: Vec<String> = lv.vals.iter().map(|v| v.to_string()).collect();
        if parts.is_empty() {
            "]".into()
        } else {
            format!("[{}]", parts.join(","))
        }
    }
}

impl fmt::Display for NumberRangePrefixTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.tree.name())
    }
}

impl SpatialPrefixTree for NumberRangePrefixTree {
    fn spatial_context(&self) -> &Arc<SpatialContext> {
        static CTX: OnceLock<Arc<SpatialContext>> = OnceLock::new();
        CTX.get_or_init(dummy_ctx)
    }

    fn max_levels(&self) -> i32 {
        self.tree.base().max_levels()
    }

    /// Always the full precision: this tree does no approximation.
    fn level_for_distance(&self, _dist: f64) -> i32 {
        self.max_levels()
    }

    fn distance_for_level(&self, _level: i32) -> Result<f64> {
        Err(Error::UnsupportedOperation(Some("Not applicable.".into())))
    }

    fn world_cell(&self) -> Box<dyn Cell> {
        Box::new(self.unit(Vec::new()))
    }

    fn read_cell(&self, term: &[u8]) -> Result<Box<dyn Cell>> {
        if term.is_empty() {
            return Ok(self.world_cell());
        }
        let (vals, is_leaf) = self.tree.base().decode(term)?;
        let mut cell = self.unit(vals);
        if is_leaf {
            cell.leaf = true;
        }
        Ok(Box::new(cell))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// `NRShape`: what `toRangeShape`/`parseShape` return.
#[derive(Debug, Clone)]
pub enum NRShape {
    Unit(UnitNRShape),
    Span(SpanUnitsNRShape),
}

impl NRShape {
    /// The shape, for strategies and queries.
    pub fn into_shape(self) -> Arc<dyn Shape> {
        match self {
            NRShape::Unit(u) => Arc::new(u),
            NRShape::Span(s) => Arc::new(s),
        }
    }

    /// `roundToLevel(targetLevel)`.
    pub fn round_to_level(&self, target_level: i32) -> Result<NRShape> {
        match self {
            NRShape::Unit(u) => Ok(NRShape::Unit(u.round_to_level(target_level))),
            NRShape::Span(s) => s.round_to_level(target_level),
        }
    }
}

impl fmt::Display for NRShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NRShape::Unit(u) => write!(f, "{u}"),
            NRShape::Span(s) => write!(f, "{s}"),
        }
    }
}

/// `UnitNRShape` (Java's `NRCell`): a cell of the tree, as a shape.
#[derive(Clone)]
pub struct UnitNRShape {
    tree: NumberRangePrefixTree,
    vals: Vec<i32>,
    leaf: bool,
    shape_rel: Option<SpatialRelation>,
    /// The filter the cell was iterated with (`iterFilter`), for `relate`'s
    /// identity shortcut.
    iter_filter: Option<Arc<dyn Shape>>,
}

impl fmt::Debug for UnitNRShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "UnitNRShape({:?})", self.vals)
    }
}

impl UnitNRShape {
    /// `getLevel()`.
    pub fn level(&self) -> i32 {
        self.vals.len() as i32
    }

    /// `getValAtLevel(level)` (1-based).
    pub fn val_at_level(&self, level: i32) -> i32 {
        self.vals[level as usize - 1]
    }

    /// The cell numbers, level 1 first.
    pub fn vals(&self) -> &[i32] {
        &self.vals
    }

    /// `getShapeAtLevel(level)`: the ancestor at `level` (no cell state).
    pub fn shape_at_level(&self, level: i32) -> UnitNRShape {
        self.tree.unit(self.vals[..level as usize].to_vec())
    }

    /// `roundToLevel(targetLevel)`.
    pub fn round_to_level(&self, target_level: i32) -> UnitNRShape {
        if self.level() <= target_level {
            self.clone()
        } else {
            self.shape_at_level(target_level)
        }
    }

    /// `clone()` (`UnitNRShape.clone`): the same cell numbers, no cell state.
    pub fn unit_clone(&self) -> UnitNRShape {
        self.tree.unit(self.vals.clone())
    }

    /// `compareTo(o)`: prefix order, then shorter first.
    pub fn compare_to(&self, o: &UnitNRShape) -> Ordering {
        let cmp = compare_prefix(&self.vals, &o.vals);
        if cmp != 0 {
            cmp.cmp(&0)
        } else {
            self.vals.len().cmp(&o.vals.len())
        }
    }

    /// The tree this cell is of.
    pub fn tree(&self) -> &NumberRangePrefixTree {
        &self.tree
    }

    /// `relate(UnitNRShape)`.
    fn relate_unit(&self, lv: &UnitNRShape) -> SpatialRelation {
        if compare_prefix(&self.vals, &lv.vals) != 0 {
            return SpatialRelation::Disjoint;
        }
        if self.level() > lv.level() {
            return SpatialRelation::Within;
        }
        SpatialRelation::Contains
    }

    /// `relate(SpanUnitsNRShape)`.
    fn relate_span(&self, span: &SpanUnitsNRShape) -> SpatialRelation {
        let start_cmp = compare_prefix(&span.min.vals, &self.vals);
        if start_cmp > 0 {
            return SpatialRelation::Disjoint;
        }
        let end_cmp = compare_prefix(&span.max.vals, &self.vals);
        if end_cmp < 0 {
            return SpatialRelation::Disjoint;
        }
        let mut nr_min_level = span.min.level();
        let mut nr_max_level = span.max.level();
        if (start_cmp < 0 || start_cmp == 0 && nr_min_level <= self.level())
            && (end_cmp > 0 || end_cmp == 0 && nr_max_level <= self.level())
        {
            return SpatialRelation::Within;
        }
        if start_cmp != 0 || end_cmp != 0 {
            return SpatialRelation::Intersects;
        }
        while nr_min_level < self.level() {
            if self.val_at_level(nr_min_level + 1) != 0 {
                return SpatialRelation::Intersects;
            }
            nr_min_level += 1;
        }
        while nr_max_level < self.level() {
            let max = self
                .tree
                .prefix_sub_cells(&self.vals[..nr_max_level as usize])
                - 1;
            if self.val_at_level(nr_max_level + 1) != max {
                return SpatialRelation::Intersects;
            }
            nr_max_level += 1;
        }
        SpatialRelation::Contains
    }

    fn unsupported<T>() -> Result<T> {
        Err(Error::UnsupportedOperation(None))
    }
}

impl fmt::Display for UnitNRShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.tree.tree.unit_to_string(&self.vals))
    }
}

impl Shape for UnitNRShape {
    fn relate(&self, other: &dyn Shape) -> Result<SpatialRelation> {
        if let (Some(f), Some(rel)) = (&self.iter_filter, self.shape_rel) {
            if std::ptr::addr_eq(Arc::as_ptr(f), other as *const dyn Shape) {
                return Ok(rel);
            }
        }
        if let Some(u) = other.as_any().downcast_ref::<UnitNRShape>() {
            return Ok(self.relate_unit(u));
        }
        if let Some(s) = other.as_any().downcast_ref::<SpanUnitsNRShape>() {
            return Ok(self.relate_span(s));
        }
        relate_transposed(self, other)
    }

    fn bounding_box(&self) -> Result<Arc<dyn Rectangle>> {
        Self::unsupported()
    }

    fn has_area(&self) -> bool {
        true
    }

    fn area(&self, _ctx: Option<&SpatialContext>) -> Result<f64> {
        Self::unsupported()
    }

    fn center(&self) -> Result<Arc<dyn Point>> {
        Self::unsupported()
    }

    fn buffered(&self, _d: f64, _ctx: &Arc<SpatialContext>) -> Result<Arc<dyn Shape>> {
        Self::unsupported()
    }

    fn is_empty(&self) -> bool {
        false
    }

    /// `NRCell.equals`: the same level and term.
    fn equals(&self, other: &dyn Shape) -> bool {
        other
            .as_any()
            .downcast_ref::<UnitNRShape>()
            .is_some_and(|o| o.vals == self.vals)
    }

    fn context(&self) -> Option<&Arc<SpatialContext>> {
        Some(self.tree.spatial_context())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Cell for UnitNRShape {
    fn shape_rel(&self) -> Option<SpatialRelation> {
        self.shape_rel
    }

    fn set_shape_rel(&mut self, rel: Option<SpatialRelation>) {
        self.shape_rel = rel;
    }

    fn is_leaf(&self) -> bool {
        self.leaf
    }

    fn set_leaf(&mut self) {
        self.leaf = true;
    }

    fn token_bytes_with_leaf(&self) -> Vec<u8> {
        let mut t = self.token_bytes_no_leaf();
        if self.leaf {
            t.push(0);
        }
        t
    }

    fn token_bytes_no_leaf(&self) -> Vec<u8> {
        self.tree.tree.base().encode(&self.vals)
    }

    fn level(&self) -> i32 {
        self.vals.len() as i32
    }

    fn next_level_cells(
        &self,
        shape_filter: Option<&Arc<dyn Shape>>,
    ) -> Result<Box<dyn CellIterator>> {
        // Java takes `cellsByLevel[cellLevel + 1]` from a `maxLevels + 1`
        // stack first, so a last-level cell throws there.
        let stack = self.tree.tree.base().max_levels() + 1;
        if self.level() + 1 >= stack {
            return Err(Error::ArrayIndexOutOfBounds(format!(
                "Index {} out of bounds for length {stack}",
                self.level() + 1
            )));
        }
        Ok(Box::new(NRCellIterator::new(self, shape_filter)?))
    }

    fn shape(&self) -> Result<Arc<dyn Shape>> {
        Ok(Arc::new(self.clone()))
    }

    fn is_prefix_of(&self, c: &dyn Cell) -> bool {
        let mine = self.token_bytes_no_leaf();
        let other = c.token_bytes_no_leaf();
        other.len() >= mine.len() && other[..mine.len()] == mine[..]
    }

    fn compare_to_no_leaf(&self, from_cell: &dyn Cell) -> i32 {
        super::compare_bytes(
            &self.token_bytes_no_leaf(),
            &from_cell.token_bytes_no_leaf(),
        )
    }

    fn sub_cells_size(&self) -> Option<i32> {
        None
    }

    fn clone_box(&self) -> Box<dyn Cell> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// `NRCell`'s iteration over a cell's children.
pub struct NRCellIterator {
    parent: UnitNRShape,
    filter: Option<Arc<dyn Shape>>,
    first: i32,
    first_is_intersects: bool,
    last: i32,
    last_is_intersects: bool,
    cell_number: i32,
    st: IterState,
}

impl NRCellIterator {
    /// `getNextLevelCells(filter)` / `initIter(filter)`.
    fn new(parent: &UnitNRShape, filter: Option<&Arc<dyn Shape>>) -> Result<Self> {
        let mut filter = filter.cloned();
        if filter
            .as_ref()
            .and_then(|f| f.as_any().downcast_ref::<UnitNRShape>())
            .is_some_and(|u| u.level() == 0)
        {
            filter = None; // world means everything -- no filter
        }
        let parent = parent.unit_clone();
        let level = parent.level() + 1;
        let num_sub = parent.tree.num_sub_cells(&parent)?;
        let mut it = NRCellIterator {
            parent,
            filter: filter.clone(),
            first: 0,
            first_is_intersects: false,
            last: num_sub - 1,
            last_is_intersects: false,
            cell_number: -1,
            st: IterState::default(),
        };
        let Some(filter) = filter else {
            return Ok(it);
        };
        let (min_lv, max_lv) = if let Some(s) = filter.as_any().downcast_ref::<SpanUnitsNRShape>() {
            (s.min.clone(), s.max.clone())
        } else if let Some(u) = filter.as_any().downcast_ref::<UnitNRShape>() {
            (u.unit_clone(), u.unit_clone())
        } else {
            return Err(Error::ClassCast(format!(
                "{filter} is not a NumberRangePrefixTree.UnitNRShape"
            )));
        };
        let start_cmp = compare_prefix(&min_lv.vals, &it.parent.vals);
        let end_cmp = compare_prefix(&max_lv.vals, &it.parent.vals);
        if start_cmp > 0 || end_cmp < 0 {
            it.first = 0;
            it.last = -1; // so ends early (no cells)
            return Ok(it);
        }
        if start_cmp < 0 || min_lv.level() < level {
            it.first = 0;
            it.first_is_intersects = false;
        } else {
            it.first = min_lv.val_at_level(level);
            it.first_is_intersects = min_lv.level() > level;
        }
        if end_cmp > 0 || max_lv.level() < level {
            it.last = num_sub - 1;
            it.last_is_intersects = false;
        } else {
            it.last = max_lv.val_at_level(level);
            it.last_is_intersects = max_lv.level() > level;
        }
        if it.first == it.last {
            if it.last_is_intersects {
                it.first_is_intersects = true;
            } else if it.first_is_intersects {
                it.last_is_intersects = true;
            }
        }
        Ok(it)
    }
}

impl CellIterator for NRCellIterator {
    fn has_next(&mut self) -> Result<bool> {
        self.st.this_cell = None;
        if self.st.next_cell.is_some() {
            return Ok(true);
        }
        if self.cell_number >= self.last {
            return Ok(false);
        }
        self.cell_number = if self.cell_number < self.first {
            self.first
        } else {
            self.cell_number + 1
        };
        let has_children = (self.cell_number == self.first && self.first_is_intersects)
            || (self.cell_number == self.last && self.last_is_intersects);
        let mut vals = self.parent.vals.clone();
        vals.push(self.cell_number);
        let mut cell = self.parent.tree.unit(vals);
        cell.iter_filter = self.filter.clone();
        if !has_children {
            cell.leaf = true;
            cell.shape_rel = Some(SpatialRelation::Within);
        } else if self.first == self.last {
            cell.shape_rel = Some(SpatialRelation::Contains);
        } else {
            cell.shape_rel = Some(SpatialRelation::Intersects);
        }
        self.st.next_cell = Some(Box::new(cell));
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
}

/// `SpanUnitsNRShape`: the range from one unit to another.
#[derive(Clone)]
pub struct SpanUnitsNRShape {
    min: UnitNRShape,
    max: UnitNRShape,
    last_level_in_common: i32,
}

impl fmt::Debug for SpanUnitsNRShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SpanUnitsNRShape({:?}, {:?})",
            self.min.vals, self.max.vals
        )
    }
}

impl SpanUnitsNRShape {
    fn new(min: UnitNRShape, max: UnitNRShape) -> Self {
        let mut level = 1;
        while level <= min.level() && level <= max.level() {
            if min.val_at_level(level) != max.val_at_level(level) {
                break;
            }
            level += 1;
        }
        SpanUnitsNRShape {
            min,
            max,
            last_level_in_common: level - 1,
        }
    }

    /// `getMinUnit()`.
    pub fn min_unit(&self) -> &UnitNRShape {
        &self.min
    }

    /// `getMaxUnit()`.
    pub fn max_unit(&self) -> &UnitNRShape {
        &self.max
    }

    /// `getLevelsInCommon()`.
    pub fn levels_in_common(&self) -> i32 {
        self.last_level_in_common
    }

    /// `roundToLevel(targetLevel)`.
    pub fn round_to_level(&self, target_level: i32) -> Result<NRShape> {
        self.min.tree.clone().to_range_shape(
            &self.min.round_to_level(target_level),
            &self.max.round_to_level(target_level),
        )
    }

    /// `relate(SpanUnitsNRShape)`.
    fn relate_span(&self, ext: &SpanUnitsNRShape) -> SpatialRelation {
        if compare_prefix(&ext.min.vals, &self.max.vals) > 0 {
            return SpatialRelation::Disjoint;
        }
        if compare_prefix(&ext.max.vals, &self.min.vals) < 0 {
            return SpatialRelation::Disjoint;
        }
        let ext_min_int_min = compare_prefix(&ext.min.vals, &self.min.vals);
        let ext_max_int_max = compare_prefix(&ext.max.vals, &self.max.vals);
        if (ext_min_int_min > 0 || ext_min_int_min == 0 && ext.min.level() >= self.min.level())
            && (ext_max_int_max < 0 || ext_max_int_max == 0 && ext.max.level() >= self.max.level())
        {
            return SpatialRelation::Contains;
        }
        if (ext_min_int_min < 0 || ext_min_int_min == 0 && ext.min.level() <= self.min.level())
            && (ext_max_int_max > 0 || ext_max_int_max == 0 && ext.max.level() <= self.max.level())
        {
            return SpatialRelation::Within;
        }
        SpatialRelation::Intersects
    }
}

impl fmt::Display for SpanUnitsNRShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{} TO {}]", self.min, self.max)
    }
}

impl Shape for SpanUnitsNRShape {
    fn relate(&self, other: &dyn Shape) -> Result<SpatialRelation> {
        if let Some(s) = other.as_any().downcast_ref::<SpanUnitsNRShape>() {
            return Ok(self.relate_span(s));
        }
        relate_transposed(self, other)
    }

    fn bounding_box(&self) -> Result<Arc<dyn Rectangle>> {
        UnitNRShape::unsupported()
    }

    fn has_area(&self) -> bool {
        true
    }

    fn area(&self, _ctx: Option<&SpatialContext>) -> Result<f64> {
        UnitNRShape::unsupported()
    }

    fn center(&self) -> Result<Arc<dyn Point>> {
        UnitNRShape::unsupported()
    }

    fn buffered(&self, _d: f64, _ctx: &Arc<SpatialContext>) -> Result<Arc<dyn Shape>> {
        UnitNRShape::unsupported()
    }

    fn is_empty(&self) -> bool {
        false
    }

    fn equals(&self, other: &dyn Shape) -> bool {
        other
            .as_any()
            .downcast_ref::<SpanUnitsNRShape>()
            .is_some_and(|o| o.max.vals == self.max.vals && o.min.vals == self.min.vals)
    }

    fn context(&self) -> Option<&Arc<SpatialContext>> {
        Some(self.min.tree.spatial_context())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
