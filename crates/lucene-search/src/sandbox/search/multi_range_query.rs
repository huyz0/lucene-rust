//! `MultiRangeQuery`: documents with a point inside any of several ranges
//! of one point field, at a constant score.
//!
//! A port of `MultiRangeQuery`: one-dimensional queries rewrite by merging
//! overlapping ranges (`mergeOverlappingRanges`); each segment then walks
//! the field's BKD tree with a visitor relating cells to a single range or
//! to an interval tree of them (`RangeTree`, built by median selection,
//! splitting on each dimension in turn), or matches every document at once
//! when every document has a point and the field's bounds lie inside.
//!
//! Deliberate differences:
//! - `RangeTree` picks each node's median with `select_nth_unstable_by`
//!   rather than `ArrayUtil.select`'s introselect. Which range lands at a
//!   node can differ on ties; the relation and the hits cannot, since a
//!   tree answers "inside" or "crosses" only from a range that is.
//! - A query with no clauses fails with an error where Java's dereferences
//!   a `null` tree (`NullPointerException`), and a clause whose bounds are
//!   not `numDims * bytesPerDim` bytes fails with one where Java's indexes
//!   out of bounds.
//! - `count(LeafReaderContext)`, `toString` and `equals`/`hashCode` are not
//!   ported.

use std::cmp::Ordering;

use lucene_codecs::points::{IntersectVisitor, Relation};

use super::saturated;
use crate::collector::ScoringCollector;
use crate::document::{collect_live, field_info, reader, DocumentQuery};
use crate::multi_segment::OpenSegment;
use crate::{Error, Result};

fn illegal(message: impl Into<String>) -> Error {
    Error::DocumentQuery(message.into())
}

/// `MultiRangeQuery.RangeClause`: one range's packed lower and upper
/// corners, inclusive.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RangeClause {
    pub lower_value: Vec<u8>,
    pub upper_value: Vec<u8>,
}

impl RangeClause {
    pub fn new(lower_value: Vec<u8>, upper_value: Vec<u8>) -> Self {
        RangeClause {
            lower_value,
            upper_value,
        }
    }
}

/// `MultiRangeQuery.Builder`: the clauses of one field's query. The typed
/// builders (`IntPointMultiRangeBuilder` and the rest) add the value checks.
#[derive(Debug, Clone)]
pub struct MultiRangeQueryBuilder {
    pub(crate) field: String,
    pub(crate) bytes_per_dim: usize,
    pub(crate) num_dims: usize,
    pub(crate) clauses: Vec<RangeClause>,
}

impl MultiRangeQueryBuilder {
    /// `new Builder(field, bytesPerDim, numDims)`, with its checks.
    pub fn new(field: impl Into<String>, bytes_per_dim: usize, num_dims: usize) -> Result<Self> {
        if bytes_per_dim == 0 {
            return Err(illegal("bytesPerDim should be a valid value"));
        }
        if num_dims == 0 {
            return Err(illegal("numDims should be a valid value"));
        }
        Ok(MultiRangeQueryBuilder {
            field: field.into(),
            bytes_per_dim,
            num_dims,
            clauses: Vec::new(),
        })
    }

    /// `add(RangeClause)`.
    pub fn add_clause(&mut self, clause: RangeClause) -> &mut Self {
        self.clauses.push(clause);
        self
    }

    /// `add(byte[] lowerValue, byte[] upperValue)`: packed corners, taken as
    /// they are.
    pub fn add(&mut self, lower_value: Vec<u8>, upper_value: Vec<u8>) -> &mut Self {
        self.add_clause(RangeClause::new(lower_value, upper_value))
    }

    /// `build()`.
    pub fn build(&self) -> MultiRangeQuery {
        MultiRangeQuery {
            field: self.field.clone(),
            num_dims: self.num_dims,
            bytes_per_dim: self.bytes_per_dim,
            range_clauses: self.clauses.clone(),
        }
    }
}

/// `MultiRangeQuery`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiRangeQuery {
    pub field: String,
    pub num_dims: usize,
    pub bytes_per_dim: usize,
    pub range_clauses: Vec<RangeClause>,
}

/// `ArrayUtil.getUnsignedComparator(bytesPerDim)`: the `width` bytes at
/// `a_off` of `a` against those at `b_off` of `b`, unsigned.
#[inline]
fn cmp_at(a: &[u8], a_off: usize, b: &[u8], b_off: usize, width: usize) -> Ordering {
    a[a_off..a_off + width].cmp(&b[b_off..b_off + width])
}

impl MultiRangeQuery {
    /// `getField()`.
    pub fn field(&self) -> &str {
        &self.field
    }

    /// `mergeOverlappingRanges(rangeClauses, bytesPerDim)`: the ranges
    /// sorted by (lower, upper) and every run of overlapping ones merged;
    /// `None` when nothing merged (Java returns the very same list).
    pub(crate) fn merge_overlapping_ranges(
        clauses: &[RangeClause],
        bytes_per_dim: usize,
    ) -> Option<Vec<RangeClause>> {
        if clauses.len() <= 1 {
            return None;
        }
        let w = bytes_per_dim;
        let mut origin = clauses.to_vec();
        // `List.sort`: stable, as Java's TimSort.
        origin.sort_by(|a, b| {
            cmp_at(&a.lower_value, 0, &b.lower_value, 0, w)
                .then_with(|| cmp_at(&a.upper_value, 0, &b.upper_value, 0, w))
        });
        let mut out = Vec::new();
        let mut iter = origin.into_iter();
        let mut current = iter.next()?;
        for next in iter {
            if cmp_at(&next.lower_value, 0, &current.upper_value, 0, w) == Ordering::Greater {
                out.push(std::mem::replace(&mut current, next));
            } else if cmp_at(&next.upper_value, 0, &current.upper_value, 0, w) == Ordering::Greater
            {
                current = RangeClause::new(current.lower_value, next.upper_value);
            }
        }
        out.push(current);
        (out.len() != clauses.len()).then_some(out)
    }

    /// Every clause's corners are `numDims * bytesPerDim` bytes (Java
    /// indexes past a short one).
    fn check_clauses(&self) -> Result<()> {
        let want = self.num_dims.saturating_mul(self.bytes_per_dim);
        for c in &self.range_clauses {
            if c.lower_value.len() != want || c.upper_value.len() != want {
                return Err(illegal(format!(
                    "field=\"{}\": a range clause's bounds must be {want} bytes (numDims={} * \
                     bytesPerDim={}), got {} and {}",
                    self.field,
                    self.num_dims,
                    self.bytes_per_dim,
                    c.lower_value.len(),
                    c.upper_value.len()
                )));
            }
        }
        Ok(())
    }
}

impl DocumentQuery for MultiRangeQuery {
    /// `rewrite`: a one-dimensional query with overlapping ranges becomes
    /// the query of the merged ranges.
    fn rewrite(&self, _leaves: &[OpenSegment<'_>]) -> Result<Option<Box<dyn DocumentQuery>>> {
        if self.num_dims != 1 {
            return Ok(None);
        }
        Ok(
            Self::merge_overlapping_ranges(&self.range_clauses, self.bytes_per_dim).map(|merged| {
                Box::new(MultiRangeQuery {
                    range_clauses: merged,
                    ..self.clone()
                }) as Box<dyn DocumentQuery>
            }),
        )
    }

    fn score_leaf(
        &self,
        leaf: &OpenSegment<'_>,
        boost: f32,
        collector: &mut dyn ScoringCollector,
    ) -> Result<()> {
        let Some(info) = field_info(leaf, &self.field)? else {
            return Ok(());
        };
        if info.point_dimension_count == 0 {
            return Ok(());
        }
        let r = reader(leaf)?;
        let points = r.points_reader()?;
        let Some(values) = points.field(info.number) else {
            return Ok(());
        };
        if usize::try_from(values.num_index_dims).ok() != Some(self.num_dims) {
            return Err(illegal(format!(
                "field=\"{}\" was indexed with numIndexDimensions={} but this query has numDims={}",
                self.field, values.num_index_dims, self.num_dims
            )));
        }
        if usize::try_from(values.bytes_per_dim).ok() != Some(self.bytes_per_dim) {
            return Err(illegal(format!(
                "field=\"{}\" was indexed with bytesPerDim={} but this query has bytesPerDim={}",
                self.field, values.bytes_per_dim, self.bytes_per_dim
            )));
        }
        self.check_clauses()?;
        let range = Relatable::new(&self.range_clauses, self.num_dims, self.bytes_per_dim)
            .ok_or_else(|| {
                illegal(format!(
                    "field=\"{}\": a MultiRangeQuery needs at least one range clause",
                    self.field
                ))
            })?;
        if saturated(collector, boost) {
            return Ok(());
        }
        let max_doc = r.max_doc;
        if values.doc_count == max_doc
            && range.relate(&values.min_packed_value, &values.max_packed_value)
                == Relation::CellInsideQuery
        {
            // Every document has a value and every value is inside.
            for doc in 0..max_doc {
                collect_live(leaf, doc, boost, collector);
                if saturated(collector, boost) {
                    break;
                }
            }
            return Ok(());
        }
        let mut visitor = Visitor {
            range: &range,
            docs: Vec::new(),
        };
        points.intersect(info.number, &mut visitor)?;
        let mut docs = visitor.docs;
        docs.sort_unstable();
        docs.dedup();
        for doc in docs {
            if !(0..max_doc).contains(&doc) {
                return Err(illegal(format!(
                    "points of field {} name document {doc}, outside the segment's 0..{max_doc}",
                    self.field
                )));
            }
            collect_live(leaf, doc, boost, collector);
            if saturated(collector, boost) {
                return Ok(());
            }
        }
        Ok(())
    }
}

/// The `IntersectVisitor` of the weight: `DocIdSetBuilder`'s documents.
struct Visitor<'r> {
    range: &'r Relatable<'r>,
    docs: Vec<i32>,
}

impl IntersectVisitor for Visitor<'_> {
    fn compare(&mut self, min_packed: &[u8], max_packed: &[u8]) -> Relation {
        self.range.relate(min_packed, max_packed)
    }

    fn visit(&mut self, doc_id: i32) {
        self.docs.push(doc_id);
    }

    fn visit_with_value(&mut self, doc_id: i32, packed_value: &[u8]) {
        if self.range.matches(packed_value) {
            self.docs.push(doc_id);
        }
    }

    fn grow(&mut self, count: usize) {
        self.docs.reserve(count.min(1 << 20));
    }
}

/// `getRange(clause, ...)`: one range's `Relatable`.
#[derive(Debug, Clone, Copy)]
struct Range<'c> {
    clause: &'c RangeClause,
    num_dims: usize,
    bytes_per_dim: usize,
}

impl Range<'_> {
    fn matches(&self, packed: &[u8]) -> bool {
        let (w, c) = (self.bytes_per_dim, self.clause);
        (0..self.num_dims).all(|dim| {
            let off = dim.saturating_mul(w);
            cmp_at(packed, off, &c.lower_value, off, w) != Ordering::Less
                && cmp_at(packed, off, &c.upper_value, off, w) != Ordering::Greater
        })
    }

    fn relate(&self, min: &[u8], max: &[u8]) -> Relation {
        let (w, c) = (self.bytes_per_dim, self.clause);
        let mut crosses = false;
        for dim in 0..self.num_dims {
            let off = dim.saturating_mul(w);
            if cmp_at(min, off, &c.upper_value, off, w) == Ordering::Greater
                || cmp_at(max, off, &c.lower_value, off, w) == Ordering::Less
            {
                return Relation::CellOutsideQuery;
            }
            crosses |= cmp_at(min, off, &c.lower_value, off, w) == Ordering::Less
                || cmp_at(max, off, &c.upper_value, off, w) == Ordering::Greater;
        }
        if crosses {
            Relation::CellCrossesQuery
        } else {
            Relation::CellInsideQuery
        }
    }
}

/// `RangeTree`: an interval tree node -- its range, the dimension it split
/// on, and the per-dimension maximum of its whole subtree.
#[derive(Debug)]
struct RangeTree<'c> {
    max_packed_value: Vec<u8>,
    left: Option<Box<RangeTree<'c>>>,
    right: Option<Box<RangeTree<'c>>>,
    split: usize,
    component: Range<'c>,
}

impl RangeTree<'_> {
    fn below_max(&self, value: &[u8]) -> bool {
        let w = self.component.bytes_per_dim;
        (0..self.component.num_dims).all(|i| {
            let off = i.saturating_mul(w);
            cmp_at(value, off, &self.max_packed_value, off, w) != Ordering::Greater
        })
    }

    /// Whether the right subtree can hold `value` on the split dimension.
    fn right_reaches(&self, value: &[u8]) -> bool {
        let w = self.component.bytes_per_dim;
        let off = self.split.saturating_mul(w);
        cmp_at(value, off, &self.component.clause.lower_value, off, w) != Ordering::Less
    }

    fn matches(&self, packed: &[u8]) -> bool {
        if !self.below_max(packed) {
            return false;
        }
        if self.component.matches(packed) {
            return true;
        }
        if self.left.as_ref().is_some_and(|l| l.matches(packed)) {
            return true;
        }
        self.right
            .as_ref()
            .is_some_and(|r| self.right_reaches(packed) && r.matches(packed))
    }

    fn relate(&self, min: &[u8], max: &[u8]) -> Relation {
        if self.below_max(min) {
            let relation = self.component.relate(min, max);
            if relation != Relation::CellOutsideQuery {
                return relation;
            }
            if let Some(l) = &self.left {
                let relation = l.relate(min, max);
                if relation != Relation::CellOutsideQuery {
                    return relation;
                }
            }
            if let Some(r) = &self.right {
                if self.right_reaches(max) {
                    let relation = r.relate(min, max);
                    if relation != Relation::CellOutsideQuery {
                        return relation;
                    }
                }
            }
        }
        Relation::CellOutsideQuery
    }
}

/// `Relatable`: one range, or the tree of several.
#[derive(Debug)]
enum Relatable<'c> {
    One(Range<'c>),
    Tree(Box<RangeTree<'c>>),
}

impl<'c> Relatable<'c> {
    /// `rangeClauses.size() == 1 ? getRange(..) : createTree(..)`; `None`
    /// for no clauses (Java's `null` tree).
    fn new(clauses: &'c [RangeClause], num_dims: usize, bytes_per_dim: usize) -> Option<Self> {
        let ranges: Vec<Range<'c>> = clauses
            .iter()
            .map(|clause| Range {
                clause,
                num_dims,
                bytes_per_dim,
            })
            .collect();
        if let [one] = ranges.as_slice() {
            return Some(Relatable::One(*one));
        }
        let mut ranges = ranges;
        create_tree(&mut ranges, 0).map(Relatable::Tree)
    }

    fn matches(&self, packed: &[u8]) -> bool {
        match self {
            Relatable::One(r) => r.matches(packed),
            Relatable::Tree(t) => t.matches(packed),
        }
    }

    fn relate(&self, min: &[u8], max: &[u8]) -> Relation {
        match self {
            Relatable::One(r) => r.relate(min, max),
            Relatable::Tree(t) => t.relate(min, max),
        }
    }
}

/// `createTree(components, low, high, split, ...)` over `components` (the
/// whole slice is `low..=high`).
fn create_tree<'c>(components: &mut [Range<'c>], split: usize) -> Option<Box<RangeTree<'c>>> {
    let first = *components.first()?;
    let (num_dims, w) = (first.num_dims, first.bytes_per_dim);
    let mid = (components.len() - 1) / 2;
    if components.len() > 1 {
        let off = split.saturating_mul(w);
        components.select_nth_unstable_by(mid, |a, b| {
            cmp_at(&a.clause.lower_value, off, &b.clause.lower_value, off, w)
                .then_with(|| cmp_at(&a.clause.upper_value, off, &b.clause.upper_value, off, w))
        });
    }
    let component = components[mid];
    let next_split = if split + 1 == num_dims { 0 } else { split + 1 };
    let (low, rest) = components.split_at_mut(mid);
    let left = create_tree(low, next_split);
    let right = create_tree(&mut rest[1..], next_split);
    let mut max_packed_value = component.clause.upper_value.clone();
    for child in [&left, &right].into_iter().flatten() {
        for i in 0..num_dims {
            let off = i.saturating_mul(w);
            if cmp_at(&max_packed_value, off, &child.max_packed_value, off, w) == Ordering::Less {
                max_packed_value[off..off + w]
                    .copy_from_slice(&child.max_packed_value[off..off + w]);
            }
        }
    }
    Some(Box::new(RangeTree {
        max_packed_value,
        left,
        right,
        split,
        component,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn int(v: i32) -> Vec<u8> {
        ((v as u32) ^ 0x8000_0000).to_be_bytes().to_vec()
    }

    fn clause(lo: i32, hi: i32) -> RangeClause {
        RangeClause::new(int(lo), int(hi))
    }

    #[test]
    fn builder_checks() {
        assert!(MultiRangeQueryBuilder::new("f", 0, 1)
            .unwrap_err()
            .to_string()
            .contains("bytesPerDim should be a valid value"));
        assert!(MultiRangeQueryBuilder::new("f", 4, 0)
            .unwrap_err()
            .to_string()
            .contains("numDims should be a valid value"));
        let mut b = MultiRangeQueryBuilder::new("f", 4, 1).unwrap();
        b.add(int(1), int(2)).add_clause(clause(5, 6));
        let q = b.build();
        assert_eq!(q.field(), "f");
        assert_eq!(q.range_clauses.len(), 2);
    }

    #[test]
    fn merge_overlapping() {
        let merged = MultiRangeQuery::merge_overlapping_ranges(
            &[
                clause(5, 10),
                clause(-3, 0),
                clause(0, 2),
                clause(8, 9),
                clause(11, 12),
            ],
            4,
        )
        .unwrap();
        assert_eq!(merged, vec![clause(-3, 2), clause(5, 10), clause(11, 12)]);
        assert!(
            MultiRangeQuery::merge_overlapping_ranges(&[clause(1, 2), clause(3, 4)], 4).is_none()
        );
        assert!(MultiRangeQuery::merge_overlapping_ranges(&[clause(1, 2)], 4).is_none());
        let q = MultiRangeQuery {
            field: "f".into(),
            num_dims: 2,
            bytes_per_dim: 2,
            range_clauses: vec![],
        };
        assert!(
            q.rewrite(&[]).unwrap().is_none(),
            "two dimensions never merge"
        );
    }

    #[test]
    fn tree_relates_like_the_union() {
        let clauses: Vec<RangeClause> = (0..9).map(|i| clause(i * 10, i * 10 + 4)).collect();
        let tree = Relatable::new(&clauses, 1, 4).unwrap();
        assert!(matches!(tree, Relatable::Tree(_)));
        for v in -5..100 {
            assert_eq!(
                tree.matches(&int(v)),
                v >= 0 && v % 10 <= 4 && v < 90,
                "{v}"
            );
        }
        assert_eq!(tree.relate(&int(21), &int(23)), Relation::CellInsideQuery);
        assert_eq!(tree.relate(&int(18), &int(23)), Relation::CellCrossesQuery);
        assert_eq!(tree.relate(&int(5), &int(9)), Relation::CellOutsideQuery);
        assert_eq!(
            tree.relate(&int(200), &int(300)),
            Relation::CellOutsideQuery
        );
        assert!(Relatable::new(&[], 1, 4).is_none());
        let one = Relatable::new(&clauses[..1], 1, 4).unwrap();
        assert!(matches!(one, Relatable::One(_)));
        assert!(one.matches(&int(3)));
        assert_eq!(one.relate(&int(-1), &int(3)), Relation::CellCrossesQuery);
    }

    #[test]
    fn short_clause_is_an_error() {
        let q = MultiRangeQuery {
            field: "f".into(),
            num_dims: 1,
            bytes_per_dim: 4,
            range_clauses: vec![RangeClause::new(vec![1], int(2))],
        };
        let e = q.check_clauses().unwrap_err();
        assert!(e.to_string().contains("must be 4 bytes"), "{e}");
    }
}
