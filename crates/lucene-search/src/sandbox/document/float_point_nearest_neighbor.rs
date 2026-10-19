//! `FloatPointNearestNeighbor`: the `topN` points of a `FloatPoint` field
//! nearest an origin by Euclidean distance, over every segment's BKD cells.
//!
//! A port of `FloatPointNearestNeighbor.nearest`: cells wait in a queue by
//! the squared distance from the origin to their bounding box, the nearest
//! is expanded into its children (only those no farther than the current
//! `topN`-th hit) or, at a leaf, has its points visited, and the walk stops
//! at the first cell farther than that hit. A point's squared distance is
//! summed in `double`s dimension by dimension, abandoned as soon as it
//! passes the bottom hit's; hits tie on the larger document losing. The
//! answer is `TopFieldDocs`-shaped: each hit's distance as
//! `(float) Math.sqrt(distanceSquared)`, and the total of every segment's
//! `getDocCount()` (deletions included, as Java counts them).
//!
//! The walk's order does not decide the hits -- pruning only drops cells
//! and points that cannot beat the bottom hit -- so the cell queue is a
//! plain [`BinaryHeap`]; Java's `java.util.PriorityQueue` may expand
//! equal-distance cells in another order, to the same answer.
//!
//! Deliberate difference: Java decodes `origin.length` dimensions of
//! whatever the field holds and fails on a short value; an origin with more
//! dimensions than the field indexes, or a field whose dimensions are not
//! four bytes wide, is refused with an error here.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use lucene_codecs::points::{IntersectVisitor, PointTreeNode, PointsReader, Relation};
use lucene_index::document::FloatPoint;
use lucene_util::fixed_bit_set::FixedBitSet;

use super::illegal;
use crate::document::{field_info, reader};
use crate::multi_segment::OpenSegment;
use crate::Result;

/// `FloatPointNearestNeighbor.nearest`'s `TopFieldDocs`.
#[derive(Debug, Clone, PartialEq)]
pub struct NearestNeighborHits {
    /// Every segment's `getDocCount()` for the field (`EQUAL_TO`).
    pub total_hits: i64,
    /// `(global doc id, distance)`, nearest first, the larger document last
    /// on ties. A document with several points can appear once per point.
    pub hits: Vec<(i32, f32)>,
}

/// `FloatPointNearestNeighbor`.
#[derive(Debug, Clone, Copy)]
pub struct FloatPointNearestNeighbor;

/// `NearestHit`, ordered worst first: the larger distance, then the larger
/// document (the hit queue's comparator, reversed for a max-heap).
#[derive(Debug, Clone, Copy)]
struct NearestHit {
    doc_id: i32,
    distance_squared: f64,
}

impl PartialEq for NearestHit {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}
impl Eq for NearestHit {}
impl PartialOrd for NearestHit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for NearestHit {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance_squared
            .total_cmp(&other.distance_squared)
            .then(self.doc_id.cmp(&other.doc_id))
    }
}

/// A queued `Cell`, nearest first out of the max-heap.
#[derive(Debug, Clone, Copy)]
struct Cell {
    distance_squared: f64,
    reader: usize,
    node: usize,
}

impl PartialEq for Cell {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}
impl Eq for Cell {}
impl PartialOrd for Cell {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Cell {
    fn cmp(&self, other: &Self) -> Ordering {
        // `Double.compare`, reversed; then the arena slot, for a total order.
        other
            .distance_squared
            .total_cmp(&self.distance_squared)
            .then(other.node.cmp(&self.node))
    }
}

/// `NearestVisitor`.
struct NearestVisitor<'o> {
    field: &'o str,
    cur_doc_base: i32,
    cur_max_doc: i32,
    cur_live_docs: Option<&'o FixedBitSet>,
    top_n: usize,
    hit_queue: BinaryHeap<NearestHit>,
    origin: &'o [f32],
    bottom_nearest_distance_squared: f64,
    bottom_nearest_distance_doc: i32,
    error: Option<crate::Error>,
}

impl NearestVisitor<'_> {
    /// `updateBottomNearestDistance()`.
    fn update_bottom(&mut self) {
        if let Some(b) = self.hit_queue.peek() {
            self.bottom_nearest_distance_squared = b.distance_squared;
            self.bottom_nearest_distance_doc = b.doc_id;
        }
    }
}

impl IntersectVisitor for NearestVisitor<'_> {
    fn compare(&mut self, min: &[u8], max: &[u8]) -> Relation {
        if self.hit_queue.len() == self.top_n
            && point_to_rectangle_distance_squared(min, max, self.origin)
                > self.bottom_nearest_distance_squared
        {
            return Relation::CellOutsideQuery;
        }
        Relation::CellCrossesQuery
    }

    /// Java's `visit(int)` throws: `compare` never answers "inside".
    fn visit(&mut self, _doc_id: i32) {}

    fn visit_with_value(&mut self, doc_id: i32, packed: &[u8]) {
        if doc_id < 0 || doc_id >= self.cur_max_doc {
            self.error.get_or_insert_with(|| {
                illegal(format!(
                    "points of field {} name document {doc_id}, outside the segment's 0..{}",
                    self.field, self.cur_max_doc
                ))
            });
            return;
        }
        if self.cur_live_docs.is_some_and(|b| !b.get_doc(doc_id)) {
            return;
        }
        let mut distance_squared = 0.0f64;
        for (d, &o) in self.origin.iter().enumerate() {
            let off = d.saturating_mul(4);
            let diff = f64::from(FloatPoint::decode_dimension(&packed[off..])) - f64::from(o);
            distance_squared += diff * diff;
            if distance_squared > self.bottom_nearest_distance_squared {
                return;
            }
        }
        let full_doc_id = self.cur_doc_base.wrapping_add(doc_id);
        let hit = NearestHit {
            doc_id: full_doc_id,
            distance_squared,
        };
        if self.hit_queue.len() == self.top_n {
            if distance_squared == self.bottom_nearest_distance_squared
                && full_doc_id > self.bottom_nearest_distance_doc
            {
                return;
            }
            // `poll()` then `offer(bottom)`.
            self.hit_queue.pop();
            self.hit_queue.push(hit);
            self.update_bottom();
        } else {
            self.hit_queue.push(hit);
            if self.hit_queue.len() == self.top_n {
                self.update_bottom();
            }
        }
    }
}

/// `pointToRectangleDistanceSquared(min, max, value)`.
fn point_to_rectangle_distance_squared(min: &[u8], max: &[u8], value: &[f32]) -> f64 {
    let mut sum = 0.0f64;
    for (i, &v) in value.iter().enumerate() {
        let off = i.saturating_mul(4);
        let lo = f64::from(FloatPoint::decode_dimension(&min[off..]));
        if f64::from(v) < lo {
            let diff = lo - f64::from(v);
            sum += diff * diff;
            continue;
        }
        let hi = f64::from(FloatPoint::decode_dimension(&max[off..]));
        if f64::from(v) > hi {
            let diff = hi - f64::from(v);
            sum += diff * diff;
        }
    }
    sum
}

/// One segment's points for the walk.
struct Segment<'a> {
    points: PointsReader<'a>,
    field_number: i32,
    live_docs: Option<&'a FixedBitSet>,
    doc_base: i32,
    max_doc: i32,
}

impl FloatPointNearestNeighbor {
    /// `nearest(searcher, field, topN, origin...)` over `leaves`.
    ///
    /// # Errors
    /// `topN < 1`, with Java's message; a field whose points are not four
    /// bytes a dimension, or an origin with more dimensions than the field
    /// indexes; an index that does not decode.
    pub fn nearest(
        leaves: &[OpenSegment<'_>],
        field: &str,
        top_n: i32,
        origin: &[f32],
    ) -> Result<NearestNeighborHits> {
        if top_n < 1 {
            return Err(illegal(format!("topN must be at least 1; got {top_n}")));
        }
        let mut segments = Vec::new();
        let mut total_hits = 0i64;
        for leaf in leaves {
            let Some(info) = field_info(leaf, field)? else {
                continue;
            };
            if info.point_dimension_count == 0 {
                continue;
            }
            let r = reader(leaf)?;
            let points = r.points_reader()?;
            let Some(values) = points.field(info.number) else {
                continue;
            };
            if values.bytes_per_dim != 4
                || usize::try_from(values.num_index_dims).map_or(true, |n| origin.len() > n)
            {
                return Err(illegal(format!(
                    "field=\"{field}\" holds points of {} indexed dimensions of {} bytes, not \
                     the {} float dimensions of the origin",
                    values.num_index_dims,
                    values.bytes_per_dim,
                    origin.len()
                )));
            }
            total_hits = total_hits.saturating_add(i64::from(values.doc_count));
            segments.push(Segment {
                points,
                field_number: info.number,
                live_docs: leaf.live_docs,
                doc_base: leaf.doc_base,
                max_doc: r.max_doc,
            });
        }
        let top_n = usize::try_from(top_n).unwrap_or(1);
        let hits = nearest_hits(field, &segments, top_n, origin)?;
        Ok(NearestNeighborHits {
            total_hits,
            hits: hits
                .into_iter()
                .map(|h| (h.doc_id, h.distance_squared.sqrt() as f32))
                .collect(),
        })
    }
}

/// `nearest(readers, liveDocs, docBases, topN, origin)`: the hits, nearest
/// first.
fn nearest_hits(
    field: &str,
    segments: &[Segment<'_>],
    top_n: usize,
    origin: &[f32],
) -> Result<Vec<NearestHit>> {
    let mut visitor = NearestVisitor {
        field,
        cur_doc_base: 0,
        cur_max_doc: 0,
        cur_live_docs: None,
        top_n,
        hit_queue: BinaryHeap::with_capacity(top_n.min(1 << 16)),
        origin,
        bottom_nearest_distance_squared: f64::INFINITY,
        bottom_nearest_distance_doc: i32::MAX,
        error: None,
    };
    let mut arena: Vec<PointTreeNode> = Vec::new();
    let mut cells: BinaryHeap<Cell> = BinaryHeap::new();
    let offer = |cells: &mut BinaryHeap<Cell>,
                 arena: &mut Vec<PointTreeNode>,
                 reader: usize,
                 node: PointTreeNode,
                 distance_squared: f64| {
        arena.push(node);
        cells.push(Cell {
            distance_squared,
            reader,
            node: arena.len() - 1,
        });
    };
    for (i, s) in segments.iter().enumerate() {
        let root = s.points.point_tree(s.field_number)?;
        let d = point_to_rectangle_distance_squared(root.min_packed(), root.max_packed(), origin);
        offer(&mut cells, &mut arena, i, root, d);
    }
    while let Some(cell) = cells.pop() {
        if cell.distance_squared > visitor.bottom_nearest_distance_squared {
            break;
        }
        let s = &segments[cell.reader];
        let node = &arena[cell.node];
        match s.points.point_tree_children(node)? {
            None => {
                // A leaf: visit its points.
                visitor.cur_doc_base = s.doc_base;
                visitor.cur_max_doc = s.max_doc;
                visitor.cur_live_docs = s.live_docs;
                s.points.visit_leaf(node, &mut visitor)?;
                if let Some(e) = visitor.error.take() {
                    return Err(e);
                }
            }
            Some((left, right)) => {
                for child in [left, right] {
                    let d = point_to_rectangle_distance_squared(
                        child.min_packed(),
                        child.max_packed(),
                        origin,
                    );
                    if d <= visitor.bottom_nearest_distance_squared {
                        offer(&mut cells, &mut arena, cell.reader, child, d);
                    }
                }
            }
        }
    }
    // `poll()` into the array from its end: nearest first.
    Ok(visitor.hit_queue.into_sorted_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hits_and_cells_order_by_distance_then_slot() {
        let hit = |doc_id, distance_squared| NearestHit {
            doc_id,
            distance_squared,
        };
        assert!(hit(1, 2.0) == hit(1, 2.0));
        assert!(hit(1, 2.0) < hit(2, 2.0) && hit(9, 1.0) < hit(1, 2.0));
        let cell = |distance_squared, node| Cell {
            distance_squared,
            reader: 0,
            node,
        };
        assert!(cell(1.0, 3) == cell(1.0, 3));
        // The max-heap pops the nearest cell, then the earlier slot.
        assert!(cell(1.0, 3) > cell(2.0, 0) && cell(1.0, 0) > cell(1.0, 3));
    }

    #[test]
    fn rectangle_distance_sums_the_outside_dimensions() {
        let pack = |v: &[f32]| FloatPoint::pack(v).unwrap();
        let (min, max) = (pack(&[0.0, 0.0]), pack(&[1.0, 1.0]));
        assert_eq!(
            point_to_rectangle_distance_squared(&min, &max, &[0.5, 0.5]),
            0.0
        );
        assert_eq!(
            point_to_rectangle_distance_squared(&min, &max, &[-1.0, 3.0]),
            5.0
        );
    }
}
