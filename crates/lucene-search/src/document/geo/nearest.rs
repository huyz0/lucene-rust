//! `NearestNeighbor` and `LatLonPoint.nearest`: the `n` points nearest an
//! origin, by a best-first walk over every segment's BKD cells.
//!
//! A port of `NearestNeighbor.nearest`: cells wait in a priority queue by
//! the approximate distance of their closest corner (`approxBestDistance`),
//! the nearest is expanded into its children (or, at a leaf, has its points
//! visited), and once `n` hits are held each better hit shrinks a bounding
//! box (`maybeUpdateBBox`) that prunes the cells and points that can no
//! longer compete. The cell queue is `java.util.PriorityQueue`'s binary
//! heap, ported operation for operation, so cells with equal distances are
//! expanded in Java's order; a point's distance is the haversine sort key.
//!
//! Rust shape: a cell owns its [`PointTreeNode`] (Java's cloned
//! `PointTree`); the queue holds indices into an arena of them, so the
//! heap moves `Copy` entries as Java's moves references.

use lucene_codecs::points::{IntersectVisitor, PointTreeNode, PointsReader, Relation};
use lucene_util::fixed_bit_set::FixedBitSet;
use lucene_util::geo::{GeoEncodingUtils, GeoUtils, Rectangle};
use lucene_util::sloppy_math;

use super::{geo, illegal};
use crate::document::{field_info, reader};
use crate::multi_segment::OpenSegment;
use crate::Result;

/// `NearestNeighbor.NearestHit`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NearestHit {
    /// The global doc id.
    pub doc_id: i32,
    /// `SloppyMath.haversinSortKey` of the hit's point to the origin.
    pub distance_sort_key: f64,
}

/// `LatLonPoint.nearest`'s `TopFieldDocs`.
#[derive(Debug, Clone, PartialEq)]
pub struct NearestHits {
    /// The documents with a point for the field, over every segment
    /// (`TotalHits.Relation.EQUAL_TO`).
    pub total_hits: i64,
    /// `(global doc id, distance in meters)`, nearest first. A document
    /// with several points can appear once per point.
    pub hits: Vec<(i32, f64)>,
}

/// A queued cell: `NearestNeighbor.Cell` (its tree node in the arena).
#[derive(Debug, Clone, Copy)]
struct Cell {
    reader: usize,
    node: usize,
    distance_sort_key: f64,
}

impl Cell {
    /// `compareTo`: `Double.compare` of the keys.
    #[inline]
    fn cmp(&self, other: &Cell) -> std::cmp::Ordering {
        self.distance_sort_key.total_cmp(&other.distance_sort_key)
    }
}

/// `java.util.PriorityQueue<Cell>` (natural order): `offer` and `poll` as
/// `siftUpComparable` / `siftDownComparable` do them.
#[derive(Debug, Default)]
struct JavaPriorityQueue {
    es: Vec<Cell>,
}

impl JavaPriorityQueue {
    fn offer(&mut self, key: Cell) {
        let mut k = self.es.len();
        self.es.push(key);
        while k > 0 {
            let parent = (k - 1) >> 1;
            let e = self.es[parent];
            if key.cmp(&e).is_ge() {
                break;
            }
            self.es[k] = e;
            k = parent;
        }
        self.es[k] = key;
    }

    fn poll(&mut self) -> Option<Cell> {
        let result = *self.es.first()?;
        let x = self.es.pop().expect("non-empty");
        let n = self.es.len();
        if n > 0 {
            let mut k = 0;
            let half = n >> 1;
            while k < half {
                let mut child = (k << 1) + 1;
                let mut c = self.es[child];
                let right = child + 1;
                if right < n && c.cmp(&self.es[right]).is_gt() {
                    child = right;
                    c = self.es[child];
                }
                if x.cmp(&c).is_le() {
                    break;
                }
                self.es[k] = c;
                k = child;
            }
            self.es[k] = x;
        }
        Some(result)
    }
}

/// A hit in the queue, ordered worst first (`NearestHitQueue.lessThan`:
/// the larger key, then the larger doc).
#[derive(Debug, Clone, Copy)]
struct Hit(NearestHit);

impl PartialEq for Hit {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}
impl Eq for Hit {}
impl PartialOrd for Hit {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Hit {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .distance_sort_key
            .total_cmp(&other.0.distance_sort_key)
            .then(self.0.doc_id.cmp(&other.0.doc_id))
    }
}

/// `NearestNeighbor.NearestVisitor`.
struct NearestVisitor<'l> {
    /// The field walked, for the corruption error.
    field: &'l str,
    cur_doc_base: i32,
    /// The current segment's `maxDoc`: a leaf naming a document at or past
    /// it (a corrupt `.kdd`) is an error, not a hit in another segment.
    cur_max_doc: i32,
    cur_live_docs: Option<&'l FixedBitSet>,
    top_n: usize,
    hit_queue: std::collections::BinaryHeap<Hit>,
    point_lat: f64,
    point_lon: f64,
    set_bottom_counter: i32,
    min_lon: f64,
    max_lon: f64,
    min_lat: f64,
    max_lat: f64,
    min_lon2: f64,
    error: Option<crate::Error>,
}

impl NearestVisitor<'_> {
    /// `maybeUpdateBBox()`.
    fn maybe_update_bbox(&mut self) {
        let c = self.set_bottom_counter;
        if c < 1024 || (c & 0x3F) == 0x3F {
            let Some(hit) = self.hit_queue.peek() else {
                return;
            };
            let radius = sloppy_math::haversin_meters_from_sort_key(hit.0.distance_sort_key);
            match Rectangle::from_point_distance(self.point_lat, self.point_lon, radius) {
                Ok(b) => {
                    self.min_lat = b.min_lat;
                    self.max_lat = b.max_lat;
                    if b.crosses_dateline() {
                        self.min_lon = f64::NEG_INFINITY;
                        self.max_lon = b.max_lon;
                        self.min_lon2 = b.min_lon;
                    } else {
                        self.min_lon = b.min_lon;
                        self.max_lon = b.max_lon;
                        self.min_lon2 = f64::INFINITY;
                    }
                }
                Err(e) => {
                    self.error.get_or_insert(geo(e));
                }
            }
        }
        self.set_bottom_counter = c.saturating_add(1);
    }
}

impl IntersectVisitor for NearestVisitor<'_> {
    fn compare(&mut self, min: &[u8], max: &[u8]) -> Relation {
        let cell_min_lat = GeoEncodingUtils::decode_latitude_bytes(min, 0);
        let cell_min_lon = GeoEncodingUtils::decode_longitude_bytes(min, 4);
        let cell_max_lat = GeoEncodingUtils::decode_latitude_bytes(max, 0);
        let cell_max_lon = GeoEncodingUtils::decode_longitude_bytes(max, 4);
        if cell_max_lat < self.min_lat
            || self.max_lat < cell_min_lat
            || ((cell_max_lon < self.min_lon || self.max_lon < cell_min_lon)
                && cell_max_lon < self.min_lon2)
        {
            return Relation::CellOutsideQuery;
        }
        Relation::CellCrossesQuery
    }

    /// Java's `visit(int)` throws: `compare` never answers "inside".
    fn visit(&mut self, _doc_id: i32) {}

    fn visit_with_value(&mut self, doc_id: i32, packed: &[u8]) {
        if doc_id < 0 || doc_id >= self.cur_max_doc {
            self.error
                .get_or_insert(super::out_of_segment(self.field, doc_id, self.cur_max_doc));
            return;
        }
        if self.cur_live_docs.is_some_and(|b| !b.get_doc(doc_id)) {
            return;
        }
        let lat = GeoEncodingUtils::decode_latitude_bytes(packed, 0);
        let lon = GeoEncodingUtils::decode_longitude_bytes(packed, 4);
        if lat < self.min_lat || lat > self.max_lat {
            return;
        }
        if (lon < self.min_lon || lon > self.max_lon) && lon < self.min_lon2 {
            return;
        }
        let key = sloppy_math::haversin_sort_key(self.point_lat, self.point_lon, lat, lon);
        let full = self.cur_doc_base.saturating_add(doc_id);
        if self.hit_queue.len() == self.top_n {
            let replace = self.hit_queue.peek().is_some_and(|top| {
                key < top.0.distance_sort_key
                    || (key == top.0.distance_sort_key && full < top.0.doc_id)
            });
            if replace {
                if let Some(mut top) = self.hit_queue.peek_mut() {
                    *top = Hit(NearestHit {
                        doc_id: full,
                        distance_sort_key: key,
                    });
                }
                self.maybe_update_bbox();
            }
        } else {
            self.hit_queue.push(Hit(NearestHit {
                doc_id: full,
                distance_sort_key: key,
            }));
        }
    }
}

/// `approxBestDistance(minPackedValue, maxPackedValue, pointLat, pointLon)`.
fn approx_best_distance(min: &[u8], max: &[u8], point_lat: f64, point_lon: f64) -> f64 {
    let min_lat = GeoEncodingUtils::decode_latitude_bytes(min, 0);
    let min_lon = GeoEncodingUtils::decode_longitude_bytes(min, 4);
    let max_lat = GeoEncodingUtils::decode_latitude_bytes(max, 0);
    let max_lon = GeoEncodingUtils::decode_longitude_bytes(max, 4);
    if point_lat >= min_lat && point_lat <= max_lat && point_lon >= min_lon && point_lon <= max_lon
    {
        return 0.0;
    }
    let key = sloppy_math::haversin_sort_key;
    let d1 = key(point_lat, point_lon, min_lat, min_lon);
    let d2 = key(point_lat, point_lon, min_lat, max_lon);
    let d3 = key(point_lat, point_lon, max_lat, max_lon);
    let d4 = key(point_lat, point_lon, max_lat, min_lon);
    d1.min(d2).min(d3.min(d4))
}

/// One segment's points for the walk.
struct Segment<'a> {
    points: PointsReader<'a>,
    field_number: i32,
    live_docs: Option<&'a FixedBitSet>,
    doc_base: i32,
    max_doc: i32,
}

/// `NearestNeighbor.nearest(pointLat, pointLon, readers, liveDocs,
/// docBases, n)` over `leaves`: the hits, nearest first.
fn nearest_hits(
    field: &str,
    point_lat: f64,
    point_lon: f64,
    segments: &[Segment<'_>],
    n: usize,
) -> Result<Vec<NearestHit>> {
    let mut cells = JavaPriorityQueue::default();
    let mut visitor = NearestVisitor {
        field,
        cur_doc_base: 0,
        cur_max_doc: 0,
        cur_live_docs: None,
        top_n: n,
        hit_queue: std::collections::BinaryHeap::with_capacity(n.min(1 << 16)),
        point_lat,
        point_lon,
        set_bottom_counter: 0,
        min_lon: f64::NEG_INFINITY,
        max_lon: f64::INFINITY,
        min_lat: f64::NEG_INFINITY,
        max_lat: f64::INFINITY,
        min_lon2: f64::INFINITY,
        error: None,
    };
    let mut arena: Vec<PointTreeNode> = Vec::new();
    let offer = |cells: &mut JavaPriorityQueue, arena: &mut Vec<PointTreeNode>, reader, node| {
        let n: &PointTreeNode = &node;
        let key = approx_best_distance(n.min_packed(), n.max_packed(), point_lat, point_lon);
        arena.push(node);
        cells.offer(Cell {
            reader,
            node: arena.len() - 1,
            distance_sort_key: key,
        });
    };
    for (i, s) in segments.iter().enumerate() {
        let root = s.points.point_tree(s.field_number)?;
        offer(&mut cells, &mut arena, i, root);
    }
    while let Some(cell) = cells.poll() {
        let s = &segments[cell.reader];
        let node = &arena[cell.node];
        if visitor.compare(node.min_packed(), node.max_packed()) == Relation::CellOutsideQuery {
            continue;
        }
        match s.points.point_tree_children(node)? {
            None => {
                visitor.cur_doc_base = s.doc_base;
                visitor.cur_max_doc = s.max_doc;
                visitor.cur_live_docs = s.live_docs;
                s.points.visit_leaf(node, &mut visitor)?;
                if let Some(e) = visitor.error.take() {
                    return Err(e);
                }
            }
            Some((left, right)) => {
                offer(&mut cells, &mut arena, cell.reader, left);
                offer(&mut cells, &mut arena, cell.reader, right);
            }
        }
    }
    Ok(visitor
        .hit_queue
        .into_sorted_vec()
        .into_iter()
        .map(|h| h.0)
        .collect())
}

/// `LatLonPoint.nearest(searcher, field, latitude, longitude, n)`: the `n`
/// nearest points of `field` to the origin over `leaves` (deleted documents
/// skipped), with their distances in meters.
///
/// # Errors
/// An invalid origin, or `n < 1`, with Java's message; an index that does
/// not decode.
pub fn nearest(
    leaves: &[OpenSegment<'_>],
    field: &str,
    latitude: f64,
    longitude: f64,
    n: i32,
) -> Result<NearestHits> {
    GeoUtils::check_latitude(latitude).map_err(geo)?;
    GeoUtils::check_longitude(longitude).map_err(geo)?;
    if n < 1 {
        return Err(illegal(format!("n must be at least 1; got {n}")));
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
        let points = reader(leaf)?.points_reader()?;
        let Some(values) = points.field(info.number) else {
            continue;
        };
        // Java decodes whatever the field holds (and fails on a short
        // value); a field that is not a geo point's shape is refused.
        super::point_queries::check_points_shape(field, values)?;
        total_hits = total_hits.saturating_add(i64::from(values.doc_count));
        segments.push(Segment {
            points,
            field_number: info.number,
            live_docs: leaf.live_docs,
            doc_base: leaf.doc_base,
            max_doc: reader(leaf)?.max_doc,
        });
    }
    let n = usize::try_from(n).unwrap_or(1);
    let hits = nearest_hits(field, latitude, longitude, &segments, n)?;
    Ok(NearestHits {
        total_hits,
        hits: hits
            .into_iter()
            .map(|h| {
                (
                    h.doc_id,
                    sloppy_math::haversin_meters_from_sort_key(h.distance_sort_key),
                )
            })
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cell(k: f64, node: usize) -> Cell {
        Cell {
            reader: 0,
            node,
            distance_sort_key: k,
        }
    }

    #[test]
    fn java_priority_queue_order() {
        let mut q = JavaPriorityQueue::default();
        assert!(q.poll().is_none());
        for (i, k) in [5.0, 1.0, 3.0, 1.0, 0.0, 9.0, 3.0].into_iter().enumerate() {
            q.offer(cell(k, i));
        }
        let mut keys = Vec::new();
        let mut nodes = Vec::new();
        while let Some(c) = q.poll() {
            keys.push(c.distance_sort_key);
            nodes.push(c.node);
        }
        assert_eq!(keys, vec![0.0, 1.0, 1.0, 3.0, 3.0, 5.0, 9.0]);
        // java.util.PriorityQueue's tie order for this offer sequence.
        assert_eq!(nodes, vec![4, 1, 3, 6, 2, 0, 5]);
    }

    #[test]
    fn approx_best_distance_is_zero_inside() {
        let enc = |lat, lon| lucene_index::document::LatLonPoint::encode(lat, lon).unwrap();
        assert_eq!(
            approx_best_distance(&enc(-1.0, -1.0), &enc(1.0, 1.0), 0.0, 0.0),
            0.0
        );
        let d = approx_best_distance(&enc(10.0, 10.0), &enc(11.0, 11.0), 0.0, 0.0);
        let corner = sloppy_math::haversin_sort_key(
            0.0,
            0.0,
            GeoEncodingUtils::decode_latitude_bytes(&enc(10.0, 10.0), 0),
            GeoEncodingUtils::decode_longitude_bytes(&enc(10.0, 10.0), 4),
        );
        assert_eq!(d, corner);
    }

    #[test]
    fn validation() {
        let e = nearest(&[], "f", 0.0, 0.0, 0).unwrap_err();
        assert!(e.to_string().contains("n must be at least 1; got 0"), "{e}");
        assert!(nearest(&[], "f", 91.0, 0.0, 1).is_err());
        assert!(nearest(&[], "f", 0.0, 181.0, 1).is_err());
        let r = nearest(&[], "f", 0.0, 0.0, 3).unwrap();
        assert_eq!(r.total_hits, 0);
        assert!(r.hits.is_empty());
    }

    #[test]
    fn hit_order_and_bbox() {
        let a = Hit(NearestHit {
            doc_id: 1,
            distance_sort_key: 0.5,
        });
        let b = Hit(NearestHit {
            doc_id: 2,
            distance_sort_key: 0.5,
        });
        assert!(a < b);
        assert_eq!(a, a);
        let mut v = NearestVisitor {
            field: "p",
            cur_doc_base: 0,
            cur_max_doc: 10,
            cur_live_docs: None,
            top_n: 1,
            hit_queue: std::collections::BinaryHeap::new(),
            point_lat: 0.0,
            point_lon: 179.9,
            set_bottom_counter: 0,
            min_lon: f64::NEG_INFINITY,
            max_lon: f64::INFINITY,
            min_lat: f64::NEG_INFINITY,
            max_lat: f64::INFINITY,
            min_lon2: f64::INFINITY,
            error: None,
        };
        v.maybe_update_bbox();
        assert_eq!(v.set_bottom_counter, 0, "an empty queue sets no box");
        let enc = |lat, lon| lucene_index::document::LatLonPoint::encode(lat, lon).unwrap();
        v.visit(0);
        v.visit_with_value(3, &enc(0.0, 170.0));
        v.visit_with_value(4, &enc(0.0, -179.95));
        assert_eq!(v.hit_queue.peek().unwrap().0.doc_id, 4);
        assert_eq!(v.min_lon, f64::NEG_INFINITY, "the box crosses the dateline");
        assert_eq!(
            v.compare(&enc(10.0, 0.0), &enc(20.0, 10.0)),
            Relation::CellOutsideQuery
        );
        assert_eq!(
            v.compare(&enc(-1.0, 179.0), &enc(1.0, 180.0)),
            Relation::CellCrossesQuery
        );
        v.visit_with_value(5, &enc(50.0, 0.0));
        assert_eq!(v.hit_queue.len(), 1);
        // A leaf naming a document past this segment's `maxDoc` (a corrupt
        // `.kdd`) is an error, not a hit at `docBase + doc` in a later
        // segment; nor is a negative one.
        assert!(v.error.is_none());
        v.cur_doc_base = 100;
        v.visit_with_value(10, &enc(0.0, -179.95));
        v.visit_with_value(-1, &enc(0.0, -179.95));
        assert_eq!(v.hit_queue.peek().unwrap().0.doc_id, 4, "nothing added");
        let e = v.error.take().expect("recorded");
        assert!(
            e.to_string()
                .contains("points of field p name document 10, outside the segment's 0..10"),
            "{e}"
        );
    }
}
