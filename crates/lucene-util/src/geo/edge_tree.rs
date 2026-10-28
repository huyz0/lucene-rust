//! Port of `org.apache.lucene.geo.EdgeTree`: an interval tree over a
//! polygon's or line's edges, keyed on their y-range, used by `Polygon2D`
//! and `Line2D`.
//!
//! # Layout
//!
//! Java links one heap node per edge. Here Java's tree (the edges sorted
//! by `(low, max)`, the node of the sorted range `[lo, hi)` at
//! `mid = (lo + hi - 1) >> 1`) is one array in pre-order: a node's left
//! child is the next element and only the right child's offset is stored.
//!
//! # Queries
//!
//! Every query's answer depends only on the edges whose own y-range meets
//! the query's y-range:
//!
//! - each per-edge test is false for any other edge (`containsPnPoly`'s
//!   crossing test and `isPointOnLine` need the point's `y` inside the
//!   edge's range; the `crosses*` tests start with an `outside` check, or
//!   an endpoint inside the box, that fails for it);
//! - Java's walk reaches every edge that meets the range (it prunes only a
//!   subtree whose `max` is below `min_y`, and a right subtree whose edges
//!   all start above `max_y`);
//! - the answer combines the per-edge results in an order-free way: an OR,
//!   or for `containsPnPoly` the XOR of the crossings unless an edge holds
//!   the point (Java returns `ON_EDGE` as soon as one does, whatever it
//!   XORed before).
//!
//! So instead of walking the tree, a query looks the edges up in
//! [`Slabs`]: the polygon's y-extent cut into equal slabs, each listing the
//! edges whose range meets it. A point lies in one slab, whose few edges
//! are all a contains test reads -- no root-to-leaf path, no pruning tests.
//! The tree walk ([`walk`]) remains for a query whose y-bounds are NaN or
//! inverted, where Java's pruning, not the edges, decides the answer.
#![allow(clippy::too_many_arguments)]

use super::geo_utils::GeoUtils;
use super::java_max;
use super::rectangle::Rectangle;

/// One edge, with its node's subtree `max` and link.
#[derive(Debug, Clone, Copy)]
struct Edge {
    // X-Y pair (in original order) of the two vertices
    y1: f64,
    y2: f64,
    x1: f64,
    x2: f64,
    /// The edge's own min y.
    low: f64,
    /// The subtree's max y (Java pulls it up from the children).
    max: f64,
    /// The right child's offset from this node; 0 for none.
    right: u32,
    /// Whether the left child (always the next element) exists.
    has_left: bool,
}

/// Port of `EdgeTree`: the root edge's first vertex, the tree, and the
/// slab index over it.
#[derive(Debug, Clone)]
pub(crate) struct EdgeTree {
    /// The root edge's first vertex (`tree.x1`/`tree.y1` in Java).
    pub(crate) x1: f64,
    pub(crate) y1: f64,
    edges: Box<[Edge]>,
    slabs: Slabs,
}

/// The edges by y-slab: slab `b` covers the `y` with `slab(y) == b` and
/// lists (as indices into the tree) the edges whose own range meets it.
#[derive(Debug, Clone)]
struct Slabs {
    y0: f64,
    /// Slabs per unit of y.
    scale: f64,
    /// The last slab's index.
    last: usize,
    /// `entries[start[b]..start[b + 1]]` are slab `b`'s edges.
    start: Box<[u32]>,
    entries: Box<[u32]>,
}

/// How many slab entries per edge an index may hold: an edge spanning
/// several slabs is listed in each, so the slab count shrinks until the
/// copies fit this budget.
const MAX_ENTRIES_PER_EDGE: usize = 8;

impl Slabs {
    /// The slab holding `y`: non-decreasing in `y` (a difference and a
    /// product by a positive constant round monotonically), so an edge
    /// listed in every slab from `slab(low)` to `slab(high)` is in the slab
    /// of every `y` between.
    #[inline(always)]
    fn slab(&self, y: f64) -> usize {
        let t = (y - self.y0) * self.scale;
        if t >= self.last as f64 {
            self.last
        } else if t >= 0.0 {
            t as usize
        } else {
            0
        }
    }

    /// Slab `b`'s edges.
    #[inline(always)]
    fn edges(&self, b: usize) -> &[u32] {
        &self.entries[self.start[b] as usize..self.start[b + 1] as usize]
    }

    fn build(edges: &[Edge]) -> Slabs {
        let n = edges.len();
        let ranges: Vec<(f64, f64)> = edges
            .iter()
            .map(|e| (e.low, java_max(e.y1, e.y2)))
            .collect();
        let y0 = ranges.iter().fold(f64::INFINITY, |m, r| m.min(r.0));
        let y_max = ranges.iter().fold(f64::NEG_INFINITY, |m, r| m.max(r.1));
        let height = y_max - y0;
        let mut count = n.next_power_of_two();
        let mut slabs = loop {
            let s = Slabs {
                y0,
                scale: if count > 1 && height > 0.0 && height.is_finite() {
                    count as f64 / height
                } else {
                    0.0
                },
                last: count - 1,
                start: Box::new([]),
                entries: Box::new([]),
            };
            let total: usize = ranges
                .iter()
                .map(|&(lo, hi)| s.slab(hi) - s.slab(lo) + 1)
                .sum();
            if count == 1 || total <= MAX_ENTRIES_PER_EDGE * n {
                break s;
            }
            count /= 2;
        };
        // counting sort of (slab, edge) pairs
        let mut start = vec![0u32; count + 1];
        for &(lo, hi) in &ranges {
            for b in slabs.slab(lo)..=slabs.slab(hi) {
                start[b + 1] += 1;
            }
        }
        for b in 0..count {
            start[b + 1] += start[b];
        }
        let mut fill: Vec<u32> = start[..count].to_vec();
        let mut entries = vec![0u32; start[count] as usize];
        for (i, &(lo, hi)) in ranges.iter().enumerate() {
            for b in slabs.slab(lo)..=slabs.slab(hi) {
                entries[fill[b] as usize] = i as u32;
                fill[b] += 1;
            }
        }
        slabs.start = start.into_boxed_slice();
        slabs.entries = entries.into_boxed_slice();
        slabs
    }
}

/// The node of the non-empty half-open range `[lo, hi)`.
#[inline(always)]
fn mid(lo: usize, hi: usize) -> usize {
    (lo + hi - 1) >> 1
}

impl EdgeTree {
    /// `contains(x, y)`: point in polygon, boundary included.
    #[inline]
    pub(crate) fn contains(&self, x: f64, y: f64) -> bool {
        // NaN: Java's root test `y <= max` fails, so FALSE
        if y.is_nan() {
            return false;
        }
        let mut res = false;
        for &i in self.slabs.edges(self.slabs.slab(y)) {
            match pn_poly_edge(&self.edges[i as usize], x, y) {
                PnPoly::OnEdge => return true,
                PnPoly::Crosses => res = !res,
                PnPoly::Miss => {}
            }
        }
        res
    }

    /// `isPointOnLine(x, y)`.
    pub(crate) fn is_point_on_line(&self, x: f64, y: f64) -> bool {
        if y.is_nan() {
            return false;
        }
        self.slabs
            .edges(self.slabs.slab(y))
            .iter()
            .any(|&i| point_on_edge(&self.edges[i as usize], x, y))
    }

    /// Whether `f` holds for an edge meeting `[min_y, max_y]` (Java's walk,
    /// whose pruning decides alone when a bound is NaN or the range is
    /// inverted).
    #[inline]
    fn any_edge(&self, min_y: f64, max_y: f64, f: impl Fn(&Edge) -> bool) -> bool {
        if min_y.is_nan() || max_y.is_nan() || min_y > max_y {
            return walk(&self.edges, min_y, max_y, f);
        }
        let s = &self.slabs;
        let (b0, b1) = (s.slab(min_y), s.slab(max_y));
        if b0 == b1 {
            return s.edges(b0).iter().any(|&i| f(&self.edges[i as usize]));
        }
        // an edge listed in several of the slabs is tested in the first
        (b0..=b1).any(|b| {
            s.edges(b).iter().any(|&i| {
                let e = &self.edges[i as usize];
                s.slab(e.low).max(b0) == b && f(e)
            })
        })
    }

    /// `crossesTriangle`.
    pub(crate) fn crosses_triangle(
        &self,
        min_x: f64,
        max_x: f64,
        min_y: f64,
        max_y: f64,
        ax: f64,
        ay: f64,
        bx: f64,
        by: f64,
        cx: f64,
        cy: f64,
        include_boundary: bool,
    ) -> bool {
        self.any_edge(min_y, max_y, |e| {
            let (dy, ey, dx, ex) = (e.y1, e.y2, e.x1, e.x2);
            let outside = (dy < min_y && ey < min_y)
                || (dy > max_y && ey > max_y)
                || (dx < min_x && ex < min_x)
                || (dx > max_x && ex > max_x);
            if outside {
                return false;
            }
            if include_boundary {
                GeoUtils::line_crosses_line_with_boundary(dx, dy, ex, ey, ax, ay, bx, by)
                    || GeoUtils::line_crosses_line_with_boundary(dx, dy, ex, ey, bx, by, cx, cy)
                    || GeoUtils::line_crosses_line_with_boundary(dx, dy, ex, ey, cx, cy, ax, ay)
            } else {
                GeoUtils::line_crosses_line(dx, dy, ex, ey, ax, ay, bx, by)
                    || GeoUtils::line_crosses_line(dx, dy, ex, ey, bx, by, cx, cy)
                    || GeoUtils::line_crosses_line(dx, dy, ex, ey, cx, cy, ax, ay)
            }
        })
    }

    /// `crossesBox`.
    pub(crate) fn crosses_box(
        &self,
        min_x: f64,
        max_x: f64,
        min_y: f64,
        max_y: f64,
        include_boundary: bool,
    ) -> bool {
        self.any_edge(min_y, max_y, |e| {
            let (cy, dy, cx, dx) = (e.y1, e.y2, e.x1, e.x2);
            // optimization: see if either end of the line segment is contained by the rectangle
            if Rectangle::contains_point(cy, cx, min_y, max_y, min_x, max_x)
                || Rectangle::contains_point(dy, dx, min_y, max_y, min_x, max_x)
            {
                return true;
            }
            let outside = (cy < min_y && dy < min_y)
                || (cy > max_y && dy > max_y)
                || (cx < min_x && dx < min_x)
                || (cx > max_x && dx > max_x);
            if outside {
                return false;
            }
            if include_boundary {
                GeoUtils::line_crosses_line_with_boundary(
                    cx, cy, dx, dy, min_x, min_y, max_x, min_y,
                ) || GeoUtils::line_crosses_line_with_boundary(
                    cx, cy, dx, dy, max_x, min_y, max_x, max_y,
                ) || GeoUtils::line_crosses_line_with_boundary(
                    cx, cy, dx, dy, max_x, max_y, min_x, max_y,
                ) || GeoUtils::line_crosses_line_with_boundary(
                    cx, cy, dx, dy, min_x, max_y, min_x, min_y,
                )
            } else {
                GeoUtils::line_crosses_line(cx, cy, dx, dy, min_x, min_y, max_x, min_y)
                    || GeoUtils::line_crosses_line(cx, cy, dx, dy, max_x, min_y, max_x, max_y)
                    || GeoUtils::line_crosses_line(cx, cy, dx, dy, max_x, max_y, min_x, max_y)
                    || GeoUtils::line_crosses_line(cx, cy, dx, dy, min_x, max_y, min_x, min_y)
            }
        })
    }

    /// `crossesLine`.
    pub(crate) fn crosses_line(
        &self,
        min_x: f64,
        max_x: f64,
        min_y: f64,
        max_y: f64,
        a2x: f64,
        a2y: f64,
        b2x: f64,
        b2y: f64,
        include_boundary: bool,
    ) -> bool {
        self.any_edge(min_y, max_y, |e| {
            let (a1x, a1y, b1x, b1y) = (e.x1, e.y1, e.x2, e.y2);
            let outside = (a1y < min_y && b1y < min_y)
                || (a1y > max_y && b1y > max_y)
                || (a1x < min_x && b1x < min_x)
                || (a1x > max_x && b1x > max_x);
            if outside {
                return false;
            }
            if include_boundary {
                GeoUtils::line_crosses_line_with_boundary(a1x, a1y, b1x, b1y, a2x, a2y, b2x, b2y)
            } else {
                GeoUtils::line_crosses_line(a1x, a1y, b1x, b1y, a2x, a2y, b2x, b2y)
            }
        })
    }

    /// `createTree(x, y)`: sorts the edges by `(low, max)` and builds a
    /// balanced tree. Panics on fewer than two points (Java would throw on
    /// the empty array; every caller has validated its geometry first).
    pub(crate) fn create_tree(x: &[f64], y: &[f64]) -> EdgeTree {
        assert!(
            x.len() >= 2 && y.len() >= x.len(),
            "a line has at least one edge"
        );
        let mut edges: Vec<Edge> = x
            .windows(2)
            .zip(y.windows(2))
            .map(|(xs, ys)| {
                let (x1, x2, y1, y2) = (xs[0], xs[1], ys[0], ys[1]);
                Edge {
                    y1,
                    y2,
                    x1,
                    x2,
                    low: super::java_min(y1, y2),
                    max: java_max(y1, y2),
                    right: 0,
                    has_left: false,
                }
            })
            .collect();
        // Arrays.sort over objects is a stable merge sort, as is sort_by;
        // ties keep their input order, which decides the root edge.
        edges.sort_by(|l, r| l.low.total_cmp(&r.low).then(l.max.total_cmp(&r.max)));
        let n = edges.len();
        assert!(u32::try_from(n).is_ok(), "too many edges");
        let mut tree = Vec::with_capacity(n);
        build_pre_order(&edges, 0, n, &mut tree);
        let slabs = Slabs::build(&tree);
        EdgeTree {
            x1: tree[0].x1,
            y1: tree[0].y1,
            edges: tree.into_boxed_slice(),
            slabs,
        }
    }
}

/// `createTreeRange(edges, lo, hi - 1)`: appends the subtree of the sorted
/// range `[lo, hi)` to `out` in pre-order, pulling each node's max up from
/// its children (left, then right, as Java folds them); returns the
/// subtree's max.
fn build_pre_order(sorted: &[Edge], lo: usize, hi: usize, out: &mut Vec<Edge>) -> f64 {
    let m = mid(lo, hi);
    let at = out.len();
    out.push(sorted[m]);
    let mut max = sorted[m].max;
    if lo < m {
        out[at].has_left = true;
        max = java_max(max, build_pre_order(sorted, lo, m, out));
    }
    if m + 1 < hi {
        out[at].right = (out.len() - at) as u32;
        max = java_max(max, build_pre_order(sorted, m + 1, hi, out));
    }
    out[at].max = max;
    max
}

/// Java's recursive walk for a query whose y-range is `[min_y, max_y]`, in
/// its order (a node, its left subtree, its right subtree): a node is
/// examined when `min_y <= max` of its subtree, its right subtree entered
/// only when `max_y >= low`. True as soon as `f` is.
fn walk(edges: &[Edge], min_y: f64, max_y: f64, f: impl Fn(&Edge) -> bool) -> bool {
    // At most one pending right subtree per level, and a tree of < 2^32
    // edges is at most 33 levels deep.
    let mut stack = [0usize; 40];
    let mut sp = 0;
    let mut i = 0;
    loop {
        let e = &edges[i];
        if min_y <= e.max {
            if f(e) {
                return true;
            }
            if e.right != 0 && max_y >= e.low {
                stack[sp] = i + e.right as usize;
                sp += 1;
            }
            if e.has_left {
                i += 1;
                continue;
            }
        }
        if sp == 0 {
            return false;
        }
        sp -= 1;
        i = stack[sp];
    }
}

/// One edge's part of `containsPnPoly`.
enum PnPoly {
    Miss,
    Crosses,
    OnEdge,
}

/// One edge's part of `containsPnPoly` (W. Randolph Franklin's crossing
/// test, with an on-edge answer), Java's comparisons as written.
#[inline(always)]
fn pn_poly_edge(e: &Edge, x: f64, y: f64) -> PnPoly {
    let (y1, y2, x1, x2) = (e.y1, e.y2, e.x1, e.x2);
    // `&` rather than `&&`: plain comparisons, one branch instead of a
    // chain of unpredictable ones on the hot path.
    if (y == y1) & (y == y2) | (((y <= y1) & (y >= y2)) != ((y >= y1) & (y <= y2))) {
        if (x == x1 && x == x2)
            || (((x <= x1) & (x >= x2)) != ((x >= x1) & (x <= x2))
                && GeoUtils::orient(x1, y1, x2, y2, x, y) == 0)
        {
            return PnPoly::OnEdge;
        } else if (y1 > y) != (y2 > y) && x < (x2 - x1) * (y - y1) / (y2 - y1) + x1 {
            return PnPoly::Crosses;
        }
    }
    PnPoly::Miss
}

/// One edge's part of `isPointOnLine`.
#[inline(always)]
fn point_on_edge(e: &Edge, x: f64, y: f64) -> bool {
    let (a1x, a1y, b1x, b1y) = (e.x1, e.y1, e.x2, e.y2);
    let outside = (a1y < y && b1y < y)
        || (a1y > y && b1y > y)
        || (a1x < x && b1x < x)
        || (a1x > x && b1x > x);
    !outside && GeoUtils::orient(a1x, a1y, b1x, b1y, x, y) == 0
}

#[cfg(test)]
#[path = "edge_tree_reference.rs"]
mod reference;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_queries() {
        let t = EdgeTree::create_tree(&[0.0, 2.0, 2.0, 0.0, 0.0], &[0.0, 0.0, 2.0, 2.0, 0.0]);
        assert!(t.contains(1.0, 1.0));
        assert!(t.contains(0.0, 1.0)); // on edge
        assert!(t.contains(2.0, 2.0)); // vertex
        assert!(!t.contains(3.0, 1.0));
        assert!(t.is_point_on_line(1.0, 0.0));
        assert!(!t.is_point_on_line(1.0, 1.0));
        assert!(t.crosses_box(1.0, 3.0, 1.0, 3.0, false));
        assert!(t.crosses_box(-1.0, 3.0, 0.5, 1.5, false));
        assert!(!t.crosses_box(0.5, 1.5, 0.5, 1.5, true));
        assert!(t.crosses_line(-1.0, 3.0, 1.0, 1.0, -1.0, 1.0, 3.0, 1.0, false));
        assert!(t.crosses_line(-1.0, 0.0, 1.0, 1.0, -1.0, 1.0, 0.0, 1.0, true));
        assert!(!t.crosses_line(-1.0, 0.0, 1.0, 1.0, -1.0, 1.0, 0.0, 1.0, false));
        assert!(t.crosses_triangle(-1.0, 3.0, -1.0, 0.5, -1.0, 0.5, 3.0, 0.5, 1.0, -1.0, false));
        assert!(!t.crosses_triangle(0.5, 1.5, 0.5, 1.5, 0.5, 0.5, 1.5, 0.5, 1.0, 1.5, true));
        assert!(t.crosses_triangle(0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, true));
    }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        /// A coordinate on a coarse grid in `[-4, 4]`, so vertices, edges,
        /// slab bounds and queries coincide often (on-edge answers,
        /// `low`/`max` ties, signed zeros).
        fn grid(&mut self) -> f64 {
            let v = (self.next() % 33) as f64 * 0.25 - 4.0;
            if v == 0.0 && self.next().is_multiple_of(2) {
                -0.0
            } else {
                v
            }
        }
        /// A query bound: mostly the grid, sometimes NaN or infinite.
        fn bound(&mut self) -> f64 {
            match self.next() % 40 {
                0 => f64::NAN,
                1 => f64::INFINITY,
                2 => f64::NEG_INFINITY,
                _ => self.grid(),
            }
        }
    }

    /// The slab lookup answers every query exactly as Java's boxed-node
    /// tree (the port this replaced), over random closed rings and open
    /// lines whose vertices share coordinates, flat lines, and query
    /// bounds that are NaN, infinite or inverted.
    #[test]
    fn slabs_match_the_boxed_tree() {
        let mut rng = Rng(0x5eed);
        for case in 0..600 {
            let n = 2 + (rng.next() % 60) as usize;
            let flat = case % 7 == 6;
            let mut xs: Vec<f64> = (0..n).map(|_| rng.grid()).collect();
            let mut ys: Vec<f64> = (0..n)
                .map(|_| if flat { 1.5 } else { rng.grid() })
                .collect();
            if case % 2 == 0 {
                xs.push(xs[0]);
                ys.push(ys[0]);
            }
            let fast = EdgeTree::create_tree(&xs, &ys);
            let boxed = reference::EdgeTree::create_tree(&xs, &ys);
            assert_eq!(fast.x1.to_bits(), boxed.x1.to_bits(), "case {case}");
            assert_eq!(fast.y1.to_bits(), boxed.y1.to_bits(), "case {case}");
            for _ in 0..60 {
                let (x, y) = (rng.bound() - 0.125 * (rng.next() % 2) as f64, rng.bound());
                assert_eq!(fast.contains(x, y), boxed.contains(x, y), "case {case}");
                assert_eq!(
                    fast.is_point_on_line(x, y),
                    boxed.is_point_on_line(x, y),
                    "case {case}"
                );
                let (x2, y2) = (rng.bound(), rng.bound());
                // sometimes inverted
                let (min_x, max_x, min_y, max_y) = if rng.next().is_multiple_of(8) {
                    (x, x2, y, y2)
                } else {
                    (x.min(x2), x.max(x2), y.min(y2), y.max(y2))
                };
                for ib in [false, true] {
                    assert_eq!(
                        fast.crosses_box(min_x, max_x, min_y, max_y, ib),
                        boxed.crosses_box(min_x, max_x, min_y, max_y, ib),
                        "case {case} box {min_x} {max_x} {min_y} {max_y}"
                    );
                    assert_eq!(
                        fast.crosses_line(min_x, max_x, min_y, max_y, x, y, x2, y2, ib),
                        boxed.crosses_line(min_x, max_x, min_y, max_y, x, y, x2, y2, ib),
                        "case {case}"
                    );
                    let (cx, cy) = (rng.grid(), rng.grid());
                    let (tmin_x, tmax_x) = (min_x.min(cx), max_x.max(cx));
                    let (tmin_y, tmax_y) = (min_y.min(cy), max_y.max(cy));
                    assert_eq!(
                        fast.crosses_triangle(
                            tmin_x, tmax_x, tmin_y, tmax_y, x, y, x2, y2, cx, cy, ib
                        ),
                        boxed.crosses_triangle(
                            tmin_x, tmax_x, tmin_y, tmax_y, x, y, x2, y2, cx, cy, ib
                        ),
                        "case {case}"
                    );
                }
            }
        }
    }

    /// A ring with one very tall edge among many short ones: the slab
    /// count shrinks until the tall edge's copies fit the budget, and the
    /// answers stay the tree's.
    #[test]
    fn slab_budget_and_comb() {
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        // a comb: every tooth spans the whole height
        for i in 0..200 {
            xs.push(i as f64);
            ys.push(if i % 2 == 0 { 0.0 } else { 100.0 });
        }
        xs.push(0.0);
        ys.push(0.0);
        let t = EdgeTree::create_tree(&xs, &ys);
        let r = reference::EdgeTree::create_tree(&xs, &ys);
        assert!(t.slabs.entries.len() <= MAX_ENTRIES_PER_EDGE * t.edges.len());
        assert_eq!(t.slabs.start.len(), t.slabs.last + 2);
        for k in 0..400 {
            let (x, y) = (k as f64 * 0.5 + 0.25, (k % 101) as f64);
            assert_eq!(t.contains(x, y), r.contains(x, y), "{x} {y}");
            assert_eq!(
                t.crosses_box(x, x + 3.0, y, y + 7.0, true),
                r.crosses_box(x, x + 3.0, y, y + 7.0, true)
            );
        }
    }

    /// Point in polygon agrees with a brute-force even-odd test away from
    /// the boundary, on a large jittered ring (many slabs).
    #[test]
    fn contains_matches_brute_force_on_a_large_ring() {
        let mut rng = Rng(99);
        let n = 5_000;
        let (mut xs, mut ys) = (Vec::new(), Vec::new());
        for i in 0..n {
            let t = i as f64 / n as f64 * std::f64::consts::TAU;
            let r = 4.0 + (rng.next() % 1000) as f64 / 1000.0;
            xs.push(r * t.cos());
            ys.push(r * t.sin());
        }
        xs.push(xs[0]);
        ys.push(ys[0]);
        let tree = EdgeTree::create_tree(&xs, &ys);
        assert!(tree.slabs.last >= 255, "{}", tree.slabs.last);
        for _ in 0..2_000 {
            let x = (rng.next() % 10_000) as f64 / 1000.0 - 5.0 + 1e-7;
            let y = (rng.next() % 10_000) as f64 / 1000.0 - 5.0 + 1e-7;
            let mut inside = false;
            for i in 0..n {
                let (x1, y1, x2, y2) = (xs[i], ys[i], xs[i + 1], ys[i + 1]);
                if (y1 > y) != (y2 > y) && x < (x2 - x1) * (y - y1) / (y2 - y1) + x1 {
                    inside = !inside;
                }
            }
            assert_eq!(tree.contains(x, y), inside, "({x}, {y})");
        }
    }
}
