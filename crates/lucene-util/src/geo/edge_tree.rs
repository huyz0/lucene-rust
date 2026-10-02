//! Port of `org.apache.lucene.geo.EdgeTree`: an interval tree over a
//! polygon's or line's edges, keyed on their y-range, used by `Polygon2D`
//! and `Line2D`.
#![allow(clippy::too_many_arguments)]

use super::geo_utils::GeoUtils;
use super::java_max;
use super::rectangle::Rectangle;

const FALSE: u8 = 0x00;
const TRUE: u8 = 0x01;
const ON_EDGE: u8 = 0x02;

/// One edge and its subtree.
#[derive(Debug, Clone)]
pub(crate) struct EdgeTree {
    // X-Y pair (in original order) of the two vertices
    pub(crate) y1: f64,
    pub(crate) y2: f64,
    pub(crate) x1: f64,
    pub(crate) x2: f64,
    low: f64,
    max: f64,
    left: Option<Box<EdgeTree>>,
    right: Option<Box<EdgeTree>>,
}

impl EdgeTree {
    fn new(x1: f64, y1: f64, x2: f64, y2: f64, low: f64, max: f64) -> EdgeTree {
        EdgeTree {
            y1,
            y2,
            x1,
            x2,
            low,
            max,
            left: None,
            right: None,
        }
    }

    /// `contains(x, y)`: point in polygon, boundary included.
    pub(crate) fn contains(&self, x: f64, y: f64) -> bool {
        self.contains_pn_poly(x, y) > FALSE
    }

    /// `containsPnPoly`: W. Randolph Franklin's crossing test, with an
    /// on-edge answer.
    fn contains_pn_poly(&self, x: f64, y: f64) -> u8 {
        let mut res = FALSE;
        if y <= self.max {
            if y == self.y1 && y == self.y2
                || (y <= self.y1 && y >= self.y2) != (y >= self.y1 && y <= self.y2)
            {
                if (x == self.x1 && x == self.x2)
                    || ((x <= self.x1 && x >= self.x2) != (x >= self.x1 && x <= self.x2)
                        && GeoUtils::orient(self.x1, self.y1, self.x2, self.y2, x, y) == 0)
                {
                    return ON_EDGE;
                } else if (self.y1 > y) != (self.y2 > y) {
                    res = if x < (self.x2 - self.x1) * (y - self.y1) / (self.y2 - self.y1) + self.x1
                    {
                        TRUE
                    } else {
                        FALSE
                    };
                }
            }
            if let Some(left) = &self.left {
                res ^= left.contains_pn_poly(x, y);
                if (res & 0x02) == 0x02 {
                    return ON_EDGE;
                }
            }
            if let Some(right) = &self.right {
                if y >= self.low {
                    res ^= right.contains_pn_poly(x, y);
                    if (res & 0x02) == 0x02 {
                        return ON_EDGE;
                    }
                }
            }
        }
        res
    }

    /// `isPointOnLine(x, y)`.
    pub(crate) fn is_point_on_line(&self, x: f64, y: f64) -> bool {
        if y <= self.max {
            let a1x = self.x1;
            let a1y = self.y1;
            let b1x = self.x2;
            let b1y = self.y2;
            let outside = (a1y < y && b1y < y)
                || (a1y > y && b1y > y)
                || (a1x < x && b1x < x)
                || (a1x > x && b1x > x);
            if !outside && GeoUtils::orient(a1x, a1y, b1x, b1y, x, y) == 0 {
                return true;
            }
            if let Some(left) = &self.left {
                if left.is_point_on_line(x, y) {
                    return true;
                }
            }
            if let Some(right) = &self.right {
                if y >= self.low && right.is_point_on_line(x, y) {
                    return true;
                }
            }
        }
        false
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
        if min_y <= self.max {
            let dy = self.y1;
            let ey = self.y2;
            let dx = self.x1;
            let ex = self.x2;
            let outside = (dy < min_y && ey < min_y)
                || (dy > max_y && ey > max_y)
                || (dx < min_x && ex < min_x)
                || (dx > max_x && ex > max_x);
            if !outside {
                if include_boundary {
                    if GeoUtils::line_crosses_line_with_boundary(dx, dy, ex, ey, ax, ay, bx, by)
                        || GeoUtils::line_crosses_line_with_boundary(dx, dy, ex, ey, bx, by, cx, cy)
                        || GeoUtils::line_crosses_line_with_boundary(dx, dy, ex, ey, cx, cy, ax, ay)
                    {
                        return true;
                    }
                } else if GeoUtils::line_crosses_line(dx, dy, ex, ey, ax, ay, bx, by)
                    || GeoUtils::line_crosses_line(dx, dy, ex, ey, bx, by, cx, cy)
                    || GeoUtils::line_crosses_line(dx, dy, ex, ey, cx, cy, ax, ay)
                {
                    return true;
                }
            }
            if let Some(left) = &self.left {
                if left.crosses_triangle(
                    min_x,
                    max_x,
                    min_y,
                    max_y,
                    ax,
                    ay,
                    bx,
                    by,
                    cx,
                    cy,
                    include_boundary,
                ) {
                    return true;
                }
            }
            if let Some(right) = &self.right {
                if max_y >= self.low
                    && right.crosses_triangle(
                        min_x,
                        max_x,
                        min_y,
                        max_y,
                        ax,
                        ay,
                        bx,
                        by,
                        cx,
                        cy,
                        include_boundary,
                    )
                {
                    return true;
                }
            }
        }
        false
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
        if min_y <= self.max {
            let cy = self.y1;
            let dy = self.y2;
            let cx = self.x1;
            let dx = self.x2;
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
            if !outside {
                if include_boundary {
                    if GeoUtils::line_crosses_line_with_boundary(
                        cx, cy, dx, dy, min_x, min_y, max_x, min_y,
                    ) || GeoUtils::line_crosses_line_with_boundary(
                        cx, cy, dx, dy, max_x, min_y, max_x, max_y,
                    ) || GeoUtils::line_crosses_line_with_boundary(
                        cx, cy, dx, dy, max_x, max_y, min_x, max_y,
                    ) || GeoUtils::line_crosses_line_with_boundary(
                        cx, cy, dx, dy, min_x, max_y, min_x, min_y,
                    ) {
                        return true;
                    }
                } else if GeoUtils::line_crosses_line(cx, cy, dx, dy, min_x, min_y, max_x, min_y)
                    || GeoUtils::line_crosses_line(cx, cy, dx, dy, max_x, min_y, max_x, max_y)
                    || GeoUtils::line_crosses_line(cx, cy, dx, dy, max_x, max_y, min_x, max_y)
                    || GeoUtils::line_crosses_line(cx, cy, dx, dy, min_x, max_y, min_x, min_y)
                {
                    return true;
                }
            }
            if let Some(left) = &self.left {
                if left.crosses_box(min_x, max_x, min_y, max_y, include_boundary) {
                    return true;
                }
            }
            if let Some(right) = &self.right {
                if max_y >= self.low
                    && right.crosses_box(min_x, max_x, min_y, max_y, include_boundary)
                {
                    return true;
                }
            }
        }
        false
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
        if min_y <= self.max {
            let a1x = self.x1;
            let a1y = self.y1;
            let b1x = self.x2;
            let b1y = self.y2;
            let outside = (a1y < min_y && b1y < min_y)
                || (a1y > max_y && b1y > max_y)
                || (a1x < min_x && b1x < min_x)
                || (a1x > max_x && b1x > max_x);
            if !outside {
                if include_boundary {
                    if GeoUtils::line_crosses_line_with_boundary(
                        a1x, a1y, b1x, b1y, a2x, a2y, b2x, b2y,
                    ) {
                        return true;
                    }
                } else if GeoUtils::line_crosses_line(a1x, a1y, b1x, b1y, a2x, a2y, b2x, b2y) {
                    return true;
                }
            }
            if let Some(left) = &self.left {
                if left.crosses_line(
                    min_x,
                    max_x,
                    min_y,
                    max_y,
                    a2x,
                    a2y,
                    b2x,
                    b2y,
                    include_boundary,
                ) {
                    return true;
                }
            }
            if let Some(right) = &self.right {
                if max_y >= self.low
                    && right.crosses_line(
                        min_x,
                        max_x,
                        min_y,
                        max_y,
                        a2x,
                        a2y,
                        b2x,
                        b2y,
                        include_boundary,
                    )
                {
                    return true;
                }
            }
        }
        false
    }

    /// `createTree(x, y)`: sorts the edges by `(low, max)` and builds a
    /// balanced tree. Panics on fewer than two points (Java would throw on
    /// the empty array; every caller has validated its geometry first).
    pub(crate) fn create_tree(x: &[f64], y: &[f64]) -> EdgeTree {
        let mut edges: Vec<Option<EdgeTree>> = Vec::with_capacity(x.len().saturating_sub(1));
        for i in 1..x.len() {
            let x1 = x[i - 1];
            let y1 = y[i - 1];
            let x2 = x[i];
            let y2 = y[i];
            edges.push(Some(EdgeTree::new(
                x1,
                y1,
                x2,
                y2,
                super::java_min(y1, y2),
                java_max(y1, y2),
            )));
        }
        // Arrays.sort over objects is a stable merge sort, as is sort_by.
        edges.sort_by(|l, r| {
            let (l, r) = (l.as_ref().expect("edge"), r.as_ref().expect("edge"));
            l.low.total_cmp(&r.low).then(l.max.total_cmp(&r.max))
        });
        let high = edges.len() as isize - 1;
        *Self::create_tree_range(&mut edges, 0, high).expect("a line has at least one edge")
    }

    fn create_tree_range(
        edges: &mut [Option<EdgeTree>],
        low: isize,
        high: isize,
    ) -> Option<Box<EdgeTree>> {
        if low > high {
            return None;
        }
        // add midpoint
        let mid = ((low + high) as usize >> 1) as isize;
        let mut new_node = edges[mid as usize].take().expect("each edge used once");
        // add children
        new_node.left = Self::create_tree_range(edges, low, mid - 1);
        new_node.right = Self::create_tree_range(edges, mid + 1, high);
        // pull up max values to this node
        if let Some(left) = &new_node.left {
            new_node.max = java_max(new_node.max, left.max);
        }
        if let Some(right) = &new_node.right {
            new_node.max = java_max(new_node.max, right.max);
        }
        Some(Box::new(new_node))
    }
}

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
}
