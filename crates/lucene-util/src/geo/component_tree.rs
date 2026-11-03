//! Port of `org.apache.lucene.geo.ComponentTree`: a 2D interval tree of
//! components (multi-polygons, polygon holes, dateline-split boxes),
//! alternating its split between y and x.
#![allow(clippy::too_many_arguments)]

use std::cmp::Ordering;

use super::component2d::{self, Component2D, WithinRelation};
use super::{java_max, java_min, GeoError, Relation};
use crate::sorter::{intro_select, IntroTarget};
use crate::splittable_random::SplittableRandom;

const ROOT_SPLITX: bool = false;

/// Port of `org.apache.lucene.geo.ComponentTree`.
#[derive(Debug)]
pub(crate) struct ComponentTree {
    min_y: f64,
    max_y: f64,
    min_x: f64,
    max_x: f64,
    /// The component's own `getMinX()`/`getMinY()` (the root's `min_x`/
    /// `min_y` are pulled down to the global min), cached so the per-node
    /// `right_reachable` test is not two virtual calls.
    comp_min_x: f64,
    comp_min_y: f64,
    /// The component's own `getMaxX()`/`getMaxY()` (`max_x`/`max_y` are
    /// pulled up from the children). With `comp_min_*` they skip the
    /// component's virtual call for a query outside its bounding box, which
    /// every component answers false (or `CELL_OUTSIDE_QUERY`) itself.
    comp_max_x: f64,
    comp_max_y: f64,
    left: Option<Box<ComponentTree>>,
    right: Option<Box<ComponentTree>>,
    component: Box<dyn Component2D>,
}

/// A component waiting to be placed, with its bounding box read once
/// (`[min_x, max_x, min_y, max_y]`).
struct Slot {
    component: Option<Box<dyn Component2D>>,
    bbox: [f64; 4],
}

/// `ArrayUtil.select`'s `IntroSelector` over the components, ordered by
/// `X_COMPARATOR` or `Y_COMPARATOR` (`Double.compare` on min, then max).
struct Select<'a> {
    components: &'a mut [Slot],
    split_x: bool,
    pivot: (f64, f64),
}

impl Select<'_> {
    #[inline]
    fn key(&self, i: usize) -> (f64, f64) {
        let b = &self.components[i].bbox;
        if self.split_x {
            (b[0], b[1])
        } else {
            (b[2], b[3])
        }
    }
}

impl IntroTarget for Select<'_> {
    fn swap(&mut self, i: usize, j: usize) {
        self.components.swap(i, j);
    }
    fn set_pivot(&mut self, i: usize) {
        self.pivot = self.key(i);
    }
    // SENTINEL: none -- `-1` is "less than", a comparator result.
    fn compare_pivot(&mut self, j: usize) -> i32 {
        let (a0, a1) = self.pivot;
        let (b0, b1) = self.key(j);
        match a0.total_cmp(&b0).then(a1.total_cmp(&b1)) {
            Ordering::Less => -1,
            Ordering::Equal => 0,
            Ordering::Greater => 1,
        }
    }
}

impl ComponentTree {
    fn new(component: Box<dyn Component2D>, bbox: [f64; 4]) -> ComponentTree {
        let [min_x, max_x, min_y, max_y] = bbox;
        ComponentTree {
            min_y,
            max_y,
            min_x,
            max_x,
            comp_min_x: min_x,
            comp_min_y: min_y,
            comp_max_x: max_x,
            comp_max_y: max_y,
            left: None,
            right: None,
            component,
        }
    }

    /// `create(components)`: a single component is returned as is.
    pub(crate) fn create(components: Vec<Box<dyn Component2D>>) -> Box<dyn Component2D> {
        if components.len() == 1 {
            return components.into_iter().next().expect("one component");
        }
        let mut slots: Vec<Slot> = components
            .into_iter()
            .map(|c| Slot {
                bbox: [c.min_x(), c.max_x(), c.min_y(), c.max_y()],
                component: Some(c),
            })
            .collect();
        // Java's root min-pull-up folds `Math.min` over every component
        // after the select has permuted them; `Math.min` is commutative and
        // associative (NaN wins, -0.0 beats 0.0), so folding in input order
        // gives the same bits.
        let mut min_y = f64::INFINITY;
        let mut min_x = f64::INFINITY;
        for s in &slots {
            min_y = java_min(min_y, s.bbox[2]);
            min_x = java_min(min_x, s.bbox[0]);
        }
        let high = slots.len() as isize - 1;
        let mut root =
            Self::create_tree(&mut slots, 0, high, ROOT_SPLITX).expect("at least two components");
        root.min_y = java_min(root.min_y, min_y);
        root.min_x = java_min(root.min_x, min_x);
        root
    }

    fn create_tree(
        components: &mut [Slot],
        low: isize,
        high: isize,
        split_x: bool,
    ) -> Option<Box<ComponentTree>> {
        if low > high {
            return None;
        }
        let mid = ((low + high) as usize >> 1) as isize;
        if low < high {
            let mut select = Select {
                components,
                split_x,
                pivot: (0.0, 0.0),
            };
            // Java's IntroSelector shuffles with an unseeded random only on
            // pathological inputs; a fixed seed keeps the port reproducible.
            let mut random = SplittableRandom::new(0);
            intro_select(
                &mut select,
                low as usize,
                high as usize + 1,
                mid as usize,
                &mut random,
            );
        }
        let slot = &mut components[mid as usize];
        let component = slot.component.take().expect("each component used once");
        let mut new_node = Box::new(ComponentTree::new(component, slot.bbox));
        // find children
        new_node.left = Self::create_tree(components, low, mid - 1, !split_x);
        new_node.right = Self::create_tree(components, mid + 1, high, !split_x);
        // pull up max values to this node
        if let Some(left) = &new_node.left {
            new_node.max_x = java_max(new_node.max_x, left.max_x);
            new_node.max_y = java_max(new_node.max_y, left.max_y);
        }
        if let Some(right) = &new_node.right {
            new_node.max_x = java_max(new_node.max_x, right.max_x);
            new_node.max_y = java_max(new_node.max_y, right.max_y);
        }
        Some(new_node)
    }

    /// Whether the right subtree can hold a match: Java's
    /// `(splitX == false && maxY >= component.getMinY()) || (splitX && maxX >= component.getMinX())`.
    #[inline]
    fn right_reachable(&self, split_x: bool, x: f64, y: f64) -> bool {
        (!split_x && y >= self.comp_min_y) || (split_x && x >= self.comp_min_x)
    }

    /// Whether the query box can meet the component's own bounding box:
    /// every component's `relate`, `intersects*` and `contains*` start
    /// with this disjoint test (or one implied by it), so a box that fails
    /// it gets their false / `CELL_OUTSIDE_QUERY` without the call.
    #[inline]
    fn comp_may_meet(&self, min_x: f64, max_x: f64, min_y: f64, max_y: f64) -> bool {
        !component2d::disjoint(
            self.comp_min_x,
            self.comp_max_x,
            self.comp_min_y,
            self.comp_max_y,
            min_x,
            max_x,
            min_y,
            max_y,
        )
    }

    /// [`Self::comp_may_meet`] for `containsLine`/`containsTriangle`:
    /// `Rectangle2D` answers those with a `within` test alone, which an
    /// inverted query box can pass while disjoint, so such a box always
    /// reaches the component.
    #[inline]
    fn comp_may_contain(&self, min_x: f64, max_x: f64, min_y: f64, max_y: f64) -> bool {
        !(min_x <= max_x && min_y <= max_y) || self.comp_may_meet(min_x, max_x, min_y, max_y)
    }

    fn contains_split(&self, x: f64, y: f64, split_x: bool) -> bool {
        if y <= self.max_y && x <= self.max_x {
            if component2d::contains_point(
                x,
                y,
                self.comp_min_x,
                self.comp_max_x,
                self.comp_min_y,
                self.comp_max_y,
            ) && self.component.contains(x, y)
            {
                return true;
            }
            if let Some(left) = &self.left {
                if left.contains_split(x, y, !split_x) {
                    return true;
                }
            }
            if let Some(right) = &self.right {
                if self.right_reachable(split_x, x, y) {
                    return right.contains_split(x, y, !split_x);
                }
            }
        }
        false
    }

    fn intersects_line_split(
        &self,
        min_x: f64,
        max_x: f64,
        min_y: f64,
        max_y: f64,
        a_x: f64,
        a_y: f64,
        b_x: f64,
        b_y: f64,
        split_x: bool,
    ) -> bool {
        if min_y <= self.max_y && min_x <= self.max_x {
            if self.comp_may_meet(min_x, max_x, min_y, max_y)
                && self
                    .component
                    .intersects_line_bbox(min_x, max_x, min_y, max_y, a_x, a_y, b_x, b_y)
            {
                return true;
            }
            if let Some(left) = &self.left {
                if left
                    .intersects_line_split(min_x, max_x, min_y, max_y, a_x, a_y, b_x, b_y, !split_x)
                {
                    return true;
                }
            }
            if let Some(right) = &self.right {
                if self.right_reachable(split_x, max_x, max_y) {
                    return right.intersects_line_split(
                        min_x, max_x, min_y, max_y, a_x, a_y, b_x, b_y, !split_x,
                    );
                }
            }
        }
        false
    }

    fn intersects_triangle_split(
        &self,
        min_x: f64,
        max_x: f64,
        min_y: f64,
        max_y: f64,
        a_x: f64,
        a_y: f64,
        b_x: f64,
        b_y: f64,
        c_x: f64,
        c_y: f64,
        split_x: bool,
    ) -> bool {
        if min_y <= self.max_y && min_x <= self.max_x {
            if self.comp_may_meet(min_x, max_x, min_y, max_y)
                && self.component.intersects_triangle_bbox(
                    min_x, max_x, min_y, max_y, a_x, a_y, b_x, b_y, c_x, c_y,
                )
            {
                return true;
            }
            if let Some(left) = &self.left {
                if left.intersects_triangle_split(
                    min_x, max_x, min_y, max_y, a_x, a_y, b_x, b_y, c_x, c_y, !split_x,
                ) {
                    return true;
                }
            }
            if let Some(right) = &self.right {
                if self.right_reachable(split_x, max_x, max_y) {
                    return right.intersects_triangle_split(
                        min_x, max_x, min_y, max_y, a_x, a_y, b_x, b_y, c_x, c_y, !split_x,
                    );
                }
            }
        }
        false
    }

    fn contains_line_split(
        &self,
        min_x: f64,
        max_x: f64,
        min_y: f64,
        max_y: f64,
        a_x: f64,
        a_y: f64,
        b_x: f64,
        b_y: f64,
        split_x: bool,
    ) -> bool {
        if min_y <= self.max_y && min_x <= self.max_x {
            if self.comp_may_contain(min_x, max_x, min_y, max_y)
                && self
                    .component
                    .contains_line_bbox(min_x, max_x, min_y, max_y, a_x, a_y, b_x, b_y)
            {
                return true;
            }
            if let Some(left) = &self.left {
                if left
                    .contains_line_split(min_x, max_x, min_y, max_y, a_x, a_y, b_x, b_y, !split_x)
                {
                    return true;
                }
            }
            if let Some(right) = &self.right {
                if self.right_reachable(split_x, max_x, max_y) {
                    return right.contains_line_split(
                        min_x, max_x, min_y, max_y, a_x, a_y, b_x, b_y, !split_x,
                    );
                }
            }
        }
        false
    }

    fn contains_triangle_split(
        &self,
        min_x: f64,
        max_x: f64,
        min_y: f64,
        max_y: f64,
        a_x: f64,
        a_y: f64,
        b_x: f64,
        b_y: f64,
        c_x: f64,
        c_y: f64,
        split_x: bool,
    ) -> bool {
        if min_y <= self.max_y && min_x <= self.max_x {
            if self.comp_may_contain(min_x, max_x, min_y, max_y)
                && self.component.contains_triangle_bbox(
                    min_x, max_x, min_y, max_y, a_x, a_y, b_x, b_y, c_x, c_y,
                )
            {
                return true;
            }
            if let Some(left) = &self.left {
                if left.contains_triangle_split(
                    min_x, max_x, min_y, max_y, a_x, a_y, b_x, b_y, c_x, c_y, !split_x,
                ) {
                    return true;
                }
            }
            if let Some(right) = &self.right {
                if self.right_reachable(split_x, max_x, max_y) {
                    return right.contains_triangle_split(
                        min_x, max_x, min_y, max_y, a_x, a_y, b_x, b_y, c_x, c_y, !split_x,
                    );
                }
            }
        }
        false
    }

    fn relate_split(
        &self,
        min_x: f64,
        max_x: f64,
        min_y: f64,
        max_y: f64,
        split_x: bool,
    ) -> Relation {
        if min_y <= self.max_y && min_x <= self.max_x {
            if self.comp_may_meet(min_x, max_x, min_y, max_y) {
                let relation = self.component.relate(min_x, max_x, min_y, max_y);
                if relation != Relation::CellOutsideQuery {
                    return relation;
                }
            }
            if let Some(left) = &self.left {
                let relation = left.relate_split(min_x, max_x, min_y, max_y, !split_x);
                if relation != Relation::CellOutsideQuery {
                    return relation;
                }
            }
            if let Some(right) = &self.right {
                if self.right_reachable(split_x, max_x, max_y) {
                    return right.relate_split(min_x, max_x, min_y, max_y, !split_x);
                }
            }
        }
        Relation::CellOutsideQuery
    }

    fn single(&self, what: &str) -> Result<(), GeoError> {
        if self.left.is_some() || self.right.is_some() {
            return Err(GeoError::illegal(format!(
                "{what} is not supported for shapes with more than one component"
            )));
        }
        Ok(())
    }
}

impl Component2D for ComponentTree {
    fn min_x(&self) -> f64 {
        self.min_x
    }
    fn max_x(&self) -> f64 {
        self.max_x
    }
    fn min_y(&self) -> f64 {
        self.min_y
    }
    fn max_y(&self) -> f64 {
        self.max_y
    }

    fn contains(&self, x: f64, y: f64) -> bool {
        self.contains_split(x, y, ROOT_SPLITX)
    }

    fn relate(&self, min_x: f64, max_x: f64, min_y: f64, max_y: f64) -> Relation {
        self.relate_split(min_x, max_x, min_y, max_y, ROOT_SPLITX)
    }

    fn intersects_line_bbox(
        &self,
        min_x: f64,
        max_x: f64,
        min_y: f64,
        max_y: f64,
        a_x: f64,
        a_y: f64,
        b_x: f64,
        b_y: f64,
    ) -> bool {
        self.intersects_line_split(min_x, max_x, min_y, max_y, a_x, a_y, b_x, b_y, ROOT_SPLITX)
    }

    fn intersects_triangle_bbox(
        &self,
        min_x: f64,
        max_x: f64,
        min_y: f64,
        max_y: f64,
        a_x: f64,
        a_y: f64,
        b_x: f64,
        b_y: f64,
        c_x: f64,
        c_y: f64,
    ) -> bool {
        self.intersects_triangle_split(
            min_x,
            max_x,
            min_y,
            max_y,
            a_x,
            a_y,
            b_x,
            b_y,
            c_x,
            c_y,
            ROOT_SPLITX,
        )
    }

    fn contains_line_bbox(
        &self,
        min_x: f64,
        max_x: f64,
        min_y: f64,
        max_y: f64,
        a_x: f64,
        a_y: f64,
        b_x: f64,
        b_y: f64,
    ) -> bool {
        self.contains_line_split(min_x, max_x, min_y, max_y, a_x, a_y, b_x, b_y, ROOT_SPLITX)
    }

    fn contains_triangle_bbox(
        &self,
        min_x: f64,
        max_x: f64,
        min_y: f64,
        max_y: f64,
        a_x: f64,
        a_y: f64,
        b_x: f64,
        b_y: f64,
        c_x: f64,
        c_y: f64,
    ) -> bool {
        self.contains_triangle_split(
            min_x,
            max_x,
            min_y,
            max_y,
            a_x,
            a_y,
            b_x,
            b_y,
            c_x,
            c_y,
            ROOT_SPLITX,
        )
    }

    fn within_point(&self, x: f64, y: f64) -> Result<WithinRelation, GeoError> {
        self.single("withinPoint")?;
        self.component.within_point(x, y)
    }

    fn within_line_bbox(
        &self,
        min_x: f64,
        max_x: f64,
        min_y: f64,
        max_y: f64,
        a_x: f64,
        a_y: f64,
        ab: bool,
        b_x: f64,
        b_y: f64,
    ) -> Result<WithinRelation, GeoError> {
        self.single("withinLine")?;
        self.component
            .within_line_bbox(min_x, max_x, min_y, max_y, a_x, a_y, ab, b_x, b_y)
    }

    fn within_triangle_bbox(
        &self,
        min_x: f64,
        max_x: f64,
        min_y: f64,
        max_y: f64,
        a_x: f64,
        a_y: f64,
        ab: bool,
        b_x: f64,
        b_y: f64,
        bc: bool,
        c_x: f64,
        c_y: f64,
        ca: bool,
    ) -> Result<WithinRelation, GeoError> {
        self.single("withinTriangle")?;
        self.component.within_triangle_bbox(
            min_x, max_x, min_y, max_y, a_x, a_y, ab, b_x, b_y, bc, c_x, c_y, ca,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geo::{XYCircle, XYGeometry, XYLine, XYPoint, XYPolygon, XYRectangle};

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        /// A coordinate on a coarse grid, so bounding boxes touch often.
        fn grid(&mut self) -> f32 {
            (self.next() % 41) as f32 * 0.5 - 10.0
        }
        /// A query coordinate: mostly the grid, sometimes NaN.
        fn bound(&mut self) -> f64 {
            if self.next().is_multiple_of(50) {
                f64::NAN
            } else {
                f64::from(self.grid()) + 0.25 * (self.next() % 2) as f64
            }
        }
    }

    fn geometry(rng: &mut Rng) -> XYGeometry {
        let (x, y) = (rng.grid(), rng.grid());
        let w = 0.5 + (rng.next() % 6) as f32 * 0.5;
        match rng.next() % 5 {
            0 => XYGeometry::Polygon(
                XYPolygon::new(&[x, x + w, x, x], &[y, y, y + w, y], vec![]).unwrap(),
            ),
            1 => XYGeometry::Rectangle(XYRectangle::new(x, x + w, y, y + w).unwrap()),
            2 => XYGeometry::Circle(XYCircle::new(x, y, w).unwrap()),
            3 => XYGeometry::Point(XYPoint::new(x, y).unwrap()),
            _ => XYGeometry::Line(XYLine::new(&[x, x + w, x + w], &[y, y + w, y]).unwrap()),
        }
    }

    /// What the tree's bounding-box skip relies on: every component, for
    /// any query box (inverted and NaN ones included), answers false (or
    /// `CELL_OUTSIDE_QUERY`) when the box is disjoint from its own bounding
    /// box -- `containsLine`/`containsTriangle` only for a box that is not
    /// inverted, which is when the tree skips them. And over proper boxes
    /// the tree's answers are any component's.
    #[test]
    fn bbox_skip_keeps_every_components_answer() {
        let mut rng = Rng(0xc0de);
        for case in 0..300 {
            let n = 2 + (rng.next() % 12) as usize;
            let geoms: Vec<XYGeometry> = (0..n).map(|_| geometry(&mut rng)).collect();
            let parts: Vec<Box<dyn Component2D>> =
                geoms.iter().map(|g| g.to_component2d().unwrap()).collect();
            let tree =
                ComponentTree::create(geoms.iter().map(|g| g.to_component2d().unwrap()).collect());
            for _ in 0..80 {
                let (ax, ay, bx, by, cx, cy) = (
                    rng.bound(),
                    rng.bound(),
                    rng.bound(),
                    rng.bound(),
                    rng.bound(),
                    rng.bound(),
                );
                let inverted = rng.next().is_multiple_of(4);
                let (min_x, max_x, min_y, max_y) = if inverted {
                    (ax, bx, ay, by)
                } else {
                    (
                        ax.min(bx).min(cx),
                        ax.max(bx).max(cx),
                        ay.min(by).min(cy),
                        ay.max(by).max(cy),
                    )
                };
                for p in &parts {
                    let bbox = (p.min_x(), p.max_x(), p.min_y(), p.max_y());
                    if !component2d::contains_point(ax, ay, bbox.0, bbox.1, bbox.2, bbox.3) {
                        assert!(!p.contains(ax, ay), "case {case}");
                    }
                    let outside = component2d::disjoint(
                        bbox.0, bbox.1, bbox.2, bbox.3, min_x, max_x, min_y, max_y,
                    );
                    if !outside {
                        continue;
                    }
                    assert_eq!(
                        p.relate(min_x, max_x, min_y, max_y),
                        Relation::CellOutsideQuery,
                        "case {case}"
                    );
                    assert!(
                        !p.intersects_line_bbox(min_x, max_x, min_y, max_y, ax, ay, bx, by),
                        "case {case}"
                    );
                    assert!(
                        !p.intersects_triangle_bbox(
                            min_x, max_x, min_y, max_y, ax, ay, bx, by, cx, cy
                        ),
                        "case {case}"
                    );
                    if min_x <= max_x && min_y <= max_y {
                        assert!(
                            !p.contains_line_bbox(min_x, max_x, min_y, max_y, ax, ay, bx, by),
                            "case {case}"
                        );
                        assert!(
                            !p.contains_triangle_bbox(
                                min_x, max_x, min_y, max_y, ax, ay, bx, by, cx, cy
                            ),
                            "case {case}"
                        );
                    }
                }
                if inverted || [min_x, max_x, min_y, max_y].iter().any(|v| v.is_nan()) {
                    continue;
                }
                let any = |f: &dyn Fn(&dyn Component2D) -> bool| parts.iter().any(|p| f(&**p));
                assert_eq!(
                    tree.contains(ax, ay),
                    any(&|p| p.contains(ax, ay)),
                    "case {case}"
                );
                assert_eq!(
                    tree.relate(min_x, max_x, min_y, max_y) == Relation::CellOutsideQuery,
                    !any(&|p| p.relate(min_x, max_x, min_y, max_y) != Relation::CellOutsideQuery),
                    "case {case}"
                );
                assert_eq!(
                    tree.intersects_line_bbox(min_x, max_x, min_y, max_y, ax, ay, bx, by),
                    any(&|p| p.intersects_line_bbox(min_x, max_x, min_y, max_y, ax, ay, bx, by)),
                    "case {case}"
                );
                assert_eq!(
                    tree.contains_line_bbox(min_x, max_x, min_y, max_y, ax, ay, bx, by),
                    any(&|p| p.contains_line_bbox(min_x, max_x, min_y, max_y, ax, ay, bx, by)),
                    "case {case}"
                );
                assert_eq!(
                    tree.intersects_triangle_bbox(
                        min_x, max_x, min_y, max_y, ax, ay, bx, by, cx, cy
                    ),
                    any(&|p| p.intersects_triangle_bbox(
                        min_x, max_x, min_y, max_y, ax, ay, bx, by, cx, cy
                    )),
                    "case {case}"
                );
                assert_eq!(
                    tree.contains_triangle_bbox(min_x, max_x, min_y, max_y, ax, ay, bx, by, cx, cy),
                    any(&|p| p.contains_triangle_bbox(
                        min_x, max_x, min_y, max_y, ax, ay, bx, by, cx, cy
                    )),
                    "case {case}"
                );
            }
        }
    }
}
