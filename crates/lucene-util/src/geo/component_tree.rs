//! Port of `org.apache.lucene.geo.ComponentTree`: a 2D interval tree of
//! components (multi-polygons, polygon holes, dateline-split boxes),
//! alternating its split between y and x.
#![allow(clippy::too_many_arguments)]

use std::cmp::Ordering;

use super::component2d::{Component2D, WithinRelation};
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

    fn contains_split(&self, x: f64, y: f64, split_x: bool) -> bool {
        if y <= self.max_y && x <= self.max_x {
            if self.component.contains(x, y) {
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
            if self
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
            if self
                .component
                .intersects_triangle_bbox(min_x, max_x, min_y, max_y, a_x, a_y, b_x, b_y, c_x, c_y)
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
            if self
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
            if self
                .component
                .contains_triangle_bbox(min_x, max_x, min_y, max_y, a_x, a_y, b_x, b_y, c_x, c_y)
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
            let relation = self.component.relate(min_x, max_x, min_y, max_y);
            if relation != Relation::CellOutsideQuery {
                return relation;
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
