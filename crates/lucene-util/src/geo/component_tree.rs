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
    left: Option<Box<ComponentTree>>,
    right: Option<Box<ComponentTree>>,
    component: Box<dyn Component2D>,
}

/// `ArrayUtil.select`'s `IntroSelector` over the components, ordered by
/// `X_COMPARATOR` or `Y_COMPARATOR` (`Double.compare` on min, then max).
struct Select<'a> {
    components: &'a mut [Option<Box<dyn Component2D>>],
    split_x: bool,
    pivot: (f64, f64),
}

impl Select<'_> {
    fn key(&self, i: usize) -> (f64, f64) {
        let c = self.components[i].as_ref().expect("component present");
        if self.split_x {
            (c.min_x(), c.max_x())
        } else {
            (c.min_y(), c.max_y())
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
    fn new(component: Box<dyn Component2D>) -> ComponentTree {
        ComponentTree {
            min_y: component.min_y(),
            max_y: component.max_y(),
            min_x: component.min_x(),
            max_x: component.max_x(),
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
        let mut slots: Vec<Option<Box<dyn Component2D>>> =
            components.into_iter().map(Some).collect();
        let high = slots.len() as isize - 1;
        // Java's root min-pull-up reads every component; read them before
        // the tree takes ownership.
        let mut root =
            Self::create_tree(&mut slots, 0, high, ROOT_SPLITX).expect("at least two components");
        let mut stack: Vec<&ComponentTree> = Vec::new();
        let (mut min_y, mut min_x) = (root.min_y, root.min_x);
        stack.push(&root);
        let mut all = Vec::new();
        while let Some(n) = stack.pop() {
            all.push((n.component.min_y(), n.component.min_x()));
            if let Some(l) = &n.left {
                stack.push(l);
            }
            if let Some(r) = &n.right {
                stack.push(r);
            }
        }
        // min is order-independent except for NaN/-0.0 ties, which a
        // validated geometry never produces.
        for (cy, cx) in all {
            min_y = java_min(min_y, cy);
            min_x = java_min(min_x, cx);
        }
        root.min_y = min_y;
        root.min_x = min_x;
        root
    }

    fn create_tree(
        components: &mut [Option<Box<dyn Component2D>>],
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
        let component = components[mid as usize]
            .take()
            .expect("each component used once");
        let mut new_node = Box::new(ComponentTree::new(component));
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
        (!split_x && y >= self.component.min_y()) || (split_x && x >= self.component.min_x())
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
