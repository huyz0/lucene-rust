#![cfg(test)]
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
                    !p.intersects_triangle_bbox(min_x, max_x, min_y, max_y, ax, ay, bx, by, cx, cy),
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
                tree.intersects_triangle_bbox(min_x, max_x, min_y, max_y, ax, ay, bx, by, cx, cy),
                any(&|p| p
                    .intersects_triangle_bbox(min_x, max_x, min_y, max_y, ax, ay, bx, by, cx, cy)),
                "case {case}"
            );
            assert_eq!(
                tree.contains_triangle_bbox(min_x, max_x, min_y, max_y, ax, ay, bx, by, cx, cy),
                any(&|p| p
                    .contains_triangle_bbox(min_x, max_x, min_y, max_y, ax, ay, bx, by, cx, cy)),
                "case {case}"
            );
        }
    }
}
