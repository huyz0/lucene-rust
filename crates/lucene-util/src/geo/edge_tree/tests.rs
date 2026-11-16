#![cfg(test)]
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
                    fast.crosses_triangle(tmin_x, tmax_x, tmin_y, tmax_y, x, y, x2, y2, cx, cy, ib),
                    boxed
                        .crosses_triangle(tmin_x, tmax_x, tmin_y, tmax_y, x, y, x2, y2, cx, cy, ib),
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
    assert!(t.slabs.segs.len() <= MAX_ENTRIES_PER_EDGE * t.edges.len());
    assert_eq!(t.slabs.first.len(), t.slabs.segs.len());
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
