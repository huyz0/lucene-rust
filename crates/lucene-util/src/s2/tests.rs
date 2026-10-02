//! Unit tests for the S2 subset: the cell-id arithmetic's invariants and
//! the edges the fixtures (`tests/spatial4j_fixtures.rs`) do not reach.

use super::projections::*;
use super::*;

#[test]
fn point_normalize_and_components() {
    let p = S2Point::new(3.0, 0.0, 4.0);
    assert_eq!(p.norm2(), 25.0);
    let n = p.normalize();
    assert!((n.x - 0.6).abs() < 1e-15 && n.y == 0.0 && (n.z - 0.8).abs() < 1e-15);
    assert_eq!(
        S2Point::new(0.0, 0.0, 0.0).normalize(),
        S2Point::new(0.0, 0.0, 0.0)
    );
    assert_eq!(p.get(0), 3.0);
    assert_eq!(p.get(1), 0.0);
    assert_eq!(p.get(2), 4.0);
    assert_eq!(p.largest_abs_component(), 2);
    assert_eq!(S2Point::new(-5.0, 1.0, 1.0).largest_abs_component(), 0);
    assert_eq!(S2Point::new(1.0, -5.0, 1.0).largest_abs_component(), 1);
    assert_eq!(S2Point::new(5.0, 1.0, 9.0).largest_abs_component(), 2);
}

#[test]
fn projections_round_trip() {
    for &s in &[-1.0, -0.5, 0.0, 0.25, 1.0] {
        let u = st_to_uv(s);
        assert!((uv_to_st(u) - s).abs() < 1e-15, "{s}");
    }
    for face in 0..6 {
        let p = face_uv_to_xyz(face, 0.25, -0.5);
        assert_eq!(xyz_to_face(&p), face);
        let (u, v) = valid_face_xyz_to_uv(face, &p);
        assert!(
            (u - 0.25).abs() < 1e-15 && (v + 0.5).abs() < 1e-15,
            "{face}"
        );
    }
}

#[test]
fn metric_levels() {
    assert_eq!(MAX_WIDTH.deriv(), MAX_ANGLE_SPAN_DERIV);
    assert_eq!(MAX_WIDTH.get_min_level(0.0), MAX_LEVEL);
    assert_eq!(MAX_WIDTH.get_max_level(-1.0), MAX_LEVEL);
    for level in 0..=MAX_LEVEL {
        let v = MAX_WIDTH.get_value(level);
        assert_eq!(MAX_WIDTH.get_min_level(v), level, "{level}");
        assert_eq!(MAX_WIDTH.get_max_level(v), level, "{level}");
    }
    assert_eq!(exp(0.0), 0);
    assert_eq!(exp(1.0), 1);
    assert_eq!(exp(0.5), 0);
    let area = Metric::new(2, 1.0);
    assert_eq!(area.get_value(1), 1.0);
    assert_eq!(area.get_max_level(0.25), 2);
    assert_eq!(scalb(1.0, 1100), f64::INFINITY);
    assert_eq!(scalb(1.0, -1074), f64::from_bits(1));
    assert_eq!(scalb(1.0, -3000), 0.0);
    assert_eq!(scalb(3.0, 2000), f64::INFINITY);
    assert_eq!(scalb(2f64.powi(1000), -1500), 2f64.powi(-500));
}

#[test]
fn cell_id_hierarchy() {
    let leaf = S2CellId::from_lat_lng(&S2LatLng::from_degrees(37.0, -122.0));
    assert!(leaf.is_leaf() && leaf.is_valid());
    assert_eq!(leaf.level(), MAX_LEVEL);
    let mut child = leaf;
    for level in (0..MAX_LEVEL).rev() {
        let parent = child.parent(level);
        assert_eq!(parent.level(), level);
        assert_eq!(child.parent_one(), parent);
        assert!(parent.contains(&child) && parent.intersects(&child));
        assert!(parent.range_min() <= child && child <= parent.range_max());
        assert_eq!(parent.child_begin(level + 1), parent.child_begin_one());
        let end = parent.child_end(level + 1);
        let mut c = parent.child_begin(level + 1);
        let mut n = 0;
        while c != end {
            assert_eq!(c.next().prev(), c);
            c = c.next();
            n += 1;
        }
        assert_eq!(n, 4);
        child = parent;
    }
    assert!(child.is_face());
    assert_eq!(child.face(), leaf.face());
    assert_eq!(child.pos(), 1 << (POS_BITS - 1));
    assert_eq!(S2CellId::none().to_token(), "X");
    assert!(!S2CellId::none().is_valid());
    assert_eq!(S2CellId::from_face_pos_level(3, 0, 0).to_token(), "7");
    assert!(S2CellId::new(-1) > S2CellId::new(1), "unsigned order");
}

#[test]
fn face_ij_round_trip_and_cell_vertices() {
    for (face, i, j) in [
        (0, 0, 0),
        (2, 12345, 678_901),
        (5, MAX_SIZE - 1, MAX_SIZE - 1),
    ] {
        let id = S2CellId::from_face_ij(face, i, j);
        let (f, i2, j2, _) = id.to_face_ij_orientation();
        assert_eq!((f, i2, j2), (face, i, j));
    }
    let cell = S2Cell::new(S2CellId::from_face_pos_level(1, 0, 0));
    assert_eq!(cell.level, 0);
    let v = cell.vertex(0);
    assert!((v.norm2() - 1.0).abs() < 1e-15);
}

#[test]
fn java_round_ties_up() {
    assert_eq!(java_round(0.5), 1);
    assert_eq!(java_round(-0.5), 0);
    assert_eq!(java_round(0.49999999999999994), 0);
    assert_eq!(java_round(f64::NAN), 0);
    assert_eq!(java_round(-2.5), -2);
}
