#![allow(clippy::arithmetic_side_effects)]

use super::*;
use proptest::prelude::*;

fn tri(kind: TriangleType, v: [i32; 6], ab: bool, bc: bool, ca: bool) -> DecodedTriangle {
    DecodedTriangle::new(kind, v[0], v[1], ab, v[2], v[3], bc, v[4], v[5], ca)
}

/// The vertices of a triangle as a set, for comparing a rotation.
fn vertices(t: &DecodedTriangle) -> Vec<(i32, i32)> {
    let mut v = vec![(t.a_x, t.a_y), (t.b_x, t.b_y), (t.c_x, t.c_y)];
    v.sort_unstable();
    v
}

proptest! {
    /// Encoding keeps the vertices (rotated, made counter-clockwise) and
    /// re-encoding the decoded triangle gives the same bytes.
    #[test]
    fn encode_decode_round_trips(
        v in prop::array::uniform6(-50i32..50),
        ab in any::<bool>(), bc in any::<bool>(), ca in any::<bool>(),
    ) {
        let bytes = ShapeField::encode_triangle(v[1], v[0], ab, v[3], v[2], bc, v[5], v[4], ca).unwrap();
        let t = ShapeField::decode_triangle(&bytes);
        let mut want = vec![(v[0], v[1]), (v[2], v[3]), (v[4], v[5])];
        want.sort_unstable();
        if t.kind == TriangleType::Triangle {
            prop_assert_eq!(vertices(&t), want);
            // counter-clockwise or flat
            prop_assert!(orient(t.a_x, t.a_y, t.b_x, t.b_y, t.c_x, t.c_y) >= 0);
            let again = ShapeField::encode_triangle(t.a_y, t.a_x, t.ab, t.b_y, t.b_x, t.bc, t.c_y, t.c_x, t.ca).unwrap();
            prop_assert_eq!(again, bytes);
        }
        // the bounding box is the first four dimensions
        let xs = [v[0], v[2], v[4]];
        let ys = [v[1], v[3], v[5]];
        prop_assert_eq!(dim(&bytes, 0), *ys.iter().min().unwrap());
        prop_assert_eq!(dim(&bytes, 1), *xs.iter().min().unwrap());
        prop_assert_eq!(dim(&bytes, 2), *ys.iter().max().unwrap());
        prop_assert_eq!(dim(&bytes, 3), *xs.iter().max().unwrap());
    }
}

#[test]
fn every_layout_decodes() {
    // One triangle per layout code, as the encoder chooses them.
    let mut seen = [false; 8];
    for v in [
        [0, 0, 10, 10, 2, 5],
        [0, 10, 10, 0, 8, 7],
        [0, 0, 5, 3, 10, 10],
        [0, 10, 5, 3, 10, 0],
        [0, 10, 10, 0, 5, 3],
        [0, 5, 3, 0, 10, 10],
        [0, 5, 10, 0, 8, 10],
        [0, 10, 3, 0, 10, 5],
        [0, 0, 10, 4, 3, 10],
        [0, 0, 10, 2, 4, 10],
        [0, 4, 8, 0, 10, 10],
        [0, 6, 10, 0, 2, 10],
    ] {
        let bytes =
            ShapeField::encode_triangle(v[1], v[0], true, v[3], v[2], false, v[5], v[4], true)
                .unwrap();
        let code = (dim(&bytes, 6) & 7) as usize;
        seen[code] = true;
        let t = ShapeField::decode_triangle(&bytes);
        let mut want = vec![(v[0], v[1]), (v[2], v[3]), (v[4], v[5])];
        want.sort_unstable();
        assert_eq!(vertices(&t), want, "{v:?} code {code}");
    }
    assert_eq!(seen, [true; 8], "every layout");
}

#[test]
fn rotation_keeps_edge_flags_with_their_edges() {
    // b has the minimum x: the triangle rotates to start at b, and the
    // flags rotate with it.
    let bytes = ShapeField::encode_triangle(0, 5, true, 0, 0, false, 5, 2, false).unwrap();
    let t = ShapeField::decode_triangle(&bytes);
    assert_eq!((t.a_x, t.a_y), (0, 0));
    assert!(!t.ab || !t.bc || !t.ca);
    // c has the minimum x
    let bytes = ShapeField::encode_triangle(0, 5, false, 5, 6, false, 1, 0, true).unwrap();
    let t = ShapeField::decode_triangle(&bytes);
    assert_eq!((t.a_x, t.a_y), (0, 1));
    // one meridian: the minimum y first, from b and from c
    for (a, b, c) in [((0, 5), (0, 1), (0, 9)), ((0, 5), (0, 9), (0, 1))] {
        let bytes =
            ShapeField::encode_triangle(a.1, a.0, true, b.1, b.0, false, c.1, c.0, true).unwrap();
        let t = ShapeField::decode_triangle(&bytes);
        assert_eq!(t.a_y, 1);
        assert_eq!(t.kind, TriangleType::Triangle);
    }
}

#[test]
fn degenerate_triangles_resolve_to_points_and_lines() {
    let p = ShapeField::decode_triangle(
        &ShapeField::encode_triangle(3, 4, true, 3, 4, true, 3, 4, true).unwrap(),
    );
    assert_eq!(p.kind, TriangleType::Point);
    assert_eq!((p.a_x, p.a_y), (4, 3));
    // a flat line (a, b, a), as LatLonShape indexes a segment
    let l = ShapeField::decode_triangle(
        &ShapeField::encode_triangle(0, 0, true, 7, 9, true, 0, 0, true).unwrap(),
    );
    assert_eq!(l.kind, TriangleType::Line);
    assert_eq!((l.c_x, l.c_y), (l.a_x, l.a_y));
    assert!(l.ab);
    // the three ways a pair can coincide
    let mut t = tri(
        TriangleType::Triangle,
        [1, 1, 1, 1, 5, 5],
        false,
        true,
        false,
    );
    ShapeField::resolve_triangle_type(&mut t);
    assert_eq!(
        (t.kind, t.ab, t.b_x, t.c_x),
        (TriangleType::Line, true, 5, 1)
    );
    let mut t = tri(
        TriangleType::Triangle,
        [1, 1, 5, 5, 1, 1],
        false,
        true,
        false,
    );
    ShapeField::resolve_triangle_type(&mut t);
    assert_eq!((t.kind, t.ab), (TriangleType::Line, true));
    let mut t = tri(
        TriangleType::Triangle,
        [1, 1, 5, 5, 5, 5],
        false,
        false,
        true,
    );
    ShapeField::resolve_triangle_type(&mut t);
    assert_eq!((t.kind, t.ab, t.c_x), (TriangleType::Line, true, 1));
}

#[test]
fn decoded_triangle_equality_display_and_types() {
    let a = tri(
        TriangleType::Triangle,
        [1, 2, 3, 4, 5, 6],
        true,
        false,
        true,
    );
    let mut b = a;
    b.kind = TriangleType::Line;
    assert_eq!(a, b, "the type is not compared");
    b.bc = true;
    assert_ne!(a, b);
    assert_eq!(a.to_string(), "1, 2 3, 4 5, 6 [true,false,true]");
    for (t, name, ord) in [
        (TriangleType::Point, "POINT", 0),
        (TriangleType::Line, "LINE", 1),
        (TriangleType::Triangle, "TRIANGLE", 2),
    ] {
        assert_eq!(t.to_string(), name);
        assert_eq!(t.ordinal(), ord);
        assert_eq!(TriangleType::from_ordinal(ord), Some(t));
    }
    assert_eq!(TriangleType::from_ordinal(3), None);
    assert_eq!(TriangleType::from_ordinal(-1), None);
    assert_eq!(DecodedTriangle::default().kind, TriangleType::Triangle);
}

#[test]
fn triangle_fields_are_seven_dimension_points() {
    let ft = ShapeField::field_type_of();
    assert_eq!(ft.point_dimension_count(), 7);
    assert_eq!(ft.point_index_dimension_count(), 4);
    assert_eq!(ft.point_num_bytes(), 4);
    let mut t = ShapeTriangle::new("f", 1, 2, 3, 4, 1, 2).unwrap();
    assert_eq!(t.field_name(), "f");
    assert_eq!(IndexableField::name(&t), "f");
    assert_eq!(t.field_type(), &ft);
    assert_eq!(t.binary_value().unwrap().as_ref(), t.packed());
    assert!(t.token_stream(&Analyzer::standard(None)).unwrap().is_none());
    let before = *t.packed();
    t.set_triangle_value(0, 0, true, 9, 0, false, 0, 9, true)
        .unwrap();
    assert_ne!(*t.packed(), before);
    let d = ShapeField::decode_triangle(t.packed());
    assert_eq!(d.kind, TriangleType::Triangle);
}

fn square() -> Polygon {
    Polygon::new(
        &[0.0, 0.0, 10.0, 10.0, 0.0],
        &[0.0, 10.0, 10.0, 0.0, 0.0],
        vec![],
    )
    .unwrap()
}

#[test]
fn lat_lon_factories() {
    let poly = square();
    let fields = LatLonShape::create_indexable_fields("f", &poly).unwrap();
    assert_eq!(fields.len(), 2);
    assert_eq!(
        LatLonShape::create_indexable_fields_checked("f", &poly, true).unwrap(),
        fields
    );
    let line = Line::new(&[0.0, 1.0, 2.0], &[0.0, 1.0, 0.0]).unwrap();
    let lf = LatLonShape::create_line_fields("f", &line).unwrap();
    assert_eq!(lf.len(), 2);
    assert!(lf
        .iter()
        .all(|t| ShapeField::decode_triangle(t.packed()).kind == TriangleType::Line));
    let pf = LatLonShape::create_point_fields("f", 45.0, 90.0).unwrap();
    assert_eq!(
        ShapeField::decode_triangle(pf[0].packed()).kind,
        TriangleType::Point
    );
    assert!(LatLonShape::create_point_fields("f", 91.0, 0.0).is_err());
    assert!(LatLonShape::create_point_doc_value_field("f", 0.0, 181.0).is_err());

    // doc values: every form, and back from their own bytes
    let dv = LatLonShape::create_doc_value_field("f", &poly).unwrap();
    assert_eq!(dv.number_of_terms(), 2);
    assert_eq!(dv.highest_dimension_type(), TriangleType::Triangle);
    let bb = dv.bounding_box();
    assert!(bb.min_lat < 1e-6 && bb.max_lat > 9.99 && bb.min_lon < 1e-6 && bb.max_lon > 9.99);
    let c = dv.centroid();
    assert!(
        (c.lat() - 5.0).abs() < 1e-6 && (c.lon() - 5.0).abs() < 1e-6,
        "{c:?}"
    );
    let checked = LatLonShape::create_doc_value_field_checked("f", &poly, true).unwrap();
    assert_eq!(checked.binary_value(), dv.binary_value());
    let bytes = dv.binary_value().unwrap().into_owned();
    let back = LatLonShape::create_doc_value_field_from_bytes("f", bytes.clone()).unwrap();
    assert_eq!(back.binary_value().unwrap().as_ref(), &bytes[..]);
    assert_eq!(back.centroid(), dv.centroid());
    assert_eq!(dv.field_type(), &LatLonShapeDocValuesField::field_type_of());
    assert_eq!(dv.field_type().doc_values_type(), DocValuesType::Binary);
    assert!(dv.field_type().omit_norms());
    assert_eq!(IndexableField::name(&dv), "f");
    assert!(dv
        .token_stream(&Analyzer::standard(None))
        .unwrap()
        .is_none());
    let read = LatLonShape::create_lat_lon_shape_doc_values(&bytes).unwrap();
    assert_eq!(read.bounding_box(), dv.bounding_box());
    assert_eq!(read.values().number_of_terms(), 2);
    assert_eq!(dv.doc_values().values().encoding(), ShapeEncoding::LatLon);

    let ldv = LatLonShape::create_line_doc_value_field("f", &line).unwrap();
    assert_eq!(ldv.highest_dimension_type(), TriangleType::Line);
    let pdv = LatLonShape::create_point_doc_value_field("f", 45.0, 90.0).unwrap();
    assert_eq!(pdv.highest_dimension_type(), TriangleType::Point);
    assert_eq!(pdv.number_of_terms(), 1);
    let mut all = fields.clone();
    all.extend(lf);
    let multi = LatLonShape::create_doc_value_field_from_fields("f", &all).unwrap();
    assert_eq!(multi.number_of_terms(), 4);
    let tris: Vec<DecodedTriangle> = all
        .iter()
        .map(|t| ShapeField::decode_triangle(t.packed()))
        .collect();
    let same = LatLonShape::create_doc_value_field_from_triangles("f", &tris).unwrap();
    assert_eq!(same.binary_value(), multi.binary_value());
    assert!(LatLonShape::create_doc_value_field_from_triangles("f", &[]).is_err());
    assert!(LatLonShape::create_doc_value_field_from_bytes("f", vec![0]).is_err());

    // a polygon the tessellator rejects
    let bow = Polygon::new(
        &[0.0, 10.0, 0.0, 10.0, 0.0],
        &[0.0, 10.0, 10.0, 0.0, 0.0],
        vec![],
    )
    .unwrap();
    assert!(LatLonShape::create_indexable_fields_checked("f", &bow, true).is_err());
    assert!(LatLonShape::create_doc_value_field_checked("f", &bow, true).is_err());
}

#[test]
fn xy_factories() {
    let poly = XYPolygon::new(
        &[0.0, 10.0, 10.0, 0.0, 0.0],
        &[0.0, 0.0, 10.0, 10.0, 0.0],
        vec![],
    )
    .unwrap();
    let fields = XYShape::create_indexable_fields("f", &poly).unwrap();
    assert_eq!(fields.len(), 2);
    assert_eq!(
        XYShape::create_indexable_fields_checked("f", &poly, true).unwrap(),
        fields
    );
    let line = XYLine::new(&[0.0, 1.0], &[0.0, 1.0]).unwrap();
    assert_eq!(XYShape::create_line_fields("f", &line).unwrap().len(), 1);
    assert_eq!(
        XYShape::create_point_fields("f", 1.0, 2.0).unwrap().len(),
        1
    );
    assert!(XYShape::create_point_fields("f", f32::NAN, 2.0).is_err());
    assert!(XYShape::create_point_doc_value_field("f", f32::INFINITY, 2.0).is_err());

    let dv = XYShape::create_doc_value_field("f", &poly).unwrap();
    assert_eq!(dv.number_of_terms(), 2);
    assert_eq!(dv.centroid().x(), 5.0);
    assert_eq!(dv.bounding_box().max_y, 10.0);
    assert_eq!(dv.field_type(), &XYShapeDocValuesField::field_type_of());
    let bytes = dv.binary_value().unwrap().into_owned();
    let back = XYShape::create_doc_value_field_from_bytes("f", bytes.clone()).unwrap();
    assert_eq!(back.bounding_box(), dv.bounding_box());
    let read = XYShape::create_xy_shape_doc_values(&bytes).unwrap();
    assert_eq!(read.centroid(), dv.centroid());
    assert_eq!(read.values().encoding(), ShapeEncoding::XY);
    assert_eq!(
        XYShape::create_doc_value_field_checked("f", &poly, true)
            .unwrap()
            .binary_value(),
        dv.binary_value()
    );
    let ldv = XYShape::create_line_doc_value_field("f", &line).unwrap();
    assert_eq!(ldv.highest_dimension_type(), TriangleType::Line);
    let pdv = XYShape::create_point_doc_value_field("f", 1.0, 2.0).unwrap();
    assert_eq!(pdv.centroid().y(), 2.0);
    let tris: Vec<DecodedTriangle> = fields
        .iter()
        .map(|t| ShapeField::decode_triangle(t.packed()))
        .collect();
    let a = XYShape::create_doc_value_field_from_triangles("f", &tris).unwrap();
    let b = XYShape::create_doc_value_field_from_fields("f", &fields).unwrap();
    assert_eq!(a.binary_value(), b.binary_value());
    assert_eq!(a.doc_values().values().number_of_terms(), 2);
    let bow = XYPolygon::new(
        &[0.0, 10.0, 0.0, 10.0, 0.0],
        &[0.0, 10.0, 10.0, 0.0, 0.0],
        vec![],
    )
    .unwrap();
    assert!(XYShape::create_indexable_fields_checked("f", &bow, true).is_err());
    assert!(XYShape::create_doc_value_field_checked("f", &bow, true).is_err());
    assert!(XYShape::create_doc_value_field_from_bytes("f", vec![]).is_err());
}

#[test]
fn polygon_doc_values_carry_lucenes_edge_flags() {
    // `createDocValueField(name, polygon)` takes the second edge's flag from
    // the first edge, as Lucene does.
    let poly = square();
    let tess = tessellator::tessellate(&poly, false).unwrap();
    let tris = doc_value_triangles(&tess);
    for (t, d) in tess.iter().zip(&tris) {
        assert_eq!(d.ab, t.is_edge_from_polygon(0));
        assert_eq!(d.bc, t.is_edge_from_polygon(0));
        assert_eq!(d.ca, t.is_edge_from_polygon(2));
        assert_eq!(d.kind, TriangleType::Triangle);
    }
}

#[test]
fn shape_doc_values_reject_invalid_centroid_and_box() {
    // A header whose bounding box has minY > maxY: Java's `Rectangle` only
    // asserts the order (so a lat/lon value reads), `XYRectangle` throws.
    let mut b = vec![VERSION_BYTE];
    let mut push = |v: i64| {
        let mut v = v as u64;
        while v & !0x7F != 0 {
            b.push((v & 0x7F) as u8 | 0x80);
            v >>= 7;
        }
        b.push(v as u8);
    };
    push(1);
    for v in [0i64, 10, 10, 0, 5, 5] {
        push(v - i64::from(i32::MIN));
    }
    push(0);
    assert!(LatLonShapeDocValues::new(Cow::Borrowed(&b)).is_ok());
    assert!(XYShapeDocValues::new(Cow::Borrowed(&b)).is_err());
}

const VERSION_BYTE: u8 = super::super::shape_doc_values::VERSION;
