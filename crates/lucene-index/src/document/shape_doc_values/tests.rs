#![allow(clippy::arithmetic_side_effects)]

use super::*;
use lucene_util::geo::{LatLonGeometry, Rectangle, XYGeometry, XYRectangle};

fn t(kind: TriangleType, v: [i32; 6]) -> DecodedTriangle {
    DecodedTriangle::new(kind, v[0], v[1], true, v[2], v[3], false, v[4], v[5], true)
}

fn latlon_box(min_lat: f64, max_lat: f64, min_lon: f64, max_lon: f64) -> Box<dyn Component2D> {
    LatLonGeometry::create(&[LatLonGeometry::Rectangle(
        Rectangle::new(min_lat, max_lat, min_lon, max_lon).unwrap(),
    )])
    .unwrap()
}

fn enc(lat: f64, lon: f64) -> (i32, i32) {
    (
        GeoEncodingUtils::encode_longitude(lon).unwrap(),
        GeoEncodingUtils::encode_latitude(lat).unwrap(),
    )
}

/// Where [`strip`]'s triangle `i` starts.
fn lon_of(i: usize) -> f64 {
    -100.0 + i as f64 * 0.7
}

/// `count` small triangles along the equator, 0.7 degrees apart.
fn strip(count: i32) -> Vec<DecodedTriangle> {
    (0..count)
        .map(|i| {
            let x = lon_of(i as usize);
            let (ax, ay) = enc(0.0, x);
            let (bx, by) = enc(0.0, x + 0.35);
            let (cx, cy) = enc(0.5, x);
            t(TriangleType::Triangle, [ax, ay, bx, by, cx, cy])
        })
        .collect()
}

#[test]
fn var_int_sizes() {
    assert_eq!(v_long_size(0), 1);
    assert_eq!(v_long_size(127), 1);
    assert_eq!(v_long_size(128), 2);
    assert_eq!(v_long_size(i64::MAX), 9);
    assert_eq!(v_long_size(-1), 10);
    assert_eq!(v_int_size(0), 1);
    assert_eq!(v_int_size(1 << 14), 3);
    assert_eq!(v_int_size(-1), 5);
    for v in [0i64, 1, 127, 128, 300, 1 << 35, i64::MAX] {
        let mut out = Vec::new();
        write_vlong(&mut out, v);
        assert_eq!(out.len() as i32, v_long_size(v));
        let mut r = Reader { data: &out, pos: 0 };
        assert_eq!(r.read_vlong().unwrap(), v);
    }
    for v in [0i32, 1, 127, 128, 1 << 21, i32::MAX, -1, i32::MIN] {
        let mut out = Vec::new();
        write_vint(&mut out, v);
        assert_eq!(out.len() as i32, v_int_size(v));
        let mut r = Reader { data: &out, pos: 0 };
        assert_eq!(r.read_vint().unwrap(), v);
    }
}

#[test]
fn reader_refuses_malformed_varints_and_overflow() {
    let mut r = Reader {
        data: &[0x80, 0x80, 0x80, 0x80, 0x10],
        pos: 0,
    };
    assert!(r.read_vint().is_err(), "a fifth byte with high bits");
    let mut r = Reader {
        data: &[0xFF; 9],
        pos: 0,
    };
    assert!(r.read_vlong().is_err(), "a ninth byte with its high bit");
    let mut r = Reader { data: &[], pos: 0 };
    assert!(r.read_byte().is_err());
    // `toIntExact`: a value that does not fit an int
    let mut out = Vec::new();
    write_vlong(&mut out, i64::from(u32::MAX) + 5);
    let mut r = Reader { data: &out, pos: 0 };
    assert!(r.read_translated().is_err());
    let mut r = Reader { data: &out, pos: 0 };
    assert!(r.read_relative(0).is_err());
    let mut r = Reader {
        data: &[1, 2],
        pos: 0,
    };
    assert!(r.skip_bytes(3).is_err());
    assert!(r.skip_bytes(-1).is_err());
    assert!(r.skip_bytes(2).is_ok());
    assert_eq!(r.pos, 2);
}

#[test]
fn encodings() {
    let ll = ShapeEncoding::LatLon;
    assert_eq!(ll.encode_x(180.0).unwrap(), i32::MAX);
    assert_eq!(ll.encode_y(-90.0).unwrap(), i32::MIN);
    assert!(ll.encode_x(181.0).is_err());
    assert!(ll.encode_y(f64::NAN).is_err());
    assert_eq!(ll.decode_x(0), 0.0);
    let xy = ShapeEncoding::XY;
    let e = xy.encode_x(1.5).unwrap();
    assert_eq!(xy.decode_x(e), 1.5);
    assert_eq!(xy.decode_y(xy.encode_y(-2.25).unwrap()), -2.25);
    assert!(xy.encode_y(f64::INFINITY).is_err());
}

#[test]
fn a_tessellation_round_trips_through_its_header() {
    for n in [1, 2, 3, 7, 40, 257] {
        let tris = strip(n);
        let dv = ShapeDocValues::from_triangles(ShapeEncoding::LatLon, &tris).unwrap();
        assert_eq!(dv.number_of_terms(), n);
        assert_eq!(dv.highest_dimension(), TriangleType::Triangle);
        let min_x = tris.iter().map(|t| t.a_x).min().unwrap();
        let max_x = tris.iter().map(|t| t.b_x).max().unwrap();
        assert_eq!((dv.encoded_min_x(), dv.encoded_max_x()), (min_x, max_x));
        assert_eq!(dv.encoded_min_y(), tris[0].a_y);
        assert_eq!(dv.encoded_max_y(), tris[0].c_y);
        let again =
            ShapeDocValues::from_bytes(ShapeEncoding::LatLon, Cow::Borrowed(dv.binary_value()))
                .unwrap();
        assert_eq!(again.encoded_centroid_x(), dv.encoded_centroid_x());
        assert_eq!(again.encoded_centroid_y(), dv.encoded_centroid_y());
        // every triangle is found by a box around it alone, and a box beside
        // the strip finds nothing
        for (i, tri) in tris.iter().enumerate().step_by(7) {
            let x = lon_of(i);
            let q = latlon_box(0.1, 0.2, x + 0.05, x + 0.1);
            assert_eq!(
                dv.relate(q.as_ref()).unwrap(),
                Relation::CellCrossesQuery,
                "{n} {tri}"
            );
        }
        let q = latlon_box(-5.0, -4.0, -100.0, -99.0);
        assert_eq!(dv.relate(q.as_ref()).unwrap(), Relation::CellOutsideQuery);
        let q = latlon_box(-5.0, 5.0, -101.0, 179.0);
        assert_eq!(dv.relate(q.as_ref()).unwrap(), Relation::CellInsideQuery);
        // between two triangles: inside the bounding box, touching none
        if n > 2 {
            let q = latlon_box(0.4, 0.45, lon_of(1) + 0.4, lon_of(1) + 0.6);
            assert_eq!(dv.relate(q.as_ref()).unwrap(), Relation::CellOutsideQuery);
        }
    }
}

#[test]
fn centroids_by_dimension() {
    // points only: the plain average
    let (ax, ay) = enc(10.0, 20.0);
    let (bx, by) = enc(20.0, 40.0);
    let pts = [
        t(TriangleType::Point, [ax, ay, ax, ay, ax, ay]),
        t(TriangleType::Point, [bx, by, bx, by, bx, by]),
    ];
    let dv = ShapeDocValues::from_triangles(ShapeEncoding::LatLon, &pts).unwrap();
    assert_eq!(dv.highest_dimension(), TriangleType::Point);
    let e = ShapeEncoding::LatLon;
    assert!((e.decode_x(dv.encoded_centroid_x()) - 30.0).abs() < 1e-6);
    assert!((e.decode_y(dv.encoded_centroid_y()) - 15.0).abs() < 1e-6);
    // a point and a line: the line wins
    let line = t(TriangleType::Line, [ax, ay, bx, by, ax, ay]);
    let dv = ShapeDocValues::from_triangles(ShapeEncoding::LatLon, &[pts[0], line]).unwrap();
    assert_eq!(dv.highest_dimension(), TriangleType::Line);
    assert!((e.decode_x(dv.encoded_centroid_x()) - 30.0).abs() < 1e-6);
    // a zero-length line and a zero-area triangle alone keep a zero centroid
    let flat = t(TriangleType::Line, [ax, ay, ax, ay, ax, ay]);
    let dv = ShapeDocValues::from_triangles(ShapeEncoding::LatLon, &[flat]).unwrap();
    assert_eq!(dv.encoded_centroid_x(), 0);
    let dv = ShapeDocValues::from_triangles(ShapeEncoding::LatLon, &[flat, flat]).unwrap();
    assert_eq!(dv.encoded_centroid_x(), 0);
    let zero = t(TriangleType::Triangle, [ax, ay, bx, by, ax, ay]);
    let dv = ShapeDocValues::from_triangles(ShapeEncoding::LatLon, &[zero]).unwrap();
    assert_eq!(dv.encoded_centroid_x(), 0);
    let dv = ShapeDocValues::from_triangles(ShapeEncoding::LatLon, &[zero, zero]).unwrap();
    assert_eq!(dv.encoded_centroid_x(), 0);
    // a single line and a single triangle divide by their own weight
    let dv = ShapeDocValues::from_triangles(ShapeEncoding::LatLon, &[line]).unwrap();
    assert!((e.decode_x(dv.encoded_centroid_x()) - 30.0).abs() < 1e-6);
    let tri = strip(1);
    let dv = ShapeDocValues::from_triangles(ShapeEncoding::LatLon, &tri).unwrap();
    assert!((e.decode_y(dv.encoded_centroid_y()) - 0.5 / 3.0).abs() < 1e-6);
    assert!(ShapeDocValues::from_triangles(ShapeEncoding::LatLon, &[]).is_err());
}

#[test]
fn points_and_lines_relate() {
    let (ax, ay) = enc(10.0, 20.0);
    let (bx, by) = enc(20.0, 40.0);
    let (cx, cy) = enc(-10.0, -20.0);
    let shapes = [
        t(TriangleType::Point, [ax, ay, ax, ay, ax, ay]),
        t(TriangleType::Line, [bx, by, cx, cy, bx, by]),
        t(TriangleType::Point, [cx, cy, cx, cy, cx, cy]),
    ];
    let dv = ShapeDocValues::from_triangles(ShapeEncoding::LatLon, &shapes).unwrap();
    for (q, want) in [
        (
            latlon_box(9.0, 11.0, 19.0, 21.0),
            Relation::CellCrossesQuery,
        ),
        (latlon_box(4.0, 6.0, 9.0, 11.0), Relation::CellCrossesQuery),
        (
            latlon_box(-11.0, -9.0, -21.0, -19.0),
            Relation::CellCrossesQuery,
        ),
        (
            latlon_box(12.0, 13.0, 19.0, 21.0),
            Relation::CellOutsideQuery,
        ),
    ] {
        assert_eq!(dv.relate(q.as_ref()).unwrap(), want);
    }
    // cartesian
    let xy = ShapeDocValues::from_triangles(ShapeEncoding::XY, &strip(9)).unwrap();
    let q = XYGeometry::create(&[XYGeometry::Rectangle(
        XYRectangle::new(-1e30, 1e30, -1e30, 1e30).unwrap(),
    )])
    .unwrap();
    assert_eq!(xy.relate(q.as_ref()).unwrap(), Relation::CellInsideQuery);
    assert_eq!(xy.encoding(), ShapeEncoding::XY);
}

#[test]
fn corrupt_values_are_errors() {
    let dv = ShapeDocValues::from_triangles(ShapeEncoding::LatLon, &strip(40)).unwrap();
    let bytes = dv.binary_value().to_vec();
    // in a gap between two triangles: the walk reads or skips every node,
    // so a truncation fails to read the header or fails the walk -- and
    // never panics
    let q = latlon_box(0.4, 0.45, lon_of(1) + 0.4, lon_of(1) + 0.6);
    assert_eq!(dv.relate(q.as_ref()).unwrap(), Relation::CellOutsideQuery);
    let mut failed = 0;
    for len in 0..bytes.len() {
        let cut = &bytes[..len];
        match ShapeDocValues::from_bytes(ShapeEncoding::LatLon, Cow::Borrowed(cut)) {
            Ok(v) => failed += usize::from(v.relate(q.as_ref()).is_err()),
            Err(_) => failed += 1,
        }
    }
    assert_eq!(failed, bytes.len());
    // an unknown dimension type
    let mut bad = bytes.clone();
    let header_len = {
        let mut r = Reader {
            data: &bytes,
            pos: 0,
        };
        r.read_byte().unwrap();
        r.read_vint().unwrap();
        for _ in 0..6 {
            r.read_vlong().unwrap();
        }
        r.pos
    };
    bad[header_len] = 3;
    let e = ShapeDocValues::from_bytes(ShapeEncoding::LatLon, Cow::Owned(bad)).unwrap_err();
    assert!(e.to_string().contains("out of bounds"), "{e}");
    // a subtree size that skips past the end
    let mut big_skip = bytes.clone();
    // the root's left subtree size follows the root component; make every
    // byte of the tree after the header huge instead and expect an error,
    // never a panic
    for b in big_skip.iter_mut().skip(header_len + 1) {
        *b = 0xFF;
    }
    if let Ok(v) = ShapeDocValues::from_bytes(ShapeEncoding::LatLon, Cow::Owned(big_skip)) {
        let wide = latlon_box(-90.0, 90.0, -90.0, -89.9999);
        let r = v.relate(wide.as_ref());
        assert!(
            r.is_err(),
            "{r:?} {:?}",
            &v.binary_value()[..header_len + 3]
        );
    }
}

/// A tree of single left children, `depth` deep: header, then each node a
/// size, four bounds, a header with only a left child, a point component.
fn left_chain(depth: usize) -> Vec<u8> {
    let mut out = vec![VERSION];
    write_vint(&mut out, i32::try_from(depth + 1).unwrap());
    for v in [0, 10, 0, 10, 5, 5] {
        write_vlong(&mut out, translate(v));
    }
    write_vint(&mut out, 0);
    write_vint(&mut out, 0x02 | 0x04);
    write_vlong(&mut out, 10);
    write_vlong(&mut out, 10);
    for i in 0..depth {
        write_vint(&mut out, 1000);
        for _ in 0..4 {
            write_vlong(&mut out, 0);
        }
        let header = if i + 1 == depth { 0x04 } else { 0x02 | 0x04 };
        write_vint(&mut out, header);
        write_vlong(&mut out, 10);
        write_vlong(&mut out, 10);
    }
    out
}

#[test]
fn a_tree_nested_past_the_limit_is_refused() {
    // a box that does not contain the origin point every node holds
    let q = ShapeEncoding::XY;
    let query = XYGeometry::create(&[XYGeometry::Rectangle(
        XYRectangle::new(
            q.decode_x(1) as f32,
            q.decode_x(20) as f32,
            q.decode_y(1) as f32,
            q.decode_y(20) as f32,
        )
        .unwrap(),
    )])
    .unwrap();
    let ok = ShapeDocValues::from_bytes(ShapeEncoding::XY, Cow::Owned(left_chain(10))).unwrap();
    assert_eq!(
        ok.relate(query.as_ref()).unwrap(),
        Relation::CellOutsideQuery
    );
    let deep = ShapeDocValues::from_bytes(
        ShapeEncoding::XY,
        Cow::Owned(left_chain(MAX_DEPTH as usize + 5)),
    )
    .unwrap();
    let e = deep.relate(query.as_ref()).unwrap_err();
    assert!(e.to_string().contains("too deep"), "{e}");
}
