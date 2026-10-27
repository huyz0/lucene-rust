//! `BaseGeoPointTestCase`'s point streams and `nextGeometry` against the
//! actual Java ones: `fixtures/src/testframework/GenBaseGeoPoint.java`
//! seeded each test method with a `java.util.Random`, captured the arrays
//! its verify hands the four verifiers (quantizing as the identity), and
//! recorded their `Arrays.hashCode` and the stream's next int.

use lucene_index::document::Document;
use lucene_search::query::Clause;
use lucene_test_framework::geo::base_geo_point::{BaseGeoPoint, GeoPointCase};
use lucene_test_framework::geo::geo_test_util::Rect;
use lucene_util::geo::{LatLonGeometry, Polygon};
use lucene_util::java_random::JavaRandom;

fn fixture(name: &str) -> String {
    std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/testframework/data")
            .join(format!("base_geo_point_{name}.tsv")),
    )
    .expect("run scripts/gen-test-framework-fixtures.sh --only GenBaseGeoPoint")
}

/// Generates only: indexing or querying is a bug in the fixture's use.
struct Capture;

impl GeoPointCase for Capture {
    fn add_point_to_doc(&self, _: &str, _: &mut Document, _: f64, _: f64) -> Result<(), String> {
        unreachable!("capture must not index")
    }
    fn new_rect_query(&self, _: &str, _: &Rect) -> Result<Clause, String> {
        unreachable!("capture must not query")
    }
    fn new_distance_query(&self, _: &str, _: f64, _: f64, _: f64) -> Result<Clause, String> {
        unreachable!("capture must not query")
    }
    fn new_polygon_query(&self, _: &str, _: &[Polygon]) -> Result<Clause, String> {
        unreachable!("capture must not query")
    }
    fn new_geometry_query(&self, _: &str, _: &[LatLonGeometry]) -> Result<Clause, String> {
        unreachable!("capture must not query")
    }
}

/// Java's `Arrays.hashCode(double[])`.
fn java_hash(values: &[f64]) -> i32 {
    values.iter().fold(1i32, |h, v| {
        // `doubleToLongBits`: every NaN is the canonical one.
        let bits = if v.is_nan() {
            0x7ff8_0000_0000_0000u64
        } else {
            v.to_bits()
        };
        #[allow(clippy::cast_possible_truncation)]
        let e = (bits ^ (bits >> 32)) as i32;
        h.wrapping_mul(31).wrapping_add(e)
    })
}

fn bits(values: &[f64]) -> String {
    values
        .iter()
        .map(|v| v.to_bits().to_string())
        .collect::<Vec<_>>()
        .join(",")
}

fn geometry(g: &LatLonGeometry) -> String {
    match g {
        LatLonGeometry::Rectangle(r) => {
            format!(
                "box:{}",
                bits(&[r.min_lat, r.max_lat, r.min_lon, r.max_lon])
            )
        }
        LatLonGeometry::Circle(c) => format!("circle:{}", bits(&[c.lat(), c.lon(), c.radius()])),
        LatLonGeometry::Polygon(p) => {
            assert!(p.holes().is_empty());
            format!("poly:{};{}", bits(p.poly_lats()), bits(p.poly_lons()))
        }
        other => panic!("nextGeometry does not make {other:?}"),
    }
}

#[test]
fn point_streams_match_java() {
    let mut rows = 0;
    for line in fixture("points").lines() {
        let f: Vec<&str> = line.split('\t').collect();
        let seed: i64 = f[1].parse().unwrap();
        let mut r = JavaRandom::new(seed);
        let mut base = BaseGeoPoint::new(Capture, &mut r, 1);
        let (lats, lons) = match f[0] {
            "random-tiny" => base.random_points(10),
            "random-medium" => base.random_points(1000),
            "same-point" => base.same_point_points(),
            "low-cardinality" => base.low_cardinality_points(),
            "all-lat" => base.all_one_equal_points(true),
            "all-lon" => base.all_one_equal_points(false),
            other => panic!("{other}"),
        };
        assert_eq!(f[3], "rectangles,distances,polygons,geometries");
        assert_eq!(lats.len(), f[2].parse::<usize>().unwrap(), "{line}");
        assert_eq!(java_hash(&lats), f[4].parse::<i32>().unwrap(), "{line}");
        assert_eq!(java_hash(&lons), f[5].parse::<i32>().unwrap(), "{line}");
        assert_eq!(r.next_int(), f[6].parse::<i32>().unwrap(), "{line}");
        rows += 1;
    }
    assert_eq!(rows, 20 * 6);
}

#[test]
fn next_geometry_matches_java() {
    let mut rows = 0;
    let mut stream: Option<(i64, JavaRandom)> = None;
    for line in fixture("geometry").lines() {
        let f: Vec<&str> = line.split('\t').collect();
        let (seed, at): (i64, u32) = (f[0].parse().unwrap(), f[1].parse().unwrap());
        if at == 0 {
            stream = Some((seed, JavaRandom::new(seed)));
        }
        let (s, r) = stream.as_mut().unwrap();
        assert_eq!(*s, seed);
        let got: Vec<String> = BaseGeoPoint::new(Capture, r, 1)
            .next_geometry()
            .iter()
            .map(geometry)
            .collect();
        assert_eq!(got.join("|"), f[2], "{line}");
        assert_eq!(r.next_int(), f[3].parse::<i32>().unwrap(), "{line}");
        rows += 1;
    }
    assert_eq!(rows, 20 * 8);
}

#[test]
fn java_hash_canonicalizes_nan() {
    let other_nan = f64::from_bits(0x7ff0_0000_0000_0001);
    assert_eq!(java_hash(&[other_nan]), java_hash(&[f64::NAN]));
    assert_eq!(java_hash(&[]), 1);
}
