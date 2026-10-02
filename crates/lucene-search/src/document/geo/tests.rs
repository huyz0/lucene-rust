//! The geo queries' own edges over small indexes this port writes: schema
//! mismatches, fields missing from a segment, the dense, inverse and
//! all-documents paths, the comparator source, and the factories' special
//! cases. Their agreement with Lucene is `tests/geo_points_fixtures.rs`.

use lucene_index::document::{self as d, Document, IndexableField, Store};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_store::FsDirectory;
use lucene_util::geo::{Circle, LatLonGeometry, Line, Point, Polygon, Rectangle, XYGeometry};
use lucene_util::test_support::TempDir;

use super::*;
use crate::directory_reader::DirectoryReader;
use crate::document::{search_all, MatchAllDocs};

const VERSION: LuceneVersion = LuceneVersion {
    major: 10,
    minor: 5,
    bugfix: 0,
};

/// Writes `segments` (one commit each) and opens the index.
fn index(tmp: &TempDir, segments: Vec<Vec<Vec<Box<dyn IndexableField>>>>) -> DirectoryReader {
    let dir = FsDirectory::open(tmp.path());
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
    for seg in segments {
        for fields in seg {
            let mut doc = Document::new();
            for f in fields {
                doc.add_boxed(f);
            }
            w.add_fields_document(&doc).unwrap();
        }
        w.commit().unwrap();
    }
    drop(w);
    DirectoryReader::open(&dir).unwrap()
}

fn ll(field: &str, lat: f64, lon: f64) -> Vec<Box<dyn IndexableField>> {
    vec![
        Box::new(d::LatLonPoint::new(field, lat, lon).unwrap()),
        Box::new(d::LatLonDocValuesField::new(field, lat, lon).unwrap()),
    ]
}

fn docs(leaves: &[OpenSegment<'_>], q: &dyn DocumentQuery) -> Vec<i32> {
    search_all(leaves, q)
        .unwrap()
        .into_iter()
        .map(|h| h.doc_id)
        .collect()
}

#[test]
fn wrong_schemas_are_errors_and_missing_fields_match_nothing() {
    let tmp = TempDir::new("geo-schema");
    let r = index(
        &tmp,
        vec![vec![vec![
            Box::new(d::IntPoint::new("int", &[1]).unwrap()),
            Box::new(d::SortedDocValuesField::new("kw", "x")),
            Box::new(d::StringField::new("s", "v", Store::No)),
        ]]],
    );
    let opened = r.open_segments().unwrap();
    let leaves = opened.as_open_segments();
    let e = search_all(
        &leaves,
        &LatLonPointDistanceQuery::new("int", 0.0, 0.0, 10.0).unwrap(),
    )
    .unwrap_err();
    assert!(e.to_string().contains("numDims=1"), "{e}");
    let e = search_all(
        &leaves,
        &xy_point_field::new_box_query("int", 0.0, 1.0, 0.0, 1.0).unwrap(),
    )
    .unwrap_err();
    assert!(e.to_string().contains("really a XYPoint"), "{e}");
    let sort = lat_lon_doc_values_field::new_distance_sort("kw", 0.0, 0.0).unwrap();
    let e = sort.search(&leaves, &MatchAllDocs, 3).unwrap_err();
    assert!(e.to_string().contains("docValuesType=SORTED"), "{e}");
    // `n == 0` is refused before the field is looked at, as
    // `TopFieldCollectorManager` refuses it; for both sorts, any index.
    let e = sort.search(&leaves, &MatchAllDocs, 0).unwrap_err();
    assert!(e.to_string().contains("numHits must be > 0"), "{e}");
    let e = xy_doc_values_field::new_distance_sort("kw", 0.0, 0.0)
        .search(&[], &MatchAllDocs, 0)
        .unwrap_err();
    assert!(e.to_string().contains("numHits must be > 0"), "{e}");
    let e = xy_doc_values_field::new_distance_sort("kw", 0.0, 0.0)
        .search(&leaves, &MatchAllDocs, 3)
        .unwrap_err();
    assert!(e.to_string().contains("XYDocValuesField"), "{e}");
    let feature = LatLonPointDistanceFeatureQuery::new("kw", 0.0, 0.0, 1.0).unwrap();
    assert!(docs(&leaves, &feature).is_empty(), "no points, no scorer");
    // A doc-values query over a field of another type matches nothing.
    let dv = lat_lon_doc_values_field::new_slow_box_query("kw", -1.0, 1.0, -1.0, 1.0).unwrap();
    assert!(docs(&leaves, dv.as_ref()).is_empty());
    // Fields the segment never saw.
    for q in [
        lat_lon_point::new_distance_query("none", 0.0, 0.0, 1.0).unwrap(),
        lat_lon_point::new_polygon_query(
            "s",
            &[Polygon::new(&[0.0, 0.0, 1.0, 0.0], &[0.0, 1.0, 1.0, 0.0], vec![]).unwrap()],
        )
        .unwrap(),
        Box::new(LatLonPointDistanceFeatureQuery::new("none", 0.0, 0.0, 1.0).unwrap()),
        Box::new(xy_point_field::new_distance_query("none", 0.0, 0.0, 1.0).unwrap()),
        Box::new(xy_doc_values_field::new_slow_distance_query("s", 0.0, 0.0, 1.0).unwrap()),
    ] {
        assert!(docs(&leaves, q.as_ref()).is_empty());
    }
    let s = lat_lon_doc_values_field::new_distance_sort("none", 0.0, 0.0)
        .unwrap()
        .search(&leaves, &MatchAllDocs, 3)
        .unwrap();
    assert_eq!(s.hits, vec![(0, f64::INFINITY)]);
    let e = lat_lon_point::nearest(&leaves, "int", 0.0, 0.0, 2).unwrap_err();
    assert!(e.to_string().contains("not a geo point's"), "{e}");
    // The spatial and feature queries refuse it too, where Java would fail
    // decoding a four-byte value as two dimensions.
    let poly = lat_lon_point::new_polygon_query(
        "int",
        &[Polygon::new(&[0.0, 0.0, 1.0, 0.0], &[0.0, 1.0, 1.0, 0.0], vec![]).unwrap()],
    )
    .unwrap();
    let e = search_all(&leaves, poly.as_ref()).unwrap_err();
    assert!(e.to_string().contains("not a geo point's"), "{e}");
    let feature = LatLonPointDistanceFeatureQuery::new("int", 0.0, 0.0, 1.0).unwrap();
    assert!(search_all(&leaves, &feature).is_err());
    let n = lat_lon_point::nearest(&leaves, "s", 0.0, 0.0, 2).unwrap();
    assert_eq!(n.total_hits, 0);
}

/// Every document of a segment has exactly one point: the dense and inverse
/// paths of the distance and spatial queries run.
#[test]
fn dense_single_valued_segments_take_the_inverse_paths() {
    let tmp = TempDir::new("geo-dense");
    let pts: Vec<(f64, f64)> = (0..300)
        .map(|i| (f64::from(i % 30) - 15.0, f64::from(i / 30) * 3.0 - 15.0))
        .collect();
    let r = index(
        &tmp,
        vec![pts.iter().map(|&(a, b)| ll("p", a, b)).collect()],
    );
    let opened = r.open_segments().unwrap();
    let leaves = opened.as_open_segments();
    // Most documents within 2000 km: the inverse visitor.
    let big = lat_lon_point::new_distance_query("p", 0.0, 0.0, 2_000_000.0).unwrap();
    let small = lat_lon_point::new_distance_query("p", 0.0, 0.0, 100_000.0).unwrap();
    let want = |q: &dyn Fn(f64, f64) -> bool| -> Vec<i32> {
        (0..pts.len())
            .filter(|&i| q(pts[i].0, pts[i].1))
            .map(|i| i as i32)
            .collect()
    };
    let within = |rad: f64| {
        move |a: f64, b: f64| lucene_util::sloppy_math::haversin_meters(0.0, 0.0, a, b) <= rad
    };
    assert_eq!(docs(&leaves, big.as_ref()), want(&within(2_000_000.0)));
    assert_eq!(docs(&leaves, small.as_ref()), want(&within(100_000.0)));
    // A box covering everything: every document, without a walk.
    let rect = LatLonGeometry::Rectangle(Rectangle::new(-80.0, 80.0, -170.0, 170.0).unwrap());
    let poly = LatLonGeometry::Polygon(
        Polygon::new(
            &[-80.0, -80.0, 80.0, 80.0, -80.0],
            &[-170.0, 170.0, 170.0, -170.0, -170.0],
            vec![],
        )
        .unwrap(),
    );
    let all =
        lat_lon_point::new_geometry_query("p", QueryRelation::Within, std::slice::from_ref(&poly))
            .unwrap();
    assert_eq!(docs(&leaves, all.as_ref()).len(), 300);
    let none = lat_lon_point::new_geometry_query("p", QueryRelation::Disjoint, &[poly]).unwrap();
    assert!(docs(&leaves, none.as_ref()).is_empty());
    // DISJOINT from a small box: the inverse sparse scorer.
    let small_box = LatLonGeometry::Rectangle(Rectangle::new(-1.5, 1.5, -1.5, 1.5).unwrap());
    let dis = lat_lon_point::new_geometry_query(
        "p",
        QueryRelation::Disjoint,
        std::slice::from_ref(&small_box),
    )
    .unwrap();
    let inside = |a: f64, b: f64| (-1.5..=1.5).contains(&a) && (-1.5..=1.5).contains(&b);
    assert_eq!(docs(&leaves, dis.as_ref()), want(&|a, b| !inside(a, b)));
    let int = lat_lon_point::new_geometry_query("p", QueryRelation::Within, &[small_box]).unwrap();
    assert_eq!(docs(&leaves, int.as_ref()), want(&inside));
    let _ = rect;
}

/// Several points per document: `WITHIN`/`DISJOINT` need every point, so the
/// dense scorers (and `hasAnyHits`) run.
#[test]
fn multi_valued_documents_take_the_dense_paths() {
    let tmp = TempDir::new("geo-multi");
    let segment = || {
        let mut seg = Vec::new();
        for i in 0..200 {
            let a = f64::from(i % 20) - 10.0;
            let mut fields = ll("p", a, a);
            if i % 3 == 0 {
                fields.extend(ll("p", -a, a + 0.5));
            }
            if i % 7 == 0 {
                // no point at all
                fields = vec![Box::new(d::StringField::new("s", "x", Store::No))];
            }
            seg.push(fields);
        }
        seg
    };
    let r = index(&tmp, vec![segment(), segment()]);
    let opened = r.open_segments().unwrap();
    let leaves = opened.as_open_segments();
    let square = |h: f64| {
        LatLonGeometry::Polygon(
            Polygon::new(&[-h, -h, h, h, -h], &[-h, h, h, -h, -h], vec![]).unwrap(),
        )
    };
    let count = |rel: QueryRelation, h: f64| {
        let q = lat_lon_point::new_geometry_query("p", rel, &[square(h)]).unwrap();
        let dvq =
            lat_lon_doc_values_field::new_slow_geometry_query("p", rel, &[square(h)]).unwrap();
        let a = docs(&leaves, q.as_ref());
        assert_eq!(a, docs(&leaves, dvq.as_ref()), "{rel} {h}");
        a.len()
    };
    let within = count(QueryRelation::Within, 5.0);
    let intersects = count(QueryRelation::Intersects, 5.0);
    let disjoint = count(QueryRelation::Disjoint, 5.0);
    assert!(within > 0 && within < intersects);
    assert_eq!(intersects + disjoint, 2 * (200 - 29));
    // A square holding no point: `hasAnyHits` says so before any scorer.
    let far = LatLonGeometry::Polygon(
        Polygon::new(
            &[50.0, 50.0, 51.0, 51.0, 50.0],
            &[50.0, 51.0, 51.0, 50.0, 50.0],
            vec![],
        )
        .unwrap(),
    );
    let q = lat_lon_point::new_geometry_query("p", QueryRelation::Within, &[far]).unwrap();
    assert!(docs(&leaves, q.as_ref()).is_empty());
    assert_eq!(count(QueryRelation::Disjoint, 50.0), 0);
    // CONTAINS a point some documents hold.
    let at = d::LatLonPoint::encode(3.0, 3.0).unwrap();
    let p = LatLonGeometry::Point(
        Point::new(
            lucene_util::geo::GeoEncodingUtils::decode_latitude_bytes(&at, 0),
            lucene_util::geo::GeoEncodingUtils::decode_longitude_bytes(&at, 4),
        )
        .unwrap(),
    );
    let q =
        lat_lon_point::new_geometry_query("p", QueryRelation::Contains, &[p.clone(), p.clone()])
            .unwrap();
    let dvq = lat_lon_doc_values_field::new_slow_geometry_query(
        "p",
        QueryRelation::Contains,
        std::slice::from_ref(&p),
    )
    .unwrap();
    let hits = docs(&leaves, q.as_ref());
    assert!(!hits.is_empty());
    assert_eq!(hits, docs(&leaves, dvq.as_ref()));
    // CONTAINS anything but points matches nothing.
    let none =
        lat_lon_point::new_geometry_query("p", QueryRelation::Contains, &[square(1.0)]).unwrap();
    assert!(docs(&leaves, none.as_ref()).is_empty());
    let none = lat_lon_doc_values_field::new_slow_geometry_query(
        "p",
        QueryRelation::Contains,
        &[square(1.0)],
    )
    .unwrap();
    assert!(docs(&leaves, none.as_ref()).is_empty());
    // The deleted-free sorted search and the comparator source agree.
    let sort = lat_lon_doc_values_field::new_distance_sort("p", 1.0, 1.0).unwrap();
    let top = sort.search(&leaves, &MatchAllDocs, 5).unwrap();
    assert_eq!(top.total_hits, 400);
    assert_eq!(top.hits.len(), 5);
    let src = sort.comparator_source();
    let cmp = src.new_comparator("p", 5, false);
    let leaf = cmp
        .leaf(crate::top_field::LeafCtx {
            reader: &r.segment_readers()[0],
            doc_base: 0,
        })
        .ok()
        .unwrap();
    let _ = leaf;
    let x = xy_doc_values_field::new_distance_sort("p", 0.0, 0.0).comparator_source();
    let xc = x.new_comparator("p", 1, false);
    assert_eq!(
        xc.compare_values(
            &crate::top_field::SortValue::Long(1),
            &crate::top_field::SortValue::Long(2)
        ),
        std::cmp::Ordering::Less
    );
    assert_eq!(
        xc.compare_values(
            &crate::top_field::SortValue::Bytes(None),
            &crate::top_field::SortValue::Long(2)
        ),
        std::cmp::Ordering::Equal
    );
}

#[test]
fn factories_special_cases() {
    // `minLatitude == 90`, `minLongitude == maxLongitude == 180`: nothing.
    for (q, dv) in [
        (
            lat_lon_point::new_box_query("p", 90.0, 90.0, 0.0, 1.0).unwrap(),
            lat_lon_doc_values_field::new_slow_box_query("p", 90.0, 90.0, 0.0, 1.0).unwrap(),
        ),
        (
            lat_lon_point::new_box_query("p", 0.0, 1.0, 180.0, 180.0).unwrap(),
            lat_lon_doc_values_field::new_slow_box_query("p", 0.0, 1.0, 180.0, 180.0).unwrap(),
        ),
    ] {
        assert!(format!("{q:?}").contains("MatchNoDocs"));
        assert!(format!("{dv:?}").contains("MatchNoDocs"));
    }
    // `minLongitude == 180` with a smaller max wraps to -180.
    let q = lat_lon_point::new_box_query("p", 0.0, 1.0, 180.0, 10.0).unwrap();
    assert!(format!("{q:?}").contains("PointRangeQuery"));
    let q = lat_lon_doc_values_field::new_slow_box_query("p", 0.0, 1.0, 180.0, 10.0).unwrap();
    assert!(format!("{q:?}").contains("crosses_dateline: false"));
    assert!(lat_lon_point::new_box_query("p", 0.0, 91.0, 0.0, 1.0).is_err());
    assert!(lat_lon_point::new_box_query("p", -91.0, 0.0, 0.0, 1.0).is_err());
    // A single rectangle or circle intersected is a box or distance query.
    let rect = LatLonGeometry::Rectangle(Rectangle::new(0.0, 1.0, 0.0, 1.0).unwrap());
    let q = lat_lon_point::new_geometry_query(
        "p",
        QueryRelation::Intersects,
        std::slice::from_ref(&rect),
    )
    .unwrap();
    assert!(format!("{q:?}").contains("PointRangeQuery"));
    let q =
        lat_lon_doc_values_field::new_slow_geometry_query("p", QueryRelation::Intersects, &[rect])
            .unwrap();
    assert!(format!("{q:?}").contains("LatLonDocValuesBoxQuery"));
    let circle = LatLonGeometry::Circle(Circle::new(0.0, 0.0, 10.0).unwrap());
    let q = lat_lon_point::new_geometry_query("p", QueryRelation::Intersects, &[circle]).unwrap();
    assert!(format!("{q:?}").contains("LatLonPointDistanceQuery"));
    let line = LatLonGeometry::Line(Line::new(&[0.0, 1.0], &[0.0, 1.0]).unwrap());
    assert!(lat_lon_point::new_geometry_query("p", QueryRelation::Within, &[line]).is_err());
    assert!(lat_lon_doc_values_field::new_slow_distance_query("p", 0.0, 0.0, -1.0).is_err());
    let q = lat_lon_point::new_distance_feature_query("p", 2.0, 0.0, 0.0, 10.0).unwrap();
    assert!(format!("{q:?}").contains("Boosted"));
    assert!(lat_lon_point::new_distance_feature_query("p", 1.0, 0.0, 0.0, -1.0).is_err());
    // The cartesian factories validate.
    assert!(xy_point_field::new_box_query("p", 1.0, 0.0, 0.0, 1.0).is_err());
    assert!(xy_point_field::new_distance_query("p", 0.0, 0.0, -1.0).is_err());
    assert!(xy_doc_values_field::new_slow_box_query("p", 1.0, 0.0, 0.0, 1.0).is_err());
    assert!(xy_doc_values_field::new_slow_distance_query("p", 0.0, 0.0, -1.0).is_err());
    let poly =
        lucene_util::geo::XYPolygon::new(&[0.0, 1.0, 1.0, 0.0], &[0.0, 0.0, 1.0, 0.0], vec![])
            .unwrap();
    assert!(xy_point_field::new_polygon_query("p", std::slice::from_ref(&poly)).is_ok());
    assert!(xy_doc_values_field::new_slow_polygon_query("p", std::slice::from_ref(&poly)).is_ok());
    assert!(xy_point_field::new_geometry_query("p", &[XYGeometry::Polygon(poly)]).is_ok());
    assert_eq!(QueryRelation::Disjoint.to_string(), "DISJOINT");
    assert_eq!(QueryRelation::Contains.to_string(), "CONTAINS");
    assert_eq!(QueryRelation::Intersects.to_string(), "INTERSECTS");
}

#[test]
fn cartesian_queries_and_sorts() {
    let tmp = TempDir::new("geo-xy");
    let seg: Vec<Vec<Box<dyn IndexableField>>> = (0..100)
        .map(|i| {
            let (x, y) = (i as f32, -(i as f32));
            vec![
                Box::new(d::XYPointField::new("xy", x, y).unwrap()) as Box<dyn IndexableField>,
                Box::new(d::XYDocValuesField::new("xy", x, y).unwrap()),
            ]
        })
        .collect();
    let r = index(&tmp, vec![seg]);
    let opened = r.open_segments().unwrap();
    let leaves = opened.as_open_segments();
    let q = xy_point_field::new_box_query("xy", 10.0, 20.0, -15.0, 0.0).unwrap();
    let dvq = xy_doc_values_field::new_slow_box_query("xy", 10.0, 20.0, -15.0, 0.0).unwrap();
    assert_eq!(docs(&leaves, &q), (10..=15).collect::<Vec<_>>());
    assert_eq!(docs(&leaves, &q), docs(&leaves, &dvq));
    let sort = xy_doc_values_field::new_distance_sort("xy", 50.0, -50.0);
    let top = sort.search(&leaves, &MatchAllDocs, 3).unwrap();
    assert_eq!(top.hits[0], (50, 0.0));
    assert_eq!(top.hits.len(), 3);
    // Past 1024 bottoms the box is rebuilt every 64th time only: a reverse
    // walk keeps replacing the bottom.
    let far = xy_doc_values_field::new_distance_sort("xy", 1000.0, -1000.0);
    let top = far.search(&leaves, &MatchAllDocs, 1).unwrap();
    assert_eq!(top.hits[0].0, 99);
}

#[test]
fn numeric_doc_values_read_as_a_singleton_and_other_types_are_refused() {
    let tmp = TempDir::new("geo-numeric-dv");
    let v = d::LatLonDocValuesField::encode(10.0, 20.0).unwrap();
    let r = index(
        &tmp,
        vec![vec![
            vec![
                Box::new(d::LatLonPoint::new("n", 10.0, 20.0).unwrap()) as Box<dyn IndexableField>,
                Box::new(d::NumericDocValuesField::new("n", v)),
            ],
            vec![Box::new(d::BinaryDocValuesField::new("b", vec![1u8]))],
        ]],
    );
    let opened = r.open_segments().unwrap();
    let leaves = opened.as_open_segments();
    // `DocValues.getSortedNumeric` reads a NUMERIC column as a singleton.
    let feature = LatLonPointDistanceFeatureQuery::new("n", 10.0, 20.0, 100.0).unwrap();
    let hits = search_all(&leaves, &feature).unwrap();
    assert_eq!(hits.len(), 1);
    assert!(hits[0].score > 0.99, "{hits:?}");
    // The sort refuses it (`checkCompatible`), and any other type too.
    let e = lat_lon_doc_values_field::new_distance_sort("n", 10.0, 20.0)
        .unwrap()
        .search(&leaves, &MatchAllDocs, 2)
        .unwrap_err();
    assert!(e.to_string().contains("docValuesType=NUMERIC"), "{e}");
    let info = crate::document::field_info(&leaves[0], "b")
        .unwrap()
        .unwrap();
    let e = sorted_numeric(&leaves[0], info).err().unwrap();
    assert!(
        e.to_string().contains("unexpected docvalues type BINARY"),
        "{e}"
    );
    assert_eq!(QueryRelation::Within.to_string(), "WITHIN");
}

#[test]
fn geo3d_query_schema_and_comparator_sources() {
    use crate::top_field::{LeafCtx, SortValue};
    use lucene_util::spatial3d::PlanetModel;
    let pm = PlanetModel::wgs84();
    let tmp = TempDir::new("geo3d-schema");
    let g3 = |lat: f64, lon: f64| -> Vec<Box<dyn IndexableField>> {
        let p = d::geo3d::from_degrees;
        let point = lucene_util::spatial3d::GeoPoint::from_lat_lon(&pm, p(lat), p(lon)).unwrap();
        vec![
            Box::new(d::Geo3DPoint::new("p", lat, lon).unwrap()),
            Box::new(d::Geo3DDocValuesField::new("p", &point, &pm).unwrap()),
            Box::new(d::LatLonPoint::new("ll", lat, lon).unwrap()),
        ]
    };
    let r = index(&tmp, vec![vec![g3(0.0, 0.0), g3(0.0, 1.0), vec![]]]);
    let opened = r.open_segments().unwrap();
    let leaves = opened.as_open_segments();
    let q = geo3d::geo3d_point::new_distance_query("p", &pm, 0.0, 0.0, 50_000.0).unwrap();
    assert_eq!(docs(&leaves, q.as_ref()), vec![0]);
    // A field of two-dimension points is not a Geo3DPoint's.
    let wrong = geo3d::geo3d_point::new_distance_query("ll", &pm, 0.0, 0.0, 50_000.0).unwrap();
    let e = search_all(&leaves, wrong.as_ref()).unwrap_err();
    assert!(e.to_string().contains("not a Geo3DPoint's"), "{e}");
    // The comparator sources report each document's key; missing is last.
    let ctx = || LeafCtx {
        reader: &r.segment_readers()[0],
        doc_base: 0,
    };
    let near = geo3d::geo3d_doc_values_field::new_distance_sort("p", 0.0, 0.0, 1e6, &pm)
        .unwrap()
        .comparator_source();
    let outside = geo3d::geo3d_doc_values_field::new_outside_distance_sort("p", 0.0, 0.0, 1e4, &pm)
        .unwrap()
        .comparator_source();
    let mut keys = Vec::new();
    for src in [near, outside] {
        let cmp = src.new_comparator("p", 3, false);
        let mut leaf = cmp.leaf(ctx()).ok().unwrap();
        for doc in 0..3 {
            match leaf.value(doc, 0.0).unwrap() {
                SortValue::Long(v) => keys.push(v),
                other => panic!("{other:?}"),
            }
        }
    }
    assert!(keys[0] < keys[1] && keys[1] < keys[2], "{keys:?}");
    assert!(keys[3] < keys[4] && keys[4] < keys[5], "{keys:?}");
}
