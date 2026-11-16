#![cfg(test)]
use super::*;
use lucene_index::document::{LatLonDocValuesField, XYDocValuesField};

#[test]
fn missing_values_and_validation() {
    let mut s = LatLonPointSortField::new("f", 1.0, 2.0).unwrap();
    assert!(s.set_missing_value(f64::INFINITY).is_ok());
    let e = s.set_missing_value(0.0).unwrap_err();
    assert!(e.to_string().contains("but got 0.0"), "{e}");
    assert!(LatLonPointSortField::new("f", 100.0, 0.0).is_err());
    assert!(LatLonPointSortField::new("f", 0.0, -200.0).is_err());
    let mut x = XYPointSortField::new("f", 1.0, 2.0);
    assert!(x.set_missing_value(f64::INFINITY).is_ok());
    assert!(x.set_missing_value(f64::NEG_INFINITY).is_err());
}

#[test]
fn helpers_follow_java() {
    assert_eq!(double_compare(1.0, 2.0), Ordering::Less);
    assert_eq!(double_compare(2.0, 2.0), Ordering::Equal);
    assert_eq!(double_compare(f64::INFINITY, 2.0), Ordering::Greater);
    assert!(java_min(f64::NAN, 1.0).is_nan());
    assert!(java_min(0.0, -0.0).is_sign_negative());
    assert!(java_min(-0.0, 0.0).is_sign_negative());
    assert_eq!(java_min(3.0, 2.0), 2.0);
    assert_eq!(haversin2(f64::INFINITY), f64::INFINITY);
}

#[test]
fn top_n_keeps_the_nearest_and_breaks_ties_by_doc() {
    let v = |lat, lon| LatLonDocValuesField::encode(lat, lon).unwrap();
    let mut top = DistanceTopN {
        n: 2,
        queue: BinaryHeap::new(),
        distance: LatLonDistance::new(0.0, 0.0),
        total: 0,
        error: None,
    };
    top.collect(0, &[v(0.0, 3.0)]).unwrap();
    top.collect(1, &[]).unwrap();
    top.collect(2, &[v(0.0, 1.0), v(0.0, 50.0)]).unwrap();
    top.collect(3, &[v(0.0, 1.0)]).unwrap();
    top.collect(4, &[v(0.0, 2.0)]).unwrap();
    top.collect(5, &[]).unwrap();
    let r = top.finish();
    assert_eq!(r.total_hits, 6);
    assert_eq!(r.hits.iter().map(|h| h.0).collect::<Vec<_>>(), vec![2, 3]);
    assert!((r.hits[0].1 - 111_195.0).abs() < 100.0, "{:?}", r.hits);

    let mut xy = DistanceTopN {
        n: 1,
        queue: BinaryHeap::new(),
        distance: XYDistance::new(0.0, 0.0),
        total: 0,
        error: None,
    };
    let p = |x, y| XYDocValuesField::encode(x, y).unwrap();
    xy.collect(0, &[]).unwrap();
    xy.collect(1, &[p(3.0, 4.0)]).unwrap();
    xy.collect(2, &[p(30.0, 40.0), p(0.0, 1.0)]).unwrap();
    xy.collect(3, &[p(100.0, 100.0)]).unwrap();
    let r = xy.finish();
    assert_eq!(r.hits, vec![(2, 1.0)]);
    let empty = DistanceTopN {
        n: 0,
        queue: BinaryHeap::new(),
        distance: XYDistance::new(0.0, 0.0),
        total: 0,
        error: None,
    };
    assert!(empty.finish().hits.is_empty());
}

#[test]
fn geo3d_sorts_and_their_distances() {
    use crate::document::geo::geo3d::{from_distance, from_polygon};
    use lucene_util::geo::Polygon;
    let pm = PlanetModel::wgs84();
    let circle = from_distance(&pm, 0.0, 0.0, 100_000.0).unwrap();
    let mut s = Geo3DPointSortField::new("f", &pm, circle.clone());
    assert!(s.set_missing_value(f64::INFINITY).is_ok());
    assert!(s.set_missing_value(1.0).is_err());
    assert_eq!(format!("{s:?}"), "Geo3DPointSortField(\"f\")");
    let square = Polygon::new(
        &[0.0, 1.0, 1.0, 0.0, 0.0],
        &[0.0, 0.0, 1.0, 1.0, 0.0],
        vec![],
    )
    .unwrap();
    let polygon = from_polygon(&pm, &[square]).unwrap();
    let mut o = Geo3DPointOutsideSortField::new("f", &pm, polygon);
    assert!(o.set_missing_value(f64::INFINITY).is_ok());
    assert!(o.set_missing_value(f64::NEG_INFINITY).is_err());
    assert_eq!(format!("{o:?}"), "Geo3DPointOutsideSortField(\"f\")");

    let enc = pm.doc_value_encoder();
    let value = |lat: f64, lon: f64| {
        let p = lucene_spatial3d::GeoPoint::from_lat_lon(&pm, lat.to_radians(), lon.to_radians())
            .unwrap();
        enc.encode_point(&p).unwrap()
    };
    // Past the bottom's bounds (here the circle's own) a value is
    // skipped, not computed.
    let mut d = s.distance();
    d.set_bottom(1e-4).unwrap();
    assert!(d.bounds.is_some_and(|b| b[3] < 0.05), "{:?}", d.bounds);
    assert_eq!(d.compare_bottom(1e-4, &[value(0.0, 5.0)]), Ordering::Less);
    assert_eq!(
        d.compare_bottom(1e-4, &[value(0.0, 0.0001)]),
        Ordering::Greater
    );
    let key = d
        .sort_key_checked(&[value(0.0, 0.5), value(0.0, 0.0)])
        .unwrap();
    assert!(key < 1e-5, "{key}");
    let od = o.distance();
    assert_eq!(od.sort_key_checked(&[value(0.5, 0.5)]).unwrap(), 0.0);
    assert!(od.sort_key_checked(&[value(5.0, 5.0)]).unwrap() > 0.0);
    let e = s3d(lucene_spatial3d::Error::Runtime("boom".into()));
    assert!(e.to_string().contains("boom"), "{e}");
}

/// OpenSearch's comparator: every value against every origin, sorted,
/// then the mode's pick; missing documents at `+Infinity`.
#[test]
fn opensearch_geo_distance_modes() {
    let v = |lat, lon| LatLonDocValuesField::encode(lat, lon).unwrap();
    let point = |e: i64| {
        (
            GeoEncodingUtils::decode_latitude(doc_value_high(e)),
            GeoEncodingUtils::decode_longitude(doc_value_low(e)),
        )
    };
    let values = [v(0.0, 1.0), v(0.0, 3.0), v(10.0, 0.0)];
    let origins = vec![(0.0, 0.0), (0.0, 2.0)];
    let mut want: Vec<f64> = values
        .iter()
        .flat_map(|&e| {
            let (lat, lon) = point(e);
            origins
                .iter()
                .map(move |&(a, b)| sloppy_math::haversin_meters(a, b, lat, lon) / 1000.0)
        })
        .collect();
    want.sort_by(f64::total_cmp);
    let sort = |mode| OpenSearchGeoDistanceSort {
        origins: origins.clone(),
        unit_meters: 1000.0,
        mode,
        plane: false,
    };
    let mut scratch = Vec::new();
    let at = |mode| sort(mode).value(&values, &mut Vec::new());
    assert_eq!(at(DistanceMode::Min), want[0]);
    assert_eq!(at(DistanceMode::Max), want[5]);
    let sum = want.iter().fold(0.0, |t, v| t + v);
    assert_eq!(at(DistanceMode::Sum), sum);
    assert_eq!(at(DistanceMode::Avg), sum / 6.0);
    assert_eq!(at(DistanceMode::Median), (want[2] + want[3]) / 2.0);
    // Two values: the mean of both; three (below): the middle one.
    let odd = sort(DistanceMode::Median).value(&values[..1], &mut scratch);
    let (lat, lon) = point(values[0]);
    let mut two = [
        sloppy_math::haversin_meters(0.0, 0.0, lat, lon) / 1000.0,
        sloppy_math::haversin_meters(0.0, 2.0, lat, lon) / 1000.0,
    ];
    two.sort_by(f64::total_cmp);
    assert_eq!(odd, (two[0] + two[1]) / 2.0);
    let one = OpenSearchGeoDistanceSort {
        origins: vec![(0.0, 0.0)],
        unit_meters: 1.0,
        mode: DistanceMode::Median,
        plane: false,
    };
    assert_eq!(
        one.value(&values, &mut scratch),
        sloppy_math::haversin_meters(0.0, 0.0, point(values[1]).0, point(values[1]).1)
    );
    assert_eq!(one.value(&[], &mut scratch), f64::INFINITY);
}

/// The comparator over a real segment: values as sortable longs, a
/// document without the field at `+Infinity`, a field of another
/// doc-values type an error.
#[test]
fn opensearch_geo_distance_comparator_reads_the_segment() {
    use crate::directory_reader::DirectoryReader;
    use lucene_index::document::{self as d, Document};
    use lucene_index::index_writer::IndexWriter;
    use lucene_index::segment_info::LuceneVersion;
    use lucene_store::FsDirectory;
    use lucene_util::test_support::TempDir;
    let tmp = TempDir::new("os-geo-sort");
    let dir = FsDirectory::open(tmp.path());
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", version).unwrap();
    for lat in [Some(1.0), None, Some(-3.0)] {
        let mut doc = Document::new();
        if let Some(lat) = lat {
            doc.add_boxed(Box::new(
                d::LatLonDocValuesField::new("loc", lat, 0.0).unwrap(),
            ));
        }
        doc.add_boxed(Box::new(d::SortedDocValuesField::new("kw", "x")));
        w.add_fields_document(&doc).unwrap();
    }
    w.commit().unwrap();
    drop(w);
    let r = DirectoryReader::open(&dir).unwrap();
    let seg = &r.segment_readers()[0];
    let sort = OpenSearchGeoDistanceSort {
        origins: vec![(0.0, 0.0)],
        unit_meters: 1.0,
        mode: DistanceMode::Min,
        plane: false,
    };
    let src = sort.clone();
    let cmp = src.new_comparator("loc", 3, false);
    let mut leaf = cmp
        .leaf(LeafCtx {
            reader: seg,
            doc_base: 0,
        })
        .unwrap();
    let got: Vec<SortValue> = (0..3).map(|d| leaf.value(d, 0.0).unwrap()).collect();
    let SortValue::Long(missing) = got[1] else {
        panic!("{got:?}")
    };
    assert_eq!(missing, double_to_sortable_long(f64::INFINITY));
    assert_eq!(cmp.compare_values(&got[0], &got[2]), Ordering::Less);
    assert_eq!(
        cmp.compare_values(&got[0], &SortValue::Bytes(None)),
        Ordering::Equal
    );
    // No such field: every document missing.
    let mut none = src
        .new_comparator("nope", 3, false)
        .leaf(LeafCtx {
            reader: seg,
            doc_base: 0,
        })
        .unwrap();
    assert_eq!(none.value(0, 0.0).unwrap(), SortValue::Long(missing));
    // A field of another doc-values type.
    assert!(src
        .new_comparator("kw", 3, false)
        .leaf(LeafCtx {
            reader: seg,
            doc_base: 0,
        })
        .is_err());
}

/// In a segment of 10,000 documents a distance comparator reads its
/// points from the segment's decoded copy from the second use on: the
/// same values as the per-document read, documents without any
/// included; and `compareBottom` answers as `value` compares.
#[test]
fn decoded_points_column_reads_as_the_doc_values() {
    use crate::directory_reader::DirectoryReader;
    use lucene_index::document::{self as d, Document};
    use lucene_index::index_writer::IndexWriter;
    use lucene_index::segment_info::LuceneVersion;
    use lucene_store::FsDirectory;
    use lucene_util::test_support::TempDir;
    let tmp = TempDir::new("geo-sort-column");
    let dir = FsDirectory::open(tmp.path());
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", version).unwrap();
    for i in 0..10_050u32 {
        let mut doc = Document::new();
        for k in 0..(i % 3) {
            let lat = f64::from((i * 7 + k) % 170) - 85.0;
            let lon = f64::from((i * 13 + k) % 350) - 175.0;
            doc.add_boxed(Box::new(
                d::LatLonDocValuesField::new("loc", lat, lon).unwrap(),
            ));
        }
        w.add_fields_document(&doc).unwrap();
    }
    w.commit().unwrap();
    drop(w);
    let r = DirectoryReader::open(&dir).unwrap();
    let seg = &r.segment_readers()[0];
    assert!(seg.max_doc >= 10_000);
    let ctx = LeafCtx {
        reader: seg,
        doc_base: 0,
    };
    let lucene = LatLonPointSortField::new("loc", 10.0, 20.0)
        .unwrap()
        .comparator_source()
        .new_comparator("loc", 10, false);
    let os = OpenSearchGeoDistanceSort {
        origins: vec![(10.0, 20.0), (-5.0, 170.0)],
        unit_meters: 1609.344,
        mode: DistanceMode::Avg,
        plane: false,
    };
    let os = os.new_comparator("loc", 10, false);
    for cmp in [lucene, os] {
        let all = |cmp: &dyn FieldComparator| {
            let mut leaf = cmp.leaf(ctx).unwrap();
            (0..seg.max_doc)
                .map(|d| leaf.value(d, 0.0).unwrap())
                .collect::<Vec<_>>()
        };
        // First use: per document; second: the column is built; third:
        // read from it.
        let first = all(cmp.as_ref());
        assert_eq!(all(cmp.as_ref()), first);
        assert_eq!(all(cmp.as_ref()), first);
        assert!(matches!(
            GeoColumn::open(seg, "loc").unwrap(),
            GeoColumn::Decoded(_)
        ));
        // `compareBottom` against a middling bottom, as `value` compares.
        let bottom = first[5000].clone();
        let mut leaf = cmp.leaf(ctx).unwrap();
        leaf.set_bottom(&bottom).unwrap();
        for d in (0..seg.max_doc).step_by(7) {
            let want = cmp.compare_values(&bottom, &first[idx(d)]);
            let got = leaf.compare_bottom(&bottom, d, 0.0).unwrap().unwrap();
            // Outside the bottom's box Lucene's comparator answers
            // "not competitive" unmeasured, which `value` agrees with.
            assert_eq!(got, want, "doc {d}");
        }
        assert_eq!(
            leaf.compare_bottom(&SortValue::Bytes(None), 0, 0.0)
                .unwrap(),
            None
        );
    }
    // The other distance sorts compare by value.
    let xy = XYPointSortField::new("nope", 1.0, 2.0)
        .comparator_source()
        .new_comparator("nope", 10, false);
    let mut leaf = xy.leaf(ctx).unwrap();
    let bottom = SortValue::Long(0);
    leaf.set_bottom(&bottom).unwrap();
    assert_eq!(leaf.compare_bottom(&bottom, 0, 0.0).unwrap(), None);
}

#[test]
fn lat_lon_bottom_box_across_the_dateline() {
    let mut d = LatLonDistance::new(0.0, 179.9);
    let key = sloppy_math::haversin_sort_key(0.0, 179.9, 0.0, -179.0);
    d.set_bottom(key).unwrap();
    assert_eq!(d.min_lon, i32::MIN);
    assert_ne!(d.min_lon2, i32::MAX);
    let v = |lat, lon| LatLonDocValuesField::encode(lat, lon).unwrap();
    assert_eq!(d.compare_bottom(key, &[v(0.0, 0.0)]), Ordering::Less);
    assert_eq!(d.compare_bottom(key, &[v(0.0, 179.95)]), Ordering::Greater);
    for _ in 0..1100 {
        d.set_bottom(key).unwrap();
    }
    assert_eq!(d.set_bottom_counter, 1101);
}
