#![cfg(test)]
use super::*;

/// A visitor whose answers are fixed: `relate` says `rel`, `intersects`
/// says `hit`, `contains` says `within` (or fails).
struct Fixed {
    rel: Relation,
    hit: bool,
    within: Option<WithinRelation>,
}

impl SpatialVisitor for Fixed {
    fn relate(&self, _min: &[u8], _max: &[u8]) -> Relation {
        self.rel
    }
    fn intersects(&self, _packed: &[u8]) -> bool {
        self.hit
    }
    fn contains(&self, _packed: &[u8]) -> std::result::Result<WithinRelation, GeoError> {
        self.within
            .ok_or_else(|| GeoError::IllegalArgument("no withinPoint".into()))
    }
}

#[test]
fn spatial_walks_set_clear_and_exclude_as_java() {
    let v = Fixed {
        rel: Relation::CellCrossesQuery,
        hit: true,
        within: Some(WithinRelation::NotWithin),
    };
    let p = [0u8; 8];
    // CONTAINS: a cell inside excludes; NOTWITHIN excludes; CANDIDATE sets.
    let mut w = SpatialWalk::new(&v, QueryRelation::Contains, Walk::ContainsDense, 8);
    w.visit(1);
    w.visit_with_value(2, &p);
    assert!(get_doc(&w.excluded, 1) && get_doc(&w.excluded, 2));
    assert!(!get_doc(&w.result, 2));
    let cand = Fixed {
        within: Some(WithinRelation::Candidate),
        ..v
    };
    let mut w = SpatialWalk::new(&cand, QueryRelation::Contains, Walk::ContainsDense, 8);
    w.visit_with_value(3, &p);
    assert!(get_doc(&w.result, 3));
    assert_eq!(cand.leaf(QueryRelation::Contains, &p), Ok(true));
    // A failing `contains` is kept, and the walk goes on.
    let bad = Fixed { within: None, ..v };
    let mut w = SpatialWalk::new(&bad, QueryRelation::Contains, Walk::ContainsDense, 8);
    w.visit_with_value(4, &p);
    assert!(w.error.is_some());
    let mut w = SpatialWalk::new(&bad, QueryRelation::Contains, Walk::Forward, 8);
    w.visit_with_value(4, &p);
    assert!(w.error.is_some() && !get_doc(&w.result, 4));
    // The estimate walk only relates; any-hits stops at the first hit.
    let mut w = SpatialWalk::new(&v, QueryRelation::Within, Walk::Estimate, 8);
    w.visit(0);
    w.visit_with_value(0, &p);
    assert_eq!(w.compare(&p, &p), Relation::CellCrossesQuery);
    let inside = Fixed {
        rel: Relation::CellInsideQuery,
        ..v
    };
    let mut w = SpatialWalk::new(&inside, QueryRelation::Within, Walk::AnyHits, 0);
    assert_eq!(w.compare(&p, &p), Relation::CellOutsideQuery);
    assert!(w.found);
    assert_eq!(w.compare(&p, &p), Relation::CellOutsideQuery);
    let mut w = SpatialWalk::new(&v, QueryRelation::Within, Walk::AnyHits, 0);
    w.visit(5);
    assert!(w.found);
    // Dense: a failing point excludes its document for good.
    let miss = Fixed { hit: false, ..v };
    let mut w = SpatialWalk::new(&miss, QueryRelation::Within, Walk::Dense, 8);
    w.visit_with_value(6, &p);
    w.visit(7);
    assert!(get_doc(&w.excluded, 6) && get_doc(&w.result, 7));
    // The shallow inverse walk clears inside cells only.
    let mut w = SpatialWalk::new(&v, QueryRelation::Disjoint, Walk::ShallowInverse, 8);
    w.result.set_range(0, 8);
    w.visit_with_value(1, &p);
    w.visit(2);
    assert!(get_doc(&w.result, 1) && !get_doc(&w.result, 2));
    // Out-of-segment doc ids from a corrupt tree set nothing; the first
    // is remembered, and fails the walk.
    let mut w = SpatialWalk::new(&v, QueryRelation::Intersects, Walk::Forward, 8);
    w.visit(9);
    w.visit(-1);
    assert_eq!(w.result.cardinality(), 0);
    assert_eq!(w.bad, Some(9));
    let e = check_walk(w.bad, "f", 8).unwrap_err();
    assert!(
        e.to_string()
            .contains("points of field f name document 9, outside the segment's 0..8"),
        "{e}"
    );
    let mut w = SpatialWalk::new(&v, QueryRelation::Disjoint, Walk::Inverse, 8);
    w.visit_many(&[3, 99]);
    assert_eq!(w.bad, Some(99));
    let mut w = SpatialWalk::new(&v, QueryRelation::Within, Walk::ContainsDense, 8);
    w.visit_many(&[-5]);
    assert_eq!(w.bad, Some(-5));
    assert!(check_walk(None, "f", 8).is_ok());

    // Runs, ranges and bit-set leaves write what the same ids one by one
    // would.
    let mut w = SpatialWalk::new(&v, QueryRelation::Intersects, Walk::Forward, 200);
    w.visit_run(&[1, 2], &p);
    assert!(w.visit_range(10, 5));
    assert!(w.visit_bitset(&[0b101], 128, 2));
    let want: Vec<usize> = vec![1, 2, 10, 11, 12, 13, 14, 128, 130];
    let got: Vec<usize> = (0..200).filter(|&i| w.result.get(i)).collect();
    assert_eq!(got, want);
    assert_eq!(w.bad, None);
    // A bit-set leaf off a word boundary falls back to the id list.
    assert!(!w.visit_bitset(&[1], 3, 1));
    // The inverse walk clears; a run failing the leaf test (under
    // DISJOINT, one that intersects) is cleared, one passing it kept.
    let mut w = SpatialWalk::new(&v, QueryRelation::Disjoint, Walk::Inverse, 200);
    w.result.set_range(0, 200);
    w.visit_run(&[4], &p);
    assert!(w.visit_range(10, 5));
    assert!(w.visit_bitset(&[u64::MAX], 64, 64));
    assert_eq!(w.result.cardinality(), 200 - 1 - 5 - 64);
    assert!(!w.result.get(4) && !w.result.get(12) && !w.result.get(100));
    let mut w = SpatialWalk::new(&miss, QueryRelation::Disjoint, Walk::Inverse, 200);
    w.result.set_range(0, 200);
    w.visit_run(&[4], &p);
    assert!(w.result.get(4));
    // The dense walk sends a run to `excluded` on a miss.
    let mut w = SpatialWalk::new(&miss, QueryRelation::Within, Walk::Dense, 8);
    w.visit_run(&[5, 6], &p);
    assert!(get_doc(&w.excluded, 5) && get_doc(&w.excluded, 6));
    assert_eq!(w.result.cardinality(), 0);
    // Out-of-segment ranges and bits set nothing past the end and fail
    // the walk.
    let mut w = SpatialWalk::new(&v, QueryRelation::Intersects, Walk::Forward, 100);
    assert!(w.visit_range(98, 5));
    assert_eq!((w.result.cardinality(), w.bad), (0, Some(98)));
    let mut w = SpatialWalk::new(&v, QueryRelation::Intersects, Walk::Forward, 100);
    assert!(w.visit_bitset(&[u64::MAX], 64, 64));
    assert_eq!((w.result.cardinality(), w.bad), (36, Some(64)));
    let mut w = SpatialWalk::new(&v, QueryRelation::Within, Walk::AnyHits, 8);
    assert!(w.visit_range(0, 1));
    assert!(w.found);
}

#[test]
fn transpose_swaps_inside_and_outside() {
    assert_eq!(
        transpose(Relation::CellInsideQuery),
        Relation::CellOutsideQuery
    );
    assert_eq!(
        transpose(Relation::CellOutsideQuery),
        Relation::CellInsideQuery
    );
    assert_eq!(
        transpose(Relation::CellCrossesQuery),
        Relation::CellCrossesQuery
    );
}

#[test]
fn distance_query_validates_like_java() {
    let e = LatLonPointDistanceQuery::new("f", 0.0, 0.0, -1.0).unwrap_err();
    assert!(
        e.to_string().contains("radiusMeters: '-1.0' is invalid"),
        "{e}"
    );
    assert!(LatLonPointDistanceQuery::new("f", 0.0, 0.0, f64::NAN).is_err());
    assert!(LatLonPointDistanceQuery::new("f", 91.0, 0.0, 1.0).is_err());
    assert!(LatLonPointDistanceQuery::new("f", 0.0, 181.0, 1.0).is_err());
    let q = LatLonPointDistanceQuery::new("f", 10.0, 179.9, 50_000.0).unwrap();
    assert_eq!(q.weight.min_lon, i32::MIN, "the box crosses the dateline");
    let near = lucene_index::document::LatLonPoint::encode(10.0, -179.95).unwrap();
    let far = lucene_index::document::LatLonPoint::encode(10.0, 170.0).unwrap();
    assert!(q.matches(&near));
    assert!(!q.matches(&far));
    let lo = lucene_index::document::LatLonPoint::encode(-50.0, -50.0).unwrap();
    let hi = lucene_index::document::LatLonPoint::encode(-40.0, -40.0).unwrap();
    assert_eq!(q.relate(&lo, &hi), Relation::CellOutsideQuery);
}

#[test]
fn point_query_rejects_unsupported_relations() {
    use lucene_util::geo::{Line, Point};
    let line = LatLonGeometry::Line(Line::new(&[0.0, 1.0], &[0.0, 1.0]).unwrap());
    let e =
        LatLonPointQuery::new("f", QueryRelation::Within, std::slice::from_ref(&line)).unwrap_err();
    assert!(e.to_string().contains("WITHIN queries with line"), "{e}");
    let e = LatLonPointQuery::new("f", QueryRelation::Contains, &[line]).unwrap_err();
    assert!(
        e.to_string().contains("CONTAINS queries with non-points"),
        "{e}"
    );
    assert!(LatLonPointQuery::new("f", QueryRelation::Intersects, &[]).is_err());
    let at = lucene_index::document::LatLonPoint::encode(1.0, 2.0).unwrap();
    let p = LatLonGeometry::Point(
        Point::new(
            GeoEncodingUtils::decode_latitude_bytes(&at, 0),
            GeoEncodingUtils::decode_longitude_bytes(&at, 4),
        )
        .unwrap(),
    );
    let q = LatLonPointQuery::new("f", QueryRelation::Contains, &[p]).unwrap();
    assert_eq!(q.visitor.contains(&at).unwrap(), WithinRelation::Candidate);
    assert!(format!("{q:?}").contains("LatLonPointVisitor"));
    assert!(XYPointInGeometryQuery::new("f", &[]).is_err());
}

#[test]
fn check_compatible_messages() {
    let e = check_shape("f", 1, 4, "LatLonPoint").unwrap_err();
    assert!(e.to_string().contains("numDims=1"), "{e}");
    let e = check_shape("f", 2, 8, "XYPoint").unwrap_err();
    assert!(
        e.to_string().contains("bytesPerDim=8") && e.to_string().contains("XYPoint"),
        "{e}"
    );
    assert!(check_shape("f", 2, 4, "XYPoint").is_ok());
    assert!(check_shape("f", 0, 0, "XYPoint").is_ok());
}
