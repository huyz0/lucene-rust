//! Unit tests for the spatial strategies' own surface: construction and
//! indexing errors, the value sources' descriptions, explanations and
//! defaults, the heatmap's arithmetic edges, buffering, and the query paths
//! the differential fixture (`tests/spatial_strategies_fixtures.rs`) does
//! not reach.

use std::sync::Arc;

use lucene_index::document::{Document, FieldType, IndexableField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_store::FsDirectory;
use lucene_util::spatial4j::{Point, Shape, SpatialContext, SpatialContextFactory};
use lucene_util::spatial_extras::prefix_tree::{
    Calendar, DateRangePrefixTree, QuadPrefixTree, SpatialPrefixTree,
};
use lucene_util::spatial_extras::query::{SpatialArgs, SpatialOperation};
use lucene_util::test_support::TempDir;

use super::bbox::{BBoxOverlapRatioValueSource, BBoxStrategy};
use super::prefix::facets::{self, Heatmap};
use super::prefix::query::{
    buffer_shape, java_max, java_min, ContainsPrefixTreeQuery, WithinPrefixTreeQuery,
};
use super::util::{
    CachingDoubleValueSource, ReciprocalDoubleValuesSource, ShapeAreaValueSource, ShapeFieldCache,
    ShapeValuesPredicate,
};
use super::*;
use crate::directory_reader::DirectoryReader;
use crate::document::search_all;
use crate::explain::Explanation;
use crate::index_searcher::IndexSearcher;
use crate::multi_segment::OpenSegment;
use crate::values_source::{constant, ValuesContext};

const VERSION: LuceneVersion = LuceneVersion {
    major: 10,
    minor: 5,
    bugfix: 0,
};

fn geo() -> Arc<SpatialContext> {
    SpatialContext::geo_context()
}

fn flat() -> Arc<SpatialContext> {
    let mut f = SpatialContextFactory::new();
    f.geo = false;
    f.world_bounds = Some([-100.0, 100.0, -100.0, 100.0]);
    f.new_spatial_context().unwrap()
}

fn wkt(ctx: &Arc<SpatialContext>, s: &str) -> Arc<dyn Shape> {
    ctx.read_shape_from_wkt(s).unwrap()
}

fn pt(ctx: &Arc<SpatialContext>, x: f64, y: f64) -> Arc<dyn Point> {
    ctx.point_xy(x, y).unwrap()
}

/// Writes one segment of documents (each the fields of its shapes) and
/// opens it.
fn index(tmp: &TempDir, docs: Vec<Vec<Box<dyn IndexableField>>>) -> DirectoryReader {
    let dir = FsDirectory::open(tmp.path());
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
    for fields in docs {
        let mut doc = Document::new();
        for f in fields {
            doc.add_boxed(f);
        }
        w.add_fields_document(&doc).unwrap();
    }
    w.commit().unwrap();
    drop(w);
    DirectoryReader::open(&dir).unwrap()
}

fn hits(leaves: &[OpenSegment<'_>], q: &dyn crate::document::DocumentQuery) -> Vec<i32> {
    let mut h: Vec<i32> = search_all(leaves, q)
        .unwrap()
        .into_iter()
        .map(|h| h.doc_id)
        .collect();
    h.sort_unstable();
    h
}

fn class(e: &crate::Error) -> &'static str {
    match e {
        crate::Error::Spatial(s) => s.java_class(),
        crate::Error::IllegalArgument(_) => "java.lang.IllegalArgumentException",
        crate::Error::IllegalState(_) => "java.lang.IllegalStateException",
        _ => "other",
    }
}

fn quad(ctx: &Arc<SpatialContext>, levels: i32) -> Arc<dyn SpatialPrefixTree> {
    Arc::new(QuadPrefixTree::new(ctx.clone(), levels).unwrap())
}

#[test]
fn construction_and_indexing_errors() {
    let ctx = geo();
    assert!(check_field_name("").is_err());
    assert!(RecursivePrefixTreeStrategy::new(quad(&ctx, 4), "").is_err());
    assert!(BBoxStrategy::new_instance(ctx.clone(), "").is_err());
    assert!(PointVectorStrategy::new_instance(ctx.clone(), "").is_err());
    assert!(SerializedDVStrategy::new(ctx.clone(), "").is_err());

    // points only: a box is refused (when not pruning: Java's pruning
    // traversal does not check)
    let mut rpt = RecursivePrefixTreeStrategy::new(quad(&ctx, 4), "f").unwrap();
    rpt.base_mut().set_points_only(true);
    assert!(rpt
        .create_indexable_fields(&wkt(&ctx, "ENVELOPE(0, 10, 10, 0)"))
        .is_ok());
    rpt.set_prune_leafy_branches(false);
    assert!(rpt.base().is_points_only());
    let e = rpt
        .create_indexable_fields(&wkt(&ctx, "ENVELOPE(0, 10, 10, 0)"))
        .err()
        .unwrap();
    assert!(
        e.to_string().contains("pointsOnly is true yet a class"),
        "{e}"
    );
    // a point vector indexes points only
    let pv = PointVectorStrategy::new_instance(ctx.clone(), "pv").unwrap();
    let e = pv
        .create_indexable_fields(&wkt(&ctx, "ENVELOPE(0, 10, 10, 0)"))
        .err()
        .unwrap();
    assert_eq!(class(&e), "java.lang.UnsupportedOperationException");
    // and queries boxes and circles only
    let args = SpatialArgs::new(SpatialOperation::Intersects, wkt(&ctx, "POINT(1 2)"));
    let e = pv.make_query(&args).err().unwrap();
    assert!(e.to_string().contains("found [class "), "{e}");
    // a BBox queries by rectangle only, and needs points to query at all
    let bb = BBoxStrategy::new_instance(ctx.clone(), "bb").unwrap();
    assert!(bb.make_query(&args).is_err());
    let mut dv_only = FieldType::new();
    dv_only
        .set_doc_values_type(lucene_index::document::DocValuesType::Numeric)
        .unwrap();
    let no_points = BBoxStrategy::new(ctx.clone(), "nb", dv_only.clone()).unwrap();
    assert_eq!(
        no_points.field_type().doc_values_type(),
        dv_only.doc_values_type()
    );
    let rect_args = SpatialArgs::new(
        SpatialOperation::Intersects,
        wkt(&ctx, "ENVELOPE(0, 1, 1, 0)"),
    );
    for op in [SpatialOperation::Intersects, SpatialOperation::IsEqualTo] {
        let mut a = rect_args.clone();
        a.operation = op;
        let e = no_points.make_query(&a).err().unwrap();
        assert!(e.to_string().contains("An index is required"), "{e}");
    }
    let pv_dv = PointVectorStrategy::new(ctx.clone(), "pvd", dv_only).unwrap();
    assert!(pv_dv.make_query(&rect_args).is_err());
    // the overlap ratio's proportion
    assert!(bb
        .make_overlap_ratio_value_source(ctx.rect(0.0, 1.0, 0.0, 1.0).unwrap(), 1.5)
        .is_err());
    // composite: no distances, no BBox operations, no disjoint
    let cmp = CompositeSpatialStrategy::new(
        "cmp",
        RecursivePrefixTreeStrategy::new(quad(&ctx, 4), "c_rpt").unwrap(),
        SerializedDVStrategy::new(ctx.clone(), "c_sdv").unwrap(),
    )
    .unwrap();
    assert!(cmp.is_optimize_predicates());
    assert!(cmp
        .make_distance_value_source(&pt(&ctx, 0.0, 0.0), 1.0)
        .is_err());
    for op in [
        SpatialOperation::BBoxIntersects,
        SpatialOperation::BBoxWithin,
        SpatialOperation::IsDisjointTo,
    ] {
        let mut a = rect_args.clone();
        a.operation = op;
        assert!(cmp.make_query(&a).is_err());
    }
    assert!(CompositeSpatialStrategy::new(
        "",
        cmp.index_strategy().clone(),
        cmp.geometry_strategy().clone()
    )
    .is_err());
}

#[test]
fn descriptions() {
    let ctx = geo();
    let rpt = RecursivePrefixTreeStrategy::new(quad(&ctx, 4), "f").unwrap();
    assert_eq!(rpt.field_name(), "f");
    assert!(Arc::ptr_eq(rpt.spatial_context(), &ctx));
    assert!(rpt.is_prune_leafy_branches() && rpt.is_multi_overlapping_indexed_shapes());
    assert!(!format!("{:?}", rpt.base()).is_empty());
    assert!(Arc::ptr_eq(rpt.base().spatial_context(), &ctx));
    assert!((rpt.base().dist_err_pct() - 0.025).abs() < 1e-12);
    let tq = TermQueryPrefixTreeStrategy::new(quad(&ctx, 4), "t").unwrap();
    assert_eq!(tq.field_name(), "t");
    assert!(Arc::ptr_eq(tq.spatial_context(), &ctx));
    assert!(tq.as_prefix_tree().is_some());
    assert_eq!(tq.base().field_name(), "t");
    let bb = BBoxStrategy::new_instance(ctx.clone(), "bb").unwrap();
    assert_eq!(bb.field_name(), "bb");
    assert!(bb.as_prefix_tree().is_none());
    assert!(format!("{bb:?}").starts_with("BBoxStrategy field:bb"));
    assert_eq!(bb.make_shape_value_source().to_string(), "bboxShape(bb)");
    let pv = PointVectorStrategy::new_instance(ctx.clone(), "pv").unwrap();
    assert_eq!(pv.field_name(), "pv");
    assert!(format!("{pv:?}").starts_with("PointVectorStrategy"));
    let sdv = SerializedDVStrategy::new(ctx.clone(), "s").unwrap();
    assert_eq!(sdv.field_name(), "s");
    assert!(format!("{sdv:?}").starts_with("SerializedDVStrategy"));
    assert_eq!(sdv.make_shape_value_source().to_string(), "shapeDocVal(s)");
    let cmp = CompositeSpatialStrategy::new("cmp", rpt.clone(), sdv.clone()).unwrap();
    assert_eq!(cmp.field_name(), "cmp");
    assert!(format!("{cmp:?}").starts_with("CompositeSpatialStrategy"));
    let dates = Arc::new(DateRangePrefixTree::new(&Calendar::new_default()).unwrap());
    let mut dr = NumberRangePrefixTreeStrategy::for_dates(dates, "dr").unwrap();
    assert_eq!(dr.field_name(), "dr");
    assert_eq!(dr.rpt().prefix_grid_scan_level(), 7);
    dr.rpt_mut().set_prefix_grid_scan_level(6);
    assert!(dr.to_string().contains("prefixGridScanLevel:6"));
    assert_eq!(dr.number_range_tree().to_string(), "DateRangePrefixTree");
    assert!(dr
        .make_distance_value_source(&pt(&ctx, 0.0, 0.0), 1.0)
        .is_err());

    let p = pt(&ctx, 1.0, 2.0);
    let dist = make_distance_value_source(&bb, &p).unwrap();
    assert!(dist
        .describe()
        .starts_with("distance(Pt(x=1.0,y=2.0) to bboxShape(bb))*1.0"));
    assert!(!dist.needs_scores());
    let area = ShapeAreaValueSource::new(bb.make_shape_value_source(), ctx.clone(), true, 1.0);
    assert_eq!(area.describe(), "area(bboxShape(bb),geo=true)");
    assert!(!area.needs_scores());
    let recip = ReciprocalDoubleValuesSource::new(2.0, constant(3.0));
    assert!(recip.describe().starts_with("recip(2.0, "));
    assert!(!recip.needs_scores());
    let cached = CachingDoubleValueSource::new(constant(1.0));
    assert!(cached.describe().starts_with("Cached["));
    assert!(!cached.needs_scores());
    let field_cache = rpt.base().make_distance_value_source(&p, 1.0);
    assert!(field_cache
        .describe()
        .starts_with("ShapeFieldCacheDistanceValueSource(PointPrefixTreeFieldCacheProvider(f)"));
    assert!(!field_cache.needs_scores());
    let pv_dist = pv.make_distance_value_source(&p, 1.0).unwrap();
    assert!(pv_dist
        .describe()
        .starts_with("DistanceValueSource(PointVectorStrategy"));
    assert!(!pv_dist.needs_scores());
    let overlap = BBoxOverlapRatioValueSource::with_defaults(
        bb.make_shape_value_source(),
        ctx.rect(0.0, 10.0, 0.0, 10.0).unwrap(),
    )
    .unwrap();
    assert_eq!(
        overlap.describe(),
        "BBoxOverlapRatioValueSource(bboxShape(bb),Rect(minX=0.0,maxX=10.0,minY=0.0,maxY=10.0),0.25)"
    );
    assert!(!overlap.needs_scores());
    let pred = ShapeValuesPredicate::new(
        sdv.make_shape_value_source(),
        SpatialOperation::Intersects,
        p.clone(),
    );
    assert_eq!(
        pred.to_string(),
        "shapeDocVal(s) Intersects Pt(x=1.0,y=2.0)"
    );
    assert_eq!(format!("{pred:?}"), pred.to_string());
}

#[test]
fn java_min_and_max() {
    assert!(java_max(f64::NAN, 1.0).is_nan());
    assert!(java_min(1.0, f64::NAN).is_nan());
    assert!(java_max(-0.0, 0.0).is_sign_positive());
    assert!(java_max(0.0, -0.0).is_sign_positive());
    assert!(java_min(-0.0, 0.0).is_sign_negative());
    assert!(java_min(0.0, -0.0).is_sign_negative());
    assert_eq!(java_max(1.0, 2.0), 2.0);
    assert_eq!(java_min(1.0, 2.0), 1.0);
}

#[test]
fn buffering_query_shapes() {
    let g = geo();
    assert!(buffer_shape(&g, &wkt(&g, "POINT(1 2)"), 0.0).is_err());
    let c = buffer_shape(&g, &wkt(&g, "POINT(1 2)"), 1.0).unwrap();
    assert!(c.as_circle().is_some());
    let c = buffer_shape(&g, &wkt(&g, "BUFFER(POINT(1 2), 170)"), 20.0).unwrap();
    assert_eq!(c.as_circle().unwrap().radius(), 180.0);
    let r = buffer_shape(&g, &wkt(&g, "ENVELOPE(170, 175, 10, 0)"), 10.0).unwrap();
    let r = r.as_rectangle().unwrap();
    assert_eq!((r.min_x(), r.max_x()), (160.0, -175.0));
    let r = buffer_shape(&g, &wkt(&g, "ENVELOPE(0, 5, 85, 80)"), 10.0).unwrap();
    let r = r.as_rectangle().unwrap();
    assert_eq!((r.min_x(), r.max_x(), r.max_y()), (-180.0, 180.0, 90.0));
    let r = buffer_shape(&g, &wkt(&g, "ENVELOPE(0, 5, -80, -85)"), 10.0).unwrap();
    assert_eq!(r.as_rectangle().unwrap().min_y(), -90.0);
    let f = flat();
    let r = buffer_shape(&f, &wkt(&f, "ENVELOPE(90, 95, 95, 90)"), 10.0).unwrap();
    let r = r.as_rectangle().unwrap();
    assert_eq!((r.max_x(), r.max_y()), (100.0, 100.0));
}

#[test]
fn heatmap_arithmetic() {
    let g = geo();
    let hm = Heatmap {
        columns: 2,
        rows: 3,
        counts: vec![1, 2, 3, 4, 5, 6],
        region: g.rect(0.0, 10.0, 0.0, 10.0).unwrap(),
    };
    assert_eq!(hm.get_count(1, 2), 6);
    assert_eq!(hm.get_count(5, 5), 0);
    assert_eq!(
        hm.to_string(),
        "Heatmap{2x3 Rect(minX=0.0,maxX=10.0,minY=0.0,maxY=10.0)}"
    );
    let rpt = RecursivePrefixTreeStrategy::new(quad(&g, 4), "f").unwrap();
    let e = facets::calc_heatmap(rpt.base(), &[], None, None, 2, i32::MAX).unwrap_err();
    assert_eq!(class(&e), "java.lang.IllegalArgumentException");
    // a point: one cell
    let p = wkt(&g, "POINT(10 20)");
    let hm = rpt.base().calc_facets(&[], None, Some(&p), 3, 100).unwrap();
    assert_eq!((hm.columns, hm.rows), (1, 1));
}

#[test]
fn queries_over_a_small_index() {
    let tmp = TempDir::new("spatial-unit");
    let g = geo();
    let mut pts = RecursivePrefixTreeStrategy::new(quad(&g, 6), "pts").unwrap();
    pts.base_mut().set_points_only(true);
    pts.base_mut().set_default_field_values_array_len(1);
    let rpt = RecursivePrefixTreeStrategy::new(quad(&g, 6), "rpt").unwrap();
    let sdv = SerializedDVStrategy::new(g.clone(), "sdv").unwrap();
    let bb = BBoxStrategy::new_instance(g.clone(), "bb").unwrap();
    let pv = PointVectorStrategy::new_instance(g.clone(), "pv").unwrap();
    let shapes = ["POINT(10 20)", "POINT(10 20)", "POINT(-30 -40)"];
    let docs: Vec<Vec<Box<dyn IndexableField>>> = shapes
        .iter()
        .map(|s| {
            let shape = wkt(&g, s);
            let mut f = pts.create_indexable_fields(&shape).unwrap();
            f.extend(rpt.create_indexable_fields(&shape).unwrap());
            f.extend(sdv.create_indexable_fields(&shape).unwrap());
            f.extend(bb.create_indexable_fields(&shape).unwrap());
            f.extend(pv.create_indexable_fields(&shape).unwrap());
            f
        })
        .collect();
    let reader = index(&tmp, docs);
    let mut opened = reader.open_segments().unwrap();
    opened.open_points().unwrap();
    let leaves = opened.as_open_segments();

    // the composite's general path over a points-only (term) index query
    let mut cmp = CompositeSpatialStrategy::new("cmp", pts.clone(), sdv.clone()).unwrap();
    cmp.set_optimize_predicates(false);
    let point = wkt(&g, "POINT(10 20)");
    let q = cmp
        .make_query(&SpatialArgs::new(
            SpatialOperation::Intersects,
            point.clone(),
        ))
        .unwrap();
    assert_eq!(hits(&leaves, q.as_ref()), vec![0, 1]);
    // within, buffered
    let within = WithinPrefixTreeQuery::new(point.clone(), "rpt", quad(&g, 6), 6, 4, 1.0).unwrap();
    assert_eq!(hits(&leaves, &within), vec![0, 1]);
    // contains, both documents' shapes at one cell
    let contains = ContainsPrefixTreeQuery::new(point.clone(), "rpt", quad(&g, 6), 6, true);
    assert_eq!(hits(&leaves, &contains), vec![0, 1]);
    // a field that is not there
    let missing = ContainsPrefixTreeQuery::new(point.clone(), "nope", quad(&g, 6), 6, true);
    assert!(hits(&leaves, &missing).is_empty());

    // value sources: explanations, cacheability, defaults
    let norms = vec![None; leaves.len()];
    let searcher = IndexSearcher::new(&leaves, &norms).unwrap();
    let ctx = ValuesContext::new(&searcher);
    let p = pt(&g, 10.0, 20.0);
    let overlap = BBoxOverlapRatioValueSource::new(
        bb.make_shape_value_source(),
        true,
        g.rect(0.0, 20.0, 10.0, 30.0).unwrap(),
        0.5,
        0.5,
    )
    .unwrap();
    assert!(overlap.is_cacheable(&ctx, 0));
    let e = overlap
        .explain(&ctx, 0, 0, &Explanation::match_(1.0, "s"))
        .unwrap();
    assert!(e.value > 0.0);
    let recip =
        ReciprocalDoubleValuesSource::new(1.0, make_distance_value_source(&pv, &p).unwrap());
    assert!(recip.is_cacheable(&ctx, 0));
    let e = recip
        .explain(&ctx, 0, 0, &Explanation::match_(1.0, "s"))
        .unwrap();
    assert_eq!(e.value, 1.0);
    let cached = CachingDoubleValueSource::new(make_distance_value_source(&sdv, &p).unwrap());
    assert!(cached.is_cacheable(&ctx, 0));
    let mut v = cached.get_values(&ctx, 0, None).unwrap();
    assert!(v.advance_exact(2).unwrap());
    let first = v.double_value().unwrap();
    assert_eq!(v.double_value().unwrap(), first, "from the cache");
    assert!(cached
        .explain(&ctx, 0, 0, &Explanation::match_(1.0, "s"))
        .is_ok());
    let area = ShapeAreaValueSource::new(sdv.make_shape_value_source(), g.clone(), false, 1.0);
    assert!(area.is_cacheable(&ctx, 0));
    let field_cache = pts.base().make_distance_value_source(&p, 1.0);
    assert!(field_cache.is_cacheable(&ctx, 0));
    // a field without terms: every document has the default
    let empty = RecursivePrefixTreeStrategy::new(quad(&g, 6), "nope")
        .unwrap()
        .base()
        .make_distance_value_source(&p, 2.0);
    let mut v = empty.get_values(&ctx, 0, None).unwrap();
    assert!(v.advance_exact(0).unwrap());
    assert_eq!(v.double_value().unwrap(), 360.0);
    let pv_dist = pv.make_distance_value_source(&p, 1.0).unwrap();
    assert!(pv_dist.is_cacheable(&ctx, 0));
    let dts = sdv.make_distance_value_source(&p, 1.0).unwrap();
    assert!(dts.is_cacheable(&ctx, 0));
    let pred = ShapeValuesPredicate::new(
        sdv.make_shape_value_source(),
        SpatialOperation::Intersects,
        p.clone(),
    );
    assert!(pred.is_cacheable(&ctx, 0));
    // a document without the field: no shape, the defaults
    let none = SerializedDVStrategy::new(g.clone(), "none").unwrap();
    let mut shapes = none.make_shape_value_source().get_values(&ctx, 0).unwrap();
    assert!(!shapes.advance_exact(0).unwrap());
    let nb = BBoxStrategy::new_instance(g.clone(), "none").unwrap();
    let mut boxes = nb.make_shape_value_source().get_values(&ctx, 0).unwrap();
    assert!(!boxes.advance_exact(0).unwrap());
    let npv = PointVectorStrategy::new_instance(g.clone(), "none").unwrap();
    let mut d = npv
        .make_distance_value_source(&p, 1.0)
        .unwrap()
        .get_values(&ctx, 0, None)
        .unwrap();
    assert!(d.advance_exact(0).unwrap());
    assert_eq!(d.double_value().unwrap(), 180.0);
}

#[test]
fn shape_field_cache() {
    let mut c: ShapeFieldCache<i32> = ShapeFieldCache::new(3, 2);
    c.add(1, 7);
    c.add(1, 8);
    c.add(9, 1); // outside: ignored
    c.add(-1, 1);
    assert_eq!(c.get_shapes(1), Some(&[7, 8][..]));
    assert_eq!(c.get_shapes(0), None);
    assert_eq!(c.get_shapes(-1), None);
}

#[test]
fn shape_classes() {
    let g = geo();
    let f = flat();
    for (shape, class) in [
        (wkt(&g, "POINT(1 2)"), "PointImpl"),
        (wkt(&g, "ENVELOPE(0, 1, 1, 0)"), "RectangleImpl"),
        (wkt(&g, "BUFFER(POINT(1 2), 1)"), "GeoCircle"),
        (wkt(&f, "BUFFER(POINT(1 2), 1)"), "CircleImpl"),
        (
            wkt(&g, "GEOMETRYCOLLECTION(POINT(1 2), POINT(3 4))"),
            "ShapeCollection",
        ),
        (wkt(&g, "LINESTRING(1 2, 3 4)"), "BufferedLineString"),
    ] {
        assert!(
            shape_class(&*shape).ends_with(class),
            "{}",
            shape_class(&*shape)
        );
    }
    let dates = DateRangePrefixTree::new(&Calendar::new_default()).unwrap();
    let unit = dates.parse_shape("2014").unwrap().into_shape();
    assert!(shape_class(&*unit).ends_with("$NRCell"));
    let span = dates.parse_shape("[2014 TO 2015]").unwrap().into_shape();
    assert!(shape_class(&*span).ends_with("$SpanUnitsNRShape"));
    assert_eq!(
        class(&class_cast(&*span, "X")),
        "java.lang.ClassCastException"
    );
}

#[test]
fn heatmap_helpers() {
    use facets::{calc_rows_or_cols, increment_range, intersect_interval};
    // a zero-size request is one interval; an absurd one saturates
    assert_eq!(calc_rows_or_cols(1.0, 0.0, 0.0, 0.0, 360.0), 1);
    assert_eq!(calc_rows_or_cols(1e-300, 0.0, 1.0, 0.0, 1.0), i32::MAX);
    assert_eq!(
        calc_rows_or_cols(1e-12, 0.0, 1e-3, 0.0, 1e300),
        1_000_000_000
    );
    assert_eq!(intersect_interval(0.0, 10.0, 1.0, 10, -5.0, 20.0), (0, 9));
    assert_eq!(intersect_interval(0.0, 10.0, 1.0, 10, 2.0, 5.0), (2, 4));
    let mut hm = Heatmap {
        columns: 3,
        rows: 3,
        counts: vec![0; 9],
        region: geo().rect(0.0, 3.0, 0.0, 3.0).unwrap(),
    };
    // negative starts are clamped, the ends moved with them (Java's)
    increment_range(&mut hm, -1, 1, -1, 1, 5);
    assert_eq!(hm.counts, vec![5, 0, 0, 0, 0, 0, 0, 0, 0]);
    // an empty row range does nothing
    increment_range(&mut hm, 0, 2, 2, 1, 7);
    assert_eq!(hm.counts.iter().sum::<i32>(), 5);
}

#[test]
fn facets_over_a_small_index() {
    let tmp = TempDir::new("spatial-facets");
    let g = geo();
    let rpt = RecursivePrefixTreeStrategy::new(quad(&g, 5), "rpt").unwrap();
    let mut pv_type = super::bbox::default_field_type();
    pv_type = {
        let mut t = FieldType::copy_of(&pv_type);
        t.set_stored(true).unwrap();
        t
    };
    let pv = PointVectorStrategy::new(g.clone(), "pv", pv_type).unwrap();
    let shapes = [
        "POINT(10 20)",
        "POINT(11 21)",
        "ENVELOPE(-170, 170, 80, -80)",
    ];
    let docs: Vec<Vec<Box<dyn IndexableField>>> = shapes
        .iter()
        .map(|s| {
            let shape = wkt(&g, s);
            let mut f = rpt.create_indexable_fields(&shape).unwrap();
            if shape.as_point().is_some() {
                f.extend(pv.create_indexable_fields(&shape).unwrap());
            }
            f
        })
        .collect();
    let reader = index(&tmp, docs);
    let mut opened = reader.open_segments().unwrap();
    opened.open_points().unwrap();
    let leaves = opened.as_open_segments();
    // no deletions: counts are document frequencies
    let hm = rpt
        .base()
        .calc_facets(&leaves, None, None, 2, 1000)
        .unwrap();
    assert_eq!(hm.counts.len(), (hm.columns * hm.rows) as usize);
    assert!(
        hm.counts.iter().all(|&c| c >= 1),
        "the box covers every cell"
    );
    // level 0: no corner cell (Java's NullPointerException)
    assert!(rpt
        .base()
        .calc_facets(&leaves, None, None, 0, 1000)
        .is_err());
    // a non-number-range tree's cells are not units
    let world: Arc<dyn Shape> = Arc::new(g.world_bounds());
    let e = facets::calc_number_range_facets(rpt.base(), &leaves, None, &world, 2).unwrap_err();
    assert_eq!(class(&e), "java.lang.ClassCastException");
    // a circle query of the stored point vector
    let circle = wkt(&g, "BUFFER(POINT(10 20), 2)");
    let q = pv
        .make_query(&SpatialArgs::new(SpatialOperation::IsWithin, circle))
        .unwrap();
    assert_eq!(hits(&leaves, q.as_ref()), vec![0, 1]);
}
