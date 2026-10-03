//! The spatial-extras benchmark pair (M9 T9.5), against
//! `benchmarks/micro/java/SpatialExtrasMicro.java`: RPT indexing a Geo3D
//! polygon (every token consumed), RPT intersects of boxes and circles, BBox
//! intersects with the overlap-ratio similarity of every hit, heatmaps and
//! date-range queries, over the 100 000-document index the Java side builds
//! (`SpatialExtrasMicro build <dir>`), replaying its `spatial-queries.tsv`.
//! Each case prints a `#check` digest the report compares before it shows a
//! ratio.

use std::collections::BTreeMap;
use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;

use lucene_search::collector::{ScoreMode, ScoringCollector};
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::document::DocumentQuery;
use lucene_search::index_searcher::IndexSearcher;
use lucene_search::multi_segment::OpenSegment;
use lucene_search::spatial::bbox::BBoxStrategy;
use lucene_search::spatial::{
    NumberRangePrefixTreeStrategy, RecursivePrefixTreeStrategy, SpatialStrategy,
};
use lucene_search::values_source::{DoubleValuesSource, ValuesContext};
use lucene_store::FsDirectory;
use lucene_util::spatial4j::{Shape, SpatialContext, SpatialContextFactory};
use lucene_util::spatial_extras::prefix_tree::{Calendar, DateRangePrefixTree, QuadPrefixTree};
use lucene_util::spatial_extras::query::{SpatialArgs, SpatialOperation};

use super::measure;

/// FNV-1a over 64-bit words, identical to `SpatialExtrasMicro.Fnv`.
struct Fnv(u64);

impl Fnv {
    fn new() -> Self {
        Fnv(0xcbf2_9ce4_8422_2325)
    }
    fn add(&mut self, x: i64) {
        self.0 ^= x as u64;
        self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

fn check(case: &str, d: &Fnv, n: u64) {
    println!("#check\t{case}\t{:016x}\t{n}", d.0);
}

/// Counts every hit, needs no score.
struct Count(i64);

impl ScoringCollector for Count {
    fn collect(&mut self, _doc_id: i32, _score: f32) {
        self.0 += 1;
    }
    fn score_mode(&self) -> ScoreMode {
        ScoreMode::CompleteNoScores
    }
}

fn count(leaves: &[OpenSegment<'_>], q: &dyn DocumentQuery) -> i64 {
    let rewritten = lucene_search::document::rewrite(q, leaves).unwrap();
    let q = rewritten.as_deref().unwrap_or(q);
    let mut c = Count(0);
    for leaf in leaves {
        q.score_leaf(leaf, 1.0, &mut c).unwrap();
    }
    c.0
}

/// `SpatialExtrasMicro.Sum`: a value source summed over a query's hits.
struct Sum<'v> {
    values: Box<dyn lucene_search::values_source::DoubleValues + 'v>,
    sum: f64,
    n: i64,
}

impl ScoringCollector for Sum<'_> {
    fn collect(&mut self, doc_id: i32, _score: f32) {
        if self.values.advance_exact(doc_id).unwrap() {
            self.sum += self.values.double_value().unwrap();
        }
        self.n += 1;
    }
    fn score_mode(&self) -> ScoreMode {
        ScoreMode::CompleteNoScores
    }
}

fn similarity(
    leaves: &[OpenSegment<'_>],
    ctx: &ValuesContext<'_>,
    q: &dyn DocumentQuery,
    src: &Arc<dyn DoubleValuesSource>,
) -> (i64, f64) {
    let (mut n, mut sum) = (0, 0.0);
    for (i, leaf) in leaves.iter().enumerate() {
        let mut s = Sum {
            values: src.get_values(ctx, i, None).unwrap(),
            sum: 0.0,
            n: 0,
        };
        q.score_leaf(leaf, 1.0, &mut s).unwrap();
        n += s.n;
        sum += s.sum;
    }
    (n, sum)
}

/// `SpatialExtrasMicro.tokens`: every token of a polygon's RPT field.
fn tokens(rpt: &RecursivePrefixTreeStrategy, p: &Arc<dyn Shape>) -> i64 {
    let analyzer = lucene_analysis::Analyzer::standard(None);
    let mut n: i64 = 0;
    let mut h: i64 = 0;
    for f in rpt.create_indexable_fields(p).unwrap() {
        let ts = f.token_stream(&analyzer).unwrap().unwrap();
        for t in &ts.tokens {
            n += 1;
            let last = t.term.last().map_or(0, |&b| i64::from(b as i8));
            h = h.wrapping_mul(31).wrapping_add(t.term.len() as i64 + last);
        }
    }
    n * 1_000_003 + (h & 0xffff)
}

/// Java's `Math.round(double)`.
fn round(v: f64) -> i64 {
    (v + 0.5).floor() as i64
}

pub fn bench_spatial_extras(w: Duration, m: Duration, dir: &str) {
    let text = std::fs::read_to_string(format!("{dir}/spatial-queries.tsv"))
        .expect("run SpatialExtrasMicro build first (scripts/bench-micro.sh --bench spatial_extras)");
    let geo = SpatialContext::geo_context();
    let mut args = BTreeMap::new();
    args.insert(
        "spatialContextFactory".to_string(),
        "org.apache.lucene.spatial.spatial4j.Geo3dSpatialContextFactory".to_string(),
    );
    let g3 = SpatialContextFactory::make_spatial_context(&args).unwrap();
    let rpt = RecursivePrefixTreeStrategy::new(
        Arc::new(QuadPrefixTree::new(geo.clone(), 11).unwrap()),
        "rpt",
    )
    .unwrap();
    let rpt3 = RecursivePrefixTreeStrategy::new(
        Arc::new(QuadPrefixTree::new(g3.clone(), 11).unwrap()),
        "rpt3",
    )
    .unwrap();
    let bb = BBoxStrategy::new_instance(geo.clone(), "bb").unwrap();
    let dates = Arc::new(DateRangePrefixTree::new(&Calendar::new_default()).unwrap());
    let dr = NumberRangePrefixTreeStrategy::for_dates(dates.clone(), "dr").unwrap();

    let (mut polys, mut rects, mut circles, mut boxes, mut heat, mut date_shapes) =
        (vec![], vec![], vec![], vec![], vec![], vec![]);
    for line in text.lines() {
        let a: Vec<&str> = line.split('\t').collect();
        match a[0] {
            "poly" => polys.push(g3.read_shape_from_wkt(a[1]).unwrap()),
            "rect" => rects.push(geo.read_shape_from_wkt(a[1]).unwrap()),
            "circle" => circles.push(geo.read_shape_from_wkt(a[1]).unwrap()),
            "bbox" => boxes.push(geo.read_shape_from_wkt(a[1]).unwrap()),
            "heat" => heat.push((
                (a[1] != "-").then(|| geo.read_shape_from_wkt(a[1]).unwrap()),
                a[2].parse::<i32>().unwrap(),
            )),
            "date" => date_shapes.push(dates.parse_shape(a[1]).unwrap().into_shape()),
            other => panic!("{other}"),
        }
    }

    // RPT indexing a polygon: every token of its field
    let mut f = Fnv::new();
    for p in &polys {
        f.add(tokens(&rpt3, p));
    }
    check("spx_rpt_index_polygon", &f, polys.len() as u64);
    measure("spx_rpt_index_polygon", w, m, || {
        let mut n = 0;
        for p in black_box(&polys) {
            n += tokens(&rpt3, p);
        }
        black_box(n);
        polys.len() as u64
    });

    let reader = DirectoryReader::open(&FsDirectory::open(std::path::Path::new(dir))).unwrap();
    let mut opened = reader.open_segments().unwrap();
    opened.open_points().unwrap();
    let leaves = opened.as_open_segments();

    for (name, shapes) in [
        ("spx_rpt_intersects_rect", &rects),
        ("spx_rpt_intersects_circle", &circles),
    ] {
        let qs: Vec<Box<dyn DocumentQuery>> = shapes
            .iter()
            .map(|s| {
                rpt.make_query(&SpatialArgs::new(SpatialOperation::Intersects, s.clone()))
                    .unwrap()
            })
            .collect();
        let mut f = Fnv::new();
        for q in &qs {
            f.add(count(&leaves, q.as_ref()));
        }
        check(name, &f, qs.len() as u64);
        measure(name, w, m, || {
            let mut n = 0;
            for q in black_box(&qs) {
                n += count(&leaves, q.as_ref());
            }
            black_box(n);
            qs.len() as u64
        });
    }

    // BBox intersects, every hit's overlap ratio
    let norms = vec![None; leaves.len()];
    let searcher = IndexSearcher::new(&leaves, &norms).unwrap();
    let ctx = ValuesContext::new(&searcher);
    let bq: Vec<Box<dyn DocumentQuery>> = boxes
        .iter()
        .map(|b| {
            bb.make_query(&SpatialArgs::new(SpatialOperation::BBoxIntersects, b.clone()))
                .unwrap()
        })
        .collect();
    let bs: Vec<Arc<dyn DoubleValuesSource>> = boxes
        .iter()
        .map(|b| {
            bb.make_overlap_ratio_value_source(b.bounding_box().unwrap(), 0.25)
                .unwrap()
        })
        .collect();
    let mut f = Fnv::new();
    for (q, s) in bq.iter().zip(&bs) {
        let (n, sum) = similarity(&leaves, &ctx, q.as_ref(), s);
        f.add(n);
        f.add(round(sum * 1e6));
    }
    check("spx_bbox_similarity", &f, bq.len() as u64);
    measure("spx_bbox_similarity", w, m, || {
        for (q, s) in black_box(bq.iter().zip(&bs)) {
            black_box(similarity(&leaves, &ctx, q.as_ref(), s));
        }
        bq.len() as u64
    });

    // heatmaps
    let pt = rpt.as_prefix_tree().unwrap();
    let mut f = Fnv::new();
    for (shape, level) in &heat {
        let hm = pt
            .calc_facets(&leaves, None, shape.as_ref(), *level, 1_000_000)
            .unwrap();
        f.add(i64::from(hm.columns));
        f.add(i64::from(hm.rows));
        f.add(hm.counts.iter().map(|&c| i64::from(c)).sum());
    }
    check("spx_heatmap", &f, heat.len() as u64);
    measure("spx_heatmap", w, m, || {
        for (shape, level) in black_box(&heat) {
            black_box(
                pt.calc_facets(&leaves, None, shape.as_ref(), *level, 1_000_000)
                    .unwrap()
                    .counts
                    .len(),
            );
        }
        heat.len() as u64
    });

    // date ranges
    let dq: Vec<Box<dyn DocumentQuery>> = date_shapes
        .iter()
        .map(|s| {
            dr.make_query(&SpatialArgs::new(SpatialOperation::Intersects, s.clone()))
                .unwrap()
        })
        .collect();
    let mut f = Fnv::new();
    for q in &dq {
        f.add(count(&leaves, q.as_ref()));
    }
    check("spx_date_range", &f, dq.len() as u64);
    measure("spx_date_range", w, m, || {
        let mut n = 0;
        for q in black_box(&dq) {
            n += count(&leaves, q.as_ref());
        }
        black_box(n);
        dq.len() as u64
    });
}
