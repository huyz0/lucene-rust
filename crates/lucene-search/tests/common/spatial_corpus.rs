//! `SpatialExtrasCorpus.java`'s strategies, in Rust: shared by
//! `tests/spatial_strategies_fixtures.rs` and the
//! `write_spatial_strategies_fixture` example (`VerifySpatialExtras`).

#![allow(dead_code)]

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use lucene_index::document::{Document, IndexableField, Store, StringField};
use lucene_search::spatial::bbox::BBoxStrategy;
use lucene_search::spatial::{
    CompositeSpatialStrategy, NumberRangePrefixTreeStrategy, PointVectorStrategy,
    RecursivePrefixTreeStrategy, SerializedDVStrategy, SpatialStrategy,
    TermQueryPrefixTreeStrategy,
};
use lucene_util::spatial4j::{Shape, SpatialContext, SpatialContextFactory};
use lucene_util::spatial_extras::prefix_tree::{
    Calendar, DateRangePrefixTree, GeohashPrefixTree, PackedQuadPrefixTree, QuadPrefixTree,
    S2PrefixTree,
};

fn ctx(kv: &[(&str, &str)]) -> Arc<SpatialContext> {
    let m: BTreeMap<String, String> = kv
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    SpatialContextFactory::make_spatial_context(&m).unwrap()
}

/// `SpatialExtrasCorpus.S`.
pub struct Corpus {
    pub geo: Arc<SpatialContext>,
    pub g3: Arc<SpatialContext>,
    pub flat: Arc<SpatialContext>,
    pub dates: Arc<DateRangePrefixTree>,
    pub order: Vec<(&'static str, &'static str)>,
    pub s: HashMap<&'static str, Box<dyn SpatialStrategy>>,
    pub bb: HashMap<&'static str, BBoxStrategy>,
    pub sdv: HashMap<&'static str, SerializedDVStrategy>,
    pub dr: NumberRangePrefixTreeStrategy,
}

pub fn corpus() -> Corpus {
    let geo = SpatialContext::geo_context();
    let g3 = ctx(&[(
        "spatialContextFactory",
        "org.apache.lucene.spatial.spatial4j.Geo3dSpatialContextFactory",
    )]);
    let flat = ctx(&[
        ("geo", "false"),
        ("worldBounds", "ENVELOPE(-1000, 1000, 1000, -1000)"),
    ]);
    let dates = Arc::new(DateRangePrefixTree::new(&Calendar::new_default()).unwrap());
    let mut s: HashMap<&'static str, Box<dyn SpatialStrategy>> = HashMap::new();
    let rpt = |grid: Arc<dyn lucene_util::spatial_extras::prefix_tree::SpatialPrefixTree>,
               name: &str| { RecursivePrefixTreeStrategy::new(grid, name).unwrap() };
    s.insert(
        "rgh",
        Box::new(rpt(
            Arc::new(GeohashPrefixTree::new(geo.clone(), 4).unwrap()),
            "rgh",
        )),
    );
    s.insert(
        "rq",
        Box::new(rpt(
            Arc::new(QuadPrefixTree::new(geo.clone(), 8).unwrap()),
            "rq",
        )),
    );
    let mut rqn = rpt(
        Arc::new(QuadPrefixTree::new(geo.clone(), 8).unwrap()),
        "rqn",
    );
    rqn.set_prune_leafy_branches(false);
    rqn.set_prefix_grid_scan_level(3);
    rqn.set_multi_overlapping_indexed_shapes(false);
    rqn.base_mut().set_dist_err_pct(0.1);
    s.insert("rqn", Box::new(rqn));
    s.insert(
        "rpq",
        Box::new(rpt(
            Arc::new(PackedQuadPrefixTree::new(geo.clone(), 10).unwrap()),
            "rpq",
        )),
    );
    s.insert(
        "rs2",
        Box::new(rpt(
            Arc::new(S2PrefixTree::new(g3.clone(), 5, 1).unwrap()),
            "rs2",
        )),
    );
    s.insert(
        "rfl",
        Box::new(rpt(
            Arc::new(QuadPrefixTree::new(flat.clone(), 9).unwrap()),
            "rfl",
        )),
    );
    let mut pts = rpt(
        Arc::new(QuadPrefixTree::new(geo.clone(), 10).unwrap()),
        "pts",
    );
    pts.base_mut().set_points_only(true);
    s.insert("pts", Box::new(pts));
    s.insert(
        "tq",
        Box::new(
            TermQueryPrefixTreeStrategy::new(
                Arc::new(GeohashPrefixTree::new(geo.clone(), 5).unwrap()),
                "tq",
            )
            .unwrap(),
        ),
    );
    let mut bb = HashMap::new();
    bb.insert("bb", BBoxStrategy::new_instance(geo.clone(), "bb").unwrap());
    let mut all = lucene_index::document::FieldType::copy_of(
        &lucene_search::spatial::bbox::default_field_type(),
    );
    all.set_stored(true).unwrap();
    bb.insert("bbf", BBoxStrategy::new(flat.clone(), "bbf", all).unwrap());
    for (k, v) in &bb {
        s.insert(k, Box::new(v.clone()));
    }
    s.insert(
        "pv",
        Box::new(PointVectorStrategy::new_instance(geo.clone(), "pv").unwrap()),
    );
    let mut sdv = HashMap::new();
    sdv.insert(
        "sdv",
        SerializedDVStrategy::new(geo.clone(), "sdv").unwrap(),
    );
    sdv.insert("sd3", SerializedDVStrategy::new(g3.clone(), "sd3").unwrap());
    for (k, v) in &sdv {
        s.insert(k, Box::new(v.clone()));
    }
    let cmp = CompositeSpatialStrategy::new(
        "cmp",
        rpt(
            Arc::new(QuadPrefixTree::new(geo.clone(), 8).unwrap()),
            "cmp_rpt",
        ),
        SerializedDVStrategy::new(geo.clone(), "cmp_sdv").unwrap(),
    )
    .unwrap();
    let mut cmpn = CompositeSpatialStrategy::new(
        "cmpn",
        cmp.index_strategy().clone(),
        cmp.geometry_strategy().clone(),
    )
    .unwrap();
    cmpn.set_optimize_predicates(false);
    s.insert("cmp", Box::new(cmp));
    s.insert("cmpn", Box::new(cmpn));
    let dr = NumberRangePrefixTreeStrategy::for_dates(dates.clone(), "dr").unwrap();
    s.insert("dr", Box::new(dr.clone()));
    let order = vec![
        ("rgh", "g"),
        ("rq", "g"),
        ("rqn", "g"),
        ("rpq", "g"),
        ("rs2", "t"),
        ("rfl", "f"),
        ("pts", "p"),
        ("tq", "g"),
        ("bb", "g"),
        ("bbf", "f"),
        ("pv", "p"),
        ("sdv", "g"),
        ("sd3", "t"),
        ("cmp", "g"),
        ("cmpn", "-"),
        ("dr", "d"),
    ];
    Corpus {
        geo,
        g3,
        flat,
        dates,
        order,
        s,
        bb,
        sdv,
        dr,
    }
}

impl Corpus {
    pub fn context_of(&self, family: &str) -> &Arc<SpatialContext> {
        match family {
            "t" => &self.g3,
            "f" => &self.flat,
            _ => &self.geo,
        }
    }

    /// A spec's shape.
    pub fn shape(&self, spec: &str) -> lucene_search::Result<Arc<dyn Shape>> {
        let (family, text) = spec.split_once(':').unwrap();
        if family == "d" {
            return Ok(self.dates.parse_shape(text)?.into_shape());
        }
        Ok(self.context_of(family).read_shape_from_wkt(text)?)
    }

    pub fn fields(
        &self,
        strategy: &str,
        spec: &str,
    ) -> lucene_search::Result<Vec<Box<dyn IndexableField>>> {
        self.s[strategy].create_indexable_fields(&self.shape(spec)?)
    }

    pub fn document(&self, line: &str) -> Document {
        let mut p = line.split('\t');
        let mut doc = Document::new();
        doc.add(StringField::new("id", p.next().unwrap(), Store::Yes));
        for f in p {
            let (name, spec) = f.split_once('=').unwrap();
            for field in self.fields(name, spec).unwrap() {
                doc.add_boxed(field);
            }
        }
        doc
    }
}
