//! M13 T13.4b's facet-counting pair (the Java twin is
//! `benchmarks/micro/java/FacetCountsMicro.java`), with the same case names,
//! over the 200,000-document, four-segment index and taxonomy the Java side
//! builds (`FacetCountsMicro build <dir>`): a flat `Author`, a three-level
//! hierarchical `Category`, a multi-valued `Tag`, int and float associations
//! in their own index fields, and `q:rare` on every 100th document. Cases (the
//! matching documents collected beforehand, except in `facet_collect_all`):
//! `facet_collect_all`, a `FacetsCollectorManager` search over
//! `MatchAllDocsQuery`, units: hits; `facet_count_all` and `facet_count_rare`,
//! `FastTaxonomyFacetCounts` over every document (dense counts) and over
//! `q:rare` (1%: the sparse counts), units: hits; `facet_top_children`, four
//! `getTopChildren(10, ..)` calls over the match-all counts, units: calls;
//! `facet_int_assoc_sum` and `facet_float_assoc_sum`, the association facets'
//! `SUM` over every document, units: hits. Each prints a `#check` digest of
//! its result first.

use std::collections::HashMap;
use std::hint::black_box;
use std::path::Path;
use std::time::Duration;

use lucene_search::directory_reader::DirectoryReader;
use lucene_search::facets::taxonomy::{
    AssociationAggregationFunction, DirectoryTaxonomyReader, FastTaxonomyFacetCounts,
    TaxonomyFacetFloatAssociations, TaxonomyFacetIntAssociations,
};
use lucene_search::facets::{
    FacetResult, FacetValue, Facets, FacetsCollector, FacetsCollectorManager, FacetsConfig,
};
use lucene_search::field_norms::FieldNorms;
use lucene_search::index_searcher::{IndexSearcher, SegmentNorms};
use lucene_search::query::{BooleanQuery, Clause, MatchAllDocsQuery, TermQuery};
use lucene_store::FsDirectory;

use super::measure;

/// FNV-1a over 64-bit words, identical to `FacetCountsMicro.Fnv`.
pub(crate) struct Fnv(u64);

impl Fnv {
    pub(crate) fn new() -> Self {
        Fnv(0xcbf2_9ce4_8422_2325)
    }
    pub(crate) fn add(&mut self, x: i64) {
        self.0 ^= x as u64;
        self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
    }
    fn str(&mut self, s: &str) {
        for &b in s.as_bytes() {
            self.add(i64::from(b as i8));
        }
        self.add(-1);
    }
    fn num(&mut self, v: FacetValue) {
        match v {
            FacetValue::Float(f) => {
                let bits = if f.is_nan() { 0x7fc0_0000 } else { f.to_bits() };
                self.add(i64::from(bits as i32));
            }
            FacetValue::Int(i) => self.add(i64::from(i)),
        }
    }
    pub(crate) fn result(&mut self, r: Option<FacetResult>) {
        let Some(r) = r else {
            self.add(-2);
            return;
        };
        self.str(&r.dim);
        for p in &r.path {
            self.str(p);
        }
        self.num(r.value);
        self.add(i64::from(r.child_count));
        for lv in &r.label_values {
            self.str(&lv.label);
            self.num(lv.value);
        }
    }
}

/// `FacetCountsMicro.config()`.
pub(crate) fn config() -> FacetsConfig {
    let mut c = FacetsConfig::new();
    c.set_hierarchical("Category", true);
    c.set_multi_valued("Tag", true);
    c.set_index_field_name("Score", "$int_assoc");
    c.set_multi_valued("Score", true);
    c.set_index_field_name("Weight", "$float_assoc");
    c.set_multi_valued("Weight", true);
    c
}

pub(crate) fn cases(name: &str, w: Duration, m: Duration, run: &dyn Fn(&mut Fnv) -> u64) {
    // `MICRO_CASE`: the one case alone, its check included.
    if std::env::var("MICRO_CASE").is_ok_and(|only| only != name) {
        return;
    }
    let mut f = Fnv::new();
    let units = run(&mut f);
    println!("#check\t{name}\t{:016x}\t{units}", f.0);
    measure(name, w, m, || {
        let mut g = Fnv::new();
        let n = run(&mut g);
        black_box(g.0);
        n
    });
}

fn hits(fc: &FacetsCollector<'_>) -> u64 {
    fc.matching_docs().iter().map(|m| m.total_hits as u64).sum()
}

fn tops(f: &mut Fnv, facets: &dyn Facets) {
    f.result(facets.get_top_children(10, "Author", &[]).unwrap());
    f.result(facets.get_top_children(10, "Category", &[]).unwrap());
    f.result(facets.get_top_children(10, "Category", &["c3"]).unwrap());
    f.result(facets.get_top_children(10, "Tag", &[]).unwrap());
}

pub fn bench_facet_counts(w: Duration, m: Duration, dir: &str) {
    let dir = Path::new(dir);
    assert!(
        dir.join("built").exists(),
        "run FacetCountsMicro build first (scripts/bench-micro.sh --bench facet_counts)"
    );
    let reader = DirectoryReader::open(&FsDirectory::open(dir.join("index"))).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let owned = reader.field_norms_by_field(&["q".to_string()]);
    let norms: Vec<SegmentNorms<'_, '_>> = owned
        .iter()
        .map(|m: &HashMap<String, FieldNorms<'_>>| Some(m))
        .collect();
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();
    let taxo_dir = FsDirectory::open(dir.join("taxo"));
    let taxo = DirectoryTaxonomyReader::open(&taxo_dir).unwrap();
    let config = config();
    let all = BooleanQuery {
        must: vec![Clause::MatchAllDocs(MatchAllDocsQuery::new(
            reader.max_doc(),
        ))],
        ..Default::default()
    };
    let rare = BooleanQuery {
        must: vec![Clause::Term(TermQuery::new("q", "rare"))],
        ..Default::default()
    };
    let fcm = FacetsCollectorManager::new(false);
    let fc_all = fcm.collect(&searcher, &all).unwrap();
    let fc_rare = fcm.collect(&searcher, &rare).unwrap();
    let (hits_all, hits_rare) = (hits(&fc_all), hits(&fc_rare));
    let sum = AssociationAggregationFunction::Sum;

    cases("facet_collect_all", w, m, &|f| {
        let fc = fcm.collect(&searcher, &all).unwrap();
        let n = hits(&fc);
        f.add(n as i64);
        n
    });
    cases("facet_count_all", w, m, &|f| {
        let c = FastTaxonomyFacetCounts::new(&taxo, &config, &fc_all).unwrap();
        f.result(c.get_top_children(10, "Author", &[]).unwrap());
        hits_all
    });
    cases("facet_count_rare", w, m, &|f| {
        let c = FastTaxonomyFacetCounts::new(&taxo, &config, &fc_rare).unwrap();
        f.result(c.get_top_children(10, "Author", &[]).unwrap());
        hits_rare
    });
    let counts = FastTaxonomyFacetCounts::new(&taxo, &config, &fc_all).unwrap();
    cases("facet_top_children", w, m, &|f| {
        tops(f, &counts);
        4
    });
    cases("facet_int_assoc_sum", w, m, &|f| {
        let c =
            TaxonomyFacetIntAssociations::with_field("$int_assoc", &taxo, &config, &fc_all, sum)
                .unwrap();
        f.result(c.get_top_children(10, "Score", &[]).unwrap());
        hits_all
    });
    cases("facet_float_assoc_sum", w, m, &|f| {
        let c = TaxonomyFacetFloatAssociations::with_field(
            "$float_assoc",
            &taxo,
            &config,
            &fc_all,
            sum,
        )
        .unwrap();
        f.result(c.get_top_children(10, "Weight", &[]).unwrap());
        hits_all
    });
}
