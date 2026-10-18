//! M13 T13.4c's drill-sideways pair (the Java twin is
//! `benchmarks/micro/java/DrillSidewaysMicro.java`), with the same case
//! names, over the corpus `FacetCountsMicro build <dir>` writes. Each case is
//! one sequential `DrillSideways::search(ddq, 10)`: the `DrillSidewaysQuery`
//! search and the `FastTaxonomyFacetCounts` of the drill-down and of every
//! sideways dim. One case per `DrillSidewaysScorer` strategy:
//! `drill_sideways_union` (match-all base, `Category/c3`),
//! `drill_sideways_advance` (match-all base, `Author/a5` and `Tag/t7`: the
//! second dim far cheaper than the base) and `drill_sideways_query_first`
//! (`q:rare` base, `Category/c3` and `Tag/t7`: the base far cheaper than the
//! dims). Units: searches. Each prints a `#check` digest of its hits and of
//! `getTopChildren(10, dim)` for each dim first.

use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

use lucene_search::directory_reader::DirectoryReader;
use lucene_search::facets::taxonomy::DirectoryTaxonomyReader;
use lucene_search::facets::{DrillDownQuery, DrillSideways, DrillSidewaysResult, Facets};
use lucene_search::field_norms::FieldNorms;
use lucene_search::index_searcher::{IndexSearcher, SegmentNorms};
use lucene_search::query::{Clause, MatchAllDocsQuery, TermQuery};
use lucene_search::top_docs::TopDocs;
use lucene_store::FsDirectory;

use super::micro_facet_counts::{cases, config, Fnv};

/// `DrillSidewaysMicro.digest`.
fn digest(f: &mut Fnv, r: &DrillSidewaysResult<'_, '_, TopDocs>, dims: &[&str]) {
    f.add(r.hits.total_hits.value as i64);
    for sd in &r.hits.score_docs {
        f.add(i64::from(sd.doc));
        f.add(i64::from(sd.score.to_bits() as i32));
    }
    for d in dims {
        f.result(r.facets.get_top_children(10, d, &[]).unwrap());
    }
}

pub fn bench_drill_sideways(w: Duration, m: Duration, dir: &str) {
    let dir = Path::new(dir);
    assert!(
        dir.join("built").exists(),
        "run FacetCountsMicro build first (scripts/bench-micro.sh --bench drill_sideways)"
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
    let ds = DrillSideways::new(&searcher, &config, &taxo);
    let all = || Clause::MatchAllDocs(MatchAllDocsQuery::new(reader.max_doc()));
    let rare = || Clause::Term(TermQuery::new("q", "rare"));

    cases("drill_sideways_union", w, m, &|f| {
        let mut q = DrillDownQuery::with_base(&config, all());
        q.add("Category", &["c3"]).unwrap();
        digest(f, &ds.search(&q, 10).unwrap(), &["Category", "Author"]);
        1
    });
    cases("drill_sideways_advance", w, m, &|f| {
        let mut q = DrillDownQuery::with_base(&config, all());
        q.add("Author", &["a5"]).unwrap();
        q.add("Tag", &["t7"]).unwrap();
        digest(f, &ds.search(&q, 10).unwrap(), &["Author", "Tag", "Category"]);
        1
    });
    cases("drill_sideways_query_first", w, m, &|f| {
        let mut q = DrillDownQuery::with_base(&config, rare());
        q.add("Category", &["c3"]).unwrap();
        q.add("Tag", &["t7"]).unwrap();
        digest(f, &ds.search(&q, 10).unwrap(), &["Category", "Tag", "Author"]);
        1
    });
}
