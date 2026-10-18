//! M13 T13.4d1's value- and range-facet pair (the Java twin is
//! `benchmarks/micro/java/ValueFacetsMicro.java`), with the same case
//! names, over the 200,000-document, four-segment index
//! `ValueFacetsMicro build <dir>` writes: `l` a single-valued long, `lm` 1-3
//! longs, `s` one of 2,000 strings (`SORTED`) and `d` a double. Every case
//! counts the match-all hits, collected beforehand, and reads the result:
//! `value_facet_long` and `value_facet_long_multi` (`LongValueFacetCounts`
//! over `l` and `lm`, `getTopChildren(10, ..)`), `value_facet_string`
//! (`StringValueFacetCounts` over `s`, its state built beforehand),
//! `value_facet_long_range` (`l` into five exclusive ranges),
//! `value_facet_long_range_overlap` (`lm` into five overlapping ones) and
//! `value_facet_double_range` (`d` into four ranges), each range count then
//! `getAllChildren`. Units: hits. Each prints a `#check` digest of its result
//! first.

use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

use lucene_search::directory_reader::DirectoryReader;
use lucene_search::facets::range::{
    DoubleRange, DoubleRangeFacetCounts, LongRange, LongRangeFacetCounts,
};
use lucene_search::facets::{
    Facets, FacetsCollectorManager, LongValueFacetCounts, StringDocValuesReaderState,
    StringValueFacetCounts,
};
use lucene_search::field_norms::FieldNorms;
use lucene_search::index_searcher::{IndexSearcher, SegmentNorms};
use lucene_search::query::{BooleanQuery, Clause, MatchAllDocsQuery};
use lucene_store::FsDirectory;

use super::micro_facet_counts::cases;

pub fn bench_value_facets(w: Duration, m: Duration, dir: &str) {
    let dir = Path::new(dir);
    assert!(
        dir.join("built").exists(),
        "run ValueFacetsMicro build first (scripts/bench-micro.sh --bench value_facets)"
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
    let all_query = BooleanQuery {
        must: vec![Clause::MatchAllDocs(MatchAllDocsQuery::new(
            reader.max_doc(),
        ))],
        ..Default::default()
    };
    let all = FacetsCollectorManager::new(false)
        .collect(&searcher, &all_query)
        .unwrap();
    let hits: u64 = all
        .matching_docs()
        .iter()
        .map(|m| m.total_hits as u64)
        .sum();
    let state = StringDocValuesReaderState::new(&reader, "s").unwrap();
    let l = |label, min, mi, max, ma| LongRange::new(label, min, mi, max, ma).unwrap();
    let exclusive = [
        l("lt100", i64::MIN, true, 100, false),
        l("100-999", 100, true, 1000, false),
        l("1k-5k", 1000, true, 5000, false),
        l("5k-9k", 5000, true, 9000, false),
        l("ge9k", 9000, true, i64::MAX, true),
    ];
    let overlapping = [
        l("lt10", 0, true, 10, false),
        l("lt100", 0, true, 100, false),
        l("lt500", 0, true, 500, false),
        l("mid", 250, true, 750, true),
        l("top", 900, true, 1000, false),
    ];
    let d = |label, min, mi, max, ma| DoubleRange::new(label, min, mi, max, ma).unwrap();
    let doubles = [
        d("neg", f64::NEG_INFINITY, true, 0.0, false),
        d("0-10", 0.0, true, 10.0, false),
        d("10-50", 10.0, true, 50.0, false),
        d("ge50", 50.0, true, f64::INFINITY, true),
    ];

    cases("value_facet_long", w, m, &|f| {
        let c = LongValueFacetCounts::new("l", &all).unwrap();
        f.result(c.get_top_children(10, "l", &[]).unwrap());
        hits
    });
    cases("value_facet_long_multi", w, m, &|f| {
        let c = LongValueFacetCounts::new("lm", &all).unwrap();
        f.result(c.get_top_children(10, "lm", &[]).unwrap());
        hits
    });
    cases("value_facet_string", w, m, &|f| {
        let c = StringValueFacetCounts::with_hits(&state, &all).unwrap();
        f.result(c.get_top_children(10, "s", &[]).unwrap());
        hits
    });
    cases("value_facet_long_range", w, m, &|f| {
        let c = LongRangeFacetCounts::new("l", &all, &exclusive).unwrap();
        f.result(c.get_all_children("l", &[]).unwrap());
        hits
    });
    cases("value_facet_long_range_overlap", w, m, &|f| {
        let c = LongRangeFacetCounts::new("lm", &all, &overlapping).unwrap();
        f.result(c.get_all_children("lm", &[]).unwrap());
        hits
    });
    cases("value_facet_double_range", w, m, &|f| {
        let c = DoubleRangeFacetCounts::new("d", &all, &doubles).unwrap();
        f.result(c.get_all_children("d", &[]).unwrap());
        hits
    });
}
