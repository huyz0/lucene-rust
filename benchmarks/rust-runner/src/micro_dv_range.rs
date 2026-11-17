//! Doc-values range queries counted over the value-facets corpus (the Java
//! twin is `benchmarks/micro/java/DvRangeMicro.java`), with the same case
//! names: `dv_range_long_{narrow,mid,wide}` (`NumericDocValuesField`'s slow
//! range query over `l`), `dv_range_multi_{narrow,wide}` (`SortedNumeric..` over
//! `lm`) and `dv_range_ord_{narrow,wide}` (`SortedDocValuesField` over `s`).
//! Units: hits.

use std::path::Path;
use std::time::Duration;

use lucene_search::collector::ScoringCollector;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::document::doc_values_queries::{
    numeric_doc_values_field, sorted_doc_values_field, sorted_numeric_doc_values_field,
};
use lucene_search::document::DocumentQuery;
use lucene_store::FsDirectory;

use super::micro_facet_counts::cases;

struct Count(u64);

impl ScoringCollector for Count {
    fn collect(&mut self, _doc_id: i32, _score: f32) {
        self.0 += 1;
    }
}

pub fn bench_dv_range(w: Duration, m: Duration, dir: &str) {
    let dir = Path::new(dir);
    assert!(
        dir.join("built").exists(),
        "run DvRangeMicro build first (scripts/bench-micro.sh --bench dv_range)"
    );
    let reader = DirectoryReader::open(&FsDirectory::open(dir.join("index"))).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let count = |q: &dyn DocumentQuery| -> u64 {
        let mut c = Count(0);
        for leaf in &segments {
            q.score_leaf(leaf, 1.0, &mut c).unwrap();
        }
        c.0
    };
    let run = |name: &str, q: &dyn DocumentQuery| {
        cases(name, w, m, &|f| {
            let n = count(q);
            f.add(n as i64);
            n
        });
    };
    run(
        "dv_range_long_narrow",
        &numeric_doc_values_field::new_slow_range_query("l", 100, 199),
    );
    run(
        "dv_range_long_mid",
        &numeric_doc_values_field::new_slow_range_query("l", 1000, 4999),
    );
    run(
        "dv_range_long_wide",
        &numeric_doc_values_field::new_slow_range_query("l", 100, 9899),
    );
    run(
        "dv_range_multi_narrow",
        &sorted_numeric_doc_values_field::new_slow_range_query("lm", 10, 19),
    );
    run(
        "dv_range_multi_wide",
        &sorted_numeric_doc_values_field::new_slow_range_query("lm", 100, 899),
    );
    run(
        "dv_range_ord_narrow",
        &sorted_doc_values_field::new_slow_range_query(
            "s",
            Some(b"v100"),
            Some(b"v104"),
            true,
            true,
        ),
    );
    run(
        "dv_range_ord_wide",
        &sorted_doc_values_field::new_slow_range_query(
            "s",
            Some(b"v1"),
            Some(b"v1zzz"),
            true,
            true,
        ),
    );
}
