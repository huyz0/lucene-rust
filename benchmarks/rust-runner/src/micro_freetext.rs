//! M13 T13.3c's FreeText pair (the Java twin is
//! `benchmarks/micro/java/FreeTextSuggestMicro.java`) over the phrases and
//! queries of `micro_analyzing`. `freetext_lookup`: top 5 for every query with
//! a token, units: lookups; `freetext_build`: a build over all phrases, units:
//! phrases.

use std::hint::black_box;
use std::time::Duration;

use lucene_search::suggest::free_text::FreeTextSuggester;
use lucene_search::suggest::lookup::Lookup;

use super::measure;
use super::micro_analyzing::{analyzer, entries, lookups, phrases, queries};

pub fn bench_freetext(w: Duration, m: Duration) {
    let rows = phrases();
    let mut queries = queries(&rows);
    let mut f = FreeTextSuggester::new(analyzer());
    f.build(&mut entries(&rows)).unwrap();
    // a query of stop words alone analyzes to no tokens, which both engines
    // refuse: time the others
    queries.retain(|q| f.lookup(q, None, false, 5).is_ok());
    lookups("freetext_lookup", &f, &queries, w, m);
    measure("freetext_build", w, m, || {
        let mut b = FreeTextSuggester::new(analyzer());
        b.build(&mut entries(&rows)).unwrap();
        black_box(b.get_count().unwrap());
        rows.len() as u64
    });
}
