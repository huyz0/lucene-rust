//! M13 T13.3c's infix pair (the Java twin is
//! `benchmarks/micro/java/InfixSuggestMicro.java`): the phrases of
//! `micro_analyzing` repeated to [`SCALE`] suggestions, each with its number
//! appended, so a lookup has hundreds of matches to rank. `infix_lookup`: top 5
//! for every query of `micro_analyzing`, units: lookups; `infix_build`: a build
//! over the whole dictionary, units: suggestions.

use std::hint::black_box;
use std::time::Duration;

use lucene_search::suggest::infix::AnalyzingInfixSuggester;
use lucene_search::suggest::lookup::Lookup;

use super::measure;
use super::micro_analyzing::{analyzer, entries, lookups, phrases, queries};

/// The dictionary size.
const SCALE: usize = 20_000;

/// The phrases repeated to `SCALE` rows: `phrase i`, weight `w + i % 97`.
fn scaled(rows: &[(String, i64)]) -> Vec<(String, i64)> {
    (0..SCALE)
        .map(|i| {
            let (text, weight) = &rows[i % rows.len()];
            (format!("{text} {i}"), weight + (i % 97) as i64)
        })
        .collect()
}

pub fn bench_infix(w: Duration, m: Duration) {
    let phrases = phrases();
    let queries = queries(&phrases);
    let rows = scaled(&phrases);
    let mut s = AnalyzingInfixSuggester::new(analyzer());
    s.build(&mut entries(&rows)).unwrap();
    lookups("infix_lookup", &s, &queries, w, m);
    measure("infix_build", w, m, || {
        let mut b = AnalyzingInfixSuggester::new(analyzer());
        b.build(&mut entries(&rows)).unwrap();
        black_box(b.get_count().unwrap());
        rows.len() as u64
    });
}
