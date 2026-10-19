#![allow(clippy::arithmetic_side_effects)]
//! M14 T14.2: an expression as a `DoubleValuesSource`, sort and rescorer over
//! a real (Lucene-written) index -- `values_rescore_index`, two segments, the
//! first with deletions, sparse `n` (long), `i` (int), `f` (float) and `d`
//! (double) doc values. The expected values are computed here from the plain
//! field sources, so the test pins what the expression layer adds: variables
//! bound to fields, a missing value reading `0`, nested expressions, scores,
//! the sort field and the rescorer. (The compiled arithmetic itself is
//! differential: `expressions_fixtures.rs`.) `expressions_match_java_over_an_index`
//! is differential too: values, sorted hits, rescoring and explanations
//! against Java's over `fixtures/data/expressions/index`.

mod m7support;

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use lucene_search::directory_reader::DirectoryReader;
use lucene_search::expressions::{
    compile, ExpressionRescorer, ExpressionValueSource, SimpleBindings,
};
use lucene_search::field_norms::FieldNorms;
use lucene_search::index_searcher::IndexSearcher;
use lucene_search::query::{BooleanQuery, Clause, MatchAllDocsQuery, TermQuery};
use lucene_search::rescorer::Rescorer;
use lucene_search::values_source::{self as vs, DoubleValuesSource, ValuesContext};
use lucene_store::FsDirectory;
use m7support::fixture;

/// Every document's value under `source`, `None` where it has none.
fn read(
    ctx: &ValuesContext<'_>,
    source: &dyn DoubleValuesSource,
    max_docs: &[i32],
) -> Vec<Option<f64>> {
    let mut out = Vec::new();
    for (leaf, &max_doc) in max_docs.iter().enumerate() {
        let mut v = source.get_values(ctx, leaf, None).unwrap();
        for doc in 0..max_doc {
            out.push(
                v.advance_exact(doc)
                    .unwrap()
                    .then(|| v.double_value().unwrap()),
            );
        }
    }
    out
}

#[test]
fn expressions_over_a_real_index() {
    let dir = fixture("values_rescore_index");
    let reader = DirectoryReader::open(&FsDirectory::open(&dir)).expect("open reader");
    let mut opened = reader.open_segments().expect("open postings");
    opened.open_points().expect("open points");
    let segments = opened.as_open_segments();
    let owned: Vec<HashMap<String, FieldNorms<'_>>> = reader
        .field_norms("body")
        .into_iter()
        .map(|n| n.into_iter().map(|n| ("body".to_string(), n)).collect())
        .collect();
    let norms: Vec<Option<&HashMap<String, FieldNorms<'_>>>> = owned.iter().map(Some).collect();
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();
    let max_docs: Vec<i32> = reader.segment_readers().iter().map(|r| r.max_doc).collect();
    let ctx = ValuesContext::new(&searcher);

    let n = read(&ctx, vs::from_long_field("n").as_ref(), &max_docs);
    let d = read(&ctx, vs::from_double_field("d").as_ref(), &max_docs);
    assert!(n.iter().any(Option::is_none) && n.iter().any(Option::is_some));
    assert!(d.iter().any(Option::is_none) && d.iter().any(Option::is_some));

    // Variables bound to fields; a missing value reads 0.
    let mut bindings = SimpleBindings::new();
    bindings.add_source("n", vs::from_long_field("n"));
    bindings.add_source("d", vs::from_double_field("d"));
    bindings.add_source("_score", vs::scores());
    let e = Arc::new(compile("n * 2 + d").unwrap());
    let source = e.double_values_source(&bindings).unwrap();
    assert_eq!(source.describe(), "expr(n * 2 + d)");
    let got = read(&ctx, source.as_ref(), &max_docs);
    let want: Vec<f64> = n
        .iter()
        .zip(&d)
        .map(|(n, d)| n.unwrap_or(0.0) * 2.0 + d.unwrap_or(0.0))
        .collect();
    assert_eq!(got.len(), want.len());
    for (doc, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(g.unwrap().to_bits(), w.to_bits(), "doc {doc}");
    }

    // One expression bound as another's variable, plain and caching.
    let outer = Arc::new(compile("inner - n + sqrt(abs(d))").unwrap());
    bindings.add_expression("inner", e.clone());
    bindings.validate().unwrap();
    for caching in [false, true] {
        let source: Arc<dyn DoubleValuesSource> = if caching {
            Arc::new(ExpressionValueSource::caching(&bindings, outer.clone()).unwrap())
        } else {
            Arc::new(ExpressionValueSource::new(&bindings, outer.clone()).unwrap())
        };
        let got = read(&ctx, source.as_ref(), &max_docs);
        for (doc, g) in got.iter().enumerate() {
            let (n, d) = (n[doc].unwrap_or(0.0), d[doc].unwrap_or(0.0));
            let want = (n * 2.0 + d) - n + d.abs().sqrt();
            assert_eq!(
                g.unwrap().to_bits(),
                want.to_bits(),
                "caching={caching} doc {doc}"
            );
        }
    }

    // A variable with no binding is refused when the source is made.
    let missing = Arc::new(compile("n + nobody").unwrap());
    assert!(missing.double_values_source(&bindings).is_err());

    // The sort field orders by the expression (descending), the missing
    // documents (value 0 there) included.
    let first = searcher.search(&match_all(), 40).unwrap();
    let rescorer = ExpressionRescorer::new(e.clone(), &bindings).unwrap();
    let rescored = rescorer.rescore(&ctx, &first, 40).unwrap();
    assert_eq!(rescored.score_docs.len(), first.score_docs.len());
    let by_expression = |doc: i32| want[doc as usize];
    let mut expected: Vec<i32> = first.score_docs.iter().map(|h| h.doc).collect();
    expected.sort_by(|a, b| {
        by_expression(*b)
            .total_cmp(&by_expression(*a))
            .then(a.cmp(b))
    });
    let docs: Vec<i32> = rescored.score_docs.iter().map(|h| h.doc).collect();
    assert_eq!(docs, expected);
    // `topN` cuts the list.
    assert_eq!(
        rescorer.rescore(&ctx, &first, 3).unwrap().score_docs.len(),
        3
    );
    assert_eq!(rescorer.expression().source_text, "n * 2 + d");
}

/// A source that counts the `get_values` calls made on it.
struct Counting(Arc<dyn DoubleValuesSource>, Arc<AtomicUsize>);

impl DoubleValuesSource for Counting {
    fn get_values<'c>(
        &self,
        ctx: &ValuesContext<'c>,
        leaf: usize,
        scores: Option<vs::BoxDoubleValues<'c>>,
    ) -> lucene_search::Result<vs::BoxDoubleValues<'c>> {
        self.1.fetch_add(1, Ordering::Relaxed);
        self.0.get_values(ctx, leaf, scores)
    }
    fn needs_scores(&self) -> bool {
        false
    }
    fn is_cacheable(&self, _ctx: &ValuesContext<'_>, _leaf: usize) -> bool {
        false
    }
    fn describe(&self) -> String {
        format!("counting({})", self.0.describe())
    }
}

/// `CachingExpressionValueSource`: two caching expressions bound as
/// variables that both name `c` share one reading of `c` per leaf under a
/// caching source; a plain one reads it once per nested expression. The
/// values are the same either way.
#[test]
fn a_caching_source_reads_a_shared_variable_once() {
    let dir = fixture("values_rescore_index");
    let reader = DirectoryReader::open(&FsDirectory::open(&dir)).expect("open reader");
    let mut opened = reader.open_segments().expect("open postings");
    opened.open_points().expect("open points");
    let segments = opened.as_open_segments();
    let norms: Vec<Option<&HashMap<String, FieldNorms<'_>>>> = vec![None; segments.len()];
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();
    let max_docs: Vec<i32> = reader.segment_readers().iter().map(|r| r.max_doc).collect();
    let ctx = ValuesContext::new(&searcher);

    let reads = Arc::new(AtomicUsize::new(0));
    let mut bindings = SimpleBindings::new();
    bindings.add_source(
        "c",
        Arc::new(Counting(vs::from_long_field("n"), reads.clone())),
    );
    for (name, text) in [("a", "c * 2"), ("b", "c + 1")] {
        let nested =
            ExpressionValueSource::caching(&bindings, Arc::new(compile(text).unwrap())).unwrap();
        bindings.add_source(name, Arc::new(nested));
    }
    let outer = Arc::new(compile("a + b").unwrap());
    let n = read(&ctx, vs::from_long_field("n").as_ref(), &max_docs);
    let mut values = Vec::new();
    for (caching, per_leaf) in [(true, 1), (false, 2)] {
        let source = if caching {
            ExpressionValueSource::caching(&bindings, outer.clone()).unwrap()
        } else {
            ExpressionValueSource::new(&bindings, outer.clone()).unwrap()
        };
        reads.store(0, Ordering::Relaxed);
        let got = read(&ctx, &source, &max_docs);
        assert_eq!(
            reads.load(Ordering::Relaxed),
            per_leaf * max_docs.len(),
            "caching={caching}"
        );
        for (doc, g) in got.iter().enumerate() {
            let c = n[doc].unwrap_or(0.0);
            assert_eq!(g.unwrap(), c * 2.0 + (c + 1.0), "doc {doc}");
        }
        values.push(got);
    }
    assert_eq!(values[0], values[1]);
}

fn match_all() -> BooleanQuery {
    BooleanQuery {
        must: vec![Clause::MatchAllDocs(MatchAllDocsQuery::new(i32::MAX))],
        ..BooleanQuery::default()
    }
}

/// `AnalysisRows.esc`'s inverse (the explanation text holds only ASCII, line
/// breaks and backslashes).
fn unesc(s: &str) -> String {
    let mut out = String::new();
    let mut it = s.chars();
    while let Some(c) = it.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match it.next() {
            Some('t') => out.push('\t'),
            Some('n') => out.push('\n'),
            Some('r') => out.push('\r'),
            Some(o) => out.push(o),
            None => out.push('\\'),
        }
    }
    out
}

/// `NumericUtils.doubleToSortableLong` of raw bits.
fn sortable(bits: i64) -> i64 {
    bits ^ ((bits >> 63) & 0x7fff_ffff_ffff_ffff)
}

/// `fixtures/data/expressions/index.tsv` (`GenExpressions.indexRows`): every
/// document's value (`X`), the top hits sorted by the expression (`S`),
/// `ExpressionRescorer`'s rescoring (`R`) and its explanations (`P`) --
/// expressions over a long and a double field, `_score` and a nested
/// expression, compared with Java's output bit for bit.
#[test]
fn expressions_match_java_over_an_index() {
    let dir = fixture("expressions/index");
    let reader = DirectoryReader::open(&FsDirectory::open(&dir)).expect("open reader");
    let mut opened = reader.open_segments().expect("open postings");
    opened.open_points().expect("open points");
    let segments = opened.as_open_segments();
    let owned: Vec<HashMap<String, FieldNorms<'_>>> = reader
        .field_norms("body")
        .into_iter()
        .map(|n| n.into_iter().map(|n| ("body".to_string(), n)).collect())
        .collect();
    let norms: Vec<Option<&HashMap<String, FieldNorms<'_>>>> = owned.iter().map(Some).collect();
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();
    let max_docs: Vec<i32> = reader.segment_readers().iter().map(|r| r.max_doc).collect();
    let ctx = ValuesContext::new(&searcher);

    let mut bindings = SimpleBindings::new();
    bindings.add_source("n", vs::from_long_field("n"));
    bindings.add_source("d", vs::from_double_field("d"));
    bindings.add_source("_score", vs::scores());
    bindings.add_expression("inner", Arc::new(compile("n * 2 + d").unwrap()));
    let query = BooleanQuery {
        must: vec![Clause::Term(TermQuery::new("body", b"w1".to_vec()))],
        ..BooleanQuery::default()
    };

    let text = std::fs::read_to_string(fixture("expressions/index.tsv")).unwrap();
    let mut values: HashMap<String, Vec<Option<f64>>> = HashMap::new();
    let mut checked = [0usize; 4];
    for line in text.lines() {
        let c: Vec<&str> = line.split('\t').collect();
        let e = Arc::new(compile(c[1]).unwrap());
        match c[0] {
            "X" => {
                let got = values.entry(c[1].to_string()).or_insert_with(|| {
                    let source = e.double_values_source(&bindings).unwrap();
                    assert!(!source.needs_scores(), "{}", c[1]);
                    read(&ctx, source.as_ref(), &max_docs)
                });
                let doc: usize = c[2].parse().unwrap();
                let got = got[doc].map(|v| format!("{:x}", v.to_bits()));
                assert_eq!(got.as_deref().unwrap_or("-"), c[3], "X {} doc {doc}", c[1]);
                checked[0] += 1;
            }
            "S" => {
                let sort = vec![e.sort_field(&bindings, true).unwrap()];
                let sort = lucene_search::top_field::rewrite_sort(&sort, &ctx).unwrap();
                let got = lucene_search::top_field::search_sorted(
                    &segments,
                    reader.segment_readers(),
                    &query,
                    &norms,
                    &sort,
                    12,
                    u64::MAX,
                    None,
                )
                .unwrap();
                let got: Vec<String> = got
                    .hits
                    .iter()
                    .map(|h| format!("{}:{}", h.doc, h.values[0]))
                    .collect();
                let want: Vec<String> = c[2]
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(|s| {
                        let (doc, bits) = s.split_once(':').unwrap();
                        let bits = u64::from_str_radix(bits, 16).unwrap() as i64;
                        format!("{doc}:{}", sortable(bits))
                    })
                    .collect();
                assert_eq!(got, want, "S {}", c[1]);
                checked[1] += 1;
            }
            "R" => {
                let rescorer = ExpressionRescorer::new(e.clone(), &bindings).unwrap();
                let first = searcher.search(&query, 40).unwrap();
                let got = rescorer.rescore(&ctx, &first, 10).unwrap();
                let got: Vec<String> = got
                    .score_docs
                    .iter()
                    .map(|h| format!("{}:{:x}", h.doc, h.score.to_bits()))
                    .collect();
                let want: Vec<&str> = c[2].split(',').filter(|s| !s.is_empty()).collect();
                assert_eq!(got, want, "R {}", c[1]);
                checked[2] += 1;
            }
            "P" => {
                let rescorer = ExpressionRescorer::new(e.clone(), &bindings).unwrap();
                let doc: i32 = c[2].parse().unwrap();
                let first = searcher.explain(&query, doc).unwrap();
                let got = rescorer.explain(&ctx, &first, doc).unwrap().to_string();
                assert_eq!(got, unesc(c[3]), "P {} doc {doc}", c[1]);
                checked[3] += 1;
            }
            other => panic!("row kind {other}"),
        }
    }
    assert_eq!(checked, [150, 5, 5, 15]);
}
