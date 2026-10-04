//! The function-query benchmark pair (M10 T10.5), against
//! `benchmarks/micro/java/FunctionMicro.java`: per word, one top-10 search
//! -- `FunctionScoreQuery` over `body:word` with a field source and
//! `boostByValue` with a composite source, `FunctionQuery` over a field and
//! a composite arithmetic source (every document), `FunctionRangeQuery`
//! alone and as a filter, `FunctionQuery(termfreq)`, `tf * idf` under
//! `ClassicSimilarity`, and `FunctionMatchQuery` -- over the
//! 200 000-document, four-segment index the Java side builds
//! (`FunctionMicro build <dir>`). Each case prints a `#check` digest of its
//! hits the report compares before it shows a ratio.

use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;

use lucene_search::directory_reader::DirectoryReader;
use lucene_search::function::valuesource::{
    ConstValueSource, FloatFieldSource, IDFValueSource, IntFieldSource, LinearFloatFunction,
    LongFieldSource, ProductFloatFunction, ReciprocalFloatFunction, SumFloatFunction,
    TFValueSource, TermFreqValueSource,
};
use lucene_search::function::{
    as_double_values_source, FunctionMatchQuery, FunctionQuery, FunctionRangeQuery,
    FunctionScoreQuery, ValueSource,
};
use lucene_search::index_searcher::{IndexSearcher, SegmentNorms};
use lucene_search::query::{BooleanQuery, Clause, TermQuery};
use lucene_search::similarities::ClassicSimilarity;
use lucene_search::top_docs::TopDocs;
use lucene_search::values_source;
use lucene_store::FsDirectory;

use super::measure;

/// FNV-1a over 64-bit words, identical to `FunctionMicro.Fnv`.
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

fn digest(f: &mut Fnv, td: &TopDocs) {
    f.add(td.total_hits.value as i64);
    for d in &td.score_docs {
        f.add(i64::from(d.doc));
        let bits = if d.score.is_nan() {
            0x7fc0_0000
        } else {
            d.score.to_bits()
        };
        f.add(i64::from(bits as i32));
    }
}

fn term(word: &str) -> Clause {
    Clause::Term(TermQuery::new("body", word.as_bytes().to_vec()))
}

fn one(c: Clause) -> BooleanQuery {
    BooleanQuery {
        must: vec![c],
        ..Default::default()
    }
}

/// `sum(product(int(i),0.5), linear(float(f),2,1), recip(long(n),0.001,10,1))`.
fn composite() -> Arc<dyn ValueSource> {
    Arc::new(SumFloatFunction::new(vec![
        Arc::new(ProductFloatFunction::new(vec![
            Arc::new(IntFieldSource::new("i")),
            Arc::new(ConstValueSource::new(0.5)),
        ])),
        Arc::new(LinearFloatFunction::new(
            Arc::new(FloatFieldSource::new("f")),
            2.0,
            1.0,
        )),
        Arc::new(ReciprocalFloatFunction::new(
            Arc::new(LongFieldSource::new("n")),
            0.001,
            10.0,
            1.0,
        )),
    ]))
}

fn cases(
    name: &str,
    words: &[String],
    s: &IndexSearcher<'_, '_>,
    w: Duration,
    m: Duration,
    query: &dyn Fn(&str) -> BooleanQuery,
) {
    if std::env::var("MICRO_CASE").is_ok_and(|only| only != name) {
        return;
    }
    // The queries are built once per word, as the Java side's are per search;
    // building them is not what is measured.
    let queries: Vec<BooleanQuery> = words.iter().map(|w| query(w)).collect();
    let mut f = Fnv::new();
    for q in &queries {
        digest(&mut f, &s.search(q, 10).unwrap());
    }
    println!("#check\t{name}\t{:016x}\t{}", f.0, words.len());
    measure(name, w, m, || {
        let mut g = Fnv::new();
        for q in black_box(&queries) {
            digest(&mut g, &s.search(q, 10).unwrap());
        }
        black_box(g.0);
        words.len() as u64
    });
}

pub fn bench_function(w: Duration, m: Duration, dir: &str) {
    let words: Vec<String> = std::fs::read_to_string(format!("{dir}/function-words.tsv"))
        .expect("run FunctionMicro build first (scripts/bench-micro.sh --bench function)")
        .lines()
        .map(str::to_string)
        .collect();
    let reader = DirectoryReader::open(&FsDirectory::open(std::path::Path::new(dir))).unwrap();
    let opened = reader.open_segments().unwrap();
    // No query cache, as the Java side's `setQueryCache(null)`: a cached
    // filter's set changes where a top-10 search stops counting hits.
    let mut segments = opened.as_open_segments();
    for seg in &mut segments {
        seg.cache = None;
    }
    let owned = reader.field_norms_by_field(&["body".to_string()]);
    let norms: Vec<SegmentNorms<'_, '_>> = owned.iter().map(Some).collect();
    let s = IndexSearcher::new(&segments, &norms).unwrap();
    let sim = ClassicSimilarity::default();
    let mut classic = IndexSearcher::new(&segments, &norms).unwrap();
    classic.set_similarity(&sim);

    cases("fn_score_field", &words, &s, w, m, &|word| {
        one(FunctionScoreQuery::new(term(word), values_source::from_float_field("f")).into())
    });
    cases("fn_boost_composite", &words, &s, w, m, &|word| {
        one(
            FunctionScoreQuery::boost_by_value(term(word), as_double_values_source(composite()))
                .into(),
        )
    });
    cases("fn_query_field", &words, &s, w, m, &|_| {
        one(FunctionQuery::new(Arc::new(FloatFieldSource::new("f"))).into())
    });
    cases("fn_query_composite", &words, &s, w, m, &|_| {
        one(FunctionQuery::new(composite()).into())
    });
    cases("fn_range", &words, &s, w, m, &|_| {
        one(FunctionRangeQuery::new(
            Arc::new(IntFieldSource::new("i")),
            Some("100"),
            Some("180"),
            true,
            false,
        )
        .into())
    });
    cases("fn_range_filter", &words, &s, w, m, &|word| BooleanQuery {
        must: vec![term(word)],
        filter: vec![FunctionRangeQuery::new(
            Arc::new(IntFieldSource::new("i")),
            Some("100"),
            Some("500"),
            true,
            true,
        )
        .into()],
        ..Default::default()
    });
    cases("fn_termfreq", &words, &s, w, m, &|word| {
        one(FunctionQuery::new(Arc::new(TermFreqValueSource::new(
            "body",
            word,
            "body",
            word.as_bytes(),
        )))
        .into())
    });
    cases("fn_tf_idf", &words, &classic, w, m, &|word| {
        let product: Arc<dyn ValueSource> = Arc::new(ProductFloatFunction::new(vec![
            Arc::new(TFValueSource::new("body", word, "body", word.as_bytes())),
            Arc::new(IDFValueSource::new("body", word, "body", word.as_bytes())),
        ]));
        one(FunctionScoreQuery::new(term(word), as_double_values_source(product)).into())
    });
    cases("fn_match", &words, &s, w, m, &|word| BooleanQuery {
        must: vec![term(word)],
        filter: vec![FunctionMatchQuery::new(
            values_source::from_int_field("i"),
            Arc::new(|v| v > 500.0),
        )
        .into()],
        ..Default::default()
    });
}
