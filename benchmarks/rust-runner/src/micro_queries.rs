//! The M10 T10.6 benchmark pair, against
//! `benchmarks/micro/java/QueriesMicro.java`: per word, one top-10 search of
//! each case -- interval queries (ordered, phrase, unordered under
//! `maxgaps`, a phrase over a disjunction, containing, at-least, a prefix,
//! a payload filter), `CommonTermsQuery`, `MoreLikeThisQuery` and the span
//! and payload queries -- over
//! the 200 000-document, four-segment index the Java side builds
//! (`QueriesMicro build <dir>`). Each case prints a `#check` digest of its
//! hits the report compares before it shows a ratio.

use std::hint::black_box;
use std::time::Duration;

use lucene_search::common_terms::CommonTermsQuery;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::index_searcher::{IndexSearcher, SegmentNorms};
use lucene_search::extended_query::{MultiTermQuery, MultiTermSource, RewriteMethod};
use lucene_search::mlt::MoreLikeThisQuery;
use lucene_search::spans::payloads::{
    PayloadDecoder, PayloadFunction, PayloadScoreQuery, SpanPayloadCheckQuery,
};
use lucene_search::spans::SpanNode;
use lucene_search::intervals::{IntervalQuery, Intervals, IntervalsSource, PayloadFilter};
use lucene_search::query::{BooleanQuery, Clause, PrefixQuery};
use lucene_search::query_visitor::Occur;
use lucene_search::top_docs::TopDocs;
use lucene_store::FsDirectory;

use super::measure;

/// FNV-1a over 64-bit words, identical to `QueriesMicro.Fnv`.
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
        f.add(i64::from(d.score.to_bits() as i32));
    }
}

fn one(c: Clause) -> BooleanQuery {
    BooleanQuery {
        must: vec![c],
        ..Default::default()
    }
}

fn iq(field: &str, s: IntervalsSource) -> BooleanQuery {
    one(IntervalQuery::new(field, s).into())
}

fn t(w: &str) -> IntervalsSource {
    Intervals::term(w)
}

type Case = dyn Fn(&str, &str, &str) -> BooleanQuery;

fn cases(name: &str, words: &[String], s: &IndexSearcher<'_, '_>, w: Duration, m: Duration, query: &Case) {
    if std::env::var("MICRO_CASE").is_ok_and(|only| only != name) {
        return;
    }
    let n = words.len();
    let queries: Vec<BooleanQuery> = (0..n)
        .map(|i| query(&words[i], &words[(i + 1) % n], &words[(i + 2) % n]))
        .collect();
    let mut f = Fnv::new();
    for q in &queries {
        digest(&mut f, &s.search(q, 10).unwrap());
    }
    println!("#check\t{name}\t{:016x}\t{n}", f.0);
    measure(name, w, m, || {
        let mut g = Fnv::new();
        for q in black_box(&queries) {
            digest(&mut g, &s.search(q, 10).unwrap());
        }
        black_box(g.0);
        n as u64
    });
}

pub fn bench_queries(w: Duration, m: Duration, dir: &str) {
    let words: Vec<String> = std::fs::read_to_string(format!("{dir}/queries-words.tsv"))
        .expect("run QueriesMicro build first (scripts/bench-micro.sh --bench queries)")
        .lines()
        .map(str::to_string)
        .collect();
    let reader = DirectoryReader::open(&FsDirectory::open(std::path::Path::new(dir))).unwrap();
    let opened = reader.open_segments().unwrap();
    // No query cache, as the Java side's `setQueryCache(null)`.
    let mut segments = opened.as_open_segments();
    for seg in &mut segments {
        seg.cache = None;
    }
    let owned = reader.field_norms_by_field(&["body".to_string(), "pay".to_string()]);
    let norms: Vec<SegmentNorms<'_, '_>> = owned.iter().map(Some).collect();
    let s = IndexSearcher::new(&segments, &norms).unwrap();

    cases("iv_ordered", &words, &s, w, m, &|a, b, _| {
        iq("body", Intervals::ordered(vec![t(a), t(b)]))
    });
    cases("iv_phrase", &words, &s, w, m, &|a, b, _| {
        iq("body", Intervals::phrase_terms(&[a, b]).unwrap())
    });
    cases("iv_unordered_maxgaps", &words, &s, w, m, &|a, b, _| {
        iq(
            "body",
            Intervals::maxgaps(2, Intervals::unordered(vec![t(a), t(b)])).unwrap(),
        )
    });
    cases("iv_or_phrase", &words, &s, w, m, &|a, b, c| {
        iq(
            "body",
            Intervals::phrase(vec![Intervals::or(vec![t(a), t(b)]).unwrap(), t(c)]).unwrap(),
        )
    });
    cases("iv_containing", &words, &s, w, m, &|a, b, c| {
        iq(
            "body",
            Intervals::containing(Intervals::maxwidth(6, Intervals::ordered(vec![t(a), t(c)])), t(b))
                .unwrap(),
        )
    });
    cases("iv_atleast", &words, &s, w, m, &|a, b, c| {
        iq("body", Intervals::maxwidth(5, Intervals::at_least(2, vec![t(a), t(b), t(c)])))
    });
    cases("iv_prefix", &words, &s, w, m, &|a, b, _| {
        iq(
            "body",
            Intervals::ordered(vec![Intervals::prefix(&a.as_bytes()[..2], 128).unwrap(), t(b)]),
        )
    });
    cases("iv_payload", &words, &s, w, m, &|a, b, _| {
        let filter = PayloadFilter::new(|p| p.is_some_and(|p| (p[0] as i8) >= 2));
        iq(
            "pay",
            Intervals::ordered(vec![Intervals::term_with_payload_filter(a, filter), t(b)]),
        )
    });
    let st = |f: &str, w: &str| SpanNode::term(f, w);
    let sp = |q: SpanNode| one(q.into());
    cases("sp_first", &words, &s, w, m, &move |a, _, _| sp(SpanNode::first(st("body", a), 3)));
    cases("sp_pos_range", &words, &s, w, m, &move |a, b, _| {
        let near = SpanNode::near(vec![st("body", a), st("body", b)], 1, true).unwrap();
        sp(SpanNode::position_range(near, 2, 10))
    });
    cases("sp_not", &words, &s, w, m, &move |a, b, _| {
        sp(SpanNode::not(st("body", a), st("body", b), 1, 1).unwrap())
    });
    cases("sp_containing", &words, &s, w, m, &move |a, b, c| {
        let near = SpanNode::near(vec![st("body", a), st("body", c)], 4, true).unwrap();
        sp(SpanNode::containing(near, st("body", b)).unwrap())
    });
    cases("sp_within", &words, &s, w, m, &move |a, b, c| {
        let near = SpanNode::near(vec![st("body", a), st("body", c)], 4, false).unwrap();
        sp(SpanNode::within(near, st("body", b)).unwrap())
    });
    cases("sp_multi", &words, &s, w, m, &move |a, b, _| {
        let prefix = SpanNode::multi_term(MultiTermQuery::new(
            MultiTermSource::Prefix(PrefixQuery::new("body", &a[..2])),
            RewriteMethod::default(),
        ));
        sp(SpanNode::near(vec![prefix, st("body", b)], 2, true).unwrap())
    });
    cases("sp_check", &words, &s, w, m, &move |a, _, _| {
        sp(SpanNode::PayloadCheck(Box::new(SpanPayloadCheckQuery::new(
            st("pay", a),
            vec![Some(vec![1])],
        ))))
    });
    cases("sp_pscore", &words, &s, w, m, &move |a, b, _| {
        let near = SpanNode::near(vec![st("pay", a), st("pay", b)], 2, true).unwrap();
        sp(SpanNode::PayloadScore(Box::new(PayloadScoreQuery::new(
            near,
            PayloadFunction::Sum,
            PayloadDecoder::Float,
            true,
        ))))
    });
    let analyzer = std::sync::Arc::new(lucene_analysis::Analyzer::standard(None));
    cases("mlt_query", &words, &s, w, m, &move |a, b, c| {
        one(MoreLikeThisQuery::new(
            format!("{a} {b} {c} {a} {c} {a}"),
            vec!["body".to_string()],
            std::sync::Arc::clone(&analyzer),
            "body",
        )
        .into())
    });
    cases("ct_split", &words, &s, w, m, &|a, b, c| {
        let mut q = CommonTermsQuery::new(Occur::Should, Occur::Should, 0.5).unwrap();
        for t in [a, b, c] {
            q.add("body", t);
        }
        one(q.into())
    });
    cases("ct_msm", &words, &s, w, m, &|a, b, c| {
        let mut q = CommonTermsQuery::new(Occur::Should, Occur::Should, 0.45).unwrap();
        for t in [a, b, c, "missing"] {
            q.add("body", t);
        }
        q.low_freq_min_nr_should_match = 0.5;
        one(q.into())
    });
}
