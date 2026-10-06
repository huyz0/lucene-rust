//! M10 T10.6, differentially against Lucene 10.5.0:
//! `fixtures/src/GenSpans.java`.
//!
//! `queries.tsv`: the span queries of `lucene-queries`' `spans` package
//! (term, near, or, first, position range, not, containing, within, field
//! masking, the multi-term wrapper) and the `payloads` package's
//! (`SpanPayloadCheckQuery` over one-byte, int, float and string payloads
//! under each match operation; `PayloadScoreQuery` with each payload
//! function, with and without the span score) over a four-segment index with
//! deletions, a stop word's position holes and a pulsed singleton term --
//! each query's `toString`; under five variants (plain, boosted, a
//! boolean's required clause, its filter, its prohibited clause) every hit's
//! score bits and five documents' explanations; then `Weight.matches` of the hits and the
//! explained documents -- each span's positions and offsets and the query
//! it reports, its terms as
//! sub-matches with their `TermQuery`. The file is rebuilt here line for
//! line and compared with Lucene's.
//!
//! One deliberate difference is checked rather than skipped: Java's
//! `SpanPayloadCheckQuery.toString` throws on a `null` payload
//! (`Term.toString(null)`), so its explanations do too; this port prints
//! `null`. Those lines are compared as "Java threw, and the query has a
//! `null` payload".

// Test fixtures' own arithmetic -- see `docs/arithmetic-gate.md`'s "Test code".
#![allow(clippy::arithmetic_side_effects)]

use lucene_search::directory_reader::DirectoryReader;
use lucene_search::extended_query::{
    ExtendedQuery, MultiTermQuery, MultiTermSource, RewriteMethod, TermRangeQuery,
};
use lucene_search::index_searcher::{IndexSearcher, SegmentNorms};
use lucene_search::matches::{matches, BoxMatches};
use lucene_search::query::{
    BooleanQuery, BoostQuery, Clause, PrefixQuery, RegexpQuery, TermQuery, WildcardQuery,
};
use lucene_search::spans::payloads::{
    MatchOperation, PayloadDecoder, PayloadFunction, PayloadScoreQuery, PayloadType,
    SpanPayloadCheckQuery,
};
use lucene_search::spans::SpanNode;
use lucene_search::{Error, Result};
use lucene_store::FsDirectory;

fn data() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/spans")
}

// ---------------------------------------------------------------------------
// The spec grammar
// ---------------------------------------------------------------------------

fn split(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let (mut depth, mut start) = (0i32, 0usize);
    for (i, c) in s.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => depth -= 1,
            ',' if depth == 0 => {
                out.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    if start < s.len() || !out.is_empty() {
        out.push(&s[start..]);
    }
    out
}

fn name(spec: &str) -> &str {
    spec.split_once('(').map_or(spec, |(n, _)| n)
}

fn args(spec: &str) -> Vec<&str> {
    match spec.find('(') {
        None => Vec::new(),
        Some(p) => split(&spec[p + 1..spec.len() - 1]),
    }
}

fn int(s: &str) -> i32 {
    s.parse().unwrap()
}

fn queries(a: &[&str], from: usize) -> Result<Vec<SpanNode>> {
    a[from..].iter().map(|s| query(s)).collect()
}

fn function(name: &str) -> PayloadFunction {
    match name {
        "min" => PayloadFunction::Min,
        "max" => PayloadFunction::Max,
        "avg" => PayloadFunction::Average,
        "sum" => PayloadFunction::Sum,
        _ => panic!("{name}"),
    }
}

/// A payload to match, as its type encodes it: `None` for `null`.
fn payload_arg(kind: &str, v: &str) -> Option<Vec<u8>> {
    if v == "null" {
        return None;
    }
    Some(match kind {
        "INT" => v.parse::<i32>().unwrap().to_be_bytes().to_vec(),
        "FLOAT" => v.parse::<f32>().unwrap().to_be_bytes().to_vec(),
        "BYTE" => vec![v.parse::<i8>().unwrap() as u8],
        _ => v.replace("e'", "\u{e9}").into_bytes(),
    })
}

fn multi(source: MultiTermSource, top: Option<usize>) -> SpanNode {
    let rewrite = match top {
        Some(n) => RewriteMethod::TopTermsScoringBoolean(n),
        None => RewriteMethod::default(),
    };
    SpanNode::multi_term(MultiTermQuery::new(source, rewrite))
}

/// A bare word is a term of `body`; `t(field,word)` one of another field.
fn query(spec: &str) -> Result<SpanNode> {
    let a = args(spec);
    Ok(match name(spec) {
        "t" => SpanNode::term(a[0], a[1]),
        "near" => SpanNode::near(queries(&a, 2)?, int(a[0]), a[1] == "true")?,
        "or" => SpanNode::or(queries(&a, 0)?)?,
        "first" => SpanNode::first(query(a[1])?, int(a[0])),
        "range" => SpanNode::position_range(query(a[2])?, int(a[0]), int(a[1])),
        "not" => SpanNode::not(query(a[0])?, query(a[1])?, int(a[2]), int(a[3]))?,
        "containing" => SpanNode::containing(query(a[0])?, query(a[1])?)?,
        "within" => SpanNode::within(query(a[0])?, query(a[1])?)?,
        "mask" => SpanNode::field_masking(query(a[1])?, a[0]),
        "prefix" => multi(MultiTermSource::Prefix(PrefixQuery::new(a[0], a[1])), None),
        "wildcard" => multi(
            MultiTermSource::Wildcard(WildcardQuery::new(a[0], a[1])),
            None,
        ),
        "regexp" => multi(MultiTermSource::Regexp(RegexpQuery::new(a[0], a[1])), None),
        "trange" => multi(
            MultiTermSource::TermRange(TermRangeQuery::new(
                a[0],
                Some(a[1].as_bytes().to_vec()),
                Some(a[2].as_bytes().to_vec()),
                a[3] == "true",
                a[4] == "true",
            )),
            None,
        ),
        "topprefix" => multi(
            MultiTermSource::Prefix(PrefixQuery::new(a[1], a[2])),
            Some(a[0].parse().unwrap()),
        ),
        "check" => {
            let pays = a[3].split('|').map(|v| payload_arg(a[0], v)).collect();
            let kind = match a[0] {
                "INT" => PayloadType::Int,
                "FLOAT" => PayloadType::Float,
                _ => PayloadType::String,
            };
            let op = match a[1] {
                "EQ" => MatchOperation::Eq,
                "GT" => MatchOperation::Gt,
                "GTE" => MatchOperation::Gte,
                "LT" => MatchOperation::Lt,
                _ => MatchOperation::Lte,
            };
            SpanNode::PayloadCheck(Box::new(SpanPayloadCheckQuery::with(
                query(a[2])?,
                pays,
                kind,
                op,
            )))
        }
        "pscore" => SpanNode::PayloadScore(Box::new(PayloadScoreQuery::new(
            query(a[2])?,
            function(a[0]),
            PayloadDecoder::Float,
            a[1] == "true",
        ))),
        _ => {
            assert!(a.is_empty(), "{spec}");
            SpanNode::term("body", spec)
        }
    })
}

// ---------------------------------------------------------------------------
// queries.tsv
// ---------------------------------------------------------------------------

fn hex32(f: f32) -> String {
    format!("{:x}", f.to_bits())
}

fn err(e: &Error) -> String {
    match e {
        Error::Unsupported(_) => "!UnsupportedOperationException".into(),
        Error::IllegalArgument(_) => "!IllegalArgumentException".into(),
        Error::IllegalState(_) => "!IllegalStateException".into(),
        other => format!("!{other:?}"),
    }
}

fn clean(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('\t', "\\t")
        .replace('\n', "\\n")
}

const VARIANTS: [&str; 5] = ["plain", "boost", "bool", "filter", "mustnot"];
const EXPLAIN: [i32; 5] = [0, 7, 17, 33, 60];

fn variant(v: &str, s: &SpanNode) -> BooleanQuery {
    match v {
        "plain" => as_boolean(Clause::from(s.clone())),
        "boost" => as_boolean(Clause::Boost(Box::new(BoostQuery::new(
            Clause::from(s.clone()),
            2.5,
        )))),
        "bool" => BooleanQuery {
            must: vec![Clause::from(s.clone())],
            should: vec![Clause::Term(TermQuery::new("body", "egg"))],
            ..Default::default()
        },
        "filter" => BooleanQuery {
            filter: vec![Clause::from(s.clone())],
            should: vec![Clause::Term(TermQuery::new("body", "egg"))],
            ..Default::default()
        },
        "mustnot" => BooleanQuery {
            should: vec![Clause::Term(TermQuery::new("body", "egg"))],
            must_not: vec![Clause::from(s.clone())],
            ..Default::default()
        },
        _ => panic!("{v}"),
    }
}

fn as_boolean(c: Clause) -> BooleanQuery {
    BooleanQuery {
        must: vec![c],
        ..Default::default()
    }
}

fn run(searcher: &IndexSearcher<'_, '_>, spec: &str, out: &mut Vec<String>) {
    let s = match query(spec) {
        Ok(s) => s,
        Err(e) => {
            out.push(format!("{spec}\tquery\t{}", err(&e)));
            return;
        }
    };
    out.push(format!("{spec}\tquery\t{}", clean(&s.to_string())));
    for v in VARIANTS {
        let q = variant(v, &s);
        match searcher.search(&q, 1000) {
            Ok(td) => {
                let mut b = format!("{} ", td.total_hits.value);
                for sd in &td.score_docs {
                    b.push_str(&format!("{}:{},", sd.doc, hex32(sd.score)));
                }
                out.push(format!("{spec}\t{v}\thits\t{b}"));
            }
            Err(e) => {
                out.push(format!("{spec}\t{v}\thits\t{}", err(&e)));
                continue;
            }
        }
        for doc in EXPLAIN {
            let e = match searcher.explain(&q, doc) {
                Ok(e) => clean(&e.to_string()),
                Err(e) => err(&e),
            };
            out.push(format!("{spec}\t{v}\texplain {doc}\t{e}"));
        }
    }
    matches_lines(searcher, spec, &s, out);
}

/// `GenSpans.render`: each field's matches with the query they report, a
/// sub-match with its term.
fn render(m: Option<BoxMatches>) -> Result<String> {
    let Some(m) = m else {
        return Ok("none".into());
    };
    let mut b = String::new();
    for f in m.fields() {
        if !b.is_empty() {
            b.push(';');
        }
        b.push_str(&f);
        b.push('|');
        let mut it = m.get_matches(&f)?;
        let mut first = true;
        while let Some(i) = it.as_mut() {
            if !i.next()? {
                break;
            }
            if !first {
                b.push(',');
            }
            first = false;
            // `MatchesIterator.getQuery()`: the span weight's query.
            let Clause::Extended(q) = i.query() else {
                panic!("a span match reports its span query");
            };
            let ExtendedQuery::Span(q) = q.as_ref() else {
                panic!("a span match reports its span query");
            };
            b.push_str(&format!(
                "{}:{}:{}:{}@{q}",
                i.start_position(),
                i.end_position(),
                i.start_offset(),
                i.end_offset()
            ));
            if let Some(mut sub) = i.sub_matches()? {
                b.push('[');
                let mut f2 = true;
                while sub.next()? {
                    if !f2 {
                        b.push(' ');
                    }
                    f2 = false;
                    let Clause::Term(t) = sub.query() else {
                        panic!("a span's sub-match is a term's");
                    };
                    b.push_str(&format!(
                        "{}:{}:{}:{}={}:{}",
                        sub.start_position(),
                        sub.end_position(),
                        sub.start_offset(),
                        sub.end_offset(),
                        t.field,
                        String::from_utf8_lossy(&t.term)
                    ));
                }
                b.push(']');
            }
        }
    }
    Ok(b)
}

/// `Weight.matches` of the plain query's hits and the explained documents,
/// over the query `IndexSearcher.rewrite` gives.
fn matches_lines(
    searcher: &IndexSearcher<'_, '_>,
    spec: &str,
    s: &SpanNode,
    out: &mut Vec<String>,
) {
    let mut docs: std::collections::BTreeSet<i32> = EXPLAIN.into_iter().collect();
    if let Ok(td) = searcher.search(&variant("plain", s), 1000) {
        docs.extend(td.score_docs.iter().map(|sd| sd.doc));
    }
    let rewritten = s.rewrite(searcher).map(Clause::from);
    for doc in docs {
        let m = rewritten.as_ref().map_err(err).and_then(|q| {
            matches(searcher, q, doc)
                .and_then(render)
                .map_err(|e| err(&e))
        });
        let m = m.unwrap_or_else(|e| e);
        out.push(format!("{spec}\tmatches {doc}\t{m}"));
    }
}

/// Java threw `Term.toString(null)` where this port printed the `null`
/// payload: the line is Java's exception, and ours names the query, whose
/// payloads include `null`.
fn null_payload_divergence(java: &str, rust: &str) -> bool {
    let (Some((jhead, jtail)), Some((rhead, rtail))) =
        (java.rsplit_once('\t'), rust.rsplit_once('\t'))
    else {
        return false;
    };
    jhead == rhead
        && jtail == "!NullPointerException"
        && jhead.starts_with("check(")
        && jhead
            .split('\t')
            .next()
            .is_some_and(|spec| spec.contains("null"))
        && rtail.contains("payloadRef: ")
}

#[test]
fn spans_and_payload_queries_match_lucene() {
    let dir = data();
    let reader = DirectoryReader::open(&FsDirectory::open(dir.join("index"))).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let fields: Vec<String> = ["body", "body2", "pay", "ipay", "fpay", "spay"]
        .iter()
        .map(|f| f.to_string())
        .collect();
    let owned = reader.field_norms_by_field(&fields);
    let norms: Vec<SegmentNorms<'_, '_>> = owned.iter().map(Some).collect();
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();
    let text = std::fs::read_to_string(dir.join("queries.tsv")).unwrap();
    let want: Vec<&str> = text.lines().collect();
    let mut specs: Vec<&str> = Vec::new();
    for line in &want {
        let spec = line.split('\t').next().unwrap();
        if specs.last() != Some(&spec) {
            specs.push(spec);
        }
    }
    let mut got = Vec::new();
    for spec in &specs {
        run(&searcher, spec, &mut got);
    }
    let (mut bad, mut diverged) = (0, 0);
    for (i, (w, g)) in want.iter().zip(&got).enumerate() {
        if *w == g {
            continue;
        }
        if null_payload_divergence(w, g) {
            diverged += 1;
            continue;
        }
        bad += 1;
        if bad <= 20 {
            eprintln!("line {}:\n  java {w}\n  rust {g}", i + 1);
        }
    }
    assert_eq!(want.len(), got.len(), "line counts");
    assert_eq!(bad, 0, "{bad} of {} lines differ", want.len());
    // Every one of Java's `toString` exceptions, and nothing else, is that.
    let thrown = want
        .iter()
        .filter(|l| l.ends_with("\t!NullPointerException"))
        .count();
    assert_eq!(diverged, thrown, "null-payload lines");
}
