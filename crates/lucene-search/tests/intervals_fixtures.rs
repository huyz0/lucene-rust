//! M10 T10.6, differentially against Lucene 10.5.0:
//! `fixtures/src/GenIntervals.java`.
//!
//! `queries.tsv`: every `Intervals` factory's source over a four-segment
//! index with deletions, a stop word's position holes, offsets, payloads and
//! a pulsed singleton term -- its `toString` and `minExtent`; under five
//! scoring variants (plain, pivot, sigmoid, boosted, a boolean's required
//! clause) every hit's score bits and five documents' explanations; and the
//! `Matches` of every hit and of the five documents, positions, offsets and
//! sub-matches. The file is
//! rebuilt here line for line and compared with Lucene's.

// Test fixtures' own arithmetic -- see `docs/arithmetic-gate.md`'s "Test code".
#![allow(clippy::arithmetic_side_effects)]

use std::collections::HashSet;

use lucene_search::directory_reader::DirectoryReader;
use lucene_search::index_searcher::{IndexSearcher, SegmentNorms};
use lucene_search::intervals::iterators::root_shape_counts;
use lucene_search::intervals::{IntervalQuery, Intervals, IntervalsSource, PayloadFilter};
use lucene_search::matches::{matches, BoxMatches};
use lucene_search::query::{BooleanQuery, BoostQuery, Clause, TermQuery};
use lucene_search::{Error, Result};
use lucene_store::FsDirectory;

fn data() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/intervals")
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

fn filter(name: &str) -> PayloadFilter {
    match name {
        "even" => PayloadFilter::new(|b| b.is_some_and(|b| (b[0] as i8) % 2 == 0)),
        "null" => PayloadFilter::new(|b| b.is_none()),
        "ge1" => PayloadFilter::new(|b| b.is_some_and(|b| (b[0] as i8) >= 1)),
        "neg" => PayloadFilter::new(|b| b.is_some_and(|b| (b[0] as i8) < 0)),
        _ => panic!("{name}"),
    }
}

fn sources(a: &[&str], from: usize) -> Result<Vec<IntervalsSource>> {
    a[from..].iter().map(|s| source(s)).collect()
}

fn int(s: &str) -> i32 {
    s.parse().unwrap()
}

fn bytes(s: &str) -> Option<Vec<u8>> {
    (s != "*").then(|| s.as_bytes().to_vec())
}

/// A bare word is a term.
fn source(spec: &str) -> Result<IntervalsSource> {
    let a = args(spec);
    Ok(match name(spec) {
        "pt" => Intervals::term_with_payload_filter(a[0], filter(a[1])),
        "phrase" => Intervals::phrase(sources(&a, 0)?)?,
        "phraset" => Intervals::phrase_terms(&a)?,
        "or" => Intervals::or(sources(&a, 0)?)?,
        "ornr" => Intervals::or_with_rewrite(false, sources(&a, 0)?)?,
        "ordered" => Intervals::ordered(sources(&a, 0)?),
        "unordered" => Intervals::unordered(sources(&a, 0)?),
        "uno" => Intervals::unordered_no_overlaps(source(a[0])?, source(a[1])?)?,
        "maxgaps" => Intervals::maxgaps(int(a[0]), source(a[1])?)?,
        "maxwidth" => Intervals::maxwidth(int(a[0]), source(a[1])?),
        "extend" => Intervals::extend(source(a[0])?, int(a[1]), int(a[2])),
        "containing" => Intervals::containing(source(a[0])?, source(a[1])?)?,
        "notcontaining" => Intervals::not_containing(source(a[0])?, source(a[1])?)?,
        "containedby" => Intervals::contained_by(source(a[0])?, source(a[1])?)?,
        "notcontainedby" => Intervals::not_contained_by(source(a[0])?, source(a[1])?)?,
        "overlapping" => Intervals::overlapping(source(a[0])?, source(a[1])?),
        "nonoverlapping" => Intervals::non_overlapping(source(a[0])?, source(a[1])?),
        "before" => Intervals::before(source(a[0])?, source(a[1])?)?,
        "after" => Intervals::after(source(a[0])?, source(a[1])?)?,
        "within" => Intervals::within(source(a[0])?, int(a[1]), source(a[2])?)?,
        "notwithin" => Intervals::not_within(source(a[0])?, int(a[1]), source(a[2])?),
        "atleast" => Intervals::at_least(int(a[0]), sources(&a, 1)?),
        "fix" => Intervals::fix_field(a[0], source(a[1])?),
        "prefix" => {
            let max = a.get(1).map_or(128, |m| m.parse().unwrap());
            Intervals::prefix(a[0], max)?
        }
        "wildcard" => Intervals::wildcard(a[0], 128)?,
        "regexp" => Intervals::regexp(a[0], 128)?,
        "range" => Intervals::range(
            bytes(a[0]),
            bytes(a[1]),
            a[2] == "true",
            a[3] == "true",
            128,
        )?,
        "fuzzy" => {
            if a.len() == 2 {
                Intervals::fuzzy_term(a[0], int(a[1]), 0, true, 128)?
            } else {
                Intervals::fuzzy_term(
                    a[0],
                    int(a[1]),
                    int(a[2]),
                    a[3] == "true",
                    a[4].parse().unwrap(),
                )?
            }
        }
        "none" => Intervals::no_intervals(a[0]),
        "analyzed" => {
            let stop: HashSet<String> = ["the".to_string()].into_iter().collect();
            let analyzer = lucene_analysis::Analyzer::standard(Some(&stop));
            lucene_search::intervals::builder::analyzed_text(
                &a[0].replace('_', " "),
                &analyzer,
                "body",
                int(a[1]),
                a[2] == "true",
            )?
        }
        _ => {
            assert!(a.is_empty(), "{spec}");
            Intervals::term(spec)
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

const VARIANTS: [&str; 5] = ["plain", "pivot", "sigmoid", "boost", "bool"];
const EXPLAIN: [i32; 5] = [0, 7, 17, 33, 60];

fn variant(v: &str, field: &str, s: &IntervalsSource) -> BooleanQuery {
    let iq = |q: IntervalQuery| Clause::from(q);
    match v {
        "plain" => as_boolean(iq(IntervalQuery::new(field, s.clone()))),
        "pivot" => as_boolean(iq(IntervalQuery::with_pivot(field, s.clone(), 2.5).unwrap())),
        "sigmoid" => as_boolean(iq(IntervalQuery::with_pivot_and_exp(
            field,
            s.clone(),
            1.5,
            2.0,
        )
        .unwrap())),
        "boost" => as_boolean(Clause::Boost(Box::new(BoostQuery::new(
            iq(IntervalQuery::new(field, s.clone())),
            3.0,
        )))),
        "bool" => BooleanQuery {
            must: vec![iq(IntervalQuery::new(field, s.clone()))],
            should: vec![Clause::Term(TermQuery::new("body", "egg"))],
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
            b.push_str(&format!(
                "{}:{}:{}:{}",
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
                    b.push_str(&format!(
                        "{}:{}:{}:{}",
                        sub.start_position(),
                        sub.end_position(),
                        sub.start_offset(),
                        sub.end_offset()
                    ));
                }
                b.push(']');
            }
        }
    }
    Ok(b)
}

fn run(searcher: &IndexSearcher<'_, '_>, field: &str, spec: &str, out: &mut Vec<String>) {
    let head = format!("{field}\t{spec}");
    let s = match source(spec) {
        Ok(s) => s,
        Err(e) => {
            out.push(format!("{head}\tsource\t{}", err(&e)));
            return;
        }
    };
    out.push(format!(
        "{head}\tsource\t{}\t{}",
        clean(&s.to_string()),
        s.min_extent()
    ));
    let mut hits = Vec::new();
    for v in VARIANTS {
        let q = variant(v, field, &s);
        match searcher.search(&q, 1000) {
            Ok(td) => {
                let mut b = format!("{} ", td.total_hits.value);
                for sd in &td.score_docs {
                    b.push_str(&format!("{}:{},", sd.doc, hex32(sd.score)));
                    if v == "plain" {
                        hits.push(sd.doc);
                    }
                }
                out.push(format!("{head}\t{v}\thits\t{b}"));
            }
            Err(e) => {
                out.push(format!("{head}\t{v}\thits\t{}", err(&e)));
                continue;
            }
        }
        for doc in EXPLAIN {
            let e = match searcher.explain(&q, doc) {
                Ok(e) => clean(&e.to_string()),
                Err(e) => err(&e),
            };
            out.push(format!("{head}\t{v}\texplain {doc}\t{e}"));
        }
    }
    // The hits, and the explained documents whether they match or not.
    hits.extend(EXPLAIN);
    hits.sort_unstable();
    hits.dedup();
    let plain = Clause::from(IntervalQuery::new(field, s.clone()));
    for doc in hits {
        let m = match matches(searcher, &plain, doc).and_then(render) {
            Ok(m) => m,
            Err(e) => err(&e),
        };
        out.push(format!("{head}\tmatches {doc}\t{m}"));
    }
}

#[test]
fn intervals_match_lucene() {
    let dir = data();
    let reader = DirectoryReader::open(&FsDirectory::open(dir.join("index"))).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let owned = reader.field_norms_by_field(&["body".to_string(), "pay".to_string()]);
    let norms: Vec<SegmentNorms<'_, '_>> = owned.iter().map(Some).collect();
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();
    let text = std::fs::read_to_string(dir.join("queries.tsv")).unwrap();
    let want: Vec<&str> = text.lines().collect();
    let mut specs: Vec<(&str, &str)> = Vec::new();
    for line in &want {
        let mut p = line.split('\t');
        let (f, s) = (p.next().unwrap(), p.next().unwrap());
        if specs.last() != Some(&(f, s)) {
            specs.push((f, s));
        }
    }
    let mut got = Vec::new();
    for (f, s) in specs {
        run(&searcher, f, s, &mut got);
    }
    let mut bad = 0;
    for (i, (w, g)) in want.iter().zip(&got).enumerate() {
        if *w != g {
            bad += 1;
            if bad <= 25 {
                eprintln!("line {}:\n  java {w}\n  rust {g}", i + 1);
            }
        }
    }
    assert_eq!(want.len(), got.len(), "line counts");
    assert_eq!(bad, 0, "{bad} of {} lines differ", want.len());
    // Every construction a query's root iterator can take -- each a
    // different monomorphised scorer -- compared with Lucene above at least
    // once. Counted on this thread, so no other test's queries count.
    let unreached: Vec<_> = root_shape_counts()
        .into_iter()
        .filter(|&(_, n)| n == 0)
        .map(|(s, _)| s)
        .collect();
    assert!(
        unreached.is_empty(),
        "no fixture query reaches {unreached:?}"
    );
}
