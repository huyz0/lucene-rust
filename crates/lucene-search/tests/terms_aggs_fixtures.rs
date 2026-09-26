#![allow(clippy::arithmetic_side_effects)]
//! **The `terms` aggregation on keyword fields, against Lucene-computed
//! expectations.**
//!
//! `fixtures/src/GenTermsAggs.java` writes four segments, three with
//! deletions, and five keyword fields (single- and multi-valued `SORTED_SET`,
//! a sparse high-cardinality one, raw bytes with `0x00`/`0xff`, a `SORTED`
//! one), runs seven queries and records, per field and `shard_size`, the
//! shard's `terms` result: the kept buckets by term and `otherDocCount` --
//! over the whole shard, and per slice of a concurrent search over the
//! non-contiguous slices `[[0,2],[1,3]]`. Every bucket and count must match.

use std::collections::HashMap;

use lucene_search::directory_reader::DirectoryReader;
use lucene_search::query::{MatchAllDocsQuery, PointsRangeQuery};
use lucene_search::terms_agg::{terms, terms_sliced, TermsResult};
use lucene_search::{BooleanQuery, Clause, PhraseQuery, TermQuery};
use lucene_store::FsDirectory;

fn fixture_dir() -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/terms_aggs_index"
    )
    .to_string()
}

fn manifest() -> HashMap<String, String> {
    let text = std::fs::read_to_string(format!("{}/manifest.properties", fixture_dir()))
        .expect("run scripts/gen-fixtures.sh --only GenTermsAggs");
    text.lines()
        .filter_map(|l| l.split_once('='))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

fn parse(toks: &[String], at: &mut usize) -> Clause {
    let mut next = || {
        *at += 1;
        toks[*at - 1].clone()
    };
    assert_eq!(next(), "(");
    let op = next();
    let q = match op.as_str() {
        "all" => Clause::MatchAllDocs(MatchAllDocsQuery::new(0)),
        "t" => Clause::Term(TermQuery::new("body", next().into_bytes())),
        "p" => {
            let mut words = Vec::new();
            while toks[*at] != ")" {
                *at += 1;
                words.push(toks[*at - 1].clone());
            }
            Clause::Phrase(PhraseQuery::new("body", words))
        }
        "r" => {
            let min = next().parse().unwrap();
            let max = next().parse().unwrap();
            Clause::PointsRange(PointsRangeQuery::new("r", min, max))
        }
        "b" => {
            let mut b = BooleanQuery::new();
            b.minimum_should_match = next().parse().unwrap();
            while toks[*at] == "(" {
                *at += 1;
                let occur = toks[*at].clone();
                *at += 1;
                let c = parse(toks, at);
                match occur.as_str() {
                    "+" => b.must.push(c),
                    "#" => b.filter.push(c),
                    "?" => b.should.push(c),
                    "-" => b.must_not.push(c),
                    other => panic!("occur {other}"),
                }
                *at += 1;
            }
            Clause::Boolean(Box::new(b))
        }
        other => panic!("op {other}"),
    };
    *at += 1;
    q
}

fn query(text: &str) -> BooleanQuery {
    let toks: Vec<String> = text
        .replace('(', " ( ")
        .replace(')', " ) ")
        .split_whitespace()
        .map(str::to_string)
        .collect();
    match parse(&toks, &mut 0) {
        Clause::Boolean(b) => *b,
        other => {
            let mut b = BooleanQuery::new();
            b.must.push(other);
            b
        }
    }
}

fn record(r: &TermsResult) -> String {
    let buckets: Vec<String> = r
        .buckets
        .iter()
        .map(|(t, n)| {
            let hex: String = t.iter().map(|b| format!("{b:02x}")).collect();
            format!("{hex}:{n}")
        })
        .collect();
    format!("{}|{}", r.other_doc_count, buckets.join(","))
}

#[test]
fn terms_aggregations_match_opensearch() {
    let m = manifest();
    let reader = DirectoryReader::open(&FsDirectory::open(fixture_dir())).expect("open");
    assert_eq!(reader.segment_readers().len(), 4);
    let mut opened = reader.open_segments().expect("open postings");
    opened.open_points().expect("open points");
    let segments = opened.as_open_segments();
    let slices = [vec![0, 2], vec![1, 3]];
    let runs: usize = m["run_count"].parse().unwrap();
    assert!(runs >= 7);
    let mut failures = Vec::new();
    let mut checked = 0;
    let mut pruned = 0;
    for r in 0..runs {
        let text = &m[&format!("run.{r}.query")];
        for field in ["kw", "mkw", "hk", "bk", "sk"] {
            for shard_size in [1usize, 3, 25, 1000] {
                let prefix = format!("run.{r}.{field}.{shard_size}.");
                let whole = terms(
                    &segments,
                    reader.segment_readers(),
                    &query(text),
                    field,
                    shard_size,
                )
                .unwrap_or_else(|e| panic!("{text} {field}: {e}"));
                let sliced = terms_sliced(
                    &segments,
                    reader.segment_readers(),
                    &query(text),
                    field,
                    shard_size,
                    &slices,
                )
                .unwrap_or_else(|e| panic!("{text} {field} sliced: {e}"));
                pruned += usize::from(whole.other_doc_count > 0);
                for (label, got) in std::iter::once(("all".to_string(), &whole))
                    .chain(sliced.iter().enumerate().map(|(i, s)| (i.to_string(), s)))
                {
                    let want = &m[&format!("{prefix}{label}")];
                    let mine = record(got);
                    checked += 1;
                    if &mine != want {
                        failures.push(format!(
                            "{text} {field} shard_size {shard_size} slice {label}:\n  got    {}\n  Lucene {}",
                            &mine[..mine.len().min(300)],
                            &want[..want.len().min(300)]
                        ));
                    }
                }
            }
        }
    }
    assert!(checked > 400, "{checked}");
    assert!(pruned > 50, "runs past shard_size: {pruned}");
    assert!(
        failures.is_empty(),
        "{} of {checked} disagree:\n{}",
        failures.len(),
        failures
            .iter()
            .take(8)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[test]
fn a_field_with_other_doc_values_is_refused() {
    let dir = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/metric_aggs_index"
    );
    let reader = DirectoryReader::open(&FsDirectory::open(dir)).expect("open");
    let opened = reader.open_segments().expect("open postings");
    let segments = opened.as_open_segments();
    let all = query("(all)");
    let err = terms(&segments, reader.segment_readers(), &all, "l", 10).unwrap_err();
    assert!(err.to_string().contains("non-keyword doc values"), "{err}");
    // A field the index does not have counts nothing.
    let none = terms(&segments, reader.segment_readers(), &all, "nosuch", 10).unwrap();
    assert_eq!(none, TermsResult::default());
    // A slice naming a segment the reader lacks.
    assert!(terms_sliced(
        &segments,
        reader.segment_readers(),
        &all,
        "nosuch",
        10,
        &[vec![9]]
    )
    .is_err());
}
