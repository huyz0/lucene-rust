#![allow(clippy::arithmetic_side_effects)]
//! **Numeric metric aggregations, against Lucene-computed expectations.**
//!
//! `fixtures/src/GenMetricAggs.java` writes four segments, three with deletions, and
//! six numeric fields (sparse, multi-valued, `NaN`/`±0`/`±inf`, values that
//! round when widened to `double`), runs seven queries and records, per
//! field, what OpenSearch's `min`/`max`/`sum`/`avg`/`value_count`/`stats`
//! aggregators compute over the live matches: the value count, the
//! `CompensatedSum` value and delta, the minimum and maximum over every value
//! and over each document's first and last. Every bit must match.
//!
//! For the match-all run it also records what `min`/`max` answer from the
//! points (`findLeafMinValue`/`findLeafMaxValue`), which [`Source::PointsMin`]
//! and [`Source::PointsMax`] must reproduce.

use std::collections::HashMap;

use lucene_search::aggs::{metric_states, MetricSpec, Source, ValueKind};
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::query::{MatchAllDocsQuery, PointsRangeQuery};
use lucene_search::{BooleanQuery, Clause, PhraseQuery, TermQuery};
use lucene_store::FsDirectory;

fn fixture_dir() -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/metric_aggs_index"
    )
    .to_string()
}

fn manifest() -> HashMap<String, String> {
    let text = std::fs::read_to_string(format!("{}/manifest.properties", fixture_dir()))
        .expect("run scripts/gen-fixtures.sh --only GenMetricAggs");
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

#[test]
fn metric_aggregations_match_opensearch_bit_for_bit() {
    let m = manifest();
    let reader = DirectoryReader::open(&FsDirectory::open(fixture_dir())).expect("open");
    assert_eq!(reader.segment_readers().len(), 4);
    let mut opened = reader.open_segments().expect("open postings");
    opened.open_points().expect("open points");
    let segments = opened.as_open_segments();
    let fields = [
        ("l", ValueKind::Long),
        ("ml", ValueKind::Long),
        ("d", ValueKind::Double),
        ("md", ValueKind::Double),
        ("f", ValueKind::Float),
        ("i", ValueKind::Long),
        ("e", ValueKind::Double),
    ];
    let specs: Vec<MetricSpec> = fields
        .iter()
        .map(|&(f, kind)| MetricSpec {
            field: f.to_string(),
            kind,
            source: Source::DocValues,
        })
        .collect();
    let runs: usize = m["run_count"].parse().unwrap();
    assert!(runs >= 7);
    let mut failures = Vec::new();
    let mut nonempty = 0;
    for r in 0..runs {
        let text = &m[&format!("run.{r}.query")];
        let got = metric_states(&segments, reader.segment_readers(), &query(text), &specs)
            .unwrap_or_else(|e| panic!("{text}: {e}"));
        for ((field, _), state) in fields.iter().zip(&got) {
            let want = &m[&format!("run.{r}.{field}")];
            let hex = |d: f64| format!("{:x}", d.to_bits());
            let mine = format!(
                "{}:{}:{}:{}:{}:{}:{}",
                state.count,
                hex(state.sum),
                hex(state.delta),
                hex(state.min),
                hex(state.max),
                hex(state.min_of_mins),
                hex(state.max_of_maxes)
            );
            nonempty += usize::from(state.count > 0);
            if &mine != want {
                failures.push(format!("{text} {field}:\n  got    {mine}\n  Lucene {want}"));
            }
        }
    }
    assert!(nonempty > 30, "{nonempty} non-empty field states");
    assert!(failures.is_empty(), "{}", failures.join("\n"));

    // A field asked for twice (`sum` and `avg` of it, say) is read once and
    // reported twice, the same.
    let twice: Vec<MetricSpec> = specs.iter().chain(&specs).cloned().collect();
    let text = &m["run.0.query"];
    let got = metric_states(&segments, reader.segment_readers(), &query(text), &twice).unwrap();
    let (a, b) = got.split_at(specs.len());
    // Debug formatting compares NaN as NaN (PartialEq does not).
    assert_eq!(format!("{a:?}"), format!("{b:?}"));
    let once = metric_states(&segments, reader.segment_readers(), &query(text), &specs).unwrap();
    assert_eq!(format!("{a:?}"), format!("{once:?}"));

    // The points shortcut, over the match-all run: every field once as `min`
    // and once as `max`, in one pass (fields without points read doc values).
    let r = (0..runs)
        .find(|r| m[&format!("run.{r}.query")] == "(all)")
        .expect("a match-all run");
    let bounds: Vec<MetricSpec> = fields
        .iter()
        .flat_map(|&(f, kind)| {
            [Source::PointsMin, Source::PointsMax].map(|source| MetricSpec {
                field: f.to_string(),
                kind,
                source,
            })
        })
        .collect();
    let got = metric_states(
        &segments,
        reader.segment_readers(),
        &query("(all)"),
        &bounds,
    )
    .expect("points bounds");
    for (k, (field, _)) in fields.iter().enumerate() {
        let hex = |d: f64| format!("{:x}", d.to_bits());
        let mine = format!(
            "{}:{}",
            hex(got[2 * k].min_of_mins),
            hex(got[2 * k + 1].max_of_maxes)
        );
        let want = &m[&format!("run.{r}.{field}.points")];
        assert_eq!(&mine, want, "{field}: points bounds");
    }
    // Where the answers differ: `d` holds documents whose one value is NaN,
    // which Math.min over the documents keeps and the points sort last.
    let d = fields.iter().position(|f| f.0 == "d").unwrap();
    assert!(got[2 * d].min_of_mins == f64::NEG_INFINITY);
    assert!(m[&format!("run.{r}.d")].split(':').nth(5) == Some("7ff8000000000000"));
    // And where the points give up: `e`'s lowest points in the fourth segment
    // are all deleted, so that segment is read from doc values, NaN and all.
    let e = fields.iter().position(|f| f.0 == "e").unwrap();
    assert!(got[2 * e].min_of_mins.is_nan());
}
