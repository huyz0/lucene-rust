#![allow(clippy::arithmetic_side_effects)]
//! **`terminate_after` against real Lucene.**
//!
//! `fixtures/src/GenTerminateAfter.java` writes four segments of 3,000
//! documents, two with deletions, and records a sequential search behind
//! OpenSearch's `EarlyTerminatingCollector` (reproduced there) beside Lucene's
//! own top-docs collectors: for nine queries -- two matching only in the
//! first two segments, one matching nothing -- `n` over fixed values and each
//! query's match count and its neighbours, five sorts (the unsorted
//! `TopScoreDocCollector` among them), with and without `track_scores`, and
//! the `size: 0` count (`TotalHitCountCollector`, which takes a segment's
//! `Weight.count` whole).
//!
//! Each run compares the hits (documents, sort values, score bits), the total,
//! the documents let through and whether the search ended early.

use std::collections::HashMap;

use lucene_search::directory_reader::DirectoryReader;
use lucene_search::field_norms::FieldNorms;
use lucene_search::query::{MatchAllDocsQuery, PointsRangeQuery};
use lucene_search::terminate::{count_until, search_sorted_until, terminate_after};
use lucene_search::top_field::{FieldDoc, Selector, SortField, SortType};
use lucene_search::{BooleanQuery, Clause, PhraseQuery, TermQuery};
use lucene_store::FsDirectory;

fn fixture_dir() -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/terminate_after_index"
    )
    .to_string()
}

struct Manifest(HashMap<String, String>);

impl Manifest {
    fn load() -> Self {
        let text = std::fs::read_to_string(format!("{}/manifest.properties", fixture_dir()))
            .expect("run scripts/gen-fixtures.sh --only GenTerminateAfter");
        Manifest(
            text.lines()
                .filter_map(|l| l.split_once('='))
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        )
    }

    fn get(&self, key: &str) -> &str {
        self.0
            .get(key)
            .unwrap_or_else(|| panic!("manifest key {key} missing"))
    }
}

fn field_doc(s: &str) -> FieldDoc {
    let mut parts = s.split(':');
    let doc = parts.next().unwrap().parse().unwrap();
    FieldDoc {
        doc,
        values: parts.map(|v| v.parse().unwrap()).collect(),
        terms: Vec::new(),
    }
}

fn hits(s: &str) -> Vec<FieldDoc> {
    if s.is_empty() {
        return Vec::new();
    }
    s.split(',').map(field_doc).collect()
}

fn sort(spec: &str) -> Vec<SortField> {
    spec.split(',')
        .map(|k| {
            let p: Vec<&str> = k.split(':').collect();
            SortField {
                field: p[0].to_string(),
                ty: match p[1] {
                    "score" => SortType::Score,
                    "doc" => SortType::Doc,
                    "long" => SortType::Long,
                    "int" => SortType::Int,
                    "double" => SortType::Double,
                    "float" => SortType::Float,
                    other => panic!("type {other}"),
                },
                selector: if p[2] == "max" {
                    Selector::Max
                } else {
                    Selector::Min
                },
                reverse: p[3] == "true",
                missing: p[4].parse().unwrap(),
            }
        })
        .collect()
}

/// GenMixedBooleanScoring's grammar, as far as these queries use it.
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
                assert_eq!(toks[*at], ")");
                *at += 1;
            }
            Clause::Boolean(Box::new(b))
        }
        other => panic!("op {other}"),
    };
    assert_eq!(toks[*at], ")");
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

/// The queries whose `Weight.count` the port has (a lone term, match-all).
fn countable(text: &str) -> bool {
    text == "(all)" || text.starts_with("(t ")
}

/// Whether Lucene collects this search document by document -- the searches
/// the plugin runs natively: the top-docs collector needs scores (so the
/// search runs `COMPLETE`), and the query is not one whose bulk scorer hands
/// out ranges (a constant-score query, a conjunction of filters).
fn per_document(text: &str, spec: &str, track: bool) -> bool {
    (spec.contains("score") || track)
        && text != "(all)"
        && text != "(b 0 (# (t w0)) (# (r 0 5999)))"
}

#[test]
fn terminate_after_matches_real_lucene() {
    let m = Manifest::load();
    let reader = DirectoryReader::open(&FsDirectory::open(fixture_dir())).expect("open reader");
    assert_eq!(reader.segment_readers().len(), 4, "four-segment fixture");
    let mut opened = reader.open_segments().expect("open postings");
    opened.open_points().expect("open points");
    let segments = opened.as_open_segments();
    assert!(segments.iter().any(|s| s.live_docs.is_some()));
    assert!(segments.iter().any(|s| s.live_docs.is_none()));
    let owned: Vec<HashMap<String, FieldNorms<'_>>> = reader
        .field_norms("body")
        .into_iter()
        .map(|n| n.into_iter().map(|n| ("body".to_string(), n)).collect())
        .collect();
    let norms: Vec<Option<&HashMap<String, FieldNorms<'_>>>> = owned.iter().map(Some).collect();

    let runs: usize = m.get("run_count").parse().unwrap();
    assert!(runs > 500, "the fixture records every combination");
    let mut failures = Vec::new();
    let (mut terminated, mut not_terminated, mut counted, mut tracked, mut ranged) =
        (0, 0, 0, 0, 0);
    for r in 0..runs {
        let k = format!("run.{r}");
        let text = m.get(&format!("{k}.query"));
        let q = query(text);
        let n: u64 = m.get(&format!("{k}.n")).parse().unwrap();
        let total: u64 = m.get(&format!("{k}.total")).parse().unwrap();
        let collected: u64 = m.get(&format!("{k}.collected")).parse().unwrap();
        let want_terminated = m.get(&format!("{k}.terminated")) == "true";
        if want_terminated {
            terminated += 1;
        } else {
            not_terminated += 1;
        }
        let spec = m.get(&format!("{k}.sort"));
        if spec == "count" {
            let cut = terminate_after(&segments, &q, n).unwrap();
            let got = count_until(&segments, &q, &cut).unwrap();
            if countable(text) {
                counted += 1;
                if got != Some(total)
                    || cut.collected != collected
                    || cut.terminated != want_terminated
                {
                    failures.push(format!(
                        "run {r}: {text} n={n} count: got {got:?} collected {} terminated {}, Lucene {total} {collected} {want_terminated}",
                        cut.collected, cut.terminated
                    ));
                }
            } else if got.is_some() {
                failures.push(format!(
                    "run {r}: {text}: a segment count for a query without one"
                ));
            }
            continue;
        }
        let top_n: usize = m.get(&format!("{k}.top_n")).parse().unwrap();
        let track = m.0.contains_key(&format!("{k}.track"));
        if !per_document(text, spec, track) {
            // Lucene hands this search's matches out in collectRange batches
            // (DenseConjunctionBulkScorer), and the top-docs collector loses
            // the batch the terminating collector throws in: the plugin runs
            // it on Lucene. Recorded all the same, to show the difference is
            // real.
            ranged += 1;
            continue;
        }
        tracked += usize::from(track);
        let (got, cut) = search_sorted_until(
            &segments,
            reader.segment_readers(),
            &q,
            &norms,
            &sort(spec),
            top_n,
            None,
            track,
            n,
        )
        .unwrap_or_else(|e| panic!("{text} by {spec}: {e}"));
        let want = hits(m.get(&format!("{k}.hits")));
        assert_eq!(
            m.get(&format!("{k}.relation")),
            "eq",
            "run {r}: every let-through document is counted"
        );
        let max_ok = !track
            || match m.get(&format!("{k}.max_score")) {
                "nan" => got.max_score.is_nan(),
                bits => got.max_score.to_bits() as i32 == bits.parse::<i32>().unwrap(),
            };
        if got.hits != want
            || cut.collected != total
            || cut.collected != collected
            || cut.terminated != want_terminated
            || !max_ok
        {
            failures.push(format!(
                "run {r}: {text} by {spec}, n={n}, track {track}\n  got    {:?} collected {} terminated {} max {}\n  Lucene {:?} total {total} collected {collected} terminated {want_terminated}",
                got.hits.iter().take(4).collect::<Vec<_>>(),
                cut.collected,
                cut.terminated,
                got.max_score,
                want.iter().take(4).collect::<Vec<_>>(),
            ));
        }
    }
    assert!(
        terminated > 100 && not_terminated > 50,
        "both outcomes: {terminated} / {not_terminated}"
    );
    assert!(counted > 20, "segment-counted runs: {counted}");
    assert!(tracked > 50, "tracked max-score runs: {tracked}");
    assert!(ranged > 50, "runs left to Lucene: {ranged}");
    assert!(
        failures.is_empty(),
        "{} of {runs} runs disagree with Lucene:\n{}",
        failures.len(),
        failures
            .iter()
            .take(20)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}

/// A concurrent `size: 0` count's `terminated_early`: per slice, OpenSearch's
/// non-forced `EarlyTerminatingCollector` around Lucene's
/// `TotalHitCountCollector`, the segments it iterates being those whose
/// `Weight.count` is `-1` (recorded from Lucene's own weight; the plugin asks
/// the same weight at run time).
#[test]
fn concurrent_count_termination_matches_real_lucene() {
    let m = Manifest::load();
    let reader = DirectoryReader::open(&FsDirectory::open(fixture_dir())).expect("open reader");
    let mut opened = reader.open_segments().expect("open postings");
    opened.open_points().expect("open points");
    let segments = opened.as_open_segments();
    let counts: usize = m.get("count_count").parse().unwrap();
    assert!(counts > 200, "count records: {counts}");
    let (mut terminated, mut failures) = (0, Vec::new());
    for c in 0..counts {
        let k = format!("count.{c}");
        let text = m.get(&format!("{k}.query"));
        let n: u64 = m.get(&format!("{k}.n")).parse().unwrap();
        let iterate: Vec<bool> = m
            .get(&format!("{k}.iterate"))
            .chars()
            .map(|ch| ch == '1')
            .collect();
        let slices: Vec<Vec<usize>> = m
            .get(&format!("{k}.slices"))
            .split('|')
            .map(|s| s.split(',').map(|i| i.parse().unwrap()).collect())
            .collect();
        let want = m.get(&format!("{k}.terminated")) == "true";
        terminated += usize::from(want);
        let got = lucene_search::terminate::count_terminates(
            &segments,
            &query(text),
            &slices,
            &iterate,
            n,
        )
        .unwrap();
        if got != want {
            failures.push(format!("count {c}: {text} n={n} slices {slices:?} iterate {iterate:?}: got {got}, Lucene {want}"));
        }
    }
    assert!(
        terminated > 50 && terminated < counts - 50,
        "both outcomes: {terminated} of {counts}"
    );
    assert!(
        failures.is_empty(),
        "{} of {counts} disagree:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
