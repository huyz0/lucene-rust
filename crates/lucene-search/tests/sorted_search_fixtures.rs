#![allow(clippy::arithmetic_side_effects)]
//! **Sorted searches against real Lucene.**
//!
//! `fixtures/src/GenSortedSearch.java` writes a three-segment, 60,000
//! document corpus with deletions and numeric fields indexed as doc values
//! and points, and records `TopFieldCollector`'s answer for every
//! combination of seven queries (one with a phrase), eighteen sorts (numeric types, selectors,
//! directions, missing values, several keys, the score and the document), two
//! page sizes and two total-hits thresholds -- then the next page twice, once
//! after the last hit and once after its values alone (OpenSearch's
//! `search_after`).
//!
//! Each run compares the hits' documents and sort values, and the total with
//! its relation: under a threshold of 100 the numeric comparators skip with
//! the points, and the count Lucene reaches before they do is compared too.

use std::collections::HashMap;

use lucene_search::directory_reader::DirectoryReader;
use lucene_search::field_norms::FieldNorms;
use lucene_search::query::{MatchAllDocsQuery, PointsRangeQuery};
use lucene_search::top_field::{
    search_sorted_sliced, search_sorted_tracking, FieldDoc, Selector, SortField, SortType,
};
use lucene_search::{BooleanQuery, Clause, PhraseQuery, TermQuery};
use lucene_store::FsDirectory;

fn fixture_dir() -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/sorted_search_index"
    )
    .to_string()
}

struct Manifest(HashMap<String, String>);

impl Manifest {
    fn load() -> Self {
        let text = std::fs::read_to_string(format!("{}/manifest.properties", fixture_dir()))
            .expect("run scripts/gen-fixtures.sh --only GenSortedSearch");
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

#[test]
fn sorted_searches_match_real_lucene() {
    let m = Manifest::load();
    let reader = DirectoryReader::open(&FsDirectory::open(fixture_dir())).expect("open reader");
    assert_eq!(reader.segment_readers().len(), 3, "three-segment fixture");
    let mut opened = reader.open_segments().expect("open postings");
    opened.open_points().expect("open points");
    let segments = opened.as_open_segments();
    assert!(segments.iter().any(|s| s.live_docs.is_some()));
    let owned: Vec<HashMap<String, FieldNorms<'_>>> = reader
        .field_norms("body")
        .into_iter()
        .map(|n| n.into_iter().map(|n| ("body".to_string(), n)).collect())
        .collect();
    let norms: Vec<Option<&HashMap<String, FieldNorms<'_>>>> = owned.iter().map(Some).collect();

    let runs: usize = m.get("run_count").parse().unwrap();
    assert!(runs > 1000, "the fixture records every combination");
    let mut failures = Vec::new();
    let mut pruned = 0;
    let mut tracked = 0;
    let mut sliced_runs = 0;
    for r in 0..runs {
        let k = format!("run.{r}");
        let text = m.get(&format!("{k}.query"));
        let spec = m.get(&format!("{k}.sort"));
        let top_n: usize = m.get(&format!("{k}.top_n")).parse().unwrap();
        let threshold: u64 = match m.get(&format!("{k}.threshold")) {
            "max" => u64::MAX,
            n => n.parse().unwrap(),
        };
        let after = m.0.get(&format!("{k}.after")).map(|s| field_doc(s));
        let track = m.0.contains_key(&format!("{k}.track"));
        tracked += usize::from(track);
        let got = search_sorted_tracking(
            &segments,
            reader.segment_readers(),
            &query(text),
            &norms,
            &sort(spec),
            top_n,
            threshold,
            after.as_ref(),
            track,
        )
        .unwrap_or_else(|e| panic!("{text} by {spec}: {e}"));
        let want = hits(m.get(&format!("{k}.hits")));
        let total: u64 = m.get(&format!("{k}.total")).parse().unwrap();
        let gte = m.get(&format!("{k}.relation")) == "gte";
        pruned += usize::from(gte);
        let got_gte =
            got.total.relation == lucene_search::collector::TotalHitsRelation::GreaterThanOrEqualTo;
        // An exact count must match. Past the threshold both engines report
        // a lower bound, and how far past depends on when each starts
        // skipping (Lucene's match-all and filter conjunctions collect whole
        // 4096-document windows first); all that is promised is that it
        // exceeds the threshold.
        let total_ok = if track {
            // Beside a MaxScoreCollector nothing is skipped on either side:
            // the count is the same, bound or not.
            got_gte == gte && got.total.value == total
        } else if gte {
            got_gte && got.total.value > threshold
        } else {
            !got_gte && got.total.value == total
        };
        let max_ok = !track
            || match m.get(&format!("{k}.max_score")) {
                "nan" => got.max_score.is_nan(),
                bits => got.max_score.to_bits() as i32 == bits.parse::<i32>().unwrap(),
            };
        if !max_ok {
            failures.push(format!(
                "run {r}: {text} by {spec}: max score {} ({:#x}), Lucene {}",
                got.max_score,
                got.max_score.to_bits(),
                m.get(&format!("{k}.max_score"))
            ));
        }
        // As a concurrent search runs it: slices of the segments, each its
        // own collector, the hits merged.
        if !track {
            let n = segments.len();
            // Given out of order: a slice's segments are searched by doc base.
            let slices = [(n / 2..n).rev().collect::<Vec<_>>(), (0..n / 2).collect()];
            let sliced = search_sorted_sliced(
                &segments,
                reader.segment_readers(),
                &query(text),
                &norms,
                &sort(spec),
                top_n,
                threshold,
                after.as_ref(),
                &slices,
            )
            .unwrap_or_else(|e| panic!("{text} by {spec}, sliced: {e}"));
            sliced_runs += 1;
            let sliced_gte = sliced.total.relation
                == lucene_search::collector::TotalHitsRelation::GreaterThanOrEqualTo;
            // Each slice counts to its own threshold: past Lucene's, the sum
            // is past it too (exact or not); below it, it is exact.
            let sliced_total_ok = if gte {
                sliced.total.value > threshold
            } else {
                !sliced_gte && sliced.total.value == total
            };
            if sliced.hits != want || !sliced_total_ok {
                failures.push(format!(
                    "run {r}: {text} by {spec}, sliced {slices:?}\n  got    {:?} total {} gte {sliced_gte}\n  Lucene {:?} total {total} gte {gte}",
                    sliced.hits.iter().take(4).collect::<Vec<_>>(),
                    sliced.total.value,
                    want.iter().take(4).collect::<Vec<_>>(),
                ));
            }
        }
        if got.hits != want || !total_ok {
            failures.push(format!(
                "run {r}: {text} by {spec}, top {top_n}, threshold {threshold}, after {after:?}\n  got    {:?} total {} gte {got_gte}\n  Lucene {:?} total {total} gte {gte}",
                got.hits.iter().take(4).collect::<Vec<_>>(),
                got.total.value,
                want.iter().take(4).collect::<Vec<_>>(),
            ));
        }
    }
    assert!(pruned > 100, "the threshold runs must prune: {pruned}");
    assert!(sliced_runs > 1000, "sliced runs: {sliced_runs}");
    assert!(tracked > 200, "tracked max-score runs: {tracked}");
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
