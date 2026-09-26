#![allow(clippy::arithmetic_side_effects, dead_code)]
//! **`min_score` against real Lucene.**
//!
//! `fixtures/src/GenMinScore.java` writes three segments of 3,000 documents,
//! two with deletions, and records OpenSearch's `MinimumScoreCollector`
//! (reproduced there) around Lucene's own `TopScoreDocCollector`: for nine
//! queries (a match-all, terms, a conjunction with an exclusion, a
//! disjunction, a filtered term, a phrase beside a term, a
//! `minimum_should_match`, nothing), minimums at fractions of each query's top
//! score and at its fifth hit's score exactly, under two total-hits
//! thresholds -- and, scored `COMPLETE`, the passing documents' count and the
//! sum of their `v` (each document's id).
//!
//! Each run compares the top hits (documents and score bits), the total and
//! its relation, the `size: 0` count, and the aggregations' view: a
//! `value_count` and a `sum` of `v` over the passing documents.

use std::collections::HashMap;

use lucene_search::aggs::{
    aggregate_sliced_min_score, MetricSpec, MinScore, Source, ValueKind, NEED_ALL,
};
use lucene_search::collector::TotalHitsRelation;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::field_norms::FieldNorms;
use lucene_search::multi_segment::{
    count_boolean_query_min_score, search_boolean_query_multi_segment_min_score,
};
use lucene_search::query::{MatchAllDocsQuery, PointsRangeQuery};
use lucene_search::top_field::{FieldDoc, Selector, SortField, SortType};
use lucene_search::{BooleanQuery, Clause, PhraseQuery, TermQuery};
use lucene_store::FsDirectory;

fn fixture_dir() -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/min_score_index"
    )
    .to_string()
}

struct Manifest(HashMap<String, String>);

impl Manifest {
    fn load() -> Self {
        let text = std::fs::read_to_string(format!("{}/manifest.properties", fixture_dir()))
            .expect("run scripts/gen-fixtures.sh --only GenMinScore");
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
fn min_score_matches_real_lucene() {
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
    let specs = [MetricSpec {
        field: "v".to_string(),
        kind: ValueKind::Long,
        source: Source::DocValues,
        needs: NEED_ALL,
    }];
    let all: Vec<usize> = (0..segments.len()).collect();

    let runs: usize = m.get("run_count").parse().unwrap();
    assert!(runs > 100, "the fixture records every combination");
    let (mut failures, mut pruned, mut none_pass) = (Vec::new(), 0, 0);
    for r in 0..runs {
        let k = format!("run.{r}");
        let text = m.get(&format!("{k}.query"));
        let q = query(text);
        let min = f32::from_bits(m.get(&format!("{k}.min")).parse::<i32>().unwrap() as u32);
        let threshold: u64 = match m.get(&format!("{k}.threshold")) {
            "max" => u64::MAX,
            n => n.parse().unwrap(),
        };
        let want: Vec<(i32, u32)> = match m.get(&format!("{k}.hits")) {
            "" => Vec::new(),
            s => s
                .split(',')
                .map(|h| {
                    let (d, b) = h.split_once(':').unwrap();
                    (d.parse().unwrap(), b.parse::<i32>().unwrap() as u32)
                })
                .collect(),
        };
        let total: u64 = m.get(&format!("{k}.total")).parse().unwrap();
        let gte = m.get(&format!("{k}.relation")) == "gte";
        pruned += usize::from(gte);
        let passing: u64 = m.get(&format!("{k}.passing")).parse().unwrap();
        let v_sum: f64 = m.get(&format!("{k}.passing_v_sum")).parse().unwrap();
        none_pass += usize::from(passing == 0);

        let (hits, got_total) =
            search_boolean_query_multi_segment_min_score(&segments, &q, &norms, 10, threshold, min)
                .unwrap();
        let got: Vec<(i32, u32)> = hits.iter().map(|h| (h.doc_id, h.score.to_bits())).collect();
        let got_gte = got_total.relation == TotalHitsRelation::GreaterThanOrEqualTo;
        let total_ok = if gte {
            got_gte && got_total.value > threshold
        } else {
            !got_gte && got_total.value == total
        };
        if got != want || !total_ok {
            failures.push(format!(
                "run {r}: {text} min {min} threshold {threshold}: got {:?} total {} gte {got_gte}, Lucene {:?} total {total} gte {gte}",
                got.iter().take(3).collect::<Vec<_>>(),
                got_total.value,
                want.iter().take(3).collect::<Vec<_>>()
            ));
        }
        let (count, lower) =
            count_boolean_query_min_score(&segments, &q, &norms, min, u64::MAX).unwrap();
        if count != passing || lower {
            failures.push(format!(
                "run {r}: {text} min {min}: count {count}, Lucene {passing}"
            ));
        }
        let min_score = MinScore { min, norms: &norms };
        let states = aggregate_sliced_min_score(
            &segments,
            reader.segment_readers(),
            &q,
            &specs,
            &[],
            &[],
            &[all.clone()],
            Some(&min_score),
        )
        .unwrap();
        let s = &states[0].0[0];
        if s.count != passing || s.sum + s.delta != v_sum {
            failures.push(format!(
                "run {r}: {text} min {min}: aggregated {} docs summing {}, Lucene {passing} summing {v_sum}",
                s.count,
                s.sum + s.delta
            ));
        }
    }
    assert!(pruned > 5, "threshold runs that prune: {pruned}");
    assert!(none_pass > 5, "runs nothing passes: {none_pass}");
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
