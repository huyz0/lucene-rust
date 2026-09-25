#![allow(clippy::arithmetic_side_effects)]
//! **Mixed boolean queries, pruned and exhaustive, against real Lucene.**
//!
//! `fixtures/src/GenMixedBooleanScoring.java` writes a two-segment, 24,000
//! document Zipf corpus with deletions, and records Lucene's top hits for
//! sixty-eight queries that mix every `Occur`, `minimum_should_match`, boosts,
//! `constant_score`, dismax and nesting -- the shapes the scorer tree in
//! `lucene-search`'s `exec` module runs. Each query is recorded twice: with a
//! total-hits threshold of 100, so the collector publishes a minimum
//! competitive score almost at once and every block-max path (`WANDScorer`,
//! `BlockMaxConjunctionScorer`, `ReqOptSumScorer`'s impacts approximation,
//! `ImpactsDISI`, `ConstantScoreScorer`'s emptying) prunes; and with an exact
//! count, which prunes nothing.
//!
//! Hits compare on doc id **and raw `f32` score bits**. The queries are
//! written in the generator's S-expression grammar, parsed here into
//! [`Clause`]s, so the two sides cannot drift apart on what was asked.

use std::collections::HashMap;

use lucene_search::directory_reader::DirectoryReader;
use lucene_search::field_norms::FieldNorms;
use lucene_search::multi_segment::search_boolean_query_multi_segment_maxscore_counting;
use lucene_search::query::{BoostQuery, ConstantScoreQuery, DisjunctionMaxQuery};
use lucene_search::{BooleanQuery, Clause, TermQuery};
use lucene_store::FsDirectory;

fn fixture_dir() -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/mixed_boolean_scoring_index"
    )
    .to_string()
}

struct Manifest(HashMap<String, String>);

impl Manifest {
    fn load() -> Self {
        let text = std::fs::read_to_string(format!("{}/manifest.properties", fixture_dir()))
            .expect("run scripts/gen-fixtures.sh --only GenMixedBooleanScoring");
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

    fn hits(&self, key: &str) -> Vec<(i32, u32)> {
        let raw = self.get(key);
        if raw.is_empty() {
            return Vec::new();
        }
        raw.split(',')
            .map(|pair| {
                let (doc, bits) = pair.split_once(':').expect("doc:bits");
                (doc.parse().unwrap(), bits.parse().unwrap())
            })
            .collect()
    }
}

/// The generator's grammar; see `GenMixedBooleanScoring`'s class doc.
struct Tokens {
    toks: Vec<String>,
    at: usize,
}

impl Tokens {
    fn new(s: &str) -> Self {
        Tokens {
            toks: s
                .replace('(', " ( ")
                .replace(')', " ) ")
                .split_whitespace()
                .map(str::to_string)
                .collect(),
            at: 0,
        }
    }

    fn next(&mut self) -> String {
        self.at += 1;
        self.toks[self.at - 1].clone()
    }

    fn peek(&self) -> &str {
        &self.toks[self.at]
    }

    fn expect(&mut self, t: &str) {
        assert_eq!(self.next(), t);
    }
}

fn parse(t: &mut Tokens) -> Clause {
    t.expect("(");
    let op = t.next();
    let q = match op.as_str() {
        "t" => Clause::Term(TermQuery::new("body", t.next().into_bytes())),
        "boost" => {
            let f: f32 = t.next().parse().unwrap();
            BoostQuery::new(parse(t), f).into()
        }
        "const" => ConstantScoreQuery::new(parse(t), 1.0).into(),
        "dismax" => {
            let tie: f32 = t.next().parse().unwrap();
            let mut ds = Vec::new();
            while t.peek() == "(" {
                ds.push(parse(t));
            }
            DisjunctionMaxQuery::new(ds, tie).into()
        }
        "b" => {
            let mut b = BooleanQuery::new();
            b.minimum_should_match = t.next().parse().unwrap();
            while t.peek() == "(" {
                t.expect("(");
                let occur = t.next();
                let c = parse(t);
                match occur.as_str() {
                    "+" => b.must.push(c),
                    "#" => b.filter.push(c),
                    "?" => b.should.push(c),
                    "-" => b.must_not.push(c),
                    other => panic!("bad occur {other}"),
                }
                t.expect(")");
            }
            Clause::Boolean(Box::new(b))
        }
        other => panic!("unknown op {other}"),
    };
    t.expect(")");
    q
}

/// A top-level query as the search entry point takes it: a boolean as is,
/// anything else as the lone `MUST` clause of one (which Lucene rewrites
/// back to the clause itself).
fn root(clause: Clause) -> BooleanQuery {
    match clause {
        Clause::Boolean(b) => *b,
        other => {
            let mut b = BooleanQuery::new();
            b.must.push(other);
            b
        }
    }
}

#[test]
fn mixed_boolean_queries_match_real_lucene_pruned_and_exact() {
    let m = Manifest::load();
    let reader = DirectoryReader::open(&FsDirectory::open(fixture_dir())).expect("open reader");
    assert_eq!(reader.segment_readers().len(), 2, "two-segment fixture");
    let opened = reader.open_segments().expect("open postings");
    let segments = opened.as_open_segments();
    assert!(
        segments.iter().any(|s| s.live_docs.is_some()),
        "the fixture deletes documents, so live docs are in play"
    );
    let owned: Vec<HashMap<String, FieldNorms<'_>>> = reader
        .field_norms("body")
        .into_iter()
        .map(|n| n.into_iter().map(|n| ("body".to_string(), n)).collect())
        .collect();
    let norms: Vec<Option<&HashMap<String, FieldNorms<'_>>>> = owned.iter().map(Some).collect();

    let top_n: usize = m.get("top_n").parse().unwrap();
    let threshold: u64 = m.get("threshold").parse().unwrap();
    let count: usize = m.get("query_count").parse().unwrap();
    let mut failures = Vec::new();
    for i in 0..count {
        let text = m.get(&format!("query.{i}"));
        let query = root(parse(&mut Tokens::new(text)));
        for (mode, limit) in [("pruned", threshold), ("exact", u64::MAX)] {
            let (hits, total) = search_boolean_query_multi_segment_maxscore_counting(
                &segments, &query, &norms, top_n, limit,
            )
            .unwrap_or_else(|e| panic!("{text}: {e}"));
            let got: Vec<(i32, u32)> = hits.iter().map(|h| (h.doc_id, h.score.to_bits())).collect();
            let expected = m.hits(&format!("query.{i}.{mode}"));
            if got != expected {
                failures.push(format!(
                    "{text} [{mode}]\n  got    {:?}\n  Lucene {:?}",
                    decode(&got),
                    decode(&expected)
                ));
            }
            if mode == "exact" {
                let lucene_total: u64 = m.get(&format!("query.{i}.total")).parse().unwrap();
                if total.value != lucene_total {
                    failures.push(format!(
                        "{text}: total {} vs Lucene {lucene_total}",
                        total.value
                    ));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} runs disagree with Lucene:\n{}",
        failures.len(),
        count * 2,
        failures.join("\n")
    );
}

fn decode(hits: &[(i32, u32)]) -> Vec<(i32, f32)> {
    hits.iter().map(|&(d, b)| (d, f32::from_bits(b))).collect()
}
