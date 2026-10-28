#![allow(clippy::arithmetic_side_effects)]
//! **Mixed boolean queries, pruned and exhaustive, against real Lucene.**
//!
//! `fixtures/src/GenMixedBooleanScoring.java` writes a two-segment, 24,000
//! document Zipf corpus with deletions, and records Lucene's top hits for
//! ninety-one queries that mix every `Occur`, `minimum_should_match`, boosts,
//! `constant_score`, dismax and nesting -- the shapes the scorer tree in
//! `lucene-search`'s `exec` module runs. Each query is recorded twice: with a
//! total-hits threshold of 100, so the collector publishes a minimum
//! competitive score almost at once and every block-max path (`WANDScorer`,
//! `BlockMaxConjunctionScorer`, `ReqOptSumScorer`'s impacts approximation,
//! `ImpactsDISI`, `ConstantScoreScorer`'s emptying) prunes; and with an exact
//! count, which prunes nothing. `AppendPrunedTotalsManifest` adds the
//! pruned run's lower-bound total, and `IndexSearcher.search(query, 10)`'s
//! total and relation (its default threshold of 1000): the collector counts
//! every document the scorer tree hands it, so these totals pin down exactly
//! which documents each block-max path skips, which top hits cannot.
//!
//! Hits compare on doc id **and raw `f32` score bits**. The queries are
//! written in the generator's S-expression grammar, parsed here into
//! [`Clause`]s, so the two sides cannot drift apart on what was asked.

use std::collections::HashMap;

use lucene_search::collector::TotalHitsRelation;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::field_norms::FieldNorms;
use lucene_search::multi_segment::search_boolean_query_multi_segment_maxscore_counting;
use lucene_search::query::{
    BoostQuery, ConstantScoreQuery, DisjunctionMaxQuery, PointsRangeQuery, PrefixQuery,
    RegexpQuery, TermInSetQuery, WildcardQuery,
};
use lucene_search::{BooleanQuery, Clause, PhraseQuery, TermQuery};
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
        "r" => {
            let min: i64 = t.next().parse().unwrap();
            let max: i64 = t.next().parse().unwrap();
            Clause::PointsRange(PointsRangeQuery::new("n", min, max))
        }
        "pre" => Clause::Prefix(PrefixQuery::new("body", t.next().into_bytes())),
        "wc" => Clause::Wildcard(WildcardQuery::new("body", t.next().into_bytes())),
        "re" => Clause::Regexp(RegexpQuery::new("body", t.next())),
        "ts" => {
            let mut terms = Vec::new();
            while t.peek() != ")" {
                terms.push(t.next().into_bytes());
            }
            Clause::TermInSet(TermInSetQuery::new("body", terms))
        }
        "p" | "ps" => {
            let slop: u32 = if op == "ps" {
                t.next().parse().unwrap()
            } else {
                0
            };
            let mut words = Vec::new();
            while t.peek() != ")" {
                words.push(t.next());
            }
            Clause::Phrase(PhraseQuery::new("body", words).with_slop(slop))
        }
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
    let mut opened = reader.open_segments().expect("open postings");
    opened.open_points().expect("open points");
    let segments = opened.as_open_segments();
    assert!(
        segments.iter().any(|s| s.live_docs.is_some()),
        "the fixture deletes documents, so live docs are in play"
    );
    let owned: Vec<HashMap<String, FieldNorms<'_>>> = reader
        .field_norms("body")
        .unwrap()
        .into_iter()
        .map(|n| n.into_iter().map(|n| ("body".to_string(), n)).collect())
        .collect();
    let norms: Vec<Option<&HashMap<String, FieldNorms<'_>>>> = owned.iter().map(Some).collect();

    let top_n: usize = m.get("top_n").parse().unwrap();
    let threshold: u64 = m.get("threshold").parse().unwrap();
    let count: usize = m.get("query_count").parse().unwrap();
    let mut failures = Vec::new();
    // `IndexSearcher.search` rewrites first, and the lower-bound totals
    // depend on the bulk scorer the rewritten query gets; the free search
    // functions score the query as given, as the JVM hands it over.
    let mut runs: Vec<(String, String, BooleanQuery)> = (0..count)
        .map(|i| {
            let text = m.get(&format!("query.{i}")).to_string();
            let query = root(parse(&mut Tokens::new(&text)).rewrite());
            (format!("query.{i}"), text, query)
        })
        .collect();
    // `+a ?(b c)`, not rewritten: `BooleanQuery.rewrite` flattens the nested
    // disjunction, so Lucene's answer is the one it recorded for `+a ?b ?c`
    // -- which the free functions must give for the nested query as handed
    // over (`Child::TermSum`, the same `DisjunctionSumScorer`).
    let nested = nested_disjunction_runs(&m, count);
    assert_eq!(nested.len(), 3, "the fixture has three `+a ?b ?c`");
    runs.extend(nested);
    // `AppendPresentClausesManifest`: booleans written with several scoring
    // clauses of which one is present in the segment. `setTopLevelScoringClause`
    // counts the present ones, so that one is built top-level.
    let present: usize = m.get("present_count").parse().unwrap();
    assert!(present > 0, "the fixture has present-clause queries");
    runs.extend((0..present).map(|i| {
        let text = m.get(&format!("present.{i}")).to_string();
        let query = root(parse(&mut Tokens::new(&text)).rewrite());
        (format!("present.{i}"), text, query)
    }));
    // `AppendFilteredMaxScoreManifest`: filtered disjunctions whose filter is
    // sparse, so `MaxScoreBulkScorer` collects the filter into a bit set per
    // window (`fillScoreBufferViaBitSet`) instead of leap-frogging.
    let filtered: usize = m.get("filtered_ms_count").parse().unwrap();
    assert!(filtered > 0, "the fixture has filtered MaxScore queries");
    runs.extend((0..filtered).map(|i| {
        let text = m.get(&format!("filtered_ms.{i}")).to_string();
        let query = root(parse(&mut Tokens::new(&text)).rewrite());
        (format!("filtered_ms.{i}"), text, query)
    }));
    for (key, text, query) in &runs {
        let text = text.as_str();
        for (mode, limit) in [
            ("pruned", threshold),
            ("searcher", 1000),
            ("exact", u64::MAX),
        ] {
            let (hits, total) = search_boolean_query_multi_segment_maxscore_counting(
                &segments, query, &norms, top_n, limit,
            )
            .unwrap_or_else(|e| panic!("{text}: {e}"));
            let got: Vec<(i32, u32)> = hits.iter().map(|h| (h.doc_id, h.score.to_bits())).collect();
            let expected = match mode {
                // `search(query, 10)` returns the same top ten as the pruned
                // run: both collect exactly the ten best.
                "searcher" => m.hits(&format!("{key}.pruned")),
                _ => m.hits(&format!("{key}.{mode}")),
            };
            if got != expected {
                failures.push(format!(
                    "{text} [{mode}]\n  got    {:?}\n  Lucene {:?}",
                    decode(&got),
                    decode(&expected)
                ));
            }
            let (total_key, relation) = match mode {
                "pruned" => (
                    format!("{key}.pruned.total"),
                    m.get(&format!("{key}.pruned.relation")),
                ),
                "searcher" => (
                    format!("{key}.searcher.total"),
                    m.get(&format!("{key}.searcher.relation")),
                ),
                _ => (format!("{key}.total"), "eq"),
            };
            let lucene_total: u64 = m.get(&total_key).parse().unwrap();
            let got_relation = match total.relation {
                TotalHitsRelation::EqualTo => "eq",
                TotalHitsRelation::GreaterThanOrEqualTo => "gte",
            };
            if total.value != lucene_total || got_relation != relation {
                failures.push(format!(
                    "{text} [{mode}]: total {} {got_relation} vs Lucene {lucene_total} {relation}",
                    total.value
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} runs disagree with Lucene:\n{}",
        failures.len(),
        runs.len() * 3,
        failures.join("\n")
    );
}

/// Each manifest query `(b 0 (+ a) (? b) (? c))` of three terms, as
/// `(b 0 (+ a) (? (b 0 (? b) (? c))))` -- not rewritten -- with the index of
/// the flat query whose recorded answers it must give.
///
/// Not `#a ?(b c)`: there the nested boolean is the only scoring clause as
/// written, so `setTopLevelScoringClause` makes it a `WANDScorer`, where
/// Lucene, which only ever scores it flattened, has two scoring clauses and a
/// `DisjunctionSumScorer` -- the pruned totals differ (177 vs 10,343).
fn nested_disjunction_runs(m: &Manifest, count: usize) -> Vec<(String, String, BooleanQuery)> {
    let mut out = Vec::new();
    for i in 0..count {
        let text = m.get(&format!("query.{i}"));
        let Clause::Boolean(b) = parse(&mut Tokens::new(text)) else {
            continue;
        };
        let terms = |cs: &[Clause]| cs.iter().all(|c| matches!(c, Clause::Term(_)));
        let shape = b.minimum_should_match == 0
            && b.must.len() == 1
            && b.filter.is_empty()
            && b.should.len() == 2
            && b.must_not.is_empty()
            && terms(&b.must)
            && terms(&b.should);
        if !shape {
            continue;
        }
        let mut inner = BooleanQuery::new();
        inner.should = b.should.clone();
        let mut outer = BooleanQuery::new();
        outer.must = b.must.clone();
        outer.should.push(Clause::Boolean(Box::new(inner)));
        out.push((
            format!("query.{i}"),
            format!("{text} as nested ?(b c)"),
            outer,
        ));
    }
    out
}

fn decode(hits: &[(i32, u32)]) -> Vec<(i32, f32)> {
    hits.iter().map(|&(d, b)| (d, f32::from_bits(b))).collect()
}
