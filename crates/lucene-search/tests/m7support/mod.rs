//! Shared helpers for M7's search-infrastructure fixture tests: manifest
//! loading and the query grammar the Java generators share
//! (`GenMixedBooleanScoring.parse` / `GenSortedSearch.parse`).
#![allow(dead_code, clippy::arithmetic_side_effects)]

use std::collections::HashMap;

use lucene_search::query::{
    ConstantScoreQuery, DisjunctionMaxQuery, MatchAllDocsQuery, PointsRangeQuery, PrefixQuery,
    RegexpQuery, TermInSetQuery, WildcardQuery,
};
use lucene_search::{BooleanQuery, BoostQuery, Clause, PhraseQuery, TermQuery};

pub fn fixture(name: &str) -> String {
    format!("{}/../../fixtures/data/{name}", env!("CARGO_MANIFEST_DIR"))
}

pub struct Manifest(pub HashMap<String, String>);

impl Manifest {
    pub fn load(path: &str) -> Self {
        let text = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("{path}: {e} (run scripts/gen-fixtures.sh)"));
        Manifest(
            text.lines()
                .filter(|l| !l.starts_with('#'))
                .filter_map(|l| l.split_once('='))
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        )
    }

    pub fn get(&self, key: &str) -> &str {
        self.0
            .get(key)
            .unwrap_or_else(|| panic!("manifest key {key} missing"))
    }

    pub fn opt(&self, key: &str) -> Option<&str> {
        self.0.get(key).map(String::as_str)
    }
}

/// `doc:scoreBits,doc:scoreBits,...`.
pub fn scored_hits(s: &str) -> Vec<(i32, u32)> {
    if s.is_empty() {
        return Vec::new();
    }
    s.split(',')
        .map(|h| {
            let (d, b) = h.split_once(':').unwrap();
            (d.parse().unwrap(), b.parse::<i32>().unwrap() as u32)
        })
        .collect()
}

pub struct Grammar {
    /// The text field every term-level op reads.
    pub text: &'static str,
    /// The points field `(r lo hi)` reads.
    pub range: &'static str,
}

impl Grammar {
    fn parse(&self, toks: &[String], at: &mut usize) -> Clause {
        let next = |at: &mut usize| {
            *at += 1;
            toks[*at - 1].clone()
        };
        assert_eq!(next(at), "(");
        let op = next(at);
        let field = self.text;
        let q = match op.as_str() {
            "all" => Clause::MatchAllDocs(MatchAllDocsQuery::new(0)),
            "t" => Clause::Term(TermQuery::new(field, next(at).into_bytes())),
            "tf" => {
                let f = next(at);
                Clause::Term(TermQuery::new(f, next(at).into_bytes()))
            }
            "pf" => {
                let f = next(at);
                let slop = next(at).parse().unwrap();
                let mut words = Vec::new();
                while toks[*at] != ")" {
                    words.push(next(at));
                }
                Clause::Phrase(PhraseQuery::new(f, words).with_slop(slop))
            }
            "pre" => {
                let p = if toks[*at] == ")" {
                    String::new()
                } else {
                    next(at)
                };
                Clause::Prefix(PrefixQuery::new(field, p.into_bytes()))
            }
            "none" => Clause::MatchNoDocs(lucene_search::query::MatchNoDocsQuery::new()),
            "exists" => Clause::Exists(lucene_search::query::FieldExistsQuery::new(next(at))),
            "fz" => {
                let term = next(at);
                let edits: u8 = next(at).parse().unwrap();
                let mut q = lucene_search::FuzzyQuery::new(field, term);
                q.max_edits = edits;
                Clause::Fuzzy(q)
            }
            "mp" => {
                let mut positions = Vec::new();
                while toks[*at] != ")" {
                    positions.push(
                        next(at)
                            .split('|')
                            .map(|s| s.as_bytes().to_vec())
                            .collect::<Vec<_>>(),
                    );
                }
                Clause::MultiPhrase(lucene_search::MultiPhraseQuery::new(field, positions))
            }
            "st" | "snear" | "sor" => {
                *at -= 2;
                let s = self.span(toks, at);
                return Clause::Span(s);
            }
            "wc" => Clause::Wildcard(WildcardQuery::new(field, next(at).into_bytes())),
            "re" => Clause::Regexp(RegexpQuery::new(field, next(at))),
            "ts" => {
                let mut terms = Vec::new();
                while toks[*at] != ")" {
                    terms.push(next(at).into_bytes());
                }
                Clause::TermInSet(TermInSetQuery::new(field, terms))
            }
            "p" | "ps" => {
                let slop = if op == "ps" {
                    next(at).parse().unwrap()
                } else {
                    0
                };
                let mut words = Vec::new();
                while toks[*at] != ")" {
                    words.push(next(at));
                }
                Clause::Phrase(PhraseQuery::new(field, words).with_slop(slop))
            }
            "r" => {
                let min = next(at).parse().unwrap();
                let max = next(at).parse().unwrap();
                Clause::PointsRange(PointsRangeQuery::new(self.range, min, max))
            }
            "boost" => {
                let f: f32 = next(at).parse().unwrap();
                let inner = self.parse(toks, at);
                Clause::Boost(Box::new(BoostQuery::new(inner, f)))
            }
            "const" => {
                let inner = self.parse(toks, at);
                Clause::ConstantScore(Box::new(ConstantScoreQuery::new(inner, 1.0)))
            }
            "dismax" => {
                let tie: f32 = next(at).parse().unwrap();
                let mut ds = Vec::new();
                while toks[*at] == "(" {
                    ds.push(self.parse(toks, at));
                }
                Clause::DisjunctionMax(Box::new(DisjunctionMaxQuery::new(ds, tie)))
            }
            "b" => {
                let mut b = BooleanQuery::new();
                b.minimum_should_match = next(at).parse().unwrap();
                while toks[*at] == "(" {
                    *at += 1;
                    let occur = next(at);
                    let c = self.parse(toks, at);
                    match occur.as_str() {
                        "+" => b.must.push(c),
                        "#" => b.filter.push(c),
                        "?" => b.should.push(c),
                        "-" => b.must_not.push(c),
                        other => panic!("occur {other}"),
                    }
                    assert_eq!(next(at), ")");
                }
                Clause::Boolean(Box::new(b))
            }
            other => panic!("op {other}"),
        };
        assert_eq!(next(at), ")");
        q
    }

    /// A span query: `(st t)`, `(snear slop in_order ...)`, `(sor ...)`.
    fn span(&self, toks: &[String], at: &mut usize) -> lucene_search::SpanQuery {
        use lucene_search::SpanQuery;
        let next = |at: &mut usize| {
            *at += 1;
            toks[*at - 1].clone()
        };
        assert_eq!(next(at), "(");
        let op = next(at);
        let q = match op.as_str() {
            "st" => SpanQuery::span_term(self.text, next(at).into_bytes()),
            "snear" => {
                let slop = next(at).parse().unwrap();
                let in_order = next(at) == "true";
                let mut cs = Vec::new();
                while toks[*at] == "(" {
                    cs.push(self.span(toks, at));
                }
                SpanQuery::span_near(cs, slop, in_order)
            }
            "sor" => {
                let mut cs = Vec::new();
                while toks[*at] == "(" {
                    cs.push(self.span(toks, at));
                }
                SpanQuery::span_or(cs)
            }
            other => panic!("span op {other}"),
        };
        assert_eq!(next(at), ")");
        q
    }

    pub fn clause(&self, text: &str) -> Clause {
        let toks: Vec<String> = text
            .replace('(', " ( ")
            .replace(')', " ) ")
            .split_whitespace()
            .map(str::to_string)
            .collect();
        let mut at = 0;
        let c = self.parse(&toks, &mut at);
        assert_eq!(at, toks.len(), "trailing tokens in {text}");
        c
    }

    /// The query as the boolean the search functions take: a boolean
    /// itself, else a lone `MUST` clause (`BooleanQuery.rewrite` of one
    /// clause is that clause).
    pub fn query(&self, text: &str) -> BooleanQuery {
        match self.clause(text) {
            Clause::Boolean(b) => *b,
            other => {
                let mut b = BooleanQuery::new();
                b.must.push(other);
                b
            }
        }
    }
}
