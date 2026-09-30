#![allow(clippy::arithmetic_side_effects)]
//! **`Query.visit` and `QueryVisitor` against real Lucene.**
//!
//! `fixtures/src/GenQueryVisitor.java` visits 31 queries (not rewritten):
//! with `QueryVisitor.termCollector`, and with a tracing visitor that logs
//! each call -- `consumeTerms` with its terms, `consumeTermsMatching` with
//! the matcher's answers over a list of probe terms, `visitLeaf`,
//! `getSubVisitor` with its occurrence and parent -- at its depth, once
//! accepting every field and once only `body`. Every query kind this port has
//! is covered: terms, phrases, multi-phrases, prefixes (empty too), wildcards
//! (literal, escaped, match-all), regexps (literal, empty language), term
//! sets of one and several terms, fuzzy, spans, match-all/none, points,
//! exists, and booleans (all four occurrences, nested prohibited clauses),
//! dis-max, constant score and boost around them.

mod m7support;

use std::cell::RefCell;
use std::rc::Rc;

use lucene_search::query::SpanQuery;
use lucene_search::query_visitor::{
    extract_terms, visit, Occur, QueryVisitor, SubVisitor, Term, TermMatcher,
};
use lucene_search::Clause;
use m7support::{fixture, Grammar, Manifest};

const GRAMMAR: Grammar = Grammar {
    text: "body",
    range: "r",
};

/// The Java class a clause is.
fn name(c: &Clause) -> &'static str {
    match c {
        Clause::Span(SpanQuery::SpanTerm { .. }) => "SpanTermQuery",
        Clause::Span(SpanQuery::SpanNear { .. }) => "SpanNearQuery",
        Clause::Span(SpanQuery::SpanOr { .. }) => "SpanOrQuery",
        // `LongPoint.newRangeQuery` is an anonymous subclass: no simple name.
        Clause::PointsRange(_) => "",
        other => lucene_search::matches::clause_name(other),
    }
}

fn occur(o: Occur) -> &'static str {
    match o {
        Occur::Must => "+",
        Occur::Filter => "#",
        Occur::Should => "",
        Occur::MustNot => "-",
    }
}

fn terms_str(terms: &[Term]) -> String {
    terms
        .iter()
        .map(|t| format!("{}:{}", t.field, String::from_utf8_lossy(&t.bytes)))
        .collect::<Vec<_>>()
        .join(",")
}

/// `GenQueryVisitor.Trace`.
struct Trace {
    depth: usize,
    log: Rc<RefCell<Vec<String>>>,
    only: Option<&'static str>,
    probes: Rc<Vec<String>>,
}

impl QueryVisitor for Trace {
    fn consume_terms(&mut self, query: &Clause, terms: &[Term]) {
        self.log.borrow_mut().push(format!(
            "d{} terms {} {}",
            self.depth,
            name(query),
            terms_str(terms)
        ));
    }
    fn consume_terms_matching(&mut self, query: &Clause, field: &str, matcher: &dyn TermMatcher) {
        let answers: String = self
            .probes
            .iter()
            .map(|p| {
                if matcher.matches(p.as_bytes()) {
                    '1'
                } else {
                    '0'
                }
            })
            .collect();
        self.log.borrow_mut().push(format!(
            "d{} match {} {field} {answers}",
            self.depth,
            name(query)
        ));
    }
    fn visit_leaf(&mut self, query: &Clause) {
        self.log
            .borrow_mut()
            .push(format!("d{} leaf {}", self.depth, name(query)));
    }
    fn accept_field(&self, field: &str) -> bool {
        self.only.is_none_or(|o| o == field)
    }
    fn sub_visitor(&mut self, o: Occur, parent: &Clause) -> SubVisitor {
        self.log
            .borrow_mut()
            .push(format!("d{} sub {} {}", self.depth, occur(o), name(parent)));
        if o == Occur::MustNot {
            return SubVisitor::Empty;
        }
        SubVisitor::Other(Box::new(Trace {
            depth: self.depth + 1,
            log: Rc::clone(&self.log),
            only: self.only,
            probes: Rc::clone(&self.probes),
        }))
    }
}

#[test]
fn query_visitor_matches_real_lucene() {
    let m = Manifest::load(&fixture("query_visitor/cases.txt"));
    let probes: Rc<Vec<String>> = Rc::new(m.get("probes").split(' ').map(str::to_string).collect());
    let n: usize = m.get("query_count").parse().unwrap();
    let mut failures = Vec::new();
    for i in 0..n {
        let text = m.get(&format!("q.{i}"));
        let clause = GRAMMAR.clause(text);
        let terms: Vec<String> = extract_terms(&clause)
            .iter()
            .map(|t| format!("{}:{}", t.field, String::from_utf8_lossy(&t.bytes)))
            .collect();
        let want_terms = m.get(&format!("terms.{i}"));
        if terms.join(",") != want_terms {
            failures.push(format!(
                "{text}: terms {:?} vs {want_terms}",
                terms.join(",")
            ));
        }
        for (key, only) in [("trace", None), ("trace_body", Some("body"))] {
            let log = Rc::new(RefCell::new(Vec::new()));
            let mut t = Trace {
                depth: 0,
                log: Rc::clone(&log),
                only,
                probes: Rc::clone(&probes),
            };
            visit(&clause, &mut t);
            let mut got = log.borrow().clone();
            got.sort();
            let got = got.join(" | ");
            let want = m.get(&format!("{key}.{i}"));
            if got != want {
                failures.push(format!("{text} ({key}):\n  got  {got}\n  want {want}"));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} disagreements:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
