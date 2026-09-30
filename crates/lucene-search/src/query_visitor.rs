//! `QueryVisitor` and `Query.visit`: walking a query tree for its terms and
//! leaves, with per-field filtering and per-clause sub-visitors
//! (`QueryVisitor.termCollector`, `EMPTY_VISITOR`).
//!
//! A port of Lucene 10.5.0's `QueryVisitor` and of the `visit` method of every
//! query this port has as a [`Clause`]: terms report their terms
//! (`consumeTerms`), multi-term queries a matcher over the terms they accept
//! (`consumeTermsMatching`, or `consumeTerms` of the one term a literal
//! pattern accepts, as `CompiledAutomaton.visit` decides from the automaton's
//! type), and everything else is a leaf (`visitLeaf`); `BooleanQuery`,
//! `DisjunctionMaxQuery`, `ConstantScoreQuery`, `BoostQuery`, `PhraseQuery`,
//! `MultiPhraseQuery` and span queries hand their clauses a sub-visitor for
//! the clause's occurrence.
//!
//! Sub-visitors own their state (`SubVisitor::Other` is `'static`): a
//! sub-visitor that reports to its parent shares state through `Rc`/`Arc`,
//! as a Java sub-visitor captures its outer instance.
//!
//! Verified against Lucene by `tests/query_visitor_fixtures.rs`
//! (`fixtures/src/GenQueryVisitor.java`).

use std::collections::BTreeSet;

use lucene_util::automaton::{self, AutomatonType, CompiledAutomaton};

use crate::query::{Clause, SpanQuery};

/// `BooleanClause.Occur`, as a sub-visitor is asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Occur {
    Must,
    Filter,
    Should,
    MustNot,
}

/// `Term`: a field and its bytes.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Term {
    pub field: String,
    pub bytes: Vec<u8>,
}

impl Term {
    pub fn new(field: impl Into<String>, bytes: impl Into<Vec<u8>>) -> Self {
        Self {
            field: field.into(),
            bytes: bytes.into(),
        }
    }
}

/// `consumeTermsMatching`'s `Supplier<ByteRunAutomaton>`: whether a term is
/// one the query accepts.
pub trait TermMatcher {
    fn matches(&self, term: &[u8]) -> bool;
}

impl<F: Fn(&[u8]) -> bool> TermMatcher for F {
    fn matches(&self, term: &[u8]) -> bool {
        self(term)
    }
}

/// What `getSubVisitor` returns.
pub enum SubVisitor {
    /// `return this`.
    Same,
    /// `QueryVisitor.EMPTY_VISITOR`.
    Empty,
    /// Another visitor.
    Other(Box<dyn QueryVisitor>),
}

/// `QueryVisitor`.
pub trait QueryVisitor {
    /// `consumeTerms(query, terms)`.
    fn consume_terms(&mut self, _query: &Clause, _terms: &[Term]) {}
    /// `consumeTermsMatching(query, field, automaton)`: by default a leaf.
    fn consume_terms_matching(&mut self, query: &Clause, _field: &str, _matcher: &dyn TermMatcher) {
        self.visit_leaf(query);
    }
    /// `visitLeaf(query)`.
    fn visit_leaf(&mut self, _query: &Clause) {}
    /// `acceptField(field)`.
    fn accept_field(&self, _field: &str) -> bool {
        true
    }
    /// `getSubVisitor(occur, parent)`: by default this visitor, except for a
    /// `MUST_NOT` clause, whose terms are not the query's.
    fn sub_visitor(&mut self, occur: Occur, _parent: &Clause) -> SubVisitor {
        if occur == Occur::MustNot {
            SubVisitor::Empty
        } else {
            SubVisitor::Same
        }
    }
}

/// `QueryVisitor.EMPTY_VISITOR`: accepts no field.
#[derive(Debug, Default, Clone, Copy)]
pub struct EmptyVisitor;

impl QueryVisitor for EmptyVisitor {
    fn accept_field(&self, _field: &str) -> bool {
        false
    }
}

/// `QueryVisitor.termCollector(termSet)`: every term the query's non-negated
/// clauses name.
#[derive(Debug, Default, Clone)]
pub struct TermCollector {
    pub terms: BTreeSet<Term>,
}

impl QueryVisitor for TermCollector {
    fn consume_terms(&mut self, _query: &Clause, terms: &[Term]) {
        self.terms.extend(terms.iter().cloned());
    }
}

/// Runs `f` with the visitor `getSubVisitor(occur, parent)` gives.
fn with_sub(
    visitor: &mut dyn QueryVisitor,
    occur: Occur,
    parent: &Clause,
    f: &mut dyn FnMut(&mut dyn QueryVisitor),
) {
    match visitor.sub_visitor(occur, parent) {
        SubVisitor::Same => f(visitor),
        SubVisitor::Empty => f(&mut EmptyVisitor),
        SubVisitor::Other(mut v) => f(v.as_mut()),
    }
}

/// `CompiledAutomaton.visit`: by the automaton's type, nothing, one term, or
/// a matcher; an automaton that cannot be built is a leaf.
fn visit_automaton(
    visitor: &mut dyn QueryVisitor,
    query: &Clause,
    field: &str,
    compiled: Option<CompiledAutomaton>,
) {
    let Some(c) = compiled else {
        visitor.visit_leaf(query);
        return;
    };
    visit_compiled(&c, visitor, query, field);
}

/// `CompiledAutomaton.visit(visitor, parent, field)`: when the visitor
/// accepts `field`, by the automaton's type nothing (`NONE`), the one term
/// (`SINGLE`, `consumeTerms`), or a matcher over the terms it accepts
/// (`ALL`, `NORMAL`: `consumeTermsMatching`). The terms enumeration half,
/// `getTermsEnum`, is `lucene_codecs::blocktree::FieldTerms::compiled_terms`.
pub fn visit_compiled(
    c: &CompiledAutomaton,
    visitor: &mut dyn QueryVisitor,
    query: &Clause,
    field: &str,
) {
    if !visitor.accept_field(field) {
        return;
    }
    match c.automaton_type {
        AutomatonType::NONE => {}
        AutomatonType::SINGLE => {
            let term = c.term.clone().unwrap_or_default();
            visitor.consume_terms(query, &[Term::new(field, term)]);
        }
        AutomatonType::ALL => visitor.consume_terms_matching(query, field, &|_: &[u8]| true),
        AutomatonType::NORMAL => {
            let run = |t: &[u8]| c.get_byte_runnable().is_some_and(|r| r.run(t));
            visitor.consume_terms_matching(query, field, &run);
        }
    }
}

/// `PrefixQuery.toAutomaton(prefix)`: the prefix's bytes, then any bytes.
fn prefix_automaton(prefix: &[u8]) -> automaton::Automaton {
    let mut a = automaton::Automaton::new();
    let mut last = a.create_state();
    for &b in prefix {
        let s = a.create_state();
        a.add_transition(last, s, i32::from(b), i32::from(b));
        last = s;
    }
    a.set_accept(last, true);
    a.add_transition(last, last, 0, 255);
    a.finish_state();
    a
}

/// `WildcardQuery.toAutomaton(term, workLimit)`.
fn wildcard_automaton(pattern: &[u8]) -> Option<automaton::Automaton> {
    let text = String::from_utf8_lossy(pattern);
    let chars: Vec<char> = text.chars().collect();
    let mut parts = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        match c {
            '*' => parts.push(automaton::automata::make_any_string()),
            '?' => parts.push(automaton::automata::make_any_char()),
            '\\' if i + 1 < chars.len() => {
                i += 1;
                parts.push(automaton::automata::make_char(chars[i] as i32));
            }
            _ => parts.push(automaton::automata::make_char(c as i32)),
        }
        i += 1;
    }
    let refs: Vec<&automaton::Automaton> = parts.iter().collect();
    automaton::operations::determinize(
        &automaton::operations::concatenate(&refs),
        automaton::DEFAULT_DETERMINIZE_WORK_LIMIT,
    )
    .ok()
}

/// `Query.visit(visitor)` for `clause`.
pub fn visit(clause: &Clause, visitor: &mut dyn QueryVisitor) {
    match clause {
        Clause::Term(t) => {
            if visitor.accept_field(&t.field) {
                visitor.consume_terms(clause, &[Term::new(t.field.clone(), t.term.clone())]);
            }
        }
        Clause::Phrase(p) => {
            if !visitor.accept_field(&p.field) {
                return;
            }
            let terms: Vec<Term> = p
                .terms
                .iter()
                .map(|t| Term::new(p.field.clone(), t.clone()))
                .collect();
            with_sub(visitor, Occur::Must, clause, &mut |v| {
                v.consume_terms(clause, &terms)
            });
        }
        Clause::MultiPhrase(p) => {
            if !visitor.accept_field(&p.field) {
                return;
            }
            with_sub(visitor, Occur::Must, clause, &mut |v| {
                for position in &p.term_arrays {
                    let terms: Vec<Term> = position
                        .iter()
                        .map(|t| Term::new(p.field.clone(), t.clone()))
                        .collect();
                    with_sub(v, Occur::Should, clause, &mut |sv| {
                        sv.consume_terms(clause, &terms)
                    });
                }
            });
        }
        Clause::Boolean(b) => {
            // `clauseSets` in `Occur` order: MUST, FILTER, SHOULD, MUST_NOT.
            with_sub(visitor, Occur::Must, clause, &mut |sub| {
                for c in &b.must {
                    visit(c, sub);
                }
                for (occur, clauses) in [
                    (Occur::Filter, &b.filter),
                    (Occur::Should, &b.should),
                    (Occur::MustNot, &b.must_not),
                ] {
                    if clauses.is_empty() {
                        continue;
                    }
                    with_sub(sub, occur, clause, &mut |v| {
                        for c in clauses {
                            visit(c, v);
                        }
                    });
                }
            });
        }
        Clause::DisjunctionMax(d) => {
            with_sub(visitor, Occur::Should, clause, &mut |v| {
                for c in &d.disjuncts {
                    visit(c, v);
                }
            });
        }
        Clause::ConstantScore(c) => {
            with_sub(visitor, Occur::Filter, clause, &mut |v| visit(&c.inner, v));
        }
        Clause::Boost(b) => {
            with_sub(visitor, Occur::Must, clause, &mut |v| visit(&b.inner, v));
        }
        Clause::Prefix(q) => {
            if visitor.accept_field(&q.field) {
                let c = CompiledAutomaton::with_options(
                    &prefix_automaton(&q.prefix),
                    false,
                    true,
                    true,
                )
                .ok();
                visit_automaton(visitor, clause, &q.field, c);
            }
        }
        Clause::Wildcard(q) => {
            if visitor.accept_field(&q.field) {
                let c = wildcard_automaton(&q.pattern)
                    .and_then(|a| CompiledAutomaton::with_options(&a, false, true, false).ok());
                visit_automaton(visitor, clause, &q.field, c);
            }
        }
        Clause::Regexp(q) => {
            if visitor.accept_field(&q.field) {
                let c = automaton::RegExp::new(&q.pattern)
                    .ok()
                    .and_then(|r| r.to_automaton().ok())
                    .and_then(|a| {
                        automaton::operations::determinize(
                            &a,
                            automaton::DEFAULT_DETERMINIZE_WORK_LIMIT,
                        )
                        .ok()
                    })
                    .and_then(|a| CompiledAutomaton::with_options(&a, false, true, false).ok());
                visit_automaton(visitor, clause, &q.field, c);
            }
        }
        Clause::Fuzzy(q) => {
            if visitor.accept_field(&q.field) {
                let m = lucene_codecs::fuzzy::FuzzyMatch::new(
                    &q.term,
                    q.max_edits,
                    q.prefix_length,
                    q.transpositions,
                );
                visitor.consume_terms_matching(clause, &q.field, &|t: &[u8]| m.matches(t));
            }
        }
        Clause::TermInSet(q) => {
            if !visitor.accept_field(&q.field) {
                return;
            }
            let set: BTreeSet<&Vec<u8>> = q.terms.iter().collect();
            if set.len() == 1 {
                let only = set.iter().next().map(|t| (*t).clone()).unwrap_or_default();
                visitor.consume_terms(clause, &[Term::new(q.field.clone(), only)]);
            } else if set.len() > 1 {
                visitor.consume_terms_matching(clause, &q.field, &|t: &[u8]| {
                    set.contains(&t.to_vec())
                });
            }
        }
        Clause::Span(s) => visit_span(s, clause, visitor),
        Clause::PointsRange(q) => {
            if visitor.accept_field(&q.field) {
                visitor.visit_leaf(clause);
            }
        }
        Clause::Exists(q) => {
            if visitor.accept_field(&q.field) {
                visitor.visit_leaf(clause);
            }
        }
        // `MatchAllDocsQuery`, `MatchNoDocsQuery`, and any query without
        // terms of its own: a leaf.
        #[allow(unreachable_patterns)]
        _ => visitor.visit_leaf(clause),
    }
}

/// The field a span query is over (`SpanQuery.getField`).
fn span_field(s: &SpanQuery) -> Option<&str> {
    match s {
        SpanQuery::SpanTerm { field, .. } => Some(field),
        SpanQuery::SpanNear { clauses, .. } | SpanQuery::SpanOr { clauses } => {
            clauses.first().and_then(span_field)
        }
    }
}

/// `SpanTermQuery`/`SpanNearQuery`/`SpanOrQuery.visit`. A sub-span's
/// `Query` is its own [`Clause::Span`].
fn visit_span(s: &SpanQuery, clause: &Clause, visitor: &mut dyn QueryVisitor) {
    let Some(field) = span_field(s) else {
        return;
    };
    if !visitor.accept_field(field) {
        return;
    }
    match s {
        SpanQuery::SpanTerm { field, term } => {
            visitor.consume_terms(clause, &[Term::new(field.clone(), term.clone())]);
        }
        SpanQuery::SpanNear { clauses, .. } | SpanQuery::SpanOr { clauses } => {
            let occur = if matches!(s, SpanQuery::SpanNear { .. }) {
                Occur::Must
            } else {
                Occur::Should
            };
            with_sub(visitor, occur, clause, &mut |v| {
                for c in clauses {
                    let sub = Clause::Span(c.clone());
                    visit_span(c, &sub, v);
                }
            });
        }
    }
}

/// `query.visit(QueryVisitor.termCollector(set))`.
pub fn extract_terms(clause: &Clause) -> BTreeSet<Term> {
    let mut c = TermCollector::default();
    visit(clause, &mut c);
    c.terms
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::{BooleanQuery, TermQuery};

    #[test]
    fn empty_visitor_accepts_nothing_and_must_not_is_skipped() {
        let mut b = BooleanQuery::new();
        b.must
            .push(Clause::Term(TermQuery::new("f", b"a".to_vec())));
        b.must_not
            .push(Clause::Term(TermQuery::new("f", b"b".to_vec())));
        let terms = extract_terms(&Clause::Boolean(Box::new(b)));
        assert_eq!(terms.len(), 1);
        assert!(terms.contains(&Term::new("f", b"a".to_vec())));
        let mut e = EmptyVisitor;
        assert!(!e.accept_field("f"));
        visit(&Clause::Term(TermQuery::new("f", b"a".to_vec())), &mut e);
        e.visit_leaf(&Clause::Term(TermQuery::new("f", b"a".to_vec())));
        assert!(matches!(
            e.sub_visitor(
                Occur::Must,
                &Clause::Term(TermQuery::new("f", b"a".to_vec()))
            ),
            SubVisitor::Same
        ));
    }

    #[test]
    fn wildcard_escapes_and_prefix_automata() {
        let a = wildcard_automaton(b"a\\*b").unwrap();
        let c = CompiledAutomaton::with_options(&a, false, true, false).unwrap();
        assert_eq!(c.automaton_type, AutomatonType::SINGLE);
        assert_eq!(c.term.as_deref(), Some(&b"a*b"[..]));
        let trailing = wildcard_automaton(b"ab\\").unwrap();
        let c = CompiledAutomaton::with_options(&trailing, false, true, false).unwrap();
        assert_eq!(c.term.as_deref(), Some(&b"ab\\"[..]));
        let all =
            CompiledAutomaton::with_options(&prefix_automaton(b""), false, true, true).unwrap();
        assert_eq!(all.automaton_type, AutomatonType::ALL);
    }

    /// `CompiledAutomaton.visit`: a field the visitor rejects sees nothing;
    /// `NONE` nothing; `SINGLE` its term; `NORMAL` -- determinized or not --
    /// a matcher over the accepted terms.
    #[test]
    fn a_compiled_automaton_visits_by_its_type() {
        #[derive(Default)]
        struct Seen {
            terms: Vec<Term>,
            matched: Vec<bool>,
            reject: bool,
        }
        impl QueryVisitor for Seen {
            fn consume_terms(&mut self, _q: &Clause, terms: &[Term]) {
                self.terms.extend_from_slice(terms);
            }
            fn consume_terms_matching(&mut self, _q: &Clause, _f: &str, m: &dyn TermMatcher) {
                self.matched = vec![m.matches(b"ab"), m.matches(b"ac"), m.matches(b"b")];
            }
            fn accept_field(&self, _field: &str) -> bool {
                !self.reject
            }
        }
        let q = Clause::Term(TermQuery::new("f", b"x".to_vec()));
        // a(b|c), non-deterministic: two `a` transitions.
        let mut nfa = automaton::Automaton::new();
        let (s0, s1, s2, s3) = (
            nfa.create_state(),
            nfa.create_state(),
            nfa.create_state(),
            nfa.create_state(),
        );
        nfa.add_transition(s0, s1, i32::from(b'a'), i32::from(b'a'));
        nfa.add_transition(s0, s2, i32::from(b'a'), i32::from(b'a'));
        nfa.add_transition(s1, s3, i32::from(b'b'), i32::from(b'b'));
        nfa.add_transition(s2, s3, i32::from(b'c'), i32::from(b'c'));
        nfa.set_accept(s3, true);
        nfa.finish_state();
        let c = CompiledAutomaton::with_options(&nfa, false, false, true).unwrap();
        assert_eq!(c.automaton_type, AutomatonType::NORMAL);
        let mut seen = Seen::default();
        visit_compiled(&c, &mut seen, &q, "f");
        assert_eq!(seen.matched, vec![true, true, false]);
        let mut rejecting = Seen {
            reject: true,
            ..Seen::default()
        };
        visit_compiled(&c, &mut rejecting, &q, "f");
        assert!(rejecting.matched.is_empty());
        let none = CompiledAutomaton::with_options(&automaton::Automaton::new(), false, true, true)
            .unwrap();
        assert_eq!(none.automaton_type, AutomatonType::NONE);
        let mut seen = Seen::default();
        visit_compiled(&none, &mut seen, &q, "f");
        assert!(seen.terms.is_empty() && seen.matched.is_empty());
    }
}
