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
use std::sync::Arc;

use lucene_util::automaton::{self, AutomatonType, CompiledAutomaton};

use crate::extended_query::MultiTermSource;
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

    /// A matcher of the same terms that outlives the visit, for a visitor
    /// that keeps it (Java's keeps the `ByteRunAutomaton` it was supplied);
    /// `None`, the default, when this one cannot give it.
    fn to_owned_matcher(&self) -> Option<Box<dyn TermMatcher + Send + Sync>> {
        None
    }
}

/// The matcher of an automaton that accepts every term.
struct AcceptAll;

impl TermMatcher for AcceptAll {
    fn matches(&self, _term: &[u8]) -> bool {
        true
    }
    fn to_owned_matcher(&self) -> Option<Box<dyn TermMatcher + Send + Sync>> {
        Some(Box::new(AcceptAll))
    }
}

/// The matcher of a `NORMAL` compiled automaton: its runnable form.
struct Runs<'a>(&'a CompiledAutomaton);

impl TermMatcher for Runs<'_> {
    fn matches(&self, term: &[u8]) -> bool {
        self.0.byte_runnable().is_some_and(|r| r.run(term))
    }
    fn to_owned_matcher(&self) -> Option<Box<dyn TermMatcher + Send + Sync>> {
        // the NFA form keeps a cache it cannot share; the DFA is plain data
        match (&self.0.nfa_run_automaton, &self.0.run_automaton) {
            (None, Some(r)) => Some(Box::new(OwnedRun(r.clone()))),
            _ => None,
        }
    }
}

/// [`Runs`] over a determinized automaton, owned.
struct OwnedRun(automaton::ByteRunAutomaton);

impl TermMatcher for OwnedRun {
    fn matches(&self, term: &[u8]) -> bool {
        automaton::ByteRunnable::run(&self.0, term)
    }
    fn to_owned_matcher(&self) -> Option<Box<dyn TermMatcher + Send + Sync>> {
        Some(Box::new(OwnedRun(self.0.clone())))
    }
}

/// `FuzzyQuery`'s matcher: its pattern, built once.
struct Fuzzy<'a>(lucene_codecs::fuzzy::FuzzyMatch<'a>);

impl Fuzzy<'_> {
    fn of(q: &crate::query::FuzzyQuery) -> Fuzzy<'_> {
        Fuzzy(lucene_codecs::fuzzy::FuzzyMatch::new(
            &q.term,
            q.max_edits,
            q.prefix_length,
            q.transpositions,
        ))
    }
}

impl TermMatcher for Fuzzy<'_> {
    fn matches(&self, term: &[u8]) -> bool {
        self.0.matches(term)
    }
    fn to_owned_matcher(&self) -> Option<Box<dyn TermMatcher + Send + Sync>> {
        Some(Box::new(OwnedFuzzy(Arc::new(self.0.clone().into_owned()))))
    }
}

/// [`Fuzzy`], owned.
struct OwnedFuzzy(Arc<lucene_codecs::fuzzy::FuzzyMatch<'static>>);

impl TermMatcher for OwnedFuzzy {
    fn matches(&self, term: &[u8]) -> bool {
        self.0.matches(term)
    }
    fn to_owned_matcher(&self) -> Option<Box<dyn TermMatcher + Send + Sync>> {
        Some(Box::new(OwnedFuzzy(Arc::clone(&self.0))))
    }
}

/// `TermInSetQuery`'s matcher: membership of its terms.
struct InSet(BTreeSet<Vec<u8>>);

impl TermMatcher for InSet {
    fn matches(&self, term: &[u8]) -> bool {
        self.0.contains(term)
    }
    fn to_owned_matcher(&self) -> Option<Box<dyn TermMatcher + Send + Sync>> {
        Some(Box::new(InSet(self.0.clone())))
    }
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

/// A doc-values or point query's `visit`: a leaf when its field is accepted.
fn leaf_on(visitor: &mut dyn QueryVisitor, field: &str, clause: &Clause) {
    if visitor.accept_field(field) {
        visitor.visit_leaf(clause);
    }
}

/// A recorded `visit` of `clause`, done again on `visitor`.
fn replay(
    record: &crate::extended_query::VisitRecord,
    clause: &Clause,
    visitor: &mut dyn QueryVisitor,
) {
    use crate::extended_query::VisitEvent;
    let accepts = |v: &dyn QueryVisitor, guard: &Option<String>| {
        guard.as_deref().is_none_or(|f| v.accept_field(f))
    };
    for event in &record.events {
        match event {
            VisitEvent::Terms { guard, terms } => {
                if accepts(visitor, guard) {
                    visitor.consume_terms(clause, terms);
                }
            }
            VisitEvent::Leaf { guard } => {
                if accepts(visitor, guard) {
                    visitor.visit_leaf(clause);
                }
            }
            VisitEvent::Sub(occur, sub) => {
                with_sub(visitor, *occur, clause, &mut |v| replay(sub, clause, v));
            }
            VisitEvent::Node(node) => visit(node, visitor),
        }
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

/// The `visit` of a multi-term query by its source: `AutomatonQuery.visit`
/// (prefix, wildcard, regexp, term range, automaton: the compiled
/// automaton, when the visitor accepts the field), or `MultiTermQuery.visit`
/// (`lucene-join`'s term set: a leaf).
fn visit_source(source: &MultiTermSource, query: &Clause, visitor: &mut dyn QueryVisitor) {
    if !visitor.accept_field(source.field()) {
        return;
    }
    let c = match source {
        MultiTermSource::Prefix(q) => {
            CompiledAutomaton::with_options(&prefix_automaton(&q.prefix), false, true, true).ok()
        }
        MultiTermSource::Wildcard(q) => wildcard_automaton(&q.pattern)
            .and_then(|a| CompiledAutomaton::with_options(&a, false, true, false).ok()),
        MultiTermSource::Regexp(q) => automaton::RegExp::new(&q.pattern)
            .ok()
            .and_then(|r| r.to_automaton().ok())
            .and_then(|a| {
                automaton::operations::determinize(&a, automaton::DEFAULT_DETERMINIZE_WORK_LIMIT)
                    .ok()
            })
            .and_then(|a| CompiledAutomaton::with_options(&a, false, true, false).ok()),
        // `TermRangeQuery.toAutomaton`: an open end is inclusive
        MultiTermSource::TermRange(r) => automaton::automata::make_binary_interval(
            r.lower.as_deref(),
            r.include_lower || r.lower.is_none(),
            r.upper.as_deref(),
            r.include_upper || r.upper.is_none(),
        )
        .ok()
        .and_then(|a| CompiledAutomaton::with_options(&a, false, true, true).ok()),
        MultiTermSource::Automaton(q) => {
            CompiledAutomaton::with_options(&q.automaton, false, true, q.binary).ok()
        }
        MultiTermSource::TermSet(_) => return visitor.visit_leaf(query),
    };
    visit_automaton(visitor, query, source.field(), c);
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
        AutomatonType::ALL => visitor.consume_terms_matching(query, field, &AcceptAll),
        AutomatonType::NORMAL => visitor.consume_terms_matching(query, field, &Runs(c)),
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

/// `WildcardQuery.toAutomaton(term, workLimit)`, `None` when it is too
/// complex to determinize.
fn wildcard_automaton(pattern: &[u8]) -> Option<automaton::Automaton> {
    wildcard_automaton_checked(pattern).ok()
}

/// `WildcardQuery.toAutomaton(term, DEFAULT_DETERMINIZE_WORK_LIMIT)`: the
/// `TooComplexToDeterminizeException` it throws, or the automaton.
pub fn wildcard_automaton_checked(
    pattern: &[u8],
) -> Result<automaton::Automaton, automaton::TooComplexToDeterminize> {
    wildcard_automaton_within(pattern, automaton::DEFAULT_DETERMINIZE_WORK_LIMIT)
}

/// `WildcardQuery.toAutomaton(term, determinizeWorkLimit)`: the pattern's
/// automaton, determinized within `work_limit`.
pub fn wildcard_automaton_within(
    pattern: &[u8],
    work_limit: i32,
) -> Result<automaton::Automaton, automaton::TooComplexToDeterminize> {
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
    automaton::operations::determinize(&automaton::operations::concatenate(&refs), work_limit)
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
        Clause::Prefix(q) => visit_source(&MultiTermSource::Prefix(q.clone()), clause, visitor),
        Clause::Wildcard(q) => {
            visit_source(&MultiTermSource::Wildcard(q.clone()), clause, visitor);
        }
        Clause::Regexp(q) => visit_source(&MultiTermSource::Regexp(q.clone()), clause, visitor),
        Clause::Fuzzy(q) => {
            if visitor.accept_field(&q.field) {
                visitor.consume_terms_matching(clause, &q.field, &Fuzzy::of(q));
            }
        }
        Clause::TermInSet(q) => {
            if !visitor.accept_field(&q.field) {
                return;
            }
            let set: BTreeSet<Vec<u8>> = q.terms.iter().cloned().collect();
            if set.len() == 1 {
                let only = set.into_iter().next().unwrap_or_default();
                visitor.consume_terms(clause, &[Term::new(q.field.clone(), only)]);
            } else if set.len() > 1 {
                visitor.consume_terms_matching(clause, &q.field, &InSet(set));
            }
        }
        Clause::Span(s) => visit_span(s, clause, visitor),
        Clause::Extended(e) => match e.as_ref() {
            #[cfg(any(test, feature = "asserting"))]
            crate::extended_query::ExtendedQuery::Test(q) if q.delegates_query_semantics() => {
                visit(q.query(), visitor);
            }
            // `SynonymQuery.visit`
            crate::extended_query::ExtendedQuery::Synonym(q) => {
                if visitor.accept_field(&q.field) {
                    let terms: Vec<Term> = q
                        .terms
                        .iter()
                        .map(|(t, _)| Term::new(q.field.clone(), t.clone()))
                        .collect();
                    visitor.consume_terms(clause, &terms);
                }
            }
            crate::extended_query::ExtendedQuery::Span(node) => {
                visit_span_node(node, clause, visitor);
            }
            crate::extended_query::ExtendedQuery::MultiTerm(m) => {
                visit_source(&m.source, clause, visitor);
            }
            // `IndexOrDocValuesQuery.visit`: both of its queries, as `MUST`
            crate::extended_query::ExtendedQuery::IndexOrDocValues(q) => {
                with_sub(visitor, Occur::Must, clause, &mut |v| {
                    visit(&q.index_query, v);
                    visit(&q.dv_query, v);
                });
            }
            // the doc-values and point queries' `visit`: a leaf on an accepted field
            crate::extended_query::ExtendedQuery::NumericDocValuesRange(q) => {
                leaf_on(visitor, &q.field, clause);
            }
            crate::extended_query::ExtendedQuery::IndexSortRange(q) => {
                leaf_on(visitor, &q.field, clause);
            }
            crate::extended_query::ExtendedQuery::PointRange(q) => {
                leaf_on(visitor, &q.field, clause);
            }
            crate::extended_query::ExtendedQuery::PointInSet(q) => {
                leaf_on(visitor, &q.field, clause);
            }
            // OpenSearch's `FunctionScoreQuery.visit`: its query, as `MUST`
            crate::extended_query::ExtendedQuery::VisitOnly(inner) => {
                with_sub(visitor, Occur::Must, clause, &mut |v| visit(inner, v));
            }
            crate::extended_query::ExtendedQuery::Visited(record) => {
                replay(record, clause, visitor);
            }
            // `MultiPhrasePrefixQuery.visit`: every position but the prefix one
            crate::extended_query::ExtendedQuery::MultiPhrasePrefix(q) => {
                if !visitor.accept_field(&q.field) {
                    return;
                }
                let whole = &q.term_arrays[..q.term_arrays.len().saturating_sub(1)];
                with_sub(visitor, Occur::Must, clause, &mut |v| {
                    for position in whole {
                        let terms: Vec<Term> = position
                            .iter()
                            .map(|t| Term::new(q.field.clone(), t.clone()))
                            .collect();
                        if terms.len() == 1 {
                            v.consume_terms(clause, &terms);
                        } else {
                            with_sub(v, Occur::Should, clause, &mut |sv| {
                                sv.consume_terms(clause, &terms)
                            });
                        }
                    }
                });
            }
            crate::extended_query::ExtendedQuery::Document(d) => {
                if d.query.field().is_none_or(|f| visitor.accept_field(f)) {
                    visitor.visit_leaf(clause);
                }
            }
            _ => visitor.visit_leaf(clause),
        },
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

/// The `visit` of the span queries of `lucene-queries` (`SpanTermQuery`,
/// `SpanNearQuery`, `SpanOrQuery`, `SpanPositionCheckQuery`, `SpanNotQuery`,
/// `SpanContainQuery`, `FieldMaskingSpanQuery`, `SpanMultiTermQueryWrapper`).
fn visit_span_node(node: &crate::spans::SpanNode, clause: &Clause, visitor: &mut dyn QueryVisitor) {
    use crate::spans::SpanNode as N;
    let sub = |n: &N| Clause::from(n.clone());
    match node {
        N::Term { field, term } => {
            if visitor.accept_field(field) {
                visitor.consume_terms(clause, &[Term::new(field.clone(), term.clone())]);
            }
        }
        N::Near { clauses, .. } | N::Or { clauses } => {
            if !node.field().is_some_and(|f| visitor.accept_field(f)) {
                return;
            }
            let occur = if matches!(node, N::Near { .. }) {
                Occur::Must
            } else {
                Occur::Should
            };
            with_sub(visitor, occur, clause, &mut |v| {
                for c in clauses {
                    visit_span_node(c, &sub(c), v);
                }
            });
        }
        N::First { inner, .. } | N::PositionRange { inner, .. } => {
            if node.field().is_some_and(|f| visitor.accept_field(f)) {
                with_sub(visitor, Occur::Must, clause, &mut |v| {
                    visit_span_node(inner, &sub(inner), v);
                });
            }
        }
        N::Not {
            include, exclude, ..
        } => {
            if !node.field().is_some_and(|f| visitor.accept_field(f)) {
                return;
            }
            with_sub(visitor, Occur::Must, clause, &mut |v| {
                visit_span_node(include, &sub(include), v);
            });
            with_sub(visitor, Occur::MustNot, clause, &mut |v| {
                visit_span_node(exclude, &sub(exclude), v);
            });
        }
        N::Containing { big, little } | N::Within { big, little } => {
            if !node.field().is_some_and(|f| visitor.accept_field(f)) {
                return;
            }
            with_sub(visitor, Occur::Must, clause, &mut |v| {
                visit_span_node(big, &sub(big), v);
                visit_span_node(little, &sub(little), v);
            });
        }
        N::FieldMasking { inner, .. } => visit_span_node(inner, &sub(inner), visitor),
        // `SpanMultiTermQueryWrapper.visit`: the wrapped query, as a
        // `MUST`
        N::MultiTerm(m) => {
            if visitor.accept_field(m.source.field()) {
                let inner = match &m.source {
                    MultiTermSource::Prefix(q) => Clause::Prefix(q.clone()),
                    MultiTermSource::Wildcard(q) => Clause::Wildcard(q.clone()),
                    MultiTermSource::Regexp(q) => Clause::Regexp(q.clone()),
                    _ => Clause::Extended(Box::new(
                        crate::extended_query::ExtendedQuery::MultiTerm(m.clone()),
                    )),
                };
                with_sub(visitor, Occur::Must, clause, &mut |v| visit(&inner, v));
            }
        }
        // `SpanPayloadCheckQuery.visit`: its match, as a `MUST`
        N::PayloadCheck(p) => {
            if p.inner.field().is_some_and(|f| visitor.accept_field(f)) {
                with_sub(visitor, Occur::Must, clause, &mut |v| {
                    visit_span_node(&p.inner, &sub(&p.inner), v);
                });
            }
        }
        // `PayloadScoreQuery.visit`: the wrapped query, as a `MUST`
        N::PayloadScore(p) => with_sub(visitor, Occur::Must, clause, &mut |v| {
            visit_span_node(&p.inner, &sub(&p.inner), v);
        }),
        // `SpanGapQuery.visit` and `SpanMatchNoDocsQuery.visit`: a leaf
        N::Gap { .. } | N::NoMatch { .. } => visitor.visit_leaf(clause),
    }
}

/// `query.visit(QueryVisitor.termCollector(set))`.
pub fn extract_terms(clause: &Clause) -> BTreeSet<Term> {
    let mut c = TermCollector::default();
    visit(clause, &mut c);
    c.terms
}

#[cfg(test)]
mod tests;
