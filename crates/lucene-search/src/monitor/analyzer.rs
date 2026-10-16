//! `org.apache.lucene.monitor.QueryAnalyzer` and `QueryDecomposer`: a
//! query's term tree, and the simpler queries it decomposes into, for the
//! presearcher to index.
//!
//! `QueryAnalyzer` walks a query's clauses the way Java's `QueryVisitor`
//! does: a `MUST`/`FILTER` clause is a conjunction child, a `SHOULD` clause a
//! disjunction child (unless the parent has required clauses, which make it
//! redundant), a `MUST_NOT` clause none, and a leaf query its terms or, for an
//! unknown leaf, the any-token marker.

use crate::monitor::tree::{QueryTree, Term, TermWeightor};
use crate::query::{BooleanQuery, Clause};

/// One child of a conjunction or disjunction being built: `Java`'s
/// `Function<TermWeightor, QueryTree>`, resolved once the weightor is known.
enum Child {
    Term(Term),
    Any(String),
    Node(Node),
}

/// A conjunction or a disjunction under construction.
struct Node {
    disjunction: bool,
    children: Vec<Child>,
}

impl Node {
    fn apply(self, weightor: &TermWeightor) -> Result<QueryTree, String> {
        let children: Vec<QueryTree> = self
            .children
            .into_iter()
            .map(|c| match c {
                Child::Term(t) => Ok(QueryTree::term(t, weightor)),
                Child::Any(reason) => Ok(QueryTree::any_term(reason)),
                Child::Node(n) => n.apply(weightor),
            })
            .collect::<Result<_, String>>()?;
        if self.disjunction {
            disjunction(children)
        } else {
            conjunction(children)
        }
    }
}

/// `QueryTree.conjunction(children, weightor)`'s node: `children` sorted
/// heaviest first, and a conjunction of unweighted children collapsed to
/// the first.
fn conjunction(mut children: Vec<QueryTree>) -> Result<QueryTree, String> {
    if children.is_empty() {
        return Err("Cannot build a conjunction with no children".to_string());
    }
    if children.len() == 1 {
        return Ok(children.swap_remove(0));
    }
    let mut weighted = Vec::with_capacity(children.len());
    for c in children {
        weighted.push((c.weight()?, c));
    }
    if weighted.iter().all(|(w, _)| *w <= 0.0) {
        return Ok(weighted.swap_remove(0).1);
    }
    weighted.sort_by(|a, b| b.0.total_cmp(&a.0));
    Ok(QueryTree::Conjunction(
        weighted.into_iter().map(|(_, c)| c).collect(),
    ))
}

/// `QueryTree.disjunction(children, weightor)`'s node: a disjunction with an
/// unweighted child is unweighted, and `children` sorted lightest first.
fn disjunction(mut children: Vec<QueryTree>) -> Result<QueryTree, String> {
    if children.is_empty() {
        return Err("Cannot build a disjunction with no children".to_string());
    }
    if children.len() == 1 {
        return Ok(children.swap_remove(0));
    }
    let mut weighted = Vec::with_capacity(children.len());
    for c in children {
        weighted.push((c.weight()?, c));
    }
    if let Some(i) = weighted.iter().position(|(w, _)| *w == 0.0) {
        return Ok(weighted.swap_remove(i).1);
    }
    weighted.sort_by(|a, b| a.0.total_cmp(&b.0));
    Ok(QueryTree::Disjunction(
        weighted.into_iter().map(|(_, c)| c).collect(),
    ))
}

/// `QueryAnalyzer`: a query to its term tree.
#[derive(Debug, Clone, Default)]
pub struct QueryAnalyzer;

impl QueryAnalyzer {
    /// `buildTree(luceneQuery, weightor)`.
    pub fn build_tree(&self, query: &Clause, weightor: &TermWeightor) -> Result<QueryTree, String> {
        // `QueryBuilder.apply`: a conjunction over the query's children -- but
        // a boolean whose clauses are all `SHOULD` is a disjunction of them,
        // as its `Disjunction` sub-visitor is the visited boolean's own node.
        let disjunction = matches!(
            query,
            Clause::Boolean(b) if b.must.is_empty() && b.filter.is_empty() && !b.should.is_empty()
        );
        let mut root = Node {
            disjunction,
            children: Vec::new(),
        };
        collect(query, &mut root);
        root.apply(weightor)
    }
}

/// `QueryBuilder.visit`: a query's children added to `node`. A boolean adds
/// one child per clause, by its occurrence (`getSubVisitor`); any other query
/// is a leaf, which contributes its terms (`consumeTerms`), or, when it has
/// none, the any-token marker (`visitLeaf` with no mapped tree).
fn collect(query: &Clause, node: &mut Node) {
    match query {
        Clause::Boolean(b) => {
            for c in &b.must {
                add_clause(c, b, Occur::Must, node);
            }
            for c in &b.filter {
                add_clause(c, b, Occur::Must, node);
            }
            for c in &b.must_not {
                add_clause(c, b, Occur::MustNot, node);
            }
            for c in &b.should {
                add_clause(c, b, Occur::Should, node);
            }
        }
        Clause::Boost(b) => {
            // `BoostQuery.visit`: the wrapped query, through a `MUST` child
            let mut n = Node {
                disjunction: is_shoulds_only(&b.inner),
                children: Vec::new(),
            };
            collect(&b.inner, &mut n);
            node.children.push(Child::Node(n));
        }
        leaf => {
            let terms = visit_order_terms(leaf);
            if terms.is_empty() {
                node.children.push(Child::Any(leaf_string(leaf)));
            }
            for t in terms {
                node.children.push(Child::Term(t));
            }
        }
    }
}

enum Occur {
    Must,
    MustNot,
    Should,
}

/// `getSubVisitor(occur, parent)`: where one clause of `parent` goes.
fn add_clause(clause: &Clause, parent: &BooleanQuery, occur: Occur, node: &mut Node) {
    match occur {
        Occur::Must => {
            let mut n = Node {
                disjunction: is_shoulds_only(clause),
                children: Vec::new(),
            };
            collect(clause, &mut n);
            node.children.push(Child::Node(n));
        }
        Occur::MustNot => {
            // a pure negative boolean matches every document: the marker
            if parent.should.is_empty() && parent.must.is_empty() && parent.filter.is_empty() {
                node.children.push(Child::Any(format!(
                    "PURE NEGATIVE QUERY[{}]",
                    leaf_string(&Clause::Boolean(Box::new(parent.clone())))
                )));
            }
        }
        Occur::Should => {
            // beside a required clause a SHOULD adds nothing to the tree
            if !parent.must.is_empty() || !parent.filter.is_empty() {
                return;
            }
            let mut n = Node {
                disjunction: is_shoulds_only(clause),
                children: Vec::new(),
            };
            collect(clause, &mut n);
            node.children.push(Child::Node(n));
        }
    }
}

/// Whether a clause is a boolean of `SHOULD` clauses only: its visited node
/// is a disjunction, as `BooleanQuery`'s own `Disjunction` sub-visitor is.
fn is_shoulds_only(clause: &Clause) -> bool {
    matches!(
        clause,
        Clause::Boolean(b) if b.must.is_empty() && b.filter.is_empty() && !b.should.is_empty()
    )
}

/// `QueryVisitor`'s `consumeTerms`, in the order the query visits them, as
/// `QueryBuilder` adds them, duplicates included (`fox AND fox` is two terms):
/// a `Vec`, where the shared `TermCollector` keeps a sorted set.
fn visit_order_terms(query: &Clause) -> Vec<Term> {
    struct InOrder(Vec<Term>);
    impl crate::query_visitor::QueryVisitor for InOrder {
        fn consume_terms(&mut self, _query: &Clause, terms: &[crate::query_visitor::Term]) {
            self.0.extend(terms.iter().map(|t| Term {
                field: t.field.clone(),
                bytes: t.bytes.clone(),
            }));
        }
    }
    let mut v = InOrder(Vec::new());
    crate::query_visitor::visit(query, &mut v);
    v.0
}

/// `Query.toString()` with no default field: how the marker reads a query it
/// could not index.
pub(crate) fn leaf_string(query: &Clause) -> String {
    crate::queryparser::ParsedQuery::from_clause(query.clone()).to_string("\u{0}")
}

/// `QueryDecomposer`: a query as the set of simpler queries that together
/// match what it does, for the presearcher to index each on its own.
#[derive(Debug, Default)]
pub struct QueryDecomposer;

impl QueryDecomposer {
    /// `decompose(q)`: the distinct sub-queries, in first-seen order.
    pub fn decompose(&self, q: &Clause) -> Vec<Clause> {
        match q {
            Clause::Boolean(b) => self.decompose_boolean(b),
            Clause::DisjunctionMax(d) => {
                let mut out = Vec::new();
                for sub in &d.disjuncts {
                    push_distinct(&mut out, self.decompose(sub));
                }
                out
            }
            Clause::Boost(b) => self.decompose_boost(b.inner.as_ref(), b.boost),
            other => vec![other.clone()],
        }
    }

    /// `decomposeBoostQuery`: a boost of `1` is the query itself.
    fn decompose_boost(&self, inner: &Clause, boost: f32) -> Vec<Clause> {
        if boost == 1.0 {
            return self.decompose(inner);
        }
        self.decompose(inner)
            .into_iter()
            .map(|sub| {
                Clause::Boost(Box::new(crate::query::BoostQuery {
                    inner: Box::new(sub),
                    boost,
                }))
            })
            .collect()
    }

    /// `decomposeBoolean`: a boolean with a required clause and disjunctions
    /// cannot be split, nor one with several required clauses; a single
    /// required clause and no disjunctions decomposes that clause; exclusions
    /// go into every sub-query.
    fn decompose_boolean(&self, q: &BooleanQuery) -> Vec<Clause> {
        if q.minimum_should_match > 1 {
            return vec![Clause::Boolean(Box::new(q.clone()))];
        }
        let mut subqueries: Vec<Clause> = Vec::new();
        let mut exclusions: Vec<Clause> = Vec::new();
        let mut mandatory: Vec<Clause> = Vec::new();
        for c in q.must.iter().chain(q.filter.iter()) {
            push_distinct(&mut mandatory, vec![c.clone()]);
        }
        for c in &q.must_not {
            push_distinct(&mut exclusions, vec![c.clone()]);
        }
        for c in &q.should {
            push_distinct(&mut subqueries, self.decompose(c));
        }
        if mandatory.len() > 1 || (mandatory.len() == 1 && !subqueries.is_empty()) {
            return vec![Clause::Boolean(Box::new(q.clone()))];
        }
        if let Some(m) = mandatory.first() {
            push_distinct(&mut subqueries, self.decompose(m));
        }
        if exclusions.is_empty() {
            return subqueries;
        }
        let mut rewritten = Vec::new();
        for sub in subqueries {
            let mut bq = BooleanQuery::new();
            bq.must.push(sub);
            bq.must_not.extend(exclusions.iter().cloned());
            push_distinct(&mut rewritten, vec![Clause::Boolean(Box::new(bq))]);
        }
        rewritten
    }
}

/// Appends each of `items` not already in `out`: a `LinkedHashSet`'s order.
fn push_distinct(out: &mut Vec<Clause>, items: Vec<Clause>) {
    for item in items {
        if !out.contains(&item) {
            out.push(item);
        }
    }
}
