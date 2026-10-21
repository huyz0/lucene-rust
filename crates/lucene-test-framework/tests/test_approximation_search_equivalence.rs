//! Lucene's `TestApproximationSearchEquivalence`: boolean queries score the
//! same whether their clauses are plain or behind a
//! `RandomApproximationQuery`'s two-phase view.
//!
//! Java hands every `RandomApproximationQuery` the test's own `random()`;
//! here each gets a `JavaRandom` seeded from the test's stream.

mod explanation_dsl;

use explanation_dsl::{bq, csq, t, M, N, O, S};
use lucene_search::query::Clause;
use lucene_search::test_support::TestQuery;
use lucene_test_framework::search::search_equivalence::{random_term, SearchEquivalence};
use lucene_test_framework::util::lucene_test_case::LuceneTestCase;
use lucene_util::java_random::JavaRandom;

struct Case {
    tc: LuceneTestCase,
    s: SearchEquivalence,
}

impl Case {
    fn new(name: &str) -> Self {
        let mut tc = LuceneTestCase::new(name);
        let (nightly, multiplier) = (tc.nightly(), tc.multiplier());
        let s = SearchEquivalence::new(tc.random(), nightly, multiplier);
        Self { tc, s }
    }

    fn term(&mut self) -> String {
        String::from_utf8(random_term(self.tc.random())).expect("a letter")
    }

    /// `t1`, a `t2 != t1`, and `t3`.
    fn three_terms(&mut self) -> (String, String, String) {
        let t1 = self.term();
        let mut t2 = self.term();
        while t2 == t1 {
            t2 = self.term();
        }
        let t3 = self.term();
        (t1, t2, t3)
    }

    /// `new RandomApproximationQuery(q, random())`.
    fn approx(&mut self, q: Clause) -> Clause {
        TestQuery::random_approximation(q, JavaRandom::new(self.tc.random().next_long()))
    }

    fn same_scores(&mut self, q1: &Clause, q2: &Clause) {
        self.s.assert_same_scores(self.tc.random(), q1, q2);
    }

    /// The two-clause shape: `(o, q1) (o2, q2)` plain and approximated.
    fn pair(&mut self, occurs: [O; 2]) {
        let (a, b) = (self.term(), self.term());
        let plain = bq(vec![(occurs[0], t(&a)), (occurs[1], t(&b))]);
        let qa = self.approx(t(&a));
        let qb = self.approx(t(&b));
        let approx = bq(vec![(occurs[0], qa), (occurs[1], qb)]);
        self.same_scores(&plain, &approx);
    }

    /// The nested shape: `(outer, [(inner, q1) (inner2, q2)]) (outer, q3)`.
    fn nested(&mut self, inner: [O; 2], outer: O) {
        let (a, b, c) = self.three_terms();
        let plain = bq(vec![
            (outer, bq(vec![(inner[0], t(&a)), (inner[1], t(&b))])),
            (outer, t(&c)),
        ]);
        let qa = self.approx(t(&a));
        let qb = self.approx(t(&b));
        let approx = bq(vec![
            (outer, bq(vec![(inner[0], qa), (inner[1], qb)])),
            (outer, t(&c)),
        ]);
        self.same_scores(&plain, &approx);
    }
}

#[test]
fn conjunction() {
    Case::new("conjunction").pair([M, M]);
}

#[test]
fn nested_conjunction() {
    Case::new("nested_conjunction").nested([M, M], M);
}

#[test]
fn disjunction() {
    Case::new("disjunction").pair([S, S]);
}

#[test]
fn nested_disjunction() {
    Case::new("nested_disjunction").nested([S, S], S);
}

#[test]
fn disjunction_in_conjunction() {
    Case::new("disjunction_in_conjunction").nested([S, S], M);
}

#[test]
fn conjunction_in_disjunction() {
    Case::new("conjunction_in_disjunction").nested([M, M], S);
}

#[test]
fn constant_score() {
    let mut c = Case::new("constant_score");
    let (a, b) = (c.term(), c.term());
    let plain = bq(vec![(M, csq(t(&a))), (M, csq(t(&b)))]);
    let qa = c.approx(t(&a));
    let qb = c.approx(t(&b));
    let approx = bq(vec![(M, csq(qa)), (M, csq(qb))]);
    c.same_scores(&plain, &approx);
}

#[test]
fn exclusion() {
    Case::new("exclusion").pair([M, N]);
}

#[test]
fn nested_exclusion() {
    let mut c = Case::new("nested_exclusion");
    let (a, b, d) = c.three_terms();
    let plain = bq(vec![(M, bq(vec![(M, t(&a)), (N, t(&b))])), (M, t(&d))]);
    // both the required and the excluded clause, then each alone, have an
    // approximation
    for (approx_req, approx_excl) in [(true, true), (true, false), (false, true)] {
        let req = if approx_req { c.approx(t(&a)) } else { t(&a) };
        let excl = if approx_excl { c.approx(t(&b)) } else { t(&b) };
        let approx = bq(vec![(M, bq(vec![(M, req), (N, excl)])), (M, t(&d))]);
        c.same_scores(&plain, &approx);
    }
}

#[test]
fn req_opt() {
    Case::new("req_opt").nested([M, S], M);
}
