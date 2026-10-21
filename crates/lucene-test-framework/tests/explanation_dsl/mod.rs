//! Query builders the explanation suites share, so each Java test reads
//! as one line: `bq(vec![(M, t("w1")), (S, a("w2"))])` is a `BooleanQuery`
//! of `+field:w1 alt:w2`.
#![allow(dead_code, unused_imports)]

use lucene_search::extended_query::SynonymQuery;
use lucene_search::query::{
    BooleanQuery, BoostQuery, Clause, ConstantScoreQuery, DisjunctionMaxQuery, MatchAllDocsQuery,
    MultiPhraseQuery, PhraseQuery, TermQuery,
};
use lucene_test_framework::search::base_explanation::{ALTFIELD, DOC_FIELDS, FIELD};

pub use lucene_test_framework::search::base_explanation::match_these_items;

/// A `BooleanClause.Occur`.
#[derive(Clone, Copy)]
pub enum O {
    M,
    S,
    N,
    F,
}
pub use O::{F, M, N, S};

/// `new TermQuery(new Term(FIELD, s))`.
pub fn t(s: &str) -> Clause {
    Clause::Term(TermQuery::new(FIELD, s))
}

/// `new TermQuery(new Term(ALTFIELD, s))`.
pub fn a(s: &str) -> Clause {
    Clause::Term(TermQuery::new(ALTFIELD, s))
}

/// `new BoostQuery(q, b)`.
pub fn boost(q: Clause, b: f32) -> Clause {
    BoostQuery::new(q, b).into()
}

/// `MatchAllDocsQuery.INSTANCE` over the four documents.
pub fn match_all() -> Clause {
    Clause::MatchAllDocs(MatchAllDocsQuery::new(
        i32::try_from(DOC_FIELDS.len()).expect("four documents"),
    ))
}

/// `new ConstantScoreQuery(q)`.
pub fn csq(q: Clause) -> Clause {
    ConstantScoreQuery::new(q, 1.0).into()
}

/// `new DisjunctionMaxQuery(disjuncts, tie)`.
pub fn dmq(disjuncts: Vec<Clause>, tie_breaker: f32) -> Clause {
    DisjunctionMaxQuery {
        disjuncts,
        tie_breaker,
    }
    .into()
}

/// `new PhraseQuery(slop, FIELD, terms...)`.
pub fn phrase(slop: u32, terms: &[&str]) -> Clause {
    PhraseQuery::new(FIELD, terms.iter().map(|s| s.as_bytes().to_vec()))
        .with_slop(slop)
        .into()
}

/// `new PhraseQuery(slop, ALTFIELD, terms...)`.
pub fn alt_phrase(slop: u32, terms: &[&str]) -> Clause {
    PhraseQuery::new(ALTFIELD, terms.iter().map(|s| s.as_bytes().to_vec()))
        .with_slop(slop)
        .into()
}

/// A `MultiPhraseQuery` of `FIELD`, one array of alternatives per position.
pub fn mpq(slop: u32, arrays: &[&[&str]]) -> Clause {
    let mut q = MultiPhraseQuery::new(
        FIELD,
        arrays.iter().map(|alts| {
            alts.iter()
                .map(|s| s.as_bytes().to_vec())
                .collect::<Vec<_>>()
        }),
    );
    q.slop = slop;
    q.into()
}

/// A `SynonymQuery` of `FIELD`, every term at boost 1.
pub fn synonym(terms: &[&str]) -> Clause {
    SynonymQuery::new(FIELD, terms.iter().map(|s| (s.as_bytes().to_vec(), 1.0)))
        .expect("valid synonyms")
        .into()
}

/// A `BooleanQuery` of `(occur, clause)` pairs.
pub fn bq(clauses: Vec<(O, Clause)>) -> Clause {
    bq_msm(0, clauses)
}

/// [`bq`] with `setMinimumNumberShouldMatch(msm)`.
pub fn bq_msm(msm: usize, clauses: Vec<(O, Clause)>) -> Clause {
    let mut b = BooleanQuery {
        minimum_should_match: msm,
        ..BooleanQuery::default()
    };
    for (o, c) in clauses {
        match o {
            M => b.must.push(c),
            S => b.should.push(c),
            N => b.must_not.push(c),
            F => b.filter.push(c),
        }
    }
    b.into()
}

/// One `#[test]` per `name: query => [docs];`, each `qtest` over a fresh
/// `BaseExplanationTestCase` index, drawing from the test's own stream.
#[macro_export]
macro_rules! qtests {
    ($($name:ident: $q:expr => [$($d:expr),* $(,)?];)*) => {
        $(
            #[test]
            fn $name() {
                let mut tc = lucene_test_framework::util::lucene_test_case::LuceneTestCase::new(
                    stringify!($name),
                );
                let base = lucene_test_framework::search::base_explanation::BaseExplanation::new(
                    tc.random(),
                );
                base.qtest(tc.random(), $q, &[$($d),*]);
            }
        )*
    };
}
