//! `GenSimilaritySearch`' prefix query grammar and similarity names, shared
//! by `tests/similarity_search_fixtures.rs` (hits and score bits against
//! Lucene under every similarity) and the `similarity` micro benchmark
//! (`benchmarks/rust-runner/src/micro_m7.rs`, which includes this file by
//! path). The Java twin is `GenSimilaritySearch.parse`/`sims`, which
//! `benchmarks/micro/java/M7Micro.java` calls.
#![allow(dead_code, clippy::arithmetic_side_effects)]

use std::sync::Arc;

use lucene_search::query::{
    BooleanQuery, BoostQuery, Clause, ConstantScoreQuery, DisjunctionMaxQuery, FuzzyQuery,
    MultiPhraseQuery, PhraseQuery, SpanQuery, TermQuery,
};
use lucene_search::similarities::*;

/// `GenSimilaritySearch.sims()`, by the same names.
pub fn sim(name: &str) -> Arc<dyn Similarity> {
    let lmjm = || -> Arc<dyn Similarity> {
        Arc::new(LmJelinekMercerSimilarity::new(CollectionModel::Default, true, 0.7).unwrap())
    };
    match name {
        "bm25" => Arc::new(Bm25Similarity::default()),
        "bm25_k2_b03" => Arc::new(Bm25Similarity::new(2.0, 0.3, true).unwrap()),
        "classic" => Arc::new(ClassicSimilarity::default()),
        "boolean" => Arc::new(BooleanSimilarity),
        "dfr_In_B_H2" => Arc::new(
            DfrSimilarity::new(BasicModel::In, AfterEffect::B, Normalization::H2_DEFAULT).unwrap(),
        ),
        "ib_LL_DF_H2" => Arc::new(
            IbSimilarity::new(Distribution::LL, Lambda::DF, Normalization::H2_DEFAULT).unwrap(),
        ),
        "dfi_saturated" => Arc::new(DfiSimilarity::new(Independence::Saturated)),
        "lmdirichlet" => Arc::new(LmDirichletSimilarity::default()),
        "lmjm_0.7" => lmjm(),
        "ax_f2exp" => {
            Arc::new(AxiomaticSimilarity::new(AxiomaticVariant::F2Exp, true, 0.5, 1, 0.2).unwrap())
        }
        "multi" => Arc::new(
            MultiSimilarity::new(vec![
                Arc::new(Bm25Similarity::default()),
                Arc::new(ClassicSimilarity::default()),
            ])
            .unwrap(),
        ),
        "perfield" => Arc::new(
            PerFieldSimilarity::new(lmjm())
                .with_field("title", Arc::new(ClassicSimilarity::default())),
        ),
        other => panic!("unknown similarity {other}"),
    }
}

/// `GenSimilaritySearch.parse`: the queries' prefix syntax.
pub fn parse(tok: &mut std::slice::Iter<'_, &str>) -> Clause {
    let mut next = || *tok.next().expect("truncated query");
    let op = next();
    match op {
        "T" => {
            let field = next();
            Clause::Term(TermQuery::new(field, next()))
        }
        "P" => {
            let field = next();
            let slop: u32 = next().parse().unwrap();
            let n: usize = next().parse().unwrap();
            let terms: Vec<&str> = (0..n).map(|_| next()).collect();
            Clause::Phrase(PhraseQuery::new(field, terms).with_slop(slop))
        }
        "B" => {
            let must: usize = next().parse().unwrap();
            let should: usize = next().parse().unwrap();
            let must_not: usize = next().parse().unwrap();
            let must: Vec<Clause> = (0..must).map(|_| parse(tok)).collect();
            let should: Vec<Clause> = (0..should).map(|_| parse(tok)).collect();
            let must_not: Vec<Clause> = (0..must_not).map(|_| parse(tok)).collect();
            Clause::Boolean(Box::new(
                BooleanQuery::new()
                    .with_must(must)
                    .with_should(should)
                    .with_must_not(must_not),
            ))
        }
        "X" => {
            let boost: f32 = next().parse().unwrap();
            Clause::Boost(Box::new(BoostQuery::new(parse(tok), boost)))
        }
        "C" => {
            let score: f32 = next().parse().unwrap();
            Clause::ConstantScore(Box::new(ConstantScoreQuery::new(parse(tok), score)))
        }
        "D" => {
            let tie: f32 = next().parse().unwrap();
            let n: usize = next().parse().unwrap();
            let disjuncts: Vec<Clause> = (0..n).map(|_| parse(tok)).collect();
            Clause::DisjunctionMax(Box::new(DisjunctionMaxQuery::new(disjuncts, tie)))
        }
        op @ ("S" | "N" | "O") => Clause::Span(parse_span_op(op, tok)),
        "F" => {
            let field = next();
            let term = next();
            let edits: u8 = next().parse().unwrap();
            let prefix: usize = next().parse().unwrap();
            let max: usize = next().parse().unwrap();
            Clause::Fuzzy(
                FuzzyQuery::new(field, term)
                    .with_max_edits(edits)
                    .with_prefix_length(prefix)
                    .with_max_expansions(max),
            )
        }
        "M" => {
            let field = next();
            let slop: u32 = next().parse().unwrap();
            let n: usize = next().parse().unwrap();
            let mut arrays: Vec<Vec<Vec<u8>>> = Vec::new();
            for _ in 0..n {
                let k: usize = next().parse().unwrap();
                arrays.push((0..k).map(|_| next().as_bytes().to_vec()).collect());
            }
            Clause::MultiPhrase(MultiPhraseQuery::new(field, arrays).with_slop(slop))
        }
        "G" => {
            let must: usize = next().parse().unwrap();
            let filter: usize = next().parse().unwrap();
            let must: Vec<Clause> = (0..must).map(|_| parse(tok)).collect();
            let filter: Vec<Clause> = (0..filter).map(|_| parse(tok)).collect();
            Clause::Boolean(Box::new(
                BooleanQuery::new().with_must(must).with_filter(filter),
            ))
        }
        other => panic!("query op {other}"),
    }
}

/// `GenSimilaritySearch.span`: `S field term`, `N slop inOrder n span...`,
/// `O n span...`.
pub fn parse_span(tok: &mut std::slice::Iter<'_, &str>) -> SpanQuery {
    let op = *tok.next().expect("truncated span");
    parse_span_op(op, tok)
}

pub fn parse_span_op(op: &str, tok: &mut std::slice::Iter<'_, &str>) -> SpanQuery {
    let mut next = || *tok.next().expect("truncated span");
    match op {
        "S" => {
            let field = next();
            SpanQuery::span_term(field, next())
        }
        "N" => {
            let slop: u32 = next().parse().unwrap();
            let in_order = next() == "1";
            let n: usize = next().parse().unwrap();
            let clauses: Vec<SpanQuery> = (0..n).map(|_| parse_span(tok)).collect();
            SpanQuery::span_near(clauses, slop, in_order)
        }
        "O" => {
            let n: usize = next().parse().unwrap();
            let clauses: Vec<SpanQuery> = (0..n).map(|_| parse_span(tok)).collect();
            SpanQuery::span_or(clauses)
        }
        other => panic!("span op {other}"),
    }
}

pub fn query(text: &str) -> BooleanQuery {
    let tokens: Vec<&str> = text.split(' ').collect();
    let mut it = tokens.iter();
    let clause = parse(&mut it);
    assert!(it.next().is_none(), "trailing tokens in {text}");
    match clause {
        Clause::Boolean(b) => *b,
        // `BooleanQuery.rewrite`: a lone `MUST` clause is that clause.
        other => BooleanQuery::new().with_must([other]),
    }
}
