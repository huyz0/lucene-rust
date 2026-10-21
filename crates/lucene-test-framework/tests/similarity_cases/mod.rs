//! Each `BaseSimilarityTestCase` subclass's `getSimilarity(Random)`, by
//! the Java test's name without its `Test` prefix.
#![allow(dead_code)]

use std::sync::Arc;

use lucene_search::similarities::{
    AfterEffect, AxiomaticSimilarity, AxiomaticVariant, BasicModel, Bm25Similarity,
    BooleanSimilarity, ClassicSimilarity, CollectionModel, DfiSimilarity, DfrSimilarity,
    Distribution, IbSimilarity, Independence, IndriDirichletSimilarity, Lambda,
    LmDirichletSimilarity, LmJelinekMercerSimilarity, Normalization, RawTfSimilarity, Similarity,
};
use lucene_test_framework::search::similarities::FLOAT_MIN_VALUE;
use lucene_util::java_random::JavaRandom;

/// A subclass's `getSimilarity`.
pub type GetSimilarity = fn(&mut JavaRandom) -> Arc<dyn Similarity>;

/// `Integer.MAX_VALUE` as the `float` Java's `int * float` promotes it to.
const INT_MAX_F: f32 = i32::MAX as f32;

/// `0`, `Float.MIN_VALUE`, `Integer.MAX_VALUE` or `Integer.MAX_VALUE *
/// nextFloat()`: k1, mu, c.
fn unbounded(r: &mut JavaRandom) -> f32 {
    match r.next_int_bounded(4) {
        0 => 0.0,
        1 => FLOAT_MIN_VALUE,
        2 => INT_MAX_F,
        _ => INT_MAX_F * r.next_float(),
    }
}

/// `0`, `Float.MIN_VALUE`, `1` or `nextFloat()`: b, s, k.
fn unit(r: &mut JavaRandom) -> f32 {
    match r.next_int_bounded(4) {
        0 => 0.0,
        1 => FLOAT_MIN_VALUE,
        2 => 1.0,
        _ => r.next_float(),
    }
}

/// `BasicModelTestCase`/`DistributionTestCase`'s normalization: c, z, mu
/// drawn, then one of the five picked.
fn normalization(r: &mut JavaRandom) -> Normalization {
    let c = unbounded(r);
    let z = match r.next_int_bounded(3) {
        0 => FLOAT_MIN_VALUE,
        1 => 0.5f32.next_down(),
        _ => {
            let z = r.next_float() / 2.0;
            if z == 0.0 {
                z.next_up()
            } else {
                z
            }
        }
    };
    let mu = unbounded(r);
    match r.next_int_bounded(5) {
        0 => Normalization::None,
        1 => Normalization::H1(c),
        2 => Normalization::H2(c),
        3 => Normalization::H3(mu),
        _ => Normalization::Z(z),
    }
}

fn basic_model(r: &mut JavaRandom, model: BasicModel) -> Arc<dyn Similarity> {
    let after = if r.next_int_bounded(2) == 0 {
        AfterEffect::L
    } else {
        AfterEffect::B
    };
    Arc::new(DfrSimilarity::new(model, after, normalization(r)).expect("valid"))
}

fn distribution(r: &mut JavaRandom, distribution: Distribution) -> Arc<dyn Similarity> {
    let lambda = if r.next_boolean() {
        Lambda::DF
    } else {
        Lambda::TTF
    };
    Arc::new(IbSimilarity::new(distribution, lambda, normalization(r)).expect("valid"))
}

/// `AxiomaticTestCase.getSimilarity`: s, queryLen, k drawn; the subclass
/// keeps what its constructor takes (F1/F2: queryLen 1; LOG and F3EXP's
/// test: k 0.35).
fn axiomatic(r: &mut JavaRandom, variant: AxiomaticVariant) -> Arc<dyn Similarity> {
    let s = unit(r);
    let query_len = match r.next_int_bounded(4) {
        0 => 0,
        1 => 1,
        2 => i32::MAX,
        _ => r.next_int_bounded(i32::MAX),
    };
    let k = unit(r);
    let (query_len, k) = match variant {
        AxiomaticVariant::F1Exp | AxiomaticVariant::F2Exp => (1, k),
        AxiomaticVariant::F1Log | AxiomaticVariant::F2Log => (1, 0.35),
        AxiomaticVariant::F3Exp | AxiomaticVariant::F3Log => (query_len, 0.35),
    };
    Arc::new(AxiomaticSimilarity::new(variant, true, s, query_len, k).expect("valid"))
}

/// Every subclass, in the Java twin's order.
pub const CASES: [(&str, GetSimilarity); 22] = [
    ("BM25", |r| {
        let k1 = unbounded(r);
        let b = unit(r);
        Arc::new(Bm25Similarity::new(k1, b, true).expect("valid"))
    }),
    ("Boolean", |_| Arc::new(BooleanSimilarity)),
    ("Classic", |_| Arc::new(ClassicSimilarity::default())),
    ("IndependenceChiSquared", |_| {
        Arc::new(DfiSimilarity::new(Independence::ChiSquared))
    }),
    ("IndependenceSaturated", |_| {
        Arc::new(DfiSimilarity::new(Independence::Saturated))
    }),
    ("IndependenceStandardized", |_| {
        Arc::new(DfiSimilarity::new(Independence::Standardized))
    }),
    ("IndriDirichlet", |r| {
        // `IndriDirichletSimilarity(mu)`: the default collection model
        Arc::new(IndriDirichletSimilarity::new(
            CollectionModel::Default,
            true,
            unbounded(r),
        ))
    }),
    ("LMDirichlet", |r| {
        Arc::new(
            LmDirichletSimilarity::new(CollectionModel::Default, true, unbounded(r))
                .expect("valid"),
        )
    }),
    ("LMJelinekMercer", |r| {
        let lambda = match r.next_int_bounded(3) {
            0 => FLOAT_MIN_VALUE,
            1 => 1.0,
            _ => r.next_float(),
        };
        Arc::new(
            LmJelinekMercerSimilarity::new(CollectionModel::Default, true, lambda).expect("valid"),
        )
    }),
    ("RawTF", |_| Arc::new(RawTfSimilarity::default())),
    ("AxiomaticF1EXP", |r| axiomatic(r, AxiomaticVariant::F1Exp)),
    ("AxiomaticF1LOG", |r| axiomatic(r, AxiomaticVariant::F1Log)),
    ("AxiomaticF2EXP", |r| axiomatic(r, AxiomaticVariant::F2Exp)),
    ("AxiomaticF2LOG", |r| axiomatic(r, AxiomaticVariant::F2Log)),
    ("AxiomaticF3EXP", |r| axiomatic(r, AxiomaticVariant::F3Exp)),
    ("AxiomaticF3LOG", |r| axiomatic(r, AxiomaticVariant::F3Log)),
    ("BasicModelG", |r| basic_model(r, BasicModel::G)),
    ("BasicModelIF", |r| basic_model(r, BasicModel::IF)),
    ("BasicModelIn", |r| basic_model(r, BasicModel::In)),
    ("BasicModelIne", |r| basic_model(r, BasicModel::Ine)),
    ("DistributionLL", |r| distribution(r, Distribution::LL)),
    ("DistributionSPL", |r| distribution(r, Distribution::SPL)),
];

/// The subclass called `name`.
pub fn get(name: &str) -> GetSimilarity {
    CASES
        .iter()
        .find(|(n, _)| *n == name)
        .unwrap_or_else(|| panic!("no similarity test {name}"))
        .1
}
