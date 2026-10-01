//! Every similarity in Lucene 10.5.0's `search/similarities`, differentially
//! against Java: `fixtures/src/GenSimilarities.java` scored random statistics
//! and `(freq, norm)` pairs with each configuration, and recorded the raw
//! bits; this scores the same cases with `lucene_search::similarities` and
//! requires the same bits (any `NaN` for a `NaN`: Java's `(float)` of a
//! `double` `NaN` does not pin a payload).

use std::sync::Arc;

use lucene_search::similarities::*;

fn root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/similarities")
}

/// The Rust counterpart of `GenSimilarities.sims()`, by the same names.
fn sim(name: &str) -> Arc<dyn Similarity> {
    let norm = |n: &str| match n {
        "none" => Normalization::None,
        "H1" => Normalization::H1_DEFAULT,
        "H1_2.5" => Normalization::H1(2.5),
        "H2" => Normalization::H2_DEFAULT,
        "H2_0.3" => Normalization::H2(0.3),
        "H3" => Normalization::H3_DEFAULT,
        "H3_100" => Normalization::H3(100.0),
        "Z" => Normalization::Z_DEFAULT,
        "Z_0.1" => Normalization::Z(0.1),
        other => panic!("normalization {other}"),
    };
    let parts: Vec<&str> = name.split('_').collect();
    match parts.as_slice() {
        ["bm25"] => Arc::new(Bm25Similarity::default()),
        ["bm25", "k2", "b03"] => Arc::new(Bm25Similarity::new(2.0, 0.3, true).unwrap()),
        ["bm25", "k0", "b1"] => Arc::new(Bm25Similarity::new(0.0, 1.0, true).unwrap()),
        ["bm25", "nodiscount"] => Arc::new(Bm25Similarity::new(1.2, 0.75, false).unwrap()),
        ["classic"] => Arc::new(ClassicSimilarity::default()),
        ["classic", "nodiscount"] => Arc::new(ClassicSimilarity::new(false)),
        ["boolean"] => Arc::new(BooleanSimilarity),
        ["rawtf"] => Arc::new(RawTfSimilarity::default()),
        ["dfr", bm, ae, n @ ..] => {
            let bm = match *bm {
                "G" => BasicModel::G,
                "IF" => BasicModel::IF,
                "In" => BasicModel::In,
                "Ine" => BasicModel::Ine,
                other => panic!("basic model {other}"),
            };
            let ae = match *ae {
                "B" => AfterEffect::B,
                "L" => AfterEffect::L,
                other => panic!("after effect {other}"),
            };
            Arc::new(DfrSimilarity::new(bm, ae, norm(&n.join("_"))).unwrap())
        }
        ["ib", d, l, n @ ..] => {
            let d = match *d {
                "LL" => Distribution::LL,
                "SPL" => Distribution::SPL,
                other => panic!("distribution {other}"),
            };
            let l = match *l {
                "DF" => Lambda::DF,
                "TTF" => Lambda::TTF,
                other => panic!("lambda {other}"),
            };
            Arc::new(IbSimilarity::new(d, l, norm(&n.join("_"))).unwrap())
        }
        ["dfi", "standardized"] => Arc::new(DfiSimilarity::new(Independence::Standardized)),
        ["dfi", "saturated"] => Arc::new(DfiSimilarity::new(Independence::Saturated)),
        ["dfi", "chisquared"] => Arc::new(DfiSimilarity::new(Independence::ChiSquared)),
        ["lmdirichlet"] => Arc::new(LmDirichletSimilarity::default()),
        ["lmdirichlet", "100"] => {
            Arc::new(LmDirichletSimilarity::new(CollectionModel::Default, true, 100.0).unwrap())
        }
        ["lmdirichlet", "indri", "500"] => {
            Arc::new(LmDirichletSimilarity::new(CollectionModel::Indri, true, 500.0).unwrap())
        }
        ["lmjm", l] => Arc::new(
            LmJelinekMercerSimilarity::new(CollectionModel::Default, true, l.parse().unwrap())
                .unwrap(),
        ),
        ["indri"] => Arc::new(IndriDirichletSimilarity::default()),
        // `IndriDirichletSimilarity(float mu)`: the default collection model.
        ["indri", "100"] => Arc::new(IndriDirichletSimilarity::new(
            CollectionModel::Default,
            true,
            100.0,
        )),
        ["ax", v, rest @ ..] => {
            use AxiomaticVariant::*;
            let (variant, s, q, k) = match (*v, rest) {
                ("f1exp", []) => (F1Exp, 0.25, 1, 0.35),
                ("f1exp", ["0.5", "0.2"]) => (F1Exp, 0.5, 1, 0.2),
                ("f1log", []) => (F1Log, 0.1, 1, 0.35),
                ("f2exp", []) => (F2Exp, 0.5, 1, 0.2),
                ("f2log", []) => (F2Log, 0.25, 1, 0.35),
                ("f3exp", []) => (F3Exp, 0.25, 3, 0.35),
                ("f3log", []) => (F3Log, 0.4, 2, 0.35),
                other => panic!("axiomatic {other:?}"),
            };
            Arc::new(AxiomaticSimilarity::new(variant, true, s, q, k).unwrap())
        }
        ["multi"] => Arc::new(
            MultiSimilarity::new(vec![
                Arc::new(Bm25Similarity::default()),
                Arc::new(ClassicSimilarity::default()),
                Arc::new(
                    DfrSimilarity::new(BasicModel::G, AfterEffect::B, Normalization::H2_DEFAULT)
                        .unwrap(),
                ),
            ])
            .unwrap(),
        ),
        other => panic!("unknown similarity {other:?}"),
    }
}

fn f32_hex(s: &str) -> f32 {
    f32::from_bits(u32::from_str_radix(s, 16).unwrap())
}

#[test]
fn every_similarity_scores_bit_for_bit_with_lucene() {
    let text = std::fs::read_to_string(root().join("scores.tsv")).unwrap();
    let mut failures = Vec::new();
    let mut cases = 0;
    let mut sims = std::collections::BTreeSet::new();
    for line in text.lines() {
        let c: Vec<&str> = line.split('\t').collect();
        let [name, coll, terms, boost, freq, norm, want] = c.as_slice() else {
            panic!("bad line {line}");
        };
        let n: Vec<i64> = coll.split(',').map(|v| v.parse().unwrap()).collect();
        let collection = CollectionStatistics::new(n[0], n[1], n[2], n[3]).unwrap();
        let terms: Vec<TermStatistics> = terms
            .split(';')
            .map(|t| {
                let (df, ttf) = t.split_once(':').unwrap();
                TermStatistics::new(df.parse().unwrap(), ttf.parse().unwrap()).unwrap()
            })
            .collect();
        let scorer = sim(name).scorer("f", f32_hex(boost), &collection, &terms);
        let got = scorer.score(f32_hex(freq), norm.parse().unwrap());
        let want = f32_hex(want);
        cases += 1;
        sims.insert(name.to_string());
        let same = got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan());
        if !same {
            failures.push(format!("{line}\n    rust {:08x} ({got})", got.to_bits()));
        }
    }
    assert!(
        cases > 30_000 && sims.len() > 130,
        "{cases} cases, {} similarities",
        sims.len()
    );
    assert!(
        failures.is_empty(),
        "{} of {cases} differ:\n{}",
        failures.len(),
        failures
            .iter()
            .take(40)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[test]
fn compute_norm_matches_lucene() {
    let text = std::fs::read_to_string(root().join("norms.tsv")).unwrap();
    let mut n = 0;
    for line in text.lines() {
        let c: Vec<&str> = line.split('\t').collect();
        let [name, docs_only, length, overlap, unique, want] = c.as_slice() else {
            panic!("bad line {line}");
        };
        let state = FieldInvertState {
            docs_only: docs_only.parse().unwrap(),
            length: length.parse().unwrap(),
            num_overlap: overlap.parse().unwrap(),
            unique_term_count: unique.parse().unwrap(),
            ..FieldInvertState::default()
        };
        assert_eq!(
            sim(name).compute_norm("f", &state),
            want.parse::<i64>().unwrap(),
            "{line}"
        );
        n += 1;
    }
    assert!(n >= 1500, "{n}");
}
