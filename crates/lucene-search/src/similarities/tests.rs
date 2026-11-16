#![cfg(test)]
use super::*;

/// A source that answers with the statistics it is given back.
struct Echo;

impl crate::index_searcher::StatisticsSource for Echo {
    fn term_statistics(
        &self,
        _field: &str,
        _term: &[u8],
        local: TermStatistics,
    ) -> crate::Result<TermStatistics> {
        Ok(local)
    }

    fn collection_statistics(
        &self,
        _field: &str,
        local: Option<CollectionStatistics>,
    ) -> crate::Result<Option<CollectionStatistics>> {
        Ok(local)
    }
}

/// The BM25 fast path scores with the norms' own average length, so a per-field
/// similarity that replaces the statistics (an aggregated DFS) is not it.
#[test]
fn a_statistics_source_leaves_the_default_bm25_fast_path() {
    let plain = PerFieldSimilarity::new(Arc::new(Bm25Similarity::default()));
    assert!(plain.is_default_bm25());
    assert!(!plain.with_statistics(Arc::new(Echo)).is_default_bm25());
}

fn coll() -> CollectionStatistics {
    CollectionStatistics::new(100, 90, 5000, 900).unwrap()
}

fn term() -> TermStatistics {
    TermStatistics::new(10, 40).unwrap()
}

/// `MultiSimScorer.explain`: "sum of:" each sub-scorer's explanation,
/// valued as the summed score -- for `MultiSimilarity` and for a
/// `SimilarityBase` over several terms.
#[test]
fn multi_scorers_explain_as_a_sum() {
    let freq = || Explanation::match_(3.0, "freq, occurrences of term within document");
    let multi = MultiSimilarity::new(vec![
        Arc::new(Bm25Similarity::default()),
        Arc::new(BooleanSimilarity),
    ])
    .unwrap();
    let scorer = multi.scorer("f", 2.0, &coll(), &[term()]);
    let e = scorer.explain(freq(), 7);
    assert_eq!(e.description, "sum of:");
    assert_eq!(e.value, scorer.score(3.0, 7));
    assert_eq!(e.details.len(), 2);
    assert!(e.details[0]
        .description
        .starts_with("score(freq=3.0), computed as boost * idf * tf"));
    assert_eq!(
        e.details[1].description,
        "score(BooleanWeight), computed from:"
    );

    let dfi = DfiSimilarity::new(Independence::Saturated);
    let scorer = dfi.scorer("f", 1.0, &coll(), &[term(), term()]);
    let e = scorer.explain(freq(), 7);
    assert_eq!(e.description, "sum of:");
    assert_eq!(e.value, scorer.score(3.0, 7));
    assert!(e.details[0]
        .description
        .starts_with("score(DFISimilarity, freq=3.0)"));
}

/// `LegacyBM25Similarity` scores BM25 at `boost * (1 + k1)`.
#[test]
fn legacy_bm25_scales_the_boost_by_k1_plus_one() {
    let legacy = LegacyBm25Similarity::new(1.5, 0.6, true).unwrap();
    let bm25 = Bm25Similarity::new(1.5, 0.6, true).unwrap();
    let (l, b) = (
        legacy.scorer("f", 2.0, &coll(), &[term()]),
        bm25.scorer("f", 2.0 * 2.5, &coll(), &[term()]),
    );
    for (freq, norm) in [(1.0, 1), (3.0, 7), (9.0, 200)] {
        assert_eq!(l.score(freq, norm).to_bits(), b.score(freq, norm).to_bits());
    }
    assert_eq!(legacy.simple_name(), "LegacyBM25Similarity");
    assert!(!legacy.is_default_bm25());
}

/// Each similarity reports its Java class's simple name, which an
/// explanation prints.
#[test]
fn simple_names_are_the_java_classes() {
    use AxiomaticVariant::*;
    let none = Normalization::None;
    let mut sims: Vec<(Arc<dyn Similarity>, &str)> = vec![
        (Arc::new(Bm25Similarity::default()), "BM25Similarity"),
        (Arc::new(ClassicSimilarity::default()), "ClassicSimilarity"),
        (Arc::new(BooleanSimilarity), "BooleanSimilarity"),
        (Arc::new(RawTfSimilarity::default()), "RawTFSimilarity"),
        (
            Arc::new(DfrSimilarity::new(BasicModel::G, AfterEffect::B, none).unwrap()),
            "DFRSimilarity",
        ),
        (
            Arc::new(IbSimilarity::new(Distribution::LL, Lambda::DF, none).unwrap()),
            "IBSimilarity",
        ),
        (
            Arc::new(DfiSimilarity::new(Independence::Saturated)),
            "DFISimilarity",
        ),
        (
            Arc::new(LmDirichletSimilarity::default()),
            "LMDirichletSimilarity",
        ),
        (
            Arc::new(LmJelinekMercerSimilarity::new(CollectionModel::Default, true, 0.5).unwrap()),
            "LMJelinekMercerSimilarity",
        ),
        (
            Arc::new(IndriDirichletSimilarity::default()),
            "IndriDirichletSimilarity",
        ),
    ];
    for (v, name) in [
        (F1Exp, "AxiomaticF1EXP"),
        (F1Log, "AxiomaticF1LOG"),
        (F2Exp, "AxiomaticF2EXP"),
        (F2Log, "AxiomaticF2LOG"),
        (F3Exp, "AxiomaticF3EXP"),
        (F3Log, "AxiomaticF3LOG"),
    ] {
        sims.push((
            Arc::new(AxiomaticSimilarity::new(v, true, 0.25, 1, 0.35).unwrap()),
            name,
        ));
    }
    let bm25: Arc<dyn Similarity> = Arc::new(Bm25Similarity::default());
    sims.push((
        Arc::new(MultiSimilarity::new(vec![Arc::clone(&bm25)]).unwrap()),
        "MultiSimilarity",
    ));
    sims.push((
        Arc::new(PerFieldSimilarity::new(bm25)),
        "PerFieldSimilarityWrapper",
    ));
    for (sim, name) in &sims {
        assert_eq!(sim.simple_name(), *name);
    }
}

/// The constructors refuse what Java's refuse, with Java's messages.
#[test]
fn arguments_are_checked_as_java_checks_them() {
    assert!(CollectionStatistics::new(0, 0, 0, 0).is_err());
    assert!(CollectionStatistics::new(10, 11, 20, 20).is_err());
    assert!(CollectionStatistics::new(10, 5, 20, 4).is_err());
    assert!(CollectionStatistics::new(10, 5, 4, 5).is_err());
    assert!(TermStatistics::new(0, 1).is_err());
    assert!(TermStatistics::new(5, 4).is_err());
    assert!(Bm25Similarity::new(-1.0, 0.5, true).is_err());
    assert!(Bm25Similarity::new(1.2, 1.5, true).is_err());
    assert!(Bm25Similarity::new(f32::INFINITY, 0.5, true).is_err());
    for bad in [
        Normalization::H1(-1.0),
        Normalization::H2(f32::NAN),
        Normalization::H3(-0.5),
        Normalization::Z(0.5),
        Normalization::Z(0.0),
    ] {
        assert!(
            DfrSimilarity::new(BasicModel::G, AfterEffect::B, bad).is_err(),
            "{bad:?}"
        );
        assert!(IbSimilarity::new(Distribution::LL, Lambda::DF, bad).is_err());
    }
    assert!(LmDirichletSimilarity::new(CollectionModel::Default, true, -1.0).is_err());
    assert!(LmJelinekMercerSimilarity::new(CollectionModel::Default, true, 0.0).is_err());
    assert!(LmJelinekMercerSimilarity::new(CollectionModel::Default, true, 1.5).is_err());
    use AxiomaticVariant::F1Exp;
    assert!(AxiomaticSimilarity::new(F1Exp, true, 1.5, 1, 0.35).is_err());
    assert!(AxiomaticSimilarity::new(F1Exp, true, 0.25, 1, -0.1).is_err());
    assert!(AxiomaticSimilarity::new(F1Exp, true, 0.25, -1, 0.35).is_err());
    assert!(MultiSimilarity::new(vec![]).is_err());
}

/// `PerFieldSimilarityWrapper` scores and norms with the field's own
/// similarity, and falls back to the default for any other field.
#[test]
fn per_field_dispatches_by_field_name() {
    let per_field = PerFieldSimilarity::new(Arc::new(Bm25Similarity::default()))
        .with_field("title", Arc::new(ClassicSimilarity::new(false)));
    let (c, t) = (coll(), [term()]);
    let classic = ClassicSimilarity::new(false).scorer("title", 1.0, &c, &t);
    let bm25 = Bm25Similarity::default().scorer("body", 1.0, &c, &t);
    assert_eq!(
        per_field
            .scorer("title", 1.0, &c, &t)
            .score(3.0, 7)
            .to_bits(),
        classic.score(3.0, 7).to_bits()
    );
    assert_eq!(
        per_field
            .scorer("body", 1.0, &c, &t)
            .score(3.0, 7)
            .to_bits(),
        bm25.score(3.0, 7).to_bits()
    );
    let state = FieldInvertState {
        docs_only: false,
        length: 10,
        num_overlap: 4,
        unique_term_count: 3,
        ..FieldInvertState::default()
    };
    // Classic without discounting counts the overlaps; BM25 does not.
    assert_ne!(
        per_field.compute_norm("title", &state),
        per_field.compute_norm("body", &state)
    );
    assert!(!per_field.is_default_bm25());
    assert!(PerFieldSimilarity::new(Arc::new(Bm25Similarity::default())).is_default_bm25());
}

#[test]
fn default_bm25_is_recognised_and_bulk_scoring_is_per_document() {
    assert!(Bm25Similarity::default().is_default_bm25());
    assert!(!Bm25Similarity::new(2.0, 0.75, true)
        .unwrap()
        .is_default_bm25());
    assert!(!ClassicSimilarity::default().is_default_bm25());
    let s = DfiSimilarity::new(Independence::Saturated).scorer("f", 1.0, &coll(), &[term()]);
    let mut out = Vec::new();
    s.score_bulk(&[1.0, 5.0, 30.0], &[1, 40, -3], &mut out);
    let one: Vec<u32> = [(1.0, 1), (5.0, 40), (30.0, -3)]
        .iter()
        .map(|&(f, n)| s.score(f, n).to_bits())
        .collect();
    assert_eq!(out.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), one);
    assert_eq!(java_max(0.0, f64::NAN).to_bits(), f64::NAN.to_bits());
    assert_eq!(java_max(-0.0, 0.0).to_bits(), 0.0f64.to_bits());
    assert_eq!(java_max(0.0, -0.0).to_bits(), 0.0f64.to_bits());
    assert_eq!(java_max(1.0, 2.0), 2.0);
    assert_eq!(java_max(3.0, 2.0), 3.0);
}
