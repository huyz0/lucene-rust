//! The extended queries' scorers agree with themselves across score modes,
//! over `GenFunction`'s four-segment index with deletions: a top-10 search
//! (`TOP_SCORES`: block-max skipping through each scorer's `max_score` and
//! `advance_shallow`) keeps exactly the ten best of an exhaustive
//! `COMPLETE` scoring, score bits included, and a count with the query as a
//! filter (`COMPLETE_NO_SCORES`, where several of them rewrite to a plain
//! disjunction) counts exactly the exhaustive scoring's live matches. Lucene's
//! own scores for these queries are pinned by the M7 differential suites;
//! this checks the paths those suites' shapes do not reach -- two-phase
//! sub-clauses, weighted fields, one- and no-clause fusions.

// Test fixtures' own arithmetic -- see `docs/arithmetic-gate.md`'s "Test code".
#![allow(clippy::arithmetic_side_effects)]

use lucene_search::directory_reader::DirectoryReader;
use lucene_search::extended_query::{
    BayesianScoreQuery, BlendedRewrite, BlendedTermQuery, CombinedFieldQuery, IndriAndQuery,
    LogOddsFusionQuery, MultiTermQuery, MultiTermSource, RewriteMethod, SynonymQuery,
};
use lucene_search::index_searcher::{IndexSearcher, SegmentNorms};
use lucene_search::query::{
    BooleanQuery, Clause, MultiPhraseQuery, PhraseQuery, PrefixQuery, RegexpQuery, TermQuery,
    WildcardQuery,
};
use lucene_store::FsDirectory;

fn t(w: &str) -> Clause {
    Clause::Term(TermQuery::new("body", w.as_bytes().to_vec()))
}

fn phrase(a: &str, b: &str, slop: u32) -> Clause {
    Clause::from(PhraseQuery::new("body", [a, b]).with_slop(slop))
}

fn queries() -> Vec<(&'static str, Clause)> {
    vec![
        (
            "synonym",
            SynonymQuery::new("body", [(b"red".to_vec(), 1.0), (b"blue".to_vec(), 0.5)])
                .unwrap()
                .into(),
        ),
        (
            "combined",
            CombinedFieldQuery::new(b"red".to_vec(), [("body", 1.0)])
                .unwrap()
                .into(),
        ),
        (
            "combined weighted",
            CombinedFieldQuery::new(b"green".to_vec(), [("body", 2.5)])
                .unwrap()
                .into(),
        ),
        (
            "blended dismax",
            BlendedTermQuery::new(
                [
                    ("body", b"red".to_vec(), 1.0),
                    ("body", b"old".to_vec(), 2.0),
                ],
                BlendedRewrite::DisjunctionMax(0.1),
            )
            .unwrap()
            .into(),
        ),
        (
            "blended boolean",
            BlendedTermQuery::new(
                [
                    ("body", b"fast".to_vec(), 1.0),
                    ("body", b"slow".to_vec(), 1.0),
                ],
                BlendedRewrite::Boolean,
            )
            .unwrap()
            .into(),
        ),
        (
            "blended one",
            BlendedTermQuery::new([("body", b"big".to_vec(), 1.0)], BlendedRewrite::default())
                .unwrap()
                .into(),
        ),
        (
            "multi-phrase one position",
            Clause::MultiPhrase(MultiPhraseQuery::new(
                "body",
                [vec![b"red".to_vec(), b"small".to_vec()]],
            )),
        ),
        (
            "phrase one term",
            Clause::from(PhraseQuery::new("body", ["blue"])),
        ),
        (
            "positional phrase one term",
            Clause::from(PhraseQuery::with_positions("body", [("blue", 3)]).unwrap()),
        ),
        (
            "positional phrase with a gap",
            Clause::from(PhraseQuery::with_positions("body", [("red", 0), ("old", 2)]).unwrap()),
        ),
        (
            "prefix blended",
            MultiTermQuery::new(
                MultiTermSource::Prefix(PrefixQuery::new("body", b"s".to_vec())),
                RewriteMethod::ConstantScoreBlended,
            )
            .into(),
        ),
        (
            "wildcard blended",
            MultiTermQuery::new(
                MultiTermSource::Wildcard(WildcardQuery::new("body", b"*e*".to_vec())),
                RewriteMethod::ConstantScoreBlended,
            )
            .into(),
        ),
        (
            "regexp blended",
            MultiTermQuery::new(
                MultiTermSource::Regexp(RegexpQuery::new("body", "[bg].*")),
                RewriteMethod::ConstantScoreBlended,
            )
            .into(),
        ),
        (
            "top terms",
            MultiTermQuery::new(
                MultiTermSource::Prefix(PrefixQuery::new("body", b"".to_vec())),
                RewriteMethod::TopTermsScoringBoolean(2),
            )
            .into(),
        ),
        (
            "indri and",
            IndriAndQuery::new([t("red"), phrase("big", "old", 2)]).into(),
        ),
        (
            "fusion two-phase",
            LogOddsFusionQuery::new([t("red"), phrase("red", "blue", 3)], 0.5, None, None)
                .unwrap()
                .into(),
        ),
        (
            "fusion weighted",
            LogOddsFusionQuery::new(
                [t("green"), t("slow"), phrase("old", "new", 1)],
                0.0,
                Some(vec![0.5, 0.25, 0.25]),
                None,
            )
            .unwrap()
            .into(),
        ),
        (
            "fusion one",
            LogOddsFusionQuery::new([t("fast")], 1.0, None, None)
                .unwrap()
                .into(),
        ),
        (
            "fusion none",
            LogOddsFusionQuery::new(Vec::<Clause>::new(), 0.5, None, None)
                .unwrap()
                .into(),
        ),
        (
            "bayesian phrase",
            BayesianScoreQuery::new(phrase("big", "small", 4), 1.5, 0.5, 0.1)
                .unwrap()
                .into(),
        ),
        (
            "bayesian term",
            BayesianScoreQuery::new(t("red"), 0.8, -0.25, 0.0)
                .unwrap()
                .into(),
        ),
    ]
}

/// `q`'s live matches scored exhaustively (`COMPLETE`), best first; the
/// top `n` searches (`TOP_SCORES`) and the count (`COMPLETE_NO_SCORES`)
/// checked against them.
fn check(
    searcher: &IndexSearcher<'_, '_>,
    segments: &[lucene_search::multi_segment::OpenSegment<'_>],
    q: &BooleanQuery,
    what: &str,
) -> Vec<(i32, f32)> {
    let mut all: Vec<(i32, f32)> = Vec::new();
    for (i, seg) in segments.iter().enumerate() {
        for (doc, score) in searcher.leaf_scores(q, i, false).unwrap() {
            all.push((seg.doc_base + doc, score));
        }
    }
    all.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    // Small `n`s publish a minimum competitive score early, which is what
    // makes the scorers skip by their `max_score`.
    for n in [1, 3, 10] {
        let top = searcher.search(q, n).unwrap();
        let got: Vec<(i32, u32)> = top
            .score_docs
            .iter()
            .map(|d| (d.doc, d.score.to_bits()))
            .collect();
        let want: Vec<(i32, u32)> = all.iter().take(n).map(|&(d, s)| (d, s.to_bits())).collect();
        assert_eq!(got, want, "{what}: the top {n}");
    }
    assert_eq!(
        searcher.count(q).unwrap(),
        all.len() as u64,
        "{what}: the count"
    );
    all
}

#[test]
fn extended_queries_agree_across_score_modes() {
    let dir =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/function/index");
    let reader = DirectoryReader::open(&FsDirectory::open(dir)).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let owned = reader.field_norms_by_field(&["body".to_string()]);
    let norms: Vec<SegmentNorms<'_, '_>> = owned.iter().map(Some).collect();
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();
    let mut matched_any = 0;
    for (name, clause) in queries() {
        // Alone, and as one clause of a disjunction and of a conjunction --
        // where its scorer is a sub-scorer the parent advances, skips by
        // block maxima and asks to confirm two-phase matches.
        let boosted = Clause::Boost(Box::new(lucene_search::query::BoostQuery::new(
            clause.clone(),
            2.0,
        )));
        let shapes = [
            (
                "alone",
                BooleanQuery {
                    must: vec![clause.clone()],
                    ..Default::default()
                },
            ),
            (
                "or",
                BooleanQuery {
                    should: vec![boosted, t("slow"), t("new")],
                    ..Default::default()
                },
            ),
            (
                "and",
                BooleanQuery {
                    must: vec![clause.clone(), t("big")],
                    ..Default::default()
                },
            ),
        ];
        for (shape, q) in shapes {
            let all = check(&searcher, &segments, &q, &format!("{name} {shape}"));
            if shape == "alone" && !all.is_empty() {
                matched_any += 1;
            }
            if shape == "alone" {
                let filtered = BooleanQuery {
                    filter: vec![clause.clone()],
                    ..Default::default()
                };
                assert_eq!(
                    searcher.count(&filtered).unwrap(),
                    all.len() as u64,
                    "{name}: the count as a filter"
                );
            }
        }
    }
    assert!(
        matched_any >= 18,
        "only {matched_any} queries matched anything"
    );
}
