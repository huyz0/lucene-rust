//! The disabling wrapper must select the default bulk loop, preserve query
//! delegation and agree with ordinary searches over deleted, segmented data.

use std::sync::Arc;

use lucene_analysis::core_analysis::WhitespaceAnalyzer;
use lucene_analysis::Analyzer;
use lucene_index::document::{Document, Store, StringField, TextField};
use lucene_search::collector::{ScoreMode, ScoringCollector};
use lucene_search::query::{BooleanQuery, Clause, MatchAllDocsQuery, PhraseQuery, TermQuery};
use lucene_search::{query_visitor, segment_cacheable, test_support};
use lucene_test_framework::search::disabling_bulk_scorer_query::wrap;
use lucene_test_framework::search::test_index::TestIndex;

fn index(deleted: bool) -> TestIndex {
    let segments = [vec!["a b", "a a", "b"], vec!["a b a", "c", "a"]]
        .into_iter()
        .enumerate()
        .map(|(segment, texts)| {
            texts
                .into_iter()
                .enumerate()
                .map(|(i, text)| {
                    let mut doc = Document::new();
                    doc.add(StringField::new("id", format!("{segment}:{i}"), Store::No));
                    doc.add(TextField::new("body", text, Store::No));
                    doc
                })
                .collect()
        })
        .collect::<Vec<_>>();
    TestIndex::build(
        Arc::new(Analyzer::new(WhitespaceAnalyzer::default())),
        &segments,
        if deleted { &[("id", "0:1")] } else { &[] },
        &["body"],
    )
}

fn top_level(clause: Clause) -> BooleanQuery {
    BooleanQuery {
        must: vec![clause],
        ..Default::default()
    }
}

struct Hits {
    mode: ScoreMode,
    values: Vec<(i32, u32)>,
}

impl ScoringCollector for Hits {
    fn collect(&mut self, doc: i32, score: f32) {
        self.values.push((
            doc,
            if self.mode.needs_scores() {
                score.to_bits()
            } else {
                0
            },
        ));
    }

    fn score_mode(&self) -> ScoreMode {
        self.mode
    }
}

#[test]
fn selects_default_bulk_scorer_and_preserves_live_hits_and_scores() {
    index(true).with_searcher(|searcher| {
        let term = Clause::Term(TermQuery::new("body", "a"));
        for (query, default) in [
            (top_level(term.clone()), false),
            (top_level(wrap(term.clone())), true),
        ] {
            test_support::with_leaf(searcher, &query, 0, |leaf| {
                let bulk = leaf.bulk_scorer(ScoreMode::Complete)?.unwrap();
                assert_eq!(bulk.uses_default_scorer(), default);
                Ok(())
            })
            .unwrap();
        }
        for clause in [
            term,
            TermQuery::new("body", "absent").into(),
            MatchAllDocsQuery::new(6).into(),
            PhraseQuery::new("body", [b"a".to_vec(), b"b".to_vec()]).into(),
            BooleanQuery {
                should: vec![
                    TermQuery::new("body", "a").into(),
                    TermQuery::new("body", "b").into(),
                ],
                ..Default::default()
            }
            .into(),
            test_support::TestQuery::random_approximation(
                TermQuery::new("body", "a"),
                lucene_util::java_random::JavaRandom::new(73),
            ),
        ] {
            let ordinary = top_level(clause.clone());
            let wrapped = top_level(wrap(clause));
            for mode in [
                ScoreMode::Complete,
                ScoreMode::CompleteNoScores,
                ScoreMode::TopScores,
                ScoreMode::TopDocs,
                ScoreMode::TopDocsWithScores,
            ] {
                let mut expected = Hits {
                    mode,
                    values: vec![],
                };
                let mut actual = Hits {
                    mode,
                    values: vec![],
                };
                searcher.search_collector(&ordinary, &mut expected).unwrap();
                searcher.search_collector(&wrapped, &mut actual).unwrap();
                assert_eq!(actual.values, expected.values, "{mode:?}: {ordinary:?}");
            }
            assert_eq!(
                searcher.count(&ordinary).unwrap(),
                searcher.count(&wrapped).unwrap()
            );
            assert_eq!(
                searcher.search(&ordinary, 10).unwrap(),
                searcher.search(&wrapped, 10).unwrap()
            );
            // QueryUtils compares raw scorer iteration to collected hits and
            // currently assumes no deletions. Live-document equivalence is
            // checked above on the deleted index; scorer contracts use all docs.
            index(false).with_searcher(|contract_searcher| {
                lucene_test_framework::search::query_utils::check_with_searcher(
                    &mut lucene_util::java_random::JavaRandom::new(123),
                    &wrapped,
                    contract_searcher,
                )
                .unwrap();
            });
        }
    });
}

#[test]
fn delegates_terms_cacheability_rewrite_explanations_and_matches() {
    let clause = Clause::Term(TermQuery::new("body", "a"));
    let wrapped = wrap(clause.clone());
    assert_eq!(
        query_visitor::extract_terms(&wrapped),
        query_visitor::extract_terms(&clause)
    );
    assert!(segment_cacheable::is_cacheable(&wrapped, None));
    assert_eq!(wrap(top_level(clause.clone())).rewrite(), wrapped);
    let approximated = test_support::TestQuery::random_approximation(
        clause,
        lucene_util::java_random::JavaRandom::new(19),
    );
    assert!(!segment_cacheable::is_cacheable(&wrap(approximated), None));
    index(false).with_searcher(|searcher| {
        lucene_test_framework::search::query_utils::check_with_searcher(
            &mut lucene_util::java_random::JavaRandom::new(101),
            &top_level(wrapped),
            searcher,
        )
        .unwrap();
    });
}

#[test]
fn wrapper_preserves_inner_and_has_distinct_value_identity() {
    let query = Clause::Term(TermQuery::new("body", "term"));
    let wrapped = wrap(query.clone());
    assert_eq!(wrapped, wrap(query.clone()));
    assert_ne!(wrapped, query);
    assert_ne!(wrapped, wrap(TermQuery::new("body", "other")));
    assert!(matches!(wrapped, Clause::Extended(q) if matches!(
        q.as_ref(),
        lucene_search::extended_query::ExtendedQuery::Test(q)
            if q.query() == &query && q.name() == "DisablingBulkScorerQuery"
    )));
}
