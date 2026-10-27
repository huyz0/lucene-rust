//! Calls Java's actual wrapper on the same sources and indexed queries.
use std::cell::RefCell;
use std::sync::Arc;

use lucene_analysis::core_analysis::WhitespaceAnalyzer;
use lucene_analysis::Analyzer;
use lucene_index::document::{Document, Store, StringField};
use lucene_search::collector::{ScoreMode, ScoringCollector};
use lucene_search::query::{
    BooleanQuery, Clause, ConstantScoreQuery, DisjunctionMaxQuery, MatchAllDocsQuery, TermQuery,
};
use lucene_search::test_support::with_leaf;
use lucene_test_framework::search::bulk_scorer_wrapper_scorer::{
    BulkScorer, BulkScorerWrapperScorer, Error, LeafBulkSource, NO_MORE_DOCS,
};
use lucene_test_framework::search::test_index::TestIndex;

const DOCS: [i32; 10] = [0, 1, 3, 4, 9, 21, 22, 70, 71, 95];
const BITS: [u32; 10] = [
    0, 0x80000000, 1, 0x3f800000, 0xbf800000, 0x7f800000, 0xff800000, 0x7fc01234, 0x3eaaaaab,
    0x7f7fffff,
];

struct Source {
    empty: bool,
    index: usize,
    calls: usize,
    fail_at: usize,
    fail_after: usize,
    failure: Arc<std::io::Error>,
    trace: RefCell<Vec<String>>,
    cost: u64,
}
impl Source {
    fn new(label: &str) -> Self {
        Self {
            empty: label == "empty",
            index: 0,
            calls: 0,
            fail_at: match label {
                "failure" => 2,
                "partialFailure" => 1,
                _ => usize::MAX,
            },
            fail_after: if label == "partialFailure" { 2 } else { 0 },
            failure: Arc::new(std::io::Error::other("injected bulk failure")),
            trace: RefCell::new(Vec::new()),
            cost: 0x123456789,
        }
    }
    fn take_trace(&self) -> String {
        std::mem::take(&mut *self.trace.borrow_mut()).join(",")
    }
}
impl BulkScorer for Source {
    type Error = Arc<std::io::Error>;
    fn cost(&self) -> u64 {
        self.cost
    }
    fn score(
        &mut self,
        collector: &mut dyn ScoringCollector,
        min: i32,
        max: i32,
    ) -> Result<i32, Self::Error> {
        self.trace.borrow_mut().push(format!("{min}:{max}"));
        self.calls += 1;
        if self.calls == self.fail_at && self.fail_after == 0 {
            return Err(self.failure.clone());
        }
        let len = if self.empty { 0 } else { DOCS.len() };
        while self.index < len && DOCS[self.index] < min {
            self.index += 1;
        }
        while self.index < len && DOCS[self.index] < max {
            collector.collect(DOCS[self.index], f32::from_bits(BITS[self.index]));
            self.index += 1;
            if self.calls == self.fail_at && self.index == self.fail_after {
                return Err(self.failure.clone());
            }
        }
        Ok(if self.index == len { NO_MORE_DOCS } else { max })
    }
}

#[test]
fn java_cursor_refill_scores_errors_and_shared_iterators() {
    let fixture = include_str!("../../../fixtures/testframework/data/bulk_scorer_wrapper.tsv");
    let mut current = None;
    let mut steps = 0;
    for line in fixture.lines() {
        let c: Vec<_> = line.split('\t').collect();
        match c[0] {
            "initial" => {
                let wrapper =
                    BulkScorerWrapperScorer::new(Source::new(c[1]), c[2].parse().unwrap()).unwrap();
                assert_eq!(wrapper.doc_id(), c[3].parse::<i32>().unwrap());
                assert_eq!(wrapper.doc_id(), c[4].parse::<i32>().unwrap());
                assert_eq!(wrapper.cost(), c[5].parse::<u64>().unwrap());
                assert_eq!(
                    wrapper.max_score(-1).to_bits(),
                    c[6].parse::<u32>().unwrap()
                );
                current = Some(wrapper);
            }
            "step" | "failure" => {
                let wrapper = current.as_mut().unwrap();
                let target = c[2].parse::<i32>().unwrap();
                let result = if target < 0 {
                    wrapper.iterator().next_doc()
                } else {
                    wrapper.iterator().advance(target)
                };
                if c[0] == "failure" {
                    let Error::Source(error) = result.unwrap_err() else {
                        panic!("{line}")
                    };
                    assert_eq!(
                        Arc::ptr_eq(&error, &wrapper.source().failure),
                        c[3] == "true"
                    );
                    assert_eq!(wrapper.doc_id(), c[4].parse::<i32>().unwrap());
                    assert_eq!(wrapper.source().take_trace(), c[5]);
                } else {
                    let doc = result.unwrap();
                    assert_eq!(doc, c[3].parse::<i32>().unwrap(), "{line}");
                    assert_eq!(wrapper.doc_id(), c[4].parse::<i32>().unwrap());
                    assert_eq!(wrapper.iterator().doc_id(), c[5].parse::<i32>().unwrap());
                    if c[6] != "-" {
                        assert_eq!(wrapper.score().to_bits(), c[6].parse::<u32>().unwrap());
                    }
                    assert_eq!(wrapper.source().take_trace(), c[7], "{line}");
                }
                steps += 1;
            }
            "shared" => {
                let mut source = Source::new("shared");
                source.cost = 10;
                let mut wrapper = BulkScorerWrapperScorer::new(source, 7).unwrap();
                let actual = [
                    wrapper.iterator().next_doc().unwrap(),
                    wrapper.iterator().doc_id(),
                    wrapper.iterator().next_doc().unwrap(),
                    wrapper.iterator().doc_id(),
                    wrapper.doc_id(),
                ];
                for (actual, expected) in actual.into_iter().zip(&c[1..6]) {
                    assert_eq!(actual, expected.parse::<i32>().unwrap());
                }
                assert_eq!(wrapper.score().to_bits(), c[6].parse::<u32>().unwrap());
                assert_eq!(wrapper.cost(), 10);
            }
            _ => {}
        }
    }
    assert_eq!(steps, 152);
}

#[test]
fn java_actual_indexed_bulk_scorers() {
    let fixture = include_str!("../../../fixtures/testframework/data/bulk_scorer_wrapper.tsv");
    let rows: Vec<Vec<_>> = fixture
        .lines()
        .filter(|line| line.starts_with("indexed"))
        .map(|line| line.split('\t').collect())
        .collect();
    let docs = (0..97)
        .map(|id| {
            let mut doc = Document::new();
            doc.add(StringField::new(
                "tag",
                if id % 11 == 0 { "hit" } else { "miss" },
                Store::No,
            ));
            doc
        })
        .collect();
    let index = TestIndex::build(
        Arc::new(Analyzer::new(WhitespaceAnalyzer::default())),
        &[docs],
        &[],
        &[],
    );
    let mut row = 0;
    let mut cases = 0;
    while row < rows.len() {
        let header = &rows[row];
        row += 1;
        assert_eq!(header[0], "indexed");
        let clause = if header[1] == "all" {
            Clause::MatchAllDocs(MatchAllDocsQuery::new(97))
        } else {
            Clause::ConstantScore(Box::new(ConstantScoreQuery::new(
                TermQuery::new("tag", "hit"),
                1.0,
            )))
        };
        let query = BooleanQuery {
            must: vec![clause],
            ..Default::default()
        };
        let java_cost = header[4].parse::<u64>().unwrap();
        index.with_searcher(|searcher| {
            with_leaf(searcher, &query, 0, |leaf| {
                let bulk = leaf.bulk_scorer(ScoreMode::Complete)?.unwrap();
                let mut wrapper = BulkScorerWrapperScorer::new(
                    LeafBulkSource::new(bulk),
                    header[2].parse().unwrap(),
                )
                .unwrap();
                assert_eq!(wrapper.cost(), java_cost);
                let mut target = 0;
                loop {
                    let expected = &rows[row];
                    row += 1;
                    assert_eq!(expected[0], "indexedStep");
                    let doc = if header[3] == "true" {
                        wrapper.advance(target)
                    } else {
                        wrapper.next_doc()
                    }
                    .unwrap();
                    assert_eq!(doc, expected[1].parse::<i32>().unwrap());
                    if doc == NO_MORE_DOCS {
                        break;
                    }
                    assert_eq!(
                        wrapper.score().to_bits(),
                        expected[2].parse::<u32>().unwrap()
                    );
                    target = target.max(doc) + 7;
                }
                Ok(())
            })
            .unwrap();
        });
        cases += 1;
    }
    assert_eq!(cases, 16);
}

fn t(term: &str) -> Clause {
    Clause::Term(TermQuery::new("body", term))
}

/// A boolean of `must`, `filter`, `should` and `must_not` clauses.
fn bool_of(
    must: Vec<Clause>,
    filter: Vec<Clause>,
    should: Vec<Clause>,
    not: Vec<Clause>,
) -> Clause {
    Clause::Boolean(Box::new(BooleanQuery {
        must,
        filter,
        should,
        must_not: not,
        ..Default::default()
    }))
}

/// Each query shape's real bulk scorer reports Java's `BulkScorer.cost()`
/// under every score mode, whichever bulk variant scores it.
#[test]
fn java_bulk_scorer_costs() {
    let fixture = include_str!("../../../fixtures/testframework/data/bulk_scorer_wrapper.tsv");
    let docs = (0..3000)
        .map(|id| {
            let mut doc = Document::new();
            for (term, m) in [("a", 2), ("b", 3), ("c", 5), ("d", 7)] {
                if id % m == 0 {
                    doc.add(StringField::new("body", term, Store::No));
                }
            }
            doc.add(StringField::new("body", "z", Store::No));
            doc
        })
        .collect();
    let index = TestIndex::build(
        Arc::new(Analyzer::new(WhitespaceAnalyzer::default())),
        &[docs],
        &[],
        &[],
    );
    let (a, b, c, d) = (t("a"), t("b"), t("c"), t("d"));
    let query = |label: &str| -> Clause {
        match label {
            "a" => bool_of(vec![a.clone()], vec![], vec![], vec![]),
            "+a +b" => bool_of(vec![a.clone(), b.clone()], vec![], vec![], vec![]),
            "+a #b" => bool_of(vec![a.clone()], vec![b.clone()], vec![], vec![]),
            "a b" => bool_of(vec![], vec![], vec![a.clone(), b.clone()], vec![]),
            "a b c d msm2" => {
                let Clause::Boolean(mut q) = bool_of(
                    vec![],
                    vec![],
                    vec![a.clone(), b.clone(), c.clone(), d.clone()],
                    vec![],
                ) else {
                    unreachable!()
                };
                q.minimum_should_match = 2;
                Clause::Boolean(q)
            }
            "+a ?b" => bool_of(vec![a.clone()], vec![], vec![b.clone()], vec![]),
            "+a ?b ?c" => bool_of(vec![a.clone()], vec![], vec![b.clone(), c.clone()], vec![]),
            "#a ?b ?c" => bool_of(vec![], vec![a.clone()], vec![b.clone(), c.clone()], vec![]),
            "+a -b" => bool_of(vec![a.clone()], vec![], vec![], vec![b.clone()]),
            "a b -c" => bool_of(vec![], vec![], vec![a.clone(), b.clone()], vec![c.clone()]),
            "+a +b ?c -d" => bool_of(
                vec![a.clone(), b.clone()],
                vec![],
                vec![c.clone()],
                vec![d.clone()],
            ),
            "+(a b) +c" => bool_of(
                vec![
                    bool_of(vec![], vec![], vec![a.clone(), b.clone()], vec![]),
                    c.clone(),
                ],
                vec![],
                vec![],
                vec![],
            ),
            "(+a +b) (+c +d)" => bool_of(
                vec![],
                vec![],
                vec![
                    bool_of(vec![a.clone(), b.clone()], vec![], vec![], vec![]),
                    bool_of(vec![c.clone(), d.clone()], vec![], vec![], vec![]),
                ],
                vec![],
            ),
            "dismax(a b)" => bool_of(
                vec![Clause::DisjunctionMax(Box::new(DisjunctionMaxQuery {
                    disjuncts: vec![a.clone(), b.clone()],
                    tie_breaker: 0.0,
                }))],
                vec![],
                vec![],
                vec![],
            ),
            "+const(a) ?b" => bool_of(
                vec![Clause::ConstantScore(Box::new(ConstantScoreQuery::new(
                    a.clone(),
                    1.0,
                )))],
                vec![],
                vec![b.clone()],
                vec![],
            ),
            "+z #d" => bool_of(vec![t("z")], vec![d.clone()], vec![], vec![]),
            other => panic!("unknown query {other}"),
        }
    };
    let mut cases = 0;
    for line in fixture.lines().filter(|l| l.starts_with("cost\t")) {
        let row: Vec<_> = line.split('\t').collect();
        let Clause::Boolean(q) = query(row[1]) else {
            unreachable!()
        };
        let mode = match row[2] {
            "COMPLETE" => ScoreMode::Complete,
            "COMPLETE_NO_SCORES" => ScoreMode::CompleteNoScores,
            "TOP_SCORES" => ScoreMode::TopScores,
            other => panic!("unknown mode {other}"),
        };
        let cost = index.with_searcher(|searcher| {
            with_leaf(searcher, &q, 0, |leaf| {
                Ok(leaf.bulk_scorer(mode)?.map(|bulk| bulk.cost()))
            })
            .unwrap()
        });
        let expected = (row[3] != "-").then(|| row[3].parse::<i64>().unwrap());
        assert_eq!(cost, expected, "{} {}", row[1], row[2]);
        cases += 1;
    }
    assert_eq!(cases, 48);
}
