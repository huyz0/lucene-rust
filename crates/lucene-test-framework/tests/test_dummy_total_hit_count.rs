//! Java `DummyTotalHitCountCollector` over the same corpus, segments and deletes.
use std::sync::Arc;

use lucene_analysis::core_analysis::WhitespaceAnalyzer;
use lucene_analysis::Analyzer;
use lucene_index::document::{Document, Store, StringField};
use lucene_search::query::{BooleanQuery, Clause, MatchAllDocsQuery, TermQuery};
use lucene_test_framework::search::dummy_total_hit_count::DummyTotalHitCountCollector;
use lucene_test_framework::search::test_index::TestIndex;

#[test]
fn java_live_search_and_manager_counts() {
    let fixture = include_str!("../../../fixtures/testframework/data/dummy_total_hit_count.txt");
    let mut checked = 0;
    for line in fixture.lines().filter(|line| line.starts_with("search\t")) {
        let cells: Vec<_> = line.split('\t').collect();
        let segments: usize = cells[1].parse().unwrap();
        let deleted = cells[2] == "true";
        let docs: Vec<Vec<_>> = (0..segments)
            .map(|segment| {
                (segment..9)
                    .step_by(segments)
                    .map(|id| {
                        let mut doc = Document::new();
                        doc.add(StringField::new("id", id.to_string(), Store::No));
                        doc.add(StringField::new(
                            "tag",
                            if id % 2 == 0 { "even" } else { "odd" },
                            Store::No,
                        ));
                        doc
                    })
                    .collect()
            })
            .collect();
        let deletes = if deleted {
            vec![("id", "2"), ("id", "7")]
        } else {
            vec![]
        };
        let index = TestIndex::build(
            Arc::new(Analyzer::new(WhitespaceAnalyzer::default())),
            &docs,
            &deletes,
            &[],
        );
        let clause = match cells[3] {
            "all" => Clause::MatchAllDocs(MatchAllDocsQuery::new(9)),
            term => Clause::Term(TermQuery::new("tag", term)),
        };
        let query = BooleanQuery {
            must: vec![clause],
            ..Default::default()
        };
        index.with_searcher(|searcher| {
            let mut collector = DummyTotalHitCountCollector::new();
            searcher.search_collector(&query, &mut collector).unwrap();
            assert_eq!(
                collector.total_hits(),
                cells[4].parse::<i32>().unwrap(),
                "{line}"
            );
            assert_eq!(
                searcher
                    .search_manager(&query, &DummyTotalHitCountCollector::create_manager())
                    .unwrap(),
                cells[5].parse::<i32>().unwrap(),
                "{line}"
            );
        });
        checked += 1;
    }
    assert_eq!(checked, 12);
}
