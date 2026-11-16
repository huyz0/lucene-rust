#![cfg(test)]
use super::*;
use lucene_analysis::StandardAnalyzer;
use lucene_search::query::{Clause, TermQuery};

fn monitor_with(queries: &[(&str, Clause)]) -> Monitor {
    let mut m = Monitor::new(Arc::new(Analyzer::new(StandardAnalyzer::new())));
    m.register(
        queries
            .iter()
            .map(|(id, q)| MonitorQuery::new(*id, q.clone(), None, BTreeMap::new()).unwrap())
            .collect(),
    )
    .unwrap();
    m
}

fn term(text: &str) -> Clause {
    Clause::Term(TermQuery::new("f", text.as_bytes().to_vec()))
}

#[test]
fn hits_are_equal_and_ordered_by_position_alone() {
    let a = Hit {
        start_position: 1,
        start_offset: 4,
        end_position: 1,
        end_offset: 9,
    };
    let b = Hit {
        start_position: 1,
        start_offset: -1,
        end_position: 1,
        end_offset: -1,
    };
    let c = Hit {
        start_position: 2,
        start_offset: 0,
        end_position: 2,
        end_offset: 1,
    };
    assert_eq!(a, b);
    assert!(a < c);
    assert_eq!(a.to_string(), "1(4)->1(9)");
    let mut set = BTreeSet::new();
    set.insert(a);
    set.insert(b);
    assert_eq!(set.len(), 1);
}

#[test]
fn two_matches_of_one_query_resolve_by_summing_and_merging() {
    let mut hits_a = Hits::new();
    hits_a.entry("f".into()).or_default().insert(Hit {
        start_position: 0,
        start_offset: 0,
        end_position: 0,
        end_offset: 1,
    });
    let mut hits_b = Hits::new();
    hits_b.entry("f".into()).or_default().insert(Hit {
        start_position: 3,
        start_offset: 5,
        end_position: 3,
        end_offset: 6,
    });
    hits_b.entry("g".into()).or_default().insert(Hit {
        start_position: 1,
        start_offset: 2,
        end_position: 1,
        end_offset: 3,
    });
    let a = QueryMatch {
        query_id: "q".into(),
        score: Some(1.5),
        explanation: Some(Explanation::match_(1.5, "a")),
        hits: Some(hits_a),
    };
    let b = QueryMatch {
        query_id: "q".into(),
        score: Some(0.25),
        explanation: Some(Explanation::match_(0.25, "b")),
        hits: Some(hits_b),
    };
    let r = a.resolve(b);
    assert_eq!(r.score, Some(1.75));
    let ex = r.explanation.unwrap();
    assert_eq!(ex.description, "sum of:");
    assert_eq!(ex.details.len(), 2);
    assert_eq!(r.hits.unwrap()["f"].len(), 2);
    // a match without detail keeps the other's
    let plain = QueryMatch::plain("q");
    let scored = QueryMatch {
        score: Some(2.0),
        ..QueryMatch::plain("q")
    };
    assert_eq!(plain.resolve(scored).score, Some(2.0));
}

#[test]
fn an_empty_batch_is_refused_and_a_single_document_matches() {
    let m = monitor_with(&[("a", term("fox")), ("b", term("dog"))]);
    let err = m.match_docs(&[], &Matcher::Simple).err().unwrap();
    assert_eq!(
        err.to_string(),
        "A DocumentBatch must contain at least one document"
    );
    let one = m.match_doc(&[("f", "the quick fox")]).unwrap();
    assert_eq!(one.match_count(), 1);
    assert!(one.matches("a").is_some() && one.matches("b").is_none());
    assert_eq!(one.queries_run, 1);
    assert!(one.errors.is_empty());
}

#[test]
fn a_batch_scores_with_its_own_statistics_and_reports_counts() {
    let m = monitor_with(&[("a", term("fox"))]);
    let docs: [DocFields<'_>; 2] = [&[("f", "fox")], &[("f", "dog")]];
    let r = m.match_docs(&docs, &Matcher::Scoring(None)).unwrap();
    assert_eq!(r.batch_size, 2);
    assert_eq!(r.match_count(0), 1);
    assert_eq!(r.match_count(1), 0);
    assert_eq!(r.match_count(9), 0);
    assert!(r.matches("a", 0).unwrap().score.unwrap() > 0.0);
    assert!(r.matches("a", 1).is_none());
    // the same batch under the explaining and highlighting matchers
    let e = m.match_docs(&docs, &Matcher::Explaining).unwrap();
    assert!(
        e.matches("a", 0)
            .unwrap()
            .explanation
            .as_ref()
            .unwrap()
            .matched
    );
    let h = m.match_docs(&docs, &Matcher::Highlights).unwrap();
    assert_eq!(
        h.matches("a", 0).unwrap().hits.as_ref().unwrap()["f"].len(),
        1
    );
}

#[test]
fn registration_surface_and_similarity_pass_through() {
    let mut m = monitor_with(&[("a", term("fox")), ("b", term("dog"))]);
    assert_eq!((m.query_count(), m.disjunct_count()), (2, 2));
    assert_eq!(m.query_ids(), vec!["a".to_string(), "b".to_string()]);
    assert!(m.get_query("a").is_some());
    m.delete_by_id(&["a"]).unwrap();
    assert_eq!(m.query_count(), 1);
    m.clear().unwrap();
    assert_eq!(m.query_count(), 0);
    let m = monitor_with(&[("a", term("fox"))]);
    let sim: Arc<dyn Similarity> =
        Arc::new(lucene_search::similarities::ClassicSimilarity::default());
    let classic = m
        .match_doc_with_similarity(&[("f", "fox fox")], sim)
        .unwrap();
    let bm25 = m.match_doc_scored(&[("f", "fox fox")]).unwrap();
    assert_ne!(classic.matches[0].score, bm25.matches[0].score);
}
