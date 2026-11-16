#![cfg(test)]
use super::*;

/// The matches iterators before their first and after their last match
/// report `-1`s, and a sub-match's query is its term's (the span
/// query's before the first).
#[test]
fn span_matches_iterators_outside_their_matches() {
    use crate::matches::MatchesIterator;
    let span_q = Arc::new(crate::query::Clause::from(SpanNode::term("f", "a")));
    let term = Arc::new(crate::query::Clause::Term(crate::TermQuery::new("f", "a")));
    let mut it = SpanMatchesIterator {
        query: Arc::clone(&span_q),
        spans: Arc::new(vec![SpanMatch {
            start: 2,
            end: 4,
            terms: vec![
                TermMatch {
                    query: Arc::clone(&term),
                    position: 2,
                    start_offset: 5,
                    end_offset: 6,
                },
                TermMatch {
                    query: Arc::clone(&term),
                    position: 4,
                    start_offset: 9,
                    end_offset: 12,
                },
            ],
        }]),
        at: 0,
    };
    let outside = |it: &SpanMatchesIterator| {
        [
            it.start_position(),
            it.end_position(),
            it.start_offset(),
            it.end_offset(),
        ]
    };
    assert_eq!(outside(&it), [-1; 4]);
    assert!(it.next().unwrap());
    assert_eq!(outside(&it), [2, 4, 5, 12]);
    assert_eq!(it.query(), span_q.as_ref());
    let mut sub = it.sub_matches().unwrap().unwrap();
    assert_eq!(sub.query(), span_q.as_ref(), "before the first term");
    assert_eq!(sub.start_position(), -1);
    assert!(sub.next().unwrap());
    assert_eq!((sub.start_position(), sub.end_offset()), (2, 6));
    assert_eq!(sub.query(), term.as_ref());
    assert!(sub.next().unwrap());
    assert_eq!((sub.end_position(), sub.start_offset()), (4, 9));
    assert!(!sub.next().unwrap());
    assert!(!it.next().unwrap());
    assert_eq!(outside(&it), [2, 4, 5, 12], "stays on the last");
}

/// No matches where the segment has no spans for the query, or the
/// document has none.
#[test]
fn span_matches_of_a_missing_term_or_document_are_none() {
    let reader = reader();
    let opened = reader.open_segments().unwrap();
    let segs = opened.as_open_segments();
    let seg = &segs[0];
    let ctx = LeafContext {
        fields: seg.fields,
        doc_in: seg.doc_in,
        pos_in: seg.pos_in,
        pay_in: seg.pay_in,
        live_docs: None,
        points: None,
        norms: None,
        global: None,
        max_doc: seg.max_doc,
        cache: None,
        reader: seg.reader,
        similarity: seg.similarity,
    };
    let q = Arc::new(crate::query::Clause::MatchNoDocs(Default::default()));
    let missing = SpanNode::term("body", "nosuch");
    assert!(span_matches(&ctx, &missing, Arc::clone(&q), 0)
        .unwrap()
        .is_none());
    let empty_or = SpanNode::Or {
        clauses: Vec::new(),
    };
    assert!(span_matches(&ctx, &empty_or, Arc::clone(&q), 0)
        .unwrap()
        .is_none());
    let apple = SpanNode::term("body", "apple");
    let found = (0..seg.max_doc.unwrap())
        .filter(|&d| {
            span_matches(&ctx, &apple, Arc::clone(&q), d)
                .unwrap()
                .is_some()
        })
        .count();
    assert!(found > 0 && found < usize::try_from(seg.max_doc.unwrap()).unwrap());
}

fn reader() -> crate::directory_reader::DirectoryReader {
    let dir = std::path::Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/spans/index"
    ));
    crate::directory_reader::DirectoryReader::open(&lucene_store::FsDirectory::open(dir)).unwrap()
}

/// The weight's statistics where Java's `simScorer` is `null`; a scorer
/// without norms explains BM25 over unnormed lengths; the scorer's
/// bounds and costs.
#[test]
fn statistics_bounds_and_unnormed_explanations() {
    let reader = reader();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let seg = &segments[0];
    let ctx = LeafContext {
        fields: seg.fields,
        doc_in: seg.doc_in,
        pos_in: seg.pos_in,
        pay_in: seg.pay_in,
        live_docs: seg.live_docs,
        points: None,
        norms: None,
        global: None,
        max_doc: seg.max_doc,
        cache: None,
        reader: None,
        similarity: seg.similarity,
    };
    let empty = SpanNode::Near {
        clauses: Vec::new(),
        slop: 0,
        in_order: true,
    };
    assert!(sim_inputs(&ctx, &empty).unwrap().is_none());
    assert!(sim_inputs(&ctx, &SpanNode::term("body", "nosuch"))
        .unwrap()
        .is_none());
    assert!(sim_inputs(&ctx, &SpanNode::term("nofield", "a"))
        .unwrap()
        .is_none());
    assert!(span_node(&ctx, &empty, 1.0, Mode::Complete)
        .unwrap()
        .is_none());

    let q = SpanNode::term("body", "apple");
    let mut s = build_scorer(&ctx, &q, 2.0, Mode::Complete)
        .unwrap()
        .unwrap();
    assert_eq!(s.max_score(NO_MORE_DOCS).unwrap(), f32::INFINITY);
    assert!(!s.two_phase());
    assert!(s.cost() > 0 && s.match_cost() > 0.0);
    let doc = s.next_doc().unwrap();
    let score = s.score().unwrap();
    let e = explain_span_node(&ctx, &q, 2.0, doc).unwrap();
    assert_eq!(e.value, score);
    let text = e.to_string();
    assert!(text.contains("2.0 = boost"), "{text}");
    assert!(text.contains("1.0 = dl, length of field"), "{text}");
    assert_eq!(
        explain_span_node(&ctx, &q, 1.0, NO_MORE_DOCS - 1)
            .unwrap()
            .description,
        "no matching term"
    );
    // Without scores: frequency 1, no similarity.
    let mut n = build_scorer(&ctx, &q, 1.0, Mode::NoScores)
        .unwrap()
        .unwrap();
    n.next_doc().unwrap();
    assert_eq!(n.score().unwrap(), 0.0);
    let e = explain_span_unscored(&ctx, &q, doc).unwrap();
    assert_eq!(
        e.description,
        format!("match body:apple in {doc} without score")
    );
    assert!(
        !explain_span_unscored(&ctx, &q, NO_MORE_DOCS - 1)
            .unwrap()
            .matched
    );
    assert!(
        !explain_span_unscored(&ctx, &SpanNode::term("body", "nosuch"), 0)
            .unwrap()
            .matched
    );
}
