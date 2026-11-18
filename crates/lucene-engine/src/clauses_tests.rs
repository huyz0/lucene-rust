//! Which fields' norms a scored search loads for each clause.
use super::clause_field_names;
use lucene_search::extended_query::ExtendedQuery;
use lucene_search::query::{BooleanQuery, Clause};
use lucene_search::spans::SpanNode;

fn span_clause(node: SpanNode) -> BooleanQuery {
    BooleanQuery::new().with_must([Clause::Extended(Box::new(ExtendedQuery::Span(node)))])
}

/// A top-level field-masking span query scores with its masked query's weight, so the norms
/// it reads are the masked query's field's, not the field it reports.
#[test]
fn top_level_field_masking_loads_the_masked_querys_norms() {
    let masked = SpanNode::field_masking(SpanNode::term("title", b"tau".to_vec()), "body");
    assert_eq!(clause_field_names(&span_clause(masked)), vec!["title"]);
}

/// Inside a near query the masking clause reports its field: the near weight's field is the
/// first clause's.
#[test]
fn nested_field_masking_keeps_the_reported_field() {
    let masked = SpanNode::field_masking(SpanNode::term("title", b"tau".to_vec()), "body");
    let near = SpanNode::near(
        vec![masked, SpanNode::term("body", b"alpha".to_vec())],
        3,
        false,
    )
    .unwrap();
    assert_eq!(clause_field_names(&span_clause(near)), vec!["body"]);
}
