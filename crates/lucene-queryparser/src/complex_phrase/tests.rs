#![cfg(test)]
use super::*;

#[test]
fn a_phrase_prints_its_field_text_and_slop() {
    let q = ComplexPhraseQuery {
        field: "f".into(),
        text: "a b*".into(),
        slop: 2,
        in_order: true,
        contents: None,
    };
    assert_eq!(q.to_string("f"), "\"a b*\"~2");
    assert_eq!(q.to_string("g"), "f:\"a b*\"~2");
    let q = ComplexPhraseQuery { slop: 0, ..q };
    assert_eq!(q.to_string("f"), "\"a b*\"");
}

#[test]
fn span_builders_refuse_clauses_of_two_fields() {
    let a = SpanNode::term("f", b"a".to_vec());
    let g = SpanNode::term("g", b"a".to_vec());
    let msg = |r: Result<SpanNode, QueryParserError>| r.unwrap_err().message;
    assert_eq!(
        msg(span_not(a.clone(), g.clone())),
        "Clauses must have same field."
    );
    assert_eq!(
        msg(span_or(vec![a.clone(), g.clone()])),
        "Clauses must have same field."
    );
    assert_eq!(
        msg(span_near(vec![a.clone(), g.clone()], 0, true)),
        "Clauses must have same field."
    );
    // a span of no field (a span-first here) is checked against nothing
    let first = SpanNode::first(g, 1);
    assert_eq!(span_field(&first), None);
    assert!(span_not(a.clone(), first.clone()).is_ok());
    assert!(span_or(vec![first, a]).is_ok());
}

#[test]
fn phrases_are_equal_by_field_text_slop_and_order_not_contents() {
    let a = ComplexPhraseQuery {
        field: "f".into(),
        text: "a b".into(),
        slop: 0,
        in_order: true,
        contents: None,
    };
    let parsed = ComplexPhraseQuery {
        contents: Some(ParsedQuery::MatchAll),
        ..a.clone()
    };
    assert_eq!(a, parsed);
    assert_ne!(
        a,
        ComplexPhraseQuery {
            slop: 1,
            ..a.clone()
        }
    );
    assert_ne!(
        a,
        ComplexPhraseQuery {
            in_order: false,
            ..a.clone()
        }
    );
    assert_ne!(
        a,
        ComplexPhraseQuery {
            text: "a".into(),
            ..a.clone()
        }
    );
    assert_ne!(
        a,
        ComplexPhraseQuery {
            field: "g".into(),
            ..a.clone()
        }
    );
}

/// Shapes `BooleanQuery.rewrite` never leaves in a rewritten clause (an
/// all-negative boolean, a match-nothing clause, a match-all one), which
/// Java's `addComplexPhraseClause` still handles.
#[test]
fn boolean_clauses_become_span_ors_with_the_dummy_for_nothing() {
    let phrase = ComplexPhraseQuery {
        field: "f".into(),
        text: "x".into(),
        slop: 0,
        in_order: true,
        contents: None,
    };
    let term = |t: &str| ParsedQuery::Term(TermQuery::new("f", t.as_bytes().to_vec()));
    let boolean = |clauses: Vec<(Occur, ParsedQuery)>| ParsedBoolean {
        clauses,
        minimum_should_match: 0,
    };
    let dummy = SpanNode::term("f", DUMMY.as_bytes().to_vec());
    assert_eq!(
        phrase
            .boolean_span(&boolean(vec![(Occur::MustNot, term("a"))]))
            .unwrap(),
        dummy
    );
    assert_eq!(
        phrase
            .boolean_span(&boolean(vec![
                (Occur::Should, ParsedQuery::MatchNoDocs(String::new())),
                (Occur::Should, term("a")),
            ]))
            .unwrap(),
        SpanNode::Or {
            clauses: vec![dummy, SpanNode::term("f", b"a".to_vec())]
        }
    );
    // a nested boolean's words join the span-or, a boost is looked
    // through, a negated word is the span-not's exclude
    let nested = ParsedQuery::Boolean(boolean(vec![(Occur::Should, term("b"))]));
    assert_eq!(
        phrase
            .boolean_span(&boolean(vec![
                (Occur::Should, nested),
                (Occur::Should, ParsedQuery::Boost(Box::new(term("c")), 2.0)),
                (Occur::MustNot, term("d")),
            ]))
            .unwrap(),
        SpanNode::Not {
            include: Box::new(SpanNode::Or {
                clauses: vec![
                    SpanNode::Or {
                        clauses: vec![SpanNode::term("f", b"b".to_vec())]
                    },
                    SpanNode::term("f", b"c".to_vec())
                ]
            }),
            exclude: Box::new(SpanNode::Or {
                clauses: vec![SpanNode::term("f", b"d".to_vec())]
            }),
            pre: 0,
            post: 0,
        }
    );
    assert_eq!(
        phrase
            .boolean_span(&boolean(vec![(Occur::Should, ParsedQuery::MatchAll)]))
            .unwrap_err()
            .message,
        "Unknown query type:org.apache.lucene.search.MatchAllDocsQuery"
    );
}

#[test]
fn an_unrewritten_phrase_matches_nothing_and_says_why() {
    let q = ParsedQuery::ComplexPhrase(Box::new(ComplexPhraseQuery {
        field: "f".into(),
        text: "a".into(),
        slop: 0,
        in_order: true,
        contents: None,
    }));
    let Clause::MatchNoDocs(m) = q.to_clause(3) else {
        panic!("not a match-no-docs");
    };
    assert_eq!(
        m.reason,
        "ComplexPhraseQuery: call rewrite_complex_phrases first"
    );
}
