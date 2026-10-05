use super::*;
use crate::extended_query::{MultiTermSource, RewriteMethod};
use crate::query::{PrefixQuery, SpanQuery};

fn t(w: &str) -> SpanNode {
    SpanNode::term("f", w)
}

/// Every span query's `toString`, as Lucene prints it.
#[test]
fn the_queries_print_as_java_does() {
    let near = SpanNode::near(vec![t("a"), t("b")], 2, true).unwrap();
    assert_eq!(near.to_string(), "spanNear([f:a, f:b], 2, true)");
    let or = SpanNode::or(vec![t("a"), near.clone()]).unwrap();
    assert_eq!(
        or.to_string(),
        "spanOr([f:a, spanNear([f:a, f:b], 2, true)])"
    );
    assert_eq!(SpanNode::first(t("a"), 3).to_string(), "spanFirst(f:a, 3)");
    assert_eq!(
        SpanNode::position_range(t("a"), 1, 4).to_string(),
        "spanPosRange(f:a, 1, 4)"
    );
    assert_eq!(
        SpanNode::not(t("a"), t("b"), -1, 2).unwrap().to_string(),
        "spanNot(f:a, f:b, -1, 2)"
    );
    assert_eq!(
        SpanNode::not_within(t("a"), t("b"), 3).unwrap().to_string(),
        "spanNot(f:a, f:b, 3, 3)"
    );
    assert_eq!(
        SpanNode::containing(near.clone(), t("c"))
            .unwrap()
            .to_string(),
        "SpanContaining(spanNear([f:a, f:b], 2, true), f:c)"
    );
    assert_eq!(
        SpanNode::within(near, t("c")).unwrap().to_string(),
        "SpanWithin(spanNear([f:a, f:b], 2, true), f:c)"
    );
    assert_eq!(
        SpanNode::field_masking(SpanNode::term("g", "x"), "f").to_string(),
        "mask(g:x) as f"
    );
    let prefix = SpanNode::multi_term(MultiTermQuery::new(
        MultiTermSource::Prefix(PrefixQuery::new("f", "ap")),
        RewriteMethod::default(),
    ));
    assert_eq!(prefix.to_string(), "SpanMultiTermQueryWrapper(f:ap*)");
    assert_eq!(term_to_string(b"abc"), "abc");
    assert_eq!(term_to_string(&[0x61, 0xff, 0x01]), "[61 ff 1]");
}

/// The constructors' field checks; a field-less clause (none here) and
/// negative distances are allowed, as Java's are.
#[test]
fn clauses_of_different_fields_are_refused() {
    let other = SpanNode::term("g", "a");
    assert!(SpanNode::near(vec![t("a"), other.clone()], 0, true).is_err());
    assert!(SpanNode::or(vec![t("a"), other.clone()]).is_err());
    assert!(SpanNode::not(t("a"), other.clone(), 0, 0).is_err());
    assert!(SpanNode::containing(t("a"), other.clone()).is_err());
    assert!(SpanNode::within(t("a"), other.clone()).is_err());
    assert!(SpanNode::near(Vec::new(), 0, true).is_ok());
    assert!(SpanNode::not(t("a"), t("b"), -5, -5).is_ok());
    // A masked clause reports the masking field.
    let masked = SpanNode::field_masking(other, "f");
    assert_eq!(masked.field(), Some("f"));
    assert!(SpanNode::near(vec![t("a"), masked], 0, true).is_ok());
    assert_eq!(
        SpanNode::Near {
            clauses: Vec::new(),
            slop: 0,
            in_order: true
        }
        .field(),
        None
    );
}

/// A multi-term wrapper anywhere in the tree asks for the searcher's
/// rewrite; a query of the core span queries converts as it is.
#[test]
fn rewrite_needs_and_conversions() {
    let prefix = SpanNode::multi_term(MultiTermQuery::new(
        MultiTermSource::Prefix(PrefixQuery::new("f", "ap")),
        RewriteMethod::default(),
    ));
    assert!(prefix.needs_rewrite());
    assert!(SpanNode::first(prefix.clone(), 2).needs_rewrite());
    assert!(SpanNode::not(t("a"), prefix.clone(), 0, 0)
        .unwrap()
        .needs_rewrite());
    let check = SpanNode::PayloadCheck(Box::new(payloads::SpanPayloadCheckQuery::new(
        prefix.clone(),
        Vec::new(),
    )));
    assert!(check.needs_rewrite());
    assert!(!SpanNode::near(vec![t("a"), t("b")], 1, false)
        .unwrap()
        .needs_rewrite());
    assert_eq!(check.field(), Some("f"));

    let core = SpanQuery::span_near(
        [
            SpanQuery::span_term("f", "a"),
            SpanQuery::span_or([SpanQuery::span_term("f", "b")]),
        ],
        3,
        false,
    );
    assert_eq!(
        SpanNode::from(&core),
        SpanNode::Near {
            clauses: vec![
                t("a"),
                SpanNode::Or {
                    clauses: vec![t("b")]
                }
            ],
            slop: 3,
            in_order: false,
        }
    );
    let clause = crate::query::Clause::from(t("a"));
    assert!(matches!(clause, crate::query::Clause::Extended(_)));
}

/// The statistics a weight takes: a not query's include side only; a
/// containment's both sides; a masked query's own; the statistics pass reads
/// the exclude side too.
#[test]
fn weight_terms_follow_extract_term_states() {
    let q = SpanNode::not(SpanNode::containing(t("a"), t("b")).unwrap(), t("c"), 0, 0).unwrap();
    let mut w = Vec::new();
    weight_terms(&q, &mut w);
    let names: Vec<&[u8]> = w.iter().map(|(_, t)| t.as_slice()).collect();
    assert_eq!(names, [b"a".as_slice(), b"b"]);
    let mut all = Vec::new();
    all_terms(&q, &mut all);
    assert_eq!(all.len(), 3);
    let masked = SpanNode::field_masking(SpanNode::term("g", "x"), "f");
    assert_eq!(weight_query(&masked), &SpanNode::term("g", "x"));
    let mut m = Vec::new();
    weight_terms(&masked, &mut m);
    assert_eq!(m, [("g".to_string(), b"x".to_vec())]);
}
