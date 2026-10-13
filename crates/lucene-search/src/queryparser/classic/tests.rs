use std::collections::HashMap;
use std::sync::Arc;

use lucene_analysis::{AnalysisError, Analyzer, StandardAnalyzer};

use super::*;
use crate::query_visitor::Occur;

fn standard() -> Arc<Analyzer> {
    Arc::new(Analyzer::new(StandardAnalyzer::new()))
}

fn parse(p: &QueryParser, q: &str) -> String {
    match p.parse(q) {
        Ok(q) => q.to_string("f"),
        Err(e) => e.to_string(),
    }
}

#[test]
fn settings_round_trip() {
    let mut p = QueryParser::new("f", standard());
    assert_eq!(p.field(), Some("f"));
    let _ = p.analyzer();
    assert_eq!(p.default_operator(), Operator::Or);
    p.set_default_operator(Operator::And);
    assert_eq!(p.default_operator(), Operator::And);
    assert!(!p.allow_leading_wildcard());
    p.set_allow_leading_wildcard(true);
    assert!(p.allow_leading_wildcard());
    p.set_phrase_slop(3);
    assert_eq!(p.phrase_slop(), 3);
    assert_eq!(p.fuzzy_min_sim(), 2.0);
    p.set_fuzzy_min_sim(0.5);
    assert_eq!(p.fuzzy_min_sim(), 0.5);
    p.set_fuzzy_prefix_length(1);
    assert_eq!(p.fuzzy_prefix_length(), 1);
    p.set_enable_graph_queries(false);
    p.set_auto_generate_multi_term_synonyms_phrase_query(true);
    assert!(!p.split_on_whitespace());
    assert!(!p.auto_generate_phrase_queries());
}

#[test]
fn auto_phrase_and_no_split_exclude_each_other() {
    let mut p = QueryParser::new("f", standard());
    let e = p.set_auto_generate_phrase_queries(true).unwrap_err();
    assert_eq!(e.class, "IllegalArgumentException");
    assert!(!e.is_parse_exception());
    p.set_split_on_whitespace(true).unwrap();
    p.set_auto_generate_phrase_queries(true).unwrap();
    assert!(p.auto_generate_phrase_queries());
    let e = p.set_split_on_whitespace(false).unwrap_err();
    assert_eq!(
        e.to_string(),
        "IllegalArgumentException: setSplitOnWhitespace(false) is disallowed when getAutoGeneratePhraseQueries() == true"
    );
    p.set_auto_generate_phrase_queries(false).unwrap();
    p.set_split_on_whitespace(false).unwrap();
}

#[test]
fn escape_backslashes_every_syntax_character() {
    assert_eq!(
        QueryParser::escape("a+b-c!d(e)f:g^h[i]j\"k{l}m~n*o?p|q&r/s\\t"),
        "a\\+b\\-c\\!d\\(e\\)f\\:g\\^h\\[i\\]j\\\"k\\{l\\}m\\~n\\*o\\?p\\|q\\&r\\/s\\\\t"
    );
    let p = QueryParser::new("f", standard());
    // an escaped string parses back to its words
    assert_eq!(parse(&p, &QueryParser::escape("a:b")), "a:b");
}

#[test]
fn too_many_clauses_is_a_parse_exception() {
    let p = QueryParser::new("f", standard());
    let q: Vec<String> = (0..1025).map(|i| format!("+t{i}")).collect();
    let e = p.parse(&q.join(" ")).unwrap_err();
    assert!(e.is_parse_exception());
    assert!(
        e.message.ends_with("': too many boolean clauses"),
        "{}",
        e.message
    );
    let q: Vec<String> = (0..1024).map(|i| format!("+t{i}")).collect();
    assert!(p.parse(&q.join(" ")).is_ok());
}

#[test]
fn a_negative_phrase_slop_is_refused_by_the_phrase() {
    let mut p = QueryParser::new("f", standard());
    p.set_phrase_slop(-1);
    // no phrase is made: the slop is never set
    assert_eq!(parse(&p, "a"), "a");
    assert_eq!(
        parse(&p, "\"a b\""),
        "IllegalArgumentException: Slop must be >= 0, got -1"
    );
}

#[test]
fn a_negative_fuzzy_prefix_length_is_refused() {
    let mut p = QueryParser::new("f", standard());
    p.set_fuzzy_prefix_length(-1);
    assert_eq!(
        parse(&p, "a~1"),
        "IllegalArgumentException: prefixLength cannot be negative."
    );
}

#[test]
fn multi_field_parser_has_no_field_of_its_own() {
    let p = QueryParser::multi_field(&["a", "b"], standard());
    assert_eq!(p.field(), None);
    assert_eq!(parse(&p, "x"), "a:x b:x");
    assert_eq!(parse(&p, "c:x"), "c:x");
    // a phrase made from no words is no clause
    assert_eq!(parse(&p, "\"\""), "");
    let p = QueryParser::multi_field_with_boosts(
        &["a", "b"],
        standard(),
        HashMap::from([("a".to_string(), 3.0)]),
    );
    assert_eq!(parse(&p, "x*"), "(a:x*)^3.0 b:x*");
    assert_eq!(parse(&p, "\"x y\"~2"), "(a:\"x y\"~2)^3.0 b:\"x y\"~2");
    assert_eq!(parse(&p, "a:x*"), "(a:x*)^3.0");
    assert_eq!(
        parse(&p, "/x/ x~1 [p TO q]"),
        "((a:/x/)^3.0 b:/x/) ((a:x~1)^3.0 b:x~1) ((a:[p TO q])^3.0 b:[p TO q])"
    );
}

#[test]
fn parse_helpers_keep_the_flags() {
    let q =
        QueryParser::parse_with_flags("x", &["a", "b"], &[Occur::Must, Occur::Filter], standard())
            .unwrap();
    assert_eq!(q.to_string("f"), "+a:x #b:x");
}

#[test]
fn errors_convert_from_the_search_and_analysis_errors() {
    let e = QueryParserError::from_search(crate::Error::IllegalArgument("x".into()));
    assert_eq!(e.class, "IllegalArgumentException");
    let e = QueryParserError::from_search(crate::Error::InvalidQuery("y".into()));
    assert_eq!(e.class, "RuntimeException");
    let e = QueryParserError::from_analysis(AnalysisError::IllegalArgument("z".into()));
    assert_eq!(e.to_string(), "IllegalArgumentException: z");
    let e = QueryParserError::from_analysis(AnalysisError::IllegalState("w".into()));
    assert_eq!(e.class, "RuntimeException");
    let e: &dyn std::error::Error = &QueryParserError::parse("p");
    assert_eq!(e.to_string(), "ParseException: p");
}
