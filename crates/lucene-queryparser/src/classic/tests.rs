use std::collections::HashMap;
use std::sync::Arc;

use lucene_analysis::{AnalysisError, Analyzer, StandardAnalyzer};

use super::*;
use lucene_search::query_visitor::Occur;

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
    let e = QueryParserError::from_search(lucene_search::Error::IllegalArgument("x".into()));
    assert_eq!(e.class, "IllegalArgumentException");
    let e = QueryParserError::from_search(lucene_search::Error::InvalidQuery("y".into()));
    assert_eq!(e.class, "RuntimeException");
    let e = QueryParserError::from_analysis(AnalysisError::IllegalArgument("z".into()));
    assert_eq!(e.to_string(), "IllegalArgumentException: z");
    let e = QueryParserError::from_analysis(AnalysisError::IllegalState("w".into()));
    assert_eq!(e.class, "RuntimeException");
    let e: &dyn std::error::Error = &QueryParserError::parse("p");
    assert_eq!(e.to_string(), "ParseException: p");
}

#[test]
fn deeply_nested_groups_fail_to_parse_instead_of_overflowing_the_stack() {
    // The limits fit an unoptimised build on a 2 MiB thread; a coverage-
    // instrumented build's frames are larger, so run on a big stack.
    std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(|| {
            let p = QueryParser::new("f", standard());
            let nested = |depth: usize| format!("{}a{}", "(".repeat(depth), ")".repeat(depth));
            assert_eq!(parse(&p, &nested(parser::MAX_GROUP_DEPTH)), "a");
            let err = parse(&p, &nested(parser::MAX_GROUP_DEPTH + 1));
            assert!(err.contains("nested groups"), "{err}");
            let err = parse(&p, &nested(200_000));
            assert!(err.contains("nested groups"), "{err}");
        })
        .expect("spawn")
        .join()
        .expect("no panic");
}

/// Lucene 10.6 (#16446): the default operator joins the term positions of one
/// analysed text across fields, not just the fields of one position.
#[test]
fn multi_field_parser_honors_the_default_operator_between_term_positions() {
    let mut p = QueryParser::multi_field(&["a", "b"], standard());
    assert_eq!(parse(&p, "x y"), "(a:x b:x) (a:y b:y)");
    p.set_default_operator(Operator::And);
    assert_eq!(parse(&p, "x y"), "+(a:x b:x) +(a:y b:y)");
}

/// Overrides that name the method called, with its arguments.
struct Named;

fn tag(s: String) -> Result<Option<ParsedQuery>, QueryParserError> {
    Ok(Some(ParsedQuery::MatchNoDocs(s)))
}

impl Overrides for Named {
    fn field_query(
        &self,
        f: Option<&str>,
        t: &str,
        q: bool,
    ) -> Result<Option<ParsedQuery>, QueryParserError> {
        tag(format!("field {f:?} {t} {q}"))
    }
    fn field_query_slop(
        &self,
        f: Option<&str>,
        t: &str,
        s: i32,
    ) -> Result<Option<ParsedQuery>, QueryParserError> {
        tag(format!("slop {f:?} {t} {s}"))
    }
    fn range_query(
        &self,
        f: Option<&str>,
        a: Option<&str>,
        b: Option<&str>,
        si: bool,
        ei: bool,
    ) -> Result<Option<ParsedQuery>, QueryParserError> {
        tag(format!("range {f:?} {a:?} {b:?} {si} {ei}"))
    }
    fn wildcard_query(
        &self,
        f: Option<&str>,
        t: &str,
    ) -> Result<Option<ParsedQuery>, QueryParserError> {
        tag(format!("wildcard {f:?} {t}"))
    }
    fn regexp_query(
        &self,
        f: Option<&str>,
        t: &str,
    ) -> Result<Option<ParsedQuery>, QueryParserError> {
        tag(format!("regexp {f:?} {t}"))
    }
    fn prefix_query(
        &self,
        f: Option<&str>,
        t: &str,
    ) -> Result<Option<ParsedQuery>, QueryParserError> {
        tag(format!("prefix {f:?} {t}"))
    }
    fn fuzzy_query(
        &self,
        f: Option<&str>,
        t: &str,
        s: f32,
    ) -> Result<Option<ParsedQuery>, QueryParserError> {
        tag(format!("fuzzy {f:?} {t} {s}"))
    }
    fn bare_fuzzy(
        &self,
        f: Option<&str>,
        slop: &str,
        t: &str,
    ) -> Result<Option<ParsedQuery>, QueryParserError> {
        tag(format!("bare {f:?} {slop} {t}"))
    }
    fn boolean_query(
        &self,
        clauses: Vec<(Occur, ParsedQuery)>,
    ) -> Result<Option<ParsedQuery>, QueryParserError> {
        let n = clauses.len();
        tag(format!("bool {n}"))
    }
}

fn named() -> QueryParser {
    QueryParser::new("f", standard())
}

fn reasons(p: &QueryParser, q: &str) -> String {
    match p.parse_with_overrides(q, &Named) {
        Ok(ParsedQuery::MatchNoDocs(s)) => s,
        Ok(other) => format!("{other:?}"),
        Err(e) => e.to_string(),
    }
}

#[test]
fn overrides_build_each_leaf() {
    let p = named();
    assert_eq!(reasons(&p, "a:b"), "field Some(\"a\") b false");
    assert_eq!(reasons(&p, "a:\"b c\"~3"), "slop Some(\"a\") b c 3");
    assert_eq!(
        reasons(&p, "a:[b TO c}"),
        "range Some(\"a\") Some(\"b\") Some(\"c\") true false"
    );
    assert_eq!(reasons(&p, "a:b*c"), "wildcard Some(\"a\") b*c");
    assert_eq!(reasons(&p, "a:/b+/"), "regexp Some(\"a\") b+");
    assert_eq!(reasons(&p, "a:b*"), "prefix Some(\"a\") b");
    assert_eq!(reasons(&p, "a:b~1"), "bare Some(\"a\") ~1 b");
    assert_eq!(reasons(&p, "a:b c:d"), "bool 2");
}

#[test]
fn overrides_without_a_field_see_none() {
    let p = named();
    assert_eq!(reasons(&p, "b"), "field Some(\"f\") b false");
    let m = QueryParser::multi_field(&["x"], standard());
    assert_eq!(reasons(&m, "b"), "field None b false");
    assert_eq!(reasons(&m, "b~"), "bare None ~ b");
}

struct Failing(bool);

impl Overrides for Failing {
    fn field_query(
        &self,
        _: Option<&str>,
        _: &str,
        _: bool,
    ) -> Result<Option<ParsedQuery>, QueryParserError> {
        if self.0 {
            Err(QueryParserError::parse_exception("bad query"))
        } else {
            Err(QueryParserError::illegal_argument("bad argument"))
        }
    }
    fn field_query_slop(
        &self,
        _: Option<&str>,
        _: &str,
        _: i32,
    ) -> Result<Option<ParsedQuery>, QueryParserError> {
        Ok(None)
    }
    fn range_query(
        &self,
        _: Option<&str>,
        _: Option<&str>,
        _: Option<&str>,
        _: bool,
        _: bool,
    ) -> Result<Option<ParsedQuery>, QueryParserError> {
        Ok(None)
    }
    fn wildcard_query(
        &self,
        _: Option<&str>,
        _: &str,
    ) -> Result<Option<ParsedQuery>, QueryParserError> {
        Ok(None)
    }
    fn regexp_query(
        &self,
        _: Option<&str>,
        _: &str,
    ) -> Result<Option<ParsedQuery>, QueryParserError> {
        Ok(None)
    }
    fn prefix_query(
        &self,
        _: Option<&str>,
        _: &str,
    ) -> Result<Option<ParsedQuery>, QueryParserError> {
        Ok(None)
    }
    fn fuzzy_query(
        &self,
        _: Option<&str>,
        _: &str,
        _: f32,
    ) -> Result<Option<ParsedQuery>, QueryParserError> {
        Ok(None)
    }
    fn bare_fuzzy(
        &self,
        _: Option<&str>,
        _: &str,
        _: &str,
    ) -> Result<Option<ParsedQuery>, QueryParserError> {
        Ok(None)
    }
    fn boolean_query(
        &self,
        _: Vec<(Occur, ParsedQuery)>,
    ) -> Result<Option<ParsedQuery>, QueryParserError> {
        Ok(None)
    }
}

#[test]
fn an_override_error_is_the_parsers_own() {
    let p = QueryParser::new("f", standard());
    let e = p
        .parse_with_overrides("a", &Failing(true))
        .expect_err("fails");
    assert!(e.is_parse_exception(), "{e}");
    let e = p
        .parse_with_overrides("a", &Failing(false))
        .expect_err("fails");
    assert!(!e.is_parse_exception(), "{e}");
}
