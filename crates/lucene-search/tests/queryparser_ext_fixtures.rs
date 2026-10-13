//! M13 T13.1: `ExtendableQueryParser` against Lucene 10.5.0
//! (`fixtures/src/GenQueryParserExt.java`): the extension corpus and the
//! classic one, with five extensions (an upper-cased term, a nested parse,
//! `null`, a `ParseException`, a boosted term) under the `:` and `#`
//! delimiters and with none registered; and `Extensions`' field helpers.

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use lucene_analysis::{Analyzer, StandardAnalyzer};
use lucene_search::query::{BoostQuery, Clause, TermQuery};
use lucene_search::queryparser::classic::{QueryParser, QueryParserError};
use lucene_search::queryparser::ext::{
    ExtendableQueryParser, ExtensionQuery, Extensions, ParserExtension,
};
use lucene_search::queryparser::ParsedQuery;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/queryparser_ext")
}

/// `AnalysisRows.esc` undone.
fn unesc(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut it = s.chars();
    while let Some(c) = it.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match it.next() {
            Some('t') => out.push('\t'),
            Some('n') => out.push('\n'),
            Some('r') => out.push('\r'),
            Some('u') => {
                let h: String = it.by_ref().take(4).collect();
                out.push(char::from_u32(u32::from_str_radix(&h, 16).unwrap()).unwrap());
            }
            Some(o) => out.push(o),
            None => out.push('\\'),
        }
    }
    out
}

fn term(field: &str, text: &str) -> ParsedQuery {
    ParsedQuery::from_clause(Clause::Term(TermQuery::new(
        field,
        text.as_bytes().to_vec(),
    )))
}

struct Upper;
impl ParserExtension for Upper {
    fn parse(&self, q: &ExtensionQuery<'_>) -> Result<Option<ParsedQuery>, QueryParserError> {
        Ok(Some(term(q.field, &q.raw_query_string.to_uppercase())))
    }
}

struct Nested;
impl ParserExtension for Nested {
    fn parse(&self, q: &ExtensionQuery<'_>) -> Result<Option<ParsedQuery>, QueryParserError> {
        q.top_level_parser.parse(q.raw_query_string).map(Some)
    }
}

struct Null;
impl ParserExtension for Null {
    fn parse(&self, _: &ExtensionQuery<'_>) -> Result<Option<ParsedQuery>, QueryParserError> {
        Ok(None)
    }
}

struct Failing;
impl ParserExtension for Failing {
    fn parse(&self, q: &ExtensionQuery<'_>) -> Result<Option<ParsedQuery>, QueryParserError> {
        Err(QueryParserError::parse_exception(format!(
            "extension e failed: {}",
            q.raw_query_string
        )))
    }
}

struct Boosted;
impl ParserExtension for Boosted {
    fn parse(&self, q: &ExtensionQuery<'_>) -> Result<Option<ParsedQuery>, QueryParserError> {
        let t = Clause::Term(TermQuery::new(
            q.field,
            q.raw_query_string.as_bytes().to_vec(),
        ));
        Ok(Some(ParsedQuery::from_clause(Clause::Boost(Box::new(
            BoostQuery::new(t, 2.5),
        )))))
    }
}

fn extensions(delimiter: char) -> Extensions {
    let mut e = Extensions::with_delimiter(delimiter);
    e.add("t", Arc::new(Upper));
    e.add("p", Arc::new(Nested));
    e.add("n", Arc::new(Null));
    e.add("e", Arc::new(Failing));
    e.add("b", Arc::new(Boosted));
    e
}

fn parser(config: &str) -> QueryParser {
    let analyzer = Arc::new(Analyzer::new(StandardAnalyzer::new()));
    match config {
        "colon" => ExtendableQueryParser::with_extensions("f", analyzer, extensions(':')),
        "hash" => ExtendableQueryParser::with_extensions("f", analyzer, extensions('#')),
        "none" => ExtendableQueryParser::without_extensions("f", analyzer),
        other => panic!("unknown configuration {other}"),
    }
}

fn row(r: Result<ParsedQuery, QueryParserError>) -> String {
    match r {
        Ok(q) => format!("Q\t{}\t{}", q.to_string("f"), q.describe("f")),
        Err(e) if e.is_parse_exception() => format!("E\t{}", e.message),
        Err(e) => format!("X\t{}: {}", e.class, e.message),
    }
}

#[test]
fn extendable_query_parser_matches_lucene() {
    let mut total = 0;
    let mut bad = Vec::new();
    for config in ["colon", "hash", "none"] {
        let p = parser(config);
        let text = fs::read_to_string(dir().join(format!("{config}.tsv"))).unwrap();
        for line in text.lines() {
            let cols: Vec<&str> = line.split('\t').collect();
            let query = unesc(cols[0]);
            total += 1;
            let got = row(p.parse(&query));
            let want = match cols[1] {
                "Q" => format!("Q\t{}\t{}", unesc(cols[2]), unesc(cols[3])),
                k => format!("{k}\t{}", unesc(cols[2])),
            };
            if got != want {
                bad.push(format!(
                    "{config} {query:?}\n  want {want:?}\n  got  {got:?}"
                ));
            }
        }
    }
    assert!(total > 700, "only {total} rows compared");
    assert!(
        bad.is_empty(),
        "{} of {total} rows differ:\n{}",
        bad.len(),
        bad.join("\n")
    );
}

#[test]
fn field_helpers_match_lucene() {
    let e = extensions(':');
    assert_eq!(e.extension_field_delimiter(), ':');
    let text = fs::read_to_string(dir().join("helpers.tsv")).unwrap();
    let mut seen = 0;
    for line in text.lines() {
        let cols: Vec<&str> = line.split('\t').collect();
        if cols[0] == "split" {
            let field = unesc(cols[1]);
            let (cur, cud) = e.split_extension_field("dflt", &field);
            assert_eq!(cur, cols[2], "{field}");
            assert_eq!(cud.unwrap_or("<null>"), cols[3], "{field}");
        } else {
            let key = unesc(cols[0]);
            assert_eq!(e.build_extension_field(&key), unesc(cols[1]), "{key}");
            assert_eq!(
                e.build_extension_field_for(&key, "fld"),
                unesc(cols[2]),
                "{key}"
            );
            assert_eq!(e.escape_extension_field(&key), unesc(cols[3]), "{key}");
        }
        seen += 1;
    }
    assert_eq!(seen, 10);
    assert!(e.extension("t").is_some());
    assert!(e.extension("zz").is_none());
    assert!(Extensions::default().extension("t").is_none());
    assert_eq!(Extensions::default().extension_field_delimiter(), ':');
}
