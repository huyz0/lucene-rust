//! M13 T13.1: `SimpleQueryParser` against Lucene 10.5.0
//! (`fixtures/src/GenQueryParserSimple.java`): the classic parser's corpus,
//! the simple parser's own (`fixtures/corpus/queryparser-simple.txt`) and
//! 3,000 seeded joins of operator fragments, under fourteen configurations
//! (default operator, analyzers, field weights and every flag switched off
//! in turn), to the same query tree (`toString("f")` and the class tree).

use std::fs;
use std::path::PathBuf;

use lucene_analysis::core_analysis::WhitespaceAnalyzer;
use lucene_analysis::{Analyzer, StandardAnalyzer};
use lucene_search::query_visitor::Occur;
use lucene_search::queryparser::simple::*;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/queryparser_simple")
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

fn standard() -> Analyzer {
    Analyzer::new(StandardAnalyzer::new())
}

/// The parser of `GenQueryParserSimple`'s configuration `name`.
fn parser(name: &str) -> SimpleQueryParser {
    let one =
        |flags: i32| SimpleQueryParser::with_weights(standard(), vec![("f".into(), 1.0)], flags);
    match name {
        "std" => SimpleQueryParser::new(standard(), "f"),
        "std_must" => {
            let mut p = SimpleQueryParser::new(standard(), "f");
            p.set_default_operator(Occur::Must).unwrap();
            p
        }
        "ws" => SimpleQueryParser::new(Analyzer::new(WhitespaceAnalyzer::default()), "f"),
        "weights" => SimpleQueryParser::with_weights(
            standard(),
            vec![("f".into(), 1.0), ("g".into(), 2.5)],
            -1,
        ),
        "no_whitespace" => one(!WHITESPACE_OPERATOR),
        "no_precedence" => one(!PRECEDENCE_OPERATORS),
        "no_escape" => one(!ESCAPE_OPERATOR),
        "no_phrase_near" => one(!(PHRASE_OPERATOR | NEAR_OPERATOR)),
        "no_near" => one(!NEAR_OPERATOR),
        "no_fuzzy_prefix" => one(!(FUZZY_OPERATOR | PREFIX_OPERATOR)),
        "no_bool" => one(!(AND_OPERATOR | OR_OPERATOR | NOT_OPERATOR)),
        "none" => one(0),
        other => panic!("unknown configuration {other}"),
    }
}

fn check(file: &str, config: &str) -> (usize, Vec<String>) {
    let p = parser(config);
    let text = fs::read_to_string(dir().join(file)).unwrap();
    let mut rows = 0;
    let mut bad = Vec::new();
    for line in text.lines() {
        let cols: Vec<&str> = line.split('\t').collect();
        let query = unesc(cols[0]);
        rows += 1;
        let got = match p.parse(&query) {
            Ok(q) => format!("Q\t{}\t{}", q.to_string("f"), q.describe("f")),
            Err(e) => format!("X\t{}: {}", e.class, e.message),
        };
        let want = match cols[1] {
            "Q" => format!("Q\t{}\t{}", unesc(cols[2]), unesc(cols[3])),
            k => format!("{k}\t{}", unesc(cols[2])),
        };
        if got != want {
            bad.push(format!("{file} {query:?}\n  want {want:?}\n  got  {got:?}"));
        }
    }
    (rows, bad)
}

#[test]
fn simple_query_parser_matches_lucene() {
    let mut total = 0;
    let mut all_bad = Vec::new();
    for config in [
        "std",
        "std_must",
        "ws",
        "weights",
        "no_whitespace",
        "no_precedence",
        "no_escape",
        "no_phrase_near",
        "no_near",
        "no_fuzzy_prefix",
        "no_bool",
        "none",
    ] {
        let (n, bad) = check(&format!("{config}.tsv"), config);
        total += n;
        all_bad.extend(bad);
    }
    for config in ["std", "std_must", "no_whitespace", "no_escape"] {
        let (n, bad) = check(&format!("frag_{config}.tsv"), config);
        total += n;
        all_bad.extend(bad);
    }
    assert!(total > 15_000, "only {total} rows compared");
    let shown: Vec<&str> = all_bad.iter().take(40).map(String::as_str).collect();
    assert!(
        all_bad.is_empty(),
        "{} of {total} rows differ:\n{}",
        all_bad.len(),
        shown.join("\n")
    );
}

#[test]
fn only_should_and_must_are_default_operators() {
    let mut p = SimpleQueryParser::new(standard(), "f");
    assert_eq!(p.default_operator(), Occur::Should);
    let e = p.set_default_operator(Occur::MustNot).unwrap_err();
    assert_eq!(
        e.to_string(),
        "IllegalArgumentException: invalid operator: only SHOULD or MUST are allowed"
    );
    assert!(p.set_default_operator(Occur::Filter).is_err());
    p.set_default_operator(Occur::Must).unwrap();
    assert_eq!(p.default_operator(), Occur::Must);
}
