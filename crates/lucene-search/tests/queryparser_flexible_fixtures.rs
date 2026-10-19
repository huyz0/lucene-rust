//! M13 T13.1: `StandardQueryParser` (the flexible query parser) against
//! Lucene 10.5.0 (`fixtures/src/GenQueryParserFlexible.java`): every query of
//! `fixtures/corpus/queryparser-classic.txt` and `queryparser-flexible.txt`,
//! under eleven configurations, to the same query (`toString("f")` and the
//! class tree) or the same exception class and message; and the range
//! bounds of `GenQueryParserClassic.DATES` read as dates (`date_*.tsv`).

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use lucene_analysis::charfilter::{MappingCharFilter, NormalizeCharMap, NormalizeCharMapBuilder};
use lucene_analysis::core_analysis::StopAnalyzer;
use lucene_analysis::miscellaneous::DateLocale;
use lucene_analysis::synonym::{SynonymGraphFilter, SynonymMap, SynonymMapBuilder};
use lucene_analysis::util::WhitespaceTokenizer;
use lucene_analysis::{
    english_stop_words, AnalysisError, Analyzer, AnalyzerDefinition, CharArraySet, CharReader,
    LowerCaseFilter, StandardAnalyzer, TokenStream, TokenStreamComponents,
};
use lucene_index::document::Resolution;
use lucene_search::extended_query::RewriteMethod;
use lucene_search::queryparser::flexible::{Operator, StandardQueryParser};

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/queryparser_flexible")
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
                out.push(
                    char::from_u32(u32::from_str_radix(&h, 16).unwrap()).unwrap_or('\u{fffd}'),
                );
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

/// `GenQueryParserClassic.synonyms()`: whitespace, lower case, and the
/// single-word synonyms fast = quick, car = auto.
struct Synonyms(Arc<SynonymMap>);

impl AnalyzerDefinition for Synonyms {
    fn create_components(&self, _field: &str) -> Result<TokenStreamComponents, AnalysisError> {
        Ok(TokenStreamComponents::new(SynonymGraphFilter::new(
            LowerCaseFilter::new(WhitespaceTokenizer::new()),
            Arc::clone(&self.0),
            true,
        )))
    }
}

fn synonyms() -> Analyzer {
    let mut b = SynonymMapBuilder::new(true);
    b.add("fast", "quick", true).unwrap();
    b.add("car", "auto", true).unwrap();
    Analyzer::new(Synonyms(Arc::new(b.build().unwrap())))
}

/// `GenQueryParserFlexible.mapping()`: whitespace, lower case, and a
/// `MappingCharFilter` on both readers rewriting `\*` to `S` and `\?` to `Q`
/// -- a normaliser that sees across a wildcard chunk boundary.
struct Mapping(Arc<NormalizeCharMap>);

impl AnalyzerDefinition for Mapping {
    fn create_components(&self, _field: &str) -> Result<TokenStreamComponents, AnalysisError> {
        Ok(TokenStreamComponents::new(LowerCaseFilter::new(
            WhitespaceTokenizer::new(),
        )))
    }

    fn normalize(&self, _field: &str, input: Box<dyn TokenStream>) -> Box<dyn TokenStream> {
        Box::new(LowerCaseFilter::new(input))
    }

    fn init_reader(&self, _field: &str, reader: Box<dyn CharReader>) -> Box<dyn CharReader> {
        Box::new(MappingCharFilter::new(Arc::clone(&self.0), reader))
    }

    fn init_reader_for_normalization(
        &self,
        _field: &str,
        reader: Box<dyn CharReader>,
    ) -> Box<dyn CharReader> {
        Box::new(MappingCharFilter::new(Arc::clone(&self.0), reader))
    }
}

fn mapping() -> Analyzer {
    let mut b = NormalizeCharMapBuilder::new();
    b.add("\\*", "S").unwrap();
    b.add("\\?", "Q").unwrap();
    Analyzer::new(Mapping(Arc::new(b.build())))
}

/// `GenQueryParserFlexible.parser(config)`.
fn parser(config: &str) -> StandardQueryParser {
    let analyzer = match config {
        "syn" => Some(synonyms()),
        "stop" => Some(Analyzer::new(StopAnalyzer::new(Arc::new(
            CharArraySet::from(&english_stop_words()),
        )))),
        "noanalyzer" => None,
        "mapping" => Some(mapping()),
        _ => Some(standard()),
    };
    let mut p = if config.starts_with("prec") {
        StandardQueryParser::precedence(None)
    } else {
        StandardQueryParser::new()
    };
    p.set_analyzer(analyzer.map(Arc::new));
    match config {
        "and" | "prec_and" => p.set_default_operator(Operator::And),
        "multi" => p.set_multi_fields(&["title", "body"]),
        "options" => {
            p.set_allow_leading_wildcard(true);
            p.set_phrase_slop(2);
            p.set_fuzzy_prefix_length(1);
            p.set_fuzzy_min_sim(0.5);
            p.set_enable_position_increments(false);
            p.set_fields_boost(&[("f", 2.0), ("title", 0.5)]);
        }
        "negative" => {
            p.set_fuzzy_prefix_length(-1);
            p.set_phrase_slop(-1);
            p.set_fields_boost(&[("f", -1.0)]);
        }
        "rewrite" => p.set_multi_term_rewrite_method(RewriteMethod::ScoringBoolean),
        "topterms" => p.set_multi_term_rewrite_method(RewriteMethod::TopTermsBoostOnlyBoolean(5)),
        _ => {}
    }
    p
}

#[test]
fn standard_query_parser_matches_lucene() {
    let mut total = 0;
    let mut bad = Vec::new();
    for config in [
        "std",
        "and",
        "noanalyzer",
        "multi",
        "options",
        "syn",
        "stop",
        "negative",
        "rewrite",
        "topterms",
        "prec",
        "prec_and",
        "mapping",
    ] {
        let p = parser(config);
        let field = (config != "multi").then_some("f");
        let tsv = fs::read_to_string(dir().join(format!("{config}.tsv"))).unwrap();
        for line in tsv.lines() {
            let cols: Vec<&str> = line.splitn(4, '\t').collect();
            let query = unesc(cols[0]);
            let want = match cols[1] {
                "Q" => format!("Q\t{}\t{}", unesc(cols[2]), unesc(cols[3])),
                _ => format!("X\t{}", unesc(cols[2])),
            };
            let got = match p.parse(&query, field) {
                Ok(q) => format!("Q\t{}\t{}", q.to_string("f"), q.describe("f")),
                Err(e) => format!("X\t{}: {}", e.class, e.message),
            };
            total += 1;
            if got != want {
                bad.push(format!(
                    "{config} {query:?}\n  want {want:?}\n  got  {got:?}"
                ));
            }
        }
    }
    // the classic parser's seeded fragment joins under std: query, the
    // token columns, then the parse
    let p = parser("std");
    let tsv = fs::read_to_string(dir().join("fragments.tsv")).unwrap();
    for line in tsv.lines() {
        let cols: Vec<&str> = line.split('\t').collect();
        let query = unesc(cols[0]);
        let want = match cols[3] {
            "Q" => format!("Q\t{}\t{}", unesc(cols[4]), unesc(cols[5])),
            _ => format!("X\t{}", unesc(cols[4])),
        };
        let got = match p.parse(&query, Some("f")) {
            Ok(q) => format!("Q\t{}\t{}", q.to_string("f"), q.describe("f")),
            Err(e) => format!("X\t{}: {}", e.class, e.message),
        };
        total += 1;
        if got != want {
            bad.push(format!(
                "fragments {query:?}\n  want {want:?}\n  got  {got:?}"
            ));
        }
    }
    assert!(total > 5000, "only {total} rows");
    let shown: Vec<&str> = bad.iter().take(40).map(String::as_str).collect();
    assert!(
        bad.is_empty(),
        "{} of {total} differ:\n{}",
        bad.len(),
        shown.join("\n")
    );
}

/// `GenQueryParserFlexible.writeDates`' configurations, in its default
/// time zone (GMT-05:00) and locale (`en-US`), at its clock
/// (`date_clock.txt`), which its two-digit years were read against.
#[test]
fn date_ranges_match_lucene() {
    const HOUR: i32 = 3_600_000;
    let clock: i64 = fs::read_to_string(dir().join("date_clock.txt"))
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    let mut total = 0;
    let mut bad = Vec::new();
    for config in ["us_day", "de_map", "root_ms", "none"] {
        let mut p = StandardQueryParser::with_analyzer(standard());
        p.set_default_time_zone(-5 * HOUR);
        p.set_date_clock(Some(clock));
        let locale = |tag: &str| DateLocale::for_language_tag(tag).unwrap();
        match config {
            "us_day" => {
                p.set_date_resolution(Some(Resolution::Day));
                p.set_time_zone(2 * HOUR);
            }
            "de_map" => {
                p.set_locale(locale("de-DE"));
                p.set_date_resolution_map(HashMap::from([("d".to_string(), Resolution::Minute)]));
            }
            "root_ms" => {
                p.set_locale(locale("und"));
                p.set_date_resolution(Some(Resolution::Millisecond));
                p.set_time_zone(5 * HOUR + HOUR / 2);
            }
            _ => {}
        }
        let tsv = fs::read_to_string(dir().join(format!("date_{config}.tsv"))).unwrap();
        for line in tsv.lines() {
            let cols: Vec<&str> = line.splitn(4, '\t').collect();
            let query = unesc(cols[0]);
            let want = match cols[1] {
                "Q" => format!("Q\t{}\t{}", unesc(cols[2]), unesc(cols[3])),
                _ => format!("X\t{}", unesc(cols[2])),
            };
            let got = match p.parse(&query, Some("f")) {
                Ok(q) => format!("Q\t{}\t{}", q.to_string("f"), q.describe("f")),
                Err(e) => format!("X\t{}: {}", e.class, e.message),
            };
            total += 1;
            if got != want {
                bad.push(format!(
                    "{config} {query:?}\n  want {want:?}\n  got  {got:?}"
                ));
            }
        }
    }
    assert!(total > 150, "only {total} rows");
    assert!(
        bad.is_empty(),
        "{} of {total} differ:\n{}",
        bad.len(),
        bad.join("\n")
    );
}

/// `QueryParserUtil`'s helpers over `GenQueryParserFlexible.util()`'s calls.
#[test]
fn query_parser_util_matches_lucene() {
    use lucene_search::query_visitor::Occur;
    use lucene_search::queryparser::flexible::query_parser_util as util;
    let a = || Arc::new(standard());
    let calls = [
        util::parse(&["a", "b c"], &["f", "g"], a()),
        util::parse(&["a"], &["f", "g"], a()),
        util::parse_fields("a b", &["f", "g"], &[Occur::Must, Occur::MustNot], a()),
        util::parse_fields("a", &["f"], &[Occur::Must, Occur::MustNot], a()),
        util::parse_all(
            &["a", "\"b c\""],
            &["f", "g"],
            &[Occur::Should, Occur::Filter],
            a(),
        ),
        util::parse_all(&["a"], &["f", "g"], &[Occur::Should], a()),
        util::parse(&["a("], &["f"], a()),
    ];
    let tsv = fs::read_to_string(dir().join("util.tsv")).unwrap();
    let rows: Vec<&str> = tsv.lines().collect();
    assert_eq!(rows.len(), calls.len() + 1);
    for (row, got) in rows.iter().zip(calls) {
        let cols: Vec<String> = row.split('\t').map(unesc).collect();
        let want = cols[1..].join("\t");
        let got = match got {
            Ok(q) => format!("{}\t{}", q.to_string("f"), q.describe("f")),
            Err(e) => format!("{}: {}", e.class, e.message),
        };
        assert_eq!(got, want);
    }
    let escape = unesc(rows[rows.len() - 1].split('\t').nth(1).unwrap());
    assert_eq!(
        util::escape("a+b-c!(d):e^[f]\"g{h}~i*j?k|l&m/n\\o p"),
        escape
    );
}
