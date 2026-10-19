//! M13 T13.1: the classic `QueryParser` and `MultiFieldQueryParser`
//! against Lucene 10.5.0 (`fixtures/src/GenQueryParserClassic.java`): every
//! query string of `fixtures/corpus/queryparser-classic.txt` and 3,000
//! seeded joins of syntax fragments, under sixteen parser configurations, to
//! the same query tree (`toString("f")` and the class tree) or the same
//! exception message (a `ParseException`'s token, column and expected
//! tokens included); `MultiFieldQueryParser`'s static `parse` helpers; and
//! range bounds read as dates (`date_*.tsv`) under seven locale, resolution
//! and time zone configurations.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use lucene_analysis::core_analysis::{StopAnalyzer, WhitespaceAnalyzer};
use lucene_analysis::miscellaneous::{DateLocale, PerFieldAnalyzerWrapper};
use lucene_analysis::synonym::{SynonymGraphFilter, SynonymMap, SynonymMapBuilder};
use lucene_analysis::util::WhitespaceTokenizer;
use lucene_analysis::{
    english_stop_words, AnalysisError, Analyzer, AnalyzerDefinition, CharArraySet, LowerCaseFilter,
    StandardAnalyzer, TokenStreamComponents,
};
use lucene_index::document::Resolution;
use lucene_search::extended_query::RewriteMethod;
use lucene_search::query_visitor::Occur;
use lucene_search::queryparser::classic::{Operator, QueryParser, QueryParserError};
use lucene_search::queryparser::ParsedQuery;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/queryparser_classic")
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

fn stop() -> Analyzer {
    let words = CharArraySet::from(&english_stop_words());
    Analyzer::new(StopAnalyzer::new(Arc::new(words)))
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

/// The parser of `GenQueryParserClassic`'s configuration `name`.
fn parser(name: &str) -> QueryParser {
    match name {
        "std" => QueryParser::new("f", standard()),
        "std_and" => {
            let mut p = QueryParser::new("f", standard());
            p.set_default_operator(Operator::And);
            p
        }
        "ws" => QueryParser::new("f", Analyzer::new(WhitespaceAnalyzer::default())),
        "std_leading" => {
            let mut p = QueryParser::new("f", standard());
            p.set_allow_leading_wildcard(true);
            p
        }
        "std_autophrase" => {
            let mut p = QueryParser::new("f", standard());
            p.set_split_on_whitespace(true).unwrap();
            p.set_auto_generate_phrase_queries(true).unwrap();
            p
        }
        "std_split" => {
            let mut p = QueryParser::new("f", standard());
            p.set_split_on_whitespace(true).unwrap();
            p
        }
        "std_tuned" => {
            let mut p = QueryParser::new("f", standard());
            p.set_fuzzy_min_sim(1.0);
            p.set_fuzzy_prefix_length(2);
            p.set_phrase_slop(1);
            p.set_enable_position_increments(false);
            p
        }
        "stop" => QueryParser::new("f", stop()),
        "keyword" => QueryParser::new("f", Analyzer::keyword()),
        "multi" => QueryParser::multi_field(&["title", "body"], standard()),
        "multi_boost" => QueryParser::multi_field_with_boosts(
            &["title", "body"],
            standard(),
            HashMap::from([("title".to_string(), 2.0), ("body".to_string(), 0.5)]),
        ),
        "syn" => QueryParser::new("f", synonyms()),
        "std_negative" => {
            let mut p = QueryParser::new("f", standard());
            p.set_fuzzy_prefix_length(-1);
            p.set_phrase_slop(-1);
            p
        }
        "multi_negative" => {
            let mut p = QueryParser::multi_field(&["title", "body"], standard());
            p.set_phrase_slop(-1);
            p
        }
        "multi_perfield" => QueryParser::multi_field(
            &["title", "body"],
            PerFieldAnalyzerWrapper::new(
                standard(),
                HashMap::from([("title".to_string(), Analyzer::keyword())]),
            )
            .into_analyzer(),
        ),
        "multi_and" => {
            let mut p = QueryParser::multi_field(&["title", "body"], standard());
            p.set_default_operator(Operator::And);
            p
        }
        "std_rewrite" => {
            let mut p = QueryParser::new("f", standard());
            p.set_multi_term_rewrite_method(RewriteMethod::ScoringBoolean);
            p
        }
        "std_topterms" => {
            let mut p = QueryParser::new("f", standard());
            p.set_multi_term_rewrite_method(RewriteMethod::TopTermsScoringBoolean(10));
            p.set_determinize_work_limit(8);
            p
        }
        other => panic!("unknown configuration {other}"),
    }
}

/// An hour in milliseconds, for the fixed time zone offsets.
const HOUR: i32 = 3_600_000;

/// `GenQueryParserClassic.dateParser(name)`, in the generator's default
/// time zone (GMT-05:00) and at its clock (`date_clock.txt`), which its
/// two-digit years were read against.
fn date_parser(name: &str) -> QueryParser {
    let mut p = QueryParser::new("f", standard());
    p.set_default_time_zone(-5 * HOUR);
    let clock = fs::read_to_string(dir().join("date_clock.txt")).unwrap();
    p.set_date_clock(Some(clock.trim().parse().unwrap()));
    let locale = |tag: &str| DateLocale::for_language_tag(tag).unwrap();
    match name {
        "us_day" => {
            p.set_locale(locale("en-US"));
            p.set_date_resolution(Some(Resolution::Day));
            p.set_time_zone(2 * HOUR);
        }
        "us_ms" => {
            p.set_locale(locale("en-US"));
            p.set_date_resolution(Some(Resolution::Millisecond));
        }
        "de_field" => {
            p.set_locale(locale("de-DE"));
            p.set_field_date_resolution("d", Resolution::Minute);
            p.set_time_zone(-3 * HOUR);
        }
        "root_year" => {
            p.set_locale(locale("und"));
            p.set_date_resolution(Some(Resolution::Year));
            p.set_field_date_resolution("d", Resolution::Second);
            p.set_time_zone(5 * HOUR + HOUR / 2);
        }
        "ja_month" => {
            p.set_locale(locale("ja-JP"));
            p.set_date_resolution(Some(Resolution::Month));
        }
        "fr_hour" => {
            p.set_locale(locale("fr-FR"));
            p.set_date_resolution(Some(Resolution::Hour));
            p.set_time_zone(-3 * HOUR);
        }
        "none" => p.set_locale(locale("en-US")),
        other => panic!("unknown date configuration {other}"),
    }
    p
}

#[test]
fn date_ranges_match_lucene() {
    let mut total = 0;
    let mut bad = Vec::new();
    for config in [
        "us_day",
        "us_ms",
        "de_field",
        "root_year",
        "ja_month",
        "fr_hour",
        "none",
    ] {
        let p = date_parser(config);
        let file = format!("date_{config}.tsv");
        let text = fs::read_to_string(dir().join(&file)).unwrap();
        for line in text.lines() {
            let cols: Vec<&str> = line.split('\t').collect();
            let query = unesc(cols[0]);
            total += 1;
            let got = row(p.parse(&query));
            let want = want(&cols);
            if got != want {
                bad.push(format!("{file} {query:?}\n  want {want:?}\n  got  {got:?}"));
            }
        }
    }
    assert!(total > 300, "only {total} rows compared");
    assert!(
        bad.is_empty(),
        "{} of {total} rows differ:\n{}",
        bad.len(),
        bad.join("\n")
    );
}

/// A parse's outcome as a fixture row prints it.
fn row(r: Result<ParsedQuery, QueryParserError>) -> String {
    match r {
        Ok(q) => format!("Q\t{}\t{}", q.to_string("f"), q.describe("f")),
        Err(e) if e.is_parse_exception() => format!("E\t{}", e.message),
        Err(e) => format!("X\t{}: {}", e.class, e.message),
    }
}

/// The row a fixture line expects, after its first (key) column.
fn want(cols: &[&str]) -> String {
    match cols[1] {
        "Q" => format!("Q\t{}\t{}", unesc(cols[2]), unesc(cols[3])),
        k => format!("{k}\t{}", unesc(cols[2])),
    }
}

/// Parses every row of `file` with configuration `config`; the row count
/// and the mismatches.
fn check(file: &str, config: &str) -> (usize, Vec<String>) {
    let p = parser(config);
    let text = fs::read_to_string(dir().join(file)).unwrap();
    let mut rows = 0;
    let mut bad = Vec::new();
    for line in text.lines() {
        let cols: Vec<&str> = line.split('\t').collect();
        let query = unesc(cols[0]);
        rows += 1;
        let got = row(p.parse(&query));
        let want = want(&cols);
        if got != want {
            bad.push(format!("{file} {query:?}\n  want {want:?}\n  got  {got:?}"));
        }
    }
    (rows, bad)
}

#[test]
fn classic_query_parser_matches_lucene() {
    let mut total = 0;
    let mut all_bad = Vec::new();
    for config in [
        "std",
        "std_and",
        "ws",
        "std_leading",
        "std_autophrase",
        "std_split",
        "std_tuned",
        "stop",
        "keyword",
        "multi",
        "multi_boost",
        "multi_and",
        "syn",
        "multi_perfield",
        "std_negative",
        "multi_negative",
        "std_rewrite",
        "std_topterms",
    ] {
        let (n, bad) = check(&format!("{config}.tsv"), config);
        total += n;
        all_bad.extend(bad);
    }
    for config in [
        "std",
        "std_and",
        "ws",
        "std_leading",
        "multi",
        "multi_boost",
        "multi_and",
        "syn",
    ] {
        let (n, bad) = check(&format!("frag_{config}.tsv"), config);
        total += n;
        all_bad.extend(bad);
    }
    assert!(total > 28_000, "only {total} rows compared");
    let shown: Vec<&str> = all_bad.iter().take(40).map(String::as_str).collect();
    assert!(
        all_bad.is_empty(),
        "{} of {total} rows differ:\n{}",
        all_bad.len(),
        shown.join("\n")
    );
}

/// `MultiFieldQueryParser`'s static `parse` helpers (`static.tsv`).
#[test]
fn multi_field_static_helpers_match_lucene() {
    let std = || Arc::new(standard());
    let two = ["title", "body"];
    let (s, m, n) = (Occur::Should, Occur::Must, Occur::MustNot);
    let text = fs::read_to_string(dir().join("static.tsv")).unwrap();
    let mut seen = 0;
    for line in text.lines() {
        let cols: Vec<&str> = line.split('\t').collect();
        let got = row(match cols[0] {
            "fields" => QueryParser::parse_fields(&["foo", "bar baz"], &two, std()),
            "fields_stop" => QueryParser::parse_fields(&["the", "x"], &two, stop()),
            "fields_mismatch" => QueryParser::parse_fields(&["a"], &two, std()),
            "flags" => QueryParser::parse_with_flags("foo bar", &two, &[m, n], std()),
            "flags_mismatch" => QueryParser::parse_with_flags("foo", &two, &[m], std()),
            "fields_flags" => {
                QueryParser::parse_fields_with_flags(&["a", "the"], &two, &[s, m], stop())
            }
            "fields_flags_mismatch" => {
                QueryParser::parse_fields_with_flags(&["a", "b"], &two, &[s], std())
            }
            "fields_error" => QueryParser::parse_fields(&["a", "(b"], &two, std()),
            other => panic!("unknown row {other}"),
        });
        assert_eq!(got, want(&cols), "{}", cols[0]);
        seen += 1;
    }
    assert_eq!(seen, 8);
}
