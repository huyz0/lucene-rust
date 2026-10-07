//! M11 T11.6: `ReverseStringFilter`, `CSVUtil`, the deprecated
//! `WordDelimiterFilter` and `DateRecognizerFilter` against
//! `fixtures/src/GenAnalysisMisc.java`.

mod support;

use std::sync::Arc;

use lucene_analysis::miscellaneous::{
    DateRecognizerFilter, SetKeywordMarkerFilter, SimpleDateFormat, WordDelimiterFilter,
    CATENATE_ALL, CATENATE_NUMBERS, CATENATE_WORDS, GENERATE_NUMBER_PARTS, GENERATE_WORD_PARTS,
    IGNORE_KEYWORDS, PRESERVE_ORIGINAL, SPLIT_ON_CASE_CHANGE, SPLIT_ON_NUMERICS,
    STEM_ENGLISH_POSSESSIVE,
};
use lucene_analysis::pattern::PatternReplaceFilter;
use lucene_analysis::reverse::{self, ReverseStringFilter};
use lucene_analysis::util::{csv_util, JavaPattern, WhitespaceTokenizer};
use lucene_analysis::{Analyzer, CharArraySet, KeywordTokenizer};
use support::{chain, comps, esc, unesc};

fn build(name: &str) -> Option<Analyzer> {
    let marker = match name {
        "ws_reverse" => None,
        "ws_reverse_soh" => Some(reverse::START_OF_HEADING_MARKER),
        "ws_reverse_pua" => Some(reverse::PUA_EC00_MARKER),
        "ws_reverse_rtl" => Some(reverse::RTL_DIRECTION_MARKER),
        "ws_reverse_is" => Some(reverse::INFORMATION_SEPARATOR_MARKER),
        _ => return None,
    };
    Some(chain(move || {
        comps(match marker {
            None => ReverseStringFilter::new(WhitespaceTokenizer::new()),
            Some(m) => ReverseStringFilter::with_marker(WhitespaceTokenizer::new(), m),
        })
    }))
}

/// The generator's `WDF_DEFAULT`.
const WDF_DEFAULT: i32 = GENERATE_WORD_PARTS
    | GENERATE_NUMBER_PARTS
    | SPLIT_ON_CASE_CHANGE
    | SPLIT_ON_NUMERICS
    | STEM_ENGLISH_POSSESSIVE;

/// The generator's `all`.
const WDF_ALL: i32 =
    WDF_DEFAULT | CATENATE_WORDS | CATENATE_NUMBERS | CATENATE_ALL | PRESERVE_ORIGINAL;

fn set(words: &[&str]) -> Arc<CharArraySet> {
    Arc::new(CharArraySet::from_words(words, false))
}

fn build_wdf(name: &str) -> Option<Analyzer> {
    Some(match name {
        "wdf_default" => chain(|| {
            comps(WordDelimiterFilter::new(
                WhitespaceTokenizer::new(),
                WDF_DEFAULT,
                None,
            ))
        }),
        "wdf_all" => chain(|| {
            comps(WordDelimiterFilter::new(
                WhitespaceTokenizer::new(),
                WDF_ALL,
                Some(set(&["protected-word", "AT&T"])),
            ))
        }),
        "wdf_catenate_only" => chain(|| {
            comps(WordDelimiterFilter::new(
                WhitespaceTokenizer::new(),
                CATENATE_WORDS | CATENATE_NUMBERS | CATENATE_ALL,
                None,
            ))
        }),
        "wdf_ignore_keywords" => chain(|| {
            comps(WordDelimiterFilter::new(
                SetKeywordMarkerFilter::new(WhitespaceTokenizer::new(), set(&["keyword-term"])),
                WDF_DEFAULT | IGNORE_KEYWORDS | CATENATE_ALL,
                None,
            ))
        }),
        "wdf_illegal_offsets" => chain(|| {
            comps(WordDelimiterFilter::new(
                PatternReplaceFilter::new(
                    WhitespaceTokenizer::new(),
                    JavaPattern::compile("x")?,
                    Some("yy"),
                    true,
                ),
                WDF_ALL,
                None,
            ))
        }),
        "wdf_no_case_numerics" => chain(|| {
            comps(WordDelimiterFilter::new(
                WhitespaceTokenizer::new(),
                GENERATE_WORD_PARTS | GENERATE_NUMBER_PARTS | CATENATE_ALL,
                None,
            ))
        }),
        _ => return None,
    })
}

fn lines(file: &str) -> Vec<String> {
    std::fs::read_to_string(support::data_dir("analysis_misc") + file)
        .unwrap()
        .lines()
        .map(unesc)
        .collect()
}

#[test]
fn reverse_chains_match_lucene() {
    let names = support::fixture_names("analysis_misc")
        .into_iter()
        .filter(|n| !n.starts_with("wdf_") && !n.starts_with("date_"))
        .collect();
    assert_eq!(
        support::check_chains_over("analysis_misc", &lines("lines.txt"), &names, build),
        5
    );
}

#[test]
fn word_delimiter_chains_match_lucene() {
    let names = support::fixture_names("analysis_misc")
        .into_iter()
        .filter(|n| n.starts_with("wdf_"))
        .collect();
    assert_eq!(
        support::check_chains_over("analysis_misc", &lines("wdf_lines.txt"), &names, build_wdf),
        6
    );
}

#[test]
fn csv_util_matches_lucene() {
    let text = std::fs::read_to_string(support::data_dir("analysis_misc") + "csv.words").unwrap();
    for row in text.lines() {
        let f: Vec<&str> = row.split('\t').collect();
        let input = unesc(f[0]);
        assert_eq!(esc(&csv_util::quote_escape(&input)), f[1], "{input}");
        let parsed: Vec<String> = csv_util::parse(&input).iter().map(|s| esc(s)).collect();
        assert_eq!(parsed, f[2..], "{input}");
    }
}

fn build_date(name: &str) -> Option<Analyzer> {
    Some(match name {
        "date_default_kw" => chain(|| comps(DateRecognizerFilter::new(KeywordTokenizer::new()))),
        "date_iso_ws" => chain(|| {
            comps(DateRecognizerFilter::with_format(
                WhitespaceTokenizer::new(),
                SimpleDateFormat::new("yyyy-MM-dd")?,
            ))
        }),
        _ => return None,
    })
}

#[test]
fn date_recognizer_chains_match_lucene() {
    let names = support::fixture_names("analysis_misc")
        .into_iter()
        .filter(|n| n.starts_with("date_"))
        .collect();
    assert_eq!(
        support::check_chains_over(
            "analysis_misc",
            &lines("date_lines.txt"),
            &names,
            build_date
        ),
        2
    );
}

/// `DateFormat.parse` succeeding or not, input by input, per pattern.
#[test]
fn simple_date_format_parses_as_lucene() {
    let text =
        std::fs::read_to_string(support::data_dir("analysis_misc") + "date_formats.txt").unwrap();
    let (mut format, mut pattern) = (None, String::new());
    let (mut checked, mut bad) = (0, Vec::new());
    for row in text.lines() {
        let (a, b) = row.split_once('\t').unwrap();
        if a == "#pattern" {
            pattern = unesc(b);
            format = Some(if pattern == "DEFAULT" {
                SimpleDateFormat::english_default()
            } else {
                SimpleDateFormat::new(&pattern).unwrap()
            });
            continue;
        }
        let input = unesc(a);
        let parses = format.as_ref().unwrap().parses(&input);
        if parses != (b == "1") {
            bad.push(format!("{pattern}: {row}"));
        }
        checked += 1;
    }
    assert!(
        bad.is_empty(),
        "{} of {checked} differ:\n{}",
        bad.len(),
        bad.join("\n")
    );
    assert_eq!(checked, 14_555);
}
