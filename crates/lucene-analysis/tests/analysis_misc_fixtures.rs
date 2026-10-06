//! M11 T11.6: `ReverseStringFilter` and `CSVUtil` against
//! `fixtures/src/GenAnalysisMisc.java`.

mod support;

use lucene_analysis::reverse::{self, ReverseStringFilter};
use lucene_analysis::util::{csv_util, WhitespaceTokenizer};
use lucene_analysis::Analyzer;
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

#[test]
fn reverse_chains_match_lucene() {
    let lines: Vec<String> =
        std::fs::read_to_string(support::data_dir("analysis_misc") + "lines.txt")
            .unwrap()
            .lines()
            .map(unesc)
            .collect();
    let names = support::fixture_names("analysis_misc");
    assert_eq!(
        support::check_chains_over("analysis_misc", &lines, &names, build),
        5
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
