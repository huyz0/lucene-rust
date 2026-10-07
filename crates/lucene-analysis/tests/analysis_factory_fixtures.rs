//! M11 T11.7: the factories and `CustomAnalyzer` against Lucene, from the same
//! configuration text.
//!
//! `fixtures/src/GenAnalysisFactories.java` builds every configuration of
//! `fixtures/corpus/analysis-factories.conf` with `CustomAnalyzer.builder`
//! and writes `fixtures/data/analysis_factories/<name>.tsv`: the build's
//! exception (class and message), or `toString()`, every token of every line
//! of `corpus/analysis-factories.txt` and `normalize` of every line. This
//! test builds the same configurations with the port's builder and compares
//! row for row, and checks that the port registers exactly Lucene's SPI names
//! (`names.txt`).
//!
//! The configurations naming `Word2VecSynonym` are run by
//! `lucene-search/tests/word2vec_factory_fixtures.rs`, where that filter
//! lives.

#[path = "support/factory_config.rs"]
mod factory_config;
mod support;

use std::collections::BTreeSet;

use factory_config::{build, configs, parse, stable};
use lucene_analysis::factory::spi;
use support::{analyze_line, corpus, data_dir, esc, normalise_expected};

/// The filter `lucene-search` registers.
const SEARCH_ONLY: &str = "Word2VecSynonym";

/// The rows `GenAnalysisFactories` writes for one configuration.
pub fn rows(fields: &[&str], lines: &[String]) -> Vec<String> {
    let analyzer = match build(&parse(fields)) {
        Ok(a) => a,
        Err(e) => {
            return vec![format!(
                "B\t{}\t{}",
                e.java_class(),
                esc(&stable(e.java_class(), &e.message))
            )]
        }
    };
    let mut out = vec![format!("S\t{analyzer}")];
    for (ln, line) in lines.iter().enumerate() {
        analyze_line(&analyzer, ln, line, &mut out);
    }
    for (ln, line) in lines.iter().enumerate() {
        let n = match analyzer.normalize("f", line) {
            Ok(bytes) => bytes.iter().map(|b| format!("{b:02x}")).collect::<String>(),
            Err(e) => format!("X:{}", support::exception_name(&e)),
        };
        out.push(format!("N\t{ln}\t{n}"));
    }
    out
}

fn expected(name: &str) -> Vec<String> {
    let text = std::fs::read_to_string(format!("{}{name}.tsv", data_dir("analysis_factories")))
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    text.lines()
        .map(|l| {
            let l = normalise_expected(l);
            // The PatternSyntaxException message is not compared (see stable()).
            match l.strip_prefix("B\tPatternSyntaxException\t") {
                Some(_) => "B\tPatternSyntaxException\t".to_string(),
                None => l,
            }
        })
        .collect()
}

#[test]
fn every_configuration_matches_lucene() {
    let lines = corpus("analysis-factories.txt");
    let mut checked = 0;
    let mut built = 0;
    for (name, fields) in configs() {
        if fields.iter().any(|f| f.contains(SEARCH_ONLY)) {
            continue;
        }
        let fields: Vec<&str> = fields.iter().map(String::as_str).collect();
        let actual = rows(&fields, &lines);
        let expected = expected(&name);
        for (i, (a, e)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(a, e, "{name}: row {i} differs");
        }
        assert_eq!(actual.len(), expected.len(), "{name}: row count");
        if actual[0].starts_with('S') {
            built += 1;
        }
        checked += 1;
    }
    assert!(checked >= 290, "{checked} configurations checked");
    assert!(built >= 190, "{built} configurations built");
}

#[test]
fn every_fixture_has_a_configuration() {
    let names: BTreeSet<String> = configs().into_iter().map(|(n, _)| n).collect();
    let fixtures = support::fixture_names("analysis_factories");
    assert_eq!(names, fixtures);
}

/// The SPI names Lucene registers, by kind (`T`, `F`, `C`).
pub fn lucene_names(kind: &str) -> BTreeSet<String> {
    std::fs::read_to_string(format!("{}names.txt", data_dir("analysis_factories")))
        .unwrap()
        .lines()
        .filter_map(|l| l.strip_prefix(kind).and_then(|r| r.strip_prefix('\t')))
        .map(str::to_string)
        .collect()
}

fn set(names: Vec<&str>) -> BTreeSet<String> {
    names.into_iter().map(str::to_string).collect()
}

#[test]
fn the_registered_names_are_lucenes() {
    assert_eq!(set(spi::available_tokenizers()), lucene_names("T"));
    assert_eq!(set(spi::available_char_filters()), lucene_names("C"));
    let mut filters = set(spi::available_token_filters());
    filters.insert(SEARCH_ONLY.to_string());
    assert_eq!(filters, lucene_names("F"));
}
