//! M12 T12.2: Stempel against Lucene 10.5.0.
//!
//! `fixtures/src/GenAnalysisStempel.java` writes `analysis_stempel/`:
//! `stems.tsv` (30,000+ words through Lucene's default Polish table),
//! `tables/<method>.tbl` (tables Egothor's `Compile` built from
//! `corpus/analysis-stempel-train.txt`, every kind and optimiser) with
//! `tables/<method>.tsv` (words through each), `PolishAnalyzer` and
//! `StempelFilter` chains and the factory over `corpus/analysis-stempel.txt`.
//! Every stem and row must be equal.
#![allow(clippy::arithmetic_side_effects)] // test code: no value read off disk

#[path = "../../lucene-analysis/tests/support/factory_config.rs"]
mod factory_config;
#[path = "../../lucene-analysis/tests/support/mod.rs"]
mod support;

use std::sync::Arc;

use lucene_analysis::util::WhitespaceTokenizer;
use lucene_analysis::{Analyzer, CharArraySet, LowerCaseFilter};
use lucene_analysis_stempel::stemmer::default_table;
use lucene_analysis_stempel::{PolishAnalyzer, StempelFilter, StempelStemmer};
use support::{analyze_line, chain, comps, corpus, data_dir, esc, normalise_expected, unesc};

fn stem_value(s: &StempelStemmer, word: &str) -> String {
    let units: Vec<u16> = word.encode_utf16().collect();
    match s.stem(&units) {
        Ok(Some(v)) => format!("={}", esc(&String::from_utf16_lossy(&v))),
        Ok(None) => "~".to_string(),
        Err(_) => "!StringIndexOutOfBoundsException".to_string(),
    }
}

fn check_stems(s: &StempelStemmer, file: &str) -> usize {
    let text = std::fs::read_to_string(data_dir("analysis_stempel") + file).unwrap();
    let mut failures = Vec::new();
    let mut n = 0;
    for line in text.lines() {
        let (word, expected) = line.split_once('\t').unwrap();
        let word = unesc(word);
        let actual = stem_value(s, &word);
        n += 1;
        if actual != normalise_expected(expected) && failures.len() < 30 {
            failures.push(format!("{word:?}: java {expected} rust {actual}"));
        }
    }
    assert!(failures.is_empty(), "{file}:\n{}", failures.join("\n"));
    n
}

#[test]
fn default_table_stems_as_lucene() {
    let n = check_stems(&StempelStemmer::new(default_table()), "stems.tsv");
    assert!(n > 30_000, "{n} words");
}

#[test]
fn compiled_tables_read_and_stem_as_lucene() {
    let dir = data_dir("analysis_stempel") + "tables/";
    let mut tables = 0;
    for e in std::fs::read_dir(&dir).unwrap() {
        let name = e.unwrap().file_name().into_string().unwrap();
        let Some(stem) = name.strip_suffix(".tbl") else {
            continue;
        };
        let bytes = std::fs::read(format!("{dir}{name}")).unwrap();
        let s = StempelStemmer::load(&bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
        check_stems(&s, &format!("tables/{stem}.tsv"));
        // Every truncation of a table is an error, never a panic.
        for cut in (0..bytes.len()).step_by(97) {
            assert!(
                StempelStemmer::load(&bytes[..cut]).is_err(),
                "{name} cut at {cut}"
            );
        }
        // Every single-byte flip reads or fails, and what reads stems
        // without panicking.
        for i in 0..bytes.len() {
            let mut flipped = bytes.clone();
            flipped[i] ^= 0xFF;
            if let Ok(s) = StempelStemmer::load(&flipped) {
                for w in ["kotami", "psa", "dobrego", "a"] {
                    let _ = s.stem(&w.encode_utf16().collect::<Vec<_>>());
                }
            }
        }
        tables += 1;
    }
    assert_eq!(tables, 8);
}

fn analyzer(name: &str) -> Analyzer {
    let table = default_table();
    match name {
        "polish" => Analyzer::new(PolishAnalyzer::default()),
        "polish_exclusions" => {
            let mut ex = CharArraySet::with_capacity(4, false);
            for w in ["kot", "domu", "dzieci", "Łodzi"] {
                ex.add(w);
            }
            Analyzer::new(PolishAnalyzer::with_exclusions(
                &lucene_analysis_stempel::polish::default_stop_set(),
                &ex,
            ))
        }
        "polish_nostop" => Analyzer::new(PolishAnalyzer::new(&CharArraySet::empty())),
        "stempel_min1" => chain(move || {
            let s = StempelStemmer::new(Arc::clone(&table));
            comps(StempelFilter::with_min_length(
                LowerCaseFilter::new(WhitespaceTokenizer::new()),
                s,
                1,
            )?)
        }),
        "stempel_min5" => chain(move || {
            let s = StempelStemmer::new(Arc::clone(&table));
            comps(StempelFilter::with_min_length(
                WhitespaceTokenizer::new(),
                s,
                5,
            )?)
        }),
        other => panic!("no chain {other}"),
    }
}

#[test]
fn chains_match_lucene() {
    let names = support::fixture_names("analysis_stempel");
    let chains: Vec<&String> = names
        .iter()
        .filter(|n| !n.starts_with("factory_") && *n != "stems")
        .collect();
    assert_eq!(chains.len(), 5);
    for name in chains {
        let lines = corpus("analysis-stempel.txt");
        let text =
            std::fs::read_to_string(format!("{}{name}.tsv", data_dir("analysis_stempel"))).unwrap();
        let expected: Vec<String> = text.lines().map(normalise_expected).collect();
        support::check_rows(name, &analyzer(name), &lines, &expected);
    }
}

#[test]
fn factories_match_lucene() {
    lucene_analysis_stempel::register_factories().unwrap();
    let lines = corpus("analysis-stempel.txt");
    let configs: [&[&str]; 3] = [
        &[
            "factory_stempel",
            "tok:whitespace",
            "tf:lowercase",
            "tf:stempelPolishStem",
        ],
        &[
            "factory_stempel_keyword",
            "tok:whitespace",
            "tf:keywordMarker",
            "pattern=[A-Z].*",
            "tf:stempelPolishStem",
        ],
        &[
            "factory_stempel_param",
            "tok:whitespace",
            "tf:stempelPolishStem",
            "x=1",
        ],
    ];
    for fields in configs {
        let name = fields[0];
        let text =
            std::fs::read_to_string(format!("{}{name}.tsv", data_dir("analysis_stempel"))).unwrap();
        let expected: Vec<String> = text.lines().map(normalise_expected).collect();
        let actual = match factory_config::build(&factory_config::parse(fields)) {
            Err(e) => vec![format!(
                "B\t{}\t{}",
                e.java_class(),
                esc(&factory_config::stable(e.java_class(), &e.message))
            )],
            Ok(a) => {
                let mut out = vec![format!("S\t{a}")];
                for (ln, line) in lines.iter().enumerate() {
                    analyze_line(&a, ln, line, &mut out);
                }
                for (ln, line) in lines.iter().enumerate() {
                    let n = a.normalize("f", line).unwrap();
                    out.push(format!(
                        "N\t{ln}\t{}",
                        n.iter().map(|b| format!("{b:02x}")).collect::<String>()
                    ));
                }
                out
            }
        };
        assert_eq!(actual, expected, "{name}");
    }
}
