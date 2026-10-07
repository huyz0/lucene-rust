//! M11 T11.7: `Word2VecSynonymFilterFactory` against Lucene. The
//! `Word2VecSynonym` configurations of `fixtures/corpus/analysis-factories.conf`
//! (`GenAnalysisFactories.java`) are built here, after
//! [`register_factories`] adds the factory to `lucene-analysis`' registry,
//! and compared row for row with `fixtures/data/analysis_factories/`; and
//! the registry then holds exactly Lucene's SPI names.

#[path = "../../lucene-analysis/tests/support/factory_config.rs"]
mod factory_config;
#[path = "../../lucene-analysis/tests/support/mod.rs"]
mod support;

use std::collections::BTreeSet;

use factory_config::{build, configs, parse, stable};
use lucene_analysis::factory::spi;
use lucene_search::word2vec::register_factories;
use support::{analyze_line, corpus, data_dir, esc, normalise_expected};

fn rows(fields: &[&str], lines: &[String]) -> Vec<String> {
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

#[test]
fn word2vec_configurations_match_lucene() {
    register_factories().unwrap();
    let lines = corpus("analysis-factories.txt");
    let mut checked = 0;
    for (name, fields) in configs() {
        if !fields.iter().any(|f| f.contains("Word2VecSynonym")) {
            continue;
        }
        let fields: Vec<&str> = fields.iter().map(String::as_str).collect();
        let actual = rows(&fields, &lines);
        let text = std::fs::read_to_string(format!("{}{name}.tsv", data_dir("analysis_factories")))
            .unwrap();
        let expected: Vec<String> = text.lines().map(normalise_expected).collect();
        for (i, (a, e)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(a, e, "{name}: row {i} differs");
        }
        assert_eq!(actual.len(), expected.len(), "{name}: row count");
        checked += 1;
    }
    assert_eq!(checked, 5);
}

#[test]
fn the_registry_then_holds_lucenes_names() {
    register_factories().unwrap();
    let names = |kind: &str| -> BTreeSet<String> {
        std::fs::read_to_string(format!("{}names.txt", data_dir("analysis_factories")))
            .unwrap()
            .lines()
            .filter_map(|l| l.strip_prefix(kind).and_then(|r| r.strip_prefix('\t')))
            .map(str::to_string)
            .collect()
    };
    let set = |v: Vec<&str>| -> BTreeSet<String> { v.into_iter().map(str::to_string).collect() };
    assert_eq!(set(spi::available_token_filters()), names("F"));
    assert_eq!(set(spi::available_tokenizers()), names("T"));
    assert_eq!(set(spi::available_char_filters()), names("C"));
}
