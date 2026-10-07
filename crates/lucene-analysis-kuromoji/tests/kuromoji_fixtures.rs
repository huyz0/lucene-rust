//! M12 T12.1: Kuromoji against Lucene 10.5.0.
//!
//! `fixtures/src/GenAnalysisKuromoji.java` writes `analysis_kuromoji/`:
//! token rows of `JapaneseTokenizer` in every mode, with punctuation,
//! compounds, n-best costs and a user dictionary, over a Japanese corpus
//! written for this project and a seeded stress corpus (with every Kuromoji
//! attribute's reflected values for four of them); the Graphviz lattice of
//! each corpus line; `calcNBestCost` results; a sample of the system
//! dictionary's entries; and `UserDictionary.lookup` results. Every row must
//! be equal.
#![allow(clippy::arithmetic_side_effects)] // test code: no value read off disk

#[path = "../../lucene-analysis/tests/support/mod.rs"]
mod support;

use std::sync::Arc;

use lucene_analysis::morph::{GraphvizFormatter, MorphData};
use lucene_analysis::reader::StrReader;
use lucene_analysis::{TokenStream, Tokenizer};
use lucene_analysis_kuromoji::dict::{ConnectionCosts, TokenInfoDictionary};
use lucene_analysis_kuromoji::{JapaneseTokenizer, Mode, UserDictionary};
use support::{corpus, data_dir, esc, normalise_expected};

fn dir() -> String {
    data_dir("analysis_kuromoji")
}

fn user_dict() -> Arc<UserDictionary> {
    let text = std::fs::read_to_string(format!(
        "{}/../../fixtures/corpus/analysis-japanese-userdict.txt",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap();
    Arc::new(UserDictionary::open(&text).unwrap().unwrap())
}

fn stress() -> Vec<String> {
    let text = std::fs::read_to_string(dir() + "stress.txt").unwrap();
    let mut lines: Vec<String> = text.split('\n').map(str::to_string).collect();
    lines.pop();
    lines
}

fn opt(v: Option<String>) -> String {
    v.map_or_else(|| "null".to_string(), |s| esc(&s))
}

/// `GenAnalysisKuromoji.rows`.
fn rows(t: &mut JapaneseTokenizer, lines: &[String], attributes: bool) -> Vec<String> {
    let mut out = Vec::new();
    let mut keys = None;
    for (ln, line) in lines.iter().enumerate() {
        let mark = out.len();
        let run = |t: &mut JapaneseTokenizer, out: &mut Vec<String>, keys: &mut Option<String>| {
            t.set_reader(Box::new(StrReader::new(line.clone())))?;
            t.reset()?;
            while t.increment_token()? {
                let a = t.attributes();
                let mut row = format!(
                    "T\t{ln}\t{}\t{}\t{}\t{}\t{}",
                    esc(a.term()),
                    a.start_offset(),
                    a.end_offset(),
                    a.position_increment(),
                    a.position_length()
                );
                if attributes {
                    let mut k = String::from("K");
                    a.reflect_custom(&mut |class, key, value| {
                        let simple = class.rsplit('.').next().unwrap();
                        k.push_str(&format!("\t{simple}#{key}"));
                        row.push('\t');
                        row.push_str(&match value.to_string() {
                            v if v == "null" => v,
                            v => esc(&v),
                        });
                    });
                    keys.get_or_insert(k);
                }
                out.push(row);
            }
            t.end()?;
            let a = t.attributes();
            out.push(format!(
                "E\t{ln}\t{}\t{}\t{}",
                a.start_offset(),
                a.end_offset(),
                a.position_increment()
            ));
            Ok::<(), lucene_analysis::AnalysisError>(())
        };
        if let Err(e) = run(t, &mut out, &mut keys) {
            out.truncate(mark);
            out.push(format!("X\t{ln}\t{}", support::exception_name(&e)));
        }
        let _ = t.close();
    }
    if let Some(k) = keys {
        out.insert(0, k);
    }
    out
}

fn expected(name: &str) -> Vec<String> {
    std::fs::read_to_string(format!("{}{name}", dir()))
        .unwrap_or_else(|e| panic!("{name}: {e}"))
        .lines()
        .map(normalise_expected)
        .collect()
}

fn compare(name: &str, actual: &[String], expected: &[String]) {
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(a, e, "{name}: row {i}");
    }
    assert_eq!(actual.len(), expected.len(), "{name}: row count");
}

fn lines() -> Vec<String> {
    let mut l = corpus("analysis-japanese.txt");
    l.extend(stress());
    l
}

fn check(name: &str, mut t: JapaneseTokenizer, attributes: bool) {
    let actual = rows(&mut t, &lines(), attributes);
    compare(name, &actual, &expected(&format!("tok_{name}.tsv")));
}

#[test]
fn normal_mode_matches_lucene() {
    check(
        "normal",
        JapaneseTokenizer::with_options(None, true, true, Mode::Normal),
        true,
    );
    check(
        "normal_punct",
        JapaneseTokenizer::with_options(None, false, true, Mode::Normal),
        false,
    );
}

#[test]
fn search_mode_matches_lucene() {
    check(
        "search",
        JapaneseTokenizer::with_options(None, true, true, Mode::Search),
        false,
    );
    check(
        "search_compound",
        JapaneseTokenizer::with_options(None, true, false, Mode::Search),
        true,
    );
    check(
        "search_punct_compound",
        JapaneseTokenizer::with_options(None, false, false, Mode::Search),
        false,
    );
}

#[test]
fn extended_mode_matches_lucene() {
    check(
        "extended",
        JapaneseTokenizer::with_options(None, true, true, Mode::Extended),
        false,
    );
    check(
        "extended_punct_compound",
        JapaneseTokenizer::with_options(None, false, false, Mode::Extended),
        false,
    );
}

#[test]
fn user_dictionary_matches_lucene() {
    let u = user_dict();
    check(
        "user_normal",
        JapaneseTokenizer::with_options(Some(u.clone()), true, true, Mode::Normal),
        false,
    );
    check(
        "user_search_compound",
        JapaneseTokenizer::with_options(Some(u.clone()), false, false, Mode::Search),
        true,
    );
    check(
        "user_extended",
        JapaneseTokenizer::with_options(Some(u.clone()), true, true, Mode::Extended),
        false,
    );
    let text = std::fs::read_to_string(dir() + "userdict_lookup.tsv").unwrap();
    for (line, want) in corpus("analysis-japanese.txt").iter().zip(text.lines()) {
        let units: Vec<u16> = line.encode_utf16().collect();
        let mut row = esc(line);
        for [a, b, c] in u.lookup(&units) {
            row.push_str(&format!("\t{a},{b},{c}"));
        }
        assert_eq!(row, want);
    }
}

#[test]
fn nbest_matches_lucene() {
    let u = user_dict();
    for (name, mode, cost, attrs) in [
        ("nbest_normal_500", Mode::Normal, 500, false),
        ("nbest_normal_2000", Mode::Normal, 2000, true),
        ("nbest_search_2000", Mode::Search, 2000, false),
        ("nbest_extended_1000", Mode::Extended, 1000, false),
        ("nbest_normal_10000", Mode::Normal, 10000, false),
    ] {
        let mut t = JapaneseTokenizer::with_options(None, true, true, mode);
        t.set_n_best_cost(cost);
        check(name, t, attrs);
    }
    let mut t = JapaneseTokenizer::with_options(Some(u), false, false, Mode::Search);
    t.set_n_best_cost(2000);
    check("nbest_user_punct_2000", t, false);

    let text = std::fs::read_to_string(dir() + "nbest_examples.tsv").unwrap();
    for row in text.lines() {
        let f: Vec<&str> = row.split('\t').collect();
        let mode = Mode::value_of(f[1]).unwrap();
        let mut t = JapaneseTokenizer::with_options(None, true, true, mode);
        let actual = match t.calc_n_best_cost(&support::unesc(f[0])) {
            Ok(v) => v.to_string(),
            Err(_) => "!RuntimeException".to_string(),
        };
        assert_eq!(actual, f[2], "{row}");
    }
}

#[test]
fn graphviz_lattices_match_lucene() {
    let u = user_dict();
    for (name, mode, user) in [
        ("normal", Mode::Normal, None),
        ("user_search", Mode::Search, Some(u)),
    ] {
        let mut actual = String::new();
        for (ln, line) in corpus("analysis-japanese.txt").iter().enumerate() {
            let mut t = JapaneseTokenizer::with_options(user.clone(), false, true, mode);
            t.set_graphviz_formatter(GraphvizFormatter::new(ConnectionCosts::instance()));
            t.set_reader(Box::new(StrReader::new(line.clone())))
                .unwrap();
            t.reset().unwrap();
            while t.increment_token().unwrap() {}
            t.end().unwrap();
            t.close().unwrap();
            actual.push_str(&format!("== {ln}\n{}\n", t.graphviz().unwrap().finish()));
        }
        let want = std::fs::read_to_string(format!("{}graphviz_{name}.txt", dir())).unwrap();
        compare(
            name,
            &actual.lines().map(str::to_string).collect::<Vec<_>>(),
            &want.lines().map(str::to_string).collect::<Vec<_>>(),
        );
    }
}

#[test]
fn dictionary_entries_match_lucene() {
    let dict = TokenInfoDictionary::instance();
    let fst = dict.fst();
    let m = dict.morph_attributes();
    let text = std::fs::read_to_string(dir() + "dictionary.tsv").unwrap();
    let mut n = 0;
    for row in text.lines() {
        let f: Vec<&str> = row.split('\t').collect();
        let surface: Vec<u16> = support::unesc(f[0]).encode_utf16().collect();
        let mut arc = fst.first_arc();
        let mut output = 0i64;
        for (i, &u) in surface.iter().enumerate() {
            arc = fst
                .find_target_arc(i32::from(u), &arc, i == 0)
                .unwrap()
                .unwrap();
            output += arc.output();
        }
        assert!(arc.is_final());
        output += arc.next_final_output();
        assert_eq!(output.to_string(), f[1], "{row}");
        let id: i32 = f[2].parse().unwrap();
        assert!(dict.lookup_word_ids(output as i32).contains(&id));
        let len = surface.len() as i32;
        let o = |v: Option<&str>| v.map_or("null".to_string(), esc);
        let actual = [
            m.left_id(id).to_string(),
            m.right_id(id).to_string(),
            m.word_cost(id).to_string(),
            o(m.part_of_speech(id)),
            o(m.inflection_type(id)),
            o(m.inflection_form(id)),
            opt(m.base_form(id, &surface, 0, len)),
            esc(&m.reading(id, &surface, 0, len)),
            esc(&m.pronunciation(id, &surface, 0, len, false)),
        ];
        assert_eq!(actual.join("\t"), f[3..].join("\t"), "{row}");
        n += 1;
    }
    assert!(n > 4000, "{n}");
}

fn chain_lines() -> Vec<String> {
    let mut l = corpus("analysis-japanese.txt");
    l.extend(corpus("analysis-japanese-filters.txt"));
    l
}

fn normalized(a: &lucene_analysis::Analyzer, lines: &[String], out: &mut Vec<String>) {
    for (ln, line) in lines.iter().enumerate() {
        let n = match a.normalize("f", line) {
            Ok(bytes) => bytes.iter().map(|b| format!("{b:02x}")).collect::<String>(),
            Err(e) => format!("X:{}", support::exception_name(&e)),
        };
        out.push(format!("N\t{ln}\t{n}"));
    }
}

fn chain_rows(a: &lucene_analysis::Analyzer, lines: &[String], out: &mut Vec<String>) {
    for (ln, line) in lines.iter().enumerate() {
        support::analyze_line(a, ln, line, out);
    }
    normalized(a, lines, out);
}

#[test]
fn analyzers_match_lucene() {
    use lucene_analysis::{Analyzer, CharArraySet};
    use lucene_analysis_kuromoji::analyzer::{default_stop_set, default_stop_tags};
    use lucene_analysis_kuromoji::completion::CompletionMode;
    use lucene_analysis_kuromoji::{JapaneseAnalyzer, JapaneseCompletionAnalyzer};
    let u = user_dict();
    let lines = chain_lines();
    let cases: Vec<(&str, Analyzer)> = vec![
        (
            "analyzer_default",
            Analyzer::new(JapaneseAnalyzer::default()),
        ),
        (
            "analyzer_user_normal",
            Analyzer::new(JapaneseAnalyzer::new(
                Some(u.clone()),
                Mode::Normal,
                default_stop_set(),
                default_stop_tags(),
            )),
        ),
        (
            "analyzer_extended_nostop",
            Analyzer::new(JapaneseAnalyzer::new(
                None,
                Mode::Extended,
                Arc::new(CharArraySet::empty()),
                Arc::new(Default::default()),
            )),
        ),
        (
            "completion_index",
            Analyzer::new(JapaneseCompletionAnalyzer::default()),
        ),
        (
            "completion_query",
            Analyzer::new(JapaneseCompletionAnalyzer::new(
                Some(u),
                CompletionMode::Query,
            )),
        ),
    ];
    for (name, a) in cases {
        let mut actual = Vec::new();
        chain_rows(&a, &lines, &mut actual);
        compare(name, &actual, &expected(&format!("chains/{name}.tsv")));
    }
}

#[path = "../../lucene-analysis/tests/support/factory_config.rs"]
mod factory_config;

#[test]
fn factories_match_lucene() {
    lucene_analysis_kuromoji::register_factories().unwrap();
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/corpus/analysis-kuromoji.conf"
    );
    let lines = chain_lines();
    let mut n = 0;
    for config in std::fs::read_to_string(path).unwrap().lines() {
        if config.is_empty() || config.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = config.split('\t').collect();
        let actual = match factory_config::build(&factory_config::parse(&fields)) {
            Err(e) => vec![format!(
                "B\t{}\t{}",
                e.java_class(),
                esc(&factory_config::stable(e.java_class(), &e.message))
            )],
            Ok(a) => {
                let mut out = vec![format!("S\t{a}")];
                chain_rows(&a, &lines, &mut out);
                out
            }
        };
        compare(
            fields[0],
            &actual,
            &expected(&format!("chains/{}.tsv", fields[0])),
        );
        n += 1;
    }
    assert_eq!(n, 40);
}

#[test]
fn to_string_util_matches_lucene() {
    use lucene_analysis_kuromoji::dict::to_string_util::{
        inflected_form_translation, inflection_type_translation, pos_translation, romanization,
    };
    let text = std::fs::read_to_string(dir() + "romanization.tsv").unwrap();
    let mut n = 0;
    for row in text.lines() {
        let (input, want) = row.split_once('\t').unwrap();
        assert_eq!(romanization(input), want, "{input}");
        n += 1;
    }
    assert_eq!(n, 18624);
    let text = std::fs::read_to_string(dir() + "translations.tsv").unwrap();
    for row in text.lines() {
        let f: Vec<&str> = row.split('\t').collect();
        let actual = match f[0] {
            "posTranslations" => pos_translation(f[1]),
            "inflTypeTranslations" => inflection_type_translation(f[1]),
            _ => inflected_form_translation(f[1]),
        };
        assert_eq!(actual, Some(f[2]), "{row}");
    }
}
