//! M12 T12.1: Nori against Lucene 10.5.0.
//!
//! `fixtures/src/GenAnalysisNori.java` writes `analysis_nori/`: token rows
//! of `KoreanTokenizer` in every decompound mode, with punctuation, unknown
//! unigrams and a user dictionary, over a Korean corpus written for this
//! project and a seeded stress corpus (with every Nori attribute's
//! reflected values for three of them); the Graphviz lattice of each corpus
//! line; a sample of the system dictionary's entries; `UserDictionary.lookup`
//! results; and analyzer and factory chains. Every row must be equal.
#![allow(clippy::arithmetic_side_effects)] // test code: no value read off disk

#[path = "../../lucene-analysis/tests/support/mod.rs"]
mod support;

#[path = "../../lucene-analysis/tests/support/factory_config.rs"]
mod factory_config;

use std::sync::Arc;

use lucene_analysis::morph::{GraphvizFormatter, MorphData};
use lucene_analysis::reader::StrReader;
use lucene_analysis::{Analyzer, TokenStream, Tokenizer};
use lucene_analysis_nori::dict::{ConnectionCosts, TokenInfoDictionary};
use lucene_analysis_nori::pos::Tag;
use lucene_analysis_nori::{DecompoundMode, KoreanAnalyzer, KoreanTokenizer, UserDictionary};
use support::{corpus, data_dir, esc, normalise_expected};

fn dir() -> String {
    data_dir("analysis_nori")
}

fn user_dict() -> Arc<UserDictionary> {
    let text = std::fs::read_to_string(format!(
        "{}/../../fixtures/corpus/analysis-korean-userdict.txt",
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

/// `GenAnalysisNori.rows`.
fn rows(t: &mut KoreanTokenizer, lines: &[String], attributes: bool) -> Vec<String> {
    let mut out = Vec::new();
    let mut keys = None;
    for (ln, line) in lines.iter().enumerate() {
        let mark = out.len();
        let run = |t: &mut KoreanTokenizer, out: &mut Vec<String>, keys: &mut Option<String>| {
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

fn check(name: &str, mut t: KoreanTokenizer, attributes: bool) {
    let mut lines = corpus("analysis-korean.txt");
    lines.extend(stress());
    let actual = rows(&mut t, &lines, attributes);
    compare(name, &actual, &expected(&format!("tok_{name}.tsv")));
}

#[test]
fn decompound_modes_match_lucene() {
    use DecompoundMode::{Discard, Mixed, None};
    check(
        "discard",
        KoreanTokenizer::new(Option::None, Discard, false, true),
        true,
    );
    check(
        "none",
        KoreanTokenizer::new(Option::None, None, false, true),
        false,
    );
    check(
        "mixed",
        KoreanTokenizer::new(Option::None, Mixed, false, true),
        false,
    );
    check(
        "mixed_punct",
        KoreanTokenizer::new(Option::None, Mixed, false, false),
        true,
    );
    check(
        "discard_unigrams_punct",
        KoreanTokenizer::new(Option::None, Discard, true, false),
        false,
    );
}

#[test]
fn user_dictionary_matches_lucene() {
    let u = user_dict();
    check(
        "user_discard",
        KoreanTokenizer::new(Some(u.clone()), DecompoundMode::Discard, false, true),
        false,
    );
    check(
        "user_mixed_punct",
        KoreanTokenizer::new(Some(u.clone()), DecompoundMode::Mixed, false, false),
        true,
    );
    check(
        "user_none_unigrams",
        KoreanTokenizer::new(Some(u.clone()), DecompoundMode::None, true, true),
        false,
    );
    let text = std::fs::read_to_string(dir() + "userdict_lookup.tsv").unwrap();
    for (line, want) in corpus("analysis-korean.txt").iter().zip(text.lines()) {
        let units: Vec<u16> = line.encode_utf16().collect();
        let mut row = esc(line);
        for id in u.lookup(&units) {
            row.push_str(&format!("\t{id}"));
        }
        assert_eq!(row, want);
    }
}

#[test]
fn graphviz_lattices_match_lucene() {
    let u = user_dict();
    for (name, mode, user) in [
        ("discard", DecompoundMode::Discard, None),
        ("user_mixed", DecompoundMode::Mixed, Some(u)),
    ] {
        let mut actual = String::new();
        for (ln, line) in corpus("analysis-korean.txt").iter().enumerate() {
            let mut t = KoreanTokenizer::new(user.clone(), mode, false, false);
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

fn tag(t: Option<Tag>) -> String {
    t.map_or("null".to_string(), |t| t.name().to_string())
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
        let morphemes = match m.morphemes(id, &surface, 0, surface.len() as i32) {
            None => "null".to_string(),
            Some(ms) => ms
                .iter()
                .map(|x| format!("{}/{}", esc(&x.surface_form), x.pos_tag.name()))
                .collect::<Vec<_>>()
                .join("+"),
        };
        let actual = [
            m.left_id(id).to_string(),
            m.right_id(id).to_string(),
            m.word_cost(id).to_string(),
            m.pos_type(id).name().to_string(),
            tag(m.left_pos(id)),
            tag(m.right_pos(id)),
            m.reading(id).map_or("null".to_string(), |r| esc(&r)),
            morphemes,
        ];
        assert_eq!(actual.join("\t"), f[3..].join("\t"), "{row}");
        n += 1;
    }
    assert!(n > 2000, "{n}");
}

fn chain_rows(a: &Analyzer, lines: &[String], out: &mut Vec<String>) {
    for (ln, line) in lines.iter().enumerate() {
        support::analyze_line(a, ln, line, out);
    }
    for (ln, line) in lines.iter().enumerate() {
        let n = match a.normalize("f", line) {
            Ok(bytes) => bytes.iter().map(|b| format!("{b:02x}")).collect::<String>(),
            Err(e) => format!("X:{}", support::exception_name(&e)),
        };
        out.push(format!("N\t{ln}\t{n}"));
    }
}

#[test]
fn analyzers_match_lucene() {
    use lucene_analysis_nori::pos_stop::default_stop_tags;
    let lines = corpus("analysis-korean.txt");
    let cases: Vec<(&str, Analyzer)> = vec![
        ("analyzer_default", Analyzer::new(KoreanAnalyzer::default())),
        (
            "analyzer_user_mixed",
            Analyzer::new(KoreanAnalyzer::new(
                Some(user_dict()),
                DecompoundMode::Mixed,
                Arc::new([Tag::Jks, Tag::Jko, Tag::Ef, Tag::Sf].into_iter().collect()),
                false,
            )),
        ),
        (
            "analyzer_none_unigrams",
            Analyzer::new(KoreanAnalyzer::new(
                None,
                DecompoundMode::None,
                default_stop_tags(),
                true,
            )),
        ),
    ];
    for (name, a) in cases {
        let mut actual = Vec::new();
        chain_rows(&a, &lines, &mut actual);
        compare(name, &actual, &expected(&format!("chains/{name}.tsv")));
    }
}

#[test]
fn factories_match_lucene() {
    lucene_analysis_nori::register_factories().unwrap();
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/corpus/analysis-nori.conf"
    );
    let lines = corpus("analysis-korean.txt");
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
    assert_eq!(n, 21);
}

/// The hostile sweep (`sweep.tsv`): `hostile.txt`, a seeded corpus of text
/// no tokenizer is tuned for, through every combination of decompound mode,
/// unknown unigrams, punctuation and user dictionary (24), with every
/// attribute, each line's rows checked against Lucene's by digest.
#[test]
fn hostile_sweep_matches_lucene() {
    let lines = support::data_lines("analysis_nori", "hostile.txt");
    let u = user_dict();
    let groups = support::sweep_digests("analysis_nori");
    let mut n = 0;
    for (mode, name) in [
        (DecompoundMode::None, "none"),
        (DecompoundMode::Discard, "discard"),
        (DecompoundMode::Mixed, "mixed"),
    ] {
        for unigrams in [false, true] {
            for punct in [true, false] {
                for user in [false, true] {
                    let config = format!(
                        "{name}_un{}_dp{}_ud{}",
                        u8::from(unigrams),
                        u8::from(punct),
                        u8::from(user)
                    );
                    let (want_config, expected) = &groups[n];
                    assert_eq!(want_config, &config);
                    let mut t =
                        KoreanTokenizer::new(user.then(|| u.clone()), mode, unigrams, punct);
                    let actual = rows(&mut t, &lines, true);
                    support::check_digests(&config, &actual, &lines, expected);
                    n += 1;
                }
            }
        }
    }
    assert_eq!(n, groups.len());
}
