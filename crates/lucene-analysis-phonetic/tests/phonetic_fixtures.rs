//! M12 T12.3: the phonetic encoders and filters against Commons Codec 1.17.2
//! and Lucene 10.5.0.
//!
//! `fixtures/src/GenAnalysisPhonetic.java` writes `analysis_phonetic/`:
//! `words.tsv` and `bm.tsv` (a word list through every encoder and option,
//! a column each), `prefixes.txt` (Beider-Morse's name prefixes in Java's
//! `HashSet` order) and one `<config>.tsv` per configuration of
//! `corpus/analysis-phonetic.conf`, built with `CustomAnalyzer` and run over
//! `corpus/analysis-phonetic.txt`. Every value and row must be equal.
#![allow(clippy::arithmetic_side_effects)] // test code: no value read off disk

#[path = "../../lucene-analysis/tests/support/factory_config.rs"]
mod factory_config;
#[path = "../../lucene-analysis/tests/support/mod.rs"]
mod support;

use std::collections::BTreeSet;

use lucene_analysis_phonetic::bm::{name_prefixes, NameType, PhoneticEngine, RuleType};
use lucene_analysis_phonetic::caverphone::{caverphone1, caverphone2};
use lucene_analysis_phonetic::cologne::cologne_phonetic;
use lucene_analysis_phonetic::daitch_mokotoff::DaitchMokotoffSoundex;
use lucene_analysis_phonetic::double_metaphone::DoubleMetaphone;
use lucene_analysis_phonetic::match_rating::match_rating_encode;
use lucene_analysis_phonetic::metaphone::Metaphone;
use lucene_analysis_phonetic::nysiis::Nysiis;
use lucene_analysis_phonetic::soundex::{RefinedSoundex, Soundex};
use lucene_analysis_phonetic::EncoderError;
use support::{analyze_line, corpus, data_dir, esc, normalise_expected, unesc};

type Column = Box<dyn Fn(&[u16]) -> Result<Option<Vec<u16>>, EncoderError>>;

fn ok(v: Vec<u16>) -> Result<Option<Vec<u16>>, EncoderError> {
    Ok(Some(v))
}

/// `GenAnalysisPhonetic.val`.
fn val(r: Result<Option<Vec<u16>>, EncoderError>) -> String {
    match r {
        Ok(Some(v)) => format!("={}", esc(&String::from_utf16_lossy(&v))),
        Ok(None) => "~".to_string(),
        Err(e) => format!("!{}", e.java_class()),
    }
}

/// `GenAnalysisPhonetic.columns`.
fn column(name: &str) -> Column {
    let max = |prefix: &str| {
        name.strip_prefix(prefix)
            .and_then(|m| m.parse::<i32>().ok())
    };
    if let Some(m) = max("metaphone") {
        let mut e = Metaphone::default();
        e.set_max_code_len(m);
        return Box::new(move |s| ok(e.metaphone(s)));
    }
    for (prefix, alt) in [("dmp", false), ("dma", true)] {
        if let Some(m) = max(prefix) {
            let mut e = DoubleMetaphone::default();
            e.set_max_code_len(m);
            return Box::new(move |s| Ok(e.double_metaphone(s, alt)));
        }
    }
    match name {
        "soundex" => Box::new(|s| Soundex::default().soundex(s).map(Some)),
        "soundex_simplified" => Box::new(|s| Soundex::us_english_simplified().soundex(s).map(Some)),
        "soundex_genealogy" => Box::new(|s| Soundex::us_english_genealogy().soundex(s).map(Some)),
        "refined" => Box::new(|s| ok(RefinedSoundex::default().soundex(s))),
        "caverphone1" => Box::new(|s| ok(caverphone1(s))),
        "caverphone2" => Box::new(|s| ok(caverphone2(s))),
        "cologne" => Box::new(|s| ok(cologne_phonetic(s))),
        "nysiis" => Box::new(|s| ok(Nysiis::default().nysiis(s))),
        "nysiis_loose" => Box::new(|s| ok(Nysiis::new(false).nysiis(s))),
        "mra" => Box::new(|s| ok(match_rating_encode(s))),
        "dms_encode" => Box::new(|s| ok(DaitchMokotoffSoundex::default().encode(s))),
        "dms_soundex" => Box::new(|s| ok(DaitchMokotoffSoundex::default().soundex(s))),
        "dms_nofold" => Box::new(|s| ok(DaitchMokotoffSoundex::new(false).soundex(s))),
        _ => bm_column(name),
    }
}

/// `GenAnalysisPhonetic.bmColumns`.
fn bm_column(name: &str) -> Column {
    let nt = |n: &str| match n {
        "ash" => NameType::Ashkenazi,
        "gen" => NameType::Generic,
        "sep" => NameType::Sephardic,
        other => panic!("no name type {other}"),
    };
    let (prefix, rest) = name.split_once('_').unwrap();
    let languages = |langs: &[&str]| -> Column {
        let e = PhoneticEngine::new(NameType::Generic, RuleType::Approx, true).unwrap();
        let names: Vec<String> = langs.iter().map(|s| s.to_string()).collect();
        Box::new(move |s| e.encode_with(s, &e.languages(&names)).map(Some))
    };
    match rest {
        "approx" | "exact" => {
            let rt = if rest == "approx" {
                RuleType::Approx
            } else {
                RuleType::Exact
            };
            let e = PhoneticEngine::new(nt(prefix), rt, true).unwrap();
            Box::new(move |s| e.encode(s).map(Some))
        }
        "approx_noconcat" => {
            let e = PhoneticEngine::new(nt(prefix), RuleType::Approx, false).unwrap();
            Box::new(move |s| e.encode(s).map(Some))
        }
        "guess" => {
            let lang = lucene_analysis_phonetic::bm::Lang::instance(nt(prefix));
            Box::new(move |s| ok(lang.guess_language(s).encode_utf16().collect()))
        }
        "english" => languages(&["english"]),
        "german_polish" => languages(&["german", "polish"]),
        "klingon_english" => languages(&["klingon", "english"]),
        "exact_max3" => {
            let e = PhoneticEngine::with_max_phonemes(NameType::Generic, RuleType::Exact, true, 3)
                .unwrap();
            Box::new(move |s| e.encode(s).map(Some))
        }
        other => panic!("no column {other}"),
    }
}

/// Every row of a `words.tsv`-shaped table through every column; returns the
/// number of values compared. Mismatches are collected per column so a
/// failure names the encoder and a few words.
fn check_table(file: &str) -> usize {
    let text = std::fs::read_to_string(data_dir("analysis_phonetic") + file).unwrap();
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().unwrap().split('\t').collect();
    assert_eq!(header[0], "#word");
    let columns: Vec<(&str, Column)> = header[1..].iter().map(|&h| (h, column(h))).collect();
    let mut compared = 0;
    let mut failures: Vec<String> = Vec::new();
    for line in lines {
        let fields: Vec<&str> = line.split('\t').collect();
        assert_eq!(fields.len(), header.len(), "{file}: {line:?}");
        let word = unesc(fields[0]);
        let units: Vec<u16> = word.encode_utf16().collect();
        for ((name, col), expected) in columns.iter().zip(&fields[1..]) {
            let actual = val(col(&units));
            compared += 1;
            if actual != normalise_expected(expected) && failures.len() < 40 {
                failures.push(format!("{name}({word:?}): java {expected} rust {actual}"));
            }
        }
    }
    assert!(failures.is_empty(), "{file}:\n{}", failures.join("\n"));
    compared
}

#[test]
fn every_encoder_matches_commons_codec() {
    let n = check_table("words.tsv");
    assert!(n > 400_000, "{n} values");
}

#[test]
fn beider_morse_matches_commons_codec() {
    let n = check_table("bm.tsv");
    assert!(n > 30_000, "{n} values");
}

#[test]
fn name_prefixes_iterate_in_javas_order() {
    let text = std::fs::read_to_string(data_dir("analysis_phonetic") + "prefixes.txt").unwrap();
    for line in text.lines() {
        let (nt, list) = line.split_once('\t').unwrap();
        let nt = NameType::value_of(nt).unwrap();
        let ours: Vec<String> = name_prefixes(nt)
            .iter()
            .map(|p| String::from_utf16(p).unwrap())
            .collect();
        assert_eq!(ours.join(","), list, "{nt:?}");
    }
}

/// The phonetic configurations: `(name, fields)`.
fn configs() -> Vec<(String, Vec<String>)> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/corpus/analysis-phonetic.conf"
    );
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| {
            let fields: Vec<String> = l.split('\t').map(str::to_string).collect();
            (fields[0].clone(), fields)
        })
        .collect()
}

/// `GenAnalysisPhonetic.stable`.
fn stable(class: &str, message: &str) -> String {
    let m = factory_config::stable(class, message);
    const LIST: &str = "must be full class name or one of ";
    match m.find(LIST) {
        Some(i) => m[..i + LIST.len()].to_string(),
        None => m,
    }
}

fn rows(fields: &[&str], lines: &[String]) -> Vec<String> {
    let analyzer = match factory_config::build(&factory_config::parse(fields)) {
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
fn every_configuration_matches_lucene() {
    lucene_analysis_phonetic::register_factories().unwrap();
    let lines = corpus("analysis-phonetic.txt");
    let mut built = 0;
    let all = configs();
    for (name, fields) in &all {
        let fields: Vec<&str> = fields.iter().map(String::as_str).collect();
        let actual = rows(&fields, &lines);
        let text = std::fs::read_to_string(format!("{}{name}.tsv", data_dir("analysis_phonetic")))
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let expected: Vec<String> = text.lines().map(normalise_expected).collect();
        for (i, (a, e)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(a, e, "{name}: row {i} differs");
        }
        assert_eq!(actual.len(), expected.len(), "{name}: row count");
        built += usize::from(actual[0].starts_with('S'));
    }
    assert_eq!(all.len(), 52);
    assert!(built >= 38, "{built} built");
}

#[test]
fn every_fixture_has_a_configuration() {
    let mut names: BTreeSet<String> = configs().into_iter().map(|(n, _)| n).collect();
    names.extend(["words", "bm"].map(String::from));
    let fixtures: BTreeSet<String> = std::fs::read_dir(data_dir("analysis_phonetic"))
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .filter(|f| f != "prefixes.txt")
        .map(|f| f.trim_end_matches(".tsv").to_string())
        .collect();
    assert_eq!(names, fixtures);
}
