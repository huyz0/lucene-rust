//! M12 T12.3: Morfologik against Lucene 10.5.0 and Morfologik 2.1.9.
//!
//! `fixtures/src/GenAnalysisMorfologik.java` writes `analysis_morfologik/`:
//! `lookups_<lang>.tsv` (words through the real Polish and Ukrainian
//! dictionaries), `dicts/` (dictionaries built with Morfologik's own
//! builders -- every encoder, both formats, two charsets, conversions,
//! tagless entries -- and words through each), and analyzer, filter and
//! factory rows with each token's tags. Every lemma, tag and row must be
//! equal.
#![allow(clippy::arithmetic_side_effects)] // test code: no value read off disk

#[path = "../../lucene-analysis/tests/support/factory_config.rs"]
mod factory_config;
#[path = "../../lucene-analysis/tests/support/mod.rs"]
mod support;

use std::sync::Arc;

use lucene_analysis::factory::CustomAnalyzer;
use lucene_analysis::miscellaneous::SetKeywordMarkerFilter;
use lucene_analysis::util::WhitespaceTokenizer;
use lucene_analysis::{AnalysisError, Analyzer, CharArraySet, TokenStream};
use lucene_analysis_morfologik::analyzer::{polish_dictionary, ukrainian_dictionary};
use lucene_analysis_morfologik::{
    Dictionary, DictionaryLookup, MorfologikAnalyzer, MorfologikFilter,
    MorphosyntacticTagsAttribute, UkrainianMorfologikAnalyzer,
};
use support::{chain, comps, corpus, data_dir, esc, normalise_expected};

fn opt(v: &Option<Vec<u16>>) -> String {
    match v {
        Some(v) => esc(&String::from_utf16_lossy(v)),
        None => "~".into(),
    }
}

/// `AnalysisRows.esc`'s inverse into UTF-16 units (a word may hold a lone
/// surrogate, escaped `\uD800`).
fn unesc_units(s: &str) -> Vec<u16> {
    let mut out = Vec::new();
    let mut it = s.chars();
    while let Some(c) = it.next() {
        if c != '\\' {
            let mut b = [0u16; 2];
            out.extend_from_slice(c.encode_utf16(&mut b));
            continue;
        }
        match it.next() {
            Some('t') => out.push(9),
            Some('n') => out.push(10),
            Some('r') => out.push(13),
            Some('u') => {
                let h: String = it.by_ref().take(4).collect();
                out.push(u16::from_str_radix(&h, 16).unwrap());
            }
            Some(o) => out.push(o as u16),
            None => out.push(u16::from(b'\\')),
        }
    }
    out
}

/// `GenAnalysisMorfologik.lookupRow`.
fn lookup_row(lookup: &mut DictionaryLookup, units: &[u16], escaped: &str) -> String {
    let mut row = escaped.to_string();
    match lookup.lookup(units) {
        Ok(forms) => {
            for f in forms {
                row.push_str(&format!("\t{}\t{}", opt(&f.stem), opt(&f.tag)));
            }
        }
        Err(e) => {
            let class = e.message().split(':').next().unwrap_or("").to_string();
            row.push_str(&format!("\t!{class}"));
        }
    }
    row
}

fn check_lookups(dict: Arc<Dictionary>, file: &str) -> usize {
    let text = std::fs::read_to_string(data_dir("analysis_morfologik") + file).unwrap();
    let mut lookup = DictionaryLookup::new(dict);
    let mut failures = Vec::new();
    let mut n = 0;
    for line in text.lines() {
        let escaped = line.split('\t').next().unwrap();
        let actual = lookup_row(&mut lookup, &unesc_units(escaped), escaped);
        n += 1;
        if actual != line && failures.len() < 20 {
            failures.push(format!("java {line:?}\nrust {actual:?}"));
        }
    }
    assert!(failures.is_empty(), "{file}:\n{}", failures.join("\n"));
    n
}

#[test]
fn polish_lookups_match_morfologik() {
    assert!(check_lookups(polish_dictionary(), "lookups_polish.tsv") > 40_000);
}

#[test]
fn ukrainian_lookups_match_morfologik() {
    assert!(check_lookups(ukrainian_dictionary(), "lookups_ukrainian.tsv") > 1_000);
}

fn dicts_dir() -> String {
    data_dir("analysis_morfologik") + "dicts/"
}

fn own(name: &str) -> Arc<Dictionary> {
    let dir = dicts_dir();
    let fsa = std::fs::read(format!("{dir}{name}.dict")).unwrap();
    let info = std::fs::read_to_string(format!("{dir}{name}.info")).unwrap();
    Arc::new(Dictionary::read(&fsa, &info).unwrap_or_else(|e| panic!("{name}: {e}")))
}

#[test]
fn own_dictionaries_match_morfologik() {
    let mut checked = 0;
    for e in std::fs::read_dir(dicts_dir()).unwrap() {
        let name = e.unwrap().file_name().into_string().unwrap();
        let Some(stem) = name.strip_suffix(".dict") else {
            continue;
        };
        check_lookups(own(stem), &format!("dicts/{stem}.tsv"));
        // Every truncation of the automaton reads or fails, never panics;
        // a lookup in what reads may fail but not panic either.
        let bytes = std::fs::read(format!("{}{name}", dicts_dir())).unwrap();
        let info = std::fs::read_to_string(format!("{}{stem}.info", dicts_dir())).unwrap();
        // And every single-byte flip.
        for i in 0..bytes.len() {
            let mut flipped = bytes.clone();
            flipped[i] ^= 0xFF;
            if let Ok(d) = Dictionary::read(&flipped, &info) {
                let mut l = DictionaryLookup::new(Arc::new(d));
                for w in ["kotami", "dobra", "zamek", "a", "ab"] {
                    let _ = l.lookup(&w.encode_utf16().collect::<Vec<_>>());
                }
            }
        }
        for cut in 0..bytes.len() {
            if let Ok(d) = Dictionary::read(&bytes[..cut], &info) {
                let mut l = DictionaryLookup::new(Arc::new(d));
                for w in ["kotami", "dobra", "zamek", "a"] {
                    let _ = l.lookup(&w.encode_utf16().collect::<Vec<_>>());
                }
            }
        }
        checked += 1;
    }
    assert_eq!(checked, 8);
}

/// `GenAnalysisMorfologik.rowsWithTags` for one analyzer.
fn rows_with_tags(a: &Analyzer, lines: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    for (ln, line) in lines.iter().enumerate() {
        let run = |out: &mut Vec<String>| -> Result<(), AnalysisError> {
            let mut ts = a.token_stream("f", line)?;
            ts.reset()?;
            while ts.increment_token()? {
                let at = ts.attributes();
                let tags = match at
                    .custom::<MorphosyntacticTagsAttribute>()
                    .and_then(|t| t.tags.as_ref())
                {
                    None => "-".to_string(),
                    Some(t) => format!(
                        "[{}]",
                        t.iter().map(|s| esc(s)).collect::<Vec<_>>().join("|")
                    ),
                };
                out.push(format!(
                    "T\t{ln}\t{}\t{}\t{}\t{}\t{}\t{tags}",
                    esc(at.term()),
                    at.start_offset(),
                    at.end_offset(),
                    at.position_increment(),
                    u8::from(at.is_keyword())
                ));
            }
            ts.end()?;
            let at = ts.attributes();
            out.push(format!(
                "E\t{ln}\t{}\t{}\t{}",
                at.start_offset(),
                at.end_offset(),
                at.position_increment()
            ));
            ts.close()
        };
        let mark = out.len();
        if let Err(e) = run(&mut out) {
            out.truncate(mark);
            out.push(format!("X\t{ln}\t{}", support::exception_name(&e)));
        }
    }
    out
}

fn expected(name: &str) -> Vec<String> {
    std::fs::read_to_string(format!("{}{name}.tsv", data_dir("analysis_morfologik")))
        .unwrap_or_else(|e| panic!("{name}: {e}"))
        .lines()
        .map(normalise_expected)
        .collect()
}

#[test]
fn chains_match_lucene() {
    let pl = corpus("analysis-stempel.txt");
    let uk = corpus("analysis-ukrainian.txt");
    let own_lines = corpus("analysis-morfologik.txt");
    let mut ex = CharArraySet::with_capacity(3, false);
    for w in ["київ", "школі", "дітей"] {
        ex.add(w);
    }
    let suffix = own("suffix_cfsa2");
    let cases: Vec<(&str, Analyzer, &[String])> = vec![
        (
            "morfologik_polish",
            Analyzer::new(MorfologikAnalyzer::default()),
            &pl,
        ),
        (
            "ukrainian_analyzer",
            Analyzer::new(UkrainianMorfologikAnalyzer::default()),
            &uk,
        ),
        (
            "ukrainian_exclusions",
            Analyzer::new(UkrainianMorfologikAnalyzer::with_exclusions(
                &CharArraySet::empty(),
                &ex,
            )),
            &uk,
        ),
        (
            "filter_own_dict",
            chain(move || {
                let mut kw = CharArraySet::with_capacity(1, false);
                kw.add("koty");
                comps(MorfologikFilter::new(
                    SetKeywordMarkerFilter::new(WhitespaceTokenizer::new(), Arc::new(kw)),
                    Arc::clone(&suffix),
                ))
            }),
            &own_lines,
        ),
    ];
    for (name, a, lines) in cases {
        let actual = rows_with_tags(&a, lines);
        let expected = expected(name);
        for (i, (x, y)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(x, y, "{name}: row {i}");
        }
        assert_eq!(actual.len(), expected.len(), "{name}");
    }
}

#[test]
fn factories_match_lucene() {
    lucene_analysis_morfologik::register_factories().unwrap();
    let lines = corpus("analysis-morfologik.txt");
    let configs: [&[&str]; 5] = [
        &["factory_default", "tok:standard", "tf:morfologik"],
        &[
            "factory_own",
            "tok:whitespace",
            "tf:morfologik",
            "dictionary=suffix_fsa5.dict",
        ],
        &[
            "factory_missing",
            "tok:whitespace",
            "tf:morfologik",
            "dictionary=missing.dict",
        ],
        &[
            "factory_resource_param",
            "tok:whitespace",
            "tf:morfologik",
            "dictionary-resource=x",
        ],
        &["factory_param", "tok:whitespace", "tf:morfologik", "x=1"],
    ];
    for fields in configs {
        let name = fields[0];
        let built = (|| {
            let mut b = CustomAnalyzer::builder_with_dir(dicts_dir())?;
            for s in factory_config::parse(fields) {
                let p: Vec<&str> = s.params.iter().map(String::as_str).collect();
                b = match s.kind.as_str() {
                    "tok" => b.with_tokenizer(&s.name, &p)?,
                    _ => b.add_token_filter(&s.name, &p)?,
                };
            }
            b.build()
        })();
        let actual = match built {
            Err(e) => vec![format!("B\t{}", e.java_class())],
            Ok(a) => rows_with_tags(&a, &lines),
        };
        assert_eq!(actual, expected(name), "{name}");
    }
}
