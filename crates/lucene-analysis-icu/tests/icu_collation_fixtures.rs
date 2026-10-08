//! ICU collation sort keys against ICU4J 77.1 (`fixtures/src/GenAnalysisIcuCollation.java`):
//! every collation bundle and type, fallback IDs, attribute keywords and
//! setter combinations over 1,687 texts, block digests of the keys; full
//! keys for seven collators; Lucene's `ICUCollationKeyAnalyzer` and
//! `ICUCollationDocValuesField` bytes.

use std::path::PathBuf;

use lucene_analysis::{Analyzer, TokenStream};
use lucene_analysis_icu::{
    Collator, ICUCollationDocValuesField, ICUCollationKeyAnalyzer, IcuError, IcuErrorKind,
};

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/analysis_icu_collation")
}

fn read(name: &str) -> String {
    std::fs::read_to_string(dir().join(name)).unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn strings() -> Vec<Vec<u16>> {
    read("coll_strings.txt")
        .lines()
        .map(|l| {
            l.split(' ')
                .filter(|u| !u.is_empty())
                .map(|u| u16::from_str_radix(u, 16).unwrap())
                .collect()
        })
        .collect()
}

fn exception(e: &IcuError) -> &'static str {
    match e.kind() {
        IcuErrorKind::IllegalArgument => "IllegalArgumentException",
        IcuErrorKind::IllegalIcuArgument => "IllegalIcuArgumentException",
        IcuErrorKind::MissingResource => "MissingResourceException",
        IcuErrorKind::Io => "ICUUncheckedIOException",
        IcuErrorKind::UnsupportedOperation => "UnsupportedOperationException",
    }
}

/// `GenAnalysisIcuCollation.collator(spec)`.
fn collator(spec: &str) -> Result<Collator, IcuError> {
    let mut parts = spec.split('|');
    let mut c = Collator::get_instance(parts.next().unwrap())?;
    for p in parts {
        let (k, v) = p.split_once('=').unwrap();
        match k {
            "strength" => c.set_strength(v.parse().unwrap())?,
            "decomposition" => c.set_decomposition(v.parse().unwrap())?,
            "french" => c.set_french_collation(v == "1"),
            "caseLevel" => c.set_case_level(v == "1"),
            "upper" => c.set_upper_case_first(v == "1"),
            "lower" => c.set_lower_case_first(v == "1"),
            "shifted" => c.set_alternate_handling_shifted(v == "1"),
            "numeric" => c.set_numeric_collation(v == "1"),
            "maxVariable" => c.set_max_variable(v.parse().unwrap())?,
            "reorder" => {
                let codes: Vec<i32> = v
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(|s| s.parse().unwrap())
                    .collect();
                c.set_reorder_codes(&codes)?
            }
            _ => panic!("{k}"),
        }
    }
    Ok(c)
}

fn fnv(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h = (h ^ u64::from(b)).wrapping_mul(0x100000001b3);
    }
    h
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[test]
fn collators_and_keys() {
    let strings = strings();
    let mut failures = Vec::new();
    let mut rows = 0;
    for line in read("coll_configs.tsv").lines() {
        rows += 1;
        let cols: Vec<&str> = line.split('\t').collect();
        let spec = cols[0];
        let c = match collator(spec) {
            Ok(c) => c,
            Err(e) => {
                if cols[1] != "X" || cols[2] != exception(&e) {
                    failures.push(format!(
                        "{spec}: Rust {} ({e}), Java {}",
                        exception(&e),
                        cols[1..].join(" ")
                    ));
                }
                continue;
            }
        };
        if cols[1] == "X" {
            failures.push(format!(
                "{spec}: Rust built a collator, Java threw {}",
                cols[2]
            ));
            continue;
        }
        if c.valid_locale() != cols[1] || c.actual_locale() != cols[2] {
            failures.push(format!(
                "{spec}: locales Rust {:?}/{:?}, Java {:?}/{:?}",
                c.valid_locale(),
                c.actual_locale(),
                cols[1],
                cols[2]
            ));
        }
        for (b, block) in strings.chunks(100).enumerate() {
            let mut h = 0xcbf29ce484222325u64;
            for s in block {
                h = fnv(h, &c.raw_collation_key_utf16(s));
            }
            if format!("{h:x}") != cols[3 + b] {
                failures.push(format!("{spec}: block {b} differs"));
                break;
            }
        }
        if failures.len() > 40 {
            break;
        }
    }
    assert!(rows > 1800, "{rows} rows");
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn full_keys() {
    let strings = strings();
    for (name, spec) in [
        ("root", ""),
        ("de_phonebook", "de@collation=phonebook"),
        ("ja_identical", "ja|strength=15"),
        ("en_shifted_quaternary", "en|shifted=1|strength=3"),
        ("fr_ca", "fr_CA"),
        ("th_numeric_caselevel", "th|numeric=1|caseLevel=1|upper=1"),
        ("da_fcd", "da|decomposition=17"),
    ] {
        let c = collator(spec).unwrap();
        let expected = read(&format!("coll_keys_{name}.tsv"));
        for (i, (s, want)) in strings.iter().zip(expected.lines()).enumerate() {
            let got = hex(&c.raw_collation_key_utf16(s));
            assert_eq!(got, want, "{name}: string {i} {:04x?}", s);
        }
    }
}

#[test]
fn lucene_classes() {
    let corpus = std::fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/corpus/analysis-icu.txt"),
    )
    .unwrap();
    let lines: Vec<&str> = corpus.lines().collect();
    let expected = read("coll_lucene.tsv");
    let mut rows = expected.lines();
    for loc in ["", "de@collation=phonebook", "ja"] {
        let c = Collator::get_instance(loc).unwrap();
        let a = Analyzer::new(ICUCollationKeyAnalyzer::new(c.clone()));
        let mut field = ICUCollationDocValuesField::new("f", c);
        for line in &lines {
            let row = rows.next().unwrap();
            let cols: Vec<&str> = row.split('\t').collect();
            assert_eq!(cols[0], loc);
            let mut ts = a.token_stream("f", line).unwrap();
            ts.reset().unwrap();
            let mut terms = Vec::new();
            while ts.increment_token().unwrap() {
                terms.push(hex(ts.attributes().term_bytes()));
            }
            ts.end().unwrap();
            drop(ts);
            assert_eq!(terms.join(","), cols[1], "{loc}: {line}");
            field.set_string_value(line);
            assert_eq!(hex(field.binary_value()), cols[2], "{loc}: {line}");
        }
    }
    assert!(rows.next().is_none());
}

#[test]
fn rules_are_refused() {
    let e = Collator::from_rules("&a<b").unwrap_err();
    assert_eq!(e.kind(), IcuErrorKind::UnsupportedOperation);
}
