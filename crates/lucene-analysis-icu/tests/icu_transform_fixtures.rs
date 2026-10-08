//! ICU transliterators against ICU4J 77.1 and Lucene's `ICUTransformFilter`
//! (`fixtures/src/GenAnalysisIcuTransform.java`): every ID ICU4J lists plus
//! compound, filtered, inverse and malformed IDs over the texts (block
//! digests and `getID()`), full outputs and source sets of the
//! transliterators Lucene's tests use, rules built with `createFromRules`,
//! and the filter over the corpus.
#![allow(clippy::arithmetic_side_effects)] // test code: no value read off disk

#[path = "../../lucene-analysis/tests/support/mod.rs"]
mod support;

use lucene_analysis::util::WhitespaceTokenizer;
use lucene_analysis::KeywordTokenizer;
use lucene_analysis_icu::icu4j::translit::{FORWARD, REVERSE};
use lucene_analysis_icu::{ICUTransformFilter, IcuError, IcuErrorKind, Transliterator};
use support::{chain, comps, unesc};

const DIR: &str = "analysis_icu_transform";

fn read(name: &str) -> String {
    std::fs::read_to_string(format!("{}/{name}", support::data_dir(DIR)))
        .unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn units(hex: &str) -> Vec<u16> {
    hex.split(' ')
        .filter(|u| !u.is_empty())
        .map(|u| u16::from_str_radix(u, 16).unwrap())
        .collect()
}

fn strings() -> Vec<Vec<u16>> {
    read("tr_strings.txt").lines().map(units).collect()
}

/// IDs ICU4J builds that this port refuses by design (`UnsupportedOperation`):
/// the escape and name transliterators, and the specs that name a locale.
pub fn refused_by_design(spec: &str) -> bool {
    let id = spec.trim_end_matches("|R");
    id.contains("Hex")
        || id.contains("Name")
        || id.starts_with("Any-Any")
        || id == "Any-"
        || id.split(['-', '/', ';']).any(is_locale)
}

fn is_locale(part: &str) -> bool {
    let language = part.trim().split('_').next().unwrap();
    (language.len() == 2 || language.len() == 3) && language.bytes().all(|b| b.is_ascii_lowercase())
}

fn instance(spec: &str) -> Result<Transliterator, IcuError> {
    match spec.strip_suffix("|R") {
        Some(id) => Transliterator::get_instance(id, REVERSE),
        None => Transliterator::get_instance(spec, FORWARD),
    }
}

fn fnv(mut h: u64, s: &[u16]) -> u64 {
    for &c in s {
        h = (h ^ u64::from(c & 0xff)).wrapping_mul(0x100000001b3);
        h = (h ^ u64::from(c >> 8)).wrapping_mul(0x100000001b3);
    }
    h = (h ^ 0xff).wrapping_mul(0x100000001b3);
    (h ^ 0xff).wrapping_mul(0x100000001b3)
}

fn run(t: &Transliterator, s: &[u16]) -> Vec<u16> {
    let mut v = s.to_vec();
    t.transliterate_units(&mut v).unwrap();
    v
}

#[test]
fn ids_match_icu4j() {
    let texts = strings();
    let mut bad = Vec::new();
    let mut refused = 0;
    let mut checked = 0;
    for row in read("tr_ids.tsv").lines() {
        let f: Vec<&str> = row.split('\t').collect();
        let spec = f[0];
        let got = instance(spec);
        if f[1] == "X" {
            if got.is_ok() {
                bad.push(format!("{spec}: Java refused ({}), Rust built it", f[2]));
            }
            continue;
        }
        let t = match got {
            Ok(t) => t,
            Err(e) if e.kind() == IcuErrorKind::UnsupportedOperation && refused_by_design(spec) => {
                refused += 1;
                continue;
            }
            Err(e) => {
                bad.push(format!("{spec}: {e}"));
                continue;
            }
        };
        if t.id() != f[1] {
            bad.push(format!("{spec}: id {} != {}", t.id(), f[1]));
        }
        let mut h = 0xcbf29ce484222325u64;
        let mut digests = Vec::new();
        for (i, s) in texts.iter().enumerate() {
            let mut v = s.clone();
            match t.transliterate_units(&mut v) {
                Ok(()) => h = fnv(h, &v),
                Err(e) => {
                    bad.push(format!("{spec}: line {i}: {e}"));
                    break;
                }
            }
            if (i + 1) % 50 == 0 || i + 1 == texts.len() {
                digests.push(format!("{h:x}"));
                h = 0xcbf29ce484222325;
            }
        }
        let first_bad = digests.iter().zip(&f[2..]).position(|(a, b)| a != b);
        if let Some(block) = first_bad {
            bad.push(format!("{spec}: block {block} differs"));
        }
        checked += 1;
    }
    assert!(
        bad.is_empty(),
        "{} mismatches:\n{}",
        bad.len(),
        bad.join("\n")
    );
    assert!(checked > 750, "{checked}");
    assert!(refused < 130, "{refused}");
}

#[test]
fn full_outputs_and_source_sets_match_icu4j() {
    let texts = strings();
    let mut current: Option<(String, Transliterator)> = None;
    let mut bad = Vec::new();
    for row in read("tr_full.tsv").lines() {
        let f: Vec<&str> = row.splitn(3, '\t').collect();
        if current.as_ref().map(|(s, _)| s.as_str()) != Some(f[0]) {
            current = Some((f[0].to_string(), instance(f[0]).unwrap()));
        }
        let t = &current.as_ref().unwrap().1;
        let i: usize = f[1].parse().unwrap();
        let got = run(t, &texts[i]);
        if got != units(f[2]) && bad.len() < 20 {
            bad.push(format!(
                "{} line {i}: {:?} -> {:?}, want {:?}",
                f[0],
                String::from_utf16_lossy(&texts[i]),
                String::from_utf16_lossy(&got),
                String::from_utf16_lossy(&units(f[2]))
            ));
        }
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
    for row in read("tr_sources.tsv").lines() {
        let (spec, ranges) = row.split_once('\t').unwrap();
        let set = instance(spec).unwrap().source_set().unwrap();
        let mut want = Vec::new();
        for r in ranges.split(',').filter(|r| !r.starts_with('{')) {
            let (a, b) = r.split_once('-').unwrap();
            want.push((
                u32::from_str_radix(a, 16).unwrap(),
                u32::from_str_radix(b, 16).unwrap(),
            ));
        }
        assert_eq!(set.ranges(), &want[..], "{spec}");
    }
}

#[test]
fn rules_match_icu4j() {
    const INPUTS: &[&str] = &[
        "",
        "a",
        "ab",
        "abc",
        "aab",
        "ba",
        "xay",
        "aaaa",
        "bbb",
        "c",
        "cab",
        "a1",
        "b2c3",
        "Hello World",
        "ABC abc",
        "zzz",
        "é",
        "e\u{301}",
        "αβγ",
        "Москва",
        "x a y",
        "aXb",
        "123",
        "a b c",
        "aa bb cc",
    ];
    let mut t: Option<Result<Transliterator, IcuError>> = None;
    let mut rules = String::new();
    let mut n = 0;
    let mut bad: Vec<String> = Vec::new();
    for row in read("tr_rules.tsv").lines() {
        let f: Vec<&str> = row.split('\t').collect();
        let dir = if f[1].ends_with('R') {
            REVERSE
        } else {
            FORWARD
        };
        let key = f[1];
        match f[0] {
            "R" => {
                rules = unesc(f[2]);
                t = Some(Transliterator::create_from_rules("Test", &rules, dir));
                n += 1;
            }
            "I" => match t.as_ref().unwrap() {
                Ok(got) if got.id() == f[2] => {}
                Ok(got) => bad.push(format!("{key} {rules:?}: id {} != {}", got.id(), f[2])),
                Err(e) => bad.push(format!("{key} {rules:?}: refused: {e}")),
            },
            "O" => {
                let Some(Ok(tr)) = t.as_ref() else {
                    continue;
                };
                let i: usize = f[2].parse().unwrap();
                let input: Vec<u16> = INPUTS[i].encode_utf16().collect();
                let want = f.get(3).copied().unwrap_or("");
                let mut got = input.clone();
                let result = tr.transliterate_units(&mut got);
                match result {
                    Err(_) if want.starts_with('!') => {}
                    Ok(()) if want.starts_with('!') => {
                        bad.push(format!("{key} {rules:?} on {:?}: Java threw", INPUTS[i]))
                    }
                    Err(e) => bad.push(format!("{key} {rules:?} on {:?}: {e}", INPUTS[i])),
                    Ok(()) if got != units(want) => bad.push(format!(
                        "{key} {rules:?} on {:?}: {:?}, want {:?}",
                        INPUTS[i],
                        String::from_utf16_lossy(&got),
                        String::from_utf16_lossy(&units(want))
                    )),
                    Ok(()) => {}
                }
            }
            "X" => {
                if let Some(Ok(_)) = t.as_ref() {
                    bad.push(format!("{key} {rules:?}: Java threw {}", f[2]));
                }
            }
            _ => panic!("{row}"),
        }
    }
    bad.dedup();
    assert!(
        bad.is_empty(),
        "{} mismatches:\n{}",
        bad.len(),
        bad.join("\n")
    );
    assert!(n > 200);
}

#[test]
fn filter_matches_lucene() {
    let names: std::collections::BTreeSet<String> = support::fixture_names(DIR)
        .into_iter()
        .filter(|n| n.starts_with("tr_lucene_"))
        .collect();
    let build = |name: &str| {
        let (id, dir, ws) = match name {
            "tr_lucene_ws_any_latin" => ("Any-Latin", FORWARD, true),
            "tr_lucene_ws_trad_simp" => ("Traditional-Simplified", FORWARD, true),
            "tr_lucene_ws_kata_hira" => ("Katakana-Hiragana", FORWARD, true),
            "tr_lucene_ws_full_half" => ("Fullwidth-Halfwidth", FORWARD, true),
            "tr_lucene_ws_strip_marks" => ("NFD; [:Nonspacing Mark:] Remove", FORWARD, true),
            "tr_lucene_ws_han_latin" => ("Han-Latin", FORWARD, true),
            "tr_lucene_ws_casefold" => ("CaseFold", FORWARD, true),
            "tr_lucene_ws_cyrl_latn" => ("Cyrillic-Latin", FORWARD, true),
            "tr_lucene_ws_latn_cyrl_rev" => ("Cyrillic-Latin", REVERSE, true),
            "tr_lucene_ws_null" => ("Null", FORWARD, true),
            "tr_lucene_kw_any_latin_ascii" => ("Any-Latin; Latin-ASCII; Lower", FORWARD, false),
            "tr_lucene_ws_rules" => ("", FORWARD, true),
            _ => return None,
        };
        let t = if id.is_empty() {
            Transliterator::create_from_rules(
                "test",
                "a > b; [:Lu:] > X; (é) > &Any-Upper($1);",
                FORWARD,
            )
        } else {
            Transliterator::get_instance(id, dir)
        }
        .unwrap();
        Some(if ws {
            chain(move || {
                comps(ICUTransformFilter::new(
                    WhitespaceTokenizer::new(),
                    t.clone(),
                ))
            })
        } else {
            chain(move || comps(ICUTransformFilter::new(KeywordTokenizer::new(), t.clone())))
        })
    };
    let lines = support::corpus("analysis-icu.txt");
    assert_eq!(
        support::check_chains_over(DIR, &lines, &names, build),
        names.len()
    );
    assert_eq!(names.len(), 12);
}
