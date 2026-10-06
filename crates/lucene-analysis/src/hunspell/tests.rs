//! Boundary and error handling the Java-fixture test
//! (`tests/hunspell_fixtures.rs`) does not reach: internal limits, API
//! accessors and inputs Lucene rejects before Hunspell sees them.

use std::sync::Arc;

use super::dictionary::{index_of_str, java_split, u};
use super::flags::{FlagEnumerator, FlagParsing};
use super::word_storage::WordStorageBuilder;
use super::*;
use crate::token_stream::TokenStream;
use crate::util::canned::Canned;

const AFF: &[u8] = b"SET UTF-8\nPFX R Y 1\nPFX R 0 re .\nSFX S Y 1\nSFX S 0 s .\n";
const DIC: &[u8] = b"2\nwalk/RS\nrun po:verb st:run\n";

fn dictionary() -> Dictionary {
    Dictionary::new(AFF, &[DIC], false).unwrap()
}

#[test]
fn flag_parsing_errors_are_javas() {
    assert!(matches!(
        FlagParsing::Simple.parse_flag(&[]),
        Err(HunspellError::IndexOutOfBounds(_))
    ));
    assert!(matches!(
        FlagParsing::Num.parse_flags(&u("99999999999")),
        Err(HunspellError::NumberFormat(_))
    ));
    assert!(matches!(
        FlagParsing::Num.parse_flags(&u("65510")),
        Err(HunspellError::IllegalArgument(_))
    ));
    // Java's `getBytes(ISO_8859_1)` writes `?` for a unit it cannot map.
    assert_eq!(
        FlagParsing::DefaultAsUtf8.parse_flags(&[0x0100]).unwrap(),
        u("?")
    );
    let mut enumerator = FlagEnumerator::new();
    let mut too_many = vec![1u16; usize::from(u16::MAX) + 1];
    assert!(matches!(
        enumerator.add(&mut too_many),
        Err(HunspellError::IllegalArgument(_))
    ));
}

#[test]
fn word_storage_rejects_what_lucene_rejects() {
    let mut enumerator = FlagEnumerator::new();
    let mut b = WordStorageBuilder::new(2, 1.0, false, &mut enumerator, vec![]);
    b.add(&u("b"), vec![], 0).unwrap();
    assert!(matches!(
        b.add(&u("a"), vec![], 0),
        Err(HunspellError::IllegalArgument(_))
    ));

    let mut enumerator = FlagEnumerator::new();
    let mut b = WordStorageBuilder::new(1, 1.0, false, &mut enumerator, vec![]);
    b.add(&u("a"), vec![], 0).unwrap();
    b.add(&u("b"), vec![], 0).unwrap();
    assert!(matches!(
        b.add(&u("c"), vec![], 0),
        Err(HunspellError::IllegalState(_))
    ));

    // One slot: the 21st word overflows its chain.
    let mut enumerator = FlagEnumerator::new();
    let mut b = WordStorageBuilder::new(30, 1.0 / 30.0, false, &mut enumerator, vec![]);
    let mut result = Ok(());
    for i in 0..22 {
        result = result.and_then(|()| b.add(&u(&format!("w{i:02}")), vec![], 0));
    }
    let result = result.and_then(|()| b.finish().map(|_| ()));
    assert!(matches!(result, Err(HunspellError::IllegalState(_))));
}

#[test]
fn dictionary_accessors_and_unsupported_charsets() {
    let d = dictionary();
    assert!(!d.ignore_case());
    assert_eq!(d.lookup_entries(""), None);
    let run = &d.lookup_entries("run").unwrap()[0];
    assert_eq!(run.morphological_values("st:"), vec!["run".to_string()]);
    assert!(run.morphological_values("is:").is_empty());
    let walk = &d.lookup_entries("walk").unwrap()[0];
    assert!(walk.morphological_values("st:").is_empty());

    assert!(matches!(
        Dictionary::new(b"SET KOI8-R\n", &[DIC], false),
        Err(HunspellError::Unsupported(_))
    ));
    let e = Dictionary::new(
        b"FLAG num\nSFX 1x Y 1\nSFX 1x 0 s/99999999999 .\n",
        &[DIC],
        false,
    )
    .unwrap_err();
    assert!(!e.to_string().is_empty());
}

#[test]
fn affixed_words_name_their_flags() {
    let d = dictionary();
    let forms = WordFormGenerator::new(&d).get_all_word_forms("walk");
    let rewalks = forms.iter().find(|w| w.word == "rewalks").unwrap();
    assert_eq!(rewalks.prefixes[0].flag(), "R");
    assert_eq!(rewalks.suffixes[0].flag(), "S");
}

#[test]
fn string_helpers_follow_java() {
    assert_eq!(index_of_str(&u("ab"), &[], 1), Some(1));
    assert_eq!(index_of_str(&u("ab"), &[], 3), None);
    assert!(java_split(&u("  "), true).is_empty());
    assert_eq!(java_split(&u("a  "), true), vec![&u("a")[..]]);
}

#[test]
fn n_gram_fragment_checker_bounds() {
    assert!(matches!(
        NGramFragmentChecker::from_words(5, &["abc"]),
        Err(HunspellError::IllegalArgument(_))
    ));
    // One word sizes the set at 64 bits; every bigram over `a..=h` fills
    // 48 of them, past Lucene's two-thirds limit.
    let letters = b'a'..=b'h';
    let word: String = letters
        .clone()
        .flat_map(|a| {
            letters
                .clone()
                .flat_map(move |b| [char::from(a), char::from(b)])
        })
        .collect();
    assert!(matches!(
        NGramFragmentChecker::from_words(2, &[word.as_str()]),
        Err(HunspellError::IllegalArgument(_))
    ));
    let checker = NGramFragmentChecker::from_words(2, &["walk"]).unwrap();
    assert!(!checker.has_impossible_fragment_around(&u("w"), 0, 1));
    assert!(!EverythingPossible.has_impossible_fragment_around(&u("zz"), 0, 2));
}

#[test]
fn stem_filter_defaults_and_keywords() {
    let d = Arc::new(dictionary());
    let mut canned = Canned::parse("walks:0:5:1:1 rewalks:6:13:1:1");
    canned.set_keywords(&[true, false]);
    let mut f = HunspellStemFilter::with_defaults(canned, d);
    f.reset().unwrap();
    let mut out = Vec::new();
    while f.increment_token().unwrap() {
        out.push(f.attributes().term().to_string());
    }
    assert_eq!(out, ["walks", "walk"]);
}

#[test]
fn stemmer_unique_stems() {
    let d = dictionary();
    assert_eq!(Stemmer::new(&d).unique_stems("walks"), ["walk"]);
    // Past the stack buffer for a stripped word.
    let long = format!("{}s", "walk".repeat(17));
    assert!(Stemmer::new(&d).stem(&long).is_empty());
}

#[test]
fn alias_counts_do_not_size_allocations() {
    // Java allocates `new String[count]` up front; a hostile header must not
    // make the port allocate 2e9 slots (or abort trying).
    let aff = b"AF 2000000000\nAF AB\nAM 2000000000\nAM po:noun\nSFX A Y 1\nSFX A 0 s .\n";
    let d = Dictionary::new(aff, &[b"1\nwalk/1\t1\n"], false).unwrap();
    assert!(Hunspell::new(&d).spell("walks"));
    assert_eq!(
        d.lookup_entries("walk").unwrap()[0].to_string(),
        "walk/AB po:noun"
    );
    // An alias number inside the announced count but past the lines given:
    // Java's slot is `null` (a `NullPointerException` later); the port
    // reports the bad alias number.
    assert!(matches!(
        Dictionary::new(aff, &[b"1\nwalk/2\n"], false),
        Err(HunspellError::IllegalArgument(_))
    ));
    // More lines than announced: Java's `ArrayIndexOutOfBoundsException`.
    for aff in [&b"AF 1\nAF A\nAF B\n"[..], b"AM 1\nAM a:b\nAM c:d\n"] {
        assert!(matches!(
            Dictionary::new(aff, &[b"0\n"], false),
            Err(HunspellError::IndexOutOfBounds(_))
        ));
    }
    // A negative count: Java's `NegativeArraySizeException`.
    for aff in [&b"AF -1\n"[..], b"AM -1\n"] {
        assert!(matches!(
            Dictionary::new(aff, &[b"0\n"], false),
            Err(HunspellError::NegativeArraySize(_))
        ));
    }
}
