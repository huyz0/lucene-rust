//! M13 T13.3e against Lucene 10.5.0 (`fixtures/src/GenSpellWordBreak.java`):
//! `WordBreakSpellChecker`'s breaks and combinations over a three-segment
//! index, ten configurations x every query x every mode, sort method and
//! size.

use std::path::PathBuf;

use lucene_search::directory_reader::DirectoryReader;
use lucene_search::query_visitor::Term;
use lucene_search::spell::suggest_word::{SuggestMode, SuggestWord};
use lucene_search::spell::word_break::{
    separator_term, BreakSuggestionSortMethod, WordBreakSpellChecker,
};
use lucene_store::FsDirectory;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/spell_word_break")
}

fn reader() -> DirectoryReader {
    DirectoryReader::open(&FsDirectory::open(dir().join("index"))).unwrap()
}

fn checker(name: &str) -> WordBreakSpellChecker {
    let mut s = WordBreakSpellChecker::new();
    match name {
        "default" | "nulls" => {}
        "max_changes_2" => s.set_max_changes(2),
        "max_changes_3" => s.set_max_changes(3),
        "min_break_2" => {
            s.set_max_changes(3);
            s.set_min_break_word_length(2);
        }
        "min_break_0" => s.set_min_break_word_length(0),
        "min_freq_2" => {
            s.set_max_changes(2);
            s.set_min_suggestion_frequency(2);
        }
        "max_evaluations_3" => {
            s.set_max_changes(3);
            s.set_max_evaluations(3);
        }
        "max_evaluations_10" => {
            s.set_max_changes(3);
            s.set_max_evaluations(10);
        }
        "max_combine_3" => {
            s.set_max_changes(3);
            s.set_max_combine_word_length(3);
        }
        "max_combine_5" => {
            s.set_max_changes(2);
            s.set_max_combine_word_length(5);
        }
        other => panic!("configuration {other}"),
    }
    s
}

fn mode(name: &str) -> Option<SuggestMode> {
    match name {
        "SUGGEST_WHEN_NOT_IN_INDEX" => Some(SuggestMode::SuggestWhenNotInIndex),
        "SUGGEST_MORE_POPULAR" => Some(SuggestMode::SuggestMorePopular),
        "SUGGEST_ALWAYS" => Some(SuggestMode::SuggestAlways),
        "null" => None,
        other => panic!("mode {other}"),
    }
}

fn sort(name: &str) -> Option<BreakSuggestionSortMethod> {
    match name {
        "NUM_CHANGES_THEN_SUMMED_FREQUENCY" => {
            Some(BreakSuggestionSortMethod::NumChangesThenSummedFrequency)
        }
        "NUM_CHANGES_THEN_MAX_FREQUENCY" => {
            Some(BreakSuggestionSortMethod::NumChangesThenMaxFrequency)
        }
        "null" => None,
        other => panic!("sort {other}"),
    }
}

fn word(w: &SuggestWord) -> String {
    format!("{}:{}:{:x}", w.string, w.freq, w.score.to_bits())
}

#[test]
fn word_breaks_match_lucene() {
    let reader = reader();
    assert_eq!(reader.segment_readers().len(), 3);
    let text = std::fs::read_to_string(dir().join("breaks.tsv"))
        .expect("run scripts/gen-fixtures.sh --only GenSpellWordBreak");
    let (mut rows, mut suggested) = (0, 0);
    for line in text.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        let s = checker(f[0]);
        let got: Vec<String> = s
            .suggest_word_breaks(
                &Term::new("wb", f[1].as_bytes()),
                f[4].parse().unwrap(),
                &reader,
                mode(f[2]),
                sort(f[3]),
            )
            .unwrap()
            .iter()
            .map(|sugg| sugg.iter().map(word).collect::<Vec<_>>().join(" "))
            .collect();
        suggested += got.len();
        assert_eq!(got, f[5..], "{line}");
        rows += 1;
    }
    assert_eq!(rows, 10 * 15 * 3 * 2 * 5 + 1);
    assert!(suggested > 500, "{suggested}");
}

#[test]
fn word_combinations_match_lucene() {
    let reader = reader();
    let text = std::fs::read_to_string(dir().join("combinations.tsv")).unwrap();
    let (mut rows, mut suggested) = (0, 0);
    for line in text.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        let s = checker(f[0]);
        let terms: Vec<Term> = f[1]
            .split(' ')
            .map(|t| {
                if t == "|" {
                    separator_term()
                } else {
                    Term::new("wb", t.as_bytes())
                }
            })
            .collect();
        let got: Vec<String> = s
            .suggest_word_combinations(&terms, f[3].parse().unwrap(), &reader, mode(f[2]).unwrap())
            .unwrap()
            .iter()
            .map(|c| {
                let idx: Vec<String> = c
                    .original_term_indexes
                    .iter()
                    .map(|i| i.to_string())
                    .collect();
                format!("{}={}", idx.join(","), word(&c.suggestion))
            })
            .collect();
        suggested += got.len();
        assert_eq!(got, f[4..], "{line}");
        rows += 1;
    }
    assert_eq!(rows, 10 * 10 * 3 * 4);
    assert!(suggested > 200, "{suggested}");
}

#[test]
fn the_settings_read_back() {
    let mut s = WordBreakSpellChecker::default();
    assert_eq!(
        (
            s.get_min_suggestion_frequency(),
            s.get_min_break_word_length(),
            s.get_max_combine_word_length(),
            s.get_max_changes(),
            s.get_max_evaluations()
        ),
        (1, 1, 20, 1, 1000)
    );
    s.set_min_suggestion_frequency(3);
    s.set_min_break_word_length(2);
    s.set_max_combine_word_length(9);
    s.set_max_changes(4);
    s.set_max_evaluations(7);
    assert_eq!(
        (
            s.get_min_suggestion_frequency(),
            s.get_min_break_word_length(),
            s.get_max_combine_word_length(),
            s.get_max_changes(),
            s.get_max_evaluations()
        ),
        (3, 2, 9, 4, 7)
    );
}
