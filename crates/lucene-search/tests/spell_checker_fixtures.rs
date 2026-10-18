//! M13 T13.3e against Lucene 10.5.0 (`fixtures/src/GenSpellChecker.java`):
//! `DirectSpellChecker` over a four-segment index with deletions, 22
//! configurations x every query x the three modes x three sizes; the
//! dictionaries a `LuceneDictionary` and a `HighFrequencyDictionary` read off
//! it; and the setters' errors.

use std::path::PathBuf;
use std::sync::Arc;

use lucene_search::directory_reader::DirectoryReader;
use lucene_search::spell::direct::DirectSpellChecker;
use lucene_search::spell::distance::{
    JaroWinklerDistance, LevenshteinDistance, LuceneLevenshteinDistance, NGramDistance,
};
use lucene_search::spell::index_dictionary::{HighFrequencyDictionary, LuceneDictionary};
use lucene_search::spell::suggest_word::{frequency_comparator, SuggestMode};
use lucene_search::spell::Dictionary;
use lucene_search::suggest::input::InputIterator;
use lucene_store::FsDirectory;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/spell_checker")
}

fn reader() -> DirectoryReader {
    DirectoryReader::open(&FsDirectory::open(dir().join("index"))).unwrap()
}

/// The configuration a fixture row names, and the accuracy it passes to the
/// five-argument `suggestSimilar`, if any.
fn checker(name: &str) -> (DirectSpellChecker, Option<f32>) {
    let mut s = DirectSpellChecker::new();
    let mut arg = None;
    match name {
        "default" => {}
        "max_edits_1" => s.set_max_edits(1).unwrap(),
        "min_prefix_0" => s.set_min_prefix(0),
        "min_prefix_2" => s.set_min_prefix(2),
        "inspections_1" => s.set_max_inspections(1),
        "inspections_20" => s.set_max_inspections(20),
        "accuracy_0" => s.set_accuracy(0.0),
        "accuracy_0.8" => s.set_accuracy(0.8),
        "accuracy_arg_0.3" => arg = Some(0.3),
        "threshold_0.05" => s.set_threshold_frequency(0.05).unwrap(),
        "threshold_3" => s.set_threshold_frequency(3.0).unwrap(),
        "max_query_freq_0" => s.set_max_query_frequency(0.0).unwrap(),
        "max_query_freq_0.5" => s.set_max_query_frequency(0.5).unwrap(),
        "max_query_freq_5" => s.set_max_query_frequency(5.0).unwrap(),
        "query_length_1_5" => {
            s.set_min_query_length(1).unwrap();
            s.set_max_query_length(5).unwrap();
        }
        "no_lowercase" => s.set_lower_case_terms(false),
        "frequency_comparator" => s.set_comparator(frequency_comparator()),
        "distance_levenshtein" => s.set_distance(Arc::new(LevenshteinDistance::new())),
        "distance_jarowinkler" => s.set_distance(Arc::new(JaroWinklerDistance::new())),
        "distance_ngram" => s.set_distance(Arc::new(NGramDistance::new())),
        "distance_lucene_levenshtein" => s.set_distance(Arc::new(LuceneLevenshteinDistance::new())),
        "mixed" => {
            s.set_max_edits(1).unwrap();
            s.set_min_prefix(0);
            s.set_accuracy(0.3);
            s.set_max_inspections(2);
            s.set_distance(Arc::new(NGramDistance::with_size(3)));
            s.set_comparator(frequency_comparator());
            s.set_threshold_frequency(2.0).unwrap();
            s.set_max_query_frequency(0.9).unwrap();
        }
        other => panic!("configuration {other}"),
    }
    (s, arg)
}

fn mode(name: &str) -> SuggestMode {
    match name {
        "SUGGEST_WHEN_NOT_IN_INDEX" => SuggestMode::SuggestWhenNotInIndex,
        "SUGGEST_MORE_POPULAR" => SuggestMode::SuggestMorePopular,
        "SUGGEST_ALWAYS" => SuggestMode::SuggestAlways,
        other => panic!("mode {other}"),
    }
}

#[test]
fn direct_spell_checker_suggests_as_lucene_does() {
    let reader = reader();
    assert_eq!(reader.segment_readers().len(), 4);
    let text = std::fs::read_to_string(dir().join("direct.tsv"))
        .expect("run scripts/gen-fixtures.sh --only GenSpellChecker");
    let (mut rows, mut suggested) = (0, 0);
    for line in text.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        let num: usize = f[3].parse().unwrap();
        let got = if let Some(field) = f[0].strip_prefix("field_") {
            DirectSpellChecker::new()
                .suggest_similar(field, f[1], num, &reader)
                .unwrap()
                .iter()
                .map(|w| format!("{}:{}", w.string, w.freq))
                .collect::<Vec<_>>()
        } else {
            let (s, arg) = checker(f[0]);
            let words = match arg {
                Some(a) => {
                    s.suggest_similar_with_accuracy("body", f[1], num, &reader, mode(f[2]), a)
                }
                None => s.suggest_similar_with_mode("body", f[1], num, &reader, mode(f[2])),
            }
            .unwrap();
            words
                .iter()
                .map(|w| format!("{}:{}:{:x}", w.string, w.freq, w.score.to_bits()))
                .collect()
        };
        suggested += got.len();
        assert_eq!(got, f[4..], "{line}");
        rows += 1;
    }
    assert_eq!(rows, 22 * 37 * 3 * 3 + 2);
    assert!(suggested > 1000, "{suggested}");
}

fn dump(name: &str, it: &mut dyn InputIterator) -> String {
    let mut s = format!("{name}\t{}\t{}", it.has_payloads(), it.has_contexts());
    while let Some(b) = it.next() {
        let hex: String = b.iter().map(|x| format!("{x:02x}")).collect();
        s.push_str(&format!("\t{hex}:{}", it.weight()));
        assert!(it.payload().is_none() && it.contexts().is_none());
    }
    s
}

#[test]
fn index_dictionaries_read_the_terms_lucene_does() {
    let reader = reader();
    let text = std::fs::read_to_string(dir().join("dictionaries.tsv")).unwrap();
    let want: Vec<&str> = text.lines().collect();
    let mut got = Vec::new();
    for field in ["body", "other", "missing"] {
        let d = LuceneDictionary::new(&reader, field);
        got.push(dump(
            &format!("lucene_{field}"),
            &mut *d.get_entry_iterator().unwrap(),
        ));
    }
    for (th, name) in [
        (0.0, "0.0"),
        (0.05, "0.05"),
        (0.1, "0.1"),
        (0.3, "0.3"),
        (2.0, "2.0"),
    ] {
        let d = HighFrequencyDictionary::new(&reader, "body", th);
        got.push(dump(
            &format!("high_frequency_body_{name}"),
            &mut *d.get_entry_iterator().unwrap(),
        ));
    }
    let d = HighFrequencyDictionary::new(&reader, "missing", 0.0);
    got.push(dump(
        "high_frequency_missing",
        &mut *d.get_entry_iterator().unwrap(),
    ));
    assert_eq!(got, want);
}

#[test]
fn the_setters_refuse_what_lucene_refuses() {
    let text = std::fs::read_to_string(dir().join("errors.tsv")).unwrap();
    let mut rows = 0;
    for line in text.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        let mut s = DirectSpellChecker::new();
        let r = match (f[0], f[1]) {
            ("max_edits", v) => s.set_max_edits(v.parse().unwrap()),
            ("threshold_frequency", v) => s.set_threshold_frequency(v.parse().unwrap()),
            ("max_query_frequency", v) => s.set_max_query_frequency(v.parse().unwrap()),
            ("min_query_length", _) => s
                .set_max_query_length(5)
                .and_then(|()| s.set_min_query_length(10)),
            ("max_query_length", v) => s.set_max_query_length(v.parse().unwrap()),
            other => panic!("{other:?}"),
        };
        let got = match r {
            Ok(()) => "ok".to_string(),
            Err(e) => e.to_string(),
        };
        // Java's class name, then its message
        let want = f[2].split_once(": ").map_or(f[2], |(_, m)| m);
        assert!(got.ends_with(want), "{line}: {got}");
        rows += 1;
    }
    assert_eq!(rows, 11);
}

#[test]
fn the_settings_read_back_with_lucenes_defaults() {
    let mut s = DirectSpellChecker::default();
    assert_eq!(
        (
            s.get_max_edits(),
            s.get_min_prefix(),
            s.get_max_inspections(),
            s.get_accuracy()
        ),
        (2, 1, 5, 0.5)
    );
    assert_eq!(
        (
            s.get_threshold_frequency(),
            s.get_min_query_length(),
            s.get_max_query_length()
        ),
        (0.0, 4, i32::MAX)
    );
    assert_eq!(
        (s.get_max_query_frequency(), s.get_lower_case_terms()),
        (0.01, true)
    );
    assert!(format!("{:?}", s.get_distance()).contains("Levenshtein"));
    assert!(format!("{s:?}").starts_with("DirectSpellChecker"));
    s.set_max_edits(1).unwrap();
    s.set_min_prefix(0);
    s.set_max_inspections(9);
    s.set_accuracy(0.25);
    s.set_threshold_frequency(3.0).unwrap();
    s.set_min_query_length(2).unwrap();
    s.set_max_query_length(8).unwrap();
    s.set_max_query_frequency(0.5).unwrap();
    s.set_lower_case_terms(false);
    s.set_distance(Arc::new(NGramDistance::with_size(3)));
    s.set_comparator(frequency_comparator());
    assert_eq!(
        (
            s.get_max_edits(),
            s.get_min_prefix(),
            s.get_max_inspections(),
            s.get_accuracy()
        ),
        (1, 0, 9, 0.25)
    );
    assert_eq!(
        (
            s.get_threshold_frequency(),
            s.get_min_query_length(),
            s.get_max_query_length()
        ),
        (3.0, 2, 8)
    );
    assert_eq!(
        (s.get_max_query_frequency(), s.get_lower_case_terms()),
        (0.5, false)
    );
    assert!(format!("{:?}", s.get_distance()).contains("NGram"));
    let _ = s.get_comparator();
    // numSug 0: Java's queue would throw; nothing is suggested
    let reader = reader();
    assert!(DirectSpellChecker::new()
        .suggest_similar("body", "lucenee", 0, &reader)
        .unwrap()
        .is_empty());
}
