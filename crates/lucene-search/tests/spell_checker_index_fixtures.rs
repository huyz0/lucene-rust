//! M13 T13.3e against Lucene 10.5.0 (`fixtures/src/GenSpellCheckerIndex.java`):
//! the classic `SpellChecker`, over the spell indexes Lucene wrote and over
//! the ones this port writes from the same dictionaries, with and without a
//! user index, eight configurations x every query x three sizes x every mode.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use lucene_search::directory_reader::DirectoryReader;
use lucene_search::spell::distance::{
    JaroWinklerDistance, LevenshteinDistance, LuceneLevenshteinDistance, NGramDistance,
};
use lucene_search::spell::index_dictionary::LuceneDictionary;
use lucene_search::spell::plain_text::PlainTextDictionary;
use lucene_search::spell::spell_checker::SpellChecker;
use lucene_search::spell::suggest_word::{frequency_comparator, score_comparator, SuggestMode};
use lucene_search::Error;
use lucene_store::{ByteBuffersDirectory, Directory, FsDirectory};

/// `GenSpellCheckerIndex.DICTIONARY`.
const DICTIONARY: &str =
    "lucene\nlucine\nlucent\nsearch\nsearches\nsearcher\nseach\napache\napeche\nindex\nindexes\n\
spell\nspelling\nsmell\nshell\nspill\nchecker\ncheckers\ncafé\ncafe\ncaffe\nnaïve\n\
naive\n日本語\nab\nabc\nabcd\nword\nworld\nsword\nwords\nbreakfast\nbreak\nfast\n\
lucene\n😀xyz\n😀xzz\nfive5\nfives\nsixsix\nsixsis\nwordy\nworldly\n\nzz\nz\n";

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/spell_checker_index")
}

fn user() -> DirectoryReader {
    DirectoryReader::open(&FsDirectory::open(dir().join("user"))).unwrap()
}

fn configure(s: &mut SpellChecker<'_>, name: &str) -> Option<f32> {
    match name {
        "default" => {}
        "accuracy_0" => s.set_accuracy(0.0),
        "accuracy_0.8" => s.set_accuracy(0.8),
        "accuracy_arg_0.3" => return Some(0.3),
        "frequency_comparator" => s.set_comparator(frequency_comparator()),
        "distance_jarowinkler" => s.set_string_distance(Arc::new(JaroWinklerDistance::new())),
        "distance_ngram" => s.set_string_distance(Arc::new(NGramDistance::new())),
        "distance_lucene_levenshtein" => {
            s.set_string_distance(Arc::new(LuceneLevenshteinDistance::new()))
        }
        other => panic!("configuration {other}"),
    }
    None
}

fn mode(name: &str) -> SuggestMode {
    match name {
        "SUGGEST_WHEN_NOT_IN_INDEX" => SuggestMode::SuggestWhenNotInIndex,
        "SUGGEST_MORE_POPULAR" => SuggestMode::SuggestMorePopular,
        "SUGGEST_ALWAYS" => SuggestMode::SuggestAlways,
        other => panic!("mode {other}"),
    }
}

/// Every `suggest.tsv` row of `stage` against the spell index in `spell`.
fn check_stage(stage: &str, spell: &dyn Directory, user: &DirectoryReader) -> usize {
    let text = std::fs::read_to_string(dir().join("suggest.tsv"))
        .expect("run scripts/gen-fixtures.sh --only GenSpellCheckerIndex");
    let (mut rows, mut suggested) = (0, 0);
    let mut checkers: BTreeMap<String, (SpellChecker<'_>, Option<f32>)> = BTreeMap::new();
    for line in text.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        if f[0] != stage {
            continue;
        }
        let (s, arg) = checkers.entry(f[1].to_string()).or_insert_with(|| {
            let mut s = SpellChecker::new(spell).unwrap();
            let arg = configure(&mut s, f[1]);
            (s, arg)
        });
        let num: usize = f[3].parse().unwrap();
        let got = match (f[4], *arg) {
            ("-", None) => s.suggest_similar(f[2], num),
            ("-", Some(a)) => s.suggest_similar_with_accuracy(f[2], num, a),
            (m, None) => s.suggest_similar_in(f[2], num, Some((user, "body")), mode(m)),
            (m, Some(a)) => {
                s.suggest_similar_in_with_accuracy(f[2], num, Some((user, "body")), mode(m), a)
            }
        }
        .unwrap();
        suggested += got.len();
        assert_eq!(got, f[5..], "{stage}: {line}");
        rows += 1;
    }
    assert_eq!(rows, 8 * 30 * 3 * 4, "{stage}");
    assert!(suggested > 1000, "{suggested}");
    rows
}

fn check_exist(stage: &str, spell: &dyn Directory) {
    let s = SpellChecker::new(spell).unwrap();
    let text = std::fs::read_to_string(dir().join("exist.tsv")).unwrap();
    let mut rows = 0;
    for line in text
        .lines()
        .filter(|l| l.starts_with(&format!("{stage}\t")))
    {
        let f: Vec<&str> = line.split('\t').collect();
        assert_eq!(s.exist(f[1]).unwrap().to_string(), f[2], "{line}");
        rows += 1;
    }
    assert_eq!(rows, 30);
}

#[test]
fn suggestions_over_lucenes_spell_indexes_match_lucene() {
    let user = user();
    for stage in ["first", "second"] {
        let spell = FsDirectory::open(dir().join(stage));
        check_stage(stage, &spell, &user);
        check_exist(stage, &spell);
    }
}

#[test]
fn spell_indexes_this_port_writes_answer_as_lucenes_do() {
    let user = user();
    let first = ByteBuffersDirectory::new();
    let second = ByteBuffersDirectory::new();
    for (stage, spell) in [("first", &first), ("second", &second)] {
        let mut s = SpellChecker::new(spell).unwrap();
        s.index_dictionary(&PlainTextDictionary::from_text(DICTIONARY), true)
            .unwrap();
        if stage == "second" {
            s.index_dictionary(&LuceneDictionary::new(&user, "body"), false)
                .unwrap();
        }
        s.close().unwrap();
        check_stage(stage, spell, &user);
        check_exist(stage, spell);
    }
    // the port's spell index has Lucene's shape: one merged segment, then
    // the user index's new words in a second
    assert_eq!(
        DirectoryReader::open(&first)
            .unwrap()
            .segment_readers()
            .len(),
        1
    );
    assert_eq!(
        DirectoryReader::open(&second)
            .unwrap()
            .segment_readers()
            .len(),
        2
    );
}

#[test]
fn the_life_cycle_matches_lucenes() {
    let dir = ByteBuffersDirectory::new();
    let mut s = SpellChecker::with_distance(&dir, Arc::new(LevenshteinDistance::new())).unwrap();
    assert!(!s.is_closed());
    // an empty spell index was committed, and answers nothing
    assert!(lucene_index::segment_infos::read_latest(&dir).is_ok());
    assert_eq!(
        s.suggest_similar("lucene", 5).unwrap(),
        Vec::<String>::new()
    );
    assert!(!s.exist("lucene").unwrap());
    // numSug 0: Lucene's searcher refuses 0 hits
    assert!(
        matches!(s.suggest_similar("lucene", 0), Err(Error::IllegalArgument(m)) if m == "numHits must be > 0")
    );

    s.index_dictionary(
        &PlainTextDictionary::from_text("lucene\nlucent\nab\n"),
        false,
    )
    .unwrap();
    assert!(s.exist("lucene").unwrap() && !s.exist("ab").unwrap());
    assert_eq!(
        s.suggest_similar("lucena", 5).unwrap(),
        ["lucene", "lucent"]
    );
    // an empty word asks for no n-grams, which match nothing
    assert!(s.suggest_similar("", 5).unwrap().is_empty());
    // indexing the same words again adds nothing
    s.index_dictionary(&PlainTextDictionary::from_text("lucene\nlucent\n"), true)
        .unwrap();
    assert_eq!(
        s.suggest_similar("lucena", 5).unwrap(),
        ["lucene", "lucent"]
    );

    // the word itself is never suggested; the user index decides the mode
    assert_eq!(s.suggest_similar("lucene", 5).unwrap(), ["lucent"]);
    assert_eq!(s.get_accuracy(), 0.5);
    s.set_accuracy(0.9);
    assert_eq!(
        s.suggest_similar("lucena", 5).unwrap(),
        Vec::<String>::new()
    );
    assert_eq!(
        s.suggest_similar_with_accuracy("lucena", 5, 0.1).unwrap(),
        ["lucene", "lucent"]
    );
    s.set_comparator(score_comparator());
    let _ = s.get_comparator();
    s.set_string_distance(Arc::new(NGramDistance::new()));
    assert_eq!(s.get_string_distance().get_distance("ab", "ab"), 1.0);
    assert!(format!("{s:?}").contains("SpellChecker"));

    // another spell index, created when empty
    let other = ByteBuffersDirectory::new();
    s.set_spell_index(&other).unwrap();
    assert!(!s.exist("lucene").unwrap());
    s.set_spell_index(&dir).unwrap();
    assert!(s.exist("lucene").unwrap());

    s.clear_index().unwrap();
    assert!(!s.exist("lucene").unwrap());
    assert_eq!(
        s.suggest_similar_with_accuracy("lucena", 5, 0.1).unwrap(),
        Vec::<String>::new()
    );

    s.close().unwrap();
    assert!(s.is_closed());
    let closed = |r: Result<(), Error>| {
        assert!(matches!(r, Err(Error::AlreadyClosed(m)) if m == "Spellchecker has been closed"))
    };
    closed(s.close());
    closed(s.clear_index());
    closed(s.exist("x").map(drop));
    closed(s.suggest_similar("x", 1).map(drop));
    closed(s.index_dictionary(&PlainTextDictionary::from_text("abc\n"), false));
    closed(s.set_spell_index(&other));
}
