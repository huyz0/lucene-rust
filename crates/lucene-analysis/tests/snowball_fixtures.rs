//! M11 T11.3: every Snowball stemmer against Lucene's `SnowballFilter`.
//!
//! `fixtures/src/GenSnowball.java` stems, per language, every string of the
//! stemmer's `Among` tables with random stems around it, random stem + suffix
//! chains and the corpus's words, through Lucene 10.5.0's `SnowballFilter`
//! over a `KeywordTokenizer`, and writes `snowball/<Language>.words` (`word`
//! or `word\tstem`). This runs the same chain in Rust and requires every stem
//! to be equal.

use lucene_analysis::snowball::{SnowballFilter, SnowballStemmer};
use lucene_analysis::{
    AnalysisError, Analyzer, AnalyzerDefinition, KeywordTokenizer, TokenStream,
    TokenStreamComponents,
};

fn dir() -> &'static str {
    concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures/data/snowball/")
}

struct Chain(&'static str);

impl AnalyzerDefinition for Chain {
    fn create_components(&self, _field: &str) -> Result<TokenStreamComponents, AnalysisError> {
        let filter = SnowballFilter::with_name(KeywordTokenizer::new(), self.0)?;
        Ok(TokenStreamComponents::new(filter))
    }
}

fn stem(a: &Analyzer, word: &str) -> String {
    let mut ts = a.token_stream("f", word).expect("token stream");
    ts.reset().expect("reset");
    assert!(
        ts.increment_token().expect("token"),
        "no token for {word:?}"
    );
    let s = ts.attributes().term().to_string();
    assert!(!ts.increment_token().expect("token"));
    ts.end().expect("end");
    ts.close().expect("close");
    s
}

#[test]
fn every_snowball_stemmer_matches_lucenes_snowball_filter() {
    let mut languages = 0;
    let mut words = 0;
    for name in SnowballStemmer::names() {
        let text = std::fs::read_to_string(format!("{}{name}.words", dir()))
            .unwrap_or_else(|e| panic!("{name}.words: {e}"));
        let a = Analyzer::new(Chain(name));
        let mut direct = SnowballStemmer::for_name(name).expect("a shipped stemmer");
        assert_eq!(direct.name(), name);
        let mut mismatches = Vec::new();
        for (i, line) in text.lines().enumerate() {
            let (word, expected) = line.split_once('\t').unwrap_or((line, line));
            let got = stem(&a, word);
            if got != expected {
                mismatches.push(format!("{word} -> {got} (Lucene: {expected})"));
            }
            if i % 5 == 0 {
                direct.set_current(word);
                direct.stem();
                assert_eq!(direct.current(), expected, "{name} direct: {word}");
            }
            words += 1;
        }
        assert!(
            mismatches.is_empty(),
            "{name}: {} of {} stems differ, first: {:?}",
            mismatches.len(),
            text.lines().count(),
            &mismatches[..mismatches.len().min(10)]
        );
        languages += 1;
    }
    assert_eq!(languages, 30);
    assert!(words > 80_000, "{words} words");
}

/// Words with a character outside the Basic Multilingual Plane, where a
/// UTF-8 runtime would see one character and Java's sees two: the runtime
/// counts UTF-16 units as Java does, so every stem is Lucene's.
#[test]
fn supplementary_characters_stem_as_in_lucene() {
    let text = std::fs::read_to_string(format!("{}supplementary.words", dir())).unwrap();
    let mut n = 0;
    for line in text.lines() {
        let mut f = line.split('\t');
        let (lang, word, expected) = (f.next().unwrap(), f.next().unwrap(), f.next().unwrap());
        let mut s = SnowballStemmer::for_name(lang).unwrap();
        assert!(s.set_current(word));
        s.stem();
        assert_eq!(s.current(), expected, "{lang}: {word}");
        n += 1;
    }
    assert_eq!(n, 30 * 8);
}

/// Random strings over every script the 30 stemmers handle (and some they do
/// not: digits, punctuation, emoji, the empty string) never panic a stemmer:
/// a panic here would cross the FFI as a dead JVM.
#[test]
fn no_stemmer_panics_on_random_strings() {
    let alphabet: Vec<char> = "abcdefghijklmnopqrstuvwxyzáéíóúàèìòùâêîôûäëïöüßçñœæøåčćđšžğışőű\
        αβγδεζηθικλμνξοπρστυφχψωάέήίόύώϊϋΐΰς\
        абвгдежзийклмнопрстуфхцчшщъыьэюяёђјљњћџ\
        ابتثجحخدذرزسشصضطظعغفقكلمنهويىةءآأؤإئ٠١٢\
        אבגדהווזחטיכלמנסעפצקרשתךםןףץײױ\
        कखगघचछजझटठडढणतथदधनपफबभमयरलवशषसहािीुूेैोौंः्\
        கஙசஞடணதநபமயரலவழளறனாிீுூெேைொோௌ்\
        աբգդեզէըթժիլխծկհձղճմյնշոչպջռսվտրցւփքօֆ\
        0123456789'’-_😀𝒜"
        .chars()
        .collect();
    let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = |n: usize| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed % n as u64) as usize
    };
    for name in SnowballStemmer::names() {
        let mut s = SnowballStemmer::for_name(name).unwrap();
        for _ in 0..3000 {
            let len = next(14);
            let word: String = (0..len).map(|_| alphabet[next(alphabet.len())]).collect();
            let mut term = word.clone();
            s.stem_in_place(&mut term);
            assert!(
                term.len() <= 4 * word.len() + 16,
                "{name}: {word} -> {term}"
            );
        }
    }
}

/// `fixtures/src/GenSnowballFuzz.java`: seeded random strings (any BMP
/// character, letters of every script, supplementary characters, chains of
/// among strings with emoji between) through Lucene's `SnowballFilter`; the
/// port's stems are Lucene's, an unpaired surrogate a stemmer leaves being
/// U+FFFD on both sides.
#[test]
fn random_strings_stem_as_in_lucene() {
    let fuzz = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/snowball-fuzz/"
    );
    let mut words = 0;
    for name in SnowballStemmer::names() {
        let text = std::fs::read_to_string(format!("{fuzz}{name}.tsv"))
            .unwrap_or_else(|e| panic!("{name}.tsv: {e}"));
        let a = Analyzer::new(Chain(name));
        let mut mismatches = Vec::new();
        for line in text.lines() {
            let (word, expected) = line.split_once('\t').expect("word\\tstem");
            let got = stem(&a, word);
            if got != expected {
                mismatches.push(format!("{word:?} -> {got:?} (Lucene: {expected:?})"));
            }
            words += 1;
        }
        assert!(
            mismatches.is_empty(),
            "{name}: {} differ, first: {:?}",
            mismatches.len(),
            &mismatches[..mismatches.len().min(5)]
        );
    }
    assert!(words >= 30 * 1900, "{words} words");
}
