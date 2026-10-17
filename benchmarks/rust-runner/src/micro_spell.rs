//! M13 T13.3e's spell pair (the Java twin is
//! `benchmarks/micro/java/SpellMicro.java`) over 5,000 pseudo-words from a
//! shared LCG, word `i` in `i % 5 + 1` documents of a one-segment index.
//! `direct_suggest`: `DirectSpellChecker` top 5 for 500 misspellings, units:
//! lookups; `spell_suggest`: `SpellChecker` top 5 for the same, units:
//! lookups; `spell_index`: a spell index built from every word, units: words;
//! `word_break`: `WordBreakSpellChecker` breaks of 500 word pairs run
//! together (max changes 2), units: lookups.

use std::hint::black_box;
use std::time::Duration;

use lucene_index::document::{Document, Store, StringField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::merge_policy::upgrade::LATEST;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::query_visitor::Term;
use lucene_search::spell::direct::DirectSpellChecker;
use lucene_search::spell::plain_text::PlainTextDictionary;
use lucene_search::spell::spell_checker::SpellChecker;
use lucene_search::spell::word_break::WordBreakSpellChecker;
use lucene_store::ByteBuffersDirectory;

use super::measure;

/// `SpellMicro.words()`: 5,000 words of 4 to 10 letters from a 64-bit LCG.
pub(super) fn words() -> Vec<String> {
    let mut state: u64 = 42;
    let mut next = |bound: u64| {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 33) % bound
    };
    (0..5000)
        .map(|_| {
            let len = 4 + next(7);
            (0..len).map(|_| char::from(b'a' + next(26) as u8)).collect()
        })
        .collect()
}

/// `SpellMicro.misspellings()`: every tenth word with one letter dropped
/// (even) or two swapped (odd).
pub(super) fn misspellings(words: &[String]) -> Vec<String> {
    words
        .iter()
        .step_by(10)
        .enumerate()
        .map(|(i, w)| {
            let mut b = w.as_bytes().to_vec();
            let at = i % (b.len() - 1);
            if i % 2 == 0 {
                b.remove(at);
            } else {
                b.swap(at, at + 1);
            }
            String::from_utf8(b).unwrap()
        })
        .collect()
}

pub fn bench_spell(w: Duration, m: Duration) {
    let words = words();
    let queries = misspellings(&words);
    let user = ByteBuffersDirectory::new();
    {
        let mut iw = IndexWriter::open(&user, Vec::new(), "Lucene104", LATEST).unwrap();
        for (i, word) in words.iter().enumerate() {
            for _ in 0..=i % 5 {
                let mut doc = Document::new();
                doc.add(StringField::new("body", word.as_str(), Store::No));
                iw.add_fields_document(&doc).unwrap();
            }
        }
        iw.commit().unwrap();
        iw.force_merge(1).unwrap();
        iw.commit().unwrap();
    }
    let reader = DirectoryReader::open(&user).unwrap();

    let direct = DirectSpellChecker::new();
    measure("direct_suggest", w, m, || {
        for q in &queries {
            black_box(direct.suggest_similar("body", q, 5, &reader).unwrap().len());
        }
        queries.len() as u64
    });

    let text: String = words.iter().map(|w| format!("{w}\n")).collect();
    let spell_dir = ByteBuffersDirectory::new();
    let mut spell = SpellChecker::new(&spell_dir).unwrap();
    spell.index_dictionary(&PlainTextDictionary::from_text(&text), true).unwrap();
    measure("spell_suggest", w, m, || {
        for q in &queries {
            black_box(spell.suggest_similar(q, 5).unwrap().len());
        }
        queries.len() as u64
    });
    measure("spell_index", w, m, || {
        let d = ByteBuffersDirectory::new();
        let mut s = SpellChecker::new(&d).unwrap();
        s.index_dictionary(&PlainTextDictionary::from_text(&text), true).unwrap();
        black_box(s.exist(&words[0]).unwrap());
        words.len() as u64
    });

    let pairs: Vec<Term> = (0..500)
        .map(|i| Term::new("body", format!("{}{}", words[i * 7], words[i * 7 + 3]).as_bytes()))
        .collect();
    let mut wb = WordBreakSpellChecker::new();
    wb.set_max_changes(2);
    measure("word_break", w, m, || {
        for t in &pairs {
            black_box(wb.suggest_word_breaks(t, 5, &reader, None, None).unwrap().len());
        }
        pairs.len() as u64
    });
}
