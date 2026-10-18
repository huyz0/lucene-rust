//! The document API's indexing cost, the twin of
//! `benchmarks/micro/java/DocIndexMicro.java`: each case builds a fresh
//! in-memory index of 5,000 documents through
//! `IndexWriter::add_fields_document`, commits and closes it. Units:
//! documents. `doc_strings`: `SpellChecker`'s document shape (the word,
//! stored, and its start/inner/end 3- and 4-grams as unstored
//! `StringField`s); `doc_text`: a stored `StringField` id and a 30-word
//! `TextField` through the standard analyzer; `doc_values`: a numeric, a
//! sorted and three sorted-set doc values and a `LongPoint`.

use std::hint::black_box;
use std::time::Duration;

use lucene_index::document::{
    Document, LongPoint, NumericDocValuesField, SortedDocValuesField, SortedSetDocValuesField,
    Store, StringField, TextField,
};
use lucene_index::index_writer::IndexWriter;
use lucene_index::merge_policy::upgrade::LATEST;
use lucene_store::{ByteBuffersDirectory, Directory};

use super::measure;
use super::micro_spell::words;

const DOCS: usize = 5000;

/// `DocIndexMicro.next`: `SpellMicro`'s LCG on its own state.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self, bound: u64) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) % bound
    }
}

fn string_docs(words: &[String]) -> Vec<Document> {
    words[..DOCS]
        .iter()
        .map(|w| {
            let mut d = Document::new();
            d.add(StringField::new("word", w.as_str(), Store::Yes));
            for n in 3..=4 {
                if w.len() < n {
                    continue;
                }
                d.add(StringField::new(format!("start{n}"), &w[..n], Store::No));
                for k in 0..=w.len() - n {
                    d.add(StringField::new(format!("gram{n}"), &w[k..k + n], Store::No));
                }
                d.add(StringField::new(format!("end{n}"), &w[w.len() - n..], Store::No));
            }
            d
        })
        .collect()
}

fn text_docs(words: &[String], rng: &mut Lcg) -> Vec<Document> {
    (0..DOCS)
        .map(|i| {
            let body: Vec<&str> = (0..30)
                .map(|_| words[rng.next(2000) as usize].as_str())
                .collect();
            let mut d = Document::new();
            d.add(StringField::new("id", i.to_string(), Store::Yes));
            d.add(TextField::new("body", body.join(" "), Store::No));
            d
        })
        .collect()
}

fn value_docs(words: &[String], rng: &mut Lcg) -> Vec<Document> {
    (0..DOCS)
        .map(|_| {
            let mut d = Document::new();
            d.add(NumericDocValuesField::new("n", rng.next(1_000_000) as i64));
            d.add(SortedDocValuesField::new(
                "s",
                words[rng.next(500) as usize].as_bytes(),
            ));
            for _ in 0..3 {
                d.add(SortedSetDocValuesField::new(
                    "tags",
                    words[rng.next(100) as usize].as_bytes(),
                ));
            }
            d.add(LongPoint::new("p", &[rng.next(1_000_000_000) as i64]).unwrap());
            d
        })
        .collect()
}

fn index(docs: &[Document]) -> u64 {
    let dir = ByteBuffersDirectory::new();
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", LATEST).unwrap();
    for d in docs {
        w.add_fields_document(d).unwrap();
    }
    w.commit().unwrap();
    w.close().unwrap();
    black_box(dir.list_all().unwrap().len());
    docs.len() as u64
}

pub fn bench_doc_index(w: Duration, m: Duration) {
    let words = words();
    let mut rng = Lcg(7);
    let strings = string_docs(&words);
    let text = text_docs(&words, &mut rng);
    let values = value_docs(&words, &mut rng);
    measure("doc_strings", w, m, || index(&strings));
    measure("doc_text", w, m, || index(&text));
    measure("doc_values", w, m, || index(&values));
}
