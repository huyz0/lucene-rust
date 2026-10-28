//! What opening and polling a many-segment index costs, and two per-call
//! reader paths that used to redo open-time work:
//!
//! - `open` -- `DirectoryReader::open` over a 30-segment index with deletes
//!   (segments opened, `.liv` files parsed, doc bases assigned).
//! - `open_if_changed_noop` -- a poll when nothing changed.
//! - `num_docs` -- `DirectoryReader::num_docs`, which counted every
//!   segment's live-docs bits on each call.
//! - `term_vectors_all_docs` -- `LeafReader::term_vectors(doc)` over every
//!   document, which re-opened (and re-checksummed `.tvm`) per document.
//!
//! The index is built in memory by this port's own `IndexWriter`, so the
//! bench needs no corpus. Run with
//! `cargo bench -p lucene-search --bench reader_open`.

use std::sync::Arc;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lucene_analysis::Analyzer;
use lucene_index::buffered_updates::Term;
use lucene_index::document::{Document, Field, FieldType, Store, StringField, TextField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::reader::LeafReader;
use lucene_store::ByteBuffersDirectory;

const SEGMENTS: usize = 30;
const DOCS_PER_SEGMENT: usize = 200;

const WORDS: &[&str] = &[
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet",
    "kilo", "lima", "mike", "november", "oscar", "papa", "quebec", "romeo", "sierra", "tango",
];

fn build() -> ByteBuffersDirectory {
    let dir = ByteBuffersDirectory::new();
    let version = LuceneVersion {
        major: 10,
        minor: 0,
        bugfix: 0,
    };
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", version).expect("open writer");
    w.set_analyzer(Some(Arc::new(Analyzer::standard(None))));
    w.set_use_compound_file(false);
    let mut body_type = FieldType::copy_of(&TextField::type_stored());
    body_type.set_store_term_vectors(true).expect("tv");
    body_type.set_store_term_vector_positions(true).expect("tv");
    let body_type = body_type.frozen();
    let mut id = 0usize;
    for seg in 0..SEGMENTS {
        for _ in 0..DOCS_PER_SEGMENT {
            let mut d = Document::new();
            d.add(StringField::new("id", id.to_string(), Store::Yes));
            let text: Vec<&str> = (0..12)
                .map(|k| WORDS[(id.wrapping_mul(7).wrapping_add(k * 3 + seg)) % WORDS.len()])
                .collect();
            d.add(Field::from_string("body", text.join(" "), body_type.clone()).expect("field"));
            w.add_fields_document(&d).expect("add");
            id += 1;
        }
        w.commit().expect("commit");
    }
    // Deletes in every segment: each gets a `.liv`.
    let terms: Vec<Term> = (0..id)
        .step_by(7)
        .map(|i| Term::new("id", i.to_string()))
        .collect();
    w.delete_documents_by_term(&terms).expect("delete");
    w.commit().expect("commit");
    drop(w);
    dir
}

fn bench(c: &mut Criterion) {
    let dir = build();
    let reader = DirectoryReader::open(&dir).expect("open");
    assert_eq!(reader.segment_readers().len(), SEGMENTS);
    assert!(reader.num_docs() < reader.max_doc());

    c.bench_function("open", |b| {
        b.iter(|| black_box(DirectoryReader::open(&dir).expect("open")))
    });
    c.bench_function("open_if_changed_noop", |b| {
        b.iter(|| black_box(reader.open_if_changed(&dir).expect("poll").is_none()))
    });
    c.bench_function("num_docs", |b| b.iter(|| black_box(reader.num_docs())));
    c.bench_function("term_vectors_all_docs", |b| {
        b.iter(|| {
            let mut terms = 0usize;
            for seg in reader.segment_readers() {
                for doc in 0..seg.max_doc {
                    if let Some(tv) = seg.term_vectors(doc).expect("tv") {
                        terms += tv.fields.len();
                    }
                }
            }
            black_box(terms)
        })
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
