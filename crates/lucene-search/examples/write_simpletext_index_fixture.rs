//! Writes `VerifySimpleTextWritePath`'s indexes: IndexWriter publishing
//! through [`SimpleTextDirectory`], as Java's `IndexWriter` with
//! `SimpleTextCodec` writes them. `<output-dir>/flushed` holds two segments
//! (the second compound) with deletions and two doc-values update
//! generations; `<output-dir>/merged` holds the same force-merged into one.
//! `<output-dir>/appended` is written in two sessions, the second a writer
//! opened on the first's SimpleText index (another update round, a third
//! segment, deletions); `<output-dir>/appended-merged` ends force-merged.
//!
//! Usage: `write_simpletext_index_fixture <output-dir>`.
// Test-support code opts out of the arithmetic gate at the file boundary:
// see `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::field_infos::FieldInfo;
use lucene_index::buffered_updates::Term;
use lucene_index::document::{
    BinaryDocValuesField, Document, Field, FieldType, IntPoint, KnnByteVectorField,
    KnnFloatVectorField, NumericDocValuesField, SortedDocValuesField, SortedNumericDocValuesField,
    SortedSetDocValuesField, Store, StoredValue, StringField, TextField, VectorSimilarityFunction,
};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::simpletext_writer::SimpleTextDirectory;
use lucene_store::{Directory, FsDirectory};

/// The version every segment records.
const VERSION: LuceneVersion = LuceneVersion {
    major: 10,
    minor: 5,
    bugfix: 0,
};

/// `VerifySimpleTextWritePath.PER_SEGMENT`.
const PER_SEGMENT: usize = 60;

/// `VerifySimpleTextWritePath.doc`.
fn doc(i: usize) -> Document {
    let mut d = Document::new();
    d.add(StringField::new("id", format!("d{i}"), Store::Yes));
    let mut body = FieldType::copy_of(&TextField::type_stored());
    body.set_store_term_vectors(true).unwrap();
    body.set_store_term_vector_positions(true).unwrap();
    body.set_store_term_vector_offsets(true).unwrap();
    body.freeze();
    d.add(
        Field::from_string(
            "body",
            format!("w{} x{} common t{i} w{}", i % 5, i % 7, i % 3),
            body,
        )
        .unwrap(),
    );
    d.add(Field::stored("n", StoredValue::Int(i as i32)));
    if i % 4 != 1 {
        d.add(IntPoint::new("pt", &[i as i32, -(i as i32) * 3]).unwrap());
        d.add(NumericDocValuesField::new("num", i as i64 * 7 - 50));
        d.add(SortedDocValuesField::new("sorted", format!("s{}", i % 6)));
        d.add(BinaryDocValuesField::new("bin", vec![i as u8; i % 5]));
    }
    d.add(SortedNumericDocValuesField::new("snum", i as i64 % 9));
    d.add(SortedNumericDocValuesField::new("snum", -(i as i64)));
    d.add(SortedSetDocValuesField::new("sset", format!("a{}", i % 4)));
    d.add(SortedSetDocValuesField::new("sset", format!("b{}", i % 3)));
    if i % 3 != 2 {
        d.add(
            KnnFloatVectorField::new(
                "vf",
                vec![i as f32 * 0.5, 1.0, -(i as i32) as f32, 0.25],
                VectorSimilarityFunction::Euclidean,
            )
            .unwrap(),
        );
        d.add(
            KnnByteVectorField::new(
                "vb",
                vec![i as u8, 255 - i as u8, 7],
                VectorSimilarityFunction::Euclidean,
            )
            .unwrap(),
        );
    }
    d
}

/// The documents the first two segments lose.
fn deleted() -> Vec<Term> {
    (0..2 * PER_SEGMENT)
        .filter(|i| i % 11 == 4)
        .map(|i| Term::new("id", format!("d{i}")))
        .collect()
}

/// `VerifySimpleTextWritePath.write`.
fn write(dir: &dyn Directory, merge: bool) {
    let mut w = IndexWriter::open(dir, Vec::new(), "Lucene104", VERSION).unwrap();
    w.set_max_full_flush_merge_wait_millis(0);
    for seg in 0..2 {
        w.set_use_compound_file(seg == 1);
        for i in seg * PER_SEGMENT..(seg + 1) * PER_SEGMENT {
            w.add_fields_document(&doc(i)).unwrap();
        }
        w.commit().unwrap();
    }
    w.delete_documents_by_term(&deleted()).unwrap();
    w.commit().unwrap();
    w.update_numeric_doc_value(Term::new("id", "d3"), "num", 1000)
        .unwrap();
    w.update_numeric_doc_value(Term::new("id", "d71"), "num", -1000)
        .unwrap();
    w.commit().unwrap();
    w.update_numeric_doc_value(Term::new("id", "d3"), "num", 2000)
        .unwrap();
    w.update_binary_doc_value(Term::new("id", "d7"), "bin", &[9, 9])
        .unwrap();
    w.commit().unwrap();
    if merge {
        w.force_merge(1).unwrap();
        w.commit().unwrap();
    }
    w.close().unwrap();
}

/// `VerifySimpleTextWritePath.writeFirst`: the first session of an
/// appended index.
fn write_first(dir: &dyn Directory) {
    let mut w = IndexWriter::open(dir, Vec::new(), "Lucene104", VERSION).unwrap();
    w.set_max_full_flush_merge_wait_millis(0);
    for seg in 0..2 {
        w.set_use_compound_file(seg == 1);
        for i in seg * PER_SEGMENT..(seg + 1) * PER_SEGMENT {
            w.add_fields_document(&doc(i)).unwrap();
        }
        w.commit().unwrap();
    }
    w.delete_documents_by_term(&deleted()).unwrap();
    w.commit().unwrap();
    w.update_numeric_doc_value(Term::new("id", "d3"), "num", 1000)
        .unwrap();
    w.update_numeric_doc_value(Term::new("id", "d71"), "num", -1000)
        .unwrap();
    w.commit().unwrap();
    w.close().unwrap();
}

/// The fields of `dir`'s index, as a writer opened on it declares them
/// (this port's writer takes its field list at open).
fn existing_fields(dir: &dyn Directory) -> Vec<FieldInfo> {
    let reader = DirectoryReader::open(dir).unwrap();
    let mut fields: Vec<FieldInfo> = Vec::new();
    for segment in reader.segment_readers() {
        for field in &segment.field_infos().fields {
            if fields.iter().all(|f| f.name != field.name) {
                let mut field = field.clone();
                field.doc_values_gen = -1;
                field.attributes.clear();
                fields.push(field);
            }
        }
    }
    fields.sort_by_key(|f| f.number);
    fields
}

/// `VerifySimpleTextWritePath.writeSecond`: a writer opened on
/// [`write_first`]'s index.
fn write_second(dir: &dyn Directory, merge: bool) {
    let mut w = IndexWriter::open(dir, existing_fields(dir), "Lucene104", VERSION).unwrap();
    w.set_max_full_flush_merge_wait_millis(0);
    w.update_numeric_doc_value(Term::new("id", "d3"), "num", 2000)
        .unwrap();
    w.update_binary_doc_value(Term::new("id", "d75"), "bin", &[9, 9])
        .unwrap();
    w.commit().unwrap();
    w.set_use_compound_file(false);
    for i in 2 * PER_SEGMENT..2 * PER_SEGMENT + 30 {
        w.add_fields_document(&doc(i)).unwrap();
    }
    w.commit().unwrap();
    w.delete_documents_by_term(&[Term::new("id", "d5"), Term::new("id", "d130")])
        .unwrap();
    w.update_numeric_doc_value(Term::new("id", "d125"), "num", 7)
        .unwrap();
    w.commit().unwrap();
    if merge {
        w.force_merge(1).unwrap();
        w.commit().unwrap();
    }
    w.close().unwrap();
}

fn main() {
    let out = std::path::PathBuf::from(std::env::args().nth(1).expect("output dir"));
    for (name, merge) in [("flushed", false), ("merged", true)] {
        let path = out.join(name);
        std::fs::create_dir_all(&path).unwrap();
        let real = FsDirectory::open(&path);
        let dir = SimpleTextDirectory::new(&real).unwrap();
        write(&dir, merge);
    }
    for (name, merge) in [("appended", false), ("appended-merged", true)] {
        let path = out.join(name);
        std::fs::create_dir_all(&path).unwrap();
        let real = FsDirectory::open(&path);
        write_first(&SimpleTextDirectory::new(&real).unwrap());
        write_second(&SimpleTextDirectory::new(&real).unwrap(), merge);
    }
}
