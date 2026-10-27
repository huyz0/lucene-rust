//! `TestUtil`'s codec pickers: each `always*Format` codec installed in
//! `IndexWriter` writes its family's files in the given format and the rest
//! as `Asserting`, and the index reads back clean through `checkReader`.

use std::sync::Arc;

use lucene_codecs::codec::Lucene104Codec;
use lucene_codecs::per_field_doc_values::Lucene90DocValuesFormat;
use lucene_codecs::per_field_knn_vectors::KnnVectorsFormat;
use lucene_index::document::{
    Document, KnnFloatVectorField, NumericDocValuesField, SortedDocValuesField, Store, StringField,
    TextField, VectorSimilarityFunction,
};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::reader::IndexReader;
use lucene_store::{ByteBuffersDirectory, Directory};
use lucene_test_framework::util::check_reader::check_reader;
use lucene_test_framework::util::test_util::{
    always_doc_values_format, always_knn_vectors_format, always_postings_format,
    get_default_postings_format, get_postings_format_with_ords,
};
use lucene_util::java_random::JavaRandom;

fn doc(i: usize) -> Document {
    let mut d = Document::new();
    d.add(StringField::new("id", format!("d{i}"), Store::Yes));
    d.add(TextField::new(
        "body",
        format!("w{} x{} common t{i}", i % 5, i % 7),
        Store::No,
    ));
    d.add(NumericDocValuesField::new("n", i as i64));
    d.add(SortedDocValuesField::new("s", format!("s{}", i % 4)));
    let f = (i % 10) as f32;
    d.add(
        KnnFloatVectorField::new("v", vec![f, 1.0, -f], VectorSimilarityFunction::Euclidean)
            .unwrap(),
    );
    d
}

/// Three flushed segments, then (when asked) a force-merge to one; returns
/// the directory's file names.
fn write(dir: &ByteBuffersDirectory, codec: Arc<dyn Lucene104Codec>, merge: bool) -> Vec<String> {
    lucene_test_framework::codecs::register_codecs();
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut w = IndexWriter::open(dir, Vec::new(), "Lucene104", version).unwrap();
    w.set_codec(Some(codec)).unwrap();
    w.set_max_full_flush_merge_wait_millis(0);
    for seg in 0..3 {
        for i in seg * 50..(seg + 1) * 50 {
            w.add_fields_document(&doc(i)).unwrap();
        }
        w.commit().unwrap();
    }
    if merge {
        w.force_merge(1).unwrap();
        w.commit().unwrap();
    }
    drop(w);
    dir.list_all().unwrap()
}

fn read_back(dir: &ByteBuffersDirectory) {
    let reader = DirectoryReader::open(dir).unwrap();
    assert_eq!(reader.num_docs(), 150);
    check_reader(&reader).unwrap();
    let total: usize = reader
        .leaves()
        .iter()
        .map(|l| l.reader.terms("body").unwrap().map_or(0, |_| 1))
        .sum();
    assert_eq!(total, reader.leaves().len());
}

fn has_file(files: &[String], suffix: &str) -> bool {
    files.iter().any(|f| f.ends_with(suffix))
}

fn check(codec: Arc<dyn Lucene104Codec>, expected: &[&str], absent: &[&str]) {
    for merge in [false, true] {
        let dir = ByteBuffersDirectory::new();
        let files = write(&dir, codec.clone(), merge);
        for suffix in expected {
            assert!(
                has_file(&files, suffix),
                "merge={merge}: no *{suffix} in {files:?}"
            );
        }
        for suffix in absent {
            assert!(
                !has_file(&files, suffix),
                "merge={merge}: *{suffix} in {files:?}"
            );
        }
        read_back(&dir);
    }
}

#[test]
fn always_postings_format_routes_every_field() {
    // Small seeds' first `nextInt(2)` draws agree, so walk until both
    // formats have been drawn.
    let mut names = Vec::new();
    for seed in (0..1000).map(|s| s * 7919) {
        if names.len() == 2 {
            break;
        }
        let format = get_postings_format_with_ords(&mut JavaRandom::new(seed));
        if names.contains(&format.name()) {
            continue;
        }
        names.push(format.name());
        let ext = if format.name() == "LuceneFixedGap" {
            "tib"
        } else {
            "tio"
        };
        let codec = always_postings_format(format);
        assert_eq!(codec.name(), "Asserting");
        check(
            codec,
            &[
                &format!("_{}_0.{ext}", format.name()),
                "_Asserting_0.dvd",
                "_Asserting_0.vec",
            ],
            &["_Asserting_0.tim"],
        );
    }
    names.sort_unstable();
    assert_eq!(names, ["BlockTreeOrds", "LuceneFixedGap"]);
    let sized = get_default_postings_format();
    check(
        always_postings_format(sized),
        &["_Lucene104_0.tim"],
        &["_Asserting_0.tim"],
    );
}

#[test]
fn always_doc_values_format_routes_every_field() {
    let codec = always_doc_values_format(Lucene90DocValuesFormat::new(1024).unwrap());
    check(
        codec,
        &["_Lucene90_0.dvd", "_Asserting_0.tim", "_Asserting_0.vec"],
        &["_Asserting_0.dvd"],
    );
}

#[test]
fn always_knn_vectors_format_routes_every_field() {
    let codec = always_knn_vectors_format(KnnVectorsFormat::hnsw(8, 40).unwrap());
    check(
        codec,
        &[
            "_Lucene99HnswVectorsFormat_0.vec",
            "_Asserting_0.tim",
            "_Asserting_0.dvd",
        ],
        &["_Asserting_0.vec"],
    );
}
