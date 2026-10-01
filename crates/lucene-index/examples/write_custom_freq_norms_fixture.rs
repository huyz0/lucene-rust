//! Writes a `DOCS_AND_CUSTOM_FREQS` field that keeps its norms for
//! `VerifyCustomFreqNorms`, through
//! [`IndexWriter::add_document_with_custom_freq_terms`].
//!
//! Document `i` carries [`terms`]`(i)`: up to five distinct terms, each with
//! its own frequency, and none for every seventh document. The verifier
//! indexes the same pairs through Lucene's own `IndexWriter` and compares
//! every norm and every posting.
//!
//! `<out>/flushed` holds one flushed segment; `<out>/merged` three, force-
//! merged into one.
//!
//! Usage: `write_custom_freq_norms_fixture <output-dir>`.
// Test-support code opts out of the arithmetic gate at the file boundary:
// every value is a small constant-derived index.
#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::field_infos::{
    DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions, VectorEncoding,
    VectorSimilarityFunction,
};
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_store::FsDirectory;

/// Documents per commit; `VerifyCustomFreqNorms.DOCS` must agree.
const DOCS: usize = 120;

/// Document `i`'s `(term, freq)` pairs, as `VerifyCustomFreqNorms.terms`.
fn terms(i: usize) -> Vec<(String, i32)> {
    if i.is_multiple_of(7) {
        return Vec::new();
    }
    // `3k mod 13` is distinct for `k < 13`, so a document never repeats a
    // term (Lucene's `DuplicateTermException`).
    (0..1 + i % 5)
        .map(|k| {
            (
                format!("t{}", (i + 3 * k) % 13),
                ((i * 31 + k * 17) % 50 + 1) as i32,
            )
        })
        .collect()
}

fn field(number: i32, name: &str, index_options: IndexOptions) -> FieldInfo {
    FieldInfo {
        name: name.to_string(),
        number,
        store_term_vectors: false,
        omit_norms: false,
        store_payloads: false,
        soft_deletes_field: false,
        parent_field: false,
        index_options,
        doc_values_type: DocValuesType::None,
        doc_values_skip_index_type: DocValuesSkipIndexType::None,
        doc_values_gen: -1,
        attributes: Vec::new(),
        point_dimension_count: 0,
        point_index_dimension_count: 0,
        point_num_bytes: 0,
        vector_dimension: 0,
        vector_encoding: VectorEncoding::Float32,
        vector_similarity_function: VectorSimilarityFunction::Euclidean,
    }
}

fn write(dir: &FsDirectory, commits: usize) {
    let fields = vec![
        field(0, "id", IndexOptions::None),
        field(1, "score", IndexOptions::DocsAndCustomFreqs),
    ];
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut w = IndexWriter::open(dir, fields, "Lucene104", version).unwrap();
    w.set_custom_freq_postings_field(Some("score")).unwrap();
    for c in 0..commits {
        for i in c * DOCS..(c + 1) * DOCS {
            let doc = Document {
                fields: vec![StoredField {
                    field_number: 0,
                    value: FieldValue::String(i.to_string()),
                }],
            };
            w.add_document_with_custom_freq_terms(doc, terms(i))
                .unwrap();
        }
        w.commit().unwrap();
    }
    if commits > 1 {
        w.force_merge(1).unwrap();
        w.commit().unwrap();
    }
}

fn main() {
    let out = std::env::args()
        .nth(1)
        .expect("usage: write_custom_freq_norms_fixture <output-dir>");
    for (sub, commits) in [("flushed", 1), ("merged", 3)] {
        let path = std::path::Path::new(&out).join(sub);
        std::fs::create_dir_all(&path).unwrap();
        write(&FsDirectory::open(&path), commits);
    }
}
