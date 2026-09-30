//! Writes two indexes whose postings are split across two
//! `Lucene104PostingsFormat` instances, for `VerifyPerFieldFormats`: `key`
//! and `tag` are routed to `Lucene104PostingsFormat(10, 20)` with
//! [`IndexWriter::set_postings_format_for_field`], `body` stays on the
//! default -- so every segment has a `_Lucene104_0` and a `_Lucene104_1` set
//! of postings files, recorded per field in the `.fnm` as
//! `PerFieldPostingsFormat` records them.
//!
//! `<out>/flushed` holds three flushed segments (the last two compound) and
//! a delete by a `tag` term resolved against all three, which reads the
//! routed format's dictionary back; `<out>/merged` the same, force-merged
//! into one segment, the merge reading both formats of every source and
//! writing both again (`PerFieldMergeState`).
//!
//! Usage: `write_per_field_formats_fixture <output-dir>`.
// Test-support code opts out of the arithmetic gate at the file boundary:
// see `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::field_infos::{
    DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions, VectorEncoding,
    VectorSimilarityFunction,
};
use lucene_codecs::per_field_postings::Lucene104PostingsFormat;
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
use lucene_index::buffered_updates::Term;
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_store::FsDirectory;

/// Documents per flushed segment; `VerifyPerFieldFormats` recomputes them.
const PER_SEGMENT: usize = 400;
const SEGMENTS: usize = 3;

fn field(name: &str, number: i32, index_options: IndexOptions) -> FieldInfo {
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
        attributes: vec![],
        point_dimension_count: 0,
        point_index_dimension_count: 0,
        point_num_bytes: 0,
        vector_dimension: 0,
        vector_encoding: VectorEncoding::Byte,
        vector_similarity_function: VectorSimilarityFunction::Euclidean,
    }
}

fn write(dir: &FsDirectory, merge: bool) {
    let fields = vec![
        field("id", 0, IndexOptions::None),
        field("body", 1, IndexOptions::DocsAndFreqsAndPositions),
        field("tag", 2, IndexOptions::Docs),
        field("key", 3, IndexOptions::DocsAndFreqs),
    ];
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut writer = IndexWriter::open(dir, fields, "Lucene104", version).expect("open writer");
    for name in ["body", "tag", "key"] {
        writer.add_postings_field(name).expect("postings");
    }
    let small = Lucene104PostingsFormat::new(10, 20).expect("block sizes");
    writer.set_postings_format_for_field("tag", small);
    writer.set_postings_format_for_field("key", small);
    for i in 0..PER_SEGMENT * SEGMENTS {
        let text = |field_number: i32, s: String| StoredField {
            field_number,
            value: FieldValue::String(s),
        };
        writer
            .add_document(Document {
                fields: vec![
                    text(0, format!("doc{i}")),
                    text(1, format!("shared a{}", i % 13)),
                    text(2, format!("t{}", i % 7)),
                    text(3, format!("k{}", i % 50)),
                ],
            })
            .expect("add");
        if (i + 1) % PER_SEGMENT == 0 {
            if i + 1 == PER_SEGMENT * (SEGMENTS - 1) {
                writer.set_use_compound_file(true);
            }
            writer.commit().expect("commit");
        }
    }
    writer
        .delete_documents_by_term(&[Term {
            field: "tag".to_string(),
            bytes: b"t3".to_vec(),
        }])
        .expect("delete");
    writer.commit().expect("commit deletes");
    if merge {
        writer.force_merge(1).expect("force merge");
        writer.commit().expect("commit merge");
    }
}

fn main() {
    let out = std::env::args()
        .nth(1)
        .expect("usage: write_per_field_formats_fixture <output-dir>");
    for (sub, merge) in [("flushed", false), ("merged", true)] {
        let path = format!("{out}/{sub}");
        std::fs::create_dir_all(&path).expect("create output dir");
        write(&FsDirectory::open(&path), merge);
    }
    println!("wrote per-field postings format indexes to {out}");
}
