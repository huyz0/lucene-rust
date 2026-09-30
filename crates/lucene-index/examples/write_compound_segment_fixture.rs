//! Writes two indexes with compound segments for `VerifyCompoundSegments`:
//! `<out>/flushed` holds three segments flushed with
//! `IndexWriter::set_use_compound_file(true)` (`DocumentsWriterPerThread.
//! sealFlushedSegment` -> `createCompoundFile`), and `<out>/merged` the same
//! three merged into one by a merge policy whose `useCompoundFile` says yes
//! (`noCFSRatio` 1.0), the merged segment packed as `IndexWriter.mergeMiddle`
//! packs it. Every document carries a stored id, an indexed body with
//! positions, and a numeric doc value, so the compound archive holds stored
//! fields, postings, norms and doc values alike.
//!
//! Usage: `write_compound_segment_fixture <output-dir>`.
// Test-support code opts out of the arithmetic gate at the file boundary:
// see `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use std::sync::Arc;

use lucene_codecs::field_infos::{
    DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions, VectorEncoding,
    VectorSimilarityFunction,
};
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::merge_policy::{MergePolicy, TieredMergePolicy};
use lucene_index::segment_info::LuceneVersion;
use lucene_store::FsDirectory;

/// Documents per flushed segment; `VerifyCompoundSegments` recomputes them.
const PER_SEGMENT: usize = 400;
const SEGMENTS: usize = 3;

fn field(name: &str, number: i32) -> FieldInfo {
    FieldInfo {
        name: name.to_string(),
        number,
        store_term_vectors: false,
        omit_norms: false,
        store_payloads: false,
        soft_deletes_field: false,
        parent_field: false,
        index_options: IndexOptions::None,
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
        field("id", 0),
        FieldInfo {
            index_options: IndexOptions::DocsAndFreqsAndPositions,
            ..field("body", 1)
        },
        FieldInfo {
            doc_values_type: DocValuesType::Numeric,
            ..field("num", 2)
        },
    ];
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut writer = IndexWriter::open(dir, fields, "Lucene104", version).expect("open writer");
    writer.set_use_compound_file(true);
    writer.set_postings_field(Some("body")).expect("postings");
    writer
        .set_doc_values_field(Some("num"))
        .expect("doc values");
    for i in 0..PER_SEGMENT * SEGMENTS {
        let body = format!("shared a{} b{}", i % 13, i % 101);
        writer
            .add_document(Document {
                fields: vec![
                    StoredField {
                        field_number: 0,
                        value: FieldValue::String(format!("doc{i}")),
                    },
                    StoredField {
                        field_number: 1,
                        value: FieldValue::String(body),
                    },
                    StoredField {
                        field_number: 2,
                        value: FieldValue::Long(i as i64 * 7 - 500),
                    },
                ],
            })
            .expect("add");
        if (i + 1) % PER_SEGMENT == 0 {
            writer.commit().expect("commit");
        }
    }
    if merge {
        let mut policy = TieredMergePolicy::default();
        policy
            .compound_file_settings_mut()
            .set_no_cfs_ratio(1.0)
            .expect("ratio");
        writer.set_pluggable_merge_policy(Some(Arc::new(policy)));
        writer.force_merge(1).expect("force merge");
        writer.commit().expect("commit");
    }
}

fn main() {
    let out = std::env::args()
        .nth(1)
        .expect("usage: write_compound_segment_fixture <output-dir>");
    for (sub, merge) in [("flushed", false), ("merged", true)] {
        let path = format!("{out}/{sub}");
        std::fs::create_dir_all(&path).expect("create output dir");
        write(&FsDirectory::open(&path), merge);
    }
    println!("wrote compound indexes to {out}");
}
