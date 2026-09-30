//! Writes indexes whose segments are **compound** (`.cfs`/`.cfe`,
//! `IndexWriterConfig.setUseCompoundFile(true)`) for
//! `fixtures/src/VerifyCompoundSegment.java` to read with real Lucene.
//!
//! - `<out>/flushed`: 1 000 documents flushed as three compound segments,
//!   then every tenth document deleted (the `.liv` generation stays outside
//!   the compound file, as Java's does).
//! - `<out>/merged`: the same, force-merged into one segment under a
//!   `TieredMergePolicy` with `noCFSRatio = 1.0`, so the merged segment is
//!   packed by the merge's own `useCompoundFile` decision.
//!
//! Every format this writer produces goes into the compound file: stored
//! fields, postings with positions, norms, term vectors, all five doc-values
//! types, points and HNSW vectors. Document `i` (`id = "doc{i}"`) has:
//!
//! | field | value |
//! |---|---|
//! | `body` (text, term vectors) | `"w{i%13} w{i%7} common"` |
//! | `num` (NUMERIC) | `3i`, when `i % 4 != 0` |
//! | `cat` (SORTED) | `"c{i%7}"` |
//! | `tags` (SORTED_SET) | `"t{i%5}"`, `"t{i%3}"` |
//! | `blob` (BINARY) | `"b{i}"` |
//! | `pt` (1-D long point) | `i` |
//! | `v` (float vector, dim 4) | `[i, i%10, 1, 0]` |
//!
//! `VerifyCompoundSegment.java` has the same table.
// Example code: see `docs/arithmetic-gate.md`'s "Test code" section.
#![allow(clippy::arithmetic_side_effects)]

use std::sync::Arc;

use lucene_codecs::field_infos::{
    DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions, VectorEncoding,
    VectorSimilarityFunction,
};
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
use lucene_index::buffered_updates::Term;
use lucene_index::index_writer::{DocumentVector, IndexWriter};
use lucene_index::merge_policy::{MergePolicy, MergePolicyConfig, TieredMergePolicy};
use lucene_index::segment_info::LuceneVersion;
use lucene_store::FsDirectory;

/// Must match `VerifyCompoundSegment.java`.
const NUM_DOCS: usize = 1_000;
const DOCS_PER_SEGMENT: i32 = 400;

fn fields() -> Vec<FieldInfo> {
    let dv = |name: &str, number, t| {
        FieldInfo::new(name, number)
            .with_omit_norms(true)
            .with_doc_values(t, DocValuesSkipIndexType::None, -1)
    };
    vec![
        FieldInfo::new("id", 0)
            .with_index_options(IndexOptions::Docs)
            .with_omit_norms(true),
        FieldInfo::new("body", 1)
            .with_index_options(IndexOptions::DocsAndFreqsAndPositions)
            .with_store_term_vectors(true),
        dv("num", 2, DocValuesType::Numeric),
        dv("cat", 3, DocValuesType::Sorted),
        dv("tags", 4, DocValuesType::SortedSet),
        dv("blob", 5, DocValuesType::Binary),
        FieldInfo {
            point_dimension_count: 1,
            point_index_dimension_count: 1,
            point_num_bytes: 8,
            ..FieldInfo::new("pt", 6).with_omit_norms(true)
        },
        FieldInfo {
            vector_dimension: 4,
            vector_encoding: VectorEncoding::Float32,
            vector_similarity_function: VectorSimilarityFunction::Euclidean,
            ..FieldInfo::new("v", 7).with_omit_norms(true)
        },
    ]
}

fn document(i: usize) -> Document {
    let mut fields = Vec::new();
    let mut add = |field_number, value| {
        fields.push(StoredField {
            field_number,
            value,
        })
    };
    add(0, FieldValue::String(format!("doc{i}")));
    add(
        1,
        FieldValue::String(format!("w{} w{} common", i % 13, i % 7)),
    );
    if !i.is_multiple_of(4) {
        add(2, FieldValue::Long(3 * i as i64));
    }
    add(3, FieldValue::String(format!("c{}", i % 7)));
    add(4, FieldValue::String(format!("t{}", i % 5)));
    add(4, FieldValue::String(format!("t{}", i % 3)));
    add(5, FieldValue::Binary(format!("b{i}").into_bytes()));
    add(6, FieldValue::Long(i as i64));
    Document { fields }
}

fn write(dir_path: &str, merge: bool) -> usize {
    std::fs::create_dir_all(dir_path).expect("create dir");
    let dir = FsDirectory::open(dir_path);
    let mut writer = IndexWriter::open(
        &dir,
        fields(),
        "Lucene104",
        LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        },
    )
    .expect("open writer");
    writer.set_use_compound_file(true);
    writer
        .set_max_buffered_docs(DOCS_PER_SEGMENT)
        .expect("max buffered docs");
    writer.set_ram_buffer_size_mb(4096.0).expect("ram buffer");
    writer.set_postings_field(Some("id")).expect("id");
    writer.add_postings_field("body").expect("body");
    writer.set_term_vector_field(Some("body")).expect("tv");
    writer.set_doc_values_field(Some("num")).expect("num");
    for f in ["cat", "tags", "blob"] {
        writer.add_doc_values_field(f).expect(f);
    }
    writer.add_points_field("pt").expect("pt");
    writer.set_vector_field(Some("v")).expect("v");
    for i in 0..NUM_DOCS {
        let vector = DocumentVector::float32("v", vec![i as f32, (i % 10) as f32, 1.0, 0.0]);
        writer
            .add_document_with_vectors(document(i), vec![vector])
            .expect("add document");
    }
    writer.commit().expect("commit");
    let deleted: Vec<Term> = (0..NUM_DOCS)
        .filter(|i| i.is_multiple_of(10))
        .map(|i| Term::new("id", format!("doc{i}")))
        .collect();
    writer.delete_documents_by_term(&deleted).expect("delete");
    writer.commit().expect("commit deletes");
    if merge {
        let mut policy = TieredMergePolicy::new(MergePolicyConfig::default());
        policy
            .compound_file_settings_mut()
            .set_no_cfs_ratio(1.0)
            .expect("noCFSRatio");
        writer.set_pluggable_merge_policy(Some(Arc::new(policy)));
        writer.force_merge(1).expect("force merge");
        writer.commit().expect("commit");
    }
    for sci in &writer.segment_infos().segments {
        let si = lucene_index::segment_info::parse(
            &std::fs::read(format!("{dir_path}/{}.si", sci.segment_name)).expect("si"),
            &sci.segment_id,
        )
        .expect("parse si");
        assert!(si.is_compound_file, "{} is not compound", sci.segment_name);
    }
    writer.segment_infos().segments.len()
}

fn main() {
    let out = std::env::args()
        .nth(1)
        .expect("usage: write_compound_segment_fixture <output-dir>");
    assert_eq!(write(&format!("{out}/flushed"), false), 3);
    assert_eq!(write(&format!("{out}/merged"), true), 1);
    println!("wrote {NUM_DOCS} documents in compound segments, flushed and merged");
}
