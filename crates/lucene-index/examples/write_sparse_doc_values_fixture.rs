//! Writes two indexes whose doc values are **sparse in every type** --
//! NUMERIC, BINARY, SORTED, SORTED_NUMERIC and SORTED_SET, all five in one
//! multi-field `.dvm`/`.dvd` -- for `fixtures/src/VerifySparseDocValues.java`
//! to read with real Lucene.
//!
//! - `<out>/flushed`: 3 000 documents in three flushed segments. Every field
//!   is missing on a different stride of documents, so each segment's column
//!   is written through its sparse (`IndexedDISI`) variant.
//! - `<out>/merged`: the same documents, merged into one segment. The merge
//!   must carry every missing value across as missing -- the sparse merge
//!   path, which for the four non-numeric types did not exist before M4.
//!
//! Document `i` (in insertion order, stored as `id = "doc{i}"`) has:
//!
//! | field | present when | value |
//! |---|---|---|
//! | `num` (NUMERIC) | `i % 3 != 0` | `7i - 1000` |
//! | `bin` (BINARY) | `i % 5 != 0` | `"b{i}"` |
//! | `sorted` (SORTED) | `i % 7 != 0` | `"s{i % 50}"` |
//! | `snum` (SORTED_NUMERIC) | `i % 11 != 0` | `{i % 13, i, -i}` |
//! | `sset` (SORTED_SET) | `i % 13 != 0` | `{"t{i % 17}", "t{i % 19}"}` |
//!
//! `VerifySparseDocValues.java` has the same table.
// Example code: see `docs/arithmetic-gate.md`'s "Test code" section.
#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::field_infos::{
    DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions, VectorEncoding,
    VectorSimilarityFunction,
};
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::merge_policy::MergePolicyConfig;
use lucene_index::segment_info::LuceneVersion;
use lucene_store::FsDirectory;

/// Must match `VerifySparseDocValues.java`.
const NUM_DOCS: usize = 3_000;
const DOCS_PER_SEGMENT: i32 = 1_000;

fn field(name: &str, number: i32, dv: DocValuesType) -> FieldInfo {
    FieldInfo {
        name: name.to_string(),
        number,
        store_term_vectors: false,
        omit_norms: true,
        store_payloads: false,
        soft_deletes_field: false,
        parent_field: false,
        index_options: IndexOptions::None,
        doc_values_type: dv,
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

fn document(i: usize) -> Document {
    let mut fields = vec![StoredField {
        field_number: 0,
        value: FieldValue::String(format!("doc{i}")),
    }];
    let mut add = |field_number, value| {
        fields.push(StoredField {
            field_number,
            value,
        })
    };
    if !i.is_multiple_of(3) {
        add(1, FieldValue::Long(7 * i as i64 - 1000));
    }
    if !i.is_multiple_of(5) {
        add(2, FieldValue::Binary(format!("b{i}").into_bytes()));
    }
    if !i.is_multiple_of(7) {
        add(3, FieldValue::String(format!("s{}", i % 50)));
    }
    if !i.is_multiple_of(11) {
        for v in [(i % 13) as i64, i as i64, -(i as i64)] {
            add(4, FieldValue::Long(v));
        }
    }
    if !i.is_multiple_of(13) {
        for v in [format!("t{}", i % 17), format!("t{}", i % 19)] {
            add(5, FieldValue::String(v));
        }
    }
    Document { fields }
}

fn write(dir_path: &str, merge: bool) -> usize {
    std::fs::create_dir_all(dir_path).expect("create dir");
    let dir = FsDirectory::open(dir_path);
    let fields = vec![
        field("id", 0, DocValuesType::None),
        field("num", 1, DocValuesType::Numeric),
        field("bin", 2, DocValuesType::Binary),
        field("sorted", 3, DocValuesType::Sorted),
        field("snum", 4, DocValuesType::SortedNumeric),
        field("sset", 5, DocValuesType::SortedSet),
    ];
    let mut writer = IndexWriter::open(
        &dir,
        fields,
        "Lucene104",
        LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        },
    )
    .expect("open writer");
    writer
        .set_max_buffered_docs(DOCS_PER_SEGMENT)
        .expect("max buffered docs");
    writer.set_doc_values_field(Some("num")).expect("num");
    for name in ["bin", "sorted", "snum", "sset"] {
        writer.add_doc_values_field(name).expect(name);
    }
    for i in 0..NUM_DOCS {
        writer.add_document(document(i)).expect("add document");
    }
    writer.commit().expect("commit");
    if merge {
        writer.set_merge_policy(Some(MergePolicyConfig {
            max_merge_at_once: 10,
            segments_per_tier: 2,
            max_merged_segment_size: u64::MAX / 4,
            floor_segment_size: 1 << 30,
            ..MergePolicyConfig::default()
        }));
        writer.commit().expect("commit triggers the merge");
    }
    writer.segment_infos().segments.len()
}

fn main() {
    let out = std::env::args()
        .nth(1)
        .expect("usage: write_sparse_doc_values_fixture <output-dir>");
    let flushed = write(&format!("{out}/flushed"), false);
    assert!(
        flushed >= 3,
        "expected several flushed segments, got {flushed}"
    );
    let merged = write(&format!("{out}/merged"), true);
    assert_eq!(merged, 1, "expected the segments to merge into one");
    println!("wrote {NUM_DOCS} sparse-doc-values documents: {flushed} segments and 1 merged");
}
