//! Writes three indexes whose documents carry **points**, through the real
//! `IndexWriter`, for `fixtures/src/VerifyPointsSegment.java` to read with
//! real Lucene (M4's T4.2):
//!
//! - `<out>/flushed`: 20 000 documents in three flushed segments;
//! - `<out>/merged`: the same documents merged into one segment;
//! - `<out>/sorted`: the same documents under an index sort on `rank`
//!   (a permutation of the document numbers), flushed in three sorted
//!   segments and then merged -- so every point goes through the sorted
//!   merge's doc-id remap, the one path that remap had no test for.
//!
//! Document `i` (stored as `id = "doc{i}"`) has:
//!
//! | field | shape | value(s) |
//! |---|---|---|
//! | `lp` | `LongPoint`, multi-valued | `7i - 1000`, and `-i` too when `i % 4 == 0` |
//! | `ip` | `IntPoint`, present when `i % 3 != 0` | `i % 1000` |
//! | `dp` | `DoublePoint` | `i / 8.0 - 100.0` |
//! | `xy` | two `int` dimensions, packed | `(i % 97, i % 89 - 44)` |
//! | `rank` | NUMERIC doc values (the sort key) | `(i * 7919) % 20000` |
//! | `none` | `LongPoint`, declared but on no document | -- |
//!
//! `VerifyPointsSegment.java` has the same table.
// Example code: see `docs/arithmetic-gate.md`'s "Test code" section.
#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::field_infos::{
    DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions, VectorEncoding,
    VectorSimilarityFunction,
};
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::merge_policy::MergePolicyConfig;
use lucene_index::segment_info::{IndexSortField, LuceneVersion};
use lucene_store::FsDirectory;

/// Must match `VerifyPointsSegment.java`.
const NUM_DOCS: usize = 20_000;
const DOCS_PER_SEGMENT: i32 = 7_000;

fn field(name: &str, number: i32) -> FieldInfo {
    FieldInfo {
        name: name.to_string(),
        number,
        store_term_vectors: false,
        omit_norms: true,
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

fn point_field(name: &str, number: i32, dims: i32, bytes: i32) -> FieldInfo {
    FieldInfo {
        point_dimension_count: dims,
        point_index_dimension_count: dims,
        point_num_bytes: bytes,
        ..field(name, number)
    }
}

/// `IntPoint.pack`: each dimension `intToSortableBytes`.
fn pack_ints(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&v| ((v as u32) ^ 0x8000_0000).to_be_bytes())
        .collect()
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
    add(1, FieldValue::Long(7 * i as i64 - 1000));
    if i.is_multiple_of(4) {
        add(1, FieldValue::Long(-(i as i64)));
    }
    if !i.is_multiple_of(3) {
        add(2, FieldValue::Int((i % 1000) as i32));
    }
    add(3, FieldValue::Double(i as f64 / 8.0 - 100.0));
    add(
        4,
        FieldValue::Binary(pack_ints(&[(i % 97) as i32, (i % 89) as i32 - 44])),
    );
    add(5, FieldValue::Long(((i * 7919) % NUM_DOCS) as i64));
    Document { fields }
}

fn write(dir_path: &str, merge: bool, sorted: bool) -> usize {
    std::fs::create_dir_all(dir_path).expect("create dir");
    let dir = FsDirectory::open(dir_path);
    let fields = vec![
        field("id", 0),
        point_field("lp", 1, 1, 8),
        point_field("ip", 2, 1, 4),
        point_field("dp", 3, 1, 8),
        point_field("xy", 4, 2, 4),
        FieldInfo {
            doc_values_type: DocValuesType::Numeric,
            ..field("rank", 5)
        },
        // Declared and opted in, but no document carries it: every flushed
        // and merged `.fnm` must drop its point shape, since no `.kdm` entry
        // backs it.
        point_field("none", 6, 1, 8),
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
    writer.set_ram_buffer_size_mb(4096.0).expect("ram buffer");
    for name in ["lp", "ip", "dp", "xy", "none"] {
        writer.add_points_field(name).expect(name);
    }
    writer.set_doc_values_field(Some("rank")).expect("rank");
    if sorted {
        writer
            .set_index_sort(Some(&[IndexSortField::long("rank", false, None)]))
            .expect("index sort");
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
        .expect("usage: write_points_segment_fixture <output-dir>");
    let flushed = write(&format!("{out}/flushed"), false, false);
    assert!(flushed >= 3, "expected several segments, got {flushed}");
    assert_eq!(write(&format!("{out}/merged"), true, false), 1);
    assert_eq!(write(&format!("{out}/sorted"), true, true), 1);
    println!("wrote {NUM_DOCS} documents with points: {flushed} segments, merged, sorted-merged");
}
