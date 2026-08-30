//! Writes an **index-sorted** index through the real write path
//! (`IndexWriter::set_index_sort` + `add_document_with_vectors` + `commit`)
//! for `fixtures/src/VerifySortedSegment.java` to open with a real
//! `DirectoryReader`.
//!
//! An index sort is the one property of a segment that is *entirely* an
//! invariant between files: every file is well-formed, every checksum is
//! valid, and every doc id is in range no matter which permutation the writer
//! applied -- or whether it applied one to some formats and not to others.
//! The only things that catch it are (a) `LeafMetaData.sort()` versus the
//! physical order, which is `CheckIndex.testSort`, and (b) reading each
//! document's *own* stored value, doc-values value, postings term, norm and
//! vector back and checking they still describe the same document.
//!
//! So the fixture gives every document a body term, a norm-distinguishing
//! length, two doc-values columns and a vector that all encode its identity,
//! and the sort is deliberately the awkward one: a **reversed** first tier
//! whose missing documents therefore belong at the *front*, and a second tier
//! that breaks its many ties.
//!
//! Usage: `write_sorted_segment_fixture <output-dir>`.
// Test-support code opts out of the arithmetic gate at the file boundary:
// the gate exists for values read off disk in production decode paths, not
// for a fixture builder's own index arithmetic. See
// `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::field_infos::{
    DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions, VectorEncoding,
    VectorSimilarityFunction,
};
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
use lucene_index::index_writer::{DocumentVector, IndexWriter};
use lucene_index::segment_info::{IndexSortField, LuceneVersion, SortMissingValue};
use lucene_store::FsDirectory;

/// Past a stored-fields chunk boundary and many postings blocks, and past
/// `HNSW_GRAPH_THRESHOLD` so the vector field builds a real graph over the
/// *sorted* ordinal space.
const NUM_DOCS: usize = 2_000;
/// Every `MISSING_EVERY`-th document has no `rank`, so the reversed
/// missing-last tier has to put a whole block of documents at the front.
const MISSING_EVERY: usize = 37;
const DIM: usize = 8;

fn base(name: &str, number: i32) -> FieldInfo {
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
        vector_encoding: VectorEncoding::Float32,
        vector_similarity_function: VectorSimilarityFunction::Euclidean,
    }
}

/// `rank`, or `None` for the documents that deliberately have none.
pub fn rank_of(i: usize) -> Option<i64> {
    if i.is_multiple_of(MISSING_EVERY) {
        None
    } else {
        // Many duplicates, so the second tier decides most comparisons.
        Some(((i * 7919) % 50) as i64 - 20)
    }
}

/// `tie`: unique per document, so the whole order is total and the verifier
/// can predict it exactly.
pub fn tie_of(i: usize) -> i64 {
    ((i * 104_729) % NUM_DOCS) as i64
}

/// A deterministic vector whose first component is the document's ordinal, so
/// a vector that drifted onto another document is visible without any
/// similarity reasoning.
pub fn vector_of(i: usize) -> Vec<f32> {
    let mut v = vec![i as f32; DIM];
    for (k, slot) in v.iter_mut().enumerate().skip(1) {
        *slot = ((i * 31 + k * 17) % 1000) as f32 / 1000.0;
    }
    v
}

fn main() {
    let out_dir = std::env::args()
        .nth(1)
        .expect("usage: write_sorted_segment_fixture <output-dir>");
    std::fs::create_dir_all(&out_dir).expect("create output dir");

    let dir = FsDirectory::open(&out_dir);
    let fields = vec![
        base("id", 0),
        FieldInfo {
            doc_values_type: DocValuesType::Numeric,
            ..base("rank", 1)
        },
        FieldInfo {
            doc_values_type: DocValuesType::Numeric,
            ..base("tie", 2)
        },
        FieldInfo {
            index_options: IndexOptions::DocsAndFreqs,
            ..base("body", 3)
        },
        FieldInfo {
            vector_dimension: DIM as i32,
            ..base("v", 4)
        },
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
        .set_postings_field(Some("body"))
        .expect("set postings field");
    writer
        .set_norms_field(Some("body"))
        .expect("set norms field");
    writer
        .set_doc_values_field(Some("rank"))
        .expect("set doc values field");
    writer
        .add_doc_values_field("tie")
        .expect("add second doc values field");
    writer
        .set_vector_field(Some("v"))
        .expect("set vector field");
    // `rank` descending with missing last -- i.e. the missing documents take
    // `Long.MAX_VALUE` and, reversed, come first. `tie` ascending breaks the
    // ties.
    writer
        .set_index_sort(Some(&[
            IndexSortField {
                field: "rank".to_string(),
                reverse: true,
                missing: SortMissingValue::Last,
            },
            IndexSortField {
                field: "tie".to_string(),
                reverse: false,
                missing: SortMissingValue::First,
            },
        ]))
        .expect("set index sort");

    for i in 0..NUM_DOCS {
        let mut fields = vec![
            StoredField {
                field_number: 0,
                value: FieldValue::String(format!("doc{i}")),
            },
            StoredField {
                field_number: 2,
                value: FieldValue::Long(tie_of(i)),
            },
            StoredField {
                field_number: 3,
                // A term unique to this document, plus a shared one, plus a
                // repeat count that makes the field length -- and therefore
                // the norm -- a function of the document.
                value: FieldValue::String(format!("shared u{i}{}", " pad".repeat(i % 5))),
            },
        ];
        if let Some(rank) = rank_of(i) {
            fields.insert(
                1,
                StoredField {
                    field_number: 1,
                    value: FieldValue::Long(rank),
                },
            );
        }
        writer
            .add_document_with_vectors(
                Document { fields },
                vec![DocumentVector::float32("v", vector_of(i))],
            )
            .expect("add document");
    }
    writer.commit().expect("commit");

    println!("wrote a {NUM_DOCS}-document index-sorted index to {out_dir}");
}
