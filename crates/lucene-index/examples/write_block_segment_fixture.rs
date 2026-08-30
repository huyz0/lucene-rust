//! Writes an index whose documents were added as **blocks**
//! (`IndexWriter::add_documents`), so real Lucene can confirm the resulting
//! segment's `SegmentInfo.hasBlocks` flag is set and the segment is otherwise
//! intact.
//!
//! `write_full_segment_fixture` already proves a normally-flushed segment
//! opens; this fixture exists because `hasBlocks` is a *single byte* in the
//! `.si` that nothing else in the write path sets, and because that byte is
//! what makes parent/child join queries legal against the segment. A segment
//! that carries blocks but reports `hasBlocks=false` reads back perfectly and
//! is silently wrong -- `LeafMetaData.hasBlocks()` is the only place the
//! difference surfaces, so the verifier asserts on it directly rather than
//! relying on `CheckIndex` alone.
//!
//! It also issues one **buffered delete** before committing, against a final
//! block whose body carries a unique term. That is the only real-Lucene
//! validation of the `.liv` this port's buffered-delete path writes: unlike the
//! doc-values-update overlay (whose format is this port's own invention),
//! `.liv` *is* real Lucene format, and `CheckIndex` cross-checks it against the
//! `delCount` recorded in `segments_N` and the segment's `maxDoc`. A delete
//! whose `.liv` and whose bookkeeping disagree is exactly the kind of defect
//! that reads back fine through this port's own reader.
//!
//! Deliberately **no index sort and no parent field**: Lucene only requires a
//! parent field for a block-carrying segment when an index sort is present
//! (`CheckIndex.testSort`, `IndexWriter.mergeMiddle`'s
//! `hasBlocksButNoParentField`), and this port has no doc-values write path for
//! a parent field yet. The unsorted case is the one this port can produce
//! today, and it is a case real Lucene accepts.
//!
//! Usage: `write_block_segment_fixture <output-dir>`.
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
use lucene_index::buffered_updates::Term;
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_store::FsDirectory;

/// Blocks of one parent plus three children, repeated -- enough to cross a
/// postings block (128 docs) so the block adds are not all inside one
/// degenerate first block.
const NUM_BLOCKS: usize = 300;
const BLOCK_SIZE: usize = 4;
/// One more block, added last and then deleted by term, so real Lucene has a
/// `.liv` to validate. Its bodies deliberately avoid the `parent`/`child`
/// terms so the contiguity check above it is unaffected.
const TOMBSTONE_TERM: &str = "tombstone";
/// Must match `VerifyBlockSegment.java`.
const NUM_DOCS: usize = (NUM_BLOCKS + 1) * BLOCK_SIZE;
const NUM_DELETED: usize = BLOCK_SIZE;

fn field(name: &str, number: i32, indexed: bool) -> FieldInfo {
    FieldInfo {
        name: name.to_string(),
        number,
        store_term_vectors: false,
        omit_norms: false,
        store_payloads: false,
        soft_deletes_field: false,
        parent_field: false,
        index_options: if indexed {
            IndexOptions::DocsAndFreqs
        } else {
            IndexOptions::None
        },
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

fn doc(id: &str, body: &str) -> Document {
    Document {
        fields: vec![
            StoredField {
                field_number: 0,
                value: FieldValue::String(id.to_string()),
            },
            StoredField {
                field_number: 1,
                value: FieldValue::String(body.to_string()),
            },
        ],
    }
}

fn main() {
    let out_dir = std::env::args()
        .nth(1)
        .expect("usage: write_block_segment_fixture <output-dir>");
    std::fs::create_dir_all(&out_dir).expect("create output dir");

    let dir = FsDirectory::open(&out_dir);
    let fields = vec![field("id", 0, false), field("body", 1, true)];
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

    for b in 0..NUM_BLOCKS {
        let mut block = Vec::with_capacity(BLOCK_SIZE);
        block.push(doc(&format!("parent{b}"), "parent shared"));
        for c in 1..BLOCK_SIZE {
            block.push(doc(&format!("child{b}_{c}"), "child shared"));
        }
        // One call, one sequence number, contiguous doc IDs, and the flag.
        writer.add_documents(block).expect("add block");
    }

    // One last block, then a buffered delete of it. The delete is issued while
    // the whole batch is still in the buffer, so it becomes the segment's own
    // private packet and is resolved against the segment this flush writes --
    // the path `docs/sweep/m2/c7-delete-queue.md` F-3 describes.
    let tombstones: Vec<Document> = (0..BLOCK_SIZE)
        .map(|i| doc(&format!("tombstone{i}"), TOMBSTONE_TERM))
        .collect();
    writer
        .add_documents(tombstones)
        .expect("add tombstone block");
    writer
        .delete_documents_by_term(&[Term::new("body", TOMBSTONE_TERM.as_bytes())])
        .expect("buffer delete");

    writer.commit().expect("commit");

    println!("wrote a {NUM_DOCS}-document block index ({NUM_DELETED} deleted) to {out_dir}");
}
