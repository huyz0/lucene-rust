//! Writes an index, then **merges** it, and leaves the result for real Lucene
//! to open -- the reverse-direction check for the merge path's two fast
//! stored-fields strategies.
//!
//! `write_full_segment_fixture` proves a *flushed* segment is readable. It
//! cannot see the merge, and the merge is where this port stopped writing its
//! own bytes: `Lucene90CompressingStoredFieldsWriter.merge`'s BULK path copies
//! a source segment's already-compressed chunks straight through, rewriting
//! only each chunk's `docBase` vint and the `.fdx` index entry. A wrong chunk
//! boundary there produces a segment that still opens and still reads back
//! plausible documents -- just the wrong ones -- so round-tripping through
//! this port's own reader is exactly the check that cannot catch it.
//!
//! The index is built so both fast paths are exercised in one merge:
//!
//! - segments `_0` and `_2` have no deletions and identical field numbering,
//!   so they take **BULK** (whole compressed chunks copied verbatim);
//! - segment `_1` has real `.liv` deletions, which rules out chunk copying, so
//!   it takes **DOC** (each surviving document's serialized bytes copied
//!   without being parsed into fields).
//!
//! The `body` field also stores **term vectors**, whose merge has the same
//! two-way shape (`Lucene90CompressingTermVectorsWriter.merge`'s `copyChunks`
//! versus `addAllDocVectors`) over a different chunk format -- 4 096 bytes /
//! 128 documents rather than stored fields' 80 kB / 1 024, so each segment
//! holds ~19 term-vector chunks and the copy loop runs over many more
//! boundaries than the stored-fields one does. `CheckIndex` walks every
//! document's vectors, so a copied chunk landing at the wrong `docBase` is
//! caught here and nowhere in-port.
//!
//! Each segment is large enough to hold several full chunks plus a trailing
//! dirty one, so the copy loop runs over real chunk boundaries rather than a
//! single degenerate chunk.
//!
//! Usage: `write_merged_segment_fixture <output-dir>`.
// Test-support code opts out of the arithmetic gate at the file boundary:
// the gate exists for values read off disk in production decode paths, not
// for a fixture builder's own index arithmetic. See
// `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::blocktree;
use lucene_codecs::field_infos::{
    DocValuesSkipIndexType, DocValuesType, FieldInfo, FieldInfos, IndexOptions, VectorEncoding,
    VectorSimilarityFunction,
};
use lucene_codecs::postings::DocInput;
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
use lucene_index::index_writer::{
    per_field_codec_suffix, per_field_segment, DeleteSource, IndexWriter, POSTINGS_FORMAT_NAME,
};
use lucene_index::merge_policy::MergePolicyConfig;
use lucene_index::segment_info::LuceneVersion;
use lucene_store::{Directory, FsDirectory};

/// Well past the 1024-document stored-fields chunk cap, so each flushed
/// segment holds two full chunks plus a trailing dirty one.
const DOCS_PER_SEGMENT: usize = 2_400;
const NUM_SEGMENTS: usize = 3;
/// Every document whose ordinal within its segment is a multiple of this gets
/// the extra term `doomed` in its body; deleting that term is what gives
/// segment `_1` its deletions.
const DOOMED_EVERY: usize = 100;

fn field(name: &str, number: i32, indexed: bool) -> FieldInfo {
    FieldInfo {
        name: name.to_string(),
        number,
        store_term_vectors: indexed,
        // True on the indexed field: this port's automatic merge does not open
        // a source's norms, so a merged `.fnm` promising them would describe
        // files the merged segment does not carry.
        omit_norms: true,
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

/// The body text for the `n`th document of segment `seg` -- deterministic, and
/// the Java verifier recomputes it independently.
fn body(seg: usize, n: usize, vocab: &[String]) -> String {
    let mut text = format!(
        "shared {} {}",
        vocab[n % vocab.len()],
        vocab[(n / 7) % vocab.len()]
    );
    if n.is_multiple_of(DOOMED_EVERY) {
        text.push_str(" doomed");
    }
    let _ = seg;
    text
}

fn main() {
    let out_dir = std::env::args()
        .nth(1)
        .expect("usage: write_merged_segment_fixture <output-dir>");
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
        .set_term_vector_field(Some("body"))
        .expect("set term-vector field");

    let vocab: Vec<String> = (0..500)
        .map(|i| format!("{}{i:03}", (b'a' + (i % 26) as u8) as char))
        .collect();

    for seg in 0..NUM_SEGMENTS {
        for n in 0..DOCS_PER_SEGMENT {
            writer
                .add_document(Document {
                    fields: vec![
                        StoredField {
                            field_number: 0,
                            value: FieldValue::String(format!("doc{seg}-{n}")),
                        },
                        StoredField {
                            field_number: 1,
                            value: FieldValue::String(body(seg, n, &vocab)),
                        },
                    ],
                })
                .unwrap();
        }
        // One commit per segment, with no merge policy set, so exactly
        // NUM_SEGMENTS segments exist when the merge is triggered below.
        writer.commit().expect("commit");
    }

    // Deletions on segment _1 only: term `body:doomed`, resolved through that
    // segment's own real postings. That is what puts _1 on the DOC path while
    // _0 and _2 stay on BULK.
    let seg_name = "_1";
    let postings_seg = per_field_segment(seg_name, POSTINGS_FORMAT_NAME);
    let tim = dir.open(&format!("{postings_seg}.tim")).unwrap().to_vec();
    let tip = dir.open(&format!("{postings_seg}.tip")).unwrap().to_vec();
    let tmd = dir.open(&format!("{postings_seg}.tmd")).unwrap().to_vec();
    let doc_bytes = dir.open(&format!("{postings_seg}.doc")).unwrap().to_vec();
    let segment_id = writer
        .segment_infos()
        .segments
        .iter()
        .find(|s| s.segment_name == seg_name)
        .expect("segment _1 committed")
        .segment_id;
    let suffix = per_field_codec_suffix(POSTINGS_FORMAT_NAME);
    let postings_field_infos = FieldInfos {
        fields: vec![field("body", 1, true)],
    };
    let bt = blocktree::open(
        &tim,
        &tip,
        &tmd,
        &postings_field_infos,
        &segment_id,
        &suffix,
        DOCS_PER_SEGMENT as i32,
    )
    .expect("open term dictionary");
    let doc_in = DocInput::open(&doc_bytes, &segment_id, &suffix).expect("open .doc");
    let sources = [DeleteSource {
        segment_name: seg_name,
        fields: &bt,
        doc_in: Some(&doc_in),
        live_docs: None,
        max_doc: DOCS_PER_SEGMENT,
    }];
    let del_count = writer
        .delete_documents_with_sources(&sources, "body", b"doomed")
        .expect("delete doomed documents")
        .segments
        .iter()
        .find(|s| s.segment_name == seg_name)
        .expect("segment _1 still committed")
        .del_count;
    assert_eq!(del_count as usize, DOCS_PER_SEGMENT.div_ceil(DOOMED_EVERY));

    // Now merge. `floor_segment_size`/`max_merged_segment_size` are set well
    // above/below these segments so the tiered policy proposes exactly one
    // merge of all three.
    writer.set_merge_policy(Some(MergePolicyConfig {
        max_merge_at_once: 10,
        segments_per_tier: 2,
        max_merged_segment_size: u64::MAX / 4,
        floor_segment_size: 1 << 30,
        ..MergePolicyConfig::default()
    }));
    writer.commit().expect("commit triggers the merge");

    let segments = &writer.segment_infos().segments;
    assert_eq!(
        segments.len(),
        1,
        "expected the three segments to merge into one, got {}",
        segments.len()
    );

    println!(
        "merged {NUM_SEGMENTS} segments ({} docs, {del_count} deleted) into {} in {out_dir}",
        NUM_SEGMENTS * DOCS_PER_SEGMENT,
        segments[0].segment_name
    );
}
