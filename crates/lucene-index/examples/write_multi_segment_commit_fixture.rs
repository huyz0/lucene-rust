//! Writes a real **multi-segment** index -- two independently-flushed
//! segments (`_0`, `_1`), each with its own `.fdt`/`.fdx`/`.fdm`/`.fnm`/`.si`,
//! described together by a single `segments_N` commit -- to the directory
//! given as the first CLI argument.
//!
//! This is the next slice after `write_segment_infos_fixture.rs`'s
//! single-segment commit: it proves the multi-segment shape of a real
//! `IndexWriter.commit()` (several `DocumentsWriterPerThread.flush()` calls
//! folded into one commit) without needing real RAM-triggered flushing,
//! merging, or concurrent buffering -- see `lucene_index::segment_writer`'s
//! module docs for exactly what is and isn't in scope here.
//!
//! `fixtures/src/VerifySegmentInfos.java` (unchanged, already
//! segment-count-agnostic: it only reads `manifest.properties` and calls
//! `DirectoryReader.open` + `StoredFields.document(docId)` across the whole
//! reader) verifies this fixture the same way it verifies the single-segment
//! one -- succeeding here is proof that real Lucene's `DirectoryReader`
//! federates two Rust-written segments into one coherent doc-id space.
//!
//! Run: `cargo run -p lucene-index --example write_multi_segment_commit_fixture -- <dir>`

use lucene_codecs::field_infos::{
    DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions, VectorEncoding,
    VectorSimilarityFunction,
};
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
use lucene_index::segment_info::LuceneVersion;
use lucene_index::segment_infos::{self, LuceneVersion as SisLuceneVersion, SegmentInfos};
use lucene_index::segment_writer::flush_stored_only_segment;
use lucene_store::FsDirectory;
use std::io::Write;

const COMMIT_ID: [u8; 16] = *b"rustwrittenmcmt0";
const CODEC_NAME: &str = "Lucene104";

fn lucene_version() -> LuceneVersion {
    LuceneVersion {
        major: 10,
        minor: 0,
        bugfix: 0,
    }
}

fn sis_lucene_version() -> SisLuceneVersion {
    let v = lucene_version();
    SisLuceneVersion {
        major: v.major,
        minor: v.minor,
        bugfix: v.bugfix,
    }
}

fn stored_only_field(name: &str, number: i32) -> FieldInfo {
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
        .expect("usage: write_multi_segment_commit_fixture <output-dir>");
    std::fs::create_dir_all(&out_dir).unwrap();
    let dir = FsDirectory::open(&out_dir);

    let fields = vec![stored_only_field("id", 0), stored_only_field("body", 1)];

    // Two independent "flushes" -- distinct segment names and segment ids,
    // disjoint document sets, each written by its own
    // `flush_stored_only_segment` call.
    let segment0_docs = vec![
        doc("1", "the quick brown fox"),
        doc("2", "jumps over the lazy dog"),
    ];
    let segment1_docs = vec![
        doc("3", "pack my box with five dozen liquor jugs"),
        doc("4", "how vexingly quick daft zebras jump"),
        doc("5", "sphinx of black quartz judge my vow"),
    ];

    let sci0 = flush_stored_only_segment(
        &dir,
        "_0",
        *b"rustwrittenseg00",
        CODEC_NAME,
        lucene_version(),
        &fields,
        &segment0_docs,
        false,
    )
    .unwrap();
    let sci1 = flush_stored_only_segment(
        &dir,
        "_1",
        *b"rustwrittenseg01",
        CODEC_NAME,
        lucene_version(),
        &fields,
        &segment1_docs,
        false,
    )
    .unwrap();

    // One segments_N describing both segments -- `SegmentInfos::segments` is
    // already a `Vec`, and both `parse`/`write` already loop over it, so
    // nothing in `segment_infos.rs` itself needed to change for this.
    let sis = SegmentInfos {
        id: COMMIT_ID,
        generation: 1,
        format_version: 0, // unused by write(); always emits VERSION_CURRENT
        lucene_version: sis_lucene_version(),
        index_created_version_major: lucene_version().major,
        version: 2,
        counter: 2,
        min_segment_lucene_version: Some(sis_lucene_version()),
        segments: vec![sci0, sci1],
        user_data: vec![("lucene-rust-test".to_string(), "true".to_string())],
    };
    let segments_file_name = segment_infos::write(&sis, &dir).unwrap();

    // Doc ids in a freshly-opened DirectoryReader are assigned by
    // concatenating segments in `SegmentInfos` order, so the manifest's flat
    // `doc.<n>` numbering below matches `segment0_docs` followed by
    // `segment1_docs`.
    let all_docs: Vec<&Document> = segment0_docs.iter().chain(segment1_docs.iter()).collect();
    let max_doc = all_docs.len() as i32;

    let mut manifest = std::fs::File::create(format!("{out_dir}/manifest.properties")).unwrap();
    writeln!(manifest, "segments_file_name={segments_file_name}").unwrap();
    writeln!(manifest, "num_segments=2").unwrap();
    writeln!(manifest, "max_doc={max_doc}").unwrap();
    writeln!(manifest, "num_docs={max_doc}").unwrap();
    for (doc_id, d) in all_docs.iter().enumerate() {
        let id_field = match &d.fields[0].value {
            FieldValue::String(s) => s.clone(),
            _ => unreachable!(),
        };
        let body_field = match &d.fields[1].value {
            FieldValue::String(s) => s.clone(),
            _ => unreachable!(),
        };
        writeln!(manifest, "doc.{doc_id}.id={id_field}").unwrap();
        writeln!(manifest, "doc.{doc_id}.body={body_field}").unwrap();
    }

    println!(
        "wrote a real 2-segment index ({} + {} docs) to {out_dir} (segments file: {segments_file_name})",
        segment0_docs.len(),
        segment1_docs.len()
    );
}
