//! Writes a genuinely self-contained, single-segment index -- stored fields
//! (`.fdt`/`.fdx`/`.fdm`), field infos (`.fnm`), segment info (`.si`), and the
//! commit file itself (`segments_N`) -- to the directory given as the first
//! CLI argument, entirely through real on-disk `Directory`/`IndexOutput`
//! writes.
//!
//! This is the milestone the previous three write-path slices built towards:
//! `fixtures/src/VerifySegmentInfos.java` opens the result via real,
//! high-level `DirectoryReader.open(FSDirectory.open(path))` -- ordinary
//! application code, not hand-built codec-level access -- and checks doc
//! count and stored field values through `IndexReader`/`StoredFields`.
//!
//! To make that possible without a postings/doc-values/points/vectors writer
//! (none ported yet), every field here is stored-only
//! (`IndexOptions::None`, no doc values, no term vectors, no points, no
//! vectors): `SegmentCoreReaders` only opens a `FieldsProducer` when
//! `FieldInfos.hasPostings()` is true (see
//! `org.apache.lucene.index.SegmentCoreReaders`), so a segment with zero
//! indexed fields needs no `.tim`/`.tip`/`.doc` files at all, and none of the
//! other per-field producers are opened either -- stored fields are the one
//! per-document format this port can write end-to-end today. This is a real
//! constraint of what's implemented so far, not a shortcut in the verifier:
//! see `docs/parity.md` for what a fully-indexed segment would still need.
//!
//! Run: `cargo run -p lucene-index --example write_segment_infos_fixture -- <dir>`

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

const SEGMENT_ID: [u8; 16] = *b"rustwrittensis00";
const COMMIT_ID: [u8; 16] = *b"rustwrittencommt";
const SEGMENT_NAME: &str = "_0";
const CODEC_NAME: &str = "Lucene104";

fn lucene_version() -> LuceneVersion {
    // Matches the version already established (and real-Lucene-verified) by
    // `write_segment_info_fixture.rs` -- kept identical here rather than
    // reading it from the runtime JAR, since both fixtures target the same
    // pinned Lucene version (see PLAN.md).
    LuceneVersion {
        major: 10,
        minor: 0,
        bugfix: 0,
    }
}

/// `segment_infos::LuceneVersion` is a distinct type from
/// `segment_info::LuceneVersion` (same shape, no shared crate to hang a
/// common type off -- `lucene-index` owns both `.si` and `segments_N`
/// modules directly, so there's no natural third home for a shared version
/// type yet); trivial conversion here rather than changing either module's
/// existing, already-verified representation.
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

fn main() {
    let out_dir = std::env::args()
        .nth(1)
        .expect("usage: write_segment_infos_fixture <output-dir>");
    std::fs::create_dir_all(&out_dir).unwrap();
    let dir = FsDirectory::open(&out_dir);

    let fields = vec![stored_only_field("id", 0), stored_only_field("body", 1)];

    let docs = vec![
        Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String("1".to_string()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::String("the quick brown fox".to_string()),
                },
            ],
        },
        Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String("2".to_string()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::String("jumps over the lazy dog".to_string()),
                },
            ],
        },
        Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String("3".to_string()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::String(
                        "pack my box with five dozen liquor jugs".to_string(),
                    ),
                },
            ],
        },
    ];
    let max_doc = docs.len() as i32;

    // .fdt/.fdx/.fdm + .fnm + .si -- one "flush" via the shared
    // single-segment builder (see `lucene_index::segment_writer`, also
    // exercised twice per commit by `write_multi_segment_commit_fixture.rs`).
    let sci = flush_stored_only_segment(
        &dir,
        SEGMENT_NAME,
        SEGMENT_ID,
        CODEC_NAME,
        lucene_version(),
        &fields,
        &docs,
        false,
    )
    .unwrap();

    // segments_N -- the piece this slice ports.
    let sis = SegmentInfos {
        id: COMMIT_ID,
        generation: 1,
        format_version: 0, // unused by write(); always emits VERSION_CURRENT
        lucene_version: sis_lucene_version(),
        index_created_version_major: lucene_version().major,
        version: 2,
        counter: 1,
        min_segment_lucene_version: Some(sis_lucene_version()),
        segments: vec![sci],
        user_data: vec![("lucene-rust-test".to_string(), "true".to_string())],
    };
    let segments_file_name = segment_infos::write(&sis, &dir).unwrap();

    let mut manifest = std::fs::File::create(format!("{out_dir}/manifest.properties")).unwrap();
    writeln!(manifest, "segments_file_name={segments_file_name}").unwrap();
    writeln!(manifest, "segment_name={SEGMENT_NAME}").unwrap();
    writeln!(manifest, "max_doc={max_doc}").unwrap();
    writeln!(manifest, "num_docs={max_doc}").unwrap();
    for (doc_id, doc) in docs.iter().enumerate() {
        let id_field = match &doc.fields[0].value {
            FieldValue::String(s) => s.clone(),
            _ => unreachable!(),
        };
        let body_field = match &doc.fields[1].value {
            FieldValue::String(s) => s.clone(),
            _ => unreachable!(),
        };
        writeln!(manifest, "doc.{doc_id}.id={id_field}").unwrap();
        writeln!(manifest, "doc.{doc_id}.body={body_field}").unwrap();
    }

    println!(
        "wrote a complete single-segment index to {out_dir} (segments file: {segments_file_name})"
    );
}
