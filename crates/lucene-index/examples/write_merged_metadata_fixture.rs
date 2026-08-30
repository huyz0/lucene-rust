//! Merges three segments **real Lucene wrote** and leaves the result for real
//! Lucene to read back -- the differential case for the two facts
//! `SegmentMerger` folds across its readers rather than copying from the
//! merging writer: `SegmentInfo.minVersion` and `SegmentInfo.hasBlocks`.
//!
//! Every other merge fixture in `scripts/verify-write-path.sh` merges segments
//! *this port* flushed, which is exactly the blind spot: this port stamps its
//! own version onto every segment it writes, so a merge of its own output
//! cannot tell "the minimum over the sources" apart from "the merging writer's
//! version". They are only different when a source was written by something
//! else -- which is the entire point of a compatible port, and which is what
//! `fixtures/data/merge_metadata/` supplies: three genuine Lucene 10.5.0
//! segments whose `.si` files record `minVersion` 10.2.0, 10.0.0 and 10.1.0,
//! with `hasBlocks` set on the middle one only (it was built with
//! `addDocuments`).
//!
//! The expected merged answers, per `SegmentMerger`'s constructor and
//! `IndexWriter.mergeMiddle`, are `minVersion = 10.0.0` (the oldest source)
//! and `hasBlocks = true` (any source). `VerifyMergedMetadata` reads both back
//! through `LeafMetaData`.
//!
//! Usage: `write_merged_metadata_fixture <output-dir>`.
// Test-support code opts out of the arithmetic gate at the file boundary:
// the gate exists for values read off disk in production decode paths, not
// for a fixture builder's own index arithmetic. See
// `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::field_infos::{
    DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions, VectorEncoding,
    VectorSimilarityFunction,
};
use lucene_index::index_writer::IndexWriter;
use lucene_index::merge_policy::MergePolicyConfig;
use lucene_index::segment_info::LuceneVersion;
use lucene_store::FsDirectory;

/// The Java-written source index, relative to this crate's manifest so the
/// example does not depend on the caller's working directory.
const SOURCE_FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../fixtures/data/merge_metadata"
);

/// The one field `GenMergeMetadata` puts in every document: stored only, so
/// the merged segment is about nothing but the two metadata fields under
/// test. Field number 0 and the name `id` must match the `.fnm` real Lucene
/// wrote, because this port's merge takes the merged schema from the caller
/// rather than from each source's `.fnm`.
fn id_field() -> FieldInfo {
    FieldInfo {
        name: "id".to_string(),
        number: 0,
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

fn main() {
    let out_dir = std::env::args()
        .nth(1)
        .expect("usage: write_merged_metadata_fixture <output-dir>");
    std::fs::create_dir_all(&out_dir).expect("create output dir");

    // Copy the Java-written index into the output directory: the merge runs
    // over a private copy, so the committed fixture is never mutated.
    let source = std::path::Path::new(SOURCE_FIXTURE);
    let mut copied = 0usize;
    for entry in std::fs::read_dir(source).unwrap_or_else(|e| {
        panic!("run scripts/gen-fixtures.sh --only GenMergeMetadata first ({SOURCE_FIXTURE}): {e}")
    }) {
        let entry = entry.expect("read fixture entry");
        let name = entry.file_name();
        let name = name.to_string_lossy();
        // `manifest.properties` is the fixture's description, not part of the
        // index; `write.lock` is an IndexWriter artifact.
        if name == "manifest.properties" || name == "write.lock" {
            continue;
        }
        std::fs::copy(entry.path(), std::path::Path::new(&out_dir).join(&*name))
            .expect("copy fixture file");
        copied += 1;
    }
    assert!(copied > 0, "no files copied from {SOURCE_FIXTURE}");

    let dir = FsDirectory::open(&out_dir);
    let mut writer = IndexWriter::open(
        &dir,
        vec![id_field()],
        "Lucene104",
        // The merging writer is *newer* than every source, which is what makes
        // "the writer's own version" and "the minimum over the sources"
        // distinguishable at all.
        LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        },
    )
    .expect("open the Java-written index");

    assert_eq!(
        writer.segment_infos().segments.len(),
        3,
        "the fixture must hand this writer exactly three segments to merge"
    );

    // Tight enough that three tiny segments are over budget, so `find_merges`
    // proposes them all in one group.
    writer.set_merge_policy(Some(MergePolicyConfig {
        max_merge_at_once: 10,
        segments_per_tier: 2,
        max_merged_segment_size: 1_000_000,
        reclaim_weight: 1.0,
        floor_segment_size: 100_000,
        force_merge_deletes_pct_allowed: 10.0,
        ..MergePolicyConfig::default()
    }));
    writer.commit().expect("commit, which runs the merge");

    let segments = &writer.segment_infos().segments;
    assert_eq!(
        segments.len(),
        1,
        "the three sources must have merged into one segment, got {:?}",
        segments
            .iter()
            .map(|s| s.segment_name.clone())
            .collect::<Vec<_>>()
    );

    println!(
        "wrote merged-metadata fixture to {out_dir} (segment {})",
        segments[0].segment_name
    );
}
