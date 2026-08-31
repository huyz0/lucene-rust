//! Alternating A/B measurement of `IndexFileDeleter`'s **checkpoint** (ledger
//! item 25b): the same segment list reference-counted twice, once with each
//! segment's file set already in memory and once with the deleter having to
//! open and parse every `.si` to get it.
//!
//! What the change is. Java's `SegmentCommitInfo` holds its `SegmentInfo`, so
//! `SegmentInfos.files()` -- and therefore every checkpoint -- reads a
//! segment's file set out of memory. This port's `SegmentCommitInfo`
//! deliberately does not own the parsed `.si`, so the deleter opened the file
//! and ran `segment_info::parse` over it (index-header check and a CRC over
//! the whole file included). `c36-merge-metadata` left exactly one such read
//! per flushed segment and pinned it with a counting `Directory`;
//! `IndexFileDeleter::record_segment_files` removes it, fed from the same
//! in-memory `SegmentInfo` `seal_flushed_segment` encodes.
//!
//! Both arms are shipped code paths and run in one process from one build:
//! arm A is a deleter the writer has told about its segments, arm B is one
//! that has not been. There is no second binary and no rebuild between arms --
//! the trap `c42-readpath-perf` found in `scripts/bench-micro.sh`, where the
//! container had been timing a months-old binary. Criterion is not used: it
//! reported 83/91/129 µs for identical code on this machine
//! (`docs/sweep/m2/c24-arith-codecs.md`), so every figure is a **min of N
//! alternating repetitions**.
//!
//! Run: `cargo run --release -p lucene-index --example deleter_checkpoint`
// Benchmark support code opts out of the arithmetic gate at the file
// boundary, as the fixture writers do. See `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use std::hint::black_box;
use std::time::Instant;

use lucene_codecs::field_infos::{
    DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions, VectorEncoding,
    VectorSimilarityFunction,
};
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
use lucene_index::index_file_deleter::{DeletionPolicy, IndexFileDeleter};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::{self, LuceneVersion};
use lucene_index::segment_infos::{self, SegmentInfos};
use lucene_store::{Directory, FsDirectory};
use lucene_util::test_support::TempDir;

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

/// Builds a committed index of `segments` one-document segments.
fn build(dir: &FsDirectory, segments: usize) {
    let fields = vec![
        field("id", 0),
        FieldInfo {
            index_options: IndexOptions::DocsAndFreqs,
            ..field("body", 1)
        },
    ];
    let mut writer = IndexWriter::open(
        dir,
        fields,
        "Lucene104",
        LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        },
    )
    .expect("open writer");
    writer.set_postings_field(Some("body")).unwrap();
    for i in 0..segments {
        writer
            .add_document(Document {
                fields: vec![
                    StoredField {
                        field_number: 0,
                        value: FieldValue::String(format!("d{i}")),
                    },
                    StoredField {
                        field_number: 1,
                        value: FieldValue::String(format!("term{i} shared")),
                    },
                ],
            })
            .unwrap();
        writer.flush().unwrap();
    }
    writer.commit().unwrap();
}

/// Every segment's `.si` file list, parsed once outside the timed region --
/// this is what `IndexWriter` has in hand for free (the `SegmentInfo` it just
/// encoded) and what `record_segment_files` takes.
fn si_file_lists(dir: &FsDirectory, infos: &SegmentInfos) -> Vec<Vec<String>> {
    infos
        .segments
        .iter()
        .map(|sci| {
            let bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
            segment_info::parse(&bytes, &sci.segment_id).unwrap().files
        })
        .collect()
}

fn run(label: &str, segments: usize, reps: usize) {
    let tmp = TempDir::new(&format!("c43-deleter-{segments}"));
    let dir = FsDirectory::open(&tmp);
    build(&dir, segments);
    let infos = segment_infos::read_latest(&dir).expect("read commit");
    assert_eq!(infos.segments.len(), segments, "one segment per flush");
    let lists = si_file_lists(&dir, &infos);

    let mut best_memory = u128::MAX;
    let mut best_disk = u128::MAX;
    for _ in 0..reps {
        // Arm A: the writer told the deleter each segment's files, so the
        // checkpoint reads nothing. A fresh deleter per repetition, because
        // the cache is what is being measured.
        let mut deleter =
            IndexFileDeleter::open(&dir, &infos, DeletionPolicy::KeepOnlyLastCommit).unwrap();
        for (sci, files) in infos.segments.iter().zip(&lists) {
            deleter.record_segment_files(sci, files);
        }
        let t = Instant::now();
        deleter.checkpoint(black_box(&infos), false).unwrap();
        best_memory = best_memory.min(t.elapsed().as_nanos());

        // Arm B: the pre-`c43-final-cleanup` situation -- the deleter has to
        // open and parse every `.si` itself.
        let mut deleter =
            IndexFileDeleter::open(&dir, &infos, DeletionPolicy::KeepOnlyLastCommit).unwrap();
        deleter.forget_segment_files();
        let t = Instant::now();
        deleter.checkpoint(black_box(&infos), false).unwrap();
        best_disk = best_disk.min(t.elapsed().as_nanos());
    }

    println!(
        "{label:<26} from memory {:>9.1} us   from .si {:>9.1} us   {:.2}x   ({:.2} us/segment saved)",
        best_memory as f64 / 1000.0,
        best_disk as f64 / 1000.0,
        best_disk as f64 / best_memory as f64,
        (best_disk.saturating_sub(best_memory)) as f64 / 1000.0 / segments as f64,
    );
}

fn main() {
    let reps = 40;
    println!("min of {reps} alternating repetitions, both arms in one process\n");
    run("8 segments", 8, reps);
    run("32 segments", 32, reps);
    run("128 segments", 128, reps);
}
