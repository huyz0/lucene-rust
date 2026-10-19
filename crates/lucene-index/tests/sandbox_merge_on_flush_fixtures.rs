//! lucene-sandbox's `MergeOnFlushMergePolicy` against Java:
//! `GenSandboxMergeOnFlush` records the segments of every commit and every
//! near-real-time reader of a writer whose policy merges, on each full
//! flush, the segments below 0.008 MB -- waiting for those merges
//! (`maxFullFlushMergeWaitMillis` 500) and not (0). This runs the same steps
//! through [`IndexWriter`] and must produce the same segment lists -- names,
//! document counts and order -- and the same documents in the same order.

// Test-support arithmetic over small in-memory counters; see
// `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use std::sync::Arc;

use lucene_codecs::field_infos::{
    DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions, VectorEncoding,
    VectorSimilarityFunction,
};
use lucene_codecs::stored_fields::{self, Document, FieldValue, StoredField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::merge_policy::{
    CompoundFileSettings, MergeOnFlushMergePolicy, MergePolicy, MergePolicyConfig,
    TieredMergePolicy,
};
use lucene_index::segment_info::{self, LuceneVersion};
use lucene_index::segment_infos;
use lucene_store::directory::{Directory, FsDirectory};
use lucene_util::test_support::TempDir;

/// `GenSandboxMergeOnFlush.STEPS`.
const STEPS: [&str; 8] = ["ssc", "bsc", "sr", "bbc", "sssc", "bsbr", "c", "sc"];

fn stored_field(name: &str, number: i32) -> FieldInfo {
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

/// `GenSandboxMergeOnFlush.big`: 16 KiB of xorshift letters.
fn big(n: i64) -> String {
    let mut x = 0x9E37_79B9_7F4A_7C15_u64.wrapping_mul((n + 1) as u64);
    let mut s = String::with_capacity(16384);
    while s.len() < 16384 {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        s.push((b'a' + ((x >> 33) % 26) as u8) as char);
    }
    s
}

fn max_doc(dir: &dyn Directory, sci: &segment_infos::SegmentCommitInfo) -> i32 {
    let si = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
    segment_info::parse(&si, &sci.segment_id).unwrap().doc_count
}

fn segments(dir: &dyn Directory, sis: &segment_infos::SegmentInfos) -> String {
    sis.segments
        .iter()
        .map(|s| format!(" {}:{}", s.segment_name, max_doc(dir, s)))
        .collect()
}

/// The ids of the latest commit, in doc order.
fn docs(dir: &dyn Directory) -> String {
    let sis = segment_infos::read_latest(dir).unwrap();
    let mut out = String::from("docs");
    for sci in &sis.segments {
        let read = |ext: &str| dir.open(&format!("{}.{ext}", sci.segment_name)).unwrap();
        let (fdt, fdx, fdm) = (read("fdt"), read("fdx"), read("fdm"));
        let reader = stored_fields::open(&fdt, &fdx, &fdm, &sci.segment_id, "").unwrap();
        for d in 0..reader.max_doc() {
            let doc = reader.document(d).unwrap();
            let id = doc.fields.iter().find(|f| f.field_number == 0).unwrap();
            if let FieldValue::String(s) = &id.value {
                out.push(' ');
                out.push_str(s);
            }
        }
    }
    out
}

fn run(dir: &FsDirectory, wait_millis: i64) -> Vec<String> {
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let fields = vec![stored_field("id", 0), stored_field("big", 1)];
    let mut w = IndexWriter::open(dir, fields, "Lucene104", version).unwrap();
    let mut tmp = TieredMergePolicy::new(MergePolicyConfig {
        segments_per_tier: 50,
        max_merge_at_once: 50,
        ..MergePolicyConfig::default()
    });
    tmp.compound_file_settings_mut()
        .set_no_cfs_ratio(0.0)
        .unwrap();
    let mut mp = MergeOnFlushMergePolicy::new(Box::new(tmp));
    mp.set_small_segment_threshold_mb(0.008);
    w.set_pluggable_merge_policy(Some(Arc::new(mp)));
    w.set_max_full_flush_merge_wait_millis(wait_millis);
    let mut lines = Vec::new();
    let mut next = 0i64;
    for step in STEPS {
        for k in step[..step.len() - 1].chars() {
            let mut fields = vec![StoredField {
                field_number: 0,
                value: FieldValue::String(format!("d{next:03}")),
            }];
            if k == 'b' {
                fields.push(StoredField {
                    field_number: 1,
                    value: FieldValue::String(big(next)),
                });
            }
            next += 1;
            w.add_document(Document { fields }).unwrap();
            w.flush().unwrap();
        }
        if step.ends_with('c') {
            w.commit().unwrap();
            let sis = segment_infos::read_latest(dir).unwrap();
            lines.push(format!("commit {}{}", sis.generation, segments(dir, &sis)));
        } else {
            let snapshot = w.nrt_snapshot(true, false).unwrap();
            lines.push(format!("reader{}", segments(dir, &snapshot.segment_infos)));
        }
    }
    w.close().unwrap();
    lines.push(docs(dir));
    lines
}

fn expected(which: &str) -> Vec<String> {
    let path = format!(
        "{}/../../fixtures/data/sandbox_merge_on_flush/{which}/manifest.txt",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read_to_string(path)
        .expect("run fixtures generator first (GenSandboxMergeOnFlush)")
        .lines()
        .map(str::to_string)
        .collect()
}

#[test]
fn small_segments_merge_on_commit_and_refresh_as_java_does() {
    let tmp = TempDir::new("merge-on-flush-wait");
    let dir = FsDirectory::open(&tmp);
    assert_eq!(expected("wait"), run(&dir, 500));
}

#[test]
fn a_zero_wait_keeps_the_flushed_segments_as_java_does() {
    let tmp = TempDir::new("merge-on-flush-nowait");
    let dir = FsDirectory::open(&tmp);
    assert_eq!(expected("nowait"), run(&dir, 0));
}

#[test]
fn the_policy_reads_its_settings_back() {
    let p = MergeOnFlushMergePolicy::new(Box::new(TieredMergePolicy::default()));
    assert_eq!(p.small_segment_threshold_mb(), 100.0);
    let _: CompoundFileSettings = p.compound_file_settings();
}
