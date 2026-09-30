//! Merge-on-commit and merge-on-refresh against Java: `GenMergeOnCommit`
//! records, step by step, the segments of every commit and every
//! near-real-time reader of a writer that waits for `findFullFlushMerges`
//! merges (`maxFullFlushMergeWaitMillis` 500, Lucene's default) and of one
//! that does not (0). This runs the same steps through [`IndexWriter`] with
//! the same policy and must produce the same segment lists -- names, sizes
//! and order -- and the same documents in the same order.
//!
//! The policy is `TieredMergePolicy` (`segmentsPerTier` 2, `maxMergeAtOnce`
//! 2) proposing merges only for the `COMMIT` and `GET_READER` triggers, so
//! every merge is a point-in-time one (see the generator's doc).

// Test-support arithmetic over small in-memory counters; see
// `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use std::collections::HashMap;
use std::sync::Arc;

use lucene_codecs::field_infos::{
    DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions, VectorEncoding,
    VectorSimilarityFunction,
};
use lucene_codecs::stored_fields::{self, Document, FieldValue, StoredField};
use lucene_index::concurrent_writer::ConcurrentIndexWriter;
use lucene_index::index_writer::{IndexWriter, DEFAULT_MAX_FULL_FLUSH_MERGE_WAIT_MILLIS};
use lucene_index::merge_policy::{
    self, CompoundFileSettings, MergeContext, MergePolicy, MergePolicyConfig, MergeSegment,
    MergeSpecification, MergeTrigger, TieredMergePolicy,
};
use lucene_index::nrt::NrtSource;
use lucene_index::segment_info::{self, LuceneVersion};
use lucene_index::segment_infos;
use lucene_store::directory::{Directory, FsDirectory};
use lucene_util::test_support::TempDir;

/// `GenMergeOnCommit.FullFlushOnly`.
#[derive(Debug)]
struct FullFlushOnly(TieredMergePolicy);

impl MergePolicy for FullFlushOnly {
    fn find_merges(
        &self,
        trigger: MergeTrigger,
        infos: &[MergeSegment],
        ctx: &dyn MergeContext,
    ) -> merge_policy::api::Result<Option<MergeSpecification>> {
        if !matches!(trigger, MergeTrigger::Commit | MergeTrigger::GetReader) {
            return Ok(None);
        }
        self.0.find_merges(trigger, infos, ctx)
    }

    fn find_forced_merges(
        &self,
        infos: &[MergeSegment],
        max_segment_count: i32,
        segments_to_merge: &HashMap<String, bool>,
        ctx: &dyn MergeContext,
    ) -> merge_policy::api::Result<Option<MergeSpecification>> {
        self.0
            .find_forced_merges(infos, max_segment_count, segments_to_merge, ctx)
    }

    fn find_forced_deletes_merges(
        &self,
        infos: &[MergeSegment],
        ctx: &dyn MergeContext,
    ) -> merge_policy::api::Result<Option<MergeSpecification>> {
        self.0.find_forced_deletes_merges(infos, ctx)
    }

    fn max_full_flush_merge_size(&self) -> i64 {
        self.0.max_full_flush_merge_size()
    }

    fn compound_file_settings(&self) -> CompoundFileSettings {
        self.0.compound_file_settings()
    }

    fn compound_file_settings_mut(&mut self) -> &mut CompoundFileSettings {
        self.0.compound_file_settings_mut()
    }
}

/// `GenMergeOnCommit.STEPS`.
const STEPS: [&str; 11] = [
    "c1", "c1", "c1", "c1", "r1", "c1", "c1", "r1", "r1", "c2", "c1",
];

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
        vector_encoding: VectorEncoding::Float32,
        vector_similarity_function: VectorSimilarityFunction::Euclidean,
    }
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
            if let FieldValue::String(s) = &reader.document(d).unwrap().fields[0].value {
                out.push(' ');
                out.push_str(s);
            }
        }
    }
    out
}

/// The writer both runs start from: the generator's policy and wait.
fn writer(dir: &FsDirectory, wait_millis: Option<i64>) -> IndexWriter<'_> {
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut w = IndexWriter::open(dir, vec![id_field()], "Lucene104", version).unwrap();
    let mut tmp = TieredMergePolicy::new(MergePolicyConfig {
        segments_per_tier: 2,
        max_merge_at_once: 2,
        ..MergePolicyConfig::default()
    });
    tmp.compound_file_settings_mut()
        .set_no_cfs_ratio(0.0)
        .unwrap();
    w.set_pluggable_merge_policy(Some(Arc::new(FullFlushOnly(tmp))));
    if let Some(wait) = wait_millis {
        w.set_max_full_flush_merge_wait_millis(wait);
    }
    w
}

fn doc(n: usize) -> Document {
    Document {
        fields: vec![StoredField {
            field_number: 0,
            value: FieldValue::String(format!("d{n:03}")),
        }],
    }
}

/// Runs `GenMergeOnCommit`'s steps and returns the manifest lines.
fn run(dir: &FsDirectory, wait_millis: Option<i64>) -> Vec<String> {
    let mut w = writer(dir, wait_millis);
    let mut lines = Vec::new();
    let mut next = 0;
    for step in STEPS {
        let n: usize = step[1..].parse().unwrap();
        for _ in 0..n {
            w.add_document(doc(next)).unwrap();
            next += 1;
            w.flush().unwrap();
        }
        if step.starts_with('c') {
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

/// The same steps through a [`ConcurrentIndexWriter`] (one slot): its
/// commit and its near-real-time reader run the same point-in-time merges.
fn run_concurrent(dir: &FsDirectory, wait_millis: Option<i64>) -> Vec<String> {
    let w = ConcurrentIndexWriter::new(writer(dir, wait_millis), 1).unwrap();
    let mut lines = Vec::new();
    let mut next = 0;
    for step in STEPS {
        let n: usize = step[1..].parse().unwrap();
        for _ in 0..n {
            w.add_document(doc(next)).unwrap();
            next += 1;
            w.flush().unwrap();
        }
        if step.starts_with('c') {
            w.commit().unwrap();
            let sis = segment_infos::read_latest(dir).unwrap();
            lines.push(format!("commit {}{}", sis.generation, segments(dir, &sis)));
        } else {
            let snapshot = NrtSource::nrt_snapshot(&w, true, false).unwrap();
            lines.push(format!("reader{}", segments(dir, &snapshot.segment_infos)));
        }
    }
    w.into_writer().unwrap().close().unwrap();
    lines.push(docs(dir));
    lines
}

fn expected(which: &str) -> Vec<String> {
    let path = format!(
        "{}/../../fixtures/data/merge_on_commit/{which}/manifest.txt",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read_to_string(path)
        .expect("run fixtures generator first (GenMergeOnCommit)")
        .lines()
        .map(str::to_string)
        .collect()
}

/// Lucene's default wait: every commit and reader holds the merged segments.
#[test]
fn merge_on_commit_and_refresh_match_java() {
    let tmp = TempDir::new("merge-on-commit-wait");
    let dir = FsDirectory::open(&tmp);
    assert_eq!(expected("wait"), run(&dir, None));
}

/// A wait of 0 turns it off: every flushed segment reaches the commit.
#[test]
fn a_zero_wait_commits_the_flushed_segments_as_java_does() {
    let tmp = TempDir::new("merge-on-commit-nowait");
    let dir = FsDirectory::open(&tmp);
    assert_eq!(expected("nowait"), run(&dir, Some(0)));
}

#[test]
fn the_concurrent_writer_merges_on_commit_and_refresh_as_java_does() {
    for (which, wait) in [("wait", None), ("nowait", Some(0))] {
        let tmp = TempDir::new("merge-on-commit-concurrent");
        let dir = FsDirectory::open(&tmp);
        assert_eq!(expected(which), run_concurrent(&dir, wait), "{which}");
    }
}

#[test]
fn the_default_wait_is_javas() {
    let tmp = TempDir::new("merge-on-commit-default");
    let dir = FsDirectory::open(&tmp);
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let w = IndexWriter::open(&dir, vec![id_field()], "Lucene104", version).unwrap();
    assert_eq!(w.max_full_flush_merge_wait_millis(), 500);
    assert_eq!(DEFAULT_MAX_FULL_FLUSH_MERGE_WAIT_MILLIS, 500);
}
