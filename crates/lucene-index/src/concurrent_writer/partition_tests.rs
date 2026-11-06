// Test arithmetic on small constants; see docs/arithmetic-gate.md.
#![allow(clippy::arithmetic_side_effects)]

//! Segment partitions through the concurrent writer: every segment holds one
//! partition and records it, and merges keep partitions apart.

use std::collections::BTreeMap;

use lucene_codecs::field_infos::{FieldInfo, IndexOptions};
use lucene_codecs::stored_fields::{self, FieldValue, StoredField};
use lucene_store::FsDirectory;
use lucene_util::test_support::TempDir;

use super::*;
use crate::index_writer::{
    ExplicitDocument, ExplicitFields, InvertedField, InvertedTerm, SegmentPartitioning,
};
use crate::merge_policy::api::{MergePolicy, TieredMergePolicy};
use crate::merge_policy::MergePolicyConfig;
use crate::segment_info::LuceneVersion;
use crate::{segment_info, segment_infos};

const VERSION: LuceneVersion = LuceneVersion {
    major: 10,
    minor: 5,
    bugfix: 0,
};

fn open(
    dir: &FsDirectory,
    slots: usize,
    partitioning: Option<SegmentPartitioning>,
) -> ConcurrentIndexWriter<'_> {
    let mut w = IndexWriter::open(dir, Vec::new(), "Lucene104", VERSION).unwrap();
    w.set_ram_buffer_size_mb(16.0).unwrap();
    w.enable_explicit_documents().unwrap();
    w.register_field(
        FieldInfo::new("_id", 0)
            .with_index_options(IndexOptions::Docs)
            .with_omit_norms(true),
    )
    .unwrap();
    w.set_merge_policy(Some(MergePolicyConfig {
        max_merge_at_once: 4,
        segments_per_tier: 2.0,
        floor_segment_size: 1 << 30,
        ..MergePolicyConfig::default()
    }));
    w.set_max_full_flush_merge_wait_millis(0);
    w.set_segment_partitioning(partitioning);
    ConcurrentIndexWriter::new(w, slots).unwrap()
}

/// Document `i` of partition `p`: `_id` "p{p}-{i}".
fn doc(p: usize, i: usize) -> ExplicitDocument {
    let id = format!("p{p}-{i}");
    ExplicitDocument {
        stored: vec![StoredField {
            field_number: 0,
            value: FieldValue::String(id.clone()),
        }],
        fields: ExplicitFields {
            inverted: vec![InvertedField {
                field_number: 0,
                terms: vec![InvertedTerm {
                    term: id.into_bytes(),
                    freq: 1,
                    positions: Vec::new(),
                    offsets: Vec::new(),
                    payloads: Vec::new(),
                }]
                .into(),
                norm: None,
            }],
            ..ExplicitFields::default()
        },
    }
}

fn partition(p: usize) -> Option<Arc<str>> {
    Some(Arc::from(format!("p{p}").as_str()))
}

/// Every segment of the latest commit: its `bucket` attribute, the
/// `_id` field's, and the partitions (the `_id` prefixes) of its documents.
struct Seen {
    bucket: Option<String>,
    id_bucket: Option<String>,
    prefixes: Vec<String>,
    merged: bool,
}

fn segments(dir: &FsDirectory) -> Vec<Seen> {
    let infos = segment_infos::read_latest(dir).unwrap();
    infos
        .segments
        .iter()
        .map(|sci| {
            let name = &sci.segment_name;
            let si =
                segment_info::parse(&dir.open(&format!("{name}.si")).unwrap(), &sci.segment_id)
                    .unwrap();
            let attr = |attrs: &[(String, String)]| {
                attrs
                    .iter()
                    .find(|(k, _)| k == "bucket")
                    .map(|(_, v)| v.clone())
            };
            let fnm = lucene_codecs::field_infos::parse(
                &dir.open(&format!("{name}.fnm")).unwrap(),
                &sci.segment_id,
                "",
            )
            .unwrap();
            let fdt = dir.open(&format!("{name}.fdt")).unwrap();
            let fdx = dir.open(&format!("{name}.fdx")).unwrap();
            let fdm = dir.open(&format!("{name}.fdm")).unwrap();
            let reader = stored_fields::open(&fdt, &fdx, &fdm, &sci.segment_id, "").unwrap();
            let mut prefixes: Vec<String> = (0..si.doc_count)
                .map(|d| match &reader.document(d).unwrap().fields[0].value {
                    FieldValue::String(s) => s.split('-').next().unwrap().to_string(),
                    other => panic!("{other:?}"),
                })
                .collect();
            prefixes.sort();
            prefixes.dedup();
            Seen {
                bucket: attr(&si.attributes),
                id_bucket: fnm.field_by_name("_id").and_then(|f| attr(&f.attributes)),
                prefixes,
                merged: si
                    .diagnostics
                    .iter()
                    .any(|(k, v)| k == "source" && v == "merge"),
            }
        })
        .collect()
}

/// Asserts every segment holds one partition and records it, on the
/// segment and on `_id`; returns the documents per partition.
fn assert_one_partition_each(dir: &FsDirectory) -> BTreeMap<String, usize> {
    let mut per = BTreeMap::new();
    for seen in segments(dir) {
        assert_eq!(
            seen.prefixes.len(),
            1,
            "a segment mixes partitions: {:?}",
            seen.prefixes
        );
        assert_eq!(seen.bucket.as_deref(), Some(seen.prefixes[0].as_str()));
        assert_eq!(seen.id_bucket, seen.bucket);
        *per.entry(seen.prefixes[0].clone()).or_insert(0) += 1;
    }
    per
}

#[test]
fn interleaved_partitions_flush_to_segments_of_one_partition_each() {
    let tmp = TempDir::new("partitions-interleaved");
    let dir = FsDirectory::open(&tmp);
    // Fewer slots than partitions: a buffer holding another partition is
    // flushed to make room.
    let w = open(&dir, 2, Some(SegmentPartitioning::opensearch()));
    for i in 0..60 {
        let p = i % 3;
        w.add_explicit_documents_in(partition(p), vec![doc(p, i)])
            .unwrap();
    }
    w.commit().unwrap();
    let per = assert_one_partition_each(&dir);
    assert_eq!(per.keys().cloned().collect::<Vec<_>>(), ["p0", "p1", "p2"]);
}

#[test]
fn documents_added_without_a_partition_record_the_default() {
    let tmp = TempDir::new("partitions-default");
    let dir = FsDirectory::open(&tmp);
    let w = open(&dir, 4, Some(SegmentPartitioning::opensearch()));
    w.add_explicit_documents(vec![doc(9, 0)]).unwrap();
    w.commit().unwrap();
    let seen = segments(&dir);
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0].bucket.as_deref(), Some("-2"));
}

#[test]
fn without_partitioning_nothing_is_recorded_and_buffers_mix() {
    let tmp = TempDir::new("partitions-off");
    let dir = FsDirectory::open(&tmp);
    let w = open(&dir, 1, None);
    for i in 0..6 {
        w.add_explicit_documents_in(partition(i % 2), vec![doc(i % 2, i)])
            .unwrap();
    }
    w.commit().unwrap();
    let seen = segments(&dir);
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0].bucket, None);
    assert_eq!(seen[0].prefixes, ["p0", "p1"]);
}

/// Several commits of every partition, so each has many small segments.
fn many_segments(dir: &FsDirectory) -> ConcurrentIndexWriter<'_> {
    let w = open(dir, 4, Some(SegmentPartitioning::opensearch()));
    for round in 0..6 {
        for p in 0..3 {
            for i in 0..5 {
                w.add_explicit_documents_in(partition(p), vec![doc(p, round * 10 + i)])
                    .unwrap();
            }
        }
        w.commit().unwrap();
    }
    w
}

#[test]
fn natural_merges_take_one_partition_at_a_time() {
    let tmp = TempDir::new("partitions-natural-merge");
    let dir = FsDirectory::open(&tmp);
    let w = many_segments(&dir);
    let before = segments(&dir).len();
    while w.maybe_merge().unwrap() > 0 {}
    w.commit().unwrap();
    let after = segments(&dir);
    assert!(after.len() < before, "{} -> {}", before, after.len());
    assert!(after.iter().any(|s| s.merged));
    assert_one_partition_each(&dir);
}

#[test]
fn a_forced_merge_to_one_segment_leaves_one_per_partition() {
    for pluggable in [false, true] {
        let tmp = TempDir::new("partitions-forced-merge");
        let dir = FsDirectory::open(&tmp);
        let w = many_segments(&dir);
        w.with_writer_merges_aborted(|iw| {
            if pluggable {
                let mut tiered = TieredMergePolicy::new(MergePolicyConfig::default());
                // Loose files, which the test reads directly.
                tiered
                    .compound_file_settings_mut()
                    .set_no_cfs_ratio(0.0)
                    .unwrap();
                iw.set_pluggable_merge_policy(Some(Arc::new(tiered)));
            }
            iw.force_merge(1)
        })
        .unwrap();
        w.commit().unwrap();
        let per = assert_one_partition_each(&dir);
        assert_eq!(
            per,
            BTreeMap::from([("p0".into(), 1), ("p1".into(), 1), ("p2".into(), 1)]),
            "pluggable: {pluggable}"
        );
        assert!(
            segments(&dir).iter().all(|s| s.merged),
            "pluggable: {pluggable}"
        );
    }
}

#[test]
fn many_threads_keep_partitions_apart() {
    let tmp = TempDir::new("partitions-threads");
    let dir = FsDirectory::open(&tmp);
    let w = open(&dir, 3, Some(SegmentPartitioning::opensearch()));
    std::thread::scope(|scope| {
        for t in 0..4 {
            let w = &w;
            scope.spawn(move || {
                for i in 0..200 {
                    let p = (i + t) % 5;
                    w.add_explicit_documents_in(partition(p), vec![doc(p, t * 1000 + i)])
                        .unwrap();
                }
            });
        }
    });
    w.commit().unwrap();
    let per = assert_one_partition_each(&dir);
    assert_eq!(per.len(), 5);
}

#[test]
fn a_forced_merge_without_a_merge_policy_keeps_partitions_apart() {
    let tmp = TempDir::new("partitions-forced-no-policy");
    let dir = FsDirectory::open(&tmp);
    let w = many_segments(&dir);
    w.with_writer_merges_aborted(|iw| {
        iw.set_merge_policy(None);
        iw.force_merge(1)
    })
    .unwrap();
    w.commit().unwrap();
    let per = assert_one_partition_each(&dir);
    assert_eq!(per.values().copied().collect::<Vec<_>>(), [1, 1, 1]);
}

#[test]
fn an_add_refused_after_its_slot_was_flushed_keeps_the_flushed_documents() {
    let tmp = TempDir::new("partitions-refused-add");
    let dir = FsDirectory::open(&tmp);
    let mut iw = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
    iw.enable_explicit_documents().unwrap();
    iw.register_field(
        FieldInfo::new("_id", 0)
            .with_index_options(IndexOptions::Docs)
            .with_omit_norms(true),
    )
    .unwrap();
    iw.set_segment_partitioning(Some(SegmentPartitioning::opensearch()));
    iw.set_max_docs(2);
    let w = ConcurrentIndexWriter::new(iw, 1).unwrap();
    w.add_explicit_documents_in(partition(0), vec![doc(0, 0)])
        .unwrap();
    w.add_explicit_documents_in(partition(0), vec![doc(0, 1)])
        .unwrap();
    // The one slot holds partition 0: it is flushed, then the add is over
    // the document limit.
    assert!(w
        .add_explicit_documents_in(partition(1), vec![doc(1, 0)])
        .is_err());
    w.commit().unwrap();
    let per = assert_one_partition_each(&dir);
    assert_eq!(per, BTreeMap::from([("p0".into(), 1)]));
    let infos = segment_infos::read_latest(&dir).unwrap();
    assert_eq!(infos.segments.len(), 1);
}

#[test]
fn many_interleaved_partitions_each_keep_one_buffer() {
    let tmp = TempDir::new("partitions-many");
    let dir = FsDirectory::open(&tmp);
    let w = open(&dir, 64, Some(SegmentPartitioning::opensearch()));
    for i in 0..400 {
        let p = i % 40;
        w.add_explicit_documents_in(partition(p), vec![doc(p, i)])
            .unwrap();
    }
    w.commit().unwrap();
    let per = assert_one_partition_each(&dir);
    assert_eq!(per.len(), 40);
    assert!(per.values().all(|&n| n == 1), "{per:?}");
}
