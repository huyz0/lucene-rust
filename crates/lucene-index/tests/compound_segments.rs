//! Compound segments written by this port's `IndexWriter`
//! (`IndexWriterConfig.setUseCompoundFile`, `MergePolicy.useCompoundFile`):
//! the file layout, and every later operation on a compound segment --
//! deletes, doc-values updates, merges -- through this port's `CheckIndex`.
//! Real Lucene reads the same layout in `scripts/verify-write-path.sh`
//! (`VerifyCompoundSegment`).
// Test code: see `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use std::sync::Arc;

use lucene_codecs::field_infos::{DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions};
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
use lucene_index::buffered_updates::Term;
use lucene_index::check_index;
use lucene_index::index_writer::IndexWriter;
use lucene_index::merge_policy::{MergePolicy, MergePolicyConfig, TieredMergePolicy};
use lucene_index::segment_info::{self, LuceneVersion};
use lucene_store::directory::{Directory, FsDirectory};
use lucene_util::test_support::TempDir;

fn writer(dir: &FsDirectory) -> IndexWriter<'_> {
    let fields = vec![
        FieldInfo::new("id", 0)
            .with_index_options(IndexOptions::Docs)
            .with_omit_norms(true),
        FieldInfo::new("body", 1).with_index_options(IndexOptions::DocsAndFreqsAndPositions),
        FieldInfo::new("num", 2)
            .with_omit_norms(true)
            .with_doc_values(DocValuesType::Numeric, DocValuesSkipIndexType::None, -1),
    ];
    let mut w = IndexWriter::open(
        dir,
        fields,
        "Lucene104",
        LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        },
    )
    .unwrap();
    w.set_postings_field(Some("id")).unwrap();
    w.add_postings_field("body").unwrap();
    w.set_doc_values_field(Some("num")).unwrap();
    w.set_max_buffered_docs(10).unwrap();
    w.set_ram_buffer_size_mb(1024.0).unwrap();
    w
}

fn doc(i: usize) -> Document {
    Document {
        fields: vec![
            StoredField {
                field_number: 0,
                value: FieldValue::String(format!("d{i}")),
            },
            StoredField {
                field_number: 1,
                value: FieldValue::String(format!("w{} common", i % 3)),
            },
            StoredField {
                field_number: 2,
                value: FieldValue::Long(i as i64),
            },
        ],
    }
}

/// `(is_compound, files)` of every segment of the writer's current view.
fn layout(dir: &FsDirectory, w: &IndexWriter<'_>) -> Vec<(bool, Vec<String>)> {
    w.segment_infos()
        .segments
        .iter()
        .map(|s| {
            let si = segment_info::parse(
                &dir.open(&format!("{}.si", s.segment_name)).unwrap(),
                &s.segment_id,
            )
            .unwrap();
            let mut files = si.files.clone();
            files.sort();
            (si.is_compound_file, files)
        })
        .collect()
}

fn assert_clean(dir: &FsDirectory) {
    for r in check_index::check_directory(dir).unwrap() {
        assert!(r.all_passed(), "{}: {:?}", r.segment_name, r.failures());
    }
}

#[test]
fn flushed_segments_are_packed_and_stay_usable() {
    let tmp = TempDir::new("compound-flush");
    let dir = FsDirectory::open(&tmp);
    let mut w = writer(&dir);
    assert!(!w.use_compound_file(), "off by default");
    w.set_use_compound_file(true);
    assert!(w.use_compound_file());
    for i in 0..25 {
        w.add_document(doc(i)).unwrap();
    }
    w.commit().unwrap();
    let layout0 = layout(&dir, &w);
    assert_eq!(layout0.len(), 3);
    for (i, (compound, files)) in layout0.iter().enumerate() {
        assert!(compound);
        assert_eq!(
            files,
            &[format!("_{i}.cfe"), format!("_{i}.cfs"), format!("_{i}.si")]
        );
    }
    // No loose codec file survived the packing.
    let loose: Vec<String> = dir
        .list_all()
        .unwrap()
        .into_iter()
        .filter(|f| {
            f.starts_with('_')
                && !f.ends_with(".cfs")
                && !f.ends_with(".cfe")
                && !f.ends_with(".si")
        })
        .collect();
    assert!(loose.is_empty(), "{loose:?}");
    assert_clean(&dir);

    // Deletes and doc-values updates land beside the compound file.
    w.delete_documents_by_term(&[Term::new("id", "d3")])
        .unwrap();
    w.update_numeric_doc_value(Term::new("id", "d4"), "num", 400)
        .unwrap();
    w.commit().unwrap();
    assert_eq!(w.segment_infos().segments[0].del_count, 1);
    assert!(w.segment_infos().segments[0].doc_values_gen > 0);
    assert_clean(&dir);

    // Turned off, the next flush is loose again.
    w.set_use_compound_file(false);
    w.add_document(doc(99)).unwrap();
    w.commit().unwrap();
    let last = layout(&dir, &w).pop().unwrap();
    assert!(!last.0);
    assert!(last.1.iter().any(|f| f.ends_with(".fdt")));
    assert_clean(&dir);
}

/// A merge is packed when the merge policy's `useCompoundFile` says so:
/// `noCFSRatio` 1.0 always, 0.0 never; the built-in policy only with
/// compound files on, at `TieredMergePolicy`'s 0.1.
#[test]
fn merges_follow_the_policys_use_compound_file() {
    for (ratio, packed) in [(1.0, true), (0.0, false)] {
        let tmp = TempDir::new("compound-merge");
        let dir = FsDirectory::open(&tmp);
        let mut w = writer(&dir);
        for i in 0..30 {
            w.add_document(doc(i)).unwrap();
        }
        w.commit().unwrap();
        let mut policy = TieredMergePolicy::new(MergePolicyConfig::default());
        policy
            .compound_file_settings_mut()
            .set_no_cfs_ratio(ratio)
            .unwrap();
        w.set_pluggable_merge_policy(Some(Arc::new(policy)));
        w.delete_documents_by_term(&[Term::new("id", "d7")])
            .unwrap();
        w.commit().unwrap();
        w.force_merge(1).unwrap();
        let merged = layout(&dir, &w);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].0, packed, "noCFSRatio {ratio}");
        assert_clean(&dir);
        // And a compound merged segment is merged again, and deleted from.
        w.add_document(doc(100)).unwrap();
        w.commit().unwrap();
        w.delete_documents_by_term(&[Term::new("id", "d8")])
            .unwrap();
        w.force_merge(1).unwrap();
        assert_clean(&dir);
    }

    // The built-in policy with compound files on: the merged segment is the
    // whole index, above 10% of it, so not packed; with them off, never.
    let tmp = TempDir::new("compound-merge-builtin");
    let dir = FsDirectory::open(&tmp);
    let mut w = writer(&dir);
    w.set_use_compound_file(true);
    for i in 0..30 {
        w.add_document(doc(i)).unwrap();
    }
    w.commit().unwrap();
    w.force_merge(1).unwrap();
    assert!(!layout(&dir, &w)[0].0);
    assert_clean(&dir);
}
