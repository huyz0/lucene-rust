//! Rich documents through the concurrent writer
//! ([`ConcurrentIndexWriter::add_fields_documents`] and
//! [`ConcurrentIndexWriter::update_fields_documents`]): they reach its
//! flush, commit and NRT snapshot, and its merge threads merge their
//! segments, with the same result as the single-threaded writer.

// Test fixtures' own arithmetic, not values read off disk -- see
// `docs/arithmetic-gate.md`'s "Test code" section.
#![allow(clippy::arithmetic_side_effects)]

use std::collections::BTreeSet;

use super::*;
use crate::document::{Document as FieldsDocument, NumericDocValuesField, Store, StringField};
use crate::merge_policy::MergePolicyConfig;
use crate::nrt::NrtSource;
use crate::segment_info::LuceneVersion;
use crate::{deletes, segment_info, segment_infos};
use lucene_codecs::stored_fields::{self, FieldValue};
use lucene_store::FsDirectory;
use lucene_util::test_support::TempDir;

const VERSION: LuceneVersion = LuceneVersion {
    major: 10,
    minor: 5,
    bugfix: 0,
};

fn writer(dir: &FsDirectory) -> IndexWriter<'_> {
    let mut w = IndexWriter::open(dir, Vec::new(), "Lucene104", VERSION).unwrap();
    w.set_max_full_flush_merge_wait_millis(0);
    w
}

fn rich(id: usize, version: usize) -> FieldsDocument {
    let mut d = FieldsDocument::new();
    d.add(StringField::new("id", format!("d{id}"), Store::Yes));
    d.add(StringField::new("v", version.to_string(), Store::Yes));
    d.add(NumericDocValuesField::new("n", version as i64));
    d
}

/// `id:version` of every live document of the latest commit.
fn live(dir: &FsDirectory) -> BTreeSet<String> {
    let infos = segment_infos::read_latest(dir).unwrap();
    let mut out = BTreeSet::new();
    for sci in &infos.segments {
        let name = &sci.segment_name;
        let si = segment_info::parse(&dir.open(&format!("{name}.si")).unwrap(), &sci.segment_id)
            .unwrap();
        let fdt = dir.open(&format!("{name}.fdt")).unwrap();
        let fdx = dir.open(&format!("{name}.fdx")).unwrap();
        let fdm = dir.open(&format!("{name}.fdm")).unwrap();
        let reader = stored_fields::open(&fdt, &fdx, &fdm, &sci.segment_id, "").unwrap();
        let liv = (sci.del_gen >= 0).then(|| {
            lucene_codecs::live_docs::parse(
                &dir.open(&deletes::liv_file_name(name, sci.del_gen))
                    .unwrap(),
                &sci.segment_id,
                sci.del_gen,
                si.doc_count as usize,
                sci.del_count as usize,
            )
            .unwrap()
        });
        for d in 0..si.doc_count {
            if liv.as_ref().is_some_and(|l| !l.get(d as usize)) {
                continue;
            }
            let text: Vec<String> = reader
                .document(d)
                .unwrap()
                .fields
                .iter()
                .map(|f| match &f.value {
                    FieldValue::String(s) => s.clone(),
                    other => panic!("{other:?}"),
                })
                .collect();
            assert!(out.insert(text.join(":")), "live twice: {text:?}");
        }
    }
    out
}

fn check(dir: &FsDirectory) {
    for r in crate::check_index::check_directory(dir).unwrap() {
        assert!(r.all_passed(), "{}: {:?}", r.segment_name, r.failures());
    }
}

fn leaked(tmp: &TempDir) -> &'static FsDirectory {
    Box::leak(Box::new(FsDirectory::open(tmp)))
}

/// The same adds, updates and deletes through both writers leave the same
/// live documents, and the concurrent writer's index checks clean.
#[test]
fn rich_updates_leave_what_the_single_threaded_writer_leaves() {
    let tmp_a = TempDir::new("fields-concurrent-a");
    let tmp_b = TempDir::new("fields-concurrent-b");
    let (a, b) = (leaked(&tmp_a), leaked(&tmp_b));
    let mut single = writer(a);
    let conc = ConcurrentIndexWriter::new(writer(b), 2).unwrap();
    for round in 0..4 {
        for id in 0..50 {
            let term = Term::new("id", format!("d{id}").as_bytes());
            single
                .update_fields_documents(term.clone(), &[rich(id, round)])
                .unwrap();
            conc.update_fields_documents(term, &[rich(id, round)])
                .unwrap();
        }
        let gone = Term::new("id", format!("d{}", round * 3).as_bytes());
        single
            .delete_documents_by_term(std::slice::from_ref(&gone))
            .unwrap();
        conc.delete_documents_by_term(&[gone]).unwrap();
        single.commit().unwrap();
        conc.commit().unwrap();
    }
    assert_eq!(live(a), live(b));
    // Only the last delete (d9) is not re-added by a later round.
    assert_eq!(live(b).len(), 49);
    check(b);
}

/// A snapshot sees rich documents added since the last one, and an update
/// replaces the earlier version in it.
#[test]
fn an_nrt_snapshot_sees_rich_documents() {
    let tmp = TempDir::new("fields-concurrent-nrt");
    let dir = leaked(&tmp);
    let w = ConcurrentIndexWriter::new(writer(dir), 1).unwrap();
    let docs = |s: &crate::nrt::NrtSnapshot| -> i64 {
        s.segment_infos
            .segments
            .iter()
            .map(|sci| {
                let si = segment_info::parse(
                    &dir.open(&format!("{}.si", sci.segment_name)).unwrap(),
                    &sci.segment_id,
                )
                .unwrap();
                i64::from(si.doc_count) - i64::from(sci.del_count)
            })
            .sum()
    };
    assert_eq!(docs(&w.nrt_snapshot(true, false).unwrap()), 0);
    w.add_fields_documents(&[rich(1, 0), rich(2, 0)]).unwrap();
    assert_eq!(docs(&w.nrt_snapshot(true, false).unwrap()), 2);
    w.update_fields_documents(Term::new("id", b"d1"), &[rich(1, 1)])
        .unwrap();
    assert_eq!(docs(&w.nrt_snapshot(true, false).unwrap()), 2);
    w.commit().unwrap();
    assert_eq!(
        live(dir),
        BTreeSet::from(["d1:1".to_string(), "d2:0".to_string()])
    );
}

/// Refresh-flushed rich segments are merged by the scheduler's threads,
/// and a refused document leaves nothing buffered.
#[test]
fn rich_refresh_segments_are_merged_by_the_scheduler() {
    let tmp = TempDir::new("fields-concurrent-merge");
    let dir = leaked(&tmp);
    let cms = crate::merge_scheduler::ConcurrentMergeScheduler::new();
    cms.set_max_merges_and_threads(3, 2).unwrap();
    let mut iw = writer(dir);
    iw.set_merge_policy(Some(MergePolicyConfig {
        max_merge_at_once: 4,
        segments_per_tier: 3.0,
        floor_segment_size: 1 << 30,
        ..MergePolicyConfig::default()
    }));
    let w = ConcurrentIndexWriter::with_merge_scheduler(iw, 2, Arc::new(cms)).unwrap();
    for i in 0..300 {
        w.update_fields_documents(Term::new("id", format!("d{i}").as_bytes()), &[rich(i, 0)])
            .unwrap();
        if i % 3 == 2 {
            w.nrt_snapshot(true, false).unwrap();
        }
    }
    w.close_merges().unwrap();
    assert!(lock(&w.core).writer.segment_count() < 30);
    w.commit().unwrap();
    assert_eq!(live(dir).len(), 300);
    check(dir);
    // A document the writer refuses leaves no buffered trace.
    let mut bad = FieldsDocument::new();
    bad.add(StringField::new("id", "x", Store::Yes));
    bad.add(NumericDocValuesField::new("id", 1));
    assert!(w.add_fields_documents(&[bad]).is_err());
    assert_eq!(w.pending_doc_count(), 0);
}

fn texty(i: usize) -> FieldsDocument {
    let mut d = FieldsDocument::new();
    d.add(StringField::new("id", format!("t{i}"), Store::Yes));
    d.add(crate::document::TextField::new(
        "body",
        format!("quick brown fox {i} jumps over the lazy dog fox"),
        Store::No,
    ));
    // A second value of the same field: positions continue across values.
    d.add(crate::document::TextField::new(
        "body",
        format!("fox {}", i % 3),
        Store::No,
    ));
    d
}

type TermRow = (i32, Vec<u8>, i32, Vec<i32>, Vec<(i32, i32)>);

fn sorted_terms(e: &crate::index_writer::ExplicitDocument) -> Vec<TermRow> {
    let mut out = Vec::new();
    for inv in &e.fields.inverted {
        for t in inv.terms.iter() {
            out.push((
                inv.field_number,
                t.term.to_vec(),
                t.freq,
                t.positions.to_vec(),
                t.offsets.to_vec(),
            ));
        }
    }
    out.sort();
    out
}

/// The one-pass read-back of a block's postings gives each document the
/// terms, frequencies, positions and offsets that inverting it alone does.
#[test]
fn a_blocks_terms_read_back_as_each_document_alone() {
    let tmp_a = TempDir::new("fields-readback-a");
    let tmp_b = TempDir::new("fields-readback-b");
    let (a, b) = (leaked(&tmp_a), leaked(&tmp_b));
    let docs: Vec<FieldsDocument> = (0..20).map(texty).collect();
    let mut block = writer(a);
    let (explicit, _) = block.invert_fields_documents_detached(&docs).unwrap();
    let mut alone = writer(b);
    alone.enable_explicit_documents().unwrap();
    for (i, doc) in docs.iter().enumerate() {
        let (expected, _) = alone.invert_fields_document(doc).unwrap();
        let got = sorted_terms(&explicit[i]);
        assert!(!got.is_empty());
        assert_eq!(got, sorted_terms(&expected), "document {i}");
    }
}

/// A block whose last document fails during inversion leaves nothing in the
/// wrapped writer: a later block reads back as on a fresh writer.
#[test]
fn a_refused_block_leaves_no_postings_behind() {
    let tmp_a = TempDir::new("fields-rollback-a");
    let tmp_b = TempDir::new("fields-rollback-b");
    let (a, b) = (leaked(&tmp_a), leaked(&tmp_b));
    let mut bad = texty(99);
    bad.add(NumericDocValuesField::new("id", 1));
    let mut dirty = writer(a);
    let refused = [texty(1), texty(2), bad];
    assert!(dirty.invert_fields_documents_detached(&refused).is_err());
    let good = [texty(3), texty(4)];
    let (after, _) = dirty.invert_fields_documents_detached(&good).unwrap();
    let mut fresh = writer(b);
    let (clean, _) = fresh.invert_fields_documents_detached(&good).unwrap();
    // Field numbers may differ only if registration leaked; they must not.
    for (x, y) in after.iter().zip(&clean) {
        assert_eq!(sorted_terms(x), sorted_terms(y));
    }
}

/// With a parent field, the last document of every rich add is the parent;
/// a document carrying the reserved name is refused; an update with no
/// documents is a delete only.
#[test]
fn parent_fields_and_empty_updates_follow_the_single_threaded_writer() {
    let tmp = TempDir::new("fields-parent");
    let dir = leaked(&tmp);
    let mut iw = writer(dir);
    iw.set_parent_field(Some("_parent")).unwrap();
    let w = ConcurrentIndexWriter::new(iw, 1).unwrap();
    w.add_fields_documents(&[rich(1, 0), rich(2, 0), rich(3, 0)])
        .unwrap();
    w.update_fields_documents(Term::new("id", b"d3"), &[rich(3, 1)])
        .unwrap();
    // Deletes the block's parent, as the single-threaded writer does.
    w.update_fields_documents(Term::new("id", b"d1"), &[])
        .unwrap();
    let mut reserved = rich(9, 0);
    reserved.add(NumericDocValuesField::new("_parent", 1));
    assert!(w.add_fields_documents(&[reserved]).is_err());
    assert_eq!(w.pending_doc_count(), 4);
    w.commit().unwrap();
    assert_eq!(
        live(dir),
        BTreeSet::from(["d2:0".to_string(), "d3:1".to_string()])
    );
    check(dir);
}
