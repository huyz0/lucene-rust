#![cfg(test)]
// The arithmetic gate is about values read off disk; a test's `i + 1` is
// not one. See docs/arithmetic-gate.md.
#![allow(clippy::arithmetic_side_effects)]

use std::collections::BTreeMap;

use crate::merge_policy::MergePolicyConfig;

use lucene_codecs::field_infos::{FieldInfo, IndexOptions};
use lucene_codecs::stored_fields::{self, FieldValue, StoredField};
use lucene_store::FsDirectory;
use lucene_util::test_support::TempDir;

use super::*;
use crate::index_writer::Error;
use crate::segment_info::LuceneVersion;
use crate::{deletes, segment_info, segment_infos};

fn fields() -> Vec<FieldInfo> {
    vec![
        FieldInfo {
            index_options: IndexOptions::Docs,
            omit_norms: true,
            ..FieldInfo::new("id", 0)
        },
        FieldInfo {
            index_options: IndexOptions::DocsAndFreqs,
            ..FieldInfo::new("body", 1)
        },
    ]
}

fn doc(id: &str, version: u32) -> Document {
    Document {
        fields: vec![
            StoredField {
                field_number: 0,
                value: FieldValue::String(id.to_string()),
            },
            StoredField {
                field_number: 1,
                value: FieldValue::String(format!("v{version} common")),
            },
        ],
    }
}

fn writer(dir: &FsDirectory, max_buffered_docs: i32) -> IndexWriter<'_> {
    let mut w = IndexWriter::open(
        dir,
        fields(),
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
    w.set_max_buffered_docs(max_buffered_docs).unwrap();
    w.set_ram_buffer_size_mb(DISABLE_AUTO_FLUSH_MB).unwrap();
    w.set_merge_policy(Some(MergePolicyConfig {
        max_merge_at_once: 4,
        segments_per_tier: 3.0,
        floor_segment_size: 1 << 30,
        ..MergePolicyConfig::default()
    }));
    // These tests drive merges themselves; merge-on-commit has its own.
    w.set_max_full_flush_merge_wait_millis(0);
    w
}

use crate::index_writer::{DEFAULT_MAX_FULL_FLUSH_MERGE_WAIT_MILLIS, DISABLE_AUTO_FLUSH_MB};

/// id -> body of every live document of the latest commit, read through
/// the stored fields and `.liv` files -- plus the segment count.
fn live_documents(dir: &FsDirectory) -> (BTreeMap<String, String>, usize) {
    let infos = segment_infos::read_latest(dir).unwrap();
    let mut out = BTreeMap::new();
    for sci in &infos.segments {
        let name = &sci.segment_name;
        let si = segment_info::parse(&dir.open(&format!("{name}.si")).unwrap(), &sci.segment_id)
            .unwrap();
        let fdt = dir.open(&format!("{name}.fdt")).unwrap();
        let fdx = dir.open(&format!("{name}.fdx")).unwrap();
        let fdm = dir.open(&format!("{name}.fdm")).unwrap();
        let reader = stored_fields::open(&fdt, &fdx, &fdm, &sci.segment_id, "").unwrap();
        let live = (sci.del_gen >= 0).then(|| {
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
            if live.as_ref().is_some_and(|l| !l.get(d as usize)) {
                continue;
            }
            let stored = reader.document(d).unwrap();
            let text = |n: i32| match &stored
                .fields
                .iter()
                .find(|f| f.field_number == n)
                .unwrap()
                .value
            {
                FieldValue::String(s) => s.clone(),
                other => panic!("{other:?}"),
            };
            if let Some(previous) = out.insert(text(0), text(1)) {
                panic!(
                    "id {} is live twice: {previous:?} and {:?} (segment {name}, doc {d})",
                    text(0),
                    text(1)
                );
            }
        }
    }
    (out, infos.segments.len())
}

fn assert_clean(dir: &FsDirectory) {
    for result in crate::check_index::check_directory(dir).unwrap() {
        assert!(result.all_passed(), "{:?}", result.failures());
    }
}

#[test]
fn many_threads_add_while_merges_run() {
    let tmp = TempDir::new("concurrent-add");
    let dir = FsDirectory::open(&tmp);
    let w = ConcurrentIndexWriter::new(writer(&dir, 7), 4).unwrap();
    let stop = AtomicBool::new(false);
    let merges = std::thread::scope(|scope| {
        let merger = scope.spawn(|| w.run_merges(&stop).unwrap());
        let indexers: Vec<_> = (0..4)
            .map(|t| {
                let w = &w;
                scope.spawn(move || {
                    for i in 0..500 {
                        w.add_document(doc(&format!("t{t}x{i}"), 0)).unwrap();
                        if i % 97 == 0 {
                            w.commit().unwrap();
                        }
                    }
                })
            })
            .collect();
        for h in indexers {
            h.join().unwrap();
        }
        w.commit().unwrap();
        stop.store(true, Ordering::Release);
        merger.join().unwrap()
    });
    // However the merge thread was scheduled, one more round runs here.
    let merges = merges + w.maybe_merge().unwrap();
    w.commit().unwrap();
    let (docs, segments) = live_documents(&dir);
    assert_eq!(docs.len(), 2000);
    assert!(merges > 0, "nothing was merged");
    assert!(
        segments < 2000 / 7,
        "{segments} segments: nothing was merged"
    );
    assert_clean(&dir);
}

#[test]
fn updates_and_deletes_from_many_threads_leave_exactly_the_last_versions() {
    let tmp = TempDir::new("concurrent-update");
    let dir = FsDirectory::open(&tmp);
    let w = ConcurrentIndexWriter::new(writer(&dir, 5), 3).unwrap();
    let stop = AtomicBool::new(false);
    // Each thread owns its ids, so the expected end state is exact: id k
    // of thread t is updated k % 4 times, and deleted when k % 5 == 0.
    std::thread::scope(|scope| {
        let merger = scope.spawn(|| w.run_merges(&stop).unwrap());
        let threads: Vec<_> = (0..3)
            .map(|t| {
                let w = &w;
                scope.spawn(move || {
                    for k in 0..150u32 {
                        let id = format!("t{t}x{k}");
                        w.add_document(doc(&id, 0)).unwrap();
                        for v in 1..=(k % 4) {
                            w.update_document(
                                Term::new("id", id.clone().into_bytes()),
                                doc(&id, v),
                            )
                            .unwrap();
                        }
                        if k % 5 == 0 {
                            w.delete_documents_by_term(&[Term::new("id", id.into_bytes())])
                                .unwrap();
                        }
                        if k % 41 == 0 {
                            w.commit().unwrap();
                        }
                    }
                })
            })
            .collect();
        for h in threads {
            h.join().unwrap();
        }
        w.commit().unwrap();
        stop.store(true, Ordering::Release);
        merger.join().unwrap();
    });
    w.commit().unwrap();
    let (docs, _) = live_documents(&dir);
    let mut expected = BTreeMap::new();
    for t in 0..3 {
        for k in 0..150u32 {
            if k % 5 != 0 {
                expected.insert(format!("t{t}x{k}"), format!("v{} common", k % 4));
            }
        }
    }
    assert_eq!(docs, expected);
    assert_clean(&dir);
}

#[test]
fn a_thread_goes_back_to_the_slot_it_last_filled() {
    // The main thread holds slot 0, so the other thread's first add takes
    // slot 1; with every slot free again, its next add goes back there
    // rather than to the lowest free slot.
    let tmp = TempDir::new("concurrent-affinity");
    let dir = FsDirectory::open(&tmp);
    let w = ConcurrentIndexWriter::new(writer(&dir, 1000), 2).unwrap();
    let (added, released) = (std::sync::mpsc::channel(), std::sync::mpsc::channel());
    let mut held = Some(write_slot(&w.slots[0]));
    std::thread::scope(|s| {
        let (added_tx, released_rx, w) = (added.0, released.1, &w);
        s.spawn(move || {
            w.add_document(doc("a", 0)).unwrap();
            added_tx.send(()).unwrap();
            released_rx.recv().unwrap();
            w.add_document(doc("b", 1)).unwrap();
        });
        added.1.recv().unwrap();
        held.take();
        released.0.send(()).unwrap();
    });
    assert_eq!(write_slot(&w.slots[0]).num_docs(), 0);
    assert_eq!(write_slot(&w.slots[1]).num_docs(), 2);
}

#[test]
fn a_delete_reaches_a_segment_that_was_being_built_when_it_was_issued() {
    // Two slots: slot A's buffer is ticketed (taken for a segment) before
    // the delete, and published after it. The delete must still reach it.
    let tmp = TempDir::new("concurrent-inflight");
    let dir = FsDirectory::open(&tmp);
    let w = ConcurrentIndexWriter::new(writer(&dir, 1000), 2).unwrap();
    w.add_document(doc("a", 0)).unwrap();
    let (i, mut dwpt) = w.acquire_slot();
    let batch = w.begin_flush(i, &mut dwpt);
    drop(dwpt);
    let batch = batch.or_else(|| {
        // The document went to the other slot.
        let j = 1 - i;
        let mut other = write_slot(&w.slots[j]);
        w.begin_flush(j, &mut other)
    });
    let batch = batch.expect("one slot held the document");
    w.delete_documents_by_term(&[Term::new("id", b"a".to_vec())])
        .unwrap();
    w.complete_flush(batch).unwrap();
    w.commit().unwrap();
    let (docs, _) = live_documents(&dir);
    assert!(docs.is_empty(), "{docs:?}");
    assert_clean(&dir);
}

/// A commit is a clean cut through concurrent updates: an update's delete
/// and its new document are both in a commit or both out of it, so every
/// id is live exactly once in every commit.
#[test]
fn every_commit_holds_each_updated_id_exactly_once() {
    let tmp = TempDir::new("concurrent-cut");
    let dir = FsDirectory::open(&tmp);
    let w = ConcurrentIndexWriter::new(writer(&dir, 3), 3).unwrap();
    for k in 0..30 {
        w.add_document(doc(&format!("u{k}"), 0)).unwrap();
    }
    w.commit().unwrap();
    let stop = AtomicBool::new(false);
    std::thread::scope(|scope| {
        for t in 0..2u32 {
            let (w, stop) = (&w, &stop);
            scope.spawn(move || {
                let mut v = 1;
                while !stop.load(Ordering::Acquire) {
                    for k in (t..30).step_by(2) {
                        let id = format!("u{k}");
                        w.update_document(Term::new("id", id.clone().into_bytes()), doc(&id, v))
                            .unwrap();
                    }
                    v += 1;
                }
            });
        }
        // Stops the updaters however this loop ends, so a failed
        // assertion fails the test instead of hanging it.
        let _stop = StopOnDrop(&stop);
        for _ in 0..40 {
            w.commit().unwrap();
            // Panics on an id live twice.
            let (docs, _) = live_documents(&dir);
            assert_eq!(docs.len(), 30, "an update was cut in half: {docs:?}");
        }
    });
    w.commit().unwrap();
    assert_clean(&dir);
}

struct StopOnDrop<'a>(&'a AtomicBool);

impl Drop for StopOnDrop<'_> {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

/// Two segments in flight, a delete issued between their tickets: the
/// first segment gets it only through the *second* ticket's frozen
/// packet, pushed when that one publishes -- after the first, although
/// the second finished building first.
#[test]
fn a_delete_between_two_tickets_reaches_the_first_through_the_second() {
    let tmp = TempDir::new("concurrent-two-tickets");
    let dir = FsDirectory::open(&tmp);
    let w = ConcurrentIndexWriter::new(writer(&dir, 1000), 2).unwrap();
    write_slot(&w.slots[0]).docs.push(doc("a", 0));
    write_slot(&w.slots[1]).docs.push(doc("b", 0));
    let first = w.begin_flush(0, &mut write_slot(&w.slots[0])).unwrap();
    w.delete_documents_by_term(&[Term::new("id", b"a".to_vec())])
        .unwrap();
    w.delete_documents_by_term(&[Term::new("id", b"b".to_vec())])
        .unwrap();
    let second = w.begin_flush(1, &mut write_slot(&w.slots[1])).unwrap();
    std::thread::scope(|scope| {
        let later = scope.spawn(|| w.complete_flush(second));
        // Let the second build finish and wait for its turn.
        std::thread::sleep(Duration::from_millis(50));
        w.complete_flush(first).unwrap();
        later.join().unwrap().unwrap();
    });
    // Published, not yet committed: both deletes already applied.
    {
        let core = lock(&w.core);
        let live = core.writer.live_infos();
        assert!(
            live.segments.is_empty(),
            "both segments are fully deleted and dropped"
        );
    }
    w.add_document(doc("c", 0)).unwrap();
    w.commit().unwrap();
    let (docs, _) = live_documents(&dir);
    assert_eq!(docs.keys().collect::<Vec<_>>(), ["c"]);
    assert_clean(&dir);
}

fn two_committed_segments(dir: &FsDirectory) -> ConcurrentIndexWriter<'_> {
    let mut single = sorted_writer(dir);
    single.set_index_sort(None).unwrap();
    single.set_merge_policy(Some(MergePolicyConfig {
        max_merge_at_once: 2,
        segments_per_tier: 2.0,
        floor_segment_size: 1 << 30,
        ..MergePolicyConfig::default()
    }));
    single.set_max_full_flush_merge_wait_millis(0);
    let w = ConcurrentIndexWriter::new(single, 1).unwrap();
    for (id, rank) in [("m0", 1), ("m1", 2)] {
        w.add_document(ranked(id, 0, rank)).unwrap();
    }
    w.commit().unwrap();
    for (id, rank) in [("m2", 3), ("m3", 4)] {
        w.add_document(ranked(id, 0, rank)).unwrap();
    }
    w.commit().unwrap();
    w.add_document(ranked("m4", 0, 5)).unwrap();
    w.commit().unwrap();
    w
}

/// `commitMergedDeletesAndUpdates`: a delete published on a source while
/// the merge ran is carried onto the merged segment through the merge's
/// doc map.
#[test]
fn a_delete_made_during_a_merge_is_carried_onto_the_merged_segment() {
    let tmp = TempDir::new("concurrent-merge-carry");
    let dir = FsDirectory::open(&tmp);
    let w = two_committed_segments(&dir);
    let plan = {
        let mut core = lock(&w.core);
        let names = core
            .writer
            .next_merge(
                &core.writer.merge_policy().cloned().unwrap(),
                &HashSet::new(),
            )
            .unwrap()
            .unwrap();
        core.writer.begin_merge(&names).unwrap()
    };
    let outcome = w.cfg().run_merge(&dir, &plan).unwrap();
    // Meanwhile: deletes reach the first two segments -- at least one of
    // them a source, whichever the policy chose -- and are published
    // there. (Emptying a source would abandon the merge instead.)
    let sources = plan.names.len();
    for id in ["m0", "m2"] {
        w.delete_documents_by_term(&[Term::new("id", id.as_bytes().to_vec())])
            .unwrap();
    }
    w.flush().unwrap();
    let generation = segment_infos::read_latest(&dir).unwrap().generation;
    assert!(lock(&w.core).writer.finish_merge(plan, outcome).unwrap());
    // `commitMerge`: published in memory; nothing durable changes until
    // the next commit.
    assert_eq!(
        segment_infos::read_latest(&dir).unwrap().generation,
        generation
    );
    w.commit().unwrap();
    let (docs, segments) = live_documents(&dir);
    assert!(segments <= 3 - sources + 1, "{segments} segments");
    assert_eq!(docs.keys().collect::<Vec<_>>(), ["m1", "m3", "m4"]);
    assert_clean(&dir);
}

/// A doc-values update on a source while its merge ran is not carried:
/// the merge is abandoned, its files deleted, the sources kept.
#[test]
fn a_doc_values_update_during_a_merge_abandons_it() {
    let tmp = TempDir::new("concurrent-merge-abandon");
    let dir = FsDirectory::open(&tmp);
    let w = two_committed_segments(&dir);
    let plan = {
        let mut core = lock(&w.core);
        let names = core
            .writer
            .next_merge(
                &core.writer.merge_policy().cloned().unwrap(),
                &HashSet::new(),
            )
            .unwrap()
            .unwrap();
        core.writer.begin_merge(&names).unwrap()
    };
    let merged_name = plan.merged_name.clone();
    let outcome = w.cfg().run_merge(&dir, &plan).unwrap();
    for id in ["m0", "m2", "m4"] {
        w.update_numeric_doc_value(Term::new("id", id.as_bytes().to_vec()), "score", 7)
            .unwrap();
    }
    w.flush().unwrap();
    assert!(!lock(&w.core).writer.finish_merge(plan, outcome).unwrap());
    let leftovers: Vec<String> = dir
        .list_all()
        .unwrap()
        .into_iter()
        .filter(|f| {
            f.starts_with(&format!("{merged_name}.")) || f.starts_with(&format!("{merged_name}_"))
        })
        .collect();
    assert!(leftovers.is_empty(), "{leftovers:?}");
    w.commit().unwrap();
    let (docs, segments) = live_documents(&dir);
    assert_eq!(segments, 3);
    assert_eq!(docs.len(), 5);
    assert_clean(&dir);
}

/// A writer over `id`, `body`, `rank` and `score` (NUMERIC doc values),
/// sorted by `rank` ascending, so a flush permutes its buffer and every
/// private delete's limit has to go through the sort map.
fn sorted_writer(dir: &FsDirectory) -> IndexWriter<'_> {
    let mut fields = fields();
    fields.push(FieldInfo {
        doc_values_type: lucene_codecs::field_infos::DocValuesType::Numeric,
        ..FieldInfo::new("rank", 2)
    });
    fields.push(FieldInfo {
        doc_values_type: lucene_codecs::field_infos::DocValuesType::Numeric,
        ..FieldInfo::new("score", 3)
    });
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
    w.set_doc_values_field(Some("rank")).unwrap();
    w.add_doc_values_field("score").unwrap();
    w.set_index_sort(Some(&[segment_info::IndexSortField::long(
        "rank",
        false,
        Some(i64::MAX),
    )]))
    .unwrap();
    w.set_max_buffered_docs(1000).unwrap();
    w.set_ram_buffer_size_mb(DISABLE_AUTO_FLUSH_MB).unwrap();
    w
}

fn ranked(id: &str, version: u32, rank: i64) -> Document {
    let mut d = doc(id, version);
    d.fields.push(StoredField {
        field_number: 2,
        value: FieldValue::Long(rank),
    });
    d.fields.push(StoredField {
        field_number: 3,
        value: FieldValue::Long(rank * 100),
    });
    d
}

/// A buffer's own deletes of every kind, resolved by the flushing thread
/// before the publish: term deletes against the terms just inverted,
/// query deletes and doc-values updates against the written segment --
/// all through the sort map, since the segment is sorted by `rank`.
#[test]
fn a_flush_resolves_its_own_deletes_and_updates_through_the_sort() {
    let tmp = TempDir::new("concurrent-sorted");
    let dir = FsDirectory::open(&tmp);
    let w = ConcurrentIndexWriter::new(sorted_writer(&dir), 1).unwrap();
    w.add_document(ranked("a", 0, 3)).unwrap();
    w.add_document(ranked("b", 0, 1)).unwrap();
    w.add_document(ranked("c", 0, 2)).unwrap();
    w.delete_documents_by_query(&[DeleteQuery::Term(Term::new("id", b"a".to_vec()))])
        .unwrap();
    w.update_numeric_doc_value(Term::new("id", b"b".to_vec()), "score", 10)
        .unwrap();
    w.update_document(Term::new("id", b"c".to_vec()), ranked("c", 1, 0))
        .unwrap();
    // After the delete: survives it.
    w.add_document(ranked("a", 1, 5)).unwrap();
    w.commit().unwrap();

    let (docs, segments) = live_documents(&dir);
    assert_eq!(segments, 1);
    let want: BTreeMap<String, String> = [("a", 1), ("b", 0), ("c", 1)]
        .into_iter()
        .map(|(id, v)| (id.to_string(), format!("v{v} common")))
        .collect();
    assert_eq!(docs, want);
    let infos = segment_infos::read_latest(&dir).unwrap();
    let scores = crate::index_writer::tests::read_numeric_column(&dir, &infos.segments[0], 3);
    // Sorted by rank (c1=0, b=1, c0=2, a0=3, a1=5), each score 100 x its
    // rank; only b's changed.
    assert_eq!(scores, [Some(0), Some(10), Some(200), Some(300), Some(500)]);
    assert_clean(&dir);
}

/// `DocumentsWriterPerThread.updateDocuments`: a block in a sorted index
/// without a parent field is refused before anything is buffered, and
/// the writer goes on (a buffered one would fail every later flush).
#[test]
fn a_block_in_a_sorted_index_without_a_parent_field_is_refused_at_add() {
    let tmp = TempDir::new("concurrent-sorted-block");
    let dir = FsDirectory::open(&tmp);
    let w = ConcurrentIndexWriter::new(sorted_writer(&dir), 2).unwrap();
    w.add_document(ranked("a", 0, 2)).unwrap();
    let before = w.pending_doc_count();
    assert!(matches!(
        w.add_documents(vec![ranked("b", 0, 1), ranked("c", 0, 3)]),
        Err(Error::BlocksWithIndexSortNeedParentField)
    ));
    assert_eq!(w.pending_doc_count(), before);
    w.add_documents(vec![ranked("d", 0, 1)]).unwrap();
    w.commit().unwrap();
    let (docs, _) = live_documents(&dir);
    assert_eq!(docs.keys().collect::<Vec<_>>(), ["a", "d"]);
    assert_clean(&dir);
}

/// A writer with a parent field and an index sort: each add marks its
/// last document as the parent, and a flush orders whole blocks by
/// their parent's key, as Java's `IndexingChain.maybeSortSegment` does.
#[test]
fn explicit_blocks_are_sorted_whole_by_their_parent() {
    use crate::index_writer::{ExplicitDocument, ExplicitFields};
    use lucene_codecs::field_infos::{DocValuesSkipIndexType, DocValuesType};
    use lucene_codecs::stored_fields::{self, FieldValue, StoredField};
    let tmp = TempDir::new("concurrent-parent-sort");
    let dir = FsDirectory::open(&tmp);
    let mut w = IndexWriter::open(
        &dir,
        Vec::new(),
        "Lucene104",
        LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        },
    )
    .unwrap();
    w.set_parent_field(Some("_parent")).unwrap();
    let id = w.register_field(FieldInfo::new("id", 0)).unwrap();
    let v = w
        .register_field(FieldInfo::new("v", 0).with_doc_values(
            DocValuesType::Numeric,
            DocValuesSkipIndexType::None,
            -1,
        ))
        .unwrap();
    w.set_index_sort(Some(&[segment_info::IndexSortField::long(
        "v", false, None,
    )]))
    .unwrap();
    let w = ConcurrentIndexWriter::new(w, 2).unwrap();
    let doc = |name: &str, key: Option<i64>| ExplicitDocument {
        stored: vec![StoredField {
            field_number: id,
            value: FieldValue::String(name.to_string()),
        }],
        fields: ExplicitFields {
            doc_values: key
                .map(|k| StoredField {
                    field_number: v,
                    value: FieldValue::Long(k),
                })
                .into_iter()
                .collect(),
            ..ExplicitFields::default()
        },
    };
    w.add_explicit_documents(vec![doc("c5", None), doc("b5", Some(5))])
        .unwrap();
    w.add_explicit_documents(vec![doc("s1", Some(1))]).unwrap();
    w.add_explicit_documents(vec![doc("c3a", None), doc("c3b", None), doc("b3", Some(3))])
        .unwrap();
    w.commit().unwrap();
    let infos = segment_infos::read_latest(&dir).unwrap();
    assert_eq!(infos.segments.len(), 1);
    let sci = &infos.segments[0];
    let name = &sci.segment_name;
    let si =
        segment_info::parse(&dir.open(&format!("{name}.si")).unwrap(), &sci.segment_id).unwrap();
    assert!(si.has_blocks);
    assert!(si.index_sort.is_some());
    let fnm = lucene_codecs::field_infos::parse(
        &dir.open(&format!("{name}.fnm")).unwrap(),
        &sci.segment_id,
        "",
    )
    .unwrap();
    assert_eq!(fnm.parent_field(), Some("_parent"));
    let fdt = dir.open(&format!("{name}.fdt")).unwrap();
    let fdx = dir.open(&format!("{name}.fdx")).unwrap();
    let fdm = dir.open(&format!("{name}.fdm")).unwrap();
    let reader = stored_fields::open(&fdt, &fdx, &fdm, &sci.segment_id, "").unwrap();
    let order: Vec<String> = (0..si.doc_count)
        .map(|d| match &reader.document(d).unwrap().fields[0].value {
            FieldValue::String(s) => s.clone(),
            other => panic!("{other:?}"),
        })
        .collect();
    assert_eq!(order, ["s1", "c3a", "c3b", "b3", "c5", "b5"]);
    assert_clean(&dir);
}

/// `DocumentsWriterPerThread.abort`: a buffer whose segment cannot be
/// built loses its documents and leaves no file behind, the published
/// segments still get the deletes its ticket froze, and the writer goes
/// on.
#[test]
fn a_failed_build_is_abandoned_and_the_writer_goes_on() {
    let tmp = TempDir::new("concurrent-abort");
    let dir = FsDirectory::open(&tmp);
    let mut single = sorted_writer(&dir);
    single.set_merge_policy(Some(MergePolicyConfig {
        max_merge_at_once: 2,
        segments_per_tier: 2.0,
        floor_segment_size: 1 << 30,
        ..MergePolicyConfig::default()
    }));
    single.set_max_buffered_docs(2).unwrap();
    let w = ConcurrentIndexWriter::new(single, 1).unwrap();
    w.add_document(ranked("old", 0, 1)).unwrap();
    w.add_document(ranked("kept", 0, 1)).unwrap();
    w.commit().unwrap();
    w.add_document(ranked("also", 0, 1)).unwrap();
    w.commit().unwrap();
    // The delete is frozen into the next ticket, whose build then fails:
    // a document whose sort value is not a number fills the buffer
    // (`max_buffered_docs` is 2) and the automatic flush cannot sort it.
    w.delete_documents_by_term(&[Term::new("id", b"old".to_vec())])
        .unwrap();
    w.add_document(ranked("p", 0, 2)).unwrap();
    let mut bad = doc("q", 0);
    bad.fields.push(StoredField {
        field_number: 2,
        value: FieldValue::String("not a number".to_string()),
    });
    assert!(matches!(
        w.add_document(bad),
        Err(Error::NonNumericDocValue(..))
    ));
    let failed: Vec<String> = dir
        .list_all()
        .unwrap()
        .into_iter()
        .filter(|f| f.starts_with("_2"))
        .collect();
    assert!(failed.is_empty(), "the failed build left {failed:?}");
    // Nothing is left pending for a merge's segment to take by mistake
    // (`begin_merge` asserts the stream is empty).
    assert_eq!(w.maybe_merge().unwrap(), 1);
    w.add_document(ranked("r", 0, 4)).unwrap();
    w.commit().unwrap();
    let (docs, _) = live_documents(&dir);
    assert_eq!(docs.keys().collect::<Vec<_>>(), ["also", "kept", "r"]);
    assert_clean(&dir);
}

/// A directory whose `create_output` waits while its gate is shut: holds
/// a flush in the middle of its build.
struct GatedDirectory<'a> {
    inner: &'a FsDirectory,
    open: Mutex<bool>,
    opened: Condvar,
}

impl GatedDirectory<'_> {
    fn set_open(&self, open: bool) {
        *lock(&self.open) = open;
        self.opened.notify_all();
    }
}

impl Directory for GatedDirectory<'_> {
    fn list_all(&self) -> lucene_store::Result<Vec<String>> {
        self.inner.list_all()
    }
    fn open(&self, name: &str) -> lucene_store::Result<Input> {
        self.inner.open(name)
    }
    fn create_output(&self, name: &str) -> lucene_store::Result<FsIndexOutput> {
        let mut open = lock(&self.open);
        while !*open {
            open = self.opened.wait(open).unwrap();
        }
        drop(open);
        self.inner.create_output(name)
    }
    fn sync(&self, names: &[String]) -> lucene_store::Result<()> {
        self.inner.sync(names)
    }
    fn rename(&self, source: &str, dest: &str) -> lucene_store::Result<()> {
        self.inner.rename(source, dest)
    }
    fn delete_file(&self, name: &str) -> lucene_store::Result<()> {
        self.inner.delete_file(name)
    }
    fn sync_meta_data(&self) -> lucene_store::Result<()> {
        self.inner.sync_meta_data()
    }
    fn obtain_lock(&self, name: &str) -> lucene_store::Result<Box<dyn lucene_store::Lock>> {
        self.inner.obtain_lock(name)
    }
}

/// A directory that records every name it syncs.
struct SyncLogDirectory<'a> {
    inner: &'a FsDirectory,
    synced: Mutex<Vec<String>>,
}

impl Directory for SyncLogDirectory<'_> {
    fn list_all(&self) -> lucene_store::Result<Vec<String>> {
        self.inner.list_all()
    }
    fn open(&self, name: &str) -> lucene_store::Result<Input> {
        self.inner.open(name)
    }
    fn create_output(&self, name: &str) -> lucene_store::Result<FsIndexOutput> {
        self.inner.create_output(name)
    }
    fn sync(&self, names: &[String]) -> lucene_store::Result<()> {
        self.inner.sync(names)?;
        lock(&self.synced).extend(names.iter().cloned());
        Ok(())
    }
    fn rename(&self, source: &str, dest: &str) -> lucene_store::Result<()> {
        self.inner.rename(source, dest)
    }
    fn delete_file(&self, name: &str) -> lucene_store::Result<()> {
        self.inner.delete_file(name)
    }
    fn sync_meta_data(&self) -> lucene_store::Result<()> {
        self.inner.sync_meta_data()
    }
    fn obtain_lock(&self, name: &str) -> lucene_store::Result<Box<dyn lucene_store::Lock>> {
        self.inner.obtain_lock(name)
    }
}

/// The concurrent writer's flushes run on many threads but publish
/// through the one inner writer, whose deleter tracks what is synced: no
/// flush syncs anything, the commit syncs each file it names once, and a
/// commit with nothing new writes no generation and syncs nothing.
#[test]
fn concurrent_flushes_sync_nothing_and_the_commit_syncs_each_file_once() {
    let tmp = TempDir::new("concurrent-sync-at-commit");
    let fs = FsDirectory::open(&tmp);
    let dir = SyncLogDirectory {
        inner: &fs,
        synced: Mutex::new(Vec::new()),
    };
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut single = IndexWriter::open(&dir, fields(), "Lucene104", version).unwrap();
    single.set_postings_field(Some("id")).unwrap();
    single.add_postings_field("body").unwrap();
    single.set_max_buffered_docs(5).unwrap();
    single
        .set_ram_buffer_size_mb(DISABLE_AUTO_FLUSH_MB)
        .unwrap();
    let w = ConcurrentIndexWriter::new(single, 3).unwrap();
    std::thread::scope(|scope| {
        for t in 0..3 {
            let w = &w;
            scope.spawn(move || {
                for k in 0..20 {
                    w.add_document(doc(&format!("t{t}k{k:02}"), 0)).unwrap();
                }
            });
        }
    });
    w.flush().unwrap();
    assert!(lock(&dir.synced).is_empty(), "a flush synced files");

    w.commit().unwrap();
    let infos = segment_infos::read_latest(&fs).unwrap();
    assert!(infos.segments.len() > 1);
    let mut named = crate::index_file_deleter::files_of_infos(&fs, &infos, false).unwrap();
    named.sort();
    let mut synced: Vec<String> = lock(&dir.synced)
        .iter()
        .filter(|n| !n.contains("segments"))
        .cloned()
        .collect();
    synced.sort();
    assert_eq!(synced, named, "each named file synced exactly once");

    let before = lock(&dir.synced).len();
    w.commit().unwrap();
    assert_eq!(
        segment_infos::read_latest(&fs).unwrap().generation,
        infos.generation,
        "nothing changed, so no new generation"
    );
    assert_eq!(lock(&dir.synced).len(), before);
    let (docs, _) = live_documents(&fs);
    assert_eq!(docs.len(), 60);
}

/// Waits (bounded) until `cond` holds.
fn eventually(what: &str, cond: impl Fn() -> bool) {
    let start = std::time::Instant::now();
    while !cond() {
        assert!(
            start.elapsed() < std::time::Duration::from_secs(20),
            "never: {what}"
        );
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}

/// `DocumentsWriterFlushControl.updateStallState` +
/// `DocumentsWriter.preUpdate`: while flushes are held mid-build and the
/// slots keep filling, RAM in slots and flushes passes twice the buffer
/// and indexing stalls; the other threads' adds wait, and go on once the
/// flushes complete and lift the stall. No document is lost.
///
/// Each flush is taken by the thread whose add reached the RAM buffer
/// (the largest slot, `FlushByRamOrCountsPolicy`) and holds that thread
/// at the gate, so three held flushes of about four documents each plus
/// nine documents in the slots pass the twenty-document stall limit
/// while the slots alone stay under it.
#[test]
fn indexing_stalls_while_flushing_falls_behind() {
    let tmp = TempDir::new("concurrent-stall");
    let fs = FsDirectory::open(&tmp);
    let gated = GatedDirectory {
        inner: &fs,
        open: Mutex::new(true),
        opened: Condvar::new(),
    };
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut single = IndexWriter::open(&gated, fields(), "Lucene104", version).unwrap();
    single.set_postings_field(Some("id")).unwrap();
    single.add_postings_field("body").unwrap();
    single.set_max_buffered_docs(DISABLE_AUTO_FLUSH).unwrap();
    // Ten documents fill a slot.
    let doc_bytes = document_ram_bytes(&doc("t1x00", 0));
    let mb = (doc_bytes * 10) as f64 / (1024.0 * 1024.0);
    single.set_ram_buffer_size_mb(mb).unwrap();
    let w = ConcurrentIndexWriter::new(single, 3).unwrap();
    assert!(w.stall_control().is_healthy());

    gated.set_open(false);
    /// Opens the gate however the scope ends: a failed assertion must
    /// not leave a thread blocked in `create_output` for the scope to
    /// wait on forever.
    struct OpenOnDrop<'a, 'b>(&'a GatedDirectory<'b>);
    impl Drop for OpenOnDrop<'_, '_> {
        fn drop(&mut self) {
            self.0.set_open(true);
        }
    }
    std::thread::scope(|scope| {
        let _open = OpenOnDrop(&gated);
        let fillers: Vec<_> = (0..4)
            .map(|t| {
                let w = &w;
                scope.spawn(move || {
                    for k in 0..30 {
                        w.add_document(doc(&format!("t{t}x{k:02}"), 0)).unwrap();
                    }
                })
            })
            .collect();
        eventually("the held flushes stall indexing", || {
            w.stall_control().any_stalled_threads()
        });
        eventually("a thread waits on the stall", || {
            w.stall_control().has_blocked()
        });
        assert!(fillers.iter().any(|f| !f.is_finished()));
        gated.set_open(true);
        for f in fillers {
            f.join().unwrap();
        }
    });
    assert!(w.stall_control().was_stalled());
    assert!(
        w.stall_control().is_healthy(),
        "the flushes lifted the stall"
    );
    w.commit().unwrap();
    let (docs, _) = live_documents(&fs);
    assert_eq!(docs.len(), 120);
}

/// A concurrent writer flushing by RAM over three slots, the buffer
/// `docs` documents of `doc("t1x00", 0)`'s size, and no document limit.
fn ram_writer(dir: &FsDirectory, docs: usize) -> ConcurrentIndexWriter<'_> {
    let mut single = writer(dir, 1_000_000);
    let doc_bytes = document_ram_bytes(&doc("t1x00", 0));
    single
        .set_ram_buffer_size_mb((doc_bytes * docs) as f64 / (1024.0 * 1024.0))
        .unwrap();
    single.set_max_buffered_docs(DISABLE_AUTO_FLUSH).unwrap();
    ConcurrentIndexWriter::new(single, 3).unwrap()
}

/// `FlushByRamOrCountsPolicy.markLargestWriterPending`: the RAM buffer
/// bounds every slot together, and reaching it flushes the slot holding
/// the most, not the one the last document went into. One thread fills
/// the slots in turn (4, 3, 3 documents): the tenth document fills the
/// buffer and the first slot, the largest, is flushed.
#[test]
fn the_ram_buffer_bounds_all_slots_and_flushes_the_largest() {
    let tmp = TempDir::new("concurrent-largest");
    let dir = FsDirectory::open(&tmp);
    let w = ram_writer(&dir, 10);
    w.rotate_slots.store(true, Ordering::Relaxed);
    for k in 0..9 {
        w.add_document(doc(&format!("t1x{k:02}"), 0)).unwrap();
    }
    assert_eq!(w.pending_doc_count(), 9);
    w.add_document(doc("t1x09", 0)).unwrap();
    assert_eq!(w.pending_doc_count(), 6, "the four-document slot flushed");
    let counts: Vec<usize> = w.slots.iter().map(|s| read_slot(s).num_docs()).collect();
    assert_eq!(counts, [0, 3, 3]);
    let ram = lock(&w.ram);
    assert_eq!(ram.flush_bytes(), 0, "the flush is done");
    assert_eq!(ram.num_pending(), 0);
}

/// `FlushByRamOrCountsPolicy`: a marked slot the add did not go into is
/// flushed by the adding thread (`postUpdate`'s `maybeFlush`) while
/// `checkPendingFlushOnUpdate` is on; with it off the slot stays marked
/// until a thread indexes into it.
#[test]
fn a_marked_slot_is_flushed_by_whoever_checks_pending_flushes() {
    for check in [true, false] {
        let tmp = TempDir::new("concurrent-pending");
        let dir = FsDirectory::open(&tmp);
        let w = ram_writer(&dir, 10);
        w.rotate_slots.store(true, Ordering::Relaxed);
        w.set_check_pending_flush_on_update(check);
        let big = Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String("big".into()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::String(
                        "x".repeat(document_ram_bytes(&doc("t1x00", 0)) * 15 / 2),
                    ),
                },
            ],
        };
        // Slots 0, 1, 2 in turn; the big document (about 8.5 small ones)
        // is the largest, and the buffer fills on the third add, into
        // slot 2.
        w.add_document(doc("a", 0)).unwrap();
        w.add_document(big.clone()).unwrap();
        w.add_document(doc("c", 0)).unwrap();
        let counts: Vec<usize> = w.slots.iter().map(|s| read_slot(s).num_docs()).collect();
        if check {
            assert_eq!(counts, [1, 0, 1], "the adding thread flushed slot 1");
            assert!(!w.any_pending());
        } else {
            assert_eq!(counts, [1, 1, 1], "slot 1 waits, marked");
            assert!(w.pending[1].load(Ordering::Acquire));
            // Slot 0 next: the marked RAM no longer counts, so nothing
            // more is marked, and slot 1 still waits.
            w.add_document(doc("d", 0)).unwrap();
            let counts: Vec<usize> = w.slots.iter().map(|s| read_slot(s).num_docs()).collect();
            assert_eq!(counts, [2, 1, 1]);
            assert!(w.pending[1].load(Ordering::Acquire));
            // Into slot 1: the thread indexing there flushes it.
            w.add_document(doc("e", 0)).unwrap();
            let counts: Vec<usize> = w.slots.iter().map(|s| read_slot(s).num_docs()).collect();
            assert_eq!(counts, [2, 0, 1]);
            assert!(!w.any_pending());
        }
        w.commit().unwrap();
        let (docs, _) = live_documents(&dir);
        assert_eq!(docs.len(), if check { 3 } else { 5 });
    }
}

/// `FlushByRamOrCountsPolicy.flushDeletes` + `DocumentsWriter
/// .applyAllDeletes`: once buffered deletes alone reach the RAM buffer
/// they are applied to the published segments at once -- their `.liv`
/// written before any flush or commit -- and stop counting.
#[test]
fn deletes_over_the_ram_buffer_are_applied_without_a_flush() {
    let tmp = TempDir::new("concurrent-delete-ram");
    let dir = FsDirectory::open(&tmp);
    let mut single = writer(&dir, 1_000_000);
    for k in 0..20 {
        single.add_document(doc(&format!("d{k:02}"), 0)).unwrap();
    }
    single.commit().unwrap();
    let node = DeleteNode::terms(vec![Term::new("id", "d00")]);
    // Three one-term deletes fill the buffer.
    single
        .set_ram_buffer_size_mb((node.ram_bytes() * 3) as f64 / (1024.0 * 1024.0))
        .unwrap();
    single.set_max_buffered_docs(DISABLE_AUTO_FLUSH).unwrap();
    let w = ConcurrentIndexWriter::new(single, 2).unwrap();
    let liv = || dir.list_all().unwrap().iter().any(|f| f.ends_with(".liv"));
    w.delete_documents_by_term(&[Term::new("id", "d00")])
        .unwrap();
    w.delete_documents_by_term(&[Term::new("id", "d01")])
        .unwrap();
    assert!(!liv(), "two deletes are under the buffer");
    assert_eq!(lock(&w.log).unhanded_bytes, node.ram_bytes() * 2);
    w.delete_documents_by_term(&[Term::new("id", "d02")])
        .unwrap();
    assert!(liv(), "the third applied all three");
    assert_eq!(lock(&w.log).unhanded_bytes, 0);
    assert_eq!(lock(&w.ram).delete_bytes_used(), 0);
    w.delete_documents_by_term(&[Term::new("id", "d03")])
        .unwrap();
    w.commit().unwrap();
    let (docs, _) = live_documents(&dir);
    assert_eq!(docs.len(), 16);
    assert!(!docs.contains_key("d02") && docs.contains_key("d04"));
}

/// With no RAM buffer (flushing by document count) nothing ever stalls.
#[test]
fn indexing_never_stalls_without_a_ram_buffer() {
    let tmp = TempDir::new("concurrent-no-stall");
    let dir = FsDirectory::open(&tmp);
    let w = ConcurrentIndexWriter::new(writer(&dir, 2), 2).unwrap();
    for k in 0..20 {
        w.add_document(doc(&format!("n{k}"), 0)).unwrap();
    }
    w.commit().unwrap();
    assert!(!w.stall_control().was_stalled());
}

/// `DocumentsWriterPerThread.flush`: a buffer its own deletes leave empty
/// never becomes a published segment.
#[test]
fn a_buffer_its_own_deletes_empty_is_not_published() {
    let tmp = TempDir::new("concurrent-empty");
    let dir = FsDirectory::open(&tmp);
    let w = ConcurrentIndexWriter::new(writer(&dir, 1000), 1).unwrap();
    w.add_document(doc("x", 0)).unwrap();
    w.delete_documents_by_term(&[Term::new("id", b"x".to_vec())])
        .unwrap();
    w.commit().unwrap();
    let (docs, segments) = live_documents(&dir);
    assert!(docs.is_empty());
    assert_eq!(segments, 0);
    let leftovers: Vec<String> = dir
        .list_all()
        .unwrap()
        .into_iter()
        .filter(|f| f.starts_with("_0"))
        .collect();
    assert!(leftovers.is_empty(), "{leftovers:?}");
}

/// Updates spread over several slots from one thread: each update's
/// delete has to reach the slot the old version went to.
#[test]
fn one_thread_updating_across_slots_leaves_the_last_versions() {
    let tmp = TempDir::new("concurrent-repro");
    let dir = FsDirectory::open(&tmp);
    let w = ConcurrentIndexWriter::new(writer(&dir, 5), 3).unwrap();
    let mut expected = BTreeMap::new();
    for k in 0..40u32 {
        let id = format!("k{k}");
        w.add_document(doc(&id, 0)).unwrap();
        for v in 1..=(k % 4) {
            w.update_document(Term::new("id", id.clone().into_bytes()), doc(&id, v))
                .unwrap();
        }
        expected.insert(id, format!("v{} common", k % 4));
    }
    w.commit().unwrap();
    let (docs, _) = live_documents(&dir);
    assert_eq!(docs, expected);
    assert_clean(&dir);
}

#[test]
fn a_delete_does_not_reach_a_document_added_after_it() {
    let tmp = TempDir::new("concurrent-order");
    let dir = FsDirectory::open(&tmp);
    let w = ConcurrentIndexWriter::new(writer(&dir, 1000), 2).unwrap();
    w.add_document(doc("x", 0)).unwrap();
    w.delete_documents_by_term(&[Term::new("id", b"x".to_vec())])
        .unwrap();
    let added = w.add_document(doc("x", 1)).unwrap();
    let committed = w.commit().unwrap();
    // `commit` names the last operation it holds.
    assert_eq!(committed, added);
    assert!(w.add_document(doc("y", 0)).unwrap() > committed);
    let (docs, _) = live_documents(&dir);
    assert_eq!(
        docs.into_iter().collect::<Vec<_>>(),
        [("x".to_string(), "v1 common".to_string())]
    );
    assert_clean(&dir);
}

/// `IndexWriter.deleteAll()` on the concurrent writer: committed,
/// flushed and buffered documents all go, deletes buffered before it
/// reach nothing added after it, a `MatchAllDocsQuery` delete is the
/// same operation, and a merge the scheduler had not started is dropped.
#[test]
fn the_max_docs_limit_holds_across_slots_flushes_and_delete_all() {
    let tmp = TempDir::new("concurrent-max-docs");
    let dir = FsDirectory::open(&tmp);
    let mut inner = writer(&dir, 3);
    inner.set_max_docs(7);
    inner.add_document(doc("a0", 0)).unwrap();
    let w = ConcurrentIndexWriter::new(inner, 2).unwrap();
    for k in 1..7 {
        w.add_document(doc(&format!("a{k}"), 0)).unwrap();
    }
    assert!(matches!(
        w.add_document(doc("a7", 0)),
        Err(crate::index_writer::Error::TooManyDocs(7))
    ));
    assert!(matches!(
        w.add_documents(vec![doc("b0", 0), doc("b1", 0)]),
        Err(crate::index_writer::Error::TooManyDocs(7))
    ));
    w.commit().unwrap();
    assert!(w.add_document(doc("a7", 0)).is_err());
    w.delete_all().unwrap();
    for k in 0..7 {
        w.add_document(doc(&format!("c{k}"), 0)).unwrap();
    }
    assert!(w.add_document(doc("c7", 0)).is_err());
}

#[test]
fn delete_all_drops_every_segment_buffer_and_delete() {
    let tmp = TempDir::new("concurrent-delete-all");
    let dir = FsDirectory::open(&tmp);
    let w = ConcurrentIndexWriter::new(writer(&dir, 3), 2).unwrap();
    for k in 0..10 {
        w.add_document(doc(&format!("a{k}"), 0)).unwrap();
    }
    w.commit().unwrap();
    for k in 0..5 {
        w.add_document(doc(&format!("b{k}"), 0)).unwrap();
    }
    w.delete_documents_by_term(&[Term::new("id", "c0")])
        .unwrap();
    assert!(w.pending_doc_count() > 0);
    let before = w.add_document(doc("gone", 0)).unwrap();
    let seq = w.delete_all().unwrap();
    assert!(seq > before);
    assert_eq!(w.pending_doc_count(), 0);
    assert_eq!(lock(&w.log).nodes.len(), 0);
    // Not durable yet.
    assert_eq!(live_documents(&dir).0.len(), 10);
    // "c0" was deleted before the delete-all: the new one survives.
    w.add_document(doc("c0", 0)).unwrap();
    w.add_document(doc("c1", 0)).unwrap();
    w.commit().unwrap();
    let (docs, _) = live_documents(&dir);
    assert_eq!(docs.keys().collect::<Vec<_>>(), ["c0", "c1"]);
    assert_clean(&dir);

    // A merge registered for a scheduler but not taken is dropped.
    lock(&w.core).merging.insert("_zz".into());
    w.pending_merges
        .push(Arc::new(ScheduledMerge::new(vec!["_zz".into()], 0)));
    let seq2 = w
        .delete_documents_by_query(&[DeleteQuery::MatchAll])
        .unwrap();
    assert!(seq2 > seq);
    assert!(w.pending_merges.is_empty());
    assert!(lock(&w.core).merging.is_empty());
    w.commit().unwrap();
    assert!(live_documents(&dir).0.is_empty());
    assert_clean(&dir);
}

/// Handing the writer back: pending work, a delete not yet applied, and
/// sequence numbers that keep climbing across both hand-overs.
#[test]
fn into_writer_hands_back_pending_work_and_the_numbering() {
    let tmp = TempDir::new("concurrent-into");
    let dir = FsDirectory::open(&tmp);
    let mut single = writer(&dir, 1000);
    let before = single.add_document(doc("q", 0)).unwrap();
    let w = ConcurrentIndexWriter::new(single, 2).unwrap();
    let during = w.add_document(doc("p", 0)).unwrap();
    assert!(during > before, "{during} after {before}");
    w.delete_documents_by_term(&[Term::new("id", b"q".to_vec())])
        .unwrap();
    assert_eq!(w.pending_doc_count(), 1);
    let mut single = w.into_writer().unwrap();
    let after = single.add_document(doc("r", 0)).unwrap();
    assert!(after > during + 1, "{after} after {during} and a delete");
    single.commit().unwrap();
    let (docs, _) = live_documents(&dir);
    assert_eq!(docs.keys().collect::<Vec<_>>(), ["p", "r"]);
    assert_clean(&dir);
}

/// **Merges and flushes between commits leave no per-segment state
/// behind.** The concurrent writer merges and flushes without stamping a
/// `segments_N`, so the writer's version cache had only a commit to prune
/// it: it held every segment each merge retired or each delete emptied
/// until then. Here one commit comes every 50 rounds, and every round must
/// hold the cache to the writer's view, and the deleter and the directory
/// to the view plus the last commit.
#[test]
fn merges_between_commits_keep_per_segment_state_to_the_view() {
    let tmp = TempDir::new("concurrent-resource-bounds");
    let dir = FsDirectory::open(&tmp);
    let w = ConcurrentIndexWriter::new(writer(&dir, 5), 1).unwrap();
    let mut merges = 0;
    for round in 0..300u32 {
        for k in 0..10u32 {
            let id = format!("{}", (round * 10 + k) % 400);
            w.update_document(Term::new("id", id.clone().into_bytes()), doc(&id, round))
                .unwrap();
        }
        merges += w.maybe_merge().unwrap();
        if round % 50 == 49 {
            w.commit().unwrap();
        }
        let (view, versions, tracked) = lock(&w.core).writer.per_segment_sizes();
        assert!(
            versions <= view,
            "round {round}: {versions} versions for {view} segments"
        );
        // A caller-driven merge keeps its sources' files while the last
        // commit names them; the next commit releases them. So the bound
        // is the view plus that commit -- not every segment ever written.
        let committed = segment_infos::read_latest(&dir).map_or(0, |i| i.segments.len());
        assert!(
            tracked <= view + committed,
            "round {round}: {tracked} file sets for {view} segments + {committed} committed"
        );
        let on_disk = std::fs::read_dir(tmp.path())
            .unwrap()
            .filter(|e| {
                e.as_ref()
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .ends_with(".si")
            })
            .count();
        assert!(
            on_disk <= view + committed,
            "round {round}: {on_disk} segments on disk for {view} + {committed} committed"
        );
    }
    assert!(merges > 50, "the policy must have merged: {merges}");
    w.commit().unwrap();
    assert_clean(&dir);
}

/// A directory that lives for the rest of the test process -- what a
/// merge scheduler's threads need to hold the writer (the directory
/// itself is removed by its `TempDir` guard).
fn static_dir(tmp: &TempDir) -> &'static FsDirectory {
    Box::leak(Box::new(FsDirectory::open(tmp)))
}

/// `ConcurrentMergeScheduler` behind the writer: merges are registered
/// after flushes and commits and run on merge threads while four threads
/// index; nothing here runs a merge itself. After `close_merges` the
/// writer can be taken back out of its `Arc`.
#[test]
fn a_concurrent_merge_scheduler_merges_while_threads_index() {
    let tmp = TempDir::new("concurrent-cms");
    let dir = static_dir(&tmp);
    let cms = crate::merge_scheduler::ConcurrentMergeScheduler::new();
    cms.set_max_merges_and_threads(3, 2).unwrap();
    let w = ConcurrentIndexWriter::with_merge_scheduler(writer(dir, 7), 4, Arc::new(cms.clone()))
        .unwrap();
    std::thread::scope(|scope| {
        for t in 0..4 {
            let w = &w;
            scope.spawn(move || {
                for i in 0..400 {
                    w.add_document(doc(&format!("t{t}x{i}"), 0)).unwrap();
                    if i % 61 == 0 {
                        w.commit().unwrap();
                    }
                }
            });
        }
    });
    w.commit().unwrap();
    assert_eq!(w.maybe_merge().unwrap(), 0, "merges are the scheduler's");
    w.close_merges().unwrap();
    assert_eq!(cms.merge_thread_count(), 0);
    w.commit().unwrap();
    let (docs, segments) = live_documents(dir);
    assert_eq!(docs.len(), 1600);
    assert!(
        segments < 1600 / 7 / 2,
        "{segments} segments: the scheduler merged too little"
    );
    assert_clean(dir);
    let w = Arc::try_unwrap(w)
        .ok()
        .expect("no merge thread still holds the writer");
    let mut single = w.into_writer().unwrap();
    single.commit().unwrap();
}

/// Refreshes without a commit: the merge scheduler merges the segments
/// they flush, as Java's `segmentInfos` holds them, and with a full-flush
/// wait a refresh's own merges run on the merge threads.
#[test]
fn refreshes_without_commits_are_merged_by_the_scheduler() {
    use crate::nrt::NrtSource;
    for wait in [0, 10_000] {
        let tmp = TempDir::new("concurrent-nrt-merges");
        let dir = static_dir(&tmp);
        let cms = crate::merge_scheduler::ConcurrentMergeScheduler::new();
        cms.set_max_merges_and_threads(3, 2).unwrap();
        let mut iw = writer(dir, 1000);
        iw.set_max_full_flush_merge_wait_millis(wait);
        let w = ConcurrentIndexWriter::with_merge_scheduler(iw, 2, Arc::new(cms.clone())).unwrap();
        for i in 0..300 {
            w.add_document(doc(&format!("d{i}"), 0)).unwrap();
            if i % 3 == 2 {
                w.nrt_snapshot(true, false).unwrap();
            }
        }
        w.close_merges().unwrap();
        let segments = lock(&w.core).writer.segment_count();
        assert!(
            segments < 30,
            "wait {wait}: {segments} segments after 100 refreshes"
        );
        w.commit().unwrap();
        assert_eq!(live_documents(dir).0.len(), 300);
        assert_clean(dir);
    }
}

/// `SerialMergeScheduler` behind the writer: a commit's merges run on the
/// committing thread before `commit` returns.
#[test]
fn a_serial_merge_scheduler_merges_on_the_committing_thread() {
    let tmp = TempDir::new("concurrent-serial");
    let dir = static_dir(&tmp);
    let w = ConcurrentIndexWriter::with_merge_scheduler(
        writer(dir, 5),
        1,
        Arc::new(crate::merge_scheduler::SerialMergeScheduler::new()),
    )
    .unwrap();
    for i in 0..60 {
        w.add_document(doc(&format!("d{i}"), 0)).unwrap();
        if i % 5 == 4 {
            w.commit().unwrap();
        }
    }
    // Twelve flushed segments, merged by the commits themselves.
    let merged = lock(&w.core).writer.segment_infos().segments.len();
    assert!(merged < 12, "{merged} segments: nothing was merged");
    w.close_merges().unwrap();
    w.commit().unwrap();
    assert_eq!(live_documents(dir).0.len(), 60);
    assert_clean(dir);

    // Without a policy, a scheduler has nothing to do.
    let tmp2 = TempDir::new("concurrent-serial-nopolicy");
    let dir2 = static_dir(&tmp2);
    let mut plain = writer(dir2, 5);
    plain.set_merge_policy(None);
    let w = ConcurrentIndexWriter::with_merge_scheduler(
        plain,
        1,
        Arc::new(crate::merge_scheduler::SerialMergeScheduler::new()),
    )
    .unwrap();
    for i in 0..20 {
        w.add_document(doc(&format!("d{i}"), 0)).unwrap();
    }
    w.commit().unwrap();
    w.flush().unwrap();
    assert_eq!(w.maybe_merge().unwrap(), 0);
    w.close_merges().unwrap();
    assert_eq!(lock(&w.core).writer.segment_infos().segments.len(), 4);
}

/// Records what a [`crate::index_writer::MergeListener`] hears.
#[derive(Default)]
struct Heard {
    events: Mutex<Vec<(bool, crate::index_writer::MergeEvent, bool)>>,
}

impl crate::index_writer::MergeListener for Heard {
    fn before_merge(&self, merge: &crate::index_writer::MergeEvent) {
        self.events
            .lock()
            .unwrap()
            .push((true, merge.clone(), false));
    }

    fn after_merge(
        &self,
        merge: &crate::index_writer::MergeEvent,
        _times: crate::index_writer::MergeTimes,
        failure: Option<&Error>,
    ) {
        self.events
            .lock()
            .unwrap()
            .push((false, merge.clone(), failure.is_some()));
    }
}

/// `OpenSearchConcurrentMergeScheduler.beforeMerge`/`afterMerge`: the
/// listener hears every scheduled merge and every forced one, each before
/// and after, with the sources' documents summed.
#[test]
fn a_merge_listener_hears_every_merge_before_and_after() {
    let tmp = TempDir::new("concurrent-listener");
    let dir = static_dir(&tmp);
    let heard = Arc::new(Heard::default());
    let mut iw = writer(dir, 5);
    iw.set_merge_listener(Some(heard.clone()));
    assert!(iw.merge_listener().is_some());
    let w = ConcurrentIndexWriter::with_merge_scheduler(
        iw,
        1,
        Arc::new(crate::merge_scheduler::SerialMergeScheduler::new()),
    )
    .unwrap();
    for i in 0..60 {
        w.add_document(doc(&format!("d{i}"), 0)).unwrap();
        if i % 5 == 4 {
            w.commit().unwrap();
        }
    }
    assert!(!w.has_pending_merges());
    let natural = heard.events.lock().unwrap().len();
    assert!(natural >= 2, "no merge was heard");
    lock(&w.core).writer.force_merge(1).unwrap();
    let events = heard.events.lock().unwrap().clone();
    assert!(events.len() > natural, "the forced merge was not heard");
    assert_eq!(events.len() % 2, 0);
    for pair in events.chunks(2) {
        let ((before, b, _), (after, a, failed)) = (&pair[0], &pair[1]);
        assert!(*before && !*after && !*failed);
        assert_eq!(a, b);
        assert!(b.total_docs > 0 && b.total_bytes > 0);
        assert_eq!(b.total_docs as usize % 5, 0, "{b:?}");
    }
    let (_, last, _) = events.last().unwrap();
    assert_eq!(last.max_num_segments, 1);
    assert_eq!(last.total_docs, 60);
    assert!(events[0].1.id < last.id);
}

/// A merge that unwinds before it reports its end is heard as failed.
#[test]
fn a_merge_that_unwinds_is_heard_as_failed() {
    let heard = Arc::new(Heard::default());
    let event = crate::index_writer::MergeEvent {
        id: 1,
        segments: vec!["_0".to_string()],
        total_docs: 1,
        total_bytes: 1,
        max_num_segments: -1,
    };
    let listener: Arc<dyn crate::index_writer::MergeListener> = heard.clone();
    drop(crate::index_writer::MergeReport::start(
        Some((listener, event)),
        None,
    ));
    let events = heard.events.lock().unwrap();
    assert_eq!(events.len(), 2);
    assert!(events[0].0 && !events[1].0 && events[1].2);
}

/// A pluggable policy picks the merges the scheduler runs, and its
/// forced merges interleave their sources when shuffling is on.
#[test]
fn a_pluggable_policy_picks_the_scheduled_and_forced_merges() {
    let tmp = TempDir::new("concurrent-pluggable");
    let dir = static_dir(&tmp);
    let mut iw = writer(dir, 5);
    let mut log = crate::merge_policy::log::LogByteSizeMergePolicy::new();
    log.set_merge_factor(2).unwrap();
    log.min_merge_size = 0;
    iw.set_pluggable_merge_policy(Some(Arc::new(log)));
    iw.set_shuffle_forced_merges(true);
    let w = ConcurrentIndexWriter::with_merge_scheduler(
        iw,
        1,
        Arc::new(crate::merge_scheduler::SerialMergeScheduler::new()),
    )
    .unwrap();
    for i in 0..40 {
        w.add_document(doc(&format!("d{i}"), 0)).unwrap();
        if i % 5 == 4 {
            w.commit().unwrap();
        }
    }
    let segments = || lock(&w.core).writer.segment_infos().segments.len();
    assert!(segments() < 8, "no scheduled merge ran: {}", segments());
    // Merges wait for the forced one from here on.
    lock(&w.core)
        .writer
        .set_pluggable_merge_policy(Some(Arc::new(
            crate::merge_policy::log::LogByteSizeMergePolicy::new(),
        )));
    for i in 40..50 {
        w.add_document(doc(&format!("d{i}"), 0)).unwrap();
        if i % 5 == 4 {
            w.commit().unwrap();
        }
    }
    assert!(segments() >= 2);
    lock(&w.core).writer.force_merge(1).unwrap();
    assert_eq!(segments(), 1);
    let core = lock(&w.core);
    let sci = &core.writer.segment_infos().segments[0];
    let si = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
    let si = crate::segment_info::parse(&si, &sci.segment_id).unwrap();
    assert!(si
        .diagnostics
        .iter()
        .any(|(k, _)| k == crate::merge::SHUFFLE_MERGE_KEY));
}

/// `onMergeFinished` releases a merge the scheduler took but will not
/// run, and a writer with no scheduler has nothing to close.
#[test]
fn a_merge_the_scheduler_gives_back_releases_its_segments() {
    let tmp = TempDir::new("concurrent-sched-release");
    let dir = static_dir(&tmp);
    let w = ConcurrentIndexWriter::with_merge_scheduler(
        writer(dir, 5),
        1,
        Arc::new(crate::merge_scheduler::NoMergeScheduler),
    )
    .unwrap();
    for i in 0..40 {
        w.add_document(doc(&format!("d{i}"), 0)).unwrap();
        if i % 5 == 4 {
            w.commit().unwrap();
        }
    }
    // `NoMergeScheduler` leaves what the commits registered pending.
    assert!(w.has_pending_merges());
    let merge = w.next_merge().expect("a registered merge");
    assert!(merge
        .segments
        .iter()
        .all(|n| lock(&w.core).merging.contains(n)));
    w.on_merge_finished(&merge);
    assert!(merge
        .segments
        .iter()
        .all(|n| !lock(&w.core).merging.contains(n)));

    let tmp2 = TempDir::new("concurrent-no-sched");
    let dir2 = FsDirectory::open(&tmp2);
    let plain = ConcurrentIndexWriter::new(writer(&dir2, 5), 1).unwrap();
    plain.close_merges().unwrap();
}

/// A force merge aborts a scheduled merge of the same segments -- here
/// one stopped by a zero rate, which would otherwise never finish -- and
/// merges them itself, leaving no file of the aborted merge behind.
#[test]
fn a_force_merge_aborts_a_running_scheduled_merge() {
    use lucene_store::RateLimiter as _;
    let tmp = TempDir::new("concurrent-force-aborts");
    let dir = static_dir(&tmp);
    let w = ConcurrentIndexWriter::with_merge_scheduler(
        writer(dir, 5),
        1,
        Arc::new(crate::merge_scheduler::NoMergeScheduler),
    )
    .unwrap();
    for i in 0..40 {
        w.add_document(doc(&format!("d{i}"), 0)).unwrap();
        if i % 5 == 4 {
            w.commit().unwrap();
        }
    }
    let merge = w.next_merge().expect("a registered merge");
    let limiter = Arc::new(MergeRateLimiter::new(Arc::clone(merge.progress())));
    limiter.set_mb_per_sec(0.0);
    std::thread::scope(|scope| {
        let stuck = scope.spawn(|| w.merge(&merge, Some(limiter)));
        // Once the merge holds the control lock's `running` entry, or
        // before it starts: either way it is aborted.
        while lock(&w.core).running.is_empty() && !stuck.is_finished() {
            std::thread::yield_now();
        }
        w.with_writer_merges_aborted(|iw| iw.force_merge(1))
            .unwrap();
        assert!(matches!(stuck.join().unwrap(), Err(Error::MergeAborted)));
    });
    assert!(merge.is_aborted());
    assert!(lock(&w.core).running.is_empty());
    assert!(lock(&w.core).merging.is_empty());
    assert!(!w.has_pending_merges());
    w.commit().unwrap();
    let (docs, segments) = live_documents(dir);
    assert_eq!(docs.len(), 40);
    assert_eq!(segments, 1);
    assert_clean(dir);
}

/// `delete_all` aborts a running scheduled merge instead of waiting for
/// it: one stopped by a zero rate would otherwise never let it return.
#[test]
fn delete_all_aborts_a_running_scheduled_merge() {
    use lucene_store::RateLimiter as _;
    let tmp = TempDir::new("concurrent-delete-all-aborts");
    let dir = static_dir(&tmp);
    let w = ConcurrentIndexWriter::with_merge_scheduler(
        writer(dir, 5),
        1,
        Arc::new(crate::merge_scheduler::NoMergeScheduler),
    )
    .unwrap();
    for i in 0..40 {
        w.add_document(doc(&format!("d{i}"), 0)).unwrap();
        if i % 5 == 4 {
            w.commit().unwrap();
        }
    }
    let merge = w.next_merge().expect("a registered merge");
    let limiter = Arc::new(MergeRateLimiter::new(Arc::clone(merge.progress())));
    limiter.set_mb_per_sec(0.0);
    std::thread::scope(|scope| {
        let stuck = scope.spawn(|| w.merge(&merge, Some(limiter)));
        while lock(&w.core).running.is_empty() && !stuck.is_finished() {
            std::thread::yield_now();
        }
        w.delete_all().unwrap();
        assert!(matches!(stuck.join().unwrap(), Err(Error::MergeAborted)));
    });
    w.commit().unwrap();
    assert_eq!(live_documents(dir).0.len(), 0);
    assert_clean(dir);
}

/// Merge-on-commit (`maxFullFlushMergeWaitMillis`) leaves the segments
/// a scheduled merge holds alone -- `registerMerge` rejects a merge over
/// them -- and merges the rest into the commit.
#[test]
fn merge_on_commit_skips_segments_a_scheduled_merge_holds() {
    let tmp = TempDir::new("concurrent-merge-on-commit");
    let dir = static_dir(&tmp);
    let w = ConcurrentIndexWriter::with_merge_scheduler(
        writer(dir, 5),
        1,
        Arc::new(crate::merge_scheduler::NoMergeScheduler),
    )
    .unwrap();
    for i in 0..20 {
        w.add_document(doc(&format!("d{i}"), 0)).unwrap();
        if i % 5 == 4 {
            w.commit().unwrap();
        }
    }
    let held: HashSet<String> = lock(&w.core).merging.clone();
    assert!(!held.is_empty(), "the commits registered a merge");
    lock(&w.core)
        .writer
        .set_max_full_flush_merge_wait_millis(DEFAULT_MAX_FULL_FLUSH_MERGE_WAIT_MILLIS);
    for i in 20..40 {
        w.add_document(doc(&format!("d{i}"), 0)).unwrap();
        if i % 5 == 4 {
            w.flush().unwrap();
        }
    }
    let before: Vec<String> = lock(&w.core)
        .writer
        .live_infos()
        .segments
        .iter()
        .map(|s| s.segment_name.clone())
        .collect();
    w.commit().unwrap();
    let committed: Vec<String> = crate::segment_infos::read_latest(dir)
        .unwrap()
        .segments
        .iter()
        .map(|s| s.segment_name.clone())
        .collect();
    assert!(
        held.iter().all(|n| committed.contains(n)),
        "{held:?} {committed:?}"
    );
    assert!(
        committed.len() < before.len(),
        "the free segments were merged into the commit: {before:?} -> {committed:?}"
    );
    assert!(committed.iter().any(|n| !before.contains(n)));
}

/// A full-flush merge policy (OpenSearch's `merge-on-flush`) merges the
/// segments several slots flushed into the one segment the commit names,
/// and leaves the writer's other merges alone.
#[test]
fn full_flush_merge_policy_merges_every_slot_into_the_commit() {
    let tmp = TempDir::new("concurrent-full-flush-policy");
    let dir = FsDirectory::open(&tmp);
    let mut iw = writer(&dir, 1000);
    iw.set_max_full_flush_merge_wait_millis(5000);
    iw.set_full_flush_merge_policy(Some(Arc::new(
        crate::merge_policy::MergeOnFlushMergePolicy::new(Box::new(
            crate::merge_policy::TieredMergePolicy::new(MergePolicyConfig::default()),
        )),
    )));
    let w = ConcurrentIndexWriter::new(iw, 4).unwrap();
    // Every add takes the next slot, so all four buffers fill however
    // the threads would have been scheduled.
    w.rotate_slots.store(true, Ordering::Relaxed);
    for i in 0..200 {
        w.add_document(doc(&format!("d{i}"), 0)).unwrap();
    }
    w.flush().unwrap();
    assert_eq!(lock(&w.core).writer.live_infos().segments.len(), 4);
    w.commit().unwrap();
    let committed = crate::segment_infos::read_latest(&dir).unwrap();
    assert_eq!(committed.segments.len(), 1);
    assert_eq!(lock(&w.core).writer.committed_doc_count().unwrap(), 200);
}

/// One thread indexing alone keeps filling one buffer, so a flush writes
/// one segment, as Java reuses a free `DocumentsWriterPerThread`.
#[test]
fn one_thread_fills_one_buffer() {
    let tmp = TempDir::new("concurrent-one-thread");
    let dir = FsDirectory::open(&tmp);
    let w = ConcurrentIndexWriter::new(writer(&dir, 1000), 4).unwrap();
    for i in 0..20 {
        w.add_document(doc(&format!("d{i}"), 0)).unwrap();
    }
    let counts: Vec<usize> = w.slots.iter().map(|s| read_slot(s).num_docs()).collect();
    assert_eq!(counts, [20, 0, 0, 0]);
    w.commit().unwrap();
    assert_eq!(
        crate::segment_infos::read_latest(&dir)
            .unwrap()
            .segments
            .len(),
        1
    );
}

/// `flushNextBuffer` takes the fullest slot; `tryDeleteDocument` and
/// `tryUpdateDocValue` act on a snapshot's doc ids and are numbered from
/// this writer's log.
#[test]
fn flush_next_buffer_and_try_delete_by_doc_id() {
    use crate::nrt::NrtSource;
    let tmp = TempDir::new("concurrent-try");
    let dir = FsDirectory::open(&tmp);
    let w = ConcurrentIndexWriter::new(writer(&dir, 1000), 2).unwrap();
    w.rotate_slots.store(true, Ordering::Relaxed);
    assert!(!w.flush_next_buffer().unwrap());
    for i in 0..3 {
        w.add_document(doc(&format!("a{i}"), 0)).unwrap();
    }
    // Round-robin over two slots: the fuller one (two documents) goes
    // first, then the other.
    assert!(w.flush_next_buffer().unwrap());
    assert_eq!(w.pending_doc_count(), 1);
    assert!(w.flush_next_buffer().unwrap());
    assert_eq!(w.pending_doc_count(), 0);
    let snap = w.nrt_snapshot(true, false).unwrap();
    let before = lock(&w.log).next_seq;
    let seq = w.try_delete_document(&snap.segment_infos, 1).unwrap();
    assert_eq!(seq, Some(before));
    let update = DocValuesUpdate::Numeric {
        term: Term::new("", ""),
        field: "id".into(),
        value: Some(1),
    };
    // `id` has no doc values: refused before anything is written.
    assert!(w
        .try_update_doc_value(&snap.segment_infos, 0, &[update])
        .is_err());
    w.commit().unwrap();
    let (docs, _) = live_documents(&dir);
    assert_eq!(docs.len(), 2);
    // Doc 1 of the snapshot is the second of the first segment flushed,
    // which (round-robin, fuller slot first) holds a0 and a2.
    assert!(!docs.contains_key("a2"), "{docs:?}");
    assert_clean(&dir);
}

/// `addDocument` with vectors or explicit term frequencies through the
/// concurrent writer answers what the single-threaded writer answers for
/// the same writer configuration: accepted where it is (no vectors, no
/// custom terms), refused with the same error where it is not (among a
/// vector field the writer was not given and custom frequencies without a
/// custom-frequency field, whichever of those it refuses) -- and both
/// commit the same documents.
#[test]
fn vectors_and_custom_frequencies_go_through_as_the_single_writer_takes_them() {
    let (a, b) = (TempDir::new("cw-vectors-a"), TempDir::new("cw-vectors-b"));
    let (da, db) = (FsDirectory::open(&a), FsDirectory::open(&b));
    let mut single = writer(&da, 100);
    let concurrent = ConcurrentIndexWriter::new(writer(&db, 100), 2).unwrap();
    let outcome = |r: Result<SeqNo>| r.map(|_| ()).map_err(|e| e.to_string());
    let vectors = || vec![DocumentVector::float32("nosuch", vec![1.0, 2.0])];
    let terms = || vec![("alpha".to_string(), 3)];
    type OnSingle = Box<dyn Fn(&mut IndexWriter<'_>) -> Result<SeqNo>>;
    type OnConcurrent = Box<dyn Fn(&ConcurrentIndexWriter<'_>) -> Result<SeqNo>>;
    let cases: [(&str, OnSingle, OnConcurrent); 4] = [
        (
            "no vectors",
            Box::new(|w| w.add_document_with_vectors(doc("a", 0), Vec::new())),
            Box::new(|w| w.add_document_with_vectors(doc("a", 0), Vec::new())),
        ),
        (
            "unknown vector field",
            Box::new(move |w| w.add_document_with_vectors(doc("b", 0), vectors())),
            Box::new(move |w| w.add_document_with_vectors(doc("b", 0), vectors())),
        ),
        (
            "no custom terms",
            Box::new(|w| w.add_document_with_custom_freq_terms(doc("c", 0), Vec::new())),
            Box::new(|w| w.add_document_with_custom_freq_terms(doc("c", 0), Vec::new())),
        ),
        (
            "custom terms without the field",
            Box::new(move |w| w.add_document_with_custom_freq_terms(doc("d", 0), terms())),
            Box::new(move |w| w.add_document_with_custom_freq_terms(doc("d", 0), terms())),
        ),
    ];
    let mut refused = 0;
    for (what, on_single, on_concurrent) in &cases {
        let want = outcome(on_single(&mut single));
        assert_eq!(outcome(on_concurrent(&concurrent)), want, "{what}");
        refused += usize::from(want.is_err());
    }
    assert!((1..4).contains(&refused), "{refused} refused");
    single.commit().unwrap();
    concurrent.commit().unwrap();
    let concurrent = concurrent.into_writer().unwrap();
    assert_eq!(
        concurrent.committed_doc_count().unwrap(),
        single.committed_doc_count().unwrap()
    );
    assert_eq!(single.committed_doc_count().unwrap(), 4 - refused);
}

/// `OneMerge.reorder` reversing every merge, and a diagnostic saying it ran.
#[derive(Debug)]
struct ReverseHooks;

impl crate::merge_policy::MergeHooks for ReverseHooks {
    fn prepare(
        &self,
        dir: &dyn lucene_store::Directory,
        sources: &crate::segment_infos::SegmentInfos,
        may_reorder: bool,
    ) -> std::result::Result<crate::merge_policy::PreparedMerge, String> {
        let mut total = 0i32;
        for sci in &sources.segments {
            let si = dir
                .open(&format!("{}.si", sci.segment_name))
                .map_err(|e| e.to_string())?;
            total += segment_info::parse(&si, &sci.segment_id)
                .map_err(|e| e.to_string())?
                .doc_count;
        }
        Ok(crate::merge_policy::PreparedMerge {
            live_docs: vec![None; sources.segments.len()],
            new_to_old: may_reorder.then(|| (0..total).rev().collect()),
            diagnostics: vec![("test.reversed".to_string(), "true".to_string())],
        })
    }
}

/// A tiered policy whose natural merges carry [`ReverseHooks`].
#[derive(Debug)]
struct Reversing(crate::merge_policy::TieredMergePolicy);

impl crate::merge_policy::MergePolicy for Reversing {
    fn find_merges(
        &self,
        trigger: MergeTrigger,
        infos: &[crate::merge_policy::MergeSegment],
        ctx: &dyn crate::merge_policy::MergeContext,
    ) -> crate::merge_policy::api::Result<Option<crate::merge_policy::MergeSpecification>> {
        let Some(spec) = self.0.find_merges(trigger, infos, ctx)? else {
            return Ok(None);
        };
        let hooks: Arc<dyn crate::merge_policy::MergeHooks> = Arc::new(ReverseHooks);
        let mut out = crate::merge_policy::MergeSpecification::new();
        for merge in spec.merges {
            out.add(merge.with_hooks(hooks.clone()));
        }
        Ok(Some(out))
    }

    fn find_forced_merges(
        &self,
        infos: &[crate::merge_policy::MergeSegment],
        max_segment_count: i32,
        segments_to_merge: &std::collections::HashMap<String, bool>,
        ctx: &dyn crate::merge_policy::MergeContext,
    ) -> crate::merge_policy::api::Result<Option<crate::merge_policy::MergeSpecification>> {
        self.0
            .find_forced_merges(infos, max_segment_count, segments_to_merge, ctx)
    }

    fn find_forced_deletes_merges(
        &self,
        infos: &[crate::merge_policy::MergeSegment],
        ctx: &dyn crate::merge_policy::MergeContext,
    ) -> crate::merge_policy::api::Result<Option<crate::merge_policy::MergeSpecification>> {
        self.0.find_forced_deletes_merges(infos, ctx)
    }

    fn compound_file_settings(&self) -> crate::merge_policy::CompoundFileSettings {
        self.0.compound_file_settings()
    }

    fn compound_file_settings_mut(&mut self) -> &mut crate::merge_policy::CompoundFileSettings {
        self.0.compound_file_settings_mut()
    }
}

fn reversing_writer(dir: &FsDirectory) -> IndexWriter<'_> {
    let mut iw = writer(dir, 7);
    let mut tiered =
        crate::merge_policy::TieredMergePolicy::new(iw.merge_policy().cloned().unwrap());
    crate::merge_policy::MergePolicy::compound_file_settings_mut(&mut tiered)
        .set_no_cfs_ratio(0.0)
        .unwrap();
    iw.set_pluggable_merge_policy(Some(Arc::new(Reversing(tiered))));
    iw
}

/// Each committed segment's stored ids in doc-id order, and whether its
/// `.si` records [`ReverseHooks`].
fn segments_and_hooks(dir: &FsDirectory) -> Vec<(Vec<String>, bool)> {
    let infos = segment_infos::read_latest(dir).unwrap();
    infos
        .segments
        .iter()
        .map(|sci| {
            let name = &sci.segment_name;
            let si =
                segment_info::parse(&dir.open(&format!("{name}.si")).unwrap(), &sci.segment_id)
                    .unwrap();
            let fdt = dir.open(&format!("{name}.fdt")).unwrap();
            let fdx = dir.open(&format!("{name}.fdx")).unwrap();
            let fdm = dir.open(&format!("{name}.fdm")).unwrap();
            let reader = stored_fields::open(&fdt, &fdx, &fdm, &sci.segment_id, "").unwrap();
            let ids = (0..reader.max_doc())
                .map(|d| match &reader.document(d).unwrap().fields[0].value {
                    FieldValue::String(s) => s.clone(),
                    other => panic!("{other:?}"),
                })
                .collect();
            let hooked = si
                .diagnostics
                .iter()
                .any(|(k, v)| k == "test.reversed" && v == "true");
            (ids, hooked)
        })
        .collect()
}

/// A merged segment of documents that were indexed in id order, holding
/// them in descending order: the hooks reordered it.
fn assert_reversed_merges(dir: &FsDirectory, docs: usize) {
    let segments = segments_and_hooks(dir);
    assert_eq!(
        segments.iter().map(|(ids, _)| ids.len()).sum::<usize>(),
        docs
    );
    let merged: Vec<_> = segments.iter().filter(|(_, hooked)| *hooked).collect();
    assert!(!merged.is_empty(), "no merge ran the hooks: {segments:?}");
    assert!(
        merged
            .iter()
            .any(|(ids, _)| ids.windows(2).any(|w| w[0] > w[1])),
        "no hooked merge reordered its documents: {merged:?}"
    );
    assert_clean(dir);
}

/// The merge scheduler's merges come from the pluggable policy and run
/// its hooks: reordered, with the hook's diagnostics in the merged `.si`.
#[test]
fn scheduled_merges_run_the_pluggable_policys_hooks() {
    let tmp = TempDir::new("concurrent-cms-hooks");
    let dir = static_dir(&tmp);
    let cms = crate::merge_scheduler::ConcurrentMergeScheduler::new();
    cms.set_max_merges_and_threads(3, 2).unwrap();
    let w = ConcurrentIndexWriter::with_merge_scheduler(
        reversing_writer(dir),
        1,
        Arc::new(cms.clone()),
    )
    .unwrap();
    for i in 0..120 {
        w.add_document(doc(&format!("d{i:04}"), 0)).unwrap();
        if i % 7 == 6 {
            w.commit().unwrap();
        }
    }
    w.commit().unwrap();
    w.close_merges().unwrap();
    w.commit().unwrap();
    assert_reversed_merges(dir, 120);
}

/// Without a scheduler, [`ConcurrentIndexWriter::maybe_merge`] runs the
/// pluggable policy's merges, hooks included.
#[test]
fn maybe_merge_runs_the_pluggable_policys_hooks() {
    let tmp = TempDir::new("concurrent-maybe-merge-hooks");
    let dir = FsDirectory::open(&tmp);
    let w = ConcurrentIndexWriter::new(reversing_writer(&dir), 1).unwrap();
    for i in 0..120 {
        w.add_document(doc(&format!("d{i:04}"), 0)).unwrap();
        if i % 7 == 6 {
            w.commit().unwrap();
        }
    }
    w.commit().unwrap();
    assert!(w.maybe_merge().unwrap() > 0);
    w.commit().unwrap();
    assert_reversed_merges(&dir, 120);
}
