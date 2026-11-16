#![cfg(test)]
use super::*;
use lucene_store::codec_util::ID_LENGTH;
use lucene_store::directory::FsDirectory;
use lucene_store::{ByteBuffersDirectory, DataOutput};

#[test]
fn failed_live_docs_callback_cleans_renamed_creations_and_preserves_renamed_preexisting_file() {
    use lucene_codecs::live_docs_writer::LiveDocsWriter;
    #[derive(Debug)]
    struct RenameThenFail;
    impl Lucene104Codec for RenameThenFail {
        fn live_docs_writer(&self) -> Option<&dyn LiveDocsWriter> {
            Some(self)
        }
    }
    impl LiveDocsWriter for RenameThenFail {
        fn write_live_docs(
            &self,
            _: &FixedBitSet,
            state: LiveDocsWriteState<'_>,
            _: usize,
        ) -> std::io::Result<()> {
            let mut output = state
                .directory
                .create_output("owned")
                .map_err(std::io::Error::other)?;
            output.write_bytes(&[1, 2]);
            output.close().map_err(std::io::Error::other)?;
            state
                .directory
                .rename("owned", "owned_renamed")
                .map_err(std::io::Error::other)?;
            state
                .directory
                .rename("preexisting", "renamed")
                .map_err(std::io::Error::other)?;
            Err(std::io::Error::other("after rename"))
        }
    }
    let dir = ByteBuffersDirectory::new();
    let mut output = dir.create_output("preexisting").unwrap();
    output.write_bytes(&[8, 5]);
    output.close().unwrap();
    let info = sci("_0", -1, 0);
    let error =
        apply_deletes_with_codec(&dir, &info, None, 1, [0], Some(&RenameThenFail)).unwrap_err();
    assert_eq!(error.to_string(), "after rename");
    assert_eq!(dir.list_all().unwrap(), ["renamed"]);
    assert_eq!(&*dir.open("renamed").unwrap(), &[8, 5]);
    assert_eq!(info.del_gen, -1);
    assert_eq!(info.del_count, 0);
}

#[test]
fn copy_failures_before_creation_preserve_unowned_destination() {
    let dir = ByteBuffersDirectory::new();
    for (name, bytes) in [("source", &[1, 2][..]), ("occupied", &[8, 5][..])] {
        let mut output = dir.create_output(name).unwrap();
        output.write_bytes(bytes);
        output.close().unwrap();
    }
    let tracked = CreatedOutputs::new(&dir);
    for source in ["source", "missing"] {
        assert!(tracked.copy_from(&dir, source, "occupied").is_err());
        assert_eq!(&*dir.open("occupied").unwrap(), &[8, 5]);
        assert_eq!(&*dir.open("source").unwrap(), &[1, 2]);
        assert!(tracked.files().is_empty());
    }
}

struct CopyFault {
    fail_write: bool,
}
impl lucene_store::index_output::OutputFaults for CopyFault {
    fn before_write(&mut self, _: &str, _: usize) -> lucene_store::index_output::WriteVerdict {
        use lucene_store::index_output::WriteVerdict;
        if self.fail_write {
            WriteVerdict::Fail {
                keep: 1,
                error: std::io::Error::other("copy write"),
            }
        } else {
            WriteVerdict::Write
        }
    }
    fn on_close(&mut self, _: &str) -> lucene_store::index_output::CloseVerdict {
        use lucene_store::index_output::CloseVerdict;
        if self.fail_write {
            CloseVerdict::Close
        } else {
            CloseVerdict::FailAfterFlush(std::io::Error::other("copy close"))
        }
    }
}
struct CopyFaultDirectory {
    inner: ByteBuffersDirectory,
    fail_write: bool,
    fail_delete: std::sync::atomic::AtomicBool,
}
impl Directory for CopyFaultDirectory {
    fn list_all(&self) -> lucene_store::Result<Vec<String>> {
        self.inner.list_all()
    }
    fn open(&self, name: &str) -> lucene_store::Result<lucene_store::Input> {
        self.inner.open(name)
    }
    fn file_length(&self, name: &str) -> lucene_store::Result<u64> {
        self.inner.file_length(name)
    }
    fn create_output(&self, name: &str) -> lucene_store::Result<lucene_store::FsIndexOutput> {
        let mut output = self.inner.create_output(name)?;
        output.set_faults(Box::new(CopyFault {
            fail_write: self.fail_write,
        }));
        Ok(output)
    }
    fn sync(&self, names: &[String]) -> lucene_store::Result<()> {
        self.inner.sync(names)
    }
    fn rename(&self, source: &str, dest: &str) -> lucene_store::Result<()> {
        self.inner.rename(source, dest)
    }
    fn delete_file(&self, name: &str) -> lucene_store::Result<()> {
        if self.fail_delete.load(std::sync::atomic::Ordering::SeqCst) {
            return Err(lucene_store::Error::Io(std::io::Error::other(
                "cleanup delete",
            )));
        }
        self.inner.delete_file(name)
    }
    fn sync_meta_data(&self) -> lucene_store::Result<()> {
        self.inner.sync_meta_data()
    }
    fn obtain_lock(&self, name: &str) -> lucene_store::Result<Box<dyn lucene_store::Lock>> {
        self.inner.obtain_lock(name)
    }
}
#[test]
fn failed_copy_cleans_its_partial_output_and_retains_failed_cleanup_for_retry() {
    let source = ByteBuffersDirectory::new();
    let mut output = source.create_output("source").unwrap();
    output.write_bytes(&[1, 2, 3]);
    output.close().unwrap();
    for fail_write in [false, true] {
        for fail_delete in [false, true] {
            let dir = CopyFaultDirectory {
                inner: ByteBuffersDirectory::new(),
                fail_write,
                fail_delete: std::sync::atomic::AtomicBool::new(fail_delete),
            };
            let tracked = CreatedOutputs::new(&dir);
            let error = tracked.copy_from(&source, "source", "dest").unwrap_err();
            assert_eq!(
                error.to_string(),
                if fail_write {
                    "copy write"
                } else {
                    "copy close"
                }
            );
            if fail_delete {
                assert_eq!(
                    *tracked.files(),
                    std::collections::BTreeSet::from(["dest".into()])
                );
                assert_eq!(
                    dir.file_length("dest").unwrap(),
                    if fail_write { 1 } else { 3 }
                );
                assert_eq!(
                    &*dir.open("dest").unwrap(),
                    if fail_write { &[1][..] } else { &[1, 2, 3][..] }
                );
                dir.fail_delete
                    .store(false, std::sync::atomic::Ordering::SeqCst);
                tracked.delete_file("dest").unwrap();
            }
            assert!(tracked.files().is_empty());
            assert!(dir.list_all().unwrap().is_empty());
            assert_eq!(&*source.open("source").unwrap(), &[1, 2, 3]);
        }
    }
}

#[test]
fn tracked_outputs_forward_directory_policy_and_track_successful_changes() {
    use lucene_store::{DataOutput, Directory, IndexOutput};
    let tmp = lucene_util::test_support::TempDir::new("live-docs-output-tracking");
    let dir = FsDirectory::open(&tmp);
    let tracked = CreatedOutputs::new(&dir);
    let mut output = tracked.create_output("original").unwrap();
    output.write_bytes(&[1, 2, 3]);
    output.close().unwrap();
    assert_eq!(tracked.file_length("original").unwrap(), 3);
    assert_eq!(&*tracked.open("original").unwrap(), &[1, 2, 3]);
    assert_eq!(tracked.list_all().unwrap(), ["original"]);
    tracked.sync(&["original".into()]).unwrap();
    tracked.sync_meta_data().unwrap();
    assert_eq!(tracked.fs_directory_path(), dir.fs_directory_path());
    assert!(tracked.pending_deletions().unwrap().is_empty());
    assert!(!tracked.disable_virus_checker());
    drop(tracked.obtain_lock("write.lock").unwrap());
    tracked.copy_from(&dir, "original", "copy").unwrap();
    tracked.rename("copy", "renamed").unwrap();
    tracked.delete_file("original").unwrap();
    assert_eq!(
        *tracked.files(),
        std::collections::BTreeSet::from(["renamed".into()])
    );
    assert!(tracked.rename("missing", "never").is_err());
    assert!(tracked.delete_file("missing").is_err());
    assert!(tracked.copy_from(&dir, "missing", "never").is_err());
    let estimated = tracked
        .create_output_with_estimate("estimated", Some(12))
        .unwrap();
    estimated.close().unwrap();
    let temp = tracked.create_temp_output("live", "docs").unwrap();
    let temp_name = temp.name().to_owned();
    temp.close().unwrap();
    let estimated_temp = tracked
        .create_temp_output_with_estimate("live", "docs", Some(10))
        .unwrap();
    let estimated_temp_name = estimated_temp.name().to_owned();
    estimated_temp.close().unwrap();
    assert_eq!(
        *tracked.files(),
        std::collections::BTreeSet::from([
            "renamed".into(),
            "estimated".into(),
            temp_name,
            estimated_temp_name,
        ])
    );

    let memory = lucene_store::ByteBuffersDirectory::new();
    let tracked = CreatedOutputs::new(&memory);
    memory.create_output("occupied").unwrap().close().unwrap();
    assert!(tracked.create_output("occupied").is_err());
    assert!(tracked.files().is_empty());
}

fn sci(segment_name: &str, del_gen: i64, del_count: i32) -> SegmentCommitInfo {
    SegmentCommitInfo {
        segment_name: segment_name.to_string(),
        segment_id: [7u8; ID_LENGTH],
        codec_name: "Lucene104".to_string(),
        del_gen,
        del_count,
        ..Default::default()
    }
}

// --- liv_file_name ---

#[test]
fn liv_file_name_matches_ifn_convention() {
    assert_eq!(liv_file_name("_0", 1), "_0_1.liv");
    // Generation 0 has no suffix at all -- `IndexFileNames
    // .fileNameFromGeneration`'s dedicated `gen == 0` branch, which
    // returns `segmentFileName(base, "", ext)`.
    assert_eq!(liv_file_name("_3", 0), "_3.liv");
    // Generation is base-36, not decimal -- this is the case decimal and
    // base36 diverge, so it's the one that actually proves the encoding.
    assert_eq!(liv_file_name("_0", 36), "_0_10.liv");
    assert_eq!(liv_file_name("_0", 100), "_0_2s.liv");
}

/// The all-live bitset is built by filling whole `u64` words and masking
/// the tail, not by setting `max_doc` bits one at a time. A `max_doc`
/// that is not a multiple of 64 is the case that catches a wrong mask --
/// `cardinality()` counts raw words, so a stray set bit past `max_doc`
/// would show up as a live doc that does not exist.
#[test]
fn all_live_bitset_masks_bits_past_max_doc() {
    for max_doc in [1usize, 63, 64, 65, 130, 256] {
        let (bits, newly) = mark_deleted(None, max_doc, []).unwrap();
        assert_eq!(newly, 0);
        assert_eq!(bits.cardinality(), max_doc, "max_doc={max_doc}");
        assert!((0..max_doc).all(|i| bits.get(i)), "max_doc={max_doc}");
    }
}

/// `ReadersAndUpdates.writeLiveDocs` ends in `advanceDelGen()`, whose
/// `generationAdvanced()` re-derives `SegmentCommitInfo.id`. Two rounds
/// of deletes against one segment therefore give three distinct ids.
/// Carrying `sci_id` straight across would tell a replication or NRT
/// consumer keying on the id that the segment had not changed.
#[test]
fn each_round_of_deletes_gives_the_segment_commit_a_new_id() {
    let tmp = lucene_util::test_support::TempDir::new("deletes-sci-id-advances");
    let dir = FsDirectory::open(&tmp);
    let mut info = sci("_0", -1, 0);
    info.sci_id = Some([1u8; ID_LENGTH]);

    let mut live = FixedBitSet::new(8);
    for i in 0..8 {
        live.set(i);
    }

    let after_first = apply_deletes(&dir, &info, Some(&live), 8, [0]).unwrap();
    assert_ne!(after_first.sci_id, info.sci_id);
    assert_eq!(after_first.del_gen, 1);
    assert_eq!(after_first.del_count, 1);
    assert_eq!(after_first.next_write_del_gen(), 2);

    let mut live = FixedBitSet::new(8);
    for i in 1..8 {
        live.set(i);
    }
    let after_second = apply_deletes(&dir, &after_first, Some(&live), 8, [1]).unwrap();
    assert_ne!(after_second.sci_id, after_first.sci_id);
    assert_ne!(after_second.sci_id, info.sci_id);
    assert_eq!(after_second.del_gen, 2);
    assert_eq!(after_second.del_count, 2);
}

/// Fails the close of every output created while `fail` is set (after
/// its bytes reached the file); everything else passes through.
struct FailingClose<'a> {
    inner: &'a FsDirectory,
    fail: std::sync::atomic::AtomicBool,
}

struct CloseFault;
impl lucene_store::index_output::OutputFaults for CloseFault {
    fn before_write(&mut self, _: &str, _: usize) -> lucene_store::index_output::WriteVerdict {
        lucene_store::index_output::WriteVerdict::Write
    }
    fn on_close(&mut self, _: &str) -> lucene_store::index_output::CloseVerdict {
        lucene_store::index_output::CloseVerdict::FailAfterFlush(std::io::Error::other("close"))
    }
}

impl Directory for FailingClose<'_> {
    fn list_all(&self) -> lucene_store::Result<Vec<String>> {
        self.inner.list_all()
    }
    fn open(&self, name: &str) -> lucene_store::Result<lucene_store::directory::Input> {
        self.inner.open(name)
    }
    fn create_output(&self, name: &str) -> lucene_store::Result<lucene_store::FsIndexOutput> {
        let mut output = self.inner.create_output(name)?;
        if self.fail.load(std::sync::atomic::Ordering::SeqCst) {
            output.set_faults(Box::new(CloseFault));
        }
        Ok(output)
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

/// `PendingDeletes.writeLiveDocs` deletes the files a failed attempt
/// created. Leaving the partial `.liv` behind made the retry at the same
/// generation fail, since `createOutput` refuses an existing name.
#[test]
fn a_failed_live_docs_write_leaves_no_partial_file_behind() {
    let tmp = lucene_util::test_support::TempDir::new("deletes-failed-liv-cleaned");
    let fs = FsDirectory::open(&tmp);
    let dir = FailingClose {
        inner: &fs,
        fail: std::sync::atomic::AtomicBool::new(true),
    };
    let info = sci("_0", -1, 0);
    assert!(apply_deletes(&dir, &info, None, 4, [0]).is_err());
    assert!(!fs.list_all().unwrap().contains(&liv_file_name("_0", 1)));

    dir.fail.store(false, std::sync::atomic::Ordering::SeqCst);
    let updated = apply_deletes(&dir, &info, None, 4, [0]).unwrap();
    assert_eq!(updated.del_gen, 1);
    assert_eq!(updated.del_count, 1);
}

/// A commit can never record more deletions than the segment has docs --
/// `SegmentInfos.write` refuses to serialize one, so building it is a
/// bug worth surfacing where it happens.
#[test]
fn del_count_past_max_doc_is_rejected() {
    let tmp = lucene_util::test_support::TempDir::new("deletes-del-count-past-max-doc");
    let dir = FsDirectory::open(&tmp);
    // A segment of 3 docs that already claims 3 deletions; deleting one
    // more would make 4.
    let info = sci("_0", 1, 3);
    let mut live = FixedBitSet::new(3);
    live.set(0);
    let err = apply_deletes(&dir, &info, Some(&live), 3, [0]).unwrap_err();
    assert!(
        matches!(
            &err,
            Error::DelCountExceedsMaxDoc {
                del_count: 4,
                max_doc: 3,
                ..
            }
        ),
        "unexpected error: {err}"
    );
    std::fs::remove_dir_all(&tmp).ok();
}

/// `del_count` comes off `segments_N`, where the only bound this port can
/// apply is `>= 0` — the `maxDoc` half of Java's check lives in the `.si`,
/// which `segment_infos` deliberately does not read. So `del_count +
/// newly_deleted` can overflow `i32`: a **panic** in a debug build, and in
/// a release build a *negative* count that sails straight past the
/// `> max_doc` test and into the commit.
#[test]
fn del_count_overflow_is_an_error_not_a_wrap() {
    let tmp = lucene_util::test_support::TempDir::new("deletes-del-count-overflow");
    let dir = FsDirectory::open(&tmp);
    let info = sci("_0", 1, i32::MAX);
    let mut live = FixedBitSet::new(3);
    live.set(0);
    let err = apply_deletes(&dir, &info, Some(&live), 3, [0]).unwrap_err();
    assert!(
        matches!(
            &err,
            Error::DelCountExceedsMaxDoc {
                del_count: i32::MAX,
                max_doc: 3,
                ..
            }
        ),
        "unexpected error: {err}"
    );
    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn negative_previous_delete_count_is_rejected_before_output() {
    let directory = lucene_store::ByteBuffersDirectory::new();
    let info = sci("_0", 1, -1);
    assert!(matches!(
        apply_deletes(&directory, &info, None, 3, [0]),
        Err(Error::DelCountExceedsMaxDoc { .. })
    ));
    assert!(directory.list_all().unwrap().is_empty());
}

// --- mark_deleted ---

#[test]
fn deleting_from_all_live_none_state() {
    let (bits, newly) = mark_deleted(None, 5, [1, 3]).unwrap();
    assert_eq!(newly, 2);
    assert!(bits.get(0));
    assert!(!bits.get(1));
    assert!(bits.get(2));
    assert!(!bits.get(3));
    assert!(bits.get(4));
    assert_eq!(bits.cardinality(), 3);
}

#[test]
fn deleting_from_existing_partially_deleted_bitset() {
    let mut existing = FixedBitSet::new(4);
    existing.set(0);
    existing.set(1);
    existing.set(2);
    existing.set(3);
    existing.clear(2); // doc 2 already deleted from a prior generation

    let (bits, newly) = mark_deleted(Some(&existing), 4, [0]).unwrap();
    assert_eq!(newly, 1);
    assert!(!bits.get(0));
    assert!(bits.get(1));
    assert!(!bits.get(2)); // stays deleted
    assert!(bits.get(3));
}

#[test]
fn deleting_an_already_deleted_doc_is_idempotent() {
    let mut existing = FixedBitSet::new(3);
    existing.set(0);
    existing.set(1);
    existing.set(2);
    existing.clear(1);

    let (bits, newly) = mark_deleted(Some(&existing), 3, [1, 1]).unwrap();
    assert_eq!(newly, 0); // doc 1 was already deleted, both calls no-op
    assert!(!bits.get(1));
}

#[test]
fn boundary_doc_ids_zero_and_max_doc_minus_one() {
    let (bits, newly) = mark_deleted(None, 5, [0, 4]).unwrap();
    assert_eq!(newly, 2);
    assert!(!bits.get(0));
    assert!(bits.get(1));
    assert!(!bits.get(4));
}

#[test]
fn out_of_range_doc_id_is_an_error_not_silent_or_panic() {
    let result = mark_deleted(None, 5, [5]);
    assert!(matches!(
        result,
        Err(Error::DocOutOfRange {
            doc_id: 5,
            max_doc: 5
        })
    ));
}

#[test]
fn negative_doc_id_is_an_error() {
    let result = mark_deleted(None, 5, [-1]);
    assert!(matches!(
        result,
        Err(Error::DocOutOfRange {
            doc_id: -1,
            max_doc: 5
        })
    ));
}

#[test]
fn empty_doc_ids_from_none_state_is_a_no_op() {
    let (bits, newly) = mark_deleted(None, 3, []).unwrap();
    assert_eq!(newly, 0);
    assert_eq!(bits.cardinality(), 3);
}

// --- apply_deletes: full round-trip via real Directory I/O ---

use lucene_util::test_support::TempDir;

/// A scratch directory that removes itself when the test ends -- unless
/// the test is panicking, in which case its bytes stay for inspection.
fn tempdir() -> TempDir {
    TempDir::new("deletes")
}

#[test]
fn first_delete_round_writes_gen_one_and_bumps_del_count() {
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let info = sci("_0", -1, 0);

    let updated = apply_deletes(&dir, &info, None, 4, [1, 3]).unwrap();
    assert_eq!(updated.del_gen, 1);
    assert_eq!(updated.del_count, 2);

    let bytes = std::fs::read(tmp.join("_0_1.liv")).unwrap();
    let parsed = live_docs::parse(&bytes, &info.segment_id, 1, 4, 2).unwrap();
    assert!(parsed.get(0));
    assert!(!parsed.get(1));
    assert!(parsed.get(2));
    assert!(!parsed.get(3));
}

#[test]
fn second_delete_round_increments_gen_and_unions_with_first() {
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let info = sci("_0", -1, 0);

    let after_first = apply_deletes(&dir, &info, None, 5, [1]).unwrap();
    assert_eq!(after_first.del_gen, 1);
    assert_eq!(after_first.del_count, 1);

    let first_liv = std::fs::read(tmp.join("_0_1.liv")).unwrap();
    let first_bits = live_docs::parse(&first_liv, &info.segment_id, 1, 5, 1).unwrap();

    let after_second = apply_deletes(&dir, &after_first, Some(&first_bits), 5, [3]).unwrap();
    assert_eq!(after_second.del_gen, 2);
    assert_eq!(after_second.del_count, 2);

    let second_liv = std::fs::read(tmp.join("_0_2.liv")).unwrap();
    let second_bits = live_docs::parse(&second_liv, &info.segment_id, 2, 5, 2).unwrap();
    // Union of both rounds: doc 1 (first round) and doc 3 (second round)
    // both stay deleted; everything else stays live.
    assert!(second_bits.get(0));
    assert!(!second_bits.get(1));
    assert!(second_bits.get(2));
    assert!(!second_bits.get(3));
    assert!(second_bits.get(4));
}

#[test]
fn redeleting_same_doc_in_a_later_generation_does_not_double_count() {
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let info = sci("_0", -1, 0);

    let after_first = apply_deletes(&dir, &info, None, 3, [0]).unwrap();
    let first_liv = std::fs::read(tmp.join("_0_1.liv")).unwrap();
    let first_bits = live_docs::parse(&first_liv, &info.segment_id, 1, 3, 1).unwrap();

    let after_second = apply_deletes(&dir, &after_first, Some(&first_bits), 3, [0]).unwrap();
    assert_eq!(after_second.del_gen, 2);
    assert_eq!(after_second.del_count, 1); // not double-counted
}

#[test]
fn out_of_range_doc_id_propagates_as_error_without_writing_a_file() {
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let info = sci("_0", -1, 0);

    let result = apply_deletes(&dir, &info, None, 3, [9]);
    assert!(result.is_err());
    assert!(!tmp.join("_0_1.liv").exists());
}
