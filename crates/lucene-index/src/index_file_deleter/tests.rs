#![cfg(test)]
use super::*;
use crate::segment_info::{self, LuceneVersion, SegmentInfo};
use lucene_store::data_output::DataOutput;
use lucene_store::directory::FsDirectory;

use lucene_util::test_support::TempDir;

/// A scratch directory that removes itself when the test ends -- unless
/// the test is panicking, in which case its bytes stay for inspection.
fn tempdir(tag: &str) -> TempDir {
    TempDir::new(&format!("file-deleter-{tag}"))
}

fn version() -> LuceneVersion {
    LuceneVersion {
        major: 10,
        minor: 0,
        bugfix: 0,
    }
}

fn id(seed: u8) -> [u8; ID_LENGTH] {
    [seed; ID_LENGTH]
}

fn write(dir: &FsDirectory, name: &str, bytes: &[u8]) {
    let mut out = dir.create_output(name).unwrap();
    out.write_bytes(bytes);
    out.close().unwrap();
}

/// Writes a minimal but real `.si` for `name` listing itself plus
/// `extra_files`, and returns the matching [`SegmentCommitInfo`]. Those
/// extra files are created empty, so every name the deleter refcounts is a
/// name that actually exists.
fn seed_segment(
    dir: &FsDirectory,
    name: &str,
    seed: u8,
    extra_files: &[&str],
) -> SegmentCommitInfo {
    let mut files: Vec<String> = vec![format!("{name}.si")];
    for f in extra_files {
        files.push((*f).to_string());
        write(dir, f, b"x");
    }
    let si = SegmentInfo {
        id: id(seed),
        version: version(),
        min_version: None,
        doc_count: 1,
        is_compound_file: false,
        has_blocks: false,
        diagnostics: vec![],
        files,
        attributes: vec![],
        index_sort: None,
    };
    write(dir, &format!("{name}.si"), &segment_info::write(&si, ""));
    SegmentCommitInfo {
        segment_name: name.to_string(),
        segment_id: id(seed),
        codec_name: "Lucene104".to_string(),
        del_gen: -1,
        del_count: 0,
        field_infos_gen: -1,
        doc_values_gen: -1,
        soft_del_count: 0,
        sci_id: None,
        field_infos_files: vec![],
        dv_update_files: vec![],
        ..Default::default()
    }
}

fn infos(generation: i64, segments: Vec<SegmentCommitInfo>) -> SegmentInfos {
    SegmentInfos {
        id: id(0xAA),
        generation,
        format_version: segment_infos::VERSION_86,
        lucene_version: segment_infos::LuceneVersion {
            major: 10,
            minor: 0,
            bugfix: 0,
        },
        index_created_version_major: 10,
        version: generation,
        counter: 10,
        min_segment_lucene_version: None,
        segments,
        user_data: vec![],
    }
}

fn listing(dir: &FsDirectory) -> Vec<String> {
    let mut v = dir.list_all().unwrap();
    v.sort();
    v
}

#[test]
fn codec_file_pattern_accepts_lucene_names_and_rejects_everything_else() {
    for name in [
        "_0.si",
        "_0.fdt",
        "_1a.cfs",
        "_0_1.liv",
        "_0_Lucene104_0.tim",
        "_ff_Lucene90_0.dvd",
        "segments",
        "segments_2",
        "segments.gen",
        "pending_segments_3",
    ] {
        assert!(is_index_file_name(name), "{name} should be an index file");
    }
    for name in [
        "write.lock",
        "some.write.lock",
        "README.txt",
        "_0",
        "_.si",
        "_A.si",
        "_0_nodot",
        "notours",
        "",
    ] {
        assert!(
            !is_index_file_name(name),
            "{name} should not be an index file"
        );
    }
}

#[test]
fn codec_pattern_uses_java_line_terminators_only_for_regex_branch() {
    for terminator in ['\n', '\r', '\u{0085}', '\u{2028}', '\u{2029}'] {
        assert!(!is_index_file_name(&format!("_0.{terminator}")));
        assert!(!is_index_file_name(&format!("_0_{terminator}.si")));
        assert!(is_index_file_name(&format!("segments_{terminator}")));
        assert!(is_index_file_name(&format!(
            "pending_segments_{terminator}"
        )));
    }
    for name in ["_0.", "_0_\u{000b}.si", "_0.\u{000c}"] {
        assert!(is_index_file_name(name));
    }
}

#[test]
fn simpletext_file_sets_reach_all_deleter_and_commit_listing_paths() {
    use lucene_store::ByteBuffersDirectory;
    use std::collections::BTreeSet;
    for stage in ["flush", "merged"] {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/data/codecs_simpletext/codec")
            .join(stage)
            .join("index");
        let dir = ByteBuffersDirectory::new();
        for entry in std::fs::read_dir(root).unwrap() {
            let entry = entry.unwrap();
            let mut out = dir
                .create_output(entry.file_name().to_str().unwrap())
                .unwrap();
            out.write_bytes(&std::fs::read(entry.path()).unwrap());
            out.close().unwrap();
        }
        let infos = segment_infos::read_latest(&dir).unwrap();
        let expected: BTreeSet<_> = dir.list_all().unwrap().into_iter().collect();
        assert_eq!(
            files_of_infos(&dir, &infos, true)
                .unwrap()
                .into_iter()
                .collect::<BTreeSet<_>>(),
            expected
        );
        let commits = list_commits(&dir).unwrap();
        assert_eq!(commits.len(), 1);
        assert_eq!(
            commits[0]
                .file_names()
                .iter()
                .cloned()
                .collect::<BTreeSet<_>>(),
            expected
        );
        let mut deleter = IndexFileDeleter::open(&dir, &infos, DeletionPolicy::KeepAll).unwrap();
        deleter.forget_segment_files();
        deleter.checkpoint(&infos, false).unwrap();
        assert_eq!(
            dir.list_all().unwrap().into_iter().collect::<BTreeSet<_>>(),
            expected
        );
        let mut wrong_id = infos.segments[0].clone();
        wrong_id.segment_id = [0; ID_LENGTH];
        assert!(segment_files(&dir, &wrong_id).is_err());
        let name = format!("{}.si", wrong_id.segment_name);
        dir.delete_file(&name).unwrap();
        let mut output = dir.create_output(&name).unwrap();
        output.write_bytes(b"truncated");
        output.close().unwrap();
        assert!(segment_files(&dir, &infos.segments[0]).is_err());
    }
}

#[test]
fn segments_file_name_test_excludes_the_legacy_pointer_file() {
    assert!(is_segments_file_name("segments"));
    assert!(is_segments_file_name("segments_9"));
    // Lucene 4.0 dropped `segments.gen`; it is not a commit, so the deleter
    // treats it as an ordinary unreferenced index file (and reclaims it)
    // rather than trying to read a commit out of it.
    assert!(!is_segments_file_name("segments.gen"));
    assert!(!is_segments_file_name("pending_segments_9"));
}

#[test]
fn parse_segment_name_stops_at_the_generation_or_the_extension() {
    assert_eq!(parse_segment_name("_0.si"), "_0");
    assert_eq!(parse_segment_name("_1a_1.liv"), "_1a");
    assert_eq!(parse_segment_name("_0_Lucene104_0.tim"), "_0");
    assert_eq!(parse_segment_name("_0"), "_0");
}

fn named_sci(segment_name: &str, del_gen: i64) -> SegmentCommitInfo {
    SegmentCommitInfo {
        segment_name: segment_name.to_string(),
        segment_id: id(0x11),
        codec_name: "Lucene104".to_string(),
        del_gen,
        ..Default::default()
    }
}

#[test]
fn inflate_gens_pushes_both_counters_past_everything_on_disk() {
    let mut current = infos(2, vec![]);
    let files: Vec<String> = [
        "segments_2",
        "pending_segments_7",
        "_0.si",
        "_a.fdt",
        "_zz.tmp",
        "trash",
        "write.lock",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();

    IndexFileDeleter::inflate_gens(&files, &mut current);
    assert_eq!(
        current.generation, 7,
        "the leftover pending commit must win"
    );
    // `_a` is 10 in base 36, so the next name has to be at least 11. `_zz`
    // is skipped: Java refuses to read a generation out of a `.tmp` name.
    assert_eq!(current.counter, 11);
}

#[test]
fn inflate_gens_pushes_a_segments_next_write_gens_past_a_crashed_sessions_liv() {
    // The exact crash this guards: the commit records `_0` at delGen 1, but
    // a `_0_3.liv` from a session that died before committing is still on
    // disk. Without the per-segment half the next delete would derive
    // `del_gen + 1 == 2`, then `3` -- landing straight on the orphan.
    let mut current = infos(1, vec![named_sci("_0", 1)]);
    assert_eq!(current.segments[0].next_write_del_gen(), 2);

    let files: Vec<String> = ["segments_1", "_0.si", "_0_1.liv", "_0_3.liv"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    IndexFileDeleter::inflate_gens(&files, &mut current);

    assert_eq!(current.segments[0].next_write_del_gen(), 4);
    // Java's own comment: the maximum is the *union* of the live-docs,
    // field-infos and doc-values generations, so all three move together.
    assert_eq!(current.segments[0].next_write_field_infos_gen(), 4);
    assert_eq!(current.segments[0].next_write_doc_values_gen(), 4);
    // `del_gen` itself is untouched -- only the *next write* moves.
    assert_eq!(current.segments[0].del_gen, 1);
}

#[test]
fn inflate_gens_never_lowers_a_segments_next_write_gen() {
    // A segment already at delGen 5 with nothing above `_0_5.liv` on disk
    // keeps its derived next-write generation of 6.
    let mut current = infos(1, vec![named_sci("_0", 5)]);
    let files: Vec<String> = ["segments_1", "_0.si", "_0_5.liv"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    IndexFileDeleter::inflate_gens(&files, &mut current);
    assert_eq!(current.segments[0].next_write_del_gen(), 6);
}

#[test]
fn inflate_gens_leaves_a_segment_with_no_files_on_disk_at_its_derived_gens() {
    let mut current = infos(1, vec![named_sci("_9", -1)]);
    IndexFileDeleter::inflate_gens(&["segments_1".to_string()], &mut current);
    assert_eq!(current.segments[0].next_write_del_gen(), 1);
}

/// A generation is parsed out of a **file name**, in base 36, so any
/// process that can create a file in the index directory can hand
/// `inflate_gens` an `i64::MAX` — `1y2p0ij32e8e7` is `Long.MAX_VALUE` in
/// base 36, and it fits Lucene's own `_<seg>_<gen>.<ext>` shape exactly.
/// Java's `genLong + 1` merely wraps; the bare `+ 1` here **panicked** in a
/// debug build, taking the JVM down at index-open time. Such a name is
/// trash, in the same sense Java's own comment uses for an unparsable one,
/// so it contributes nothing rather than being followed.
#[test]
fn a_trash_file_name_claiming_an_absurd_generation_is_ignored_not_followed() {
    let cap = lucene_util::base36::to_base36(segment_infos::MAX_GENERATION);
    let mut current = infos(1, vec![named_sci("_0", 1)]);
    let files: Vec<String> = [
        "segments_1".to_string(),
        "_0.si".to_string(),
        "_0_2.liv".to_string(),
        // `i64::MAX` and `i64::MIN` in base 36.
        "_0_1y2p0ij32e8e7.liv".to_string(),
        "_0_-1y2p0ij32e8e8.liv".to_string(),
        // ...and the boundary, which is the one that actually mattered:
        // `MAX_GENERATION` is small enough to survive the `+ 1` in
        // `inflate_gens` but produces a `nextWriteDelGen` of
        // `MAX_GENERATION + 1`, which `advance_del_gen` would then write
        // into `segments_N` for `segment_infos::parse` to refuse. An
        // index this port wrote and could no longer open.
        format!("_0_{cap}.liv"),
    ]
    .to_vec();
    IndexFileDeleter::inflate_gens(&files, &mut current);

    // The one *usable* generation on disk still wins: 2 -> next write 3.
    assert_eq!(current.segments[0].next_write_del_gen(), 3);
    assert_eq!(current.segments[0].next_write_field_infos_gen(), 3);
    assert_eq!(current.segments[0].next_write_doc_values_gen(), 3);
}

/// The same shape for the commit-wide counters: a `segments_<absurd>` and
/// a `_<absurd>.si` in the listing must not drive `generation`/`counter`
/// to a value the next commit's `+ 1` cannot represent.
#[test]
fn trash_segment_and_commit_names_do_not_inflate_the_commit_counters() {
    let cap = lucene_util::base36::to_base36(segment_infos::MAX_GENERATION);
    let mut current = infos(1, vec![named_sci("_0", -1)]);
    let files: Vec<String> = [
        "segments_1".to_string(),
        "segments_1y2p0ij32e8e7".to_string(),
        "_0.si".to_string(),
        "_1y2p0ij32e8e7.si".to_string(),
        // The boundary: `_<base36(MAX_GENERATION)>.si` drove
        // `infos.counter` to `MAX_GENERATION + 1`, which
        // `segment_infos::write` serializes and `segment_infos::parse`
        // then refuses.
        format!("_{cap}.si"),
        format!("segments_{cap}"),
    ]
    .to_vec();
    IndexFileDeleter::inflate_gens(&files, &mut current);

    assert_eq!(current.generation, 1);
    assert!(
        current.counter <= segment_infos::MAX_GENERATION,
        "counter {} escaped the cap",
        current.counter
    );
}

#[test]
fn parse_generation_reads_javas_four_file_name_shapes() {
    // `segment.ext` and `segment_codec_suffix.ext` carry no generation.
    assert_eq!(parse_generation("_0.si"), 0);
    assert_eq!(parse_generation("_0_Lucene104_0.tim"), 0);
    // `segment_gen.ext` and `segment_gen_codec_suffix.ext` do, in base 36.
    assert_eq!(parse_generation("_0_3.liv"), 3);
    assert_eq!(parse_generation("_0_a.liv"), 10);
    assert_eq!(parse_generation("_0_2_Lucene90_0.dvd"), 2);
    // Trash tails read as no generation rather than erroring, matching
    // Java's `catch (NumberFormatException)`.
    assert_eq!(parse_generation("_0_!!.liv"), 0);
    assert_eq!(parse_generation("write.lock"), 0);
    // The generational `.fnm` and `.dvm`/`.dvd`/`.dvs` a doc-values
    // update writes (`crate::field_updates`), which is where the
    // four-part `segment_gen_codec_suffix.ext` shape actually comes from.
    assert_eq!(parse_generation("_0_1.fnm"), 1);
    assert_eq!(parse_generation("_1a_z_Lucene90_0.dvm"), 35);
}

#[test]
fn inflate_gens_never_lowers_a_counter_and_tolerates_an_empty_directory() {
    let mut current = infos(5, vec![]);
    let before_counter = current.counter;
    IndexFileDeleter::inflate_gens(&[], &mut current);
    assert_eq!(current.generation, 5);
    assert_eq!(current.counter, before_counter);

    // A file whose base-36 tail is unparsable is trash, not a counter.
    let files = vec!["segments_zzzzzzzzzzzzzzzzzzzz".to_string()];
    IndexFileDeleter::inflate_gens(&files, &mut current);
    assert_eq!(current.generation, 5);
    assert_eq!(current.counter, before_counter);
}

#[test]
fn opening_on_an_empty_directory_holds_nothing_and_deletes_nothing() {
    let tmp = tempdir("empty");
    let dir = FsDirectory::open(&tmp);
    let deleter =
        IndexFileDeleter::open(&dir, &infos(0, vec![]), DeletionPolicy::KeepOnlyLastCommit)
            .unwrap();
    assert_eq!(deleter.commit_count(), 0);
    assert!(deleter.commit_file_names().is_empty());
    assert_eq!(deleter.ref_count("_0.si"), 0);
    assert!(!deleter.exists("_0.si"));
    assert!(listing(&dir).is_empty());
}

#[test]
fn a_non_commit_checkpoint_holds_the_new_view_and_releases_the_previous_one() {
    let tmp = tempdir("noncommit-checkpoint");
    let dir = FsDirectory::open(&tmp);
    let a = seed_segment(&dir, "_0", 1, &["_0.fdt"]);
    let b = seed_segment(&dir, "_1", 2, &["_1.fdt"]);

    let mut deleter =
        IndexFileDeleter::open(&dir, &infos(0, vec![]), DeletionPolicy::KeepOnlyLastCommit)
            .unwrap();
    // The init sweep already reclaimed both, since nothing referenced them.
    assert!(listing(&dir).is_empty());

    let a = {
        let _ = a;
        seed_segment(&dir, "_0", 1, &["_0.fdt"])
    };
    deleter
        .checkpoint(&infos(0, vec![a.clone()]), false)
        .unwrap();
    assert_eq!(deleter.ref_count("_0.si"), 1);
    assert_eq!(deleter.ref_count("_0.fdt"), 1);
    assert!(deleter.exists("_0.si"));

    let b = {
        let _ = b;
        seed_segment(&dir, "_1", 2, &["_1.fdt"])
    };
    // Replacing the view with a different segment releases the first one.
    deleter
        .checkpoint(&infos(0, vec![b.clone()]), false)
        .unwrap();
    assert_eq!(deleter.ref_count("_0.si"), 0);
    assert_eq!(deleter.ref_count("_1.si"), 1);
    assert_eq!(listing(&dir), vec!["_1.fdt", "_1.si"]);
    // ...and forgets its recorded file set, which would otherwise be kept
    // for every segment ever written.
    assert!(!deleter.si_files.contains_key("_0"));
    assert!(deleter.si_files.contains_key("_1"));
}

#[test]
fn a_commit_checkpoint_keeps_only_the_last_commit_and_deletes_the_rest() {
    let tmp = tempdir("commit-checkpoint");
    let dir = FsDirectory::open(&tmp);
    let mut deleter =
        IndexFileDeleter::open(&dir, &infos(0, vec![]), DeletionPolicy::KeepOnlyLastCommit)
            .unwrap();

    let a = seed_segment(&dir, "_0", 1, &["_0.fdt"]);
    let first = infos(1, vec![a.clone()]);
    write(&dir, "segments_1", b"placeholder");
    deleter.checkpoint(&first, true).unwrap();
    assert_eq!(deleter.commit_count(), 1);
    assert_eq!(deleter.commit_file_names(), vec!["segments_1"]);
    assert_eq!(deleter.ref_count("segments_1"), 1);

    // A second commit that drops `_0` entirely.
    let b = seed_segment(&dir, "_1", 2, &["_1.fdt"]);
    let second = infos(2, vec![b.clone()]);
    write(&dir, "segments_2", b"placeholder");
    deleter.checkpoint(&second, true).unwrap();

    assert_eq!(deleter.commit_count(), 1, "only the last commit survives");
    assert_eq!(deleter.commit_file_names(), vec!["segments_2"]);
    assert_eq!(
        listing(&dir),
        vec!["_1.fdt", "_1.si", "segments_2"],
        "the superseded commit and every file only it named must be gone"
    );
}

#[test]
fn the_keep_all_policy_never_drops_a_commit_point() {
    let tmp = tempdir("keep-all");
    let dir = FsDirectory::open(&tmp);
    let mut deleter =
        IndexFileDeleter::open(&dir, &infos(0, vec![]), DeletionPolicy::KeepAll).unwrap();

    let a = seed_segment(&dir, "_0", 1, &[]);
    write(&dir, "segments_1", b"placeholder");
    deleter.checkpoint(&infos(1, vec![a]), true).unwrap();
    let b = seed_segment(&dir, "_1", 2, &[]);
    write(&dir, "segments_2", b"placeholder");
    deleter
        .checkpoint(&infos(2, vec![b.clone()]), true)
        .unwrap();

    assert_eq!(deleter.commit_count(), 2);
    assert_eq!(
        listing(&dir),
        vec!["_0.si", "_1.si", "segments_1", "segments_2"]
    );

    // Tightening the policy reclaims what it was holding.
    deleter
        .set_policy(DeletionPolicy::KeepOnlyLastCommit)
        .unwrap();
    assert_eq!(deleter.commit_count(), 1);
    assert_eq!(listing(&dir), vec!["_1.si", "segments_2"]);
}

#[test]
fn refresh_and_delete_new_files_reclaim_only_unreferenced_files() {
    let tmp = tempdir("refresh");
    let dir = FsDirectory::open(&tmp);
    let mut deleter =
        IndexFileDeleter::open(&dir, &infos(0, vec![]), DeletionPolicy::KeepOnlyLastCommit)
            .unwrap();

    let a = seed_segment(&dir, "_0", 1, &["_0.fdt"]);
    deleter.checkpoint(&infos(0, vec![a]), false).unwrap();

    write(&dir, "_5.fdt", b"orphan");
    write(&dir, "pending_segments_4", b"orphan");
    std::fs::write(tmp.join("README.txt"), b"not ours").unwrap();

    deleter.refresh().unwrap();
    assert_eq!(listing(&dir), vec!["README.txt", "_0.fdt", "_0.si"]);

    // `deleteNewFiles` only removes the names it is given, and only the
    // unreferenced ones.
    write(&dir, "_6.fdt", b"orphan");
    deleter
        .delete_new_files(&["_6.fdt".to_string(), "_0.fdt".to_string()])
        .unwrap();
    assert_eq!(
        listing(&dir),
        vec!["README.txt", "_0.fdt", "_0.si"],
        "a referenced file must survive deleteNewFiles"
    );
}

#[test]
fn opening_over_an_existing_commit_reclaims_orphans_and_holds_the_commit() {
    let tmp = tempdir("open-existing");
    let dir = FsDirectory::open(&tmp);
    let a = seed_segment(&dir, "_0", 1, &["_0.fdt"]);
    let committed = infos(1, vec![a]);
    segment_infos::write(&committed, &dir).unwrap();

    // Orphans from a "crash".
    write(&dir, "_7.fdt", b"orphan");
    write(&dir, "pending_segments_9", b"orphan");

    let deleter =
        IndexFileDeleter::open(&dir, &committed, DeletionPolicy::KeepOnlyLastCommit).unwrap();

    assert_eq!(deleter.commit_count(), 1);
    assert_eq!(deleter.commit_file_names(), vec!["segments_1"]);
    assert_eq!(listing(&dir), vec!["_0.fdt", "_0.si", "segments_1"]);
    // The commit holds one reference and the in-memory view another.
    assert_eq!(deleter.ref_count("_0.si"), 2);
    assert_eq!(deleter.ref_count("segments_1"), 1);
}

#[test]
fn a_policy_dropping_the_opened_commit_keeps_the_files_the_writer_holds() {
    #[derive(Debug)]
    struct DropAll;
    impl crate::deletion_policy::IndexDeletionPolicy for DropAll {
        fn on_init(
            &self,
            commits: &mut [crate::deletion_policy::IndexCommit],
        ) -> crate::deletion_policy::Result<()> {
            commits.iter_mut().for_each(|c| c.delete());
            Ok(())
        }
        fn on_commit(
            &self,
            _commits: &mut [crate::deletion_policy::IndexCommit],
        ) -> crate::deletion_policy::Result<()> {
            Ok(())
        }
    }
    let tmp = tempdir("policy-drops-opened");
    let dir = FsDirectory::open(&tmp);
    let a = seed_segment(&dir, "_0", 1, &["_0.fdt"]);
    let committed = infos(1, vec![a]);
    segment_infos::write(&committed, &dir).unwrap();

    let deleter = IndexFileDeleter::open_with_policy(&dir, &committed, Arc::new(DropAll)).unwrap();

    assert_eq!(deleter.commit_count(), 0);
    // The commit file went; the segment the writer opened on did not.
    assert_eq!(listing(&dir), vec!["_0.fdt", "_0.si"]);
    assert_eq!(deleter.ref_count("_0.si"), 1);
}

#[test]
fn a_commit_file_under_a_non_canonical_name_is_rejected_rather_than_deleted() {
    let tmp = tempdir("noncanonical-commit");
    let dir = FsDirectory::open(&tmp);
    let a = seed_segment(&dir, "_0", 1, &[]);
    let committed = infos(1, vec![a]);
    segment_infos::write(&committed, &dir).unwrap();
    // `segments_01` parses to generation 1, but the commit it holds names
    // itself `segments_1`, so nothing ever increfs `segments_01`. Java hits
    // the same case with `IllegalStateException("... refCount=0, which
    // should never happen on init")`; deleting a commit file we do not
    // understand is not an option.
    dir.rename("segments_1", "segments_01").unwrap();

    let err = match IndexFileDeleter::open(&dir, &committed, DeletionPolicy::KeepOnlyLastCommit) {
        Err(e) => e,
        Ok(_) => panic!("a commit file nothing references must be rejected"),
    };
    assert!(
        matches!(&err, Error::UnreferencedCommitFile(n) if n == "segments_01"),
        "{err:?}"
    );
    assert!(listing(&dir).contains(&"segments_01".to_string()));
}

#[test]
fn a_segment_whose_si_is_missing_surfaces_as_an_error_not_a_silent_skip() {
    let tmp = tempdir("missing-si");
    let dir = FsDirectory::open(&tmp);
    let mut deleter =
        IndexFileDeleter::open(&dir, &infos(0, vec![]), DeletionPolicy::KeepOnlyLastCommit)
            .unwrap();
    let phantom = SegmentCommitInfo {
        segment_name: "_ghost".to_string(),
        segment_id: id(9),
        codec_name: "Lucene104".to_string(),
        del_gen: -1,
        del_count: 0,
        field_infos_gen: -1,
        doc_values_gen: -1,
        soft_del_count: 0,
        sci_id: None,
        field_infos_files: vec![],
        dv_update_files: vec![],
        ..Default::default()
    };
    let err = deleter
        .checkpoint(&infos(0, vec![phantom]), false)
        .unwrap_err();
    assert!(matches!(err, Error::Store(_)), "{err:?}");
}

#[test]
fn a_corrupt_commit_file_is_an_error_at_open_not_a_fresh_index() {
    let tmp = tempdir("corrupt-commit");
    let dir = FsDirectory::open(&tmp);
    write(&dir, "segments_1", b"not a commit file at all");
    let err = match IndexFileDeleter::open(&dir, &infos(1, vec![]), DeletionPolicy::KeepAll) {
        Err(e) => e,
        Ok(_) => panic!("a corrupt commit file must be an error"),
    };
    assert!(matches!(err, Error::SegmentInfos(_)), "{err:?}");
}

#[test]
fn generational_liv_field_infos_and_doc_values_files_are_all_refcounted() {
    let tmp = tempdir("generational-files");
    let dir = FsDirectory::open(&tmp);
    let mut deleter =
        IndexFileDeleter::open(&dir, &infos(0, vec![]), DeletionPolicy::KeepOnlyLastCommit)
            .unwrap();

    let mut sci = seed_segment(&dir, "_0", 1, &[]);
    sci.del_gen = 3;
    sci.field_infos_gen = 2;
    sci.field_infos_files = vec!["_0_2.fnm".to_string()];
    sci.dv_update_files = vec![(0, vec!["_0_2_Lucene90_0.dvd".to_string()])];
    for name in ["_0_3.liv", "_0_2.fnm", "_0_2_Lucene90_0.dvd"] {
        write(&dir, name, b"x");
    }

    deleter
        .checkpoint(&infos(0, vec![sci.clone()]), false)
        .unwrap();
    for name in ["_0.si", "_0_3.liv", "_0_2.fnm", "_0_2_Lucene90_0.dvd"] {
        assert_eq!(deleter.ref_count(name), 1, "{name} must be refcounted");
    }

    // Dropping the segment reclaims every generational file with it -- the
    // exact leak `SegmentCommitInfo.files()` exists to prevent.
    deleter.checkpoint(&infos(0, vec![]), false).unwrap();
    assert!(listing(&dir).is_empty(), "{:?}", listing(&dir));
}

/// `IndexWriter.addIndexes(Directory...)` copies a segment's `.si` byte for
/// byte under the new segment name, so the file list in it still names the
/// old segment: a read names every file for the segment it belongs to
/// (`SegmentInfo.setFiles`), and the deleter keeps them all.
#[test]
fn a_copied_si_lists_files_named_for_the_segment_it_now_belongs_to() {
    let tmp = tempdir("copied-si");
    let dir = FsDirectory::open(&tmp);
    let original = seed_segment(&dir, "_0", 1, &["_0.fdt", "_0_Lucene90_0.dvd"]);
    for (from, to) in [
        ("_0.si", "_5.si"),
        ("_0.fdt", "_5.fdt"),
        ("_0_Lucene90_0.dvd", "_5_Lucene90_0.dvd"),
    ] {
        write(&dir, to, &std::fs::read(tmp.join(from)).unwrap());
        dir.delete_file(from).unwrap();
    }
    let copied = SegmentCommitInfo {
        segment_name: "_5".to_string(),
        ..original
    };
    assert_eq!(
        segment_info_files(&dir, &copied).unwrap(),
        ["_5.si", "_5.fdt", "_5_Lucene90_0.dvd"]
    );
    let committed = infos(1, vec![copied]);
    segment_infos::write(&committed, &dir).unwrap();
    IndexFileDeleter::open(&dir, &committed, DeletionPolicy::KeepOnlyLastCommit).unwrap();
    assert_eq!(
        listing(&dir),
        vec!["_5.fdt", "_5.si", "_5_Lucene90_0.dvd", "segments_1"]
    );
}
