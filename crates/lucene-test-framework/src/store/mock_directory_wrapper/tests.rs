#![cfg(test)]
use super::{unknown_commit_generation, Throttle};

#[test]
fn unknown_commit_generations_follow_java_utf16_and_radix36_rules() {
    for (name, expected) in [
        ("segments", 0),
        ("segments_4", 4),
        ("segmentsX4", 4),
        ("segments§4", 4),
        ("segments\n4", 4),
        ("segments_+4", 4),
        ("segments_-4", -4),
        ("segments_٤", 4),
        ("segments_４", 4),
        ("segments_ａ", 10),
        ("segments_Ｚ", 35),
        ("segments_aZ", 395),
        ("segments_+000", 0),
        ("segments_-0", 0),
        ("segments_1y2p0ij32e8e7", i64::MAX),
        ("segments_-1y2p0ij32e8e8", i64::MIN),
    ] {
        assert_eq!(
            unknown_commit_generation(name).unwrap(),
            expected,
            "{name:?}"
        );
    }
    for name in [
        "other_4",
        "segments.gen",
        "segments_",
        "segments_+",
        "segments_-",
        "segments_++4",
        "segments_ 4",
        "segments_4 ",
        "segments_4\n",
        "segments_＋4",
        "segments_𝟜",
        "segments😀4",
        "segments_1y2p0ij32e8e8",
        "segments_-1y2p0ij32e8e9",
        "segments_zzzzzzzzzzzzzzzzzzzz",
    ] {
        let error = unknown_commit_generation(name).unwrap_err();
        assert!(matches!(error, super::Error::Corrupted(_)), "{error}");
        assert!(error.to_string().contains(&format!("{name:?}")), "{error}");
    }
}

fn validation_case() -> (
    lucene_index::segment_infos::SegmentInfos,
    lucene_index::segment_infos::SegmentCommitInfo,
    lucene_index::segment_info::SegmentInfo,
) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/data/merge_policies/commits_index");
    let infos =
        lucene_index::segment_infos::parse(&std::fs::read(path.join("segments_3")).unwrap(), 3)
            .unwrap();
    let commit = infos.segments[0].clone();
    let si = lucene_index::segment_info::parse_for_codec(
        &std::fs::read(path.join(format!("{}.si", commit.segment_name))).unwrap(),
        &commit.segment_id,
        &commit.codec_name,
    )
    .unwrap();
    (infos, commit, si)
}

#[test]
fn unknown_scan_dispatches_simpletext_metadata_and_uses_liv_suffix() {
    use lucene_index::segment_info::{self, LuceneVersion};
    use lucene_store::{ByteBuffersDirectory, DataOutput, Directory};
    let bytes = include_bytes!("../../../../../fixtures/data/simpletext_index_sort/_1.si");
    let id = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let mut text = segment_info::parse_simpletext(bytes, "_1", &id).unwrap();
    let (mut infos, mut commit, _) = validation_case();
    commit.segment_name = "_1".into();
    commit.segment_id = id;
    commit.codec_name = "SimpleText".into();
    commit.del_count = 0;
    commit.soft_del_count = 0;
    commit.del_gen = 1;
    commit.field_infos_files.clear();
    commit.dv_update_files.clear();
    infos.segments = vec![commit.clone()];
    let parsed = super::unknown_segment_info(bytes, &commit).unwrap();
    assert_eq!(parsed.doc_count, 7);
    assert!(
        parsed.index_sort.is_some(),
        "typed SimpleText sort must be decoded"
    );
    let error = super::validate_unknown_segment(&infos, &commit, &parsed, &mut 0).unwrap_err();
    assert!(error.to_string().contains("minVersion"), "{error}");
    text.version = lucene_util::version::Version::new(10, 0, 0, 2).unwrap();
    text.min_version = Some(lucene_util::version::Version::LUCENE_9_0_0);
    infos.min_segment_lucene_version = Some(lucene_index::segment_infos::LuceneVersion {
        major: 10,
        minor: 0,
        bugfix: 0,
    });
    let bytes = segment_info::write_simpletext(&mut text).unwrap();
    let parsed = super::unknown_segment_info(&bytes, &commit).unwrap();
    assert_eq!(
        parsed.version,
        LuceneVersion {
            major: 10,
            minor: 0,
            bugfix: 0
        }
    );
    let dir = ByteBuffersDirectory::new();
    let mut output = dir.create_output("_1.si").unwrap();
    output.write_bytes(&bytes);
    output.close().unwrap();
    assert_eq!(
        super::unknown_commit_files(&dir, &infos).unwrap(),
        std::collections::HashSet::from([
            "segments_3".to_owned(),
            "_1.si".to_owned(),
            "_1.other".to_owned(),
            "_1_1.liv".to_owned(),
        ])
    );
    assert!(super::unknown_segment_info(&bytes[..20], &commit).is_err());
    commit.segment_id = [0; 16];
    assert!(super::unknown_segment_info(&bytes, &commit).is_err());
}
#[test]
fn unknown_scan_normalizes_names_like_java() {
    for (file, expected) in [
        ("_old.si", "_new.si"),
        ("_old_1_Lucene90_0.dvd", "_new_1_Lucene90_0.dvd"),
        ("_old.ext_suffix.more", "_new_suffix.more"),
        ("plain.ext", "_new.ext"),
        ("plain", "_newplain"),
        ("", "_new"),
        ("_", "_new_"),
        ("__", "_new_"),
        ("😀_suffix.ext", "_new_suffix.ext"),
    ] {
        assert_eq!(
            super::unknown_named_for_segment("_new", file),
            expected,
            "{file:?}"
        );
    }
}

#[test]
fn unknown_scan_unions_normalized_names_and_validates_only_si_names() {
    let (_, mut commit, mut si) = validation_case();
    commit.segment_name = "_new".into();
    commit.del_gen = 1;
    commit.field_infos_files = vec![
        "_other.fnm".into(),
        "_old_1.dvd".into(),
        "no.dot_TMP".into(),
    ];
    commit.dv_update_files = vec![
        (3, vec!["_ignored_99.dvd".into()]),
        (3, vec!["_older_1.dvd".into(), "_other.TMP".into()]),
    ];
    si.files = vec!["_old.si".into(), "_other.si".into(), "_old_1.dvd".into()];
    assert_eq!(
        super::unknown_segment_files(&commit, &si).unwrap(),
        std::collections::HashSet::from([
            "_new.si".to_owned(),
            "_new_1.dvd".to_owned(),
            "_new.fnm".to_owned(),
            "_new_1.liv".to_owned(),
            "_new_TMP".to_owned(),
            "_new.TMP".to_owned(),
        ])
    );
    commit.del_gen = -1;
    si.files.clear();
    assert!(
        !super::unknown_segment_files(&commit, &si)
            .unwrap()
            .contains("_new.si"),
        "Java does not synthesize a missing .si self-listing"
    );
    for bad in ["not-a-codec", "_old.TMP", "_old.tMp", "_old.x\n"] {
        si.files = vec![bad.into()];
        let error = super::unknown_segment_files(&commit, &si).unwrap_err();
        assert!(
            error.to_string().contains("invalid codec filename"),
            "{error}"
        );
    }
}
#[test]
fn unknown_scan_validates_created_major_even_for_empty_commits() {
    let (mut infos, _, _) = validation_case();
    infos.segments.clear();
    let dir = lucene_store::ByteBuffersDirectory::new();
    for major in [5, 11] {
        infos.index_created_version_major = major;
        let error = super::unknown_commit_files(&dir, &infos).unwrap_err();
        assert!(
            error.to_string().contains("indexCreatedVersionMajor"),
            "{error}"
        );
    }
    for major in [6, 10] {
        infos.index_created_version_major = major;
        assert_eq!(
            super::unknown_commit_files(&dir, &infos).unwrap(),
            std::collections::HashSet::from(["segments_3".to_owned()])
        );
    }
}
#[test]
fn unknown_scan_rejects_invalid_deletion_counts() {
    let (infos, mut commit, si) = validation_case();
    for (hard, soft) in [
        (-1, 0),
        (0, -1),
        (si.doc_count + 1, 0),
        (0, si.doc_count + 1),
        (si.doc_count, 1),
    ] {
        commit.del_count = hard;
        commit.soft_del_count = soft;
        let mut total = 0;
        let error = super::validate_unknown_segment(&infos, &commit, &si, &mut total).unwrap_err();
        assert!(
            error.to_string().contains("invalid deletion counts"),
            "{error}"
        );
        assert_eq!(total, 0, "failed validation must not advance total");
    }
    commit.del_count = si.doc_count;
    commit.soft_del_count = 0;
    let mut total = 0;
    super::validate_unknown_segment(&infos, &commit, &si, &mut total).unwrap();
    assert_eq!(total, si.doc_count as u64);
}

#[test]
fn unknown_scan_checks_segment_versions_against_commit() {
    let (mut infos, commit, mut si) = validation_case();
    let original_si = si.clone();
    let original_infos = infos.clone();
    let min = infos.min_segment_lucene_version.as_mut().unwrap();
    *min = lucene_index::segment_infos::LuceneVersion {
        major: si.version.major,
        minor: si.version.minor,
        bugfix: si.version.bugfix + 1,
    };
    let error = super::validate_unknown_segment(&infos, &commit, &si, &mut 0).unwrap_err();
    assert!(
        error.to_string().contains("minSegmentLuceneVersion"),
        "{error}"
    );
    infos = original_infos;
    infos.min_segment_lucene_version = None;
    infos.index_created_version_major = si.version.major + 1;
    let error = super::validate_unknown_segment(&infos, &commit, &si, &mut 0).unwrap_err();
    assert!(
        error.to_string().contains("indexCreatedVersionMajor"),
        "{error}"
    );
    infos.index_created_version_major = 6;
    for min_version in [
        None,
        Some(lucene_index::segment_info::LuceneVersion {
            major: 8,
            minor: 11,
            bugfix: 0,
        }),
    ] {
        si.min_version = min_version;
        let error = super::validate_unknown_segment(&infos, &commit, &si, &mut 0).unwrap_err();
        assert!(error.to_string().contains("minVersion"), "{error}");
    }
    si = original_si;
    si.min_version = Some(lucene_index::segment_info::LuceneVersion {
        major: 9,
        minor: 0,
        bugfix: 0,
    });
    let mut total = 0;
    super::validate_unknown_segment(&infos, &commit, &si, &mut total).unwrap();
    assert_eq!(total, si.doc_count as u64);
}

#[test]
fn unknown_scan_enforces_total_max_doc_including_deleted_documents() {
    let (infos, mut commit, mut si) = validation_case();
    let max = lucene_index::index_writer::MAX_DOCS as u64;
    commit.del_count = 0;
    commit.soft_del_count = 0;
    si.doc_count = max as i32;
    let mut total = 0;
    super::validate_unknown_segment(&infos, &commit, &si, &mut total).unwrap();
    assert_eq!(total, max);
    si.doc_count = 1;
    commit.del_count = 1;
    let error = super::validate_unknown_segment(&infos, &commit, &si, &mut total).unwrap_err();
    assert!(error.to_string().contains("total maxDoc"), "{error}");
    assert_eq!(total, max);
    si.doc_count = max as i32 + 1;
    let mut total = 0;
    let error = super::validate_unknown_segment(&infos, &commit, &si, &mut total).unwrap_err();
    assert!(error.to_string().contains("total maxDoc"), "{error}");
    assert_eq!(total, 0);
}
/// `ThrottledIndexOutput.getDelay`, quirks included: the observed rate
/// is `(elapsedNanos / pendingBytes) * 1e9` in integer arithmetic, so a
/// delegate faster than one nanosecond per byte reads as 0 bytes/s and
/// is never slowed, and the delay is whole milliseconds.
#[test]
fn delay_follows_java_including_its_integer_rate() {
    let mut t = Throttle::new(5_000_000, 3);
    // Under `minBytesWritten`, and not closing: no delay yet.
    t.pending_bytes = 1000;
    t.time_elapsed_nanos = 10_000;
    assert_eq!(t.delay(false), 0);
    // 100 kB written at 2 ns/byte against 5 MB/s: 20 ms expected, the
    // 0.2 ms already spent rounds away.
    t.pending_bytes = 100_000;
    t.time_elapsed_nanos = 200_000;
    assert_eq!(t.delay(false), 20);
    assert_eq!((t.pending_bytes, t.time_elapsed_nanos), (0, 0));
    // Faster than 1 ns/byte: Java's rate truncates to 0, no delay.
    t.pending_bytes = 100_000;
    t.time_elapsed_nanos = 50_000;
    assert_eq!(t.delay(true), 0);
    // Closing ignores `minBytesWritten`.
    let mut t = Throttle::new(1000, 0);
    t.pending_bytes = 10;
    t.time_elapsed_nanos = 100;
    assert_eq!(t.delay(true), 10);
    Throttle::sleep(0);
    Throttle::sleep(1);
}
