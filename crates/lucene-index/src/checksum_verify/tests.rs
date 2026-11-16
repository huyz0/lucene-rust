#![cfg(test)]
// Test code opts out of the arithmetic gate at the module boundary: the
// gate exists for values read off disk, not for a test's own index
// arithmetic. See `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use super::*;
use lucene_codecs::field_infos::{
    DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions, VectorEncoding,
    VectorSimilarityFunction,
};
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
use lucene_store::FsDirectory;

fn stored_only_field(name: &str, number: i32) -> FieldInfo {
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

fn doc(id: &str, body: &str) -> Document {
    Document {
        fields: vec![
            StoredField {
                field_number: 0,
                value: FieldValue::String(id.to_string()),
            },
            StoredField {
                field_number: 1,
                value: FieldValue::String(body.to_string()),
            },
        ],
    }
}

fn lucene_version() -> crate::segment_info::LuceneVersion {
    crate::segment_info::LuceneVersion {
        major: 10,
        minor: 0,
        bugfix: 0,
    }
}

fn sis_lucene_version() -> segment_infos::LuceneVersion {
    let v = lucene_version();
    segment_infos::LuceneVersion {
        major: v.major,
        minor: v.minor,
        bugfix: v.bugfix,
    }
}

/// Matches `check_index.rs`'s convention -- and, like it, returns a guard
/// that removes the directory unless the test is panicking. The old form
/// leaked one directory per test into a tmpfs; see
/// [`lucene_util::test_support`].
fn tempdir() -> lucene_util::test_support::TempDir {
    lucene_util::test_support::TempDir::new("checksum-verify")
}

/// Writes a real, valid 2-segment commit (mirroring
/// `examples/write_multi_segment_commit_fixture.rs`'s shape) to a fresh
/// temp directory, returning the directory path.
fn write_multi_segment_commit() -> lucene_util::test_support::TempDir {
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), stored_only_field("body", 1)];

    let segment0_docs = vec![doc("1", "the quick brown fox"), doc("2", "lazy dog")];
    let segment1_docs = vec![doc("3", "pack my box"), doc("4", "quick daft zebras")];

    let sci0 = crate::segment_writer::flush_stored_only_segment(
        &dir,
        "_0",
        *b"rustwrittenseg00",
        "Lucene104",
        lucene_version(),
        &fields,
        &segment0_docs,
        false,
    )
    .unwrap();
    let sci1 = crate::segment_writer::flush_stored_only_segment(
        &dir,
        "_1",
        *b"rustwrittenseg01",
        "Lucene104",
        lucene_version(),
        &fields,
        &segment1_docs,
        false,
    )
    .unwrap();

    let sis = segment_infos::SegmentInfos {
        id: *b"chksumtestcommit",
        generation: 1,
        format_version: 0,
        lucene_version: sis_lucene_version(),
        index_created_version_major: lucene_version().major,
        version: 2,
        counter: 2,
        min_segment_lucene_version: Some(sis_lucene_version()),
        segments: vec![sci0, sci1],
        user_data: vec![],
    };
    segment_infos::write(&sis, &dir).unwrap();
    tmp
}

#[test]
fn clean_directory_all_pass() {
    let tmp = write_multi_segment_commit();
    let dir = FsDirectory::open(&tmp);
    let report = verify_directory(&dir).unwrap();

    assert!(report.total() > 0);
    assert!(report.all_passed());
    assert_eq!(report.failed_count(), 0);
    assert!(report.failures().is_empty());
    // Two segments' worth of files: at minimum .si/.fdt/.fdx/.fdm/.fnm each.
    assert!(report.total() >= 8);
}

#[test]
fn corrupted_payload_byte_is_detected() {
    let tmp = write_multi_segment_commit();
    let dir_path = &tmp;

    // Find a real segment data file (not .si, not segments_N) and flip a
    // byte inside its payload (not the footer) -- exactly the "hardware
    // problem"/bit-rot case `check_footer`'s doc comment describes.
    let mut target: Option<std::path::PathBuf> = None;
    for entry in std::fs::read_dir(dir_path).unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name().into_string().unwrap();
        if name.ends_with(".fdt") {
            target = Some(entry.path());
            break;
        }
    }
    let target = target.expect("expected at least one .fdt file");

    let mut bytes = std::fs::read(&target).unwrap();
    assert!(bytes.len() > codec_util::FOOTER_LENGTH + 1);
    // Flip a byte well before the footer so the payload (not the
    // footer's own magic/algorithm/checksum fields) is corrupted.
    let flip_at = bytes.len() - codec_util::FOOTER_LENGTH - 1;
    bytes[flip_at] ^= 0xFF;
    std::fs::write(&target, &bytes).unwrap();

    let dir = FsDirectory::open(dir_path);
    let report = verify_directory(&dir).unwrap();

    assert!(!report.all_passed());
    assert_eq!(report.failed_count(), 1);
    let failure = &report.failures()[0];
    assert!(failure.file_name.ends_with(".fdt"));
    assert!(failure.message.contains("checksum failed"));
}

#[test]
fn missing_file_is_reported_as_failure() {
    let tmp = write_multi_segment_commit();
    let dir_path = &tmp;

    let mut target: Option<std::path::PathBuf> = None;
    for entry in std::fs::read_dir(dir_path).unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name().into_string().unwrap();
        if name.ends_with(".fnm") {
            target = Some(entry.path());
            break;
        }
    }
    let target = target.expect("expected at least one .fnm file");
    std::fs::remove_file(&target).unwrap();

    let dir = FsDirectory::open(dir_path);
    let report = verify_directory(&dir).unwrap();

    assert!(!report.all_passed());
    assert_eq!(report.failed_count(), 1);
    assert!(report.failures()[0].file_name.ends_with(".fnm"));
}

#[test]
fn truncated_file_is_reported_as_failure() {
    let tmp = write_multi_segment_commit();
    let dir_path = &tmp;

    let mut target: Option<std::path::PathBuf> = None;
    for entry in std::fs::read_dir(dir_path).unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name().into_string().unwrap();
        if name.ends_with(".fdx") {
            target = Some(entry.path());
            break;
        }
    }
    let target = target.expect("expected at least one .fdx file");
    let bytes = std::fs::read(&target).unwrap();
    std::fs::write(&target, &bytes[..2]).unwrap();

    let dir = FsDirectory::open(dir_path);
    let report = verify_directory(&dir).unwrap();

    assert!(!report.all_passed());
    assert_eq!(report.failed_count(), 1);
    assert!(report.failures()[0].message.contains("truncated"));
}

/// A `.liv` file lives on the `SegmentCommitInfo`, never in the `.si`'s
/// own file list -- verifying only `si.files` skipped it entirely, which
/// meant a corrupted live-docs file (the most recently written file in
/// any index that has taken deletes) passed a "clean" checksum run.
#[test]
fn live_docs_file_is_verified_and_its_corruption_detected() {
    let tmp = write_multi_segment_commit();
    let dir = FsDirectory::open(&tmp);

    // Delete a doc from _0 so a real `_0_1.liv` exists, then republish the
    // commit with the bumped SegmentCommitInfo.
    let mut infos = segment_infos::read_latest(&dir).unwrap();
    let updated = crate::deletes::apply_deletes(&dir, &infos.segments[0], None, 2, [1]).unwrap();
    let liv_name = crate::deletes::liv_file_name(&updated.segment_name, updated.del_gen);
    infos.segments[0] = updated;
    infos.generation = 2;
    segment_infos::write(&infos, &dir).unwrap();

    let report = verify_directory(&dir).unwrap();
    assert!(
        report.files.iter().any(|f| f.file_name == liv_name),
        "the .liv file must be in the verified set: {:?}",
        report
            .files
            .iter()
            .map(|f| &f.file_name)
            .collect::<Vec<_>>()
    );
    assert!(report.all_passed(), "{:?}", report.failures());

    // Corrupt it and confirm the checksum pass catches it.
    let path = tmp.join(&liv_name);
    let mut bytes = std::fs::read(&path).unwrap();
    let flip_at = bytes.len() - codec_util::FOOTER_LENGTH - 1;
    bytes[flip_at] ^= 0xFF;
    std::fs::write(&path, &bytes).unwrap();

    let report = verify_directory(&dir).unwrap();
    assert_eq!(report.failed_count(), 1);
    assert_eq!(report.failures()[0].file_name, liv_name);
}

/// The commit file itself is in no segment's file list, so it has to be
/// added explicitly or the report silently omits the one file whose loss
/// makes the index unopenable.
#[test]
fn segments_file_itself_is_in_the_verified_set() {
    let tmp = write_multi_segment_commit();
    let dir = FsDirectory::open(&tmp);
    let report = verify_directory(&dir).unwrap();
    assert!(
        report.files.iter().any(|f| f.file_name == "segments_1"),
        "segments_1 must be verified"
    );
}

#[test]
fn missing_commit_is_an_error() {
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    assert!(verify_directory(&dir).is_err());
}

#[test]
fn report_defaults_to_empty_and_all_passed() {
    let report = VerifyReport::default();
    assert_eq!(report.total(), 0);
    assert_eq!(report.failed_count(), 0);
    assert!(report.all_passed());
}

/// A segment whose `.si` is *missing* must be reported as a single failed
/// `.si` entry, not silently skipped: without the `.si` this verifier has
/// no file list for that segment, so skipping it would report "all
/// passed" for an index whose whole segment is unreadable.
#[test]
fn a_missing_si_is_reported_and_the_other_segment_still_verified() {
    let tmp = write_multi_segment_commit();
    let dir = FsDirectory::open(&tmp);
    let infos = segment_infos::read_latest(&dir).unwrap();
    let victim = infos.segments[0].segment_name.clone();
    let survivor = infos.segments[1].segment_name.clone();
    let clean_total = verify_directory(&dir).unwrap().total();
    std::fs::remove_file(tmp.join(format!("{victim}.si"))).unwrap();

    let dir = FsDirectory::open(&tmp);
    let report = verify_directory(&dir).unwrap();
    assert!(!report.all_passed());
    let failures = report.failures();
    assert_eq!(failures.len(), 1, "{failures:?}");
    assert_eq!(failures[0].file_name, format!("{victim}.si"));
    assert_eq!(failures[0].segment_name, victim);
    // The rest of the commit is still verified rather than abandoned.
    assert!(
        report
            .files
            .iter()
            .any(|f| f.segment_name == survivor && f.passed),
        "the surviving segment's files must still be checked: {:?}",
        report
            .files
            .iter()
            .map(|f| &f.file_name)
            .collect::<Vec<_>>()
    );
    assert!(report.total() < clean_total);

    std::fs::remove_dir_all(&tmp).ok();
}

/// The same, for a `.si` that exists but does not parse. This is a
/// separate arm from the missing-file one above (a different `match`
/// branch) and a different failure message, and it is the more likely of
/// the two in practice -- a torn write leaves the file present.
#[test]
fn an_unparsable_si_is_reported_rather_than_skipping_the_segment() {
    let tmp = write_multi_segment_commit();
    let dir = FsDirectory::open(&tmp);
    let infos = segment_infos::read_latest(&dir).unwrap();
    let victim = infos.segments[0].segment_name.clone();
    let si_path = tmp.join(format!("{victim}.si"));

    // Truncate to the codec header alone: the header still validates, so
    // this reaches `segment_info::parse`'s own decode failure rather than
    // the open failure above.
    let bytes = std::fs::read(&si_path).unwrap();
    std::fs::write(&si_path, &bytes[..40.min(bytes.len())]).unwrap();

    let dir = FsDirectory::open(&tmp);
    let report = verify_directory(&dir).unwrap();
    let failures = report.failures();
    assert_eq!(failures.len(), 1, "{failures:?}");
    assert_eq!(failures[0].file_name, format!("{victim}.si"));
    assert!(
        !failures[0].message.is_empty(),
        "the parse error must be carried through to the report"
    );

    std::fs::remove_dir_all(&tmp).ok();
}
