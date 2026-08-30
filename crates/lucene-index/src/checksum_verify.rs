//! A standalone, fast checksum-only verifier (task #217): the lighter
//! sibling of [`crate::check_index`]'s much deeper `CheckIndex`-equivalent.
//!
//! Real Lucene's `CheckIndex` supports a `-fastCheck`-style mode that only
//! verifies each file's codec footer checksum, skipping every structural
//! cross-check (`.si` doc counts vs `.liv`, `.fnm` flags vs which files
//! exist, postings re-derivation, points-tree invariants, etc). This module
//! is that mode's equivalent: it opens the latest commit in a directory,
//! and for every file every segment owns -- [`crate::segment_infos::SegmentCommitInfo::files`],
//! i.e. the `.si`'s own list *plus* the commit-level `.liv`/field-infos/
//! doc-values-update files, plus the `segments_N` file itself -- re-reads the
//! whole file and recomputes its CRC-32 via
//! [`lucene_store::codec_util::check_whole_file_footer`] -- the same
//! `CodecUtil.checksumEntireFile` helper `merge.rs` runs on a merge source
//! before a byte-copy and that `check_index`'s `file:*` checks use (not
//! [`lucene_store::codec_util::retrieve_checksum`], which only validates the
//! footer's *shape* without recomputing the CRC over the payload -- this
//! tool's whole point is catching bit-rot/truncation/corruption in the
//! payload bytes themselves, which `retrieve_checksum` by design does not
//! detect).
//!
//! This is deliberately **not** a replacement for `check_index.rs`: it does
//! not open `.fnm`/`.liv`/stored-fields/postings/points and cross-checks
//! nothing between files. It is useful precisely because it is cheap -- one
//! sequential read and one CRC-32 pass per file, no format-specific
//! decoding at all -- so it's a reasonable thing to run as a fast pre-flight
//! before the full `check_index.rs` (or in a tight loop / on every commit)
//! where the deeper tool would be too slow.
// This module is *audited* against the arithmetic gate: no `+`/`-`/`*`/`<<`
// in it may panic, because a verifier that panics on a corrupt file has
// failed at its one job (and through the FFI a panic in a debug build takes
// the JVM with it). Every operation here is `checked_*` with a reported
// corruption, `saturating_*` where the saturation is unreachable or is itself
// the honest answer for a counter, or a plain operator under an `#[allow]`
// carrying an `// ARITH:` proof. See `docs/arithmetic-gate.md`.
#![deny(clippy::arithmetic_side_effects)]

use lucene_store::codec_util;
use lucene_store::Directory;

use crate::segment_infos;

/// One file's checksum-verification outcome.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileCheck {
    /// The segment this file belongs to (e.g. `_0`), or `""` for the
    /// commit-level `segments_N` file, which belongs to no segment.
    pub segment_name: String,
    /// The file name as recorded in `SegmentCommitInfo::files` (e.g.
    /// `_0.fdt`, `_0_1.liv`).
    pub file_name: String,
    pub passed: bool,
    /// `"ok"` on success, or the error message on failure (missing file,
    /// truncated footer, checksum mismatch, etc).
    pub message: String,
}

/// Every file check performed against a directory's latest commit, plus a
/// summary. Mirrors `check_index.rs::CheckResult`'s "list of independent
/// named outcomes" shape, but flat across all segments since there is only
/// one kind of check here.
#[derive(Debug, Clone, Default)]
pub struct VerifyReport {
    pub files: Vec<FileCheck>,
}

impl VerifyReport {
    /// Total files checked (attempted), regardless of outcome.
    pub fn total(&self) -> usize {
        self.files.len()
    }

    /// Number of files that failed checksum verification.
    pub fn failed_count(&self) -> usize {
        self.files.iter().filter(|f| !f.passed).count()
    }

    /// Whether every file checked passed. Vacuously `true` for a commit
    /// with zero files (an empty index), matching `Iterator::all`'s
    /// convention.
    pub fn all_passed(&self) -> bool {
        self.files.iter().all(|f| f.passed)
    }

    /// The subset of files that failed, for reporting.
    pub fn failures(&self) -> Vec<&FileCheck> {
        self.files.iter().filter(|f| !f.passed).collect()
    }
}

/// Verifies every file every segment in `dir`'s latest commit declares.
///
/// Returns `Err` only if the commit itself (`segments_N`) can't be found or
/// parsed -- a per-file open/checksum failure is *not* an `Err` here, it's
/// recorded as a failed [`FileCheck`] in the returned report so callers see
/// every failure, not just the first one.
pub fn verify_directory(dir: &dyn Directory) -> segment_infos::Result<VerifyReport> {
    let infos = segment_infos::read_latest(dir)?;
    let mut files = Vec::new();
    for commit in &infos.segments {
        // A `.si` that won't even parse means we don't know which files
        // belong to this segment; that's still worth surfacing as a single
        // failed "file" entry rather than silently skipping the segment.
        let si = match crate::segment_info::parse(
            &match dir.open(&format!("{}.si", commit.segment_name)) {
                Ok(b) => b,
                Err(e) => {
                    files.push(FileCheck {
                        segment_name: commit.segment_name.clone(),
                        file_name: format!("{}.si", commit.segment_name),
                        passed: false,
                        message: e.to_string(),
                    });
                    continue;
                }
            },
            &commit.segment_id,
        ) {
            Ok(si) => si,
            Err(e) => {
                files.push(FileCheck {
                    segment_name: commit.segment_name.clone(),
                    file_name: format!("{}.si", commit.segment_name),
                    passed: false,
                    message: e.to_string(),
                });
                continue;
            }
        };

        // `SegmentCommitInfo.files()`, not `SegmentInfo.files`: the `.liv`,
        // generational field-infos and doc-values-update files a delete or
        // DV-update round produced are recorded on the *commit*, never in the
        // already-written `.si`. Verifying only `si.files` silently skipped
        // exactly the files most likely to have been written last.
        for file_name in &commit.files(&si.files) {
            files.push(verify_file(dir, &commit.segment_name, file_name));
        }
    }

    // The commit file itself. `segments_N` is the one file whose corruption
    // makes the whole index unopenable, and it is in no segment's file list.
    // `read_latest` above already verified its checksum on the way in, so a
    // failure here is unreachable in practice -- it is recorded anyway so the
    // report's file list is the commit's real file list.
    if let Some(name) = lucene_store::directory::segments_file_name(infos.generation) {
        files.push(verify_file(dir, "", &name));
    }

    Ok(VerifyReport { files })
}

/// Opens and checksum-verifies a single file, returning a [`FileCheck`]
/// rather than propagating an error -- every file in a directory should be
/// checked even if an earlier one failed.
fn verify_file(dir: &dyn Directory, segment_name: &str, file_name: &str) -> FileCheck {
    let bytes = match dir.open(file_name) {
        Ok(b) => b,
        Err(e) => {
            return FileCheck {
                segment_name: segment_name.to_string(),
                file_name: file_name.to_string(),
                passed: false,
                message: e.to_string(),
            };
        }
    };

    // `CodecUtil.checksumEntireFile` -- the same helper `merge.rs` calls
    // before a byte-copy merge (batch c4's `StoredFieldsReader::check_integrity`)
    // and the same one `check_index`'s `file:*` checks now use, rather than a
    // third hand-rolled seek-to-footer-then-`check_footer` sequence.
    // `check_whole_file_footer` reports the "file shorter than a footer" case
    // itself, via `check_footer`'s own `total_len < FOOTER_LENGTH` guard.
    let total_len = bytes.len();
    let payload_end = total_len.saturating_sub(codec_util::FOOTER_LENGTH);
    let result =
        codec_util::check_whole_file_footer(&bytes, payload_end).map_err(|e| e.to_string());

    match result {
        Ok(_) => FileCheck {
            segment_name: segment_name.to_string(),
            file_name: file_name.to_string(),
            passed: true,
            message: "ok".to_string(),
        },
        Err(message) => FileCheck {
            segment_name: segment_name.to_string(),
            file_name: file_name.to_string(),
            passed: false,
            message,
        },
    }
}

#[cfg(test)]
mod tests {
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
        let updated =
            crate::deletes::apply_deletes(&dir, &infos.segments[0], None, 2, [1]).unwrap();
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
}
