//! A standalone, fast checksum-only verifier: the lighter
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
    use rayon::prelude::*;
    let infos = segment_infos::read_latest(dir)?;
    // Collected first and verified in parallel below: every file's CRC is
    // independent, and a report in the same order as before.
    enum Entry {
        Done(FileCheck),
        Verify(String, String),
    }
    let mut files: Vec<Entry> = Vec::new();
    for commit in &infos.segments {
        // A `.si` that won't even parse means we don't know which files
        // belong to this segment; that's still worth surfacing as a single
        // failed "file" entry rather than silently skipping the segment.
        let si = match crate::segment_info::parse_for_commit(
            &match dir.open(&format!("{}.si", commit.segment_name)) {
                Ok(b) => b,
                Err(e) => {
                    files.push(Entry::Done(FileCheck {
                        segment_name: commit.segment_name.clone(),
                        file_name: format!("{}.si", commit.segment_name),
                        passed: false,
                        message: e.to_string(),
                    }));
                    continue;
                }
            },
            commit,
        ) {
            Ok(si) => si,
            Err(e) => {
                files.push(Entry::Done(FileCheck {
                    segment_name: commit.segment_name.clone(),
                    file_name: format!("{}.si", commit.segment_name),
                    passed: false,
                    message: e.to_string(),
                }));
                continue;
            }
        };

        // `SegmentCommitInfo.files()`, not `SegmentInfo.files`: the `.liv`,
        // generational field-infos and doc-values-update files a delete or
        // DV-update round produced are recorded on the *commit*, never in the
        // already-written `.si`. Verifying only `si.files` silently skipped
        // exactly the files most likely to have been written last.
        for file_name in commit.files(&si.files) {
            files.push(Entry::Verify(commit.segment_name.clone(), file_name));
        }
    }

    // The commit file itself. `segments_N` is the one file whose corruption
    // makes the whole index unopenable, and it is in no segment's file list.
    // `read_latest` above already verified its checksum on the way in, so a
    // failure here is unreachable in practice -- it is recorded anyway so the
    // report's file list is the commit's real file list.
    if let Some(name) = lucene_store::directory::segments_file_name(infos.generation) {
        files.push(Entry::Verify(String::new(), name));
    }

    let files = files
        .into_par_iter()
        .map(|entry| match entry {
            Entry::Done(check) => check,
            Entry::Verify(segment, file) => verify_file(dir, &segment, &file),
        })
        .collect();
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
    // before a byte-copy merge (`StoredFieldsReader::check_integrity`)
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
mod tests;
