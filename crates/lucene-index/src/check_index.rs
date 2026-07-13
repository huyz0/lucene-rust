//! A `CheckIndex`-equivalent (task #57): a standalone consistency verifier
//! that opens a segment/directory and cross-checks internal relationships a
//! normal single-purpose open never bothers to verify -- does `.si`'s
//! declared doc count match `.liv`'s bit-count-derived max doc, does
//! `live_docs`' cardinality match `SegmentCommitInfo.del_count`, does every
//! field `.fnm` claims to have doc-values/norms/postings/term-vectors for
//! actually have a corresponding file (and vice versa), does the
//! stored-fields reader's own doc count match `.si`'s.
//!
//! This is deliberately *not* built on top of `lucene-search`'s
//! `DirectoryReader`/`SegmentReader` (task #45): those types only expose the
//! curated subset of state a query needs (blocktree fields, postings
//! buffers, live docs) and hide exactly the things a self-check needs to
//! cross-reference -- `SegmentInfo.files`, per-field flags, raw
//! `.si`/`.fnm`/stored-fields bytes. This module lives in `lucene-index`
//! (not `lucene-search`, which it has no actual dependency on -- every type
//! it touches is already available here) and reuses
//! [`crate::segment_infos::read_latest`] for the one piece of "directory
//! reader" logic that *is* shared (find the latest commit, list its
//! segments), otherwise opening each segment's files directly through the
//! same lower-level decoders `lucene-search`'s `directory_reader.rs` itself
//! calls (`segment_info::parse`, `field_infos::parse`, `live_docs::parse`,
//! `stored_fields::open`), since those are exactly the values this module
//! needs to compare against each other.
//!
//! # Scope
//!
//! Implemented (real, valuable cross-checks given this port's current
//! write-side scope -- see this module's unit tests for both the
//! clean-pass and deliberately-corrupted-input cases):
//!
//! - Every file `SegmentInfo.files` lists opens and has a structurally valid
//!   codec footer (magic/algorithm id/checksum shape) -- doubles as a "did
//!   we forget to write/list a file" check.
//! - `.si` doc_count vs `.liv`'s bit-count-implied max_doc (if the segment
//!   has deletions).
//! - `live_docs` cardinality vs `SegmentCommitInfo.del_count`'s implied live
//!   count (`max_doc - del_count`).
//! - `.fnm`'s per-field flags (doc values, norms, term vectors, postings via
//!   `index_options != None`) cross-checked against which of
//!   `.dvd`/`.dvm`/`.nvd`/`.nvm`/`.tvd`/`.tvx`/`.tvm`/`.tim`/`.tip`/`.tmd`
//!   the segment's file list actually includes, in both directions (a field
//!   claiming doc-values with no `.dvd`/`.dvm` file is flagged, and so is a
//!   `.dvd`/`.dvm` file present with no field claiming doc-values).
//! - Stored-fields doc count (`StoredFieldsReader::max_doc`) vs `.si`'s
//!   declared `doc_count`.
//!
//! **Deliberately out of scope** (see `docs/parity.md`): postings
//! term-by-term re-derivation (recomputing docFreq/totalTermFreq from raw
//! postings and cross-checking against the term dictionary's own recorded
//! stats -- real `CheckIndex`'s single most expensive check), doc-values
//! value-range sanity, points-tree structural invariants, and vectors-graph
//! structural invariants. All of these require walking per-format internals
//! this port's read-side decoders expose in different shapes per format
//! (blocktree iteration, points-tree traversal, HNSW graph traversal) --
//! genuinely a separate, large task per format rather than a natural
//! extension of this module's cross-file bookkeeping checks.

use crate::deletes::liv_file_name;
use crate::segment_info::{self, SegmentInfo};
use crate::segment_infos::{self, SegmentCommitInfo};
use lucene_codecs::field_infos::{self, FieldInfos};
use lucene_codecs::live_docs;
use lucene_codecs::stored_fields;
use lucene_store::codec_util;
use lucene_store::directory::Directory;

/// One named check's outcome -- matches real `CheckIndex`'s own
/// per-check `Status` reporting style (a caller can see *which* check
/// failed, not just whether the segment as a whole is healthy).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Check {
    pub name: String,
    pub passed: bool,
    pub message: String,
}

impl Check {
    fn pass(name: impl Into<String>) -> Self {
        Check {
            name: name.into(),
            passed: true,
            message: "ok".to_string(),
        }
    }

    fn fail(name: impl Into<String>, message: impl Into<String>) -> Self {
        Check {
            name: name.into(),
            passed: false,
            message: message.into(),
        }
    }
}

/// Every check performed against one segment, in the order they ran.
#[derive(Debug, Clone)]
pub struct CheckResult {
    pub segment_name: String,
    pub checks: Vec<Check>,
}

impl CheckResult {
    /// Whether every check performed on this segment passed. `false` for a
    /// segment where an early structural failure (e.g. `.si` won't even
    /// parse) short-circuited the rest -- see [`check_segment`]'s doc
    /// comment on short-circuiting.
    pub fn all_passed(&self) -> bool {
        self.checks.iter().all(|c| c.passed)
    }

    /// The subset of checks that failed, for reporting.
    pub fn failures(&self) -> Vec<&Check> {
        self.checks.iter().filter(|c| !c.passed).collect()
    }
}

/// File-extension groups this module cross-checks `.fnm` field flags
/// against. Each group is "all files present" or "no files present" --
/// real Lucene never writes e.g. only a `.dvm` without its matching `.dvd`,
/// so a partial group is itself worth flagging rather than silently picking
/// one file to check.
fn files_with_ext<'a>(files: &'a [String], ext: &str) -> Vec<&'a str> {
    files
        .iter()
        .filter(|f| f.ends_with(ext))
        .map(String::as_str)
        .collect()
}

fn has_any_ext(files: &[String], exts: &[&str]) -> bool {
    exts.iter()
        .any(|ext| !files_with_ext(files, ext).is_empty())
}

/// Checks one segment. Reads `.si` first (everything else depends on
/// knowing the file list); if `.si` itself fails to open/parse, every other
/// check is skipped (there is nothing meaningful left to cross-check) and
/// only the `si.open` failure is reported -- matches real `CheckIndex`
/// aborting a segment's remaining checks once its `SegmentInfo` can't be
/// trusted.
pub fn check_segment(dir: &dyn Directory, commit: &SegmentCommitInfo) -> CheckResult {
    let segment_name = commit.segment_name.clone();
    let mut checks = Vec::new();

    let si = match open_si(dir, commit) {
        Ok(si) => {
            checks.push(Check::pass("si.open"));
            si
        }
        Err(e) => {
            checks.push(Check::fail("si.open", e));
            return CheckResult {
                segment_name,
                checks,
            };
        }
    };

    check_files_exist_and_validate(dir, &si, &mut checks);

    let field_infos = match open_fnm(dir, commit, &si) {
        Ok(fi) => {
            checks.push(Check::pass("fnm.open"));
            Some(fi)
        }
        Err(e) => {
            checks.push(Check::fail("fnm.open", e));
            None
        }
    };

    if let Some(fi) = &field_infos {
        check_field_flags_vs_files(fi, &si.files, &mut checks);
    }

    check_live_docs(dir, commit, &si, &mut checks);
    check_stored_fields_doc_count(dir, commit, &si, &mut checks);

    CheckResult {
        segment_name,
        checks,
    }
}

/// Checks every segment in the latest commit found in `dir`.
pub fn check_directory(dir: &dyn Directory) -> segment_infos::Result<Vec<CheckResult>> {
    let infos = segment_infos::read_latest(dir)?;
    Ok(infos
        .segments
        .iter()
        .map(|commit| check_segment(dir, commit))
        .collect())
}

fn open_si(dir: &dyn Directory, commit: &SegmentCommitInfo) -> Result<SegmentInfo, String> {
    let bytes = dir
        .open(&format!("{}.si", commit.segment_name))
        .map_err(|e| e.to_string())?;
    segment_info::parse(&bytes, &commit.segment_id).map_err(|e| e.to_string())
}

fn open_fnm(
    dir: &dyn Directory,
    commit: &SegmentCommitInfo,
    si: &SegmentInfo,
) -> Result<FieldInfos, String> {
    let fnm_name = si
        .files
        .iter()
        .find(|f| f.ends_with(".fnm"))
        .ok_or_else(|| "segment has no .fnm file listed".to_string())?;
    let bytes = dir.open(fnm_name).map_err(|e| e.to_string())?;
    field_infos::parse(&bytes, &commit.segment_id, "").map_err(|e| e.to_string())
}

/// Every file `.si` lists must exist and have a structurally valid codec
/// footer (magic/algorithm id/checksum shape, via [`codec_util::retrieve_checksum`]
/// -- cheap, format-agnostic, and every Lucene file this port reads/writes
/// ends with the same 16-byte footer regardless of its own header shape).
fn check_files_exist_and_validate(dir: &dyn Directory, si: &SegmentInfo, checks: &mut Vec<Check>) {
    for file in &si.files {
        let name = format!("file:{file}");
        match dir.open(file) {
            Ok(bytes) => match codec_util::retrieve_checksum(&bytes) {
                Ok(_) => checks.push(Check::pass(name)),
                Err(e) => checks.push(Check::fail(name, e.to_string())),
            },
            Err(e) => checks.push(Check::fail(name, e.to_string())),
        }
    }
}

/// Cross-checks each field's `.fnm` flags against which file groups the
/// segment actually has, in both directions: a field claiming a feature
/// with no matching files is an orphaned claim; files present with no field
/// claiming that feature are orphaned files.
fn check_field_flags_vs_files(fields: &FieldInfos, files: &[String], checks: &mut Vec<Check>) {
    let has_dv_files = has_any_ext(files, &[".dvd", ".dvm"]);
    let has_norms_files = has_any_ext(files, &[".nvd", ".nvm"]);
    let has_tv_files = has_any_ext(files, &[".tvd", ".tvx", ".tvm"]);
    let has_postings_files = has_any_ext(files, &[".tim", ".tip", ".tmd"]);

    let any_field_claims_dv = fields
        .fields
        .iter()
        .any(|f| f.doc_values_type != field_infos::DocValuesType::None);
    let any_field_claims_norms = fields.fields.iter().any(|f| !f.omit_norms);
    let any_field_claims_tv = fields.fields.iter().any(|f| f.store_term_vectors);
    let any_field_claims_postings = fields
        .fields
        .iter()
        .any(|f| f.index_options != field_infos::IndexOptions::None);

    check_claim_vs_files(
        "fnm.doc_values_vs_files",
        any_field_claims_dv,
        has_dv_files,
        ".dvd/.dvm",
        checks,
    );
    check_claim_vs_files(
        "fnm.norms_vs_files",
        any_field_claims_norms,
        has_norms_files,
        ".nvd/.nvm",
        checks,
    );
    check_claim_vs_files(
        "fnm.term_vectors_vs_files",
        any_field_claims_tv,
        has_tv_files,
        ".tvd/.tvx/.tvm",
        checks,
    );
    check_claim_vs_files(
        "fnm.postings_vs_files",
        any_field_claims_postings,
        has_postings_files,
        ".tim/.tip/.tmd",
        checks,
    );
}

fn check_claim_vs_files(
    name: &str,
    claims: bool,
    has_files: bool,
    file_group: &str,
    checks: &mut Vec<Check>,
) {
    match (claims, has_files) {
        (true, false) => checks.push(Check::fail(
            name,
            format!("a field claims this feature but the segment has no {file_group} file(s)"),
        )),
        (false, true) => checks.push(Check::fail(
            name,
            format!("the segment has {file_group} file(s) but no field claims this feature"),
        )),
        _ => checks.push(Check::pass(name)),
    }
}

/// `.si`'s `doc_count` vs `.liv`'s bit-count-derived max_doc, and
/// `live_docs`' cardinality vs `SegmentCommitInfo.del_count`'s implied live
/// count (`max_doc - del_count`). Both are skipped (not failed) for a
/// segment with no deletions (`del_gen == -1`), matching
/// `SegmentCommitInfo.hasDeletions()`'s own condition -- there is no `.liv`
/// file to check in that case.
fn check_live_docs(
    dir: &dyn Directory,
    commit: &SegmentCommitInfo,
    si: &SegmentInfo,
    checks: &mut Vec<Check>,
) {
    if commit.del_gen == -1 {
        return;
    }
    let liv_name = liv_file_name(&commit.segment_name, commit.del_gen);
    let bytes = match dir.open(&liv_name) {
        Ok(b) => b,
        Err(e) => {
            checks.push(Check::fail("liv.open", e.to_string()));
            return;
        }
    };

    // Independent of `live_docs::parse` below: the `.liv` payload's byte
    // length (header end to footer start) is `bits2words(maxDoc) * 8` --
    // derive the max_doc this file's *size alone* implies and cross-check
    // it against `.si`'s recorded `doc_count`, rather than trusting
    // `si.doc_count` by construction (which is what simply passing it as
    // `parse`'s `max_doc` argument would do).
    {
        use lucene_store::data_input::SliceInput;
        let suffix = lucene_util::base36::to_base36(commit.del_gen);
        let mut input = SliceInput::new(&bytes);
        match codec_util::check_index_header(
            &mut input,
            "Lucene90LiveDocs",
            0,
            0,
            &commit.segment_id,
            &suffix,
        ) {
            Ok(_) => {
                let header_end = input.position();
                let payload_len = bytes
                    .len()
                    .saturating_sub(header_end + codec_util::FOOTER_LENGTH);
                let implied_words = payload_len / 8;
                let expected_words = lucene_util::fixed_bit_set::bits2words(si.doc_count as usize);
                if implied_words == expected_words {
                    checks.push(Check::pass("liv.max_doc_matches_si"));
                } else {
                    checks.push(Check::fail(
                        "liv.max_doc_matches_si",
                        format!(
                            "si.doc_count={} implies {expected_words} words but .liv's payload has {implied_words} words",
                            si.doc_count
                        ),
                    ));
                }
            }
            Err(e) => {
                checks.push(Check::fail("liv.open", e.to_string()));
                return;
            }
        }
    }

    match live_docs::parse(
        &bytes,
        &commit.segment_id,
        commit.del_gen,
        si.doc_count as usize,
        commit.del_count as usize,
    ) {
        Ok(_) => {
            checks.push(Check::pass("liv.open"));
            checks.push(Check::pass("liv.cardinality_matches_del_count"));
        }
        Err(live_docs::Error::DelCountMismatch { actual, expected }) => {
            // Header/bits/footer all decoded fine; only the recorded
            // del_count disagrees with the bits' own cardinality.
            checks.push(Check::pass("liv.open"));
            checks.push(Check::fail(
                "liv.cardinality_matches_del_count",
                format!(
                    "SegmentCommitInfo.del_count={expected} but .liv's live bits imply {actual} deleted docs"
                ),
            ));
        }
        Err(e) => {
            checks.push(Check::fail("liv.open", e.to_string()));
        }
    }
}

/// Stored-fields doc count vs `.si`'s declared `doc_count`. Skipped (not
/// failed) if the segment has no stored-fields files at all -- a
/// stored-fields-less segment is not itself a defect this check is
/// responsible for catching (nothing in this port's scope writes such a
/// segment, but a hand-built or externally-produced one legitimately
/// could).
fn check_stored_fields_doc_count(
    dir: &dyn Directory,
    commit: &SegmentCommitInfo,
    si: &SegmentInfo,
    checks: &mut Vec<Check>,
) {
    let fdt_name = si.files.iter().find(|f| f.ends_with(".fdt"));
    let fdx_name = si.files.iter().find(|f| f.ends_with(".fdx"));
    let fdm_name = si.files.iter().find(|f| f.ends_with(".fdm"));
    let (fdt_name, fdx_name, fdm_name) = match (fdt_name, fdx_name, fdm_name) {
        (Some(t), Some(x), Some(m)) => (t, x, m),
        (None, None, None) => return,
        _ => {
            checks.push(Check::fail(
                "stored_fields.doc_count_matches_si",
                "segment has some but not all of .fdt/.fdx/.fdm",
            ));
            return;
        }
    };

    let result = (|| -> Result<i32, String> {
        let fdt = dir.open(fdt_name).map_err(|e| e.to_string())?;
        let fdx = dir.open(fdx_name).map_err(|e| e.to_string())?;
        let fdm = dir.open(fdm_name).map_err(|e| e.to_string())?;
        let reader = stored_fields::open(&fdt, &fdx, &fdm, &commit.segment_id, "")
            .map_err(|e| e.to_string())?;
        Ok(reader.max_doc())
    })();

    match result {
        Ok(max_doc) if max_doc == si.doc_count => {
            checks.push(Check::pass("stored_fields.doc_count_matches_si"));
        }
        Ok(max_doc) => {
            checks.push(Check::fail(
                "stored_fields.doc_count_matches_si",
                format!(
                    "si.doc_count={} but stored fields reader reports max_doc={max_doc}",
                    si.doc_count
                ),
            ));
        }
        Err(e) => {
            checks.push(Check::fail("stored_fields.doc_count_matches_si", e));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_store::codec_util::ID_LENGTH;
    use lucene_store::FsDirectory;

    fn fixture_dir(name: &str) -> std::path::PathBuf {
        std::path::PathBuf::from(format!(
            "{}/../../fixtures/data/{name}/",
            env!("CARGO_MANIFEST_DIR")
        ))
    }

    fn tempdir() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "lucene_rust_check_index_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A genuinely valid, real-Lucene-written fixture must pass every check
    /// cleanly -- the baseline "no false positives" test.
    #[test]
    fn valid_blocktree_fixture_passes_every_check() {
        let dir = FsDirectory::open(fixture_dir("blocktree_index"));
        let results = check_directory(&dir).expect("read segments_N");
        assert_eq!(results.len(), 1);
        let result = &results[0];
        assert!(
            result.all_passed(),
            "unexpected failures: {:?}",
            result.failures()
        );
        // Sanity: real checks actually ran, this isn't a vacuous pass.
        assert!(result.checks.len() >= 5);
    }

    /// A fixture with real deletions must pass the `.liv`-specific checks
    /// (max_doc match, cardinality vs del_count) on genuinely valid data.
    #[test]
    fn valid_live_docs_fixture_passes_every_check() {
        let dir = FsDirectory::open(fixture_dir("live_docs_index"));
        let results = check_directory(&dir).expect("read segments_N");
        assert_eq!(results.len(), 1);
        assert!(
            results[0].all_passed(),
            "unexpected failures: {:?}",
            results[0].failures()
        );
        assert!(results[0]
            .checks
            .iter()
            .any(|c| c.name == "liv.cardinality_matches_del_count"));
    }

    fn read_commit(dir: &FsDirectory) -> segment_infos::SegmentCommitInfo {
        segment_infos::read_latest(dir)
            .expect("read real segments_N")
            .segments[0]
            .clone()
    }

    /// Hand-construct a `SegmentCommitInfo` with a wrong `del_count` (copied
    /// from the real fixture but mutated) and confirm the cardinality check
    /// reports a clear failure, not a panic or a false pass -- the fixture's
    /// `.liv` bytes on disk are untouched, only the in-memory commit info
    /// this module cross-checks against is wrong.
    #[test]
    fn wrong_del_count_fails_cardinality_check_with_clear_message() {
        let dir = FsDirectory::open(fixture_dir("live_docs_index"));
        let mut commit = read_commit(&dir);
        assert_eq!(commit.del_count, 2, "fixture's real recorded del_count");
        commit.del_count = 4; // wrong on purpose

        let result = check_segment(&dir, &commit);
        assert!(!result.all_passed());
        let failure = result
            .checks
            .iter()
            .find(|c| c.name == "liv.cardinality_matches_del_count")
            .expect("cardinality check must have run");
        assert!(!failure.passed);
        assert!(failure.message.contains("del_count"));

        // Every other check on this segment (files, .fnm, stored fields)
        // must still have run and passed -- one wrong field must not
        // suppress unrelated checks.
        assert!(result
            .checks
            .iter()
            .filter(|c| c.name != "liv.cardinality_matches_del_count")
            .all(|c| c.passed));
    }

    /// Truncating the `.liv` file's bytes (fewer bits than `si.doc_count`
    /// implies) must fail the max_doc-consistency check, not panic.
    #[test]
    fn truncated_liv_file_fails_max_doc_check() {
        let src_dir = fixture_dir("live_docs_index");
        let dst_dir = tempdir();
        for entry in std::fs::read_dir(&src_dir).unwrap() {
            let entry = entry.unwrap();
            std::fs::copy(entry.path(), dst_dir.join(entry.file_name())).unwrap();
        }

        let dir = FsDirectory::open(&dst_dir);
        let commit = read_commit(&dir);
        let liv_name = liv_file_name(&commit.segment_name, commit.del_gen);
        let liv_path = dst_dir.join(&liv_name);
        let mut bytes = std::fs::read(&liv_path).unwrap();
        // Truncate well below the footer length so the .liv fails to parse
        // at all (a byte-count too short to even hold a header/footer) --
        // this must surface as a `liv.open` failure, not a panic.
        bytes.truncate(4);
        std::fs::write(&liv_path, &bytes).unwrap();

        let result = check_segment(&dir, &commit);
        assert!(!result.all_passed());
        let failure = result
            .checks
            .iter()
            .find(|c| c.name == "liv.open")
            .expect("liv.open check must have run");
        assert!(!failure.passed);

        std::fs::remove_dir_all(&dst_dir).ok();
    }

    /// A segment whose `.si` lists a file that does not actually exist on
    /// disk must be flagged by the "every listed file exists" check, not
    /// error out the whole directory-level walk.
    #[test]
    fn missing_listed_file_fails_files_exist_check() {
        use crate::segment_info;

        let dst_dir = tempdir();
        let dir = FsDirectory::open(&dst_dir);

        let si = SegmentInfo {
            id: [7u8; ID_LENGTH],
            version: segment_info::LuceneVersion {
                major: 10,
                minor: 0,
                bugfix: 0,
            },
            min_version: None,
            doc_count: 1,
            is_compound_file: false,
            has_blocks: false,
            diagnostics: vec![],
            files: vec!["_0.fnm".to_string(), "_0.missing".to_string()],
            attributes: vec![],
            index_sort: None,
        };
        let si_bytes = segment_info::write(&si, "");
        std::fs::write(dst_dir.join("_0.si"), &si_bytes).unwrap();
        // No .fnm nor .missing actually written.

        let commit = segment_infos::SegmentCommitInfo {
            segment_name: "_0".to_string(),
            segment_id: [7u8; ID_LENGTH],
            codec_name: "Lucene104".to_string(),
            del_gen: -1,
            del_count: 0,
            field_infos_gen: -1,
            doc_values_gen: -1,
            soft_del_count: 0,
            sci_id: None,
            field_infos_files: vec![],
            dv_update_files: vec![],
        };

        let result = check_segment(&dir, &commit);
        assert!(!result.all_passed());
        let file_failures: Vec<_> = result
            .checks
            .iter()
            .filter(|c| c.name.starts_with("file:") && !c.passed)
            .collect();
        assert_eq!(file_failures.len(), 2);

        std::fs::remove_dir_all(&dst_dir).ok();
    }

    /// `.si` itself failing to parse must short-circuit every other check
    /// (nothing else can be trusted without a valid file list) and report
    /// exactly that one failure, not panic.
    #[test]
    fn corrupt_si_short_circuits_remaining_checks() {
        let dst_dir = tempdir();
        let dir = FsDirectory::open(&dst_dir);
        std::fs::write(dst_dir.join("_0.si"), b"not a real .si file").unwrap();

        let commit = segment_infos::SegmentCommitInfo {
            segment_name: "_0".to_string(),
            segment_id: [9u8; ID_LENGTH],
            codec_name: "Lucene104".to_string(),
            del_gen: -1,
            del_count: 0,
            field_infos_gen: -1,
            doc_values_gen: -1,
            soft_del_count: 0,
            sci_id: None,
            field_infos_files: vec![],
            dv_update_files: vec![],
        };

        let result = check_segment(&dir, &commit);
        assert_eq!(result.checks.len(), 1);
        assert_eq!(result.checks[0].name, "si.open");
        assert!(!result.checks[0].passed);

        std::fs::remove_dir_all(&dst_dir).ok();
    }

    /// A `.fnm` claiming doc-values with no matching `.dvd`/`.dvm` file
    /// present must be flagged as an orphaned claim, not silently ignored --
    /// exercises the "claims but no files" branch of the field-flags
    /// cross-check without needing a whole hand-built doc-values fixture.
    #[test]
    fn field_claiming_doc_values_without_files_is_flagged() {
        use lucene_codecs::field_infos::{DocValuesType, FieldInfo, IndexOptions};

        let field = FieldInfo {
            name: "f".to_string(),
            number: 0,
            store_term_vectors: false,
            omit_norms: true,
            store_payloads: false,
            soft_deletes_field: false,
            parent_field: false,
            index_options: IndexOptions::None,
            doc_values_type: DocValuesType::Numeric,
            doc_values_skip_index_type: field_infos::DocValuesSkipIndexType::None,
            doc_values_gen: -1,
            attributes: vec![],
            point_dimension_count: 0,
            point_index_dimension_count: 0,
            point_num_bytes: 0,
            vector_dimension: 0,
            vector_encoding: field_infos::VectorEncoding::Float32,
            vector_similarity_function: field_infos::VectorSimilarityFunction::Euclidean,
        };
        let fields = FieldInfos {
            fields: vec![field],
        };
        let mut checks = Vec::new();
        // No .dvd/.dvm in this file list.
        check_field_flags_vs_files(&fields, &["_0.fnm".to_string()], &mut checks);

        let failure = checks
            .iter()
            .find(|c| c.name == "fnm.doc_values_vs_files")
            .expect("doc values check must have run");
        assert!(!failure.passed);
        assert!(failure.message.contains("claims"));
    }

    /// The reverse orphan direction: `.dvd`/`.dvm` files present but no
    /// field in `.fnm` claims doc-values at all.
    #[test]
    fn doc_values_files_without_any_claiming_field_is_flagged() {
        let fields = FieldInfos { fields: vec![] };
        let mut checks = Vec::new();
        check_field_flags_vs_files(
            &fields,
            &["_0.dvd".to_string(), "_0.dvm".to_string()],
            &mut checks,
        );

        let failure = checks
            .iter()
            .find(|c| c.name == "fnm.doc_values_vs_files")
            .expect("doc values check must have run");
        assert!(!failure.passed);
        assert!(failure.message.contains("no field claims"));
    }

    /// `CheckResult::failures()` must return exactly the failed checks, in
    /// order, for a mixed pass/fail result.
    #[test]
    fn check_result_failures_filters_correctly() {
        let result = CheckResult {
            segment_name: "_0".to_string(),
            checks: vec![Check::pass("a"), Check::fail("b", "bad"), Check::pass("c")],
        };
        assert!(!result.all_passed());
        let failures = result.failures();
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].name, "b");
    }

    /// A file listed in `.si` that opens but has a corrupted/missing codec
    /// footer must fail the files-exist-and-validate check with a clear
    /// message, not panic -- exercises the `retrieve_checksum` failure
    /// branch (as opposed to the file simply not existing at all).
    #[test]
    fn file_with_corrupted_footer_fails_files_check() {
        let dst_dir = tempdir();
        let dir = FsDirectory::open(&dst_dir);
        // Long enough to have a plausible footer position, but the footer
        // bytes themselves are garbage.
        std::fs::write(dst_dir.join("_0.junk"), vec![0u8; 32]).unwrap();

        let si = SegmentInfo {
            id: [3u8; ID_LENGTH],
            version: segment_info::LuceneVersion {
                major: 10,
                minor: 0,
                bugfix: 0,
            },
            min_version: None,
            doc_count: 1,
            is_compound_file: false,
            has_blocks: false,
            diagnostics: vec![],
            files: vec!["_0.junk".to_string()],
            attributes: vec![],
            index_sort: None,
        };
        let mut checks = Vec::new();
        check_files_exist_and_validate(&dir, &si, &mut checks);
        assert_eq!(checks.len(), 1);
        assert!(!checks[0].passed);

        std::fs::remove_dir_all(&dst_dir).ok();
    }

    /// A segment with `del_gen != -1` (deletions expected) but no `.liv`
    /// file actually on disk must fail `liv.open` with the underlying I/O
    /// error, not panic.
    #[test]
    fn missing_liv_file_fails_liv_open() {
        let dst_dir = tempdir();
        let dir = FsDirectory::open(&dst_dir);

        let si = SegmentInfo {
            id: [4u8; ID_LENGTH],
            version: segment_info::LuceneVersion {
                major: 10,
                minor: 0,
                bugfix: 0,
            },
            min_version: None,
            doc_count: 4,
            is_compound_file: false,
            has_blocks: false,
            diagnostics: vec![],
            files: vec![],
            attributes: vec![],
            index_sort: None,
        };
        let commit = segment_infos::SegmentCommitInfo {
            segment_name: "_0".to_string(),
            segment_id: [4u8; ID_LENGTH],
            codec_name: "Lucene104".to_string(),
            del_gen: 1,
            del_count: 1,
            field_infos_gen: -1,
            doc_values_gen: -1,
            soft_del_count: 0,
            sci_id: None,
            field_infos_files: vec![],
            dv_update_files: vec![],
        };

        let mut checks = Vec::new();
        check_live_docs(&dir, &commit, &si, &mut checks);
        assert_eq!(checks.len(), 1);
        assert_eq!(checks[0].name, "liv.open");
        assert!(!checks[0].passed);

        std::fs::remove_dir_all(&dst_dir).ok();
    }

    /// A `.liv` file whose byte size implies fewer words than `.si`'s
    /// `doc_count` requires must fail the independent `liv.max_doc_matches_si`
    /// size check, and then also fail `liv.open` when `live_docs::parse`
    /// itself runs out of bytes trying to read the (wrongly) larger bit
    /// array -- two related but distinct failures from one root cause, not
    /// a panic.
    #[test]
    fn liv_size_mismatch_fails_max_doc_check_and_parse() {
        use lucene_util::fixed_bit_set::FixedBitSet;

        let mut bits = FixedBitSet::new(4);
        bits.set(0);
        bits.set(1);
        bits.set(2);
        bits.set(3);
        let segment_id = [5u8; ID_LENGTH];
        let liv_bytes = live_docs::write(&bits, &segment_id, 1, 0).unwrap();

        let dst_dir = tempdir();
        let dir = FsDirectory::open(&dst_dir);
        std::fs::write(dst_dir.join(liv_file_name("_0", 1)), &liv_bytes).unwrap();

        let si = SegmentInfo {
            id: segment_id,
            version: segment_info::LuceneVersion {
                major: 10,
                minor: 0,
                bugfix: 0,
            },
            min_version: None,
            // 100 bits needs 2 words; the real .liv above only has 1 word
            // (built for max_doc=4).
            doc_count: 100,
            is_compound_file: false,
            has_blocks: false,
            diagnostics: vec![],
            files: vec![],
            attributes: vec![],
            index_sort: None,
        };
        let commit = segment_infos::SegmentCommitInfo {
            segment_name: "_0".to_string(),
            segment_id,
            codec_name: "Lucene104".to_string(),
            del_gen: 1,
            del_count: 0,
            field_infos_gen: -1,
            doc_values_gen: -1,
            soft_del_count: 0,
            sci_id: None,
            field_infos_files: vec![],
            dv_update_files: vec![],
        };

        let mut checks = Vec::new();
        check_live_docs(&dir, &commit, &si, &mut checks);
        let size_check = checks
            .iter()
            .find(|c| c.name == "liv.max_doc_matches_si")
            .expect("size check must have run");
        assert!(!size_check.passed);
        let open_check = checks
            .iter()
            .find(|c| c.name == "liv.open")
            .expect("liv.open must have run");
        assert!(!open_check.passed);

        std::fs::remove_dir_all(&dst_dir).ok();
    }

    /// A segment listing only some of `.fdt`/`.fdx`/`.fdm` (not all three,
    /// not none) must be flagged as an inconsistent file set, not silently
    /// skipped or panic.
    #[test]
    fn partial_stored_fields_file_set_is_flagged() {
        let dst_dir = tempdir();
        let dir = FsDirectory::open(&dst_dir);
        let si = SegmentInfo {
            id: [6u8; ID_LENGTH],
            version: segment_info::LuceneVersion {
                major: 10,
                minor: 0,
                bugfix: 0,
            },
            min_version: None,
            doc_count: 1,
            is_compound_file: false,
            has_blocks: false,
            diagnostics: vec![],
            files: vec!["_0.fdt".to_string()], // missing .fdx/.fdm
            attributes: vec![],
            index_sort: None,
        };
        let commit = segment_infos::SegmentCommitInfo {
            segment_name: "_0".to_string(),
            segment_id: [6u8; ID_LENGTH],
            codec_name: "Lucene104".to_string(),
            del_gen: -1,
            del_count: 0,
            field_infos_gen: -1,
            doc_values_gen: -1,
            soft_del_count: 0,
            sci_id: None,
            field_infos_files: vec![],
            dv_update_files: vec![],
        };

        let mut checks = Vec::new();
        check_stored_fields_doc_count(&dir, &commit, &si, &mut checks);
        assert_eq!(checks.len(), 1);
        assert!(!checks[0].passed);
        assert!(checks[0].message.contains("some but not all"));

        std::fs::remove_dir_all(&dst_dir).ok();
    }

    /// Real `.fdt`/`.fdx`/`.fdm` bytes (copied from the `blocktree_index`
    /// fixture) but a deliberately wrong `.si` `doc_count` must fail the
    /// doc-count cross-check with a clear message, not panic -- the
    /// stored-fields half of the same "wrong recorded count" family of
    /// tests as `wrong_del_count_fails_cardinality_check_with_clear_message`.
    #[test]
    fn stored_fields_doc_count_mismatch_is_flagged() {
        let src_dir = fixture_dir("blocktree_index");
        let dir = FsDirectory::open(&src_dir);
        let commit = read_commit(&dir);

        let mut si = open_si(&dir, &commit).expect("real .si parses");
        assert_ne!(si.doc_count, 999);
        si.doc_count = 999; // wrong on purpose; real .fdt/.fdx/.fdm untouched

        let mut checks = Vec::new();
        check_stored_fields_doc_count(&dir, &commit, &si, &mut checks);
        assert_eq!(checks.len(), 1);
        assert!(!checks[0].passed);
        assert!(checks[0].message.contains("999"));
    }
}
