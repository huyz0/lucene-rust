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
//! - Postings term-by-term re-derivation (revisited; previously deferred --
//!   see below): for every field with postings and every term in that
//!   field's dictionary, walks the term's *actual* postings via
//!   [`blocktree::BlockTreeFields`]/[`DocInput::read_postings`] (the same
//!   read-side API `lucene-search`'s `directory_reader.rs` uses for real
//!   queries) and independently recomputes `totalTermFreq` (sum of decoded
//!   per-doc freqs), cross-checking it against the `.tmd`/`.tim`-recorded
//!   [`lucene_codecs::postings::TermStats`] for that exact term -- a
//!   metadata/data consistency check, not a re-validation of already-checked
//!   block encoding: it would catch a dictionary claiming `totalTermFreq=50`
//!   for a term whose actual per-doc freqs only sum to 49. Each decoded doc
//!   ID is also checked for being in-range and strictly increasing (see
//!   "why not a plain docFreq recount" below for why this, not a `docFreq`
//!   recount, is `docFreq`'s meaningful proxy here).
//!
//! ## Revisited scope decision: postings re-derivation
//!
//! This check was **deliberately deferred** in task #57 (see the prior
//! revision of this doc comment / `PLAN.md`'s task #57 entry) with the
//! stated reason "requires walking per-format internals this port's
//! read-side decoders expose in different shapes per format -- genuinely a
//! separate, large task". Re-examined now: that blocker no longer holds for
//! postings specifically. `blocktree::FieldTerms::iter()` already yields
//! every `(term, TermStats)` pair in a field in one pass,
//! `blocktree::BlockTreeFields::iter_fields()` (added by this task) yields
//! every field's dictionary, and `DocInput::read_postings`/
//! `postings::singleton_postings` already fully materialize a term's
//! `(docID, freq)` pairs -- every piece this check needs was already built
//! and already exercised by `lucene-search`'s query path (task #45) before
//! this task started; only a one-line accessor (`iter_fields`) was missing.
//! Nothing about this check requires new decode logic, matching the
//! `differential-testing` skill's precedent that this module's checks are
//! self-consistency logic over already-differentially-verified decoders,
//! not new byte parsing.
//!
//! **Why not a plain `docFreq` recount**: investigating this port's decode
//! API (`DocInput::read_postings`/`postings::singleton_postings`) turned up
//! a structural fact worth being explicit about rather than silently
//! glossing over -- both are *parameterized by* the term dictionary's own
//! claimed `docFreq` (it drives how many full 256-doc blocks vs. how large a
//! tail block to decode), exactly like real Lucene's own
//! `PostingsEnum.reset`/`BlockDocsEnum` (`TermState.docFreq` plays the same
//! role there). That means `postings.docs.len()` is *always* exactly equal
//! to the claimed `docFreq` whenever decoding succeeds at all -- a plain
//! recount can never disagree, so it would be a vacuous, always-passing
//! check dressed up as real verification. What a genuinely wrong claimed
//! `docFreq` actually produces is the reader consuming a different number of
//! bytes than the writer intended and wandering into unrelated bytes (the
//! next term's data, or past the buffer) -- observable as a decoded doc ID
//! that is out of the segment's valid `0..doc_count` range or not strictly
//! increasing, which `postings.doc_ids_valid:<field>` checks directly, or as
//! an outright decode error (already surfaced via this function's
//! `postings.open` failure path). This is the same reason real `CheckIndex`
//! catches this class of bug the way it does, not a limitation invented for
//! this port.
//!
//! **Known, honest limitation carried over rather than papered over**: a
//! term with `docFreq == 1` stores no per-doc freq on disk at all --
//! `singleton_postings` reconstructs its one `(docID, freq)` pair from
//! `TermMetadata.singleton_doc_id` and the term dictionary's own recorded
//! `total_term_freq` (see `blocktree.rs`'s `postings()` and
//! `postings::singleton_postings`'s doc comment). Re-deriving stats for such
//! a term from "postings" therefore trivially reproduces the claimed
//! `total_term_freq` rather than independently verifying it -- this mirrors
//! real Lucene's own format (a singleton's freq genuinely isn't stored
//! independently anywhere), not a gap specific to this port.
//!
//! **The same vacuity also applies to any `IndexOptions::Docs` (freq-less)
//! field, not just singleton terms**: `blocktree.rs`'s meta parsing sets
//! `total_term_freq = doc_freq` for such a field (no independent
//! `total_term_freq` vlong is ever written for it -- see
//! `postings_writer.rs`'s `IndexOptions::Docs` branch), and the postings
//! decoder itself synthesizes freq `1` for every doc when the field has no
//! stored freqs (never reading it from the wire). So for a `Docs`-only
//! field with `docFreq > 1`, `postings.total_term_freq:<field>` compares
//! `doc_freq` against `doc_freq` -- always trivially true, the same class
//! of vacuity as the singleton case above, just for a different reason
//! (field-wide format choice vs. per-term encoding). `postings.doc_ids_valid`
//! remains meaningful for such fields regardless, since it only depends on
//! decoded doc IDs, not freqs.
//!
//! **Still out of scope** (unchanged from before, and for the reason
//! originally given): doc-values value-range sanity, points-tree structural
//! invariants, and vectors-graph structural invariants. Each of those checks
//! a genuinely different per-format internal shape (points-tree traversal,
//! HNSW graph traversal) that this task did not touch -- a separate task per
//! format, not a natural extension of postings re-derivation.

use crate::deletes::liv_file_name;
use crate::segment_info::{self, SegmentInfo};
use crate::segment_infos::{self, SegmentCommitInfo};
use lucene_codecs::blocktree;
use lucene_codecs::field_infos::{self, FieldInfos};
use lucene_codecs::live_docs;
use lucene_codecs::postings::DocInput;
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
    if let Some(fi) = &field_infos {
        check_postings_term_stats(dir, commit, &si, fi, &mut checks);
    }

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

/// For every field with postings and every term in that field, walks the
/// term's *actual* postings and independently recomputes `totalTermFreq`
/// (the sum of each doc's decoded freq), cross-checking it against the term
/// dictionary's own recorded [`lucene_codecs::postings::TermStats`] -- see
/// this module's doc comment ("Revisited scope decision: postings
/// re-derivation") for why this is now implemented rather than deferred,
/// and its known singleton-term limitation.
///
/// **`docFreq` is deliberately *not* cross-checked against a plain recount**
/// here -- see the doc comment's "why not a plain docFreq recount" note.
/// Instead, every decoded doc ID is checked for being in-range
/// (`0 <= id < si.doc_count`) and strictly increasing
/// (`postings.doc_ids_valid`): the observable symptom a wrong claimed
/// `docFreq` actually produces with this decode API (wandering into
/// unrelated bytes, not a clean short/over count).
///
/// Skipped (not failed) when: the segment is a compound (`.cfs`/`.cfe`)
/// segment (this module has no compound-file support anywhere, matching its
/// existing scope); the segment has none of `.tim`/`.tip`/`.tmd` (no
/// postings at all, nothing to check); or the segment has only some of
/// `.tim`/`.tip`/`.tmd` (already flagged by
/// [`check_field_flags_vs_files`]'s `fnm.postings_vs_files` orphan check --
/// this function does not duplicate that failure). A field with postings
/// but no `.doc` file present (needed for any term with `docFreq > 1`) is
/// reported as a single `postings.doc_open` failure rather than silently
/// skipped.
/// Builds one named `Check` from a field's collected list of problem
/// messages (empty -> pass, non-empty -> fail listing at most the first 5,
/// with a total count) -- shared by [`check_postings_term_stats`]'s two
/// per-field checks so the "how many terms, show a few" reporting shape
/// isn't duplicated.
fn named_field_check(name: &str, problems: &[String], num_terms: i64) -> Check {
    if problems.is_empty() {
        Check::pass(name)
    } else {
        let shown = problems.len().min(5);
        Check::fail(
            name,
            format!(
                "{} of {num_terms} terms affected; first {shown}: {}",
                problems.len(),
                problems[..shown].join("; ")
            ),
        )
    }
}

fn check_postings_term_stats(
    dir: &dyn Directory,
    commit: &SegmentCommitInfo,
    si: &SegmentInfo,
    field_infos: &FieldInfos,
    checks: &mut Vec<Check>,
) {
    if si.is_compound_file {
        return;
    }
    let tim_name = si.files.iter().find(|f| f.ends_with(".tim"));
    let tip_name = si.files.iter().find(|f| f.ends_with(".tip"));
    let tmd_name = si.files.iter().find(|f| f.ends_with(".tmd"));
    let (tim_name, tip_name, tmd_name) = match (tim_name, tip_name, tmd_name) {
        (Some(t), Some(p), Some(m)) => (t, p, m),
        (None, None, None) => return,
        _ => return,
    };

    let result = (|| -> Result<Vec<Check>, String> {
        let tim = dir.open(tim_name).map_err(|e| e.to_string())?;
        let tip = dir.open(tip_name).map_err(|e| e.to_string())?;
        let tmd = dir.open(tmd_name).map_err(|e| e.to_string())?;

        // The postings codec suffix is embedded in the sub-file's own name:
        // strip the `<segment_name>_` prefix (e.g. `_0_Lucene104_0.tim` ->
        // `Lucene104_0`) and the `.tim` extension -- same derivation
        // `lucene-search`'s `directory_reader.rs` (`SegmentReader::open`)
        // uses, duplicated here rather than shared since that logic lives in
        // a crate this module has no dependency on (see this module's own
        // top doc comment on why it doesn't build on `lucene-search`).
        let segment_suffix = tim_name
            .strip_prefix(&format!("{}_", commit.segment_name))
            .or_else(|| tim_name.strip_prefix('_'))
            .and_then(|s| s.strip_suffix(".tim"))
            .unwrap_or_default()
            .to_string();

        let fields = blocktree::open(
            &tim,
            &tip,
            &tmd,
            field_infos,
            &commit.segment_id,
            &segment_suffix,
            si.doc_count,
        )
        .map_err(|e| e.to_string())?;

        let doc_bytes = si
            .files
            .iter()
            .find(|f| f.ends_with(".doc"))
            .map(|name| dir.open(name).map_err(|e| e.to_string()))
            .transpose()?;
        let doc_in = doc_bytes
            .as_ref()
            .map(|bytes| DocInput::open(bytes, &commit.segment_id, &segment_suffix))
            .transpose()
            .map_err(|e| e.to_string())?;

        let mut field_checks = Vec::new();
        let mut any_needs_doc_file = false;
        for (field_name, field_terms) in fields.iter_fields() {
            let mut freq_mismatches: Vec<String> = Vec::new();
            let mut doc_id_problems: Vec<String> = Vec::new();
            let mut terms = field_terms.iter();
            while let Some((term, claimed)) = terms.next() {
                if claimed.doc_freq > 1 && doc_in.is_none() {
                    any_needs_doc_file = true;
                    continue;
                }
                let postings = field_terms
                    .postings(term, doc_in.as_ref())
                    .map_err(|e| e.to_string())?
                    .ok_or_else(|| {
                        format!(
                            "field {field_name:?}: term {term:?} enumerated by iter() but not \
                             found by postings() seek"
                        )
                    })?;

                // `totalTermFreq` re-derivation: genuinely independent data
                // (each doc's decoded freq comes straight off the wire, not
                // from anything this loop already assumed) -- exactly the
                // "dictionary metadata vs. actual postings" cross-check task
                // #57 originally deferred.
                let actual_total_term_freq: i64 = postings.freqs.iter().map(|&f| f as i64).sum();
                if actual_total_term_freq != claimed.total_term_freq {
                    freq_mismatches.push(format!(
                        "field {field_name:?} term {term:?}: dictionary claims \
                         totalTermFreq={}, but postings actually sum to {actual_total_term_freq}",
                        claimed.total_term_freq
                    ));
                }

                // `docFreq` proxy: `postings.docs.len()` always equals
                // `claimed.doc_freq` by construction whenever decode
                // succeeds at all (`read_postings`/`singleton_postings` are
                // parameterized by the claimed count, exactly like real
                // Lucene's own `PostingsEnum.reset`) -- a plain recount could
                // never disagree, so it would be a no-op check. What a wrong
                // claimed `docFreq` actually does on decode is make the
                // reader consume a different number of bytes than the
                // writer intended, wandering into unrelated bytes -- caught
                // here as an out-of-range or non-monotonically-increasing
                // decoded doc ID, not as a count mismatch.
                let mut prev_doc_id = -1i32;
                for &doc_id in &postings.docs {
                    if doc_id <= prev_doc_id || doc_id >= si.doc_count {
                        doc_id_problems.push(format!(
                            "field {field_name:?} term {term:?}: decoded doc ID {doc_id} is not \
                             in the valid strictly-increasing 0..{} range (previous was \
                             {prev_doc_id})",
                            si.doc_count
                        ));
                        break;
                    }
                    prev_doc_id = doc_id;
                }
            }
            field_checks.push(named_field_check(
                &format!("postings.total_term_freq:{field_name}"),
                &freq_mismatches,
                field_terms.num_terms,
            ));
            field_checks.push(named_field_check(
                &format!("postings.doc_ids_valid:{field_name}"),
                &doc_id_problems,
                field_terms.num_terms,
            ));
        }
        if any_needs_doc_file {
            field_checks.push(Check::fail(
                "postings.doc_open",
                "a term with docFreq > 1 needs the segment's .doc file, but none was found",
            ));
        }
        Ok(field_checks)
    })();

    match result {
        Ok(field_checks) => checks.extend(field_checks),
        Err(e) => checks.push(Check::fail("postings.open", e)),
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

    // -- postings term-by-term re-derivation (task: "revisit scope") --

    const POSTINGS_SEG_ID: [u8; ID_LENGTH] = [11u8; ID_LENGTH];
    const POSTINGS_SUFFIX: &str = "Lucene104_0";

    fn postings_field_info(index_options: field_infos::IndexOptions) -> field_infos::FieldInfo {
        field_infos::FieldInfo {
            name: "body".to_string(),
            number: 0,
            store_term_vectors: false,
            omit_norms: true,
            store_payloads: false,
            soft_deletes_field: false,
            parent_field: false,
            index_options,
            doc_values_type: field_infos::DocValuesType::None,
            doc_values_skip_index_type: field_infos::DocValuesSkipIndexType::None,
            doc_values_gen: -1,
            attributes: vec![],
            point_dimension_count: 0,
            point_index_dimension_count: 0,
            point_num_bytes: 0,
            vector_dimension: 0,
            vector_encoding: field_infos::VectorEncoding::Float32,
            vector_similarity_function: field_infos::VectorSimilarityFunction::Euclidean,
        }
    }

    /// Writes a minimal, self-contained, non-compound one-field segment
    /// (`.si`/`.fnm`/`.tim`/`.tip`/`.tmd`/`.doc`) into `dst_dir` from
    /// `postings_writer::write_single_field`'s output, and returns the
    /// `SegmentCommitInfo` to open it with. `doc_bytes_override` lets a test
    /// substitute a *different* `.doc` buffer than the one that naturally
    /// matches `terms` -- the mechanism the corruption test below uses to
    /// build a term dictionary that claims one `totalTermFreq` while the
    /// actual `.doc` bytes sum to a different one, without any raw byte
    /// surgery.
    fn write_postings_fixture(
        dst_dir: &std::path::Path,
        terms: &[lucene_codecs::postings_writer::TermPostings],
        field_doc_count: i32,
        max_doc: i32,
        doc_bytes_override: Option<&[u8]>,
    ) -> segment_infos::SegmentCommitInfo {
        use lucene_codecs::field_infos::IndexOptions;
        use lucene_codecs::postings_writer::{write_single_field, FieldPostingsInput};

        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqs,
            doc_count: field_doc_count,
            has_payloads: false,
            terms,
        };
        let output = write_single_field(&input, &POSTINGS_SEG_ID, POSTINGS_SUFFIX)
            .expect("hand-built postings must write cleanly");
        let doc_bytes = doc_bytes_override.unwrap_or(&output.doc);

        let fields = field_infos::write(
            &[postings_field_info(IndexOptions::DocsAndFreqs)],
            &POSTINGS_SEG_ID,
            "",
        );
        std::fs::write(dst_dir.join("_0.fnm"), &fields).unwrap();
        std::fs::write(
            dst_dir.join(format!("_0_{POSTINGS_SUFFIX}.tim")),
            &output.tim,
        )
        .unwrap();
        std::fs::write(
            dst_dir.join(format!("_0_{POSTINGS_SUFFIX}.tip")),
            &output.tip,
        )
        .unwrap();
        std::fs::write(
            dst_dir.join(format!("_0_{POSTINGS_SUFFIX}.tmd")),
            &output.tmd,
        )
        .unwrap();
        std::fs::write(dst_dir.join(format!("_0_{POSTINGS_SUFFIX}.doc")), doc_bytes).unwrap();

        let si = SegmentInfo {
            id: POSTINGS_SEG_ID,
            version: segment_info::LuceneVersion {
                major: 10,
                minor: 0,
                bugfix: 0,
            },
            min_version: None,
            doc_count: max_doc,
            is_compound_file: false,
            has_blocks: false,
            diagnostics: vec![],
            files: vec![
                "_0.fnm".to_string(),
                format!("_0_{POSTINGS_SUFFIX}.tim"),
                format!("_0_{POSTINGS_SUFFIX}.tip"),
                format!("_0_{POSTINGS_SUFFIX}.tmd"),
                format!("_0_{POSTINGS_SUFFIX}.doc"),
            ],
            attributes: vec![],
            index_sort: None,
        };
        std::fs::write(dst_dir.join("_0.si"), segment_info::write(&si, "")).unwrap();

        segment_infos::SegmentCommitInfo {
            segment_name: "_0".to_string(),
            segment_id: POSTINGS_SEG_ID,
            codec_name: "Lucene104".to_string(),
            del_gen: -1,
            del_count: 0,
            field_infos_gen: -1,
            doc_values_gen: -1,
            soft_del_count: 0,
            sci_id: None,
            field_infos_files: vec![],
            dv_update_files: vec![],
        }
    }

    /// The real `blocktree_index` fixture (genuine Java-written postings, a
    /// mix of singleton and multi-doc terms) must pass the new re-derivation
    /// checks cleanly -- the "no false positives on real data" baseline,
    /// same role `valid_blocktree_fixture_passes_every_check` plays for the
    /// rest of this module.
    #[test]
    fn valid_blocktree_fixture_passes_postings_re_derivation() {
        let dir = FsDirectory::open(fixture_dir("blocktree_index"));
        let results = check_directory(&dir).expect("read segments_N");
        let result = &results[0];
        assert!(
            result.all_passed(),
            "unexpected failures: {:?}",
            result.failures()
        );
        assert!(
            result
                .checks
                .iter()
                .any(|c| c.name.starts_with("postings.total_term_freq:") && c.passed),
            "expected a passing postings.total_term_freq:<field> check, got: {:?}",
            result.checks
        );
        assert!(
            result
                .checks
                .iter()
                .any(|c| c.name.starts_with("postings.doc_ids_valid:") && c.passed),
            "expected a passing postings.doc_ids_valid:<field> check, got: {:?}",
            result.checks
        );
    }

    /// A hand-built, genuinely self-consistent segment (real writer output,
    /// not raw-byte surgery) must pass the re-derivation checks -- proves
    /// the machinery works on this port's own writer output, not just the
    /// one real-Lucene fixture above.
    #[test]
    fn hand_built_consistent_postings_pass_re_derivation() {
        use lucene_codecs::postings_writer::TermPostings;

        let terms = vec![
            TermPostings {
                term: b"apple".to_vec(),
                docs: vec![(0, 2), (2, 1), (5, 3)],
                ..Default::default()
            },
            TermPostings {
                term: b"kiwi".to_vec(),
                docs: vec![(1, 1)], // singleton
                ..Default::default()
            },
        ];

        // Distinct docs across both terms: {0, 1, 2, 5} -> field doc_count 4;
        // max_doc must exceed the highest doc ID (5) -> 6.
        let dst_dir = tempdir();
        let commit = write_postings_fixture(&dst_dir, &terms, 4, 6, None);
        let dir = FsDirectory::open(&dst_dir);

        let result = check_segment(&dir, &commit);
        assert!(
            result.all_passed(),
            "unexpected failures: {:?}",
            result.failures()
        );

        std::fs::remove_dir_all(&dst_dir).ok();
    }

    /// The actual proof this check does something real: the term
    /// dictionary (`.tim`/`.tip`/`.tmd`) is built from `claimed_terms` (which
    /// says term `"apple"`'s `totalTermFreq` is 60), but the `.doc` bytes it
    /// points at are swapped for `actual_terms`' real postings (whose three
    /// per-doc freqs actually sum to 6) -- same doc IDs and doc count in
    /// both (so decoding itself succeeds cleanly; this is a metadata/data
    /// disagreement, not a corrupt/truncated file), yet
    /// `postings.total_term_freq:body` must fail and name the exact
    /// mismatch, while `postings.doc_ids_valid:body` (an unrelated
    /// dimension) must still pass -- proving the new check independently
    /// recomputes from the actual postings rather than trusting the
    /// dictionary's own claim.
    #[test]
    fn corrupted_total_term_freq_is_caught_by_re_derivation() {
        use lucene_codecs::field_infos::IndexOptions;
        use lucene_codecs::postings_writer::{
            write_single_field, FieldPostingsInput, TermPostings,
        };

        let actual_terms = vec![TermPostings {
            term: b"apple".to_vec(),
            docs: vec![(0, 2), (2, 1), (5, 3)], // real per-doc freqs, sum = 6
            ..Default::default()
        }];
        let claimed_terms = vec![TermPostings {
            term: b"apple".to_vec(),
            // Same doc IDs/doc count (docFreq stays consistent -- this test
            // isolates totalTermFreq disagreement), different per-doc freqs
            // so the dictionary's recorded totalTermFreq (60) disagrees with
            // what the swapped-in real `.doc` bytes below actually contain.
            docs: vec![(0, 20), (2, 10), (5, 30)],
            ..Default::default()
        }];

        // Distinct docs {0, 2, 5} -> field doc_count 3; max_doc must exceed
        // the highest doc ID (5) -> 6.
        let actual_output = write_single_field(
            &FieldPostingsInput {
                field_number: 0,
                index_options: IndexOptions::DocsAndFreqs,
                doc_count: 3,
                has_payloads: false,
                terms: &actual_terms,
            },
            &POSTINGS_SEG_ID,
            POSTINGS_SUFFIX,
        )
        .unwrap();
        assert!(!actual_output.doc.is_empty());

        let dst_dir = tempdir();
        // `write_postings_fixture` builds .tim/.tip/.tmd from
        // `claimed_terms` (dictionary says totalTermFreq=60) but the `.doc`
        // file on disk is overridden to `actual_output.doc` (real bytes
        // summing to 6) -- both used the same doc IDs/doc_count, so
        // `meta.doc_start_fp` still points at the right offset and decoding
        // succeeds; only the recorded stat disagrees with the real data.
        let commit =
            write_postings_fixture(&dst_dir, &claimed_terms, 3, 6, Some(&actual_output.doc));
        let dir = FsDirectory::open(&dst_dir);

        let result = check_segment(&dir, &commit);
        assert!(!result.all_passed());

        let freq_check = result
            .checks
            .iter()
            .find(|c| c.name == "postings.total_term_freq:body")
            .expect("total_term_freq check must have run");
        assert!(!freq_check.passed);
        assert!(freq_check.message.contains("totalTermFreq=60"));
        assert!(freq_check.message.contains("sum to 6"));

        // An unrelated dimension (doc ID validity) must still pass -- one
        // wrong stat must not suppress or corrupt an unrelated check.
        let doc_ids_check = result
            .checks
            .iter()
            .find(|c| c.name == "postings.doc_ids_valid:body")
            .expect("doc_ids_valid check must have run");
        assert!(doc_ids_check.passed);

        std::fs::remove_dir_all(&dst_dir).ok();
    }

    /// The `postings.doc_ids_valid` proxy's own actual proof: unlike
    /// `total_term_freq` above, this swaps in `.doc` bytes whose per-doc
    /// freqs still sum correctly (so `total_term_freq` passes) but whose
    /// decoded doc IDs include one at/past `si.doc_count` -- exactly the
    /// "wrong claimed docFreq made the reader wander into unrelated bytes"
    /// symptom the doc comment above describes as this check's real
    /// purpose. Without this test, the check added specifically to catch
    /// docFreq corruption had never actually been exercised on its failure
    /// path.
    #[test]
    fn corrupted_doc_id_is_caught_by_doc_ids_valid_check() {
        use lucene_codecs::field_infos::IndexOptions;
        use lucene_codecs::postings_writer::{
            write_single_field, FieldPostingsInput, TermPostings,
        };

        let claimed_terms = vec![TermPostings {
            term: b"apple".to_vec(),
            docs: vec![(0, 2), (2, 1), (5, 3)], // sum = 6, doc IDs all < max_doc (6)
            ..Default::default()
        }];
        let actual_terms = vec![TermPostings {
            term: b"apple".to_vec(),
            // Same per-doc freqs in the same order (sum still 6, so
            // total_term_freq must still agree) but the third doc ID is
            // 9, past this segment's max_doc of 6 -- doc_ids_valid must
            // catch it even though total_term_freq does not.
            docs: vec![(0, 2), (2, 1), (9, 3)],
            ..Default::default()
        }];

        let actual_output = write_single_field(
            &FieldPostingsInput {
                field_number: 0,
                index_options: IndexOptions::DocsAndFreqs,
                doc_count: 3,
                has_payloads: false,
                terms: &actual_terms,
            },
            &POSTINGS_SEG_ID,
            POSTINGS_SUFFIX,
        )
        .unwrap();

        let dst_dir = tempdir();
        let commit =
            write_postings_fixture(&dst_dir, &claimed_terms, 3, 6, Some(&actual_output.doc));
        let dir = FsDirectory::open(&dst_dir);

        let result = check_segment(&dir, &commit);
        assert!(!result.all_passed());

        let doc_ids_check = result
            .checks
            .iter()
            .find(|c| c.name == "postings.doc_ids_valid:body")
            .expect("doc_ids_valid check must have run");
        assert!(!doc_ids_check.passed);
        assert!(doc_ids_check.message.contains("doc ID 9"));

        // total_term_freq is an unrelated dimension here (both sides sum
        // to 6) -- must still pass, proving the two checks are
        // independent.
        let freq_check = result
            .checks
            .iter()
            .find(|c| c.name == "postings.total_term_freq:body")
            .expect("total_term_freq check must have run");
        assert!(freq_check.passed);

        std::fs::remove_dir_all(&dst_dir).ok();
    }

    /// A field claiming postings whose segment is missing the `.doc` file
    /// (needed for any term with `docFreq > 1`) must be flagged as
    /// `postings.doc_open`, not panic -- exercises the "term needs `.doc`
    /// bytes but none were found" branch distinctly from a plain I/O error.
    #[test]
    fn missing_doc_file_for_multi_doc_term_fails_doc_open_check() {
        use lucene_codecs::postings_writer::TermPostings;

        let terms = vec![TermPostings {
            term: b"apple".to_vec(),
            docs: vec![(0, 2), (2, 1), (5, 3)],
            ..Default::default()
        }];

        // Distinct docs {0, 2, 5} -> field doc_count 3; max_doc must exceed
        // the highest doc ID (5) -> 6.
        let dst_dir = tempdir();
        let commit = write_postings_fixture(&dst_dir, &terms, 3, 6, None);

        // Make the .doc file genuinely absent, not just unlisted: delete it
        // from disk *and* drop it from `.si`'s file list, then rewrite
        // `.si` -- so this is "the segment legitimately has no .doc file"
        // from this function's point of view, not an I/O error on an
        // expected file (that's a different, already-covered failure mode).
        let dir_ro = FsDirectory::open(&dst_dir);
        let mut si = open_si(&dir_ro, &commit).expect("hand-built .si parses");
        si.files.retain(|f| !f.ends_with(".doc"));
        std::fs::write(dst_dir.join("_0.si"), segment_info::write(&si, "")).unwrap();
        std::fs::remove_file(dst_dir.join(format!("_0_{POSTINGS_SUFFIX}.doc"))).unwrap();

        let dir = FsDirectory::open(&dst_dir);
        let result = check_segment(&dir, &commit);
        let doc_open_check = result
            .checks
            .iter()
            .find(|c| c.name == "postings.doc_open")
            .expect("postings.doc_open check must have run");
        assert!(!doc_open_check.passed);

        std::fs::remove_dir_all(&dst_dir).ok();
    }
}
