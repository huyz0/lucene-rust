//! A `CheckIndex` equivalent: a standalone verifier that opens a commit's
//! segments and cross-checks relationships a normal open never verifies.
//!
//! It is not built on `lucene-search`'s `DirectoryReader`/`SegmentReader`:
//! those expose only what a query needs and hide what a self-check must
//! compare (`SegmentInfo.files`, per-field flags, raw `.si`/`.fnm`/stored
//! fields bytes). It reuses [`crate::segment_infos::read_latest`] to find the
//! commit and opens each segment's files through the same decoders the
//! reader calls (`segment_info::parse`, `field_infos::parse`,
//! `live_docs::parse`, `stored_fields::open`).
//!
//! # Checks
//!
//! Per file and per commit:
//!
//! - Every file `SegmentInfo.files` lists opens, and its CRC-32 is recomputed
//!   over every byte (`checksumEntireFile`), not just the footer's shape.
//! - The `.si` lists itself (`si.files_lists_itself`), and the commit-level
//!   files (`.liv`, generational field-infos and doc-values-update files) that
//!   live on `SegmentCommitInfo` are checked too.
//! - `SegmentInfos.readCommit`'s cross-`.si` validations (`commit.*`):
//!   `delCount`/`softDelCount` within `maxDoc`, `delCount == 0` without a
//!   delete generation, segment versions against `minSegmentLuceneVersion`
//!   and `indexCreatedVersionMajor`, `minVersion` recorded once
//!   `indexCreatedVersionMajor >= 7`, total docs within
//!   `IndexWriter.MAX_DOCS`, unique segment names, and `SegmentInfos.counter`
//!   ahead of every name (`validCounter`).
//! - `.si` doc count against `.liv`'s length, and `live_docs` cardinality
//!   against `del_count`.
//! - `.fnm`'s per-field flags (doc values, norms, term vectors, postings)
//!   against the files the segment lists, in both directions.
//! - Compound (`.cfs`) segments are read through
//!   [`crate::compound_reader::CompoundReader`] like any other.
//!
//! Per format (Java's `test*` methods):
//!
//! - **Stored fields** (`stored_fields.*`): doc count against `.si`, and
//!   every document decoded, deleted ones included.
//! - **Term vectors** (`term_vectors.*`): doc count, every document's vectors
//!   decoded, fields against `.fnm`'s `storeTermVectors`, and each vector
//!   against the inverted index (`.self_consistent`, `.match_postings`).
//! - **Doc values** (`doc_values.*`): every value decoded, SORTED/SORTED_SET
//!   ordinals bounds-checked and SORTED_SET ordinals strictly increasing
//!   within a doc, BINARY lengths within `.dvm`'s min/max, and the
//!   docs-with-a-value count against `numDocsWithField`.
//! - **Norms** (`norms.*`): every value read, the docs-with-a-value count
//!   against `.nvm`, and a live doc's norm non-zero exactly when it has terms
//!   in the field's postings.
//! - **Postings** (`postings.*`): for every term, the postings are walked and
//!   `totalTermFreq` recomputed against the dictionary's
//!   [`lucene_codecs::blocktree::TermStats`]; doc ids in range and strictly
//!   increasing; terms sorted, `docFreq > 0`, and `.tmd`'s field summary
//!   re-derived; positions, offsets, seek, advance and intersect agreement;
//!   `checkImpacts` and `checkDocIDRuns` over the lazy cursor. See
//!   [`check_postings`] for the full table and the Java checks that are
//!   vacuous in this representation.
//! - **Points** (`points.*`): every point within the field's `.kdm` bounds,
//!   each multi-dimension leaf's bounding box within the field's, and the
//!   leaves' point counts summing to `point_count`.
//! - **Vectors** (`vectors.*`) and HNSW graphs (`hnsw.*`).
//! - **Index sort** (`sort.docs_in_index_sort_order`), using the
//!   `segment_info::SortKeyComparator` the writer sorts with.
//! - **Soft deletes** (`soft_deletes.count_matches`): live docs with a value
//!   for the soft-deletes field equal `softDelCount`.
//!
//! # Notes
//!
//! **Why not a plain `docFreq` recount.** The postings decoder is driven by
//! the dictionary's own `docFreq` (it decides how many full blocks and how
//! large a tail block to read), as Lucene's `BlockDocsEnum` is, so a decoded
//! doc count always equals the claimed one. A wrong `docFreq` shows instead
//! as the reader wandering into unrelated bytes: a doc id out of range or
//! not increasing (`postings.doc_ids_valid`), or a decode error
//! (`postings.open`).
//!
//! **Where `totalTermFreq` cannot be checked.** A `docFreq == 1` term stores
//! no freq; its one posting is rebuilt from the dictionary's
//! `total_term_freq`. A freq-less (`IndexOptions::Docs`) field has
//! `total_term_freq = doc_freq` and decodes freq `1` everywhere. In both
//! cases the comparison is trivially true; `postings.doc_ids_valid` still
//! applies.
//!
//! **Single-dimension points leaves** carry no bounding box on disk, so
//! `points.leaf_bounds_subset_of_field` runs only for multi-dimension fields.
//!
//! **The `hnsw.*` checks go beyond 10.5.0.** Its `CheckIndex` has no graph
//! check (`testHnswGraphs` came later). They are diagnostic-only and catch
//! real graph corruption (see [`check_hnsw_graphs`]); they do not describe
//! 10.5.0's behaviour.
// This module is *audited* against the arithmetic gate: no `+`/`-`/`*`/`<<`
// in it may panic, because a verifier that panics on a corrupt file has
// failed at its one job (and through the FFI a panic in a debug build takes
// the JVM with it). Every operation here is `checked_*` with a reported
// corruption, `saturating_*` where the saturation is unreachable or is itself
// the honest answer for a counter, or a plain operator under an `#[allow]`
// carrying an `// ARITH:` proof. See `docs/arithmetic-gate.md`.
#![deny(clippy::arithmetic_side_effects)]

use crate::compound_reader::CompoundReader;
use crate::deletes::liv_file_name;
use crate::segment_info::{self, SegmentInfo};
use crate::segment_infos::{self, SegmentCommitInfo, SegmentInfos};
use lucene_codecs::backward_codecs::hnsw_vectors::{RetiredHnswFormat, RetiredHnswVectorsReader};
use lucene_codecs::backward_codecs::quantized_vectors::{
    QuantizedFiles, QuantizedFormat, QuantizedReader, QuantizedVectorsReader,
};
use lucene_codecs::blocktree;
use lucene_codecs::doc_values;
use lucene_codecs::field_infos::{self, FieldInfos};
use lucene_codecs::hnsw::HnswGraphView;
use lucene_codecs::hnsw_vectors;
use lucene_codecs::live_docs;
use lucene_codecs::norms;
use lucene_codecs::per_field_postings;
use lucene_codecs::points;
use lucene_codecs::postings::{self, DocInput};
use lucene_codecs::stored_fields;
use lucene_codecs::term_vectors;
use lucene_codecs::vectors;
use lucene_store::codec_util;
use lucene_store::directory::Directory;

/// `IndexWriter.MAX_DOCS`/`getActualMaxDocs()`: the hard ceiling on the total
/// number of documents across a commit (`Integer.MAX_VALUE - 128`).
/// `SegmentInfos.readCommit` enforces it as LUCENE-6299's "check we are in
/// bounds"; this port's `segment_infos::parse` cannot (it never opens the
/// `.si` files it would need `maxDoc` from), so the check lands here instead.
const MAX_DOCS: i64 = i32::MAX as i64 - 128;

/// One named check's outcome -- matches real `CheckIndex`'s own
/// per-check `Status` reporting style (a caller can see *which* check
/// failed, not just whether the segment as a whole is healthy).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Check {
    pub name: String,
    pub outcome: Outcome,
    pub message: String,
}

/// What a named check actually did. **Three outcomes, not two.**
///
/// A verifier's most dangerous failure is not a missing check; it is a check
/// that *could not run* being indistinguishable, from the outside, from a
/// check that passed. Almost every family here depends on an `*.open` step,
/// and when that step fails a family that contributed **nothing** to the
/// result would make a segment whose `.fnm` would not parse report a failed
/// `fnm.open` and then, silently, no postings checks at all -- or, worse,
/// report *passes* because the problem lists were empty for the simple reason
/// that nothing had looked.
///
/// [`Outcome::Skipped`] is that state made visible: the check is named, the
/// prerequisite that took it down is named, and [`CheckResult::all_passed`]
/// is `false`. It is emitted **only** when a prerequisite actually failed --
/// a format the segment legitimately does not have (no vectors, no points)
/// produces no check at all, because there is nothing there to be unguarded.
///
/// For example, a `.fnm` that does not parse must not make every postings
/// check in the segment vanish silently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Passed,
    Failed,
    /// Not run, because something it needed failed first. Not a pass.
    Skipped,
}

impl Check {
    fn pass(name: impl Into<String>) -> Self {
        Check {
            name: name.into(),
            outcome: Outcome::Passed,
            message: "ok".to_string(),
        }
    }

    fn fail(name: impl Into<String>, message: impl Into<String>) -> Self {
        Check {
            name: name.into(),
            outcome: Outcome::Failed,
            message: message.into(),
        }
    }

    /// A check that did not run because `because` (the name of the
    /// prerequisite check that failed) took it down with it.
    fn skipped(name: impl Into<String>, because: &str) -> Self {
        Check {
            name: name.into(),
            outcome: Outcome::Skipped,
            message: format!("not run: {because} failed"),
        }
    }

    /// True only for [`Outcome::Passed`]. A skipped check is not a pass.
    pub fn passed(&self) -> bool {
        matches!(self.outcome, Outcome::Passed)
    }

    /// True when the check could not run because a prerequisite failed.
    pub fn was_skipped(&self) -> bool {
        matches!(self.outcome, Outcome::Skipped)
    }
}

/// Records that `families` did not run because `because` failed.
///
/// One entry per *family* rather than per field: the caller needs to know
/// which classes of invariant are unguarded, and the per-field expansion is
/// unknowable anyway once the file that names the fields would not open.
fn skip_families(checks: &mut Vec<Check>, families: &[&str], because: &str) {
    for family in families {
        checks.push(Check::skipped(*family, because));
    }
}

/// Every check family [`check_segment_in_commit`] can produce that depends on
/// the `.si` parsing, in the order it would have run them. Used to report
/// what a failed `si.open` takes down with it.
const FAMILIES_BELOW_SI: &[&str] = &[
    "file:*",
    "si.files_lists_itself",
    "fnm.open",
    "fnm.*_vs_files",
    "commit.del_count_within_max_doc",
    "liv.*",
    "stored_fields.*",
    "postings.*",
    "term_vectors.*",
    "norms.*",
    "doc_values.*",
    "sort.docs_in_index_sort_order",
    "soft_deletes.count_matches",
    "points.*",
    "vectors.*",
    "hnsw.*",
];

/// Every family that needs `.fnm` to have parsed. `term_vectors.*` is not
/// here because its own reader still opens and its `doc_count_matches_si`
/// and `every_doc_decodes` checks still run; the three families that go
/// silent are skipped at their own site instead.
const FAMILIES_BELOW_FNM: &[&str] = &[
    "fnm.*_vs_files",
    "postings.*",
    "norms.*",
    "doc_values.*",
    "sort.docs_in_index_sort_order",
    "soft_deletes.count_matches",
    "points.*",
    "vectors.*",
    "hnsw.*",
];

/// Every family that needs the postings files to have opened. The last two
/// are the ones that make this worth naming: they are *cross-checks*, and a
/// cross-check with one side missing reports a clean pass over a comparison
/// nobody made.
const FAMILIES_BELOW_POSTINGS: &[&str] = &[
    "postings.*",
    "term_vectors.match_postings",
    "norms.agree_with_postings",
];

/// The counts real `CheckIndex` prints alongside each `OK` line, kept so a
/// caller (and this module's tests) can compare them against Java's own
/// output on the same index. Java's `test: terms, freq, prox...OK [N terms;
/// M terms/docs pairs; K tokens]` is
/// [`Self::term_count`]/[`Self::term_doc_pairs`]/[`Self::token_count`];
/// `test: field norms.........OK [N fields]` is [`Self::norm_fields`];
/// `test: term vectors........OK [N total term vector count]` is
/// [`Self::term_vector_fields`]; `test: vectors.............OK [N fields, M
/// vectors]` is [`Self::vector_fields`]/[`Self::vector_values`].
///
/// A count that disagrees with Java's is itself a finding: it means one side
/// enumerated something the other did not.
#[derive(Debug, Clone, Default)]
pub struct CheckStats {
    /// `Status.TermIndexStatus.termCount`: terms with at least one
    /// non-deleted document.
    pub term_count: i64,
    /// `Status.TermIndexStatus.delTermCount`: terms all of whose documents
    /// are deleted.
    pub del_term_count: i64,
    /// `Status.TermIndexStatus.totFreq`: (term, live doc) pairs.
    pub term_doc_pairs: i64,
    /// `Status.TermIndexStatus.totPos`: the sum of every (term, live doc)
    /// pair's frequency -- Java prints it as "tokens".
    pub token_count: i64,
    /// `Status.FieldNormStatus.totFields`.
    pub norm_fields: i64,
    /// `Status.TermVectorStatus.totVectors`: (live doc, field) pairs with a
    /// term vector.
    pub term_vector_fields: i64,
    /// `Status.VectorValuesStatus.totalKnnVectorFields`.
    pub vector_fields: i64,
    /// `Status.VectorValuesStatus.totalVectorValues`.
    pub vector_values: i64,
}

/// Every check performed against one segment, in the order they ran.
#[derive(Debug, Clone)]
pub struct CheckResult {
    pub segment_name: String,
    /// The segment's `.si`-declared `doc_count`, or `None` when the `.si`
    /// could not be opened (and for the commit-level result, which is not a
    /// segment). [`check_directory`] sums these for the total-doc bound.
    pub max_doc: Option<i32>,
    pub checks: Vec<Check>,
    /// The counts real `CheckIndex` prints; empty for the commit-level
    /// result and for a segment whose `.si` would not parse.
    pub stats: CheckStats,
}

impl CheckResult {
    /// Whether every check performed on this segment **passed** -- which a
    /// check that could not run did not. A [`Outcome::Skipped`] entry makes
    /// this `false`, deliberately: the whole point of modelling the third
    /// outcome is that "nothing looked at this" must never read as "this is
    /// fine".
    pub fn all_passed(&self) -> bool {
        self.checks.iter().all(|c| c.passed())
    }

    /// Every check that is not a pass -- failures **and** the families that
    /// could not run. For reporting.
    pub fn failures(&self) -> Vec<&Check> {
        self.checks.iter().filter(|c| !c.passed()).collect()
    }

    /// Just the families that could not run because a prerequisite failed,
    /// and what each one leaves unguarded.
    pub fn skipped(&self) -> Vec<&Check> {
        self.checks.iter().filter(|c| c.was_skipped()).collect()
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
    check_segment_in_commit(dir, None, commit)
}

/// [`check_segment`] plus the checks that need the enclosing commit's own
/// header fields (`SegmentInfos.readCommit`'s cross-`.si` validations, which
/// `segment_infos::parse` cannot perform because it never opens a `.si`).
/// Pass `None` for `infos` to run only the segment-local checks.
pub fn check_segment_in_commit(
    dir: &dyn Directory,
    infos: Option<&SegmentInfos>,
    commit: &SegmentCommitInfo,
) -> CheckResult {
    check_segment_with_si(dir, infos, commit, read_si(dir, commit))
}

/// [`check_segment_in_commit`] over the outcome of reading the segment's
/// `.si`. A failure is reported as `CorruptSegmentInfoException`'s message,
/// with the root cause (which names the file) appended.
fn check_segment_with_si(
    dir: &dyn Directory,
    infos: Option<&SegmentInfos>,
    commit: &SegmentCommitInfo,
    si: segment_info::Result<SegmentInfo>,
) -> CheckResult {
    let segment_name = commit.segment_name.clone();
    let mut checks = Vec::new();

    let si = match si {
        Ok(si) => {
            checks.push(Check::pass("si.open"));
            si
        }
        Err(e) => {
            checks.push(Check::fail(
                "si.open",
                format!(
                    "segment info file: {segment_name}.si cannot be read - it may be missing or \
                     corrupt: {e}"
                ),
            ));
            // Everything below needs the `.si`'s file list. Naming what did
            // not run is the whole point: without it the result is one
            // failure and sixteen families of silence that read, to any
            // caller counting failures, exactly like sixteen families of
            // agreement.
            skip_families(&mut checks, FAMILIES_BELOW_SI, "si.open");
            return CheckResult {
                segment_name,
                max_doc: None,
                checks,
                stats: CheckStats::default(),
            };
        }
    };

    check_files_exist_and_validate(dir, commit, &si, &mut checks);

    // `SegmentCoreReaders`: a compound segment's codec files live inside its
    // `.cfs`, and its `.si` lists only `.cfs`/`.cfe`/`.si` -- every segment
    // real Lucene flushes by default. From here on the checks read the
    // members through the archive, and see the member names in `si.files`,
    // exactly as they would a loose segment's files.
    let compound = if si.is_compound_file {
        match CompoundReader::open(dir, &commit.segment_name, &commit.segment_id) {
            Ok(reader) => {
                checks.push(Check::pass("cfs.open"));
                Some(reader)
            }
            Err(e) => {
                checks.push(Check::fail("cfs.open", e.to_string()));
                skip_families(&mut checks, FAMILIES_BELOW_SI, "cfs.open");
                return CheckResult {
                    segment_name,
                    max_doc: None,
                    checks,
                    stats: CheckStats::default(),
                };
            }
        }
    } else {
        None
    };
    let (dir, si): (&dyn Directory, SegmentInfo) = match &compound {
        Some(reader) => {
            let files = si
                .files
                .iter()
                .filter(|f| !f.ends_with(".cfs") && !f.ends_with(".cfe"))
                .cloned()
                .chain(reader.member_files())
                .collect();
            // Seen through the archive, the segment *is* a loose one: its
            // members are its files.
            (
                reader,
                SegmentInfo {
                    files,
                    is_compound_file: false,
                    ..si
                },
            )
        }
        None => (dir, si),
    };

    let field_infos = match open_fnm(dir, commit, &si) {
        Ok(fi) => {
            checks.push(Check::pass("fnm.open"));
            Some(fi)
        }
        Err(e) => {
            checks.push(Check::fail("fnm.open", e));
            skip_families(&mut checks, FAMILIES_BELOW_FNM, "fnm.open");
            None
        }
    };

    if let Some(fi) = &field_infos {
        check_field_flags_vs_files(fi, &si.files, &mut checks);
    }

    check_deletion_counts(commit, &si, &mut checks);
    if let Some(infos) = infos {
        check_segment_vs_commit_header(infos, commit, &si, &mut checks);
    }
    let live_docs = check_live_docs(dir, commit, &si, &mut checks);
    check_stored_fields_doc_count(dir, commit, &si, &mut checks);

    // The postings files are opened once and shared by three consumers:
    // `check_postings` itself, `testFieldNorms`' terms-vs-norms cross-check
    // (which needs the per-field set of docs that have terms), and
    // `testTermVectors`' vectors-vs-inverted-index cross-check. Real
    // `CheckIndex` gets the same sharing from one `CodecReader`.
    let postings_bytes = field_infos
        .as_ref()
        .as_ref()
        .and_then(|fi| open_postings_bytes(dir, &si, &commit.segment_name, Some(fi)));
    let postings = match (&postings_bytes, &field_infos) {
        (Some(Ok(bytes)), Some(fi)) => match bytes.handles(commit, fi, &si) {
            Ok(h) => Some(h),
            Err(e) => {
                checks.push(Check::fail("postings.open", e));
                skip_families(&mut checks, FAMILIES_BELOW_POSTINGS, "postings.open");
                None
            }
        },
        (Some(Err(e)), _) => {
            checks.push(Check::fail("postings.open", e.clone()));
            skip_families(&mut checks, FAMILIES_BELOW_POSTINGS, "postings.open");
            None
        }
        // No `.tim`/`.tip`/`.tmd` at all: a legitimate
        // absence, already reported by `fnm.postings_vs_files` if a field
        // claims postings. Nothing is left unguarded, so nothing is skipped.
        _ => None,
    };

    let mut stats = CheckStats::default();
    // `checkFields`' `visitedDocs`, per postings field: internal plumbing
    // from the postings walk to the terms-vs-norms cross-check, so it stays
    // local rather than riding along on every `CheckResult` for the life of
    // the run (one `FixedBitSet` per field per segment).
    let mut docs_with_terms = DocsWithTerms::new();
    if let (Some(fi), Some(postings)) = (&field_infos, &postings) {
        let (s, visited) = check_postings(&si, fi, live_docs.as_ref(), postings, &mut checks);
        stats = s;
        docs_with_terms = visited;
    }
    check_term_vectors(
        dir,
        commit,
        &si,
        field_infos.as_ref(),
        postings.as_ref(),
        live_docs.as_ref(),
        &mut stats,
        &mut checks,
    );
    if let Some(fi) = &field_infos {
        check_field_norms(
            dir,
            commit,
            &si,
            fi,
            live_docs.as_ref(),
            &docs_with_terms,
            &mut stats,
            &mut checks,
        );
        check_doc_values(dir, commit, &si, fi, &mut checks);
        check_index_sort(dir, commit, &si, fi, &mut checks);
        check_soft_deletes(dir, commit, &si, fi, &mut checks);
        check_points_structural_invariants(dir, commit, &si, fi, &mut checks);
        check_vectors(dir, commit, &si, fi, &mut stats, &mut checks);
    }

    CheckResult {
        segment_name,
        max_doc: Some(si.doc_count),
        checks,
        stats,
    }
}

/// `SegmentInfos.readCommit`'s per-segment deletion-count validation:
/// `delCount` and `softDelCount` must each be within `[0, maxDoc]` **and**
/// must not sum past `maxDoc`. Java performs these while parsing
/// `segments_N`, where it has already opened the `.si` and therefore knows
/// `maxDoc`; this port's `segment_infos::parse` deliberately does not open
/// `.si` files, so without this check nothing anywhere validated a
/// `delCount` against the segment it belongs to -- a commit claiming 500
/// deletions in a 3-document segment parsed clean.
fn check_deletion_counts(commit: &SegmentCommitInfo, si: &SegmentInfo, checks: &mut Vec<Check>) {
    let max_doc = si.doc_count;
    if commit.del_count > max_doc {
        checks.push(Check::fail(
            "commit.del_count_within_max_doc",
            format!(
                "invalid deletion count: {} vs maxDoc={max_doc}",
                commit.del_count
            ),
        ));
    } else {
        checks.push(Check::pass("commit.del_count_within_max_doc"));
    }
    if commit.soft_del_count > max_doc {
        checks.push(Check::fail(
            "commit.soft_del_count_within_max_doc",
            format!(
                "invalid deletion count: {} vs maxDoc={max_doc}",
                commit.soft_del_count
            ),
        ));
    } else {
        checks.push(Check::pass("commit.soft_del_count_within_max_doc"));
    }
    // Both are `i32` off `segments_N`; widening first makes the sum exact
    // for every pair of inputs, so the saturation can never be reached.
    let total = i64::from(commit.del_count).saturating_add(i64::from(commit.soft_del_count));
    if total > max_doc as i64 {
        checks.push(Check::fail(
            "commit.del_plus_soft_del_within_max_doc",
            format!("invalid deletion count: {total} vs maxDoc={max_doc}"),
        ));
    } else {
        checks.push(Check::pass("commit.del_plus_soft_del_within_max_doc"));
    }
    // `SegmentCommitInfo.hasDeletions()` is `delGen != -1`; a commit with no
    // delete generation cannot have deleted anything.
    if commit.del_gen == -1 && commit.del_count != 0 {
        checks.push(Check::fail(
            "commit.del_count_zero_without_del_gen",
            format!(
                "segment has delGen=-1 (no .liv file) but records del_count={}",
                commit.del_count
            ),
        ));
    } else {
        checks.push(Check::pass("commit.del_count_zero_without_del_gen"));
    }
}

/// The three `SegmentInfos.readCommit` checks that compare a segment's `.si`
/// header against the commit's own header: the segment cannot predate the
/// commit's recorded `minSegmentLuceneVersion`, cannot predate
/// `indexCreatedVersionMajor`, and must record a `minVersion` at all once
/// `indexCreatedVersionMajor >= 7`.
fn check_segment_vs_commit_header(
    infos: &SegmentInfos,
    commit: &SegmentCommitInfo,
    si: &SegmentInfo,
    checks: &mut Vec<Check>,
) {
    let seg = (si.version.major, si.version.minor, si.version.bugfix);
    if let Some(min) = infos.min_segment_lucene_version {
        let min = (min.major, min.minor, min.bugfix);
        if seg < min {
            checks.push(Check::fail(
                "commit.segment_version_at_or_after_min",
                format!(
                    "segments file recorded minSegmentLuceneVersion={min:?} but segment {} has older version={seg:?}",
                    commit.segment_name
                ),
            ));
        } else {
            checks.push(Check::pass("commit.segment_version_at_or_after_min"));
        }
    }
    if infos.index_created_version_major >= 7 {
        if si.version.major < infos.index_created_version_major {
            checks.push(Check::fail(
                "commit.segment_version_at_or_after_created",
                format!(
                    "segments file recorded indexCreatedVersionMajor={} but segment {} has older version={seg:?}",
                    infos.index_created_version_major, commit.segment_name
                ),
            ));
        } else {
            checks.push(Check::pass("commit.segment_version_at_or_after_created"));
        }
        if si.min_version.is_none() {
            checks.push(Check::fail(
                "commit.segment_records_min_version",
                format!(
                    "segments infos must record minVersion with indexCreatedVersionMajor={}",
                    infos.index_created_version_major
                ),
            ));
        } else {
            checks.push(Check::pass("commit.segment_records_min_version"));
        }
    }
}

/// Checks every segment in the latest commit found in `dir`, **plus** the
/// commit-level invariants no single segment can see.
///
/// The returned vector's first entry is the commit itself (its
/// `segment_name` is the `segments_N` file name, which is not a segment
/// name and so can never collide with one); the rest are the segments, in
/// commit order.
///
/// # Errors
/// The commit cannot be read: `segments_N` fails, or a segment's `.si`
/// fails with an I/O error that is not evidence of corruption (Java's
/// `readCommit` propagates it as-is, so `CheckIndex` blames no segment for
/// it; Lucene 10.6, #16788). A missing, truncated or corrupt `.si` is that
/// segment's `si.open` failure instead.
pub fn check_directory(dir: &dyn Directory) -> segment_infos::Result<Vec<CheckResult>> {
    let infos = segment_infos::read_latest(dir)?;
    let mut results = vec![check_commit(&infos)];
    // Segments are checked in parallel, as Java's CheckIndex does with
    // `-threadCount`; each check reads only its own segment's files, and the
    // results keep commit order.
    {
        use rayon::prelude::*;
        let segments = infos
            .segments
            .par_iter()
            .map(|commit| match read_si(dir, commit) {
                Err(segment_info::Error::Store(e)) if !is_segment_info_corruption(&e) => Err(e),
                si => Ok(check_segment_with_si(dir, Some(&infos), commit, si)),
            })
            .collect::<Result<Vec<CheckResult>, lucene_store::Error>>()?;
        results.extend(segments);
    }
    // `SegmentInfos.readCommit`'s LUCENE-6299 total-doc bound needs every
    // segment's maxDoc, which only becomes available once the `.si` files
    // above have been opened -- so it is appended to the commit's own result
    // rather than computed up front.
    let total_docs: i64 = results
        .iter()
        .skip(1)
        .filter_map(|r| r.max_doc)
        .map(i64::from)
        .sum();
    let commit_result = &mut results[0];
    if total_docs > MAX_DOCS {
        commit_result.checks.push(Check::fail(
            "commit.total_max_doc_within_bounds",
            format!("too many documents: an index cannot exceed {MAX_DOCS} but readers have total maxDoc={total_docs}"),
        ));
    } else {
        commit_result
            .checks
            .push(Check::pass("commit.total_max_doc_within_bounds"));
    }
    Ok(results)
}

/// `CheckIndex.exorciseIndex(status)` (`-exorcise`): writes the next
/// `segments_N` holding only the segments `results` -- a
/// [`check_directory`] of `dir`'s latest commit -- found sound, so every
/// document of a broken segment is lost and the rest of the index opens.
/// Returns the new commit's file name, or `None` when no segment is broken
/// and nothing is written (Java's "No problems were detected").
///
/// As Java's: under the index's write lock (`CheckIndex` holds it for its
/// whole life), the commit is the latest one read again, its version bumped
/// (`changed()`), and its segment counter moved past every remaining name
/// when the check found it behind (`newSegments.counter = maxSegmentName +
/// 1`). The dropped segments' files are left for the next writer's deleter.
///
/// # Errors
/// `results` does not describe `dir`'s latest commit (a different
/// commit, or a subset -- Java's "can only exorcise an index that was fully
/// checked"), the lock is held, or the read or write fails.
pub fn exorcise(
    dir: &dyn Directory,
    results: &[CheckResult],
) -> segment_infos::Result<Option<String>> {
    let _lock = dir.obtain_lock("write.lock")?;
    let mut infos = segment_infos::read_latest(dir)?;
    let commit_name = lucene_store::directory::segments_file_name(infos.generation);
    let segment_results = results.get(1..).unwrap_or_default();
    let covers = results.first().map(|r| Some(&r.segment_name)) == Some(commit_name.as_ref())
        && segment_results.len() == infos.segments.len()
        && segment_results
            .iter()
            .zip(&infos.segments)
            .all(|(r, s)| r.segment_name == s.segment_name);
    if !covers {
        return Err(segment_infos::Error::Store(lucene_store::Error::Io(
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "can only exorcise an index that was fully checked (these results are not a check \
             of every segment of the latest commit)",
            ),
        )));
    }
    if segment_results.iter().all(CheckResult::all_passed) {
        return Ok(None);
    }
    let mut kept = Vec::with_capacity(infos.segments.len());
    for (result, segment) in segment_results
        .iter()
        .zip(std::mem::take(&mut infos.segments))
    {
        if result.all_passed() {
            kept.push(segment);
        }
    }
    infos.segments = kept;
    let max_segment_name = infos
        .segments
        .iter()
        .filter_map(|c| {
            c.segment_name
                .strip_prefix('_')
                .and_then(|n| i64::from_str_radix(n, 36).ok())
        })
        .max()
        .unwrap_or(-1);
    if infos.counter <= max_segment_name {
        infos.counter = max_segment_name.saturating_add(1);
    }
    infos.generation = infos.generation.saturating_add(1);
    infos.version = infos.version.saturating_add(1);
    infos.id = crate::index_writer::generate_segment_id(infos.generation);
    segment_infos::write(&infos, dir).map(Some)
}

/// The `segments_N`-level invariants: no two segments share a name, and
/// `SegmentInfos.counter` is strictly greater than every segment name's
/// base-36 numeric suffix (real `CheckIndex`'s `validCounter` /
/// `maxSegmentName` check -- a counter that has fallen behind makes the next
/// flush reuse a live segment's name and silently overwrite its files).
pub fn check_commit(infos: &SegmentInfos) -> CheckResult {
    let mut checks = Vec::new();
    let mut seen: Vec<&str> = Vec::with_capacity(infos.segments.len());
    let mut dupes: Vec<&str> = Vec::new();
    for commit in &infos.segments {
        let name = commit.segment_name.as_str();
        if seen.contains(&name) {
            dupes.push(name);
        } else {
            seen.push(name);
        }
    }
    if dupes.is_empty() {
        checks.push(Check::pass("commit.segment_names_unique"));
    } else {
        checks.push(Check::fail(
            "commit.segment_names_unique",
            format!("duplicate segment name(s) in the commit: {dupes:?}"),
        ));
    }

    // `CheckIndex.updateMaxSegmentName`: `Long.parseLong(name.substring(1), 36)`.
    let mut max_segment_name: i64 = -1;
    let mut unparsable: Vec<&str> = Vec::new();
    for commit in &infos.segments {
        match commit
            .segment_name
            .strip_prefix('_')
            .and_then(|suffix| i64::from_str_radix(suffix, 36).ok())
        {
            Some(n) => max_segment_name = max_segment_name.max(n),
            None => unparsable.push(&commit.segment_name),
        }
    }
    if unparsable.is_empty() {
        checks.push(Check::pass("commit.segment_names_well_formed"));
    } else {
        checks.push(Check::fail(
            "commit.segment_names_well_formed",
            format!("segment name(s) are not `_<base36>`: {unparsable:?}"),
        ));
    }
    if max_segment_name < infos.counter {
        checks.push(Check::pass("commit.counter_ahead_of_segment_names"));
    } else {
        checks.push(Check::fail(
            "commit.counter_ahead_of_segment_names",
            format!(
                "next segment name counter {} is not greater than max segment name {max_segment_name}",
                infos.counter
            ),
        ));
    }

    CheckResult {
        segment_name: lucene_store::directory::segments_file_name(infos.generation)
            .unwrap_or_else(|| format!("<invalid generation {}>", infos.generation)),
        max_doc: None,
        checks,
        stats: CheckStats::default(),
    }
}

#[cfg(test)]
fn open_si(dir: &dyn Directory, commit: &SegmentCommitInfo) -> Result<SegmentInfo, String> {
    read_si(dir, commit).map_err(|e| e.to_string())
}

/// `codec.segmentInfoFormat().read(directory, segName, segmentID, ...)`.
fn read_si(dir: &dyn Directory, commit: &SegmentCommitInfo) -> segment_info::Result<SegmentInfo> {
    let bytes = dir.open(&format!("{}.si", commit.segment_name))?;
    segment_info::parse_for_codec(&bytes, &commit.segment_id, &commit.codec_name)
}

/// The `.si` failures `SegmentInfos.readCommit` wraps in a
/// `CorruptSegmentInfoException` (Lucene 10.6, branch_10_6, #16476 + #16788):
/// `CorruptIndexException | EOFException | NoSuchFileException |
/// FileNotFoundException`. Every other [`segment_info::Error`] is a decode
/// error (`CorruptIndexException`); a store error of any other kind (a
/// transient I/O failure, a closed directory, ...) "is no evidence of
/// corruption and must propagate as-is".
fn is_segment_info_corruption(e: &lucene_store::Error) -> bool {
    match e {
        lucene_store::Error::Eof { .. }
        | lucene_store::Error::MalformedVarint
        | lucene_store::Error::Corrupted(_) => true,
        lucene_store::Error::Io(_) => e.is_no_such_file(),
        _ => false,
    }
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

/// Every file `.si` lists must exist and pass `CodecUtil.checksumEntireFile`
/// -- the whole payload re-read and its CRC-32 recomputed, then compared
/// against the footer's stored checksum.
///
/// This is real `CheckIndex`'s `test: check integrity` step, which runs
/// `reader.checkIntegrity()` (every format's `checkIntegrity` bottoms out in
/// `CodecUtil.checksumEntireFile`) on every segment file.
/// [`codec_util::retrieve_checksum`] would not do: it validates only the
/// footer's *shape* (magic/algorithm id/checksum field), so a flipped byte in
/// the middle of a `.tim`/`.fdt`/`.dvd` would pass cleanly and be caught, if at
/// all, by whichever decoder happened to trip over it. It shares
/// [`codec_util::check_whole_file_footer`] with `merge.rs`'s pre-bulk-copy
/// `check_integrity` and with [`crate::checksum_verify`].
fn check_files_exist_and_validate(
    dir: &dyn Directory,
    commit: &SegmentCommitInfo,
    si: &SegmentInfo,
    checks: &mut Vec<Check>,
) {
    // `Lucene99SegmentInfoFormat.write` does `si.addFile(fileName)` before
    // writing, so a real Lucene `.si` always lists itself. `IndexFileDeleter`
    // reference-counts from exactly this set, so a `.si` missing from its own
    // file list is a file nothing holds a reference to.
    let si_name = format!("{}.si", commit.segment_name);
    if si.files.contains(&si_name) {
        checks.push(Check::pass("si.files_lists_itself"));
    } else {
        checks.push(Check::fail(
            "si.files_lists_itself",
            format!("SegmentInfo.files does not list {si_name}"),
        ));
    }

    // `SegmentCommitInfo.files()`, not `SegmentInfo.files`: a `.liv` or a
    // generational field-infos/doc-values-update file is recorded on the
    // commit and never in the already-written `.si`, so checking only
    // `si.files` skipped exactly the files a delete/update round wrote.
    for file in &commit.files(&si.files) {
        let name = format!("file:{file}");
        if per_field_postings::is_unchecksummed_dictionary_file(file) {
            // `RAMOnly`'s token: its format's `checkIntegrity` is a no-op.
            match dir.file_length(file) {
                Ok(_) => checks.push(Check::pass(name)),
                Err(e) => checks.push(Check::fail(name, e.to_string())),
            }
            continue;
        }
        match dir.open(file) {
            Ok(bytes) => {
                let payload_end = bytes.len().saturating_sub(codec_util::FOOTER_LENGTH);
                match codec_util::check_whole_file_footer(&bytes, payload_end) {
                    Ok(_) => checks.push(Check::pass(name)),
                    Err(e) => checks.push(Check::fail(name, e.to_string())),
                }
            }
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
    let has_postings_files = has_any_ext(files, &[".tim", ".tip", ".tmd"])
        || per_field_postings::has_postings_files(files);

    let any_field_claims_dv = fields
        .fields
        .iter()
        .any(|f| f.doc_values_type != field_infos::DocValuesType::None);
    // Norms only ever exist for an *indexed* field -- a non-indexed field's
    // `omit_norms` flag is meaningless noise real Lucene doesn't act on
    // (confirmed against a real fixture: `Lucene90PointsWriter`-backed,
    // non-indexed numeric-point fields are written with `omit_norms=false`
    // by default even though no norms file is ever produced for them, since
    // `index_options == None` already means "no norms, full stop" -- see
    // `Lucene94FieldInfosFormat`/`FieldInvertState`). Without the
    // `index_options != None` guard this check would falsely flag every
    // points-only (or doc-values-only) field as an orphaned norms claim.
    let any_field_claims_norms = fields
        .fields
        .iter()
        .any(|f| !f.omit_norms && f.index_options != field_infos::IndexOptions::None);
    let any_field_claims_tv = fields.fields.iter().any(|f| f.store_term_vectors);
    // Java's FreqProxTermsWriter.flush only routes fields with terms. A
    // zero-token field remains indexed (and may have norms) without any
    // postings files. Routing attributes claim files even after a merge
    // removes the last term; legacy direct formats have no such attributes,
    // so their indexed fields claim postings only when files are present.
    let any_field_claims_postings = fields.fields.iter().any(|f| {
        f.index_options != field_infos::IndexOptions::None
            && (has_postings_files
                || f.attributes.iter().any(|(key, _)| {
                    key == lucene_codecs::per_field_postings::PER_FIELD_FORMAT_KEY
                        || key == lucene_codecs::per_field_postings::PER_FIELD_SUFFIX_KEY
                }))
    });

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
    match lucene_codecs::per_field_postings::validate_routed_groups(&fields.fields, files) {
        Err(message) => checks.push(Check::fail("fnm.postings_vs_files", message)),
        Ok(()) => check_claim_vs_files(
            "fnm.postings_vs_files",
            any_field_claims_postings,
            has_postings_files,
            ".tim/.tip/.tmd",
            checks,
        ),
    }
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
) -> Option<lucene_util::fixed_bit_set::FixedBitSet> {
    if commit.del_gen == -1 {
        return None;
    }
    let liv_name = liv_file_name(&commit.segment_name, commit.del_gen);
    let bytes = match dir.open(&liv_name) {
        Ok(b) => b,
        Err(e) => {
            checks.push(Check::fail("liv.open", e.to_string()));
            skip_families(checks, LIV_FAMILIES, "liv.open");
            return None;
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
                    .saturating_sub(header_end)
                    .saturating_sub(codec_util::FOOTER_LENGTH);
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
                skip_families(checks, LIV_FAMILIES, "liv.open");
                return None;
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
        Ok(bits) => {
            checks.push(Check::pass("liv.open"));
            checks.push(Check::pass("liv.cardinality_matches_del_count"));
            return Some(bits);
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
            skip_families(checks, &["liv.cardinality_matches_del_count"], "liv.open");
        }
    }
    // Returning `None` here is itself a degradation worth knowing about, and
    // it is *not* modelled as a skip because nothing is skipped: every check
    // below runs, but against a segment it now believes has no deletions.
    // `liv.open`'s own failure is the signal, and `all_passed()` is already
    // false because of it.
    None
}

/// The two families a failed `liv.open` takes down.
const LIV_FAMILIES: &[&str] = &[
    "liv.max_doc_matches_si",
    "liv.cardinality_matches_del_count",
];

/// Is `doc` live, given the segment's `.liv` bitset (`None` = no deletions)?
///
/// The crate rule in `docs/arithmetic-gate.md`: **never index a `FixedBitSet`
/// with an index bounded against anything other than that bitset's own
/// `len()`.** Every caller here has a doc id bounded against `.si`'s
/// `doc_count`, while the bitset comes out of the `.liv` -- two independent
/// files. They agree today only because [`check_live_docs`] hands
/// `si.doc_count` to `live_docs::parse` as `max_doc`, which is one line away
/// from not being true. `FixedBitSet::get` checks its bound in release as
/// well as debug, so the failure would be a panic -- in a
/// *verifier*, which is meant to report damage, not die on it.
///
/// A doc id outside the bitset counts as **live**, deliberately: this is a
/// verifier, an out-of-range doc id is already reported by the caller that
/// produced it (`postings.doc_ids_valid`, or the loop's own `0..doc_count`
/// bound), and quietly treating it as deleted would suppress every check that
/// only looks at live documents.
fn is_live_at(live_docs: Option<&lucene_util::fixed_bit_set::FixedBitSet>, doc: usize) -> bool {
    live_docs.is_none_or(|bits| doc >= bits.len() || bits.get(doc))
}

/// [`is_live_at`] for a doc id that is still a (possibly negative) `i32`.
fn is_live(live_docs: Option<&lucene_util::fixed_bit_set::FixedBitSet>, doc: i32) -> bool {
    usize::try_from(doc).map_or(true, |d| is_live_at(live_docs, d))
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

    let result = (|| -> Result<(i32, Result<usize, String>), String> {
        let fdt = dir.open(fdt_name).map_err(|e| e.to_string())?;
        let fdx = dir.open(fdx_name).map_err(|e| e.to_string())?;
        let fdm = dir.open(fdm_name).map_err(|e| e.to_string())?;
        // A `CompressingCodec` format's files carry its segment suffix.
        let suffix = crate::index_writer::format_suffix(&si.files, &commit.segment_name, "fdt")
            .unwrap_or_default();
        let reader = stored_fields::open(&fdt, &fdx, &fdm, &commit.segment_id, &suffix)
            .map_err(|e| e.to_string())?;
        let max_doc = reader.max_doc();
        // `CheckIndex.testStoredFields` pulls *every* document, deleted ones
        // included ("to make sure they too are not corrupt"). Comparing
        // `max_doc` alone touches only the `.fdm` metadata and the `.fdx`
        // index -- it never decompresses a single chunk, so a corrupted
        // `.fdt` body passed cleanly. Decoding every document is what
        // actually exercises the LZ4/DEFLATE chunk bodies and the per-doc
        // field-number/type stream.
        let decode = (|| -> Result<usize, String> {
            let mut total_fields = 0usize;
            for doc_id in 0..max_doc {
                let doc = reader
                    .document(doc_id)
                    .map_err(|e| format!("docID={doc_id}: {e}"))?;
                total_fields = total_fields.saturating_add(doc.fields.len());
            }
            Ok(total_fields)
        })();
        Ok((max_doc, decode))
    })();

    match result {
        Ok((max_doc, decode)) => {
            if max_doc == si.doc_count {
                checks.push(Check::pass("stored_fields.doc_count_matches_si"));
            } else {
                checks.push(Check::fail(
                    "stored_fields.doc_count_matches_si",
                    format!(
                        "si.doc_count={} but stored fields reader reports max_doc={max_doc}",
                        si.doc_count
                    ),
                ));
            }
            match decode {
                Ok(total_fields) => checks.push(Check {
                    name: "stored_fields.every_doc_decodes".to_string(),
                    outcome: Outcome::Passed,
                    message: format!("{max_doc} docs, {total_fields} stored field values"),
                }),
                Err(e) => checks.push(Check::fail("stored_fields.every_doc_decodes", e)),
            }
        }
        Err(e) => {
            checks.push(Check::fail("stored_fields.doc_count_matches_si", e));
            skip_families(
                checks,
                &["stored_fields.every_doc_decodes"],
                "stored_fields.doc_count_matches_si",
            );
        }
    }
}

/// `CheckIndex.testTermVectors`. Reads *every* doc's vectors -- deleted ones
/// included, "to make sure they too are not corrupt" -- checks that every
/// field a document actually carries vectors for is a field `.fnm` marks
/// `storeTermVectors`, and, at Java's slow level, cross-checks each vector
/// against the inverted index:
///
/// - `term_vectors.doc_count_matches_si`, `.every_doc_decodes`,
///   `.fields_marked_in_fnm` -- as before.
/// - `term_vectors.self_consistent` -- `checkFields(tfv, ..., isVectors =
///   true)` applied to the one-document `Fields` a vector is: terms strictly
///   increasing, `freq > 0` and equal to the number of positions, positions
///   non-decreasing, `endOffset >= startOffset`, and the per-field
///   `hasPositions`/`hasOffsets`/`hasPayloads` flags agreeing with what the
///   terms actually carry.
/// - `term_vectors.match_postings` -- Java's `level >=
///   MIN_LEVEL_FOR_SLOW_CHECKS` block: the vector's field must exist in the
///   postings, every vector term must exist there too, that term's postings
///   must contain this doc, and the freq / positions / offsets / payloads
///   recorded in the vector must equal the ones in the inverted index.
///
/// The reverse `.fnm`-claims-vectors direction is covered segment-wide by
/// [`check_field_flags_vs_files`]'s `fnm.term_vectors_vs_files`.
///
/// Skipped (not failed) for a segment with no `.tvd`/`.tvx`/`.tvm` at
/// all. A partial file group is reported rather than skipped.
#[allow(clippy::too_many_arguments)]
fn check_term_vectors(
    dir: &dyn Directory,
    commit: &SegmentCommitInfo,
    si: &SegmentInfo,
    field_infos: Option<&FieldInfos>,
    postings: Option<&PostingsHandles<'_>>,
    live_docs: Option<&lucene_util::fixed_bit_set::FixedBitSet>,
    stats: &mut CheckStats,
    checks: &mut Vec<Check>,
) {
    let tvd = si.files.iter().find(|f| f.ends_with(".tvd"));
    let tvx = si.files.iter().find(|f| f.ends_with(".tvx"));
    let tvm = si.files.iter().find(|f| f.ends_with(".tvm"));
    let (tvd, tvx, tvm) = match (tvd, tvx, tvm) {
        (Some(d), Some(x), Some(m)) => (d, x, m),
        (None, None, None) => return,
        _ => {
            checks.push(Check::fail(
                "term_vectors.open",
                "segment has some but not all of .tvd/.tvx/.tvm",
            ));
            return;
        }
    };

    let result = (|| -> Result<(Vec<Check>, i64), String> {
        let tvd = dir.open(tvd).map_err(|e| e.to_string())?;
        let tvx = dir.open(tvx).map_err(|e| e.to_string())?;
        let tvm = dir.open(tvm).map_err(|e| e.to_string())?;
        let suffix = crate::index_writer::format_suffix(&si.files, &commit.segment_name, "tvd")
            .unwrap_or_default();
        let reader = term_vectors::open(&tvd, &tvx, &tvm, &commit.segment_id, &suffix)
            .map_err(|e| e.to_string())?;

        let mut out = Vec::new();
        if reader.max_doc() == si.doc_count {
            out.push(Check::pass("term_vectors.doc_count_matches_si"));
        } else {
            out.push(Check::fail(
                "term_vectors.doc_count_matches_si",
                format!(
                    "si.doc_count={} but term vectors reader reports max_doc={}",
                    si.doc_count,
                    reader.max_doc()
                ),
            ));
        }

        let mut decode_problems: Vec<String> = Vec::new();
        let mut flag_problems: Vec<String> = Vec::new();
        let mut self_problems: Vec<String> = Vec::new();
        let mut postings_problems: Vec<String> = Vec::new();
        let mut docs_with_vectors = 0i64;
        let mut tot_vectors = 0i64;
        let mut memo: TermPostingsMemo = std::collections::HashMap::new();
        let mut memo_elements = 0usize;
        // Field number -> `FieldInfo`, built once rather than scanned per
        // document per vector field. The first entry wins, as `find` did.
        let by_number: std::collections::HashMap<i32, &field_infos::FieldInfo> = field_infos
            .map(|fi| {
                let mut m = std::collections::HashMap::with_capacity(fi.fields.len());
                for f in &fi.fields {
                    m.entry(f.number).or_insert(f);
                }
                m
            })
            .unwrap_or_default();
        for doc_id in 0..si.doc_count.min(reader.max_doc()) {
            match reader.document(doc_id) {
                Ok(None) => {}
                Ok(Some(document)) => {
                    docs_with_vectors = docs_with_vectors.saturating_add(1);
                    let live = is_live(live_docs, doc_id);
                    if field_infos.is_none() {
                        continue;
                    }
                    for field in &document.fields {
                        let info = by_number.get(&field.field_number).copied();
                        match info.map(|f| f.store_term_vectors) {
                            Some(true) => {}
                            Some(false) => flag_problems.push(format!(
                                "docID={doc_id} has term vectors for field number {} but \
                                 FieldInfo has storeTermVector=false",
                                field.field_number
                            )),
                            None => flag_problems.push(format!(
                                "docID={doc_id} has term vectors for field number {} which is \
                                 not in .fnm at all",
                                field.field_number
                            )),
                        }
                        if live {
                            tot_vectors = tot_vectors.saturating_add(1);
                        }
                        check_one_vector_field(
                            doc_id,
                            field,
                            info,
                            postings,
                            &mut memo,
                            &mut memo_elements,
                            &mut self_problems,
                            &mut postings_problems,
                        );
                    }
                }
                Err(e) => decode_problems.push(format!("docID={doc_id}: {e}")),
            }
        }
        out.push(named_field_check(
            "term_vectors.every_doc_decodes",
            &decode_problems,
            si.doc_count as i64,
            "docs",
        ));
        // These three walk *fields*, and the per-document loop above skips
        // every field when there is no `.fnm` to name them -- so with the
        // `.fnm` unreadable their problem lists are empty for the one reason
        // that must never read as a pass: nothing looked. Same for the
        // cross-check when the postings side is missing. This is the case
        // of `fnm.open` failing and the segment still reporting
        // clean term vectors).
        if field_infos.is_none() {
            skip_families(
                &mut out,
                &[
                    "term_vectors.fields_marked_in_fnm",
                    "term_vectors.self_consistent",
                    "term_vectors.match_postings",
                ],
                "fnm.open",
            );
            return Ok((out, tot_vectors));
        }
        out.push(named_field_check(
            "term_vectors.fields_marked_in_fnm",
            &flag_problems,
            docs_with_vectors,
            "docs with vectors",
        ));
        out.push(named_field_check(
            "term_vectors.self_consistent",
            &self_problems,
            docs_with_vectors,
            "docs with vectors",
        ));
        // No vectors means nothing to cross-check, postings or not: a
        // segment whose term-vector field produced no tokens has neither.
        if postings.is_none() && docs_with_vectors > 0 {
            // A cross-check with one side missing is not a cross-check.
            // (`postings.open` already reported the reason; if the segment
            // simply has no term dictionary, `fnm.postings_vs_files` did.)
            out.push(Check::skipped(
                "term_vectors.match_postings",
                "the postings side",
            ));
        } else {
            out.push(named_field_check(
                "term_vectors.match_postings",
                &postings_problems,
                docs_with_vectors,
                "docs with vectors",
            ));
        }
        Ok((out, tot_vectors))
    })();

    match result {
        Ok((cs, tot_vectors)) => {
            checks.extend(cs);
            stats.term_vector_fields = stats.term_vector_fields.saturating_add(tot_vectors);
        }
        Err(e) => {
            checks.push(Check::fail("term_vectors.open", e));
            skip_families(
                checks,
                &[
                    "term_vectors.doc_count_matches_si",
                    "term_vectors.every_doc_decodes",
                    "term_vectors.fields_marked_in_fnm",
                    "term_vectors.self_consistent",
                    "term_vectors.match_postings",
                ],
                "term_vectors.open",
            );
        }
    }
}

/// One `(document, field)` term vector: `checkFields(..., isVectors = true)`
/// plus the vectors-vs-inverted-index cross-check. Split out of
/// [`check_term_vectors`] purely to keep that function's nesting readable.
#[allow(clippy::too_many_arguments)]
fn check_one_vector_field(
    doc_id: i32,
    field: &term_vectors::TermVectorField,
    info: Option<&field_infos::FieldInfo>,
    postings: Option<&PostingsHandles<'_>>,
    memo: &mut TermPostingsMemo,
    memo_elements: &mut usize,
    self_problems: &mut Vec<String>,
    postings_problems: &mut Vec<String>,
) {
    let field_name = info.map(|f| f.name.as_str()).unwrap_or("<unknown>");
    // `checkFields`, `isVectors == true` branch: term order, `freq > 0`,
    // positions non-decreasing, `endOffset >= startOffset`, and the
    // per-field flags matching what the terms actually carry.
    let mut prev: Option<&[u8]> = None;
    for t in &field.terms {
        if let Some(p) = prev {
            if t.term.as_slice() <= p {
                self_problems.push(format!(
                    "docID={doc_id} field {field_name:?}: vector terms out of order \
                     ({:?} after {p:?})",
                    t.term
                ));
            }
        }
        prev = Some(&t.term);
        if t.freq <= 0 {
            self_problems.push(format!(
                "docID={doc_id} field {field_name:?} term {:?}: freq {} is out of bounds",
                t.term, t.freq
            ));
        }
        if let Some(positions) = &t.positions {
            if positions.len() as i32 != t.freq {
                self_problems.push(format!(
                    "docID={doc_id} field {field_name:?} term {:?}: freq={} but {} positions",
                    t.term,
                    t.freq,
                    positions.len()
                ));
            }
            let mut last = -1i32;
            for &pos in positions {
                if pos < 0 || pos < last {
                    self_problems.push(format!(
                        "docID={doc_id} field {field_name:?} term {:?}: position {pos} is out \
                         of bounds or before the previous position {last}",
                        t.term
                    ));
                    break;
                }
                last = pos;
            }
        } else if field.has_positions {
            self_problems.push(format!(
                "docID={doc_id} field {field_name:?} term {:?}: the field claims positions but \
                 the term carries none",
                t.term
            ));
        }
        match (&t.start_offsets, &t.end_offsets) {
            (Some(starts), Some(ends)) => {
                if starts.len() != ends.len() || starts.len() as i32 != t.freq {
                    self_problems.push(format!(
                        "docID={doc_id} field {field_name:?} term {:?}: freq={} but {} start / \
                         {} end offsets",
                        t.term,
                        t.freq,
                        starts.len(),
                        ends.len()
                    ));
                }
                for (&s, &e) in starts.iter().zip(ends.iter()) {
                    if s < 0 || e < s {
                        self_problems.push(format!(
                            "docID={doc_id} field {field_name:?} term {:?}: offsets [{s}, {e}] \
                             are out of bounds",
                            t.term
                        ));
                        break;
                    }
                }
            }
            (None, None) => {
                if field.has_offsets {
                    self_problems.push(format!(
                        "docID={doc_id} field {field_name:?} term {:?}: the field claims \
                         offsets but the term carries none",
                        t.term
                    ));
                }
            }
            _ => self_problems.push(format!(
                "docID={doc_id} field {field_name:?} term {:?}: start and end offsets disagree \
                 about being present",
                t.term
            )),
        }
        if t.payloads.is_some() && !field.has_payloads {
            self_problems.push(format!(
                "docID={doc_id} field {field_name:?} term {:?}: payloads present on a vector \
                 field whose header says hasPayloads=false",
                t.term
            ));
        }
    }

    // `testTermVectors`' slow-level block: the vector must agree with the
    // inverted index, term for term.
    let (Some(p), Some(fi)) = (postings, info) else {
        return;
    };
    // A field with no postings at all cannot be cross-checked; Java throws
    // "vector field=... does not exist in postings" only because in Java a
    // term-vector'd field is always indexed.
    let Some(field_terms) = p.fields.field(&fi.name) else {
        if fi.index_options != field_infos::IndexOptions::None {
            postings_problems.push(format!(
                "docID={doc_id}: vector field {field_name:?} does not exist in the postings"
            ));
        }
        return;
    };
    let postings_has_freqs = !matches!(
        fi.index_options,
        field_infos::IndexOptions::None | field_infos::IndexOptions::Docs
    );
    let postings_has_positions = matches!(
        fi.index_options,
        field_infos::IndexOptions::DocsAndFreqsAndPositions
            | field_infos::IndexOptions::DocsAndFreqsAndPositionsAndOffsets
    );
    let postings_has_offsets = matches!(
        fi.index_options,
        field_infos::IndexOptions::DocsAndFreqsAndPositionsAndOffsets
    );
    for t in &field.terms {
        let key = (fi.number, t.term.clone());
        if !memo.contains_key(&key) {
            // Java re-pulls a `PostingsEnum` per (doc, term) and skips to the
            // doc; this port's decoders materialize a term's whole postings
            // list, so doing the same would be quadratic in `docFreq`. The
            // memo makes the cross-check cost one decode per *distinct* term
            // per field instead of one per (doc, term) pair -- i.e. O(sum of
            // docFreq) overall, the same total as `check_postings`' own pass.
            // It is bounded so a huge vocabulary cannot pin an unbounded
            // amount of positions data in memory; past the cap the memo is
            // dropped and refills, which degrades the constant factor and
            // never the result.
            if *memo_elements >= TERM_VECTOR_POSTINGS_MEMO_ELEMENTS {
                memo.clear();
                *memo_elements = 0;
            }
            let mut cursor = field_terms.iter();
            let found = match cursor.try_seek_ceil(&t.term) {
                Ok(blocktree::SeekStatus::Found) => true,
                Ok(_) => false,
                Err(e) => {
                    postings_problems.push(format!(
                        "docID={doc_id} field {field_name:?}: seeking vector term {:?} in the \
                         postings failed: {e}",
                        t.term
                    ));
                    continue;
                }
            };
            if !found {
                postings_problems.push(format!(
                    "docID={doc_id} field {field_name:?}: vector term {:?} does not exist in \
                     the postings",
                    t.term
                ));
                continue;
            }
            let decoded = if postings_has_positions {
                match p.pos_in.as_ref() {
                    Some(pos_in) => cursor
                        .try_current_postings_and_positions(
                            p.doc_in.as_ref(),
                            pos_in,
                            p.pay_in.as_ref(),
                        )
                        .map(|o| o.map(|(d, pos)| (d, Some(pos)))),
                    None => Ok(None),
                }
            } else {
                cursor
                    .try_current_postings(p.doc_in.as_ref())
                    .map(|o| o.map(|d| (d, None)))
            };
            match decoded {
                Ok(Some((docs, positions))) => {
                    *memo_elements = memo_elements
                        .saturating_add(docs.docs.len())
                        .saturating_add(positions.as_ref().map_or(0, |p| {
                            p.iter().map(|v| v.len()).fold(0, usize::saturating_add)
                        }));
                    memo.insert(key.clone(), (docs, positions));
                }
                Ok(None) => continue,
                Err(e) => {
                    postings_problems.push(format!(
                        "docID={doc_id} field {field_name:?} term {:?}: decoding its postings \
                         failed: {e}",
                        t.term
                    ));
                    continue;
                }
            }
        }
        let (doc_postings, positions) = &memo[&key];
        // `docs` is strictly increasing (the format guarantees it and
        // `postings.doc_ids_valid` verifies it), so this is Java's
        // `postingsDocs.advance(j)` at the same cost -- a linear scan here
        // would leave the cross-check quadratic in `docFreq`, which is
        // exactly what the memo above exists to avoid.
        let Ok(at) = doc_postings.docs.binary_search(&doc_id) else {
            postings_problems.push(format!(
                "docID={doc_id} field {field_name:?}: vector term {:?} was not found in the \
                 postings for this doc",
                t.term
            ));
            continue;
        };
        // Java's `postingsHasFreq` guard: a field may store term vectors
        // while its postings omit frequencies (`IndexOptions.DOCS`), in
        // which case the postings decoder synthesizes freq 1 for every
        // document while the vector carries the real one. Comparing them
        // would fail on a healthy segment.
        if postings_has_freqs {
            let postings_freq = doc_postings.freqs.get(at).copied().unwrap_or(0);
            if postings_freq != t.freq {
                postings_problems.push(format!(
                    "docID={doc_id} field {field_name:?} term {:?}: vector freq={} differs \
                     from postings freq={postings_freq}",
                    t.term, t.freq
                ));
            }
        }
        let Some(post_positions) = positions.as_ref().and_then(|p| p.get(at)) else {
            continue;
        };
        if postings_has_positions {
            if let Some(vec_positions) = &t.positions {
                let post: Vec<i32> = post_positions.iter().map(|o| o.position).collect();
                if &post != vec_positions {
                    postings_problems.push(format!(
                        "docID={doc_id} field {field_name:?} term {:?}: vector positions {:?} \
                         differ from postings positions {post:?}",
                        t.term, vec_positions
                    ));
                }
            }
        }
        if postings_has_offsets {
            if let (Some(starts), Some(ends)) = (&t.start_offsets, &t.end_offsets) {
                let post_starts: Vec<i32> = post_positions.iter().map(|o| o.start_offset).collect();
                let post_ends: Vec<i32> = post_positions.iter().map(|o| o.end_offset).collect();
                if &post_starts != starts || &post_ends != ends {
                    postings_problems.push(format!(
                        "docID={doc_id} field {field_name:?} term {:?}: vector offsets \
                         {starts:?}/{ends:?} differ from postings offsets \
                         {post_starts:?}/{post_ends:?}",
                        t.term
                    ));
                }
            }
        }
        if fi.store_payloads {
            if let Some(vec_payloads) = &t.payloads {
                let post_payloads: Vec<&[u8]> = post_positions
                    .iter()
                    .map(|o| o.payload.as_slice())
                    .collect();
                let vec_slices: Vec<&[u8]> = vec_payloads.iter().map(|p| p.as_slice()).collect();
                if post_payloads != vec_slices {
                    postings_problems.push(format!(
                        "docID={doc_id} field {field_name:?} term {:?}: vector payloads differ \
                         from postings payloads",
                        t.term
                    ));
                }
            }
        }
    }
}

/// How many decoded elements -- documents plus position occurrences -- the
/// term-vector cross-check's memo may hold at once. Bounded in *elements*
/// rather than entries because one high-`docFreq` term with positions can
/// be larger than thousands of singletons, so an entry count would not
/// actually bound the memory. See the memo's own comment: purely a
/// constant-factor bound, never a change of result.
const TERM_VECTOR_POSTINGS_MEMO_ELEMENTS: usize = 1 << 20;

type TermPostingsMemo = std::collections::HashMap<
    (i32, Vec<u8>),
    (
        postings::Postings,
        Option<Vec<Vec<lucene_codecs::postings::Position>>>,
    ),
>;

/// Builds one named `Check` from a field's collected list of problem
/// messages (empty -> pass, non-empty -> fail listing at most the first 5,
/// with a total count) -- shared by every per-field check here so the "how
/// many terms, show a few" reporting shape isn't duplicated.
fn named_field_check<P: ProblemList + ?Sized>(
    name: &str,
    problems: &P,
    num_units: i64,
    unit: &str,
) -> Check {
    counted_field_check(name, problems.total(), problems.shown(), num_units, unit)
}

/// What [`named_field_check`] reports from: how many problems there were and
/// the messages it spells out.
trait ProblemList {
    fn total(&self) -> usize;
    fn shown(&self) -> &[String];
}

impl ProblemList for [String] {
    fn total(&self) -> usize {
        self.len()
    }
    fn shown(&self) -> &[String] {
        &self[..self.len().min(SHOWN_PROBLEMS)]
    }
}

impl ProblemList for Vec<String> {
    fn total(&self) -> usize {
        self.len()
    }
    fn shown(&self) -> &[String] {
        self.as_slice().shown()
    }
}

impl ProblemList for Problems {
    fn total(&self) -> usize {
        self.total
    }
    fn shown(&self) -> &[String] {
        &self.shown
    }
}

/// How many problem messages a failed [`named_field_check`] spells out.
const SHOWN_PROBLEMS: usize = 5;

/// [`named_field_check`] over a total and the messages it shows.
fn counted_field_check(
    name: &str,
    total: usize,
    shown: &[String],
    num_units: i64,
    unit: &str,
) -> Check {
    if total == 0 {
        Check::pass(name)
    } else {
        Check::fail(
            name,
            format!(
                "{total} of {num_units} {unit} affected; first {}: {}",
                shown.len(),
                shown.join("; ")
            ),
        )
    }
}

/// A problem list for a check that can fire once per document, point or
/// node: on a thoroughly corrupt file that is millions of messages of which
/// [`named_field_check`] shows five. Keeps the messages it shows and counts
/// the rest, so the report is identical and the memory is bounded.
#[derive(Default)]
struct Problems {
    shown: Vec<String>,
    total: usize,
}

impl Problems {
    fn new() -> Self {
        Self::default()
    }

    fn push(&mut self, message: String) {
        self.total = self.total.saturating_add(1);
        if self.shown.len() < SHOWN_PROBLEMS {
            self.shown.push(message);
        }
    }

    /// Appends another list's problems: its shown messages while there is
    /// room, and its whole count.
    fn absorb(&mut self, other: Problems) {
        let extra = other.total.saturating_sub(other.shown.len());
        for message in other.shown {
            self.push(message);
        }
        self.total = self.total.saturating_add(extra);
    }
}

/// `IndexWriter.MAX_POSITION`: the largest token position `IndexWriter` will
/// accept (`Integer.MAX_VALUE - 128`). `checkFields` rejects anything past
/// it; a position beyond this could only come from a corrupt `.pos` or a
/// writer that never checked.
///
/// Re-exported from the **writer's** definition rather than duplicated. The two
/// halves of one rule -- what [`crate::indexing_chain::advance_position`]
/// clamps to and what `postings.positions_valid` rejects -- must move together,
/// and a verifier with its own copy of the writer's ceiling is a verifier that
/// can silently stop agreeing with the writer it verifies.
use crate::indexing_chain::MAX_POSITION;

/// The segment's postings files, read into memory once so the term
/// dictionary, the `.doc`/`.pos`/`.pay` streams and the term-vector
/// cross-check can all borrow the same bytes -- the equivalent of real
/// `CheckIndex` handing one `CodecReader` to `testPostings` and
/// `testTermVectors`.
struct PostingsFileBytes {
    tim: lucene_store::directory::Input,
    tip: lucene_store::directory::Input,
    tmd: lucene_store::directory::Input,
    doc: Option<lucene_store::directory::Input>,
    pos: Option<lucene_store::directory::Input>,
    pay: Option<lucene_store::directory::Input>,
    /// The postings codec's per-segment suffix, e.g. `Lucene104_0`, taken
    /// from the `.tim`'s own file name.
    segment_suffix: String,
    /// `PerFieldPostingsFormat` with more than one format: each group's
    /// suffix, dictionary files and where its `.doc`/`.pos`/`.pay` start in
    /// `doc`/`pos`/`pay`, which then hold every group's files concatenated,
    /// each validated as it was read. Empty for a single-format segment.
    groups: Vec<PostingsGroupBytes>,
}

/// One postings format's dictionary of a multi-format segment.
struct PostingsGroupBytes {
    suffix: String,
    /// In `per_field_postings::dictionary_extensions` order.
    dictionary: Vec<lucene_store::directory::Input>,
    base: [u64; 3],
    /// The dictionary is already a block tree: a re-encoded group
    /// (`per_field_postings::reencode_group`).
    block_tree: bool,
}

/// The opened readers over [`PostingsFileBytes`].
struct PostingsHandles<'a> {
    fields: blocktree::BlockTreeFields,
    doc_in: Option<DocInput<'a>>,
    pos_in: Option<postings::PosInput<'a>>,
    pay_in: Option<postings::PayInput<'a>>,
}

/// Reads the segment's `.tim`/`.tip`/`.tmd` (plus `.doc`/`.pos`/`.pay` when
/// present) into memory.
///
/// `None` means "nothing to check here", not "ok": a segment with none/only
/// some of
/// `.tim`/`.tip`/`.tmd` -- a partial group is already reported by
/// [`check_field_flags_vs_files`]'s `fnm.postings_vs_files`, and this
/// function does not duplicate that failure.
fn open_postings_bytes(
    dir: &dyn Directory,
    si: &SegmentInfo,
    segment_name: &str,
    field_infos: Option<&FieldInfos>,
) -> Option<Result<PostingsFileBytes, String>> {
    let find = |ext: &str| si.files.iter().find(|f| f.ends_with(ext));
    let markers = group_markers(si, segment_name);
    if markers.len() > 1
        || markers.iter().any(|(_, suffix)| {
            per_field_postings::is_foreign_format(per_field_postings::format_of_suffix(suffix))
        })
    {
        return Some(open_postings_groups(dir, si, &markers, field_infos));
    }
    let (tim_name, tip_name, tmd_name) = match (find(".tim"), find(".tip"), find(".tmd")) {
        (Some(a), Some(b), Some(c)) => (a.clone(), b.clone(), c.clone()),
        _ => return None,
    };
    // The postings codec suffix is embedded in the sub-file's own name:
    // strip the `<segment_name>_` prefix (e.g. `_0_Lucene104_0.tim` ->
    // `Lucene104_0`) and the `.tim` extension -- the same derivation
    // `lucene-search`'s `directory_reader.rs` (`SegmentReader::open`) uses,
    // duplicated here rather than shared since that logic lives in a crate
    // this module has no dependency on (see this module's own top doc
    // comment on why it doesn't build on `lucene-search`).
    let segment_suffix = suffix_of(&tim_name, segment_name, ".tim");
    let open = |name: &str| dir.open(name).map_err(|e| e.to_string());
    Some((|| {
        Ok(PostingsFileBytes {
            tim: open(&tim_name)?,
            tip: open(&tip_name)?,
            tmd: open(&tmd_name)?,
            doc: find(".doc").map(|n| open(n)).transpose()?,
            pos: find(".pos").map(|n| open(n)).transpose()?,
            pay: find(".pay").map(|n| open(n)).transpose()?,
            segment_suffix,
            groups: Vec::new(),
        })
    })())
}

/// Every postings group's marker file -- its `.tmd`, or a foreign
/// dictionary's first file (`per_field_postings::marker_extension`) -- with
/// the group's codec suffix, in file order.
fn group_markers<'a>(si: &'a SegmentInfo, segment_name: &str) -> Vec<(&'a String, String)> {
    si.files
        .iter()
        .filter_map(|f| {
            let (_, ext) = f.rsplit_once('.')?;
            let suffix = suffix_of(f, segment_name, &format!(".{ext}"));
            let format = per_field_postings::format_of_suffix(&suffix);
            (per_field_postings::marker_extension(format) == ext
                && (ext == "tmd" || per_field_postings::is_foreign_format(format)))
            .then_some((f, suffix))
        })
        .collect()
}

/// [`open_postings_bytes`] for a segment with one group per postings
/// format, or with a group of a dictionary other than the block tree:
/// `PerFieldPostingsFormat.FieldsReader`, every group's header and footer
/// checked here, their `.doc`/`.pos`/`.pay` concatenated.
fn open_postings_groups(
    dir: &dyn Directory,
    si: &SegmentInfo,
    markers: &[(&String, String)],
    field_infos: Option<&FieldInfos>,
) -> Result<PostingsFileBytes, String> {
    let id = &si.id;
    let err = |e: &dyn std::fmt::Display| e.to_string();
    let open = |name: &str| dir.open(name).map_err(|e| err(&e));
    let (mut doc, mut pos, mut pay) = (Vec::new(), Vec::new(), Vec::new());
    let mut groups = Vec::with_capacity(markers.len());
    for (marker, suffix) in markers {
        let suffix = suffix.clone();
        let stem = marker
            .rsplit_once('.')
            .map_or(marker.as_str(), |(stem, _)| stem);
        let file = |ext: &str| format!("{stem}.{ext}");
        let has = |ext: &str| si.files.contains(&file(ext));
        let offset = |v: &Vec<u8>| u64::try_from(v.len()).unwrap_or(u64::MAX);
        let base = [offset(&doc), offset(&pos), offset(&pay)];
        let format = per_field_postings::format_of_suffix(&suffix);
        if let Some(Some(reencoded)) = field_infos
            .map(|fi| {
                // A format keeping its own postings (`RAMOnly`): read as the
                // block tree and postings it re-encodes to.
                let dictionary = per_field_postings::dictionary_extensions(format)
                    .iter()
                    .map(|ext| {
                        if has(ext) {
                            open(&file(ext))
                        } else {
                            Ok(lucene_store::directory::Input::Owned(Vec::new()))
                        }
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                let group = per_field_postings::GroupFiles {
                    suffix: &suffix,
                    dictionary: dictionary.iter().map(|d| &**d).collect(),
                    doc: None,
                    pos: None,
                    pay: None,
                };
                per_field_postings::reencode_group(&group, fi, id, si.doc_count)
                    .transpose()
                    .map_err(|e| err(&e))
            })
            .transpose()?
        {
            doc.extend_from_slice(&reencoded.doc);
            pos.extend_from_slice(&reencoded.pos);
            pay.extend_from_slice(&reencoded.pay);
            groups.push(PostingsGroupBytes {
                dictionary: [reencoded.tim, reencoded.tip, reencoded.tmd]
                    .into_iter()
                    .map(lucene_store::directory::Input::Owned)
                    .collect(),
                suffix,
                base,
                block_tree: true,
            });
            continue;
        }
        if has("doc") {
            let bytes = open(&file("doc"))?;
            DocInput::open(&bytes, id, &suffix).map_err(|e| err(&e))?;
            doc.extend_from_slice(&bytes);
        }
        if has("pos") {
            let bytes = open(&file("pos"))?;
            postings::PosInput::open(&bytes, id, &suffix).map_err(|e| err(&e))?;
            pos.extend_from_slice(&bytes);
        }
        if has("pay") {
            let bytes = open(&file("pay"))?;
            postings::PayInput::open(&bytes, id, &suffix).map_err(|e| err(&e))?;
            pay.extend_from_slice(&bytes);
        }
        let dictionary = per_field_postings::dictionary_extensions(format)
            .iter()
            .map(|ext| {
                if has(ext) || per_field_postings::dictionary_file_required(format, ext) {
                    open(&file(ext))
                } else {
                    Ok(lucene_store::directory::Input::Owned(Vec::new()))
                }
            })
            .collect::<Result<Vec<_>, String>>()?;
        groups.push(PostingsGroupBytes {
            dictionary,
            suffix,
            base,
            block_tree: false,
        });
    }
    let owned = |bytes: Vec<u8>| {
        (!bytes.is_empty()).then_some(lucene_store::directory::Input::Owned(bytes))
    };
    let empty = || lucene_store::directory::Input::Owned(Vec::new());
    Ok(PostingsFileBytes {
        // Unused: [`PostingsFileBytes::group_handles`] reads `groups`.
        tim: empty(),
        tip: empty(),
        tmd: empty(),
        doc: owned(doc),
        pos: owned(pos),
        pay: owned(pay),
        segment_suffix: String::new(),
        groups,
    })
}

/// The postings codec suffix embedded in a sub-file's own name.
fn suffix_of(file_name: &str, segment_name: &str, ext: &str) -> String {
    let stem = file_name.strip_suffix(ext).unwrap_or(file_name);
    // A loose file is `<segment>_<suffix>.ext` or `<segment>.ext`; a compound
    // member drops the segment name (`_<suffix>.ext`, `.ext`).
    let rest = stem.strip_prefix(segment_name).unwrap_or(stem);
    rest.strip_prefix('_').unwrap_or(rest).to_string()
}

impl PostingsFileBytes {
    fn handles(
        &self,
        commit: &SegmentCommitInfo,
        field_infos: &FieldInfos,
        si: &SegmentInfo,
    ) -> Result<PostingsHandles<'_>, String> {
        if !self.groups.is_empty() {
            return self.group_handles(commit, field_infos, si);
        }
        let fields = blocktree::open(
            &self.tim,
            &self.tip,
            &self.tmd,
            field_infos,
            &commit.segment_id,
            &self.segment_suffix,
            si.doc_count,
        )
        .map_err(|e| e.to_string())?;
        let doc_in = self
            .doc
            .as_ref()
            .map(|b| DocInput::open(b, &commit.segment_id, &self.segment_suffix))
            .transpose()
            .map_err(|e| e.to_string())?;
        let pos_in = self
            .pos
            .as_ref()
            .map(|b| postings::PosInput::open(b, &commit.segment_id, &self.segment_suffix))
            .transpose()
            .map_err(|e| e.to_string())?;
        let pay_in = self
            .pay
            .as_ref()
            .map(|b| postings::PayInput::open(b, &commit.segment_id, &self.segment_suffix))
            .transpose()
            .map_err(|e| e.to_string())?;
        Ok(PostingsHandles {
            fields,
            doc_in,
            pos_in,
            pay_in,
        })
    }

    /// [`Self::handles`] for a multi-format segment: every group's
    /// dictionary combined into one, decoding at its group's offsets into
    /// the concatenated, already-validated `.doc`/`.pos`/`.pay`.
    fn group_handles(
        &self,
        commit: &SegmentCommitInfo,
        field_infos: &FieldInfos,
        si: &SegmentInfo,
    ) -> Result<PostingsHandles<'_>, String> {
        let mut opened = Vec::with_capacity(self.groups.len());
        for g in &self.groups {
            let dictionary: Vec<&[u8]> = g.dictionary.iter().map(|d| &**d).collect();
            let rewritten = if !g.block_tree
                && per_field_postings::is_foreign_format(per_field_postings::format_of_suffix(
                    &g.suffix,
                )) {
                Some(
                    per_field_postings::rewrite_group_as_block_tree(
                        &per_field_postings::GroupFiles {
                            suffix: &g.suffix,
                            dictionary: dictionary.clone(),
                            doc: None,
                            pos: None,
                            pay: None,
                        },
                        field_infos,
                        &commit.segment_id,
                        si.doc_count,
                    )
                    .map_err(|e| e.to_string())?,
                )
            } else {
                None
            };
            let (tim, tip, tmd) = match &rewritten {
                Some((tim, tip, tmd)) => (&tim[..], &tip[..], &tmd[..]),
                None => match dictionary[..] {
                    [tim, tip, tmd] => (tim, tip, tmd),
                    _ => return Err(format!("postings group {} is incomplete", g.suffix)),
                },
            };
            let fields = blocktree::open(
                tim,
                tip,
                tmd,
                field_infos,
                &commit.segment_id,
                &g.suffix,
                si.doc_count,
            )
            .map_err(|e| e.to_string())?;
            opened.push((fields, g.base));
        }
        Ok(PostingsHandles {
            fields: blocktree::BlockTreeFields::combine(opened).map_err(|e| e.to_string())?,
            doc_in: self.doc.as_deref().map(DocInput::validated),
            pos_in: self.pos.as_deref().map(postings::PosInput::validated),
            pay_in: self.pay.as_deref().map(postings::PayInput::validated),
        })
    }
}

/// A term's fully decoded postings: the reference the cursor walks are
/// checked against. `freqs` is empty for a field without frequencies (every
/// frequency is then `1`).
struct TermDocs<'a> {
    docs: &'a [i32],
    freqs: &'a [i32],
}

impl TermDocs<'_> {
    /// The index of the first document `>= target`.
    fn first_at_or_after(&self, target: i32) -> usize {
        self.docs.partition_point(|&d| d < target)
    }

    fn doc(&self, i: usize) -> i32 {
        self.docs.get(i).copied().unwrap_or(postings::NO_MORE_DOCS)
    }

    fn freq(&self, i: usize) -> i32 {
        self.freqs.get(i).copied().unwrap_or(1)
    }
}

/// `CheckIndex.checkDocIDRuns`: every document in `[doc, docIDRunEnd())`
/// must be the next ones returned, and the runs end at the last document.
fn check_doc_id_runs(cursor: &mut postings::LazyDocsCursor<'_>) -> Result<(), String> {
    let mut prev_doc = -1i32;
    let mut run_end = 0i32;
    loop {
        let doc = cursor.next_doc().map_err(|e| e.to_string())?;
        if doc == postings::NO_MORE_DOCS {
            break;
        }
        let next = prev_doc.saturating_add(1);
        if next < run_end && doc != next {
            return Err(format!(
                "Run end is {run_end} but next doc after {prev_doc} is {doc}"
            ));
        }
        let new_run_end = cursor.doc_id_run_end();
        if new_run_end <= doc {
            return Err(format!("Run end {new_run_end} is <= doc ID {doc}"));
        }
        run_end = run_end.max(new_run_end);
        prev_doc = doc;
    }
    if run_end != prev_doc.saturating_add(1) {
        return Err(format!("Run end is {run_end} but last doc is {prev_doc}"));
    }
    Ok(())
}

/// `ImpactsEnum.getImpacts()` of the Lucene104 `BlockImpactsEnum`, as
/// `(getDocIdUpTo(level), getImpacts(level))` per level: level 0 is the
/// current block's (a tail block, whose extent is unknown, answers
/// `(MAX_VALUE, 1)`), level 1 the current level-1 span's when the term has
/// one; a field without frequencies has the one level `(1, 1)` up to
/// `NO_MORE_DOCS`.
fn impacts_levels(
    cursor: &mut postings::LazyDocsCursor<'_>,
    has_freqs: bool,
) -> Vec<(i32, Vec<postings::Impact>)> {
    let dummy = |freq| vec![postings::Impact { freq, norm: 1 }];
    if !has_freqs {
        return vec![(postings::NO_MORE_DOCS, dummy(1))];
    }
    let level0_up_to = cursor.level0_last_doc_id();
    let level0 = if level0_up_to == postings::NO_MORE_DOCS {
        dummy(i32::MAX)
    } else {
        cursor.level0_impacts().to_vec()
    };
    let mut levels = vec![(level0_up_to, level0)];
    let level1_up_to = cursor.level1_last_doc_id();
    if level1_up_to != postings::NO_MORE_DOCS {
        levels.push((level1_up_to, cursor.level1_impacts().to_vec()));
    }
    levels
}

/// Java: `CheckIndex.checkImpacts(impacts, lastTarget)`, over owned impact
/// snapshots. Shared by this index checker and the test framework's asserting
/// impacts iterator. Equal level bounds are allowed.
///
/// Ported as written, including that the within-level ordering loop never
/// advances its "previous" impact, so it compares every impact with the
/// first one rather than with its predecessor. The cross-level loop advances
/// the upper impact at most once for each lower impact, without retrying it.
/// Java copies mutable frequency/norm buffers before comparing levels; these
/// snapshots already own their immutable pairs.
///
/// # Errors
/// An empty level list, insufficient or decreasing bounds, invalid frequency
/// or norm, ordering violation, or exhausted cross-level comparison returns
/// Java's diagnostic message.
pub fn check_impacts(
    levels: &[(i32, Vec<postings::Impact>)],
    last_target: i32,
) -> Result<(), String> {
    let Some((doc_id_up_to0, _)) = levels.first() else {
        return Err("The number of impact levels must be >= 1, got 0".to_string());
    };
    if *doc_id_up_to0 < last_target {
        return Err(format!(
            "getDocIdUpTo returned {doc_id_up_to0} on level 0, which is less than the target \
             {last_target}"
        ));
    }
    for (previous_level, pair) in levels.windows(2).enumerate() {
        if let [(previous, _), (up_to, _)] = pair {
            if up_to < previous {
                return Err(format!(
                    "Decreasing return for getDocIdUpTo: level {previous_level} returned \
                     {previous} but level {} returned {up_to} for target {last_target}",
                    previous_level.saturating_add(1)
                ));
            }
        }
    }
    for (level, (_, impacts)) in levels.iter().enumerate() {
        let Some(first) = impacts.first() else {
            return Err(format!("Got empty list of impacts on level {level}"));
        };
        if first.freq < 1 {
            return Err(format!("First impact had a freq <= 0: {}", first.freq));
        }
        if first.norm == 0 {
            return Err(format!("First impact had a norm == 0: {}", first.norm));
        }
        let (prev_freq, prev_norm) = (first.freq, first.norm);
        for impact in impacts.iter().skip(1) {
            if impact.freq <= prev_freq || (impact.norm as u64) <= (prev_norm as u64) {
                return Err(format!(
                    "Impacts are not ordered or contain dups, got ({prev_freq},{prev_norm}) \
                     then ({},{})",
                    impact.freq, impact.norm
                ));
            }
        }
        if level > 0 {
            // Every impact of the level below must be matched by one of this
            // level's that scores at least as well.
            let prev = &levels[level.saturating_sub(1)].1;
            let mut index = 1usize;
            let (mut freq, mut norm) = (first.freq, first.norm as u64);
            for p in prev {
                if p.freq <= freq && (p.norm as u64) >= norm {
                    continue;
                }
                let Some(next) = impacts.get(index) else {
                    let freqs: Vec<_> = impacts.iter().map(|impact| impact.freq).collect();
                    let norms: Vec<_> = impacts.iter().map(|impact| impact.norm).collect();
                    return Err(format!(
                        "Found impact ({},{}) on level {} but no impact on level {level} \
                         triggers a better score: freqs={freqs:?} norms={norms:?}",
                        p.freq,
                        p.norm,
                        level.saturating_sub(1)
                    ));
                };
                (freq, norm) = (next.freq, next.norm as u64);
                index = index.saturating_add(1);
            }
        }
    }
    Ok(())
}

/// The largest frequency the impacts allow up to `max`: that of the first
/// level reaching it, `MAX_VALUE` when none does.
fn max_freq_up_to(levels: &[(i32, Vec<postings::Impact>)], max: i32) -> i32 {
    levels
        .iter()
        .find(|(up_to, _)| *up_to >= max)
        .and_then(|(_, impacts)| impacts.last())
        .map_or(i32::MAX, |i| i.freq)
}

/// `String.hashCode()`, which Java's impacts walk derives its targets from.
fn java_string_hash(s: &str) -> i32 {
    s.encode_utf16()
        .fold(0i32, |h, c| h.wrapping_mul(31).wrapping_add(i32::from(c)))
}

/// `checkFields`' impacts walks for one term, each against `reference`: the
/// slow level's walk of every document (`nextDoc`, then `advanceShallow` and
/// [`check_impacts`] whenever the document passes the last level-0 extent,
/// no frequency above the block's largest), and the walk that alternates
/// `nextDoc` and `advance` to targets hashed from the field name, calling
/// `advanceShallow` ahead of the cursor. `open` gives a fresh cursor.
fn check_term_impacts<'d>(
    field: &str,
    has_freqs: bool,
    reference: &TermDocs<'_>,
    mut open: impl FnMut() -> Result<postings::LazyDocsCursor<'d>, String>,
) -> Result<(), String> {
    const NO_MORE: i32 = postings::NO_MORE_DOCS;
    let freq_of = |c: &postings::LazyDocsCursor<'_>| c.freq().unwrap_or(1);

    let mut cursor = open()?;
    let (mut max, mut max_freq) = (-1i32, 0i32);
    for i in 0usize.. {
        let doc = cursor.next_doc().map_err(|e| e.to_string())?;
        if doc != reference.doc(i) {
            return Err(format!(
                "Wrong next doc: {doc}, expected {}",
                reference.doc(i)
            ));
        }
        if doc == NO_MORE {
            break;
        }
        if freq_of(&cursor) != reference.freq(i) {
            return Err(format!(
                "Wrong freq, expected {}, but got {}",
                reference.freq(i),
                freq_of(&cursor)
            ));
        }
        if doc > max {
            cursor.advance_shallow(doc).map_err(|e| e.to_string())?;
            let levels = impacts_levels(&mut cursor, has_freqs);
            check_impacts(&levels, doc)?;
            max = levels[0].0;
            max_freq = levels[0].1.last().map_or(0, |i| i.freq);
        }
        if freq_of(&cursor) > max_freq {
            return Err(format!(
                "freq {} is greater than the max freq according to impacts {max_freq}",
                freq_of(&cursor)
            ));
        }
    }

    let hash = java_string_hash(field);
    let mut cursor = open()?;
    let (mut max, mut max_freq) = (-1i32, 0i32);
    let mut doc = -1i32;
    loop {
        let (advance, target) = if hash.wrapping_add(doc) & 1 == 1 {
            (false, doc.wrapping_add(1))
        } else {
            let hashed = hash.wrapping_mul(31).wrapping_add(doc) & 0x1ff;
            let delta = hashed.wrapping_add(1).min(NO_MORE.wrapping_sub(doc));
            (true, doc.wrapping_add(delta))
        };
        if target > max && target.rem_euclid(2) == 1 {
            let hashed = hash.wrapping_mul(31).wrapping_add(target) & 0x1ff;
            max = target.wrapping_add(hashed.min(NO_MORE.wrapping_sub(target)));
            cursor.advance_shallow(target).map_err(|e| e.to_string())?;
            let levels = impacts_levels(&mut cursor, has_freqs);
            check_impacts(&levels, doc)?;
            max_freq = max_freq_up_to(&levels, max);
        }
        doc = if advance {
            cursor.advance(target)
        } else {
            cursor.next_doc()
        }
        .map_err(|e| e.to_string())?;
        let i = reference.first_at_or_after(target);
        if doc != reference.doc(i) {
            return Err(format!(
                "Impacts do not advance to the same document as postings for target {target}, \
                 postings: {}, impacts: {doc}",
                reference.doc(i)
            ));
        }
        if doc == NO_MORE {
            break;
        }
        if freq_of(&cursor) != reference.freq(i) {
            return Err(format!(
                "Wrong freq, expected {}, but got {}",
                reference.freq(i),
                freq_of(&cursor)
            ));
        }
        if doc >= max {
            // Java hashes the document reached here (`31 * hash + doc`),
            // not the target the advance asked for.
            let hashed = hash.wrapping_mul(31).wrapping_add(doc) & 0x1ff;
            max = doc.wrapping_add(hashed.min(NO_MORE.wrapping_sub(doc)));
            cursor.advance_shallow(doc).map_err(|e| e.to_string())?;
            let levels = impacts_levels(&mut cursor, has_freqs);
            check_impacts(&levels, doc)?;
            max_freq = max_freq_up_to(&levels, max);
        }
        if freq_of(&cursor) > max_freq {
            return Err(format!(
                "Term frequency {} is greater than the max freq according to impacts {max_freq}",
                freq_of(&cursor)
            ));
        }
    }
    Ok(())
}

/// `checkFields`' per-occurrence positional validation, extracted so it can
/// be unit-tested directly against synthetic [`postings::Position`] values.
///
/// That indirection is deliberate and worth stating: **byte corruption of a
/// real `.pos`/`.pay` cannot reach these predicates in this port.** Position
/// deltas are unsigned in both the packed and the vint encoding, and
/// `read_positions` restarts the accumulator at zero for every document, so
/// a decoded position is non-negative and non-decreasing *by construction*;
/// a corrupt bits-per-value header or block length is rejected by the
/// `for_util`/`SliceInput` layer first and surfaces as
/// `postings.terms_decode`. (The negative-control test
/// `no_single_byte_corruption_of_pos_or_pay_is_silently_accepted` pins that
/// property.) What these predicates actually guard is a *writer* emitting
/// bad positions or offsets -- the case Java's own message points at when it
/// suggests "the FixBrokenOffsets tool in Lucene's backward-codecs module"
/// -- so they are proved by feeding them exactly that.
fn check_occurrences(
    field_name: &str,
    term: &[u8],
    doc_id: i32,
    occurrences: &[postings::Position],
    has_offsets: bool,
    position_problems: &mut Vec<String>,
    offset_problems: &mut Vec<String>,
) {
    let mut last_pos = -1i32;
    let mut last_start_offset = 0i32;
    for occ in occurrences {
        if occ.position < 0 {
            position_problems.push(format!(
                "field {field_name:?} term {term:?} doc {doc_id}: position {} is out of bounds",
                occ.position
            ));
        } else if occ.position > MAX_POSITION {
            position_problems.push(format!(
                "field {field_name:?} term {term:?} doc {doc_id}: position {} > \
                 IndexWriter.MAX_POSITION={MAX_POSITION}",
                occ.position
            ));
        } else if occ.position < last_pos {
            position_problems.push(format!(
                "field {field_name:?} term {term:?} doc {doc_id}: position {} < previous \
                 position {last_pos}",
                occ.position
            ));
        }
        last_pos = occ.position;

        if has_offsets {
            if occ.start_offset < 0 {
                offset_problems.push(format!(
                    "field {field_name:?} term {term:?} doc {doc_id} pos {}: startOffset {} is \
                     out of bounds",
                    occ.position, occ.start_offset
                ));
            } else if occ.start_offset < last_start_offset {
                offset_problems.push(format!(
                    "field {field_name:?} term {term:?} doc {doc_id} pos {}: startOffset {} < \
                     previous startOffset {last_start_offset}",
                    occ.position, occ.start_offset
                ));
            }
            if occ.end_offset < 0 {
                offset_problems.push(format!(
                    "field {field_name:?} term {term:?} doc {doc_id} pos {}: endOffset {} is \
                     out of bounds",
                    occ.position, occ.end_offset
                ));
            } else if occ.end_offset < occ.start_offset {
                offset_problems.push(format!(
                    "field {field_name:?} term {term:?} doc {doc_id} pos {}: endOffset {} < \
                     startOffset {}",
                    occ.position, occ.end_offset, occ.start_offset
                ));
            }
            last_start_offset = occ.start_offset;
        }
    }
}

/// `checkFields`' per-field `visitedDocs`, by field name, for the fields
/// with norms (the only ones the terms-vs-norms cross-check reads).
type DocsWithTerms = std::collections::HashMap<String, lucene_util::fixed_bit_set::FixedBitSet>;

/// The automata `postings.intersect_agrees` walks every field with. They
/// depend on nothing but constants, so they are built once per
/// [`check_postings`] call rather than once per field. (Not once per
/// process: a compiled automaton carries a `RefCell` run cache and so is not
/// `Sync`, and segments are checked in parallel.)
struct IntersectAutomata {
    any_binary: Option<(
        lucene_util::automaton::CompiledAutomaton,
        lucene_util::automaton::ByteRunAutomaton,
    )>,
    regexp: Option<(
        lucene_codecs::regexp::RegexpPattern,
        lucene_codecs::automaton::ByteDfa,
    )>,
}

impl IntersectAutomata {
    fn build() -> Self {
        IntersectAutomata {
            any_binary: any_binary_a_to_e_any_binary(),
            regexp: lucene_codecs::regexp::RegexpPattern::new(b".*[a-e].*")
                .ok()
                .and_then(|p| p.to_dfa().map(|d| (p, d))),
        }
    }
}

/// `checkTermsIntersect`'s first automaton, `concatenate(makeAnyBinary(),
/// makeCharRange('a', 'e'), makeAnyBinary())` determinized, as the walk's
/// `CompiledAutomaton(automaton, false, true, true)` and the scan's
/// `ByteRunAutomaton(automaton, true)`. `None` only if the automaton API
/// refused it, which it cannot for this one.
fn any_binary_a_to_e_any_binary() -> Option<(
    lucene_util::automaton::CompiledAutomaton,
    lucene_util::automaton::ByteRunAutomaton,
)> {
    use lucene_util::automaton::{
        automata, operations, ByteRunAutomaton, CompiledAutomaton, DEFAULT_DETERMINIZE_WORK_LIMIT,
    };
    let any = automata::make_any_binary();
    let a_to_e = automata::make_char_range(i32::from(b'a'), i32::from(b'e'));
    let automaton = operations::concatenate(&[&any, &a_to_e, &any]);
    let automaton = operations::determinize(&automaton, DEFAULT_DETERMINIZE_WORK_LIMIT).ok()?;
    let compiled = CompiledAutomaton::with_options(&automaton, false, true, true).ok()?;
    let run = ByteRunAutomaton::new(&automaton, true).ok()?;
    Some((compiled, run))
}

/// One `checkTermsIntersect` round: walks the field's terms linearly and the
/// pruning `intersect` iterator side by side, reporting the first place they
/// disagree. Neither side is collected, so the memory cost is one term.
fn compare_intersect_with_scan(
    field_name: &str,
    field_terms: &blocktree::FieldTerms,
    label: &str,
    matches: impl Fn(&[u8]) -> bool,
    mut intersected: impl Iterator<Item = blocktree::Result<(Vec<u8>, blocktree::TermStats)>>,
    problems: &mut Vec<String>,
) {
    let mut scan = field_terms.iter();
    loop {
        let expected = loop {
            // The scan compares *term bytes* with the intersect walk's, so
            // it takes the stats-free `next()` Java's own `TermsEnum` has --
            // otherwise this check would decode every term's postings
            // metadata to throw it away (item 21).
            match scan.try_next_term() {
                Ok(Some(t)) if matches(t) => break Some(t.to_vec()),
                Ok(Some(_)) => continue,
                Ok(None) => break None,
                Err(e) => {
                    problems.push(format!(
                        "field {field_name:?}: {label}: the linear scan failed: {e}"
                    ));
                    return;
                }
            }
        };
        let actual = match intersected.next().transpose() {
            Ok(next) => next.map(|(t, _)| t),
            Err(e) => {
                problems.push(format!(
                    "field {field_name:?}: {label}: the intersect walk failed: {e}"
                ));
                return;
            }
        };
        match (expected, actual) {
            (None, None) => return,
            (Some(e), Some(a)) if e == a => {}
            (e, a) => {
                problems.push(format!(
                    "field {field_name:?}: {label}: intersect yields {a:?} where a linear scan \
                     matches {e:?}"
                ));
                return;
            }
        }
    }
}

/// `CheckIndex.testDocValues`' core: for every field `.fnm` says has
/// doc-values, decode **every** document's value out of `.dvd` and check the
/// per-field invariants Java checks -- ordinals inside the terms
/// dictionary's range, sorted-set ordinals strictly increasing within a doc,
/// and the number of docs that actually have a value matching what `.dvm`
/// recorded.
///
/// Without this, nothing during a check would read a single doc-values
/// *value*, so a corrupted `.dvd` payload would pass a clean `CheckIndex` run
/// -- the "verifier that does not check something silently passes corrupt
/// segments" failure mode.
///
/// Skipped (not failed) for a segment where no field claims doc-values. A field
/// that claims them while the segment lacks `.dvd`/`.dvm` is already reported
/// by [`check_field_flags_vs_files`]'s `fnm.doc_values_vs_files`, so this
/// function returns quietly in that case rather than duplicating the failure.
fn check_doc_values(
    dir: &dyn Directory,
    commit: &SegmentCommitInfo,
    si: &SegmentInfo,
    field_infos: &FieldInfos,
    checks: &mut Vec<Check>,
) {
    // `testDocValues` walks `reader.getFieldInfos()` -- the *current*
    // generation's field infos -- and reads every field through
    // `SegmentDocValuesProducer`, which hands a field with an update
    // generation to that generation's producer and every other field to the
    // base one of its own `PerFieldDocValuesFormat` instance. Reading only
    // the first `.dvm` the `.si` lists decoded one format's fields and none
    // of an update's.
    let current = if commit.field_infos_gen == -1 {
        None
    } else {
        match crate::field_updates::read_current_field_infos(dir, commit, &si.files) {
            Ok(current) => Some(current),
            Err(e) => {
                checks.push(Check::fail("doc_values.open", e.to_string()));
                skip_families(
                    checks,
                    &[
                        "doc_values.values_decode",
                        "doc_values.skipper",
                        "doc_values.ords_dense",
                        "doc_values.terms_sorted",
                    ],
                    "doc_values.open",
                );
                return;
            }
        }
    };
    let lookup = current.as_ref().unwrap_or(field_infos);
    let dv_fields: Vec<&field_infos::FieldInfo> = lookup
        .fields
        .iter()
        .filter(|f| f.doc_values_type != field_infos::DocValuesType::None)
        .collect();
    if dv_fields.is_empty() {
        return;
    }
    // Only an update generation can hold a column without a base `.dvm`.
    if dv_fields.iter().all(|f| f.doc_values_gen == -1)
        && !si.files.iter().any(|f| f.ends_with(".dvm"))
    {
        return; // already reported by fnm.doc_values_vs_files
    }

    let result = (|| -> Result<Vec<Check>, String> {
        // One opened column per `(generation, format instance)`, shared by
        // every field that lives in it.
        let mut columns: Vec<((i64, String), DocValuesColumn)> = Vec::new();
        let mut out = Vec::new();
        for fi in &dv_fields {
            let key = column_key(fi);
            let at = match columns.iter().position(|(k, _)| *k == key) {
                Some(at) => at,
                None => {
                    // `.dvs` is opened when *any* field read from this
                    // column declares a skip index, not just the one that
                    // happens to open it: the column is shared.
                    let wants_dvs = dv_fields.iter().any(|g| {
                        g.doc_values_skip_index_type != field_infos::DocValuesSkipIndexType::None
                            && column_key(g) == key
                    });
                    let column =
                        open_field_column(dir, commit, si, field_infos, fi, wants_dvs, true)?;
                    columns.push((key, column));
                    columns.len().saturating_sub(1)
                }
            };
            let DocValuesColumn {
                meta,
                dvd,
                dvs,
                suffix,
            } = &columns[at].1;
            let (meta, dvd, dvs, suffix) = (meta, &dvd[..], dvs.as_deref(), suffix.as_str());
            let name = &fi.name;
            let mut problems = Problems::new();
            let mut docs_with_value = 0i64;
            let mut expected_docs_with_value: Option<i64> = None;
            // `checkSortedDocValues`/`checkSortedSetDocValues`' ordinal
            // bookkeeping: which ordinals any document actually uses, and
            // the ordinal-to-term dictionary they index into.
            let mut seen_ords: Option<lucene_util::fixed_bit_set::FixedBitSet> = None;
            let mut ord_dict: Option<(i64, &lucene_codecs::terms_dict::TermsDictEntry)> = None;

            match fi.doc_values_type {
                field_infos::DocValuesType::Numeric => {
                    let Some(entry) = meta.numeric_entry(fi.number) else {
                        out.push(Check::fail(
                            format!("doc_values.entry_present:{name}"),
                            "field claims NUMERIC doc values in .fnm but has no .dvm entry",
                        ));
                        // The entry is how every value of this field is
                        // addressed, so without it the field's whole column
                        // goes unread -- and nothing else in the module reads
                        // `.dvd`.
                        skip_families(
                            &mut out,
                            &[&format!("doc_values.values_decode:{name}")],
                            &format!("doc_values.entry_present:{name}"),
                        );
                        continue;
                    };
                    // One stateful reader for the whole walk, as the norms
                    // check and the parent walk hold one: the free
                    // per-document functions re-walk a sparse field's
                    // `IndexedDISI` from its start on every call.
                    let mut reader = doc_values::NumericReader::new(dvd, entry);
                    for doc in 0..si.doc_count {
                        match reader.value(doc) {
                            Ok(Some(_)) => docs_with_value = docs_with_value.saturating_add(1),
                            Ok(None) => {}
                            Err(e) => {
                                problems.push(format!("docID={doc}: {e}"));
                                // A failed lookup can leave the cursor mid-block; the
                                // next document starts from a fresh one, as the free
                                // per-document functions always did.
                                reader = doc_values::NumericReader::new(dvd, entry);
                            }
                        }
                    }
                }
                field_infos::DocValuesType::Binary => {
                    let Some(entry) = meta.binary_entry(fi.number) else {
                        out.push(Check::fail(
                            format!("doc_values.entry_present:{name}"),
                            "field claims BINARY doc values in .fnm but has no .dvm entry",
                        ));
                        // The entry is how every value of this field is
                        // addressed, so without it the field's whole column
                        // goes unread -- and nothing else in the module reads
                        // `.dvd`.
                        skip_families(
                            &mut out,
                            &[&format!("doc_values.values_decode:{name}")],
                            &format!("doc_values.entry_present:{name}"),
                        );
                        continue;
                    };
                    expected_docs_with_value = Some(entry.num_docs_with_field as i64);
                    let mut reader = doc_values::BinaryReader::new(dvd, entry);
                    for doc in 0..si.doc_count {
                        match reader.value_ref(doc) {
                            Ok(Some(v)) => {
                                docs_with_value = docs_with_value.saturating_add(1);
                                let len = v.len() as i32;
                                if len < entry.min_length || len > entry.max_length {
                                    problems.push(format!(
                                        "docID={doc}: value length {len} outside .dvm's declared \
                                         [{}, {}]",
                                        entry.min_length, entry.max_length
                                    ));
                                }
                            }
                            Ok(None) => {}
                            Err(e) => {
                                problems.push(format!("docID={doc}: {e}"));
                                // A failed lookup can leave the cursor mid-block; the
                                // next document starts from a fresh one, as the free
                                // per-document functions always did.
                                reader = doc_values::BinaryReader::new(dvd, entry);
                            }
                        }
                    }
                }
                field_infos::DocValuesType::Sorted => {
                    let Some(entry) = meta.sorted_entry(fi.number) else {
                        out.push(Check::fail(
                            format!("doc_values.entry_present:{name}"),
                            "field claims SORTED doc values in .fnm but has no .dvm entry",
                        ));
                        // The entry is how every value of this field is
                        // addressed, so without it the field's whole column
                        // goes unread -- and nothing else in the module reads
                        // `.dvd`.
                        skip_families(
                            &mut out,
                            &[
                                &format!("doc_values.values_decode:{name}"),
                                &format!("doc_values.ords_dense:{name}"),
                                &format!("doc_values.terms_sorted:{name}"),
                            ],
                            &format!("doc_values.entry_present:{name}"),
                        );
                        continue;
                    };
                    let value_count = entry.terms.terms_dict_size;
                    let mut seen =
                        lucene_util::fixed_bit_set::FixedBitSet::new(value_count.max(0) as usize);
                    // `sorted_ord` is `numeric_value` over the entry's `ords`.
                    let mut reader = doc_values::NumericReader::new(dvd, &entry.ords);
                    for doc in 0..si.doc_count {
                        match reader.value(doc) {
                            Ok(Some(ord)) => {
                                docs_with_value = docs_with_value.saturating_add(1);
                                if ord < 0 || ord >= value_count {
                                    problems.push(format!(
                                        "docID={doc}: ordinal {ord} outside the terms \
                                         dictionary's 0..{value_count} range"
                                    ));
                                } else {
                                    // FBS: `seen` is
                                    // `FixedBitSet::new(value_count.max(0))` in
                                    // this same block, so `value_count` is its
                                    // `len()`.
                                    seen.set(ord as usize);
                                }
                            }
                            Ok(None) => {}
                            Err(e) => {
                                problems.push(format!("docID={doc}: {e}"));
                                // A failed lookup can leave the cursor mid-block; the
                                // next document starts from a fresh one, as the free
                                // per-document functions always did.
                                reader = doc_values::NumericReader::new(dvd, &entry.ords);
                            }
                        }
                    }
                    seen_ords = Some(seen);
                    ord_dict = Some((value_count, &entry.terms));
                }
                field_infos::DocValuesType::SortedNumeric => {
                    let Some(entry) = meta.sorted_numeric_entry(fi.number) else {
                        out.push(Check::fail(
                            format!("doc_values.entry_present:{name}"),
                            "field claims SORTED_NUMERIC doc values in .fnm but has no .dvm entry",
                        ));
                        // The entry is how every value of this field is
                        // addressed, so without it the field's whole column
                        // goes unread -- and nothing else in the module reads
                        // `.dvd`.
                        skip_families(
                            &mut out,
                            &[&format!("doc_values.values_decode:{name}")],
                            &format!("doc_values.entry_present:{name}"),
                        );
                        continue;
                    };
                    expected_docs_with_value = Some(entry.num_docs_with_field as i64);
                    let mut reader = doc_values::SortedNumericReader::new(dvd, entry);
                    let mut values = Vec::new();
                    for doc in 0..si.doc_count {
                        match reader.values(doc, &mut values) {
                            Ok(()) => {
                                if !values.is_empty() {
                                    docs_with_value = docs_with_value.saturating_add(1);
                                }
                                // `CheckIndex.checkSortedNumericDocValues`:
                                // `if (value < previous) throw new
                                // CheckIndexException("values out of order: "
                                // + value + " < " + previous + " for doc: " +
                                // docID)`. A document's values are written
                                // ascending by
                                // `SortedNumericDocValuesWriter.finishCurrentDoc`,
                                // and `SortedNumericSelector.MIN`/`MAX` are
                                // literally the first and the last stored
                                // value -- so an unsorted column silently
                                // makes an index sort over it pick the wrong
                                // value, and nothing else here would notice.
                                for pair in values.windows(2) {
                                    if pair[1] < pair[0] {
                                        problems.push(format!(
                                            "docID={doc}: values out of order: {} < {}",
                                            pair[1], pair[0]
                                        ));
                                    }
                                }
                            }
                            Err(e) => {
                                problems.push(format!("docID={doc}: {e}"));
                                // A failed lookup can leave the cursor mid-block; the
                                // next document starts from a fresh one, as the free
                                // per-document functions always did.
                                reader = doc_values::SortedNumericReader::new(dvd, entry);
                            }
                        }
                    }
                }
                field_infos::DocValuesType::SortedSet => {
                    let Some(entry) = meta.sorted_set_entry(fi.number) else {
                        out.push(Check::fail(
                            format!("doc_values.entry_present:{name}"),
                            "field claims SORTED_SET doc values in .fnm but has no .dvm entry",
                        ));
                        // The entry is how every value of this field is
                        // addressed, so without it the field's whole column
                        // goes unread -- and nothing else in the module reads
                        // `.dvd`.
                        skip_families(
                            &mut out,
                            &[
                                &format!("doc_values.values_decode:{name}"),
                                &format!("doc_values.ords_dense:{name}"),
                                &format!("doc_values.terms_sorted:{name}"),
                            ],
                            &format!("doc_values.entry_present:{name}"),
                        );
                        continue;
                    };
                    match &entry.kind {
                        doc_values::SortedSetKind::Single(single) => {
                            let value_count = single.terms.terms_dict_size;
                            let mut seen = lucene_util::fixed_bit_set::FixedBitSet::new(
                                value_count.max(0) as usize,
                            );
                            let mut reader = doc_values::NumericReader::new(dvd, &single.ords);
                            for doc in 0..si.doc_count {
                                match reader.value(doc) {
                                    Ok(Some(ord)) => {
                                        docs_with_value = docs_with_value.saturating_add(1);
                                        if ord < 0 || ord >= value_count {
                                            problems.push(format!(
                                                "docID={doc}: ordinal {ord} outside the terms \
                                                 dictionary's 0..{value_count} range"
                                            ));
                                        } else {
                                            // FBS: `seen` is
                                            // `FixedBitSet::new(value_count.max(0))`
                                            // in this same block, so `value_count`
                                            // is its `len()`.
                                            seen.set(ord as usize);
                                        }
                                    }
                                    Ok(None) => {}
                                    Err(e) => {
                                        problems.push(format!("docID={doc}: {e}"));
                                        // A failed lookup can leave the cursor mid-block; the
                                        // next document starts from a fresh one, as the free
                                        // per-document functions always did.
                                        reader = doc_values::NumericReader::new(dvd, &single.ords);
                                    }
                                }
                            }
                            seen_ords = Some(seen);
                            ord_dict = Some((value_count, &single.terms));
                        }
                        doc_values::SortedSetKind::Multi { ords, terms } => {
                            let value_count = terms.terms_dict_size;
                            let mut seen = lucene_util::fixed_bit_set::FixedBitSet::new(
                                value_count.max(0) as usize,
                            );
                            expected_docs_with_value = Some(ords.num_docs_with_field as i64);
                            let mut reader = doc_values::SortedNumericReader::new(dvd, ords);
                            let mut values = Vec::new();
                            for doc in 0..si.doc_count {
                                match reader.values(doc, &mut values) {
                                    Ok(()) => {
                                        if values.is_empty() {
                                            continue;
                                        }
                                        docs_with_value = docs_with_value.saturating_add(1);
                                        // `SortedSetDocValues` ordinals are a
                                        // *set*: strictly increasing per doc.
                                        let mut prev = -1i64;
                                        for &ord in &values {
                                            if ord < 0 || ord >= value_count {
                                                problems.push(format!(
                                                    "docID={doc}: ordinal {ord} outside the terms \
                                                     dictionary's 0..{value_count} range"
                                                ));
                                            } else if ord <= prev {
                                                problems.push(format!(
                                                    "docID={doc}: ordinals are not strictly \
                                                     increasing ({ord} after {prev})"
                                                ));
                                            } else {
                                                // FBS: `seen` is
                                                // `FixedBitSet::new(value_count.max(0))`
                                                // in this same block, so
                                                // `value_count` is its `len()`.
                                                seen.set(ord as usize);
                                            }
                                            prev = ord;
                                        }
                                    }
                                    Err(e) => {
                                        problems.push(format!("docID={doc}: {e}"));
                                        // A failed lookup can leave the cursor mid-block; the
                                        // next document starts from a fresh one, as the free
                                        // per-document functions always did.
                                        reader = doc_values::SortedNumericReader::new(dvd, ords);
                                    }
                                }
                            }
                            seen_ords = Some(seen);
                            ord_dict = Some((value_count, terms));
                        }
                    }
                }
                field_infos::DocValuesType::None => unreachable!("filtered above"),
            }

            if let Some(expected) = expected_docs_with_value {
                if expected != docs_with_value {
                    problems.push(format!(
                        ".dvm records numDocsWithField={expected} but {docs_with_value} docs \
                         actually decode a value"
                    ));
                }
            }
            out.push(named_field_check(
                &format!("doc_values.values_decode:{name}"),
                &problems,
                si.doc_count as i64,
                "docs",
            ));

            // `checkDocValueSkipper(fi, dvReader.getSkipper(fi))`, run for
            // exactly the fields Java runs it for.
            if fi.doc_values_skip_index_type != field_infos::DocValuesSkipIndexType::None {
                let name = &fi.name;
                let mut skip_problems: Vec<String> = Vec::new();
                let mut intervals = 0i64;
                match (dvs, meta.skipper_meta(fi.number)) {
                    (Some(dvs), Some(skipper_meta)) => {
                        match doc_values::parse_skip_index(
                            dvs,
                            &commit.segment_id,
                            suffix,
                            skipper_meta,
                        ) {
                            Ok(index) => {
                                intervals = index.intervals.len() as i64;
                                skip_problems = check_doc_value_skipper(&index);
                            }
                            Err(e) => skip_problems.push(e.to_string()),
                        }
                    }
                    (None, _) => skip_problems.push(
                        "field declares a doc-values skip index but the segment has no \
                               .dvs file"
                            .to_string(),
                    ),
                    (_, None) => skip_problems.push(
                        "field declares a doc-values skip index in .fnm but has no .dvm \
                         skipper entry"
                            .to_string(),
                    ),
                }
                out.push(named_field_check(
                    &format!("doc_values.skipper:{name}"),
                    &skip_problems,
                    intervals,
                    "skip intervals",
                ));
            }

            if let (Some(seen), Some((value_count, terms))) = (seen_ords, ord_dict) {
                // `checkSortedDocValues`: `maxOrd` must actually be reached
                // and the ordinal space must have no holes -- an ordinal no
                // document uses is a term in the dictionary that nothing can
                // ever match, and `valueCount` is what every ordinal-based
                // comparator and facet counter sizes its arrays from.
                let mut density: Vec<String> = Vec::new();
                let used = seen.cardinality() as i64;
                if used != value_count {
                    density.push(format!(
                        ".dvm declares valueCount={value_count} but only {used} distinct \
                         ordinals are used by any document"
                    ));
                }
                out.push(named_field_check(
                    &format!("doc_values.ords_dense:{name}"),
                    &density,
                    value_count,
                    "ordinals",
                ));

                // `checkSortedDocValues`' `lookupOrd` walk: the
                // ordinal-to-term dictionary must be strictly increasing (it
                // is what makes ordinal comparison equivalent to term
                // comparison, which every `SortedDocValues` range query and
                // index sort relies on) and must hold exactly `valueCount`
                // terms.
                let mut order: Vec<String> = Vec::new();
                match lucene_codecs::terms_dict::decode_all_terms(dvd, terms) {
                    Ok(all) => {
                        if all.len() as i64 != value_count {
                            order.push(format!(
                                ".dvm declares valueCount={value_count} but the ordinal-to-term \
                                 dictionary decodes {} terms",
                                all.len()
                            ));
                        }
                        for pair in all.windows(2) {
                            if pair[1] <= pair[0] {
                                order.push(format!(
                                    "terms are not strictly increasing: {:?} does not come \
                                     after {:?}",
                                    pair[1], pair[0]
                                ));
                                break;
                            }
                        }
                    }
                    Err(e) => order.push(e.to_string()),
                }
                out.push(named_field_check(
                    &format!("doc_values.terms_sorted:{name}"),
                    &order,
                    value_count,
                    "ordinals",
                ));
            }
        }
        Ok(out)
    })();

    match result {
        Ok(cs) => checks.extend(cs),
        Err(e) => {
            checks.push(Check::fail("doc_values.open", e));
            skip_families(
                checks,
                &[
                    "doc_values.values_decode",
                    "doc_values.skipper",
                    "doc_values.ords_dense",
                    "doc_values.terms_sorted",
                ],
                "doc_values.open",
            );
        }
    }
}

/// One doc-values column as [`check_doc_values`] reads it: a base
/// `PerFieldDocValuesFormat` instance's `.dvm`/`.dvd`(/`.dvs`), or one update
/// generation's, with the segment suffix every one of its headers carries.
struct DocValuesColumn {
    meta: doc_values::DocValuesMeta,
    dvd: lucene_store::directory::Input,
    dvs: Option<lucene_store::directory::Input>,
    suffix: String,
}

/// Which column a field is read from: its update generation (`-1`: the
/// base) and its `PerFieldDocValuesFormat` component (`Lucene90_1`), empty
/// for a field whose `.fnm` entry carries no format attributes.
fn column_key(fi: &field_infos::FieldInfo) -> (i64, String) {
    (
        fi.doc_values_gen,
        crate::field_updates::field_per_field_component(fi).unwrap_or_default(),
    )
}

/// `SegmentDocValuesProducer`'s choice of producer for one field, opened:
/// the update generation `docValuesGen` names (whose `.dvm` describes that
/// one field, so it is parsed against it alone, as Java hands the
/// generation's producer a one-field `FieldInfos`), or the base column of the
/// format instance the field's attributes name. A base field with no
/// attributes falls back to the first `.dvm` the `.si` lists, the one column
/// a single-format segment has.
///
/// `wants_dvs` opens the column's `.dvs` skip-index file: true when any
/// field that reads this column declares a skip index. `verify_dvd` checks
/// the `.dvd`'s header and footer, which the value checks do once and the
/// sort and soft-deletes checks, re-reading the same column, do not repeat.
fn open_field_column(
    dir: &dyn Directory,
    commit: &SegmentCommitInfo,
    si: &SegmentInfo,
    base_infos: &FieldInfos,
    fi: &field_infos::FieldInfo,
    wants_dvs: bool,
    verify_dvd: bool,
) -> Result<DocValuesColumn, String> {
    let component = crate::field_updates::field_per_field_component(fi);
    let (dvm_name, dvd_name, dvs_name, dvp_name, suffix, only);
    if fi.doc_values_gen != -1 {
        let component = component.ok_or_else(|| {
            format!(
                "field {:?} has doc-values generation {} but no PerFieldDocValuesFormat \
                 attributes",
                fi.name, fi.doc_values_gen
            )
        })?;
        let gen = fi.doc_values_gen;
        let name = |ext| {
            crate::field_updates::generation_file_name(&commit.segment_name, gen, &component, ext)
        };
        dvm_name = name("dvm");
        dvd_name = name("dvd");
        dvs_name = Some(name("dvs"));
        dvp_name = Some(name("dvp"));
        suffix = crate::field_updates::generation_segment_suffix(gen, &component);
        only = Some(FieldInfos {
            fields: vec![fi.clone()],
        });
    } else {
        let base = |ext: &str| -> Option<String> {
            match &component {
                Some(c) => {
                    let want = format!("{}_{c}.{ext}", commit.segment_name);
                    si.files.iter().find(|f| **f == want).cloned()
                }
                None => si
                    .files
                    .iter()
                    .find(|f| f.ends_with(&format!(".{ext}")))
                    .cloned(),
            }
        };
        let missing = || {
            format!(
                "field {:?}'s doc-values format instance {:?} has no .dvm/.dvd in the segment",
                fi.name,
                component.as_deref().unwrap_or("")
            )
        };
        dvm_name = base("dvm").ok_or_else(missing)?;
        dvd_name = base("dvd").ok_or_else(missing)?;
        dvs_name = base("dvs");
        dvp_name = base("dvp");
        // The per-format suffix is embedded in the file name, exactly as
        // `check_postings_term_stats` derives the postings one.
        suffix = if dvm_name == format!("{}.dvm", commit.segment_name) {
            String::new()
        } else {
            dvm_name
                .strip_prefix(&format!("{}_", commit.segment_name))
                .or_else(|| dvm_name.strip_prefix('_'))
                .and_then(|s| s.strip_suffix(".dvm"))
                .unwrap_or_default()
                .to_string()
        };
        only = None;
    }
    let dvm = dir
        .open(&dvm_name)
        .map_err(|e| format!("{dvm_name}: {e}"))?;
    let dvd = dir
        .open(&dvd_name)
        .map_err(|e| format!("{dvd_name}: {e}"))?;
    let (version, mut meta) = doc_values::parse_meta(
        &dvm,
        &commit.segment_id,
        &suffix,
        only.as_ref().unwrap_or(base_infos),
    )
    .map_err(|e| e.to_string())?;
    if verify_dvd {
        doc_values::check_data_header_footer(&dvd, &commit.segment_id, &suffix)
            .map_err(|e| e.to_string())?;
    }
    if doc_values::has_disi_file(version) {
        let name = dvp_name.ok_or_else(|| {
            format!("doc values are format {version} but the segment has no .dvp file")
        })?;
        let dvp = dir.open(&name).map_err(|e| format!("{name}: {e}"))?;
        if verify_dvd {
            doc_values::check_disi_header_footer(&dvp, &commit.segment_id, &suffix, version)
                .map_err(|e| e.to_string())?;
        }
        meta.attach_disi(doc_values::DisiBytes::new(std::sync::Arc::new(dvp)));
    }
    // `.dvs` is a separate file, present only when some field asked for a
    // skip index (`Lucene90DocValuesConsumer`'s
    // `VERSION_SKIPPER_SEPARATE_FILE`); its absence is reported per field.
    let dvs = match dvs_name {
        Some(name) if wants_dvs => Some(dir.open(&name).map_err(|e| format!("{name}: {e}"))?),
        _ => None,
    };
    Ok(DocValuesColumn {
        meta,
        dvd,
        dvs,
        suffix,
    })
}

/// `CheckIndex.checkDocValueSkipper`: the `.dvs` skip index's own semantic
/// invariants, which no amount of CRC checking can reach.
///
/// A skipper is a *promise* about documents the reader will then not look at:
/// `DocValuesSkipper.advance` trusts every level's `[minDocID, maxDocID]` and
/// `[minValue, maxValue]` to skip whole subtrees. A skipper whose bounds are
/// too narrow silently drops matching documents from every range query that
/// uses it, and nothing else in `CheckIndex` would notice -- the per-document
/// values all still decode, because they are in `.dvd`, not here.
///
/// Java's checks, in order:
///
/// - a fresh skipper reports `maxDocID(0) == -1` (nothing consumed yet);
/// - the global value range is not inverted, and `maxValueCount` is neither
///   below `-1` (the "written by a codec too old to record it" sentinel) nor
///   non-zero for a field with no documents;
/// - walking interval by interval (`advance(maxDocID(0) + 1)`), each
///   interval starts at or after the doc it was asked for, every level's doc
///   range is not inverted, and every level's value range is both
///   non-inverted and *nested inside* the global one;
/// - the level-0 document counts add up to the global `docCount`.
///
/// Returns one message per violation, in `named_field_check`'s shape.
fn check_doc_value_skipper(index: &doc_values::DocValuesSkipIndex) -> Vec<String> {
    let mut problems = Vec::new();
    // Java's `assert skipper.maxDocID(0) == -1` on a fresh skipper is not
    // reproduced: Java is handed a `DocValuesSkipper` the codec produced and
    // that a caller may already have advanced, whereas this one is
    // constructed on the line above and `DocValuesSkipper::new` sets every
    // level's `max_doc_id` to `-1` unconditionally. The check could only ever
    // pass.
    let mut skipper = doc_values::DocValuesSkipper::new(index);

    if skipper.global_doc_count() > 0 && skipper.global_min_value() > skipper.global_max_value() {
        problems.push(format!(
            "inverted global value range: {} > {}",
            skipper.global_min_value(),
            skipper.global_max_value()
        ));
    }
    if skipper.max_value_count() < -1 {
        problems.push(format!(
            "invalid maxValueCount {}",
            skipper.max_value_count()
        ));
    }
    if skipper.global_doc_count() == 0 && skipper.max_value_count() != 0 {
        problems.push(format!(
            "maxValueCount is {} for a field with no documents",
            skipper.max_value_count()
        ));
    }

    let mut doc_count: i64 = 0;
    // Every iteration consumes at least one interval, so this is a bound, not
    // a heuristic -- and `check_index` must not be able to hang on a corrupt
    // file even where the skipper's own state machine would.
    for _ in 0..=index.intervals.len() {
        // Saturating, not because the overflow is reachable but because
        // clippy cannot see that it is not: `NO_MORE_DOCS` *is* `i32::MAX`, a
        // fresh skipper starts at `-1`, and `advance` leaves `max_doc_id(0)`
        // as either an on-disk interval bound or the sentinel -- which the
        // `break` below catches one line before the next increment. Rule 2 of
        // `docs/arithmetic-gate.md` (saturation unreachable).
        let doc = skipper.max_doc_id(0).saturating_add(1);
        skipper.advance(doc);
        if skipper.max_doc_id(0) == doc_values::NO_MORE_DOCS {
            break;
        }
        if skipper.min_doc_id(0) < doc {
            problems.push(format!(
                "interval starting at docID={} was reached by advancing to {doc}",
                skipper.min_doc_id(0)
            ));
        }
        for level in 0..skipper.num_levels() {
            if skipper.min_doc_id(level) > skipper.max_doc_id(level) {
                problems.push(format!(
                    "level {level}: inverted doc range {} > {}",
                    skipper.min_doc_id(level),
                    skipper.max_doc_id(level)
                ));
            }
            if skipper.global_min_value() > skipper.min_value(level) {
                problems.push(format!(
                    "level {level}: minValue {} is below the global minValue {}",
                    skipper.min_value(level),
                    skipper.global_min_value()
                ));
            }
            if skipper.global_max_value() < skipper.max_value(level) {
                problems.push(format!(
                    "level {level}: maxValue {} is above the global maxValue {}",
                    skipper.max_value(level),
                    skipper.global_max_value()
                ));
            }
            if skipper.min_value(level) > skipper.max_value(level) {
                problems.push(format!(
                    "level {level}: inverted value range {} > {}",
                    skipper.min_value(level),
                    skipper.max_value(level)
                ));
            }
        }
        doc_count = doc_count.saturating_add(i64::from(skipper.doc_count(0)));
    }
    if skipper.global_doc_count() as i64 != doc_count {
        problems.push(format!(
            ".dvm declares docCount={} but the intervals cover {doc_count} documents",
            skipper.global_doc_count()
        ));
    }
    problems
}

/// The base doc-values column `fi` is read from -- the
/// `PerFieldDocValuesFormat` instance its attributes name, as
/// `SegmentDocValuesProducer` hands a field without an update generation to
/// its own format's producer -- or, for a field without those attributes,
/// the first `.dvm` the `.si` lists. `None` when the segment has no
/// doc-values files at all. Reading only the first `.dvm` the `.si` lists
/// failed a sort or soft-deletes field that lives in a second format
/// instance.
///
/// Shared by the index-sort check, the soft-deletes check and the writer's
/// block tests, all of which read actual per-doc values. The `.dvd` header
/// and footer are not re-verified here: `check_doc_values` does that.
pub(crate) fn open_base_column(
    dir: &dyn Directory,
    commit: &SegmentCommitInfo,
    si: &SegmentInfo,
    field_infos: &FieldInfos,
    fi: &field_infos::FieldInfo,
) -> Option<Result<(doc_values::DocValuesMeta, lucene_store::directory::Input), String>> {
    if !si.files.iter().any(|f| f.ends_with(".dvm"))
        || !si.files.iter().any(|f| f.ends_with(".dvd"))
    {
        return None;
    }
    let base;
    let fi = if fi.doc_values_gen == -1 {
        fi
    } else {
        base = field_infos::FieldInfo {
            doc_values_gen: -1,
            ..fi.clone()
        };
        &base
    };
    Some(open_field_column(dir, commit, si, field_infos, fi, false, false).map(|c| (c.meta, c.dvd)))
}

/// One opened base column, keyed by [`column_key`].
type BaseColumn = (
    (i64, String),
    (doc_values::DocValuesMeta, lucene_store::directory::Input),
);

/// [`open_base_column`] through a per-check cache, so several fields of one
/// format instance open and parse its `.dvm` once.
fn cached_base_column<'c>(
    columns: &'c mut Vec<BaseColumn>,
    dir: &dyn Directory,
    commit: &SegmentCommitInfo,
    si: &SegmentInfo,
    field_infos: &FieldInfos,
    fi: &field_infos::FieldInfo,
) -> Result<(&'c doc_values::DocValuesMeta, &'c [u8]), String> {
    let key = (
        -1,
        crate::field_updates::field_per_field_component(fi).unwrap_or_default(),
    );
    let at = match columns.iter().position(|(k, _)| *k == key) {
        Some(at) => at,
        None => {
            let column = open_base_column(dir, commit, si, field_infos, fi)
                .ok_or_else(|| "the segment has no doc-values files".to_string())??;
            columns.push((key, column));
            columns.len().saturating_sub(1)
        }
    };
    let (meta, dvd) = &columns[at].1;
    Ok((meta, &dvd[..]))
}

/// Which documents carry a value for `fi` -- `DocValuesIterator`'s
/// "advanceExact returns true", which is what
/// `PendingSoftDeletes.countSoftDeletes` counts over the soft-deletes field.
///
/// Only the two doc-values types a soft-deletes field can have; separate from
/// [`sort_key_values`], which reads one of three columns depending on the
/// `SortField` kind.
fn doc_values_presence(
    dvd: &[u8],
    meta: &doc_values::DocValuesMeta,
    fi: &field_infos::FieldInfo,
    max_doc: i32,
) -> Result<Vec<bool>, String> {
    let mut present = Vec::with_capacity(max_doc.max(0) as usize);
    match fi.doc_values_type {
        field_infos::DocValuesType::Numeric => {
            let entry = meta
                .numeric_entry(fi.number)
                .ok_or("no NUMERIC doc-values entry")?;
            let mut reader = doc_values::NumericReader::new(dvd, entry);
            for doc in 0..max_doc {
                present.push(reader.value(doc).map_err(|e| e.to_string())?.is_some());
            }
        }
        field_infos::DocValuesType::SortedNumeric => {
            let entry = meta
                .sorted_numeric_entry(fi.number)
                .ok_or("no SORTED_NUMERIC doc-values entry")?;
            let mut reader = doc_values::SortedNumericReader::new(dvd, entry);
            let mut values = Vec::new();
            for doc in 0..max_doc {
                reader.values(doc, &mut values).map_err(|e| e.to_string())?;
                present.push(!values.is_empty());
            }
        }
        other => return Err(format!("field has doc-values type {other:?}")),
    }
    Ok(present)
}

/// Reads one field's per-doc sort key as a single `Option<i64>` -- the shape
/// [`segment_info::SortKeyComparator`] compares, and the shape every index
/// sort whose key is not raw bytes reduces to.
///
/// Which doc-values column and which reduction is the sort field's own
/// business, exactly as it is Java's: `SortField(field, LONG)` reads NUMERIC,
/// `SortedNumericSortField` reads SORTED_NUMERIC and applies its selector
/// (`SortedNumericSelector.Type.MIN`/`MAX`), and a STRING sort reads SORTED
/// and yields the **term ordinal** (`IndexSorter.StringSorter` compares ords,
/// not bytes). `Err` names the combination rather than guessing.
///
/// A `SortedSetSortField` reads SORTED_SET and applies its
/// `SortedSetSelector`, and a `BinarySortField` reads BINARY; both, like
/// STRING, reduce to the rank of the chosen value among the segment's values
/// ([`crate::index_sorter`]) -- for a SORTED_SET that is its ordinal, for
/// BINARY the order `BytesRef.compareTo` gives.
fn sort_key_values(
    dvd: &[u8],
    meta: &doc_values::DocValuesMeta,
    sf: &segment_info::IndexSortField,
    fi: &field_infos::FieldInfo,
    max_doc: i32,
) -> Result<Vec<Option<i64>>, String> {
    use segment_info::{IndexSortKind, SortedNumericSelector};
    let mut keys = Vec::with_capacity(max_doc.max(0) as usize);
    match (&sf.kind, fi.doc_values_type) {
        (IndexSortKind::Numeric(_), field_infos::DocValuesType::Numeric) => {
            let entry = meta
                .numeric_entry(fi.number)
                .ok_or("no NUMERIC doc-values entry")?;
            let mut reader = doc_values::NumericReader::new(dvd, entry);
            for doc in 0..max_doc {
                keys.push(reader.value(doc).map_err(|e| e.to_string())?);
            }
        }
        (
            IndexSortKind::SortedNumeric { selector, .. },
            field_infos::DocValuesType::SortedNumeric,
        ) => {
            let entry = meta
                .sorted_numeric_entry(fi.number)
                .ok_or("no SORTED_NUMERIC doc-values entry")?;
            let mut reader = doc_values::SortedNumericReader::new(dvd, entry);
            let mut values = Vec::new();
            for doc in 0..max_doc {
                reader.values(doc, &mut values).map_err(|e| e.to_string())?;
                // `SortedNumericSelector.MinValue`/`MaxValue`: the *first*
                // and the *last* stored value of the document, not its
                // smallest and largest. The two coincide only for an
                // ascending column, which is what
                // `doc_values.sorted_numeric_ascending` separately checks --
                // re-deriving min/max here would mask an unsorted column
                // rather than expose it.
                keys.push(match selector {
                    SortedNumericSelector::Min => values.first().copied(),
                    SortedNumericSelector::Max => values.last().copied(),
                });
            }
        }
        (IndexSortKind::String(_), field_infos::DocValuesType::Sorted) => {
            let entry = meta
                .sorted_entry(fi.number)
                .ok_or("no SORTED doc-values entry")?;
            // `sorted_ord` is `numeric_value` over the entry's `ords`.
            let mut reader = doc_values::NumericReader::new(dvd, &entry.ords);
            for doc in 0..max_doc {
                keys.push(reader.value(doc).map_err(|e| e.to_string())?);
            }
        }
        (IndexSortKind::SortedSet { .. }, field_infos::DocValuesType::SortedSet)
        | (IndexSortKind::Binary(_), field_infos::DocValuesType::Binary) => {
            let terms = crate::index_sorter::read_segment_terms(dvd, meta, sf, fi.number, max_doc)
                .map_err(|e| e.to_string())?
                .ok_or(
                    if fi.doc_values_type == field_infos::DocValuesType::Binary {
                        "no BINARY doc-values entry"
                    } else {
                        "no SORTED_SET doc-values entry"
                    },
                )?;
            keys = crate::index_sorter::rank_terms(&[terms])
                .pop()
                .unwrap_or_default();
        }
        (kind, dv) => {
            return Err(format!(
                "sort kind {kind:?} does not match the field's doc-values type {dv:?}"
            ))
        }
    }
    Ok(keys)
}

/// `CheckIndex.testSort`: a segment that declares an index sort must
/// actually *be* sorted by it. Java rebuilds the sort's comparators
/// (`IndexSorter.getDocComparator` per `SortField`) and walks adjacent doc
/// ids asserting `cmp <= 0`; this rebuilds them from the same `.si` via
/// [`segment_info::SortKeyComparator`] -- the exact comparator the
/// sort-on-flush writer and the sort-preserving merge use to *produce* the
/// order, applied in reverse as a verifier.
///
/// Not run for an unsorted segment, and skipped (not failed) for a segment
/// with no doc-values files -- the index is openable and everything else
/// about it is checked, but this one property is unverified and saying so is
/// the difference between a check that passed and one that never ran. Every
/// sort kind is verified, the byte-keyed ones through
/// [`crate::index_sorter`].
pub(crate) fn check_index_sort(
    dir: &dyn Directory,
    commit: &SegmentCommitInfo,
    si: &SegmentInfo,
    field_infos: &FieldInfos,
    checks: &mut Vec<Check>,
) {
    let Some(sort_fields) = &si.index_sort else {
        return;
    };
    if !si.files.iter().any(|f| f.ends_with(".dvm"))
        || !si.files.iter().any(|f| f.ends_with(".dvd"))
    {
        // The sharpest of the skip cases, and the reason this is modelled at
        // all: a segment that *declares* an index sort but carries no
        // doc-values files has nothing this check can read, and the sort is
        // then verified by nothing anywhere. Reporting it keeps `all_passed()` from
        // calling the segment fine over a `.si` whose
        // declared order every merge and every early-terminating query will
        // trust. `fnm.doc_values_vs_files` does not cover it: a sort field
        // that is not in `.fnm` at all leaves nothing to claim doc values,
        // so that check passes too.
        checks.push(Check::skipped(
            "sort.docs_in_index_sort_order",
            "the segment's doc-values files, which the sort keys live in,",
        ));
        return;
    };
    let result = (|| -> Result<Vec<String>, String> {
        // Each field is read from its own `PerFieldDocValuesFormat`
        // instance's column, opened once per instance.
        let mut columns: Vec<BaseColumn> = Vec::new();
        let mut per_field: Vec<(Vec<Option<i64>>, segment_info::SortKeyComparator)> =
            Vec::with_capacity(sort_fields.len());
        for sf in sort_fields {
            let fi = field_infos
                .fields
                .iter()
                .find(|f| f.name == sf.field)
                .ok_or_else(|| format!("sort field {:?} is not in .fnm", sf.field))?;
            let cmp = segment_info::SortKeyComparator::new(sf);
            let (meta, dvd) = cached_base_column(&mut columns, dir, commit, si, field_infos, fi)
                .map_err(|e| format!("sort field {:?}: {e}", sf.field))?;
            per_field.push((
                sort_key_values(dvd, meta, sf, fi, si.doc_count)
                    .map_err(|e| format!("sort field {:?}: {e}", sf.field))?,
                cmp,
            ));
        }

        // A segment with blocks is sorted parent by parent: Java walks the
        // parent field's documents (`reader.getNumericDocValues(parentField)`)
        // instead of every one, and refuses blocks without a parent field.
        let walked: Vec<i32> = if si.has_blocks {
            let parent = field_infos
                .fields
                .iter()
                .find(|f| f.parent_field)
                .ok_or_else(|| {
                    "parent field is not set but the index has document blocks".to_string()
                })?;
            // `reader.getNumericDocValues(parentField)`: Java walks it
            // unconditionally, so a block segment without the column fails
            // the check rather than passing over an empty walk.
            let (meta, dvd) =
                cached_base_column(&mut columns, dir, commit, si, field_infos, parent)
                    .map_err(|e| format!("parent field: {e}"))?;
            match meta.numeric_entry(parent.number) {
                None => {
                    return Err(format!(
                        "the index has document blocks but parent field {:?} has no doc values",
                        parent.name
                    ))
                }
                Some(entry) => {
                    let mut reader = doc_values::NumericReader::new(dvd, entry);
                    let mut docs = Vec::new();
                    for doc in 0..si.doc_count {
                        if reader
                            .value(doc)
                            .map_err(|e| format!("parent field: {e}"))?
                            .is_some()
                        {
                            docs.push(doc);
                        }
                    }
                    docs
                }
            }
        } else {
            (0..si.doc_count).collect()
        };
        let mut problems = Vec::new();
        for pair in walked.windows(2) {
            let (prev, doc) = (pair[0], pair[1]);
            let mut ordering = std::cmp::Ordering::Equal;
            for (keys, cmp) in &per_field {
                ordering = cmp.compare(keys[prev as usize], keys[doc as usize]);
                if ordering != std::cmp::Ordering::Equal {
                    break;
                }
            }
            if ordering == std::cmp::Ordering::Greater {
                problems.push(format!(
                    "segment declares an index sort but docID={prev} sorts after docID={doc}"
                ));
            }
        }
        Ok(problems)
    })();

    match result {
        Ok(problems) => checks.push(named_field_check(
            "sort.docs_in_index_sort_order",
            &problems,
            si.doc_count as i64,
            "docs",
        )),
        Err(e) => checks.push(Check::fail("sort.docs_in_index_sort_order", e)),
    }
}

/// `CheckIndex.checkSoftDeletes`: the number of *live* documents that carry
/// a value for the soft-deletes field must equal the commit's recorded
/// `softDelCount` (Java's `PendingSoftDeletes.countSoftDeletes(iterator,
/// liveDocs)`). Nothing else in this port ever validated `soft_del_count`
/// against the data it summarises.
///
/// Skipped (not failed) when `.fnm` marks no field as the soft-deletes
/// field, or the segment has no doc-values files.
fn check_soft_deletes(
    dir: &dyn Directory,
    commit: &SegmentCommitInfo,
    si: &SegmentInfo,
    field_infos: &FieldInfos,
    checks: &mut Vec<Check>,
) {
    // Java asks the reader's *current* field infos
    // (`reader.getFieldInfos().getSoftDeletesField()`). A soft delete that
    // lands on a segment already written is a doc-values update: a new `.fnm`
    // generation, and a new `.dvm`/`.dvd` generation for the field
    // (`_4o_1_Lucene90_0.dvm`) -- possibly the first the segment ever had of
    // it, the field absent from the base `.fnm` altogether. Reading only the
    // base `.fnm` and column skipped the check for such segments and
    // miscounted the others, failing segments real Lucene's `CheckIndex`
    // passes.
    let current = if commit.field_infos_gen == -1 {
        None
    } else {
        match crate::field_updates::read_current_field_infos(dir, commit, &si.files) {
            Ok(current) => Some(current),
            Err(e) => {
                checks.push(Check::fail("soft_deletes.count_matches", e.to_string()));
                return;
            }
        }
    };
    let lookup = current.as_ref().unwrap_or(field_infos);
    let Some(index) = lookup.fields.iter().position(|f| f.soft_deletes_field) else {
        return;
    };
    let fi = &lookup.fields[index];
    // A field with no update generation is read from the base column, as
    // `SegmentDocValuesProducer` hands it to the base producer.
    let opened = match &current {
        Some(current) if fi.doc_values_gen != -1 => {
            let per_field = crate::field_updates::field_per_field_component(fi).unwrap_or_default();
            crate::field_updates::read_current_column(
                dir, commit, &si.files, current, index, &per_field,
            )
            .map_err(|e| e.to_string())
            .transpose()
        }
        _ => open_base_column(dir, commit, si, field_infos, fi),
    };
    let Some(column) = opened else {
        // Same shape as the index sort above: `.fnm` names a soft-deletes
        // field, so `softDelCount` is a claim about data -- and with no
        // doc-values files there is no data to check it against. Reported,
        // not skipped silently.
        checks.push(Check::skipped(
            "soft_deletes.count_matches",
            "the segment's doc-values files, which the soft-deletes field lives in,",
        ));
        return;
    };
    let result = (|| -> Result<i32, String> {
        let (meta, dvd) = column?;
        let live = if commit.del_gen == -1 {
            None
        } else {
            let liv_name = liv_file_name(&commit.segment_name, commit.del_gen);
            let bytes = dir.open(&liv_name).map_err(|e| e.to_string())?;
            Some(
                live_docs::parse(
                    &bytes,
                    &commit.segment_id,
                    commit.del_gen,
                    si.doc_count as usize,
                    commit.del_count as usize,
                )
                .map_err(|e| e.to_string())?,
            )
        };
        let present = doc_values_presence(&dvd, &meta, fi, si.doc_count)
            .map_err(|e| format!("soft-deletes field {:?}: {e}", fi.name))?;
        Ok(present
            .iter()
            .enumerate()
            .filter(|(doc, has_value)| **has_value && is_live_at(live.as_ref(), *doc))
            .count() as i32)
    })();

    match result {
        Ok(actual) if actual == commit.soft_del_count => {
            checks.push(Check::pass("soft_deletes.count_matches"))
        }
        Ok(actual) => checks.push(Check::fail(
            "soft_deletes.count_matches",
            format!(
                "actual soft deletes: {actual} but expected: {}",
                commit.soft_del_count
            ),
        )),
        Err(e) => checks.push(Check::fail("soft_deletes.count_matches", e)),
    }
}

/// For every field with points data (`.fnm`'s `point_dimension_count != 0`),
/// walks the field's actual BKD tree leaf-by-leaf via
/// [`lucene_codecs::points::PointsReader::decode_leaves`] -- reusing this
/// port's existing BKD decoder rather than re-parsing `.kdm`/`.kdi`/`.kdd` --
/// and verifies real structural invariants a corrupted or buggy writer could
/// violate silently:
///
/// - every point's packed value (over the index dimensions) actually falls
///   within the field's own `.kdm`-declared `min_packed_value`/
///   `max_packed_value` (an unsigned, per-dimension byte-wise range check)
///   -- `points.value_within_field_bounds:<field>`. This is the check a
///   corrupted `.kdd` point value (or a writer that mis-tracked its own
///   min/max) would fail, since the field-level bound comes from `.kdm`, a
///   file entirely separate from the `.kdd` bytes a point's value is
///   decoded from.
/// - when a field has more than one index dimension, `.kdd` leaf blocks
///   embed their own (tighter) per-leaf bounding box (see
///   [`lucene_codecs::points::Leaf::bound`]); that box must itself be a
///   subset of the field-level bound above -- `points.leaf_bounds_subset_of_field:<field>`.
///   Skipped for single-index-dimension fields, since no such box exists on
///   disk in that case (see `points.rs`'s `read_leaf_block` doc comment).
/// - the leaves' decoded point counts must sum to `.kdm`'s own declared
///   field-level `point_count` -- `points.point_count_matches:<field>`.
///
/// Skipped (not failed) when no field in `.fnm` claims points at all. A field
/// that *does* claim points but whose segment is missing one of
/// `.kdm`/`.kdi`/`.kdd` is reported as a single `points.open` failure rather
/// than silently skipped -- points files are optional at the segment level
/// (most segments have none), but once one field commits to having them, all
/// three must be present and parse.
fn check_points_structural_invariants(
    dir: &dyn Directory,
    commit: &SegmentCommitInfo,
    si: &SegmentInfo,
    field_infos: &FieldInfos,
    checks: &mut Vec<Check>,
) {
    let points_fields: Vec<&field_infos::FieldInfo> = field_infos
        .fields
        .iter()
        .filter(|f| f.point_dimension_count != 0)
        .collect();
    if points_fields.is_empty() {
        return;
    }

    let kdm_name = si.files.iter().find(|f| f.ends_with(".kdm"));
    let kdi_name = si.files.iter().find(|f| f.ends_with(".kdi"));
    let kdd_name = si.files.iter().find(|f| f.ends_with(".kdd"));
    let (kdm_name, kdi_name, kdd_name) = match (kdm_name, kdi_name, kdd_name) {
        (Some(m), Some(i), Some(d)) => (m, i, d),
        _ => {
            checks.push(Check::fail(
                "points.open",
                "a field claims points but the segment is missing one or more of .kdm/.kdi/.kdd",
            ));
            skip_families(checks, POINTS_FAMILIES, "points.open");
            return;
        }
    };

    let result = (|| -> Result<Vec<Check>, String> {
        let kdm = dir.open(kdm_name).map_err(|e| e.to_string())?;
        let kdi = dir.open(kdi_name).map_err(|e| e.to_string())?;
        let kdd = dir.open(kdd_name).map_err(|e| e.to_string())?;
        let reader =
            points::open(&kdm, &kdi, &kdd, &commit.segment_id, "").map_err(|e| e.to_string())?;

        let mut field_checks = Vec::new();
        for field_info in &points_fields {
            let field_name = &field_info.name;
            let field = match reader.field(field_info.number) {
                Some(f) => f,
                None => {
                    // CheckIndex.testPoints skips a null PointValues. A merge
                    // retains the field's dimensions even when every point
                    // was deleted, so an absent BKD entry is legitimate. The
                    // triple was opened above; there are no leaves to inspect.
                    field_checks.push(Check::pass(format!("points.field_empty:{field_name}")));
                    continue;
                }
            };

            let leaves = match reader.decode_leaves(field_info.number) {
                Ok(l) => l,
                Err(e) => {
                    field_checks.push(Check::fail(
                        format!("points.decode:{field_name}"),
                        e.to_string(),
                    ));
                    skip_families(
                        &mut field_checks,
                        &POINTS_PER_FIELD
                            .iter()
                            .map(|f| format!("{f}:{field_name}"))
                            .collect::<Vec<_>>()
                            .iter()
                            .map(String::as_str)
                            .collect::<Vec<_>>(),
                        &format!("points.decode:{field_name}"),
                    );
                    continue;
                }
            };

            let num_index_dims = field.num_index_dims as usize;
            let bytes_per_dim = field.bytes_per_dim as usize;

            let mut value_problems = Problems::new();
            let mut bound_problems = Problems::new();
            let mut doc_problems = Problems::new();
            let mut docs_seen =
                lucene_util::fixed_bit_set::FixedBitSet::new(si.doc_count.max(0) as usize);
            let mut total_points = 0i64;
            for (leaf_idx, leaf) in leaves.iter().enumerate() {
                total_points = total_points.saturating_add(leaf.points.len() as i64);
                for point in &leaf.points {
                    // `VerifyPointsVisitor.visit(docID, packedValue)`'s doc
                    // bookkeeping.
                    if point.doc_id < 0 || point.doc_id >= si.doc_count {
                        doc_problems.push(format!(
                            "field {field_name:?} leaf {leaf_idx}: docID={} is outside 0..{}",
                            point.doc_id, si.doc_count
                        ));
                    } else {
                        // FBS: `docs_seen` is
                        // `FixedBitSet::new(si.doc_count.max(0))` in this same
                        // function, so `si.doc_count` is its `len()`.
                        docs_seen.set(point.doc_id as usize);
                    }
                    // ARITH: `points::open` allocates the packed min/max
                    // values as exactly `numIndexDims * bytesPerDim` bytes
                    // and rejects a `.kdm` whose product overflows, and
                    // `decode_leaves` rejects a point whose packed value is
                    // not `numDims * bytesPerDim` bytes with
                    // `numIndexDims <= numDims`. So `dim < num_index_dims`
                    // bounds every index here, and the product itself was
                    // checked before this function ever saw the field.
                    #[allow(clippy::arithmetic_side_effects)]
                    for dim in 0..num_index_dims {
                        let lo = dim * bytes_per_dim;
                        let hi = lo + bytes_per_dim;
                        let value = &point.packed_value[lo..hi];
                        let field_min = &field.min_packed_value[lo..hi];
                        let field_max = &field.max_packed_value[lo..hi];
                        if value < field_min || value > field_max {
                            value_problems.push(format!(
                                "field {field_name:?} leaf {leaf_idx} doc {}: dim {dim} value is \
                                 outside the field's declared min/max packed value",
                                point.doc_id
                            ));
                        }
                    }
                }
                if let Some((min_bound, max_bound)) = &leaf.bound {
                    // ARITH: as above; `read_leaf_block` reads each bound
                    // as `numIndexDims * bytesPerDim` bytes.
                    #[allow(clippy::arithmetic_side_effects)]
                    for dim in 0..num_index_dims {
                        let lo = dim * bytes_per_dim;
                        let hi = lo + bytes_per_dim;
                        if min_bound[lo..hi] < field.min_packed_value[lo..hi]
                            || max_bound[lo..hi] > field.max_packed_value[lo..hi]
                        {
                            bound_problems.push(format!(
                                "field {field_name:?} leaf {leaf_idx}: leaf's own bounding box for \
                                 dim {dim} is not a subset of the field's declared bounding box"
                            ));
                        }
                    }
                }
            }

            // `CheckIndex.VerifyPointsVisitor`, driven through the BKD
            // tree's own pruning walk exactly as Java's
            // `values.intersect(visitor)`: every cell the walk hands to
            // `compare` must be non-inverted and inside the field's bounds,
            // every point must fall inside the last cell compared, and a
            // one-dimensional field's values must be one ascending sweep
            // with ties broken by ascending docID. The leaf decode above
            // cannot see any of this: it checks points against the field's
            // bounds only, and cells exist only in the walk.
            {
                let mut cells = Problems::new();
                let mut visitor = VerifyPointsVisitor::new(field_name, field, &mut cells);
                let walk = reader.intersect(field_info.number, &mut visitor);
                let VerifyPointsVisitor {
                    order: order_problems,
                    in_cell: in_cell_problems,
                    ..
                } = visitor;
                if let Err(e) = walk {
                    value_problems.push(format!(
                        "field {field_name:?}: the BKD intersect walk failed: {e}"
                    ));
                }
                for p in [in_cell_problems, order_problems] {
                    value_problems.absorb(p);
                }
                // A one-dimensional field has no
                // `points.leaf_bounds_subset_of_field` check to report in.
                if num_index_dims == 1 {
                    value_problems.absorb(cells);
                } else {
                    bound_problems.absorb(cells);
                }
            }

            field_checks.push(named_field_check(
                &format!("points.value_within_field_bounds:{field_name}"),
                &value_problems,
                leaves.len() as i64,
                "leaves",
            ));
            if num_index_dims != 1 {
                field_checks.push(named_field_check(
                    &format!("points.leaf_bounds_subset_of_field:{field_name}"),
                    &bound_problems,
                    leaves.len() as i64,
                    "leaves",
                ));
            }
            // `VerifyPointsVisitor`'s constructor guards plus
            // `testPoints`' `getDocCountSeen() != docCount` check; the doc count
            // feeds a `PointRangeQuery`'s cost estimate and `IndexSearcher`'s
            // pruning.
            if field.doc_count as i64 > field.point_count {
                doc_problems.push(format!(
                    "field {field_name:?} claims docCount={} with only point_count={} points",
                    field.doc_count, field.point_count
                ));
            }
            if field.doc_count > si.doc_count {
                doc_problems.push(format!(
                    "field {field_name:?} claims docCount={} but the segment has maxDoc={}",
                    field.doc_count, si.doc_count
                ));
            }
            let docs_seen_count = docs_seen.cardinality() as i32;
            if docs_seen_count != field.doc_count {
                doc_problems.push(format!(
                    "field {field_name:?} claims docCount={} but its leaves reference {docs_seen_count} distinct doc IDs",
                    field.doc_count
                ));
            }
            field_checks.push(named_field_check(
                &format!("points.doc_count_matches:{field_name}"),
                &doc_problems,
                leaves.len() as i64,
                "leaves",
            ));
            if total_points == field.point_count {
                field_checks.push(Check::pass(format!(
                    "points.point_count_matches:{field_name}"
                )));
            } else {
                field_checks.push(Check::fail(
                    format!("points.point_count_matches:{field_name}"),
                    format!(
                        "field declares point_count={} but its leaves decoded {total_points} points",
                        field.point_count
                    ),
                ));
            }
        }
        Ok(field_checks)
    })();

    match result {
        Ok(field_checks) => checks.extend(field_checks),
        Err(e) => {
            checks.push(Check::fail("points.open", e));
            skip_families(checks, POINTS_FAMILIES, "points.open");
        }
    }
}

/// `CheckIndex.VerifyPointsVisitor`'s cell and ordering checks, as an
/// [`points::IntersectVisitor`] that answers `CELL_CROSSES_QUERY` for every
/// cell so the walk hands it every point with its value. (Its doc and point
/// counting is [`check_points_structural_invariants`]' leaf walk.)
struct VerifyPointsVisitor<'a> {
    field_name: &'a str,
    num_data_dims: i32,
    num_index_dims: usize,
    bytes_per_dim: usize,
    global_min: &'a [u8],
    global_max: &'a [u8],
    last_min: Vec<u8>,
    last_max: Vec<u8>,
    /// The last point's first-dimension value and docID, for the
    /// one-dimensional sweep check.
    last_value: Vec<u8>,
    last_doc: i32,
    cells: &'a mut Problems,
    in_cell: Problems,
    order: Problems,
}

impl<'a> VerifyPointsVisitor<'a> {
    fn new(field_name: &'a str, field: &'a points::PointsField, cells: &'a mut Problems) -> Self {
        let bytes_per_dim = usize::try_from(field.bytes_per_dim).unwrap_or(0);
        let num_index_dims = usize::try_from(field.num_index_dims).unwrap_or(0);
        Self {
            field_name,
            num_data_dims: field.num_dims,
            num_index_dims,
            bytes_per_dim,
            global_min: &field.min_packed_value,
            global_max: &field.max_packed_value,
            last_min: field.min_packed_value.clone(),
            last_max: field.max_packed_value.clone(),
            last_value: vec![0; bytes_per_dim],
            last_doc: -1,
            cells,
            in_cell: Problems::new(),
            order: Problems::new(),
        }
    }

    /// Dimension `dim`'s bytes of a packed value, empty when the value is
    /// too short to have them (which `points::open` and the leaf decoder
    /// already reject).
    fn dim_of(&self, packed: &'a [u8], dim: usize) -> &'a [u8] {
        let lo = dim.saturating_mul(self.bytes_per_dim);
        packed
            .get(lo..lo.saturating_add(self.bytes_per_dim))
            .unwrap_or(&[])
    }
}

impl points::IntersectVisitor for VerifyPointsVisitor<'_> {
    fn compare(&mut self, min_packed: &[u8], max_packed: &[u8]) -> points::Relation {
        self.last_min.clear();
        self.last_min.extend_from_slice(min_packed);
        self.last_max.clear();
        self.last_max.extend_from_slice(max_packed);
        let name = self.field_name;
        for dim in 0..self.num_index_dims {
            let lo = dim.saturating_mul(self.bytes_per_dim);
            let range = lo..lo.saturating_add(self.bytes_per_dim);
            let (Some(min), Some(max)) = (min_packed.get(range.clone()), max_packed.get(range))
            else {
                continue;
            };
            if min > max {
                self.cells.push(format!(
                    "field {name:?}: packed points cell minPackedValue {min:?} is out-of-bounds \
                     of the cell's maxPackedValue {max:?} dim={dim}"
                ));
            }
            if min < self.dim_of(self.global_min, dim) {
                self.cells.push(format!(
                    "field {name:?}: packed points cell minPackedValue {min:?} is out-of-bounds \
                     of the global minimum dim={dim}"
                ));
            }
            if max > self.dim_of(self.global_max, dim) {
                self.cells.push(format!(
                    "field {name:?}: packed points cell maxPackedValue {max:?} is out-of-bounds \
                     of the global maximum dim={dim}"
                ));
            }
        }
        points::Relation::CellCrossesQuery
    }

    fn visit(&mut self, doc_id: i32) {
        self.in_cell.push(format!(
            "field {:?}: codec called IntersectVisitor.visit without a packed value for \
             docID={doc_id}",
            self.field_name
        ));
    }

    fn visit_with_value(&mut self, doc_id: i32, packed_value: &[u8]) {
        let name = self.field_name;
        for dim in 0..self.num_index_dims {
            let lo = dim.saturating_mul(self.bytes_per_dim);
            let range = lo..lo.saturating_add(self.bytes_per_dim);
            let (Some(value), Some(min), Some(max)) = (
                packed_value.get(range.clone()),
                self.last_min.get(range.clone()),
                self.last_max.get(range),
            ) else {
                continue;
            };
            if value < min || value > max {
                self.in_cell.push(format!(
                    "field {name:?} docID={doc_id}: packed points value {value:?} is \
                     out-of-bounds of the last cell min={min:?} max={max:?} dim={dim}"
                ));
            }
        }
        // `numDataDims == 1`: one in-order sweep, ties by increasing docID.
        // (With more data dimensions a leaf is sorted by whichever
        // dimension compresses best, so there is no global order.)
        if self.num_data_dims == 1 {
            let Some(value) = packed_value.get(..self.bytes_per_dim) else {
                return;
            };
            match self.last_value.as_slice().cmp(value) {
                std::cmp::Ordering::Greater => self.order.push(format!(
                    "field {name:?} docID={doc_id}: packed points value {value:?} is \
                     out-of-order vs the previous document's value {:?}",
                    self.last_value
                )),
                std::cmp::Ordering::Equal if doc_id < self.last_doc => self.order.push(format!(
                    "field {name:?}: packed points value is the same, but docID={doc_id} \
                         is out of order vs previous docID={}",
                    self.last_doc
                )),
                _ => {}
            }
            self.last_value.clear();
            self.last_value.extend_from_slice(value);
            self.last_doc = doc_id;
        }
    }
}

/// The per-field checks a field's own `points.decode`
/// failure takes down -- everything that walks the field's leaves.
const POINTS_PER_FIELD: &[&str] = &[
    "points.value_within_field_bounds",
    "points.leaf_bounds_subset_of_field",
    "points.doc_count_matches",
    "points.point_count_matches",
];

/// The per-field families a failed `points.open` takes down.
const POINTS_FAMILIES: &[&str] = &[
    "points.field_empty",
    "points.value_within_field_bounds",
    "points.leaf_bounds_subset_of_field",
    "points.doc_count_matches",
    "points.point_count_matches",
];

/// `CheckIndex.testFieldNorms`: for every field `.fnm` says has norms, read
/// **every** norm value out of `.nvd`, plus `checkFields`' terms-vs-norms
/// cross-check.
///
/// Checking only that `.nvd`/`.nvm` exist would let a corrupted norms
/// payload pass and silently change every BM25 score that reads it.
///
/// Three checks per field:
///
/// - `norms.entry_present:<f>` -- `.nvm` has an entry for a field `.fnm`
///   says has norms (Java's `normsReader.getNorms(info)` returning null).
/// - `norms.values_decode:<f>` -- every doc's value decodes, and the number
///   of docs that actually carry one equals `.nvm`'s own
///   `numDocsWithField` (Java's `checkNumericDocValues`, which walks the
///   iterator and counts).
/// - `norms.agree_with_postings:<f>` -- `checkFields`' "Cross-check terms
///   with norms": a live doc with a non-zero norm must have terms in that
///   field's postings, a live doc with a zero norm must not, and the two
///   counts must match. Deleted docs are exempt, exactly as in Java ("norms
///   may only be out of sync with terms on deleted documents").
///
/// A sparse norms field is walked with one forward-only
/// [`lucene_codecs::indexed_disi::DisiCursor`] rather than `norms::norm_value` per doc:
/// that helper builds a fresh cursor per lookup, which would make this an
/// O(maxDoc x blocks) scan instead of O(maxDoc).
#[allow(clippy::too_many_arguments)]
fn check_field_norms(
    dir: &dyn Directory,
    commit: &SegmentCommitInfo,
    si: &SegmentInfo,
    field_infos: &FieldInfos,
    live_docs: Option<&lucene_util::fixed_bit_set::FixedBitSet>,
    docs_with_terms: &DocsWithTerms,
    stats: &mut CheckStats,
    checks: &mut Vec<Check>,
) {
    let with_norms: Vec<&field_infos::FieldInfo> = field_infos
        .fields
        .iter()
        .filter(|f| !f.omit_norms && f.index_options != field_infos::IndexOptions::None)
        .collect();
    if with_norms.is_empty() {
        return;
    }
    let nvd_name = si.files.iter().find(|f| f.ends_with(".nvd"));
    let nvm_name = si.files.iter().find(|f| f.ends_with(".nvm"));
    // A claiming field with no norms files at all is already reported by
    // `fnm.norms_vs_files`; don't duplicate it.
    let (Some(nvd_name), Some(nvm_name)) = (nvd_name, nvm_name) else {
        return;
    };

    let opened = (|| -> Result<(lucene_store::directory::Input, norms::Norms), String> {
        let nvd = dir.open(nvd_name).map_err(|e| e.to_string())?;
        let nvm = dir.open(nvm_name).map_err(|e| e.to_string())?;
        let (_meta_version, norms) =
            norms::parse_meta(&nvm, &commit.segment_id, "").map_err(|e| e.to_string())?;
        // Java's `Lucene90NormsProducer` compares the `.nvm`'s format version
        // against the `.nvd`'s. That comparison cannot fire here and is not
        // ported: `norms::VERSION_START == VERSION_CURRENT == 0`, and both
        // headers are validated against that one-element range by
        // `check_index_header`, so both versions are `0` or the open already
        // failed. (A second norms format version would make it a real check
        // again -- see c25.)
        norms::check_data_header_footer(&nvd, &commit.segment_id, "").map_err(|e| e.to_string())?;
        Ok((nvd, norms))
    })();
    let (nvd, norms) = match opened {
        Ok(v) => v,
        Err(e) => {
            checks.push(Check::fail("norms.open", e));
            // No check other than these reads `.nvd`,
            // so a skipped norms family is a completely unguarded file.
            skip_families(
                checks,
                &[
                    "norms.entries_name_real_norms_fields",
                    "norms.entry_present",
                    "norms.values_decode",
                    "norms.agree_with_postings",
                ],
                "norms.open",
            );
            return;
        }
    };

    // `Lucene90NormsProducer.readFields`' other half: every `.nvm` entry must
    // name a field the `.fnm` has, and one that actually has norms. Java
    // rejects the segment at open; this port's `norms::parse_meta` does not
    // take `FieldInfos` (see `norms::validate_fields` for why), so the
    // diagnostic lives here -- which is `CheckIndex`'s job anyway. The
    // `norms.entry_present` loop below is the *converse* check (a field with
    // no entry); neither implies the other.
    checks.push(match norms::validate_fields(&norms, field_infos) {
        Ok(()) => Check::pass("norms.entries_name_real_norms_fields"),
        Err(e) => Check::fail("norms.entries_name_real_norms_fields", e.to_string()),
    });

    for fi in with_norms {
        let Some(entry) = norms.entry(fi.number) else {
            checks.push(Check::fail(
                format!("norms.entry_present:{}", fi.name),
                format!(
                    "field {:?} has omitNorms=false but .nvm carries no entry for field number {}",
                    fi.name, fi.number
                ),
            ));
            // Without an entry there is no column to read, so this field's
            // norms are checked by nothing at all: nothing else in the
            // module reads `.nvd`.
            skip_families(
                checks,
                &[
                    &format!("norms.values_decode:{}", fi.name),
                    &format!("norms.agree_with_postings:{}", fi.name),
                ],
                &format!("norms.entry_present:{}", fi.name),
            );
            continue;
        };
        checks.push(Check::pass(format!("norms.entry_present:{}", fi.name)));
        stats.norm_fields = stats.norm_fields.saturating_add(1);

        // `checkFields`' terms-vs-norms cross-check runs in the same pass:
        // Java pulls `visitedDocs` from the postings walk it has already
        // done, and so do we. `None` means the field has no term dictionary
        // at all, in which case Java's `checkFields` never reaches the
        // cross-check either (it lives inside `checkFields`' per-field
        // loop) -- so the check is skipped rather than run against an empty
        // bitset, matching Java.
        let visited = docs_with_terms.get(&fi.name);

        let mut decode_problems = Problems::new();
        let mut agree_problems = Problems::new();
        let mut docs_with_norm = 0i64;
        // `actual`: live docs with a non-zero norm. `expected`: live docs
        // that have terms.
        let mut actual_count = 0i64;
        let mut expected_count = 0i64;

        // A dense entry (`docsWithFieldOffset == -1`) means *every* document
        // has a norm, so `numDocsWithField` must be `maxDoc` -- checked
        // directly rather than by counting, because iterating
        // `0..num_docs_with_field` and then comparing the count against
        // `num_docs_with_field` would be a tautology that also silently
        // skipped the tail documents.
        if entry.is_dense() && entry.num_docs_with_field != si.doc_count {
            decode_problems.push(format!(
                ".nvm declares a dense field (docsWithFieldOffset=-1) with \
                 numDocsWithField={} but the segment has maxDoc={}",
                entry.num_docs_with_field, si.doc_count
            ));
        }

        // One forward-only `DisiCursor` for a sparse field rather than
        // `norms::norm_value` per document: that helper builds a fresh
        // cursor per lookup, which would make this O(maxDoc x blocks).
        let sparse_region = if entry.is_empty_field() || entry.is_dense() {
            None
        } else {
            // Both halves come straight off `.nvm`, so the *sum* is as
            // untrusted as the operands: `offset + length` on two `i64`s a
            // corrupt file chose overflows before any range is built, and an
            // overflow here is a panic inside the verifier rather than a
            // reported corruption. `norms::sparse_region` is the one place
            // that rule lives -- the read path uses the same helper, so the
            // two cannot drift.
            match norms::sparse_region(&nvd, entry) {
                Ok(region) => Some(region),
                Err(_) => {
                    decode_problems.push(format!(
                        "the .nvm entry's docs-with-field region [{}, +{}) is past the end of a \
                         {} byte .nvd",
                        entry.docs_with_field_offset,
                        entry.docs_with_field_length,
                        nvd.len()
                    ));
                    checks.push(named_field_check(
                        &format!("norms.values_decode:{}", fi.name),
                        &decode_problems,
                        si.doc_count as i64,
                        "docs",
                    ));
                    continue;
                }
            }
        };
        let mut cursor = sparse_region.map(|region| {
            lucene_codecs::indexed_disi::DisiCursor::new(
                region,
                entry.dense_rank_power,
                entry.jump_table_entry_count,
            )
        });

        for doc in 0..si.doc_count {
            let ordinal = if entry.is_empty_field() {
                None
            } else if entry.is_dense() {
                Some(doc as i64)
            } else {
                match cursor
                    .as_mut()
                    .expect("sparse entry has a cursor")
                    .advance_exact(doc)
                {
                    Ok(o) => o.map(|o| o as i64),
                    Err(e) => {
                        decode_problems.push(format!("docID={doc}: {e}"));
                        break;
                    }
                }
            };
            let norm = match ordinal {
                Some(ordinal) => match norms::read_value_at_ordinal(&nvd, entry, ordinal) {
                    Ok(v) => {
                        docs_with_norm = docs_with_norm.saturating_add(1);
                        Some(v)
                    }
                    Err(e) => {
                        decode_problems.push(format!("docID={doc}: {e}"));
                        None
                    }
                },
                None => None,
            };

            let Some(visited) = visited else { continue };
            if !is_live(live_docs, doc) {
                continue;
            }
            let has_terms = visited.get(doc as usize);
            if has_terms {
                expected_count = expected_count.saturating_add(1);
            }
            match norm {
                Some(norm) if norm != 0 => {
                    actual_count = actual_count.saturating_add(1);
                    if !has_terms {
                        agree_problems.push(format!(
                            "docID={doc} has no terms according to the postings but its norm is \
                             {norm}, which is not zero"
                        ));
                    }
                }
                Some(_) if has_terms => agree_problems.push(format!(
                    "docID={doc} has terms according to the postings but its norm is 0, which \
                     may only be used on documents that have no terms"
                )),
                _ => {}
            }
        }

        if docs_with_norm != entry.num_docs_with_field as i64 {
            decode_problems.push(format!(
                ".nvm records numDocsWithField={} but {docs_with_norm} docs actually carry a norm",
                entry.num_docs_with_field
            ));
        }
        checks.push(named_field_check(
            &format!("norms.values_decode:{}", fi.name),
            &decode_problems,
            si.doc_count as i64,
            "docs",
        ));

        if visited.is_some() {
            if expected_count != actual_count {
                agree_problems.push(format!(
                    "actual norm count: {actual_count} but expected: {expected_count}"
                ));
            }
            checks.push(named_field_check(
                &format!("norms.agree_with_postings:{}", fi.name),
                &agree_problems,
                si.doc_count as i64,
                "docs",
            ));
        }
    }
}

/// `CheckIndex.testVectors`, plus the graph checks in [`check_hnsw_graphs`]
/// (which are **this port's own addition**, not a port of 10.5.0 -- see
/// there).
///
/// `.vec`/`.vemf` are `Lucene99FlatVectorsFormat` bytes and `.vem`/`.vex`
/// `Lucene99HnswVectorsFormat` ones.
///
/// `testVectors`:
///
/// - Java's `dimension <= 0` guard is *not* ported: it is unfalsifiable in
///   Java as well as here (see the comment at its site), and this module ships
///   no check that can only pass.
/// - `vectors.field_entry_matches_fnm:<f>` -- the `.vemf` entry's
///   dimension/encoding/similarity agreeing with `.fnm`'s.
/// - `vectors.values_decode:<f>` -- every one of the field's `size`
///   ordinals decodes to a vector of exactly the declared dimension
///   (`checkFloatVectorValues`/`checkByteVectorValues`). Honest note: like
///   Java's, this is one property, not two -- the `count != values.size()`
///   clause can only fire when the per-ordinal decode already failed, and
///   the per-ordinal decode itself is unfalsifiable by byte corruption
///   because `.vemf`'s `size * dim * byteSize == vectorDataLength` guard
///   rejects an inconsistent entry at open. It guards a *writer*, which is
///   the regression it exists for.
/// - `vectors.ord_to_doc:<f>` -- every ordinal's document is in
///   `0..maxDoc` and strictly increasing in ordinal order. Java gets this
///   for free from `DocIdSetIterator`'s own contract; here the sparse
///   ord-to-doc mapping is decoded, so it is worth asserting.
///
/// The graph checks, when the segment has a `.vex` -- an addition beyond
/// 10.5.0, whose `CheckIndex` has none (see [`check_hnsw_graphs`]):
///
/// - `hnsw.neighbors_on_level:<f>` -- every neighbour of a node on level L
///   is itself on level L.
/// - `hnsw.neighbors_sorted:<f>` -- a node's neighbours are strictly
///   increasing (Java rejects both out-of-order and repeated neighbours).
/// - `hnsw.entry_point_reachable:<f>` -- Java *reports* connectedness
///   (`N/M connected`) without failing on it, and neither do we: the ratio
///   goes in the passing message. The one exception is the degenerate case
///   Java's report would show as `1/N` -- an entry point that reaches
///   nothing but itself on a level with more than one node, i.e. a graph
///   whose search can never return more than one document. Java's
///   `node < 0 || node > size - 1` guard is deliberately *not* repeated:
///   `read_field_entry` rejects an out-of-range level-node ordinal while
///   parsing `.vem`, so a check here could never fire.
fn check_vectors(
    dir: &dyn Directory,
    commit: &SegmentCommitInfo,
    si: &SegmentInfo,
    field_infos: &FieldInfos,
    stats: &mut CheckStats,
    checks: &mut Vec<Check>,
) {
    let with_vectors: Vec<&field_infos::FieldInfo> = field_infos
        .fields
        .iter()
        .filter(|f| f.vector_dimension != 0)
        .collect();
    if with_vectors.is_empty() {
        return;
    }
    // Fields a retired 9.0-9.8 HNSW format wrote (`PerFieldKnnVectorsFormat`
    // names it) keep their vectors in a `.vem`/`.vec`/`.vex` triple of their
    // own; each (format, suffix) group is checked through its own reader.
    let (retired, with_vectors): (Vec<_>, Vec<_>) = with_vectors
        .into_iter()
        .partition(|fi| retired_vectors_format(fi).is_some());
    let mut groups: Vec<(RetiredHnswFormat, String)> = Vec::new();
    for fi in &retired {
        if let Some(group) = retired_vectors_format(fi) {
            if !groups.contains(&group) {
                groups.push(group);
            }
        }
    }
    for (format, suffix) in groups {
        let fields: Vec<&field_infos::FieldInfo> = retired
            .iter()
            .copied()
            .filter(|fi| retired_vectors_format(fi).is_some_and(|g| g == (format, suffix.clone())))
            .collect();
        check_retired_vectors(dir, commit, si, &fields, format, &suffix, stats, checks);
    }
    // The same for the retired quantized formats (Lucene 9.9-10.3), whose
    // groups hold raw vectors, codes and (for an HNSW wrapper) a graph.
    let (quantized, with_vectors): (Vec<_>, Vec<_>) = with_vectors
        .into_iter()
        .partition(|fi| quantized_vectors_format(fi).is_some());
    let mut qgroups: Vec<(QuantizedFormat, String)> = Vec::new();
    for fi in &quantized {
        if let Some(group) = quantized_vectors_format(fi) {
            if !qgroups.contains(&group) {
                qgroups.push(group);
            }
        }
    }
    for (format, suffix) in qgroups {
        let fields: Vec<&field_infos::FieldInfo> = quantized
            .iter()
            .copied()
            .filter(|fi| {
                quantized_vectors_format(fi).is_some_and(|g| g == (format, suffix.clone()))
            })
            .collect();
        check_quantized_vectors(
            dir,
            commit,
            si,
            field_infos,
            &fields,
            format,
            &suffix,
            stats,
            checks,
        );
    }
    if with_vectors.is_empty() {
        return;
    }
    // The current formats' groups (`Lucene99HnswVectorsFormat`,
    // `Lucene104HnswScalarQuantizedVectorsFormat`,
    // `Lucene104ScalarQuantizedVectorsFormat`): each field is read through
    // the instance its `PerFieldKnnVectorsFormat` attributes name, as
    // `FieldsReader` opens one reader per suffix. A field without the
    // attributes is read through the first current group the segment has.
    let listed_groups =
        crate::index_writer::current_vector_suffixes(&si.files, &commit.segment_name);
    let mut groups: Vec<(String, Vec<&field_infos::FieldInfo>)> = Vec::new();
    for fi in with_vectors {
        let attr = |key: &str| {
            fi.attributes
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v.as_str())
        };
        let suffix = match (
            attr("PerFieldKnnVectorsFormat.format"),
            attr("PerFieldKnnVectorsFormat.suffix"),
        ) {
            (Some(format), Some(n)) => format!("{format}_{n}"),
            _ => listed_groups.first().cloned().unwrap_or_default(),
        };
        match groups.iter_mut().find(|(s, _)| *s == suffix) {
            Some((_, fields)) => fields.push(fi),
            None => groups.push((suffix, vec![fi])),
        }
    }
    for (suffix, fields) in groups {
        check_current_vector_group(dir, commit, si, &fields, &suffix, stats, checks);
    }
}

/// [`check_vectors`] for one current-format `PerFieldKnnVectorsFormat`
/// instance: the flat checks over its raw vectors, `vectors.quantized:<f>`
/// for a scalar-quantized one (codes for exactly the raw vectors' ordinals,
/// each mapped to the same document), and the graph checks when it has a
/// graph.
fn check_current_vector_group(
    dir: &dyn Directory,
    commit: &SegmentCommitInfo,
    si: &SegmentInfo,
    fields: &[&field_infos::FieldInfo],
    suffix: &str,
    stats: &mut CheckStats,
    checks: &mut Vec<Check>,
) {
    let name = |ext: &str| match suffix {
        // A codec whose vectors format is not per-field (`ConfigurableMCodec`).
        "" => format!("{}.{ext}", commit.segment_name),
        _ => format!("{}_{suffix}.{ext}", commit.segment_name),
    };
    let (vec_name, vemf_name) = (name("vec"), name("vemf"));
    if !si.files.contains(&vec_name) || !si.files.contains(&vemf_name) {
        checks.push(Check::fail(
            "vectors.open",
            format!(
                "a field declares vector values but the segment has no .vec/.vemf files for \
                 {suffix:?}"
            ),
        ));
        skip_families(checks, VECTOR_FAMILIES, "vectors.open");
        return;
    }
    let opened = (|| -> Result<
        (
            lucene_store::directory::Input,
            lucene_store::directory::Input,
        ),
        String,
    > {
        let vemf = dir.open(&vemf_name).map_err(|e| e.to_string())?;
        let vec = dir.open(&vec_name).map_err(|e| e.to_string())?;
        Ok((vemf, vec))
    })();
    let (vemf, vec) = match opened {
        Ok(v) => v,
        Err(e) => {
            checks.push(Check::fail("vectors.open", e));
            skip_families(checks, VECTOR_FAMILIES, "vectors.open");
            return;
        }
    };
    // `open` only retrieves the `.vec` checksum, as Java's reader does;
    // `checkIntegrity` is what re-reads the whole file, and a damaged `.vec`
    // is this subsystem's own failure to report.
    let flat = match vectors::FlatVectorsReader::open(&vemf, &vec, &commit.segment_id, suffix)
        .and_then(|r| r.check_integrity().map(|()| r))
        .and_then(|mut r| {
            r.resolve_similarities(|number| {
                fields
                    .iter()
                    .find(|f| f.number == number)
                    .map(|f| f.vector_similarity_function)
            })
            .map(|()| r)
        }) {
        Ok(r) => r,
        Err(e) => {
            checks.push(Check::fail("vectors.open", e.to_string()));
            skip_families(checks, VECTOR_FAMILIES, "vectors.open");
            return;
        }
    };

    check_flat_vector_fields(&flat, fields, si, stats, checks);

    if si.files.contains(&name("vemq")) {
        check_current_quantized(dir, commit, &flat, fields, suffix, &name, checks);
    }

    check_hnsw_graphs(dir, commit, si, fields, suffix, checks);
}

/// `vectors.quantized:<field>` for a `Lucene104` scalar-quantized instance:
/// the `.vemq`/`.veq` open, and every `FLOAT32` field has codes for exactly
/// the raw vectors' ordinals, each readable and mapped to the same document.
fn check_current_quantized(
    dir: &dyn Directory,
    commit: &SegmentCommitInfo,
    flat: &vectors::FlatVectorsReader<'_>,
    fields: &[&field_infos::FieldInfo],
    suffix: &str,
    name: &dyn Fn(&str) -> String,
    checks: &mut Vec<Check>,
) {
    let opened = dir
        .open(&name("vemq"))
        .and_then(|vemq| Ok((vemq, dir.open(&name("veq"))?)))
        .map_err(|e| e.to_string());
    let reader = opened
        .as_ref()
        .map_err(Clone::clone)
        .and_then(|(vemq, veq)| {
            lucene_codecs::scalar_quantized_vectors::ScalarQuantizedVectorsReader::open(
                vemq,
                veq,
                &commit.segment_id,
                suffix,
            )
            .map_err(|e| e.to_string())
        });
    for fi in fields
        .iter()
        .filter(|f| f.vector_encoding == field_infos::VectorEncoding::Float32)
    {
        let mut size = 0i64;
        let result = reader.as_ref().map_err(Clone::clone).and_then(|q| {
            let codes = q
                .quantized_vector_values(fi.number)
                .map_err(|e| e.to_string())?;
            let raw = flat
                .float_vector_values(fi.number)
                .map_err(|e| e.to_string())?;
            size = i64::from(codes.size());
            if codes.size() != raw.size() {
                return Err(format!(
                    ".vemq holds {} codes but .vemf {} vectors",
                    codes.size(),
                    raw.size()
                ));
            }
            for ord in 0..codes.size() {
                codes.vector(ord).map_err(|e| format!("ord={ord}: {e}"))?;
                codes
                    .corrective_terms(ord)
                    .map_err(|e| format!("ord={ord}: {e}"))?;
                let a = codes.ord_to_doc(ord).map_err(|e| e.to_string())?;
                let b = raw.ord_to_doc(ord).map_err(|e| e.to_string())?;
                if a != b {
                    return Err(format!(
                        "ord={ord}: codes map to doc {a}, raw vectors to doc {b}"
                    ));
                }
            }
            Ok(())
        });
        let problems: Vec<String> = result.err().into_iter().collect();
        checks.push(named_field_check(
            &format!("vectors.quantized:{}", fi.name),
            &problems,
            size,
            "vectors",
        ));
    }
}

/// The per-field half of [`check_vectors`], over whichever reader serves the
/// fields' vectors: `vectors.field_entry_matches_fnm`,
/// `vectors.values_decode` and `vectors.ord_to_doc`.
fn check_flat_vector_fields(
    flat: &vectors::FlatVectorsReader<'_>,
    with_vectors: &[&field_infos::FieldInfo],
    si: &SegmentInfo,
    stats: &mut CheckStats,
    checks: &mut Vec<Check>,
) {
    for fi in with_vectors {
        let name = &fi.name;
        // Java's `dimension <= 0` guard is deliberately *not* reproduced here.
        // It is unfalsifiable in this port and, on inspection, in Java too:
        // `hasVectorValues()` is `vectorDimension > 0`, so the loop it guards
        // can never see a non-positive dimension, and `FieldInfo`'s own
        // constructor rejects a negative one before that. This port has the
        // same two facts -- `with_vectors` filters `vector_dimension != 0`,
        // and `field_infos::FieldInfo::check_consistency` (a direct port of
        // Java's) rejects `vector_dimension < 0` inside `field_infos::parse`,
        // which is where `fnm.open` gets this `FieldInfos` from. Shipping it
        // would be a check that can only ever pass.
        let Some(entry) = flat.field(fi.number) else {
            checks.push(Check::fail(
                format!("vectors.field_entry_matches_fnm:{name}"),
                format!("field {name:?} has vector values in .fnm but no .vemf entry"),
            ));
            // No entry means no `size`, no offsets and no ordinal space, so
            // every vector of this field goes unread.
            skip_families(
                checks,
                &[
                    &format!("vectors.values_decode:{name}"),
                    &format!("vectors.ord_to_doc:{name}"),
                ],
                &format!("vectors.field_entry_matches_fnm:{name}"),
            );
            continue;
        };
        let mut entry_problems: Vec<String> = Vec::new();
        if entry.dimension != fi.vector_dimension {
            entry_problems.push(format!(
                ".vemf says dimension={} but .fnm says {}",
                entry.dimension, fi.vector_dimension
            ));
        }
        if entry.encoding != fi.vector_encoding {
            entry_problems.push(format!(
                ".vemf says encoding={:?} but .fnm says {:?}",
                entry.encoding, fi.vector_encoding
            ));
        }
        if entry.similarity != fi.vector_similarity_function {
            entry_problems.push(format!(
                ".vemf says similarity={:?} but .fnm says {:?}",
                entry.similarity, fi.vector_similarity_function
            ));
        }
        checks.push(named_field_check(
            &format!("vectors.field_entry_matches_fnm:{name}"),
            &entry_problems,
            1,
            "field",
        ));

        stats.vector_fields = stats.vector_fields.saturating_add(1);
        let size = entry.size;
        let mut decode_problems: Vec<String> = Vec::new();
        let mut ord_problems: Vec<String> = Vec::new();
        let mut counted = 0i64;
        // The two `Err(e) => decode_problems.push(..)` arms below, and the
        // byte branch's per-ordinal `Err`, **cannot fire today**:
        // `flat.field()` already returned `Some` and this `match` is on that
        // entry's own encoding, so `{float,byte}_vector_values` can only fail
        // on the `vectorDataOffset + vectorDataLength <= .vec length` bound
        // that `read_field_entry` has already proved; and `bytes(ord, ..)`
        // rejects only an out-of-range ordinal, which `0..size` excludes.
        //
        // They are kept, unlike never-firing `Check::fail` arms. The distinction
        // is what the arm *claims*: a `Check::fail` that cannot fire is a
        // false claim of coverage, so it is deleted; an `Err` arm that cannot
        // fire is total error handling over a `Result` the decoder's
        // signature forces this caller to handle, and removing it would mean
        // ignoring the error or unwrapping it in a verifier whose one
        // contract is not to panic. The float branch's `vector_into` error
        // *is* firable, on a `.fnm`/`.vemf` dimension disagreement, and the
        // byte branch reports that same disagreement through the length
        // comparison instead -- which is the asymmetry the two branches have.
        match entry.encoding {
            field_infos::VectorEncoding::Float32 => match flat.float_vector_values(fi.number) {
                Ok(values) => {
                    let mut buf = vec![0f32; fi.vector_dimension as usize];
                    for ord in 0..size {
                        match values.vector_into(ord, &mut buf) {
                            Ok(()) => counted = counted.saturating_add(1),
                            Err(e) => {
                                decode_problems.push(format!("ord={ord}: {e}"));
                                break;
                            }
                        }
                    }
                    check_ord_to_doc(&values, size, si.doc_count, &mut ord_problems);
                }
                Err(e) => decode_problems.push(e.to_string()),
            },
            field_infos::VectorEncoding::Byte => match flat.byte_vector_values(fi.number) {
                Ok(values) => {
                    for ord in 0..size {
                        match values.vector(ord) {
                            Ok(v) => {
                                if v.len() != fi.vector_dimension as usize {
                                    decode_problems.push(format!(
                                        "ord={ord}: vector has {} bytes, not the field's \
                                         dimension {}",
                                        v.len(),
                                        fi.vector_dimension
                                    ));
                                }
                                counted = counted.saturating_add(1);
                            }
                            Err(e) => {
                                decode_problems.push(format!("ord={ord}: {e}"));
                                break;
                            }
                        }
                    }
                    check_ord_to_doc(&values, size, si.doc_count, &mut ord_problems);
                }
                Err(e) => decode_problems.push(e.to_string()),
            },
        }
        if counted != size as i64 {
            decode_problems.push(format!(
                "field has size={size} but when iterated, returns {counted} vectors"
            ));
        }
        stats.vector_values = stats.vector_values.saturating_add(counted);
        checks.push(named_field_check(
            &format!("vectors.values_decode:{name}"),
            &decode_problems,
            size as i64,
            "vectors",
        ));
        checks.push(named_field_check(
            &format!("vectors.ord_to_doc:{name}"),
            &ord_problems,
            size as i64,
            "vectors",
        ));
    }
}

/// The retired HNSW format and file suffix a vector field was written with,
/// from its `PerFieldKnnVectorsFormat` attributes; `None` for any other
/// format (or a field without the attributes).
fn retired_vectors_format(fi: &field_infos::FieldInfo) -> Option<(RetiredHnswFormat, String)> {
    let attr = |key: &str| {
        fi.attributes
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    };
    let format = RetiredHnswFormat::for_name(attr("PerFieldKnnVectorsFormat.format")?)?;
    Some((format, attr("PerFieldKnnVectorsFormat.suffix")?.to_string()))
}

/// The retired quantized format and file suffix a vector field was written
/// with, from its `PerFieldKnnVectorsFormat` attributes; `None` for any other.
fn quantized_vectors_format(fi: &field_infos::FieldInfo) -> Option<(QuantizedFormat, String)> {
    let attr = |key: &str| {
        fi.attributes
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    };
    let format = QuantizedFormat::for_name(attr("PerFieldKnnVectorsFormat.format")?)?;
    Some((format, attr("PerFieldKnnVectorsFormat.suffix")?.to_string()))
}

/// [`check_vectors`] for fields a retired quantized format wrote: the flat
/// checks over the raw vectors, the graph checks for an HNSW wrapper, and
/// `vectors.quantized:<field>` -- the quantized file's checksum
/// (`checkIntegrity`), and for every `FLOAT32` field codes for exactly the
/// raw vectors' ordinals, each readable and mapped to the same document.
#[allow(clippy::too_many_arguments)]
fn check_quantized_vectors(
    dir: &dyn Directory,
    commit: &SegmentCommitInfo,
    si: &SegmentInfo,
    field_infos: &FieldInfos,
    fields: &[&field_infos::FieldInfo],
    format: QuantizedFormat,
    suffix: &str,
    stats: &mut CheckStats,
    checks: &mut Vec<Check>,
) {
    let segment_suffix = format!("{}_{suffix}", format.name());
    let name = |ext: &str| format!("{}_{segment_suffix}.{ext}", commit.segment_name);
    let (meta_ext, data_ext) = format.quantized_extensions();
    let open = |ext: &str| dir.open(&name(ext)).map_err(|e| e.to_string());
    let opened = (|| {
        let graph = if format.has_graph() {
            Some((open("vem")?, open("vex")?))
        } else {
            None
        };
        Ok::<_, String>((
            open("vemf")?,
            open("vec")?,
            open(meta_ext)?,
            open(data_ext)?,
            graph,
        ))
    })();
    let (vemf, vec, qmeta, qdata, graph) = match opened {
        Ok(v) => v,
        Err(e) => {
            checks.push(Check::fail("vectors.open", e));
            skip_families(checks, VECTOR_FAMILIES, "vectors.open");
            return;
        }
    };
    let reader = match QuantizedVectorsReader::open(
        format,
        QuantizedFiles {
            vemf: &vemf,
            vec: &vec,
            quantized_meta: &qmeta,
            quantized_data: &qdata,
            graph: graph.as_ref().map(|(m, x)| (&m[..], &x[..])),
        },
        field_infos,
        &commit.segment_id,
        &segment_suffix,
    ) {
        Ok(r) => r,
        Err(e) => {
            checks.push(Check::fail("vectors.open", e.to_string()));
            skip_families(checks, VECTOR_FAMILIES, "vectors.open");
            return;
        }
    };
    check_flat_vector_fields(reader.flat(), fields, si, stats, checks);
    let integrity = reader.check_integrity().err().map(|e| e.to_string());
    for fi in fields {
        let mut problems: Vec<String> = integrity.iter().cloned().collect();
        if fi.vector_encoding == field_infos::VectorEncoding::Float32 {
            if let Err(e) = check_quantized_field(&reader, fi.number) {
                problems.push(e);
            }
        }
        checks.push(named_field_check(
            &format!("vectors.quantized:{}", fi.name),
            &problems,
            1,
            "field",
        ));
        if let Some(graphs) = reader.graph() {
            match graphs.graph(fi.number) {
                Ok(Some(graph)) => check_one_graph(&graph, &fi.name, true, checks),
                Ok(None) => {}
                Err(e) => checks.push(Check::fail(format!("hnsw.open:{}", fi.name), e.to_string())),
            }
        }
    }
}

/// One `FLOAT32` field of a quantized group: as many codes as raw vectors,
/// each ordinal's codes and corrections readable, on the same document.
fn check_quantized_field(reader: &QuantizedVectorsReader<'_>, field: i32) -> Result<(), String> {
    let raw = reader
        .flat()
        .float_vector_values(field)
        .map_err(|e| e.to_string())?;
    // Every ordinal's codes and corrections decoded, and its document.
    let docs: Vec<i32> = match reader.quantized() {
        QuantizedReader::Scalar(r) => {
            let v = r
                .quantized_vector_values(field)
                .map_err(|e| e.to_string())?;
            let mut codes = Vec::new();
            (0..v.size())
                .map(|ord| {
                    v.vector_into(ord, &mut codes)?;
                    v.score_correction_constant(ord)?;
                    v.ord_to_doc(ord)
                })
                .collect::<Result<_, _>>()
                .map_err(|e| e.to_string())?
        }
        QuantizedReader::Binary(r) => {
            let v = r
                .binarized_vector_values(field)
                .map_err(|e| e.to_string())?;
            (0..v.size())
                .map(|ord| {
                    v.vector(ord)?;
                    v.corrective_terms(ord)?;
                    v.ord_to_doc(ord)
                })
                .collect::<Result<_, _>>()
                .map_err(|e| e.to_string())?
        }
    };
    let size = docs.len() as i32;
    if size != raw.size() {
        return Err(format!(
            "{size} quantized vectors for {} raw ones",
            raw.size()
        ));
    }
    for (ord, &q) in (0..size).zip(&docs) {
        let r = raw.ord_to_doc(ord).map_err(|e| e.to_string())?;
        if q != r {
            return Err(format!("ord={ord}: quantized doc {q}, raw doc {r}"));
        }
    }
    Ok(())
}

/// [`check_vectors`] for fields a retired `Lucene90`..`Lucene95` HNSW format
/// wrote: the same flat checks over the vectors its `.vem` describes, and the
/// same graph checks over its `.vex`.
#[allow(clippy::too_many_arguments)]
fn check_retired_vectors(
    dir: &dyn Directory,
    commit: &SegmentCommitInfo,
    si: &SegmentInfo,
    fields: &[&field_infos::FieldInfo],
    format: RetiredHnswFormat,
    suffix: &str,
    stats: &mut CheckStats,
    checks: &mut Vec<Check>,
) {
    let segment_suffix = format!("{}_{suffix}", format.name());
    let name = |ext: &str| format!("{}_{segment_suffix}.{ext}", commit.segment_name);
    let opened = (|| -> Result<
        (
            lucene_store::directory::Input,
            lucene_store::directory::Input,
            lucene_store::directory::Input,
        ),
        String,
    > {
        let vem = dir.open(&name("vem")).map_err(|e| e.to_string())?;
        let vec = dir.open(&name("vec")).map_err(|e| e.to_string())?;
        let vex = dir.open(&name("vex")).map_err(|e| e.to_string())?;
        Ok((vem, vec, vex))
    })();
    let (vem, vec, vex) = match opened {
        Ok(v) => v,
        Err(e) => {
            checks.push(Check::fail("vectors.open", e));
            skip_families(checks, VECTOR_FAMILIES, "vectors.open");
            return;
        }
    };
    let reader = match RetiredHnswVectorsReader::open(
        format,
        &vem,
        &vec,
        &vex,
        &commit.segment_id,
        &segment_suffix,
    ) {
        Ok(r) => r,
        Err(e) => {
            checks.push(Check::fail("vectors.open", e.to_string()));
            skip_families(checks, VECTOR_FAMILIES, "vectors.open");
            return;
        }
    };
    check_flat_vector_fields(reader.flat(), fields, si, stats, checks);
    for fi in fields {
        match reader.graph(fi.number) {
            Ok(graph) => check_one_graph(&graph, &fi.name, format.neighbors_sorted(), checks),
            Err(e) => checks.push(Check::fail(format!("hnsw.open:{}", fi.name), e.to_string())),
        }
    }
}

/// The per-field families a failed `vectors.open` takes down. The graph
/// checks go with them: `check_hnsw_graphs` is only reached from the bottom
/// of `check_vectors`.
const VECTOR_FAMILIES: &[&str] = &[
    "vectors.field_entry_matches_fnm",
    "vectors.values_decode",
    "vectors.ord_to_doc",
    "hnsw.neighbors_on_level",
    "hnsw.neighbors_sorted",
    "hnsw.entry_point_reachable",
];

/// `values.ordToDoc(ord)` for every ordinal: in `0..maxDoc` and strictly
/// increasing, which is what makes the flat vector store addressable as a
/// `DocIdSetIterator` at all.
fn check_ord_to_doc<V: VectorOrdToDoc>(
    values: &V,
    size: i32,
    max_doc: i32,
    problems: &mut Vec<String>,
) {
    let mut prev = -1i32;
    for ord in 0..size {
        match values.ord_to_doc_at(ord) {
            Ok(doc) => {
                if doc <= prev || doc >= max_doc {
                    problems.push(format!(
                        "ord={ord} maps to docID={doc}, which is not in the valid \
                         strictly-increasing 0..{max_doc} range (previous was {prev})"
                    ));
                    break;
                }
                prev = doc;
            }
            Err(e) => {
                problems.push(format!("ord={ord}: {e}"));
                break;
            }
        }
    }
}

/// The one method [`check_ord_to_doc`] needs from either vector-values
/// flavour; `lucene-codecs` generates `ord_to_doc` on both types via a macro
/// rather than a shared trait, so this is the local adapter.
trait VectorOrdToDoc {
    fn ord_to_doc_at(&self, ord: i32) -> Result<i32, String>;
}

impl VectorOrdToDoc for vectors::FloatVectorValues<'_> {
    fn ord_to_doc_at(&self, ord: i32) -> Result<i32, String> {
        self.ord_to_doc(ord).map_err(|e| e.to_string())
    }
}

impl VectorOrdToDoc for vectors::ByteVectorValues<'_> {
    fn ord_to_doc_at(&self, ord: i32) -> Result<i32, String> {
        self.ord_to_doc(ord).map_err(|e| e.to_string())
    }
}

/// HNSW graph structure checks, for every vector field whose segment carries
/// a `.vex` graph.
///
/// **Not a port of 10.5.0.** The pinned version's `CheckIndex` has no HNSW
/// check of any kind; `testHnswGraphs`/`testHnswGraph` were added to Lucene
/// after 10.5.0. These stay because
/// they are diagnostic-only, cost nothing a caller does not ask for, and do
/// catch real corruption -- `corrupted_hnsw_graph_bytes_are_never_silently_accepted`
/// and `corrupted_hnsw_neighbours_are_caught_by_the_graph_checks` are their
/// evidence -- and the three properties checked here are the same ones
/// Lucene's own later `testHnswGraph` settled on. Nothing below describes
/// 10.5.0 behaviour.
fn check_hnsw_graphs(
    dir: &dyn Directory,
    commit: &SegmentCommitInfo,
    si: &SegmentInfo,
    with_vectors: &[&field_infos::FieldInfo],
    suffix: &str,
    checks: &mut Vec<Check>,
) {
    // This instance's graph files; none is a flat (exhaustive-search)
    // format, not a defect.
    let own = |ext: &str| {
        let want = format!("{}_{suffix}.{ext}", commit.segment_name);
        si.files.iter().find(|f| **f == want)
    };
    let (vem_name, vex_name) = (own("vem"), own("vex"));
    let (Some(vem_name), Some(vex_name)) = (vem_name, vex_name) else {
        return;
    };
    let opened = (|| -> Result<
        (
            lucene_store::directory::Input,
            lucene_store::directory::Input,
        ),
        String,
    > {
        let vem = dir.open(vem_name).map_err(|e| e.to_string())?;
        let vex = dir.open(vex_name).map_err(|e| e.to_string())?;
        Ok((vem, vex))
    })();
    let (vem, vex) = match opened {
        Ok(v) => v,
        Err(e) => {
            checks.push(Check::fail("hnsw.open", e));
            skip_families(checks, HNSW_FAMILIES, "hnsw.open");
            return;
        }
    };
    let reader = match hnsw_vectors::HnswVectorsReader::open(&vem, &vex, &commit.segment_id, suffix)
    {
        Ok(r) => r,
        Err(e) => {
            checks.push(Check::fail("hnsw.open", e.to_string()));
            skip_families(checks, HNSW_FAMILIES, "hnsw.open");
            return;
        }
    };

    for fi in with_vectors {
        let name = &fi.name;
        let graph = match reader.graph(fi.number) {
            // `HnswGraph.EMPTY`: the field exists but carries no graph.
            Ok(None) => continue,
            Ok(Some(g)) => g,
            Err(e) => {
                checks.push(Check::fail(format!("hnsw.open:{name}"), e.to_string()));
                skip_families(
                    checks,
                    &HNSW_FAMILIES
                        .iter()
                        .map(|f| format!("{f}:{name}"))
                        .collect::<Vec<_>>()
                        .iter()
                        .map(String::as_str)
                        .collect::<Vec<_>>(),
                    &format!("hnsw.open:{name}"),
                );
                continue;
            }
        };
        check_one_graph(&graph, name, true, checks);
    }
}

/// The graph checks for one field's graph, whatever format it came from.
/// `sorted` says whether the format's writer sorted each neighbour list:
/// `Lucene99`/`Lucene95`/`Lucene90` store deltas and did; `Lucene91`..`94`
/// stored a node's `NeighborArray` in score order, so for those a repeat is
/// looked for in a sorted copy instead.
fn check_one_graph<G: HnswGraphView>(graph: &G, name: &str, sorted: bool, checks: &mut Vec<Check>) {
    let size = graph.size();
    let mut level_problems = Problems::new();
    let mut order_problems = Problems::new();
    let mut reachable = String::new();
    let mut degenerate: Vec<String> = Vec::new();
    let mut neighbors: Vec<i32> = Vec::new();

    for level in (0..graph.num_levels()).rev() {
        // Same class as `check_vectors`' `Err` arms above: unreachable
        // (this loop's bound is `graph.num_levels()`, and
        // `read_field_entry` sizes `nodes_by_level` to exactly that), but
        // kept because it is error handling rather than a claimed check.
        let nodes = match graph.sorted_nodes_on_level(level) {
            Ok(n) => n,
            Err(e) => {
                level_problems.push(format!("level {level}: {e}"));
                continue;
            }
        };
        let mut on_this_level = lucene_util::fixed_bit_set::FixedBitSet::new(size.max(0) as usize);
        for &n in &nodes {
            // Java's `node < 0 || node > size - 1` guard is not repeated
            // here: `read_field_entry` already rejects an out-of-range
            // level-node ordinal while parsing `.vem` (this port
            // validates on the way in where Java validates on the way
            // out), and level 0's node set is the implicit `0..size`.
            // The guard that *can* fire is the neighbour one below.
            // FBS: `on_this_level` is `FixedBitSet::new(size.max(0))` a few
            // lines above, so `size` is its `len()`.
            if n >= 0 && n < size {
                on_this_level.set(n as usize);
            }
        }
        for &node in &nodes {
            // No `node < 0 || node >= size` guard. Level 0's node set is
            // literally `0..size`, and `read_field_entry` validates an
            // upper level's first node, its deltas and its last node
            // against `size` while parsing `.vem` -- modulo the plain
            // `i32` accumulator it builds them with, which a release
            // build lets wrap, so an *interior* node is validated only
            // transitively. Either way the guard was the wrong response:
            // a node that turns out to be unusable is reported by
            // `neighbors_into` below (which range-checks it again), where
            // the guard silently skipped it and reported nothing.
            if let Err(e) = graph.neighbors_into(level, node, &mut neighbors) {
                level_problems.push(format!("field {name:?} node {node} level {level}: {e}"));
                continue;
            }
            if !sorted {
                // Score order on disk: look for the repeat in ordinal order.
                neighbors.sort_unstable();
            }
            let mut last = -1i32;
            for &nbr in &neighbors {
                // FBS: `on_this_level` is `FixedBitSet::new(size.max(0))`
                // in this same function, so `size` is its `len()`.
                if nbr < 0 || nbr >= size || !on_this_level.get(nbr as usize) {
                    level_problems.push(format!(
                        "field {name:?} has node {node} with a neighbor {nbr} which is not \
                         on its level ({level})"
                    ));
                }
                // Java rejects both out-of-order and repeated neighbours.
                // Only the repeat is falsifiable here: `neighbors_into`
                // decodes a neighbour list as a running sum of
                // *unsigned* deltas, so the list it hands back is
                // non-decreasing by construction and `nbr < last` cannot
                // happen. A zero delta still can, and that is the repeat.
                if nbr == last {
                    order_problems.push(format!(
                        "field {name:?} has repeated neighbors of node {node} with value \
                         {nbr}"
                    ));
                }
                last = nbr;
            }
        }

        // Connectedness from the entry point, Java's
        // `getConnectedNodesOnLevel`. Java computes it per level because
        // it *prints* one line per level; this reports level 0 only, so
        // the walk runs only there.
        if level == 0 {
            let connected = connected_nodes_on_level(graph, level, size);
            reachable = format!("{connected}/{} nodes reachable on level 0", nodes.len());
            // Java never fails on connectedness (it tolerates
            // historically-disconnected graphs) and neither do we, with
            // one exception: an entry point that reaches nothing *but
            // itself* on a level with more than one node is a graph
            // whose search can only ever return one document. That is a
            // corrupt or empty neighbour list, not a quality issue.
            if connected <= 1 && nodes.len() > 1 {
                degenerate.push(format!(
                    "field {name:?}: the level-0 entry point {} reaches {connected} of {} \
                     nodes -- no search of this graph can return more than that",
                    graph.entry_node(),
                    nodes.len()
                ));
            }
        }
    }

    checks.push(named_field_check(
        &format!("hnsw.neighbors_on_level:{name}"),
        &level_problems,
        size as i64,
        "nodes",
    ));
    checks.push(named_field_check(
        &format!("hnsw.neighbors_sorted:{name}"),
        &order_problems,
        size as i64,
        "nodes",
    ));
    if degenerate.is_empty() {
        checks.push(Check {
            name: format!("hnsw.entry_point_reachable:{name}"),
            outcome: Outcome::Passed,
            message: reachable,
        });
    } else {
        checks.push(named_field_check(
            &format!("hnsw.entry_point_reachable:{name}"),
            &degenerate,
            size as i64,
            "nodes",
        ));
    }
}

/// The families a failed `hnsw.open` takes down.
const HNSW_FAMILIES: &[&str] = &[
    "hnsw.neighbors_on_level",
    "hnsw.neighbors_sorted",
    "hnsw.entry_point_reachable",
];

/// Java's `getConnectedNodesOnLevel`: how many nodes a depth-first walk from
/// the entry point reaches on `level`.
///
/// There is no `entry < 0 || entry >= size` guard: `entry_node` is `0` for a single-level
/// graph and `nodes_by_level[top][0]` otherwise, and `read_field_entry`
/// validates that ordinal into `0..size` -- so the guard could only fire for
/// `size == 0`, and a `size == 0` graph never reaches this function at all.
/// A field with `numLevels <= 1` contributes `size` node offsets, so
/// `size == 0` leaves `numberOfOffsets == 0`, no `offsetsMeta`, and
/// `OffHeapHnswGraph::new` rejects the entry as "graph has data but no node
/// offsets" (reported as `hnsw.open:<field>`); a field with `numLevels >= 2`
/// needs `0 < numNodesOnLevel <= size` on its upper level, so `size >= 1`.
/// The loop below still range-checks every node it pops, which is what keeps
/// the walk total.
fn connected_nodes_on_level<G: HnswGraphView>(graph: &G, level: i32, size: i32) -> usize {
    let entry = graph.entry_node();
    let mut seen = lucene_util::fixed_bit_set::FixedBitSet::new(size.max(0) as usize);
    let mut stack = vec![entry];
    let mut neighbors: Vec<i32> = Vec::new();
    let mut count = 0usize;
    while let Some(node) = stack.pop() {
        // FBS: `seen` is `FixedBitSet::new(size.max(0))` at the top of this
        // function, so `size` is its `len()`.
        if node < 0 || node >= size || seen.get(node as usize) {
            continue;
        }
        seen.set(node as usize);
        count = count.saturating_add(1);
        if graph.neighbors_into(level, node, &mut neighbors).is_err() {
            continue;
        }
        stack.extend_from_slice(&neighbors);
    }
    count
}

mod postings_check;
use postings_check::check_postings;

#[cfg(test)]
mod tests;
