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
//! - Points-tree (BKD) structural invariants (revisited; previously
//!   deferred -- see below): for every field with points data (`.fnm`'s
//!   `point_dimension_count != 0`), walks the field's actual BKD tree
//!   leaf-by-leaf via [`lucene_codecs::points::PointsReader::decode_leaves`]
//!   (reusing this port's existing `.kdm`/`.kdi`/`.kdd` decoder, not a new
//!   parser) and checks: every point's packed value (over the index
//!   dimensions) falls within the field's own `.kdm`-declared
//!   `min_packed_value`/`max_packed_value`
//!   (`points.value_within_field_bounds:<field>`); for fields with more
//!   than one index dimension, each leaf's own embedded bounding box (see
//!   [`lucene_codecs::points::Leaf::bound`]) is itself a subset of that
//!   field-level bound (`points.leaf_bounds_subset_of_field:<field>`); and
//!   the leaves' decoded point counts sum to `.kdm`'s declared field-level
//!   `point_count` (`points.point_count_matches:<field>`).
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
//! ## Revisited scope decision: points-tree structural invariants
//!
//! Also **deliberately deferred** in task #57 for the same "separate,
//! per-format task" reason as postings, and closed here for the same
//! reason that blocker no longer holds: [`lucene_codecs::points`]'s
//! existing read-side decoder already does all the packed-index/leaf-block
//! parsing (used today by `lucene-search`'s points range queries, task
//! #199) -- only a per-leaf accessor
//! ([`lucene_codecs::points::PointsReader::decode_leaves`], added by this
//! task, alongside a small [`lucene_codecs::points::Leaf`] type) was
//! missing, plus actually reading (rather than skip-decoding) the
//! multi-index-dim per-leaf bounding box `read_leaf_block` had always
//! parsed past but discarded. Nothing about this check requires new BKD
//! parsing logic, the same precedent as postings re-derivation above.
//!
//! **What this catches**: a corrupted or buggy writer producing a `.kdd`
//! point value outside the field's own declared `.kdm` bounds (a
//! metadata/data disagreement across two separate files, exactly the class
//! of bug `postings.total_term_freq` catches for postings), a leaf whose
//! own embedded bounding box has grown looser than its parent field's
//! bound (a tree-invariant violation a range query's pruning silently
//! relies on being true), or a leaf whose actually-decoded point count
//! disagrees with `.kdm`'s own declared field-level `point_count`.
//!
//! **Known scope note**: `points.leaf_bounds_subset_of_field` only runs for
//! fields with more than one index dimension, since a single-index-dimension
//! leaf has no embedded bounding box on disk at all (see `points.rs`'s
//! `read_leaf_block` doc comment) -- there is nothing there to cross-check
//! beyond the field-wide bound `points.value_within_field_bounds` already
//! covers for every field regardless of dimensionality.
//!
//! ## Checks added in the `b11-index-meta` sweep
//!
//! The sweep against `org/apache/lucene/index/CheckIndex.java` found that a
//! number of Java's per-segment tests had no counterpart here, and a
//! verifier that does not check something silently passes corrupt segments.
//! Now also performed:
//!
//! - **`testStoredFields`'s actual document scan** (`stored_fields.every_doc_decodes`):
//!   every document is decoded, deleted ones included ("to make sure they too
//!   are not corrupt"). Comparing `max_doc` alone never decompressed a single
//!   chunk, so a corrupted `.fdt` body passed cleanly.
//! - **`testTermVectors`** (`term_vectors.*`): doc count vs `.si`, every
//!   document's vectors decoded, and every field a document carries vectors
//!   for cross-checked against `.fnm`'s `storeTermVectors`.
//! - **`testDocValues`** (`doc_values.*`): every field's every per-doc value
//!   decoded out of `.dvd`, with SORTED/SORTED_SET ordinals bounds-checked
//!   against the terms dictionary's size, SORTED_SET ordinals checked for
//!   being a strictly increasing set within a doc, BINARY value lengths
//!   checked against `.dvm`'s declared min/max, and the decoded
//!   docs-with-a-value count checked against `.dvm`'s own
//!   `numDocsWithField`.
//! - **`testSort`** (`sort.docs_in_index_sort_order`): a segment declaring an
//!   `indexSort` really is in that order, verified by reading the sort
//!   fields' doc values and applying `segment_info::SortKeyComparator` -- the
//!   same comparator the sort-on-flush writer used to produce the order.
//! - **`checkSoftDeletes`** (`soft_deletes.count_matches`): the number of
//!   live docs carrying a value for `.fnm`'s soft-deletes field equals the
//!   commit's `softDelCount`.
//! - **`testPostings`' field-level and term-level invariants**
//!   (`postings.terms_sorted`, `postings.doc_freq_positive`,
//!   `postings.field_summary`): terms strictly increasing within a field,
//!   `docFreq > 0`, and `.tmd`'s `numTerms`/`sumDocFreq`/`sumTotalTermFreq`/
//!   `docCount`/`minTerm`/`maxTerm` each re-derived from the dictionary and
//!   its postings.
//! - **`SegmentInfos.readCommit`'s cross-`.si` validations**
//!   (`commit.*`), which this port's `segment_infos::parse` structurally
//!   cannot perform because it never opens a `.si`: `delCount`/`softDelCount`
//!   within `maxDoc` and not summing past it, `delCount == 0` without a
//!   delete generation, the segment's version at or after the commit's
//!   `minSegmentLuceneVersion` and `indexCreatedVersionMajor`, `minVersion`
//!   recorded once `indexCreatedVersionMajor >= 7`, total docs within
//!   `IndexWriter.MAX_DOCS`, unique segment names, and
//!   `SegmentInfos.counter` ahead of every segment name (real `CheckIndex`'s
//!   `validCounter`).
//! - **The `.si`'s own self-reference** (`si.files_lists_itself`) and the
//!   commit-level files (`.liv`, generational field-infos/doc-values-update
//!   files) that live on `SegmentCommitInfo` and never in the `.si`
//!   (`SegmentCommitInfo.files()`).
//!
//! ## Checks added in the `c9-check-index` sweep
//!
//! b11 left three gaps, all closed here:
//!
//! - **`testFieldNorms`'s value scan and `checkFields`' terms-vs-norms
//!   cross-check** (`norms.entry_present:*`, `norms.values_decode:*`,
//!   `norms.agree_with_postings:*`): every norm value is read out of `.nvd`,
//!   the docs-with-a-value count is checked against `.nvm`'s own
//!   `numDocsWithField`, and a live doc's norm is required to be non-zero
//!   exactly when that doc has terms in the field's postings -- Java's
//!   "doesn't have terms according to postings but has a norm value that is
//!   not zero" / "has terms according to postings but its norm value is 0".
//! - **`checkFields`' positional, statistical, seek and intersect
//!   invariants** (`postings.positions_valid:*`, `.offsets_valid:*`,
//!   `.term_stats:*`, `.term_dict_shape:*`, `.field_in_fnm:*`,
//!   `.advance_agrees:*`, `.seek_agrees:*`, `.intersect_agrees:*`,
//!   `.terms_decode:*`) -- see [`check_postings`]' own table, and its list
//!   of the Java checks that are structurally vacuous in this port's
//!   representation and why.
//! - **`testVectors`** (`vectors.*`). b11 recorded it as unreachable because
//!   "this port has no vector/HNSW write path at all"; batch c5 built one, so
//!   there is now writer output to check.
//! - **The `hnsw.*` checks are an addition beyond 10.5.0, not a port.** The
//!   pinned version's `CheckIndex` has no HNSW graph check at all --
//!   `testHnswGraphs`/`testHnswGraph` were added to Lucene *after* 10.5.0,
//!   and c9 mislabelled these as a port of them. They are kept because they
//!   are diagnostic-only and do catch real graph corruption (see
//!   [`check_hnsw_graphs`]), and because Lucene's own later equivalent
//!   checks the same three properties -- but nothing here should be read as
//!   describing what 10.5.0's `CheckIndex` does. (c18 version audit.)
//!
//! Two further gaps found while sweeping and closed here:
//!
//! - **The per-file check was `retrieve_checksum`, not
//!   `checksumEntireFile`.** Real `CheckIndex`'s `test: check integrity`
//!   step re-reads every byte of every segment file and recomputes its
//!   CRC-32. This module only validated the footer's *shape*, so a byte
//!   flipped in the middle of a `.kdd`/`.fdt`/`.tim` passed the `file:*`
//!   check outright.
//! - **`testTermVectors` never cross-checked a vector against the inverted
//!   index** (`term_vectors.self_consistent`, `.match_postings`).
//!
//! **Still out of scope**: `checkImpacts`/`checkDocIDRuns` (this port has no
//! separate `ImpactsEnum` and no `docIDRunEnd` for the postings enum to
//! disagree with -- see [`check_postings`]); and compound (`.cfs`) segments,
//! which this module has never supported and skips rather than mis-reports.
//! A previous version of this list also named the `Float16` vector encoding:
//! that was a `main`-ism. 10.5.0's `VectorEncoding` has exactly `BYTE` and
//! `FLOAT32`, so there is no third encoding to be out of scope -- and this
//! port's reader correctly rejects a third ordinal. (c18 version audit.)
// This module is *audited* against the arithmetic gate: no `+`/`-`/`*`/`<<`
// in it may panic, because a verifier that panics on a corrupt file has
// failed at its one job (and through the FFI a panic in a debug build takes
// the JVM with it). Every operation here is `checked_*` with a reported
// corruption, `saturating_*` where the saturation is unreachable or is itself
// the honest answer for a counter, or a plain operator under an `#[allow]`
// carrying an `// ARITH:` proof. See `docs/arithmetic-gate.md`.
#![deny(clippy::arithmetic_side_effects)]

use crate::deletes::liv_file_name;
use crate::segment_info::{self, SegmentInfo};
use crate::segment_infos::{self, SegmentCommitInfo, SegmentInfos};
use lucene_codecs::blocktree;
use lucene_codecs::doc_values;
use lucene_codecs::field_infos::{self, FieldInfos};
use lucene_codecs::hnsw::HnswGraphView;
use lucene_codecs::hnsw_vectors;
use lucene_codecs::live_docs;
use lucene_codecs::norms;
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
/// and when that step fails the family used to contribute **nothing** to the
/// result -- so a segment whose `.fnm` would not parse reported a failed
/// `fnm.open` and then, silently, no postings checks at all. Worse, three of
/// the term-vector families were pushed as *passes* in that state, because
/// their problem lists were empty for the simple reason that nothing had
/// looked.
///
/// [`Outcome::Skipped`] is that state made visible: the check is named, the
/// prerequisite that took it down is named, and [`CheckResult::all_passed`]
/// is `false`. It is emitted **only** when a prerequisite actually failed --
/// a format the segment legitimately does not have (no vectors, no points, a
/// compound segment) produces no check at all, as before, because there is
/// nothing there to be unguarded.
///
/// (Found by c23: a `.fnm` this port wrote but its own parser rejected made
/// `check_index` skip every postings check in the segment. See c25.)
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
    let segment_name = commit.segment_name.clone();
    let mut checks = Vec::new();

    let si = match open_si(dir, commit) {
        Ok(si) => {
            checks.push(Check::pass("si.open"));
            si
        }
        Err(e) => {
            checks.push(Check::fail("si.open", e));
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
        .and_then(|_| open_postings_bytes(dir, &si));
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
        // No `.tim`/`.tip`/`.tmd` at all, or a compound segment: a legitimate
        // absence, already reported by `fnm.postings_vs_files` if a field
        // claims postings. Nothing is left unguarded, so nothing is skipped.
        _ => None,
    };

    let mut stats = CheckStats::default();
    // `checkFields`' `visitedDocs`, per postings field: internal plumbing
    // from the postings walk to the terms-vs-norms cross-check, so it stays
    // local rather than riding along on every `CheckResult` for the life of
    // the run (one `FixedBitSet` per field per segment).
    let mut docs_with_terms = Vec::new();
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
pub fn check_directory(dir: &dyn Directory) -> segment_infos::Result<Vec<CheckResult>> {
    let infos = segment_infos::read_latest(dir)?;
    let mut results = vec![check_commit(&infos)];
    for commit in &infos.segments {
        results.push(check_segment_in_commit(dir, Some(&infos), commit));
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

/// Every file `.si` lists must exist and pass `CodecUtil.checksumEntireFile`
/// -- the whole payload re-read and its CRC-32 recomputed, then compared
/// against the footer's stored checksum.
///
/// This is real `CheckIndex`'s `test: check integrity` step, which runs
/// `reader.checkIntegrity()` (every format's `checkIntegrity` bottoms out in
/// `CodecUtil.checksumEntireFile`) on every segment file. It previously used
/// [`codec_util::retrieve_checksum`], which validates the footer's *shape*
/// (magic/algorithm id/checksum field) and never touches the payload -- so a
/// flipped byte in the middle of a `.tim`/`.fdt`/`.dvd` passed this check
/// cleanly and only got caught later by whichever decoder happened to trip
/// over it (and not at all for the bytes no decoder reads). It shares
/// [`codec_util::check_whole_file_footer`] with `merge.rs`'s pre-bulk-copy
/// `check_integrity` (batch c4) and with [`crate::checksum_verify`].
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
    let has_postings_files = has_any_ext(files, &[".tim", ".tip", ".tmd"]);

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
/// files, and exactly the pairing c28 found live twice in `deletes`/
/// `term_delete`. They agree today only because [`check_live_docs`] hands
/// `si.doc_count` to `live_docs::parse` as `max_doc`, which is one line away
/// from not being true; `FixedBitSet::get` indexes `words[index >> 6]` behind
/// a bare `debug_assert`, so the failure would be a ghost bit in release and
/// an index panic in *both* profiles once the id is 64 or more past the end.
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
        let reader = stored_fields::open(&fdt, &fdx, &fdm, &commit.segment_id, "")
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
/// Skipped (not failed) for a compound segment (this module has no
/// compound-file support) and for a segment with no `.tvd`/`.tvx`/`.tvm` at
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
    if si.is_compound_file {
        return;
    }
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
        let reader = term_vectors::open(&tvd, &tvx, &tvm, &commit.segment_id, "")
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
        for doc_id in 0..si.doc_count.min(reader.max_doc()) {
            match reader.document(doc_id) {
                Ok(None) => {}
                Ok(Some(document)) => {
                    docs_with_vectors = docs_with_vectors.saturating_add(1);
                    let live = is_live(live_docs, doc_id);
                    let Some(fi) = field_infos else { continue };
                    for field in &document.fields {
                        let info = fi.fields.iter().find(|f| f.number == field.field_number);
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
        // cross-check when the postings side is missing. This is the exact
        // shape c23 hit (`fnm.open` failing and the segment still reporting
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
        if postings.is_none() {
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
fn named_field_check(name: &str, problems: &[String], num_units: i64, unit: &str) -> Check {
    if problems.is_empty() {
        Check::pass(name)
    } else {
        let shown = problems.len().min(5);
        Check::fail(
            name,
            format!(
                "{} of {num_units} {unit} affected; first {shown}: {}",
                problems.len(),
                problems[..shown].join("; ")
            ),
        )
    }
}

/// `IndexWriter.MAX_POSITION`: the largest token position `IndexWriter` will
/// accept (`Integer.MAX_VALUE - 128`). `checkFields` rejects anything past
/// it; a position beyond this could only come from a corrupt `.pos` or a
/// writer that never checked.
///
/// Re-exported from the **writer's** definition rather than duplicated (c28's
/// carry-over). The two halves of one rule -- what
/// [`crate::indexing_chain::advance_position`] clamps to and what
/// `postings.positions_valid` rejects -- must move together, and a verifier
/// with its own copy of the writer's ceiling is a verifier that can silently
/// stop agreeing with the writer it verifies.
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
/// `None` means "nothing to check here", not "ok": a compound segment (this
/// module has no compound-file support), or a segment with none/only some of
/// `.tim`/`.tip`/`.tmd` -- a partial group is already reported by
/// [`check_field_flags_vs_files`]'s `fnm.postings_vs_files`, and this
/// function does not duplicate that failure.
fn open_postings_bytes(
    dir: &dyn Directory,
    si: &SegmentInfo,
) -> Option<Result<PostingsFileBytes, String>> {
    if si.is_compound_file {
        return None;
    }
    let find = |ext: &str| si.files.iter().find(|f| f.ends_with(ext));
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
    let segment_suffix = tim_name
        .rsplit_once('_')
        .map(|(_, tail)| tail)
        .and_then(|tail| tail.strip_suffix(".tim"))
        .map(|counter| {
            // ARITH: `rsplit_once('_')` succeeded and `strip_suffix(".tim")`
            // succeeded, so `tim_name` is exactly `<stem>_<counter>.tim` and
            // `len - counter.len() - 4 - 1 == stem.len() >= 0`. `i` is a byte
            // index `rfind` returned, so `i + 1 <= stem.len()`. Both are char
            // boundaries because both split on ASCII `_`.
            #[allow(clippy::arithmetic_side_effects)]
            let stem = &tim_name[..tim_name.len() - counter.len() - ".tim".len() - 1];
            // ARITH: `i` is a byte index `rfind` returned, so
            // `i + 1 <= stem.len()`, and it is a char boundary because the
            // split is on ASCII `_`.
            #[allow(clippy::arithmetic_side_effects)]
            match stem.rfind('_') {
                Some(i) => format!("{}_{counter}", &stem[i + 1..]),
                None => counter.to_string(),
            }
        })
        .unwrap_or_default();
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
        })
    })())
}

impl PostingsFileBytes {
    fn handles(
        &self,
        commit: &SegmentCommitInfo,
        field_infos: &FieldInfos,
        si: &SegmentInfo,
    ) -> Result<PostingsHandles<'_>, String> {
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
}

/// `CheckIndex.testPostings` -> `checkFields(fields, liveDocs, maxDoc,
/// fieldInfos, normsProducer, true, false, ...)`, the deepest single check
/// real `CheckIndex` performs.
///
/// Per field it re-derives, from the postings themselves, everything the
/// term dictionary and `.tmd` *claim*, and walks the positional streams:
///
/// | check | Java |
/// |---|---|
/// | `postings.field_in_fnm:<f>` | `isIndexed == false` (the `fieldsEnum inconsistent with fieldInfos` half is unfirable here -- see below) |
/// | `postings.term_dict_shape:<f>` | `hasPayloads`/`hasOffsets` vs `.fnm` (`docCount > maxDoc` and `minTerm`/`maxTerm` both-or-neither are unfirable -- see below) |
/// | `postings.terms_sorted:<f>` | `terms out of order` |
/// | `postings.doc_freq_positive:<f>` | `docfreq: N is out of bounds` |
/// | `postings.term_stats:<f>` | `totalTermFreq <= 0`, `totalTermFreq < docFreq`, term outside `[minTerm, maxTerm]`, and the accumulator overflowing `i64` |
/// | `postings.total_term_freq:<f>` | `totalTermFreq != recomputed totalTermFreq` |
/// | `postings.doc_ids_valid:<f>` | `doc <= lastDoc`, `doc >= maxDoc`, `freq <= 0` |
/// | `postings.positions_valid:<f>` | `pos < 0`, `pos > MAX_POSITION`, `pos < lastPos` |
/// | `postings.offsets_valid:<f>` | `startOffset < 0`, `startOffset < lastOffset`, `endOffset < 0`, `endOffset < startOffset` |
/// | `postings.advance_agrees:<f>` | the `Test skipping` block's seven `advance(maxDoc*i/8)` probes |
/// | `postings.seek_agrees:<f>` | `Test seeking by ord` / `seek to last term` / `seek to existing term ... failed` |
/// | `postings.intersect_agrees:<f>` | `checkTermsIntersect` |
/// | `postings.field_summary:<f>` | `sumDocFreq`/`sumTotalTermFreq`/`docCount`/`termCount`/`minTerm`/`maxTerm` vs recomputed |
///
/// Returns the counts real `CheckIndex` prints for `test: terms, freq,
/// prox...OK [N terms; M terms/docs pairs; K tokens]` plus, per field, the
/// set of documents that carry at least one term (`checkFields`'
/// `visitedDocs`), which [`check_field_norms`] consumes.
///
/// **Deliberately not ported, with reasons** (rather than silently omitted):
///
/// - *`docCount != docFreq` per term.* This port's decode API is
///   parameterized by the claimed `docFreq` (like Java's own
///   `BlockDocsEnum`/`TermState.docFreq`), so a recount can never disagree
///   -- see this module's top comment. The observable symptom of a wrong
///   `docFreq` is an out-of-range or non-increasing doc ID, which
///   `postings.doc_ids_valid` checks directly.
/// - *Every check that reads back the `.fnm`'s **frequency** flag* --
///   `Terms.hasFreqs()` vs `IndexOptions`, `!hasFreqs => totalTermFreq ==
///   docFreq`, `!hasFreqs => freq == 1`, `!hasFreqs => sumTotalTermFreq ==
///   sumDocFreq`. In Lucene these getters are *derived from*
///   `FieldInfo.getIndexOptions()`, and so is this port's decode path:
///   `blocktree` sets `totalTermFreq = docFreq` for an `IndexOptions::Docs`
///   field, `read_freq_pair` returns `(sumDocFreq, sumDocFreq)` for one, and
///   `read_tail_block`/`refill_full_block` do `freqs.fill(1)`. All four are
///   `x == x`, and after c25 removed them nothing in this function reads
///   `has_freqs` at all -- **the `.fnm`'s frequency flag has no independent
///   on-disk witness in this port.** `hasPayloads`/`hasOffsets` *are*
///   checked, because those do have one: whether the segment carries a `.pay`
///   file.
/// - *A field in the dictionary with no `.fnm` entry, and `.tmd`'s `docCount
///   > maxDoc` / `sumTotalTermFreq < sumDocFreq`.* `blocktree::open` rejects
///   all three while parsing (`InvalidFieldNumber`, `InvalidDocCount`,
///   `InvalidSumTotalTermFreq`), the first because it takes each field's
///   *name* from the very `FieldInfos` a check here would compare against.
///   A file that would trip them is reported as `postings.open`. (c25)
/// - *`payload.length < 1`, and any payload check at all.* Java guards
///   against a codec handing back a zero-length `BytesRef`; this port
///   represents "no payload at this position" as an empty `Vec<u8>`, so a
///   zero-length payload and no payload are the same value. Nor can a
///   payload appear on a field whose `.fnm` says `storePayloads = false`:
///   `read_positions` is *told* whether to read payloads by the very same
///   `FieldInfo` flag, so the comparison is `x == x`. Payloads *are*
///   cross-checked, in the one place two independent copies of them exist
///   on disk -- `term_vectors.match_postings`.
/// - *`freq` vs. the number of decoded positions.* `read_positions` chops
///   the flat position stream into per-document groups *using* `freqs`, so
///   the group's length is `freq` by construction or the decode fails
///   outright. Same `x == x` shape.
/// - *`impacts`/`checkImpacts`/`checkDocIDRuns`.* Java's slow level
///   cross-checks the `ImpactsEnum` against the `PostingsEnum` and the
///   `docIDRunEnd` API. This port's `PostingsCursor`/`LazyDocsCursor` expose
///   impacts (`level0_impacts`/`level1_impacts`) but have no `docIDRunEnd`
///   and no separate impacts enum to disagree with the postings enum: the
///   impacts come off the same decoded block. `postings.advance_agrees`
///   covers the part that *does* have two independent implementations here
///   -- the skip-list-driven `advance` versus the fully decoded doc list.
fn check_postings(
    si: &SegmentInfo,
    field_infos: &FieldInfos,
    live_docs: Option<&lucene_util::fixed_bit_set::FixedBitSet>,
    p: &PostingsHandles<'_>,
    checks: &mut Vec<Check>,
) -> (
    CheckStats,
    Vec<(String, lucene_util::fixed_bit_set::FixedBitSet)>,
) {
    let mut stats = CheckStats::default();
    let mut docs_with_terms: Vec<(String, lucene_util::fixed_bit_set::FixedBitSet)> = Vec::new();
    // Reported once for the segment, but tracked *per field*: a field that
    // needs a missing `.doc` must not suppress the next field's `docCount`
    // re-derivation and `visitedDocs` (which would make that field's
    // `norms.agree_with_postings` silently vanish rather than fail).
    let mut any_field_needs_doc_file = false;
    let has_pay_file = p.pay_in.is_some();

    for (field_name, field_terms) in p.fields.iter_fields() {
        // Java's "fieldsEnum inconsistent with fieldInfos" -- a field in the
        // dictionary that `.fnm` has no entry for -- is **not** ported: it
        // cannot fire. `blocktree::open` resolves each `.tmd` record's field
        // *number* through `FieldInfos::field_by_number` (rejecting an
        // unknown one as `Error::InvalidFieldNumber`) and names the field
        // from what it found, so every name `iter_fields` yields came out of
        // this very `FieldInfos`. The lookup below is `x == x`; only the
        // `indexOptions=None` half has an independent claim to compare.
        let Some(fi) = field_infos.fields.iter().find(|f| f.name == field_name) else {
            debug_assert!(
                false,
                "blocktree::open names every field from this FieldInfos, and rejects \
                 duplicate names, so the lookup cannot miss"
            );
            continue;
        };
        let mut shape_problems: Vec<String> = Vec::new();
        if fi.index_options == field_infos::IndexOptions::None {
            shape_problems.push(format!(
                "field {field_name:?} has a term dictionary but .fnm says indexOptions=None"
            ));
        }
        checks.push(named_field_check(
            &format!("postings.field_in_fnm:{field_name}"),
            &shape_problems,
            1,
            "field",
        ));

        let index_options = fi.index_options;
        // No `has_freqs` here: every one of `checkFields`' three
        // freq-flag checks turned out to be `x == x` in this port and is
        // documented as such where it used to live (see `postings.term_stats`
        // above, `postings.doc_ids_valid` below and `postings.field_summary`
        // at the end of this loop). `blocktree`'s decoder derives
        // `totalTermFreq`, `freq` and `sumTotalTermFreq` from the *same*
        // `.fnm` `indexOptions` this function would compare them against.
        //
        // The flag is not *unwitnessed* -- `index_has_freq` selects the
        // `.doc` block layout, so a `.fnm` that disagrees with the file that
        // was written misparses the stream -- but the witness is a decode
        // failure (`postings.terms_decode`, `postings.doc_ids_valid`), not a
        // value this function can compare against anything.
        let has_positions = matches!(
            index_options,
            field_infos::IndexOptions::DocsAndFreqsAndPositions
                | field_infos::IndexOptions::DocsAndFreqsAndPositionsAndOffsets
        );
        let has_offsets = matches!(
            index_options,
            field_infos::IndexOptions::DocsAndFreqsAndPositionsAndOffsets
        );

        let mut dict_problems: Vec<String> = Vec::new();
        // `checkFields`' "docCount > maxDoc for field" is **not** ported: it
        // cannot fire here. `blocktree::open` is handed `si.doc_count` as its
        // `max_doc` and rejects a `.tmd` whose `docCount` is outside
        // `0..=max_doc` (`Error::InvalidDocCount`) while parsing, so a
        // dictionary that would trip this check never reaches
        // `check_postings` at all -- it is reported as `postings.open`. See
        // c9 finding 9 / c25 for this module's rule on never-firing checks.
        //
        // Java's `(minTerm == null) != (maxTerm == null)` has no
        // counterpart here: this port stores both as (possibly empty) byte
        // strings rather than nullables, so "absent" and "the empty term"
        // are the same value and the asymmetry cannot be observed. The
        // stronger property is checked instead, by
        // `postings.field_summary`: `.tmd`'s minTerm/maxTerm must equal the
        // first and last terms the dictionary actually enumerates.
        // `.pay` exists exactly when the field indexes offsets or payloads
        // (`Lucene104PostingsWriter`), so `.fnm`'s flags have an independent
        // on-disk witness here even though `hasFreqs`/`hasPositions` do not.
        if (has_offsets || fi.store_payloads) && has_positions && !has_pay_file {
            dict_problems.push(format!(
                "field {field_name:?}: .fnm says hasOffsets={has_offsets} \
                 storePayloads={} but the segment has no .pay file",
                fi.store_payloads
            ));
        }
        if has_positions && p.pos_in.is_none() {
            dict_problems.push(format!(
                "field {field_name:?}: .fnm says the field indexes positions but the segment \
                 has no .pos file"
            ));
        }
        checks.push(named_field_check(
            &format!("postings.term_dict_shape:{field_name}"),
            &dict_problems,
            field_terms.num_terms,
            "terms",
        ));

        let mut freq_mismatches: Vec<String> = Vec::new();
        let mut doc_id_problems: Vec<String> = Vec::new();
        let mut order_problems: Vec<String> = Vec::new();
        let mut doc_freq_problems: Vec<String> = Vec::new();
        let mut term_stat_problems: Vec<String> = Vec::new();
        let mut position_problems: Vec<String> = Vec::new();
        let mut offset_problems: Vec<String> = Vec::new();
        let mut advance_problems: Vec<String> = Vec::new();
        let mut needs_doc_file = false;
        let mut prev_term: Option<Vec<u8>> = None;
        let mut first_term: Option<Vec<u8>> = None;
        let mut counted_terms = 0i64;
        let mut summed_doc_freq = 0i64;
        let mut summed_total_term_freq = 0i64;
        let mut visited_docs =
            lucene_util::fixed_bit_set::FixedBitSet::new(si.doc_count.max(0) as usize);
        // Terms sampled for the seek/reseek round, at most 10 000 of them --
        // Java's own `seekCount = min(10000, termCount)` cap.
        let mut seek_samples: Vec<Vec<u8>> = Vec::new();
        let seek_stride = (field_terms.num_terms / 10_000).max(1);

        let mut terms = field_terms.iter();
        let mut term_buf: Vec<u8> = Vec::new();
        let mut decode_error: Option<String> = None;
        loop {
            let stats_now = match terms.try_next() {
                Ok(Some((term, s))) => {
                    term_buf.clear();
                    term_buf.extend_from_slice(term);
                    Some(s)
                }
                Ok(None) => None,
                Err(e) => {
                    decode_error = Some(format!("field {field_name:?}: {e}"));
                    None
                }
            };
            let Some(claimed) = stats_now else { break };
            let term = term_buf.as_slice();

            counted_terms = counted_terms.saturating_add(1);
            // `doc_freq`/`total_term_freq` come straight off `.tim`/`.tmd`, so
            // a corrupt dictionary can hand this loop `i64::MAX` repeatedly.
            //
            // `summed_doc_freq` saturates: `doc_freq` is an `i32`, so reaching
            // `i64::MAX` would take 2^32 terms, and a saturated value cannot
            // accidentally equal the `.tmd`'s own claim in any real file.
            //
            // `summed_total_term_freq` must **not** saturate. Its operand is
            // itself an `i64` off disk and it is compared against
            // `sum_total_term_freq`, another `i64` off disk -- so a `.tmd`
            // claiming `i64::MAX` paired with per-term values that overflow
            // would compare *equal*, and the check would pass on exactly the
            // file it exists to reject. Report the overflow instead.
            summed_doc_freq = summed_doc_freq.saturating_add(i64::from(claimed.doc_freq));
            match summed_total_term_freq.checked_add(claimed.total_term_freq) {
                Some(sum) => summed_total_term_freq = sum,
                None => {
                    term_stat_problems.push(format!(
                        "the per-term totalTermFreq values overflow i64 at term {:?}",
                        String::from_utf8_lossy(term)
                    ));
                    break;
                }
            }
            if first_term.is_none() {
                first_term = Some(term.to_vec());
            }
            // ARITH: `seek_stride` is `(num_terms / 10_000).max(1)`, so it is
            // at least 1 -- neither a divide-by-zero nor the `i64::MIN % -1`
            // overflow is reachable.
            #[allow(clippy::arithmetic_side_effects)]
            let sample = counted_terms % seek_stride == 0;
            if sample && seek_samples.len() < 10_000 {
                seek_samples.push(term.to_vec());
            }
            if let Some(prev) = &prev_term {
                if term <= prev.as_slice() {
                    order_problems.push(format!(
                        "field {field_name:?}: term {term:?} does not come after previous \
                         term {prev:?}"
                    ));
                }
            }
            prev_term = Some(term.to_vec());
            if claimed.doc_freq <= 0 {
                doc_freq_problems.push(format!(
                    "field {field_name:?} term {term:?}: docFreq={} is not > 0",
                    claimed.doc_freq
                ));
            }
            // `checkFields`' term-statistic bounds.
            if claimed.total_term_freq <= 0 {
                term_stat_problems.push(format!(
                    "field {field_name:?} term {term:?}: totalTermFreq={} is not > 0",
                    claimed.total_term_freq
                ));
            }
            if claimed.total_term_freq < claimed.doc_freq as i64 {
                term_stat_problems.push(format!(
                    "field {field_name:?} term {term:?}: totalTermFreq={} < docFreq={}",
                    claimed.total_term_freq, claimed.doc_freq
                ));
            }
            if !field_terms.min_term.is_empty() && term < field_terms.min_term.as_slice() {
                term_stat_problems.push(format!(
                    "field {field_name:?}: term {term:?} sorts before .tmd's minTerm {:?}",
                    field_terms.min_term
                ));
            }
            if !field_terms.max_term.is_empty() && term > field_terms.max_term.as_slice() {
                term_stat_problems.push(format!(
                    "field {field_name:?}: term {term:?} sorts after .tmd's maxTerm {:?}",
                    field_terms.max_term
                ));
            }

            if claimed.doc_freq > 1 && p.doc_in.is_none() {
                needs_doc_file = true;
                any_field_needs_doc_file = true;
                continue;
            }

            // One metadata decode for both the docs/freqs and, when the
            // field indexes them, the positions/offsets/payloads.
            let decoded = if has_positions {
                match p.pos_in.as_ref() {
                    Some(pos_in) => terms
                        .try_current_postings_and_positions(
                            p.doc_in.as_ref(),
                            pos_in,
                            p.pay_in.as_ref(),
                        )
                        .map(|o| o.map(|(d, pos)| (d, Some(pos)))),
                    // Already reported by `postings.term_dict_shape`.
                    None => Ok(None),
                }
            } else {
                terms
                    .try_current_postings(p.doc_in.as_ref())
                    .map(|o| o.map(|d| (d, None)))
            };
            let (postings_of_term, positions) = match decoded {
                Ok(Some(v)) => v,
                Ok(None) => continue,
                Err(e) => {
                    decode_error = Some(format!("field {field_name:?} term {term:?}: {e}"));
                    break;
                }
            };

            // `totalTermFreq` re-derivation: each doc's freq comes straight
            // off the wire, so this is genuinely independent of the claim.
            let actual_total_term_freq: i64 =
                postings_of_term.freqs.iter().map(|&f| f as i64).sum();
            if actual_total_term_freq != claimed.total_term_freq {
                freq_mismatches.push(format!(
                    "field {field_name:?} term {term:?}: dictionary claims \
                     totalTermFreq={}, but postings actually sum to {actual_total_term_freq}",
                    claimed.total_term_freq
                ));
            }

            let mut prev_doc_id = -1i32;
            let mut term_is_live = false;
            for (i, &doc_id) in postings_of_term.docs.iter().enumerate() {
                if doc_id >= 0 && doc_id < si.doc_count {
                    visited_docs.set(doc_id as usize);
                }
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
                let freq = postings_of_term.freqs.get(i).copied().unwrap_or(0);
                if freq <= 0 {
                    doc_id_problems.push(format!(
                        "field {field_name:?} term {term:?}: doc {doc_id} has freq {freq}, \
                         which is out of bounds"
                    ));
                }
                let live = is_live(live_docs, doc_id);
                if live {
                    term_is_live = true;
                    stats.term_doc_pairs = stats.term_doc_pairs.saturating_add(1);
                    stats.token_count = stats.token_count.saturating_add(i64::from(freq.max(0)));
                }

                // Positions/offsets/payloads for this doc.
                if let Some(per_doc) = positions.as_ref().and_then(|p| p.get(i)) {
                    check_occurrences(
                        field_name,
                        term,
                        doc_id,
                        per_doc,
                        has_offsets,
                        &mut position_problems,
                        &mut offset_problems,
                    );
                }
            }
            if term_is_live {
                stats.term_count = stats.term_count.saturating_add(1);
            } else {
                stats.del_term_count = stats.del_term_count.saturating_add(1);
            }

            // `checkFields`' "Test skipping" block: seven `advance` probes
            // spread across the doc-id space. Java only checks the returned
            // doc is `>= target` and that `nextDoc` moves forward; this port
            // already holds the term's whole decoded doc list, so it checks
            // the stronger property -- the skip-list-driven `advance` lands
            // on exactly the doc a linear scan of that list would. That is a
            // real cross-check of the `.doc` level-0/level-1 skip data
            // against the block payload it indexes.
            if let Some(doc_in) = p.doc_in.as_ref() {
                if claimed.doc_freq > 1 {
                    match field_terms.lazy_postings(term, doc_in) {
                        Ok(Some(mut cursor)) => {
                            // ARITH: `idx` is 0..7 and `doc_count` is an
                            // `i32`, so the product is at most 8 * 2^31 --
                            // three orders of magnitude inside `i64`.
                            #[allow(clippy::arithmetic_side_effects)]
                            for idx in 0..7i64 {
                                let target = ((idx + 1) * i64::from(si.doc_count) / 8)
                                    .clamp(0, i32::MAX as i64)
                                    as i32;
                                let expected = postings_of_term
                                    .docs
                                    .iter()
                                    .copied()
                                    .find(|&d| d >= target)
                                    .unwrap_or(postings::NO_MORE_DOCS);
                                match cursor.advance(target) {
                                    Ok(got) if got == expected => {}
                                    Ok(got) => {
                                        advance_problems.push(format!(
                                            "field {field_name:?} term {term:?}: \
                                             advance({target}) returned {got}, but the decoded \
                                             doc list's first doc >= {target} is {expected}"
                                        ));
                                        break;
                                    }
                                    Err(e) => {
                                        advance_problems.push(format!(
                                            "field {field_name:?} term {term:?}: \
                                             advance({target}) failed: {e}"
                                        ));
                                        break;
                                    }
                                }
                                if expected == postings::NO_MORE_DOCS {
                                    break;
                                }
                            }
                        }
                        Ok(None) => advance_problems.push(format!(
                            "field {field_name:?}: term {term:?} was enumerated but a re-seek \
                             for its postings found nothing"
                        )),
                        Err(e) => advance_problems.push(format!(
                            "field {field_name:?} term {term:?}: re-seek failed: {e}"
                        )),
                    }
                }
            }
        }

        if let Some(e) = decode_error {
            checks.push(Check::fail(
                format!("postings.terms_decode:{field_name}"),
                e,
            ));
        }

        // `checkFields`' seek round: every sampled term must be findable
        // again by an independent `seekExact`/`seekCeil` from the trie root,
        // returning the same statistics the forward scan reported. `try_*`
        // is deliberate (batch c1): a corrupt block must surface as an error
        // here, not as "term not found".
        let mut seek_problems: Vec<String> = Vec::new();
        {
            let mut seek_enum = field_terms.iter();
            for sample in &seek_samples {
                match field_terms.try_seek_exact(sample) {
                    Ok(Some(_)) => {}
                    Ok(None) => seek_problems.push(format!(
                        "field {field_name:?}: term {sample:?} was enumerated but \
                         seekExact cannot find it"
                    )),
                    Err(e) => seek_problems.push(format!(
                        "field {field_name:?}: seekExact({sample:?}) failed: {e}"
                    )),
                }
                match seek_enum.try_seek_ceil(sample) {
                    Ok(blocktree::SeekStatus::Found) => match seek_enum.try_current() {
                        Ok(Some((landed, _))) if landed == sample.as_slice() => {}
                        Ok(Some((landed, _))) => seek_problems.push(format!(
                            "field {field_name:?}: seekCeil({sample:?}) reported FOUND but \
                             landed on {landed:?}"
                        )),
                        Ok(None) => seek_problems.push(format!(
                            "field {field_name:?}: seekCeil({sample:?}) reported FOUND but the \
                             cursor is on no term"
                        )),
                        Err(e) => seek_problems.push(format!(
                            "field {field_name:?}: seekCeil({sample:?}) landed on an \
                             undecodable term: {e}"
                        )),
                    },
                    Ok(other) => seek_problems.push(format!(
                        "field {field_name:?}: seekCeil({sample:?}) returned {other:?}, not FOUND"
                    )),
                    Err(e) => seek_problems.push(format!(
                        "field {field_name:?}: seekCeil({sample:?}) failed: {e}"
                    )),
                }
            }
            // Java's dedicated "seek to last term" case. Its accompanying
            // `docFreq` recount is *not* ported: `postings()` is
            // parameterized by the claimed `docFreq` (see this module's top
            // comment), so `docs.len() == doc_freq` whenever the decode
            // succeeds at all and the comparison could never disagree.
            if let Some(last) = &prev_term {
                match field_terms.try_seek_exact(last) {
                    Ok(Some(_)) => {}
                    Ok(None) => seek_problems.push(format!(
                        "field {field_name:?}: seek to last term {last:?} failed"
                    )),
                    Err(e) => seek_problems.push(format!(
                        "field {field_name:?}: seek to last term {last:?} failed: {e}"
                    )),
                }
            }
        }
        checks.push(named_field_check(
            &format!("postings.seek_agrees:{field_name}"),
            &seek_problems,
            seek_samples.len() as i64,
            "sampled terms",
        ));

        // `checkTermsIntersect`: the pruning term walker must return exactly
        // the terms a linear scan filtered by the same matcher returns, in
        // the same order. Compared in lockstep rather than by collecting
        // both sides: `.*[a-e].*` matches most natural-language terms, so
        // materializing them would be close to a second copy of the whole
        // dictionary.
        let mut intersect_problems: Vec<String> = Vec::new();
        if let Ok(pattern) = lucene_codecs::regexp::RegexpPattern::new(b".*[a-e].*") {
            compare_intersect_with_scan(
                field_name,
                field_terms,
                "regexp `.*[a-e].*`",
                |t| pattern.matches(t),
                field_terms.regexp_intersect(&pattern),
                &mut intersect_problems,
            );
        }
        {
            let pattern = lucene_codecs::wildcard::WildcardPattern::new(b"*e*");
            compare_intersect_with_scan(
                field_name,
                field_terms,
                "wildcard `*e*`",
                |t| pattern.matches(t),
                field_terms.intersect(&pattern),
                &mut intersect_problems,
            );
        }
        checks.push(named_field_check(
            &format!("postings.intersect_agrees:{field_name}"),
            &intersect_problems,
            field_terms.num_terms,
            "terms",
        ));

        for (name, problems) in [
            ("postings.total_term_freq", &freq_mismatches),
            ("postings.doc_ids_valid", &doc_id_problems),
            ("postings.terms_sorted", &order_problems),
            ("postings.doc_freq_positive", &doc_freq_problems),
            ("postings.term_stats", &term_stat_problems),
            ("postings.advance_agrees", &advance_problems),
        ] {
            checks.push(named_field_check(
                &format!("{name}:{field_name}"),
                problems,
                field_terms.num_terms,
                "terms",
            ));
        }
        if has_positions {
            checks.push(named_field_check(
                &format!("postings.positions_valid:{field_name}"),
                &position_problems,
                field_terms.num_terms,
                "terms",
            ));
        }
        if has_offsets {
            checks.push(named_field_check(
                &format!("postings.offsets_valid:{field_name}"),
                &offset_problems,
                field_terms.num_terms,
                "terms",
            ));
        }

        // Field-level totals. These come from `.tmd` and are what a
        // scorer's IDF/normalisation reads.
        let mut summary_problems: Vec<String> = Vec::new();
        if counted_terms != field_terms.num_terms {
            summary_problems.push(format!(
                "field {field_name:?}: .tmd claims numTerms={} but the dictionary enumerates \
                 {counted_terms}",
                field_terms.num_terms
            ));
        }
        if summed_doc_freq != field_terms.sum_doc_freq {
            summary_problems.push(format!(
                "field {field_name:?}: .tmd claims sumDocFreq={} but the terms' docFreqs sum \
                 to {summed_doc_freq}",
                field_terms.sum_doc_freq
            ));
        }
        if summed_total_term_freq != field_terms.sum_total_term_freq {
            summary_problems.push(format!(
                "field {field_name:?}: .tmd claims sumTotalTermFreq={} but the terms' \
                 totalTermFreqs sum to {summed_total_term_freq}",
                field_terms.sum_total_term_freq
            ));
        }
        if let Some(first) = &first_term {
            if first.as_slice() != field_terms.min_term.as_slice() {
                summary_problems.push(format!(
                    "field {field_name:?}: .tmd claims minTerm={:?} but the first enumerated \
                     term is {first:?}",
                    field_terms.min_term
                ));
            }
        }
        if let Some(last) = &prev_term {
            if last.as_slice() != field_terms.max_term.as_slice() {
                summary_problems.push(format!(
                    "field {field_name:?}: .tmd claims maxTerm={:?} but the last enumerated \
                     term is {last:?}",
                    field_terms.max_term
                ));
            }
        }
        if !needs_doc_file {
            // `visited_docs` is only complete when every term's postings
            // were actually decoded.
            let visited = visited_docs.cardinality() as i32;
            if visited != field_terms.doc_count {
                summary_problems.push(format!(
                    "field {field_name:?}: .tmd claims docCount={} but {visited} distinct doc \
                     IDs appear in its postings",
                    field_terms.doc_count
                ));
            }
            docs_with_terms.push((field_name.to_string(), visited_docs));
        }
        checks.push(named_field_check(
            &format!("postings.field_summary:{field_name}"),
            &summary_problems,
            field_terms.num_terms,
            "terms",
        ));
    }
    if any_field_needs_doc_file {
        checks.push(Check::fail(
            "postings.doc_open",
            "a term with docFreq > 1 needs the segment's .doc file, but none was found",
        ));
    }
    (stats, docs_with_terms)
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

/// One `checkTermsIntersect` round: walks the field's terms linearly and the
/// pruning `intersect` iterator side by side, reporting the first place they
/// disagree. Neither side is collected, so the memory cost is one term.
fn compare_intersect_with_scan(
    field_name: &str,
    field_terms: &blocktree::FieldTerms,
    label: &str,
    matches: impl Fn(&[u8]) -> bool,
    mut intersected: impl Iterator<Item = (Vec<u8>, blocktree::TermStats)>,
    problems: &mut Vec<String>,
) {
    let mut scan = field_terms.iter();
    loop {
        let expected = loop {
            match scan.try_next() {
                Ok(Some((t, _))) if matches(t) => break Some(t.to_vec()),
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
        let actual = intersected.next().map(|(t, _)| t);
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
/// This was previously listed as out of scope ("doc-values value-range
/// sanity ... a separate task"). It is not: `lucene-codecs::doc_values`
/// already exposes `parse_meta` plus a per-doc accessor for every kind, and
/// `lucene-search`'s `directory_reader` already calls them. Without this,
/// nothing anywhere read a single doc-values *value* during a check, so a
/// corrupted `.dvd` payload passed a clean `CheckIndex` run -- the exact
/// "verifier that does not check something silently passes corrupt
/// segments" failure mode.
///
/// Skipped (not failed) for a compound segment (this module has no
/// compound-file support) and for a segment where no field claims
/// doc-values. A field that claims them while the segment lacks
/// `.dvd`/`.dvm` is already reported by [`check_field_flags_vs_files`]'s
/// `fnm.doc_values_vs_files`, so this function returns quietly in that case
/// rather than duplicating the failure.
fn check_doc_values(
    dir: &dyn Directory,
    commit: &SegmentCommitInfo,
    si: &SegmentInfo,
    field_infos: &FieldInfos,
    checks: &mut Vec<Check>,
) {
    if si.is_compound_file {
        return;
    }
    let dv_fields: Vec<&field_infos::FieldInfo> = field_infos
        .fields
        .iter()
        .filter(|f| f.doc_values_type != field_infos::DocValuesType::None)
        .collect();
    if dv_fields.is_empty() {
        return;
    }
    let (Some(dvm_name), Some(dvd_name)) = (
        si.files.iter().find(|f| f.ends_with(".dvm")),
        si.files.iter().find(|f| f.ends_with(".dvd")),
    ) else {
        return; // already reported by fnm.doc_values_vs_files
    };

    let result = (|| -> Result<Vec<Check>, String> {
        let dvm = dir.open(dvm_name).map_err(|e| e.to_string())?;
        let dvd = dir.open(dvd_name).map_err(|e| e.to_string())?;
        // The per-format suffix is embedded in the file name, exactly as
        // `check_postings_term_stats` derives the postings one.
        let suffix = if *dvm_name == format!("{}.dvm", commit.segment_name) {
            String::new()
        } else {
            dvm_name
                .strip_prefix(&format!("{}_", commit.segment_name))
                .or_else(|| dvm_name.strip_prefix('_'))
                .and_then(|s| s.strip_suffix(".dvm"))
                .unwrap_or_default()
                .to_string()
        };
        let (_version, meta) =
            doc_values::parse_meta(&dvm, &commit.segment_id, &suffix, field_infos)
                .map_err(|e| e.to_string())?;
        doc_values::check_data_header_footer(&dvd, &commit.segment_id, &suffix)
            .map_err(|e| e.to_string())?;

        // `.dvs` is a separate file, present only when some field asked for a
        // skip index (`Lucene90DocValuesConsumer`'s
        // `VERSION_SKIPPER_SEPARATE_FILE`). Opened once here rather than per
        // field, like `.dvm`/`.dvd`.
        let dvs = if dv_fields
            .iter()
            .any(|f| f.doc_values_skip_index_type != field_infos::DocValuesSkipIndexType::None)
        {
            match si.files.iter().find(|f| f.ends_with(".dvs")) {
                Some(dvs_name) => Some(dir.open(dvs_name).map_err(|e| e.to_string())?),
                None => None, // already reported by fnm.doc_values_vs_files
            }
        } else {
            None
        };

        let mut out = Vec::new();
        for fi in &dv_fields {
            let name = &fi.name;
            let mut problems: Vec<String> = Vec::new();
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
                        // `.dvd` (c19 measured 0 corruptions caught elsewhere).
                        skip_families(
                            &mut out,
                            &[&format!("doc_values.values_decode:{name}")],
                            &format!("doc_values.entry_present:{name}"),
                        );
                        continue;
                    };
                    for doc in 0..si.doc_count {
                        match doc_values::numeric_value(&dvd, entry, doc) {
                            Ok(Some(_)) => docs_with_value = docs_with_value.saturating_add(1),
                            Ok(None) => {}
                            Err(e) => problems.push(format!("docID={doc}: {e}")),
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
                        // `.dvd` (c19 measured 0 corruptions caught elsewhere).
                        skip_families(
                            &mut out,
                            &[&format!("doc_values.values_decode:{name}")],
                            &format!("doc_values.entry_present:{name}"),
                        );
                        continue;
                    };
                    expected_docs_with_value = Some(entry.num_docs_with_field as i64);
                    for doc in 0..si.doc_count {
                        match doc_values::binary_value(&dvd, entry, doc) {
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
                            Err(e) => problems.push(format!("docID={doc}: {e}")),
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
                        // `.dvd` (c19 measured 0 corruptions caught elsewhere).
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
                    for doc in 0..si.doc_count {
                        match doc_values::sorted_ord(&dvd, entry, doc) {
                            Ok(Some(ord)) => {
                                docs_with_value = docs_with_value.saturating_add(1);
                                if ord < 0 || ord >= value_count {
                                    problems.push(format!(
                                        "docID={doc}: ordinal {ord} outside the terms \
                                         dictionary's 0..{value_count} range"
                                    ));
                                } else {
                                    seen.set(ord as usize);
                                }
                            }
                            Ok(None) => {}
                            Err(e) => problems.push(format!("docID={doc}: {e}")),
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
                        // `.dvd` (c19 measured 0 corruptions caught elsewhere).
                        skip_families(
                            &mut out,
                            &[&format!("doc_values.values_decode:{name}")],
                            &format!("doc_values.entry_present:{name}"),
                        );
                        continue;
                    };
                    expected_docs_with_value = Some(entry.num_docs_with_field as i64);
                    for doc in 0..si.doc_count {
                        match doc_values::sorted_numeric_values(&dvd, entry, doc) {
                            Ok(values) => {
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
                            Err(e) => problems.push(format!("docID={doc}: {e}")),
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
                        // `.dvd` (c19 measured 0 corruptions caught elsewhere).
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
                            for doc in 0..si.doc_count {
                                match doc_values::sorted_ord(&dvd, single, doc) {
                                    Ok(Some(ord)) => {
                                        docs_with_value = docs_with_value.saturating_add(1);
                                        if ord < 0 || ord >= value_count {
                                            problems.push(format!(
                                                "docID={doc}: ordinal {ord} outside the terms \
                                                 dictionary's 0..{value_count} range"
                                            ));
                                        } else {
                                            seen.set(ord as usize);
                                        }
                                    }
                                    Ok(None) => {}
                                    Err(e) => problems.push(format!("docID={doc}: {e}")),
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
                            for doc in 0..si.doc_count {
                                match doc_values::sorted_numeric_values(&dvd, ords, doc) {
                                    Ok(values) => {
                                        if values.is_empty() {
                                            continue;
                                        }
                                        docs_with_value = docs_with_value.saturating_add(1);
                                        // `SortedSetDocValues` ordinals are a
                                        // *set*: strictly increasing per doc.
                                        let mut prev = -1i64;
                                        for ord in values {
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
                                                seen.set(ord as usize);
                                            }
                                            prev = ord;
                                        }
                                    }
                                    Err(e) => problems.push(format!("docID={doc}: {e}")),
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
                match (dvs.as_ref(), meta.skipper_meta(fi.number)) {
                    (Some(dvs), Some(skipper_meta)) => {
                        match doc_values::parse_skip_index(
                            dvs,
                            &commit.segment_id,
                            &suffix,
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
                match lucene_codecs::terms_dict::decode_all_terms(&dvd, terms) {
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

/// Opens a segment's `.dvm`/`.dvd` pair, if it has one, returning the parsed
/// metadata and the raw data bytes. Shared by the doc-values-value checks,
/// the index-sort check and the soft-deletes check, all of which need to
/// read actual per-doc values.
fn open_doc_values(
    dir: &dyn Directory,
    commit: &SegmentCommitInfo,
    si: &SegmentInfo,
    field_infos: &FieldInfos,
) -> Option<Result<(doc_values::DocValuesMeta, lucene_store::directory::Input), String>> {
    let dvm_name = si.files.iter().find(|f| f.ends_with(".dvm"))?;
    let dvd_name = si.files.iter().find(|f| f.ends_with(".dvd"))?;
    let suffix = if *dvm_name == format!("{}.dvm", commit.segment_name) {
        String::new()
    } else {
        dvm_name
            .strip_prefix(&format!("{}_", commit.segment_name))
            .or_else(|| dvm_name.strip_prefix('_'))
            .and_then(|s| s.strip_suffix(".dvm"))
            .unwrap_or_default()
            .to_string()
    };
    Some((|| {
        let dvm = dir.open(dvm_name).map_err(|e| e.to_string())?;
        let dvd = dir.open(dvd_name).map_err(|e| e.to_string())?;
        let (_v, meta) = doc_values::parse_meta(&dvm, &commit.segment_id, &suffix, field_infos)
            .map_err(|e| e.to_string())?;
        Ok((meta, dvd))
    })())
}

/// Which documents carry a value for `fi` -- `DocValuesIterator`'s
/// "advanceExact returns true", which is what
/// `PendingSoftDeletes.countSoftDeletes` counts over the soft-deletes field.
///
/// Only the two doc-values types a soft-deletes field can have. Until c35
/// this shared [`sort_key_values`], which is now sort-field-shaped and reads
/// three different columns depending on the `SortField` kind -- an argument
/// the soft-deletes check has nothing to pass.
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
            for doc in 0..max_doc {
                present.push(
                    doc_values::numeric_value(dvd, entry, doc)
                        .map_err(|e| e.to_string())?
                        .is_some(),
                );
            }
        }
        field_infos::DocValuesType::SortedNumeric => {
            let entry = meta
                .sorted_numeric_entry(fi.number)
                .ok_or("no SORTED_NUMERIC doc-values entry")?;
            for doc in 0..max_doc {
                present.push(
                    !doc_values::sorted_numeric_values(dvd, entry, doc)
                        .map_err(|e| e.to_string())?
                        .is_empty(),
                );
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
/// Deliberately not supported, and reported as such rather than silently
/// mis-verified: a `SortedSetSortField` (its per-document reduction is
/// `SortedSetSelector`, which needs a SORTED_SET ordinal reader this port's
/// `doc_values` module does not expose) and a `BinarySortField` (whose keys
/// are raw bytes, so there is no `Option<i64>` to return at all -- see
/// [`segment_info::IndexSortField::key_comparison`]).
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
            for doc in 0..max_doc {
                keys.push(doc_values::numeric_value(dvd, entry, doc).map_err(|e| e.to_string())?);
            }
        }
        (
            IndexSortKind::SortedNumeric { selector, .. },
            field_infos::DocValuesType::SortedNumeric,
        ) => {
            let entry = meta
                .sorted_numeric_entry(fi.number)
                .ok_or("no SORTED_NUMERIC doc-values entry")?;
            for doc in 0..max_doc {
                let values = doc_values::sorted_numeric_values(dvd, entry, doc)
                    .map_err(|e| e.to_string())?;
                // `SortedNumericSelector.MinValue`/`MaxValue`: the *first*
                // and the *last* stored value of the document, not its
                // smallest and largest. The two coincide only for an
                // ascending column, which is what
                // `doc_values.sorted_numeric_ascending` separately checks --
                // re-deriving min/max here would mask an unsorted column
                // rather than expose it.
                keys.push(match selector {
                    SortedNumericSelector::Min => values.into_iter().next(),
                    SortedNumericSelector::Max => values.into_iter().last(),
                });
            }
        }
        (IndexSortKind::String(_), field_infos::DocValuesType::Sorted) => {
            let entry = meta
                .sorted_entry(fi.number)
                .ok_or("no SORTED doc-values entry")?;
            for doc in 0..max_doc {
                keys.push(doc_values::sorted_ord(dvd, entry, doc).map_err(|e| e.to_string())?);
            }
        }
        (IndexSortKind::SortedSet { .. }, _) => {
            return Err(
                "a SortedSetSortField's per-document ordinal needs a SORTED_SET selector \
                 reader this port does not expose"
                    .to_string(),
            )
        }
        (IndexSortKind::Binary(_), _) => {
            return Err(
                "a BinarySortField compares raw bytes, which is not a single-i64 sort key"
                    .to_string(),
            )
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
/// Skipped (not failed) for an unsorted segment, a compound segment, a
/// segment with no doc-values files, or a sort this port can read but not
/// compare (a `SortedSetSortField` or a `BinarySortField` -- see
/// [`sort_key_values`]). "Skipped" is deliberate for the last of those: the
/// index is openable and everything else about it is checked, but this one
/// property is unverified and saying so is the difference between a check
/// that passed and one that never ran.
fn check_index_sort(
    dir: &dyn Directory,
    commit: &SegmentCommitInfo,
    si: &SegmentInfo,
    field_infos: &FieldInfos,
    checks: &mut Vec<Check>,
) {
    let Some(sort_fields) = &si.index_sort else {
        return;
    };
    if si.is_compound_file {
        return;
    }
    // A sort kind this port can read but not *verify* is unverifiable before
    // any file is opened, and reporting it as a failure would call a
    // perfectly good real-Lucene index corrupt. Skipped, with the reason and
    // the field, so it is visible that the check did not run.
    if let Some(sf) = sort_fields.iter().find(|sf| {
        matches!(
            sf.kind,
            segment_info::IndexSortKind::SortedSet { .. } | segment_info::IndexSortKind::Binary(_)
        )
    }) {
        checks.push(Check::skipped(
            "sort.docs_in_index_sort_order",
            &format!(
                "a comparator for sort field {:?}, a {},",
                sf.field,
                match sf.kind {
                    segment_info::IndexSortKind::Binary(_) =>
                        "BinarySortField whose keys are raw bytes rather than one i64",
                    _ =>
                        "SortedSetSortField whose per-document ordinal needs a SORTED_SET \
                          selector reader this port does not expose",
                }
            ),
        ));
        return;
    }
    let Some(opened) = open_doc_values(dir, commit, si, field_infos) else {
        // The sharpest of the skip cases, and the reason this is modelled at
        // all: a segment that *declares* an index sort but carries no
        // doc-values files has nothing this check can read, and the sort is
        // then verified by nothing anywhere. Before c25 that was silence, and
        // `all_passed()` said the segment was fine -- over a `.si` whose
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
        let (meta, dvd) = opened?;
        let mut per_field: Vec<(Vec<Option<i64>>, segment_info::SortKeyComparator)> =
            Vec::with_capacity(sort_fields.len());
        for sf in sort_fields {
            let fi = field_infos
                .fields
                .iter()
                .find(|f| f.name == sf.field)
                .ok_or_else(|| format!("sort field {:?} is not in .fnm", sf.field))?;
            let cmp = segment_info::SortKeyComparator::new(sf)
                .expect("the unsupported kinds returned above");
            per_field.push((
                sort_key_values(&dvd, &meta, sf, fi, si.doc_count)
                    .map_err(|e| format!("sort field {:?}: {e}", sf.field))?,
                cmp,
            ));
        }

        let mut problems = Vec::new();
        for doc in 1..si.doc_count {
            // ARITH: the range starts at 1.
            #[allow(clippy::arithmetic_side_effects)]
            let prev = doc - 1;
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
/// field, or the segment is compound, or it has no doc-values files.
fn check_soft_deletes(
    dir: &dyn Directory,
    commit: &SegmentCommitInfo,
    si: &SegmentInfo,
    field_infos: &FieldInfos,
    checks: &mut Vec<Check>,
) {
    if si.is_compound_file {
        return;
    }
    let Some(fi) = field_infos.fields.iter().find(|f| f.soft_deletes_field) else {
        return;
    };
    let Some(opened) = open_doc_values(dir, commit, si, field_infos) else {
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
        let (meta, dvd) = opened?;
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
/// Skipped (not failed) when: the segment is compound (`.cfs`/`.cfe`,
/// matching this module's existing compound-file scope, same as
/// [`check_postings_term_stats`]); or no field in `.fnm` claims points at
/// all. A field that *does* claim points but whose segment is missing one
/// of `.kdm`/`.kdi`/`.kdd` is reported as a single `points.open` failure
/// rather than silently skipped -- points files are optional at the
/// segment level (most segments have none), but once one field commits to
/// having them, all three must be present and parse.
fn check_points_structural_invariants(
    dir: &dyn Directory,
    commit: &SegmentCommitInfo,
    si: &SegmentInfo,
    field_infos: &FieldInfos,
    checks: &mut Vec<Check>,
) {
    if si.is_compound_file {
        return;
    }
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
                    field_checks.push(Check::fail(
                        format!("points.field_present:{field_name}"),
                        "field claims points in .fnm but has no entry in .kdm",
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
                        &format!("points.field_present:{field_name}"),
                    );
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

            let mut value_problems: Vec<String> = Vec::new();
            let mut bound_problems: Vec<String> = Vec::new();
            let mut doc_problems: Vec<String> = Vec::new();
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
            // `testPoints`' `getDocCountSeen() != docCount` check. b11
            // ported the point-count half of this; the doc-count half --
            // which is what a `PointRangeQuery`'s `docCount`-based cost
            // estimate and `IndexSearcher`'s pruning read -- had no
            // counterpart.
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

/// The per-field checks a field's own `points.field_present`/`points.decode`
/// failure takes down -- everything that walks the field's leaves.
const POINTS_PER_FIELD: &[&str] = &[
    "points.value_within_field_bounds",
    "points.leaf_bounds_subset_of_field",
    "points.doc_count_matches",
    "points.point_count_matches",
];

/// The per-field families a failed `points.open` takes down.
const POINTS_FAMILIES: &[&str] = &[
    "points.field_present",
    "points.value_within_field_bounds",
    "points.leaf_bounds_subset_of_field",
    "points.doc_count_matches",
    "points.point_count_matches",
];

/// `CheckIndex.testFieldNorms`: for every field `.fnm` says has norms, read
/// **every** norm value out of `.nvd`, plus `checkFields`' terms-vs-norms
/// cross-check.
///
/// b11 recorded this as the one `test*` method with no counterpart here: the
/// module only cross-checked that `.nvd`/`.nvm` existed when some field
/// claimed norms, so a corrupted norms payload passed a clean run and then
/// silently changed every BM25 score that read it.
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
/// [`indexed_disi::DisiCursor`] rather than `norms::norm_value` per doc:
/// that helper builds a fresh cursor per lookup, which would make this an
/// O(maxDoc x blocks) scan instead of O(maxDoc).
#[allow(clippy::too_many_arguments)]
fn check_field_norms(
    dir: &dyn Directory,
    commit: &SegmentCommitInfo,
    si: &SegmentInfo,
    field_infos: &FieldInfos,
    live_docs: Option<&lucene_util::fixed_bit_set::FixedBitSet>,
    docs_with_terms: &[(String, lucene_util::fixed_bit_set::FixedBitSet)],
    stats: &mut CheckStats,
    checks: &mut Vec<Check>,
) {
    if si.is_compound_file {
        return;
    }
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
            // c19 measured that **no** check other than these reads `.nvd`,
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
            // norms are checked by nothing at all -- and c19 measured that
            // nothing else in the module reads `.nvd`.
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
        let visited = docs_with_terms
            .iter()
            .find(|(n, _)| *n == fi.name)
            .map(|(_, v)| v);

        let mut decode_problems: Vec<String> = Vec::new();
        let mut agree_problems: Vec<String> = Vec::new();
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
            lucene_codecs::indexed_disi::DisiCursor::new(region, entry.dense_rank_power)
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
/// b11 recorded this as out of scope because "this port has no vector/HNSW
/// write path at all". Batch c5 built one -- `.vec`/`.vemf` are real
/// `Lucene99FlatVectorsFormat` bytes and `.vem`/`.vex` real
/// `Lucene99HnswVectorsFormat` ones -- so there is now writer output to
/// check, and this is its checker.
///
/// `testVectors`:
///
/// - Java's `dimension <= 0` guard is *not* ported: it is unfalsifiable in
///   Java as well as here (see the comment at its site), and c9 finding 9
///   established that this module does not ship checks that can only pass.
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
    if si.is_compound_file {
        return;
    }
    let with_vectors: Vec<&field_infos::FieldInfo> = field_infos
        .fields
        .iter()
        .filter(|f| f.vector_dimension != 0)
        .collect();
    if with_vectors.is_empty() {
        return;
    }
    let vec_name = si.files.iter().find(|f| f.ends_with(".vec"));
    let vemf_name = si.files.iter().find(|f| f.ends_with(".vemf"));
    let (Some(vec_name), Some(vemf_name)) = (vec_name, vemf_name) else {
        checks.push(Check::fail(
            "vectors.open",
            "a field declares vector values but the segment has no .vec/.vemf files",
        ));
        skip_families(checks, VECTOR_FAMILIES, "vectors.open");
        return;
    };
    // `_0_Lucene99HnswVectorsFormat_0.vec` -> `Lucene99HnswVectorsFormat_0`.
    let suffix = vec_name
        .strip_prefix(&format!("{}_", commit.segment_name))
        .and_then(|s| s.strip_suffix(".vec"))
        .unwrap_or_default()
        .to_string();

    let opened = (|| -> Result<
        (
            lucene_store::directory::Input,
            lucene_store::directory::Input,
        ),
        String,
    > {
        let vemf = dir.open(vemf_name).map_err(|e| e.to_string())?;
        let vec = dir.open(vec_name).map_err(|e| e.to_string())?;
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
    let flat = match vectors::FlatVectorsReader::open(&vemf, &vec, &commit.segment_id, &suffix) {
        Ok(r) => r,
        Err(e) => {
            checks.push(Check::fail("vectors.open", e.to_string()));
            skip_families(checks, VECTOR_FAMILIES, "vectors.open");
            return;
        }
    };

    for fi in &with_vectors {
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
        // would be a check that can only ever pass, which c9 finding 9
        // already established this module does not do.
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
        // They are kept, unlike c25's D1-D10 and c30's D11. The distinction
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

    check_hnsw_graphs(dir, commit, si, &with_vectors, &suffix, checks);
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
/// after 10.5.0, and c9 documented these as a port of them. They stay because
/// they are diagnostic-only, cost nothing a caller does not ask for, and do
/// catch real corruption -- `corrupted_hnsw_graph_bytes_are_never_silently_accepted`
/// and `corrupted_hnsw_neighbours_are_caught_by_the_graph_checks` are their
/// evidence -- and the three properties checked here are the same ones
/// Lucene's own later `testHnswGraph` settled on. But the label was wrong and
/// is corrected here: nothing below describes 10.5.0 behaviour. (c18 version
/// audit.)
fn check_hnsw_graphs(
    dir: &dyn Directory,
    commit: &SegmentCommitInfo,
    si: &SegmentInfo,
    with_vectors: &[&field_infos::FieldInfo],
    suffix: &str,
    checks: &mut Vec<Check>,
) {
    let vem_name = si.files.iter().find(|f| f.ends_with(".vem"));
    let vex_name = si.files.iter().find(|f| f.ends_with(".vex"));
    // No graph files at all is the flat (exhaustive-search) format, not a
    // defect.
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
        let size = graph.size();
        let mut level_problems: Vec<String> = Vec::new();
        let mut order_problems: Vec<String> = Vec::new();
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
            let mut on_this_level =
                lucene_util::fixed_bit_set::FixedBitSet::new(size.max(0) as usize);
            for &n in &nodes {
                // Java's `node < 0 || node > size - 1` guard is not repeated
                // here: `read_field_entry` already rejects an out-of-range
                // level-node ordinal while parsing `.vem` (this port
                // validates on the way in where Java validates on the way
                // out), and level 0's node set is the implicit `0..size`.
                // The guard that *can* fire is the neighbour one below.
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
                let mut last = -1i32;
                for &nbr in &neighbors {
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
                let connected = connected_nodes_on_level(&graph, level, size);
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
/// c25 left an `if entry < 0 || entry >= size { return 0 }` guard here as a
/// D-list candidate. It is deleted: `entry_node` is `0` for a single-level
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
fn connected_nodes_on_level(
    graph: &hnsw_vectors::OffHeapHnswGraph<'_>,
    level: i32,
    size: i32,
) -> usize {
    let entry = graph.entry_node();
    let mut seen = lucene_util::fixed_bit_set::FixedBitSet::new(size.max(0) as usize);
    let mut stack = vec![entry];
    let mut neighbors: Vec<i32> = Vec::new();
    let mut count = 0usize;
    while let Some(node) = stack.pop() {
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

#[cfg(test)]
mod tests {
    // Test code opts out of the arithmetic gate at the module boundary: the
    // gate exists for values read off disk, not for a test's own index
    // arithmetic. See `docs/arithmetic-gate.md`.
    #![allow(clippy::arithmetic_side_effects)]

    use super::*;
    use lucene_store::codec_util::ID_LENGTH;
    use lucene_store::FsDirectory;

    fn fixture_dir(name: &str) -> std::path::PathBuf {
        std::path::PathBuf::from(format!(
            "{}/../../fixtures/data/{name}/",
            env!("CARGO_MANIFEST_DIR")
        ))
    }

    /// A scratch directory that actually goes away. This used to hand back a
    /// bare `PathBuf` that nothing ever removed -- and `/tmp` is a 16 GB
    /// tmpfs, i.e. RAM, so ~21 000 leaked directories from repeated runs ate
    /// memory until the kernel started killing processes. See
    /// [`lucene_util::test_support`].
    fn tempdir() -> lucene_util::test_support::TempDir {
        lucene_util::test_support::TempDir::new("check-index")
    }

    /// A genuinely valid, real-Lucene-written fixture must pass every check
    /// cleanly -- the baseline "no false positives" test.
    #[test]
    fn valid_blocktree_fixture_passes_every_check() {
        let dir = FsDirectory::open(fixture_dir("blocktree_index"));
        let results = check_directory(&dir).expect("read segments_N");
        // results[0] is the commit-level result; results[1..] are the segments.
        assert_eq!(results.len(), 2);
        let result = &results[1];
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
        assert_eq!(results.len(), 2);
        assert!(
            results[1].all_passed(),
            "unexpected failures: {:?}",
            results[1].failures()
        );
        assert!(results[1]
            .checks
            .iter()
            .any(|c| c.name == "liv.cardinality_matches_del_count"));
    }

    /// Every real-Lucene index fixture in the tree must pass every check
    /// cleanly. This is the no-false-positives baseline for the checks added
    /// beyond the original scope (doc values, term vectors, postings field
    /// summaries, commit-level invariants) -- each of these fixtures was
    /// written by a real `IndexWriter`, so any failure is ours.
    #[test]
    fn every_real_lucene_index_fixture_passes_every_check() {
        for name in [
            "blocktree_index",
            "live_docs_index",
            "doc_values_index",
            "sorted_dv_index",
            "multi_valued_dv_index",
            "term_vectors_index",
            "norms_index",
            "points_index",
            "doc_values_skip_index",
            "doc_values_varying_bpv",
            "vectors_index",
            // A real Java-written index that declares an **index sort**, so
            // `check_index_sort`'s success path is proved against Lucene's
            // own sort-on-flush output and not only against this port's.
            "sorted_index",
        ] {
            let dir = FsDirectory::open(fixture_dir(name));
            let results = check_directory(&dir).unwrap_or_else(|e| panic!("{name}: {e}"));
            for result in &results {
                assert!(
                    result.all_passed(),
                    "{name} segment {}: unexpected failures: {:?}",
                    result.segment_name,
                    result.failures()
                );
            }
        }
    }

    /// The doc-values checks must actually run (and be non-vacuous) on a
    /// fixture that has doc values -- a check that silently never fires is
    /// worse than no check.
    #[test]
    fn doc_values_checks_actually_run_on_a_doc_values_fixture() {
        let dir = FsDirectory::open(fixture_dir("doc_values_index"));
        let results = check_directory(&dir).unwrap();
        let names: Vec<&str> = results[1].checks.iter().map(|c| c.name.as_str()).collect();
        assert!(
            names
                .iter()
                .any(|n| n.starts_with("doc_values.values_decode:")),
            "expected doc_values checks, got {names:?}"
        );
    }

    /// Copies a fixture directory into a fresh temp dir, dropping the
    /// `.raw`/manifest side-files the fixture ships for other tests.
    fn copy_fixture(name: &str) -> lucene_util::test_support::TempDir {
        let src = fixture_dir(name);
        let dst = tempdir();
        for entry in std::fs::read_dir(&src).unwrap() {
            let entry = entry.unwrap();
            let file_name = entry.file_name();
            let s = file_name.to_string_lossy().to_string();
            if s.ends_with(".raw") || s.contains("manifest") || s.ends_with(".tsv") {
                continue;
            }
            if entry.file_type().unwrap().is_file() {
                std::fs::copy(entry.path(), dst.join(&s)).unwrap();
            }
        }
        dst
    }

    fn find_file(dir: &std::path::Path, ext: &str) -> std::path::PathBuf {
        std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().path())
            .find(|p| p.to_string_lossy().ends_with(ext))
            .unwrap_or_else(|| panic!("no {ext} file in {dir:?}"))
    }

    fn failed_names(results: &[CheckResult]) -> Vec<String> {
        results
            .iter()
            .flat_map(|r| r.failures())
            .map(|c| c.name.clone())
            .collect()
    }

    /// Real Java `CheckIndex -level 3` was run on each of these fixtures and
    /// its printed counts recorded here verbatim. `test: terms, freq,
    /// prox...OK [N terms; M terms/docs pairs; K tokens]` is
    /// `term_count`/`term_doc_pairs`/`token_count`; `test: field
    /// norms.........OK [N fields]` is `norm_fields`; `test: term
    /// vectors........OK [N total term vector count]` is
    /// `term_vector_fields`; `test: vectors.............OK [N fields, M
    /// vectors]` is `vector_fields`/`vector_values`.
    ///
    /// A disagreement is itself a finding: it means one side enumerated
    /// something the other did not.
    #[test]
    fn counts_match_real_java_check_index_output() {
        // (fixture, terms, terms/docs pairs, tokens, norm fields,
        //  term vector count, vector fields, vector values)
        struct Expected {
            fixture: &'static str,
            terms: i64,
            term_doc_pairs: i64,
            tokens: i64,
            norm_fields: i64,
            term_vectors: i64,
            vector_fields: i64,
            vector_values: i64,
        }
        let e = |fixture,
                 terms,
                 term_doc_pairs,
                 tokens,
                 norm_fields,
                 term_vectors,
                 vector_fields,
                 vector_values| Expected {
            fixture,
            terms,
            term_doc_pairs,
            tokens,
            norm_fields,
            term_vectors,
            vector_fields,
            vector_values,
        };
        let expected = [
            e("blocktree_index", 414, 8968, 13547, 6, 0, 0, 0),
            e("term_vectors_index", 8, 8, 9, 2, 3, 0, 0),
            e("norms_index", 24, 43, 43, 2, 0, 0, 0),
            e("points_index", 2000, 2000, 2000, 0, 0, 0, 0),
            e("vectors_index", 0, 0, 0, 0, 0, 5, 7911),
        ];
        for x in expected {
            let name = x.fixture;
            let dir = FsDirectory::open(fixture_dir(name));
            let results = check_directory(&dir).unwrap_or_else(|e| panic!("{name}: {e}"));
            let s = &results[1].stats;
            let got = [
                s.term_count,
                s.term_doc_pairs,
                s.token_count,
                s.norm_fields,
                s.term_vector_fields,
                s.vector_fields,
                s.vector_values,
            ];
            let want = [
                x.terms,
                x.term_doc_pairs,
                x.tokens,
                x.norm_fields,
                x.term_vectors,
                x.vector_fields,
                x.vector_values,
            ];
            assert_eq!(
                got, want,
                "{name}: counts disagree with real Java CheckIndex"
            );
        }
    }

    /// The norms values check must actually run on a fixture with norms, and
    /// so must the terms-vs-norms cross-check.
    #[test]
    fn norms_checks_actually_run_on_a_norms_fixture() {
        let dir = FsDirectory::open(fixture_dir("norms_index"));
        let results = check_directory(&dir).unwrap();
        let names: Vec<&str> = results[1].checks.iter().map(|c| c.name.as_str()).collect();
        assert!(
            names.iter().any(|n| n.starts_with("norms.values_decode:")),
            "expected norms value checks, got {names:?}"
        );
        assert!(
            names
                .iter()
                .any(|n| n.starts_with("norms.agree_with_postings:")),
            "expected the terms-vs-norms cross-check, got {names:?}"
        );
    }

    /// The positional checks must actually run on a fixture whose fields
    /// index positions, offsets and payloads.
    #[test]
    fn positional_checks_actually_run_on_the_blocktree_fixture() {
        let dir = FsDirectory::open(fixture_dir("blocktree_index"));
        let results = check_directory(&dir).unwrap();
        let names: Vec<&str> = results[1].checks.iter().map(|c| c.name.as_str()).collect();
        for prefix in [
            "postings.positions_valid:",
            "postings.seek_agrees:",
            "postings.intersect_agrees:",
            "postings.advance_agrees:",
            "postings.term_stats:",
            "postings.term_dict_shape:",
        ] {
            assert!(
                names.iter().any(|n| n.starts_with(prefix)),
                "expected a {prefix}* check, got {names:?}"
            );
        }
    }

    /// The vector and HNSW-graph checks must actually run on the real
    /// `Lucene99HnswVectorsFormat` fixture.
    #[test]
    fn vector_checks_actually_run_on_the_vectors_fixture() {
        let dir = FsDirectory::open(fixture_dir("vectors_index"));
        let results = check_directory(&dir).unwrap();
        let names: Vec<&str> = results[1].checks.iter().map(|c| c.name.as_str()).collect();
        for prefix in [
            "vectors.values_decode:",
            "vectors.ord_to_doc:",
            "vectors.field_entry_matches_fnm:",
            "hnsw.neighbors_on_level:",
            "hnsw.neighbors_sorted:",
            "hnsw.entry_point_reachable:",
        ] {
            assert!(
                names.iter().any(|n| n.starts_with(prefix)),
                "expected a {prefix}* check, got {names:?}"
            );
        }
        assert!(
            results[1].all_passed(),
            "unexpected failures: {:?}",
            results[1].failures()
        );
    }

    /// The term-vectors-vs-postings cross-check must actually run.
    #[test]
    fn term_vector_postings_cross_check_actually_runs() {
        let dir = FsDirectory::open(fixture_dir("term_vectors_index"));
        let results = check_directory(&dir).unwrap();
        let names: Vec<&str> = results[1].checks.iter().map(|c| c.name.as_str()).collect();
        assert!(names.contains(&"term_vectors.match_postings"));
        assert!(names.contains(&"term_vectors.self_consistent"));
        assert!(
            results[1].all_passed(),
            "unexpected failures: {:?}",
            results[1].failures()
        );
    }

    /// A hand-built one-field segment whose single term spans `num_docs`
    /// documents -- enough for the `.doc` file to carry several full
    /// 128-document blocks and therefore real level-0/level-1 skip data,
    /// which is what `postings.advance_agrees` cross-checks against the
    /// decoded blocks. Small on disk, so a byte-by-byte corruption scan over
    /// it is cheap.
    fn write_many_doc_postings_fixture(
        dst_dir: &std::path::Path,
        num_docs: i32,
    ) -> segment_infos::SegmentCommitInfo {
        use lucene_codecs::postings_writer::TermPostings;
        let terms = vec![TermPostings {
            term: b"alpha".to_vec(),
            docs: (0..num_docs).map(|d| (d, 1 + (d % 3))).collect(),
            positions: vec![],
            offsets: vec![],
            payloads: vec![],
        }];
        write_postings_fixture(dst_dir, &terms, num_docs, num_docs, None)
    }

    /// Negative control for `postings.advance_agrees`: the `.doc` file
    /// carries both the packed doc-id blocks and the level-0/level-1 skip
    /// data that indexes them, and this check is the only place the two are
    /// compared -- the skip-driven `advance` is required to land on exactly
    /// the document a linear scan of the fully decoded block would.
    #[test]
    fn corrupting_the_doc_skip_data_is_caught_by_the_advance_check() {
        let dst = tempdir();
        let commit = write_many_doc_postings_fixture(&dst, 1000);
        let doc_path = dst.join(format!("_0_{POSTINGS_SUFFIX}.doc"));
        let original = std::fs::read(&doc_path).unwrap();
        let dir = FsDirectory::open(&dst);
        assert!(
            check_segment(&dir, &commit).all_passed(),
            "the fixture must start clean: {:?}",
            check_segment(&dir, &commit).failures()
        );

        // Every corruption is re-signed, so `file:*`'s CRC cannot "catch" it
        // and only the semantic checks can. The previous shape of this test
        // left the footer stale, which made `!failed.is_empty()` trivially
        // true for every flip and reduced the real claim to
        // `caught_by_advance > 0` -- an assertion that survives a regression
        // to one caught byte in three thousand.
        //
        // This loop used to run under `catch_unwind` with a silenced panic
        // hook, because a corrupt `.doc` could trip a `debug_assert_eq!` on
        // a value read off disk and panic in a debug build -- and a panicking
        // case was `continue`d, i.e. excluded from the assertion. Batch c8
        // converted those four sites to typed `Corrupted` errors, so the
        // escape hatch is gone and every byte flip is now actually checked.
        let body_end = original.len() - lucene_store::codec_util::FOOTER_LENGTH;
        let mut caught_by_advance = 0usize;
        let mut caught_by_other = 0usize;
        let mut accepted = 0usize;
        let mut isolated: Option<(usize, u8)> = None;
        for off in 48..body_end {
            for mask in [0x01u8, 0x80, 0xff] {
                let mut bytes = original.clone();
                bytes[off] ^= mask;
                repair_checksum(&mut bytes);
                std::fs::write(&doc_path, &bytes).unwrap();
                let dir = FsDirectory::open(&dst);
                let result = check_segment(&dir, &commit);
                let failed: Vec<&str> = result.failures().iter().map(|c| c.name.as_str()).collect();
                if failed.contains(&"postings.advance_agrees:body") {
                    caught_by_advance += 1;
                    if failed.len() == 1 {
                        isolated.get_or_insert((off, mask));
                    }
                } else if failed.is_empty() {
                    accepted += 1;
                } else {
                    caught_by_other += 1;
                }
            }
        }
        std::fs::write(&doc_path, &original).unwrap();
        let total = caught_by_advance + caught_by_other + accepted;
        // Measured when this was written, over 2 034 re-signed single-byte
        // `.doc` corruptions of a 1 000-document term: **5** were rejected by
        // `postings.advance_agrees` specifically, 2 026 by another postings
        // check, and **3** were accepted by every check in the module.
        //
        // The second number is the one worth asserting, and it is the one the
        // old `caught_by_advance > 0` shape could not state: with the CRC out
        // of the picture, all but 3 of 2 034 corruptions of a `.doc` are
        // still rejected on their semantics alone.
        assert!(
            accepted <= 3,
            "{accepted} of {total} re-signed .doc corruptions were accepted by every check; \
             3 were when this test was written"
        );
        assert!(
            caught_by_advance >= 4,
            "postings.advance_agrees rejected only {caught_by_advance} of {total} re-signed \
             .doc corruptions ({caught_by_other} caught by another check, {accepted} accepted); \
             5 were rejected when this test was written"
        );
        // No isolation assertion, deliberately, and this is a real property
        // of the check rather than a gap in the test: **none** of the 2 034
        // corruptions tripped `postings.advance_agrees` alone. A `.doc` byte
        // that makes the skip list disagree with the decoded blocks almost
        // always also makes a doc ID non-increasing or out of range, which
        // `postings.doc_ids_valid` sees first. The check is therefore a
        // *second* witness rather than a sole one -- which is still worth
        // having (it is the only place the skip data and the blocks it
        // indexes are compared) but is not falsifiable in isolation by byte
        // corruption. Recorded here so the next reader does not add an
        // isolation assertion that cannot hold.
        assert!(
            isolated.is_none(),
            "a .doc corruption now trips postings.advance_agrees alone (byte {isolated:?}); \
             none did when this test was written -- if that is a real improvement, assert the \
             isolation here instead of this line"
        );

        let dir = FsDirectory::open(&dst);
        assert!(check_segment(&dir, &commit).all_passed());
    }

    /// Negative control for `postings.seek_agrees`: the `.tip` trie is the
    /// one structure a `seekExact`/`seekCeil` consults and a forward scan
    /// does not, so corrupting it is exactly the case where the two
    /// disagree -- a term the scan enumerated that a seek can no longer
    /// find, or a `seekCeil` that lands on the wrong term. `try_seek_exact`
    /// / `try_seek_ceil` (batch c1's error-carrying forms) are what let that
    /// surface as a failure instead of a silent "not found".
    #[test]
    fn corrupting_the_term_index_is_caught_by_the_seek_check() {
        let dst = copy_fixture("blocktree_index");
        let tip_path = find_file(&dst, ".tip");
        let original = std::fs::read(&tip_path).unwrap();
        let body_end = original.len() - lucene_store::codec_util::FOOTER_LENGTH;
        let mut caught_by_seek = 0usize;
        let mut caught_by_other = 0usize;
        let mut accepted = 0usize;
        let mut isolated: Option<(usize, u8)> = None;
        // Sampled rather than exhaustive: a full check of this 8 959-document
        // fixture costs ~20 ms, and an every-byte x every-mask sweep would
        // dominate the test suite. Every fourth byte still covers every
        // structural region of the trie.
        //
        // Re-signed, like every other negative control in this module since
        // c15: with a stale footer the `file:*` CRC catches every flip, so
        // `!failed.is_empty()` is trivially true and `caught_by_seek > 0` is
        // the only real claim left -- one that survives a regression to a
        // single caught byte.
        for off in (48..body_end).step_by(4) {
            for mask in [0x01u8, 0x80, 0xff] {
                let mut bytes = original.clone();
                bytes[off] ^= mask;
                repair_checksum(&mut bytes);
                std::fs::write(&tip_path, &bytes).unwrap();
                let dir = FsDirectory::open(&dst);
                // A corruption that makes the *commit* unreadable is still a
                // corruption this module rejected -- counting it as
                // "caught by another check" keeps `total` equal to the sweep's
                // own iteration count, which is what makes the floors below
                // mean what they say.
                let failed = match check_directory(&dir) {
                    Ok(r) => failed_names(&r),
                    Err(e) => vec![format!("commit.unreadable: {e}")],
                };
                let is_seek = |n: &String| n.starts_with("postings.seek_agrees:");
                if failed.iter().any(is_seek) {
                    caught_by_seek += 1;
                    if failed.iter().all(is_seek) {
                        isolated.get_or_insert((off, mask));
                    }
                } else if failed.is_empty() {
                    accepted += 1;
                } else {
                    caught_by_other += 1;
                }
            }
        }
        std::fs::write(&tip_path, &original).unwrap();
        let total = caught_by_seek + caught_by_other + accepted;
        // Measured when this was written, over 99 re-signed single-byte
        // `.tip` corruptions: **44** were rejected by `postings.seek_agrees`
        // specifically, 12 by another check, and 43 were accepted. The 43 are
        // expected and are a property of the format, not a hole: the `.tip`
        // trie is an *index* into the `.tim`, so many of its bytes only
        // change which block a seek starts scanning from, and a scan that
        // starts in the right block still finds the right term.
        assert!(
            caught_by_seek >= 38,
            "postings.seek_agrees rejected only {caught_by_seek} of {total} re-signed .tip \
             corruptions ({caught_by_other} caught by another check, {accepted} accepted); \
             44 were rejected when this test was written"
        );
        // The isolation: at least one corruption trips the seek check and
        // nothing else, which is what proves it is a witness in its own right
        // rather than riding on a neighbouring check's failure.
        let (off, mask) = isolated.expect(
            "no .tip corruption tripped postings.seek_agrees alone, so the check cannot be \
             distinguished from the checks around it",
        );
        let mut bytes = original.clone();
        bytes[off] ^= mask;
        repair_checksum(&mut bytes);
        std::fs::write(&tip_path, &bytes).unwrap();
        let dir = FsDirectory::open(&dst);
        let failed = failed_names(&check_directory(&dir).unwrap());
        std::fs::write(&tip_path, &original).unwrap();
        assert!(
            !failed.is_empty()
                && failed
                    .iter()
                    .all(|n| n.starts_with("postings.seek_agrees:")),
            "byte {off} ^ {mask:#x} was expected to trip only the seek check: {failed:?}"
        );
        let dir = FsDirectory::open(&dst);
        assert!(check_directory(&dir)
            .unwrap()
            .iter()
            .all(|r| r.all_passed()));
    }

    /// Negative control on real Java-written positional bytes: no
    /// single-byte corruption of `blocktree_index`'s `.pos` may pass every
    /// check. Before this batch nothing in `check_index` decoded a single
    /// position, so a `.pos` was only ever validated by its footer -- and
    /// the footer check was shape-only.
    #[test]
    fn no_corruption_of_a_real_pos_file_is_silently_accepted() {
        let dst = copy_fixture("blocktree_index");
        let pos_path = find_file(&dst, ".pos");
        let original = std::fs::read(&pos_path).unwrap();
        let payload = 48..(original.len() - lucene_store::codec_util::FOOTER_LENGTH);
        for off in payload {
            for mask in [0x01u8, 0x80, 0xff] {
                let mut bytes = original.clone();
                bytes[off] ^= mask;
                std::fs::write(&pos_path, &bytes).unwrap();
                let dir = FsDirectory::open(&dst);
                let results = check_directory(&dir).unwrap();
                assert!(
                    !failed_names(&results).is_empty(),
                    "flipping .pos byte {off} with {mask:#x} was silently accepted"
                );
            }
        }
        std::fs::write(&pos_path, &original).unwrap();
        let dir = FsDirectory::open(&dst);
        assert!(
            check_directory(&dir)
                .unwrap()
                .iter()
                .all(|r| r.all_passed()),
            "restoring the original bytes must restore a clean run"
        );
    }

    /// Negative control for `doc_values.terms_sorted` and
    /// `doc_values.ords_dense`: no single-byte corruption of a SORTED
    /// field's `.dvd` may be silently accepted, and at least one must be
    /// caught by the ordinal-space checks specifically -- an ordinal nothing
    /// uses, or an ordinal-to-term dictionary that is no longer sorted (in
    /// which case ordinal comparison stops being term comparison, and every
    /// `SortedDocValues` range query and index sort built on it is wrong
    /// while every other check still passes).
    #[test]
    fn corrupted_sorted_dv_ordinal_space_is_caught() {
        let dst = copy_fixture("sorted_dv_index");
        let dvd_path = find_file(&dst, ".dvd");
        let original = std::fs::read(&dvd_path).unwrap();
        let dir = FsDirectory::open(&dst);
        assert!(
            check_directory(&dir)
                .unwrap()
                .iter()
                .all(|r| r.all_passed()),
            "the fixture must start clean"
        );

        // The footer is recomputed over every corruption, so the file is
        // perfectly well-formed and only the *semantic* checks can reject it.
        // Without that, `file:*`'s CRC catches every flip on its own and a
        // `caught_by_ords > 0` assertion says nothing about the ordinal-space
        // checks -- the weakness c15 fixed in its own `.dvs` control and
        // recorded here.
        let body_end = original.len() - lucene_store::codec_util::FOOTER_LENGTH;
        let mut caught_by_ords = 0usize;
        let mut caught_by_other = 0usize;
        let mut accepted = 0usize;
        let mut isolated: Option<(usize, u8)> = None;
        for off in 48..body_end {
            for mask in [0x01u8, 0x80, 0xff] {
                let mut bytes = original.clone();
                bytes[off] ^= mask;
                repair_checksum(&mut bytes);
                std::fs::write(&dvd_path, &bytes).unwrap();
                let dir = FsDirectory::open(&dst);
                // A corruption that makes the *commit* unreadable is still a
                // corruption this module rejected -- counting it as
                // "caught by another check" keeps `total` equal to the sweep's
                // own iteration count, which is what makes the floors below
                // mean what they say.
                let failed = match check_directory(&dir) {
                    Ok(r) => failed_names(&r),
                    Err(e) => vec![format!("commit.unreadable: {e}")],
                };
                let is_ord = |n: &String| {
                    n.starts_with("doc_values.terms_sorted:")
                        || n.starts_with("doc_values.ords_dense:")
                };
                if failed.iter().any(is_ord) {
                    caught_by_ords += 1;
                    if failed.iter().all(is_ord) {
                        isolated.get_or_insert((off, mask));
                    }
                } else if failed.is_empty() {
                    accepted += 1;
                } else {
                    caught_by_other += 1;
                }
            }
        }
        std::fs::write(&dvd_path, &original).unwrap();
        // A floor with the measured number in it, not `> 0`: `> 0` survives a
        // regression that leaves a single corruption in a hundred caught.
        // Measured when this was written: of 99 re-signed single-byte `.dvd`
        // corruptions, **18** were rejected by the ordinal-space checks
        // specifically, 27 by another doc-values check, and 54 were accepted.
        // The 54 are real and expected: `sorted_dv_index`'s `.dvd` is 97
        // bytes of which most are the term dictionary's own *bytes*, and
        // changing a byte inside a term leaves the ordinal space -- which is
        // about ordinal density and term *order* -- perfectly valid. That is
        // precisely why re-signing matters: unsigned, all 99 would have been
        // "caught", by the CRC, and the number would have said nothing.
        assert!(
            caught_by_ords >= 15,
            "the ordinal-space checks rejected only {caught_by_ords} of {} re-signed .dvd \
             corruptions ({caught_by_other} caught by another check, {accepted} accepted); \
             18 were rejected when this test was written",
            caught_by_ords + caught_by_other + accepted
        );

        // The wiring, and the isolation: at least one corruption trips the
        // ordinal-space checks *and nothing else*, which is what proves the
        // checks are not riding on a neighbour's failure.
        let (off, mask) = isolated.expect(
            "no .dvd corruption tripped the ordinal-space checks alone, so they cannot be \
             distinguished from the checks around them",
        );
        let mut bytes = original.clone();
        bytes[off] ^= mask;
        repair_checksum(&mut bytes);
        std::fs::write(&dvd_path, &bytes).unwrap();
        let dir = FsDirectory::open(&dst);
        let failed = failed_names(&check_directory(&dir).unwrap());
        std::fs::write(&dvd_path, &original).unwrap();
        assert!(
            failed
                .iter()
                .all(|n| n.starts_with("doc_values.terms_sorted:")
                    || n.starts_with("doc_values.ords_dense:")),
            "byte {off} ^ {mask:#x} was expected to trip only the ordinal-space checks: \
             {failed:?}"
        );

        let dir = FsDirectory::open(&dst);
        assert!(check_directory(&dir)
            .unwrap()
            .iter()
            .all(|r| r.all_passed()));
    }

    /// The doc-values skipper check must actually run, and be non-vacuous, on
    /// the one fixture that has a `.dvs` -- a check that silently never fires
    /// is worse than no check.
    #[test]
    fn skipper_checks_actually_run_on_the_skip_index_fixture() {
        let dir = FsDirectory::open(fixture_dir("doc_values_skip_index"));
        let results = check_directory(&dir).unwrap();
        let skipper_checks: Vec<&Check> = results
            .iter()
            .flat_map(|r| r.checks.iter())
            .filter(|c| c.name.starts_with("doc_values.skipper:"))
            .collect();
        assert!(
            !skipper_checks.is_empty(),
            "expected a doc_values.skipper check on the skip-index fixture, got {:?}",
            results
                .iter()
                .flat_map(|r| r.checks.iter().map(|c| c.name.clone()))
                .collect::<Vec<_>>()
        );
        assert!(
            skipper_checks.iter().all(|c| c.passed()),
            "the skip-index fixture must pass its own skipper check: {skipper_checks:?}"
        );
        // That the walk is non-vacuous -- that it really visits intervals and
        // really tests their bounds -- is what
        // `corrupting_the_doc_values_skipper_is_caught_by_the_skipper_check`
        // proves; a passing check on a good file cannot prove it alone.
    }

    /// Negative control for `doc_values.skipper`: a `.dvs` whose *contents*
    /// lie but whose CRC is honest.
    ///
    /// Every other negative control in this module flips a byte and leaves the
    /// footer stale, which the `file:*` checksum check catches on its own --
    /// useless for proving a *semantic* check works. Here the footer is
    /// recomputed over the corrupted bytes, so the file is perfectly
    /// well-formed and only the skip index's own invariants (each level's
    /// bounds nested inside the global ones, doc ranges not inverted, the
    /// per-interval doc counts summing to the declared `docCount`) can catch
    /// it. Before this batch the `.dvs` was CRC-verified and never
    /// interpreted, so every one of these passed.
    ///
    /// The sweep runs against the check function directly, because a
    /// `check_directory` call over this fixture costs ~0.9 s and there are
    /// hundreds of corruptions; one of the corruptions it finds is then put
    /// through the whole of `check_directory` to prove the wiring.
    #[test]
    fn corrupting_the_doc_values_skipper_is_caught_by_the_skipper_check() {
        let dst = copy_fixture("doc_values_skip_index");
        let dvs_path = find_file(&dst, ".dvs");
        let original = std::fs::read(&dvs_path).unwrap();
        let dir = FsDirectory::open(&dst);
        let before = check_directory(&dir).unwrap();
        assert!(
            before.iter().all(|r| r.all_passed()),
            "the fixture must start clean: {:?}",
            failed_names(&before)
        );

        // Everything the sweep needs to parse a `.dvs` on its own, read once.
        let infos = segment_infos::read_latest(&dir).unwrap();
        let commit = &infos.segments[0];
        let si = open_si(&dir, commit).unwrap();
        let fnm = open_fnm(&dir, commit, &si).unwrap();
        let dvm_name = si.files.iter().find(|f| f.ends_with(".dvm")).unwrap();
        let suffix = dvm_name
            .strip_prefix(&format!("{}_", commit.segment_name))
            .and_then(|s| s.strip_suffix(".dvm"))
            .unwrap_or_default()
            .to_string();
        let dvm = dir.open(dvm_name).unwrap();
        let (_version, meta) =
            doc_values::parse_meta(&dvm, &commit.segment_id, &suffix, &fnm).unwrap();
        let field = fnm
            .fields
            .iter()
            .find(|f| f.doc_values_skip_index_type != field_infos::DocValuesSkipIndexType::None)
            .expect("the fixture has a skip-index field");
        let skipper_meta = meta
            .skipper_meta(field.number)
            .expect("the .dvm carries its skipper entry");

        // Past the index header (codec name, version, segment id, suffix),
        // whose own validation would catch a flip without needing any of the
        // semantic checks.
        let body_start = 64usize;
        let body_end = original.len() - lucene_store::codec_util::FOOTER_LENGTH;
        let resign = |off: usize, mask: u8| {
            let mut bytes = original[..body_end].to_vec();
            bytes[off] ^= mask;
            lucene_store::codec_util::write_footer(&mut bytes);
            bytes
        };

        let mut caught = 0usize;
        let mut accepted = 0usize;
        let mut first_caught: Option<(usize, u8)> = None;
        for off in body_start..body_end {
            for mask in [0x01u8, 0xff] {
                let bytes = resign(off, mask);
                let rejected = match doc_values::parse_skip_index(
                    &bytes,
                    &commit.segment_id,
                    &suffix,
                    skipper_meta,
                ) {
                    // A structurally impossible skip index (a zero or
                    // over-large level count, a truncated level body) is the
                    // decoder's own rejection, not this check's.
                    Err(_) => false,
                    Ok(index) => !check_doc_value_skipper(&index).is_empty(),
                };
                if rejected {
                    caught += 1;
                    first_caught.get_or_insert((off, mask));
                } else {
                    accepted += 1;
                }
            }
        }
        // A floor with the measured number in it, not `> 0`: `> 0` survives a
        // regression that leaves one check firing out of nine. 428 of 574
        // single-byte corruptions were rejected when this was written; the
        // other 146 are genuinely harmless (a bound moved but stayed nested
        // inside the global range).
        assert!(
            caught >= 400,
            "the skipper checks rejected only {caught} of {} re-signed .dvs corruptions \
             ({accepted} accepted); 428 were rejected when this test was written",
            caught + accepted
        );

        // The wiring: one corruption the checks reject must come back out of
        // `check_directory` as a `doc_values.skipper` failure -- and as
        // *only* that, since the file's CRC is honest and every other check
        // reads a different file.
        let (off, mask) = first_caught.expect("at least one corruption was caught");
        std::fs::write(&dvs_path, resign(off, mask)).unwrap();
        let dir = FsDirectory::open(&dst);
        let failed = failed_names(&check_directory(&dir).unwrap());
        std::fs::write(&dvs_path, &original).unwrap();
        assert!(
            failed.iter().any(|n| n.starts_with("doc_values.skipper:")),
            "a .dvs corruption with an honest CRC was not reported by the skipper check \
             (byte {off} ^ {mask:#x}); failures: {failed:?}"
        );
        assert!(
            failed.iter().all(|n| n.starts_with("doc_values.skipper:")),
            "the corruption was expected to trip only the skipper check: {failed:?}"
        );

        let dir = FsDirectory::open(&dst);
        assert!(check_directory(&dir)
            .unwrap()
            .iter()
            .all(|r| r.all_passed()));
    }

    /// Negative control for `norms.entries_name_real_norms_fields`: a `.nvm`
    /// whose entry names a field number the `.fnm` does not have.
    ///
    /// Java's `Lucene90NormsProducer.readFields` rejects the segment outright
    /// for this; this port accepted it, and the consequence is silent rather
    /// than loud -- the entry becomes unreachable, so every norm lookup for
    /// the *real* field falls back to "this field has no norms" and every
    /// score is computed against a default norm. Nothing else in this module
    /// notices, because `norms.entry_present` only looks the other way (a
    /// field with no entry), and the file's CRC is re-signed here so the
    /// integrity checks stay clean.
    #[test]
    fn a_nvm_entry_for_a_nonexistent_field_is_caught() {
        let dst = copy_fixture("norms_index");
        let nvm_path = find_file(&dst, ".nvm");
        let original = std::fs::read(&nvm_path).unwrap();
        let dir = FsDirectory::open(&dst);
        let before = check_directory(&dir).unwrap();
        assert!(
            before.iter().all(|r| r.all_passed()),
            "the fixture must start clean: {:?}",
            failed_names(&before)
        );

        // The first entry's field number sits immediately after the index
        // header (magic, codec name, version, segment id, empty suffix).
        let infos = segment_infos::read_latest(&dir).unwrap();
        let commit = &infos.segments[0];
        let (_version, norms) = norms::parse_meta(&original, &commit.segment_id, "").unwrap();
        let first = norms.entries[0].field_number;
        const FIELD_NUMBER_OFFSET: usize = 47;
        assert_eq!(
            i32::from_le_bytes(
                original[FIELD_NUMBER_OFFSET..FIELD_NUMBER_OFFSET + 4]
                    .try_into()
                    .unwrap()
            ),
            first,
            "the .nvm's first field number is not where this test patches it"
        );

        let body_end = original.len() - lucene_store::codec_util::FOOTER_LENGTH;
        let mut bytes = original[..body_end].to_vec();
        // 4242: a field number no `.fnm` in this fixture has.
        bytes[FIELD_NUMBER_OFFSET..FIELD_NUMBER_OFFSET + 4].copy_from_slice(&4242i32.to_le_bytes());
        lucene_store::codec_util::write_footer(&mut bytes);
        std::fs::write(&nvm_path, &bytes).unwrap();

        let dir = FsDirectory::open(&dst);
        let failed = failed_names(&check_directory(&dir).unwrap());
        std::fs::write(&nvm_path, &original).unwrap();
        assert!(
            failed
                .iter()
                .any(|n| n == "norms.entries_name_real_norms_fields"),
            "a .nvm naming a nonexistent field was not caught: {failed:?}"
        );
    }

    /// Negative control for `norms.agree_with_postings`: zeroing the norm of
    /// a document that does have terms is exactly the "norm value is 0,
    /// which may only be used on documents that have no terms" case Java
    /// rejects, and nothing else in this module would notice it.
    #[test]
    fn a_zeroed_norm_for_a_doc_with_terms_is_caught() {
        let dst = copy_fixture("norms_index");
        let nvd_path = find_file(&dst, ".nvd");
        let original = std::fs::read(&nvd_path).unwrap();
        let dir = FsDirectory::open(&dst);
        let before = check_directory(&dir).unwrap();
        assert!(
            before.iter().all(|r| r.all_passed()),
            "the fixture must start clean: {:?}",
            failed_names(&before)
        );

        // The norms payload is a flat `numDocsWithField * bytesPerNorm`
        // array; zeroing any byte of it zeroes some document's norm.
        let mut zeroed = 0usize;
        for off in 0..(original.len() - lucene_store::codec_util::FOOTER_LENGTH) {
            if original[off] == 0 {
                continue;
            }
            let mut bytes = original.clone();
            bytes[off] = 0;
            std::fs::write(&nvd_path, &bytes).unwrap();
            let dir = FsDirectory::open(&dst);
            let failed = failed_names(&check_directory(&dir).unwrap());
            if failed
                .iter()
                .any(|n| n.starts_with("norms.agree_with_postings:"))
            {
                zeroed += 1;
                break;
            }
        }
        std::fs::write(&nvd_path, &original).unwrap();
        assert!(
            zeroed > 0,
            "zeroing a norm byte was never caught by norms.agree_with_postings"
        );
    }

    /// Negative control for the norms *value* pass: truncating the `.nvd`
    /// payload region so a norm read runs off the end must be reported by
    /// `norms.values_decode`, not silently skipped.
    #[test]
    fn a_norms_file_that_cannot_be_read_is_caught_by_values_decode() {
        let dst = copy_fixture("norms_index");
        let nvm_path = find_file(&dst, ".nvm");
        let original = std::fs::read(&nvm_path).unwrap();
        let body_end = original.len() - lucene_store::codec_util::FOOTER_LENGTH;
        let mut caught_by_norms = 0usize;
        let mut caught_by_other = 0usize;
        let mut accepted = 0usize;
        let mut isolated: Option<usize> = None;
        for off in 24..body_end {
            let mut bytes = original.clone();
            bytes[off] ^= 0x11;
            repair_checksum(&mut bytes);
            std::fs::write(&nvm_path, &bytes).unwrap();
            let dir = FsDirectory::open(&dst);
            // See the note in the `.tip` sweep: an unreadable commit counts
            // as caught by another check, not as a skipped iteration.
            let failed = match check_directory(&dir) {
                Ok(r) => failed_names(&r),
                Err(e) => vec![format!("commit.unreadable: {e}")],
            };
            let is_norms = |n: &String| n.starts_with("norms.");
            if failed.iter().any(is_norms) {
                caught_by_norms += 1;
                if failed.iter().all(is_norms) {
                    isolated.get_or_insert(off);
                }
            } else if failed.is_empty() {
                accepted += 1;
            } else {
                caught_by_other += 1;
            }
        }
        std::fs::write(&nvm_path, &original).unwrap();
        let total = caught_by_norms + caught_by_other + accepted;
        // Measured when this was written, over 99 re-signed single-byte
        // `.nvm` corruptions: **85** were rejected by a `norms.*` check, 0 by
        // any other check, and 14 accepted. The zero is the interesting
        // number: the `.nvm` is read by nothing else in this module, so if
        // the norms checks do not catch a corruption of it, nothing does --
        // which is exactly why a `> 0` floor here was worth replacing.
        assert!(
            caught_by_norms >= 75,
            "the norms checks rejected only {caught_by_norms} of {total} re-signed .nvm \
             corruptions ({caught_by_other} caught by another check, {accepted} accepted); \
             85 were rejected when this test was written"
        );
        assert_eq!(
            caught_by_other, 0,
            "a .nvm corruption was reported by something other than a norms.* check; the \
             norms checks are meant to be the only reader of this file"
        );
        let off = isolated.expect("at least one .nvm corruption is caught by norms.* alone");
        let mut bytes = original.clone();
        bytes[off] ^= 0x11;
        repair_checksum(&mut bytes);
        std::fs::write(&nvm_path, &bytes).unwrap();
        let dir = FsDirectory::open(&dst);
        let failed = failed_names(&check_directory(&dir).unwrap());
        std::fs::write(&nvm_path, &original).unwrap();
        assert!(
            !failed.is_empty() && failed.iter().all(|n| n.starts_with("norms.")),
            "byte {off} was expected to trip only the norms checks: {failed:?}"
        );
    }

    /// Overwrites the codec footer's CRC-32 with the one the (possibly
    /// modified) payload actually hashes to, so a deliberately corrupted
    /// file still passes `CodecUtil.checksumEntireFile`.
    ///
    /// Without this a corruption test of `.vec`/`.vex` proves nothing about
    /// the structural checks: `FlatVectorsReader::open` and
    /// `HnswVectorsReader::open` both verify the *whole file's* checksum
    /// before parsing a single field entry (c5 ported that from
    /// `Lucene99FlatVectorsReader`/`Lucene99HnswVectorsReader`), so every
    /// byte flip surfaces as `vectors.open`/`hnsw.open` and never reaches
    /// the graph. Repairing the checksum isolates the check under test from
    /// the checksum, which is what "corrupt exactly the thing this check is
    /// meant to catch" requires here.
    fn repair_checksum(bytes: &mut [u8]) {
        let n = bytes.len();
        let crc = crc32fast::hash(&bytes[..n - 8]) as u64;
        bytes[n - 8..].copy_from_slice(&crc.to_be_bytes());
    }

    /// Negative control for the HNSW graph checks. The `.vex` graph region
    /// is corrupted and the file's checksum repaired (see
    /// [`repair_checksum`]), so the corruption reaches the graph walk rather
    /// than being stopped by the reader's whole-file CRC. At least one
    /// corruption must be reported by `hnsw.neighbors_on_level` or
    /// `hnsw.neighbors_sorted` -- a neighbour that is not on its own level,
    /// out of the ordinal range, repeated, or out of order -- which is
    /// exactly what Java's `testHnswGraph` rejects and what nothing else in
    /// this module would notice.
    #[test]
    fn corrupted_hnsw_neighbours_are_caught_by_the_graph_checks() {
        let dst = copy_fixture("vectors_index");
        let vex_path = find_file(&dst, ".vex");
        let original = std::fs::read(&vex_path).unwrap();
        let body_end = original.len() - 16;
        let mut caught_by_graph = 0usize;
        let mut caught_by_other = 0usize;
        let mut accepted = 0usize;
        let mut reached_the_graph = 0usize;
        let mut isolated: Option<(usize, u8)> = None;
        // A fixed, uniform sample of the 223 kB `.vex` rather than an
        // early-exit-at-three walk: the old shape stopped as soon as it had
        // three hits, so the count it asserted (`> 0`) could never say how
        // far the checks actually reach. Every 2 111th byte x three masks is
        // ~105 x 3 whole-directory checks over a 4 000-document vectors
        // index, which is the affordable budget here.
        for off in (48..body_end).step_by(2111) {
            for mask in [0x01u8, 0x80, 0xff] {
                let mut bytes = original.clone();
                bytes[off] ^= mask;
                repair_checksum(&mut bytes);
                std::fs::write(&vex_path, &bytes).unwrap();
                let dir = FsDirectory::open(&dst);
                // A corruption that makes the *commit* unreadable is still a
                // corruption this module rejected -- counting it as
                // "caught by another check" keeps `total` equal to the sweep's
                // own iteration count, which is what makes the floors below
                // mean what they say.
                let failed = match check_directory(&dir) {
                    Ok(r) => failed_names(&r),
                    Err(e) => vec![format!("commit.unreadable: {e}")],
                };
                if !failed.iter().any(|n| n == "hnsw.open") {
                    reached_the_graph += 1;
                }
                let is_graph = |n: &String| {
                    n.starts_with("hnsw.neighbors_on_level:")
                        || n.starts_with("hnsw.neighbors_sorted:")
                };
                if failed.iter().any(is_graph) {
                    caught_by_graph += 1;
                    if failed.iter().all(is_graph) {
                        isolated.get_or_insert((off, mask));
                    }
                } else if failed.is_empty() {
                    accepted += 1;
                } else {
                    caught_by_other += 1;
                }
            }
        }
        std::fs::write(&vex_path, &original).unwrap();
        let total = caught_by_graph + caught_by_other + accepted;
        // Measured when this was written, over 318 re-signed single-byte
        // `.vex` corruptions (106 offsets x 3 masks): **315** got past
        // `hnsw.open` into the graph itself, **138** were rejected by the
        // neighbour checks specifically, 3 by another check, and 177 were
        // accepted -- the `.vex` is mostly packed neighbour ordinals, and a
        // flip that keeps an ordinal in range, distinct and in order is a
        // different but still well-formed graph, which no structural check
        // can or should reject.
        assert!(
            reached_the_graph >= 300,
            "only {reached_the_graph} of {total} corruptions got past hnsw.open, so the graph \
             checks were barely exercised; 315 did when this test was written"
        );
        assert!(
            caught_by_graph >= 120,
            "the neighbour checks rejected only {caught_by_graph} of {total} re-signed .vex \
             corruptions ({caught_by_other} caught by another check, {accepted} accepted); \
             138 were rejected when this test was written"
        );
        let (off, mask) =
            isolated.expect("at least one .vex corruption is caught by the neighbour checks alone");
        let mut bytes = original.clone();
        bytes[off] ^= mask;
        repair_checksum(&mut bytes);
        std::fs::write(&vex_path, &bytes).unwrap();
        let dir = FsDirectory::open(&dst);
        let failed = failed_names(&check_directory(&dir).unwrap());
        std::fs::write(&vex_path, &original).unwrap();
        assert!(
            !failed.is_empty()
                && failed
                    .iter()
                    .all(|n| n.starts_with("hnsw.neighbors_on_level:")
                        || n.starts_with("hnsw.neighbors_sorted:")),
            "byte {off} ^ {mask:#x} was expected to trip only the neighbour checks: {failed:?}"
        );
        let dir = FsDirectory::open(&dst);
        assert!(check_directory(&dir)
            .unwrap()
            .iter()
            .all(|r| r.all_passed()));
    }

    /// The complement: an *unrepaired* `.vex` byte flip must still be caught,
    /// by the reader's own whole-file checksum and by the `file:*` integrity
    /// check. Nothing gets through either way.
    #[test]
    fn corrupted_hnsw_graph_bytes_are_never_silently_accepted() {
        let dst = copy_fixture("vectors_index");
        let vex_path = find_file(&dst, ".vex");
        let original = std::fs::read(&vex_path).unwrap();
        for off in (48..(original.len() - 16)).step_by(997) {
            let mut bytes = original.clone();
            bytes[off] ^= 0x37;
            std::fs::write(&vex_path, &bytes).unwrap();
            let dir = FsDirectory::open(&dst);
            let failed = failed_names(&check_directory(&dir).unwrap());
            assert!(
                failed.iter().any(|n| n == "hnsw.open"),
                "flipping .vex byte {off} was not reported by hnsw.open: {failed:?}"
            );
        }
        std::fs::write(&vex_path, &original).unwrap();
    }

    /// Negative control for `vectors.ord_to_doc`: the ordinal-to-document
    /// mapping lives in `.vemf` (a `DirectMonotonic` block plus, for a
    /// sparse field, an `IndexedDISI`), so that is what gets corrupted --
    /// with the checksum repaired (see [`repair_checksum`]) so the
    /// corruption reaches the mapping rather than being stopped by
    /// `FlatVectorsReader::open`'s whole-file CRC. Nothing may be silently
    /// accepted, and at least one corruption must be reported by a
    /// `vectors.*` check.
    #[test]
    fn corrupted_vector_ord_to_doc_mapping_is_caught() {
        let dst = copy_fixture("vectors_index");
        let vemf_path = find_file(&dst, ".vemf");
        let original = std::fs::read(&vemf_path).unwrap();
        let mut caught_by_vectors = 0;
        let mut reached_the_fields = 0;
        'sweep: for off in 48..(original.len() - 16) {
            for mask in [0x01u8, 0x80, 0xff] {
                if caught_by_vectors >= 3 && reached_the_fields >= 3 {
                    break 'sweep;
                }
                let mut bytes = original.clone();
                bytes[off] ^= mask;
                repair_checksum(&mut bytes);
                std::fs::write(&vemf_path, &bytes).unwrap();
                let dir = FsDirectory::open(&dst);
                // A corruption that makes the *commit* unreadable is still a
                // corruption this module rejected -- counting it as
                // "caught by another check" keeps `total` equal to the sweep's
                // own iteration count, which is what makes the floors below
                // mean what they say.
                let failed = match check_directory(&dir) {
                    Ok(r) => failed_names(&r),
                    Err(e) => vec![format!("commit.unreadable: {e}")],
                };
                // Note the *absence* of a "nothing is silently accepted"
                // assertion here, unlike the `.pos`/`.tim`/`.doc` sweeps:
                // `.vemf` is pure metadata with no redundancy, so some of
                // its bytes (a `DirectMonotonic` block shift that still
                // decodes to a monotone in-range doc list, say) have no
                // second copy to disagree with. Their guard is the
                // checksum, which is exactly why
                // `Lucene99FlatVectorsReader` verifies the whole file at
                // open and why `file:*` now does a full CRC too --
                // `corrupted_vector_data_is_caught` pins that direction.
                // Repairing the checksum removes that guard on purpose, to
                // reach the per-field checks at all.
                if !failed.iter().any(|n| n == "vectors.open") {
                    reached_the_fields += 1;
                }
                if failed.iter().any(|n| {
                    n.starts_with("vectors.ord_to_doc:")
                        || n.starts_with("vectors.field_entry_matches_fnm:")
                        || n.starts_with("vectors.values_decode:")
                }) {
                    caught_by_vectors += 1;
                }
            }
        }
        std::fs::write(&vemf_path, &original).unwrap();
        assert!(
            reached_the_fields > 0,
            "no corruption got past vectors.open, so the field checks were never exercised"
        );
        assert!(
            caught_by_vectors > 0,
            "no .vemf corruption was caught by a per-field vectors check"
        );
        let dir = FsDirectory::open(&dst);
        assert!(check_directory(&dir)
            .unwrap()
            .iter()
            .all(|r| r.all_passed()));
    }

    /// Negative control for the vector-values pass: corrupting the `.vec`
    /// payload must be reported, and the vectors reader's own whole-file
    /// checksum is what reports it (`vectors.open`) -- the point of the test
    /// is that the vectors are opened and read at all, which before this
    /// batch they were not.
    #[test]
    fn corrupted_vector_data_is_caught() {
        let dst = copy_fixture("vectors_index");
        let vec_path = find_file(&dst, ".vec");
        let original = std::fs::read(&vec_path).unwrap();
        let mut bytes = original.clone();
        let off = original.len() / 2;
        bytes[off] ^= 0xff;
        std::fs::write(&vec_path, &bytes).unwrap();
        let dir = FsDirectory::open(&dst);
        let failed = failed_names(&check_directory(&dir).unwrap());
        assert!(
            failed.iter().any(|n| n == "vectors.open"),
            "expected vectors.open to fail, got {failed:?}"
        );
        std::fs::write(&vec_path, &original).unwrap();
    }

    /// Negative control for `term_vectors.match_postings`: rewriting the
    /// segment's postings so a term vector's term is no longer in the
    /// inverted index must be reported. Corrupting the `.tvd` alone would
    /// be caught by the term-vectors decoder; corrupting the postings side
    /// is what exercises the cross-check.
    #[test]
    fn a_term_vector_term_missing_from_the_postings_is_caught() {
        let dst = copy_fixture("term_vectors_index");
        let tim_path = find_file(&dst, ".tim");
        let original = std::fs::read(&tim_path).unwrap();
        let mut caught = 0;
        for off in 40..(original.len() - lucene_store::codec_util::FOOTER_LENGTH) {
            let mut bytes = original.clone();
            bytes[off] ^= 0x21;
            std::fs::write(&tim_path, &bytes).unwrap();
            let dir = FsDirectory::open(&dst);
            // See the note in the `.tip` sweep: an unreadable commit counts
            // as caught by another check, not as a skipped iteration.
            let failed = match check_directory(&dir) {
                Ok(r) => failed_names(&r),
                Err(e) => vec![format!("commit.unreadable: {e}")],
            };
            assert!(
                !failed.is_empty(),
                "flipping .tim byte {off} was silently accepted"
            );
            if failed.iter().any(|n| n == "term_vectors.match_postings") {
                caught += 1;
            }
        }
        std::fs::write(&tim_path, &original).unwrap();
        assert!(
            caught > 0,
            "no .tim corruption was caught by term_vectors.match_postings"
        );
    }

    /// Negative control for the upgrade from `retrieve_checksum` (footer
    /// *shape* only) to `CodecUtil.checksumEntireFile` in the `file:*`
    /// checks: a byte flipped in the middle of a file, leaving every footer
    /// field intact, used to pass.
    #[test]
    fn a_payload_byte_flip_now_fails_the_file_integrity_check() {
        let dst = copy_fixture("points_index");
        let kdd_path = find_file(&dst, ".kdd");
        let original = std::fs::read(&kdd_path).unwrap();
        let mut bytes = original.clone();
        let off = original.len() / 2;
        bytes[off] ^= 0x01;
        std::fs::write(&kdd_path, &bytes).unwrap();
        let dir = FsDirectory::open(&dst);
        let failed = failed_names(&check_directory(&dir).unwrap());
        assert!(
            failed
                .iter()
                .any(|n| n.starts_with("file:") && n.ends_with(".kdd")),
            "expected the .kdd file integrity check to fail, got {failed:?}"
        );
        // The old shape-only check would not have noticed: prove that by
        // asserting the footer itself is still well-formed.
        assert!(lucene_store::codec_util::retrieve_checksum(&bytes).is_ok());
        std::fs::write(&kdd_path, &original).unwrap();
    }

    /// Same for term vectors.
    #[test]
    fn term_vector_checks_actually_run_on_a_term_vectors_fixture() {
        let dir = FsDirectory::open(fixture_dir("term_vectors_index"));
        let results = check_directory(&dir).unwrap();
        let names: Vec<&str> = results[1].checks.iter().map(|c| c.name.as_str()).collect();
        assert!(
            names.contains(&"term_vectors.every_doc_decodes"),
            "expected term_vectors checks, got {names:?}"
        );
        assert!(names.contains(&"term_vectors.fields_marked_in_fnm"));
    }

    /// Corrupting a byte inside the `.dvd` payload must be caught by the
    /// value-decode pass. Before this batch nothing in `check_index` read a
    /// single doc-values *value*, so this exact corruption passed clean.
    #[test]
    fn corrupted_doc_values_payload_is_caught() {
        let src_dir = fixture_dir("sorted_dv_index");
        let dst_dir = tempdir();
        for entry in std::fs::read_dir(&src_dir).unwrap() {
            let entry = entry.unwrap();
            std::fs::copy(entry.path(), dst_dir.join(entry.file_name())).unwrap();
        }
        let dvd = dst_dir.join("_0_Lucene90_0.dvd");
        let mut bytes = std::fs::read(&dvd).unwrap();
        // The ordinals live right after the index header; flipping the high
        // bits of an early payload byte pushes an ordinal out of the terms
        // dictionary's range.
        let at = 40;
        bytes[at] ^= 0xFF;
        std::fs::write(&dvd, &bytes).unwrap();

        let dir = FsDirectory::open(&dst_dir);
        let results = check_directory(&dir).unwrap();
        let failures: Vec<&str> = results[1]
            .failures()
            .iter()
            .map(|c| c.name.as_str())
            .collect();
        assert!(
            failures.iter().any(|n| n.starts_with("doc_values.")),
            "expected a doc_values failure, got {failures:?}"
        );

        std::fs::remove_dir_all(&dst_dir).ok();
    }

    /// A commit whose `counter` has fallen behind a live segment's name is
    /// how `IndexWriter` ends up reusing a name and clobbering a segment --
    /// real `CheckIndex`'s `validCounter`/`maxSegmentName` check.
    #[test]
    fn commit_counter_behind_segment_names_is_flagged() {
        let dir = FsDirectory::open(fixture_dir("blocktree_index"));
        let mut infos = segment_infos::read_latest(&dir).unwrap();
        infos.counter = 0; // segment `_0` parses to 0, so counter must be > 0
        let result = check_commit(&infos);
        let failure = result
            .checks
            .iter()
            .find(|c| c.name == "commit.counter_ahead_of_segment_names")
            .unwrap();
        assert!(!failure.passed(), "{failure:?}");
    }

    #[test]
    fn duplicate_segment_names_in_a_commit_are_flagged() {
        let dir = FsDirectory::open(fixture_dir("blocktree_index"));
        let mut infos = segment_infos::read_latest(&dir).unwrap();
        let dupe = infos.segments[0].clone();
        infos.segments.push(dupe);
        infos.counter = 99;
        let result = check_commit(&infos);
        assert!(!result
            .checks
            .iter()
            .find(|c| c.name == "commit.segment_names_unique")
            .unwrap()
            .passed());
    }

    /// `SegmentInfos.readCommit` refuses a `delCount` larger than the
    /// segment's own `maxDoc`. This port's `segment_infos::parse` cannot see
    /// `maxDoc`, so the check lives here -- and without it nothing validated
    /// it at all.
    #[test]
    fn del_count_larger_than_max_doc_is_flagged() {
        let dir = FsDirectory::open(fixture_dir("live_docs_index"));
        let mut commit = read_commit(&dir);
        commit.del_count = 1_000_000;
        let result = check_segment(&dir, &commit);
        assert!(!result
            .checks
            .iter()
            .find(|c| c.name == "commit.del_count_within_max_doc")
            .unwrap()
            .passed());
    }

    /// A commit that records deletions without a delete generation has no
    /// `.liv` file to hold them -- `SegmentCommitInfo.hasDeletions()` is
    /// `delGen != -1`.
    #[test]
    fn del_count_without_del_gen_is_flagged() {
        let dir = FsDirectory::open(fixture_dir("blocktree_index"));
        let mut commit = read_commit(&dir);
        assert_eq!(commit.del_gen, -1);
        commit.del_count = 3;
        let result = check_segment(&dir, &commit);
        assert!(!result
            .checks
            .iter()
            .find(|c| c.name == "commit.del_count_zero_without_del_gen")
            .unwrap()
            .passed());
    }

    // -- index sort / soft deletes --

    const SORT_SEG_ID: [u8; ID_LENGTH] = [21u8; ID_LENGTH];

    /// Writes a one-field segment (NUMERIC doc values, no postings/stored
    /// fields) declaring `index_sort` over that field, with `values` as the
    /// per-doc sort keys in doc-id order. `soft_deletes` marks the field as
    /// the soft-deletes field so the same fixture drives both new checks.
    fn write_sorted_dv_fixture(
        dst_dir: &std::path::Path,
        values: &[i64],
        sort: Option<Vec<segment_info::IndexSortField>>,
        soft_deletes: bool,
        soft_del_count: i32,
    ) -> segment_infos::SegmentCommitInfo {
        let max_doc = values.len() as i32;
        let fi = field_infos::FieldInfo {
            name: "ts".to_string(),
            number: 0,
            store_term_vectors: false,
            // A non-indexed field must not omit norms (`Lucene94FieldInfosFormat`
            // rejects that combination outright).
            omit_norms: false,
            store_payloads: false,
            soft_deletes_field: soft_deletes,
            parent_field: false,
            index_options: field_infos::IndexOptions::None,
            doc_values_type: field_infos::DocValuesType::Numeric,
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
        let fnm = field_infos::write(std::slice::from_ref(&fi), &SORT_SEG_ID, "");
        std::fs::write(dst_dir.join("_0.fnm"), &fnm).unwrap();
        let (dvm, dvd, dvs) = lucene_codecs::doc_values::write_single_dense_numeric_field(
            0,
            values,
            max_doc,
            &SORT_SEG_ID,
            "",
        )
        .unwrap();
        std::fs::write(dst_dir.join("_0.dvm"), &dvm).unwrap();
        std::fs::write(dst_dir.join("_0.dvd"), &dvd).unwrap();
        std::fs::write(dst_dir.join("_0.dvs"), &dvs).unwrap();

        let si = SegmentInfo {
            id: SORT_SEG_ID,
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
                "_0.si".to_string(),
                "_0.fnm".to_string(),
                "_0.dvm".to_string(),
                "_0.dvd".to_string(),
                "_0.dvs".to_string(),
            ],
            attributes: vec![],
            index_sort: sort,
        };
        std::fs::write(dst_dir.join("_0.si"), segment_info::write(&si, "")).unwrap();

        segment_infos::SegmentCommitInfo {
            segment_name: "_0".to_string(),
            segment_id: SORT_SEG_ID,
            codec_name: "Lucene104".to_string(),
            del_gen: -1,
            del_count: 0,
            field_infos_gen: -1,
            doc_values_gen: -1,
            soft_del_count,
            sci_id: None,
            field_infos_files: vec![],
            dv_update_files: vec![],
            ..Default::default()
        }
    }

    fn sort_asc() -> Option<Vec<segment_info::IndexSortField>> {
        Some(vec![segment_info::IndexSortField::long(
            "ts",
            false,
            Some(i64::MAX),
        )])
    }

    /// The same ascending sort over a **multi-valued** column: a
    /// `SortedNumericSortField` with the `MIN` selector, which is what Java's
    /// `SortedNumericSortField` produces and the only kind whose
    /// `getIndexSorter` reads a SORTED_NUMERIC column. A plain
    /// `SortField(ts, LONG)` over the same field is what `DocValues.getNumeric`
    /// throws on in Java, and what `sort_key_values` reports as a kind/type
    /// mismatch here.
    fn sorted_numeric_sort_asc() -> Option<Vec<segment_info::IndexSortField>> {
        Some(vec![segment_info::IndexSortField {
            field: "ts".to_string(),
            reverse: false,
            kind: segment_info::IndexSortKind::SortedNumeric {
                key: segment_info::NumericSortKey::Long(Some(i64::MAX)),
                selector: segment_info::SortedNumericSelector::Min,
            },
        }])
    }

    /// A segment whose doc values really are in the declared sort order must
    /// pass; one whose are not must fail. Before this batch nothing verified
    /// that a `.si`'s `indexSort` claim matched the docs' actual order --
    /// exactly the claim a sorted-index merge and an early-terminating query
    /// both silently rely on.
    #[test]
    fn index_sort_order_is_verified_against_the_actual_doc_values() {
        let ok_dir = tempdir();
        let commit = write_sorted_dv_fixture(&ok_dir, &[10, 20, 20, 30], sort_asc(), false, 0);
        let dir = FsDirectory::open(&ok_dir);
        let result = check_segment(&dir, &commit);
        assert!(
            result.all_passed(),
            "unexpected failures: {:?}",
            result.failures()
        );
        assert!(result
            .checks
            .iter()
            .any(|c| c.name == "sort.docs_in_index_sort_order"));
        std::fs::remove_dir_all(&ok_dir).ok();

        let bad_dir = tempdir();
        let commit = write_sorted_dv_fixture(&bad_dir, &[10, 30, 20, 40], sort_asc(), false, 0);
        let dir = FsDirectory::open(&bad_dir);
        let result = check_segment(&dir, &commit);
        let failure = result
            .checks
            .iter()
            .find(|c| c.name == "sort.docs_in_index_sort_order")
            .unwrap();
        assert!(!failure.passed(), "{failure:?}");
        assert!(failure.message.contains("sorts after"));
        std::fs::remove_dir_all(&bad_dir).ok();
    }

    /// An unsorted segment must not get a sort check at all (nothing to
    /// verify), rather than a vacuous pass.
    #[test]
    fn unsorted_segment_gets_no_sort_check() {
        let dst_dir = tempdir();
        let commit = write_sorted_dv_fixture(&dst_dir, &[3, 1, 2], None, false, 0);
        let dir = FsDirectory::open(&dst_dir);
        let result = check_segment(&dir, &commit);
        assert!(!result
            .checks
            .iter()
            .any(|c| c.name == "sort.docs_in_index_sort_order"));
        assert!(result.all_passed(), "{:?}", result.failures());
        std::fs::remove_dir_all(&dst_dir).ok();
    }

    /// `CheckIndex.checkSoftDeletes`: `softDelCount` must equal the number of
    /// live docs carrying a value for the soft-deletes field.
    #[test]
    fn soft_delete_count_is_verified_against_the_soft_deletes_field() {
        let dst_dir = tempdir();
        // Dense NUMERIC doc values: all 3 docs have a value -> 3 soft deletes.
        let commit = write_sorted_dv_fixture(&dst_dir, &[1, 1, 1], None, true, 3);
        let dir = FsDirectory::open(&dst_dir);
        let result = check_segment(&dir, &commit);
        assert!(
            result.all_passed(),
            "unexpected failures: {:?}",
            result.failures()
        );
        std::fs::remove_dir_all(&dst_dir).ok();

        let bad_dir = tempdir();
        let commit = write_sorted_dv_fixture(&bad_dir, &[1, 1, 1], None, true, 1);
        let dir = FsDirectory::open(&bad_dir);
        let result = check_segment(&dir, &commit);
        let failure = result
            .checks
            .iter()
            .find(|c| c.name == "soft_deletes.count_matches")
            .unwrap();
        assert!(!failure.passed());
        assert!(failure.message.contains("actual soft deletes: 3"));
        std::fs::remove_dir_all(&bad_dir).ok();
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
        assert!(!failure.passed());
        assert!(failure.message.contains("del_count"));

        // Every other check on this segment (files, .fnm, stored fields)
        // must still have run and passed -- one wrong field must not
        // suppress unrelated checks.
        assert!(result
            .checks
            .iter()
            .filter(|c| c.name != "liv.cardinality_matches_del_count")
            .all(|c| c.passed()));
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
        assert!(!failure.passed());

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
            ..Default::default()
        };

        let result = check_segment(&dir, &commit);
        assert!(!result.all_passed());
        let file_failures: Vec<_> = result
            .checks
            .iter()
            .filter(|c| c.name.starts_with("file:") && !c.passed())
            .collect();
        assert_eq!(file_failures.len(), 2);

        std::fs::remove_dir_all(&dst_dir).ok();
    }

    /// `.si` itself failing to parse must short-circuit every other check
    /// (nothing else can be trusted without a valid file list) -- and must
    /// **say so**, family by family, rather than returning one failure and
    /// sixteen silences. `all_passed()` is false either way here, but a
    /// caller counting failures would otherwise see one problem where
    /// sixteen classes of invariant went unchecked.
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
            ..Default::default()
        };

        let result = check_segment(&dir, &commit);
        assert_eq!(result.checks[0].name, "si.open");
        assert_eq!(result.checks[0].outcome, Outcome::Failed);
        // Exactly one failure; everything else is named as *not run*.
        assert_eq!(
            result
                .checks
                .iter()
                .filter(|c| c.outcome == Outcome::Failed)
                .count(),
            1
        );
        assert_eq!(result.skipped().len(), FAMILIES_BELOW_SI.len());
        assert!(result
            .skipped()
            .iter()
            .all(|c| c.message.contains("not run: si.open failed")));
        assert!(!result.all_passed());
        // The families a caller most needs to know went unchecked.
        let skipped: Vec<&str> = result.skipped().iter().map(|c| c.name.as_str()).collect();
        for family in ["postings.*", "doc_values.*", "norms.*", "term_vectors.*"] {
            assert!(skipped.contains(&family), "{skipped:?}");
        }

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
        assert!(!failure.passed());
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
        assert!(!failure.passed());
        assert!(failure.message.contains("no field claims"));
    }

    /// `CheckResult::failures()` must return exactly the failed checks, in
    /// order, for a mixed pass/fail result.
    #[test]
    fn check_result_failures_filters_correctly() {
        let result = CheckResult {
            segment_name: "_0".to_string(),
            max_doc: Some(3),
            checks: vec![Check::pass("a"), Check::fail("b", "bad"), Check::pass("c")],
            stats: CheckStats::default(),
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
        let commit = SegmentCommitInfo {
            segment_name: "_0".to_string(),
            segment_id: si.id,
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
        let mut checks = Vec::new();
        check_files_exist_and_validate(&dir, &commit, &si, &mut checks);
        // [0] is the `.si` self-reference check (this hand-built `.si` does
        // not list itself), [1] is the `_0.junk` footer failure.
        assert_eq!(checks.len(), 2);
        assert_eq!(checks[0].name, "si.files_lists_itself");
        assert!(!checks[0].passed());
        assert_eq!(checks[1].name, "file:_0.junk");
        assert!(!checks[1].passed());

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
            ..Default::default()
        };

        let mut checks = Vec::new();
        check_live_docs(&dir, &commit, &si, &mut checks);
        assert_eq!(checks[0].name, "liv.open");
        assert_eq!(checks[0].outcome, Outcome::Failed);
        // ... and the two checks that needed the file are reported as not
        // run, rather than vanishing from the result (c25).
        assert_eq!(
            checks[1..]
                .iter()
                .map(|c| (c.name.as_str(), c.outcome))
                .collect::<Vec<_>>(),
            [
                ("liv.max_doc_matches_si", Outcome::Skipped),
                ("liv.cardinality_matches_del_count", Outcome::Skipped),
            ]
        );

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
            ..Default::default()
        };

        let mut checks = Vec::new();
        check_live_docs(&dir, &commit, &si, &mut checks);
        let size_check = checks
            .iter()
            .find(|c| c.name == "liv.max_doc_matches_si")
            .expect("size check must have run");
        assert!(!size_check.passed());
        let open_check = checks
            .iter()
            .find(|c| c.name == "liv.open")
            .expect("liv.open must have run");
        assert!(!open_check.passed());

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
            ..Default::default()
        };

        let mut checks = Vec::new();
        check_stored_fields_doc_count(&dir, &commit, &si, &mut checks);
        assert_eq!(checks.len(), 1);
        assert!(!checks[0].passed());
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
        // Two checks: the doc-count cross-check (which must fail) and the
        // decode-every-document pass (which must still succeed -- the .fdt
        // bytes are genuinely fine, only `.si` lies).
        assert_eq!(checks.len(), 2);
        assert_eq!(checks[0].name, "stored_fields.doc_count_matches_si");
        assert!(!checks[0].passed());
        assert!(checks[0].message.contains("999"));
        assert_eq!(checks[1].name, "stored_fields.every_doc_decodes");
        assert!(checks[1].passed());
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
                // Real Lucene's `.si` always lists itself (the format's
                // `write` does `si.addFile(fileName)`), so a hand-built
                // fixture must too or `si.files_lists_itself` fails.
                "_0.si".to_string(),
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
            ..Default::default()
        }
    }

    /// A hand-built one-field segment with **positions, offsets and
    /// payloads**, big enough that at least one term's position stream fills
    /// a whole 128-value `for_util` block (the packed path, as opposed to
    /// the vint tail). The positional negative controls need that: a
    /// corrupted vint tail can only ever make positions *larger*, because
    /// vint deltas are non-negative and positions accumulate, whereas a
    /// corrupted packed block's bits-per-value header makes the decoder read
    /// arbitrary 32-bit values -- which is exactly the "position out of
    /// bounds / before the previous one / past MAX_POSITION" family Java's
    /// `checkFields` rejects.
    fn write_positional_postings_fixture(
        dst_dir: &std::path::Path,
        occurrences_in_first_doc: usize,
    ) -> segment_infos::SegmentCommitInfo {
        use lucene_codecs::field_infos::IndexOptions;
        use lucene_codecs::postings_writer::{
            write_single_field, FieldPostingsInput, TermPostings,
        };

        const OPTS: IndexOptions = IndexOptions::DocsAndFreqsAndPositionsAndOffsets;
        let big: Vec<i32> = (0..occurrences_in_first_doc as i32)
            .map(|i| i * 3)
            .collect();
        let big_offsets: Vec<(i32, i32)> = big.iter().map(|&p| (p * 4, p * 4 + 3)).collect();
        let big_payloads: Vec<Vec<u8>> = big.iter().map(|&p| vec![(p % 251) as u8; 2]).collect();
        let terms = vec![
            TermPostings {
                term: b"alpha".to_vec(),
                docs: vec![(0, occurrences_in_first_doc as i32), (2, 2)],
                positions: vec![big, vec![1, 7]],
                offsets: vec![big_offsets, vec![(4, 7), (28, 31)]],
                payloads: vec![big_payloads, vec![vec![9], vec![9]]],
            },
            TermPostings {
                term: b"beta".to_vec(),
                docs: vec![(1, 3)],
                positions: vec![vec![0, 5, 9]],
                offsets: vec![vec![(0, 4), (20, 24), (36, 40)]],
                payloads: vec![vec![vec![1], vec![2], vec![3]]],
            },
        ];
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: OPTS,
            doc_count: 3,
            has_payloads: true,
            terms: &terms,
        };
        let output = write_single_field(&input, &POSTINGS_SEG_ID, POSTINGS_SUFFIX)
            .expect("hand-built positional postings must write cleanly");

        let mut fi = postings_field_info(OPTS);
        fi.store_payloads = true;
        let fields = field_infos::write(&[fi], &POSTINGS_SEG_ID, "");
        std::fs::write(dst_dir.join("_0.fnm"), &fields).unwrap();
        for (ext, bytes) in [
            ("tim", &output.tim),
            ("tip", &output.tip),
            ("tmd", &output.tmd),
            ("doc", &output.doc),
            ("pos", &output.pos),
            ("pay", &output.pay),
        ] {
            std::fs::write(
                dst_dir.join(format!("_0_{POSTINGS_SUFFIX}.{ext}")),
                bytes.as_slice(),
            )
            .unwrap();
        }

        let si = SegmentInfo {
            id: POSTINGS_SEG_ID,
            version: segment_info::LuceneVersion {
                major: 10,
                minor: 0,
                bugfix: 0,
            },
            min_version: None,
            doc_count: 3,
            is_compound_file: false,
            has_blocks: false,
            diagnostics: vec![],
            files: ["_0.si".to_string(), "_0.fnm".to_string()]
                .into_iter()
                .chain(
                    ["tim", "tip", "tmd", "doc", "pos", "pay"]
                        .iter()
                        .map(|e| format!("_0_{POSTINGS_SUFFIX}.{e}")),
                )
                .collect(),
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
            ..Default::default()
        }
    }

    /// The hand-built positional fixture must pass every check cleanly --
    /// the "no false positives" side of the positional negative controls
    /// below.
    #[test]
    fn hand_built_positional_postings_pass_every_check() {
        let dst = tempdir();
        let commit = write_positional_postings_fixture(&dst, 300);
        let dir = FsDirectory::open(&dst);
        let result = check_segment(&dir, &commit);
        assert!(
            result.all_passed(),
            "unexpected failures: {:?}",
            result.failures()
        );
        let names: Vec<&str> = result.checks.iter().map(|c| c.name.as_str()).collect();
        assert!(names.contains(&"postings.positions_valid:body"));
        assert!(names.contains(&"postings.offsets_valid:body"));
        // 305 occurrences across 3 docs, all live.
        assert_eq!(result.stats.token_count, 305);
        assert_eq!(result.stats.term_doc_pairs, 3);
        assert_eq!(result.stats.term_count, 2);
    }

    /// Negative control for `postings.positions_valid` and
    /// `postings.offsets_valid`, driven directly at the predicate: a writer
    /// that emits a negative position, a decreasing position, a position
    /// past `IndexWriter.MAX_POSITION`, a negative or decreasing
    /// `startOffset`, or an `endOffset` before its `startOffset` must be
    /// reported -- and clean occurrences must not be. See
    /// [`check_occurrences`]' own doc comment for why this is driven at the
    /// predicate rather than by corrupting bytes.
    #[test]
    fn bad_positions_and_offsets_are_reported_by_the_predicate() {
        use lucene_codecs::postings::Position;
        let occ = |position, start_offset, end_offset| Position {
            position,
            start_offset,
            end_offset,
            payload: Vec::new(),
        };

        // Clean input: no complaint from either family.
        let mut pos = Vec::new();
        let mut off = Vec::new();
        check_occurrences(
            "body",
            b"t",
            0,
            &[occ(0, 0, 4), occ(5, 10, 14), occ(5, 10, 12)],
            true,
            &mut pos,
            &mut off,
        );
        assert!(pos.is_empty() && off.is_empty(), "{pos:?} {off:?}");

        // Each of Java's six rejected shapes, checked against the family
        // that is supposed to name it. (A bad position also perturbs the
        // offset sequence and vice versa, so the assertion is "the right
        // family complains", not "only that family does".)
        for (bad, wants_position, wants_offset) in [
            (occ(-1, 9, 11), true, false),
            (occ(MAX_POSITION + 1, 9, 11), true, false),
            (occ(3, 9, 11), true, false), // < the 7 planted before it
            (occ(9, -1, 1), false, true),
            (occ(9, 9, -1), false, true),
            (occ(9, 9, 3), false, true),
            (occ(9, 1, 5), false, true), // startOffset before the previous one
        ] {
            let mut pos = Vec::new();
            let mut off = Vec::new();
            check_occurrences(
                "body",
                b"t",
                7,
                &[occ(7, 8, 12), bad.clone()],
                true,
                &mut pos,
                &mut off,
            );
            if wants_position {
                assert!(!pos.is_empty(), "position family missed {bad:?}");
            }
            if wants_offset {
                assert!(!off.is_empty(), "offset family missed {bad:?}");
            }
        }
    }

    /// The complement of the predicate test: no single-byte corruption of a
    /// real packed `.pos`/`.pay` may be silently accepted. It is caught one
    /// layer lower than `postings.positions_valid` -- the `for_util`/
    /// `SliceInput` decoders reject a bad bits-per-value or block length
    /// before a position is ever produced, so the failure surfaces as
    /// `postings.terms_decode` (a check this batch added) or as the file's
    /// own CRC. That is the property worth pinning: nothing gets through.
    #[test]
    fn no_single_byte_corruption_of_pos_or_pay_is_silently_accepted() {
        let dst = tempdir();
        let commit = write_positional_postings_fixture(&dst, 300);
        let pos_path = dst.join(format!("_0_{POSTINGS_SUFFIX}.pos"));
        let pay_path = dst.join(format!("_0_{POSTINGS_SUFFIX}.pay"));
        let pos_original = std::fs::read(&pos_path).unwrap();
        let pay_original = std::fs::read(&pay_path).unwrap();

        for (path, original) in [(&pos_path, &pos_original), (&pay_path, &pay_original)] {
            for off in 48..(original.len() - lucene_store::codec_util::FOOTER_LENGTH) {
                for mask in [0x01u8, 0x40, 0x80, 0xff] {
                    let mut bytes = original.clone();
                    bytes[off] ^= mask;
                    std::fs::write(path, &bytes).unwrap();
                    let dir = FsDirectory::open(&dst);
                    let result = check_segment(&dir, &commit);
                    assert!(
                        !result.all_passed(),
                        "flipping {path:?} byte {off} with {mask:#x} was silently accepted"
                    );
                }
                std::fs::write(path, original).unwrap();
            }
        }
        std::fs::write(&pos_path, &pos_original).unwrap();
        std::fs::write(&pay_path, &pay_original).unwrap();
        let dir = FsDirectory::open(&dst);
        assert!(check_segment(&dir, &commit).all_passed());
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
        let result = &results[1];
        assert!(
            result.all_passed(),
            "unexpected failures: {:?}",
            result.failures()
        );
        assert!(
            result
                .checks
                .iter()
                .any(|c| c.name.starts_with("postings.total_term_freq:") && c.passed()),
            "expected a passing postings.total_term_freq:<field> check, got: {:?}",
            result.checks
        );
        assert!(
            result
                .checks
                .iter()
                .any(|c| c.name.starts_with("postings.doc_ids_valid:") && c.passed()),
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
        assert!(!freq_check.passed());
        assert!(freq_check.message.contains("totalTermFreq=60"));
        assert!(freq_check.message.contains("sum to 6"));

        // An unrelated dimension (doc ID validity) must still pass -- one
        // wrong stat must not suppress or corrupt an unrelated check.
        let doc_ids_check = result
            .checks
            .iter()
            .find(|c| c.name == "postings.doc_ids_valid:body")
            .expect("doc_ids_valid check must have run");
        assert!(doc_ids_check.passed());

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
        assert!(!doc_ids_check.passed());
        assert!(doc_ids_check.message.contains("doc ID 9"));

        // total_term_freq is an unrelated dimension here (both sides sum
        // to 6) -- must still pass, proving the two checks are
        // independent.
        let freq_check = result
            .checks
            .iter()
            .find(|c| c.name == "postings.total_term_freq:body")
            .expect("total_term_freq check must have run");
        assert!(freq_check.passed());

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
        assert!(!doc_open_check.passed());

        std::fs::remove_dir_all(&dst_dir).ok();
    }

    // -- points-tree structural invariants --

    /// The real `points_index` fixture (genuine Java-written BKD data with
    /// a single-dimension field, a 2-dimension field, and a 4-dim/2-index-
    /// dim "shape"-style field -- see `GenPoints.java`) must pass every
    /// structural-invariant check cleanly, exercising both the plain
    /// field-bounds check (all three fields) and the leaf-bounds-subset
    /// check (the two multi-index-dim fields only) on real data -- the
    /// "no false positives" baseline for this check, mirroring
    /// `valid_blocktree_fixture_passes_postings_re_derivation`'s role for
    /// postings.
    #[test]
    fn valid_points_fixture_passes_structural_invariants() {
        let dir = FsDirectory::open(fixture_dir("points_index"));
        let results = check_directory(&dir).expect("read segments_N");
        assert_eq!(results.len(), 2);
        let result = &results[1];
        assert!(
            result.all_passed(),
            "unexpected failures: {:?}",
            result.failures()
        );
        assert!(
            result
                .checks
                .iter()
                .any(|c| c.name.starts_with("points.value_within_field_bounds:") && c.passed()),
            "expected a passing points.value_within_field_bounds:<field> check, got: {:?}",
            result.checks
        );
        assert!(
            result
                .checks
                .iter()
                .any(|c| c.name.starts_with("points.leaf_bounds_subset_of_field:") && c.passed()),
            "expected a passing points.leaf_bounds_subset_of_field:<field> check \
             (the fixture has multi-index-dim fields), got: {:?}",
            result.checks
        );
        assert!(
            result
                .checks
                .iter()
                .any(|c| c.name.starts_with("points.point_count_matches:") && c.passed()),
            "expected a passing points.point_count_matches:<field> check, got: {:?}",
            result.checks
        );
    }

    const POINTS_SEG_ID: [u8; ID_LENGTH] = [13u8; ID_LENGTH];

    fn points_field_info() -> field_infos::FieldInfo {
        field_infos::FieldInfo {
            name: "loc".to_string(),
            number: 5,
            store_term_vectors: false,
            omit_norms: false,
            store_payloads: false,
            soft_deletes_field: false,
            parent_field: false,
            index_options: field_infos::IndexOptions::None,
            doc_values_type: field_infos::DocValuesType::None,
            doc_values_skip_index_type: field_infos::DocValuesSkipIndexType::None,
            doc_values_gen: -1,
            attributes: vec![],
            point_dimension_count: 1,
            point_index_dimension_count: 1,
            point_num_bytes: 8,
            vector_dimension: 0,
            vector_encoding: field_infos::VectorEncoding::Float32,
            vector_similarity_function: field_infos::VectorSimilarityFunction::Euclidean,
        }
    }

    /// Writes a minimal, self-contained, non-compound one-field segment
    /// (`.si`/`.fnm`/`.kdm`/`.kdi`/`.kdd`) from
    /// [`lucene_codecs::points::write`]'s output for a single-dimension,
    /// 8-byte-per-value field with the given `(docID, packedValue)` pairs,
    /// and returns the `SegmentCommitInfo` to open it with. `kdd_override`
    /// lets a test substitute a *different* `.kdd` buffer than the one that
    /// naturally matches `points` -- the same "swap the on-disk bytes for a
    /// legitimately-decodable but wrong one" mechanism
    /// `write_postings_fixture` uses for its own corruption tests, letting
    /// this test hand-corrupt a decoded point value without any raw-offset
    /// byte surgery.
    fn write_points_fixture(
        dst_dir: &std::path::Path,
        points: &[(i32, Vec<u8>)],
        max_doc: i32,
        kdd_override: Option<&[u8]>,
    ) -> segment_infos::SegmentCommitInfo {
        write_points_fixture_dims(dst_dir, points, max_doc, 1, 8, kdd_override, &|_| {})
    }

    /// [`write_points_fixture`] with the dimension count and a `.kdm` editor
    /// exposed.
    ///
    /// c25 recorded that `points.leaf_bounds_subset_of_field` is *skipped
    /// entirely* for a single-dimension field and that every hand-built
    /// points fixture in this file is one, so the check had never run against
    /// anything. `kdm_edit` is the same idea as `kdd_override` applied to the
    /// metadata: the field summary a `.kdm` records (`pointCount`,
    /// `docCount`) is computed by the writer from the points it is given, so
    /// it cannot be *asked* for a summary that lies -- which is precisely
    /// what `points.doc_count_matches` exists to reject.
    fn write_points_fixture_dims(
        dst_dir: &std::path::Path,
        points: &[(i32, Vec<u8>)],
        max_doc: i32,
        num_index_dims: i32,
        bytes_per_dim: i32,
        kdd_override: Option<&[u8]>,
        kdm_edit: &dyn Fn(&mut Vec<u8>),
    ) -> segment_infos::SegmentCommitInfo {
        use lucene_codecs::points::WritePointsField;

        let field = WritePointsField {
            field_number: 5,
            num_dims: num_index_dims,
            num_index_dims,
            bytes_per_dim,
            points: points.to_vec(),
        };
        let (mut kdm, kdi, kdd) = points::write(
            &[field],
            points::DEFAULT_MAX_POINTS_IN_LEAF_NODE,
            &POINTS_SEG_ID,
            "",
        )
        .expect("hand-built points must write cleanly");
        kdm_edit(&mut kdm);
        let kdd = kdd_override.unwrap_or(&kdd);

        let mut info = points_field_info();
        info.point_dimension_count = num_index_dims;
        info.point_index_dimension_count = num_index_dims;
        info.point_num_bytes = bytes_per_dim;
        let fields = field_infos::write(&[info], &POINTS_SEG_ID, "");
        std::fs::write(dst_dir.join("_0.fnm"), &fields).unwrap();
        std::fs::write(dst_dir.join("_0.kdm"), &kdm).unwrap();
        std::fs::write(dst_dir.join("_0.kdi"), &kdi).unwrap();
        std::fs::write(dst_dir.join("_0.kdd"), kdd).unwrap();

        let si = SegmentInfo {
            id: POINTS_SEG_ID,
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
                "_0.si".to_string(),
                "_0.fnm".to_string(),
                "_0.kdm".to_string(),
                "_0.kdi".to_string(),
                "_0.kdd".to_string(),
            ],
            attributes: vec![],
            index_sort: None,
        };
        std::fs::write(dst_dir.join("_0.si"), segment_info::write(&si, "")).unwrap();

        segment_infos::SegmentCommitInfo {
            segment_name: "_0".to_string(),
            segment_id: POINTS_SEG_ID,
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

    /// A hand-built, genuinely self-consistent points segment (real writer
    /// output, not raw-byte surgery) must pass the structural-invariant
    /// checks -- proves the machinery works on this port's own writer
    /// output too, not just the one real-Lucene fixture above.
    #[test]
    fn hand_built_consistent_points_pass_structural_invariants() {
        let points = vec![
            (0, vec![0, 0, 0, 0, 0, 0, 0, 1]),
            (1, vec![0, 0, 0, 0, 0, 0, 0, 2]),
            (2, vec![0, 0, 0, 0, 0, 0, 0, 3]),
        ];
        let dst_dir = tempdir();
        let commit = write_points_fixture(&dst_dir, &points, 3, None);
        let dir = FsDirectory::open(&dst_dir);

        let result = check_segment(&dir, &commit);
        assert!(
            result.all_passed(),
            "unexpected failures: {:?}",
            result.failures()
        );

        std::fs::remove_dir_all(&dst_dir).ok();
    }

    /// The actual proof this check does something real: build a segment
    /// whose `.kdm` field-level bounds are derived from the *real* points
    /// `(0,...,1)`/`(1,...,2)`/`(2,...,3)` (so `min_packed_value`/
    /// `max_packed_value` = `...01`/`...03`), but swap the `.kdd` bytes for
    /// ones decoded from a *different* point set where the middle point's
    /// value is `...FF` -- decoding still succeeds cleanly (same doc IDs,
    /// same doc count, same leaf layout), so this is a genuine "declared
    /// bound vs. actual decoded value" disagreement, not a truncated/corrupt
    /// file. `points.value_within_field_bounds:loc` must fail and name the
    /// offending doc, while `points.point_count_matches:loc` (an unrelated
    /// dimension: the leaf still decodes exactly 3 points either way) must
    /// still pass -- proving the two checks are independent, mirroring
    /// `corrupted_total_term_freq_is_caught_by_re_derivation`'s structure
    /// for postings.
    #[test]
    fn corrupted_point_value_is_caught_by_bounds_check() {
        let real_points = vec![
            (0, vec![0, 0, 0, 0, 0, 0, 0, 1]),
            (1, vec![0, 0, 0, 0, 0, 0, 0, 2]),
            (2, vec![0, 0, 0, 0, 0, 0, 0, 3]),
        ];
        let corrupted_points = vec![
            (0, vec![0, 0, 0, 0, 0, 0, 0, 1]),
            (1, vec![0xFF, 0, 0, 0, 0, 0, 0, 0xFF]), // wildly out of [..01, ..03]
            (2, vec![0, 0, 0, 0, 0, 0, 0, 3]),
        ];
        let corrupted_field = lucene_codecs::points::WritePointsField {
            field_number: 5,
            num_dims: 1,
            num_index_dims: 1,
            bytes_per_dim: 8,
            points: corrupted_points,
        };
        let (_, _, corrupted_kdd) = points::write(
            &[corrupted_field],
            points::DEFAULT_MAX_POINTS_IN_LEAF_NODE,
            &POINTS_SEG_ID,
            "",
        )
        .unwrap();

        let dst_dir = tempdir();
        let commit = write_points_fixture(&dst_dir, &real_points, 3, Some(&corrupted_kdd));
        let dir = FsDirectory::open(&dst_dir);

        let result = check_segment(&dir, &commit);
        assert!(!result.all_passed());

        let bounds_check = result
            .checks
            .iter()
            .find(|c| c.name == "points.value_within_field_bounds:loc")
            .expect("value_within_field_bounds check must have run");
        assert!(!bounds_check.passed());
        assert!(bounds_check.message.contains("doc 1"));

        // Unrelated dimension: the corrupted leaf still decodes exactly 3
        // points, same as the declared point_count, so this must still
        // pass -- one wrong value must not suppress or corrupt an unrelated
        // check.
        let count_check = result
            .checks
            .iter()
            .find(|c| c.name == "points.point_count_matches:loc")
            .expect("point_count_matches check must have run");
        assert!(count_check.passed());

        std::fs::remove_dir_all(&dst_dir).ok();
    }

    /// A field claiming points in `.fnm` whose segment is missing one of
    /// `.kdm`/`.kdi`/`.kdd` must be flagged as `points.open`, not panic --
    /// the points analogue of
    /// `missing_doc_file_for_multi_doc_term_fails_doc_open_check`.
    #[test]
    fn missing_points_file_fails_points_open_check() {
        let points = vec![(0, vec![0, 0, 0, 0, 0, 0, 0, 1])];
        let dst_dir = tempdir();
        let commit = write_points_fixture(&dst_dir, &points, 1, None);

        // Make the .kdd file genuinely absent, not just unlisted.
        let dir_ro = FsDirectory::open(&dst_dir);
        let mut si = open_si(&dir_ro, &commit).expect("hand-built .si parses");
        si.files.retain(|f| !f.ends_with(".kdd"));
        std::fs::write(dst_dir.join("_0.si"), segment_info::write(&si, "")).unwrap();
        std::fs::remove_file(dst_dir.join("_0.kdd")).unwrap();

        let dir = FsDirectory::open(&dst_dir);
        let result = check_segment(&dir, &commit);
        let points_open_check = result
            .checks
            .iter()
            .find(|c| c.name == "points.open")
            .expect("points.open check must have run");
        assert!(!points_open_check.passed());

        std::fs::remove_dir_all(&dst_dir).ok();
    }

    /// A segment where no field claims points at all must skip the points
    /// checks entirely (not fail, not run vacuously) -- exercises the
    /// early-return branch for a segment this check has nothing to verify
    /// about, using the real `blocktree_index` fixture (postings only, no
    /// points).
    #[test]
    fn segment_without_points_fields_skips_points_checks() {
        let dir = FsDirectory::open(fixture_dir("blocktree_index"));
        let results = check_directory(&dir).expect("read segments_N");
        assert!(!results[1]
            .checks
            .iter()
            .any(|c| c.name.starts_with("points.")));
    }

    /// A compound (`.cfs`/`.cfe`) segment must skip the points check
    /// entirely, matching [`check_postings_term_stats`]'s own compound-file
    /// scope -- this module has no compound-file support anywhere.
    #[test]
    fn compound_segment_skips_points_checks() {
        let field = points_field_info();
        let fields = FieldInfos {
            fields: vec![field],
        };
        let si = SegmentInfo {
            id: POINTS_SEG_ID,
            version: segment_info::LuceneVersion {
                major: 10,
                minor: 0,
                bugfix: 0,
            },
            min_version: None,
            doc_count: 1,
            is_compound_file: true,
            has_blocks: false,
            diagnostics: vec![],
            files: vec![],
            attributes: vec![],
            index_sort: None,
        };
        let commit = segment_infos::SegmentCommitInfo {
            segment_name: "_0".to_string(),
            segment_id: POINTS_SEG_ID,
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
        let dst_dir = tempdir();
        let dir = FsDirectory::open(&dst_dir);
        let mut checks = Vec::new();
        check_points_structural_invariants(&dir, &commit, &si, &fields, &mut checks);
        assert!(checks.is_empty());
        std::fs::remove_dir_all(&dst_dir).ok();
    }

    /// A field whose `.fnm` entry claims points but whose field *number*
    /// doesn't match any field actually recorded in `.kdm` must be flagged
    /// as `points.field_present`, not panic or silently skip -- exercises
    /// the "claimed but not actually present in the BKD tree" branch
    /// distinctly from a missing-file `points.open` failure.
    #[test]
    fn mismatched_field_number_fails_field_present_check() {
        let points = vec![(0, vec![0, 0, 0, 0, 0, 0, 0, 1])];
        let dst_dir = tempdir();
        // write_points_fixture always writes the .kdm field under number 5;
        // overwrite .fnm afterwards with a field claiming a *different*
        // number (99) so `reader.field(99)` finds nothing.
        let commit = write_points_fixture(&dst_dir, &points, 1, None);
        let mut mismatched_field = points_field_info();
        mismatched_field.number = 99;
        let fields = field_infos::write(&[mismatched_field], &POINTS_SEG_ID, "");
        std::fs::write(dst_dir.join("_0.fnm"), &fields).unwrap();

        let dir = FsDirectory::open(&dst_dir);
        let result = check_segment(&dir, &commit);
        let check = result
            .checks
            .iter()
            .find(|c| c.name == "points.field_present:loc")
            .expect("field_present check must have run");
        assert!(!check.passed());

        std::fs::remove_dir_all(&dst_dir).ok();
    }

    /// The leaf's own decoded point count must sum to `.kdm`'s declared
    /// `point_count` for the field: build a segment whose `.kdm`/`.kdi`
    /// claim 3 points (from `real_points`) but swap in `.kdd` bytes
    /// re-encoded from only 2 points -- decoding still succeeds (same
    /// field shape, one leaf), but the leaf actually yields 2 points, not
    /// 3, so `points.point_count_matches:loc` must fail while
    /// `points.value_within_field_bounds:loc` (checked against the
    /// looser, real-points-derived field bounds) still passes.
    /// Negative control for `points.doc_count_matches`: a `.kdd` whose
    /// points carry fewer *distinct* doc IDs than `.kdm`'s declared
    /// `docCount`, with the point count and every packed value left intact
    /// -- so only Java's `getDocCountSeen() != docCount` check can catch it.
    #[test]
    fn fewer_distinct_docs_than_declared_fails_doc_count_check() {
        let real_points = vec![
            (0, vec![0, 0, 0, 0, 0, 0, 0, 1]),
            (1, vec![0, 0, 0, 0, 0, 0, 0, 2]),
            (2, vec![0, 0, 0, 0, 0, 0, 0, 3]),
        ];
        // Same three packed values, all on doc 0: point_count is unchanged
        // and every value is still inside the field's declared bounds.
        let same_doc_points = vec![
            (0, vec![0, 0, 0, 0, 0, 0, 0, 1]),
            (0, vec![0, 0, 0, 0, 0, 0, 0, 2]),
            (0, vec![0, 0, 0, 0, 0, 0, 0, 3]),
        ];
        let same_doc_field = lucene_codecs::points::WritePointsField {
            field_number: 5,
            num_dims: 1,
            num_index_dims: 1,
            bytes_per_dim: 8,
            points: same_doc_points,
        };
        let (_, _, same_doc_kdd) = points::write(
            &[same_doc_field],
            points::DEFAULT_MAX_POINTS_IN_LEAF_NODE,
            &POINTS_SEG_ID,
            "",
        )
        .unwrap();

        let dst_dir = tempdir();
        let commit = write_points_fixture(&dst_dir, &real_points, 3, Some(&same_doc_kdd));
        let dir = FsDirectory::open(&dst_dir);
        let result = check_segment(&dir, &commit);
        let failed: Vec<&str> = result.failures().iter().map(|c| c.name.as_str()).collect();
        assert!(
            failed.contains(&"points.doc_count_matches:loc"),
            "expected the doc-count check to fail, got {failed:?}"
        );
        assert!(
            !failed.contains(&"points.point_count_matches:loc"),
            "the point count is unchanged, so only the doc count may fail: {failed:?}"
        );
    }

    #[test]
    fn fewer_actual_points_than_declared_fails_point_count_check() {
        let real_points = vec![
            (0, vec![0, 0, 0, 0, 0, 0, 0, 1]),
            (1, vec![0, 0, 0, 0, 0, 0, 0, 2]),
            (2, vec![0, 0, 0, 0, 0, 0, 0, 3]),
        ];
        let fewer_points = vec![
            (0, vec![0, 0, 0, 0, 0, 0, 0, 1]),
            (1, vec![0, 0, 0, 0, 0, 0, 0, 2]),
        ];
        let fewer_field = lucene_codecs::points::WritePointsField {
            field_number: 5,
            num_dims: 1,
            num_index_dims: 1,
            bytes_per_dim: 8,
            points: fewer_points,
        };
        let (_, _, fewer_kdd) = points::write(
            &[fewer_field],
            points::DEFAULT_MAX_POINTS_IN_LEAF_NODE,
            &POINTS_SEG_ID,
            "",
        )
        .unwrap();

        let dst_dir = tempdir();
        let commit = write_points_fixture(&dst_dir, &real_points, 3, Some(&fewer_kdd));
        let dir = FsDirectory::open(&dst_dir);

        let result = check_segment(&dir, &commit);
        assert!(!result.all_passed());

        let count_check = result
            .checks
            .iter()
            .find(|c| c.name == "points.point_count_matches:loc")
            .expect("point_count_matches check must have run");
        assert!(!count_check.passed());
        assert!(count_check.message.contains("point_count=3"));
        assert!(count_check.message.contains("decoded 2"));

        std::fs::remove_dir_all(&dst_dir).ok();
    }

    // ----------------------------------------------------------------------
    // Hand-built structures for the checks whose *failure* arms a real
    // segment never reaches.
    //
    // c19's coverage audit found that `check_index.rs`'s uncovered lines were
    // almost entirely `problems.push(..)`/`Check::fail(..)` arms: 4 000 lines
    // of verifier in which ~150 individual checks had never once been seen to
    // fire. That is the failure mode this module exists to prevent -- a check
    // that silently does nothing looks exactly like a check that passes.
    //
    // The two functions below take plain data structures rather than files,
    // so a hand-built input reaches every arm directly; that is the same
    // "test-only builder" pattern the `test-coverage` skill sanctions, and it
    // is far cheaper than synthesizing a `.tvd`/`.dvs` per arm.
    // ----------------------------------------------------------------------

    fn vector_term(term: &[u8], freq: i32) -> term_vectors::TermVectorTerm {
        term_vectors::TermVectorTerm {
            term: term.to_vec(),
            freq,
            positions: None,
            start_offsets: None,
            end_offsets: None,
            payloads: None,
        }
    }

    /// Every self-consistency arm of `check_one_vector_field` -- Java's
    /// `checkFields(..., isVectors = true)` block -- fires on a hand-built
    /// term vector that breaks exactly that invariant, and *only* that one.
    #[test]
    fn every_term_vector_self_consistency_arm_reports_its_own_invariant() {
        // `postings = None`, so only the self-consistency half runs.
        let check = |field: &term_vectors::TermVectorField| -> Vec<String> {
            let mut memo: TermPostingsMemo = std::collections::HashMap::new();
            let mut memo_elements = 0usize;
            let mut self_problems = Vec::new();
            let mut postings_problems = Vec::new();
            check_one_vector_field(
                7,
                field,
                None,
                None,
                &mut memo,
                &mut memo_elements,
                &mut self_problems,
                &mut postings_problems,
            );
            assert!(
                postings_problems.is_empty(),
                "no postings were supplied, so no cross-check may fire: {postings_problems:?}"
            );
            self_problems
        };
        let field = |terms: Vec<term_vectors::TermVectorTerm>,
                     positions: bool,
                     offsets: bool,
                     payloads: bool| {
            term_vectors::TermVectorField {
                field_number: 0,
                has_positions: positions,
                has_offsets: offsets,
                has_payloads: payloads,
                terms,
            }
        };

        // A clean vector: nothing fires, and the field name falls back to
        // `<unknown>` only when it does (`info = None` here).
        let mut clean = vector_term(b"aaa", 2);
        clean.positions = Some(vec![0, 4]);
        clean.start_offsets = Some(vec![0, 8]);
        clean.end_offsets = Some(vec![3, 11]);
        clean.payloads = Some(vec![vec![1], vec![2]]);
        assert!(check(&field(vec![clean.clone()], true, true, true)).is_empty());

        // 1. terms out of order (`bbb` then `aaa`), and 2. a duplicate term
        //    (`<=`, not `<`).
        let out_of_order = check(&field(
            vec![vector_term(b"bbb", 1), vector_term(b"aaa", 1)],
            false,
            false,
            false,
        ));
        assert_eq!(out_of_order.len(), 1, "{out_of_order:?}");
        assert!(out_of_order[0].contains("vector terms out of order"));
        let duplicate = check(&field(
            vec![vector_term(b"aaa", 1), vector_term(b"aaa", 1)],
            false,
            false,
            false,
        ));
        assert_eq!(duplicate.len(), 1, "{duplicate:?}");
        assert!(duplicate[0].contains("vector terms out of order"));

        // 3. freq <= 0.
        let zero_freq = check(&field(vec![vector_term(b"aaa", 0)], false, false, false));
        assert_eq!(zero_freq.len(), 1, "{zero_freq:?}");
        assert!(zero_freq[0].contains("freq 0 is out of bounds"));
        assert!(zero_freq[0].contains("<unknown>"), "{zero_freq:?}");

        // 4. positions present but their count disagrees with `freq`.
        let mut t = vector_term(b"aaa", 3);
        t.positions = Some(vec![0, 4]);
        let wrong_count = check(&field(vec![t], true, false, false));
        assert_eq!(wrong_count.len(), 1, "{wrong_count:?}");
        assert!(wrong_count[0].contains("freq=3 but 2 positions"));

        // 5. a negative position, and 6. a position before the previous one.
        let mut t = vector_term(b"aaa", 2);
        t.positions = Some(vec![-1, 4]);
        let negative = check(&field(vec![t], true, false, false));
        assert_eq!(negative.len(), 1, "{negative:?}");
        assert!(negative[0].contains("position -1 is out of bounds"));
        let mut t = vector_term(b"aaa", 3);
        t.positions = Some(vec![0, 9, 4]);
        let backwards = check(&field(vec![t], true, false, false));
        assert_eq!(backwards.len(), 1, "{backwards:?}");
        assert!(backwards[0].contains("before the previous position 9"));

        // 7. the field claims positions and the term carries none.
        let no_positions = check(&field(vec![vector_term(b"aaa", 1)], true, false, false));
        assert_eq!(no_positions.len(), 1, "{no_positions:?}");
        assert!(no_positions[0].contains("claims positions but"));

        // 8. offset counts disagreeing with `freq` and with each other.
        let mut t = vector_term(b"aaa", 2);
        t.start_offsets = Some(vec![0]);
        t.end_offsets = Some(vec![3, 9]);
        let offset_counts = check(&field(vec![t], false, true, false));
        assert_eq!(offset_counts.len(), 1, "{offset_counts:?}");
        assert!(offset_counts[0].contains("freq=2 but 1 start / 2 end offsets"));

        // 9. a negative start offset, and 10. an end before its start.
        let mut t = vector_term(b"aaa", 1);
        t.start_offsets = Some(vec![-4]);
        t.end_offsets = Some(vec![3]);
        let negative_offset = check(&field(vec![t], false, true, false));
        assert_eq!(negative_offset.len(), 1, "{negative_offset:?}");
        assert!(negative_offset[0].contains("offsets [-4, 3] are out of bounds"));
        let mut t = vector_term(b"aaa", 1);
        t.start_offsets = Some(vec![9]);
        t.end_offsets = Some(vec![3]);
        let inverted = check(&field(vec![t], false, true, false));
        assert_eq!(inverted.len(), 1, "{inverted:?}");
        assert!(inverted[0].contains("offsets [9, 3] are out of bounds"));

        // 11. the field claims offsets and the term carries none.
        let no_offsets = check(&field(vec![vector_term(b"aaa", 1)], false, true, false));
        assert_eq!(no_offsets.len(), 1, "{no_offsets:?}");
        assert!(no_offsets[0].contains("claims offsets but"));

        // 12. start offsets present, end offsets absent (and the mirror).
        let mut t = vector_term(b"aaa", 1);
        t.start_offsets = Some(vec![0]);
        let half = check(&field(vec![t], false, true, false));
        assert_eq!(half.len(), 1, "{half:?}");
        assert!(half[0].contains("start and end offsets disagree"));
        let mut t = vector_term(b"aaa", 1);
        t.end_offsets = Some(vec![3]);
        let other_half = check(&field(vec![t], false, true, false));
        assert_eq!(other_half.len(), 1, "{other_half:?}");
        assert!(other_half[0].contains("start and end offsets disagree"));

        // 13. payloads on a field whose header says it has none.
        let mut t = vector_term(b"aaa", 1);
        t.payloads = Some(vec![vec![7]]);
        let stray_payloads = check(&field(vec![t], false, false, false));
        assert_eq!(stray_payloads.len(), 1, "{stray_payloads:?}");
        assert!(stray_payloads[0].contains("hasPayloads=false"));
    }

    /// Every cross-check arm of `check_one_vector_field` -- Java's
    /// `testTermVectors` slow level, the block that requires the stored
    /// vector to agree with the inverted index term for term -- fires on a
    /// hand-built vector that disagrees with real, hand-written postings in
    /// exactly one way.
    ///
    /// Driven at the function rather than by corrupting a `.tvd`: this port
    /// has no term-vector *writer* that can be told to emit a wrong freq, and
    /// a byte flip in a real `.tvd` overwhelmingly fails the decode long
    /// before it produces a well-formed vector that merely disagrees.
    #[test]
    fn every_term_vector_postings_cross_check_arm_reports_its_own_disagreement() {
        let dst = tempdir();
        let commit = write_positional_postings_fixture(&dst, 4);
        let dir = FsDirectory::open(&dst);
        let si = open_si(&dir, &commit).unwrap();
        let fnm = open_fnm(&dir, &commit, &si).unwrap();
        let bytes = open_postings_bytes(&dir, &si).unwrap().unwrap();
        let handles = bytes.handles(&commit, &fnm, &si).unwrap();
        let info = fnm.fields.iter().find(|f| f.number == 0).unwrap().clone();

        // The fixture: term `alpha` occurs 4x in doc 0 at positions 0/3/6/9
        // with offsets (p*4, p*4+3) and 2-byte payloads, and 2x in doc 2.
        // Term `beta` occurs 3x in doc 1.
        let check = |field: &term_vectors::TermVectorField,
                     doc_id: i32,
                     info: Option<&field_infos::FieldInfo>| {
            let mut memo: TermPostingsMemo = std::collections::HashMap::new();
            let mut memo_elements = 0usize;
            let mut self_problems = Vec::new();
            let mut postings_problems = Vec::new();
            check_one_vector_field(
                doc_id,
                field,
                info,
                Some(&handles),
                &mut memo,
                &mut memo_elements,
                &mut self_problems,
                &mut postings_problems,
            );
            (self_problems, postings_problems)
        };
        let field = |terms: Vec<term_vectors::TermVectorTerm>| term_vectors::TermVectorField {
            field_number: 0,
            has_positions: true,
            has_offsets: true,
            has_payloads: true,
            terms,
        };
        let alpha_in_doc0 = || {
            let mut t = vector_term(b"alpha", 4);
            t.positions = Some(vec![0, 3, 6, 9]);
            t.start_offsets = Some(vec![0, 12, 24, 36]);
            t.end_offsets = Some(vec![3, 15, 27, 39]);
            t.payloads = Some(vec![vec![0, 0], vec![3, 3], vec![6, 6], vec![9, 9]]);
            t
        };

        // Baseline: a vector that matches the postings exactly is silent.
        let (self_p, cross) = check(&field(vec![alpha_in_doc0()]), 0, Some(&info));
        assert!(
            self_p.is_empty() && cross.is_empty(),
            "{self_p:?} {cross:?}"
        );

        // 1. a field the postings do not have at all, on an indexed field.
        let mut missing_field_info = info.clone();
        missing_field_info.name = "no_such_field".to_string();
        let (_, cross) = check(&field(vec![alpha_in_doc0()]), 0, Some(&missing_field_info));
        assert_eq!(cross.len(), 1, "{cross:?}");
        assert!(cross[0].contains("does not exist in the postings"));

        // ... and the same field with `IndexOptions::None`, where Java does
        // not complain either, because the field is not indexed.
        let mut unindexed = missing_field_info.clone();
        unindexed.index_options = field_infos::IndexOptions::None;
        let (_, cross) = check(&field(vec![alpha_in_doc0()]), 0, Some(&unindexed));
        assert!(cross.is_empty(), "{cross:?}");

        // 2. a term the postings do not carry.
        let mut t = vector_term(b"zzz", 1);
        t.positions = Some(vec![0]);
        t.start_offsets = Some(vec![0]);
        t.end_offsets = Some(vec![1]);
        t.payloads = Some(vec![vec![1, 1]]);
        let (_, cross) = check(&field(vec![t]), 0, Some(&info));
        assert_eq!(cross.len(), 1, "{cross:?}");
        assert!(cross[0].contains("does not exist in the postings"));

        // 3. a term the postings carry, but not for this document.
        let (_, cross) = check(&field(vec![alpha_in_doc0()]), 1, Some(&info));
        assert_eq!(cross.len(), 1, "{cross:?}");
        assert!(cross[0].contains("was not found in the postings for this doc"));

        // 4. a freq the postings disagree with.
        let mut t = alpha_in_doc0();
        t.freq = 3;
        t.positions = Some(vec![0, 3, 6]);
        t.start_offsets = Some(vec![0, 12, 24]);
        t.end_offsets = Some(vec![3, 15, 27]);
        t.payloads = Some(vec![vec![0, 0], vec![3, 3], vec![6, 6]]);
        let (_, cross) = check(&field(vec![t]), 0, Some(&info));
        assert_eq!(cross.len(), 4, "{cross:?}");
        assert!(cross[0].contains("vector freq=3 differs from postings freq=4"));
        assert!(cross[1].contains("vector positions"));
        assert!(cross[2].contains("vector offsets"));
        assert!(cross[3].contains("vector payloads differ"));

        // 5. positions only.
        let mut t = alpha_in_doc0();
        t.positions = Some(vec![0, 3, 6, 11]);
        let (_, cross) = check(&field(vec![t]), 0, Some(&info));
        assert_eq!(cross.len(), 1, "{cross:?}");
        assert!(cross[0].contains("vector positions"));

        // 6. offsets only.
        let mut t = alpha_in_doc0();
        t.end_offsets = Some(vec![3, 15, 27, 99]);
        let (_, cross) = check(&field(vec![t]), 0, Some(&info));
        assert_eq!(cross.len(), 1, "{cross:?}");
        assert!(cross[0].contains("vector offsets"));

        // 7. payloads only.
        let mut t = alpha_in_doc0();
        t.payloads = Some(vec![vec![0, 0], vec![3, 3], vec![6, 6], vec![9, 8]]);
        let (_, cross) = check(&field(vec![t]), 0, Some(&info));
        assert_eq!(cross.len(), 1, "{cross:?}");
        assert!(cross[0].contains("vector payloads differ"));

        // 8. the memo really is a memo: two documents' worth of the same term
        //    decode the postings once, and the cap resets it rather than
        //    growing without bound.
        let mut memo: TermPostingsMemo = std::collections::HashMap::new();
        let mut memo_elements = TERM_VECTOR_POSTINGS_MEMO_ELEMENTS;
        memo.insert((99, b"stale".to_vec()), (Default::default(), None));
        let mut self_problems = Vec::new();
        let mut postings_problems = Vec::new();
        check_one_vector_field(
            0,
            &field(vec![alpha_in_doc0()]),
            Some(&info),
            Some(&handles),
            &mut memo,
            &mut memo_elements,
            &mut self_problems,
            &mut postings_problems,
        );
        assert!(postings_problems.is_empty(), "{postings_problems:?}");
        assert!(
            !memo.contains_key(&(99, b"stale".to_vec())),
            "past the element cap the memo must be dropped, not grown"
        );
        assert!(memo_elements > 0 && memo_elements < TERM_VECTOR_POSTINGS_MEMO_ELEMENTS);

        std::fs::remove_dir_all(&dst).ok();
    }

    /// Every *global* arm of `check_doc_value_skipper` --
    /// `CheckIndex.checkDocValueSkipper`'s four pre-walk guards -- fires on a
    /// hand-built skip index. The per-level arms are already covered by
    /// `corrupting_the_doc_values_skipper_is_caught_by_the_skipper_check`'s
    /// sweep over real `.dvs` bytes; these four are not reachable that way,
    /// because they read `.dvm`-derived summary fields the sweep does not
    /// corrupt.
    #[test]
    fn the_doc_values_skipper_global_guards_each_fire_on_their_own_input() {
        let level = |min_doc_id, max_doc_id, min_value, max_value, doc_count| {
            doc_values::SkipIndexLevelInterval {
                min_doc_id,
                max_doc_id,
                min_value,
                max_value,
                doc_count,
            }
        };
        let one_interval = || doc_values::SkipIndexInterval {
            levels: vec![level(0, 3, -5, 5, 4)],
        };
        let clean = doc_values::DocValuesSkipIndex {
            min_value: -5,
            max_value: 5,
            doc_count: 4,
            max_doc_id: 3,
            max_value_count: 1,
            intervals: vec![one_interval()],
        };
        assert!(
            check_doc_value_skipper(&clean).is_empty(),
            "{:?}",
            check_doc_value_skipper(&clean)
        );

        // 1. an inverted global value range.
        let mut bad = clean.clone();
        bad.min_value = 6;
        let problems = check_doc_value_skipper(&bad);
        assert!(
            problems
                .iter()
                .any(|p| p.contains("inverted global value range: 6 > 5")),
            "{problems:?}"
        );

        // 2. `maxValueCount < -1`.
        let mut bad = clean.clone();
        bad.max_value_count = -2;
        let problems = check_doc_value_skipper(&bad);
        assert!(
            problems
                .iter()
                .any(|p| p.contains("invalid maxValueCount -2")),
            "{problems:?}"
        );

        // 3. a non-zero `maxValueCount` on a field with no documents.
        let empty = doc_values::DocValuesSkipIndex {
            min_value: 0,
            max_value: 0,
            doc_count: 0,
            max_doc_id: -1,
            max_value_count: 3,
            intervals: Vec::new(),
        };
        let problems = check_doc_value_skipper(&empty);
        assert!(
            problems
                .iter()
                .any(|p| p.contains("maxValueCount is 3 for a field with no documents")),
            "{problems:?}"
        );

        // 4. intervals that do not add up to the declared `docCount`.
        let mut bad = clean.clone();
        bad.doc_count = 9;
        let problems = check_doc_value_skipper(&bad);
        assert!(
            problems
                .iter()
                .any(|p| p.contains("declares docCount=9 but the intervals cover 4")),
            "{problems:?}"
        );

        // 5. `maxDocID(0) == i32::MAX` on a fresh skipper: the walk's
        //    `max_doc_id(0) + 1` used to be a plain `+`, so a `.dvs` whose
        //    first interval already sits at the sentinel panicked the
        //    verifier in a debug build instead of reporting the file.
        let sentinel = doc_values::DocValuesSkipIndex {
            min_value: 0,
            max_value: 0,
            doc_count: 1,
            max_doc_id: doc_values::NO_MORE_DOCS,
            max_value_count: 1,
            intervals: vec![doc_values::SkipIndexInterval {
                levels: vec![level(
                    doc_values::NO_MORE_DOCS,
                    doc_values::NO_MORE_DOCS,
                    0,
                    0,
                    1,
                )],
            }],
        };
        let problems = check_doc_value_skipper(&sentinel);
        assert!(
            !problems.is_empty(),
            "a sentinel-only skip index is corrupt"
        );
    }

    /// `doc_values.entry_present:<f>`, all five arms: a `.fnm` that claims a
    /// doc-values type the `.dvm` has no entry for.
    ///
    /// This is not a contrived shape. `Lucene90DocValuesProducer` routes each
    /// `.dvm` entry by the **type byte in the `.dvm` itself**, while every
    /// reader asks for a field's values by the type in the **`.fnm`**. When
    /// the two disagree the entry becomes unreachable and the field silently
    /// reads as "no doc values at all" -- every range query, sort and facet on
    /// it quietly returns nothing, and nothing else in this module notices,
    /// because the `.dvd` still decodes perfectly. Each of the five types has
    /// its own arm, so each needs its own case.
    #[test]
    fn a_fnm_claiming_a_doc_values_type_the_dvm_lacks_is_caught_for_every_type() {
        use field_infos::DocValuesType;
        let dst = copy_fixture("doc_values_index");
        let fnm_path = dst.join("_0.fnm");
        let original = std::fs::read(&fnm_path).unwrap();
        let dir = FsDirectory::open(&dst);
        let infos = segment_infos::read_latest(&dir).unwrap();
        let commit = infos.segments[0].clone();
        assert!(
            check_directory(&dir)
                .unwrap()
                .iter()
                .all(|r| r.all_passed()),
            "the fixture must start clean"
        );
        let parsed = field_infos::parse(&original, &commit.segment_id, "").unwrap();

        // `varying` is NUMERIC and `bin_var` is BINARY in the fixture, so
        // between them every one of the five claimed types has a case where
        // the `.dvm` entry is filed under a different one.
        for (field, claimed) in [
            ("varying", DocValuesType::Binary),
            ("varying", DocValuesType::Sorted),
            ("varying", DocValuesType::SortedNumeric),
            ("varying", DocValuesType::SortedSet),
            ("bin_var", DocValuesType::Numeric),
        ] {
            let mut fields = parsed.fields.clone();
            let fi = fields
                .iter_mut()
                .find(|f| f.name == field)
                .expect("fixture field");
            assert_ne!(fi.doc_values_type, claimed);
            fi.doc_values_type = claimed;
            std::fs::write(
                &fnm_path,
                field_infos::write(&fields, &commit.segment_id, ""),
            )
            .unwrap();

            let dir = FsDirectory::open(&dst);
            let failed = failed_names(&check_directory(&dir).unwrap());
            assert!(
                failed.contains(&format!("doc_values.entry_present:{field}")),
                "a .fnm claiming {claimed:?} for {field:?} with no matching .dvm entry was \
                 not caught: {failed:?}"
            );
        }

        std::fs::write(&fnm_path, &original).unwrap();
        let dir = FsDirectory::open(&dst);
        assert!(check_directory(&dir)
            .unwrap()
            .iter()
            .all(|r| r.all_passed()));
        std::fs::remove_dir_all(&dst).ok();
    }

    /// Every arm of `vectors.field_entry_matches_fnm`, driven by rewriting the
    /// `.fnm` of a real Java-written vectors segment.
    ///
    /// The `.vemf` records a vector field's dimension, encoding and
    /// similarity, and so does the `.fnm`; nothing on the wire ties the two
    /// together. A disagreement is silent and consequential -- the reader
    /// sizes its buffers from one and interprets distances with the other --
    /// and until this test none of the four arms that catch it had ever run.
    #[test]
    fn a_fnm_disagreeing_with_the_vemf_about_a_vector_field_is_caught() {
        let dst = copy_fixture("vectors_index");
        let fnm_path = dst.join("_0.fnm");
        let original = std::fs::read(&fnm_path).unwrap();
        let dir = FsDirectory::open(&dst);
        let infos = segment_infos::read_latest(&dir).unwrap();
        let commit = infos.segments[0].clone();
        assert!(
            check_directory(&dir)
                .unwrap()
                .iter()
                .all(|r| r.all_passed()),
            "the fixture must start clean"
        );
        let parsed = field_infos::parse(&original, &commit.segment_id, "").unwrap();
        let vector_field = parsed
            .fields
            .iter()
            .find(|f| f.vector_dimension > 0)
            .expect("the fixture has a vector field")
            .clone();

        let rewrite = |mutate: &dyn Fn(&mut field_infos::FieldInfo)| -> Vec<String> {
            let mut fields = parsed.fields.clone();
            let fi = fields
                .iter_mut()
                .find(|f| f.number == vector_field.number)
                .unwrap();
            mutate(fi);
            std::fs::write(
                &fnm_path,
                field_infos::write(&fields, &commit.segment_id, ""),
            )
            .unwrap();
            let dir = FsDirectory::open(&dst);
            failed_names(&check_directory(&dir).unwrap())
        };
        let name = &vector_field.name;

        // 1. a dimension the `.vemf` disagrees with.
        let failed = rewrite(&|fi| fi.vector_dimension += 1);
        assert!(
            failed.contains(&format!("vectors.field_entry_matches_fnm:{name}")),
            "{failed:?}"
        );

        // 2. an encoding the `.vemf` disagrees with.
        let flipped_encoding = match vector_field.vector_encoding {
            field_infos::VectorEncoding::Float32 => field_infos::VectorEncoding::Byte,
            field_infos::VectorEncoding::Byte => field_infos::VectorEncoding::Float32,
        };
        let failed = rewrite(&|fi| fi.vector_encoding = flipped_encoding);
        assert!(
            failed.contains(&format!("vectors.field_entry_matches_fnm:{name}")),
            "{failed:?}"
        );

        // 3. a similarity function the `.vemf` disagrees with. Nothing else
        //    in the segment can catch this one: the distances still compute,
        //    they are just the wrong distances.
        let flipped_similarity = match vector_field.vector_similarity_function {
            field_infos::VectorSimilarityFunction::Euclidean => {
                field_infos::VectorSimilarityFunction::DotProduct
            }
            _ => field_infos::VectorSimilarityFunction::Euclidean,
        };
        let failed = rewrite(&|fi| fi.vector_similarity_function = flipped_similarity);
        assert!(
            failed.contains(&format!("vectors.field_entry_matches_fnm:{name}")),
            "{failed:?}"
        );

        std::fs::write(&fnm_path, &original).unwrap();
        let dir = FsDirectory::open(&dst);
        assert!(check_directory(&dir)
            .unwrap()
            .iter()
            .all(|r| r.all_passed()));
        std::fs::remove_dir_all(&dst).ok();
    }

    /// `norms.entry_present:<f>`: a `.fnm` field that claims norms the `.nvm`
    /// has no entry for.
    ///
    /// Silent in normal use -- the field scores every document against a
    /// default norm, and nothing else notices, because
    /// `norms.entries_name_real_norms_fields` only looks the other way (an
    /// entry naming a field that does not exist). `.fnm` and `.nvm` are
    /// written by different consumers with nothing on the wire tying them
    /// together, which is why `CheckIndex` cross-checks them at all -- and
    /// why this arm needs a `.fnm` rewrite rather than a byte flip to reach.
    #[test]
    fn a_fnm_claiming_norms_the_nvm_does_not_have_is_caught() {
        let dst = copy_fixture("blocktree_index");
        let fnm_path = dst.join("_0.fnm");
        let original = std::fs::read(&fnm_path).unwrap();
        let dir = FsDirectory::open(&dst);
        let infos = segment_infos::read_latest(&dir).unwrap();
        let commit = infos.segments[0].clone();
        assert!(
            check_directory(&dir)
                .unwrap()
                .iter()
                .all(|r| r.all_passed()),
            "the fixture must start clean"
        );
        let parsed = field_infos::parse(&original, &commit.segment_id, "").unwrap();

        let rewrite = |mutate: &dyn Fn(&mut Vec<field_infos::FieldInfo>)| -> Vec<String> {
            let mut fields = parsed.fields.clone();
            mutate(&mut fields);
            std::fs::write(
                &fnm_path,
                field_infos::write(&fields, &commit.segment_id, ""),
            )
            .unwrap();
            let dir = FsDirectory::open(&dst);
            failed_names(&check_directory(&dir).unwrap())
        };

        // 1. a new indexed field that claims norms. The `.nvm` has no entry
        //    for it, so every score for the field would silently be computed
        //    against a default norm. (Adding a field rather than flipping
        //    `omitNorms` on an existing one: every indexed field in this
        //    fixture already has norms, and `FieldInfo::check_consistency`
        //    refuses norms on a *non*-indexed field, so there is no existing
        //    field to flip.)
        let template = parsed
            .fields
            .iter()
            .find(|f| f.index_options != field_infos::IndexOptions::None)
            .expect("the fixture has an indexed field")
            .clone();
        let ghost_number = parsed.fields.iter().map(|f| f.number).max().unwrap() + 1;
        let failed = rewrite(&|fields| {
            let mut ghost = template.clone();
            ghost.name = "ghost".to_string();
            ghost.number = ghost_number;
            ghost.omit_norms = false;
            ghost.doc_values_type = field_infos::DocValuesType::None;
            ghost.doc_values_skip_index_type = field_infos::DocValuesSkipIndexType::None;
            ghost.store_term_vectors = false;
            ghost.point_dimension_count = 0;
            ghost.point_index_dimension_count = 0;
            ghost.point_num_bytes = 0;
            ghost.vector_dimension = 0;
            fields.push(ghost);
        });
        assert!(
            failed.contains(&"norms.entry_present:ghost".to_string()),
            "a field claiming norms the .nvm lacks was not caught: {failed:?}"
        );

        std::fs::write(&fnm_path, &original).unwrap();
        let dir = FsDirectory::open(&dst);
        assert!(check_directory(&dir)
            .unwrap()
            .iter()
            .all(|r| r.all_passed()));
        std::fs::remove_dir_all(&dst).ok();
    }

    /// A SORTED_SET field where every document has exactly one value.
    ///
    /// `Lucene90DocValuesConsumer.addSortedSetField` collapses that shape to a
    /// plain `SortedEntry` (`multiValued = 0`), and the reader's
    /// `SortedSetKind::Single` branch decodes it -- so `check_doc_values` has
    /// a whole second SORTED_SET code path, with its own ordinal bookkeeping,
    /// that no fixture in this repo exercised. Every Java-written SORTED_SET
    /// fixture here happens to be genuinely multi-valued.
    fn write_single_valued_sorted_set_fixture(
        dst_dir: &std::path::Path,
        values: &[&[u8]],
    ) -> segment_infos::SegmentCommitInfo {
        let max_doc = values.len() as i32;
        let fi = field_infos::FieldInfo {
            name: "tags".to_string(),
            number: 0,
            store_term_vectors: false,
            omit_norms: false,
            store_payloads: false,
            soft_deletes_field: false,
            parent_field: false,
            index_options: field_infos::IndexOptions::None,
            doc_values_type: field_infos::DocValuesType::SortedSet,
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
        let fnm = field_infos::write(std::slice::from_ref(&fi), &SORT_SEG_ID, "");
        std::fs::write(dst_dir.join("_0.fnm"), &fnm).unwrap();
        let per_doc: Vec<Vec<Vec<u8>>> = values.iter().map(|v| vec![v.to_vec()]).collect();
        let (dvm, dvd, dvs) = lucene_codecs::doc_values::write_single_dense_sorted_set_field(
            0,
            &per_doc,
            max_doc,
            &SORT_SEG_ID,
            "",
        )
        .unwrap();
        std::fs::write(dst_dir.join("_0.dvm"), &dvm).unwrap();
        std::fs::write(dst_dir.join("_0.dvd"), &dvd).unwrap();
        std::fs::write(dst_dir.join("_0.dvs"), &dvs).unwrap();

        let si = SegmentInfo {
            id: SORT_SEG_ID,
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
                "_0.si".to_string(),
                "_0.fnm".to_string(),
                "_0.dvm".to_string(),
                "_0.dvd".to_string(),
                "_0.dvs".to_string(),
            ],
            attributes: vec![],
            index_sort: None,
        };
        std::fs::write(dst_dir.join("_0.si"), segment_info::write(&si, "")).unwrap();

        segment_infos::SegmentCommitInfo {
            segment_name: "_0".to_string(),
            segment_id: SORT_SEG_ID,
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

    /// The `SortedSetKind::Single` half of `check_doc_values`: a clean
    /// single-valued SORTED_SET passes every check, and a corrupted one is
    /// rejected by the ordinal-space checks that branch owns.
    #[test]
    fn a_single_valued_sorted_set_field_is_checked_through_the_single_branch() {
        let dst = tempdir();
        let commit = write_single_valued_sorted_set_fixture(
            &dst,
            &[b"alpha", b"beta", b"gamma", b"alpha", b"delta"],
        );
        let dir = FsDirectory::open(&dst);
        let result = check_segment(&dir, &commit);
        assert!(
            result.all_passed(),
            "a single-valued SORTED_SET must pass cleanly: {:?}",
            result.failures()
        );
        let names: Vec<&str> = result.checks.iter().map(|c| c.name.as_str()).collect();
        for expected in [
            "doc_values.values_decode:tags",
            "doc_values.ords_dense:tags",
            "doc_values.terms_sorted:tags",
        ] {
            assert!(
                names.contains(&expected),
                "{expected} did not run: {names:?}"
            );
        }

        // The branch's own ordinal bookkeeping really is live: re-signed
        // single-byte corruptions of the `.dvd` must be rejected by it.
        let dvd_path = dst.join("_0.dvd");
        let original = std::fs::read(&dvd_path).unwrap();
        let body_end = original.len() - lucene_store::codec_util::FOOTER_LENGTH;
        let mut caught = 0usize;
        let mut total = 0usize;
        for off in 48..body_end {
            for mask in [0x01u8, 0xff] {
                let mut bytes = original.clone();
                bytes[off] ^= mask;
                repair_checksum(&mut bytes);
                std::fs::write(&dvd_path, &bytes).unwrap();
                let dir = FsDirectory::open(&dst);
                total += 1;
                // Attributed to the ordinal-space checks specifically, not to
                // "some check failed": `doc_values.values_decode` alone would
                // satisfy a bare `all_passed()` counter, and the assertion
                // below would then say nothing about the `Single` branch.
                if check_segment(&dir, &commit).failures().iter().any(|c| {
                    c.name == "doc_values.ords_dense:tags"
                        || c.name == "doc_values.terms_sorted:tags"
                }) {
                    caught += 1;
                }
            }
        }
        std::fs::write(&dvd_path, &original).unwrap();
        // 17 of 54 when this was written. The dictionary is five short
        // terms, so most byte flips land inside a term's bytes and leave a
        // different but perfectly valid single-valued set behind -- the same
        // reason the `.dvd` control on `sorted_dv_index` accepts 54 of 99.
        assert!(total >= 50, "the sweep must actually run: {total}");
        assert!(
            caught >= 14,
            "only {caught} of {total} re-signed .dvd corruptions were rejected by the \
             SortedSetKind::Single branch's checks; 17 were when this test was written"
        );

        std::fs::remove_dir_all(&dst).ok();
    }

    /// `check_term_vectors`' three "the segment disagrees with itself" arms,
    /// on a real Java-written term-vectors segment: a `.si` that lists only
    /// some of `.tvd`/`.tvx`/`.tvm`, a `.si` whose `docCount` disagrees with
    /// the term-vectors reader's own `maxDoc`, and a `.fnm` that says a field
    /// carrying vectors does not store them.
    ///
    /// All three are cross-file: `.si`, `.fnm` and the `.tv*` triple are
    /// written by different consumers, and nothing on the wire ties them
    /// together -- which is why `CheckIndex` compares them and why none of
    /// these arms is reachable by corrupting bytes inside a single file.
    #[test]
    fn a_segment_that_disagrees_with_its_own_term_vectors_is_caught() {
        let dst = copy_fixture("term_vectors_index");
        let dir = FsDirectory::open(&dst);
        let infos = segment_infos::read_latest(&dir).unwrap();
        let commit = infos.segments[0].clone();
        let si_path = dst.join(format!("{}.si", commit.segment_name));
        let original_si = std::fs::read(&si_path).unwrap();
        let fnm_path = dst.join(format!("{}.fnm", commit.segment_name));
        let original_fnm = std::fs::read(&fnm_path).unwrap();
        let si = segment_info::parse(&original_si, &commit.segment_id).unwrap();
        assert!(
            check_segment(&dir, &commit).all_passed(),
            "the fixture must start clean: {:?}",
            check_segment(&dir, &commit).failures()
        );

        let write_si = |mutate: &dyn Fn(&mut SegmentInfo)| -> Vec<String> {
            let mut mutated = si.clone();
            mutate(&mut mutated);
            std::fs::write(&si_path, segment_info::write(&mutated, "")).unwrap();
            let dir = FsDirectory::open(&dst);
            check_segment(&dir, &commit)
                .failures()
                .iter()
                .map(|c| c.name.clone())
                .collect()
        };

        // 1. the `.si` lists `.tvd`/`.tvx` but not `.tvm`.
        let failed = write_si(&|s| s.files.retain(|f| !f.ends_with(".tvm")));
        assert!(
            failed.contains(&"term_vectors.open".to_string()),
            "a partial .tv* file set was not caught: {failed:?}"
        );

        // 2. the `.si` claims more documents than the term-vectors reader
        //    has. `doc_count_matches_si` is the only check that compares the
        //    two, and a reader that stops early would otherwise look clean.
        let failed = write_si(&|s| s.doc_count += 3);
        assert!(
            failed.contains(&"term_vectors.doc_count_matches_si".to_string()),
            "a .si docCount above the term-vectors reader's maxDoc was not caught: {failed:?}"
        );
        std::fs::write(&si_path, &original_si).unwrap();

        // 3. the `.fnm` says a field that carries vectors does not store
        //    them. Java reports this per document, and so does this port.
        let parsed = field_infos::parse(&original_fnm, &commit.segment_id, "").unwrap();
        let vectored = parsed
            .fields
            .iter()
            .find(|f| f.store_term_vectors)
            .expect("the fixture has a term-vector field")
            .name
            .clone();
        let mut fields = parsed.fields.clone();
        fields
            .iter_mut()
            .find(|f| f.name == vectored)
            .unwrap()
            .store_term_vectors = false;
        std::fs::write(
            &fnm_path,
            field_infos::write(&fields, &commit.segment_id, ""),
        )
        .unwrap();
        let dir = FsDirectory::open(&dst);
        let failed: Vec<String> = check_segment(&dir, &commit)
            .failures()
            .iter()
            .map(|c| c.name.clone())
            .collect();
        assert!(
            failed.contains(&"term_vectors.fields_marked_in_fnm".to_string()),
            "a .fnm denying term vectors a document actually carries was not caught: {failed:?}"
        );
        std::fs::write(&fnm_path, &original_fnm).unwrap();

        let dir = FsDirectory::open(&dst);
        assert!(check_segment(&dir, &commit).all_passed());
        std::fs::remove_dir_all(&dst).ok();
    }

    /// The NUMERIC/BINARY halves of `check_doc_values`' per-document decode,
    /// driven by re-signed single-byte corruptions of a real Java-written
    /// `.dvd` carrying dense numeric, GCD-compressed numeric, sparse numeric,
    /// fixed-length binary, variable-length binary and sparse binary fields.
    ///
    /// `corrupted_doc_values_payload_is_caught` covers the SORTED fixture;
    /// this one covers the five other entry shapes, whose decode-failure arms
    /// are separate code. Re-signed, so the `file:*` CRC cannot stand in for
    /// the value checks.
    #[test]
    fn no_re_signed_corruption_of_a_numeric_or_binary_dvd_goes_unnoticed() {
        let dst = copy_fixture("doc_values_index");
        let dvd_path = find_file(&dst, ".dvd");
        let original = std::fs::read(&dvd_path).unwrap();
        let dir = FsDirectory::open(&dst);
        assert!(
            check_directory(&dir)
                .unwrap()
                .iter()
                .all(|r| r.all_passed()),
            "the fixture must start clean"
        );

        let body_end = original.len() - lucene_store::codec_util::FOOTER_LENGTH;
        let mut caught_by_dv = 0usize;
        let mut caught_by_other = 0usize;
        let mut accepted = 0usize;
        let mut isolated: Option<(usize, u8)> = None;
        for off in 48..body_end {
            for mask in [0x01u8, 0x80, 0xff] {
                let mut bytes = original.clone();
                bytes[off] ^= mask;
                repair_checksum(&mut bytes);
                std::fs::write(&dvd_path, &bytes).unwrap();
                let dir = FsDirectory::open(&dst);
                // A corruption that makes the *commit* unreadable is still a
                // corruption this module rejected -- counting it as
                // "caught by another check" keeps `total` equal to the sweep's
                // own iteration count, which is what makes the floors below
                // mean what they say.
                let failed = match check_directory(&dir) {
                    Ok(r) => failed_names(&r),
                    Err(e) => vec![format!("commit.unreadable: {e}")],
                };
                let is_dv = |n: &String| n.starts_with("doc_values.");
                if failed.iter().any(is_dv) {
                    caught_by_dv += 1;
                    if failed.iter().all(is_dv) {
                        isolated.get_or_insert((off, mask));
                    }
                } else if failed.is_empty() {
                    accepted += 1;
                } else {
                    caught_by_other += 1;
                }
            }
        }
        std::fs::write(&dvd_path, &original).unwrap();
        let total = caught_by_dv + caught_by_other + accepted;
        // Measured when this was written: of 261 re-signed corruptions, 69
        // were rejected by a `doc_values.*` check, **0** by anything else,
        // and 192 accepted. The zero matters as much as the 69: nothing but
        // these checks reads the `.dvd`, so a corruption they miss is a
        // corruption nothing catches. The 192 are the same phenomenon as the
        // `.tip` and single-valued-SORTED_SET controls -- most of this file
        // is raw values, and a different value is still a valid one.
        assert!(
            caught_by_dv >= 60,
            "the doc-values checks rejected only {caught_by_dv} of {total} re-signed .dvd \
             corruptions ({caught_by_other} caught by another check, {accepted} accepted); \
             69 were rejected when this test was written"
        );
        assert_eq!(
            caught_by_other, 0,
            "a .dvd corruption was reported by something other than a doc_values.* check"
        );
        let (off, mask) = isolated.expect("at least one corruption is caught by doc_values alone");
        let mut bytes = original.clone();
        bytes[off] ^= mask;
        repair_checksum(&mut bytes);
        std::fs::write(&dvd_path, &bytes).unwrap();
        let dir = FsDirectory::open(&dst);
        let failed = failed_names(&check_directory(&dir).unwrap());
        std::fs::write(&dvd_path, &original).unwrap();
        assert!(
            !failed.is_empty() && failed.iter().all(|n| n.starts_with("doc_values.")),
            "byte {off} ^ {mask:#x} was expected to trip only the doc-values checks: {failed:?}"
        );

        let dir = FsDirectory::open(&dst);
        assert!(check_directory(&dir)
            .unwrap()
            .iter()
            .all(|r| r.all_passed()));
        std::fs::remove_dir_all(&dst).ok();
    }

    /// `checkSoftDeletes` on a segment that has *both* soft deletes and hard
    /// deletes.
    ///
    /// Java's `PendingSoftDeletes.countSoftDeletes` counts only documents that
    /// are still live: a document that is both soft-deleted and hard-deleted
    /// must not be counted twice. That is the `del_gen != -1` branch, which
    /// reads the `.liv` and intersects it with the soft-deletes field --
    /// untested until now, because every soft-deletes fixture here had
    /// `del_gen == -1`, so the check only ever ran with `live = None`.
    #[test]
    fn soft_deletes_are_counted_only_among_live_documents() {
        use lucene_util::fixed_bit_set::FixedBitSet;

        let dst = tempdir();
        // All four docs carry the soft-deletes field, but doc 3 is also hard
        // deleted -- so the live soft-delete count is 3, not 4.
        let mut commit = write_sorted_dv_fixture(&dst, &[1, 1, 1, 1], None, true, 3);
        let mut live = FixedBitSet::new(4);
        for doc in 0..3 {
            live.set(doc);
        }
        let liv = live_docs::write(&live, &commit.segment_id, 1, 1).unwrap();
        std::fs::write(dst.join(liv_file_name("_0", 1)), &liv).unwrap();
        commit.del_gen = 1;
        commit.del_count = 1;

        let dir = FsDirectory::open(&dst);
        let result = check_segment(&dir, &commit);
        let check = result
            .checks
            .iter()
            .find(|c| c.name == "soft_deletes.count_matches")
            .expect("the soft-deletes check must run");
        assert!(
            check.passed(),
            "a hard-deleted doc must not count towards softDelCount: {}",
            check.message
        );

        // And the count really is live-aware: claiming 4 (every doc, ignoring
        // the hard delete) must now fail.
        commit.soft_del_count = 4;
        let dir = FsDirectory::open(&dst);
        let result = check_segment(&dir, &commit);
        let check = result
            .checks
            .iter()
            .find(|c| c.name == "soft_deletes.count_matches")
            .unwrap();
        assert!(!check.passed());
        assert!(
            check.message.contains("actual soft deletes: 3"),
            "{}",
            check.message
        );

        std::fs::remove_dir_all(&dst).ok();
    }

    /// Every subsystem's `*.open` arm: a file the `.si` lists, replaced by a
    /// well-formed file of a *different* kind from the same segment.
    ///
    /// This is the failure `check_index` most needs to survive gracefully.
    /// Each decoder validates a codec header naming its own format, so the
    /// wrong file is rejected at open -- and the module's contract is that it
    /// then *reports* the failure under that subsystem's own name and keeps
    /// checking the rest of the segment, rather than panicking or abandoning
    /// the segment. Every one of these `Err(e) => checks.push(Check::fail(
    /// "<x>.open", ..))` arms existed unexercised; a decoder change that
    /// turned one of them into a panic would have gone unnoticed.
    #[test]
    fn a_file_replaced_by_another_kind_is_reported_by_its_own_open_check() {
        // (fixture, victim extension, donor extension, expected check prefix)
        let cases: &[(&str, &str, &str, &str)] = &[
            ("vectors_index", ".vemf", ".vec", "vectors.open"),
            ("vectors_index", ".vex", ".vec", "hnsw.open"),
            ("points_index", ".kdm", ".fdt", "points.open"),
            ("points_index", ".kdi", ".fdt", "points."),
            ("term_vectors_index", ".tvm", ".fdt", "term_vectors.open"),
            ("blocktree_index", ".fdt", ".nvd", "stored_fields."),
            ("blocktree_index", ".nvm", ".fdt", "norms.open"),
            ("blocktree_index", ".tim", ".nvd", "postings.open"),
            ("doc_values_index", ".dvm", ".fdt", "doc_values."),
            ("doc_values_skip_index", ".dvs", ".dvd", "doc_values."),
            ("blocktree_index", ".tip", ".nvd", "postings."),
            ("blocktree_index", ".tmd", ".nvd", "postings."),
            ("blocktree_index", ".doc", ".nvd", "postings."),
            ("vectors_index", ".vec", ".vemf", "vectors."),
            ("term_vectors_index", ".tvd", ".fdt", "term_vectors."),
            ("points_index", ".kdd", ".fdt", "points."),
        ];
        for (fixture, victim_ext, donor_ext, expected) in cases {
            let dst = copy_fixture(fixture);
            let victim = find_file(&dst, victim_ext);
            let donor = std::fs::read(find_file(&dst, donor_ext)).unwrap();
            let original = std::fs::read(&victim).unwrap();
            std::fs::write(&victim, &donor).unwrap();

            let dir = FsDirectory::open(&dst);
            // `check_directory` must not panic, and must not give up on the
            // segment: the commit-level result is always produced.
            let results = check_directory(&dir).expect("the commit itself is untouched");
            let failed = failed_names(&results);
            assert!(
                failed.iter().any(|n| n.starts_with(expected)),
                "{fixture}: {victim_ext} replaced by {donor_ext} was not reported by \
                 {expected}*; failures: {failed:?}"
            );

            std::fs::write(&victim, &original).unwrap();
            let dir = FsDirectory::open(&dst);
            assert!(
                check_directory(&dir)
                    .unwrap()
                    .iter()
                    .all(|r| r.all_passed()),
                "{fixture} must be clean again after restoring {victim_ext}"
            );
            std::fs::remove_dir_all(&dst).ok();
        }
    }

    /// `testSort` over a **SORTED_NUMERIC** sort field.
    ///
    /// `sort_key_values` has a separate arm per doc-values type, and Lucene
    /// sorts on the *minimum* value of a multi-valued SORTED_NUMERIC field
    /// (`SortedNumericSortField`'s `MIN` selector, which is what
    /// `IndexWriterConfig.setIndexSort` produces). Every index-sort fixture
    /// here used a plain NUMERIC field, so that arm -- and the `min()` that
    /// makes multi-valued sorting well-defined at all -- had never run.
    #[test]
    fn an_index_sort_on_a_multi_valued_sorted_numeric_field_is_verified() {
        let write = |dst: &std::path::Path, values: &[Vec<i64>]| {
            let fi = field_infos::FieldInfo {
                name: "ts".to_string(),
                number: 0,
                store_term_vectors: false,
                omit_norms: false,
                store_payloads: false,
                soft_deletes_field: false,
                parent_field: false,
                index_options: field_infos::IndexOptions::None,
                doc_values_type: field_infos::DocValuesType::SortedNumeric,
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
            std::fs::write(
                dst.join("_0.fnm"),
                field_infos::write(std::slice::from_ref(&fi), &SORT_SEG_ID, ""),
            )
            .unwrap();
            let (dvm, dvd, dvs) =
                lucene_codecs::doc_values::write_single_dense_sorted_numeric_field(
                    0,
                    values,
                    &SORT_SEG_ID,
                    "",
                )
                .unwrap();
            std::fs::write(dst.join("_0.dvm"), &dvm).unwrap();
            std::fs::write(dst.join("_0.dvd"), &dvd).unwrap();
            std::fs::write(dst.join("_0.dvs"), &dvs).unwrap();
            let si = SegmentInfo {
                id: SORT_SEG_ID,
                version: segment_info::LuceneVersion {
                    major: 10,
                    minor: 0,
                    bugfix: 0,
                },
                min_version: None,
                doc_count: values.len() as i32,
                is_compound_file: false,
                has_blocks: false,
                diagnostics: vec![],
                files: vec![
                    "_0.si".to_string(),
                    "_0.fnm".to_string(),
                    "_0.dvm".to_string(),
                    "_0.dvd".to_string(),
                    "_0.dvs".to_string(),
                ],
                attributes: vec![],
                index_sort: sorted_numeric_sort_asc(),
            };
            std::fs::write(dst.join("_0.si"), segment_info::write(&si, "")).unwrap();
            segment_infos::SegmentCommitInfo {
                segment_name: "_0".to_string(),
                segment_id: SORT_SEG_ID,
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
        };

        // Sorted by each document's *minimum* value: 1, 3, 3, 9.
        let good = tempdir();
        let commit = write(&good, &[vec![1, 50], vec![3, 4], vec![3], vec![9, 9]]);
        let dir = FsDirectory::open(&good);
        let result = check_segment(&dir, &commit);
        assert!(
            result.all_passed(),
            "a correctly sorted SORTED_NUMERIC segment must pass: {:?}",
            result.failures()
        );
        assert!(result
            .checks
            .iter()
            .any(|c| c.name == "sort.docs_in_index_sort_order"));
        std::fs::remove_dir_all(&good).ok();

        // Sorted by each document's *maximum* instead: 50, 4, 3, 9 -- which
        // is out of order under `MIN`, and is exactly the mistake a writer
        // that picked the wrong selector would make.
        let bad = tempdir();
        let commit = write(&bad, &[vec![1, 50], vec![3, 4], vec![3], vec![0, 9]]);
        let dir = FsDirectory::open(&bad);
        let result = check_segment(&dir, &commit);
        let check = result
            .checks
            .iter()
            .find(|c| c.name == "sort.docs_in_index_sort_order")
            .expect("the sort check must run");
        assert!(!check.passed(), "{}", check.message);
        std::fs::remove_dir_all(&bad).ok();

        // A document whose values are stored *descending*.
        // `SortedNumericDocValuesWriter.finishCurrentDoc` sorts them, so real
        // Lucene never writes this and real
        // `CheckIndex.checkSortedNumericDocValues` throws `"values out of
        // order"` on it; the column is also what makes
        // `SortedNumericSelector.MIN`/`MAX` -- the first and the last stored
        // value -- mean what they say.
        let unsorted = tempdir();
        let commit = write(&unsorted, &[vec![1, 50], vec![4, 3], vec![9], vec![9, 9]]);
        let dir = FsDirectory::open(&unsorted);
        let result = check_segment(&dir, &commit);
        let check = result
            .checks
            .iter()
            .find(|c| c.name == "doc_values.values_decode:ts")
            .expect("the doc-values check must run");
        assert!(!check.passed(), "{}", check.message);
        assert!(
            check.message.contains("values out of order: 3 < 4"),
            "{}",
            check.message
        );
        std::fs::remove_dir_all(&unsorted).ok();
    }

    // ---------------------------------------------------------------------
    // c25: the never-fired check arms.
    //
    // Every `Check::fail`/`problems.push` below had been observed to fire
    // exactly zero times before this batch, which makes it indistinguishable
    // -- from outside -- from a check that passes. Each test drives one
    // family and asserts the *named* check fails, not merely that something
    // did.
    // ---------------------------------------------------------------------

    /// A `FieldInfo` that claims **every** per-format feature at once, so one
    /// segment can drive all seven format checks' guards in a single pass.
    fn all_claiming_field_info() -> field_infos::FieldInfo {
        field_infos::FieldInfo {
            name: "everything".to_string(),
            number: 0,
            store_term_vectors: true,
            omit_norms: false,
            store_payloads: false,
            soft_deletes_field: true,
            parent_field: false,
            index_options: field_infos::IndexOptions::DocsAndFreqs,
            doc_values_type: field_infos::DocValuesType::Numeric,
            doc_values_skip_index_type: field_infos::DocValuesSkipIndexType::None,
            doc_values_gen: -1,
            attributes: vec![],
            point_dimension_count: 1,
            point_index_dimension_count: 1,
            point_num_bytes: 8,
            vector_dimension: 4,
            vector_encoding: field_infos::VectorEncoding::Float32,
            vector_similarity_function: field_infos::VectorSimilarityFunction::Euclidean,
        }
    }

    const CLAIMS_SEG_ID: [u8; ID_LENGTH] = [31u8; ID_LENGTH];

    fn claiming_si(is_compound_file: bool, files: Vec<String>) -> SegmentInfo {
        SegmentInfo {
            id: CLAIMS_SEG_ID,
            version: segment_info::LuceneVersion {
                major: 10,
                minor: 0,
                bugfix: 0,
            },
            min_version: None,
            doc_count: 2,
            is_compound_file,
            has_blocks: false,
            diagnostics: vec![],
            files,
            attributes: vec![],
            index_sort: sort_asc(),
        }
    }

    /// Every codec file the claiming field's formats would open. The files
    /// do not exist on disk: the point is that each check gets *past* its
    /// "this segment has no such files" early return, so the only thing that
    /// can still keep it quiet is the guard under test.
    fn claiming_files() -> Vec<String> {
        [
            "_0.si",
            "_0.fnm",
            "_0.dvm",
            "_0.dvd",
            "_0.nvm",
            "_0.nvd",
            "_0.tvd",
            "_0.tvx",
            "_0.tvm",
            "_0.kdm",
            "_0.kdi",
            "_0.kdd",
            "_0_Lucene99HnswVectorsFormat_0.vec",
            "_0_Lucene99HnswVectorsFormat_0.vemf",
            "_0_Lucene104_0.tim",
            "_0_Lucene104_0.tip",
            "_0_Lucene104_0.tmd",
        ]
        .into_iter()
        .map(str::to_string)
        .collect()
    }

    fn claiming_commit() -> segment_infos::SegmentCommitInfo {
        segment_infos::SegmentCommitInfo {
            segment_name: "_0".to_string(),
            segment_id: CLAIMS_SEG_ID,
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

    /// This module has no compound-file (`.cfs`/`.cfe`) support, so every
    /// per-format check returns **silently** for a compound segment rather
    /// than reporting a missing file. That is a deliberate, documented scope
    /// decision -- and only one of the seven guards that implement it
    /// (`check_points_structural_invariants`') had ever been executed by a
    /// test. The other six could have been reporting spurious failures on
    /// every compound segment in existence, and nothing would have said so.
    #[test]
    fn a_compound_segment_skips_every_format_check() {
        let fields = FieldInfos {
            fields: vec![all_claiming_field_info()],
        };
        let commit = claiming_commit();
        let dst = tempdir();
        let dir = FsDirectory::open(&dst);

        // Run every per-format check over one `.si`, twice: once compound and
        // once not. The file list names every codec file the claiming field's
        // formats look for and **none of them exists on disk**, so the
        // non-compound run must report -- which is what makes the compound
        // run's silence mean "the guard fired" rather than "there was nothing
        // to do anyway". Without this pairing, five of the seven assertions
        // below would also hold with the guards deleted, because each check's
        // "this segment has no such files" early return would take over. That
        // is exactly the c19 weakness this batch exists to remove, and the
        // first version of this test had it.
        let run = |is_compound_file: bool| -> Vec<Check> {
            let si = claiming_si(is_compound_file, claiming_files());
            let mut stats = CheckStats::default();
            let mut checks = Vec::new();
            check_term_vectors(
                &dir,
                &commit,
                &si,
                Some(&fields),
                None,
                None,
                &mut stats,
                &mut checks,
            );
            check_doc_values(&dir, &commit, &si, &fields, &mut checks);
            check_index_sort(&dir, &commit, &si, &fields, &mut checks);
            check_soft_deletes(&dir, &commit, &si, &fields, &mut checks);
            check_field_norms(
                &dir,
                &commit,
                &si,
                &fields,
                None,
                &[],
                &mut stats,
                &mut checks,
            );
            check_vectors(&dir, &commit, &si, &fields, &mut stats, &mut checks);
            check_points_structural_invariants(&dir, &commit, &si, &fields, &mut checks);
            checks
        };

        // The control: with the compound flag off, every one of the seven
        // reports the files it cannot open, under its own subsystem's name.
        let reported = run(false);
        let mut names: Vec<&str> = reported
            .iter()
            .filter(|c| c.outcome == Outcome::Failed)
            .map(|c| {
                c.name
                    .split_once(':')
                    .map_or(c.name.as_str(), |(family, _)| family)
            })
            .collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(
            names,
            [
                "doc_values.open",
                "norms.open",
                "points.open",
                "soft_deletes.count_matches",
                "sort.docs_in_index_sort_order",
                "term_vectors.open",
                "vectors.open",
            ],
            "the non-compound control must reach all seven walks: {reported:?}"
        );
        // Each of those seven takes its own family down with it, and says so.
        assert!(
            reported.iter().filter(|c| c.was_skipped()).count() >= 20,
            "a failed *.open must name what it took down: {reported:?}"
        );

        // The subject: the same segment marked compound reports nothing at
        // all.
        let skipped = run(true);
        assert!(
            skipped.is_empty(),
            "a compound segment must produce no per-format check at all: {:?}",
            skipped.iter().map(|c| &c.name).collect::<Vec<_>>()
        );
        // The postings walk is skipped at the same boundary. Its control is
        // the same shape: with the flag off it finds the `.tim`/`.tip`/`.tmd`
        // the `.si` lists and reports that they cannot be opened.
        assert!(
            open_postings_bytes(&dir, &claiming_si(true, claiming_files())).is_none(),
            "a compound segment must not open postings"
        );
        assert!(
            matches!(
                open_postings_bytes(&dir, &claiming_si(false, claiming_files())),
                Some(Err(_))
            ),
            "the non-compound control must try, and fail, to open the postings"
        );
        std::fs::remove_dir_all(&dst).ok();
    }

    /// A field that claims a format whose files the segment does not have is
    /// reported **once**, by `fnm.*_vs_files`, and the format's own walk is
    /// then skipped rather than reporting the same absence a second time
    /// under a different name. Vectors are the one deliberate exception: a
    /// `.vec`/`.vemf`-less vector field has no `fnm.vectors_vs_files`
    /// counterpart (there is no such check), so `vectors.open` is the only
    /// place it can be reported and it *must* fire.
    ///
    /// Five of these six early returns had never run, which means the
    /// no-duplicate-reporting decision was entirely unverified.
    #[test]
    fn a_field_claiming_a_format_with_no_files_is_not_walked_twice() {
        let fields = FieldInfos {
            fields: vec![all_claiming_field_info()],
        };
        let si = claiming_si(false, vec!["_0.si".to_string(), "_0.fnm".to_string()]);
        let commit = claiming_commit();
        let dst = tempdir();
        let dir = FsDirectory::open(&dst);

        // First prove the claims are *live*, so the silence asserted below is
        // a decision and not an accident of a `FieldInfo` that forgot to
        // claim anything. This is the `fnm.*_vs_files` half of "reported
        // once": all four families report here, and nowhere else.
        let mut claims = Vec::new();
        check_field_flags_vs_files(&fields, &si.files, &mut claims);
        let claimed: Vec<&str> = claims
            .iter()
            .filter(|c| !c.passed())
            .map(|c| c.name.as_str())
            .collect();
        assert_eq!(
            claimed,
            [
                "fnm.doc_values_vs_files",
                "fnm.norms_vs_files",
                "fnm.term_vectors_vs_files",
                "fnm.postings_vs_files",
            ],
            "the field must actually claim all four file groups"
        );
        assert!(
            si.index_sort.is_some(),
            "the .si must declare an index sort"
        );
        assert!(
            fields.fields[0].soft_deletes_field,
            "the field must be the soft-deletes field"
        );

        let mut checks = Vec::new();
        check_doc_values(&dir, &commit, &si, &fields, &mut checks);
        let mut stats = CheckStats::default();
        check_field_norms(
            &dir,
            &commit,
            &si,
            &fields,
            None,
            &[],
            &mut stats,
            &mut checks,
        );
        assert!(
            checks.is_empty(),
            "doc values and norms must stay quiet when the segment has no \
             .dvm/.dvd/.nvm/.nvd -- `fnm.doc_values_vs_files` and \
             `fnm.norms_vs_files` above already reported it: {:?}",
            checks.iter().map(|c| &c.name).collect::<Vec<_>>()
        );

        // The index sort and the soft-deletes count are different: their
        // *claims* live in the `.si` and `.fnm`, not in the doc-values files,
        // so `fnm.doc_values_vs_files` does not cover them and staying quiet
        // would leave a declared sort order verified by nothing at all. They
        // report a skip -- which is not a pass.
        check_index_sort(&dir, &commit, &si, &fields, &mut checks);
        check_soft_deletes(&dir, &commit, &si, &fields, &mut checks);
        assert_eq!(
            checks
                .iter()
                .map(|c| (c.name.as_str(), c.outcome))
                .collect::<Vec<_>>(),
            [
                ("sort.docs_in_index_sort_order", Outcome::Skipped),
                ("soft_deletes.count_matches", Outcome::Skipped),
            ],
            "{checks:?}"
        );
        assert!(!checks.iter().any(|c| c.passed()));
        checks.clear();
        // Vectors and points are the two deliberate exceptions: neither has
        // an `fnm.*_vs_files` counterpart above, so their own `open` check is
        // the only place a missing file group can be reported -- and it must
        // fire, or the absence goes unreported entirely.
        check_vectors(&dir, &commit, &si, &fields, &mut stats, &mut checks);
        check_points_structural_invariants(&dir, &commit, &si, &fields, &mut checks);
        let failed: Vec<&str> = checks
            .iter()
            .filter(|c| c.outcome == Outcome::Failed)
            .map(|c| c.name.as_str())
            .collect();
        assert_eq!(failed, ["vectors.open", "points.open"], "{checks:?}");
        // ... and each names the families it took down, rather than leaving
        // them absent from the result.
        assert_eq!(
            checks.iter().filter(|c| c.was_skipped()).count(),
            VECTOR_FAMILIES.len() + POINTS_FAMILIES.len(),
            "{checks:?}"
        );
        assert!(!checks.iter().any(|c| c.passed()));
        std::fs::remove_dir_all(&dst).ok();
    }

    /// `SegmentInfos.readCommit`'s three cross-`.si` header validations, none
    /// of which had ever failed in a test: a segment older than the commit's
    /// own recorded `minSegmentLuceneVersion`, a segment older than
    /// `indexCreatedVersionMajor`, and a segment with no `minVersion` at all
    /// once `indexCreatedVersionMajor >= 7`.
    ///
    /// This port's `segment_infos::parse` never opens a `.si`, so these are
    /// the only place the two headers are ever compared -- a commit that
    /// claims to have been written by a version newer than the segments it
    /// contains would otherwise parse, open and query perfectly while lying
    /// about what wrote it (which is what every back-compat gate reads).
    #[test]
    fn a_commit_header_ahead_of_its_own_segment_is_caught() {
        use lucene_codecs::postings_writer::TermPostings;
        let dst = tempdir();
        let commit = write_postings_fixture(
            &dst,
            &[TermPostings {
                term: b"alpha".to_vec(),
                docs: vec![(0, 1), (1, 2)],
                positions: vec![],
                offsets: vec![],
                payloads: vec![],
            }],
            2,
            2,
            None,
        );
        let dir = FsDirectory::open(&dst);

        // The same segment, checked against a commit header that agrees with
        // it, must not report any of the three.
        let agreeing = SegmentInfos {
            id: CLAIMS_SEG_ID,
            generation: 1,
            format_version: 10,
            lucene_version: segment_infos::LuceneVersion {
                major: 10,
                minor: 0,
                bugfix: 0,
            },
            index_created_version_major: 6,
            version: 1,
            counter: 99,
            min_segment_lucene_version: Some(segment_infos::LuceneVersion {
                major: 9,
                minor: 0,
                bugfix: 0,
            }),
            segments: vec![],
            user_data: vec![],
        };
        let result = check_segment_in_commit(&dir, Some(&agreeing), &commit);
        assert!(
            result.all_passed(),
            "an agreeing commit header must pass: {:?}",
            result.failures()
        );

        // `indexCreatedVersionMajor = 6` above is deliberate: it is below
        // Java's `>= 7` gate, so the second and third checks do not even run.
        assert!(!result
            .checks
            .iter()
            .any(|c| c.name == "commit.segment_records_min_version"));

        let mut disagreeing = agreeing;
        disagreeing.min_segment_lucene_version = Some(segment_infos::LuceneVersion {
            major: 11,
            minor: 0,
            bugfix: 0,
        });
        disagreeing.index_created_version_major = 11;
        let result = check_segment_in_commit(&dir, Some(&disagreeing), &commit);
        for name in [
            "commit.segment_version_at_or_after_min",
            "commit.segment_version_at_or_after_created",
            "commit.segment_records_min_version",
        ] {
            let check = result
                .checks
                .iter()
                .find(|c| c.name == name)
                .unwrap_or_else(|| panic!("{name} must have run"));
            assert!(!check.passed(), "{name} should have failed: {check:?}");
        }
        std::fs::remove_dir_all(&dst).ok();
    }

    /// `SegmentInfos.readCommit`'s `softDelCount` bound. `del_count`'s twin
    /// was tested; this one -- and the combined
    /// `delCount + softDelCount <= maxDoc` bound -- were not, so a commit
    /// claiming more soft deletes than the segment has documents was
    /// validated by nothing at all.
    #[test]
    fn soft_del_count_larger_than_max_doc_is_flagged() {
        let dir = FsDirectory::open(fixture_dir("live_docs_index"));
        let mut commit = read_commit(&dir);
        commit.soft_del_count = 1_000_000;
        let result = check_segment(&dir, &commit);
        for name in [
            "commit.soft_del_count_within_max_doc",
            "commit.del_plus_soft_del_within_max_doc",
        ] {
            let check = result.checks.iter().find(|c| c.name == name).unwrap();
            assert!(!check.passed(), "{check:?}");
        }
    }

    /// `CheckIndex.updateMaxSegmentName` parses every segment name as
    /// `_<base36>`; a name it cannot parse leaves `maxSegmentName` at its
    /// initial value, which silently turns the counter check into a
    /// tautology. This port reports the unparsable name instead -- an arm
    /// that had never fired, because every fixture in the repo is written by
    /// a real `IndexWriter` and so is always well-formed.
    #[test]
    fn a_segment_name_that_is_not_base36_is_flagged() {
        let dir = FsDirectory::open(fixture_dir("blocktree_index"));
        let mut infos = segment_infos::read_latest(&dir).unwrap();
        infos.segments[0].segment_name = "not-a-segment".to_string();
        let result = check_commit(&infos);
        let check = result
            .checks
            .iter()
            .find(|c| c.name == "commit.segment_names_well_formed")
            .unwrap();
        assert!(!check.passed(), "{check:?}");
        assert!(check.message.contains("not-a-segment"), "{check:?}");
    }

    /// `SegmentInfos.readCommit`'s LUCENE-6299 bound: the readers' total
    /// `maxDoc` cannot exceed `IndexWriter.MAX_DOCS`. It is the one
    /// commit-level check that needs every segment's `.si` to have been
    /// opened, which is why it is appended by `check_directory` rather than
    /// computed in `check_commit` -- and it had never been observed to fire,
    /// so the append was unverified too.
    #[test]
    fn a_commit_whose_segments_exceed_max_docs_is_flagged() {
        let dst = tempdir();
        let si = SegmentInfo {
            id: CLAIMS_SEG_ID,
            version: segment_info::LuceneVersion {
                major: 10,
                minor: 0,
                bugfix: 0,
            },
            min_version: None,
            // One document past `MAX_DOCS`, still inside `i32`.
            doc_count: (MAX_DOCS + 1) as i32,
            is_compound_file: false,
            has_blocks: false,
            diagnostics: vec![],
            files: vec!["_0.si".to_string()],
            attributes: vec![],
            index_sort: None,
        };
        std::fs::write(dst.join("_0.si"), segment_info::write(&si, "")).unwrap();
        let infos = SegmentInfos {
            id: CLAIMS_SEG_ID,
            generation: 1,
            format_version: 10,
            lucene_version: segment_infos::LuceneVersion {
                major: 10,
                minor: 0,
                bugfix: 0,
            },
            index_created_version_major: 10,
            version: 1,
            counter: 99,
            min_segment_lucene_version: None,
            segments: vec![claiming_commit()],
            user_data: vec![],
        };
        let dir = FsDirectory::open(&dst);
        segment_infos::write(&infos, &dir).expect("write segments_1");

        let dir = FsDirectory::open(&dst);
        let results = check_directory(&dir).expect("read segments_N");
        let check = results[0]
            .checks
            .iter()
            .find(|c| c.name == "commit.total_max_doc_within_bounds")
            .expect("the total-maxDoc check must run");
        assert!(!check.passed(), "{check:?}");
        assert_eq!(results[1].max_doc, Some((MAX_DOCS + 1) as i32));
        std::fs::remove_dir_all(&dst).ok();
    }

    // -- the term dictionary's own claims, rewritten -------------------------
    //
    // `postings_writer` computes every term statistic from the postings it is
    // given, so it cannot be *asked* for a dictionary that lies -- which is
    // precisely the file `check_postings` exists to reject. These two helpers
    // decode the written `.tim` term-statistics region and the `.tmd`
    // field-summary record, let a test change one claim, and re-encode. That
    // is a negative control in c15's shape: the bytes are real writer output
    // apart from the one semantic claim under test, and every checksum is
    // re-signed so `file:*`'s CRC cannot "catch" the corruption first.

    const TIM_CODEC: &str = "BlockTreeTermsDict";
    const TMD_CODEC: &str = "BlockTreeTermsMeta";
    const TMD_POSTINGS_CODEC: &str = "Lucene104PostingsWriterTerms";

    /// A minimal varint writer over the same encoding
    /// `lucene_store::data_output` uses, spelled out here because the
    /// negative deltas below are exactly what a `write_vlong` refuses to
    /// emit: `.tim` stores `totalTermFreq - docFreq` as an *unsigned* vlong,
    /// so a `totalTermFreq < docFreq` can only be spelled as the ten-byte
    /// encoding of a negative `i64` -- and `DataInput::read_vlong` accepts
    /// it, which is what makes that check falsifiable at all.
    fn push_raw_vlong(out: &mut Vec<u8>, value: i64) {
        let mut v = value as u64;
        loop {
            let byte = (v & 0x7f) as u8;
            v >>= 7;
            if v == 0 {
                out.push(byte);
                return;
            }
            out.push(byte | 0x80);
        }
    }

    fn push_vint(out: &mut Vec<u8>, value: i32) {
        push_raw_vlong(out, i64::from(value) & 0xffff_ffff);
    }

    /// Rewrites the per-term `(docFreq, totalTermFreq)` pairs in a hand-built
    /// fixture's single `.tim` leaf block, returning the new `.tim` bytes and
    /// the new whole-file length the `.tmd` must record for it.
    fn patch_tim_stats(
        tim: &[u8],
        suffix: &str,
        mut edit: impl FnMut(usize, &mut i32, &mut i64),
    ) -> Vec<u8> {
        use lucene_store::data_input::{DataInput, SliceInput};
        let start = codec_util::index_header_length(TIM_CODEC, suffix);
        let mut input = SliceInput::new(&tim[..tim.len() - codec_util::FOOTER_LENGTH]);
        input.seek(start).unwrap();
        let code = input.read_vint().unwrap();
        let ent_count = (code as u32 >> 1) as usize;
        let code_l = input.read_vlong().unwrap();
        let suffix_len = (code_l as u64 >> 3) as usize;
        let suffix_at = input.position();
        input.seek(suffix_at + suffix_len).unwrap();
        let suffix_bytes = &tim[suffix_at..suffix_at + suffix_len];
        let sl_code = input.read_vint().unwrap();
        let sl_len = (sl_code as u32 >> 1) as usize;
        let sl_at = input.position();
        input.seek(sl_at + sl_len).unwrap();
        let stats_len = input.read_vint().unwrap() as usize;
        let stats_at = input.position();
        input.seek(stats_at + stats_len).unwrap();
        let meta_len = input.read_vint().unwrap() as usize;
        let meta_at = input.position();

        // Decode the stats region. This writer never singleton-run-encodes,
        // so every entry is `vint(docFreq << 1)` plus, for a field with
        // freqs, `vlong(totalTermFreq - docFreq)`.
        //
        // **Only valid for a field that indexes freqs.** A DOCS-only field
        // writes no delta at all (`write_tim_block`'s
        // `index_options != Docs` guard, and `blocktree` reads it back the
        // same way), so the "read a delta if bytes are left" rule below would
        // consume the *next* term's `docFreq` token. Re-encoding is canonical,
        // so the identity round trip would still be byte-identical and would
        // not catch it -- hence the assertion rather than a comment.
        let mut stats_in = SliceInput::new(&tim[stats_at..stats_at + stats_len]);
        let mut pairs: Vec<(i32, i64)> = Vec::new();
        for _ in 0..ent_count {
            let token = stats_in.read_vint().unwrap();
            assert_eq!(
                token & 1,
                0,
                "singleton runs are not written by this writer"
            );
            let doc_freq = (token as u32 >> 1) as i32;
            assert!(
                stats_in.position() < stats_len,
                "patch_tim_stats only handles a field that indexes freqs"
            );
            let delta = stats_in.read_vlong().unwrap();
            pairs.push((doc_freq, i64::from(doc_freq) + delta));
        }

        let mut stats = Vec::new();
        for (i, (doc_freq, total_term_freq)) in pairs.iter_mut().enumerate() {
            edit(i, doc_freq, total_term_freq);
            push_vint(&mut stats, (*doc_freq as u32 as i32) << 1);
            push_raw_vlong(
                &mut stats,
                total_term_freq.wrapping_sub(i64::from(*doc_freq)),
            );
        }

        let mut out = tim[..start].to_vec();
        push_vint(&mut out, code);
        push_raw_vlong(&mut out, code_l);
        out.extend_from_slice(suffix_bytes);
        push_vint(&mut out, sl_code);
        out.extend_from_slice(&tim[sl_at..sl_at + sl_len]);
        push_vint(&mut out, stats.len() as i32);
        out.extend_from_slice(&stats);
        push_vint(&mut out, meta_len as i32);
        out.extend_from_slice(&tim[meta_at..meta_at + meta_len]);
        out.extend_from_slice(&[0u8; codec_util::FOOTER_LENGTH]);
        let n = out.len();
        out[n - codec_util::FOOTER_LENGTH..n - 8]
            .copy_from_slice(&tim[tim.len() - codec_util::FOOTER_LENGTH..tim.len() - 8]);
        repair_checksum(&mut out);
        out
    }

    /// The `.tmd`'s one field-summary record, decoded so a test can change a
    /// single claim.
    struct TmdRecord {
        num_terms: i64,
        sum_total_term_freq: i64,
        sum_doc_freq: i64,
        doc_count: i32,
        min_term: Vec<u8>,
        max_term: Vec<u8>,
        /// The whole `.tim` file's length, which the `.tmd` records so the
        /// reader can checksum it -- so a test that rewrites the `.tim` has
        /// to keep this in step.
        terms_length: i64,
    }

    fn patch_tmd(tmd: &[u8], suffix: &str, edit: impl FnOnce(&mut TmdRecord)) -> Vec<u8> {
        use lucene_store::data_input::{DataInput, SliceInput};
        let start = codec_util::index_header_length(TMD_CODEC, suffix)
            + codec_util::index_header_length(TMD_POSTINGS_CODEC, suffix);
        let body_end = tmd.len() - codec_util::FOOTER_LENGTH;
        let mut input = SliceInput::new(&tmd[..body_end]);
        input.seek(start).unwrap();
        let block_size = input.read_vint().unwrap();
        let num_fields = input.read_vint().unwrap();
        assert_eq!(num_fields, 1, "this helper handles one-field fixtures only");
        let field_number = input.read_vint().unwrap();
        let num_terms = input.read_vlong().unwrap();
        // `read_freq_pair` reads *one* vlong for a DOCS-only field and
        // returns it as both sums, so reading two here would consume
        // `docCount` as `sumDocFreq` and shift the whole record. The
        // re-encoding is canonical, so the identity round trip would still be
        // byte-identical and would not catch it -- assert instead. (c25
        // review, A4.)
        let sum_total_term_freq = input.read_vlong().unwrap();
        let sum_doc_freq = input.read_vlong().unwrap();
        let doc_count = input.read_vint().unwrap();
        let read_bytes = |input: &mut SliceInput<'_>| {
            let len = input.read_vint().unwrap() as usize;
            let at = input.position();
            input.seek(at + len).unwrap();
            tmd[at..at + len].to_vec()
        };
        let min_term = read_bytes(&mut input);
        let max_term = read_bytes(&mut input);
        let index_start = input.read_vlong().unwrap();
        let root_fp = input.read_vlong().unwrap();
        let index_end = input.read_vlong().unwrap();
        // `write_i64` is little-endian in Lucene 9+ (`DataOutput.writeLong`),
        // which is only observable once a test changes the value: a
        // big-endian round trip is byte-identical and silently wrong.
        let index_length = i64::from_le_bytes(tmd[body_end - 16..body_end - 8].try_into().unwrap());
        let terms_length = i64::from_le_bytes(tmd[body_end - 8..body_end].try_into().unwrap());

        let mut record = TmdRecord {
            num_terms,
            sum_total_term_freq,
            sum_doc_freq,
            doc_count,
            min_term,
            max_term,
            terms_length,
        };
        edit(&mut record);

        let mut out = tmd[..start].to_vec();
        push_vint(&mut out, block_size);
        push_vint(&mut out, num_fields);
        push_vint(&mut out, field_number);
        push_raw_vlong(&mut out, record.num_terms);
        push_raw_vlong(&mut out, record.sum_total_term_freq);
        push_raw_vlong(&mut out, record.sum_doc_freq);
        push_vint(&mut out, record.doc_count);
        push_vint(&mut out, record.min_term.len() as i32);
        out.extend_from_slice(&record.min_term);
        push_vint(&mut out, record.max_term.len() as i32);
        out.extend_from_slice(&record.max_term);
        push_raw_vlong(&mut out, index_start);
        push_raw_vlong(&mut out, root_fp);
        push_raw_vlong(&mut out, index_end);
        out.extend_from_slice(&index_length.to_le_bytes());
        out.extend_from_slice(&record.terms_length.to_le_bytes());
        out.extend_from_slice(&[0u8; codec_util::FOOTER_LENGTH]);
        let n = out.len();
        out[n - codec_util::FOOTER_LENGTH..n - 8].copy_from_slice(&tmd[body_end..tmd.len() - 8]);
        repair_checksum(&mut out);
        out
    }

    /// Two terms with freqs, the shape both patchers work on.
    fn two_term_postings() -> Vec<lucene_codecs::postings_writer::TermPostings> {
        use lucene_codecs::postings_writer::TermPostings;
        vec![
            TermPostings {
                term: b"alpha".to_vec(),
                docs: vec![(0, 2), (1, 3)],
                positions: vec![],
                offsets: vec![],
                payloads: vec![],
            },
            TermPostings {
                term: b"beta".to_vec(),
                docs: vec![(1, 1), (2, 4)],
                positions: vec![],
                offsets: vec![],
                payloads: vec![],
            },
        ]
    }

    /// The two patchers must be *identity* transforms when nothing is
    /// changed. Without this the negative controls below would be worthless:
    /// a test that fails because the re-encoding is broken proves nothing
    /// about the check it names.
    #[test]
    fn rewriting_the_tim_and_tmd_unchanged_leaves_the_segment_clean() {
        let dst = tempdir();
        let commit = write_postings_fixture(&dst, &two_term_postings(), 3, 3, None);
        let tim_path = dst.join(format!("_0_{POSTINGS_SUFFIX}.tim"));
        let tmd_path = dst.join(format!("_0_{POSTINGS_SUFFIX}.tmd"));
        let tim = std::fs::read(&tim_path).unwrap();
        let tmd = std::fs::read(&tmd_path).unwrap();
        let round_tripped_tim = patch_tim_stats(&tim, POSTINGS_SUFFIX, |_, _, _| {});
        let round_tripped_tmd = patch_tmd(&tmd, POSTINGS_SUFFIX, |_| {});
        assert_eq!(round_tripped_tim, tim, ".tim round-trip changed the bytes");
        assert_eq!(round_tripped_tmd, tmd, ".tmd round-trip changed the bytes");
        let dir = FsDirectory::open(&dst);
        let result = check_segment(&dir, &commit);
        assert!(result.all_passed(), "{:?}", result.failures());
        std::fs::remove_dir_all(&dst).ok();
    }

    /// `checkFields`' per-term statistic bounds, one arm per case. Every one
    /// of these had never fired: the only way to reach them is a `.tim`
    /// stats region that disagrees with the postings it indexes, and no
    /// writer in this port can be asked to produce one.
    ///
    /// The second case is the interesting one. `.tim` stores
    /// `totalTermFreq - docFreq` as an unsigned vlong, so `totalTermFreq <
    /// docFreq` looks unreachable -- and it would be, except that this port's
    /// `read_vlong` accepts the ten-byte encoding whose top bit lands in bit
    /// 63 (a deliberate divergence from Java's nine-byte cap, recorded in
    /// `data_input.rs`). So the check *is* falsifiable, and this pins the
    /// reasoning: if `read_vlong` ever gains Java's cap, this case becomes
    /// unreachable and the arm should go the way of the five deleted in c25.
    #[test]
    fn every_per_term_statistic_bound_reports_its_own_violation() {
        // (label, edit, the check that must fail, the checks that must not)
        struct Case {
            label: &'static str,
            doc_freq: i32,
            total_term_freq: i64,
            expected_message: &'static str,
        }
        let cases = [
            Case {
                label: "docFreq and totalTermFreq both zero",
                doc_freq: 0,
                total_term_freq: 0,
                expected_message: "totalTermFreq=0 is not > 0",
            },
            Case {
                label: "totalTermFreq below docFreq",
                doc_freq: 2,
                total_term_freq: 1,
                expected_message: "totalTermFreq=1 < docFreq=2",
            },
        ];
        for case in cases {
            let dst = tempdir();
            let commit = write_postings_fixture(&dst, &two_term_postings(), 3, 3, None);
            let tim_path = dst.join(format!("_0_{POSTINGS_SUFFIX}.tim"));
            let tim = std::fs::read(&tim_path).unwrap();
            let patched = patch_tim_stats(&tim, POSTINGS_SUFFIX, |i, df, ttf| {
                if i == 0 {
                    *df = case.doc_freq;
                    *ttf = case.total_term_freq;
                }
            });
            std::fs::write(&tim_path, &patched).unwrap();
            let tmd_path = dst.join(format!("_0_{POSTINGS_SUFFIX}.tmd"));
            let tmd = std::fs::read(&tmd_path).unwrap();
            let new_len = patched.len() as i64;
            std::fs::write(
                &tmd_path,
                patch_tmd(&tmd, POSTINGS_SUFFIX, |r| r.terms_length = new_len),
            )
            .unwrap();

            let dir = FsDirectory::open(&dst);
            let result = check_segment(&dir, &commit);
            let check = result
                .checks
                .iter()
                .find(|c| c.name == "postings.term_stats:body")
                .unwrap_or_else(|| panic!("{}: {:?}", case.label, result.failures()));
            assert!(!check.passed(), "{}: {check:?}", case.label);
            assert!(
                check.message.contains(case.expected_message),
                "{}: expected {:?} in {:?}",
                case.label,
                case.expected_message,
                check.message
            );
            std::fs::remove_dir_all(&dst).ok();
        }

        // `docFreq <= 0` is a separate named check, and a term with
        // `docFreq == 0` is the only input that reaches it.
        let dst = tempdir();
        let commit = write_postings_fixture(&dst, &two_term_postings(), 3, 3, None);
        let tim_path = dst.join(format!("_0_{POSTINGS_SUFFIX}.tim"));
        let tim = std::fs::read(&tim_path).unwrap();
        let patched = patch_tim_stats(&tim, POSTINGS_SUFFIX, |i, df, ttf| {
            if i == 0 {
                *df = 0;
                *ttf = 0;
            }
        });
        std::fs::write(&tim_path, &patched).unwrap();
        let tmd_path = dst.join(format!("_0_{POSTINGS_SUFFIX}.tmd"));
        let tmd = std::fs::read(&tmd_path).unwrap();
        let new_len = patched.len() as i64;
        std::fs::write(
            &tmd_path,
            patch_tmd(&tmd, POSTINGS_SUFFIX, |r| r.terms_length = new_len),
        )
        .unwrap();
        let dir = FsDirectory::open(&dst);
        let result = check_segment(&dir, &commit);
        let check = result
            .checks
            .iter()
            .find(|c| c.name == "postings.doc_freq_positive:body")
            .expect("the docFreq check must run");
        assert!(!check.passed(), "{check:?}");
        assert!(check.message.contains("docFreq=0 is not > 0"), "{check:?}");
        std::fs::remove_dir_all(&dst).ok();
    }

    /// The `totalTermFreq` accumulator's overflow arm (c19's F8: this sum is
    /// `checked_add` precisely so a `.tmd` claiming `i64::MAX` cannot be made
    /// to agree with itself). The arm reports and stops the walk; it had
    /// never run, so the `checked_add` that F8 introduced was itself
    /// unverified.
    #[test]
    fn per_term_total_term_freqs_that_overflow_i64_are_reported_not_summed() {
        let dst = tempdir();
        let commit = write_postings_fixture(&dst, &two_term_postings(), 3, 3, None);
        let tim_path = dst.join(format!("_0_{POSTINGS_SUFFIX}.tim"));
        let tim = std::fs::read(&tim_path).unwrap();
        let patched = patch_tim_stats(&tim, POSTINGS_SUFFIX, |_, _, ttf| *ttf = i64::MAX);
        std::fs::write(&tim_path, &patched).unwrap();
        let tmd_path = dst.join(format!("_0_{POSTINGS_SUFFIX}.tmd"));
        let tmd = std::fs::read(&tmd_path).unwrap();
        let new_len = patched.len() as i64;
        std::fs::write(
            &tmd_path,
            patch_tmd(&tmd, POSTINGS_SUFFIX, |r| r.terms_length = new_len),
        )
        .unwrap();

        let dir = FsDirectory::open(&dst);
        let result = check_segment(&dir, &commit);
        let check = result
            .checks
            .iter()
            .find(|c| c.name == "postings.term_stats:body")
            .expect("the term-statistic check must run");
        assert!(!check.passed(), "{check:?}");
        assert!(
            check
                .message
                .contains("the per-term totalTermFreq values overflow i64"),
            "{check:?}"
        );
        std::fs::remove_dir_all(&dst).ok();
    }

    /// A `.tmd` whose `maxTerm` is smaller than a term the dictionary
    /// actually holds. `minTerm`/`maxTerm` are what a `TermRangeQuery`'s
    /// pruning and `Terms.getMax()` read without ever looking at the
    /// dictionary, so a wrong one silently drops matches -- and the
    /// `maxTerm` half of the pair had never been observed to fire.
    #[test]
    fn a_tmd_max_term_below_the_dictionary_is_caught() {
        let dst = tempdir();
        let commit = write_postings_fixture(&dst, &two_term_postings(), 3, 3, None);
        let tmd_path = dst.join(format!("_0_{POSTINGS_SUFFIX}.tmd"));
        let tmd = std::fs::read(&tmd_path).unwrap();
        std::fs::write(
            &tmd_path,
            patch_tmd(&tmd, POSTINGS_SUFFIX, |r| r.max_term = b"aaa".to_vec()),
        )
        .unwrap();

        let dir = FsDirectory::open(&dst);
        let result = check_segment(&dir, &commit);
        let stats = result
            .checks
            .iter()
            .find(|c| c.name == "postings.term_stats:body")
            .unwrap();
        assert!(!stats.passed(), "{stats:?}");
        assert!(
            stats.message.contains("sorts after .tmd's maxTerm"),
            "{stats:?}"
        );
        let summary = result
            .checks
            .iter()
            .find(|c| c.name == "postings.field_summary:body")
            .unwrap();
        assert!(!summary.passed(), "{summary:?}");
        assert!(summary.message.contains("maxTerm"), "{summary:?}");
        std::fs::remove_dir_all(&dst).ok();
    }

    /// A field whose `.fnm` says it indexes positions (and offsets, and
    /// payloads) in a segment whose file list has no `.pos`/`.pay`.
    ///
    /// `Lucene104PostingsWriter` writes a `.pay` exactly when some field
    /// indexes offsets or stores payloads, and a `.pos` exactly when some
    /// field indexes positions -- so those two files are the only
    /// *independent on-disk witness* the `.fnm`'s positional flags have
    /// (`hasFreqs`/`hasPositions` have none, which is why c25 deleted the
    /// checks that pretended otherwise). Both arms, and the two decode paths
    /// that then decline to read positions rather than panicking, had never
    /// run.
    #[test]
    fn a_positional_field_whose_segment_lists_no_pos_or_pay_is_caught() {
        let dst = tempdir();
        let mut commit = write_positional_postings_fixture(&dst, 300);
        // Re-write the `.si` with `.pos`/`.pay` dropped from its file list,
        // which is what an interrupted flush (or a lost file) looks like.
        let si_bytes = std::fs::read(dst.join("_0.si")).unwrap();
        let mut si = segment_info::parse(&si_bytes, &commit.segment_id).unwrap();
        si.files
            .retain(|f| !f.ends_with(".pos") && !f.ends_with(".pay"));
        std::fs::write(dst.join("_0.si"), segment_info::write(&si, "")).unwrap();
        commit.segment_name = "_0".to_string();

        let dir = FsDirectory::open(&dst);
        let result = check_segment(&dir, &commit);
        let shape = result
            .checks
            .iter()
            .find(|c| c.name == "postings.term_dict_shape:body")
            .expect("the term-dictionary shape check must run");
        assert!(!shape.passed(), "{shape:?}");
        assert!(shape.message.contains("no .pay file"), "{shape:?}");
        assert!(shape.message.contains("no .pos file"), "{shape:?}");
        // The walk still finishes: no positional check is reported, because
        // there are no positions to check, and nothing panics on the way.
        assert!(!result
            .checks
            .iter()
            .any(|c| c.name == "postings.positions_valid:body" && !c.passed()));
        std::fs::remove_dir_all(&dst).ok();
    }

    /// A `.tim` that is listed by the `.si` but absent from the directory:
    /// the postings *file set* fails to open, as opposed to the postings
    /// *readers* failing to parse. Those are two different arms and only the
    /// second had ever run.
    #[test]
    fn a_listed_but_missing_tim_is_reported_by_postings_open() {
        let dst = tempdir();
        let commit = write_postings_fixture(&dst, &two_term_postings(), 3, 3, None);
        std::fs::remove_file(dst.join(format!("_0_{POSTINGS_SUFFIX}.tim"))).unwrap();
        let dir = FsDirectory::open(&dst);
        let result = check_segment(&dir, &commit);
        let check = result
            .checks
            .iter()
            .find(|c| c.name == "postings.open")
            .expect("postings.open must be reported");
        assert!(!check.passed(), "{check:?}");
        // ... and the rest of the segment is still checked.
        assert!(result
            .checks
            .iter()
            .any(|c| c.name == "fnm.open" && c.passed()));
        std::fs::remove_dir_all(&dst).ok();
    }

    /// A points field whose leaves reference documents the segment does not
    /// have. Java's `VerifyPointsVisitor.visit` rejects both the per-point
    /// doc id and the field-level `docCount`; here the two arms sit next to
    /// each other and neither had fired -- so a `.kdd` pointing at documents
    /// past `maxDoc` (every `PointRangeQuery` on it then collecting garbage
    /// doc ids) would have been reported only by the weaker distinct-count
    /// mismatch.
    #[test]
    fn points_leaves_referencing_documents_past_max_doc_are_caught() {
        let points: Vec<(i32, Vec<u8>)> = (0..5)
            .map(|d| (d, vec![0, 0, 0, 0, 0, 0, 0, d as u8]))
            .collect();
        let dst = tempdir();
        // The tree is written for five documents; the `.si` claims two.
        let commit = write_points_fixture(&dst, &points, 2, None);
        let dir = FsDirectory::open(&dst);
        let result = check_segment(&dir, &commit);
        let check = result
            .checks
            .iter()
            .find(|c| c.name == "points.doc_count_matches:loc")
            .expect("the points doc-count check must run");
        assert!(!check.passed(), "{check:?}");
        assert!(check.message.contains("is outside 0..2"), "{check:?}");
        assert!(
            check.message.contains("but the segment has maxDoc=2"),
            "{check:?}"
        );
        std::fs::remove_dir_all(&dst).ok();
    }

    /// A vector segment whose `.si` lists `.vec`/`.vemf`/`.vem`/`.vex` files
    /// that are not in the directory. Every one of the four is opened by a
    /// separate `dir.open(..)` whose failure arm reports under its own
    /// subsystem name -- and those arms are *different* from the ones c19's
    /// file-replacement test drove, which reach the readers' own parse
    /// failures. A file that is simply gone is the more common accident of
    /// the two (an interrupted copy, a partially-restored snapshot).
    #[test]
    fn vector_files_listed_but_absent_are_reported_by_their_own_open_checks() {
        for (victim, expected) in [(".vec", "vectors.open"), (".vex", "hnsw.open")] {
            let dst = copy_fixture("vectors_index");
            std::fs::remove_file(find_file(&dst, victim)).unwrap();
            let dir = FsDirectory::open(&dst);
            let results = check_directory(&dir).expect("the commit itself is untouched");
            let failed = failed_names(&results);
            assert!(
                failed.iter().any(|n| n == expected),
                "a missing {victim} was not reported by {expected}: {failed:?}"
            );
            std::fs::remove_dir_all(&dst).ok();
        }
    }

    /// A field that declares a doc-values skip index in `.fnm` while the
    /// segment's file list has no `.dvs`. Java's `checkDocValueSkipper` is
    /// only ever handed a skipper that exists; this port has to decide what
    /// to do when the `.fnm` promises one and the segment does not carry it,
    /// and it reports rather than skipping -- a decision that had never been
    /// executed.
    #[test]
    fn a_field_declaring_a_skip_index_with_no_dvs_file_is_caught() {
        let dst = copy_fixture("doc_values_skip_index");
        let si_path = find_file(&dst, ".si");
        let commit = {
            let dir = FsDirectory::open(&dst);
            read_commit(&dir)
        };
        let si_bytes = std::fs::read(&si_path).unwrap();
        let mut si = segment_info::parse(&si_bytes, &commit.segment_id).unwrap();
        assert!(si.files.iter().any(|f| f.ends_with(".dvs")));
        si.files.retain(|f| !f.ends_with(".dvs"));
        std::fs::write(&si_path, segment_info::write(&si, "")).unwrap();

        let dir = FsDirectory::open(&dst);
        let results = check_directory(&dir).expect("the commit itself is untouched");
        let failed: Vec<String> = failed_names(&results)
            .into_iter()
            .filter(|n| n.starts_with("doc_values.skipper:"))
            .collect();
        assert!(!failed.is_empty(), "{:?}", failed_names(&results));
        let message = results
            .iter()
            .flat_map(|r| &r.checks)
            .find(|c| c.name == failed[0])
            .map(|c| c.message.clone())
            .unwrap();
        assert!(message.contains(".dvs file"), "{message}");
        std::fs::remove_dir_all(&dst).ok();
    }

    /// A `.si` that declares an index sort over a field its own `.fnm` does
    /// not have. `check_index_sort`'s error arm reports it as a single failed
    /// `sort.docs_in_index_sort_order` rather than panicking on the missing
    /// field or -- worse -- skipping the sort check for a segment that
    /// claims to be sorted. It had never run.
    #[test]
    fn an_index_sort_naming_a_field_the_fnm_lacks_is_caught() {
        let dst = tempdir();
        let commit = write_sorted_dv_fixture(&dst, &[1, 2, 3], sort_asc(), false, 0);
        let si_path = dst.join("_0.si");
        let si_bytes = std::fs::read(&si_path).unwrap();
        let mut si = segment_info::parse(&si_bytes, &commit.segment_id).unwrap();
        for sf in si.index_sort.iter_mut().flatten() {
            sf.field = "no-such-field".to_string();
        }
        std::fs::write(&si_path, segment_info::write(&si, "")).unwrap();

        let dir = FsDirectory::open(&dst);
        let result = check_segment(&dir, &commit);
        let check = result
            .checks
            .iter()
            .find(|c| c.name == "sort.docs_in_index_sort_order")
            .expect("the sort check must run");
        assert!(!check.passed(), "{check:?}");
        assert!(check.message.contains("no-such-field"), "{check:?}");
        std::fs::remove_dir_all(&dst).ok();
    }

    /// The soft-deletes count check's error arm: a segment whose `.fnm`
    /// marks a field as the soft-deletes field but whose `.dvm` carries no
    /// entry for it. Java reaches the same state through
    /// `DocValues.getNumeric` returning an empty iterator, and counts zero;
    /// this port reports it, because a soft-deletes field with no column is
    /// a `softDelCount` nothing can be verified against.
    #[test]
    fn a_soft_deletes_field_with_no_doc_values_entry_is_caught() {
        let dst = tempdir();
        // The fixture writes one NUMERIC column for field number 0 and marks
        // it as the soft-deletes field; renumbering the `.fnm` entry leaves
        // the `.dvm` entry unreachable.
        let commit = write_sorted_dv_fixture(&dst, &[1, 2, 3], None, true, 3);
        let fnm_path = dst.join("_0.fnm");
        let fnm = std::fs::read(&fnm_path).unwrap();
        let mut fields = field_infos::parse(&fnm, &commit.segment_id, "").unwrap();
        fields.fields[0].number = 7;
        std::fs::write(
            &fnm_path,
            field_infos::write(&fields.fields, &commit.segment_id, ""),
        )
        .unwrap();

        let dir = FsDirectory::open(&dst);
        let result = check_segment(&dir, &commit);
        let check = result
            .checks
            .iter()
            .find(|c| c.name == "soft_deletes.count_matches")
            .expect("the soft-deletes check must run");
        assert!(!check.passed(), "{check:?}");
        std::fs::remove_dir_all(&dst).ok();
    }

    /// `hnsw.entry_point_reachable`'s one *failing* arm: an entry point that
    /// reaches nothing but itself on a level that has more than one node.
    ///
    /// Java reports connectedness without ever failing on it (it tolerates
    /// historically-disconnected graphs), and so does this port -- except for
    /// this degenerate case, which is not a quality issue but a graph whose
    /// search can never return more than one document however large `k` is.
    /// That distinction is the whole point of the check and it had never been
    /// executed: the `.vex` corruption sweep cannot reach it, because
    /// emptying the *entry node's* neighbour list specifically is not
    /// something a byte flip does often.
    ///
    /// Driven through this port's own `.vem`/`.vex` writer rather than by
    /// corrupting bytes, for the same reason c19 drove the term-vector arms
    /// at the function: a graph is a shape, not a byte, and this is the only
    /// way to ask for a *specific* one.
    #[test]
    fn an_hnsw_entry_point_that_reaches_only_itself_is_caught() {
        use lucene_codecs::hnsw::{NeighborArray, OnHeapHnswGraph};
        use lucene_codecs::hnsw_vectors::{write_hnsw_vectors, HnswVectorsField};

        const GRAPH_SEG_ID: [u8; ID_LENGTH] = [41u8; ID_LENGTH];
        const GRAPH_SUFFIX: &str = "Lucene99HnswVectorsFormat_0";

        // `connect_entry` decides whether node 0 -- the entry point of a
        // single-level graph -- has any neighbours at all. Everything else
        // about the two graphs is identical, so the only difference the check
        // can be reacting to is the one under test.
        let build = |connect_entry: bool| {
            let mut graph = OnHeapHnswGraph::with_size(16, 3);
            for node in 0..3 {
                graph.add_node(0, node);
            }
            // One level, entry point node 0 -- the shape every small graph
            // this port builds has (`entry_level` defaults to 1, which would
            // describe a second, empty level).
            assert!(graph.try_set_new_entry_node(0, 0));
            let link = |g: &mut OnHeapHnswGraph, from: i32, to: &[i32]| {
                let mut arr = NeighborArray::new(33, true);
                for &n in to {
                    arr.add_in_order(n, 1.0);
                }
                *g.neighbors_mut(0, from) = arr;
            };
            if connect_entry {
                link(&mut graph, 0, &[1, 2]);
            } else {
                link(&mut graph, 0, &[]);
            }
            link(&mut graph, 1, &[2]);
            link(&mut graph, 2, &[1]);
            graph
        };

        let field = field_infos::FieldInfo {
            name: "vec".to_string(),
            number: 0,
            store_term_vectors: false,
            omit_norms: true,
            store_payloads: false,
            soft_deletes_field: false,
            parent_field: false,
            index_options: field_infos::IndexOptions::None,
            doc_values_type: field_infos::DocValuesType::None,
            doc_values_skip_index_type: field_infos::DocValuesSkipIndexType::None,
            doc_values_gen: -1,
            attributes: vec![],
            point_dimension_count: 0,
            point_index_dimension_count: 0,
            point_num_bytes: 0,
            vector_dimension: 4,
            vector_encoding: field_infos::VectorEncoding::Float32,
            vector_similarity_function: field_infos::VectorSimilarityFunction::Euclidean,
        };

        let run = |connect_entry: bool| -> Vec<Check> {
            let graph = build(connect_entry);
            let (vex, vem) = write_hnsw_vectors(
                &[HnswVectorsField {
                    field_number: 0,
                    encoding: field_infos::VectorEncoding::Float32,
                    similarity: field_infos::VectorSimilarityFunction::Euclidean,
                    dimension: 4,
                    count: 3,
                    graph: Some(&graph),
                    m: 16,
                }],
                &GRAPH_SEG_ID,
                GRAPH_SUFFIX,
            )
            .expect("hand-built HNSW graph must write cleanly");
            let dst = tempdir();
            std::fs::write(dst.join(format!("_0_{GRAPH_SUFFIX}.vex")), &vex).unwrap();
            std::fs::write(dst.join(format!("_0_{GRAPH_SUFFIX}.vem")), &vem).unwrap();
            let si = SegmentInfo {
                id: GRAPH_SEG_ID,
                version: segment_info::LuceneVersion {
                    major: 10,
                    minor: 0,
                    bugfix: 0,
                },
                min_version: None,
                doc_count: 3,
                is_compound_file: false,
                has_blocks: false,
                diagnostics: vec![],
                files: vec![
                    format!("_0_{GRAPH_SUFFIX}.vex"),
                    format!("_0_{GRAPH_SUFFIX}.vem"),
                ],
                attributes: vec![],
                index_sort: None,
            };
            let commit = segment_infos::SegmentCommitInfo {
                segment_name: "_0".to_string(),
                segment_id: GRAPH_SEG_ID,
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
            let dir = FsDirectory::open(&dst);
            let mut checks = Vec::new();
            check_hnsw_graphs(&dir, &commit, &si, &[&field], GRAPH_SUFFIX, &mut checks);
            std::fs::remove_dir_all(&dst).ok();
            checks
        };

        // Positive control: the same three nodes, with the entry point wired
        // into them, pass and report the connectedness ratio in the message.
        let good = run(true);
        let reachable = good
            .iter()
            .find(|c| c.name == "hnsw.entry_point_reachable:vec")
            .unwrap_or_else(|| panic!("the connectedness check must run: {good:?}"));
        assert!(reachable.passed(), "{reachable:?}");
        assert_eq!(reachable.message, "3/3 nodes reachable on level 0");
        assert!(good.iter().all(|c| c.passed()), "{good:?}");

        let bad = run(false);
        let reachable = bad
            .iter()
            .find(|c| c.name == "hnsw.entry_point_reachable:vec")
            .expect("the connectedness check must run");
        assert!(!reachable.passed(), "{reachable:?}");
        assert!(
            reachable.message.contains("reaches 1 of 3"),
            "{reachable:?}"
        );
        // ... and nothing else: an isolated entry point is not an
        // out-of-level or repeated neighbour, and must not be reported as one.
        assert!(
            bad.iter()
                .filter(|c| !c.passed())
                .all(|c| c.name == "hnsw.entry_point_reachable:vec"),
            "{bad:?}"
        );
    }

    /// Re-signed body corruption of `.fdt` and `.tvd`: the two "decode every
    /// document, deleted ones included" walks Java's `testStoredFields` and
    /// `testTermVectors` perform. Both walks existed and both had only ever
    /// been observed *succeeding* -- the arm that reports a document that
    /// will not decode had never run for either, which is the half of the
    /// check that does the work.
    ///
    /// c19's file-replacement control reaches these subsystems by swapping a
    /// whole file for another kind, which fails at `open`. A flipped byte in
    /// the middle of a compressed chunk is the failure that gets past `open`
    /// -- and past `file:*`, because the footer is re-signed -- and lands in
    /// the per-document decode. It found two real defects in
    /// `lucene-codecs/src/term_vectors.rs` on its first run: a
    /// `prefixLength` off disk slicing the *previous* term (a panic) and a
    /// chunk's claimed decompressed length sizing a `vec![0u8; n]` (an
    /// **abort**, which no `catch_unwind` at the FFI boundary can intercept).
    /// Both are fixed; this test is what keeps them fixed.
    ///
    /// Measured floors, in the shape c19's controls use: the number that was
    /// measured when this was written is in the failure message, the floor is
    /// ~85% of it, and every corruption is accounted for as caught-here,
    /// caught-elsewhere or accepted so the denominator equals the sweep's own
    /// iteration count.
    #[test]
    fn a_re_signed_body_corruption_is_reported_by_the_per_document_decode() {
        // (fixture, extension, the check that must fail, measured floor)
        let cases: &[(&str, &str, &str, usize)] = &[
            (
                "stored_fields_index",
                ".fdt",
                "stored_fields.every_doc_decodes",
                28,
            ),
            (
                "term_vectors_index",
                ".tvd",
                "term_vectors.every_doc_decodes",
                12,
            ),
        ];
        for (fixture, ext, expected, floor) in cases {
            let dst = copy_fixture(fixture);
            let path = find_file(&dst, ext);
            let original = std::fs::read(&path).unwrap();
            let body_end = original.len() - lucene_store::codec_util::FOOTER_LENGTH;
            let mut caught_here = 0usize;
            let mut caught_by_other = 0usize;
            let mut accepted = 0usize;
            // A fixed uniform sample rather than every byte: the point is the
            // rate, and the sample has to be reproducible run to run.
            let stride = ((body_end - 64) / 40).max(1);
            let mut off = 64;
            while off < body_end {
                let mut bytes = original.clone();
                bytes[off] ^= 0xff;
                repair_checksum(&mut bytes);
                std::fs::write(&path, &bytes).unwrap();
                let dir = FsDirectory::open(&dst);
                let failed = match check_directory(&dir) {
                    Ok(r) => failed_names(&r),
                    Err(e) => vec![format!("commit.unreadable: {e}")],
                };
                if failed.iter().any(|n| n == expected) {
                    caught_here += 1;
                } else if failed.is_empty() {
                    accepted += 1;
                } else {
                    caught_by_other += 1;
                }
                off += stride;
            }
            std::fs::write(&path, &original).unwrap();
            let total = caught_here + caught_by_other + accepted;
            assert!(
                total >= 8,
                "{fixture}: the {ext} sample was only {total} corruptions"
            );
            assert!(
                caught_here >= *floor,
                "{fixture}: only {caught_here} of {total} re-signed {ext} corruptions were \
                 reported by {expected} ({caught_by_other} caught by another check, \
                 {accepted} accepted); the floor is the measured number when this was written"
            );
            if *ext == ".fdt" {
                // As with c19's `.nvm` and `.dvd` rows: nothing else in the
                // module reads `.fdt`, so a corruption this walk misses is a
                // corruption nothing catches. The 14 it does miss are
                // stored-field *values*, which have no second copy anywhere
                // on disk -- a different value is still a valid one, the same
                // phenomenon as the `.tip` row.
                assert_eq!(
                    caught_by_other, 0,
                    "{fixture}: a .fdt corruption was reported by something other than \
                     {expected}"
                );
            }
            let dir = FsDirectory::open(&dst);
            assert!(
                check_directory(&dir)
                    .unwrap()
                    .iter()
                    .all(|r| r.all_passed()),
                "{fixture} must be clean again"
            );
            std::fs::remove_dir_all(&dst).ok();
        }
    }

    /// A document that carries term vectors for a field number the `.fnm`
    /// does not have at all.
    ///
    /// `term_vectors.fields_marked_in_fnm` has two arms -- the field exists
    /// but says `storeTermVectors=false`, and the field is not in `.fnm` --
    /// and c19's control drove only the first. The second is what a `.fnm`
    /// rewritten by a field-infos *update* generation (or a lost `.fnm`
    /// generation) actually looks like, and it is worse than the first: the
    /// vectors are unaddressable rather than merely unexpected.
    #[test]
    fn term_vectors_for_a_field_number_the_fnm_lacks_are_caught() {
        let dst = copy_fixture("term_vectors_index");
        let fnm_path = find_file(&dst, ".fnm");
        let commit = {
            let dir = FsDirectory::open(&dst);
            read_commit(&dir)
        };
        let fnm = std::fs::read(&fnm_path).unwrap();
        let mut fields = field_infos::parse(&fnm, &commit.segment_id, "").unwrap();
        for fi in &mut fields.fields {
            fi.number += 50;
        }
        std::fs::write(
            &fnm_path,
            field_infos::write(&fields.fields, &commit.segment_id, ""),
        )
        .unwrap();

        let dir = FsDirectory::open(&dst);
        let results = check_directory(&dir).expect("the commit itself is untouched");
        let check = results
            .iter()
            .flat_map(|r| &r.checks)
            .find(|c| c.name == "term_vectors.fields_marked_in_fnm")
            .expect("the term-vector field-flag check must run");
        assert!(!check.passed(), "{check:?}");
        assert!(check.message.contains("not in .fnm at all"), "{check:?}");
        std::fs::remove_dir_all(&dst).ok();
    }

    /// A segment whose `.fnm` says a field is not indexed at all while the
    /// segment still carries that field's term dictionary.
    ///
    /// This is the survivable half of Java's "fieldsEnum inconsistent with
    /// fieldInfos" pair (the other half is `x == x` here -- see
    /// `check_postings`' comment). It is what a `.fnm` rewritten by a
    /// field-infos update generation looks like when the update drops a
    /// field's `IndexOptions`: every term in the dictionary becomes
    /// unreachable, because every reader asks whether the field is indexed
    /// before it opens the terms at all.
    #[test]
    fn a_term_dictionary_for_a_field_the_fnm_says_is_not_indexed_is_caught() {
        let dst = tempdir();
        let commit = write_postings_fixture(&dst, &two_term_postings(), 3, 3, None);
        let fnm_path = dst.join("_0.fnm");
        let fnm = std::fs::read(&fnm_path).unwrap();
        let mut fields = field_infos::parse(&fnm, &commit.segment_id, "").unwrap();
        fields.fields[0].index_options = field_infos::IndexOptions::None;
        std::fs::write(
            &fnm_path,
            field_infos::write(&fields.fields, &commit.segment_id, ""),
        )
        .unwrap();

        let dir = FsDirectory::open(&dst);
        let result = check_segment(&dir, &commit);
        let check = result
            .checks
            .iter()
            .find(|c| c.name == "postings.field_in_fnm:body")
            .expect("the field-shape check must run");
        assert!(!check.passed(), "{check:?}");
        assert!(check.message.contains("indexOptions=None"), "{check:?}");
        std::fs::remove_dir_all(&dst).ok();
    }

    /// **The c23 scenario**: a `.fnm` this port wrote and its own parser
    /// rejects. `fnm.open` fails, and before c25 that was the *end* of the
    /// report -- every postings, doc-values, norms, points, vector and
    /// field-flag check simply never appeared, and three term-vector families
    /// appeared as **passes**, because their problem lists were empty for the
    /// one reason that must never read as agreement: nothing had looked.
    ///
    /// A verifier that reports safety over a region it did not examine is
    /// worse than one that omits the check, because the omission is at least
    /// visible in the check list. This pins the fix: every family a failed
    /// prerequisite takes down is named, and `all_passed()` is false.
    #[test]
    fn a_prerequisite_failure_names_every_family_it_takes_down() {
        let dst = copy_fixture("term_vectors_index");
        let fnm_path = find_file(&dst, ".fnm");
        // Truncating the `.fnm` body is the cheapest stand-in for c23's
        // "we wrote it, we reject it": any `.fnm` that will not parse
        // reproduces the state exactly.
        std::fs::write(&fnm_path, b"not a field-infos file").unwrap();

        let dir = FsDirectory::open(&dst);
        let results = check_directory(&dir).expect("the commit itself is untouched");
        let segment = &results[1];

        assert!(!segment.all_passed());
        let failed: Vec<&str> = segment
            .checks
            .iter()
            .filter(|c| c.outcome == Outcome::Failed)
            .map(|c| c.name.as_str())
            .collect();
        assert!(failed.contains(&"fnm.open"), "{failed:?}");

        // The heart of it: the families that could not run are *present* in
        // the result, named, and not passes.
        let skipped: Vec<&str> = segment.skipped().iter().map(|c| c.name.as_str()).collect();
        for family in FAMILIES_BELOW_FNM {
            assert!(
                skipped.contains(family),
                "{family} must be reported as not run: {skipped:?}"
            );
        }
        // And specifically the three that used to be reported as *passing*.
        for family in [
            "term_vectors.fields_marked_in_fnm",
            "term_vectors.self_consistent",
            "term_vectors.match_postings",
        ] {
            let check = segment
                .checks
                .iter()
                .find(|c| c.name == family)
                .unwrap_or_else(|| panic!("{family} must appear in the result at all"));
            assert_eq!(
                check.outcome,
                Outcome::Skipped,
                "{family} must not report a pass over vectors nothing cross-checked"
            );
        }
        // Nothing that did run is claimed to have run against the `.fnm`:
        // `term_vectors.every_doc_decodes` still works (it needs no `.fnm`)
        // and must still be a real pass, so the skip is targeted, not a
        // blanket "give up".
        assert!(segment
            .checks
            .iter()
            .any(|c| c.name == "term_vectors.every_doc_decodes" && c.passed()));

        std::fs::remove_dir_all(&dst).ok();
    }

    /// A segment that **declares an index sort** but carries no doc-values
    /// files: the one skip case where `all_passed()` used to return `true`.
    ///
    /// `fnm.doc_values_vs_files` does not cover it -- a sort field that is
    /// not in `.fnm` at all leaves no field claiming doc values, so that
    /// check passes too. The result was a segment whose declared sort order
    /// (which every merge, every early-terminating query and Lucene's own
    /// `testSort` trust) had been verified by nothing whatsoever, reported as
    /// entirely healthy.
    #[test]
    fn a_declared_index_sort_with_no_doc_values_is_reported_as_unverified() {
        let dst = tempdir();
        let commit = write_sorted_dv_fixture(&dst, &[1, 2, 3], sort_asc(), false, 0);

        // The healthy control: with its doc values, the segment passes and
        // the sort check really ran.
        let dir = FsDirectory::open(&dst);
        let result = check_segment(&dir, &commit);
        assert!(result.all_passed(), "{:?}", result.failures());
        assert!(result
            .checks
            .iter()
            .any(|c| c.name == "sort.docs_in_index_sort_order" && c.passed()));

        // Now take the doc-values files out of the `.si`'s file list *and*
        // the field out of `.fnm`, which is what makes `fnm.doc_values_vs_files`
        // pass and leaves nothing else to notice.
        let si_bytes = std::fs::read(dst.join("_0.si")).unwrap();
        let mut si = segment_info::parse(&si_bytes, &commit.segment_id).unwrap();
        si.files
            .retain(|f| !f.ends_with(".dvm") && !f.ends_with(".dvd"));
        std::fs::write(dst.join("_0.si"), segment_info::write(&si, "")).unwrap();
        std::fs::write(
            dst.join("_0.fnm"),
            field_infos::write(&[], &commit.segment_id, ""),
        )
        .unwrap();

        let dir = FsDirectory::open(&dst);
        let result = check_segment(&dir, &commit);
        let check = result
            .checks
            .iter()
            .find(|c| c.name == "sort.docs_in_index_sort_order")
            .expect("the sort check must appear in the result even when it cannot run");
        assert_eq!(check.outcome, Outcome::Skipped, "{check:?}");
        assert!(
            !result.all_passed(),
            "a segment whose declared sort order was verified by nothing must not report \
             as healthy: {:?}",
            result.checks
        );
        // Nothing else noticed -- which is exactly why the skip has to exist.
        assert_eq!(
            result
                .checks
                .iter()
                .filter(|c| c.outcome == Outcome::Failed)
                .count(),
            0,
            "{:?}",
            result.failures()
        );
        std::fs::remove_dir_all(&dst).ok();
    }

    /// The suppression inventory, pinned so it cannot drift silently.
    ///
    /// Eleven prerequisites in this module can take a family of checks down
    /// with them. The danger is not that they do -- there is genuinely
    /// nothing to check once the file will not open -- it is that before c25
    /// they did it *invisibly*, so a caller could not tell a region that was
    /// examined and found healthy from one nothing looked at. Every one of
    /// them now names its casualties.
    ///
    /// This test exists so that adding a new `*.open` arm without its
    /// `skip_families` call fails the suite rather than quietly reintroducing
    /// the hole. The lists are also what the batch report's
    /// "what each leaves unguarded" table is derived from.
    #[test]
    fn every_prerequisite_names_the_families_it_takes_down() {
        // (prerequisite, the families it suppresses)
        let inventory: &[(&str, &[&str])] = &[
            ("si.open", FAMILIES_BELOW_SI),
            ("fnm.open", FAMILIES_BELOW_FNM),
            ("postings.open", FAMILIES_BELOW_POSTINGS),
            ("liv.open", LIV_FAMILIES),
            ("points.open", POINTS_FAMILIES),
            ("points.field_present:<f>", POINTS_PER_FIELD),
            ("points.decode:<f>", POINTS_PER_FIELD),
            ("vectors.open", VECTOR_FAMILIES),
            ("hnsw.open", HNSW_FAMILIES),
            ("hnsw.open:<f>", HNSW_FAMILIES),
            // These four have their lists inline at the site, because each is
            // a single arm rather than a shared constant.
            (
                "term_vectors.open",
                &[
                    "term_vectors.doc_count_matches_si",
                    "term_vectors.every_doc_decodes",
                    "term_vectors.fields_marked_in_fnm",
                    "term_vectors.self_consistent",
                    "term_vectors.match_postings",
                ],
            ),
            (
                "stored_fields.doc_count_matches_si",
                &["stored_fields.every_doc_decodes"],
            ),
            (
                "doc_values.open",
                &[
                    "doc_values.values_decode",
                    "doc_values.skipper",
                    "doc_values.ords_dense",
                    "doc_values.terms_sorted",
                ],
            ),
            (
                "norms.open",
                &[
                    "norms.entries_name_real_norms_fields",
                    "norms.entry_present",
                    "norms.values_decode",
                    "norms.agree_with_postings",
                ],
            ),
            // Three per-field prerequisites: a `.dvm`/`.nvm`/`.vemf` that
            // parses but has no entry for a field the `.fnm` says has the
            // format. The field's whole column then goes unread while every
            // other field's checks still run.
            (
                "doc_values.entry_present:<f>",
                &[
                    "doc_values.values_decode",
                    "doc_values.ords_dense",
                    "doc_values.terms_sorted",
                ],
            ),
            (
                "norms.entry_present:<f>",
                &["norms.values_decode", "norms.agree_with_postings"],
            ),
            (
                "vectors.field_entry_matches_fnm:<f>",
                &["vectors.values_decode", "vectors.ord_to_doc"],
            ),
        ];

        assert_eq!(
            inventory.len(),
            17,
            "a prerequisite was added or removed without updating the audit"
        );
        let mut families: Vec<&str> = inventory.iter().flat_map(|(_, f)| *f).copied().collect();
        families.sort_unstable();
        families.dedup();
        // 43 names, of which 9 are the `<subsystem>.*` roll-ups the `.si`
        // and `.fnm` failures use: once the file that *names the fields* is
        // gone, the per-field expansion is unknowable, so the roll-up is the
        // honest granularity. The other 34 are concrete check families.
        assert_eq!(
            families.len(),
            43,
            "the set of suppressible check families changed: {families:?}"
        );
        assert_eq!(families.iter().filter(|f| f.ends_with(".*")).count(), 9);
        // No prerequisite may suppress itself -- that would be a check
        // reporting that it did not run because it did not run.
        for (because, suppressed) in inventory {
            assert!(
                !suppressed.contains(because),
                "{because} lists itself among what it takes down"
            );
        }
        // `si.open` is the widest: it is the only one that can leave the
        // whole segment unexamined, which is why it enumerates every family.
        assert_eq!(FAMILIES_BELOW_SI.len(), 16);
        // The three cross-checks are the sharpest, because a cross-check with
        // one side missing is the shape that used to report a pass.
        for cross in [
            "term_vectors.match_postings",
            "norms.agree_with_postings",
            "postings.*",
        ] {
            assert!(FAMILIES_BELOW_POSTINGS.contains(&cross));
        }
    }

    /// Every file the `.si` lists that is *gone* from the directory must be
    /// reported by the check that opens it, by name.
    ///
    /// c25 drove two of these arms (`.tim`, `.vec`/`.vex`). The rest --
    /// `.fdt`/`.fdx`/`.fdm`, `.tvd`/`.tvx`/`.tvm`, `.dvm`/`.dvd`, `.nvd`,
    /// `.kdm`/`.kdi`/`.kdd`, `.tip`/`.tmd`/`.doc`/`.pos` -- had never run,
    /// which mattered because every one of them sits *inside* a
    /// `(|| -> Result<..>{ ... })()` block whose failure arm is the only
    /// thing standing between a missing file and a silently empty walk.
    /// A partially-restored snapshot or an interrupted copy is the common
    /// way to produce one, and it must not read as agreement.
    ///
    /// Each row also asserts what c19's controls call the isolation
    /// property: the *named* check fails. `file:<name>` fails too (the file
    /// is genuinely absent, and `check_files_exist_and_validate` says so
    /// first), which is correct and is not what this test is about.
    #[test]
    fn every_listed_file_that_goes_missing_is_reported_by_its_own_open_check() {
        // (fixture, extension to delete, the check that must fail)
        let cases: &[(&str, &str, &str)] = &[
            (
                "blocktree_index",
                ".fdt",
                "stored_fields.doc_count_matches_si",
            ),
            (
                "blocktree_index",
                ".fdx",
                "stored_fields.doc_count_matches_si",
            ),
            (
                "blocktree_index",
                ".fdm",
                "stored_fields.doc_count_matches_si",
            ),
            ("term_vectors_index", ".tvd", "term_vectors.open"),
            ("term_vectors_index", ".tvx", "term_vectors.open"),
            ("term_vectors_index", ".tvm", "term_vectors.open"),
            ("doc_values_index", ".dvm", "doc_values.open"),
            ("doc_values_index", ".dvd", "doc_values.open"),
            ("doc_values_skip_index", ".dvs", "doc_values.open"),
            ("norms_index", ".nvd", "norms.open"),
            ("norms_index", ".nvm", "norms.open"),
            ("points_index", ".kdm", "points.open"),
            ("points_index", ".kdi", "points.open"),
            ("points_index", ".kdd", "points.open"),
            ("blocktree_index", ".tip", "postings.open"),
            ("blocktree_index", ".tmd", "postings.open"),
            ("blocktree_index", ".doc", "postings.open"),
            ("blocktree_index", ".pos", "postings.open"),
            ("blocktree_index", ".pay", "postings.open"),
            ("vectors_index", ".vemf", "vectors.open"),
            ("vectors_index", ".vem", "hnsw.open"),
        ];
        for (fixture, victim, expected) in cases {
            let dst = copy_fixture(fixture);
            std::fs::remove_file(find_file(&dst, victim)).unwrap();
            let dir = FsDirectory::open(&dst);
            let results = check_directory(&dir).expect("the commit itself is untouched");
            let failed = failed_names(&results);
            assert!(
                failed.iter().any(|n| n == expected),
                "{fixture}: a missing {victim} was not reported by {expected}: {failed:?}"
            );
            std::fs::remove_dir_all(&dst).ok();
        }
    }

    /// The three commit-level degradations that are *not* a corrupt segment:
    /// a directory with no `segments_N` at all, a `.si` file that is listed
    /// but absent (as opposed to present and unparsable, which
    /// `corrupt_si_short_circuits_remaining_checks` covers), and a
    /// `SegmentInfos` whose generation cannot be turned back into a file
    /// name.
    ///
    /// The last is the one worth having: `check_commit`'s result is labelled
    /// with the `segments_N` name, and a negative generation makes
    /// `segments_file_name` return `None`. Without the fallback the label
    /// would be a panic inside the verifier.
    #[test]
    fn commit_level_degradations_are_reported_rather_than_panicking() {
        // (1) No commit at all.
        let empty = tempdir();
        let dir = FsDirectory::open(&empty);
        assert!(
            check_directory(&dir).is_err(),
            "a directory with no segments_N must not report a clean index"
        );
        std::fs::remove_dir_all(&empty).ok();

        // (2) A `.si` that is not there at all.
        let missing = tempdir();
        let dir = FsDirectory::open(&missing);
        let commit = segment_infos::SegmentCommitInfo {
            segment_name: "_0".to_string(),
            segment_id: [7u8; ID_LENGTH],
            codec_name: "Lucene104".to_string(),
            del_gen: -1,
            ..Default::default()
        };
        let result = check_segment(&dir, &commit);
        assert_eq!(result.checks[0].name, "si.open");
        assert!(!result.checks[0].passed(), "{:?}", result.checks[0]);
        std::fs::remove_dir_all(&missing).ok();

        // (3) A commit whose generation has no file name.
        let infos = SegmentInfos {
            id: [3u8; ID_LENGTH],
            generation: -1,
            format_version: 10,
            lucene_version: segment_infos::LuceneVersion {
                major: 10,
                minor: 0,
                bugfix: 0,
            },
            index_created_version_major: 10,
            version: 1,
            counter: 1,
            min_segment_lucene_version: None,
            segments: vec![],
            user_data: vec![],
        };
        let result = check_commit(&infos);
        assert_eq!(result.segment_name, "<invalid generation -1>");
    }

    const BYTE_VEC_SEG_ID: [u8; ID_LENGTH] = [23u8; ID_LENGTH];
    const BYTE_VEC_SUFFIX: &str = "Lucene99HnswVectorsFormat_0";

    /// A **byte**-encoded flat vector segment, written through this port's
    /// own `vectors::write_flat_vectors`.
    ///
    /// Every `.vec` fixture in this repo is `Float32`, so the whole
    /// `VectorEncoding::Byte` arm of `check_vectors` -- its decode loop, its
    /// per-vector length validation and its `ordToDoc` walk -- had never run
    /// against anything at all. That is a *format* gap as much as a coverage
    /// one (c25's carry-over).
    ///
    /// `writer_max_doc` and `si_max_doc` are separate on purpose: the
    /// `ordToDoc` check compares the `.vec`'s doc mapping against the `.si`'s
    /// `maxDoc`, and those are two different files that can disagree.
    /// `fnm_dim` is likewise separate from `dim`, because `.fnm` and `.vemf`
    /// each record the dimension independently.
    #[allow(clippy::too_many_arguments)]
    fn write_byte_vectors_fixture(
        dst: &std::path::Path,
        docs: &[i32],
        dim: i32,
        writer_max_doc: i32,
        si_max_doc: i32,
        fnm_dim: i32,
        extra_files: &[String],
        vemf_edit: impl FnOnce(&mut Vec<u8>),
    ) -> segment_infos::SegmentCommitInfo {
        use lucene_codecs::vectors::{FieldVectorData, FlatVectorsField};

        let values: Vec<u8> = (0..docs.len() as i32)
            .flat_map(|ord| (0..dim).map(move |c| (ord * 16 + c + 1) as u8))
            .collect();
        let (vec, mut vemf) = lucene_codecs::vectors::write_flat_vectors(
            &[FlatVectorsField {
                field_number: 0,
                similarity: field_infos::VectorSimilarityFunction::Euclidean,
                dimension: dim,
                docs: docs.to_vec(),
                values: FieldVectorData::Byte(values),
            }],
            writer_max_doc,
            &BYTE_VEC_SEG_ID,
            BYTE_VEC_SUFFIX,
        )
        .expect("hand-built byte vectors must write cleanly");
        vemf_edit(&mut vemf);

        let field = field_infos::FieldInfo {
            name: "bvec".to_string(),
            number: 0,
            store_term_vectors: false,
            omit_norms: true,
            store_payloads: false,
            soft_deletes_field: false,
            parent_field: false,
            index_options: field_infos::IndexOptions::None,
            doc_values_type: field_infos::DocValuesType::None,
            doc_values_skip_index_type: field_infos::DocValuesSkipIndexType::None,
            doc_values_gen: -1,
            attributes: vec![],
            point_dimension_count: 0,
            point_index_dimension_count: 0,
            point_num_bytes: 0,
            vector_dimension: fnm_dim,
            vector_encoding: field_infos::VectorEncoding::Byte,
            vector_similarity_function: field_infos::VectorSimilarityFunction::Euclidean,
        };
        std::fs::write(
            dst.join("_0.fnm"),
            field_infos::write(&[field], &BYTE_VEC_SEG_ID, ""),
        )
        .unwrap();
        std::fs::write(dst.join(format!("_0_{BYTE_VEC_SUFFIX}.vemf")), &vemf).unwrap();
        std::fs::write(dst.join(format!("_0_{BYTE_VEC_SUFFIX}.vec")), &vec).unwrap();

        let si = SegmentInfo {
            id: BYTE_VEC_SEG_ID,
            version: segment_info::LuceneVersion {
                major: 10,
                minor: 0,
                bugfix: 0,
            },
            min_version: None,
            doc_count: si_max_doc,
            is_compound_file: false,
            has_blocks: false,
            diagnostics: vec![],
            files: [
                "_0.si".to_string(),
                "_0.fnm".to_string(),
                format!("_0_{BYTE_VEC_SUFFIX}.vemf"),
                format!("_0_{BYTE_VEC_SUFFIX}.vec"),
            ]
            .into_iter()
            .chain(extra_files.iter().cloned())
            .collect(),
            attributes: vec![],
            index_sort: None,
        };
        std::fs::write(dst.join("_0.si"), segment_info::write(&si, "")).unwrap();

        segment_infos::SegmentCommitInfo {
            segment_name: "_0".to_string(),
            segment_id: BYTE_VEC_SEG_ID,
            codec_name: "Lucene104".to_string(),
            del_gen: -1,
            ..Default::default()
        }
    }

    /// The `VectorEncoding::Byte` half of `check_vectors`, and the flat
    /// (graph-less) shape of `check_hnsw_graphs`.
    ///
    /// Three cases against one writer, differing in exactly one claim each:
    ///
    /// 1. the healthy control -- every check passes, and `check_hnsw_graphs`
    ///    returns silently because a segment with no `.vem`/`.vex` is the
    ///    flat exhaustive-search format, not a defect;
    /// 2. a `.fnm` whose dimension disagrees with the `.vemf`'s -- the byte
    ///    branch reports it through a *length* comparison where the float
    ///    branch reports it as a decode error, an asymmetry that had never
    ///    been exercised on the byte side;
    /// 3. a `.vec` whose ord->doc map points past the `.si`'s `maxDoc` --
    ///    the flat vector store is addressable as a `DocIdSetIterator` only
    ///    if that map stays inside `0..maxDoc` and strictly increases, and
    ///    the two numbers live in two different files.
    #[test]
    fn a_byte_encoded_vector_field_is_checked_through_the_byte_branch() {
        // (1) Healthy.
        let dst = tempdir();
        let commit = write_byte_vectors_fixture(&dst, &[0, 1, 2], 4, 3, 3, 4, &[], |_| {});
        let dir = FsDirectory::open(&dst);
        let result = check_segment(&dir, &commit);
        assert!(
            result.all_passed(),
            "a byte-encoded vector segment must pass: {:?}",
            result.failures()
        );
        let decode = result
            .checks
            .iter()
            .find(|c| c.name == "vectors.values_decode:bvec")
            .expect("the byte decode walk must run");
        assert_eq!(decode.message, "ok");
        assert!(result
            .checks
            .iter()
            .any(|c| c.name == "vectors.ord_to_doc:bvec" && c.passed()));
        // A flat, graph-less vector segment produces no `hnsw.*` check at
        // all -- the absence is the format, not a failure.
        assert!(
            !result.checks.iter().any(|c| c.name.starts_with("hnsw.")),
            "{:?}",
            result.checks
        );
        std::fs::remove_dir_all(&dst).ok();

        // (2) `.fnm` says dimension 5, `.vemf` says 4.
        let dst = tempdir();
        let commit = write_byte_vectors_fixture(&dst, &[0, 1, 2], 4, 3, 3, 5, &[], |_| {});
        let dir = FsDirectory::open(&dst);
        let result = check_segment(&dir, &commit);
        let decode = result
            .checks
            .iter()
            .find(|c| c.name == "vectors.values_decode:bvec")
            .expect("the byte decode walk must run");
        assert!(!decode.passed(), "{decode:?}");
        assert!(
            decode
                .message
                .contains("vector has 4 bytes, not the field's dimension 5"),
            "{decode:?}"
        );
        // The disagreement itself is reported separately, and the ord->doc
        // map -- which the dimension does not touch -- still passes.
        assert!(result
            .checks
            .iter()
            .any(|c| c.name == "vectors.field_entry_matches_fnm:bvec" && !c.passed()));
        assert!(result
            .checks
            .iter()
            .any(|c| c.name == "vectors.ord_to_doc:bvec" && c.passed()));
        std::fs::remove_dir_all(&dst).ok();

        // (3) The `.vec` maps ordinal 2 to doc 4; the `.si` says maxDoc=3.
        let dst = tempdir();
        let commit = write_byte_vectors_fixture(&dst, &[0, 1, 4], 4, 5, 3, 4, &[], |_| {});
        let dir = FsDirectory::open(&dst);
        let result = check_segment(&dir, &commit);
        let ord = result
            .checks
            .iter()
            .find(|c| c.name == "vectors.ord_to_doc:bvec")
            .expect("the ord->doc walk must run");
        assert!(!ord.passed(), "{ord:?}");
        assert!(ord.message.contains("ord=2 maps to docID=4"), "{ord:?}");
        // ... and nothing else: an out-of-range doc mapping is not a decode
        // failure and must not be reported as one.
        assert!(result
            .checks
            .iter()
            .any(|c| c.name == "vectors.values_decode:bvec" && c.passed()));
        std::fs::remove_dir_all(&dst).ok();
    }

    /// A re-signed sweep over the **sparse** byte-vector segment's `.vemf`,
    /// the metadata file that says where in the `.vec` every region lives.
    ///
    /// This is c19's negative-control shape applied to a file c25's sweeps
    /// never reached with a byte encoding: overwrite one 8-byte field with a
    /// hostile value, re-sign the footer so `file:*`'s CRC cannot claim the
    /// catch, and require that the verifier **reports** rather than panics.
    /// The `ordToDoc` addresses region is the interesting one -- its offset
    /// and length are read off `.vemf` and used to slice the `.vec` with
    /// nothing on the wire relating them to it.
    #[test]
    fn no_re_signed_vemf_field_overwrite_crashes_the_vector_checks() {
        let dst = tempdir();
        let commit = write_byte_vectors_fixture(&dst, &[0, 1, 4], 4, 5, 5, 4, &[], |_| {});
        let vemf_path = dst.join(format!("_0_{BYTE_VEC_SUFFIX}.vemf"));
        let original = std::fs::read(&vemf_path).unwrap();
        let body_end = original.len() - 16;
        let mut total = 0usize;
        let mut caught_by_vectors = 0usize;
        let mut caught_by_other = 0usize;
        let mut accepted = 0usize;
        let mut ord_to_doc_errors = 0usize;
        for off in 0..body_end.saturating_sub(8) {
            for value in [i64::MIN, -1i64, i64::MAX, 1 << 40] {
                total += 1;
                let mut bytes = original.clone();
                bytes[off..off + 8].copy_from_slice(&value.to_be_bytes());
                repair_checksum(&mut bytes);
                std::fs::write(&vemf_path, &bytes).unwrap();
                let dir = FsDirectory::open(&dst);
                let result = check_segment(&dir, &commit);
                let failed: Vec<&str> = result.failures().iter().map(|c| c.name.as_str()).collect();
                let by_vectors = failed
                    .iter()
                    .any(|n| n.starts_with("vectors.") || n.starts_with("hnsw."));
                if by_vectors {
                    caught_by_vectors += 1;
                } else if !failed.is_empty() {
                    caught_by_other += 1;
                } else {
                    accepted += 1;
                }
                if failed.contains(&"vectors.ord_to_doc:bvec") {
                    ord_to_doc_errors += 1;
                }
            }
        }
        std::fs::write(&vemf_path, &original).unwrap();
        std::fs::remove_dir_all(&dst).ok();
        assert_eq!(caught_by_vectors + caught_by_other + accepted, total);
        // Measured when this was written: 566 of 616 caught by the vector
        // checks, **0 by any other check** -- nothing else in the module
        // reads the `.vemf`, so a corruption these miss is one nothing
        // catches. The 50 it accepts are fields whose overwritten value
        // still describes a well-formed region (a similarity ordinal, a
        // block shift that still decodes) -- the `.tip` phenomenon, and the
        // reason `Lucene99FlatVectorsReader` verifies the whole file at
        // open. The floor is ~85% of the measured rate.
        assert!(
            caught_by_vectors >= 481,
            "only {caught_by_vectors} of {total} hostile .vemf field overwrites were rejected \
             by a vector check (was 566); {caught_by_other} elsewhere, {accepted} accepted"
        );
        assert_eq!(
            caught_by_other, 0,
            "the `.vemf` now has a second reader; re-measure the table in c30"
        );
        assert!(
            ord_to_doc_errors > 0,
            "no .vemf overwrite reached the ordToDoc walk's error arm"
        );
    }

    /// A `.vem` that parses cleanly but whose *per-field* graph does not
    /// open: `hnsw.open:<field>`, and the three families it takes down.
    ///
    /// The whole-file `hnsw.open` arm has had a test since c19; this is the
    /// per-field one, and it is the one with a `skip_families` call attached
    /// -- so until now nothing proved that a field whose graph is
    /// unreadable *names* the checks that did not run on it. A segment can
    /// carry several vector fields and lose the graph of exactly one.
    ///
    /// The corruption is a single byte and it is the honest minimum: a field
    /// written with **no** graph records `vectorIndexLength = 0` (which the
    /// reader correctly reads as "flat field, no graph"), and flipping that
    /// vlong to `1` makes the entry claim graph bytes while still recording
    /// `numLevels = 0`, i.e. no node offsets. `OffHeapHnswGraph::new` rejects
    /// exactly that pair. Everything else in the file, the `.vec`/`.vemf`
    /// included, is real writer output.
    #[test]
    fn a_vem_whose_per_field_graph_cannot_open_names_what_it_takes_down() {
        use lucene_codecs::hnsw::OnHeapHnswGraph;
        use lucene_codecs::hnsw_vectors::{write_hnsw_vectors, HnswVectorsField};

        let dst = tempdir();
        let (vex, mut vem) = write_hnsw_vectors(
            &[HnswVectorsField {
                field_number: 0,
                encoding: field_infos::VectorEncoding::Byte,
                similarity: field_infos::VectorSimilarityFunction::Euclidean,
                dimension: 4,
                count: 3,
                graph: None::<&OnHeapHnswGraph>,
                m: 16,
            }],
            &BYTE_VEC_SEG_ID,
            BYTE_VEC_SUFFIX,
        )
        .expect("a graph-less .vem must write cleanly");

        // The entry's tail is `[vlong vectorIndexLength][vint dimension]
        // [i32 size][vint M][vint numLevels]`, then the `-1` field
        // terminator and the footer. With dimension=4 and M=16 every vint
        // here is one byte, so the length's vlong sits eight bytes before
        // the terminator -- asserted, not assumed.
        let terminator = vem.len() - lucene_store::codec_util::FOOTER_LENGTH - 4;
        let len_pos = terminator - 8;
        assert_eq!(
            vem[len_pos], 0,
            "expected a zero vectorIndexLength vlong at {len_pos}: {vem:?}"
        );
        vem[len_pos] = 1;
        repair_checksum(&mut vem);

        std::fs::write(dst.join(format!("_0_{BYTE_VEC_SUFFIX}.vex")), &vex).unwrap();
        std::fs::write(dst.join(format!("_0_{BYTE_VEC_SUFFIX}.vem")), &vem).unwrap();
        let commit = write_byte_vectors_fixture(
            &dst,
            &[0, 1, 2],
            4,
            3,
            3,
            4,
            &[
                format!("_0_{BYTE_VEC_SUFFIX}.vem"),
                format!("_0_{BYTE_VEC_SUFFIX}.vex"),
            ],
            |_| {},
        );

        let dir = FsDirectory::open(&dst);
        let result = check_segment(&dir, &commit);
        let open = result
            .checks
            .iter()
            .find(|c| c.name == "hnsw.open:bvec")
            .unwrap_or_else(|| panic!("hnsw.open:bvec must be reported: {:?}", result.checks));
        assert!(!open.passed(), "{open:?}");
        assert!(
            open.message.contains("graph has data but no node offsets"),
            "{open:?}"
        );
        // The three graph families are named as *not run*, per field.
        let skipped: Vec<&str> = result.skipped().iter().map(|c| c.name.as_str()).collect();
        for family in HNSW_FAMILIES {
            assert!(
                skipped.contains(&format!("{family}:bvec").as_str()),
                "{family}:bvec was not named as skipped: {skipped:?}"
            );
        }
        // The *flat* vector store is untouched and still fully checked --
        // the skip is targeted at the graph, not a blanket give-up.
        assert!(result
            .checks
            .iter()
            .any(|c| c.name == "vectors.values_decode:bvec" && c.passed()));
        assert!(result
            .checks
            .iter()
            .any(|c| c.name == "vectors.ord_to_doc:bvec" && c.passed()));
        std::fs::remove_dir_all(&dst).ok();
    }

    /// The `.liv`-aware arms of the term-vector and norms walks, plus the
    /// `FixedBitSet` bound rule they now go through.
    ///
    /// Three separate places consult `live_docs` while walking a doc id that
    /// came from a *different* file -- `check_term_vectors`' per-document
    /// loop, `check_postings`' per-posting loop and `check_field_norms`'
    /// terms-vs-norms cross-check -- and only the postings one had ever been
    /// reached, because no fixture in the tree carried both deletions and
    /// term vectors or norms. That is exactly the combination a real index
    /// has after its first delete.
    #[test]
    fn a_segment_with_deletions_still_checks_its_vectors_and_norms() {
        use lucene_util::fixed_bit_set::FixedBitSet;

        let dst = copy_fixture("term_vectors_index");
        let mut commit = {
            let dir = FsDirectory::open(&dst);
            read_commit(&dir)
        };
        let max_doc = {
            let dir = FsDirectory::open(&dst);
            let si = open_si(&dir, &commit).expect("the fixture's .si parses");
            si.doc_count
        };
        assert!(max_doc > 1, "the fixture needs more than one document");
        let mut live = FixedBitSet::new(max_doc as usize);
        for doc in 1..max_doc as usize {
            live.set(doc);
        }
        let liv = live_docs::write(&live, &commit.segment_id, 1, 1).unwrap();
        std::fs::write(dst.join(liv_file_name("_0", 1)), &liv).unwrap();
        commit.del_gen = 1;
        commit.del_count = 1;

        let dir = FsDirectory::open(&dst);
        let result = check_segment(&dir, &commit);
        assert!(
            result.all_passed(),
            "deleting document 0 must not make a healthy segment fail: {:?}",
            result.failures()
        );
        // The three walks really ran, and really saw the deletion: the
        // term-vector and postings statistics are live-only, so they must
        // now be *smaller* than on the undeleted control.
        let deleted_stats = result.stats;
        let mut undeleted = commit.clone();
        undeleted.del_gen = -1;
        undeleted.del_count = 0;
        let control = check_segment(&dir, &undeleted);
        assert!(control.all_passed(), "{:?}", control.failures());
        assert!(
            deleted_stats.term_doc_pairs < control.stats.term_doc_pairs,
            "the postings walk did not skip the deleted document: {deleted_stats:?} vs {:?}",
            control.stats
        );
        assert!(
            deleted_stats.term_vector_fields < control.stats.term_vector_fields,
            "the term-vector walk did not skip the deleted document: {deleted_stats:?} vs {:?}",
            control.stats
        );
        std::fs::remove_dir_all(&dst).ok();
    }

    /// The bound rule itself, at the two helpers that implement it: a doc id
    /// past the end of the `.liv` bitset must read as **live**, not index the
    /// bitset.
    ///
    /// `FixedBitSet::get` does `words[index >> 6]` behind a bare
    /// `debug_assert`, so an unbounded id 64 or more past the end panics in a
    /// release build too -- c28's F6, in the shape `docs/arithmetic-gate.md`
    /// now names as a crate rule. Both modes are covered: an *empty* bitset
    /// (`words` is empty, so any index is a real panic) and a short one
    /// (`words` has room, so the read would silently return a ghost bit).
    #[test]
    fn a_doc_id_past_the_end_of_the_live_docs_bitset_is_treated_as_live() {
        use lucene_util::fixed_bit_set::FixedBitSet;

        let empty = FixedBitSet::new(0);
        assert!(is_live(Some(&empty), 0));
        assert!(is_live(Some(&empty), 1_000_000));
        assert!(is_live_at(Some(&empty), usize::MAX));

        // Two bits, one live and one deleted, then the ghost-bit range.
        let mut short = FixedBitSet::new(2);
        short.set(1);
        assert!(!is_live(Some(&short), 0));
        assert!(is_live(Some(&short), 1));
        assert!(is_live(Some(&short), 2), "a ghost bit must not be read");
        assert!(is_live(Some(&short), 63), "a ghost bit must not be read");
        assert!(
            is_live(Some(&short), 64),
            "an index past `words` must not panic"
        );

        // A negative doc id -- `as usize` would sign-extend it to
        // `usize::MAX`, which is the c28 shape exactly.
        assert!(is_live(Some(&short), -1));
        // And no bitset at all means nothing is deleted.
        assert!(is_live(None, i32::MAX));
    }

    /// A re-signed `.kdd` corruption that gets past `points::open` and lands
    /// in `decode_leaves`: `points.decode:<field>`, and the four per-field
    /// families it takes down.
    ///
    /// c25 drove `points.field_present` (a `.kdm` with no entry for the
    /// field) but not this one, which is the more common failure: the tree's
    /// *metadata* is fine and its **leaf blocks** are not. Every check below
    /// it -- the value/bounds check, the doc-count check, the point-count
    /// check -- reads the decoded leaves, so all four have to be named as
    /// not run rather than silently omitted.
    ///
    /// Reported as a rate, in c19's shape: the footer is re-signed over each
    /// corruption so `file:*`'s CRC cannot claim the catch.
    #[test]
    fn a_re_signed_kdd_corruption_is_reported_by_the_points_decode() {
        let points: Vec<(i32, Vec<u8>)> = (0..64)
            .map(|d| (d, (d as u64).to_be_bytes().to_vec()))
            .collect();
        let dst = tempdir();
        let clean = write_points_fixture(&dst, &points, 64, None);
        let original = std::fs::read(dst.join("_0.kdd")).unwrap();
        {
            let dir = FsDirectory::open(&dst);
            let result = check_segment(&dir, &clean);
            assert!(result.all_passed(), "{:?}", result.failures());
        }

        let body_start = 32;
        let body_end = original.len() - lucene_store::codec_util::FOOTER_LENGTH;
        let mut total = 0usize;
        let mut caught_by_decode = 0usize;
        let mut caught_by_other = 0usize;
        let mut accepted = 0usize;
        let mut named_skips = 0usize;
        for off in (body_start..body_end).step_by(3) {
            total += 1;
            let mut bytes = original.clone();
            bytes[off] ^= 0xff;
            repair_checksum(&mut bytes);
            let commit = write_points_fixture(&dst, &points, 64, Some(&bytes));
            let dir = FsDirectory::open(&dst);
            let result = check_segment(&dir, &commit);
            let failed: Vec<&str> = result.failures().iter().map(|c| c.name.as_str()).collect();
            if failed.contains(&"points.decode:loc") {
                caught_by_decode += 1;
                // Every per-field family below the decode must be named.
                if POINTS_PER_FIELD
                    .iter()
                    .all(|f| failed.contains(&format!("{f}:loc").as_str()))
                {
                    named_skips += 1;
                }
            } else if !failed.is_empty() {
                caught_by_other += 1;
            } else {
                accepted += 1;
            }
        }
        std::fs::remove_dir_all(&dst).ok();
        assert_eq!(caught_by_decode + caught_by_other + accepted, total);
        // Measured when this was written: **200 of 200 rejected**, of which
        // only **2** by the decode walk and 198 by another points check.
        // That is the opposite of the `.fdt`/`.dvd`/`.nvm` rows, and it is
        // the finding: a BKD tree carries three independent redundancies
        // over the same bytes (the field's declared min/max packed value,
        // its `docCount` and its `pointCount`), so almost every flipped byte
        // is caught by a *cross-check* rather than by failing to decode. The
        // decode arm is still the only one that can report a leaf that does
        // not parse at all, which is why it needs its own driver.
        assert!(
            caught_by_decode >= 2,
            "only {caught_by_decode} of {total} re-signed .kdd corruptions reached \
             points.decode (was 2); {caught_by_other} elsewhere, {accepted} accepted"
        );
        assert_eq!(
            accepted, 0,
            "a re-signed .kdd corruption was silently accepted"
        );
        assert_eq!(
            named_skips, caught_by_decode,
            "a failed points.decode did not name every family it takes down"
        );
    }

    /// The two points arms c25 left open because no fixture in the tree
    /// could reach them.
    ///
    /// 1. **`points.leaf_bounds_subset_of_field`** is skipped outright for a
    ///    single-index-dimension field (Lucene's `VerifyPointsVisitor` gets
    ///    the same bound from the packed values in that case), and every
    ///    points fixture here was one-dimensional. It matters for a
    ///    multi-dimensional field because a `PointRangeQuery` prunes whole
    ///    subtrees on the *leaf's own* bounding box without ever reading a
    ///    point: a leaf box outside the field box silently drops matches.
    /// 2. **`docCount > pointCount`** -- a field claiming more documents than
    ///    it has points at all. `docCount` is what `IndexSearcher`'s cost
    ///    estimate reads, and the writer computes it from the points, so only
    ///    a hand-edited `.kdm` can say otherwise.
    #[test]
    fn a_multi_dimension_points_field_checks_its_leaf_boxes_and_its_doc_count() {
        // Two index dimensions, one byte each, and enough points to force
        // more than one leaf (so the leaves carry their own bounding boxes).
        let real: Vec<(i32, Vec<u8>)> = (0..1024i32)
            .map(|d| (d, vec![(d / 32) as u8, (d % 32) as u8]))
            .collect();
        // The same doc ids and the same leaf layout, but every value shifted
        // into a range the `.kdm`'s declared box does not contain.
        let wider: Vec<(i32, Vec<u8>)> = real
            .iter()
            .map(|(d, v)| (*d, vec![v[0], v[1].wrapping_add(200)]))
            .collect();

        // Control: a genuinely consistent two-dimensional field passes, and
        // the leaf-bounds check really runs on it (it does not exist at all
        // for a one-dimensional field).
        let dst = tempdir();
        let commit = write_points_fixture_dims(&dst, &real, 1024, 2, 1, None, &|_| {});
        let dir = FsDirectory::open(&dst);
        let result = check_segment(&dir, &commit);
        assert!(result.all_passed(), "{:?}", result.failures());
        assert!(
            result
                .checks
                .iter()
                .any(|c| c.name == "points.leaf_bounds_subset_of_field:loc"),
            "the leaf-bounds check must run for a 2-dimension field: {:?}",
            result.checks
        );
        std::fs::remove_dir_all(&dst).ok();

        // (1) `.kdm`/`.kdi` from `real`, `.kdd` from `wider`.
        let dst = tempdir();
        let (_, _, wider_kdd) = points::write(
            &[lucene_codecs::points::WritePointsField {
                field_number: 5,
                num_dims: 2,
                num_index_dims: 2,
                bytes_per_dim: 1,
                points: wider,
            }],
            points::DEFAULT_MAX_POINTS_IN_LEAF_NODE,
            &POINTS_SEG_ID,
            "",
        )
        .expect("the wider point set must write cleanly");
        let commit = write_points_fixture_dims(&dst, &real, 1024, 2, 1, Some(&wider_kdd), &|_| {});
        let dir = FsDirectory::open(&dst);
        let result = check_segment(&dir, &commit);
        let bounds = result
            .checks
            .iter()
            .find(|c| c.name == "points.leaf_bounds_subset_of_field:loc")
            .expect("the leaf-bounds check must run");
        assert!(!bounds.passed(), "{bounds:?}");
        assert!(
            bounds.message.contains("bounding box for dim 1"),
            "{bounds:?}"
        );
        std::fs::remove_dir_all(&dst).ok();

        // (2) A `.kdm` claiming one more document than it has points.
        let few: Vec<(i32, Vec<u8>)> = (0..5i32).map(|d| (d, vec![d as u8, 0])).collect();
        let dst = tempdir();
        let commit = write_points_fixture_dims(&dst, &few, 8, 2, 1, None, &|kdm| {
            patch_kdm_doc_count(kdm, 6)
        });
        let dir = FsDirectory::open(&dst);
        let result = check_segment(&dir, &commit);
        let docs = result
            .checks
            .iter()
            .find(|c| c.name == "points.doc_count_matches:loc")
            .expect("the doc-count check must run");
        assert!(!docs.passed(), "{docs:?}");
        assert!(
            docs.message
                .contains("claims docCount=6 with only point_count=5"),
            "{docs:?}"
        );
        std::fs::remove_dir_all(&dst).ok();
    }

    /// Rewrites the per-field `docCount` vint in a `.kdm` and re-signs the
    /// footer, so only the semantic invariant can fire.
    ///
    /// The per-field record ends
    /// `[vlong pointCount][vint docCount][vint numIndexBytes][i64
    /// minLeafBlockFP][i64 indexStartPointer]`, and the file then ends with
    /// the `-1` field terminator, the `.kdi`/`.kdd` total lengths and the
    /// codec footer -- so the two vints are found by walking *backwards* from
    /// a fixed offset, a vint's last byte being the one with its high bit
    /// clear.
    fn patch_kdm_doc_count(kdm: &mut [u8], doc_count: u8) {
        assert!(doc_count < 0x80, "single-byte vints only");
        // footer, `.kdd` length, `.kdi` length, `-1`, indexStartPointer,
        // minLeafBlockFP.
        let after_vints = kdm.len() - lucene_store::codec_util::FOOTER_LENGTH - 8 - 8 - 4 - 8 - 8;
        let mut num_index_bytes_start = after_vints - 1;
        assert_eq!(kdm[num_index_bytes_start] & 0x80, 0);
        while num_index_bytes_start > 0 && kdm[num_index_bytes_start - 1] & 0x80 != 0 {
            num_index_bytes_start -= 1;
        }
        let doc_count_pos = num_index_bytes_start - 1;
        assert_eq!(
            kdm[doc_count_pos] & 0x80,
            0,
            "docCount must be a single-byte vint"
        );
        kdm[doc_count_pos] = doc_count;
        repair_checksum(kdm);
    }

    /// A re-signed `.dvm` sweep: the metadata a doc-values column is decoded
    /// *through*, corrupted one byte at a time with the footer re-signed so
    /// only a semantic check can fire.
    ///
    /// This is c25's `patch_dvm` carry-over, taken as a sweep rather than as
    /// a typed editor. The reason a sweep is the right shape here is that the
    /// arms it has to reach are not one invariant but a family: a per-document
    /// decode that fails, an ordinal outside the terms dictionary, a
    /// non-increasing SORTED_SET ordinal run, a dictionary whose size
    /// disagrees with `valueCount`. Every one of them needs a `.dvm` that
    /// *parses* and then disagrees with its `.dvd`, and a byte flip in the
    /// entry region produces exactly that -- it moves an offset, a count or a
    /// bit width without touching the payload those describe.
    ///
    /// c19 measured **0** of 261 `.dvd` corruptions caught by any check other
    /// than `doc_values.*`; the same holds for the `.dvm` and is asserted, so
    /// a skipped `doc_values` family really is the column's only reader.
    #[test]
    fn no_re_signed_dvm_corruption_of_a_doc_values_index_goes_unnoticed() {
        // (fixture, the rejection floor -- ~85% of what was measured when
        // this was written: 315/520, 225/380, 91/218, with **0** caught by
        // any non-`doc_values` check in all three.)
        for (fixture, floor) in [
            ("doc_values_index", 267usize),
            ("multi_valued_dv_index", 191),
            ("sorted_dv_index", 77),
        ] {
            let dst = copy_fixture(fixture);
            let dvm_path = find_file(&dst, ".dvm");
            let original = std::fs::read(&dvm_path).unwrap();
            let body_end = original.len() - lucene_store::codec_util::FOOTER_LENGTH;
            let mut total = 0usize;
            let mut caught_by_doc_values = 0usize;
            let mut caught_by_other = 0usize;
            let mut accepted = 0usize;
            for off in (48..body_end).step_by(2) {
                for mask in [0x01u8, 0x80] {
                    total += 1;
                    let mut bytes = original.clone();
                    bytes[off] ^= mask;
                    repair_checksum(&mut bytes);
                    std::fs::write(&dvm_path, &bytes).unwrap();
                    let dir = FsDirectory::open(&dst);
                    // A corruption the *commit* cannot survive still counts
                    // as rejected, so `total` stays equal to the sweep's own
                    // iteration count.
                    let failed: Vec<String> = match check_directory(&dir) {
                        Ok(results) => failed_names(&results),
                        Err(e) => vec![format!("commit.unreadable: {e}")],
                    };
                    if failed.iter().any(|n| n.starts_with("doc_values.")) {
                        caught_by_doc_values += 1;
                    } else if !failed.is_empty() {
                        caught_by_other += 1;
                    } else {
                        accepted += 1;
                    }
                }
            }
            std::fs::write(&dvm_path, &original).unwrap();
            let dir = FsDirectory::open(&dst);
            assert!(check_directory(&dir)
                .unwrap()
                .iter()
                .all(|r| r.all_passed()));
            std::fs::remove_dir_all(&dst).ok();
            assert_eq!(caught_by_doc_values + caught_by_other + accepted, total);
            assert!(
                caught_by_doc_values >= floor,
                "{fixture}: only {caught_by_doc_values} of {total} re-signed .dvm corruptions \
                 were caught by a doc_values check; {caught_by_other} elsewhere, \
                 {accepted} accepted"
            );
            assert_eq!(
                caught_by_other, 0,
                "{fixture}: the .dvm now has a second reader; re-measure the table in c30"
            );
        }
    }

    /// The error and skip paths of the two checks that read *per-document
    /// sort keys* -- `sort.docs_in_index_sort_order` and
    /// `soft_deletes.count_matches`. Every failure arm of both was unfired:
    /// the only segments that reached them were healthy ones.
    ///
    /// 1. A `SortedSetSortField` sort. `segment_info` can *read* it (that is
    ///    what lets this port open an index Lucene wrote with one), but
    ///    reducing a SORTED_SET column by a `SortedSetSelector` needs an
    ///    ordinal reader `doc_values` does not expose -- so the check must
    ///    report itself **skipped**, naming the field. Failing it would call
    ///    a healthy real-Lucene index corrupt.
    /// 2. A sort whose *kind* disagrees with the field's doc-values type --
    ///    a numeric `SortField` over a SORTED_SET column, which is what
    ///    `DocValues.getNumeric` throws on in Java. A `.si` and a `.fnm`
    ///    disagreeing, and the consequence is the sharp one: the declared
    ///    order that every merge and every early-terminating query trusts
    ///    would be verified by nothing.
    /// 3. A `.dvd` the `.si` lists and the directory does not have -- the
    ///    `dir.open` arm inside `open_doc_values`, distinct from the
    ///    `doc_values.open` one c30 drives elsewhere because this is the
    ///    *second*, independent open of the same pair.
    /// 4. The same for the soft-deletes field, whose `softDelCount` is then a
    ///    claim about data that cannot be read.
    #[test]
    fn a_sort_or_soft_deletes_field_whose_values_cannot_be_read_is_reported() {
        let sorted_set_sort_on_tags = || {
            Some(vec![segment_info::IndexSortField {
                field: "tags".to_string(),
                reverse: false,
                kind: segment_info::IndexSortKind::SortedSet {
                    selector: segment_info::SortedSetSelector::Min,
                    missing: segment_info::StringMissingValue::None,
                },
            }])
        };
        let numeric_sort_on_tags = || {
            Some(vec![segment_info::IndexSortField::long(
                "tags",
                false,
                Some(i64::MAX),
            )])
        };
        let sort_check = |dir: &FsDirectory, commit: &segment_infos::SegmentCommitInfo| {
            check_segment(dir, commit)
                .checks
                .into_iter()
                .find(|c| c.name == "sort.docs_in_index_sort_order")
                .expect("the sort check must run")
        };
        let restamp_sort =
            |dst: &std::path::Path,
             commit: &segment_infos::SegmentCommitInfo,
             sort: Option<Vec<segment_info::IndexSortField>>| {
                let si_path = dst.join("_0.si");
                let mut si =
                    segment_info::parse(&std::fs::read(&si_path).unwrap(), &commit.segment_id)
                        .expect("the fixture's .si parses");
                si.index_sort = sort;
                std::fs::write(&si_path, segment_info::write(&si, "")).unwrap();
            };

        // (1) A SORTED_SET sort: skipped, not failed, and it says which field.
        let dst = tempdir();
        let commit = write_single_valued_sorted_set_fixture(&dst, &[b"a", b"b"]);
        restamp_sort(&dst, &commit, sorted_set_sort_on_tags());
        let dir = FsDirectory::open(&dst);
        let sort = sort_check(&dir, &commit);
        assert!(
            sort.was_skipped(),
            "a real-Lucene SortedSetSortField index must not be called corrupt: {sort:?}"
        );
        assert!(sort.message.contains("SortedSetSortField"), "{sort:?}");
        assert!(sort.message.contains("\"tags\""), "{sort:?}");

        // (2) A numeric sort over the same SORTED_SET column: a real
        // `.si`/`.fnm` disagreement, and a failure.
        restamp_sort(&dst, &commit, numeric_sort_on_tags());
        let dir = FsDirectory::open(&dst);
        let sort = sort_check(&dir, &commit);
        assert!(!sort.passed() && !sort.was_skipped(), "{sort:?}");
        assert!(
            sort.message.contains("doc-values type SortedSet"),
            "{sort:?}"
        );

        // (3) ... and with the `.dvd` gone, the same check reports the open
        // failure instead. (Same directory, one file removed.)
        std::fs::remove_file(dst.join("_0.dvd")).unwrap();
        let dir = FsDirectory::open(&dst);
        let sort = sort_check(&dir, &commit);
        assert!(!sort.passed(), "{sort:?}");
        assert!(
            !sort.message.contains("doc-values type"),
            "the missing file must be reported, not the field type: {sort:?}"
        );
        std::fs::remove_dir_all(&dst).ok();

        // (2) A SORTED_SET soft-deletes field.
        let dst = tempdir();
        let mut commit = write_single_valued_sorted_set_fixture(&dst, &[b"a", b"b"]);
        commit.soft_del_count = 2;
        let fi = field_infos::FieldInfo {
            name: "tags".to_string(),
            number: 0,
            store_term_vectors: false,
            omit_norms: false,
            store_payloads: false,
            soft_deletes_field: true,
            parent_field: false,
            index_options: field_infos::IndexOptions::None,
            doc_values_type: field_infos::DocValuesType::SortedSet,
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
        std::fs::write(
            dst.join("_0.fnm"),
            field_infos::write(std::slice::from_ref(&fi), &commit.segment_id, ""),
        )
        .unwrap();
        let dir = FsDirectory::open(&dst);
        let result = check_segment(&dir, &commit);
        let soft = result
            .checks
            .iter()
            .find(|c| c.name == "soft_deletes.count_matches")
            .expect("the soft-deletes check must run");
        assert!(!soft.passed(), "{soft:?}");
        assert!(
            soft.message
                .contains("soft-deletes field \"tags\": field has doc-values type SortedSet"),
            "{soft:?}"
        );
        std::fs::remove_dir_all(&dst).ok();
    }

    /// A re-signed `.dvm` sweep over the **hand-built single-valued
    /// SORTED_SET** fixture -- the second SORTED_SET code path
    /// (`Lucene90DocValuesConsumer.addSortedSetField` collapses a
    /// one-value-per-doc set to a plain `SortedEntry`), whose ordinal
    /// bookkeeping is separate code from the multi-valued branch and which
    /// no Java-written fixture in this repo produces.
    ///
    /// The arms this reaches are the ordinal-space ones: an ordinal outside
    /// the terms dictionary, a document whose ordinal will not decode, and a
    /// dictionary whose decoded size disagrees with `valueCount`. Those are
    /// the checks that catch a dictionary nothing can ever match -- a
    /// `SortedDocValues` range query or an index sort reads the ordinal
    /// space without ever looking at the terms.
    #[test]
    fn no_re_signed_dvm_corruption_of_a_single_valued_sorted_set_goes_unnoticed() {
        let dst = tempdir();
        let commit = write_single_valued_sorted_set_fixture(
            &dst,
            &[b"alpha", b"beta", b"delta", b"gamma", b"omega"],
        );
        let dvm_path = dst.join("_0.dvm");
        let original = std::fs::read(&dvm_path).unwrap();
        {
            let dir = FsDirectory::open(&dst);
            let result = check_segment(&dir, &commit);
            assert!(result.all_passed(), "{:?}", result.failures());
        }

        let body_end = original.len() - lucene_store::codec_util::FOOTER_LENGTH;
        let mut total = 0usize;
        let mut caught_by_doc_values = 0usize;
        let mut caught_by_other = 0usize;
        let mut accepted = 0usize;
        let mut families: std::collections::BTreeSet<String> = Default::default();
        for off in 48..body_end {
            for mask in [0x01u8, 0x80, 0xff] {
                total += 1;
                let mut bytes = original.clone();
                bytes[off] ^= mask;
                repair_checksum(&mut bytes);
                std::fs::write(&dvm_path, &bytes).unwrap();
                let dir = FsDirectory::open(&dst);
                let result = check_segment(&dir, &commit);
                let failed: Vec<&str> = result.failures().iter().map(|c| c.name.as_str()).collect();
                if failed.iter().any(|n| n.starts_with("doc_values.")) {
                    caught_by_doc_values += 1;
                    families.extend(
                        failed
                            .iter()
                            .filter(|n| n.starts_with("doc_values."))
                            .map(|n| (*n).to_string()),
                    );
                } else if !failed.is_empty() {
                    caught_by_other += 1;
                } else {
                    accepted += 1;
                }
            }
        }
        std::fs::write(&dvm_path, &original).unwrap();
        std::fs::remove_dir_all(&dst).ok();
        assert_eq!(caught_by_doc_values + caught_by_other + accepted, total);
        // Measured when this was written: 256 of 624 caught by a doc-values
        // check, **0** by anything else. The 368 it accepts are the `.dvm`'s
        // padding and its many bit-width/offset fields whose flipped value
        // still describes a self-consistent column -- pure metadata with no
        // second copy, which is why `Lucene90DocValuesProducer` verifies the
        // whole file at open and why `file:*` does a full CRC. Floor at
        // ~85% of the measured rate.
        assert!(
            caught_by_doc_values >= 217,
            "only {caught_by_doc_values} of {total} re-signed .dvm corruptions were caught by \
             a doc_values check (was 256); {caught_by_other} elsewhere, {accepted} accepted"
        );
        assert_eq!(caught_by_other, 0);
        // The sweep must reach the *ordinal-space* checks, not just the
        // per-document decode -- those are the ones with no second reader.
        for family in [
            "doc_values.values_decode:tags",
            "doc_values.ords_dense:tags",
            "doc_values.terms_sorted:tags",
        ] {
            assert!(
                families.contains(family),
                "{family} was never reached: {families:?}"
            );
        }
    }

    /// The remaining failure arms of the two sort-key readers: a `.liv` the
    /// soft-deletes count needs and cannot open or cannot parse, a `.dvd` the
    /// `.si` does not list at all, and a `.dvd` whose per-document numeric
    /// values will not decode.
    ///
    /// `check_soft_deletes` opens the `.liv` a **second** time, independently
    /// of `check_live_docs`, because the count it computes is over *live*
    /// documents -- so its own open and parse failures are separate arms, and
    /// a soft-delete count computed as if nothing were deleted is exactly the
    /// wrong answer (it is the number `IndexWriter` uses to decide when a
    /// segment is fully deleted and can be dropped).
    #[test]
    fn the_sort_key_readers_report_every_way_their_input_can_fail() {
        // (1) A soft-deletes segment whose `.liv` is not there.
        let dst = tempdir();
        let mut commit = write_sorted_dv_fixture(&dst, &[1, 2, 3, 4], sort_asc(), true, 3);
        commit.del_gen = 1;
        commit.del_count = 1;
        let dir = FsDirectory::open(&dst);
        let result = check_segment(&dir, &commit);
        let soft = result
            .checks
            .iter()
            .find(|c| c.name == "soft_deletes.count_matches")
            .expect("the soft-deletes check must run");
        assert!(!soft.passed(), "{soft:?}");

        // (2) ... and with a `.liv` that is present but will not parse.
        std::fs::write(dst.join(liv_file_name("_0", 1)), b"not a .liv file").unwrap();
        let dir = FsDirectory::open(&dst);
        let result = check_segment(&dir, &commit);
        let soft = result
            .checks
            .iter()
            .find(|c| c.name == "soft_deletes.count_matches")
            .expect("the soft-deletes check must run");
        assert!(!soft.passed(), "{soft:?}");
        std::fs::remove_dir_all(&dst).ok();

        // (3) A `.si` that declares an index sort and lists a `.dvm` but no
        // `.dvd`: nothing can be read, so the sort is *unverified* -- the
        // c25 hole, reached through a different door (a half-listed file set
        // rather than no doc-values files at all).
        let dst = tempdir();
        let commit = write_sorted_dv_fixture(&dst, &[1, 2, 3, 4], sort_asc(), false, 0);
        let si_path = dst.join("_0.si");
        let mut si = segment_info::parse(&std::fs::read(&si_path).unwrap(), &commit.segment_id)
            .expect("the fixture's .si parses");
        si.files.retain(|f| !f.ends_with(".dvd"));
        std::fs::write(&si_path, segment_info::write(&si, "")).unwrap();
        let dir = FsDirectory::open(&dst);
        let result = check_segment(&dir, &commit);
        let sort = result
            .checks
            .iter()
            .find(|c| c.name == "sort.docs_in_index_sort_order")
            .expect("the sort check must be reported, not omitted");
        assert!(sort.was_skipped(), "{sort:?}");
        std::fs::remove_dir_all(&dst).ok();

        // (4) A `.dvd` that is *shorter* than the `.dvm` says its column is.
        // `open_doc_values` parses only the `.dvm`, so the pair opens and the
        // failure lands in `sort_key_values`' per-document read -- the arm
        // that turns "the sort keys cannot be read" into a reported check
        // rather than a silent pass over an unverified order.
        let dst = tempdir();
        let commit = write_sorted_dv_fixture(&dst, &[1, 2, 3, 4], sort_asc(), false, 0);
        let dvd_path = dst.join("_0.dvd");
        let original = std::fs::read(&dvd_path).unwrap();
        std::fs::write(&dvd_path, &original[..original.len() / 2]).unwrap();
        let dir = FsDirectory::open(&dst);
        let result = check_segment(&dir, &commit);
        let sort = result
            .checks
            .iter()
            .find(|c| c.name == "sort.docs_in_index_sort_order")
            .expect("the sort check must run");
        assert!(!sort.passed(), "{sort:?}");
        assert!(
            sort.message.contains("sort field \"ts\""),
            "the failure must name the sort field: {sort:?}"
        );
        std::fs::remove_dir_all(&dst).ok();
    }
}
