//! Port of `org.apache.lucene.index.TieredMergePolicy` -- **the decision
//! function only**: given a segment list's stats, which segments (if any)
//! should be merged together next. This module does not execute merges (see
//! [`crate::merge`] for that -- its own module doc explicitly calls out "not
//! a merge policy" as out of scope, which is exactly the gap this module
//! fills) and does not run anything in a background thread (real Lucene's
//! `MergeScheduler`/`ConcurrentMergeScheduler` is out of scope here too --
//! this is a pure, synchronous, side-effect-free function of its input).
//!
//! # Fidelity
//!
//! This is a **faithful, line-by-line port** of `TieredMergePolicy`'s three
//! decision entry points and their shared machinery:
//!
//! | Java | Rust |
//! |---|---|
//! | `getSortedBySegmentSize` | [`sorted_by_segment_size`] |
//! | `findMerges` | [`find_merges`] / [`find_merges_excluding`] |
//! | `doFindMerges` | [`do_find_merges`] |
//! | `score` | [`merge_score`] |
//! | `findForcedMerges` | [`find_forced_merges`] |
//! | `findForcedDeletesMerges` | [`find_forced_delete_merges`] |
//! | `getMaxAllowedDocs` | [`max_allowed_docs`] |
//! | `MergePolicy.size` (pro-rated bytes) | [`prorated_size`] |
//! | `floorSize` | [`floor_size`] |
//!
//! That means the real budget computation (the level-by-level
//! `allowedSegCount` walk), the real `MergeScore` (`skew *
//! totAfterMergeBytes^0.05 * nonDelRatio^2`), the real "don't rewrite the
//! biggest input into something barely bigger" 1.5x guard, the real
//! `deletesPctAllowed` accounting, the real `targetSearchConcurrency`
//! handling, and the real bin-packing of `findForcedMerges` are all here.
//!
//! # Size units
//!
//! Real Lucene distinguishes two byte quantities per segment and this port
//! does too:
//! - `SegmentCommitInfo.sizeInBytes()` -- the **raw** sum of the segment's
//!   file lengths. That is [`SegmentStat::size_bytes`], filled either by
//!   [`commit_byte_size`] (byte-accurate, `.liv` included, needs a
//!   [`Directory`]; [`segment_byte_size`] for a segment no commit names
//!   yet) or by a caller's own
//!   doc-count approximation ([`SegmentStat::from_segment_info`]).
//! - `MergePolicy.size(info, ctx)` -- the raw size **pro-rated by the live-doc
//!   fraction**, `bytes * (1 - delCount/maxDoc)`. That is [`prorated_size`],
//!   derived here rather than stored, and it is what every size comparison in
//!   the algorithm below uses (exactly as in Java, where
//!   `SegmentSizeAndDocs.sizeInBytes` is the pro-rated figure and only
//!   `score()`'s `totBeforeMergeBytes` uses the raw one).
//!
//! The algorithm only requires these be monotonic, comparable size units, so
//! a caller using the doc-count approximation still gets sensible behaviour
//! -- but the byte-denominated defaults (`maxMergedSegmentMB`,
//! `floorSegmentMB`) are then meaningless as doc counts and should be
//! overridden.
//!
//! # Deliberate scope boundaries (documented, not silently missing)
//!
//! - **The free functions take no `MergeContext`.** They read `del_count`
//!   straight off [`SegmentStat`] and take the currently-merging set as an
//!   argument ([`find_merges_excluding`]). The [`MergePolicy`] trait
//!   ([`api`]) is the context-driven form: [`TieredMergePolicy`] fills each
//!   stat's `del_count` from `MergeContext.numDeletesToMerge`, and adds
//!   `findFullFlushMerges` (inputs below `floorSegmentBytes`),
//!   `useCompoundFile` (`noCFSRatio` 0.1) and `isMerged`, which
//!   `findForcedMerges`' "already merged" bail-out consults through
//!   [`forced_merges_full`] together with Java's `segmentsToMerge` "original"
//!   flags. [`find_forced_merges`] keeps the old shape: every segment
//!   original, none merging, a delete-free segment merged.
//! - **Other policies** live beside this one: [`log`] (`LogMergePolicy`,
//!   `LogByteSizeMergePolicy`, `LogDocMergePolicy`), [`temporal`]
//!   (`TemporalMergePolicy`) and [`filter`] (`FilterMergePolicy`,
//!   `OneMergeWrappingMergePolicy`).
//! - **`segmentsPerTier` is an integer here**, not a `double`. Real Lucene
//!   validates `>= 2.0`; this port clamps to 2 for the same reason (the
//!   level walk below cannot terminate otherwise).
//! - **`max_merge_at_once` is Java's own `maxMergeAtOnce` knob** (default 10),
//!   from which Java derives `mergeFactor = (int) Math.min(maxMergeAtOnce,
//!   segsPerTier)`. It is both a *hard* cap on how many segments one merge may
//!   contain and (through `mergeFactor`) the *soft* cap that a below-floor
//!   merge is allowed to exceed.
//! - **Degenerate inputs where Java would produce `NaN`** are given a neutral
//!   value instead, since this port's `size_bytes` is unit-agnostic and a
//!   zero-`doc_count` or zero-byte segment is constructible here where real
//!   Lucene never produces one: `totalDelPct` and `segDelPct` become `0.0`
//!   rather than `100 * 0.0 / 0`, `nonDelRatio` becomes `1.0` rather than
//!   `0.0 / 0.0`, and `levelSize` is floored at 1 rather than dividing by
//!   zero. Java's `NaN` fails every `<=` comparison, so the substitutions are
//!   the conservative direction (exclude nothing, win nothing).

use std::collections::HashSet;

use crate::segment_info::SegmentInfo;

pub mod api;
pub mod filter;
pub mod log;
pub mod merge_on_flush;
pub mod temporal;
pub mod upgrade;

pub use api::{
    BasicMergeContext, CompoundFileSettings, MergeContext, MergeHooks, MergePolicy, MergeSegment,
    MergeSpecification, MergeTrigger, NoMergePolicy, OneMerge, PreparedMerge, TieredMergePolicy,
};
use lucene_store::directory::Directory;
pub use merge_on_flush::MergeOnFlushMergePolicy;

/// The stats [`find_merges`] needs about one segment -- the port of Java's
/// `TieredMergePolicy.SegmentSizeAndDocs` input. Deliberately not
/// `SegmentCommitInfo` directly: `SegmentCommitInfo` (`segment_infos.rs`)
/// carries `del_count` but not doc count or byte size -- those live in the
/// separate per-segment `.si` file (`SegmentInfo`, `segment_info.rs`). A
/// caller that has both parsed already builds a `SegmentStat` from them (see
/// [`SegmentStat::from_segment_info`]); a caller with only a `Directory` can
/// use [`commit_byte_size`] to fill in `size_bytes`.
#[derive(Debug, Clone, PartialEq)]
pub struct SegmentStat {
    pub name: String,
    /// Total (including deleted) doc count -- real Lucene's `maxDoc`.
    pub doc_count: i32,
    pub del_count: i32,
    /// **Raw** size in whatever unit the caller chose -- real Lucene's
    /// `SegmentCommitInfo.sizeInBytes()`, i.e. *not* pro-rated by deletes.
    /// The pro-rated figure the algorithm actually compares against
    /// (`MergePolicy.size`) is derived from this by [`prorated_size`]. Real
    /// on-disk bytes if obtained via [`commit_byte_size`], or a doc-count
    /// approximation otherwise.
    pub size_bytes: u64,
}

impl SegmentStat {
    /// Approximates `size_bytes` from `doc_count` (one unit per doc) when no
    /// `Directory` is available to compute real on-disk size. Documented
    /// approximation, not a byte-accurate figure -- see this module's doc
    /// comment.
    pub fn from_segment_info(name: impl Into<String>, info: &SegmentInfo, del_count: i32) -> Self {
        SegmentStat {
            name: name.into(),
            doc_count: info.doc_count,
            del_count,
            size_bytes: info.doc_count.max(0) as u64,
        }
    }

    fn del_ratio(&self) -> f64 {
        if self.doc_count <= 0 {
            0.0
        } else {
            self.del_count as f64 / self.doc_count as f64
        }
    }
}

/// Real Lucene's `MergePolicy.size(SegmentCommitInfo, MergeContext)`: the
/// segment's raw byte size pro-rated by the fraction of live docs,
/// `(long) (byteSize * (1.0 - delRatio))`, or the raw size unchanged when
/// `maxDoc <= 0`. This -- not [`SegmentStat::size_bytes`] -- is what every
/// size comparison in [`find_merges`]/[`do_find_merges`] uses, matching Java
/// where `SegmentSizeAndDocs.sizeInBytes` holds exactly this value.
pub fn prorated_size(stat: &SegmentStat) -> u64 {
    if stat.doc_count <= 0 {
        stat.size_bytes
    } else {
        (stat.size_bytes as f64 * (1.0 - stat.del_ratio())) as u64
    }
}

/// Sums real on-disk file lengths for a segment's files (`SegmentInfo::files`)
/// via the existing [`Directory`] trait -- the honest, byte-accurate way to
/// fill [`SegmentStat::size_bytes`] when a `Directory` is available. Missing
/// files are skipped (matches "some auxiliary files are optional" rather
/// than erroring the whole computation over one absent file).
pub fn segment_byte_size(dir: &dyn Directory, info: &SegmentInfo) -> u64 {
    files_byte_size(dir, &info.files)
}

/// Port of `SegmentCommitInfo.sizeInBytes()`: [`segment_byte_size`] over
/// [`SegmentCommitInfo::files`](crate::segment_infos::SegmentCommitInfo::files),
/// so the commit-only files -- the current `.liv`, the generational field
/// infos and doc-values update files -- count too. A merge policy sizes a
/// committed segment with this: leaving the `.liv` out once ordered two
/// nearly equal segments the other way round from Java's
/// `TieredMergePolicy`, and so merged their documents in a different order.
pub fn commit_byte_size(
    dir: &dyn Directory,
    commit: &crate::segment_infos::SegmentCommitInfo,
    info: &SegmentInfo,
) -> u64 {
    files_byte_size(dir, &commit.files(&info.files))
}

fn files_byte_size(dir: &dyn Directory, files: &[String]) -> u64 {
    let mut total = 0u64;
    for file in files {
        // `file_length`, not `open(..).len()`: `FsDirectory::open` reads the
        // whole file, and this runs over every segment on every merge pass.
        if let Ok(len) = dir.file_length(file) {
            // Saturating rather than a bare `+`: `info.files` is parsed off
            // the `.si`, so the *number* of terms is file-controlled even
            // though each term is the real byte length of a file that opened.
            // Saturation is the honest failure here -- it can only make a
            // segment look larger, and a larger segment is one this policy
            // excludes from merging, never one it merges wrongly.
            total = total.saturating_add(len);
        }
    }
    total
}

/// Tunables for [`find_merges`]/[`find_forced_merges`]/
/// [`find_forced_delete_merges`]. Every default is real
/// `TieredMergePolicy`'s own default as of Lucene 10.5.0.
#[derive(Debug, Clone, PartialEq)]
pub struct MergePolicyConfig {
    /// Real Lucene's `maxMergeAtOnce` (`setMaxMergeAtOnce`, default 10): a
    /// **hard** cap on how many segments one natural/force-merge-deletes
    /// merge may contain. Java also derives its soft cap from it,
    /// `mergeFactor = (int) Math.min(maxMergeAtOnce, segsPerTier)` -- see
    /// [`MergePolicyConfig::merge_factor`]. The soft cap is what a merge is
    /// allowed to exceed while it is still under `floor_segment_size` (real
    /// Lucene deliberately packs more than `mergeFactor` tiny segments into
    /// one merge); this hard cap is never exceeded.
    pub max_merge_at_once: usize,
    /// Real Lucene's `segsPerTier` (`setSegmentsPerTier`, default 8.0).
    /// Clamped to `>= 2` (Java rejects anything smaller outright).
    pub segments_per_tier: f64,
    /// Real Lucene's `maxMergedSegmentBytes` (`setMaxMergedSegmentMB`,
    /// default 5 GB). Note the exclusion rule is `size > this / 2`, not
    /// `size >= this` -- see [`find_merges_excluding`].
    pub max_merged_segment_size: u64,
    /// The exponent real Lucene hardcodes as `2` in `score()`'s
    /// `mergeScore *= Math.pow(nonDelRatio, 2)` term -- how strongly a merge
    /// that reclaims deletes is favoured. Default `2.0` reproduces Java
    /// exactly; `0.0` disables reclaim-weighting (pure skew/size scoring).
    pub reclaim_weight: f64,
    /// Real Lucene's `floorSegmentBytes` (`setFloorSegmentMB`, default
    /// 16MB): segments smaller than this are rounded up to it for scoring
    /// (`floorSize`), and a merge is allowed to exceed `max_merge_at_once`
    /// inputs while it is still below this size.
    pub floor_segment_size: u64,
    /// Real Lucene's `forceMergeDeletesPctAllowed`
    /// (`setForceMergeDeletesPctAllowed`, default 10.0): only segments whose
    /// deleted-doc percentage (`100.0 * del_count / doc_count`) *strictly
    /// exceeds* this are eligible for [`find_forced_delete_merges`].
    /// A percentage (`0.0..=100.0`), matching Java's own unit.
    pub force_merge_deletes_pct_allowed: f64,
    /// Real Lucene's `deletesPctAllowed` (`setDeletesPctAllowed`, default
    /// 20.0): the maximum percentage of the doc-id space allowed to be
    /// deleted docs before natural merging kicks in purely to reclaim them.
    /// Also gates which over-half-max-size segments are excluded from
    /// merging, and the `score()` bypass of the 1.5x growth guard.
    pub deletes_pct_allowed: f64,
    /// Real Lucene's `targetSearchConcurrency` (`setTargetSearchConcurrency`,
    /// default 1): prevents creating segments bigger than
    /// `maxDoc / targetSearchConcurrency`, so search work can be split into
    /// that many similarly-sized slices.
    pub target_search_concurrency: usize,
    /// `MergePolicy.keepFullyDeletedSegment(IOSupplier<CodecReader>)`: keep a
    /// segment whose every document has been hard-deleted instead of dropping
    /// it when the deletes are applied.
    ///
    /// Java's base implementation returns `false` and `TieredMergePolicy` does
    /// not override it, so `false` is the default here too, matching
    /// `IndexWriterConfig`'s own default policy. The one override in the
    /// Lucene tree is `SoftDeletesRetentionMergePolicy`, which returns `true`
    /// when its retention query still matches a document the *hard* delete
    /// count has written off -- so a writer that keeps soft-deleted documents
    /// alive for a retention window must set this, or the drop discards
    /// documents its policy promised to retain.
    ///
    /// **Reduced to a flag on purpose.** Java's hook takes a `CodecReader`
    /// supplier and may run a query over the segment; this port has no
    /// `MergePolicy` trait and no reader supplier to hand it (see this
    /// module's doc comment), so the honest port of a hook whose two in-tree
    /// implementations are "always false" and "true iff a retention query
    /// matches" is a boolean the caller sets. When a reader-aware policy
    /// arrives this becomes its method.
    pub keep_fully_deleted_segments: bool,
}

impl Default for MergePolicyConfig {
    fn default() -> Self {
        MergePolicyConfig {
            // maxMergeAtOnce = 10 (mergeFactor is then min(10, 8.0) = 8).
            max_merge_at_once: 10,
            // segsPerTier = 8.0
            segments_per_tier: 8.0,
            // maxMergedSegmentBytes = 5 * 1024 * 1024 * 1024L (5 GiB, NOT
            // 5000 MiB -- setMaxMergedSegmentMB(5000) would be a different
            // number).
            max_merged_segment_size: 5 * 1024 * 1024 * 1024,
            // score()'s hardcoded `Math.pow(nonDelRatio, 2)` exponent.
            reclaim_weight: 2.0,
            // floorSegmentBytes = 16 * 1024 * 1024L
            floor_segment_size: 16 * 1024 * 1024,
            // forceMergeDeletesPctAllowed = 10.0
            force_merge_deletes_pct_allowed: 10.0,
            // deletesPctAllowed = 20.0
            deletes_pct_allowed: 20.0,
            // targetSearchConcurrency = 1
            target_search_concurrency: 1,
            // `MergePolicy.keepFullyDeletedSegment` returns false, and
            // `TieredMergePolicy` does not override it.
            keep_fully_deleted_segments: false,
        }
    }
}

impl MergePolicyConfig {
    /// Java validates `segsPerTier >= 2.0` in the setter; this port clamps
    /// instead so an out-of-range caller value can't wedge the level walk in
    /// [`find_merges_excluding`] (which subtracts `segs_per_tier * levelSize`
    /// per iteration and would never terminate at 0).
    fn segs_per_tier(&self) -> f64 {
        self.segments_per_tier.max(2.0)
    }

    /// Java's `mergeFactor = (int) Math.min(maxMergeAtOnce, segsPerTier)` --
    /// the *soft* per-merge cap, which a merge still under
    /// `floor_segment_size` is allowed to exceed, and the level-walk /
    /// `score()` skew denominator.
    fn merge_factor(&self) -> usize {
        self.max_merge_at_once
            .min(self.segs_per_tier() as usize)
            .max(1)
    }
}

/// Real Lucene's `TieredMergePolicy.floorSize`: `Math.max(floorSegmentBytes,
/// bytes)`. Scoring-only -- it never changes which segments are eligible.
fn floor_size(bytes: u64, floor_segment_size: u64) -> u64 {
    bytes.max(floor_segment_size)
}

/// Real Lucene's `getMaxAllowedDocs`:
/// `Math.ceilDiv(totalMaxDoc - totalDelDocs, targetSearchConcurrency)`.
fn max_allowed_docs(
    total_max_doc: i64,
    total_del_docs: i64,
    target_search_concurrency: usize,
) -> i64 {
    // `as i64` would turn a `usize` above `i64::MAX` into a *negative*
    // divisor -- `usize::MAX as i64` is `-1`, and `i64::MIN.div_euclid(-1)`
    // panics in release as well as debug. `try_from` keeps the divisor
    // positive, which is what both the `div_euclid` and the ceiling below
    // assume.
    let concurrency = i64::try_from(target_search_concurrency)
        .unwrap_or(i64::MAX)
        .max(1);
    // Saturating because both operands are caller-supplied sums (see
    // `find_merges_excluding`); a saturated `live` yields an *unbounded*
    // doc budget, which is exactly Java's `targetSearchConcurrency == 1`
    // default and can only make this policy merge less aggressively.
    let live = total_max_doc.saturating_sub(total_del_docs);
    // ARITH: `concurrency >= 1`, so this is neither a division by zero nor
    // the one overflowing division (`i64::MIN / -1`). The quotient is at
    // most `live`, and it equals `i64::MAX` only when `live == i64::MAX` and
    // `concurrency == 1` -- in which case `rem_euclid` is `0` and nothing is
    // added. So the `+` cannot overflow.
    #[allow(clippy::arithmetic_side_effects)]
    {
        live.div_euclid(concurrency) + i64::from(live.rem_euclid(concurrency) != 0)
    }
}

/// Real Lucene's `100.0 * delCount / maxDoc` -- a percentage
/// (`0.0..=100.0`), not a `0.0..=1.0` ratio.
fn pct_deletes(stat: &SegmentStat) -> f64 {
    stat.del_ratio() * 100.0
}

/// Java's `TieredMergePolicy.SegmentSizeAndDocs` record: a segment plus the
/// **pro-rated** size the algorithm sorts and budgets by, resolved once up
/// front (Java's comment: "the size can change concurrently while we are
/// running here ... so we call size() once per segment and sort by that").
#[derive(Debug, Clone, Copy)]
struct SegmentSizeAndDocs<'a> {
    stat: &'a SegmentStat,
    /// `MergePolicy.size(info)` -- pro-rated by live-doc fraction.
    size_in_bytes: u64,
}

impl<'a> SegmentSizeAndDocs<'a> {
    fn name(&self) -> &'a str {
        &self.stat.name
    }
    fn del_count(&self) -> i64 {
        self.stat.del_count as i64
    }
    fn max_doc(&self) -> i64 {
        self.stat.doc_count as i64
    }
    /// `SegmentCommitInfo.sizeInBytes()` -- the raw, *not* pro-rated size,
    /// used only by `score()`'s `totBeforeMergeBytes` and by
    /// `findForcedMerges`' bin packing (both of which read the raw figure in
    /// Java too).
    fn raw_size_in_bytes(&self) -> u64 {
        self.stat.size_bytes
    }

    /// The pro-rated size as a **non-negative** `i64`, for the byte budgets
    /// that are signed because Java's are `long`s.
    ///
    /// [`SegmentStat::size_bytes`] is a `pub u64`, so a caller can hand this
    /// module a size above `i64::MAX`. A bare `as i64` turns that into a
    /// *negative* number, and every budget below then treats the largest
    /// segment in the index as one that costs nothing: it passes the
    /// `bytes_this_merge + seg_bytes > max_merged` bound, is packed into a
    /// merge, and `tot_index_bytes` goes down when it should go up. Clamping
    /// keeps the sign, and a segment at `i64::MAX` bytes is excluded by every
    /// size bound rather than let into every merge.
    fn size_i64(&self) -> i64 {
        i64::try_from(self.size_in_bytes).unwrap_or(i64::MAX)
    }

    /// [`Self::size_i64`] for the raw (not pro-rated) size, for
    /// `findForcedMerges`' bin packing.
    fn raw_size_i64(&self) -> i64 {
        i64::try_from(self.raw_size_in_bytes()).unwrap_or(i64::MAX)
    }

    /// `maxDoc - delCount`: the segment's live-document count, the figure
    /// `findMerges` counts for an already-merging segment and `doFindMerges`
    /// budgets a candidate merge by.
    ///
    /// ARITH: [`SegmentStat`]'s two counts are `i32`s widened losslessly to
    /// `i64`, so each has magnitude below `2^31` and their difference lands
    /// in `-2^31..2^31`. This subtraction is the only one in the module that
    /// needs no clamp of its own, and every caller relies on that.
    #[allow(clippy::arithmetic_side_effects)]
    fn live_doc_count(&self) -> i64 {
        self.max_doc() - self.del_count()
    }
}

/// Port of `TieredMergePolicy.getSortedBySegmentSize`: **descending** by
/// pro-rated size, ties broken by segment name ascending (Java's
/// `o1.name.compareTo(o2.name)`), so the result is a deterministic total
/// order regardless of the caller's input order.
fn sorted_by_segment_size(segments: &[SegmentStat]) -> Vec<SegmentSizeAndDocs<'_>> {
    let mut sorted: Vec<SegmentSizeAndDocs<'_>> = segments
        .iter()
        .map(|stat| SegmentSizeAndDocs {
            stat,
            size_in_bytes: prorated_size(stat),
        })
        .collect();
    sorted.sort_by(|a, b| {
        b.size_in_bytes
            .cmp(&a.size_in_bytes)
            .then_with(|| a.name().cmp(b.name()))
    });
    sorted
}

/// Java's `TieredMergePolicy.MERGE_TYPE`, minus its `FORCE_MERGE` variant:
/// Java declares that one but never passes it to `doFindMerges` either --
/// `findForcedMerges` has its own bin-packing loop (see
/// [`find_forced_merges`]) -- so carrying a variant nothing can construct
/// would be shape parity for its own sake.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MergeType {
    Natural,
    ForceMergeDeletes,
}

/// Port of `TieredMergePolicy.findMerges(MergeTrigger, SegmentInfos,
/// MergeContext)` with an empty currently-merging set -- see
/// [`find_merges_excluding`] for the full form and the algorithm's
/// documentation. Returns zero or more groups of segment names, each an
/// independent merge (Java's `MergeSpecification` of `OneMerge`s); an empty
/// result means "no merge needed right now".
pub fn find_merges(segments: &[SegmentStat], config: &MergePolicyConfig) -> Vec<Vec<String>> {
    find_merges_excluding(segments, &HashSet::new(), config)
}

/// Port of `TieredMergePolicy.findMerges`, including the
/// `mergeContext.getMergingSegments()` exclusion Java performs: a segment
/// named in `merging` is already being merged, so it is removed from the
/// eligible set, its bytes are accumulated into `mergingBytes` (which gates
/// whether another max-size merge may start), and only its *live* docs count
/// towards the index's total doc-id space (its deletes are already being
/// reclaimed by the in-flight merge).
///
/// The algorithm, following Java step for step:
/// 1. Sort by pro-rated size descending ([`sorted_by_segment_size`]) and
///    accumulate `totIndexBytes`, `minSegmentBytes`, `totalDelDocs`,
///    `totalMaxDoc`, `mergingBytes`. Merging segments still count towards
///    `totIndexBytes`/`minSegmentBytes` (Java updates those *after* the
///    merging branch) but are dropped from the eligible list.
/// 2. Drop segments larger than half `max_merged_segment_size` whose deletes
///    (or the index's overall deletes) are within `deletes_pct_allowed` --
///    merging them further is not worth it. Each such "too big" segment
///    reduces `totIndexBytes` and the delete budget. Then reserve a whole
///    segment slot for each of the first `target_search_concurrency - 1`
///    remaining segments.
/// 3. Walk size levels upward from `max(minSegmentBytes, floorSegmentBytes)`,
///    multiplying by `mergeFactor` each level, accumulating
///    `allowedSegCount` -- the budget for how many segments the index is
///    allowed to have.
/// 4. Hand the eligible list to [`do_find_merges`], which repeatedly picks
///    the best-scoring merge until the index is back within budget.
pub fn find_merges_excluding(
    segments: &[SegmentStat],
    merging: &HashSet<String>,
    config: &MergePolicyConfig,
) -> Vec<Vec<String>> {
    // Step 1.
    let mut tot_index_bytes: i64 = 0;
    let mut min_segment_bytes = u64::MAX;
    let mut total_del_docs: i64 = 0;
    let mut total_max_doc: i64 = 0;
    let mut merging_bytes: u64 = 0;
    let mut kept: Vec<SegmentSizeAndDocs<'_>> = Vec::with_capacity(segments.len());
    for entry in sorted_by_segment_size(segments) {
        let seg_bytes = entry.size_in_bytes;
        if merging.contains(entry.name()) {
            merging_bytes = merging_bytes.saturating_add(seg_bytes);
            // If this segment is merging, its deletes are being reclaimed
            // already: only count live docs in the total max doc.
            total_max_doc = total_max_doc.saturating_add(entry.live_doc_count());
        } else {
            total_del_docs = total_del_docs.saturating_add(entry.del_count());
            total_max_doc = total_max_doc.saturating_add(entry.max_doc());
            kept.push(entry);
        }
        min_segment_bytes = min_segment_bytes.min(seg_bytes);
        tot_index_bytes = tot_index_bytes.saturating_add(entry.size_i64());
    }
    if kept.is_empty() {
        // Java's doFindMerges returns null for an empty eligible list.
        return Vec::new();
    }

    let total_del_pct = if total_max_doc > 0 {
        100.0 * total_del_docs as f64 / total_max_doc as f64
    } else {
        0.0
    };
    let mut allowed_del_count = (config.deletes_pct_allowed * total_max_doc as f64 / 100.0) as i64;

    // Step 2.
    let mut too_big_count: i64 = 0;
    let mut concurrency_count: i64 = 0;
    let mut allowed_seg_count: f64 = 0.0;
    // Same `as i64` trap as `max_allowed_docs`': a `usize` above `i64::MAX`
    // would become negative, and `target_concurrency - 1` below is then a
    // bound no segment can be under.
    let target_concurrency = i64::try_from(config.target_search_concurrency)
        .unwrap_or(i64::MAX)
        .max(1);
    let mut eligible: Vec<SegmentSizeAndDocs<'_>> = Vec::with_capacity(kept.len());
    for entry in kept {
        let seg_del_pct = if entry.max_doc() > 0 {
            100.0 * entry.del_count() as f64 / entry.max_doc() as f64
        } else {
            0.0
        };
        if entry.size_in_bytes > config.max_merged_segment_size / 2
            && (total_del_pct <= config.deletes_pct_allowed
                || seg_del_pct <= config.deletes_pct_allowed)
        {
            // ARITH: `too_big_count` and `concurrency_count` each step by one
            // per iteration of a loop over `kept`, whose length is a `Vec`
            // length and therefore at most `isize::MAX`.
            #[allow(clippy::arithmetic_side_effects)]
            {
                too_big_count += 1;
            }
            tot_index_bytes = tot_index_bytes.saturating_sub(entry.size_i64());
            allowed_del_count = allowed_del_count.saturating_sub(entry.del_count());
            continue;
        }
        // ARITH: both operands are bounded by `kept.len()` (see above), so
        // the sum is far inside `i64`; `target_concurrency - 1` cannot
        // underflow because `target_concurrency >= 1`.
        #[allow(clippy::arithmetic_side_effects)]
        let under_concurrency_reserve = concurrency_count + too_big_count < target_concurrency - 1;
        if under_concurrency_reserve {
            // Count a whole segment for the first targetSearchConcurrency-1
            // segments, to avoid over-merging on the lower levels.
            // ARITH: as above -- one step per `kept` entry.
            #[allow(clippy::arithmetic_side_effects)]
            {
                concurrency_count += 1;
            }
            allowed_seg_count += 1.0;
            tot_index_bytes = tot_index_bytes.saturating_sub(entry.size_i64());
        }
        eligible.push(entry);
    }
    allowed_del_count = allowed_del_count.max(0);

    // Step 3.
    let segs_per_tier = config.segs_per_tier();
    let merge_factor = config.merge_factor();
    // `max(.., 1)` guards a degenerate all-zero-size input, which would make
    // the level walk divide by zero and never terminate. Java cannot hit
    // this because real segment files are never zero bytes.
    let mut level_size = min_segment_bytes.max(config.floor_segment_size).max(1);
    let mut bytes_left = tot_index_bytes;
    loop {
        let seg_count_level = bytes_left as f64 / level_size as f64;
        if seg_count_level < segs_per_tier || level_size == config.max_merged_segment_size {
            allowed_seg_count += seg_count_level.ceil();
            break;
        }
        allowed_seg_count += segs_per_tier;
        // Saturating: `bytes_left` starts at `tot_index_bytes` and only ever
        // decreases, and the loop exits the moment it goes negative (a
        // negative `seg_count_level` is below `segs_per_tier`, which is
        // clamped to `>= 2.0`). So at most one subtraction can run against a
        // value near `i64::MIN`, and saturating it ends the walk one
        // iteration earlier rather than wrapping the budget positive.
        bytes_left = bytes_left.saturating_sub((segs_per_tier * level_size as f64) as i64);
        level_size = config
            .max_merged_segment_size
            .min(level_size.saturating_mul(merge_factor as u64));
    }
    // allowedSegCount may occasionally be less than segsPerTier if segment
    // sizes are below the floor size.
    allowed_seg_count = allowed_seg_count.max(segs_per_tier);
    // No need to merge if the total segment count (including too-big
    // segments) is at or below the target search concurrency.
    allowed_seg_count =
        allowed_seg_count.max(target_concurrency.saturating_sub(too_big_count) as f64);
    let allowed_doc_count = max_allowed_docs(
        total_max_doc,
        total_del_docs,
        config.target_search_concurrency,
    );

    // Step 4.
    do_find_merges(
        &eligible,
        config,
        MergeBudget {
            merge_factor,
            allowed_seg_count: allowed_seg_count as i64,
            allowed_del_count,
            allowed_doc_count,
            merge_type: MergeType::Natural,
            max_merge_is_running: merging_bytes >= config.max_merged_segment_size,
        },
    )
}

/// The five budget values `findMerges`/`findForcedDeletesMerges` compute and
/// hand to `doFindMerges`, plus the merge type -- Java passes them as six
/// positional parameters; grouping them keeps the two call sites readable and
/// self-documenting.
#[derive(Debug, Clone, Copy)]
struct MergeBudget {
    /// Java's `mergeFactor` (`(int) segsPerTier` for a natural merge,
    /// `Integer.MAX_VALUE` for force-merge-deletes).
    merge_factor: usize,
    allowed_seg_count: i64,
    allowed_del_count: i64,
    allowed_doc_count: i64,
    merge_type: MergeType,
    /// Java's `mergingBytes >= maxMergedSegmentBytes`: a max-size merge is
    /// already in flight, so don't start another.
    max_merge_is_running: bool,
}

/// Port of `TieredMergePolicy.doFindMerges` -- the shared engine behind
/// `findMerges` and `findForcedDeletesMerges`. Repeatedly:
/// - drops segments already claimed by a previously-selected merge;
/// - for a NATURAL merge, stops once the segment count is within
///   `allowed_seg_count` *and* the remaining deletes are within
///   `allowed_del_count`;
/// - considers a candidate merge starting at every index of the
///   size-descending list, packing segments in until `merge_factor` is hit
///   (or the merge is still below `floor_segment_size`), the merge would
///   exceed `max_merged_segment_size`, or it would exceed
///   `allowed_doc_count`;
/// - skips candidates that would only grow the biggest input by less than
///   50% (Java's guard against O(N^2) rewrite-the-same-segment merging)
///   unless that biggest input is itself delete-heavy;
/// - skips a singleton candidate with no deletes (nothing to reclaim);
/// - picks the lowest [`merge_score`], and emits at most one "too large"
///   merge per call.
fn do_find_merges<'a>(
    sorted_eligible: &[SegmentSizeAndDocs<'a>],
    config: &MergePolicyConfig,
    budget: MergeBudget,
) -> Vec<Vec<String>> {
    let MergeBudget {
        merge_factor,
        allowed_seg_count,
        allowed_del_count,
        allowed_doc_count,
        merge_type,
        max_merge_is_running,
    } = budget;
    let mut spec: Vec<Vec<String>> = Vec::new();
    if sorted_eligible.is_empty() {
        return spec;
    }
    // `as i64` on a `pub u64` config would make either bound *negative*,
    // and a negative `max_merged` is a bound `bytes_this_merge` can never be
    // under -- the packing loop would then propose nothing at all, silently,
    // for every index. Clamping keeps "absurdly large" meaning "unbounded".
    let max_merged = i64::try_from(config.max_merged_segment_size).unwrap_or(i64::MAX);
    let floor = i64::try_from(config.floor_segment_size).unwrap_or(i64::MAX);

    let mut to_be_merged: HashSet<&'a str> = HashSet::new();
    let mut have_one_large_merge = false;
    let mut remaining: Vec<SegmentSizeAndDocs<'a>> = sorted_eligible.to_vec();

    loop {
        remaining.retain(|entry| !to_be_merged.contains(entry.name()));
        if remaining.is_empty() {
            return spec;
        }
        // `sum()` would panic on overflow in a debug build; the count is a
        // budget comparison, and a saturated one only makes this loop keep
        // looking for a merge.
        let remaining_del_count: i64 = remaining
            .iter()
            .fold(0i64, |acc, e| acc.saturating_add(e.del_count()));
        if merge_type == MergeType::Natural
            && remaining.len() as i64 <= allowed_seg_count
            && remaining_del_count <= allowed_del_count
        {
            return spec;
        }

        // Over budget -- find the best merge.
        let mut best: Option<Vec<SegmentSizeAndDocs<'a>>> = None;
        let mut best_score = f64::INFINITY;
        let mut best_too_large = false;

        for start_idx in 0..remaining.len() {
            let mut candidate: Vec<SegmentSizeAndDocs<'a>> = Vec::new();
            let mut hit_too_large = false;
            let mut bytes_this_merge: i64 = 0;
            let mut doc_count_this_merge: i64 = 0;
            let mut idx = start_idx;
            while idx < remaining.len()
                && candidate.len() < config.max_merge_at_once
                // We allow merging more than mergeFactor segments together
                // if the merged segment would still be below the floor
                // segment size -- those are merged aggressively, so they
                // need to grow as fast as possible.
                && (candidate.len() < merge_factor || bytes_this_merge < floor)
                && bytes_this_merge < max_merged
                && (bytes_this_merge < floor || doc_count_this_merge <= allowed_doc_count)
            {
                let entry = remaining[idx];
                let seg_bytes = entry.size_i64();
                let seg_doc_count = entry.live_doc_count();
                // `checked_add` rather than `bytes_this_merge + seg_bytes`:
                // this is `docs/arithmetic-gate.md`'s "the guard forms the
                // very sum it exists to guard" shape, and the two bounds it
                // guards are the *hard* ones -- exceeding either produces an
                // oversized merged segment rather than a bad heuristic. An
                // overflowing sum is unambiguously "too large", so `None`
                // takes the same branch a sum past the bound would.
                let would_exceed_bytes = bytes_this_merge
                    .checked_add(seg_bytes)
                    .is_none_or(|total| total > max_merged);
                let would_exceed_docs = doc_count_this_merge
                    .checked_add(seg_doc_count)
                    .is_none_or(|total| total > allowed_doc_count);
                if would_exceed_bytes || (bytes_this_merge > floor && would_exceed_docs) {
                    // Only set hitTooLarge when reaching the maximum byte
                    // size: that creates a segment that will not be
                    // eligible for merging again for a long time.
                    hit_too_large |= would_exceed_bytes;
                    if !candidate.is_empty() {
                        // Keep going, to try "packing" smaller segments into
                        // this merge and get closer to the max size.
                        // ARITH: `idx` is bounded by `remaining.len()`, a
                        // `Vec` length, by this `while`'s own condition.
                        #[allow(clippy::arithmetic_side_effects)]
                        {
                            idx += 1;
                        }
                        continue;
                    }
                }
                candidate.push(entry);
                // Saturating: reachable only on the one iteration the two
                // guards above just declared "too large", which is also the
                // last -- `bytes_this_merge < max_merged` then fails and the
                // `while` exits.
                bytes_this_merge = bytes_this_merge.saturating_add(seg_bytes);
                doc_count_this_merge = doc_count_this_merge.saturating_add(seg_doc_count);
                // ARITH: as above -- `idx < remaining.len()` holds here.
                #[allow(clippy::arithmetic_side_effects)]
                {
                    idx += 1;
                }
            }
            if candidate.is_empty() {
                // Java asserts this is impossible (too-large segments are
                // pre-excluded); reachable here only for a degenerate config
                // such as max_merged_segment_size == 0.
                continue;
            }

            // The list is size-descending, so candidate[0] is the biggest.
            let max_candidate = candidate[0];
            if !hit_too_large
                && merge_type == MergeType::Natural
                && (bytes_this_merge as f64) < max_candidate.size_in_bytes as f64 * 1.5
                && (max_candidate.del_count() as f64)
                    < max_candidate.max_doc() as f64 * config.deletes_pct_allowed / 100.0
            {
                // Ignore a merge whose result is not at least 50% larger
                // than its biggest input: otherwise we hit pathological
                // O(N^2) merging that keeps rewriting the biggest input into
                // a segment barely bigger. Exception: when the merge would
                // reclaim lots of deletes from that biggest segment.
                continue;
            }

            // A singleton merge with no deletes makes no sense.
            if candidate.len() == 1 && max_candidate.del_count() == 0 {
                continue;
            }

            // If we didn't find a too-large merge and the candidate is
            // shorter than the merge factor, we've reached the tail of the
            // list and will only find smaller merges. Stop.
            if best.is_some() && !hit_too_large && candidate.len() < merge_factor {
                break;
            }

            let score = merge_score(&candidate, hit_too_large, config);
            if (best.is_none() || score < best_score) && (!hit_too_large || !max_merge_is_running) {
                best_score = score;
                best_too_large = hit_too_large;
                best = Some(candidate);
            }
        }

        let Some(best) = best else {
            return spec;
        };
        // The trigger point for total deleted documents leads to a bunch of
        // large segment merges at the same time, so only put one large merge
        // in the list per cycle -- we'll pick up another next time around.
        if !have_one_large_merge || !best_too_large || merge_type == MergeType::ForceMergeDeletes {
            have_one_large_merge |= best_too_large;
            spec.push(best.iter().map(|e| e.name().to_string()).collect());
        }
        // Whether or not we returned it in the spec, remove it from
        // consideration on the next loop.
        for entry in &best {
            to_be_merged.insert(entry.name());
        }
    }
}

/// Port of `TieredMergePolicy.score` -- lower is better:
///
/// ```text
/// skew        = hitTooLarge ? 1.0 / mergeFactor
///                           : floorSize(biggest) / sum(floorSize(seg))
/// mergeScore  = skew
///             * totAfterMergeBytes ^ 0.05        // gently favour smaller merges
///             * nonDelRatio ^ reclaim_weight     // strongly favour reclaiming deletes
/// nonDelRatio = sum(proratedSize) / sum(rawSize)
/// ```
///
/// `mergeFactor` in the `hitTooLarge` branch is Java's `(int) segsPerTier`,
/// not the `mergeFactor` parameter threaded through `doFindMerges` (which is
/// `Integer.MAX_VALUE` for a force-merge-deletes pass).
fn merge_score(
    candidate: &[SegmentSizeAndDocs<'_>],
    hit_too_large: bool,
    config: &MergePolicyConfig,
) -> f64 {
    let mut tot_before_merge_bytes: f64 = 0.0;
    let mut tot_after_merge_bytes: f64 = 0.0;
    let mut tot_after_merge_bytes_floored: f64 = 0.0;
    for entry in candidate {
        tot_after_merge_bytes += entry.size_in_bytes as f64;
        tot_after_merge_bytes_floored +=
            floor_size(entry.size_in_bytes, config.floor_segment_size) as f64;
        tot_before_merge_bytes += entry.raw_size_in_bytes() as f64;
    }

    // Roughly measure "skew": how balanced the merge is, from
    // 1.0/numSegsBeingMerged (good) to 1.0 (poor). Heavily lopsided merges
    // mean O(N^2) merge cost over time.
    let skew = if hit_too_large {
        // Pretend the merge has perfect skew: it will not cascade, so it
        // cannot lead to N^2 merge cost over time.
        1.0 / config.merge_factor() as f64
    } else if tot_after_merge_bytes_floored > 0.0 {
        floor_size(candidate[0].size_in_bytes, config.floor_segment_size) as f64
            / tot_after_merge_bytes_floored
    } else {
        1.0
    };

    let mut merge_score = skew;
    // Gently favour smaller merges over bigger ones.
    merge_score *= tot_after_merge_bytes.powf(0.05);
    // Strongly favour merges that reclaim deletes.
    let non_del_ratio = if tot_before_merge_bytes > 0.0 {
        tot_after_merge_bytes / tot_before_merge_bytes
    } else {
        1.0
    };
    merge_score *= non_del_ratio.powf(config.reclaim_weight);
    merge_score
}

/// Pass as `max_segment_count` to [`find_forced_merges`] for Java's
/// `Integer.MAX_VALUE` "no explicit segment-count target" case, where the
/// scoring is left to decide how much to merge.
pub const UNLIMITED_SEGMENT_COUNT: usize = usize::MAX;

/// Port of `TieredMergePolicy.findForcedMerges` -- `IndexWriter.forceMerge(n)`:
/// merge down to at most `max_segment_count` segments.
///
/// Faithful to Java's structure:
/// - `maxMergeBytes` is `Long.MAX_VALUE` for `max_segment_count == 1`,
///   otherwise `max(totalMergeBytes / maxSegmentCount, maxMergedSegmentBytes)
///   * 1.25` (the 25% fudge that avoids needing a second pass).
/// - Delete-free segments already at or above `maxMergeBytes` are dropped.
/// - If nothing has deletes and we are already within `max_segment_count`,
///   nothing is proposed.
/// - Merging down to one segment when everything fits is a single group.
/// - Otherwise candidates are bin-packed **from the smallest end upward**
///   (Java walks the size-descending list backwards), each bin capped at
///   `maxMergeBytes` but always allowed at least two segments, stopping as
///   soon as the projected surviving segment count reaches
///   `max_segment_count`.
///
/// Every supplied segment is treated as "original" and none as merging;
/// [`forced_merges_full`] (reached through [`TieredMergePolicy`]) takes Java's
/// `segmentsToMerge` and `getMergingSegments()` as given.
pub fn find_forced_merges(
    segments: &[SegmentStat],
    max_segment_count: usize,
    config: &MergePolicyConfig,
) -> Vec<Vec<String>> {
    let all_original: std::collections::HashMap<String, bool> =
        segments.iter().map(|s| (s.name.clone(), true)).collect();
    let is_merged = |name: &str| -> std::result::Result<bool, std::convert::Infallible> {
        Ok(segments.iter().any(|s| s.name == name && s.del_count == 0))
    };
    match forced_merges_full(
        segments,
        max_segment_count,
        &all_original,
        &HashSet::new(),
        is_merged,
        config,
    ) {
        Ok(groups) => groups,
        Err(never) => match never {},
    }
}

/// The whole of `TieredMergePolicy.findForcedMerges`, which
/// [`find_forced_merges`] calls with every segment "original" and nothing
/// merging: `segments_to_merge` is Java's `segmentsToMerge` (name -> "is
/// original"; a segment absent from it is not merged), `merging` is
/// `getMergingSegments()` (any of them among the segments to merge means a
/// forced merge is already running, and nothing new is proposed), and
/// `is_merged` answers `isMerged(infos, info, ctx)` for a segment name.
///
/// `is_merged` is asked lazily, at most once, exactly where Java's
/// short-circuiting condition asks it: for the single remaining segment of a
/// delete-free merge down to one segment that `segments_to_merge` does not
/// already name. Answering it for every segment up front costs a
/// `useCompoundFile` sizing per segment and can surface errors Java never
/// would.
pub(crate) fn forced_merges_full<E>(
    segments: &[SegmentStat],
    max_segment_count: usize,
    segments_to_merge: &std::collections::HashMap<String, bool>,
    merging: &HashSet<String>,
    is_merged: impl FnOnce(&str) -> std::result::Result<bool, E>,
    config: &MergePolicyConfig,
) -> std::result::Result<Vec<Vec<String>>, E> {
    let max_segment_count = max_segment_count.max(1);
    let mut sorted = sorted_by_segment_size(segments);
    let mut force_merge_running = false;
    sorted.retain(|e| {
        if !segments_to_merge.contains_key(e.name()) {
            return false;
        }
        if merging.contains(e.name()) {
            force_merge_running = true;
            return false;
        }
        true
    });
    if sorted.is_empty() {
        return Ok(Vec::new());
    }
    // `sum()` panics on overflow in a debug build. Saturating here is the
    // same call `segment_byte_size` makes: an index whose segments really do
    // total `i64::MAX` bytes wants the *largest* bin size this can express,
    // which is what saturation gives.
    let total_merge_bytes: i64 = sorted
        .iter()
        .fold(0i64, |acc, e| acc.saturating_add(e.size_i64()));

    // Same clamp as `do_find_merges`: `as i64` on a `pub u64` config would
    // make the ceiling negative, which drops every delete-free segment.
    let max_merged_segment_size = i64::try_from(config.max_merged_segment_size).unwrap_or(i64::MAX);
    let mut max_merge_bytes: i64 = max_merged_segment_size;
    if max_segment_count == 1 {
        max_merge_bytes = i64::MAX;
    } else if max_segment_count != UNLIMITED_SEGMENT_COUNT {
        // ARITH: `max_segment_count` is `>= 2` here (`== 1` took the branch
        // above and the parameter was clamped to `>= 1`), and `try_from`
        // keeps the divisor positive, so this is neither a division by zero
        // nor `i64::MIN / -1`.
        let divisor = i64::try_from(max_segment_count).unwrap_or(i64::MAX).max(1);
        // ARITH: `divisor >= 1` by the `.max(1)` on the line above, so the
        // division can neither trap on zero nor on `i64::MIN / -1`.
        #[allow(clippy::arithmetic_side_effects)]
        {
            max_merge_bytes = (total_merge_bytes / divisor).max(max_merged_segment_size);
        }
        // Fudge up a bit so we have a better chance of not needing a second
        // merging pass to reach the requested target segment count. The
        // `as i64` back is saturating in Rust, so the 25% cannot wrap.
        max_merge_bytes = (max_merge_bytes as f64 * 1.25) as i64;
    }

    let mut found_deletes = false;
    sorted.retain(|entry| {
        let is_original = segments_to_merge.get(entry.name()).copied();
        if entry.del_count() != 0 {
            // This is forceMerge: every segment with deleted docs is merged.
            if is_original == Some(true) {
                found_deletes = true;
            }
            return true;
        }
        if max_segment_count == UNLIMITED_SEGMENT_COUNT && is_original == Some(false) {
            return false;
        }
        // Don't try to merge a delete-free segment that's over the max size.
        !(max_segment_count != UNLIMITED_SEGMENT_COUNT && entry.size_i64() >= max_merge_bytes)
    });

    if sorted.is_empty() {
        return Ok(Vec::new());
    }

    // We only bail if there are no deletions.
    if !found_deletes {
        let info_zero = sorted[0].name();
        let already_merged = (max_segment_count != UNLIMITED_SEGMENT_COUNT
            && max_segment_count > 1
            && sorted.len() <= max_segment_count)
            || (max_segment_count == 1
                && sorted.len() == 1
                && (segments_to_merge.contains_key(info_zero) || is_merged(info_zero)?));
        if already_merged {
            return Ok(Vec::new());
        }
    }

    if force_merge_running {
        return Ok(Vec::new());
    }

    let starting_segment_count = sorted.len();

    // Special case: merging down to a single segment that fits.
    if max_segment_count == 1 && total_merge_bytes < max_merge_bytes {
        return Ok(vec![sorted.iter().map(|e| e.name().to_string()).collect()]);
    }

    let mut spec: Vec<Vec<String>> = Vec::new();
    // ARITH: `starting_segment_count` is `sorted.len()`, a `Vec` length and
    // therefore at most `isize::MAX`, so the widening and the `- 1` are both
    // exact. `index` then only decreases, and never below `-1` because the
    // `while` guards `index >= 0` before each step.
    #[allow(clippy::arithmetic_side_effects)]
    let mut index: isize = starting_segment_count as isize - 1;
    let mut resulting_segments = starting_segment_count;
    loop {
        let mut candidate: Vec<&str> = Vec::new();
        let mut current_candidate_bytes: i64 = 0;
        while index >= 0 && resulting_segments > max_segment_count {
            let current = sorted[index as usize];
            let initial_candidate_size = candidate.len();
            // Java reads the RAW size here (`current.sizeInBytes()`), not
            // the pro-rated one, so bin packing accounts for real disk cost.
            let current_segment_size = current.raw_size_i64();
            // Add to the bin either because there's room, or because this is
            // the smallest possible bin (decrementing moves to even larger
            // segments). `checked_add` for the same reason `do_find_merges`
            // uses one: an overflowing sum must read as "the bin is full",
            // not as "there is room".
            let fits = current_candidate_bytes
                .checked_add(current_segment_size)
                .is_some_and(|total| total <= max_merge_bytes);
            if fits || initial_candidate_size < 2 {
                candidate.push(current.name());
                // ARITH: guarded by `index >= 0` immediately above.
                #[allow(clippy::arithmetic_side_effects)]
                {
                    index -= 1;
                }
                // Saturating: only reachable on the `initial_candidate_size
                // < 2` branch that ignored `fits`, i.e. at most twice per
                // bin, and an over-full bin is bounded by nothing else here.
                current_candidate_bytes =
                    current_candidate_bytes.saturating_add(current_segment_size);
                if initial_candidate_size > 0 {
                    // Any merge of two or more segments reduces the segment
                    // count by (handled - 1).
                    // ARITH: `resulting_segments > max_segment_count >= 1`
                    // is this `while`'s own condition, so it is at least 2.
                    #[allow(clippy::arithmetic_side_effects)]
                    {
                        resulting_segments -= 1;
                    }
                }
            } else {
                break;
            }
        }
        if candidate.len() > 1 {
            spec.push(candidate.into_iter().map(str::to_string).collect());
        } else {
            return Ok(spec);
        }
    }
}

/// Port of `TieredMergePolicy.findForcedDeletesMerges` --
/// `IndexWriter.forceMergeDeletes()`: reclaim space from segments whose
/// deleted-doc percentage *strictly exceeds*
/// `config.force_merge_deletes_pct_allowed`, leaving lower-deletion segments
/// untouched.
///
/// Faithful to Java: a quick "is there any work" pass first, then the
/// over-threshold segments are handed to the very same [`do_find_merges`]
/// engine `findMerges` uses, with `mergeFactor` and `allowedSegCount`
/// unbounded, `allowedDelCount == 0` (every remaining delete is worth
/// reclaiming) and the real `allowedDocCount`. That matters: unlike the
/// previous simplified implementation, the merges produced here **respect
/// `max_merged_segment_size`** ("findForcedDeletesMerges should never
/// produce segments greater than maxSegmentSize", per Java's own class
/// javadoc), so a huge pile of delete-heavy segments comes back as several
/// bounded merges rather than one unbounded one.
///
/// A group of a single qualifying segment is valid here (unlike
/// [`find_merges`]): real Lucene does rewrite one heavily-deleted segment on
/// its own to physically drop its deleted docs. `do_find_merges`' singleton
/// guard only rejects singletons with *no* deletes, which by construction
/// cannot appear in this eligible set.
pub fn find_forced_delete_merges(
    segments: &[SegmentStat],
    config: &MergePolicyConfig,
) -> Vec<Vec<String>> {
    find_forced_delete_merges_excluding(segments, &HashSet::new(), config)
}

/// [`find_forced_delete_merges`] with Java's
/// `mergeContext.getMergingSegments()` exclusion made explicit -- a segment
/// already being merged neither counts as "work to do" nor is eligible.
pub fn find_forced_delete_merges_excluding(
    segments: &[SegmentStat],
    merging: &HashSet<String>,
    config: &MergePolicyConfig,
) -> Vec<Vec<String>> {
    // First a quick check that there's any work to do.
    let mut have_work = false;
    let mut total_del_count: i64 = 0;
    let mut total_max_doc: i64 = 0;
    for stat in segments {
        // Saturating for the same reason `find_merges_excluding`'s
        // accumulators are: these two feed `max_allowed_docs`, a heuristic
        // budget, and `segments` is a caller-supplied slice whose length this
        // module does not bound.
        total_del_count = total_del_count.saturating_add(i64::from(stat.del_count));
        total_max_doc = total_max_doc.saturating_add(i64::from(stat.doc_count));
        have_work = have_work
            || (pct_deletes(stat) > config.force_merge_deletes_pct_allowed
                && !merging.contains(&stat.name));
    }
    if !have_work {
        return Vec::new();
    }

    let mut sorted = sorted_by_segment_size(segments);
    sorted.retain(|entry| {
        !merging.contains(entry.name())
            && pct_deletes(entry.stat) > config.force_merge_deletes_pct_allowed
    });

    do_find_merges(
        &sorted,
        config,
        MergeBudget {
            merge_factor: usize::MAX,
            allowed_seg_count: i64::MAX,
            allowed_del_count: 0,
            allowed_doc_count: max_allowed_docs(
                total_max_doc,
                total_del_count,
                config.target_search_concurrency,
            ),
            merge_type: MergeType::ForceMergeDeletes,
            max_merge_is_running: false,
        },
    )
}

#[cfg(test)]
mod tests;
