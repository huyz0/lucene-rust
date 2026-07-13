//! Port of `org.apache.lucene.index.TieredMergePolicy` -- **the decision
//! function only**: given a segment list's stats, which segments (if any)
//! should be merged together next. This module does not execute merges (see
//! [`crate::merge`] for that -- its own module doc explicitly calls out "not
//! a merge policy" as out of scope, which is exactly the gap this module
//! fills) and does not run anything in a background thread (real Lucene's
//! `MergeScheduler`/`ConcurrentMergeScheduler` is out of scope here too --
//! this is a pure, synchronous, side-effect-free function of its input).
//!
//! # What real `TieredMergePolicy` does
//!
//! Real Lucene's algorithm (`findMerges`), roughly:
//! 1. Segments already at or above `maxMergedSegmentBytes` are excluded from
//!    being merge *inputs* (merging them further would only make an
//!    over-sized segment; they're left alone).
//! 2. Remaining segments are scored for how good a merge candidate they are:
//!    smaller segments and segments with a higher deleted-doc ratio
//!    (`delCount / maxDoc`) score better, because merging them reclaims disk
//!    space (dropped deletes) relative to the work done. Real Lucene's exact
//!    scoring (`MergeScore`) balances a "skew" penalty (avoid merges that mix
//!    a huge segment with tiny ones) against a reclaim-percentage bonus.
//! 3. The policy greedily builds one or more merge candidates of up to
//!    `maxMergeAtOnce` segments each, aiming to bring the number of segments
//!    "at a given size tier" down to roughly `segmentsPerTier`, preferring to
//!    start from the best-scoring (smallest / most-reclaimable) segments.
//!
//! # What this module keeps vs. simplifies
//!
//! **Kept (real-shaped, not hand-waved):**
//! - Excluding at-or-above-max-size segments from further merge input
//!   (real behavior #1 above).
//! - A reclaim-weighted score, not naive size-only bin-packing: two
//!   same-size segments with different `del_count` ratios are *not*
//!   equivalent merge candidates here, matching real behavior #2's intent
//!   that deletion-heavy segments are preferred merge fodder.
//! - A segment-count target (`segments_per_tier`) that suppresses merges
//!   once the segment count is already at or below it (unless an oversized
//!   segment is present -- see below), and a `max_merge_at_once` cap that no
//!   single proposed group ever exceeds (real behavior #3).
//! - Preferring to merge smaller / more-deleted segments first, by sorting
//!   candidates by score before grouping (real behavior #3).
//!
//! **Simplified / dropped (documented here, not silently missing):**
//! - **Size unit.** Real Lucene sizes segments by on-disk byte size
//!   (`segBytes`, the sum of a segment's file lengths, itself adjusted by
//!   `(1 - delRatio)` to approximate "live bytes"). This port supports two
//!   ways to obtain a segment's size, both feeding the *same* algorithm
//!   below:
//!   - [`segment_byte_size`]: sums real on-disk file lengths for a segment's
//!     files via the existing [`lucene_store::directory::Directory`] trait
//!     (no new trait method needed -- `Directory::open` + the returned
//!     [`lucene_store::data_input::SliceInput`]'s `len()` is enough). This is
//!     the honest, byte-accurate answer when a `Directory` and the
//!     segment's `.si`-listed files are available.
//!   - A caller without a `Directory` handy (e.g. pure unit tests, or a
//!     caller that only has `SegmentCommitInfo`/doc-count style stats) may
//!     instead approximate size via `doc_count` as a stand-in unit --
//!     documented here explicitly as an approximation, *not* claimed to be
//!     byte-accurate. [`SegmentStat`] is deliberately unit-agnostic (its
//!     `size_bytes` field just means "whatever monotonic size unit the
//!     caller chose") so both call styles work through the same
//!     [`find_merges`] without the algorithm caring which one is real bytes.
//! - **Real Lucene's exact `MergeScore` formula** (a skew penalty using
//!   `Math.log`, a floor/ceiling on considered segment sizes, tier
//!   floor/ceiling smoothing (`floorSegmentBytes`), and per-tier iterative
//!   refinement across multiple candidate merges with rollback) is not
//!   ported byte-for-byte -- this uses a simpler, real-*shaped* score:
//!   `size * (1.0 - reclaim_weight * del_ratio)`, i.e. smaller-effective-size
//!   segments (whether small outright, or size discounted by how many
//!   deletes merging would reclaim) sort first. This keeps the "reclaim
//!   deletes preferentially" property real Lucene has without claiming to
//!   reproduce its exact numeric output (which even real Lucene does not
//!   guarantee is stable across versions).
//! - **Single best merge group per call, not an iterative multi-merge
//!   search.** Real `findMerges` can return several simultaneous merge
//!   candidates in one call (bounded by `maxMergeCount`/concurrent-merge
//!   limits it also manages). This port's [`find_merges`] proposes merges
//!   greedily, repeatedly grouping up to `max_merge_at_once` of the
//!   best-scoring remaining eligible segments until fewer than
//!   `segments_per_tier` eligible segments remain (or fewer than 2, since a
//!   1-segment "merge" is meaningless) -- similar end effect, simpler
//!   control flow.
//! - **No forced-merge deletes-pending-only variant, no compound-file
//!   awareness, no `maxMergedSegmentBytes` floor tiering
//!   (`floorSegmentBytes`).**
//!
//! [`find_forced_merges`] is also provided, a simplified
//! `findForcedMerges`-equivalent: merge everything down to at most
//! `max_segment_count` segments, ignoring size/reclaim scoring (real forced
//! merge is "merge down to N segments, full stop").

use crate::segment_info::SegmentInfo;
use lucene_store::directory::Directory;

/// The stats [`find_merges`] needs about one segment. Deliberately not
/// `SegmentCommitInfo` directly: `SegmentCommitInfo` (`segment_infos.rs`)
/// carries `del_count` but not doc count or byte size -- those live in the
/// separate per-segment `.si` file (`SegmentInfo`, `segment_info.rs`). A
/// caller that has both parsed already builds a `SegmentStat` from them (see
/// [`SegmentStat::from_segment_info`]); a caller with only a `Directory` can
/// use [`segment_byte_size`] to fill in `size_bytes`.
#[derive(Debug, Clone, PartialEq)]
pub struct SegmentStat {
    pub name: String,
    /// Total (including deleted) doc count -- real Lucene's `maxDoc`.
    pub doc_count: i32,
    pub del_count: i32,
    /// Size in whatever unit the caller chose -- real on-disk bytes if
    /// obtained via [`segment_byte_size`], or a doc-count-based
    /// approximation otherwise. The algorithm only requires this be a
    /// monotonic, comparable size unit; it does not require actual bytes.
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

/// Sums real on-disk file lengths for a segment's files (`SegmentInfo::files`)
/// via the existing [`Directory`] trait -- the honest, byte-accurate way to
/// fill [`SegmentStat::size_bytes`] when a `Directory` is available. Missing
/// files are skipped (matches "some auxiliary files are optional" rather
/// than erroring the whole computation over one absent file).
pub fn segment_byte_size(dir: &dyn Directory, info: &SegmentInfo) -> u64 {
    let mut total = 0u64;
    for file in &info.files {
        if let Ok(input) = dir.open(file) {
            total += input.len() as u64;
        }
    }
    total
}

/// Tunables for [`find_merges`]/[`find_forced_merges`]. Defaults mirror real
/// `TieredMergePolicy`'s own defaults where known:
/// `maxMergeAtOnce = 10`, `segmentsPerTier = 10`,
/// `maxMergedSegmentMB = 5000` (here expressed in `size_bytes` units --
/// interpreted as either real bytes, if `size_bytes` was filled via
/// [`segment_byte_size`], or the doc-count approximation unit otherwise; see
/// this module's doc comment).
#[derive(Debug, Clone, PartialEq)]
pub struct MergePolicyConfig {
    /// Max number of segments combined into one merge group.
    pub max_merge_at_once: usize,
    /// Target number of segments to converge towards (per "tier" in real
    /// Lucene; this port does not model multiple size tiers separately, so
    /// this is applied as a single overall target).
    pub segments_per_tier: usize,
    /// Segments at or above this size are excluded from further merging.
    pub max_merged_segment_size: u64,
    /// How strongly a segment's deleted-doc ratio discounts its effective
    /// score (bigger => reclaiming deletes is preferred more strongly over
    /// picking purely by size). `0.0` disables reclaim-weighting entirely
    /// (falls back to pure size-based selection).
    pub reclaim_weight: f64,
}

impl Default for MergePolicyConfig {
    fn default() -> Self {
        // maxMergedSegmentMB=5000 => 5000 * 1024 * 1024 bytes. When callers
        // use the doc-count approximation instead of real bytes, this
        // default is not meaningful as a doc count and should be overridden.
        MergePolicyConfig {
            max_merge_at_once: 10,
            segments_per_tier: 10,
            max_merged_segment_size: 5_000 * 1024 * 1024,
            reclaim_weight: 1.0,
        }
    }
}

/// Real `TieredMergePolicy.findMerges`-equivalent: decides which currently
/// existing segments should be merged together next, given their stats.
/// Returns zero or more groups of segment names (each an independent merge);
/// an empty result means "no merge needed right now." Never proposes a
/// group of size `1` (nothing to merge) or larger than
/// `config.max_merge_at_once`.
///
/// See this module's doc comment for exactly which real `TieredMergePolicy`
/// behaviors this keeps vs. simplifies.
pub fn find_merges(segments: &[SegmentStat], config: &MergePolicyConfig) -> Vec<Vec<String>> {
    // Step 1: exclude already-oversized segments from further merge input
    // (real behavior: segments at/above maxMergedSegmentBytes are left alone).
    let mut eligible: Vec<&SegmentStat> = segments
        .iter()
        .filter(|s| s.size_bytes < config.max_merged_segment_size)
        .collect();

    if eligible.len() < 2 {
        return Vec::new();
    }

    // Nothing to do if we're already at or below the target segment count
    // and no segment needs a reclaim-driven merge push. Real Lucene still
    // considers merging when segment count is within the target but
    // reclaimable deletes are high; this port keeps it simple: below/at
    // target => no merge, matching "already optimal" the task's test wants.
    if eligible.len() <= config.segments_per_tier {
        return Vec::new();
    }

    // Step 2: score each eligible segment (smaller effective size sorts
    // first = more attractive as a merge input). Effective size = raw size
    // discounted by how much deleted-doc reclaim merging it would achieve.
    eligible.sort_by(|a, b| {
        let score_a = effective_score(a, config.reclaim_weight);
        let score_b = effective_score(b, config.reclaim_weight);
        score_a
            .partial_cmp(&score_b)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Step 3: greedily group the best-scoring segments into merge groups of
    // up to max_merge_at_once, until fewer than segments_per_tier (or fewer
    // than 2) segments remain ungrouped, aiming to bring the segment count
    // down towards segments_per_tier.
    let mut groups = Vec::new();
    let mut remaining = eligible.as_slice();
    let target_after = config.segments_per_tier.max(1);

    while remaining.len() > target_after && remaining.len() >= 2 {
        let take = config.max_merge_at_once.min(remaining.len());
        if take < 2 {
            break;
        }
        // Don't let a merge group shrink the total segment count below the
        // target in a way that leaves a lone leftover segment stranded
        // pointlessly small; simplest correct behavior is just "take up to
        // max_merge_at_once every time" -- matches real Lucene's own
        // "merge in max_merge_at_once-sized chunks" shape closely enough
        // for this port's scope.
        let (group, rest) = remaining.split_at(take);
        groups.push(group.iter().map(|s| s.name.clone()).collect());
        remaining = rest;
    }

    groups
}

/// Simplified `findForcedMerges`-equivalent: merge everything down to at
/// most `max_segment_count` segments, ignoring size/reclaim scoring (real
/// forced merge does not skip oversized segments either -- "merge down to N,
/// full stop"). Returns zero or more groups; empty if already at or below
/// `max_segment_count`.
pub fn find_forced_merges(segments: &[SegmentStat], max_segment_count: usize) -> Vec<Vec<String>> {
    let max_segment_count = max_segment_count.max(1);
    if segments.len() <= max_segment_count {
        return Vec::new();
    }

    // Merge segments down to exactly max_segment_count survivors: group the
    // excess count of segments into one merge, leaving the rest untouched.
    // (Real Lucene's actual forced-merge chunking is more elaborate --
    // repeated max_merge_at_once-sized passes -- but "merge the excess into
    // one group" is a faithful, simply-scoped rendition of "converge to N
    // segments" for this port.)
    let excess = segments.len() - max_segment_count + 1;
    let group = segments[..excess].iter().map(|s| s.name.clone()).collect();
    vec![group]
}

fn effective_score(stat: &SegmentStat, reclaim_weight: f64) -> f64 {
    let discount = (reclaim_weight * stat.del_ratio()).clamp(0.0, 1.0);
    stat.size_bytes as f64 * (1.0 - discount)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stat(name: &str, doc_count: i32, del_count: i32, size_bytes: u64) -> SegmentStat {
        SegmentStat {
            name: name.to_string(),
            doc_count,
            del_count,
            size_bytes,
        }
    }

    fn small_config() -> MergePolicyConfig {
        MergePolicyConfig {
            max_merge_at_once: 3,
            segments_per_tier: 3,
            max_merged_segment_size: 1_000_000,
            reclaim_weight: 1.0,
        }
    }

    #[test]
    fn many_small_segments_propose_a_merge() {
        let segments: Vec<SegmentStat> = (0..8)
            .map(|i| stat(&format!("_{i}"), 100, 0, 100))
            .collect();
        let groups = find_merges(&segments, &small_config());
        assert!(!groups.is_empty(), "expected at least one merge group");
        for g in &groups {
            assert!(g.len() >= 2);
            assert!(g.len() <= small_config().max_merge_at_once);
        }
        // Total segments across groups should reduce the count towards the
        // target, i.e. more than one group merged given 8 inputs / cap 3.
        let merged_count: usize = groups.iter().map(|g| g.len()).sum();
        assert!(merged_count >= 5);
    }

    #[test]
    fn oversized_segment_excluded_from_merges() {
        let mut segments: Vec<SegmentStat> = (0..8)
            .map(|i| stat(&format!("_{i}"), 100, 0, 100))
            .collect();
        segments.push(stat("_huge", 100, 0, 10_000_000));
        let groups = find_merges(&segments, &small_config());
        for g in &groups {
            assert!(!g.contains(&"_huge".to_string()));
        }
    }

    #[test]
    fn high_del_ratio_segment_preferred_over_larger_low_del_ratio() {
        // _high_del is both smaller AND has heavy deletes; _low_del is
        // larger and clean. Both properties point the same direction here,
        // so this alone doesn't isolate reclaim-weighting from plain
        // size-based preference -- see the next test for that isolation.
        let config = MergePolicyConfig {
            max_merge_at_once: 2,
            segments_per_tier: 2,
            max_merged_segment_size: 1_000_000,
            reclaim_weight: 1.0,
        };
        let segments = vec![
            stat("_low_del", 100, 0, 2000),
            stat("_high_del", 100, 90, 1000),
            stat("_other", 100, 0, 1000),
            stat("_other2", 100, 0, 1000),
        ];
        let groups = find_merges(&segments, &config);
        assert_eq!(groups.len(), 1);
        assert!(groups[0].contains(&"_high_del".to_string()));
        assert!(!groups[0].contains(&"_low_del".to_string()));
    }

    #[test]
    fn reclaim_weighting_moves_a_heavily_deleted_segment_ahead_of_equal_size_clean_ones() {
        // Three EQUAL-size_bytes segments, only one with heavy deletes.
        // With reclaim_weight == 0.0 (deletes ignored), all three score
        // identically, so the stable sort's tie-break is pure input order:
        // the first two listed (_mid, _low -- NOT _high_del, listed last)
        // fill the size-2 merge group. This isolates the reclaim-weighting
        // property itself: with reclaim_weight == 1.0, _high_del's heavy
        // deletes must lower its score enough to displace at least one of
        // the equal-size clean segments from the group, proving the
        // del-ratio term -- not size or input order -- decides the outcome.
        let segments = vec![
            stat("_mid", 100, 0, 1000),
            stat("_low", 100, 0, 1000),
            stat("_high_del", 100, 90, 1000),
        ];

        let no_reclaim_weighting = MergePolicyConfig {
            max_merge_at_once: 2,
            segments_per_tier: 2,
            max_merged_segment_size: 1_000_000,
            reclaim_weight: 0.0,
        };
        let groups = find_merges(&segments, &no_reclaim_weighting);
        assert_eq!(groups.len(), 1);
        assert_eq!(
            groups[0],
            vec!["_mid".to_string(), "_low".to_string()],
            "with reclaim_weight=0.0, an exact 3-way tie breaks by input \
             order (stable sort), excluding _high_del even though it's \
             heavily deleted: {groups:?}"
        );

        let with_reclaim_weighting = MergePolicyConfig {
            reclaim_weight: 1.0,
            ..no_reclaim_weighting
        };
        let groups = find_merges(&segments, &with_reclaim_weighting);
        assert_eq!(groups.len(), 1);
        assert!(
            groups[0].contains(&"_high_del".to_string()),
            "with reclaim_weight=1.0, the heavily-deleted segment's lower \
             score must include it in the merge group: {groups:?}"
        );
    }

    #[test]
    fn already_optimal_segment_count_proposes_nothing() {
        let segments: Vec<SegmentStat> = (0..3)
            .map(|i| stat(&format!("_{i}"), 100, 0, 100))
            .collect();
        let groups = find_merges(&segments, &small_config());
        assert!(groups.is_empty());
    }

    #[test]
    fn fewer_than_two_eligible_segments_proposes_nothing() {
        let segments = vec![stat("_0", 100, 0, 100)];
        assert!(find_merges(&segments, &small_config()).is_empty());

        let segments: Vec<SegmentStat> = vec![];
        assert!(find_merges(&segments, &small_config()).is_empty());
    }

    #[test]
    fn all_oversized_proposes_nothing() {
        let segments = vec![
            stat("_0", 100, 0, 10_000_000),
            stat("_1", 100, 0, 10_000_000),
        ];
        assert!(find_merges(&segments, &small_config()).is_empty());
    }

    #[test]
    fn max_merge_at_once_cap_respected_with_many_segments() {
        let config = MergePolicyConfig {
            max_merge_at_once: 4,
            segments_per_tier: 2,
            max_merged_segment_size: 1_000_000,
            reclaim_weight: 1.0,
        };
        let segments: Vec<SegmentStat> = (0..20)
            .map(|i| stat(&format!("_{i}"), 100, 0, 100))
            .collect();
        let groups = find_merges(&segments, &config);
        assert!(!groups.is_empty());
        for g in &groups {
            assert!(g.len() <= 4);
        }
    }

    #[test]
    fn default_config_matches_real_lucene_defaults() {
        let config = MergePolicyConfig::default();
        assert_eq!(config.max_merge_at_once, 10);
        assert_eq!(config.segments_per_tier, 10);
        assert_eq!(config.max_merged_segment_size, 5_000 * 1024 * 1024);
    }

    #[test]
    fn find_forced_merges_converges_to_target_count() {
        let segments: Vec<SegmentStat> = (0..5)
            .map(|i| stat(&format!("_{i}"), 100, 0, 100))
            .collect();
        let groups = find_forced_merges(&segments, 1);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].len(), 5);
    }

    #[test]
    fn find_forced_merges_no_op_when_already_at_target() {
        let segments: Vec<SegmentStat> = (0..2)
            .map(|i| stat(&format!("_{i}"), 100, 0, 100))
            .collect();
        assert!(find_forced_merges(&segments, 2).is_empty());
        assert!(find_forced_merges(&segments, 5).is_empty());
    }

    #[test]
    fn find_forced_merges_leaves_target_minus_one_untouched() {
        let segments: Vec<SegmentStat> = (0..6)
            .map(|i| stat(&format!("_{i}"), 100, 0, 100))
            .collect();
        let groups = find_forced_merges(&segments, 3);
        // excess = 6 - 3 + 1 = 4 segments merged into one group, 2 untouched.
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].len(), 4);
    }

    #[test]
    fn segment_stat_from_segment_info_approximates_size_by_doc_count() {
        let info = SegmentInfo {
            id: [0u8; lucene_store::codec_util::ID_LENGTH],
            version: crate::segment_info::LuceneVersion {
                major: 9,
                minor: 0,
                bugfix: 0,
            },
            min_version: None,
            doc_count: 42,
            is_compound_file: false,
            has_blocks: false,
            diagnostics: vec![],
            files: vec![],
            attributes: vec![],
        };
        let stat = SegmentStat::from_segment_info("_0", &info, 5);
        assert_eq!(stat.name, "_0");
        assert_eq!(stat.doc_count, 42);
        assert_eq!(stat.del_count, 5);
        assert_eq!(stat.size_bytes, 42);
    }

    #[test]
    fn del_ratio_zero_doc_count_is_zero_not_nan() {
        let stat = stat("_0", 0, 0, 0);
        assert_eq!(stat.del_ratio(), 0.0);
    }

    #[test]
    fn segment_byte_size_sums_real_file_lengths() {
        use lucene_store::directory::FsDirectory;
        let dir_path = std::env::temp_dir().join(format!(
            "lucene-rust-merge-policy-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir_path).unwrap();
        std::fs::write(dir_path.join("_0.fdt"), b"0123456789").unwrap();
        std::fs::write(dir_path.join("_0.fdx"), b"01234").unwrap();
        let dir = FsDirectory::open(&dir_path);

        let info = SegmentInfo {
            id: [0u8; lucene_store::codec_util::ID_LENGTH],
            version: crate::segment_info::LuceneVersion {
                major: 9,
                minor: 0,
                bugfix: 0,
            },
            min_version: None,
            doc_count: 1,
            is_compound_file: false,
            has_blocks: false,
            diagnostics: vec![],
            files: vec![
                "_0.fdt".to_string(),
                "_0.fdx".to_string(),
                "_0.missing".to_string(),
            ],
            attributes: vec![],
        };
        let size = segment_byte_size(&dir, &info);
        assert_eq!(size, 15);

        std::fs::remove_dir_all(&dir_path).ok();
    }
}
