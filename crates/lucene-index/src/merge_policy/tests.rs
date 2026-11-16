#![cfg(test)]
// The arithmetic gate is about values read off disk; a fixture builder's
// own index arithmetic is not one (see `docs/arithmetic-gate.md`).
#![allow(clippy::arithmetic_side_effects)]

use super::*;

fn stat(name: &str, doc_count: i32, del_count: i32, size_bytes: u64) -> SegmentStat {
    SegmentStat {
        name: name.to_string(),
        doc_count,
        del_count,
        size_bytes,
    }
}

/// A byte-denominated config scaled down so that tests can use
/// three-digit "byte" sizes and still exercise the real thresholds.
fn small_config() -> MergePolicyConfig {
    MergePolicyConfig {
        max_merge_at_once: 3,
        segments_per_tier: 3.0,
        max_merged_segment_size: 1_000_000,
        floor_segment_size: 0,
        ..MergePolicyConfig::default()
    }
}

#[test]
fn default_config_matches_real_lucene_defaults() {
    let config = MergePolicyConfig::default();
    // TieredMergePolicy field initialisers, Lucene 10.5.0.
    assert_eq!(config.max_merged_segment_size, 5 * 1024 * 1024 * 1024);
    assert_eq!(config.floor_segment_size, 16 * 1024 * 1024);
    assert_eq!(config.segments_per_tier, 8.0);
    assert_eq!(config.max_merge_at_once, 10);
    // mergeFactor = (int) Math.min(maxMergeAtOnce, segsPerTier)
    assert_eq!(config.merge_factor(), 8);
    assert_eq!(config.force_merge_deletes_pct_allowed, 10.0);
    assert_eq!(config.deletes_pct_allowed, 20.0);
    assert_eq!(config.target_search_concurrency, 1);
    // score()'s hardcoded Math.pow(nonDelRatio, 2) exponent.
    assert_eq!(config.reclaim_weight, 2.0);
}

#[test]
fn prorated_size_matches_merge_policy_size() {
    // MergePolicy.size: byteSize * (1 - delCount/maxDoc), truncated.
    assert_eq!(prorated_size(&stat("_0", 100, 0, 1000)), 1000);
    assert_eq!(prorated_size(&stat("_0", 100, 25, 1000)), 750);
    assert_eq!(prorated_size(&stat("_0", 100, 100, 1000)), 0);
    // maxDoc <= 0 => raw size, no proration (and no divide by zero).
    assert_eq!(prorated_size(&stat("_0", 0, 0, 1234)), 1234);
}

#[test]
fn sorted_by_segment_size_is_size_descending_then_name() {
    let segments = vec![
        stat("_b", 100, 0, 100),
        stat("_a", 100, 0, 100),
        stat("_big", 100, 0, 500),
        // Pro-rated down to 100, so it ties with _a/_b on size.
        stat("_c", 100, 50, 200),
    ];
    let sorted = sorted_by_segment_size(&segments);
    let names: Vec<&str> = sorted.iter().map(|e| e.name()).collect();
    assert_eq!(names, vec!["_big", "_a", "_b", "_c"]);
}

#[test]
fn many_small_segments_propose_a_merge() {
    let segments: Vec<SegmentStat> = (0..8)
        .map(|i| stat(&format!("_{i}"), 100, 0, 100))
        .collect();
    let config = small_config();
    let groups = find_merges(&segments, &config);
    assert!(!groups.is_empty(), "expected at least one merge group");
    for g in &groups {
        assert!(g.len() >= 2, "no singleton merges without deletes: {g:?}");
        assert!(g.len() <= config.max_merge_at_once);
    }
    // Groups must be disjoint -- a segment can only be in one merge.
    let mut seen = HashSet::new();
    for g in &groups {
        for name in g {
            assert!(seen.insert(name.clone()), "segment {name} in two merges");
        }
    }
}

#[test]
fn oversized_segment_excluded_from_merges() {
    // Java excludes segments over HALF max_merged_segment_size (when
    // their deletes are within deletes_pct_allowed), not at/above it.
    let mut segments: Vec<SegmentStat> = (0..8)
        .map(|i| stat(&format!("_{i}"), 100, 0, 100))
        .collect();
    segments.push(stat("_huge", 100, 0, 600_000));
    let groups = find_merges(&segments, &small_config());
    assert!(!groups.is_empty());
    for g in &groups {
        assert!(!g.contains(&"_huge".to_string()));
    }
}

#[test]
fn over_half_max_size_segment_with_heavy_deletes_is_still_merged() {
    // The exclusion in step 2 only applies when the index's overall
    // delete percentage AND the segment's own are within
    // deletes_pct_allowed. A delete-heavy big segment in a delete-heavy
    // index stays eligible so its deletes can be reclaimed.
    let config = MergePolicyConfig {
        max_merge_at_once: 3,
        segments_per_tier: 3.0,
        max_merged_segment_size: 1_000_000,
        floor_segment_size: 0,
        deletes_pct_allowed: 20.0,
        ..MergePolicyConfig::default()
    };
    // Raw 1_000_000 with 50% deletes pro-rates to 500_000, which is not
    // > max/2, so it isn't excluded on size either -- use a raw size
    // whose pro-rated value still clears half the max.
    let segments = vec![
        stat("_big_dirty", 1000, 500, 2_000_000),
        stat("_a", 100, 50, 100),
        stat("_b", 100, 50, 100),
        stat("_c", 100, 50, 100),
        stat("_d", 100, 50, 100),
    ];
    let groups = find_merges(&segments, &config);
    assert!(
        groups.iter().any(|g| g.contains(&"_big_dirty".to_string())),
        "a delete-heavy over-half-max segment must stay eligible: {groups:?}"
    );
}

#[test]
fn already_optimal_segment_count_proposes_nothing() {
    let segments: Vec<SegmentStat> = (0..3)
        .map(|i| stat(&format!("_{i}"), 100, 0, 100))
        .collect();
    assert!(find_merges(&segments, &small_config()).is_empty());
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
fn merging_segments_are_excluded_from_the_eligible_set() {
    let segments: Vec<SegmentStat> = (0..8)
        .map(|i| stat(&format!("_{i}"), 100, 0, 100))
        .collect();
    let merging: HashSet<String> = ["_0".to_string(), "_1".to_string()].into_iter().collect();
    let groups = find_merges_excluding(&segments, &merging, &small_config());
    assert!(!groups.is_empty());
    for g in &groups {
        assert!(!g.contains(&"_0".to_string()), "{g:?}");
        assert!(!g.contains(&"_1".to_string()), "{g:?}");
    }
}

#[test]
fn merge_factor_cap_respected_unless_below_floor() {
    // maxMergeAtOnce = 10, segsPerTier = 4, so
    // mergeFactor = (int) min(10, 4) = 4. floor_segment_size == 0 turns
    // off the "keep packing while below the floor" escape hatch, so the
    // soft mergeFactor cap is what binds.
    let config = MergePolicyConfig {
        max_merge_at_once: 10,
        segments_per_tier: 4.0,
        max_merged_segment_size: 1_000_000,
        floor_segment_size: 0,
        ..MergePolicyConfig::default()
    };
    let segments: Vec<SegmentStat> = (0..20)
        .map(|i| stat(&format!("_{i:02}"), 100, 0, 100))
        .collect();
    let groups = find_merges(&segments, &config);
    assert!(!groups.is_empty());
    for g in &groups {
        assert!(g.len() <= 4, "{g:?}");
    }

    // With a floor above the whole merge, Java deliberately allows MORE
    // than mergeFactor inputs so tiny segments grow fast -- but never
    // more than maxMergeAtOnce, which stays a hard cap.
    let floored = MergePolicyConfig {
        floor_segment_size: 100_000,
        ..config
    };
    let groups = find_merges(&segments, &floored);
    assert!(
        groups.iter().any(|g| g.len() > 4),
        "below the floor, a merge may exceed mergeFactor: {groups:?}"
    );
    for g in &groups {
        assert!(
            g.len() <= config.max_merge_at_once,
            "maxMergeAtOnce is a hard cap even below the floor: {g:?}"
        );
    }
}

/// Java 10.5.0's `doFindMerges` bounds every candidate by
/// `candidate.size() < maxMergeAtOnce` *in addition to* the
/// `mergeFactor`-or-below-floor bound. Lucene `main` deleted the knob and
/// with it that hard bound; 10.5.0 still has it, so a below-floor run of
/// tiny segments packs exactly `maxMergeAtOnce` of them and no more.
#[test]
fn max_merge_at_once_is_a_hard_cap_below_the_floor() {
    let config = MergePolicyConfig {
        max_merge_at_once: 10,
        segments_per_tier: 2.0,
        max_merged_segment_size: 1_000_000,
        // Every candidate stays far below the floor, so only the hard
        // cap can stop the packing loop.
        floor_segment_size: 500_000,
        ..MergePolicyConfig::default()
    };
    let segments: Vec<SegmentStat> = (0..40)
        .map(|i| stat(&format!("_{i:02}"), 100, 0, 100))
        .collect();
    let groups = find_merges(&segments, &config);
    assert!(!groups.is_empty());
    assert!(
        groups.iter().any(|g| g.len() == 10),
        "expected a candidate packed to exactly maxMergeAtOnce: {groups:?}"
    );
    for g in &groups {
        assert!(g.len() <= 10, "exceeded maxMergeAtOnce: {g:?}");
    }
}

#[test]
fn max_merged_segment_size_caps_a_merge() {
    // Java-expressible config (`mergeFactor == (int) segsPerTier`): the
    // 1000-byte cap, not the merge factor of 10, is what limits each
    // group to three 300-byte segments.
    let config = MergePolicyConfig {
        max_merge_at_once: 10,
        segments_per_tier: 10.0,
        max_merged_segment_size: 1000,
        floor_segment_size: 0,
        ..MergePolicyConfig::default()
    };
    let segments: Vec<SegmentStat> = (0..30)
        .map(|i| stat(&format!("_{i:02}"), 100, 0, 300))
        .collect();
    let groups = find_merges(&segments, &config);
    assert!(!groups.is_empty());
    for g in &groups {
        assert!(
            g.len() * 300 <= 1000,
            "merge exceeded the byte cap, so the cap did not bind: {g:?}"
        );
        assert!(
            g.len() < config.max_merge_at_once,
            "the byte cap, not the merge factor, must be what bound: {g:?}"
        );
    }
}

/// `mergeFactor == (int) segsPerTier == 2`, i.e. a config real Lucene can
/// actually be set to, for the pair of growth-guard tests below.
fn growth_guard_config() -> MergePolicyConfig {
    MergePolicyConfig {
        max_merge_at_once: 2,
        segments_per_tier: 2.0,
        max_merged_segment_size: 1_000_000,
        floor_segment_size: 0,
        ..MergePolicyConfig::default()
    }
}

#[test]
fn pathological_growth_guard_skips_a_merge_that_barely_grows_the_biggest_input() {
    // Merging the 1000-byte segment with one 400-byte sibling would grow
    // it to 1400 -- less than 1.5x -- and it has no deletes to reclaim,
    // so Java skips every candidate that starts at it. The index is over
    // its segment budget, so merges do happen; they just never touch the
    // dominant segment.
    let mut segments = vec![stat("_dominant", 1000, 0, 1000)];
    segments.extend((0..8).map(|i| stat(&format!("_speck_{i}"), 100, 0, 400)));
    let groups = find_merges(&segments, &growth_guard_config());
    assert!(
        !groups.is_empty(),
        "the index is over budget, so something must merge: {groups:?}"
    );
    for g in &groups {
        assert!(
            !g.contains(&"_dominant".to_string()),
            "merging the dominant segment with a barely-bigger result is \
                 the O(N^2) pattern Java's 1.5x guard exists to prevent: {groups:?}"
        );
    }
}

#[test]
fn growth_guard_is_bypassed_when_the_biggest_input_is_delete_heavy() {
    // Same 1000-byte pro-rated dominant segment and the same <1.5x
    // growth, but now it is 50% deleted -- well over deletes_pct_allowed
    // -- so Java takes the merge anyway to reclaim those deletes. (The
    // deletes are also what pushes the index over budget here: two
    // 400-byte siblings alone would not.)
    let segments = vec![
        stat("_dominant", 1000, 500, 2000),
        stat("_speck_0", 100, 0, 400),
        stat("_speck_1", 100, 0, 400),
    ];
    let groups = find_merges(&segments, &growth_guard_config());
    assert!(
        groups.iter().any(|g| g.contains(&"_dominant".to_string())),
        "a delete-heavy biggest input bypasses the 1.5x guard: {groups:?}"
    );
}

#[test]
fn deletes_alone_can_trigger_a_merge_within_the_segment_count_budget() {
    // Two segments, well within any segment-count budget, but 60% of the
    // doc-id space is deleted -- over deletes_pct_allowed (20%) -- so
    // Java keeps merging until the delete budget is met.
    let config = MergePolicyConfig {
        max_merge_at_once: 10,
        segments_per_tier: 10.0,
        max_merged_segment_size: 1_000_000,
        floor_segment_size: 0,
        deletes_pct_allowed: 20.0,
        ..MergePolicyConfig::default()
    };
    let segments = vec![stat("_a", 100, 60, 1000), stat("_b", 100, 60, 1000)];
    let groups = find_merges(&segments, &config);
    assert!(
        !groups.is_empty(),
        "over-budget deletes must trigger a merge even at a fine \
             segment count: {groups:?}"
    );

    // The same two segments with no deletes are left alone.
    let clean = vec![stat("_a", 100, 0, 1000), stat("_b", 100, 0, 1000)];
    assert!(find_merges(&clean, &config).is_empty());
}

#[test]
fn target_search_concurrency_bounds_merged_doc_count() {
    // Java-expressible config (`mergeFactor == (int) segsPerTier == 10`)
    // with target_search_concurrency = 4: no merge may produce a segment
    // holding more than ceil(liveDocs / 4) docs, and here that cap (4
    // segments) binds well before the merge factor (10) does.
    let config = MergePolicyConfig {
        max_merge_at_once: 10,
        segments_per_tier: 10.0,
        max_merged_segment_size: 10_000_000,
        floor_segment_size: 0,
        target_search_concurrency: 4,
        ..MergePolicyConfig::default()
    };
    let segments: Vec<SegmentStat> = (0..16)
        .map(|i| stat(&format!("_{i:02}"), 100, 0, 1000))
        .collect();
    let groups = find_merges(&segments, &config);
    let allowed = max_allowed_docs(1600, 0, 4);
    assert_eq!(allowed, 400);
    assert!(!groups.is_empty());
    for g in &groups {
        assert!(
            (g.len() * 100) as i64 <= allowed,
            "merge of {} docs exceeds allowed_doc_count {allowed}: {g:?}",
            g.len() * 100
        );
        assert!(
            g.len() < config.max_merge_at_once,
            "the doc-count cap, not the merge factor, must be what bound: {g:?}"
        );
    }
}

#[test]
fn max_allowed_docs_is_ceiling_division() {
    assert_eq!(max_allowed_docs(100, 0, 1), 100);
    assert_eq!(max_allowed_docs(100, 0, 3), 34);
    assert_eq!(max_allowed_docs(100, 10, 3), 30);
    assert_eq!(max_allowed_docs(0, 0, 4), 0);
}

#[test]
fn a_target_search_concurrency_above_i64_max_does_not_invert_the_doc_budget() {
    // `target_search_concurrency` is a `pub usize` config. `as i64` turns
    // anything above `i64::MAX` **negative** -- `usize::MAX as i64` is
    // `-1` -- and `100.div_euclid(-1)` is `-100`: a *negative* document
    // budget, which every candidate merge then exceeds, so the policy
    // silently stops proposing merges for the whole index. The same cast
    // makes `i64::MIN.div_euclid(-1)` a panic in release as well as debug.
    //
    // Java's `Math.ceilDiv(100, Integer.MAX_VALUE)` is `1`; clamping the
    // divisor is what reproduces it.
    assert_eq!(max_allowed_docs(100, 0, usize::MAX), 1);
    assert_eq!(max_allowed_docs(100, 0, (i64::MAX as usize) + 1), 1);
    // And the ordinary cases are unchanged.
    assert_eq!(max_allowed_docs(100, 0, 1), 100);
    assert_eq!(max_allowed_docs(0, 0, usize::MAX), 0);
    // A `total_del_docs` above `total_max_doc` (two files disagreeing)
    // must not wrap: Java's `ceilDiv` of a negative numerator is negative,
    // and so is this.
    assert_eq!(max_allowed_docs(10, 100, 1), -90);
}

#[test]
fn a_segment_size_above_i64_max_is_treated_as_huge_not_as_negative() {
    // `SegmentStat::size_bytes` is a `pub u64`. A bare `as i64` on a value
    // above `i64::MAX` yields a *negative* byte count, and every signed
    // budget in this module then treats the largest segment in the index
    // as one that costs nothing.
    //
    // Two such segments make `bytes_this_merge + seg_bytes` overflow
    // outright (`i64::MIN + i64::MIN`), which is a panic in a debug build
    // -- this test aborts without the fix.
    let config = MergePolicyConfig::default();
    let huge = |name: &str| stat(name, 100, 50, u64::MAX);
    let segments = vec![huge("_0"), huge("_1"), stat("_2", 100, 50, 1_000)];

    let spec = find_merges(&segments, &config);
    assert!(
        !spec.is_empty(),
        "a delete-heavy index still has merges to propose"
    );
    assert_eq!(
        spec[0],
        vec!["_0".to_string()],
        "an over-i64::MAX segment is its own too-large merge, not a free \
             rider packed in with the rest"
    );

    // `findForcedMerges` reads the *raw* size for its bin packing, and the
    // same cast there decided whether a delete-free giant is dropped as
    // "already over the max size". `-1 >= maxMergeBytes` is false, so the
    // unfixed code kept it and packed it into a bin.
    let clean = vec![
        stat("_0", 100, 0, u64::MAX),
        stat("_1", 100, 0, 1_000),
        stat("_2", 100, 0, 1_000),
        stat("_3", 100, 0, 1_000),
    ];
    let forced = find_forced_merges(&clean, 2, &config);
    assert!(
        forced
            .iter()
            .all(|group| !group.contains(&"_0".to_string())),
        "a delete-free segment above the max merged size is dropped, not \
             bin-packed: {forced:?}"
    );
}

#[test]
fn merge_score_matches_the_java_formula() {
    let config = MergePolicyConfig {
        floor_segment_size: 0,
        reclaim_weight: 2.0,
        segments_per_tier: 8.0,
        ..MergePolicyConfig::default()
    };
    // Two clean segments, sizes 400 and 100 (raw == pro-rated).
    let a = stat("_a", 100, 0, 400);
    let b = stat("_b", 100, 0, 100);
    let candidate = vec![
        SegmentSizeAndDocs {
            stat: &a,
            size_in_bytes: 400,
        },
        SegmentSizeAndDocs {
            stat: &b,
            size_in_bytes: 100,
        },
    ];
    // skew = 400/500 = 0.8; totAfter = 500; nonDelRatio = 500/500 = 1.
    let expected = 0.8 * 500f64.powf(0.05) * 1f64.powf(2.0);
    let got = merge_score(&candidate, false, &config);
    assert!((got - expected).abs() < 1e-12, "{got} vs {expected}");

    // hitTooLarge pretends perfect skew: 1/segsPerTier.
    let expected_too_large = (1.0 / 8.0) * 500f64.powf(0.05);
    let got = merge_score(&candidate, true, &config);
    assert!(
        (got - expected_too_large).abs() < 1e-12,
        "{got} vs {expected_too_large}"
    );
}

#[test]
fn merge_score_favours_reclaiming_deletes() {
    let config = MergePolicyConfig {
        floor_segment_size: 0,
        ..MergePolicyConfig::default()
    };
    // Same pro-rated sizes, but the second pair got there by deleting
    // half of a twice-as-big segment -- a much better merge.
    let clean_a = stat("_a", 100, 0, 250);
    let clean_b = stat("_b", 100, 0, 250);
    let dirty_a = stat("_c", 100, 50, 500);
    let dirty_b = stat("_d", 100, 50, 500);
    let clean = vec![
        SegmentSizeAndDocs {
            stat: &clean_a,
            size_in_bytes: 250,
        },
        SegmentSizeAndDocs {
            stat: &clean_b,
            size_in_bytes: 250,
        },
    ];
    let dirty = vec![
        SegmentSizeAndDocs {
            stat: &dirty_a,
            size_in_bytes: 250,
        },
        SegmentSizeAndDocs {
            stat: &dirty_b,
            size_in_bytes: 250,
        },
    ];
    assert!(
        merge_score(&dirty, false, &config) < merge_score(&clean, false, &config),
        "reclaiming deletes must score better (lower)"
    );

    // reclaim_weight == 0.0 disables that preference entirely.
    let no_reclaim = MergePolicyConfig {
        reclaim_weight: 0.0,
        ..config
    };
    assert_eq!(
        merge_score(&dirty, false, &no_reclaim),
        merge_score(&clean, false, &no_reclaim)
    );
}

#[test]
fn merge_score_favours_lower_skew() {
    let config = MergePolicyConfig {
        floor_segment_size: 0,
        ..MergePolicyConfig::default()
    };
    let bal_a = stat("_a", 100, 0, 250);
    let bal_b = stat("_b", 100, 0, 250);
    let skew_a = stat("_c", 100, 0, 490);
    let skew_b = stat("_d", 100, 0, 10);
    let balanced = vec![
        SegmentSizeAndDocs {
            stat: &bal_a,
            size_in_bytes: 250,
        },
        SegmentSizeAndDocs {
            stat: &bal_b,
            size_in_bytes: 250,
        },
    ];
    let lopsided = vec![
        SegmentSizeAndDocs {
            stat: &skew_a,
            size_in_bytes: 490,
        },
        SegmentSizeAndDocs {
            stat: &skew_b,
            size_in_bytes: 10,
        },
    ];
    assert!(merge_score(&balanced, false, &config) < merge_score(&lopsided, false, &config));
}

#[test]
fn floor_size_is_a_max_clamp() {
    assert_eq!(floor_size(400, 1000), 1000);
    assert_eq!(floor_size(1000, 1000), 1000);
    assert_eq!(floor_size(1400, 1000), 1400);
    assert_eq!(floor_size(400, 0), 400);
}

#[test]
fn floor_segment_size_flattens_skew_among_tiny_segments() {
    // Below the floor every segment scores as the floor, so skew for any
    // pair of tiny segments collapses to the same value regardless of
    // their raw size difference.
    let config = MergePolicyConfig {
        floor_segment_size: 1000,
        reclaim_weight: 0.0,
        ..MergePolicyConfig::default()
    };
    let a = stat("_a", 100, 0, 900);
    let b = stat("_b", 100, 0, 100);
    let c = stat("_c", 100, 0, 500);
    let d = stat("_d", 100, 0, 500);
    let uneven = vec![
        SegmentSizeAndDocs {
            stat: &a,
            size_in_bytes: 900,
        },
        SegmentSizeAndDocs {
            stat: &b,
            size_in_bytes: 100,
        },
    ];
    let even = vec![
        SegmentSizeAndDocs {
            stat: &c,
            size_in_bytes: 500,
        },
        SegmentSizeAndDocs {
            stat: &d,
            size_in_bytes: 500,
        },
    ];
    // Both floor to 1000 + 1000, biggest floors to 1000 => skew 0.5 for
    // both; only the totAfterMergeBytes^0.05 term (unfloored) differs,
    // and both totals are 1000.
    assert_eq!(
        merge_score(&uneven, false, &config),
        merge_score(&even, false, &config)
    );

    // Without the floor, the lopsided pair scores strictly worse.
    let no_floor = MergePolicyConfig {
        floor_segment_size: 0,
        ..config
    };
    assert!(merge_score(&even, false, &no_floor) < merge_score(&uneven, false, &no_floor));
}

#[test]
fn find_forced_merges_down_to_one_segment() {
    let segments: Vec<SegmentStat> = (0..5)
        .map(|i| stat(&format!("_{i}"), 100, 0, 100))
        .collect();
    let groups = find_forced_merges(&segments, 1, &MergePolicyConfig::default());
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].len(), 5);
}

#[test]
fn find_forced_merges_no_op_when_already_at_target() {
    let segments: Vec<SegmentStat> = (0..2)
        .map(|i| stat(&format!("_{i}"), 100, 0, 100))
        .collect();
    let config = MergePolicyConfig::default();
    assert!(find_forced_merges(&segments, 2, &config).is_empty());
    assert!(find_forced_merges(&segments, 5, &config).is_empty());
}

#[test]
fn find_forced_merges_single_clean_segment_to_one_is_a_no_op() {
    // Java bails when maxSegmentCount == 1 and the single remaining
    // segment is already merged (no deletes, right compound-file mode).
    let segments = vec![stat("_0", 100, 0, 100)];
    assert!(find_forced_merges(&segments, 1, &MergePolicyConfig::default()).is_empty());
}

#[test]
fn find_forced_merges_rewrites_a_lone_deleted_segment() {
    // With deletes present Java does NOT bail, and merges down to one.
    let segments = vec![stat("_0", 100, 40, 100)];
    let groups = find_forced_merges(&segments, 1, &MergePolicyConfig::default());
    assert_eq!(groups, vec![vec!["_0".to_string()]]);
}

#[test]
fn find_forced_merges_bin_packs_from_the_smallest_end() {
    // Java walks the size-descending list backwards, so the smallest
    // segments are merged first and the biggest are left alone.
    let segments = vec![
        stat("_big", 100, 0, 900),
        stat("_mid", 100, 0, 500),
        stat("_small_a", 100, 0, 100),
        stat("_small_b", 100, 0, 100),
    ];
    let config = MergePolicyConfig {
        max_merged_segment_size: 1000,
        ..MergePolicyConfig::default()
    };
    let groups = find_forced_merges(&segments, 3, &config);
    assert_eq!(groups.len(), 1);
    assert_eq!(
        groups[0],
        vec!["_small_b".to_string(), "_small_a".to_string()],
        "expected the two smallest, smallest-first: {groups:?}"
    );
}

#[test]
fn find_forced_merges_drops_oversized_clean_segments() {
    // A delete-free segment already at/above maxMergeBytes is dropped
    // from consideration (it's already as merged as it needs to be).
    let config = MergePolicyConfig {
        max_merged_segment_size: 1000,
        ..MergePolicyConfig::default()
    };
    let segments = vec![
        // maxMergeBytes = max(totalBytes/2, 1000) * 1.25.
        stat("_huge", 100, 0, 100_000),
        stat("_a", 100, 0, 100),
        stat("_b", 100, 0, 100),
        stat("_c", 100, 0, 100),
    ];
    let groups = find_forced_merges(&segments, 2, &config);
    for g in &groups {
        assert!(!g.contains(&"_huge".to_string()), "{g:?}");
    }
}

#[test]
fn find_forced_merges_empty_input() {
    assert!(find_forced_merges(&[], 1, &MergePolicyConfig::default()).is_empty());
}

#[test]
fn find_forced_delete_merges_selects_only_over_threshold_segments() {
    // pct_deletes: _clean=0%, _light=5%, _heavy=50%, _mostly_gone=90%.
    // Threshold 10.0 -> only _heavy and _mostly_gone qualify.
    let segments = vec![
        stat("_clean", 100, 0, 100),
        stat("_light", 100, 5, 100),
        stat("_heavy", 100, 50, 100),
        stat("_mostly_gone", 100, 90, 100),
    ];
    let config = MergePolicyConfig {
        floor_segment_size: 0,
        ..MergePolicyConfig::default()
    };
    let groups = find_forced_delete_merges(&segments, &config);
    let selected: HashSet<&String> = groups.iter().flatten().collect();
    assert_eq!(selected.len(), 2, "{groups:?}");
    assert!(selected.contains(&"_heavy".to_string()));
    assert!(selected.contains(&"_mostly_gone".to_string()));
    assert!(!selected.contains(&"_clean".to_string()));
    assert!(!selected.contains(&"_light".to_string()));
}

#[test]
fn find_forced_delete_merges_boundary_at_exact_threshold_is_excluded() {
    // Real Lucene's condition is `pctDeletes > forceMergeDeletesPctAllowed`
    // (strictly greater), so a segment exactly at the threshold is not
    // selected.
    let segments = vec![
        stat("_at_threshold", 100, 10, 100),
        stat("_just_over", 1000, 101, 100),
    ];
    let config = MergePolicyConfig {
        floor_segment_size: 0,
        ..MergePolicyConfig::default()
    };
    let groups = find_forced_delete_merges(&segments, &config);
    assert_eq!(groups, vec![vec!["_just_over".to_string()]]);
}

#[test]
fn find_forced_delete_merges_zero_deletions_selects_nothing() {
    let segments: Vec<SegmentStat> = (0..5)
        .map(|i| stat(&format!("_{i}"), 100, 0, 100))
        .collect();
    assert!(find_forced_delete_merges(&segments, &MergePolicyConfig::default()).is_empty());
}

#[test]
fn find_forced_delete_merges_single_qualifying_segment_still_selected() {
    // Unlike find_merges, a lone over-threshold segment is worth
    // rewriting on its own to drop its deletes.
    let segments = vec![stat("_clean", 100, 0, 100), stat("_heavy", 100, 90, 100)];
    let config = MergePolicyConfig {
        floor_segment_size: 0,
        ..MergePolicyConfig::default()
    };
    let groups = find_forced_delete_merges(&segments, &config);
    assert_eq!(groups, vec![vec!["_heavy".to_string()]]);
}

#[test]
fn find_forced_delete_merges_respects_max_merged_segment_size() {
    // Java's class javadoc: "findForcedDeletesMerges should never produce
    // segments greater than maxSegmentSize." The old simplified
    // implementation returned one unbounded group; this must not.
    let config = MergePolicyConfig {
        max_merged_segment_size: 1000,
        floor_segment_size: 0,
        ..MergePolicyConfig::default()
    };
    let segments: Vec<SegmentStat> = (0..10)
        .map(|i| stat(&format!("_{i}"), 100, 50, 800))
        .collect();
    let groups = find_forced_delete_merges(&segments, &config);
    assert!(!groups.is_empty());
    for g in &groups {
        // Each input pro-rates to 400 bytes, so at most two fit under
        // the 1000-byte cap.
        assert!(
            g.len() * 400 <= 1000,
            "forced-deletes merge exceeded max_merged_segment_size: {g:?}"
        );
    }
    // Every over-threshold segment ends up in exactly one merge.
    let mut seen = HashSet::new();
    for g in &groups {
        for name in g {
            assert!(seen.insert(name.clone()), "{name} selected twice");
        }
    }
}

#[test]
fn find_forced_delete_merges_excludes_merging_segments() {
    let segments = vec![stat("_a", 100, 90, 100), stat("_b", 100, 90, 100)];
    let config = MergePolicyConfig {
        floor_segment_size: 0,
        ..MergePolicyConfig::default()
    };
    let merging: HashSet<String> = ["_a".to_string()].into_iter().collect();
    let groups = find_forced_delete_merges_excluding(&segments, &merging, &config);
    assert_eq!(groups, vec![vec!["_b".to_string()]]);

    // If every over-threshold segment is already merging there's no work.
    let merging: HashSet<String> = ["_a".to_string(), "_b".to_string()].into_iter().collect();
    assert!(find_forced_delete_merges_excluding(&segments, &merging, &config).is_empty());
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
        index_sort: None,
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
    assert_eq!(pct_deletes(&stat), 0.0);
}

#[test]
fn degenerate_zero_size_segments_terminate() {
    // All-zero sizes would divide by zero in the level walk; the port
    // clamps level_size to at least 1 so this terminates instead of
    // spinning (Java can't hit it: real files are never zero bytes).
    let segments: Vec<SegmentStat> = (0..20).map(|i| stat(&format!("_{i}"), 100, 0, 0)).collect();
    let config = MergePolicyConfig {
        floor_segment_size: 0,
        max_merged_segment_size: 1000,
        ..MergePolicyConfig::default()
    };
    let _ = find_merges(&segments, &config);
}

#[test]
fn segments_per_tier_below_two_is_clamped() {
    let config = MergePolicyConfig {
        segments_per_tier: 0.0,
        ..small_config()
    };
    assert_eq!(config.segs_per_tier(), 2.0);
    let segments: Vec<SegmentStat> = (0..8)
        .map(|i| stat(&format!("_{i}"), 100, 0, 100))
        .collect();
    // Terminates (would spin with segs_per_tier == 0) and proposes work.
    assert!(!find_merges(&segments, &config).is_empty());
}

#[test]
fn segment_byte_size_sums_real_file_lengths() {
    use lucene_store::directory::FsDirectory;
    let dir_path = lucene_util::test_support::TempDir::new("merge-policy");
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
        index_sort: None,
    };
    let size = segment_byte_size(&dir, &info);
    assert_eq!(size, 15);

    // `SegmentCommitInfo.sizeInBytes` adds the files only the commit
    // names: the current `.liv`, and the generational field infos and
    // doc-values files; a file both lists name is counted once.
    std::fs::write(dir_path.join("_0_2.liv"), b"012").unwrap();
    std::fs::write(dir_path.join("_0_1.fnm"), b"0123456").unwrap();
    std::fs::write(dir_path.join("_0_1_Lucene90_0.dvd"), b"01").unwrap();
    let mut commit = crate::segment_infos::SegmentCommitInfo {
        segment_name: "_0".to_string(),
        ..Default::default()
    };
    assert_eq!(
        commit_byte_size(&dir, &commit, &info),
        15,
        "no deletes, no updates"
    );
    commit.del_gen = 2;
    commit.field_infos_files = vec!["_0_1.fnm".to_string(), "_0.fdt".to_string()];
    commit.dv_update_files = vec![(0, vec!["_0_1_Lucene90_0.dvd".to_string()])];
    assert_eq!(commit_byte_size(&dir, &commit, &info), 15 + 3 + 7 + 2);

    std::fs::remove_dir_all(&dir_path).ok();
}
