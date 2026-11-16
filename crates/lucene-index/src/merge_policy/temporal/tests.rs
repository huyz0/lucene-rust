#![cfg(test)]
#![allow(clippy::arithmetic_side_effects)]
use super::super::api::BasicMergeContext;
use super::*;

const HOUR_MS: i64 = 3_600_000;

fn policy(ranges: &[(&str, i64)]) -> TemporalMergePolicy {
    let mut p = TemporalMergePolicy::new();
    p.set_temporal_field("ts").unwrap();
    p.disable_exponential_buckets();
    p.set_now_millis(Some(1_000 * HOUR_MS));
    p.set_segment_date_range_overrides(Some(
        ranges
            .iter()
            .map(|(n, max)| {
                (
                    n.to_string(),
                    SegmentDateRange {
                        min_date: max - 1,
                        max_date: *max,
                    },
                )
            })
            .collect(),
    ));
    p
}

fn seg(name: &str, docs: i32, del: i32) -> MergeSegment {
    MergeSegment::new(name, docs, del, 100)
}

#[test]
fn setters_validate_like_java() {
    let mut p = TemporalMergePolicy::new();
    assert!(p.set_temporal_field(" ").is_err());
    assert!(p.set_base_time_in_seconds(0).is_err());
    assert!(p.set_min_threshold(1).is_err());
    assert!(p.set_min_threshold(9).is_err());
    assert!(p.set_max_threshold(3).is_err());
    assert!(p.set_compaction_ratio(0.5).is_err());
    assert!(p.set_max_window_size_seconds(0).is_err());
    assert!(p.set_max_age_seconds(0).is_err());
    assert!(p.set_force_merge_deletes_pct_allowed(-1.0).is_err());
    p.set_temporal_field("t").unwrap();
    p.set_base_time_in_seconds(10).unwrap();
    p.set_max_threshold(12).unwrap();
    p.set_min_threshold(6).unwrap();
    p.set_compaction_ratio(1.0).unwrap();
    p.set_max_window_size_seconds(100).unwrap();
    p.set_max_age_seconds(1000).unwrap();
    p.set_force_merge_deletes_pct_allowed(5.0).unwrap();
    assert_eq!(p.temporal_field(), "t");
    assert_eq!(p.base_time_in_seconds(), 10);
    assert_eq!(p.min_threshold(), 6);
    assert_eq!(p.max_threshold(), 12);
    assert_eq!(p.compaction_ratio(), 1.0);
    assert_eq!(p.max_window_size_seconds(), 100);
    assert_eq!(p.max_age_seconds(), 1000);
    assert_eq!(p.force_merge_deletes_pct_allowed(), 5.0);
    assert!(p.use_exponential_buckets());
    assert!(format!("{p:?}").contains("temporal_field"));
}

#[test]
fn no_field_or_no_ranges_means_no_merges() {
    let ctx = BasicMergeContext::default();
    let infos = vec![seg("_a", 1, 0)];
    let p = TemporalMergePolicy::new();
    assert!(p
        .find_merges(MergeTrigger::Explicit, &infos, &ctx)
        .unwrap()
        .is_none());
    let p = policy(&[]);
    assert!(p
        .find_merges(MergeTrigger::Explicit, &infos, &ctx)
        .unwrap()
        .is_none());
    assert!(p
        .find_forced_merges(&infos, 1, &HashMap::new(), &ctx)
        .unwrap()
        .is_none());
    assert!(p
        .find_forced_deletes_merges(&infos, &ctx)
        .unwrap()
        .is_none());
    assert!(p
        .find_forced_merges(&infos, 0, &HashMap::new(), &ctx)
        .is_err());
}

#[test]
fn same_window_segments_merge_newest_first() {
    let ctx = BasicMergeContext::default();
    let names = ["_a", "_b", "_c", "_d", "_e"];
    let ranges: Vec<(&str, i64)> = names
        .iter()
        .enumerate()
        .map(|(i, n)| (*n, 10 * HOUR_MS + i as i64))
        .collect();
    let p = policy(&ranges);
    let infos: Vec<MergeSegment> = names.iter().map(|n| seg(n, 10, 0)).collect();
    let spec = p
        .find_merges(MergeTrigger::Explicit, &infos, &ctx)
        .unwrap()
        .unwrap();
    // Newest first; 1.2 * 10 = 12 docs reached at the minimum of 4.
    assert_eq!(spec.groups(), vec![vec!["_e", "_d", "_c", "_b"]]);
}

#[test]
fn bucket_sizes_grow_exponentially_and_cap() {
    let mut p = TemporalMergePolicy::new();
    // age 0 -> base window.
    assert_eq!(p.bucket_for_timestamp(7_300, 7_300), 7_200);
    // Old: the window grows by minThreshold until it covers the age.
    assert_eq!(p.bucket_for_timestamp(0, 4 * 3600), 0);
    p.set_max_age_seconds(10).unwrap();
    assert_eq!(p.bucket_for_timestamp(0, 11), -1);
    p.set_max_age_seconds(i64::MAX).unwrap();
    p.set_max_window_size_seconds(5_000).unwrap();
    assert_eq!(p.bucket_for_timestamp(0, 1_000_000_000), 0);
    assert_eq!(p.bucket_for_timestamp(12_000, 1_000_000_000), 10_000);
}

/// A base window whose growth overflows `i64`: `2^61 * 8` wraps to
/// zero, and the old `wrapping_mul` loop then spun forever.
#[test]
fn exponential_bucket_growth_stops_at_i64_overflow() {
    let mut p = TemporalMergePolicy::new();
    p.set_base_time_in_seconds(1 << 61).unwrap();
    p.set_min_threshold(8).unwrap();
    p.set_max_window_size_seconds(i64::MAX).unwrap();
    p.set_max_age_seconds(i64::MAX).unwrap();
    assert_eq!(p.bucket_for_timestamp(0, i64::MAX), 0);
    assert_eq!(p.bucket_for_timestamp((1 << 62) + 5, i64::MAX), 1 << 62);
}

#[test]
fn divisor_detects_units() {
    assert_eq!(to_millis(1, 2).max_date, 2_000);
    assert_eq!(to_millis(1, 200_000_000_000).max_date, 200_000_000_000);
    assert_eq!(
        to_millis(1_000, 200_000_000_000_000).max_date,
        200_000_000_000
    );
    assert_eq!(decode_long_dimension(&[0x80, 0, 0, 0, 0, 0, 0, 5]), Some(5));
    assert_eq!(decode_long_dimension(&[0x80]), None);
}

#[test]
fn resolver_and_directory_ranges() {
    let mut p = TemporalMergePolicy::new();
    p.set_temporal_field("ts").unwrap();
    p.set_now_millis(Some(0));
    let r = DirectoryDateRanges::default();
    assert!(r.get("_a").is_none());
    p.with_date_range_resolver(Arc::new(|s: &MergeSegment| {
        (s.name != "_z").then_some(SegmentDateRange {
            min_date: 0,
            max_date: 0,
        })
    }));
    let infos: Vec<MergeSegment> = ["_a", "_b", "_c", "_d", "_z"]
        .iter()
        .map(|n| seg(n, 1, 0))
        .collect();
    let spec = p
        .find_merges(
            MergeTrigger::Explicit,
            &infos,
            &BasicMergeContext::default(),
        )
        .unwrap()
        .unwrap();
    assert_eq!(spec.groups(), vec![vec!["_a", "_b", "_c", "_d"]]);
}
