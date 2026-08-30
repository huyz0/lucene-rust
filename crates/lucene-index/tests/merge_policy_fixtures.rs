//! Differential test for `lucene_index::merge_policy` against real Lucene.
//!
//! `fixtures/src/GenMergePolicy.java` runs `TieredMergePolicy`'s three decision
//! entry points (`findMerges`, `findForcedMerges`, `findForcedDeletesMerges`)
//! over a table of `(config, segments, currently-merging set)` scenarios and
//! records exactly which segments Lucene chose to merge, in which groups, in
//! which order. This replays every scenario through the Rust port and asserts
//! the identical answer.
//!
//! This is the ground truth the port's own unit tests cannot provide: they were
//! written from the same reading of `TieredMergePolicy.java` that produced the
//! Rust, so a shared misreading -- a flipped comparison, a truncation in the
//! wrong place, a level-walk off-by-one -- would pass both. Here the expected
//! values come from running Lucene itself.
//!
//! Regenerate with `scripts/gen-fixtures.sh`.
// Test-support code opts out of the arithmetic gate at the file boundary:
// the gate exists for values read off disk in production decode paths, not
// for a fixture builder's own index arithmetic. See
// `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use std::collections::HashSet;

use lucene_index::merge_policy::{
    find_forced_delete_merges_excluding, find_forced_merges, find_merges_excluding,
    MergePolicyConfig, SegmentStat,
};

fn manifest_path() -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/merge_policy/merge_policy.manifest.properties"
    )
    .to_string()
}

/// Which `TieredMergePolicy` entry point a scenario exercises. The variants
/// deliberately keep Java's method names.
#[derive(Debug)]
#[allow(clippy::enum_variant_names)]
enum Op {
    FindMerges,
    FindForcedMerges(usize),
    FindForcedDeletesMerges,
}

#[derive(Debug)]
struct Scenario {
    name: String,
    op: Op,
    config: MergePolicyConfig,
    segments: Vec<SegmentStat>,
    merging: HashSet<String>,
    expected: Vec<Vec<String>>,
}

fn parse_manifest(text: &str) -> Vec<Scenario> {
    let kv: Vec<(&str, &str)> = text
        .lines()
        .filter(|l| !l.starts_with('#'))
        .filter_map(|l| l.split_once('='))
        .collect();
    let get = |key: &str| -> &str {
        kv.iter()
            .find(|(k, _)| *k == key)
            .unwrap_or_else(|| panic!("missing manifest key {key}"))
            .1
    };

    let count: usize = get("scenarios").parse().unwrap();
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let p = format!("scenario.{i}.");
        let name = get(&format!("{p}name")).to_string();

        let op_raw = get(&format!("{p}op"));
        let op = match op_raw.split_once(':') {
            Some(("findForcedMerges", n)) => Op::FindForcedMerges(n.parse().unwrap()),
            _ => match op_raw {
                "findMerges" => Op::FindMerges,
                "findForcedDeletesMerges" => Op::FindForcedDeletesMerges,
                other => panic!("unknown op {other}"),
            },
        };

        // maxMergedSegmentBytes,floorSegmentBytes,segsPerTier,
        // deletesPctAllowed,forceMergeDeletesPctAllowed,targetSearchConcurrency
        let cfg: Vec<&str> = get(&format!("{p}config")).split(',').collect();
        assert_eq!(cfg.len(), 6, "scenario {name}: bad config line");
        let segs_per_tier: usize = cfg[2].parse().unwrap();
        let config = MergePolicyConfig {
            max_merged_segment_size: cfg[0].parse().unwrap(),
            floor_segment_size: cfg[1].parse().unwrap(),
            segments_per_tier: segs_per_tier,
            // The generator leaves `maxMergeAtOnce` at Java's default of 10;
            // `mergeFactor` is then `(int) min(10, segsPerTier)`.
            max_merge_at_once: 10,
            deletes_pct_allowed: cfg[3].parse().unwrap(),
            force_merge_deletes_pct_allowed: cfg[4].parse().unwrap(),
            target_search_concurrency: cfg[5].parse().unwrap(),
            // score()'s hardcoded Math.pow(nonDelRatio, 2) exponent.
            reclaim_weight: 2.0,
        };

        let segs_raw = get(&format!("{p}segments"));
        let segments: Vec<SegmentStat> = if segs_raw.is_empty() {
            Vec::new()
        } else {
            segs_raw
                .split(';')
                .map(|entry| {
                    let f: Vec<&str> = entry.split(':').collect();
                    assert_eq!(f.len(), 4, "scenario {name}: bad segment entry {entry}");
                    SegmentStat {
                        name: f[0].to_string(),
                        doc_count: f[1].parse().unwrap(),
                        del_count: f[2].parse().unwrap(),
                        size_bytes: f[3].parse().unwrap(),
                    }
                })
                .collect()
        };

        let merging_raw = get(&format!("{p}merging"));
        let merging: HashSet<String> = if merging_raw.is_empty() {
            HashSet::new()
        } else {
            merging_raw.split(',').map(str::to_string).collect()
        };

        let expected_raw = get(&format!("{p}expected"));
        let expected: Vec<Vec<String>> = if expected_raw.is_empty() {
            Vec::new()
        } else {
            expected_raw
                .split('|')
                .map(|g| g.split(',').map(str::to_string).collect())
                .collect()
        };

        out.push(Scenario {
            name,
            op,
            config,
            segments,
            merging,
            expected,
        });
    }
    out
}

fn run(scenario: &Scenario) -> Vec<Vec<String>> {
    match scenario.op {
        Op::FindMerges => {
            find_merges_excluding(&scenario.segments, &scenario.merging, &scenario.config)
        }
        Op::FindForcedMerges(max_segment_count) => {
            assert!(
                scenario.merging.is_empty(),
                "the port's find_forced_merges models no concurrently running merge"
            );
            find_forced_merges(&scenario.segments, max_segment_count, &scenario.config)
        }
        Op::FindForcedDeletesMerges => find_forced_delete_merges_excluding(
            &scenario.segments,
            &scenario.merging,
            &scenario.config,
        ),
    }
}

#[test]
fn matches_real_lucene_tiered_merge_policy() {
    let text = std::fs::read_to_string(manifest_path())
        .expect("run scripts/gen-fixtures.sh first (GenMergePolicy)");
    let scenarios = parse_manifest(&text);
    assert!(
        scenarios.len() >= 30,
        "expected the full scenario table, got {}",
        scenarios.len()
    );

    let mut failures = Vec::new();
    for scenario in &scenarios {
        let got = run(scenario);
        if got != scenario.expected {
            failures.push(format!(
                "  {} ({:?})\n    lucene: {:?}\n    rust  : {:?}",
                scenario.name, scenario.op, scenario.expected, got
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} scenarios disagree with real Lucene:\n{}",
        failures.len(),
        scenarios.len(),
        failures.join("\n")
    );
}

/// The fixture is only ground truth if it actually exercises the interesting
/// branches: guard against a future edit that quietly reduces it to a handful
/// of no-op scenarios.
#[test]
fn fixture_covers_every_entry_point_and_both_outcomes() {
    let text = std::fs::read_to_string(manifest_path())
        .expect("run scripts/gen-fixtures.sh first (GenMergePolicy)");
    let scenarios = parse_manifest(&text);

    let natural = scenarios
        .iter()
        .filter(|s| matches!(s.op, Op::FindMerges))
        .count();
    let forced = scenarios
        .iter()
        .filter(|s| matches!(s.op, Op::FindForcedMerges(_)))
        .count();
    let forced_deletes = scenarios
        .iter()
        .filter(|s| matches!(s.op, Op::FindForcedDeletesMerges))
        .count();
    assert!(natural >= 15, "natural-merge scenarios: {natural}");
    assert!(forced >= 5, "forced-merge scenarios: {forced}");
    assert!(
        forced_deletes >= 5,
        "forced-deletes scenarios: {forced_deletes}"
    );

    assert!(
        scenarios.iter().any(|s| s.expected.is_empty()),
        "no scenario expects 'nothing to merge'"
    );
    assert!(
        scenarios.iter().any(|s| s.expected.len() > 1),
        "no scenario expects several simultaneous merges"
    );
    assert!(
        scenarios.iter().any(|s| !s.merging.is_empty()),
        "no scenario exercises the currently-merging exclusion"
    );
    assert!(
        scenarios
            .iter()
            .any(|s| s.segments.iter().any(|seg| seg.del_count > 0)),
        "no scenario has deleted docs, so nothing pro-rates or reclaims"
    );
    assert!(
        scenarios
            .iter()
            .any(|s| s.config.target_search_concurrency > 1),
        "no scenario exercises target_search_concurrency"
    );
    assert!(
        scenarios
            .iter()
            .any(|s| s.expected.iter().any(|g| g.len() == 1)),
        "no scenario produces a singleton (delete-reclaiming) merge"
    );
}
