//! Differential test for the pluggable merge policies
//! (`lucene_index::merge_policy::{api, log, temporal, filter}`) against real
//! Lucene.
//!
//! `fixtures/src/GenMergePolicies.java` runs `LogByteSizeMergePolicy`,
//! `LogDocMergePolicy`, `FilterMergePolicy`, `OneMergeWrappingMergePolicy`,
//! `TemporalMergePolicy` and `TieredMergePolicy` (through `findFullFlushMerges`,
//! `useCompoundFile` and `findForcedMerges`' `segmentsToMerge` flags) over a
//! table of segment descriptions, and a `TemporalMergePolicy` over a real
//! index whose date ranges it reads from the segments' own points. Every
//! scenario is replayed here through the Rust policy built from the same
//! description, and the merges must be identical, in the identical order.
//!
//! Regenerate with `scripts/gen-fixtures.sh --only GenMergePolicies`.
#![allow(clippy::arithmetic_side_effects)]

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use lucene_index::merge_policy::filter::{FilterMergePolicy, OneMergeWrappingMergePolicy};
use lucene_index::merge_policy::log::{LogByteSizeMergePolicy, LogDocMergePolicy, LogMergePolicy};
use lucene_index::merge_policy::temporal::{
    DirectoryDateRanges, SegmentDateRange, TemporalMergePolicy,
};
use lucene_index::merge_policy::{
    BasicMergeContext, MergePolicy, MergePolicyConfig, MergeSegment, MergeSpecification,
    MergeTrigger, OneMerge, TieredMergePolicy,
};
use lucene_store::directory::FsDirectory;

fn data(rel: &str) -> String {
    format!(
        "{}/../../fixtures/data/merge_policies/{rel}",
        env!("CARGO_MANIFEST_DIR")
    )
}

fn props(text: &str) -> HashMap<String, String> {
    text.lines()
        .filter(|l| !l.starts_with('#'))
        .filter_map(|l| l.split_once('='))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

fn kv(spec: &str) -> (String, HashMap<String, String>) {
    let mut parts = spec.split(';');
    let kind = parts.next().unwrap().to_string();
    let map = parts
        .map(|p| {
            let (k, v) = p.split_once('=').unwrap();
            (k.to_string(), v.to_string())
        })
        .collect();
    (kind, map)
}

fn mb(bytes: &str) -> f64 {
    bytes.parse::<i64>().unwrap() as f64 / (1024.0 * 1024.0)
}

fn apply_log(p: &mut LogMergePolicy, kv: &HashMap<String, String>) {
    if let Some(v) = kv.get("mergeFactor") {
        p.set_merge_factor(v.parse().unwrap()).unwrap();
    }
    if let Some(v) = kv.get("maxMergeDocs") {
        p.max_merge_docs = v.parse().unwrap();
    }
    if let Some(v) = kv.get("calibrate") {
        p.calibrate_size_by_deletes = v.parse().unwrap();
    }
    if let Some(v) = kv.get("tsc") {
        p.set_target_search_concurrency(v.parse().unwrap()).unwrap();
    }
    if let Some(v) = kv.get("noCFSRatio") {
        p.compound_file_settings_mut()
            .set_no_cfs_ratio(v.parse().unwrap())
            .unwrap();
    }
}

fn log_doc(kv: &HashMap<String, String>) -> LogDocMergePolicy {
    let mut p = LogDocMergePolicy::new();
    apply_log(&mut p, kv);
    if let Some(v) = kv.get("minDocs") {
        p.set_min_merge_docs(v.parse().unwrap());
    }
    p
}

fn build(spec: &str, ranges: &HashMap<String, SegmentDateRange>) -> Box<dyn MergePolicy> {
    let (kind, kv) = kv(spec);
    match kind.as_str() {
        "logbyte" => {
            let mut p = LogByteSizeMergePolicy::new();
            apply_log(&mut p, &kv);
            if let Some(v) = kv.get("minBytes") {
                p.set_min_merge_mb(mb(v));
            }
            if let Some(v) = kv.get("maxBytes") {
                p.set_max_merge_mb(mb(v));
            }
            if let Some(v) = kv.get("maxForcedBytes") {
                p.set_max_merge_mb_for_forced_merge(mb(v));
            }
            Box::new(p)
        }
        "logdoc" => Box::new(log_doc(&kv)),
        "filter" => Box::new(FilterMergePolicy::new(Box::new(log_doc(&kv)))),
        "wrapreverse" => Box::new(OneMergeWrappingMergePolicy::new(
            Box::new(log_doc(&kv)),
            Arc::new(|m: OneMerge| {
                let mut segments = m.segments;
                segments.reverse();
                OneMerge::new(segments)
            }),
        )),
        "tiered" => {
            let mut config = MergePolicyConfig::default();
            if let Some(v) = kv.get("maxMergedBytes") {
                config.max_merged_segment_size = v.parse().unwrap();
            }
            if let Some(v) = kv.get("floorBytes") {
                config.floor_segment_size = v.parse().unwrap();
            }
            if let Some(v) = kv.get("segsPerTier") {
                config.segments_per_tier = v.parse::<f64>().unwrap() as usize;
            }
            let mut p = TieredMergePolicy::new(config);
            if let Some(v) = kv.get("noCFSRatio") {
                p.compound_file_settings_mut()
                    .set_no_cfs_ratio(v.parse().unwrap())
                    .unwrap();
            }
            if let Some(v) = kv.get("maxCFSBytes") {
                p.compound_file_settings_mut()
                    .set_max_cfs_segment_size_mb(mb(v))
                    .unwrap();
            }
            Box::new(p)
        }
        "temporal" => {
            let mut p = temporal(&kv);
            if !ranges.is_empty() {
                p.set_segment_date_range_overrides(Some(ranges.clone()));
            }
            Box::new(p)
        }
        other => panic!("unknown policy {other}"),
    }
}

fn temporal(kv: &HashMap<String, String>) -> TemporalMergePolicy {
    let mut p = TemporalMergePolicy::new();
    p.set_temporal_field("ts").unwrap();
    if let Some(v) = kv.get("minThreshold") {
        p.set_min_threshold(v.parse().unwrap()).unwrap();
    }
    if let Some(v) = kv.get("maxThreshold") {
        p.set_max_threshold(v.parse().unwrap()).unwrap();
    }
    if let Some(v) = kv.get("baseTime") {
        p.set_base_time_in_seconds(v.parse().unwrap()).unwrap();
    }
    if let Some(v) = kv.get("maxWindow") {
        p.set_max_window_size_seconds(v.parse().unwrap()).unwrap();
    }
    if let Some(v) = kv.get("maxAge") {
        p.set_max_age_seconds(v.parse().unwrap()).unwrap();
    }
    if let Some(v) = kv.get("ratio") {
        p.set_compaction_ratio(v.parse().unwrap()).unwrap();
    }
    if let Some(v) = kv.get("forceDeletesPct") {
        p.set_force_merge_deletes_pct_allowed(v.parse().unwrap())
            .unwrap();
    }
    if kv.get("exponential").map(String::as_str) == Some("false") {
        p.disable_exponential_buckets();
    }
    p
}

/// Java's `render`: `null`, `` for an empty specification, or the groups.
fn render(spec: Option<MergeSpecification>, sort: bool) -> String {
    let Some(spec) = spec else {
        return "null".into();
    };
    let mut groups: Vec<String> = spec
        .groups()
        .into_iter()
        .map(|mut g| {
            if sort {
                g.sort();
            }
            g.join(",")
        })
        .collect();
    if sort {
        groups.sort();
    }
    groups.join("|")
}

fn run_op(
    policy: &dyn MergePolicy,
    op: &str,
    kind: &str,
    infos: &[MergeSegment],
    ctx: &BasicMergeContext,
    to_merge: &HashMap<String, bool>,
) -> String {
    if let Some(name) = op.strip_prefix("useCompoundFile:") {
        let merged = infos.iter().find(|s| s.name == name).unwrap();
        return policy
            .use_compound_file(infos, merged, ctx)
            .unwrap()
            .to_string();
    }
    match op {
        "findMerges" => render(
            policy
                .find_merges(MergeTrigger::Explicit, infos, ctx)
                .unwrap(),
            false,
        ),
        "findFullFlushMerges" => render(
            policy
                .find_full_flush_merges(MergeTrigger::FullFlush, infos, ctx)
                .unwrap(),
            false,
        ),
        "findForcedDeletesMerges" => render(
            policy.find_forced_deletes_merges(infos, ctx).unwrap(),
            false,
        ),
        _ => {
            let n: i32 = op
                .strip_prefix("findForcedMerges:")
                .unwrap_or_else(|| panic!("unknown op {op}"))
                .parse()
                .unwrap();
            render(
                policy.find_forced_merges(infos, n, to_merge, ctx).unwrap(),
                kind == "temporal",
            )
        }
    }
}

#[test]
fn every_policy_scenario_matches_lucene() {
    let p = props(&std::fs::read_to_string(data("merge_policies.manifest.properties")).unwrap());
    let count: usize = p["scenarios"].parse().unwrap();
    assert!(
        count >= 50,
        "the generator writes more scenarios than {count}"
    );
    let mut failures = Vec::new();
    for i in 0..count {
        let get = |k: &str| p[&format!("scenario.{i}.{k}")].clone();
        let name = get("name");
        let infos: Vec<MergeSegment> = get("segments")
            .split(';')
            .filter(|s| !s.is_empty())
            .map(|s| {
                let f: Vec<&str> = s.split(':').collect();
                MergeSegment::new(
                    f[0],
                    f[1].parse().unwrap(),
                    f[2].parse().unwrap(),
                    f[3].parse().unwrap(),
                )
                .with_compound_file(f[4] == "true")
            })
            .collect();
        let merging: HashSet<String> = get("merging")
            .split(',')
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect();
        let to_merge: HashMap<String, bool> = get("toMerge")
            .split(',')
            .filter(|s| !s.is_empty())
            .map(|s| {
                let (n, o) = s.split_once(':').unwrap();
                (n.to_string(), o == "true")
            })
            .collect();
        let ranges: HashMap<String, SegmentDateRange> = get("ranges")
            .split(';')
            .filter(|s| !s.is_empty())
            .map(|s| {
                let f: Vec<&str> = s.split(':').collect();
                (
                    f[0].to_string(),
                    SegmentDateRange {
                        min_date: f[1].parse().unwrap(),
                        max_date: f[2].parse().unwrap(),
                    },
                )
            })
            .collect();
        let spec = get("policy");
        let policy = build(&spec, &ranges);
        let kind = kv(&spec).0;
        let ctx = BasicMergeContext::new(merging);
        let got = run_op(policy.as_ref(), &get("op"), &kind, &infos, &ctx, &to_merge);
        let expected = get("expected");
        if got != expected {
            failures.push(format!(
                "{name} ({}): lucene={expected:?} rust={got:?}",
                get("op")
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {count} scenarios disagree:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn temporal_policy_reads_date_ranges_from_real_segments() {
    let dir = FsDirectory::open(data("temporal_index"));
    let infos = lucene_index::segment_infos::read_latest(&dir).unwrap();
    let p = props(&std::fs::read_to_string(data("temporal_index/manifest.properties")).unwrap());

    let mut segments = Vec::new();
    for sci in &infos.segments {
        let si_bytes =
            lucene_store::directory::Directory::open(&dir, &format!("{}.si", sci.segment_name))
                .unwrap();
        let si = lucene_index::segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
        segments.push(
            MergeSegment::new(sci.segment_name.clone(), si.doc_count, sci.del_count, 0)
                .with_compound_file(si.is_compound_file),
        );
    }
    let listed: Vec<String> = segments
        .iter()
        .map(|s| format!("{}:{}", s.name, s.max_doc))
        .collect();
    assert_eq!(listed.join(","), p["segments"]);

    let ranges = DirectoryDateRanges::read(&dir, &infos.segments, "ts");
    // The segment without the field has no range; the others all do, read
    // through `.cfs` for the compound ones.
    assert!(ranges.get("_c").is_none());
    for s in &segments[..12] {
        assert!(ranges.get(&s.name).is_some(), "no range for {}", s.name);
    }
    assert!(segments.iter().any(|s| s.use_compound_file));

    let ctx = BasicMergeContext::default();
    let all: HashMap<String, bool> = segments.iter().map(|s| (s.name.clone(), true)).collect();
    for case in [
        "natural",
        "natural_min3",
        "natural_exponential",
        "forced_one",
    ] {
        let (_, kv) = kv(&p[&format!("case.{case}.policy")]);
        let mut policy = temporal(&kv);
        policy.with_date_range_resolver(ranges.clone().into_resolver());
        let op = &p[&format!("case.{case}.op")];
        let got = run_op(&policy, op, "temporal", &segments, &ctx, &all);
        assert_eq!(got, p[&format!("case.{case}.expected")], "case {case}");
    }
}
