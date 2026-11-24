#![cfg(test)]
use super::date_rounding::*;
use super::*;

/// The grid places every date in the bucket `DateRounding::bucket` gives
/// it, for each uniform rounding, zone offset and date (negative, huge, on
/// and next to a boundary), and is absent for a calendar month.
#[test]
fn date_grid_matches_rounding() {
    let zone = [0, 19_800_000, -25_200_000];
    let kinds = [
        RoundingKind::Interval(86_400_000),
        RoundingKind::Interval(7),
        RoundingKind::Interval(1_000_000_007),
        RoundingKind::Interval(1 << 20),
        RoundingKind::Unit(DateUnit::Day),
        RoundingKind::Unit(DateUnit::Hour),
        RoundingKind::Unit(DateUnit::Week),
        RoundingKind::Unit(DateUnit::Second),
    ];
    let mut x = 0x9E37_79B9_7F4A_7C15_u64;
    let mut next = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    for kind in kinds {
        for zone_ms in zone {
            for offset in [0, 3_600_000] {
                let r = DateRounding {
                    kind,
                    zone_ms,
                    region: None,
                    offset,
                };
                let first = 1_735_700_000_000_i64;
                let grid = DateGrid::new(&r, first).expect("uniform");
                let mut values = vec![i64::MIN, i64::MAX, 0, -1, first, first - 1];
                for _ in 0..4000 {
                    let spread =
                        [1_000_i64, 1_000_000_000, 100_000_000_000_000][(next() % 3) as usize];
                    values.push(first + (next() % (2 * spread as u64)) as i64 - spread);
                }
                for v in values {
                    let Some(p) = grid.position(v) else { continue };
                    let from = grid.start(p);
                    let Some((f, t)) = r.bucket(v) else { continue };
                    assert_eq!(
                        (from, from + grid.width),
                        (f, t),
                        "{kind:?} {zone_ms} {offset} {v}"
                    );
                }
            }
        }
    }
    let month = DateRounding {
        kind: RoundingKind::Unit(DateUnit::Month),
        zone_ms: 0,
        region: None,
        offset: 0,
    };
    assert!(DateGrid::new(&month, 1_735_700_000_000).is_none());
}

/// The ordinal table hands out what a map of `(owning, key)` to the next
/// ordinal would, across its growth.
#[test]
fn key_ords_number_keys_as_a_map_does() {
    key_ords_match_a_map(None);
}

/// With owner 0's keys below 5000 looked up densely, the rest hashed
/// (other owners, negatives, keys past the array), every key keeps one
/// bucket, and the table holds only its own keys.
#[test]
fn dense_key_ords_number_keys_as_a_map_does() {
    key_ords_match_a_map(Some(5000));
    assert!(KeyOrds::dense_pays(10_000, 2_000));
    assert!(!KeyOrds::dense_pays(10_000, 1_000));
    assert!(!KeyOrds::dense_pays(0, 1_000));
}

fn key_ords_match_a_map(dense: Option<usize>) {
    let mut ords = KeyOrds::default();
    if let Some(n) = dense {
        ords.densify(n);
        assert_eq!(ords.dense.len(), n);
    }
    let mut keys = Vec::new();
    let mut want: FastMap<(u32, i64), u32> = FastMap::default();
    let mut x = 0x2545_f491_4f6c_dd1du64;
    for _ in 0..200_000 {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        // Few owners, keys clustered and spread, negatives among them.
        let owning = (x % 3) as u32;
        let key = match x >> 62 {
            0 => (x >> 40) as i64 % 5000,
            1 => -((x >> 20) as i64 % 70_000),
            _ => (x >> 1) as i64,
        };
        let next = want.len() as u32;
        let w = *want.entry((owning, key)).or_insert(next);
        // A hint: asking early for any key, present or not, changes nothing.
        ords.prefetch(owning, key ^ 1);
        assert_eq!(ords.ord(&mut keys, owning, key), w);
    }
    assert!(ords.slots.len() >= KeyOrds::PREFETCH_SLOTS);
    KeyOrds::default().prefetch(0, 7);
    assert_eq!(keys.len(), want.len());
    for (&k, &o) in &want {
        assert_eq!(keys[o as usize], k);
    }
    assert!(ords.slots.len() * 5 >= ords.hashed * 8);
    let n = dense.unwrap_or(0);
    let in_dense = keys
        .iter()
        .filter(|&&(o, k)| o == 0 && usize::try_from(k).is_ok_and(|k| k < n))
        .count();
    assert_eq!(ords.hashed + in_dense, keys.len());
}

/// Bounded selection keeps what sorting every bucket and truncating
/// keeps, in every order, for sizes around the batch it holds.
#[test]
fn top_buckets_keep_what_a_full_sort_keeps() {
    let mut x = 0x9e37_79b9_7f4a_7c15u64;
    let buckets: Vec<(i64, u64, u32)> = (0..5000u32)
        .map(|b| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (i64::from(b) * 7 - 9000, x % 40, b)
        })
        .collect();
    for order in [
        TermsOrder::CountDesc,
        TermsOrder::CountAsc,
        TermsOrder::KeyAsc,
        TermsOrder::KeyDesc,
    ] {
        for n in [0, 1, 10, 31, 32, 33, 100, 4999, 5000, 9000] {
            let mut selected = buckets.clone();
            keep_best(&mut selected, n, order);
            selected.sort_unstable_by(|a, b| order.compare((a.0, a.1), (b.0, b.1)));
            let mut sorted = buckets.clone();
            sorted.sort_unstable_by(|a, b| order.compare((a.0, a.1), (b.0, b.1)));
            sorted.truncate(n);
            assert_eq!(selected, sorted, "keep_best {order:?} {n}");
            let mut all = buckets.clone();
            all.sort_unstable_by(|a, b| order.compare((a.0, a.1), (b.0, b.1)));
            all.truncate(n);
            let total: u64 = buckets.iter().map(|b| b.1).sum();
            assert_eq!(
                top_buckets(buckets.iter().copied(), n, order),
                (total, all),
                "{order:?} {n}"
            );
        }
    }
}

#[test]
fn histogram_buckets_sort_as_collection_util_sorts_them() {
    let sorted = |mut list: Vec<(f64, u64, u32)>| {
        let len = list.len();
        lucene_util::sorter::intro_sort(
            &mut ByBucketKey {
                list: &mut list,
                interval: 10.0,
                offset: 0.0,
                pivot: 0.0,
            },
            0,
            len,
        );
        list.iter().map(|k| k.2).collect::<Vec<_>>()
    };
    // `-0.0 * 10 + 0.0` is `0.0`: the two keys tie and keep their ord order.
    assert_eq!(
        sorted(vec![(0.0, 1, 0), (-0.0, 1, 1), (-1.0, 1, 2)]),
        [2, 0, 1]
    );
    assert_eq!(sorted(vec![(-0.0, 1, 0), (0.0, 1, 1)]), [0, 1]);
    use std::cmp::Ordering::{Equal, Greater, Less};
    assert_eq!(java_double_compare(-0.0, 0.0), Less);
    assert_eq!(java_double_compare(f64::NAN, f64::INFINITY), Greater);
    assert_eq!(java_double_compare(f64::NAN, -f64::NAN), Equal);
}

/// A leaf over no column whose window holds `vals` (one per document
/// from 0; `multi`: each document's values at `offsets`).
fn windowed(vals: Vec<i64>, multi: Option<Vec<u32>>) -> Leaf<'static> {
    let len = multi.as_ref().map_or(vals.len(), |o| o.len() - 1);
    Leaf {
        col: Col::None,
        buf: Vec::new(),
        max_to: Vec::new(),
        memo: (1, 0, 0, 0, u32::MAX),
        done: false,
        zoned: false,
        pre_base: 0,
        pre_len: len,
        pre_bits: vec![u64::MAX; len.div_ceil(64)],
        pre_vals: vals,
        pre_multi: multi.is_some(),
        pre_offsets: multi.unwrap_or_default(),
        win_docs: Vec::new(),
        win_ords: Vec::new(),
        term_buckets: Vec::new(),
        term_seen: 0,
        key_cache: KeyCache::default(),
        date_grid: None,
        grid_tried: false,
        calendar_buckets: Default::default(),
        zone_ords: Vec::new(),
        folded: 0,
        missing: None,
        missing_key: None,
        filter: None,
        composite: None,
        fed: None,
        nested: None,
        text: None,
        subs: Vec::new(),
    }
}

/// A multi-valued window answers `read` with each document's values
/// and never `prefetched` (a single value per document only).
#[test]
fn a_multi_valued_window_reads_each_documents_values() {
    let mut leaf = windowed(vec![4, 5, 9], Some(vec![0, 2, 2, 3]));
    assert!(leaf.read(0).unwrap());
    assert_eq!(leaf.buf, [4, 5]);
    assert!(!leaf.read(1).unwrap(), "no values");
    assert!(leaf.read(2).unwrap());
    assert_eq!(leaf.buf, [9]);
    assert_eq!(leaf.prefetched(0), None);
    let single = windowed(vec![7, 8], None);
    assert_eq!(single.prefetched(1), Some(Some(8)));
    assert_eq!(single.prefetched(2), None, "past the window");
}

/// The documents of a window [`Leaf::each_prefetched`] visits: straight
/// over the values only for the whole window with a value everywhere,
/// otherwise each document's own (a gap, a subset, a repeat, the bits
/// past the window's end ignored).
#[test]
fn each_prefetched_visits_the_documents_with_values() {
    let seen = |leaf: &Leaf<'_>, docs: &[i32]| {
        let mut out = Vec::new();
        leaf.each_prefetched(docs, |i, v| out.push((i, v)));
        out
    };
    let vals: Vec<i64> = (0..70).map(|v| v * 10).collect();
    let all: Vec<i32> = (0..70).collect();
    let mut leaf = windowed(vals.clone(), None);
    // The last word's bits past the window are unset, as a column leaves them.
    leaf.pre_bits[1] = (1 << 6) - 1;
    let want: Vec<(usize, i64)> = vals.iter().copied().enumerate().collect();
    assert_eq!(seen(&leaf, &all), want);
    leaf.pre_bits[0] &= !(1 << 3);
    let gap: Vec<(usize, i64)> = want.iter().copied().filter(|&(i, _)| i != 3).collect();
    assert_eq!(seen(&leaf, &all), gap, "a document without a value");
    leaf.pre_bits[0] |= 1 << 3;
    let mut repeat = all.clone();
    repeat[1] = 0;
    let mut want_repeat = want.clone();
    want_repeat[1] = (1, 0);
    assert_eq!(
        seen(&leaf, &repeat),
        want_repeat,
        "a repeat, as long as the window"
    );
    assert_eq!(seen(&leaf, &[2, 69, 70]), [(0, 20), (1, 690)], "a subset");
    leaf.pre_base = 100;
    let shifted: Vec<i32> = (100..170).collect();
    assert_eq!(seen(&leaf, &shifted), want, "a window from its base");
}

/// Each bucket arm's window collection places what `collect` places
/// document by document: the same keys in the same order, the same
/// counts and the same metric states, for owner 0's small keys (looked
/// up by index, again in a second window), large and negative ones,
/// `-0.0` beside `0.0`, NaN, owners past 0, with and without
/// sub-aggregations, over a whole window and one with a gap.
#[test]
fn window_collection_matches_collecting_each_document() {
    use metric_aggs::MetricSlot;
    let dbl = |v: f64| lucene_util::numeric_utils::double_to_sortable_long(v);
    let sub = AggNode::Metric {
        script: None,
        missing: None,
        field: "m".into(),
        kind: ValueKind::Long,
        source: Source::DocValues,
        needs: metric_aggs::NEED_COUNT | metric_aggs::NEED_SUM,
    };
    let terms = |kind, subs: &Vec<AggNode>| AggNode::NumericTerms {
        script: None,
        field: "f".into(),
        missing: None,
        kind,
        shard_size: 10,
        order: TermsOrder::CountDesc,
        subs: subs.clone(),
    };
    let histogram = |kind, interval, hard_bounds, subs: &Vec<AggNode>| AggNode::Histogram {
        script: None,
        field: "f".into(),
        missing: None,
        kind,
        interval,
        offset: 0.0,
        hard_bounds,
        subs: subs.clone(),
    };
    let range = |kind, ranges: Vec<(f64, f64)>, subs: &Vec<AggNode>| AggNode::Range {
        script: None,
        field: "f".into(),
        missing: None,
        kind,
        ranges,
        subs: subs.clone(),
    };
    let longs = vec![-1, 0, 5, 4095, 4096, i64::MAX, 5, 0, i64::MIN, 3];
    let doubles: Vec<i64> = [
        -0.0,
        0.0,
        0.5,
        4095.5,
        4096.0,
        -3.0,
        f64::NAN,
        -0.0,
        7.25,
        f64::NEG_INFINITY,
    ]
    .into_iter()
    .map(dbl)
    .collect();
    let many: Vec<(f64, f64)> = (0..20)
        .map(|i| (f64::from(i), f64::from(i) + 0.5))
        .collect();
    let disjoint = vec![(f64::NEG_INFINITY, 0.0), (0.0, 10.0), (10.0, 5000.0)];
    let overlapping = vec![(-5.0, 1.0), (0.0, 8.0), (4000.0, f64::INFINITY)];
    let (long, double) = (ValueKind::Long, ValueKind::Double);
    let docs: Vec<i32> = (0..10).collect();
    let owners_sets = [vec![0u32; 10], vec![0, 0, 1, 0, 2, 0, 1, 0, 0, 1]];
    let summary = |state: &State| -> (Vec<(u32, i64)>, Vec<u64>, String) {
        let (keys, counts, subs) = match state {
            State::Keyed {
                keys, docs, subs, ..
            } => (keys.clone(), docs.clone(), subs),
            State::Fixed { docs, subs, .. } => (Vec::new(), docs.clone(), subs),
            _ => panic!("a bucket state"),
        };
        let metrics: Vec<MetricState> = match subs.first() {
            Some(State::Metric(col)) => {
                with_metric_col!(col, v => v.iter().map(|s| s.state()).collect())
            }
            _ => Vec::new(),
        };
        (keys, counts, format!("{metrics:?}"))
    };
    for subs in [vec![sub.clone()], Vec::new()] {
        let cases: Vec<(AggNode, &Vec<i64>)> = vec![
            (terms(long, &subs), &longs),
            (terms(double, &subs), &doubles),
            (histogram(double, 1.0, (None, None), &subs), &doubles),
            (
                histogram(double, 0.25, (Some(-2.0), Some(4095.0)), &subs),
                &doubles,
            ),
            (histogram(long, 2.0, (None, None), &subs), &longs),
            (range(double, disjoint.clone(), &subs), &doubles),
            (range(double, overlapping.clone(), &subs), &doubles),
            (range(double, many.clone(), &subs), &doubles),
            (range(long, disjoint.clone(), &subs), &longs),
        ];
        for (node, vals) in &cases {
            for owners in &owners_sets {
                for gap in [false, true] {
                    let leaf_for = || {
                        let mut leaf = windowed(vals.to_vec(), None);
                        if gap {
                            leaf.pre_bits[0] &= !(1 << 8);
                        }
                        leaf.max_to = match node {
                            AggNode::Range { ranges, .. } => {
                                let mut max = f64::NEG_INFINITY;
                                ranges
                                    .iter()
                                    .map(|r| {
                                        max = max.max(r.1);
                                        max
                                    })
                                    .collect()
                            }
                            _ => Vec::new(),
                        };
                        if !subs.is_empty() {
                            leaf.subs = vec![windowed((0..10).map(|v| v * 3).collect(), None)];
                        }
                        leaf
                    };
                    let (mut window, mut each) = (State::new(node), State::new(node));
                    // Two windows into one state: the second finds the
                    // first's buckets.
                    let mut leaf = leaf_for();
                    for _ in 0..2 {
                        collect_window(node, &mut window, &mut leaf, &docs, owners).unwrap();
                    }
                    let mut leaf = leaf_for();
                    for _ in 0..2 {
                        for (&doc, &owner) in docs.iter().zip(owners) {
                            collect(node, &mut each, &mut leaf, doc, owner).unwrap();
                        }
                    }
                    let (w, e) = (summary(&window), summary(&each));
                    assert_eq!(w, e, "{owners:?} gap {gap} subs {}", subs.len());
                }
            }
        }
    }
    // `filters`: three filters (one matching everything), with and
    // without the `other` bucket, over owners past 0.
    for (other, subs) in [
        (false, vec![sub.clone()]),
        (true, vec![sub.clone()]),
        (true, Vec::new()),
    ] {
        let node = AggNode::Filters {
            filters: vec![BooleanQuery::default(); 3],
            other,
            subs: subs.clone(),
        };
        let bits = vec![
            Some(vec![0b0000_1010_0101u64]),
            None,
            Some(vec![0b0011_0000_0000u64]),
        ];
        for owners in &owners_sets {
            let leaf_for = || {
                let mut leaf = windowed(vec![0; 10], None);
                leaf.col = Col::Filters(bits.clone());
                if !subs.is_empty() {
                    leaf.subs = vec![windowed((0..10).map(|v| v * 3).collect(), None)];
                }
                leaf
            };
            let (mut window, mut each) = (State::new(&node), State::new(&node));
            let mut leaf = leaf_for();
            for _ in 0..2 {
                collect_window(&node, &mut window, &mut leaf, &docs, owners).unwrap();
            }
            let mut leaf = leaf_for();
            for _ in 0..2 {
                for (&doc, &owner) in docs.iter().zip(owners) {
                    collect(&node, &mut each, &mut leaf, doc, owner).unwrap();
                }
            }
            assert_eq!(
                summary(&window),
                summary(&each),
                "filters {other} {owners:?}"
            );
        }
    }
    assert!((-0.0f64).to_bits() >= SMALL_KEYS_F64_BITS);
    assert!(f64::NAN.to_bits() >= SMALL_KEYS_F64_BITS);
    assert!(4095.0f64.to_bits() < SMALL_KEYS_F64_BITS);
    assert!(4096.0f64.to_bits() >= SMALL_KEYS_F64_BITS);
}

/// A cardinality node whose state or column disagrees with its kind is
/// an internal inconsistency, reported rather than miscounted.
#[test]
fn a_cardinality_state_that_disagrees_with_its_kind_is_an_error() {
    let node = |kind| AggNode::Cardinality {
        field: "f".into(),
        kind,
        precision: None,
    };
    let numeric = node(CardinalityKind::Numeric(ValueKind::Long));
    let keyword = node(CardinalityKind::Keyword);
    let cases: [(&AggNode, State, &str); 3] = [
        (
            &numeric,
            State::OrdBits(Vec::new()),
            "without an ordinal column",
        ),
        (
            &keyword,
            State::Sketches {
                p: 14,
                sketches: Vec::new(),
            },
            "no direct sketch",
        ),
        (
            &keyword,
            State::Cardinality(Vec::new()),
            "counted as values",
        ),
    ];
    for (node, mut state, want) in cases {
        let mut leaf = windowed(vec![3], None);
        match collect(node, &mut state, &mut leaf, 0, 0) {
            Err(lucene_search::Error::TermsAggType(m)) => assert!(m.contains(want), "{m}"),
            other => panic!("{want}: {other:?}"),
        }
    }
    // A document without values collects nothing, whatever the state.
    let mut leaf = windowed(vec![3], Some(vec![0, 0]));
    let mut state = State::OrdBits(Vec::new());
    assert!(collect(&numeric, &mut state, &mut leaf, 0, 0).is_ok());
}

/// Only a metric answered from the points reads them, at any depth.
#[test]
fn reads_points_finds_a_points_metric() {
    let metric = |source| AggNode::Metric {
        script: None,
        missing: None,
        field: "n".into(),
        kind: ValueKind::Long,
        source,
        needs: metric_aggs::NEED_MIN,
    };
    assert!(metric(Source::PointsMin).reads_points());
    assert!(!metric(Source::DocValues).reads_points());
    let terms = |sub| AggNode::Terms {
        accepted: None,
        missing: None,
        zero_buckets: false,
        field: "k".into(),
        shard_size: 10,
        order: TermsOrder::CountDesc,
        subs: vec![sub],
    };
    assert!(terms(metric(Source::PointsMax)).reads_points());
    assert!(!terms(metric(Source::DocValues)).reads_points());
}

/// The hasher's byte and word paths agree with its per-word mixing, and
/// the month tables are `DateUtilsRounding`'s (checked at run time too:
/// the constants are built by a `const fn`).
#[test]
fn the_fx_hasher_and_month_tables() {
    let mut a = Fx::default();
    a.write(&[1, 2]);
    let mut b = Fx::default();
    b.write_u64(1);
    b.write_u64(2);
    assert_eq!(a.finish(), b.finish());
    let mut c = Fx::default();
    c.write_u32(7);
    let mut d = Fx::default();
    d.write_i64(7);
    assert_eq!(c.finish(), d.finish());
    assert_eq!(month_starts(false), MIN_TOTAL_MILLIS_BY_MONTH);
    assert_eq!(month_starts(true), MAX_TOTAL_MILLIS_BY_MONTH);
    assert_eq!(
        month_starts(true)[2] - month_starts(false)[2],
        MILLIS_PER_DAY
    );
}
use crate::metric_aggs::{MetricSpec, NEED_ALL};
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::query::{Clause, MatchAllDocsQuery, PointsRangeQuery, TermQuery};
use lucene_store::FsDirectory;

const DAY: i64 = MILLIS_PER_DAY;

/// Days since 1970-01-01 of a proleptic Gregorian date (Hinnant's
/// `days_from_civil`): an oracle independent of the Joda-style port.
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// `(year, month)` of a day count (`civil_from_days`).
fn civil_from_days(z: i64) -> (i64, i64) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (yoe + era * 400 + i64::from(m <= 2), m)
}

#[test]
fn calendar_units_round_as_opensearch_does() {
    let t = 1_700_000_000_000; // 2023-11-14T22:13:20Z, a Tuesday
    let r = |unit| DateRounding {
        kind: RoundingKind::Unit(unit),
        zone_ms: 0,
        region: None,
        offset: 0,
    };
    assert_eq!(r(DateUnit::Second).round(t), t);
    assert_eq!(r(DateUnit::Minute).round(t), 1_699_999_980_000);
    assert_eq!(r(DateUnit::Hour).round(t), 1_699_999_200_000);
    assert_eq!(r(DateUnit::Day).round(t), 1_699_920_000_000);
    assert_eq!(r(DateUnit::Week).round(t), 1_699_833_600_000, "Monday 13th");
    assert_eq!(r(DateUnit::Month).round(t), 1_698_796_800_000);
    assert_eq!(r(DateUnit::Quarter).round(t), 1_696_118_400_000);
    assert_eq!(r(DateUnit::Year).round(t), 1_672_531_200_000);
    // Before the epoch, and across a leap day.
    assert_eq!(r(DateUnit::Day).round(-1), -DAY);
    assert_eq!(r(DateUnit::Second).round(-1), -1_000);
    assert_eq!(r(DateUnit::Year).round(-1), -365 * DAY);
    assert_eq!(r(DateUnit::Month).round(-1), -31 * DAY);
    assert_eq!(
        r(DateUnit::Week).round(0),
        -3 * DAY,
        "1970-01-01 was a Thursday"
    );
    let leap = days_from_civil(2024, 2, 29) * DAY + 43_200_000;
    assert_eq!(
        r(DateUnit::Month).round(leap),
        days_from_civil(2024, 2, 1) * DAY
    );
    // Every unit against the oracle, over four centuries either side.
    let mut x: u64 = 0x9e37_79b9_7f4a_7c15;
    for _ in 0..20_000 {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        let t = (x % (800 * 366 * DAY as u64)) as i64 - 400 * 366 * DAY;
        let days = t.div_euclid(DAY);
        let (y, m) = civil_from_days(days);
        assert_eq!(r(DateUnit::Day).round(t), days * DAY, "{t}");
        assert_eq!(
            r(DateUnit::Month).round(t),
            days_from_civil(y, m, 1) * DAY,
            "{t}"
        );
        let q = (m - 1) / 3 * 3 + 1;
        assert_eq!(
            r(DateUnit::Quarter).round(t),
            days_from_civil(y, q, 1) * DAY,
            "{t}"
        );
        assert_eq!(
            r(DateUnit::Year).round(t),
            days_from_civil(y, 1, 1) * DAY,
            "{t}"
        );
        // Monday on or before: 1970-01-05 was a Monday.
        let monday = days - (days - 4).rem_euclid(7);
        assert_eq!(r(DateUnit::Week).round(t), monday * DAY, "{t}");
    }
}

#[test]
fn intervals_zones_and_offsets_round_as_opensearch_does() {
    let t = 1_700_000_000_000;
    // A fixed interval of 90 minutes, 17 minutes in.
    let fixed = DateRounding {
        kind: RoundingKind::Interval(90 * 60_000),
        zone_ms: 0,
        region: None,
        offset: 17 * 60_000,
    };
    let v = t - 17 * 60_000;
    assert_eq!(fixed.round(t), v - v.rem_euclid(90 * 60_000) + 17 * 60_000);
    assert_eq!(fixed.round(fixed.round(t)), fixed.round(t));
    // Negative values round down, not toward zero.
    let neg = DateRounding {
        kind: RoundingKind::Interval(1_000),
        zone_ms: 0,
        region: None,
        offset: 0,
    };
    assert_eq!(neg.round(-1), -1_000);
    assert_eq!(neg.round(-1_000), -1_000);
    // A day in +05:30 starts at 18:30 UTC the day before.
    let india = DateRounding {
        kind: RoundingKind::Unit(DateUnit::Day),
        zone_ms: 19_800_000,
        region: None,
        offset: 0,
    };
    assert_eq!(india.round(t), 1_699_986_600_000);
    let hour = DateRounding {
        kind: RoundingKind::Unit(DateUnit::Hour),
        zone_ms: 19_800_000,
        region: None,
        offset: 0,
    };
    assert_eq!(hour.round(t), 1_699_999_200_000 - 1_800_000);
}

#[test]
fn a_bucket_is_the_run_of_values_rounding_the_same() {
    let mut x: u64 = 0x2545_f491_4f6c_dd1d;
    let kinds = [
        RoundingKind::Unit(DateUnit::Week),
        RoundingKind::Unit(DateUnit::Year),
        RoundingKind::Unit(DateUnit::Quarter),
        RoundingKind::Unit(DateUnit::Month),
        RoundingKind::Unit(DateUnit::Day),
        RoundingKind::Unit(DateUnit::Hour),
        RoundingKind::Unit(DateUnit::Minute),
        RoundingKind::Unit(DateUnit::Second),
        RoundingKind::Interval(7_777),
    ];
    for kind in kinds {
        for (zone_ms, offset) in [(0, 0), (19_800_000, 0), (-3_600_000, 17 * 60_000)] {
            let r = DateRounding {
                kind,
                zone_ms,
                region: None,
                offset,
            };
            for _ in 0..500 {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                let t = (x % (400 * 366 * DAY as u64)) as i64 - 200 * 366 * DAY;
                let (from, to) = r.bucket(t).unwrap();
                assert_eq!(from, r.round(t), "{kind:?} {t}");
                assert!(from <= t && t < to, "{kind:?} {t}");
                assert_eq!(r.round(to - 1), from, "{kind:?} {t}: the last value in it");
                assert_eq!(
                    r.round(to),
                    to,
                    "{kind:?} {t}: the next bucket starts at to"
                );
            }
        }
    }
    // At the edges of the range there is no bucket to name.
    let day = DateRounding {
        kind: RoundingKind::Unit(DateUnit::Day),
        zone_ms: 0,
        region: None,
        offset: 0,
    };
    assert_eq!(day.bucket(i64::MAX - 5), None);
    let far = DateRounding {
        kind: RoundingKind::Unit(DateUnit::Day),
        zone_ms: 5,
        region: None,
        offset: i64::MIN,
    };
    assert_eq!(far.bucket(0), None);
}

#[test]
fn matched_ranges_are_the_candidates_and_nothing_else() {
    let ranges = [
        (f64::NEG_INFINITY, 0.0),
        (0.0, 100.0),
        (50.0, 250.0),
        (250.0, f64::INFINITY),
    ];
    let mt = max_to(&ranges);
    assert_eq!(mt, vec![0.0, 100.0, 250.0, f64::INFINITY]);
    let hits = |v: f64, low: usize| {
        let (s, e) = matched_range(&ranges, low, v, &mt);
        (s..e)
            .filter(|&i| v >= ranges[i].0 && v < ranges[i].1)
            .collect::<Vec<_>>()
    };
    assert_eq!(hits(-5.0, 0), vec![0]);
    assert_eq!(hits(0.0, 0), vec![1]);
    assert_eq!(hits(75.0, 0), vec![1, 2]);
    assert_eq!(hits(75.0, 2), vec![2], "searching on from a previous value");
    assert_eq!(hits(250.0, 0), vec![3]);
    assert_eq!(hits(f64::NAN, 0), Vec::<usize>::new());
    // Nothing can hold a value past every range.
    let bounded = [(0.0, 1.0), (2.0, 3.0)];
    let bt = max_to(&bounded);
    assert_eq!(matched_range(&bounded, 0, 5.0, &bt), (2, 2));
    assert_eq!(matched_range(&bounded, 0, -5.0, &bt), (0, 0));
    assert_eq!(matched_range(&bounded, 0, 1.5, &bt), (1, 1));
}

/// A state's parts as bits (`NaN` equal to itself).
fn bits(s: &MetricState) -> [u64; 7] {
    [
        s.count,
        s.sum.to_bits(),
        s.delta.to_bits(),
        s.min.to_bits(),
        s.max.to_bits(),
        s.min_of_mins.to_bits(),
        s.max_of_maxes.to_bits(),
    ]
}

fn fixture(name: &str) -> DirectoryReader {
    let dir = format!("{}/../../fixtures/data/{name}", env!("CARGO_MANIFEST_DIR"));
    DirectoryReader::open(&FsDirectory::open(dir)).expect("open fixture")
}

fn all() -> BooleanQuery {
    BooleanQuery {
        must: vec![Clause::MatchAllDocs(MatchAllDocsQuery::new(0))],
        ..Default::default()
    }
}

fn body(term: &str) -> BooleanQuery {
    BooleanQuery {
        must: vec![Clause::Term(TermQuery::new(
            "body",
            term.as_bytes().to_vec(),
        ))],
        ..Default::default()
    }
}

fn metric(field: &str, kind: ValueKind) -> AggNode {
    AggNode::Metric {
        script: None,
        missing: None,
        field: field.to_string(),
        kind,
        source: Source::DocValues,
        needs: NEED_ALL,
    }
}

/// Every live document matching `query` with its values of `field`,
/// segment by segment: the oracle the bucketed results are held to.
fn scan(
    reader: &DirectoryReader,
    query: &BooleanQuery,
    field: &str,
) -> Vec<(usize, i32, Vec<i64>)> {
    let mut opened = reader.open_segments().unwrap();
    opened.open_points().unwrap();
    let segments = opened.as_open_segments();
    let mut out = Vec::new();
    let mut buf = Vec::new();
    for (i, (seg, r)) in segments.iter().zip(reader.segment_readers()).enumerate() {
        let clause = segment_match::lone_clause(query);
        let docs: Vec<i32> = match segment_match::segment_matches(
            &segment_match::plain_context(seg),
            query,
            &clause,
            seg.live_docs,
            &mut buf,
        )
        .unwrap()
        {
            None => Vec::new(),
            Some(Some(d)) => d.to_vec(),
            Some(None) => (0..r.max_doc)
                .filter(|&d| seg.live_docs.is_none_or(|l| l.get_doc(d)))
                .collect(),
        };
        let mut values = metric_aggs::open_values(r, field).unwrap();
        for doc in docs {
            let mut v = Vec::new();
            match &mut values {
                Values::Absent => {}
                Values::Single(c) => v.extend(c.value(doc).unwrap()),
                Values::Multi(c) => c.values(doc, &mut v).unwrap(),
            }
            out.push((i, doc, v));
        }
    }
    out
}

fn run(
    reader: &DirectoryReader,
    query: &BooleanQuery,
    nodes: &[AggNode],
    slices: &[Vec<usize>],
) -> Result<Vec<Vec<AggResult>>> {
    run_limited(reader, query, nodes, slices, usize::MAX)
}

fn run_limited(
    reader: &DirectoryReader,
    query: &BooleanQuery,
    nodes: &[AggNode],
    slices: &[Vec<usize>],
    bucket_limit: usize,
) -> Result<Vec<Vec<AggResult>>> {
    let mut opened = reader.open_segments().unwrap();
    opened.open_points().unwrap();
    let segments = opened.as_open_segments();
    let mut fields = Vec::new();
    for n in nodes {
        n.keyword_fields(&mut fields);
    }
    let ords: HashMap<String, Arc<GlobalOrds>> = fields
        .into_iter()
        .map(|f| (f.to_string(), reader.global_ords(f).unwrap()))
        .collect();
    aggregate_tree(
        &segments,
        reader.segment_readers(),
        query,
        nodes,
        &Globals::new(&ords).with_bucket_limit(bucket_limit),
        slices,
        None,
    )
}

fn whole(reader: &DirectoryReader) -> Vec<Vec<usize>> {
    vec![(0..reader.segment_readers().len()).collect()]
}

/// Numeric `terms` keeps what `NumericTermsAggregator` keeps: each
/// document's distinct keys counted once, the `shard_size` best by count
/// then key, the rest as `otherDocCount`, listed by key -- and each kept
/// bucket's sub-aggregation sees exactly its documents.
#[test]
fn numeric_terms_keep_the_top_keys_of_a_scan() {
    let reader = fixture("metric_aggs_index");
    for (q, field, kind) in [
        (all(), "d", ValueKind::Double),
        (body("w1"), "md", ValueKind::Double),
        (all(), "ml", ValueKind::Long),
        (body("w4"), "f", ValueKind::Float),
        (all(), "i", ValueKind::Long),
    ] {
        let rows = scan(&reader, &q, field);
        let mut counts: HashMap<i64, u64> = HashMap::new();
        for (_, _, values) in &rows {
            let keys: HashSet<i64> = values.iter().map(|&v| numeric_terms_key(kind, v)).collect();
            for k in keys {
                *counts.entry(k).or_default() += 1;
            }
        }
        let orders = [
            TermsOrder::CountDesc,
            TermsOrder::CountAsc,
            TermsOrder::KeyAsc,
            TermsOrder::KeyDesc,
        ];
        for (shard_size, order) in [1, 3, 10_000]
            .into_iter()
            .flat_map(|s| orders.map(|o| (s, o)))
        {
            let mut ranked: Vec<(i64, u64)> = counts.iter().map(|(&k, &n)| (k, n)).collect();
            match order {
                TermsOrder::CountDesc => ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0))),
                TermsOrder::CountAsc => ranked.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0))),
                TermsOrder::KeyAsc => ranked.sort_by_key(|k| k.0),
                TermsOrder::KeyDesc => ranked.sort_by_key(|k| std::cmp::Reverse(k.0)),
            }
            let total: u64 = ranked.iter().map(|k| k.1).sum();
            ranked.truncate(shard_size);
            let other = total - ranked.iter().map(|k| k.1).sum::<u64>();
            // Listed by key, or in the order when it is the key's.
            if order == TermsOrder::KeyDesc {
                ranked.sort_by_key(|k| std::cmp::Reverse(k.0));
            } else {
                ranked.sort_by_key(|k| k.0);
            }
            let node = AggNode::NumericTerms {
                script: None,
                missing: None,
                field: field.to_string(),
                kind,
                shard_size,
                order,
                subs: vec![AggNode::Filters {
                    filters: vec![all()],
                    other: false,
                    subs: Vec::new(),
                }],
            };
            let got = run(&reader, &q, std::slice::from_ref(&node), &whole(&reader))
                .unwrap()
                .remove(0)
                .remove(0);
            let AggResult::NumericTerms { buckets, subs } = got else {
                panic!("{got:?}")
            };
            assert_eq!(
                buckets,
                vec![(other, ranked.clone())],
                "{field} shard_size {shard_size} {order:?}"
            );
            let AggResult::Fixed { docs, .. } = &subs[0] else {
                panic!()
            };
            let want: Vec<u64> = ranked.iter().map(|k| k.1).collect();
            assert_eq!(
                docs, &want,
                "{field}: each bucket's sub-aggregation sees its documents"
            );
            // Without sub-aggregations a multi-valued window counts
            // straight from the values read ahead: the same buckets.
            let bare = AggNode::NumericTerms {
                script: None,
                missing: None,
                field: field.to_string(),
                kind,
                shard_size,
                order,
                subs: Vec::new(),
            };
            let got = run(&reader, &q, std::slice::from_ref(&bare), &whole(&reader))
                .unwrap()
                .remove(0)
                .remove(0);
            assert!(
                matches!(&got, AggResult::NumericTerms { buckets: b, .. } if *b == vec![(other, ranked.clone())]),
                "{field} shard_size {shard_size} without subs: {got:?}"
            );
        }
    }
    // A double's key is its sortable long: -0.0 below 0.0, both below 1.0.
    let keys = [-0.0f64, 0.0, 1.0].map(|d| {
        numeric_terms_key(
            ValueKind::Double,
            lucene_util::numeric_utils::double_to_sortable_long(d),
        )
    });
    assert!(keys[0] < keys[1] && keys[1] < keys[2], "{keys:?}");
    assert_eq!(numeric_terms_key(ValueKind::Long, -7), -7);
}

/// The filter rewrite counts from points only what it can count
/// exactly: a top-level `date_histogram` or `range` without
/// sub-aggregations over a single-valued, one-dimensional points field
/// of the node's value type; anything else is left to the scan.
#[test]
fn counting_from_points_declines_what_it_cannot_count_exactly() {
    let reader = fixture("metric_aggs_index");
    let range = |field: &str, kind, subs: Vec<AggNode>| AggNode::Range {
        script: None,
        missing: None,
        field: field.to_string(),
        kind,
        ranges: vec![(f64::NEG_INFINITY, 0.0), (0.0, f64::INFINITY)],
        subs,
    };
    let mut plain = reader.open_segments().unwrap();
    {
        let segments = plain.as_open_segments();
        // No points opened for the segment at all.
        let r = range("d", ValueKind::Double, vec![]);
        assert!(count_from_points(&r, &segments[0], None).unwrap().is_none());
    }
    plain.open_points().unwrap();
    let segments = plain.as_open_segments();
    let seg = &segments[0];
    let terms = AggNode::Terms {
        accepted: None,
        missing: None,
        zero_buckets: false,
        field: "d".to_string(),
        shard_size: 1,
        order: TermsOrder::CountDesc,
        subs: vec![],
    };
    let declined = [
        terms,
        range(
            "d",
            ValueKind::Double,
            vec![AggNode::Global { subs: vec![] }],
        ),
        AggNode::Range {
            script: None,
            missing: None,
            field: "d".to_string(),
            kind: ValueKind::Double,
            ranges: vec![(f64::NAN, 1.0)],
            subs: vec![],
        },
        range("no-such-field", ValueKind::Double, vec![]),
        // Multi-valued: more points than documents.
        range("md", ValueKind::Double, vec![]),
        // A double field read as a float: the widths disagree.
        range("d", ValueKind::Float, vec![]),
    ];
    for node in &declined {
        assert!(
            count_from_points(node, seg, None).unwrap().is_none(),
            "{node:?}"
        );
    }
    assert!(
        count_from_points(&range("d", ValueKind::Double, vec![]), seg, None)
            .unwrap()
            .is_some()
    );
}

#[test]
fn a_match_all_filter_and_global_see_what_the_metrics_see() {
    let reader = fixture("metric_aggs_index");
    let mut opened = reader.open_segments().unwrap();
    opened.open_points().unwrap();
    let segments = opened.as_open_segments();
    for q in [all(), body("w1"), body("w4")] {
        let specs: Vec<MetricSpec> = [
            ("d", ValueKind::Double),
            ("md", ValueKind::Double),
            ("ml", ValueKind::Long),
            ("f", ValueKind::Float),
        ]
        .iter()
        .map(|&(f, kind)| MetricSpec {
            field: f.to_string(),
            kind,
            source: Source::DocValues,
            needs: NEED_ALL,
        })
        .collect();
        let want =
            crate::metric_aggs::metric_states(&segments, reader.segment_readers(), &q, &specs)
                .unwrap();
        let subs: Vec<AggNode> = specs.iter().map(|s| metric(&s.field, s.kind)).collect();
        let nodes = vec![
            AggNode::Filters {
                filters: vec![all()],
                other: true,
                subs: subs.clone(),
            },
            AggNode::Global { subs: subs.clone() },
        ];
        let got = run(&reader, &q, &nodes, &whole(&reader)).unwrap().remove(0);
        let AggResult::Fixed {
            width: 2,
            docs,
            subs: f,
        } = &got[0]
        else {
            panic!("{got:?}")
        };
        let matched = scan(&reader, &q, "d").len() as u64;
        assert_eq!(
            docs,
            &vec![matched, 0],
            "every match in the filter, none in other"
        );
        for (i, s) in f.iter().enumerate() {
            let AggResult::Metric(states) = s else {
                panic!()
            };
            assert_eq!(
                bits(&states[0]),
                bits(&want[i]),
                "{i}: bit for bit, the documents in the same order"
            );
            assert_eq!(states[1], MetricState::default(), "the empty other bucket");
        }
        let AggResult::Fixed {
            width: 1,
            docs,
            subs: g,
        } = &got[1]
        else {
            panic!()
        };
        assert_eq!(docs, &vec![matched]);
        for (i, s) in g.iter().enumerate() {
            let AggResult::Metric(g) = s else { panic!() };
            assert_eq!(bits(&g[0]), bits(&want[i]));
        }
    }
}

#[test]
fn a_pass_over_the_bucket_limit_stops_before_building_results() {
    let reader = fixture("metric_aggs_index");
    let histogram = |subs| AggNode::Histogram {
        script: None,
        missing: None,
        field: "i".to_string(),
        kind: ValueKind::Long,
        interval: 1.0,
        offset: 0.0,
        hard_bounds: (None, None),
        subs,
    };
    for node in [
        histogram(Vec::new()),
        histogram(vec![metric("ml", ValueKind::Long)]),
        AggNode::Range {
            script: None,
            missing: None,
            field: "i".to_string(),
            kind: ValueKind::Long,
            ranges: vec![(f64::NEG_INFINITY, 0.0), (0.0, f64::INFINITY)],
            subs: vec![histogram(Vec::new())],
        },
    ] {
        let nodes = std::slice::from_ref(&node);
        let all_buckets = |r: &AggResult| -> usize {
            fn count(r: &AggResult) -> usize {
                match r {
                    AggResult::Histogram { buckets, subs } => {
                        buckets.iter().map(Vec::len).sum::<usize>()
                            + subs.iter().map(count).sum::<usize>()
                    }
                    AggResult::Fixed { docs, subs, .. } => {
                        docs.len() + subs.iter().map(count).sum::<usize>()
                    }
                    _ => 0,
                }
            }
            count(r)
        };
        let full = run(&reader, &all(), nodes, &whole(&reader)).unwrap();
        let made = all_buckets(&full[0][0]);
        assert!(made > 2, "{made} buckets");
        let at = run_limited(&reader, &all(), nodes, &whole(&reader), made).unwrap();
        assert_eq!(at, full, "a limit the buckets reach is not passed");
        let over = run_limited(&reader, &all(), nodes, &whole(&reader), made - 1);
        assert!(
            matches!(over, Err(lucene_search::Error::TooManyBuckets(n, l)) if n > l && l == made - 1),
            "{over:?}"
        );
    }
}

// A nested `terms` holds a key for every (owning bucket, value) pair it
// meets, while its result keeps `shard_size` of them: the pairs, not the
// kept buckets, are bounded by the limit, at the pair that would pass it.
#[test]
fn a_nested_terms_stops_at_the_limit_on_its_pairs_not_its_kept_buckets() {
    let reader = fixture("metric_aggs_index");
    let node = AggNode::Histogram {
        script: None,
        missing: None,
        field: "i".to_string(),
        kind: ValueKind::Long,
        interval: 1.0,
        offset: 0.0,
        hard_bounds: (None, None),
        subs: vec![AggNode::NumericTerms {
            script: None,
            missing: None,
            field: "ml".to_string(),
            kind: ValueKind::Long,
            shard_size: 1,
            order: TermsOrder::CountDesc,
            subs: Vec::new(),
        }],
    };
    let nodes = std::slice::from_ref(&node);
    let full = run(&reader, &all(), nodes, &whole(&reader)).unwrap();
    let AggResult::Histogram { buckets, .. } = &full[0][0] else {
        panic!("{:?}", full[0][0]);
    };
    let histogram_buckets = buckets.iter().map(Vec::len).sum::<usize>();
    // What the pass counts from below: the histogram's buckets and one kept
    // terms bucket (shard_size 1).
    let limit = histogram_buckets + 1;
    let over = run_limited(&reader, &all(), nodes, &whole(&reader), limit);
    assert!(
        matches!(over, Err(lucene_search::Error::TooManyBuckets(n, l)) if n > l && l == limit),
        "{over:?}"
    );
    // A limit the pairs fit in passes.
    let fits = run_limited(&reader, &all(), nodes, &whole(&reader), usize::MAX - 1).unwrap();
    assert_eq!(fits, full);
}

#[test]
fn histograms_and_ranges_bucket_what_a_scan_buckets() {
    let reader = fixture("metric_aggs_index");
    for (q, field, kind) in [
        (all(), "d", ValueKind::Double),
        (body("w1"), "md", ValueKind::Double),
        (all(), "ml", ValueKind::Long),
        (body("w4"), "f", ValueKind::Float),
        (all(), "i", ValueKind::Long),
    ] {
        let rows = scan(&reader, &q, field);
        for (interval, offset, bounds) in [
            (10.0, 0.0, (None, None)),
            (3.5, 1.25, (Some(-20.0), Some(40.0))),
            (1e9, 0.0, (None, None)),
        ] {
            let node = AggNode::Histogram {
                script: None,
                missing: None,
                field: field.to_string(),
                kind,
                interval,
                offset,
                hard_bounds: bounds,
                subs: vec![
                    metric("ml", ValueKind::Long),
                    AggNode::Cardinality {
                        field: field.to_string(),
                        kind: CardinalityKind::Numeric(kind),
                        precision: None,
                    },
                ],
            };
            // The oracle: each distinct key of a document's values once.
            let mut want: std::collections::BTreeMap<i64, u64> = std::collections::BTreeMap::new();
            let mut distinct: HashMap<i64, HashSet<i64>> = HashMap::new();
            for (_, _, values) in &rows {
                // Java's loop: a key equal to the previous one -- which
                // starts at -Infinity, so a -Infinity value never counts,
                // and moves on past out-of-bounds keys too -- is skipped.
                let mut previous = f64::NEG_INFINITY;
                for &v in values {
                    let k = ((metric_aggs::to_double(kind, v) - offset) / interval).floor();
                    if k == previous {
                        continue;
                    }
                    previous = k;
                    let b = k * interval;
                    if bounds.0.is_some_and(|m| b < m) || bounds.1.is_some_and(|m| b > m) {
                        continue;
                    }
                    *want.entry(java_double_bits(k)).or_default() += 1;
                    let set = distinct.entry(java_double_bits(k)).or_default();
                    // A whole number field's longs as they are, not widened.
                    for &v in values {
                        set.insert(match kind {
                            ValueKind::Long => v,
                            _ => java_double_bits(metric_aggs::to_double(kind, v)),
                        });
                    }
                }
            }
            let got = run(&reader, &q, std::slice::from_ref(&node), &whole(&reader))
                .unwrap()
                .remove(0)
                .remove(0);
            let AggResult::Histogram { buckets, subs } = got else {
                panic!()
            };
            let mut got_counts: Vec<(i64, u64)> = buckets[0]
                .iter()
                .map(|&(k, n)| (java_double_bits(k), n))
                .collect();
            got_counts.sort_unstable();
            let mut want_counts: Vec<(i64, u64)> = want.into_iter().collect();
            want_counts.sort_unstable();
            assert_eq!(got_counts, want_counts, "{field} / {interval}");
            assert!(
                buckets[0]
                    .windows(2)
                    .all(|w| w[0].0.total_cmp(&w[1].0).is_lt()),
                "listed by key"
            );
            let AggResult::Cardinality(sets) = &subs[1] else {
                panic!()
            };
            for ((k, _), values) in buckets[0].iter().zip(sets) {
                assert_eq!(values.len(), distinct[&java_double_bits(*k)].len());
            }
        }
        // Ranges: overlapping, open-ended, one past every value; and
        // disjoint ones, an empty one among them. With a sub-aggregation
        // the documents are read (no count from the points).
        let sets = [
            vec![
                (f64::NEG_INFINITY, 0.0),
                (-5.0, 5.0),
                (0.0, 1e9),
                (1e9, f64::INFINITY),
            ],
            vec![
                (f64::NEG_INFINITY, -5.0),
                (-5.0, 2.5),
                (2.5, 2.5),
                (7.0, 40.0),
                (40.0, f64::INFINITY),
            ],
        ];
        for (ranges, subs) in sets
            .iter()
            .flat_map(|r| [(r, vec![]), (r, vec![metric(field, kind)])])
        {
            let node = AggNode::Range {
                script: None,
                missing: None,
                field: field.to_string(),
                kind,
                ranges: ranges.clone(),
                subs,
            };
            let mut want = vec![0u64; ranges.len()];
            for (_, _, values) in &rows {
                for (i, &(from, to)) in ranges.iter().enumerate() {
                    if values.iter().any(|&v| {
                        let d = metric_aggs::to_double(kind, v);
                        d >= from && d < to
                    }) {
                        want[i] += 1;
                    }
                }
            }
            let got = run(&reader, &q, &[node], &whole(&reader))
                .unwrap()
                .remove(0)
                .remove(0);
            let AggResult::Fixed { width, docs, .. } = got else {
                panic!("{got:?}")
            };
            assert_eq!((width, docs), (ranges.len(), want), "{field}");
        }
    }
}

/// One walk over a tree of many leaves counts every bucket exactly:
/// leaves inside one bucket by size, leaves crossing bucket boundaries
/// value by value (runs of equal values included), with and without hard
/// bounds -- the same counts as bucketing each value.
#[test]
fn counting_buckets_from_points_matches_bucketing_each_value() {
    use lucene_index::document::{Document, LongPoint};
    use lucene_index::index_writer::IndexWriter;
    use lucene_index::merge_policy::upgrade::LATEST;

    let dir = lucene_store::ByteBuffersDirectory::new();
    let mut values = Vec::new();
    {
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", LATEST).unwrap();
        w.set_ram_buffer_size_mb(1024.0).unwrap();
        for i in 0..20_000i64 {
            // Spread over ~40 buckets of 1000, with runs of one value.
            let v = if i % 7 == 0 {
                12_345
            } else {
                (i * 7919) % 40_000 - 300
            };
            values.push(v);
            let mut d = Document::new();
            d.add(LongPoint::new("ts", &[v]).unwrap());
            w.add_fields_document(&d).unwrap();
        }
        w.commit().unwrap();
    }
    let reader = DirectoryReader::open(&dir).unwrap();
    let mut opened = reader.open_segments().unwrap();
    opened.open_points().unwrap();
    let segments = opened.as_open_segments();
    assert_eq!(segments.len(), 1);
    for interval in [1000i64, 397, 50_000] {
        let rounding = DateRounding {
            kind: RoundingKind::Interval(interval),
            zone_ms: 0,
            region: None,
            offset: 3,
        };
        for bounds in [(None, None), (Some(0), Some(20_000))] {
            let node = AggNode::DateHistogram {
                script: None,
                missing: None,
                field: "ts".to_string(),
                rounding,
                hard_bounds: bounds,
                subs: vec![],
            };
            let mut want: std::collections::BTreeMap<i64, u64> = std::collections::BTreeMap::new();
            for &v in &values {
                let k = rounding.round(v);
                if bounds.0.is_some_and(|m| k < m) || bounds.1.is_some_and(|m| k >= m) {
                    continue;
                }
                *want.entry(k).or_default() += 1;
            }
            let Some(PointCounts::Keyed(got)) =
                count_from_points(&node, &segments[0], None).unwrap()
            else {
                panic!("interval {interval}: not counted from the points")
            };
            assert_eq!(
                got,
                want.into_iter().collect::<Vec<_>>(),
                "interval {interval} bounds {bounds:?}"
            );
        }
        // Under a points range on the same field: only the values it holds count.
        let node = AggNode::DateHistogram {
            script: None,
            missing: None,
            field: "ts".to_string(),
            rounding,
            hard_bounds: (None, None),
            subs: vec![],
        };
        for clip in [
            (0i64, 5_000i64),
            (1234, 1234),
            (-50, 30_000),
            (7_001, 7_999),
            (i64::MIN, 4_321),
            (9_000, i64::MAX),
            (60_000, 70_000),
        ] {
            let mut want: std::collections::BTreeMap<i64, u64> = std::collections::BTreeMap::new();
            for &v in values.iter().filter(|&&v| v >= clip.0 && v <= clip.1) {
                *want.entry(rounding.round(v)).or_default() += 1;
            }
            let Some(PointCounts::Keyed(got)) =
                count_from_points(&node, &segments[0], Some(clip)).unwrap()
            else {
                panic!("interval {interval} clip {clip:?}: not counted from the points")
            };
            assert_eq!(
                got,
                want.into_iter().collect::<Vec<_>>(),
                "interval {interval} clip {clip:?}"
            );
        }
    }
}

/// Months differ in length and a zone shifts their edges: the one walk
/// still counts what bucketing each value counts, and hard bounds that
/// keep a few months out of years decode nothing outside them.
#[test]
fn counting_month_buckets_in_a_zone_from_points_matches_bucketing_each_value() {
    use lucene_index::document::{Document, LongPoint};
    use lucene_index::index_writer::IndexWriter;
    use lucene_index::merge_policy::upgrade::LATEST;

    const DAY: i64 = 86_400_000;
    let dir = lucene_store::ByteBuffersDirectory::new();
    let mut values = Vec::new();
    {
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", LATEST).unwrap();
        w.set_ram_buffer_size_mb(1024.0).unwrap();
        for i in 0..20_000i64 {
            // About three years from 2020, some on month edges.
            let v = 1_577_836_800_000 + (i * 7919) % (1100 * DAY) + if i % 11 == 0 { 0 } else { 1 };
            values.push(v);
            let mut d = Document::new();
            d.add(LongPoint::new("ts", &[v]).unwrap());
            w.add_fields_document(&d).unwrap();
        }
        w.commit().unwrap();
    }
    let reader = DirectoryReader::open(&dir).unwrap();
    let mut opened = reader.open_segments().unwrap();
    opened.open_points().unwrap();
    let segments = opened.as_open_segments();
    for zone_ms in [0i64, 19_800_000, -18_000_000] {
        let rounding = DateRounding {
            kind: RoundingKind::Unit(DateUnit::Month),
            zone_ms,
            region: None,
            offset: 0,
        };
        let mid = rounding.round(values[5000]);
        for bounds in [(None, None), (Some(mid), Some(mid + 90 * DAY))] {
            let node = AggNode::DateHistogram {
                script: None,
                missing: None,
                field: "ts".to_string(),
                rounding,
                hard_bounds: bounds,
                subs: vec![],
            };
            let mut want: std::collections::BTreeMap<i64, u64> = std::collections::BTreeMap::new();
            for &v in &values {
                let k = rounding.round(v);
                if bounds.0.is_some_and(|m| k < m) || bounds.1.is_some_and(|m| k >= m) {
                    continue;
                }
                *want.entry(k).or_default() += 1;
            }
            let Some(PointCounts::Keyed(got)) =
                count_from_points(&node, &segments[0], None).unwrap()
            else {
                panic!("zone {zone_ms}: not counted from the points")
            };
            assert_eq!(
                got,
                want.into_iter().collect::<Vec<_>>(),
                "zone {zone_ms} bounds {bounds:?}"
            );
        }
    }
}

#[test]
fn date_histograms_bucket_each_rounded_value_once() {
    let reader = fixture("metric_aggs_index");
    // `l` has one point per document: its segment without deletions is
    // counted from the points; `ml`, multi-valued, never is.
    for field in ["ml", "l"] {
        let rows = scan(&reader, &all(), field);
        for kind in [
            RoundingKind::Interval(7),
            RoundingKind::Unit(DateUnit::Second),
            RoundingKind::Interval(1),
            // Few enough buckets to count from the points.
            RoundingKind::Interval(1 << 56),
            RoundingKind::Unit(DateUnit::Year),
        ] {
            let rounding = DateRounding {
                kind,
                zone_ms: 0,
                region: None,
                offset: 3,
            };
            for bounds in [(None, None), (Some(0), Some(500))] {
                let subs = if field == "l" {
                    vec![]
                } else {
                    vec![metric("d", ValueKind::Double)]
                };
                let node = AggNode::DateHistogram {
                    script: None,
                    missing: None,
                    field: field.to_string(),
                    rounding,
                    hard_bounds: bounds,
                    subs,
                };
                let mut want: std::collections::BTreeMap<i64, u64> =
                    std::collections::BTreeMap::new();
                for (_, _, values) in &rows {
                    let mut keys: Vec<i64> = values.iter().map(|&v| rounding.round(v)).collect();
                    keys.dedup();
                    for k in keys {
                        if bounds.0.is_some_and(|m| k < m) || bounds.1.is_some_and(|m| k >= m) {
                            continue;
                        }
                        *want.entry(k).or_default() += 1;
                    }
                }
                let got = run(&reader, &all(), &[node], &whole(&reader))
                    .unwrap()
                    .remove(0)
                    .remove(0);
                let AggResult::DateHistogram { buckets, subs } = got else {
                    panic!()
                };
                assert_eq!(buckets[0], want.into_iter().collect::<Vec<_>>());
                if let Some(AggResult::Metric(states)) = subs.first() {
                    assert_eq!(states.len(), buckets[0].len());
                }
            }
        }
    }
}

/// Reading columns a window at a time and collecting a window per node
/// (`collect_window`) answers what document-at-a-time collection does:
/// nested buckets with metrics, a cardinality and multi-valued columns,
/// over every document (windows) and a sparse query (documents one by
/// one), with sparse single-valued columns and `-Infinity` values.
#[test]
fn windows_answer_what_documents_one_at_a_time_answer() {
    let reader = fixture("metric_aggs_index");
    let rounding = DateRounding {
        kind: RoundingKind::Interval(7),
        zone_ms: 0,
        region: None,
        offset: 3,
    };
    let date = |field: &str, subs| AggNode::DateHistogram {
        script: None,
        missing: None,
        field: field.to_string(),
        rounding,
        hard_bounds: (None, None),
        subs,
    };
    let histogram = |field: &str, kind, subs| AggNode::Histogram {
        script: None,
        missing: None,
        field: field.to_string(),
        kind,
        interval: 3.5,
        offset: 1.25,
        hard_bounds: (Some(-20.0), Some(40.0)),
        subs,
    };
    let trees = vec![
        date(
            "l",
            vec![
                metric("d", ValueKind::Double),
                metric("ml", ValueKind::Long),
            ],
        ),
        histogram(
            "d",
            ValueKind::Double,
            vec![
                date("i", vec![metric("f", ValueKind::Float)]),
                AggNode::Cardinality {
                    field: "e".to_string(),
                    kind: CardinalityKind::Numeric(ValueKind::Double),
                    precision: None,
                },
            ],
        ),
        histogram(
            "e",
            ValueKind::Double,
            vec![
                metric("l", ValueKind::Long),
                AggNode::Cardinality {
                    field: "d".to_string(),
                    kind: CardinalityKind::Numeric(ValueKind::Double),
                    precision: Some(5),
                },
                AggNode::Cardinality {
                    field: "ml".to_string(),
                    kind: CardinalityKind::Numeric(ValueKind::Long),
                    precision: Some(14),
                },
            ],
        ),
        histogram("md", ValueKind::Double, vec![metric("i", ValueKind::Long)]),
        AggNode::Range {
            script: None,
            missing: None,
            field: "d".to_string(),
            kind: ValueKind::Double,
            ranges: vec![(f64::NEG_INFINITY, 0.0), (0.0, 10.0), (5.0, f64::INFINITY)],
            subs: vec![date("l", vec![metric("e", ValueKind::Double)])],
        },
        metric("f", ValueKind::Float),
        // A lone `min` or `max` of longs: folded as integers when the
        // window has one owner (top level), per document under buckets.
        only("l", metric_aggs::NEED_MIN),
        only("i", metric_aggs::NEED_MAX),
        date(
            "l",
            vec![
                only("i", metric_aggs::NEED_MIN),
                only("l", metric_aggs::NEED_MAX),
            ],
        ),
        AggNode::Global {
            subs: vec![
                metric("l", ValueKind::Long),
                date("l", vec![metric("d", ValueKind::Double)]),
            ],
        },
    ];
    fn only(field: &str, needs: u8) -> AggNode {
        AggNode::Metric {
            script: None,
            missing: None,
            field: field.to_string(),
            kind: ValueKind::Long,
            source: Source::DocValues,
            needs,
        }
    }
    let both = |reader: &DirectoryReader, q: &BooleanQuery, tree: &AggNode| {
        let nodes = std::slice::from_ref(tree);
        NO_WINDOWS.with(|w| w.set(true));
        let one = run(reader, q, nodes, &whole(reader)).unwrap();
        NO_WINDOWS.with(|w| w.set(false));
        let windowed = run(reader, q, nodes, &whole(reader)).unwrap();
        assert_eq!(format!("{windowed:?}"), format!("{one:?}"), "{tree:?}");
    };
    for q in [all(), body("w1"), body("w4")] {
        for tree in &trees {
            both(&reader, &q, tree);
        }
    }
    // Keyword columns: single- and multi-valued ordinals, as terms and as
    // cardinalities, with and without a precision.
    let reader = fixture("terms_aggs_index");
    for (field, other) in [("kw", "mkw"), ("mkw", "kw")] {
        let tree = AggNode::Terms {
            accepted: None,
            missing: None,
            zero_buckets: false,
            field: field.to_string(),
            shard_size: 5,
            order: TermsOrder::CountDesc,
            subs: vec![
                AggNode::Cardinality {
                    field: other.to_string(),
                    kind: CardinalityKind::Keyword,
                    precision: None,
                },
                AggNode::Cardinality {
                    field: other.to_string(),
                    kind: CardinalityKind::Keyword,
                    precision: Some(12),
                },
            ],
        };
        for q in [all(), body("w1")] {
            both(&reader, &q, &tree);
        }
    }
    // Single-valued terms under single-valued terms: each window's owner
    // changes from document to document.
    let keyword = |field: &str, subs| AggNode::Terms {
        accepted: None,
        missing: None,
        zero_buckets: false,
        field: field.to_string(),
        shard_size: 5,
        order: TermsOrder::CountDesc,
        subs,
    };
    let nested = keyword("sk", vec![keyword("kw", vec![]), keyword("hk", vec![])]);
    for q in [all(), body("w1")] {
        both(&reader, &q, &nested);
    }
}

/// `SortedNumericReader::fill_window` is `values` document by document,
/// on the fixtures' multi-valued columns (numbers and keyword ordinals,
/// dense and sparse): aligned windows across each segment, windows that
/// are not aligned, one running past the documents, and a rewind.
#[test]
fn a_multi_valued_window_is_each_documents_values() {
    let (mut checked, mut declined) = (0, 0);
    for (index, field, keyword) in [
        ("metric_aggs_index", "ml", false),
        ("metric_aggs_index", "md", false),
        ("terms_aggs_index", "mkw", true),
    ] {
        let reader = fixture(index);
        for r in reader.segment_readers() {
            let open = || -> Option<Box<lucene_codecs::doc_values::SortedNumericReader<'_>>> {
                if keyword {
                    match lucene_search::global_ords::open_ords(r, field).unwrap().0 {
                        Ords::Multi(m) => Some(m),
                        _ => None,
                    }
                } else {
                    match metric_aggs::open_values(r, field).unwrap() {
                        Values::Multi(m) => Some(m),
                        _ => None,
                    }
                }
            };
            let Some(mut windowed) = open() else {
                continue;
            };
            let max = r.max_doc;
            let mut windows: Vec<(i32, usize)> = (0..max)
                .step_by(WINDOW as usize)
                .map(|b| (b, WINDOW as usize))
                .collect();
            windows.extend([(3, 100), (max - 5, 64), (0, 64), (max, 0)]);
            let (mut offsets, mut values) = (Vec::new(), Vec::new());
            for (base, len) in windows {
                if !windowed
                    .fill_window(base, len, &mut offsets, &mut values)
                    .unwrap()
                {
                    declined += 1;
                    continue;
                }
                assert_eq!(offsets.len(), len + 1);
                let mut by_doc = open().unwrap();
                let mut want = Vec::new();
                for i in 0..len {
                    let doc = base + i as i32;
                    if doc < max {
                        by_doc.values(doc, &mut want).unwrap();
                    } else {
                        want.clear();
                    }
                    let got = &values[offsets[i] as usize..offsets[i + 1] as usize];
                    assert_eq!(got, &want[..], "{field} doc {doc} window {base}+{len}");
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 1000, "{checked}");
    // Unaligned windows over a sparse column are left to the caller.
    assert!(declined > 0);
}

/// With a precision, a `cardinality` answers each bucket's sketch: the
/// one its distinct values build when hashed as `CardinalityAggregator`
/// hashes them (`cardinality_sketch`'s tests hold the sketch to Java's).
#[test]
fn a_cardinality_with_a_precision_is_its_values_sketch() {
    use crate::cardinality_sketch::{mix64, murmur3_h1, Sketch};
    let sketch = |p: u32, values: &[CardinalityValue]| -> Vec<u8> {
        if values.is_empty() {
            return Vec::new();
        }
        let mut s = Sketch::new(p).unwrap();
        for v in values {
            s.collect(match v {
                CardinalityValue::Term(t) => murmur3_h1(t, 0),
                CardinalityValue::Long(l) => mix64(*l),
            });
        }
        let mut out = Vec::new();
        s.write_to(&mut out);
        out
    };
    for (index, field, kind) in [
        ("terms_aggs_index", "kw", CardinalityKind::Keyword),
        ("terms_aggs_index", "mkw", CardinalityKind::Keyword),
        (
            "metric_aggs_index",
            "ml",
            CardinalityKind::Numeric(ValueKind::Long),
        ),
        (
            "metric_aggs_index",
            "d",
            CardinalityKind::Numeric(ValueKind::Double),
        ),
    ] {
        let reader = fixture(index);
        // A numeric one is fed every value in document order, as
        // `DirectCollector` feeds it; a keyword one its distinct terms
        // in term order, as `OrdinalsCollector` does.
        let rows = match kind {
            CardinalityKind::Numeric(_) => scan(&reader, &all(), field),
            CardinalityKind::Keyword => Vec::new(),
        };
        let in_doc_order = |p: u32| -> Vec<u8> {
            let CardinalityKind::Numeric(k) = kind else {
                unreachable!()
            };
            let mut s = Sketch::new(p).unwrap();
            for (_, _, values) in &rows {
                for &v in values {
                    s.collect(numeric_hash(k, v));
                }
            }
            let mut out = Vec::new();
            s.write_to(&mut out);
            out
        };
        let want = |p: u32, values: &[CardinalityValue]| match kind {
            CardinalityKind::Numeric(_) if !values.is_empty() => in_doc_order(p),
            _ => sketch(p, values),
        };
        for p in [4, 10, 14] {
            let node = |precision| AggNode::Cardinality {
                field: field.to_string(),
                kind,
                precision,
            };
            // Under a filter bucket that matches nothing too: an empty
            // bucket answers no sketch.
            let nodes = vec![
                node(None),
                node(Some(p)),
                AggNode::Filters {
                    filters: vec![all(), body("no-such-term")],
                    other: false,
                    subs: vec![node(None), node(Some(p))],
                },
            ];
            let got = run(&reader, &all(), &nodes, &whole(&reader))
                .unwrap()
                .remove(0);
            let (AggResult::Cardinality(values), AggResult::CardinalitySketch(sketches)) =
                (&got[0], &got[1])
            else {
                panic!("{got:?}")
            };
            assert_eq!(sketches[0], want(p, &values[0]), "{field} p {p}");
            assert!(!sketches[0].is_empty());
            let AggResult::Fixed { subs, .. } = &got[2] else {
                panic!()
            };
            let (AggResult::Cardinality(values), AggResult::CardinalitySketch(sketches)) =
                (&subs[0], &subs[1])
            else {
                panic!()
            };
            assert_eq!(sketches.len(), 2);
            for (v, s) in values.iter().zip(sketches) {
                assert_eq!(s, &want(p, v), "{field} p {p} under filters");
            }
            assert!(sketches[1].is_empty(), "the empty filter bucket");
        }
    }
    // A precision OpenSearch would refuse is an error, not a panic.
    let reader = fixture("terms_aggs_index");
    let bad = AggNode::Cardinality {
        field: "kw".to_string(),
        kind: CardinalityKind::Keyword,
        precision: Some(40),
    };
    assert!(run(&reader, &all(), &[bad], &whole(&reader)).is_err());
}

/// A `missing` value is what every matching document without a value
/// reads (`MissingValues.replaceMissing`), on the window read ahead
/// (single-valued columns), document by document (multi-valued ones),
/// and under a parent bucket alike.
#[test]
fn missing_values_stand_in_for_documents_without_one() {
    use lucene_util::numeric_utils::{double_to_sortable_long, float_to_sortable_int};
    let reader = fixture("metric_aggs_index");
    let mut absent = 0;
    for (q, field, kind, missing) in [
        (all(), "i", ValueKind::Long, 7),
        (all(), "ml", ValueKind::Long, -3),
        (
            body("w1"),
            "d",
            ValueKind::Double,
            double_to_sortable_long(2.5),
        ),
        (
            all(),
            "md",
            ValueKind::Double,
            double_to_sortable_long(-0.0),
        ),
        (
            body("w4"),
            "f",
            ValueKind::Float,
            i64::from(float_to_sortable_int(1.5)),
        ),
        (all(), "no_such_field", ValueKind::Long, 11),
    ] {
        let mut counts: HashMap<i64, u64> = HashMap::new();
        for (_, _, values) in scan(&reader, &q, field) {
            let values = if values.is_empty() {
                absent += 1;
                vec![missing]
            } else {
                values
            };
            let keys: HashSet<i64> = values.iter().map(|&v| numeric_terms_key(kind, v)).collect();
            for k in keys {
                *counts.entry(k).or_default() += 1;
            }
        }
        let mut want: Vec<(i64, u64)> = counts.into_iter().collect();
        want.sort_unstable();
        for subs in [Vec::new(), vec![all()]] {
            let node = AggNode::NumericTerms {
                script: None,
                field: field.to_string(),
                missing: Some(missing),
                kind,
                shard_size: 1 << 20,
                order: TermsOrder::KeyAsc,
                subs: subs
                    .iter()
                    .map(|f: &BooleanQuery| AggNode::Filters {
                        filters: vec![f.clone()],
                        other: false,
                        subs: Vec::new(),
                    })
                    .collect(),
            };
            // Under a filter bucket too: a sub-aggregation's leaf.
            let parent = AggNode::Filters {
                filters: vec![all()],
                other: false,
                subs: vec![node.clone()],
            };
            for (n, nested) in [(node, false), (parent, true)] {
                let mut got = run(&reader, &q, &[n], &whole(&reader))
                    .unwrap()
                    .remove(0)
                    .remove(0);
                if nested {
                    let AggResult::Fixed { subs, .. } = got else {
                        panic!("{got:?}")
                    };
                    got = subs.into_iter().next().unwrap();
                }
                let AggResult::NumericTerms { buckets, .. } = got else {
                    panic!("{got:?}")
                };
                assert_eq!(buckets, vec![(0, want.clone())], "{field} nested {nested}");
            }
        }
    }
    assert!(absent > 0, "some documents lack a value");
}

/// `min_doc_count: 0` makes every term of the shard's global ordinals a
/// candidate: the counted ones with their counts, every other one empty.
#[test]
fn keyword_terms_with_min_doc_count_zero_hold_every_term() {
    let reader = fixture("terms_aggs_index");
    let mut opened = reader.open_segments().unwrap();
    opened.open_points().unwrap();
    let segments = opened.as_open_segments();
    for field in ["kw", "mkw"] {
        let terms_count =
            lucene_search::global_ords::GlobalOrds::build(reader.segment_readers(), field)
                .unwrap()
                .value_count();
        let q = BooleanQuery {
            must: vec![Clause::PointsRange(PointsRangeQuery::new("r", 0, 2))],
            ..Default::default()
        };
        let counted =
            crate::terms_agg::terms(&segments, reader.segment_readers(), &q, field, 1 << 20)
                .unwrap()
                .buckets;
        assert!(counted.len() < terms_count, "{field}: some terms uncounted");
        let node = |order, shard_size| AggNode::Terms {
            accepted: None,
            missing: None,
            field: field.to_string(),
            shard_size,
            order,
            zero_buckets: true,
            subs: Vec::new(),
        };
        let got = run(
            &reader,
            &q,
            &[node(TermsOrder::KeyAsc, 1 << 20)],
            &whole(&reader),
        )
        .unwrap()
        .remove(0)
        .remove(0);
        let AggResult::Terms { buckets, .. } = got else {
            panic!("{got:?}")
        };
        let (other, list) = &buckets[0];
        assert_eq!(*other, 0);
        assert_eq!(list.len(), terms_count, "{field}");
        assert!(list.windows(2).all(|w| w[0].0 < w[1].0), "{field}: by term");
        let nonzero: Vec<_> = list.iter().filter(|b| b.1 > 0).cloned().collect();
        assert_eq!(nonzero, counted, "{field}");
        // `_count` ascending keeps empty buckets first, ties by term.
        let got = run(
            &reader,
            &q,
            &[node(TermsOrder::CountAsc, 2)],
            &whole(&reader),
        )
        .unwrap()
        .remove(0)
        .remove(0);
        let AggResult::Terms { buckets, .. } = got else {
            panic!("{got:?}")
        };
        let zero: Vec<_> = list.iter().filter(|b| b.1 == 0).take(2).cloned().collect();
        let total: u64 = counted.iter().map(|b| b.1).sum();
        assert_eq!(buckets, vec![(total, zero)], "{field}");
    }
}

/// A keyword `terms` with a `missing` term counts the documents without
/// a value under it: into its bucket when a segment holds it, otherwise
/// a bucket of its own in term order (`replaceMissingOrdinals`).
#[test]
fn keyword_missing_term_stands_in_for_documents_without_one() {
    let reader = fixture("terms_aggs_index");
    let q = BooleanQuery {
        must: vec![Clause::PointsRange(PointsRangeQuery::new("r", 0, 2))],
        ..Default::default()
    };
    let node = |missing: Option<&[u8]>, order, zero_buckets| AggNode::Terms {
        accepted: None,
        field: "mkw".to_string(),
        missing: missing.map(<[u8]>::to_vec),
        shard_size: 1 << 20,
        order,
        zero_buckets,
        subs: Vec::new(),
    };
    let list = |n: AggNode| {
        let got = run(&reader, &q, &[n], &whole(&reader))
            .unwrap()
            .remove(0)
            .remove(0);
        let AggResult::Terms { mut buckets, .. } = got else {
            panic!("{got:?}")
        };
        buckets.remove(0).1
    };
    let base = list(node(None, TermsOrder::KeyAsc, false));
    let (first, first_count) = base[0].clone();
    let merged = list(node(Some(&first), TermsOrder::KeyAsc, false));
    assert_eq!(merged.len(), base.len());
    assert_eq!(merged[1..], base[1..]);
    let absent = merged[0].1 - first_count;
    assert!(absent > 0, "some matched documents lack mkw");
    // Just after the first term: its own bucket, second.
    let mut after = first.clone();
    after.push(0);
    let inserted = list(node(Some(&after), TermsOrder::KeyAsc, false));
    let mut want = base.clone();
    want.insert(1, (after.clone(), absent));
    assert_eq!(inserted, want);
    // Before every term, and past every term.
    let got = list(node(Some(b""), TermsOrder::KeyAsc, false));
    assert_eq!(got[0], (Vec::new(), absent));
    assert_eq!(got[1..], base[..]);
    let got = list(node(Some(b"\xff"), TermsOrder::KeyDesc, false));
    assert_eq!(got[0], (b"\xff".to_vec(), absent));
    // Every bucket kept by count, listed by term.
    let got = list(node(Some(&after), TermsOrder::CountDesc, false));
    assert_eq!(got, inserted);
    // `min_doc_count: 0`: the inserted term is one of the empty buckets'.
    let every = list(node(Some(&after), TermsOrder::KeyAsc, true));
    let terms_count =
        lucene_search::global_ords::GlobalOrds::build(reader.segment_readers(), "mkw")
            .unwrap()
            .value_count();
    assert_eq!(every.len(), terms_count + 1);
    assert!(every.contains(&(after, absent)));
    assert!(every.windows(2).all(|w| w[0].0 < w[1].0));
}

/// A `composite` whose first source is read sorted (its terms in order,
/// stopping after a term that adds no competitive key) keeps what the
/// document pass keeps: buckets, counts and sub-aggregations, on
/// segments with deletions, paged with `after`.
#[test]
fn composite_sorted_lead_keeps_the_document_pass_buckets() {
    let reader = fixture("terms_aggs_index");
    {
        let opened = reader.open_segments().unwrap();
        let segments = opened.as_open_segments();
        assert!(segments.iter().all(|s| s.fields.field("pk").is_some()));
        assert!(segments.iter().any(|s| s.live_docs.is_some()));
    }
    let source = |field: &str| CompositeSource {
        field: field.to_string(),
        values: CompositeValues::Keyword,
        reverse: false,
        missing_bucket: false,
        missing_order: MissingOrder::Default,
    };
    let segments = reader.segment_readers().len();
    let sub = || AggNode::Terms {
        field: "mkw".to_string(),
        missing: None,
        accepted: None,
        shard_size: 3,
        order: TermsOrder::CountDesc,
        zero_buckets: false,
        subs: Vec::new(),
    };
    for second in ["kw", "mkw"] {
        let node =
            |size, after: Option<Vec<CompositeAfter>>, sorted: bool, subs| AggNode::Composite {
                sources: vec![source("pk"), source(second)],
                size,
                after,
                sorted_lead: if sorted {
                    vec![true; segments]
                } else {
                    Vec::new()
                },
                subs,
            };
        for size in [1, 5, 40, 1 << 20] {
            let mut after = None;
            for _ in 0..4 {
                let got = |sorted| {
                    run(
                        &reader,
                        &all(),
                        &[node(size, after.clone(), sorted, vec![sub()])],
                        &whole(&reader),
                    )
                    .unwrap()
                    .remove(0)
                    .remove(0)
                };
                let (want, sorted) = (got(false), got(true));
                assert_eq!(
                    format!("{sorted:?}"),
                    format!("{want:?}"),
                    "{second} size {size}"
                );
                let AggResult::Composite { buckets, .. } = want else {
                    panic!("{want:?}")
                };
                let Some(last) = buckets.last() else {
                    break;
                };
                after = Some(
                    last.0
                        .iter()
                        .map(|v| match v {
                            CompositeKeyValue::Term(t) => CompositeAfter::Term(t.clone()),
                            v => panic!("{v:?}"),
                        })
                        .collect(),
                );
            }
        }
    }
}

/// The sorted walk stops on a term only when the queue is full and the
/// term offered keys none of which was competitive: a term whose
/// documents lack the second source offers none, and the walk goes on.
#[test]
fn composite_sorted_lead_goes_past_a_term_without_keys() {
    use lucene_index::document::{Document, KeywordField, Store};
    use lucene_index::index_writer::IndexWriter;
    use lucene_index::merge_policy::upgrade::LATEST;

    let dir = lucene_store::ByteBuffersDirectory::new();
    {
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", LATEST).unwrap();
        for (lead, second) in [
            ("a", Some("x")),
            ("b", None),
            ("c", Some("y")),
            ("d", Some("z")),
        ] {
            let mut d = Document::new();
            d.add(KeywordField::new("lead", lead, Store::No));
            if let Some(v) = second {
                d.add(KeywordField::new("second", v, Store::No));
            }
            w.add_fields_document(&d).unwrap();
        }
        w.commit().unwrap();
    }
    let reader = DirectoryReader::open(&dir).unwrap();
    let source = |field: &str| CompositeSource {
        field: field.to_string(),
        values: CompositeValues::Keyword,
        reverse: false,
        missing_bucket: false,
        missing_order: MissingOrder::Default,
    };
    for size in [1, 2, 10] {
        let got = |sorted: bool| {
            let node = AggNode::Composite {
                sources: vec![source("lead"), source("second")],
                size,
                after: None,
                sorted_lead: vec![sorted],
                subs: Vec::new(),
            };
            let r = run(&reader, &all(), &[node], &whole(&reader)).unwrap();
            format!("{:?}", r[0][0])
        };
        assert_eq!(got(true), got(false), "size {size}");
    }
    let node = AggNode::Composite {
        sources: vec![source("lead"), source("second")],
        size: 10,
        after: None,
        sorted_lead: vec![true],
        subs: Vec::new(),
    };
    let AggResult::Composite { buckets, .. } = run(&reader, &all(), &[node], &whole(&reader))
        .unwrap()
        .remove(0)
        .remove(0)
    else {
        panic!()
    };
    assert_eq!(buckets.len(), 3);
}

/// A numeric first source read sorted from its points (stopping once the
/// queue is full and a bucket places nothing) lists what the document
/// pass lists, paged with `after`, with a keyword second source, a
/// sub-aggregation, a multi-valued first field and deleted documents.
#[test]
fn composite_sorted_points_lead_keeps_the_document_pass_buckets() {
    use lucene_index::document::{
        Document, KeywordField, LongPoint, SortedNumericDocValuesField, Store,
    };
    use lucene_index::index_writer::IndexWriter;
    use lucene_index::merge_policy::upgrade::LATEST;

    let dir = lucene_store::ByteBuffersDirectory::new();
    {
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", LATEST).unwrap();
        for i in 0..600i64 {
            let mut d = Document::new();
            let lead = (i * 7919) % 53 - 20;
            for v in if i % 11 == 0 {
                vec![lead, lead + 3]
            } else {
                vec![lead]
            } {
                d.add(LongPoint::new("lead", &[v]).unwrap());
                d.add(SortedNumericDocValuesField::new("lead", v));
            }
            if i % 13 != 0 {
                d.add(KeywordField::new(
                    "second",
                    format!("k{}", i % 5),
                    Store::No,
                ));
            }
            d.add(SortedNumericDocValuesField::new("m", i));
            w.add_fields_document(&d).unwrap();
        }
        w.commit().unwrap();
    }
    let reader = DirectoryReader::open(&dir).unwrap();
    let sources = |second: bool| {
        let mut v = vec![CompositeSource {
            field: "lead".to_string(),
            values: CompositeValues::Long,
            reverse: false,
            missing_bucket: false,
            missing_order: MissingOrder::Default,
        }];
        if second {
            v.push(CompositeSource {
                field: "second".to_string(),
                values: CompositeValues::Keyword,
                reverse: false,
                missing_bucket: false,
                missing_order: MissingOrder::Default,
            });
        }
        v
    };
    let sub = AggNode::Metric {
        script: None,
        missing: None,
        field: "m".to_string(),
        kind: ValueKind::Long,
        source: Source::DocValues,
        needs: 0xff,
    };
    for second in [false, true] {
        for size in [1, 3, 10, 40, 1 << 20] {
            let mut after: Option<Vec<CompositeAfter>> = None;
            for _ in 0..6 {
                let got = |sorted: bool| {
                    let node = AggNode::Composite {
                        sources: sources(second),
                        size,
                        after: after.clone(),
                        sorted_lead: vec![sorted],
                        subs: vec![sub.clone()],
                    };
                    run(&reader, &all(), &[node], &whole(&reader))
                        .unwrap()
                        .remove(0)
                        .remove(0)
                };
                let (want, sorted) = (got(false), got(true));
                // Counts differ for a document with two leading values: both
                // passes count it once per distinct value, as OpenSearch does.
                assert_eq!(
                    format!("{sorted:?}"),
                    format!("{want:?}"),
                    "second {second} size {size}"
                );
                let AggResult::Composite { buckets, .. } = want else {
                    panic!("{want:?}")
                };
                let Some(last) = buckets.last() else {
                    break;
                };
                after = Some(
                    last.0
                        .iter()
                        .map(|v| match v {
                            CompositeKeyValue::Long(l) => CompositeAfter::Value(*l),
                            CompositeKeyValue::Term(t) => CompositeAfter::Term(t.clone()),
                            v => panic!("{v:?}"),
                        })
                        .collect(),
                );
            }
        }
    }
}

/// The two-keyword window pass collects what the document pass does, for
/// each page size, `after`, order and sub-aggregation.
#[test]
fn composite_two_keyword_window_pass_matches_the_document_pass() {
    let reader = fixture("terms_aggs_index");
    let source = |field: &str, reverse, missing_bucket| CompositeSource {
        field: field.to_string(),
        values: CompositeValues::Keyword,
        reverse,
        missing_bucket,
        missing_order: MissingOrder::Default,
    };
    let both = |tree: &AggNode| {
        let nodes = std::slice::from_ref(tree);
        NO_WINDOWS.with(|w| w.set(true));
        let one = run(&reader, &all(), nodes, &whole(&reader)).unwrap();
        NO_WINDOWS.with(|w| w.set(false));
        let windowed = run(&reader, &all(), nodes, &whole(&reader)).unwrap();
        assert_eq!(format!("{windowed:?}"), format!("{one:?}"), "{tree:?}");
        windowed
    };
    for (first, second) in [
        (source("kw", false, false), source("sk", false, false)),
        (source("kw", true, false), source("sk", false, true)),
        (source("sk", false, true), source("kw", true, false)),
    ] {
        let sub = AggNode::Metric {
            script: None,
            missing: None,
            field: "ml".to_string(),
            kind: ValueKind::Long,
            source: Source::DocValues,
            needs: 0xff,
        };
        let node = |size, after, subs| AggNode::Composite {
            sources: vec![first.clone(), second.clone()],
            size,
            after,
            sorted_lead: Vec::new(),
            subs,
        };
        let every = both(&node(1 << 20, None, Vec::new()));
        assert!(format!("{every:?}").len() > 100);
        for size in [1, 3, 8, 1 << 20] {
            both(&node(size, None, Vec::new()));
            both(&node(size, None, vec![sub.clone()]));
            let after = vec![
                CompositeAfter::Term(b"w2".to_vec()),
                CompositeAfter::Term(b"s1".to_vec()),
            ];
            both(&node(size, Some(after), Vec::new()));
        }
    }
}

/// A `composite` paged with `after` lists the same buckets as one page
/// holding them all, whatever its sources' orders and missing buckets,
/// and with `size` small enough that the queue drops buckets.
#[test]
fn composite_pages_list_every_bucket_once() {
    let source = |field: &str, values, reverse, missing_bucket, missing_order| CompositeSource {
        field: field.to_string(),
        values,
        reverse,
        missing_bucket,
        missing_order,
    };
    let cases = [
        (
            "terms_aggs_index",
            vec![
                source(
                    "kw",
                    CompositeValues::Keyword,
                    false,
                    false,
                    MissingOrder::Default,
                ),
                source(
                    "mkw",
                    CompositeValues::Keyword,
                    true,
                    true,
                    MissingOrder::Default,
                ),
            ],
        ),
        (
            "terms_aggs_index",
            vec![
                source(
                    "sk",
                    CompositeValues::Keyword,
                    true,
                    true,
                    MissingOrder::First,
                ),
                source(
                    "kw",
                    CompositeValues::Keyword,
                    false,
                    false,
                    MissingOrder::Default,
                ),
            ],
        ),
        (
            "metric_aggs_index",
            vec![
                source("ml", CompositeValues::Long, true, true, MissingOrder::Last),
                source(
                    "d",
                    CompositeValues::Histogram {
                        kind: ValueKind::Double,
                        interval: 25.0,
                    },
                    false,
                    true,
                    MissingOrder::Default,
                ),
            ],
        ),
        (
            "metric_aggs_index",
            vec![source(
                "md",
                CompositeValues::Double(ValueKind::Double),
                false,
                false,
                MissingOrder::Default,
            )],
        ),
    ];
    // Buckets compared with their doubles as bits: a histogram of NaN is
    // keyed NaN.
    fn bits(buckets: &[(Vec<CompositeKeyValue>, u64)]) -> Vec<String> {
        buckets
            .iter()
            .map(|(key, docs)| {
                let parts: Vec<String> = key
                    .iter()
                    .map(|v| match v {
                        CompositeKeyValue::Double(d) => format!("d{:x}", d.to_bits()),
                        v => format!("{v:?}"),
                    })
                    .collect();
                format!("{parts:?} {docs}")
            })
            .collect()
    }
    for (index, sources) in cases {
        let reader = fixture(index);
        let node = |size, after, subs| AggNode::Composite {
            sources: sources.clone(),
            size,
            after,
            sorted_lead: Vec::new(),
            subs,
        };
        let page = |n: AggNode| {
            let got = run(&reader, &all(), &[n], &whole(&reader))
                .unwrap()
                .remove(0)
                .remove(0);
            let AggResult::Composite { buckets, subs } = got else {
                panic!("{got:?}")
            };
            (buckets, subs)
        };
        let (every, _) = page(node(1 << 20, None, Vec::new()));
        assert!(every.len() > 20, "{index}: {} buckets", every.len());
        let total: u64 = every.iter().map(|b| b.1).sum();
        assert!(total > 0);
        let to_after = |key: &[CompositeKeyValue]| -> Vec<CompositeAfter> {
            key.iter()
                .zip(&sources)
                .map(|(v, s)| match v {
                    CompositeKeyValue::Missing => CompositeAfter::Missing,
                    CompositeKeyValue::Term(t) => CompositeAfter::Term(t.clone()),
                    CompositeKeyValue::Long(l) => CompositeAfter::Value(*l),
                    CompositeKeyValue::Double(d) => {
                        assert!(!matches!(s.values, CompositeValues::Keyword));
                        CompositeAfter::Value(numeric_terms_key(
                            ValueKind::Double,
                            lucene_util::numeric_utils::double_to_sortable_long(*d),
                        ))
                    }
                })
                .collect()
        };
        // The first pages, and a page size that reaches the end.
        for (size, pages) in [(1, 12), (7, 12), (every.len() / 3 + 1, 5)] {
            let mut paged = Vec::new();
            let mut after = None;
            for _ in 0..pages {
                let (buckets, _) = page(node(size, after.clone(), Vec::new()));
                assert!(buckets.len() <= size);
                let Some(last) = buckets.last() else {
                    break;
                };
                after = Some(to_after(&last.0));
                paged.extend(buckets);
            }
            let n = paged.len().min(every.len());
            assert_eq!(bits(&paged), bits(&every[..n]), "{index} size {size}");
            assert!(n == every.len() || n == size * pages, "{index} size {size}");
        }
        // An `after` term none of the ordinals holds: the buckets past it.
        if let Some(i) = every
            .iter()
            .position(|b| matches!(b.0[0], CompositeKeyValue::Term(_)))
        {
            let mut after = to_after(&every[i].0);
            if let CompositeAfter::Term(t) = &mut after[0] {
                t.push(0);
            }
            let (got, _) = page(node(1 << 20, Some(after), Vec::new()));
            let first = &every[i].0[0];
            let reverse = sources[0].reverse;
            let want: Vec<_> = every
                .iter()
                .filter(|b| {
                    let (CompositeKeyValue::Term(a), CompositeKeyValue::Term(b)) = (first, &b.0[0])
                    else {
                        // The missing value sorts first in these cases.
                        return false;
                    };
                    // The `after` term sorts just past `a`.
                    if reverse {
                        b <= a
                    } else {
                        b > a
                    }
                })
                .cloned()
                .collect();
            assert_eq!(bits(&got), bits(&want), "{index}: after an absent term");
        }
        // A sub-aggregation sees each kept bucket's documents.
        let (with_subs, subs) = page(node(5, None, vec![metric("d", ValueKind::Double)]));
        assert_eq!(bits(&with_subs), bits(&every[..5]));
        let AggResult::Metric(states) = &subs[0] else {
            panic!("{subs:?}")
        };
        assert_eq!(states.len(), 5);
        if index == "metric_aggs_index" {
            // Every document holds a `d`.
            for (b, st) in with_subs.iter().zip(states) {
                assert!(st.count >= b.1, "{index}: {b:?} {st:?}");
            }
        }
    }
}

/// A keyword `terms` with `include`/`exclude` keeps the accepted terms'
/// buckets alone, in the ordinals numbered with an inserted `missing`
/// term among them, empty ones too under `min_doc_count: 0`.
#[test]
fn keyword_terms_keep_the_accepted_terms() {
    let reader = fixture("terms_aggs_index");
    let terms_count =
        lucene_search::global_ords::GlobalOrds::build(reader.segment_readers(), "mkw")
            .unwrap()
            .value_count();
    let every_third = |count: usize| {
        let mut bits = vec![0u64; count.div_ceil(64)];
        for o in (0..count).step_by(3) {
            bits[o >> 6] |= 1 << (o & 63);
        }
        AcceptedOrds {
            count: count as u64,
            bits: bits.into(),
        }
    };
    let node = |missing: Option<&[u8]>, accepted, zero_buckets, subs| AggNode::Terms {
        field: "mkw".to_string(),
        missing: missing.map(<[u8]>::to_vec),
        accepted,
        shard_size: 1 << 20,
        order: TermsOrder::KeyAsc,
        zero_buckets,
        subs,
    };
    let list = |n: AggNode| {
        let got = run(&reader, &all(), &[n], &whole(&reader))
            .unwrap()
            .remove(0)
            .remove(0);
        let AggResult::Terms { mut buckets, .. } = got else {
            panic!("{got:?}")
        };
        buckets.remove(0)
    };
    let sub = || vec![metric("d", ValueKind::Double)];
    // Every term, empty ones included: the third ones kept.
    let (_, every) = list(node(None, None, true, Vec::new()));
    assert_eq!(every.len(), terms_count);
    let want: Vec<_> = every
        .iter()
        .step_by(3)
        .filter(|b| b.1 > 0)
        .cloned()
        .collect();
    for subs in [Vec::new(), sub()] {
        let (other, got) = list(node(None, Some(every_third(terms_count)), false, subs));
        assert_eq!(other, 0);
        assert_eq!(got, want);
    }
    let (_, got) = list(node(None, Some(every_third(terms_count)), true, Vec::new()));
    assert_eq!(got, every.iter().step_by(3).cloned().collect::<Vec<_>>());
    // An inserted `missing` term, first: ordinal 0, accepted; the rest
    // one up.
    let (_, with_missing) = list(node(Some(b""), None, true, Vec::new()));
    assert_eq!(with_missing.len(), terms_count + 1);
    let want: Vec<_> = with_missing.iter().step_by(3).cloned().collect();
    for subs in [Vec::new(), sub()] {
        let (_, got) = list(node(
            Some(b""),
            Some(every_third(terms_count + 1)),
            subs.is_empty(),
            subs.clone(),
        ));
        let want: Vec<_> = if subs.is_empty() {
            want.clone()
        } else {
            want.iter().filter(|b| b.1 > 0).cloned().collect()
        };
        assert_eq!(got, want);
    }
    // Bits over the wrong number of ordinals are refused.
    assert!(run(
        &reader,
        &all(),
        &[node(
            Some(b""),
            Some(every_third(terms_count)),
            false,
            Vec::new()
        )],
        &whole(&reader)
    )
    .is_err());
}

/// A keyword `terms` in each order keeps the `shard_size` best buckets
/// of that order -- by count with the term ascending as tie-breaker, or
/// by term -- listed by term (descending for `_key` descending).
#[test]
fn keyword_terms_keep_the_best_buckets_of_their_order() {
    let reader = fixture("terms_aggs_index");
    let mut opened = reader.open_segments().unwrap();
    opened.open_points().unwrap();
    let segments = opened.as_open_segments();
    for q in [all(), body("w1")] {
        for field in ["kw", "mkw"] {
            let every =
                crate::terms_agg::terms(&segments, reader.segment_readers(), &q, field, 1 << 20)
                    .unwrap();
            assert_eq!(every.other_doc_count, 0);
            let total: u64 = every.buckets.iter().map(|b| b.1).sum();
            for order in [
                TermsOrder::CountDesc,
                TermsOrder::CountAsc,
                TermsOrder::KeyAsc,
                TermsOrder::KeyDesc,
            ] {
                for shard_size in [1, 3, 1000] {
                    let mut want = every.buckets.clone();
                    match order {
                        TermsOrder::CountDesc => {
                            want.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)))
                        }
                        TermsOrder::CountAsc => {
                            want.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)))
                        }
                        TermsOrder::KeyAsc => want.sort_by(|a, b| a.0.cmp(&b.0)),
                        TermsOrder::KeyDesc => want.sort_by(|a, b| b.0.cmp(&a.0)),
                    }
                    want.truncate(shard_size);
                    if order == TermsOrder::KeyDesc {
                        want.sort_by(|a, b| b.0.cmp(&a.0));
                    } else {
                        want.sort_by(|a, b| a.0.cmp(&b.0));
                    }
                    let node = AggNode::Terms {
                        accepted: None,
                        missing: None,
                        zero_buckets: false,
                        field: field.to_string(),
                        shard_size,
                        order,
                        subs: Vec::new(),
                    };
                    let got = run(&reader, &q, &[node], &whole(&reader))
                        .unwrap()
                        .remove(0)
                        .remove(0);
                    let AggResult::Terms { buckets, .. } = got else {
                        panic!("{got:?}")
                    };
                    let kept: u64 = want.iter().map(|b| b.1).sum();
                    assert_eq!(
                        buckets,
                        vec![(total - kept, want)],
                        "{field} {order:?} shard_size {shard_size}"
                    );
                }
            }
        }
    }
}

#[test]
fn terms_under_a_bucket_are_the_top_level_terms_of_its_documents() {
    let reader = fixture("terms_aggs_index");
    let mut opened = reader.open_segments().unwrap();
    opened.open_points().unwrap();
    let segments = opened.as_open_segments();
    let n = reader.segment_readers().len();
    for q in [all(), body("w1")] {
        for (field, shard_size) in [("kw", 5), ("mkw", 3), ("bk", 1000), ("sk", 2)] {
            let want =
                crate::terms_agg::terms(&segments, reader.segment_readers(), &q, field, shard_size)
                    .unwrap();
            let terms = AggNode::Terms {
                accepted: None,
                missing: None,
                zero_buckets: false,
                field: field.to_string(),
                shard_size,
                order: TermsOrder::CountDesc,
                subs: vec![AggNode::Cardinality {
                    field: field.to_string(),
                    kind: CardinalityKind::Keyword,
                    precision: None,
                }],
            };
            let nodes = vec![
                terms.clone(),
                AggNode::Filters {
                    filters: vec![
                        all(),
                        BooleanQuery {
                            must: vec![Clause::PointsRange(PointsRangeQuery::new("r", 0, 99))],
                            ..Default::default()
                        },
                    ],
                    other: false,
                    subs: vec![terms],
                },
            ];
            let got = run(&reader, &q, &nodes, &whole(&reader)).unwrap().remove(0);
            let AggResult::Terms { buckets, subs } = &got[0] else {
                panic!()
            };
            assert_eq!(buckets[0].0, want.other_doc_count, "{field}");
            assert_eq!(buckets[0].1, want.buckets, "{field}");
            // A bucket's cardinality of its own field is its one term.
            let AggResult::Cardinality(sets) = &subs[0] else {
                panic!()
            };
            for ((term, _), set) in buckets[0].1.iter().zip(sets) {
                if field != "mkw" {
                    assert_eq!(set, &vec![CardinalityValue::Term(term.clone())]);
                } else {
                    assert!(set.contains(&CardinalityValue::Term(term.clone())));
                }
            }
            let AggResult::Fixed { subs, .. } = &got[1] else {
                panic!()
            };
            let AggResult::Terms { buckets: under, .. } = &subs[0] else {
                panic!()
            };
            assert_eq!(under[0].1, want.buckets, "under a match-all filter");
            assert_eq!(under[0].0, want.other_doc_count);
            assert_eq!(under.len(), 2, "one owning bucket per filter");
        }
        // Sliced: every slice from scratch; the counts add up.
        let node = AggNode::Terms {
            accepted: None,
            missing: None,
            zero_buckets: false,
            field: "kw".to_string(),
            shard_size: 10_000,
            order: TermsOrder::CountDesc,
            subs: vec![],
        };
        let slices: Vec<Vec<usize>> = (0..n).rev().map(|i| vec![i]).collect();
        let per = run(&reader, &q, std::slice::from_ref(&node), &slices).unwrap();
        let total: u64 = per
            .iter()
            .map(|r| match &r[0] {
                AggResult::Terms { buckets, .. } => buckets[0].1.iter().map(|b| b.1).sum::<u64>(),
                _ => 0,
            })
            .sum();
        let one = run(&reader, &q, &[node], &whole(&reader))
            .unwrap()
            .remove(0)
            .remove(0);
        let AggResult::Terms { buckets, .. } = one else {
            panic!()
        };
        assert_eq!(total, buckets[0].1.iter().map(|b| b.1).sum::<u64>());
    }
}

#[test]
fn a_bad_tree_or_slice_is_an_error() {
    let reader = fixture("terms_aggs_index");
    let nodes = [AggNode::Terms {
        accepted: None,
        missing: None,
        zero_buckets: false,
        field: "kw".to_string(),
        shard_size: 1,
        order: TermsOrder::CountDesc,
        subs: vec![],
    }];
    assert!(
        run(&reader, &all(), &nodes, &[vec![99]]).is_err(),
        "no such segment"
    );
    let mut opened = reader.open_segments().unwrap();
    opened.open_points().unwrap();
    let segments = opened.as_open_segments();
    let empty = HashMap::new();
    let missing = aggregate_tree(
        &segments,
        reader.segment_readers(),
        &all(),
        &nodes,
        &Globals::new(&empty),
        &whole(&reader),
        None,
    );
    assert!(missing.is_err(), "no global ordinals for the field");
    // A numeric field under terms is a mapping error.
    let numeric = [AggNode::Terms {
        accepted: None,
        missing: None,
        zero_buckets: false,
        field: "r".to_string(),
        shard_size: 1,
        order: TermsOrder::CountDesc,
        subs: vec![],
    }];
    let _ = run(&reader, &all(), &numeric, &whole(&reader));
    assert!(AggNode::Metric {
        script: None,
        missing: None,
        field: "x".into(),
        kind: ValueKind::Long,
        source: Source::PointsMin,
        needs: NEED_ALL,
    }
    .reads_points());
    let mut qs = Vec::new();
    let g = AggNode::Global {
        subs: vec![AggNode::Filters {
            filters: vec![all()],
            other: false,
            subs: vec![],
        }],
    };
    g.filter_queries(&mut qs);
    assert_eq!(qs.len(), 1);
}

/// A top-level `min`/`max` read from each segment's points bound
/// (OpenSearch's shortcut over a match-all), and the tree behind
/// `min_score`: both answer what the flat metric pass answers -- a
/// minimum of 0 every match clears, one none does.
#[test]
fn points_bounds_and_min_score_answer_what_the_metric_pass_answers() {
    let reader = fixture("metric_aggs_index");
    let mut opened = reader.open_segments().unwrap();
    opened.open_points().unwrap();
    let segments = opened.as_open_segments();
    for (field, kind) in [
        ("l", ValueKind::Long),
        ("d", ValueKind::Double),
        ("f", ValueKind::Float),
    ] {
        for source in [Source::PointsMin, Source::PointsMax] {
            let spec = MetricSpec {
                field: field.to_string(),
                kind,
                source,
                needs: NEED_ALL,
            };
            let want = crate::metric_aggs::metric_states(
                &segments,
                reader.segment_readers(),
                &all(),
                &[spec],
            )
            .unwrap();
            let node = AggNode::Metric {
                script: None,
                missing: None,
                field: field.to_string(),
                kind,
                source,
                needs: NEED_ALL,
            };
            let got = run(&reader, &all(), &[node], &whole(&reader))
                .unwrap()
                .remove(0);
            let AggResult::Metric(states) = &got[0] else {
                panic!("{got:?}")
            };
            assert_eq!(bits(&states[0]), bits(&want[0]), "{field} {source:?}");
        }
    }

    let owned = reader.field_norms_by_field(&["body".to_string()]).unwrap();
    let norms: Vec<Option<&HashMap<String, lucene_search::field_norms::FieldNorms<'_>>>> =
        owned.iter().map(Some).collect();
    let behind = |q: &BooleanQuery, min: f32| {
        let m = lucene_search::segment_match::MinScore { min, norms: &norms };
        aggregate_tree(
            &segments,
            reader.segment_readers(),
            q,
            &[metric("d", ValueKind::Double)],
            &Globals::new(&HashMap::new()),
            &whole(&reader),
            Some(&m),
        )
        .unwrap()
        .remove(0)
    };
    for q in [body("w0"), body("w3")] {
        let plain = run(
            &reader,
            &q,
            &[metric("d", ValueKind::Double)],
            &whole(&reader),
        )
        .unwrap()
        .remove(0);
        let (AggResult::Metric(want), AggResult::Metric(got)) = (&plain[0], &behind(&q, 0.0)[0])
        else {
            panic!()
        };
        assert!(want[0].count > 0);
        assert_eq!(bits(&got[0]), bits(&want[0]));
        let AggResult::Metric(none) = &behind(&q, f32::MAX)[0] else {
            panic!()
        };
        assert_eq!(none[0].count, 0);
    }
}

/// `top_hits` by score keeps what one `TopScoreDocCollector` over the
/// same matches keeps: the searcher's top hits, by global document, at
/// the top level, under `global` and per slice -- and behind `min_score`
/// only the hits that pass it.
#[test]
fn top_hits_are_the_searchers_top_docs() {
    let reader = fixture("metric_aggs_index");
    let mut opened = reader.open_segments().unwrap();
    opened.open_points().unwrap();
    let segments = opened.as_open_segments();
    let owned = reader.field_norms_by_field(&["body".to_string()]).unwrap();
    let norms: Vec<Option<&HashMap<String, lucene_search::field_norms::FieldNorms<'_>>>> =
        owned.iter().map(Some).collect();
    let searcher = lucene_search::index_searcher::IndexSearcher::new(&segments, &norms).unwrap();
    let readers = reader.segment_readers();
    assert!(readers.len() > 1, "a multi-segment fixture");
    let bases: Vec<i32> = readers
        .iter()
        .scan(0, |next, r| {
            let b = *next;
            *next += r.max_doc;
            Some(b)
        })
        .collect();
    let run_scored = |q: &BooleanQuery, nodes: &[AggNode], slices: &[Vec<usize>], min: f32| {
        let m = lucene_search::segment_match::MinScore { min, norms: &norms };
        aggregate_tree(
            &segments,
            readers,
            q,
            nodes,
            &Globals::new(&HashMap::new()),
            slices,
            Some(&m),
        )
        .unwrap()
    };
    let hits = |r: &AggResult| -> Option<(u64, Vec<(i32, u32)>)> {
        let AggResult::TopHits(per) = r else {
            panic!("{r:?}")
        };
        per[0].as_ref().map(|(total, h)| {
            (
                *total,
                h.iter().map(|d| (d.doc_id, d.score.to_bits())).collect(),
            )
        })
    };
    // Scored a whole segment's range at a time, and in ranges of a few
    // documents (not aligned to the collection windows).
    for ((q, top_n), width) in [(body("w1"), 3), (body("w3"), 50), (all(), 5)]
        .into_iter()
        .flat_map(|c| [0, 7, 100].map(move |w| (c.clone(), w)))
    {
        TEST_SCORED_RANGE.store(width, std::sync::atomic::Ordering::Relaxed);
        let every = searcher.search(&q, 100_000).unwrap();
        let oracle: Vec<(i32, u32)> = every
            .score_docs
            .iter()
            .map(|d| (d.doc, d.score.to_bits()))
            .collect();
        assert!(!oracle.is_empty());
        let want = (
            oracle.len() as u64,
            oracle.iter().copied().take(top_n).collect::<Vec<_>>(),
        );
        let top = [AggNode::TopHits { top_n }];
        let got = run_scored(&q, &top, &whole(&reader), f32::NEG_INFINITY);
        assert_eq!(hits(&got[0][0]), Some(want.clone()), "{q:?}");
        let global = [AggNode::Global {
            subs: vec![AggNode::TopHits { top_n }],
        }];
        let got = run_scored(&q, &global, &whole(&reader), f32::NEG_INFINITY);
        let AggResult::Fixed { subs, .. } = &got[0][0] else {
            panic!("{got:?}")
        };
        assert_eq!(hits(&subs[0]), Some(want.clone()));
        // A slice per segment, last first: each keeps its own segment's.
        let slices: Vec<Vec<usize>> = (0..readers.len()).rev().map(|i| vec![i]).collect();
        let got = run_scored(&q, &top, &slices, f32::NEG_INFINITY);
        for (slice, results) in slices.iter().zip(&got) {
            let i = slice[0];
            let end = bases[i] + readers[i].max_doc;
            let mine: Vec<(i32, u32)> = oracle
                .iter()
                .copied()
                .filter(|&(d, _)| d >= bases[i] && d < end)
                .collect();
            let want = (!mine.is_empty()).then(|| {
                (
                    mine.len() as u64,
                    mine.iter().copied().take(top_n).collect(),
                )
            });
            assert_eq!(hits(&results[0]), want, "segment {i}");
        }
        // Behind `min_score`: the hits at or above the median score.
        let median = f32::from_bits(oracle[oracle.len() / 2].1);
        let passing: Vec<(i32, u32)> = oracle
            .iter()
            .copied()
            .filter(|&(_, s)| f32::from_bits(s) >= median)
            .collect();
        let got = run_scored(&q, &top, &whole(&reader), median);
        assert_eq!(
            hits(&got[0][0]),
            Some((
                passing.len() as u64,
                passing.into_iter().take(top_n).collect()
            ))
        );
    }
    // Scores are the caller's to supply.
    assert!(run(
        &reader,
        &all(),
        &[AggNode::TopHits { top_n: 1 }],
        &whole(&reader)
    )
    .is_err());
}

/// `significant_terms` collects what `GlobalOrdinalsStringTermsAggregator`
/// does: every matching document counted for its bucket, and each global
/// ordinal it reads with its count, in the order first collected --
/// segment by segment, document by document, a document's ordinals in
/// order.
#[test]
fn significant_terms_collect_in_first_seen_order() {
    for (index, field, numeric) in [
        ("terms_aggs_index", "kw", false),
        ("terms_aggs_index", "mkw", false),
        ("metric_aggs_index", "ml", true),
        ("metric_aggs_index", "i", true),
    ] {
        let reader = fixture(index);
        let readers = reader.segment_readers();
        let globals = (!numeric)
            .then(|| lucene_search::global_ords::GlobalOrds::build(readers, field).unwrap());
        for q in [all(), body("w1")] {
            let mut opened = reader.open_segments().unwrap();
            opened.open_points().unwrap();
            let segments = opened.as_open_segments();
            let oracle = |slice: &[usize]| {
                let (mut subset, mut seen) = (0u64, Vec::<(i64, u64)>::new());
                let mut buf = Vec::new();
                for &i in slice {
                    let (seg, r) = (&segments[i], &readers[i]);
                    let clause = segment_match::lone_clause(&q);
                    let docs: Vec<i32> = match segment_match::segment_matches(
                        &segment_match::plain_context(seg),
                        &q,
                        &clause,
                        seg.live_docs,
                        &mut buf,
                    )
                    .unwrap()
                    {
                        None => Vec::new(),
                        Some(Some(d)) => d.to_vec(),
                        Some(None) => (0..r.max_doc)
                            .filter(|&d| seg.live_docs.is_none_or(|l| l.get_doc(d)))
                            .collect(),
                    };
                    for doc in docs {
                        subset += 1;
                        let mut v = Vec::new();
                        if let Some(g) = &globals {
                            let map = g.segment_map(i).unwrap_or(&[]);
                            match &mut lucene_search::global_ords::open_ords(r, field).unwrap().0 {
                                Ords::Absent => {}
                                Ords::Single(c) => v.extend(c.value(doc).unwrap()),
                                Ords::Multi(c) => c.values(doc, &mut v).unwrap(),
                            }
                            for o in &mut v {
                                *o = map[*o as usize];
                            }
                        } else {
                            match &mut metric_aggs::open_values(r, field).unwrap() {
                                Values::Absent => {}
                                Values::Single(c) => v.extend(c.value(doc).unwrap()),
                                Values::Multi(c) => c.values(doc, &mut v).unwrap(),
                            }
                            v.dedup();
                        }
                        for g in v {
                            match seen.iter_mut().find(|(k, _)| *k == g) {
                                Some((_, n)) => *n += 1,
                                None => seen.push((g, 1)),
                            }
                        }
                    }
                }
                (subset, seen)
            };
            let mut ords = HashMap::new();
            if let Some(g) = &globals {
                ords.insert(field.to_string(), Arc::new(g.clone()));
            }
            let node = [AggNode::SignificantTerms {
                field: field.to_string(),
                numeric,
                terms: false,
                missing_term: None,
                missing: None,
                text: None,
                subs: vec![],
            }];
            let slices: Vec<Vec<usize>> = vec![
                (0..readers.len()).collect(),
                (0..readers.len()).rev().collect(),
            ];
            for slice in &slices {
                let got = aggregate_tree(
                    &segments,
                    readers,
                    &q,
                    &node,
                    &Globals::new(&ords),
                    std::slice::from_ref(slice),
                    None,
                )
                .unwrap();
                let AggResult::Significant { buckets, subs } = &got[0][0] else {
                    panic!("{got:?}")
                };
                assert!(subs.is_empty());
                let (subset, seen) = oracle(slice);
                assert!(subset > 0 && !seen.is_empty());
                let want: Vec<(SignificantKey, u64)> = seen
                    .into_iter()
                    .map(|(k, n)| (SignificantKey::Long(k), n))
                    .collect();
                assert_eq!(buckets, &vec![(subset, want)], "{field} {q:?} {slice:?}");
            }
        }
    }
}

/// A sorted `top_hits` keeps what one `TopFieldCollector` over the same
/// matches keeps: the sorted search's hits and count, at the top level and
/// under `global`; with `track_scores`, each kept hit's score and the max.
#[test]
fn sorted_top_hits_are_the_sorted_searchs_hits() {
    use lucene_search::top_field::{SortField, SortType};
    for (index, q, sort, track) in [
        (
            "metric_aggs_index",
            body("w1"),
            vec![SortField::numeric("d", SortType::Double, true)],
            false,
        ),
        (
            "metric_aggs_index",
            all(),
            vec![
                SortField::numeric("i", SortType::Int, false),
                SortField::doc(),
            ],
            true,
        ),
        (
            "metric_aggs_index",
            body("w3"),
            vec![
                SortField::score(),
                SortField::numeric("f", SortType::Float, false),
            ],
            true,
        ),
        (
            "terms_aggs_index",
            all(),
            vec![SortField::string("kw", false), SortField::doc()],
            false,
        ),
        (
            "terms_aggs_index",
            body("w1"),
            vec![SortField::string("kw", true)],
            true,
        ),
    ] {
        let reader = fixture(index);
        let readers = reader.segment_readers();
        let mut opened = reader.open_segments().unwrap();
        opened.open_points().unwrap();
        let segments = opened.as_open_segments();
        let owned = reader.field_norms_by_field(&["body".to_string()]).unwrap();
        let norms: Vec<Option<&HashMap<String, lucene_search::field_norms::FieldNorms<'_>>>> =
            owned.iter().map(Some).collect();
        let sort = Arc::new(sort);
        for top_n in [1, 4, 50] {
            let want = lucene_search::top_field::search_sorted(
                &segments,
                readers,
                &q,
                &norms,
                &sort,
                top_n,
                u64::MAX,
                None,
            )
            .unwrap();
            let node = AggNode::SortedTopHits {
                top_n,
                sort: Arc::clone(&sort),
                track_scores: track,
            };
            let m = lucene_search::segment_match::MinScore {
                min: f32::NEG_INFINITY,
                norms: &norms,
            };
            for (tree, width) in [
                vec![node.clone()],
                vec![AggNode::Global {
                    subs: vec![node.clone()],
                }],
            ]
            .into_iter()
            .flat_map(|t| [0, 7].map(move |w| (t.clone(), w)))
            {
                TEST_SCORED_RANGE.store(width, std::sync::atomic::Ordering::Relaxed);
                let got = aggregate_tree(
                    &segments,
                    readers,
                    &q,
                    &tree,
                    &Globals::new(&HashMap::new()),
                    &whole(&reader),
                    tree[0].needs_scores().then_some(&m),
                )
                .unwrap();
                let r = match &got[0][0] {
                    AggResult::Fixed { subs, .. } => &subs[0],
                    r => r,
                };
                let AggResult::SortedTopHits(per) = r else {
                    panic!("{r:?}")
                };
                let hits = per[0].as_ref().unwrap();
                assert_eq!(hits.top.hits, want.hits, "{index} {sort:?} {top_n}");
                assert_eq!(hits.top.total.value, want.total.value);
                assert_eq!(hits.scores.len(), want.hits.len());
                if track {
                    let all = searcher_scores(&segments, &norms, &q);
                    for (h, s) in want.hits.iter().zip(&hits.scores) {
                        assert_eq!(s.to_bits(), all[&h.doc].to_bits());
                    }
                    let max = all.values().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                    assert_eq!(hits.max_score.to_bits(), max.to_bits());
                } else {
                    assert!(hits.scores.iter().all(|s| s.is_nan()) && hits.max_score.is_nan());
                }
            }
        }
    }
}

/// Every match's score, by global document, from the searcher.
fn searcher_scores(
    segments: &[OpenSegment<'_>],
    norms: &[Option<&HashMap<String, lucene_search::field_norms::FieldNorms<'_>>>],
    q: &BooleanQuery,
) -> HashMap<i32, f32> {
    let searcher = lucene_search::index_searcher::IndexSearcher::new(segments, norms).unwrap();
    searcher
        .search(q, 1_000_000)
        .unwrap()
        .score_docs
        .iter()
        .map(|d| (d.doc, d.score))
        .collect()
}
/// `nested` hands each owning bucket the children between a collected
/// parent and the one before it (`getParentAndChildId`), and
/// `reverse_nested` under it each of their parents once per bucket --
/// with one owner and with several (`BufferingNestedLeafBucketCollector`).
#[test]
fn nested_joins_each_parents_children_and_reverse_nested_their_parents() {
    let reader = fixture("metric_aggs_index");
    let readers = reader.segment_readers();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    // Parents: the documents with `w1`; children: those with `w2`.
    let (parents, children) = (body("w1"), body("w2"));
    let matches = |seg: &OpenSegment<'_>, q: &BooleanQuery| {
        lucene_search::top_field::nested_docs(seg, q).unwrap()
    };
    let nested = |subs| AggNode::Nested {
        parents: parents.clone(),
        children: children.clone(),
        subs,
    };
    let reverse = AggNode::ReverseNested {
        parents: parents.clone(),
        subs: vec![],
    };
    // Per filter (an owning bucket): children joined, parents reached.
    let filters = [all(), body("w3")];
    let mut want = vec![(0u64, 0u64); filters.len()];
    for seg in &segments {
        let ps = matches(seg, &parents);
        let cs = matches(seg, &children);
        let live = |d: i32| seg.live_docs.is_none_or(|l| l.get_doc(d));
        for (f, w) in filters.iter().zip(&mut want) {
            let fs = matches(seg, f);
            let mut prev = -1;
            for &p in &ps {
                if live(p) && fs.binary_search(&p).is_ok() {
                    let n = cs.iter().filter(|&&c| c > prev && c < p).count() as u64;
                    w.0 += n;
                    w.1 += u64::from(n > 0);
                }
                prev = p;
            }
        }
    }
    assert!(want[0].0 > 0 && want[1].0 > 0 && want[1].0 < want[0].0);
    let docs_of = |r: &AggResult| match r {
        AggResult::Fixed { docs, subs, .. } => (docs.clone(), subs.clone()),
        r => panic!("{r:?}"),
    };
    let run = |tree: Vec<AggNode>| {
        aggregate_tree(
            &segments,
            readers,
            &parents,
            &tree,
            &Globals::new(&HashMap::new()),
            &whole(&reader),
            None,
        )
        .unwrap()
        .remove(0)
        .remove(0)
    };
    // One owner.
    let (docs, subs) = docs_of(&run(vec![nested(vec![reverse.clone()])]));
    assert_eq!(docs, vec![want[0].0]);
    assert_eq!(docs_of(&subs[0]).0, vec![want[0].1]);
    // Two owners, each document in both or one.
    let tree = vec![AggNode::Filters {
        filters: filters.to_vec(),
        other: false,
        subs: vec![nested(vec![reverse])],
    }];
    let (_, subs) = docs_of(&run(tree));
    let (docs, rev) = docs_of(&subs[0]);
    assert_eq!(docs, vec![want[0].0, want[1].0]);
    assert_eq!(docs_of(&rev[0]).0, vec![want[0].1, want[1].1]);
}

/// A filter whose query holds a function query is prepared over the whole
/// reader first, as the request's own query is: its bits count the same
/// documents as that query run as the request's.
#[test]
fn a_filter_with_a_function_query_is_prepared_reader_wide() {
    use lucene_search::function::valuesource::IntFieldSource;
    use lucene_search::function::{FunctionRangeQuery, ValueSource};
    let dir = format!(
        "{}/../../fixtures/data/function/index",
        env!("CARGO_MANIFEST_DIR")
    );
    let reader = DirectoryReader::open(&FsDirectory::open(dir)).expect("open fixture");
    let source: Arc<dyn ValueSource> = Arc::new(IntFieldSource::new("i"));
    let function = || BooleanQuery {
        must: vec![Clause::from(FunctionRangeQuery::new(
            Arc::clone(&source),
            Some("1"),
            None,
            true,
            true,
        ))],
        ..Default::default()
    };
    let count = |q: &BooleanQuery, filter: BooleanQuery| {
        let node = AggNode::Filters {
            filters: vec![filter],
            other: false,
            subs: Vec::new(),
        };
        let got = run(&reader, q, std::slice::from_ref(&node), &whole(&reader))
            .unwrap()
            .remove(0)
            .remove(0);
        let AggResult::Fixed { docs, .. } = got else {
            panic!("{got:?}")
        };
        docs[0]
    };
    let matched = count(&function(), all());
    assert!(matched > 0, "the fixture has documents with i >= 1");
    assert_eq!(count(&all(), function()), matched);
}

/// `gallop` is `partition_point` for any hint, ascending or not.
#[test]
fn galloping_finds_the_partition_point_from_any_hint() {
    let v: Vec<i32> = (0..500).map(|i| i * 3 + (i % 7)).collect();
    for hint in [0usize, 1, 17, 250, 499, 500, 9999] {
        for doc in [-5, 0, 1, 2, 3, 700, 1000, 1497, 1498, 1600, 5000] {
            let cell = std::cell::Cell::new(hint);
            assert_eq!(
                gallop(&v, &cell, |d| d <= doc),
                v.partition_point(|&d| d <= doc),
                "hint {hint} doc {doc}"
            );
            assert_eq!(cell.get(), v.partition_point(|&d| d <= doc));
        }
    }
    assert_eq!(gallop(&[], &std::cell::Cell::new(3), |_| true), 0);
}

/// A calendar day table answers the bucket each day was put under, in any zone shift, and grows
/// both ways; days outside the span it keeps are simply unknown.
#[test]
fn calendar_days_answer_the_buckets_put_in() {
    const DAY: i64 = 86_400_000;
    for shift in [0i64, 19_800_000, -18_000_000, 3 * 3_600_000] {
        let r = DateRounding {
            kind: RoundingKind::Unit(DateUnit::Month),
            zone_ms: shift,
            region: None,
            offset: 0,
        };
        let mut days = CalendarDays::default();
        // Buckets met out of order, so the table grows at the front and the back.
        let starts = [
            1_700_000_000_000i64,
            1_650_000_000_000,
            1_800_000_000_000,
            1_600_000_000_000,
        ];
        for (ord, &v) in starts.iter().enumerate() {
            let (from, to) = r.bucket(v).unwrap();
            days.put(from, to, shift, ord as u32);
        }
        for (ord, &v) in starts.iter().enumerate() {
            let (from, to) = r.bucket(v).unwrap();
            for t in [from, from + 1, from + DAY, to - 1, to - DAY / 2] {
                assert_eq!(
                    days.get(t, shift).map(|x| x.0),
                    Some(ord as u32),
                    "shift {shift} {t}"
                );
            }
            assert_ne!(days.get(to, shift).map(|x| x.0), Some(ord as u32));
        }
        assert_eq!(days.get(1_000_000_000_000, shift), None);
    }
}

/// A keyword `cardinality` over a segment every document of which matches, none deleted, takes the terms
/// dictionary whole; a segment with deletions is read document by document. Either way the distinct terms are
/// those of the live matches, which `terms` lists.
#[test]
fn a_keyword_cardinality_over_whole_segments_equals_the_live_terms() {
    let reader = fixture("terms_aggs_index");
    let mut opened = reader.open_segments().unwrap();
    opened.open_points().unwrap();
    let segments = opened.as_open_segments();
    let with_deletes = segments.iter().filter(|s| s.live_docs.is_some()).count();
    assert!(
        with_deletes > 0 && with_deletes < segments.len(),
        "the fixture has segments of both kinds"
    );
    let node = |field: &str| AggNode::Cardinality {
        field: field.to_string(),
        kind: CardinalityKind::Keyword,
        precision: None,
    };
    for field in ["kw", "mkw", "hk", "sk"] {
        // Each segment alone, then all of them: the live terms of the matches.
        let mut slices: Vec<Vec<usize>> = (0..segments.len()).map(|i| vec![i]).collect();
        slices.push((0..segments.len()).collect());
        for slice in slices {
            let got = run(
                &reader,
                &all(),
                &[node(field)],
                std::slice::from_ref(&slice),
            )
            .unwrap()
            .remove(0)
            .remove(0);
            let AggResult::Cardinality(mut got) = got else {
                panic!("{got:?}")
            };
            let got: HashSet<Vec<u8>> = got
                .pop()
                .unwrap_or_default()
                .into_iter()
                .map(|v| match v {
                    CardinalityValue::Term(t) => t,
                    other => panic!("{other:?}"),
                })
                .collect();
            let chosen: Vec<_> = slice.iter().map(|&i| &segments[i]).collect();
            let want: HashSet<Vec<u8>> = crate::terms_agg::terms_sliced(
                &segments,
                reader.segment_readers(),
                &all(),
                field,
                usize::MAX,
                std::slice::from_ref(&slice),
            )
            .unwrap()
            .remove(0)
            .buckets
            .into_iter()
            .map(|(t, _)| t)
            .collect();
            assert_eq!(
                got,
                want,
                "{field} over segments {slice:?} ({} with deletions)",
                chosen.iter().filter(|s| s.live_docs.is_some()).count()
            );
        }
    }
}

fn region(id: &str) -> RegionZone {
    RegionZone::named(id).unwrap()
}

/// A region zone is named as the tz database spells it: a fixed offset, a
/// misspelt case or an unknown id is not one.
#[test]
fn region_zones_are_named_as_the_tz_database_spells_them() {
    let paris = region("Europe/Paris");
    assert_eq!(paris.id(), "Europe/Paris");
    assert_eq!(paris, region("Europe/Paris"));
    assert_ne!(paris, region("America/New_York"));
    assert_eq!(format!("{paris:?}"), "RegionZone(Europe/Paris)");
    for bad in ["europe/paris", "Europe/Nowhere", "+02:00", "UTC+2", ""] {
        assert!(RegionZone::named(bad).is_none(), "{bad:?}");
    }
    assert_eq!(paris.offset_ms(1_700_000_000_000), 3_600_000);
    assert_eq!(paris.offset_ms(1_690_000_000_000), 7_200_000);
    // Dates past the tz database's range take its nearest rules.
    assert_eq!(
        paris.offset_ms(i64::MAX),
        paris.offset_ms(253_402_207_199_999)
    );
    assert_eq!(
        paris.offset_ms(i64::MIN),
        paris.offset_ms(-377_705_116_800_000)
    );
}

/// A day is 23 hours long when the clocks go forward and 25 when they go
/// back; a day, week, month and hour in Paris start where the local clock
/// says, and an offset shifts them.
#[test]
fn region_zone_buckets_follow_the_clock() {
    const H: i64 = 3_600_000;
    let paris = region("Europe/Paris");
    let day =
        |utc| DateRounding::in_region(RoundingKind::Unit(DateUnit::Day), paris, 0).bucket(utc);
    // 2021-03-28 (spring forward): midnight CET is 2021-03-27T23:00Z.
    let spring = 1_616_886_000_000;
    assert_eq!(day(spring + 5 * H), Some((spring, spring + 23 * H)));
    // 2021-10-31 (fall back): midnight CEST is 2021-10-30T22:00Z.
    let fall = 1_635_631_200_000;
    assert_eq!(day(fall + 5 * H), Some((fall, fall + 25 * H)));
    // An ordinary day.
    let (f, t) = day(fall - 30 * H).unwrap();
    assert_eq!(t - f, 24 * H);
    let month = DateRounding::in_region(RoundingKind::Unit(DateUnit::Month), paris, 0);
    // October 2021 in Paris starts at 2021-09-30T22:00Z and has 745 hours.
    let oct = 1_633_039_200_000;
    assert_eq!(month.bucket(fall), Some((oct, oct + 745 * H)));
    // A 3-hour offset moves the day to start at 03:00 local.
    let shifted = DateRounding::in_region(RoundingKind::Unit(DateUnit::Day), paris, 3 * H);
    assert_eq!(shifted.round(spring + 5 * H), spring + 3 * H);
    assert_eq!(shifted.bucket(spring + 5 * H).unwrap().0, spring + 3 * H);
    // A bucket that overflows is no bucket.
    assert_eq!(month.bucket(i64::MAX), None);
    assert_eq!(shifted.bucket(i64::MIN), None);
    // A region zone has no uniform grid and no calendar cache.
    assert!(DateGrid::new(&shifted, spring).is_none());
}

/// Counting a region-zone day histogram from the points tree gives the
/// buckets of rounding each value, across both daylight-saving changes.
#[test]
fn counting_region_zone_buckets_from_points_matches_bucketing_each_value() {
    use lucene_index::document::{Document, LongPoint};
    use lucene_index::index_writer::IndexWriter;
    use lucene_index::merge_policy::upgrade::LATEST;

    let dir = lucene_store::ByteBuffersDirectory::new();
    let start = 1_609_459_200_000_i64; // 2021-01-01T00:00Z
    let mut values = Vec::new();
    {
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", LATEST).unwrap();
        w.set_ram_buffer_size_mb(1024.0).unwrap();
        for i in 0..9_000i64 {
            let v = start + i * 3_600_000 * 11 / 5 + (i % 3) * 60_000;
            values.push(v);
            let mut d = Document::new();
            d.add(LongPoint::new("ts", &[v]).unwrap());
            w.add_fields_document(&d).unwrap();
        }
        w.commit().unwrap();
    }
    let reader = DirectoryReader::open(&dir).unwrap();
    let mut opened = reader.open_segments().unwrap();
    opened.open_points().unwrap();
    let segments = opened.as_open_segments();
    let mut counted = 0;
    for zone in ["Europe/Paris", "America/Sao_Paulo", "Australia/Lord_Howe"] {
        for unit in [DateUnit::Day, DateUnit::Hour, DateUnit::Month] {
            let rounding = DateRounding::in_region(RoundingKind::Unit(unit), region(zone), 0);
            let node = AggNode::DateHistogram {
                script: None,
                missing: None,
                field: "ts".to_string(),
                rounding,
                hard_bounds: (None, None),
                subs: vec![],
            };
            let mut want: std::collections::BTreeMap<i64, u64> = std::collections::BTreeMap::new();
            for &v in &values {
                *want.entry(rounding.round(v)).or_default() += 1;
            }
            // The walk counts from the points or declines; where it counts, it is exact.
            if let Some(PointCounts::Keyed(got)) =
                count_from_points(&node, &segments[0], None).unwrap()
            {
                assert_eq!(got, want.into_iter().collect::<Vec<_>>(), "{zone} {unit:?}");
                counted += 1;
            }
        }
    }
    assert!(counted > 0, "the walk counted no region-zone histogram");
}
