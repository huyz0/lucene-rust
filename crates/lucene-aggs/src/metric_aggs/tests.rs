#![cfg(test)]

/// `half_float` stores `HalfFloatPoint.halfFloatToSortableShort`; `scaled_float` the rounded product.
#[test]
fn half_float_and_scaled_float_values_decode_to_the_doubles_java_reads() {
    // sortable short of a half float: positive as is, negative flips the low 15 bits.
    let sortable = |h: u16| i64::from((h as i16) ^ (((h as i16) >> 15) & 0x7fff));
    for (bits, want) in [
        (0x0000u16, 0.0),
        (0x3c00, 1.0),
        (0xbc00, -1.0),
        (0x4900, 10.0),
        (0xc500, -5.0),
        (0x0001, 5.960464477539063e-8),
        (0x7bff, 65504.0),
        (0xfbff, -65504.0),
        (0x7c00, f64::INFINITY),
    ] {
        assert_eq!(
            to_double(ValueKind::HalfFloat, sortable(bits)),
            want,
            "{bits:#06x}"
        );
    }
    let kind = ValueKind::Scaled(100f64.to_bits());
    assert_eq!(to_double(kind, 1234), 12.34);
    assert_eq!(to_double(kind, -5), -0.05);
}
use super::*;

/// `NumericUtils.doubleToSortableLong`: how a `double` field stores `d`.
fn enc(d: f64) -> i64 {
    let bits = d.to_bits() as i64;
    bits ^ ((bits >> 63) & 0x7fff_ffff_ffff_ffff)
}

/// `extended_stats`' sum of squares is the compensated sum of each value's
/// square, whether the values come one at a time or a document's at once,
/// and never rides on a zone's statistics.
#[test]
fn the_sum_of_squares_is_the_compensated_sum_of_squared_values() {
    use lucene_codecs::adaptive_zones::ZoneStats;
    let values: Vec<i64> = (0..257).map(|i| (i * 7919) % 1001 - 400).collect();
    let (mut sum, mut delta) = (0.0, 0.0);
    for &v in &values {
        let d = v as f64;
        compensated_add(&mut sum, &mut delta, d * d);
    }
    const N: u8 = NEED_COUNT | NEED_SUM | NEED_MIN_MAX | NEED_SQ;
    let mut one = MetricState::for_needs(N);
    for &v in &values {
        one.one::<N>(v as f64);
    }
    let mut many = MetricState::for_needs(N);
    many.many::<N>(ValueKind::Long, &values);
    for s in [&one, &many] {
        assert_eq!(s.sum_of_squares().to_bits(), sum.to_bits());
        assert_eq!(s.count, values.len() as u64);
        assert_eq!((s.min, s.max), (-400.0, 600.0));
    }
    let stats = ZoneStats {
        min: -400,
        max: 600,
        sum: Some(1),
    };
    assert!(fold_zone_needs(N, one, stats, 1, ValueKind::Long).is_none());
    assert!(valid_needs(N) && !valid_needs(N | NEED_MIN) && !valid_needs(NEED_SQ << 1));
}

/// **A long window folded as integers lands on the very bits the
/// per-value compensated path does**, for every window it accepts, and
/// is refused (left to `one`) when the bound could be crossed or the
/// state carries a rounding delta.
#[test]
fn a_folded_long_window_matches_value_by_value_bits() {
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let same = |a: &MetricState, b: &MetricState| {
        a.count == b.count
            && a.sum.to_bits() == b.sum.to_bits()
            && a.delta.to_bits() == b.delta.to_bits()
            && a.min.to_bits() == b.min.to_bits()
            && a.max.to_bits() == b.max.to_bits()
            && a.min_of_mins.to_bits() == b.min_of_mins.to_bits()
            && a.max_of_maxes.to_bits() == b.max_of_maxes.to_bits()
    };
    let mut accepted = 0;
    let mut refused = 0;
    for round in 0..400 {
        // Magnitudes from tiny to past the bound, windows from 0 to 70.
        let bits = 1 + (next() % 62) as u32;
        let len = (next() % 71) as usize;
        let values: Vec<i64> = (0..len)
            .map(|_| {
                let m = (next() >> (64 - bits)) as i64;
                if next() & 1 == 0 {
                    m
                } else {
                    -m
                }
            })
            .collect();
        let mut by_value = MetricState::default();
        let mut folded = MetricState::default();
        // Half the rounds start from a state that already holds values,
        // some with a pending delta from inexact additions.
        if round % 2 == 1 {
            for v in [1_i64 << 60, 3, -(1 << 59), 7] {
                by_value.one::<NEED_ALL>(v as f64);
                folded.one::<NEED_ALL>(v as f64);
            }
        }
        let mut f = LongFold::EMPTY;
        values.iter().for_each(|&v| f.push(v));
        for &v in &values {
            by_value.one::<NEED_ALL>(v as f64);
        }
        if folded.fold_longs(&f) {
            accepted += 1;
            assert!(
                same(&by_value, &folded),
                "round {round}: {by_value:?} vs {folded:?}"
            );
        } else {
            refused += 1;
            assert!(f.count > 0, "an empty window is always taken");
        }
    }
    assert!(
        accepted > 50 && refused > 50,
        "{accepted} accepted, {refused} refused"
    );
    // A state whose running sum is already at the bound refuses.
    let mut at_bound = CountSum::default();
    at_bound.one(9_007_199_254_740_992.0);
    let mut f = LongFold::EMPTY;
    f.push(1);
    assert!(!at_bound.fold_longs(&f));
    // `Count` takes any window.
    let mut c = Count::default();
    assert!(c.fold_longs(&f));
    assert_eq!(c.0, 1);
}

/// A document's values, as a `double` column holds them.
fn doc(s: &mut MetricState, values: &[f64]) {
    let stored: Vec<i64> = values.iter().map(|&d| enc(d)).collect();
    s.many::<NEED_ALL>(ValueKind::Double, &stored);
}

/// **A NaN sum is Java's `Double.NaN`, in every build profile.** On x86
/// `inf + -inf` is the negative default NaN, and a release build (which
/// may commute the addition) returned `0xfff8...` where Java's fixture
/// records `0x7ff8...`; `metric_aggs_fixtures` failed under `--release`
/// only. The values go through `black_box` so the additions run at run
/// time, not folded; CI runs this test in release as well.
#[test]
fn a_nan_sum_has_javas_nan_bits() {
    let neg_nan = f64::from_bits(0xfff8_0000_0000_0000);
    let payload_nan = f64::from_bits(0x7ff8_0000_0000_0001);
    for values in [
        [f64::INFINITY, f64::NEG_INFINITY, 1.0],
        [f64::NEG_INFINITY, 2.0, f64::INFINITY],
        [neg_nan, 1.0, 1.0],
        [f64::INFINITY, neg_nan, f64::NEG_INFINITY],
        [payload_nan, f64::INFINITY, neg_nan],
        [f64::NAN, f64::NEG_INFINITY, f64::INFINITY],
    ] {
        let mut s = MetricState::default();
        for v in values {
            s.one::<NEED_SUM>(std::hint::black_box(v));
        }
        assert_eq!(s.sum.to_bits(), 0x7ff8_0000_0000_0000, "{values:?}");
    }
}

#[test]
fn the_sum_is_compensated_as_opensearch_compensates_it() {
    // 1 + 1e-16 ten times: a naive sum stays 1.0, Kahan's does not.
    let mut s = MetricState::default();
    doc(&mut s, &[1.0]);
    for _ in 0..10 {
        doc(&mut s, &[1e-16]);
    }
    assert!(s.sum > 1.0);
    assert_eq!(s.count, 11);
    // A non-finite value turns the tally non-finite, and it stays so.
    let mut s = MetricState::default();
    doc(&mut s, &[1.0, f64::INFINITY]);
    assert_eq!(s.sum, f64::INFINITY);
    doc(&mut s, &[5.0]);
    assert_eq!(s.sum, f64::INFINITY);
    let mut s = MetricState::default();
    doc(&mut s, &[f64::NAN]);
    assert!(s.sum.is_nan());
    // A single-valued document feeds every part the same way.
    let (mut one, mut many) = (MetricState::default(), MetricState::default());
    for v in [3.0, -0.0, 0.0, 1e300, f64::NEG_INFINITY] {
        one.one::<NEED_ALL>(v);
        doc(&mut many, &[v]);
    }
    assert_eq!(format!("{one:?}"), format!("{many:?}"));
}

#[test]
fn a_documents_own_minimum_and_maximum_feed_min_and_max() {
    // `SORTED_NUMERIC` keeps NaN last: `min` reads the first value and
    // never sees it; `stats` reads every value and does.
    let mut s = MetricState::default();
    doc(&mut s, &[1.0, f64::NAN]);
    doc(&mut s, &[]);
    doc(&mut s, &[-2.0, 4.0]);
    assert_eq!(s.min_of_mins, -2.0);
    assert!(s.max_of_maxes.is_nan());
    assert!(s.min.is_nan());
    assert_eq!(s.count, 4);
}

/// A window folded from its statistics against the same values one by
/// one: equal bit for bit when it folds, untouched when it must not.
fn zone_vs_values<const N: u8>(start: MetricState, kind: ValueKind, values: &[i64]) -> bool {
    use lucene_codecs::adaptive_zones::ZoneStats;
    let stats = ZoneStats {
        min: *values.iter().min().unwrap(),
        max: *values.iter().max().unwrap(),
        sum: values.iter().try_fold(0i64, |a, &v| a.checked_add(v)),
    };
    let mut one = start;
    for &v in values {
        one.one::<N>(to_double(kind, v));
    }
    match fold_zone::<N>(start, stats, values.len(), kind) {
        Some(zone) => {
            assert_eq!(format!("{zone:?}"), format!("{one:?}"));
            assert_eq!(zone.sum.to_bits(), one.sum.to_bits());
            assert_eq!(zone.min.to_bits(), one.min.to_bits());
            assert_eq!(zone.min_of_mins.to_bits(), one.min_of_mins.to_bits());
            assert_eq!(zone.max_of_maxes.to_bits(), one.max_of_maxes.to_bits());
            true
        }
        None => false,
    }
}

#[test]
fn a_window_folds_from_its_statistics_exactly_or_not_at_all() {
    let longs: Vec<i64> = (0..1024).map(|i| i * 3 - 700).collect();
    assert!(zone_vs_values::<NEED_ALL>(
        MetricState::default(),
        ValueKind::Long,
        &longs
    ));
    // The exact-integer bound, at the edge and one past it.
    let big = vec![1i64 << 40; 1024];
    let edge = MetricState {
        sum: ((1u64 << 53) - (1u64 << 50)) as f64,
        ..MetricState::default()
    };
    assert!(zone_vs_values::<NEED_SUM>(edge, ValueKind::Long, &big));
    let past = MetricState {
        sum: edge.sum + 1.0,
        ..edge
    };
    assert!(!zone_vs_values::<NEED_SUM>(past, ValueKind::Long, &big));
    // ...but its count and extremes still fold.
    assert!(zone_vs_values::<{ NEED_COUNT | NEED_MIN_MAX }>(
        past,
        ValueKind::Long,
        &big
    ));
    // A sum already carrying a correction, or not an integer.
    let carried = MetricState {
        delta: 1e-17,
        ..MetricState::default()
    };
    assert!(!zone_vs_values::<NEED_SUM>(
        carried,
        ValueKind::Long,
        &longs
    ));
    let half = MetricState {
        sum: 0.5,
        ..MetricState::default()
    };
    assert!(!zone_vs_values::<NEED_SUM>(half, ValueKind::Long, &longs));

    // Doubles: count and extremes, both zeros on either side, never a
    // sum, and no NaN of either sign.
    let doubles = |ds: &[f64]| ds.iter().map(|&d| enc(d)).collect::<Vec<i64>>();
    for ds in [
        [-0.0, 0.0, 1.5, 2.0],
        [0.0, -0.0, -1.5, -2.0],
        [0.0, -0.0, 0.0, -0.0],
    ] {
        let v = doubles(&ds);
        assert!(zone_vs_values::<
            { NEED_COUNT | NEED_MIN_MAX | NEED_MIN | NEED_MAX },
        >(MetricState::default(), ValueKind::Double, &v));
        assert!(!zone_vs_values::<NEED_SUM>(
            MetricState::default(),
            ValueKind::Double,
            &v
        ));
    }
    let negative_nan = f64::from_bits(0xfff8_0000_0000_0001);
    for nan in [f64::NAN, negative_nan] {
        let v = doubles(&[1.0, nan, -3.0]);
        assert!(!zone_vs_values::<NEED_MIN>(
            MetricState::default(),
            ValueKind::Double,
            &v
        ));
        assert!(!zone_vs_values::<NEED_MAX>(
            MetricState::default(),
            ValueKind::Double,
            &v
        ));
        // A count alone needs neither extreme.
        assert!(zone_vs_values::<NEED_COUNT>(
            MetricState::default(),
            ValueKind::Double,
            &v
        ));
    }

    // Floats: sign-extended sortable ints fold; anything wider does not.
    let float = |f: f32| {
        let bits = f.to_bits() as i32;
        i64::from(bits ^ ((bits >> 31) & 0x7fff_ffff))
    };
    let v: Vec<i64> = [-2.5f32, -0.0, 0.0, 7.25]
        .iter()
        .map(|&f| float(f))
        .collect();
    assert!(!zone_vs_values::<NEED_ALL>(
        MetricState::default(),
        ValueKind::Float,
        &v
    ));
    assert!(zone_vs_values::<{ NEED_MIN | NEED_MAX }>(
        MetricState::default(),
        ValueKind::Float,
        &v
    ));
    assert!(!zone_vs_values::<NEED_MIN>(
        MetricState::default(),
        ValueKind::Float,
        &[1, i64::from(u32::MAX)]
    ));
}

#[test]
fn stored_longs_become_the_doubles_an_aggregation_reads() {
    assert_eq!(to_double(ValueKind::Long, -7), -7.0);
    let d = -1.5f64;
    let bits = d.to_bits() as i64;
    let sortable = bits ^ ((bits >> 63) & 0x7fff_ffff_ffff_ffff);
    assert_eq!(to_double(ValueKind::Double, sortable), d);
    let f = 2.25f32;
    let ibits = f.to_bits() as i32;
    let sortable = ibits ^ ((ibits >> 31) & 0x7fff_ffff);
    assert_eq!(to_double(ValueKind::Float, i64::from(sortable)), 2.25);
}
