//! Lucene's `BaseRangeFieldQueryTestCase` subclasses:
//! `TestIntRangeFieldQueries`, `TestLongRangeFieldQueries`,
//! `TestFloatRangeFieldQueries`, `TestDoubleRangeFieldQueries` and
//! `TestInetAddressRangeQueries`, each over its range field and queries.

use std::net::{IpAddr, Ipv6Addr};
use std::sync::Arc;

use lucene_index::document::{
    DoubleRange, FloatRange, IndexableField, InetAddressRange, IntRange, LongRange,
};
use lucene_search::document::{
    double_range, float_range, inet_address_range, int_range, long_range, RangeFieldQuery,
};
use lucene_search::extended_query::DocumentClause;
use lucene_search::query::Clause;
use lucene_test_framework::search::base_range_field::{
    expected_bbox_query_result, BaseRangeField, BoxRange, QueryType, Range, RangeFieldCase,
};
use lucene_test_framework::util::lucene_test_case::LuceneTestCase;
use lucene_test_framework::util::test_util;
use lucene_util::java_random::JavaRandom;

const FIELD: &str = "rangeField";

fn clause(q: RangeFieldQuery) -> Clause {
    DocumentClause::new(format!("{q:?}"), Arc::new(q)).into()
}

/// One numeric subclass: `$draw` is its `next*Internal()`.
macro_rules! numeric_case {
    ($case:ident, $prim:ty, $field:ident, $factories:ident, $draw:expr) => {
        struct $case;

        impl RangeFieldCase for $case {
            type R = BoxRange<$prim>;

            fn new_range_field(&self, r: &Self::R) -> Box<dyn IndexableField> {
                Box::new($field::new(FIELD, &r.min, &r.max).expect("a valid box"))
            }

            fn new_query(&self, r: &Self::R, query_type: QueryType) -> Clause {
                let q = match query_type {
                    QueryType::Intersects => {
                        $factories::new_intersects_query(FIELD, &r.min, &r.max)
                    }
                    QueryType::Contains => $factories::new_contains_query(FIELD, &r.min, &r.max),
                    QueryType::Within => $factories::new_within_query(FIELD, &r.min, &r.max),
                    QueryType::Crosses => $factories::new_crosses_query(FIELD, &r.min, &r.max),
                };
                clause(q.expect("a valid query"))
            }

            fn next_range(&self, random: &mut JavaRandom, dimensions: usize) -> Self::R {
                let draw: fn(&mut JavaRandom) -> $prim = $draw;
                let (mut a, mut b) = (Vec::new(), Vec::new());
                for _ in 0..dimensions {
                    a.push(draw(random));
                    b.push(draw(random));
                }
                BoxRange::ordered(a, b)
            }
        }
    };
}

numeric_case!(
    IntCase,
    i32,
    IntRange,
    int_range,
    |r| match r.next_int_bounded(5) {
        0 => i32::MIN,
        1 => i32::MAX,
        _ => {
            // `bpv` is below 32, so Java's `case 32` never runs
            let bpv = r.next_int_bounded(32);
            let mut v = test_util::next_int(r, 0, (1i32 << bpv).wrapping_sub(1));
            if bpv > 0 {
                v -= 1 << (bpv - 1);
            }
            v
        }
    }
);

numeric_case!(
    LongCase,
    i64,
    LongRange,
    long_range,
    |r| match r.next_int_bounded(5) {
        0 => i64::MIN,
        1 => i64::MAX,
        _ => {
            let bpv = r.next_int_bounded(64);
            let mut v = test_util::next_long(r, 0, (1i64 << bpv).wrapping_sub(1));
            if bpv > 0 {
                v -= 1 << (bpv - 1);
            }
            v
        }
    }
);

numeric_case!(
    FloatCase,
    f32,
    FloatRange,
    float_range,
    |r| match r.next_int_bounded(5) {
        0 => f32::NEG_INFINITY,
        1 => f32::INFINITY,
        _ => {
            if r.next_boolean() {
                r.next_float()
            } else {
                (r.next_int_bounded(15) - 7) as f32 / 3.0
            }
        }
    }
);

numeric_case!(
    DoubleCase,
    f64,
    DoubleRange,
    double_range,
    |r| match r.next_int_bounded(5) {
        0 => f64::NEG_INFINITY,
        1 => f64::INFINITY,
        _ => {
            if r.next_boolean() {
                r.next_double()
            } else {
                f64::from(r.next_int_bounded(15) - 7) / 3.0
            }
        }
    }
);

/// `TestInetAddressRangeQueries`: one dimension of 16-byte encoded
/// addresses, compared unsigned.
struct InetCase;

/// `InetAddressPoint.encode`: IPv4 as its IPv4-mapped IPv6 form.
fn encode(b: &[u8]) -> [u8; 16] {
    let mut out = [0u8; 16];
    if b.len() == 4 {
        out[10] = 0xff;
        out[11] = 0xff;
        out[12..].copy_from_slice(b);
    } else {
        out.copy_from_slice(b);
    }
    out
}

fn addr(b: [u8; 16]) -> IpAddr {
    IpAddr::V6(Ipv6Addr::from(b))
}

impl RangeFieldCase for InetCase {
    type R = BoxRange<[u8; 16]>;

    fn new_range_field(&self, r: &Self::R) -> Box<dyn IndexableField> {
        Box::new(
            InetAddressRange::new(FIELD, addr(r.min[0]), addr(r.max[0])).expect("a valid range"),
        )
    }

    fn new_query(&self, r: &Self::R, query_type: QueryType) -> Clause {
        let (min, max) = (addr(r.min[0]), addr(r.max[0]));
        let q = match query_type {
            QueryType::Intersects => inet_address_range::new_intersects_query(FIELD, min, max),
            QueryType::Contains => inet_address_range::new_contains_query(FIELD, min, max),
            QueryType::Within => inet_address_range::new_within_query(FIELD, min, max),
            QueryType::Crosses => inet_address_range::new_crosses_query(FIELD, min, max),
        };
        clause(q.expect("a valid query"))
    }

    /// `nextRange` ignores `dimensions`: an address range is one dimension.
    fn next_range(&self, random: &mut JavaRandom, _dimensions: usize) -> Self::R {
        let mut next = || {
            let mut b = vec![0u8; if random.next_boolean() { 4 } else { 16 }];
            match random.next_int_bounded(5) {
                0 => {}
                1 => b.fill(0xff),
                2 => b.fill(42),
                _ => random.next_bytes(&mut b),
            }
            encode(&b)
        };
        let (min, max) = (next(), next());
        BoxRange::ordered(vec![min], vec![max])
    }
}

/// Every test method of the base, for one subclass, each with its own
/// stream as each Java test method has; `testRandomBig` only under
/// `tests.nightly`.
fn run_all<C: RangeFieldCase>(name: &str, make: impl Fn() -> C) {
    type Method<C> = fn(&mut BaseRangeField<'_, C>);
    let methods: [(&str, Method<C>, bool); 6] = [
        ("random_tiny", |b| b.random_tiny(), false),
        ("random_medium", |b| b.random_medium(), false),
        ("random_big", |b| b.random_big(), true),
        ("multi_valued", |b| b.multi_valued(), false),
        ("all_equal", |b| b.all_equal(), false),
        ("low_cardinality", |b| b.low_cardinality(), false),
    ];
    for (method, run, nightly) in methods {
        let mut tc = LuceneTestCase::new(&format!("{name}::{method}"));
        if nightly && !tc.nightly() {
            continue;
        }
        let multiplier = tc.multiplier();
        run(&mut BaseRangeField::new(make(), tc.random(), multiplier));
    }
}

#[test]
fn int_range_field_queries() {
    run_all("TestIntRangeFieldQueries", || IntCase);
}

#[test]
fn long_range_field_queries() {
    run_all("TestLongRangeFieldQueries", || LongCase);
}

#[test]
fn float_range_field_queries() {
    run_all("TestFloatRangeFieldQueries", || FloatCase);
}

#[test]
fn double_range_field_queries() {
    run_all("TestDoubleRangeFieldQueries", || DoubleCase);
}

#[test]
fn inet_address_range_queries() {
    run_all("TestInetAddressRangeQueries", || InetCase);
}

/// `TestIntRangeFieldQueries.testBasics`, through the base's relations:
/// the hand-computed relation agrees with Java's expected counts.
#[test]
fn int_basics() {
    let boxes = [
        ([-10, -10], [9, 10]),
        ([10, -10], [20, 10]),
        ([-20, -20], [30, 30]),
        ([-11, -11], [1, 11]),
        ([12, 1], [15, 29]),
        ([-122, 1], [-115, 29]),
        ([i32::MIN, 1], [-11, 29]),
        ([-11, -15], [15, 20]),
    ];
    let boxes: Vec<BoxRange<i32>> = boxes
        .iter()
        .map(|(min, max)| BoxRange {
            min: min.to_vec(),
            max: max.to_vec(),
        })
        .collect();
    let q = BoxRange {
        min: vec![-11, -15],
        max: vec![15, 20],
    };
    let count = |t| {
        boxes
            .iter()
            .filter(|b| expected_bbox_query_result(&q, b, t))
            .count()
    };
    assert_eq!(count(QueryType::Intersects), 7);
    assert_eq!(count(QueryType::Within), 3);
    assert_eq!(count(QueryType::Contains), 2);
    assert_eq!(count(QueryType::Crosses), 4);
}

/// Java's `setMin`/`setMax`: a copied bound past the other one replaces
/// it instead, so the box stays ordered.
#[test]
fn copied_bounds_keep_the_box_ordered() {
    let mut r = BoxRange {
        min: vec![0],
        max: vec![10],
    };
    let other = BoxRange {
        min: vec![20],
        max: vec![-5],
    };
    r.copy_min(0, &other);
    assert_eq!((r.min[0], r.max[0]), (0, 20));
    r.copy_max(0, &other);
    assert_eq!((r.min[0], r.max[0]), (-5, 20));
    r.copy_min(
        0,
        &BoxRange {
            min: vec![-9],
            max: vec![0],
        },
    );
    assert_eq!(r.min[0], -9);
    r.copy_max(
        0,
        &BoxRange {
            min: vec![0],
            max: vec![30],
        },
    );
    assert_eq!(r.max[0], 30);
    assert_eq!(r.num_dimensions(), 1);
}

/// A query that hits the wrong documents fails `verify`.
#[test]
fn verify_fails_on_a_wrong_query() {
    struct Wrong;
    impl RangeFieldCase for Wrong {
        type R = BoxRange<i32>;
        fn new_range_field(&self, r: &Self::R) -> Box<dyn IndexableField> {
            IntCase.new_range_field(r)
        }
        // always intersects, whatever was asked
        fn new_query(&self, r: &Self::R, _: QueryType) -> Clause {
            IntCase.new_query(r, QueryType::Intersects)
        }
        fn next_range(&self, random: &mut JavaRandom, dims: usize) -> Self::R {
            IntCase.next_range(random, dims)
        }
    }
    let err = std::panic::catch_unwind(|| {
        let mut r = JavaRandom::new(3);
        BaseRangeField::new(Wrong, &mut r, 1).random_medium();
    })
    .expect_err("an intersects query answers no other relation");
    let msg = err.downcast_ref::<String>().cloned().unwrap_or_default();
    assert!(msg.contains("FAIL (iter"), "{msg}");
}
