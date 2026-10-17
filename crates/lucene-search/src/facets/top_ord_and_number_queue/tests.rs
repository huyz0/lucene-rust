//! Unit tests.
#![allow(clippy::arithmetic_side_effects)]

use super::*;

fn iv(ord: i32, v: i32) -> OrdAndValue {
    OrdAndValue {
        ord,
        value: FacetValue::Int(v),
    }
}

fn fv(ord: i32, v: f32) -> OrdAndValue {
    OrdAndValue {
        ord,
        value: FacetValue::Float(v),
    }
}

fn drain(q: &mut TopOrdAndNumberQueue) -> Vec<i32> {
    let mut out = Vec::new();
    while let Some(e) = q.pop() {
        out.push(e.ord);
    }
    out
}

#[test]
fn keeps_the_best_and_pops_worst_first() {
    let mut q = TopOrdAndNumberQueue::int(3);
    for (ord, v) in [(1, 5), (2, 9), (3, 1), (4, 7), (5, 9)] {
        q.insert_with_overflow(iv(ord, v));
    }
    assert_eq!(q.size(), 3);
    assert_eq!(q.top().unwrap().ord, 4);
    // Worst first: 7, then the tie at 9 where the greater ordinal is worse.
    assert_eq!(drain(&mut q), vec![4, 5, 2]);
    assert!(q.pop().is_none());
    assert!(q.top().is_none());
}

#[test]
fn insert_with_overflow_returns_what_fell_out() {
    let mut q = TopOrdAndNumberQueue::int(1);
    assert_eq!(q.insert_with_overflow(iv(1, 3)), None);
    // Not better than the top: the element itself comes back.
    assert_eq!(q.insert_with_overflow(iv(2, 3)), Some(iv(2, 3)));
    assert_eq!(q.insert_with_overflow(iv(3, 2)), Some(iv(3, 2)));
    // Better: the old top comes back.
    assert_eq!(q.insert_with_overflow(iv(0, 3)), Some(iv(1, 3)));
    assert_eq!(drain(&mut q), vec![0]);
}

#[test]
fn a_zero_sized_queue_keeps_nothing() {
    let mut q = TopOrdAndNumberQueue::float(0);
    assert_eq!(q.insert_with_overflow(fv(1, 1.0)), Some(fv(1, 1.0)));
    assert_eq!(q.size(), 0);
}

#[test]
fn float_values_order_by_value_and_nan_ties_on_the_ordinal() {
    let mut q = TopOrdAndNumberQueue::float(10);
    for (ord, v) in [(1, 0.5), (2, -1.0), (3, 2.0), (4, 0.5)] {
        q.insert_with_overflow(fv(ord, v));
    }
    assert_eq!(drain(&mut q), vec![2, 4, 1, 3]);
    // `NaN` compares neither less nor greater: only ordinals decide.
    assert!(ord_and_value_less_than(&fv(2, f32::NAN), &fv(1, 5.0)));
    assert!(!ord_and_value_less_than(&fv(1, f32::NAN), &fv(2, 5.0)));
    // Mixed kinds compare as floats.
    assert!(ord_and_value_less_than(&iv(1, 1), &fv(1, 1.5)));
    assert!(!ord_and_value_less_than(&fv(1, 1.5), &iv(1, 1)));
}

#[test]
fn the_heap_matches_lucenes_priority_queue_under_update_top() {
    // A queue over plain ints, updated through `top_mut` + `update_top`, as
    // `getTopDims` replaces its worst entry in place.
    let mut q = PriorityQueue::new(4, |a: &i32, b: &i32| a < b);
    for v in [5, 3, 8, 1] {
        q.add(v);
    }
    assert_eq!(q.top(), Some(&1));
    *q.top_mut().unwrap() = 9;
    q.update_top();
    assert_eq!(q.top(), Some(&3));
    let mut out = Vec::new();
    while let Some(v) = q.pop() {
        out.push(v);
    }
    assert_eq!(out, vec![3, 5, 8, 9]);
    q.update_top();
    assert!(q.top_mut().is_none());
}

#[test]
fn heap_order_survives_many_random_inserts() {
    let mut q = TopOrdAndNumberQueue::int(16);
    let mut x: u32 = 12345;
    let mut all = Vec::new();
    for ord in 0..500 {
        x = x.wrapping_mul(1_103_515_245).wrapping_add(12345);
        let v = (x >> 16) as i32 % 50;
        all.push((v, ord));
        q.insert_with_overflow(iv(ord, v));
    }
    // Best 16: greatest value, then least ordinal.
    all.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    let mut want: Vec<i32> = all.iter().take(16).map(|p| p.1).collect();
    want.reverse();
    assert_eq!(drain(&mut q), want);
}

#[test]
fn skipping_under_a_full_queues_bottom_keeps_the_same_queue() {
    // The dense count walk skips a count under `full_int_bottom`; the queue
    // must end as `insertWithOverflow` alone leaves it.
    for size in [0, 1, 5, 16] {
        let (mut plain, mut skipping) = (
            TopOrdAndNumberQueue::int(size),
            TopOrdAndNumberQueue::int(size),
        );
        let mut x: u32 = 777;
        for _ in 0..2000 {
            x = x.wrapping_mul(1_103_515_245).wrapping_add(12345);
            let ord = (x >> 8) as i32 % 300;
            let v = (x >> 20) as i32 % 40;
            plain.insert_with_overflow(iv(ord, v));
            if skipping.full_int_bottom().is_none_or(|b| v >= b) {
                skipping.insert_with_overflow(iv(ord, v));
            }
        }
        let mut a = Vec::new();
        while let Some(e) = plain.pop() {
            a.push(e);
        }
        let mut b = Vec::new();
        while let Some(e) = skipping.pop() {
            b.push(e);
        }
        assert_eq!(a, b, "size {size}");
    }
    // A float queue has no int bottom.
    let mut f = TopOrdAndNumberQueue::float(1);
    f.insert_with_overflow(fv(1, 2.0));
    assert_eq!(f.full_int_bottom(), None);
}
