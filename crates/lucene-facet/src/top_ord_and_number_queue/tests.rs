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

/// The queue as it was ported first (1-based positions over a `fn`
/// pointer), kept as the reference the 0-based heap must match element for
/// element -- ties and `NaN`s included, where only the sift steps decide
/// what stays.
struct ReferenceQueue {
    heap: Vec<OrdAndValue>,
    max_size: usize,
}

impl ReferenceQueue {
    fn less(&self, a: usize, b: usize) -> bool {
        ord_and_value_less_than(&self.heap[a - 1], &self.heap[b - 1])
    }

    fn up_heap(&mut self, mut i: usize) {
        let mut j = i >> 1;
        while j > 0 && self.less(i, j) {
            self.heap.swap(i - 1, j - 1);
            i = j;
            j >>= 1;
        }
    }

    fn down_heap(&mut self, mut i: usize) {
        let size = self.heap.len();
        let mut j = i * 2;
        let mut k = j + 1;
        if k <= size && self.less(k, j) {
            j = k;
        }
        while j <= size && self.less(j, i) {
            self.heap.swap(i - 1, j - 1);
            i = j;
            j = i * 2;
            k = j + 1;
            if k <= size && self.less(k, j) {
                j = k;
            }
        }
    }

    fn insert_with_overflow(&mut self, e: OrdAndValue) -> Option<OrdAndValue> {
        if self.heap.len() < self.max_size {
            self.heap.push(e);
            let last = self.heap.len();
            self.up_heap(last);
            None
        } else if !self.heap.is_empty() && ord_and_value_less_than(&self.heap[0], &e) {
            let old = std::mem::replace(&mut self.heap[0], e);
            self.down_heap(1);
            Some(old)
        } else {
            Some(e)
        }
    }

    fn pop(&mut self) -> Option<OrdAndValue> {
        if self.heap.is_empty() {
            return None;
        }
        let r = self.heap.swap_remove(0);
        if !self.heap.is_empty() {
            self.down_heap(1);
        }
        Some(r)
    }
}

#[test]
fn the_heap_matches_the_one_based_reference_on_ties_and_nans() {
    let mut x: u64 = 0x2545_f491;
    let mut next = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    for size in [1usize, 2, 3, 7, 10, 64, 1000] {
        for float in [false, true] {
            let mut q = if float {
                TopOrdAndNumberQueue::float(size)
            } else {
                TopOrdAndNumberQueue::int(size)
            };
            let mut r = ReferenceQueue {
                heap: Vec::new(),
                max_size: size,
            };
            for step in 0..20_000 {
                // Few distinct values (many ties), the same ordinal now and
                // then, a `NaN` in the float queue.
                let ord = (next() % 5000) as i32;
                let v = (next() % 8) as i32;
                let e = if !float {
                    iv(ord, v)
                } else if v == 7 {
                    fv(ord, f32::NAN)
                } else {
                    fv(ord, v as f32 * 0.5)
                };
                let (a, b) = (q.insert_with_overflow(e), r.insert_with_overflow(e));
                // `NaN != NaN`: compare the bits.
                assert_eq!(bits(a), bits(b), "size {size} step {step}");
                if step % 997 == 0 {
                    assert_eq!(bits(q.pop()), bits(r.pop()));
                }
                assert_eq!(bits(q.top().copied()), bits(r.heap.first().copied()));
            }
            loop {
                let (a, b) = (q.pop(), r.pop());
                assert_eq!(bits(a), bits(b), "size {size}");
                if a.is_none() {
                    break;
                }
            }
        }
    }
}

/// An element as comparable bits (a `NaN` equal to itself).
fn bits(e: Option<OrdAndValue>) -> Option<(i32, u32, bool)> {
    e.map(|e| match e.value {
        FacetValue::Int(v) => (e.ord, v as u32, false),
        FacetValue::Float(v) => (e.ord, v.to_bits(), true),
    })
}
