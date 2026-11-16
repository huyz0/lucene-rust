#![cfg(test)]
#![allow(clippy::needless_range_loop)]

use super::*;

fn model(n: usize, seed: u64) -> (FixedBitSet, Vec<bool>) {
    let mut s = seed;
    let mut b = FixedBitSet::new(n);
    let mut m = vec![false; n];
    for i in 0..n {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        if (s >> 60) & 1 == 1 {
            b.set(i);
            m[i] = true;
        }
    }
    (b, m)
}

fn check(b: &FixedBitSet, m: &[bool]) {
    for (i, &v) in m.iter().enumerate() {
        assert_eq!(b.get(i), v, "bit {i}");
    }
    assert_eq!(b.cardinality(), m.iter().filter(|&&v| v).count());
}

#[test]
fn ranges_flip_set_clear_match_a_model() {
    for n in [1usize, 63, 64, 65, 200, 1000] {
        for (from, to) in [
            (0usize, n),
            (0, 0),
            (1, n.min(3)),
            (n / 3, n - n / 5),
            (n.saturating_sub(1), n),
        ] {
            if from > to {
                continue;
            }
            let (mut b, mut m) = model(n, n as u64 + from as u64);
            b.flip_range(from, to);
            for v in &mut m[from..to] {
                *v = !*v;
            }
            check(&b, &m);
            b.set_range(from, to);
            m[from..to].fill(true);
            check(&b, &m);
            b.clear_range(from, to);
            m[from..to].fill(false);
            check(&b, &m);
            let (b2, m2) = model(n, 7);
            assert_eq!(
                b2.cardinality_range(from, to),
                m2[from..to].iter().filter(|&&v| v).count()
            );
        }
    }
}

#[test]
fn single_bit_ops_and_navigation() {
    let (mut b, m) = model(300, 3);
    for i in (0..300).step_by(7) {
        let prev = m[..=i].iter().rposition(|&v| v);
        assert_eq!(b.prev_set_bit(i), prev, "prev {i}");
        let next_clear = m[i..].iter().position(|&v| !v).map(|p| p + i);
        assert_eq!(b.next_clear_bit(i), next_clear);
        assert_eq!(
            b.next_set_bit_in_range(i, i + 3),
            m[i..(i + 3).min(300)]
                .iter()
                .position(|&v| v)
                .map(|p| p + i)
        );
    }
    let was = b.get(5);
    assert_eq!(b.get_and_set(5), was);
    assert!(b.get(5));
    assert!(b.get_and_clear(5));
    assert!(!b.get(5));
    b.flip(5);
    assert!(b.get(5));
    let mut full = FixedBitSet::new(130);
    full.set_range(0, 130);
    assert_eq!(full.next_clear_bit(0), None);
    assert_eq!(full.next_clear_bit_in_range(0, 64), None);
    assert_eq!(FixedBitSet::new(10).prev_set_bit(9), None);
}

#[test]
fn counts_and_set_algebra() {
    let (a, ma) = model(500, 11);
    let (b, mb) = model(300, 12);
    let union = (0..500).filter(|&i| ma[i] || (i < 300 && mb[i])).count();
    let andnot = (0..500).filter(|&i| ma[i] && !(i < 300 && mb[i])).count();
    assert_eq!(FixedBitSet::union_count(&a, &b), union);
    assert_eq!(FixedBitSet::and_not_count(&a, &b), andnot);
    assert!(a.intersects(&b));
    let mut x = a.clone();
    x.xor(&b);
    for i in 0..500 {
        assert_eq!(x.get(i), ma[i] ^ (i < 300 && mb[i]));
    }
    assert!(!FixedBitSet::new(10).intersects(&FixedBitSet::new(10)));
    assert!(FixedBitSet::new(100).scan_is_empty());
    assert!(!a.scan_is_empty());
    assert_eq!(a.approximate_cardinality(), a.cardinality());
    // Big enough to sample: 2048 words.
    let mut big = FixedBitSet::new(2048 * 64);
    big.set_range(0, 2048 * 64);
    let approx = big.approximate_cardinality();
    assert_eq!(approx, 2048 * 64);
    let mut arr = vec![0i32; 500];
    let n = a.into_array(10, 400, 1000, &mut arr);
    let want: Vec<i32> = (10..400)
        .filter(|&i| ma[i])
        .map(|i| 1000 + i as i32)
        .collect();
    assert_eq!(&arr[..n], &want[..]);
    let mut seen = Vec::new();
    a.for_each_in_range(10, 400, 1000, |d| seen.push(d));
    assert_eq!(seen, want);
    assert_eq!(a, a.clone());
    assert_ne!(a, b);
    assert_eq!(a.java_hash_code(), a.clone().java_hash_code());
}

#[test]
fn masks_ranges_and_capacity() {
    let mut b = FixedBitSet::new(200);
    b.or_mask(60, 0b1111, 4);
    assert!(b.get(60) && b.get(63) && b.get(62) && !b.get(64));
    b.or_mask(62, 0b1111, 4);
    assert!(b.get(65) && !b.get(66));
    let (src, ms) = model(300, 5);
    let mut dst = FixedBitSet::new(300);
    FixedBitSet::or_range(&src, 17, &mut dst, 3, 250);
    for i in 0..250 {
        assert_eq!(dst.get(3 + i), ms[17 + i]);
    }
    let mut all = FixedBitSet::new(300);
    all.set_range(0, 300);
    FixedBitSet::and_range(&src, 17, &mut all, 3, 250);
    for i in 0..250 {
        assert_eq!(all.get(3 + i), ms[17 + i]);
    }
    assert!(all.get(0) && all.get(299));

    let grown = FixedBitSet::ensure_capacity(src.clone(), 1000);
    assert!(grown.len() > 1000);
    for i in 0..300 {
        assert_eq!(grown.get(i), ms[i]);
    }
    let same = FixedBitSet::ensure_capacity(src.clone(), 10);
    assert_eq!(same.len(), 300);
    let cleared = FixedBitSet::ensure_capacity_and_clear(src.clone(), 10);
    assert!(cleared.scan_is_empty());
    let fresh = FixedBitSet::ensure_capacity_and_clear(src.clone(), 5000);
    assert!(fresh.len() > 5000 && fresh.scan_is_empty());

    let live = FixedBitSet::copy_of(100, |i| i % 3 != 0);
    let mut docs = FixedBitSet::new(50);
    docs.set_range(0, 50);
    live.apply_mask(&mut docs, 10).unwrap();
    for i in 0..50 {
        assert_eq!(docs.get(i), (i + 10) % 3 != 0);
    }
    let mut past = FixedBitSet::new(50);
    past.set(49);
    assert!(live.apply_mask(&mut past, 80).is_err());
    let mut ok = FixedBitSet::new(50);
    ok.set(1);
    assert!(live.apply_mask(&mut ok, 60).is_ok());
}
