#![cfg(test)]
#![allow(clippy::needless_range_loop, clippy::identity_op)]

use super::*;

#[test]
fn check_structure_sees_each_corruption() {
    let mut s = SparseFixedBitSet::new(10_000);
    for i in [3, 70, 4100, 9000] {
        s.set(i);
    }
    assert_eq!(s.check_structure(), Ok(()));
    let mut short = s.clone();
    // Block 0 holds words 0 and 1 (bits 3 and 70).
    short.bits[0].as_mut().unwrap().truncate(1);
    assert!(short.check_structure().unwrap_err().contains("words for"));
    let mut extra = s.clone();
    let block = extra.bits[0].as_mut().unwrap();
    block.resize(block.len().max(3), 0);
    block[2] = 1;
    assert!(extra
        .check_structure()
        .unwrap_err()
        .contains("past its count"));
    let mut count = s.clone();
    count.non_zero_long_count += 1;
    assert!(count
        .check_structure()
        .unwrap_err()
        .contains("nonZeroLongCount"));
}

fn rng(s: &mut u64) -> u64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *s >> 17
}

#[test]
fn random_ops_match_a_dense_model() {
    for &n in &[1usize, 100, 4096, 4097, 20000] {
        let mut s = n as u64;
        let mut b = SparseFixedBitSet::new(n);
        let mut m = vec![false; n];
        for step in 0..3000 {
            let i = (rng(&mut s) % n as u64) as usize;
            match step % 7 {
                0..=3 => {
                    assert_eq!(b.get_and_set(i), m[i]);
                    m[i] = true;
                }
                4 => {
                    b.clear(i);
                    m[i] = false;
                }
                5 => {
                    let j = (i + (rng(&mut s) % 300) as usize).min(n);
                    b.set_range(i, j);
                    m[i..j].fill(true);
                }
                _ => {
                    let j = (i + (rng(&mut s) % 9000) as usize).min(n);
                    b.clear_range(i, j);
                    m[i..j].fill(false);
                }
            }
        }
        let card = m.iter().filter(|&&v| v).count();
        assert_eq!(b.cardinality(), card, "n={n}");
        for i in 0..n {
            assert_eq!(b.get(i), m[i]);
        }
        for i in (0..n).step_by(37) {
            assert_eq!(
                b.next_set_bit(i),
                m[i..].iter().position(|&v| v).map(|p| p + i),
                "next {i}"
            );
            assert_eq!(
                b.prev_set_bit(i),
                m[..=i].iter().rposition(|&v| v),
                "prev {i}"
            );
            assert_eq!(
                b.next_clear_bit(i),
                m[i..].iter().position(|&v| !v).map(|p| p + i)
            );
            let up = (i + 100).min(n);
            assert_eq!(
                b.next_set_bit_in_range(i, up),
                m[i..up].iter().position(|&v| v).map(|p| p + i)
            );
        }
        let approx = b.approximate_cardinality();
        assert!(approx <= n);
        assert!(b.ram_bytes_used() > 0);
        assert!(b.non_zero_long_count() <= n.div_ceil(64));
        assert!(!b.is_empty());
        assert_eq!(b.len(), n);
        b.clear_all();
        assert_eq!(b.cardinality(), 0);
        assert_eq!(b.next_set_bit(0), None);
    }
}

#[test]
fn union_and_iterator_or() {
    let mut a = SparseFixedBitSet::new(10000);
    let mut b = SparseFixedBitSet::new(10000);
    a.or_iter([1, 5000, 9999]);
    b.or_iter([2, 5000, 6000, 64, 65]);
    a.or(&b);
    let got: Vec<usize> =
        std::iter::successors(a.next_set_bit(0), |&i| a.next_set_bit(i + 1)).collect();
    assert_eq!(got, vec![1, 2, 64, 65, 5000, 6000, 9999]);
    assert!(a.to_string().starts_with("SparseFixedBitSet(size=10000"));
}

#[test]
fn block_growth_follows_oversize() {
    assert_eq!(oversize(2), 3);
    assert_eq!(oversize(34), 64);
    assert_eq!(oversize(40), 64);
    let mut b = SparseFixedBitSet::new(4096);
    for w in 0..64 {
        b.set(w * 64);
    }
    assert_eq!(b.non_zero_long_count(), 64);
    assert_eq!(b.cardinality(), 64);
    // Removing words shifts the block down.
    for w in (0..64).step_by(2) {
        b.clear(w * 64);
    }
    assert_eq!(b.non_zero_long_count(), 32);
    assert_eq!(b.prev_set_bit(4095), Some(63 * 64));
    assert_eq!(b.prev_set_bit(63), None);
}

#[test]
#[should_panic(expected = "length needs to be >= 1")]
fn zero_length_panics() {
    SparseFixedBitSet::new(0);
}

#[test]
fn next_clear_bit_word_wise_matches_model() {
    let n = 10_000;
    let mut b = SparseFixedBitSet::new(n);
    b.set_range(0, 4200);
    b.set_range(5000, 5130);
    b.set(9999);
    b.clear(70);
    let model = |i: usize| b.get(i);
    for start in (0..n)
        .step_by(13)
        .chain([4095, 4096, 4199, 4200, 9998, 9999])
    {
        for ub in [start, start + 1, start + 64, start + 5000, n] {
            let want = (start..ub.min(n)).find(|&i| !model(i));
            assert_eq!(b.next_clear_bit_in_range(start, ub), want, "{start} {ub}");
        }
    }
    assert_eq!(b.next_clear_bit(0), Some(70));
    assert_eq!(b.next_clear_bit(71), Some(4200));
    assert_eq!(b.next_clear_bit(5000), Some(5130));
    assert_eq!(b.next_clear_bit(9999), None);
    let mut full = SparseFixedBitSet::new(130);
    full.set_range(0, 130);
    assert_eq!(full.next_clear_bit(0), None);
    assert_eq!(full.next_clear_bit_in_range(3, 3), None);
}

#[test]
#[should_panic(expected = "index=10,length=10")]
fn get_and_set_out_of_range_panics() {
    SparseFixedBitSet::new(10).get_and_set(10);
}

#[test]
fn get_and_set_edges_of_blocks_and_words() {
    // Exercises the unchecked reads at the first and last word of a
    // block, the last block, and a block grown past its first word.
    let n = 3 * 4096 + 5;
    let mut b = SparseFixedBitSet::new(n);
    for &i in &[0usize, 63, 64, 4095, 4096, 8191, 4032, n - 1, n - 5, 1] {
        assert!(!b.get(i));
        assert!(!b.get_and_set(i));
        assert!(b.get(i));
        assert!(b.get_and_set(i));
    }
    assert_eq!(b.check_structure(), Ok(()));
    assert_eq!(b.cardinality(), 10);
}
