#![cfg(test)]
#![allow(clippy::needless_range_loop, clippy::identity_op)]

use super::*;
#[test]
fn into_words_is_the_inverse_of_from_words() {
    let mut b = FixedBitSet::new(130);
    b.set(0);
    b.set(129);
    let backing = b.clone().into_words();
    assert_eq!(backing.len(), bits2words(130));
    assert_eq!(FixedBitSet::from_words(backing, 130), b);
}

#[test]
fn next_clear_bit_in_words_finds_the_first_gap() {
    let words = [u64::MAX, 0b1011, 0];
    assert_eq!(next_clear_bit_in_words(&words, 0), 66);
    assert_eq!(next_clear_bit_in_words(&words, 64), 66);
    assert_eq!(next_clear_bit_in_words(&words, 67), 68);
    assert_eq!(next_clear_bit_in_words(&words, 130), 130);
    // All set from `from` on: the end of the words.
    assert_eq!(next_clear_bit_in_words(&[u64::MAX, u64::MAX], 5), 128);
    assert_eq!(next_clear_bit_in_words(&[u64::MAX], 64), 64);
    assert_eq!(next_clear_bit_in_words(&[], 0), 0);
    // Bits below `from` in its word do not count as clear.
    assert_eq!(next_clear_bit_in_words(&[!1u64], 1), 64);
}

#[test]
fn bits2words_matches_java_formula() {
    assert_eq!(bits2words(0), 0);
    assert_eq!(bits2words(1), 1);
    assert_eq!(bits2words(64), 1);
    assert_eq!(bits2words(65), 2);
    assert_eq!(bits2words(128), 2);
    assert_eq!(bits2words(129), 3);
}

#[test]
fn clear_all_unsets_every_bit_and_keeps_the_allocation() {
    let mut bs = FixedBitSet::new(200);
    for i in [0, 63, 64, 199] {
        bs.set(i);
    }
    assert_eq!(bs.cardinality(), 4);
    bs.clear_all();
    assert_eq!(bs.cardinality(), 0);
    assert_eq!(bs.len(), 200);
    // Still usable afterwards.
    bs.set(7);
    assert!(bs.get(7));
}

#[test]
fn is_empty_and_len() {
    assert!(FixedBitSet::new(0).is_empty());
    assert_eq!(FixedBitSet::new(0).len(), 0);
    let bs = FixedBitSet::new(5);
    assert!(!bs.is_empty());
    assert_eq!(bs.len(), 5);
}

#[test]
fn or_and_not_words_at_match_bit_by_bit_and_drop_ghost_bits() {
    let mut a = FixedBitSet::new(200);
    let words = [0x8000_0000_0000_0001u64, u64::MAX];
    assert!(a.or_words_at(1, &words));
    for i in 0..200 {
        let want = i == 64 || i == 127 || (128..192).contains(&i);
        assert_eq!(a.get(i), want, "bit {i}");
    }
    // Word 3 holds bits 192..256 but only 192..200 exist.
    assert!(!a.or_words_at(3, &[u64::MAX]));
    assert_eq!(a.cardinality(), 2 + 64 + 8);
    assert!(!a.or_words_at(4, &[1]));
    assert!(a.or_words_at(9, &[0]));
    assert!(a.and_not_words_at(2, &[u64::MAX]));
    assert_eq!(a.cardinality(), 2 + 8);
    assert!(!a.and_not_words_at(3, &[u64::MAX]));
    assert_eq!(a.cardinality(), 2);
}

#[test]
fn words_exposes_backing_storage() {
    let mut bs = FixedBitSet::new(70); // 2 words
    bs.set(0);
    bs.set(64);
    assert_eq!(bs.words().len(), 2);
    assert_eq!(bs.words()[0], 1);
    assert_eq!(bs.words()[1], 1);
}

#[test]
fn from_words_wraps_disk_bytes_directly() {
    // Mirrors how `live_docs::parse` constructs a FixedBitSet from raw i64
    // words read off disk, without going through set()/clear().
    let bs = FixedBitSet::from_words(vec![0b1011], 4);
    assert!(bs.get(0));
    assert!(bs.get(1));
    assert!(!bs.get(2));
    assert!(bs.get(3));
    assert_eq!(bs.cardinality(), 3);
}

#[test]
fn from_words_accepts_a_full_final_word() {
    // num_bits a multiple of 64: there are no ghost bits to check, and an
    // all-ones final word is legal.
    let bs = FixedBitSet::from_words(vec![u64::MAX], 64);
    assert_eq!(bs.cardinality(), 64);
}

#[test]
#[should_panic(expected = "bits beyond num_bits")]
fn from_words_rejects_ghost_bits_in_debug() {
    // Java asserts the same invariant in its `FixedBitSet(long[], int)`
    // constructor: bit 4 is outside a 4-bit set, and counting it would
    // inflate cardinality() (and hence a segment's live-doc count).
    FixedBitSet::from_words(vec![0b1_0000], 4);
}

#[test]
fn set_get_clear_cardinality() {
    let mut bs = FixedBitSet::new(130);
    assert_eq!(bs.cardinality(), 0);
    bs.set(0);
    bs.set(63);
    bs.set(64);
    bs.set(129);
    assert!(bs.get(0));
    assert!(bs.get(63));
    assert!(bs.get(64));
    assert!(bs.get(129));
    assert!(!bs.get(1));
    assert_eq!(bs.cardinality(), 4);
    bs.clear(64);
    assert!(!bs.get(64));
    assert_eq!(bs.cardinality(), 3);
}

/// The bound is checked in **release** as well as debug, so the three
/// tests below would still fail if the check were weakened back to a
/// `debug_assert!`. Each uses an index that is past `num_bits` but still
/// inside the backing word -- the *ghost bit* range, where `words[index >>
/// 6]` on its own catches nothing.
#[test]
#[should_panic(expected = "index 5 is out of range for a bitset of 4 bits")]
fn get_past_num_bits_but_inside_the_word_panics() {
    FixedBitSet::from_words(vec![0b1011], 4).get(5);
}

#[test]
#[should_panic(expected = "index 5 is out of range for a bitset of 4 bits")]
fn set_past_num_bits_but_inside_the_word_panics() {
    // A write here would leave a set bit above `num_bits`, which
    // `cardinality()` (a whole-word popcount, as Java's is) would then
    // count forever after.
    FixedBitSet::new(4).set(5);
}

#[test]
#[should_panic(expected = "index 5 is out of range for a bitset of 4 bits")]
fn clear_past_num_bits_but_inside_the_word_panics() {
    FixedBitSet::new(4).clear(5);
}

/// A deterministic bitset with a mix of empty, full and sparse words, so
/// the word-skipping paths in `next_set_bit` are all exercised.
fn sample(num_bits: usize, seed: u64) -> FixedBitSet {
    let mut bs = FixedBitSet::new(num_bits);
    let mut s = seed;
    for i in 0..num_bits {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        // Leave words 3..6 empty and fill word 8 completely.
        let w = i / 64;
        if (3..6).contains(&w) {
            continue;
        }
        if w == 8 || s.is_multiple_of(7) {
            bs.set(i);
        }
    }
    bs
}

fn naive_bits(bs: &FixedBitSet) -> Vec<usize> {
    (0..bs.len()).filter(|&i| bs.get(i)).collect()
}

#[test]
fn next_set_bit_walks_every_set_bit_in_order() {
    for num_bits in [1usize, 63, 64, 65, 700, 1000] {
        let bs = sample(num_bits, 0x1234 + num_bits as u64);
        let expected = naive_bits(&bs);
        let mut got = Vec::new();
        let mut i = bs.next_set_bit(0);
        while let Some(b) = i {
            got.push(b);
            i = bs.next_set_bit(b + 1);
        }
        assert_eq!(got, expected, "num_bits={num_bits}");
        let mut each = Vec::new();
        bs.for_each_set_bit(|b| each.push(b));
        assert_eq!(each, expected, "for_each num_bits={num_bits}");
    }
}

#[test]
fn next_set_bit_from_the_middle_of_a_word_and_past_the_end() {
    let mut bs = FixedBitSet::new(200);
    bs.set(3);
    bs.set(70);
    bs.set(199);
    assert_eq!(bs.next_set_bit(0), Some(3));
    assert_eq!(bs.next_set_bit(3), Some(3));
    assert_eq!(bs.next_set_bit(4), Some(70));
    assert_eq!(bs.next_set_bit(71), Some(199));
    assert_eq!(bs.next_set_bit(199), Some(199));
    assert_eq!(bs.next_set_bit(200), None);
    assert_eq!(bs.next_set_bit(usize::MAX), None);
    assert_eq!(FixedBitSet::new(0).next_set_bit(0), None);
    assert_eq!(FixedBitSet::new(130).next_set_bit(5), None);
}

#[test]
fn cardinality_matches_a_naive_count_for_every_tail_length() {
    for num_bits in [0usize, 1, 64, 128, 191, 256, 257, 1000] {
        let bs = sample(num_bits, 99 + num_bits as u64);
        assert_eq!(
            bs.cardinality(),
            naive_bits(&bs).len(),
            "num_bits={num_bits}"
        );
    }
}

#[test]
fn or_and_and_not_and_intersection_count_match_naive_set_algebra() {
    for num_bits in [5usize, 64, 300, 1000] {
        let a = sample(num_bits, 7 + num_bits as u64);
        let b = sample(num_bits, 1_000_003 + num_bits as u64);
        let (na, nb) = (naive_bits(&a), naive_bits(&b));
        let both: Vec<usize> = na.iter().copied().filter(|i| nb.contains(i)).collect();
        assert_eq!(FixedBitSet::intersection_count(&a, &b), both.len());

        let mut or = a.clone();
        or.or(&b);
        let mut union: Vec<usize> = na.iter().chain(&nb).copied().collect();
        union.sort_unstable();
        union.dedup();
        assert_eq!(naive_bits(&or), union);

        let mut and = a.clone();
        and.and(&b);
        assert_eq!(naive_bits(&and), both);

        let mut diff = a.clone();
        diff.and_not(&b);
        let only_a: Vec<usize> = na.iter().copied().filter(|i| !nb.contains(i)).collect();
        assert_eq!(naive_bits(&diff), only_a);
    }
}

#[test]
fn or_accepts_a_shorter_bitset_and_and_clears_past_it() {
    let mut long = FixedBitSet::new(300);
    long.set(10);
    long.set(250);
    let mut short = FixedBitSet::new(100);
    short.set(10);
    short.set(20);
    let mut or = long.clone();
    or.or(&short);
    assert_eq!(naive_bits(&or), vec![10, 20, 250]);
    let mut and = long.clone();
    and.and(&short);
    assert_eq!(naive_bits(&and), vec![10]);
    assert_eq!(FixedBitSet::intersection_count(&long, &short), 1);
}

#[test]
#[should_panic(expected = "or: other has 300 bits, this bitset only 100")]
fn or_with_a_longer_bitset_panics() {
    FixedBitSet::new(100).or(&FixedBitSet::new(300));
}

#[test]
#[should_panic(expected = "index 0 is out of range for a bitset of 0 bits")]
fn an_empty_bitset_has_no_index_at_all() {
    // `bits2words(0) == 0`, so `words` is empty and `words[0]` would panic
    // on its own -- but with the message of a slice index, not of a
    // bitset bound.
    FixedBitSet::new(0).get(0);
}

mod range_ops_props {
    use super::super::FixedBitSet;
    use proptest::prelude::*;

    fn bitset(words: &[u64], len: usize) -> FixedBitSet {
        FixedBitSet::copy_of(len, |i| words[i >> 6] >> (i & 63) & 1 == 1)
    }

    proptest! {
        #[test]
        fn range_ops_match_bit_by_bit(
            sw in proptest::collection::vec(any::<u64>(), 4),
            dw in proptest::collection::vec(any::<u64>(), 4),
            slen in 0usize..=256, dlen in 0usize..=256,
            a in 0usize..=256, b in 0usize..=256, c in 0usize..=256,
        ) {
            let src = bitset(&sw, slen);
            let dst = bitset(&dw, dlen);
            for i in 0..slen {
                prop_assert_eq!(src.get(i), sw[i >> 6] >> (i & 63) & 1 == 1);
            }
            let sf = a.min(slen);
            let df = b.min(dlen);
            let length = c.min(slen - sf).min(dlen - df);
            let mut or = dst.clone();
            FixedBitSet::or_range(&src, sf, &mut or, df, length);
            let mut and = dst.clone();
            FixedBitSet::and_range(&src, sf, &mut and, df, length);
            for i in 0..dlen {
                let inside = i >= df && i < df + length;
                let s = inside && src.get(sf + i - df);
                prop_assert_eq!(or.get(i), dst.get(i) || s, "or bit {}", i);
                prop_assert_eq!(and.get(i), dst.get(i) && (!inside || s), "and bit {}", i);
            }
            let (from, to) = (a.min(slen), b.min(slen));
            let (from, to) = (from.min(to), from.max(to));
            let want: Vec<i32> = (from..to).filter(|&i| src.get(i)).map(|i| i as i32 + 7).collect();
            prop_assert_eq!(src.cardinality_range(from, to), want.len());
            let mut seen = Vec::new();
            src.for_each_in_range(from, to, 7, |d| seen.push(d));
            prop_assert_eq!(&seen, &want);
            let mut arr = vec![0i32; c % 9];
            let n = src.into_array(from, to, 7, &mut arr);
            prop_assert_eq!(n, want.len().min(arr.len()));
            prop_assert_eq!(&arr[..n], &want[..n]);
        }
    }
}
