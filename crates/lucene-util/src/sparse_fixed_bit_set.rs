//! Port of `org.apache.lucene.util.SparseFixedBitSet`: a bit set that only
//! stores the 64-bit words that have a bit set. The index space is cut into
//! blocks of 4096 bits; per block, a 64-bit `index` says which of its 64
//! words are non-zero and `bits` holds exactly those words, in order.
//!
//! Memory is proportional to the number of non-zero words, which is what
//! makes it the right `visited` set for a graph search that touches a few
//! thousand of millions of nodes, and the right accept set for a sparse
//! filter (`BitSet.of` picks it below `maxDoc / 128` documents).
//!
//! The array growth policy (`oversize`) and the word placement are Java's, so
//! `ram_bytes_used` tracks the same allocations Java's accounting does (with
//! Rust's `Vec` headers instead of JVM array headers).

/// `MASK_4096`.
const MASK_4096: usize = (1 << 12) - 1;

/// `SparseFixedBitSet.blockCount`.
fn block_count(length: usize) -> usize {
    length.div_ceil(4096)
}

/// `SparseFixedBitSet.oversize`: grow by half, straight to 64 past 50.
fn oversize(s: usize) -> usize {
    let n = s + (s >> 1);
    if n > 50 {
        64
    } else {
        n
    }
}

/// `SparseFixedBitSet`.
#[derive(Debug, Clone)]
pub struct SparseFixedBitSet {
    indices: Vec<u64>,
    /// `None` is Java's `null` block (no word set).
    bits: Vec<Option<Vec<u64>>>,
    length: usize,
    non_zero_long_count: usize,
}

impl SparseFixedBitSet {
    /// `new SparseFixedBitSet(length)`. Panics below 1, as Java throws.
    pub fn new(length: usize) -> Self {
        assert!(length >= 1, "length needs to be >= 1");
        let blocks = block_count(length);
        SparseFixedBitSet {
            indices: vec![0; blocks],
            bits: vec![None; blocks],
            length,
            non_zero_long_count: 0,
        }
    }

    /// `length()`.
    pub fn len(&self) -> usize {
        self.length
    }

    /// Never true (the length is at least 1); here for clippy's `len` rule.
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    #[inline]
    fn check(&self, i: usize) {
        assert!(i < self.length, "index={i},length={}", self.length);
    }

    /// `clear()`: every bit unset.
    pub fn clear_all(&mut self) {
        self.bits.iter_mut().for_each(|b| *b = None);
        self.indices.fill(0);
        self.non_zero_long_count = 0;
    }

    /// `cardinality()`.
    pub fn cardinality(&self) -> usize {
        self.bits
            .iter()
            .flatten()
            .flat_map(|a| a.iter())
            .map(|w| w.count_ones() as usize)
            .sum()
    }

    /// `approximateCardinality()`: estimated from the number of non-zero
    /// words, assuming random placement (linear counting).
    pub fn approximate_cardinality(&self) -> usize {
        let total_longs = self.length.div_ceil(64);
        let zero_longs = total_longs - self.non_zero_long_count;
        let estimate = crate::vector_util::java_round_f64(
            total_longs as f64 * (total_longs as f64 / zero_longs as f64).ln(),
        );
        (self.length as i64).min(estimate) as usize
    }

    /// `get(i)`.
    #[inline]
    pub fn get(&self, i: usize) -> bool {
        self.check(i);
        let i4096 = i >> 12;
        let index = self.indices[i4096];
        let i64bit = 1u64 << ((i >> 6) & 63);
        if index & i64bit == 0 {
            return false;
        }
        let o = (index & (i64bit - 1)).count_ones() as usize;
        let block = self.bits[i4096]
            .as_ref()
            .expect("non-zero index has a block");
        block[o] & (1u64 << (i & 63)) != 0
    }

    /// `getAndSet(i)`.
    pub fn get_and_set(&mut self, i: usize) -> bool {
        self.check(i);
        let i4096 = i >> 12;
        let index = self.indices[i4096];
        let i64bit = 1u64 << ((i >> 6) & 63);
        if index & i64bit != 0 {
            let o = (index & (i64bit - 1)).count_ones() as usize;
            let bit = 1u64 << (i & 63);
            let block = self.bits[i4096].as_mut().expect("block");
            let was = block[o] & bit != 0;
            block[o] |= bit;
            was
        } else {
            self.insert(i4096, i64bit, 1u64 << (i & 63), index);
            false
        }
    }

    /// `set(i)`.
    #[inline]
    pub fn set(&mut self, i: usize) {
        self.get_and_set(i);
    }

    /// `insertBlock` / `insertLong(Value)`: add a new non-zero word.
    fn insert(&mut self, i4096: usize, i64bit: u64, value: u64, index: u64) {
        if index == 0 {
            self.indices[i4096] = i64bit;
            debug_assert!(self.bits[i4096].is_none());
            self.bits[i4096] = Some(vec![value]);
            self.non_zero_long_count += 1;
            return;
        }
        self.indices[i4096] |= i64bit;
        let o = (index & (i64bit - 1)).count_ones() as usize;
        let block = self.bits[i4096]
            .as_mut()
            .expect("non-zero index has a block");
        if *block.last().expect("non-empty") == 0 {
            // A free slot at the end: shift right by one.
            let n = block.len();
            block.copy_within(o..n - 1, o + 1);
            block[o] = value;
        } else {
            let new_size = oversize(block.len() + 1);
            let mut grown = vec![0u64; new_size];
            grown[..o].copy_from_slice(&block[..o]);
            grown[o] = value;
            grown[o + 1..block.len() + 1].copy_from_slice(&block[o..]);
            *block = grown;
        }
        self.non_zero_long_count += 1;
    }

    /// `set(startIndex, endIndex)`.
    pub fn set_range(&mut self, start: usize, end: usize) {
        if end <= start {
            return;
        }
        assert!(start < self.length && end <= self.length);
        let first = start >> 12;
        let last = (end - 1) >> 12;
        if first == last {
            self.set_within_block(first, start & MASK_4096, (end - 1) & MASK_4096);
        } else {
            self.set_within_block(first, start & MASK_4096, MASK_4096);
            for i in first + 1..last {
                self.set_within_block(i, 0, MASK_4096);
            }
            self.set_within_block(last, 0, (end - 1) & MASK_4096);
        }
    }

    /// `clear(i)`.
    pub fn clear(&mut self, i: usize) {
        self.check(i);
        self.and(i >> 12, (i >> 6) & 63, !(1u64 << (i & 63)));
    }

    /// `and(i4096, i64, mask)`.
    fn and(&mut self, i4096: usize, i64: usize, mask: u64) {
        let index = self.indices[i4096];
        if index & (1u64 << i64) != 0 {
            let o = (index & ((1u64 << i64) - 1)).count_ones() as usize;
            let block = self.bits[i4096].as_mut().expect("block");
            let b = block[o] & mask;
            if b == 0 {
                self.remove_long(i4096, i64, index, o);
            } else {
                block[o] = b;
            }
        }
    }

    /// `orLong(i4096, i64, newBits)`.
    fn or_long(&mut self, i4096: usize, i64: usize, new_bits: u64) {
        if new_bits == 0 {
            return;
        }
        let index = self.indices[i4096];
        let i64bit = 1u64 << i64;
        if index & i64bit != 0 {
            let o = (index & (i64bit - 1)).count_ones() as usize;
            self.bits[i4096].as_mut().expect("block")[o] |= new_bits;
        } else {
            self.insert(i4096, i64bit, new_bits, index);
        }
    }

    /// `removeLong`.
    fn remove_long(&mut self, i4096: usize, i64: usize, mut index: u64, o: usize) {
        index &= !(1u64 << i64);
        self.indices[i4096] = index;
        if index == 0 {
            self.bits[i4096] = None;
        } else {
            let length = index.count_ones() as usize;
            let block = self.bits[i4096].as_mut().expect("block");
            block.copy_within(o + 1..length + 1, o);
            block[length] = 0;
        }
        self.non_zero_long_count -= 1;
    }

    /// `clear(from, to)`.
    pub fn clear_range(&mut self, from: usize, to: usize) {
        if from >= to {
            return;
        }
        assert!(to <= self.length);
        let first = from >> 12;
        let last = (to - 1) >> 12;
        if first == last {
            self.clear_within_block(first, from & MASK_4096, (to - 1) & MASK_4096);
        } else {
            self.clear_within_block(first, from & MASK_4096, MASK_4096);
            for i in first + 1..last {
                self.non_zero_long_count -= self.indices[i].count_ones() as usize;
                self.indices[i] = 0;
                self.bits[i] = None;
            }
            self.clear_within_block(last, 0, (to - 1) & MASK_4096);
        }
    }

    /// `mask(from, to)`: bits `from..=to` of a word (`to` inclusive).
    fn mask(from: usize, to: usize) -> u64 {
        // Java: ((1L << (to - from) << 1) - 1) << from, with shifts mod 64.
        let span = to - from + 1;
        let m = if span >= 64 {
            u64::MAX
        } else {
            (1u64 << span) - 1
        };
        m << (from & 63)
    }

    fn clear_within_block(&mut self, i4096: usize, from: usize, to: usize) {
        let first = from >> 6;
        let last = to >> 6;
        if first == last {
            self.and(i4096, first, !Self::mask(from & 63, to & 63));
        } else {
            self.and(i4096, last, !Self::mask(0, to & 63));
            for i in (first + 1..last).rev() {
                self.and(i4096, i, 0);
            }
            self.and(i4096, first, !Self::mask(from & 63, 63));
        }
    }

    fn set_within_block(&mut self, i4096: usize, from: usize, to: usize) {
        let first = from >> 6;
        let last = to >> 6;
        if first == last {
            self.or_long(i4096, first, Self::mask(from & 63, to & 63));
        } else {
            self.or_long(i4096, last, Self::mask(0, to & 63));
            for i in first + 1..last {
                self.or_long(i4096, i, u64::MAX);
            }
            self.or_long(i4096, first, Self::mask(from & 63, 63));
        }
    }

    /// `firstDoc(i4096, upper)`.
    fn first_doc(&self, mut i4096: usize, upper: usize) -> Option<usize> {
        while i4096 < upper {
            let index = self.indices[i4096];
            if index != 0 {
                let i64 = index.trailing_zeros() as usize;
                let w = self.bits[i4096].as_ref().expect("block")[0];
                return Some((i4096 << 12) | (i64 << 6) | w.trailing_zeros() as usize);
            }
            i4096 += 1;
        }
        None
    }

    /// `nextSetBit(i)`: `None` for `NO_MORE_DOCS`.
    pub fn next_set_bit(&self, i: usize) -> Option<usize> {
        if i >= self.length {
            return None;
        }
        self.next_set_bit_in_range(i, self.length)
    }

    /// `nextSetBit(start, upperBound)`.
    pub fn next_set_bit_in_range(&self, start: usize, upper_bound: usize) -> Option<usize> {
        if start >= upper_bound {
            return None;
        }
        let i4096 = start >> 12;
        let index = self.indices[i4096];
        let mut i64 = (start >> 6) & 63;
        let i64bit = 1u64 << i64;
        let mut o = (index & (i64bit - 1)).count_ones() as usize;
        if index & i64bit != 0 {
            let b = self.bits[i4096].as_ref().expect("block")[o] >> (start & 63);
            if b != 0 {
                let r = start + b.trailing_zeros() as usize;
                return (r < upper_bound).then_some(r);
            }
            o += 1;
        }
        // `index >>> i64 >>> 1`
        let index_bits = (index >> i64) >> 1;
        let r = if index_bits == 0 {
            let upper = if upper_bound == self.length {
                self.indices.len()
            } else {
                block_count(upper_bound)
            };
            self.first_doc(i4096 + 1, upper)?
        } else {
            i64 += 1 + index_bits.trailing_zeros() as usize;
            let w = self.bits[i4096].as_ref().expect("block")[o];
            (i4096 << 12) | (i64 << 6) | w.trailing_zeros() as usize
        };
        (r < upper_bound).then_some(r)
    }

    /// `nextClearBit(index)`.
    pub fn next_clear_bit(&self, index: usize) -> Option<usize> {
        self.next_clear_bit_in_range(index, self.length)
    }

    /// `nextClearBit(start, upperBound)`: the first clear bit in
    /// `start..upper_bound`.
    pub fn next_clear_bit_in_range(&self, start: usize, upper_bound: usize) -> Option<usize> {
        let upper_bound = upper_bound.min(self.length);
        (start..upper_bound).find(|&i| !self.get(i))
    }

    /// `lastDoc(i4096)`.
    fn last_doc(&self, i4096: Option<usize>) -> Option<usize> {
        let mut i4096 = i4096?;
        loop {
            let index = self.indices[i4096];
            if index != 0 {
                let i64 = 63 - index.leading_zeros() as usize;
                let block = self.bits[i4096].as_ref().expect("block");
                let w = block[index.count_ones() as usize - 1];
                return Some((i4096 << 12) | (i64 << 6) | (63 - w.leading_zeros() as usize));
            }
            i4096 = i4096.checked_sub(1)?;
        }
    }

    /// `prevSetBit(i)`: `None` for Java's `-1`.
    pub fn prev_set_bit(&self, i: usize) -> Option<usize> {
        self.check(i);
        let i4096 = i >> 12;
        let index = self.indices[i4096];
        let mut i64 = (i >> 6) & 63;
        let index_bits = index & ((1u64 << i64) - 1);
        let o = index_bits.count_ones() as usize;
        if index & (1u64 << i64) != 0 {
            let sub = i & 63;
            let mask = if sub == 63 {
                u64::MAX
            } else {
                (1u64 << (sub + 1)) - 1
            };
            let b = self.bits[i4096].as_ref().expect("block")[o] & mask;
            if b != 0 {
                return Some((i4096 << 12) | (i64 << 6) | (63 - b.leading_zeros() as usize));
            }
        }
        if index_bits == 0 {
            return self.last_doc(i4096.checked_sub(1));
        }
        i64 = 63 - index_bits.leading_zeros() as usize;
        let w = self.bits[i4096].as_ref().expect("block")[o - 1];
        Some((i4096 << 12) | (i64 << 6) | (63 - w.leading_zeros() as usize))
    }

    /// `or(DocIdSetIterator)`: sets every document of an ascending stream.
    /// (Java picks between per-document `set` and a block-at-a-time dense
    /// path by the iterator's cost; both set the same bits.)
    pub fn or_iter(&mut self, docs: impl IntoIterator<Item = usize>) {
        for d in docs {
            self.set(d);
        }
    }

    /// `or(SparseFixedBitSet)`: the union, block by block.
    pub fn or(&mut self, other: &SparseFixedBitSet) {
        for i4096 in 0..other.indices.len().min(self.indices.len()) {
            let index = other.indices[i4096];
            if index == 0 {
                continue;
            }
            let block = other.bits[i4096].as_ref().expect("block");
            let mut idx = index;
            let mut o = 0;
            while idx != 0 {
                let i64 = idx.trailing_zeros() as usize;
                self.or_long(i4096, i64, block[o]);
                o += 1;
                idx &= idx - 1;
            }
        }
    }

    /// Heap bytes held (Java's `ramBytesUsed`, measured for Rust).
    pub fn ram_bytes_used(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.indices.capacity() * 8
            + self.bits.capacity() * std::mem::size_of::<Option<Vec<u64>>>()
            + self
                .bits
                .iter()
                .flatten()
                .map(|b| b.capacity() * 8)
                .sum::<usize>()
    }

    /// The number of non-zero words stored (Java's `nonZeroLongCount`).
    pub fn non_zero_long_count(&self) -> usize {
        self.non_zero_long_count
    }
}

impl std::fmt::Display for SparseFixedBitSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SparseFixedBitSet(size={},cardinality=~{}",
            self.length,
            self.approximate_cardinality()
        )
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::needless_range_loop, clippy::identity_op)]

    use super::*;

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
}
