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

/// Out of line and `#[cold]`, so the bound check in `get`/`get_and_set` is a
/// single never-taken branch rather than inlined formatting machinery.
#[cold]
#[inline(never)]
fn out_of_range(i: usize, length: usize) -> ! {
    panic!("index={i},length={length}");
}

/// `SparseFixedBitSet`.
///
/// Invariant (every method keeps it; `check_structure` verifies it):
/// `indices.len() == bits.len() == block_count(length)`, and for every block
/// `k`, `indices[k] != 0` implies `bits[k]` is `Some(block)` with
/// `block.len() >= indices[k].count_ones()`. The unchecked reads in
/// [`SparseFixedBitSet::get`] and [`SparseFixedBitSet::get_and_set`] rely on
/// it.
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
        if i >= self.length {
            out_of_range(i, self.length);
        }
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
        debug_assert!(i4096 < self.indices.len() && i4096 < self.bits.len());
        // SAFETY: `i < length` (checked above), so `i >> 12 <
        // block_count(length) == indices.len()` by the type invariant.
        let index = unsafe { *self.indices.get_unchecked(i4096) };
        let i64bit = 1u64 << ((i >> 6) & 63);
        if index & i64bit == 0 {
            return false;
        }
        let o = (index & (i64bit - 1)).count_ones() as usize;
        // SAFETY: `i4096 < bits.len()` as above; `index != 0` (its `i64bit`
        // is set), so by the invariant `bits[i4096]` is `Some` and holds at
        // least `index.count_ones()` words, more than `o`, the count of the
        // index bits strictly below `i64bit`.
        unsafe {
            let block = self.bits.get_unchecked(i4096);
            debug_assert!(block.as_ref().is_some_and(|b| o < b.len()));
            let block = block.as_ref().unwrap_unchecked();
            *block.get_unchecked(o) & (1u64 << (i & 63)) != 0
        }
    }

    /// `getAndSet(i)`.
    #[inline]
    pub fn get_and_set(&mut self, i: usize) -> bool {
        self.check(i);
        let i4096 = i >> 12;
        debug_assert!(i4096 < self.indices.len() && i4096 < self.bits.len());
        // SAFETY: as in `get`: `i < length`, so `i4096 < indices.len()`.
        let index = unsafe { *self.indices.get_unchecked(i4096) };
        let i64bit = 1u64 << ((i >> 6) & 63);
        if index & i64bit != 0 {
            let o = (index & (i64bit - 1)).count_ones() as usize;
            let bit = 1u64 << (i & 63);
            // SAFETY: as in `get`: `index` has `i64bit` set, so the block
            // exists and has more than `o` words.
            let word = unsafe {
                let block = self.bits.get_unchecked_mut(i4096);
                debug_assert!(block.as_ref().is_some_and(|b| o < b.len()));
                block.as_mut().unwrap_unchecked().get_unchecked_mut(o)
            };
            let was = *word & bit != 0;
            *word |= bit;
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

    /// The 64-bit word `w` (bits `64 * w ..`), zero when not stored.
    /// `w` must be below `len().div_ceil(64)`.
    #[inline]
    pub fn word(&self, w: usize) -> u64 {
        let i4096 = w >> 6;
        let index = self.indices[i4096];
        let i64bit = 1u64 << (w & 63);
        if index & i64bit == 0 {
            return 0;
        }
        let o = (index & (i64bit - 1)).count_ones() as usize;
        self.bits[i4096].as_ref().expect("block")[o]
    }

    /// `nextClearBit(index)`.
    pub fn next_clear_bit(&self, index: usize) -> Option<usize> {
        self.next_clear_bit_in_range(index, self.length)
    }

    /// `nextClearBit(start, upperBound)`: the first clear bit in
    /// `start..upper_bound`.
    pub fn next_clear_bit_in_range(&self, start: usize, upper_bound: usize) -> Option<usize> {
        let upper_bound = upper_bound.min(self.length);
        if start >= upper_bound {
            return None;
        }
        // Word by word: a word absent from its block's index is all clear;
        // otherwise the first zero of the stored word (masked from `start`).
        let mut w = start >> 6;
        let mut from = start & 63;
        let last = (upper_bound - 1) >> 6;
        while w <= last {
            let i4096 = w >> 6;
            let index = self.indices[i4096];
            let i64bit = 1u64 << (w & 63);
            let r = if index & i64bit == 0 {
                (w << 6) + from
            } else {
                let o = (index & (i64bit - 1)).count_ones() as usize;
                let word = self.bits[i4096].as_ref().expect("block")[o];
                let clear = !word & (u64::MAX << from);
                if clear == 0 {
                    w += 1;
                    from = 0;
                    continue;
                }
                (w << 6) + clear.trailing_zeros() as usize
            };
            return (r < upper_bound).then_some(r);
        }
        None
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

    /// `TestSparseFixedBitSet.assertEquals`'s structural half, for the test
    /// framework: each block holds at least as many words as its index has
    /// bits, every word past that count is zero, and the counts sum to
    /// `nonZeroLongCount`. `Err` names the first violation.
    #[cfg(any(test, feature = "test-support"))]
    pub fn check_structure(&self) -> Result<(), String> {
        let mut total = 0;
        for (i, (&index, block)) in self.indices.iter().zip(&self.bits).enumerate() {
            let n = index.count_ones() as usize;
            if n == 0 {
                continue;
            }
            total += n;
            let block = block.as_deref().unwrap_or(&[]);
            if block.len() < n {
                return Err(format!(
                    "block {i} holds {} words for {n} index bits",
                    block.len()
                ));
            }
            if let Some(j) = block[n..].iter().position(|&w| w != 0) {
                return Err(format!(
                    "block {i} word {} is set past its count {n}",
                    n + j
                ));
            }
        }
        if total != self.non_zero_long_count {
            return Err(format!(
                "nonZeroLongCount {} != {total}",
                self.non_zero_long_count
            ));
        }
        Ok(())
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
mod tests;
