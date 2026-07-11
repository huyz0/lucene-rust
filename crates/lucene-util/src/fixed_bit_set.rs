//! Port of `org.apache.lucene.util.FixedBitSet` — a fixed-length bitset backed by
//! `u64` words, little-endian bit order within each word (bit `i` of word `w`
//! covers doc id `w*64 + i`), matching Lucene's own layout exactly. This is the
//! in-memory shape `.liv` (live docs) files decode into.

/// Number of `u64` words needed to hold `num_bits` bits — `FixedBitSet.bits2words`.
pub fn bits2words(num_bits: usize) -> usize {
    // Lucene: (numBits - 1 >> 6) + 1, guarding numBits == 0 -> 0 words.
    if num_bits == 0 {
        0
    } else {
        ((num_bits - 1) >> 6) + 1
    }
}

#[derive(Debug, Clone)]
pub struct FixedBitSet {
    words: Vec<u64>,
    num_bits: usize,
}

impl FixedBitSet {
    /// Wraps already-decoded words (e.g. read directly off disk), matching
    /// Lucene's `new FixedBitSet(long[] storedBits, int numBits)` constructor.
    /// `words.len()` must equal `bits2words(num_bits)`.
    pub fn from_words(words: Vec<u64>, num_bits: usize) -> Self {
        debug_assert_eq!(words.len(), bits2words(num_bits));
        Self { words, num_bits }
    }

    pub fn new(num_bits: usize) -> Self {
        Self {
            words: vec![0u64; bits2words(num_bits)],
            num_bits,
        }
    }

    pub fn len(&self) -> usize {
        self.num_bits
    }

    pub fn is_empty(&self) -> bool {
        self.num_bits == 0
    }

    #[inline]
    pub fn get(&self, index: usize) -> bool {
        debug_assert!(index < self.num_bits);
        let word = self.words[index >> 6];
        (word >> (index & 63)) & 1 != 0
    }

    #[inline]
    pub fn set(&mut self, index: usize) {
        debug_assert!(index < self.num_bits);
        self.words[index >> 6] |= 1u64 << (index & 63);
    }

    #[inline]
    pub fn clear(&mut self, index: usize) {
        debug_assert!(index < self.num_bits);
        self.words[index >> 6] &= !(1u64 << (index & 63));
    }

    /// Port of `FixedBitSet.cardinality()`: total number of set bits.
    pub fn cardinality(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    pub fn words(&self) -> &[u64] {
        &self.words
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
