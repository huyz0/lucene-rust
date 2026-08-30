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

/// Port of `FixedBitSet.verifyGhostBitsClear()` (Java runs it as a
/// constructor `assert`): the bits above `num_bits` in the final word must be
/// zero. `cardinality()` counts whole words, exactly as Lucene's does, so a
/// `.liv` file whose trailing word carries junk would silently inflate the
/// live-doc count rather than being rejected.
fn ghost_bits_clear(words: &[u64], num_bits: usize) -> bool {
    if num_bits & 0x3f == 0 {
        return true;
    }
    let mask = u64::MAX << (num_bits & 0x3f);
    words[bits2words(num_bits) - 1] & mask == 0
}

/// Out of line and `#[cold]` so the bound check above costs a single
/// never-taken branch in the hot loops that index a bitset per document or
/// per graph node, rather than inlining a panic's formatting machinery into
/// each of them.
#[cold]
#[inline(never)]
fn out_of_range(index: usize, num_bits: usize) -> ! {
    panic!("FixedBitSet index {index} is out of range for a bitset of {num_bits} bits");
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
    /// Java additionally allows `storedBits` to be *longer* than
    /// `bits2words(numBits)` (the surplus words must be zero); this port
    /// requires the exact length, since every caller here decodes exactly
    /// `bits2words(numBits)` words off disk and `cardinality()`/`words()`
    /// would otherwise have to distinguish "words in use" from "words
    /// allocated" for no gain.
    pub fn from_words(words: Vec<u64>, num_bits: usize) -> Self {
        debug_assert_eq!(words.len(), bits2words(num_bits));
        debug_assert!(
            ghost_bits_clear(&words, num_bits),
            "bits beyond num_bits={num_bits} are set in the last word"
        );
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

    /// `FixedBitSet.get(index)`.
    ///
    /// # Panics
    ///
    /// If `index >= len()`. Java's `FixedBitSet` carries the same bound as an
    /// `assert`, which is off in production; this one is **not** a
    /// `debug_assert!`, and the difference matters. `words[index >> 6]` alone
    /// only catches an index 64 or more past the end: one merely past
    /// `num_bits` still lands inside the final word and reads a *ghost bit* --
    /// a silently wrong live/dead answer no caller can detect, which is the
    /// half of this defect class that costs a wrong answer rather than a
    /// crash (see `docs/arithmetic-gate.md`). A panic is containable
    /// (`lucene_ffi`'s `guard` catches it and reports `FfiStatus::Panic`); a
    /// ghost bit is not. Callers bound the index against **this bitset's own
    /// `len()`** -- see `docs/mechanical-gates.md`'s `fixed-bitset-bound`
    /// rule, which is what checks that they do.
    #[inline]
    pub fn get(&self, index: usize) -> bool {
        if index >= self.num_bits {
            out_of_range(index, self.num_bits);
        }
        let word = self.words[index >> 6];
        (word >> (index & 63)) & 1 != 0
    }

    /// Is the bit for `doc` set, where `doc` is a **doc id or ordinal that did
    /// not come from this bitset**?
    ///
    /// This is the sanctioned way to ask a live-docs / accept-ords bitset
    /// about an id produced somewhere else -- a postings walk, a BKD leaf, a
    /// vector store's ordinal range -- and it exists because writing the bound
    /// by hand at each of those call sites is the defect
    /// `docs/mechanical-gates.md`'s `fixed-bitset-bound` rule was built to
    /// catch, found by hand three times (c28 twice, c30 once) and mechanically
    /// thirty more.
    ///
    /// Two things it gets right that `get(doc as usize)` does not:
    ///
    /// - **A negative `doc` is not live.** `as usize` sign-extends, so a
    ///   negative id off a corrupt `.doc`/`.kdd` becomes `usize::MAX`.
    /// - **A `doc` past this bitset is not live**, rather than a ghost bit or
    ///   a panic. Java's `Bits.get` throws for both; answering "not live" is
    ///   what every caller in this port wants and what `Bits` means when the
    ///   two sides disagree about `maxDoc`.
    ///
    /// Use [`FixedBitSet::get`] when the index provably came from this
    /// bitset's own `len()` -- it is the cheaper call and the one the gate
    /// asks you to prove.
    #[inline]
    pub fn get_doc(&self, doc: i32) -> bool {
        match usize::try_from(doc) {
            Ok(index) => index < self.num_bits && self.get(index),
            Err(_) => false,
        }
    }

    /// `FixedBitSet.set(index)`.
    ///
    /// # Panics
    ///
    /// If `index >= len()` -- see [`FixedBitSet::get`]. A write past
    /// `num_bits` is worse than a read: it leaves a set ghost bit behind, and
    /// `cardinality()` counts whole words, so every later count is wrong too.
    #[inline]
    pub fn set(&mut self, index: usize) {
        if index >= self.num_bits {
            out_of_range(index, self.num_bits);
        }
        self.words[index >> 6] |= 1u64 << (index & 63);
    }

    /// `FixedBitSet.clear(index)`.
    ///
    /// # Panics
    ///
    /// If `index >= len()` -- see [`FixedBitSet::get`].
    #[inline]
    pub fn clear(&mut self, index: usize) {
        if index >= self.num_bits {
            out_of_range(index, self.num_bits);
        }
        self.words[index >> 6] &= !(1u64 << (index & 63));
    }

    /// Port of `FixedBitSet.clear()` (the no-argument overload): unsets every
    /// bit, keeping the allocation. The HNSW searcher clears its `visited` set
    /// once per level per query, so reallocating instead would put an
    /// allocation on the hottest loop in vector search.
    #[inline]
    pub fn clear_all(&mut self) {
        self.words.fill(0);
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

    #[test]
    #[should_panic(expected = "index 0 is out of range for a bitset of 0 bits")]
    fn an_empty_bitset_has_no_index_at_all() {
        // `bits2words(0) == 0`, so `words` is empty and `words[0]` would panic
        // on its own -- but with the message of a slice index, not of a
        // bitset bound.
        FixedBitSet::new(0).get(0);
    }
}
