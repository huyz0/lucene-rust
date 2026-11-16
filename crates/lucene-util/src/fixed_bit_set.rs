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

/// `FROM_BIT[k]` has bits `k..64` set: the mask [`FixedBitSet::next_set_bit`]
/// applies to the word holding its start.
static FROM_BIT: [u64; 64] = {
    let mut t = [0u64; 64];
    let mut k = 0;
    while k < 64 {
        t[k] = u64::MAX << k;
        k += 1;
    }
    t
};

/// The first set bit at or after `from` in `words`, or `None` -- the
/// word-array half of [`FixedBitSet::next_set_bit`], for callers that hold
/// raw words (a postings block kept as its bit set). The same tuned shape:
/// one loop, the first word masked through [`FROM_BIT`].
#[inline(always)]
pub fn next_set_bit_in_words(words: &[u64], from: usize) -> Option<usize> {
    let mut i = from >> 6;
    let mut word = *words.get(i)? & FROM_BIT[from & 63];
    loop {
        if word != 0 {
            return Some((i << 6) + word.trailing_zeros() as usize);
        }
        i = i.wrapping_add(1);
        word = *words.get(i)?;
    }
}

/// The index of the first clear bit at or after `from` in `words`, or
/// `words.len() * 64` when every bit from `from` on is set -- the
/// complement's `nextSetBit`, over a bare word slice.
pub fn next_clear_bit_in_words(words: &[u64], from: usize) -> usize {
    let end = words.len().saturating_mul(64);
    if from >= end {
        return end;
    }
    let mut i = from >> 6;
    // Set every bit below `from` in its word, so they do not read as clear.
    let mut word = words[i] | !FROM_BIT[from & 63];
    loop {
        if word != u64::MAX {
            return (i << 6) + (!word).trailing_zeros() as usize;
        }
        i += 1;
        match words.get(i) {
            Some(&w) => word = w,
            None => return end,
        }
    }
}

use crate::popcount::{self, Op};

/// Total set bits of `words`.
#[inline]
fn popcount_words(words: &[u64]) -> usize {
    popcount::words(words)
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
    ///
    /// # Panics
    ///
    /// If `words` is shorter than `bits2words(num_bits)`: the single-bit
    /// accessors index `words` without a second bounds check, which is sound
    /// only because every bit below `num_bits` has its word.
    pub fn from_words(words: Vec<u64>, num_bits: usize) -> Self {
        assert!(
            words.len() >= bits2words(num_bits),
            "{} words cannot hold {num_bits} bits",
            words.len()
        );
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
        // SAFETY: `index < num_bits`, and every constructor gives `words` at
        // least `bits2words(num_bits)` entries, so `index >> 6` is in bounds.
        let word = unsafe { *self.words.get_unchecked(index >> 6) };
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
    /// catch.
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
            Ok(index) => index < self.len() && self.get(index),
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
        // SAFETY: as in `get`.
        unsafe { *self.words.get_unchecked_mut(index >> 6) |= 1u64 << (index & 63) };
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
        // SAFETY: as in `get`.
        unsafe { *self.words.get_unchecked_mut(index >> 6) &= !(1u64 << (index & 63)) };
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
    ///
    /// Four independent accumulators rather than one `map().sum()`: a single
    /// running sum serialises every `popcnt` behind the previous add, and the
    /// loop is then bound by that dependency chain rather than by the
    /// instruction's throughput. Java's `Long.bitCount` loop gets the same
    /// unrolling from C2.
    pub fn cardinality(&self) -> usize {
        popcount_words(&self.words)
    }

    /// Port of `FixedBitSet.nextSetBit(index)`: the index of the first set bit
    /// at or after `index`, or `None` where Java returns `NO_MORE_DOCS`.
    ///
    /// Unlike Java, an `index` at or past `len()` is not an assertion failure:
    /// it simply has no set bit after it. That is what every iteration loop
    /// wants at its last step (`next_set_bit(i + 1)` after the final bit) and
    /// saves each caller the bound Java makes them write.
    #[inline]
    pub fn next_set_bit(&self, index: usize) -> Option<usize> {
        if index >= self.num_bits {
            return None;
        }
        let mut i = index >> 6;
        // Bits below `index` in its own word are masked off; bits above
        // `num_bits` are ghost bits, which the type invariant keeps clear.
        //
        // SAFETY: `index < num_bits` was just checked, and `words.len() ==
        // bits2words(num_bits)` is this type's invariant, so `index >> 6` is a
        // valid word index. This is the bounds check `FixedBitSet.nextSetBit`
        // never pays either (its `assert` is off in production); the one
        // comparison above is what makes it sound here.
        //
        // One loop for the first word and every later one -- one branch for the
        // predictor to learn rather than two -- with the bits below `index`
        // masked off. The mask comes from a table: written as `MAX << k`,
        // LLVM lowers `word & mask` to a shift down and back up, two dependent
        // instructions after the load instead of one `and` (the table load runs
        // alongside the word's). Measured on a 10%-dense set.
        let mut word = unsafe { *self.words.get_unchecked(i) } & FROM_BIT[index & 63];
        let words = self.words.as_slice();
        loop {
            if word != 0 {
                // `+` rather than `|`: a caller's `+ 1` then folds into one `lea`.
                return Some((i << 6) + word.trailing_zeros() as usize);
            }
            i = i.wrapping_add(1);
            word = *words.get(i)?;
        }
    }

    /// Calls `f` with every set bit, ascending -- the loop Java writes as
    /// `for (i = nextSetBit(0); i != NO_MORE_DOCS; i = nextSetBit(i + 1))`,
    /// done word by word so each word is loaded once rather than once per bit.
    #[inline]
    pub fn for_each_set_bit(&self, mut f: impl FnMut(usize)) {
        for (w, &word) in self.words.iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                f((w << 6) | bits.trailing_zeros() as usize);
                bits &= bits.wrapping_sub(1);
            }
        }
    }

    /// Port of `FixedBitSet.or(FixedBitSet other)`: sets every bit that is set
    /// in `other`. `other` may be shorter than `self`, as in Java; a longer
    /// `other` panics, where Java's `assert` would.
    pub fn or(&mut self, other: &FixedBitSet) {
        assert!(
            other.num_bits <= self.num_bits,
            "or: other has {} bits, this bitset only {}",
            other.num_bits,
            self.num_bits
        );
        // `other.words.len() <= self.words.len()` by the assert, so every one
        // of `other`'s words is or-ed in.
        crate::simd::or_words(&mut self.words, &other.words);
    }

    /// Port of `FixedBitSet.and(FixedBitSet other)`: clears every bit not set
    /// in `other`; words past `other`'s end are cleared, as in Java.
    pub fn and(&mut self, other: &FixedBitSet) {
        let n = self.words.len().min(other.words.len());
        let (head, tail) = self.words.split_at_mut(n);
        for (a, b) in head.iter_mut().zip(&other.words) {
            *a &= *b;
        }
        tail.fill(0);
    }

    /// Port of `FixedBitSet.andNot(FixedBitSet other)`: clears every bit set
    /// in `other`.
    pub fn and_not(&mut self, other: &FixedBitSet) {
        for (a, b) in self.words.iter_mut().zip(&other.words) {
            *a &= !*b;
        }
    }

    /// `FixedBitSet.or(DocIdSetIterator)` for a `DocBaseBitSetIterator`:
    /// sets bit `word_offset * 64 + i` for every bit `i` set in `words`, a
    /// word at a time. Bits that fall at or past `len()` are dropped, and the
    /// answer is `false` when there were any.
    pub fn or_words_at(&mut self, word_offset: usize, words: &[u64]) -> bool {
        self.merge_words_at(word_offset, words, |a, b| *a |= b)
    }

    /// `FixedBitSet.andNot(DocIdSetIterator)` for a `DocBaseBitSetIterator`:
    /// clears bit `word_offset * 64 + i` for every bit `i` set in `words`, as
    /// [`Self::or_words_at`] sets them.
    pub fn and_not_words_at(&mut self, word_offset: usize, words: &[u64]) -> bool {
        self.merge_words_at(word_offset, words, |a, b| *a &= !b)
    }

    #[inline]
    fn merge_words_at(
        &mut self,
        word_offset: usize,
        words: &[u64],
        op: impl Fn(&mut u64, u64),
    ) -> bool {
        let own = bits2words(self.num_bits);
        let fit = own.saturating_sub(word_offset).min(words.len());
        let mut in_range = words[fit..].iter().all(|&w| w == 0);
        if fit == 0 {
            return in_range;
        }
        let dst = &mut self.words[word_offset..word_offset + fit];
        let (last_dst, head_dst) = dst.split_last_mut().expect("fit > 0");
        for (a, &b) in head_dst.iter_mut().zip(&words[..fit - 1]) {
            op(a, b);
        }
        let mut last = words[fit - 1];
        if word_offset + fit == own && self.num_bits & 63 != 0 {
            let keep = (1u64 << (self.num_bits & 63)) - 1;
            in_range &= last & !keep == 0;
            last &= keep;
        }
        op(last_dst, last);
        in_range
    }

    /// Port of `FixedBitSet.intersectionCount(a, b)`: the number of bits set
    /// in both, without materializing the intersection.
    pub fn intersection_count(a: &FixedBitSet, b: &FixedBitSet) -> usize {
        popcount::zip(&a.words, &b.words, Op::And)
    }

    pub fn words(&self) -> &[u64] {
        &self.words
    }

    /// The backing words, for a caller that fills them in bulk and hands
    /// them back through [`Self::from_words`] (the inverse; the length is
    /// `bits2words(len())`).
    pub fn into_words(self) -> Vec<u64> {
        self.words
    }
}

/// The rest of `FixedBitSet`'s API: ranges, flips, the counting helpers, the
/// `BitSet` operations (`prevSetBit`, `getAndSet`, `approximateCardinality`)
/// and capacity management. Word-level algorithms are Java's; Java's shifts
/// are mod 64, which the masks below spell out where it matters.
impl FixedBitSet {
    #[inline]
    fn check(&self, index: usize) {
        if index >= self.num_bits {
            out_of_range(index, self.num_bits);
        }
    }

    /// Java's `-1L << startIndex` / `-1L >>> -endIndex`: the masks of the
    /// first and last word of `start..end` (`end > start`).
    #[inline]
    fn range_masks(start: usize, end: usize) -> (u64, u64) {
        let startmask = u64::MAX << (start & 63);
        // `-1L >>> -endIndex`: the low `end % 64` bits, or all 64 when `end`
        // is a multiple of 64.
        let endmask = u64::MAX >> ((64 - (end & 63)) & 63);
        (startmask, endmask)
    }

    /// `FixedBitSet.getAndSet(index)`.
    #[inline]
    pub fn get_and_set(&mut self, index: usize) -> bool {
        self.check(index);
        let mask = 1u64 << (index & 63);
        let word = &mut self.words[index >> 6];
        let was = *word & mask != 0;
        *word |= mask;
        was
    }

    /// `FixedBitSet.getAndClear(index)`.
    #[inline]
    pub fn get_and_clear(&mut self, index: usize) -> bool {
        self.check(index);
        let mask = 1u64 << (index & 63);
        let word = &mut self.words[index >> 6];
        let was = *word & mask != 0;
        *word &= !mask;
        was
    }

    /// `FixedBitSet.flip(index)`.
    #[inline]
    pub fn flip(&mut self, index: usize) {
        self.check(index);
        self.words[index >> 6] ^= 1u64 << (index & 63);
    }

    /// `FixedBitSet.flip(startIndex, endIndex)`.
    pub fn flip_range(&mut self, start: usize, end: usize) {
        if end <= start {
            return;
        }
        assert!(
            end <= self.num_bits,
            "endIndex={end}, numBits={}",
            self.num_bits
        );
        let (sw, ew) = (start >> 6, (end - 1) >> 6);
        let (startmask, endmask) = Self::range_masks(start, end);
        if sw == ew {
            self.words[sw] ^= startmask & endmask;
            return;
        }
        self.words[sw] ^= startmask;
        for w in &mut self.words[sw + 1..ew] {
            *w = !*w;
        }
        self.words[ew] ^= endmask;
    }

    /// `FixedBitSet.set(startIndex, endIndex)`.
    pub fn set_range(&mut self, start: usize, end: usize) {
        if end <= start {
            return;
        }
        assert!(
            end <= self.num_bits,
            "endIndex={end}, numBits={}",
            self.num_bits
        );
        let (sw, ew) = (start >> 6, (end - 1) >> 6);
        let (startmask, endmask) = Self::range_masks(start, end);
        if sw == ew {
            self.words[sw] |= startmask & endmask;
            return;
        }
        self.words[sw] |= startmask;
        self.words[sw + 1..ew].fill(u64::MAX);
        self.words[ew] |= endmask;
    }

    /// `FixedBitSet.clear(startIndex, endIndex)`.
    pub fn clear_range(&mut self, start: usize, end: usize) {
        if end <= start {
            return;
        }
        assert!(
            end <= self.num_bits,
            "endIndex={end}, numBits={}",
            self.num_bits
        );
        let (sw, ew) = (start >> 6, (end - 1) >> 6);
        let (startmask, endmask) = Self::range_masks(start, end);
        let (startmask, endmask) = (!startmask, !endmask);
        if sw == ew {
            self.words[sw] &= startmask | endmask;
            return;
        }
        self.words[sw] &= startmask;
        self.words[sw + 1..ew].fill(0);
        self.words[ew] &= endmask;
    }

    /// `FixedBitSet.prevSetBit(index)`: the last set bit at or before
    /// `index`, `None` for Java's `-1`.
    pub fn prev_set_bit(&self, index: usize) -> Option<usize> {
        self.check(index);
        let mut i = index >> 6;
        let sub = index & 63;
        let word = self.words[i] << (63 - sub);
        if word != 0 {
            return Some((i << 6) + sub - word.leading_zeros() as usize);
        }
        while i > 0 {
            i -= 1;
            let word = self.words[i];
            if word != 0 {
                return Some((i << 6) + 63 - word.leading_zeros() as usize);
            }
        }
        None
    }

    /// `FixedBitSet.nextSetBit(start, upperBound)`: `None` when the first set
    /// bit is at or past `upper_bound`. Like Java, the scan stops at the word
    /// holding `upper_bound - 1` instead of running to the end of the set.
    #[inline]
    pub fn next_set_bit_in_range(&self, start: usize, upper_bound: usize) -> Option<usize> {
        let ub = upper_bound.min(self.num_bits);
        if start >= ub {
            return None;
        }
        // `start < ub <= num_bits`, so `bits2words(ub) <= words.len()`.
        next_set_bit_in_words(&self.words[..bits2words(ub)], start).filter(|&b| b < ub)
    }

    /// `FixedBitSet.nextClearBit(index)`: `None` for Java's `NO_MORE_DOCS`.
    pub fn next_clear_bit(&self, index: usize) -> Option<usize> {
        self.next_clear_bit_in_range(index, self.num_bits)
    }

    /// `FixedBitSet.nextClearBit(start, upperBound)`.
    pub fn next_clear_bit_in_range(&self, start: usize, upper_bound: usize) -> Option<usize> {
        if start >= upper_bound.min(self.num_bits) {
            return None;
        }
        let ub = upper_bound.min(self.num_bits);
        // Scan only the words that hold `start..ub`, as Java does.
        let b = next_clear_bit_in_words(&self.words[..bits2words(ub)], start);
        (b < ub).then_some(b)
    }

    /// `FixedBitSet.cardinality(from, to)`: set bits in `from..to`.
    pub fn cardinality_range(&self, from: usize, to: usize) -> usize {
        assert!(
            from <= to && to <= self.num_bits,
            "range {from}..{to} of {}",
            self.num_bits
        );
        if from == to {
            return 0;
        }
        // Java 10's shape: a masked head word, the plain popcount kernel over
        // the whole words between, a masked tail word.
        let (fw, tw) = (from >> 6, (to - 1) >> 6);
        let (startmask, endmask) = Self::range_masks(from, to);
        if fw == tw {
            return (self.words[fw] & startmask & endmask).count_ones() as usize;
        }
        (self.words[fw] & startmask).count_ones() as usize
            + popcount_words(&self.words[fw + 1..tw])
            + (self.words[tw] & endmask).count_ones() as usize
    }

    /// `FixedBitSet.approximateCardinality()`: exact up to 1024 words, else
    /// the popcount of the first 16 words of every 1024, scaled.
    pub fn approximate_cardinality(&self) -> usize {
        const RANGE: usize = 16;
        const INTERVAL: usize = 1024;
        let num_words = self.words.len();
        if num_words <= INTERVAL {
            return self.cardinality();
        }
        let mut pop: i64 = 0;
        let mut max_word = 0usize;
        while max_word + INTERVAL < num_words {
            for i in 0..RANGE {
                pop += self.words[max_word + i].count_ones() as i64;
            }
            max_word += INTERVAL;
        }
        // Java: popCount *= (interval / rangeLength) * numWords / maxWord, in
        // int arithmetic on the right-hand side.
        pop *= ((INTERVAL / RANGE) as i64 * num_words as i64) / max_word as i64;
        pop as i32 as usize
    }

    /// `FixedBitSet.unionCount(a, b)`.
    pub fn union_count(a: &FixedBitSet, b: &FixedBitSet) -> usize {
        let n = a.words.len().min(b.words.len());
        popcount::zip(&a.words, &b.words, Op::Or)
            + popcount_words(&a.words[n..])
            + popcount_words(&b.words[n..])
    }

    /// `FixedBitSet.andNotCount(a, b)`.
    pub fn and_not_count(a: &FixedBitSet, b: &FixedBitSet) -> usize {
        let n = a.words.len().min(b.words.len());
        popcount::zip(&a.words, &b.words, Op::AndNot) + popcount_words(&a.words[n..])
    }

    /// `FixedBitSet.xor(FixedBitSet)`: `other` no longer than `self`.
    pub fn xor(&mut self, other: &FixedBitSet) {
        assert!(
            other.words.len() <= self.words.len(),
            "xor with a longer bitset"
        );
        for (a, b) in self.words.iter_mut().zip(&other.words) {
            *a ^= *b;
        }
    }

    /// `FixedBitSet.intersects(other)`.
    pub fn intersects(&self, other: &FixedBitSet) -> bool {
        self.words.iter().zip(&other.words).any(|(a, b)| a & b != 0)
    }

    /// `FixedBitSet.scanIsEmpty()`.
    pub fn scan_is_empty(&self) -> bool {
        self.words.iter().all(|&w| w == 0)
    }

    /// `FixedBitSet.orMask(startBit, mask, maskLen)`: ORs `mask`'s low
    /// `mask_len` bits in at `start_bit`.
    pub fn or_mask(&mut self, start_bit: usize, mask: u64, mask_len: usize) {
        assert!(
            start_bit + mask_len <= self.num_bits,
            "startBit={start_bit}, maskLen={mask_len}"
        );
        let w = start_bit >> 6;
        let off = start_bit & 63;
        self.words[w] |= mask << off;
        if off + mask_len > 64 {
            self.words[w + 1] |= mask >> (64 - off);
        }
    }

    /// Java's package-private `getBits(from, numBits)`: bits
    /// `from..from + n` as the low `n` bits of a word (`1 <= n <= 64`, the
    /// range inside the set).
    #[inline]
    fn bits_at(&self, from: usize, n: usize) -> u64 {
        debug_assert!((1..=64).contains(&n) && from + n <= self.num_bits);
        let (w, sh) = (from >> 6, from & 63);
        let mut v = self.words[w] >> sh;
        if sh + n > 64 {
            v |= self.words[w + 1] << (64 - sh);
        }
        if n < 64 {
            v &= (1u64 << n) - 1;
        }
        v
    }

    /// The word-at-a-time walk behind [`Self::or_range`] and
    /// [`Self::and_range`] (Java 10's): align `dest_from` to a word, combine
    /// whole words (the source shifted into place), then the tail. `head`
    /// combines `bits` (the low `n` bits) into a dest word at bit `sh`;
    /// `word` combines a whole aligned source word.
    #[inline(always)]
    fn range_op(
        source: &FixedBitSet,
        mut source_from: usize,
        dest: &mut FixedBitSet,
        mut dest_from: usize,
        mut length: usize,
        head: impl Fn(&mut u64, u64, usize, usize),
        word: impl Fn(&mut u64, u64),
    ) {
        assert!(
            source_from
                .checked_add(length)
                .is_some_and(|e| e <= source.len())
                && dest_from.checked_add(length).is_some_and(|e| e <= dest.len()),
            "range out of bounds: source {source_from}+{length} of {}, dest {dest_from}+{length} of {}",
            source.len(),
            dest.len()
        );
        if length == 0 {
            return;
        }
        if dest_from & 63 != 0 {
            let n = (64 - (dest_from & 63)).min(length);
            let bits = source.bits_at(source_from, n);
            head(&mut dest.words[dest_from >> 6], bits, dest_from & 63, n);
            source_from += n;
            dest_from += n;
            length -= n;
            if length == 0 {
                return;
            }
        }
        let full = length >> 6;
        let (sw, dw) = (source_from >> 6, dest_from >> 6);
        let d = &mut dest.words[dw..dw + full];
        let shift = source_from & 63;
        if shift == 0 {
            for (x, &y) in d.iter_mut().zip(&source.words[sw..sw + full]) {
                word(x, y);
            }
        } else if full > 0 {
            // The last full word reads source bits up to
            // `source_from + 64 * full - 1 < source.len()`, which lie in word
            // `sw + full` because `shift > 0`: that word exists.
            for (x, y) in d.iter_mut().zip(source.words[sw..=sw + full].windows(2)) {
                word(x, (y[0] >> shift) | (y[1] << (64 - shift)));
            }
        }
        source_from += full << 6;
        dest_from += full << 6;
        length -= full << 6;
        if length > 0 {
            let bits = source.bits_at(source_from, length);
            head(&mut dest.words[dest_from >> 6], bits, 0, length);
        }
    }

    /// `FixedBitSet.orRange(source, sourceFrom, dest, destFrom, length)`.
    pub fn or_range(
        source: &FixedBitSet,
        source_from: usize,
        dest: &mut FixedBitSet,
        dest_from: usize,
        length: usize,
    ) {
        Self::range_op(
            source,
            source_from,
            dest,
            dest_from,
            length,
            |d, bits, sh, _| *d |= bits << sh,
            |d, w| *d |= w,
        );
    }

    /// `FixedBitSet.andRange(source, sourceFrom, dest, destFrom, length)`:
    /// clears the bits of `dest[dest_from..+length]` whose source bit is clear.
    pub fn and_range(
        source: &FixedBitSet,
        source_from: usize,
        dest: &mut FixedBitSet,
        dest_from: usize,
        length: usize,
    ) {
        Self::range_op(
            source,
            source_from,
            dest,
            dest_from,
            length,
            |d, bits, sh, n| {
                // Keep every dest bit outside `sh..sh + n`.
                let span = if n == 64 { u64::MAX } else { (1u64 << n) - 1 };
                *d &= (bits << sh) | !(span << sh);
            },
            |d, w| *d &= w,
        );
    }

    /// `FixedBitSet.ensureCapacity(bits, desiredBit)`: `bits` itself when it
    /// can hold `desired_bit`, else a larger copy (Java's growth policy:
    /// `ArrayUtil.grow` to at least `bits2words(desiredBit) + 1` words).
    pub fn ensure_capacity(bits: FixedBitSet, desired_bit: usize) -> FixedBitSet {
        Self::ensure_capacity_internal(bits, desired_bit, true)
    }

    /// `FixedBitSet.ensureCapacityAndClear(bits, desiredBit)`.
    pub fn ensure_capacity_and_clear(bits: FixedBitSet, desired_bit: usize) -> FixedBitSet {
        Self::ensure_capacity_internal(bits, desired_bit, false)
    }

    fn ensure_capacity_internal(
        mut bits: FixedBitSet,
        desired_bit: usize,
        preserve: bool,
    ) -> FixedBitSet {
        if desired_bit < bits.num_bits {
            if !preserve {
                bits.clear_all();
            }
            return bits;
        }
        let num_words = bits2words(desired_bit);
        let mut words = if preserve { bits.words } else { Vec::new() };
        if num_words >= words.len() {
            // ArrayUtil.oversize(numWords + 1, 8): grow by 1/8, at least 3.
            let min = num_words + 1;
            let extra = (min >> 3).max(3);
            words.resize(min + extra, 0);
        }
        let num_bits = words.len() << 6;
        FixedBitSet { words, num_bits }
    }

    /// `FixedBitSet.hashCode()`: Java's rotate-and-xor over the words.
    pub fn java_hash_code(&self) -> i32 {
        let mut h: i64 = 0;
        for &w in self.words.iter().rev() {
            h ^= w as i64;
            h = h.rotate_left(1);
        }
        (((h >> 32) ^ h) as i32).wrapping_add(0x98761234u32 as i32)
    }

    /// `FixedBitSet.intoArray(from, to, base, array)`: the set bits of
    /// `from..to`, plus `base`, into `array`; returns how many. Stops when
    /// `array` is full, as Java's `word2Array` copies at most the room left:
    /// the first `array.len()` bits, in order.
    pub fn into_array(&self, from: usize, to: usize, base: i32, array: &mut [i32]) -> usize {
        assert!(from <= to && to <= self.num_bits);
        let mut n = 0;
        self.for_each_word_in_range(from, to, |w, mut bits| {
            while bits != 0 && n < array.len() {
                array[n] = base + ((w << 6) | bits.trailing_zeros() as usize) as i32;
                n += 1;
                bits &= bits - 1;
            }
            n < array.len()
        });
        n
    }

    /// `FixedBitSet.forEach(from, to, base, consumer)`.
    pub fn for_each_in_range(&self, from: usize, to: usize, base: i32, mut f: impl FnMut(i32)) {
        assert!(from <= to && to <= self.num_bits);
        self.for_each_word_in_range(from, to, |w, mut bits| {
            while bits != 0 {
                f(base + ((w << 6) | bits.trailing_zeros() as usize) as i32);
                bits &= bits - 1;
            }
            true
        });
    }

    /// Calls `f(word_index, bits)` for each word overlapping `from..to`, with
    /// the bits outside the range masked off; stops when `f` returns false.
    #[inline]
    fn for_each_word_in_range(
        &self,
        from: usize,
        to: usize,
        mut f: impl FnMut(usize, u64) -> bool,
    ) {
        if from >= to {
            return;
        }
        let (first, last) = (from >> 6, (to - 1) >> 6);
        let tail = u64::MAX >> (63 - ((to - 1) & 63));
        for w in first..=last {
            let mut bits = self.words[w];
            if w == first {
                bits &= u64::MAX << (from & 63);
            }
            if w == last {
                bits &= tail;
            }
            if !f(w, bits) {
                return;
            }
        }
    }

    /// `FixedBitSet.applyMask(bitSet, offset)`: clears every bit of `bit_set`
    /// whose counterpart at `offset + i` here is clear. `Err` when `bit_set`
    /// has bits set past this bitset's end (Java's `IllegalArgumentException`).
    pub fn apply_mask(&self, bit_set: &mut FixedBitSet, offset: usize) -> Result<(), String> {
        let length = bit_set.num_bits.min(self.num_bits.saturating_sub(offset));
        if offset <= self.num_bits {
            Self::and_range(self, offset, bit_set, 0, length);
        }
        if length < bit_set.num_bits && bit_set.next_set_bit(length).is_some() {
            return Err("Some bits are set beyond the end of live docs".into());
        }
        Ok(())
    }

    /// `FixedBitSet.copyOf(Bits)` over any bit source of `len` bits.
    pub fn copy_of(len: usize, get: impl Fn(usize) -> bool) -> FixedBitSet {
        let mut words = vec![0u64; bits2words(len)];
        for (w, word) in words.iter_mut().enumerate() {
            let base = w << 6;
            let n = (len - base).min(64);
            let mut acc = 0u64;
            for b in 0..n {
                acc |= u64::from(get(base + b)) << b;
            }
            *word = acc;
        }
        FixedBitSet {
            words,
            num_bits: len,
        }
    }
}

impl PartialEq for FixedBitSet {
    /// `FixedBitSet.equals`: same length, same words.
    fn eq(&self, other: &Self) -> bool {
        self.num_bits == other.num_bits && self.words == other.words
    }
}

impl Eq for FixedBitSet {}

#[cfg(test)]
mod ext_tests;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod range_tests {
    #![allow(clippy::needless_range_loop)]

    use super::*;

    fn model(n: usize, seed: u64) -> (FixedBitSet, Vec<bool>) {
        let mut s = seed;
        let mut b = FixedBitSet::new(n);
        let mut m = vec![false; n];
        for (i, slot) in m.iter_mut().enumerate() {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            if (s >> 60) & 1 == 1 {
                b.set(i);
                *slot = true;
            }
        }
        (b, m)
    }

    #[test]
    fn word_wise_range_ops_match_bitwise_model() {
        let (src, sm) = model(700, 11);
        for &(sf, df, len) in &[
            (0, 0, 0),
            (0, 0, 700),
            (3, 17, 600),
            (64, 0, 128),
            (0, 64, 300),
            (5, 5, 64),
            (63, 1, 70),
            (1, 63, 1),
            (130, 200, 499),
            (17, 3, 683),
        ] {
            let (d0, dm0) = model(720, 23);
            let mut d = d0.clone();
            FixedBitSet::or_range(&src, sf, &mut d, df, len);
            let mut dm = dm0.clone();
            for i in 0..len {
                dm[df + i] |= sm[sf + i];
            }
            for i in 0..720 {
                assert_eq!(d.get(i), dm[i], "or {sf} {df} {len} bit {i}");
            }
            let mut d = d0.clone();
            FixedBitSet::and_range(&src, sf, &mut d, df, len);
            let mut dm = dm0;
            for i in 0..len {
                dm[df + i] &= sm[sf + i];
            }
            for i in 0..720 {
                assert_eq!(d.get(i), dm[i], "and {sf} {df} {len} bit {i}");
            }
            assert!(ghost_bits_clear(d.words(), d.len()));
        }
    }

    #[test]
    #[should_panic(expected = "range out of bounds")]
    fn or_range_rejects_overrun() {
        let s = FixedBitSet::new(10);
        let mut d = FixedBitSet::new(100);
        FixedBitSet::or_range(&s, 5, &mut d, 0, 6);
    }

    #[test]
    fn bounded_scans_and_counts_match_model() {
        let (b, m) = model(1000, 5);
        let (c, cm) = model(900, 6);
        for start in (0..1000).step_by(7) {
            for ub in [
                start,
                start + 1,
                start + 63,
                start + 64,
                start + 200,
                1000,
                5000,
            ] {
                let want = (start..ub.min(1000)).find(|&i| m[i]);
                assert_eq!(b.next_set_bit_in_range(start, ub), want, "{start} {ub}");
                let want = (start..ub.min(1000)).find(|&i| !m[i]);
                assert_eq!(b.next_clear_bit_in_range(start, ub), want, "{start} {ub}");
            }
            for to in [start, start + 1, start + 64, 1000] {
                if to <= 1000 {
                    let want = (start..to).filter(|&i| m[i]).count();
                    assert_eq!(b.cardinality_range(start, to), want);
                }
            }
        }
        let u = (0..1000).filter(|&i| m[i] || (i < 900 && cm[i])).count();
        assert_eq!(FixedBitSet::union_count(&b, &c), u);
        assert_eq!(FixedBitSet::union_count(&c, &b), u);
        let an = (0..1000).filter(|&i| m[i] && !(i < 900 && cm[i])).count();
        assert_eq!(FixedBitSet::and_not_count(&b, &c), an);
        let an = (0..900).filter(|&i| cm[i] && !m[i]).count();
        assert_eq!(FixedBitSet::and_not_count(&c, &b), an);
        let mut full = FixedBitSet::new(130);
        full.set_range(0, 130);
        assert_eq!(full.next_clear_bit_in_range(0, 130), None);
        assert_eq!(full.next_clear_bit(64), None);
        assert_eq!(FixedBitSet::new(130).next_set_bit_in_range(0, 130), None);
    }
}
