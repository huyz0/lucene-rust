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
    ///
    /// Four independent accumulators rather than one `map().sum()`: a single
    /// running sum serialises every `popcnt` behind the previous add, and the
    /// loop is then bound by that dependency chain rather than by the
    /// instruction's throughput. Java's `Long.bitCount` loop gets the same
    /// unrolling from C2.
    pub fn cardinality(&self) -> usize {
        let mut chunks = self.words.chunks_exact(4);
        let (mut a, mut b, mut c, mut d) = (0usize, 0usize, 0usize, 0usize);
        for w in &mut chunks {
            a = a.wrapping_add(w[0].count_ones() as usize);
            b = b.wrapping_add(w[1].count_ones() as usize);
            c = c.wrapping_add(w[2].count_ones() as usize);
            d = d.wrapping_add(w[3].count_ones() as usize);
        }
        let tail: usize = chunks
            .remainder()
            .iter()
            .map(|w| w.count_ones() as usize)
            .fold(0, usize::wrapping_add);
        a.wrapping_add(b)
            .wrapping_add(c)
            .wrapping_add(d)
            .wrapping_add(tail)
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

    /// Port of `FixedBitSet.intersectionCount(a, b)`: the number of bits set
    /// in both, without materializing the intersection.
    pub fn intersection_count(a: &FixedBitSet, b: &FixedBitSet) -> usize {
        let mut acc = [0usize; 4];
        let n = a.words.len().min(b.words.len());
        let (aw, bw) = (&a.words[..n], &b.words[..n]);
        let mut ac = aw.chunks_exact(4);
        let mut bc = bw.chunks_exact(4);
        for (x, y) in (&mut ac).zip(&mut bc) {
            for k in 0..4 {
                acc[k] = acc[k].wrapping_add((x[k] & y[k]).count_ones() as usize);
            }
        }
        let tail = ac
            .remainder()
            .iter()
            .zip(bc.remainder())
            .map(|(x, y)| (x & y).count_ones() as usize)
            .fold(0, usize::wrapping_add);
        acc.iter().fold(tail, |s, v| s.wrapping_add(*v))
    }

    pub fn words(&self) -> &[u64] {
        &self.words
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
    /// bit is at or past `upper_bound`.
    pub fn next_set_bit_in_range(&self, start: usize, upper_bound: usize) -> Option<usize> {
        self.next_set_bit(start).filter(|&b| b < upper_bound)
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
        let b = next_clear_bit_in_words(&self.words, start);
        (b < upper_bound.min(self.num_bits)).then_some(b)
    }

    /// `FixedBitSet.cardinality(from, to)`: set bits in `from..to`.
    pub fn cardinality_range(&self, from: usize, to: usize) -> usize {
        assert!(
            from <= to && to <= self.num_bits,
            "range {from}..{to} of {}",
            self.num_bits
        );
        let mut count = 0usize;
        let mut i = from;
        while i < to {
            let w = i >> 6;
            let lo = i & 63;
            let span = (to - i).min(64 - lo);
            let mask = if span == 64 {
                u64::MAX
            } else {
                ((1u64 << span) - 1) << lo
            };
            count += (self.words[w] & mask).count_ones() as usize;
            i += span;
        }
        count
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
        let mut tot: usize = (0..n)
            .map(|i| (a.words[i] | b.words[i]).count_ones() as usize)
            .sum();
        tot += a.words[n..]
            .iter()
            .map(|w| w.count_ones() as usize)
            .sum::<usize>();
        tot += b.words[n..]
            .iter()
            .map(|w| w.count_ones() as usize)
            .sum::<usize>();
        tot
    }

    /// `FixedBitSet.andNotCount(a, b)`.
    pub fn and_not_count(a: &FixedBitSet, b: &FixedBitSet) -> usize {
        let n = a.words.len().min(b.words.len());
        let mut tot: usize = (0..n)
            .map(|i| (a.words[i] & !b.words[i]).count_ones() as usize)
            .sum();
        tot += a.words[n..]
            .iter()
            .map(|w| w.count_ones() as usize)
            .sum::<usize>();
        tot
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

    /// `FixedBitSet.orRange(source, sourceFrom, dest, destFrom, length)`.
    pub fn or_range(
        source: &FixedBitSet,
        source_from: usize,
        dest: &mut FixedBitSet,
        dest_from: usize,
        length: usize,
    ) {
        assert!(source_from + length <= source.len() && dest_from + length <= dest.len());
        for i in 0..length {
            if source.get(source_from + i) {
                dest.set(dest_from + i);
            }
        }
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
        assert!(source_from + length <= source.len() && dest_from + length <= dest.len());
        for i in 0..length {
            if !source.get(source_from + i) {
                dest.clear(dest_from + i);
            }
        }
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
    /// `from..to`, plus `base`, into `array`; returns how many.
    pub fn into_array(&self, from: usize, to: usize, base: i32, array: &mut [i32]) -> usize {
        assert!(from <= to && to <= self.num_bits);
        let mut n = 0;
        let mut i = self.next_set_bit(from);
        while let Some(b) = i.filter(|&b| b < to) {
            array[n] = base + b as i32;
            n += 1;
            i = self.next_set_bit(b + 1);
        }
        n
    }

    /// `FixedBitSet.forEach(from, to, base, consumer)`.
    pub fn for_each_in_range(&self, from: usize, to: usize, base: i32, mut f: impl FnMut(i32)) {
        assert!(from <= to && to <= self.num_bits);
        let mut i = self.next_set_bit(from);
        while let Some(b) = i.filter(|&b| b < to) {
            f(base + b as i32);
            i = self.next_set_bit(b + 1);
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
        let mut out = FixedBitSet::new(len);
        for i in 0..out.len() {
            if get(i) {
                out.set(i);
            }
        }
        out
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
mod ext_tests {
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
}

#[cfg(test)]
mod tests {
    #![allow(clippy::needless_range_loop, clippy::identity_op)]

    use super::*;

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
}
