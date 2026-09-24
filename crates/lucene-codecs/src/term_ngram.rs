//! A term-level n-gram index: for one field's term dictionary, which terms
//! contain each two- and three-byte sequence -- the index behind Google Code
//! Search's regexp matching (Russ Cox, "Regular Expression Matching with a
//! Trigram Index") and Elasticsearch's `wildcard` field, built over the
//! *terms* rather than the documents.
//!
//! A regexp with no literal prefix (`.*zz.*`, `.*a0`) gives the term
//! dictionary walk nothing to skip by: every entry has to be run through the
//! automaton. What such a pattern does have is literal text every match must
//! contain ([`crate::regexp::RegexpPattern::required_grams`]); this index
//! turns that into the few terms that contain all of it, and only those are
//! run through the automaton. Lucene has no counterpart -- its
//! `IntersectTermsEnum` walks the whole dictionary for these patterns.
//!
//! Grams are over bytes with a boundary symbol at each end of a term, so
//! "ends in `z`" is the gram `z` + end and "starts with `ab`" is start + `ab`.
//! Postings are term ordinals in dictionary order, delta-varint encoded, so
//! candidates come out sorted, as every term enumeration must.
//!
//! Built on demand -- [`crate::blocktree::FieldTerms`] builds one only for a
//! field that keeps being asked such queries -- and costs, per term, its
//! bytes plus a byte or two for each gram it contains.

use std::collections::HashMap;

/// The start/end-of-term symbol: one past every byte.
pub const BOUNDARY: u32 = 256;

/// A gram's key: two or three 9-bit symbols (bytes or [`BOUNDARY`]) and the
/// length in the top bits.
// ARITH: every symbol is at most 256 < 2^9, so three shifted into 27 bits
// plus the length tag at bit 30 fit a `u32`.
#[allow(clippy::arithmetic_side_effects)]
#[inline]
pub fn gram_key(symbols: &[u32]) -> u32 {
    match *symbols {
        [a, b] => (1 << 30) | (a << 9) | b,
        [a, b, c] => (2 << 30) | (a << 18) | (b << 9) | c,
        _ => 0,
    }
}

/// The two- and three-symbol windows of `symbols`, as keys.
pub fn grams_of(symbols: &[u32], out: &mut Vec<u32>) {
    for w in symbols.windows(2) {
        out.push(gram_key(w));
    }
    for w in symbols.windows(3) {
        out.push(gram_key(w));
    }
}

/// `term` padded with [`BOUNDARY`] at both ends.
pub fn padded(term: &[u8], out: &mut Vec<u32>) {
    out.clear();
    out.push(BOUNDARY);
    out.extend(term.iter().map(|&b| u32::from(b)));
    out.push(BOUNDARY);
}

#[derive(Debug, Default)]
pub struct TermNgramIndex {
    /// Every term's bytes, back to back; term `i` is
    /// `bytes[offsets[i]..offsets[i + 1]]`.
    bytes: Vec<u8>,
    offsets: Vec<u32>,
    /// Gram key -> `(start, byte length, term count)` of its posting list in
    /// `postings`.
    grams: HashMap<u32, (u32, u32, u32)>,
    postings: Vec<u8>,
}

impl TermNgramIndex {
    /// Builds the index from `terms`, which must come in dictionary order.
    /// `None` when the dictionary is too large to address with `u32`
    /// offsets. Streams: each term is copied once into the index's own
    /// buffer, and each gram's posting list is written compressed as terms
    /// arrive, so the build's peak is about the finished index's size.
    pub fn build<'t>(terms: impl Iterator<Item = &'t [u8]>) -> Option<TermNgramIndex> {
        let mut b = TermNgramBuilder::default();
        for term in terms {
            b.push(term)?;
        }
        b.finish()
    }

    pub fn num_terms(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// Term `ord`'s bytes.
    // ARITH: `ord < num_terms()`, so `ord + 1 < offsets.len()`.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn term(&self, ord: u32) -> &[u8] {
        let (lo, hi) = (self.offsets[ord as usize], self.offsets[ord as usize + 1]);
        &self.bytes[lo as usize..hi as usize]
    }

    /// How many terms contain the gram, 0 for one that none does.
    pub fn count(&self, key: u32) -> u32 {
        self.grams.get(&key).map_or(0, |&(_, _, n)| n)
    }

    /// The ordinals of the terms containing every gram in `keys`, ascending
    /// -- the candidates, a superset of the matches. `None` past `limit`
    /// candidates, so the caller can decide the index was not selective
    /// enough after all, and for no grams at all.
    // ARITH: `j` only advances while `j < other.len()`, so it stays within
    // a slice length.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn candidates(&self, keys: &[u32], limit: usize) -> Option<Vec<u32>> {
        // No gram, no filter: every term is a candidate, which this index
        // cannot improve on.
        if keys.is_empty() {
            return None;
        }
        let mut lists: Vec<Vec<u32>> = Vec::with_capacity(keys.len());
        let mut order: Vec<u32> = keys.to_vec();
        order.sort_unstable_by_key(|&k| self.count(k));
        order.dedup();
        for (i, &k) in order.iter().enumerate() {
            let list = self.decode(k);
            if i == 0 && list.len() > limit {
                return None;
            }
            lists.push(list);
            if lists.last().is_some_and(Vec::is_empty) {
                return Some(Vec::new());
            }
        }
        let mut out = lists.first().cloned().unwrap_or_default();
        for other in &lists[1..] {
            let mut j = 0;
            out.retain(|&o| {
                while j < other.len() && other[j] < o {
                    j += 1;
                }
                j < other.len() && other[j] == o
            });
            if out.is_empty() {
                break;
            }
        }
        Some(out)
    }

    // ARITH: `start + len` is a range this index wrote into `postings`; the
    // running ordinal is a sum of deltas that reproduces values that were
    // themselves `u32`s.
    #[allow(clippy::arithmetic_side_effects)]
    fn decode(&self, key: u32) -> Vec<u32> {
        let Some(&(start, len, n)) = self.grams.get(&key) else {
            return Vec::new();
        };
        let mut out = Vec::with_capacity(n as usize);
        let mut buf = &self.postings[start as usize..(start + len) as usize];
        let mut prev = 0u32;
        while let Some((d, rest)) = read_vu32(buf) {
            prev += d;
            out.push(prev);
            buf = rest;
        }
        out
    }

    /// Bytes this index holds, for memory accounting.
    // ARITH: sums of in-memory buffer sizes.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn heap_bytes(&self) -> usize {
        self.bytes.capacity()
            + self.offsets.capacity() * 4
            + self.postings.capacity()
            + self.grams.capacity() * 16
    }
}

/// [`TermNgramIndex::build`], one term at a time.
#[derive(Debug, Default)]
pub struct TermNgramBuilder {
    bytes: Vec<u8>,
    offsets: Vec<u32>,
    /// Per gram: the last ordinal written, the term count, and the
    /// delta-varint postings so far.
    lists: HashMap<u32, (u32, u32, Vec<u8>)>,
    sym: Vec<u32>,
    keys: Vec<u32>,
}

impl TermNgramBuilder {
    /// Adds the next term in dictionary order; `None` once the dictionary
    /// outgrows `u32` offsets.
    // ARITH: ordinals ascend, so `ord - last` is non-negative; the count is
    // at most the number of terms, itself checked to fit a `u32`.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn push(&mut self, term: &[u8]) -> Option<()> {
        if self.offsets.is_empty() {
            self.offsets.push(0);
        }
        let ord = u32::try_from(self.offsets.len() - 1).ok()?;
        self.bytes.extend_from_slice(term);
        self.offsets.push(u32::try_from(self.bytes.len()).ok()?);
        padded(term, &mut self.sym);
        self.keys.clear();
        grams_of(&self.sym, &mut self.keys);
        self.keys.sort_unstable();
        self.keys.dedup();
        for &k in &self.keys {
            let (last, count, buf) = self.lists.entry(k).or_default();
            write_vu32(buf, ord - *last);
            *last = ord;
            *count += 1;
        }
        Some(())
    }

    /// The finished index; `None` when its postings outgrow `u32` offsets.
    pub fn finish(mut self) -> Option<TermNgramIndex> {
        if self.offsets.is_empty() {
            self.offsets.push(0);
        }
        let total: usize = self.lists.values().map(|(_, _, b)| b.len()).sum();
        let mut postings = Vec::with_capacity(total);
        let mut grams = HashMap::with_capacity(self.lists.len());
        for (k, (_, count, buf)) in self.lists {
            let start = u32::try_from(postings.len()).ok()?;
            postings.extend_from_slice(&buf);
            grams.insert(k, (start, u32::try_from(buf.len()).ok()?, count));
        }
        self.bytes.shrink_to_fit();
        Some(TermNgramIndex {
            bytes: self.bytes,
            offsets: self.offsets,
            grams,
            postings,
        })
    }
}

// ARITH: a `u32` needs at most five 7-bit groups; the shift is below 32.
#[allow(clippy::arithmetic_side_effects)]
fn write_vu32(out: &mut Vec<u8>, mut v: u32) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

// ARITH: at most five groups are read (`shift <= 28`); a longer run is
// rejected rather than shifted past 32 bits.
#[allow(clippy::arithmetic_side_effects)]
fn read_vu32(buf: &[u8]) -> Option<(u32, &[u8])> {
    let mut v = 0u32;
    for (i, &b) in buf.iter().enumerate().take(5) {
        v |= u32::from(b & 0x7F) << (7 * i);
        if b < 0x80 {
            return Some((v, &buf[i + 1..]));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidates_are_the_terms_containing_every_gram() {
        let terms: Vec<&[u8]> = vec![
            b"abc",
            b"abd",
            b"bcd",
            b"xabcx",
            b"zz",
            b"azz",
            &[0xFF, b'z'],
        ];
        let idx = TermNgramIndex::build(terms.iter().copied()).unwrap();
        assert_eq!(idx.num_terms(), 7);
        assert_eq!(idx.term(3), b"xabcx");
        let mut sym = Vec::new();
        let key = |s: &[u32]| gram_key(s);
        // "bc" anywhere.
        assert_eq!(idx.candidates(&[key(&[98, 99])], 100), Some(vec![0, 2, 3]));
        // "abc" anywhere and "c" at the end: only "abc".
        assert_eq!(
            idx.candidates(&[key(&[97, 98, 99]), key(&[99, BOUNDARY])], 100),
            Some(vec![0])
        );
        // Ends in "z": boundary-anchored bigram.
        assert_eq!(
            idx.candidates(&[key(&[122, BOUNDARY])], 100),
            Some(vec![4, 5, 6])
        );
        // A byte above 0x7F stays distinct from the boundary.
        assert_eq!(
            idx.candidates(&[key(&[BOUNDARY, 0xFF])], 100),
            Some(vec![6])
        );
        // No term: empty; too many: `None`.
        assert_eq!(idx.candidates(&[key(&[1, 2])], 100), Some(vec![]));
        assert_eq!(idx.candidates(&[key(&[98, 99])], 2), None);
        assert_eq!(idx.candidates(&[], 100), None);
        padded(b"ab", &mut sym);
        assert_eq!(sym, vec![BOUNDARY, 97, 98, BOUNDARY]);
        assert!(idx.heap_bytes() > 0);
    }

    #[test]
    fn varints_round_trip() {
        for v in [0u32, 1, 127, 128, 16_383, 16_384, u32::MAX] {
            let mut b = Vec::new();
            write_vu32(&mut b, v);
            assert_eq!(read_vu32(&b), Some((v, &[][..])));
        }
        assert_eq!(read_vu32(&[0x80, 0x80]), None);
    }
}
