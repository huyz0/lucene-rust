//! [`InvertedTerms`]: one document's terms of one field, stored flat -- every
//! term's bytes in one buffer and every occurrence's position, offsets and
//! payload in one array each -- so a document costs a handful of
//! allocations per field rather than two or three per term.

use super::explicit::InvertedTerm;
use crate::inverter::hash_bytes;

/// Where one term's data ends in each of [`InvertedTerms`]' arrays; it starts
/// where the previous term's ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TermEnds {
    bytes: u32,
    positions: u32,
    offsets: u32,
    payloads: u32,
    freq: i32,
}

impl TermEnds {
    const ZERO: TermEnds = TermEnds {
        bytes: 0,
        positions: 0,
        offsets: 0,
        payloads: 0,
        freq: 0,
    };
}

/// [`InvertedTerms::iter`].
#[derive(Debug, Clone)]
pub struct InvertedTermsIter<'a> {
    terms: &'a InvertedTerms,
    i: usize,
    prev: TermEnds,
}

impl<'a> Iterator for InvertedTermsIter<'a> {
    type Item = TermRef<'a>;

    #[inline]
    fn next(&mut self) -> Option<TermRef<'a>> {
        let e = *self.terms.ends.get(self.i)?;
        let t = self.terms.term_between(self.i, self.prev, e);
        self.prev = e;
        self.i = self.i.saturating_add(1);
        Some(t)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.terms.len().saturating_sub(self.i);
        (n, Some(n))
    }
}

impl ExactSizeIterator for InvertedTermsIter<'_> {}

/// One document's terms of one field, each distinct term once, in the order
/// they were pushed: see the module doc.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct InvertedTerms {
    ends: Vec<TermEnds>,
    bytes: Vec<u8>,
    positions: Vec<i32>,
    offsets: Vec<(i32, i32)>,
    /// Where each payload ends in `payload_bytes`.
    payload_ends: Vec<u32>,
    payload_bytes: Vec<u8>,
    /// Each term's [`hash_bytes`], taken once as it is pushed: the writer's
    /// duplicate check and dictionary lookup both reuse it.
    hashes: Vec<u64>,
}

/// One term of [`InvertedTerms`], read in place; the fields mean what
/// [`InvertedTerm`]'s do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TermRef<'a> {
    pub term: &'a [u8],
    /// [`hash_bytes`] of `term`.
    pub hash: u64,
    pub freq: i32,
    pub positions: &'a [i32],
    pub offsets: &'a [(i32, i32)],
    pub payloads: Payloads<'a>,
}

/// A term's payloads, one per occurrence or none at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Payloads<'a> {
    /// The end of the payload before the first, then each payload's end.
    start: u32,
    ends: &'a [u32],
    bytes: &'a [u8],
}

impl<'a> Payloads<'a> {
    pub fn len(&self) -> usize {
        self.ends.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ends.is_empty()
    }

    /// Occurrence `i`'s payload; empty for none.
    #[inline]
    pub fn get(&self, i: usize) -> &'a [u8] {
        let Some(&end) = self.ends.get(i) else {
            return &[];
        };
        let start = i
            .checked_sub(1)
            .and_then(|p| self.ends.get(p).copied())
            .unwrap_or(self.start);
        self.bytes
            .get(start as usize..end as usize)
            .unwrap_or_default()
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &'a [u8]> + '_ {
        (0..self.len()).map(|i| self.get(i))
    }
}

fn len_u32(len: usize) -> u32 {
    u32::try_from(len).unwrap_or(u32::MAX)
}

impl InvertedTerms {
    pub fn new() -> Self {
        Self::default()
    }

    /// Room for `terms` terms of `bytes` bytes and `occurrences` positions in
    /// all.
    pub fn with_capacity(terms: usize, bytes: usize, occurrences: usize) -> Self {
        InvertedTerms {
            ends: Vec::with_capacity(terms),
            hashes: Vec::with_capacity(terms),
            bytes: Vec::with_capacity(bytes),
            positions: Vec::with_capacity(occurrences),
            ..Self::default()
        }
    }

    /// Appends a term: its bytes, its frequency in the document, and its
    /// positions, offsets and payloads (each empty when the field does not
    /// index it; see [`InvertedTerm`]).
    pub fn push<P: AsRef<[u8]>>(
        &mut self,
        term: &[u8],
        freq: i32,
        positions: &[i32],
        offsets: &[(i32, i32)],
        payloads: &[P],
    ) {
        self.hashes.push(hash_bytes(term));
        self.bytes.extend_from_slice(term);
        self.positions.extend_from_slice(positions);
        self.offsets.extend_from_slice(offsets);
        for p in payloads {
            self.payload_bytes.extend_from_slice(p.as_ref());
            self.payload_ends.push(len_u32(self.payload_bytes.len()));
        }
        self.ends.push(TermEnds {
            bytes: len_u32(self.bytes.len()),
            positions: len_u32(self.positions.len()),
            offsets: len_u32(self.offsets.len()),
            payloads: len_u32(self.payload_ends.len()),
            freq,
        });
    }

    /// Appends a term built in place: its bytes now, then its occurrences
    /// one at a time through the returned [`TermBuilder`], which records the
    /// term when dropped.
    pub fn start(&mut self, term: &[u8]) -> TermBuilder<'_> {
        self.start_hashed(term, hash_bytes(term))
    }

    /// [`Self::start`] for a term whose [`hash_bytes`] the caller already
    /// has.
    pub fn start_hashed(&mut self, term: &[u8], hash: u64) -> TermBuilder<'_> {
        debug_assert_eq!(hash, hash_bytes(term));
        self.hashes.push(hash);
        self.bytes.extend_from_slice(term);
        TermBuilder {
            terms: self,
            freq: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.ends.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ends.is_empty()
    }

    /// Drops every term, keeping the allocations.
    pub fn clear(&mut self) {
        self.ends.clear();
        self.hashes.clear();
        self.bytes.clear();
        self.positions.clear();
        self.offsets.clear();
        self.payload_ends.clear();
        self.payload_bytes.clear();
    }

    /// Term `i`, in push order.
    ///
    /// # Panics
    /// When `i >= self.len()`.
    #[inline]
    pub fn get(&self, i: usize) -> TermRef<'_> {
        let s = i.checked_sub(1).map_or(TermEnds::ZERO, |p| self.ends[p]);
        self.term_between(i, s, self.ends[i])
    }

    /// Term `i`, whose data ends at `e` in each array and starts at `s`
    /// (the previous term's ends).
    #[inline]
    fn term_between(&self, i: usize, s: TermEnds, e: TermEnds) -> TermRef<'_> {
        let payload_start = (s.payloads as usize)
            .checked_sub(1)
            .map_or(0, |p| self.payload_ends[p]);
        TermRef {
            term: &self.bytes[s.bytes as usize..e.bytes as usize],
            hash: self.hashes[i],
            freq: e.freq,
            positions: &self.positions[s.positions as usize..e.positions as usize],
            offsets: &self.offsets[s.offsets as usize..e.offsets as usize],
            payloads: Payloads {
                start: payload_start,
                ends: &self.payload_ends[s.payloads as usize..e.payloads as usize],
                bytes: &self.payload_bytes,
            },
        }
    }

    /// The terms in push order; each step reuses the previous term's ends.
    #[inline]
    pub fn iter(&self) -> InvertedTermsIter<'_> {
        InvertedTermsIter {
            terms: self,
            i: 0,
            prev: TermEnds::ZERO,
        }
    }

    /// Each term's [`hash_bytes`], in push order.
    #[inline]
    pub fn hashes(&self) -> &[u64] {
        &self.hashes
    }

    /// The terms as owned [`InvertedTerm`]s.
    pub fn to_vec(&self) -> Vec<InvertedTerm> {
        self.iter()
            .map(|t| InvertedTerm {
                term: t.term.to_vec(),
                freq: t.freq,
                positions: t.positions.to_vec(),
                offsets: t.offsets.to_vec(),
                payloads: t.payloads.iter().map(<[u8]>::to_vec).collect(),
            })
            .collect()
    }

    /// Runs `edit` on the terms as owned [`InvertedTerm`]s and stores the
    /// result -- a convenience for building unusual documents, not a fast
    /// path.
    pub fn edit<R>(&mut self, edit: impl FnOnce(&mut Vec<InvertedTerm>) -> R) -> R {
        let mut terms = self.to_vec();
        let out = edit(&mut terms);
        *self = terms.into();
        out
    }

    /// Heap bytes held.
    pub(crate) fn ram_bytes(&self) -> usize {
        self.ends
            .capacity()
            .saturating_mul(std::mem::size_of::<TermEnds>())
            .saturating_add(self.bytes.capacity())
            .saturating_add(self.positions.capacity().saturating_mul(4))
            .saturating_add(self.offsets.capacity().saturating_mul(8))
            .saturating_add(self.payload_ends.capacity().saturating_mul(4))
            .saturating_add(self.payload_bytes.capacity())
            .saturating_add(self.hashes.capacity().saturating_mul(8))
    }
}

/// See [`InvertedTerms::start`].
pub struct TermBuilder<'a> {
    terms: &'a mut InvertedTerms,
    freq: i32,
}

impl TermBuilder<'_> {
    /// One occurrence at `position` (for a field that indexes positions).
    pub fn position(&mut self, position: i32) -> &mut Self {
        self.terms.positions.push(position);
        self
    }

    /// The current occurrence's offsets (for a field that indexes them).
    pub fn offsets(&mut self, start: i32, end: i32) -> &mut Self {
        self.terms.offsets.push((start, end));
        self
    }

    /// The current occurrence's payload; every occurrence of a term with
    /// payloads needs one, empty for none.
    pub fn payload(&mut self, payload: &[u8]) -> &mut Self {
        self.terms.payload_bytes.extend_from_slice(payload);
        let at = len_u32(self.terms.payload_bytes.len());
        self.terms.payload_ends.push(at);
        self
    }

    /// The term's frequency in the document.
    pub fn freq(&mut self, freq: i32) -> &mut Self {
        self.freq = freq;
        self
    }
}

impl Drop for TermBuilder<'_> {
    fn drop(&mut self) {
        let t = &mut *self.terms;
        t.ends.push(TermEnds {
            bytes: len_u32(t.bytes.len()),
            positions: len_u32(t.positions.len()),
            offsets: len_u32(t.offsets.len()),
            payloads: len_u32(t.payload_ends.len()),
            freq: self.freq,
        });
    }
}

impl From<Vec<InvertedTerm>> for InvertedTerms {
    fn from(terms: Vec<InvertedTerm>) -> Self {
        let mut out = InvertedTerms::with_capacity(
            terms.len(),
            terms.iter().map(|t| t.term.len()).sum(),
            terms.iter().map(|t| t.positions.len()).sum(),
        );
        for t in &terms {
            out.push(&t.term, t.freq, &t.positions, &t.offsets, &t.payloads);
        }
        out
    }
}

impl FromIterator<InvertedTerm> for InvertedTerms {
    fn from_iter<I: IntoIterator<Item = InvertedTerm>>(iter: I) -> Self {
        iter.into_iter().collect::<Vec<_>>().into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terms_read_back_as_pushed() {
        let owned = vec![
            InvertedTerm {
                term: b"a".to_vec(),
                freq: 2,
                positions: vec![0, 3],
                offsets: vec![(0, 1), (6, 7)],
                payloads: vec![b"x".to_vec(), Vec::new()],
            },
            InvertedTerm {
                term: b"bc".to_vec(),
                freq: 1,
                positions: vec![1],
                offsets: vec![(2, 4)],
                payloads: Vec::new(),
            },
            InvertedTerm {
                term: b"d".to_vec(),
                freq: 1,
                positions: vec![2],
                offsets: vec![(5, 6)],
                payloads: vec![b"yz".to_vec()],
            },
        ];
        let flat = InvertedTerms::from(owned.clone());
        assert_eq!(flat.len(), 3);
        assert_eq!(flat.to_vec(), owned);
        let d = flat.get(2);
        assert_eq!((d.term, d.payloads.get(0)), (&b"d"[..], &b"yz"[..]));
        assert!(flat.get(1).payloads.is_empty());

        let mut built = InvertedTerms::new();
        built
            .start(b"a")
            .freq(2)
            .position(0)
            .offsets(0, 1)
            .payload(b"x")
            .position(3)
            .offsets(6, 7)
            .payload(b"");
        built.start(b"bc").freq(1).position(1).offsets(2, 4);
        built
            .start(b"d")
            .freq(1)
            .position(2)
            .offsets(5, 6)
            .payload(b"yz");
        assert_eq!(built, flat);
    }
}
