//! Port of the composite term views over a multi-segment reader:
//! `org.apache.lucene.index.MultiFields` (the indexed field names),
//! `MultiTerms` (one field's terms across segments, with summed statistics),
//! `MultiTermsEnum` (their union, in term order) and `MultiPostingsEnum` (a
//! term's postings across segments, in top-level doc ids).
//!
//! Built over the segments [`crate::directory_reader::DirectoryReader::open_segments`]
//! opens ([`OpenSegment`]): each segment's term dictionary, `.doc` input and
//! doc base.
//!
//! # What differs from Java
//!
//! - `MultiTermsEnum` keeps its subs in a priority queue; here the smallest
//!   current term is found by a scan of the subs -- the same answers, and a
//!   reader has few segments.
//! - [`MultiPostingsEnum`] carries docs and freqs. Positions, offsets and
//!   payloads (`nextPosition`, `startOffset`, ...) are not offered; a caller
//!   needing them reads the segment's own positions.
//! - Like Java's, the postings and statistics include deleted documents:
//!   `PostingsEnum` never filters live docs (see [`crate::multi_bits`]).
//! - `ord()`/`seekExact(long)` are unsupported in Java too; `intersect` and
//!   `impacts` are not offered.

use lucene_codecs::blocktree::{FieldTerms, SeekStatus, TermsEnum};
use lucene_codecs::postings::DocInput;

use crate::multi_segment::OpenSegment;
use crate::Result;

/// `DocIdSetIterator.NO_MORE_DOCS`.
pub const NO_MORE_DOCS: i32 = i32::MAX;

/// `MultiFields`' field iteration: the names of the fields with terms in
/// any of `segments`, sorted, each once.
pub fn indexed_fields(segments: &[OpenSegment<'_>]) -> Vec<String> {
    let mut names: Vec<String> = segments
        .iter()
        .flat_map(|s| s.fields.iter_fields().map(|(name, _)| name.to_string()))
        .collect();
    names.sort();
    names.dedup();
    names
}

/// `MultiTerms`: one field's terms across the segments that have it.
pub struct MultiTerms<'a> {
    /// Each segment holding the field: its terms, `.doc` input and doc base
    /// (`subs` with their `ReaderSlice`s).
    subs: Vec<(&'a FieldTerms, Option<&'a DocInput<'a>>, i32)>,
}

impl<'a> MultiTerms<'a> {
    /// `MultiTerms.getTerms(reader, field)`: `None` when no segment has
    /// terms for `field`.
    pub fn get_terms(segments: &'a [OpenSegment<'a>], field: &str) -> Option<Self> {
        let subs: Vec<_> = segments
            .iter()
            .filter_map(|s| {
                s.fields
                    .field(field)
                    .map(|terms| (terms, s.doc_in, s.doc_base))
            })
            .collect();
        (!subs.is_empty()).then_some(MultiTerms { subs })
    }

    /// `getSumTotalTermFreq()`.
    pub fn sum_total_term_freq(&self) -> i64 {
        self.subs.iter().fold(0i64, |acc, (t, _, _)| {
            acc.saturating_add(t.sum_total_term_freq)
        })
    }

    /// `getSumDocFreq()`.
    pub fn sum_doc_freq(&self) -> i64 {
        self.subs
            .iter()
            .fold(0i64, |acc, (t, _, _)| acc.saturating_add(t.sum_doc_freq))
    }

    /// `getDocCount()`.
    pub fn doc_count(&self) -> i32 {
        self.subs
            .iter()
            .fold(0i32, |acc, (t, _, _)| acc.saturating_add(t.doc_count))
    }

    /// `getMin()`: the smallest term of any segment.
    pub fn min(&self) -> &[u8] {
        self.subs
            .iter()
            .map(|(t, _, _)| t.min_term.as_slice())
            .min()
            .unwrap_or_default()
    }

    /// `getMax()`: the largest term of any segment.
    pub fn max(&self) -> &[u8] {
        self.subs
            .iter()
            .map(|(t, _, _)| t.max_term.as_slice())
            .max()
            .unwrap_or_default()
    }

    /// `size()`: unknown (`-1`), as in Java -- terms shared by segments are
    /// counted once, which only a full enumeration can tell.
    // SENTINEL: `-1` = unknown, Java's `Terms.size()` contract; no caller
    // indexes with it.
    pub fn size(&self) -> i64 {
        -1
    }

    /// `iterator()`: every term of the field, in byte order, each once.
    pub fn iterator(&self) -> MultiTermsEnum<'a> {
        MultiTermsEnum {
            subs: self
                .subs
                .iter()
                .map(|&(terms, doc_in, doc_base)| Sub {
                    te: terms.iter(),
                    term: None,
                    doc_in,
                    doc_base,
                })
                .collect(),
            current: None,
            top: Vec::new(),
            started: false,
        }
    }
}

struct Sub<'a> {
    te: TermsEnum<'a>,
    /// The sub's current term; `None` before it starts and once exhausted.
    term: Option<Vec<u8>>,
    doc_in: Option<&'a DocInput<'a>>,
    doc_base: i32,
}

impl Sub<'_> {
    fn advance(&mut self) -> Result<()> {
        self.term = self.te.try_next_term()?.map(<[u8]>::to_vec);
        Ok(())
    }

    fn try_seek_ceil(&mut self, target: &[u8]) -> Result<()> {
        self.term = match self.te.try_seek_ceil(target)? {
            SeekStatus::End => None,
            SeekStatus::Found | SeekStatus::NotFound => self.te.term().map(<[u8]>::to_vec),
        };
        Ok(())
    }
}

/// `MultiTermsEnum`.
pub struct MultiTermsEnum<'a> {
    subs: Vec<Sub<'a>>,
    /// `current`: the term the enum stands on.
    current: Option<Vec<u8>>,
    /// `top`: the subs standing on `current`, ascending (segment order).
    top: Vec<usize>,
    started: bool,
}

impl MultiTermsEnum<'_> {
    /// `pullTop()`: the smallest term among the subs, and every sub on it.
    fn pull_top(&mut self) {
        let min = self
            .subs
            .iter()
            .filter_map(|s| s.term.as_deref())
            .min()
            .map(<[u8]>::to_vec);
        self.top = match &min {
            None => Vec::new(),
            Some(m) => (0..self.subs.len())
                .filter(|&i| self.subs[i].term.as_deref() == Some(m.as_slice()))
                .collect(),
        };
        self.current = min;
    }

    /// `next()`: the next term, or `None` past the last.
    pub fn try_next(&mut self) -> Result<Option<&[u8]>> {
        if !self.started {
            self.started = true;
            for sub in &mut self.subs {
                sub.advance()?;
            }
        } else {
            // `pushTop()`: the subs on the current term move on.
            for &i in &self.top {
                self.subs[i].advance()?;
            }
        }
        self.pull_top();
        Ok(self.current.as_deref())
    }

    /// `seekCeil(target)`: positions every sub at its smallest term `>=
    /// target`, and the enum on the smallest of those.
    pub fn try_seek_ceil(&mut self, target: &[u8]) -> Result<SeekStatus> {
        self.started = true;
        for sub in &mut self.subs {
            sub.try_seek_ceil(target)?;
        }
        self.pull_top();
        Ok(match &self.current {
            None => SeekStatus::End,
            Some(t) if t.as_slice() == target => SeekStatus::Found,
            Some(_) => SeekStatus::NotFound,
        })
    }

    /// `seekExact(term)`: whether `term` exists in any segment; the enum is
    /// positioned on it when it does.
    pub fn try_seek_exact(&mut self, term: &[u8]) -> Result<bool> {
        Ok(self.try_seek_ceil(term)? == SeekStatus::Found)
    }

    /// `term()`: the current term.
    pub fn term(&self) -> Option<&[u8]> {
        self.current.as_deref()
    }

    /// `docFreq()`: summed over the segments holding the current term.
    pub fn doc_freq(&mut self) -> Result<i32> {
        let mut sum = 0i32;
        for &i in &self.top {
            let stats = self.subs[i].te.try_stats()?.map_or(0, |s| s.doc_freq);
            sum = sum.saturating_add(stats);
        }
        Ok(sum)
    }

    /// `totalTermFreq()`: summed over the segments holding the current term.
    pub fn total_term_freq(&mut self) -> Result<i64> {
        let mut sum = 0i64;
        for &i in &self.top {
            let stats = self.subs[i]
                .te
                .try_stats()?
                .map_or(0, |s| s.total_term_freq);
            sum = sum.saturating_add(stats);
        }
        Ok(sum)
    }

    /// `postings(null, FREQS)`: the current term's documents across the
    /// segments holding it, in top-level doc ids, with their freqs.
    pub fn postings(&mut self) -> Result<MultiPostingsEnum> {
        let mut docs = Vec::new();
        let mut freqs = Vec::new();
        for &i in &self.top {
            let sub = &mut self.subs[i];
            let Some(postings) = sub.te.try_current_postings(sub.doc_in)? else {
                continue;
            };
            for (k, &doc) in postings.docs.iter().enumerate() {
                docs.push(doc.saturating_add(sub.doc_base));
                // A field indexed without freqs reads as freq 1 (Java's
                // `PostingsEnum.freq()` for DOCS postings).
                freqs.push(postings.freqs.get(k).copied().unwrap_or(1));
            }
        }
        Ok(MultiPostingsEnum {
            docs,
            freqs,
            upto: None,
        })
    }
}

/// `MultiPostingsEnum`, docs and freqs.
#[derive(Debug, Clone)]
pub struct MultiPostingsEnum {
    docs: Vec<i32>,
    freqs: Vec<i32>,
    /// The current position; `None` before the first doc.
    upto: Option<usize>,
}

impl MultiPostingsEnum {
    /// `docID()`: `-1` before the first doc, [`NO_MORE_DOCS`] past the last.
    // SENTINEL: `-1` = unpositioned, `DocIdSetIterator.docID()`'s contract.
    pub fn doc_id(&self) -> i32 {
        match self.upto {
            None => -1,
            Some(i) => self.docs.get(i).copied().unwrap_or(NO_MORE_DOCS),
        }
    }

    /// `freq()` of the current doc (`0` when unpositioned or exhausted).
    pub fn freq(&self) -> i32 {
        self.upto
            .and_then(|i| self.freqs.get(i))
            .copied()
            .unwrap_or(0)
    }

    /// `nextDoc()`.
    pub fn next_doc(&mut self) -> i32 {
        let next = self.upto.map_or(0, |i| i.saturating_add(1));
        self.upto = Some(next.min(self.docs.len()));
        self.doc_id()
    }

    /// `advance(target)`: the first doc `>= target` from the next one on.
    pub fn advance(&mut self, target: i32) -> i32 {
        let from = self
            .upto
            .map_or(0, |i| i.saturating_add(1))
            .min(self.docs.len());
        let offset = self.docs[from..].partition_point(|&d| d < target);
        self.upto = Some(from.saturating_add(offset));
        self.doc_id()
    }

    /// `cost()`: how many docs it holds.
    pub fn cost(&self) -> usize {
        self.docs.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn postings_iterate_and_advance() {
        let mut p = MultiPostingsEnum {
            docs: vec![1, 4, 9],
            freqs: vec![2, 1, 3],
            upto: None,
        };
        assert_eq!(p.doc_id(), -1);
        assert_eq!(p.freq(), 0);
        assert_eq!(p.cost(), 3);
        assert_eq!(p.next_doc(), 1);
        assert_eq!(p.freq(), 2);
        assert_eq!(p.advance(5), 9);
        assert_eq!(p.freq(), 3);
        assert_eq!(p.next_doc(), NO_MORE_DOCS);
        assert_eq!(p.next_doc(), NO_MORE_DOCS);
        assert_eq!(p.advance(100), NO_MORE_DOCS);
        assert_eq!(p.freq(), 0);
    }
}
