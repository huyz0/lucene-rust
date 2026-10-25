//! Lucene's `BulkScorerWrapperScorer`: buffer bulk hits for a document iterator.
//!
//! The two arrays, cursor, refill loop and binary search follow Java. After
//! measuring the faithful port, the next buffered hit is checked before the
//! binary-search fallback, avoiding a search for sequential iteration. Rust's
//! collector receives the score with each document instead of `setScorer`.
//! `iterator()` borrows the wrapper exclusively; successive borrows share its
//! cursor just as Java's anonymous iterators do. Zero/negative buffer sizes
//! are rejected (Java can loop forever with zero), and overflowing refill
//! windows return an error instead of wrapping. The real search adapter needs
//! the caller's known cost: `LeafBulkScorer` has no common cost accessor.

use std::fmt;

use lucene_search::collector::ScoringCollector;
use lucene_search::test_support::LeafBulkScorer;

/// Lucene's terminal document ID.
pub const NO_MORE_DOCS: i32 = i32::MAX;

/// The portion of Java's `BulkScorer` consumed by the wrapper.
pub trait BulkScorer {
    /// The original source failure, retained without conversion.
    type Error;

    /// Java: `BulkScorer.cost()`.
    fn cost(&self) -> u64;

    /// Collect `[min, max)` and return a lower bound for the next document.
    /// Java's wrapper always passes null `acceptDocs`.
    fn score(
        &mut self,
        collector: &mut dyn ScoringCollector,
        min: i32,
        max: i32,
    ) -> Result<i32, Self::Error>;
}

/// Adapts an actual indexed query's bulk scorer, with its caller-known cost.
pub struct LeafBulkSource<'a> {
    inner: LeafBulkScorer<'a>,
    known_cost: u64,
}

impl<'a> LeafBulkSource<'a> {
    /// Keep this exact bulk scorer; no second scorer is constructed for cost.
    pub fn new(inner: LeafBulkScorer<'a>, known_cost: u64) -> Self {
        Self { inner, known_cost }
    }
}

impl BulkScorer for LeafBulkSource<'_> {
    type Error = lucene_search::Error;

    fn cost(&self) -> u64 {
        self.known_cost
    }

    fn score(
        &mut self,
        collector: &mut dyn ScoringCollector,
        min: i32,
        max: i32,
    ) -> Result<i32, Self::Error> {
        self.inner.score(collector, None, min, max)
    }
}

/// An invalid constructor buffer size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidBufferSize(pub i32);

impl fmt::Display for InvalidBufferSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "buffer size must be positive: {}", self.0)
    }
}

impl std::error::Error for InvalidBufferSize {}

/// A bulk-source failure or a refill window Java would overflow.
#[derive(Debug, PartialEq, Eq)]
pub enum Error<E> {
    /// The source error, preserved intact.
    Source(E),
    /// `min + buffer_size` exceeds the document-ID range.
    WindowOverflow {
        /// Start of the requested refill.
        min: i32,
        /// Number of document IDs in a refill window.
        buffer_size: i32,
    },
}

impl<E: fmt::Display> fmt::Display for Error<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Source(error) => error.fmt(f),
            Self::WindowOverflow { min, buffer_size } => {
                write!(f, "bulk refill window overflows: {min} + {buffer_size}")
            }
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for Error<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Source(error) => Some(error),
            Self::WindowOverflow { .. } => None,
        }
    }
}

/// A document scorer backed by buffered bulk collection.
pub struct BulkScorerWrapperScorer<B> {
    scorer: B,
    docs: Box<[i32]>,
    scores: Box<[f32]>,
    buffer_size: i32,
    buffer_length: usize,
    i: i32,
    doc: i32,
    next: i32,
}

impl<B: BulkScorer> BulkScorerWrapperScorer<B> {
    /// Java: `BulkScorerWrapperScorer(scorer, bufferSize)`.
    pub fn new(scorer: B, buffer_size: i32) -> Result<Self, InvalidBufferSize> {
        if buffer_size <= 0 {
            return Err(InvalidBufferSize(buffer_size));
        }
        Ok(Self {
            scorer,
            docs: vec![0; buffer_size as usize].into_boxed_slice(),
            scores: vec![0.0; buffer_size as usize].into_boxed_slice(),
            buffer_size,
            buffer_length: 0,
            i: -1,
            doc: -1,
            next: 0,
        })
    }

    /// Java: `iterator()`; every borrow shares this wrapper's cursor.
    pub fn iterator(&mut self) -> &mut Self {
        self
    }

    /// Java: `docID()`.
    pub fn doc_id(&self) -> i32 {
        self.doc
    }

    /// Java: `iterator().cost()`.
    pub fn cost(&self) -> u64 {
        self.scorer.cost()
    }

    /// Java: `score()`; requires a current document, as in Java.
    pub fn score(&self) -> f32 {
        self.scores[self.i as usize]
    }

    /// Java: `getMaxScore(upTo)`.
    pub fn max_score(&self, _up_to: i32) -> f32 {
        f32::INFINITY
    }

    /// Inspect the wrapped source, including its collection trace.
    pub fn source(&self) -> &B {
        &self.scorer
    }

    /// Java: `iterator().nextDoc()`.
    pub fn next_doc(&mut self) -> Result<i32, Error<B::Error>> {
        self.advance(self.doc.wrapping_add(1))
    }

    /// Java: `iterator().advance(target)`, including its binary-search range.
    pub fn advance(&mut self, target: i32) -> Result<i32, Error<B::Error>> {
        if self.buffer_length == 0 || self.docs[self.buffer_length - 1] < target {
            self.refill(target)?;
        }
        let mut low = self.i.wrapping_add(1) as usize;
        let mut high = self.buffer_length;
        // The measured nextDoc/sparse-advance cases usually need this first
        // remaining hit. The strictly increasing bulk hits make it the same
        // insertion point the binary search would return.
        if low < high && self.docs[low] >= target {
            self.i = low as i32;
            self.doc = self.docs[low];
            return Ok(self.doc);
        }
        while low < high {
            let mid = low + (high - low) / 2;
            match self.docs[mid].cmp(&target) {
                std::cmp::Ordering::Less => low = mid + 1,
                std::cmp::Ordering::Greater => high = mid,
                std::cmp::Ordering::Equal => {
                    low = mid;
                    break;
                }
            }
        }
        self.i = low as i32;
        self.doc = if low == self.buffer_length {
            NO_MORE_DOCS
        } else {
            self.docs[low]
        };
        Ok(self.doc)
    }

    // Java: `refill(target)`.
    fn refill(&mut self, target: i32) -> Result<(), Error<B::Error>> {
        self.buffer_length = 0;
        while self.next != NO_MORE_DOCS && self.buffer_length == 0 {
            let min = target.max(self.next);
            let max = min
                .checked_add(self.buffer_size)
                .ok_or(Error::WindowOverflow {
                    min,
                    buffer_size: self.buffer_size,
                })?;
            let mut collector = BufferCollector {
                docs: &mut self.docs,
                scores: &mut self.scores,
                length: &mut self.buffer_length,
            };
            self.next = self
                .scorer
                .score(&mut collector, min, max)
                .map_err(Error::Source)?;
        }
        self.i = -1;
        Ok(())
    }
}

struct BufferCollector<'a> {
    docs: &'a mut [i32],
    scores: &'a mut [f32],
    length: &'a mut usize,
}

impl ScoringCollector for BufferCollector<'_> {
    fn collect(&mut self, doc_id: i32, score: f32) {
        self.docs[*self.length] = doc_id;
        self.scores[*self.length] = score;
        *self.length += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Empty;
    impl BulkScorer for Empty {
        type Error = std::io::Error;
        fn cost(&self) -> u64 {
            0
        }
        fn score(
            &mut self,
            _: &mut dyn ScoringCollector,
            _: i32,
            _: i32,
        ) -> Result<i32, Self::Error> {
            Ok(NO_MORE_DOCS)
        }
    }

    #[test]
    fn empty_source_has_zero_cost_and_exhausts() {
        let mut wrapper = BulkScorerWrapperScorer::new(Empty, 4).unwrap();
        assert_eq!(wrapper.cost(), 0);
        assert_eq!(wrapper.next_doc().unwrap(), NO_MORE_DOCS);
        assert_eq!(wrapper.doc_id(), NO_MORE_DOCS);
    }

    #[test]
    fn invalid_sizes_and_overflow_are_explicit() {
        for size in [i32::MIN, -1, 0] {
            let error = BulkScorerWrapperScorer::new(Empty, size).err().unwrap();
            assert_eq!(error, InvalidBufferSize(size));
            assert_eq!(
                error.to_string(),
                format!("buffer size must be positive: {size}")
            );
        }
        let mut wrapper = BulkScorerWrapperScorer::new(Empty, 4).unwrap();
        let error = wrapper.advance(i32::MAX - 2).unwrap_err();
        assert!(
            matches!(error, Error::WindowOverflow { min, buffer_size: 4 } if min == i32::MAX - 2)
        );
        assert_eq!(
            error.to_string(),
            "bulk refill window overflows: 2147483645 + 4"
        );
        assert!(std::error::Error::source(&error).is_none());
        assert_eq!(wrapper.doc_id(), -1);
        let source = Error::Source(std::io::Error::other("original"));
        assert_eq!(source.to_string(), "original");
        assert_eq!(
            std::error::Error::source(&source).unwrap().to_string(),
            "original"
        );
    }
}
