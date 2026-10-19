//! `org.apache.lucene.sandbox.facet.iterators`: ordinal iterators that
//! count, filter and rank other iterators' ordinals.

use std::sync::Arc;

use crate::facets::taxonomy::FacetLabel;
use crate::java_priority_queue::JavaPriorityQueue;
use crate::{Error, Result};

use super::comparable::ComparableSupplier;
use super::labels::LabelToOrd;
use super::{from_array, ArrayOrdinalIterator, FacetRecorder, OrdinalIterator, NO_MORE_ORDS};
use crate::facets::taxonomy::ParallelTaxonomyArrays;

/// `LengthOrdinalIterator`: its delegate's ordinals, counted.
pub struct LengthOrdinalIterator<I> {
    delegate: I,
    complete: bool,
    length: i32,
}

impl<I: OrdinalIterator> LengthOrdinalIterator<I> {
    /// `new LengthOrdinalIterator(delegate)`.
    pub fn new(delegate: I) -> Self {
        LengthOrdinalIterator {
            delegate,
            complete: false,
            length: 0,
        }
    }

    /// `length()`: how many ordinals the delegate had.
    ///
    /// # Errors
    /// [`Error::IllegalState`] before the delegate is consumed.
    pub fn length(&self) -> Result<i32> {
        if self.complete {
            Ok(self.length)
        } else {
            Err(Error::IllegalState(
                "You can't request count before the ordinal iterator was consumed".into(),
            ))
        }
    }
}

impl<I: OrdinalIterator> OrdinalIterator for LengthOrdinalIterator<I> {
    fn next_ord(&mut self) -> Result<i32> {
        let ord = self.delegate.next_ord()?;
        if ord == NO_MORE_ORDS {
            self.complete = true;
        } else {
            self.length = self.length.wrapping_add(1);
        }
        Ok(ord)
    }
}

/// `TaxonomyChildrenOrdinalIterator`: the source's ordinals whose parent is
/// `parent_ord`.
pub struct TaxonomyChildrenOrdinalIterator<I> {
    source: I,
    parents: Arc<dyn ParallelTaxonomyArrays>,
    parent_ord: i32,
}

impl<I: OrdinalIterator> TaxonomyChildrenOrdinalIterator<I> {
    /// `new TaxonomyChildrenOrdinalIterator(sourceOrds, parents, parentOrd)`;
    /// `arrays` are the taxonomy's parallel arrays, whose `parents()` it
    /// reads. An `INVALID_ORD` parent (Java's assertion) matches only the
    /// root, whose parent it is, as Java without assertions.
    pub fn new(source: I, arrays: Arc<dyn ParallelTaxonomyArrays>, parent_ord: i32) -> Self {
        TaxonomyChildrenOrdinalIterator {
            source,
            parents: arrays,
            parent_ord,
        }
    }
}

impl<I: OrdinalIterator> OrdinalIterator for TaxonomyChildrenOrdinalIterator<I> {
    fn next_ord(&mut self) -> Result<i32> {
        loop {
            let ord = self.source.next_ord()?;
            if ord == NO_MORE_ORDS {
                return Ok(NO_MORE_ORDS);
            }
            if self.parents.parents().get(ord) == self.parent_ord {
                return Ok(ord);
            }
        }
    }
}

/// `CandidateSetOrdinalIterator`: the candidates' ordinals (in candidate
/// order, repeats kept, unknown labels dropped) that the recorder has.
pub struct CandidateSetOrdinalIterator<'r> {
    candidates: ArrayOrdinalIterator,
    recorder: &'r dyn FacetRecorder,
}

impl<'r> CandidateSetOrdinalIterator<'r> {
    /// `new CandidateSetOrdinalIterator(facetRecorder, candidateLabels,
    /// labelToOrd)`: an empty recorder looks no label up.
    ///
    /// # Errors
    /// What the recorder or the lookup reports.
    pub fn new(
        recorder: &'r dyn FacetRecorder,
        candidate_labels: &[FacetLabel],
        label_to_ord: &dyn LabelToOrd,
    ) -> Result<Self> {
        let candidates = if recorder.is_empty()? {
            from_array(Vec::new())
        } else {
            from_array(label_to_ord.get_ords(candidate_labels)?)
        };
        Ok(CandidateSetOrdinalIterator {
            candidates,
            recorder,
        })
    }
}

impl OrdinalIterator for CandidateSetOrdinalIterator<'_> {
    fn next_ord(&mut self) -> Result<i32> {
        loop {
            let ord = self.candidates.next_ord()?;
            if ord == NO_MORE_ORDS || self.recorder.contains(ord)? {
                return Ok(ord);
            }
        }
    }
}

/// `TopnOrdinalIterator`: the source's top `n` ordinals by the supplier's
/// comparables, greatest first, chosen through Lucene's `PriorityQueue`.
pub struct TopnOrdinalIterator<I, S> {
    source: I,
    supplier: S,
    top_n: usize,
    result: Option<std::vec::IntoIter<i32>>,
}

impl<I: OrdinalIterator, S: ComparableSupplier> TopnOrdinalIterator<I, S> {
    /// `new TopnOrdinalIterator(sourceOrds, comparableSupplier, topN)`.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] when `top_n` is not positive.
    pub fn new(source: I, supplier: S, top_n: i32) -> Result<Self> {
        match usize::try_from(top_n) {
            Ok(n) if n > 0 => Ok(TopnOrdinalIterator {
                source,
                supplier,
                top_n: n,
                result: None,
            }),
            _ => Err(Error::IllegalArgument(format!(
                "topN must be > 0 (got: {top_n})"
            ))),
        }
    }

    /// `getTopN()`.
    fn top_n(&mut self) -> Result<Vec<i32>> {
        let mut queue = JavaPriorityQueue::new(
            self.top_n,
            |a: &(i32, S::Comparable), b: &(i32, S::Comparable)| a.1 < b.1,
        );
        loop {
            let ord = self.source.next_ord()?;
            if ord == NO_MORE_ORDS {
                break;
            }
            queue.insert_with_overflow((ord, self.supplier.create_comparable(ord)?));
        }
        // The queue pops the least first.
        let mut result = Vec::with_capacity(queue.size());
        while let Some((ord, _)) = queue.pop() {
            result.push(ord);
        }
        result.reverse();
        Ok(result)
    }
}

impl<I: OrdinalIterator, S: ComparableSupplier> OrdinalIterator for TopnOrdinalIterator<I, S> {
    fn next_ord(&mut self) -> Result<i32> {
        if self.result.is_none() {
            self.result = Some(self.top_n()?.into_iter());
        }
        Ok(self
            .result
            .as_mut()
            .and_then(Iterator::next)
            .unwrap_or(NO_MORE_ORDS))
    }
}
