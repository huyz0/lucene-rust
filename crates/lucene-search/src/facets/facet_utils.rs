//! `FacetUtils`, and the iterator conjunction the facet counters intersect
//! their matching documents with doc values through
//! (`ConjunctionUtils.intersectIterators`).

use lucene_util::fixed_bit_set::FixedBitSet;

use crate::facets::facets_collector::DocIdSet;
use lucene_codecs::field_infos::DocValuesType;

use crate::reader::{DocIdSetIterator, DocValuesIterator, LeafReader, NO_MORE_DOCS};
use crate::{Error, Result};

/// `FacetUtils.liveDocsDISI(it, liveDocs)`: `it` without the deleted
/// documents.
pub fn live_docs_disi<'i>(
    it: Box<dyn DocIdSetIterator + 'i>,
    live_docs: &'i FixedBitSet,
) -> LiveDocsIterator<'i> {
    LiveDocsIterator { it, live_docs }
}

/// [`live_docs_disi`]'s iterator.
pub struct LiveDocsIterator<'i> {
    it: Box<dyn DocIdSetIterator + 'i>,
    live_docs: &'i FixedBitSet,
}

impl LiveDocsIterator<'_> {
    /// `doNext(doc)`: the first live document from `doc` on.
    fn do_next(&mut self, mut doc: i32) -> Result<i32> {
        while doc != NO_MORE_DOCS && !self.live_docs.get_doc(doc) {
            doc = self.it.next_doc()?;
        }
        Ok(doc)
    }
}

impl DocIdSetIterator for LiveDocsIterator<'_> {
    fn doc_id(&self) -> i32 {
        self.it.doc_id()
    }

    fn next_doc(&mut self) -> Result<i32> {
        let doc = self.it.next_doc()?;
        self.do_next(doc)
    }

    fn advance(&mut self, target: i32) -> Result<i32> {
        let doc = self.it.advance(target)?;
        self.do_next(doc)
    }

    fn cost(&self) -> i64 {
        self.it.cost()
    }
}

/// `ConjunctionUtils.intersectIterators(iterators)`: the documents every
/// iterator has, led by the cheapest.
pub struct ConjunctionIterator<'i> {
    /// Cheapest first.
    its: Vec<Box<dyn DocIdSetIterator + 'i>>,
    doc: i32,
}

impl<'i> ConjunctionIterator<'i> {
    /// Over `its`, which must not be empty for the conjunction to match
    /// anything (an empty conjunction matches nothing).
    pub fn new(mut its: Vec<Box<dyn DocIdSetIterator + 'i>>) -> Self {
        its.sort_by_key(|it| it.cost());
        ConjunctionIterator { its, doc: -1 }
    }

    /// `doNext(doc)`: from the lead's `doc`, the next document all have.
    fn do_next(&mut self, mut doc: i32) -> Result<i32> {
        'advance_head: loop {
            if doc == NO_MORE_DOCS {
                self.doc = NO_MORE_DOCS;
                return Ok(doc);
            }
            for i in 1..self.its.len() {
                let other = &mut self.its[i];
                let mut next = other.doc_id();
                if next < doc {
                    next = other.advance(doc)?;
                }
                if next > doc {
                    doc = self.its[0].advance(next)?;
                    continue 'advance_head;
                }
            }
            self.doc = doc;
            return Ok(doc);
        }
    }
}

impl DocIdSetIterator for ConjunctionIterator<'_> {
    fn doc_id(&self) -> i32 {
        self.doc
    }

    fn next_doc(&mut self) -> Result<i32> {
        let doc = match self.its.first_mut() {
            Some(lead) => lead.next_doc()?,
            None => NO_MORE_DOCS,
        };
        self.do_next(doc)
    }

    fn advance(&mut self, target: i32) -> Result<i32> {
        let doc = match self.its.first_mut() {
            Some(lead) => lead.advance(target)?,
            None => NO_MORE_DOCS,
        };
        self.do_next(doc)
    }

    fn cost(&self) -> i64 {
        self.its.first().map_or(0, |it| it.cost())
    }
}

/// The two-iterator conjunction the counters run: `f` is called with `b`
/// positioned on each document both iterators have, in order (so it can read
/// `b`'s values there). The set and order are
/// [`ConjunctionIterator`]'s.
#[inline]
pub(crate) fn for_each_in_both<A, B>(
    a: &mut A,
    b: &mut B,
    mut f: impl FnMut(&mut B) -> Result<()>,
) -> Result<()>
where
    A: DocIdSetIterator + ?Sized,
    B: DocIdSetIterator + ?Sized,
{
    let mut doc = a.next_doc()?;
    while doc != NO_MORE_DOCS {
        let mut other = b.doc_id();
        if other < doc {
            other = b.advance(doc)?;
        }
        if other == doc {
            f(b)?;
            doc = a.next_doc()?;
        } else {
            doc = a.advance(other)?;
        }
    }
    Ok(())
}

/// [`for_each_in_both`] over a segment's hits and doc values `b`, with the
/// hits walked directly rather than through their iterator: `f` is called
/// with `b` positioned on each hit `b` has a value for, in order. The lead is
/// the cheaper side, as `ConjunctionDISI`'s is: a bitset at least as large as
/// `b` follows `b`'s documents and tests each against the bits; otherwise
/// each hit is `advance_exact`ed to.
#[inline]
pub(crate) fn for_each_hit<B>(
    hits: &DocIdSet,
    b: &mut B,
    mut f: impl FnMut(&mut B) -> Result<()>,
) -> Result<()>
where
    B: DocValuesIterator + ?Sized,
{
    match hits {
        DocIdSet::Empty => {}
        DocIdSet::Array(docs) => {
            for &doc in docs {
                if b.advance_exact(doc)? {
                    f(b)?;
                }
            }
        }
        DocIdSet::Bits { bits, cost } if *cost >= b.cost() => {
            let words = bits.words();
            let mut doc = b.next_doc()?;
            while doc != NO_MORE_DOCS {
                if word_bit(words, doc) {
                    f(b)?;
                }
                doc = b.next_doc()?;
            }
        }
        DocIdSet::Bits { bits, .. } => {
            try_for_each_set_bit(bits, |doc| {
                if b.advance_exact(doc)? {
                    f(b)?;
                }
                Ok(())
            })?;
        }
    }
    Ok(())
}

/// [`for_each_hit`] over the hits in `lo..hi` only (one chunk of a segment
/// counted on its own task), `b` a fresh iterator: the same calls, in the
/// same order, that [`for_each_hit`] makes for those documents.
pub(crate) fn for_each_hit_in<B>(
    hits: &DocIdSet,
    b: &mut B,
    lo: i32,
    hi: i32,
    mut f: impl FnMut(&mut B) -> Result<()>,
) -> Result<()>
where
    B: DocValuesIterator + ?Sized,
{
    let lo = lo.max(0);
    if lo >= hi {
        return Ok(());
    }
    match hits {
        DocIdSet::Empty => {}
        DocIdSet::Array(docs) => {
            let start = docs.partition_point(|&d| d < lo);
            for &doc in docs.get(start..).unwrap_or_default() {
                if doc >= hi {
                    break;
                }
                if b.advance_exact(doc)? {
                    f(b)?;
                }
            }
        }
        DocIdSet::Bits { bits, cost } if *cost >= b.cost() => {
            let words = bits.words();
            let mut doc = b.advance(lo)?;
            while doc < hi {
                if word_bit(words, doc) {
                    f(b)?;
                }
                doc = b.next_doc()?;
            }
        }
        DocIdSet::Bits { bits, .. } => {
            let words = bits.words();
            // Both ends are non-negative here.
            let (lo, hi) = (lo as usize, hi as usize);
            let last = hi.div_ceil(64).min(words.len());
            let first = lo / 64;
            for (wi, &word) in (first..).zip(words.get(first..last).unwrap_or_default()) {
                let base = wi.wrapping_mul(64);
                let mut w = word;
                // Only the first word starts under `lo` (by under 64), and
                // only the last ends past `hi` (by under 64).
                if base < lo {
                    w &= u64::MAX.wrapping_shl(lo.wrapping_sub(base) as u32);
                }
                let below_hi = hi.wrapping_sub(base);
                if below_hi < 64 {
                    w &= 1u64.wrapping_shl(below_hi as u32).wrapping_sub(1);
                }
                while w != 0 {
                    let Ok(doc) = i32::try_from(base | w.trailing_zeros() as usize) else {
                        return Ok(());
                    };
                    if b.advance_exact(doc)? {
                        f(b)?;
                    }
                    w &= w.wrapping_sub(1);
                }
            }
        }
    }
    Ok(())
}

/// `DocValues.unwrapSingleton(DocValues.getSortedNumeric(reader, field)) !=
/// null`: a field the leaf lacks (`emptySortedNumeric()` is a singleton), a
/// `NUMERIC` field, or a `SORTED_NUMERIC` field in the codec's
/// one-value-per-document form.
pub(crate) fn sorted_numeric_is_single_valued<R: LeafReader + ?Sized>(
    reader: &R,
    field: &str,
) -> bool {
    match reader.field_infos().field_by_name(field) {
        None => true,
        Some(fi) => match fi.doc_values_type {
            DocValuesType::SortedNumeric => reader.sorted_numeric_is_singleton(field),
            _ => true,
        },
    }
}

/// The single-dim facets' `validateDimAndPathForGetChildren(dim, path)`.
pub(crate) fn validate_dim_and_path(field: &str, dim: &str, path: &[&str]) -> Result<()> {
    if dim != field {
        return Err(Error::IllegalArgument(format!(
            "invalid dim \"{dim}\"; should be \"{field}\""
        )));
    }
    if !path.is_empty() {
        return Err(Error::IllegalArgument("path.length should be 0".into()));
    }
    Ok(())
}

/// Each document of `set`, in order.
pub(crate) fn for_each_doc(set: &DocIdSet, mut f: impl FnMut(i32) -> Result<()>) -> Result<()> {
    match set {
        DocIdSet::Empty => {}
        DocIdSet::Array(docs) => {
            for &doc in docs {
                f(doc)?;
            }
        }
        DocIdSet::Bits { bits, .. } => try_for_each_set_bit(bits, f)?,
    }
    Ok(())
}

/// Bit `doc` of a bit set's words: a document outside them (negative, or
/// past the last word) is not set. The words past `len()` hold no set bit.
#[inline(always)]
pub(crate) fn word_bit(words: &[u64], doc: i32) -> bool {
    let Ok(i) = usize::try_from(doc) else {
        return false;
    };
    words.get(i >> 6).is_some_and(|&w| (w >> (i & 63)) & 1 != 0)
}

/// `f(doc)` for every set bit of `bits`, ascending, stopping at the first
/// error: the bits popped word by word (each word loaded once, the lowest
/// set bit taken and cleared) rather than a `next_set_bit` search per hit.
#[inline]
pub(crate) fn try_for_each_set_bit(
    bits: &FixedBitSet,
    mut f: impl FnMut(i32) -> Result<()>,
) -> Result<()> {
    for (wi, &word) in bits.words().iter().enumerate() {
        let mut w = word;
        while w != 0 {
            let Ok(doc) = i32::try_from((wi << 6) | w.trailing_zeros() as usize) else {
                return Ok(());
            };
            f(doc)?;
            w &= w.wrapping_sub(1);
        }
    }
    Ok(())
}

/// Each document of `it`, in order.
pub(crate) fn for_each_iter(
    it: &mut (dyn DocIdSetIterator + '_),
    mut f: impl FnMut(i32) -> Result<()>,
) -> Result<()> {
    let mut doc = it.next_doc()?;
    while doc != NO_MORE_DOCS {
        f(doc)?;
        doc = it.next_doc()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
