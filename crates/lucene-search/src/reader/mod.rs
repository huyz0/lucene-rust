//! Lucene's public reader API: `IndexReader`, `LeafReader`, `CodecReader`,
//! `CompositeReader` and the per-leaf access objects they hand out (`Terms`,
//! `TermsEnum`, `PostingsEnum`, the five doc-values iterators, `PointValues`,
//! stored fields, term vectors and KNN vector values) -- as Rust traits, so a
//! caller can read any leaf the same way whether it is a segment
//! ([`crate::directory_reader::SegmentReader`]) or a view over other leaves
//! (the wrappers in this module's children).
//!
//! OpenSearch builds on exactly this layer: it wraps readers for soft deletes,
//! field-level security and timeouts ([`exitable`]), joins them
//! ([`multi_reader`], [`parallel`]) and re-sorts them for `addIndexes`
//! ([`sorting`], [`slow_codec`]).
//!
//! # What differs from Java
//!
//! - **One trait per Java abstract class, access objects boxed.** `terms`,
//!   `getNumericDocValues`, ... return `Box<dyn Trait + '_>` borrowing the
//!   reader, so a wrapper can wrap what its inner reader returns
//!   (`FilterLeafReader`'s whole purpose) without generics leaking into every
//!   signature.
//! - **Postings are owned** (`Box<dyn PostingsEnum>`, no borrow): the segment
//!   implementation decodes a term's postings (and, when asked, positions) up
//!   front, as [`crate::multi_terms`] already did. Java decodes lazily; the
//!   scorer tree ([`crate::exec`]) keeps using the codec's lazy cursors
//!   directly and does not go through this layer.
//! - **Doc-values values are read when the iterator is positioned**: `longValue()`,
//!   `binaryValue()`, `ordValue()` and friends are infallible accessors here,
//!   and any decode error surfaces from `nextDoc`/`advance`/`advanceExact`.
//!   Every segment iterator reads through the column readers
//!   (`NumericReader`, `BinaryReader`, `SortedNumericReader`), never the
//!   per-call `doc_values::numeric_value` (docs/mechanical-gates.md,
//!   "doc-values per doc").
//! - **Reference counting is `Arc`.** `incRef`/`decRef`/`close` are the
//!   `Arc`'s clone and drop; a reader-closed listener
//!   ([`CacheHelper::add_closed_listener`]) runs when the last handle sharing
//!   its cache key is dropped.
//! - **Field numbers.** A [`StoredFieldVisitor`] and a term-vectors document
//!   name fields by number, so a view that renumbers fields (a
//!   [`parallel::ParallelLeafReader`], a
//!   [`slow_codec::SlowCompositeCodecReaderWrapper`]) translates the numbers
//!   through field names, as Java's `remap(FieldInfo)` does.

pub mod exitable;
pub mod filter;
pub mod filtered_terms_enum;
pub mod merge_readers;
pub mod multi_doc_values;
pub mod multi_reader;
pub mod parallel;
mod segment;
pub mod slow_codec;
pub mod sorting;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use lucene_codecs::blocktree::SeekStatus;
use lucene_codecs::field_infos::FieldInfos;
pub use lucene_codecs::points::{IntersectVisitor, Relation};
pub use lucene_codecs::postings::{Impact, Position};
pub use lucene_codecs::stored_fields::{StoredFieldVisitor, VisitStatus};
pub use lucene_codecs::term_vectors::TermVectorsDocument;
use lucene_index::segment_info::IndexSortField;
use lucene_util::fixed_bit_set::FixedBitSet;

use crate::{Error, Result};

/// `DocIdSetIterator.NO_MORE_DOCS`.
pub const NO_MORE_DOCS: i32 = i32::MAX;

// ---------------------------------------------------------------------------
// Iterators
// ---------------------------------------------------------------------------

/// `DocIdSetIterator`.
pub trait DocIdSetIterator {
    /// `docID()`: `-1` before the first `next_doc`/`advance`,
    /// [`NO_MORE_DOCS`] once exhausted.
    fn doc_id(&self) -> i32;
    /// `nextDoc()`.
    fn next_doc(&mut self) -> Result<i32>;
    /// `advance(target)`: the first document `>= target`; `target` must be
    /// past the current document.
    fn advance(&mut self, target: i32) -> Result<i32>;
    /// `cost()`.
    fn cost(&self) -> i64;
}

/// `PostingsEnum.flags`: what a caller of [`TermsEnum::postings`] will read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum PostingsFlags {
    /// `PostingsEnum.NONE`: doc ids only (`freq()` then reads `1`).
    None,
    /// `PostingsEnum.FREQS`.
    #[default]
    Freqs,
    /// `PostingsEnum.POSITIONS`.
    Positions,
    /// `PostingsEnum.OFFSETS`.
    Offsets,
    /// `PostingsEnum.PAYLOADS`.
    Payloads,
    /// `PostingsEnum.ALL`.
    All,
}

impl PostingsFlags {
    /// `PostingsEnum.featureRequested(flags, POSITIONS)`.
    pub fn wants_positions(self) -> bool {
        self >= PostingsFlags::Positions
    }
}

/// `PostingsEnum`.
pub trait PostingsEnum: DocIdSetIterator {
    /// `freq()` of the current document.
    fn freq(&self) -> i32;
    /// `nextPosition()`: the next position in the current document, `-1`
    /// when positions were not indexed or not requested.
    // SENTINEL: `-1` = no positions (`PostingsEnum.nextPosition`'s contract);
    // callers only compare it, never index with it.
    fn next_position(&mut self) -> Result<i32>;
    /// `startOffset()` of the last position read, `-1` without offsets.
    // SENTINEL: `-1` = no offsets; compared only.
    fn start_offset(&self) -> i32;
    /// `endOffset()` of the last position read, `-1` without offsets.
    // SENTINEL: `-1` = no offsets; compared only.
    fn end_offset(&self) -> i32;
    /// `getPayload()` of the last position read.
    fn payload(&self) -> Option<&[u8]>;
}

/// `ImpactsEnum`: a [`PostingsEnum`] that also reports, per block of
/// documents, the competitive `(freq, norm)` pairs a scorer prunes with.
pub trait ImpactsEnum: PostingsEnum {
    /// `advanceShallow(target)`: positions the impacts (not the iterator) on
    /// the block holding `target`, returning that block's last document.
    fn advance_shallow(&mut self, target: i32) -> Result<i32>;
    /// `getImpacts()`: `(levels)`, each `(docIdUpTo, impacts)`.
    fn impacts(&self) -> Vec<(i32, Vec<Impact>)>;
}

/// A term's postings decoded into memory: the [`PostingsEnum`] every
/// implementation in this module hands out.
#[derive(Debug, Clone, Default)]
pub struct MaterializedPostings {
    docs: Vec<i32>,
    freqs: Vec<i32>,
    /// One list per document, when positions were requested and indexed.
    positions: Option<Vec<Vec<Position>>>,
    upto: Option<usize>,
    /// The next position of the current document to hand out.
    pos_upto: usize,
}

impl MaterializedPostings {
    /// Postings over `docs` (ascending) with `freqs` (same length) and,
    /// optionally, each document's positions.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] when the lengths disagree.
    pub fn new(
        docs: Vec<i32>,
        freqs: Vec<i32>,
        positions: Option<Vec<Vec<Position>>>,
    ) -> Result<Self> {
        if docs.len() != freqs.len() || positions.as_ref().is_some_and(|p| p.len() != docs.len()) {
            return Err(Error::IllegalArgument(format!(
                "postings: {} docs, {} freqs, {:?} position lists",
                docs.len(),
                freqs.len(),
                positions.as_ref().map(Vec::len)
            )));
        }
        Ok(Self {
            docs,
            freqs,
            positions,
            upto: None,
            pos_upto: 0,
        })
    }

    /// Every document, freq and position list, consuming the enum.
    pub fn into_parts(self) -> (Vec<i32>, Vec<i32>, Option<Vec<Vec<Position>>>) {
        (self.docs, self.freqs, self.positions)
    }

    fn current_position(&self) -> Option<&Position> {
        let i = self.upto?;
        let list = self.positions.as_ref()?.get(i)?;
        list.get(self.pos_upto.checked_sub(1)?)
    }
}

impl DocIdSetIterator for MaterializedPostings {
    // SENTINEL: `-1` = unpositioned, `DocIdSetIterator.docID()`'s contract.
    fn doc_id(&self) -> i32 {
        match self.upto {
            None => -1,
            Some(i) => self.docs.get(i).copied().unwrap_or(NO_MORE_DOCS),
        }
    }

    fn next_doc(&mut self) -> Result<i32> {
        let next = self.upto.map_or(0, |i| i.saturating_add(1));
        self.upto = Some(next.min(self.docs.len()));
        self.pos_upto = 0;
        Ok(self.doc_id())
    }

    fn advance(&mut self, target: i32) -> Result<i32> {
        let from = self
            .upto
            .map_or(0, |i| i.saturating_add(1))
            .min(self.docs.len());
        let offset = self.docs[from..].partition_point(|&d| d < target);
        self.upto = Some(from.saturating_add(offset));
        self.pos_upto = 0;
        Ok(self.doc_id())
    }

    fn cost(&self) -> i64 {
        self.docs.len() as i64
    }
}

impl PostingsEnum for MaterializedPostings {
    fn freq(&self) -> i32 {
        self.upto
            .and_then(|i| self.freqs.get(i))
            .copied()
            .unwrap_or(0)
    }

    // SENTINEL: `-1` = no positions, `PostingsEnum.nextPosition`'s contract;
    // callers compare it, never index with it.
    fn next_position(&mut self) -> Result<i32> {
        let Some(list) = self
            .upto
            .and_then(|i| self.positions.as_ref().and_then(|p| p.get(i)))
        else {
            return Ok(-1);
        };
        let Some(p) = list.get(self.pos_upto) else {
            return Err(Error::IllegalState(
                "nextPosition called more than freq() times".into(),
            ));
        };
        let position = p.position;
        self.pos_upto = self.pos_upto.saturating_add(1);
        Ok(position)
    }

    fn start_offset(&self) -> i32 {
        self.current_position().map_or(-1, |p| p.start_offset)
    }

    fn end_offset(&self) -> i32 {
        self.current_position().map_or(-1, |p| p.end_offset)
    }

    fn payload(&self) -> Option<&[u8]> {
        self.current_position()
            .map(|p| p.payload.as_slice())
            .filter(|p| !p.is_empty())
    }
}

/// `SlowImpactsEnum`: impacts that claim nothing (`freq = Integer.MAX_VALUE`,
/// `norm = 1`) over any postings -- what `TermsEnum.impacts` returns for a
/// view that cannot know its real impacts (`MultiTermsEnum`, a filtered or
/// sorted view).
pub struct SlowImpactsEnum {
    postings: Box<dyn PostingsEnum>,
}

impl SlowImpactsEnum {
    pub fn new(postings: Box<dyn PostingsEnum>) -> Self {
        Self { postings }
    }
}

impl DocIdSetIterator for SlowImpactsEnum {
    fn doc_id(&self) -> i32 {
        self.postings.doc_id()
    }
    fn next_doc(&mut self) -> Result<i32> {
        self.postings.next_doc()
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        self.postings.advance(target)
    }
    fn cost(&self) -> i64 {
        self.postings.cost()
    }
}

impl PostingsEnum for SlowImpactsEnum {
    fn freq(&self) -> i32 {
        self.postings.freq()
    }
    fn next_position(&mut self) -> Result<i32> {
        self.postings.next_position()
    }
    fn start_offset(&self) -> i32 {
        self.postings.start_offset()
    }
    fn end_offset(&self) -> i32 {
        self.postings.end_offset()
    }
    fn payload(&self) -> Option<&[u8]> {
        self.postings.payload()
    }
}

impl ImpactsEnum for SlowImpactsEnum {
    fn advance_shallow(&mut self, _target: i32) -> Result<i32> {
        Ok(NO_MORE_DOCS)
    }
    fn impacts(&self) -> Vec<(i32, Vec<Impact>)> {
        vec![(
            NO_MORE_DOCS,
            vec![Impact {
                freq: i32::MAX,
                norm: 1,
            }],
        )]
    }
}

/// `TermsEnum`.
pub trait TermsEnum {
    /// `next()`: the next term, `None` past the last.
    fn next(&mut self) -> Result<Option<&[u8]>>;
    /// `term()`: the current term, `None` when unpositioned or exhausted.
    fn term(&self) -> Option<&[u8]>;
    /// `seekCeil(target)`.
    fn try_seek_ceil(&mut self, target: &[u8]) -> Result<SeekStatus>;
    /// `seekExact(target)`.
    fn try_seek_exact(&mut self, target: &[u8]) -> Result<bool> {
        Ok(self.try_seek_ceil(target)? == SeekStatus::Found)
    }
    /// `docFreq()` of the current term.
    fn doc_freq(&mut self) -> Result<i32>;
    /// `totalTermFreq()` of the current term.
    fn total_term_freq(&mut self) -> Result<i64>;
    /// `postings(null, flags)` of the current term.
    fn postings(&mut self, flags: PostingsFlags) -> Result<Box<dyn PostingsEnum>>;
    /// `impacts(flags)`: a [`SlowImpactsEnum`] unless the implementation
    /// knows the real impacts.
    fn impacts(&mut self, flags: PostingsFlags) -> Result<Box<dyn ImpactsEnum>> {
        Ok(Box::new(SlowImpactsEnum::new(self.postings(flags)?)))
    }
    /// `ord()`: unsupported unless the dictionary is ordinal-addressed.
    fn ord(&self) -> Result<i64> {
        Err(Error::Unsupported(
            "this TermsEnum does not support term ordinals".into(),
        ))
    }
    /// `seekExact(long ord)`.
    fn seek_exact_ord(&mut self, _ord: i64) -> Result<()> {
        Err(Error::Unsupported(
            "this TermsEnum does not support seekExact(ord)".into(),
        ))
    }
}

/// `Terms`: one field's term dictionary and its statistics.
pub trait Terms {
    /// `iterator()`.
    fn iterator(&self) -> Result<Box<dyn TermsEnum + '_>>;
    /// `intersect(compiled, startTerm)`: the terms `dfa` accepts, strictly
    /// after `start_term` when one is given. The default is Java's
    /// `Terms.intersect`: an [`filtered_terms_enum::AutomatonTermsEnum`] over
    /// [`Self::iterator`].
    fn intersect<'s>(
        &'s self,
        dfa: &'s lucene_codecs::automaton::ByteDfa,
        start_term: Option<&[u8]>,
    ) -> Result<Box<dyn TermsEnum + 's>> {
        Ok(Box::new(filtered_terms_enum::AutomatonTermsEnum::new(
            self.iterator()?,
            dfa,
            start_term,
        )))
    }
    /// `size()`: the number of terms, `-1` when unknown.
    // SENTINEL: `-1` = unknown, `Terms.size()`'s contract; compared only.
    fn size(&self) -> i64;
    /// `getSumTotalTermFreq()`.
    fn sum_total_term_freq(&self) -> i64;
    /// `getSumDocFreq()`.
    fn sum_doc_freq(&self) -> i64;
    /// `getDocCount()`.
    fn doc_count(&self) -> i32;
    /// `hasFreqs()`.
    fn has_freqs(&self) -> bool;
    /// `hasOffsets()`.
    fn has_offsets(&self) -> bool;
    /// `hasPositions()`.
    fn has_positions(&self) -> bool;
    /// `hasPayloads()`.
    fn has_payloads(&self) -> bool;
    /// `getMin()`: the smallest term, `None` for an empty field.
    fn min(&self) -> Result<Option<Vec<u8>>> {
        let mut te = self.iterator()?;
        Ok(te.next()?.map(<[u8]>::to_vec))
    }
    /// `getMax()`: the largest term, `None` for an empty field.
    fn max(&self) -> Result<Option<Vec<u8>>> {
        let mut te = self.iterator()?;
        let mut last = None;
        while let Some(t) = te.next()? {
            last = Some(t.to_vec());
        }
        Ok(last)
    }
}

/// `DocValuesIterator`: a [`DocIdSetIterator`] that can also be positioned
/// exactly on a document.
pub trait DocValuesIterator: DocIdSetIterator {
    /// `advanceExact(target)`: whether `target` has a value; the iterator is
    /// on `target` afterwards either way.
    fn advance_exact(&mut self, target: i32) -> Result<bool>;
}

/// `NumericDocValues`.
pub trait NumericDocValues: DocValuesIterator {
    /// `longValue()` of the current document.
    fn long_value(&self) -> i64;
}

/// `BinaryDocValues`.
pub trait BinaryDocValues: DocValuesIterator {
    /// `binaryValue()` of the current document.
    fn binary_value(&self) -> &[u8];
}

/// `SortedDocValues`.
pub trait SortedDocValues: DocValuesIterator {
    /// `ordValue()` of the current document.
    fn ord_value(&self) -> i32;
    /// `lookupOrd(ord)`.
    fn lookup_ord(&mut self, ord: i32) -> Result<Vec<u8>>;
    /// `getValueCount()`.
    fn value_count(&self) -> i32;
    /// `lookupTerm(key)`: `key`'s ordinal, or `-insertionPoint - 1`.
    // SENTINEL: negative = absent, `-insertionPoint-1` (`SortedDocValues.lookupTerm`).
    fn lookup_term(&mut self, key: &[u8]) -> Result<i32> {
        let (mut low, mut high) = (0i32, self.value_count().saturating_sub(1));
        while low <= high {
            let mid = low + (high - low) / 2;
            let term = self.lookup_ord(mid)?;
            match term.as_slice().cmp(key) {
                std::cmp::Ordering::Less => low = mid + 1,
                std::cmp::Ordering::Greater => high = mid - 1,
                std::cmp::Ordering::Equal => return Ok(mid),
            }
        }
        Ok(-(low + 1))
    }
}

/// `SortedNumericDocValues`.
pub trait SortedNumericDocValues: DocValuesIterator {
    /// `docValueCount()` of the current document.
    fn doc_value_count(&self) -> i32;
    /// `nextValue()`: the current document's values, ascending, one per call.
    fn next_value(&mut self) -> Result<i64>;
}

/// `SortedSetDocValues`.
pub trait SortedSetDocValues: DocValuesIterator {
    /// `docValueCount()` of the current document.
    fn doc_value_count(&self) -> i32;
    /// `nextOrd()`: the current document's ordinals, ascending, one per call.
    fn next_ord(&mut self) -> Result<i64>;
    /// `lookupOrd(ord)`.
    fn lookup_ord(&mut self, ord: i64) -> Result<Vec<u8>>;
    /// `getValueCount()`.
    fn value_count(&self) -> i64;
}

/// `PointValues` of one field.
pub trait PointValues {
    /// `intersect(visitor)`.
    fn intersect(&self, visitor: &mut dyn IntersectVisitor) -> Result<()>;
    /// `estimatePointCount(visitor)`.
    fn estimate_point_count(&self, visitor: &mut dyn IntersectVisitor) -> Result<i64>;
    /// `getMinPackedValue()`.
    fn min_packed_value(&self) -> &[u8];
    /// `getMaxPackedValue()`.
    fn max_packed_value(&self) -> &[u8];
    /// `getNumDimensions()`.
    fn num_dimensions(&self) -> i32;
    /// `getNumIndexDimensions()`.
    fn num_index_dimensions(&self) -> i32;
    /// `getBytesPerDimension()`.
    fn bytes_per_dimension(&self) -> i32;
    /// `size()`: the number of points.
    fn size(&self) -> i64;
    /// `getDocCount()`.
    fn doc_count(&self) -> i32;
}

/// `FloatVectorValues`: a field's vectors by ordinal, ordinals ascending
/// with their documents.
pub trait FloatVectorValues {
    /// `dimension()`.
    fn dimension(&self) -> usize;
    /// `size()`.
    fn size(&self) -> i32;
    /// `ordToDoc(ord)`.
    fn ord_to_doc(&self, ord: i32) -> Result<i32>;
    /// `vectorValue(ord)`.
    fn vector_value(&self, ord: i32) -> Result<Vec<f32>>;
}

/// `ByteVectorValues`.
pub trait ByteVectorValues {
    /// `dimension()`.
    fn dimension(&self) -> usize;
    /// `size()`.
    fn size(&self) -> i32;
    /// `ordToDoc(ord)`.
    fn ord_to_doc(&self, ord: i32) -> Result<i32>;
    /// `vectorValue(ord)`.
    fn vector_value(&self, ord: i32) -> Result<Vec<u8>>;
}

/// A `&mut dyn IntersectVisitor` as the sized visitor the codec's
/// generic `intersect` takes.
pub(crate) struct DynVisitor<'a>(pub(crate) &'a mut dyn IntersectVisitor);

impl IntersectVisitor for DynVisitor<'_> {
    fn compare(&mut self, min_packed: &[u8], max_packed: &[u8]) -> Relation {
        self.0.compare(min_packed, max_packed)
    }
    fn visit(&mut self, doc_id: i32) {
        self.0.visit(doc_id);
    }
    fn visit_many(&mut self, doc_ids: &[i32]) {
        self.0.visit_many(doc_ids);
    }
    fn visit_with_value(&mut self, doc_id: i32, packed_value: &[u8]) {
        self.0.visit_with_value(doc_id, packed_value);
    }
    fn grow(&mut self, count: usize) {
        self.0.grow(count);
    }
}

// ---------------------------------------------------------------------------
// Cache helpers and closed listeners
// ---------------------------------------------------------------------------

/// `IndexReader.CacheKey`: identity of a reader (or of a segment core), equal
/// only to itself.
#[derive(Clone)]
pub struct CacheKey(Arc<ClosedState>);

impl CacheKey {
    fn id(&self) -> usize {
        Arc::as_ptr(&self.0) as usize
    }
}

impl PartialEq for CacheKey {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for CacheKey {}

impl std::hash::Hash for CacheKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id().hash(state);
    }
}

impl std::fmt::Debug for CacheKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CacheKey({:#x})", self.id())
    }
}

/// `IndexReader.ClosedListener`: told the id of the key being closed.
pub type ClosedListener = Box<dyn FnOnce(usize) + Send>;

#[derive(Default)]
struct ClosedState {
    listeners: Mutex<Vec<ClosedListener>>,
}

impl Drop for ClosedState {
    fn drop(&mut self) {
        let id = self as *const Self as usize;
        let listeners = std::mem::take(self.listeners.get_mut().unwrap_or_else(|e| e.into_inner()));
        for l in listeners {
            l(id);
        }
    }
}

/// `IndexReader.CacheHelper`: a cache key plus the listeners told when the
/// last handle sharing it goes away.
///
/// The key a listener hears is [`CacheHelper::key_id`]'s number: the key
/// itself is gone by then (holding it would keep the reader alive).
pub struct CacheHelper {
    /// Shared by every clone of the reader it describes: the last one
    /// dropped runs the listeners.
    state: Arc<ClosedState>,
}

impl Default for CacheHelper {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for CacheHelper {
    /// A clone shares the key: a cloned segment reader is the same reader.
    fn clone(&self) -> Self {
        Self {
            state: Arc::clone(&self.state),
        }
    }
}

impl std::fmt::Debug for CacheHelper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CacheHelper({:#x})", self.key_id())
    }
}

impl CacheHelper {
    /// A fresh key.
    pub fn new() -> Self {
        Self {
            state: Arc::default(),
        }
    }

    /// `getKey()`.
    pub fn key(&self) -> CacheKey {
        CacheKey(Arc::clone(&self.state))
    }

    /// The key's identity, as listeners hear it.
    pub fn key_id(&self) -> usize {
        Arc::as_ptr(&self.state) as usize
    }

    /// `addClosedListener(listener)`: runs once, when the last handle sharing
    /// this key is dropped. Holding a [`CacheKey`] counts as a handle.
    pub fn add_closed_listener(&self, listener: ClosedListener) {
        self.state
            .listeners
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(listener);
    }
}

// ---------------------------------------------------------------------------
// Readers
// ---------------------------------------------------------------------------

/// `LeafReaderContext`: a leaf, its position among a top-level reader's
/// leaves and its first document there.
#[derive(Clone, Copy)]
pub struct LeafReaderContext<'a> {
    pub reader: &'a dyn LeafReader,
    /// `ord`: the leaf's index in `leaves()`.
    pub ord: usize,
    /// `docBase`.
    pub doc_base: i32,
}

/// The methods Java's `IndexReader` gives every reader, leaf or composite.
pub trait IndexReader: Send + Sync {
    /// `maxDoc()`.
    fn max_doc(&self) -> i32;
    /// `numDocs()`.
    fn num_docs(&self) -> i32;
    /// `leaves()`: every leaf, in document order, with its doc base.
    fn leaves(&self) -> Vec<LeafReaderContext<'_>>;
    /// `getReaderCacheHelper()`.
    fn reader_cache_helper(&self) -> Option<&CacheHelper>;

    /// `hasDeletions()`.
    fn has_deletions(&self) -> bool {
        self.num_deleted_docs() > 0
    }
    /// `numDeletedDocs()`.
    fn num_deleted_docs(&self) -> i32 {
        self.max_doc() - self.num_docs()
    }
    /// `docFreq(term)`: summed over the leaves (deleted documents included).
    fn doc_freq(&self, field: &str, term: &[u8]) -> Result<i64> {
        let mut total = 0i64;
        for leaf in self.leaves() {
            if let Some(terms) = leaf.reader.terms(field)? {
                let mut te = terms.iterator()?;
                if te.try_seek_exact(term)? {
                    total = total.saturating_add(i64::from(te.doc_freq()?));
                }
            }
        }
        Ok(total)
    }
    /// `totalTermFreq(term)`.
    fn total_term_freq(&self, field: &str, term: &[u8]) -> Result<i64> {
        let mut total = 0i64;
        for leaf in self.leaves() {
            if let Some(terms) = leaf.reader.terms(field)? {
                let mut te = terms.iterator()?;
                if te.try_seek_exact(term)? {
                    total = total.saturating_add(te.total_term_freq()?);
                }
            }
        }
        Ok(total)
    }
    /// `getSumDocFreq(field)`.
    fn sum_doc_freq(&self, field: &str) -> Result<i64> {
        self.leaves().iter().try_fold(0i64, |acc, l| {
            Ok(acc.saturating_add(l.reader.terms(field)?.map_or(0, |t| t.sum_doc_freq())))
        })
    }
    /// `getDocCount(field)`.
    fn doc_count(&self, field: &str) -> Result<i32> {
        self.leaves().iter().try_fold(0i32, |acc, l| {
            Ok(acc.saturating_add(l.reader.terms(field)?.map_or(0, |t| t.doc_count())))
        })
    }
    /// `getSumTotalTermFreq(field)`.
    fn sum_total_term_freq(&self, field: &str) -> Result<i64> {
        self.leaves().iter().try_fold(0i64, |acc, l| {
            Ok(acc.saturating_add(
                l.reader
                    .terms(field)?
                    .map_or(0, |t| t.sum_total_term_freq()),
            ))
        })
    }
    /// `storedFields().document(docID, visitor)` in top-level doc ids.
    fn stored_document(&self, doc: i32, visitor: &mut dyn StoredFieldVisitor) -> Result<()> {
        let (leaf, local) = leaf_for_doc(&self.leaves(), doc)?;
        leaf.reader.document(local, visitor)
    }
    /// `termVectors().get(docID)` in top-level doc ids.
    fn document_term_vectors(&self, doc: i32) -> Result<Option<TermVectorsDocument>> {
        let (leaf, local) = leaf_for_doc(&self.leaves(), doc)?;
        leaf.reader.term_vectors(local)
    }
}

/// `ReaderUtil.subIndex`: the leaf holding top-level `doc`, and `doc` in its
/// space.
///
/// # Errors
/// [`Error::IllegalArgument`] for a document outside every leaf.
pub fn leaf_for_doc<'a>(
    leaves: &[LeafReaderContext<'a>],
    doc: i32,
) -> Result<(LeafReaderContext<'a>, i32)> {
    let i = leaves.partition_point(|l| l.doc_base <= doc);
    let leaf = i
        .checked_sub(1)
        .and_then(|i| leaves.get(i))
        .filter(|l| doc - l.doc_base < l.reader.max_doc())
        .ok_or_else(|| Error::IllegalArgument(format!("docID {doc} is out of bounds")))?;
    Ok((*leaf, doc - leaf.doc_base))
}

/// `LeafReader`: one segment, or a view that reads like one.
///
/// Every accessor is by field name and returns `None` for a field the leaf
/// does not have in that form, as Java returns `null`.
pub trait LeafReader: IndexReader {
    /// `getFieldInfos()`.
    fn field_infos(&self) -> &FieldInfos;
    /// `getLiveDocs()`: `None` when no document is deleted.
    fn live_docs(&self) -> Option<&FixedBitSet>;
    /// `terms(field)`.
    fn terms(&self, field: &str) -> Result<Option<Box<dyn Terms + '_>>>;
    /// `getNumericDocValues(field)`.
    fn numeric_doc_values(&self, field: &str) -> Result<Option<Box<dyn NumericDocValues + '_>>>;
    /// `getBinaryDocValues(field)`.
    fn binary_doc_values(&self, field: &str) -> Result<Option<Box<dyn BinaryDocValues + '_>>>;
    /// `getSortedDocValues(field)`.
    fn sorted_doc_values(&self, field: &str) -> Result<Option<Box<dyn SortedDocValues + '_>>>;
    /// `getSortedNumericDocValues(field)`.
    fn sorted_numeric_doc_values(
        &self,
        field: &str,
    ) -> Result<Option<Box<dyn SortedNumericDocValues + '_>>>;
    /// `getSortedSetDocValues(field)`.
    fn sorted_set_doc_values(
        &self,
        field: &str,
    ) -> Result<Option<Box<dyn SortedSetDocValues + '_>>>;
    /// `getNormValues(field)`.
    fn norm_values(&self, field: &str) -> Result<Option<Box<dyn NumericDocValues + '_>>>;
    /// `getPointValues(field)`.
    fn point_values(&self, field: &str) -> Result<Option<Box<dyn PointValues + '_>>>;
    /// `getFloatVectorValues(field)`.
    fn float_vector_values(&self, field: &str) -> Result<Option<Box<dyn FloatVectorValues + '_>>>;
    /// `getByteVectorValues(field)`.
    fn byte_vector_values(&self, field: &str) -> Result<Option<Box<dyn ByteVectorValues + '_>>>;
    /// `storedFields().document(docID, visitor)`: nothing is visited when the
    /// leaf stores no fields.
    fn document(&self, doc: i32, visitor: &mut dyn StoredFieldVisitor) -> Result<()>;
    /// `termVectors().get(docID)`: `None` when the document has none.
    fn term_vectors(&self, doc: i32) -> Result<Option<TermVectorsDocument>>;
    /// `getMetaData().sort()`: the index sort, `None` for an unsorted leaf.
    fn index_sort(&self) -> Option<&[IndexSortField]>;
    /// `getCoreCacheHelper()`: `None` when the leaf cannot be cached per
    /// core (a view whose content differs from its inner reader's).
    fn core_cache_helper(&self) -> Option<&CacheHelper>;

    /// `getDocValuesSkipper(field)`: the field's doc-values skip index,
    /// `None` when it has none -- and, by default, for a view, as Java's
    /// sorting and composite views answer.
    fn doc_values_skipper(
        &self,
        _field: &str,
    ) -> Result<Option<lucene_codecs::doc_values::DocValuesSkipper<'_>>> {
        Ok(None)
    }

    /// `checkIntegrity()`: verifies the checksums of the files behind this
    /// leaf. A view with no files of its own checks its inner readers; the
    /// default, for a leaf holding none, has nothing to check.
    fn check_integrity(&self) -> Result<()> {
        Ok(())
    }

    /// `LeafReader.postings(term, flags)`: `None` when the field or term is
    /// absent.
    fn postings(
        &self,
        field: &str,
        term: &[u8],
        flags: PostingsFlags,
    ) -> Result<Option<Box<dyn PostingsEnum>>> {
        let Some(terms) = self.terms(field)? else {
            return Ok(None);
        };
        let mut te = terms.iterator()?;
        if !te.try_seek_exact(term)? {
            return Ok(None);
        }
        te.postings(flags).map(Some)
    }
}

/// `CodecReader`: a leaf that reads straight through a codec's producers --
/// or a view re-presenting them (`FilterCodecReader`, `SortingCodecReader`,
/// the slow wrappers) -- which is what a merge consumes (`addIndexes`,
/// `OneMerge.wrapForMerge`).
///
/// It adds no methods: in Java the producers (`getPostingsReader`, ...) are
/// the `LeafReader` methods' implementation, and here the per-format readers
/// are `lucene_codecs` types a [`crate::directory_reader::SegmentReader`]
/// holds directly. The trait is the type-level promise a merge input needs.
pub trait CodecReader: LeafReader {}

/// `numDocs()` of a leaf whose deletions are `live_docs`.
pub fn num_docs_of(max_doc: i32, live_docs: Option<&FixedBitSet>) -> i32 {
    live_docs.map_or(max_doc, |bits| bits.cardinality() as i32)
}

/// Implements [`IndexReader`] for a leaf type: one leaf at doc base 0,
/// `numDocs` from its live docs.
#[macro_export]
#[doc(hidden)]
macro_rules! impl_leaf_index_reader {
    ($ty:ty, |$s:ident| max_doc: $max:expr, reader_cache_helper: $helper:expr) => {
        impl $crate::reader::IndexReader for $ty {
            fn max_doc(&self) -> i32 {
                let $s = self;
                $max
            }
            fn num_docs(&self) -> i32 {
                $crate::reader::num_docs_of(
                    $crate::reader::IndexReader::max_doc(self),
                    $crate::reader::LeafReader::live_docs(self),
                )
            }
            fn leaves(&self) -> Vec<$crate::reader::LeafReaderContext<'_>> {
                vec![$crate::reader::LeafReaderContext {
                    reader: self,
                    ord: 0,
                    doc_base: 0,
                }]
            }
            fn reader_cache_helper(&self) -> Option<&$crate::reader::CacheHelper> {
                let $s = self;
                $helper
            }
        }
    };
}

/// One child of a [`CompositeReader`].
#[derive(Clone, Copy)]
pub enum SubReader<'a> {
    Leaf(&'a dyn LeafReader),
    Composite(&'a dyn CompositeReader),
}

impl SubReader<'_> {
    fn max_doc(&self) -> i32 {
        match self {
            SubReader::Leaf(l) => l.max_doc(),
            SubReader::Composite(c) => c.max_doc(),
        }
    }
}

/// An owned child handle, what a composite built by the caller holds.
#[derive(Clone)]
pub enum ReaderHandle {
    Leaf(Arc<dyn LeafReader>),
    Composite(Arc<dyn CompositeReader>),
}

impl ReaderHandle {
    /// The borrowed form.
    pub fn as_sub(&self) -> SubReader<'_> {
        match self {
            ReaderHandle::Leaf(l) => SubReader::Leaf(l.as_ref()),
            ReaderHandle::Composite(c) => SubReader::Composite(c.as_ref()),
        }
    }

    /// `maxDoc()`.
    pub fn max_doc(&self) -> i32 {
        self.as_sub().max_doc()
    }

    /// `numDocs()`.
    pub fn num_docs(&self) -> i32 {
        match self {
            ReaderHandle::Leaf(l) => l.num_docs(),
            ReaderHandle::Composite(c) => c.num_docs(),
        }
    }
}

/// `CompositeReader`: a reader made of other readers.
pub trait CompositeReader: IndexReader {
    /// `getSequentialSubReaders()`.
    fn sequential_sub_readers(&self) -> Vec<SubReader<'_>>;
    /// Every leaf as an owned handle, in `leaves()` order -- what a view
    /// that outlives this borrow (a [`parallel::ParallelCompositeReader`])
    /// holds on to.
    fn leaf_handles(&self) -> Vec<Arc<dyn LeafReader>>;
}

/// `CompositeReaderContext`'s leaf walk: every leaf under `subs`, depth
/// first, each with its doc base.
pub fn composite_leaves<'a>(subs: &[SubReader<'a>]) -> Vec<LeafReaderContext<'a>> {
    fn walk<'a>(subs: &[SubReader<'a>], base: &mut i32, out: &mut Vec<LeafReaderContext<'a>>) {
        for sub in subs {
            match *sub {
                SubReader::Leaf(reader) => {
                    out.push(LeafReaderContext {
                        reader,
                        ord: out.len(),
                        doc_base: *base,
                    });
                    *base = base.saturating_add(reader.max_doc());
                }
                SubReader::Composite(c) => walk(&c.sequential_sub_readers(), base, out),
            }
        }
    }
    let mut out = Vec::new();
    let mut base = 0;
    walk(subs, &mut base, &mut out);
    out
}

/// `FieldInfos.getMergedFieldInfos(reader)`: every field of every leaf, each
/// once, numbered by `FieldInfos.Builder` (a field keeps its number unless an
/// earlier field took it).
pub fn merged_field_infos(leaves: &[LeafReaderContext<'_>]) -> FieldInfos {
    let mut builder = FieldInfosBuilder::default();
    for leaf in leaves {
        for fi in &leaf.reader.field_infos().fields {
            builder.add(fi);
        }
    }
    builder.finish()
}

/// `FieldInfos.Builder` over `FieldInfos.FieldNumbers`: the numbering half,
/// which is all a view that joins leaves needs.
#[derive(Default)]
pub(crate) struct FieldInfosBuilder {
    by_name: HashMap<String, usize>,
    taken: std::collections::HashSet<i32>,
    /// `lowestUnassignedFieldNumber`, `-1` before the first assignment.
    lowest_unassigned: i32,
    started: bool,
    fields: Vec<lucene_codecs::field_infos::FieldInfo>,
}

impl FieldInfosBuilder {
    /// `add(fieldInfo)`: a field seen before keeps its first entry.
    pub(crate) fn add(&mut self, fi: &lucene_codecs::field_infos::FieldInfo) {
        if self.by_name.contains_key(&fi.name) {
            return;
        }
        if !self.started {
            self.started = true;
            self.lowest_unassigned = -1;
        }
        // `FieldNumbers.addOrGet`: the preferred number when free, else the
        // lowest unassigned one.
        let number = if fi.number >= 0 && !self.taken.contains(&fi.number) {
            fi.number
        } else {
            loop {
                self.lowest_unassigned += 1;
                if !self.taken.contains(&self.lowest_unassigned) {
                    break self.lowest_unassigned;
                }
            }
        };
        self.taken.insert(number);
        let mut merged = fi.clone();
        merged.number = number;
        self.by_name.insert(fi.name.clone(), self.fields.len());
        self.fields.push(merged);
    }

    /// `finish()`: the fields in number order.
    pub(crate) fn finish(mut self) -> FieldInfos {
        self.fields.sort_by_key(|f| f.number);
        FieldInfos {
            fields: self.fields,
        }
    }
}

/// A [`StoredFieldVisitor`] that hands `inner` field numbers renumbered
/// from one [`FieldInfos`] to another by name -- Java's `remap(FieldInfo)`.
pub(crate) struct RemappingVisitor<'a> {
    pub(crate) inner: &'a mut dyn StoredFieldVisitor,
    pub(crate) from: &'a FieldInfos,
    pub(crate) to: &'a FieldInfos,
}

impl RemappingVisitor<'_> {
    fn map(&self, number: i32) -> i32 {
        self.from
            .field_by_number(number)
            .and_then(|fi| self.to.field_by_name(&fi.name))
            .map_or(number, |fi| fi.number)
    }
}

type StoredResult<T> = lucene_codecs::stored_fields::Result<T>;

impl StoredFieldVisitor for RemappingVisitor<'_> {
    fn needs_field(&mut self, field_number: i32) -> StoredResult<VisitStatus> {
        let n = self.map(field_number);
        self.inner.needs_field(n)
    }
    fn string_field(&mut self, field_number: i32, value: &str) -> StoredResult<()> {
        let n = self.map(field_number);
        self.inner.string_field(n, value)
    }
    fn binary_field(&mut self, field_number: i32, value: &[u8]) -> StoredResult<()> {
        let n = self.map(field_number);
        self.inner.binary_field(n, value)
    }
    fn int_field(&mut self, field_number: i32, value: i32) -> StoredResult<()> {
        let n = self.map(field_number);
        self.inner.int_field(n, value)
    }
    fn long_field(&mut self, field_number: i32, value: i64) -> StoredResult<()> {
        let n = self.map(field_number);
        self.inner.long_field(n, value)
    }
    fn float_field(&mut self, field_number: i32, value: f32) -> StoredResult<()> {
        let n = self.map(field_number);
        self.inner.float_field(n, value)
    }
    fn double_field(&mut self, field_number: i32, value: f64) -> StoredResult<()> {
        let n = self.map(field_number);
        self.inner.double_field(n, value)
    }
}

/// Renumbers a term-vectors document's fields from `from` to `to` by name.
pub(crate) fn remap_term_vectors(
    mut doc: TermVectorsDocument,
    from: &FieldInfos,
    to: &FieldInfos,
) -> TermVectorsDocument {
    for field in &mut doc.fields {
        if let Some(n) = from
            .field_by_number(field.field_number)
            .and_then(|fi| to.field_by_name(&fi.name))
        {
            field.field_number = n.number;
        }
    }
    doc
}

#[cfg(test)]
mod tests;
