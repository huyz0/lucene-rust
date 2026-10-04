//! The reader API over one segment: [`SegmentReader`] as a [`LeafReader`]
//! and [`CodecReader`], and [`DirectoryReader`] as a [`CompositeReader`].
//!
//! Every access object reads through the codec readers the segment already
//! holds -- `blocktree` for terms and postings, the doc-values column readers
//! (`NumericReader`, `BinaryReader`, `SortedNumericReader`, the terms
//! dictionary), `NormsReader`, the BKD `PointsReader`, stored fields, term
//! vectors and flat vectors. No format is decoded here that the codec crate
//! does not already decode.

use std::sync::Arc;

use lucene_codecs::blocktree::{self, FieldTerms, SeekStatus};
use lucene_codecs::doc_values::{
    self, BinaryReader, NumericEntry, NumericReader, SortedNumericReader, SortedSetKind,
};
use lucene_codecs::field_infos::{DocValuesType, FieldInfo, FieldInfos, IndexOptions};
use lucene_codecs::postings::{DocInput, PayInput, PosInput};
use lucene_codecs::terms_dict::TermsDict;
use lucene_index::segment_info::IndexSortField;
use lucene_util::fixed_bit_set::FixedBitSet;

use super::{
    BinaryDocValues, ByteVectorValues, CacheHelper, CodecReader, CompositeReader, DocIdSetIterator,
    DocValuesIterator, DynVisitor, FloatVectorValues, IndexReader, IntersectVisitor, LeafReader,
    LeafReaderContext, MaterializedPostings, NumericDocValues, PointValues, PostingsEnum,
    PostingsFlags, SortedDocValues, SortedNumericDocValues, SortedSetDocValues, StoredFieldVisitor,
    SubReader, TermVectorsDocument, Terms, TermsEnum, NO_MORE_DOCS,
};
use crate::directory_reader::{DirectoryReader, SegmentReader};
use crate::{Error, Result};

// ---------------------------------------------------------------------------
// Documents with a value
// ---------------------------------------------------------------------------

/// The documents of a doc-values (or norms) field that have a value: the
/// `IndexedDISI` a sparse field records, decoded once, or the whole segment.
#[derive(Debug, Clone)]
pub(crate) enum DocSet {
    Empty,
    /// Every document below this bound.
    Dense(i32),
    Sparse(Arc<SparseDocs>),
}

/// A sparse field's documents, ascending, with a rank index over them: a
/// bit per document of the segment and, per 64-bit word, how many documents
/// come before it -- so a document's position among them (what
/// `IndexedDISI`'s rank tables give Java) is a lookup, not a search.
#[derive(Debug)]
pub(crate) struct SparseDocs {
    docs: Vec<i32>,
    words: Vec<u64>,
    ranks: Vec<u32>,
}

impl SparseDocs {
    /// The rank index over `docs`, a set within a segment of `max_doc`
    /// documents. Everything is sized from `max_doc`, never from a document:
    /// the documents come off disk, and one near `i32::MAX` in a ten-document
    /// segment would otherwise size a ~400 MB bit set. A document outside
    /// `0..max_doc`, or not strictly above the one before it, is corruption.
    pub(crate) fn new(docs: Vec<i32>, max_doc: i32) -> Result<Self> {
        let mut last = -1i32;
        for &d in &docs {
            if d <= last || d >= max_doc {
                return Err(lucene_store::Error::Corrupted(format!(
                    "docs-with-field document {d} after {last} in a segment of {max_doc}: \
                     out of order or past the segment"
                ))
                .into());
            }
            last = d;
        }
        // ALLOC: sized from the segment, which every document was checked
        // against above.
        let mut words = vec![0u64; usize::try_from(max_doc).unwrap_or(0).div_ceil(64)];
        for &d in &docs {
            // Checked in `0..max_doc` above, so every word exists.
            if let Some(w) = usize::try_from(d).ok().and_then(|d| words.get_mut(d >> 6)) {
                *w |= 1u64 << (d & 63);
            }
        }
        let mut ranks = Vec::with_capacity(words.len());
        let mut before = 0u32;
        for w in &words {
            ranks.push(before);
            before = before.saturating_add(w.count_ones());
        }
        Ok(Self { docs, words, ranks })
    }

    fn len(&self) -> usize {
        self.docs.len()
    }

    fn get(&self, i: usize) -> Option<i32> {
        self.docs.get(i).copied()
    }

    /// How many documents are below `target`, and whether `target` is one.
    fn rank(&self, target: i32) -> (usize, bool) {
        let Ok(t) = usize::try_from(target) else {
            return (0, false);
        };
        let w = t >> 6;
        match (self.words.get(w), self.ranks.get(w)) {
            (Some(&word), Some(&before)) => {
                let bit = 1u64 << (t & 63);
                let below = (word & (bit - 1)).count_ones();
                let rank = usize::try_from(before.saturating_add(below)).unwrap_or(usize::MAX);
                (rank, word & bit != 0)
            }
            _ => (self.docs.len(), false),
        }
    }
}

impl DocSet {
    /// `docsWithFieldOffset`'s three shapes: `-2` empty, `-1` dense, else the
    /// `IndexedDISI` at `[offset, offset + length)` of `data`.
    pub(crate) fn read(
        data: &[u8],
        offset: i64,
        length: i64,
        dense_rank_power: u8,
        max_doc: i32,
    ) -> Result<Self> {
        match offset {
            -2 => Ok(DocSet::Empty),
            -1 => Ok(DocSet::Dense(max_doc)),
            _ => {
                let region = usize::try_from(offset)
                    .ok()
                    .zip(usize::try_from(length).ok())
                    .and_then(|(o, l)| data.get(o..o.checked_add(l)?))
                    .ok_or_else(|| {
                        lucene_store::Error::Corrupted(format!(
                            "docs-with-field region {offset}+{length} outside its file"
                        ))
                    })?;
                // Bounded by the segment as it decodes: a corrupt region
                // cannot grow the list past `max_doc` entries.
                let docs = lucene_codecs::indexed_disi::decode_doc_ids_below(
                    region,
                    dense_rank_power,
                    max_doc,
                )?;
                Ok(DocSet::Sparse(Arc::new(SparseDocs::new(docs, max_doc)?)))
            }
        }
    }

    /// [`Self::read`] through `reader`'s cache of decoded sets: a sparse
    /// field's documents are decoded once per reader, not once per iterator
    /// (stage 3, M10 T10.4: the grouping collectors open a field's values in
    /// every segment for every selector, and decoding them was a tenth of a
    /// grouping search).
    pub(crate) fn cached(
        reader: &SegmentReader,
        data: &[u8],
        offset: i64,
        length: i64,
        dense_rank_power: u8,
    ) -> Result<Self> {
        if offset < 0 {
            return Self::read(data, offset, length, dense_rank_power, reader.max_doc);
        }
        let key = (data.as_ptr() as usize, offset);
        let cache = &reader.docs_with_field;
        let lock = || {
            cache
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
        };
        if let Some(docs) = lock().get(&key) {
            return Ok(DocSet::Sparse(Arc::clone(docs)));
        }
        let set = Self::read(data, offset, length, dense_rank_power, reader.max_doc)?;
        if let DocSet::Sparse(docs) = &set {
            lock().insert(key, Arc::clone(docs));
        }
        Ok(set)
    }

    fn of_numeric(reader: &SegmentReader, data: &[u8], e: &NumericEntry) -> Result<Self> {
        Self::cached(
            reader,
            data,
            e.docs_with_field_offset,
            e.docs_with_field_length,
            e.dense_rank_power,
        )
    }

    fn len(&self) -> i64 {
        match self {
            DocSet::Empty => 0,
            DocSet::Dense(n) => i64::from(*n),
            DocSet::Sparse(v) => v.len() as i64,
        }
    }
}

/// A forward cursor over a [`DocSet`]: the `DocIdSetIterator` half every
/// segment doc-values iterator shares.
#[derive(Debug, Clone)]
pub(crate) struct DvCursor {
    docs: DocSet,
    /// Sparse: the index of the next entry past the current document.
    next: usize,
    doc: i32,
}

impl DvCursor {
    pub(crate) fn new(docs: DocSet) -> Self {
        Self {
            docs,
            next: 0,
            doc: -1,
        }
    }

    pub(crate) fn next_doc(&mut self) -> i32 {
        self.doc = match &self.docs {
            DocSet::Empty => NO_MORE_DOCS,
            DocSet::Dense(n) => {
                let d = self.doc.saturating_add(1);
                if d < *n {
                    d
                } else {
                    NO_MORE_DOCS
                }
            }
            DocSet::Sparse(v) => match v.get(self.next) {
                Some(d) => {
                    self.next += 1;
                    d
                }
                None => NO_MORE_DOCS,
            },
        };
        self.doc
    }

    pub(crate) fn advance(&mut self, target: i32) -> i32 {
        self.doc = match &self.docs {
            DocSet::Empty => NO_MORE_DOCS,
            DocSet::Dense(n) => {
                if target < *n {
                    target
                } else {
                    NO_MORE_DOCS
                }
            }
            DocSet::Sparse(v) => {
                // Not behind the current position (the iterator contract).
                let at = v.rank(target).0.max(self.next.min(v.len()));
                match v.get(at) {
                    Some(d) => {
                        self.next = at + 1;
                        d
                    }
                    None => {
                        self.next = v.len();
                        NO_MORE_DOCS
                    }
                }
            }
        };
        self.doc
    }

    /// The current document's position among a sparse field's documents
    /// (the `IndexedDISI` rank the cursor already found); `None` for a
    /// dense field, whose readers index by document.
    fn sparse_index(&self) -> Option<i64> {
        match &self.docs {
            DocSet::Sparse(_) => i64::try_from(self.next.checked_sub(1)?).ok(),
            _ => None,
        }
    }

    pub(crate) fn advance_exact(&mut self, target: i32) -> bool {
        self.doc = target;
        match &self.docs {
            DocSet::Empty => false,
            DocSet::Dense(n) => target < *n,
            DocSet::Sparse(v) => {
                let (at, found) = v.rank(target);
                self.next = if found { at + 1 } else { at };
                found
            }
        }
    }
}

/// Implements `DocIdSetIterator` and `DocValuesIterator` for a struct with a
/// `cur: DvCursor`, a `load(&mut self) -> Result<()>` that reads the
/// value of `cur.doc` and a `miss(&mut self)` that forgets it (a document
/// `advanceExact` found without a value).
macro_rules! dv_iterator {
    ($ty:ident) => {
        impl DocIdSetIterator for $ty<'_> {
            fn doc_id(&self) -> i32 {
                self.cur.doc
            }
            fn next_doc(&mut self) -> Result<i32> {
                let d = self.cur.next_doc();
                if d != NO_MORE_DOCS {
                    self.load()?;
                }
                Ok(d)
            }
            fn advance(&mut self, target: i32) -> Result<i32> {
                let d = self.cur.advance(target);
                if d != NO_MORE_DOCS {
                    self.load()?;
                }
                Ok(d)
            }
            fn cost(&self) -> i64 {
                self.cur.docs.len()
            }
        }
        impl DocValuesIterator for $ty<'_> {
            fn advance_exact(&mut self, target: i32) -> Result<bool> {
                let found = self.cur.advance_exact(target);
                if found {
                    self.load()?;
                } else {
                    self.miss();
                }
                Ok(found)
            }
        }
    };
}

struct SegNumeric<'a> {
    cur: DvCursor,
    reader: NumericReader<'a>,
    value: i64,
}

impl SegNumeric<'_> {
    fn miss(&mut self) {
        self.value = 0;
    }

    fn load(&mut self) -> Result<()> {
        self.value = match self.cur.sparse_index() {
            Some(i) => self.reader.value_at_index(i)?,
            None => self.reader.value(self.cur.doc)?.unwrap_or(0),
        };
        Ok(())
    }
}
dv_iterator!(SegNumeric);

impl NumericDocValues for SegNumeric<'_> {
    fn long_value(&self) -> i64 {
        self.value
    }
}

struct SegNorms<'a> {
    cur: DvCursor,
    reader: lucene_codecs::norms::NormsReader<'a>,
    value: i64,
}

impl SegNorms<'_> {
    fn miss(&mut self) {
        self.value = 0;
    }

    fn load(&mut self) -> Result<()> {
        self.value = self.reader.value(self.cur.doc)?.unwrap_or(0);
        Ok(())
    }
}
dv_iterator!(SegNorms);

impl NumericDocValues for SegNorms<'_> {
    fn long_value(&self) -> i64 {
        self.value
    }
}

struct SegBinary<'a> {
    cur: DvCursor,
    reader: BinaryReader<'a>,
    value: &'a [u8],
}

impl SegBinary<'_> {
    fn miss(&mut self) {
        self.value = &[];
    }

    fn load(&mut self) -> Result<()> {
        self.value = self.reader.value(self.cur.doc)?.unwrap_or_default();
        Ok(())
    }
}
dv_iterator!(SegBinary);

impl BinaryDocValues for SegBinary<'_> {
    fn binary_value(&self) -> &[u8] {
        self.value
    }
}

struct SegSorted<'a> {
    cur: DvCursor,
    ords: NumericReader<'a>,
    terms: TermsDict<'a>,
    ord: i32,
}

impl SegSorted<'_> {
    fn miss(&mut self) {
        self.ord = -1;
    }

    fn load(&mut self) -> Result<()> {
        let ord = match self.cur.sparse_index() {
            Some(i) => self.ords.value_at_index(i)?,
            None => self.ords.value(self.cur.doc)?.unwrap_or(0),
        };
        self.ord = i32::try_from(ord)
            .map_err(|_| lucene_store::Error::Corrupted(format!("sorted ordinal {ord}")))?;
        Ok(())
    }
}
dv_iterator!(SegSorted);

impl SortedDocValues for SegSorted<'_> {
    fn ord_value(&self) -> i32 {
        self.ord
    }
    fn lookup_ord(&mut self, ord: i32) -> Result<Vec<u8>> {
        Ok(self.terms.seek_ord(i64::from(ord))?.to_vec())
    }
    fn value_count(&self) -> i32 {
        i32::try_from(self.terms.size()).unwrap_or(i32::MAX)
    }
    // SENTINEL: negative = absent, `-insertionPoint-1` (`SortedDocValues.lookupTerm`).
    fn lookup_term(&mut self, key: &[u8]) -> Result<i32> {
        let ord = dict_lookup_term(&mut self.terms, key)?;
        Ok(i32::try_from(ord).unwrap_or(i32::MIN))
    }
}

/// `Lucene90DocValuesProducer`'s `lookupTerm(key)`: `TermsDict.seekCeil`
/// (the terms index, then one block) -- the ordinal when found, else
/// `-ord - 1` with `ord` the insertion point (the value count past the end).
// SENTINEL: negative = absent, `-insertionPoint-1`.
fn dict_lookup_term(terms: &mut TermsDict<'_>, key: &[u8]) -> Result<i64> {
    use lucene_codecs::terms_dict::SeekStatus;
    // `TermsDict::seek_ceil` spelled out: the doc-values dictionary's own,
    // already fallible -- not the block tree's infallible one.
    Ok(match TermsDict::seek_ceil(terms, key)? {
        SeekStatus::Found => terms.ord(),
        SeekStatus::NotFound => terms.ord().saturating_neg().saturating_sub(1),
        SeekStatus::End => terms.size().saturating_neg().saturating_sub(1),
    })
}

struct SegSortedNumeric<'a> {
    cur: DvCursor,
    reader: SortedNumericReader<'a>,
    values: Vec<i64>,
    upto: usize,
}

impl SegSortedNumeric<'_> {
    fn miss(&mut self) {
        self.values.clear();
    }

    fn load(&mut self) -> Result<()> {
        self.reader.values(self.cur.doc, &mut self.values)?;
        self.upto = 0;
        Ok(())
    }
}
dv_iterator!(SegSortedNumeric);

impl SortedNumericDocValues for SegSortedNumeric<'_> {
    fn doc_value_count(&self) -> i32 {
        self.values.len() as i32
    }
    fn next_value(&mut self) -> Result<i64> {
        let v = self.values.get(self.upto).copied().ok_or_else(|| {
            Error::IllegalState("nextValue called more than docValueCount times".into())
        })?;
        self.upto += 1;
        Ok(v)
    }
}

/// A SORTED_SET field, in either on-disk shape.
enum SetOrds<'a> {
    Single(NumericReader<'a>),
    Multi(SortedNumericReader<'a>),
}

struct SegSortedSet<'a> {
    cur: DvCursor,
    ords: SetOrds<'a>,
    terms: TermsDict<'a>,
    values: Vec<i64>,
    upto: usize,
}

impl SegSortedSet<'_> {
    fn miss(&mut self) {
        self.values.clear();
    }

    fn load(&mut self) -> Result<()> {
        match &mut self.ords {
            SetOrds::Single(r) => {
                self.values.clear();
                if let Some(o) = r.value(self.cur.doc)? {
                    self.values.push(o);
                }
            }
            SetOrds::Multi(r) => r.values(self.cur.doc, &mut self.values)?,
        }
        self.upto = 0;
        Ok(())
    }
}
dv_iterator!(SegSortedSet);

impl SortedSetDocValues for SegSortedSet<'_> {
    fn doc_value_count(&self) -> i32 {
        self.values.len() as i32
    }
    fn next_ord(&mut self) -> Result<i64> {
        let v = self.values.get(self.upto).copied().ok_or_else(|| {
            Error::IllegalState("nextOrd called more than docValueCount times".into())
        })?;
        self.upto += 1;
        Ok(v)
    }
    fn lookup_ord(&mut self, ord: i64) -> Result<Vec<u8>> {
        Ok(self.terms.seek_ord(ord)?.to_vec())
    }
    fn value_count(&self) -> i64 {
        self.terms.size()
    }
    // SENTINEL: negative = absent, `-insertionPoint-1` (`SortedSetDocValues.lookupTerm`).
    fn lookup_term(&mut self, key: &[u8]) -> Result<i64> {
        dict_lookup_term(&mut self.terms, key)
    }
}

// ---------------------------------------------------------------------------
// Terms and postings
// ---------------------------------------------------------------------------

/// One field's terms in a segment, with the postings inputs they decode from.
struct SegmentTerms<'a> {
    field: &'a FieldTerms,
    doc_in: Option<DocInput<'a>>,
    pos_in: Option<PosInput<'a>>,
    pay_in: Option<PayInput<'a>>,
}

impl Terms for SegmentTerms<'_> {
    fn iterator(&self) -> Result<Box<dyn TermsEnum + '_>> {
        Ok(Box::new(SegmentTermsEnum {
            te: self.field.iter(),
            terms: self,
        }))
    }
    fn size(&self) -> i64 {
        self.field.num_terms
    }
    fn sum_total_term_freq(&self) -> i64 {
        self.field.sum_total_term_freq
    }
    fn sum_doc_freq(&self) -> i64 {
        self.field.sum_doc_freq
    }
    fn doc_count(&self) -> i32 {
        self.field.doc_count
    }
    fn has_freqs(&self) -> bool {
        !matches!(
            self.field.index_options(),
            IndexOptions::None | IndexOptions::Docs
        )
    }
    fn has_offsets(&self) -> bool {
        self.field.index_options() == IndexOptions::DocsAndFreqsAndPositionsAndOffsets
    }
    fn has_positions(&self) -> bool {
        matches!(
            self.field.index_options(),
            IndexOptions::DocsAndFreqsAndPositions
                | IndexOptions::DocsAndFreqsAndPositionsAndOffsets
        )
    }
    fn has_payloads(&self) -> bool {
        self.field.has_payloads()
    }
    fn min(&self) -> Result<Option<Vec<u8>>> {
        Ok(Some(self.field.min_term.clone()))
    }
    fn max(&self) -> Result<Option<Vec<u8>>> {
        Ok(Some(self.field.max_term.clone()))
    }
}

struct SegmentTermsEnum<'s> {
    te: blocktree::TermsEnum<'s>,
    terms: &'s SegmentTerms<'s>,
}

impl SegmentTermsEnum<'_> {
    fn stats(&mut self) -> Result<blocktree::TermStats> {
        self.te
            .try_stats()?
            .ok_or_else(|| Error::IllegalState("the TermsEnum is not positioned on a term".into()))
    }
}

impl TermsEnum for SegmentTermsEnum<'_> {
    fn next(&mut self) -> Result<Option<&[u8]>> {
        Ok(self.te.try_next_term()?)
    }
    fn term(&self) -> Option<&[u8]> {
        self.te.term()
    }
    fn try_seek_ceil(&mut self, target: &[u8]) -> Result<SeekStatus> {
        Ok(self.te.try_seek_ceil(target)?)
    }
    fn doc_freq(&mut self) -> Result<i32> {
        Ok(self.stats()?.doc_freq)
    }
    fn total_term_freq(&mut self) -> Result<i64> {
        Ok(self.stats()?.total_term_freq)
    }
    fn postings(&mut self, flags: PostingsFlags) -> Result<Box<dyn PostingsEnum>> {
        let t = self.terms;
        let positional = flags.wants_positions() && t.has_positions();
        let (docs, freqs, positions) = match (&t.pos_in, positional) {
            (Some(pos_in), true) => {
                let (p, positions) = self
                    .te
                    .try_current_postings_and_positions(
                        t.doc_in.as_ref(),
                        pos_in,
                        t.pay_in.as_ref(),
                    )?
                    .ok_or_else(|| {
                        Error::IllegalState("the TermsEnum is not positioned on a term".into())
                    })?;
                (p.docs, p.freqs, Some(positions))
            }
            _ => {
                let p = self
                    .te
                    .try_current_postings(t.doc_in.as_ref())?
                    .ok_or_else(|| {
                        Error::IllegalState("the TermsEnum is not positioned on a term".into())
                    })?;
                (p.docs, p.freqs, None)
            }
        };
        // A field without freqs, or a caller that asked for none, reads 1.
        let freqs = if freqs.len() == docs.len() && flags != PostingsFlags::None {
            freqs
        } else {
            vec![1; docs.len()]
        };
        Ok(Box::new(MaterializedPostings::new(docs, freqs, positions)?))
    }
}

// ---------------------------------------------------------------------------
// Points and vectors
// ---------------------------------------------------------------------------

struct SegPoints<'a> {
    reader: lucene_codecs::points::PointsReader<'a>,
    number: i32,
}

impl SegPoints<'_> {
    fn field(&self) -> &lucene_codecs::points::PointsField {
        self.reader
            .field(self.number)
            .expect("checked when the values were opened")
    }
}

impl PointValues for SegPoints<'_> {
    fn intersect(&self, visitor: &mut dyn IntersectVisitor) -> Result<()> {
        Ok(self
            .reader
            .intersect(self.number, &mut DynVisitor(visitor))?)
    }
    fn estimate_point_count(&self, visitor: &mut dyn IntersectVisitor) -> Result<i64> {
        Ok(self
            .reader
            .estimate_point_count(self.number, &mut DynVisitor(visitor))?)
    }
    fn min_packed_value(&self) -> &[u8] {
        &self.field().min_packed_value
    }
    fn max_packed_value(&self) -> &[u8] {
        &self.field().max_packed_value
    }
    fn num_dimensions(&self) -> i32 {
        self.field().num_dims
    }
    fn num_index_dimensions(&self) -> i32 {
        self.field().num_index_dims
    }
    fn bytes_per_dimension(&self) -> i32 {
        self.field().bytes_per_dim
    }
    fn size(&self) -> i64 {
        self.field().point_count
    }
    fn doc_count(&self) -> i32 {
        self.field().doc_count
    }
}

struct SegFloatVectors<'a>(lucene_codecs::vectors::FloatVectorValues<'a>);

impl FloatVectorValues for SegFloatVectors<'_> {
    fn dimension(&self) -> usize {
        self.0.dimension()
    }
    fn size(&self) -> i32 {
        self.0.size()
    }
    fn ord_to_doc(&self, ord: i32) -> Result<i32> {
        Ok(self.0.ord_to_doc(ord)?)
    }
    fn vector_value(&self, ord: i32) -> Result<Vec<f32>> {
        Ok(self.0.vector(ord)?)
    }
}

struct SegByteVectors<'a>(lucene_codecs::vectors::ByteVectorValues<'a>);

impl ByteVectorValues for SegByteVectors<'_> {
    fn dimension(&self) -> usize {
        self.0.dimension()
    }
    fn size(&self) -> i32 {
        self.0.size()
    }
    fn ord_to_doc(&self, ord: i32) -> Result<i32> {
        Ok(self.0.ord_to_doc(ord)?)
    }
    fn vector_value(&self, ord: i32) -> Result<Vec<u8>> {
        Ok(self.0.vector(ord)?.to_vec())
    }
}

// ---------------------------------------------------------------------------
// SegmentReader
// ---------------------------------------------------------------------------

impl SegmentReader {
    /// `fieldInfo(field)` when its doc-values type is `kind`, with the column
    /// (meta, data) serving it.
    fn dv_field<'s, E>(
        &'s self,
        field: &str,
        kind: DocValuesType,
        entry: impl FnOnce(&'s doc_values::DocValuesMeta, i32) -> Option<&'s E>,
    ) -> Option<(&'s E, &'s [u8])> {
        let fi: &FieldInfo = self.field_infos().field_by_name(field)?;
        if fi.doc_values_type != kind {
            return None;
        }
        let (meta, data) = self.doc_values_for_field(fi.number)?;
        Some((entry(meta, fi.number)?, data))
    }
}

crate::impl_leaf_index_reader!(SegmentReader, |s| max_doc: s.max_doc, reader_cache_helper: Some(s.reader_cache_helper()));

impl LeafReader for SegmentReader {
    fn field_infos(&self) -> &FieldInfos {
        SegmentReader::field_infos(self)
    }

    fn live_docs(&self) -> Option<&FixedBitSet> {
        SegmentReader::live_docs(self)
    }

    fn terms(&self, field: &str) -> Result<Option<Box<dyn Terms + '_>>> {
        let Some(terms) = self.block_tree_fields().field(field) else {
            return Ok(None);
        };
        let (doc_in, pos_in, pay_in) = self.postings_inputs()?;
        Ok(Some(Box::new(SegmentTerms {
            field: terms,
            doc_in,
            pos_in,
            pay_in,
        })))
    }

    fn numeric_doc_values(&self, field: &str) -> Result<Option<Box<dyn NumericDocValues + '_>>> {
        let Some((e, data)) =
            self.dv_field(field, DocValuesType::Numeric, |m, n| m.numeric_entry(n))
        else {
            return Ok(None);
        };
        Ok(Some(Box::new(SegNumeric {
            cur: DvCursor::new(DocSet::of_numeric(self, data, e)?),
            reader: NumericReader::new(data, e),
            value: 0,
        })))
    }

    fn binary_doc_values(&self, field: &str) -> Result<Option<Box<dyn BinaryDocValues + '_>>> {
        let Some((e, data)) = self.dv_field(field, DocValuesType::Binary, |m, n| m.binary_entry(n))
        else {
            return Ok(None);
        };
        let docs = DocSet::read(
            data,
            e.docs_with_field_offset,
            e.docs_with_field_length,
            e.dense_rank_power,
            self.max_doc,
        )?;
        Ok(Some(Box::new(SegBinary {
            cur: DvCursor::new(docs),
            reader: BinaryReader::new(data, e),
            value: &[],
        })))
    }

    fn sorted_doc_values(&self, field: &str) -> Result<Option<Box<dyn SortedDocValues + '_>>> {
        let Some((e, data)) = self.dv_field(field, DocValuesType::Sorted, |m, n| m.sorted_entry(n))
        else {
            return Ok(None);
        };
        Ok(Some(Box::new(SegSorted {
            cur: DvCursor::new(DocSet::of_numeric(self, data, &e.ords)?),
            ords: NumericReader::new(data, &e.ords),
            terms: TermsDict::open(data, &e.terms)?,
            ord: -1,
        })))
    }

    fn sorted_numeric_doc_values(
        &self,
        field: &str,
    ) -> Result<Option<Box<dyn SortedNumericDocValues + '_>>> {
        let Some((e, data)) = self.dv_field(field, DocValuesType::SortedNumeric, |m, n| {
            m.sorted_numeric_entry(n)
        }) else {
            return Ok(None);
        };
        Ok(Some(Box::new(SegSortedNumeric {
            cur: DvCursor::new(DocSet::of_numeric(self, data, &e.numeric)?),
            reader: SortedNumericReader::new(data, e),
            values: Vec::new(),
            upto: 0,
        })))
    }

    fn sorted_set_doc_values(
        &self,
        field: &str,
    ) -> Result<Option<Box<dyn SortedSetDocValues + '_>>> {
        let Some((e, data)) = self.dv_field(field, DocValuesType::SortedSet, |m, n| {
            m.sorted_set_entry(n)
        }) else {
            return Ok(None);
        };
        let (docs, ords, terms) = match &e.kind {
            SortedSetKind::Single(s) => (
                DocSet::of_numeric(self, data, &s.ords)?,
                SetOrds::Single(NumericReader::new(data, &s.ords)),
                &s.terms,
            ),
            SortedSetKind::Multi { ords, terms } => (
                DocSet::of_numeric(self, data, &ords.numeric)?,
                SetOrds::Multi(SortedNumericReader::new(data, ords)),
                terms,
            ),
        };
        Ok(Some(Box::new(SegSortedSet {
            cur: DvCursor::new(docs),
            ords,
            terms: TermsDict::open(data, terms)?,
            values: Vec::new(),
            upto: 0,
        })))
    }

    fn norm_values(&self, field: &str) -> Result<Option<Box<dyn NumericDocValues + '_>>> {
        let Some(fi) = SegmentReader::field_infos(self).field_by_name(field) else {
            return Ok(None);
        };
        if fi.omit_norms || fi.index_options == IndexOptions::None {
            return Ok(None);
        }
        let (Some(e), Some(data)) = (self.norms_entry(fi.number), self.norms_data()) else {
            return Ok(None);
        };
        // Norms record their documents exactly as doc values do.
        let docs = DocSet::read(
            data,
            e.docs_with_field_offset,
            e.docs_with_field_length,
            e.dense_rank_power,
            self.max_doc,
        )?;
        Ok(Some(Box::new(SegNorms {
            cur: DvCursor::new(docs),
            reader: lucene_codecs::norms::NormsReader::new(data, e),
            value: 0,
        })))
    }

    fn point_values(&self, field: &str) -> Result<Option<Box<dyn PointValues + '_>>> {
        let Some(fi) = SegmentReader::field_infos(self).field_by_name(field) else {
            return Ok(None);
        };
        if fi.point_dimension_count == 0 {
            return Ok(None);
        }
        let reader = self.points_reader()?;
        Ok(reader.field(fi.number).is_some().then(|| {
            Box::new(SegPoints {
                reader,
                number: fi.number,
            }) as Box<dyn PointValues + '_>
        }))
    }

    fn float_vector_values(&self, field: &str) -> Result<Option<Box<dyn FloatVectorValues + '_>>> {
        let Some(fi) = SegmentReader::field_infos(self).field_by_name(field) else {
            return Ok(None);
        };
        if fi.vector_dimension == 0
            || fi.vector_encoding != lucene_codecs::field_infos::VectorEncoding::Float32
        {
            return Ok(None);
        }
        let Some(flat) = self.flat_vectors_reader_for(fi.number)? else {
            return Ok(None);
        };
        Ok(Some(Box::new(SegFloatVectors(
            flat.float_vector_values(fi.number)?,
        ))))
    }

    fn byte_vector_values(&self, field: &str) -> Result<Option<Box<dyn ByteVectorValues + '_>>> {
        let Some(fi) = SegmentReader::field_infos(self).field_by_name(field) else {
            return Ok(None);
        };
        if fi.vector_dimension == 0
            || fi.vector_encoding != lucene_codecs::field_infos::VectorEncoding::Byte
        {
            return Ok(None);
        }
        let Some(flat) = self.flat_vectors_reader_for(fi.number)? else {
            return Ok(None);
        };
        Ok(Some(Box::new(SegByteVectors(
            flat.byte_vector_values(fi.number)?,
        ))))
    }

    fn document(&self, doc: i32, visitor: &mut dyn StoredFieldVisitor) -> Result<()> {
        let _any = self.visit_stored_document(doc, visitor)?;
        Ok(())
    }

    fn term_vectors(&self, doc: i32) -> Result<Option<TermVectorsDocument>> {
        match self.term_vectors_reader()? {
            None => Ok(None),
            Some(r) => Ok(r.document(doc)?),
        }
    }

    fn index_sort(&self) -> Option<&[IndexSortField]> {
        SegmentReader::index_sort(self)
    }

    fn core_cache_helper(&self) -> Option<&CacheHelper> {
        Some(SegmentReader::core_cache_helper(self))
    }

    fn doc_values_skipper(&self, field: &str) -> Result<Option<doc_values::DocValuesSkipper<'_>>> {
        let Some(fi) = SegmentReader::field_infos(self).field_by_name(field) else {
            return Ok(None);
        };
        Ok(self
            .doc_values_skip_index(fi.number)?
            .map(doc_values::DocValuesSkipper::new))
    }

    fn check_integrity(&self) -> Result<()> {
        SegmentReader::check_integrity(self)
    }
}

impl CodecReader for SegmentReader {}

// ---------------------------------------------------------------------------
// DirectoryReader
// ---------------------------------------------------------------------------

impl IndexReader for DirectoryReader {
    fn max_doc(&self) -> i32 {
        DirectoryReader::max_doc(self)
    }
    fn num_docs(&self) -> i32 {
        DirectoryReader::num_docs(self)
    }
    fn leaves(&self) -> Vec<LeafReaderContext<'_>> {
        super::composite_leaves(&self.sequential_sub_readers())
    }
    fn reader_cache_helper(&self) -> Option<&CacheHelper> {
        Some(DirectoryReader::reader_cache_helper(self))
    }
}

impl CompositeReader for DirectoryReader {
    fn sequential_sub_readers(&self) -> Vec<SubReader<'_>> {
        self.segment_readers()
            .iter()
            .map(|s| SubReader::Leaf(s))
            .collect()
    }
    fn leaf_handles(&self) -> Vec<std::sync::Arc<dyn LeafReader>> {
        self.segment_readers()
            .iter()
            .map(|s| std::sync::Arc::new(s.clone()) as std::sync::Arc<dyn LeafReader>)
            .collect()
    }
}
