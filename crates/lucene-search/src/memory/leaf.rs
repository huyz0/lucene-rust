//! `MemoryIndexReader`'s `LeafReader` half: Java's `MemoryFields`,
//! `MemoryTermsEnum`, `MemoryPostingsEnum`, `MemoryDocValuesIterator`,
//! `MemoryIndexPointValues` and stored-fields visit, over the arrays
//! [`super::MemoryIndex`] hands the reader -- and `Weight.matches`
//! ([`crate::matches::MatchesLeaf`]) read from the same arrays.
//!
//! # What differs from Java
//!
//! The postings are read the way the codec form of the document (which this
//! reader replaced on these paths) reads them, so the two forms agree
//! exactly (`leaf_duels_the_segment` in the tests): `freq()` is `1` for a
//! caller that asked for `PostingsEnum.NONE` and positions are handed out
//! only to a caller that asked for them, as a codec's postings do, where
//! Java's `MemoryPostingsEnum` reads both regardless; a field without
//! offsets reports `-1`, the `PostingsEnum` contract, where Java's reports
//! `0`; and `hasPayloads()` is whether a payload was stored, not the index's
//! `storePayloads`. No term vectors are served (`termVectors()` has none),
//! nor vectors, which [`super::MemoryIndex`] does not hold.

use lucene_codecs::blocktree::SeekStatus;
use lucene_codecs::field_infos::{DocValuesType, FieldInfos, IndexOptions};
use lucene_codecs::postings::Position;
use lucene_codecs::postings_writer::TermPostings;
use lucene_codecs::regexp::RegexpPattern;
use lucene_codecs::stored_fields::{FieldValue, StoredFieldVisitor, VisitStatus};
use lucene_codecs::wildcard::WildcardPattern;
use lucene_index::segment_info::IndexSortField;
use lucene_util::fixed_bit_set::FixedBitSet;

use super::reader::{FieldPostings, MemoryIndexReader};
use crate::matches::{BoxMatches, MatchesLeaf};
use crate::query::{Clause, FuzzyQuery};
use crate::reader::{
    BinaryDocValues, ByteVectorValues, CacheHelper, DocIdSetIterator, DocValuesIterator,
    FloatVectorValues, IndexReader, IntersectVisitor, LeafReader, LeafReaderContext,
    MaterializedPostings, NumericDocValues, PointValues, PostingsEnum, PostingsFlags, Relation,
    SortedDocValues, SortedNumericDocValues, SortedSetDocValues, TermVectorsDocument, Terms,
    TermsEnum, NO_MORE_DOCS,
};
use crate::{Error, Result};

/// A term's occurrences in the document, as `PostingsEnum.OFFSETS` reads
/// them: offsets `-1` without offsets, an empty payload without one.
fn occurrences_of(t: &TermPostings) -> Vec<Position> {
    let positions = t.positions.first().map_or(&[][..], Vec::as_slice);
    let offsets = t.offsets.first().map_or(&[][..], Vec::as_slice);
    let mut at = 0usize;
    positions
        .iter()
        .enumerate()
        .map(|(i, &position)| {
            let (start_offset, end_offset) = offsets.get(i).copied().unwrap_or((-1, -1));
            let len = t.payload_lengths.get(i).map_or(0, |&l| l as usize);
            let end = at.saturating_add(len).min(t.payload_bytes.len());
            let payload = t
                .payload_bytes
                .get(at..end)
                .map_or_else(Vec::new, <[u8]>::to_vec);
            at = end;
            Position {
                position,
                start_offset,
                end_offset,
                payload,
            }
        })
        .collect()
}

/// The term's entry in the field, by binary search.
fn find_term<'f>(f: &'f FieldPostings, term: &[u8]) -> Option<&'f TermPostings> {
    f.terms
        .binary_search_by(|t| t.term.as_slice().cmp(term))
        .ok()
        .map(|i| &f.terms[i])
}

impl MemoryIndexReader {
    /// The field when it has terms (`MemoryFields.terms`: `null` for
    /// `numTokens <= 0`).
    fn terms_field(&self, name: &str) -> Option<&FieldPostings> {
        self.field(name).filter(|f| !f.terms.is_empty())
    }

    /// The field's doc values when they are of type `ty`
    /// (`getInfoForExpectedDocValuesType`).
    fn doc_values_of(&self, name: &str, ty: DocValuesType) -> Option<Vec<&FieldValue>> {
        let f = self.field(name).filter(|f| f.info.doc_values_type == ty)?;
        let number = f.info.number;
        Some(
            self.doc
                .doc_values
                .iter()
                .filter(|v| v.field_number == number)
                .map(|v| &v.value)
                .collect(),
        )
    }

    /// The field's numeric doc values, ascending
    /// (`NumericDocValuesProducer.prepareForUsage`).
    fn longs_of(&self, name: &str, ty: DocValuesType) -> Option<Vec<i64>> {
        let mut values: Vec<i64> = self
            .doc_values_of(name, ty)?
            .into_iter()
            .filter_map(|v| match v {
                FieldValue::Long(l) => Some(*l),
                _ => None,
            })
            .collect();
        values.sort_unstable();
        Some(values)
    }

    /// The field's binary doc values, in the order they are held (sorted
    /// for `SORTED_SET`).
    fn bytes_of(&self, name: &str, ty: DocValuesType) -> Option<Vec<&[u8]>> {
        Some(
            self.doc_values_of(name, ty)?
                .into_iter()
                .filter_map(|v| match v {
                    FieldValue::Binary(b) => Some(b.as_slice()),
                    _ => None,
                })
                .collect(),
        )
    }
}

impl MatchesLeaf for MemoryIndexReader {
    fn field_index_options(&self, field: &str) -> Option<IndexOptions> {
        self.terms_field(field).map(|f| f.info.index_options)
    }

    fn reads_positions(&self) -> bool {
        // every field of a memory index is indexed with positions
        true
    }

    fn occurrences(&self, field: &str, term: &[u8], doc: i32) -> Result<Option<Vec<Position>>> {
        if doc != 0 {
            return Ok(None);
        }
        Ok(self
            .terms_field(field)
            .and_then(|f| find_term(f, term))
            .map(occurrences_of))
    }

    fn expanded_terms(&self, clause: &Clause) -> Result<Option<(String, Vec<Vec<u8>>)>> {
        /// The clause's acceptance test, before it is compiled.
        enum Kind<'q> {
            Set(&'q [Vec<u8>]),
            Prefix(&'q [u8]),
            Wildcard(&'q [u8]),
            Regexp(&'q str),
        }
        let (field, kind) = match clause {
            Clause::TermInSet(q) => (&q.field, Kind::Set(&q.terms)),
            Clause::Prefix(q) => (&q.field, Kind::Prefix(&q.prefix)),
            Clause::Wildcard(q) => (&q.field, Kind::Wildcard(&q.pattern)),
            Clause::Regexp(q) => (&q.field, Kind::Regexp(&q.pattern)),
            _ => return Ok(None),
        };
        // the field first: a regexp over an absent field is not compiled
        let Some(f) = self.terms_field(field) else {
            return Ok(None);
        };
        type Accept<'q> = Box<dyn Fn(&[u8]) -> bool + 'q>;
        let accept: Accept<'_> = match kind {
            Kind::Set(terms) => {
                let mut wanted: Vec<&[u8]> = terms.iter().map(Vec::as_slice).collect();
                wanted.sort_unstable();
                Box::new(move |t| wanted.binary_search(&t).is_ok())
            }
            Kind::Prefix(prefix) => {
                let pattern = WildcardPattern::prefix(prefix);
                Box::new(move |t| pattern.matches(t))
            }
            Kind::Wildcard(p) => {
                let pattern = WildcardPattern::new(p);
                Box::new(move |t| pattern.matches(t))
            }
            Kind::Regexp(p) => {
                let pattern = RegexpPattern::new(p.as_bytes())?;
                Box::new(move |t| pattern.matches(t))
            }
        };
        let terms = f
            .terms
            .iter()
            .filter(|t| accept(&t.term))
            .map(|t| t.term.clone())
            .collect();
        Ok(Some((field.clone(), terms)))
    }

    fn fuzzy_terms(&self, query: &FuzzyQuery) -> Result<Option<Vec<Vec<u8>>>> {
        Ok(self.terms_field(&query.field).map(|f| {
            crate::fuzzy_expanded_terms_in_memory(f.terms.iter().map(|t| t.term.as_slice()), query)
        }))
    }

    fn codec_matches(&self, clause: &Clause, doc: i32) -> Result<Option<BoxMatches>> {
        MemoryIndexReader::codec_matches(self, clause, doc)
    }
}

// ---------------------------------------------------------------------------
// Terms and postings
// ---------------------------------------------------------------------------

/// `MemoryFields.terms(field)`'s `Terms`.
struct MemoryTerms<'a> {
    field: &'a FieldPostings,
}

impl Terms for MemoryTerms<'_> {
    fn iterator(&self) -> Result<Box<dyn TermsEnum + '_>> {
        Ok(Box::new(MemoryTermsEnum {
            terms: &self.field.terms,
            upto: None,
        }))
    }
    fn size(&self) -> i64 {
        self.field.terms.len() as i64
    }
    fn sum_total_term_freq(&self) -> i64 {
        self.field
            .terms
            .iter()
            .map(|t| i64::from(t.docs[0].1))
            .sum()
    }
    /// Each term has `docFreq` 1.
    fn sum_doc_freq(&self) -> i64 {
        self.size()
    }
    fn doc_count(&self) -> i32 {
        1
    }
    fn has_freqs(&self) -> bool {
        true
    }
    fn has_offsets(&self) -> bool {
        self.field.info.index_options.subsumes_offsets()
    }
    fn has_positions(&self) -> bool {
        true
    }
    fn has_payloads(&self) -> bool {
        self.field.has_payloads
    }
}

/// `MemoryTermsEnum`: the field's terms by ordinal.
struct MemoryTermsEnum<'a> {
    terms: &'a [TermPostings],
    /// `termUpto`: `None` before the first `next` or after a failed
    /// `seekExact`, `terms.len()` once exhausted.
    upto: Option<usize>,
}

impl MemoryTermsEnum<'_> {
    /// The ordinal of the term the enum is on.
    fn current_ord(&self) -> Result<usize> {
        self.upto
            .filter(|&u| u < self.terms.len())
            .ok_or_else(|| Error::IllegalState("the TermsEnum is not positioned on a term".into()))
    }

    fn on_term(&self) -> Result<&TermPostings> {
        Ok(&self.terms[self.current_ord()?])
    }
}

impl TermsEnum for MemoryTermsEnum<'_> {
    fn next(&mut self) -> Result<Option<&[u8]>> {
        let next = self
            .upto
            .map_or(0, |u| u.saturating_add(1))
            .min(self.terms.len());
        self.upto = Some(next);
        Ok(self.terms.get(next).map(|t| t.term.as_slice()))
    }
    fn term(&self) -> Option<&[u8]> {
        self.on_term().ok().map(|t| t.term.as_slice())
    }
    fn try_seek_ceil(&mut self, target: &[u8]) -> Result<SeekStatus> {
        match self
            .terms
            .binary_search_by(|t| t.term.as_slice().cmp(target))
        {
            Ok(i) => {
                self.upto = Some(i);
                Ok(SeekStatus::Found)
            }
            Err(i) => {
                self.upto = Some(i);
                Ok(if i < self.terms.len() {
                    SeekStatus::NotFound
                } else {
                    SeekStatus::End
                })
            }
        }
    }
    fn try_seek_exact(&mut self, target: &[u8]) -> Result<bool> {
        self.upto = self
            .terms
            .binary_search_by(|t| t.term.as_slice().cmp(target))
            .ok();
        Ok(self.upto.is_some())
    }
    fn doc_freq(&mut self) -> Result<i32> {
        self.on_term().map(|_| 1)
    }
    fn total_term_freq(&mut self) -> Result<i64> {
        self.on_term().map(|t| i64::from(t.docs[0].1))
    }
    fn postings(&mut self, flags: PostingsFlags) -> Result<Box<dyn PostingsEnum>> {
        let t = self.on_term()?;
        // a caller that asked for no freqs reads 1, as from a codec
        let freq = if flags == PostingsFlags::None {
            1
        } else {
            t.docs[0].1
        };
        let positions = flags.wants_positions().then(|| vec![occurrences_of(t)]);
        Ok(Box::new(MaterializedPostings::new(
            vec![0],
            vec![freq],
            positions,
        )?))
    }
    fn ord(&self) -> Result<i64> {
        Ok(self.current_ord()? as i64)
    }
    fn seek_exact_ord(&mut self, ord: i64) -> Result<()> {
        let u = usize::try_from(ord)
            .ok()
            .filter(|&u| u < self.terms.len())
            .ok_or_else(|| {
                Error::IllegalArgument(format!(
                    "ord {ord} is out of bounds for {} terms",
                    self.terms.len()
                ))
            })?;
        self.upto = Some(u);
        Ok(())
    }
    /// An `OrdTermState` of `termUpto` (`-1` before the first term).
    fn term_state(&mut self) -> Result<crate::reader::TermState> {
        let ord = self
            .upto
            .map_or(-1, |u| i64::try_from(u).unwrap_or(i64::MAX));
        Ok(crate::reader::TermState::ord(ord))
    }
    /// `seekExact(state.ord)`.
    fn seek_exact_state(&mut self, _term: &[u8], state: &crate::reader::TermState) -> Result<()> {
        let ord = state
            .as_ord()
            .ok_or_else(|| Error::IllegalArgument("not an OrdTermState".into()))?;
        self.seek_exact_ord(ord)
    }
}

// ---------------------------------------------------------------------------
// Doc values and norms
// ---------------------------------------------------------------------------

/// `MemoryDocValuesIterator`: the one document `0`.
#[derive(Debug)]
struct OneDoc {
    doc: i32,
}

impl OneDoc {
    fn new() -> Self {
        OneDoc { doc: -1 }
    }
    fn advance(&mut self, target: i32) -> i32 {
        self.doc = target;
        self.doc_id()
    }
    fn next_doc(&mut self) -> i32 {
        self.doc = self.doc.saturating_add(1);
        self.doc_id()
    }
    fn doc_id(&self) -> i32 {
        if self.doc > 0 {
            NO_MORE_DOCS
        } else {
            self.doc
        }
    }
}

/// The [`DocIdSetIterator`] and [`DocValuesIterator`] halves of a doc-values
/// type over [`OneDoc`]; `advanceExact` rewinds the per-document values.
macro_rules! one_doc_iterator {
    ($ty:ident $(<$lt:lifetime>)?) => {
        impl$(<$lt>)? DocIdSetIterator for $ty $(<$lt>)? {
            fn doc_id(&self) -> i32 {
                self.it.doc_id()
            }
            fn next_doc(&mut self) -> Result<i32> {
                Ok(self.it.next_doc())
            }
            fn advance(&mut self, target: i32) -> Result<i32> {
                Ok(self.it.advance(target))
            }
            fn cost(&self) -> i64 {
                1
            }
        }
        impl$(<$lt>)? DocValuesIterator for $ty $(<$lt>)? {
            fn advance_exact(&mut self, target: i32) -> Result<bool> {
                self.rewind();
                Ok(self.it.advance(target) == target)
            }
        }
    };
}

/// `numericDocValues(long)`: a `NUMERIC` value, or a norm.
struct OneLong {
    it: OneDoc,
    value: i64,
}

impl OneLong {
    fn new(value: i64) -> Self {
        OneLong {
            it: OneDoc::new(),
            value,
        }
    }
    fn rewind(&mut self) {}
}

one_doc_iterator!(OneLong);

impl NumericDocValues for OneLong {
    fn long_value(&self) -> i64 {
        self.value
    }
}

/// `sortedDocValues(BytesRef)`, and the `BINARY` view of it.
struct OneBytes<'a> {
    it: OneDoc,
    value: &'a [u8],
}

impl OneBytes<'_> {
    fn rewind(&mut self) {}
}

one_doc_iterator!(OneBytes<'a>);

impl BinaryDocValues for OneBytes<'_> {
    fn binary_value(&self) -> &[u8] {
        self.value
    }
}

impl SortedDocValues for OneBytes<'_> {
    fn ord_value(&self) -> i32 {
        0
    }
    /// The value, whatever the ordinal (Java's `lookupOrd`).
    fn lookup_ord(&mut self, _ord: i32) -> Result<Vec<u8>> {
        Ok(self.value.to_vec())
    }
    fn value_count(&self) -> i32 {
        1
    }
}

/// `numericDocValues(long[], int)`: the `SORTED_NUMERIC` values, ascending.
struct ManyLongs {
    it: OneDoc,
    values: Vec<i64>,
    upto: usize,
}

impl ManyLongs {
    fn rewind(&mut self) {
        self.upto = 0;
    }
}

one_doc_iterator!(ManyLongs);

impl SortedNumericDocValues for ManyLongs {
    fn doc_value_count(&self) -> i32 {
        i32::try_from(self.values.len()).unwrap_or(i32::MAX)
    }
    fn next_value(&mut self) -> Result<i64> {
        let v = self.values.get(self.upto).copied().ok_or_else(|| {
            Error::IllegalState("nextValue called more than docValueCount() times".into())
        })?;
        self.upto = self.upto.saturating_add(1);
        Ok(v)
    }
}

/// `sortedSetDocValues(BytesRefHash, int[])`: the `SORTED_SET` values,
/// sorted, ordinals `0..n`.
struct ManyBytes<'a> {
    it: OneDoc,
    values: Vec<&'a [u8]>,
    upto: i64,
}

impl ManyBytes<'_> {
    fn rewind(&mut self) {
        self.upto = 0;
    }
}

one_doc_iterator!(ManyBytes<'a>);

impl SortedSetDocValues for ManyBytes<'_> {
    fn doc_value_count(&self) -> i32 {
        i32::try_from(self.values.len()).unwrap_or(i32::MAX)
    }
    fn next_ord(&mut self) -> Result<i64> {
        let ord = self.upto;
        self.upto = self.upto.saturating_add(1);
        Ok(ord)
    }
    fn lookup_ord(&mut self, ord: i64) -> Result<Vec<u8>> {
        usize::try_from(ord)
            .ok()
            .and_then(|o| self.values.get(o))
            .map(|v| v.to_vec())
            .ok_or_else(|| Error::IllegalArgument(format!("ord {ord} is out of bounds")))
    }
    fn value_count(&self) -> i64 {
        self.values.len() as i64
    }
}

// ---------------------------------------------------------------------------
// Points
// ---------------------------------------------------------------------------

/// `MemoryIndexPointValues`: a one-node tree over the field's points.
struct MemoryPoints<'a> {
    /// Sorted for one dimension (`PointInSetQuery`'s visitor expects
    /// ascending values), in add order otherwise.
    values: Vec<&'a [u8]>,
    min: Vec<u8>,
    max: Vec<u8>,
    dims: i32,
    bytes_per_dim: i32,
}

impl<'a> MemoryPoints<'a> {
    /// `prepareDocValuesAndPointValues`' points half: `None` without points.
    fn new(mut values: Vec<&'a [u8]>, dims: i32, bytes_per_dim: i32) -> Option<Self> {
        let first = *values.first()?;
        let (mut min, mut max) = (first.to_vec(), first.to_vec());
        if dims == 1 {
            values.sort_unstable();
            min = values[0].to_vec();
            max = values.last().map_or_else(Vec::new, |v| v.to_vec());
        } else {
            let width = usize::try_from(bytes_per_dim).unwrap_or(0).max(1);
            for v in &values {
                for ((lo, hi), dim) in min
                    .chunks_mut(width)
                    .zip(max.chunks_mut(width))
                    .zip(v.chunks(width))
                {
                    if dim < &*lo {
                        lo.copy_from_slice(dim);
                    }
                    if dim > &*hi {
                        hi.copy_from_slice(dim);
                    }
                }
            }
        }
        Some(MemoryPoints {
            values,
            min,
            max,
            dims,
            bytes_per_dim,
        })
    }
}

impl PointValues for MemoryPoints<'_> {
    /// `PointValues.intersect` over the one-node tree: `visitDocIDs` inside
    /// the query, `visitDocValues` crossing it.
    fn intersect(&self, visitor: &mut dyn IntersectVisitor) -> Result<()> {
        let relation = visitor.compare(&self.min, &self.max);
        if relation == Relation::CellOutsideQuery {
            return Ok(());
        }
        visitor.grow(self.values.len());
        for v in &self.values {
            if relation == Relation::CellInsideQuery {
                visitor.visit(0);
            } else {
                visitor.visit_with_value(0, v);
            }
        }
        Ok(())
    }
    /// `PointValues.estimatePointCount` over the one-node tree.
    fn estimate_point_count(&self, visitor: &mut dyn IntersectVisitor) -> Result<i64> {
        Ok(match visitor.compare(&self.min, &self.max) {
            Relation::CellOutsideQuery => 0,
            Relation::CellInsideQuery => self.size(),
            Relation::CellCrossesQuery => (self.size() + 1) / 2,
        })
    }
    fn min_packed_value(&self) -> &[u8] {
        &self.min
    }
    fn max_packed_value(&self) -> &[u8] {
        &self.max
    }
    fn num_dimensions(&self) -> i32 {
        self.dims
    }
    /// The dimension count, as Java's reader answers.
    fn num_index_dimensions(&self) -> i32 {
        self.dims
    }
    fn bytes_per_dimension(&self) -> i32 {
        self.bytes_per_dim
    }
    fn size(&self) -> i64 {
        self.values.len() as i64
    }
    fn doc_count(&self) -> i32 {
        1
    }
}

// ---------------------------------------------------------------------------
// The reader
// ---------------------------------------------------------------------------

impl IndexReader for MemoryIndexReader {
    fn max_doc(&self) -> i32 {
        1
    }
    fn num_docs(&self) -> i32 {
        1
    }
    fn leaves(&self) -> Vec<LeafReaderContext<'_>> {
        vec![LeafReaderContext {
            reader: self,
            ord: 0,
            doc_base: 0,
        }]
    }
    /// Java's reader has no cache helper.
    fn reader_cache_helper(&self) -> Option<&CacheHelper> {
        None
    }
}

impl LeafReader for MemoryIndexReader {
    fn field_infos(&self) -> &FieldInfos {
        &self.field_infos
    }
    fn live_docs(&self) -> Option<&FixedBitSet> {
        None
    }
    fn terms(&self, field: &str) -> Result<Option<Box<dyn Terms + '_>>> {
        Ok(self
            .terms_field(field)
            .map(|field| Box::new(MemoryTerms { field }) as Box<dyn Terms + '_>))
    }
    fn numeric_doc_values(&self, field: &str) -> Result<Option<Box<dyn NumericDocValues + '_>>> {
        Ok(self
            .longs_of(field, DocValuesType::Numeric)
            .and_then(|v| v.first().copied())
            .map(|v| Box::new(OneLong::new(v)) as Box<dyn NumericDocValues>))
    }
    fn binary_doc_values(&self, field: &str) -> Result<Option<Box<dyn BinaryDocValues + '_>>> {
        Ok(self
            .bytes_of(field, DocValuesType::Binary)
            .and_then(|v| v.first().copied())
            .map(|value| {
                Box::new(OneBytes {
                    it: OneDoc::new(),
                    value,
                }) as Box<dyn BinaryDocValues + '_>
            }))
    }
    fn sorted_doc_values(&self, field: &str) -> Result<Option<Box<dyn SortedDocValues + '_>>> {
        Ok(self
            .bytes_of(field, DocValuesType::Sorted)
            .and_then(|v| v.first().copied())
            .map(|value| {
                Box::new(OneBytes {
                    it: OneDoc::new(),
                    value,
                }) as Box<dyn SortedDocValues + '_>
            }))
    }
    fn sorted_numeric_doc_values(
        &self,
        field: &str,
    ) -> Result<Option<Box<dyn SortedNumericDocValues + '_>>> {
        Ok(self
            .longs_of(field, DocValuesType::SortedNumeric)
            .map(|values| {
                Box::new(ManyLongs {
                    it: OneDoc::new(),
                    values,
                    upto: 0,
                }) as Box<dyn SortedNumericDocValues>
            }))
    }
    fn sorted_set_doc_values(
        &self,
        field: &str,
    ) -> Result<Option<Box<dyn SortedSetDocValues + '_>>> {
        Ok(self
            .bytes_of(field, DocValuesType::SortedSet)
            .map(|values| {
                Box::new(ManyBytes {
                    it: OneDoc::new(),
                    values,
                    upto: 0,
                }) as Box<dyn SortedSetDocValues + '_>
            }))
    }
    /// `getNormValues`: `None` for a field that omits norms.
    fn norm_values(&self, field: &str) -> Result<Option<Box<dyn NumericDocValues + '_>>> {
        Ok(self
            .field(field)
            .and_then(|f| f.norm)
            .map(|n| Box::new(OneLong::new(n)) as Box<dyn NumericDocValues>))
    }
    fn point_values(&self, field: &str) -> Result<Option<Box<dyn PointValues + '_>>> {
        let Some(f) = self.field(field) else {
            return Ok(None);
        };
        let number = f.info.number;
        let values = self
            .doc
            .points
            .iter()
            .filter(|p| p.field_number == number)
            .filter_map(|p| match &p.value {
                FieldValue::Binary(b) => Some(b.as_slice()),
                _ => None,
            })
            .collect();
        Ok(
            MemoryPoints::new(values, f.info.point_dimension_count, f.info.point_num_bytes)
                .map(|p| Box::new(p) as Box<dyn PointValues + '_>),
        )
    }
    fn float_vector_values(&self, _field: &str) -> Result<Option<Box<dyn FloatVectorValues + '_>>> {
        Ok(None)
    }
    fn byte_vector_values(&self, _field: &str) -> Result<Option<Box<dyn ByteVectorValues + '_>>> {
        Ok(None)
    }
    /// `storedFields().document(docID, visitor)`: every field in name order
    /// is offered to `needsField`, and the stored values of each it wants
    /// visited -- whatever `docID`, as Java's reader does.
    fn document(&self, _doc: i32, visitor: &mut dyn StoredFieldVisitor) -> Result<()> {
        for &i in &self.by_name {
            let number = self.doc.fields[i].info.number;
            match visitor.needs_field(number)? {
                VisitStatus::Stop => return Ok(()),
                VisitStatus::No => continue,
                VisitStatus::Yes => {}
            }
            for stored in self.doc.stored.iter().filter(|s| s.field_number == number) {
                match &stored.value {
                    FieldValue::Binary(b) => visitor.binary_field(number, b)?,
                    FieldValue::Double(d) => visitor.double_field(number, *d)?,
                    FieldValue::Float(f) => visitor.float_field(number, *f)?,
                    FieldValue::Long(l) => visitor.long_field(number, *l)?,
                    FieldValue::Int(n) => visitor.int_field(number, *n)?,
                    FieldValue::String(s) => visitor.string_field(number, s)?,
                }
            }
        }
        Ok(())
    }
    /// The term vectors are not served (see the module docs).
    fn term_vectors(&self, _doc: i32) -> Result<Option<TermVectorsDocument>> {
        Ok(None)
    }
    fn index_sort(&self) -> Option<&[IndexSortField]> {
        None
    }
    /// Java's reader has no cache helper.
    fn core_cache_helper(&self) -> Option<&CacheHelper> {
        None
    }
}
