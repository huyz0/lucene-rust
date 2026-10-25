//! `LeafReader` access to the SimpleText component readers. Terms use the
//! codec's FST and enums; postings have the same owned, materialized shape
//! as the binary segment's generic `TermsEnum` adapter.
//! Differs: numeric/binary values are decoded on the fallible positioning
//! calls because the generic reader's value getters are infallible.

use lucene_codecs::postings::Position;
use lucene_codecs::simpletext::{self as text, fields_reader};

use super::{
    BinaryDocValues, DocIdSetIterator, DocValuesIterator, NumericDocValues, SortedDocValues,
    SortedNumericDocValues, SortedSetDocValues, NO_MORE_DOCS,
};
use super::{MaterializedPostings, PostingsEnum, PostingsFlags, Terms, TermsEnum};
use crate::{Error, Result};
use text::doc_values_reader::{
    SimpleTextBinaryDocValues, SimpleTextNumericDocValues, SimpleTextSortedDocValues,
    SimpleTextSortedNumericDocValues, SimpleTextSortedSetDocValues,
};

fn error(error: text::Error) -> Error {
    crate::directory_reader::Error::SimpleText(error).into()
}

pub(super) struct SimpleTextTerms<'a>(pub(super) &'a fields_reader::SimpleTextTerms);

impl Terms for SimpleTextTerms<'_> {
    fn iterator(&self) -> Result<Box<dyn TermsEnum + '_>> {
        Ok(Box::new(SimpleTextTermsEnum {
            cursor: self.0.iterator(),
            positioned: false,
            has_positions: self.0.has_positions(),
        }))
    }
    fn size(&self) -> i64 {
        self.0.size()
    }
    fn sum_total_term_freq(&self) -> i64 {
        self.0.sum_total_term_freq()
    }
    fn sum_doc_freq(&self) -> i64 {
        self.0.sum_doc_freq()
    }
    fn doc_count(&self) -> i32 {
        self.0.doc_count()
    }
    fn has_freqs(&self) -> bool {
        self.0.has_freqs()
    }
    fn has_positions(&self) -> bool {
        self.0.has_positions()
    }
    fn has_offsets(&self) -> bool {
        self.0.has_offsets()
    }
    fn has_payloads(&self) -> bool {
        self.0.has_payloads()
    }
}

struct SimpleTextTermsEnum<'a> {
    cursor: fields_reader::SimpleTextTermsEnum<'a>,
    positioned: bool,
    has_positions: bool,
}

impl SimpleTextTermsEnum<'_> {
    fn check_positioned(&self) -> Result<()> {
        if !self.positioned {
            return Err(Error::IllegalState(
                "the TermsEnum is not positioned on a term".into(),
            ));
        }
        Ok(())
    }
}

impl TermsEnum for SimpleTextTermsEnum<'_> {
    fn next(&mut self) -> Result<Option<&[u8]>> {
        let term = self.cursor.next().map_err(error)?;
        self.positioned = term.is_some();
        Ok(term)
    }
    fn term(&self) -> Option<&[u8]> {
        self.positioned.then(|| self.cursor.term())
    }
    fn try_seek_ceil(&mut self, target: &[u8]) -> Result<text::SeekStatus> {
        let status = self.cursor.seek_ceil(target).map_err(error)?;
        self.positioned = status != text::SeekStatus::End;
        Ok(status)
    }
    fn try_seek_exact(&mut self, target: &[u8]) -> Result<bool> {
        self.positioned = self.cursor.seek_exact(target).map_err(error)?;
        Ok(self.positioned)
    }
    fn doc_freq(&mut self) -> Result<i32> {
        self.check_positioned()?;
        Ok(self.cursor.doc_freq())
    }
    fn total_term_freq(&mut self) -> Result<i64> {
        self.check_positioned()?;
        Ok(self.cursor.total_term_freq())
    }
    fn postings(&mut self, flags: PostingsFlags) -> Result<Box<dyn PostingsEnum>> {
        self.check_positioned()?;
        let raw_flags = match flags {
            PostingsFlags::None => text::NONE,
            PostingsFlags::Freqs => text::FREQS,
            PostingsFlags::Positions => text::POSITIONS,
            PostingsFlags::Offsets => text::OFFSETS,
            PostingsFlags::Payloads => text::PAYLOADS,
            PostingsFlags::All => text::ALL,
        };
        let mut postings = self.cursor.postings(raw_flags).map_err(error)?;
        let mut docs = Vec::new();
        let mut freqs = Vec::new();
        let mut positions = (flags.wants_positions() && self.has_positions).then(Vec::new);
        loop {
            let doc = postings.next_doc().map_err(error)?;
            if doc == text::NO_MORE_DOCS {
                break;
            }
            docs.push(doc);
            let freq = if flags == PostingsFlags::None {
                1
            } else {
                postings.freq()
            };
            freqs.push(freq);
            if let Some(positions) = &mut positions {
                let mut occurrences = Vec::new();
                for _ in 0..freq {
                    let position = postings.next_position().map_err(error)?;
                    occurrences.push(Position {
                        position,
                        start_offset: postings.start_offset(),
                        end_offset: postings.end_offset(),
                        payload: postings.payload().unwrap_or_default().to_vec(),
                    });
                }
                positions.push(occurrences);
            }
        }
        Ok(Box::new(MaterializedPostings::new(docs, freqs, positions)?))
    }
    fn ord(&self) -> Result<i64> {
        self.cursor.ord().map_err(error)
    }
    fn seek_exact_ord(&mut self, ord: i64) -> Result<()> {
        self.cursor.seek_exact_ord(ord).map_err(error)
    }
}

// The generic reader's numeric/binary getters are infallible. Decode at
// the fallible positioning call so malformed values still reach the caller.
macro_rules! iterator {
    ($ty:ident) => {
        impl DocIdSetIterator for $ty<'_> {
            fn doc_id(&self) -> i32 {
                self.cursor.doc_id()
            }
            fn cost(&self) -> i64 {
                self.cursor.cost()
            }
            fn next_doc(&mut self) -> Result<i32> {
                let doc = self.cursor.next_doc().map_err(error)?;
                if doc != NO_MORE_DOCS {
                    self.load()?;
                } else {
                    self.miss();
                }
                Ok(doc)
            }
            fn advance(&mut self, target: i32) -> Result<i32> {
                let doc = self.cursor.advance(target).map_err(error)?;
                if doc != NO_MORE_DOCS {
                    self.load()?;
                } else {
                    self.miss();
                }
                Ok(doc)
            }
        }
        impl DocValuesIterator for $ty<'_> {
            fn advance_exact(&mut self, target: i32) -> Result<bool> {
                let found = self.cursor.advance_exact(target).map_err(error)?;
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

pub(super) struct Numeric<'a> {
    cursor: SimpleTextNumericDocValues<'a>,
    value: i64,
}
impl<'a> Numeric<'a> {
    pub(super) fn new(cursor: SimpleTextNumericDocValues<'a>) -> Self {
        Self { cursor, value: 0 }
    }
    fn load(&mut self) -> Result<()> {
        self.value = self.cursor.long_value().map_err(error)?;
        Ok(())
    }
    fn miss(&mut self) {
        self.value = 0;
    }
}
iterator!(Numeric);
impl NumericDocValues for Numeric<'_> {
    fn long_value(&self) -> i64 {
        self.value
    }
}

pub(super) struct Binary<'a> {
    cursor: SimpleTextBinaryDocValues<'a>,
    value: Vec<u8>,
}
impl<'a> Binary<'a> {
    pub(super) fn new(cursor: SimpleTextBinaryDocValues<'a>) -> Self {
        Self {
            cursor,
            value: Vec::new(),
        }
    }
    fn load(&mut self) -> Result<()> {
        self.value = self.cursor.binary_value().map_err(error)?;
        Ok(())
    }
    fn miss(&mut self) {
        self.value.clear();
    }
}
iterator!(Binary);
impl BinaryDocValues for Binary<'_> {
    fn binary_value(&self) -> &[u8] {
        &self.value
    }
}

pub(super) struct Sorted<'a> {
    cursor: SimpleTextSortedDocValues<'a>,
}
impl<'a> Sorted<'a> {
    pub(super) fn new(cursor: SimpleTextSortedDocValues<'a>) -> Self {
        Self { cursor }
    }
    fn load(&mut self) -> Result<()> {
        Ok(())
    }
    fn miss(&mut self) {}
}
iterator!(Sorted);
impl SortedDocValues for Sorted<'_> {
    fn ord_value(&self) -> i32 {
        self.cursor.ord_value()
    }
    fn value_count(&self) -> i32 {
        self.cursor.value_count()
    }
    fn lookup_ord(&mut self, ord: i32) -> Result<Vec<u8>> {
        self.cursor
            .lookup_ord(ord)
            .map(<[u8]>::to_vec)
            .map_err(error)
    }
    // SENTINEL: negative = absent, `-insertionPoint-1`.
    fn lookup_term(&mut self, key: &[u8]) -> Result<i32> {
        Ok(match self.cursor.lookup_term(key).map_err(error)? {
            Ok(ord) => ord,
            Err(insertion) => insertion.saturating_neg().saturating_sub(1),
        })
    }
}

pub(super) struct SortedNumeric<'a> {
    cursor: SimpleTextSortedNumericDocValues<'a>,
    count: i32,
}
impl<'a> SortedNumeric<'a> {
    pub(super) fn new(cursor: SimpleTextSortedNumericDocValues<'a>) -> Self {
        Self { cursor, count: 0 }
    }
    fn load(&mut self) -> Result<()> {
        self.count = value_count(self.cursor.doc_value_count())?;
        Ok(())
    }
    fn miss(&mut self) {
        self.count = 0;
    }
}
iterator!(SortedNumeric);
impl SortedNumericDocValues for SortedNumeric<'_> {
    fn doc_value_count(&self) -> i32 {
        self.count
    }
    fn next_value(&mut self) -> Result<i64> {
        self.cursor.next_value().map_err(error)
    }
}

pub(super) struct SortedSet<'a> {
    cursor: SimpleTextSortedSetDocValues<'a>,
    count: i32,
}
impl<'a> SortedSet<'a> {
    pub(super) fn new(cursor: SimpleTextSortedSetDocValues<'a>) -> Self {
        Self { cursor, count: 0 }
    }
    fn load(&mut self) -> Result<()> {
        self.count = value_count(self.cursor.doc_value_count())?;
        Ok(())
    }
    fn miss(&mut self) {
        self.count = 0;
    }
}
iterator!(SortedSet);
impl SortedSetDocValues for SortedSet<'_> {
    fn doc_value_count(&self) -> i32 {
        self.count
    }
    fn next_ord(&mut self) -> Result<i64> {
        self.cursor.next_ord().map_err(error)
    }
    fn value_count(&self) -> i64 {
        self.cursor.value_count()
    }
    fn lookup_ord(&mut self, ord: i64) -> Result<Vec<u8>> {
        self.cursor
            .lookup_ord(ord)
            .map(<[u8]>::to_vec)
            .map_err(error)
    }
    // SENTINEL: negative = absent, `-insertionPoint-1`.
    fn lookup_term(&mut self, key: &[u8]) -> Result<i64> {
        Ok(match self.cursor.lookup_term(key).map_err(error)? {
            Ok(ord) => ord,
            Err(insertion) => insertion.saturating_neg().saturating_sub(1),
        })
    }
}

fn value_count(count: usize) -> Result<i32> {
    i32::try_from(count).map_err(|_| {
        lucene_store::Error::Corrupted(format!("SimpleText value count {count} exceeds i32")).into()
    })
}

pub(super) struct Points<'a>(pub(super) text::bkd_reader::SimpleTextBkdReader<'a>);

pub(super) struct FloatVectors<'a>(
    pub(super) text::knn_vectors_reader::SimpleTextVectorValues<'a, f32>,
);
pub(super) struct ByteVectors<'a>(
    pub(super) text::knn_vectors_reader::SimpleTextVectorValues<'a, u8>,
);

fn vector_ord_error(ord: i32) -> Error {
    Error::IllegalArgument(format!("vector ordinal {ord} out of bounds"))
}

impl super::FloatVectorValues for FloatVectors<'_> {
    fn dimension(&self) -> usize {
        self.0.dimension() as usize
    }
    fn size(&self) -> i32 {
        self.0.size() as i32
    }
    fn ord_to_doc(&self, ord: i32) -> Result<i32> {
        usize::try_from(ord)
            .ok()
            .and_then(|ord| self.0.ord_to_doc(ord))
            .ok_or_else(|| vector_ord_error(ord))
    }
    fn vector_value(&self, ord: i32) -> Result<Vec<f32>> {
        usize::try_from(ord)
            .ok()
            .and_then(|ord| self.0.vector_value(ord))
            .map(<[f32]>::to_vec)
            .ok_or_else(|| vector_ord_error(ord))
    }
}

impl super::ByteVectorValues for ByteVectors<'_> {
    fn dimension(&self) -> usize {
        self.0.dimension() as usize
    }
    fn size(&self) -> i32 {
        self.0.size() as i32
    }
    fn ord_to_doc(&self, ord: i32) -> Result<i32> {
        usize::try_from(ord)
            .ok()
            .and_then(|ord| self.0.ord_to_doc(ord))
            .ok_or_else(|| vector_ord_error(ord))
    }
    fn vector_value(&self, ord: i32) -> Result<Vec<u8>> {
        usize::try_from(ord)
            .ok()
            .and_then(|ord| self.0.vector_value(ord))
            .map(<[u8]>::to_vec)
            .ok_or_else(|| vector_ord_error(ord))
    }
}

impl super::PointValues for Points<'_> {
    fn intersect(&self, visitor: &mut dyn super::IntersectVisitor) -> Result<()> {
        self.0
            .intersect(&mut super::DynVisitor(visitor))
            .map_err(error)
    }
    fn estimate_point_count(&self, visitor: &mut dyn super::IntersectVisitor) -> Result<i64> {
        self.0
            .estimate_point_count(&mut super::DynVisitor(visitor))
            .map_err(error)
    }
    fn min_packed_value(&self) -> &[u8] {
        self.0.min_packed_value()
    }
    fn max_packed_value(&self) -> &[u8] {
        self.0.max_packed_value()
    }
    fn num_dimensions(&self) -> i32 {
        self.0.num_dimensions() as i32
    }
    fn num_index_dimensions(&self) -> i32 {
        self.0.num_index_dimensions() as i32
    }
    fn bytes_per_dimension(&self) -> i32 {
        self.0.bytes_per_dimension() as i32
    }
    fn size(&self) -> i64 {
        self.0.size()
    }
    fn doc_count(&self) -> i32 {
        self.0.doc_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_codecs::field_infos::{DocValuesType, FieldInfo};
    use text::doc_values_reader::SimpleTextDocValuesReader;
    use text::doc_values_writer::SimpleTextDocValuesWriter;

    fn numeric_file() -> Vec<u8> {
        let mut field = FieldInfo::new("n", 0);
        field.doc_values_type = DocValuesType::Numeric;
        let mut writer = SimpleTextDocValuesWriter::new(2);
        writer.add_numeric_field(&field, &[(0, 7)]).unwrap();
        writer.finish().unwrap()
    }

    #[test]
    fn numeric_getter_caches_values_and_clears_missing_documents() {
        let bytes = numeric_file();
        let reader = SimpleTextDocValuesReader::open(&bytes, 2).unwrap();
        let mut values = Numeric::new(reader.numeric("n").unwrap());
        assert!(values.advance_exact(0).unwrap());
        assert_eq!(values.long_value(), 7);
        assert!(!values.advance_exact(1).unwrap());
        assert_eq!(values.long_value(), 0);
        assert_eq!(values.next_doc().unwrap(), NO_MORE_DOCS);
        assert_eq!(values.long_value(), 0);
    }

    #[test]
    fn malformed_numeric_values_fail_at_every_positioning_entry_point() {
        let mut bytes = numeric_file();
        let reader = SimpleTextDocValuesReader::open(&bytes, 2)
            .unwrap()
            .rebind(&[]);
        bytes[reader.field("n").unwrap().data_start] = b'x';
        let mut next = Numeric::new(reader.numeric_from_data("n", &bytes).unwrap());
        assert!(next.next_doc().is_err());
        let mut advance = Numeric::new(reader.numeric_from_data("n", &bytes).unwrap());
        assert!(advance.advance(0).is_err());
        let mut exact = Numeric::new(reader.numeric_from_data("n", &bytes).unwrap());
        assert!(exact.advance_exact(0).is_err());
    }

    #[test]
    fn generic_value_count_refuses_unrepresentable_counts() {
        assert_eq!(value_count(0).unwrap(), 0);
        assert_eq!(value_count(i32::MAX as usize).unwrap(), i32::MAX);
        assert!(value_count((i32::MAX as usize).saturating_add(1)).is_err());
    }
}
