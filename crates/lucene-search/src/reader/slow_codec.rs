//! Ports of `org.apache.lucene.index.SlowCodecReaderWrapper` (any leaf as a
//! [`CodecReader`], what `addIndexes(CodecReader...)` takes) and
//! `SlowCompositeCodecReaderWrapper` (several codec readers as one, their
//! documents numbered one after another -- what `addIndexes` and
//! `OneMerge.reorder` read a whole merge through).
//!
//! The composite view is built from the multi-reader pieces: terms through
//! [`MultiTerms`], doc values and norms through [`multi_doc_values`] (sorted
//! ordinals through an ordinal map), live docs as one bitset, stored fields
//! and term vectors routed to the leaf holding the document, points and
//! vectors concatenated with each leaf's doc (and ordinal) base.
//!
//! # What differs from Java
//!
//! `SlowCodecReaderWrapper.wrap` returns its argument when that already is a
//! `CodecReader`; a `dyn LeafReader` cannot be asked that, so
//! [`SlowCodecReaderWrapper::wrap`] always wraps, and a caller holding a
//! [`CodecReader`] simply uses it.

use std::sync::Arc;

use lucene_codecs::field_infos::FieldInfos;
use lucene_index::segment_info::IndexSortField;
use lucene_util::fixed_bit_set::FixedBitSet;

use super::multi_doc_values;
use super::multi_reader::MultiReader;
use super::{
    merged_field_infos, remap_term_vectors, BinaryDocValues, ByteVectorValues, CacheHelper,
    CodecReader, FloatVectorValues, IndexReader, IntersectVisitor, LeafReader, LeafReaderContext,
    NumericDocValues, PointValues, ReaderHandle, Relation, RemappingVisitor, SortedDocValues,
    SortedNumericDocValues, SortedSetDocValues, StoredFieldVisitor, TermVectorsDocument, Terms,
};
use crate::multi_terms::MultiTerms;
use crate::{Error, Result};

/// `SlowCodecReaderWrapper`: a leaf as a [`CodecReader`], every read
/// delegated.
pub struct SlowCodecReaderWrapper {
    in_: Arc<dyn LeafReader>,
}

impl SlowCodecReaderWrapper {
    /// `SlowCodecReaderWrapper.wrap(reader)`.
    pub fn wrap(reader: Arc<dyn LeafReader>) -> Arc<dyn CodecReader> {
        Arc::new(Self { in_: reader })
    }
}

impl IndexReader for SlowCodecReaderWrapper {
    fn max_doc(&self) -> i32 {
        self.in_.max_doc()
    }
    fn num_docs(&self) -> i32 {
        self.in_.num_docs()
    }
    fn leaves(&self) -> Vec<LeafReaderContext<'_>> {
        vec![LeafReaderContext {
            reader: self,
            ord: 0,
            doc_base: 0,
        }]
    }
    fn reader_cache_helper(&self) -> Option<&CacheHelper> {
        self.in_.reader_cache_helper()
    }
}

impl LeafReader for SlowCodecReaderWrapper {
    fn field_infos(&self) -> &FieldInfos {
        self.in_.field_infos()
    }
    fn live_docs(&self) -> Option<&FixedBitSet> {
        self.in_.live_docs()
    }
    fn terms(&self, field: &str) -> Result<Option<Box<dyn Terms + '_>>> {
        self.in_.terms(field)
    }
    fn numeric_doc_values(&self, field: &str) -> Result<Option<Box<dyn NumericDocValues + '_>>> {
        self.in_.numeric_doc_values(field)
    }
    fn binary_doc_values(&self, field: &str) -> Result<Option<Box<dyn BinaryDocValues + '_>>> {
        self.in_.binary_doc_values(field)
    }
    fn sorted_doc_values(&self, field: &str) -> Result<Option<Box<dyn SortedDocValues + '_>>> {
        self.in_.sorted_doc_values(field)
    }
    fn sorted_numeric_doc_values(
        &self,
        field: &str,
    ) -> Result<Option<Box<dyn SortedNumericDocValues + '_>>> {
        self.in_.sorted_numeric_doc_values(field)
    }
    fn sorted_set_doc_values(
        &self,
        field: &str,
    ) -> Result<Option<Box<dyn SortedSetDocValues + '_>>> {
        self.in_.sorted_set_doc_values(field)
    }
    fn norm_values(&self, field: &str) -> Result<Option<Box<dyn NumericDocValues + '_>>> {
        self.in_.norm_values(field)
    }
    fn point_values(&self, field: &str) -> Result<Option<Box<dyn PointValues + '_>>> {
        self.in_.point_values(field)
    }
    fn float_vector_values(&self, field: &str) -> Result<Option<Box<dyn FloatVectorValues + '_>>> {
        self.in_.float_vector_values(field)
    }
    fn byte_vector_values(&self, field: &str) -> Result<Option<Box<dyn ByteVectorValues + '_>>> {
        self.in_.byte_vector_values(field)
    }
    fn document(&self, doc: i32, visitor: &mut dyn StoredFieldVisitor) -> Result<()> {
        self.in_.document(doc, visitor)
    }
    fn term_vectors(&self, doc: i32) -> Result<Option<TermVectorsDocument>> {
        self.in_.term_vectors(doc)
    }
    fn index_sort(&self) -> Option<&[IndexSortField]> {
        self.in_.index_sort()
    }
    fn core_cache_helper(&self) -> Option<&CacheHelper> {
        self.in_.core_cache_helper()
    }
    fn doc_values_skipper(
        &self,
        field: &str,
    ) -> Result<Option<lucene_codecs::doc_values::DocValuesSkipper<'_>>> {
        self.in_.doc_values_skipper(field)
    }
    fn check_integrity(&self) -> Result<()> {
        self.in_.check_integrity()
    }
}

impl CodecReader for SlowCodecReaderWrapper {}

/// `SlowCompositeCodecReaderWrapper`.
pub struct SlowCompositeCodecReaderWrapper {
    readers: Vec<Arc<dyn CodecReader>>,
    /// The readers as a [`MultiReader`], which the doc-values views read.
    multi: MultiReader,
    /// `docStarts`: each reader's first document, then `maxDoc`.
    doc_starts: Vec<i32>,
    field_infos: FieldInfos,
    live_docs: Option<FixedBitSet>,
}

impl SlowCompositeCodecReaderWrapper {
    /// `SlowCompositeCodecReaderWrapper.wrap(readers)`: one reader is
    /// returned as is.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] for no readers at all, or more documents
    /// than a reader may hold.
    pub fn wrap(readers: Vec<Arc<dyn CodecReader>>) -> Result<Arc<dyn CodecReader>> {
        match readers.len() {
            0 => Err(Error::IllegalArgument(
                "Must take at least one reader, got 0".into(),
            )),
            1 => Ok(readers.into_iter().next().expect("one reader")),
            _ => Ok(Arc::new(Self::new(readers)?)),
        }
    }

    fn new(readers: Vec<Arc<dyn CodecReader>>) -> Result<Self> {
        let multi = MultiReader::new(
            readers
                .iter()
                .map(|r| ReaderHandle::Leaf(Arc::clone(r) as Arc<dyn LeafReader>))
                .collect(),
        )?;
        let leaves = multi.leaves();
        let mut doc_starts: Vec<i32> = leaves.iter().map(|l| l.doc_base).collect();
        doc_starts.push(multi.max_doc());
        let field_infos = merged_field_infos(&leaves);
        // `MultiBits.getLiveDocs`: none unless some reader has deletions.
        let live_docs = if leaves.iter().any(|l| l.reader.live_docs().is_some()) {
            let max_doc = usize::try_from(multi.max_doc()).unwrap_or(0);
            let mut bits = FixedBitSet::new(max_doc);
            for leaf in &leaves {
                let base = leaf.doc_base as usize;
                let n = leaf.reader.max_doc() as usize;
                let live = leaf.reader.live_docs();
                for d in 0..n {
                    if live.is_none_or(|l| l.get_doc(d as i32)) && base + d < bits.len() {
                        bits.set(base + d);
                    }
                }
            }
            Some(bits)
        } else {
            None
        };
        drop(leaves);
        Ok(Self {
            readers,
            multi,
            doc_starts,
            field_infos,
            live_docs,
        })
    }

    /// `docIdToReaderId(doc)` and the document in that reader's space.
    fn locate(&self, doc: i32) -> Result<(&dyn CodecReader, usize, i32)> {
        let n = self.readers.len();
        if doc < 0 || doc >= self.doc_starts[n] {
            return Err(Error::IllegalArgument(format!(
                "docID {doc} out of bounds for length {}",
                self.doc_starts[n]
            )));
        }
        let i = self.doc_starts[..n]
            .partition_point(|&s| s <= doc)
            .saturating_sub(1);
        Ok((self.readers[i].as_ref(), i, doc - self.doc_starts[i]))
    }
}

impl IndexReader for SlowCompositeCodecReaderWrapper {
    fn max_doc(&self) -> i32 {
        self.multi.max_doc()
    }
    fn num_docs(&self) -> i32 {
        self.multi.num_docs()
    }
    fn leaves(&self) -> Vec<LeafReaderContext<'_>> {
        vec![LeafReaderContext {
            reader: self,
            ord: 0,
            doc_base: 0,
        }]
    }
    fn reader_cache_helper(&self) -> Option<&CacheHelper> {
        None
    }
}

impl LeafReader for SlowCompositeCodecReaderWrapper {
    fn field_infos(&self) -> &FieldInfos {
        &self.field_infos
    }
    fn live_docs(&self) -> Option<&FixedBitSet> {
        self.live_docs.as_ref()
    }
    fn terms(&self, field: &str) -> Result<Option<Box<dyn Terms + '_>>> {
        let mut subs = Vec::new();
        for (i, r) in self.readers.iter().enumerate() {
            if let Some(t) = r.terms(field)? {
                subs.push((t, self.doc_starts[i]));
            }
        }
        Ok(if subs.is_empty() {
            None
        } else {
            Some(Box::new(MultiTerms::new(subs)))
        })
    }
    fn numeric_doc_values(&self, field: &str) -> Result<Option<Box<dyn NumericDocValues + '_>>> {
        multi_doc_values::numeric_values(&self.multi, field)
    }
    fn binary_doc_values(&self, field: &str) -> Result<Option<Box<dyn BinaryDocValues + '_>>> {
        multi_doc_values::binary_values(&self.multi, field)
    }
    fn sorted_doc_values(&self, field: &str) -> Result<Option<Box<dyn SortedDocValues + '_>>> {
        multi_doc_values::sorted_values(&self.multi, field)
    }
    fn sorted_numeric_doc_values(
        &self,
        field: &str,
    ) -> Result<Option<Box<dyn SortedNumericDocValues + '_>>> {
        multi_doc_values::sorted_numeric_values(&self.multi, field)
    }
    fn sorted_set_doc_values(
        &self,
        field: &str,
    ) -> Result<Option<Box<dyn SortedSetDocValues + '_>>> {
        multi_doc_values::sorted_set_values(&self.multi, field)
    }
    fn norm_values(&self, field: &str) -> Result<Option<Box<dyn NumericDocValues + '_>>> {
        multi_doc_values::norm_values(&self.multi, field)
    }
    fn point_values(&self, field: &str) -> Result<Option<Box<dyn PointValues + '_>>> {
        let mut subs = Vec::new();
        for (i, r) in self.readers.iter().enumerate() {
            let has = r
                .field_infos()
                .field_by_name(field)
                .is_some_and(|fi| fi.point_dimension_count > 0);
            if !has {
                continue;
            }
            if let Some(v) = r.point_values(field)? {
                subs.push((v, self.doc_starts[i]));
            }
        }
        if subs.is_empty() {
            return Ok(None);
        }
        Ok(Some(Box::new(MergedPoints::new(subs))))
    }
    fn float_vector_values(&self, field: &str) -> Result<Option<Box<dyn FloatVectorValues + '_>>> {
        let mut subs = Vec::new();
        for (i, r) in self.readers.iter().enumerate() {
            if let Some(v) = r.float_vector_values(field)? {
                subs.push((v, self.doc_starts[i]));
            }
        }
        Ok(MergedVectors::<dyn FloatVectorValues>::new(subs)
            .map(|m| Box::new(m) as Box<dyn FloatVectorValues + '_>))
    }
    fn byte_vector_values(&self, field: &str) -> Result<Option<Box<dyn ByteVectorValues + '_>>> {
        let mut subs = Vec::new();
        for (i, r) in self.readers.iter().enumerate() {
            if let Some(v) = r.byte_vector_values(field)? {
                subs.push((v, self.doc_starts[i]));
            }
        }
        Ok(MergedVectors::<dyn ByteVectorValues>::new(subs)
            .map(|m| Box::new(m) as Box<dyn ByteVectorValues + '_>))
    }
    fn document(&self, doc: i32, visitor: &mut dyn StoredFieldVisitor) -> Result<()> {
        let (r, _, local) = self.locate(doc)?;
        let mut remap = RemappingVisitor {
            inner: visitor,
            from: r.field_infos(),
            to: &self.field_infos,
        };
        r.document(local, &mut remap)
    }
    fn term_vectors(&self, doc: i32) -> Result<Option<TermVectorsDocument>> {
        let (r, _, local) = self.locate(doc)?;
        Ok(r.term_vectors(local)?
            .map(|d| remap_term_vectors(d, r.field_infos(), &self.field_infos)))
    }
    fn index_sort(&self) -> Option<&[IndexSortField]> {
        None
    }
    fn core_cache_helper(&self) -> Option<&CacheHelper> {
        None
    }
    fn check_integrity(&self) -> Result<()> {
        for r in &self.readers {
            r.check_integrity()?;
        }
        Ok(())
    }
}

impl CodecReader for SlowCompositeCodecReaderWrapper {}

/// The merged `PointValues`: one tree node holding every reader's points.
struct MergedPoints<'a> {
    subs: Vec<(Box<dyn PointValues + 'a>, i32)>,
    min: Vec<u8>,
    max: Vec<u8>,
}

impl<'a> MergedPoints<'a> {
    fn new(subs: Vec<(Box<dyn PointValues + 'a>, i32)>) -> Self {
        let mut min: Option<Vec<u8>> = None;
        let mut max: Option<Vec<u8>> = None;
        for (s, _) in &subs {
            let bpd = s.bytes_per_dimension().max(0) as usize;
            let dims = s.num_index_dimensions().max(0) as usize;
            fold(
                &mut min,
                s.min_packed_value(),
                dims,
                bpd,
                std::cmp::Ordering::Less,
            );
            fold(
                &mut max,
                s.max_packed_value(),
                dims,
                bpd,
                std::cmp::Ordering::Greater,
            );
        }
        Self {
            subs,
            min: min.unwrap_or_default(),
            max: max.unwrap_or_default(),
        }
    }

    fn size(&self) -> i64 {
        self.subs
            .iter()
            .fold(0i64, |a, (s, _)| a.saturating_add(s.size()))
    }
}

/// Per index dimension, keeps the smaller (`Less`) or larger (`Greater`) of
/// the two values, as `getMinPackedValue`/`getMaxPackedValue` do.
fn fold(acc: &mut Option<Vec<u8>>, v: &[u8], dims: usize, bpd: usize, keep: std::cmp::Ordering) {
    let Some(a) = acc else {
        *acc = Some(v.to_vec());
        return;
    };
    for d in 0..dims {
        let r = d * bpd..(d + 1) * bpd;
        if let (Some(x), Some(y)) = (v.get(r.clone()), a.get(r.clone())) {
            if x.cmp(y) == keep {
                a[r].copy_from_slice(x);
            }
        }
        // (A dimension missing from either value -- impossible for points of
        // one field -- is left as it is.)
    }
}

/// Every point of a reader, doc ids shifted by its base: `visitDocValues`
/// (`inside == false`) or `visitDocIDs` (`inside == true`).
struct ShiftAll<'v> {
    inner: &'v mut dyn IntersectVisitor,
    base: i32,
    inside: bool,
}

impl IntersectVisitor for ShiftAll<'_> {
    fn compare(&mut self, _min: &[u8], _max: &[u8]) -> Relation {
        if self.inside {
            Relation::CellInsideQuery
        } else {
            Relation::CellCrossesQuery
        }
    }
    fn visit(&mut self, doc_id: i32) {
        self.inner.visit(doc_id + self.base);
    }
    fn visit_with_value(&mut self, doc_id: i32, packed_value: &[u8]) {
        self.inner
            .visit_with_value(doc_id + self.base, packed_value);
    }
}

impl PointValues for MergedPoints<'_> {
    /// `PointValues.intersect` over the one-node tree: the visitor compares
    /// the merged bounds, then sees every document (inside) or every point
    /// (crossing).
    fn intersect(&self, visitor: &mut dyn IntersectVisitor) -> Result<()> {
        let inside = match visitor.compare(&self.min, &self.max) {
            Relation::CellOutsideQuery => return Ok(()),
            Relation::CellInsideQuery => true,
            Relation::CellCrossesQuery => false,
        };
        for (s, base) in &self.subs {
            s.intersect(&mut ShiftAll {
                inner: &mut *visitor,
                base: *base,
                inside,
            })?;
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
        self.subs[0].0.num_dimensions()
    }
    fn num_index_dimensions(&self) -> i32 {
        self.subs[0].0.num_index_dimensions()
    }
    fn bytes_per_dimension(&self) -> i32 {
        self.subs[0].0.bytes_per_dimension()
    }
    fn size(&self) -> i64 {
        MergedPoints::size(self)
    }
    fn doc_count(&self) -> i32 {
        self.subs
            .iter()
            .fold(0i32, |a, (s, _)| a.saturating_add(s.doc_count()))
    }
}

/// `MergedFloatVectorValues`/`MergedByteVectorValues`: every reader's
/// vectors, ordinals concatenated in reader order.
struct MergedVectors<T: ?Sized> {
    subs: Vec<(Box<T>, i32)>,
    /// Each sub's first ordinal, then the total size.
    ord_starts: Vec<i32>,
    dimension: usize,
}

macro_rules! merged_vectors {
    ($tr:ident, $elem:ty) => {
        impl<'a> MergedVectors<dyn $tr + 'a> {
            fn new(subs: Vec<(Box<dyn $tr + 'a>, i32)>) -> Option<Self> {
                let dimension = subs.first()?.0.dimension();
                let mut ord_starts = Vec::with_capacity(subs.len() + 1);
                let mut total = 0i32;
                for (s, _) in &subs {
                    ord_starts.push(total);
                    total = total.saturating_add(s.size());
                }
                ord_starts.push(total);
                Some(Self {
                    subs,
                    ord_starts,
                    dimension,
                })
            }

            fn locate(&self, ord: i32) -> Result<(&(Box<dyn $tr + 'a>, i32), i32)> {
                let n = self.subs.len();
                if ord < 0 || ord >= self.ord_starts[n] {
                    return Err(Error::IllegalArgument(format!(
                        "vector ordinal {ord} out of range"
                    )));
                }
                let i = self.ord_starts[..n]
                    .partition_point(|&s| s <= ord)
                    .saturating_sub(1);
                Ok((&self.subs[i], ord - self.ord_starts[i]))
            }
        }

        impl<'a> $tr for MergedVectors<dyn $tr + 'a> {
            fn dimension(&self) -> usize {
                self.dimension
            }
            fn size(&self) -> i32 {
                self.ord_starts[self.subs.len()]
            }
            fn ord_to_doc(&self, ord: i32) -> Result<i32> {
                let ((sub, base), local) = self.locate(ord)?;
                Ok(sub.ord_to_doc(local)? + base)
            }
            fn vector_value(&self, ord: i32) -> Result<Vec<$elem>> {
                let ((sub, _), local) = self.locate(ord)?;
                sub.vector_value(local)
            }
        }
    };
}

merged_vectors!(FloatVectorValues, f32);
merged_vectors!(ByteVectorValues, u8);
