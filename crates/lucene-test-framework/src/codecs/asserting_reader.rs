//! Java `AssertingCodec`'s read side: the producers its formats open for a
//! segment written under it, checking each answer as
//! `AssertingPostingsFormat.AssertingFieldsProducer`,
//! `AssertingDocValuesFormat.AssertingDocValuesProducer`,
//! `AssertingNormsFormat.AssertingNormsProducer`,
//! `AssertingPointsFormat.AssertingPointsReader`,
//! `AssertingKnnVectorsFormat.AssertingKnnVectorsReader`,
//! `AssertingStoredFieldsFormat.AssertingStoredFieldsReader` and
//! `AssertingLiveDocsFormat.readLiveDocs` do, reached the way Java's
//! `SegmentReader` (a `CodecReader`) reaches them.
//!
//! [`AssertingCodecReader::open`] opens an index and gives every segment
//! whose codec is `Asserting` an [`AssertingCodecLeafReader`]; other
//! segments are read as they are, as Java reads them through their own
//! codec. The checks panic where Java's `assert` would throw.
//!
//! Differs:
//!
//! - The port's segment reader decodes its files itself, so the checks wrap
//!   what it answers rather than sitting between it and the codec's
//!   producers; a merge reads its sources through the index crate's own
//!   decoders, so no merge instances (`getMergeInstance`, `finishMerge`) are
//!   opened, and the merging-thread checks have nothing to guard.
//! - `assertThread` has no counterpart: the iterators borrow their reader
//!   and are not `Send`.
//! - Double `close()` is the reader's `Drop`.
//! - The vector values have no iterator, so Java's `iterator().docID() ==
//!   -1` check has none; a term-vectors document is owned data with no
//!   `Fields` to wrap (`AssertingFields`).
//! - Points: one asserting visitor spans each `intersect` walk, where Java
//!   wraps each `visitDocIDs`/`visitDocValues` call in a fresh one, so the
//!   cell-bounds, budget and one-dimensional order checks run across the
//!   whole walk.

use std::collections::HashSet;
use std::sync::Arc;

use lucene_codecs::field_infos::{
    DocValuesSkipIndexType, DocValuesType, FieldInfo, FieldInfos, IndexOptions, VectorEncoding,
};
use lucene_codecs::stored_fields::StoredFieldVisitor;
use lucene_index::segment_info::IndexSortField;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::reader::DocValuesSkipper;
use lucene_search::reader::{
    BinaryDocValues, ByteVectorValues, CacheHelper, CompositeReader, FloatVectorValues,
    IndexReader, IntersectVisitor, LeafReader, LeafReaderContext, NumericDocValues, PointValues,
    Relation, SortedDocValues, SortedNumericDocValues, SortedSetDocValues, SubReader,
    TermVectorsDocument, Terms,
};
use lucene_search::Result;
use lucene_store::Directory;
use lucene_util::fixed_bit_set::FixedBitSet;

use super::asserting_installed::NAME;
use crate::index::asserting_leaf_reader::{
    AssertingBinaryDocValues, AssertingDocValuesSkipper, AssertingNumericDocValues,
    AssertingSortedDocValues, AssertingSortedNumericDocValues, AssertingSortedSetDocValues,
    AssertingTerms,
};

/// A segment of an `Asserting`-coded index read through the asserting
/// producers.
pub struct AssertingCodecLeafReader {
    in_: Arc<dyn LeafReader>,
}

impl AssertingCodecLeafReader {
    /// The producers over `in_`, a segment written under `Asserting`.
    pub fn new(in_: Arc<dyn LeafReader>) -> Self {
        Self { in_ }
    }

    /// The wrapped segment.
    pub fn delegate(&self) -> &Arc<dyn LeafReader> {
        &self.in_
    }

    fn field(&self, field: &str) -> Option<&FieldInfo> {
        self.in_.field_infos().field_by_name(field)
    }

    /// `CodecReader.getXDocValues`: the producer is asked only for a field
    /// whose info has that doc-values type, and must then answer
    /// (`assert values != null`); for any other field the reader answers
    /// `null` itself.
    fn check_doc_values<T>(&self, field: &str, ty: DocValuesType, values: &Option<T>) {
        let expected = self.field(field).is_some_and(|fi| fi.doc_values_type == ty);
        if expected {
            assert!(
                values.is_some(),
                "the {ty:?} doc-values producer has no values for {field}"
            );
        } else {
            assert!(
                values.is_none(),
                "{ty:?} doc values for {field}, whose field info has none"
            );
        }
    }
}

/// `FieldInfo.hasNorms()`.
fn has_norms(fi: &FieldInfo) -> bool {
    fi.index_options != IndexOptions::None && !fi.omit_norms
}

impl IndexReader for AssertingCodecLeafReader {
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

impl LeafReader for AssertingCodecLeafReader {
    fn field_infos(&self) -> &FieldInfos {
        self.in_.field_infos()
    }

    /// `AssertingLiveDocsFormat.readLiveDocs`: the bits cover the segment and
    /// clear exactly its deleted documents.
    fn live_docs(&self) -> Option<&FixedBitSet> {
        let live = self.in_.live_docs();
        if let Some(bits) = live {
            let max_doc = self.max_doc();
            assert!(
                usize::try_from(max_doc).is_ok_and(|m| m == bits.len()),
                "maxDoc={max_doc} but live docs hold {} bits",
                bits.len()
            );
            let deleted = max_doc - self.num_docs();
            let cleared = bits.len() - bits.cardinality();
            assert!(
                usize::try_from(deleted).is_ok_and(|d| d == cleared),
                "{deleted} deleted documents but live docs clear {cleared}"
            );
        }
        live
    }

    /// `AssertingFieldsProducer.terms`.
    fn terms(&self, field: &str) -> Result<Option<Box<dyn Terms + '_>>> {
        Ok(self
            .in_
            .terms(field)?
            .map(|t| Box::new(AssertingTerms::new(t)) as Box<dyn Terms + '_>))
    }

    fn numeric_doc_values(&self, field: &str) -> Result<Option<Box<dyn NumericDocValues + '_>>> {
        let dv = self.in_.numeric_doc_values(field)?;
        self.check_doc_values(field, DocValuesType::Numeric, &dv);
        let max_doc = self.max_doc();
        Ok(dv.map(|dv| {
            Box::new(AssertingNumericDocValues::new(dv, max_doc)) as Box<dyn NumericDocValues + '_>
        }))
    }

    fn binary_doc_values(&self, field: &str) -> Result<Option<Box<dyn BinaryDocValues + '_>>> {
        let dv = self.in_.binary_doc_values(field)?;
        self.check_doc_values(field, DocValuesType::Binary, &dv);
        let max_doc = self.max_doc();
        Ok(dv.map(|dv| {
            Box::new(AssertingBinaryDocValues::new(dv, max_doc)) as Box<dyn BinaryDocValues + '_>
        }))
    }

    fn sorted_doc_values(&self, field: &str) -> Result<Option<Box<dyn SortedDocValues + '_>>> {
        let dv = self.in_.sorted_doc_values(field)?;
        self.check_doc_values(field, DocValuesType::Sorted, &dv);
        let max_doc = self.max_doc();
        Ok(dv.map(|dv| {
            Box::new(AssertingSortedDocValues::new(dv, max_doc)) as Box<dyn SortedDocValues + '_>
        }))
    }

    fn sorted_numeric_doc_values(
        &self,
        field: &str,
    ) -> Result<Option<Box<dyn SortedNumericDocValues + '_>>> {
        let dv = self.in_.sorted_numeric_doc_values(field)?;
        self.check_doc_values(field, DocValuesType::SortedNumeric, &dv);
        let max_doc = self.max_doc();
        Ok(dv.map(|dv| {
            Box::new(AssertingSortedNumericDocValues::new(dv, max_doc))
                as Box<dyn SortedNumericDocValues + '_>
        }))
    }

    fn sorted_set_doc_values(
        &self,
        field: &str,
    ) -> Result<Option<Box<dyn SortedSetDocValues + '_>>> {
        let dv = self.in_.sorted_set_doc_values(field)?;
        self.check_doc_values(field, DocValuesType::SortedSet, &dv);
        let max_doc = self.max_doc();
        Ok(dv.map(|dv| {
            Box::new(AssertingSortedSetDocValues::new(dv, max_doc))
                as Box<dyn SortedSetDocValues + '_>
        }))
    }

    /// `AssertingNormsProducer.getNorms`, asked only for a field with norms.
    fn norm_values(&self, field: &str) -> Result<Option<Box<dyn NumericDocValues + '_>>> {
        let norms = self.in_.norm_values(field)?;
        if self.field(field).is_some_and(has_norms) {
            assert!(
                norms.is_some(),
                "the norms producer has no norms for {field}"
            );
        } else {
            assert!(
                norms.is_none(),
                "norms for {field}, whose field info has none"
            );
        }
        let max_doc = self.max_doc();
        Ok(norms.map(|dv| {
            Box::new(AssertingNumericDocValues::new(dv, max_doc)) as Box<dyn NumericDocValues + '_>
        }))
    }

    /// `AssertingPointsReader.getValues`, asked only for a field with
    /// points; `AssertingPointValues` checks the statistics.
    fn point_values(&self, field: &str) -> Result<Option<Box<dyn PointValues + '_>>> {
        let values = self.in_.point_values(field)?;
        if values.is_some() {
            assert!(
                self.field(field)
                    .is_some_and(|fi| fi.point_dimension_count > 0),
                "point values for {field}, whose field info has no point dimensions"
            );
        }
        let max_doc = self.max_doc();
        Ok(values.map(|v| {
            Box::new(AssertingCodecPointValues::new(v, max_doc)) as Box<dyn PointValues + '_>
        }))
    }

    /// `AssertingKnnVectorsReader.getFloatVectorValues`.
    fn float_vector_values(&self, field: &str) -> Result<Option<Box<dyn FloatVectorValues + '_>>> {
        let values = self.in_.float_vector_values(field)?;
        let fi = self.field(field);
        if fi.is_some_and(|fi| {
            fi.vector_dimension > 0 && fi.vector_encoding == VectorEncoding::Float32
        }) {
            let v = values
                .as_ref()
                .unwrap_or_else(|| panic!("the vectors reader has no float vectors for {field}"));
            assert!(v.size() >= 0, "{field} holds {} vectors", v.size());
            assert!(v.dimension() > 0, "{field}'s vectors have no dimensions");
        } else {
            assert!(
                values.is_none(),
                "float vectors for {field}, whose field info has none"
            );
        }
        Ok(values)
    }

    /// `AssertingKnnVectorsReader.getByteVectorValues`.
    fn byte_vector_values(&self, field: &str) -> Result<Option<Box<dyn ByteVectorValues + '_>>> {
        let values = self.in_.byte_vector_values(field)?;
        let fi = self.field(field);
        if fi
            .is_some_and(|fi| fi.vector_dimension > 0 && fi.vector_encoding == VectorEncoding::Byte)
        {
            let v = values
                .as_ref()
                .unwrap_or_else(|| panic!("the vectors reader has no byte vectors for {field}"));
            assert!(v.size() >= 0, "{field} holds {} vectors", v.size());
            assert!(v.dimension() > 0, "{field}'s vectors have no dimensions");
        } else {
            assert!(
                values.is_none(),
                "byte vectors for {field}, whose field info has none"
            );
        }
        Ok(values)
    }

    /// `AssertingStoredFieldsReader.document`: `assert n >= 0 && n < maxDoc`.
    fn document(&self, doc: i32, visitor: &mut dyn StoredFieldVisitor) -> Result<()> {
        let max_doc = self.max_doc();
        assert!(
            doc >= 0 && doc < max_doc,
            "stored fields of doc {doc} of {max_doc}"
        );
        self.in_.document(doc, visitor)
    }

    /// `AssertingTermVectorsReader.get`.
    fn term_vectors(&self, doc: i32) -> Result<Option<TermVectorsDocument>> {
        self.in_.term_vectors(doc)
    }

    fn index_sort(&self) -> Option<&[IndexSortField]> {
        self.in_.index_sort()
    }

    fn core_cache_helper(&self) -> Option<&CacheHelper> {
        self.in_.core_cache_helper()
    }

    /// `AssertingDocValuesProducer.getSkipper`, asked only for a field with a
    /// skip index.
    fn doc_values_skipper(&self, field: &str) -> Result<Option<DocValuesSkipper<'_>>> {
        let skipper = self.in_.doc_values_skipper(field)?;
        let indexed = self
            .field(field)
            .is_some_and(|fi| fi.doc_values_skip_index_type != DocValuesSkipIndexType::None);
        if indexed {
            let skipper = skipper
                .unwrap_or_else(|| panic!("the doc-values producer has no skipper for {field}"));
            Ok(Some(DocValuesSkipper::from_impl(Box::new(
                AssertingDocValuesSkipper::new(skipper),
            ))))
        } else {
            assert!(
                skipper.is_none(),
                "a skipper for {field}, whose field info has no skip index"
            );
            Ok(None)
        }
    }

    fn sorted_numeric_is_singleton(&self, field: &str) -> bool {
        self.in_.sorted_numeric_is_singleton(field)
    }

    fn check_integrity(&self) -> Result<()> {
        self.in_.check_integrity()
    }
}

/// `AssertingLeafReader.AssertingPointValues` over the reader's point
/// values.
pub struct AssertingCodecPointValues<'a> {
    in_: Box<dyn PointValues + 'a>,
}

impl<'a> AssertingCodecPointValues<'a> {
    /// `assertStats(maxDoc)`.
    pub fn new(in_: Box<dyn PointValues + 'a>, max_doc: i32) -> Self {
        let (size, doc_count) = (in_.size(), in_.doc_count());
        assert!(size > 0, "point values of size {size}");
        assert!(doc_count > 0, "point values in {doc_count} documents");
        assert!(
            i64::from(doc_count) <= size,
            "{doc_count} documents hold {size} points"
        );
        assert!(
            doc_count <= max_doc,
            "points in {doc_count} documents of {max_doc}"
        );
        Self { in_ }
    }

    fn visitor<'v>(&self, in_: &'v mut dyn IntersectVisitor) -> AssertingIntersectVisitor<'v> {
        AssertingIntersectVisitor::new(
            self.in_.num_dimensions(),
            self.in_.num_index_dimensions(),
            self.in_.bytes_per_dimension(),
            in_,
        )
    }
}

impl PointValues for AssertingCodecPointValues<'_> {
    fn intersect(&self, visitor: &mut dyn IntersectVisitor) -> Result<()> {
        self.in_.intersect(&mut self.visitor(visitor))
    }
    fn estimate_point_count(&self, visitor: &mut dyn IntersectVisitor) -> Result<i64> {
        self.in_.estimate_point_count(&mut self.visitor(visitor))
    }
    /// `Objects.requireNonNull(in.getMinPackedValue())`.
    fn min_packed_value(&self) -> &[u8] {
        self.in_.min_packed_value()
    }
    fn max_packed_value(&self) -> &[u8] {
        self.in_.max_packed_value()
    }
    fn num_dimensions(&self) -> i32 {
        self.in_.num_dimensions()
    }
    fn num_index_dimensions(&self) -> i32 {
        self.in_.num_index_dimensions()
    }
    fn bytes_per_dimension(&self) -> i32 {
        self.in_.bytes_per_dimension()
    }
    fn size(&self) -> i64 {
        self.in_.size()
    }
    fn doc_count(&self) -> i32 {
        self.in_.doc_count()
    }
}

/// `AssertingLeafReader.AssertingIntersectVisitor`.
pub struct AssertingIntersectVisitor<'a> {
    in_: &'a mut dyn IntersectVisitor,
    num_data_dims: usize,
    num_index_dims: usize,
    bytes_per_dim: usize,
    /// `lastDocValue`, kept for one-dimensional points only.
    last_doc_value: Option<Vec<u8>>,
    last_min_packed_value: Vec<u8>,
    last_max_packed_value: Vec<u8>,
    last_compare_result: Option<Relation>,
    last_doc_id: i32,
    doc_budget: i64,
}

impl<'a> AssertingIntersectVisitor<'a> {
    /// `new AssertingIntersectVisitor(numDataDims, numIndexDims, bytesPerDim,
    /// in)`.
    pub fn new(
        num_data_dims: i32,
        num_index_dims: i32,
        bytes_per_dim: i32,
        in_: &'a mut dyn IntersectVisitor,
    ) -> Self {
        let dims = |n: i32| usize::try_from(n).expect("a dimension count");
        let (num_data_dims, num_index_dims, bytes_per_dim) = (
            dims(num_data_dims),
            dims(num_index_dims),
            dims(bytes_per_dim),
        );
        let packed = num_data_dims * bytes_per_dim;
        Self {
            in_,
            num_data_dims,
            num_index_dims,
            bytes_per_dim,
            last_doc_value: (num_data_dims == 1).then(|| vec![0; bytes_per_dim]),
            last_min_packed_value: vec![0; packed],
            last_max_packed_value: vec![0; packed],
            last_compare_result: None,
            last_doc_id: -1,
            doc_budget: 0,
        }
    }

    fn spend(&mut self, docs: usize) {
        self.doc_budget -= i64::try_from(docs).unwrap_or(i64::MAX);
        assert!(
            self.doc_budget >= 0,
            "called add() more times than the last call to grow() reserved"
        );
    }

    fn dim(&self, dim: usize) -> std::ops::Range<usize> {
        dim * self.bytes_per_dim..(dim + 1) * self.bytes_per_dim
    }
}

impl IntersectVisitor for AssertingIntersectVisitor<'_> {
    fn compare(&mut self, min_packed: &[u8], max_packed: &[u8]) -> Relation {
        for dim in 0..self.num_index_dims {
            let r = self.dim(dim);
            assert!(
                min_packed[r.clone()] <= max_packed[r],
                "cell min above max in dim {dim}"
            );
        }
        let n = self.num_index_dims * self.bytes_per_dim;
        self.last_max_packed_value[..n].copy_from_slice(&max_packed[..n]);
        self.last_min_packed_value[..n].copy_from_slice(&min_packed[..n]);
        let relation = self.in_.compare(min_packed, max_packed);
        self.last_compare_result = Some(relation);
        relation
    }

    fn visit(&mut self, doc_id: i32) {
        self.spend(1);
        assert!(
            matches!(
                self.last_compare_result,
                None | Some(Relation::CellInsideQuery)
            ),
            "visit(doc) after {:?}",
            self.last_compare_result
        );
        self.in_.visit(doc_id);
    }

    fn visit_many(&mut self, doc_ids: &[i32]) {
        self.spend(doc_ids.len());
        assert!(
            matches!(
                self.last_compare_result,
                None | Some(Relation::CellInsideQuery)
            ),
            "visit(docs) after {:?}",
            self.last_compare_result
        );
        self.in_.visit_many(doc_ids);
    }

    fn visit_with_value(&mut self, doc_id: i32, packed_value: &[u8]) {
        self.spend(1);
        assert!(
            matches!(
                self.last_compare_result,
                None | Some(Relation::CellCrossesQuery)
            ),
            "visit(doc, value) after {:?}",
            self.last_compare_result
        );
        if self.last_compare_result.is_some() {
            for dim in 0..self.num_index_dims {
                let r = self.dim(dim);
                assert!(
                    self.last_min_packed_value[r.clone()] <= packed_value[r.clone()]
                        && self.last_max_packed_value[r.clone()] >= packed_value[r],
                    "dim={dim} of {} value={packed_value:?}",
                    self.num_data_dims
                );
            }
            self.last_compare_result = None;
        }
        assert_eq!(
            packed_value.len(),
            self.num_data_dims * self.bytes_per_dim,
            "a packed value's length"
        );
        if let Some(last) = &mut self.last_doc_value {
            let value = &packed_value[..self.bytes_per_dim];
            match (*last.as_slice()).cmp(value) {
                std::cmp::Ordering::Less => {}
                std::cmp::Ordering::Equal => assert!(
                    self.last_doc_id <= doc_id,
                    "doc ids are out of order when point values are the same!"
                ),
                std::cmp::Ordering::Greater => panic!("point values are out of order"),
            }
            last.copy_from_slice(value);
            self.last_doc_id = doc_id;
        }
        self.in_.visit_with_value(doc_id, packed_value);
    }

    fn grow(&mut self, count: usize) {
        self.in_.grow(count);
        self.doc_budget = i64::try_from(count).unwrap_or(i64::MAX);
    }
}

/// An index whose `Asserting`-coded segments are read through
/// [`AssertingCodecLeafReader`]s.
pub struct AssertingCodecReader {
    in_: Arc<DirectoryReader>,
    leaves: Vec<Arc<dyn LeafReader>>,
    /// Its own cache key, as a `FilterDirectoryReader`'s.
    reader_cache: CacheHelper,
}

impl AssertingCodecReader {
    /// Opens the latest commit of `dir`.
    ///
    /// # Errors
    /// What [`DirectoryReader::open`] reports.
    pub fn open(dir: &dyn Directory) -> Result<Self> {
        Ok(Self::new(Arc::new(DirectoryReader::open(dir)?)))
    }

    /// `in_`'s segments, the `Asserting`-coded ones wrapped.
    pub fn new(in_: Arc<DirectoryReader>) -> Self {
        let asserting: HashSet<&str> = in_
            .segment_infos
            .segments
            .iter()
            .filter(|s| s.codec_name == NAME)
            .map(|s| s.segment_name.as_str())
            .collect();
        let leaves = in_
            .segment_readers()
            .iter()
            .map(|s| {
                let leaf: Arc<dyn LeafReader> = Arc::new(s.clone());
                if asserting.contains(s.segment_name.as_str()) {
                    Arc::new(AssertingCodecLeafReader::new(leaf)) as Arc<dyn LeafReader>
                } else {
                    leaf
                }
            })
            .collect();
        Self {
            in_,
            leaves,
            reader_cache: CacheHelper::new(),
        }
    }

    /// The reader underneath.
    pub fn delegate(&self) -> &Arc<DirectoryReader> {
        &self.in_
    }

    /// The leaves, in order.
    pub fn leaf_readers(&self) -> &[Arc<dyn LeafReader>] {
        &self.leaves
    }

    /// `DirectoryReader.openIfChanged`, read the same way.
    ///
    /// # Errors
    /// What reopening reports.
    pub fn open_if_changed(&self, dir: &dyn Directory) -> Result<Option<Self>> {
        Ok(self
            .in_
            .open_if_changed(dir)?
            .map(|r| Self::new(Arc::new(r))))
    }
}

impl IndexReader for AssertingCodecReader {
    fn max_doc(&self) -> i32 {
        self.in_.max_doc()
    }
    fn num_docs(&self) -> i32 {
        self.in_.num_docs()
    }
    fn leaves(&self) -> Vec<LeafReaderContext<'_>> {
        lucene_search::reader::composite_leaves(&self.sequential_sub_readers())
    }
    fn reader_cache_helper(&self) -> Option<&CacheHelper> {
        Some(&self.reader_cache)
    }
}

impl CompositeReader for AssertingCodecReader {
    fn sequential_sub_readers(&self) -> Vec<SubReader<'_>> {
        self.leaves
            .iter()
            .map(|l| SubReader::Leaf(l.as_ref()))
            .collect()
    }
    fn leaf_handles(&self) -> Vec<Arc<dyn LeafReader>> {
        self.leaves.clone()
    }
}
