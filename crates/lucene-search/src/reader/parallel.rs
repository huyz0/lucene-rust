//! Ports of `org.apache.lucene.index.ParallelLeafReader` and
//! `ParallelCompositeReader`: readers over indexes that hold the **same
//! documents** (same `maxDoc`, same doc ids) with different fields, joined
//! field-wise. Each field is read from the first reader that has it; stored
//! fields come from the stored-fields readers, in order; live docs are the
//! first reader's.
//!
//! The joined `FieldInfos` numbers fields as `FieldInfos.Builder` does (a
//! field keeps its number unless an earlier field took it), and stored-field
//! visitors and term vectors see those numbers -- Java's visitors see the
//! sub-reader's `FieldInfo`, whose *name* is what they key on; renumbering
//! through names is the same thing in this port's number-keyed visitor.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use lucene_codecs::field_infos::{FieldInfos, IndexOptions};
use lucene_index::segment_info::IndexSortField;
use lucene_util::fixed_bit_set::FixedBitSet;

use super::{
    composite_leaves, remap_term_vectors, BinaryDocValues, ByteVectorValues, CacheHelper,
    CompositeReader, FieldInfosBuilder, FloatVectorValues, IndexReader, LeafReader,
    LeafReaderContext, NumericDocValues, PointValues, RemappingVisitor, SortedDocValues,
    SortedNumericDocValues, SortedSetDocValues, StoredFieldVisitor, SubReader, TermVectorsDocument,
    Terms,
};
use crate::{Error, Result};

/// `ParallelLeafReader`.
pub struct ParallelLeafReader {
    field_infos: FieldInfos,
    parallel_readers: Vec<Arc<dyn LeafReader>>,
    stored_fields_readers: Vec<Arc<dyn LeafReader>>,
    max_doc: i32,
    num_docs: i32,
    has_deletions: bool,
    index_sort: Option<Vec<IndexSortField>>,
    /// `fieldToReader`: each field's first reader.
    field_to_reader: HashMap<String, usize>,
    /// `termsFieldToReader`: the indexed fields' readers.
    terms_field_to_reader: HashMap<String, usize>,
    /// `tvFieldToReader`: the term-vector fields' readers (each field's
    /// first reader only), in field-name order.
    tv_field_to_reader: BTreeMap<String, usize>,
}

impl ParallelLeafReader {
    /// `new ParallelLeafReader(readers)`: every reader also serves stored
    /// fields.
    ///
    /// # Errors
    /// As [`Self::with_stored_fields_readers`].
    pub fn new(readers: Vec<Arc<dyn LeafReader>>) -> Result<Self> {
        let stored = readers.clone();
        Self::with_stored_fields_readers(readers, stored)
    }

    /// `new ParallelLeafReader(closeSubReaders, readers, storedFieldsReaders)`.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] for stored-fields readers without a main
    /// reader, readers whose `maxDoc`s differ, or readers with different
    /// index sorts.
    pub fn with_stored_fields_readers(
        readers: Vec<Arc<dyn LeafReader>>,
        stored_fields_readers: Vec<Arc<dyn LeafReader>>,
    ) -> Result<Self> {
        if readers.is_empty() && !stored_fields_readers.is_empty() {
            return Err(Error::IllegalArgument(
                "There must be at least one main reader if storedFieldsReaders are used.".into(),
            ));
        }
        let (max_doc, num_docs, has_deletions) = match readers.first() {
            Some(first) => (first.max_doc(), first.num_docs(), first.has_deletions()),
            None => (0, 0, false),
        };
        for r in readers.iter().chain(&stored_fields_readers) {
            if r.max_doc() != max_doc {
                return Err(Error::IllegalArgument(format!(
                    "All readers must have same maxDoc: {max_doc}!={}",
                    r.max_doc()
                )));
            }
        }
        let mut builder = FieldInfosBuilder::default();
        let mut index_sort: Option<Vec<IndexSortField>> = None;
        let mut field_to_reader = HashMap::new();
        let mut terms_field_to_reader = HashMap::new();
        let mut tv_field_to_reader = BTreeMap::new();
        for (i, reader) in readers.iter().enumerate() {
            if let Some(sort) = reader.index_sort() {
                match &index_sort {
                    None => index_sort = Some(sort.to_vec()),
                    Some(s) if s.as_slice() != sort => {
                        return Err(Error::IllegalArgument(format!(
                            "cannot combine LeafReaders that have different index sorts: saw \
                             both sort={s:?} and {sort:?}"
                        )))
                    }
                    Some(_) => {}
                }
            }
            for fi in &reader.field_infos().fields {
                if field_to_reader.contains_key(&fi.name) {
                    continue;
                }
                builder.add(fi);
                field_to_reader.insert(fi.name.clone(), i);
                if fi.index_options != IndexOptions::None {
                    terms_field_to_reader.insert(fi.name.clone(), i);
                }
                if fi.store_term_vectors {
                    tv_field_to_reader.insert(fi.name.clone(), i);
                }
            }
        }
        Ok(Self {
            field_infos: builder.finish(),
            parallel_readers: readers,
            stored_fields_readers,
            max_doc,
            num_docs,
            has_deletions,
            index_sort,
            field_to_reader,
            terms_field_to_reader,
            tv_field_to_reader,
        })
    }

    /// `getParallelReaders()`.
    pub fn parallel_readers(&self) -> &[Arc<dyn LeafReader>] {
        &self.parallel_readers
    }

    /// `getStoredFieldsReaders()`.
    pub fn stored_fields_readers(&self) -> &[Arc<dyn LeafReader>] {
        &self.stored_fields_readers
    }

    fn reader_for(&self, field: &str) -> Option<&dyn LeafReader> {
        self.field_to_reader
            .get(field)
            .map(|&i| self.parallel_readers[i].as_ref())
    }

    /// Whether this reader is exactly one reader serving everything, the
    /// only case whose cache helpers are its reader's.
    fn single(&self) -> Option<&dyn LeafReader> {
        match (
            self.parallel_readers.as_slice(),
            self.stored_fields_readers.as_slice(),
        ) {
            ([a], [b]) if Arc::ptr_eq(a, b) => Some(a.as_ref()),
            _ => None,
        }
    }
}

impl IndexReader for ParallelLeafReader {
    fn max_doc(&self) -> i32 {
        self.max_doc
    }
    fn num_docs(&self) -> i32 {
        self.num_docs
    }
    fn leaves(&self) -> Vec<LeafReaderContext<'_>> {
        vec![LeafReaderContext {
            reader: self,
            ord: 0,
            doc_base: 0,
        }]
    }
    fn reader_cache_helper(&self) -> Option<&CacheHelper> {
        self.single().and_then(|r| r.reader_cache_helper())
    }
}

impl LeafReader for ParallelLeafReader {
    fn field_infos(&self) -> &FieldInfos {
        &self.field_infos
    }
    fn live_docs(&self) -> Option<&FixedBitSet> {
        if self.has_deletions {
            self.parallel_readers[0].live_docs()
        } else {
            None
        }
    }
    fn terms(&self, field: &str) -> Result<Option<Box<dyn Terms + '_>>> {
        match self.terms_field_to_reader.get(field) {
            None => Ok(None),
            Some(&i) => self.parallel_readers[i].terms(field),
        }
    }
    fn numeric_doc_values(&self, field: &str) -> Result<Option<Box<dyn NumericDocValues + '_>>> {
        self.reader_for(field)
            .map_or(Ok(None), |r| r.numeric_doc_values(field))
    }
    fn binary_doc_values(&self, field: &str) -> Result<Option<Box<dyn BinaryDocValues + '_>>> {
        self.reader_for(field)
            .map_or(Ok(None), |r| r.binary_doc_values(field))
    }
    fn sorted_doc_values(&self, field: &str) -> Result<Option<Box<dyn SortedDocValues + '_>>> {
        self.reader_for(field)
            .map_or(Ok(None), |r| r.sorted_doc_values(field))
    }
    fn sorted_numeric_doc_values(
        &self,
        field: &str,
    ) -> Result<Option<Box<dyn SortedNumericDocValues + '_>>> {
        self.reader_for(field)
            .map_or(Ok(None), |r| r.sorted_numeric_doc_values(field))
    }
    fn sorted_set_doc_values(
        &self,
        field: &str,
    ) -> Result<Option<Box<dyn SortedSetDocValues + '_>>> {
        self.reader_for(field)
            .map_or(Ok(None), |r| r.sorted_set_doc_values(field))
    }
    fn norm_values(&self, field: &str) -> Result<Option<Box<dyn NumericDocValues + '_>>> {
        self.reader_for(field)
            .map_or(Ok(None), |r| r.norm_values(field))
    }
    fn point_values(&self, field: &str) -> Result<Option<Box<dyn PointValues + '_>>> {
        self.reader_for(field)
            .map_or(Ok(None), |r| r.point_values(field))
    }
    fn float_vector_values(&self, field: &str) -> Result<Option<Box<dyn FloatVectorValues + '_>>> {
        self.reader_for(field)
            .map_or(Ok(None), |r| r.float_vector_values(field))
    }
    fn byte_vector_values(&self, field: &str) -> Result<Option<Box<dyn ByteVectorValues + '_>>> {
        self.reader_for(field)
            .map_or(Ok(None), |r| r.byte_vector_values(field))
    }
    /// Every stored-fields reader, in order, into the one visitor.
    fn document(&self, doc: i32, visitor: &mut dyn StoredFieldVisitor) -> Result<()> {
        for r in &self.stored_fields_readers {
            let mut remap = RemappingVisitor {
                inner: &mut *visitor,
                from: r.field_infos(),
                to: &self.field_infos,
            };
            r.document(doc, &mut remap)?;
        }
        Ok(())
    }
    /// The term vectors of each term-vector field, read from the field's
    /// own reader (`tvFieldToReader`), in field-name order; `None` when no
    /// field has a vector for `doc`.
    fn term_vectors(&self, doc: i32) -> Result<Option<TermVectorsDocument>> {
        let mut per_reader: BTreeMap<usize, Option<TermVectorsDocument>> = BTreeMap::new();
        for &i in self.tv_field_to_reader.values() {
            if let std::collections::btree_map::Entry::Vacant(e) = per_reader.entry(i) {
                let r = &self.parallel_readers[i];
                e.insert(
                    r.term_vectors(doc)?
                        .map(|d| remap_term_vectors(d, r.field_infos(), &self.field_infos)),
                );
            }
        }
        let mut fields = BTreeMap::new();
        for (i, d) in per_reader {
            let Some(d) = d else { continue };
            for f in d.fields {
                let Some(fi) = self.field_infos.field_by_number(f.field_number) else {
                    continue;
                };
                if self.tv_field_to_reader.get(&fi.name) == Some(&i) {
                    fields.insert(fi.name.clone(), f);
                }
            }
        }
        Ok((!fields.is_empty()).then(|| TermVectorsDocument {
            fields: fields.into_values().collect(),
        }))
    }
    fn index_sort(&self) -> Option<&[IndexSortField]> {
        self.index_sort.as_deref()
    }
    fn core_cache_helper(&self) -> Option<&CacheHelper> {
        self.single().and_then(|r| r.core_cache_helper())
    }
    fn doc_values_skipper(
        &self,
        field: &str,
    ) -> Result<Option<crate::reader::DocValuesSkipper<'_>>> {
        self.reader_for(field)
            .map_or(Ok(None), |r| r.doc_values_skipper(field))
    }
    fn sorted_numeric_is_singleton(&self, field: &str) -> bool {
        self.reader_for(field)
            .is_some_and(|r| r.sorted_numeric_is_singleton(field))
    }
    /// Every reader, main and stored-fields alike (`completeReaderSet`).
    fn check_integrity(&self) -> Result<()> {
        for r in self
            .parallel_readers
            .iter()
            .chain(&self.stored_fields_readers)
        {
            r.check_integrity()?;
        }
        Ok(())
    }
}

/// `ParallelCompositeReader`: composites with the same leaf structure,
/// joined leaf by leaf into [`ParallelLeafReader`]s.
pub struct ParallelCompositeReader {
    leaves: Vec<Arc<ParallelLeafReader>>,
    /// The readers it joins, kept for their lifetime (`completeReaderSet`).
    readers: Vec<Arc<dyn CompositeReader>>,
    stored_fields_readers: Vec<Arc<dyn CompositeReader>>,
}

impl ParallelCompositeReader {
    /// `new ParallelCompositeReader(readers)`.
    ///
    /// # Errors
    /// As [`Self::with_stored_fields_readers`].
    pub fn new(readers: Vec<Arc<dyn CompositeReader>>) -> Result<Self> {
        let stored = readers.clone();
        Self::with_stored_fields_readers(readers, stored)
    }

    /// `new ParallelCompositeReader(closeSubReaders, readers,
    /// storedFieldReaders)`.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] when the readers' `maxDoc`s, leaf counts
    /// or leaf `maxDoc`s differ, and what [`ParallelLeafReader`] refuses.
    pub fn with_stored_fields_readers(
        readers: Vec<Arc<dyn CompositeReader>>,
        stored_fields_readers: Vec<Arc<dyn CompositeReader>>,
    ) -> Result<Self> {
        let Some(first) = readers.first() else {
            if !stored_fields_readers.is_empty() {
                return Err(Error::IllegalArgument(
                    "There must be at least one main reader if storedFieldsReaders are used."
                        .into(),
                ));
            }
            return Ok(Self {
                leaves: Vec::new(),
                readers,
                stored_fields_readers,
            });
        };
        let max_doc = first.max_doc();
        let leaf_max_doc: Vec<i32> = first.leaves().iter().map(|l| l.reader.max_doc()).collect();
        for r in readers.iter().chain(&stored_fields_readers) {
            if r.max_doc() != max_doc {
                return Err(Error::IllegalArgument(format!(
                    "All readers must have same maxDoc: {max_doc}!={}",
                    r.max_doc()
                )));
            }
            let leaves = r.leaves();
            if leaves.len() != leaf_max_doc.len() {
                return Err(Error::IllegalArgument(
                    "All readers must have same number of leaf readers".into(),
                ));
            }
            if leaves
                .iter()
                .zip(&leaf_max_doc)
                .any(|(l, &m)| l.reader.max_doc() != m)
            {
                return Err(Error::IllegalArgument(
                    "All leaf readers must have same corresponding subReader maxDoc".into(),
                ));
            }
        }
        let handles: Vec<Vec<Arc<dyn LeafReader>>> =
            readers.iter().map(|r| r.leaf_handles()).collect();
        let stored_handles: Vec<Vec<Arc<dyn LeafReader>>> = stored_fields_readers
            .iter()
            .map(|r| r.leaf_handles())
            .collect();
        let leaves = (0..leaf_max_doc.len())
            .map(|i| {
                let subs = handles.iter().map(|h| Arc::clone(&h[i])).collect();
                let stored = stored_handles.iter().map(|h| Arc::clone(&h[i])).collect();
                ParallelLeafReader::with_stored_fields_readers(subs, stored).map(Arc::new)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            leaves,
            readers,
            stored_fields_readers,
        })
    }

    fn single(&self) -> Option<&dyn CompositeReader> {
        match (
            self.readers.as_slice(),
            self.stored_fields_readers.as_slice(),
        ) {
            ([a], [b]) if Arc::ptr_eq(a, b) => Some(a.as_ref()),
            _ => None,
        }
    }
}

impl IndexReader for ParallelCompositeReader {
    fn max_doc(&self) -> i32 {
        self.leaves
            .iter()
            .fold(0i32, |a, l| a.saturating_add(l.max_doc()))
    }
    fn num_docs(&self) -> i32 {
        self.leaves
            .iter()
            .fold(0i32, |a, l| a.saturating_add(l.num_docs()))
    }
    fn leaves(&self) -> Vec<LeafReaderContext<'_>> {
        composite_leaves(&self.sequential_sub_readers())
    }
    fn reader_cache_helper(&self) -> Option<&CacheHelper> {
        self.single().and_then(|r| r.reader_cache_helper())
    }
}

impl CompositeReader for ParallelCompositeReader {
    fn sequential_sub_readers(&self) -> Vec<SubReader<'_>> {
        self.leaves
            .iter()
            .map(|l| SubReader::Leaf(l.as_ref()))
            .collect()
    }
    fn leaf_handles(&self) -> Vec<Arc<dyn LeafReader>> {
        self.leaves
            .iter()
            .map(|l| Arc::clone(l) as Arc<dyn LeafReader>)
            .collect()
    }
}
