//! The test framework's leaf and composite reader wrappers:
//! `AllDeletedFilterReader`, `FieldFilterLeafReader`,
//! `MismatchedLeafReader`/`MismatchedCodecReader`/`MismatchedDirectoryReader`
//! and `OwnCacheKeyMultiReader`.
//!
//! Each leaf wrapper is a struct holding its inner reader and implementing
//! [`LeafView`], the handful of `FilterLeafReader` methods these wrappers
//! override (field infos, the field filter, live docs, stored fields, term
//! vectors, cache helpers); `impl_view_leaf!` delegates the rest. Java's
//! filters change the `FieldInfos` a leaf reports, which
//! `lucene_search::reader::filter::FilterLeafReader`'s hooks cannot, hence a
//! view of their own.
//!
//! Differs:
//!
//! - Readers are read by field name, so `MismatchedCodecReader`'s doc-values
//!   and norms producers (which map a shuffled `FieldInfo` back to the
//!   original) have nothing to remap; its stored fields renumber as
//!   `MismatchedLeafReader`'s do, and term vectors, which carry field
//!   numbers here, are renumbered too.
//! - `FieldFilterLeafReader`'s term vectors keep the filtered fields, `None`
//!   when none is left, as Java's; `size()` returning -1 has no counterpart.

use std::collections::HashSet;
use std::sync::{Arc, Mutex, PoisonError};

use lucene_codecs::field_infos::FieldInfos;
use lucene_codecs::stored_fields::{StoredFieldVisitor, VisitStatus};
use lucene_index::segment_info::IndexSortField;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::reader::filter::{FilterDirectoryReader, SubReaderWrapper};
use lucene_search::reader::multi_reader::MultiReader;
use lucene_search::reader::{
    BinaryDocValues, ByteVectorValues, CacheHelper, CodecReader, CompositeReader,
    FloatVectorValues, IndexReader, LeafReader, LeafReaderContext, NumericDocValues, PointValues,
    ReaderHandle, SortedDocValues, SortedNumericDocValues, SortedSetDocValues, SubReader,
    TermVectorsDocument, Terms,
};
use lucene_search::Result;
use lucene_util::fixed_bit_set::FixedBitSet;
use lucene_util::java_random::JavaRandom;

use super::merge_policies::shuffle;

/// The `FilterLeafReader` methods the test wrappers override, each
/// defaulting to the inner reader's answer.
pub trait LeafView: Send + Sync {
    /// The inner reader.
    fn inner(&self) -> &dyn LeafReader;
    /// `getFieldInfos()`.
    fn view_field_infos(&self) -> &FieldInfos {
        self.inner().field_infos()
    }
    /// Whether `terms`, the binary, sorted, sorted-numeric and sorted-set
    /// doc values and the norms of `field` are visible (Java's
    /// `FieldFilterLeafReader` gates exactly those).
    fn has_field(&self, _field: &str) -> bool {
        true
    }
    /// `getLiveDocs()`.
    fn view_live_docs(&self) -> Option<&FixedBitSet> {
        self.inner().live_docs()
    }
    /// `storedFields().document(doc, visitor)`.
    fn view_document(&self, doc: i32, visitor: &mut dyn StoredFieldVisitor) -> Result<()> {
        self.inner().document(doc, visitor)
    }
    /// `termVectors().get(doc)`.
    fn view_term_vectors(&self, doc: i32) -> Result<Option<TermVectorsDocument>> {
        self.inner().term_vectors(doc)
    }
    /// `getCoreCacheHelper()`.
    fn view_core_cache_helper(&self) -> Option<&CacheHelper> {
        self.inner().core_cache_helper()
    }
    /// `getReaderCacheHelper()`.
    fn view_reader_cache_helper(&self) -> Option<&CacheHelper> {
        self.inner().reader_cache_helper()
    }
}

macro_rules! gated {
    ($self:ident, $field:ident, $call:expr) => {
        if $self.has_field($field) {
            $call
        } else {
            Ok(None)
        }
    };
}

/// `IndexReader` and `LeafReader` for a [`LeafView`], delegating what the
/// view does not override.
macro_rules! impl_view_leaf {
    ($ty:ty) => {
        impl IndexReader for $ty {
            fn max_doc(&self) -> i32 {
                self.inner().max_doc()
            }
            fn num_docs(&self) -> i32 {
                lucene_search::reader::num_docs_of(self.max_doc(), self.view_live_docs())
            }
            fn leaves(&self) -> Vec<LeafReaderContext<'_>> {
                vec![LeafReaderContext {
                    reader: self,
                    ord: 0,
                    doc_base: 0,
                }]
            }
            fn reader_cache_helper(&self) -> Option<&CacheHelper> {
                self.view_reader_cache_helper()
            }
        }

        impl LeafReader for $ty {
            fn field_infos(&self) -> &FieldInfos {
                self.view_field_infos()
            }
            fn live_docs(&self) -> Option<&FixedBitSet> {
                self.view_live_docs()
            }
            fn terms(&self, field: &str) -> Result<Option<Box<dyn Terms + '_>>> {
                gated!(self, field, self.inner().terms(field))
            }
            fn numeric_doc_values(
                &self,
                field: &str,
            ) -> Result<Option<Box<dyn NumericDocValues + '_>>> {
                self.inner().numeric_doc_values(field)
            }
            fn binary_doc_values(
                &self,
                field: &str,
            ) -> Result<Option<Box<dyn BinaryDocValues + '_>>> {
                gated!(self, field, self.inner().binary_doc_values(field))
            }
            fn sorted_doc_values(
                &self,
                field: &str,
            ) -> Result<Option<Box<dyn SortedDocValues + '_>>> {
                gated!(self, field, self.inner().sorted_doc_values(field))
            }
            fn sorted_numeric_doc_values(
                &self,
                field: &str,
            ) -> Result<Option<Box<dyn SortedNumericDocValues + '_>>> {
                gated!(self, field, self.inner().sorted_numeric_doc_values(field))
            }
            fn sorted_set_doc_values(
                &self,
                field: &str,
            ) -> Result<Option<Box<dyn SortedSetDocValues + '_>>> {
                gated!(self, field, self.inner().sorted_set_doc_values(field))
            }
            fn norm_values(&self, field: &str) -> Result<Option<Box<dyn NumericDocValues + '_>>> {
                gated!(self, field, self.inner().norm_values(field))
            }
            fn point_values(&self, field: &str) -> Result<Option<Box<dyn PointValues + '_>>> {
                self.inner().point_values(field)
            }
            fn float_vector_values(
                &self,
                field: &str,
            ) -> Result<Option<Box<dyn FloatVectorValues + '_>>> {
                self.inner().float_vector_values(field)
            }
            fn byte_vector_values(
                &self,
                field: &str,
            ) -> Result<Option<Box<dyn ByteVectorValues + '_>>> {
                self.inner().byte_vector_values(field)
            }
            fn document(&self, doc: i32, visitor: &mut dyn StoredFieldVisitor) -> Result<()> {
                self.view_document(doc, visitor)
            }
            fn term_vectors(&self, doc: i32) -> Result<Option<TermVectorsDocument>> {
                self.view_term_vectors(doc)
            }
            fn index_sort(&self) -> Option<&[IndexSortField]> {
                self.inner().index_sort()
            }
            fn core_cache_helper(&self) -> Option<&CacheHelper> {
                self.view_core_cache_helper()
            }
            fn check_integrity(&self) -> Result<()> {
                self.inner().check_integrity()
            }
        }
    };
}

// ---------------------------------------------------------------------------
// AllDeletedFilterReader
// ---------------------------------------------------------------------------

/// `AllDeletedFilterReader`: every document deleted.
pub struct AllDeletedFilterReader {
    in_: Arc<dyn LeafReader>,
    live_docs: FixedBitSet,
}

impl AllDeletedFilterReader {
    /// `new AllDeletedFilterReader(in)`.
    pub fn new(in_: Arc<dyn LeafReader>) -> Self {
        let live_docs = FixedBitSet::new(in_.max_doc().max(0) as usize);
        Self { in_, live_docs }
    }
}

impl LeafView for AllDeletedFilterReader {
    fn inner(&self) -> &dyn LeafReader {
        self.in_.as_ref()
    }
    fn view_live_docs(&self) -> Option<&FixedBitSet> {
        Some(&self.live_docs)
    }
    fn view_reader_cache_helper(&self) -> Option<&CacheHelper> {
        None
    }
}

impl_view_leaf!(AllDeletedFilterReader);

// ---------------------------------------------------------------------------
// FieldFilterLeafReader
// ---------------------------------------------------------------------------

/// `FieldFilterLeafReader`: only `fields` (or, with `negate`, every field
/// but them) visible.
pub struct FieldFilterLeafReader {
    in_: Arc<dyn LeafReader>,
    fields: HashSet<String>,
    negate: bool,
    field_infos: FieldInfos,
}

impl FieldFilterLeafReader {
    /// `new FieldFilterLeafReader(in, fields, negate)`.
    pub fn new(in_: Arc<dyn LeafReader>, fields: HashSet<String>, negate: bool) -> Self {
        let kept = in_
            .field_infos()
            .fields
            .iter()
            .filter(|fi| negate ^ fields.contains(&fi.name))
            .cloned()
            .collect();
        Self {
            field_infos: FieldInfos::new(kept).expect("a subset of consistent field infos"),
            in_,
            fields,
            negate,
        }
    }

    fn keeps(&self, field: &str) -> bool {
        self.negate ^ self.fields.contains(field)
    }
}

impl std::fmt::Display for FieldFilterLeafReader {
    /// Java's `toString`, with the fields sorted.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut fields: Vec<&str> = self.fields.iter().map(String::as_str).collect();
        fields.sort_unstable();
        let not = if self.negate { "!" } else { "" };
        write!(
            f,
            "FieldFilterLeafReader(fields={not}[{}])",
            fields.join(", ")
        )
    }
}

/// The stored-fields visitor `FieldFilterLeafReader` reads through: a field
/// filtered out is never asked for.
struct FilteringVisitor<'a> {
    reader: &'a FieldFilterLeafReader,
    in_: &'a mut dyn StoredFieldVisitor,
}

impl StoredFieldVisitor for FilteringVisitor<'_> {
    fn needs_field(
        &mut self,
        field_number: i32,
    ) -> lucene_codecs::stored_fields::Result<VisitStatus> {
        let keep = self
            .reader
            .in_
            .field_infos()
            .field_by_number(field_number)
            .is_some_and(|fi| self.reader.keeps(&fi.name));
        if keep {
            self.in_.needs_field(field_number)
        } else {
            Ok(VisitStatus::No)
        }
    }
    fn string_field(&mut self, n: i32, v: &str) -> lucene_codecs::stored_fields::Result<()> {
        self.in_.string_field(n, v)
    }
    fn binary_field(&mut self, n: i32, v: &[u8]) -> lucene_codecs::stored_fields::Result<()> {
        self.in_.binary_field(n, v)
    }
    fn int_field(&mut self, n: i32, v: i32) -> lucene_codecs::stored_fields::Result<()> {
        self.in_.int_field(n, v)
    }
    fn long_field(&mut self, n: i32, v: i64) -> lucene_codecs::stored_fields::Result<()> {
        self.in_.long_field(n, v)
    }
    fn float_field(&mut self, n: i32, v: f32) -> lucene_codecs::stored_fields::Result<()> {
        self.in_.float_field(n, v)
    }
    fn double_field(&mut self, n: i32, v: f64) -> lucene_codecs::stored_fields::Result<()> {
        self.in_.double_field(n, v)
    }
}

impl LeafView for FieldFilterLeafReader {
    fn inner(&self) -> &dyn LeafReader {
        self.in_.as_ref()
    }
    fn view_field_infos(&self) -> &FieldInfos {
        &self.field_infos
    }
    fn has_field(&self, field: &str) -> bool {
        self.keeps(field)
    }
    fn view_document(&self, doc: i32, visitor: &mut dyn StoredFieldVisitor) -> Result<()> {
        self.in_.document(
            doc,
            &mut FilteringVisitor {
                reader: self,
                in_: visitor,
            },
        )
    }
    fn view_term_vectors(&self, doc: i32) -> Result<Option<TermVectorsDocument>> {
        let Some(mut tv) = self.in_.term_vectors(doc)? else {
            return Ok(None);
        };
        let infos = self.in_.field_infos();
        tv.fields.retain(|f| {
            infos
                .field_by_number(f.field_number)
                .is_some_and(|fi| self.keeps(&fi.name))
        });
        // we need to check for emptiness, so we can return null
        Ok((!tv.fields.is_empty()).then_some(tv))
    }
    fn view_core_cache_helper(&self) -> Option<&CacheHelper> {
        None
    }
    fn view_reader_cache_helper(&self) -> Option<&CacheHelper> {
        None
    }
}

impl_view_leaf!(FieldFilterLeafReader);

// ---------------------------------------------------------------------------
// Mismatched readers
// ---------------------------------------------------------------------------

/// `MismatchedLeafReader.shuffleInfos`: the fields in `Collections.shuffle`
/// order, renumbered from 0.
pub fn shuffle_infos(infos: &FieldInfos, random: &mut JavaRandom) -> FieldInfos {
    let mut shuffled = infos.fields.clone();
    shuffle(random, &mut shuffled);
    for (i, fi) in shuffled.iter_mut().enumerate() {
        fi.number = i as i32;
    }
    FieldInfos::new(shuffled).expect("renumbered consistent field infos")
}

/// `MismatchedLeafReader.MismatchedVisitor`: the inner reader's field
/// numbers turned into the shuffled ones.
struct MismatchedVisitor<'a> {
    orig: &'a FieldInfos,
    shuffled: &'a FieldInfos,
    in_: &'a mut dyn StoredFieldVisitor,
}

impl MismatchedVisitor<'_> {
    fn renumber(&self, original: i32) -> i32 {
        self.orig
            .field_by_number(original)
            .and_then(|fi| self.shuffled.field_by_name(&fi.name))
            .map(|fi| fi.number)
            .expect("stored fields sending bogus infos!")
    }
}

impl StoredFieldVisitor for MismatchedVisitor<'_> {
    fn needs_field(&mut self, n: i32) -> lucene_codecs::stored_fields::Result<VisitStatus> {
        let n = self.renumber(n);
        self.in_.needs_field(n)
    }
    fn string_field(&mut self, n: i32, v: &str) -> lucene_codecs::stored_fields::Result<()> {
        let n = self.renumber(n);
        self.in_.string_field(n, v)
    }
    fn binary_field(&mut self, n: i32, v: &[u8]) -> lucene_codecs::stored_fields::Result<()> {
        let n = self.renumber(n);
        self.in_.binary_field(n, v)
    }
    fn int_field(&mut self, n: i32, v: i32) -> lucene_codecs::stored_fields::Result<()> {
        let n = self.renumber(n);
        self.in_.int_field(n, v)
    }
    fn long_field(&mut self, n: i32, v: i64) -> lucene_codecs::stored_fields::Result<()> {
        let n = self.renumber(n);
        self.in_.long_field(n, v)
    }
    fn float_field(&mut self, n: i32, v: f32) -> lucene_codecs::stored_fields::Result<()> {
        let n = self.renumber(n);
        self.in_.float_field(n, v)
    }
    fn double_field(&mut self, n: i32, v: f64) -> lucene_codecs::stored_fields::Result<()> {
        let n = self.renumber(n);
        self.in_.double_field(n, v)
    }
}

fn mismatched_document(
    inner: &dyn LeafReader,
    shuffled: &FieldInfos,
    doc: i32,
    visitor: &mut dyn StoredFieldVisitor,
) -> Result<()> {
    inner.document(
        doc,
        &mut MismatchedVisitor {
            orig: inner.field_infos(),
            shuffled,
            in_: visitor,
        },
    )
}

fn mismatched_term_vectors(
    inner: &dyn LeafReader,
    shuffled: &FieldInfos,
    doc: i32,
) -> Result<Option<TermVectorsDocument>> {
    let Some(mut tv) = inner.term_vectors(doc)? else {
        return Ok(None);
    };
    let orig = inner.field_infos();
    for f in &mut tv.fields {
        f.field_number = orig
            .field_by_number(f.field_number)
            .and_then(|fi| shuffled.field_by_name(&fi.name))
            .map(|fi| fi.number)
            .expect("term vectors sending bogus infos!");
    }
    Ok(Some(tv))
}

/// `MismatchedLeafReader`: the inner reader with its fields shuffled and
/// renumbered.
pub struct MismatchedLeafReader {
    in_: Arc<dyn LeafReader>,
    shuffled: FieldInfos,
}

impl MismatchedLeafReader {
    /// `new MismatchedLeafReader(in, random)`.
    pub fn new(in_: Arc<dyn LeafReader>, random: &mut JavaRandom) -> Self {
        let shuffled = shuffle_infos(in_.field_infos(), random);
        Self { in_, shuffled }
    }
}

impl LeafView for MismatchedLeafReader {
    fn inner(&self) -> &dyn LeafReader {
        self.in_.as_ref()
    }
    fn view_field_infos(&self) -> &FieldInfos {
        &self.shuffled
    }
    fn view_document(&self, doc: i32, visitor: &mut dyn StoredFieldVisitor) -> Result<()> {
        mismatched_document(self.in_.as_ref(), &self.shuffled, doc, visitor)
    }
    fn view_term_vectors(&self, doc: i32) -> Result<Option<TermVectorsDocument>> {
        mismatched_term_vectors(self.in_.as_ref(), &self.shuffled, doc)
    }
}

impl_view_leaf!(MismatchedLeafReader);

/// `MismatchedCodecReader`: [`MismatchedLeafReader`] over a codec reader,
/// itself one (what `MockRandomMergePolicy` hands a merge).
pub struct MismatchedCodecReader {
    in_: Arc<dyn CodecReader>,
    shuffled: FieldInfos,
}

impl MismatchedCodecReader {
    /// `new MismatchedCodecReader(in, random)`.
    pub fn new(in_: Arc<dyn CodecReader>, random: &mut JavaRandom) -> Self {
        let shuffled = shuffle_infos(in_.field_infos(), random);
        Self { in_, shuffled }
    }
}

impl LeafView for MismatchedCodecReader {
    fn inner(&self) -> &dyn LeafReader {
        self.in_.as_ref()
    }
    fn view_field_infos(&self) -> &FieldInfos {
        &self.shuffled
    }
    fn view_document(&self, doc: i32, visitor: &mut dyn StoredFieldVisitor) -> Result<()> {
        mismatched_document(self.in_.as_ref(), &self.shuffled, doc, visitor)
    }
    fn view_term_vectors(&self, doc: i32) -> Result<Option<TermVectorsDocument>> {
        mismatched_term_vectors(self.in_.as_ref(), &self.shuffled, doc)
    }
}

impl_view_leaf!(MismatchedCodecReader);

impl CodecReader for MismatchedCodecReader {}

/// `MismatchedDirectoryReader.MismatchedSubReaderWrapper`: every leaf
/// shuffled from one shared stream.
struct MismatchedSubReaderWrapper {
    random: Mutex<JavaRandom>,
}

impl SubReaderWrapper for MismatchedSubReaderWrapper {
    fn wrap(&self, reader: Arc<dyn LeafReader>) -> Result<Arc<dyn LeafReader>> {
        let mut random = self.random.lock().unwrap_or_else(PoisonError::into_inner);
        Ok(Arc::new(MismatchedLeafReader::new(reader, &mut random)))
    }
}

/// `new MismatchedDirectoryReader(in, random)`: every leaf of `in_` a
/// [`MismatchedLeafReader`]. (Java's `doWrapDirectoryReader` wraps a
/// reopened reader in an `AssertingDirectoryReader`;
/// [`FilterDirectoryReader::open_if_changed`] wraps it with the same
/// wrapper.)
///
/// # Errors
/// None in practice: the wrapper does not fail.
pub fn mismatched_directory_reader(
    in_: Arc<DirectoryReader>,
    random: JavaRandom,
) -> Result<FilterDirectoryReader> {
    FilterDirectoryReader::new(
        in_,
        Arc::new(MismatchedSubReaderWrapper {
            random: Mutex::new(random),
        }),
    )
}

// ---------------------------------------------------------------------------
// OwnCacheKeyMultiReader
// ---------------------------------------------------------------------------

/// `OwnCacheKeyMultiReader`: a [`MultiReader`] with a cache key of its own
/// (whose closed listeners run when the last handle to it goes away).
pub struct OwnCacheKeyMultiReader {
    in_: MultiReader,
    cache_helper: CacheHelper,
}

impl OwnCacheKeyMultiReader {
    /// `new OwnCacheKeyMultiReader(subReaders...)`.
    ///
    /// # Errors
    /// As [`MultiReader::new`].
    pub fn new(subs: Vec<ReaderHandle>) -> Result<Self> {
        Ok(Self {
            in_: MultiReader::new(subs)?,
            cache_helper: CacheHelper::new(),
        })
    }

    /// The multi reader underneath.
    pub fn multi_reader(&self) -> &MultiReader {
        &self.in_
    }
}

impl IndexReader for OwnCacheKeyMultiReader {
    fn max_doc(&self) -> i32 {
        self.in_.max_doc()
    }
    fn num_docs(&self) -> i32 {
        self.in_.num_docs()
    }
    fn leaves(&self) -> Vec<LeafReaderContext<'_>> {
        self.in_.leaves()
    }
    fn reader_cache_helper(&self) -> Option<&CacheHelper> {
        Some(&self.cache_helper)
    }
}

impl CompositeReader for OwnCacheKeyMultiReader {
    fn sequential_sub_readers(&self) -> Vec<SubReader<'_>> {
        self.in_.sequential_sub_readers()
    }
    fn leaf_handles(&self) -> Vec<Arc<dyn LeafReader>> {
        self.in_.leaf_handles()
    }
}
