//! `IndexRearranger` and `BinaryDocValueSelector`: copy an index into another
//! with a chosen geometry -- one segment per document selector -- and then
//! mark the documents a deletes selector picks as deleted.
//!
//! Rearranging works in three steps, as in Java: every document of the input
//! is taken as live and each segment selector makes one segment of the
//! documents it selects (`addIndexes` over a reader whose live docs are the
//! selection); the output's segments are checked to be the selectors'; and
//! the deletes selector's documents are deleted from the rearranged index.
//! A segment all of whose documents are selected for deletion is discarded.

use std::collections::HashSet;

use lucene_store::directory::Directory;

use super::{index_error, open_writer};
use crate::directory_reader::DirectoryReader;
use crate::reader::{CodecReader, LeafReader};
use crate::{Error, Result};

/// `IndexRearranger.DocumentSelector`: which documents of a segment a
/// selector picks, one flag per document (Java's `BitSet`).
pub trait DocumentSelector {
    /// `getFilteredDocs(reader)`.
    fn get_filtered_docs(&self, reader: &dyn CodecReader) -> Result<Vec<bool>>;
}

/// `IndexRearranger`.
pub struct IndexRearranger<'d> {
    input: &'d dyn Directory,
    output: &'d dyn Directory,
    segment_selectors: Vec<Box<dyn DocumentSelector + 'd>>,
    deleted_docs_selector: Option<Box<dyn DocumentSelector + 'd>>,
}

impl<'d> IndexRearranger<'d> {
    /// `IndexRearranger(input, output, config, segmentSelectors,
    /// deletedDocsSelector)`: each segment selector makes one segment, in
    /// order; `deleted_docs_selector` picks the documents to delete after.
    pub fn new(
        input: &'d dyn Directory,
        output: &'d dyn Directory,
        segment_selectors: Vec<Box<dyn DocumentSelector + 'd>>,
        deleted_docs_selector: Option<Box<dyn DocumentSelector + 'd>>,
    ) -> Self {
        Self {
            input,
            output,
            segment_selectors,
            deleted_docs_selector,
        }
    }

    /// `execute()`.
    pub fn execute(&self) -> Result<()> {
        self.create_rearranged_index()?;
        self.finalize_rearrange()?;
        self.apply_deletes()
    }

    /// `createRearrangedIndex`: one segment per selector, assuming every
    /// document of the input is live. The output's merge policy is off, so
    /// each `addIndexes` stays a segment of its own.
    fn create_rearranged_index(&self) -> Result<()> {
        let reader = DirectoryReader::open(self.input)?;
        let mut w = open_writer(self.output)?;
        for selector in &self.segment_selectors {
            let selected = reader
                .segment_readers()
                .iter()
                .map(|s| selector.get_filtered_docs(s))
                .collect::<Result<Vec<_>>>()?;
            w.add_indexes_filtered(&[self.input], &|_, segment, doc| {
                selected[segment][usize::try_from(doc).unwrap_or(0)]
            })
            .map_err(index_error)?;
        }
        w.close().map_err(index_error)
    }

    /// `finalizeRearrange`: puts the segments in the order of the selectors
    /// that made them and checks that each selector matches one segment at
    /// most. Segments are made in selector order here, where Java makes them
    /// concurrently and must sort them afterwards, so the check is all that
    /// is left to do.
    fn finalize_rearrange(&self) -> Result<()> {
        let reader = DirectoryReader::open(self.output)?;
        for (n, selector) in self.segment_selectors.iter().enumerate() {
            let mut found: Option<usize> = None;
            for (ord, segment) in reader.segment_readers().iter().enumerate() {
                if selector.get_filtered_docs(segment)?.contains(&true) {
                    if let Some(first) = found {
                        return Err(Error::IllegalState(format!(
                            "Document selector #{n} has matched more than 1 segments. \
                             Matched segments order: {first}, {ord}"
                        )));
                    }
                    found = Some(ord);
                }
            }
        }
        Ok(())
    }

    /// `applyDeletes`: deletes, by document id, what the deletes selector
    /// picks in each segment of the rearranged index.
    fn apply_deletes(&self) -> Result<()> {
        let Some(selector) = &self.deleted_docs_selector else {
            return Ok(());
        };
        let mut w = open_writer(self.output)?;
        let reader = DirectoryReader::open(self.output)?;
        let mut base = 0i32;
        for segment in reader.segment_readers() {
            let deleted = selector.get_filtered_docs(segment)?;
            for (doc, delete) in deleted.iter().enumerate() {
                if *delete {
                    let id = base + i32::try_from(doc).map_err(index_error)?;
                    if w.try_delete_document(&reader.segment_infos, id)
                        .map_err(index_error)?
                        .is_none()
                    {
                        return Err(Error::IllegalState(
                            "tryDeleteDocument has failed. This should never happen, \
                             since merging is disabled."
                                .into(),
                        ));
                    }
                }
            }
            base += segment.max_doc;
        }
        w.close().map_err(index_error)
    }
}

/// `BinaryDocValueSelector`: selects the documents whose `BinaryDocValues`
/// value, for `field`, is one of a set of keys -- for an index whose
/// documents are told apart by such a value.
#[derive(Debug, Clone)]
pub struct BinaryDocValueSelector {
    field: String,
    key_set: HashSet<Vec<u8>>,
}

impl BinaryDocValueSelector {
    /// `BinaryDocValueSelector(field, keySet)`.
    pub fn new(field: impl Into<String>, key_set: HashSet<Vec<u8>>) -> Self {
        Self {
            field: field.into(),
            key_set,
        }
    }

    /// `createDeleteSelectorFromIndex(field, directory)`: a selector of the
    /// keys of the documents `directory`'s index has deleted, to apply to a
    /// rearranged copy.
    ///
    /// # Errors
    /// [`Error::IllegalState`] for a deleted document without a key (Java's
    /// `AssertionError`).
    pub fn create_delete_selector_from_index(
        field: &str,
        directory: &dyn Directory,
    ) -> Result<Self> {
        let reader = DirectoryReader::open(directory)?;
        let mut keys = HashSet::new();
        for segment in reader.segment_readers() {
            let Some(live) = segment.live_docs() else {
                continue;
            };
            let mut values = segment.binary_doc_values(field)?;
            for doc in 0..segment.max_doc {
                if live.get(usize::try_from(doc).map_err(index_error)?) {
                    continue;
                }
                keys.insert(Self::key(values.as_deref_mut(), doc, field)?);
            }
        }
        Ok(Self::new(field, keys))
    }

    /// `createLiveSelectorsFromIndex(field, directory)`: one selector per
    /// segment of `directory`'s index, of the keys of that segment's
    /// documents (deleted ones too), which reproduce its geometry when given
    /// to an [`IndexRearranger`].
    pub fn create_live_selectors_from_index(
        field: &str,
        directory: &dyn Directory,
    ) -> Result<Vec<Self>> {
        let reader = DirectoryReader::open(directory)?;
        let mut selectors = Vec::new();
        for segment in reader.segment_readers() {
            let mut values = segment.binary_doc_values(field)?;
            let mut keys = HashSet::new();
            for doc in 0..segment.max_doc {
                keys.insert(Self::key(values.as_deref_mut(), doc, field)?);
            }
            selectors.push(Self::new(field, keys));
        }
        Ok(selectors)
    }

    /// The key of `doc`, or Java's `AssertionError` as an error.
    fn key(
        values: Option<&mut (dyn crate::reader::BinaryDocValues + '_)>,
        doc: i32,
        field: &str,
    ) -> Result<Vec<u8>> {
        if let Some(v) = values {
            if v.advance_exact(doc)? {
                return Ok(v.binary_value().to_vec());
            }
        }
        Err(Error::IllegalState(format!(
            "Document {doc} doesn't have key {field}"
        )))
    }
}

impl DocumentSelector for BinaryDocValueSelector {
    fn get_filtered_docs(&self, reader: &dyn CodecReader) -> Result<Vec<bool>> {
        let max_doc = usize::try_from(reader.max_doc()).map_err(index_error)?;
        let mut bits = vec![false; max_doc];
        let Some(mut values) = reader.binary_doc_values(&self.field)? else {
            return Ok(bits);
        };
        for (doc, bit) in bits.iter_mut().enumerate() {
            let doc = i32::try_from(doc).map_err(index_error)?;
            *bit = values.advance_exact(doc)? && self.key_set.contains(values.binary_value());
        }
        Ok(bits)
    }
}
