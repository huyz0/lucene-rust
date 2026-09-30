//! Port of `org.apache.lucene.index.MultiReader` and the
//! `BaseCompositeReader` it extends: a composite over independent readers --
//! leaves or composites, typically several indexes' `DirectoryReader`s --
//! whose documents are numbered one after another.
//!
//! `closeSubReaders` is ownership here: a [`MultiReader`] holds its
//! children by `Arc`, and whether dropping it "closes" them is whether it
//! held the last handle.

use super::{
    composite_leaves, CacheHelper, CompositeReader, IndexReader, LeafReaderContext, ReaderHandle,
    SubReader,
};
use crate::{Error, Result};

/// `IndexWriter.MAX_DOCS`: `Integer.MAX_VALUE - 128`.
pub const MAX_DOCS: i64 = i32::MAX as i64 - 128;

/// `MultiReader`.
pub struct MultiReader {
    subs: Vec<ReaderHandle>,
    /// `starts`: each child's first document, then `maxDoc`.
    starts: Vec<i32>,
    max_doc: i32,
}

impl MultiReader {
    /// `new MultiReader(subReaders)`.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] when the children hold more than
    /// [`MAX_DOCS`] documents together ("Too many documents").
    pub fn new(subs: Vec<ReaderHandle>) -> Result<Self> {
        let mut starts = Vec::with_capacity(subs.len() + 1);
        let mut max_doc: i64 = 0;
        for s in &subs {
            starts.push(max_doc as i32);
            max_doc += i64::from(s.max_doc());
            if max_doc > MAX_DOCS {
                return Err(Error::IllegalArgument(format!(
                    "Too many documents: composite IndexReaders cannot exceed {MAX_DOCS} but \
                     readers have total maxDoc={max_doc}"
                )));
            }
        }
        let max_doc = max_doc as i32;
        starts.push(max_doc);
        Ok(Self {
            subs,
            starts,
            max_doc,
        })
    }

    /// `new MultiReader(subReaders, subReadersSorter, closeSubReaders)`: the
    /// children in `cmp`'s order.
    ///
    /// # Errors
    /// As [`Self::new`].
    pub fn with_sorter(
        mut subs: Vec<ReaderHandle>,
        cmp: impl FnMut(&ReaderHandle, &ReaderHandle) -> std::cmp::Ordering,
    ) -> Result<Self> {
        subs.sort_by(cmp);
        Self::new(subs)
    }

    /// The children, in document order.
    pub fn sub_readers(&self) -> &[ReaderHandle] {
        &self.subs
    }

    /// `readerIndex(docID)`: the child holding top-level `doc`.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] for a document outside the reader.
    pub fn reader_index(&self, doc: i32) -> Result<usize> {
        if doc < 0 || doc >= self.max_doc {
            return Err(Error::IllegalArgument(format!(
                "docID must be >= 0 and < maxDoc={} (got docID={doc})",
                self.max_doc
            )));
        }
        Ok(self.starts[..self.subs.len()]
            .partition_point(|&s| s <= doc)
            .saturating_sub(1))
    }

    /// `readerBase(readerIndex)`.
    pub fn reader_base(&self, index: usize) -> Option<i32> {
        self.starts.get(index).copied()
    }
}

impl IndexReader for MultiReader {
    fn max_doc(&self) -> i32 {
        self.max_doc
    }
    fn num_docs(&self) -> i32 {
        self.subs
            .iter()
            .fold(0i32, |a, s| a.saturating_add(s.num_docs()))
    }
    fn leaves(&self) -> Vec<LeafReaderContext<'_>> {
        composite_leaves(&self.sequential_sub_readers())
    }
    /// Only a single child's helper: a composite of several has no key that
    /// outlives a change to any one of them.
    fn reader_cache_helper(&self) -> Option<&CacheHelper> {
        match self.subs.as_slice() {
            [ReaderHandle::Leaf(l)] => l.reader_cache_helper(),
            [ReaderHandle::Composite(c)] => c.reader_cache_helper(),
            _ => None,
        }
    }
}

impl CompositeReader for MultiReader {
    fn sequential_sub_readers(&self) -> Vec<SubReader<'_>> {
        self.subs.iter().map(ReaderHandle::as_sub).collect()
    }
    fn leaf_handles(&self) -> Vec<std::sync::Arc<dyn super::LeafReader>> {
        self.subs
            .iter()
            .flat_map(|s| match s {
                ReaderHandle::Leaf(l) => vec![std::sync::Arc::clone(l)],
                ReaderHandle::Composite(c) => c.leaf_handles(),
            })
            .collect()
    }
}
