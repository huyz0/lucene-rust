//! `ReindexingEnrichedDirectoryTaxonomyWriter`: a [`DirectoryTaxonomyWriter`]
//! that adds caller fields to each category's document (an
//! [`OrdinalDataAppender`]), and can rewrite every category with new ones
//! while keeping each category's ordinal.
//!
//! Differs: the writer is a wrapper that dereferences to the
//! [`DirectoryTaxonomyWriter`] rather than a subclass, and the hook is
//! installed after the inner writer is built -- as in Java, where the field
//! is assigned after `super(d)` added the root, so the root's first
//! document carries no fields of the appender's.

use std::ops::Deref;

use lucene_store::Directory;

use super::directory_taxonomy_writer::{DirectoryTaxonomyWriter, OrdinalDataAppender};
use super::DirectoryTaxonomyReader;
use crate::facets::taxonomy::{FacetLabel, TaxonomyReader, TaxonomyWriter};
use crate::reader::IndexReader;
use crate::Result;

/// `ReindexingEnrichedDirectoryTaxonomyWriter`.
pub struct ReindexingEnrichedDirectoryTaxonomyWriter<'d> {
    inner: DirectoryTaxonomyWriter<'d>,
}

impl std::fmt::Debug for ReindexingEnrichedDirectoryTaxonomyWriter<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReindexingEnrichedDirectoryTaxonomyWriter")
            .finish_non_exhaustive()
    }
}

impl<'d> ReindexingEnrichedDirectoryTaxonomyWriter<'d> {
    /// `new ReindexingEnrichedDirectoryTaxonomyWriter(d,
    /// ordinalDataAppender)`.
    ///
    /// # Errors
    /// As [`DirectoryTaxonomyWriter::open`].
    pub fn new(dir: &'d dyn Directory, appender: Option<OrdinalDataAppender>) -> Result<Self> {
        let inner = DirectoryTaxonomyWriter::open(dir)?;
        inner.set_enrich(appender);
        Ok(ReindexingEnrichedDirectoryTaxonomyWriter { inner })
    }

    /// The writer underneath.
    pub fn writer(&self) -> &DirectoryTaxonomyWriter<'d> {
        &self.inner
    }

    /// `reindexWithNewOrdinalData(ordinalDataAppender)`: every category of
    /// the last commit added again, in ordinal order (so each keeps its
    /// ordinal) and through `appender`, then a commit. The taxonomy's epoch
    /// moves on, as `deleteAll` moves it.
    ///
    /// # Errors
    /// [`crate::Error::AlreadyClosed`] after close; whatever reading the
    /// commit, adding or committing reports.
    pub fn reindex_with_new_ordinal_data(
        &self,
        appender: Option<OrdinalDataAppender>,
    ) -> Result<()> {
        self.inner.ensure_open()?;
        self.inner.set_enrich(appender);
        let ordinal_to_path = record_paths_in_order(self.inner.directory())?;
        self.inner.delete_all()?;
        for path in &ordinal_to_path {
            self.inner.add_category(path)?;
        }
        self.inner.commit()?;
        Ok(())
    }
}

impl<'d> Deref for ReindexingEnrichedDirectoryTaxonomyWriter<'d> {
    type Target = DirectoryTaxonomyWriter<'d>;

    fn deref(&self) -> &DirectoryTaxonomyWriter<'d> {
        &self.inner
    }
}

/// `recordPathsInOrder(d)`: the last commit's categories, by ordinal.
fn record_paths_in_order(dir: &dyn Directory) -> Result<Vec<FacetLabel>> {
    let taxo = DirectoryTaxonomyReader::open(dir)?;
    let mut paths = Vec::new();
    for ctx in taxo.internal_index_reader()?.leaves() {
        let ordinals: Vec<i32> = (0..ctx.reader.max_doc())
            .map(|i| ctx.doc_base.saturating_add(i))
            .collect();
        paths.extend(taxo.get_bulk_path(&ordinals)?);
    }
    taxo.close()?;
    Ok(paths)
}

#[cfg(test)]
mod tests;
