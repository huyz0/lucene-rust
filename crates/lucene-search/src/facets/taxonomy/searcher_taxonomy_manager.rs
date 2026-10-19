//! `SearcherTaxonomyManager`: a reference manager over a searcher and a
//! taxonomy reader that are refreshed together, so a search always counts
//! against a taxonomy at least as new as its index -- near-real-time from an
//! [`NrtSource`] writer and a [`DirectoryTaxonomyWriter`], or from the
//! directories' commits.
//!
//! Differs:
//! * the "searcher" is the index's [`DirectoryReader`] (an
//!   [`crate::index_searcher::IndexSearcher`] borrows its reader's opened
//!   segments, so one is built per search): there is no `SearcherFactory`;
//! * the manager borrows its writers and directories for `'t` instead of
//!   holding them, so it is not a [`crate::reference_manager::ReferenceManager`]
//!   (whose refresher must own everything) but the same protocol: an
//!   [`Arc`] per reference, acquire clones it, release drops it;
//!   `maybeRefresh` is `true` whenever no other refresh was running; there
//!   are no refresh listeners and no `RefreshCommitSupplier`;
//! * a manager over a reader and a taxonomy reader also takes the index's
//!   directory, which Java's `DirectoryReader.openIfChanged` reads off the
//!   reader;
//! * over commits, "current" compares the latest commit's generation with
//!   the reader's (Java's `isCurrent` compares the `SegmentInfos` version;
//!   a new commit changes both).

use std::sync::{Arc, Mutex, RwLock};

use lucene_index::nrt::NrtSource;
use lucene_store::Directory;

use super::directory::directory_taxonomy_writer::TaxoNrtSource;
use super::{DirectoryTaxonomyReader, DirectoryTaxonomyWriter};
use crate::directory_reader::DirectoryReader;
use crate::{Error, Result};

/// `SearcherTaxonomyManager.SearcherAndTaxonomy`.
#[derive(Debug)]
pub struct SearcherAndTaxonomy<'t> {
    /// `searcher`: its reader.
    pub searcher: DirectoryReader,
    /// `taxonomyReader`, shared with the next reference when only the index
    /// changed.
    pub taxonomy_reader: Arc<DirectoryTaxonomyReader<'t>>,
}

/// Where the index reader refreshes from.
enum Source<'t> {
    /// `DirectoryReader.openIfChanged(reader, writer, applyAllDeletes)`.
    Writer {
        writer: &'t dyn NrtSource,
        apply_all_deletes: bool,
    },
    /// `DirectoryReader.openIfChanged(reader)`: the latest commit.
    Directory(&'t dyn Directory),
}

/// `SearcherTaxonomyManager`.
pub struct SearcherTaxonomyManager<'t> {
    current: RwLock<Option<Arc<SearcherAndTaxonomy<'t>>>>,
    /// `refreshLock`.
    refresh_lock: Mutex<()>,
    source: Source<'t>,
    /// `taxoWriter`, for the `replaceTaxonomy` check.
    taxo_writer: Option<&'t dyn TaxoNrtSource>,
    /// `taxoEpoch`: the writer's epoch when the manager was made, `-1`
    /// without one.
    taxo_epoch: i64,
}

impl std::fmt::Debug for SearcherTaxonomyManager<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SearcherTaxonomyManager")
            .field("taxo_epoch", &self.taxo_epoch)
            .finish_non_exhaustive()
    }
}

fn closed_error() -> Error {
    Error::AlreadyClosed("this ReferenceManager is closed".to_string())
}

impl<'t> SearcherTaxonomyManager<'t> {
    /// `new SearcherTaxonomyManager(writer, searcherFactory, taxoWriter)`:
    /// near-real-time, every buffered delete applied.
    ///
    /// # Errors
    /// Whatever opening the readers reports.
    pub fn from_writers<'d: 't>(
        writer: &'t dyn NrtSource,
        taxo_writer: &'t DirectoryTaxonomyWriter<'d>,
    ) -> Result<Self> {
        Self::from_writers_with(writer, true, taxo_writer)
    }

    /// `new SearcherTaxonomyManager(writer, applyAllDeletes,
    /// searcherFactory, taxoWriter)`.
    ///
    /// # Errors
    /// Whatever opening the readers reports.
    pub fn from_writers_with<'d: 't>(
        writer: &'t dyn NrtSource,
        apply_all_deletes: bool,
        taxo_writer: &'t DirectoryTaxonomyWriter<'d>,
    ) -> Result<Self> {
        let taxonomy_reader = DirectoryTaxonomyReader::from_writer(taxo_writer)?;
        let searcher = DirectoryReader::open_nrt(writer, apply_all_deletes, false)?;
        let taxo_epoch = taxo_writer.taxonomy_epoch();
        Ok(Self::over(
            SearcherAndTaxonomy {
                searcher,
                taxonomy_reader: Arc::new(taxonomy_reader),
            },
            Source::Writer {
                writer,
                apply_all_deletes,
            },
            Some(taxo_writer as &dyn TaxoNrtSource),
            taxo_epoch,
        ))
    }

    /// `new SearcherTaxonomyManager(indexDir, taxoDir, searcherFactory)`:
    /// the directories' latest commits.
    ///
    /// # Errors
    /// Whatever opening the readers reports.
    pub fn from_directories(
        index_dir: &'t dyn Directory,
        taxo_dir: &'t dyn Directory,
    ) -> Result<Self> {
        let taxonomy_reader = DirectoryTaxonomyReader::open(taxo_dir)?;
        let searcher = DirectoryReader::open(index_dir)?;
        Ok(Self::from_readers(index_dir, searcher, taxonomy_reader))
    }

    /// `new SearcherTaxonomyManager(reader, taxoReader, searcherFactory)`:
    /// `reader` is of `index_dir`, and refreshes from its commits.
    pub fn from_readers(
        index_dir: &'t dyn Directory,
        reader: DirectoryReader,
        taxonomy_reader: DirectoryTaxonomyReader<'t>,
    ) -> Self {
        Self::over(
            SearcherAndTaxonomy {
                searcher: reader,
                taxonomy_reader: Arc::new(taxonomy_reader),
            },
            Source::Directory(index_dir),
            None,
            -1,
        )
    }

    fn over(
        first: SearcherAndTaxonomy<'t>,
        source: Source<'t>,
        taxo_writer: Option<&'t dyn TaxoNrtSource>,
        taxo_epoch: i64,
    ) -> Self {
        SearcherTaxonomyManager {
            current: RwLock::new(Some(Arc::new(first))),
            refresh_lock: Mutex::new(()),
            source,
            taxo_writer,
            taxo_epoch,
        }
    }

    fn current(&self) -> Option<Arc<SearcherAndTaxonomy<'t>>> {
        self.current
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// `acquire()`.
    ///
    /// # Errors
    /// [`Error::AlreadyClosed`] after [`Self::close`].
    pub fn acquire(&self) -> Result<Arc<SearcherAndTaxonomy<'t>>> {
        self.current().ok_or_else(closed_error)
    }

    /// `release(reference)`.
    pub fn release(&self, reference: Arc<SearcherAndTaxonomy<'t>>) {
        drop(reference);
    }

    /// `close()`: later acquires fail; references already out stay usable.
    pub fn close(&self) {
        *self.current.write().unwrap_or_else(|e| e.into_inner()) = None;
    }

    /// `maybeRefresh()`: `false` when another refresh is running, else
    /// `true` once this one ran (whether or not anything changed).
    ///
    /// # Errors
    /// [`Error::IllegalState`] when the taxonomy writer's
    /// `replaceTaxonomy` ran since the manager was made; whatever reopening
    /// reports.
    pub fn maybe_refresh(&self) -> Result<bool> {
        let Ok(_guard) = self.refresh_lock.try_lock() else {
            return Ok(false);
        };
        self.do_maybe_refresh()?;
        Ok(true)
    }

    /// `maybeRefreshBlocking()`.
    ///
    /// # Errors
    /// As [`Self::maybe_refresh`].
    pub fn maybe_refresh_blocking(&self) -> Result<()> {
        let _guard = self.refresh_lock.lock().unwrap_or_else(|e| e.into_inner());
        self.do_maybe_refresh()
    }

    fn do_maybe_refresh(&self) -> Result<()> {
        let current = self.acquire()?;
        if let Some(new) = self.refresh_if_needed(&current)? {
            let mut cur = self.current.write().unwrap_or_else(|e| e.into_inner());
            if cur.is_none() {
                return Err(closed_error());
            }
            *cur = Some(Arc::new(new));
        }
        Ok(())
    }

    /// `refreshIfNeeded(ref)`.
    fn refresh_if_needed(
        &self,
        current: &SearcherAndTaxonomy<'t>,
    ) -> Result<Option<SearcherAndTaxonomy<'t>>> {
        let r = &current.searcher;
        let new_reader = match &self.source {
            Source::Writer {
                writer,
                apply_all_deletes,
            } => r.open_if_changed_nrt_with(*writer, *apply_all_deletes, false)?,
            Source::Directory(dir) => r.open_if_changed(*dir)?,
        };
        let Some(new_reader) = new_reader else {
            return Ok(None);
        };
        let taxonomy_reader = match current.taxonomy_reader.open_if_changed()? {
            None => Arc::clone(&current.taxonomy_reader),
            Some(tr) => {
                if self
                    .taxo_writer
                    .is_some_and(|w| w.taxonomy_epoch() != self.taxo_epoch)
                {
                    return Err(Error::IllegalState(
                        "DirectoryTaxonomyWriter.replaceTaxonomy was called, which is not \
                         allowed when using SearcherTaxonomyManager"
                            .into(),
                    ));
                }
                Arc::new(tr)
            }
        };
        Ok(Some(SearcherAndTaxonomy {
            searcher: new_reader,
            taxonomy_reader,
        }))
    }

    /// `getSearcherCommitGeneration()`.
    ///
    /// # Errors
    /// [`Error::AlreadyClosed`] after [`Self::close`].
    pub fn searcher_commit_generation(&self) -> Result<i64> {
        Ok(self.acquire()?.searcher.segment_infos.generation)
    }

    /// `isSearcherCurrent()`: the index reader sees everything (the writer
    /// indexed, or the directory's latest commit).
    ///
    /// # Errors
    /// [`Error::AlreadyClosed`] after [`Self::close`]; whatever reading the
    /// latest commit reports.
    pub fn is_searcher_current(&self) -> Result<bool> {
        let sat = self.acquire()?;
        match &self.source {
            Source::Writer { writer, .. } => Ok(sat.searcher.is_current_nrt(*writer)?),
            Source::Directory(dir) => {
                let latest = lucene_index::segment_infos::read_latest(*dir)
                    .map_err(crate::directory_reader::Error::from)?;
                Ok(latest.generation == sat.searcher.segment_infos.generation)
            }
        }
    }

    /// `isTaxonomyCurrent()`.
    ///
    /// # Errors
    /// As [`Self::is_searcher_current`].
    pub fn is_taxonomy_current(&self) -> Result<bool> {
        self.acquire()?.taxonomy_reader.is_current()
    }
}

#[cfg(test)]
mod tests;
