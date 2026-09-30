//! Port of `org.apache.lucene.index.ReaderManager`: a
//! [`ReferenceManager`] of [`DirectoryReader`]s whose refresh is
//! `DirectoryReader.openIfChanged` -- the `ReaderManager(Directory)`,
//! `ReaderManager(DirectoryReader)` and near-real-time
//! `ReaderManager(IndexWriter[, applyAllDeletes, writeAllDeletes])`
//! constructors.
//!
//! The reference counting, refresh locking and listeners are the one
//! generic [`ReferenceManager`] in [`crate::reference_manager`], which
//! [`crate::reference_manager::SearcherManager`] shares; a `ReaderManager`
//! only supplies the refresh (Java's `ReaderManager extends
//! ReferenceManager<DirectoryReader>`), and derefs to its manager.
//!
//! The writer form takes any [`NrtSource`] (a `Mutex<IndexWriter>` or a
//! `ConcurrentIndexWriter`): each refresh is
//! `DirectoryReader.openIfChanged(reader, writer, applyAllDeletes,
//! writeAllDeletes)` ([`DirectoryReader::open_if_changed_nrt_with`]).

use std::sync::Arc;

use lucene_index::nrt::NrtSource;
use lucene_store::directory::Directory;

use crate::directory_reader::DirectoryReader;
pub use crate::reference_manager::{ReferenceManager, RefreshListener, Refresher};
use crate::Result;

/// `ReaderManager.refreshIfNeeded`: `DirectoryReader.openIfChanged`.
struct DirectoryRefresher {
    dir: Arc<dyn Directory>,
}

impl Refresher<DirectoryReader> for DirectoryRefresher {
    fn refresh_if_needed(
        &self,
        current: &Arc<DirectoryReader>,
    ) -> Result<Option<Arc<DirectoryReader>>> {
        Ok(current.open_if_changed(self.dir.as_ref())?.map(Arc::new))
    }
}

/// `ReaderManager.refreshIfNeeded` over a writer:
/// `DirectoryReader.openIfChanged(reader, writer, ...)`.
struct NrtRefresher {
    writer: Arc<dyn NrtSource>,
    apply_all_deletes: bool,
    write_all_deletes: bool,
}

impl Refresher<DirectoryReader> for NrtRefresher {
    fn refresh_if_needed(
        &self,
        current: &Arc<DirectoryReader>,
    ) -> Result<Option<Arc<DirectoryReader>>> {
        Ok(current
            .open_if_changed_nrt_with(
                self.writer.as_ref(),
                self.apply_all_deletes,
                self.write_all_deletes,
            )?
            .map(Arc::new))
    }
}

/// `ReaderManager`: every refresh opens the latest commit and reuses every
/// unchanged segment of the reader it replaces.
pub struct ReaderManager {
    manager: ReferenceManager<DirectoryReader>,
}

impl ReaderManager {
    /// `new ReaderManager(dir)`: starts from the latest commit in `dir`.
    ///
    /// # Errors
    /// Opening the first reader fails.
    pub fn open(dir: Arc<dyn Directory>) -> Result<Self> {
        let reader = DirectoryReader::open(dir.as_ref())?;
        Ok(Self::from_reader(dir, reader))
    }

    /// `new ReaderManager(reader)`: starts from `reader`, refreshing from
    /// `dir`, the directory it was opened on.
    pub fn from_reader(dir: Arc<dyn Directory>, reader: DirectoryReader) -> Self {
        Self {
            manager: ReferenceManager::new(Arc::new(reader), Box::new(DirectoryRefresher { dir })),
        }
    }

    /// `new ReaderManager(writer)`: near-real-time over `writer`, applying
    /// every buffered delete on each refresh and writing none.
    ///
    /// # Errors
    /// Opening the first reader fails.
    pub fn open_from_writer(writer: Arc<dyn NrtSource>) -> Result<Self> {
        Self::from_writer(writer, true, false)
    }

    /// `new ReaderManager(writer, applyAllDeletes, writeAllDeletes)`.
    ///
    /// # Errors
    /// Opening the first reader fails.
    pub fn from_writer(
        writer: Arc<dyn NrtSource>,
        apply_all_deletes: bool,
        write_all_deletes: bool,
    ) -> Result<Self> {
        let reader =
            DirectoryReader::open_nrt(writer.as_ref(), apply_all_deletes, write_all_deletes)?;
        Ok(Self {
            manager: ReferenceManager::new(
                Arc::new(reader),
                Box::new(NrtRefresher {
                    writer,
                    apply_all_deletes,
                    write_all_deletes,
                }),
            ),
        })
    }
}

impl std::ops::Deref for ReaderManager {
    type Target = ReferenceManager<DirectoryReader>;

    fn deref(&self) -> &Self::Target {
        &self.manager
    }
}
