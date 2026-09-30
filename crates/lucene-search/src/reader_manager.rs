//! Port of `org.apache.lucene.index.ReaderManager`: a
//! [`ReferenceManager`] of [`DirectoryReader`]s whose refresh is
//! `DirectoryReader.openIfChanged` -- the `ReaderManager(Directory)` and
//! `ReaderManager(DirectoryReader)` constructors.
//!
//! The reference counting, refresh locking and listeners are the one
//! generic [`ReferenceManager`] in [`crate::reference_manager`], which
//! [`crate::reference_manager::SearcherManager`] shares; a `ReaderManager`
//! only supplies the refresh (Java's `ReaderManager extends
//! ReferenceManager<DirectoryReader>`), and derefs to its manager.
//!
//! The `ReaderManager(IndexWriter)` (near-real-time) form is not here: this
//! port has no NRT reader over a live writer (`docs/parity.md`,
//! `DirectoryReader`).

use std::sync::Arc;

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
}

impl std::ops::Deref for ReaderManager {
    type Target = ReferenceManager<DirectoryReader>;

    fn deref(&self) -> &Self::Target {
        &self.manager
    }
}
