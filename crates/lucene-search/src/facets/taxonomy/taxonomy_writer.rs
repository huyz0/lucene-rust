//! `TaxonomyWriter`: the write side of a taxonomy, which hands each new
//! category the next ordinal (adding its ancestors first).

use super::FacetLabel;
use crate::Result;

/// `TaxonomyWriter` (with `TwoPhaseCommit` and `Closeable`). Methods take
/// `&self`: a writer is shared by the threads indexing documents, and
/// serialises itself (Java's `synchronized`).
pub trait TaxonomyWriter {
    /// `addCategory(categoryPath)`: the category's ordinal, adding it (and
    /// every missing ancestor) when it is new.
    fn add_category(&self, category_path: &FacetLabel) -> Result<i32>;

    /// `getParent(ordinal)`.
    fn get_parent(&self, ordinal: i32) -> Result<i32>;

    /// `getSize()`: the number of categories, the root included.
    fn get_size(&self) -> Result<i32>;

    /// `setLiveCommitData(commitUserData)`.
    fn set_live_commit_data(&self, commit_user_data: Vec<(String, String)>) -> Result<()>;

    /// `getLiveCommitData()`.
    fn live_commit_data(&self) -> Result<Vec<(String, String)>>;

    /// `prepareCommit()`.
    fn prepare_commit(&self) -> Result<i64>;

    /// `commit()`.
    fn commit(&self) -> Result<i64>;

    /// `rollback()`: drops everything since the last commit and closes.
    fn rollback(&self) -> Result<()>;

    /// `close()`: commits and closes.
    fn close(&self) -> Result<()>;
}
