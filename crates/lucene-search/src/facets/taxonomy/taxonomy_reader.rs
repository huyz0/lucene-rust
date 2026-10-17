//! `TaxonomyReader`: the read side of a taxonomy -- ordinal <-> path, the
//! parent/child arrays, and Java's reference counting.

use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::Arc;

use super::parallel_taxonomy_arrays::ParallelTaxonomyArrays;
use super::FacetLabel;
use crate::{Error, Result};

/// `TaxonomyReader.ROOT_ORDINAL`.
pub const ROOT_ORDINAL: i32 = 0;

/// `TaxonomyReader.INVALID_ORDINAL`.
pub const INVALID_ORDINAL: i32 = -1;

/// `TaxonomyReader.ChildrenIterator`: an ordinal's children, youngest
/// (largest ordinal) first. As an [`Iterator`] it ends where Java's `next()`
/// starts answering [`INVALID_ORDINAL`].
#[derive(Debug, Clone)]
pub struct ChildrenIterator {
    arrays: Arc<dyn ParallelTaxonomyArrays>,
    child: i32,
}

impl ChildrenIterator {
    pub(crate) fn new(child: i32, arrays: Arc<dyn ParallelTaxonomyArrays>) -> Self {
        ChildrenIterator { arrays, child }
    }

    /// Java's `next()`: the next child, or [`INVALID_ORDINAL`] once there
    /// are no more.
    pub fn next_ordinal(&mut self) -> i32 {
        let res = self.child;
        if self.child != INVALID_ORDINAL {
            self.child = self.arrays.siblings().get(self.child);
        }
        res
    }
}

impl Iterator for ChildrenIterator {
    type Item = i32;

    fn next(&mut self) -> Option<i32> {
        match self.next_ordinal() {
            INVALID_ORDINAL => None,
            ord => Some(ord),
        }
    }
}

/// `TaxonomyReader`'s `refCount` and `closed`.
#[derive(Debug)]
pub struct RefCount {
    count: AtomicI32,
    closed: AtomicBool,
}

impl Default for RefCount {
    fn default() -> Self {
        RefCount {
            count: AtomicI32::new(1),
            closed: AtomicBool::new(false),
        }
    }
}

/// `TaxonomyReader`.
pub trait TaxonomyReader {
    /// The reader's reference count.
    fn ref_counter(&self) -> &RefCount;

    /// `doClose()`: the reference count reached zero.
    fn do_close(&self) -> Result<()>;

    /// `getParallelTaxonomyArrays()`.
    fn parallel_taxonomy_arrays(&self) -> Result<Arc<dyn ParallelTaxonomyArrays>>;

    /// `getCommitUserData()`.
    fn commit_user_data(&self) -> Result<Vec<(String, String)>>;

    /// `getOrdinal(categoryPath)`: the ordinal, or [`INVALID_ORDINAL`].
    fn get_ordinal(&self, category_path: &FacetLabel) -> Result<i32>;

    /// `getPath(ordinal)`.
    fn get_path(&self, ordinal: i32) -> Result<FacetLabel>;

    /// `getSize()`: the number of categories, the root included.
    fn get_size(&self) -> Result<i32>;

    /// `ensureOpen()`.
    fn ensure_open(&self) -> Result<()> {
        if self.ref_count() <= 0 {
            return Err(Error::AlreadyClosed(
                "this TaxonomyReader is closed".to_string(),
            ));
        }
        Ok(())
    }

    /// `getRefCount()`.
    fn ref_count(&self) -> i32 {
        self.ref_counter().count.load(Ordering::SeqCst)
    }

    /// `incRef()`.
    fn inc_ref(&self) -> Result<()> {
        self.ensure_open()?;
        self.ref_counter().count.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    /// `tryIncRef()`.
    fn try_inc_ref(&self) -> bool {
        let rc = &self.ref_counter().count;
        let mut count = rc.load(Ordering::SeqCst);
        while count > 0 {
            match rc.compare_exchange(
                count,
                count.saturating_add(1),
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => return true,
                Err(now) => count = now,
            }
        }
        false
    }

    /// `decRef()`: closes the reader when the count reaches zero.
    fn dec_ref(&self) -> Result<()> {
        self.ensure_open()?;
        let rc = self
            .ref_counter()
            .count
            .fetch_sub(1, Ordering::SeqCst)
            .saturating_sub(1);
        if rc == 0 {
            if let Err(e) = self.do_close() {
                self.ref_counter().count.fetch_add(1, Ordering::SeqCst);
                return Err(e);
            }
            self.ref_counter().closed.store(true, Ordering::SeqCst);
        } else if rc < 0 {
            return Err(Error::IllegalState(format!(
                "too many decRef calls: refCount is {rc} after decrement"
            )));
        }
        Ok(())
    }

    /// `close()`: drops the caller's reference, once.
    fn close(&self) -> Result<()> {
        let closed = &self.ref_counter().closed;
        if !closed.load(Ordering::SeqCst) {
            self.dec_ref()?;
            closed.store(true, Ordering::SeqCst);
        }
        Ok(())
    }

    /// `getChildren(ordinal)`: a negative ordinal has none.
    fn get_children(&self, ordinal: i32) -> Result<ChildrenIterator> {
        let arrays = self.parallel_taxonomy_arrays()?;
        let child = if ordinal >= 0 {
            if ordinal >= arrays.children().length() {
                return Err(Error::IllegalArgument(format!(
                    "Index {ordinal} out of bounds for length {}",
                    arrays.children().length()
                )));
            }
            arrays.children().get(ordinal)
        } else {
            INVALID_ORDINAL
        };
        Ok(ChildrenIterator::new(child, arrays))
    }

    /// `getBulkOrdinals(categoryPaths...)`.
    fn get_bulk_ordinals(&self, category_paths: &[FacetLabel]) -> Result<Vec<i32>> {
        category_paths.iter().map(|p| self.get_ordinal(p)).collect()
    }

    /// `getOrdinal(dim, path...)`.
    fn get_ordinal_of(&self, dim: &str, path: &[&str]) -> Result<i32> {
        self.get_ordinal(&FacetLabel::from_dim_path(dim, path)?)
    }

    /// `getBulkPath(ordinals...)`.
    fn get_bulk_path(&self, ordinals: &[i32]) -> Result<Vec<FacetLabel>> {
        ordinals.iter().map(|&o| self.get_path(o)).collect()
    }
}

#[cfg(test)]
mod tests;
