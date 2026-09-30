//! Port of `org.apache.lucene.search.ReferenceManager` and
//! `org.apache.lucene.index.ReaderManager`: hand out the current reader to
//! any number of threads, and swap in a fresh one when the index changes,
//! with one refresh at a time.
//!
//! # Shape
//!
//! Java counts references by hand (`tryIncRef`/`decRef`, `acquire` looping
//! until an increment sticks); here the current reference is an `Arc`, so
//! [`ReferenceManager::acquire`] is a clone and [`ReferenceManager::release`]
//! a drop -- a reader swapped out stays usable by whoever still holds it,
//! and closes when the last holder lets go, which is what Java's counting
//! achieves.
//!
//! [`ReaderManager`] refreshes through [`DirectoryReader::open_if_changed`]
//! -- the `ReaderManager(Directory)`/`ReaderManager(DirectoryReader)`
//! constructors. The `ReaderManager(IndexWriter)` (near-real-time) form is
//! not here: this port has no NRT reader over a live writer
//! (`docs/parity.md`, `DirectoryReader`).

use std::sync::{Arc, Mutex, MutexGuard, RwLock};

use lucene_store::directory::Directory;

use crate::directory_reader::{self, DirectoryReader};

/// What a [`ReferenceManager`] reports.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// `AlreadyClosedException("this ReferenceManager is closed")`.
    #[error("this ReferenceManager is closed")]
    AlreadyClosed,
    /// Opening the refreshed reader failed.
    #[error(transparent)]
    Reader(#[from] directory_reader::Error),
    /// A [`RefreshListener`] failed (Java's `IOException` out of it).
    #[error("refresh listener: {0}")]
    Listener(String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// `ReferenceManager.RefreshListener`: told before every refresh attempt and
/// after it, with whether a new reference was swapped in.
pub trait RefreshListener: Send + Sync {
    /// `beforeRefresh()`.
    fn before_refresh(&self) -> std::result::Result<(), String>;
    /// `afterRefresh(didRefresh)`: runs however the refresh ended, failures
    /// included (`false` then).
    fn after_refresh(&self, did_refresh: bool) -> std::result::Result<(), String>;
}

/// How a [`ReferenceManager`] makes a new reference: `refreshIfNeeded`.
pub trait Refresher<G>: Send + Sync {
    /// `refreshIfNeeded(referenceToRefresh)`: a new reference if `current`
    /// is stale, `None` if it is current.
    fn refresh_if_needed(&self, current: &G) -> Result<Option<G>>;
}

/// `ReferenceManager<G>`.
pub struct ReferenceManager<G, R: Refresher<G>> {
    /// `current`; `None` once closed.
    current: RwLock<Option<Arc<G>>>,
    /// `refreshLock`: one refresh at a time.
    refresh_lock: Mutex<()>,
    /// `refreshListeners`.
    listeners: RwLock<Vec<Arc<dyn RefreshListener>>>,
    refresher: R,
}

fn read<T>(l: &RwLock<T>) -> std::sync::RwLockReadGuard<'_, T> {
    l.read().unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn write<T>(l: &RwLock<T>) -> std::sync::RwLockWriteGuard<'_, T> {
    l.write().unwrap_or_else(std::sync::PoisonError::into_inner)
}

impl<G, R: Refresher<G>> ReferenceManager<G, R> {
    /// A manager whose current reference is `initial`.
    pub fn new(initial: G, refresher: R) -> Self {
        ReferenceManager {
            current: RwLock::new(Some(Arc::new(initial))),
            refresh_lock: Mutex::new(()),
            listeners: RwLock::new(Vec::new()),
            refresher,
        }
    }

    /// `acquire()`: the current reference, held until the returned `Arc` is
    /// dropped (or passed to [`Self::release`]).
    pub fn acquire(&self) -> Result<Arc<G>> {
        read(&self.current).clone().ok_or(Error::AlreadyClosed)
    }

    /// `release(reference)`: gives back what [`Self::acquire`] handed out.
    pub fn release(&self, reference: Arc<G>) {
        drop(reference);
    }

    /// `maybeRefresh()`: refreshes unless another thread is refreshing right
    /// now, in which case it returns `false` at once. `true` means this call
    /// did the check (and swapped in a new reference if one was due).
    pub fn maybe_refresh(&self) -> Result<bool> {
        self.ensure_open()?;
        let guard = match self.refresh_lock.try_lock() {
            Ok(guard) => guard,
            Err(std::sync::TryLockError::Poisoned(p)) => p.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => return Ok(false),
        };
        self.do_maybe_refresh(guard)?;
        Ok(true)
    }

    /// `maybeRefreshBlocking()`: waits for any refresh in progress, then
    /// refreshes.
    pub fn maybe_refresh_blocking(&self) -> Result<()> {
        self.ensure_open()?;
        let guard = self
            .refresh_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.do_maybe_refresh(guard)
    }

    /// `doMaybeRefresh()`, holding the refresh lock.
    fn do_maybe_refresh(&self, _guard: MutexGuard<'_, ()>) -> Result<()> {
        let reference = self.acquire()?;
        let mut refreshed = false;
        let outcome = (|| {
            self.notify_before()?;
            if let Some(new_reference) = self.refresher.refresh_if_needed(&reference)? {
                self.swap_reference(new_reference)?;
                refreshed = true;
            }
            Ok(())
        })();
        self.release(reference);
        // In Java's `finally`: listeners hear the outcome however it went.
        let after = self.notify_after(refreshed);
        outcome.and(after)
    }

    /// `swapReference(newReference)`: the old reference is released; a
    /// closed manager refuses (and drops `new_reference`).
    fn swap_reference(&self, new_reference: G) -> Result<()> {
        let mut current = write(&self.current);
        if current.is_none() {
            return Err(Error::AlreadyClosed);
        }
        *current = Some(Arc::new(new_reference));
        Ok(())
    }

    fn ensure_open(&self) -> Result<()> {
        if read(&self.current).is_none() {
            return Err(Error::AlreadyClosed);
        }
        Ok(())
    }

    /// `close()`: releases the current reference; `acquire` and refreshes
    /// fail from here on. References already acquired stay usable. Closing
    /// twice is a no-op, as in Java.
    pub fn close(&self) {
        write(&self.current).take();
    }

    /// `addListener(listener)`.
    pub fn add_listener(&self, listener: Arc<dyn RefreshListener>) {
        write(&self.listeners).push(listener);
    }

    /// `removeListener(listener)`: the first registration of this very
    /// listener (by identity).
    pub fn remove_listener(&self, listener: &Arc<dyn RefreshListener>) {
        let mut listeners = write(&self.listeners);
        if let Some(i) = listeners.iter().position(|l| Arc::ptr_eq(l, listener)) {
            listeners.remove(i);
        }
    }

    fn notify_before(&self) -> Result<()> {
        let listeners = read(&self.listeners).clone();
        for l in listeners {
            l.before_refresh().map_err(Error::Listener)?;
        }
        Ok(())
    }

    fn notify_after(&self, did_refresh: bool) -> Result<()> {
        let listeners = read(&self.listeners).clone();
        for l in listeners {
            l.after_refresh(did_refresh).map_err(Error::Listener)?;
        }
        Ok(())
    }
}

/// `ReaderManager`'s `refreshIfNeeded`: `DirectoryReader.openIfChanged`.
pub struct DirectoryRefresher<'d> {
    dir: &'d dyn Directory,
}

impl Refresher<DirectoryReader> for DirectoryRefresher<'_> {
    fn refresh_if_needed(&self, current: &DirectoryReader) -> Result<Option<DirectoryReader>> {
        Ok(current.open_if_changed(self.dir)?)
    }
}

/// `ReaderManager`: a [`ReferenceManager`] of [`DirectoryReader`]s over one
/// directory, each refresh opening the latest commit and reusing every
/// unchanged segment of the reader it replaces.
pub type ReaderManager<'d> = ReferenceManager<DirectoryReader, DirectoryRefresher<'d>>;

impl<'d> ReferenceManager<DirectoryReader, DirectoryRefresher<'d>> {
    /// `new ReaderManager(dir)`: starts from the latest commit in `dir`.
    pub fn open(dir: &'d dyn Directory) -> Result<Self> {
        Ok(Self::from_reader(dir, DirectoryReader::open(dir)?))
    }

    /// `new ReaderManager(reader)`: starts from `reader`, refreshing from
    /// `dir`, the directory it was opened on.
    pub fn from_reader(dir: &'d dyn Directory, reader: DirectoryReader) -> Self {
        ReferenceManager::new(reader, DirectoryRefresher { dir })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A refresher over a counter: a new reference whenever `source` moved.
    struct Counter {
        source: AtomicUsize,
        fail: std::sync::atomic::AtomicBool,
    }

    impl Refresher<usize> for &Counter {
        fn refresh_if_needed(&self, current: &usize) -> Result<Option<usize>> {
            if self.fail.load(Ordering::SeqCst) {
                return Err(Error::Listener("refresh failed".into()));
            }
            let now = self.source.load(Ordering::SeqCst);
            Ok((now != *current).then_some(now))
        }
    }

    #[derive(Default)]
    struct Recording {
        events: Mutex<Vec<String>>,
        fail_before: std::sync::atomic::AtomicBool,
    }

    impl RefreshListener for Recording {
        fn before_refresh(&self) -> std::result::Result<(), String> {
            self.events.lock().unwrap().push("before".into());
            if self.fail_before.load(Ordering::SeqCst) {
                return Err("no".into());
            }
            Ok(())
        }
        fn after_refresh(&self, did_refresh: bool) -> std::result::Result<(), String> {
            self.events
                .lock()
                .unwrap()
                .push(format!("after({did_refresh})"));
            Ok(())
        }
    }

    #[test]
    fn refresh_swaps_in_a_new_reference_and_old_ones_stay_usable() {
        let counter = Counter {
            source: AtomicUsize::new(0),
            fail: Default::default(),
        };
        let m = ReferenceManager::new(0usize, &counter);
        let listener = Arc::new(Recording::default());
        let as_dyn: Arc<dyn RefreshListener> = listener.clone();
        m.add_listener(as_dyn.clone());

        let old = m.acquire().unwrap();
        assert!(m.maybe_refresh().unwrap());
        assert!(Arc::ptr_eq(&old, &m.acquire().unwrap()), "nothing changed");
        counter.source.store(7, Ordering::SeqCst);
        m.maybe_refresh_blocking().unwrap();
        assert_eq!(*m.acquire().unwrap(), 7);
        assert_eq!(*old, 0, "a reference swapped out is still held");
        m.release(old);
        assert_eq!(
            *listener.events.lock().unwrap(),
            ["before", "after(false)", "before", "after(true)"]
        );

        // A failed refresh keeps the reference and still tells the listener.
        counter.source.store(8, Ordering::SeqCst);
        counter.fail.store(true, Ordering::SeqCst);
        assert!(m.maybe_refresh().is_err());
        assert_eq!(*m.acquire().unwrap(), 7);
        counter.fail.store(false, Ordering::SeqCst);
        listener.fail_before.store(true, Ordering::SeqCst);
        assert!(matches!(m.maybe_refresh(), Err(Error::Listener(_))));
        assert_eq!(*m.acquire().unwrap(), 7);
        assert_eq!(listener.events.lock().unwrap().len(), 8);

        m.remove_listener(&as_dyn);
        listener.fail_before.store(false, Ordering::SeqCst);
        m.maybe_refresh().unwrap();
        assert_eq!(listener.events.lock().unwrap().len(), 8, "removed");
        assert_eq!(*m.acquire().unwrap(), 8);
    }

    #[test]
    fn a_refresh_in_progress_makes_maybe_refresh_return_false() {
        let counter = Counter {
            source: AtomicUsize::new(0),
            fail: Default::default(),
        };
        let m = ReferenceManager::new(0usize, &counter);
        let held = m.refresh_lock.lock().unwrap();
        assert!(!m.maybe_refresh().unwrap());
        drop(held);
        assert!(m.maybe_refresh().unwrap());
    }

    #[test]
    fn a_closed_manager_refuses_everything_but_close() {
        let counter = Counter {
            source: AtomicUsize::new(0),
            fail: Default::default(),
        };
        let m = ReferenceManager::new(0usize, &counter);
        let held = m.acquire().unwrap();
        m.close();
        m.close();
        assert!(matches!(m.acquire(), Err(Error::AlreadyClosed)));
        assert!(matches!(m.maybe_refresh(), Err(Error::AlreadyClosed)));
        assert!(matches!(
            m.maybe_refresh_blocking(),
            Err(Error::AlreadyClosed)
        ));
        assert!(matches!(m.swap_reference(3), Err(Error::AlreadyClosed)));
        assert_eq!(*held, 0);
    }
}
