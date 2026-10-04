//! Decoded sparse doc-values document sets ([`SparseDocs`]), kept across
//! iterators under one process-wide byte budget.
//!
//! Java has no such cache: `IndexedDISI` walks a sparse field's documents
//! lazily, per iterator. This port decodes them once into a rank index
//! (stage 3, M10 T10.4: the grouping collectors open a field's values in every
//! segment for every selector, and decoding was a tenth of a grouping
//! search), and keeps the decoded set for the next iterator over the same
//! field of the same segment core.
//!
//! A set costs about `4 * docsWithField + maxDoc / 8 * 1.5` bytes, and a
//! reader can touch every sparse field of every segment, so what is kept is
//! bounded: at most [`SPARSE_DOCS_CACHE_BYTES`] across every segment core of
//! the process, the least recently used set going first when a new one would
//! pass it. A set larger than a quarter of the budget is not kept at all --
//! it is decoded per iterator, as before the cache, rather than pushing out
//! everything else. A segment core's sets go when its last reader drops.
//!
//! Entries are keyed by the segment core (an id per opened segment, shared
//! by every reader reopened from it with only new deletions, since those
//! read the same doc-values bytes) and by the set's address in the data
//! the core holds. A reader opened afresh -- a new doc-values generation
//! included -- gets a new id, so it never sees another core's sets.

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use crate::reader::SparseDocs;

/// The most bytes of decoded sets the process keeps: a few hundred sparse
/// fields of 1M-document segments.
pub(crate) const SPARSE_DOCS_CACHE_BYTES: usize = 64 << 20;

/// Where in a core's doc-values data a set lives: the data's address and
/// the set's offset in it.
type SetKey = (usize, i64);

struct Entry {
    docs: Arc<SparseDocs>,
    bytes: usize,
    /// The entry's position in [`Inner::lru`].
    tick: u64,
}

#[derive(Default)]
struct Inner {
    cores: HashMap<u64, HashMap<SetKey, Entry>>,
    /// Every entry by when it was last used, oldest first.
    lru: BTreeMap<u64, (u64, SetKey)>,
    tick: u64,
    bytes: usize,
}

/// A byte-budgeted store of decoded sets, shared by every segment core that
/// points at it (the process-wide [`GLOBAL`] outside tests).
pub(crate) struct SparseDocsPool {
    budget: usize,
    inner: Mutex<Option<Inner>>,
}

static GLOBAL: SparseDocsPool = SparseDocsPool::new(SPARSE_DOCS_CACHE_BYTES);

impl SparseDocsPool {
    pub(crate) const fn new(budget: usize) -> Self {
        Self {
            budget,
            inner: Mutex::new(None),
        }
    }

    fn with<R>(&self, f: impl FnOnce(&mut Inner) -> R) -> R {
        let mut guard = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
        f(guard.get_or_insert_with(Inner::default))
    }

    fn get(&self, core: u64, key: SetKey) -> Option<Arc<SparseDocs>> {
        self.with(|inner| {
            let entry = inner.cores.get_mut(&core)?.get_mut(&key)?;
            inner.lru.remove(&entry.tick);
            inner.tick = inner.tick.wrapping_add(1);
            entry.tick = inner.tick;
            inner.lru.insert(entry.tick, (core, key));
            Some(Arc::clone(&entry.docs))
        })
    }

    fn insert(&self, core: u64, key: SetKey, docs: &Arc<SparseDocs>) {
        let bytes = docs.heap_bytes();
        if bytes > self.budget / 4 {
            return;
        }
        self.with(|inner| {
            if inner.cores.get(&core).is_some_and(|c| c.contains_key(&key)) {
                return;
            }
            while inner.bytes.saturating_add(bytes) > self.budget {
                let Some((_, (c, k))) = inner.lru.pop_first() else {
                    break;
                };
                if let Some(sets) = inner.cores.get_mut(&c) {
                    if let Some(old) = sets.remove(&k) {
                        inner.bytes = inner.bytes.saturating_sub(old.bytes);
                    }
                    if sets.is_empty() {
                        inner.cores.remove(&c);
                    }
                }
            }
            inner.tick = inner.tick.wrapping_add(1);
            let tick = inner.tick;
            inner.lru.insert(tick, (core, key));
            inner.bytes = inner.bytes.saturating_add(bytes);
            inner.cores.entry(core).or_default().insert(
                key,
                Entry {
                    docs: Arc::clone(docs),
                    bytes,
                    tick,
                },
            );
        });
    }

    fn purge(&self, core: u64) {
        self.with(|inner| {
            if let Some(sets) = inner.cores.remove(&core) {
                for e in sets.values() {
                    inner.lru.remove(&e.tick);
                    inner.bytes = inner.bytes.saturating_sub(e.bytes);
                }
            }
        });
    }

    /// How many sets `core` has kept, and the pool's bytes in all (tests).
    #[cfg(test)]
    fn held(&self, core: u64) -> (usize, usize) {
        self.with(|inner| (inner.cores.get(&core).map_or(0, HashMap::len), inner.bytes))
    }
}

/// One segment core's handle on the pool: what `SegmentReader` holds, shared
/// (`Arc`) by every reader of the same core. Dropping the last one drops the
/// core's sets.
pub(crate) struct DocsWithFieldCache {
    core: u64,
    pool: &'static SparseDocsPool,
}

impl DocsWithFieldCache {
    fn in_pool(pool: &'static SparseDocsPool) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        Self {
            core: NEXT.fetch_add(1, Ordering::Relaxed),
            pool,
        }
    }

    /// The set at `key` of this core, if it is kept.
    pub(crate) fn get(&self, key: SetKey) -> Option<Arc<SparseDocs>> {
        self.pool.get(self.core, key)
    }

    /// Keeps `docs` as the set at `key`, within the pool's budget.
    pub(crate) fn insert(&self, key: SetKey, docs: &Arc<SparseDocs>) {
        self.pool.insert(self.core, key, docs);
    }

    /// How many sets this core has kept (tests).
    #[cfg(test)]
    pub(crate) fn held(&self) -> usize {
        self.pool.held(self.core).0
    }
}

impl Default for DocsWithFieldCache {
    fn default() -> Self {
        Self::in_pool(&GLOBAL)
    }
}

impl Drop for DocsWithFieldCache {
    fn drop(&mut self) {
        self.pool.purge(self.core);
    }
}

impl std::fmt::Debug for DocsWithFieldCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DocsWithFieldCache")
            .field("core", &self.core)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::arithmetic_side_effects)]
    use super::*;

    fn set(max_doc: i32, every: i32) -> Arc<SparseDocs> {
        Arc::new(SparseDocs::new((0..max_doc).step_by(every as usize).collect(), max_doc).unwrap())
    }

    fn pool(budget: usize) -> &'static SparseDocsPool {
        Box::leak(Box::new(SparseDocsPool::new(budget)))
    }

    #[test]
    fn the_least_recently_used_set_goes_first_and_the_budget_holds() {
        let one = set(64 * 1024, 2).heap_bytes();
        // Room for ten sets of this size, not eleven.
        let pool = pool(one * 10 + one / 2);
        let a = DocsWithFieldCache::in_pool(pool);
        let b = DocsWithFieldCache::in_pool(pool);
        for k in 0..10 {
            a.insert((1, k), &set(64 * 1024, 2));
        }
        assert_eq!(a.held(), 10);
        // Touching the oldest makes the second the one to go.
        assert!(a.get((1, 0)).is_some());
        b.insert((1, 0), &set(64 * 1024, 2));
        assert!(a.get((1, 0)).is_some());
        assert!(
            a.get((1, 1)).is_none(),
            "the least recently used is evicted"
        );
        assert!(a.get((1, 2)).is_some());
        assert_eq!((a.held(), b.held()), (9, 1));
        assert_eq!(pool.held(u64::MAX).1, one * 10, "within the budget");
        // Keys are per core: `b`'s set is not `a`'s.
        assert!(Arc::ptr_eq(
            &b.get((1, 0)).unwrap(),
            &b.get((1, 0)).unwrap()
        ));
        assert!(!Arc::ptr_eq(
            &a.get((1, 0)).unwrap(),
            &b.get((1, 0)).unwrap()
        ));
        // Inserting a kept key again keeps the first set.
        let first = a.get((1, 2)).unwrap();
        a.insert((1, 2), &set(64 * 1024, 2));
        assert!(Arc::ptr_eq(&first, &a.get((1, 2)).unwrap()));
        // A core's sets go with its last handle, and their bytes with them.
        drop(a);
        assert_eq!(pool.held(u64::MAX).1, one);
        drop(b);
        assert_eq!(pool.held(u64::MAX).1, 0);
    }

    #[test]
    fn a_set_over_a_quarter_of_the_budget_is_not_kept() {
        let big = set(64 * 1024, 1);
        let pool = pool(big.heap_bytes() * 4 - 1);
        let c = DocsWithFieldCache::in_pool(pool);
        c.insert((1, 0), &big);
        assert_eq!(c.held(), 0);
        assert!(c.get((1, 0)).is_none());
        let small = set(64 * 1024, 64);
        c.insert((1, 1), &small);
        assert_eq!(c.held(), 1);
        assert!(format!("{c:?}").starts_with("DocsWithFieldCache"));
    }
}
