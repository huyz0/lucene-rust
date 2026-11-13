//! Process-wide handle registries. Native objects never cross the boundary
//! as pointers: each lives in a global [`SlotMap`] behind a `u64` handle the
//! caller carries between calls. Each registry is an `RwLock`, so lookups from
//! many JVM threads share the lock and only open and close take it
//! exclusively; entries are `Arc`s, so a search clones its handle out and runs
//! with no lock held.

use std::sync::{OnceLock, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

use lucene_util::fixed_bit_set::FixedBitSet;

use crate::handle::{RegistryTag, SlotMap};

/// Takes `lock`'s exclusive guard, recovering the value if an earlier
/// holder panicked. Sound because [`SlotMap`]'s `insert` and `remove` never
/// panic mid-mutation, so a poisoned registry is never half-written; handle
/// validation is unchanged, and one panicking call cannot wedge every later
/// call on the registry.
pub(crate) fn lock_recovering<T>(lock: &RwLock<T>) -> RwLockWriteGuard<'_, T> {
    lock.write().unwrap_or_else(PoisonError::into_inner)
}

/// Takes `lock`'s shared guard, recovering from poison as [`lock_recovering`] does.
pub(crate) fn read_recovering<T>(lock: &RwLock<T>) -> RwLockReadGuard<'_, T> {
    lock.read().unwrap_or_else(PoisonError::into_inner)
}

/// A `ConcurrentIndexWriter` together with the directory it borrows and the
/// merge scheduler whose threads merge for it.
///
/// The writer borrows a `&dyn Directory`, so the handle owns the directory
/// on the heap and erases the borrow to `'static` (`engine_writer.rs`). The
/// `Box` keeps the directory at a stable address. The scheduler's threads
/// hold the writer while they merge, so dropping the handle first stops
/// merging and waits for those threads; fields then drop in declaration
/// order, `writer` before `dir`.
pub struct WriterHandle {
    pub writer: std::sync::Arc<lucene_index::concurrent_writer::ConcurrentIndexWriter<'static>>,
    pub merges: lucene_index::merge_scheduler::ConcurrentMergeScheduler,
    // Only keeps the allocation `writer` borrows alive.
    #[allow(dead_code)]
    pub dir: Box<dyn lucene_store::directory::Directory>,
}

impl Drop for WriterHandle {
    fn drop(&mut self) {
        self.writer.stop_merges();
        // A merge's failure no longer matters to a writer being dropped.
        let _ = self.merges.sync();
    }
}

// SAFETY: `WriterHandle` is `!Send`/`!Sync` only because of the erased
// `dyn Directory` borrow. The writer's shared state is behind its own locks
// (`lucene-index` forbids unsafe code), and the borrowed directory is the
// `Send + Sync` one boxed in this struct.
unsafe impl Send for WriterHandle {}
// SAFETY: as for `Send` above.
unsafe impl Sync for WriterHandle {}

/// A `DirectoryReader` opened over a JVM reader's segments, plus that JVM
/// reader's live docs (`jvm_reader.rs`). `live_docs[i]` is segment `i`'s
/// mask, `None` meaning no deletions; it **replaces** whatever the segment
/// reader decoded from `.liv`, because an NRT reader's deletions live in the
/// JVM's memory, not on disk.
pub struct JvmReaderHandle {
    pub reader: lucene_search::directory_reader::DirectoryReader,
    pub live_docs: Vec<Option<FixedBitSet>>,
    /// Deleted documents per segment (`maxDoc - live_docs.cardinality()`),
    /// kept alongside the mask so total-hits lower bounds need not recount it
    /// per query.
    pub deleted: Vec<i64>,
    /// `live_docs` with the deletes issued since the reader's commit
    /// applied, as far as the last live search read them
    /// ([`crate::live::CommitLive`]).
    pub commit_live: std::sync::Mutex<Option<crate::live::CommitLive>>,
    /// How many of the reader's segments, from the first, the aggregation
    /// segment cache may keep parts of: the committed ones. A realtime view's
    /// buffers, appended after them, change with every view.
    pub cacheable: usize,
    /// Each segment's identity in the aggregation segment cache, found on
    /// first use: `None` past [`Self::cacheable`].
    pub agg_identities:
        std::sync::OnceLock<Vec<Option<lucene_aggs::bucket_aggs::segment_cache::SegmentIdentity>>>,
    /// The shard path the reader's files are under, noted for each call
    /// that reads them ([`crate::error::note_shard`]).
    pub path: std::sync::Arc<str>,
}

impl JvmReaderHandle {
    /// [`Self::agg_identities`], found now when no search has yet.
    pub fn agg_identities(
        &self,
    ) -> &[Option<lucene_aggs::bucket_aggs::segment_cache::SegmentIdentity>] {
        self.agg_identities.get_or_init(|| {
            self.reader
                .segment_readers()
                .iter()
                .zip(&self.live_docs)
                .enumerate()
                .map(|(i, (r, live))| {
                    (i < self.cacheable).then(|| {
                        lucene_aggs::bucket_aggs::segment_cache::SegmentIdentity::of(
                            r,
                            live.as_ref(),
                        )
                    })
                })
                .collect()
        })
    }
}

/// Handles are `Arc`s so a search clones one out and runs with no lock held.
pub fn jvm_readers() -> &'static RwLock<SlotMap<std::sync::Arc<JvmReaderHandle>>> {
    static REGISTRY: OnceLock<RwLock<SlotMap<std::sync::Arc<JvmReaderHandle>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(SlotMap::new(RegistryTag::JvmReader)))
}

/// The OpenSearch engine's writers (`engine_writer.rs`): one per shard. The
/// registry lock is held only to clone the `Arc`.
pub fn engine_writers(
) -> &'static RwLock<SlotMap<std::sync::Arc<crate::engine_writer::EngineWriter>>> {
    static REGISTRY: OnceLock<RwLock<SlotMap<std::sync::Arc<crate::engine_writer::EngineWriter>>>> =
        OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(SlotMap::new(RegistryTag::EngineWriter)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// [`lock_recovering`] must hand back the inner value after a writer
    /// panicked holding it, rather than propagating the poison as a second
    /// panic -- otherwise one bad call would wedge every future call on that
    /// registry, defeating `catch_unwind`'s isolation.
    #[test]
    fn lock_recovering_recovers_a_poisoned_write_guard() {
        let lock = RwLock::new(7i32);
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut guard = lock.write().unwrap();
            *guard = 9;
            panic!("poison it");
        }));
        assert!(lock.is_poisoned());
        assert_eq!(*lock_recovering(&lock), 9);
        assert_eq!(*read_recovering(&lock), 9);
    }

    /// Two threads can hold read guards on the
    /// same registry at once; with an exclusive lock the test would deadlock.
    #[test]
    fn two_readers_hold_the_same_registry_concurrently() {
        static LOCK: OnceLock<RwLock<i32>> = OnceLock::new();
        let lock = LOCK.get_or_init(|| RwLock::new(5));
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        let b2 = std::sync::Arc::clone(&barrier);
        let other = std::thread::spawn(move || {
            let guard = read_recovering(lock);
            // Only proceeds if the *other* thread also got its read guard.
            b2.wait();
            *guard
        });
        let guard = read_recovering(lock);
        barrier.wait();
        assert_eq!(*guard, 5);
        assert_eq!(other.join().unwrap(), 5);
    }
}
