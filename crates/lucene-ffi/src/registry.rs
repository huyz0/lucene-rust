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

/// An `IndexWriter` together with the directory it borrows.
///
/// `IndexWriter` borrows a `&dyn Directory`, so the handle owns the directory
/// on the heap and erases the borrow to `'static` (`engine_writer.rs`). The
/// `Box` keeps the directory at a stable address, and fields drop in
/// declaration order, so `writer` is dropped before `dir`.
pub struct WriterHandle {
    pub writer: lucene_index::index_writer::IndexWriter<'static>,
    // Only keeps the allocation `writer` borrows alive.
    #[allow(dead_code)]
    pub dir: Box<dyn lucene_store::directory::Directory>,
}

// SAFETY: `WriterHandle` is `!Send`/`!Sync` only because of the erased
// `dyn Directory` borrow. `IndexWriter` has no interior mutability
// (`lucene-index` forbids unsafe code, and every mutation takes `&mut self`),
// and the borrowed directory is the `Send + Sync` one boxed in this struct.
unsafe impl Send for WriterHandle {}
// SAFETY: as for `Send` above; shared `&WriterHandle` reads are data-race-free.
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
}

/// Handles are `Arc`s so a search clones one out and runs with no lock held.
pub fn jvm_readers() -> &'static RwLock<SlotMap<std::sync::Arc<JvmReaderHandle>>> {
    static REGISTRY: OnceLock<RwLock<SlotMap<std::sync::Arc<JvmReaderHandle>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(SlotMap::new(RegistryTag::JvmReader)))
}

/// The OpenSearch engine's writers (`engine_writer.rs`): one per shard, each
/// behind its own mutex so that two shards never serialize on each other --
/// the registry lock is held only to clone the `Arc`.
#[allow(clippy::type_complexity)]
pub fn engine_writers(
) -> &'static RwLock<SlotMap<std::sync::Arc<std::sync::Mutex<crate::engine_writer::EngineWriter>>>>
{
    static REGISTRY: OnceLock<
        RwLock<SlotMap<std::sync::Arc<std::sync::Mutex<crate::engine_writer::EngineWriter>>>>,
    > = OnceLock::new();
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
