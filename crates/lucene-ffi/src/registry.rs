//! Process-wide handle registries. Native objects never cross the boundary
//! as pointers: each lives in a global [`SlotMap`] behind a `u64` handle the
//! caller carries between calls. Each registry is an `RwLock`, so lookups from
//! many JVM threads share the lock and only open and close take it
//! exclusively; entries are `Arc`s, so a search clones its handle out and runs
//! with no lock held.

use std::sync::{OnceLock, RwLock};

use lucene_engine::handle::{RegistryTag, SlotMap};
pub(crate) use lucene_engine::registry::{jvm_readers, lock_recovering, read_recovering};

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

pub use lucene_engine::reader_handle::JvmReaderHandle;

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
