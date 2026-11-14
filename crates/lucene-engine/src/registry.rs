//! Process-wide handle registries: each native object lives in a global
//! [`SlotMap`] behind a `u64` handle the caller carries between calls. Each
//! registry is an `RwLock`, so lookups from many JVM threads share the lock
//! and only open and close take it exclusively; entries are `Arc`s, so a
//! search clones its handle out and runs with no lock held.

use std::sync::{OnceLock, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::handle::{RegistryTag, SlotMap};
use crate::reader_handle::JvmReaderHandle;
use crate::status::{set_last_error, Status};
use crate::writer::EngineWriter;

/// Takes `lock`'s exclusive guard, recovering the value if an earlier
/// holder panicked. Sound because [`SlotMap`]'s `insert` and `remove` never
/// panic mid-mutation, so a poisoned registry is never half-written; handle
/// validation is unchanged, and one panicking call cannot wedge every later
/// call on the registry.
pub fn lock_recovering<T>(lock: &RwLock<T>) -> RwLockWriteGuard<'_, T> {
    lock.write().unwrap_or_else(PoisonError::into_inner)
}

/// Takes `lock`'s shared guard, recovering from poison as [`lock_recovering`] does.
pub fn read_recovering<T>(lock: &RwLock<T>) -> RwLockReadGuard<'_, T> {
    lock.read().unwrap_or_else(PoisonError::into_inner)
}

/// Handles are `Arc`s so a search clones one out and runs with no lock held.
pub fn jvm_readers() -> &'static RwLock<SlotMap<std::sync::Arc<JvmReaderHandle>>> {
    static REGISTRY: OnceLock<RwLock<SlotMap<std::sync::Arc<JvmReaderHandle>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(SlotMap::new(RegistryTag::JvmReader)))
}

/// Closes a JVM reader handle. Segments a later handle reused stay open
/// until that handle closes too.
pub fn close_jvm_reader(handle: u64) -> Result<(), Status> {
    let mut readers = lock_recovering(jvm_readers());
    readers.remove(handle).ok_or_else(|| {
        set_last_error("ffi_close_jvm_reader: unknown or already-closed handle");
        Status::InvalidHandle
    })?;
    // The nested queries' parent bit sets of segments no open reader
    // holds any more go with them.
    let live = readers
        .values()
        .flat_map(|h| h.reader.segment_readers())
        .map(|r| r.segment_id())
        .collect();
    drop(readers);
    crate::nodes::retain_parent_sets(&live);
    Ok(())
}

/// The reader handle `handle` names, noting its shard for the running call.
pub fn lookup_jvm_reader(
    handle: u64,
    missing: &str,
) -> Result<std::sync::Arc<JvmReaderHandle>, Status> {
    let reader = read_recovering(jvm_readers())
        .get(handle)
        .cloned()
        .ok_or_else(|| {
            set_last_error(missing);
            Status::InvalidHandle
        })?;
    crate::status::note_shard(&reader.path);
    Ok(reader)
}

/// The OpenSearch engine's writers: one per shard. The registry lock is held
/// only to clone the `Arc`.
pub fn engine_writers() -> &'static RwLock<SlotMap<std::sync::Arc<EngineWriter>>> {
    static REGISTRY: OnceLock<RwLock<SlotMap<std::sync::Arc<EngineWriter>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(SlotMap::new(RegistryTag::EngineWriter)))
}
