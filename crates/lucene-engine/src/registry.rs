//! Process-wide handle registries: each native object lives in a global
//! [`SlotMap`] behind a `u64` handle the caller carries between calls. Each
//! registry is an `RwLock`, so lookups from many JVM threads share the lock
//! and only open and close take it exclusively; entries are `Arc`s, so a
//! search clones its handle out and runs with no lock held.

use std::sync::{OnceLock, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::handle::{RegistryTag, SlotMap};
use crate::reader_handle::JvmReaderHandle;

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
