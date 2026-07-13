//! Process-wide handle registries. A JNI caller has no way to hand this
//! crate a Rust reference across calls (see the `ffi-safety` skill), so
//! every opened `Directory`/segment/result set lives in one of these global
//! [`SlotMap`]s, guarded by a `Mutex` (JNI callers may call from more than
//! one JVM thread) behind a `u64` handle the caller carries between calls.

use std::sync::{Mutex, MutexGuard, OnceLock, PoisonError};

use lucene_codecs::blocktree::BlockTreeFields;
use lucene_store::directory::FsDirectory;

use crate::handle::{RegistryTag, SlotMap};

/// Locks `mutex`, recovering the inner value if a previous holder of this
/// same lock panicked instead of propagating the poison as a second panic.
///
/// **Why this is sound for these three registries specifically**: every
/// mutation [`SlotMap`] performs (`insert`/`remove`) is a single, non-panicking
/// sequence of `Vec`/field writes with no possibility of observing a torn
/// write from within this crate (`insert`/`remove` never call into
/// arbitrary/foreign code that could panic mid-mutation -- see `handle.rs`).
/// A panic that poisons one of these mutexes therefore always happens while
/// the guard is held read-only (e.g. mid-query, borrowing a `&SegmentHandle`
/// while decoding adversarial bytes) or entirely outside any `SlotMap`
/// method body, never mid-`insert`/mid-`remove` -- so the slotmap itself is
/// never left in a half-written state, only "the operation using its
/// borrowed contents failed." Recovering it is safe: every subsequent access
/// still goes through the same generation-tag (and, since the handle-tag
/// fix, registry-tag) validation, so a wrong/stale/cross-registry handle is
/// still rejected as before -- recovery only prevents *this* mutex from
/// wedging every future call into a permanent [`crate::error::FfiStatus::Panic`]
/// (defeating `catch_unwind`'s purpose of isolating one bad call), it does
/// not weaken any handle-validation guarantee.
pub(crate) fn lock_recovering<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// One opened segment's decoded term dictionary plus the raw bytes of
/// whichever postings files were opened alongside it. `postings::DocInput`/
/// `PosInput`/`PayInput` all borrow from a byte slice (see
/// `lucene-codecs/src/postings.rs`), so this struct owns the bytes and a
/// fresh `DocInput::open(&self.doc, ...)` etc. is (cheaply -- header/footer
/// checks only) reconstructed per query call rather than stored as a
/// self-referential field.
pub struct SegmentHandle {
    pub fields: BlockTreeFields,
    pub doc_bytes: Option<Vec<u8>>,
    pub pos_bytes: Option<Vec<u8>>,
    // `.pay` (payload) support is deferred -- see `lib.rs`'s module doc --
    // so there is no `pay_bytes` field yet; `search_phrase_query` is always
    // called with `pay_in: None` in `query.rs`.
    pub segment_id: [u8; 16],
    pub segment_suffix: String,
    #[allow(dead_code)] // carried for parity with a real segment's contract; not yet read.
    pub max_doc: i32,
}

/// A completed query's collected, ascending, live doc IDs -- read back via
/// `ffi_results_len`/`ffi_results_copy`, then released via
/// `ffi_close_results`.
pub struct ResultsHandle {
    pub docs: Vec<i32>,
}

pub fn directories() -> &'static Mutex<SlotMap<FsDirectory>> {
    static REGISTRY: OnceLock<Mutex<SlotMap<FsDirectory>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(SlotMap::new(RegistryTag::Directory)))
}

pub fn segments() -> &'static Mutex<SlotMap<SegmentHandle>> {
    static REGISTRY: OnceLock<Mutex<SlotMap<SegmentHandle>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(SlotMap::new(RegistryTag::Segment)))
}

pub fn results() -> &'static Mutex<SlotMap<ResultsHandle>> {
    static REGISTRY: OnceLock<Mutex<SlotMap<ResultsHandle>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(SlotMap::new(RegistryTag::Results)))
}
