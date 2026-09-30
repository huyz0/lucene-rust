//! Port of `org.apache.lucene.index.ReaderPool`, for what this port's writer
//! reads segments for: resolving buffered deletes and doc-values updates.
//!
//! Java's writer keeps one `ReadersAndUpdates` per segment in a pool, so
//! applying a round of buffered deletes to a segment reuses the
//! `SegmentReader` the previous round opened instead of opening the
//! segment's term dictionary again. This pool keeps exactly that reusable
//! part -- a segment's opened term dictionary and its `.doc` input -- keyed by
//! the segment's identity and field-infos generation, so every delete round
//! after the first costs one `.liv` read per segment rather than a full
//! postings open.
//!
//! # What differs from Java
//!
//! - **What is pooled.** Java pools whole readers with their pending deletes
//!   and doc-values updates (`ReadersAndUpdates`); here pending deletes are
//!   resolved and written in one step (`FrozenBufferedUpdates` applied to a
//!   segment writes its `.liv` generation at once), so only the immutable
//!   postings are worth keeping. Live docs are read per round, since they are
//!   exactly what a round changes.
//! - **Not shared with NRT readers.** Near-real-time readers live in
//!   `lucene-search`, above this crate; they share unchanged segments with
//!   each other instead (see [`crate::nrt`]).
//! - **Dropping** a segment's entry happens when a merge retires it
//!   ([`ReaderPool::drop_segments`]) and, as a backstop, whenever deletes are
//!   applied to the index ([`ReaderPool::retain`]) -- Java's `ReaderPool.drop`
//!   and `dropAll`.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use lucene_codecs::blocktree::BlockTreeFields;
use lucene_store::codec_util::ID_LENGTH;
use lucene_store::directory::Input;

use crate::index_writer::Result;
use crate::segment_infos::SegmentCommitInfo;

/// One segment's postings, opened once for delete resolution.
pub(crate) struct PooledPostings {
    pub(crate) fields: BlockTreeFields,
    /// The `.doc` bytes, `None` when the segment has no postings.
    pub(crate) doc_input: Option<Input>,
    /// The postings format's per-field suffix the segment's files are framed
    /// with (its own format's, which an older segment's need not share).
    pub(crate) suffix: String,
    /// `doc_input` is the concatenation of a multi-format segment's groups'
    /// `.doc` files, each already validated by
    /// `per_field_postings::open_groups`.
    pub(crate) combined: bool,
}

/// Which segment state an entry was opened from.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PoolKey {
    name: String,
    id: [u8; ID_LENGTH],
    field_infos_gen: i64,
}

impl PoolKey {
    fn of(sci: &SegmentCommitInfo) -> Self {
        PoolKey {
            name: sci.segment_name.clone(),
            id: sci.segment_id,
            field_infos_gen: sci.field_infos_gen,
        }
    }
}

/// `ReaderPool`: see the module documentation. Shared by every thread that
/// resolves deletes for one writer.
pub struct ReaderPool {
    entries: Mutex<HashMap<PoolKey, Arc<PooledPostings>>>,
    /// `IndexWriterConfig.getReaderPooling()`; on by default.
    enabled: AtomicBool,
    /// How many times a segment's postings were opened, for tests and
    /// diagnostics.
    opens: AtomicU64,
}

impl std::fmt::Debug for ReaderPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReaderPool")
            .field("segments", &self.len())
            .field("enabled", &self.is_enabled())
            .finish()
    }
}

impl Default for ReaderPool {
    fn default() -> Self {
        ReaderPool {
            entries: Mutex::new(HashMap::new()),
            enabled: AtomicBool::new(true),
            opens: AtomicU64::new(0),
        }
    }
}

impl ReaderPool {
    fn entries(&self) -> MutexGuard<'_, HashMap<PoolKey, Arc<PooledPostings>>> {
        self.entries.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// `ReaderPool.get(info, create)`: the pooled postings of `sci`, opened by
    /// `open` on first use. With pooling off, `open` runs every time and
    /// nothing is kept.
    pub(crate) fn get_or_open(
        &self,
        sci: &SegmentCommitInfo,
        open: impl FnOnce() -> Result<PooledPostings>,
    ) -> Result<Arc<PooledPostings>> {
        if !self.is_enabled() {
            self.opens.fetch_add(1, Ordering::Relaxed);
            return open().map(Arc::new);
        }
        let key = PoolKey::of(sci);
        if let Some(hit) = self.entries().get(&key) {
            return Ok(Arc::clone(hit));
        }
        // Opened without the lock: two threads may both open one segment,
        // and one's is kept -- the same postings either way.
        self.opens.fetch_add(1, Ordering::Relaxed);
        let opened = Arc::new(open()?);
        let mut entries = self.entries();
        // An older field-infos generation of the same segment is superseded.
        entries.retain(|k, _| k.name != key.name || k == &key);
        Ok(Arc::clone(entries.entry(key).or_insert(opened)))
    }

    /// `ReaderPool.drop(info)` for every segment named: a merge retired them.
    pub(crate) fn drop_segments(&self, names: &[String]) {
        self.entries().retain(|k, _| !names.contains(&k.name));
    }

    /// Keeps only the segments still in the index.
    pub(crate) fn retain(&self, live: &HashSet<&str>) {
        self.entries().retain(|k, _| live.contains(k.name.as_str()));
    }

    /// `ReaderPool.dropAll()`.
    pub(crate) fn clear(&self) {
        self.entries().clear();
    }

    /// `IndexWriterConfig.setReaderPooling(on)`. Turning it off drops what
    /// is pooled.
    pub fn set_enabled(&self, on: bool) {
        self.enabled.store(on, Ordering::SeqCst);
        if !on {
            self.clear();
        }
    }

    /// `IndexWriterConfig.getReaderPooling()`.
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::SeqCst)
    }

    /// How many segments are pooled.
    pub fn len(&self) -> usize {
        self.entries().len()
    }

    /// The pooled segments' names, sorted.
    pub fn segment_names(&self) -> Vec<String> {
        let mut v: Vec<String> = self.entries().keys().map(|k| k.name.clone()).collect();
        v.sort();
        v
    }

    /// Whether nothing is pooled.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// How many times a segment's postings have been opened through this
    /// pool (hits excluded).
    pub fn opens(&self) -> u64 {
        self.opens.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests;
