//! Live search for the plugin: documents still in the engine writer's
//! buffers, searched and fetched without a refresh
//! ([`lucene_index::concurrent_writer`]'s live views).
//!
//! A searcher opens a view ([`ffi_engine_writer_live_view_open`]) beside its
//! reader: the buffers' documents and deletes as of then. Its queries run
//! over the reader's segments and the view's buffers together
//! ([`ffi_jvm_reader_search_live`]), a buffered document numbered after the
//! reader's `maxDoc`, and its fetches read the buffered documents' stored
//! fields ([`ffi_live_view_document_alloc`]). Closing the view
//! ([`ffi_live_view_close`]) lets its buffers go.
//!
//! # The view blob
//!
//! Little-endian: `published: i64`, `count: i32`, then per buffer `max_doc:
//! i32`, `deleted: i32` and that many deleted document numbers (`i32`,
//! ascending), then `chunks: i32` and each chunk's end (`i32`, ascending,
//! the last `max_doc`): the parts of the buffer [`ffi_live_view_reader`]
//! makes a segment each, one leaf each.
//!
//! # Chunks
//!
//! A buffer still taking documents grows between views. Its segments are
//! kept per writer and buffer ([`Chunks`]) as consecutive document ranges,
//! so a view writes only the documents added since the last one written,
//! as one more chunk: when that chunk would be at least half the size of
//! the one before, they are written as one instead, and so on back, so a
//! buffer holds a few chunks and each document is written a few times.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use lucene_codecs::stored_fields::FieldValue;
use lucene_index::buffered_updates::Term;
use lucene_index::concurrent_writer::LiveView;
use lucene_search::collector::VecCollector;
use lucene_search::live_segment::live_request;
use lucene_search::multi_segment::OpenSegment;
use lucene_search::query::{Clause, TermQuery};
use lucene_search::similarities::Similarity;
use lucene_search::{
    search_boolean_query_multi_segment_maxscore_counting, search_term_query, TotalHitsRelation,
};
use lucene_util::fixed_bit_set::FixedBitSet;

use crate::clauses::map_search_error;
use crate::engine_writer::{self, EngineWriter};
use crate::error::{guard, set_last_error, FfiStatus};
use crate::jvm_fetch::{BINARY, DOUBLE, FLOAT, INT, LONG, STRING};
use crate::jvm_reader::{boolean_of, decode_request, lookup};
use crate::raw::bytes_from_raw;
use crate::registry::JvmReaderHandle;
use lucene_index::concurrent_writer::{FieldTotals, LeanSegment, LiveRequest, ViewBuffer};
use lucene_index::segment_infos::SegmentCommitInfo;
use lucene_search::directory_reader::SegmentReader;

/// An open view and the writer it reads.
struct ViewHandle {
    writer: Arc<EngineWriter>,
    view: LiveView,
    /// Per buffer of the view, its chunks.
    plans: Vec<Vec<Planned>>,
    /// What [`ffi_engine_writer_live_view_open`] gave the caller.
    blob: Vec<u8>,
    /// The commit generation it was opened for, and when.
    commit: Option<i64>,
    opened: std::time::Instant,
    /// The reader handle [`ffi_live_view_reader`] made over a reader and
    /// this view's buffers, closed with the view.
    whole: Mutex<Whole>,
    /// The buffers as the last plain requests read them ([`prepare_plain`]),
    /// by [`request_key`], newest last: a view never changes, so neither
    /// does what a request reads of it.
    prepared: Mutex<Vec<(Vec<u8>, Arc<Prepared>)>>,
}

impl Drop for ViewHandle {
    /// Closes the view: its buffers may go once no other view reads them.
    fn drop(&mut self) {
        self.writer.writer().close_live_view(&self.view);
        let writer = self.writer.writer();
        prune_chunks(&self.writer, |id| {
            writer.live_published_segment(id).is_none()
        });
        let handles = {
            let mut whole = self
                .whole
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            whole.closed = true;
            std::mem::take(&mut whole.handles)
        };
        for (_, _, h) in handles {
            crate::jvm_reader::ffi_close_jvm_reader(h);
        }
    }
}

/// What [`ffi_live_view_reader`] made of a view.
#[derive(Default)]
struct Whole {
    /// The handles made, by the reader handle they were made over and what
    /// they hold: the buffers' every term (`None`), or a request's terms
    /// (its blobs, [`request_key`]).
    handles: Vec<(u64, Option<Vec<u8>>, u64)>,
    /// Why its buffers cannot be written: they will not come back.
    failed: Option<String>,
    closed: bool,
}

impl Whole {
    /// The handle made for `key`, or why none can be.
    fn find(&self, handle: u64, key: Option<&[u8]>) -> Result<Option<u64>, FfiStatus> {
        if self.closed {
            set_last_error("live view: unknown or already-closed handle");
            return Err(FfiStatus::InvalidHandle);
        }
        if let Some((_, _, id)) = self
            .handles
            .iter()
            .find(|(b, k, _)| *b == handle && k.as_deref() == key)
        {
            return Ok(Some(*id));
        }
        if let Some(e) = &self.failed {
            set_last_error(e.clone());
            return Err(FfiStatus::InvalidArgument);
        }
        Ok(None)
    }
}

/// Where [`build_view_reader`]'s time goes, for the realtime micro bench:
/// nothing outside tests.
pub(crate) mod phases {
    pub(crate) const CHUNKS: usize = 0;
    pub(crate) const SNAPSHOT: usize = 1;
    pub(crate) const OVERLAY: usize = 2;
    pub(crate) const APPEND: usize = 3;
    pub(crate) const LIVE_DOCS: usize = 4;
    #[cfg(test)]
    pub(crate) const NAMES: [&str; 5] = ["chunks", "snapshot", "overlay", "append", "live_docs"];

    #[cfg(test)]
    thread_local! {
        /// This thread's time in each phase, since the last [`take`].
        static SPENT: std::cell::Cell<[std::time::Duration; 5]> =
            const { std::cell::Cell::new([std::time::Duration::ZERO; 5]) };
    }

    /// The time each phase took on this thread since the last call.
    #[cfg(test)]
    pub(crate) fn take() -> [std::time::Duration; 5] {
        SPENT.with(|s| s.replace([std::time::Duration::ZERO; 5]))
    }

    const TIMED: [lucene_index::live_timing::Phase; 5] = [
        lucene_index::live_timing::Phase::ReaderChunks,
        lucene_index::live_timing::Phase::ReaderSnapshot,
        lucene_index::live_timing::Phase::ReaderOverlay,
        lucene_index::live_timing::Phase::ReaderAppend,
        lucene_index::live_timing::Phase::ReaderLiveDocs,
    ];

    pub(crate) struct Timer {
        at: Option<std::time::Instant>,
    }

    impl Timer {
        pub(crate) fn start() -> Self {
            let timed = cfg!(test) || lucene_index::live_timing::enabled();
            Timer {
                at: timed.then(std::time::Instant::now),
            }
        }

        /// Ends phase `phase`, the next starting now.
        pub(crate) fn lap(&mut self, phase: usize) {
            let Some(at) = self.at else {
                return;
            };
            let now = std::time::Instant::now();
            let took = now.saturating_duration_since(at);
            self.at = Some(now);
            if let Some(&p) = TIMED.get(phase) {
                lucene_index::live_timing::record(p, took);
            }
            #[cfg(test)]
            SPENT.with(|s| {
                let mut v = s.get();
                if let Some(d) = v.get_mut(phase) {
                    *d += took;
                }
                s.set(v);
            });
        }
    }
}

/// While searches read the buffers through chunks without postings, a
/// thread of the writer's writes the buffers' newest documents into such
/// chunks every [`WARM_EVERY`], so a search finds them written and writes
/// only the documents since: their cost leaves the searches' latency. It
/// stops [`WARM_IDLE`] after the last such search.
#[derive(Default)]
pub(crate) struct Warmer {
    /// When a search last read chunks without postings, in milliseconds
    /// since [`epoch_ms`]'s epoch; 0 when none came.
    last_lean: AtomicU64,
    running: std::sync::atomic::AtomicBool,
    /// The chunks being written now, by any thread.
    writing: Mutex<Vec<Writing>>,
    /// Notified when a chunk of `writing` is done (kept or not).
    written: std::sync::Condvar,
    /// How old, in milliseconds, a view searches may share instead of
    /// opening their own ([`ffi_engine_writer_set_live_staleness`]); 0, the
    /// default, gives each search a view of every write acknowledged before
    /// it.
    max_staleness_ms: AtomicU64,
    /// Under a staleness bound, the view searches share.
    shared: Mutex<Option<Arc<ViewHandle>>>,
    /// The commit the last search opened a view for: the warmer opens its
    /// views for it.
    asked: Mutex<Option<Option<i64>>>,
    /// See [`Self::refresh_shared`].
    fresh_from: Mutex<Option<std::time::Instant>>,
    /// When an indexing thread last wrote a chunk, as `last_lean`.
    last_inline: AtomicU64,
    /// Adds since the writer opened, to look for a chunk every
    /// [`INLINE_EVERY`] of them.
    adds: AtomicU64,
    /// Under a staleness bound, the last plain requests' blobs, newest
    /// last, each with whether it came more than once: the warmer makes
    /// the reads of those that did of each view it opens.
    recent_plain: Mutex<Vec<(Vec<u8>, bool)>>,
    /// The same for [`ffi_live_view_reader`]'s requests: the reader handle
    /// last asked with, and the request's query, tree and extra blobs.
    recent_readers: Mutex<Vec<RecentReader>>,
}

/// A request [`Warmer::recent_readers`] keeps.
#[derive(Clone)]
struct RecentReader {
    handle: u64,
    /// The commit of the view it was asked of: the handle's reader's.
    commit: Option<i64>,
    query: Vec<u8>,
    tree: Option<Vec<u8>>,
    extra: Option<Vec<u8>>,
    /// Whether it came more than once.
    repeated: bool,
}

/// A chunk of buffer `buffer`'s documents `start..end` being written.
#[derive(Clone)]
struct Writing {
    buffer: u64,
    start: i32,
    end: i32,
    postings: bool,
    /// Whether [`Warmer`]'s thread writes it.
    warm: bool,
    end_totals: Totals,
}

/// Takes its chunk off [`Warmer::writing`] when dropped.
struct WritingGuard<'a> {
    warmer: &'a Warmer,
    key: (u64, i32, i32, bool),
}

impl Drop for WritingGuard<'_> {
    fn drop(&mut self) {
        let mut writing = self
            .warmer
            .writing
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // Only this guard's entry: another thread may write the same chunk.
        if let Some(i) = writing
            .iter()
            .position(|x| (x.buffer, x.start, x.end, x.postings) == self.key)
        {
            writing.remove(i);
        }
        drop(writing);
        self.warmer.written.notify_all();
    }
}

impl Warmer {
    fn max_staleness(&self) -> Option<std::time::Duration> {
        let ms = self.max_staleness_ms.load(Ordering::Relaxed);
        (ms > 0).then(|| std::time::Duration::from_millis(ms))
    }

    /// The shared view of `commit`, while it is young enough.
    fn shared_view(&self, commit: Option<i64>) -> Option<Arc<ViewHandle>> {
        let max = self.max_staleness()?;
        let mut shared = self
            .shared
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if shared.as_ref().is_some_and(|h| h.opened.elapsed() > max) {
            let old = shared.take();
            drop(shared);
            drop(old);
            return None;
        }
        shared.as_ref().filter(|h| h.commit == commit).cloned()
    }

    /// Forgets a shared view of a commit before `generation`, which the
    /// searches read no more.
    fn unshare_before(&self, generation: i64) {
        let mut shared = self
            .shared
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let old = if shared
            .as_ref()
            .is_some_and(|h| h.commit.is_some_and(|c| c < generation))
        {
            shared.take()
        } else {
            None
        };
        drop(shared);
        drop(old);
        let mut asked = self
            .asked
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if asked.is_some_and(|c| c.is_some_and(|c| c < generation)) {
            *asked = None;
        }
    }

    /// Makes `h` the view later searches share, under a staleness bound.
    /// A view opened before [`Self::fresh_from`] is not shared.
    fn share_view(&self, h: &Arc<ViewHandle>) {
        if self.max_staleness().is_none() {
            return;
        }
        let mut shared = self
            .shared
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let fresh = *self
            .fresh_from
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // A view of an older commit, or opened before the shared one, is
        // not shared: searches would read writes they had seen go missing.
        let newer = shared.as_ref().is_none_or(|s| match (s.commit, h.commit) {
            (Some(a), Some(b)) if a != b => b > a,
            _ => h.opened >= s.opened,
        });
        let old = if fresh.is_some_and(|f| h.opened < f) || !newer {
            None
        } else {
            shared.replace(Arc::clone(h))
        };
        // The bound lifted meanwhile (the writer closing): nothing is kept,
        // as a shared view holds the writer.
        let lifted = if self.max_staleness().is_none() {
            shared.take()
        } else {
            None
        };
        drop(shared);
        // Closed, when no search reads it, outside the lock.
        drop(old);
        drop(lifted);
    }

    /// From now on searches read only views opened from now on: every write
    /// acknowledged before is in them.
    fn refresh_shared(&self) {
        *self
            .fresh_from
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(std::time::Instant::now());
        self.unshare();
    }

    /// Whether the warmer should open a new shared view: the shared one is
    /// missing, of another commit, half its bound old, or lacks documents
    /// added since.
    fn needs_view(&self, w: &EngineWriter, commit: Option<i64>) -> bool {
        let Some(max) = self.max_staleness() else {
            return false;
        };
        let shared = self
            .shared
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        let Some(h) = shared else {
            return true;
        };
        if h.commit != commit || h.opened.elapsed() >= max / 2 {
            return true;
        }
        let now = w.writer().live_slot_buffers();
        now.iter().any(|b| {
            !h.view
                .buffers
                .iter()
                .any(|v| v.id == b.id && v.max_doc == b.max_doc)
        })
    }

    fn asked(&self, commit: Option<i64>) {
        if self.max_staleness().is_some() {
            *self
                .asked
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(commit);
        }
    }

    /// Lifts the staleness bound: the shared view is forgotten and the
    /// warmer writes chunks again.
    pub(crate) fn stop_sharing(&self) {
        self.max_staleness_ms.store(0, Ordering::Relaxed);
        *self
            .asked
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
        self.unshare();
    }

    /// The buffers the shared view reads.
    fn shared_buffers(&self) -> Vec<u64> {
        self.shared
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
            .map_or_else(Vec::new, |h| h.view.buffers.iter().map(|b| b.id).collect())
    }

    /// Forgets the shared view: the writer is closing, or no search came.
    pub(crate) fn unshare(&self) {
        let old = self
            .shared
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        drop(old);
    }

    /// Notes a plain request's blob for [`Self::recent_plain`].
    fn note_plain(&self, blob: &[u8]) {
        let mut recent = self
            .recent_plain
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(i) = recent.iter().position(|(b, _)| b == blob) {
            let (b, _) = recent.remove(i);
            recent.push((b, true));
            return;
        }
        if recent.len() >= RECENT_PLAIN {
            recent.remove(0);
        }
        recent.push((blob.to_vec(), false));
    }

    /// Notes a view reader's request for [`Self::recent_readers`].
    fn note_reader(
        &self,
        handle: u64,
        commit: Option<i64>,
        query: Option<&[u8]>,
        tree: Option<&[u8]>,
        extra: Option<&[u8]>,
    ) {
        let Some(query) = query else {
            return;
        };
        let mut recent = self
            .recent_readers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let same = |r: &RecentReader| {
            r.query == query && r.tree.as_deref() == tree && r.extra.as_deref() == extra
        };
        if let Some(i) = recent.iter().position(same) {
            let mut r = recent.remove(i);
            r.handle = handle;
            r.commit = commit;
            r.repeated = true;
            recent.push(r);
            return;
        }
        if recent.len() >= RECENT_PLAIN {
            recent.remove(0);
        }
        recent.push(RecentReader {
            handle,
            commit,
            query: query.to_vec(),
            tree: tree.map(<[u8]>::to_vec),
            extra: extra.map(<[u8]>::to_vec),
            repeated: false,
        });
    }

    /// The recent view readers' requests that came more than once: a
    /// one-off is not made ahead for views no search of it will read.
    fn recent_readers(&self) -> Vec<RecentReader> {
        self.recent_readers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
            .filter(|r| r.repeated)
            .cloned()
            .collect()
    }

    /// As [`Self::recent_readers`], for plain requests.
    fn recent_plain(&self) -> Vec<Vec<u8>> {
        self.recent_plain
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
            .filter(|(_, repeated)| *repeated)
            .map(|(b, _)| b.clone())
            .collect()
    }

    fn running(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }

    fn writing(&self) -> Vec<Writing> {
        self.writing
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// Waits while `w`'s chunk is being written (with postings or without,
    /// as `w`), then marks it as being written, under one lock so two
    /// threads do not both write it.
    fn claim_writing(&self, w: Writing) -> WritingGuard<'_> {
        let key = (w.buffer, w.start, w.end, w.postings);
        let mut writing = self
            .writing
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        while writing.iter().any(|x| {
            x.buffer == w.buffer && x.start == w.start && x.end == w.end && x.postings == w.postings
        }) {
            writing = self
                .written
                .wait(writing)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        writing.push(w);
        WritingGuard { warmer: self, key }
    }

    #[cfg(test)]
    fn start_writing(&self, w: Writing) -> WritingGuard<'_> {
        let key = (w.buffer, w.start, w.end, w.postings);
        self.writing
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(w);
        WritingGuard { warmer: self, key }
    }
}

const WARM_EVERY: std::time::Duration = std::time::Duration::from_millis(5);
const WARM_IDLE: u64 = 1000;
/// Fewer new documents than this are left for the next search to write.
const WARM_MIN_DOCS: i32 = 256;

fn epoch_ms() -> u64 {
    static EPOCH: OnceLock<std::time::Instant> = OnceLock::new();
    u64::try_from(
        EPOCH
            .get_or_init(std::time::Instant::now)
            .elapsed()
            .as_millis(),
    )
    .unwrap_or(u64::MAX)
}

/// Notes a search of `w`'s buffers through chunks without postings,
/// starting `w`'s warming thread when it is not running.
fn searched_lean(w: &Arc<EngineWriter>) {
    let warmer = w.live_warmer();
    // 0 is never: see `warm_loop`.
    warmer.last_lean.store(epoch_ms().max(1), Ordering::Relaxed);
    if warmer.running.swap(true, Ordering::AcqRel) {
        return;
    }
    let weak = Arc::downgrade(w);
    let spawned = std::thread::Builder::new()
        .name("live-warm".into())
        .spawn(move || warm_loop(&weak));
    if spawned.is_err() {
        warmer.running.store(false, Ordering::Release);
    }
}

fn warm_loop(weak: &std::sync::Weak<EngineWriter>) {
    // Its work is the searches' made ahead: a search waiting for a core
    // goes first.
    crate::engine_writer::yield_to_searches();
    loop {
        std::thread::sleep(WARM_EVERY);
        let Some(w) = weak.upgrade() else {
            return;
        };
        let warmer = w.live_warmer();
        let idle = || {
            let last = warmer.last_lean.load(Ordering::Relaxed);
            last == 0 || epoch_ms().saturating_sub(last) > WARM_IDLE
        };
        if idle() {
            warmer.unshare();
            warmer.running.store(false, Ordering::Release);
            // A search between the check and the store found it running.
            if idle() || warmer.running.swap(true, Ordering::AcqRel) {
                return;
            }
        }
        // A buffer taken for a flush meanwhile, or a writer closed, is
        // tried again next time; idle, the thread ends.
        let asked = *warmer
            .asked
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match asked {
            // Under a staleness bound the warmer opens the view the searches
            // share, its chunks written.
            Some(commit) => {
                if warmer.needs_view(&w, commit) {
                    // A panic must not end the thread with the warmer marked
                    // running.
                    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        if let Ok(h) = open_view(&w, commit, true) {
                            // The last requests' reads of it are made here,
                            // before any search shares it: the searches
                            // then build nothing of the buffers.
                            for blob in warmer.recent_plain() {
                                let _ = prepare_plain_blob(&h, &blob);
                            }
                            // A reader handle closed meanwhile is refused.
                            for r in warmer
                                .recent_readers()
                                .into_iter()
                                .filter(|r| r.commit == h.commit)
                            {
                                let _ = view_reader_of(
                                    r.handle,
                                    &h,
                                    Some(&r.query),
                                    r.tree.as_deref(),
                                    r.extra.as_deref(),
                                );
                            }
                            warmer.share_view(&h);
                        }
                    }));
                }
            }
            None => drop(warm(&w)),
        }
    }
}

/// Indexing threads write a chunk once a buffer has this many documents
/// past its chunks, while lean searches come, so searches rarely wait on
/// one; the warmer writes only a tail no indexing thread cut lately.
/// `LUCENE_RUST_LIVE_INLINE_CHUNK` overrides it, 0 leaving chunks to the
/// warmer.
const INLINE_CHUNK_DOCS: i32 = 1024;

/// How often, in adds, an indexing thread looks for a chunk to write.
const INLINE_EVERY: u64 = 32;

/// How long, in milliseconds, after an indexing thread wrote a chunk the
/// warmer leaves the tails alone.
const INLINE_QUIET_MS: u64 = 20;

fn inline_chunk_docs() -> Option<i32> {
    static N: OnceLock<Option<i32>> = OnceLock::new();
    *N.get_or_init(|| {
        let n = std::env::var("LUCENE_RUST_LIVE_INLINE_CHUNK")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(INLINE_CHUNK_DOCS);
        (n > 0).then_some(n)
    })
}

/// After an add on an indexing thread: see [`INLINE_CHUNK_DOCS`].
pub(crate) fn after_add(w: &EngineWriter) {
    let Some(n) = inline_chunk_docs() else {
        return;
    };
    let warmer = w.live_warmer();
    if !warmer.running()
        || !warmer
            .adds
            .fetch_add(1, Ordering::Relaxed)
            .is_multiple_of(INLINE_EVERY)
    {
        return;
    }
    if let Ok(true) = warm_min(w, n) {
        warmer
            .last_inline
            .store(epoch_ms().max(1), Ordering::Relaxed);
    }
}

/// Writes the newest documents of `w`'s buffers into chunks without
/// postings, as the next search's plan would cut them.
fn warm(w: &EngineWriter) -> lucene_index::index_writer::Result<()> {
    let last = w.live_warmer().last_inline.load(Ordering::Relaxed);
    if inline_chunk_docs().is_some() && epoch_ms().saturating_sub(last) < INLINE_QUIET_MS {
        return Ok(());
    }
    warm_min(w, WARM_MIN_DOCS).map(drop)
}

/// [`warm`] for tails of at least `min` documents; whether it wrote one.
fn warm_min(w: &EngineWriter, min: i32) -> lucene_index::index_writer::Result<bool> {
    // A panic must not end the thread with the warmer marked running.
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut wrote = false;
        w.writer().live_slot_buffers().iter().try_for_each(|b| {
            let writing = w.live_warmer().writing();
            let plan = plan(&w.live_chunks(), b, i32::MAX, &writing);
            let tail = plan.last().map_or(0, |p| p.end.saturating_sub(p.start));
            if tail < min || writes_nothing(w, b, &plan, false) {
                return Ok(());
            }
            wrote = true;
            buffer_segments(w, b, &plan, false, true, None).map(drop)
        })?;
        Ok(wrote)
    }))
    .unwrap_or_else(|_| {
        Err(lucene_index::index_writer::Error::LiveSearch(
            "warming the buffers panicked".into(),
        ))
    })
}

/// How many plain requests' reads view `view` keeps.
#[cfg(test)]
pub(crate) fn prepared_reads(view: u64) -> usize {
    self::view(view).map_or(0, |v| {
        v.prepared
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    })
}

/// Makes the read of the plain request `blob` of view `view`, as the warmer
/// makes it for a shared view.
#[cfg(test)]
pub(crate) fn prepare_ahead(view: u64, blob: &[u8]) {
    if let Ok(v) = self::view(view) {
        let _ = prepare_plain_blob(&v, blob);
    }
}

/// Writes the chunks [`warm`] would of `w`'s buffers whose tails hold at
/// least `min` documents.
#[cfg(test)]
pub(crate) fn warm_for_tests(w: &EngineWriter, min: i32) {
    warm_min(w, min).expect("warming the buffers");
}

/// Waits for `w`'s warming thread to end, as it does once no lean search
/// came for [`WARM_IDLE`]; the next lean search starts it again.
#[cfg(test)]
pub(crate) fn idle_warmer(w: &EngineWriter) {
    w.live_warmer().last_lean.store(0, Ordering::Relaxed);
    while w.live_warmer().running() {
        std::thread::sleep(WARM_EVERY);
    }
}

/// The documents of `w`'s first buffer after its chunks, as a chunk
/// marked as being written until the guard is dropped.
#[cfg(test)]
pub(crate) fn hold_tail(w: &EngineWriter) -> Option<(i32, i32, impl Drop + '_)> {
    let view = w.writer().open_live_view(None).ok()?;
    let tail = view.buffers.first().filter(|b| b.in_slot).and_then(|b| {
        let plan = plan(&w.live_chunks(), b, 0, &[]);
        plan.last().filter(|p| p.chunk.is_none()).map(|p| Writing {
            buffer: b.id,
            start: p.start,
            end: p.end,
            postings: true,
            warm: false,
            end_totals: Arc::clone(&p.end_totals),
        })
    });
    w.writer().close_live_view(&view);
    let tail = tail?;
    Some((tail.start, tail.end, w.live_warmer().start_writing(tail)))
}

/// The most handles of requests' terms a view keeps; a request past them
/// reads the handle of every term.
const REQUEST_HANDLES: usize = 16;

/// A request's blobs as one key: each but the last after its length.
fn request_key(query: &[u8], tree: Option<&[u8]>, extra: Option<&[u8]>) -> Vec<u8> {
    let tree = tree.unwrap_or_default();
    let mut key = Vec::new();
    for part in [query, tree] {
        key.extend_from_slice(&(part.len() as u64).to_le_bytes());
        key.extend_from_slice(part);
    }
    key.extend_from_slice(extra.unwrap_or_default());
    key
}

/// A buffer's documents `start..end` as a segment.
pub(crate) struct Chunk {
    start: i32,
    end: i32,
    /// [`ViewBuffer::field_totals`] of a view of `end` documents.
    totals: Totals,
    /// Whether `reader` holds the documents' postings.
    postings: bool,
    reader: SegmentReader,
    /// Where `reader`'s files are, for a JVM reader of them ([`ffi_live_view_lucene`]).
    files: ChunkFiles,
    /// Without postings, `reader` with the terms of the last requests
    /// ([`request_key`]) overlaid, newest last: the documents and their
    /// statistics never change, so neither does a request's overlay.
    overlays: Mutex<Vec<(Vec<u8>, SegmentReader)>>,
}

/// Leads the overlay key of a plain search's request ([`ffi_jvm_reader_search_live`]),
/// apart from a view reader's of the same bytes.
const PLAIN_KEY: u8 = 0xff;

/// The most overlays a chunk keeps.
const CHUNK_OVERLAYS: usize = 8;

impl Chunk {
    fn overlay(&self, key: &[u8]) -> Option<SegmentReader> {
        self.overlays
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, r)| r.clone())
    }

    fn keep_overlay(&self, key: &[u8], reader: &SegmentReader) {
        let mut overlays = self
            .overlays
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if overlays.iter().any(|(k, _)| k == key) {
            return;
        }
        if overlays.len() >= CHUNK_OVERLAYS {
            overlays.remove(0);
        }
        overlays.push((key.to_vec(), reader.clone()));
    }
}

/// By field number, a buffer's [`FieldTotals`] as of some document.
type Totals = Arc<HashMap<i32, FieldTotals>>;

/// Per buffer id and whether they hold postings, the chunks written of a
/// buffer still in its slot: consecutive, the first from document 0. A
/// search reads chunks without postings, with the terms it asks for read
/// from the buffer ([`overlaid`]); one asking for what that cannot give
/// reads chunks with them.
pub(crate) type Chunks = HashMap<(u64, bool), Vec<Arc<Chunk>>>;

/// One chunk of a view's buffer: documents `start..end`, with the
/// [`ViewBuffer::field_totals`] of views of `start` and `end` documents.
#[derive(Clone)]
struct Planned {
    start: i32,
    end: i32,
    start_totals: Totals,
    end_totals: Totals,
    /// The chunk written for these documents when the plan was made: kept
    /// for this view even once a later chunk absorbs it.
    chunk: Option<Arc<Chunk>>,
}

/// A chunk written after `acc` new documents absorbs a previous one of at
/// most this many times `acc`'s size. At 1 the chunks pair up like a binary
/// counter's bits: a document is rewritten about log2(buffer / new) times,
/// and a buffer is about as many chunks.
const ABSORB: i32 = 1;

/// While [`Warmer`] runs, a search's plan absorbs chunks of at most this
/// many documents in all: larger ones are the warmer's to rewrite, off the
/// searches' latency.
const SEARCH_ABSORB_DOCS: i32 = 4096;

/// `b`'s chunks: those of `chunks` its documents cover (of the chain with
/// postings or the one without, whichever covers more), then the documents
/// past them, absorbing the last chunks while they are small beside them
/// and the documents absorbed stay within `absorb_docs`. A buffer not read
/// as chunks ([`ViewBuffer::chunks`]), or with no chunk written yet, is
/// one.
fn plan(chunks: &Chunks, b: &ViewBuffer, absorb_docs: i32, writing: &[Writing]) -> Vec<Planned> {
    let totals = Arc::new(b.field_totals());
    let chain_of = |postings: bool| -> Vec<&Arc<Chunk>> {
        match chunks.get(&(b.id, postings)) {
            Some(c) if b.chunks => c.iter().take_while(|c| c.end <= b.max_doc).collect(),
            _ => Vec::new(),
        }
    };
    let (lean, full) = (chain_of(false), chain_of(true));
    let reach = |c: &[&Arc<Chunk>]| c.last().map_or(0, |c| c.end);
    let chain = if reach(&full) > reach(&lean) {
        full
    } else {
        lean
    };
    let mut kept = chain.len();
    let covered = chain.last().map_or(0, |c| c.end);
    // A chunk being written from the chain's end is planned as it is, after
    // the whole chain: the view waits for it and writes only what follows.
    // One that absorbs chunks of the chain is not waited for: the chunks it
    // replaces still serve this view. Nor is a warmer's chunk of at most
    // [`SEARCH_ABSORB_DOCS`]: the view writes those documents sooner than
    // it would wait for them.
    let pending = writing.iter().find(|x| {
        b.in_slot
            && x.buffer == b.id
            && x.end > covered
            && x.end < b.max_doc
            && x.start == covered
            && !(x.warm && x.end.saturating_sub(x.start) <= SEARCH_ABSORB_DOCS)
    });
    if pending.is_none() && covered < b.max_doc {
        let mut acc = b.max_doc.saturating_sub(covered);
        while let Some(last) = kept.checked_sub(1).and_then(|i| chain.get(i)) {
            let size = last.end.saturating_sub(last.start);
            if size > acc.saturating_mul(ABSORB) || acc.saturating_add(size) > absorb_docs {
                break;
            }
            acc = acc.saturating_add(size);
            kept -= 1;
        }
    }
    let mut out = Vec::new();
    let mut from = Arc::new(HashMap::new());
    for c in chain.iter().take(kept) {
        out.push(Planned {
            start: c.start,
            end: c.end,
            start_totals: Arc::clone(&from),
            end_totals: Arc::clone(&c.totals),
            chunk: Some(Arc::clone(c)),
        });
        from = Arc::clone(&c.totals);
    }
    if let Some(x) = pending {
        out.push(Planned {
            start: x.start,
            end: x.end,
            start_totals: from,
            end_totals: Arc::clone(&x.end_totals),
            chunk: None,
        });
        from = Arc::clone(&x.end_totals);
    }
    let start = out.last().map_or(0, |p| p.end);
    if start < b.max_doc || out.is_empty() {
        out.push(Planned {
            start,
            end: b.max_doc,
            start_totals: from,
            end_totals: totals,
            chunk: None,
        });
    }
    out
}

/// Adds `chunk` to `chain` after the chunk it follows, dropping those it
/// overlaps. A chunk that follows none is not kept, nor one the chain
/// already reaches past (an older view's, written late): the chain's
/// chunks are the later views'.
fn keep_chunk(chain: &mut Vec<Arc<Chunk>>, chunk: Arc<Chunk>) {
    let follows = chunk.start == 0 || chain.iter().any(|c| c.end == chunk.start);
    if !follows || chain.last().is_some_and(|c| c.end >= chunk.end) {
        return;
    }
    let (start, end) = (chunk.start, chunk.end);
    let after: Vec<Arc<Chunk>> = chain.iter().filter(|c| c.start >= end).cloned().collect();
    chain.retain(|c| c.end <= start);
    chain.push(chunk);
    let mut at = end;
    for c in after {
        if c.start != at {
            break;
        }
        at = c.end;
        chain.push(c);
    }
}

/// Drops `w`'s chunks of the buffers no open view of it reads, unless
/// `keep` says the buffer may still be read as chunks.
fn prune_chunks(w: &Arc<EngineWriter>, keep: impl Fn(u64) -> bool) {
    let mut read: std::collections::HashSet<u64> = views()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .values()
        .filter(|h| Arc::ptr_eq(&h.writer, w))
        .flat_map(|h| h.view.buffers.iter().map(|b| b.id))
        .collect();
    // The shared view is read by searches to come.
    read.extend(w.live_warmer().shared_buffers());
    let candidates: Vec<(u64, bool)> = w
        .live_chunks()
        .keys()
        .copied()
        .filter(|(id, _)| !read.contains(id))
        .collect();
    let gone: Vec<(u64, bool)> = candidates
        .into_iter()
        .filter(|&(id, _)| !keep(id))
        .collect();
    if !gone.is_empty() {
        let mut chunks = w.live_chunks();
        for key in gone {
            chunks.remove(&key);
        }
    }
}

/// Per field, how many of documents `p.start..p.end` have it.
fn range_counts(p: &Planned) -> HashMap<i32, i32> {
    p.end_totals
        .iter()
        .map(|(&f, t)| {
            (
                f,
                t.doc_count
                    .saturating_sub(p.start_totals.get(&f).map_or(0, |s| s.doc_count)),
            )
        })
        .filter(|&(_, n)| n > 0)
        .collect()
}

/// Per segment, the documents live once the deletes since the commit are
/// applied; `None` when every document is.
type Masks = Vec<Option<Arc<FixedBitSet>>>;

/// A reader's [`Masks`] after the first `applied` of the delete terms its
/// views list ([`LiveView::commit_deletes`]): every view of one commit
/// lists the same terms first, so the next search applies only the new
/// ones. `last` is the last of them, to tell a list that does not extend
/// these.
pub(crate) struct CommitLive {
    applied: usize,
    last: Term,
    masks: Arc<Masks>,
}

fn views() -> &'static Mutex<HashMap<u64, Arc<ViewHandle>>> {
    static VIEWS: OnceLock<Mutex<HashMap<u64, Arc<ViewHandle>>>> = OnceLock::new();
    VIEWS.get_or_init(Default::default)
}

static NEXT_VIEW: AtomicU64 = AtomicU64::new(1);

fn view(id: u64) -> Result<Arc<ViewHandle>, FfiStatus> {
    views()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(&id)
        .cloned()
        .ok_or_else(|| {
            set_last_error("live view: unknown or already-closed handle");
            FfiStatus::InvalidHandle
        })
}

fn live_error(e: lucene_index::index_writer::Error) -> FfiStatus {
    set_last_error(format!("live search: {e}"));
    FfiStatus::InvalidArgument
}

/// Lets searches of `writer`'s buffers read a view up to `millis`
/// milliseconds old: the writer's warming thread opens one every few
/// milliseconds, its chunks written, and the searches meanwhile share it,
/// its readers and their caches, writing nothing themselves. A search then
/// misses the writes of at most that long. 0 (the default) gives each search
/// a view of every write acknowledged before it.
#[no_mangle]
pub extern "C" fn ffi_engine_writer_set_live_staleness(writer: u64, millis: i64) -> i32 {
    guard(|| {
        let w = engine_writer::lookup(writer)?;
        let ms = u64::try_from(millis).map_err(|_| {
            set_last_error("live staleness: negative");
            FfiStatus::InvalidArgument
        })?;
        if ms == 0 {
            w.live_warmer().stop_sharing();
        } else {
            w.live_warmer()
                .max_staleness_ms
                .store(ms, Ordering::Relaxed);
        }
        Ok(())
    })
}

/// Under a staleness bound, makes every later search read a view opened
/// from now on, so it holds every write acknowledged before this call:
/// what lets `refresh=wait_for` answer at once.
#[no_mangle]
pub extern "C" fn ffi_engine_writer_live_refresh_shared(writer: u64) -> i32 {
    guard(|| {
        engine_writer::lookup(writer)?
            .live_warmer()
            .refresh_shared();
        Ok(())
    })
}

/// Opens a view of every buffered document a reader of the commit at
/// `commit_generation` lacks (`-1`: the documents in no published segment),
/// its handle into `*out_view` and its blob (see the module doc) as an owned
/// buffer.
///
/// # Safety
/// `out_view`, `out_ptr` and `out_len` valid for one write each.
#[no_mangle]
pub unsafe extern "C" fn ffi_engine_writer_live_view_open(
    writer: u64,
    commit_generation: i64,
    out_view: *mut u64,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        if out_view.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        // SAFETY: caller contract.
        unsafe { crate::ffm_bridge::clear_out(out_ptr, out_len)? };
        let w = engine_writer::lookup(writer)?;
        let commit = (commit_generation >= 0).then_some(commit_generation);
        let warmer = w.live_warmer();
        if warmer.max_staleness().is_some() {
            // The warmer opens the views searches share, and forgets the
            // shared one once no search comes.
            searched_lean(&w);
        }
        let h = match warmer.shared_view(commit) {
            Some(h) => h,
            None => {
                warmer.asked(commit);
                open_handle(&w, commit, false)?
            }
        };
        let blob = h.blob.clone();
        let id = NEXT_VIEW.fetch_add(1, Ordering::Relaxed);
        views()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(id, h);
        // SAFETY: caller contract.
        unsafe {
            *out_view = id;
            crate::ffm_bridge::give(blob, out_ptr, out_len);
        }
        Ok(())
    })
}

/// A view of `w`'s buffers for a reader of `commit`, its plans and blob.
/// `warm`, its chunks are written first, so the views' readers find them
/// all written. Under a staleness bound the searches share it.
fn open_handle(
    w: &Arc<EngineWriter>,
    commit: Option<i64>,
    warm: bool,
) -> Result<Arc<ViewHandle>, FfiStatus> {
    let h = open_view(w, commit, warm)?;
    w.live_warmer().share_view(&h);
    Ok(h)
}

/// [`open_handle`]'s view, not yet shared.
fn open_view(
    w: &Arc<EngineWriter>,
    commit: Option<i64>,
    warm: bool,
) -> Result<Arc<ViewHandle>, FfiStatus> {
    // Before the view's cut: the view is at least this old.
    let opened = std::time::Instant::now();
    let view = w.writer().open_live_view(commit).map_err(live_error)?;
    let plan_all = |w: &EngineWriter| -> Vec<Vec<Planned>> {
        let chunks = w.live_chunks();
        let warmer = w.live_warmer();
        let absorb = if warmer.running() {
            SEARCH_ABSORB_DOCS
        } else {
            i32::MAX
        };
        let writing = warmer.writing();
        view.buffers
            .iter()
            .map(|b| plan(&chunks, b, absorb, &writing))
            .collect()
    };
    let mut plans = plan_all(w);
    if warm {
        for (b, plan) in view.buffers.iter().zip(&plans) {
            if b.in_slot && !writes_nothing(w, b, plan, false) {
                if let Err(e) = buffer_segments(w, b, plan, false, true, None) {
                    w.writer().close_live_view(&view);
                    return Err(live_error(e));
                }
            }
        }
        plans = plan_all(w);
    }
    // A buffer not read as chunks is read as one segment by later views.
    prune_chunks(w, |id| view.buffers.iter().any(|b| b.chunks && b.id == id));
    let blob = match view_blob(&view, &plans) {
        Ok(b) => b,
        Err(e) => {
            w.writer().close_live_view(&view);
            return Err(e);
        }
    };
    let h = Arc::new(ViewHandle {
        writer: Arc::clone(w),
        view,
        plans,
        blob,
        commit,
        opened,
        whole: Mutex::new(Whole::default()),
        prepared: Mutex::new(Vec::new()),
    });
    Ok(h)
}

fn view_blob(view: &LiveView, plans: &[Vec<Planned>]) -> Result<Vec<u8>, FfiStatus> {
    let mut blob = Vec::new();
    blob.extend_from_slice(
        &i64::try_from(view.published)
            .unwrap_or(i64::MAX)
            .to_le_bytes(),
    );
    blob.extend_from_slice(&len_i32(view.buffers.len())?.to_le_bytes());
    for (b, plan) in view.buffers.iter().zip(plans) {
        blob.extend_from_slice(&b.max_doc.to_le_bytes());
        blob.extend_from_slice(&len_i32(b.deleted.len())?.to_le_bytes());
        for d in &b.deleted {
            blob.extend_from_slice(&d.to_le_bytes());
        }
        blob.extend_from_slice(&len_i32(plan.len())?.to_le_bytes());
        for p in plan {
            blob.extend_from_slice(&p.end.to_le_bytes());
        }
    }
    Ok(blob)
}

fn len_i32(n: usize) -> Result<i32, FfiStatus> {
    i32::try_from(n).map_err(|_| {
        set_last_error("live view: too many entries for the JVM");
        FfiStatus::InvalidArgument
    })
}

/// Closes a view: its buffers may go once no other view reads them.
#[no_mangle]
pub extern "C" fn ffi_live_view_close(view: u64) -> i32 {
    guard(|| {
        let h = views()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&view)
            .ok_or_else(|| {
                set_last_error("live view: unknown or already-closed handle");
                FfiStatus::InvalidHandle
            })?;
        // The view closes when its last handle goes: see `ViewHandle`'s drop.
        drop(h);
        Ok(())
    })
}

/// A reader handle over the reader at `handle` and `view`'s buffers as
/// segments after its own, into `*out_handle`: every search the JVM reader
/// takes ([`crate::jvm_reader`], aggregations, sorts) then reads the
/// buffered documents too, numbered as [`ffi_jvm_reader_search_live`]
/// numbers them. The buffers are written into memory as their flush would
/// write them, once per view; the deletes since the reader's commit, and
/// the buffers' own, are its deleted documents. The handle belongs to the
/// view: closing the view closes it, and the caller must not.
///
/// With a request (`query` as [`decode_request`] reads it, `tree` an
/// aggregation tree or empty, and `extra` more queries the search runs, a
/// rescore's, each an `i32` length then the blob, or empty), the buffers
/// are written without their postings and the handle holds only the terms
/// the request reads: it serves searches of that request alone. Without
/// one (`query_len` 0, when `extra` is not read), or for a request reading
/// what that cannot give, the handle holds every term.
///
/// # Safety
/// `out_handle` valid for one write; `query`/`tree`/`extra` readable for
/// their lengths when non-zero.
#[no_mangle]
pub unsafe extern "C" fn ffi_live_view_reader(
    handle: u64,
    view_id: u64,
    query: *const u8,
    query_len: usize,
    tree: *const u8,
    tree_len: usize,
    extra: *const u8,
    extra_len: usize,
    out_handle: *mut u64,
) -> i32 {
    guard(|| {
        let _timer = lucene_index::live_timing::start(lucene_index::live_timing::Phase::ViewReader);
        if out_handle.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        let v = view(view_id)?;
        // SAFETY: caller contract.
        let extra = (extra_len > 0 && query_len > 0)
            .then(|| unsafe { bytes_from_raw(extra, extra_len) })
            .transpose()?;
        // SAFETY: caller contract.
        let query = (query_len > 0)
            .then(|| unsafe { bytes_from_raw(query, query_len) })
            .transpose()?;
        // SAFETY: caller contract.
        let tree = (tree_len > 0)
            .then(|| unsafe { bytes_from_raw(tree, tree_len) })
            .transpose()?;
        if query.is_some() && v.writer.live_warmer().max_staleness().is_some() {
            v.writer
                .live_warmer()
                .note_reader(handle, v.commit, query, tree, extra);
        }
        let id = view_reader_of(handle, &v, query, tree, extra)?;
        // SAFETY: caller contract.
        unsafe { *out_handle = id };
        Ok(())
    })
}

/// [`ffi_live_view_reader`]'s handle over the reader at `handle` and `v`'s
/// buffers for the request of `query`, `tree` and `extra`, made once per
/// view and kept.
fn view_reader_of(
    handle: u64,
    v: &ViewHandle,
    query: Option<&[u8]>,
    tree: Option<&[u8]>,
    extra: Option<&[u8]>,
) -> Result<u64, FfiStatus> {
    let request = match query {
        Some(q) => overlay_request(q, tree, extra)?,
        None => None,
    };
    let key = request
        .as_ref()
        .zip(query)
        .map(|(_, q)| request_key(q, tree, extra));
    view_reader(handle, v, request, key)
}

/// [`crate::jvm_knn::ffi_jvm_reader_knn`] over the reader at `handle` and
/// `view_id`'s buffers, numbered as [`ffi_jvm_reader_search_live`] numbers
/// them: the request, outputs and errors are that call's. The buffers are
/// read through a handle of [`ffi_live_view_reader`]'s that holds only the
/// filter's terms (every term for a nested field, whose parent filter it
/// does not know), cached with the view; their vectors are searched
/// exactly, as a realtime chunk keeps no graph.
///
/// # Safety
/// As [`crate::jvm_knn::ffi_jvm_reader_knn`].
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ffi_live_view_knn(
    handle: u64,
    view_id: u64,
    request: *const u8,
    request_len: usize,
    out_docs: *mut i32,
    out_scores: *mut f32,
    buf_len: usize,
    out_count: *mut usize,
) -> i32 {
    guard(|| {
        if out_docs.is_null() || out_scores.is_null() || out_count.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        // SAFETY: caller contract.
        let blob = unsafe { bytes_from_raw(request, request_len)? };
        let knn = crate::jvm_knn::decode(blob)?;
        if buf_len < knn.k {
            return Err(FfiStatus::BufferTooSmall);
        }
        let v = view(view_id)?;
        let (request, key) = if knn.nested.is_some() {
            (None, None)
        } else {
            let all = Clause::MatchAllDocs(lucene_search::query::MatchAllDocsQuery::new(0));
            let clause = knn.filter.as_ref().unwrap_or(&all);
            let refs = [clause];
            match lucene_search::live_segment::whole_request(&refs) {
                Some(r) => {
                    let points = lucene_search::live_segment::reads_points(&refs);
                    // Keyed by the filter's own encoding, which names a
                    // filter exactly. The view's request budget is shared
                    // with the query phase: a view searched with many
                    // distinct filters leaves later requests the reader of
                    // every term.
                    let mut key = b"knn-filter:".to_vec();
                    key.extend_from_slice(&blob[knn.filter_bytes.clone()]);
                    (Some((r, points)), Some(key))
                }
                None => (None, None),
            }
        };
        let id = view_reader(handle, &v, request, key)?;
        let h = lookup(id, "ffi_live_view_knn: the view's reader closed")?;
        let hits = crate::jvm_knn::search(&h, &knn)?;
        // SAFETY: caller contract.
        unsafe { crate::jvm_knn::write_hits(&hits, out_docs, out_scores, buf_len, out_count) }
    })
}

/// The reader handle of `v` over the reader at `handle` that serves
/// `request` (the one keyed `key`, every term without one), made when the
/// view has none yet: see [`ffi_live_view_reader`].
fn view_reader(
    handle: u64,
    v: &ViewHandle,
    mut request: Option<(LiveRequest, bool)>,
    mut key: Option<Vec<u8>>,
) -> Result<u64, FfiStatus> {
    {
        let whole = v
            .whole
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut found = whole.find(handle, key.as_deref())?;
        if found.is_none()
            && key.is_some()
            && whole
                .handles
                .iter()
                .filter(|(b, k, _)| *b == handle && k.is_some())
                .count()
                >= REQUEST_HANDLES
        {
            (request, key) = (None, None);
            found = whole.find(handle, None)?;
        }
        if let Some(id) = found {
            return Ok(id);
        }
    }
    let id = build_view_reader(handle, v, request.as_ref(), key.as_deref())?;
    let mut whole = v
        .whole
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    // The view may have closed, or another search made the same handle,
    // meanwhile.
    let found = if whole.closed {
        set_last_error("live view: unknown or already-closed handle");
        Err(FfiStatus::InvalidHandle)
    } else {
        Ok(whole
            .handles
            .iter()
            .find(|(b, k, _)| *b == handle && *k == key)
            .map(|&(_, _, id)| id))
    };
    if !matches!(found, Ok(None)) {
        drop(whole);
        crate::jvm_reader::ffi_close_jvm_reader(id);
        return Ok(found?.unwrap_or_default());
    }
    whole.handles.push((handle, key, id));
    Ok(id)
}

/// A new reader handle over the reader at `handle` and `v`'s buffers as
/// [`ffi_live_view_reader`] describes it: every term without `request`,
/// else only `request`'s.
fn build_view_reader(
    handle: u64,
    v: &ViewHandle,
    request: Option<&(LiveRequest, bool)>,
    key: Option<&[u8]>,
) -> Result<u64, FfiStatus> {
    let lean_tail = request.map(|&(_, points)| points);
    let request = request.map(|(r, _)| r);
    let h = lookup(handle, "ffi_live_view_reader: unknown or closed reader")?;
    let mut extra = Vec::new();
    let mut masks = Vec::new();
    // Each buffer's chunks are written under that buffer's own slot
    // lock: with postings, the buffers with chunks to write are written
    // side by side, one on this thread and the others on threads of their
    // own. Without them each buffer has a small chunk at most, and they are
    // written on this thread: under load, a thread per buffer makes the
    // view wait for the slowest of them.
    let w: &EngineWriter = &v.writer;
    let postings = request.is_none();
    if !postings {
        searched_lean(&v.writer);
    }
    let mut phases = phases::Timer::start();
    let built = std::thread::scope(|s| {
        let mut inline_used = false;
        let jobs: Vec<_> = v
            .view
            .buffers
            .iter()
            .zip(&v.plans)
            .map(|(b, plan)| {
                let writes = !writes_nothing(w, b, plan, postings);
                if !writes || !inline_used || !postings {
                    inline_used |= writes;
                    return Err((b, plan));
                }
                std::thread::Builder::new()
                    .name("live-view".into())
                    .spawn_scoped(s, move || {
                        buffer_segments(w, b, plan, postings, false, lean_tail)
                    })
                    .map_err(|_| (b, plan))
            })
            .collect();
        jobs.into_iter()
            .map(|job| match job {
                Ok(handle) => handle
                    .join()
                    .unwrap_or_else(|panic| std::panic::resume_unwind(panic)),
                Err((b, plan)) => buffer_segments(w, b, plan, postings, false, lean_tail),
            })
            .collect::<Vec<_>>()
    });
    phases.lap(phases::CHUNKS);
    // The overlays found now are the ones used: one evicted later must not
    // be rebuilt from a read that starts past its chunk.
    let cached: Vec<Vec<Option<SegmentReader>>> = v
        .plans
        .iter()
        .map(|plan| plan.iter().map(|p| cached_overlay(p, key)).collect())
        .collect();
    // The buffers are read only for the chunks without postings that have
    // no overlay of this request yet, each from the first of them on.
    let from: Vec<Option<i32>> = built
        .iter()
        .zip(&v.plans)
        .zip(&cached)
        .map(|((parts, plan), cached)| {
            parts.as_ref().ok().and_then(|parts| {
                parts
                    .iter()
                    .zip(plan)
                    .zip(cached)
                    .find(|((&(_, lean), _), c)| lean && c.is_none())
                    .map(|((_, p), _)| p.start)
            })
        })
        .collect();
    let snapshot = match request.filter(|_| from.iter().any(Option::is_some)) {
        Some(r) => {
            let from: Vec<i32> = from
                .iter()
                .zip(&v.view.buffers)
                .map(|(f, b)| f.unwrap_or(b.max_doc))
                .collect();
            Some(
                v.writer
                    .writer()
                    .live_view_snapshot_from(&v.view, r, &from)
                    .map_err(live_error)?,
            )
        }
        None => None,
    };
    if snapshot
        .as_ref()
        .is_some_and(|s| s.buffers.len() != v.view.buffers.len())
    {
        set_last_error("live view: a snapshot of other than the view's buffers");
        return Err(FfiStatus::Decode);
    }
    phases.lap(phases::SNAPSHOT);
    for (i, (((b, plan), parts), cached)) in v
        .view
        .buffers
        .iter()
        .zip(&v.plans)
        .zip(built)
        .zip(cached)
        .enumerate()
    {
        let parts = match parts {
            Ok(parts) => parts,
            Err(e @ lucene_index::index_writer::Error::LiveSearch(_)) => {
                let msg = format!("live search: {e}");
                // Its buffers will not come back; handles made before
                // stay the view's.
                v.whole
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .failed = Some(msg.clone());
                set_last_error(msg);
                return Err(FfiStatus::InvalidArgument);
            }
            Err(e) => return Err(live_error(e)),
        };
        let buffer = snapshot.as_ref().and_then(|s| s.buffers.get(i));
        let parts = overlaid(parts, plan, cached, buffer, key)?;
        for (p, r) in plan.iter().zip(parts) {
            let len = p.end.saturating_sub(p.start);
            if r.max_doc < len {
                set_last_error(format!(
                    "live view: a buffer of {} documents became a segment of {}",
                    len, r.max_doc
                ));
                return Err(FfiStatus::Decode);
            }
            let lo = b.deleted.partition_point(|&d| d < p.start);
            let hi = b.deleted.partition_point(|&d| d < p.end);
            let deleted: Vec<i32> = b
                .deleted
                .get(lo..hi)
                .unwrap_or_default()
                .iter()
                .map(|&d| d.saturating_sub(p.start))
                .collect();
            masks.push(buffer_mask(len, &deleted, r.max_doc));
            extra.push(r);
        }
    }
    phases.lap(phases::OVERLAY);
    let reader = h.reader.with_readers_appended(extra).map_err(|e| {
        set_last_error(format!("opening the buffered documents: {e}"));
        FfiStatus::Decode
    })?;
    phases.lap(phases::APPEND);
    let mut live_docs: Vec<Option<FixedBitSet>> = if v.view.commit_deletes.is_empty() {
        h.live_docs.clone()
    } else {
        let opened = h.reader.open_segments().map_err(|e| {
            set_last_error(format!("opening segment postings: {e}"));
            FfiStatus::Decode
        })?;
        let segments = opened.as_open_segments();
        match commit_masks(&v.view.commit_deletes, &h, &segments)? {
            Some(m) => m.iter().map(|l| l.as_deref().cloned()).collect(),
            None => h.live_docs.clone(),
        }
    };
    live_docs.extend(masks);
    let readers = reader.segment_readers();
    let deleted = readers
        .iter()
        .zip(&live_docs)
        .map(|(r, l)| {
            l.as_ref().map_or(0, |l| {
                i64::from(r.max_doc).saturating_sub(i64::try_from(l.cardinality()).unwrap_or(0))
            })
        })
        .collect();
    phases.lap(phases::LIVE_DOCS);
    crate::registry::lock_recovering(crate::registry::jvm_readers()).insert_checked(Arc::new(
        JvmReaderHandle {
            reader,
            live_docs,
            deleted,
            commit_live: Mutex::new(None),
        },
    ))
}

/// What a search Lucene answers reads the view's buffers through, into
/// `*out_ptr`/`*out_len`: the deletes since the commit of the reader at
/// `handle` (the view's base), and each leaf of the view as a segment with
/// its postings, for the JVM to open.
///
/// Little-endian: `segments: i32`, then per segment of the reader its
/// live documents as of the view (`words: i32`, `-1` when none were
/// deleted since the commit, then that many `i64`); then `leaves: i32`, and
/// per leaf `kind: u8` (`0` a segment in memory, `1` the segment its buffer
/// was flushed as, in the index's directory), the segment's name (`i32`
/// length, UTF-8), its 16-byte id, its codec's name (as the name), the leaf's first document in it
/// (`i32`), and with kind `0` its files (`count: i32`, each a name as above,
/// `len: i64` and the bytes). A leaf of a flushed buffer may be a part of
/// that segment, which also holds documents added after the view.
///
/// A buffer taken for a flush since the view opened is waited for up to
/// `wait_ms` to publish; [`FfiStatus::InvalidArgument`] when it has not.
///
/// # Safety
/// `out_ptr`/`out_len` valid for one write each.
#[no_mangle]
pub unsafe extern "C" fn ffi_live_view_lucene(
    handle: u64,
    view_id: u64,
    wait_ms: i64,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        unsafe { crate::ffm_bridge::clear_out(out_ptr, out_len)? };
        let v = view(view_id)?;
        let h = lookup(handle, "ffi_live_view_lucene: unknown or closed reader")?;
        let w: &EngineWriter = &v.writer;
        let wait = std::time::Duration::from_millis(u64::try_from(wait_ms).unwrap_or(0));
        let deadline = std::time::Instant::now()
            .checked_add(wait)
            .unwrap_or_else(std::time::Instant::now);
        let mut leaves: Vec<(ChunkFiles, i32)> = Vec::new();
        for (b, plan) in v.view.buffers.iter().zip(&v.plans) {
            match buffer_segment_files(w, b, plan, true, false, None) {
                Ok(parts) => leaves.extend(parts.into_iter().map(|(_, _, files)| (files, 0))),
                Err(lucene_index::index_writer::Error::LiveSearch(_)) => {
                    // Taken for a flush since the view opened: its documents
                    // are read from the segment it is flushed as.
                    let sci = loop {
                        if let Some(sci) = w.writer().live_published_segment(b.id) {
                            break sci;
                        }
                        if std::time::Instant::now() >= deadline {
                            set_last_error(format!(
                                "live view: buffer {} is still being flushed",
                                b.id
                            ));
                            return Err(FfiStatus::InvalidArgument);
                        }
                        std::thread::sleep(std::time::Duration::from_millis(1));
                    };
                    leaves.extend(
                        plan.iter()
                            .map(|p| (ChunkFiles::Published(sci.clone()), p.start)),
                    );
                }
                Err(e) => return Err(live_error(e)),
            }
        }
        let masks = if v.view.commit_deletes.is_empty() {
            None
        } else {
            let opened = h.reader.open_segments().map_err(|e| {
                set_last_error(format!("opening segment postings: {e}"));
                FfiStatus::Decode
            })?;
            let segments = opened.as_open_segments();
            commit_masks(&v.view.commit_deletes, &h, &segments)?
        };
        let mut out = Vec::new();
        let segments = h.reader.segment_readers().len();
        put_len(&mut out, segments)?;
        for i in 0..segments {
            match masks
                .as_ref()
                .and_then(|m| m.get(i))
                .and_then(Option::as_ref)
            {
                Some(mask) => {
                    put_len(&mut out, mask.words().len())?;
                    mask.words()
                        .iter()
                        .for_each(|w| out.extend_from_slice(&w.to_le_bytes()));
                }
                None => out.extend_from_slice(&(-1i32).to_le_bytes()),
            }
        }
        put_len(&mut out, leaves.len())?;
        for (files, offset) in &leaves {
            let (kind, sci, dir) = match files {
                ChunkFiles::Memory(dir, sci) => (0u8, sci, Some(dir)),
                ChunkFiles::Published(sci) => (1u8, sci, None),
                ChunkFiles::None => {
                    set_last_error("live view: a buffered chunk without postings");
                    return Err(FfiStatus::Decode);
                }
            };
            out.push(kind);
            put_bytes(&mut out, sci.segment_name.as_bytes())?;
            out.extend_from_slice(&sci.segment_id);
            put_bytes(&mut out, sci.codec_name.as_bytes())?;
            out.extend_from_slice(&offset.to_le_bytes());
            if let Some(dir) = dir {
                let names = lucene_store::Directory::list_all(&**dir).map_err(|e| {
                    set_last_error(format!("live view: listing a chunk's files: {e}"));
                    FfiStatus::Decode
                })?;
                put_len(&mut out, names.len())?;
                for name in names {
                    let input = lucene_store::Directory::open(&**dir, &name).map_err(|e| {
                        set_last_error(format!("live view: reading {name}: {e}"));
                        FfiStatus::Decode
                    })?;
                    put_bytes(&mut out, name.as_bytes())?;
                    out.extend_from_slice(
                        &u64::try_from(input.len()).unwrap_or(u64::MAX).to_le_bytes(),
                    );
                    out.extend_from_slice(&input);
                }
            }
        }
        // SAFETY: caller contract.
        unsafe { crate::ffm_bridge::give(out, out_ptr, out_len) };
        Ok(())
    })
}

/// Appends `n` as the `i32` the blobs count with.
fn put_len(out: &mut Vec<u8>, n: usize) -> Result<(), FfiStatus> {
    let n = i32::try_from(n).map_err(|_| {
        set_last_error("live view: more than i32::MAX entries");
        FfiStatus::Decode
    })?;
    out.extend_from_slice(&n.to_le_bytes());
    Ok(())
}

/// Appends `bytes` after their `i32` length.
fn put_bytes(out: &mut Vec<u8>, bytes: &[u8]) -> Result<(), FfiStatus> {
    put_len(out, bytes.len())?;
    out.extend_from_slice(bytes);
    Ok(())
}

/// Whether [`buffer_segments`] writes nothing for buffer `b`: it is one
/// published segment, or every chunk `plan` names is in [`Chunks`].
fn writes_nothing(w: &EngineWriter, b: &ViewBuffer, plan: &[Planned], postings: bool) -> bool {
    if let [only] = plan {
        if only.start == 0 && w.writer().live_published_segment(b.id).is_some() {
            return true;
        }
    }
    let chunks = w.live_chunks();
    let have = chunks.get(&(b.id, postings));
    plan.iter().all(|p| {
        p.chunk.as_ref().is_some_and(|c| c.postings || !postings)
            || have.is_some_and(|c| c.iter().any(|c| c.start == p.start && c.end == p.end))
    })
}

/// Where a chunk's segment files are, for a JVM reader of them.
#[derive(Clone)]
enum ChunkFiles {
    /// Written into memory for the views.
    Memory(Arc<lucene_store::ByteBuffersDirectory>, SegmentCommitInfo),
    /// The segment the buffer was flushed as, in the writer's directory.
    Published(SegmentCommitInfo),
    /// None of its own: a chunk without postings, read from columns.
    None,
}

/// Buffer `b`'s chunks as segments, in `plan`'s order, each with whether
/// it lacks its postings: the segment it was published as when it is one
/// chunk (and published), else each chunk's from [`Chunks`], written (with
/// `postings` or without) when it is not there.
fn buffer_segments(
    w: &EngineWriter,
    b: &ViewBuffer,
    plan: &[Planned],
    postings: bool,
    warm: bool,
    lean_tail: Option<bool>,
) -> lucene_index::index_writer::Result<Vec<(SegmentReader, bool)>> {
    buffer_segment_files(w, b, plan, postings, warm, lean_tail)
        .map(|parts| parts.into_iter().map(|(r, lean, _)| (r, lean)).collect())
}

/// [`buffer_segments`], each segment with where its files are.
fn buffer_segment_files(
    w: &EngineWriter,
    b: &ViewBuffer,
    plan: &[Planned],
    postings: bool,
    warm: bool,
    lean_tail: Option<bool>,
) -> lucene_index::index_writer::Result<Vec<(SegmentReader, bool, ChunkFiles)>> {
    let writer = w.writer();
    if let [only] = plan {
        if only.start == 0 {
            if let Some(sci) = writer.live_published_segment(b.id) {
                let r = SegmentReader::open_one(writer.directory(), &sci).map_err(open_error)?;
                return Ok(vec![(r, false, ChunkFiles::Published(sci))]);
            }
        }
    }
    let mut out = Vec::new();
    for (i, p) in plan.iter().enumerate() {
        // A chunk with postings serves a search without them too.
        if let Some(c) = p.chunk.as_ref().filter(|c| c.postings || !postings) {
            // Kept in the chain without postings too, so the chunks written
            // after it there follow one ([`keep_chunk`]) and are kept.
            if c.postings && !postings && b.in_slot {
                keep_chunk(
                    w.live_chunks().entry((b.id, false)).or_default(),
                    Arc::clone(c),
                );
            }
            out.push((c.reader.clone(), !c.postings, c.files.clone()));
            continue;
        }
        // The last chunk of a search without postings, which a later view
        // seldom ends at too, is written for this search alone: its doc
        // values encoded a field at a time as the search reads them, its
        // points only when the request reads points, and without waiting
        // for another thread writing the same documents.
        let lean = lean_tail.filter(|_| !postings && i + 1 == plan.len());
        // Claimed before the cache is read: a chunk another thread is
        // writing is waited for and then found there.
        let _writing = (b.in_slot && lean.is_none()).then(|| {
            w.live_warmer().claim_writing(Writing {
                buffer: b.id,
                start: p.start,
                end: p.end,
                postings,
                warm,
                end_totals: Arc::clone(&p.end_totals),
            })
        });
        let cached = w.live_chunks().get(&(b.id, postings)).and_then(|c| {
            c.iter()
                .find(|c| c.start == p.start && c.end == p.end)
                .cloned()
        });
        if let Some(c) = cached {
            out.push((c.reader.clone(), !c.postings, c.files.clone()));
            continue;
        }
        let dir = Arc::new(lucene_store::ByteBuffersDirectory::new());
        // A name the JVM reads as Lucene names segments (`_[a-z0-9]+`): a
        // second `_` would start a suffix there.
        let name = format!("_live{}x{}", p.start, p.end);
        let starts: Vec<i32> = plan.iter().map(|p| p.end).collect();
        let range = p.start..p.end;
        let (sci, lazy) = match lean {
            Some(points) => match writer.live_lean_segment(
                b.id,
                range,
                range_counts(p),
                &starts,
                points,
                &*dir,
                &name,
            )? {
                LeanSegment::Parts(parts, lazy) => {
                    let reader =
                        SegmentReader::from_lean_parts(name, *parts, lazy).map_err(open_error)?;
                    out.push((reader, true, ChunkFiles::None));
                    continue;
                }
                LeanSegment::Written(sci, lazy) => (sci, lazy),
            },
            None => (
                writer.live_buffer_segment(
                    b.id,
                    range,
                    range_counts(p),
                    &starts,
                    postings,
                    &*dir,
                    &name,
                )?,
                None,
            ),
        };
        let reader = SegmentReader::open_one(&*dir, &sci).map_err(open_error)?;
        if let Some(lazy) = lazy {
            out.push((reader.with_lazy_doc_values(lazy), true, ChunkFiles::None));
            continue;
        }
        let files = if postings {
            ChunkFiles::Memory(dir, sci)
        } else {
            ChunkFiles::None
        };
        if b.in_slot && lean.is_none() {
            let chunk = Arc::new(Chunk {
                start: p.start,
                end: p.end,
                totals: Arc::clone(&p.end_totals),
                postings,
                reader: reader.clone(),
                files: files.clone(),
                overlays: Mutex::default(),
            });
            keep_chunk(w.live_chunks().entry((b.id, postings)).or_default(), chunk);
        }
        out.push((reader, !postings, files));
    }
    Ok(out)
}

/// The terms the request in `blob` reads from the buffers when it runs
/// over chunks without postings: its query's (`blob` as
/// [`decode_request`] reads it), then an aggregation tree's filter and
/// global queries (when `tree` holds one), and `extra`'s queries (as
/// [`ffi_live_view_reader`] lists them). `None` when it reads what the
/// buffers cannot give that way (a fuzzy or span query, among others).
fn overlay_request(
    query: &[u8],
    tree: Option<&[u8]>,
    extra: Option<&[u8]>,
) -> Result<Option<(LiveRequest, bool)>, FfiStatus> {
    let (query, _, _) = decode_request(query)?;
    let main = Clause::Boolean(Box::new(boolean_of(&query)));
    let decoded = tree.map(crate::jvm_aggs::decode_tree).transpose()?;
    let mut clauses = vec![main];
    let mut rest = extra.unwrap_or_default();
    while !rest.is_empty() {
        let len = rest
            .get(..4)
            .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .and_then(|n| usize::try_from(n).ok())
            .filter(|&n| n <= rest.len().saturating_sub(4))
            .ok_or_else(|| {
                set_last_error("live view: a request query of a bad length");
                FfiStatus::Decode
            })?;
        let blob = rest.get(4..4 + len).unwrap_or_default();
        // One that does not decode leaves the search to a handle of every
        // term, which its own run reports on.
        let Ok((q, _, _)) = decode_request(blob) else {
            return Ok(None);
        };
        clauses.push(Clause::Boolean(Box::new(boolean_of(&q))));
        rest = rest.get(4 + len..).unwrap_or_default();
    }
    if let Some(t) = &decoded {
        let mut filters = Vec::new();
        for n in t
            .nodes
            .iter()
            .chain(t.global.iter().flat_map(|(_, g)| g.iter()))
        {
            n.filter_queries(&mut filters);
        }
        clauses.extend(
            filters
                .into_iter()
                .map(|f| Clause::Boolean(Box::new(f.clone()))),
        );
        if let Some((g, _)) = &t.global {
            clauses.push(Clause::Boolean(Box::new(g.clone())));
        }
    }
    let refs: Vec<&Clause> = clauses.iter().collect();
    Ok(lucene_search::live_segment::whole_request(&refs)
        .map(|r| (r, lucene_search::live_segment::reads_points(&refs))))
}

/// The overlay of the request `key` names on `p`'s chunk, when that chunk
/// keeps one.
fn cached_overlay(p: &Planned, key: Option<&[u8]>) -> Option<SegmentReader> {
    key.zip(p.chunk.as_ref()).and_then(|(k, c)| c.overlay(k))
}

/// `parts` (buffer `b`'s chunks, as [`buffer_segments`] gave them) with
/// the terms of `buffer` (`b`'s in a snapshot for the request) put in each
/// chunk that lacks its postings. `cached` is, by chunk, the overlay of
/// the request [`cached_overlay`] found before `buffer` was read: those
/// are used as they are, and `buffer` covers every other chunk.
fn overlaid(
    parts: Vec<(SegmentReader, bool)>,
    plan: &[Planned],
    cached: Vec<Option<SegmentReader>>,
    buffer: Option<&lucene_index::concurrent_writer::LiveBuffer>,
    key: Option<&[u8]>,
) -> Result<Vec<SegmentReader>, FfiStatus> {
    let _timer = lucene_index::live_timing::start(lucene_index::live_timing::Phase::Overlay);
    if cached.len() != plan.len() {
        set_last_error("live view: overlays of other than the plan's chunks");
        return Err(FfiStatus::Decode);
    }
    parts
        .into_iter()
        .zip(plan)
        .zip(cached)
        .map(|(((r, lean), p), cached)| {
            if !lean {
                return Ok(r);
            }
            if let Some(r) = cached {
                return Ok(r);
            }
            let Some(buffer) = buffer else {
                set_last_error("live view: a chunk without postings and no snapshot");
                return Err(FfiStatus::Decode);
            };
            let totals = |f: i32| {
                let at = |t: &Totals| t.get(&f).copied().unwrap_or_default();
                (at(&p.start_totals), at(&p.end_totals))
            };
            let live = lucene_search::live_segment::range_segment(buffer, p.start, p.end, &totals)
                .map_err(map_search_error)?;
            let r = r.with_postings(live).map_err(map_search_error)?;
            // A planned chunk without postings is the one read here; a chunk
            // written for this view is kept once a later view plans it.
            if let (Some(k), Some(c)) = (key, p.chunk.as_ref()) {
                if !c.postings {
                    c.keep_overlay(k, &r);
                }
            }
            Ok(r)
        })
        .collect()
}

fn open_error(e: lucene_search::directory_reader::Error) -> lucene_index::index_writer::Error {
    lucene_index::index_writer::Error::Store(lucene_store::Error::Corrupted(format!(
        "opening the buffered documents: {e}"
    )))
}

/// A buffer segment's live documents: the first `max_doc` of the
/// `written`, less `deleted`; `None` when that is all of them.
fn buffer_mask(max_doc: i32, deleted: &[i32], written: i32) -> Option<FixedBitSet> {
    if deleted.is_empty() && max_doc >= written {
        return None;
    }
    let len = usize::try_from(written).unwrap_or(0);
    let mut bits = FixedBitSet::new(len);
    bits.set_range(0, usize::try_from(max_doc).unwrap_or(0).min(len));
    for &d in deleted {
        if let Ok(d) = usize::try_from(d) {
            if d < bits.len() {
                bits.clear(d);
            }
        }
    }
    Some(bits)
}

/// Stops keeping the published buffers the commit at `generation` holds,
/// once no open view reads them.
#[no_mangle]
pub extern "C" fn ffi_engine_writer_release_live(writer: u64, generation: i64) -> i32 {
    guard(|| {
        let w = engine_writer::lookup(writer)?;
        w.live_warmer().unshare_before(generation);
        w.writer().release_live(generation).map_err(live_error)
    })
}

/// The bounds of field `field`'s points in buffer `buffer` of `view` as the view
/// opened ([`lucene_index::concurrent_writer::ConcurrentIndexWriter::live_point_bounds`]),
/// as an owned buffer: the point count as a little-endian `u64`, then the
/// packed least values, then the packed greatest; empty when the buffer
/// holds no point of the field.
///
/// # Safety
/// `out_ptr`/`out_len` valid for one write each.
#[no_mangle]
pub unsafe extern "C" fn ffi_live_view_point_bounds_alloc(
    view_id: u64,
    buffer: i32,
    field: i32,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        unsafe { crate::ffm_bridge::clear_out(out_ptr, out_len)? };
        let h = view(view_id)?;
        let buffer = usize::try_from(buffer).map_err(|_| FfiStatus::IndexOutOfBounds)?;
        let bounds = h
            .writer
            .writer()
            .live_point_bounds(&h.view, buffer, field)
            .map_err(live_error)?;
        let mut out = Vec::new();
        if let Some(b) = bounds {
            out.extend_from_slice(&b.count.to_le_bytes());
            out.extend_from_slice(&b.min);
            out.extend_from_slice(&b.max);
        }
        // SAFETY: caller contract.
        unsafe { crate::ffm_bridge::give(out, out_ptr, out_len) };
        Ok(())
    })
}

/// Document `doc` of buffer `buffer` of `view`, its stored fields encoded
/// as [`crate::jvm_fetch`]'s document blob, as an owned buffer.
///
/// # Safety
/// `out_ptr`/`out_len` valid for one write each.
#[no_mangle]
pub unsafe extern "C" fn ffi_live_view_document_alloc(
    view_id: u64,
    buffer: i32,
    doc: i32,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        unsafe { crate::ffm_bridge::clear_out(out_ptr, out_len)? };
        let h = view(view_id)?;
        let buffer = usize::try_from(buffer).map_err(|_| FfiStatus::IndexOutOfBounds)?;
        let fields = h
            .writer
            .writer()
            .live_document(&h.view, buffer, doc)
            .map_err(live_error)?;
        let mut out = len_i32(fields.len())?.to_le_bytes().to_vec();
        for f in &fields {
            out.extend_from_slice(&f.field_number.to_le_bytes());
            match &f.value {
                FieldValue::String(s) => {
                    out.push(STRING);
                    out.extend_from_slice(&len_i32(s.len())?.to_le_bytes());
                    out.extend_from_slice(s.as_bytes());
                }
                FieldValue::Binary(b) => {
                    out.push(BINARY);
                    out.extend_from_slice(&len_i32(b.len())?.to_le_bytes());
                    out.extend_from_slice(b);
                }
                FieldValue::Int(v) => {
                    out.push(INT);
                    out.extend_from_slice(&v.to_le_bytes());
                }
                FieldValue::Long(v) => {
                    out.push(LONG);
                    out.extend_from_slice(&v.to_le_bytes());
                }
                FieldValue::Float(v) => {
                    out.push(FLOAT);
                    out.extend_from_slice(&v.to_bits().to_le_bytes());
                }
                FieldValue::Double(v) => {
                    out.push(DOUBLE);
                    out.extend_from_slice(&v.to_bits().to_le_bytes());
                }
            }
        }
        // SAFETY: caller contract.
        unsafe { crate::ffm_bridge::give(out, out_ptr, out_len) };
        Ok(())
    })
}

/// Document `doc` of buffer `buffer` of `view`: its doc values of the
/// `count` fields numbered at `fields`, encoded as [`crate::jvm_fetch`]'s
/// doc-values blob, as an owned buffer. A buffer has no terms dictionary:
/// a document's terms are numbered `0..n` in order, ordinals of that
/// document alone.
///
/// # Safety
/// `fields` valid for `count` `i32` reads (null when 0); `out_ptr`/`out_len`
/// valid for one write each.
#[no_mangle]
pub unsafe extern "C" fn ffi_live_view_doc_values_alloc(
    view_id: u64,
    buffer: i32,
    doc: i32,
    fields: *const i32,
    count: usize,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        unsafe { crate::ffm_bridge::clear_out(out_ptr, out_len)? };
        if fields.is_null() && count > 0 {
            return Err(FfiStatus::NullPointer);
        }
        let fields: &[i32] = if count == 0 {
            &[]
        } else {
            // SAFETY: caller contract, non-null checked above.
            unsafe { std::slice::from_raw_parts(fields, count) }
        };
        let h = view(view_id)?;
        let buffer = usize::try_from(buffer).map_err(|_| FfiStatus::IndexOutOfBounds)?;
        let values = h
            .writer
            .writer()
            .live_doc_values(&h.view, buffer, doc, fields)
            .map_err(live_error)?;
        let out = encode_live_doc_values(fields, &values)?;
        // SAFETY: caller contract.
        unsafe { crate::ffm_bridge::give(out, out_ptr, out_len) };
        Ok(())
    })
}

/// `values` (a buffered document's, by field number) as the doc-values blob
/// of `fields`, in their order; a field without values is
/// [`crate::jvm_fetch::DV_NONE`].
fn encode_live_doc_values(
    fields: &[i32],
    values: &[(i32, lucene_index::concurrent_writer::LiveDocValue)],
) -> Result<Vec<u8>, FfiStatus> {
    use crate::jvm_fetch::{DV_NONE, DV_SORTED_NUMERIC, DV_SORTED_SET};
    use lucene_index::concurrent_writer::LiveDocValue;
    let mut out = len_i32(fields.len())?.to_le_bytes().to_vec();
    for &number in fields {
        out.extend_from_slice(&number.to_le_bytes());
        match values.iter().find(|(n, _)| *n == number).map(|(_, v)| v) {
            Some(LiveDocValue::Numbers(v)) if !v.is_empty() => {
                out.push(DV_SORTED_NUMERIC);
                out.extend_from_slice(&len_i32(v.len())?.to_le_bytes());
                for x in v {
                    out.extend_from_slice(&x.to_le_bytes());
                }
            }
            Some(LiveDocValue::Bytes(v)) if !v.is_empty() => {
                out.push(DV_SORTED_SET);
                out.extend_from_slice(&len_i32(v.len())?.to_le_bytes());
                for (ord, term) in v.iter().enumerate() {
                    out.extend_from_slice(&i64::try_from(ord).unwrap_or(i64::MAX).to_le_bytes());
                    out.extend_from_slice(&len_i32(term.len())?.to_le_bytes());
                    out.extend_from_slice(term);
                }
            }
            _ => out.push(DV_NONE),
        }
    }
    Ok(out)
}

/// [`crate::jvm_reader::ffi_jvm_reader_search`] over the reader at
/// `handle` and the buffers of `view` together: a buffered hit is
/// numbered after the reader's `maxDoc`, the buffers in the view's order.
/// Every field's average length is over both. `*out_supported` is false,
/// and nothing else is written, when the query reads something the buffers
/// cannot answer (see [`live_request`]) or carries a `min_score`; the
/// caller then makes the buffers part of a reader first.
///
/// # Safety
/// As [`crate::jvm_reader::ffi_jvm_reader_search`]; `out_supported` valid
/// for one write.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ffi_jvm_reader_search_live(
    handle: u64,
    view_id: u64,
    query: *const u8,
    query_len: usize,
    top_n: usize,
    count_limit: i64,
    out_docs: *mut i32,
    out_scores: *mut f32,
    buf_len: usize,
    out_hit_count: *mut usize,
    out_total: *mut i64,
    out_total_is_lower_bound: *mut bool,
    out_supported: *mut bool,
) -> i32 {
    guard(|| {
        let _timer =
            lucene_index::live_timing::start(lucene_index::live_timing::Phase::PlainSearch);
        if out_hit_count.is_null()
            || out_total.is_null()
            || out_total_is_lower_bound.is_null()
            || out_supported.is_null()
        {
            return Err(FfiStatus::NullPointer);
        }
        if buf_len < top_n {
            return Err(FfiStatus::BufferTooSmall);
        }
        if top_n > 0 && (out_docs.is_null() || out_scores.is_null()) {
            return Err(FfiStatus::NullPointer);
        }
        // SAFETY: caller contract.
        let blob = unsafe { bytes_from_raw(query, query_len)? };
        let (query, min_score, sim) = decode_request(blob)?;
        let q = boolean_of(&query);
        let clause = Clause::Boolean(Box::new(q.clone()));
        let request = match (min_score, live_request(&clause)) {
            (None, Some(r)) => r,
            _ => {
                // SAFETY: caller contract.
                unsafe { *out_supported = false };
                return Ok(());
            }
        };
        let h = lookup(
            handle,
            "ffi_jvm_reader_search_live: unknown or closed reader",
        )?;
        let v = view(view_id)?;
        searched_lean(&v.writer);
        let mut laps = lucene_index::live_timing::Laps::start();
        // Each buffer is read as the chunks its view's plan starts with,
        // with this request's terms (their overlays kept on the chunks),
        // then a segment of the documents after them.
        let key = {
            let mut k = vec![PLAIN_KEY];
            k.extend(request_key(blob, None, None));
            k
        };
        if v.writer.live_warmer().max_staleness().is_some() {
            v.writer.live_warmer().note_plain(blob);
        }
        let prepared = prepare_plain(&v, &request, &key)?;
        let Prepared {
            chunk_readers,
            chunk_masks,
            chunk_docs,
            live,
            prefix_lens,
        } = &*prepared;
        laps.lap(lucene_index::live_timing::Phase::PlainPrepare);
        let inputs = live
            .iter()
            .map(|l| l.inputs())
            .collect::<lucene_search::Result<Vec<_>>>()
            .map_err(map_search_error)?;

        let chunked;
        let reader = if chunk_readers.is_empty() {
            &h.reader
        } else {
            chunked = h
                .reader
                .with_readers_appended(chunk_readers.clone())
                .map_err(|e| {
                    set_last_error(format!("opening the buffered documents: {e}"));
                    FfiStatus::Decode
                })?;
            &chunked
        };
        let opened = reader.open_segments().map_err(|e| {
            set_last_error(format!("opening segment postings: {e}"));
            FfiStatus::Decode
        })?;
        laps.lap(lucene_index::live_timing::Phase::PlainOpen);
        let sim = sim.as_ref();
        let committed = h.live_docs.len();
        let mut opened_segments: Vec<OpenSegment<'_>> = opened.as_open_segments();
        let masks = commit_masks(
            &v.view.commit_deletes,
            &h,
            opened_segments.get(..committed).unwrap_or_default(),
        )?;
        for (i, s) in opened_segments.iter_mut().enumerate() {
            s.live_docs = match i.checked_sub(committed) {
                Some(c) => chunk_masks.get(c).and_then(Option::as_ref),
                None => match &masks {
                    Some(m) => m.get(i).and_then(Option::as_deref),
                    None => h.live_docs.get(i).and_then(Option::as_ref),
                },
            };
            s.similarity = sim.map(|p| p as &dyn Similarity);
        }

        let fields: Vec<String> = crate::clauses::clause_field_names(&q)
            .into_iter()
            .map(str::to_string)
            .collect();
        let extra = |field: &str| {
            let mut found = None;
            for l in live {
                if let Some([_, ttf, _, dc]) = l.field_stats(field) {
                    let (t, d) = found.unwrap_or((0i64, 0i64));
                    found = Some((t.saturating_add(ttf), d.saturating_add(dc)));
                }
            }
            found
        };
        let (reader_norms, averages) = reader
            .field_norms_by_field_with(&fields, &extra)
            .map_err(|e| map_search_error(e.into()))?;
        let live_norms: Vec<_> = live
            .iter()
            .map(|l| l.norms_with_average(&|f| averages.get(f).copied()))
            .collect();
        laps.lap(lucene_index::live_timing::Phase::PlainNorms);
        // In document order: the committed segments, then each buffer's
        // chunks and the segment after them.
        let mut segments: Vec<OpenSegment<'_>> = Vec::new();
        let mut norms = Vec::new();
        let mut opened_iter = opened_segments.into_iter().zip(reader_norms.iter());
        for _ in 0..committed {
            if let Some((s, n)) = opened_iter.next() {
                segments.push(s);
                norms.push(n);
            }
        }
        let mut doc_base = h.reader.max_doc();
        let mut chunk_docs = chunk_docs.iter().copied();
        for (((&n, l), i), ln) in prefix_lens.iter().zip(live).zip(&inputs).zip(&live_norms) {
            for _ in 0..n {
                if let (Some((mut s, n)), Some(docs)) = (opened_iter.next(), chunk_docs.next()) {
                    s.doc_base = doc_base;
                    doc_base = doc_base.saturating_add(docs);
                    segments.push(s);
                    norms.push(n);
                }
            }
            let mut s = l.segment(i, doc_base);
            s.similarity = sim.map(|p| p as &dyn Similarity);
            segments.push(s);
            norms.push(ln);
            doc_base = doc_base.saturating_add(l.max_doc());
        }
        let norms: Vec<_> = norms
            .into_iter()
            .map(|m| (!m.is_empty()).then_some(m))
            .collect();
        let threshold = u64::try_from(count_limit).unwrap_or(0);
        let (hits, total) = search_boolean_query_multi_segment_maxscore_counting(
            &segments,
            &q,
            &norms,
            top_n.max(1),
            threshold,
        )
        .map_err(map_search_error)?;
        laps.lap(lucene_index::live_timing::Phase::PlainQuery);
        let hits = if top_n == 0 { Vec::new() } else { hits };
        let (total, lower_bound) = if count_limit <= 0 {
            (-1, false)
        } else {
            (
                i64::try_from(total.value).unwrap_or(i64::MAX),
                total.relation == TotalHitsRelation::GreaterThanOrEqualTo,
            )
        };
        // SAFETY: caller contract; `hits.len() <= top_n <= buf_len`.
        unsafe {
            for (i, hit) in hits.iter().enumerate() {
                *out_docs.add(i) = hit.doc_id;
                *out_scores.add(i) = hit.score;
            }
            *out_hit_count = hits.len();
            *out_total = total;
            *out_total_is_lower_bound = lower_bound;
            *out_supported = true;
        }
        Ok(())
    })
}

/// The most plain requests' reads a view keeps ([`ViewHandle::prepared`]).
const PREPARED: usize = 8;

/// The most plain requests the warmer reads each shared view for
/// ([`Warmer::recent_plain`]).
const RECENT_PLAIN: usize = 8;

/// What a plain request reads of a view's buffers: each buffer's chunks
/// (with the request's terms overlaid) and then a segment of the documents
/// after them.
struct Prepared {
    chunk_readers: Vec<SegmentReader>,
    chunk_masks: Vec<Option<FixedBitSet>>,
    chunk_docs: Vec<i32>,
    /// By buffer, the segment after its chunks.
    live: Vec<lucene_search::live_segment::LiveSegment>,
    /// By buffer, how many of `chunk_readers` are its.
    prefix_lens: Vec<usize>,
}

/// [`Prepared`] for the request `blob` decodes to, kept on `v`; nothing for
/// a request a plain search does not answer.
fn prepare_plain_blob(v: &ViewHandle, blob: &[u8]) -> Result<(), FfiStatus> {
    let (query, min_score, _) = decode_request(blob)?;
    let clause = Clause::Boolean(Box::new(boolean_of(&query)));
    if let (None, Some(request)) = (min_score, live_request(&clause)) {
        let mut key = vec![PLAIN_KEY];
        key.extend(request_key(blob, None, None));
        prepare_plain(v, &request, &key)?;
    }
    Ok(())
}

/// What `request` (whose [`request_key`] is `key`) reads of `v`'s buffers:
/// made once per view and request, and kept.
fn prepare_plain(
    v: &ViewHandle,
    request: &LiveRequest,
    key: &[u8],
) -> Result<Arc<Prepared>, FfiStatus> {
    let lock = || {
        v.prepared
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    };
    if let Some((_, p)) = lock().iter().find(|(k, _)| k == key) {
        return Ok(Arc::clone(p));
    }
    let prefixes: Vec<&[Planned]> = v
        .view
        .buffers
        .iter()
        .zip(&v.plans)
        .map(|(b, plan)| {
            let n = if b.in_slot {
                plan.iter().take_while(|p| p.chunk.is_some()).count()
            } else {
                0
            };
            plan.get(..n).unwrap_or_default()
        })
        .collect();
    // The overlays found now are the ones used: one evicted later must
    // not be rebuilt from a read that starts past its chunk.
    let cached: Vec<Vec<Option<SegmentReader>>> = prefixes
        .iter()
        .map(|prefix| {
            prefix
                .iter()
                .map(|p| cached_overlay(p, Some(key)))
                .collect()
        })
        .collect();
    // Each buffer is read from its first chunk without this request's
    // overlay, else from the documents after its chunks.
    let from: Vec<i32> = prefixes
        .iter()
        .zip(&cached)
        .map(|(prefix, cached)| {
            let uncached = prefix
                .iter()
                .zip(cached)
                .find(|(p, c)| p.chunk.as_ref().is_some_and(|c| !c.postings) && c.is_none());
            uncached.map_or_else(|| prefix.last().map_or(0, |p| p.end), |(p, _)| p.start)
        })
        .collect();
    let snapshot = v
        .writer
        .writer()
        .live_view_snapshot_from(&v.view, request, &from)
        .map_err(live_error)?;
    let mut chunk_readers = Vec::new();
    let mut chunk_masks = Vec::new();
    let mut chunk_docs = Vec::new();
    let mut live = Vec::new();
    for ((prefix, cached), buffer) in prefixes.iter().zip(cached).zip(snapshot.buffers) {
        let parts: Vec<(SegmentReader, bool)> = prefix
            .iter()
            .filter_map(|p| p.chunk.as_ref().map(|c| (c.reader.clone(), !c.postings)))
            .collect();
        for (p, r) in prefix
            .iter()
            .zip(overlaid(parts, prefix, cached, Some(&buffer), Some(key))?)
        {
            // The snapshot's deletes: every document of a buffer
            // `delete_all` or `rollback` dropped since the view opened.
            let lo = buffer.deleted.partition_point(|&d| d < p.start);
            let hi = buffer.deleted.partition_point(|&d| d < p.end);
            let deleted: Vec<i32> = buffer
                .deleted
                .get(lo..hi)
                .unwrap_or_default()
                .iter()
                .map(|&d| d.saturating_sub(p.start))
                .collect();
            chunk_masks.push(buffer_mask(
                p.end.saturating_sub(p.start),
                &deleted,
                r.max_doc,
            ));
            chunk_docs.push(r.max_doc);
            chunk_readers.push(r);
        }
        let start = prefix.last().map_or(0, |p| p.end);
        let at_start = |f: i32| {
            prefix
                .last()
                .and_then(|p| p.end_totals.get(&f).copied())
                .unwrap_or_default()
        };
        live.push(
            lucene_search::live_segment::tail_segment(buffer, start, &at_start)
                .map_err(map_search_error)?,
        );
    }
    let prepared = Arc::new(Prepared {
        chunk_readers,
        chunk_masks,
        chunk_docs,
        live,
        prefix_lens: prefixes.iter().map(|p| p.len()).collect(),
    });
    let mut kept = lock();
    if !kept.iter().any(|(k, _)| k == key) {
        if kept.len() >= PREPARED {
            kept.remove(0);
        }
        kept.push((key.to_vec(), Arc::clone(&prepared)));
    }
    Ok(prepared)
}

/// The reader's live documents with `terms` (a view's
/// [`LiveView::commit_deletes`]) applied to `segments`, its segments; `None`
/// when there are none to apply. Kept on the reader for the next search.
fn commit_masks(
    terms: &[Term],
    h: &JvmReaderHandle,
    segments: &[OpenSegment<'_>],
) -> Result<Option<Arc<Masks>>, FfiStatus> {
    let Some(last) = terms.last() else {
        return Ok(None);
    };
    let mut cached = h
        .commit_live
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let extends = cached.as_ref().filter(|c| {
        c.applied <= terms.len()
            && c.applied
                .checked_sub(1)
                .and_then(|i| terms.get(i))
                .is_some_and(|t| *t == c.last)
    });
    let (from, mut masks) = match extends {
        Some(c) if c.applied == terms.len() => return Ok(Some(Arc::clone(&c.masks))),
        Some(c) => (c.applied, Masks::clone(&c.masks)),
        None => (
            0,
            h.live_docs
                .iter()
                .map(|l| l.clone().map(Arc::new))
                .collect(),
        ),
    };
    let mut hits = VecCollector::default();
    for t in &terms[from..] {
        let query = TermQuery::new(t.field.as_str(), t.bytes.as_slice());
        for (seg, mask) in segments.iter().zip(masks.iter_mut()) {
            hits.docs.clear();
            search_term_query(seg.fields, seg.doc_in, None, &query, &mut hits)
                .map_err(map_search_error)?;
            if hits.docs.is_empty() {
                continue;
            }
            let max_doc = usize::try_from(seg.max_doc.unwrap_or(0)).unwrap_or(0);
            let mask = mask.get_or_insert_with(|| {
                let mut all = FixedBitSet::new(max_doc);
                all.set_range(0, max_doc);
                Arc::new(all)
            });
            let mask = Arc::make_mut(mask);
            let len = mask.len();
            for &d in &hits.docs {
                if let Ok(d) = usize::try_from(d) {
                    if d < len {
                        mask.clear(d);
                    }
                }
            }
        }
    }
    let masks = Arc::new(masks);
    // A view older than the cached masks keeps its own.
    if cached.as_ref().is_none_or(|c| c.applied <= terms.len()) {
        *cached = Some(CommitLive {
            applied: terms.len(),
            last: last.clone(),
            masks: Arc::clone(&masks),
        });
    }
    Ok(Some(masks))
}
