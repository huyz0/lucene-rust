//! Near-real-time searcher management: `ReferenceManager`, `SearcherManager`,
//! `SearcherFactory`, `RefreshCommitSupplier`, `SearcherLifetimeManager`,
//! `ControlledRealTimeReopenThread` and `LiveFieldValues`.
//!
//! # How reference counting maps
//!
//! Java counts references on the `IndexReader` (`incRef`/`tryIncRef`/
//! `decRef`) and closes it at zero. Here the managed reference is an
//! [`Arc`]: [`ReferenceManager::acquire`] clones it, dropping (or
//! [`ReferenceManager::release`]) is `decRef`, and the reader is freed when the
//! last holder lets go -- so "acquire always succeeds while the manager is
//! open" (Java's `tryIncRef` loop cannot fail on an `Arc`), and the "managed
//! reference has already closed" state cannot arise.
//!
//! A searcher here is built from a reader per request ([`IndexSearcher`]
//! borrows the reader's opened segments), so what [`SearcherManager`] manages
//! is the reader, wrapped by a [`SearcherFactory`] -- Java's hook for warming a
//! new searcher and setting its similarity -- into whatever the caller
//! searches with.
//!
//! [`SearcherManager`] refreshes either from the directory's commits
//! (`SearcherManager(Directory, SearcherFactory)`,
//! `DirectoryReader.openIfChanged`) or, near-real-time, from a writer
//! (`SearcherManager(IndexWriter, applyAllDeletes, writeAllDeletes,
//! SearcherFactory)`, `DirectoryReader.openIfChanged(reader, writer)` --
//! [`SearcherManager::from_writer`], over any
//! [`lucene_index::nrt::NrtSource`]). [`ControlledRealTimeReopenThread`]
//! takes its generations from a caller-supplied source -- the writer's last
//! committed (or flushed) sequence number -- where Java reads
//! `IndexWriter.getMaxCompletedSequenceNumber`.
//!
//! [`IndexSearcher`]: crate::index_searcher::IndexSearcher

use std::collections::HashMap;
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use lucene_index::segment_infos::{self, SegmentInfos};
use lucene_store::directory::Directory;

use crate::directory_reader::DirectoryReader;
use crate::{Error, Result};

fn closed_error() -> Error {
    Error::AlreadyClosed("this ReferenceManager is closed".to_string())
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    // A poisoned lock means another thread panicked holding it; the state it
    // guards is always whole between statements.
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// `ReferenceManager.RefreshListener`.
pub trait RefreshListener: Send + Sync {
    /// `beforeRefresh()`.
    fn before_refresh(&self) -> Result<()>;
    /// `afterRefresh(didRefresh)`.
    fn after_refresh(&self, did_refresh: bool) -> Result<()>;
}

/// `ReferenceManager`'s abstract half: how a reference is refreshed.
pub trait Refresher<G>: Send + Sync {
    /// `refreshIfNeeded(referenceToRefresh)`: a new reference, or `None` when
    /// nothing changed.
    fn refresh_if_needed(&self, current: &Arc<G>) -> Result<Option<Arc<G>>>;
    /// `afterMaybeRefresh()`.
    fn after_maybe_refresh(&self) -> Result<()> {
        Ok(())
    }
    /// `afterClose()`.
    fn after_close(&self) -> Result<()> {
        Ok(())
    }
}

/// `ReferenceManager<G>`: shares one current reference between threads and
/// swaps in a refreshed one.
pub struct ReferenceManager<G> {
    current: RwLock<Option<Arc<G>>>,
    /// `refreshLock`: one refresh at a time.
    refresh_lock: Mutex<()>,
    listeners: RwLock<Vec<Arc<dyn RefreshListener>>>,
    refresher: Box<dyn Refresher<G>>,
}

impl<G: Send + Sync> ReferenceManager<G> {
    pub fn new(initial: Arc<G>, refresher: Box<dyn Refresher<G>>) -> Self {
        Self {
            current: RwLock::new(Some(initial)),
            refresh_lock: Mutex::new(()),
            listeners: RwLock::new(Vec::new()),
            refresher,
        }
    }

    fn current(&self) -> Option<Arc<G>> {
        self.current
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// `acquire()`: the current reference, held until dropped.
    ///
    /// # Errors
    /// [`Error::AlreadyClosed`] after [`Self::close`].
    pub fn acquire(&self) -> Result<Arc<G>> {
        self.current().ok_or_else(closed_error)
    }

    /// `release(reference)`: gives an acquired reference back.
    pub fn release(&self, reference: Arc<G>) {
        drop(reference);
    }

    /// Whether [`Self::close`] has run.
    pub fn is_closed(&self) -> bool {
        self.current().is_none()
    }

    /// `swapReference(newReference)`.
    fn swap_reference(&self, new: Option<Arc<G>>) -> Result<()> {
        let mut cur = self.current.write().unwrap_or_else(|e| e.into_inner());
        if cur.is_none() {
            return Err(closed_error());
        }
        // The old reference is released when this `Arc` drops.
        *cur = new;
        Ok(())
    }

    fn listeners(&self) -> Vec<Arc<dyn RefreshListener>> {
        self.listeners
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// `doMaybeRefresh()`, under the refresh lock.
    fn do_maybe_refresh(&self) -> Result<()> {
        let reference = self.acquire()?;
        for l in self.listeners() {
            l.before_refresh()?;
        }
        let mut refreshed = false;
        let result = (|| {
            if let Some(new) = self.refresher.refresh_if_needed(&reference)? {
                if Arc::ptr_eq(&new, &reference) {
                    return Err(Error::IllegalState(
                        "refreshIfNeeded should return null if refresh wasn't needed".to_string(),
                    ));
                }
                self.swap_reference(Some(new))?;
                refreshed = true;
            }
            Ok(())
        })();
        drop(reference);
        for l in self.listeners() {
            l.after_refresh(refreshed)?;
        }
        result?;
        self.refresher.after_maybe_refresh()
    }

    /// `maybeRefresh()`: refreshes unless another thread is refreshing right
    /// now; `true` when this call did (or checked).
    ///
    /// # Errors
    /// [`Error::AlreadyClosed`] after [`Self::close`], and the refresh's own.
    pub fn maybe_refresh(&self) -> Result<bool> {
        if self.is_closed() {
            return Err(closed_error());
        }
        match self.refresh_lock.try_lock() {
            Ok(_guard) => {
                self.do_maybe_refresh()?;
                Ok(true)
            }
            Err(std::sync::TryLockError::Poisoned(p)) => {
                let _guard = p.into_inner();
                self.do_maybe_refresh()?;
                Ok(true)
            }
            Err(std::sync::TryLockError::WouldBlock) => Ok(false),
        }
    }

    /// `maybeRefreshBlocking()`: waits for any refresh in progress, then
    /// refreshes.
    ///
    /// # Errors
    /// [`Error::AlreadyClosed`] after [`Self::close`], and the refresh's own.
    pub fn maybe_refresh_blocking(&self) -> Result<()> {
        if self.is_closed() {
            return Err(closed_error());
        }
        let _guard = lock(&self.refresh_lock);
        self.do_maybe_refresh()
    }

    /// `close()`: drops the current reference (holders keep theirs); later
    /// calls do nothing.
    pub fn close(&self) -> Result<()> {
        if self.is_closed() {
            return Ok(());
        }
        self.swap_reference(None)?;
        self.refresher.after_close()
    }

    /// `addListener(listener)`.
    pub fn add_listener(&self, listener: Arc<dyn RefreshListener>) {
        self.listeners
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .push(listener);
    }

    /// `removeListener(listener)`: by identity.
    pub fn remove_listener(&self, listener: &Arc<dyn RefreshListener>) {
        self.listeners
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .retain(|l| !Arc::ptr_eq(l, listener));
    }
}

// ---------------------------------------------------------------------------
// SearcherManager
// ---------------------------------------------------------------------------

/// `SearcherFactory`: makes what a [`SearcherManager`] hands out from each
/// new reader -- Java's hook to warm a searcher or set its similarity before
/// it is published.
pub trait SearcherFactory: Send + Sync {
    type Searcher: Send + Sync;
    /// `newSearcher(reader, previousReader)`: `previous` is the searcher being
    /// replaced (`None` for the first).
    fn new_searcher(
        &self,
        reader: DirectoryReader,
        previous: Option<&Self::Searcher>,
    ) -> Result<Self::Searcher>;
    /// The reader a searcher wraps (`IndexSearcher.getIndexReader()`); a
    /// factory must wrap exactly the reader it was given.
    fn reader<'s>(&self, searcher: &'s Self::Searcher) -> &'s DirectoryReader;
}

/// `new SearcherFactory()`: the reader itself.
#[derive(Debug, Default, Clone, Copy)]
pub struct DefaultSearcherFactory;

impl SearcherFactory for DefaultSearcherFactory {
    type Searcher = DirectoryReader;
    fn new_searcher(
        &self,
        reader: DirectoryReader,
        _previous: Option<&DirectoryReader>,
    ) -> Result<DirectoryReader> {
        Ok(reader)
    }
    fn reader<'s>(&self, searcher: &'s DirectoryReader) -> &'s DirectoryReader {
        searcher
    }
}

/// `RefreshCommitSupplier`: which commit a refresh opens -- `None` for the
/// latest.
pub trait RefreshCommitSupplier: Send + Sync {
    /// `getSearcherRefreshCommit(reader)`.
    fn searcher_refresh_commit(&self, _reader: &DirectoryReader) -> Result<Option<SegmentInfos>> {
        Ok(None)
    }
}

/// The default `RefreshCommitSupplier`: always the latest commit.
#[derive(Debug, Default, Clone, Copy)]
pub struct LatestCommit;

impl RefreshCommitSupplier for LatestCommit {}

/// `SegmentInfos.FindSegmentsFile.run`: `body` opens the latest commit; a
/// writer committing concurrently may delete that commit's files between
/// the listing and the open, so a failure is retried for as long as the
/// latest commit generation keeps advancing, and reported once it stops.
fn find_segments_file<T>(dir: &dyn Directory, mut body: impl FnMut() -> Result<T>) -> Result<T> {
    let mut last_gen = -1i64;
    let mut first_err = None;
    loop {
        let gen = dir
            .list_all()
            .and_then(|files| lucene_store::directory::last_commit_generation(&files))
            .map_err(|e| Error::DirectoryReader(e.into()))?;
        if gen <= last_gen {
            // No error yet only when the first listing found no commit at
            // all (`IndexNotFoundException`).
            return Err(first_err
                .unwrap_or_else(|| Error::IllegalState("no segments_N commit file found".into())));
        }
        match body() {
            Ok(v) => return Ok(v),
            Err(e) => {
                first_err.get_or_insert(e);
            }
        }
        last_gen = gen;
    }
}

/// `SearcherManager`'s `refreshIfNeeded`.
struct SearcherRefresher<F> {
    dir: Arc<dyn Directory>,
    factory: Arc<F>,
    commits: Box<dyn RefreshCommitSupplier>,
}

impl<F: SearcherFactory> Refresher<F::Searcher> for SearcherRefresher<F> {
    fn refresh_if_needed(&self, current: &Arc<F::Searcher>) -> Result<Option<Arc<F::Searcher>>> {
        let reader = self.factory.reader(current);
        let new = match self.commits.searcher_refresh_commit(reader)? {
            // `openIfChanged(reader, commit)`: nothing when it is the
            // reader's own commit.
            Some(infos) if infos.generation == reader.segment_infos.generation => None,
            Some(infos) => Some(reader.reopen_at(self.dir.as_ref(), infos)?),
            None => find_segments_file(self.dir.as_ref(), || {
                Ok(reader.open_if_changed(self.dir.as_ref())?)
            })?,
        };
        match new {
            None => Ok(None),
            Some(r) => Ok(Some(Arc::new(self.factory.new_searcher(r, Some(current))?))),
        }
    }
}

/// `SearcherManager`'s `refreshIfNeeded` for a manager opened from a writer:
/// `DirectoryReader.openIfChanged(reader, writer, applyAllDeletes,
/// writeAllDeletes)`.
struct NrtSearcherRefresher<F> {
    writer: Arc<dyn lucene_index::nrt::NrtSource>,
    factory: Arc<F>,
    apply_all_deletes: bool,
    write_all_deletes: bool,
}

impl<F: SearcherFactory> Refresher<F::Searcher> for NrtSearcherRefresher<F> {
    fn refresh_if_needed(&self, current: &Arc<F::Searcher>) -> Result<Option<Arc<F::Searcher>>> {
        let reader = self.factory.reader(current);
        match reader.open_if_changed_nrt_with(
            self.writer.as_ref(),
            self.apply_all_deletes,
            self.write_all_deletes,
        )? {
            None => Ok(None),
            Some(r) => Ok(Some(Arc::new(self.factory.new_searcher(r, Some(current))?))),
        }
    }
}

/// Where a [`SearcherManager`] refreshes from.
enum ManagedSource {
    /// The directory's latest commit.
    Directory(Arc<dyn Directory>),
    /// A writer, near-real-time.
    Writer(Arc<dyn lucene_index::nrt::NrtSource>),
}

/// `SearcherManager`: a [`ReferenceManager`] over searchers of a directory's
/// commits, or of a writer's near-real-time view.
pub struct SearcherManager<F: SearcherFactory = DefaultSearcherFactory> {
    manager: Arc<ReferenceManager<F::Searcher>>,
    factory: Arc<F>,
    source: ManagedSource,
}

impl SearcherManager<DefaultSearcherFactory> {
    /// `new SearcherManager(dir, null)`.
    pub fn open(dir: Arc<dyn Directory>) -> Result<Self> {
        Self::with_factory(dir, DefaultSearcherFactory)
    }

    /// `new SearcherManager(writer, null)`: near-real-time searchers of
    /// `writer`, every buffered delete applied.
    pub fn open_from_writer(writer: Arc<dyn lucene_index::nrt::NrtSource>) -> Result<Self> {
        Self::from_writer(writer, true, false, DefaultSearcherFactory)
    }
}

impl<F: SearcherFactory + 'static> SearcherManager<F> {
    /// `new SearcherManager(dir, searcherFactory)`: over the latest commit.
    pub fn with_factory(dir: Arc<dyn Directory>, factory: F) -> Result<Self> {
        let reader = find_segments_file(dir.as_ref(), || Ok(DirectoryReader::open(dir.as_ref())?))?;
        Self::from_reader(dir, reader, factory, Box::new(LatestCommit))
    }

    /// `new SearcherManager(reader, searcherFactory, refreshCommitSupplier)`.
    pub fn from_reader(
        dir: Arc<dyn Directory>,
        reader: DirectoryReader,
        factory: F,
        commits: Box<dyn RefreshCommitSupplier>,
    ) -> Result<Self> {
        let factory = Arc::new(factory);
        let first = Arc::new(factory.new_searcher(reader, None)?);
        let refresher = SearcherRefresher {
            dir: Arc::clone(&dir),
            factory: Arc::clone(&factory),
            commits,
        };
        Ok(Self {
            manager: Arc::new(ReferenceManager::new(first, Box::new(refresher))),
            factory,
            source: ManagedSource::Directory(dir),
        })
    }

    /// `new SearcherManager(writer, applyAllDeletes, writeAllDeletes,
    /// searcherFactory)`: the first searcher is `DirectoryReader.open(writer,
    /// applyAllDeletes, writeAllDeletes)`, and every refresh is
    /// `openIfChanged(reader, writer, ...)` -- documents the writer has
    /// indexed become searchable without a commit.
    pub fn from_writer(
        writer: Arc<dyn lucene_index::nrt::NrtSource>,
        apply_all_deletes: bool,
        write_all_deletes: bool,
        factory: F,
    ) -> Result<Self> {
        let reader =
            DirectoryReader::open_nrt(writer.as_ref(), apply_all_deletes, write_all_deletes)?;
        let factory = Arc::new(factory);
        let first = Arc::new(factory.new_searcher(reader, None)?);
        let refresher = NrtSearcherRefresher {
            writer: Arc::clone(&writer),
            factory: Arc::clone(&factory),
            apply_all_deletes,
            write_all_deletes,
        };
        Ok(Self {
            manager: Arc::new(ReferenceManager::new(first, Box::new(refresher))),
            factory,
            source: ManagedSource::Writer(writer),
        })
    }

    /// The underlying [`ReferenceManager`] (listeners, close, refresh).
    pub fn manager(&self) -> &ReferenceManager<F::Searcher> {
        &self.manager
    }

    /// The underlying [`ReferenceManager`], shared: what a
    /// [`ControlledRealTimeReopenThread`] or [`LiveFieldValues`] holds.
    pub fn shared_manager(&self) -> Arc<ReferenceManager<F::Searcher>> {
        Arc::clone(&self.manager)
    }

    /// `acquire()`.
    pub fn acquire(&self) -> Result<Arc<F::Searcher>> {
        self.manager.acquire()
    }

    /// `release(searcher)`.
    pub fn release(&self, searcher: Arc<F::Searcher>) {
        self.manager.release(searcher);
    }

    /// `maybeRefresh()`.
    pub fn maybe_refresh(&self) -> Result<bool> {
        self.manager.maybe_refresh()
    }

    /// `maybeRefreshBlocking()`.
    pub fn maybe_refresh_blocking(&self) -> Result<()> {
        self.manager.maybe_refresh_blocking()
    }

    /// `close()`.
    pub fn close(&self) -> Result<()> {
        self.manager.close()
    }

    /// `getSearcherCommitGeneration()`.
    pub fn searcher_commit_generation(&self) -> Result<i64> {
        let s = self.acquire()?;
        Ok(self.factory.reader(&s).segment_infos.generation)
    }

    /// `isSearcherCurrent()`: whether the searcher sees the directory's latest
    /// commit (`DirectoryReader.isCurrent`) -- or, for a manager opened from a
    /// writer, everything the writer has indexed.
    pub fn is_searcher_current(&self) -> Result<bool> {
        let s = self.acquire()?;
        let reader = self.factory.reader(&s);
        match &self.source {
            ManagedSource::Directory(dir) => {
                let latest = segment_infos::read_latest(dir.as_ref())
                    .map_err(crate::directory_reader::Error::from)?;
                Ok(latest.generation == reader.segment_infos.generation)
            }
            ManagedSource::Writer(writer) => Ok(reader.is_current_nrt(writer.as_ref())?),
        }
    }
}

// ---------------------------------------------------------------------------
// SearcherLifetimeManager
// ---------------------------------------------------------------------------

/// What a [`SearcherLifetimeManager`] keys a searcher by:
/// `DirectoryReader.getVersion()`.
pub trait Versioned {
    fn version(&self) -> i64;
}

impl Versioned for DirectoryReader {
    fn version(&self) -> i64 {
        self.segment_infos.version
    }
}

/// `SearcherLifetimeManager.SearcherTracker`.
struct Tracker<G> {
    searcher: Arc<G>,
    record_time: Instant,
    version: i64,
}

/// `SearcherLifetimeManager.Pruner`.
pub trait Pruner {
    /// `doPrune(ageSec, searcher)`: `age_sec` is how long since a newer
    /// searcher was recorded (`0` for the newest).
    fn do_prune(&self, age_sec: f64) -> bool;
}

/// `SearcherLifetimeManager.PruneByAge`.
#[derive(Debug, Clone, Copy)]
pub struct PruneByAge {
    max_age_sec: f64,
}

impl PruneByAge {
    /// # Errors
    /// [`Error::IllegalArgument`] for a negative age.
    pub fn new(max_age_sec: f64) -> Result<Self> {
        if max_age_sec < 0.0 {
            return Err(Error::IllegalArgument(format!(
                "maxAgeSec must be > 0 (got {max_age_sec})"
            )));
        }
        Ok(Self { max_age_sec })
    }
}

impl Pruner for PruneByAge {
    fn do_prune(&self, age_sec: f64) -> bool {
        age_sec > self.max_age_sec
    }
}

/// `SearcherLifetimeManager`: keeps recent searchers by version so a
/// follow-up request (the next page) searches the same point in time.
pub struct SearcherLifetimeManager<G> {
    searchers: Mutex<HashMap<i64, Tracker<G>>>,
    closed: std::sync::atomic::AtomicBool,
}

impl<G: Versioned> Default for SearcherLifetimeManager<G> {
    fn default() -> Self {
        Self {
            searchers: Mutex::new(HashMap::new()),
            closed: std::sync::atomic::AtomicBool::new(false),
        }
    }
}

impl<G: Versioned> SearcherLifetimeManager<G> {
    pub fn new() -> Self {
        Self::default()
    }

    fn ensure_open(&self) -> Result<()> {
        if self.closed.load(std::sync::atomic::Ordering::Acquire) {
            return Err(Error::AlreadyClosed(
                "this SearcherLifetimeManager instance is closed".to_string(),
            ));
        }
        Ok(())
    }

    /// `record(searcher)`: keeps `searcher` under its version and returns it.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] when a different searcher of the same
    /// version is already recorded; [`Error::AlreadyClosed`].
    pub fn record(&self, searcher: &Arc<G>) -> Result<i64> {
        self.ensure_open()?;
        let version = searcher.version();
        let mut map = lock(&self.searchers);
        match map.get(&version) {
            Some(t) if !Arc::ptr_eq(&t.searcher, searcher) => Err(Error::IllegalArgument(
                "the provided searcher has the same underlying reader version yet the searcher \
                 instance differs from before"
                    .to_string(),
            )),
            Some(_) => Ok(version),
            None => {
                map.insert(
                    version,
                    Tracker {
                        searcher: Arc::clone(searcher),
                        record_time: Instant::now(),
                        version,
                    },
                );
                Ok(version)
            }
        }
    }

    /// `acquire(version)`: the recorded searcher, or `None` once pruned.
    ///
    /// # Errors
    /// [`Error::AlreadyClosed`].
    pub fn acquire(&self, version: i64) -> Result<Option<Arc<G>>> {
        self.ensure_open()?;
        Ok(lock(&self.searchers)
            .get(&version)
            .map(|t| Arc::clone(&t.searcher)))
    }

    /// `release(searcher)`.
    pub fn release(&self, searcher: Arc<G>) {
        drop(searcher);
    }

    /// `prune(pruner)`: newest first; each searcher's age is the time since
    /// the next newer one was recorded.
    pub fn prune(&self, pruner: &dyn Pruner) {
        self.prune_at(pruner, Instant::now());
    }

    fn prune_at(&self, pruner: &dyn Pruner, now: Instant) {
        let mut map = lock(&self.searchers);
        let mut trackers: Vec<(Instant, i64)> =
            map.values().map(|t| (t.record_time, t.version)).collect();
        // `compareTo`: `Double.compare(other.recordTimeSec, recordTimeSec)`.
        trackers.sort_by_key(|t| std::cmp::Reverse(t.0));
        let mut last: Option<Instant> = None;
        for (record_time, version) in trackers {
            let age = last.map_or(0.0, |l| now.saturating_duration_since(l).as_secs_f64());
            if pruner.do_prune(age) {
                map.remove(&version);
            }
            last = Some(record_time);
        }
    }

    /// How many searchers are recorded.
    pub fn len(&self) -> usize {
        lock(&self.searchers).len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// `close()`: forgets every searcher; later calls fail.
    pub fn close(&self) {
        self.closed
            .store(true, std::sync::atomic::Ordering::Release);
        lock(&self.searchers).clear();
    }
}

// ---------------------------------------------------------------------------
// ControlledRealTimeReopenThread
// ---------------------------------------------------------------------------

/// The generations a [`ControlledRealTimeReopenThread`] tracks.
struct GenState {
    finish: bool,
    waiting_gen: i64,
    searching_gen: i64,
    refresh_start_gen: i64,
}

struct Gens {
    state: Mutex<GenState>,
    /// `reopenCond`: wakes the reopen thread.
    reopen: Condvar,
    /// The monitor `waitForGeneration` waits on.
    searching: Condvar,
    source: Box<dyn Fn() -> i64 + Send + Sync>,
}

/// `ControlledRealTimeReopenThread.HandleRefresh`.
struct HandleRefresh(Arc<Gens>);

impl RefreshListener for HandleRefresh {
    fn before_refresh(&self) -> Result<()> {
        let gen = (self.0.source)();
        lock(&self.0.state).refresh_start_gen = gen;
        Ok(())
    }
    fn after_refresh(&self, _did_refresh: bool) -> Result<()> {
        let mut st = lock(&self.0.state);
        st.searching_gen = st.refresh_start_gen;
        self.0.searching.notify_all();
        Ok(())
    }
}

/// `ControlledRealTimeReopenThread`: a thread that refreshes `manager` at
/// most every `target_max_stale`, and within `target_min_stale` when a caller
/// waits on a generation ([`Self::wait_for_generation`]).
pub struct ControlledRealTimeReopenThread<G: Send + Sync + 'static> {
    gens: Arc<Gens>,
    handle: Mutex<Option<JoinHandle<Result<()>>>>,
    _manager: Arc<ReferenceManager<G>>,
}

impl<G: Send + Sync + 'static> ControlledRealTimeReopenThread<G> {
    /// Starts the thread. `generation` is `IndexWriter.
    /// getMaxCompletedSequenceNumber()`: the latest generation a refresh
    /// starting now would make visible.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] when `target_max_stale < target_min_stale`.
    pub fn start(
        manager: Arc<ReferenceManager<G>>,
        generation: Box<dyn Fn() -> i64 + Send + Sync>,
        target_max_stale: Duration,
        target_min_stale: Duration,
    ) -> Result<Self> {
        if target_max_stale < target_min_stale {
            return Err(Error::IllegalArgument(format!(
                "targetMaxScaleSec (= {}) < targetMinStaleSec (={})",
                target_max_stale.as_secs_f64(),
                target_min_stale.as_secs_f64()
            )));
        }
        let gens = Arc::new(Gens {
            state: Mutex::new(GenState {
                finish: false,
                waiting_gen: 0,
                searching_gen: 0,
                refresh_start_gen: 0,
            }),
            reopen: Condvar::new(),
            searching: Condvar::new(),
            source: generation,
        });
        manager.add_listener(Arc::new(HandleRefresh(Arc::clone(&gens))));
        let (g, m) = (Arc::clone(&gens), Arc::clone(&manager));
        let handle = std::thread::Builder::new()
            .name("ControlledRealTimeReopenThread".to_string())
            .spawn(move || run(&g, &m, target_max_stale, target_min_stale))
            .map_err(|e| Error::IllegalState(format!("cannot start the reopen thread: {e}")))?;
        Ok(Self {
            gens,
            handle: Mutex::new(Some(handle)),
            _manager: manager,
        })
    }

    /// `waitForGeneration(targetGen, maxMS)`: blocks until a refresh has made
    /// `target_gen` searchable, or `max_wait` passes (`None`: forever);
    /// `false` on timeout.
    pub fn wait_for_generation(&self, target_gen: i64, max_wait: Option<Duration>) -> bool {
        let mut st = lock(&self.gens.state);
        if target_gen > st.searching_gen {
            st.waiting_gen = st.waiting_gen.max(target_gen);
            self.gens.reopen.notify_one();
            let start = Instant::now();
            while target_gen > st.searching_gen {
                match max_wait {
                    None => {
                        st = self
                            .gens
                            .searching
                            .wait(st)
                            .unwrap_or_else(|e| e.into_inner());
                    }
                    Some(max) => {
                        let Some(left) = max.checked_sub(start.elapsed()).filter(|l| !l.is_zero())
                        else {
                            return false;
                        };
                        st = self
                            .gens
                            .searching
                            .wait_timeout(st, left)
                            .unwrap_or_else(|e| e.into_inner())
                            .0;
                    }
                }
            }
        }
        true
    }

    /// `getSearchingGen()`.
    pub fn searching_gen(&self) -> i64 {
        lock(&self.gens.state).searching_gen
    }

    /// `close()`: stops and joins the thread; every waiter is released.
    ///
    /// # Errors
    /// The error a refresh on the thread failed with, if any.
    pub fn close(&self) -> Result<()> {
        {
            let mut st = lock(&self.gens.state);
            st.finish = true;
            self.gens.reopen.notify_one();
        }
        let handle = lock(&self.handle).take();
        let result = match handle {
            Some(h) => h
                .join()
                .unwrap_or_else(|_| Err(Error::IllegalState("the reopen thread panicked".into()))),
            None => Ok(()),
        };
        let mut st = lock(&self.gens.state);
        st.searching_gen = i64::MAX;
        self.gens.searching.notify_all();
        result
    }
}

impl<G: Send + Sync + 'static> Drop for ControlledRealTimeReopenThread<G> {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

/// `ControlledRealTimeReopenThread.run()`.
fn run<G: Send + Sync>(
    gens: &Gens,
    manager: &ReferenceManager<G>,
    max_stale: Duration,
    min_stale: Duration,
) -> Result<()> {
    let mut last_reopen_start = Instant::now();
    loop {
        {
            let mut st = lock(&gens.state);
            loop {
                if st.finish {
                    return Ok(());
                }
                let has_waiting = st.waiting_gen > st.searching_gen;
                let next = last_reopen_start + if has_waiting { min_stale } else { max_stale };
                let now = Instant::now();
                if next > now {
                    st = gens
                        .reopen
                        .wait_timeout(st, next - now)
                        .unwrap_or_else(|e| e.into_inner())
                        .0;
                } else {
                    break;
                }
            }
        }
        last_reopen_start = Instant::now();
        manager.maybe_refresh_blocking()?;
    }
}

// ---------------------------------------------------------------------------
// LiveFieldValues
// ---------------------------------------------------------------------------

/// `LiveFieldValues`' two maps: values written since the last refresh began
/// (`current`), and during the refresh in flight (`old`). `None` marks a
/// delete (Java's `missingValue`).
struct LiveMaps<T> {
    maps: Mutex<(ValueMap<T>, ValueMap<T>)>,
}

/// Values by id; `None` marks a delete.
type ValueMap<T> = HashMap<String, Option<T>>;

impl<T: Send> RefreshListener for LiveMaps<T> {
    /// `beforeRefresh()`: `old = current; current = new`.
    fn before_refresh(&self) -> Result<()> {
        let mut m = lock(&self.maps);
        m.1 = std::mem::take(&mut m.0);
        Ok(())
    }
    /// `afterRefresh()`: the refresh made `old`'s values searchable.
    fn after_refresh(&self, _did_refresh: bool) -> Result<()> {
        lock(&self.maps).1 = HashMap::new();
        Ok(())
    }
}

/// How [`LiveFieldValues`] reads an id not in its maps: `lookupFromSearcher`.
pub type Lookup<S, T> = dyn Fn(&S, &str) -> Result<Option<T>> + Send + Sync;

/// `LiveFieldValues<S, T>`: a field's latest values by id, for ids updated
/// since the searcher last refreshed, falling back to the searcher.
pub struct LiveFieldValues<S: Send + Sync, T: Clone + Send + 'static> {
    maps: Arc<LiveMaps<T>>,
    listener: Arc<dyn RefreshListener>,
    manager: Arc<ReferenceManager<S>>,
    lookup: Box<Lookup<S, T>>,
}

impl<S: Send + Sync, T: Clone + Send + 'static> LiveFieldValues<S, T> {
    /// `new LiveFieldValues(mgr, missingValue)` with its `lookupFromSearcher`.
    pub fn new(manager: Arc<ReferenceManager<S>>, lookup: Box<Lookup<S, T>>) -> Self {
        let maps = Arc::new(LiveMaps {
            maps: Mutex::new((HashMap::new(), HashMap::new())),
        });
        let listener: Arc<dyn RefreshListener> = maps.clone();
        manager.add_listener(Arc::clone(&listener));
        Self {
            maps,
            listener,
            manager,
            lookup,
        }
    }

    /// `add(id, value)`.
    pub fn add(&self, id: &str, value: T) {
        lock(&self.maps.maps).0.insert(id.to_string(), Some(value));
    }

    /// `delete(id)`.
    pub fn delete(&self, id: &str) {
        lock(&self.maps.maps).0.insert(id.to_string(), None);
    }

    /// `size()`.
    pub fn size(&self) -> usize {
        let m = lock(&self.maps.maps);
        m.0.len() + m.1.len()
    }

    /// `get(id)`: the value added last (`None` once deleted), else the
    /// searcher's.
    pub fn get(&self, id: &str) -> Result<Option<T>> {
        {
            let m = lock(&self.maps.maps);
            if let Some(v) = m.0.get(id) {
                return Ok(v.clone());
            }
            if let Some(v) = m.1.get(id) {
                return Ok(v.clone());
            }
        }
        let s = self.manager.acquire()?;
        (self.lookup)(&s, id)
    }

    /// `close()`: stops listening to the manager.
    pub fn close(&self) {
        self.manager.remove_listener(&self.listener);
    }
}

#[cfg(test)]
mod tests;
