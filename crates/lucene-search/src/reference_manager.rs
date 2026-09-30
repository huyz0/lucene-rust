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
//! [`SearcherManager`] refreshes from the directory's commits
//! (`SearcherManager(Directory, SearcherFactory)`,
//! `DirectoryReader.openIfChanged`): an NRT reader straight from an
//! `IndexWriter`'s unflushed buffers does not exist in this port (the writer
//! publishes segments through commits). [`ControlledRealTimeReopenThread`]
//! therefore takes its generations from a caller-supplied source -- the
//! writer's last committed sequence number, for a commit-refreshed manager --
//! where Java reads `IndexWriter.getMaxCompletedSequenceNumber`.
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

/// `SearcherManager`: a [`ReferenceManager`] over searchers of a directory's
/// commits.
pub struct SearcherManager<F: SearcherFactory = DefaultSearcherFactory> {
    manager: Arc<ReferenceManager<F::Searcher>>,
    factory: Arc<F>,
    dir: Arc<dyn Directory>,
}

impl SearcherManager<DefaultSearcherFactory> {
    /// `new SearcherManager(dir, null)`.
    pub fn open(dir: Arc<dyn Directory>) -> Result<Self> {
        Self::with_factory(dir, DefaultSearcherFactory)
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
            dir,
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
    /// commit (`DirectoryReader.isCurrent`).
    pub fn is_searcher_current(&self) -> Result<bool> {
        let s = self.acquire()?;
        let latest = segment_infos::read_latest(self.dir.as_ref())
            .map_err(crate::directory_reader::Error::from)?;
        Ok(latest.generation == self.factory.reader(&s).segment_infos.generation)
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
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicI64, AtomicUsize, Ordering};

    /// A reference that is a counter; each refresh bumps it when asked to.
    struct Counter {
        next: AtomicI64,
        change: std::sync::atomic::AtomicBool,
        closed: AtomicUsize,
        after: AtomicUsize,
    }

    impl Refresher<i64> for Arc<Counter> {
        fn refresh_if_needed(&self, current: &Arc<i64>) -> Result<Option<Arc<i64>>> {
            if self.change.swap(false, Ordering::SeqCst) {
                Ok(Some(Arc::new(
                    self.next.fetch_add(1, Ordering::SeqCst).max(**current + 1),
                )))
            } else {
                Ok(None)
            }
        }
        fn after_maybe_refresh(&self) -> Result<()> {
            self.after.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        fn after_close(&self) -> Result<()> {
            self.closed.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    fn counter() -> Arc<Counter> {
        Arc::new(Counter {
            next: AtomicI64::new(1),
            change: std::sync::atomic::AtomicBool::new(false),
            closed: AtomicUsize::new(0),
            after: AtomicUsize::new(0),
        })
    }

    struct Recorder(Mutex<Vec<String>>);

    impl RefreshListener for Recorder {
        fn before_refresh(&self) -> Result<()> {
            lock(&self.0).push("before".into());
            Ok(())
        }
        fn after_refresh(&self, did: bool) -> Result<()> {
            lock(&self.0).push(format!("after {did}"));
            Ok(())
        }
    }

    #[test]
    fn reference_manager_swaps_and_closes() {
        let c = counter();
        let m = ReferenceManager::new(Arc::new(0i64), Box::new(Arc::clone(&c)));
        let rec = Arc::new(Recorder(Mutex::new(Vec::new())));
        let l: Arc<dyn RefreshListener> = rec.clone();
        m.add_listener(Arc::clone(&l));
        let held = m.acquire().unwrap();
        assert_eq!(*held, 0);
        assert!(m.maybe_refresh().unwrap());
        assert_eq!(*m.acquire().unwrap(), 0, "nothing changed");
        c.change.store(true, Ordering::SeqCst);
        m.maybe_refresh_blocking().unwrap();
        assert_eq!(*m.acquire().unwrap(), 1);
        assert_eq!(*held, 0, "a held reference outlives the swap");
        m.release(held);
        assert_eq!(
            *lock(&rec.0),
            vec!["before", "after false", "before", "after true"]
        );
        assert_eq!(c.after.load(Ordering::SeqCst), 2);
        m.remove_listener(&l);
        m.maybe_refresh().unwrap();
        assert_eq!(lock(&rec.0).len(), 4, "removed");
        m.close().unwrap();
        m.close().unwrap();
        assert_eq!(c.closed.load(Ordering::SeqCst), 1);
        assert!(m.is_closed());
        assert!(matches!(m.acquire(), Err(Error::AlreadyClosed(_))));
        assert!(m.maybe_refresh().is_err());
        assert!(m.maybe_refresh_blocking().is_err());
    }

    #[test]
    fn a_refresh_returning_the_same_reference_is_refused() {
        struct Same;
        impl Refresher<i64> for Same {
            fn refresh_if_needed(&self, current: &Arc<i64>) -> Result<Option<Arc<i64>>> {
                Ok(Some(Arc::clone(current)))
            }
        }
        let m = ReferenceManager::new(Arc::new(0i64), Box::new(Same));
        assert!(matches!(m.maybe_refresh(), Err(Error::IllegalState(_))));
    }

    #[test]
    fn concurrent_maybe_refresh_skips_while_busy() {
        // `TestSearcherManager.testMaybeRefreshBlockingLock`'s intent: a
        // non-blocking refresh returns `false` while another refreshes.
        struct Slow(Arc<std::sync::Barrier>, Arc<std::sync::Barrier>);
        impl Refresher<i64> for Slow {
            fn refresh_if_needed(&self, _current: &Arc<i64>) -> Result<Option<Arc<i64>>> {
                self.0.wait();
                self.1.wait();
                Ok(None)
            }
        }
        let (a, b) = (
            Arc::new(std::sync::Barrier::new(2)),
            Arc::new(std::sync::Barrier::new(2)),
        );
        let m = Arc::new(ReferenceManager::new(
            Arc::new(0i64),
            Box::new(Slow(Arc::clone(&a), Arc::clone(&b))),
        ));
        let m2 = Arc::clone(&m);
        let t = std::thread::spawn(move || m2.maybe_refresh_blocking().unwrap());
        a.wait();
        assert!(!m.maybe_refresh().unwrap(), "busy: skipped");
        b.wait();
        t.join().unwrap();
    }

    #[derive(Debug)]
    struct V(i64);
    impl Versioned for V {
        fn version(&self) -> i64 {
            self.0
        }
    }

    #[test]
    fn lifetime_manager_records_acquires_prunes() {
        // `TestSearcherLifetimeManager`'s intent.
        let lm: SearcherLifetimeManager<V> = SearcherLifetimeManager::new();
        let (a, b, c) = (Arc::new(V(1)), Arc::new(V(2)), Arc::new(V(3)));
        assert_eq!(lm.record(&a).unwrap(), 1);
        assert_eq!(lm.record(&a).unwrap(), 1, "the same searcher again");
        assert!(
            lm.record(&Arc::new(V(1))).is_err(),
            "another of the same version"
        );
        lm.record(&b).unwrap();
        lm.record(&c).unwrap();
        assert_eq!(lm.len(), 3);
        assert_eq!(lm.acquire(2).unwrap().unwrap().0, 2);
        assert!(lm.acquire(9).unwrap().is_none());
        assert!(PruneByAge::new(-1.0).is_err());
        // Prune against a clock 10s on: the newest is age 0, the older two
        // age ~10s since their successors were recorded.
        lm.prune_at(
            &PruneByAge::new(5.0).unwrap(),
            Instant::now() + Duration::from_secs(10),
        );
        assert_eq!(lm.len(), 1);
        assert!(lm.acquire(3).unwrap().is_some(), "the newest survives");
        lm.prune(&PruneByAge::new(1000.0).unwrap());
        assert_eq!(lm.len(), 1);
        lm.release(a);
        lm.close();
        assert!(lm.is_empty());
        assert!(lm.acquire(3).is_err());
        assert!(lm.record(&b).is_err());
    }

    #[test]
    fn reopen_thread_waits_for_generations() {
        // `TestControlledRealTimeReopenThread`'s intent: a waiter on a
        // generation is released by a refresh that started after it.
        let c = counter();
        let m = Arc::new(ReferenceManager::new(
            Arc::new(0i64),
            Box::new(Arc::clone(&c)),
        ));
        let gen = Arc::new(AtomicI64::new(0));
        let g2 = Arc::clone(&gen);
        assert!(ControlledRealTimeReopenThread::start(
            Arc::clone(&m),
            Box::new(|| 0),
            Duration::from_millis(1),
            Duration::from_millis(2),
        )
        .is_err());
        let t = ControlledRealTimeReopenThread::start(
            Arc::clone(&m),
            Box::new(move || g2.load(Ordering::SeqCst)),
            Duration::from_secs(60),
            Duration::from_millis(5),
        )
        .unwrap();
        assert!(t.wait_for_generation(0, None), "already searchable");
        gen.store(5, Ordering::SeqCst);
        c.change.store(true, Ordering::SeqCst);
        assert!(t.wait_for_generation(5, Some(Duration::from_secs(30))));
        assert!(t.searching_gen() >= 5);
        assert_eq!(*m.acquire().unwrap(), 1, "the waiter forced a refresh");
        // A generation no refresh will reach times out.
        assert!(!t.wait_for_generation(99, Some(Duration::from_millis(30))));
        // Concurrent waiters.
        gen.store(7, Ordering::SeqCst);
        let t = Arc::new(t);
        let waiters: Vec<_> = (0..4)
            .map(|_| {
                let t = Arc::clone(&t);
                std::thread::spawn(move || t.wait_for_generation(7, Some(Duration::from_secs(30))))
            })
            .collect();
        for w in waiters {
            assert!(w.join().unwrap());
        }
        t.close().unwrap();
        assert_eq!(t.searching_gen(), i64::MAX);
        assert!(t.wait_for_generation(1000, None), "released after close");
    }

    #[test]
    fn live_field_values_prefer_recent_writes() {
        // `TestLiveFieldValues`' intent: a value added or deleted since the
        // last refresh wins over the searcher's until a refresh covers it.
        let c = counter();
        let m = Arc::new(ReferenceManager::new(
            Arc::new(0i64),
            Box::new(Arc::clone(&c)),
        ));
        let lookups = Arc::new(AtomicUsize::new(0));
        let l2 = Arc::clone(&lookups);
        let live: LiveFieldValues<i64, String> = LiveFieldValues::new(
            Arc::clone(&m),
            Box::new(move |s: &i64, id: &str| {
                l2.fetch_add(1, Ordering::SeqCst);
                Ok((id == "indexed").then(|| format!("searcher {s}")))
            }),
        );
        live.add("a", "1".into());
        live.add("b", "2".into());
        live.delete("b");
        assert_eq!(live.get("a").unwrap().as_deref(), Some("1"));
        assert_eq!(live.get("b").unwrap(), None, "deleted");
        assert_eq!(live.get("indexed").unwrap().as_deref(), Some("searcher 0"));
        assert_eq!(live.size(), 2);
        c.change.store(true, Ordering::SeqCst);
        m.maybe_refresh_blocking().unwrap();
        assert_eq!(live.size(), 0, "the refresh made them searchable");
        assert_eq!(live.get("a").unwrap(), None, "now from the searcher");
        assert_eq!(live.get("indexed").unwrap().as_deref(), Some("searcher 1"));
        assert!(lookups.load(Ordering::SeqCst) >= 3);
        live.close();
        live.add("z", "9".into());
        m.maybe_refresh_blocking().unwrap();
        assert_eq!(live.size(), 1, "no longer listening");
    }

    #[test]
    fn find_segments_file_retries_while_the_commit_generation_advances() {
        use lucene_store::byte_buffers_directory::ByteBuffersDirectory;
        use lucene_store::data_output::DataOutput;
        let dir = ByteBuffersDirectory::new();
        let commit = |name: &str| {
            let mut out = dir.create_output(name).unwrap();
            out.write_bytes(b"x");
            out.close().unwrap();
        };
        // No commit at all: `IndexNotFoundException`, the body never runs.
        let err = find_segments_file(&dir, || -> Result<()> { unreachable!() }).unwrap_err();
        assert!(matches!(err, Error::IllegalState(_)), "{err}");

        // A failure on an unchanged generation is reported, the first one.
        commit("segments_1");
        let mut calls = 0;
        let err = find_segments_file(&dir, || -> Result<()> {
            calls += 1;
            Err(Error::IllegalArgument(format!("try {calls}")))
        })
        .unwrap_err();
        assert_eq!(calls, 1);
        assert!(matches!(&err, Error::IllegalArgument(m) if m == "try 1"));

        // A commit landing while the body reads: the body runs again.
        let mut calls = 0;
        let got = find_segments_file(&dir, || {
            calls += 1;
            if calls == 1 {
                commit("segments_2");
                return Err(Error::IllegalArgument("deleted under us".into()));
            }
            Ok(calls)
        })
        .unwrap();
        assert_eq!(got, 2);
    }
}
