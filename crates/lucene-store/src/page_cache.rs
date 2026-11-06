//! A fixed-page cache for files read by range from a remote store, with an
//! optional tier on local disk (NVMe).
//!
//! A file is cut into pages of a fixed size. A read of any byte of a page
//! fetches and caches the whole page; a read that spans pages fetches every
//! page it touches, adjacent missing pages merged into one ranged request
//! ([`PageCacheConfig::max_request`] at most). The cache never holds part of
//! a page, so readers of nearby ranges always share pages.
//!
//! Pages are cut in the file's *stored* bytes. A plain file's pages start at
//! byte 0. A file stored in fixed units with a header (an encrypted file's
//! authenticated chunks, see [`PageDecoder`]) has pages of a whole number of
//! units, starting after the header, so no unit is ever split across two
//! fetches and every page decodes on its own. The disk tier holds stored
//! bytes (ciphertext, for an encrypted file); decoded bytes live only in
//! memory, in the reader's mapping ([`crate::CachedBlobDirectory`]).
//!
//! Concurrent misses on one page send one request: the first caller fetches
//! and the others wait for its result. Pages of pinned files are never
//! evicted from disk.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use crate::directory::Directory;
use crate::error::{Error, Result};
use crate::lock::lock_ignoring_poison;
use crate::page_disk::DiskTier;

/// How a stored file maps onto decoded bytes, unit by unit: stored unit `i`
/// is `stored_unit` bytes at `origin + i * stored_unit` (the last one
/// shorter) and decodes to `plain_unit` bytes at `i * plain_unit`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnitLayout {
    /// Stored bytes before unit 0 (a header).
    pub origin: u64,
    /// Stored bytes per unit.
    pub stored_unit: u64,
    /// Decoded bytes per unit.
    pub plain_unit: u64,
    /// The decoded file's length.
    pub plain_len: u64,
}

/// Decodes one stored file's units (decryption, for an encrypted file).
pub trait PageCodec: Send + Sync {
    /// The file's unit layout.
    fn layout(&self) -> UnitLayout;

    /// Decodes the consecutive stored units in `stored`, the first of them
    /// unit `first_unit`, into `out`, which holds exactly their decoded
    /// length. A unit that fails to decode (authenticate) is an error.
    fn decode(&self, first_unit: u64, stored: &[u8], out: &mut [u8]) -> Result<()>;
}

/// Builds a file's [`PageCodec`], reading what it needs (a header) through
/// [`Directory::read_at`]. An encrypting wrapper passes one to
/// [`Directory::open_decoded`].
pub trait PageDecoder: Send + Sync {
    /// The codec for `name`, whose stored length is `stored_len`.
    fn codec(&self, name: &str, stored_len: u64, dir: &dyn Directory)
        -> Result<Arc<dyn PageCodec>>;
}

/// The default page: 256 KiB, small enough that cold pages waste little
/// cache; requests are larger ([`PageCacheConfig::download_window`]).
pub const DEFAULT_PAGE_SIZE: usize = 256 << 10;
/// The download window before the tuner has measured the store: 1 MiB.
pub const DEFAULT_DOWNLOAD_WINDOW: usize = 1 << 20;
/// The default largest request: 8 MiB of adjacent pages.
pub const DEFAULT_MAX_REQUEST: usize = 8 << 20;
/// The default number of requests in flight for one read.
pub const DEFAULT_CONCURRENCY: usize = 16;

/// Extensions of the files read whole when a segment opens, or on every
/// query: metadata, the terms index (FST), norms, the stored-fields and BKD
/// indexes, live docs, compound-file tables. Their pages are pinned.
pub const DEFAULT_PINNED_EXTENSIONS: &[&str] = &[
    "si", "fnm", "cfe", "tmd", "tip", "psm", "fdm", "fdx", "dvm", "nvm", "nvd", "kdm", "kdi",
    "vem", "liv",
];

/// [`PageCache`]'s settings.
#[derive(Debug, Clone)]
pub struct PageCacheConfig {
    /// Bytes per page (plain files); a file stored in units gets the whole
    /// number of units closest to it, at least one.
    pub page_size: usize,
    /// The local disk tier's directory, or `None` for no disk tier. Its
    /// contents are deleted when the cache is created.
    pub disk_dir: Option<PathBuf>,
    /// The disk tier's budget in bytes; unpinned pages past it are evicted,
    /// least recently used first.
    pub disk_budget: u64,
    /// The aligned window a miss fetches the missing pages of (read-ahead,
    /// with a disk tier), in bytes. `None` (the default) tunes it from the
    /// store's measured latency and throughput: the bytes one request
    /// streams in its own first-byte wait, as a power of two between a page
    /// and `max_request`.
    pub download_window: Option<usize>,
    /// The largest ranged request, in bytes.
    pub max_request: usize,
    /// Requests in flight at once for one read.
    pub concurrency: usize,
    /// Extensions of files whose pages are pinned and that are read whole
    /// on open.
    pub pinned_extensions: Vec<String>,
}

impl Default for PageCacheConfig {
    fn default() -> Self {
        Self {
            page_size: DEFAULT_PAGE_SIZE,
            disk_dir: None,
            disk_budget: u64::MAX,
            download_window: None,
            max_request: DEFAULT_MAX_REQUEST,
            concurrency: DEFAULT_CONCURRENCY,
            pinned_extensions: DEFAULT_PINNED_EXTENSIONS
                .iter()
                .map(|e| (*e).to_string())
                .collect(),
        }
    }
}

/// Counters since the cache was created.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PageCacheStats {
    /// Pages served from the disk tier.
    pub disk_hits: u64,
    /// Pages fetched from the store.
    pub fetched_pages: u64,
    /// Ranged requests sent to the store.
    pub requests: u64,
    /// Bytes fetched from the store.
    pub fetched_bytes: u64,
    /// Pages a caller waited for while another caller fetched them.
    pub shared_fetches: u64,
    /// Pages fetched by read-ahead, nobody having asked for them yet.
    pub read_ahead_pages: u64,
    /// The download window in bytes ([`PageCacheConfig::download_window`]).
    pub download_window: u64,
    /// RAM held by the disk tier's page records.
    pub disk_metadata_bytes: u64,
}

#[derive(Default)]
struct Counters {
    disk_hits: AtomicU64,
    fetched_pages: AtomicU64,
    requests: AtomicU64,
    fetched_bytes: AtomicU64,
    shared_fetches: AtomicU64,
    read_ahead_pages: AtomicU64,
}

/// One file's pages in stored bytes, from its [`UnitLayout`].
#[derive(Debug, Clone)]
pub(crate) struct Geometry {
    /// The cache key of the file (store namespace, name, length, layout).
    pub(crate) key: Arc<str>,
    pub(crate) stored_len: u64,
    pub(crate) origin: u64,
    pub(crate) stored_page: u64,
    pub(crate) plain_page: u64,
    pub(crate) plain_len: u64,
    pub(crate) units_per_page: u64,
    pub(crate) pages: u64,
}

impl Geometry {
    /// A plain file: pages of `page_size` bytes from byte 0.
    pub(crate) fn plain(key: Arc<str>, len: u64, page_size: usize) -> Geometry {
        let page = (page_size as u64).max(1);
        Geometry {
            key,
            stored_len: len,
            origin: 0,
            stored_page: page,
            plain_page: page,
            plain_len: len,
            units_per_page: 1,
            pages: len.div_ceil(page),
        }
    }

    /// A file stored in units: pages of the whole number of units nearest
    /// `page_size` (at least one).
    pub(crate) fn units(
        key: Arc<str>,
        stored_len: u64,
        layout: UnitLayout,
        page_size: usize,
        name: &str,
    ) -> Result<Geometry> {
        let bad = |what: &str| Error::Corrupted(format!("{name}: {what}"));
        if layout.stored_unit == 0 || layout.plain_unit == 0 || layout.origin > stored_len {
            return Err(bad("invalid unit layout"));
        }
        let units_per_page = (page_size as u64)
            .checked_div(layout.plain_unit)
            .unwrap_or(1)
            .max(1);
        let stored_page = units_per_page
            .checked_mul(layout.stored_unit)
            .ok_or_else(|| bad("page overflows"))?;
        let plain_page = units_per_page
            .checked_mul(layout.plain_unit)
            .ok_or_else(|| bad("page overflows"))?;
        let body = stored_len.saturating_sub(layout.origin);
        let pages = body.div_ceil(stored_page).max(1);
        Ok(Geometry {
            key,
            stored_len,
            origin: layout.origin,
            stored_page,
            plain_page,
            plain_len: layout.plain_len,
            units_per_page,
            pages,
        })
    }

    /// Stored bytes `start..end` of page `page`.
    pub(crate) fn stored_range(&self, page: u64) -> (u64, u64) {
        let start = self
            .origin
            .saturating_add(page.saturating_mul(self.stored_page))
            .min(self.stored_len);
        let end = start.saturating_add(self.stored_page).min(self.stored_len);
        (start, end)
    }

    /// Decoded bytes `start..end` of page `page`.
    pub(crate) fn plain_range(&self, page: u64) -> (u64, u64) {
        let start = page.saturating_mul(self.plain_page).min(self.plain_len);
        let end = start.saturating_add(self.plain_page).min(self.plain_len);
        (start, end)
    }
}

/// A fetch error as every caller waiting on the page gets it: its kind
/// (missing file, corruption, other I/O) and message.
#[derive(Debug, Clone)]
struct FetchError {
    corrupted: bool,
    kind: std::io::ErrorKind,
    message: String,
}

impl FetchError {
    fn of(e: &Error) -> FetchError {
        match e {
            Error::Corrupted(m) => FetchError {
                corrupted: true,
                kind: std::io::ErrorKind::InvalidData,
                message: m.clone(),
            },
            Error::Io(io) => FetchError {
                corrupted: false,
                kind: io.kind(),
                message: io.to_string(),
            },
            other => FetchError {
                corrupted: false,
                kind: std::io::ErrorKind::Other,
                message: other.to_string(),
            },
        }
    }

    fn to_error(&self) -> Error {
        if self.corrupted {
            Error::Corrupted(self.message.clone())
        } else {
            Error::Io(std::io::Error::new(self.kind, self.message.clone()))
        }
    }
}

type Fetched = std::result::Result<Arc<[u8]>, FetchError>;

/// A page another caller is fetching: its result once fetched.
#[derive(Default)]
struct Slot {
    result: Mutex<Option<Fetched>>,
    ready: Condvar,
}

impl Slot {
    fn fulfill(&self, result: Fetched) {
        *lock_ignoring_poison(&self.result) = Some(result);
        self.ready.notify_all();
    }

    fn wait(&self) -> Fetched {
        let mut guard = lock_ignoring_poison(&self.result);
        loop {
            if let Some(r) = guard.as_ref() {
                return r.clone();
            }
            guard = self.ready.wait(guard).unwrap_or_else(|p| p.into_inner());
        }
    }
}

type PageKey = (Arc<str>, u64);

/// The fixed-page cache shared by every [`crate::CachedBlobDirectory`]
/// given it.
pub struct PageCache {
    config: PageCacheConfig,
    disk: Option<DiskTier>,
    inflight: Mutex<HashMap<PageKey, Arc<Slot>>>,
    counters: Counters,
    tuner: Tuner,
}

/// Measures ranged requests to size the download window: request time is
/// about `latency + bytes / throughput`, so small requests give the latency
/// and large ones the throughput.
#[derive(Default)]
struct Tuner {
    state: Mutex<TunerState>,
}

#[derive(Default, Clone, Copy)]
struct TunerState {
    /// Mean seconds of requests of at most two pages.
    small_secs: Option<f64>,
    small_bytes: f64,
    /// Mean seconds and bytes of requests of four pages or more.
    large_secs: Option<f64>,
    large_bytes: f64,
}

impl Tuner {
    /// Weight of a new sample in the running means.
    const ALPHA: f64 = 0.2;

    fn record(&self, bytes: usize, took: std::time::Duration, stored_page: u64) {
        let secs = took.as_secs_f64();
        let bytes_f = bytes as f64;
        let page = stored_page as f64;
        let mut st = lock_ignoring_poison(&self.state);
        let mix =
            |old: Option<f64>, new: f64| Some(old.map_or(new, |o| o + Self::ALPHA * (new - o)));
        if bytes_f <= 2.0 * page {
            st.small_bytes = st.small_secs.map_or(bytes_f, |_| {
                st.small_bytes + Self::ALPHA * (bytes_f - st.small_bytes)
            });
            st.small_secs = mix(st.small_secs, secs);
        } else if bytes_f >= 4.0 * page {
            st.large_bytes = st.large_secs.map_or(bytes_f, |_| {
                st.large_bytes + Self::ALPHA * (bytes_f - st.large_bytes)
            });
            st.large_secs = mix(st.large_secs, secs);
        }
    }

    fn window(&self, page: usize, max: usize) -> usize {
        let st = *lock_ignoring_poison(&self.state);
        let lo = page.max(1);
        let hi = max.max(lo);
        let (Some(small), Some(large)) = (st.small_secs, st.large_secs) else {
            return DEFAULT_DOWNLOAD_WINDOW.clamp(lo, hi);
        };
        let extra_secs = large - small;
        let extra_bytes = st.large_bytes - st.small_bytes;
        if extra_secs <= 0.0 || extra_bytes <= 0.0 {
            // Larger requests cost no more: latency dominates.
            return hi;
        }
        let throughput = extra_bytes / extra_secs;
        // A small request's time less its own transfer.
        let latency = (small - st.small_bytes / throughput).max(0.0);
        let ideal = (latency * throughput).clamp(lo as f64, hi as f64) as usize;
        ideal.next_power_of_two().clamp(lo, hi)
    }
}

impl std::fmt::Debug for PageCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PageCache")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl PageCache {
    /// A cache with `config`, emptying its disk tier's directory.
    pub fn new(config: PageCacheConfig) -> Result<PageCache> {
        if config.page_size == 0 || config.max_request == 0 {
            return Err(Error::IllegalArgument(
                "page_size and max_request must be positive".into(),
            ));
        }
        let disk = match &config.disk_dir {
            Some(dir) => Some(DiskTier::open(dir, config.disk_budget, config.page_size)?),
            None => None,
        };
        Ok(PageCache {
            config,
            disk,
            inflight: Mutex::new(HashMap::new()),
            counters: Counters::default(),
            tuner: Tuner::default(),
        })
    }

    /// The settings.
    pub fn config(&self) -> &PageCacheConfig {
        &self.config
    }

    /// The counters so far.
    pub fn stats(&self) -> PageCacheStats {
        let c = &self.counters;
        PageCacheStats {
            disk_hits: c.disk_hits.load(Ordering::Relaxed),
            fetched_pages: c.fetched_pages.load(Ordering::Relaxed),
            requests: c.requests.load(Ordering::Relaxed),
            fetched_bytes: c.fetched_bytes.load(Ordering::Relaxed),
            shared_fetches: c.shared_fetches.load(Ordering::Relaxed),
            read_ahead_pages: c.read_ahead_pages.load(Ordering::Relaxed),
            download_window: self.download_window() as u64,
            disk_metadata_bytes: self.disk.as_ref().map_or(0, |d| d.metadata_bytes() as u64),
        }
    }

    /// Whether the cache has a disk tier ([`PageCacheConfig::disk_dir`]).
    pub fn has_disk(&self) -> bool {
        self.disk.is_some()
    }

    /// Bytes held by the disk tier now.
    pub fn disk_bytes(&self) -> u64 {
        self.disk.as_ref().map_or(0, DiskTier::used)
    }

    /// Whether files named `name` are pinned (read whole, never evicted).
    pub fn is_pinned_name(&self, name: &str) -> bool {
        if name.starts_with("segments") {
            return true;
        }
        let ext = name.rsplit_once('.').map_or("", |(_, e)| e);
        self.config.pinned_extensions.iter().any(|p| p == ext)
    }

    /// Pins every page of the file `key`, now and as they are cached.
    pub(crate) fn pin(&self, key: &str) {
        if let Some(d) = &self.disk {
            d.pin(key);
        }
    }

    /// Drops every cached page of files whose key starts with `prefix`.
    pub(crate) fn forget_prefix(&self, prefix: &str) {
        if let Some(d) = &self.disk {
            d.forget_prefix(prefix);
        }
    }

    /// The stored bytes of `pages` (sorted, distinct) of the file `geom`,
    /// in order: from the disk tier where cached, else fetched through
    /// `fetch(offset, len)` (a ranged read of the stored file) in merged,
    /// parallel requests. With a disk tier, a miss also fetches the other
    /// missing pages of its aligned download window, which are cached but
    /// not returned (read-ahead).
    pub(crate) fn stored_pages(
        &self,
        geom: &Geometry,
        pages: &[u64],
        fetch: &(dyn Fn(u64, usize) -> Result<Vec<u8>> + Sync),
    ) -> Result<Vec<Arc<[u8]>>> {
        let mut out: Vec<Option<Arc<[u8]>>> = vec![None; pages.len()];
        // Pages this call fetches: (page, where it goes in `out`, slot).
        let mut mine: Vec<(u64, Option<usize>, Arc<Slot>)> = Vec::new();
        let mut theirs: Vec<(usize, Arc<Slot>)> = Vec::new();
        for (i, &page) in pages.iter().enumerate() {
            if let Some(bytes) = self.disk.as_ref().and_then(|d| d.get(&geom.key, page)) {
                self.counters.disk_hits.fetch_add(1, Ordering::Relaxed);
                out[i] = Some(bytes);
                continue;
            }
            let key = (Arc::clone(&geom.key), page);
            let mut inflight = lock_ignoring_poison(&self.inflight);
            if let Some(slot) = inflight.get(&key) {
                theirs.push((i, Arc::clone(slot)));
            } else {
                let slot = Arc::new(Slot::default());
                inflight.insert(key, Arc::clone(&slot));
                mine.push((page, Some(i), slot));
            }
        }
        self.claim_read_ahead(geom, pages, &mut mine);
        // A panicking fetch must still release the pages it claimed, or
        // every later reader of them would wait forever.
        let fetched = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.fetch_runs(geom, &mine, fetch)
        }))
        .unwrap_or_else(|_| {
            let e = FetchError::of(&Error::Io(std::io::Error::other("page fetch panicked")));
            mine.iter().map(|_| Err(e.clone())).collect()
        });
        let mut first_error = None;
        for ((page, at, slot), result) in mine.iter().zip(fetched) {
            match result {
                Ok(bytes) => {
                    slot.fulfill(Ok(Arc::clone(&bytes)));
                    if let Some(i) = at {
                        out[*i] = Some(bytes);
                    }
                }
                Err(e) => {
                    slot.fulfill(Err(e.clone()));
                    if at.is_some() {
                        first_error.get_or_insert(e);
                    }
                }
            }
            lock_ignoring_poison(&self.inflight).remove(&(Arc::clone(&geom.key), *page));
        }
        for (i, slot) in theirs {
            self.counters.shared_fetches.fetch_add(1, Ordering::Relaxed);
            match slot.wait() {
                Ok(bytes) => out[i] = Some(bytes),
                Err(e) => {
                    first_error.get_or_insert(e);
                }
            }
        }
        if let Some(e) = first_error {
            return Err(e.to_error());
        }
        out.into_iter()
            .map(|b| b.ok_or_else(|| Error::Io(std::io::Error::other("page not fetched"))))
            .collect()
    }

    /// The download window now, in bytes: the configured one, or the
    /// tuned one.
    pub fn download_window(&self) -> usize {
        match self.config.download_window {
            Some(w) => w,
            None => self
                .tuner
                .window(self.config.page_size, self.config.max_request),
        }
    }

    /// Adds to `mine` (kept sorted by page) the missing, unclaimed pages of
    /// each claimed page's aligned download window. Only with a disk tier:
    /// without one, pages nobody asked for would be dropped.
    fn claim_read_ahead(
        &self,
        geom: &Geometry,
        pages: &[u64],
        mine: &mut Vec<(u64, Option<usize>, Arc<Slot>)>,
    ) {
        let Some(disk) = &self.disk else {
            return;
        };
        if mine.is_empty() {
            return;
        }
        let per_window = (self.download_window() as u64)
            .checked_div(geom.stored_page)
            .unwrap_or(1)
            .max(1);
        if per_window <= 1 {
            return;
        }
        let mut windows: Vec<u64> = mine
            .iter()
            .filter_map(|(p, _, _)| p.checked_div(per_window))
            .collect();
        windows.dedup();
        let mut extra = Vec::new();
        for w in windows {
            let start = w.saturating_mul(per_window);
            let end = start.saturating_add(per_window).min(geom.pages);
            for page in start..end {
                if pages.binary_search(&page).is_ok() || disk.contains(&geom.key, page) {
                    continue;
                }
                let key = (Arc::clone(&geom.key), page);
                let mut inflight = lock_ignoring_poison(&self.inflight);
                if inflight.contains_key(&key) {
                    continue;
                }
                let slot = Arc::new(Slot::default());
                inflight.insert(key, Arc::clone(&slot));
                extra.push((page, None, slot));
            }
        }
        if !extra.is_empty() {
            self.counters
                .read_ahead_pages
                .fetch_add(extra.len() as u64, Ordering::Relaxed);
            mine.extend(extra);
            mine.sort_by_key(|(p, _, _)| *p);
        }
    }

    /// Fetches the claimed pages (sorted), adjacent ones merged up to
    /// `max_request` bytes, `concurrency` requests at a time. One result per
    /// claimed page, in order.
    fn fetch_runs(
        &self,
        geom: &Geometry,
        mine: &[(u64, Option<usize>, Arc<Slot>)],
        fetch: &(dyn Fn(u64, usize) -> Result<Vec<u8>> + Sync),
    ) -> Vec<Fetched> {
        // Runs of claimed pages: (index into `mine` of the first, count).
        let mut runs: Vec<(usize, usize)> = Vec::new();
        let max_pages = (self.config.max_request as u64)
            .checked_div(geom.stored_page)
            .unwrap_or(1)
            .max(1);
        for (j, (page, _, _)) in mine.iter().enumerate() {
            if let Some((start, count)) = runs.last_mut() {
                let prev = mine[start.saturating_add(count.saturating_sub(1))].0;
                if prev.checked_add(1) == Some(*page) && (*count as u64) < max_pages {
                    *count = count.saturating_add(1);
                    continue;
                }
            }
            runs.push((j, 1));
        }
        let fetch_run = |&(start, count): &(usize, usize)| -> Vec<Fetched> {
            let first = mine[start].0;
            let last = mine[start.saturating_add(count).saturating_sub(1)].0;
            let (from, _) = geom.stored_range(first);
            let (_, to) = geom.stored_range(last);
            let len = usize::try_from(to.saturating_sub(from)).unwrap_or(usize::MAX);
            self.counters.requests.fetch_add(1, Ordering::Relaxed);
            let started = std::time::Instant::now();
            let fetched = fetch(from, len).and_then(|bytes| {
                if bytes.len() == len {
                    Ok(bytes)
                } else {
                    Err(Error::Corrupted(format!(
                        "short read: {} of {len} bytes at {from}",
                        bytes.len()
                    )))
                }
            });
            match fetched {
                Ok(bytes) => {
                    self.tuner
                        .record(bytes.len(), started.elapsed(), geom.stored_page);
                    self.counters
                        .fetched_bytes
                        .fetch_add(bytes.len() as u64, Ordering::Relaxed);
                    self.counters
                        .fetched_pages
                        .fetch_add(count as u64, Ordering::Relaxed);
                    (0..count)
                        .map(|k| {
                            let page = mine[start.saturating_add(k)].0;
                            let (a, b) = geom.stored_range(page);
                            let a = usize::try_from(a.saturating_sub(from)).unwrap_or(0);
                            let b = usize::try_from(b.saturating_sub(from)).unwrap_or(0);
                            let page_bytes =
                                Arc::<[u8]>::from(&bytes[a.min(b)..b.min(bytes.len())]);
                            // Written here, on the run's own thread, so the
                            // disk writes of parallel runs overlap too.
                            if let Some(d) = &self.disk {
                                d.put(&geom.key, page, &page_bytes);
                            }
                            Ok(page_bytes)
                        })
                        .collect()
                }
                Err(e) => {
                    let e = FetchError::of(&e);
                    (0..count).map(|_| Err(e.clone())).collect()
                }
            }
        };
        let per_run: Vec<Vec<_>> = if runs.len() <= 1 {
            runs.iter().map(fetch_run).collect()
        } else {
            let mut results: Vec<Vec<_>> = Vec::with_capacity(runs.len());
            for batch in runs.chunks(self.config.concurrency.max(1)) {
                let batch_results: Vec<Vec<_>> = std::thread::scope(|s| {
                    let handles: Vec<_> =
                        batch.iter().map(|run| s.spawn(|| fetch_run(run))).collect();
                    handles
                        .into_iter()
                        .zip(batch)
                        .map(|(h, run)| {
                            h.join().unwrap_or_else(|_| {
                                let e = FetchError::of(&Error::Io(std::io::Error::other(
                                    "page fetch panicked",
                                )));
                                (0..run.1).map(|_| Err(e.clone())).collect()
                            })
                        })
                        .collect()
                });
                results.extend(batch_results);
            }
            results
        };
        per_run.into_iter().flatten().collect()
    }
}

#[cfg(test)]
mod tests {
    // Test fixtures' own arithmetic -- see `docs/arithmetic-gate.md`'s "Test code".
    #![allow(clippy::arithmetic_side_effects)]
    use super::*;
    use std::sync::atomic::AtomicUsize;

    fn bytes(len: usize) -> Vec<u8> {
        (0..len).map(|i| (i * 31 % 251) as u8).collect()
    }

    fn reader(
        data: Vec<u8>,
        calls: Arc<AtomicUsize>,
    ) -> impl Fn(u64, usize) -> Result<Vec<u8>> + Sync {
        move |off, len| {
            calls.fetch_add(1, Ordering::Relaxed);
            let s = (off as usize).min(data.len());
            let e = (s + len).min(data.len());
            Ok(data[s..e].to_vec())
        }
    }

    #[test]
    fn plain_geometry_cuts_whole_pages() {
        let g = Geometry::plain(Arc::from("k"), 2500, 1000);
        assert_eq!(g.pages, 3);
        assert_eq!(g.stored_range(0), (0, 1000));
        assert_eq!(g.stored_range(2), (2000, 2500));
        assert_eq!(g.plain_range(2), (2000, 2500));
        assert_eq!(g.stored_range(9), (2500, 2500));
    }

    #[test]
    fn unit_geometry_aligns_pages_to_whole_units_after_the_origin() {
        let layout = UnitLayout {
            origin: 10,
            stored_unit: 128,
            plain_unit: 100,
            plain_len: 450,
        };
        // 5 units: 4 full of 128 stored bytes and one of 50 + 28.
        let stored = 10 + 4 * 128 + 78;
        let g = Geometry::units(Arc::from("k"), stored, layout, 250, "f").unwrap();
        assert_eq!(g.units_per_page, 2);
        assert_eq!(g.pages, 3);
        assert_eq!(g.stored_range(0), (10, 266));
        assert_eq!(g.stored_range(2), (522, stored));
        assert_eq!(g.plain_range(2), (400, 450));
        // A page smaller than a unit still holds one whole unit.
        let g = Geometry::units(Arc::from("k"), stored, layout, 1, "f").unwrap();
        assert_eq!(g.units_per_page, 1);
        let bad = UnitLayout {
            stored_unit: 0,
            ..layout
        };
        assert!(Geometry::units(Arc::from("k"), stored, bad, 250, "f").is_err());
        let bad = UnitLayout {
            origin: stored + 1,
            ..layout
        };
        assert!(Geometry::units(Arc::from("k"), stored, bad, 250, "f").is_err());
        let huge = UnitLayout {
            stored_unit: u64::MAX,
            ..layout
        };
        assert!(Geometry::units(Arc::from("k"), stored, huge, 250, "f").is_err());
    }

    #[test]
    fn adjacent_misses_merge_into_one_request_and_disk_hits_skip_the_store() {
        let tmp = lucene_util::test_support::TempDir::new("page-cache-merge");
        let cache = PageCache::new(PageCacheConfig {
            page_size: 100,
            max_request: 300,
            download_window: Some(100),
            disk_dir: Some(tmp.path().join("cache")),
            ..PageCacheConfig::default()
        })
        .unwrap();
        let data = bytes(1000);
        let calls = Arc::new(AtomicUsize::new(0));
        let fetch = reader(data.clone(), Arc::clone(&calls));
        let g = Geometry::plain(Arc::from("ns\0f"), 1000, 100);
        // Pages 0..=4 and 7: runs [0,1,2] [3,4] (max 300 bytes) and [7].
        let got = cache.stored_pages(&g, &[0, 1, 2, 3, 4, 7], &fetch).unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), 3);
        assert_eq!(&got[5][..], &data[700..800]);
        assert_eq!(&got[3][..], &data[300..400]);
        let again = cache.stored_pages(&g, &[2, 7], &fetch).unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), 3, "served from disk");
        assert_eq!(&again[0][..], &data[200..300]);
        let s = cache.stats();
        assert_eq!((s.requests, s.fetched_pages, s.disk_hits), (3, 6, 2));
        assert_eq!(s.fetched_bytes, 600);
        assert_eq!(cache.disk_bytes(), 600);
        cache.forget_prefix("ns\0");
        assert_eq!(cache.disk_bytes(), 0);
        cache.stored_pages(&g, &[2], &fetch).unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), 4);
    }

    #[test]
    fn the_disk_tier_evicts_unpinned_pages_past_its_budget() {
        let tmp = lucene_util::test_support::TempDir::new("page-cache-evict");
        let cache = PageCache::new(PageCacheConfig {
            page_size: 100,
            download_window: Some(100),
            disk_dir: Some(tmp.path().join("cache")),
            // Three slots (a 100-byte page takes an 8 KiB slot).
            disk_budget: 3 * 8192,
            ..PageCacheConfig::default()
        })
        .unwrap();
        let data = bytes(1000);
        let calls = Arc::new(AtomicUsize::new(0));
        let fetch = reader(data, Arc::clone(&calls));
        let pinned = Geometry::plain(Arc::from("p"), 1000, 100);
        let other = Geometry::plain(Arc::from("o"), 1000, 100);
        cache.stored_pages(&pinned, &[0], &fetch).unwrap();
        cache.pin("p");
        cache.stored_pages(&pinned, &[1], &fetch).unwrap();
        for page in 0..5 {
            cache.stored_pages(&other, &[page], &fetch).unwrap();
        }
        // Budget 300: the two pinned pages and the newest other page stay.
        assert_eq!(cache.disk_bytes(), 300);
        let before = calls.load(Ordering::Relaxed);
        cache.stored_pages(&pinned, &[0, 1], &fetch).unwrap();
        cache.stored_pages(&other, &[4], &fetch).unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), before);
        cache.stored_pages(&other, &[0], &fetch).unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), before + 1);
    }

    #[test]
    fn concurrent_misses_on_one_page_send_one_request() {
        let cache = PageCache::new(PageCacheConfig {
            page_size: 100,
            ..PageCacheConfig::default()
        })
        .unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let data = bytes(1000);
        let gate = Arc::new(std::sync::Barrier::new(2));
        let g = Geometry::plain(Arc::from("f"), 1000, 100);
        let slow = {
            let calls = Arc::clone(&calls);
            let gate = Arc::clone(&gate);
            move |off: u64, len: usize| {
                if calls.fetch_add(1, Ordering::Relaxed) == 0 {
                    gate.wait();
                    std::thread::sleep(std::time::Duration::from_millis(50));
                }
                Ok(data[off as usize..off as usize + len].to_vec())
            }
        };
        std::thread::scope(|s| {
            let a = s.spawn(|| cache.stored_pages(&g, &[3], &slow).unwrap());
            gate.wait();
            let b = cache.stored_pages(&g, &[3], &slow).unwrap();
            assert_eq!(a.join().unwrap()[0], b[0]);
        });
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(cache.stats().shared_fetches, 1);
    }

    #[test]
    fn failed_and_short_fetches_are_errors_for_every_waiter() {
        let cache = PageCache::new(PageCacheConfig {
            page_size: 100,
            max_request: 100,
            ..PageCacheConfig::default()
        })
        .unwrap();
        let g = Geometry::plain(Arc::from("f"), 1000, 100);
        let failing = |_: u64, _: usize| -> Result<Vec<u8>> {
            Err(Error::Io(std::io::Error::other("store down")))
        };
        let e = cache.stored_pages(&g, &[0, 5], &failing).unwrap_err();
        assert!(e.to_string().contains("store down"), "{e}");
        let short = |_: u64, len: usize| -> Result<Vec<u8>> { Ok(vec![0; len / 2]) };
        let e = cache.stored_pages(&g, &[1], &short).unwrap_err();
        assert!(e.to_string().contains("short read"), "{e}");
        // Nothing is left claimed: a later good fetch succeeds.
        let ok = |_: u64, len: usize| -> Result<Vec<u8>> { Ok(vec![7; len]) };
        assert_eq!(cache.stored_pages(&g, &[0, 1], &ok).unwrap().len(), 2);
    }

    #[test]
    fn errors_keep_their_kind_and_a_panicking_fetch_releases_its_pages() {
        let cache = PageCache::new(PageCacheConfig {
            page_size: 100,
            ..PageCacheConfig::default()
        })
        .unwrap();
        let g = Geometry::plain(Arc::from("f"), 1000, 100);
        let gone = |_: u64, _: usize| -> Result<Vec<u8>> { Err(Error::no_such_file("f")) };
        assert!(cache
            .stored_pages(&g, &[0], &gone)
            .unwrap_err()
            .is_no_such_file());
        let short = |_: u64, _: usize| -> Result<Vec<u8>> { Ok(Vec::new()) };
        let e = cache.stored_pages(&g, &[0], &short).unwrap_err();
        assert!(matches!(e, Error::Corrupted(_)), "{e}");
        let other =
            |_: u64, _: usize| -> Result<Vec<u8>> { Err(Error::IllegalArgument("x".into())) };
        assert!(cache.stored_pages(&g, &[0], &other).is_err());
        let panics = |_: u64, _: usize| -> Result<Vec<u8>> { panic!("sdk bug") };
        assert!(cache.stored_pages(&g, &[0], &panics).is_err());
        assert!(cache.stored_pages(&g, &[1, 3], &panics).is_err());
        let ok = |_: u64, len: usize| -> Result<Vec<u8>> { Ok(vec![1; len]) };
        assert_eq!(cache.stored_pages(&g, &[0, 1, 3], &ok).unwrap().len(), 3);
    }

    #[test]
    fn config_is_checked_and_pinned_names_follow_extensions() {
        assert!(PageCache::new(PageCacheConfig {
            page_size: 0,
            ..PageCacheConfig::default()
        })
        .is_err());
        let cache = PageCache::new(PageCacheConfig::default()).unwrap();
        assert!(cache.is_pinned_name("_0.tip"));
        assert!(cache.is_pinned_name("segments_3"));
        assert!(!cache.is_pinned_name("_0.doc"));
        assert!(!cache.is_pinned_name("noext"));
        assert_eq!(cache.disk_bytes(), 0);
        cache.pin("x");
        cache.forget_prefix("x");
        assert!(format!("{cache:?}").contains("PageCache"));
    }

    #[test]
    fn a_miss_reads_ahead_its_download_window_in_one_request() {
        let tmp = lucene_util::test_support::TempDir::new("page-cache-ahead");
        let cache = PageCache::new(PageCacheConfig {
            page_size: 100,
            download_window: Some(400),
            disk_dir: Some(tmp.path().join("cache")),
            ..PageCacheConfig::default()
        })
        .unwrap();
        let data = bytes(1000);
        let calls = Arc::new(AtomicUsize::new(0));
        let fetch = reader(data.clone(), Arc::clone(&calls));
        let g = Geometry::plain(Arc::from("f"), 1000, 100);
        // Page 5 is in window 4..8: one request for all four pages.
        let got = cache.stored_pages(&g, &[5], &fetch).unwrap();
        assert_eq!(&got[0][..], &data[500..600]);
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(cache.stats().read_ahead_pages, 3);
        assert_eq!(cache.stats().fetched_bytes, 400);
        let again = cache.stored_pages(&g, &[4, 6, 7], &fetch).unwrap();
        assert_eq!(&again[2][..], &data[700..800]);
        assert_eq!(calls.load(Ordering::Relaxed), 1, "read ahead");
        // The last window is cut at the file's end; cached pages are skipped.
        cache.stored_pages(&g, &[9], &fetch).unwrap();
        assert_eq!(cache.stats().fetched_bytes, 600);
        assert_eq!(cache.stats().download_window, 400);
        // Without a disk tier there is no read-ahead.
        let bare = PageCache::new(PageCacheConfig {
            page_size: 100,
            download_window: Some(400),
            ..PageCacheConfig::default()
        })
        .unwrap();
        bare.stored_pages(&g, &[5], &fetch).unwrap();
        assert_eq!(bare.stats().fetched_bytes, 100);
    }

    #[test]
    fn the_tuned_window_is_latency_times_throughput() {
        let t = Tuner::default();
        let page = 256 << 10;
        let max = 8 << 20;
        // Unmeasured: the default.
        assert_eq!(t.window(page, max), DEFAULT_DOWNLOAD_WINDOW);
        // 30 ms latency, 100 MB/s: 3 MB, so 4 MiB.
        let ms = |m: f64| std::time::Duration::from_secs_f64(m / 1000.0);
        for _ in 0..20 {
            t.record(page, ms(30.0 + 2.62), page as u64);
            t.record(8 * page, ms(30.0 + 20.97), page as u64);
            t.record(3 * page, ms(1.0), page as u64);
        }
        assert_eq!(t.window(page, max), 4 << 20);
        // 0.5 ms latency, 1 GB/s (local disk): 0.5 MB, so 512 KiB.
        let t = Tuner::default();
        for _ in 0..20 {
            t.record(page, ms(0.5 + 0.262), page as u64);
            t.record(8 * page, ms(0.5 + 2.097), page as u64);
        }
        assert_eq!(t.window(page, max), 512 << 10);
        // Large requests no slower: the largest window.
        let t = Tuner::default();
        t.record(page, ms(5.0), page as u64);
        t.record(8 * page, ms(5.0), page as u64);
        assert_eq!(t.window(page, max), max);
        // Clamped to a page at least.
        let t = Tuner::default();
        t.record(page, ms(0.001), page as u64);
        t.record(8 * page, ms(100.0), page as u64);
        assert_eq!(t.window(page, max), page);
    }
}
