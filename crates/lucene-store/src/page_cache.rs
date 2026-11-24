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
use crate::fetch_budget::{FetchBudget, FetchPermit};
use crate::lock::lock_ignoring_poison;
use crate::page_disk::DiskTier;
pub use crate::page_disk::Resident as ResidentPage;

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
/// The default most ranged requests in flight at once over every reader
/// of one cache: 256, the I/O threads a node runs at most.
pub const DEFAULT_MAX_INFLIGHT_REQUESTS: usize = 256;
/// The default most bytes of ranged requests in flight at once over every
/// reader of one cache: 512 MiB, four reads at [`DEFAULT_CONCURRENCY`]
/// requests of [`DEFAULT_MAX_REQUEST`] bytes.
pub const DEFAULT_MAX_INFLIGHT_BYTES: usize = 512 << 20;

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
    /// The most ranged requests in flight at once over every reader of
    /// this cache: a read past it waits for one to finish, so a node's
    /// fetch threads and open connections stay bounded however many
    /// queries miss at once. At least 1.
    pub max_inflight_requests: usize,
    /// The most bytes of ranged requests in flight at once over every
    /// reader of this cache, each request counted at its length from
    /// before it is sent until its pages are cut from its buffer: the
    /// heap fetching may hold. At least [`Self::max_request`].
    pub max_inflight_bytes: usize,
    /// Extensions of files whose pages are pinned and that are read whole
    /// on open.
    pub pinned_extensions: Vec<String>,
    /// A measured profile of the store, used to size the download window
    /// when [`Self::download_window`] is `None`. With it the window is
    /// `latency * throughput` of the profile, a power of two between a page
    /// and `max_request`, and the same on every run: the live tuner (which
    /// follows request timings, so the window differs run to run) is not
    /// consulted. `None` (the default) keeps the live tuner.
    pub store_profile: Option<StoreProfile>,
    /// Whether a file may be demand-filled through `userfaultfd` where the
    /// platform supports it. `false` forces the read-through fallback of
    /// [`Self::spill_min_bytes`]: how a deployment where the faulting
    /// mapping is unwanted runs, and how the fallback is tested and
    /// benchmarked on a machine that has `userfaultfd`.
    pub demand_paging: bool,
    /// The smallest undecoded file, in bytes, that is *not* read whole into
    /// the heap when it is opened without demand paging (no `userfaultfd`,
    /// a page size that is not a multiple of 4 KiB). Pinned files stay in
    /// the heap, as before. It is streamed in chunks of at most
    /// [`SPILL_CHUNK_BYTES`] into an unlinked file under
    /// [`Self::disk_dir`] and mapped read-only: the heap holds one chunk,
    /// and the file's bytes are page cache the kernel may reclaim. `None`
    /// reads every such file whole into the heap, as before; so does a cache
    /// without a disk tier, and a file read through a decoder (its plaintext
    /// is never written to disk).
    pub spill_min_bytes: Option<u64>,
    /// The most bytes of such mapped files that may be open at once: a file
    /// that would exceed it fails to open rather than fill the disk (an I/O error; a caller may retry once other
    /// files close).
    pub spill_budget: u64,
    /// The most bytes of files held whole on the heap at once: pinned files,
    /// files of one page, and (without demand paging) files below
    /// [`Self::spill_min_bytes`] or read without a disk tier. A file that
    /// would pass it fails to open (an I/O error; a caller may retry once
    /// other files close), so the steady-state heap held by such files is bounded by
    /// this and not by how many readers are open (an open also holds its
    /// pages and the assembled file, up to about twice its length, while it
    /// reads). Two threads opening one uncached file at once each count it
    /// until one input drops. Shared opens of one file
    /// count once. `u64::MAX` (the default) is unbounded, as before, and
    /// then such files are not counted ([`PageCacheStats::heap_open_bytes`]
    /// stays 0) and read exactly as they always were; a bounded budget
    /// serves them through one indirect call per access.
    pub heap_budget: u64,
}

/// What a store costs per request and per byte, measured once and given to
/// [`PageCacheConfig::store_profile`] so the read-ahead window does not
/// depend on this run's timings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StoreProfile {
    /// First-byte latency of one ranged read.
    pub latency: std::time::Duration,
    /// Streaming throughput of one ranged read, in bytes per second.
    pub bytes_per_sec: f64,
}

impl StoreProfile {
    /// The bytes one request streams in its own first-byte wait, as a power
    /// of two between `page` and `max` (the tuner's formula, on fixed
    /// inputs).
    pub fn window(&self, page: usize, max: usize) -> usize {
        let lo = page.max(1);
        let hi = max.max(lo);
        let ideal = self.latency.as_secs_f64() * self.bytes_per_sec;
        if !ideal.is_finite() {
            return if ideal > 0.0 { hi } else { lo };
        }
        (ideal.clamp(lo as f64, hi as f64) as usize)
            .next_power_of_two()
            .clamp(lo, hi)
    }
}

/// The heap a file opened through the spill fallback may use at once while
/// it is being filled: pages are fetched and written in chunks of at most
/// this many bytes (fewer when `max_request * concurrency` is smaller).
pub const SPILL_CHUNK_BYTES: usize = 16 << 20;
/// The default [`PageCacheConfig::spill_min_bytes`]: 1 MiB.
pub const DEFAULT_SPILL_MIN_BYTES: u64 = 1 << 20;

impl Default for PageCacheConfig {
    fn default() -> Self {
        Self {
            page_size: DEFAULT_PAGE_SIZE,
            disk_dir: None,
            disk_budget: u64::MAX,
            download_window: None,
            max_request: DEFAULT_MAX_REQUEST,
            concurrency: DEFAULT_CONCURRENCY,
            max_inflight_requests: DEFAULT_MAX_INFLIGHT_REQUESTS,
            max_inflight_bytes: DEFAULT_MAX_INFLIGHT_BYTES,
            pinned_extensions: DEFAULT_PINNED_EXTENSIONS
                .iter()
                .map(|e| (*e).to_string())
                .collect(),
            store_profile: None,
            demand_paging: true,
            spill_min_bytes: Some(DEFAULT_SPILL_MIN_BYTES),
            spill_budget: u64::MAX,
            heap_budget: u64::MAX,
        }
    }
}

/// Whether files can be demand-filled through `userfaultfd` in this process:
/// `true` on Linux where the kernel and its settings
/// (`vm.unprivileged_userfaultfd`, seccomp, the container) allow a
/// `userfaultfd`, `false` elsewhere. The first call probes it (starting the
/// process's fault handler); a node reports the answer as a metric so an
/// operator knows whether remote files are demand-filled or read through
/// ([`PageCacheConfig::spill_min_bytes`], [`PageCacheConfig::heap_budget`]).
/// [`PageCacheConfig::demand_paging`] `false` turns it off per cache.
pub fn demand_paging_available() -> bool {
    #[cfg(target_os = "linux")]
    {
        crate::paged_input::Pager::global().is_some()
    }
    #[cfg(not(target_os = "linux"))]
    {
        false
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
    /// Disk the tier's slots take (each page in the smallest slot size that
    /// holds it), which [`PageCacheConfig::disk_budget`] bounds.
    pub disk_footprint_bytes: u64,
    /// Bytes of mapped spill files open now ([`PageCacheConfig::spill_min_bytes`]).
    pub spill_open_bytes: u64,
    /// Files opened through the spill fallback since the cache was created.
    pub spill_opens: u64,
    /// Ranged requests that waited before being sent, the cache's
    /// [`PageCacheConfig::max_inflight_requests`] or
    /// [`PageCacheConfig::max_inflight_bytes`] being in use.
    pub fetch_waits: u64,
    /// Bytes of files held whole on the heap now, counted only while
    /// [`PageCacheConfig::heap_budget`] is bounded.
    pub heap_open_bytes: u64,
    /// Files opened whole onto the heap since the cache was created, counted
    /// only while [`PageCacheConfig::heap_budget`] is bounded.
    pub heap_opens: u64,
    /// Files opened demand-filled through `userfaultfd` since the cache was
    /// created.
    pub demand_paged_opens: u64,
}

#[derive(Default)]
struct Counters {
    disk_hits: AtomicU64,
    fetched_pages: AtomicU64,
    requests: AtomicU64,
    fetched_bytes: AtomicU64,
    shared_fetches: AtomicU64,
    read_ahead_pages: AtomicU64,
    spill_opens: AtomicU64,
    heap_opens: AtomicU64,
    demand_paged_opens: AtomicU64,
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
    /// Bytes of spill-mapped files open now.
    spill_open: Arc<AtomicU64>,
    /// Bytes of files held whole on the heap now.
    heap_open: Arc<AtomicU64>,
    /// Requests and bytes in flight over every reader.
    budget: FetchBudget,
}

/// Holds `bytes` of [`PageCacheConfig::spill_budget`] (or of
/// [`PageCacheConfig::heap_budget`]) while a mapped spill file (or a file
/// held whole on the heap) is open; dropping it returns them.
pub(crate) struct SpillReservation {
    open: Arc<AtomicU64>,
    bytes: u64,
}

impl Drop for SpillReservation {
    fn drop(&mut self) {
        let _ = self
            .open
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |open| {
                Some(open.saturating_sub(self.bytes))
            });
    }
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
        if config.max_inflight_requests == 0 || config.max_inflight_bytes < config.max_request {
            return Err(Error::IllegalArgument(format!(
                "max_inflight_requests must be positive and max_inflight_bytes ({}) at least \
                 max_request ({})",
                config.max_inflight_bytes, config.max_request
            )));
        }
        let disk = match &config.disk_dir {
            Some(dir) => Some(DiskTier::open(dir, config.disk_budget, config.page_size)?),
            None => None,
        };
        Ok(PageCache {
            disk,
            inflight: Mutex::new(HashMap::new()),
            counters: Counters::default(),
            tuner: Tuner::default(),
            spill_open: Arc::new(AtomicU64::new(0)),
            heap_open: Arc::new(AtomicU64::new(0)),
            budget: FetchBudget::new(
                config.max_inflight_requests,
                config.max_inflight_bytes as u64,
            ),
            config,
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
            disk_footprint_bytes: self.disk.as_ref().map_or(0, DiskTier::footprint),
            spill_open_bytes: self.spill_open.load(Ordering::Relaxed),
            spill_opens: c.spill_opens.load(Ordering::Relaxed),
            fetch_waits: self.budget.waits(),
            heap_open_bytes: self.heap_open.load(Ordering::Relaxed),
            heap_opens: c.heap_opens.load(Ordering::Relaxed),
            demand_paged_opens: c.demand_paged_opens.load(Ordering::Relaxed),
        }
    }

    /// Up to `limit` pages held in the disk tier for files whose cache key
    /// starts with `prefix`, the most read first. Empty without a disk tier.
    pub fn resident_pages(&self, prefix: &str, limit: usize) -> Vec<ResidentPage> {
        self.disk
            .as_ref()
            .map_or_else(Vec::new, |d| d.resident(prefix, limit))
    }

    /// The stored bytes of page `page` of the file with cache key `key`, if
    /// the disk tier holds it. Not a read: it neither counts toward the
    /// page's hits nor spares it from eviction, and it never fetches.
    pub fn peek_page(&self, key: &str, page: u64) -> Option<Arc<[u8]>> {
        self.disk.as_ref()?.peek(key, page)
    }

    /// Stores `bytes` as page `page` of the file with cache key `key`, as
    /// if it had been fetched: for a page another node held. A page already
    /// held, one too large for a slot, or a cache with no disk tier is left
    /// as it is. The caller vouches that the bytes are the file's stored
    /// bytes at that page: a file's key names its length, and a blob is
    /// immutable, so they are the same on every node.
    pub fn insert_page(&self, key: &str, page: u64, bytes: &[u8]) {
        if let Some(d) = &self.disk {
            d.put(&Arc::from(key), page, bytes);
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
        match (self.config.download_window, self.config.store_profile) {
            (Some(w), _) => w,
            (None, Some(profile)) => profile.window(self.config.page_size, self.config.max_request),
            (None, None) => self
                .tuner
                .window(self.config.page_size, self.config.max_request),
        }
    }

    /// Counts a spill file that opened.
    pub(crate) fn note_spill_open(&self) {
        self.counters.spill_opens.fetch_add(1, Ordering::Relaxed);
    }

    /// Counts a file that opened demand-filled.
    pub(crate) fn note_demand_paged_open(&self) {
        self.counters
            .demand_paged_opens
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Counts a file that opened whole onto the heap.
    pub(crate) fn note_heap_open(&self) {
        self.counters.heap_opens.fetch_add(1, Ordering::Relaxed);
    }

    /// Reserves `bytes` of [`PageCacheConfig::heap_budget`] until the guard
    /// drops, or fails when that would pass it.
    pub(crate) fn reserve_heap(&self, bytes: u64) -> Result<SpillReservation> {
        let budget = self.config.heap_budget;
        self.heap_open
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |open| {
                open.checked_add(bytes).filter(|total| *total <= budget)
            })
            .map_err(|open| {
                Error::Io(std::io::Error::other(format!(
                    "heap budget exhausted: {open} bytes open, {bytes} more asked, budget {budget}"
                )))
            })?;
        Ok(SpillReservation {
            open: Arc::clone(&self.heap_open),
            bytes,
        })
    }

    /// Reserves `bytes` of the spill budget until the guard drops, or
    /// fails when that would pass [`PageCacheConfig::spill_budget`].
    pub(crate) fn reserve_spill(&self, bytes: u64) -> Result<SpillReservation> {
        let budget = self.config.spill_budget;
        self.spill_open
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |open| {
                open.checked_add(bytes).filter(|total| *total <= budget)
            })
            .map_err(|open| {
                Error::Io(std::io::Error::other(format!(
                    "spill budget exhausted: {open} bytes open, {bytes} more asked, budget {budget}"
                )))
            })?;
        Ok(SpillReservation {
            open: Arc::clone(&self.spill_open),
            bytes,
        })
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
        // The stored bytes a run asks for, which its permit counts.
        let span = |&(start, count): &(usize, usize)| -> u64 {
            let (from, _) = geom.stored_range(mine[start].0);
            let last = mine[start.saturating_add(count).saturating_sub(1)].0;
            geom.stored_range(last).1.saturating_sub(from)
        };
        // The permit is held until the run's pages are cut from its buffer,
        // so the buffer is inside the bound for its whole life.
        let fetch_run =
            |&(start, count): &(usize, usize), _permit: FetchPermit<'_>| -> Vec<Fetched> {
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
            runs.iter()
                .map(|run| fetch_run(run, self.budget.acquire(span(run))))
                .collect()
        } else {
            let fetch_run = &fetch_run;
            let mut results: Vec<Vec<_>> = Vec::with_capacity(runs.len());
            for batch in runs.chunks(self.config.concurrency.max(1)) {
                let batch_results: Vec<Vec<_>> = std::thread::scope(|s| {
                    // Each run's permit is taken here, before its thread
                    // starts, so the threads fetching at once over every
                    // reader are bounded too, not only their requests.
                    let handles: Vec<_> = batch
                        .iter()
                        .map(|run| {
                            let permit = self.budget.acquire(span(run));
                            s.spawn(move || fetch_run(run, permit))
                        })
                        .collect();
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
mod tests;
