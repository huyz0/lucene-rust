//! Demand-filled inputs: a file opened as an anonymous mapping whose pages
//! are filled on first touch, and dropped again past a memory budget, so an
//! open file costs its working set in RAM rather than its size.
//!
//! The mapping is registered with Linux `userfaultfd`. The first access to
//! a missing page blocks the faulting thread; one of the pager's handler
//! threads asks the file's [`PageFiller`] for the page holding it (from the
//! [`crate::PageCache`], decoded when the file is encrypted) and installs it
//! with `UFFDIO_COPY`. [`ExternalBytes::will_need`] installs a range ahead
//! of the reads, in one batch, and returns any error fetching it; that is
//! how a reader keeps remote round trips few and sees failures as errors.
//! When resident pages exceed the budget, pages are released with
//! `MADV_DONTNEED`, and the next access faults them in again. Eviction is
//! a segmented FIFO: a new page starts on probation; a page that comes back
//! soon after it was released (within about a budget's worth of later
//! evictions, as the kernel's refault distance), or that a prefetch
//! announces again, is protected, and protected pages take at most
//! [`PROTECTED_SHARE`] of the budget. A scan cannot push out the working
//! set, which the pager cannot see being read once its pages are present. Readers see a
//! plain `&[u8]` throughout.
//!
//! A memory access cannot return an error: a page that still cannot be
//! filled after retries aborts the process rather than hand a reader wrong
//! bytes. Readers that announce their reads through `will_need` get the
//! error instead.
//!
//! [`Pager::global`] is `None` where `userfaultfd` is unavailable (not
//! Linux, or a seccomp policy that blocks it); callers then read files whole.

use std::collections::{BTreeMap, VecDeque};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock, Weak};
use std::time::Duration;

use crate::directory::ExternalBytes;
use crate::error::{Error, Result};
use crate::lock::lock_ignoring_poison;

/// Takes each page a [`PageFiller`] reads: its index and bytes. It may be
/// called from several threads at once.
pub(crate) type PageSink = Arc<dyn Fn(u64, &[u8]) + Send + Sync>;

/// Supplies a demand-filled file's pages.
pub(crate) trait PageFiller: Send + Sync {
    /// The decoded bytes of each of `pages` (sorted, distinct), each
    /// `page_len` bytes but the file's last, which may be shorter.
    fn fill(&self, pages: &[u64]) -> Result<Vec<Arc<[u8]>>>;

    /// Reads `pages` (sorted, distinct) and hands each to `sink` as it
    /// arrives, so the bytes go straight from the read buffer into the
    /// mapping. The pages of one read go last to first: a reader woken by
    /// a run's first page finds the rest of the run present. On an error
    /// some pages may have been delivered.
    fn fill_into(&self, pages: &[u64], sink: &PageSink) -> Result<()> {
        let filled = self.fill(pages)?;
        if filled.len() != pages.len() {
            return Err(Error::Io(std::io::Error::other(format!(
                "{} of {} pages returned",
                filled.len(),
                pages.len()
            ))));
        }
        for (&page, bytes) in pages.iter().zip(&filled).rev() {
            sink(page, bytes);
        }
        Ok(())
    }

    /// Starts [`Self::fill_into`] on another thread and returns at once;
    /// `done` runs with its result when it ends. Returns `false`, calling
    /// neither, when this filler cannot read in the background: a fault
    /// then reads its read-ahead window together with the faulting page.
    fn fill_background(&self, pages: Vec<u64>, sink: PageSink, done: BackgroundDone) -> bool {
        let _ = (pages, sink, done);
        false
    }
}

/// Runs when a [`PageFiller::fill_background`] read ends, with its result.
pub(crate) type BackgroundDone = Box<dyn FnOnce(Result<()>) + Send>;

const UFFD_API: u64 = 0xAA;
const UFFD_USER_MODE_ONLY: libc::c_int = 1;
const UFFDIO_API: libc::c_ulong = 0xC018_AA3F;
const UFFDIO_REGISTER: libc::c_ulong = 0xC020_AA00;
const UFFDIO_UNREGISTER: libc::c_ulong = 0x8010_AA01;
const UFFDIO_WAKE: libc::c_ulong = 0x8010_AA02;
const UFFDIO_COPY: libc::c_ulong = 0xC028_AA03;
const UFFDIO_REGISTER_MODE_MISSING: u64 = 1;
const UFFD_EVENT_PAGEFAULT: u8 = 0x12;

#[repr(C)]
struct UffdioApi {
    api: u64,
    features: u64,
    ioctls: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct UffdioRange {
    start: u64,
    len: u64,
}

#[repr(C)]
struct UffdioRegister {
    range: UffdioRange,
    mode: u64,
    ioctls: u64,
}

#[repr(C)]
struct UffdioCopy {
    dst: u64,
    src: u64,
    len: u64,
    mode: u64,
    copy: i64,
}

/// `struct uffd_msg`: 32 bytes; a page fault's address is at offset 16.
#[repr(C)]
#[derive(Clone, Copy)]
struct UffdMsg {
    event: u8,
    _reserved: [u8; 7],
    flags: u64,
    address: u64,
    _feat: u64,
}

/// Bytes resident across every demand-filled file before the oldest pages
/// are released, until [`Pager::set_budget`] changes it.
pub const DEFAULT_BUDGET: u64 = 1 << 30;
/// Threads serving faults: each blocks on one page's fetch, so several let
/// faults on different pages proceed at once.
const HANDLER_THREADS: usize = 8;
/// Waits in milliseconds between attempts to fill a faulting page, about a
/// minute in all; then the process aborts.
const FAULT_RETRIES: [u64; 12] = [
    50, 100, 250, 500, 1000, 2000, 4000, 8000, 8000, 10000, 10000, 15000,
];

static GLOBAL: OnceLock<Option<&'static Pager>> = OnceLock::new();

/// The process-wide fault handler: one `userfaultfd`, its handler threads,
/// every registered file, and the resident budget.
pub struct Pager {
    uffd: OwnedFd,
    os_page: usize,
    files: RwLock<BTreeMap<usize, Weak<File>>>,
    resident: Mutex<Queues>,
    /// Bytes released so far, in [`STAMP_UNIT`]s: the clock refault
    /// distances are measured on.
    evicted_units: AtomicU64,
    resident_bytes: AtomicU64,
    budget: AtomicU64,
    faults: AtomicU64,
    refaults: AtomicU64,
    fault_nanos: AtomicU64,
    evictions: AtomicU64,
}

/// The share of the budget protected pages may take, in quarters.
pub const PROTECTED_SHARE: u64 = 3;

/// Refault distances are kept in units of this many bytes.
const STAMP_UNIT: u32 = 12;

/// A resident page: its file, index, and bytes.
type Entry = (Weak<File>, usize, u64);

/// Resident pages in eviction order, oldest first.
#[derive(Default)]
struct Queues {
    probation: VecDeque<Entry>,
    protected: VecDeque<Entry>,
    protected_bytes: u64,
}

impl Queues {
    fn protect(&mut self, entry: Entry) {
        self.protected_bytes = self.protected_bytes.saturating_add(entry.2);
        self.protected.push_back(entry);
    }

    fn pop_protected(&mut self) -> Option<Entry> {
        let entry = self.protected.pop_front()?;
        self.protected_bytes = self.protected_bytes.saturating_sub(entry.2);
        Some(entry)
    }
}

/// Counters of a [`Pager`] since it started.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PagerStats {
    /// Page faults served.
    pub faults: u64,
    /// Pages filled again after eviction released them.
    pub refaults: u64,
    /// Time the handler threads spent serving faults, in nanoseconds.
    pub fault_nanos: u64,
    /// Pages released to hold the budget.
    pub evictions: u64,
}

/// One registered file: its mapping and where its pages come from.
struct File {
    base: usize,
    map_len: usize,
    len: usize,
    page: usize,
    pages: usize,
    filler: Arc<dyn PageFiller>,
    name: String,
    pager: &'static Pager,
    /// Pages resident now, so eviction never releases one twice.
    live: Vec<AtomicBool>,
    /// Resident pages a prefetch announced since eviction last passed
    /// them: they get a second chance before being released.
    referenced: Vec<AtomicBool>,
    /// The most pages a fault reads at once when faults run in sequence
    /// (1: no read-ahead).
    readahead_max: usize,
    /// The page the last fault was on, and the read-ahead window it used.
    last_fault: AtomicU64,
    /// When eviction last released each page, on the pager's eviction
    /// clock (0: never).
    evicted_at: Vec<AtomicU32>,
    /// Pages a background read-ahead is fetching now.
    inflight: Vec<AtomicBool>,
    /// How many pages are resident: all of them lets a prefetch skip the
    /// file without looking at each page.
    live_count: AtomicUsize,
}

/// A demand-filled file, as [`ExternalBytes`].
pub(crate) struct PagedBytes {
    file: Arc<File>,
}

impl ExternalBytes for PagedBytes {
    fn bytes(&self) -> &[u8] {
        if self.file.len == 0 {
            return &[];
        }
        // SAFETY: `base..base + map_len` is a live private mapping owned by
        // `self.file` until `Drop`, and `len <= map_len`. Its contents are
        // only ever written by `UFFDIO_COPY` with the filler's bytes for the
        // same page, which are the same every time (files are immutable).
        unsafe { std::slice::from_raw_parts(self.file.base as *const u8, self.file.len) }
    }

    fn will_need(&self, offset: usize, len: usize) -> Result<()> {
        let missing = self.file.missing(&[(offset, len)]);
        self.file.fill(&missing)?;
        self.file.pager.evict();
        Ok(())
    }
}

impl File {
    /// Whether every page is resident and nothing is about to be evicted,
    /// so a prefetch has nothing to fetch and no page to mark in use.
    fn all_resident(&self) -> bool {
        self.live_count.load(Ordering::Relaxed) >= self.pages
            && self.pager.resident_bytes() <= self.pager.budget()
    }

    /// The pages of `ranges` (offsets in the file) not resident now.
    fn missing(&self, ranges: &[(usize, usize)]) -> Vec<u64> {
        let mut pages = Vec::new();
        for &(offset, len) in ranges {
            let end = offset.saturating_add(len).min(self.len);
            if offset >= end {
                continue;
            }
            let first = offset.checked_div(self.page).unwrap_or(0);
            let last = end.saturating_sub(1).checked_div(self.page).unwrap_or(0);
            for p in first..=last {
                if self.live[p].load(Ordering::Acquire) {
                    if !self.referenced[p].load(Ordering::Relaxed) {
                        self.referenced[p].store(true, Ordering::Relaxed);
                    }
                } else if !self.inflight[p].load(Ordering::Acquire) {
                    pages.push(p as u64);
                }
            }
        }
        pages.sort_unstable();
        pages.dedup();
        pages
    }

    /// Fetches `pages` (sorted, distinct) in one batch and installs them.
    fn fill(self: &Arc<Self>, pages: &[u64]) -> Result<()> {
        if pages.is_empty() {
            return Ok(());
        }
        self.filler.fill_into(pages, &self.sink())
    }

    /// Starts reading `pages` (sorted, distinct) in the background where the
    /// filler can and returns at once: a fault on one of them waits for it.
    /// Returns the pages to read here instead: none once started (or
    /// already on their way), the unclaimed ones where the filler cannot
    /// read in the background.
    fn start_background(self: &Arc<Self>, pages: &[u64]) -> Vec<u64> {
        let claimed: Vec<u64> = pages
            .iter()
            .copied()
            .filter(|&p| !self.inflight[p as usize].swap(true, Ordering::AcqRel))
            .collect();
        if claimed.is_empty() {
            return claimed;
        }
        let owner = Arc::clone(self);
        let list = claimed.clone();
        let done: BackgroundDone = Box::new(move |_| owner.finish_background(&list));
        if self
            .filler
            .fill_background(claimed.clone(), self.sink(), done)
        {
            return Vec::new();
        }
        for &p in &claimed {
            self.inflight[p as usize].store(false, Ordering::Release);
        }
        // A fault raised on one of them while it was claimed was left to
        // the read that is not coming: wake it to fault again.
        if let (Some(&first), Some(&last)) = (claimed.first(), claimed.last()) {
            self.pager.wake_pages(self, first as usize, last as usize);
        }
        claimed
    }

    /// Ends a background read of `pages`: they are no longer in flight,
    /// and a thread that faulted on one the read did not install is woken
    /// to fault again and read it itself.
    fn finish_background(&self, pages: &[u64]) {
        for &p in pages {
            self.inflight[p as usize].store(false, Ordering::Release);
        }
        if let (Some(&first), Some(&last)) = (pages.first(), pages.last()) {
            self.pager.wake_pages(self, first as usize, last as usize);
        }
        self.pager.evict();
    }

    /// A sink installing each page it is handed into this file.
    fn sink(self: &Arc<Self>) -> PageSink {
        let file = Arc::clone(self);
        Arc::new(move |page, bytes| {
            file.pager.install(&file, page as usize, bytes);
        })
    }
}

/// `IndexInput.prefetch` by address: brings in every page of `ranges` that
/// lies in a demand-filled file of the global [`Pager`] and is not resident,
/// all files' pages fetched at once, and returns the first fetch error.
/// Ranges anywhere else (heap, `mmap`) are skipped without a system call.
pub fn prefetch(ranges: &[&[u8]]) -> Result<()> {
    let Some(pager) = Pager::started() else {
        return Ok(());
    };
    // Each file touched, with its ranges as (offset, len).
    type Spans = Vec<(usize, usize)>;
    let mut wanted: Vec<(Arc<File>, Spans)> = Vec::new();
    {
        let files = match pager.files.read() {
            Ok(files) => files,
            Err(poisoned) => poisoned.into_inner(),
        };
        if files.is_empty() {
            return Ok(());
        }
        for range in ranges {
            let addr = range.as_ptr() as usize;
            // Ranges of one batch mostly fall in one file: the last file
            // found takes them without a lookup.
            if let Some((file, spans)) = wanted.last_mut() {
                let offset = addr.wrapping_sub(file.base);
                if addr >= file.base && offset < file.len {
                    if !file.all_resident() {
                        spans.push((offset, range.len().max(1)));
                    }
                    continue;
                }
            }
            let Some((_, weak)) = files.range(..=addr).next_back() else {
                continue;
            };
            let Some(file) = weak.upgrade() else {
                continue;
            };
            let offset = addr.wrapping_sub(file.base);
            if offset >= file.len {
                continue;
            }
            let span = (offset, range.len().max(1));
            let all = file.all_resident();
            match wanted.iter_mut().find(|(f, _)| Arc::ptr_eq(f, &file)) {
                Some((_, spans)) if !all => spans.push(span),
                Some(_) => {}
                None if all => wanted.push((file, Vec::new())),
                None => wanted.push((file, vec![span])),
            }
        }
    }
    let work: Vec<(Arc<File>, Vec<u64>)> = wanted
        .into_iter()
        .map(|(file, spans)| {
            let missing = file.missing(&spans);
            (file, missing)
        })
        .filter(|(_, missing)| !missing.is_empty())
        .collect();
    let result = match work.as_slice() {
        [] => Ok(()),
        [(file, missing)] => file.fill(missing),
        // A bounded number of files at a time: a query over many segments
        // must not start a thread per file.
        _ => {
            let mut first = Ok(());
            for batch in work.chunks(HANDLER_THREADS) {
                std::thread::scope(|scope| {
                    let handles: Vec<_> = batch
                        .iter()
                        .map(|(file, missing)| scope.spawn(move || file.fill(missing)))
                        .collect();
                    for handle in handles {
                        let r = handle.join().unwrap_or_else(|_| {
                            Err(Error::Io(std::io::Error::other(
                                "a prefetch thread panicked",
                            )))
                        });
                        if first.is_ok() {
                            first = r;
                        }
                    }
                });
            }
            first
        }
    };
    pager.evict();
    result
}

/// Whether `bytes` starts in a demand-filled file of the global [`Pager`]:
/// what a reader asks before spending work on a [`prefetch`] that would
/// otherwise find nothing to fetch.
pub fn is_paged(bytes: &[u8]) -> bool {
    let Some(pager) = Pager::started() else {
        return false;
    };
    let files = match pager.files.read() {
        Ok(files) => files,
        Err(poisoned) => poisoned.into_inner(),
    };
    let addr = bytes.as_ptr() as usize;
    files
        .range(..=addr)
        .next_back()
        .and_then(|(_, weak)| weak.upgrade())
        .is_some_and(|file| addr.wrapping_sub(file.base) < file.map_len.max(1))
}

impl Drop for PagedBytes {
    fn drop(&mut self) {
        if self.file.map_len == 0 {
            return;
        }
        if let Ok(mut files) = self.file.pager.files.write() {
            files.remove(&self.file.base);
        }
    }
}

impl Drop for File {
    /// Unmaps once nothing, the handler threads included, holds the file.
    fn drop(&mut self) {
        if self.map_len == 0 {
            return;
        }
        let mut range = UffdioRange {
            start: self.base as u64,
            len: self.map_len as u64,
        };
        // SAFETY: the range was registered on this uffd and mapped by
        // `Pager::open`; the last reference to it is going away.
        unsafe {
            libc::ioctl(self.pager.uffd.as_raw_fd(), UFFDIO_UNREGISTER, &mut range);
            libc::munmap(self.base as *mut libc::c_void, self.map_len);
        }
        let live = self
            .live
            .iter()
            .filter(|c| c.load(Ordering::Relaxed))
            .count();
        let bytes = (live as u64).saturating_mul(self.page as u64);
        let _ = self
            .pager
            .resident_bytes
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |r| {
                Some(r.saturating_sub(bytes))
            });
    }
}

impl Pager {
    /// The process's pager, started on first use; `None` when
    /// `userfaultfd` is unavailable.
    pub fn global() -> Option<&'static Pager> {
        *GLOBAL.get_or_init(|| Pager::spawn(false))
    }

    /// [`Pager::global`] if it has been started, without starting it.
    fn started() -> Option<&'static Pager> {
        GLOBAL.get().copied().flatten()
    }

    /// Starts a pager and its handler threads. Both live for the rest of
    /// the process.
    fn spawn(user_mode_only: bool) -> Option<&'static Pager> {
        let pager = Pager::start(user_mode_only).ok()?;
        let pager: &'static Pager = Box::leak(Box::new(pager));
        for i in 0..HANDLER_THREADS {
            std::thread::Builder::new()
                .name(format!("lucene-page-fault-{i}"))
                .spawn(move || pager.serve())
                .ok()?;
        }
        Some(pager)
    }

    fn start(user_mode_only: bool) -> std::io::Result<Pager> {
        // Full mode first (kernel accesses fault in too); user-mode-only is
        // what an unprivileged process gets when vm.unprivileged_userfaultfd
        // is 0.
        let full = libc::O_CLOEXEC | libc::O_NONBLOCK;
        let user = full | UFFD_USER_MODE_ONLY;
        let modes: &[libc::c_int] = if user_mode_only {
            &[user]
        } else {
            &[full, user]
        };
        let mut fd = -1;
        for &flags in modes {
            // SAFETY: a plain syscall with integer arguments.
            fd = unsafe { libc::syscall(libc::SYS_userfaultfd, flags) } as libc::c_int;
            if fd >= 0 {
                break;
            }
        }
        if fd < 0 {
            return Err(std::io::Error::last_os_error());
        }
        // SAFETY: `fd` is a fresh descriptor this function owns.
        let uffd = unsafe { OwnedFd::from_raw_fd(fd) };
        let mut api = UffdioApi {
            api: UFFD_API,
            features: 0,
            ioctls: 0,
        };
        // SAFETY: `api` is a valid `struct uffdio_api` for the call.
        if unsafe { libc::ioctl(fd, UFFDIO_API, &mut api) } != 0 {
            return Err(std::io::Error::last_os_error());
        }
        // SAFETY: sysconf has no preconditions.
        let os_page = usize::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) })
            .map_err(|_| std::io::Error::other("page size"))?;
        Ok(Pager {
            uffd,
            os_page,
            files: RwLock::new(BTreeMap::new()),
            resident: Mutex::new(Queues::default()),
            evicted_units: AtomicU64::new(0),
            resident_bytes: AtomicU64::new(0),
            budget: AtomicU64::new(DEFAULT_BUDGET),
            faults: AtomicU64::new(0),
            refaults: AtomicU64::new(0),
            fault_nanos: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
        })
    }

    /// Whether pages of `page_len` bytes can be demand-filled: each must be
    /// a whole number of OS pages.
    pub fn supports(&self, page_len: usize) -> bool {
        page_len >= self.os_page && page_len.is_multiple_of(self.os_page)
    }

    /// Sets the resident budget in bytes; pages past it are released on
    /// the next fill.
    pub fn set_budget(&self, bytes: u64) {
        self.budget.store(bytes, Ordering::Relaxed);
    }

    /// The resident budget in bytes.
    pub fn budget(&self) -> u64 {
        self.budget.load(Ordering::Relaxed)
    }

    /// Bytes resident now, across every file.
    pub fn resident_bytes(&self) -> u64 {
        self.resident_bytes.load(Ordering::Relaxed)
    }

    /// Page faults served since the pager started.
    pub fn faults(&self) -> u64 {
        self.faults.load(Ordering::Relaxed)
    }

    /// Fault, refill and eviction counters since the pager started.
    pub fn stats(&self) -> PagerStats {
        PagerStats {
            faults: self.faults.load(Ordering::Relaxed),
            refaults: self.refaults.load(Ordering::Relaxed),
            fault_nanos: self.fault_nanos.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
        }
    }

    /// Opens a file of `len` bytes in pages of `page_len` (see
    /// [`Pager::supports`]) filled by `filler`.
    pub(crate) fn open(
        &'static self,
        len: usize,
        page_len: usize,
        filler: Arc<dyn PageFiller>,
        name: &str,
    ) -> Result<PagedBytes> {
        self.open_with_readahead(len, page_len, filler, name, 1)
    }

    /// [`Pager::open`] for a file whose faults read ahead: a fault on the
    /// page after the previous fault's last page reads the next pages too,
    /// twice as many as the previous one did, up to `readahead_max` pages,
    /// as the kernel's read-ahead does for a mapped file read in sequence.
    pub(crate) fn open_with_readahead(
        &'static self,
        len: usize,
        page_len: usize,
        filler: Arc<dyn PageFiller>,
        name: &str,
        readahead_max: usize,
    ) -> Result<PagedBytes> {
        if !self.supports(page_len) {
            return Err(Error::IllegalArgument(format!(
                "page of {page_len} bytes is not a whole number of OS pages"
            )));
        }
        let pages = len.div_ceil(page_len);
        let map_len = len.div_ceil(self.os_page).saturating_mul(self.os_page);
        let mut base = 0usize;
        if map_len > 0 {
            // SAFETY: an anonymous private mapping; the result is checked.
            let ptr = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    map_len,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                    -1,
                    0,
                )
            };
            if ptr == libc::MAP_FAILED {
                return Err(Error::Io(std::io::Error::last_os_error()));
            }
            base = ptr as usize;
            let mut reg = UffdioRegister {
                range: UffdioRange {
                    start: base as u64,
                    len: map_len as u64,
                },
                mode: UFFDIO_REGISTER_MODE_MISSING,
                ioctls: 0,
            };
            // SAFETY: registers the mapping just created.
            if unsafe { libc::ioctl(self.uffd.as_raw_fd(), UFFDIO_REGISTER, &mut reg) } != 0 {
                let e = std::io::Error::last_os_error();
                // SAFETY: unmaps the mapping just created.
                unsafe { libc::munmap(ptr, map_len) };
                return Err(Error::Io(e));
            }
        }
        let file = Arc::new(File {
            base,
            map_len,
            len,
            page: page_len,
            pages,
            filler,
            name: name.to_string(),
            pager: self,
            live: (0..pages).map(|_| AtomicBool::new(false)).collect(),
            referenced: (0..pages).map(|_| AtomicBool::new(false)).collect(),
            readahead_max: readahead_max.max(1),
            last_fault: AtomicU64::new(u64::MAX),
            evicted_at: (0..pages).map(|_| AtomicU32::new(0)).collect(),
            inflight: (0..pages).map(|_| AtomicBool::new(false)).collect(),
            live_count: AtomicUsize::new(0),
        });
        if map_len > 0 {
            self.files
                .write()
                .map_err(|_| Error::Io(std::io::Error::other("pager registry poisoned")))?
                .insert(base, Arc::downgrade(&file));
        }
        Ok(PagedBytes { file })
    }

    /// A handler loop: waits for fault messages and resolves each.
    fn serve(&'static self) {
        let fd = self.uffd.as_raw_fd();
        let mut msgs = [UffdMsg {
            event: 0,
            _reserved: [0; 7],
            flags: 0,
            address: 0,
            _feat: 0,
        }; 4];
        // Each handler waits on its own epoll set holding the uffd as
        // EPOLLEXCLUSIVE, so a fault wakes one handler, not all of them.
        // SAFETY: plain syscalls; the result is checked.
        let ep = unsafe { libc::epoll_create1(libc::EPOLL_CLOEXEC) };
        let exclusive = ep >= 0 && {
            let mut ev = libc::epoll_event {
                events: (libc::EPOLLIN | libc::EPOLLEXCLUSIVE) as u32,
                u64: 0,
            };
            // SAFETY: `ev` is a valid event for the call.
            unsafe { libc::epoll_ctl(ep, libc::EPOLL_CTL_ADD, fd, &mut ev) == 0 }
        };
        loop {
            if exclusive {
                let mut ev = libc::epoll_event { events: 0, u64: 0 };
                // SAFETY: room for one event.
                if unsafe { libc::epoll_wait(ep, &mut ev, 1, -1) } < 0 {
                    continue;
                }
            } else {
                let mut pfd = libc::pollfd {
                    fd,
                    events: libc::POLLIN,
                    revents: 0,
                };
                // SAFETY: one valid pollfd.
                if unsafe { libc::poll(&mut pfd, 1, -1) } < 0 {
                    continue;
                }
            }
            self.take_messages(&mut msgs);
        }
    }

    /// Reads and resolves the fault messages waiting, if any.
    fn take_messages(&'static self, msgs: &mut [UffdMsg; 4]) -> bool {
        let fd = self.uffd.as_raw_fd();
        // SAFETY: reads whole `uffd_msg` records into `msgs`. The fd is
        // non-blocking, so a message another thread took first returns
        // EAGAIN here.
        let n = unsafe { libc::read(fd, msgs.as_mut_ptr().cast(), std::mem::size_of_val(msgs)) };
        if n <= 0 {
            return false;
        }
        let count = (n as usize)
            .checked_div(std::mem::size_of::<UffdMsg>())
            .unwrap_or(0)
            .min(msgs.len());
        for m in &msgs[..count] {
            if m.event == UFFD_EVENT_PAGEFAULT {
                let started = std::time::Instant::now();
                self.resolve(m.address as usize);
                let nanos = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
                self.fault_nanos.fetch_add(nanos, Ordering::Relaxed);
            }
        }
        true
    }

    /// Fills and installs the page holding `addr`.
    fn resolve(&'static self, addr: usize) {
        let file = {
            let files = match self.files.read() {
                Ok(f) => f,
                Err(p) => p.into_inner(),
            };
            files
                .range(..=addr)
                .next_back()
                .and_then(|(_, w)| w.upgrade())
        };
        let Some(file) = file else {
            return;
        };
        let Some(index) = addr
            .checked_sub(file.base)
            .and_then(|off| off.checked_div(file.page))
        else {
            return;
        };
        if index >= file.pages {
            return;
        }
        if file.live[index].load(Ordering::Acquire) {
            // Installed since the fault was raised.
            self.wake(addr);
            return;
        }
        if file.inflight[index].load(Ordering::Acquire) {
            // A background read-ahead brings it: installing it wakes the
            // faulting thread, and a failed one wakes it to fault again.
            return;
        }
        if file.readahead_max > 1 && self.fill_ahead(&file, index) {
            self.faults.fetch_add(1, Ordering::Relaxed);
            self.wake(addr);
            self.evict();
            return;
        }
        let mut attempt = 0usize;
        let sink = file.sink();
        loop {
            let filled = file.filler.fill_into(&[index as u64], &sink);
            match filled {
                Ok(()) => break,
                other => {
                    let e = other
                        .err()
                        .map_or_else(|| "no page returned".to_string(), |e| e.to_string());
                    if attempt >= FAULT_RETRIES.len() {
                        eprintln!(
                            "lucene-store: page {index} of {} could not be read ({e}); aborting",
                            file.name
                        );
                        std::process::abort();
                    }
                }
            }
            std::thread::sleep(Duration::from_millis(
                FAULT_RETRIES.get(attempt).copied().unwrap_or(15000),
            ));
            attempt = attempt.saturating_add(1);
        }
        self.faults.fetch_add(1, Ordering::Relaxed);
        // Installing woke the faulting thread, unless another thread
        // installed the page first: wake it in that case too.
        self.wake(addr);
        self.evict();
    }

    /// Wakes the threads waiting on pages `first..=last` of `file`.
    fn wake_pages(&self, file: &File, first: usize, last: usize) {
        let start = file.base.saturating_add(first.saturating_mul(file.page));
        let end = file
            .base
            .saturating_add(last.saturating_add(1).saturating_mul(file.page))
            .min(file.base.saturating_add(file.map_len));
        let mut range = UffdioRange {
            start: start as u64,
            len: end.saturating_sub(start) as u64,
        };
        // SAFETY: wakes waiters on pages of the registered mapping.
        unsafe { libc::ioctl(self.uffd.as_raw_fd(), UFFDIO_WAKE, &mut range) };
    }

    /// Wakes the threads waiting on the OS page holding `addr`.
    fn wake(&self, addr: usize) {
        let mut range = UffdioRange {
            start: (addr & !self.os_page.saturating_sub(1)) as u64,
            len: self.os_page as u64,
        };
        // SAFETY: wakes waiters on a page of the registered mapping.
        unsafe { libc::ioctl(self.uffd.as_raw_fd(), UFFDIO_WAKE, &mut range) };
    }

    /// Reads the read-ahead window after the faulting page `index` when the
    /// fault continues a run (the previous fault's window ended right
    /// before it). Where the filler reads in the background, the window
    /// after `index` is started there and `false` returned, so the faulting
    /// page is read alone and its thread runs on while the rest arrives, as
    /// the kernel's asynchronous read-ahead does. Otherwise `index` and the
    /// window are read in one read; returns whether `index` was filled.
    /// `false` leaves `index` to the single-page path, which retries.
    fn fill_ahead(&self, file: &Arc<File>, index: usize) -> bool {
        // `last_fault` packs the previous window's next page (high 40 bits)
        // and its size in pages (low 24 bits).
        let prev = file.last_fault.load(Ordering::Relaxed);
        let (next, size) = ((prev >> 24) as usize, (prev & 0xFF_FFFF) as usize);
        let window = if prev != u64::MAX && next == index {
            size.saturating_mul(2).clamp(2, file.readahead_max)
        } else {
            1
        };
        let end = index.saturating_add(window).min(file.pages);
        let mut pages: Vec<u64> = vec![index as u64];
        pages.extend(
            (index.saturating_add(1)..end)
                .filter(|&p| !file.live[p].load(Ordering::Acquire))
                .map(|p| p as u64),
        );
        let packed = ((end as u64) << 24) | (end.saturating_sub(index) as u64 & 0xFF_FFFF);
        file.last_fault.store(packed, Ordering::Relaxed);
        if pages.len() == 1 && window == 1 {
            return false;
        }
        let ahead = pages.get(1..).unwrap_or(&[]);
        let left = file.start_background(ahead);
        if left.is_empty() {
            // Started (or already on its way): the faulting page alone.
            return false;
        }
        let mut pages = vec![index as u64];
        pages.extend(left);
        // The filler hands a read's pages over last to first: installing
        // the faulting page wakes its thread, whose next read must not
        // fault on a page still on its way.
        file.filler.fill_into(&pages, &file.sink()).is_ok()
            && file.live[index].load(Ordering::Acquire)
    }

    /// Copies `bytes` (page `index`'s content) into the mapping, returning
    /// whether this call installed any of it. OS pages already present
    /// (installed by another thread) are skipped, and a partial or
    /// interrupted copy continues where it stopped, so the whole page is
    /// present when this returns `true` or the page was already resident.
    fn install(&self, file: &Arc<File>, index: usize, bytes: &[u8]) -> bool {
        let start = index.saturating_mul(file.page);
        let len = file.page.min(file.map_len.saturating_sub(start));
        if len == 0 {
            return false;
        }
        // UFFDIO_COPY copies whole OS pages: pad a short last page.
        let padded;
        let src: &[u8] = if bytes.len() >= len {
            bytes
        } else {
            let mut p = vec![0u8; len];
            p[..bytes.len()].copy_from_slice(bytes);
            padded = p;
            &padded
        };
        let mut done = 0usize;
        let mut installed = false;
        while done < len {
            let mut copy = UffdioCopy {
                dst: file.base.saturating_add(start).saturating_add(done) as u64,
                src: src[done..].as_ptr() as u64,
                len: len.saturating_sub(done) as u64,
                mode: 0,
                copy: 0,
            };
            // SAFETY: `dst..dst + len` lies inside the registered mapping and
            // `src` holds at least that many bytes.
            let rc = unsafe { libc::ioctl(self.uffd.as_raw_fd(), UFFDIO_COPY, &mut copy) };
            if rc == 0 {
                installed = true;
                break;
            }
            let errno = std::io::Error::last_os_error().raw_os_error();
            if copy.copy > 0 {
                // Part copied: continue after it.
                installed = true;
                done = done.saturating_add(usize::try_from(copy.copy).unwrap_or(0));
            } else if errno == Some(libc::EEXIST) {
                // This OS page is present already: skip it.
                done = done.saturating_add(self.os_page);
            } else if errno == Some(libc::EAGAIN) {
                std::thread::yield_now();
            } else {
                return false;
            }
        }
        if installed && !file.live[index].swap(true, Ordering::AcqRel) {
            file.live_count.fetch_add(1, Ordering::Relaxed);
            file.referenced[index].store(false, Ordering::Relaxed);
            let stamp = file.evicted_at[index].load(Ordering::Relaxed);
            // Back within a budget's worth of evictions: it would have
            // stayed in a cache that much larger, so it is in use.
            let protect = stamp != 0 && {
                self.refaults.fetch_add(1, Ordering::Relaxed);
                let distance = u64::from(self.eviction_clock().wrapping_sub(stamp)) << STAMP_UNIT;
                distance <= self.budget.load(Ordering::Relaxed)
            };
            let bytes = file.page as u64;
            let entry = (Arc::downgrade(file), index, bytes);
            let mut q = lock_ignoring_poison(&self.resident);
            if protect {
                q.protect(entry);
            } else {
                q.probation.push_back(entry);
            }
            // Counted once queued, so eviction never sees bytes over the
            // budget that it cannot find in the queues.
            self.resident_bytes.fetch_add(bytes, Ordering::Relaxed);
        }
        installed
    }

    /// The eviction clock, as stored in [`File::evicted_at`].
    fn eviction_clock(&self) -> u32 {
        self.evicted_units.load(Ordering::Relaxed) as u32
    }

    /// Releases resident pages until the budget holds. Protected pages over
    /// their share move back to probation; probation goes oldest first,
    /// except that a page a prefetch announced since it was queued is
    /// protected instead, and a protected one gets one more pass.
    fn evict(&self) {
        let budget = self.budget.load(Ordering::Relaxed);
        if self.resident_bytes.load(Ordering::Relaxed) <= budget {
            return;
        }
        let cap = (budget / 4).saturating_mul(PROTECTED_SHARE);
        let mut q = lock_ignoring_poison(&self.resident);
        // Each page can be passed over once per call, so the loop ends.
        let mut second_chances = q.probation.len().saturating_add(q.protected.len());
        while self.resident_bytes.load(Ordering::Relaxed) > budget {
            while q.protected_bytes > cap {
                let Some(entry) = q.pop_protected() else {
                    break;
                };
                q.probation.push_back(entry);
            }
            let Some(entry) = q.probation.pop_front().or_else(|| q.pop_protected()) else {
                break;
            };
            let Some(file) = entry.0.upgrade() else {
                continue;
            };
            let index = entry.1;
            if !file.live[index].load(Ordering::Acquire) {
                continue;
            }
            if second_chances > 0 && file.referenced[index].swap(false, Ordering::Relaxed) {
                second_chances = second_chances.saturating_sub(1);
                q.protect(entry);
                continue;
            }
            if !file.live[index].swap(false, Ordering::AcqRel) {
                continue;
            }
            file.live_count.fetch_sub(1, Ordering::Relaxed);
            let units = (file.page as u64 >> STAMP_UNIT).max(1);
            let now = self
                .evicted_units
                .fetch_add(units, Ordering::Relaxed)
                .wrapping_add(units) as u32;
            file.evicted_at[index].store(now.max(1), Ordering::Relaxed);
            self.evictions.fetch_add(1, Ordering::Relaxed);
            let start = file.base.saturating_add(index.saturating_mul(file.page));
            let len = file
                .page
                .min(file.map_len.saturating_sub(index.saturating_mul(file.page)));
            // SAFETY: drops the pages of one page of a live registered
            // mapping; the next access faults it back in, identical.
            unsafe { libc::madvise(start as *mut libc::c_void, len, libc::MADV_DONTNEED) };
            let _ = self
                .resident_bytes
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |r| {
                    Some(r.saturating_sub(file.page as u64))
                });
        }
    }
}

#[cfg(test)]
mod tests {
    // Test fixtures' own arithmetic -- see `docs/arithmetic-gate.md`'s "Test code".
    #![allow(clippy::arithmetic_side_effects)]
    use super::*;
    use std::sync::atomic::AtomicUsize;

    struct Pattern {
        len: usize,
        page: usize,
        fills: AtomicUsize,
        calls: AtomicUsize,
        fail_first: AtomicUsize,
        background: AtomicBool,
    }

    fn byte(i: usize) -> u8 {
        (i.wrapping_mul(2_654_435_761) >> 13) as u8
    }

    impl PageFiller for Pattern {
        fn fill(&self, pages: &[u64]) -> Result<Vec<Arc<[u8]>>> {
            if self.fail_first.load(Ordering::Relaxed) > 0 {
                self.fail_first.fetch_sub(1, Ordering::Relaxed);
                return Err(Error::Io(std::io::Error::other("transient")));
            }
            self.fills.fetch_add(pages.len(), Ordering::Relaxed);
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(pages
                .iter()
                .map(|&p| {
                    let s = p as usize * self.page;
                    let e = (s + self.page).min(self.len);
                    (s..e).map(byte).collect::<Vec<u8>>().into()
                })
                .collect())
        }

        fn fill_background(&self, pages: Vec<u64>, sink: PageSink, done: BackgroundDone) -> bool {
            if !self.background.load(Ordering::Relaxed) {
                return false;
            }
            let me = Pattern {
                len: self.len,
                page: self.page,
                fills: AtomicUsize::new(0),
                calls: AtomicUsize::new(0),
                fail_first: AtomicUsize::new(0),
                background: AtomicBool::new(false),
            };
            self.fills.fetch_add(pages.len(), Ordering::Relaxed);
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(2));
                done(me.fill_into(&pages, &sink));
            });
            true
        }
    }

    fn pattern(len: usize, page: usize) -> Arc<Pattern> {
        Arc::new(Pattern {
            len,
            page,
            fills: AtomicUsize::new(0),
            calls: AtomicUsize::new(0),
            fail_first: AtomicUsize::new(0),
            background: AtomicBool::new(false),
        })
    }

    #[test]
    fn faults_and_will_need_fill_the_same_bytes_under_a_tiny_budget() {
        // A pager of its own: the budget below would starve other tests.
        let Some(pager) = Pager::spawn(false) else {
            eprintln!("userfaultfd unavailable; skipping");
            return;
        };
        let page = 16 * 1024;
        let lens = [0usize, 1, page - 1, page, page + 1, 300_000, 1 << 20];
        let files: Vec<_> = lens
            .iter()
            .map(|&len| {
                let p = pattern(len, page);
                let input = pager.open(len, page, p.clone(), "f").unwrap();
                (len, p, input)
            })
            .collect();
        pager.set_budget(64 * 1024);
        std::thread::scope(|s| {
            for t in 0..6usize {
                let files = &files;
                s.spawn(move || {
                    for (len, _, input) in files {
                        let bytes = input.bytes();
                        assert_eq!(bytes.len(), *len);
                        if t % 2 == 0 {
                            input.will_need(0, *len).unwrap();
                        }
                        let step = 4096 * (1 + t % 5) + 17;
                        let mut i = (t * 977) % len.max(&1);
                        while i < *len {
                            assert_eq!(bytes[i], byte(i), "[{i}]");
                            i += step;
                        }
                        assert!(bytes.iter().enumerate().all(|(i, b)| *b == byte(i)));
                    }
                });
            }
        });
        pager.set_budget(DEFAULT_BUDGET);
        assert!(files
            .iter()
            .any(|(_, p, _)| p.fills.load(Ordering::Relaxed) > 0));
        assert!(pager.faults() > 0 || files.iter().all(|(l, _, _)| *l == 0));
    }

    #[test]
    fn sequential_faults_read_ahead_and_random_ones_do_not() {
        let Some(pager) = Pager::global() else {
            return;
        };
        let page = 16 * 1024;
        let len = 64 * page;
        let seq = pattern(len, page);
        let input = pager
            .open_with_readahead(len, page, seq.clone(), "seq", 8)
            .unwrap();
        let bytes = input.bytes();
        // A scan in order: windows of 1, 2, 4, 8, 8, ... pages.
        assert!(bytes.iter().enumerate().all(|(i, b)| *b == byte(i)));
        let calls = seq.calls.load(Ordering::Relaxed);
        assert!(calls <= 12, "{calls} fills for 64 pages read in order");
        assert_eq!(seq.fills.load(Ordering::Relaxed), 64);
        // Touches far apart never start a run: one page each.
        let rnd = pattern(len, page);
        let input = pager
            .open_with_readahead(len, page, rnd.clone(), "rnd", 8)
            .unwrap();
        let bytes = input.bytes();
        for p in [40usize, 3, 61, 17, 29] {
            assert_eq!(bytes[p * page + 5], byte(p * page + 5));
        }
        assert_eq!(rnd.fills.load(Ordering::Relaxed), 5);
    }

    #[test]
    fn background_read_ahead_reads_each_page_once() {
        let Some(pager) = Pager::global() else {
            return;
        };
        let page = 16 * 1024;
        let len = 64 * page;
        let seq = pattern(len, page);
        seq.background.store(true, Ordering::Relaxed);
        let input = pager
            .open_with_readahead(len, page, seq.clone(), "bg", 8)
            .unwrap();
        let bytes = input.bytes();
        assert!(bytes.iter().enumerate().all(|(i, b)| *b == byte(i)));
        assert_eq!(seq.fills.load(Ordering::Relaxed), 64);
    }

    #[test]
    fn a_page_announced_again_survives_eviction() {
        let Some(pager) = Pager::spawn(false) else {
            return;
        };
        let page = 16 * 1024;
        let len = 16 * page;
        let p = pattern(len, page);
        let input = pager.open(len, page, p.clone(), "clock").unwrap();
        pager.set_budget(4 * page as u64);
        // Pages 0..4 resident, page 0 the oldest; announcing it again marks it.
        for i in 0..4 {
            input.will_need(i * page, 1).unwrap();
        }
        input.will_need(0, 1).unwrap();
        // Page 4 needs room: page 1, not the announced page 0, goes.
        input.will_need(4 * page, 1).unwrap();
        let before = p.fills.load(Ordering::Relaxed);
        input.will_need(0, 1).unwrap();
        assert_eq!(
            p.fills.load(Ordering::Relaxed),
            before,
            "page 0 was evicted"
        );
        input.will_need(page, 1).unwrap();
        assert_eq!(p.fills.load(Ordering::Relaxed), before + 1, "page 1 stayed");
        assert!(input.bytes().iter().enumerate().all(|(i, b)| *b == byte(i)));
    }

    #[test]
    fn a_page_back_soon_after_eviction_survives_a_scan() {
        let Some(pager) = Pager::spawn(false) else {
            return;
        };
        let page = 16 * 1024;
        let len = 64 * page;
        let p = pattern(len, page);
        let input = pager.open(len, page, p.clone(), "scan").unwrap();
        pager.set_budget(8 * page as u64);
        let bytes = input.bytes();
        // Page 0 is read, pushed out by a scan, and read again: in use.
        assert_eq!(bytes[5], byte(5));
        for i in 1..12 {
            assert_eq!(bytes[i * page], byte(i * page));
        }
        // Eviction runs on the handler threads after each install.
        let waited = std::time::Instant::now();
        while input.file.live[0].load(Ordering::Acquire) || pager.resident_bytes() > pager.budget()
        {
            assert!(
                waited.elapsed() < Duration::from_secs(5),
                "page 0 never evicted"
            );
            std::thread::yield_now();
        }
        assert_eq!(bytes[5], byte(5));
        // A longer scan of pages read once does not push it out again.
        let before = p.fills.load(Ordering::Relaxed);
        for i in 12..64 {
            assert_eq!(bytes[i * page], byte(i * page));
        }
        let scanned = p.fills.load(Ordering::Relaxed);
        assert_eq!(bytes[5], byte(5));
        assert_eq!(
            p.fills.load(Ordering::Relaxed),
            scanned,
            "page 0 was evicted"
        );
        assert!(scanned - before >= 52);
    }

    #[test]
    fn will_need_reports_a_fetch_error_and_a_fault_retries_it() {
        let Some(pager) = Pager::global() else {
            return;
        };
        let page = 16 * 1024;
        let p = pattern(100_000, page);
        let input = pager.open(100_000, page, p.clone(), "f").unwrap();
        p.fail_first.store(1, Ordering::Relaxed);
        assert!(input.will_need(0, 10).is_err());
        // The next fill fails once more; the fault handler retries it.
        p.fail_first.store(1, Ordering::Relaxed);
        assert_eq!(input.bytes()[50_000], byte(50_000));
        // Resident pages are not fetched again.
        input.will_need(0, 100).unwrap();
        let before = p.fills.load(Ordering::Relaxed);
        input.will_need(0, 100).unwrap();
        assert_eq!(p.fills.load(Ordering::Relaxed), before);
        assert!(input.will_need(200_000, 10).is_ok());
        assert!(input.will_need(5, 0).is_ok());
    }

    #[test]
    fn prefetch_by_address_fetches_each_file_s_ranges_in_one_batch() {
        let Some(pager) = Pager::global() else {
            return;
        };
        let page = 16 * 1024;
        let (a, b) = (pattern(200_000, page), pattern(200_000, page));
        let fa = pager.open(200_000, page, a.clone(), "a").unwrap();
        let fb = pager.open(200_000, page, b.clone(), "b").unwrap();
        let (x, y) = (fa.bytes(), fb.bytes());
        let heap = vec![1u8; 10];
        assert!(is_paged(x) && is_paged(&y[100..]));
        assert!(!is_paged(&heap) && !is_paged(&[]));
        prefetch(&[&x[..1], &x[100_000..150_000], &y[190_000..], &heap, &[]]).unwrap();
        // Pages 0 and 6..=9 of `a`, 11..=12 of `b`: one fill call each.
        assert_eq!(a.fills.load(Ordering::Relaxed), 5);
        assert_eq!(a.calls.load(Ordering::Relaxed), 1);
        assert_eq!(b.fills.load(Ordering::Relaxed), 2);
        assert_eq!(b.calls.load(Ordering::Relaxed), 1);
        assert_eq!(x[120_000], byte(120_000));
        assert_eq!(y[199_999], byte(199_999));
        // A fetch error comes back to the caller.
        b.fail_first.store(1, Ordering::Relaxed);
        assert!(prefetch(&[&y[50_000..50_001]]).is_err());
        assert_eq!(y[50_000], byte(50_000));
    }

    #[test]
    fn pages_must_be_whole_os_pages() {
        let Some(pager) = Pager::global() else {
            return;
        };
        assert!(!pager.supports(100));
        assert!(pager.open(10, 100, pattern(10, 100), "f").is_err());
        assert!(pager.resident_bytes() <= DEFAULT_BUDGET);
    }

    #[test]
    fn a_user_mode_only_pager_serves_user_reads() {
        let Some(pager) = Pager::spawn(true) else {
            return;
        };
        let page = 16 * 1024;
        let p = pattern(200_000, page);
        let input = pager.open(200_000, page, p, "f").unwrap();
        assert!(input.bytes().iter().enumerate().all(|(i, b)| *b == byte(i)));
    }
}
