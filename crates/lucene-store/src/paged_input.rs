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
//! When resident pages exceed the budget, the oldest are released with
//! `MADV_DONTNEED`, and the next access faults them in again. Readers see a
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
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock, Weak};
use std::time::Duration;

use crate::directory::ExternalBytes;
use crate::error::{Error, Result};
use crate::lock::lock_ignoring_poison;

/// Supplies a demand-filled file's pages.
pub(crate) trait PageFiller: Send + Sync {
    /// The decoded bytes of each of `pages` (sorted, distinct), each
    /// `page_len` bytes but the file's last, which may be shorter.
    fn fill(&self, pages: &[u64]) -> Result<Vec<Arc<[u8]>>>;
}

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

/// The process-wide fault handler: one `userfaultfd`, its handler threads,
/// every registered file, and the resident budget.
pub struct Pager {
    uffd: OwnedFd,
    os_page: usize,
    files: RwLock<BTreeMap<usize, Weak<File>>>,
    resident: Mutex<VecDeque<(Weak<File>, usize)>>,
    resident_bytes: AtomicU64,
    budget: AtomicU64,
    faults: AtomicU64,
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
        let f = &self.file;
        let end = offset.saturating_add(len).min(f.len);
        if offset >= end {
            return Ok(());
        }
        let first = offset.checked_div(f.page).unwrap_or(0);
        let last = end.saturating_sub(1).checked_div(f.page).unwrap_or(0);
        let missing: Vec<u64> = (first..=last)
            .filter(|&p| !f.live[p].load(Ordering::Acquire))
            .map(|p| p as u64)
            .collect();
        if missing.is_empty() {
            return Ok(());
        }
        let filled = f.filler.fill(&missing)?;
        for (&page, bytes) in missing.iter().zip(&filled) {
            f.pager.install(&self.file, page as usize, bytes);
        }
        f.pager.evict();
        Ok(())
    }
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
        static PAGER: OnceLock<Option<&'static Pager>> = OnceLock::new();
        *PAGER.get_or_init(|| Pager::spawn(false))
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
            resident: Mutex::new(VecDeque::new()),
            resident_bytes: AtomicU64::new(0),
            budget: AtomicU64::new(DEFAULT_BUDGET),
            faults: AtomicU64::new(0),
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

    /// Bytes resident now, across every file.
    pub fn resident_bytes(&self) -> u64 {
        self.resident_bytes.load(Ordering::Relaxed)
    }

    /// Page faults served since the pager started.
    pub fn faults(&self) -> u64 {
        self.faults.load(Ordering::Relaxed)
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
        loop {
            let mut pfd = libc::pollfd {
                fd,
                events: libc::POLLIN,
                revents: 0,
            };
            // SAFETY: one valid pollfd.
            if unsafe { libc::poll(&mut pfd, 1, -1) } < 0 {
                continue;
            }
            // SAFETY: reads whole `uffd_msg` records into `msgs`. The fd is
            // non-blocking, so a message another thread took first returns
            // EAGAIN here.
            let n =
                unsafe { libc::read(fd, msgs.as_mut_ptr().cast(), std::mem::size_of_val(&msgs)) };
            if n <= 0 {
                continue;
            }
            let count = (n as usize)
                .checked_div(std::mem::size_of::<UffdMsg>())
                .unwrap_or(0)
                .min(msgs.len());
            for m in &msgs[..count] {
                if m.event == UFFD_EVENT_PAGEFAULT {
                    self.resolve(m.address as usize);
                }
            }
        }
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
        let mut attempt = 0usize;
        let bytes = loop {
            match file.filler.fill(&[index as u64]) {
                Ok(mut pages) if pages.len() == 1 => break pages.swap_remove(0),
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
        };
        self.faults.fetch_add(1, Ordering::Relaxed);
        if !self.install(&file, index, &bytes) {
            // Installed by another thread meanwhile: wake whoever waits on
            // the faulting page.
            let mut range = UffdioRange {
                start: (addr & !self.os_page.saturating_sub(1)) as u64,
                len: self.os_page as u64,
            };
            // SAFETY: wakes waiters on a page of the registered mapping.
            unsafe { libc::ioctl(self.uffd.as_raw_fd(), UFFDIO_WAKE, &mut range) };
        }
        self.evict();
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
            self.resident_bytes
                .fetch_add(file.page as u64, Ordering::Relaxed);
            lock_ignoring_poison(&self.resident).push_back((Arc::downgrade(file), index));
        }
        installed
    }

    /// Releases the oldest resident pages until the budget holds.
    fn evict(&self) {
        let budget = self.budget.load(Ordering::Relaxed);
        let mut q = lock_ignoring_poison(&self.resident);
        while self.resident_bytes.load(Ordering::Relaxed) > budget {
            let Some((w, index)) = q.pop_front() else {
                break;
            };
            let Some(file) = w.upgrade() else {
                continue;
            };
            if !file.live[index].swap(false, Ordering::AcqRel) {
                continue;
            }
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
        fail_first: AtomicUsize,
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
            Ok(pages
                .iter()
                .map(|&p| {
                    let s = p as usize * self.page;
                    let e = (s + self.page).min(self.len);
                    (s..e).map(byte).collect::<Vec<u8>>().into()
                })
                .collect())
        }
    }

    fn pattern(len: usize, page: usize) -> Arc<Pattern> {
        Arc::new(Pattern {
            len,
            page,
            fills: AtomicUsize::new(0),
            fail_first: AtomicUsize::new(0),
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
