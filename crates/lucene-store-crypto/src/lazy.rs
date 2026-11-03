//! Demand-decrypted inputs: an encrypted file opened as a mapping whose
//! pages are decrypted on first touch and dropped again past a memory
//! budget, so an open encrypted index costs its working set in RAM rather
//! than its size.
//!
//! The file's plaintext range is reserved as an anonymous mapping and
//! registered with Linux `userfaultfd`. The first access to a missing page
//! blocks the faulting thread; a handler thread decrypts the 64 KiB chunk
//! holding it from the ciphertext (itself a page-cache mapping for a
//! filesystem directory) and installs the whole chunk with `UFFDIO_COPY`.
//! When resident chunks exceed the budget, the oldest are released with
//! `MADV_DONTNEED`, and the next access faults them in again. Readers see a
//! plain `&[u8]` throughout, so codecs read it exactly as they read a
//! memory-mapped plaintext file.
//!
//! Every chunk is authenticated when the file is opened, so a wrong key or
//! a tampered file fails `open` exactly as a whole-file decryption does. A
//! chunk that later fails to decrypt means the ciphertext changed under an
//! open reader; a memory access cannot return an error, so the process
//! aborts rather than hand a reader unauthenticated bytes.
//!
//! [`Pager::global`] is `None` where `userfaultfd` is unavailable (not
//! Linux, a seccomp policy that blocks it, or a chunk size that is not a
//! whole number of pages); callers then decrypt whole files.

#![allow(unsafe_code)]

use std::collections::{BTreeMap, VecDeque};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock, Weak};

use lucene_store::directory::Input;
use lucene_store::{Error, Result};

use crate::format::{ChunkCipher, CHUNK_OVERHEAD, TAG_LEN};

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

/// Bytes resident across every demand-decrypted file before the oldest
/// chunks are released, until [`Pager::set_budget`] changes it.
pub const DEFAULT_BUDGET: u64 = 512 * 1024 * 1024;

/// The process-wide fault handler: one `userfaultfd`, one handler thread,
/// every registered file, and the resident-chunk budget.
pub struct Pager {
    uffd: OwnedFd,
    page: usize,
    files: RwLock<BTreeMap<usize, Weak<File>>>,
    resident: Mutex<VecDeque<(Weak<File>, usize)>>,
    resident_bytes: AtomicU64,
    budget: AtomicU64,
    faults: AtomicU64,
}

/// One registered file: its mapping, the ciphertext it is decrypted from,
/// and the cipher.
struct File {
    base: usize,
    map_len: usize,
    plain_len: usize,
    chunk: usize,
    chunks: usize,
    sealed: Input,
    body_start: usize,
    cipher: ChunkCipher,
    name: String,
    pager: &'static Pager,
    /// Chunks resident now, so eviction never releases one twice.
    live: Vec<AtomicUsize>,
}

/// A demand-decrypted file's plaintext, as an [`Input::external`] source.
pub struct LazyPlain {
    file: Arc<File>,
    pager: &'static Pager,
}

impl AsRef<[u8]> for LazyPlain {
    fn as_ref(&self) -> &[u8] {
        if self.file.plain_len == 0 {
            return &[];
        }
        // SAFETY: `base..base + map_len` is a live private mapping owned by
        // `self.file` until `Drop`, and `plain_len <= map_len`. Its contents
        // are only ever written by `UFFDIO_COPY` with the authenticated
        // plaintext of the same chunk, so readers always see the same bytes.
        unsafe { std::slice::from_raw_parts(self.file.base as *const u8, self.file.plain_len) }
    }
}

impl Drop for LazyPlain {
    fn drop(&mut self) {
        if self.file.map_len == 0 {
            return;
        }
        if let Ok(mut files) = self.pager.files.write() {
            files.remove(&self.file.base);
        }
    }
}

impl Drop for File {
    /// Unmaps once nothing, the handler thread included, holds the file.
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
            .filter(|c| c.load(Ordering::Relaxed) != 0)
            .count();
        let bytes = (live as u64).saturating_mul(self.chunk as u64);
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

    /// Starts a pager and its handler thread. Both live for the rest of the
    /// process.
    fn spawn(user_mode_only: bool) -> Option<&'static Pager> {
        let pager = Pager::start(user_mode_only).ok()?;
        let pager: &'static Pager = Box::leak(Box::new(pager));
        std::thread::Builder::new()
            .name("lucene-encrypted-pager".into())
            .spawn(move || pager.serve())
            .ok()?;
        Some(pager)
    }

    fn start(user_mode_only: bool) -> std::io::Result<Pager> {
        // Full mode first (kernel accesses fault in too); user-mode-only is
        // what an unprivileged process gets when vm.unprivileged_userfaultfd
        // is 0.
        let full = libc::O_CLOEXEC;
        let user = libc::O_CLOEXEC | UFFD_USER_MODE_ONLY;
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
        let page = usize::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) })
            .map_err(|_| std::io::Error::other("page size"))?;
        Ok(Pager {
            uffd,
            page,
            files: RwLock::new(BTreeMap::new()),
            resident: Mutex::new(VecDeque::new()),
            resident_bytes: AtomicU64::new(0),
            budget: AtomicU64::new(DEFAULT_BUDGET),
            faults: AtomicU64::new(0),
        })
    }

    /// Whether a file with `chunk_size`-byte chunks can be demand-decrypted:
    /// each chunk must be a whole number of pages.
    pub fn supports(&self, chunk_size: u32) -> bool {
        let chunk = chunk_size as usize;
        chunk >= self.page && chunk.is_multiple_of(self.page)
    }

    /// Sets the resident budget in bytes; chunks past it are released on
    /// the next fault.
    pub fn set_budget(&self, bytes: u64) {
        self.budget.store(bytes, Ordering::Relaxed);
    }

    /// Plaintext bytes resident now, across every file.
    pub fn resident_bytes(&self) -> u64 {
        self.resident_bytes.load(Ordering::Relaxed)
    }

    /// Chunk faults served since the pager started.
    pub fn faults(&self) -> u64 {
        self.faults.load(Ordering::Relaxed)
    }

    /// Opens the encrypted body `sealed[body_start..]` (already
    /// authenticated) as a demand-decrypted input of `plain_len` bytes.
    pub fn open(
        &'static self,
        sealed: Input,
        body_start: usize,
        plain_len: usize,
        chunk_size: u32,
        cipher: ChunkCipher,
        name: &str,
    ) -> Result<Input> {
        let chunk = chunk_size as usize;
        let body_len = sealed.len().saturating_sub(body_start);
        let per_chunk = chunk.saturating_add(CHUNK_OVERHEAD);
        let chunks = body_len.div_ceil(per_chunk).max(1);
        let map_len = plain_len.div_ceil(self.page).saturating_mul(self.page);
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
            plain_len,
            chunk,
            chunks,
            sealed,
            body_start,
            cipher,
            name: name.to_string(),
            pager: self,
            live: (0..chunks).map(|_| AtomicUsize::new(0)).collect(),
        });
        if map_len > 0 {
            self.files
                .write()
                .map_err(|_| Error::Io(std::io::Error::other("pager registry poisoned")))?
                .insert(base, Arc::downgrade(&file));
        }
        Ok(Input::external(LazyPlain { file, pager: self }))
    }

    /// The handler loop: reads fault messages and resolves each.
    fn serve(&'static self) {
        let fd = self.uffd.as_raw_fd();
        let mut msgs = [UffdMsg {
            event: 0,
            _reserved: [0; 7],
            flags: 0,
            address: 0,
            _feat: 0,
        }; 16];
        let mut buf: Vec<u8> = Vec::new();
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
            // SAFETY: reads whole `uffd_msg` records into `msgs`.
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
                    self.resolve(m.address as usize, &mut buf);
                }
            }
        }
    }

    /// Decrypts and installs the chunk holding `addr`.
    fn resolve(&'static self, addr: usize, buf: &mut Vec<u8>) {
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
        let Some(off) = addr.checked_sub(file.base) else {
            return;
        };
        let Some(index) = off.checked_div(file.chunk) else {
            return;
        };
        if index >= file.chunks {
            return;
        }
        let need = file.chunk.saturating_add(TAG_LEN);
        if buf.len() < need {
            buf.resize(need, 0);
        }
        let body = &file.sealed[file.body_start..];
        let opened = file.cipher.open_chunk(
            body,
            file.chunk as u32,
            index,
            file.chunks,
            &mut buf[..need],
            &file.name,
        );
        if let Err(e) = opened {
            eprintln!(
                "lucene-store-crypto: {e}: the ciphertext changed under an open reader; aborting"
            );
            std::process::abort();
        }
        let dst = file.base.saturating_add(index.saturating_mul(file.chunk));
        let len = file.chunk.min(
            file.map_len
                .saturating_sub(index.saturating_mul(file.chunk)),
        );
        let mut copy = UffdioCopy {
            dst: dst as u64,
            src: buf.as_ptr() as u64,
            len: len as u64,
            mode: 0,
            copy: 0,
        };
        // SAFETY: `dst..dst + len` lies inside the registered mapping and
        // `buf` holds at least `len` bytes (the tail past the plaintext is
        // whatever the buffer held, never read: `plain_len` bounds readers).
        let rc = unsafe { libc::ioctl(self.uffd.as_raw_fd(), UFFDIO_COPY, &mut copy) };
        if rc != 0 {
            // Already installed by an earlier message for the same chunk:
            // wake whoever is waiting on this page.
            let mut range = UffdioRange {
                start: (addr & !self.page.saturating_sub(1)) as u64,
                len: self.page as u64,
            };
            // SAFETY: wakes waiters on a page of the registered mapping.
            unsafe { libc::ioctl(self.uffd.as_raw_fd(), UFFDIO_WAKE, &mut range) };
            return;
        }
        self.faults.fetch_add(1, Ordering::Relaxed);
        if file.live[index].swap(1, Ordering::AcqRel) == 0 {
            self.resident_bytes
                .fetch_add(file.chunk as u64, Ordering::Relaxed);
            if let Ok(mut q) = self.resident.lock() {
                q.push_back((Arc::downgrade(&file), index));
            }
        }
        self.evict();
    }

    /// Releases the oldest resident chunks until the budget holds.
    fn evict(&self) {
        let budget = self.budget.load(Ordering::Relaxed);
        let Ok(mut q) = self.resident.lock() else {
            return;
        };
        while self.resident_bytes.load(Ordering::Relaxed) > budget {
            let Some((w, index)) = q.pop_front() else {
                break;
            };
            let Some(file) = w.upgrade() else {
                continue;
            };
            if file.live[index].swap(0, Ordering::AcqRel) == 0 {
                continue;
            }
            let start = file.base.saturating_add(index.saturating_mul(file.chunk));
            let len = file.chunk.min(
                file.map_len
                    .saturating_sub(index.saturating_mul(file.chunk)),
            );
            // SAFETY: drops the pages of one chunk of a live registered
            // mapping; the next access faults it back in, identical.
            unsafe { libc::madvise(start as *mut libc::c_void, len, libc::MADV_DONTNEED) };
            let _ = self
                .resident_bytes
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |r| {
                    Some(r.saturating_sub(file.chunk as u64))
                });
        }
    }
}

#[cfg(test)]
mod tests {
    // Test fixtures' own arithmetic -- see `docs/arithmetic-gate.md`'s "Test code".
    #![allow(clippy::arithmetic_side_effects)]
    use super::*;
    use crate::{EncryptedDirectory, KeyProvider, StaticKeyProvider};
    use lucene_store::{ByteBuffersDirectory, DataOutput, Directory};

    fn plain(len: usize, seed: u64) -> Vec<u8> {
        let mut x = seed | 1;
        (0..len)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x >> 24) as u8
            })
            .collect()
    }

    fn dir(chunk: u32) -> EncryptedDirectory<ByteBuffersDirectory> {
        let keys: Arc<dyn KeyProvider> = Arc::new(StaticKeyProvider::new("k", &[3; 32]));
        EncryptedDirectory::new(ByteBuffersDirectory::new(), keys)
            .with_chunk_size(chunk)
            .unwrap()
            .with_lazy_decrypt(true)
    }

    fn put(d: &dyn Directory, name: &str, bytes: &[u8]) {
        let mut out = d.create_output(name).unwrap();
        out.write_bytes(bytes);
        out.close().unwrap();
    }

    /// Files are read back through the pager at every tail length, from many
    /// threads at once, while a budget far below the data keeps evicting:
    /// every byte read is the plaintext written.
    #[test]
    fn concurrent_reads_under_a_tiny_budget_see_the_plaintext() {
        let Some(pager) = Pager::global() else {
            eprintln!("userfaultfd unavailable; skipping");
            return;
        };
        let d = dir(4096);
        let files: Vec<(String, Vec<u8>)> = [0usize, 1, 4095, 4096, 4097, 300_000, 1 << 20]
            .iter()
            .enumerate()
            .map(|(i, &len)| (format!("f{i}"), plain(len, i as u64 + 7)))
            .collect();
        for (name, bytes) in &files {
            put(&d, name, bytes);
        }
        let before = pager.faults();
        pager.set_budget(64 * 1024);
        std::thread::scope(|s| {
            for t in 0..8u64 {
                let d = &d;
                let files = &files;
                s.spawn(move || {
                    for round in 0..3 {
                        for (name, bytes) in files {
                            let input = d.open(name).unwrap();
                            assert_eq!(input.len(), bytes.len());
                            // Strided reads, then a full compare.
                            let step = 4096 * (1 + ((t + round) as usize % 5)) + 17;
                            let mut i = (t as usize * 977) % bytes.len().max(1);
                            while i < bytes.len() {
                                assert_eq!(input[i], bytes[i], "{name}[{i}]");
                                i += step;
                            }
                            assert!(input[..] == bytes[..], "{name}");
                        }
                    }
                });
            }
        });
        pager.set_budget(DEFAULT_BUDGET);
        assert!(pager.faults() > before, "reads went through the pager");
    }

    /// Under a user-mode-only `userfaultfd` the kernel cannot fault pages
    /// in: a `write(2)` straight from a page not yet resident fails with
    /// `EFAULT`. A filesystem output copies such bytes through a buffer, so
    /// copying a demand-decrypted file out to a plain directory works.
    #[test]
    fn a_user_mode_only_pager_still_copies_out_through_a_file_output() {
        let Some(pager) = Pager::spawn(true) else {
            return;
        };
        // Two chunks resident at most: by the time the output's checksum
        // pass reaches the end, the start has been released again.
        pager.set_budget(2 * 4096);
        let d = dir(4096);
        let bytes = plain(1 << 20, 5);
        put(&d, "f", &bytes);
        let raw = d.inner().open("f").unwrap();
        let (cipher, header, header_len) = d.cipher_for("f", &raw).unwrap();
        let input = pager
            .open(raw, header_len, bytes.len(), header.chunk_size, cipher, "f")
            .unwrap();
        let tmp = lucene_util::test_support::TempDir::new("lazy-efault");
        let direct = std::fs::write(tmp.path().join("direct"), &input[..]);
        let e = direct.expect_err("the kernel cannot fault pages in");
        assert_eq!(e.raw_os_error(), Some(libc::EFAULT), "{e}");
        drop(input);
        let raw = d.inner().open("f").unwrap();
        let (cipher, header, header_len) = d.cipher_for("f", &raw).unwrap();
        let input = pager
            .open(raw, header_len, bytes.len(), header.chunk_size, cipher, "f")
            .unwrap();
        let fs = lucene_store::FsDirectory::open(tmp.path());
        let mut out = fs.create_output("copied").unwrap();
        out.write_bytes(&input);
        out.close().unwrap();
        assert!(std::fs::read(tmp.path().join("copied")).unwrap() == bytes);
    }

    /// A tampered chunk anywhere in the file fails `open`, as a whole-file
    /// decryption does, before any byte is handed out.
    #[test]
    fn tampering_fails_open() {
        if Pager::global().is_none() {
            return;
        }
        let d = dir(4096);
        let bytes = plain(100_000, 99);
        put(&d, "f", &bytes);
        let mut raw = d.inner().open("f").unwrap().to_vec();
        let at = raw.len() - 5000;
        raw[at] ^= 1;
        d.inner().delete_file("f").unwrap();
        put(d.inner(), "f", &raw);
        let err = d.open("f").unwrap_err();
        assert!(format!("{err}").contains("failed authentication"), "{err}");
    }
}
