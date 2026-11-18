//! Port of `org.apache.lucene.store.Directory` / `FSDirectory` / `MMapDirectory`,
//! plus the generation-lookup logic from `org.apache.lucene.index.SegmentInfos`
//! (`getLastCommitGeneration`, `generationFromSegmentsFileName`) that depends only
//! on a file listing.
//!
//! Two backends, one trait:
//! - [`FsDirectory`]: `std::fs::read` — safe, no `unsafe`, always correct. Default.
//! - [`MmapDirectory`]: `memmap2` — zero-copy reads matching Lucene's own default
//!   (`MMapDirectory`) for real workloads. Contains one of this crate's two
//!   `unsafe` sites (the other is `fs_lock_factory`'s `fcntl`), documented on
//!   the call site: mapping a file is only sound if nothing else
//!   truncates/mutates it concurrently, same caveat Lucene's own Javadoc carries.
//!
//! Both return an [`Input`] — an owned-or-mapped byte buffer that `Deref`s to
//! `&[u8]`, so callers (codec_util, segment_info, segment_infos) are unchanged
//! regardless of backend.

use std::collections::BTreeSet;
use std::fs;
use std::io::Read;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::data_output::DataOutput;
use crate::error::{Error, Result};
use crate::fs_lock_factory::default_fs_lock_factory;
#[cfg(doc)]
use crate::fs_lock_factory::NativeFsLockFactory;
use crate::index_output::{self, FsIndexOutput};
use crate::lock::{Lock, LockFactory};

/// The `segments` file-name prefix (`IndexFileNames.SEGMENTS`). Excludes the
/// pre-4.0 `segments.gen` pointer file, which is not a valid commit file name.
const SEGMENTS_PREFIX: &str = "segments";
const OLD_SEGMENTS_GEN: &str = "segments.gen";
/// `IndexFileNames.PENDING_SEGMENTS`: the name a `segments_N` is written
/// under before it is renamed into place. Deliberately *not* prefixed with
/// [`SEGMENTS_PREFIX`], so a half-written commit file can never be picked up
/// by [`last_commit_generation`] -- that invisibility is the whole point of
/// Java's two-phase `prepareCommit`/`finishCommit` protocol.
const PENDING_SEGMENTS_PREFIX: &str = "pending_segments";

/// A file's bytes, however the backend obtained them.
pub enum Input {
    Owned(Vec<u8>),
    Mapped(memmap2::Mmap),
    /// Shared with the directory that holds them in memory
    /// ([`crate::ByteBuffersDirectory`]): opening costs a reference count,
    /// not a copy -- Java's `IndexInput.clone()` of the stored content.
    Shared(Arc<[u8]>),
    /// `bytes[start..start + len]` of another input, sharing it: a compound
    /// file's member (`Lucene90CompoundReader.openInput`'s `slice`), which
    /// costs a reference count on the archive rather than a copy of the
    /// member. Built by [`Input::slice_of`], which checks the bounds.
    Slice {
        base: Arc<Input>,
        start: usize,
        len: usize,
    },
    /// Bytes a third-party [`Directory`] produced and owns its own way: a
    /// blob store's downloaded object kept in a pooled buffer, a decrypted
    /// file in an anonymous mapping, a local cache file's mapping. Opening
    /// costs whatever the directory chose; reading is a slice like every
    /// other input. Build it with [`Input::external`], or with
    /// [`Input::external_bytes`] for bytes that act on
    /// [`Input::will_need`] (a demand-filled mapping).
    External(Box<dyn ExternalBytes>),
    /// Another input held open by a test harness (feature `test-util`): the
    /// guard is dropped with the bytes, which is how the test framework's
    /// `MockDirectoryWrapper` sees an input "closed".
    #[cfg(any(test, feature = "test-util"))]
    Tracked(TrackedInput),
}

/// An [`Input`] with a guard whose `Drop` marks it closed (feature
/// `test-util`): the test framework's `MockIndexInputWrapper`, reduced to
/// the open/close bookkeeping, since an input here is a byte buffer and has
/// no reads to intercept.
#[cfg(any(test, feature = "test-util"))]
pub struct TrackedInput {
    input: Box<Input>,
    _guard: Box<dyn Send + Sync>,
}

#[cfg(any(test, feature = "test-util"))]
impl TrackedInput {
    /// Wraps `input`; `guard` is dropped when the returned input is.
    pub fn new(input: Input, guard: impl Send + Sync + 'static) -> Self {
        Self {
            input: Box::new(input),
            _guard: Box::new(guard),
        }
    }

    /// The wrapped input.
    pub fn inner(&self) -> &Input {
        &self.input
    }
}

impl std::fmt::Debug for Input {
    /// Length and provenance only. The alternative is dumping a
    /// half-gigabyte mapping into a panic message.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kind = match self {
            Input::Owned(_) => "Owned",
            Input::Mapped(_) => "Mapped",
            Input::Shared(_) => "Shared",
            Input::Slice { .. } => "Slice",
            Input::External(_) => "External",
            #[cfg(any(test, feature = "test-util"))]
            Input::Tracked(_) => "Tracked",
        };
        write!(f, "Input::{kind}({} bytes)", self.len())
    }
}

impl Deref for Input {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &[u8] {
        match self {
            Input::Owned(v) => v,
            Input::Mapped(m) => m,
            Input::Shared(s) => s,
            // In bounds: `slice_of` checked `start + len <= base.len()`, and
            // `base` cannot change length behind the `Arc`.
            Input::Slice { base, start, len } => &base[*start..][..*len],
            Input::External(bytes) => bytes.bytes(),
            #[cfg(any(test, feature = "test-util"))]
            Input::Tracked(t) => &t.input,
        }
    }
}

/// The bytes behind an [`Input::External`]: a slice, plus a hint that a
/// range will be read soon.
pub trait ExternalBytes: Send + Sync {
    /// The bytes.
    fn bytes(&self) -> &[u8];

    /// Makes `offset..offset + len` resident before it is read, returning
    /// any error fetching it. The default does nothing: the bytes are
    /// already in memory. A demand-filled input fetches the range here, so
    /// a failed download reaches the caller as an error instead of at a
    /// later page fault.
    fn will_need(&self, offset: usize, len: usize) -> Result<()> {
        let _ = (offset, len);
        Ok(())
    }
}

/// [`Input::external`]'s plain bytes: nothing to fetch.
struct PlainBytes<T>(T);

impl<T: AsRef<[u8]> + Send + Sync> ExternalBytes for PlainBytes<T> {
    fn bytes(&self) -> &[u8] {
        self.0.as_ref()
    }
}

impl Input {
    /// An [`Input::External`] over `bytes`, which the input owns and drops
    /// when it is closed.
    pub fn external(bytes: impl AsRef<[u8]> + Send + Sync + 'static) -> Input {
        Input::External(Box::new(PlainBytes(bytes)))
    }

    /// An [`Input::External`] over bytes that act on [`Input::will_need`].
    pub fn external_bytes(bytes: impl ExternalBytes + 'static) -> Input {
        Input::External(Box::new(bytes))
    }

    /// Hints that `offset..offset + len` (clamped to the input) will be read
    /// soon: Lucene's `IndexInput.prefetch`. A mapping is advised
    /// `MADV_WILLNEED`; a demand-filled input ([`ExternalBytes`]) fetches
    /// the range now and reports a failure to fetch it. Bytes already in
    /// memory ignore it.
    pub fn will_need(&self, offset: usize, len: usize) -> Result<()> {
        let end = offset.saturating_add(len).min(self.len());
        if offset >= end {
            return Ok(());
        }
        let len = end.saturating_sub(offset);
        match self {
            Input::Owned(_) | Input::Shared(_) => Ok(()),
            #[cfg(unix)]
            Input::Mapped(m) => {
                // Advice only: a failure (an old kernel) changes nothing a
                // reader can see.
                if let Some(bytes) = m.get(offset..end) {
                    advise_will_need(bytes);
                }
                Ok(())
            }
            #[cfg(not(unix))]
            Input::Mapped(_) => Ok(()),
            Input::Slice { base, start, .. } => base.will_need(start.saturating_add(offset), len),
            Input::External(bytes) => bytes.will_need(offset, len),
            #[cfg(any(test, feature = "test-util"))]
            Input::Tracked(t) => t.input.will_need(offset, len),
        }
    }

    /// Whether the kernel's page cache holds `offset..offset + len` (clamped
    /// to the input) of a mapped input now (`mincore`). Inputs that are not
    /// mappings are in memory and answer `true`.
    pub fn is_resident(&self, offset: usize, len: usize) -> bool {
        let end = offset.saturating_add(len).min(self.len());
        if offset >= end {
            return true;
        }
        match self {
            #[cfg(target_os = "linux")]
            Input::Mapped(m) => m.get(offset..end).is_none_or(pages_resident),
            Input::Slice { base, start, .. } => {
                base.is_resident(start.saturating_add(offset), end.saturating_sub(offset))
            }
            #[cfg(any(test, feature = "test-util"))]
            Input::Tracked(t) => t.input.is_resident(offset, len),
            _ => true,
        }
    }

    /// Advises the kernel that a mapped input is read at random, so a page
    /// fault reads that page alone instead of the read-around window
    /// (`MADV_RANDOM`; Lucene's `ReadAdvice.RANDOM`). For files a search
    /// reads a few hundred bytes at a time at scattered offsets, the vectors
    /// a graph walk or a rerank visits: under memory pressure the default
    /// window reads mostly bytes nobody asked for and evicts the ones that
    /// are needed. A compound file's member advises its own range of the
    /// archive. Inputs that are not mappings ignore it; so does a failure
    /// (an advice changes no bytes a reader sees).
    pub fn advise_random(&self) {
        self.advise_random_range(0, self.len());
    }

    /// [`Self::advise_random`], and when the input is small next to the
    /// memory this process may use ([`memory_available`]: at most an
    /// eighth of it), [`Self::will_need`] over all of it as well. Random
    /// advice stops a fault reading ahead, so a file that would fit in the
    /// page cache would otherwise warm up one page per fault after a cold
    /// start; reading it in once, in large reads, keeps that cost to what
    /// the default advice pays. A file too large for that is left to
    /// demand faults, which is where random advice pays.
    pub fn advise_random_and_warm(&self) {
        self.advise_random();
        let len = self.len() as u64;
        if len > 0 && memory_available().is_some_and(|avail| len <= avail / 8) {
            let _ = self.will_need(0, self.len());
        }
    }

    /// Applies `advice` ([`crate::read_advice`]) to a mapped input:
    /// [`Self::advise_random_and_warm`] for random advice (without the
    /// warm-up when [`crate::read_advice::warm_random`] is off),
    /// `MADV_SEQUENTIAL` for sequential, [`Self::will_need`] over all of it
    /// for will-need, nothing for normal; auto advice is first resolved
    /// against the input's length ([`crate::read_advice::resolve`]) and,
    /// resolved to random, never warmed. Returns the advice applied.
    pub fn advise(&self, advice: crate::read_advice::ReadAdvice) -> crate::read_advice::ReadAdvice {
        use crate::read_advice::ReadAdvice;
        let resolved = crate::read_advice::resolve(advice, self.len() as u64);
        match (advice, resolved) {
            (ReadAdvice::Auto, ReadAdvice::Random) => self.advise_random(),
            (_, ReadAdvice::Normal | ReadAdvice::Auto) => {}
            (_, ReadAdvice::Random) if crate::read_advice::warm_random() => {
                self.advise_random_and_warm()
            }
            (_, ReadAdvice::Random) => self.advise_random(),
            #[cfg(unix)]
            (_, ReadAdvice::Sequential) => {
                self.advise_range(memmap2::Advice::Sequential, 0, self.len())
            }
            #[cfg(not(unix))]
            (_, ReadAdvice::Sequential) => {}
            (_, ReadAdvice::WillNeed) => {
                let _ = self.will_need(0, self.len());
            }
        }
        resolved
    }

    fn advise_random_range(&self, offset: usize, len: usize) {
        #[cfg(unix)]
        self.advise_range(memmap2::Advice::Random, offset, len);
        #[cfg(not(unix))]
        let _ = (offset, len);
    }

    #[cfg(unix)]
    fn advise_range(&self, advice: memmap2::Advice, offset: usize, len: usize) {
        match self {
            Input::Mapped(m) if len > 0 => {
                let _ = m.advise_range(advice, offset, len);
            }
            Input::Slice { base, start, .. } => {
                base.advise_range(advice, start.saturating_add(offset), len)
            }
            #[cfg(any(test, feature = "test-util"))]
            Input::Tracked(t) => t.input.advise_range(advice, offset, len),
            _ => {}
        }
    }

    /// `len` bytes of `base` from `start`, sharing `base` rather than copying
    /// them ([`Input::Slice`]). Out of bounds is corruption, as Java's
    /// `IndexInput.slice` reports it.
    ///
    /// A test harness's tracked input ([`Input::Tracked`]) is copied out
    /// instead: its guard models Java's `close()`, which closes a compound
    /// file's slices with it, and a window sharing it would keep the file
    /// "open" past its owner.
    pub fn slice_of(base: &Arc<Input>, start: usize, len: usize) -> Result<Input> {
        match start.checked_add(len) {
            #[cfg(any(test, feature = "test-util"))]
            Some(end) if end <= base.len() && matches!(**base, Input::Tracked(_)) => {
                Ok(Input::Owned(base[start..end].to_vec()))
            }
            Some(end) if end <= base.len() => Ok(Input::Slice {
                base: Arc::clone(base),
                start,
                len,
            }),
            _ => Err(Error::Corrupted(format!(
                "slice(offset={start}, length={len}) is out of bounds of a {}-byte input",
                base.len()
            ))),
        }
    }
}

/// Bytes this process can still fill without reclaiming its own memory, as
/// far as Linux says: `MemAvailable` from `/proc/meminfo`, capped by the
/// memory cgroup's limit (v2 `memory.max`, else v1
/// `memory.limit_in_bytes`) when one is set. `None` elsewhere.
pub fn memory_available() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let meminfo = fs::read_to_string("/proc/meminfo").ok()?;
        let avail_kb: u64 = meminfo
            .lines()
            .find_map(|l| l.strip_prefix("MemAvailable:"))?
            .split_whitespace()
            .next()?
            .parse()
            .ok()?;
        let mut avail = avail_kb.saturating_mul(1024);
        if let Some(limit) = cgroup_memory_limit() {
            avail = avail.min(limit);
        }
        Some(avail)
    }
    #[cfg(not(target_os = "linux"))]
    None
}

/// This process's memory cgroup limit, if it has one.
#[cfg(target_os = "linux")]
fn cgroup_memory_limit() -> Option<u64> {
    let cgroups = fs::read_to_string("/proc/self/cgroup").ok()?;
    for line in cgroups.lines() {
        let mut parts = line.splitn(3, ':');
        let (_, controllers, path) = (parts.next()?, parts.next()?, parts.next()?);
        let file = if controllers.is_empty() {
            format!("/sys/fs/cgroup{path}/memory.max")
        } else if controllers.split(',').any(|c| c == "memory") {
            format!("/sys/fs/cgroup/memory{path}/memory.limit_in_bytes")
        } else {
            continue;
        };
        if let Some(limit) = fs::read_to_string(file)
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
        {
            return Some(limit);
        }
    }
    None
}

/// Whether every page holding `bytes`, a part of a mapping, is in the page
/// cache: see [`Input::is_resident`].
#[cfg(target_os = "linux")]
fn pages_resident(bytes: &[u8]) -> bool {
    const MAX: usize = 256;
    // SAFETY: `sysconf` has no preconditions.
    let page = usize::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) })
        .ok()
        .filter(|p| p.is_power_of_two())
        .unwrap_or(4096);
    let addr = bytes.as_ptr() as usize;
    let start = addr & !page.wrapping_sub(1);
    let end = addr.saturating_add(bytes.len());
    let pages = end.saturating_sub(start).div_ceil(page);
    let mut vec = [0u8; MAX];
    let mut at = start;
    let mut left = pages;
    while left > 0 {
        let n = left.min(MAX);
        // SAFETY: `at..at + n * page` covers only whole pages of the mapping
        // that holds `bytes`, mapped while the borrow lives, and `vec` has a
        // byte for each of those `n <= MAX` pages; `mincore` only reports.
        let rc = unsafe {
            libc::mincore(
                at as *mut libc::c_void,
                n.saturating_mul(page),
                vec.as_mut_ptr(),
            )
        };
        if rc != 0 || vec.iter().take(n).any(|b| b & 1 == 0) {
            return false;
        }
        at = at.saturating_add(n.saturating_mul(page));
        left = left.saturating_sub(n);
    }
    true
}

/// `MADV_WILLNEED` over the pages holding `bytes`: see
/// [`crate::read_ahead`].
#[cfg(unix)]
pub(crate) fn advise_will_need(bytes: &[u8]) {
    // Pages already in the page cache have nothing to read in, and the
    // advice walks every page it covers: a scan over a resident file (one
    // that fits in memory) would pay a syscall per 128 KiB for nothing.
    #[cfg(target_os = "linux")]
    if pages_resident(bytes) {
        return;
    }
    // SAFETY: `sysconf` has no preconditions.
    let page = usize::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) })
        .ok()
        .filter(|p| p.is_power_of_two())
        .unwrap_or(4096);
    let addr = bytes.as_ptr() as usize;
    let start = addr & !page.wrapping_sub(1);
    let end = addr.saturating_add(bytes.len());
    // Linux reads at most the device's read-ahead window per call
    // (`force_page_cache_ra` caps it at `max(io_pages, ra_pages)`, 128 KiB
    // by default), so a larger range goes in window-sized calls; each only
    // queues its reads.
    const CALL: usize = 128 * 1024;
    let mut at = start;
    while at < end {
        let len = CALL.min(end.saturating_sub(at));
        // SAFETY: the range covers only pages that hold `bytes`, which are
        // mapped for as long as the borrow lives; `MADV_WILLNEED` schedules
        // reads and changes no contents or mapping.
        unsafe {
            libc::madvise(at as *mut libc::c_void, len, libc::MADV_WILLNEED);
        }
        at = at.saturating_add(len);
    }
}

#[cfg(not(unix))]
pub(crate) fn advise_will_need(_bytes: &[u8]) {}

/// The same bytes as [`Deref`], as an `AsRef` — the bound a type-erased
/// shared buffer needs.
///
/// `Deref` alone cannot be used behind `dyn`: `Arc<dyn Deref<Target = [u8]>>`
/// is legal but every consumer would have to name the associated type, and
/// `Arc<[u8]>` (the obvious alternative) cannot alias a mapping — it owns its
/// allocation, so handing an `Input` to one always costs a full copy. With
/// this impl an `Arc<Input>` coerces straight to
/// `Arc<dyn AsRef<[u8]> + Send + Sync>`, which is how
/// `lucene_codecs::blocktree::open_shared` takes a `.tim`/`.tip` mapping
/// without copying it (~199 µs saved on a 4.7 MB `.tim`).
impl AsRef<[u8]> for Input {
    fn as_ref(&self) -> &[u8] {
        self
    }
}

/// Directory abstraction covering both Lucene's read path (`listAll`, `open`
/// a whole file's bytes) and the write path: `createOutput`/`createTempOutput`
/// (an [`FsIndexOutput`]), `sync` (the fsync-before-durable contract),
/// `rename`/`deleteFile`/`syncMetaData` (what `SegmentInfos.prepareCommit`/
/// `finishCommit`/`rollbackCommit` need to publish a commit atomically),
/// `obtainLock` (the `write.lock` an `IndexWriter` holds) and
/// `getPendingDeletions`.
///
/// `Send + Sync`, as Java's `Directory` is by contract: a writer's indexing
/// threads flush segments into it concurrently while its merge thread reads
/// sources out of it. Every implementation here is either
/// stateless over the filesystem or guards its own state.
///
/// Java's `close()` is `Drop`: [`FsDirectory`] retries its pending deletes
/// when dropped, as `FSDirectory.close` does.
pub trait Directory: Send + Sync {
    /// Port of `Directory.listAll()`: every file name in the directory, sorted.
    fn list_all(&self) -> Result<Vec<String>>;

    /// Reads a whole file's bytes (`Directory.openInput`).
    fn open(&self, name: &str) -> Result<Input>;

    /// [`Self::open`] for a file whose blocks start every `alignment`
    /// bytes ([`crate::alignment`]): a directory reading whole pages uses
    /// pages of a power-of-two multiple of it. Others ignore it.
    fn open_aligned(&self, name: &str, alignment: usize) -> Result<Input> {
        crate::alignment::check_hint(alignment)?;
        self.open(name)
    }

    /// One member of a compound file as an input of its own: `len` bytes
    /// of `container` from `offset`, a file `member` of the segment (the
    /// name the compound file's entries use). `None` leaves the caller to
    /// slice the opened container ([`Input::slice_of`]), which every
    /// directory but a paging one does. A directory that chooses how to read
    /// a file by its kind (page size, residency, whether it is read at all)
    /// answers for the member, so a vector file in a compound segment is
    /// read as it would be as a file of its own. `offset` starts a member
    /// the format placed on a page ([`crate::alignment::ALIGN_BASE`]).
    fn open_member(
        &self,
        container: &str,
        offset: u64,
        len: u64,
        member: &str,
    ) -> Result<Option<Input>> {
        let _ = (container, offset, len, member);
        Ok(None)
    }

    /// Port of `Directory.fileLength(name)`. The default reads the file,
    /// which is correct for any implementation; the real ones override it.
    fn file_length(&self, name: &str) -> Result<u64> {
        Ok(self.open(name)?.len() as u64)
    }

    /// Port of `Directory.createOutput(name, context)`: creates (truncating
    /// any existing file of the same name) a new file for sequential
    /// writing.
    fn create_output(&self, name: &str) -> Result<FsIndexOutput>;

    /// Port of `Directory.createOutput(name, context)` for a context that
    /// carries a flush or merge size estimate (`FlushInfo`'s
    /// `estimatedSegmentSize`, `MergeInfo`'s `estimatedMergeBytes`), the only
    /// part of `IOContext` a directory's *results* can depend on
    /// ([`crate::NrtCachingDirectory`] caches small segments in memory). The
    /// default ignores the estimate; a wrapper forwards it.
    fn create_output_with_estimate(
        &self,
        name: &str,
        estimated_bytes: Option<u64>,
    ) -> Result<FsIndexOutput> {
        let _ = estimated_bytes;
        self.create_output(name)
    }

    /// Port of `Directory.createTempOutput(prefix, suffix, context)`: a new
    /// output under a fresh name built by [`temp_file_name`], never
    /// clobbering an existing file.
    ///
    /// Java declares it abstract; the default here refuses (as a read-only
    /// directory such as a compound file does), so a test double need not
    /// implement it. Every real directory in this crate does.
    fn create_temp_output(&self, prefix: &str, suffix: &str) -> Result<FsIndexOutput> {
        let _ = (prefix, suffix);
        Err(unsupported("createTempOutput"))
    }

    /// [`Directory::create_temp_output`] with a context's size estimate, as
    /// [`Directory::create_output_with_estimate`].
    fn create_temp_output_with_estimate(
        &self,
        prefix: &str,
        suffix: &str,
        estimated_bytes: Option<u64>,
    ) -> Result<FsIndexOutput> {
        let _ = estimated_bytes;
        self.create_temp_output(prefix, suffix)
    }

    /// Port of `Directory.sync(Collection<String>)`: fsyncs every named
    /// file's contents (and, best-effort, the directory entry) to disk.
    /// Callers must sync a new segment's files before referencing them from
    /// a commit file — that's Lucene's actual durability contract.
    fn sync(&self, names: &[String]) -> Result<()>;

    /// Port of `Directory.rename(source, dest)`: atomically makes `source`'s
    /// contents visible under `dest`. This is the operation Lucene's
    /// `SegmentInfos.finishCommit` relies on to publish a commit — the
    /// `pending_segments_N` file is fully written and fsynced first, then a
    /// single rename makes it the new `segments_N`, so no crash can ever
    /// expose a half-written commit file under a name a reader scans for.
    fn rename(&self, source: &str, dest: &str) -> Result<()>;

    /// Port of `Directory.deleteFile(name)`.
    fn delete_file(&self, name: &str) -> Result<()>;

    /// Port of `Directory.syncMetaData()`: fsyncs the directory itself, so a
    /// rename/create of a *name* (not just a file's contents) survives a
    /// crash. Lucene calls this on both sides of the commit rename.
    fn sync_meta_data(&self) -> Result<()>;

    /// Port of `Directory.obtainLock(name)`: acquires the named lock
    /// (`write.lock`) through this directory's [`LockFactory`], failing with
    /// [`Error::LockObtainFailed`] when it is held elsewhere.
    fn obtain_lock(&self, name: &str) -> Result<Box<dyn Lock>>;

    /// Port of `Directory.getPendingDeletions()`: files a delete was
    /// requested for but could not yet be removed. Only a filesystem
    /// directory ever has any.
    fn pending_deletions(&self) -> Result<BTreeSet<String>> {
        Ok(BTreeSet::new())
    }

    /// `len` bytes of `name` from `offset` (fewer at the end of the file),
    /// without opening the whole file: what a wrapper reads a header with.
    /// The default opens the file; a remote directory reads only the range.
    fn read_at(&self, name: &str, offset: u64, len: usize) -> Result<Vec<u8>> {
        let input = self.open(name)?;
        let start = usize::try_from(offset)
            .unwrap_or(usize::MAX)
            .min(input.len());
        let end = start.saturating_add(len).min(input.len());
        Ok(input[start..end].to_vec())
    }

    /// Opens `name` decoded page by page by `decoder` (an encrypting
    /// wrapper's decryption), when this directory fetches and caches pages
    /// itself ([`crate::CachedBlobDirectory`]). `None` (the default) means it
    /// does not, and the caller decodes [`Directory::open`]'s bytes.
    fn open_decoded(
        &self,
        name: &str,
        decoder: &dyn crate::page_cache::PageDecoder,
    ) -> Option<Result<Input>> {
        let _ = (name, decoder);
        None
    }

    /// A hint that `names` are about to be opened, as a reader opening a
    /// commit's segments knows: a directory that fetches files by range
    /// ([`crate::CachedBlobDirectory`]) fetches what those opens will read,
    /// all files at once, rather than one round trip after another. A
    /// failure here is only a missed hint; the opens report their own.
    fn prefetch_files(&self, names: &[String]) -> Result<()> {
        let _ = names;
        Ok(())
    }

    /// [`Directory::prefetch_files`] for files an encrypting wrapper opens
    /// through [`Directory::open_decoded`] with `decoder`: what those opens
    /// will read, decoded pages included.
    fn prefetch_files_decoded(
        &self,
        names: &[String],
        decoder: &dyn crate::page_cache::PageDecoder,
    ) -> Result<()> {
        let _ = (names, decoder);
        Ok(())
    }

    /// `dir instanceof FSDirectory ? dir.getDirectory() : null`: the
    /// filesystem path a [`crate::fs_lock_factory::FsLockFactory`] puts its
    /// lock file in. `None` for anything that is not a filesystem directory.
    fn fs_directory_path(&self) -> Option<&Path> {
        None
    }

    /// Disables test filesystem virus checkers beneath this directory, returning
    /// whether any checker was previously enabled. Wrappers forward this hook.
    #[cfg(any(test, feature = "test-util"))]
    fn disable_virus_checker(&self) -> bool {
        false
    }

    /// `IndexInput.close()` of an input [`Directory::open`] handed out, by a
    /// caller that closes explicitly where Java's try-with-resources would
    /// (feature `test-util`): the test framework's `MockDirectoryWrapper`
    /// lets a failure fire here, as `MockIndexInputWrapper.close` does. The
    /// default drops the input, which is how every other directory closes
    /// it. Wrappers forward this hook.
    #[cfg(any(test, feature = "test-util"))]
    fn close_input(&self, name: &str, input: Input) -> Result<()> {
        let _ = name;
        drop(input);
        Ok(())
    }

    /// Port of `Directory.copyFrom(from, src, dest, context)`: copies a file
    /// from another directory, deleting the partial `dest` on failure.
    fn copy_from(&self, from: &dyn Directory, src: &str, dest: &str) -> Result<()> {
        let copied = from.open(src).and_then(|bytes| {
            let mut out = self.create_output(dest)?;
            out.write_bytes(&bytes);
            out.close()
        });
        match copied {
            Ok(_) => Ok(()),
            Err(e) => {
                // `IOUtils.deleteFilesIgnoringExceptions(this, dest)`.
                let _ = self.delete_file(dest);
                Err(e)
            }
        }
    }
}

/// `UnsupportedOperationException` from a directory that cannot perform
/// `what`.
pub(crate) fn unsupported(what: &str) -> Error {
    Error::Io(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        format!("{what} is not supported by this directory"),
    ))
}

/// Forwards every [`Directory`] method through a pointer type, so a
/// reference or an `Arc` to a directory is a directory too -- which is what
/// lets the wrappers ([`crate::NrtCachingDirectory`],
/// [`crate::FileSwitchDirectory`], ...) own either.
macro_rules! forward_directory {
    ($($ty:ty),*) => {$(
        impl<T: Directory + ?Sized> Directory for $ty {
            fn list_all(&self) -> Result<Vec<String>> {
                (**self).list_all()
            }
            fn open(&self, name: &str) -> Result<Input> {
                (**self).open(name)
            }
            fn open_aligned(&self, name: &str, alignment: usize) -> Result<Input> {
                (**self).open_aligned(name, alignment)
            }
            fn open_member(
                &self,
                container: &str,
                offset: u64,
                len: u64,
                member: &str,
            ) -> Result<Option<Input>> {
                (**self).open_member(container, offset, len, member)
            }
            fn file_length(&self, name: &str) -> Result<u64> {
                (**self).file_length(name)
            }
            fn create_output(&self, name: &str) -> Result<FsIndexOutput> {
                (**self).create_output(name)
            }
            fn create_temp_output(&self, prefix: &str, suffix: &str) -> Result<FsIndexOutput> {
                (**self).create_temp_output(prefix, suffix)
            }
            fn create_output_with_estimate(
                &self,
                name: &str,
                estimated_bytes: Option<u64>,
            ) -> Result<FsIndexOutput> {
                (**self).create_output_with_estimate(name, estimated_bytes)
            }
            fn create_temp_output_with_estimate(
                &self,
                prefix: &str,
                suffix: &str,
                estimated_bytes: Option<u64>,
            ) -> Result<FsIndexOutput> {
                (**self).create_temp_output_with_estimate(prefix, suffix, estimated_bytes)
            }
            fn sync(&self, names: &[String]) -> Result<()> {
                (**self).sync(names)
            }
            fn rename(&self, source: &str, dest: &str) -> Result<()> {
                (**self).rename(source, dest)
            }
            fn delete_file(&self, name: &str) -> Result<()> {
                (**self).delete_file(name)
            }
            fn sync_meta_data(&self) -> Result<()> {
                (**self).sync_meta_data()
            }
            fn obtain_lock(&self, name: &str) -> Result<Box<dyn Lock>> {
                (**self).obtain_lock(name)
            }
            fn pending_deletions(&self) -> Result<BTreeSet<String>> {
                (**self).pending_deletions()
            }
            #[cfg(any(test, feature = "test-util"))]
            fn disable_virus_checker(&self) -> bool {
                (**self).disable_virus_checker()
            }
            #[cfg(any(test, feature = "test-util"))]
            fn close_input(&self, name: &str, input: Input) -> Result<()> {
                (**self).close_input(name, input)
            }
            fn fs_directory_path(&self) -> Option<&Path> {
                (**self).fs_directory_path()
            }
            fn read_at(&self, name: &str, offset: u64, len: usize) -> Result<Vec<u8>> {
                (**self).read_at(name, offset, len)
            }
            fn open_decoded(
                &self,
                name: &str,
                decoder: &dyn crate::page_cache::PageDecoder,
            ) -> Option<Result<Input>> {
                (**self).open_decoded(name, decoder)
            }
            fn prefetch_files(&self, names: &[String]) -> Result<()> {
                (**self).prefetch_files(names)
            }
            fn prefetch_files_decoded(
                &self,
                names: &[String],
                decoder: &dyn crate::page_cache::PageDecoder,
            ) -> Result<()> {
                (**self).prefetch_files_decoded(names, decoder)
            }
            fn copy_from(&self, from: &dyn Directory, src: &str, dest: &str) -> Result<()> {
                (**self).copy_from(from, src, dest)
            }
        }
    )*};
}

forward_directory!(&T, Arc<T>, Box<T>);

/// A directory every output of which is created with one size estimate:
/// the Rust shape of the single `IOContext` (`new IOContext(flushInfo)` or
/// `new IOContext(mergeInfo)`) Java's `IndexWriter` passes to every
/// `createOutput` of one flush or one merge. Everything else forwards.
pub struct EstimatedWrites<'a> {
    inner: &'a dyn Directory,
    estimated_bytes: u64,
}

impl<'a> EstimatedWrites<'a> {
    /// `inner`, with `estimated_bytes` (`FlushInfo.estimatedSegmentSize` or
    /// `MergeInfo.estimatedMergeBytes`) on every output it creates.
    pub fn new(inner: &'a dyn Directory, estimated_bytes: u64) -> Self {
        EstimatedWrites {
            inner,
            estimated_bytes,
        }
    }

    /// The estimate every output carries.
    pub fn estimated_bytes(&self) -> u64 {
        self.estimated_bytes
    }
}

impl Directory for EstimatedWrites<'_> {
    fn list_all(&self) -> Result<Vec<String>> {
        self.inner.list_all()
    }
    fn open(&self, name: &str) -> Result<Input> {
        self.inner.open(name)
    }
    fn open_aligned(&self, name: &str, alignment: usize) -> Result<Input> {
        self.inner.open_aligned(name, alignment)
    }
    fn file_length(&self, name: &str) -> Result<u64> {
        self.inner.file_length(name)
    }
    fn create_output(&self, name: &str) -> Result<FsIndexOutput> {
        self.inner
            .create_output_with_estimate(name, Some(self.estimated_bytes))
    }
    fn create_output_with_estimate(
        &self,
        name: &str,
        estimated_bytes: Option<u64>,
    ) -> Result<FsIndexOutput> {
        self.inner
            .create_output_with_estimate(name, estimated_bytes)
    }
    fn create_temp_output(&self, prefix: &str, suffix: &str) -> Result<FsIndexOutput> {
        self.inner
            .create_temp_output_with_estimate(prefix, suffix, Some(self.estimated_bytes))
    }
    fn create_temp_output_with_estimate(
        &self,
        prefix: &str,
        suffix: &str,
        estimated_bytes: Option<u64>,
    ) -> Result<FsIndexOutput> {
        self.inner
            .create_temp_output_with_estimate(prefix, suffix, estimated_bytes)
    }
    fn sync(&self, names: &[String]) -> Result<()> {
        self.inner.sync(names)
    }
    fn rename(&self, source: &str, dest: &str) -> Result<()> {
        self.inner.rename(source, dest)
    }
    fn delete_file(&self, name: &str) -> Result<()> {
        self.inner.delete_file(name)
    }
    fn sync_meta_data(&self) -> Result<()> {
        self.inner.sync_meta_data()
    }
    fn obtain_lock(&self, name: &str) -> Result<Box<dyn Lock>> {
        self.inner.obtain_lock(name)
    }
    fn pending_deletions(&self) -> Result<BTreeSet<String>> {
        self.inner.pending_deletions()
    }
    #[cfg(any(test, feature = "test-util"))]
    fn disable_virus_checker(&self) -> bool {
        self.inner.disable_virus_checker()
    }
    fn fs_directory_path(&self) -> Option<&Path> {
        self.inner.fs_directory_path()
    }
    fn read_at(&self, name: &str, offset: u64, len: usize) -> Result<Vec<u8>> {
        self.inner.read_at(name, offset, len)
    }
    fn open_decoded(
        &self,
        name: &str,
        decoder: &dyn crate::page_cache::PageDecoder,
    ) -> Option<Result<Input>> {
        self.inner.open_decoded(name, decoder)
    }
    fn prefetch_files(&self, names: &[String]) -> Result<()> {
        self.inner.prefetch_files(names)
    }
    fn prefetch_files_decoded(
        &self,
        names: &[String],
        decoder: &dyn crate::page_cache::PageDecoder,
    ) -> Result<()> {
        self.inner.prefetch_files_decoded(names, decoder)
    }
}

/// Port of `Directory.getTempFileName(prefix, suffix, counter)`:
/// `IndexFileNames.segmentFileName(prefix, suffix + "_" + base36(counter),
/// "tmp")`, i.e. `{prefix}_{suffix}_{counter}.tmp`.
pub fn temp_file_name(prefix: &str, suffix: &str, counter: u64) -> String {
    let counter = lucene_util::base36::to_base36(i64::try_from(counter).unwrap_or(i64::MAX));
    format!("{prefix}_{suffix}_{counter}.tmp")
}

/// Port of `BaseDirectory`: the part of a directory that holds its
/// [`LockFactory`] and implements `obtainLock` through it. Java's abstract
/// base class becomes a field the concrete directories
/// ([`FsDirectory`], [`MmapDirectory`], [`crate::ByteBuffersDirectory`])
/// embed.
#[derive(Debug, Clone)]
pub struct BaseDirectory {
    lock_factory: Arc<dyn LockFactory>,
}

impl BaseDirectory {
    /// `BaseDirectory(lockFactory)`.
    pub fn new(lock_factory: Arc<dyn LockFactory>) -> Self {
        Self { lock_factory }
    }

    /// `BaseDirectory.lockFactory`.
    pub fn lock_factory(&self) -> &Arc<dyn LockFactory> {
        &self.lock_factory
    }

    /// `BaseDirectory.obtainLock(name)`: `lockFactory.obtainLock(this,
    /// name)`, with `this` passed explicitly.
    pub fn obtain_lock(&self, dir: &dyn Directory, name: &str) -> Result<Box<dyn Lock>> {
        self.lock_factory.obtain_lock(dir, name)
    }
}

/// The state `FSDirectory` keeps over a filesystem path, shared by both
/// backends: the lock factory, the pending deletes, and the temp-file
/// counter. Only reads differ between [`FsDirectory`] and [`MmapDirectory`].
struct FsCore {
    root: PathBuf,
    base: BaseDirectory,
    /// `FSDirectory.pendingDeletes`: files whose delete failed with an error
    /// other than "no such file" (on Windows, a file still open elsewhere),
    /// hidden from listings and retried later.
    pending_deletes: Mutex<BTreeSet<String>>,
    /// `FSDirectory.opsSinceLastDelete`.
    ops_since_last_delete: AtomicUsize,
    /// `FSDirectory.nextTempFileCounter`.
    next_temp_file_counter: AtomicU64,
    /// The mock filesystem under this directory, if a test installed one.
    #[cfg(any(test, feature = "test-util"))]
    faults: Option<Arc<dyn FsFaults>>,
}

/// A mock filesystem under an [`FsDirectory`] or [`MmapDirectory`] (feature
/// `test-util`): the test framework's `WindowsFS` and `VirusCheckingFS`,
/// which refuse to delete or move a file the way a real filesystem can, so
/// the directory's own pending-delete handling (`FSDirectory`'s) is what
/// answers. Java installs these as `java.nio` filesystem providers under
/// the directory; this port has no filesystem layer to wrap, so the
/// directory asks the hook before the real call.
#[cfg(any(test, feature = "test-util"))]
pub trait FsFaults: Send + Sync {
    /// Before `path` is deleted: an error refuses the delete, as the
    /// filesystem's `delete(path)` throwing would.
    fn check_delete(&self, path: &Path) -> std::io::Result<()>;
    /// Before `source` is moved onto `dest`: an error refuses the move.
    fn check_rename(&self, source: &Path, dest: &Path) -> std::io::Result<()>;
    /// Disables this test filesystem's virus checker, returning its prior state.
    fn disable_virus_checker(&self) -> bool {
        false
    }
}

impl FsCore {
    #[cfg(any(test, feature = "test-util"))]
    fn disable_virus_checker(&self) -> bool {
        self.faults
            .as_ref()
            .is_some_and(|faults| faults.disable_virus_checker())
    }

    fn new(root: PathBuf, lock_factory: Arc<dyn LockFactory>) -> Self {
        Self {
            root,
            base: BaseDirectory::new(lock_factory),
            pending_deletes: Mutex::new(BTreeSet::new()),
            ops_since_last_delete: AtomicUsize::new(0),
            next_temp_file_counter: AtomicU64::new(0),
            #[cfg(any(test, feature = "test-util"))]
            faults: None,
        }
    }

    /// The mock filesystem's verdict on deleting `path`; always `Ok`
    /// without one.
    fn check_delete(&self, _path: &Path) -> std::io::Result<()> {
        #[cfg(any(test, feature = "test-util"))]
        if let Some(faults) = &self.faults {
            return faults.check_delete(_path);
        }
        Ok(())
    }

    fn pending(&self) -> std::sync::MutexGuard<'_, BTreeSet<String>> {
        crate::lock::lock_ignoring_poison(&self.pending_deletes)
    }

    fn is_pending(&self, name: &str) -> bool {
        self.pending().contains(name)
    }

    /// `FSDirectory.listAll()`: the listing, minus pending deletes.
    fn list_all(&self) -> Result<Vec<String>> {
        let mut names = list_all(&self.root)?;
        let pending = self.pending();
        if !pending.is_empty() {
            names.retain(|n| !pending.contains(n));
        }
        Ok(names)
    }

    /// `FSDirectory.fileLength(name)`.
    fn file_length(&self, name: &str) -> Result<u64> {
        if self.is_pending(name) {
            return Err(Error::no_such_file(format!(
                "file \"{name}\" is pending delete"
            )));
        }
        Ok(fs::metadata(self.root.join(name))?.len())
    }

    /// `FSDirectory.ensureCanRead(name)`.
    fn ensure_can_read(&self, name: &str) -> Result<()> {
        if self.is_pending(name) {
            return Err(Error::no_such_file(format!(
                "file \"{name}\" is pending delete and cannot be opened for read"
            )));
        }
        Ok(())
    }

    /// `FSDirectory.createOutput(name, context)`.
    fn create_output(&self, name: &str) -> Result<FsIndexOutput> {
        self.maybe_delete_pending_files()?;
        // If this file was pending delete, we are now bringing it back to
        // life.
        if self.pending().remove(name) {
            // Try again to delete it -- this is the best effort.
            self.private_delete_file(name, true)?;
            // If the delete failed it went back in; take it out again.
            self.pending().remove(name);
        }
        self.create_new(name)
    }

    /// `new FSIndexOutput(name)`: `WRITE, CREATE_NEW`, so an existing name
    /// fails with `ErrorKind::AlreadyExists` (`FileAlreadyExistsException`).
    /// A root that does not exist yet is created first -- Java's
    /// constructor creates it; this port does it on the first output.
    fn create_new(&self, name: &str) -> Result<FsIndexOutput> {
        match FsIndexOutput::create_new(&self.root, name) {
            Err(Error::Io(e))
                if e.kind() == std::io::ErrorKind::NotFound && !self.root.is_dir() =>
            {
                fs::create_dir_all(&self.root)?;
                FsIndexOutput::create_new(&self.root, name)
            }
            other => other,
        }
    }

    /// `FSDirectory.createTempOutput(prefix, suffix, context)`.
    fn create_temp_output(&self, prefix: &str, suffix: &str) -> Result<FsIndexOutput> {
        self.maybe_delete_pending_files()?;
        loop {
            let counter = self.next_temp_file_counter.fetch_add(1, Ordering::Relaxed);
            let name = temp_file_name(prefix, suffix, counter);
            if self.is_pending(&name) {
                continue;
            }
            match self.create_new(&name) {
                Err(Error::Io(e)) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                other => return other,
            }
        }
    }

    /// `FSDirectory.sync(names)`.
    fn sync(&self, names: &[String]) -> Result<()> {
        index_output::sync(&self.root, names)?;
        self.maybe_delete_pending_files()
    }

    /// `FSDirectory.rename(source, dest)`.
    fn rename(&self, source: &str, dest: &str) -> Result<()> {
        if self.is_pending(source) {
            return Err(Error::no_such_file(format!(
                "file \"{source}\" is pending delete and cannot be moved"
            )));
        }
        self.maybe_delete_pending_files()?;
        if self.pending().remove(dest) {
            self.private_delete_file(dest, true)?;
            self.pending().remove(dest);
        }
        #[cfg(any(test, feature = "test-util"))]
        if let Some(faults) = &self.faults {
            faults.check_rename(&self.root.join(source), &self.root.join(dest))?;
        }
        index_output::rename(&self.root, source, dest)
    }

    /// `FSDirectory.syncMetaData()`.
    fn sync_meta_data(&self) -> Result<()> {
        index_output::sync_meta_data(&self.root)?;
        self.maybe_delete_pending_files()
    }

    /// `FSDirectory.deleteFile(name)`.
    fn delete_file(&self, name: &str) -> Result<()> {
        if self.is_pending(name) {
            return Err(Error::no_such_file(format!(
                "file \"{name}\" is already pending delete"
            )));
        }
        self.private_delete_file(name, false)?;
        self.maybe_delete_pending_files()
    }

    /// `FSDirectory.deletePendingFiles()`: retries every pending delete.
    fn delete_pending_files(&self) -> Result<()> {
        // Clone the set, since `private_delete_file` mutates it.
        let pending: Vec<String> = self.pending().iter().cloned().collect();
        for name in pending {
            self.private_delete_file(&name, true)?;
        }
        Ok(())
    }

    /// `FSDirectory.maybeDeletePendingFiles()`: retries the pending deletes
    /// once every `pendingDeletes.size()` operations -- "a silly heuristic to
    /// try to avoid O(N^2) behaviour on Windows".
    fn maybe_delete_pending_files(&self) -> Result<()> {
        let pending = self.pending().len();
        if pending == 0 {
            return Ok(());
        }
        let count = self
            .ops_since_last_delete
            .fetch_add(1, Ordering::AcqRel)
            .saturating_add(1);
        if count >= pending {
            self.ops_since_last_delete
                .fetch_sub(count, Ordering::AcqRel);
            self.delete_pending_files()?;
        }
        Ok(())
    }

    /// `FSDirectory.privateDeleteFile(name, isPendingDelete)`.
    fn private_delete_file(&self, name: &str, is_pending_delete: bool) -> Result<()> {
        let path = self.root.join(name);
        match self
            .check_delete(&path)
            .and_then(|()| fs::remove_file(&path))
        {
            Ok(()) => {
                self.pending().remove(name);
                Ok(())
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // We were asked to delete a non-existent file.
                self.pending().remove(name);
                if is_pending_delete && cfg!(windows) {
                    // LUCENE-6684: a file can sit in a "pending delete" state
                    // on Windows, failing the first attempt with access
                    // denied and then this one with no-such-file.
                    Ok(())
                } else {
                    Err(Error::Io(e))
                }
            }
            Err(_) => {
                // On Windows a delete can fail while a handle is still open
                // against the file: record it and try again later. (Java
                // does this for every other error, on every platform -- a
                // CIFS mount on Linux can behave the same way.)
                self.pending().insert(name.to_string());
                Ok(())
            }
        }
    }

    /// `FSDirectory.getPendingDeletions()`: retries first, then reports what
    /// is still pending.
    fn pending_deletions(&self) -> Result<BTreeSet<String>> {
        self.delete_pending_files()?;
        Ok(self.pending().clone())
    }
}

impl Drop for FsCore {
    /// `FSDirectory.close()`: a last attempt at the pending deletes.
    fn drop(&mut self) {
        let _ = self.delete_pending_files();
    }
}

/// Safe, copying backend (`std::fs::read`) -- Java's `NIOFSDirectory` in
/// role. No `unsafe` on its read path.
pub struct FsDirectory {
    core: FsCore,
}

impl FsDirectory {
    /// `FSDirectory.open(path)`: locks with [`NativeFsLockFactory`]
    /// (`FSLockFactory.getDefault()`). Neither creates nor canonicalises the
    /// path up front, unlike Java's constructor: the first output (or the
    /// lock factory, when a writer first locks it) creates the directory.
    pub fn open(root: impl Into<PathBuf>) -> Self {
        Self::with_lock_factory(root, default_fs_lock_factory())
    }

    /// `FSDirectory.open(path, lockFactory)`.
    pub fn with_lock_factory(root: impl Into<PathBuf>, lock_factory: Arc<dyn LockFactory>) -> Self {
        Self {
            core: FsCore::new(root.into(), lock_factory),
        }
    }

    /// `FSDirectory.getDirectory()`.
    pub fn directory(&self) -> &Path {
        &self.core.root
    }

    /// `BaseDirectory.lockFactory`.
    pub fn lock_factory(&self) -> &Arc<dyn LockFactory> {
        self.core.base.lock_factory()
    }

    /// `FSDirectory.deletePendingFiles()`.
    pub fn delete_pending_files(&self) -> Result<()> {
        self.core.delete_pending_files()
    }

    /// Puts a mock filesystem under this directory (feature `test-util`;
    /// see [`FsFaults`]).
    #[cfg(any(test, feature = "test-util"))]
    pub fn with_fs_faults(mut self, faults: Arc<dyn FsFaults>) -> Self {
        self.core.faults = Some(faults);
        self
    }
}

impl Directory for FsDirectory {
    fn list_all(&self) -> Result<Vec<String>> {
        self.core.list_all()
    }

    fn open(&self, name: &str) -> Result<Input> {
        self.core.ensure_can_read(name)?;
        Ok(Input::Owned(fs::read(self.core.root.join(name))?))
    }

    fn file_length(&self, name: &str) -> Result<u64> {
        self.core.file_length(name)
    }

    fn create_output(&self, name: &str) -> Result<FsIndexOutput> {
        self.core.create_output(name)
    }

    fn create_temp_output(&self, prefix: &str, suffix: &str) -> Result<FsIndexOutput> {
        self.core.create_temp_output(prefix, suffix)
    }

    fn sync(&self, names: &[String]) -> Result<()> {
        self.core.sync(names)
    }

    fn rename(&self, source: &str, dest: &str) -> Result<()> {
        self.core.rename(source, dest)
    }

    fn delete_file(&self, name: &str) -> Result<()> {
        self.core.delete_file(name)
    }

    fn sync_meta_data(&self) -> Result<()> {
        self.core.sync_meta_data()
    }

    fn obtain_lock(&self, name: &str) -> Result<Box<dyn Lock>> {
        self.core.base.obtain_lock(self, name)
    }

    fn pending_deletions(&self) -> Result<BTreeSet<String>> {
        self.core.pending_deletions()
    }

    #[cfg(any(test, feature = "test-util"))]
    fn disable_virus_checker(&self) -> bool {
        self.core.disable_virus_checker()
    }
    fn fs_directory_path(&self) -> Option<&Path> {
        Some(&self.core.root)
    }
}

/// The size at or below which [`MmapDirectory`] reads a file instead of
/// mapping it.
///
/// Mapping is only cheaper than reading once the mapping is large enough to
/// amortise its own syscalls. Measured on this project's 5 M-document
/// benchmark corpus (`crates/lucene-search/examples/reader_open_profile.rs`),
/// a few-hundred-byte file costs **1.86 µs** to `open`+`mmap`+`munmap` and
/// **1.19 µs** to `open`+`read`, before the mapping's first page fault --
/// and every file this small in a segment (`.si`, `.fnm`, `.tmd`, `.dvm`,
/// `.nvm`, `.kdm`, `segments_N`) is parsed whole at open, so it takes that
/// fault immediately.
///
/// 16 KiB, not larger, because the files a reader *holds* and then accesses
/// randomly -- `.tip`, `.kdi` -- are the ones a copy would pessimise, and
/// they are above this on any index big enough for it to matter. The worst
/// case the threshold can cost is one 16 KiB `memcpy`.
///
/// Real Lucene has no equivalent: `MMapDirectory.openInput` maps
/// unconditionally. This is a Rust-side win, not a port divergence -- the
/// bytes a caller sees are identical, and [`Input`] already had both
/// representations because [`FsDirectory`] produces the owned one.
pub const SMALL_FILE_READ_THRESHOLD: u64 = 16 * 1024;

/// Zero-copy backend (`memmap2`), matching Lucene's default `MMapDirectory`.
pub struct MmapDirectory {
    core: FsCore,
    /// Files of at most this many bytes are read rather than mapped -- see
    /// [`SMALL_FILE_READ_THRESHOLD`]. `0` maps everything, which is what
    /// this backend did before and what
    /// [`MmapDirectory::with_read_threshold`] exists to reproduce.
    read_threshold: u64,
}

impl MmapDirectory {
    /// `new MMapDirectory(path)`, locking with [`NativeFsLockFactory`].
    pub fn open(root: impl Into<PathBuf>) -> Self {
        Self::with_read_threshold(root, SMALL_FILE_READ_THRESHOLD)
    }

    /// [`Self::open`] with an explicit small-file threshold: files of at most
    /// `read_threshold` bytes are read into memory instead of mapped. `0`
    /// maps every file.
    ///
    /// Exposed because it is the only way to measure the two arms in **one
    /// process**, which is how this project measures anything -- and because a caller with an unusual access pattern (a
    /// long-lived reader over many small files it seeks in repeatedly) can
    /// turn it off.
    pub fn with_read_threshold(root: impl Into<PathBuf>, read_threshold: u64) -> Self {
        Self {
            core: FsCore::new(root.into(), default_fs_lock_factory()),
            read_threshold,
        }
    }

    /// `new MMapDirectory(path, lockFactory)`.
    pub fn with_lock_factory(root: impl Into<PathBuf>, lock_factory: Arc<dyn LockFactory>) -> Self {
        Self {
            core: FsCore::new(root.into(), lock_factory),
            read_threshold: SMALL_FILE_READ_THRESHOLD,
        }
    }

    /// Refuses a read of `name` while it is pending delete, as
    /// [`Directory::open`] does: for a wrapper that opens the file itself.
    pub(crate) fn ensure_can_read(&self, name: &str) -> Result<()> {
        self.core.ensure_can_read(name)
    }

    /// `FSDirectory.getDirectory()`.
    pub fn directory(&self) -> &Path {
        &self.core.root
    }

    /// `FSDirectory.deletePendingFiles()`.
    pub fn delete_pending_files(&self) -> Result<()> {
        self.core.delete_pending_files()
    }

    /// Puts a mock filesystem under this directory (feature `test-util`;
    /// see [`FsFaults`]).
    #[cfg(any(test, feature = "test-util"))]
    pub fn with_fs_faults(mut self, faults: Arc<dyn FsFaults>) -> Self {
        self.core.faults = Some(faults);
        self
    }
}

impl Directory for MmapDirectory {
    fn list_all(&self) -> Result<Vec<String>> {
        self.core.list_all()
    }

    fn open(&self, name: &str) -> Result<Input> {
        self.core.ensure_can_read(name)?;
        let mut file = fs::File::open(self.core.root.join(name))?;
        // A small file is read, not mapped: see `SMALL_FILE_READ_THRESHOLD`.
        // The `metadata` call is one `fstat` on an already-open descriptor,
        // which is cheaper than the `munmap` it avoids.
        let len = file.metadata()?.len();
        if self.read_threshold > 0 && len <= self.read_threshold {
            // Read from the descriptor already open, not `fs::read(path)`:
            // that re-opens by name and re-`fstat`s for its own size hint, so
            // the "cheaper than a mapping" arm would have paid *two*
            // `open`+`fstat` pairs -- and would have reopened a path that can
            // have changed underneath it in between. `len` is at most
            // `read_threshold`, so the `usize` cast cannot truncate on any
            // target this crate builds for.
            let mut buf = Vec::with_capacity(len as usize);
            file.read_to_end(&mut buf)?;
            return Ok(Input::Owned(buf));
        }
        // SAFETY: mapping is only unsound if another process truncates or
        // mutates this file while it's mapped, which we do not do ourselves and
        // which Lucene's own `MMapDirectory` accepts the same risk for (see its
        // Javadoc). Lucene never rewrites a file once written, so a mapped
        // index file is immutable for as long as anyone reads it.
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        Ok(Input::Mapped(mmap))
    }

    fn file_length(&self, name: &str) -> Result<u64> {
        self.core.file_length(name)
    }

    fn create_output(&self, name: &str) -> Result<FsIndexOutput> {
        self.core.create_output(name)
    }

    fn create_temp_output(&self, prefix: &str, suffix: &str) -> Result<FsIndexOutput> {
        self.core.create_temp_output(prefix, suffix)
    }

    fn sync(&self, names: &[String]) -> Result<()> {
        self.core.sync(names)
    }

    fn rename(&self, source: &str, dest: &str) -> Result<()> {
        self.core.rename(source, dest)
    }

    fn delete_file(&self, name: &str) -> Result<()> {
        self.core.delete_file(name)
    }

    fn sync_meta_data(&self) -> Result<()> {
        self.core.sync_meta_data()
    }

    fn obtain_lock(&self, name: &str) -> Result<Box<dyn Lock>> {
        self.core.base.obtain_lock(self, name)
    }

    fn pending_deletions(&self) -> Result<BTreeSet<String>> {
        self.core.pending_deletions()
    }

    #[cfg(any(test, feature = "test-util"))]
    fn disable_virus_checker(&self) -> bool {
        self.core.disable_virus_checker()
    }
    fn fs_directory_path(&self) -> Option<&Path> {
        Some(&self.core.root)
    }
}

fn list_all(root: &Path) -> Result<Vec<String>> {
    let mut names: Vec<String> = fs::read_dir(root)?
        .map(|entry| entry.map(|e| e.file_name().to_string_lossy().into_owned()))
        .collect::<std::io::Result<_>>()?;
    names.sort();
    Ok(names)
}

/// Port of `SegmentInfos.generationFromSegmentsFileName`.
pub fn generation_from_segments_file_name(file_name: &str) -> Result<i64> {
    if file_name == OLD_SEGMENTS_GEN {
        return Err(Error::Corrupted(format!(
            "\"{OLD_SEGMENTS_GEN}\" is not a valid segment file name since 4.0"
        )));
    }
    if file_name == SEGMENTS_PREFIX {
        return Ok(0);
    }
    // Java: `Long.parseLong(fileName.substring(1 + "segments".length()), 36)`
    // -- the one character after `segments` is skipped, whatever it is.
    if let Some(rest) = file_name.strip_prefix(SEGMENTS_PREFIX) {
        let mut chars = rest.chars();
        chars.next();
        return lucene_util::base36::from_base36(chars.as_str()).ok_or_else(|| {
            Error::Corrupted(format!("fileName \"{file_name}\" is not a segments file"))
        });
    }
    Err(Error::Corrupted(format!(
        "fileName \"{file_name}\" is not a segments file"
    )))
}

/// Port of `SegmentInfos.getLastCommitGeneration(String[])`: the highest
/// generation among `segments`/`segments_N` file names (excluding the legacy
/// `segments.gen` pointer), or -1 if none exist.
///
/// Strict, like Java: an unparsable `segments*` name aborts the scan rather
/// than being skipped. Skipping it is not a safe simplification in either
/// direction — a reader would silently open the *previous* commit and hide
/// every document committed since, and a writer would read -1 from a
/// directory that does have a commit and create a fresh index over it.
pub fn last_commit_generation(files: &[String]) -> Result<i64> {
    let mut generation = -1i64;
    for file in files {
        if is_segments_candidate(file) {
            generation = generation.max(generation_from_segments_file_name(file)?);
        }
    }
    Ok(generation)
}

/// The `startsWith(SEGMENTS) && startsWith(OLD_SEGMENTS_GEN) == false` guard
/// Java applies before calling `generationFromSegmentsFileName`. Note the
/// second test is a prefix test in Java, not equality: `segments.gen_1` is
/// skipped too, not treated as a corrupt generation.
fn is_segments_candidate(file_name: &str) -> bool {
    file_name.starts_with(SEGMENTS_PREFIX) && !file_name.starts_with(OLD_SEGMENTS_GEN)
}

/// Port of `IndexFileNames.fileNameFromGeneration("segments", "", gen)`.
pub fn segments_file_name(generation: i64) -> Option<String> {
    match generation {
        -1 => None,
        0 => Some(SEGMENTS_PREFIX.to_string()),
        gen => Some(format!(
            "{SEGMENTS_PREFIX}_{}",
            lucene_util::base36::to_base36(gen)
        )),
    }
}

/// Port of `IndexFileNames.fileNameFromGeneration("pending_segments", "",
/// gen)`: the name a commit file is written under before
/// `SegmentInfos.finishCommit` renames it to [`segments_file_name`]'s name.
///
/// Same `gen == -1 -> null`, `gen == 0 -> bare base name` shape
/// [`segments_file_name`] has, since both go through the same Java helper.
/// Generation 0 is unreachable in practice (`getNextPendingGeneration()`
/// returns `1` for a never-committed index and `generation + 1` otherwise),
/// but the mapping is kept total and exact rather than special-cased away.
pub fn pending_segments_file_name(generation: i64) -> Option<String> {
    match generation {
        g if g < 0 => None,
        0 => Some(PENDING_SEGMENTS_PREFIX.to_string()),
        gen => Some(format!(
            "{PENDING_SEGMENTS_PREFIX}_{}",
            lucene_util::base36::to_base36(gen)
        )),
    }
}

/// Finds and reads the most recent `segments_N` commit file in `dir`.
/// Returns `(generation, bytes)`; callers pass both to `segment_infos::parse`.
pub fn read_latest_commit(dir: &(impl Directory + ?Sized)) -> Result<(i64, Input)> {
    let files = dir.list_all()?;
    let generation = last_commit_generation(&files)?;
    Ok((generation, open_commit_file(dir, generation)?))
}

/// Opens the `segments_N` file of commit `generation` (`-1`, no commit,
/// is an error).
pub fn open_commit_file(dir: &(impl Directory + ?Sized), generation: i64) -> Result<Input> {
    let name = segments_file_name(generation)
        .ok_or_else(|| Error::Corrupted("no segments_N commit file found".to_string()))?;
    dir.open(&name)
}

#[cfg(test)]
mod tests;
