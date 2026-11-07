//! A local-disk [`Directory`] whose reads go through the process's demand
//! pager ([`crate::paged_input`]) instead of `mmap`, so the memory index
//! files take is capped by one budget ([`Pager::set_budget`]) rather than
//! left to the kernel's page cache.
//!
//! - Writes, listing, renames, deletes, syncs and locks are
//!   [`MmapDirectory`]'s: the files on disk are plain Lucene files, and the
//!   same index opens with either directory.
//! - A file larger than one page opens as an anonymous mapping filled in
//!   fixed pages on first touch (or ahead of the reads, through
//!   [`Input::will_need`] and [`crate::prefetch`]), each page read with one
//!   `pread`, adjacent pages merged into one read; faults that run in
//!   sequence read ahead ([`PagedFsConfig::readahead`]). Past the budget, the
//!   pages least recently brought in or announced by a prefetch are
//!   released and read again on their next touch.
//! - Reads bypass the kernel's page cache (`O_DIRECT`) where the
//!   filesystem allows it, so a page is held once, in the budget. Elsewhere
//!   (tmpfs, some network filesystems) reads are buffered and the kernel's
//!   copy is dropped after each read (`POSIX_FADV_DONTNEED`).
//! - Files of at most one page, and files whose extension is pinned
//!   ([`PagedFsConfig::pinned_extensions`]: metadata, terms index, norms,
//!   the stored-fields and points indexes, live docs), are read whole into
//!   memory at open, outside the budget: they are small and read on every
//!   query. So are the HNSW graph, quantized vectors and IVF codes when
//!   each is at most an eighth of the budget ([`PagedFsConfig::pins`]).
//! - A file larger than half the budget gets small pages (4 KiB for vector
//!   data, 16 KiB otherwise; [`PagedFsConfig::page_size_for`]): it cannot
//!   stay resident, and a smaller page brings in less it does not need.
//! - Without `userfaultfd` (not Linux, or blocked by a seccomp policy)
//!   every file opens as [`MmapDirectory`] opens it; [`PagedFsDirectory::is_paged`]
//!   tells which.
//!
//! [`Pager::set_budget`]: crate::paged_input::Pager::set_budget

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use crate::directory::{Directory, Input, MmapDirectory};
use crate::error::Result;
use crate::index_output::FsIndexOutput;
use crate::lock::Lock;
use crate::page_cache::DEFAULT_PINNED_EXTENSIONS;

/// The default page of a local file: 64 KiB. Local reads have no request
/// latency to amortize, so pages are smaller than a remote store's; one
/// page is still a single efficient device read.
pub const DEFAULT_LOCAL_PAGE_SIZE: usize = 64 << 10;
/// The default most bytes a run of faults reads ahead: 1 MiB.
pub const DEFAULT_READAHEAD: usize = 1 << 20;
/// The largest single read, in bytes, of adjacent missing pages.
const MAX_READ: usize = 8 << 20;

/// [`PagedFsDirectory`]'s settings.
#[derive(Debug, Clone)]
pub struct PagedFsConfig {
    /// Bytes per page: a whole number of OS pages.
    pub page_size: usize,
    /// Read with `O_DIRECT` where the filesystem supports it.
    pub direct_io: bool,
    /// Extensions of files read whole into memory at open.
    pub pinned_extensions: Vec<String>,
    /// Extensions whose files get a page size of their own instead of
    /// [`Self::page_size`]: smaller pages for files read at random a few
    /// bytes at a time (rerank vectors), larger for files read in long runs.
    pub page_sizes: Vec<(String, usize)>,
    /// Extensions of files mapped (`mmap`) as [`MmapDirectory`] maps them,
    /// outside the budget.
    pub mapped_extensions: Vec<String>,
    /// When set, only files of these extensions are read through the
    /// pager; every other file is mapped. [`Self::vectors_only`] sets it to
    /// the vector formats' files.
    pub paged_extensions: Option<Vec<String>>,
    /// Size each file's pages, and pin the vector search structures,
    /// against the pager's budget ([`Self::page_size_for`]); explicit
    /// [`Self::page_sizes`] and [`Self::pinned_extensions`] still win.
    pub adaptive: bool,
    /// The most bytes a fault reads ahead when a file's faults run in
    /// sequence (postings, stored fields scanned in order): each fault in a
    /// run reads twice as many pages as the one before, up to this. `0`
    /// reads only the faulting page.
    pub readahead: usize,
}

impl Default for PagedFsConfig {
    fn default() -> Self {
        Self {
            page_size: DEFAULT_LOCAL_PAGE_SIZE,
            direct_io: true,
            pinned_extensions: DEFAULT_PINNED_EXTENSIONS
                .iter()
                .map(|e| (*e).to_string())
                .collect(),
            page_sizes: Vec::new(),
            mapped_extensions: Vec::new(),
            paged_extensions: None,
            readahead: DEFAULT_READAHEAD,
            adaptive: true,
        }
    }
}

/// The files of the vector formats: HNSW (`.vec` vectors, `.vex` graph,
/// `.veq` quantized vectors) and IVF (`.ivp` cluster codes, `.ivv` rerank
/// vectors), what [`PagedFsConfig::vectors_only`] reads through the pager.
pub const VECTOR_EXTENSIONS: &[&str] = &["vec", "vex", "veq", "ivp", "ivv"];

impl PagedFsConfig {
    /// The default settings with only the vector formats' files read
    /// through the pager and everything else mapped: vector search, whose
    /// random reads gain most from small pages and batched reads, within
    /// the budget; full-text files left to the kernel's page cache.
    pub fn vectors_only() -> Self {
        Self {
            paged_extensions: Some(VECTOR_EXTENSIONS.iter().map(|e| (*e).to_string()).collect()),
            ..Self::default()
        }
    }
}

/// Extensions of files read at random a few hundred bytes at a time:
/// full-precision vectors read for scoring or reranking.
const RANDOM_ACCESS_EXTENSIONS: &[&str] = &["vec", "ivv"];
/// Extensions of the structures every vector search walks: the HNSW graph,
/// quantized vectors, IVF cluster codes.
const SEARCH_STRUCTURE_EXTENSIONS: &[&str] = &["vex", "veq", "ivp"];

impl PagedFsConfig {
    /// The page size of a file named `name` of `len` bytes when the pager
    /// keeps `budget` bytes resident. An explicit [`Self::page_sizes`] entry
    /// wins. Otherwise, with [`Self::adaptive`], a file larger than half the
    /// budget, which cannot stay resident, gets small pages so a read
    /// brings in little it does not need: 4 KiB for vector data, 16 KiB for
    /// the rest. Everything else gets [`Self::page_size`].
    pub fn page_size_for(&self, name: &str, len: u64, budget: u64) -> usize {
        let ext = extension(name);
        if let Some(&(_, size)) = self.page_sizes.iter().find(|(e, _)| e == ext) {
            return size;
        }
        if self.adaptive && len > budget / 2 {
            let small = if RANDOM_ACCESS_EXTENSIONS.contains(&ext) {
                4 << 10
            } else {
                16 << 10
            };
            return small.min(self.page_size);
        }
        self.page_size
    }

    /// Whether a file named `name` of `len` bytes is read whole at open:
    /// its extension is pinned, or, with [`Self::adaptive`], it is a vector
    /// search structure of at most an eighth of the budget, which every
    /// query walks at random.
    pub fn pins(&self, name: &str, len: u64, budget: u64) -> bool {
        let ext = extension(name);
        self.pinned_extensions.iter().any(|p| p == ext)
            || (self.adaptive && SEARCH_STRUCTURE_EXTENSIONS.contains(&ext) && len <= budget / 8)
    }
}

/// A local-disk directory read through the demand pager. See the module
/// docs.
pub struct PagedFsDirectory {
    inner: MmapDirectory,
    root: PathBuf,
    config: PagedFsConfig,
}

impl std::fmt::Debug for PagedFsDirectory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PagedFsDirectory({})", self.root.display())
    }
}

impl PagedFsDirectory {
    /// A directory at `root` with the default settings.
    pub fn open(root: impl Into<PathBuf>) -> Self {
        Self::with_config(root, PagedFsConfig::default())
    }

    /// A directory at `root` with `config`.
    pub fn with_config(root: impl Into<PathBuf>, config: PagedFsConfig) -> Self {
        let root = root.into();
        Self {
            inner: MmapDirectory::open(root.clone()),
            root,
            config,
        }
    }

    /// The settings.
    pub fn config(&self) -> &PagedFsConfig {
        &self.config
    }

    /// Whether files open through the pager here (`userfaultfd` is
    /// available and the page sizes fit it); otherwise they are mapped.
    pub fn is_paged(&self) -> bool {
        #[cfg(target_os = "linux")]
        return crate::paged_input::Pager::global().is_some_and(|p| {
            p.supports(self.config.page_size)
                && self.config.page_sizes.iter().all(|&(_, s)| p.supports(s))
        });
        #[cfg(not(target_os = "linux"))]
        false
    }

    fn is_mapped(&self, name: &str) -> bool {
        let ext = extension(name);
        self.config.mapped_extensions.iter().any(|p| p == ext)
            || self
                .config
                .paged_extensions
                .as_ref()
                .is_some_and(|paged| !paged.iter().any(|p| p == ext))
    }

    #[cfg(target_os = "linux")]
    fn open_paged(&self, name: &str) -> Result<Option<Input>> {
        use crate::paged_input::{PageFiller, Pager};
        use std::sync::Arc;

        if self.is_mapped(name) {
            return Ok(None);
        }
        let Some(pager) = Pager::global() else {
            return Ok(None);
        };
        let path = self.root.join(name);
        let stored = std::fs::metadata(&path)?.len();
        let budget = pager.budget();
        let page = self.config.page_size_for(name, stored, budget);
        if !pager.supports(page) {
            return Ok(None);
        }
        let file = local::LocalFile::open(&path, page, self.config.direct_io)?;
        let len = usize::try_from(file.len())
            .map_err(|_| crate::error::Error::Corrupted(format!("{name}: too large to open")))?;
        if self.config.pins(name, file.len(), budget) || len <= page {
            return Ok(Some(Input::Owned(file.read_whole()?)));
        }
        let filler: Arc<dyn PageFiller> = Arc::new(file);
        let readahead = self.config.readahead.checked_div(page).unwrap_or(0).max(1);
        let paged = pager.open_with_readahead(len, page, filler, name, readahead)?;
        // Readers check the header and footer as they open a file: both
        // ends in one batch. Only a hint; a fault retries what it misses.
        let bytes = crate::directory::ExternalBytes::bytes(&paged);
        let footer = bytes.len().saturating_sub(crate::codec_util::FOOTER_LENGTH);
        let _ = crate::paged_input::prefetch(&[bytes.get(..1).unwrap_or(&[]), &bytes[footer..]]);
        Ok(Some(Input::external_bytes(paged)))
    }
}

/// `name`'s extension, after its last dot (empty without one).
fn extension(name: &str) -> &str {
    name.rsplit_once('.').map_or("", |(_, e)| e)
}

impl Directory for PagedFsDirectory {
    fn list_all(&self) -> Result<Vec<String>> {
        self.inner.list_all()
    }

    fn open(&self, name: &str) -> Result<Input> {
        #[cfg(target_os = "linux")]
        if let Some(input) = self.open_paged(name)? {
            return Ok(input);
        }
        self.inner.open(name)
    }

    fn file_length(&self, name: &str) -> Result<u64> {
        self.inner.file_length(name)
    }

    fn create_output(&self, name: &str) -> Result<FsIndexOutput> {
        self.inner.create_output(name)
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
        self.inner.create_temp_output(prefix, suffix)
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

    fn read_at(&self, name: &str, offset: u64, len: usize) -> Result<Vec<u8>> {
        use std::io::{Read, Seek, SeekFrom};
        let mut file = std::fs::File::open(self.root.join(name))?;
        file.seek(SeekFrom::Start(offset))?;
        let mut out = Vec::new();
        file.take(len as u64).read_to_end(&mut out)?;
        Ok(out)
    }

    fn fs_directory_path(&self) -> Option<&Path> {
        Some(&self.root)
    }

    #[cfg(any(test, feature = "test-util"))]
    fn disable_virus_checker(&self) -> bool {
        self.inner.disable_virus_checker()
    }
}

#[cfg(target_os = "linux")]
mod local {
    use std::alloc::Layout;
    use std::os::fd::AsRawFd;
    use std::os::unix::fs::{FileExt, OpenOptionsExt};
    use std::path::Path;
    use std::sync::{mpsc, Arc, Mutex, OnceLock};

    use crate::error::{Error, Result};
    use crate::paged_input::PageFiller;

    /// `O_DIRECT` buffers, offsets and lengths are aligned to this; 4 KiB
    /// covers every logical block size in use.
    const ALIGN: usize = 4096;

    /// One open local file, read page by page.
    pub(super) struct LocalFile {
        file: std::fs::File,
        len: u64,
        page: usize,
        direct: bool,
    }

    impl LocalFile {
        /// Opens `path` for page reads of `page` bytes, with `O_DIRECT` when
        /// `direct` and the filesystem takes it.
        pub(super) fn open(path: &Path, page: usize, direct: bool) -> Result<LocalFile> {
            let opened = if direct {
                std::fs::OpenOptions::new()
                    .read(true)
                    .custom_flags(libc::O_DIRECT)
                    .open(path)
                    .ok()
            } else {
                None
            };
            let (file, direct) = match opened {
                Some(f) => (f, true),
                None => (std::fs::File::open(path)?, false),
            };
            let len = file.metadata()?.len();
            Ok(LocalFile {
                file,
                len,
                page,
                direct,
            })
        }

        pub(super) fn len(&self) -> u64 {
            self.len
        }

        /// Every byte of the file.
        pub(super) fn read_whole(&self) -> Result<Vec<u8>> {
            let len = usize::try_from(self.len)
                .map_err(|_| Error::Corrupted("file too large to read".into()))?;
            let mut out = Vec::with_capacity(len);
            let mut offset = 0u64;
            while (out.len() as u64) < self.len {
                let n = MAX_CHUNK.min(len.saturating_sub(out.len()));
                self.read_into(offset, n, &mut |b| out.extend_from_slice(b))?;
                offset = offset.saturating_add(n as u64);
            }
            Ok(out)
        }

        /// Reads `len` bytes at `offset` (fewer at the end of the file) and
        /// hands them to `sink` in one slice.
        fn read_into(&self, offset: u64, len: usize, sink: &mut dyn FnMut(&[u8])) -> Result<()> {
            let want = usize::try_from(self.len.saturating_sub(offset))
                .unwrap_or(usize::MAX)
                .min(len);
            if want == 0 {
                return Ok(());
            }
            if self.direct && offset.is_multiple_of(ALIGN as u64) {
                let mut buf = AlignedBuf::new(want.div_ceil(ALIGN).saturating_mul(ALIGN))?;
                match read_full(&self.file, buf.as_mut(), offset) {
                    Ok(n) if n >= want => {
                        sink(&buf.as_mut()[..want]);
                        return Ok(());
                    }
                    Ok(n) => {
                        return Err(Error::Corrupted(format!(
                            "short read: {n} of {want} bytes at {offset}"
                        )))
                    }
                    // A filesystem that opened with O_DIRECT but refuses this
                    // read: fall through to a buffered read.
                    Err(e) if e.raw_os_error() == Some(libc::EINVAL) => {}
                    Err(e) => return Err(Error::Io(e)),
                }
            }
            let mut buf = vec![0u8; want];
            if self.direct {
                // The descriptor is O_DIRECT; read this range through a
                // buffered one.
                let plain = reopen_buffered(&self.file)?;
                let n = read_full(&plain, &mut buf, offset)?;
                if n < want {
                    return Err(Error::Corrupted(format!(
                        "short read: {n} of {want} bytes at {offset}"
                    )));
                }
            } else {
                let n = read_full(&self.file, &mut buf, offset)?;
                if n < want {
                    return Err(Error::Corrupted(format!(
                        "short read: {n} of {want} bytes at {offset}"
                    )));
                }
                // SAFETY: a plain syscall on an open descriptor.
                unsafe {
                    libc::posix_fadvise(
                        self.file.as_raw_fd(),
                        i64::try_from(offset).unwrap_or(i64::MAX),
                        i64::try_from(want).unwrap_or(i64::MAX),
                        libc::POSIX_FADV_DONTNEED,
                    );
                }
            }
            sink(&buf);
            Ok(())
        }
    }

    /// The most read whole in one call.
    const MAX_CHUNK: usize = super::MAX_READ;

    impl LocalFile {
        /// Another handle on the same open file (`dup`), for a read on an
        /// I/O thread.
        fn handle(&self) -> Result<LocalFile> {
            Ok(LocalFile {
                file: self.file.try_clone()?,
                len: self.len,
                page: self.page,
                direct: self.direct,
            })
        }

        /// The `count` pages from page `first`, read at once.
        fn read_run(&self, first: u64, count: usize) -> Result<Vec<Arc<[u8]>>> {
            let offset = first.saturating_mul(self.page as u64);
            let len = count.saturating_mul(self.page);
            let page = self.page;
            let mut out: Vec<Arc<[u8]>> = Vec::with_capacity(count);
            self.read_into(offset, len, &mut |bytes| {
                for chunk in bytes.chunks(page) {
                    out.push(Arc::from(chunk));
                }
            })?;
            if out.len() != count {
                return Err(Error::Corrupted(format!(
                    "page {first}: read past the end of the file"
                )));
            }
            Ok(out)
        }
    }

    /// Threads reading runs of a batch in parallel: a batch of scattered
    /// pages (a query's candidates) is read at a queue depth of up to this
    /// many, not one read after another.
    const IO_THREADS: usize = 16;

    type Job = Box<dyn FnOnce() + Send>;

    /// The I/O threads' queue, started on first use; `None` if no thread
    /// could start (reads then run on the caller).
    fn io_pool() -> Option<&'static Mutex<mpsc::Sender<Job>>> {
        static POOL: OnceLock<Option<Mutex<mpsc::Sender<Job>>>> = OnceLock::new();
        POOL.get_or_init(|| {
            let (tx, rx) = mpsc::channel::<Job>();
            let rx = Arc::new(Mutex::new(rx));
            let mut started = 0usize;
            for i in 0..IO_THREADS {
                let rx = Arc::clone(&rx);
                let spawned = std::thread::Builder::new()
                    .name(format!("lucene-page-io-{i}"))
                    .spawn(move || loop {
                        let job = match crate::lock::lock_ignoring_poison(&rx).recv() {
                            Ok(job) => job,
                            Err(_) => return,
                        };
                        job();
                    });
                if spawned.is_ok() {
                    started = started.saturating_add(1);
                }
            }
            (started > 0).then(|| Mutex::new(tx))
        })
        .as_ref()
    }

    type Pages = Vec<Arc<[u8]>>;

    impl PageFiller for LocalFile {
        fn fill(&self, pages: &[u64]) -> Result<Vec<Arc<[u8]>>> {
            // Runs of consecutive pages, each read at once.
            let per_read = MAX_CHUNK.checked_div(self.page).unwrap_or(1).max(1);
            let mut runs: Vec<(u64, usize)> = Vec::new();
            for &p in pages {
                match runs.last_mut() {
                    Some((first, count))
                        if *count < per_read && first.saturating_add(*count as u64) == p =>
                    {
                        *count = count.saturating_add(1);
                    }
                    _ => runs.push((p, 1)),
                }
            }
            let pool = if runs.len() > 1 { io_pool() } else { None };
            let Some(pool) = pool else {
                let mut out: Vec<Arc<[u8]>> = Vec::with_capacity(pages.len());
                for &(first, count) in &runs {
                    out.extend(self.read_run(first, count)?);
                }
                return Ok(out);
            };
            // Every run but the first on the I/O threads, the first here.
            let (tx, rx) = mpsc::channel::<(usize, Result<Pages>)>();
            let mut results: Vec<Option<Result<Pages>>> = (0..runs.len()).map(|_| None).collect();
            {
                let queue = crate::lock::lock_ignoring_poison(pool);
                for (k, &(first, count)) in runs.iter().enumerate().skip(1) {
                    let handle = self.handle()?;
                    let tx = tx.clone();
                    let job: Job = Box::new(move || {
                        let _ = tx.send((k, handle.read_run(first, count)));
                    });
                    if let Err(mpsc::SendError(job)) = queue.send(job) {
                        job();
                    }
                }
            }
            drop(tx);
            results[0] = Some(self.read_run(runs[0].0, runs[0].1));
            for (k, r) in rx {
                if let Some(slot) = results.get_mut(k) {
                    *slot = Some(r);
                }
            }
            let mut out: Vec<Arc<[u8]>> = Vec::with_capacity(pages.len());
            for r in results {
                match r {
                    Some(r) => out.extend(r?),
                    None => {
                        return Err(Error::Io(std::io::Error::other(
                            "a page read did not complete",
                        )))
                    }
                }
            }
            Ok(out)
        }
    }

    /// Reads until `buf` is full or the file ends; the bytes read.
    fn read_full(file: &std::fs::File, buf: &mut [u8], offset: u64) -> std::io::Result<usize> {
        let mut done = 0usize;
        while done < buf.len() {
            match file.read_at(&mut buf[done..], offset.saturating_add(done as u64)) {
                Ok(0) => break,
                Ok(n) => done = done.saturating_add(n),
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
        Ok(done)
    }

    fn reopen_buffered(file: &std::fs::File) -> Result<std::fs::File> {
        let path = format!("/proc/self/fd/{}", file.as_raw_fd());
        Ok(std::fs::File::open(path)?)
    }

    /// A zeroed heap buffer aligned for `O_DIRECT`.
    struct AlignedBuf {
        ptr: std::ptr::NonNull<u8>,
        len: usize,
    }

    impl AlignedBuf {
        fn new(len: usize) -> Result<AlignedBuf> {
            let layout = Layout::from_size_align(len.max(ALIGN), ALIGN)
                .map_err(|_| Error::IllegalArgument(format!("buffer of {len} bytes")))?;
            // SAFETY: `layout` has a non-zero size.
            let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
            let ptr = std::ptr::NonNull::new(ptr)
                .ok_or_else(|| Error::Io(std::io::Error::other("out of memory")))?;
            Ok(AlignedBuf { ptr, len })
        }

        fn as_mut(&mut self) -> &mut [u8] {
            // SAFETY: `ptr` holds at least `len` initialized bytes owned by
            // `self`.
            unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
        }
    }

    impl Drop for AlignedBuf {
        fn drop(&mut self) {
            // SAFETY: allocated in `new` with this layout.
            unsafe {
                std::alloc::dealloc(
                    self.ptr.as_ptr(),
                    Layout::from_size_align_unchecked(self.len.max(ALIGN), ALIGN),
                );
            }
        }
    }

    // SAFETY: the buffer is uniquely owned.
    unsafe impl Send for AlignedBuf {}
}

#[cfg(test)]
mod tests {
    // Test fixtures' own arithmetic -- see `docs/arithmetic-gate.md`'s "Test code".
    #![allow(clippy::arithmetic_side_effects)]
    use super::*;
    use crate::data_output::DataOutput;
    use lucene_util::test_support::TempDir;

    fn write(dir: &dyn Directory, name: &str, bytes: &[u8]) {
        let mut out = dir.create_output(name).unwrap();
        out.write_bytes(bytes);
        out.close().unwrap();
    }

    fn pattern(len: usize) -> Vec<u8> {
        (0..len)
            .map(|i| (i.wrapping_mul(31) ^ (i >> 9)) as u8)
            .collect()
    }

    #[test]
    fn reads_match_the_file() {
        let tmp = TempDir::new("paged-fs");
        let dir = PagedFsDirectory::open(tmp.path());
        // Not a whole number of pages, so the last page is short.
        for &len in &[0usize, 10, 4096, 64 << 10, (64 << 10) + 1, 1_000_003] {
            let name = format!("f{len}.dat");
            let bytes = pattern(len);
            write(&dir, &name, &bytes);
            let input = dir.open(&name).unwrap();
            assert_eq!(&input[..], &bytes[..], "len {len}");
            assert_eq!(dir.file_length(&name).unwrap(), len as u64);
            // Out-of-order touches fault pages in independently.
            if len > 0 {
                let mid = len / 2;
                assert_eq!(input[mid], bytes[mid]);
                assert_eq!(input[len - 1], bytes[len - 1]);
            }
        }
    }

    #[test]
    fn will_need_and_prefetch_fill_ahead() {
        let tmp = TempDir::new("paged-fs-prefetch");
        let dir = PagedFsDirectory::open(tmp.path());
        let bytes = pattern(3 << 20);
        write(&dir, "big.dat", &bytes);
        let input = dir.open("big.dat").unwrap();
        input.will_need(1 << 20, 512 << 10).unwrap();
        crate::prefetch(&[&input[2 << 20..(2 << 20) + 100]]).unwrap();
        assert_eq!(&input[..], &bytes[..]);
        if dir.is_paged() {
            assert!(crate::is_prefetchable(&input));
        }
    }

    #[test]
    fn pinned_files_are_read_whole() {
        let tmp = TempDir::new("paged-fs-pinned");
        let dir = PagedFsDirectory::open(tmp.path());
        let bytes = pattern(1 << 20);
        write(&dir, "_0.tip", &bytes);
        let input = dir.open("_0.tip").unwrap();
        assert!(matches!(input, Input::Owned(_)));
        assert_eq!(&input[..], &bytes[..]);
    }

    #[test]
    fn adaptive_pages_and_pins_follow_the_budget() {
        let c = PagedFsConfig::default();
        let budget = 100 << 20;
        // Fits: the default page; too large to stay resident: small pages.
        assert_eq!(c.page_size_for("_0.doc", 10 << 20, budget), c.page_size);
        assert_eq!(c.page_size_for("_0.doc", 60 << 20, budget), 16 << 10);
        assert_eq!(c.page_size_for("_0_X_0.vec", 60 << 20, budget), 4 << 10);
        // Search structures are pinned while small against the budget.
        assert!(c.pins("_0_X_0.vex", 12 << 20, budget));
        assert!(!c.pins("_0_X_0.vex", 13 << 20, budget));
        assert!(c.pins("_0.tip", 1 << 30, budget));
        assert!(!c.pins("_0.doc", 1, budget));
        // Explicit settings win; without `adaptive` sizes are fixed.
        let explicit = PagedFsConfig {
            page_sizes: vec![("vec".into(), 32 << 10)],
            ..PagedFsConfig::default()
        };
        assert_eq!(explicit.page_size_for("_0.vec", 1 << 30, budget), 32 << 10);
        let fixed = PagedFsConfig {
            adaptive: false,
            ..PagedFsConfig::default()
        };
        assert_eq!(
            fixed.page_size_for("_0.vec", 1 << 30, budget),
            fixed.page_size
        );
        assert!(!fixed.pins("_0.vex", 1, budget));
    }

    #[test]
    fn vectors_only_maps_the_other_files() {
        let tmp = TempDir::new("paged-fs-vectors-only");
        let dir = PagedFsDirectory::with_config(tmp.path(), PagedFsConfig::vectors_only());
        let bytes = pattern(1 << 20);
        write(&dir, "_0.doc", &bytes);
        write(&dir, "_0_X_0.vec", &bytes);
        let doc = dir.open("_0.doc").unwrap();
        let vec = dir.open("_0_X_0.vec").unwrap();
        assert!(matches!(doc, Input::Mapped(_)));
        assert_eq!(&doc[..], &bytes[..]);
        assert_eq!(&vec[..], &bytes[..]);
        if dir.is_paged() {
            assert!(crate::is_prefetchable(&vec));
            assert!(!crate::is_prefetchable(&doc));
        }
    }

    #[test]
    fn buffered_reads_match_too() {
        let tmp = TempDir::new("paged-fs-buffered");
        let dir = PagedFsDirectory::with_config(
            tmp.path(),
            PagedFsConfig {
                direct_io: false,
                page_size: 16 << 10,
                ..PagedFsConfig::default()
            },
        );
        let bytes = pattern(777_777);
        write(&dir, "b.dat", &bytes);
        assert_eq!(&dir.open("b.dat").unwrap()[..], &bytes[..]);
        assert_eq!(dir.read_at("b.dat", 1000, 50).unwrap(), &bytes[1000..1050]);
    }

    #[test]
    fn open_file_survives_delete() {
        let tmp = TempDir::new("paged-fs-delete");
        let dir = PagedFsDirectory::open(tmp.path());
        let bytes = pattern(1 << 20);
        write(&dir, "d.dat", &bytes);
        let input = dir.open("d.dat").unwrap();
        dir.delete_file("d.dat").unwrap();
        assert_eq!(&input[..], &bytes[..]);
    }
}
