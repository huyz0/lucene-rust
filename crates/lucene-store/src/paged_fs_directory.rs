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
//! - A file an encrypting wrapper opens through
//!   [`Directory::open_decoded`] is read the same way, its stored bytes
//!   with `O_DIRECT`, and each page is decoded (decrypted and
//!   authenticated) once, as it is read in: pages are whole decode units
//!   (chunks), the mapping holds plaintext, and the stored bytes never sit
//!   in the kernel's page cache. Only the header is read at open; a page
//!   that fails to decode fails the reads of it with an error
//!   ([`crate::read_failures`]), never the process.
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
use crate::page_cache::{PageDecoder, DEFAULT_PINNED_EXTENSIONS};

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
    fn open_paged(&self, name: &str, alignment: Option<usize>) -> Result<Option<Input>> {
        use crate::paged_input::{PageFiller, Pager};
        use std::sync::Arc;

        if self.is_mapped(name) {
            return Ok(None);
        }
        let Some(pager) = Pager::global() else {
            return Ok(None);
        };
        self.inner.ensure_can_read(name)?;
        let path = self.root.join(name);
        let mut file = local::LocalFile::open(&path, self.config.direct_io)?;
        let budget = pager.budget();
        let mut page = self.config.page_size_for(name, file.len(), budget);
        if let Some(hint) = alignment {
            // Whole blocks per page: a multiple of the format's alignment.
            page = crate::alignment::page_size_for(crate::alignment::check_hint(hint)?, page);
        }
        if !pager.supports(page) {
            return Ok(None);
        }
        file.set_page(page);
        let len = usize::try_from(file.len())
            .map_err(|_| crate::error::Error::Corrupted(format!("{name}: too large to open")))?;
        let whole = self.config.pins(name, file.len(), budget) || len <= page;
        let key = shared::Key::of(&file, page, false, whole)?;
        if let Some(input) = shared::reuse(&key) {
            return Ok(Some(input));
        }
        if whole {
            let bytes = file.read_whole()?;
            return Ok(Some(shared::share_whole(key, bytes)));
        }
        let filler: Arc<dyn PageFiller> = Arc::new(file);
        let readahead = self.config.readahead.checked_div(page).unwrap_or(0).max(1);
        let paged = pager.open_with_readahead(len, page, filler, name, readahead)?;
        Ok(Some(shared::share_paged(key, paged)))
    }

    /// [`Directory::open_decoded`] through the pager: `None` where the
    /// pager is unavailable or the decoder's units are not whole OS pages,
    /// and the caller then decodes [`Directory::open`]'s bytes.
    #[cfg(target_os = "linux")]
    fn open_decoded_paged(&self, name: &str, decoder: &dyn PageDecoder) -> Result<Option<Input>> {
        use crate::error::Error;
        use crate::paged_input::{PageFiller, Pager};
        use std::sync::Arc;

        let Some(pager) = Pager::global() else {
            return Ok(None);
        };
        self.inner.ensure_can_read(name)?;
        let path = self.root.join(name);
        let mut file = local::LocalFile::open(&path, self.config.direct_io)?;
        let codec = decoder.codec(name, file.len(), self)?;
        let layout = codec.layout();
        let too_large = || Error::Corrupted(format!("{name}: too large to open"));
        let unit = usize::try_from(layout.plain_unit).map_err(|_| too_large())?;
        if unit == 0 || layout.stored_unit == 0 || !pager.supports(unit) {
            return Ok(None);
        }
        let len = usize::try_from(layout.plain_len).map_err(|_| too_large())?;
        let budget = pager.budget();
        // Whole units per page, as close to the plain file's page as fits.
        let want = self.config.page_size_for(name, layout.plain_len, budget);
        let units = want.checked_div(unit).unwrap_or(1).max(1);
        let page = units.saturating_mul(unit);
        if !pager.supports(page) {
            return Ok(None);
        }
        file.set_page(page);
        file.set_decode(codec, layout, units as u64);
        let whole = self.config.pins(name, layout.plain_len, budget) || len <= page;
        let key = shared::Key::of(&file, page, true, whole)?;
        if let Some(input) = shared::reuse(&key) {
            return Ok(Some(input));
        }
        if whole {
            let bytes = file.read_whole_decoded(len)?;
            return Ok(Some(shared::share_whole(key, bytes)));
        }
        let filler: Arc<dyn PageFiller> = Arc::new(file);
        let readahead = self.config.readahead.checked_div(page).unwrap_or(0).max(1);
        // The full path names the file's failures (`read_failures`).
        let label = path.to_string_lossy();
        let paged = pager.open_with_readahead(len, page, filler, &label, readahead)?;
        Ok(Some(shared::share_paged(key, paged)))
    }
}

/// Bytes of the files read whole into memory at open by a
/// [`PagedFsDirectory`] (pinned files, and files of at most one page) and
/// still open: memory held outside the pager's budget.
pub fn whole_file_bytes() -> u64 {
    #[cfg(target_os = "linux")]
    return shared::whole_bytes();
    #[cfg(not(target_os = "linux"))]
    0
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
        if let Some(input) = self.open_paged(name, None)? {
            return Ok(input);
        }
        self.inner.open(name)
    }

    fn open_aligned(&self, name: &str, alignment: usize) -> Result<Input> {
        #[cfg(target_os = "linux")]
        if let Some(input) = self.open_paged(name, Some(alignment))? {
            return Ok(input);
        }
        self.inner.open_aligned(name, alignment)
    }

    fn file_length(&self, name: &str) -> Result<u64> {
        self.inner.file_length(name)
    }

    fn open_decoded(&self, name: &str, decoder: &dyn PageDecoder) -> Option<Result<Input>> {
        #[cfg(target_os = "linux")]
        return self.open_decoded_paged(name, decoder).transpose();
        #[cfg(not(target_os = "linux"))]
        {
            let _ = (name, decoder);
            None
        }
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
        self.inner.ensure_can_read(name)?;
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

/// Files open through the pager, shared by every open of the same file:
/// a merge reading a segment the search reader has open reads the pages
/// the reader already holds, and a file opened twice is read and held
/// once, as the kernel's page cache does for mapped files.
#[cfg(target_os = "linux")]
mod shared {
    use std::collections::HashMap;
    use std::os::unix::fs::MetadataExt;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Mutex, Weak};

    use crate::directory::{ExternalBytes, Input};
    use crate::error::Result;
    use crate::lock::lock_ignoring_poison;
    use crate::paged_input::PagedBytes;

    /// One file as opened: its identity on disk (device, inode, length,
    /// change time), how its pages are cut and whether they are decoded,
    /// and whether it was read whole. Lucene never rewrites a file in
    /// place, and an open file's inode cannot be reused while it is open.
    #[derive(Clone, PartialEq, Eq, Hash)]
    pub(super) struct Key {
        dev: u64,
        ino: u64,
        len: u64,
        ctime: (i64, i64),
        page: usize,
        decoded: bool,
        whole: bool,
    }

    impl Key {
        pub(super) fn of(
            file: &super::local::LocalFile,
            page: usize,
            decoded: bool,
            whole: bool,
        ) -> Result<Key> {
            let meta = file.metadata()?;
            Ok(Key {
                dev: meta.dev(),
                ino: meta.ino(),
                len: meta.len(),
                ctime: (meta.ctime(), meta.ctime_nsec()),
                page,
                decoded,
                whole,
            })
        }
    }

    enum Open {
        Paged(Weak<PagedBytes>),
        Whole(Weak<WholeBytes>),
    }

    static OPEN: Mutex<Option<HashMap<Key, Open>>> = Mutex::new(None);
    const PRUNE_EVERY: u64 = 256;
    static WHOLE_BYTES: AtomicU64 = AtomicU64::new(0);

    /// Bytes of files read whole into memory at open (pinned files and
    /// files of at most one page), held outside the pager's budget.
    pub(crate) fn whole_bytes() -> u64 {
        WHOLE_BYTES.load(Ordering::Relaxed)
    }

    /// A file read whole, counted in [`whole_bytes`] while open.
    pub(super) struct WholeBytes(Vec<u8>);

    impl Drop for WholeBytes {
        fn drop(&mut self) {
            WHOLE_BYTES.fetch_sub(self.0.len() as u64, Ordering::Relaxed);
        }
    }

    struct SharedWhole(Arc<WholeBytes>);

    impl ExternalBytes for SharedWhole {
        fn bytes(&self) -> &[u8] {
            &self.0 .0
        }
    }

    struct SharedPaged(Arc<PagedBytes>);

    impl ExternalBytes for SharedPaged {
        fn bytes(&self) -> &[u8] {
            self.0.bytes()
        }

        fn will_need(&self, offset: usize, len: usize) -> Result<()> {
            self.0.will_need(offset, len)
        }
    }

    /// Another input over the file `key` names, if it is open now.
    pub(super) fn reuse(key: &Key) -> Option<Input> {
        let mut open = lock_ignoring_poison(&OPEN);
        let map = open.as_mut()?;
        let input = match map.get(key)? {
            Open::Paged(weak) => weak
                .upgrade()
                .map(|p| Input::external_bytes(SharedPaged(p))),
            Open::Whole(weak) => weak
                .upgrade()
                .map(|w| Input::external_bytes(SharedWhole(w))),
        };
        if input.is_none() {
            map.remove(key);
        }
        input
    }

    /// Registers `paged`, just opened, under `key` and returns an input
    /// over it. If another open registered the same file meanwhile, that
    /// one is used and `paged` dropped.
    pub(super) fn share_paged(key: Key, paged: PagedBytes) -> Input {
        let paged = Arc::new(paged);
        {
            let mut open = lock_ignoring_poison(&OPEN);
            let map = open.get_or_insert_with(HashMap::new);
            prune(map);
            if let Some(Open::Paged(existing)) = map.get(&key) {
                if let Some(existing) = existing.upgrade() {
                    return Input::external_bytes(SharedPaged(existing));
                }
            }
            map.insert(key, Open::Paged(Arc::downgrade(&paged)));
        }
        // Readers check the header and footer as they open a file: both
        // ends in one batch. Only a hint; a fault retries what it misses.
        let bytes = paged.bytes();
        let footer = bytes.len().saturating_sub(crate::codec_util::FOOTER_LENGTH);
        let _ = crate::paged_input::prefetch(&[bytes.get(..1).unwrap_or(&[]), &bytes[footer..]]);
        Input::external_bytes(SharedPaged(paged))
    }

    /// [`share_paged`] for a file read whole.
    pub(super) fn share_whole(key: Key, bytes: Vec<u8>) -> Input {
        WHOLE_BYTES.fetch_add(bytes.len() as u64, Ordering::Relaxed);
        let whole = Arc::new(WholeBytes(bytes));
        let mut open = lock_ignoring_poison(&OPEN);
        let map = open.get_or_insert_with(HashMap::new);
        prune(map);
        if let Some(Open::Whole(existing)) = map.get(&key) {
            if let Some(existing) = existing.upgrade() {
                return Input::external_bytes(SharedWhole(existing));
            }
        }
        map.insert(key, Open::Whole(Arc::downgrade(&whole)));
        Input::external_bytes(SharedWhole(whole))
    }

    /// Drops the entries of closed files every [`PRUNE_EVERY`]
    /// registrations, so the map stays near the number of files open.
    fn prune(map: &mut HashMap<Key, Open>) {
        static INSERTS: AtomicU64 = AtomicU64::new(0);
        if !INSERTS
            .fetch_add(1, Ordering::Relaxed)
            .is_multiple_of(PRUNE_EVERY)
        {
            return;
        }
        map.retain(|_, open| match open {
            Open::Paged(w) => w.strong_count() > 0,
            Open::Whole(w) => w.strong_count() > 0,
        });
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
    use crate::page_cache::{PageCodec, UnitLayout};
    use crate::paged_input::{undecodable, BackgroundDone, PageFiller, PageSink};

    /// `O_DIRECT` buffers, offsets and lengths are aligned to this; 4 KiB
    /// covers every logical block size in use.
    const ALIGN: usize = 4096;

    /// One open local file, read page by page.
    pub(super) struct LocalFile {
        file: Arc<std::fs::File>,
        len: u64,
        page: usize,
        direct: bool,
        /// A buffered descriptor on the same file, opened the first time an
        /// `O_DIRECT` read is refused and shared by every handle after.
        buffered: Arc<OnceLock<std::fs::File>>,
        /// How stored pages decode, for a file opened decoded.
        decode: Option<Arc<Decode>>,
    }

    /// A decoded file's codec and geometry: page `p` is stored units
    /// `p * units .. (p + 1) * units`, decoded into `page` bytes.
    pub(super) struct Decode {
        codec: Arc<dyn PageCodec>,
        layout: UnitLayout,
        units: u64,
    }

    thread_local! {
        /// Decoded bytes of the run being read, before they are installed.
        static PLAIN: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
    }

    impl LocalFile {
        /// Opens `path` for page reads (of one OS page until
        /// [`Self::set_page`]), with `O_DIRECT` when `direct` and the
        /// filesystem takes it.
        pub(super) fn open(path: &Path, direct: bool) -> Result<LocalFile> {
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
                file: Arc::new(file),
                len,
                page: ALIGN,
                direct,
                buffered: Arc::new(OnceLock::new()),
                decode: None,
            })
        }

        /// Pages are decoded by `codec`: each is `units` stored units.
        pub(super) fn set_decode(
            &mut self,
            codec: Arc<dyn PageCodec>,
            layout: UnitLayout,
            units: u64,
        ) {
            self.decode = Some(Arc::new(Decode {
                codec,
                layout,
                units,
            }));
        }

        /// Every decoded byte of the file (`len` of them).
        pub(super) fn read_whole_decoded(&self, len: usize) -> Result<Vec<u8>> {
            let mut out = vec![0u8; len];
            let page = self.page.max(1);
            let pages = len.div_ceil(page) as u64;
            let per_read = MAX_CHUNK.checked_div(page).unwrap_or(1).max(1) as u64;
            let mut first = 0u64;
            while first < pages {
                let count = per_read.min(pages.saturating_sub(first));
                let into = Arc::new(Mutex::new(std::mem::take(&mut out)));
                let target = Arc::clone(&into);
                let sink: PageSink = Arc::new(move |p, bytes| {
                    let mut out = crate::lock::lock_ignoring_poison(&target);
                    let at = (p as usize).saturating_mul(page);
                    if let Some(dst) = out.get_mut(at..at.saturating_add(bytes.len())) {
                        dst.copy_from_slice(bytes);
                    }
                });
                let read = self.read_run(first, count as usize, &sink);
                drop(sink);
                out = std::mem::take(&mut *crate::lock::lock_ignoring_poison(&into));
                read.map_err(|e| match e {
                    Error::Io(ref io) if io.kind() == std::io::ErrorKind::InvalidData => {
                        Error::Corrupted(io.to_string())
                    }
                    other => other,
                })?;
                first = first.saturating_add(count);
            }
            Ok(out)
        }

        pub(super) fn len(&self) -> u64 {
            self.len
        }

        /// The open file's metadata.
        pub(super) fn metadata(&self) -> Result<std::fs::Metadata> {
            Ok(self.file.metadata()?)
        }

        /// Sets the bytes per page the pager fills.
        pub(super) fn set_page(&mut self, page: usize) {
            self.page = page;
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
            if self.direct {
                // O_DIRECT reads aligned blocks: read the ones around the
                // range (a decoded file's pages start after its header).
                let skip = offset.checked_rem(ALIGN as u64).unwrap_or(0) as usize;
                let start = offset.saturating_sub(skip as u64);
                let end = skip.saturating_add(want);
                let aligned = end.div_ceil(ALIGN).saturating_mul(ALIGN);
                let read = with_scratch(aligned, |buf| {
                    match read_full(&self.file, buf, start) {
                        Ok(n) if n >= end => {
                            sink(&buf[skip..end]);
                            Ok(true)
                        }
                        Ok(n) => Err(Error::Corrupted(format!(
                            "short read: {} of {want} bytes at {offset}",
                            n.saturating_sub(skip)
                        ))),
                        // A filesystem that opened with O_DIRECT but refuses
                        // this read: fall through to a buffered read.
                        Err(e) if e.raw_os_error() == Some(libc::EINVAL) => Ok(false),
                        Err(e) => Err(Error::Io(e)),
                    }
                })??;
                if read {
                    return Ok(());
                }
            }
            with_scratch(want, |buf| self.read_buffered(offset, buf, sink))?
        }

        /// [`Self::read_into`] through the page cache, whose copy is
        /// dropped after: the bytes are held in the budget.
        fn read_buffered(
            &self,
            offset: u64,
            buf: &mut [u8],
            sink: &mut dyn FnMut(&[u8]),
        ) -> Result<()> {
            let want = buf.len();
            // The descriptor is O_DIRECT; read this range through a buffered
            // one, opened once.
            let plain = if self.direct {
                match self.buffered.get() {
                    Some(f) => f,
                    None => {
                        let f = reopen_buffered(&self.file)?;
                        self.buffered.get_or_init(|| f)
                    }
                }
            } else {
                &self.file
            };
            let n = read_full(plain, buf, offset)?;
            if n < want {
                return Err(Error::Corrupted(format!(
                    "short read: {n} of {want} bytes at {offset}"
                )));
            }
            // The bytes are held in the budget: drop the kernel's copy.
            // SAFETY: a plain syscall on an open descriptor.
            unsafe {
                libc::posix_fadvise(
                    plain.as_raw_fd(),
                    i64::try_from(offset).unwrap_or(i64::MAX),
                    i64::try_from(want).unwrap_or(i64::MAX),
                    libc::POSIX_FADV_DONTNEED,
                );
            }
            sink(buf);
            Ok(())
        }
    }

    /// Scratch buffers up to this size stay with their thread for the
    /// next read; larger ones are freed after it.
    const KEPT_SCRATCH: usize = 2 << 20;

    thread_local! {
        static SCRATCH: std::cell::RefCell<Option<AlignedBuf>> =
            const { std::cell::RefCell::new(None) };
    }

    /// Runs `f` on an aligned buffer of `len` bytes, the thread's own when
    /// it is big enough, so a read allocates and zeroes nothing.
    fn with_scratch<R>(len: usize, f: impl FnOnce(&mut [u8]) -> R) -> Result<R> {
        SCRATCH.with(|cell| {
            let Ok(mut slot) = cell.try_borrow_mut() else {
                let mut buf = AlignedBuf::new(len)?;
                return Ok(f(&mut buf.as_mut()[..len]));
            };
            let mut buf = match slot.take() {
                Some(b) if b.len >= len => b,
                kept => {
                    // Too small for this read: keep it for the next one
                    // unless the new buffer replaces it.
                    *slot = kept;
                    AlignedBuf::new(len)?
                }
            };
            let r = f(&mut buf.as_mut()[..len]);
            if buf.len <= KEPT_SCRATCH {
                *slot = Some(buf);
            }
            Ok(r)
        })
    }

    /// The most read whole in one call.
    const MAX_CHUNK: usize = super::MAX_READ;

    impl LocalFile {
        /// Another handle on the same open file (`dup`), for a read on an
        /// I/O thread.
        fn handle(&self) -> LocalFile {
            LocalFile {
                file: Arc::clone(&self.file),
                len: self.len,
                page: self.page,
                direct: self.direct,
                buffered: Arc::clone(&self.buffered),
                decode: self.decode.clone(),
            }
        }

        /// The `count` pages from page `first`, read at once and handed to
        /// `sink` last to first.
        fn read_run(&self, first: u64, count: usize, sink: &PageSink) -> Result<()> {
            if let Some(decode) = &self.decode {
                return self.read_run_decoded(decode, first, count, sink);
            }
            let offset = first.saturating_mul(self.page as u64);
            let len = count.saturating_mul(self.page);
            let page = self.page;
            let mut delivered = 0usize;
            self.read_into(offset, len, &mut |bytes| {
                for (i, chunk) in bytes.chunks(page).enumerate().rev() {
                    sink(first.saturating_add(i as u64), chunk);
                }
                delivered = bytes.len().div_ceil(page.max(1));
            })?;
            if delivered != count {
                return Err(Error::Corrupted(format!(
                    "page {first}: read past the end of the file"
                )));
            }
            Ok(())
        }
    }

    impl LocalFile {
        /// [`Self::read_run`] for a decoded file: the run's stored units in
        /// one read, decoded at once, handed to `sink` page by page, last
        /// to first. A run that does not decode is [`undecodable`].
        fn read_run_decoded(
            &self,
            decode: &Decode,
            first: u64,
            count: usize,
            sink: &PageSink,
        ) -> Result<()> {
            let layout = decode.layout;
            let page = self.page;
            let stored_page = decode.units.saturating_mul(layout.stored_unit);
            let offset = layout
                .origin
                .saturating_add(first.saturating_mul(stored_page));
            let stored_len = usize::try_from(stored_page.saturating_mul(count as u64))
                .map_err(|_| Error::Corrupted("decoded run too large".into()))?;
            let plain_start = first.saturating_mul(page as u64);
            let plain_want = usize::try_from(
                (count as u64)
                    .saturating_mul(page as u64)
                    .min(layout.plain_len.saturating_sub(plain_start)),
            )
            .map_err(|_| Error::Corrupted("decoded run too large".into()))?;
            if plain_want == 0 {
                return Err(Error::Corrupted(format!(
                    "page {first}: read past the end of the file"
                )));
            }
            let mut outcome: Result<()> = Err(Error::Corrupted(format!(
                "page {first}: read past the end of the file"
            )));
            self.read_into(offset, stored_len, &mut |stored| {
                outcome = PLAIN.with(|cell| {
                    let mut plain = cell.borrow_mut();
                    if plain.len() < plain_want {
                        plain.resize(plain_want, 0);
                    }
                    let out = &mut plain[..plain_want];
                    decode
                        .codec
                        .decode(first.saturating_mul(decode.units), stored, out)
                        .map_err(|e| undecodable(&e))?;
                    for (i, bytes) in out.chunks(page).enumerate().rev() {
                        sink(first.saturating_add(i as u64), bytes);
                    }
                    Ok(())
                });
            })
            .map_err(|e| match e {
                // Stored bytes missing (truncated): as undecodable as bad ones.
                Error::Corrupted(_) => undecodable(&e),
                other => other,
            })?;
            outcome
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

    impl LocalFile {
        /// `pages` (sorted, distinct) as runs of consecutive pages, each
        /// read at once.
        fn runs(&self, pages: &[u64]) -> Vec<(u64, usize)> {
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
            runs
        }
    }

    impl PageFiller for LocalFile {
        fn fill(&self, pages: &[u64]) -> Result<Vec<Arc<[u8]>>> {
            type Got = Mutex<Vec<(u64, Arc<[u8]>)>>;
            let got: Arc<Got> = Arc::default();
            let into = Arc::clone(&got);
            let sink: PageSink = Arc::new(move |page, bytes| {
                crate::lock::lock_ignoring_poison(&into).push((page, Arc::from(bytes)));
            });
            self.fill_into(pages, &sink)?;
            let mut got = std::mem::take(&mut *crate::lock::lock_ignoring_poison(&got));
            got.sort_unstable_by_key(|&(page, _)| page);
            Ok(got.into_iter().map(|(_, bytes)| bytes).collect())
        }

        fn fill_into(&self, pages: &[u64], sink: &PageSink) -> Result<()> {
            let runs = self.runs(pages);
            let pool = if runs.len() > 1 { io_pool() } else { None };
            let Some(pool) = pool else {
                for &(first, count) in &runs {
                    self.read_run(first, count, sink)?;
                }
                return Ok(());
            };
            // Every run but the first on the I/O threads, the first here.
            let (tx, rx) = mpsc::channel::<Result<()>>();
            {
                let queue = crate::lock::lock_ignoring_poison(pool);
                for &(first, count) in runs.iter().skip(1) {
                    let handle = self.handle();
                    let sink = Arc::clone(sink);
                    let tx = tx.clone();
                    let job: Job = Box::new(move || {
                        let _ = tx.send(handle.read_run(first, count, &sink));
                    });
                    if let Err(mpsc::SendError(job)) = queue.send(job) {
                        job();
                    }
                }
            }
            drop(tx);
            let mut result = self.read_run(runs[0].0, runs[0].1, sink);
            let mut done = 1usize;
            for r in rx {
                done = done.saturating_add(1);
                if result.is_ok() {
                    result = r;
                }
            }
            if result.is_ok() && done != runs.len() {
                result = Err(Error::Io(std::io::Error::other(
                    "a page read did not complete",
                )));
            }
            result
        }

        fn fill_background(&self, pages: Vec<u64>, sink: PageSink, done: BackgroundDone) -> bool {
            let Some(pool) = io_pool() else {
                return false;
            };
            // One job per run, so scattered pages are read in parallel;
            // the last job to finish reports.
            struct Batch {
                left: std::sync::atomic::AtomicUsize,
                result: Mutex<Result<()>>,
                done: Mutex<Option<BackgroundDone>>,
            }
            let runs = self.runs(&pages);
            let batch = Arc::new(Batch {
                left: std::sync::atomic::AtomicUsize::new(runs.len()),
                result: Mutex::new(Ok(())),
                done: Mutex::new(Some(done)),
            });
            let queue = crate::lock::lock_ignoring_poison(pool);
            for (first, count) in runs {
                let handle = self.handle();
                let sink = Arc::clone(&sink);
                let batch = Arc::clone(&batch);
                let job: Job = Box::new(move || {
                    let r = handle.read_run(first, count, &sink);
                    if r.is_err() {
                        let mut slot = crate::lock::lock_ignoring_poison(&batch.result);
                        if slot.is_ok() {
                            *slot = r;
                        }
                    }
                    if batch.left.fetch_sub(1, std::sync::atomic::Ordering::AcqRel) == 1 {
                        let result = std::mem::replace(
                            &mut *crate::lock::lock_ignoring_poison(&batch.result),
                            Ok(()),
                        );
                        if let Some(done) = crate::lock::lock_ignoring_poison(&batch.done).take() {
                            done(result);
                        }
                    }
                });
                // No I/O thread left: run it here.
                if let Err(mpsc::SendError(job)) = queue.send(job) {
                    job();
                }
            }
            true
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
        if dir.is_paged() {
            assert!(crate::is_prefetchable(&input));
            assert!(crate::is_resident(&input[1 << 20..(1 << 20) + 10]));
            assert!(!crate::is_resident(&input[(7 << 18)..(7 << 18) + 10]));
        }
        assert_eq!(&input[..], &bytes[..]);
    }

    #[test]
    fn pinned_files_are_read_whole() {
        let tmp = TempDir::new("paged-fs-pinned");
        let dir = PagedFsDirectory::open(tmp.path());
        let bytes = pattern(1 << 20);
        write(&dir, "_0.tip", &bytes);
        let before = whole_file_bytes();
        let input = dir.open("_0.tip").unwrap();
        assert!(!crate::paged_input::is_paged(&input));
        assert_eq!(&input[..], &bytes[..]);
        assert!(whole_file_bytes() >= before + (1 << 20));
        // A second open shares the bytes read at the first.
        let again = dir.open("_0.tip").unwrap();
        assert_eq!(again.as_ptr(), input.as_ptr());
    }

    #[test]
    fn opens_of_one_file_share_its_pages() {
        if crate::paged_input::Pager::global().is_none() {
            return;
        }
        let tmp = TempDir::new("paged-fs-shared");
        let dir = PagedFsDirectory::open(tmp.path());
        let bytes = pattern(4 << 20);
        write(&dir, "_0.doc", &bytes);
        let a = dir.open("_0.doc").unwrap();
        let b = PagedFsDirectory::open(tmp.path()).open("_0.doc").unwrap();
        assert_eq!(a.as_ptr(), b.as_ptr());
        assert_eq!(b[3 << 20], bytes[3 << 20]);
        assert!(crate::is_resident(&a[3 << 20..(3 << 20) + 1]));
        drop((a, b));
        // A file rewritten under the same name is a new file.
        let other = pattern(5 << 20);
        dir.delete_file("_0.doc").unwrap();
        write(&dir, "_0.doc", &other);
        let c = dir.open("_0.doc").unwrap();
        assert_eq!(&c[..], &other[..]);
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
        if dir.is_paged() {
            assert!(crate::is_prefetchable(&vec));
            assert!(!crate::is_prefetchable(&doc));
        }
        assert_eq!(&doc[..], &bytes[..]);
        assert_eq!(&vec[..], &bytes[..]);
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
    fn aligned_files_get_whole_blocks_per_page() {
        if crate::paged_input::Pager::global().is_none() {
            return;
        }
        let tmp = TempDir::new("paged-fs-aligned");
        let dir = PagedFsDirectory::with_config(
            tmp.path(),
            PagedFsConfig {
                page_size: 4 << 10,
                adaptive: false,
                ..PagedFsConfig::default()
            },
        );
        let bytes = pattern(1 << 20);
        write(&dir, "_0.fdt", &bytes);
        let input = dir.open_aligned("_0.fdt", 16 << 10).unwrap();
        assert_eq!(&input[..], &bytes[..]);
        assert!(dir.open_aligned("_0.fdt", 12 << 10).is_err());
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

    /// A test encoding: a 37-byte header (so units start unaligned), then
    /// units of `UNIT` bytes XOR 0x5A followed by a 4-byte sum of them.
    const UNIT: usize = 4096;
    const HEADER: usize = 37;

    fn sum(index: u64, plain: &[u8]) -> u32 {
        plain.iter().fold(index as u32, |a, &b| {
            a.wrapping_mul(31).wrapping_add(u32::from(b))
        })
    }

    fn encode(plain: &[u8]) -> Vec<u8> {
        let mut out = vec![7u8; HEADER];
        for (i, unit) in plain.chunks(UNIT).enumerate() {
            out.extend(unit.iter().map(|b| b ^ 0x5A));
            out.extend_from_slice(&sum(i as u64, unit).to_le_bytes());
        }
        out
    }

    struct SumCodec {
        plain_len: u64,
    }

    impl crate::page_cache::PageCodec for SumCodec {
        fn layout(&self) -> crate::page_cache::UnitLayout {
            crate::page_cache::UnitLayout {
                origin: HEADER as u64,
                stored_unit: UNIT as u64 + 4,
                plain_unit: UNIT as u64,
                plain_len: self.plain_len,
            }
        }

        fn decode(&self, first_unit: u64, stored: &[u8], out: &mut [u8]) -> Result<()> {
            let mut at = 0;
            for (i, unit) in stored.chunks(UNIT + 4).enumerate() {
                let n = unit.len() - 4;
                let plain = &mut out[at..at + n];
                for (o, b) in plain.iter_mut().zip(&unit[..n]) {
                    *o = b ^ 0x5A;
                }
                let want = u32::from_le_bytes(unit[n..].try_into().unwrap());
                if sum(first_unit + i as u64, plain) != want {
                    return Err(crate::error::Error::Corrupted(format!(
                        "unit {} failed its check",
                        first_unit + i as u64
                    )));
                }
                at += n;
            }
            assert_eq!(at, out.len(), "the stored units decode to exactly `out`");
            Ok(())
        }
    }

    struct SumDecoder;

    impl PageDecoder for SumDecoder {
        fn codec(
            &self,
            name: &str,
            stored_len: u64,
            dir: &dyn Directory,
        ) -> Result<std::sync::Arc<dyn crate::page_cache::PageCodec>> {
            assert_eq!(dir.read_at(name, 0, HEADER)?, vec![7u8; HEADER]);
            let body = stored_len - HEADER as u64;
            let units = body.div_ceil(UNIT as u64 + 4);
            Ok(std::sync::Arc::new(SumCodec {
                plain_len: body - 4 * units,
            }))
        }
    }

    fn decoded_dir(tmp: &TempDir, page: usize, pinned: bool) -> PagedFsDirectory {
        PagedFsDirectory::with_config(
            tmp.path(),
            PagedFsConfig {
                page_size: page,
                adaptive: false,
                pinned_extensions: if pinned {
                    vec!["fdt".into()]
                } else {
                    Vec::new()
                },
                ..PagedFsConfig::default()
            },
        )
    }

    /// Decoded files read back through the pager at every tail length and
    /// with pages of several units, pinned files read whole, and the
    /// stored bytes are never what a reader sees.
    #[test]
    fn decoded_files_read_back_page_by_page() {
        if crate::paged_input::Pager::global().is_none() {
            return;
        }
        let tmp = TempDir::new("paged-fs-decoded");
        for (page, pinned) in [(UNIT, false), (3 * UNIT, false), (UNIT, true)] {
            let dir = decoded_dir(&tmp, page, pinned);
            for len in [0usize, 1, UNIT - 1, UNIT, UNIT + 1, 10 * UNIT + 5, 300_000] {
                let plain = pattern(len);
                let name = format!("_{len}_{page}_{pinned}.fdt");
                write(&dir, &name, &encode(&plain));
                let input = dir.open_decoded(&name, &SumDecoder).unwrap().unwrap();
                assert_eq!(input.len(), len);
                // Out of order first, then all of it.
                for at in (0..len).step_by(UNIT * 2 + 7).rev() {
                    assert_eq!(input[at], plain[at], "{name}[{at}]");
                }
                assert!(input[..] == plain[..], "{name}");
                input.will_need(0, len).unwrap();
            }
        }
        // Where units are not whole OS pages the caller decodes instead.
        struct Odd;
        impl PageDecoder for Odd {
            fn codec(
                &self,
                _: &str,
                _: u64,
                _: &dyn Directory,
            ) -> Result<std::sync::Arc<dyn crate::page_cache::PageCodec>> {
                struct C;
                impl crate::page_cache::PageCodec for C {
                    fn layout(&self) -> crate::page_cache::UnitLayout {
                        crate::page_cache::UnitLayout {
                            origin: 0,
                            stored_unit: 100,
                            plain_unit: 100,
                            plain_len: 1000,
                        }
                    }
                    fn decode(&self, _: u64, _: &[u8], _: &mut [u8]) -> Result<()> {
                        unreachable!()
                    }
                }
                Ok(std::sync::Arc::new(C))
            }
        }
        let dir = decoded_dir(&tmp, UNIT, false);
        write(&dir, "odd.fdt", &[0u8; 1000]);
        assert!(dir.open_decoded("odd.fdt", &Odd).is_none());
    }

    /// A unit that fails its check fails announced reads with an error and
    /// reads as zeros otherwise, recorded under the file's path until the
    /// input is dropped; the process carries on. Truncation is reported the
    /// same way.
    #[test]
    fn a_bad_unit_fails_reads_with_an_error_not_an_abort() {
        // A closed file's record goes once the pager's last hold on it
        // (a read in flight) ends.
        let cleared = |root: &str| {
            (0..200).any(|_| {
                let gone = crate::read_failures::failure_under(root, crate::read_failures::epoch())
                    .is_none();
                if !gone {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
                gone
            })
        };
        if crate::paged_input::Pager::global().is_none() {
            return;
        }
        let tmp = TempDir::new("paged-fs-decoded-bad");
        let root = tmp.path().to_string_lossy().to_string();
        let dir = decoded_dir(&tmp, UNIT, false);
        let plain = pattern(20 * UNIT);
        write(&dir, "_1.fdt", &encode(&plain));
        let input = dir.open_decoded("_1.fdt", &SumDecoder).unwrap().unwrap();
        assert_eq!(input[3 * UNIT], plain[3 * UNIT]);
        assert_eq!(
            crate::read_failures::failure_under(&root, crate::read_failures::epoch()),
            None
        );
        // Corrupt unit 7 on disk under the open input.
        let path = tmp.path().join("_1.fdt");
        let mut raw = std::fs::read(&path).unwrap();
        raw[HEADER + 7 * (UNIT + 4) + 100] ^= 1;
        std::fs::write(&path, &raw).unwrap();
        assert_eq!(&input[7 * UNIT..7 * UNIT + 16], &[0u8; 16]);
        let recorded = crate::read_failures::failure_under(&root, crate::read_failures::epoch())
            .expect("recorded");
        assert!(recorded.contains("unit 7 failed its check"), "{recorded}");
        let err = input.will_need(0, 1).unwrap_err();
        assert!(matches!(err, crate::error::Error::Corrupted(_)), "{err}");
        assert!(crate::prefetch(&[&input[..1]]).is_err());
        // Other pages still read.
        assert_eq!(input[12 * UNIT + 9], plain[12 * UNIT + 9]);
        drop(input);
        assert!(cleared(&root));

        // A unit corrupt at open time fails a pinned (whole) read.
        let pinned = decoded_dir(&tmp, UNIT, true);
        assert!(matches!(
            pinned.open_decoded("_1.fdt", &SumDecoder).unwrap(),
            Err(crate::error::Error::Corrupted(_))
        ));
        // A file truncated under an open input: its missing units fail too.
        std::fs::write(&path, encode(&plain)).unwrap();
        let input = dir.open_decoded("_1.fdt", &SumDecoder).unwrap().unwrap();
        let f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        f.set_len((HEADER + 15 * (UNIT + 4)) as u64).unwrap();
        assert!(input.will_need(17 * UNIT, 1).is_err());
        assert_eq!(input[18 * UNIT], 0);
        assert!(
            crate::read_failures::failure_under(&root, crate::read_failures::epoch()).is_some()
        );
        drop(input);
        assert!(cleared(&root));
    }
}
