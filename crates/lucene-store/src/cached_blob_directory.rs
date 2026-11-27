//! A [`Directory`] over a [`BlobStore`] that reads files by range through a
//! [`PageCache`], so an index on an object store costs local disk and memory
//! for the pages its queries touch, not for whole files.
//!
//! - Writes, listing, renames and deletes go to the store as
//!   [`BlobDirectory`] sends them.
//! - Opening a file asks the store for its length only, and not even that
//!   when the store's listing carries lengths ([`BlobStore::list_with_sizes`]). Its bytes come in
//!   fixed pages on first touch ([`crate::paged_input`], Linux
//!   `userfaultfd`) or ahead of the reads through [`Input::will_need`], each
//!   page from the cache's disk tier when it holds it, else by ranged reads
//!   ([`BlobStore::get_range_into`]) that merge adjacent pages.
//! - Files read on every query or whole at open (metadata, the terms
//!   index, norms, the stored-fields and points indexes, live docs: see
//!   [`crate::page_cache::DEFAULT_PINNED_EXTENSIONS`]) and files of one page
//!   are read whole into memory, their pages pinned on disk.
//! - An encrypting wrapper above ([`Directory::open_decoded`]) gets pages of
//!   whole authenticated units: the disk tier keeps ciphertext and only the
//!   mapping in memory holds plaintext, decrypted once per page brought in.
//! - Without `userfaultfd` (not Linux, blocked, or [`PageCacheConfig::demand_paging`]
//!   off), or with a page size that is not a multiple of 4 KiB, a file of at
//!   least [`PageCacheConfig::spill_min_bytes`] is streamed in bounded chunks
//!   into an unlinked file under the disk tier's directory and mapped
//!   read-only, so the heap never holds it whole; smaller files, pinned
//!   files, files through a decoder and caches without a disk tier are read
//!   whole.
//! - Every page's cache key carries [`BlobStore::version`] when the store
//!   has one, so a name rewritten with other content of the same length
//!   cannot be served the old pages.
//!
//! The only part a backend supplies is the [`BlobStore`]: listing, whole and
//! ranged reads, uploads, deletes.

use std::collections::{BTreeSet, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

use crate::blob_directory::{BlobDirectory, BlobStore};
use crate::directory::{Directory, Input};
use crate::error::{Error, Result};
use crate::index_output::FsIndexOutput;
use crate::lock::{lock_ignoring_poison, Lock, LockFactory, SingleInstanceLockFactory};
use crate::page_cache::SpillReservation;
use crate::page_cache::{Geometry, PageCache, PageCacheConfig, PageCodec, PageDecoder};
#[cfg(target_os = "linux")]
use crate::paged_input::PageFiller;

/// A [`Directory`] over a [`BlobStore`], read through a shared
/// [`PageCache`]. See the module docs.
pub struct CachedBlobDirectory<S: BlobStore + 'static> {
    store: Arc<S>,
    blob: BlobDirectory<S>,
    cache: Arc<PageCache>,
    namespace: Arc<str>,
    open: Mutex<HashMap<String, Vec<Opened>>>,
    /// Files deleted while open: removed from the store once no input
    /// over them is left, since an open file's pages are fetched from it.
    pending: Mutex<BTreeSet<String>>,
    /// File lengths, from the store's listing or a length request: files
    /// are written once, so a length holds until the name is deleted or
    /// written again.
    sizes: Mutex<HashMap<String, u64>>,
    /// The [`BlobStore::version`] each name had when last opened: a change
    /// drops the cached length and codec, which described the old content.
    versions: Mutex<HashMap<String, Option<Arc<str>>>>,
    /// Codecs built by a prefetch, with the stored length they were built
    /// for, taken by the open that follows so a decoder (an encrypted
    /// file's header and data key) runs once per file.
    codecs: Mutex<HashMap<String, Built>>,
    /// Bumped by every [`Self::forget`]: a listing that started before one
    /// does not put back the lengths it read.
    forgets: AtomicU64,
    /// Whether a listing that shows a commit not yet seen primes the cache from
    /// the open bundle ([`crate::open_bundle`]).
    open_bundle: bool,
    /// The commit generation the cache was last primed for.
    primed_generation: Mutex<Option<u64>>,
}

/// A codec and the stored length it was built for.
type Built = (u64, Arc<dyn PageCodec>);

/// A file open now: shared by later opens of the same bytes.
struct Opened {
    stored_len: u64,
    version: Option<Arc<str>>,
    decoded: bool,
    input: Weak<Input>,
}

impl<S: BlobStore + 'static> std::fmt::Debug for CachedBlobDirectory<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CachedBlobDirectory(namespace={})", self.namespace)
    }
}

impl<S: BlobStore + 'static> CachedBlobDirectory<S> {
    /// A directory over `store` reading through `cache`. `namespace` tells
    /// this store's files apart from other stores' in a shared cache (a
    /// bucket and prefix, a shard id).
    pub fn new(store: S, cache: Arc<PageCache>, namespace: &str) -> Self {
        Self::with_lock_factory(
            store,
            cache,
            namespace,
            Arc::new(SingleInstanceLockFactory::new()),
        )
    }

    /// [`CachedBlobDirectory::new`] whose `write.lock` is `lock_factory`'s.
    pub fn with_lock_factory(
        store: S,
        cache: Arc<PageCache>,
        namespace: &str,
        lock_factory: Arc<dyn LockFactory>,
    ) -> Self {
        let store = Arc::new(store);
        Self {
            blob: BlobDirectory::from_shared(Arc::clone(&store), lock_factory),
            store,
            cache,
            namespace: Arc::from(namespace),
            open: Mutex::new(HashMap::new()),
            pending: Mutex::new(BTreeSet::new()),
            sizes: Mutex::new(HashMap::new()),
            versions: Mutex::new(HashMap::new()),
            codecs: Mutex::new(HashMap::new()),
            forgets: AtomicU64::new(0),
            open_bundle: true,
            primed_generation: Mutex::new(None),
        }
    }

    /// Switches the open bundle on or off (on by default). On, the directory
    /// writes it with every published commit and merge, and fills the cache
    /// from it the first time a listing shows a commit, so a shard opens with
    /// one request for its small files instead of several per segment. A
    /// missing, stale or damaged bundle only means the ordinary ranged reads.
    pub fn with_open_bundle(mut self, enabled: bool) -> Self {
        self.open_bundle = enabled;
        self.blob.set_open_bundle(enabled);
        self
    }

    /// The store underneath.
    pub fn store(&self) -> &S {
        &self.store
    }

    /// The page cache.
    pub fn cache(&self) -> &Arc<PageCache> {
        &self.cache
    }

    /// Fills the cache from the store's open bundle
    /// ([`crate::open_bundle`]): one request instead of the reads opening
    /// each segment's small files would send. Entries the listing does not
    /// confirm (same name and length) are skipped, and a missing or damaged
    /// bundle leaves everything to the ordinary ranged reads.
    pub fn prime_from_open_bundle(&self) -> Result<crate::open_bundle::BundlePrimed> {
        if !self.cache.has_disk() {
            return Ok(Default::default());
        }
        match self.store.list_with_sizes()? {
            Some(listing) => self.prime_with_listing(&listing),
            None => Ok(Default::default()),
        }
    }

    fn prime_with_listing(
        &self,
        listing: &[(String, u64)],
    ) -> Result<crate::open_bundle::BundlePrimed> {
        use crate::open_bundle::{parse_open_bundle, BundlePrimed, OPEN_BUNDLE_NAME};
        let mut stats = BundlePrimed::default();
        if !self.cache.has_disk() {
            return Ok(stats);
        }
        let bundle = match self.store.get(OPEN_BUNDLE_NAME) {
            Ok(b) => b,
            Err(e) if e.is_no_such_file() => return Ok(stats),
            Err(e) => return Err(e),
        };
        let Some(entries) = parse_open_bundle(&bundle) else {
            return Ok(stats);
        };
        stats.found = true;
        let sizes: HashMap<&str, u64> = listing.iter().map(|(n, l)| (n.as_str(), *l)).collect();
        let page_size = self.cache.config().page_size;
        for (name, bytes) in entries {
            if sizes.get(name.as_str()).copied() != Some(bytes.len() as u64) {
                stats.stale = stats.stale.saturating_add(1);
                continue;
            }
            let version = self.current_version(&name)?;
            let prefix = self.key_prefix(&name);
            let key = match version.as_deref() {
                None => format!("{prefix}{}\u{0}p", bytes.len()),
                Some(v) => format!("{prefix}{}\u{0}p\u{0}v{}:{v}", bytes.len(), v.len()),
            };
            if self.cache.is_pinned_name(&name) {
                self.cache.pin(&key);
            }
            for (i, page) in bytes.chunks(page_size).enumerate() {
                self.cache.insert_page(&key, i as u64, page);
            }
            lock_ignoring_poison(&self.sizes).insert(name, bytes.len() as u64);
            stats.primed = stats.primed.saturating_add(1);
            stats.bytes = stats.bytes.saturating_add(bytes.len() as u64);
        }
        Ok(stats)
    }

    /// Primes the cache from the open bundle when `listed` shows a commit the
    /// cache was not primed for: the first listing of a shard, and each commit
    /// a reopen sees. Best-effort: any failure leaves the ranged reads.
    fn prime_for_new_commit(&self, listed: &[(String, u64)]) {
        if !self.open_bundle {
            return;
        }
        let newest = listed
            .iter()
            .filter_map(|(n, _)| u64::from_str_radix(n.strip_prefix("segments_")?, 36).ok())
            .max();
        let Some(newest) = newest else { return };
        {
            let mut primed = lock_ignoring_poison(&self.primed_generation);
            if *primed == Some(newest) {
                return;
            }
            *primed = Some(newest);
        }
        let _ = self.prime_with_listing(listed);
    }

    fn key_prefix(&self, name: &str) -> String {
        format!("{}\u{0}{name}\u{0}", self.namespace)
    }

    /// Drops what is cached of `name`.
    fn forget(&self, name: &str) {
        // Bumped before the length goes, so a listing holding the lengths'
        // lock either sees the bump or is done before the removal.
        self.forgets.fetch_add(1, Ordering::AcqRel);
        lock_ignoring_poison(&self.open).remove(name);
        lock_ignoring_poison(&self.sizes).remove(name);
        lock_ignoring_poison(&self.versions).remove(name);
        lock_ignoring_poison(&self.codecs).remove(name);
        self.cache.forget_prefix(&self.key_prefix(name));
    }

    /// `name`'s stored length, asked of the store only when not known.
    fn stored_len(&self, name: &str) -> Result<u64> {
        if let Some(&len) = lock_ignoring_poison(&self.sizes).get(name) {
            return Ok(len);
        }
        let len = self.store.size(name)?;
        lock_ignoring_poison(&self.sizes).insert(name.to_string(), len);
        Ok(len)
    }

    /// `name`'s [`BlobStore::version`] now. A version that differs from the
    /// one last seen means the object was replaced: what was cached of its
    /// length and codec is dropped (its pages sit under the old version's
    /// key and age out).
    fn current_version(&self, name: &str) -> Result<Option<Arc<str>>> {
        // Asked under the lock (the store answers from memory), so two
        // opens cannot record their answers out of order.
        let mut seen = lock_ignoring_poison(&self.versions);
        let version: Option<Arc<str>> = self.store.version(name)?.map(Arc::from);
        match seen.get(name) {
            Some(old) if *old == version => {}
            Some(_) => {
                // Bumped before the length goes, as `forget` does.
                self.forgets.fetch_add(1, Ordering::AcqRel);
                lock_ignoring_poison(&self.sizes).remove(name);
                lock_ignoring_poison(&self.codecs).remove(name);
                seen.insert(name.to_string(), version.clone());
            }
            None => {
                seen.insert(name.to_string(), version.clone());
            }
        }
        Ok(version)
    }

    /// Whether an input over `name` is open now.
    fn is_open(&self, name: &str) -> bool {
        lock_ignoring_poison(&self.open)
            .get(name)
            .is_some_and(|all| all.iter().any(|o| o.input.strong_count() > 0))
    }

    fn is_pending(&self, name: &str) -> bool {
        lock_ignoring_poison(&self.pending).contains(name)
    }

    fn refuse_pending(&self, name: &str) -> Result<()> {
        if self.is_pending(name) {
            return Err(Error::no_such_file(format!(
                "file \"{name}\" is pending delete"
            )));
        }
        Ok(())
    }

    /// Deletes the pending files no input is open over any more.
    fn delete_pending(&self) -> Result<()> {
        let ready: Vec<String> = lock_ignoring_poison(&self.pending)
            .iter()
            .filter(|n| !self.is_open(n))
            .cloned()
            .collect();
        for name in ready {
            self.forget(&name);
            match self.blob.delete_file(&name) {
                Ok(()) => {}
                Err(e) if e.is_no_such_file() => {}
                Err(e) => return Err(e),
            }
            lock_ignoring_poison(&self.pending).remove(&name);
        }
        Ok(())
    }

    /// Opens `name`, decoded by the codec `codec` builds from its stored
    /// length when there is one. An input already open over the same bytes
    /// is shared, without building a codec again.
    fn open_paged(
        &self,
        name: &str,
        codec: Option<&dyn Fn(u64) -> Result<Arc<dyn PageCodec>>>,
    ) -> Result<Input> {
        self.refuse_pending(name)?;
        let version = self.current_version(name)?;
        let stored_len = self.stored_len(name)?;
        let decoded = codec.is_some();
        let sharing = lock_ignoring_poison(&self.open).get(name).and_then(|all| {
            all.iter()
                .filter(|o| {
                    o.stored_len == stored_len && o.decoded == decoded && o.version == version
                })
                .find_map(|o| o.input.upgrade())
        });
        if let Some(shared) = sharing {
            return Input::slice_of(&shared, 0, shared.len());
        }
        let codec = match codec {
            Some(build) => match lock_ignoring_poison(&self.codecs).remove(name) {
                Some((len, built)) if len == stored_len => Some(built),
                _ => Some(build(stored_len)?),
            },
            None => None,
        };
        let (file, pinned) = self.cached_file(name, stored_len, codec, version.as_deref())?;
        let len = usize::try_from(file.geom.plain_len)
            .map_err(|_| Error::Corrupted(format!("{name}: too large to open")))?;
        let input = if pinned || file.geom.pages <= 1 {
            file.read_whole_counted(len)?
        } else {
            self.demand_filled(file, len)?
        };
        let shared = Arc::new(input);
        let mut open = lock_ignoring_poison(&self.open);
        // Every live input of a name stays tracked, whatever its version:
        // a delete is deferred while any of them is open.
        open.retain(|_, all| {
            all.retain(|o| o.input.strong_count() > 0);
            !all.is_empty()
        });
        open.entry(name.to_string()).or_default().push(Opened {
            stored_len,
            version,
            decoded,
            input: Arc::downgrade(&shared),
        });
        drop(open);
        Input::slice_of(&shared, 0, shared.len())
    }

    /// `name`'s pages as stored at `stored_len` bytes, decoded by `codec`
    /// when there is one, and whether the file is pinned (pinning it).
    fn cached_file(
        &self,
        name: &str,
        stored_len: u64,
        codec: Option<Arc<dyn PageCodec>>,
        version: Option<&str>,
    ) -> Result<(Arc<CachedFile<S>>, bool)> {
        let tag = if codec.is_some() { 'd' } else { 'p' };
        let prefix = self.key_prefix(name);
        // The version is length-prefixed so no version text can run into
        // the tag; a store without versions keeps the key it always had.
        let key: Arc<str> = Arc::from(match version {
            None => format!("{prefix}{stored_len}\u{0}{tag}"),
            Some(v) => format!("{prefix}{stored_len}\u{0}{tag}\u{0}v{}:{v}", v.len()),
        });
        let page_size = self.cache.config().page_size;
        let geom = match &codec {
            None => Geometry::plain(Arc::clone(&key), stored_len, page_size),
            Some(c) => Geometry::units(Arc::clone(&key), stored_len, c.layout(), page_size, name)?,
        };
        let pinned = self.cache.is_pinned_name(name);
        if pinned {
            self.cache.pin(&key);
        }
        let file = Arc::new(CachedFile {
            store: Arc::clone(&self.store),
            cache: Arc::clone(&self.cache),
            geom,
            codec,
            name: name.to_string(),
        });
        Ok((file, pinned))
    }

    /// What opening `name` reads, undecoded or through `decoder`: a pinned
    /// or two-page file whole, otherwise its first and last pages (header
    /// and footer). A decoder's codec is kept for the open that follows.
    fn prefetch_file(&self, name: &str, decoder: Option<&dyn PageDecoder>) -> Result<()> {
        if self.is_pending(name) || self.is_open(name) {
            return Ok(());
        }
        let version = self.current_version(name)?;
        let stored_len = self.stored_len(name)?;
        let codec = match decoder {
            Some(d) => {
                let codec = d.codec(name, stored_len, self)?;
                lock_ignoring_poison(&self.codecs)
                    .insert(name.to_string(), (stored_len, Arc::clone(&codec)));
                Some(codec)
            }
            None => None,
        };
        let (file, pinned) = self.cached_file(name, stored_len, codec, version.as_deref())?;
        let pages: Vec<u64> = if pinned || file.geom.pages <= 2 {
            (0..file.geom.pages).collect()
        } else {
            vec![0, file.geom.pages.saturating_sub(1)]
        };
        file.pages(&pages).map(|_| ())
    }

    /// [`Self::prefetch_file`] of every name at once, `concurrency` files at
    /// a time; the first error.
    fn prefetch_all(&self, names: &[String], decoder: Option<&dyn PageDecoder>) -> Result<()> {
        if !self.cache.has_disk() {
            return Ok(());
        }
        let threads = self.cache.config().concurrency.max(1);
        let mut first = Ok(());
        for batch in names.chunks(threads) {
            let results: Vec<Result<()>> = std::thread::scope(|s| {
                let handles: Vec<_> = batch
                    .iter()
                    .map(|name| s.spawn(move || self.prefetch_file(name, decoder)))
                    .collect();
                handles
                    .into_iter()
                    .map(|h| {
                        h.join().unwrap_or_else(|_| {
                            Err(Error::Io(std::io::Error::other("a prefetch panicked")))
                        })
                    })
                    .collect()
            });
            for r in results {
                if first.is_ok() {
                    first = r;
                }
            }
        }
        first
    }

    #[cfg(target_os = "linux")]
    fn demand_filled(&self, file: Arc<CachedFile<S>>, len: usize) -> Result<Input> {
        let page = usize::try_from(file.geom.plain_page).unwrap_or(usize::MAX);
        let pager = crate::paged_input::Pager::global()
            .filter(|p| self.cache.config().demand_paging && p.supports(page));
        match pager {
            Some(pager) => {
                let name = file.name.clone();
                let file_pages = file.geom.pages;
                let head_tail = Arc::clone(&file);
                let filler: Arc<dyn PageFiller> = file;
                // The header and footer pages first, without the window's
                // read-ahead: the open checks only those bytes, and a window
                // of megabytes around each, for every file of every segment,
                // is what a cold open of a many-segment shard waits for.
                // A failure is only a missed hint: the fault retries.
                if file_pages > 2 && self.cache.has_disk() {
                    let _ = head_tail.pages_with(&[0, file_pages.saturating_sub(1)], false);
                }
                let paged = pager.open(len, page, filler, &name)?;
                self.cache.note_demand_paged_open();
                // Lucene's readers check a file's header and footer as they
                // open it: both ends come in one batch, not two round trips.
                let bytes = crate::directory::ExternalBytes::bytes(&paged);
                let footer = bytes.len().saturating_sub(crate::codec_util::FOOTER_LENGTH);
                // Only a hint: a page it could not fetch is retried on fault.
                let _ = crate::paged_input::prefetch(&[
                    bytes.get(..1).unwrap_or(&[]),
                    &bytes[footer..],
                ]);
                Ok(Input::external_bytes(paged))
            }
            None => file.read_through(len),
        }
    }

    #[cfg(not(target_os = "linux"))]
    fn demand_filled(&self, file: Arc<CachedFile<S>>, len: usize) -> Result<Input> {
        file.read_through(len)
    }
}

/// One open file's pages: where they come from and how they decode.
struct CachedFile<S: BlobStore + 'static> {
    store: Arc<S>,
    cache: Arc<PageCache>,
    geom: Geometry,
    codec: Option<Arc<dyn PageCodec>>,
    name: String,
}

impl<S: BlobStore + 'static> CachedFile<S> {
    /// The decoded bytes of `pages`.
    fn pages(&self, pages: &[u64]) -> Result<Vec<Arc<[u8]>>> {
        self.pages_with(pages, true)
    }

    /// [`Self::pages`], with the read-ahead of the download window on or off.
    fn pages_with(&self, pages: &[u64], read_ahead: bool) -> Result<Vec<Arc<[u8]>>> {
        // The buffer is allocated inside the request's in-flight permit
        // ([`PageCacheConfig::max_inflight_bytes`]) and the store reads
        // into it, so fetching holds no heap the bound does not count.
        let fetch = |offset: u64, len: usize| {
            let mut buf = vec![0u8; len];
            let n = self.store.get_range_into(&self.name, offset, &mut buf)?;
            buf.truncate(n);
            Ok(buf)
        };
        let stored = self
            .cache
            .stored_pages_with(&self.geom, pages, &fetch, read_ahead)?;
        pages
            .iter()
            .zip(stored)
            .map(|(&page, bytes)| {
                let (a, b) = self.geom.plain_range(page);
                let n = usize::try_from(b.saturating_sub(a)).unwrap_or(usize::MAX);
                match &self.codec {
                    None if bytes.len() == n => Ok(bytes),
                    None => Err(Error::Corrupted(format!(
                        "{}: page {page} has {} bytes, expected {n}",
                        self.name,
                        bytes.len()
                    ))),
                    Some(codec) => {
                        let mut out = vec![0u8; n];
                        let first = page.saturating_mul(self.geom.units_per_page);
                        codec.decode(first, &bytes, &mut out)?;
                        Ok(Arc::from(out))
                    }
                }
            })
            .collect()
    }

    /// The file as an input where it cannot be demand-filled: streamed into
    /// a mapped spill file when the cache is configured for it and the file
    /// is large enough ([`PageCacheConfig::spill_min_bytes`]), else read
    /// whole into the heap.
    fn read_through(&self, len: usize) -> Result<Input> {
        let config: &PageCacheConfig = self.cache.config();
        #[cfg(unix)]
        if let (Some(min), Some(dir), None) = (
            config.spill_min_bytes,
            config.disk_dir.as_deref(),
            &self.codec,
        ) {
            if len as u64 >= min {
                return self.spill(dir, len);
            }
        }
        let _ = (config, len);
        self.read_whole_counted(len)
    }

    /// The file read whole onto the heap, its `len` bytes counted against
    /// [`PageCacheConfig::heap_budget`] until the input drops (when the
    /// budget is bounded). The budget
    /// is claimed before any byte is fetched.
    fn read_whole_counted(&self, len: usize) -> Result<Input> {
        // Unbounded (the default): the plain heap input, exactly as before,
        // with no counting and no indirection on every read of the file.
        if self.cache.config().heap_budget == u64::MAX {
            return Ok(Input::Owned(self.read_whole()?));
        }
        let reservation = self.cache.reserve_heap(len as u64)?;
        let bytes = self.read_whole()?;
        self.cache.note_heap_open();
        Ok(Input::external_bytes(HeapBytes {
            bytes,
            _reservation: reservation,
        }))
    }

    /// Streams the file, a bounded chunk of pages at a time, into an unlinked
    /// file under `dir` and maps it read-only: heap use is one chunk, not
    /// the file, and the bytes live in kernel page cache. The space returns
    /// when the input drops.
    #[cfg(unix)]
    fn spill(&self, dir: &std::path::Path, len: usize) -> Result<Input> {
        use std::os::unix::fs::FileExt;
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let reservation = self.cache.reserve_spill(len as u64)?;
        // A stale file of a crashed process with this pid and counter is
        // skipped, not an error.
        let (path, file) = loop {
            let path = dir.join(format!(
                "spill-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .open(&path)
            {
                Ok(file) => break (path, file),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(e) => return Err(e.into()),
            }
        };
        // Unlinked at once: a crash leaves nothing behind, and the space is
        // the mapping's until it drops.
        std::fs::remove_file(&path)?;
        file.set_len(len as u64)?;
        let config = self.cache.config();
        let chunk_bytes = config
            .max_request
            .saturating_mul(config.concurrency.max(1))
            .min(crate::page_cache::SPILL_CHUNK_BYTES);
        let per_chunk = (chunk_bytes as u64)
            .checked_div(self.geom.plain_page)
            .unwrap_or(1)
            .max(1);
        let mut page = 0u64;
        while page < self.geom.pages {
            let end = page.saturating_add(per_chunk).min(self.geom.pages);
            let ids: Vec<u64> = (page..end).collect();
            for (&id, bytes) in ids.iter().zip(self.pages(&ids)?) {
                file.write_all_at(&bytes, id.saturating_mul(self.geom.plain_page))?;
            }
            page = end;
        }
        // SAFETY: the file was created by this call, is unlinked, and is
        // never written after this mapping is made, so nothing can truncate
        // or change it underneath the mapping.
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        self.cache.note_spill_open();
        Ok(Input::external_bytes(SpillBytes {
            mmap,
            _reservation: reservation,
        }))
    }

    /// Every decoded byte.
    fn read_whole(&self) -> Result<Vec<u8>> {
        let all: Vec<u64> = (0..self.geom.pages).collect();
        let pages = self.pages(&all)?;
        let mut out = Vec::with_capacity(usize::try_from(self.geom.plain_len).unwrap_or(0));
        for p in pages {
            out.extend_from_slice(&p);
        }
        if out.len() as u64 != self.geom.plain_len {
            return Err(Error::Corrupted(format!(
                "{}: decoded {} bytes, expected {}",
                self.name,
                out.len(),
                self.geom.plain_len
            )));
        }
        Ok(out)
    }
}

/// A file held whole on the heap, holding its share of the heap budget.
struct HeapBytes {
    bytes: Vec<u8>,
    _reservation: SpillReservation,
}

impl crate::directory::ExternalBytes for HeapBytes {
    fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// A spill file's read-only mapping, holding its share of the spill budget.
#[cfg(unix)]
struct SpillBytes {
    mmap: memmap2::Mmap,
    _reservation: SpillReservation,
}

#[cfg(unix)]
impl crate::directory::ExternalBytes for SpillBytes {
    fn bytes(&self) -> &[u8] {
        &self.mmap
    }
}

#[cfg(target_os = "linux")]
impl<S: BlobStore + 'static> PageFiller for CachedFile<S> {
    fn fill(&self, pages: &[u64]) -> Result<Vec<Arc<[u8]>>> {
        self.pages(pages)
    }
}

impl<S: BlobStore + 'static> Directory for CachedBlobDirectory<S> {
    fn list_all(&self) -> Result<Vec<String>> {
        self.delete_pending()?;
        let forgets = self.forgets.load(Ordering::Acquire);
        let mut names = match self.store.list_with_sizes()? {
            Some(listed) => {
                self.prime_for_new_commit(&listed);
                let mut sizes = lock_ignoring_poison(&self.sizes);
                let current = self.forgets.load(Ordering::Acquire) == forgets;
                let mut names = Vec::with_capacity(listed.len());
                for (name, len) in listed {
                    if current {
                        sizes.entry(name.clone()).or_insert(len);
                    }
                    names.push(name);
                }
                names.sort_unstable();
                names.dedup();
                names
            }
            None => self.blob.list_all()?,
        };
        let pending = lock_ignoring_poison(&self.pending);
        names.retain(|n| !pending.contains(n));
        Ok(names)
    }

    fn open(&self, name: &str) -> Result<Input> {
        self.open_paged(name, None)
    }

    fn open_decoded(&self, name: &str, decoder: &dyn PageDecoder) -> Option<Result<Input>> {
        let build = |len: u64| decoder.codec(name, len, self);
        Some(self.open_paged(name, Some(&build)))
    }

    /// Fetches into the disk tier, every file at once, what opening each of
    /// `names` undecoded will read. Without a disk tier nothing would keep
    /// the pages, so nothing is fetched.
    fn prefetch_files(&self, names: &[String]) -> Result<()> {
        self.prefetch_all(names, None)
    }

    /// [`Directory::prefetch_files`] for files opened through `decoder`:
    /// each file's header (read by the decoder) and then its first and last
    /// decoded pages, every file at once.
    fn prefetch_files_decoded(&self, names: &[String], decoder: &dyn PageDecoder) -> Result<()> {
        self.prefetch_all(names, Some(decoder))
    }

    fn read_at(&self, name: &str, offset: u64, len: usize) -> Result<Vec<u8>> {
        self.refuse_pending(name)?;
        self.store.get_range(name, offset, len)
    }

    fn file_length(&self, name: &str) -> Result<u64> {
        self.refuse_pending(name)?;
        self.stored_len(name)
    }

    fn create_output(&self, name: &str) -> Result<FsIndexOutput> {
        self.delete_pending()?;
        if self.is_pending(name) {
            // Its old bytes are still being read from the store.
            return Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                format!("file \"{name}\" is pending delete and still open"),
            )));
        }
        self.forget(name);
        self.blob.create_output(name)
    }

    fn create_temp_output(&self, prefix: &str, suffix: &str) -> Result<FsIndexOutput> {
        self.blob.create_temp_output(prefix, suffix)
    }

    fn sync(&self, names: &[String]) -> Result<()> {
        self.blob.sync(names)
    }

    fn rename(&self, source: &str, dest: &str) -> Result<()> {
        self.refuse_pending(source)?;
        self.delete_pending()?;
        self.forget(dest);
        self.blob.store().copy(source, dest)?;
        self.blob.bundle_move(source, dest);
        self.delete_file(source)?;
        self.blob.bundle_publish(dest);
        Ok(())
    }

    fn delete_file(&self, name: &str) -> Result<()> {
        self.refuse_pending(name)?;
        self.delete_pending()?;
        if self.is_open(name) {
            if !self.store.exists(name)? {
                return Err(Error::no_such_file(name));
            }
            lock_ignoring_poison(&self.pending).insert(name.to_string());
            return Ok(());
        }
        self.forget(name);
        self.blob.delete_file(name)
    }

    fn sync_meta_data(&self) -> Result<()> {
        self.blob.sync_meta_data()
    }

    fn obtain_lock(&self, name: &str) -> Result<Box<dyn Lock>> {
        self.blob.obtain_lock(name)
    }

    fn pending_deletions(&self) -> Result<BTreeSet<String>> {
        self.delete_pending()?;
        Ok(lock_ignoring_poison(&self.pending).clone())
    }
}

impl<S: BlobStore + 'static> Drop for CachedBlobDirectory<S> {
    /// Deletes the pending files whose inputs are closed by now, as
    /// `FSDirectory.close` retries its pending deletes.
    fn drop(&mut self) {
        let _ = self.delete_pending();
    }
}

#[cfg(test)]
mod tests;
