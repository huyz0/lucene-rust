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
//!   ([`BlobStore::get_range`]) that merge adjacent pages.
//! - Files read on every query or whole at open (metadata, the terms
//!   index, norms, the stored-fields and points indexes, live docs: see
//!   [`crate::page_cache::DEFAULT_PINNED_EXTENSIONS`]) and files of one page
//!   are read whole into memory, their pages pinned on disk.
//! - An encrypting wrapper above ([`Directory::open_decoded`]) gets pages of
//!   whole authenticated units: the disk tier keeps ciphertext and only the
//!   mapping in memory holds plaintext, decrypted once per page brought in.
//! - Without `userfaultfd` (not Linux, or blocked), files are read whole,
//!   still page by page through the cache.
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
use crate::page_cache::{Geometry, PageCache, PageCodec, PageDecoder};
#[cfg(target_os = "linux")]
use crate::paged_input::PageFiller;

/// A [`Directory`] over a [`BlobStore`], read through a shared
/// [`PageCache`]. See the module docs.
pub struct CachedBlobDirectory<S: BlobStore + 'static> {
    store: Arc<S>,
    blob: BlobDirectory<S>,
    cache: Arc<PageCache>,
    namespace: Arc<str>,
    open: Mutex<HashMap<String, Opened>>,
    /// Files deleted while open: removed from the store once no input
    /// over them is left, since an open file's pages are fetched from it.
    pending: Mutex<BTreeSet<String>>,
    /// File lengths, from the store's listing or a length request: files
    /// are written once, so a length holds until the name is deleted or
    /// written again.
    sizes: Mutex<HashMap<String, u64>>,
    /// Codecs built by a prefetch, with the stored length they were built
    /// for, taken by the open that follows so a decoder (an encrypted
    /// file's header and data key) runs once per file.
    codecs: Mutex<HashMap<String, Built>>,
    /// Bumped by every [`Self::forget`]: a listing that started before one
    /// does not put back the lengths it read.
    forgets: AtomicU64,
}

/// A codec and the stored length it was built for.
type Built = (u64, Arc<dyn PageCodec>);

/// A file open now: shared by later opens of the same bytes.
struct Opened {
    stored_len: u64,
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
            codecs: Mutex::new(HashMap::new()),
            forgets: AtomicU64::new(0),
        }
    }

    /// The store underneath.
    pub fn store(&self) -> &S {
        &self.store
    }

    /// The page cache.
    pub fn cache(&self) -> &Arc<PageCache> {
        &self.cache
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

    /// Whether an input over `name` is open now.
    fn is_open(&self, name: &str) -> bool {
        lock_ignoring_poison(&self.open)
            .get(name)
            .is_some_and(|o| o.input.strong_count() > 0)
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
        let stored_len = self.stored_len(name)?;
        let decoded = codec.is_some();
        if let Some(o) = lock_ignoring_poison(&self.open).get(name) {
            if o.stored_len == stored_len && o.decoded == decoded {
                if let Some(shared) = o.input.upgrade() {
                    return Input::slice_of(&shared, 0, shared.len());
                }
            }
        }
        let codec = match codec {
            Some(build) => match lock_ignoring_poison(&self.codecs).remove(name) {
                Some((len, built)) if len == stored_len => Some(built),
                _ => Some(build(stored_len)?),
            },
            None => None,
        };
        let (file, pinned) = self.cached_file(name, stored_len, codec)?;
        let len = usize::try_from(file.geom.plain_len)
            .map_err(|_| Error::Corrupted(format!("{name}: too large to open")))?;
        let input = if pinned || file.geom.pages <= 1 {
            Input::Owned(file.read_whole()?)
        } else {
            self.demand_filled(file, len)?
        };
        let shared = Arc::new(input);
        let mut open = lock_ignoring_poison(&self.open);
        open.retain(|_, o| o.input.strong_count() > 0);
        open.insert(
            name.to_string(),
            Opened {
                stored_len,
                decoded,
                input: Arc::downgrade(&shared),
            },
        );
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
    ) -> Result<(Arc<CachedFile<S>>, bool)> {
        let tag = if codec.is_some() { 'd' } else { 'p' };
        let key: Arc<str> = Arc::from(format!("{}{stored_len}\u{0}{tag}", self.key_prefix(name)));
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
        let (file, pinned) = self.cached_file(name, stored_len, codec)?;
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
        match crate::paged_input::Pager::global().filter(|p| p.supports(page)) {
            Some(pager) => {
                let name = file.name.clone();
                let filler: Arc<dyn PageFiller> = file;
                let paged = pager.open(len, page, filler, &name)?;
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
            None => Ok(Input::Owned(file.read_whole()?)),
        }
    }

    #[cfg(not(target_os = "linux"))]
    fn demand_filled(&self, file: Arc<CachedFile<S>>, len: usize) -> Result<Input> {
        let _ = len;
        Ok(Input::Owned(file.read_whole()?))
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
        let fetch = |offset: u64, len: usize| self.store.get_range(&self.name, offset, len);
        let stored = self.cache.stored_pages(&self.geom, pages, &fetch)?;
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
        self.delete_file(source)
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
mod tests {
    // Test fixtures' own arithmetic -- see `docs/arithmetic-gate.md`'s "Test code".
    #![allow(clippy::arithmetic_side_effects)]
    use super::*;
    use crate::page_cache::{PageCacheConfig, UnitLayout};
    use crate::{DataOutput, InMemoryBlobStore};
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A store that counts ranged reads.
    #[derive(Clone, Default)]
    struct Counting {
        inner: InMemoryBlobStore,
        ranges: Arc<AtomicUsize>,
        wholes: Arc<AtomicUsize>,
        sizes: Arc<AtomicUsize>,
    }

    impl BlobStore for Counting {
        fn list(&self) -> Result<Vec<String>> {
            self.inner.list()
        }
        fn get(&self, name: &str) -> Result<Input> {
            self.wholes.fetch_add(1, Ordering::Relaxed);
            self.inner.get(name)
        }
        fn get_range(&self, name: &str, offset: u64, len: usize) -> Result<Vec<u8>> {
            self.ranges.fetch_add(1, Ordering::Relaxed);
            self.inner.get_range(name, offset, len)
        }
        fn size(&self, name: &str) -> Result<u64> {
            self.sizes.fetch_add(1, Ordering::Relaxed);
            self.inner.size(name)
        }
        fn list_with_sizes(&self) -> Result<Option<Vec<(String, u64)>>> {
            self.inner.list_with_sizes()
        }
        fn put(&self, name: &str, bytes: Vec<u8>) -> Result<()> {
            self.inner.put(name, bytes)
        }
        fn delete(&self, name: &str) -> Result<()> {
            self.inner.delete(name)
        }
    }

    fn data(len: usize, seed: usize) -> Vec<u8> {
        (0..len).map(|i| ((i * 131 + seed) % 251) as u8).collect()
    }

    fn put(d: &dyn Directory, name: &str, bytes: &[u8]) {
        let mut out = d.create_output(name).unwrap();
        out.write_bytes(bytes);
        out.close().unwrap();
    }

    fn dir(
        tag: &str,
        page: usize,
    ) -> (
        CachedBlobDirectory<Counting>,
        Counting,
        lucene_util::test_support::TempDir,
    ) {
        let tmp = lucene_util::test_support::TempDir::new(tag);
        let cache = Arc::new(
            PageCache::new(PageCacheConfig {
                page_size: page,
                disk_dir: Some(tmp.path().join("cache")),
                ..PageCacheConfig::default()
            })
            .unwrap(),
        );
        let store = Counting::default();
        (
            CachedBlobDirectory::new(store.clone(), cache, "idx"),
            store,
            tmp,
        )
    }

    #[test]
    fn large_files_are_fetched_by_page_and_reads_see_the_written_bytes() {
        let (d, store, _tmp) = dir("cached-blob-pages", 64 * 1024);
        let big = data(1_000_000, 3);
        put(&d, "_0.doc", &big);
        put(&d, "_0.tip", &data(70_000, 4));
        put(&d, "small.bin", &data(100, 5));
        assert_eq!(d.list_all().unwrap(), ["_0.doc", "_0.tip", "small.bin"]);
        assert_eq!(d.file_length("_0.doc").unwrap(), 1_000_000);
        let input = d.open("_0.doc").unwrap();
        assert_eq!(input.len(), big.len());
        input.will_need(500_000, 10).unwrap();
        assert_eq!(input[500_005], big[500_005]);
        assert_eq!(input[999_999], big[999_999]);
        assert!(input[..] == big[..]);
        // A second open shares the open file.
        let again = d.open("_0.doc").unwrap();
        assert!(again[..] == big[..]);
        assert_eq!(d.open("_0.tip").unwrap()[..], data(70_000, 4)[..]);
        assert_eq!(d.open("small.bin").unwrap()[..], data(100, 5)[..]);
        assert_eq!(d.read_at("small.bin", 90, 50).unwrap(), data(100, 5)[90..]);
        assert_eq!(
            store.wholes.load(Ordering::Relaxed),
            0,
            "no whole-object reads"
        );
        // Dropped and reopened: every page comes from the disk tier.
        drop((input, again));
        let before = store.ranges.load(Ordering::Relaxed);
        assert!(d.open("_0.doc").unwrap()[..] == big[..]);
        assert_eq!(store.ranges.load(Ordering::Relaxed), before);
        assert!(d.cache().stats().disk_hits > 0);
        assert!(format!("{d:?}").contains("idx"));
        assert_eq!(d.store().list().unwrap().len(), 3);
    }

    #[test]
    fn prefetched_files_open_without_further_requests() {
        let (d, store, _tmp) = dir("cached-blob-prefetch", 64 * 1024);
        let big = data(1_000_000, 3);
        put(&d, "_0.doc", &big);
        put(&d, "_0.tip", &data(70_000, 4));
        put(&d, "_0.si", &data(100, 5));
        let names = ["_0.doc", "_0.tip", "_0.si"].map(String::from);
        d.prefetch_files(&names).unwrap();
        let fetched = store.ranges.load(Ordering::Relaxed);
        assert!(fetched > 0);
        // Header and footer of the large file, the pinned and the small
        // files whole: opening them and reading both ends asks for nothing.
        let doc = d.open("_0.doc").unwrap();
        assert_eq!(doc[..16], big[..16]);
        assert_eq!(doc[big.len() - 16..], big[big.len() - 16..]);
        assert_eq!(d.open("_0.tip").unwrap()[..], data(70_000, 4)[..]);
        assert_eq!(d.open("_0.si").unwrap()[..], data(100, 5)[..]);
        assert_eq!(store.ranges.load(Ordering::Relaxed), fetched);
        // A missing file is an error; an open one is skipped.
        assert!(d.prefetch_files(&["nope".to_string()]).is_err());
        d.prefetch_files(&names).unwrap();
        assert_eq!(store.ranges.load(Ordering::Relaxed), fetched);
    }

    #[test]
    fn lengths_come_from_the_listing_and_a_prefetched_codec_serves_the_open() {
        let (d, store, _tmp) = dir("cached-blob-lengths", 16 * 1024);
        let plain = data(150_000, 9);
        put(&d, "_0.fdt", &encode(&plain));
        put(&d, "_0.si", &data(100, 5));
        assert_eq!(d.list_all().unwrap(), ["_0.fdt", "_0.si"]);
        let sizes = store.sizes.load(Ordering::Relaxed);
        d.prefetch_files_decoded(&["_0.fdt".to_string()], &XorDecoder)
            .unwrap();
        let fetched = store.ranges.load(Ordering::Relaxed);
        // The codec the prefetch built (its header read once) and both
        // decoded ends: opening the file and reading them asks the store
        // for nothing more.
        let input = d.open_decoded("_0.fdt", &XorDecoder).unwrap().unwrap();
        assert!(input[..16] == plain[..16]);
        assert!(input[plain.len() - 16..] == plain[plain.len() - 16..]);
        assert_eq!(d.file_length("_0.si").unwrap(), 100);
        assert_eq!(store.ranges.load(Ordering::Relaxed), fetched);
        assert_eq!(store.sizes.load(Ordering::Relaxed), sizes, "lengths listed");
        // A name written again is not served its old length.
        d.delete_file("_0.si").unwrap();
        put(&d, "_0.si", &data(300, 6));
        assert_eq!(d.file_length("_0.si").unwrap(), 300);
        assert_eq!(d.read_at("_0.si", 0, 400).unwrap(), data(300, 6));
        assert_eq!(d.read_at("_0.si", 290, 400).unwrap(), data(300, 6)[290..]);
    }

    #[test]
    fn writes_renames_and_deletes_reach_the_store_and_drop_cached_pages() {
        let (d, _store, _tmp) = dir("cached-blob-writes", 4096);
        put(&d, "pending_segments_1", &data(10_000, 1));
        assert_eq!(
            d.open("pending_segments_1").unwrap()[..],
            data(10_000, 1)[..]
        );
        d.rename("pending_segments_1", "segments_1").unwrap();
        assert!(d.open("pending_segments_1").is_err());
        assert_eq!(d.open("segments_1").unwrap()[..], data(10_000, 1)[..]);
        d.sync(&["segments_1".to_string()]).unwrap();
        d.sync_meta_data().unwrap();
        d.delete_file("segments_1").unwrap();
        assert!(d.open("segments_1").unwrap_err().is_no_such_file());
        assert_eq!(d.cache().disk_bytes(), 0);
        let mut tmp = d.create_temp_output("_0", "x").unwrap();
        tmp.write_bytes(b"t");
        let name = crate::IndexOutput::name(&tmp).to_string();
        tmp.close().unwrap();
        assert_eq!(&d.open(&name).unwrap()[..], b"t");
        assert!(d.pending_deletions().unwrap().is_empty());
        assert!(d.open("missing").unwrap_err().is_no_such_file());
        assert!(d.delete_file("missing").unwrap_err().is_no_such_file());
        let lock = d.obtain_lock("write.lock").unwrap();
        assert!(d.obtain_lock("write.lock").is_err());
        drop(lock);
    }

    #[test]
    fn a_file_deleted_while_open_stays_readable_until_its_inputs_close() {
        let (d, store, _tmp) = dir("cached-blob-pending", 4096);
        let bytes = data(100_000, 2);
        put(&d, "_0.doc", &bytes);
        let input = d.open("_0.doc").unwrap();
        assert_eq!(input[10], bytes[10]);
        d.delete_file("_0.doc").unwrap();
        assert_eq!(d.list_all().unwrap(), Vec::<String>::new());
        assert!(d.open("_0.doc").unwrap_err().is_no_such_file());
        assert!(d.file_length("_0.doc").is_err());
        assert!(d.read_at("_0.doc", 0, 1).is_err());
        assert!(d.delete_file("_0.doc").is_err());
        assert!(d.rename("_0.doc", "_1.doc").is_err());
        assert!(d.create_output("_0.doc").is_err());
        assert_eq!(d.pending_deletions().unwrap().len(), 1);
        // Still in the store, and every page still readable.
        assert_eq!(store.inner.list().unwrap(), ["_0.doc"]);
        assert!(input[..] == bytes[..]);
        drop(input);
        assert!(d.pending_deletions().unwrap().is_empty());
        assert!(store.inner.list().unwrap().is_empty());
        // Dropping the directory deletes what is pending by then.
        put(&d, "_2.doc", &bytes);
        let input = d.open("_2.doc").unwrap();
        d.delete_file("_2.doc").unwrap();
        drop(input);
        drop(d);
        assert!(store.inner.list().unwrap().is_empty());
    }

    /// A toy codec: units of 4200 stored bytes, the first 4096 of which are
    /// the plaintext XORed with the unit index, after a 7-byte header.
    struct Xor {
        plain_len: u64,
    }

    const UNIT: u64 = 4200;
    const PLAIN: u64 = 4096;

    impl PageCodec for Xor {
        fn layout(&self) -> UnitLayout {
            UnitLayout {
                origin: 7,
                stored_unit: UNIT,
                plain_unit: PLAIN,
                plain_len: self.plain_len,
            }
        }
        fn decode(&self, first: u64, stored: &[u8], out: &mut [u8]) -> Result<()> {
            for (k, (s, o)) in stored
                .chunks(UNIT as usize)
                .zip(out.chunks_mut(PLAIN as usize))
                .enumerate()
            {
                let key = (first + k as u64) as u8;
                if s.len() < o.len() {
                    return Err(Error::Corrupted("short unit".into()));
                }
                for (b, x) in o.iter_mut().zip(s) {
                    *b = x ^ key;
                }
            }
            Ok(())
        }
    }

    struct XorDecoder;

    impl PageDecoder for XorDecoder {
        fn codec(
            &self,
            name: &str,
            stored_len: u64,
            dir: &dyn Directory,
        ) -> Result<Arc<dyn PageCodec>> {
            if dir.read_at(name, 0, 7)? != b"header!" {
                return Err(Error::Corrupted("no header".into()));
            }
            let body = stored_len - 7;
            let units = body.div_ceil(UNIT);
            let plain_len = body - units * (UNIT - PLAIN);
            Ok(Arc::new(Xor { plain_len }))
        }
    }

    fn encode(plain: &[u8]) -> Vec<u8> {
        let mut out = b"header!".to_vec();
        for (i, c) in plain.chunks(PLAIN as usize).enumerate() {
            out.extend(c.iter().map(|b| b ^ i as u8));
            out.extend(std::iter::repeat_n(0xEE, (UNIT - PLAIN) as usize));
        }
        out
    }

    #[test]
    fn decoded_files_get_pages_of_whole_units() {
        // 16 KiB pages hold four units: 16 KiB of plaintext per page.
        let (d, store, _tmp) = dir("cached-blob-decoded", 16 * 1024);
        let plain = data(150_000, 9);
        put(&d, "_0.fdt", &encode(&plain));
        let input = d.open_decoded("_0.fdt", &XorDecoder).unwrap().unwrap();
        assert_eq!(input.len(), plain.len());
        assert!(input[..] == plain[..]);
        // The cache holds stored (encoded) bytes, never decoded ones.
        assert_eq!(d.cache().disk_bytes(), d.file_length("_0.fdt").unwrap() - 7);
        assert!(store.ranges.load(Ordering::Relaxed) >= 2);
        put(&d, "bad", b"nothead");
        assert!(d.open_decoded("bad", &XorDecoder).unwrap().is_err());
        assert!(d.open_decoded("missing", &XorDecoder).unwrap().is_err());
        // A plain open of the same file gets the stored bytes.
        assert_eq!(&d.open("_0.fdt").unwrap()[..7], b"header!");
    }
}
