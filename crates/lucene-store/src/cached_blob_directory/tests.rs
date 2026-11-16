#![cfg(test)]
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

/// A store whose ranged reads only go into the caller's buffer.
#[derive(Clone, Default)]
struct IntoOnly {
    inner: InMemoryBlobStore,
    largest: Arc<AtomicUsize>,
}

impl BlobStore for IntoOnly {
    fn list(&self) -> Result<Vec<String>> {
        self.inner.list()
    }
    fn get(&self, name: &str) -> Result<Input> {
        self.inner.get(name)
    }
    fn get_range(&self, _: &str, _: u64, _: usize) -> Result<Vec<u8>> {
        Err(Error::Io(std::io::Error::other("an owned ranged read")))
    }
    fn get_range_into(&self, name: &str, offset: u64, out: &mut [u8]) -> Result<usize> {
        self.largest.fetch_max(out.len(), Ordering::Relaxed);
        let bytes = self.inner.get_range(name, offset, out.len())?;
        out[..bytes.len()].copy_from_slice(&bytes);
        Ok(bytes.len())
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

#[test]
fn pages_are_read_into_buffers_the_cache_holds() {
    let tmp = lucene_util::test_support::TempDir::new("cached-blob-into");
    let config = PageCacheConfig {
        page_size: 64 * 1024,
        max_request: 256 * 1024,
        max_inflight_bytes: 256 * 1024,
        disk_dir: Some(tmp.path().join("cache")),
        ..PageCacheConfig::default()
    };
    for demand_paging in [true, false] {
        let cache = Arc::new(
            PageCache::new(PageCacheConfig {
                demand_paging,
                ..config.clone()
            })
            .unwrap(),
        );
        let store = IntoOnly::default();
        let d = CachedBlobDirectory::new(store.clone(), cache, "idx");
        let big = data(1_000_000, 7);
        put(&d, "_0.doc", &big);
        let input = d.open("_0.doc").unwrap();
        assert!(input[..] == big[..]);
        let largest = store.largest.load(Ordering::Relaxed);
        assert!(largest > 0 && largest <= 256 * 1024, "{largest}");
    }
}

#[test]
fn the_default_ranged_read_into_a_buffer_stops_at_the_object_end() {
    let store = InMemoryBlobStore::new();
    store.put("f", data(100, 1)).unwrap();
    let mut out = [0u8; 50];
    assert_eq!(store.get_range_into("f", 90, &mut out).unwrap(), 10);
    assert_eq!(out[..10], data(100, 1)[90..]);
    assert_eq!(store.get_range_into("f", 0, &mut out).unwrap(), 50);
    assert!(store.get_range_into("missing", 0, &mut out).is_err());
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

/// A store whose objects each have a settable version, over a shared
/// in-memory map so two directories can see different contents.
#[derive(Clone, Default)]
struct Versioned {
    inner: InMemoryBlobStore,
    versions: Arc<Mutex<HashMap<String, String>>>,
}

impl Versioned {
    fn set(&self, name: &str, bytes: Vec<u8>, version: &str) {
        self.inner.put(name, bytes).unwrap();
        lock_ignoring_poison(&self.versions).insert(name.to_string(), version.to_string());
    }
}

impl BlobStore for Versioned {
    fn list(&self) -> Result<Vec<String>> {
        self.inner.list()
    }
    fn get(&self, name: &str) -> Result<Input> {
        self.inner.get(name)
    }
    fn get_range(&self, name: &str, offset: u64, len: usize) -> Result<Vec<u8>> {
        self.inner.get_range(name, offset, len)
    }
    fn size(&self, name: &str) -> Result<u64> {
        self.inner.size(name)
    }
    fn version(&self, name: &str) -> Result<Option<String>> {
        Ok(lock_ignoring_poison(&self.versions).get(name).cloned())
    }
    fn put(&self, name: &str, bytes: Vec<u8>) -> Result<()> {
        self.inner.put(name, bytes)
    }
    fn delete(&self, name: &str) -> Result<()> {
        self.inner.delete(name)
    }
}

fn cache_in(tmp: &lucene_util::test_support::TempDir, config: PageCacheConfig) -> Arc<PageCache> {
    Arc::new(
        PageCache::new(PageCacheConfig {
            disk_dir: Some(tmp.path().join("cache")),
            ..config
        })
        .unwrap(),
    )
}

#[test]
fn a_store_version_keeps_same_length_contents_of_one_name_apart() {
    let tmp = lucene_util::test_support::TempDir::new("cached-blob-version");
    let cache = cache_in(
        &tmp,
        PageCacheConfig {
            page_size: 64 * 1024,
            ..PageCacheConfig::default()
        },
    );
    let (old, new) = (data(300_000, 1), data(300_000, 2));
    // Two directories share the cache and a namespace, over stores whose
    // `_0.doc` differs in content only (the same name and length).
    let (a, b) = (Versioned::default(), Versioned::default());
    a.set("_0.doc", old.clone(), "gen-1");
    b.set("_0.doc", new.clone(), "gen-2");
    let da = CachedBlobDirectory::new(a.clone(), Arc::clone(&cache), "idx");
    let db = CachedBlobDirectory::new(b, Arc::clone(&cache), "idx");
    let held = da.open("_0.doc").unwrap();
    assert!(held[..] == old[..]);
    assert!(
        db.open("_0.doc").unwrap()[..] == new[..],
        "no pages of gen-1"
    );
    // The object replaced under the same name and length, behind the
    // directory's back: the new version is seen on the next open.
    a.set("_0.doc", new.clone(), "gen-2");
    let fresh = da.open("_0.doc").unwrap();
    assert!(fresh[..] == new[..], "not shared with the gen-1 input");
    assert!(held[..] == old[..], "the open gen-1 input is intact");
    // Both inputs stay tracked: a delete waits for the older one too.
    drop(fresh);
    da.delete_file("_0.doc").unwrap();
    assert!(da.pending_deletions().unwrap().contains("_0.doc"));
    assert!(held[..] == old[..]);
    drop(held);
    assert!(da.pending_deletions().unwrap().is_empty());
    // Unversioned stores keep the key they always had (one namespace,
    // name and length is one file), which is the poisoning a version
    // exists to prevent.
    let (c, d) = (Versioned::default(), Versioned::default());
    c.inner.put("_1.doc", old.clone()).unwrap();
    d.inner.put("_1.doc", new.clone()).unwrap();
    let dc = CachedBlobDirectory::new(c, Arc::clone(&cache), "plain");
    let dd = CachedBlobDirectory::new(d, Arc::clone(&cache), "plain");
    assert!(dc.open("_1.doc").unwrap()[..] == old[..]);
    assert!(dd.open("_1.doc").unwrap()[..] == old[..], "shared key");
}

#[test]
fn a_version_change_drops_the_cached_length_and_a_text_cannot_alias_the_key() {
    let tmp = lucene_util::test_support::TempDir::new("cached-blob-version-len");
    let cache = cache_in(&tmp, PageCacheConfig::default());
    let store = Versioned::default();
    store.set("f.bin", data(100, 1), "v1");
    let dir = CachedBlobDirectory::new(store.clone(), cache, "idx");
    assert_eq!(dir.open("f.bin").unwrap().len(), 100);
    store.set("f.bin", data(150, 1), "v2");
    assert_eq!(dir.open("f.bin").unwrap().len(), 150);
    assert_eq!(dir.file_length("f.bin").unwrap(), 150);
    let key = |v: Option<&str>| {
        dir.cached_file("f.bin", 5, None, v)
            .unwrap()
            .0
            .geom
            .key
            .clone()
    };
    assert_ne!(key(None), key(Some("")));
    assert_ne!(key(Some("a\u{0}pv1:b")), key(Some("a")));
    assert_eq!(key(Some("x")), key(Some("x")));
}

fn spill_dir(
    tag: &str,
    config: PageCacheConfig,
) -> (
    CachedBlobDirectory<Counting>,
    Counting,
    lucene_util::test_support::TempDir,
) {
    let tmp = lucene_util::test_support::TempDir::new(tag);
    let cache = cache_in(&tmp, config);
    let store = Counting::default();
    (
        CachedBlobDirectory::new(store.clone(), cache, "idx"),
        store,
        tmp,
    )
}

#[cfg(unix)]
#[test]
fn without_demand_paging_large_files_are_mapped_from_a_spill_file_not_held_whole() {
    let config = PageCacheConfig {
        page_size: 64 * 1024,
        demand_paging: false,
        spill_min_bytes: Some(200_000),
        ..PageCacheConfig::default()
    };
    let (d, store, tmp) = spill_dir("cached-blob-spill", config);
    let big = data(1_000_003, 3);
    put(&d, "_0.doc", &big);
    put(&d, "_0.nvd", &data(400_000, 4)); // pinned: stays in the heap
    put(&d, "_0.si", &data(100_000, 5)); // pinned, under it
    put(&d, "_1.doc", &data(150_000, 6)); // under it
    let doc = d.open("_0.doc").unwrap();
    assert!(doc[..] == big[..]);
    let stats = d.cache().stats();
    assert_eq!(
        (stats.spill_opens, stats.spill_open_bytes),
        (1, big.len() as u64)
    );
    assert_eq!(d.open("_0.nvd").unwrap()[..], data(400_000, 4)[..]);
    assert_eq!(
        d.cache().stats().spill_opens,
        1,
        "pinned files stay in the heap"
    );
    put(&d, "_2.doc", &data(300_000, 8));
    assert_eq!(d.open("_2.doc").unwrap()[..], data(300_000, 8)[..]);
    assert_eq!(d.cache().stats().spill_opens, 2);
    assert_eq!(d.open("_0.si").unwrap()[..], data(100_000, 5)[..]);
    assert_eq!(d.open("_1.doc").unwrap()[..], data(150_000, 6)[..]);
    assert_eq!(
        d.cache().stats().spill_opens,
        2,
        "small files stay in the heap"
    );
    // A second open shares the mapping; dropping every input returns
    // the budget, and nothing is left on disk but the tier's slabs.
    let again = d.open("_0.doc").unwrap();
    assert_eq!(d.cache().stats().spill_opens, 2);
    drop((doc, again));
    assert_eq!(d.cache().stats().spill_open_bytes, 0);
    let leftovers = std::fs::read_dir(tmp.path().join("cache"))
        .unwrap()
        .filter(|e| {
            e.as_ref()
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with("spill-")
        })
        .count();
    assert_eq!(leftovers, 0);
    assert_eq!(store.wholes.load(Ordering::Relaxed), 0);
}

#[cfg(unix)]
#[test]
fn the_spill_budget_fails_an_open_that_would_pass_it_and_heap_is_used_only_when_configured() {
    let config = PageCacheConfig {
        page_size: 64 * 1024,
        demand_paging: false,
        spill_min_bytes: Some(1),
        spill_budget: 600_000,
        ..PageCacheConfig::default()
    };
    let (d, _store, _tmp) = spill_dir("cached-blob-spill-budget", config);
    put(&d, "_0.doc", &data(500_000, 1));
    put(&d, "_1.doc", &data(500_000, 2));
    let first = d.open("_0.doc").unwrap();
    let err = d.open("_1.doc").unwrap_err();
    assert!(err.to_string().contains("spill budget"), "{err}");
    drop(first);
    assert_eq!(d.open("_1.doc").unwrap().len(), 500_000, "budget returned");
    // `None`, or no disk tier: read whole, as before.
    let tmp = lucene_util::test_support::TempDir::new("cached-blob-heap");
    for (disk, min) in [(true, None), (false, Some(1))] {
        let cache = Arc::new(
            PageCache::new(PageCacheConfig {
                demand_paging: false,
                disk_dir: disk.then(|| tmp.path().join("heap")),
                spill_min_bytes: min,
                ..PageCacheConfig::default()
            })
            .unwrap(),
        );
        let dir = CachedBlobDirectory::new(Counting::default(), cache, "idx");
        put(&dir, "_0.doc", &data(2_000_000, 1));
        assert_eq!(dir.open("_0.doc").unwrap().len(), 2_000_000);
        assert_eq!(dir.cache().stats().spill_opens, 0);
    }
}

#[cfg(unix)]
#[test]
fn a_decoded_file_is_never_spilled_as_plaintext() {
    let config = PageCacheConfig {
        page_size: 16 * 1024,
        demand_paging: false,
        spill_min_bytes: Some(1),
        ..PageCacheConfig::default()
    };
    let (d, _store, _tmp) = spill_dir("cached-blob-spill-decoded", config);
    let plain = data(150_000, 9);
    put(&d, "_0.fdt", &encode(&plain));
    let input = d.open_decoded("_0.fdt", &XorDecoder).unwrap().unwrap();
    assert!(input[..] == plain[..]);
    assert_eq!(d.cache().stats().spill_opens, 0);
}

#[test]
fn the_heap_budget_bounds_files_held_whole_and_returns_on_drop() {
    let config = PageCacheConfig {
        page_size: 64 * 1024,
        demand_paging: false,
        spill_min_bytes: None,
        heap_budget: 300_000,
        ..PageCacheConfig::default()
    };
    let (d, store, _tmp) = spill_dir("cached-blob-heap-budget", config);
    put(&d, "_0.doc", &data(200_000, 1));
    put(&d, "_1.doc", &data(200_000, 2));
    put(&d, "_2.si", &data(50_000, 3)); // pinned
    let first = d.open("_0.doc").unwrap();
    assert!(first[..] == data(200_000, 1)[..]);
    let stats = d.cache().stats();
    assert_eq!((stats.heap_opens, stats.heap_open_bytes), (1, 200_000));
    // Refused before one byte is fetched.
    let ranges = store.ranges.load(Ordering::Relaxed);
    let err = d.open("_1.doc").unwrap_err();
    assert!(err.to_string().contains("heap budget"), "{err}");
    assert_eq!(store.ranges.load(Ordering::Relaxed), ranges);
    // A shared open of the same file counts once.
    let again = d.open("_0.doc").unwrap();
    assert_eq!(d.cache().stats().heap_open_bytes, 200_000);
    // A pinned file counts against it too.
    assert_eq!(d.open("_2.si").unwrap().len(), 50_000);
    drop((first, again));
    assert_eq!(d.cache().stats().heap_open_bytes, 0);
    assert_eq!(d.open("_1.doc").unwrap().len(), 200_000, "returned");
}

#[test]
fn demand_paged_opens_are_counted_and_availability_matches_the_pager() {
    let config = PageCacheConfig {
        page_size: 64 * 1024,
        ..PageCacheConfig::default()
    };
    let (d, _store, _tmp) = spill_dir("cached-blob-demand-count", config);
    put(&d, "_0.doc", &data(500_000, 1));
    let input = d.open("_0.doc").unwrap();
    let stats = d.cache().stats();
    if crate::page_cache::demand_paging_available() {
        assert_eq!((stats.demand_paged_opens, stats.heap_opens), (1, 0));
    } else {
        assert_eq!(stats.demand_paged_opens, 0);
    }
    assert_eq!(input.len(), 500_000);
}

#[test]
fn an_unbounded_heap_budget_reads_whole_files_uncounted_as_before() {
    let config = PageCacheConfig {
        page_size: 64 * 1024,
        demand_paging: false,
        spill_min_bytes: None,
        ..PageCacheConfig::default()
    };
    let (d, _store, _tmp) = spill_dir("cached-blob-heap-unbounded", config);
    put(&d, "_0.doc", &data(200_000, 1));
    let input = d.open("_0.doc").unwrap();
    assert!(input[..] == data(200_000, 1)[..]);
    let stats = d.cache().stats();
    assert_eq!((stats.heap_opens, stats.heap_open_bytes), (0, 0));
}
