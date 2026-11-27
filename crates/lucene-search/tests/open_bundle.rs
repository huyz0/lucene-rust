//! The open bundle written with every commit and merge, and a cached
//! reader's use of it: an index written through `BlobDirectory` carries the
//! sidecar, a reader opens from it in one request, and a stale, missing or
//! damaged bundle changes nothing but the number of requests.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use lucene_index::document::{Document, Store, StringField, TextField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::index_searcher::{IndexSearcher, SegmentNorms};
use lucene_search::query::{BooleanQuery, Clause, TermQuery};
use lucene_store::open_bundle::{parse_open_bundle, OPEN_BUNDLE_NAME};
use lucene_store::{
    BlobDirectory, BlobStore, CachedBlobDirectory, InMemoryBlobStore, Input, PageCache,
    PageCacheConfig, Result,
};
use lucene_util::test_support::TempDir;

const VERSION: LuceneVersion = LuceneVersion {
    major: 10,
    minor: 5,
    bugfix: 0,
};

/// Counts the requests that read file bytes, by kind.
#[derive(Clone)]
struct Counting {
    inner: InMemoryBlobStore,
    counts: Arc<Mutex<BTreeMap<&'static str, u64>>>,
}

impl Counting {
    fn new(inner: InMemoryBlobStore) -> Self {
        Self {
            inner,
            counts: Arc::default(),
        }
    }
    fn bump(&self, kind: &'static str) {
        *self.counts.lock().unwrap().entry(kind).or_default() += 1;
    }
    fn take(&self) -> BTreeMap<&'static str, u64> {
        std::mem::take(&mut *self.counts.lock().unwrap())
    }
}

impl BlobStore for Counting {
    fn list(&self) -> Result<Vec<String>> {
        self.inner.list()
    }
    fn get(&self, name: &str) -> Result<Input> {
        self.bump(if name == OPEN_BUNDLE_NAME {
            "bundle"
        } else {
            "get"
        });
        self.inner.get(name)
    }
    fn get_range(&self, name: &str, offset: u64, len: usize) -> Result<Vec<u8>> {
        self.bump("range");
        self.inner.get_range(name, offset, len)
    }
    fn size(&self, name: &str) -> Result<u64> {
        self.bump("size");
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

fn doc(i: usize) -> Document {
    let mut d = Document::new();
    d.add(StringField::new("id", format!("doc-{i:05}"), Store::Yes));
    d.add(TextField::new(
        "body",
        format!("alpha beta w{} w{}", i % 7, i % 13),
        Store::Yes,
    ));
    d
}

/// `commits` commits of `per` documents each, written through `dir`.
fn write<D: lucene_store::Directory>(dir: &D, first: usize, commits: usize, per: usize) {
    let mut w = IndexWriter::open(dir, Vec::new(), "Lucene104", VERSION).unwrap();
    for c in 0..commits {
        for i in 0..per {
            w.add_fields_document(&doc(first + c * per + i)).unwrap();
        }
        w.commit().unwrap();
    }
}

fn bundle_entries(store: &InMemoryBlobStore) -> BTreeMap<String, Vec<u8>> {
    let bytes = store.get(OPEN_BUNDLE_NAME).unwrap();
    parse_open_bundle(&bytes)
        .expect("an intact bundle")
        .into_iter()
        .map(|(n, b)| (n, b.to_vec()))
        .collect()
}

fn cached(store: &Counting, tmp: &TempDir) -> CachedBlobDirectory<Counting> {
    let cache = Arc::new(
        PageCache::new(PageCacheConfig {
            disk_dir: Some(tmp.path().join("nvme")),
            ..PageCacheConfig::default()
        })
        .unwrap(),
    );
    CachedBlobDirectory::new(store.clone(), cache, "idx")
}

/// Hits of `alpha` and `w3` through a reader over `dir`.
fn hits<D: lucene_store::Directory>(dir: &D) -> (usize, usize) {
    let reader = DirectoryReader::open(dir).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let owned = reader.field_norms_by_field(&["body".to_string()]).unwrap();
    let norms: Vec<SegmentNorms<'_, '_>> = owned.iter().map(Some).collect();
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();
    let count = |term: &str| {
        let q = BooleanQuery {
            must: vec![Clause::Term(TermQuery::new("body", term))],
            ..BooleanQuery::default()
        };
        searcher.search(&q, 10_000).unwrap().score_docs.len()
    };
    (count("alpha"), count("w3"))
}

#[test]
fn every_commit_and_merge_publishes_a_bundle_that_matches_the_store() {
    let inner = InMemoryBlobStore::new();
    let dir = BlobDirectory::new(inner.clone());
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
    for c in 0..3 {
        for i in 0..50 {
            w.add_fields_document(&doc(c * 50 + i)).unwrap();
        }
        w.commit().unwrap();
        let entries = bundle_entries(&inner);
        let commits: Vec<&String> = entries
            .keys()
            .filter(|n| n.starts_with("segments_"))
            .collect();
        assert_eq!(commits.len(), 1, "only the newest commit: {commits:?}");
        assert!(inner.list().unwrap().contains(commits[0]));
        for (name, bytes) in &entries {
            assert_eq!(&inner.get(name).unwrap()[..], &bytes[..], "{name}");
        }
    }
    assert!(bundle_entries(&inner).keys().any(|n| n.ends_with(".si")));

    w.force_merge(1).unwrap();
    w.commit().unwrap();
    let entries = bundle_entries(&inner);
    let live: Vec<String> = inner.list().unwrap();
    let newest = live
        .iter()
        .filter(|n| n.starts_with("segments_"))
        .max()
        .unwrap();
    assert!(entries.contains_key(newest), "newest commit is bundled");
    for (name, bytes) in &entries {
        // Entries for files the merge deleted afterwards are harmless: a reader only
        // uses an entry whose name and length match the live listing.
        if live.contains(name) {
            assert_eq!(&inner.get(name).unwrap()[..], &bytes[..], "{name}");
        }
    }
    assert!(live
        .iter()
        .any(|n| n.ends_with(".si") && entries.contains_key(n)));
}

#[test]
fn a_cached_reader_opens_from_the_bundle_in_one_request_by_default() {
    let inner = InMemoryBlobStore::new();
    write(&BlobDirectory::new(inner.clone()), 0, 3, 50);
    let plain = hits(&BlobDirectory::new(inner.clone()));

    let tmp = TempDir::new("bundle-default");
    let store = Counting::new(inner.clone());
    let with = cached(&store, &tmp);
    assert_eq!(hits(&with), plain);
    let counts = store.take();
    assert_eq!(counts.get("bundle"), Some(&1), "{counts:?}");

    let tmp2 = TempDir::new("bundle-off");
    let store2 = Counting::new(inner);
    let without = cached(&store2, &tmp2).with_open_bundle(false);
    assert_eq!(hits(&without), plain);
    let off = store2.take();
    assert_eq!(off.get("bundle"), None);
    let on_requests: u64 = counts.values().sum();
    let off_requests: u64 = off.values().sum();
    assert!(
        on_requests < off_requests,
        "bundle {counts:?} against ranged reads {off:?}"
    );
}

#[test]
fn a_commit_by_a_writer_that_knows_no_bundle_leaves_it_stale_and_changes_no_result() {
    let inner = InMemoryBlobStore::new();
    write(&BlobDirectory::new(inner.clone()), 0, 2, 50);
    let before = bundle_entries(&inner);
    // A writer that never heard of the bundle (a stock Lucene writer does the
    // same): it adds a segment and a commit and leaves the sidecar alone.
    let stock = BlobDirectory::new(inner.clone()).with_open_bundle(false);
    write(&stock, 100, 1, 50);
    assert_eq!(
        bundle_entries(&inner),
        before,
        "the sidecar was not touched"
    );
    assert!(inner
        .list()
        .unwrap()
        .iter()
        .any(|n| n.starts_with("segments_") && !before.contains_key(n)));

    let plain = hits(&BlobDirectory::new(inner.clone()));
    assert_eq!(plain.0, 150);
    let tmp = TempDir::new("bundle-stale");
    let store = Counting::new(inner);
    assert_eq!(hits(&cached(&store, &tmp)), plain);
}

#[test]
fn a_missing_or_damaged_bundle_leaves_every_read_to_ranged_requests() {
    let inner = InMemoryBlobStore::new();
    write(&BlobDirectory::new(inner.clone()), 0, 3, 50);
    let plain = hits(&BlobDirectory::new(inner.clone()));

    inner.delete(OPEN_BUNDLE_NAME).unwrap();
    let tmp = TempDir::new("bundle-missing");
    let missing = Counting::new(inner.clone());
    assert_eq!(hits(&cached(&missing, &tmp)), plain);

    let good = bundle_entries(&{
        let s = InMemoryBlobStore::new();
        write(&BlobDirectory::new(s.clone()), 0, 1, 10);
        s
    });
    assert!(!good.is_empty());
    for garbage in [vec![], vec![7u8; 5], vec![0xAB; 4096]] {
        inner.put(OPEN_BUNDLE_NAME, garbage).unwrap();
        let tmp = TempDir::new("bundle-garbage");
        let store = Counting::new(inner.clone());
        assert_eq!(hits(&cached(&store, &tmp)), plain);
    }
    let mut flipped = {
        let s = InMemoryBlobStore::new();
        write(&BlobDirectory::new(s.clone()), 0, 3, 50);
        s.get(OPEN_BUNDLE_NAME).unwrap().to_vec()
    };
    let mid = flipped.len() / 2;
    flipped[mid] ^= 0xFF;
    inner.put(OPEN_BUNDLE_NAME, flipped).unwrap();
    let tmp = TempDir::new("bundle-flipped");
    let store = Counting::new(inner);
    assert_eq!(hits(&cached(&store, &tmp)), plain);
}

#[test]
fn the_bundle_is_the_only_new_object_and_no_lucene_file_changes() {
    let with = InMemoryBlobStore::new();
    write(&BlobDirectory::new(with.clone()), 0, 2, 40);
    let without = InMemoryBlobStore::new();
    write(
        &BlobDirectory::new(without.clone()).with_open_bundle(false),
        0,
        2,
        40,
    );
    let names = |s: &InMemoryBlobStore| {
        let mut n = s.list().unwrap();
        n.sort();
        n
    };
    let extra: Vec<String> = names(&with)
        .into_iter()
        .filter(|n| !names(&without).contains(n))
        .collect();
    assert_eq!(extra, vec![OPEN_BUNDLE_NAME.to_string()]);
}
