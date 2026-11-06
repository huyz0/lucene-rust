//! A whole index -- flushes, commits, merges, deletes, `CheckIndex`, a
//! search and stored fields -- written and read through the pluggable store
//! layer: an encrypted filesystem directory, a blob store, and an encrypted
//! blob store (BYOK over object storage). The codecs never know.

// Test fixtures' own arithmetic -- see `docs/arithmetic-gate.md`'s "Test code".
#![allow(clippy::arithmetic_side_effects)]

use std::sync::Arc;

use lucene_codecs::stored_fields::FieldValue;
use lucene_index::buffered_updates::Term;
use lucene_index::document::{Document, Store, StringField};
use lucene_index::index_writer::{IndexWriter, MergePolicyConfig};
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::index_searcher::{IndexSearcher, SegmentNorms};
use lucene_search::query::{BooleanQuery, Clause, TermQuery};
use lucene_store::{BlobDirectory, Directory, FsDirectory, InMemoryBlobStore};
use lucene_store_crypto::{CachingKeyProvider, EncryptedDirectory, StaticKeyProvider};
use lucene_util::test_support::TempDir;

const VERSION: LuceneVersion = LuceneVersion {
    major: 10,
    minor: 5,
    bugfix: 0,
};

/// Indexes 400 documents with commits and merges, deletes every tenth, and
/// checks the index, a term search and the stored ids read back.
fn index_and_verify(dir: &dyn Directory) {
    index_and_verify_n(dir, 400);
}

/// [`index_and_verify`] with `n` documents (a multiple of 10).
fn index_and_verify_n(dir: &dyn Directory, n: usize) {
    let commit_every = if n == 400 { 90 } else { n / 4 };
    {
        let mut w = IndexWriter::open(dir, Vec::new(), "Lucene104", VERSION).unwrap();
        let buffered = if n == 400 { 37 } else { n as i32 / 8 };
        w.set_max_buffered_docs(buffered).unwrap();
        w.set_merge_policy(Some(MergePolicyConfig {
            max_merge_at_once: 3,
            segments_per_tier: 2.0,
            floor_segment_size: 1,
            ..MergePolicyConfig::default()
        }));
        for i in 0..n {
            let mut d = Document::new();
            d.add(StringField::new("id", format!("doc-{i:05}"), Store::Yes));
            let parity = if i % 2 == 0 { "even" } else { "odd" };
            d.add(StringField::new("parity", parity, Store::No));
            w.add_fields_document(&d).unwrap();
            if i % commit_every == commit_every - 1 {
                w.commit().unwrap();
                w.maybe_merge().unwrap();
            }
        }
        for i in (0..n).step_by(10) {
            w.delete_documents_by_term(&[Term::new("id", format!("doc-{i:05}"))])
                .unwrap();
        }
        w.commit().unwrap();
        w.maybe_merge().unwrap();
        w.commit().unwrap();
    }

    for r in lucene_index::check_index::check_directory(dir).unwrap() {
        assert!(r.all_passed(), "{}: {:?}", r.segment_name, r.failures());
    }
    let reader = DirectoryReader::open(dir).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let owned = reader
        .field_norms_by_field(&["parity".to_string()])
        .unwrap();
    let norms: Vec<SegmentNorms<'_, '_>> = owned.iter().map(Some).collect();
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();
    let even = BooleanQuery {
        must: vec![Clause::Term(TermQuery::new("parity", b"even".to_vec()))],
        ..Default::default()
    };
    // The even ids, less the deleted multiples of ten (all even); hit
    // counts past 1000 are a lower bound.
    let hits = searcher.search(&even, 10).unwrap().total_hits.value;
    let want = (n / 2 - n / 10) as u64;
    if want <= 1000 {
        assert_eq!(hits, want);
    } else {
        assert!((1000..=want).contains(&hits), "{hits}");
    }

    let mut ids = Vec::new();
    for seg in reader.segment_readers() {
        let every: Vec<i32> = (0..seg.max_doc).collect();
        seg.prefetch_stored_documents(&every).unwrap();
        for doc in 0..seg.max_doc {
            if seg.live_docs().is_some_and(|l| !l.get_doc(doc)) {
                continue;
            }
            let stored = seg.stored_document(doc).unwrap().unwrap();
            let FieldValue::String(id) = &stored.fields[0].value else {
                panic!("an id");
            };
            ids.push(id.clone());
        }
    }
    ids.sort();
    let want: Vec<String> = (0..n)
        .filter(|i| i % 10 != 0)
        .map(|i| format!("doc-{i:05}"))
        .collect();
    assert_eq!(ids, want);
}

fn keys() -> Arc<CachingKeyProvider<StaticKeyProvider>> {
    let kms = StaticKeyProvider::new("tenant-a/v1", &[0x5a; 32]);
    Arc::new(CachingKeyProvider::new(
        kms,
        std::time::Duration::from_secs(300),
        1024,
    ))
}

#[test]
fn an_encrypted_filesystem_index_round_trips_and_holds_no_plaintext() {
    let tmp = TempDir::new("crypto-index-fs");
    let dir = EncryptedDirectory::new(FsDirectory::open(tmp.path()), keys());
    index_and_verify(&dir);

    let mut files = 0;
    for entry in std::fs::read_dir(tmp.path()).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        if name == "write.lock" {
            continue;
        }
        let raw = std::fs::read(&path).unwrap();
        assert!(raw.starts_with(b"LRSE"), "{name} is not encrypted");
        for needle in [&b"doc-00042"[..], b"parity", b"Lucene104", b"segments"] {
            assert!(
                !raw.windows(needle.len()).any(|w| w == needle),
                "{name} leaks {:?}",
                String::from_utf8_lossy(needle)
            );
        }
        files += 1;
    }
    assert!(files > 3, "an index was written");

    // Without the key, the index does not open.
    let wrong = EncryptedDirectory::new(
        FsDirectory::open(tmp.path()),
        Arc::new(StaticKeyProvider::new("tenant-a/v1", &[1; 32])),
    );
    assert!(DirectoryReader::open(&wrong).is_err());
}

#[test]
fn a_blob_store_index_round_trips() {
    let dir = BlobDirectory::new(InMemoryBlobStore::new());
    index_and_verify(&dir);
    assert!(dir
        .list_all()
        .unwrap()
        .iter()
        .any(|n| n.starts_with("segments_")));
}

#[test]
fn an_encrypted_blob_store_index_round_trips() {
    let store = InMemoryBlobStore::new();
    let dir = EncryptedDirectory::new(BlobDirectory::new(store.clone()), keys());
    index_and_verify(&dir);
    let names = dir.list_all().unwrap();
    assert!(!names.is_empty());
    for name in names {
        let raw = lucene_store::BlobStore::get(&store, &name).unwrap();
        assert!(raw.starts_with(b"LRSE"), "{name}");
    }
}

/// The same index with files demand-decrypted (where `userfaultfd` is
/// available) under a budget of a few chunks, over a memory-mapped and a
/// blob directory: writes, merges, `CheckIndex` and searches all read
/// through the pager while it keeps evicting.
#[test]
fn a_demand_decrypted_index_round_trips_under_a_tiny_budget() {
    #[cfg(target_os = "linux")]
    if let Some(pager) = lucene_store_crypto::lazy::Pager::global() {
        pager.set_budget(256 * 1024);
    }
    let tmp = TempDir::new("crypto-index-lazy");
    let mmap = lucene_store::MmapDirectory::open(tmp.path());
    index_and_verify(
        &EncryptedDirectory::new(mmap, keys())
            .with_chunk_size(4096)
            .unwrap()
            .with_lazy_decrypt(true),
    );
    let blob = BlobDirectory::new(InMemoryBlobStore::new());
    index_and_verify(&EncryptedDirectory::new(blob, keys()).with_lazy_decrypt(true));
    #[cfg(target_os = "linux")]
    if let Some(pager) = lucene_store_crypto::lazy::Pager::global() {
        assert!(pager.faults() > 0, "reads went through the pager");
        pager.set_budget(lucene_store_crypto::lazy::DEFAULT_BUDGET);
    }
}

fn page_cache(tmp: &TempDir, pinned: bool) -> Arc<lucene_store::PageCache> {
    // Pages of one OS page, so even this small index's files span several.
    let mut config = lucene_store::PageCacheConfig {
        page_size: 4096,
        disk_dir: Some(tmp.path().join("nvme")),
        ..lucene_store::PageCacheConfig::default()
    };
    if !pinned {
        config.pinned_extensions.clear();
    }
    Arc::new(lucene_store::PageCache::new(config).unwrap())
}

/// Every file read on disk under `root`, recursively.
fn cached_files(root: &std::path::Path) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                stack.push(path);
            } else {
                out.push(std::fs::read(path).unwrap());
            }
        }
    }
    out
}

/// The index through the page cache over a blob store: files read by
/// range, page by page, with and without the pinned hot files, under a
/// resident budget of a few pages.
#[test]
fn an_index_read_through_the_page_cache_round_trips() {
    #[cfg(target_os = "linux")]
    if let Some(pager) = lucene_store::paged_input::Pager::global() {
        pager.set_budget(32 * 1024);
    }
    for pinned in [false, true] {
        let tmp = TempDir::new("cached-index");
        let cache = page_cache(&tmp, pinned);
        let dir = lucene_store::CachedBlobDirectory::new(
            InMemoryBlobStore::new(),
            Arc::clone(&cache),
            "idx",
        );
        index_and_verify_n(&dir, 20_000);
        let stats = cache.stats();
        assert!(stats.requests > 0 && stats.fetched_bytes > 0, "{stats:?}");
        assert!(cache.disk_bytes() > 0);
    }
    #[cfg(target_os = "linux")]
    if let Some(pager) = lucene_store::paged_input::Pager::global() {
        assert!(pager.faults() > 0, "reads went through the pager");
        pager.set_budget(lucene_store::paged_input::DEFAULT_BUDGET);
    }
}

/// BYOK over the page cache: files are encrypted in the store and in the
/// disk tier (ciphertext pages of whole chunks), decrypted only in memory.
#[test]
fn an_encrypted_index_read_through_the_page_cache_keeps_ciphertext_on_disk() {
    let tmp = TempDir::new("cached-encrypted-index");
    let cache = page_cache(&tmp, true);
    let store = InMemoryBlobStore::new();
    let cached = lucene_store::CachedBlobDirectory::new(store.clone(), Arc::clone(&cache), "idx");
    let dir = EncryptedDirectory::new(cached, keys())
        .with_chunk_size(4096)
        .unwrap();
    index_and_verify_n(&dir, 20_000);
    assert!(cache.disk_bytes() > 0);
    for page in cached_files(&tmp.path().join("nvme")) {
        for needle in [&b"doc-04242"[..], b"parity", b"Lucene104"] {
            assert!(
                !page.windows(needle.len()).any(|w| w == needle),
                "a cached page leaks {:?}",
                String::from_utf8_lossy(needle)
            );
        }
    }

    // A tampered chunk fails the read that brings its page in.
    let mut out = dir.create_output("_9.tip").unwrap();
    lucene_store::DataOutput::write_bytes(&mut out, &[7u8; 50_000]);
    out.close().unwrap();
    let mut raw = lucene_store::BlobStore::get(&store, "_9.tip")
        .unwrap()
        .to_vec();
    let at = raw.len() - 9000;
    raw[at] ^= 1;
    lucene_store::BlobStore::put(&store, "_9.tip", raw).unwrap();
    let err = dir.open("_9.tip").unwrap_err();
    assert!(format!("{err}").contains("failed authentication"), "{err}");
    // And a wrong key opens nothing.
    let wrong = EncryptedDirectory::new(
        lucene_store::CachedBlobDirectory::new(store, page_cache(&TempDir::new("w"), true), "w"),
        Arc::new(StaticKeyProvider::new("tenant-a/v1", &[1; 32])),
    );
    assert!(DirectoryReader::open(&wrong).is_err());
}
