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
    {
        let mut w = IndexWriter::open(dir, Vec::new(), "Lucene104", VERSION).unwrap();
        w.set_max_buffered_docs(37).unwrap();
        w.set_merge_policy(Some(MergePolicyConfig {
            max_merge_at_once: 3,
            segments_per_tier: 2.0,
            floor_segment_size: 1,
            ..MergePolicyConfig::default()
        }));
        for i in 0..400 {
            let mut d = Document::new();
            d.add(StringField::new("id", format!("doc-{i:05}"), Store::Yes));
            let parity = if i % 2 == 0 { "even" } else { "odd" };
            d.add(StringField::new("parity", parity, Store::No));
            w.add_fields_document(&d).unwrap();
            if i % 90 == 89 {
                w.commit().unwrap();
                w.maybe_merge().unwrap();
            }
        }
        for i in (0..400).step_by(10) {
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
    // 200 even ids, less the 40 deleted multiples of ten (all even).
    assert_eq!(searcher.search(&even, 10).unwrap().total_hits.value, 160);

    let mut ids = Vec::new();
    for seg in reader.segment_readers() {
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
    let want: Vec<String> = (0..400)
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
