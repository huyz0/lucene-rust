#![allow(clippy::arithmetic_side_effects)]
//! **The NRT managers over real commits.**
//!
//! Behaviour tests in the spirit of Lucene's `TestSearcherManager`,
//! `TestControlledRealTimeReopenThread`, `TestSearcherLifetimeManager` and
//! `TestLiveFieldValues`, run against an index this port's `IndexWriter`
//! writes and commits while searcher threads acquire and release: a refresh
//! publishes exactly the committed documents, a held searcher keeps its point
//! in time, concurrent acquirers never see the document count go backwards,
//! a waiter on a generation is released once a refresh covers it, versions
//! resolve to the searcher that was recorded, and live values win over the
//! searcher until a refresh makes them searchable.

use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use lucene_codecs::field_infos::{
    DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions, VectorEncoding,
    VectorSimilarityFunction,
};
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_index::segment_infos::SegmentInfos;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::reference_manager::{
    ControlledRealTimeReopenThread, LiveFieldValues, PruneByAge, RefreshCommitSupplier,
    SearcherFactory, SearcherLifetimeManager, SearcherManager,
};
use lucene_store::directory::Directory;
use lucene_store::FsDirectory;
use lucene_util::test_support::TempDir;

fn id_field() -> FieldInfo {
    FieldInfo {
        name: "id".to_string(),
        number: 0,
        store_term_vectors: false,
        omit_norms: false,
        store_payloads: false,
        soft_deletes_field: false,
        parent_field: false,
        index_options: IndexOptions::None,
        doc_values_type: DocValuesType::None,
        doc_values_skip_index_type: DocValuesSkipIndexType::None,
        doc_values_gen: -1,
        attributes: vec![],
        point_dimension_count: 0,
        point_index_dimension_count: 0,
        point_num_bytes: 0,
        vector_dimension: 0,
        vector_encoding: VectorEncoding::Float32,
        vector_similarity_function: VectorSimilarityFunction::Euclidean,
    }
}

fn doc(id: usize) -> Document {
    Document {
        fields: vec![StoredField {
            field_number: 0,
            value: FieldValue::String(format!("doc{id}")),
        }],
    }
}

fn version() -> LuceneVersion {
    LuceneVersion {
        major: 10,
        minor: 0,
        bugfix: 0,
    }
}

fn num_docs(r: &DirectoryReader) -> i32 {
    r.segment_readers().iter().map(|s| s.num_docs()).sum()
}

/// The stored `id` of every live document.
fn find(r: &DirectoryReader, id: &str) -> Option<i32> {
    for seg in r.segment_readers() {
        for d in 0..seg.max_doc {
            if let Some(doc) = seg.stored_document(d).unwrap() {
                if doc
                    .fields
                    .iter()
                    .any(|f| matches!(&f.value, FieldValue::String(s) if s == id))
                {
                    return Some(seg.doc_base + d);
                }
            }
        }
    }
    None
}

#[test]
fn searcher_manager_publishes_commits_to_concurrent_searchers() {
    let tmp = TempDir::new("searcher-manager");
    let dir: Arc<dyn Directory> = Arc::new(FsDirectory::open(&tmp));
    let mut writer =
        IndexWriter::open(dir.as_ref(), vec![id_field()], "Lucene104", version()).expect("writer");
    for i in 0..10 {
        writer.add_document(doc(i)).unwrap();
    }
    writer.commit().unwrap();

    let mgr = Arc::new(SearcherManager::open(Arc::clone(&dir)).unwrap());
    let first = mgr.acquire().unwrap();
    assert_eq!(num_docs(&first), 10);
    assert!(mgr.is_searcher_current().unwrap());
    let gen0 = mgr.searcher_commit_generation().unwrap();

    // Searcher threads: each acquires, checks the count never goes backwards,
    // releases -- while the main thread commits and refreshes.
    let stop = Arc::new(AtomicBool::new(false));
    let threads: Vec<_> = (0..4)
        .map(|_| {
            let (mgr, stop) = (Arc::clone(&mgr), Arc::clone(&stop));
            std::thread::spawn(move || {
                let mut last = 0;
                let mut seen = 0usize;
                while !stop.load(Ordering::SeqCst) {
                    let s = mgr.acquire().unwrap();
                    let n = num_docs(&s);
                    assert!(n >= last, "count went backwards: {n} < {last}");
                    assert_eq!(n % 10, 0, "a refresh publishes whole commits");
                    last = n;
                    mgr.release(s);
                    seen += 1;
                    // Some threads refresh too, non-blocking.
                    let _ = mgr.maybe_refresh().unwrap();
                }
                (last, seen)
            })
        })
        .collect();
    for round in 1..=5 {
        for i in 0..10 {
            writer.add_document(doc(round * 10 + i)).unwrap();
        }
        writer.commit().unwrap();
        mgr.maybe_refresh_blocking().unwrap();
        assert!(mgr.is_searcher_current().unwrap());
    }
    stop.store(true, Ordering::SeqCst);
    for t in threads {
        let (last, seen) = t.join().unwrap();
        assert!(seen > 0);
        assert!(last <= 60);
    }
    let now = mgr.acquire().unwrap();
    assert_eq!(num_docs(&now), 60);
    assert!(mgr.searcher_commit_generation().unwrap() > gen0);
    assert_eq!(
        num_docs(&first),
        10,
        "a held searcher keeps its point in time"
    );
    assert!(!mgr.maybe_refresh().is_err());

    // Uncommitted documents stay invisible.
    writer.add_document(doc(999)).unwrap();
    mgr.maybe_refresh_blocking().unwrap();
    assert_eq!(num_docs(&mgr.acquire().unwrap()), 60);
    writer.commit().unwrap();
    assert!(!mgr.is_searcher_current().unwrap());

    mgr.close().unwrap();
    assert!(mgr.acquire().is_err());
    assert_eq!(
        num_docs(&now),
        60,
        "closing the manager leaves holders alone"
    );
}

/// A factory that counts what it warms and wraps the reader.
struct Warming(AtomicI64);

struct Warmed {
    reader: DirectoryReader,
    warmed_after: Option<i32>,
}

impl SearcherFactory for Warming {
    type Searcher = Warmed;
    fn new_searcher(
        &self,
        reader: DirectoryReader,
        previous: Option<&Warmed>,
    ) -> lucene_search::Result<Warmed> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(Warmed {
            reader,
            warmed_after: previous.map(|p| num_docs(&p.reader)),
        })
    }
    fn reader<'s>(&self, s: &'s Warmed) -> &'s DirectoryReader {
        &s.reader
    }
}

/// Refreshes only to the commit it is told to.
struct Pinned(std::sync::Mutex<Option<SegmentInfos>>);

impl RefreshCommitSupplier for Pinned {
    fn searcher_refresh_commit(
        &self,
        _reader: &DirectoryReader,
    ) -> lucene_search::Result<Option<SegmentInfos>> {
        Ok(self.0.lock().unwrap().clone())
    }
}

#[test]
fn factory_warms_and_commit_supplier_chooses() {
    let tmp = TempDir::new("searcher-factory");
    let dir: Arc<dyn Directory> = Arc::new(FsDirectory::open(&tmp));
    let mut writer =
        IndexWriter::open(dir.as_ref(), vec![id_field()], "Lucene104", version()).expect("writer");
    writer.add_document(doc(0)).unwrap();
    let first_commit = writer.commit().unwrap().clone();

    let warming = Warming(AtomicI64::new(0));
    let reader = DirectoryReader::open(dir.as_ref()).unwrap();
    let pinned = Arc::new(Pinned(std::sync::Mutex::new(Some(first_commit.clone()))));
    struct Shared(Arc<Pinned>);
    impl RefreshCommitSupplier for Shared {
        fn searcher_refresh_commit(
            &self,
            r: &DirectoryReader,
        ) -> lucene_search::Result<Option<SegmentInfos>> {
            self.0.searcher_refresh_commit(r)
        }
    }
    let mgr = SearcherManager::from_reader(
        Arc::clone(&dir),
        reader,
        warming,
        Box::new(Shared(Arc::clone(&pinned))),
    )
    .unwrap();
    assert!(mgr.acquire().unwrap().warmed_after.is_none());

    writer.add_document(doc(1)).unwrap();
    let second_commit = writer.commit().unwrap().clone();
    // Pinned to the commit already open: no refresh.
    mgr.maybe_refresh_blocking().unwrap();
    assert_eq!(num_docs(&mgr.acquire().unwrap().reader), 1);
    // Pinned to the new commit: refreshed, warmed against the old searcher.
    *pinned.0.lock().unwrap() = Some(second_commit);
    mgr.maybe_refresh_blocking().unwrap();
    let s = mgr.acquire().unwrap();
    assert_eq!(num_docs(&s.reader), 2);
    assert_eq!(s.warmed_after, Some(1));
    // `None`: the latest.
    *pinned.0.lock().unwrap() = None;
    writer.add_document(doc(2)).unwrap();
    writer.commit().unwrap();
    mgr.maybe_refresh_blocking().unwrap();
    assert_eq!(num_docs(&mgr.acquire().unwrap().reader), 3);
    let _ = first_commit;
}

#[test]
fn reopen_thread_lifetime_manager_and_live_values_over_commits() {
    let tmp = TempDir::new("nrt-managers");
    let dir: Arc<dyn Directory> = Arc::new(FsDirectory::open(&tmp));
    let mut writer =
        IndexWriter::open(dir.as_ref(), vec![id_field()], "Lucene104", version()).expect("writer");
    writer.add_document(doc(0)).unwrap();
    writer.commit().unwrap();

    let mgr = Arc::new(SearcherManager::open(Arc::clone(&dir)).unwrap());
    // The generation a refresh starting now would show: the last committed
    // add's sequence number.
    let committed = Arc::new(AtomicI64::new(0));
    let live: LiveFieldValues<DirectoryReader, i32> = LiveFieldValues::new(
        mgr.shared_manager(),
        Box::new(|r: &DirectoryReader, id: &str| Ok(find(r, id))),
    );
    let lifetimes: SearcherLifetimeManager<DirectoryReader> = SearcherLifetimeManager::new();
    let v0 = lifetimes.record(&mgr.acquire().unwrap()).unwrap();

    let thread = ControlledRealTimeReopenThread::start(
        mgr.shared_manager(),
        Box::new({
            let c = Arc::clone(&committed);
            move || c.load(Ordering::SeqCst)
        }),
        Duration::from_secs(5),
        Duration::from_millis(10),
    )
    .unwrap();

    for i in 1..=3 {
        let seq = writer.add_document(doc(i)).unwrap();
        live.add(&format!("doc{i}"), -1);
        writer.commit().unwrap();
        committed.store(seq as i64, Ordering::SeqCst);
        assert!(
            thread.wait_for_generation(seq as i64, Some(Duration::from_secs(30))),
            "generation {seq} became searchable"
        );
        let s = mgr.acquire().unwrap();
        assert_eq!(num_docs(&s), i as i32 + 1);
        // After the refresh the live map is empty and the searcher answers.
        assert!(find(&s, &format!("doc{i}")).is_some());
    }
    assert_eq!(live.size(), 0);
    assert_eq!(
        live.get("doc2").unwrap(),
        find(&mgr.acquire().unwrap(), "doc2")
    );
    live.add("pending", 7);
    assert_eq!(live.get("pending").unwrap(), Some(7));
    live.delete("doc1");
    assert_eq!(live.get("doc1").unwrap(), None, "deleted since the refresh");
    assert_eq!(live.get("nosuch").unwrap(), None);

    // The first point in time is still there by version; the latest has its
    // own.
    let latest = mgr.acquire().unwrap();
    let v1 = lifetimes.record(&latest).unwrap();
    assert_ne!(v0, v1);
    assert_eq!(num_docs(&lifetimes.acquire(v0).unwrap().unwrap()), 1);
    assert_eq!(num_docs(&lifetimes.acquire(v1).unwrap().unwrap()), 4);
    lifetimes.prune(&PruneByAge::new(3600.0).unwrap());
    assert_eq!(lifetimes.len(), 2);

    thread.close().unwrap();
    live.close();
}
