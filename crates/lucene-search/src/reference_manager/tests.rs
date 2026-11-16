#![cfg(test)]
use super::*;
use std::sync::atomic::{AtomicI64, AtomicUsize, Ordering};

/// A reference that is a counter; each refresh bumps it when asked to.
struct Counter {
    next: AtomicI64,
    change: std::sync::atomic::AtomicBool,
    closed: AtomicUsize,
    after: AtomicUsize,
}

impl Refresher<i64> for Arc<Counter> {
    fn refresh_if_needed(&self, current: &Arc<i64>) -> Result<Option<Arc<i64>>> {
        if self.change.swap(false, Ordering::SeqCst) {
            Ok(Some(Arc::new(
                self.next.fetch_add(1, Ordering::SeqCst).max(**current + 1),
            )))
        } else {
            Ok(None)
        }
    }
    fn after_maybe_refresh(&self) -> Result<()> {
        self.after.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    fn after_close(&self) -> Result<()> {
        self.closed.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

fn counter() -> Arc<Counter> {
    Arc::new(Counter {
        next: AtomicI64::new(1),
        change: std::sync::atomic::AtomicBool::new(false),
        closed: AtomicUsize::new(0),
        after: AtomicUsize::new(0),
    })
}

struct Recorder(Mutex<Vec<String>>);

impl RefreshListener for Recorder {
    fn before_refresh(&self) -> Result<()> {
        lock(&self.0).push("before".into());
        Ok(())
    }
    fn after_refresh(&self, did: bool) -> Result<()> {
        lock(&self.0).push(format!("after {did}"));
        Ok(())
    }
}

#[test]
fn reference_manager_swaps_and_closes() {
    let c = counter();
    let m = ReferenceManager::new(Arc::new(0i64), Box::new(Arc::clone(&c)));
    let rec = Arc::new(Recorder(Mutex::new(Vec::new())));
    let l: Arc<dyn RefreshListener> = rec.clone();
    m.add_listener(Arc::clone(&l));
    let held = m.acquire().unwrap();
    assert_eq!(*held, 0);
    assert!(m.maybe_refresh().unwrap());
    assert_eq!(*m.acquire().unwrap(), 0, "nothing changed");
    c.change.store(true, Ordering::SeqCst);
    m.maybe_refresh_blocking().unwrap();
    assert_eq!(*m.acquire().unwrap(), 1);
    assert_eq!(*held, 0, "a held reference outlives the swap");
    m.release(held);
    assert_eq!(
        *lock(&rec.0),
        vec!["before", "after false", "before", "after true"]
    );
    assert_eq!(c.after.load(Ordering::SeqCst), 2);
    m.remove_listener(&l);
    m.maybe_refresh().unwrap();
    assert_eq!(lock(&rec.0).len(), 4, "removed");
    m.close().unwrap();
    m.close().unwrap();
    assert_eq!(c.closed.load(Ordering::SeqCst), 1);
    assert!(m.is_closed());
    assert!(matches!(m.acquire(), Err(Error::AlreadyClosed(_))));
    assert!(m.maybe_refresh().is_err());
    assert!(m.maybe_refresh_blocking().is_err());
}

#[test]
fn a_refresh_returning_the_same_reference_is_refused() {
    struct Same;
    impl Refresher<i64> for Same {
        fn refresh_if_needed(&self, current: &Arc<i64>) -> Result<Option<Arc<i64>>> {
            Ok(Some(Arc::clone(current)))
        }
    }
    let m = ReferenceManager::new(Arc::new(0i64), Box::new(Same));
    assert!(matches!(m.maybe_refresh(), Err(Error::IllegalState(_))));
}

#[test]
fn concurrent_maybe_refresh_skips_while_busy() {
    // `TestSearcherManager.testMaybeRefreshBlockingLock`'s intent: a
    // non-blocking refresh returns `false` while another refreshes.
    struct Slow(Arc<std::sync::Barrier>, Arc<std::sync::Barrier>);
    impl Refresher<i64> for Slow {
        fn refresh_if_needed(&self, _current: &Arc<i64>) -> Result<Option<Arc<i64>>> {
            self.0.wait();
            self.1.wait();
            Ok(None)
        }
    }
    let (a, b) = (
        Arc::new(std::sync::Barrier::new(2)),
        Arc::new(std::sync::Barrier::new(2)),
    );
    let m = Arc::new(ReferenceManager::new(
        Arc::new(0i64),
        Box::new(Slow(Arc::clone(&a), Arc::clone(&b))),
    ));
    let m2 = Arc::clone(&m);
    let t = std::thread::spawn(move || m2.maybe_refresh_blocking().unwrap());
    a.wait();
    assert!(!m.maybe_refresh().unwrap(), "busy: skipped");
    b.wait();
    t.join().unwrap();
}

#[derive(Debug)]
struct V(i64);
impl Versioned for V {
    fn version(&self) -> i64 {
        self.0
    }
}

#[test]
fn lifetime_manager_records_acquires_prunes() {
    // `TestSearcherLifetimeManager`'s intent.
    let lm: SearcherLifetimeManager<V> = SearcherLifetimeManager::new();
    let (a, b, c) = (Arc::new(V(1)), Arc::new(V(2)), Arc::new(V(3)));
    assert_eq!(lm.record(&a).unwrap(), 1);
    assert_eq!(lm.record(&a).unwrap(), 1, "the same searcher again");
    assert!(
        lm.record(&Arc::new(V(1))).is_err(),
        "another of the same version"
    );
    lm.record(&b).unwrap();
    lm.record(&c).unwrap();
    assert_eq!(lm.len(), 3);
    assert_eq!(lm.acquire(2).unwrap().unwrap().0, 2);
    assert!(lm.acquire(9).unwrap().is_none());
    assert!(PruneByAge::new(-1.0).is_err());
    // Prune against a clock 10s on: the newest is age 0, the older two
    // age ~10s since their successors were recorded.
    lm.prune_at(
        &PruneByAge::new(5.0).unwrap(),
        Instant::now() + Duration::from_secs(10),
    );
    assert_eq!(lm.len(), 1);
    assert!(lm.acquire(3).unwrap().is_some(), "the newest survives");
    lm.prune(&PruneByAge::new(1000.0).unwrap());
    assert_eq!(lm.len(), 1);
    lm.release(a);
    lm.close();
    assert!(lm.is_empty());
    assert!(lm.acquire(3).is_err());
    assert!(lm.record(&b).is_err());
}

#[test]
fn reopen_thread_waits_for_generations() {
    // `TestControlledRealTimeReopenThread`'s intent: a waiter on a
    // generation is released by a refresh that started after it.
    let c = counter();
    let m = Arc::new(ReferenceManager::new(
        Arc::new(0i64),
        Box::new(Arc::clone(&c)),
    ));
    let gen = Arc::new(AtomicI64::new(0));
    let g2 = Arc::clone(&gen);
    assert!(ControlledRealTimeReopenThread::start(
        Arc::clone(&m),
        Box::new(|| 0),
        Duration::from_millis(1),
        Duration::from_millis(2),
    )
    .is_err());
    let t = ControlledRealTimeReopenThread::start(
        Arc::clone(&m),
        Box::new(move || g2.load(Ordering::SeqCst)),
        Duration::from_secs(60),
        Duration::from_millis(5),
    )
    .unwrap();
    assert!(t.wait_for_generation(0, None), "already searchable");
    gen.store(5, Ordering::SeqCst);
    c.change.store(true, Ordering::SeqCst);
    assert!(t.wait_for_generation(5, Some(Duration::from_secs(30))));
    assert!(t.searching_gen() >= 5);
    assert_eq!(*m.acquire().unwrap(), 1, "the waiter forced a refresh");
    // A generation no refresh will reach times out.
    assert!(!t.wait_for_generation(99, Some(Duration::from_millis(30))));
    // Concurrent waiters.
    gen.store(7, Ordering::SeqCst);
    let t = Arc::new(t);
    let waiters: Vec<_> = (0..4)
        .map(|_| {
            let t = Arc::clone(&t);
            std::thread::spawn(move || t.wait_for_generation(7, Some(Duration::from_secs(30))))
        })
        .collect();
    for w in waiters {
        assert!(w.join().unwrap());
    }
    t.close().unwrap();
    assert_eq!(t.searching_gen(), i64::MAX);
    assert!(t.wait_for_generation(1000, None), "released after close");
}

#[test]
fn live_field_values_prefer_recent_writes() {
    // `TestLiveFieldValues`' intent: a value added or deleted since the
    // last refresh wins over the searcher's until a refresh covers it.
    let c = counter();
    let m = Arc::new(ReferenceManager::new(
        Arc::new(0i64),
        Box::new(Arc::clone(&c)),
    ));
    let lookups = Arc::new(AtomicUsize::new(0));
    let l2 = Arc::clone(&lookups);
    let live: LiveFieldValues<i64, String> = LiveFieldValues::new(
        Arc::clone(&m),
        Box::new(move |s: &i64, id: &str| {
            l2.fetch_add(1, Ordering::SeqCst);
            Ok((id == "indexed").then(|| format!("searcher {s}")))
        }),
    );
    live.add("a", "1".into());
    live.add("b", "2".into());
    live.delete("b");
    assert_eq!(live.get("a").unwrap().as_deref(), Some("1"));
    assert_eq!(live.get("b").unwrap(), None, "deleted");
    assert_eq!(live.get("indexed").unwrap().as_deref(), Some("searcher 0"));
    assert_eq!(live.size(), 2);
    c.change.store(true, Ordering::SeqCst);
    m.maybe_refresh_blocking().unwrap();
    assert_eq!(live.size(), 0, "the refresh made them searchable");
    assert_eq!(live.get("a").unwrap(), None, "now from the searcher");
    assert_eq!(live.get("indexed").unwrap().as_deref(), Some("searcher 1"));
    assert!(lookups.load(Ordering::SeqCst) >= 3);
    live.close();
    live.add("z", "9".into());
    m.maybe_refresh_blocking().unwrap();
    assert_eq!(live.size(), 1, "no longer listening");
}

#[test]
fn find_segments_file_retries_while_the_commit_generation_advances() {
    use lucene_store::byte_buffers_directory::ByteBuffersDirectory;
    use lucene_store::data_output::DataOutput;
    let dir = ByteBuffersDirectory::new();
    let commit = |name: &str| {
        let mut out = dir.create_output(name).unwrap();
        out.write_bytes(b"x");
        out.close().unwrap();
    };
    // No commit at all: `IndexNotFoundException`, the body never runs.
    let err = find_segments_file(&dir, || -> Result<()> { unreachable!() }).unwrap_err();
    assert!(matches!(err, Error::IllegalState(_)), "{err}");

    // A failure on an unchanged generation is reported, the first one.
    commit("segments_1");
    let mut calls = 0;
    let err = find_segments_file(&dir, || -> Result<()> {
        calls += 1;
        Err(Error::IllegalArgument(format!("try {calls}")))
    })
    .unwrap_err();
    assert_eq!(calls, 1);
    assert!(matches!(&err, Error::IllegalArgument(m) if m == "try 1"));

    // A commit landing while the body reads: the body runs again.
    let mut calls = 0;
    let got = find_segments_file(&dir, || {
        calls += 1;
        if calls == 1 {
            commit("segments_2");
            return Err(Error::IllegalArgument("deleted under us".into()));
        }
        Ok(calls)
    })
    .unwrap();
    assert_eq!(got, 2);
}

/// `new SearcherManager(writer, ...)`: a refresh makes what the writer
/// indexed searchable without a commit, and `isSearcherCurrent` follows
/// the writer rather than the directory's commits.
#[test]
fn a_searcher_manager_from_a_writer_refreshes_near_real_time() {
    use lucene_codecs::field_infos::{FieldInfo, IndexOptions};
    use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
    use lucene_index::concurrent_writer::ConcurrentIndexWriter;
    use lucene_index::index_writer::IndexWriter;
    use lucene_index::segment_info::LuceneVersion;

    let tmp = lucene_util::test_support::TempDir::new("searcher-manager-nrt");
    let dir: &'static lucene_store::FsDirectory =
        Box::leak(Box::new(lucene_store::FsDirectory::open(&tmp)));
    let fields = vec![FieldInfo {
        index_options: IndexOptions::Docs,
        omit_norms: true,
        ..FieldInfo::new("id", 0)
    }];
    let mut w = IndexWriter::open(
        dir,
        fields,
        "Lucene104",
        LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        },
    )
    .unwrap();
    w.set_postings_field(Some("id")).unwrap();
    let writer = Arc::new(ConcurrentIndexWriter::new(w, 2).unwrap());
    let doc = |id: &str| Document {
        fields: vec![StoredField {
            field_number: 0,
            value: FieldValue::String(id.to_string()),
        }],
    };
    writer.add_document(doc("a")).unwrap();
    let manager = SearcherManager::open_from_writer(Arc::clone(&writer) as _).unwrap();
    assert_eq!(manager.acquire().unwrap().num_docs(), 1);
    assert!(manager.is_searcher_current().unwrap());

    writer.add_document(doc("b")).unwrap();
    writer.add_document(doc("c")).unwrap();
    assert!(!manager.is_searcher_current().unwrap());
    assert!(manager.maybe_refresh().unwrap());
    let s = manager.acquire().unwrap();
    assert_eq!(s.num_docs(), 3);
    assert!(s.is_nrt());
    assert!(manager.is_searcher_current().unwrap());
    // Nothing was committed.
    assert!(lucene_index::segment_infos::read_latest(dir).is_err());
    assert!(manager.maybe_refresh().unwrap());
    assert_eq!(manager.acquire().unwrap().num_docs(), 3);
    manager.close().unwrap();
}
