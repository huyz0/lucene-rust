//! `BaseLockFactoryTestCase`: the lock contract every directory's lock
//! factory must keep -- exclusive while held, re-obtainable once released,
//! idempotent close, `ensureValid` true while held and false after -- plus
//! two concurrency checks.
//!
//! Differs:
//!
//! - `testStressLocks`' writer thread indexes through lucene-index's
//!   `IndexWriter` (field `content`, one `aaa` term per document, committed
//!   and dropped, which is Java's `close()`), and its searcher thread opens a
//!   `DirectoryReader` and every segment's readers in place of running a
//!   `TermQuery` (the open is what races the writer's deletes).
//! - Java's `assumeFalse(hasWindowsFS)` has nothing to skip: a lock suite's
//!   directory is the caller's, with no mock filesystem under it.

use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::{Barrier, Mutex};

use lucene_codecs::field_infos::{FieldInfo, IndexOptions};
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::merge_policy::upgrade::LATEST;
use lucene_search::directory_reader::DirectoryReader;
use lucene_store::{Directory, Error};
use lucene_util::test_support::TempDir;

use crate::util::lucene_test_case::LuceneTestCase;

/// `BaseLockFactoryTestCase`, over `get_directory(path)`: the directory to
/// test, at `path` when it is filesystem-based.
pub struct BaseLockFactoryTestCase<F> {
    get_directory: F,
}

impl<D: Directory, F: Fn(&Path) -> D> BaseLockFactoryTestCase<F> {
    /// The suite over `get_directory`.
    pub fn new(get_directory: F) -> Self {
        Self { get_directory }
    }

    /// `testBasics`.
    pub fn test_basics(&self, _tc: &mut LuceneTestCase) {
        let temp = TempDir::new("lock");
        let dir = (self.get_directory)(temp.path());
        let l = dir.obtain_lock("commit").unwrap();
        // shouldn't be able to get the lock twice
        assert!(matches!(
            dir.obtain_lock("commit"),
            Err(Error::LockObtainFailed(_))
        ));
        l.close().unwrap();
        // Make sure we can obtain first one again:
        let l = dir.obtain_lock("commit").unwrap();
        l.close().unwrap();
    }

    /// `testDoubleClose`.
    pub fn test_double_close(&self, _tc: &mut LuceneTestCase) {
        let temp = TempDir::new("lock");
        let dir = (self.get_directory)(temp.path());
        let l = dir.obtain_lock("commit").unwrap();
        l.close().unwrap();
        l.close().unwrap(); // close again, should be no exception
    }

    /// `testValidAfterAcquire`.
    pub fn test_valid_after_acquire(&self, _tc: &mut LuceneTestCase) {
        let temp = TempDir::new("lock");
        let dir = (self.get_directory)(temp.path());
        let l = dir.obtain_lock("commit").unwrap();
        l.ensure_valid().unwrap(); // no exception
        l.close().unwrap();
    }

    /// `testInvalidAfterClose`.
    pub fn test_invalid_after_close(&self, _tc: &mut LuceneTestCase) {
        let temp = TempDir::new("lock");
        let dir = (self.get_directory)(temp.path());
        let l = dir.obtain_lock("commit").unwrap();
        l.close().unwrap();
        assert!(matches!(l.ensure_valid(), Err(Error::AlreadyClosed(_))));
    }

    /// `testObtainConcurrently`: threads race for one lock; whoever holds it
    /// must be alone.
    pub fn test_obtain_concurrently(&self, tc: &mut LuceneTestCase) {
        let temp = TempDir::new("lock");
        let directory = (self.get_directory)(temp.path());
        let running = AtomicBool::new(true);
        let atomic_counter = AtomicI32::new(0);
        let asserting_lock = Mutex::new(());
        let num_threads = 2 + tc.random().next_int_bounded(10) as usize;
        let runs = tc.at_least(1000);
        let barrier = Barrier::new(num_threads);
        let obtained = AtomicI32::new(0);
        std::thread::scope(|s| {
            for _ in 0..num_threads {
                s.spawn(|| {
                    barrier.wait();
                    while running.load(Ordering::SeqCst) {
                        if let Ok(lock) = directory.obtain_lock("foo.lock") {
                            // Held by nobody else: `tryLock` must succeed.
                            let guard = asserting_lock.try_lock();
                            assert!(guard.is_ok(), "two holders of foo.lock");
                            obtained.fetch_add(1, Ordering::SeqCst);
                            drop(guard);
                            lock.close().unwrap();
                        }
                        if atomic_counter.fetch_add(1, Ordering::SeqCst) + 1 > runs {
                            running.store(false, Ordering::SeqCst);
                        }
                    }
                });
            }
        });
        assert!(obtained.load(Ordering::SeqCst) > 0);
    }

    /// `testStressLocks`: a writer thread appends one document at a time
    /// while a searcher thread opens readers, both over and over; neither
    /// may hit an error.
    pub fn test_stress_locks(&self, tc: &mut LuceneTestCase) {
        let temp = TempDir::new("lock");
        let dir = (self.get_directory)(temp.path());
        // First create a 1 doc index:
        {
            let mut w = open_writer(&dir).unwrap();
            add_doc(&mut w);
            w.commit().unwrap();
        }
        let (writer_hit, searcher_hit) = stress(&dir, tc.at_least(20));
        assert!(!writer_hit, "IndexWriter hit unexpected exceptions");
        assert!(!searcher_hit, "IndexSearcher hit unexpected exceptions");
    }
}

/// `testStressLocks`' two threads over `dir` for `num_iterations` rounds
/// each: whether the writer, then the searcher, hit an error.
fn stress(dir: &dyn Directory, num_iterations: i32) -> (bool, bool) {
    let writer_hit = AtomicBool::new(false);
    let searcher_hit = AtomicBool::new(false);
    std::thread::scope(|s| {
        s.spawn(|| {
            for i in 0..num_iterations {
                let mut writer = match open_writer(dir) {
                    Ok(writer) => writer,
                    Err(e) => {
                        println!("Stress Test Index Writer: creation hit unexpected exception: {e} (iter={i})");
                        writer_hit.store(true, Ordering::SeqCst);
                        break;
                    }
                };
                add_doc(&mut writer);
                if let Err(e) = writer.commit() {
                    println!("Stress Test Index Writer: close hit unexpected exception: {e}");
                    writer_hit.store(true, Ordering::SeqCst);
                    break;
                }
            }
        });
        s.spawn(|| {
            for _ in 0..num_iterations {
                let opened = DirectoryReader::open(dir).and_then(|reader| {
                    reader.open_segments()?;
                    Ok(reader.num_docs())
                });
                if let Err(e) = opened {
                    println!("Stress Test Index Searcher: create hit unexpected exception: {e}");
                    searcher_hit.store(true, Ordering::SeqCst);
                    break;
                }
            }
        });
    });
    (writer_hit.into_inner(), searcher_hit.into_inner())
}

/// An `IndexWriter` over `dir` with one indexed field, `content`: the
/// writer `testStressLocks` (and the mock directory's own tests) index
/// through until `RandomIndexWriter` is ported.
pub fn open_writer(dir: &dyn Directory) -> lucene_index::index_writer::Result<IndexWriter<'_>> {
    let content = FieldInfo {
        index_options: IndexOptions::DocsAndFreqsAndPositions,
        ..FieldInfo::new("content", 0)
    };
    let mut writer = IndexWriter::open(dir, vec![content], "Lucene104", LATEST)?;
    writer.set_postings_field(Some("content"))?;
    Ok(writer)
}

/// `addDoc(writer)`: `content:aaa`.
pub fn add_doc(writer: &mut IndexWriter<'_>) {
    writer
        .add_document(Document {
            fields: vec![StoredField {
                field_number: 0,
                value: FieldValue::String("aaa".to_string()),
            }],
        })
        .unwrap();
}

/// Declares one `#[test]` per [`BaseLockFactoryTestCase`] scenario over
/// `$suite` (an expression building the suite).
#[macro_export]
macro_rules! base_lock_factory_tests {
    ($suite:expr) => {
        $crate::base_lock_factory_tests!(@each $suite;
            test_basics, test_double_close, test_valid_after_acquire,
            test_invalid_after_close, test_obtain_concurrently, test_stress_locks);
    };
    (@each $suite:expr; $($name:ident),*) => {$(
        #[test]
        fn $name() {
            ($suite).$name(&mut $crate::util::lucene_test_case::LuceneTestCase::new(stringify!($name)));
        }
    )*};
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::mock_directory_wrapper::{
        FailOn, FailureContext, FailureOp, MockDirectoryWrapper,
    };
    use lucene_store::ByteBuffersDirectory;
    use lucene_util::java_random::JavaRandom;

    /// An index of one document under a mock directory failing `op`.
    fn failing(op: FailureOp) -> MockDirectoryWrapper {
        let dir = MockDirectoryWrapper::new(&mut JavaRandom::new(7), ByteBuffersDirectory::new());
        let mut w = open_writer(&dir).unwrap();
        add_doc(&mut w);
        w.commit().unwrap();
        drop(w);
        dir.fail_on(FailOn::new(move |ctx: &FailureContext<'_>| ctx.op == op));
        dir
    }

    // Gate rule #10: the stress check reports each thread's failure.
    #[test]
    fn the_stress_check_sees_a_failing_commit() {
        let dir = failing(FailureOp::CreateOutput);
        assert_eq!(stress(&dir, 3), (true, false));
    }

    #[test]
    fn the_stress_check_sees_failing_opens() {
        let dir = failing(FailureOp::OpenInput);
        assert_eq!(stress(&dir, 3), (true, true));
    }

    #[test]
    fn the_stress_check_passes_a_healthy_directory() {
        let dir = ByteBuffersDirectory::new();
        let mut w = open_writer(&dir).unwrap();
        add_doc(&mut w);
        w.commit().unwrap();
        drop(w);
        assert_eq!(stress(&dir, 3), (false, false));
    }
}
