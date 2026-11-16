#![cfg(test)]
use super::*;
use crate::data_output::DataOutput;
use crate::{ByteBuffersDirectory, FsDirectory};
use lucene_util::test_support::TempDir;
use std::time::Instant;

#[test]
fn sleeping_wrapper_rejects_bad_arguments() {
    let dir = ByteBuffersDirectory::new();
    assert!(matches!(
        SleepingLockWrapper::new(&dir, -2),
        Err(Error::IllegalArgument(_))
    ));
    assert!(matches!(
        SleepingLockWrapper::with_poll_interval(&dir, 10, -1),
        Err(Error::IllegalArgument(_))
    ));
    let zero = SleepingLockWrapper::with_poll_interval(&dir, 10, 0).unwrap();
    assert!(matches!(
        zero.obtain_lock("write.lock"),
        Err(Error::IllegalArgument(_))
    ));
    SleepingLockWrapper::new(
        &dir,
        SleepingLockWrapper::<&ByteBuffersDirectory>::LOCK_OBTAIN_WAIT_FOREVER,
    )
    .unwrap();
}

/// Mirrors `TestSleepingLockWrapper`: a held lock times out after the
/// wait, and a lock released during the wait is obtained.
#[test]
fn sleeping_wrapper_retries_until_timeout() {
    let dir = ByteBuffersDirectory::new();
    let held = dir.obtain_lock("write.lock").unwrap();
    let sleeping = SleepingLockWrapper::with_poll_interval(&dir, 30, 10).unwrap();
    let started = Instant::now();
    let err = sleeping.obtain_lock("write.lock").unwrap_err();
    assert!(started.elapsed() >= Duration::from_millis(30));
    assert!(
        matches!(&err, Error::LockObtainFailed(m) if m.starts_with("Lock obtain timed out") && m.contains("already obtained")),
        "{err}"
    );

    std::thread::scope(|s| {
        s.spawn(|| {
            std::thread::sleep(Duration::from_millis(30));
            held.close().unwrap();
        });
        let lock = SleepingLockWrapper::with_poll_interval(&dir, 5_000, 5)
            .unwrap()
            .obtain_lock("write.lock")
            .unwrap();
        lock.ensure_valid().unwrap();
    });
}

#[test]
fn sleeping_wrapper_forwards_everything_else() {
    let root = TempDir::new("sleeping-wrapper");
    let dir = SleepingLockWrapper::new(FsDirectory::open(&root), 0).unwrap();
    let mut out = dir.create_output("a").unwrap();
    out.write_bytes(b"xyz");
    out.close().unwrap();
    dir.create_temp_output("_0", "t").unwrap().close().unwrap();
    dir.sync(&["a".to_string()]).unwrap();
    dir.rename("a", "b").unwrap();
    dir.sync_meta_data().unwrap();
    assert_eq!(&*dir.open("b").unwrap(), b"xyz");
    assert_eq!(dir.file_length("b").unwrap(), 3);
    assert_eq!(dir.list_all().unwrap(), vec!["_0_t_0.tmp", "b"]);
    dir.delete_file("b").unwrap();
    assert!(dir.pending_deletions().unwrap().is_empty());
    assert!(dir.fs_directory_path().is_none());
    assert_eq!(dir.delegate().directory(), root.path());
    // A timeout of 0 tries exactly once more after one poll.
    let held = dir.obtain_lock("write.lock").unwrap();
    assert!(matches!(
        SleepingLockWrapper::with_poll_interval(FsDirectory::open(&root), 0, 1)
            .unwrap()
            .obtain_lock("write.lock"),
        Err(Error::LockObtainFailed(_))
    ));
    drop(held);
}

const LOCK_WAIT_FOREVER_FOR_TEST: i64 = -1;

#[test]
fn sleeping_wrapper_does_not_retry_other_errors() {
    let root = TempDir::new("sleeping-wrapper-err");
    let dir = FsDirectory::with_lock_factory(
        root.join("f").join("g"),
        crate::fs_lock_factory::default_fs_lock_factory(),
    );
    // The lock directory cannot be created under a regular file.
    std::fs::write(root.join("f"), b"").unwrap();
    let sleeping = SleepingLockWrapper::new(dir, LOCK_WAIT_FOREVER_FOR_TEST).unwrap();
    assert!(matches!(
        sleeping.obtain_lock("write.lock"),
        Err(Error::Io(_))
    ));
}

/// A directory that records the names `close_input` sees.
struct RecordsCloses {
    inner: ByteBuffersDirectory,
    closed: std::sync::Mutex<Vec<String>>,
}

impl Directory for RecordsCloses {
    fn list_all(&self) -> Result<Vec<String>> {
        self.inner.list_all()
    }
    fn open(&self, name: &str) -> Result<Input> {
        self.inner.open(name)
    }
    fn open_aligned(&self, name: &str, alignment: usize) -> Result<Input> {
        self.inner.open_aligned(name, alignment)
    }
    fn create_output(&self, name: &str) -> Result<FsIndexOutput> {
        self.inner.create_output(name)
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
    fn close_input(&self, name: &str, input: Input) -> Result<()> {
        drop(input);
        self.closed.lock().unwrap().push(name.to_string());
        Ok(())
    }
}

#[test]
fn wrappers_forward_close_input_to_the_directory_beneath() {
    let base = RecordsCloses {
        inner: ByteBuffersDirectory::new(),
        closed: Default::default(),
    };
    let mut out = base.create_output("f.dat").unwrap();
    out.write_bytes(b"x");
    out.close().unwrap();
    let sleeping = SleepingLockWrapper::new(&base, 0).unwrap();
    sleeping
        .close_input("a", sleeping.open("f.dat").unwrap())
        .unwrap();
    let lock: Arc<dyn Lock> = Arc::from(base.obtain_lock("write.lock").unwrap());
    let validating = LockValidatingDirectoryWrapper::new(&base, lock);
    validating
        .close_input("b", validating.open("f.dat").unwrap())
        .unwrap();
    let switch = crate::FileSwitchDirectory::new(
        ["dat".to_string()].into_iter().collect(),
        &base,
        ByteBuffersDirectory::new(),
    )
    .unwrap();
    switch
        .close_input("c.dat", switch.open("f.dat").unwrap())
        .unwrap();
    assert_eq!(*base.closed.lock().unwrap(), ["a", "b", "c.dat"]);
}

#[test]
fn lock_validating_wrapper_checks_before_each_mutation() {
    let root = TempDir::new("lock-validating");
    let fs_dir = FsDirectory::open(&root);
    let lock: Arc<dyn Lock> = Arc::from(fs_dir.obtain_lock("write.lock").unwrap());
    let dir = LockValidatingDirectoryWrapper::new(&fs_dir, Arc::clone(&lock));
    let mut out = dir.create_output("a").unwrap();
    out.write_bytes(b"1");
    out.close().unwrap();
    dir.sync(&["a".to_string()]).unwrap();
    dir.rename("a", "b").unwrap();
    dir.sync_meta_data().unwrap();
    let src = ByteBuffersDirectory::new();
    let mut o = src.create_output("s").unwrap();
    o.write_bytes(b"src");
    o.close().unwrap();
    dir.copy_from(&src, "s", "c").unwrap();
    assert_eq!(&*dir.open("c").unwrap(), b"src");
    assert_eq!(dir.file_length("b").unwrap(), 1);
    dir.delete_file("b").unwrap();
    dir.create_temp_output("_0", "t").unwrap().close().unwrap();
    assert!(dir.pending_deletions().unwrap().is_empty());
    assert!(dir.fs_directory_path().is_none());
    assert!(dir.delegate().fs_directory_path().is_some());
    assert!(matches!(
        dir.obtain_lock("write.lock"),
        Err(Error::LockObtainFailed(_))
    ));
    assert!(dir.list_all().unwrap().contains(&"c".to_string()));

    // Once the lock is gone, every mutation refuses; reads still work.
    lock.close().unwrap();
    let closed = |r: Result<()>| assert!(matches!(r, Err(Error::AlreadyClosed(_))));
    closed(dir.create_output("x").map(|_| ()));
    closed(dir.delete_file("c"));
    closed(dir.rename("c", "d"));
    closed(dir.sync(&["c".to_string()]));
    closed(dir.sync_meta_data());
    closed(dir.copy_from(&src, "s", "e"));
    assert_eq!(&*dir.open("c").unwrap(), b"src");
}
