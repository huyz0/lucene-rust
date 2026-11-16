#![cfg(test)]
use super::*;
use crate::{ByteBuffersDirectory, FsDirectory};
use lucene_util::test_support::TempDir;

#[test]
fn native_lock_excludes_a_second_holder_in_this_process() {
    let root = TempDir::new("native-lock");
    let dir = FsDirectory::open(&root);
    let lock = NativeFsLockFactory.obtain_lock(&dir, "write.lock").unwrap();
    lock.ensure_valid().unwrap();
    // The lock file exists, empty.
    assert_eq!(fs::metadata(root.join("write.lock")).unwrap().len(), 0);

    // A second directory instance over the same path: still excluded.
    let other = FsDirectory::open(&root);
    let err = NativeFsLockFactory
        .obtain_lock(&other, "write.lock")
        .unwrap_err();
    assert!(matches!(err, Error::LockObtainFailed(ref m) if m.contains("this process")));

    lock.close().unwrap();
    lock.close().unwrap();
    assert!(matches!(lock.ensure_valid(), Err(Error::AlreadyClosed(_))));
    // The file stays; the lock is free.
    assert!(root.join("write.lock").exists());
    let again = NativeFsLockFactory
        .obtain_lock(&other, "write.lock")
        .unwrap();
    drop(again);
    let _third = NativeFsLockFactory.obtain_lock(&dir, "write.lock").unwrap();
}

#[test]
fn native_lock_creates_a_missing_directory() {
    let root = TempDir::new("native-lock-mkdir");
    let nested = root.join("a").join("b");
    let lock = NativeFsLockFactory
        .obtain_fs_lock(&nested, "write.lock")
        .unwrap();
    assert!(nested.join("write.lock").is_file());
    assert!(format!("{lock:?}").starts_with("NativeFSLock(path="));
}

#[test]
fn native_lock_detects_external_tampering() {
    let root = TempDir::new("native-lock-tamper");
    let lock = NativeFsLockFactory
        .obtain_fs_lock(&root, "write.lock")
        .unwrap();
    // Someone writes into the lock file.
    fs::write(root.join("write.lock"), b"x").unwrap();
    let err = lock.ensure_valid().unwrap_err();
    assert!(
        err.to_string().contains("Unexpected lock file size"),
        "{err}"
    );
    lock.close().unwrap();

    // The lock itself is free again: the size only matters to a holder's
    // validity check, which the next holder fails straight away.
    let again = NativeFsLockFactory
        .obtain_fs_lock(&root, "write.lock")
        .unwrap();
    assert!(again.ensure_valid().is_err());
}

#[test]
fn native_lock_detects_a_deleted_lock_file() {
    let root = TempDir::new("native-lock-deleted");
    let lock = NativeFsLockFactory
        .obtain_fs_lock(&root, "write.lock")
        .unwrap();
    fs::remove_file(root.join("write.lock")).unwrap();
    assert!(matches!(lock.ensure_valid(), Err(Error::Io(_))));
    // Cleared from the map behind its back.
    let path = fs::canonicalize(&root).unwrap().join("write.lock");
    clear_lock_held(&path).unwrap();
    let err = lock.ensure_valid().unwrap_err();
    assert!(err.to_string().contains("cleared from map"), "{err}");
    // Closing reports the missing map entry, once.
    assert!(matches!(lock.close(), Err(Error::AlreadyClosed(_))));
    lock.close().unwrap();
}

#[test]
fn fs_factories_refuse_a_non_filesystem_directory() {
    let dir = ByteBuffersDirectory::new();
    for lf in [
        &NativeFsLockFactory as &dyn LockFactory,
        &SimpleFsLockFactory,
    ] {
        let err = lf.obtain_lock(&dir, "write.lock").unwrap_err();
        assert!(
            matches!(&err, Error::Io(e) if e.kind() == std::io::ErrorKind::Unsupported),
            "{err}"
        );
    }
    assert!(format!("{:?}", default_fs_lock_factory()).contains("NativeFsLockFactory"));
}

#[test]
fn native_lock_reports_an_unusable_lock_directory() {
    let root = TempDir::new("native-lock-file");
    let file = root.join("not-a-dir");
    fs::write(&file, b"").unwrap();
    assert!(matches!(
        NativeFsLockFactory.obtain_fs_lock(&file, "write.lock"),
        Err(Error::Io(_))
    ));
    // The lock name is itself a directory: it can be canonicalised, but
    // not opened for writing.
    fs::create_dir(root.join("dir.lock")).unwrap();
    assert!(matches!(
        NativeFsLockFactory.obtain_fs_lock(&root, "dir.lock"),
        Err(Error::Io(_))
    ));
    // ... and the failed attempt did not leave it marked as held.
    let path = fs::canonicalize(root.join("dir.lock")).unwrap();
    assert!(!is_lock_held(&path));
}

#[test]
fn simple_lock_is_the_lock_file() {
    let root = TempDir::new("simple-lock");
    let dir = FsDirectory::open(&root);
    let lock = SimpleFsLockFactory::INSTANCE
        .obtain_lock(&dir, "write.lock")
        .unwrap();
    lock.ensure_valid().unwrap();
    let err = SimpleFsLockFactory
        .obtain_lock(&dir, "write.lock")
        .unwrap_err();
    assert!(matches!(err, Error::LockObtainFailed(_)), "{err}");
    assert!(format!("{lock:?}").starts_with("SimpleFSLock(path="));
    lock.close().unwrap();
    assert!(!root.join("write.lock").exists());
    lock.close().unwrap();
    assert!(matches!(lock.ensure_valid(), Err(Error::AlreadyClosed(_))));

    // A stale file from a crashed holder keeps the directory locked.
    fs::write(root.join("write.lock"), b"").unwrap();
    assert!(matches!(
        SimpleFsLockFactory.obtain_lock(&dir, "write.lock"),
        Err(Error::LockObtainFailed(_))
    ));
}

#[test]
fn simple_lock_refuses_to_remove_a_file_it_no_longer_owns() {
    let root = TempDir::new("simple-lock-stolen");
    let lock = SimpleFsLockFactory
        .obtain_fs_lock(&root, "write.lock")
        .unwrap();
    fs::remove_file(root.join("write.lock")).unwrap();
    let err = lock.close().unwrap_err();
    assert!(matches!(err, Error::LockReleaseFailed(ref m) if m.contains("safely removed")));
    // Closed regardless.
    lock.close().unwrap();

    // Validation passes but removal fails: a "lock file" that is really
    // a directory cannot be removed as a file.
    let dir_lock = SimpleFsLock {
        path: root.to_path_buf(),
        creation_time: creation_time(&root).unwrap(),
        closed: AtomicBool::new(false),
    };
    let err = dir_lock.close().unwrap_err();
    assert!(matches!(err, Error::LockReleaseFailed(ref m) if m.contains("Unable to remove")));
    assert!(root.exists());

    // Creating the lock in a path that is a file is a plain I/O error.
    let file = root.join("f");
    fs::write(&file, b"").unwrap();
    assert!(matches!(
        SimpleFsLockFactory.obtain_fs_lock(&file, "write.lock"),
        Err(Error::Io(_))
    ));
}

/// A lock file deleted and created again behind a holder's back is a
/// different file: both factories' locks report the change. (Twenty
/// milliseconds apart, so the new file's time differs at any filesystem
/// timestamp granularity in use here.)
#[test]
fn a_lock_file_recreated_behind_the_holder_is_detected() {
    let root = TempDir::new("lock-recreated");
    let native = NativeFsLockFactory
        .obtain_fs_lock(&root, "native.lock")
        .unwrap();
    let simple = SimpleFsLockFactory
        .obtain_fs_lock(&root, "simple.lock")
        .unwrap();
    std::thread::sleep(std::time::Duration::from_millis(20));
    for name in ["native.lock", "simple.lock"] {
        fs::remove_file(root.join(name)).unwrap();
        fs::write(root.join(name), b"").unwrap();
    }
    for lock in [&native, &simple] {
        let err = lock.ensure_valid().unwrap_err();
        assert!(
            err.to_string().contains("changed by an external force"),
            "{err}"
        );
    }
    native.close().unwrap();
    // The recreated file is not the one this lock created: removing it
    // is refused.
    assert!(simple.close().is_err());
}

/// A lock name under a directory that does not exist: neither factory
/// can create the file, and the native one reports both failures.
#[test]
fn a_lock_under_a_missing_directory_is_an_io_error() {
    let root = TempDir::new("lock-missing-subdir");
    let err = NativeFsLockFactory
        .obtain_fs_lock(&root, "nosuch/write.lock")
        .unwrap_err();
    assert!(
        matches!(&err, Error::Io(e) if e.to_string().contains("creating the lock file failed too")),
        "{err}"
    );
    assert!(matches!(
        SimpleFsLockFactory.obtain_fs_lock(&root, "nosuch/write.lock"),
        Err(Error::Io(_))
    ));
}
