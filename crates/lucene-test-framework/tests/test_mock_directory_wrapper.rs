//! `TestMockDirectoryWrapper`: `BaseDirectoryTestCase` over the mock
//! directory, its own scenarios (disk full, a mock inside a mock, the
//! close-time corruption), and -- gate rule #10 -- every fault knob and
//! every check of the wrapper and the mock filesystem, each seen to fire.
//!
//! Not ported: `testAbuseClosedIndexInput` and the two clone variants -- a
//! closed (dropped) input cannot be read, by the borrow checker.

use std::collections::BTreeSet;
use std::io;
use std::path::Path;
use std::sync::Arc;

use lucene_store::directory::FsFaults;
use lucene_store::index_output::FsIndexOutput;
use lucene_store::lock::Lock;
use lucene_store::{
    ByteBuffersDirectory, DataInput, DataOutput, Directory, Error, FsDirectory, IndexOutput, Input,
    SliceInput,
};
use lucene_test_framework::base_directory_tests;
use lucene_test_framework::mockfile::mock_fs::{MockFs, MockFsConfig, EXTRA_FILE_NAME};
use lucene_test_framework::store::base_directory::BaseDirectoryTestCase;
use lucene_test_framework::store::base_directory_wrapper::BaseDirectoryWrapper;
use lucene_test_framework::store::base_lock_factory::{add_doc, open_writer};
use lucene_test_framework::store::mock_directory_wrapper::{
    fake_io_exception, is_fake_io_exception, is_injected_io_exception, FailOn, FailureContext,
    FailureOp, MockDirectoryWrapper, Throttling,
};
use lucene_test_framework::util::lucene_test_case::LuceneTestCase;
use lucene_util::java_random::JavaRandom;
use lucene_util::test_support::TempDir;

/// `TestMockDirectoryWrapper.getDirectory`: a mock over a random directory
/// or over the filesystem at `path`.
fn get_directory(path: &Path) -> MockDirectoryWrapper {
    let mut tc = LuceneTestCase::new("getDirectory");
    if tc.random().next_boolean() {
        tc.new_mock_directory()
    } else {
        tc.new_mock_fs_directory(path)
    }
}

mod base {
    use super::*;
    base_directory_tests!(
        BaseDirectoryTestCase::new(get_directory).with_fs_faults(|p, faults| {
            MockDirectoryWrapper::new(
                &mut JavaRandom::new(7),
                FsDirectory::open(p).with_fs_faults(faults),
            )
        }),
        slow_list_all = true
    );
}

fn mock(seed: i64) -> MockDirectoryWrapper {
    MockDirectoryWrapper::new(&mut JavaRandom::new(seed), ByteBuffersDirectory::new())
}

fn write(dir: &dyn Directory, name: &str, bytes: &[u8]) -> lucene_store::Result<()> {
    let mut out = dir.create_output(name)?;
    out.write_bytes(bytes);
    out.close().map(drop)
}

fn kind(e: &Error) -> Option<io::ErrorKind> {
    match e {
        Error::Io(e) => Some(e.kind()),
        _ => None,
    }
}

/// `testDiskFull`: the write past the limit fails; what fit is kept.
#[test]
fn test_disk_full() {
    let dir = mock(1);
    dir.set_max_size_in_bytes(3);
    assert_eq!(dir.max_size_in_bytes(), 3);
    assert!(format!("{dir:?}").contains("max=3"));
    write(&dir, "foo", &[1, 2]).unwrap(); // first write should succeed
    let e = write(&dir, "bar", &[1, 2]).unwrap_err();
    assert!(e.to_string().contains("fake disk full"), "{e}");
    assert!(e.to_string().contains("wrote 1 of 2 bytes"), "{e}");
    assert!(dir.max_used_size_in_bytes() >= 3);
    // A full disk with nothing left writes nothing.
    let e = write(&dir, "baz", &[1]).unwrap_err();
    assert!(!e.to_string().contains("wrote"), "{e}");
    dir.close().unwrap();
    assert_eq!(format!("{:?}", mock(1)), "MockDirectoryWrapper");
}

/// `testMDWinsideOfMDW`.
#[test]
fn test_mdw_inside_of_mdw() {
    let mut tc = LuceneTestCase::new("test_mdw_inside_of_mdw");
    let inner = tc.new_mock_directory();
    let dir = MockDirectoryWrapper::new(tc.random(), inner.clone());
    {
        let mut iw = open_writer(&dir).unwrap();
        for _ in 0..20 {
            add_doc(&mut iw);
        }
        iw.commit().unwrap();
    }
    dir.close().unwrap();
    inner.close().unwrap();
}

/// `testCorruptOnCloseIsWorking`: closing the mock crashes it, which
/// damages the file it wrote and never synced.
fn corrupt_on_close_is_working(dir: Arc<dyn Directory>) {
    let mut r = JavaRandom::new(42);
    {
        let wrapped = MockDirectoryWrapper::new(&mut r, Arc::clone(&dir));
        // otherwise MDW sometimes randomly leaves the file intact
        wrapped.set_always_corrupt(true);
        // MDW will only try to corrupt things if it sees an index:
        {
            let mut iw = open_writer(dir.as_ref()).unwrap();
            add_doc(&mut iw);
            iw.commit().unwrap();
        }
        // not sync'd!
        let mut out = wrapped.create_output("foo").unwrap();
        for i in 0..100 {
            out.write_i32(i);
        }
        out.close().unwrap();
        assert!(wrapped.unsynced_files().contains("foo"));
        // MDW.close now corrupts our unsync'd file (foo):
        wrapped.close().unwrap();
        assert!(wrapped.crashed());
    }
    let changed = match dir.open("foo") {
        Err(e) => {
            assert_eq!(kind(&e), Some(io::ErrorKind::NotFound), "{e}");
            true
        }
        Ok(input) => {
            let mut i = SliceInput::new(&input);
            (0..100).any(|n| i.read_i32().map_or(true, |x| x != n))
        }
    };
    assert!(
        changed,
        "MockDirectoryWrapper failed to corrupt an unsync'd file"
    );
}

#[test]
fn test_corrupt_on_close_is_working_fs_dir() {
    let temp = TempDir::new("corruptOnClose");
    corrupt_on_close_is_working(Arc::new(FsDirectory::open(temp.path())));
}

#[test]
fn test_corrupt_on_close_is_working_on_byte_buffers_directory() {
    corrupt_on_close_is_working(Arc::new(ByteBuffersDirectory::new()));
}

/// The close-time `CheckIndex` fires on a broken index, and is skipped
/// when turned off.
#[test]
fn close_runs_check_index_over_the_index() {
    let broken = |check: bool| {
        let dir = mock(2);
        dir.set_check_index_on_close(check);
        assert_eq!(dir.check_index_on_close(), check);
        write(&dir, "segments_1", b"not a commit").unwrap();
        dir.sync(&["segments_1".to_string()]).unwrap();
        let result = dir.close();
        assert!(!dir.is_open());
        assert!(dir.close().is_ok(), "a second close checks nothing");
        result
    };
    assert!(broken(true).is_err());
    broken(false).unwrap();
    assert!(mock(2).as_mock().is_some());
}

#[test]
fn close_fails_while_a_file_is_open() {
    let dir = mock(3);
    write(&dir, "a", b"x").unwrap();
    let input = dir.open("a").unwrap();
    assert_eq!(dir.file_handle_count(), 1);
    let e = dir.close().unwrap_err();
    assert!(e.contains("still 1 open files"), "{e}");
    drop(input);
    assert_eq!(dir.file_handle_count(), 0);
}

#[test]
fn random_io_exceptions_fail_the_first_write_and_sync() {
    let dir = mock(4);
    dir.set_random_io_exception_rate(1.0);
    assert_eq!(dir.random_io_exception_rate(), 1.0);
    // Java's random errors are plain `IOException`s, not `FakeIOException`s.
    let e = write(&dir, "a", b"abc").unwrap_err();
    assert!(
        is_injected_io_exception(&e) && !is_fake_io_exception(&e),
        "{e}"
    );
    let e = dir.sync(&["a".to_string()]).unwrap_err();
    assert!(
        is_injected_io_exception(&e) && !is_fake_io_exception(&e),
        "{e}"
    );
    dir.set_random_io_exception_rate(0.0);
    // Only the first write may fail: a second write of a clean output passes.
    let mut out = dir.create_output("b").unwrap();
    out.write_bytes(b"1");
    dir.set_random_io_exception_rate(1.0);
    out.write_bytes(b"2");
    out.close().unwrap();
    dir.set_random_io_exception_rate(0.0);
    dir.sync(&["b".to_string()]).unwrap();
    assert!(!dir.unsynced_files().contains("b"));
}

#[test]
fn random_io_exceptions_on_open_take_both_kinds() {
    let dir = mock(5);
    write(&dir, "a", b"abc").unwrap();
    dir.set_random_io_exception_rate_on_open(1.0);
    assert_eq!(dir.random_io_exception_rate_on_open(), 1.0);
    let mut kinds = BTreeSet::new();
    for _ in 0..64 {
        let e = dir.open("a").unwrap_err();
        assert!(is_injected_io_exception(&e), "{e}");
        kinds.insert(format!("{:?}", kind(&e)));
        assert!(dir.create_output("b").is_err());
        assert!(dir.create_temp_output("t", "x").is_err());
    }
    assert_eq!(kinds.len(), 2, "{kinds:?}");
    dir.set_allow_random_file_not_found_exception(false);
    for _ in 0..16 {
        let e = dir.open("a").unwrap_err();
        assert_eq!(kind(&e), Some(io::ErrorKind::Other), "{e}");
    }
}

/// Every failure point fires a registered failure, which sees the
/// operation and the name.
#[test]
fn a_failure_fires_at_every_failure_point() {
    let dir = mock(6);
    write(&dir, "a", b"abc").unwrap();
    let check = |op: FailureOp, run: &dyn Fn(&MockDirectoryWrapper) -> lucene_store::Result<()>| {
        let fail = FailOn::once(move |ctx: &FailureContext<'_>| ctx.op == op);
        dir.fail_on(fail.clone());
        let e = run(&dir).unwrap_err();
        assert!(
            is_fake_io_exception(&e) && is_injected_io_exception(&e),
            "{op:?}: {e}"
        );
        assert_eq!(fail.fired(), 1, "{op:?}");
        run(&dir).unwrap_or_else(|e| panic!("{op:?} after the failure disarmed: {e}"));
    };
    check(FailureOp::CreateOutput, &|d| {
        d.create_output("c1").map(drop)
    });
    check(FailureOp::CreateTempOutput, &|d| {
        d.create_temp_output("t", "x").map(drop)
    });
    let n = std::sync::atomic::AtomicUsize::new(0);
    check(FailureOp::Write, &|d| {
        write(
            d,
            &format!("w{}", n.fetch_add(1, std::sync::atomic::Ordering::SeqCst)),
            b"z",
        )
    });
    check(FailureOp::CloseOutput, &|d| {
        write(
            d,
            &format!("c{}", n.fetch_add(1, std::sync::atomic::Ordering::SeqCst)),
            b"z",
        )
    });
    check(FailureOp::OpenInput, &|d| d.open("a").map(drop));
    check(FailureOp::Sync, &|d| d.sync(&["a".to_string()]));
    check(FailureOp::SyncMetaData, &|d| d.sync_meta_data());
    let renamed = std::sync::atomic::AtomicBool::new(false);
    check(FailureOp::Rename, &|d| {
        let (from, to) = if renamed.load(std::sync::atomic::Ordering::SeqCst) {
            ("b", "a")
        } else {
            ("a", "b")
        };
        d.rename(from, to)?;
        renamed.store(true, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    });
    write(&dir, "d", b"").unwrap();
    check(FailureOp::DeleteFile, &|d| d.delete_file("d"));
    // An empty sync still evaluates the failures.
    let fail =
        FailOn::new(|ctx: &FailureContext<'_>| ctx.op == FailureOp::Sync && ctx.name.is_empty());
    dir.fail_on(fail.clone());
    assert!(dir.sync(&[]).is_err());
    fail.clear_do_fail();
    dir.sync(&[]).unwrap();
    fail.set_do_fail();
    assert!(dir.sync(&[]).is_err());
    assert_eq!(fail.fired(), 2);
    assert!(is_fake_io_exception(&Error::Io(fake_io_exception("x"))));
}

/// `setFailOnCreateOutput(false)`/`setFailOnOpenInput(false)` drop the
/// second evaluation, so a failure that skips its first chance never fires.
#[test]
fn fail_on_create_output_and_open_input_gate_the_second_evaluation() {
    let dir = mock(7);
    write(&dir, "a", b"abc").unwrap();
    let second_only = |op: FailureOp| {
        let seen = std::sync::atomic::AtomicUsize::new(0);
        FailOn::new(move |ctx: &FailureContext<'_>| {
            ctx.op == op && seen.fetch_add(1, std::sync::atomic::Ordering::SeqCst) % 2 == 1
        })
    };
    dir.fail_on(second_only(FailureOp::CreateOutput));
    dir.fail_on(second_only(FailureOp::OpenInput));
    assert!(dir.create_output("b").is_err());
    assert!(dir.open("a").is_err());
    dir.set_fail_on_create_output(false);
    dir.set_fail_on_open_input(false);
    dir.create_output("b").unwrap().close().unwrap();
    drop(dir.open("a").unwrap());
}

#[test]
fn a_crash_fails_every_mutation_until_cleared() {
    let dir = mock(8);
    write(&dir, "a", b"abc").unwrap();
    dir.sync(&["a".to_string()]).unwrap();
    let mut open = dir.create_output("open").unwrap();
    open.write_bytes(b"before");
    dir.crash().unwrap();
    assert!(dir.crashed());
    assert_eq!(dir.file_handle_count(), 0, "crash forgets every handle");
    // The output still open fails its writes and its close.
    open.write_bytes(b"after");
    assert!(open
        .close()
        .unwrap_err()
        .to_string()
        .contains("has crashed"));
    let mut open = dir.create_output("x").unwrap_err().to_string();
    assert!(open.contains("after crash"), "{open}");
    open = dir.create_temp_output("t", "x").unwrap_err().to_string();
    assert!(open.contains("after crash"), "{open}");
    for e in [
        dir.sync(&["a".to_string()]).unwrap_err(),
        dir.rename("a", "b").unwrap_err(),
        dir.delete_file("a").unwrap_err(),
        dir.sync_meta_data().unwrap_err(),
    ] {
        assert!(e.to_string().contains("after crash"), "{e}");
    }
    // Reading still works; the synced file is intact.
    assert_eq!(&*dir.open("a").unwrap(), b"abc");
    dir.clear_crash();
    assert!(!dir.crashed());
    dir.rename("a", "b").unwrap();
}

/// A crash with an output still open, over `ByteBuffersDirectory` (which
/// cannot read a file still being written): whatever fate each seed draws,
/// the crash succeeds, leaves no listed file unreadable, and the stale
/// output stays dead after `clear_crash`.
#[test]
fn a_crash_with_an_output_still_open_damages_cleanly_on_every_seed() {
    for seed in 0..64 {
        let dir = mock(seed);
        let mut open = dir.create_output("open").unwrap();
        open.write_bytes(b"unfinished");
        write(&dir, "done", b"finished").unwrap();
        dir.crash().unwrap_or_else(|e| panic!("seed {seed}: {e}"));
        assert!(!dir.list_all().unwrap().contains(&"open".to_string()));
        for name in dir.list_all().unwrap() {
            dir.open(&name)
                .unwrap_or_else(|e| panic!("seed {seed}: {name}: {e}"));
        }
        dir.clear_crash();
        open.write_bytes(b"after");
        assert!(open.close().is_err(), "seed {seed}");
        assert!(!dir.list_all().unwrap().contains(&"open".to_string()));
    }
}

/// Out of many unsynced files, a crash gives each one of the six fates;
/// with `alwaysCorrupt` none survives intact.
#[test]
fn a_crash_deals_every_kind_of_damage() {
    let original: Vec<u8> = (0..64u8).map(|b| b | 0x40).collect();
    let fates = |always: bool| {
        let delegate = Arc::new(ByteBuffersDirectory::new());
        let dir = MockDirectoryWrapper::new(&mut JavaRandom::new(9), Arc::clone(&delegate));
        dir.set_always_corrupt(always);
        for i in 0..120 {
            write(&dir, &format!("f{i}"), &original).unwrap();
        }
        dir.crash().unwrap();
        assert!(dir.unsynced_files().is_empty());
        let mut fates = BTreeSet::new();
        for i in 0..120 {
            let fate = match delegate.open(&format!("f{i}")) {
                Err(_) => "deleted",
                Ok(bytes) if bytes.is_empty() => "truncated",
                Ok(bytes) if bytes.len() == original.len() / 2 => {
                    assert_eq!(&bytes[..], &original[..32]);
                    "half"
                }
                Ok(bytes) if bytes.iter().all(|&b| b == 0) => "zeroed",
                Ok(bytes) if bytes[..] == original[..] => "intact",
                Ok(bytes) => {
                    let flipped: u32 = bytes
                        .iter()
                        .zip(&original)
                        .map(|(a, b)| (a ^ b).count_ones())
                        .sum();
                    assert_eq!(flipped, 1);
                    "bit"
                }
            };
            fates.insert(fate);
        }
        // No temp file is left behind.
        assert!(delegate
            .list_all()
            .unwrap()
            .iter()
            .all(|f| f.starts_with('f')));
        fates
    };
    let all = fates(false);
    assert_eq!(all.len(), 6, "{all:?}");
    let always = fates(true);
    assert!(!always.contains("intact"), "{always:?}");
}

#[test]
fn a_file_is_written_once() {
    let dir = mock(10);
    write(&dir, "a", b"x").unwrap();
    let e = dir.create_output("a").unwrap_err();
    assert_eq!(kind(&e), Some(io::ErrorKind::AlreadyExists), "{e}");
    dir.delete_file("a").unwrap();
    write(&dir, "a", b"y").unwrap();
    dir.rename("a", "b").unwrap();
    write(&dir, "a", b"z").unwrap();
    assert!(dir.create_output("b").is_err());
}

#[test]
fn a_file_open_for_writing_cannot_be_read() {
    let dir = mock(11);
    let mut out = dir.create_output("a").unwrap();
    out.write_bytes(b"x");
    let e = dir.open("a").unwrap_err();
    assert_eq!(kind(&e), Some(io::ErrorKind::PermissionDenied), "{e}");
    dir.set_allow_reading_files_still_open_for_write(true);
    // The wrapper no longer refuses; whatever comes back is the delegate's
    // (`ByteBuffersDirectory` refuses an unfinished file itself, as Java's
    // does).
    if let Err(e) = dir.open("a") {
        assert!(!e.to_string().contains("MockDirectoryWrapper"), "{e}");
    }
    out.close().unwrap();
    let e = dir.open("missing").unwrap_err();
    assert_eq!(kind(&e), Some(io::ErrorKind::NotFound), "{e}");
}

#[test]
fn assert_no_delete_open_file_refuses_deleting_renaming_and_overwriting() {
    let delegate = Arc::new(ByteBuffersDirectory::new());
    let dir = MockDirectoryWrapper::new(&mut JavaRandom::new(12), Arc::clone(&delegate));
    write(&dir, "a", b"x").unwrap();
    write(&dir, "c", b"x").unwrap();
    write(delegate.as_ref(), "outside", b"x").unwrap();
    let a = dir.open("a").unwrap();
    let outside = dir.open("outside").unwrap();
    // Off: the delete goes through and is recorded.
    assert!(!dir.assert_no_delete_open_file());
    dir.delete_file("a").unwrap();
    assert_eq!(dir.open_deleted_files(), BTreeSet::from(["a".to_string()]));
    drop(a);
    let c = dir.open("c").unwrap();
    dir.set_assert_no_delete_open_file(true);
    for e in [
        dir.delete_file("c").unwrap_err(),
        dir.rename("c", "d").unwrap_err(),
        dir.rename("outside2", "c").unwrap_err(),
        dir.create_output("outside").unwrap_err(),
    ] {
        assert!(e.to_string().contains("is still open"), "{e}");
    }
    drop((c, outside));
    dir.delete_file("c").unwrap();
    // Java's `removeOpenFile` leaves the record; only a later delete of
    // the closed name clears it.
    assert_eq!(dir.open_deleted_files(), BTreeSet::from(["a".to_string()]));
    assert!(dir.delete_file("a").is_err());
    assert!(dir.open_deleted_files().is_empty());
}

/// A mock with the given mock-filesystem behaviours over a temp dir.
fn over_fs(config: MockFsConfig) -> (MockDirectoryWrapper, Arc<MockFs>, TempDir) {
    let temp = TempDir::new("mockfs");
    let fs = MockFs::new(config);
    fs.create_directory(temp.path()).unwrap();
    let faults: Arc<dyn FsFaults> = fs.clone();
    let dir = MockDirectoryWrapper::with_mock_fs(
        &mut JavaRandom::new(13),
        FsDirectory::open(temp.path()).with_fs_faults(faults),
        Arc::clone(&fs),
    );
    (dir, fs, temp)
}

#[test]
fn the_handle_limit_refuses_a_handle_past_it() {
    let (dir, fs, _temp) = over_fs(MockFsConfig {
        handle_limit: Some(2),
        ..MockFsConfig::default()
    });
    write(&dir, "a", b"x").unwrap();
    let (one, two) = (dir.open("a").unwrap(), dir.open("a").unwrap());
    assert_eq!(fs.open_handles().get("a"), Some(&2));
    let e = dir.open("a").unwrap_err();
    assert!(e.to_string().contains("Too many open files"), "{e}");
    // `HandleLimitFS` refuses after the file exists, and the wrapper has
    // already recorded the name: both stay, and a retry is refused.
    let e = dir.create_output("b").unwrap_err();
    assert!(e.to_string().contains("Too many open files"), "{e}");
    assert!(dir.list_all().unwrap().contains(&"b".to_string()));
    assert!(dir.unsynced_files().contains("b"));
    drop((one, two));
    drop(dir.open("a").unwrap());
    let e = dir.create_output("b").unwrap_err();
    assert_eq!(kind(&e), Some(io::ErrorKind::AlreadyExists), "{e}");
}

/// `LeakFS`: a handle still open when the filesystem is checked is a leak.
#[test]
fn leaks_are_reported() {
    let (dir, fs, _temp) = over_fs(MockFsConfig {
        leak: true,
        ..MockFsConfig::default()
    });
    write(&dir, "a", b"x").unwrap();
    let input = dir.open("a").unwrap();
    assert!(fs.check_leaks().unwrap_err().contains("\"a\""));
    drop(input);
    fs.check_leaks().unwrap();
    // A handle on the filesystem the wrapper did not hand out fails close.
    fs.on_open("b").unwrap();
    let e = dir.close().unwrap_err();
    assert!(e.contains("file handle leaks"), "{e}");
    fs.on_close("b");
    dir.close().unwrap();
}

#[test]
fn windows_refuses_deleting_an_open_file_which_becomes_pending() {
    let (dir, fs, _temp) = over_fs(MockFsConfig {
        windows: true,
        ..MockFsConfig::default()
    });
    write(&dir, "a", b"x").unwrap();
    let input = dir.open("a").unwrap();
    assert_eq!(fs.open_handles().get("a"), Some(&1));
    dir.delete_file("a").unwrap();
    assert!(dir.pending_deletions().unwrap().contains("a"));
    drop(input);
    assert!(fs.open_handles().is_empty());
}

#[test]
fn the_virus_checker_holds_some_deletes_and_corrupting_turns_it_off() {
    let mut r = JavaRandom::new(16);
    let (dir, fs, _temp) = over_fs(MockFsConfig::default().with_virus_checker(&mut r));
    assert!(fs.has_virus_checker());
    let mut i = 0;
    while dir.pending_deletions().unwrap().is_empty() {
        let name = format!("v{i}");
        write(&dir, &name, b"x").unwrap();
        dir.delete_file(&name).unwrap();
        i += 1;
        assert!(i < 1000, "the virus checker never struck");
    }
    // A crash's damage deletes with the checker off: nothing more pends
    // (and the retry of the held delete now goes through).
    let pending = dir.pending_deletions().unwrap().len();
    dir.set_always_corrupt(true);
    for n in 0..30 {
        write(&dir, &format!("u{n}"), b"abcdef").unwrap();
    }
    dir.crash().unwrap();
    assert!(dir.pending_deletions().unwrap().len() <= pending);
    assert!(fs.disable_virus_checker(), "re-enabled after the crash");
}

#[test]
fn extras_add_a_file_the_size_ignores() {
    let (dir, _fs, temp) = over_fs(MockFsConfig {
        extras: Some(false),
        ..MockFsConfig::default()
    });
    assert!(temp.path().join(EXTRA_FILE_NAME).exists());
    std::fs::write(temp.path().join(EXTRA_FILE_NAME), b"0123456789").unwrap();
    assert!(dir
        .list_all()
        .unwrap()
        .contains(&EXTRA_FILE_NAME.to_string()));
    write(&dir, "a", b"abc").unwrap();
    assert_eq!(dir.size_in_bytes().unwrap(), 3);
    dir.set_track_disk_usage(true);
    write(&dir, "b", b"abcd").unwrap();
    assert_eq!(dir.max_used_size_in_bytes(), 7);
    std::fs::remove_file(temp.path().join("b")).unwrap();
    dir.reset_max_used_size_in_bytes().unwrap();
    assert_eq!(dir.max_used_size_in_bytes(), 3);
}

#[test]
fn disabled_fsync_only_checks_the_file_exists() {
    let (dir, fs, temp) = over_fs(MockFsConfig {
        disable_fsync: true,
        ..MockFsConfig::default()
    });
    assert!(fs.fsync_disabled());
    assert!(dir.mock_fs().is_some_and(|fs| fs.fsync_disabled()));
    write(&dir, "a", b"x").unwrap();
    dir.sync(&["a".to_string()]).unwrap();
    std::fs::remove_file(temp.path().join("a")).unwrap();
    let e = dir.sync(&["a".to_string()]).unwrap_err();
    assert_eq!(kind(&e), Some(io::ErrorKind::NotFound), "{e}");
    dir.sync_meta_data().unwrap();
    // With fsync on, the delegate syncs.
    let (dir, _fs, _temp) = over_fs(MockFsConfig::default());
    write(&dir, "a", b"x").unwrap();
    dir.sync(&["a".to_string()]).unwrap();
    dir.sync_meta_data().unwrap();
}

#[test]
fn throttling_slows_outputs() {
    let dir = mock(18);
    dir.set_throttling(Throttling::Always);
    // No wall-clock bound: Java's `getDelay` reads a delegate faster than
    // 1 ns/byte as not worth slowing (see the unit test on `Throttle`), so
    // whether 100 kB sleeps depends on the machine. The count is what fires.
    write(&dir, "a", &vec![7u8; 100_000]).unwrap();
    for i in 0..4 {
        let mut out = dir.create_output(&format!("small{i}")).unwrap();
        for _ in 0..600 {
            out.write_bytes(&[1, 2, 3]);
        }
        out.close().unwrap();
    }
    assert_eq!(dir.slow_and_throttled_counts().2, 5);
    dir.set_throttling(Throttling::Sometimes);
    let mut n = 0;
    while dir.slow_and_throttled_counts().2 == 5 {
        write(&dir, &format!("s{n}"), b"x").unwrap();
        n += 1;
        assert!(n < 5000);
    }
    dir.set_throttling(Throttling::Never);
    write(&dir, "never", b"x").unwrap();
}

#[test]
fn slow_open_closers_fire() {
    let dir = mock(19);
    write(&dir, "a", b"x").unwrap();
    dir.set_use_slow_open_closers(true);
    let slow = std::time::Duration::from_millis(50);
    let mut n = 0;
    loop {
        let (opens_before, closes_before, _) = dir.slow_and_throttled_counts();
        let started = std::time::Instant::now();
        let input = dir.open("a").unwrap();
        let opened = started.elapsed();
        let (opens, _, _) = dir.slow_and_throttled_counts();
        let started = std::time::Instant::now();
        drop(input);
        let closed = started.elapsed();
        let (_, closes, _) = dir.slow_and_throttled_counts();
        // A counted slow open or close really slept.
        if opens > opens_before {
            assert!(opened >= slow, "{opened:?}");
        }
        if closes > closes_before {
            assert!(closed >= slow, "{closed:?}");
        }
        if opens > 0 && closes > 0 {
            break;
        }
        n += 1;
        assert!(n < 20_000);
    }
}

/// A delegate whose temp outputs are not named `*.tmp`.
struct BadTempNames(ByteBuffersDirectory);

impl Directory for BadTempNames {
    fn list_all(&self) -> lucene_store::Result<Vec<String>> {
        self.0.list_all()
    }
    fn open(&self, name: &str) -> lucene_store::Result<Input> {
        self.0.open(name)
    }
    fn file_length(&self, name: &str) -> lucene_store::Result<u64> {
        self.0.file_length(name)
    }
    fn create_output(&self, name: &str) -> lucene_store::Result<FsIndexOutput> {
        self.0.create_output(name)
    }
    fn create_temp_output(
        &self,
        prefix: &str,
        _suffix: &str,
    ) -> lucene_store::Result<FsIndexOutput> {
        self.0.create_output(prefix)
    }
    fn sync(&self, names: &[String]) -> lucene_store::Result<()> {
        self.0.sync(names)
    }
    fn rename(&self, source: &str, dest: &str) -> lucene_store::Result<()> {
        self.0.rename(source, dest)
    }
    fn delete_file(&self, name: &str) -> lucene_store::Result<()> {
        self.0.delete_file(name)
    }
    fn sync_meta_data(&self) -> lucene_store::Result<()> {
        self.0.sync_meta_data()
    }
    fn obtain_lock(&self, name: &str) -> lucene_store::Result<Box<dyn Lock>> {
        self.0.obtain_lock(name)
    }
}

#[test]
fn a_temp_output_must_be_named_tmp() {
    let dir = MockDirectoryWrapper::new(
        &mut JavaRandom::new(20),
        BadTempNames(ByteBuffersDirectory::new()),
    );
    let e = dir.create_temp_output("plain", "x").unwrap_err();
    assert!(
        e.to_string().contains("failed to use .tmp extension"),
        "{e}"
    );
    assert!(dir
        .delegate()
        .list_all()
        .unwrap()
        .contains(&"plain".to_string()));
    drop(dir.obtain_lock("write.lock").unwrap());
    assert!(dir.fs_directory_path().is_none());
    assert_eq!(dir.file_length("plain").unwrap(), 0);
}

#[test]
fn new_directory_draws_every_shape() {
    let mut raw = 0;
    let mut mocks = 0;
    let mut fs = 0;
    for seed in 0..400 {
        let mut tc = LuceneTestCase::with_config("shapes", seed, false, 1);
        let dir = tc.new_directory();
        if dir.as_mock().is_some() {
            mocks += 1;
        } else {
            raw += 1;
        }
        if dir.fs_directory_path().is_some() {
            fs += 1;
        }
        write(dir.as_ref(), "a", b"x").unwrap();
        assert_eq!(&*dir.open("a").unwrap(), b"x");
        dir.close().unwrap();
    }
    assert!(
        raw > 0 && mocks > 0 && fs > 0,
        "raw={raw} mocks={mocks} fs={fs}"
    );
}

#[test]
fn new_fs_and_virus_checking_directories() {
    let mut virus = 0;
    for seed in 0..60 {
        let mut tc = LuceneTestCase::with_config("fsdirs", seed, false, 1);
        let temp = TempDir::new("newfs");
        let dir = tc.new_fs_directory(temp.path());
        assert_eq!(dir.fs_directory_path(), Some(temp.path()));
        write(dir.as_ref(), "a", b"x").unwrap();
        dir.close().unwrap();
        let dir = tc.new_maybe_virus_checking_directory();
        if dir
            .as_mock()
            .and_then(|m| m.mock_fs())
            .is_some_and(|fs| fs.has_virus_checker())
        {
            virus += 1;
        }
        write(dir.as_ref(), "a", b"x").unwrap();
        dir.close().unwrap();
    }
    assert!(virus > 0);
}

#[test]
fn new_directory_from_copies_the_index_files() {
    let mut tc = LuceneTestCase::with_config("copy", 21, false, 1);
    let source = ByteBuffersDirectory::new();
    for name in [
        "segments_1",
        "_0.cfs",
        "_0_Lucene90_0.dvd",
        "write.lock",
        "notes.txt",
    ] {
        write(&source, name, name.as_bytes()).unwrap();
    }
    let dir = tc.new_directory_from(&source).unwrap();
    let mut files = dir.list_all().unwrap();
    files.retain(|f| !MockFs::is_extra(f));
    assert_eq!(files, ["_0.cfs", "_0_Lucene90_0.dvd", "segments_1"]);
    dir.set_check_index_on_close(false);
    dir.close().unwrap();
}

#[test]
fn copy_from_and_lengths_forward() {
    let dir = mock(22);
    let other = ByteBuffersDirectory::new();
    write(&other, "src", b"hello").unwrap();
    dir.copy_from(&other, "src", "dst").unwrap();
    assert_eq!(dir.file_length("dst").unwrap(), 5);
    let mut out = dir.create_output("x").unwrap();
    out.write_bytes(b"abc");
    assert_eq!(out.name(), "x");
    out.close().unwrap();
    assert!(dir.pending_deletions().unwrap().is_empty());
}
