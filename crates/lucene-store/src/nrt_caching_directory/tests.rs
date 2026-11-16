#![cfg(test)]
use super::*;
use crate::data_output::DataOutput;
use crate::FsDirectory;
use lucene_util::test_support::TempDir;

fn write_sized(dir: &NrtCachingDirectory<impl Directory>, name: &str, n: usize, est: Option<u64>) {
    let mut out = dir.create_output_with_estimate(name, est).unwrap();
    out.write_bytes(&vec![7u8; n]);
    out.close().unwrap();
}

/// Mirrors `TestNRTCachingDirectory.testNRTAndCommit`'s intent: small
/// flushed files are cached (invisible on disk) until a sync moves them,
/// and the cache size tracks exactly what is cached.
#[test]
fn small_writes_are_cached_until_synced() {
    let root = TempDir::new("nrt-cache");
    let dir = NrtCachingDirectory::new(FsDirectory::open(&root), 1.0, 2.0);
    write_sized(&dir, "_0.fdt", 100, Some(1000));
    write_sized(&dir, "_0.fdx", 50, Some(1000));
    assert_eq!(dir.list_cached_files(), vec!["_0.fdt", "_0.fdx"]);
    assert_eq!(dir.ram_bytes_used(), 150);
    assert!(!root.join("_0.fdt").exists());
    assert_eq!(dir.list_all().unwrap(), vec!["_0.fdt", "_0.fdx"]);
    assert_eq!(dir.file_length("_0.fdt").unwrap(), 100);
    assert_eq!(dir.open("_0.fdx").unwrap().len(), 50);

    dir.sync(&["_0.fdt".to_string()]).unwrap();
    assert_eq!(dir.list_cached_files(), vec!["_0.fdx"]);
    assert_eq!(dir.ram_bytes_used(), 50);
    assert_eq!(std::fs::read(root.join("_0.fdt")).unwrap(), vec![7u8; 100]);
    assert_eq!(dir.file_length("_0.fdt").unwrap(), 100);
    assert_eq!(dir.open("_0.fdt").unwrap().len(), 100);
    // Listing still shows each file once.
    assert_eq!(dir.list_all().unwrap(), vec!["_0.fdt", "_0.fdx"]);
}

/// The estimate reaches the cache through [`crate::EstimatedWrites`]
/// and every forwarding wrapper, as Java's `IOContext` reaches it through
/// `FilterDirectory`s: an `IndexWriter` flush of a small segment is
/// cached, a large merge is not, a plain write is not.
#[test]
fn estimates_reach_the_cache_through_wrappers() {
    use crate::{
        EstimatedWrites, LockFactory, LockValidatingDirectoryWrapper, NoLockFactory,
        SleepingLockWrapper,
    };
    let root = TempDir::new("nrt-estimated");
    let nrt = NrtCachingDirectory::new(FsDirectory::open(&root), 1.0, 2.0);
    let sleeping = SleepingLockWrapper::new(&nrt, 0).unwrap();
    let lock: Arc<dyn Lock> = Arc::from(NoLockFactory.obtain_lock(&nrt, "write.lock").unwrap());
    let validating = LockValidatingDirectoryWrapper::new(&sleeping, lock);
    let arc: Arc<dyn Directory + '_> = Arc::new(&validating);
    let flush = EstimatedWrites::new(&arc, 1000);
    assert_eq!(flush.estimated_bytes(), 1000);
    for name in ["_0.fdt", "_0.fdx"] {
        let mut out = flush.create_output(name).unwrap();
        out.write_bytes(b"abc");
        out.close().unwrap();
    }
    let mut out = flush.create_temp_output("_0", "sort").unwrap();
    out.write_bytes(b"t");
    let temp = out.name().to_string();
    out.close().unwrap();
    assert_eq!(
        nrt.list_cached_files(),
        vec!["_0.fdt".to_string(), "_0.fdx".to_string(), temp.clone()]
    );
    assert_eq!(flush.file_length("_0.fdt").unwrap(), 3);
    assert_eq!(flush.open("_0.fdx").unwrap().len(), 3);
    assert!(flush.list_all().unwrap().contains(&temp));
    flush.delete_file(&temp).unwrap();
    // An explicit estimate on the wrapper wins over its own.
    let mut out = flush
        .create_output_with_estimate("_big.fdt", Some(1 << 30))
        .unwrap();
    out.write_bytes(b"x");
    out.close().unwrap();
    let out = flush
        .create_temp_output_with_estimate("_big", "s", None)
        .unwrap();
    out.close().unwrap();
    assert!(root.join("_big.fdt").exists());
    // A large merge goes to the delegate.
    let merge = EstimatedWrites::new(&nrt, 64 * 1024 * 1024);
    let out = merge.create_output("_1.fdt").unwrap();
    out.close().unwrap();
    assert!(root.join("_1.fdt").exists());
    merge.sync(&["_0.fdt".to_string()]).unwrap();
    merge.sync_meta_data().unwrap();
    assert!(root.join("_0.fdt").exists());
    merge.rename("_0.fdt", "_2.fdt").unwrap();
    assert!(merge.pending_deletions().unwrap().is_empty());
    assert!(merge.fs_directory_path().is_none());
    assert!(merge.obtain_lock("x.lock").is_ok());
}

#[test]
fn virus_checker_control_unwraps_nrt_and_both_lock_wrappers() {
    use crate::directory::FsFaults;
    use crate::{LockValidatingDirectoryWrapper, SleepingLockWrapper};
    struct Virus(AtomicBool);
    impl FsFaults for Virus {
        fn check_delete(&self, _path: &Path) -> std::io::Result<()> {
            if self.0.load(Ordering::SeqCst) {
                Err(std::io::Error::other("virus checker refused delete"))
            } else {
                Ok(())
            }
        }
        fn check_rename(&self, _source: &Path, _dest: &Path) -> std::io::Result<()> {
            Ok(())
        }
        fn disable_virus_checker(&self) -> bool {
            self.0.swap(false, Ordering::SeqCst)
        }
    }
    let root = TempDir::new("nrt-virus-checker");
    let faults = Arc::new(Virus(AtomicBool::new(true)));
    let fs = FsDirectory::open(&root).with_fs_faults(faults.clone());
    let nrt = NrtCachingDirectory::new(fs, 1.0, 2.0);
    let mut cached = nrt.create_output_with_estimate("cached", Some(1)).unwrap();
    cached.write_bytes(b"x");
    cached.close().unwrap();
    nrt.create_output("orphan").unwrap().close().unwrap();
    assert!(faults.check_delete(&root.join("orphan")).is_err());
    faults
        .check_rename(&root.join("orphan"), &root.join("other"))
        .unwrap();
    let lock: Arc<dyn Lock> = Arc::from(nrt.obtain_lock("write.lock").unwrap());
    let validating = LockValidatingDirectoryWrapper::new(&nrt, lock.clone());
    let sleeping = SleepingLockWrapper::new(&validating, 0).unwrap();
    // TestUtil unwraps FilterDirectory without invoking their operations:
    // neither an invalid write lock nor cached files block this control.
    lock.close().unwrap();
    assert!(sleeping.disable_virus_checker());
    assert!(!sleeping.disable_virus_checker());
    assert_eq!(nrt.list_cached_files(), vec!["cached".to_string()]);
    assert!(!root.join("cached").exists());
    assert!(matches!(
        sleeping.delete_file("orphan"),
        Err(Error::AlreadyClosed(_))
    ));
    nrt.delete_file("orphan").unwrap();
    assert!(!root.join("orphan").exists());
    assert!(nrt.pending_deletions().unwrap().is_empty());
}

#[test]
fn large_or_unestimated_writes_go_to_the_delegate() {
    let root = TempDir::new("nrt-bypass");
    let dir = NrtCachingDirectory::new(FsDirectory::open(&root), 1.0, 2.0);
    // No context: never cached.
    let mut out = dir.create_output("_1.fdt").unwrap();
    out.write_bytes(b"x");
    out.close().unwrap();
    assert!(root.join("_1.fdt").exists());
    // A segment too large for maxMergeSizeMB.
    write_sized(&dir, "_2.fdt", 1, Some(2 * 1024 * 1024));
    assert!(root.join("_2.fdt").exists());
    // Would overflow maxCachedMB.
    write_sized(&dir, "_3.fdt", 1024 * 1024, Some(1024 * 1024));
    assert_eq!(dir.ram_bytes_used(), 1024 * 1024);
    assert!(!dir.do_cache_write("_4", Some(1024 * 1024 + 1)));
    write_sized(&dir, "_4.fdt", 1, Some(1024 * 1024 + 1));
    assert!(root.join("_4.fdt").exists());
    assert_eq!(dir.list_cached_files(), vec!["_3.fdt"]);
    assert!(!dir.do_cache_write("x", None));
}

#[test]
fn delete_rename_and_close_move_or_drop_cached_files() {
    let root = TempDir::new("nrt-lifecycle");
    let dir = NrtCachingDirectory::new(FsDirectory::open(&root), 1.0, 2.0);
    write_sized(&dir, "a", 10, Some(10));
    write_sized(&dir, "b", 20, Some(10));
    write_sized(&dir, "c", 30, Some(10));
    dir.delete_file("a").unwrap();
    assert_eq!(dir.ram_bytes_used(), 50);
    // Rename moves the source to the delegate first.
    dir.rename("b", "b2").unwrap();
    assert_eq!(std::fs::read(root.join("b2")).unwrap().len(), 20);
    assert_eq!(dir.ram_bytes_used(), 30);
    // A rename onto a cached name is refused.
    write_sized(&dir, "d", 1, None);
    let err = dir.rename("d", "c").unwrap_err();
    assert!(matches!(err, Error::IllegalArgument(_)), "{err}");
    // Deleting a file that is not cached reaches the delegate.
    dir.delete_file("d").unwrap();
    assert!(!root.join("d").exists());
    dir.sync_meta_data().unwrap();
    assert!(dir.pending_deletions().unwrap().is_empty());
    assert!(dir.fs_directory_path().is_none());
    assert_eq!(dir.delegate().directory(), &*root);

    dir.close().unwrap();
    dir.close().unwrap();
    assert!(dir.list_cached_files().is_empty());
    assert_eq!(dir.ram_bytes_used(), 0);
    assert_eq!(std::fs::read(root.join("c")).unwrap().len(), 30);
}

#[test]
fn drop_flushes_the_cache_to_the_delegate() {
    let root = TempDir::new("nrt-drop");
    {
        let dir = NrtCachingDirectory::new(FsDirectory::open(&root), 1.0, 2.0);
        write_sized(&dir, "_0.si", 5, Some(5));
        assert!(!root.join("_0.si").exists());
    }
    assert_eq!(std::fs::read(root.join("_0.si")).unwrap().len(), 5);
}

#[test]
fn a_deleted_open_output_does_not_count_against_the_cache() {
    let root = TempDir::new("nrt-deleted-open");
    let dir = NrtCachingDirectory::new(FsDirectory::open(&root), 1.0, 2.0);
    let mut out = dir.create_output_with_estimate("x", Some(1)).unwrap();
    out.write_bytes(b"12345");
    dir.delete_file("x").unwrap();
    out.close().unwrap();
    assert_eq!(dir.ram_bytes_used(), 0);
}

#[test]
fn temp_outputs_avoid_names_taken_in_the_other_directory() {
    let root = TempDir::new("nrt-temp");
    let dir = NrtCachingDirectory::new(FsDirectory::open(&root), 1.0, 2.0);
    // The cache holds the name the delegate will try first.
    write_sized(&dir, "_0_sort_0.tmp", 1, Some(1));
    let out = dir.create_temp_output("_0", "sort").unwrap();
    assert_eq!(out.name(), "_0_sort_1.tmp");
    out.close().unwrap();
    // The rejected attempt was removed from the delegate.
    assert!(!root.join("_0_sort_0.tmp").exists());
    assert!(root.join("_0_sort_1.tmp").exists());

    // Cached temp output: the delegate holds the cache's first name.
    std::fs::write(root.join("_1_s_0.tmp"), b"").unwrap();
    let out = dir
        .create_temp_output_with_estimate("_1", "s", Some(1))
        .unwrap();
    assert_eq!(out.name(), "_1_s_1.tmp");
    out.close().unwrap();
    assert!(dir.list_cached_files().contains(&"_1_s_1.tmp".to_string()));
    assert!(!dir.list_cached_files().contains(&"_1_s_0.tmp".to_string()));
}

#[test]
fn temp_output_errors_propagate() {
    let root = TempDir::new("nrt-temp-err");
    // Under a plain file, the directory cannot be created.
    std::fs::write(root.join("plain_file"), b"").unwrap();
    let missing = root.join("plain_file").join("missing");
    let dir = NrtCachingDirectory::new(FsDirectory::open(&missing), 1.0, 2.0);
    assert!(dir.create_temp_output("_0", "x").is_err());
}

#[test]
fn locks_come_from_the_delegate() {
    let root = TempDir::new("nrt-lock");
    let dir = NrtCachingDirectory::new(FsDirectory::open(&root), 1.0, 2.0);
    let lock = dir.obtain_lock("write.lock").unwrap();
    assert!(root.join("write.lock").exists());
    assert!(matches!(
        FsDirectory::open(&root).obtain_lock("write.lock"),
        Err(Error::LockObtainFailed(_))
    ));
    lock.close().unwrap();
}

#[test]
fn mb_to_bytes_is_javas_cast() {
    assert_eq!(mb_to_bytes(1.0), 1024 * 1024);
    assert_eq!(mb_to_bytes(-1.0), 0);
    assert_eq!(mb_to_bytes(f64::NAN), 0);
    assert_eq!(mb_to_bytes(f64::INFINITY), u64::MAX);
}
