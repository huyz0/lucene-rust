#![cfg(test)]
use super::*;
use crate::data_output::DataOutput;
use crate::index_output::IndexOutput;
use crate::{ByteBuffersDirectory, FsDirectory};
use lucene_util::test_support::TempDir;

fn exts(e: &[&str]) -> HashSet<String> {
    e.iter().map(|s| s.to_string()).collect()
}

fn write(dir: &dyn Directory, name: &str, bytes: &[u8]) {
    let mut out = dir.create_output(name).unwrap();
    out.write_bytes(bytes);
    out.close().unwrap();
}

#[test]
fn get_extension_matches_java() {
    assert_eq!(get_extension("_0.fdt"), "fdt");
    assert_eq!(get_extension("segments_1"), "");
    assert_eq!(get_extension("write.lock"), "lock");
    assert_eq!(get_extension("a.b.c"), "c");
    assert_eq!(get_extension("trailing."), "");
    // Temp files route by the extension inside them.
    assert_eq!(get_extension("_0.fdt_sort_1.tmp"), "fdt");
    assert_eq!(get_extension("_0_sort_1.tmp"), "tmp");
    assert_eq!(get_extension(".1.tmp"), "tmp");
    assert_eq!(get_extension("x.1a.tim.tmp"), "tim");
}

/// A size estimate reaches the side a file routes to (Java passes the
/// `IOContext` through), here an `NrtCachingDirectory` that caches it.
#[test]
fn estimates_route_with_the_file() {
    let nrt = crate::NrtCachingDirectory::new(ByteBuffersDirectory::new(), 1.0, 2.0);
    let dir = FileSwitchDirectory::new(exts(&["fdt"]), &nrt, ByteBuffersDirectory::new()).unwrap();
    let out = dir.create_output_with_estimate("_0.fdt", Some(10)).unwrap();
    out.close().unwrap();
    let out = dir
        .create_temp_output_with_estimate("_0.fdt", "sort", Some(10))
        .unwrap();
    let temp = out.name().to_string();
    out.close().unwrap();
    assert_eq!(nrt.list_cached_files(), vec!["_0.fdt".to_string(), temp]);
}

#[test]
fn tmp_is_a_reserved_extension() {
    let err = FileSwitchDirectory::new(
        exts(&["tmp"]),
        ByteBuffersDirectory::new(),
        ByteBuffersDirectory::new(),
    )
    .err()
    .unwrap();
    assert!(matches!(err, Error::IllegalArgument(_)));
}

/// Mirrors `TestFileSwitchDirectory.testBasic`: files land in the
/// directory their extension picks, and the union lists both.
#[test]
fn files_route_by_extension() {
    let dir = FileSwitchDirectory::new(
        exts(&["fdt", "fdx"]),
        ByteBuffersDirectory::new(),
        ByteBuffersDirectory::new(),
    )
    .unwrap();
    write(&dir, "_0.fdt", b"stored");
    write(&dir, "_0.fdx", b"index");
    write(&dir, "_0.tim", b"terms");
    write(&dir, "segments_1", b"commit");
    assert_eq!(
        dir.primary_dir().list_all().unwrap(),
        vec!["_0.fdt", "_0.fdx"]
    );
    assert_eq!(
        dir.secondary_dir().list_all().unwrap(),
        vec!["_0.tim", "segments_1"]
    );
    assert_eq!(
        dir.list_all().unwrap(),
        vec!["_0.fdt", "_0.fdx", "_0.tim", "segments_1"]
    );
    assert_eq!(&*dir.open("_0.fdt").unwrap(), b"stored");
    assert_eq!(dir.file_length("_0.tim").unwrap(), 5);
    dir.sync(&["_0.fdt".to_string(), "_0.tim".to_string()])
        .unwrap();
    dir.sync_meta_data().unwrap();
    dir.delete_file("_0.fdx").unwrap();
    assert!(!dir.primary_dir().file_exists("_0.fdx"));
    assert!(dir.pending_deletions().unwrap().is_empty());
    assert!(dir.fs_directory_path().is_none());

    // A rename across the two sides cannot be atomic.
    let err = dir.rename("_0.fdt", "_0.tim2").unwrap_err();
    assert!(
        matches!(&err, Error::Io(e) if e.kind() == std::io::ErrorKind::CrossesDevices),
        "{err}"
    );
    dir.rename("segments_1", "segments_2").unwrap();
    assert!(dir.secondary_dir().file_exists("segments_2"));

    // Temp files go where their embedded extension says.
    let tmp = dir.create_temp_output("_0.fdt", "sort").unwrap();
    assert!(dir.primary_dir().file_exists(tmp.name()));
    tmp.close().unwrap();
    let tmp = dir.create_temp_output("_0", "sort").unwrap();
    assert!(dir.secondary_dir().file_exists(tmp.name()));

    // The lock routes by its own extension.
    let lock = dir.obtain_lock("write.lock").unwrap();
    assert!(matches!(
        dir.secondary_dir().obtain_lock("write.lock"),
        Err(Error::LockObtainFailed(_))
    ));
    lock.close().unwrap();
}

/// `TestFileSwitchDirectory.testNoDir` / LUCENE-3380: listing tolerates a
/// side whose directory does not exist yet, but not both.
#[test]
fn list_all_tolerates_one_missing_side() {
    let root = TempDir::new("file-switch");
    let missing = root.join("missing");
    let present = root.join("present");
    std::fs::create_dir(&present).unwrap();

    // Both missing: the primary's error.
    let both = FileSwitchDirectory::new(
        exts(&["fdt"]),
        FsDirectory::open(&missing),
        FsDirectory::open(root.join("also-missing")),
    )
    .unwrap();
    assert!(both.list_all().unwrap_err().is_no_such_file());

    // Primary missing, secondary empty: still an error.
    let one = FileSwitchDirectory::new(
        exts(&["fdt"]),
        FsDirectory::open(&missing),
        FsDirectory::open(&present),
    )
    .unwrap();
    assert!(one.list_all().unwrap_err().is_no_such_file());
    // ... but not once the secondary has a file.
    write(&one, "segments_1", b"c");
    assert_eq!(one.list_all().unwrap(), vec!["segments_1"]);

    // Secondary missing: fine while the primary lists something.
    let other = FileSwitchDirectory::new(
        exts(&["si"]),
        FsDirectory::open(&present),
        FsDirectory::open(&missing),
    )
    .unwrap();
    assert!(other.list_all().unwrap_err().is_no_such_file());
    write(&other, "_0.si", b"s");
    assert_eq!(other.list_all().unwrap(), vec!["_0.si"]);

    // Two views of one path list each file once.
    let same = FileSwitchDirectory::new(
        exts(&["si"]),
        FsDirectory::open(&present),
        FsDirectory::open(&present),
    )
    .unwrap();
    assert_eq!(same.list_all().unwrap(), vec!["_0.si", "segments_1"]);
}

#[test]
fn list_all_propagates_other_errors() {
    let root = TempDir::new("file-switch-err");
    let file = root.join("a-file");
    std::fs::write(&file, b"").unwrap();
    // Listing a regular file is not "no such file".
    let dir = FileSwitchDirectory::new(
        exts(&["fdt"]),
        FsDirectory::open(&file),
        ByteBuffersDirectory::new(),
    )
    .unwrap();
    assert!(!dir.list_all().unwrap_err().is_no_such_file());
    let dir = FileSwitchDirectory::new(
        exts(&["fdt"]),
        ByteBuffersDirectory::new(),
        FsDirectory::open(&file),
    )
    .unwrap();
    assert!(!dir.list_all().unwrap_err().is_no_such_file());
}
