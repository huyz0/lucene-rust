#![cfg(test)]
use super::*;
use crate::data_output::DataOutput;
use crate::index_output::IndexOutput;
use crate::lock::NoLockFactory;

fn write(dir: &dyn Directory, name: &str, bytes: &[u8]) -> u64 {
    let mut out = dir.create_output(name).unwrap();
    out.write_bytes(bytes);
    out.close().unwrap()
}

#[test]
fn written_files_read_back_and_list_sorted() {
    let dir = ByteBuffersDirectory::new();
    let checksum = write(&dir, "b.dat", b"hello");
    assert_eq!(checksum, crc32fast::hash(b"hello") as u64);
    write(&dir, "a.dat", b"");
    assert_eq!(dir.list_all().unwrap(), vec!["a.dat", "b.dat"]);
    let input = dir.open("b.dat").unwrap();
    assert_eq!(&*input, b"hello");
    assert_eq!(format!("{input:?}"), "Input::Shared(5 bytes)");
    assert_eq!(dir.file_length("b.dat").unwrap(), 5);
    assert_eq!(dir.file_length("a.dat").unwrap(), 0);
    assert!(dir.file_exists("a.dat"));
    assert!(!dir.file_exists("c.dat"));
    assert!(dir.pending_deletions().unwrap().is_empty());
    dir.sync(&["b.dat".to_string()]).unwrap();
    dir.sync_meta_data().unwrap();
    assert!(dir.fs_directory_path().is_none());
    assert!(format!("{dir:?}").contains("SingleInstanceLockFactory"));
}

#[test]
fn an_open_output_is_listed_empty_and_unreadable() {
    let dir = ByteBuffersDirectory::default();
    let mut out = dir.create_output("_0.fdt").unwrap();
    out.write_bytes(b"abc");
    assert_eq!(out.file_pointer(), 3);
    assert!(out.path().is_none());
    assert_eq!(dir.list_all().unwrap(), vec!["_0.fdt"]);
    assert_eq!(dir.file_length("_0.fdt").unwrap(), 0);
    let err = dir.open("_0.fdt").unwrap_err();
    assert!(
        matches!(&err, Error::Io(e) if e.kind() == std::io::ErrorKind::PermissionDenied),
        "{err}"
    );
    out.close().unwrap();
    assert_eq!(&*dir.open("_0.fdt").unwrap(), b"abc");
}

#[test]
fn a_file_is_written_once() {
    let dir = ByteBuffersDirectory::new();
    write(&dir, "x", b"1");
    let err = dir.create_output("x").unwrap_err();
    assert!(
        matches!(&err, Error::Io(e) if e.kind() == std::io::ErrorKind::AlreadyExists),
        "{err}"
    );
}

#[test]
fn missing_files_are_no_such_file() {
    let dir = ByteBuffersDirectory::new();
    assert!(dir.open("nope").unwrap_err().is_no_such_file());
    assert!(dir.file_length("nope").unwrap_err().is_no_such_file());
    assert!(dir.delete_file("nope").unwrap_err().is_no_such_file());
    assert!(dir.rename("nope", "x").unwrap_err().is_no_such_file());
}

#[test]
fn rename_moves_and_refuses_to_overwrite() {
    let dir = ByteBuffersDirectory::new();
    write(&dir, "pending_segments_1", b"commit");
    write(&dir, "other", b"o");
    dir.rename("pending_segments_1", "segments_1").unwrap();
    assert_eq!(dir.list_all().unwrap(), vec!["other", "segments_1"]);
    assert_eq!(&*dir.open("segments_1").unwrap(), b"commit");
    let err = dir.rename("other", "segments_1").unwrap_err();
    assert!(
        matches!(&err, Error::Io(e) if e.kind() == std::io::ErrorKind::AlreadyExists),
        "{err}"
    );
    // An output renamed while open publishes under its new name.
    let mut out = dir.create_output("tmp").unwrap();
    dir.rename("tmp", "final").unwrap();
    out.write_bytes(b"late");
    out.close().unwrap();
    assert_eq!(&*dir.open("final").unwrap(), b"late");
}

#[test]
fn delete_removes_and_a_later_close_is_harmless() {
    let dir = ByteBuffersDirectory::new();
    write(&dir, "a", b"1");
    dir.delete_file("a").unwrap();
    assert!(dir.list_all().unwrap().is_empty());
    let out = dir.create_output("b").unwrap();
    dir.delete_file("b").unwrap();
    out.close().unwrap();
    assert!(!dir.file_exists("b"));
    assert_eq!(dir.remove_entry("b"), None);
}

#[test]
fn temp_outputs_get_fresh_names() {
    let dir = ByteBuffersDirectory::new();
    // Occupy the first name the counter will produce.
    write(&dir, "_0_sort_0.tmp", b"taken");
    let a = dir.create_temp_output("_0", "sort").unwrap();
    let b = dir.create_temp_output("_0", "sort").unwrap();
    assert_eq!(a.name(), "_0_sort_1.tmp");
    assert_eq!(b.name(), "_0_sort_2.tmp");
    a.close().unwrap();
    b.close().unwrap();
    assert_eq!(dir.list_all().unwrap().len(), 3);
}

#[test]
fn copy_from_copies_between_directories() {
    let src = ByteBuffersDirectory::new();
    let dst = ByteBuffersDirectory::new();
    write(&src, "a", b"payload");
    dst.copy_from(&src, "a", "b").unwrap();
    assert_eq!(&*dst.open("b").unwrap(), b"payload");
    // A missing source fails and leaves no partial destination.
    assert!(dst.copy_from(&src, "missing", "c").is_err());
    assert!(!dst.file_exists("c"));
}

#[test]
fn default_lock_factory_is_per_directory() {
    let a = ByteBuffersDirectory::new();
    let b = ByteBuffersDirectory::new();
    let held = a.obtain_lock("write.lock").unwrap();
    assert!(matches!(
        a.obtain_lock("write.lock"),
        Err(Error::LockObtainFailed(_))
    ));
    let _other = b.obtain_lock("write.lock").unwrap();
    held.close().unwrap();
    a.obtain_lock("write.lock").unwrap();

    let none = ByteBuffersDirectory::with_lock_factory(Arc::new(NoLockFactory));
    let _x = none.obtain_lock("write.lock").unwrap();
    let _y = none.obtain_lock("write.lock").unwrap();
}

#[test]
fn publish_hook_sees_only_files_still_listed() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let hook_seen = Arc::clone(&seen);
    let dir = ByteBuffersDirectory::with_publish_hook(Arc::new(move |len| {
        hook_seen.lock().unwrap().push(len)
    }));
    write(&dir, "kept", b"1234");
    let out = dir.create_output("dropped").unwrap();
    dir.delete_file("dropped").unwrap();
    out.close().unwrap();
    assert_eq!(*seen.lock().unwrap(), vec![4]);
    assert_eq!(dir.remove_entry("kept"), Some(4));
}
