#![cfg(test)]
use super::*;
use crate::data_output::DataOutput;

/// Random-access advice and read-ahead are hints: a mapping, a compound
/// member over one, and a heap buffer read the same bytes after both,
/// over ranges larger than one read-ahead call and unaligned ones.
#[test]
fn advice_and_read_ahead_leave_every_input_readable() {
    let tmp = lucene_util::test_support::TempDir::new("store-advice");
    let root = tmp.path().to_path_buf();
    let data: Vec<u8> = (0..1_000_000u32).map(|i| (i * 7) as u8).collect();
    std::fs::write(root.join("_0.vec"), &data).unwrap();
    let mapped = Arc::new(
        MmapDirectory::with_read_threshold(&root, 0)
            .open("_0.vec")
            .unwrap(),
    );
    assert!(matches!(*mapped, Input::Mapped(_)));
    let member = Input::slice_of(&mapped, 4097, 300_000).unwrap();
    let owned = Input::Owned(data.clone());
    for input in [&*mapped, &member, &owned] {
        input.advise_random();
        input.advise_random_and_warm();
        input.will_need(13, 700_000).unwrap();
        crate::read_ahead(&input[1..]);
    }
    assert_eq!(&mapped[..], &data[..]);
    assert_eq!(&member[..], &data[4097..304_097]);
    assert_eq!(&owned[..], &data[..]);
    crate::read_ahead(&[]);
}

#[cfg(target_os = "linux")]
#[test]
fn a_read_mapping_is_resident() {
    let tmp = lucene_util::test_support::TempDir::new("store-resident");
    let root = tmp.path().to_path_buf();
    let bytes: Vec<u8> = (0..300_000u32).map(|i| i as u8).collect();
    std::fs::write(root.join("f.bin"), &bytes).unwrap();
    let input = Arc::new(
        MmapDirectory::with_read_threshold(&root, 0)
            .open("f.bin")
            .unwrap(),
    );
    assert!(matches!(*input, Input::Mapped(_)));
    // Every byte read, so every page is in the page cache (more than one
    // `mincore` batch of them).
    assert_eq!(&input[..], &bytes[..]);
    assert!(input.is_resident(0, input.len()));
    let member = Input::slice_of(&input, 4097, 200_000).unwrap();
    assert!(member.is_resident(0, member.len()));
    assert!(input.is_resident(4095, 2));
    assert!(input.is_resident(input.len(), 10));
    assert!(input.is_resident(10, usize::MAX));
    let owned = Input::Owned(vec![1, 2, 3]);
    assert!(owned.is_resident(0, 3));
}

#[test]
fn memory_available_is_known_on_linux() {
    assert!(memory_available().is_some_and(|b| b > 0));
}

#[test]
fn close_input_drops_the_input_and_forwards_through_a_box() {
    let dir: Box<dyn Directory> = Box::new(crate::ByteBuffersDirectory::new());
    let mut out = dir.create_output("f").unwrap();
    out.write_bytes(b"abc");
    out.close().unwrap();
    let input = dir.open("f").unwrap();
    assert_eq!(&input[..], b"abc");
    dir.close_input("f", input).unwrap();
}

#[test]
fn slice_of_shares_the_base_and_checks_bounds() {
    let base = Arc::new(Input::Owned(b"0123456789".to_vec()));
    let mid = Input::slice_of(&base, 3, 4).unwrap();
    assert_eq!(&mid[..], b"3456");
    assert_eq!(format!("{mid:?}"), "Input::Slice(4 bytes)");
    assert_eq!(Arc::strong_count(&base), 2);
    // A slice of a slice reads through both windows.
    let inner = Input::slice_of(&Arc::new(mid), 1, 2).unwrap();
    assert_eq!(inner.as_ref(), b"45");
    assert_eq!(&Input::slice_of(&base, 10, 0).unwrap()[..], b"");
    let tracked = Arc::new(Input::Tracked(TrackedInput::new(
        Input::Owned(b"abc".to_vec()),
        (),
    )));
    let copy = Input::slice_of(&tracked, 1, 2).unwrap();
    assert!(matches!(copy, Input::Owned(_)), "{copy:?}");
    assert_eq!((&copy[..], Arc::strong_count(&tracked)), (&b"bc"[..], 1));
    for (start, len) in [(0, 11), (11, 0), (usize::MAX, 2)] {
        let err = Input::slice_of(&base, start, len).unwrap_err();
        assert!(matches!(err, Error::Corrupted(_)), "{err:?}");
    }
}

#[test]
fn generation_from_segments_file_name_valid_cases() {
    assert_eq!(generation_from_segments_file_name("segments").unwrap(), 0);
    assert_eq!(generation_from_segments_file_name("segments_1").unwrap(), 1);
    assert_eq!(generation_from_segments_file_name("segments_2").unwrap(), 2);
    // Java skips any one character after `segments`, not only `_`.
    assert_eq!(
        generation_from_segments_file_name("segments.bak").unwrap(),
        14_636
    );
    assert_eq!(generation_from_segments_file_name("segments-7").unwrap(), 7);
    assert!(generation_from_segments_file_name("segmentsX").is_err());
    // base-36: "segments_a" -> 10
    assert_eq!(
        generation_from_segments_file_name("segments_a").unwrap(),
        10
    );
}

#[test]
fn generation_from_segments_file_name_rejects_old_pointer_file() {
    assert!(matches!(
        generation_from_segments_file_name("segments.gen"),
        Err(Error::Corrupted(_))
    ));
}

#[test]
fn generation_from_segments_file_name_rejects_garbage() {
    assert!(matches!(
        generation_from_segments_file_name("not-a-segments-file"),
        Err(Error::Corrupted(_))
    ));
    // Has the prefix but a non-base-36 suffix.
    assert!(matches!(
        generation_from_segments_file_name("segments_!!!"),
        Err(Error::Corrupted(_))
    ));
}

#[test]
fn last_commit_generation_ignores_old_pointer_and_non_segments_files() {
    let files = vec![
        "segments.gen".to_string(),
        "_0.si".to_string(),
        "segments_1".to_string(),
        "segments_3".to_string(),
        "segments_2".to_string(),
    ];
    assert_eq!(last_commit_generation(&files).unwrap(), 3);
}

#[test]
fn segments_file_name_all_branches() {
    assert_eq!(segments_file_name(-1), None);
    assert_eq!(segments_file_name(0), Some("segments".to_string()));
    assert_eq!(segments_file_name(1), Some("segments_1".to_string()));
    assert_eq!(segments_file_name(10), Some("segments_a".to_string()));
}

/// A `Directory` that can only be listed: every test using it asserts
/// that `read_latest_commit` fails during the *scan*, before any file is
/// opened, so the remaining methods must never be called.
struct ListingOnlyDir(Vec<String>);

impl Directory for ListingOnlyDir {
    fn list_all(&self) -> Result<Vec<String>> {
        Ok(self.0.clone())
    }
    fn open(&self, name: &str) -> Result<Input> {
        panic!("open({name}) must not be reached: the generation scan should have failed")
    }
    fn create_output(&self, name: &str) -> Result<FsIndexOutput> {
        panic!("create_output({name}) is not part of the read path under test")
    }
    fn sync(&self, _names: &[String]) -> Result<()> {
        panic!("sync() is not part of the read path under test")
    }
    fn rename(&self, source: &str, dest: &str) -> Result<()> {
        panic!("rename({source}, {dest}) is not part of the read path under test")
    }
    fn delete_file(&self, name: &str) -> Result<()> {
        panic!("delete_file({name}) is not part of the read path under test")
    }
    fn sync_meta_data(&self) -> Result<()> {
        panic!("sync_meta_data() is not part of the read path under test")
    }
    fn obtain_lock(&self, name: &str) -> Result<Box<dyn Lock>> {
        panic!("obtain_lock({name}) is not part of the read path under test")
    }
}

#[test]
fn input_debug_reports_provenance_and_length_not_contents() {
    // The Debug impl exists so a panic message can't dump a mapped
    // half-gigabyte file; assert it stays that way.
    let owned = Input::Owned(vec![7u8; 42]);
    assert_eq!(format!("{owned:?}"), "Input::Owned(42 bytes)");

    let root = tempdir();
    // `with_read_threshold(_, 0)`: a ten-byte file is below
    // `SMALL_FILE_READ_THRESHOLD`, so the default backend would read it.
    let dir = MmapDirectory::with_read_threshold(&root, 0);
    index_output::write_all_bytes(&root, "_0.si", b"0123456789").unwrap();
    let mapped = dir.open("_0.si").unwrap();
    assert_eq!(format!("{mapped:?}"), "Input::Mapped(10 bytes)");
    fs::remove_dir_all(&root).ok();
}

/// [`SMALL_FILE_READ_THRESHOLD`] decides *how* a file is obtained and
/// nothing else: the bytes a caller sees are identical either way.
///
/// Both sides are asserted, because the interesting failure is silent --
/// a threshold that never fires costs the syscalls it exists to remove
/// and nothing reports it, and one that fires on a large file copies a
/// mapping the reader meant to hold.
#[test]
fn small_files_are_read_and_large_ones_mapped() {
    let root = tempdir();
    let small = vec![3u8; 8];
    let large = vec![4u8; (SMALL_FILE_READ_THRESHOLD as usize) + 1];
    index_output::write_all_bytes(&root, "_0.nvm", &small).unwrap();
    index_output::write_all_bytes(&root, "_0.doc", &large).unwrap();

    let dir = MmapDirectory::open(&root);
    let got_small = dir.open("_0.nvm").unwrap();
    let got_large = dir.open("_0.doc").unwrap();
    assert!(matches!(got_small, Input::Owned(_)), "{got_small:?}");
    assert!(matches!(got_large, Input::Mapped(_)), "{got_large:?}");
    assert_eq!(&*got_small, &small[..]);
    assert_eq!(&*got_large, &large[..]);

    // Exactly at the threshold is still read (`<=`).
    let exact = vec![5u8; SMALL_FILE_READ_THRESHOLD as usize];
    index_output::write_all_bytes(&root, "_0.dvm", &exact).unwrap();
    assert!(matches!(dir.open("_0.dvm").unwrap(), Input::Owned(_)));

    // And `0` turns the whole thing off, which is the A/B arm
    // `reader_open_profile` measures against.
    let mapping = MmapDirectory::with_read_threshold(&root, 0);
    assert!(matches!(mapping.open("_0.nvm").unwrap(), Input::Mapped(_)));
    assert_eq!(&*mapping.open("_0.nvm").unwrap(), &small[..]);
    fs::remove_dir_all(&root).ok();
}

/// A **zero-length** file: the edge `SMALL_FILE_READ_THRESHOLD` moves
/// across a code path, checked to move nothing observable.
///
/// `0 <= threshold` is always true, so an empty file takes the read arm
/// rather than being mapped. Worth pinning because the obvious guess is
/// wrong: `mmap(2)` rejects a zero length, but `memmap2::Mmap::map`
/// special-cases it and hands back an empty mapping rather than an error,
/// so the two arms agree.
#[test]
fn a_zero_length_file_reads_as_empty_on_every_backend() {
    let root = tempdir();
    index_output::write_all_bytes(&root, "_0.nvm", b"").unwrap();

    // The read arm, which an empty file now takes.
    let read = MmapDirectory::open(&root).open("_0.nvm").unwrap();
    assert!(matches!(read, Input::Owned(_)), "{read:?}");
    assert!(read.is_empty());

    // The mapping arm -- also `Ok`, also empty.
    let mapped = MmapDirectory::with_read_threshold(&root, 0)
        .open("_0.nvm")
        .expect("memmap2 maps a zero-length file rather than refusing it");
    assert!(matches!(mapped, Input::Mapped(_)), "{mapped:?}");
    assert!(mapped.is_empty());

    // And the copying backend, unchanged throughout.
    assert!(FsDirectory::open(&root).open("_0.nvm").unwrap().is_empty());
    fs::remove_dir_all(&root).ok();
}

#[test]
fn pending_segments_file_name_matches_file_name_from_generation() {
    assert_eq!(pending_segments_file_name(-1), None);
    assert_eq!(
        pending_segments_file_name(0),
        Some("pending_segments".to_string())
    );
    assert_eq!(
        pending_segments_file_name(1),
        Some("pending_segments_1".to_string())
    );
    // base-36, same radix as `segments_N`.
    assert_eq!(
        pending_segments_file_name(10),
        Some("pending_segments_a".to_string())
    );
}

/// The whole point of the pending name: `getLastCommitGeneration` must not
/// see it, so a half-written commit can never become the current one.
#[test]
fn a_pending_segments_file_is_invisible_to_the_commit_generation_scan() {
    let files = vec![
        "segments_1".to_string(),
        "pending_segments_2".to_string(),
        "_0.si".to_string(),
    ];
    assert_eq!(last_commit_generation(&files).unwrap(), 1);
}

#[test]
fn rename_publishes_a_file_under_a_new_name_and_delete_file_removes_it() {
    let root = tempdir();
    let dir = FsDirectory::open(&root);

    index_output::write_all_bytes(&root, "pending_segments_1", b"commit").unwrap();
    assert!(dir
        .list_all()
        .unwrap()
        .contains(&"pending_segments_1".to_string()));

    dir.rename("pending_segments_1", "segments_1").unwrap();
    dir.sync_meta_data().unwrap();
    let listed = dir.list_all().unwrap();
    assert!(!listed.contains(&"pending_segments_1".to_string()));
    assert_eq!(&*dir.open("segments_1").unwrap(), b"commit");

    dir.delete_file("segments_1").unwrap();
    assert!(dir.list_all().unwrap().is_empty());
    fs::remove_dir_all(&root).ok();
}

#[test]
fn rename_and_delete_file_surface_io_errors_for_a_missing_source() {
    let root = tempdir();
    let dir = MmapDirectory::open(&root);
    assert!(matches!(
        dir.rename("nope", "segments_1"),
        Err(Error::Io(_))
    ));
    assert!(matches!(dir.delete_file("nope"), Err(Error::Io(_))));
    // `sync_meta_data` is best-effort by design (not every platform lets a
    // directory be fsynced) -- it reports success either way.
    dir.sync_meta_data().unwrap();
    fs::remove_dir_all(&root).ok();
}

use lucene_util::test_support::TempDir;

/// A scratch directory that removes itself when the test ends -- unless
/// the test is panicking, in which case its bytes stay for inspection.
fn tempdir() -> TempDir {
    TempDir::new("directory-write")
}

#[test]
fn fs_directory_create_output_round_trips_through_open_and_list_all() {
    let root = tempdir();
    let dir = FsDirectory::open(&root);

    let mut out = dir.create_output("_0.si").unwrap();
    out.write_bytes(b"hello lucene-rust");
    let checksum = out.close().unwrap();
    assert_eq!(checksum, crc32fast::hash(b"hello lucene-rust") as u64);

    dir.sync(&["_0.si".to_string()]).unwrap();

    assert_eq!(dir.list_all().unwrap(), vec!["_0.si".to_string()]);
    let bytes = dir.open("_0.si").unwrap();
    assert_eq!(&*bytes, b"hello lucene-rust");

    fs::remove_dir_all(&root).ok();
}

#[test]
fn mmap_directory_create_output_round_trips_through_open() {
    let root = tempdir();
    let dir = MmapDirectory::open(&root);

    let mut out = dir.create_output("_0.si").unwrap();
    out.write_bytes(b"mmap round trip");
    out.close().unwrap();

    let bytes = dir.open("_0.si").unwrap();
    assert_eq!(&*bytes, b"mmap round trip");

    assert_eq!(dir.list_all().unwrap(), vec!["_0.si".to_string()]);
    dir.sync(&["_0.si".to_string()]).unwrap();

    fs::remove_dir_all(&root).ok();
}

#[test]
fn slice_input_over_a_real_file_has_independent_file_pointers() {
    // Writes a real file to a temp dir via FsDirectory/FsIndexOutput, then
    // slices it via SliceInput::slice_input the way a merge would slice a
    // sub-range of a real on-disk `.cfs`. Guards against any real
    // OS-file-handle-sharing bug that an in-memory-only test can't catch
    // (e.g. accidentally sharing one file's read position across slices).
    use crate::data_input::{DataInput, SliceInput};

    let root = tempdir();
    let dir = FsDirectory::open(&root);
    let mut out = dir.create_output("_0.cfs").unwrap();
    out.write_bytes(b"HEADER|firstpart|secondpart|FOOTER");
    out.close().unwrap();

    let bytes = dir.open("_0.cfs").unwrap();
    let root_input = SliceInput::new(&bytes);

    // "firstpart" starts at offset 7, "secondpart" at offset 17.
    let mut first = root_input.slice_input("first", 7, 9).unwrap();
    let mut second = root_input.slice_input("second", 17, 10).unwrap();

    // Interleave reads through both real-file-backed slices.
    let mut buf1 = [0u8; 4];
    let mut buf2 = [0u8; 4];
    first.read_bytes(&mut buf1).unwrap();
    second.read_bytes(&mut buf2).unwrap();
    assert_eq!(&buf1, b"firs");
    assert_eq!(&buf2, b"seco");

    let mut buf1b = [0u8; 5];
    let mut buf2b = [0u8; 6];
    first.read_bytes(&mut buf1b).unwrap();
    second.read_bytes(&mut buf2b).unwrap();
    assert_eq!(&buf1b, b"tpart");
    assert_eq!(&buf2b, b"ndpart");

    // Both slices are now fully consumed; further reads are Eof, not a
    // leak into the other slice's or the footer's bytes.
    assert!(first.read_byte().is_err());
    assert!(second.read_byte().is_err());

    fs::remove_dir_all(&root).ok();
}

#[test]
fn fs_directory_open_nonexistent_file_is_io_error() {
    let dir = FsDirectory::open("/nonexistent-lucene-rust-test-path");
    assert!(matches!(dir.open("whatever"), Err(Error::Io(_))));
}

#[test]
fn fs_directory_list_all_nonexistent_dir_is_io_error() {
    let dir = FsDirectory::open("/nonexistent-lucene-rust-test-path");
    assert!(matches!(dir.list_all(), Err(Error::Io(_))));
}

#[test]
fn mmap_directory_open_nonexistent_file_is_io_error() {
    let dir = MmapDirectory::open("/nonexistent-lucene-rust-test-path");
    assert!(matches!(dir.open("whatever"), Err(Error::Io(_))));
}

#[test]
fn read_latest_commit_finds_highest_generation_segments_file() {
    let root = tempdir();
    let dir = FsDirectory::open(&root);
    index_output::write_all_bytes(&root, "segments_1", b"old").unwrap();
    index_output::write_all_bytes(&root, "segments_2", b"newest").unwrap();

    let (generation, bytes) = read_latest_commit(&dir).unwrap();
    assert_eq!(generation, 2);
    assert_eq!(&*bytes, b"newest");

    fs::remove_dir_all(&root).ok();
}

#[test]
fn read_latest_commit_rejects_an_unparsable_segments_file_name() {
    // Java's getLastCommitGeneration lets generationFromSegmentsFileName's
    // exception escape here. Skipping the bad name instead would open
    // segments_1 and silently drop whatever segments_2 committed.
    let files = vec![
        "segments_1".to_string(),
        "segments_zzzzzzzzzzzzz".to_string(),
    ];
    assert!(matches!(
        read_latest_commit(&ListingOnlyDir(files.clone())),
        Err(Error::Corrupted(_))
    ));
    // ... and the scan the write path shares with it is equally strict:
    // reporting generation 1 here would let `IndexWriter` create a fresh
    // index over a directory that does hold a commit.
    assert!(matches!(
        last_commit_generation(&files),
        Err(Error::Corrupted(_))
    ));
}

#[test]
fn read_latest_commit_ignores_the_legacy_pointer_file() {
    let root = tempdir();
    let dir = FsDirectory::open(&root);
    index_output::write_all_bytes(&root, "segments.gen", b"legacy").unwrap();
    index_output::write_all_bytes(&root, "segments_1", b"real").unwrap();
    let (generation, bytes) = read_latest_commit(&dir).unwrap();
    assert_eq!(generation, 1);
    assert_eq!(&*bytes, b"real");
    fs::remove_dir_all(&root).ok();
}

#[test]
fn read_latest_commit_no_segments_file_is_corrupted_error() {
    assert!(matches!(
        read_latest_commit(&ListingOnlyDir(vec!["_0.si".to_string()])),
        Err(Error::Corrupted(_))
    ));
}

/// Deletes that fail with anything but "no such file" become pending, as
/// on Windows: hidden from listings and reads, retried later, and
/// resurrected by a create or a rename onto the name. Here the failure
/// is `EISDIR` -- `remove_file` on a directory -- which is portable to
/// every Unix and needs no permissions trickery (tests run as root).
#[test]
fn a_failed_delete_becomes_pending_until_it_can_be_retried() {
    for backend in 0..2 {
        let root = tempdir();
        let dir: Box<dyn Directory> = if backend == 0 {
            Box::new(FsDirectory::open(&root))
        } else {
            Box::new(MmapDirectory::open(&root))
        };
        fs::create_dir(root.join("_0.cfs")).unwrap();
        index_output::write_all_bytes(&root, "_1.si", b"x").unwrap();
        dir.delete_file("_0.cfs").unwrap();
        assert_eq!(dir.list_all().unwrap(), vec!["_1.si".to_string()]);
        assert!(dir.open("_0.cfs").unwrap_err().is_no_such_file());
        assert!(dir.file_length("_0.cfs").unwrap_err().is_no_such_file());
        assert!(dir.delete_file("_0.cfs").unwrap_err().is_no_such_file());
        assert!(dir.rename("_0.cfs", "x").unwrap_err().is_no_such_file());
        // Still undeletable: still pending.
        assert_eq!(
            dir.pending_deletions().unwrap(),
            BTreeSet::from(["_0.cfs".to_string()])
        );
        // Once deletable, the next retry takes it.
        replace_dir_with_file(&root, "_0.cfs");
        assert!(dir.pending_deletions().unwrap().is_empty());
        assert!(!root.join("_0.cfs").exists());
        assert_eq!(dir.file_length("_1.si").unwrap(), 1);
    }
}

/// Makes a directory that failed to delete deletable.
fn replace_dir_with_file(root: &Path, name: &str) {
    fs::remove_dir(root.join(name)).unwrap();
    fs::write(root.join(name), b"old").unwrap();
}

/// Leaves `name` pending in `dir`, deletable at the next retry.
fn make_pending(dir: &FsDirectory, root: &Path, name: &str) {
    fs::create_dir(root.join(name)).unwrap();
    dir.delete_file(name).unwrap();
    assert!(dir.core.is_pending(name));
    replace_dir_with_file(root, name);
}

/// `FSDirectory.createOutput` is `CREATE_NEW`: an existing name is
/// `FileAlreadyExistsException` (`BaseDirectoryTestCase.
/// testCreateOutputForExistingFile`), the file untouched; and the first
/// output under a root that does not exist yet creates it, as Java's
/// constructor does (`TestFileSwitchDirectory` over two fresh paths).
#[test]
fn create_output_refuses_an_existing_name_and_creates_a_missing_root() {
    let root = tempdir();
    let dirs: Vec<Box<dyn Directory>> = vec![
        Box::new(FsDirectory::open(root.join("fs/nested"))),
        Box::new(MmapDirectory::open(root.join("mm/nested"))),
    ];
    for dir in &dirs {
        let mut out = dir.create_output("file").unwrap();
        out.write_bytes(b"abc");
        out.close().unwrap();
        let err = dir.create_output("file").unwrap_err();
        match &err {
            Error::Io(e) => assert_eq!(e.kind(), std::io::ErrorKind::AlreadyExists),
            other => panic!("{other}"),
        }
        assert_eq!(&*dir.open("file").unwrap(), b"abc");
        dir.delete_file("file").unwrap();
        dir.create_output("file").unwrap().close().unwrap();
    }
    let temp_root = root.join("temp/nested");
    let dir = FsDirectory::open(&temp_root);
    let out = dir.create_temp_output("p", "s").unwrap();
    out.close().unwrap();
    assert!(temp_root.is_dir());
}

#[test]
fn pending_deletes_are_retried_by_later_operations_and_on_drop() {
    let root = tempdir();
    let dir = FsDirectory::open(&root);
    // One pending file: the very next mutating op retries it.
    make_pending(&dir, &root, "a");
    dir.sync_meta_data().unwrap();
    assert!(!root.join("a").exists());

    make_pending(&dir, &root, "e");
    index_output::write_all_bytes(&root, "s", b"s").unwrap();
    dir.sync(&["s".to_string()]).unwrap();
    assert!(!root.join("e").exists());

    // Creating over a pending name brings it back to life.
    make_pending(&dir, &root, "b");
    let mut out = dir.create_output("b").unwrap();
    out.write_bytes(b"new");
    out.close().unwrap();
    assert_eq!(&*dir.open("b").unwrap(), b"new");
    assert!(dir.pending_deletions().unwrap().is_empty());

    // So does a rename onto it.
    make_pending(&dir, &root, "c");
    dir.rename("b", "c").unwrap();
    assert_eq!(&*dir.open("c").unwrap(), b"new");

    // A resurrection whose retry still fails leaves the name to the
    // create, which then fails on it -- and it is no longer pending.
    fs::create_dir(root.join("d")).unwrap();
    dir.delete_file("d").unwrap();
    assert!(dir.create_output("d").is_err());
    assert!(dir.pending_deletions().unwrap().is_empty());
    fs::remove_dir(root.join("d")).unwrap();

    // A pending file that vanished on its own is an error on retry
    // outside Windows (Java's `NoSuchFileException`), and is forgotten.
    fs::create_dir(root.join("g")).unwrap();
    dir.delete_file("g").unwrap();
    fs::remove_dir(root.join("g")).unwrap();
    assert!(dir.delete_pending_files().unwrap_err().is_no_such_file());
    assert!(dir.pending_deletions().unwrap().is_empty());

    // Drop is `close()`: a last retry.
    make_pending(&dir, &root, "f");
    drop(dir);
    assert!(!root.join("f").exists());
}

#[test]
fn create_temp_output_never_clobbers() {
    let root = tempdir();
    let dir = MmapDirectory::open(&root);
    index_output::write_all_bytes(&root, "_0_sort_0.tmp", b"taken").unwrap();
    let out = dir.create_temp_output("_0", "sort").unwrap();
    assert_eq!(crate::IndexOutput::name(&out), "_0_sort_1.tmp");
    out.close().unwrap();
    assert_eq!(&*dir.open("_0_sort_0.tmp").unwrap(), b"taken");
    let fs_dir = FsDirectory::open(&root);
    let out = fs_dir.create_temp_output("_0", "sort").unwrap();
    assert_eq!(crate::IndexOutput::name(&out), "_0_sort_2.tmp");
    // A directory that cannot be created is an error, not an endless
    // retry (a missing one is created, as Java's constructor does).
    index_output::write_all_bytes(&root, "plain_file", b"").unwrap();
    assert!(FsDirectory::open(root.join("plain_file").join("missing"))
        .create_temp_output("_0", "x")
        .is_err());
    assert_eq!(temp_file_name("_5", "fdt", 35), "_5_fdt_z.tmp");
    assert_eq!(
        temp_file_name("p", "s", u64::MAX),
        temp_file_name("p", "s", i64::MAX as u64)
    );
}

#[test]
fn fs_directories_expose_their_path_and_lock_factory() {
    let root = tempdir();
    let dir = FsDirectory::with_lock_factory(&root, Arc::new(crate::NoLockFactory));
    assert_eq!(dir.directory(), root.path());
    assert_eq!(dir.fs_directory_path(), Some(root.path()));
    assert!(format!("{:?}", dir.lock_factory()).contains("NoLockFactory"));
    let _a = dir.obtain_lock("write.lock").unwrap();
    let _b = dir.obtain_lock("write.lock").unwrap();

    let mmap = MmapDirectory::with_lock_factory(&root, Arc::new(crate::NoLockFactory));
    assert_eq!(mmap.directory(), root.path());
    assert_eq!(mmap.fs_directory_path(), Some(root.path()));
    mmap.delete_pending_files().unwrap();
    let native = MmapDirectory::open(&root);
    let held = native.obtain_lock("write.lock").unwrap();
    assert!(matches!(
        FsDirectory::open(&root).obtain_lock("write.lock"),
        Err(Error::LockObtainFailed(_))
    ));
    drop(held);
    let base = BaseDirectory::new(Arc::new(crate::NoLockFactory));
    assert!(format!("{base:?}").contains("NoLockFactory"));
    base.obtain_lock(&native, "x").unwrap();
}

#[test]
fn references_and_arcs_are_directories() {
    let root = tempdir();
    let dir = Arc::new(FsDirectory::open(&root));
    let by_ref = &*dir;
    let as_ref: &dyn Directory = &by_ref;
    let shared: &dyn Directory = &dir;
    for d in [as_ref, shared] {
        let mut out = d.create_output("a").unwrap();
        out.write_bytes(b"1");
        out.close().unwrap();
        assert_eq!(d.file_length("a").unwrap(), 1);
        assert_eq!(&*d.open("a").unwrap(), b"1");
        d.sync(&["a".to_string()]).unwrap();
        d.rename("a", "b").unwrap();
        d.copy_from(&*dir, "b", "c").unwrap();
        d.delete_file("b").unwrap();
        d.delete_file("c").unwrap();
        d.sync_meta_data().unwrap();
        d.create_temp_output("t", "s").unwrap().close().unwrap();
        assert!(d.pending_deletions().unwrap().is_empty());
        assert_eq!(d.fs_directory_path(), Some(root.path()));
        let lock = d.obtain_lock("write.lock").unwrap();
        lock.close().unwrap();
        assert!(!d.list_all().unwrap().is_empty());
    }
}

/// A directory that implements only the required methods, over an
/// in-memory one: exercises the trait's defaults.
struct RequiredOnly<'a>(&'a crate::ByteBuffersDirectory);

impl Directory for RequiredOnly<'_> {
    fn list_all(&self) -> Result<Vec<String>> {
        self.0.list_all()
    }
    fn open(&self, name: &str) -> Result<Input> {
        self.0.open(name)
    }
    fn create_output(&self, name: &str) -> Result<FsIndexOutput> {
        self.0.create_output(name)
    }
    fn sync(&self, names: &[String]) -> Result<()> {
        self.0.sync(names)
    }
    fn rename(&self, source: &str, dest: &str) -> Result<()> {
        self.0.rename(source, dest)
    }
    fn delete_file(&self, name: &str) -> Result<()> {
        self.0.delete_file(name)
    }
    fn sync_meta_data(&self) -> Result<()> {
        self.0.sync_meta_data()
    }
    fn obtain_lock(&self, name: &str) -> Result<Box<dyn Lock>> {
        self.0.obtain_lock(name)
    }
}

#[test]
fn trait_defaults_read_the_file_and_refuse_temp_outputs() {
    let bb = crate::ByteBuffersDirectory::new();
    let only = RequiredOnly(&bb);
    let err = only.create_temp_output("a", "b").unwrap_err();
    assert!(matches!(&err, Error::Io(e) if e.kind() == std::io::ErrorKind::Unsupported));
    assert!(only.pending_deletions().unwrap().is_empty());
    assert!(only.fs_directory_path().is_none());
    assert!(!only.disable_virus_checker());
    let mut out = only.create_output("f").unwrap();
    out.write_bytes(b"four");
    out.close().unwrap();
    assert_eq!(only.file_length("f").unwrap(), 4);
    assert_eq!(only.list_all().unwrap(), vec!["f"]);
    only.sync(&[]).unwrap();
    only.sync_meta_data().unwrap();
    only.copy_from(&bb, "f", "g").unwrap();
    only.rename("g", "h").unwrap();
    only.delete_file("h").unwrap();
    only.obtain_lock("l").unwrap();
}

/// Refuses deleting or moving any file whose name starts with `held`,
/// as `WindowsFS` refuses a file that is still open.
struct Refusing;

impl FsFaults for Refusing {
    fn check_delete(&self, path: &Path) -> std::io::Result<()> {
        refuse(path)
    }
    fn check_rename(&self, source: &Path, _dest: &Path) -> std::io::Result<()> {
        refuse(source)
    }
}

fn refuse(path: &Path) -> std::io::Result<()> {
    let name = path.file_name().unwrap().to_string_lossy();
    if name.starts_with("held") {
        return Err(std::io::Error::other(format!("access denied: {name}")));
    }
    Ok(())
}

#[test]
fn disabling_virus_checkers_reaches_filesystems_through_wrappers() {
    struct Virus(std::sync::atomic::AtomicBool);
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
    let root = tempdir();
    assert!(!FsDirectory::open(&root).disable_virus_checker());
    assert!(!MmapDirectory::open(&root).disable_virus_checker());
    assert!(!Refusing.disable_virus_checker());
    for mmap in [false, true] {
        let faults = Arc::new(Virus(std::sync::atomic::AtomicBool::new(true)));
        let backend: Box<dyn Directory> = if mmap {
            Box::new(MmapDirectory::open(&root).with_fs_faults(faults.clone()))
        } else {
            Box::new(FsDirectory::open(&root).with_fs_faults(faults.clone()))
        };
        backend.create_output("orphan").unwrap().close().unwrap();
        assert!(faults.check_delete(&root.join("orphan")).is_err());
        faults
            .check_rename(&root.join("orphan"), &root.join("other"))
            .unwrap();
        let shared = Arc::new(backend);
        let reference = &shared;
        let estimated = EstimatedWrites::new(&reference, 0);
        assert!(estimated.disable_virus_checker());
        assert!(!estimated.disable_virus_checker());
        estimated.delete_file("orphan").unwrap();
        assert!(!root.join("orphan").exists());
        assert!(estimated.pending_deletions().unwrap().is_empty());
    }
}

/// A refused delete becomes a pending delete -- hidden from the listing,
/// retried later -- exactly as a real one failing would, for both
/// backends; a refused move fails the rename.
#[test]
fn a_mock_filesystem_refusing_a_delete_makes_it_pending() {
    let tmp = tempdir();
    let dirs: Vec<Box<dyn Directory>> = vec![
        Box::new(FsDirectory::open(tmp.join("fs")).with_fs_faults(Arc::new(Refusing))),
        Box::new(MmapDirectory::open(tmp.join("mm")).with_fs_faults(Arc::new(Refusing))),
    ];
    for (dir, sub) in dirs.iter().zip(["fs", "mm"]) {
        std::fs::create_dir_all(tmp.join(sub)).unwrap();
        for name in ["held_a", "free_b"] {
            let mut out = dir.create_output(name).unwrap();
            out.write_bytes(b"xyz");
            out.close().unwrap();
        }
        dir.delete_file("free_b").unwrap();
        dir.delete_file("held_a").unwrap();
        assert!(dir.list_all().unwrap().is_empty(), "{sub}");
        assert!(
            tmp.join(sub).join("held_a").exists(),
            "{sub}: still on disk"
        );
        assert_eq!(
            dir.pending_deletions()
                .unwrap()
                .into_iter()
                .collect::<Vec<_>>(),
            vec!["held_a".to_string()]
        );
        let mut out = dir.create_output("held_c").unwrap();
        out.write_bytes(b"c");
        out.close().unwrap();
        let err = dir.rename("held_c", "d").unwrap_err();
        assert!(err.to_string().contains("access denied"), "{err}");
    }
}

/// A tracked input reads as the input it wraps and drops its guard with
/// it.
#[test]
fn a_tracked_input_derefs_to_its_bytes_and_drops_its_guard() {
    struct Guard(Arc<std::sync::atomic::AtomicBool>);
    impl Drop for Guard {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }
    let dropped = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let input = Input::Tracked(TrackedInput::new(
        Input::Owned(vec![1, 2, 3]),
        Guard(dropped.clone()),
    ));
    assert_eq!(&*input, &[1, 2, 3]);
    assert_eq!(format!("{input:?}"), "Input::Tracked(3 bytes)");
    let Input::Tracked(tracked) = &input else {
        unreachable!()
    };
    assert!(matches!(tracked.inner(), Input::Owned(_)));
    assert!(!dropped.load(Ordering::SeqCst));
    drop(input);
    assert!(dropped.load(Ordering::SeqCst));
}
