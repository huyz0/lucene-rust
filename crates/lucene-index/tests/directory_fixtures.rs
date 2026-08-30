//! Differential test: open the real two-commit index directory (from
//! fixtures/src/GenSegmentInfos.java) through both Directory backends and
//! confirm they agree on the listing and on locating the latest commit.
// Test-support code opts out of the arithmetic gate at the file boundary:
// the gate exists for values read off disk in production decode paths, not
// for a fixture builder's own index arithmetic. See
// `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use lucene_store::directory::{self, Directory};
use lucene_store::{FsDirectory, MmapDirectory};

fn dir_path() -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/segments_index"
    )
    .to_string()
}

#[test]
fn fs_and_mmap_agree_on_listing() {
    let fs = FsDirectory::open(dir_path());
    let mmap = MmapDirectory::open(dir_path());

    let mut fs_files = fs.list_all().unwrap();
    let mut mmap_files = mmap.list_all().unwrap();
    fs_files.sort();
    mmap_files.sort();
    assert_eq!(fs_files, mmap_files);
    assert!(fs_files.contains(&"segments_2".to_string()));
    assert!(fs_files.contains(&"_0.si".to_string()));
}

#[test]
fn locates_latest_commit_generation() {
    let fs = FsDirectory::open(dir_path());
    let files = fs.list_all().unwrap();
    assert_eq!(directory::last_commit_generation(&files).unwrap(), 2);
    assert_eq!(
        directory::segments_file_name(2),
        Some("segments_2".to_string())
    );
}

#[test]
fn read_latest_commit_matches_raw_fixture_bytes() {
    let fs = FsDirectory::open(dir_path());
    let (generation, bytes) = directory::read_latest_commit(&fs).unwrap();
    assert_eq!(generation, 2);

    let expected = std::fs::read(format!("{}/expected_segments_2.bin", dir_path())).unwrap();
    assert_eq!(&*bytes, expected.as_slice());
}

/// The two backends must hand back identical bytes for the same file.
///
/// **`with_read_threshold(_, 0)`, and the variant is asserted.** `segments_2`
/// is a few hundred bytes, i.e. under
/// `lucene_store::directory::SMALL_FILE_READ_THRESHOLD`, so the default
/// `MmapDirectory` now *reads* it -- which would leave this test comparing
/// one `Input::Owned` against another and no longer checking the mapping
/// backend at all. `c42-readpath-perf` introduced that threshold; this is
/// where the comparison it silently disabled is put back.
#[test]
fn mmap_backend_reads_same_bytes_as_fs_backend() {
    use lucene_store::Input;

    let fs = FsDirectory::open(dir_path());
    let mmap = MmapDirectory::with_read_threshold(dir_path(), 0);

    let fs_bytes = fs.open("segments_2").unwrap();
    let mmap_bytes = mmap.open("segments_2").unwrap();
    assert!(matches!(fs_bytes, Input::Owned(_)), "{fs_bytes:?}");
    assert!(matches!(mmap_bytes, Input::Mapped(_)), "{mmap_bytes:?}");
    assert_eq!(&*fs_bytes, &*mmap_bytes);

    // And the default backend, whose small-file arm is what a real reader
    // takes for this file, agrees with both.
    let read_not_mapped = MmapDirectory::open(dir_path()).open("segments_2").unwrap();
    assert!(
        matches!(read_not_mapped, Input::Owned(_)),
        "{read_not_mapped:?}"
    );
    assert_eq!(&*read_not_mapped, &*fs_bytes);
}

#[test]
fn end_to_end_parses_latest_commit_via_directory() {
    let mmap = MmapDirectory::open(dir_path());
    let (generation, bytes) = directory::read_latest_commit(&mmap).unwrap();
    let sis = lucene_index::segment_infos::parse(&bytes, generation).unwrap();
    assert_eq!(sis.segments.len(), 2);
    assert_eq!(sis.segments[0].segment_name, "_0");
    assert_eq!(sis.segments[1].segment_name, "_1");
}

#[test]
fn generation_zero_and_missing_cases() {
    assert_eq!(directory::segments_file_name(-1), None);
    assert_eq!(
        directory::segments_file_name(0),
        Some("segments".to_string())
    );
    assert_eq!(directory::last_commit_generation(&[]).unwrap(), -1);
}
