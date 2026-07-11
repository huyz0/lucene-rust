//! Differential test: open the real two-commit index directory (from
//! fixtures/src/GenSegmentInfos.java) through both Directory backends and
//! confirm they agree on the listing and on locating the latest commit.

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
    assert_eq!(directory::last_commit_generation(&files), 2);
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

    let expected = std::fs::read(format!("{}/segments_2.raw", dir_path())).unwrap();
    assert_eq!(&*bytes, expected.as_slice());
}

#[test]
fn mmap_backend_reads_same_bytes_as_fs_backend() {
    let fs = FsDirectory::open(dir_path());
    let mmap = MmapDirectory::open(dir_path());

    let fs_bytes = fs.open("segments_2").unwrap();
    let mmap_bytes = mmap.open("segments_2").unwrap();
    assert_eq!(&*fs_bytes, &*mmap_bytes);
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
    assert_eq!(directory::last_commit_generation(&[]), -1);
}
