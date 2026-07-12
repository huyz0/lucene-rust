//! Port of `org.apache.lucene.store.Directory` / `FSDirectory` / `MMapDirectory`,
//! plus the generation-lookup logic from `org.apache.lucene.index.SegmentInfos`
//! (`getLastCommitGeneration`, `generationFromSegmentsFileName`) that depends only
//! on a file listing.
//!
//! Two backends, one trait:
//! - [`FsDirectory`]: `std::fs::read` — safe, no `unsafe`, always correct. Default.
//! - [`MmapDirectory`]: `memmap2` — zero-copy reads matching Lucene's own default
//!   (`MMapDirectory`) for real workloads. Contains this crate's only `unsafe`
//!   (documented on the call site): mapping a file is only sound if nothing else
//!   truncates/mutates it concurrently, same caveat Lucene's own Javadoc carries.
//!
//! Both return an [`Input`] — an owned-or-mapped byte buffer that `Deref`s to
//! `&[u8]`, so callers (codec_util, segment_info, segment_infos) are unchanged
//! regardless of backend.

use std::fs;
use std::ops::Deref;
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};
use crate::index_output::{self, FsIndexOutput};

/// The `segments` file-name prefix (`IndexFileNames.SEGMENTS`). Excludes the
/// pre-4.0 `segments.gen` pointer file, which is not a valid commit file name.
const SEGMENTS_PREFIX: &str = "segments";
const OLD_SEGMENTS_GEN: &str = "segments.gen";

/// A file's bytes, however the backend obtained them.
pub enum Input {
    Owned(Vec<u8>),
    Mapped(memmap2::Mmap),
}

impl Deref for Input {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        match self {
            Input::Owned(v) => v,
            Input::Mapped(m) => m,
        }
    }
}

/// Directory abstraction covering both Lucene's read path (`listAll`, `open`
/// a whole file's bytes) and the write-path primitives this crate now
/// supports: `createOutput` (a real on-disk [`FsIndexOutput`]) and `sync`
/// (the fsync-before-durable contract). Locking (`NativeFSLockFactory`) and
/// the `segments_N` commit lifecycle (rename/generation bookkeeping) are
/// still deferred — see `docs/parity.md`.
pub trait Directory {
    /// Port of `Directory.listAll()`: every file name in the directory, sorted.
    fn list_all(&self) -> Result<Vec<String>>;

    /// Reads a whole file's bytes.
    fn open(&self, name: &str) -> Result<Input>;

    /// Port of `Directory.createOutput(name, context)`: creates (truncating
    /// any existing file of the same name) a new file for sequential
    /// writing, backed by a real `std::fs::File`.
    fn create_output(&self, name: &str) -> Result<FsIndexOutput>;

    /// Port of `Directory.sync(Collection<String>)`: fsyncs every named
    /// file's contents (and, best-effort, the directory entry) to disk.
    /// Callers must sync a new segment's files before referencing them from
    /// a commit file — that's Lucene's actual durability contract.
    fn sync(&self, names: &[String]) -> Result<()>;
}

/// Safe, copying backend (`std::fs::read`). No `unsafe` anywhere in this crate
/// when used exclusively.
pub struct FsDirectory {
    root: PathBuf,
}

impl FsDirectory {
    pub fn open(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }
}

impl Directory for FsDirectory {
    fn list_all(&self) -> Result<Vec<String>> {
        list_all(&self.root)
    }

    fn open(&self, name: &str) -> Result<Input> {
        Ok(Input::Owned(fs::read(self.root.join(name))?))
    }

    fn create_output(&self, name: &str) -> Result<FsIndexOutput> {
        index_output::create_output(&self.root, name)
    }

    fn sync(&self, names: &[String]) -> Result<()> {
        index_output::sync(&self.root, names)
    }
}

/// Zero-copy backend (`memmap2`), matching Lucene's default `MMapDirectory`.
pub struct MmapDirectory {
    root: PathBuf,
}

impl MmapDirectory {
    pub fn open(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }
}

impl Directory for MmapDirectory {
    fn list_all(&self) -> Result<Vec<String>> {
        list_all(&self.root)
    }

    fn open(&self, name: &str) -> Result<Input> {
        let file = fs::File::open(self.root.join(name))?;
        // SAFETY: mapping is only unsound if another process truncates or
        // mutates this file while it's mapped, which we do not do ourselves and
        // which Lucene's own `MMapDirectory` accepts the same risk for (see its
        // Javadoc). The directory is opened read-only and outlives no writer in
        // the read-only phase this crate currently implements (PLAN.md Phase 2).
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        Ok(Input::Mapped(mmap))
    }

    fn create_output(&self, name: &str) -> Result<FsIndexOutput> {
        index_output::create_output(&self.root, name)
    }

    fn sync(&self, names: &[String]) -> Result<()> {
        index_output::sync(&self.root, names)
    }
}

fn list_all(root: &Path) -> Result<Vec<String>> {
    let mut names: Vec<String> = fs::read_dir(root)?
        .map(|entry| entry.map(|e| e.file_name().to_string_lossy().into_owned()))
        .collect::<std::io::Result<_>>()?;
    names.sort();
    Ok(names)
}

/// Port of `SegmentInfos.generationFromSegmentsFileName`.
pub fn generation_from_segments_file_name(file_name: &str) -> Result<i64> {
    if file_name == OLD_SEGMENTS_GEN {
        return Err(Error::Corrupted(format!(
            "\"{OLD_SEGMENTS_GEN}\" is not a valid segment file name since 4.0"
        )));
    }
    if file_name == SEGMENTS_PREFIX {
        return Ok(0);
    }
    if let Some(suffix) = file_name.strip_prefix(&format!("{SEGMENTS_PREFIX}_")) {
        return lucene_util::base36::from_base36(suffix).ok_or_else(|| {
            Error::Corrupted(format!("fileName \"{file_name}\" is not a segments file"))
        });
    }
    Err(Error::Corrupted(format!(
        "fileName \"{file_name}\" is not a segments file"
    )))
}

/// Port of `SegmentInfos.getLastCommitGeneration(String[])`: the highest
/// generation among `segments`/`segments_N` file names (excluding the legacy
/// `segments.gen` pointer), or -1 if none exist.
pub fn last_commit_generation(files: &[String]) -> i64 {
    files
        .iter()
        .filter(|f| f.starts_with(SEGMENTS_PREFIX) && f.as_str() != OLD_SEGMENTS_GEN)
        .filter_map(|f| generation_from_segments_file_name(f).ok())
        .max()
        .unwrap_or(-1)
}

/// Port of `IndexFileNames.fileNameFromGeneration("segments", "", gen)`.
pub fn segments_file_name(generation: i64) -> Option<String> {
    match generation {
        -1 => None,
        0 => Some(SEGMENTS_PREFIX.to_string()),
        gen => Some(format!(
            "{SEGMENTS_PREFIX}_{}",
            lucene_util::base36::to_base36(gen)
        )),
    }
}

/// Finds and reads the most recent `segments_N` commit file in `dir`.
/// Returns `(generation, bytes)`; callers pass both to `segment_infos::parse`.
pub fn read_latest_commit(dir: &impl Directory) -> Result<(i64, Input)> {
    let files = dir.list_all()?;
    let generation = last_commit_generation(&files);
    let name = segments_file_name(generation)
        .ok_or_else(|| Error::Corrupted("no segments_N commit file found".to_string()))?;
    let bytes = dir.open(&name)?;
    Ok((generation, bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_output::DataOutput;

    #[test]
    fn generation_from_segments_file_name_valid_cases() {
        assert_eq!(generation_from_segments_file_name("segments").unwrap(), 0);
        assert_eq!(generation_from_segments_file_name("segments_1").unwrap(), 1);
        assert_eq!(generation_from_segments_file_name("segments_2").unwrap(), 2);
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
        assert_eq!(last_commit_generation(&files), 3);
    }

    #[test]
    fn segments_file_name_all_branches() {
        assert_eq!(segments_file_name(-1), None);
        assert_eq!(segments_file_name(0), Some("segments".to_string()));
        assert_eq!(segments_file_name(1), Some("segments_1".to_string()));
        assert_eq!(segments_file_name(10), Some("segments_a".to_string()));
    }

    fn tempdir() -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "lucene-rust-directory-write-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&p).unwrap();
        p
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
    fn read_latest_commit_no_segments_file_is_corrupted_error() {
        struct EmptyDir;
        impl Directory for EmptyDir {
            fn list_all(&self) -> Result<Vec<String>> {
                Ok(vec!["_0.si".to_string()])
            }
            fn open(&self, _name: &str) -> Result<Input> {
                unreachable!("no segments_N file should be found, so open() is never called")
            }
            fn create_output(&self, _name: &str) -> Result<FsIndexOutput> {
                unreachable!("not used by this test")
            }
            fn sync(&self, _names: &[String]) -> Result<()> {
                unreachable!("not used by this test")
            }
        }
        assert!(matches!(
            read_latest_commit(&EmptyDir),
            Err(Error::Corrupted(_))
        ));
    }
}
