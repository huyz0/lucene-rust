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

/// Minimal read-only directory abstraction. Write support (`createOutput`,
/// locking) is deferred to the write-path phase (see PLAN.md Phase 5).
pub trait Directory {
    /// Port of `Directory.listAll()`: every file name in the directory, sorted.
    fn list_all(&self) -> Result<Vec<String>>;

    /// Reads a whole file's bytes.
    fn open(&self, name: &str) -> Result<Input>;
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
