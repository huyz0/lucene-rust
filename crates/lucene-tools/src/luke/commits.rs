//! `org.apache.lucene.luke.models.commits`: the commits a directory holds,
//! their files and their segments.

use std::path::{Path, PathBuf};

use lucene_index::deletion_policy::IndexCommit;
use lucene_index::index_file_deleter::{list_commits, segment_files};
use lucene_index::segment_info;
use lucene_index::segment_infos::{self, SegmentCommitInfo, SegmentInfos};
use lucene_store::directory::Directory;

use super::{display_size, read_map_string, LukeError, Result};

/// `commits.Commit`: one commit point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Commit {
    pub generation: i64,
    pub is_deleted: bool,
    pub seg_count: usize,
    /// The user data map's `toString()`.
    pub user_data: String,
}

impl Commit {
    fn of(ic: &IndexCommit) -> Self {
        Commit {
            generation: ic.generation(),
            is_deleted: ic.is_deleted(),
            seg_count: ic.segment_count(),
            user_data: read_map_string(ic.user_data()),
        }
    }
}

/// `commits.File`: a file of a commit and its size.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct File {
    pub file_name: String,
    /// `toDisplaySize`, `unknown` when the size cannot be read.
    pub display_size: String,
}

/// `commits.Segment`: a segment of a commit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segment {
    pub name: String,
    pub max_doc: i32,
    pub del_gen: i64,
    pub del_count: i32,
    pub lucene_ver: String,
    pub codec_name: String,
    /// The size of every file of the segment; empty when it cannot be read.
    pub display_size: String,
    pub use_compound_file: bool,
}

/// What a segment's `.si` holds that the model reports.
struct SegmentDetails {
    sci: SegmentCommitInfo,
    max_doc: i32,
    version: String,
    compound: bool,
    attributes: Vec<(String, String)>,
    diagnostics: Vec<(String, String)>,
}

/// `CommitsImpl`.
pub struct Commits<'d> {
    dir: &'d dyn Directory,
    index_path: PathBuf,
    /// The commits by generation, ascending.
    commits: Vec<IndexCommit>,
}

/// The error every segment read of `generation` reports.
fn segment_error(generation: i64) -> LukeError {
    LukeError::luke(format!(
        "Failed to load segment infos for commit generation {generation}"
    ))
}

impl<'d> Commits<'d> {
    /// `new CommitsImpl(dir, indexPath)`: lists the directory's commits.
    ///
    /// # Errors
    /// `LukeException` when the commits cannot be listed.
    pub fn new(dir: &'d dyn Directory, index_path: &Path) -> Result<Self> {
        let mut commits =
            list_commits(dir).map_err(|_| LukeError::luke("Failed to get commits list."))?;
        commits.sort_by_key(IndexCommit::generation);
        Ok(Commits {
            dir,
            index_path: index_path.to_path_buf(),
            commits,
        })
    }

    fn find(&self, generation: i64) -> Option<&IndexCommit> {
        self.commits.iter().find(|c| c.generation() == generation)
    }

    /// `listCommits()`: newest first.
    pub fn list_commits(&self) -> Vec<Commit> {
        self.commits.iter().rev().map(Commit::of).collect()
    }

    /// `getCommit(commitGen)`.
    pub fn commit(&self, generation: i64) -> Option<Commit> {
        self.find(generation).map(Commit::of)
    }

    /// `getFiles(commitGen)`: the commit's files by name, each with its size.
    pub fn files(&self, generation: i64) -> Vec<File> {
        let Some(ic) = self.find(generation) else {
            return Vec::new();
        };
        let mut files: Vec<File> = ic
            .file_names()
            .iter()
            .map(|name| File {
                file_name: name.clone(),
                display_size: std::fs::metadata(self.index_path.join(name))
                    .map_or_else(|_| "unknown".to_string(), |m| display_size(m.len())),
            })
            .collect();
        files.sort_by(|a, b| a.file_name.cmp(&b.file_name));
        files
    }

    /// `findSegmentInfos`: the commit's `segments_N`, `None` for an unknown
    /// generation.
    fn segment_infos(&self, generation: i64) -> Result<Option<SegmentInfos>> {
        let Some(ic) = self.find(generation) else {
            return Ok(None);
        };
        let err = || segment_error(generation);
        let bytes = self
            .dir
            .open(ic.segments_file_name())
            .map_err(|_| err())?
            .to_vec();
        segment_infos::parse(&bytes, generation)
            .map(Some)
            .map_err(|_| err())
    }

    /// Every segment of the commit with its `.si`, by name.
    fn details(&self, generation: i64) -> Result<Vec<SegmentDetails>> {
        let Some(infos) = self.segment_infos(generation)? else {
            return Ok(Vec::new());
        };
        let mut details = Vec::with_capacity(infos.segments.len());
        for sci in infos.segments {
            let bytes = self
                .dir
                .open(&format!("{}.si", sci.segment_name))
                .map_err(|_| segment_error(generation))?
                .to_vec();
            let si = segment_info::parse_for_codec(&bytes, &sci.segment_id, &sci.codec_name)
                .map_err(|_| segment_error(generation))?;
            let v = si.version;
            details.push(SegmentDetails {
                sci,
                max_doc: si.doc_count,
                version: format!("{}.{}.{}", v.major, v.minor, v.bugfix),
                compound: si.is_compound_file,
                attributes: si.attributes,
                diagnostics: si.diagnostics,
            });
        }
        details.sort_by(|a, b| a.sci.segment_name.cmp(&b.sci.segment_name));
        Ok(details)
    }

    /// `getSegments(commitGen)`: the commit's segments by name.
    ///
    /// # Errors
    /// `LukeException` when the commit's segments cannot be read.
    pub fn segments(&self, generation: i64) -> Result<Vec<Segment>> {
        Ok(self
            .details(generation)?
            .into_iter()
            .map(|d| Segment {
                display_size: self
                    .size_in_bytes(&d.sci)
                    .map_or_else(String::new, display_size),
                name: d.sci.segment_name,
                max_doc: d.max_doc,
                del_gen: d.sci.del_gen,
                del_count: d.sci.del_count,
                lucene_ver: d.version,
                codec_name: d.sci.codec_name,
                use_compound_file: d.compound,
            })
            .collect())
    }

    /// `SegmentCommitInfo.sizeInBytes()`: every file of the segment.
    fn size_in_bytes(&self, sci: &SegmentCommitInfo) -> Option<u64> {
        let files = segment_files(self.dir, sci).ok()?;
        files.iter().try_fold(0u64, |acc, f| {
            Some(acc.saturating_add(self.dir.file_length(f).ok()?))
        })
    }

    fn segment(&self, generation: i64, name: &str) -> Result<Option<SegmentDetails>> {
        Ok(self
            .details(generation)?
            .into_iter()
            .find(|d| d.sci.segment_name == name))
    }

    /// `getSegmentAttributes(commitGen, name)`, by key.
    ///
    /// # Errors
    /// `LukeException` when the commit's segments cannot be read.
    pub fn segment_attributes(&self, generation: i64, name: &str) -> Result<Vec<(String, String)>> {
        Ok(sorted(
            self.segment(generation, name)?.map(|d| d.attributes),
        ))
    }

    /// `getSegmentDiagnostics(commitGen, name)`, by key.
    ///
    /// # Errors
    /// `LukeException` when the commit's segments cannot be read.
    pub fn segment_diagnostics(
        &self,
        generation: i64,
        name: &str,
    ) -> Result<Vec<(String, String)>> {
        Ok(sorted(
            self.segment(generation, name)?.map(|d| d.diagnostics),
        ))
    }

    /// `getSegmentCodec(commitGen, name)`: the codec's name.
    ///
    /// # Errors
    /// `LukeException` when the commit's segments cannot be read.
    pub fn segment_codec(&self, generation: i64, name: &str) -> Result<Option<String>> {
        Ok(self.segment(generation, name)?.map(|d| d.sci.codec_name))
    }
}

fn sorted(map: Option<Vec<(String, String)>>) -> Vec<(String, String)> {
    let mut map = map.unwrap_or_default();
    map.sort();
    map
}
