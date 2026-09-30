//! Port of `org.apache.lucene.index.IndexUpgrader` (M8): rewrites every
//! segment an older Lucene wrote into the current format, by a forced merge
//! under [`UpgradeIndexMergePolicy`], then commits -- the tool an operator
//! runs to finish a 9.x/10.x-to-10.5 upgrade instead of waiting for ordinary
//! merging to converge the index.
//!
//! Rust-only differences: no `InfoStream` (the `-verbose` flag is accepted
//! and ignored) and no `-dir-impl` (this port has one directory
//! implementation); `main`'s argument handling is [`IndexUpgrader::parse_args`],
//! which returns an error where Java prints the usage and exits.

use std::path::PathBuf;
use std::sync::Arc;

use lucene_store::directory::Directory;

use crate::index_file_deleter;
use crate::index_writer::{self, IndexWriter};
use crate::merge_policy::api::TieredMergePolicy;
use crate::merge_policy::upgrade::{UpgradeIndexMergePolicy, LATEST};
use crate::segment_infos;

/// What `upgrade` can fail with.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// `IndexNotFoundException`: no `segments_N` in the directory.
    #[error("no index found in the directory: {0}")]
    IndexNotFound(String),
    /// The `IllegalArgumentException` for prior commit points.
    #[error(
        "This tool was invoked to not delete prior commit points, but the following commits were found: {0:?}"
    )]
    PriorCommits(Vec<String>),
    /// `parseArgs`' usage error.
    #[error("usage: IndexUpgrader [-delete-prior-commits] [-verbose] indexDir ({0})")]
    Usage(String),
    #[error(transparent)]
    Writer(#[from] index_writer::Error),
    #[error(transparent)]
    Deleter(#[from] index_file_deleter::Error),
}

/// `IndexUpgrader`.
pub struct IndexUpgrader<'d> {
    dir: &'d dyn Directory,
    delete_prior_commits: bool,
}

/// The parsed command line: the index path and `-delete-prior-commits`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Args {
    pub path: PathBuf,
    pub delete_prior_commits: bool,
}

impl<'d> IndexUpgrader<'d> {
    /// `new IndexUpgrader(dir, infoStream, deletePriorCommits)`.
    pub fn new(dir: &'d dyn Directory, delete_prior_commits: bool) -> Self {
        IndexUpgrader {
            dir,
            delete_prior_commits,
        }
    }

    /// `parseArgs(args)`: `[-delete-prior-commits] [-verbose] indexDir`.
    pub fn parse_args(args: &[String]) -> Result<Args, Error> {
        let mut path = None;
        let mut delete_prior_commits = false;
        for arg in args {
            match arg.as_str() {
                "-delete-prior-commits" => delete_prior_commits = true,
                "-verbose" => {}
                "-dir-impl" => return Err(Error::Usage("-dir-impl is not supported".into())),
                other if path.is_none() => path = Some(PathBuf::from(other)),
                other => return Err(Error::Usage(format!("unexpected argument {other:?}"))),
            }
        }
        let path = path.ok_or_else(|| Error::Usage("missing indexDir".into()))?;
        Ok(Args {
            path,
            delete_prior_commits,
        })
    }

    /// `upgrade()`.
    pub fn upgrade(&self) -> Result<(), Error> {
        // `DirectoryReader.indexExists(dir)`.
        if segment_infos::read_latest(self.dir).is_err() {
            return Err(Error::IndexNotFound(format!("{:?}", self.dir.list_all())));
        }
        if !self.delete_prior_commits {
            let commits = index_file_deleter::list_commits(self.dir)?;
            if commits.len() > 1 {
                return Err(Error::PriorCommits(
                    commits
                        .iter()
                        .map(|c| c.segments_file_name().to_string())
                        .collect(),
                ));
            }
        }
        // `iwc.setMergePolicy(new UpgradeIndexMergePolicy(iwc.getMergePolicy()))`
        // over `IndexWriterConfig`'s default `TieredMergePolicy`, and
        // `KeepOnlyLastCommitDeletionPolicy`, which is the writer's default.
        let mut w = IndexWriter::open(self.dir, Vec::new(), "Lucene104", LATEST)?;
        w.set_pluggable_merge_policy(Some(Arc::new(UpgradeIndexMergePolicy::new(Box::new(
            TieredMergePolicy::default(),
        )))));
        w.force_merge(1)?;
        // "fake change to enforce a commit (e.g. if index has no segments)"
        let data = w.live_commit_data().to_vec();
        w.set_live_commit_data(data);
        w.commit()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    // The arithmetic gate is about values read off disk; a test's document
    // count difference is not one. See docs/arithmetic-gate.md.
    #![allow(clippy::arithmetic_side_effects)]
    use std::path::Path;

    use lucene_store::directory::FsDirectory;
    use lucene_util::test_support::TempDir;

    use super::*;
    use crate::deletion_policy::NoDeletionPolicy;

    /// A copy of `fixtures/data/bwc/<version>` (index files only).
    fn copy_fixture(version: &str) -> TempDir {
        let src = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/data/bwc")
            .join(version);
        let tmp = TempDir::new(&format!("index-upgrader-{version}"));
        for entry in std::fs::read_dir(src).unwrap() {
            let entry = entry.unwrap();
            let name = entry.file_name().to_string_lossy().to_string();
            if !name.ends_with(".txt") {
                std::fs::copy(entry.path(), tmp.path().join(&name)).unwrap();
            }
        }
        tmp
    }

    fn live_docs(dir: &FsDirectory) -> i64 {
        segment_infos::read_latest(dir)
            .unwrap()
            .segments
            .iter()
            .map(|s| {
                let si = crate::segment_info::parse_for_codec(
                    &dir.open(&format!("{}.si", s.segment_name)).unwrap(),
                    &s.segment_id,
                    &s.codec_name,
                )
                .unwrap();
                i64::from(si.doc_count) - i64::from(s.del_count)
            })
            .sum()
    }

    #[test]
    fn every_old_segment_is_rewritten_and_a_second_run_changes_nothing() {
        let tmp = copy_fixture("9.0.0");
        let dir = FsDirectory::open(tmp.path());
        let live = live_docs(&dir);
        let before = segment_infos::read_latest(&dir).unwrap();
        assert!(before.segments.iter().all(|s| s.codec_name == "Lucene90"));

        IndexUpgrader::new(&dir, false).upgrade().unwrap();
        let after = segment_infos::read_latest(&dir).unwrap();
        assert_eq!(after.segments.len(), 1, "the old segments merge into one");
        let seg = &after.segments[0];
        assert_eq!(seg.codec_name, "Lucene104");
        let si = crate::segment_info::parse(
            &dir.open(&format!("{}.si", seg.segment_name)).unwrap(),
            &seg.segment_id,
        )
        .unwrap();
        assert_eq!(si.version, LATEST);
        assert_eq!(live_docs(&dir), live);
        for r in crate::check_index::check_directory(&dir).unwrap() {
            assert!(r.failures().is_empty(), "{}", r.segment_name);
        }

        // Nothing is old any more: the segments stay as they are, and the
        // forced commit still writes a new generation.
        IndexUpgrader::new(&dir, false).upgrade().unwrap();
        let again = segment_infos::read_latest(&dir).unwrap();
        assert_eq!(again.segments.len(), 1);
        assert_eq!(again.segments[0].segment_name, seg.segment_name);
        assert!(again.generation > after.generation);
    }

    #[test]
    fn prior_commits_are_refused_unless_deletion_is_asked_for() {
        let tmp = copy_fixture("9.12.2");
        let dir = FsDirectory::open(tmp.path());
        {
            // A second commit point that keeps the first.
            let mut w = IndexWriter::open_with_deletion_policy(
                &dir,
                Vec::new(),
                "Lucene104",
                LATEST,
                Arc::new(NoDeletionPolicy),
            )
            .unwrap();
            w.set_live_commit_data(vec![("k".into(), "v".into())]);
            w.commit().unwrap();
        }
        assert!(index_file_deleter::list_commits(&dir).unwrap().len() > 1);
        match IndexUpgrader::new(&dir, false).upgrade() {
            Err(Error::PriorCommits(commits)) => assert!(commits.len() > 1, "{commits:?}"),
            other => panic!("expected PriorCommits, got {other:?}"),
        }
        IndexUpgrader::new(&dir, true).upgrade().unwrap();
        let infos = segment_infos::read_latest(&dir).unwrap();
        assert!(infos.segments.iter().all(|s| s.codec_name == "Lucene104"));
        assert_eq!(index_file_deleter::list_commits(&dir).unwrap().len(), 1);
    }

    #[test]
    fn a_directory_without_an_index_is_an_error() {
        let tmp = TempDir::new("index-upgrader-empty");
        let dir = FsDirectory::open(tmp.path());
        assert!(matches!(
            IndexUpgrader::new(&dir, false).upgrade(),
            Err(Error::IndexNotFound(_))
        ));
    }

    #[test]
    fn arguments_parse_like_javas() {
        let args = |a: &[&str]| a.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(
            IndexUpgrader::parse_args(&args(&["-delete-prior-commits", "-verbose", "/idx"]))
                .unwrap(),
            Args {
                path: PathBuf::from("/idx"),
                delete_prior_commits: true
            }
        );
        assert!(
            !IndexUpgrader::parse_args(&args(&["/idx"]))
                .unwrap()
                .delete_prior_commits
        );
        for bad in [&[][..], &["/a", "/b"][..], &["-dir-impl", "X", "/a"][..]] {
            assert!(matches!(
                IndexUpgrader::parse_args(&args(bad)),
                Err(Error::Usage(_))
            ));
        }
    }
}
