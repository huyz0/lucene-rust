//! `org.apache.lucene.luke.models.tools`: what Luke changes in an index --
//! merge it, delete or add documents, create one from a 20 Newsgroups dump
//! -- and what it checks and exports.
//!
//! Differs: `repairIndex` is not ported (Luke's `CheckIndex` repair is
//! `lucene-tools CheckIndex -exorcise`'s, through
//! [`lucene_index::check_index::exorcise`]); a `PrintStream` info stream is
//! not taken. A delete takes the [`DeleteQuery`] the writer can apply
//! rather than any `Query`.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use lucene_analysis::factory::spi;
use lucene_analysis::Analyzer;
use lucene_index::buffered_updates::DeleteQuery;
use lucene_index::check_index;
use lucene_index::deletion_policy::{
    IndexDeletionPolicy, KeepOnlyLastCommitDeletionPolicy, NoDeletionPolicy,
};
use lucene_index::document::Document;
use lucene_index::index_writer::IndexWriter;
use lucene_index::merge_policy::MergePolicyConfig;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::multi_terms::MultiTerms;
use lucene_store::directory::Directory;

use super::newsgroups::{Message, MessageFilesParser};
use super::{LukeError, Result};

/// The codec and version a writer records, as Lucene 10.5.0's defaults.
const CODEC: &str = "Lucene104";
const VERSION: LuceneVersion = LuceneVersion {
    major: 10,
    minor: 5,
    bugfix: 0,
};

/// What `CheckIndex.Status` reports that Luke shows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckStatus {
    /// `clean`: every check of the commit and its segments passed.
    pub clean: bool,
    /// `numSegments`.
    pub num_segments: usize,
    /// `numBadSegments`.
    pub num_bad_segments: usize,
    /// `totLoseDocCount`: the documents of the bad segments.
    pub tot_lose_doc_count: i64,
}

/// `IndexToolsImpl`: over a directory, and the reader opened on it for what
/// reads the index.
pub struct IndexTools<'a> {
    dir: &'a dyn Directory,
    reader: Option<&'a DirectoryReader>,
    use_compound: bool,
    keep_all_commits: bool,
}

impl<'a> IndexTools<'a> {
    /// `new IndexToolsImpl(dir, useCompound, keepAllCommits)`.
    pub fn new(dir: &'a dyn Directory, use_compound: bool, keep_all_commits: bool) -> Self {
        IndexTools {
            dir,
            reader: None,
            use_compound,
            keep_all_commits,
        }
    }

    /// `new IndexToolsImpl(reader, useCompound, keepAllCommits)`, the reader
    /// open on `dir` (`DirectoryReader.directory()`).
    pub fn with_reader(
        reader: &'a DirectoryReader,
        dir: &'a dyn Directory,
        use_compound: bool,
        keep_all_commits: bool,
    ) -> Self {
        IndexTools {
            reader: Some(reader),
            ..Self::new(dir, use_compound, keep_all_commits)
        }
    }

    /// `IndexUtils.createWriter(dir, analyzer, useCompound, keepAllCommits)`:
    /// `IndexWriterConfig`'s defaults (a `TieredMergePolicy`), the
    /// whitespace analyzer when none is given.
    fn writer(
        &self,
        analyzer: Option<Arc<Analyzer>>,
    ) -> lucene_index::index_writer::Result<IndexWriter<'a>> {
        let policy: Arc<dyn IndexDeletionPolicy> = if self.keep_all_commits {
            Arc::new(NoDeletionPolicy)
        } else {
            Arc::new(KeepOnlyLastCommitDeletionPolicy)
        };
        let mut writer =
            IndexWriter::open_with_deletion_policy(self.dir, Vec::new(), CODEC, VERSION, policy)?;
        writer.set_use_compound_file(self.use_compound);
        // This writer's `close` always writes a commit, where Java's skips
        // one with nothing pending; [`close`] does as Java's does.
        writer.set_commit_on_close(false);
        writer.set_merge_policy(Some(MergePolicyConfig::default()));
        let analyzer = analyzer.unwrap_or_else(|| {
            Arc::new(
                spi::new_analyzer("org.apache.lucene.analysis.core.WhitespaceAnalyzer")
                    .expect("WhitespaceAnalyzer is built in"),
            )
        });
        writer.set_analyzer(Some(analyzer));
        Ok(writer)
    }

    /// `reader instanceof DirectoryReader`, which `addDocument` and
    /// `deleteDocuments` require.
    fn require_reader(&self) -> Result<()> {
        match self.reader {
            Some(_) => Ok(()),
            None => Err(LukeError::luke(
                "Current reader is not an instance of DirectoryReader.",
            )),
        }
    }

    /// `optimize(expunge, maxNumSegments, ps)`: `forceMergeDeletes` or
    /// `forceMerge(maxNumSegments)`.
    ///
    /// # Errors
    /// `LukeException` without a reader or when the merge fails;
    /// `IllegalArgumentException` (from `forceMerge`, not wrapped) for
    /// `max_num_segments` under 1.
    pub fn optimize(&self, expunge: bool, max_num_segments: i32) -> Result<()> {
        if self.reader.is_none() {
            return Err(LukeError::luke("Current reader is not a DirectoryReader."));
        }
        let max_num_segments = match usize::try_from(max_num_segments) {
            Ok(m) if expunge || m >= 1 => m.max(1),
            _ => {
                return Err(LukeError::illegal_argument(format!(
                    "maxNumSegments must be >= 1; got {max_num_segments}"
                )))
            }
        };
        let run = || -> lucene_index::index_writer::Result<()> {
            let mut writer = self.writer(None)?;
            if expunge {
                writer.force_merge_deletes()?;
            } else {
                writer.force_merge(max_num_segments)?;
            }
            close(writer)
        };
        run().map_err(|e| failed("Failed to optimize index", &e))
    }

    /// `checkIndex(ps)`.
    ///
    /// # Errors
    /// `LukeException` when the index's latest commit cannot be read.
    pub fn check_index(&self) -> Result<CheckStatus> {
        let results = check_index::check_directory(self.dir)
            .map_err(|_| LukeError::luke("Failed to check index."))?;
        let (commit, segments) = results.split_first().expect("the commit's result is first");
        let bad: Vec<_> = segments.iter().filter(|r| !r.all_passed()).collect();
        Ok(CheckStatus {
            clean: commit.all_passed() && bad.is_empty(),
            num_segments: segments.len(),
            num_bad_segments: bad.len(),
            tot_lose_doc_count: bad.iter().filter_map(|r| r.max_doc).map(i64::from).sum(),
        })
    }

    /// `addDocument(doc, analyzer)`, committed.
    ///
    /// # Errors
    /// `LukeException` without a reader or when the document cannot be
    /// written.
    pub fn add_document(&self, doc: &Document, analyzer: Arc<Analyzer>) -> Result<()> {
        self.require_reader()?;
        let run = || -> lucene_index::index_writer::Result<()> {
            let mut writer = self.writer(Some(analyzer))?;
            writer.add_fields_document(doc)?;
            writer.commit()?;
            close(writer)
        };
        run().map_err(|e| failed("Failed to add document", &e))
    }

    /// `deleteDocuments(query)`, committed.
    ///
    /// # Errors
    /// `LukeException` without a reader or when the deletion cannot be
    /// written.
    pub fn delete_documents(&self, query: DeleteQuery) -> Result<()> {
        self.require_reader()?;
        let run = || -> lucene_index::index_writer::Result<()> {
            let mut writer = self.writer(None)?;
            writer.delete_documents_by_query(&[query])?;
            writer.commit()?;
            close(writer)
        };
        // Java's message, a copy of `addDocument`'s.
        run().map_err(|e| failed("Failed to add document", &e))
    }

    /// `createNewIndex(dataDir)`: a new index in the (empty) directory,
    /// holding the 20 Newsgroups messages under `data_dir` if one is given.
    ///
    /// # Errors
    /// `IllegalStateException` (no message) when the directory is not
    /// empty; `LukeException` when the index cannot be written.
    pub fn create_new_index(&self, data_dir: Option<&Path>) -> Result<()> {
        let listed = self
            .dir
            .list_all()
            .map_err(|e| failed("Cannot create new index.", &e))?;
        if !listed.is_empty() {
            return Err(LukeError::illegal_state());
        }
        let run = || -> std::result::Result<(), String> {
            let mut writer = self
                .writer(Some(Message::create_lucene_analyzer()))
                .map_err(|e| e.to_string())?;
            if let Some(data_dir) = data_dir {
                let messages = MessageFilesParser::new(data_dir)
                    .parse_all()
                    .map_err(|e| e.to_string())?;
                for message in &messages {
                    writer
                        .add_fields_document(&message.to_lucene_doc())
                        .map_err(|e| e.to_string())?;
                }
            }
            writer.commit().map_err(|e| e.to_string())?;
            close(writer).map_err(|e| e.to_string())
        };
        run().map_err(|e| failed("Cannot create new index.", &e))
    }

    /// `exportTerms(destDir, field, delimiter)`: every term of the field and
    /// its document frequency, one per line, to
    /// `terms_<field>_<millis>.out` in `dest_dir`. Returns the file's path.
    ///
    /// # Errors
    /// `LukeException` when the field has no terms or the file cannot be
    /// written.
    pub fn export_terms(&self, dest_dir: &str, field: &str, delimiter: &str) -> Result<String> {
        let millis = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_millis());
        let filename = format!("terms_{field}_{millis}.out");
        let path: PathBuf = Path::new(dest_dir).join(&filename);
        let reader = self
            .reader
            .ok_or_else(|| LukeError::luke("Current reader is not set."))?;
        let export_failed = |e: &dyn std::fmt::Display| {
            failed(
                &format!("Terms file export for field [{field}] to file [{filename}] has failed."),
                e,
            )
        };
        let terms = MultiTerms::get_terms(reader, field)
            .map_err(|e| export_failed(&e))?
            .ok_or_else(|| {
                LukeError::luke(format!(
                    "Field {field} does not contain any terms to be exported"
                ))
            })?;
        let mut out = String::new();
        let mut terms_enum = terms.iterator().map_err(|e| export_failed(&e))?;
        while let Some(term) = terms_enum.next().map_err(|e| export_failed(&e))? {
            let text = String::from_utf8_lossy(term).into_owned();
            let doc_freq = terms_enum.doc_freq().map_err(|e| export_failed(&e))?;
            out.push_str(&format!("{text}{delimiter}{doc_freq}\n"));
        }
        std::fs::write(&path, out).map_err(|e| export_failed(&e))?;
        Ok(path.to_string_lossy().into_owned())
    }
}

/// `IndexWriter.close()` under `commitOnClose`: a commit only when something
/// is pending, as Java's `commitInternal` skips one with no changes. The
/// flows above leave nothing pending -- each commits explicitly, a force
/// merge commits each merge, and this writer's commit already carries the
/// merges it triggers -- so it commits only what a flow left behind.
fn close(mut writer: IndexWriter<'_>) -> lucene_index::index_writer::Result<()> {
    if writer.has_uncommitted_changes() {
        writer.commit()?;
    }
    writer.close()
}

/// `new LukeException(message, cause)`: `getMessage()` is the message
/// alone.
fn failed(message: &str, _cause: &dyn std::fmt::Display) -> LukeError {
    LukeError::luke(message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_index::document::{Store, TextField};
    use lucene_store::directory::FsDirectory;
    use lucene_util::test_support::TempDir;

    fn doc(text: &str) -> Document {
        let mut doc = Document::new();
        doc.add(TextField::new("body", text, Store::Yes));
        doc
    }

    fn whitespace() -> Arc<Analyzer> {
        Arc::new(
            spi::new_analyzer("org.apache.lucene.analysis.core.WhitespaceAnalyzer")
                .expect("built in"),
        )
    }

    fn commits(dir: &FsDirectory) -> Vec<String> {
        let mut names: Vec<String> = dir
            .list_all()
            .unwrap()
            .into_iter()
            .filter(|n| n.starts_with("segments_"))
            .collect();
        names.sort();
        names
    }

    fn open(dir: &FsDirectory) -> DirectoryReader {
        DirectoryReader::open(dir as &dyn Directory).unwrap()
    }

    /// `keepAllCommits` keeps every commit (`NoDeletionPolicy`); without it
    /// only the last survives.
    #[test]
    fn keep_all_commits_keeps_every_commit() {
        let tmp = TempDir::new("luke-tools-keep");
        let dir = FsDirectory::open(tmp.path());
        IndexTools::new(&dir, false, false)
            .create_new_index(None)
            .unwrap();
        assert_eq!(commits(&dir), ["segments_1"]);
        let reader = open(&dir);
        let tools = IndexTools::with_reader(&reader, &dir, true, true);
        tools.add_document(&doc("a b"), whitespace()).unwrap();
        tools.add_document(&doc("c"), whitespace()).unwrap();
        assert_eq!(commits(&dir), ["segments_1", "segments_2", "segments_3"]);

        let tools = IndexTools::with_reader(&reader, &dir, false, false);
        tools.add_document(&doc("d"), whitespace()).unwrap();
        assert_eq!(commits(&dir), ["segments_4"]);
    }

    /// `close` commits what is pending, and nothing when nothing is.
    #[test]
    fn close_commits_only_pending_changes() {
        let tmp = TempDir::new("luke-tools-close");
        let dir = FsDirectory::open(tmp.path());
        let tools = IndexTools::new(&dir, false, false);
        let mut writer = tools.writer(None).unwrap();
        writer.add_fields_document(&doc("x")).unwrap();
        close(writer).unwrap();
        assert_eq!(commits(&dir), ["segments_1"]);
        close(tools.writer(None).unwrap()).unwrap();
        assert_eq!(commits(&dir), ["segments_1"]);
    }

    #[test]
    fn failures_carry_lukes_messages() {
        let tmp = TempDir::new("luke-tools-fail");
        let dir = FsDirectory::open(tmp.path());
        let tools = IndexTools::new(&dir, false, false);
        let not_reader = |e: LukeError| e.to_string();
        assert_eq!(
            not_reader(tools.optimize(false, 1).unwrap_err()),
            "LukeException: Current reader is not a DirectoryReader."
        );
        let not_instance = "LukeException: Current reader is not an instance of DirectoryReader.";
        assert_eq!(
            not_reader(tools.add_document(&doc("y"), whitespace()).unwrap_err()),
            not_instance
        );
        let term = lucene_index::buffered_updates::Term::new("body", b"y".to_vec());
        assert_eq!(
            not_reader(tools.delete_documents(DeleteQuery::Term(term)).unwrap_err()),
            not_instance
        );
        let e = tools.export_terms(".", "body", ",").unwrap_err();
        assert_eq!(e.to_string(), "LukeException: Current reader is not set.");

        tools.create_new_index(None).unwrap();
        let reader = open(&dir);
        let tools = IndexTools::with_reader(&reader, &dir, false, false);
        let e = tools.optimize(false, 0).unwrap_err();
        assert_eq!(
            e.to_string(),
            "IllegalArgumentException: maxNumSegments must be >= 1; got 0"
        );
        tools.optimize(true, 0).unwrap();
        let missing = tmp.path().join("no-such-dir");
        let e = tools
            .export_terms(&missing.to_string_lossy(), "body", ",")
            .unwrap_err();
        assert!(
            e.to_string()
                .starts_with("LukeException: Field body does not contain any terms"),
            "{e}"
        );
        tools.add_document(&doc("y"), whitespace()).unwrap();
        let reader = open(&dir);
        let tools = IndexTools::with_reader(&reader, &dir, false, false);
        let e = tools
            .export_terms(&missing.to_string_lossy(), "body", ",")
            .unwrap_err();
        assert!(
            e.to_string().starts_with(
                "LukeException: Terms file export for field [body] to file [terms_body_"
            ),
            "{e}"
        );
    }
}
