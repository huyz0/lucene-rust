//! `org.apache.lucene.misc.index`: tools that split, rearrange and reorder an
//! index. Every tool that Java builds on `IndexWriter.addIndexes(CodecReader...)`
//! over a `FilterCodecReader` that hides documents is built here on
//! [`lucene_index::index_writer::IndexWriter::add_indexes_filtered`].

pub mod multi_pass;
pub mod pk;
pub mod rearranger;
pub mod splitter;

pub use multi_pass::MultiPassIndexSplitter;
pub use pk::PkIndexSplitter;
pub use rearranger::{BinaryDocValueSelector, DocumentSelector, IndexRearranger};
pub use splitter::IndexSplitter;

use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_store::directory::Directory;

use crate::collector::{ScoreMode, ScoringCollector};
use crate::directory_reader::DirectoryReader;
use crate::index_searcher::IndexSearcher;
use crate::query::BooleanQuery;
use crate::{Error, Result};

/// The version and codec the splitters' writers record: `new
/// IndexWriterConfig(null)` on Lucene 10.5.0's default codec.
const VERSION: LuceneVersion = LuceneVersion {
    major: 10,
    minor: 5,
    bugfix: 0,
};
const CODEC: &str = "Lucene104";

pub(crate) fn index_error(e: impl std::fmt::Display) -> Error {
    Error::IllegalArgument(e.to_string())
}

/// `new IndexWriter(dir, new IndexWriterConfig(null).setOpenMode(CREATE))`.
/// The output is expected to be empty: this opens the latest commit if there
/// is one, where `CREATE` would discard it.
pub(crate) fn open_writer(dir: &dyn Directory) -> Result<IndexWriter<'_>> {
    IndexWriter::open(dir, Vec::new(), CODEC, VERSION).map_err(index_error)
}

/// Per segment of `reader`, whether each document is live.
pub(crate) fn live_flags(reader: &DirectoryReader) -> Vec<Vec<bool>> {
    reader
        .segment_readers()
        .iter()
        .map(|s| {
            (0..s.max_doc)
                .map(|d| {
                    s.live_docs()
                        .is_none_or(|l| l.get(usize::try_from(d).unwrap_or(0)))
                })
                .collect()
        })
        .collect()
}

/// Per segment of `reader`, whether each document is a live match of `query`
/// (the documents `Weight.scorer(leaf)` iterates, less the deleted ones).
pub(crate) fn match_flags(
    reader: &DirectoryReader,
    query: &BooleanQuery,
) -> Result<Vec<Vec<bool>>> {
    struct Docs(Vec<i32>);
    impl ScoringCollector for Docs {
        fn collect(&mut self, doc: i32, _score: f32) {
            self.0.push(doc);
        }
        fn score_mode(&self) -> ScoreMode {
            ScoreMode::CompleteNoScores
        }
    }
    let opened = reader.open_segments()?;
    let segments = opened.as_open_segments();
    let owned = reader.field_norms_by_field(&[]);
    let norms: Vec<_> = owned.iter().map(Some).collect();
    let searcher = IndexSearcher::new(&segments, &norms)?;
    let mut docs = Docs(Vec::new());
    searcher.search_collector(query, &mut docs)?;
    let mut flags: Vec<Vec<bool>> = reader
        .segment_readers()
        .iter()
        .map(|s| vec![false; usize::try_from(s.max_doc).unwrap_or(0)])
        .collect();
    for doc in docs.0 {
        let (segment, local) = locate(reader, doc)?;
        flags[segment][local] = true;
    }
    Ok(flags)
}

/// `ReaderUtil.subIndex` and the leaf-local id of a global document.
fn locate(reader: &DirectoryReader, doc: i32) -> Result<(usize, usize)> {
    let mut base = 0i32;
    for (i, s) in reader.segment_readers().iter().enumerate() {
        if doc < base + s.max_doc {
            return Ok((i, usize::try_from(doc - base).unwrap_or(0)));
        }
        base += s.max_doc;
    }
    Err(Error::IllegalArgument(format!(
        "document {doc} is outside the reader"
    )))
}
