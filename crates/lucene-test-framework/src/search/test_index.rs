//! A small index written to memory and searched: the corpus the search
//! test bases (`SearchEquivalenceTestBase`, `BaseExplanationTestCase`, ...)
//! build in their `beforeClass`.
//!
//! Java writes it through `RandomIndexWriter`, whose random flushes,
//! merges and codec this port does not reproduce; here every segment is
//! one commit of the documents given, under a plain `IndexWriter`.

use std::sync::Arc;

use lucene_analysis::Analyzer;
use lucene_index::buffered_updates::Term;
use lucene_index::document::Document;
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::index_searcher::{IndexSearcher, SegmentNorms};
use lucene_search::similarities::Similarity;
use lucene_store::ByteBuffersDirectory;

const VERSION: LuceneVersion = LuceneVersion {
    major: 10,
    minor: 5,
    bugfix: 0,
};

/// An index in memory, with the fields whose norms its searchers load.
pub struct TestIndex {
    dir: ByteBuffersDirectory,
    norm_fields: Vec<String>,
}

impl TestIndex {
    /// Writes `segments` (each one commit) analysed by `analyzer`, then
    /// deletes the documents `delete` names by term (`(field, term)`), in
    /// one last commit. `norm_fields` are the fields scored with norms.
    ///
    /// # Panics
    /// When writing the index fails.
    pub fn build(
        analyzer: Arc<Analyzer>,
        segments: &[Vec<Document>],
        delete: &[(&str, &str)],
        norm_fields: &[&str],
    ) -> Self {
        let dir = ByteBuffersDirectory::new();
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).expect("open writer");
        w.set_analyzer(Some(analyzer));
        for docs in segments {
            for d in docs {
                w.add_fields_document(d).expect("add document");
            }
            w.commit().expect("commit");
        }
        if !delete.is_empty() {
            let terms: Vec<Term> = delete.iter().map(|&(f, t)| Term::new(f, t)).collect();
            w.delete_documents_by_term(&terms).expect("delete");
            w.commit().expect("commit deletes");
        }
        w.close().expect("close writer");
        Self {
            dir,
            norm_fields: norm_fields.iter().map(|f| (*f).to_string()).collect(),
        }
    }

    /// Runs `f` over a searcher of the index.
    ///
    /// # Panics
    /// When opening the index fails.
    pub fn with_searcher<R>(&self, f: impl FnOnce(&IndexSearcher<'_, '_>) -> R) -> R {
        self.with_searcher_similarity(None, f)
    }

    /// [`Self::with_searcher`] with `setSimilarity(similarity)` (`None`:
    /// the default BM25).
    ///
    /// # Panics
    /// When opening the index fails.
    pub fn with_searcher_similarity<R>(
        &self,
        similarity: Option<&dyn Similarity>,
        f: impl FnOnce(&IndexSearcher<'_, '_>) -> R,
    ) -> R {
        let reader = DirectoryReader::open(&self.dir).expect("open reader");
        let opened = reader.open_segments().expect("open segments");
        let segments = opened.as_open_segments();
        let owned = reader.field_norms_by_field(&self.norm_fields);
        let norms: Vec<SegmentNorms<'_, '_>> = owned.iter().map(Some).collect();
        let mut searcher = IndexSearcher::new(&segments, &norms).expect("searcher");
        if let Some(sim) = similarity {
            searcher.set_similarity(sim);
        }
        f(&searcher)
    }
}
