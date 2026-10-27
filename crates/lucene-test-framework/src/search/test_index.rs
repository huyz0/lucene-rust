//! A small index written to memory and searched: the corpus the search
//! test bases (`SearchEquivalenceTestBase`, `BaseExplanationTestCase`, ...)
//! build in their `beforeClass`.
//!
//! Java writes it through `RandomIndexWriter`: [`TestIndex::write_random`]
//! does too, and keeps the reader its `getReader()` returns.
//! [`TestIndex::build`] is the fixed layout the differential fixtures
//! replay: every segment one commit of the documents given, under a plain
//! `IndexWriter`.

use std::sync::Arc;

use lucene_analysis::Analyzer;
use lucene_index::buffered_updates::Term;
use lucene_index::document::Document;
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::index_searcher::{IndexSearcher, SegmentNorms};
use lucene_search::multi_segment::OpenSegment;
use lucene_search::reader::{IndexReader, LeafReader};
use lucene_search::similarities::Similarity;
use lucene_store::ByteBuffersDirectory;
use lucene_util::java_random::JavaRandom;

use crate::index::index_writer_config::RandomIndexWriterConfig;
use crate::index::random_index_writer::{RandomIndexWriter, WriterEnv};

const VERSION: LuceneVersion = LuceneVersion {
    major: 10,
    minor: 5,
    bugfix: 0,
};

/// An index in memory, with the fields whose norms its searchers load.
pub struct TestIndex {
    dir: ByteBuffersDirectory,
    norm_fields: Vec<String>,
    /// The reader `getReader()` returned, for an index written at random;
    /// otherwise each searcher opens the last commit.
    reader: Option<DirectoryReader>,
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
            reader: None,
        }
    }

    /// `new RandomIndexWriter(random, dir, analyzer)`, then `write`, then
    /// `getReader()` and `close()`: the index the test bases' `beforeClass`
    /// write, searched through that reader; `write` gets the writer and
    /// `random`. Only the writer's own draws come from `random`, as Java's
    /// do from the `Random` it is handed: its environment (Java's test
    /// `random()`, which Java's `beforeClass` also hands it, and the class
    /// rule's flush mode) is a stream seeded from a copy of `random`, so
    /// drawing it leaves `random` where Java's draws from `r` leave it.
    ///
    /// # Panics
    /// When writing the index fails.
    pub fn write_random(
        random: &mut JavaRandom,
        analyzer: Arc<Analyzer>,
        norm_fields: &[&str],
        write: impl FnOnce(&mut RandomIndexWriter<'_>, &mut JavaRandom),
    ) -> Self {
        Self::write_random_with(
            random,
            |random, dir, env| RandomIndexWriter::with_analyzer(random, dir, analyzer, env),
            norm_fields,
            write,
        )
    }

    /// [`Self::write_random`] with the writer `open` makes (`new
    /// RandomIndexWriter(random, dir, config)`), handed `random`, the
    /// directory and the writer's environment.
    ///
    /// # Panics
    /// When writing the index fails.
    pub fn write_random_with(
        random: &mut JavaRandom,
        open: impl for<'d> FnOnce(
            &mut JavaRandom,
            &'d ByteBuffersDirectory,
            WriterEnv,
        )
            -> lucene_search::directory_reader::Result<RandomIndexWriter<'d>>,
        norm_fields: &[&str],
        write: impl FnOnce(&mut RandomIndexWriter<'_>, &mut JavaRandom),
    ) -> Self {
        let dir = ByteBuffersDirectory::new();
        let reader = {
            let env = WriterEnv::new(random.clone().next_long());
            let mut w = open(random, &dir, env).expect("open writer");
            write(&mut w, random);
            let reader = w.get_reader().expect("getReader");
            w.close().expect("close writer");
            reader
        };
        Self {
            dir,
            norm_fields: norm_fields.iter().map(|f| (*f).to_string()).collect(),
            reader: Some(reader),
        }
    }

    /// `new IndexWriter(dir, config)`, then `write`, then `close()`: an
    /// index written by a plain writer under a random configuration (Java's
    /// `DirectoryReader.open(w)` before `close()` sees the same documents
    /// the closing commit holds), searched through its last commit.
    ///
    /// # Panics
    /// When writing the index fails.
    pub fn write_with(
        config: &RandomIndexWriterConfig,
        norm_fields: &[&str],
        write: impl FnOnce(&mut IndexWriter<'_>),
    ) -> Self {
        let dir = ByteBuffersDirectory::new();
        {
            let mut w =
                IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).expect("open writer");
            config.apply(&mut w).expect("apply the configuration");
            write(&mut w);
            w.close().expect("close writer");
        }
        Self {
            dir,
            norm_fields: norm_fields.iter().map(|f| (*f).to_string()).collect(),
            reader: None,
        }
    }

    /// Each document's `field` numeric doc value, by global doc id (`None`
    /// for a document without one).
    ///
    /// # Panics
    /// When reading the index fails.
    pub fn numeric_values(&self, field: &str) -> Vec<Option<i64>> {
        let opened_reader;
        let reader = match &self.reader {
            Some(r) => r,
            None => {
                opened_reader = DirectoryReader::open(&self.dir).expect("open reader");
                &opened_reader
            }
        };
        let mut values = Vec::new();
        for sub in reader.segment_readers() {
            let base = values.len();
            let max_doc = usize::try_from(sub.max_doc()).expect("a maxDoc");
            values.resize(base + max_doc, None);
            if let Some(mut dv) = sub.numeric_doc_values(field).expect("doc values") {
                loop {
                    let doc = dv.next_doc().expect("nextDoc");
                    let Ok(doc) = usize::try_from(doc) else { break };
                    if doc >= max_doc {
                        break;
                    }
                    values[base + doc] = Some(dv.long_value());
                }
            }
        }
        values
    }

    /// Runs `f` over the index's segments and their norms, from which
    /// searchers are made.
    ///
    /// # Panics
    /// When opening the index fails.
    pub fn with_segments<R>(
        &self,
        f: impl FnOnce(&[OpenSegment<'_>], &[SegmentNorms<'_, '_>]) -> R,
    ) -> R {
        let opened_reader;
        let reader = match &self.reader {
            Some(r) => r,
            None => {
                opened_reader = DirectoryReader::open(&self.dir).expect("open reader");
                &opened_reader
            }
        };
        let opened = reader.open_segments().expect("open segments");
        let segments = opened.as_open_segments();
        let owned = reader.field_norms_by_field(&self.norm_fields).unwrap();
        let norms: Vec<SegmentNorms<'_, '_>> = owned.iter().map(Some).collect();
        f(&segments, &norms)
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
        self.with_segments(|segments, norms| {
            let mut searcher = IndexSearcher::new(segments, norms).expect("searcher");
            if let Some(sim) = similarity {
                searcher.set_similarity(sim);
            }
            f(&searcher)
        })
    }
}
