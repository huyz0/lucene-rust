//! `org.apache.lucene.monitor.DocumentBatch`: the documents a monitor matches
//! queries against, as one searchable segment.
//!
//! A single document is a `MemoryIndex` with offsets and payloads
//! (`SingletonDocumentBatch`); several are written through an `IndexWriter`
//! and merged to one segment (`MultiDocumentBatch`), so a batch's statistics
//! -- and so its scores -- are the batch's own.

use std::sync::Arc;

use lucene_analysis::{Analyzer, TokenStream};
use lucene_index::document::{Document, Store, TextField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::merge_policy::upgrade::LATEST;
use lucene_store::byte_buffers_directory::ByteBuffersDirectory;

use crate::directory_reader::DirectoryReader;
use crate::index_searcher::{IndexSearcher, SegmentNorms};
use crate::memory::{MemoryIndex, MemoryIndexReader};
use crate::similarities::Similarity;
use crate::{Error, Result};

/// One document of a batch: its `(field, text)` values.
pub type DocFields<'a> = &'a [(&'a str, &'a str)];

/// A field's distinct analysed terms, as the presearcher is asked about them.
pub type FieldTerms = Vec<(String, Vec<Vec<u8>>)>;

enum Kind {
    Single(Arc<MemoryIndexReader>),
    Multi {
        reader: Box<DirectoryReader>,
        fields: Vec<String>,
    },
}

/// `DocumentBatch`.
pub struct DocumentBatch {
    kind: Kind,
    similarity: Option<Arc<dyn Similarity>>,
    max_doc: usize,
    terms: FieldTerms,
}

impl DocumentBatch {
    /// `DocumentBatch.of(analyzer, docs)`. `similarity` is what a scoring
    /// matcher scores with (`searcher.setSimilarity`); `None` is BM25.
    ///
    /// # Errors
    /// An empty batch, as Java refuses one.
    pub fn of(
        analyzer: &Arc<Analyzer>,
        docs: &[DocFields<'_>],
        similarity: Option<Arc<dyn Similarity>>,
    ) -> Result<Self> {
        if docs.is_empty() {
            return Err(Error::IllegalArgument(
                "A DocumentBatch must contain at least one document".into(),
            ));
        }
        let terms = analysed_terms(analyzer, docs)?;
        let kind = if docs.len() == 1 {
            let mut mi = MemoryIndex::with_options(true, true);
            if let Some(s) = &similarity {
                mi.set_similarity(Arc::clone(s))?;
            }
            for (field, value) in docs[0] {
                mi.add_field_text(field, value, analyzer)?;
            }
            mi.freeze()?;
            Kind::Single(mi.create_reader()?)
        } else {
            multi(analyzer, docs)?
        };
        Ok(DocumentBatch {
            kind,
            similarity,
            max_doc: docs.len(),
            terms,
        })
    }

    /// `maxDoc()`: the number of documents.
    pub fn max_doc(&self) -> usize {
        self.max_doc
    }

    /// Every field's distinct analysed terms across the batch, in field
    /// order of first appearance.
    pub fn terms(&self) -> &FieldTerms {
        &self.terms
    }

    /// The batch's searcher, handed to `f`.
    pub fn with_searcher<R>(&self, f: impl FnOnce(&IndexSearcher<'_, '_>) -> R) -> Result<R> {
        match &self.kind {
            Kind::Single(reader) => reader.with_searcher(f),
            Kind::Multi { reader, fields } => {
                let mut opened = reader.open_segments()?;
                opened.open_points()?;
                let segments = opened.as_open_segments();
                let owned = reader.field_norms_by_field(fields)?;
                let norms: Vec<SegmentNorms<'_, '_>> = owned.iter().map(Some).collect();
                let mut searcher = IndexSearcher::new(&segments, &norms)?;
                if let Some(s) = &self.similarity {
                    searcher.set_similarity(s.as_ref());
                }
                Ok(f(&searcher))
            }
        }
    }
}

/// `MultiDocumentBatch`: the documents in one `IndexWriter` block, committed
/// and force-merged to a single segment.
fn multi(analyzer: &Arc<Analyzer>, docs: &[DocFields<'_>]) -> Result<Kind> {
    let index_error = |e: lucene_index::index_writer::Error| Error::IllegalArgument(e.to_string());
    let dir = ByteBuffersDirectory::new();
    let mut fields: Vec<String> = Vec::new();
    {
        let mut writer =
            IndexWriter::open(&dir, Vec::new(), "Lucene104", LATEST).map_err(index_error)?;
        writer.set_use_compound_file(false);
        writer.set_analyzer(Some(Arc::clone(analyzer)));
        let mut documents = Vec::with_capacity(docs.len());
        for doc in docs {
            let mut d = Document::new();
            for (field, value) in *doc {
                if !fields.iter().any(|f| f == field) {
                    fields.push((*field).to_string());
                }
                d.add(TextField::new(*field, *value, Store::No));
            }
            documents.push(d);
        }
        writer
            .add_fields_documents(&documents)
            .map_err(index_error)?;
        writer.commit().map_err(index_error)?;
        writer.force_merge(1).map_err(index_error)?;
    }
    let reader = DirectoryReader::open(&dir)?;
    Ok(Kind::Multi {
        reader: Box::new(reader),
        fields,
    })
}

/// Each field's distinct terms across `docs`, as the analyzer yields them.
fn analysed_terms(analyzer: &Analyzer, docs: &[DocFields<'_>]) -> Result<FieldTerms> {
    let analysis = |e: lucene_analysis::AnalysisError| Error::IllegalArgument(e.to_string());
    let mut out: FieldTerms = Vec::new();
    for doc in docs {
        for (field, value) in *doc {
            let mut ts = analyzer.token_stream(field, value).map_err(analysis)?;
            ts.reset().map_err(analysis)?;
            let mut terms = Vec::new();
            while ts.increment_token().map_err(analysis)? {
                terms.push(ts.attributes().term_bytes().to_vec());
            }
            ts.end().map_err(analysis)?;
            ts.close().map_err(analysis)?;
            let slot = match out.iter().position(|(f, _)| f == field) {
                Some(i) => i,
                None => {
                    out.push(((*field).to_string(), Vec::new()));
                    out.len() - 1
                }
            };
            for t in terms {
                if !out[slot].1.contains(&t) {
                    out[slot].1.push(t);
                }
            }
        }
    }
    Ok(out)
}
