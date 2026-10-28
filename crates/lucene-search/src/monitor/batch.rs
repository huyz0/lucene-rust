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
use crate::monitor::index::BatchDocFreqs;
use crate::similarities::Similarity;
use crate::{Error, Result};

/// One document of a batch: its `(field, text)` values.
pub type DocFields<'a> = &'a [(&'a str, &'a str)];

/// A field's distinct analysed terms, as the presearcher is asked about them.
pub type FieldTerms = Vec<(String, Vec<Vec<u8>>)>;

/// [`FieldTerms`]' document frequencies, field by field and term by term.
type DocFreqs = Vec<Vec<usize>>;

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
    doc_freqs: DocFreqs,
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
        let (kind, terms, doc_freqs) = if docs.len() == 1 {
            let mut mi = MemoryIndex::with_options(true, true);
            if let Some(s) = &similarity {
                mi.set_similarity(Arc::clone(s))?;
            }
            for (field, value) in docs[0] {
                mi.add_field_text(field, value, analyzer)?;
            }
            mi.freeze()?;
            // the index's own terms dictionary: the document analysed once,
            // as Java reads the batch's terms from the batch reader
            let terms = mi.field_terms();
            let doc_freqs = terms.iter().map(|(_, t)| vec![1; t.len()]).collect();
            (Kind::Single(mi.create_reader()?), terms, doc_freqs)
        } else {
            let (terms, doc_freqs) = analysed_terms(analyzer, docs)?;
            (multi(analyzer, docs)?, terms, doc_freqs)
        };
        Ok(DocumentBatch {
            kind,
            similarity,
            max_doc: docs.len(),
            terms,
            doc_freqs,
        })
    }

    /// `maxDoc()`: the number of documents.
    pub fn max_doc(&self) -> usize {
        self.max_doc
    }

    /// Every field's distinct analysed terms across the batch, each field's
    /// sorted.
    pub fn terms(&self) -> &FieldTerms {
        &self.terms
    }

    /// The named fields' distinct terms with the number of batch documents
    /// holding each (`TermsEnum.docFreq` over the batch reader); a field the
    /// batch does not hold is left out.
    pub fn doc_freqs(&self, fields: &[String]) -> BatchDocFreqs {
        let mut out = BatchDocFreqs::new();
        for ((field, terms), freqs) in self.terms.iter().zip(&self.doc_freqs) {
            if fields.contains(field) {
                out.insert(
                    field.clone(),
                    terms.iter().cloned().zip(freqs.iter().copied()).collect(),
                );
            }
        }
        out
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

/// Each field's distinct terms across `docs`, sorted, as the analyzer yields
/// them, and the number of documents holding each.
fn analysed_terms(analyzer: &Analyzer, docs: &[DocFields<'_>]) -> Result<(FieldTerms, DocFreqs)> {
    let analysis = |e: lucene_analysis::AnalysisError| Error::IllegalArgument(e.to_string());
    // every field's terms, each document's distinct ones once
    let mut all: FieldTerms = Vec::new();
    for doc in docs {
        let mut own: FieldTerms = Vec::new();
        for (field, value) in *doc {
            let slot = match own.iter().position(|(f, _)| f == field) {
                Some(i) => i,
                None => {
                    own.push(((*field).to_string(), Vec::new()));
                    own.len() - 1
                }
            };
            let terms = &mut own[slot].1;
            let mut ts = analyzer.token_stream(field, value).map_err(analysis)?;
            ts.reset().map_err(analysis)?;
            while ts.increment_token().map_err(analysis)? {
                terms.push(ts.attributes().term_bytes().to_vec());
            }
            ts.end().map_err(analysis)?;
            ts.close().map_err(analysis)?;
        }
        for (field, mut terms) in own {
            terms.sort_unstable();
            terms.dedup();
            match all.iter_mut().find(|(f, _)| *f == field) {
                Some((_, t)) => t.extend(terms),
                None => all.push((field, terms)),
            }
        }
    }
    let mut freqs = Vec::with_capacity(all.len());
    for (_, terms) in &mut all {
        terms.sort_unstable();
        let mut f: Vec<usize> = Vec::new();
        let mut distinct: Vec<Vec<u8>> = Vec::new();
        for t in terms.drain(..) {
            match distinct.last() {
                Some(last) if *last == t => *f.last_mut().expect("a count per term") += 1,
                _ => {
                    distinct.push(t);
                    f.push(1);
                }
            }
        }
        *terms = distinct;
        freqs.push(f);
    }
    Ok((all, freqs))
}
