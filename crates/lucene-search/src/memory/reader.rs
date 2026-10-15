//! `MemoryIndex.MemoryIndexReader`: the one document, searchable.
//!
//! Java's reader serves the `Info` structures through the `LeafReader` API;
//! this port's queries execute on codec structures
//! ([`crate::multi_segment::OpenSegment`]), so the document is encoded with
//! the codec's own writers and searched as a segment. Two ways:
//!
//! - **in memory** ([`MemorySegment`]), when the document is postings and
//!   norms only: the postings writer's `.doc`/`.pos`/`.pay` and terms
//!   dictionary bytes and the norms writer's `.nvd`, opened in place -- no
//!   directory, no segment or field-infos files, no stored fields;
//! - **as a segment** of a [`ByteBuffersDirectory`], through the writer's
//!   explicit-document path and a [`DirectoryReader`]: when the document has
//!   doc values or points, which the in-memory form does not carry, or when a
//!   query needs a segment reader (a [`crate::Error::MissingSegmentReader`]
//!   from the in-memory form) -- built on first need and kept.
//!
//! Either way the searcher sees what Java's reader reports: `maxDoc` 1,
//! every term `docFreq` 1, `docCount` 1 for a field with terms, and the norm
//! `Info` computed.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use lucene_codecs::blocktree::{self, BlockTreeFields};
use lucene_codecs::field_infos::{FieldInfo, FieldInfos, IndexOptions};
use lucene_codecs::norms::{self, NormsEntry, NormsField};
use lucene_codecs::postings::{DocInput, PayInput, PosInput};
use lucene_codecs::postings_writer::{self, FieldPostingsInput, TermPostings};
use lucene_codecs::stored_fields::StoredField;
use lucene_index::index_writer::{
    ExplicitDocument, ExplicitFields, IndexWriter, InvertedField, InvertedTerm,
};
use lucene_index::merge_policy::upgrade::LATEST;
use lucene_store::byte_buffers_directory::ByteBuffersDirectory;

use super::illegal;
use crate::directory_reader::DirectoryReader;
use crate::field_norms::FieldNorms;
use crate::index_searcher::{IndexSearcher, SegmentNorms};
use crate::multi_segment::OpenSegment;
use crate::query::{BooleanQuery, Clause};
use crate::similarities::Similarity;
use crate::{Error, Result};

/// The segment id the in-memory files are written and checked with.
const SEGMENT_ID: [u8; 16] = [0; 16];

/// One field of [`Document`]: its schema, postings (term order) and norm.
#[derive(Debug, Clone)]
pub(super) struct FieldPostings {
    pub(super) info: FieldInfo,
    pub(super) has_payloads: bool,
    pub(super) terms: Vec<TermPostings>,
    pub(super) norm: Option<i64>,
}

/// The document as [`super::MemoryIndex`] hands it over: every field, in
/// Java's numbering, and the values the in-memory form does not carry.
#[derive(Debug, Clone, Default)]
pub(super) struct Document {
    pub(super) fields: Vec<FieldPostings>,
    pub(super) doc_values: Vec<StoredField>,
    pub(super) points: Vec<StoredField>,
    pub(super) stored: Vec<StoredField>,
}

impl Document {
    /// The writer's explicit form of the document.
    fn explicit(&self) -> ExplicitDocument {
        let mut fields = ExplicitFields::default();
        for f in &self.fields {
            let terms = f
                .terms
                .iter()
                .map(|t| {
                    let mut payloads = Vec::new();
                    if f.has_payloads {
                        let mut at = 0usize;
                        for &len in &t.payload_lengths {
                            let end = at.saturating_add(len as usize).min(t.payload_bytes.len());
                            payloads.push(t.payload_bytes[at..end].to_vec());
                            at = end;
                        }
                    }
                    InvertedTerm {
                        term: t.term.clone(),
                        freq: t.docs[0].1,
                        positions: t.positions[0].clone(),
                        offsets: t.offsets[0].clone(),
                        payloads,
                    }
                })
                .collect();
            fields.inverted.push(InvertedField {
                field_number: f.info.number,
                terms,
                norm: f.norm,
            });
        }
        fields.doc_values = self.doc_values.clone();
        fields.points = self.points.clone();
        ExplicitDocument {
            stored: self.stored.clone(),
            fields,
        }
    }
}

/// One field's norm in [`MemorySegment`]: its `.nvm` entry, and the
/// `sumTotalTermFreq`/`docCount` its BM25 average length comes from.
#[derive(Debug)]
struct NormsOf {
    field: String,
    entry: NormsEntry,
    sum_total_term_freq: i64,
    doc_count: i32,
}

/// The document's postings and norms, encoded and opened in memory.
#[derive(Debug)]
struct MemorySegment {
    terms: BlockTreeFields,
    doc: Vec<u8>,
    pos: Vec<u8>,
    pay: Vec<u8>,
    nvd: Vec<u8>,
    norms: Vec<NormsOf>,
}

impl MemorySegment {
    /// Encodes `doc`'s postings and norms: `None` when no field has terms.
    fn build(doc: &Document) -> Result<Option<Self>> {
        let codec_error = |e: &dyn std::fmt::Display| illegal(e.to_string());
        let mut fields: Vec<FieldInfo> = Vec::with_capacity(doc.fields.len());
        let mut inputs: Vec<(i32, IndexOptions, bool, &[TermPostings])> = Vec::new();
        let mut norm_values: Vec<(i32, [i64; 1])> = Vec::new();
        for f in &doc.fields {
            let mut info = f.info.clone();
            info.store_payloads = f.has_payloads;
            fields.push(info);
            if f.terms.is_empty() {
                continue;
            }
            inputs.push((
                f.info.number,
                f.info.index_options,
                f.has_payloads,
                &f.terms,
            ));
            if let Some(n) = f.norm {
                norm_values.push((f.info.number, [n]));
            }
        }
        if inputs.is_empty() {
            return Ok(None);
        }
        let postings_inputs: Vec<FieldPostingsInput<'_>> = inputs
            .iter()
            .map(|&(n, options, payloads, terms)| FieldPostingsInput {
                field_number: n,
                index_options: options,
                doc_count: 1,
                has_payloads: payloads,
                terms,
            })
            .collect();
        let impacts: Vec<postings_writer::FieldNorms<'_>> = norm_values
            .iter()
            .map(|(n, v)| postings_writer::FieldNorms {
                field_number: *n,
                values: v,
            })
            .collect();
        let output =
            postings_writer::write_fields_with_norms(&postings_inputs, &impacts, &SEGMENT_ID, "")
                .map_err(|e| codec_error(&e))?;
        let field_infos = FieldInfos::new(fields).map_err(|e| codec_error(&e))?;
        let terms = blocktree::open_shared(
            Arc::new(output.tim),
            Arc::new(output.tip),
            &output.tmd,
            &field_infos,
            &SEGMENT_ID,
            "",
            1,
        )?;
        let (nvd, norms) = if norm_values.is_empty() {
            (Vec::new(), Vec::new())
        } else {
            let columns: Vec<NormsField<'_>> = norm_values
                .iter()
                .map(|(n, v)| NormsField::Dense(*n, v))
                .collect();
            let (nvm, nvd) =
                norms::write_fields(&columns, 1, &SEGMENT_ID, "").map_err(|e| codec_error(&e))?;
            let (_, meta) = norms::parse_meta(&nvm, &SEGMENT_ID, "")?;
            let mut norms = Vec::with_capacity(norm_values.len());
            for (n, _) in &norm_values {
                let name = &doc
                    .fields
                    .iter()
                    .find(|f| f.info.number == *n)
                    .ok_or_else(|| illegal("a norm without a field info"))?
                    .info
                    .name;
                let (Some(entry), Some(field_terms)) = (meta.entry(*n), terms.field(name)) else {
                    continue;
                };
                norms.push(NormsOf {
                    field: name.clone(),
                    entry: *entry,
                    sum_total_term_freq: field_terms.sum_total_term_freq,
                    doc_count: field_terms.doc_count,
                });
            }
            (nvd, norms)
        };
        Ok(Some(MemorySegment {
            terms,
            doc: output.doc,
            pos: output.pos,
            pay: output.pay,
            nvd,
            norms,
        }))
    }

    /// A searcher over the segment, handed to `f`.
    fn with_searcher<R>(
        &self,
        similarity: &dyn Similarity,
        f: impl FnOnce(&IndexSearcher<'_, '_>) -> R,
    ) -> Result<R> {
        let doc_in = DocInput::open(&self.doc, &SEGMENT_ID, "")?;
        let pos_in = match self.pos.is_empty() {
            true => None,
            false => Some(PosInput::open(&self.pos, &SEGMENT_ID, "")?),
        };
        let pay_in = match self.pay.is_empty() {
            true => None,
            false => Some(PayInput::open(&self.pay, &SEGMENT_ID, "")?),
        };
        let norms: HashMap<String, FieldNorms<'_>> = self
            .norms
            .iter()
            .map(|n| {
                (
                    n.field.clone(),
                    FieldNorms::from_field_stats(
                        &self.nvd,
                        n.entry,
                        n.sum_total_term_freq,
                        n.doc_count,
                    ),
                )
            })
            .collect();
        let segments = [OpenSegment {
            fields: &self.terms,
            doc_in: Some(&doc_in),
            pos_in: pos_in.as_ref(),
            pay_in: pay_in.as_ref(),
            live_docs: None,
            doc_base: 0,
            max_doc: Some(1),
            cache: None,
            points: None,
            reader: None,
            index_sort_prefix: false,
        }];
        let segment_norms: [SegmentNorms<'_, '_>; 1] = [Some(&norms)];
        let mut searcher = IndexSearcher::new(&segments, &segment_norms)?;
        searcher.set_similarity(similarity);
        Ok(f(&searcher))
    }
}

/// `MemoryIndex.MemoryIndexReader`: the one document, and the similarity
/// its searcher scores with. See the module docs.
#[derive(Debug)]
pub struct MemoryIndexReader {
    /// The in-memory form, unless the document needs the segment.
    memory: Option<MemorySegment>,
    /// What [`Self::segment`] writes.
    doc: Document,
    segment: OnceLock<DirectoryReader>,
    similarity: Arc<dyn Similarity>,
}

impl MemoryIndexReader {
    pub(super) fn new(doc: Document, similarity: Arc<dyn Similarity>) -> Result<Self> {
        let needs_segment = !doc.doc_values.is_empty() || !doc.points.is_empty();
        let memory = if needs_segment {
            None
        } else {
            MemorySegment::build(&doc)?
        };
        let reader = MemoryIndexReader {
            memory,
            doc,
            segment: OnceLock::new(),
            similarity,
        };
        if reader.memory.is_none() {
            reader.segment()?;
        }
        Ok(reader)
    }

    /// The document written to a [`ByteBuffersDirectory`] and opened, on
    /// first need.
    fn segment(&self) -> Result<&DirectoryReader> {
        if let Some(r) = self.segment.get() {
            return Ok(r);
        }
        let index_error = |e: lucene_index::index_writer::Error| illegal(e.to_string());
        let dir = ByteBuffersDirectory::new();
        {
            let mut writer =
                IndexWriter::open(&dir, Vec::new(), "Lucene104", LATEST).map_err(index_error)?;
            writer.set_use_compound_file(false);
            writer.enable_explicit_documents().map_err(index_error)?;
            for f in &self.doc.fields {
                writer.register_field(f.info.clone()).map_err(index_error)?;
            }
            writer
                .add_explicit_documents(vec![self.doc.explicit()])
                .map_err(index_error)?;
            writer.commit().map_err(index_error)?;
        }
        let reader = DirectoryReader::open(&dir)?;
        Ok(self.segment.get_or_init(|| reader))
    }

    /// The document as a segment, for statistics and the reader-level APIs.
    pub fn directory_reader(&self) -> Result<&DirectoryReader> {
        self.segment()
    }

    /// `createSearcher()`: a searcher over the document, scoring with the
    /// index's similarity, handed to `f`.
    pub fn with_searcher<R>(&self, f: impl FnOnce(&IndexSearcher<'_, '_>) -> R) -> Result<R> {
        if let Some(m) = &self.memory {
            return m.with_searcher(self.similarity.as_ref(), f);
        }
        self.with_segment_searcher(f)
    }

    fn with_segment_searcher<R>(&self, f: impl FnOnce(&IndexSearcher<'_, '_>) -> R) -> Result<R> {
        let reader = self.segment()?;
        let mut opened = reader.open_segments()?;
        opened.open_points()?;
        let segments = opened.as_open_segments();
        let names: Vec<String> = self
            .doc
            .fields
            .iter()
            .map(|f| f.info.name.clone())
            .collect();
        let owned = reader.field_norms_by_field(&names);
        let norms: Vec<SegmentNorms<'_, '_>> = owned.iter().map(Some).collect();
        let mut searcher = IndexSearcher::new(&segments, &norms)?;
        searcher.set_similarity(self.similarity.as_ref());
        Ok(f(&searcher))
    }

    /// `MemoryIndex.search(query)` over this reader.
    pub fn search(&self, query: &Clause) -> Result<f32> {
        let Some(bq) = as_boolean(query) else {
            return Ok(0.0);
        };
        if let Some(m) = &self.memory {
            match m.with_searcher(self.similarity.as_ref(), |s| s.search(&bq, 1))? {
                Ok(top) => return Ok(score(top)),
                // a clause that reads what only a segment reader has
                Err(Error::MissingSegmentReader(_) | Error::MissingPointsInput(_)) => {}
                Err(e) => return Err(e),
            }
        }
        self.search_segment(&bq)
    }

    /// [`Self::search`] on the segment, whatever the in-memory form would
    /// say.
    pub(super) fn search_segment(&self, bq: &BooleanQuery) -> Result<f32> {
        self.with_segment_searcher(|s| s.search(bq, 1))?.map(score)
    }
}

/// The query as the boolean a searcher runs: `None` for one that matches
/// nothing.
pub(super) fn as_boolean(query: &Clause) -> Option<BooleanQuery> {
    match query {
        Clause::MatchNoDocs(_) => None,
        Clause::Boolean(b) => Some((**b).clone()),
        other => {
            let mut b = BooleanQuery::new();
            b.must.push(other.clone());
            Some(b)
        }
    }
}

/// The one document's score, `0.0` when it does not match.
fn score(top: crate::top_docs::TopDocs) -> f32 {
    top.score_docs.first().map_or(0.0, |d| d.score)
}
