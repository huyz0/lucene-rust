//! `IndexingChain.processDocument` for the document API: one [`Document`]
//! of [`IndexableField`]s becomes one [`ExplicitDocument`].
//!
//! The steps are Java's, in Java's order:
//!
//! 1. every field updates its name's per-document schema
//!    (`updateDocFieldSchema`, `FieldSchema.set*`), which must agree across
//!    the document's instances of that name;
//! 2. every schema is registered with the writer in first-appearance order
//!    (`initializeFieldInfo` -> `FieldInfos.FieldNumbers`), which must agree
//!    with the field's schema in earlier documents;
//! 3. every field, in document order, is inverted (`PerField.invert`:
//!    `invertTokenStream` or `invertTerm`), stored, doc-valued
//!    (`indexDocValue`) and pointed (`addPackedValue`);
//! 4. every inverted field is finished (`PerField.finish`): its norm, `0` for
//!    a field with no tokens, `Similarity.computeNorm` otherwise.
//!
//! A `Document` whose inversion fails is refused whole, before anything is
//! buffered: Java marks such a document deleted instead, which leaves the same
//! live documents. Fields registered before the failure stay registered, as
//! they stay in Java's `FieldInfos`.

use std::collections::BTreeMap;

use lucene_analysis::Analyzer;
use lucene_codecs::field_infos::FieldInfo;
use lucene_codecs::stored_fields::StoredField;

use super::{
    illegal, index_options_subsumes, DocValuesSkipIndexType, DocValuesType, Document, FieldTokens,
    IndexOptions, IndexableField, InvertableType, StoredValue, VectorEncoding,
    VectorSimilarityFunction,
};
use crate::buffered_updates::{SeqNo, Term};
use crate::index_writer::pending_postings::FieldRuns;
use crate::index_writer::{
    DocumentVector, Error, ExplicitDocument, ExplicitFields, IndexWriter, InvertedField, Result,
};
use crate::indexing_chain::MAX_POSITION;
use crate::similarity::FieldInvertState;

/// The attributes `IndexingChain.invertTokenStream` reads off one token.
#[derive(Debug, Clone, Copy)]
struct TokenAttrs<'a> {
    term: &'a [u8],
    start_offset: i32,
    end_offset: i32,
    position_increment: i32,
    term_frequency: i32,
    payload: Option<&'a [u8]>,
}

/// `IndexWriter.MAX_TERM_LENGTH`: the longest term, in bytes.
pub const MAX_TERM_LENGTH: usize = 32766;
/// `IndexWriter.MAX_STORED_STRING_LENGTH`: the longest stored string, in
/// UTF-16 units (`ArrayUtil.MAX_ARRAY_LENGTH / 3`).
pub const MAX_STORED_STRING_LENGTH: usize = (i32::MAX as usize - 8) / 3;

fn doc_error(e: super::Error) -> Error {
    Error::Document(e)
}

/// `IndexingChain.FieldSchema`: one field name's structures in one document.
#[derive(Debug, Clone)]
struct FieldSchema<'n> {
    name: &'n str,
    omit_norms: bool,
    store_term_vector: bool,
    index_options: IndexOptions,
    doc_values_type: DocValuesType,
    doc_values_skip_index: DocValuesSkipIndexType,
    point_dimension_count: i32,
    point_index_dimension_count: i32,
    point_num_bytes: i32,
    vector_dimension: i32,
    vector_encoding: VectorEncoding,
    vector_similarity_function: VectorSimilarityFunction,
    attributes: BTreeMap<String, String>,
}

/// The registered schemas of one document's fields, with each field's
/// global field number, as [`IndexWriter::register_field_slots`] returns them.
type RegisteredSchemas<'n> = Vec<(FieldSchema<'n>, i32)>;

impl<'n> FieldSchema<'n> {
    fn new(name: &'n str) -> Self {
        FieldSchema {
            name,
            omit_norms: false,
            store_term_vector: false,
            index_options: IndexOptions::None,
            doc_values_type: DocValuesType::None,
            doc_values_skip_index: DocValuesSkipIndexType::None,
            point_dimension_count: 0,
            point_index_dimension_count: 0,
            point_num_bytes: 0,
            vector_dimension: 0,
            vector_encoding: VectorEncoding::Float32,
            vector_similarity_function: VectorSimilarityFunction::Euclidean,
            attributes: BTreeMap::new(),
        }
    }

    /// `FieldSchema.raiseNotSame`.
    fn not_same(&self, label: &str, expected: String, given: String) -> super::Error {
        illegal(format!(
            "Inconsistency of field data structures across documents for field [{}] of doc \
             [0]. {label}: expected '{expected}', but it has '{given}'.",
            self.name
        ))
    }

    fn same<T: PartialEq + std::fmt::Debug>(
        &self,
        label: &str,
        expected: T,
        given: T,
    ) -> super::Result<()> {
        if expected != given {
            return Err(self.not_same(label, format!("{expected:?}"), format!("{given:?}")));
        }
        Ok(())
    }

    /// `updateDocFieldSchema(fieldName, schema, fieldType)`.
    fn update(&mut self, ft: &super::FieldType) -> super::Result<()> {
        if ft.index_options() != IndexOptions::None {
            if self.index_options == IndexOptions::None {
                self.index_options = ft.index_options();
                self.omit_norms = ft.omit_norms();
                self.store_term_vector = ft.store_term_vectors();
            } else {
                self.same("index options", self.index_options, ft.index_options())?;
                self.same("omit norms", self.omit_norms, ft.omit_norms())?;
                self.same(
                    "store term vector",
                    self.store_term_vector,
                    ft.store_term_vectors(),
                )?;
            }
        } else {
            verify_unindexed_field_type(self.name, ft)?;
        }
        if ft.doc_values_type() != DocValuesType::None {
            if self.doc_values_type == DocValuesType::None {
                self.doc_values_type = ft.doc_values_type();
                self.doc_values_skip_index = ft.doc_values_skip_index_type();
            } else {
                self.same(
                    "doc values type",
                    self.doc_values_type,
                    ft.doc_values_type(),
                )?;
                self.same(
                    "doc values skip index type",
                    self.doc_values_skip_index,
                    ft.doc_values_skip_index_type(),
                )?;
            }
        } else if ft.doc_values_skip_index_type() != DocValuesSkipIndexType::None {
            return Err(illegal(format!(
                "field '{}' cannot have docValuesSkipIndexType={:?} without doc values",
                self.name,
                ft.doc_values_skip_index_type()
            )));
        }
        if ft.point_dimension_count() != 0 {
            if self.point_index_dimension_count == 0 {
                self.point_dimension_count = ft.point_dimension_count();
                self.point_index_dimension_count = ft.point_index_dimension_count();
                self.point_num_bytes = ft.point_num_bytes();
            } else {
                self.same(
                    "point dimension",
                    self.point_dimension_count,
                    ft.point_dimension_count(),
                )?;
                self.same(
                    "point index dimension",
                    self.point_index_dimension_count,
                    ft.point_index_dimension_count(),
                )?;
                self.same(
                    "point num bytes",
                    self.point_num_bytes,
                    ft.point_num_bytes(),
                )?;
            }
        }
        if ft.vector_dimension() != 0 {
            if self.vector_dimension == 0 {
                self.vector_encoding = ft.vector_encoding();
                self.vector_similarity_function = ft.vector_similarity_function();
                self.vector_dimension = ft.vector_dimension();
            } else {
                self.same(
                    "vector encoding",
                    self.vector_encoding,
                    ft.vector_encoding(),
                )?;
                self.same(
                    "vector similarity function",
                    self.vector_similarity_function,
                    ft.vector_similarity_function(),
                )?;
                self.same(
                    "vector dimension",
                    self.vector_dimension,
                    ft.vector_dimension(),
                )?;
            }
        }
        if let Some(attrs) = ft.attributes() {
            for (k, v) in attrs {
                self.attributes.insert(k.clone(), v.clone());
            }
        }
        Ok(())
    }

    /// Whether `registered` is the field [`Self::field_info`] would
    /// register -- the schema `register_field` accepts for an existing
    /// field, with nothing for it to refuse -- so the document can take its
    /// number without building a `FieldInfo`.
    fn matches(&self, registered: &FieldInfo) -> bool {
        let indexed = self.index_options != IndexOptions::None;
        !(indexed && self.store_term_vector)
            && registered.index_options == self.index_options
            && registered.omit_norms == (indexed && self.omit_norms)
            && registered.doc_values_type == self.doc_values_type
            && registered.doc_values_skip_index_type == self.doc_values_skip_index
            && registered.point_dimension_count == self.point_dimension_count
            && registered.point_index_dimension_count == self.point_index_dimension_count
            && registered.point_num_bytes == self.point_num_bytes
            && registered.vector_dimension == self.vector_dimension
            && registered.vector_encoding == self.vector_encoding
            && registered.vector_similarity_function == self.vector_similarity_function
            && !registered.soft_deletes_field
            && !registered.parent_field
    }

    /// The `FieldInfo` `initializeFieldInfo` builds from this schema.
    fn field_info(&self) -> FieldInfo {
        let mut info = FieldInfo::new(self.name, 0);
        info.index_options = self.index_options;
        if self.index_options != IndexOptions::None {
            info.omit_norms = self.omit_norms;
            info.store_term_vectors = self.store_term_vector;
        }
        info.doc_values_type = self.doc_values_type;
        info.doc_values_skip_index_type = self.doc_values_skip_index;
        info.point_dimension_count = self.point_dimension_count;
        info.point_index_dimension_count = self.point_index_dimension_count;
        info.point_num_bytes = self.point_num_bytes;
        info.vector_dimension = self.vector_dimension;
        info.vector_encoding = self.vector_encoding;
        info.vector_similarity_function = self.vector_similarity_function;
        info
    }
}

/// `verifyUnIndexedFieldType`.
fn verify_unindexed_field_type(name: &str, ft: &super::FieldType) -> super::Result<()> {
    let refuse = |what: &str| {
        Err(illegal(format!(
            "cannot store term vector {what}for a field that is not indexed (field=\"{name}\")"
        )))
    };
    if ft.store_term_vectors() {
        return Err(illegal(format!(
            "cannot store term vectors for a field that is not indexed (field=\"{name}\")"
        )));
    }
    if ft.store_term_vector_positions() {
        return refuse("positions ");
    }
    if ft.store_term_vector_offsets() {
        return refuse("offsets ");
    }
    if ft.store_term_vector_payloads() {
        return refuse("payloads ");
    }
    Ok(())
}

/// `FieldInvertState` plus the per-document half of `TermsHashPerField`.
#[derive(Debug)]
struct InvertState {
    index_options: IndexOptions,
    position: i32,
    length: i32,
    num_overlap: i32,
    offset: i32,
    last_start_offset: i32,
    last_position: i32,
    unique_term_count: i32,
    max_term_frequency: i32,
    /// Some term of this document was new to the buffered segment:
    /// `newTerm`, the only call that sets a `DOCS` field's
    /// `maxTermFrequency` (to 1).
    new_to_segment: bool,
    /// `setAttributeSource`: the last value's stream, as its `end()` left
    /// it; `None` after a binary term.
    attribute_source: Option<lucene_analysis::AttributeSource>,
}

impl InvertState {
    /// `FieldInvertState.reset()`.
    fn new(index_options: IndexOptions) -> Self {
        InvertState {
            index_options,
            position: -1,
            length: 0,
            num_overlap: 0,
            offset: 0,
            last_start_offset: 0,
            last_position: 0,
            unique_term_count: 0,
            max_term_frequency: 0,
            new_to_segment: false,
            attribute_source: None,
        }
    }

    fn has_freq(&self) -> bool {
        index_options_subsumes(self.index_options, IndexOptions::DocsAndFreqs)
    }

    fn has_prox(&self) -> bool {
        self.index_options.subsumes_positions()
    }

    fn has_offsets(&self) -> bool {
        self.index_options == IndexOptions::DocsAndFreqsAndPositionsAndOffsets
    }

    fn is_term_doc(&self) -> bool {
        self.index_options == IndexOptions::DocsAndCustomFreqs
    }

    /// `FreqProxTermsWriterPerField.getTermFreq`.
    fn term_freq(&self, field: &str, freq: i32) -> super::Result<i32> {
        if freq != 1 && self.has_prox() {
            return Err(super::Error::IllegalState(format!(
                "field \"{field}\": cannot index positions while using custom \
                 TermFrequencyAttribute"
            )));
        }
        Ok(freq)
    }

    /// `TermsHashPerField.add` -> `newTerm`/`addTerm` for document `doc`,
    /// into the buffered segment's dictionary for this field (`sink`),
    /// `maxTermFrequency` kept as they keep it.
    #[allow(clippy::too_many_arguments)]
    fn add(
        &mut self,
        sink: &mut FieldRuns,
        doc: i32,
        field: &str,
        term: &[u8],
        freq: i32,
        offsets: (i32, i32),
        payload: Option<&[u8]>,
    ) -> super::Result<()> {
        if term.len() > MAX_TERM_LENGTH {
            return Err(illegal(format!(
                "Document contains at least one immense term in field=\"{field}\" (whose UTF8 \
                 encoding is longer than the max length {MAX_TERM_LENGTH}), all of which were \
                 skipped.  Please correct the analyzer to not produce such terms."
            )));
        }
        let has_freq = self.has_freq();
        let has_prox = self.has_prox();
        let has_offsets = self.has_offsets();
        let slot = sink.occurrence(doc, term);
        match slot.freq {
            None => {
                let freq = if has_freq {
                    self.term_freq(field, freq)?
                } else {
                    1
                };
                sink.first_in_doc(slot, doc, freq);
                if has_prox {
                    let offsets = if has_offsets { offsets } else { (0, 0) };
                    sink.prox(slot, self.position, offsets, payload.unwrap_or_default());
                }
                // `newTerm`: `max(1, ...)` without frequencies, the term's
                // frequency with them.
                self.max_term_frequency = self.max_term_frequency.max(freq);
                self.new_to_segment |= slot.new_to_segment;
                self.unique_term_count = self.unique_term_count.saturating_add(1);
            }
            Some(current) => {
                if !has_freq {
                    if freq != 1 {
                        return Err(super::Error::IllegalState(format!(
                            "field \"{field}\": must index term freq while using custom \
                             TermFrequencyAttribute"
                        )));
                    }
                } else {
                    if self.is_term_doc() {
                        return Err(illegal(
                            "Document update skipped due to duplicate termdoc term",
                        ));
                    }
                    if freq != 1 && has_prox {
                        return Err(super::Error::IllegalState(format!(
                            "field \"{field}\": cannot index positions while using custom \
                             TermFrequencyAttribute"
                        )));
                    }
                    let freq = current
                        .checked_add(freq)
                        .ok_or_else(|| illegal("integer overflow"))?;
                    sink.set_freq(slot, freq);
                    self.max_term_frequency = self.max_term_frequency.max(freq);
                    if has_prox {
                        let offsets = if has_offsets { offsets } else { (0, 0) };
                        sink.prox(slot, self.position, offsets, payload.unwrap_or_default());
                    }
                }
            }
        }
        Ok(())
    }

    /// `PerField.invertTokenStream`.
    fn invert_tokens(
        &mut self,
        sink: &mut FieldRuns,
        doc: i32,
        field: &str,
        tokens: &FieldTokens,
        analyzer: &Analyzer,
        analyzed: bool,
    ) -> super::Result<()> {
        for tok in &tokens.tokens {
            let attrs = TokenAttrs {
                term: &tok.term,
                start_offset: tok.start_offset,
                end_offset: tok.end_offset,
                position_increment: tok.position_increment,
                term_frequency: tok.term_frequency,
                payload: tok.payload.as_deref(),
            };
            self.token(sink, doc, field, attrs)?;
        }
        let end = tokens.attributes_at_end();
        let ends = (tokens.final_position_increment, tokens.final_offset);
        self.end_stream(field, end, ends, analyzer, analyzed);
        Ok(())
    }

    /// [`Self::invert_tokens`] over `analyzer.tokenStream(field, text)`,
    /// each token inverted as the stream produces it.
    fn invert_analyzed(
        &mut self,
        sink: &mut FieldRuns,
        doc: i32,
        field: &str,
        analyzer: &Analyzer,
        text: &str,
        analyzed: bool,
    ) -> super::Result<()> {
        use lucene_analysis::TokenStream;
        let analysis = |e: lucene_analysis::AnalysisError| illegal(e.to_string());
        // dropped unclosed on an error, the stream closes itself
        let mut ts = analyzer.token_stream(field, text).map_err(analysis)?;
        ts.reset().map_err(analysis)?;
        while ts.increment_token().map_err(analysis)? {
            let a = ts.attributes();
            let attrs = TokenAttrs {
                term: a.term_bytes(),
                start_offset: a.start_offset(),
                end_offset: a.end_offset(),
                position_increment: a.position_increment(),
                term_frequency: a.term_frequency(),
                payload: a.payload(),
            };
            self.token(sink, doc, field, attrs)?;
        }
        ts.end().map_err(analysis)?;
        let end = ts.attributes().clone();
        ts.close().map_err(analysis)?;
        let ends = (end.position_increment(), end.end_offset());
        self.end_stream(field, end, ends, analyzer, analyzed);
        Ok(())
    }

    /// `end()`'s attributes and final increment and offset, then -- for an
    /// analyzed value -- `getPositionIncrementGap`/`getOffsetGap` before the
    /// field's next value.
    fn end_stream(
        &mut self,
        field: &str,
        end: lucene_analysis::AttributeSource,
        (final_position_increment, final_offset): (i32, i32),
        analyzer: &Analyzer,
        analyzed: bool,
    ) {
        self.attribute_source = Some(end);
        self.position = self.position.wrapping_add(final_position_increment);
        self.offset = self.offset.wrapping_add(final_offset);
        if analyzed {
            self.position = self
                .position
                .wrapping_add(analyzer.position_increment_gap_for_field(field));
            self.offset = self
                .offset
                .wrapping_add(analyzer.offset_gap_for_field(field));
        }
    }

    /// One token of the stream: `invertTokenStream`'s loop body.
    fn token(
        &mut self,
        sink: &mut FieldRuns,
        doc: i32,
        field: &str,
        tok: TokenAttrs<'_>,
    ) -> super::Result<()> {
        let pos_incr = tok.position_increment;
        self.position = self.position.wrapping_add(pos_incr);
        if self.position < self.last_position {
            if pos_incr == 0 {
                return Err(illegal(format!(
                    "first position increment must be > 0 (got 0) for field '{field}'"
                )));
            } else if pos_incr < 0 {
                return Err(illegal(format!(
                    "position increment must be >= 0 (got {pos_incr}) for field '{field}'"
                )));
            } else {
                return Err(illegal(format!(
                    "position overflowed Integer.MAX_VALUE (got posIncr={pos_incr} \
                     lastPosition={} position={}) for field '{field}'",
                    self.last_position, self.position
                )));
            }
        } else if self.position > MAX_POSITION {
            return Err(illegal(format!(
                "position {} is too large for field '{field}': max allowed position is \
                 {MAX_POSITION}",
                self.position
            )));
        }
        self.last_position = self.position;
        if pos_incr == 0 {
            self.num_overlap = self.num_overlap.saturating_add(1);
        }
        let start = self.offset.wrapping_add(tok.start_offset);
        let end = self.offset.wrapping_add(tok.end_offset);
        if start < self.last_start_offset || end < start {
            return Err(illegal(format!(
                "startOffset must be non-negative, and endOffset must be >= startOffset, \
                 and offsets must not go backwards startOffset={start},endOffset={end},\
                 lastStartOffset={} for field '{field}'",
                self.last_start_offset
            )));
        }
        self.last_start_offset = start;
        let add = if self.is_term_doc() {
            1
        } else {
            tok.term_frequency
        };
        self.length = self
            .length
            .checked_add(add)
            .ok_or_else(|| illegal(format!("too many tokens for field \"{field}\"")))?;
        self.add(
            sink,
            doc,
            field,
            tok.term,
            tok.term_frequency,
            (start, end),
            tok.payload,
        )?;
        Ok(())
    }

    /// `PerField.invertTerm`: the binary value is the one term. Java counts
    /// the length twice here (`length++` and `addExact(length, 1)`).
    fn invert_term(
        &mut self,
        sink: &mut FieldRuns,
        doc: i32,
        field: &dyn IndexableField,
    ) -> super::Result<()> {
        let name = field.name();
        let Some(value) = field.binary_value() else {
            return Err(illegal(format!(
                "Field {name} returns TERM for invertableType() and null for binaryValue(), \
                 which is illegal"
            )));
        };
        let ft = field.field_type();
        if ft.tokenized()
            || ft.index_options().subsumes_positions()
            || ft.store_term_vector_positions()
            || ft.store_term_vector_offsets()
            || ft.store_term_vector_payloads()
        {
            return Err(illegal(format!(
                "Fields that are tokenized or index proximity data must produce a non-null \
                 TokenStream, but {name} did not"
            )));
        }
        self.attribute_source = None;
        self.position = self.position.wrapping_add(1);
        self.length = self
            .length
            .checked_add(2)
            .ok_or_else(|| illegal(format!("too many tokens for field \"{name}\"")))?;
        self.add(sink, doc, name, &value, 1, (0, 0), None)
    }
}

impl IndexWriter<'_> {
    /// `IndexWriter.addDocument(Iterable<IndexableField>)`: the document
    /// API's entry point. Switches the writer to explicit documents if it is
    /// not already (see [`IndexWriter::enable_explicit_documents`]).
    pub fn add_fields_document(&mut self, doc: &Document) -> Result<SeqNo> {
        self.add_fields_documents(std::slice::from_ref(doc))
    }

    /// `IndexWriter.addDocuments(docs)`: a block, flushed into one segment
    /// together.
    pub fn add_fields_documents(&mut self, docs: &[Document]) -> Result<SeqNo> {
        self.add_fields_documents_with_vectors(docs, None)
    }

    /// `IndexWriter.updateDocuments(term, docs)`.
    pub fn update_fields_documents(&mut self, term: Term, docs: &[Document]) -> Result<SeqNo> {
        self.add_fields_documents_with_vectors(docs, Some(term))
    }

    /// `IndexWriter.softUpdateDocuments(term, docs, softDeletes...)`: `docs`
    /// are added as one block and every earlier document matching `term`
    /// gets the `soft_deletes` doc-values updates, atomically.
    pub fn soft_update_fields_documents(
        &mut self,
        term: Term,
        docs: &[Document],
        soft_deletes: &[crate::buffered_updates::DocValuesUpdate],
    ) -> Result<SeqNo> {
        self.enable_explicit_documents()?;
        self.buffer_inverted_block(docs, Vec::new(), |w, explicit, vectors| {
            w.soft_update_explicit_documents_with_vectors(term, explicit, vectors, soft_deletes)
        })
    }

    /// The shared tail of the document and column-batch entry points.
    pub(crate) fn add_fields_documents_with_vectors(
        &mut self,
        docs: &[Document],
        delete: Option<Term>,
    ) -> Result<SeqNo> {
        self.add_fields_documents_registering(docs, delete, Vec::new())
    }

    /// [`Self::add_fields_documents_with_vectors`], with `registered` field
    /// numbers put in the segment's `FieldInfos` even when no document
    /// carries a value for them (a column batch's empty columns).
    pub(crate) fn add_fields_documents_registering(
        &mut self,
        docs: &[Document],
        delete: Option<Term>,
        registered: Vec<i32>,
    ) -> Result<SeqNo> {
        self.enable_explicit_documents()?;
        self.buffer_inverted_block(docs, registered, |w, explicit, vectors| {
            w.add_explicit_documents_with_vectors(delete, explicit, vectors)
        })
    }

    /// Inverts `docs` into the buffered postings, then has `buffer` buffer
    /// them. The postings they grew count toward the RAM buffer; a block that
    /// fails before any of its documents is buffered takes them out again,
    /// so a refused block leaves no trace.
    fn buffer_inverted_block(
        &mut self,
        docs: &[Document],
        registered: Vec<i32>,
        buffer: impl FnOnce(&mut Self, Vec<ExplicitDocument>, Vec<Vec<DocumentVector>>) -> Result<SeqNo>,
    ) -> Result<SeqNo> {
        let base = self.pending_doc_id_upto();
        let checkpoint = self.pending_postings.checkpoint(base);
        let before = self.pending_postings.bytes();
        let (explicit, vectors) = match self.invert_fields_block(docs, registered) {
            Ok(inverted) => inverted,
            Err(e) => {
                self.pending_postings.rollback(checkpoint);
                return Err(e);
            }
        };
        let grown = self.pending_postings.bytes().saturating_sub(before);
        self.charge_ram_bytes(grown, true);
        let added = buffer(self, explicit, vectors);
        // `buffer` takes the whole block or none of it (its refusals all come
        // before the first document is buffered), so a failure leaves either
        // nothing to undo or nothing undone.
        if added.is_err() && self.pending_doc_id_upto() == base {
            self.pending_postings.rollback(checkpoint);
            self.charge_ram_bytes(grown, false);
        }
        added
    }

    /// `DocumentsWriterPerThread.updateDocuments`' loop: each document of a
    /// block through `IndexingChain.processDocument`, the last one as the
    /// block's parent -- whose parent field, when the writer has one, is
    /// registered before the document's own fields, as `processDocument`
    /// handles it first.
    fn invert_fields_block(
        &mut self,
        docs: &[Document],
        registered: Vec<i32>,
    ) -> Result<(Vec<ExplicitDocument>, Vec<Vec<DocumentVector>>)> {
        if let Some(parent) = self.parent_field() {
            if docs
                .iter()
                .flat_map(|d| d.fields())
                .any(|f| f.name() == parent)
            {
                return Err(doc_error(illegal(format!(
                    "\"{parent}\" is a reserved field and should not be added to any document"
                ))));
            }
        }
        let mut explicit = Vec::with_capacity(docs.len());
        let mut vectors = Vec::with_capacity(docs.len());
        let last = docs.len().saturating_sub(1);
        let mut doc_id = self.pending_doc_id_upto();
        for (i, doc) in docs.iter().enumerate() {
            if i == last {
                self.register_parent_field()?;
            }
            let (e, v) = self.invert_fields_document_at(doc, doc_id)?;
            doc_id = doc_id.saturating_add(1);
            explicit.push(e);
            vectors.push(v);
        }
        if let Some(first) = explicit.first_mut() {
            first.fields.registered = registered;
        }
        Ok((explicit, vectors))
    }

    /// `updateDocFieldSchema` over `types` (one field instance each, in
    /// order), then `initializeFieldInfo` for each name in first-appearance
    /// order: the schemas and their global numbers.
    fn register_field_types<'n>(
        &mut self,
        types: &[(&'n str, &super::FieldType)],
    ) -> Result<Vec<(FieldSchema<'n>, i32)>> {
        Ok(self.register_field_slots(types)?.0)
    }

    /// [`Self::register_field_types`], with each entry of `types`' index
    /// into the returned schemas: what the inverting loop looks a field up
    /// by, rather than by name.
    fn register_field_slots<'n>(
        &mut self,
        types: &[(&'n str, &super::FieldType)],
    ) -> Result<(RegisteredSchemas<'n>, Vec<usize>)> {
        let mut schemas: Vec<FieldSchema> = Vec::new();
        let mut slots = Vec::with_capacity(types.len());
        for (name, ft) in types {
            let i = match schemas.iter().position(|s| s.name == *name) {
                Some(i) => i,
                None => {
                    schemas.push(FieldSchema::new(name));
                    schemas.len().saturating_sub(1)
                }
            };
            schemas[i].update(ft).map_err(doc_error)?;
            slots.push(i);
        }
        let schemas = schemas
            .into_iter()
            .map(|s| {
                let n = match self.registered_field(s.name) {
                    Some(f) if s.matches(f) => f.number,
                    _ => self.register_field(s.field_info())?,
                };
                Ok((s, n))
            })
            .collect::<Result<_>>()?;
        Ok((schemas, slots))
    }

    /// `IndexingChain.processBatch`'s first pass: every column's schema,
    /// registered in column order (a batch numbers its fields by column, not
    /// by the first document that carries them).
    pub(crate) fn register_batch_field_types(
        &mut self,
        types: &[(&str, &super::FieldType)],
    ) -> Result<Vec<i32>> {
        self.enable_explicit_documents()?;
        Ok(self
            .register_field_types(types)?
            .into_iter()
            .map(|(_, n)| n)
            .collect())
    }

    /// [`Self::invert_fields_document_at`] for the next buffered document,
    /// its terms read back out of the buffered postings into each
    /// [`InvertedField`](crate::index_writer::InvertedField) -- and then
    /// discarded, the document never being buffered.
    #[cfg(test)]
    pub(crate) fn invert_fields_document(
        &mut self,
        doc: &Document,
    ) -> Result<(ExplicitDocument, Vec<DocumentVector>)> {
        let doc_id = self.pending_doc_id_upto();
        let checkpoint = self.pending_postings.checkpoint(doc_id);
        let (mut e, vectors) = self.invert_fields_document_at(doc, doc_id)?;
        for inv in &mut e.fields.inverted {
            inv.terms = self.pending_postings.doc_terms(inv.field_number, doc_id);
        }
        self.pending_postings.rollback(checkpoint);
        Ok((e, vectors))
    }

    /// `IndexingChain.processDocument` for one document, which will be
    /// buffered as `doc_id`; see the module doc. Its terms go straight into
    /// the buffered postings, and out again if it fails.
    fn invert_fields_document_at(
        &mut self,
        doc: &Document,
        doc_id: i32,
    ) -> Result<(ExplicitDocument, Vec<DocumentVector>)> {
        let checkpoint = self.pending_postings.checkpoint(doc_id);
        let inverted = self.invert_fields_into_postings(doc, doc_id);
        if inverted.is_err() {
            self.pending_postings.rollback(checkpoint);
        }
        inverted
    }

    fn invert_fields_into_postings(
        &mut self,
        doc: &Document,
        doc_id: i32,
    ) -> Result<(ExplicitDocument, Vec<DocumentVector>)> {
        // 1. Per-document schemas, first appearance first; 2. global field
        // numbers.
        let types: Vec<(&str, &super::FieldType)> = doc
            .fields()
            .iter()
            .map(|f| (f.name(), f.field_type()))
            .collect();
        let (schemas, slots) = self.register_field_slots(&types)?;
        // 3. Invert, store, doc values, points -- in document order.
        let analyzer = self.writer_analyzer();
        let mut out = ExplicitDocument::default();
        // `(number, schema slot, state)`, first appearance first.
        let mut inverted: Vec<(i32, usize, InvertState)> = Vec::new();
        let mut single_dv: Vec<i32> = Vec::new();
        let mut vectors: Vec<DocumentVector> = Vec::new();
        for (field, &slot) in doc.fields().iter().zip(&slots) {
            let field = field.as_ref();
            let name = field.name();
            let (schema, number) = &schemas[slot];
            let (number, index_options) = (*number, schema.index_options);
            let ft = field.field_type();
            if ft.index_options() != IndexOptions::None {
                let state = match inverted.iter().position(|(n, ..)| *n == number) {
                    Some(i) => &mut inverted[i].2,
                    None => {
                        inverted.push((number, slot, InvertState::new(index_options)));
                        &mut inverted.last_mut().expect("just pushed").2
                    }
                };
                let sink = self.pending_postings.field(number, index_options);
                match field.invertable_type() {
                    InvertableType::Binary => {
                        state.invert_term(sink, doc_id, field).map_err(doc_error)?
                    }
                    InvertableType::TokenStream => {
                        let analyzed = ft.tokenized();
                        if let Some(text) = field.analyzed_text() {
                            state
                                .invert_analyzed(sink, doc_id, name, &analyzer, text, analyzed)
                                .map_err(doc_error)?;
                        } else {
                            let tokens = field
                                .token_stream(&analyzer)
                                .map_err(doc_error)?
                                .ok_or_else(|| {
                                    doc_error(illegal(format!(
                                        "field {name} is indexed but produced no token stream"
                                    )))
                                })?;
                            state
                                .invert_tokens(sink, doc_id, name, &tokens, &analyzer, analyzed)
                                .map_err(doc_error)?;
                        }
                    }
                }
            }
            if ft.stored() {
                let value = field
                    .stored_value()
                    .ok_or_else(|| doc_error(illegal("Cannot store a null value")))?;
                if let StoredValue::String(s) = &value {
                    check_stored_string(name, s.encode_utf16().count()).map_err(doc_error)?;
                }
                out.stored.push(StoredField {
                    field_number: number,
                    value,
                });
            }
            let dv = ft.doc_values_type();
            if dv != DocValuesType::None {
                let value = index_doc_value(field, dv).map_err(doc_error)?;
                if matches!(
                    dv,
                    DocValuesType::Numeric | DocValuesType::Binary | DocValuesType::Sorted
                ) {
                    if single_dv.contains(&number) {
                        return Err(doc_error(illegal(format!(
                            "DocValuesField \"{name}\" appears more than once in this document \
                             (only one value is allowed per field)"
                        ))));
                    }
                    single_dv.push(number);
                }
                out.fields.doc_values.push(StoredField {
                    field_number: number,
                    value,
                });
            }
            if ft.point_dimension_count() != 0 {
                let packed = field.binary_value().ok_or_else(|| {
                    doc_error(illegal(format!(
                        "field {name} has point dimensions but no binary value"
                    )))
                })?;
                out.fields.points.push(StoredField {
                    field_number: number,
                    value: StoredValue::Binary(packed.into_owned()),
                });
            }
            if ft.vector_dimension() != 0 {
                let value = field.vector_value().ok_or_else(|| {
                    doc_error(illegal(format!(
                        "field {name} has vector dimensions but no vector value"
                    )))
                })?;
                vectors.push(DocumentVector {
                    field_name: name.to_string(),
                    value,
                });
            }
        }
        // 4. `PerField.finish`: norms. The similarity is shared, so it is
        // held apart from the writer the `DOCS` term hash below mutates.
        let cfg = self.config_snapshot();
        let similarity = cfg.norm_similarity();
        for (number, slot, state) in inverted {
            let schema = &schemas[slot].0;
            let (name, omit_norms) = (schema.name, schema.omit_norms);
            let docs_only = state.index_options == IndexOptions::Docs;
            let mut max_term_frequency = state.max_term_frequency;
            if docs_only && !omit_norms {
                // Without frequencies only `newTerm` -- a term new to the
                // buffered segment -- sets it (to 1).
                max_term_frequency = i32::from(state.new_to_segment);
            }
            let norm = if omit_norms {
                None
            } else if state.length == 0 {
                Some(0)
            } else {
                let invert_state = FieldInvertState {
                    docs_only,
                    position: state.position,
                    length: state.length,
                    num_overlap: state.num_overlap,
                    offset: state.offset,
                    max_term_frequency,
                    unique_term_count: state.unique_term_count,
                    attribute_source: state.attribute_source,
                };
                let norm = similarity.compute_norm(name, &invert_state);
                if norm == 0 {
                    return Err(Error::ZeroNorm(name.to_string()));
                }
                Some(norm)
            };
            // the terms are in the buffered postings already
            out.fields.inverted.push(InvertedField {
                field_number: number,
                terms: Vec::new(),
                norm,
            });
        }
        Ok((out, vectors))
    }
}

/// `invertAndStore`'s check of a stored string's length (UTF-16 units).
fn check_stored_string(name: &str, len: usize) -> super::Result<()> {
    if len > MAX_STORED_STRING_LENGTH {
        return Err(illegal(format!(
            "stored field \"{name}\" is too large ({len} characters) to store"
        )));
    }
    Ok(())
}

/// `IndexingChain.indexDocValue`: the value the doc-values writer takes.
fn index_doc_value(field: &dyn IndexableField, dv: DocValuesType) -> super::Result<StoredValue> {
    match dv {
        DocValuesType::Numeric | DocValuesType::SortedNumeric => field
            .numeric_value()
            .map(|n| StoredValue::Long(n.long_value()))
            .ok_or_else(|| {
                illegal(format!(
                    "field=\"{}\": null value not allowed",
                    field.name()
                ))
            }),
        DocValuesType::Binary | DocValuesType::Sorted | DocValuesType::SortedSet => field
            .binary_value()
            .map(|b| StoredValue::Binary(b.into_owned()))
            .ok_or_else(|| {
                illegal(format!(
                    "field=\"{}\": null value not allowed",
                    field.name()
                ))
            }),
        DocValuesType::None => unreachable!("only called for a doc-values field"),
    }
}

#[cfg(test)]
pub(crate) fn test_check_stored_string(name: &str, len: usize) -> super::Result<()> {
    check_stored_string(name, len)
}

impl ExplicitFields {
    /// Whether this document carries nothing but stored values.
    pub fn is_empty(&self) -> bool {
        self.inverted.is_empty()
            && self.doc_values.is_empty()
            && self.points.is_empty()
            && self.registered.is_empty()
    }
}
