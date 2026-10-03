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

use lucene_codecs::field_infos::FieldInfo;
use lucene_codecs::stored_fields::StoredField;

use super::{
    illegal, index_options_subsumes, DocValuesSkipIndexType, DocValuesType, Document, FieldTokens,
    IndexOptions, IndexableField, InvertableType, StoredValue, VectorEncoding,
    VectorSimilarityFunction,
};
use crate::buffered_updates::{SeqNo, Term};
use crate::index_writer::{
    DocumentVector, Error, ExplicitDocument, ExplicitFields, IndexWriter, InvertedField,
    InvertedTerm, Result,
};
use crate::indexing_chain::MAX_POSITION;
use crate::similarity::FieldInvertState;

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
struct FieldSchema {
    name: String,
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

impl FieldSchema {
    fn new(name: &str) -> Self {
        FieldSchema {
            name: name.to_string(),
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
            verify_unindexed_field_type(&self.name, ft)?;
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

    /// The `FieldInfo` `initializeFieldInfo` builds from this schema.
    fn field_info(&self) -> FieldInfo {
        let mut info = FieldInfo::new(self.name.clone(), 0);
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

/// One term's occurrences in one document's field.
#[derive(Debug, Default)]
struct TermAcc {
    freq: i32,
    positions: Vec<i32>,
    offsets: Vec<(i32, i32)>,
    /// Parallel to `positions`: each occurrence's `PayloadAttribute`, empty
    /// for none.
    payloads: Vec<Vec<u8>>,
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
    /// `FreqProxTermsWriterPerField.sawPayloads`: some occurrence carried a
    /// non-empty payload.
    saw_payloads: bool,
    /// `setAttributeSource`: the last value's stream, as its `end()` left
    /// it; `None` after a binary term.
    attribute_source: Option<lucene_analysis::AttributeSource>,
    terms: BTreeMap<Vec<u8>, TermAcc>,
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
            saw_payloads: false,
            attribute_source: None,
            terms: BTreeMap::new(),
        }
    }

    /// `writeProx`'s payload half: an occurrence's payload, recorded when
    /// the field indexes positions. Empty is none.
    fn push_payload(saw: &mut bool, acc: &mut TermAcc, payload: Option<&[u8]>) {
        let payload = payload.unwrap_or_default();
        if !payload.is_empty() {
            *saw = true;
        }
        acc.payloads.push(payload.to_vec());
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

    /// `TermsHashPerField.add` -> `newTerm`/`addTerm` for this document,
    /// `maxTermFrequency` kept as they keep it.
    fn add(
        &mut self,
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
        let is_term_doc = self.is_term_doc();
        let position = self.position;
        match self.terms.get_mut(term) {
            None => {
                let freq = if has_freq {
                    self.term_freq(field, freq)?
                } else {
                    1
                };
                let mut acc = TermAcc {
                    freq,
                    ..TermAcc::default()
                };
                if has_prox {
                    acc.positions.push(position);
                    Self::push_payload(&mut self.saw_payloads, &mut acc, payload);
                    if has_offsets {
                        acc.offsets.push(offsets);
                    }
                }
                // `newTerm`: `max(1, ...)` without frequencies, the term's
                // frequency with them.
                self.max_term_frequency = self.max_term_frequency.max(acc.freq);
                self.terms.insert(term.to_vec(), acc);
                self.unique_term_count = self.unique_term_count.saturating_add(1);
            }
            Some(acc) => {
                if !has_freq {
                    if freq != 1 {
                        return Err(super::Error::IllegalState(format!(
                            "field \"{field}\": must index term freq while using custom \
                             TermFrequencyAttribute"
                        )));
                    }
                } else {
                    if is_term_doc {
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
                    acc.freq = acc
                        .freq
                        .checked_add(freq)
                        .ok_or_else(|| illegal("integer overflow"))?;
                    self.max_term_frequency = self.max_term_frequency.max(acc.freq);
                    if has_prox {
                        acc.positions.push(position);
                        Self::push_payload(&mut self.saw_payloads, acc, payload);
                        if has_offsets {
                            acc.offsets.push(offsets);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// `PerField.invertTokenStream`.
    fn invert_tokens(
        &mut self,
        field: &str,
        tokens: &FieldTokens,
        analyzed: bool,
        gaps: (i32, i32),
    ) -> super::Result<()> {
        for tok in &tokens.tokens {
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
                field,
                &tok.term,
                tok.term_frequency,
                (start, end),
                tok.payload.as_deref(),
            )?;
        }
        self.attribute_source = Some(tokens.attributes_at_end());
        self.position = self.position.wrapping_add(tokens.final_position_increment);
        self.offset = self.offset.wrapping_add(tokens.final_offset);
        if analyzed {
            self.position = self.position.wrapping_add(gaps.0);
            self.offset = self.offset.wrapping_add(gaps.1);
        }
        Ok(())
    }

    /// `PerField.invertTerm`: the binary value is the one term. Java counts
    /// the length twice here (`length++` and `addExact(length, 1)`).
    fn invert_term(&mut self, field: &dyn IndexableField) -> super::Result<()> {
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
        self.add(name, &value, 1, (0, 0), None)
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
        let (explicit, vectors) = self.invert_fields_block(docs, Vec::new())?;
        self.soft_update_explicit_documents_with_vectors(term, explicit, vectors, soft_deletes)
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
        let (explicit, vectors) = self.invert_fields_block(docs, registered)?;
        self.add_explicit_documents_with_vectors(delete, explicit, vectors)
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
        for (i, doc) in docs.iter().enumerate() {
            if i == last {
                self.register_parent_field()?;
            }
            let (e, v) = self.invert_fields_document(doc)?;
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
    fn register_field_types(
        &mut self,
        types: &[(&str, &super::FieldType)],
    ) -> Result<Vec<(FieldSchema, i32)>> {
        let mut schemas: Vec<FieldSchema> = Vec::new();
        for (name, ft) in types {
            let i = match schemas.iter().position(|s| s.name == *name) {
                Some(i) => i,
                None => {
                    schemas.push(FieldSchema::new(name));
                    schemas.len().saturating_sub(1)
                }
            };
            schemas[i].update(ft).map_err(doc_error)?;
        }
        schemas
            .into_iter()
            .map(|s| {
                let n = self.register_field(s.field_info())?;
                Ok((s, n))
            })
            .collect()
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

    /// `IndexingChain.processDocument` for one document; see the module doc.
    pub(crate) fn invert_fields_document(
        &mut self,
        doc: &Document,
    ) -> Result<(ExplicitDocument, Vec<DocumentVector>)> {
        // 1. Per-document schemas, first appearance first; 2. global field
        // numbers.
        let types: Vec<(&str, &super::FieldType)> = doc
            .fields()
            .iter()
            .map(|f| (f.name(), f.field_type()))
            .collect();
        let schemas = self.register_field_types(&types)?;
        let numbers: BTreeMap<String, (i32, IndexOptions)> = schemas
            .iter()
            .map(|(s, n)| (s.name.clone(), (*n, s.index_options)))
            .collect();
        let schemas: Vec<FieldSchema> = schemas.into_iter().map(|(s, _)| s).collect();
        // 3. Invert, store, doc values, points -- in document order.
        let analyzer = self.writer_analyzer();
        let mut out = ExplicitDocument::default();
        let mut inverted: Vec<(i32, InvertState)> = Vec::new();
        let mut single_dv: Vec<i32> = Vec::new();
        let mut vectors: Vec<DocumentVector> = Vec::new();
        for field in doc.fields() {
            let field = field.as_ref();
            let name = field.name();
            let (number, index_options) = numbers[name];
            let ft = field.field_type();
            if ft.index_options() != IndexOptions::None {
                let state = match inverted.iter().position(|(n, _)| *n == number) {
                    Some(i) => &mut inverted[i].1,
                    None => {
                        inverted.push((number, InvertState::new(index_options)));
                        &mut inverted.last_mut().expect("just pushed").1
                    }
                };
                match field.invertable_type() {
                    InvertableType::Binary => state.invert_term(field).map_err(doc_error)?,
                    InvertableType::TokenStream => {
                        let tokens = field
                            .token_stream(&analyzer)
                            .map_err(doc_error)?
                            .ok_or_else(|| {
                                doc_error(illegal(format!(
                                    "field {name} is indexed but produced no token stream"
                                )))
                            })?;
                        let analyzed = ft.tokenized();
                        // `analyzer.getPositionIncrementGap(fieldInfo.name)`.
                        let gaps = (
                            analyzer.position_increment_gap_for_field(name),
                            analyzer.offset_gap_for_field(name),
                        );
                        state
                            .invert_tokens(name, &tokens, analyzed, gaps)
                            .map_err(doc_error)?;
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
        for (number, state) in inverted {
            let (name, omit_norms) = numbers
                .iter()
                .find(|(_, (n, _))| *n == number)
                .map(|(name, _)| {
                    let omit = schemas
                        .iter()
                        .find(|s| &s.name == name)
                        .is_some_and(|s| s.omit_norms);
                    (name.clone(), omit)
                })
                .expect("every inverted field was registered");
            let docs_only = state.index_options == IndexOptions::Docs;
            let mut max_term_frequency = state.max_term_frequency;
            if docs_only && !omit_norms {
                // Without frequencies only `newTerm` -- a term new to the
                // buffered segment -- sets it (to 1).
                max_term_frequency = i32::from(self.note_docs_terms(number, state.terms.keys()));
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
                    attribute_source: state.attribute_source.clone(),
                };
                let norm = similarity.compute_norm(&name, &invert_state);
                if norm == 0 {
                    return Err(Error::ZeroNorm(name));
                }
                Some(norm)
            };
            out.fields.inverted.push(InvertedField {
                field_number: number,
                terms: state
                    .terms
                    .into_iter()
                    .map(|(term, acc)| InvertedTerm {
                        term,
                        freq: acc.freq,
                        positions: acc.positions,
                        offsets: acc.offsets,
                        payloads: if state.saw_payloads {
                            acc.payloads
                        } else {
                            Vec::new()
                        },
                    })
                    .collect(),
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
