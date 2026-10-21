//! `org.apache.lucene.index.memory.MemoryIndex`: a single-document index
//! held in memory, built field by field and searched with any query --
//! the index the highlighters, the monitor and percolation run their
//! queries against.
//!
//! The front end is Java's: each field's [`Info`] collects its terms, their
//! occurrences (position, offsets, payload index -- the ints Java writes to
//! its `SlicedIntBlockPool`), the token counts its norm is computed from,
//! its doc values, points and stored values, with Java's checks and messages.
//! [`MemoryIndex::create_reader`] hands them, term-sorted, to a
//! [`MemoryIndexReader`]: like Java's, a `LeafReader` over those arrays,
//! which also answers `Weight.matches` from them. This port's scorer tree
//! executes on codec structures (`crate::multi_segment::OpenSegment`), so a
//! search encodes the one document -- terms, positions, offsets and payloads
//! as `Info` holds them, the norm as `Info` computes it -- with the codec's
//! own writers, on first need ([`MemoryIndexReader`] has the details). The
//! statistics every form reports are Java's for a field with tokens:
//! `maxDoc` 1, every term `docFreq` 1, `totalTermFreq` its frequency,
//! `docCount` 1 (Lucene's `TestMemoryIndexAgainstDirectory` duels the two).

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use lucene_analysis::Analyzer;
use lucene_codecs::field_infos::{DocValuesType, FieldInfo, IndexOptions};
use lucene_codecs::postings_writer::TermPostings;
use lucene_codecs::stored_fields::{FieldValue, StoredField};
use lucene_index::document::{
    doc_values_type_name, FieldToken, FieldTokens, FieldType, IndexableField, InvertableType,
    StoredValue,
};
use lucene_index::similarity::FieldInvertState;

use crate::query::Clause;
use crate::similarities::{Bm25Similarity, Similarity};
use crate::{Error, Result};

mod leaf;
mod reader;
mod term_hash;
#[cfg(test)]
mod tests;
mod vector;

pub use reader::MemoryIndexReader;
use reader::{Document, FieldPostings};
use term_hash::TermHash;

fn illegal(message: impl Into<String>) -> Error {
    Error::IllegalArgument(message.into())
}

/// `BytesRef.toString()`: `[hex hex ...]`, each byte unpadded.
fn bytes_ref_string(bytes: &[u8]) -> String {
    let hex: Vec<String> = bytes.iter().map(|b| format!("{b:x}")).collect();
    format!("[{}]", hex.join(" "))
}

/// `String.length()`: UTF-16 units.
fn utf16_len(s: &str) -> i32 {
    i32::try_from(s.encode_utf16().count()).unwrap_or(i32::MAX)
}

/// `MemoryIndex.Info`: one field's data.
#[derive(Debug, Clone)]
struct Info {
    field_info: FieldInfo,
    /// The norm, computed on demand and dropped when the similarity changes.
    norm: Option<i64>,
    /// `BytesRefHash terms`: each distinct term, by ord.
    terms: TermHash,
    /// `SliceByteStartArray.freq`, by ord.
    freq: Vec<i32>,
    /// Every occurrence in token order: its term's ord, then the ints Java
    /// writes to the term's slice -- the position, (with offsets) the start
    /// and end offsets, (with payloads) the payload index. [`Info::slices`]
    /// groups them by term.
    occurrences: Vec<i32>,
    /// The ints per occurrence after the ord: `1`, `+ 2` with offsets,
    /// `+ 1` with payloads.
    width: usize,
    num_tokens: i32,
    num_overlap_tokens: i32,
    sum_total_term_freq: i64,
    max_term_frequency: i32,
    last_position: i32,
    last_offset: i32,
    /// `NUMERIC`/`SORTED_NUMERIC` values, in add order.
    numeric_dv: Vec<i64>,
    /// The `BINARY`/`SORTED` value.
    binary_dv: Option<Vec<u8>>,
    /// The `SORTED_SET` values (a `BytesRefHash`: each value once).
    sorted_set_dv: BTreeSet<Vec<u8>>,
    stored: Vec<StoredValue>,
    points: Vec<Vec<u8>>,
}

impl Info {
    fn new(field_info: FieldInfo, width: usize) -> Self {
        Info {
            field_info,
            norm: None,
            terms: TermHash::default(),
            freq: Vec::new(),
            occurrences: Vec::new(),
            width,
            num_tokens: 0,
            num_overlap_tokens: 0,
            sum_total_term_freq: 0,
            max_term_frequency: 0,
            last_position: 0,
            last_offset: 0,
            numeric_dv: Vec::new(),
            binary_dv: None,
            sorted_set_dv: BTreeSet::new(),
            stored: Vec::new(),
            points: Vec::new(),
        }
    }

    /// `terms.add(term)` and the slice it continues or starts.
    fn slice_for(&mut self, term: &[u8]) -> usize {
        let ord = self.terms.add(term);
        if ord == self.freq.len() {
            self.freq.push(0);
        }
        ord
    }

    /// The bookkeeping every occurrence shares (`freq`, `maxTermFrequency`,
    /// `sumTotalTermFreq`).
    fn count(&mut self, ord: usize) {
        self.freq[ord] = self.freq[ord].saturating_add(1);
        self.max_term_frequency = self.max_term_frequency.max(self.freq[ord]);
        self.sum_total_term_freq = self.sum_total_term_freq.saturating_add(1);
    }

    /// The ords in ascending term order (`BytesRefHash.sort`: unsigned
    /// bytes).
    fn sorted_ords(&self) -> Vec<usize> {
        let mut ords: Vec<usize> = (0..self.terms.len()).collect();
        ords.sort_by(|&a, &b| self.terms.get(a).cmp(self.terms.get(b)));
        ords
    }

    /// Every term's slice (`SliceByteStartArray.start`/`end`): term `ord`'s
    /// occurrences are `ints[starts[ord]..starts[ord + 1]]`, `width` ints
    /// each, in the order they were added.
    fn slices(&self) -> (Vec<usize>, Vec<i32>) {
        let w = self.width;
        let mut starts = Vec::with_capacity(self.freq.len() + 1);
        let mut at = 0usize;
        for &f in &self.freq {
            starts.push(at);
            at += f as usize * w;
        }
        starts.push(at);
        let mut next = starts.clone();
        let mut ints = vec![0; at];
        for occ in self.occurrences.chunks_exact(w + 1) {
            let ord = occ[0] as usize;
            ints[next[ord]..next[ord] + w].copy_from_slice(&occ[1..]);
            next[ord] += w;
        }
        (starts, ints)
    }

    /// `Info.getNormDocValues`'s value: `normSimilarity.computeNorm` over
    /// the field's `FieldInvertState`.
    fn norm(&mut self, similarity: &dyn Similarity) -> i64 {
        if let Some(n) = self.norm {
            return n;
        }
        let n = self.norm_of(self.terms.len(), similarity);
        self.norm = Some(n);
        n
    }

    /// The norm of the field's statistics with `unique_terms` distinct
    /// terms.
    fn norm_of(&self, unique_terms: usize, similarity: &dyn Similarity) -> i64 {
        let state = FieldInvertState {
            docs_only: self.field_info.index_options == IndexOptions::Docs,
            position: self.last_position,
            length: self.num_tokens,
            num_overlap: self.num_overlap_tokens,
            offset: 0,
            max_term_frequency: self.max_term_frequency,
            unique_term_count: i32::try_from(unique_terms).unwrap_or(i32::MAX),
            attribute_source: None,
        };
        similarity.compute_norm(&self.field_info.name, &state)
    }
}

/// One value of a field being stored: `storeTerms`' loop state.
struct ValueWriter<'a> {
    info: &'a mut Info,
    payloads: &'a mut Vec<Vec<u8>>,
    store_offsets: bool,
    store_payloads: bool,
    pos: i32,
    offset: i32,
}

impl ValueWriter<'_> {
    /// One token: its occurrence appended to its term's slice.
    fn token(&mut self, term: &[u8], increment: i32, start: i32, end: i32, payload: Option<&[u8]>) {
        let info = &mut *self.info;
        info.num_tokens = info.num_tokens.saturating_add(1);
        if increment == 0 {
            info.num_overlap_tokens = info.num_overlap_tokens.saturating_add(1);
        }
        self.pos = self.pos.saturating_add(increment);
        let ord = info.slice_for(term);
        info.count(ord);
        let slice = &mut info.occurrences;
        slice.push(ord as i32);
        slice.push(self.pos);
        if self.store_offsets {
            slice.push(start.saturating_add(self.offset));
            slice.push(end.saturating_add(self.offset));
        }
        if self.store_payloads {
            let index = match payload {
                Some(p) if !p.is_empty() => {
                    self.payloads.push(p.to_vec());
                    i32::try_from(self.payloads.len() - 1).unwrap_or(i32::MAX)
                }
                _ => -1,
            };
            slice.push(index);
        }
    }

    /// `end()`'s final offset: where the next value's offsets start.
    fn finish(self, final_offset: i32) {
        if self.info.num_tokens > 0 {
            self.info.last_position = self.pos;
            self.info.last_offset = final_offset.saturating_add(self.offset);
        }
    }
}

/// `org.apache.lucene.index.memory.MemoryIndex`.
#[derive(Debug, Clone)]
pub struct MemoryIndex {
    store_offsets: bool,
    store_payloads: bool,
    /// `SortedMap<String, Info>`: field name order.
    fields: BTreeMap<String, Info>,
    /// `payloadsBytesRefs`.
    payloads: Vec<Vec<u8>>,
    frozen: bool,
    similarity: Arc<dyn Similarity>,
    /// The reader [`MemoryIndex::freeze`] built.
    frozen_reader: Option<Arc<MemoryIndexReader>>,
}

impl Default for MemoryIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// `IndexSearcher.getDefaultSimilarity()`.
fn default_similarity() -> Arc<dyn Similarity> {
    Arc::new(Bm25Similarity::default())
}

impl MemoryIndex {
    /// `new MemoryIndex()`: no offsets, no payloads.
    pub fn new() -> Self {
        Self::with_options(false, false)
    }

    /// `new MemoryIndex(storeOffsets, storePayloads)`.
    pub fn with_options(store_offsets: bool, store_payloads: bool) -> Self {
        MemoryIndex {
            store_offsets,
            store_payloads,
            fields: BTreeMap::new(),
            payloads: Vec::new(),
            frozen: false,
            similarity: default_similarity(),
            frozen_reader: None,
        }
    }

    /// `defaultFieldType`: tokenized, indexed with positions (and offsets),
    /// term vectors on.
    fn default_field_type(&self) -> FieldType {
        let mut ft = FieldType::new();
        let fresh = "a fresh field type is not frozen";
        ft.set_index_options(self.index_options()).expect(fresh);
        ft.set_store_term_vectors(true).expect(fresh);
        ft
    }

    fn index_options(&self) -> IndexOptions {
        if self.store_offsets {
            IndexOptions::DocsAndFreqsAndPositionsAndOffsets
        } else {
            IndexOptions::DocsAndFreqsAndPositions
        }
    }

    /// `addField(String fieldName, String text, Analyzer analyzer)`.
    pub fn add_field_text(&mut self, name: &str, text: &str, analyzer: &Analyzer) -> Result<()> {
        use lucene_analysis::TokenStream;
        let analysis = |e: lucene_analysis::AnalysisError| illegal(e.to_string());
        let mut ts = analyzer.token_stream(name, text).map_err(analysis)?;
        let ft = self.default_field_type();
        self.get_info(name, &ft)?;
        let mut w = self.value_writer(
            name,
            analyzer.position_increment_gap_for_field(name),
            analyzer.offset_gap_for_field(name),
        );
        // the stream read as it runs, each term copied only when new
        ts.reset().map_err(analysis)?;
        while ts.increment_token().map_err(analysis)? {
            let a = ts.attributes();
            w.token(
                a.term_bytes(),
                a.position_increment(),
                a.start_offset(),
                a.end_offset(),
                a.payload(),
            );
        }
        ts.end().map_err(analysis)?;
        w.finish(ts.attributes().end_offset());
        ts.close().map_err(analysis)?;
        Ok(())
    }

    /// `addField(String fieldName, TokenStream stream, int
    /// positionIncrementGap, int offsetGap)` (`addField(name, stream)` is
    /// gaps `0` and `1`).
    pub fn add_field_tokens(
        &mut self,
        name: &str,
        tokens: &FieldTokens,
        position_increment_gap: i32,
        offset_gap: i32,
    ) -> Result<()> {
        let ft = self.default_field_type();
        self.get_info(name, &ft)?;
        self.store_terms(name, tokens, position_increment_gap, offset_gap);
        Ok(())
    }

    /// `addField(IndexableField field, Analyzer analyzer)`: the field's
    /// terms, doc values, points and stored value, as its type says.
    pub fn add_field(
        &mut self,
        field: &dyn IndexableField,
        analyzer: Option<&Analyzer>,
    ) -> Result<()> {
        let name = field.name();
        let ft = field.field_type();
        // refused before anything of the field is stored
        if ft.vector_dimension() > 0 {
            return Err(illegal(format!(
                "MemoryIndex does not index KNN vector fields yet (field [{name}])"
            )));
        }
        self.get_info(name, ft)?;
        let (position_increment_gap, offset_gap) = match analyzer {
            Some(a) => (
                a.position_increment_gap_for_field(name),
                a.offset_gap_for_field(name),
            ),
            None => (0, 1),
        };
        let indexed = ft.index_options() != IndexOptions::None;
        if indexed && field.invertable_type() == InvertableType::Binary {
            // `field.tokenStream` is null for a binary-invertable field
            let Some(term) = field.binary_value() else {
                return Err(illegal(
                    "Indexed field must provide a TokenStream or a binary value",
                ));
            };
            self.store_term(name, &term);
        } else if indexed {
            let tokens = match analyzer {
                Some(a) => field.token_stream(a),
                None if ft.tokenized() => {
                    // `Field.tokenStream(null, ...)` dereferences the analyzer
                    return Err(Error::IllegalState(
                        "java.lang.NullPointerException: a tokenized field needs an analyzer"
                            .into(),
                    ));
                }
                // an untokenized field never reads it
                None => field.token_stream(&Analyzer::keyword()),
            }
            .map_err(|e| illegal(e.to_string()))?;
            if let Some(tokens) = tokens {
                self.store_terms(name, &tokens, position_increment_gap, offset_gap);
            }
        }

        let dv_type = ft.doc_values_type();
        match dv_type {
            DocValuesType::None => {}
            DocValuesType::Numeric | DocValuesType::SortedNumeric => {
                if let Some(n) = field.numeric_value() {
                    self.store_numeric_doc_value(name, dv_type, n.long_value())?;
                }
            }
            DocValuesType::Binary | DocValuesType::Sorted | DocValuesType::SortedSet => {
                if let Some(b) = field.binary_value() {
                    self.store_binary_doc_value(name, dv_type, b.into_owned())?;
                }
            }
        }

        if ft.point_dimension_count() > 0 {
            let value = field.binary_value().map(|b| b.into_owned());
            let info = self.info_mut(name);
            info.points
                .push(value.ok_or_else(|| illegal("a point field needs a packed value"))?);
        }

        if ft.stored() {
            if let Some(v) = field.stored_value() {
                self.info_mut(name).stored.push(v);
            }
        }

        Ok(())
    }

    /// `MemoryIndex.fromDocument(document, analyzer, storeOffsets,
    /// storePayloads)`.
    pub fn from_document(
        document: &lucene_index::document::Document,
        analyzer: &Analyzer,
        store_offsets: bool,
        store_payloads: bool,
    ) -> Result<Self> {
        let mut mi = Self::with_options(store_offsets, store_payloads);
        for field in document.fields() {
            mi.add_field(field.as_ref(), Some(analyzer))?;
        }
        Ok(mi)
    }

    /// `keywordTokenStream(keywords)`: a token for each keyword as is,
    /// offsets as if the keywords were joined by single blanks.
    pub fn keyword_token_stream<T: AsRef<str>>(keywords: &[T]) -> FieldTokens {
        let mut start = 0i32;
        let tokens = keywords
            .iter()
            .map(|k| {
                let k = k.as_ref();
                let len = utf16_len(k);
                let t = FieldToken::new(k.as_bytes().to_vec(), start, start.saturating_add(len));
                start = start.saturating_add(len).saturating_add(1);
                t
            })
            .collect();
        FieldTokens {
            tokens,
            final_position_increment: 0,
            final_offset: 0,
            end_attributes: None,
        }
    }

    fn info_mut(&mut self, name: &str) -> &mut Info {
        self.fields
            .get_mut(name)
            .expect("get_info registered the field")
    }

    /// `getInfo(fieldName, fieldType)`: the field's `Info`, created on first
    /// sight, its point shape and doc-values type merged in.
    fn get_info(&mut self, name: &str, ft: &FieldType) -> Result<()> {
        if self.frozen {
            return Err(illegal("Cannot call addField() when MemoryIndex is frozen"));
        }
        if !self.fields.contains_key(name) {
            let number = i32::try_from(self.fields.len()).unwrap_or(i32::MAX);
            let mut fi = FieldInfo::new(name, number);
            fi.store_term_vectors = ft.store_term_vectors();
            fi.omit_norms = ft.omit_norms();
            fi.store_payloads = self.store_payloads;
            fi.index_options = self.index_options();
            fi.doc_values_type = ft.doc_values_type();
            fi.point_dimension_count = ft.point_dimension_count();
            fi.point_index_dimension_count = ft.point_index_dimension_count();
            fi.point_num_bytes = ft.point_num_bytes();
            fi.vector_dimension = ft.vector_dimension();
            fi.vector_encoding = ft.vector_encoding();
            fi.vector_similarity_function = ft.vector_similarity_function();
            let width =
                1 + if self.store_offsets { 2 } else { 0 } + usize::from(self.store_payloads);
            self.fields.insert(name.to_string(), Info::new(fi, width));
        }
        let fi = &mut self.info_mut(name).field_info;
        let dims = ft.point_dimension_count();
        if dims != fi.point_dimension_count && dims > 0 {
            // `FieldInfo.setPointDimensions`
            if fi.point_dimension_count != 0 {
                return Err(illegal(format!(
                    "cannot change point dimension count from {} to {dims} for field=\"{name}\"",
                    fi.point_dimension_count
                )));
            }
            fi.point_dimension_count = dims;
            fi.point_index_dimension_count = ft.point_index_dimension_count();
            fi.point_num_bytes = ft.point_num_bytes();
        }
        let dv = ft.doc_values_type();
        if dv != fi.doc_values_type && dv != DocValuesType::None {
            // `FieldInfo.setDocValuesType`
            if fi.doc_values_type != DocValuesType::None {
                return Err(illegal(format!(
                    "cannot change DocValues type from {} to {} for field \"{name}\"",
                    doc_values_type_name(fi.doc_values_type),
                    doc_values_type_name(dv)
                )));
            }
            fi.doc_values_type = dv;
        }
        Ok(())
    }

    /// `storeDocValues` for `NUMERIC`/`SORTED_NUMERIC`.
    fn store_numeric_doc_value(&mut self, name: &str, ty: DocValuesType, v: i64) -> Result<()> {
        let info = self.info_mut(name);
        if ty == DocValuesType::Numeric && !info.numeric_dv.is_empty() {
            return Err(only_one_value(ty, name));
        }
        info.numeric_dv.push(v);
        Ok(())
    }

    /// `storeDocValues` for `BINARY`/`SORTED`/`SORTED_SET`.
    fn store_binary_doc_value(&mut self, name: &str, ty: DocValuesType, v: Vec<u8>) -> Result<()> {
        let info = self.info_mut(name);
        if ty == DocValuesType::SortedSet {
            info.sorted_set_dv.insert(v);
            return Ok(());
        }
        if info.binary_dv.is_some() {
            return Err(only_one_value(ty, name));
        }
        info.binary_dv = Some(v);
        Ok(())
    }

    /// `storeTerm(info, term)`: a binary term at the next position, offsets
    /// `0, 0`, no payload.
    fn store_term(&mut self, name: &str, term: &[u8]) {
        let (store_offsets, store_payloads) = (self.store_offsets, self.store_payloads);
        let info = self.info_mut(name);
        info.num_tokens = info.num_tokens.saturating_add(1);
        let ord = info.slice_for(term);
        info.count(ord);
        let pos = info.last_position;
        info.last_position = info.last_position.saturating_add(1);
        let slice = &mut info.occurrences;
        slice.push(ord as i32);
        slice.push(pos);
        if store_offsets {
            slice.extend([0, 0]);
        }
        if store_payloads {
            slice.push(-1);
        }
    }

    /// `storeTerms(info, tokenStream, positionIncrementGap, offsetGap)` over
    /// an analysed stream.
    fn store_terms(
        &mut self,
        name: &str,
        tokens: &FieldTokens,
        position_increment_gap: i32,
        offset_gap: i32,
    ) {
        let mut w = self.value_writer(name, position_increment_gap, offset_gap);
        for t in &tokens.tokens {
            w.token(
                &t.term,
                t.position_increment,
                t.start_offset,
                t.end_offset,
                t.payload.as_deref(),
            );
        }
        w.finish(tokens.final_offset);
    }

    /// The writer of one more value of field `name` (`storeTerms`' state).
    fn value_writer(
        &mut self,
        name: &str,
        position_increment_gap: i32,
        offset_gap: i32,
    ) -> ValueWriter<'_> {
        let info = self
            .fields
            .get_mut(name)
            .expect("get_info registered the field");
        let (mut pos, mut offset) = (-1i32, 0i32);
        if info.num_tokens > 0 {
            pos = info.last_position.saturating_add(position_increment_gap);
            offset = info.last_offset.saturating_add(offset_gap);
        }
        ValueWriter {
            info,
            payloads: &mut self.payloads,
            store_offsets: self.store_offsets,
            store_payloads: self.store_payloads,
            pos,
            offset,
        }
    }

    /// `setSimilarity(similarity)`: the similarity norms are computed with
    /// and the searcher scores with.
    pub fn set_similarity(&mut self, similarity: Arc<dyn Similarity>) -> Result<()> {
        if self.frozen {
            return Err(illegal("Cannot set Similarity when MemoryIndex is frozen"));
        }
        if Arc::ptr_eq(&self.similarity, &similarity) {
            return Ok(());
        }
        self.similarity = similarity;
        for info in self.fields.values_mut() {
            info.norm = None;
        }
        Ok(())
    }

    /// `freeze()`: no more fields; the reader is built once and every
    /// search shares it.
    ///
    /// Java's cannot fail; this one can, when the document holds a position
    /// a segment cannot (see the module docs) -- and then leaves the index
    /// as it was, unfrozen.
    pub fn freeze(&mut self) -> Result<()> {
        if self.frozen_reader.is_none() {
            let reader = self.build_reader()?;
            self.frozen_reader = Some(Arc::new(reader));
        }
        self.frozen = true;
        Ok(())
    }

    /// `reset()`: empty again, unfrozen, with the default similarity.
    pub fn reset(&mut self) {
        self.fields.clear();
        self.payloads.clear();
        self.similarity = default_similarity();
        self.frozen = false;
        self.frozen_reader = None;
    }

    /// `createSearcher()`'s reader: the frozen one, or a fresh one over the
    /// current fields.
    pub fn create_reader(&mut self) -> Result<Arc<MemoryIndexReader>> {
        if let Some(r) = &self.frozen_reader {
            return Ok(Arc::clone(r));
        }
        Ok(Arc::new(self.build_reader()?))
    }

    /// `search(query)`: the score of the index's one document, `0.0` when
    /// it does not match.
    pub fn search(&mut self, query: &Clause) -> Result<f32> {
        self.create_reader()?.search(query)
    }

    /// The document in the codec writers' terms: its fields in Java's
    /// numbering (creation order) -- term vectors off, since they are served
    /// from `Info`, and payloads left to the postings to record -- each
    /// field's postings and norm, and the stored values in field-name order,
    /// as `MemoryIndexReader` visits them.
    fn document(&mut self) -> Result<Document> {
        let similarity = Arc::clone(&self.similarity);
        let (store_offsets, store_payloads) = (self.store_offsets, self.store_payloads);
        let per = 1 + if store_offsets { 2 } else { 0 } + usize::from(store_payloads);
        let payload_pool = &self.payloads;
        let mut by_number: Vec<&mut Info> = self.fields.values_mut().collect();
        by_number.sort_by_key(|i| i.field_info.number);
        let mut doc = Document::default();
        for info in by_number {
            let mut fi = info.field_info.clone();
            fi.store_term_vectors = false;
            fi.store_payloads = false;
            let number = fi.number;
            let norm = (!info.field_info.omit_norms).then(|| info.norm(similarity.as_ref()));
            let mut terms = Vec::with_capacity(info.terms.len());
            let mut has_payloads = false;
            let (starts, ints) = info.slices();
            for ord in info.sorted_ords() {
                let slice = &ints[starts[ord]..starts[ord + 1]];
                let n = slice.len() / per;
                let mut positions = Vec::with_capacity(n);
                let mut offsets = Vec::with_capacity(if store_offsets { n } else { 0 });
                let mut payload_bytes = Vec::new();
                let mut payload_lengths = Vec::new();
                for occ in slice.chunks(per) {
                    let pos = occ[0];
                    if pos < 0 || positions.last().is_some_and(|&last| pos < last) {
                        return Err(illegal(format!(
                            "position {pos} of term {} in field '{}' is negative or goes backwards: the index cannot hold it",
                            bytes_ref_string(info.terms.get(ord)),
                            info.field_info.name
                        )));
                    }
                    positions.push(pos);
                    if store_offsets {
                        offsets.push((occ[1], occ[2]));
                    }
                    if store_payloads {
                        let bytes = usize::try_from(occ[per - 1])
                            .ok()
                            .and_then(|i| payload_pool.get(i))
                            .map_or(&[][..], Vec::as_slice);
                        has_payloads |= !bytes.is_empty();
                        payload_bytes.extend_from_slice(bytes);
                        payload_lengths.push(u32::try_from(bytes.len()).unwrap_or(u32::MAX));
                    }
                }
                terms.push(TermPostings {
                    term: info.terms.get(ord).to_vec(),
                    docs: vec![(0, info.freq[ord])],
                    positions: vec![positions],
                    offsets: vec![offsets],
                    payload_bytes,
                    payload_lengths,
                });
            }
            if !has_payloads {
                for t in &mut terms {
                    t.payload_bytes.clear();
                    t.payload_lengths.clear();
                }
            }
            let value = |value: FieldValue| StoredField {
                field_number: number,
                value,
            };
            match info.field_info.doc_values_type {
                DocValuesType::None => {}
                DocValuesType::Numeric | DocValuesType::SortedNumeric => doc
                    .doc_values
                    .extend(info.numeric_dv.iter().map(|&v| value(FieldValue::Long(v)))),
                DocValuesType::Binary | DocValuesType::Sorted => doc.doc_values.extend(
                    info.binary_dv
                        .iter()
                        .map(|v| value(FieldValue::Binary(v.clone()))),
                ),
                DocValuesType::SortedSet => doc.doc_values.extend(
                    info.sorted_set_dv
                        .iter()
                        .map(|v| value(FieldValue::Binary(v.clone()))),
                ),
            }
            doc.points.extend(
                info.points
                    .iter()
                    .map(|p| value(FieldValue::Binary(p.clone()))),
            );
            doc.fields.push(FieldPostings {
                info: fi,
                has_payloads,
                terms,
                norm,
            });
        }
        let numbers: HashMap<&str, i32> = self
            .fields
            .iter()
            .map(|(name, info)| (name.as_str(), info.field_info.number))
            .collect();
        for (name, info) in &self.fields {
            for v in &info.stored {
                doc.stored.push(StoredField {
                    field_number: numbers[name.as_str()],
                    value: v.clone(),
                });
            }
        }
        Ok(doc)
    }

    /// The reader over the current fields.
    fn build_reader(&mut self) -> Result<MemoryIndexReader> {
        let doc = self.document()?;
        MemoryIndexReader::new(doc, Arc::clone(&self.similarity))
    }

    /// `toStringDebug()`: every field's terms (as `BytesRef.toString()`),
    /// their frequencies and occurrences, and the totals.
    pub fn to_string_debug(&self) -> String {
        let mut out = String::with_capacity(256);
        let mut sum_positions = 0i64;
        let mut sum_terms = 0usize;
        let per = 1 + if self.store_offsets { 2 } else { 0 } + usize::from(self.store_payloads);
        let iters = if self.store_offsets { 3 } else { 1 };
        for (name, info) in &self.fields {
            out.push_str(name);
            out.push_str(":\n");
            let mut num_positions = 0i64;
            let (starts, ints) = info.slices();
            for ord in info.sorted_ords() {
                let freq = info.freq[ord];
                out.push_str(&format!(
                    "\t'{}':{freq}: [",
                    bytes_ref_string(info.terms.get(ord))
                ));
                let occs: Vec<String> = ints[starts[ord]..starts[ord + 1]]
                    .chunks(per)
                    .map(|occ| {
                        let mut s = occ[..iters]
                            .iter()
                            .map(i32::to_string)
                            .collect::<Vec<_>>()
                            .join(", ");
                        if self.store_payloads {
                            let p = occ[per - 1];
                            if let Some(bytes) =
                                usize::try_from(p).ok().and_then(|i| self.payloads.get(i))
                            {
                                s.push_str(", ");
                                s.push_str(&bytes_ref_string(bytes));
                            }
                        }
                        format!("({s})")
                    })
                    .collect();
                out.push_str(&occs.join(", "));
                out.push_str("]\n");
                num_positions += i64::from(freq);
            }
            out.push_str(&format!(
                "\tterms={}, positions={num_positions}\n",
                info.terms.len()
            ));
            sum_positions += num_positions;
            sum_terms += info.terms.len();
        }
        out.push_str(&format!(
            "\nfields={}, terms={sum_terms}, positions={sum_positions}",
            self.fields.len()
        ));
        out
    }
}

/// `storeDocValues`'s refusal of a second single value.
fn only_one_value(ty: DocValuesType, name: &str) -> Error {
    illegal(format!(
        "Only one value per field allowed for [{}] doc values field [{name}]",
        doc_values_type_name(ty)
    ))
}
