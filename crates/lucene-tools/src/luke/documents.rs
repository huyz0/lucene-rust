//! `org.apache.lucene.luke.models.documents`: one document's fields, term
//! vectors and doc values, and a walk over a field's terms and their
//! postings.
//!
//! Differs: Java's `DocumentsImpl` holds a live `TermsEnum` and
//! `PostingsEnum`. Here the model holds the field and the term it is on, and
//! re-seeks a fresh enum per call; [`Documents::first_term_doc`] reads the
//! term's postings whole, so [`Documents::term_positions`] can be asked again
//! for the same document (Java's second call reads past its positions).

use lucene_codecs::field_infos::{
    DocValuesType, FieldInfo, FieldInfos, IndexOptions, VectorSimilarityFunction,
};
use lucene_codecs::stored_fields::{DocumentVisitor, FieldValue};
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::multi_terms::MultiTerms;
use lucene_search::reader::multi_doc_values;
use lucene_search::reader::{
    leaf_for_doc, merged_field_infos, IndexReader, PostingsEnum, PostingsFlags, TermsEnum,
    NO_MORE_DOCS,
};
use lucene_util::geo::{java_double_string, java_float_string};

use super::{bytes_ref_string, decode, list_string, LukeError, Result};

/// `IndexOptions.name()`.
pub fn index_options_name(o: IndexOptions) -> &'static str {
    match o {
        IndexOptions::None => "NONE",
        IndexOptions::Docs => "DOCS",
        IndexOptions::DocsAndFreqs => "DOCS_AND_FREQS",
        IndexOptions::DocsAndFreqsAndPositions => "DOCS_AND_FREQS_AND_POSITIONS",
        IndexOptions::DocsAndFreqsAndPositionsAndOffsets => {
            "DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS"
        }
        IndexOptions::DocsAndCustomFreqs => "DOCS_AND_CUSTOM_FREQS",
    }
}

/// `DocValuesType.name()`.
pub fn doc_values_type_name(t: DocValuesType) -> &'static str {
    match t {
        DocValuesType::None => "NONE",
        DocValuesType::Numeric => "NUMERIC",
        DocValuesType::Binary => "BINARY",
        DocValuesType::Sorted => "SORTED",
        DocValuesType::SortedSet => "SORTED_SET",
        DocValuesType::SortedNumeric => "SORTED_NUMERIC",
    }
}

/// `VectorSimilarityFunction.name()`.
pub fn vector_similarity_name(f: VectorSimilarityFunction) -> &'static str {
    match f {
        VectorSimilarityFunction::Euclidean => "EUCLIDEAN",
        VectorSimilarityFunction::DotProduct => "DOT_PRODUCT",
        VectorSimilarityFunction::Cosine => "COSINE",
        VectorSimilarityFunction::MaximumInnerProduct => "MAXIMUM_INNER_PRODUCT",
    }
}

/// `FieldInfo.hasNorms()`.
fn has_norms(fi: &FieldInfo) -> bool {
    fi.index_options != IndexOptions::None && !fi.omit_norms
}

/// A stored value as Java's `Field` reports it.
#[derive(Debug, Clone, PartialEq)]
pub enum StoredValue {
    String(String),
    Binary(Vec<u8>),
    /// A number, as `Number.toString()` prints it.
    Number(String),
}

/// `DocumentField`: a field of one document -- its schema, its norm and,
/// when stored, one of its values.
#[derive(Debug, Clone, PartialEq)]
pub struct DocumentField {
    pub name: String,
    pub idx_options: IndexOptions,
    pub has_term_vectors: bool,
    pub has_payloads: bool,
    pub has_norms: bool,
    pub norm: i64,
    pub is_stored: bool,
    pub value: Option<StoredValue>,
    pub dv_type: DocValuesType,
    pub point_dimension_count: i32,
    pub point_num_bytes: i32,
    pub vector_dimension: i32,
    pub vector_similarity: VectorSimilarityFunction,
}

impl DocumentField {
    /// `getStringValue()`: a string or a number's `toString()`.
    pub fn string_value(&self) -> Option<&str> {
        match &self.value {
            Some(StoredValue::String(s) | StoredValue::Number(s)) => Some(s),
            _ => None,
        }
    }

    /// `getBinaryValue()`.
    pub fn binary_value(&self) -> Option<&[u8]> {
        match &self.value {
            Some(StoredValue::Binary(b)) => Some(b),
            _ => None,
        }
    }

    /// `getNumericValue()`.
    pub fn numeric_value(&self) -> Option<&str> {
        match &self.value {
            Some(StoredValue::Number(s)) => Some(s),
            _ => None,
        }
    }
}

impl std::fmt::Display for DocumentField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "DocumentField{{name='{}', idxOptions={}, hasTermVectors={}, isStored={}, \
             dvType={}, pointDimensionCount={}, vectorDimension={}}}",
            self.name,
            index_options_name(self.idx_options),
            self.has_term_vectors,
            self.is_stored,
            doc_values_type_name(self.dv_type),
            self.point_dimension_count,
            self.vector_dimension
        )
    }
}

/// `TermPosting`: one position of a term in a document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TermPosting {
    pub position: i32,
    /// `-1` without offsets.
    pub start_offset: i32,
    /// `-1` without offsets.
    pub end_offset: i32,
    pub payload: Option<Vec<u8>>,
}

impl std::fmt::Display for TermPosting {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TermPosting{{position={}, startOffset={}, endOffset={}, payload={}}}",
            self.position,
            self.start_offset,
            self.end_offset,
            self.payload
                .as_deref()
                .map_or_else(|| "null".to_string(), bytes_ref_string)
        )
    }
}

/// `TermVectorEntry.TermVectorPosition`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TermVectorPosition {
    pub position: i32,
    /// `-1` without offsets.
    pub start_offset: i32,
    /// `-1` without offsets.
    pub end_offset: i32,
}

impl std::fmt::Display for TermVectorPosition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TermVectorPosition{{position={}, startOffset={}, endOffset={}}}",
            self.position, self.start_offset, self.end_offset
        )
    }
}

/// `TermVectorEntry`: a term of a document's term vector.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TermVectorEntry {
    pub term_text: String,
    pub freq: i64,
    pub positions: Vec<TermVectorPosition>,
}

impl std::fmt::Display for TermVectorEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let positions: Vec<String> = self.positions.iter().map(|p| p.to_string()).collect();
        write!(
            f,
            "TermVectorEntry{{termText='{}', freq={}, positions={}}}",
            self.term_text,
            self.freq,
            positions.join(",")
        )
    }
}

/// `DocValues`: one document's doc values of a field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocValues {
    pub dv_type: DocValuesType,
    pub values: Vec<Vec<u8>>,
    pub numeric_values: Vec<i64>,
}

impl std::fmt::Display for DocValues {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let numeric: Vec<String> = self.numeric_values.iter().map(i64::to_string).collect();
        write!(
            f,
            "DocValues{{dvType={}, values={}, numericValues=[{}]}}",
            doc_values_type_name(self.dv_type),
            list_string(self.values.iter().map(|v| bytes_ref_string(v))),
            numeric.join(",")
        )
    }
}

/// A term of the walk: `Term.toString()` is `field:text`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Term {
    pub field: String,
    pub bytes: Vec<u8>,
}

impl std::fmt::Display for Term {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.field, decode(&self.bytes))
    }
}

/// The postings of the current term, read whole.
struct Postings {
    docs: Vec<(i32, Vec<TermPosting>)>,
    current: usize,
}

/// `DocumentsImpl`.
pub struct Documents<'r> {
    reader: &'r DirectoryReader,
    field_infos: FieldInfos,
    cur_field: Option<String>,
    /// The term the terms enum is on; `None` for Java's `tenum == null`.
    term: Option<Vec<u8>>,
    /// Java's `penum`; `None` for `null`.
    postings: Option<Postings>,
}

fn luke(context: String) -> impl FnOnce(lucene_search::Error) -> LukeError {
    move |_| LukeError::luke(context)
}

impl<'r> Documents<'r> {
    /// `new DocumentsImpl(reader)`.
    pub fn new(reader: &'r DirectoryReader) -> Self {
        Documents {
            reader,
            field_infos: merged_field_infos(&reader.leaves()),
            cur_field: None,
            term: None,
            postings: None,
        }
    }

    /// `getFieldNames()`.
    pub fn field_names(&self) -> Vec<String> {
        self.field_infos
            .fields
            .iter()
            .map(|f| f.name.clone())
            .collect()
    }

    /// `getMaxDoc()`.
    pub fn max_doc(&self) -> i32 {
        self.reader.max_doc()
    }

    /// `isLive(docid)`.
    pub fn is_live(&self, doc: i32) -> bool {
        match leaf_for_doc(&self.reader.leaves(), doc) {
            Ok((leaf, local)) => leaf.reader.live_docs().is_none_or(|b| b.get_doc(local)),
            Err(_) => false,
        }
    }

    /// `getDocumentFields(docid)`: for each field of the index, its stored
    /// values, or one value-less entry; nothing for a deleted document.
    ///
    /// # Errors
    /// `LukeException` when the stored fields cannot be read.
    pub fn document_fields(&self, doc: i32) -> Result<Vec<DocumentField>> {
        if !self.is_live(doc) {
            return Ok(Vec::new());
        }
        let err = || format!("Fields information not available for doc {doc}.");
        let (leaf, local) = leaf_for_doc(&self.reader.leaves(), doc).map_err(luke(err()))?;
        let mut visitor = DocumentVisitor::all();
        leaf.reader
            .document(local, &mut visitor)
            .map_err(luke(err()))?;
        let stored = visitor.into_document();
        let leaf_infos = leaf.reader.field_infos();
        let mut res = Vec::new();
        for fi in &self.field_infos.fields {
            let norm = self.norm(fi, doc).map_err(luke(err()))?;
            let values: Vec<StoredValue> = stored
                .fields
                .iter()
                .filter(|s| {
                    leaf_infos
                        .field_by_number(s.field_number)
                        .is_some_and(|l| l.name == fi.name)
                })
                .map(|s| stored_value(&s.value))
                .collect();
            let field = |value: Option<StoredValue>| DocumentField {
                name: fi.name.clone(),
                idx_options: fi.index_options,
                has_term_vectors: fi.store_term_vectors,
                has_payloads: fi.store_payloads,
                has_norms: has_norms(fi),
                norm,
                is_stored: value.is_some(),
                value,
                dv_type: fi.doc_values_type,
                point_dimension_count: fi.point_dimension_count,
                point_num_bytes: fi.point_num_bytes,
                vector_dimension: fi.vector_dimension,
                vector_similarity: fi.vector_similarity_function,
            };
            if values.is_empty() {
                res.push(field(None));
            } else {
                res.extend(values.into_iter().map(|v| field(Some(v))));
            }
        }
        Ok(res)
    }

    /// The field's norm in `doc`, `0` without one.
    fn norm(&self, fi: &FieldInfo, doc: i32) -> lucene_search::Result<i64> {
        if !has_norms(fi) {
            return Ok(0);
        }
        match multi_doc_values::norm_values(self.reader, &fi.name)? {
            Some(mut norms) => {
                if norms.advance_exact(doc)? {
                    Ok(norms.long_value())
                } else {
                    Ok(0)
                }
            }
            None => Ok(0),
        }
    }

    /// `getCurrentField()`.
    pub fn current_field(&self) -> Option<&str> {
        self.cur_field.as_deref()
    }

    /// A fresh enum over `field`'s terms, positioned on `term`.
    fn with_enum<T>(
        &self,
        field: &str,
        position: Option<&[u8]>,
        f: impl FnOnce(&mut dyn TermsEnum) -> lucene_search::Result<T>,
    ) -> lucene_search::Result<Option<T>> {
        let Some(terms) = MultiTerms::get_terms(self.reader, field)? else {
            return Ok(None);
        };
        let mut te = terms.iterator()?;
        if let Some(t) = position {
            if !te.try_seek_exact(t)? {
                return Ok(None);
            }
        }
        f(te.as_mut()).map(Some)
    }

    /// `firstTerm(field)`: the field's first term; the field becomes current
    /// unless it has no terms at all.
    ///
    /// # Errors
    /// `LukeException` when the terms cannot be read.
    pub fn first_term(&mut self, field: &str) -> Result<Option<Term>> {
        self.postings = None;
        let first = self
            .with_enum(field, None, |te| Ok(te.next()?.map(<[u8]>::to_vec)))
            .map_err(luke(format!("Terms not available for field: {field}.")));
        let first = match first {
            Ok(f) => f,
            Err(e) => {
                self.term = None;
                return Err(e);
            }
        };
        match first {
            None => {
                self.cur_field = None;
                self.term = None;
                Ok(None)
            }
            Some(first) => {
                self.cur_field = Some(field.to_string());
                self.term = first;
                Ok(self.current_term())
            }
        }
    }

    fn current_term(&self) -> Option<Term> {
        Some(Term {
            field: self.cur_field.clone()?,
            bytes: self.term.clone()?,
        })
    }

    /// `nextTerm()`.
    ///
    /// # Errors
    /// `LukeException` when the terms cannot be read.
    pub fn next_term(&mut self) -> Result<Option<Term>> {
        self.postings = None;
        let (Some(field), Some(term)) = (self.cur_field.clone(), self.term.take()) else {
            return Ok(None);
        };
        let next = self
            .with_enum(&field, Some(&term), |te| Ok(te.next()?.map(<[u8]>::to_vec)))
            .map_err(luke(format!("Terms not available for field: {field}.")))?;
        self.term = next.flatten();
        Ok(self.current_term())
    }

    /// `seekTerm(termText)`: the current field's first term at or after
    /// `text`.
    ///
    /// # Errors
    /// `LukeException` when the terms cannot be read.
    pub fn seek_term(&mut self, text: &str) -> Result<Option<Term>> {
        self.postings = None;
        let Some(field) = self.cur_field.clone() else {
            return Ok(None);
        };
        let found = self
            .with_enum(&field, None, |te| {
                Ok(match te.try_seek_ceil(text.as_bytes())? {
                    lucene_codecs::blocktree::SeekStatus::End => None,
                    _ => te.term().map(<[u8]>::to_vec),
                })
            })
            .map_err(luke(format!("Terms not available for field: {field}.")));
        self.term = None;
        self.term = found?.flatten();
        Ok(self.current_term())
    }

    /// `getDocFreq()`.
    ///
    /// # Errors
    /// `LukeException` when the term cannot be read.
    pub fn doc_freq(&self) -> Result<Option<i32>> {
        let (Some(field), Some(term)) = (&self.cur_field, &self.term) else {
            return Ok(None);
        };
        self.with_enum(field, Some(term), |te| te.doc_freq())
            .map_err(luke(format!(
                "Doc frequency not available for field: {field}."
            )))
    }

    /// `firstTermDoc()`: the current term's first document (deleted ones
    /// included).
    ///
    /// # Errors
    /// `LukeException` when the postings cannot be read.
    pub fn first_term_doc(&mut self) -> Result<Option<i32>> {
        self.postings = None;
        let (Some(field), Some(term)) = (&self.cur_field, &self.term) else {
            return Ok(None);
        };
        let docs = self
            .with_enum(field, Some(term), |te| {
                read_postings(te.postings(PostingsFlags::All)?.as_mut())
            })
            .map_err(luke(format!("Term docs not available for field: {field}.")))?
            .unwrap_or_default();
        let first = docs.first().map(|d| d.0);
        if first.is_some() {
            self.postings = Some(Postings { docs, current: 0 });
        }
        Ok(first)
    }

    /// `nextTermDoc()`.
    pub fn next_term_doc(&mut self) -> Option<i32> {
        let p = self.postings.as_mut()?;
        p.current = p.current.saturating_add(1);
        match p.docs.get(p.current) {
            Some(d) => Some(d.0),
            None => {
                self.postings = None;
                None
            }
        }
    }

    /// `getTermPositions()`: the current document's positions of the current
    /// term.
    pub fn term_positions(&self) -> Vec<TermPosting> {
        self.postings
            .as_ref()
            .and_then(|p| p.docs.get(p.current))
            .map(|d| d.1.clone())
            .unwrap_or_default()
    }

    /// `getTermVectors(docid, field)`: the document's term vector of `field`,
    /// empty when it has none.
    ///
    /// # Errors
    /// `LukeException` when the term vectors cannot be read.
    pub fn term_vectors(&self, doc: i32, field: &str) -> Result<Vec<TermVectorEntry>> {
        let err = || format!("Term vector not available for doc: #{doc} and field: {field}");
        let (leaf, local) = leaf_for_doc(&self.reader.leaves(), doc).map_err(luke(err()))?;
        let Some(number) = leaf
            .reader
            .field_infos()
            .field_by_name(field)
            .map(|f| f.number)
        else {
            return Ok(Vec::new());
        };
        let Some(tv) = leaf.reader.term_vectors(local).map_err(luke(err()))? else {
            return Ok(Vec::new());
        };
        let Some(tv_field) = tv.fields.iter().find(|f| f.field_number == number) else {
            return Ok(Vec::new());
        };
        Ok(tv_field
            .terms
            .iter()
            .map(|t| {
                let at = |v: &Option<Vec<i32>>, i: usize| -> i32 {
                    v.as_ref().and_then(|v| v.get(i)).copied().unwrap_or(-1)
                };
                let count = usize::try_from(t.freq).unwrap_or(0);
                let positions = (0..count)
                    .filter_map(|i| {
                        let position = at(&t.positions, i);
                        if position < 0 {
                            return None;
                        }
                        let (s, e) = (at(&t.start_offsets, i), at(&t.end_offsets, i));
                        let (start_offset, end_offset) =
                            if s >= 0 && e >= 0 { (s, e) } else { (-1, -1) };
                        Some(TermVectorPosition {
                            position,
                            start_offset,
                            end_offset,
                        })
                    })
                    .collect();
                TermVectorEntry {
                    term_text: decode(&t.term),
                    freq: i64::from(t.freq),
                    positions,
                }
            })
            .collect())
    }

    /// `getDocValues(docid, field)`: the document's values of `field`, by the
    /// field's doc-values type.
    ///
    /// # Errors
    /// `LukeException` when the values cannot be read.
    pub fn doc_values(&self, doc: i32, field: &str) -> Result<Option<DocValues>> {
        let dv_type = self
            .field_infos
            .field_by_name(field)
            .map_or(DocValuesType::None, |f| f.doc_values_type);
        self.read_doc_values(doc, field, dv_type).map_err(|_| {
            LukeError::luke(format!(
                "Doc values not available for doc: #{doc} and field: {field}"
            ))
        })
    }

    fn read_doc_values(
        &self,
        doc: i32,
        field: &str,
        dv_type: DocValuesType,
    ) -> lucene_search::Result<Option<DocValues>> {
        let r = self.reader;
        let of = |values: Vec<Vec<u8>>, numeric_values: Vec<i64>| DocValues {
            dv_type,
            values,
            numeric_values,
        };
        Ok(match dv_type {
            DocValuesType::None => None,
            DocValuesType::Binary => match multi_doc_values::binary_values(r, field)? {
                Some(mut v) => {
                    if v.advance_exact(doc)? {
                        Some(of(vec![v.binary_value().to_vec()], Vec::new()))
                    } else {
                        None
                    }
                }
                None => None,
            },
            DocValuesType::Numeric => match multi_doc_values::numeric_values(r, field)? {
                Some(mut v) => {
                    if v.advance_exact(doc)? {
                        Some(of(Vec::new(), vec![v.long_value()]))
                    } else {
                        None
                    }
                }
                None => None,
            },
            DocValuesType::SortedNumeric => {
                match multi_doc_values::sorted_numeric_values(r, field)? {
                    Some(mut v) => {
                        if v.advance_exact(doc)? {
                            let mut values = Vec::new();
                            for _ in 0..v.doc_value_count() {
                                values.push(v.next_value()?);
                            }
                            Some(of(Vec::new(), values))
                        } else {
                            None
                        }
                    }
                    None => None,
                }
            }
            DocValuesType::Sorted => match multi_doc_values::sorted_values(r, field)? {
                Some(mut v) => {
                    if v.advance_exact(doc)? {
                        let ord = v.ord_value();
                        Some(of(vec![v.lookup_ord(ord)?], Vec::new()))
                    } else {
                        None
                    }
                }
                None => None,
            },
            DocValuesType::SortedSet => match multi_doc_values::sorted_set_values(r, field)? {
                Some(mut v) => {
                    if v.advance_exact(doc)? {
                        let mut values = Vec::new();
                        for _ in 0..v.doc_value_count() {
                            let ord = v.next_ord()?;
                            values.push(v.lookup_ord(ord)?);
                        }
                        Some(of(values, Vec::new()))
                    } else {
                        None
                    }
                }
                None => None,
            },
        })
    }
}

/// A stored value as `DocumentStoredFieldVisitor` hands it to a `Field`.
fn stored_value(v: &FieldValue) -> StoredValue {
    match v {
        FieldValue::String(s) => StoredValue::String(s.clone()),
        FieldValue::Binary(b) => StoredValue::Binary(b.clone()),
        FieldValue::Int(i) => StoredValue::Number(i.to_string()),
        FieldValue::Long(l) => StoredValue::Number(l.to_string()),
        FieldValue::Float(f) => StoredValue::Number(java_float_string(*f)),
        FieldValue::Double(d) => StoredValue::Number(java_double_string(*d)),
    }
}

/// Every document of a postings enum with its `TermPosting`s, as
/// `getTermPositions` reads them: positions below zero (none indexed) are
/// skipped, offsets are `-1` unless both are known, an empty payload is
/// none.
fn read_postings(pe: &mut dyn PostingsEnum) -> lucene_search::Result<Vec<(i32, Vec<TermPosting>)>> {
    let mut docs = Vec::new();
    loop {
        let doc = pe.next_doc()?;
        if doc == NO_MORE_DOCS {
            return Ok(docs);
        }
        let mut postings = Vec::new();
        for _ in 0..pe.freq() {
            let position = pe.next_position()?;
            if position < 0 {
                continue;
            }
            let (s, e) = (pe.start_offset(), pe.end_offset());
            let (start_offset, end_offset) = if s >= 0 && e >= 0 { (s, e) } else { (-1, -1) };
            postings.push(TermPosting {
                position,
                start_offset,
                end_offset,
                payload: pe.payload().filter(|p| !p.is_empty()).map(<[u8]>::to_vec),
            });
        }
        docs.push((doc, postings));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enums_print_their_java_names() {
        let names: Vec<&str> = [
            IndexOptions::None,
            IndexOptions::Docs,
            IndexOptions::DocsAndFreqs,
            IndexOptions::DocsAndFreqsAndPositions,
            IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
            IndexOptions::DocsAndCustomFreqs,
        ]
        .into_iter()
        .map(index_options_name)
        .collect();
        assert_eq!(names[5], "DOCS_AND_CUSTOM_FREQS");
        assert_eq!(names[4], "DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS");
        assert_eq!(doc_values_type_name(DocValuesType::Binary), "BINARY");
        assert_eq!(
            vector_similarity_name(VectorSimilarityFunction::DotProduct),
            "DOT_PRODUCT"
        );
        assert_eq!(
            vector_similarity_name(VectorSimilarityFunction::MaximumInnerProduct),
            "MAXIMUM_INNER_PRODUCT"
        );
    }

    #[test]
    fn values_print_as_their_java_types_do() {
        assert_eq!(
            stored_value(&FieldValue::Float(4.5)),
            StoredValue::Number("4.5".into())
        );
        assert_eq!(
            stored_value(&FieldValue::Double(0.75)),
            StoredValue::Number("0.75".into())
        );
        let p = TermPosting {
            position: 1,
            start_offset: -1,
            end_offset: -1,
            payload: Some(vec![0x61]),
        };
        assert_eq!(
            p.to_string(),
            "TermPosting{position=1, startOffset=-1, endOffset=-1, payload=[61]}"
        );
        let dv = DocValues {
            dv_type: DocValuesType::SortedNumeric,
            values: Vec::new(),
            numeric_values: vec![1, 2],
        };
        assert_eq!(
            dv.to_string(),
            "DocValues{dvType=SORTED_NUMERIC, values=[], numericValues=[1,2]}"
        );
    }
}
