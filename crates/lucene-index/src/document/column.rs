//! The columnar batch API (`org.apache.lucene.document.column`,
//! `IndexWriter.addBatch`): a [`ColumnBatch`] of `num_docs` documents given
//! column by column rather than document by document.
//!
//! Each column names one field and one [`FieldType`], and yields its values
//! either densely (one per document, in order: [`Density::Dense`]) or as
//! `(batch doc id, value)` tuples ([`Density::Sparse`], ascending doc ids, a
//! doc id repeated for a multi-valued field). The kinds are Java's:
//! [`LongColumn`] (numeric doc values and 1-D points, with a
//! [`NumericKind`] saying how the long encodes an int/float/double),
//! [`BinaryColumn`] (bytes or strings: binary/sorted doc values, points,
//! terms, stored values), [`DictionaryColumn`] (ordinals into a dictionary of
//! terms), [`TokenStreamColumn`] (pre-analyzed text) and [`VectorColumn`]
//! (KNN vectors).
//!
//! # What reaches disk
//!
//! Exactly what `addDocuments` of the equivalent documents writes: Java's
//! `IndexingChain.processBatch` feeds the same per-field writers column-wise,
//! and a batch is `numDocs` independent documents (no block). This port
//! validates the batch as `processBatch` does (`ColumnValidation`, the
//! one-column-per-feature rule, doc-id range and order checks), then walks the
//! columns' cursors into each document's fields through the
//! `ColumnFieldAdapter` views -- [`LongColumnAdapter`] and
//! [`BinaryColumnAdapter`] -- and hands the writer those documents. The
//! column-at-a-time writer entry points (`addDenseValues`,
//! `addOrdinalTuples`, ...) are Java's bulk-loading speed-ups over the same
//! values, not a different format.
//!
//! Rust shapes: Java's abstract column and cursor classes are traits; the
//! batch lists its columns as [`BatchColumn`]s, Java's sealed `switch` over
//! the column type. [`VecLongTupleCursor`] and friends are ready-made
//! in-memory cursors.

use std::borrow::Cow;
use std::sync::Arc;

use lucene_analysis::Analyzer;

use super::numeric::{
    int_to_sortable_bytes, long_to_sortable_bytes, sortable_int_to_float, sortable_long_to_double,
};
use super::vectors::{KnnByteVectorField, KnnFloatVectorField};
use super::{
    illegal, DocValuesType, Document, FieldTokens, FieldType, IndexOptions, IndexableField,
    InvertableType, Number, Result, StoredValue,
};
use crate::buffered_updates::{SeqNo, Term};
use crate::index_writer::{IndexWriter, VectorValue};

/// `DocIdSetIterator.NO_MORE_DOCS`: a cursor's end.
pub const NO_MORE_DOCS: i32 = i32::MAX;

/// `Column.Density`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Density {
    /// One value for every document of the batch, in doc-id order.
    Dense,
    /// `(doc id, value)` tuples.
    Sparse,
}

/// `LongColumn.NumericKind`: what the column's longs encode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumericKind {
    /// An `int`: points of 4 bytes.
    Int,
    Long,
    /// `NumericUtils.floatToSortableInt`: points of 4 bytes.
    Float,
    /// `NumericUtils.doubleToSortableLong`.
    Double,
}

/// `StoredValue.Type`, as far as a column can store.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoredType {
    Integer,
    Long,
    Float,
    Double,
    String,
    Binary,
}

impl NumericKind {
    /// `LongColumn.storedType()`.
    pub fn stored_type(self) -> StoredType {
        match self {
            NumericKind::Int => StoredType::Integer,
            NumericKind::Long => StoredType::Long,
            NumericKind::Float => StoredType::Float,
            NumericKind::Double => StoredType::Double,
        }
    }

    fn point_bytes(self) -> i32 {
        match self {
            NumericKind::Int | NumericKind::Float => 4,
            NumericKind::Long | NumericKind::Double => 8,
        }
    }
}

/// `Column`: the name, type and density every column has.
#[derive(Debug, Clone, PartialEq)]
pub struct Column {
    name: String,
    field_type: FieldType,
    density: Density,
}

impl Column {
    /// `Column(name, fieldType, density)`.
    pub fn new(name: impl Into<String>, field_type: FieldType, density: Density) -> Self {
        Column {
            name: name.into(),
            field_type,
            density,
        }
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn field_type(&self) -> &FieldType {
        &self.field_type
    }
    pub fn density(&self) -> Density {
        self.density
    }
}

/// `LongTupleCursor`: `(doc id, long)` tuples; [`NO_MORE_DOCS`] ends it.
pub trait LongTupleCursor {
    fn next_doc(&mut self) -> i32;
    fn long_value(&self) -> i64;
}

/// `LongValuesCursor`: one long per document of a dense column.
pub trait LongValuesCursor {
    fn size(&self) -> usize;
    fn next_long(&mut self) -> i64;
}

/// `ObjectTupleCursor<T>`: `(doc id, value)` tuples.
pub trait ObjectTupleCursor<T> {
    fn next_doc(&mut self) -> i32;
    fn value(&self) -> T;
}

/// `BytesRefValuesCursor`: one byte string per document of a dense column.
pub trait BytesRefValuesCursor {
    fn size(&self) -> usize;
    fn next_value(&mut self) -> Vec<u8>;
}

/// `OrdinalsCursor`: one dictionary ordinal per document of a dense column.
pub trait OrdinalsCursor {
    fn size(&self) -> usize;
    fn next_ord(&mut self) -> i32;
}

/// `OrdinalsTupleCursor`: `(doc id, ordinal)` tuples.
pub trait OrdinalsTupleCursor {
    fn next_doc(&mut self) -> i32;
    fn ord_value(&self) -> i32;
}

/// `LongColumn`: numeric doc values and/or a 1-D numeric point, and stored
/// numbers.
pub trait LongColumn {
    fn column(&self) -> &Column;
    /// `numericKind()`; `LONG` unless given.
    fn numeric_kind(&self) -> NumericKind {
        NumericKind::Long
    }
    fn tuples(&self) -> Box<dyn LongTupleCursor + '_>;
    /// `values()`: only a dense column has them.
    fn values(&self) -> Result<Box<dyn LongValuesCursor + '_>> {
        Err(dense_only(self.column()))
    }
}

/// `BinaryColumn`: bytes for binary/sorted/sorted-set doc values, points,
/// terms and stored values.
pub trait BinaryColumn {
    fn column(&self) -> &Column;
    /// `storedType()`: `BINARY` unless given (`STRING` stores UTF-8 text).
    fn stored_type(&self) -> StoredType {
        StoredType::Binary
    }
    fn tuples(&self) -> Box<dyn ObjectTupleCursor<Vec<u8>> + '_>;
    fn values(&self) -> Result<Box<dyn BytesRefValuesCursor + '_>> {
        Err(dense_only(self.column()))
    }
}

/// `DictionaryColumn`: ordinals into a dictionary of terms.
pub trait DictionaryColumn {
    fn column(&self) -> &Column;
    fn dictionary(&self) -> &[Vec<u8>];
    fn stored_type(&self) -> StoredType {
        StoredType::Binary
    }
    fn tuples(&self) -> Box<dyn OrdinalsTupleCursor + '_>;
    fn values(&self) -> Result<Box<dyn OrdinalsCursor + '_>> {
        Err(dense_only(self.column()))
    }
}

/// `TokenStreamColumn`: pre-analyzed token streams for an inverted-only
/// field.
pub trait TokenStreamColumn {
    fn column(&self) -> &Column;
    fn tuples(&self) -> Box<dyn ObjectTupleCursor<FieldTokens> + '_>;
}

/// `VectorColumn<T>`: KNN vectors, `Vec<f32>` or `Vec<u8>`.
pub trait VectorColumn<T> {
    fn column(&self) -> &Column;
    fn tuples(&self) -> Box<dyn ObjectTupleCursor<T> + '_>;
}

fn dense_only(c: &Column) -> super::Error {
    illegal(format!(
        "values() requires density() == DENSE for column \"{}\"",
        c.name()
    ))
}

/// One column of a batch: Java's `switch (column)` over the column classes.
#[derive(Clone, Copy)]
pub enum BatchColumn<'a> {
    Long(&'a dyn LongColumn),
    Binary(&'a dyn BinaryColumn),
    Dictionary(&'a dyn DictionaryColumn),
    TokenStream(&'a dyn TokenStreamColumn),
    FloatVector(&'a dyn VectorColumn<Vec<f32>>),
    ByteVector(&'a dyn VectorColumn<Vec<u8>>),
}

impl<'a> BatchColumn<'a> {
    pub fn column(&self) -> &'a Column {
        match self {
            BatchColumn::Long(c) => c.column(),
            BatchColumn::Binary(c) => c.column(),
            BatchColumn::Dictionary(c) => c.column(),
            BatchColumn::TokenStream(c) => c.column(),
            BatchColumn::FloatVector(c) => c.column(),
            BatchColumn::ByteVector(c) => c.column(),
        }
    }
}

/// `ColumnBatch`.
pub trait ColumnBatch {
    fn num_docs(&self) -> usize;
    fn columns(&self) -> Vec<BatchColumn<'_>>;
}

/// `ColumnValidation`: the checks `processBatch` makes of each column.
#[derive(Debug, Clone, Copy)]
pub struct ColumnValidation;

impl ColumnValidation {
    pub const FEATURE_INVERSION: u8 = 1;
    pub const FEATURE_STORED: u8 = 1 << 1;
    pub const FEATURE_DOCVALUES: u8 = 1 << 2;
    pub const FEATURE_POINTS: u8 = 1 << 3;
    pub const FEATURE_VECTOR: u8 = 1 << 4;

    /// `featureMask(fieldType)`.
    pub fn feature_mask(ft: &FieldType) -> u8 {
        let mut mask = 0;
        if ft.index_options() != IndexOptions::None {
            mask |= Self::FEATURE_INVERSION;
        }
        if ft.stored() {
            mask |= Self::FEATURE_STORED;
        }
        if ft.doc_values_type() != DocValuesType::None {
            mask |= Self::FEATURE_DOCVALUES;
        }
        if ft.point_dimension_count() != 0 {
            mask |= Self::FEATURE_POINTS;
        }
        if ft.vector_dimension() != 0 {
            mask |= Self::FEATURE_VECTOR;
        }
        mask
    }

    /// `featureNames(mask)`.
    pub fn feature_names(mask: u8) -> String {
        let names = [
            (Self::FEATURE_INVERSION, "inversion"),
            (Self::FEATURE_STORED, "stored"),
            (Self::FEATURE_DOCVALUES, "doc values"),
            (Self::FEATURE_POINTS, "points"),
            (Self::FEATURE_VECTOR, "vectors"),
        ];
        let parts: Vec<&str> = names
            .iter()
            .filter(|(bit, _)| mask & bit != 0)
            .map(|(_, n)| *n)
            .collect();
        format!("[{}]", parts.join(", "))
    }

    /// `validateColumnHasIndexingFeature(fieldName, fieldType)`.
    pub fn validate_column_has_indexing_feature(name: &str, ft: &FieldType) -> Result<()> {
        if Self::feature_mask(ft) == 0 {
            return Err(illegal(format!(
                "Column \"{name}\" must have a non-NONE docValuesType, point dimensions, be \
                 stored, have index options, or have vector dimensions"
            )));
        }
        Ok(())
    }

    /// `validateLongColumn(column, fieldType)`.
    pub fn validate_long_column(column: &dyn LongColumn) -> Result<()> {
        let c = column.column();
        let ft = c.field_type();
        let dims = ft.point_dimension_count();
        if dims != 0 {
            if dims != 1 {
                return Err(illegal(format!(
                    "LongColumn \"{}\" only supports 1-dimensional point fields, got \
                     pointDimensionCount={dims}",
                    c.name()
                )));
            }
            let kind = column.numeric_kind();
            if ft.point_num_bytes() != kind.point_bytes() {
                return Err(illegal(format!(
                    "LongColumn \"{}\" numericKind={kind:?} requires pointNumBytes={}, got {}",
                    c.name(),
                    kind.point_bytes(),
                    ft.point_num_bytes()
                )));
            }
        }
        // Java lets an indexed `LongColumn` through here and then fails on
        // its `null` invertable type; refused up front.
        if ft.index_options() != IndexOptions::None {
            return Err(illegal(format!(
                "LongColumn \"{}\" cannot be inverted",
                c.name()
            )));
        }
        if !matches!(
            ft.doc_values_type(),
            DocValuesType::None | DocValuesType::Numeric | DocValuesType::SortedNumeric
        ) {
            return Err(illegal(format!(
                "LongColumn \"{}\" has incompatible docValuesType: {}",
                c.name(),
                super::doc_values_type_name(ft.doc_values_type())
            )));
        }
        Ok(())
    }

    fn check_bytes_stored(
        kind: &str,
        c: &Column,
        stored: StoredType,
        use_other: &str,
    ) -> Result<()> {
        if c.field_type().stored() && !matches!(stored, StoredType::Binary | StoredType::String) {
            return Err(illegal(format!(
                "{kind} \"{}\" storedType={stored:?} is not supported; use a {use_other} for \
                 numeric stored data",
                c.name()
            )));
        }
        Ok(())
    }

    /// `validateBinaryColumn(column, fieldType)`.
    pub fn validate_binary_column(column: &dyn BinaryColumn) -> Result<()> {
        let c = column.column();
        let dv = c.field_type().doc_values_type();
        if matches!(dv, DocValuesType::Numeric | DocValuesType::SortedNumeric) {
            return Err(illegal(format!(
                "BinaryColumn \"{}\" cannot feed docValuesType={}; use a LongColumn",
                c.name(),
                super::doc_values_type_name(dv)
            )));
        }
        Self::check_bytes_stored("BinaryColumn", c, column.stored_type(), "LongColumn")
    }

    /// `validateDictionaryColumn(column, fieldType)`.
    pub fn validate_dictionary_column(column: &dyn DictionaryColumn) -> Result<()> {
        let c = column.column();
        let ft = c.field_type();
        let dv = ft.doc_values_type();
        if column.dictionary().is_empty() {
            return Err(illegal(format!(
                "DictionaryColumn \"{}\": dictionary must not be empty",
                c.name()
            )));
        }
        if let Some(i) = column
            .dictionary()
            .iter()
            .position(|e| e.len() > super::indexing::MAX_TERM_LENGTH)
        {
            return Err(illegal(format!(
                "DictionaryColumn \"{}\": dictionary entry at index {i} is too large, must be <= \
                 {}",
                c.name(),
                super::indexing::MAX_TERM_LENGTH
            )));
        }
        if matches!(dv, DocValuesType::Numeric | DocValuesType::SortedNumeric) {
            return Err(illegal(format!(
                "DictionaryColumn \"{}\" cannot feed docValuesType={}; use a LongColumn",
                c.name(),
                super::doc_values_type_name(dv)
            )));
        }
        if dv == DocValuesType::Binary {
            return Err(illegal(format!(
                "DictionaryColumn \"{}\" cannot feed docValuesType=BINARY (the writer does not \
                 dedup terms, so the dictionary provides no benefit); use a BinaryColumn",
                c.name()
            )));
        }
        if ft.point_dimension_count() != 0 {
            return Err(illegal(format!(
                "DictionaryColumn \"{}\" does not support points (pointDimensionCount must be 0)",
                c.name()
            )));
        }
        Self::check_bytes_stored("DictionaryColumn", c, column.stored_type(), "LongColumn")
    }

    /// `validateTokenStreamColumn(column, fieldType)`.
    pub fn validate_token_stream_column(c: &Column) -> Result<()> {
        let ft = c.field_type();
        if ft.index_options() == IndexOptions::None || !ft.tokenized() {
            return Err(illegal(format!(
                "TokenStreamColumn \"{}\" requires indexOptions != NONE and tokenized == true; \
                 got indexOptions={}, tokenized={}",
                c.name(),
                super::index_options_name(ft.index_options()),
                ft.tokenized()
            )));
        }
        if ft.stored()
            || ft.doc_values_type() != DocValuesType::None
            || ft.point_dimension_count() != 0
            || ft.vector_dimension() != 0
        {
            return Err(illegal(format!(
                "TokenStreamColumn \"{}\" must be inverted-only: stored=false, \
                 docValuesType=NONE, pointDimensionCount=0, vectorDimension=0",
                c.name()
            )));
        }
        Ok(())
    }

    /// `validateVectorColumn(column, fieldType)`.
    pub fn validate_vector_column(c: &Column) -> Result<()> {
        let ft = c.field_type();
        if ft.vector_dimension() <= 0 {
            return Err(illegal(format!(
                "VectorColumn \"{}\" requires fieldType.vectorDimension() > 0; got {}",
                c.name(),
                ft.vector_dimension()
            )));
        }
        if ft.doc_values_type() != DocValuesType::None
            || ft.point_dimension_count() != 0
            || ft.stored()
            || ft.index_options() != IndexOptions::None
        {
            return Err(illegal(format!(
                "VectorColumn \"{}\" must be vector-only: docValuesType=NONE, \
                 pointDimensionCount=0, stored=false, indexOptions=NONE",
                c.name()
            )));
        }
        Ok(())
    }

    /// `checkDocID(column, batchDocID, numDocs)`.
    pub fn check_doc_id(c: &Column, batch_doc_id: i32, num_docs: usize) -> Result<usize> {
        match usize::try_from(batch_doc_id) {
            Ok(d) if d < num_docs => Ok(d),
            _ => Err(illegal(format!(
                "Column \"{}\" returned batch doc-id {batch_doc_id} which is out of range [0, \
                 {num_docs})",
                c.name()
            ))),
        }
    }

    /// `checkDenseCount(column, consumed, numDocs)`.
    pub fn check_dense_count(c: &Column, consumed: usize, num_docs: usize) -> Result<()> {
        if consumed != num_docs {
            return Err(illegal(format!(
                "Dense column \"{}\" provided {consumed} values but batch has {num_docs} \
                 documents",
                c.name()
            )));
        }
        Ok(())
    }
}

/// A column's value for one document, the view `ColumnFieldAdapter` gives
/// the indexing chain of it.
#[derive(Debug, Clone)]
enum CellValue {
    Bytes(Arc<[u8]>),
    Tokens(FieldTokens),
}

/// `ColumnFieldAdapter`'s `LongColumnAdapter`: one value of a
/// [`LongColumn`] as an [`IndexableField`] -- the long as the doc value, its
/// sortable bytes (by [`NumericKind`]) as the point, the number as the
/// stored value.
#[derive(Debug, Clone)]
pub struct LongColumnAdapter {
    name: Arc<str>,
    field_type: Arc<FieldType>,
    kind: NumericKind,
    value: i64,
}

impl LongColumnAdapter {
    pub fn new(column: &Column, kind: NumericKind, value: i64) -> Self {
        LongColumnAdapter {
            name: column.name().into(),
            field_type: Arc::new(column.field_type().clone()),
            kind,
            value,
        }
    }
}

impl IndexableField for LongColumnAdapter {
    fn name(&self) -> &str {
        &self.name
    }
    fn field_type(&self) -> &FieldType {
        &self.field_type
    }
    fn numeric_value(&self) -> Option<Number> {
        Some(Number::Long(self.value))
    }
    /// `encodeSortablePointBytes(raw, kind)`.
    fn binary_value(&self) -> Option<Cow<'_, [u8]>> {
        Some(Cow::Owned(match self.kind {
            NumericKind::Int | NumericKind::Float => {
                int_to_sortable_bytes(self.value as i32).to_vec()
            }
            NumericKind::Long | NumericKind::Double => long_to_sortable_bytes(self.value).to_vec(),
        }))
    }
    fn stored_value(&self) -> Option<StoredValue> {
        if !self.field_type.stored() {
            return None;
        }
        Some(match self.kind {
            NumericKind::Int => StoredValue::Int(self.value as i32),
            NumericKind::Long => StoredValue::Long(self.value),
            NumericKind::Float => StoredValue::Float(sortable_int_to_float(self.value as i32)),
            NumericKind::Double => StoredValue::Double(sortable_long_to_double(self.value)),
        })
    }
    fn token_stream(&self, _analyzer: &Analyzer) -> Result<Option<FieldTokens>> {
        Ok(None)
    }
}

/// `ColumnFieldAdapter`'s `BinaryColumnAdapter`: one value of a
/// [`BinaryColumn`], a [`DictionaryColumn`] (its dictionary entry) or a
/// [`TokenStreamColumn`] as an [`IndexableField`].
#[derive(Debug, Clone)]
pub struct BinaryColumnAdapter {
    name: Arc<str>,
    field_type: Arc<FieldType>,
    stored_type: Option<StoredType>,
    value: CellValue,
}

impl BinaryColumnAdapter {
    /// A bytes value, stored as `stored_type` when the type is stored.
    pub fn new(column: &Column, stored_type: StoredType, value: Arc<[u8]>) -> Self {
        BinaryColumnAdapter {
            name: column.name().into(),
            field_type: Arc::new(column.field_type().clone()),
            stored_type: column.field_type().stored().then_some(stored_type),
            value: CellValue::Bytes(value),
        }
    }

    /// A caller-supplied token stream (`TokenStreamColumn`).
    pub fn from_tokens(column: &Column, tokens: FieldTokens) -> Self {
        BinaryColumnAdapter {
            name: column.name().into(),
            field_type: Arc::new(column.field_type().clone()),
            stored_type: None,
            value: CellValue::Tokens(tokens),
        }
    }

    fn bytes(&self) -> Option<&[u8]> {
        match &self.value {
            CellValue::Bytes(b) => Some(b),
            _ => None,
        }
    }

    fn decoded(&self) -> Option<String> {
        self.bytes()
            .map(|b| String::from_utf8_lossy(b).into_owned())
    }
}

impl IndexableField for BinaryColumnAdapter {
    fn name(&self) -> &str {
        &self.name
    }
    fn field_type(&self) -> &FieldType {
        &self.field_type
    }
    fn binary_value(&self) -> Option<Cow<'_, [u8]>> {
        self.bytes().map(Cow::Borrowed)
    }
    /// The UTF-8 text, for a tokenized field only.
    fn string_value(&self) -> Option<Cow<'_, str>> {
        if self.field_type.tokenized() {
            self.decoded().map(Cow::Owned)
        } else {
            None
        }
    }
    fn stored_value(&self) -> Option<StoredValue> {
        match self.stored_type? {
            StoredType::String => self.decoded().map(StoredValue::String),
            _ => self.bytes().map(|b| StoredValue::Binary(b.to_vec())),
        }
    }
    fn invertable_type(&self) -> InvertableType {
        if self.field_type.tokenized() {
            InvertableType::TokenStream
        } else {
            InvertableType::Binary
        }
    }
    fn token_stream(&self, analyzer: &Analyzer) -> Result<Option<FieldTokens>> {
        match &self.value {
            CellValue::Tokens(t) => Ok(Some(t.clone())),
            _ if self.field_type.tokenized() => {
                Ok(self.decoded().map(|s| analyzer.analyze_stream(&s).into()))
            }
            _ => Ok(None),
        }
    }
}

/// The documents a batch describes: every column walked, checked and
/// adapted.
fn batch_documents(batch: &dyn ColumnBatch) -> Result<Vec<Document>> {
    let num_docs = batch.num_docs();
    let columns = batch.columns();
    // Validation pass: `processBatch`'s first loop.
    let mut features: Vec<(&str, u8)> = Vec::new();
    for col in &columns {
        let c = col.column();
        ColumnValidation::validate_column_has_indexing_feature(c.name(), c.field_type())?;
        match col {
            BatchColumn::Long(l) => ColumnValidation::validate_long_column(*l)?,
            BatchColumn::Binary(b) => ColumnValidation::validate_binary_column(*b)?,
            BatchColumn::Dictionary(d) => ColumnValidation::validate_dictionary_column(*d)?,
            BatchColumn::TokenStream(_) => ColumnValidation::validate_token_stream_column(c)?,
            BatchColumn::FloatVector(_) | BatchColumn::ByteVector(_) => {
                ColumnValidation::validate_vector_column(c)?
            }
        }
        let mask = ColumnValidation::feature_mask(c.field_type());
        match features.iter_mut().find(|(n, _)| *n == c.name()) {
            None => features.push((c.name(), mask)),
            Some((_, seen)) => {
                let overlap = *seen & mask;
                if overlap != 0 {
                    return Err(illegal(format!(
                        "ColumnBatch has multiple columns for field \"{}\" claiming the same \
                         indexing feature {}; each feature may appear in at most one column.",
                        c.name(),
                        ColumnValidation::feature_names(overlap)
                    )));
                }
                *seen |= mask;
            }
        }
    }

    let mut docs: Vec<Vec<Box<dyn IndexableField>>> = (0..num_docs).map(|_| Vec::new()).collect();
    for col in &columns {
        let c = col.column();
        match col {
            BatchColumn::Long(l) => {
                let kind = l.numeric_kind();
                if c.density() == Density::Dense {
                    let mut cursor = l.values()?;
                    ColumnValidation::check_dense_count(c, cursor.size(), num_docs)?;
                    for fields in &mut docs {
                        fields.push(Box::new(LongColumnAdapter::new(
                            c,
                            kind,
                            cursor.next_long(),
                        )));
                    }
                } else {
                    let mut cursor = l.tuples();
                    let mut last = -1;
                    loop {
                        let doc = cursor.next_doc();
                        if doc == NO_MORE_DOCS {
                            break;
                        }
                        let d = check_order(c, doc, &mut last, num_docs)?;
                        docs[d].push(Box::new(LongColumnAdapter::new(
                            c,
                            kind,
                            cursor.long_value(),
                        )));
                    }
                }
            }
            BatchColumn::Binary(b) => {
                let stored = b.stored_type();
                if c.density() == Density::Dense {
                    let mut cursor = b.values()?;
                    ColumnValidation::check_dense_count(c, cursor.size(), num_docs)?;
                    for fields in &mut docs {
                        let v: Arc<[u8]> = cursor.next_value().into();
                        fields.push(Box::new(BinaryColumnAdapter::new(c, stored, v)));
                    }
                } else {
                    let mut cursor = b.tuples();
                    let mut cells: Vec<(usize, Vec<u8>)> = Vec::new();
                    let mut last = -1;
                    loop {
                        let doc = cursor.next_doc();
                        if doc == NO_MORE_DOCS {
                            break;
                        }
                        let d = check_order(c, doc, &mut last, num_docs)?;
                        cells.push((d, cursor.value()));
                    }
                    for (d, v) in cells {
                        docs[d].push(Box::new(BinaryColumnAdapter::new(c, stored, v.into())));
                    }
                }
            }
            BatchColumn::Dictionary(dc) => {
                let stored = dc.stored_type();
                let dict: Vec<Arc<[u8]>> = dc
                    .dictionary()
                    .iter()
                    .map(|e| e.as_slice().into())
                    .collect();
                let entry = |ord: i32| -> Result<Arc<[u8]>> {
                    usize::try_from(ord)
                        .ok()
                        .and_then(|o| dict.get(o))
                        .cloned()
                        .ok_or_else(|| {
                            illegal(format!(
                                "DictionaryColumn \"{}\" ordinal {ord} is out of range [0, {})",
                                c.name(),
                                dict.len()
                            ))
                        })
                };
                if c.density() == Density::Dense {
                    let mut cursor = dc.values()?;
                    ColumnValidation::check_dense_count(c, cursor.size(), num_docs)?;
                    for fields in &mut docs {
                        let v = entry(cursor.next_ord())?;
                        fields.push(Box::new(BinaryColumnAdapter::new(c, stored, v)));
                    }
                } else {
                    let mut cursor = dc.tuples();
                    let mut last = -1;
                    loop {
                        let doc = cursor.next_doc();
                        if doc == NO_MORE_DOCS {
                            break;
                        }
                        let d = check_order(c, doc, &mut last, num_docs)?;
                        let v = entry(cursor.ord_value())?;
                        docs[d].push(Box::new(BinaryColumnAdapter::new(c, stored, v)));
                    }
                }
            }
            BatchColumn::TokenStream(t) => {
                let mut cursor = t.tuples();
                let mut last = -1;
                loop {
                    let doc = cursor.next_doc();
                    if doc == NO_MORE_DOCS {
                        break;
                    }
                    let d = check_order(c, doc, &mut last, num_docs)?;
                    docs[d].push(Box::new(BinaryColumnAdapter::from_tokens(
                        c,
                        cursor.value(),
                    )));
                }
            }
            BatchColumn::FloatVector(v) => {
                let mut cursor = v.tuples();
                let ft = c.field_type().clone();
                collect_vectors(
                    c,
                    num_docs,
                    || {
                        let doc = cursor.next_doc();
                        (
                            doc,
                            (doc != NO_MORE_DOCS).then(|| VectorValue::Float32(cursor.value())),
                        )
                    },
                    |d, value| {
                        if let VectorValue::Float32(v) = value {
                            docs[d].push(Box::new(KnnFloatVectorField::with_type(
                                c.name(),
                                v,
                                ft.clone(),
                            )));
                        }
                    },
                )?;
            }
            BatchColumn::ByteVector(v) => {
                let mut cursor = v.tuples();
                let ft = c.field_type().clone();
                collect_vectors(
                    c,
                    num_docs,
                    || {
                        let doc = cursor.next_doc();
                        (
                            doc,
                            (doc != NO_MORE_DOCS).then(|| VectorValue::Byte(cursor.value())),
                        )
                    },
                    |d, value| {
                        if let VectorValue::Byte(v) = value {
                            docs[d].push(Box::new(KnnByteVectorField::with_type(
                                c.name(),
                                v,
                                ft.clone(),
                            )));
                        }
                    },
                )?;
            }
        }
    }
    let docs = docs
        .into_iter()
        .map(|fields| {
            let mut d = Document::new();
            for f in fields {
                d.add_boxed(f);
            }
            d
        })
        .collect();
    Ok(docs)
}

/// A sparse cursor's doc ids: in range, never descending (a repeat is
/// another value of the same document).
fn check_order(c: &Column, doc: i32, last: &mut i32, num_docs: usize) -> Result<usize> {
    let d = ColumnValidation::check_doc_id(c, doc, num_docs)?;
    if doc < *last {
        return Err(illegal(format!(
            "Row column \"{}\" returned out-of-order batch doc-id {doc}",
            c.name()
        )));
    }
    *last = doc;
    Ok(d)
}

/// `processVectorColumn`'s checks: in range, strictly increasing, the
/// field's dimension.
fn collect_vectors(
    c: &Column,
    num_docs: usize,
    mut next: impl FnMut() -> (i32, Option<VectorValue>),
    mut on_vector: impl FnMut(usize, VectorValue),
) -> Result<()> {
    let dim = c.field_type().vector_dimension();
    let mut prev = -1;
    loop {
        let (doc, value) = next();
        let Some(value) = value else {
            return Ok(());
        };
        let d = ColumnValidation::check_doc_id(c, doc, num_docs)?;
        if doc <= prev {
            return Err(illegal(format!(
                "VectorColumn \"{}\" must yield strictly increasing batch doc-ids; got {doc} \
                 after {prev}",
                c.name()
            )));
        }
        let len = match &value {
            VectorValue::Float32(v) => v.len(),
            VectorValue::Byte(v) => v.len(),
        };
        if i32::try_from(len).ok() != Some(dim) {
            return Err(illegal(format!(
                "VectorColumn \"{}\" expected dimension {dim} but got vector of length {len} at \
                 batch doc {doc}",
                c.name()
            )));
        }
        prev = doc;
        on_vector(d, value);
    }
}

impl IndexWriter<'_> {
    /// `IndexWriter.addBatch(columnBatch)`: `num_docs` independent documents
    /// given column-wise. See the module doc.
    pub fn add_batch(&mut self, batch: &dyn ColumnBatch) -> crate::index_writer::Result<SeqNo> {
        let docs = batch_documents(batch)?;
        self.register_batch(batch)?;
        self.add_fields_documents_with_vectors(&docs, None)
    }

    /// The batch's fields, numbered in column order.
    fn register_batch(&mut self, batch: &dyn ColumnBatch) -> crate::index_writer::Result<()> {
        let columns = batch.columns();
        let types: Vec<(&str, &FieldType)> = columns
            .iter()
            .map(|c| (c.column().name(), c.column().field_type()))
            .collect();
        self.register_batch_field_types(&types)
    }

    /// `IndexWriter.updateDocuments(delTerm, columnBatch)`.
    pub fn update_batch(
        &mut self,
        term: Term,
        batch: &dyn ColumnBatch,
    ) -> crate::index_writer::Result<SeqNo> {
        let docs = batch_documents(batch)?;
        self.register_batch(batch)?;
        self.add_fields_documents_with_vectors(&docs, Some(term))
    }
}

/// A [`LongTupleCursor`] over `(doc, value)` pairs.
#[derive(Debug, Clone)]
pub struct VecLongTupleCursor<'a> {
    tuples: &'a [(i32, i64)],
    at: Option<usize>,
}

impl<'a> VecLongTupleCursor<'a> {
    pub fn new(tuples: &'a [(i32, i64)]) -> Self {
        VecLongTupleCursor { tuples, at: None }
    }
}

impl LongTupleCursor for VecLongTupleCursor<'_> {
    fn next_doc(&mut self) -> i32 {
        let next = self.at.map_or(0, |i| i.saturating_add(1));
        self.at = Some(next);
        self.tuples.get(next).map_or(NO_MORE_DOCS, |t| t.0)
    }
    fn long_value(&self) -> i64 {
        self.at.and_then(|i| self.tuples.get(i)).map_or(0, |t| t.1)
    }
}

/// A [`LongValuesCursor`] over a slice.
#[derive(Debug, Clone)]
pub struct VecLongValuesCursor<'a> {
    values: &'a [i64],
    at: usize,
}

impl<'a> VecLongValuesCursor<'a> {
    pub fn new(values: &'a [i64]) -> Self {
        VecLongValuesCursor { values, at: 0 }
    }
}

impl LongValuesCursor for VecLongValuesCursor<'_> {
    fn size(&self) -> usize {
        self.values.len()
    }
    fn next_long(&mut self) -> i64 {
        let v = self.values.get(self.at).copied().unwrap_or(0);
        self.at = self.at.saturating_add(1);
        v
    }
}

/// An [`ObjectTupleCursor`] over `(doc, value)` pairs.
#[derive(Debug, Clone)]
pub struct VecObjectTupleCursor<'a, T> {
    tuples: &'a [(i32, T)],
    at: Option<usize>,
}

impl<'a, T> VecObjectTupleCursor<'a, T> {
    pub fn new(tuples: &'a [(i32, T)]) -> Self {
        VecObjectTupleCursor { tuples, at: None }
    }
}

impl<T: Clone + Default> ObjectTupleCursor<T> for VecObjectTupleCursor<'_, T> {
    fn next_doc(&mut self) -> i32 {
        let next = self.at.map_or(0, |i| i.saturating_add(1));
        self.at = Some(next);
        self.tuples.get(next).map_or(NO_MORE_DOCS, |t| t.0)
    }
    fn value(&self) -> T {
        self.at
            .and_then(|i| self.tuples.get(i))
            .map(|t| t.1.clone())
            .unwrap_or_default()
    }
}

/// A [`BytesRefValuesCursor`] over a slice.
#[derive(Debug, Clone)]
pub struct VecBytesValuesCursor<'a> {
    values: &'a [Vec<u8>],
    at: usize,
}

impl<'a> VecBytesValuesCursor<'a> {
    pub fn new(values: &'a [Vec<u8>]) -> Self {
        VecBytesValuesCursor { values, at: 0 }
    }
}

impl BytesRefValuesCursor for VecBytesValuesCursor<'_> {
    fn size(&self) -> usize {
        self.values.len()
    }
    fn next_value(&mut self) -> Vec<u8> {
        let v = self.values.get(self.at).cloned().unwrap_or_default();
        self.at = self.at.saturating_add(1);
        v
    }
}

/// An [`OrdinalsCursor`] over a slice.
#[derive(Debug, Clone)]
pub struct VecOrdinalsCursor<'a> {
    ords: &'a [i32],
    at: usize,
}

impl<'a> VecOrdinalsCursor<'a> {
    pub fn new(ords: &'a [i32]) -> Self {
        VecOrdinalsCursor { ords, at: 0 }
    }
}

impl OrdinalsCursor for VecOrdinalsCursor<'_> {
    fn size(&self) -> usize {
        self.ords.len()
    }
    fn next_ord(&mut self) -> i32 {
        let v = self.ords.get(self.at).copied().unwrap_or(-1);
        self.at = self.at.saturating_add(1);
        v
    }
}

/// An [`OrdinalsTupleCursor`] over `(doc, ord)` pairs.
#[derive(Debug, Clone)]
pub struct VecOrdinalsTupleCursor<'a> {
    tuples: &'a [(i32, i32)],
    at: Option<usize>,
}

impl<'a> VecOrdinalsTupleCursor<'a> {
    pub fn new(tuples: &'a [(i32, i32)]) -> Self {
        VecOrdinalsTupleCursor { tuples, at: None }
    }
}

impl OrdinalsTupleCursor for VecOrdinalsTupleCursor<'_> {
    fn next_doc(&mut self) -> i32 {
        let next = self.at.map_or(0, |i| i.saturating_add(1));
        self.at = Some(next);
        self.tuples.get(next).map_or(NO_MORE_DOCS, |t| t.0)
    }
    fn ord_value(&self) -> i32 {
        self.at.and_then(|i| self.tuples.get(i)).map_or(-1, |t| t.1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::{IndexOptions, VectorEncoding, VectorSimilarityFunction};
    use crate::index_writer::IndexWriter;
    use crate::segment_info::LuceneVersion;
    use lucene_store::directory::FsDirectory;
    use lucene_util::test_support::TempDir;

    fn ft(f: impl FnOnce(&mut FieldType)) -> FieldType {
        let mut t = FieldType::new();
        f(&mut t);
        t.frozen()
    }

    struct L {
        c: Column,
        kind: NumericKind,
        tuples: Vec<(i32, i64)>,
        dense: Vec<i64>,
    }
    impl LongColumn for L {
        fn column(&self) -> &Column {
            &self.c
        }
        fn numeric_kind(&self) -> NumericKind {
            self.kind
        }
        fn tuples(&self) -> Box<dyn LongTupleCursor + '_> {
            Box::new(VecLongTupleCursor::new(&self.tuples))
        }
        fn values(&self) -> Result<Box<dyn LongValuesCursor + '_>> {
            Ok(Box::new(VecLongValuesCursor::new(&self.dense)))
        }
    }

    struct B {
        c: Column,
        stored: StoredType,
        tuples: Vec<(i32, Vec<u8>)>,
    }
    impl BinaryColumn for B {
        fn column(&self) -> &Column {
            &self.c
        }
        fn stored_type(&self) -> StoredType {
            self.stored
        }
        fn tuples(&self) -> Box<dyn ObjectTupleCursor<Vec<u8>> + '_> {
            Box::new(VecObjectTupleCursor::new(&self.tuples))
        }
    }

    struct D {
        c: Column,
        dict: Vec<Vec<u8>>,
        tuples: Vec<(i32, i32)>,
    }
    impl DictionaryColumn for D {
        fn column(&self) -> &Column {
            &self.c
        }
        fn dictionary(&self) -> &[Vec<u8>] {
            &self.dict
        }
        fn tuples(&self) -> Box<dyn OrdinalsTupleCursor + '_> {
            Box::new(VecOrdinalsTupleCursor::new(&self.tuples))
        }
    }

    struct T {
        c: Column,
        tuples: Vec<(i32, FieldTokens)>,
    }
    impl TokenStreamColumn for T {
        fn column(&self) -> &Column {
            &self.c
        }
        fn tuples(&self) -> Box<dyn ObjectTupleCursor<FieldTokens> + '_> {
            Box::new(VecObjectTupleCursor::new(&self.tuples))
        }
    }

    struct V {
        c: Column,
        tuples: Vec<(i32, Vec<u8>)>,
    }
    impl VectorColumn<Vec<u8>> for V {
        fn column(&self) -> &Column {
            &self.c
        }
        fn tuples(&self) -> Box<dyn ObjectTupleCursor<Vec<u8>> + '_> {
            Box::new(VecObjectTupleCursor::new(&self.tuples))
        }
    }

    struct Batch<'a> {
        n: usize,
        cols: Vec<BatchColumn<'a>>,
    }
    impl ColumnBatch for Batch<'_> {
        fn num_docs(&self) -> usize {
            self.n
        }
        fn columns(&self) -> Vec<BatchColumn<'_>> {
            self.cols.clone()
        }
    }

    fn numeric() -> FieldType {
        ft(|t| t.set_doc_values_type(DocValuesType::Numeric).unwrap())
    }

    fn long(name: &str, t: FieldType, density: Density, tuples: Vec<(i32, i64)>) -> L {
        L {
            c: Column::new(name, t, density),
            kind: NumericKind::Long,
            dense: tuples.iter().map(|t| t.1).collect(),
            tuples,
        }
    }

    fn err(batch: &dyn ColumnBatch) -> String {
        batch_documents(batch).unwrap_err().to_string()
    }

    #[test]
    fn columns_are_validated_as_process_batch_validates_them() {
        let none = long("a", FieldType::new(), Density::Sparse, vec![]);
        let e = err(&Batch {
            n: 1,
            cols: vec![BatchColumn::Long(&none)],
        });
        assert!(e.contains("must have a non-NONE"), "{e}");
        let two_dims = long(
            "a",
            ft(|t| t.set_dimensions(2, 8).unwrap()),
            Density::Sparse,
            vec![],
        );
        assert!(err(&Batch {
            n: 1,
            cols: vec![BatchColumn::Long(&two_dims)]
        })
        .contains("1-dimensional"));
        let mut int = long(
            "a",
            ft(|t| t.set_dimensions(1, 8).unwrap()),
            Density::Sparse,
            vec![],
        );
        int.kind = NumericKind::Int;
        assert!(err(&Batch {
            n: 1,
            cols: vec![BatchColumn::Long(&int)]
        })
        .contains("pointNumBytes=4"));
        let indexed = long(
            "a",
            ft(|t| t.set_index_options(IndexOptions::Docs).unwrap()),
            Density::Sparse,
            vec![],
        );
        assert!(err(&Batch {
            n: 1,
            cols: vec![BatchColumn::Long(&indexed)]
        })
        .contains("inverted"));
        let sorted = long(
            "a",
            ft(|t| t.set_doc_values_type(DocValuesType::Sorted).unwrap()),
            Density::Sparse,
            vec![],
        );
        assert!(err(&Batch {
            n: 1,
            cols: vec![BatchColumn::Long(&sorted)]
        })
        .contains("incompatible"));
        let bin_num = B {
            c: Column::new("b", numeric(), Density::Sparse),
            stored: StoredType::Binary,
            tuples: vec![],
        };
        assert!(err(&Batch {
            n: 1,
            cols: vec![BatchColumn::Binary(&bin_num)]
        })
        .contains("use a LongColumn"));
        let bin_stored = B {
            c: Column::new("b", ft(|t| t.set_stored(true).unwrap()), Density::Sparse),
            stored: StoredType::Long,
            tuples: vec![],
        };
        assert!(err(&Batch {
            n: 1,
            cols: vec![BatchColumn::Binary(&bin_stored)]
        })
        .contains("storedType"));
        let dict = |t: FieldType, dict: Vec<Vec<u8>>| D {
            c: Column::new("d", t, Density::Sparse),
            dict,
            tuples: vec![(0, 5)],
        };
        let sorted_dv = ft(|t| t.set_doc_values_type(DocValuesType::Sorted).unwrap());
        for (d, want) in [
            (dict(sorted_dv.clone(), vec![]), "must not be empty"),
            (dict(sorted_dv.clone(), vec![vec![0; 40000]]), "too large"),
            (dict(numeric(), vec![vec![1]]), "use a LongColumn"),
            (
                dict(
                    ft(|t| t.set_doc_values_type(DocValuesType::Binary).unwrap()),
                    vec![vec![1]],
                ),
                "BINARY",
            ),
            (
                dict(
                    ft(|t| {
                        t.set_dimensions(1, 1).unwrap();
                        t.set_doc_values_type(DocValuesType::Sorted).unwrap();
                    }),
                    vec![vec![1]],
                ),
                "does not support points",
            ),
            (
                dict(sorted_dv.clone(), vec![vec![1]]),
                "ordinal 5 is out of range",
            ),
        ] {
            let e = err(&Batch {
                n: 1,
                cols: vec![BatchColumn::Dictionary(&d)],
            });
            assert!(e.contains(want), "{e}");
        }
        let tokens = |t: FieldType| T {
            c: Column::new("t", t, Density::Sparse),
            tuples: vec![],
        };
        let t = tokens(ft(|t| t.set_stored(true).unwrap()));
        assert!(err(&Batch {
            n: 1,
            cols: vec![BatchColumn::TokenStream(&t)]
        })
        .contains("tokenized == true"));
        let t = tokens(ft(|t| {
            t.set_index_options(IndexOptions::Docs).unwrap();
            t.set_stored(true).unwrap();
        }));
        assert!(err(&Batch {
            n: 1,
            cols: vec![BatchColumn::TokenStream(&t)]
        })
        .contains("inverted-only"));
        let vec_type = ft(|t| {
            t.set_vector_attributes(
                2,
                VectorEncoding::Byte,
                VectorSimilarityFunction::DotProduct,
            )
            .unwrap()
        });
        let v = V {
            c: Column::new("v", ft(|t| t.set_stored(true).unwrap()), Density::Sparse),
            tuples: vec![],
        };
        assert!(err(&Batch {
            n: 1,
            cols: vec![BatchColumn::ByteVector(&v)]
        })
        .contains("vectorDimension() > 0"));
        let v = V {
            c: Column::new(
                "v",
                ft(|t| {
                    t.set_vector_attributes(
                        2,
                        VectorEncoding::Byte,
                        VectorSimilarityFunction::DotProduct,
                    )
                    .unwrap();
                    t.set_stored(true).unwrap();
                }),
                Density::Sparse,
            ),
            tuples: vec![],
        };
        assert!(err(&Batch {
            n: 1,
            cols: vec![BatchColumn::ByteVector(&v)]
        })
        .contains("vector-only"));
        for (tuples, want) in [
            (
                vec![(0, vec![1u8, 2]), (0, vec![1, 2])],
                "strictly increasing",
            ),
            (vec![(0, vec![1u8])], "expected dimension 2"),
            (vec![(3, vec![1u8, 2])], "out of range"),
        ] {
            let v = V {
                c: Column::new("v", vec_type.clone(), Density::Sparse),
                tuples,
            };
            let e = err(&Batch {
                n: 1,
                cols: vec![BatchColumn::ByteVector(&v)],
            });
            assert!(e.contains(want), "{e}");
        }
        // One feature per field across the batch's columns.
        let a = long("x", numeric(), Density::Sparse, vec![]);
        let b = long("x", numeric(), Density::Sparse, vec![]);
        let e = err(&Batch {
            n: 1,
            cols: vec![BatchColumn::Long(&a), BatchColumn::Long(&b)],
        });
        assert!(e.contains("[doc values]"), "{e}");
        // Doc ids in range and not going backwards; dense counts matching.
        let back = long("x", numeric(), Density::Sparse, vec![(1, 1), (0, 1)]);
        assert!(err(&Batch {
            n: 2,
            cols: vec![BatchColumn::Long(&back)]
        })
        .contains("out-of-order"));
        let neg = long("x", numeric(), Density::Sparse, vec![(-1, 1)]);
        assert!(err(&Batch {
            n: 2,
            cols: vec![BatchColumn::Long(&neg)]
        })
        .contains("out of range"));
        let short = long("x", numeric(), Density::Dense, vec![(0, 1)]);
        assert!(err(&Batch {
            n: 2,
            cols: vec![BatchColumn::Long(&short)]
        })
        .contains("provided 1 values"));
        assert_eq!(
            ColumnValidation::feature_names(
                ColumnValidation::FEATURE_INVERSION
                    | ColumnValidation::FEATURE_STORED
                    | ColumnValidation::FEATURE_POINTS
                    | ColumnValidation::FEATURE_VECTOR
            ),
            "[inversion, stored, points, vectors]"
        );
    }

    #[test]
    fn sparse_columns_without_values_have_no_dense_view() {
        struct OnlyTuples(Column);
        impl LongColumn for OnlyTuples {
            fn column(&self) -> &Column {
                &self.0
            }
            fn tuples(&self) -> Box<dyn LongTupleCursor + '_> {
                Box::new(VecLongTupleCursor::new(&[]))
            }
        }
        impl BinaryColumn for OnlyTuples {
            fn column(&self) -> &Column {
                &self.0
            }
            fn tuples(&self) -> Box<dyn ObjectTupleCursor<Vec<u8>> + '_> {
                Box::new(VecObjectTupleCursor::new(&[]))
            }
        }
        impl DictionaryColumn for OnlyTuples {
            fn column(&self) -> &Column {
                &self.0
            }
            fn dictionary(&self) -> &[Vec<u8>] {
                &[]
            }
            fn tuples(&self) -> Box<dyn OrdinalsTupleCursor + '_> {
                Box::new(VecOrdinalsTupleCursor::new(&[]))
            }
        }
        let o = OnlyTuples(Column::new("o", numeric(), Density::Dense));
        assert_eq!(LongColumn::numeric_kind(&o), NumericKind::Long);
        assert!(LongColumn::values(&o).is_err());
        assert_eq!(BinaryColumn::stored_type(&o), StoredType::Binary);
        assert!(BinaryColumn::values(&o).is_err());
        assert_eq!(DictionaryColumn::stored_type(&o), StoredType::Binary);
        assert!(DictionaryColumn::values(&o).is_err());
        assert_eq!(o.0.density(), Density::Dense);
        for k in [
            NumericKind::Int,
            NumericKind::Long,
            NumericKind::Float,
            NumericKind::Double,
        ] {
            let _ = k.stored_type();
        }
        // The in-memory cursors past their end.
        let mut c = VecLongValuesCursor::new(&[]);
        assert_eq!((c.size(), c.next_long()), (0, 0));
        let mut c = VecBytesValuesCursor::new(&[]);
        assert_eq!((c.size(), c.next_value()), (0, Vec::new()));
        let mut c = VecOrdinalsCursor::new(&[]);
        assert_eq!((c.size(), c.next_ord()), (0, -1));
        let mut c = VecOrdinalsTupleCursor::new(&[]);
        assert_eq!((c.ord_value(), c.next_doc()), (-1, NO_MORE_DOCS));
        let c = VecLongTupleCursor::new(&[]);
        assert_eq!(c.long_value(), 0);
        let c = VecObjectTupleCursor::<Vec<u8>>::new(&[]);
        assert_eq!(c.value(), Vec::<u8>::new());
    }

    #[test]
    fn adapters_view_a_cell_as_a_field() {
        let analyzer = Analyzer::standard(None);
        let stored_long = ft(|t| {
            t.set_stored(true).unwrap();
            t.set_dimensions(1, 4).unwrap();
        });
        let c = Column::new("n", stored_long, Density::Dense);
        let f = LongColumnAdapter::new(&c, NumericKind::Float, i64::from(1_065_353_216));
        assert_eq!(f.stored_value(), Some(StoredValue::Float(1.0)));
        assert_eq!(f.binary_value().unwrap().len(), 4);
        assert_eq!(f.numeric_value(), Some(Number::Long(1_065_353_216)));
        assert!(f.token_stream(&analyzer).unwrap().is_none());
        assert_eq!(f.name(), "n");
        assert_eq!(
            LongColumnAdapter::new(&c, NumericKind::Int, 5).stored_value(),
            Some(StoredValue::Int(5))
        );
        let unstored = Column::new("n", numeric(), Density::Dense);
        assert_eq!(
            LongColumnAdapter::new(&unstored, NumericKind::Long, 1).stored_value(),
            None
        );
        let text = ft(|t| {
            t.set_index_options(IndexOptions::DocsAndFreqsAndPositions)
                .unwrap();
            t.set_stored(true).unwrap();
        });
        let c = Column::new("t", text, Density::Sparse);
        let b = BinaryColumnAdapter::new(&c, StoredType::String, Arc::from(&b"Hello World"[..]));
        assert_eq!(b.string_value().as_deref(), Some("Hello World"));
        assert_eq!(b.invertable_type(), InvertableType::TokenStream);
        assert_eq!(b.token_stream(&analyzer).unwrap().unwrap().tokens.len(), 2);
        assert_eq!(
            b.stored_value(),
            Some(StoredValue::String("Hello World".into()))
        );
        let t = BinaryColumnAdapter::from_tokens(&c, FieldTokens::default());
        assert_eq!(t.binary_value(), None);
        assert_eq!(t.stored_value(), None);
        assert_eq!(
            t.token_stream(&analyzer).unwrap(),
            Some(FieldTokens::default())
        );
        let kw = ft(|t| {
            t.set_index_options(IndexOptions::Docs).unwrap();
            t.set_tokenized(false).unwrap();
            t.set_stored(true).unwrap();
        });
        let c = Column::new("k", kw, Density::Sparse);
        let k = BinaryColumnAdapter::new(&c, StoredType::Binary, Arc::from(&b"k"[..]));
        assert_eq!(k.invertable_type(), InvertableType::Binary);
        assert_eq!(k.string_value(), None);
        assert!(k.token_stream(&analyzer).unwrap().is_none());
        assert_eq!(k.stored_value(), Some(StoredValue::Binary(b"k".to_vec())));
        assert_eq!(k.name(), "k");
    }

    #[test]
    fn a_batch_writes_and_updates() {
        let tmp = TempDir::new("column-batch");
        let dir = FsDirectory::open(tmp.path());
        let version = LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        };
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", version).unwrap();
        let id = D {
            c: Column::new(
                "id",
                ft(|t| {
                    t.set_index_options(IndexOptions::Docs).unwrap();
                    t.set_tokenized(false).unwrap();
                    t.set_omit_norms(true).unwrap();
                }),
                Density::Dense,
            ),
            dict: vec![b"a".to_vec(), b"b".to_vec()],
            tuples: vec![(0, 0), (1, 1)],
        };
        struct Dense<'a>(&'a D, Vec<i32>);
        impl DictionaryColumn for Dense<'_> {
            fn column(&self) -> &Column {
                &self.0.c
            }
            fn dictionary(&self) -> &[Vec<u8>] {
                &self.0.dict
            }
            fn tuples(&self) -> Box<dyn OrdinalsTupleCursor + '_> {
                Box::new(VecOrdinalsTupleCursor::new(&self.0.tuples))
            }
            fn values(&self) -> Result<Box<dyn OrdinalsCursor + '_>> {
                Ok(Box::new(VecOrdinalsCursor::new(&self.1)))
            }
        }
        let dense = Dense(&id, vec![0, 1]);
        let v = V {
            c: Column::new(
                "v",
                ft(|t| {
                    t.set_vector_attributes(
                        2,
                        VectorEncoding::Byte,
                        VectorSimilarityFunction::DotProduct,
                    )
                    .unwrap()
                }),
                Density::Sparse,
            ),
            tuples: vec![(1, vec![1, 2])],
        };
        // A second column for `id`, claiming a different feature.
        let id_values = long("id", numeric(), Density::Dense, vec![(0, 1), (1, 2)]);
        let batch = Batch {
            n: 2,
            cols: vec![
                BatchColumn::Dictionary(&dense),
                BatchColumn::ByteVector(&v),
                BatchColumn::Long(&id_values),
            ],
        };
        w.add_batch(&batch).unwrap();
        w.update_batch(crate::buffered_updates::Term::new("id", "a"), &batch)
            .unwrap();
        w.commit().unwrap();
        assert_eq!(w.committed_doc_count().unwrap(), 4);
        let deleted: i32 = w.segment_infos().segments.iter().map(|s| s.del_count).sum();
        assert_eq!(deleted, 1, "the update deleted the first batch's \"a\"");
        assert_eq!(batch.columns()[0].column().name(), "id");
        assert_eq!(batch.columns()[1].column().name(), "v");
    }
}
